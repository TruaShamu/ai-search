"""Event-driven embedding worker — pulls batch tasks from Azure Storage Queue,
embeds books, and upserts directly to Qdrant.

Architecture:
    Storage Queue → ACA Job (scale 0→1) → embed batch → upsert to Qdrant → scale to 0

Message format (JSON):
    {
        "batch_id": "batch-0001",
        "blob_path": "input/books_augmented.jsonl",
        "start_idx": 0,
        "end_idx": 1000
    }

Environment variables:
    AZURE_STORAGE_CONNECTION_STRING  — for queue + blob access
    QUEUE_NAME                       — queue name (default: embed-tasks)
    STORAGE_CONTAINER                — blob container (default: embeddings)
    QDRANT_URL                       — Qdrant endpoint (e.g., http://qdrant:6333)
    QDRANT_COLLECTION                — collection name (default: books)
    EMBED_DIM                        — Matryoshka dimension (default: 256)

Usage:
    python -m scripts.embed_worker              # Process one message then exit
    python -m scripts.embed_worker --loop       # Process until queue is empty
    python -m scripts.embed_worker --enqueue    # Enqueue batch tasks for all books
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from azure.storage.blob import BlobServiceClient
from azure.storage.queue import QueueServiceClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.search.embed import MODEL_NAME, BATCH_SIZE, build_embedding_texts, load_books
from src.etl.clean_descriptions import clean_description

# Config from environment
QUEUE_NAME = os.getenv("QUEUE_NAME", "embed-tasks")
STORAGE_CONTAINER = os.getenv("STORAGE_CONTAINER", "embeddings")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "books")
EMBED_DIM = int(os.getenv("EMBED_DIM", "256"))
BATCH_CHUNK_SIZE = int(os.getenv("BATCH_CHUNK_SIZE", "1000"))


def get_storage_clients(connection_string: str):
    """Create queue and blob clients."""
    queue_service = QueueServiceClient.from_connection_string(connection_string)
    queue_client = queue_service.get_queue_client(QUEUE_NAME)
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service.get_container_client(STORAGE_CONTAINER)
    return queue_client, blob_client


def download_books_slice(blob_client, blob_path: str, start_idx: int, end_idx: int) -> list[dict]:
    """Download JSONL from blob and extract slice [start_idx, end_idx)."""
    print(f"Downloading {blob_path} from blob storage...")
    download_stream = blob_client.download_blob(blob_path)
    content = download_stream.readall().decode("utf-8")

    books = []
    for i, line in enumerate(content.strip().split("\n")):
        if i < start_idx:
            continue
        if i >= end_idx:
            break
        book = json.loads(line)
        if book.get("tier", 99) <= 1 and book.get("description"):
            # Clean description before embedding
            book["description"] = clean_description(book["description"])
            books.append(book)

    print(f"  Loaded {len(books)} tier-1 books from indices [{start_idx}, {end_idx})")
    return books


def embed_books(books: list[dict], dim: int = EMBED_DIM) -> np.ndarray:
    """Embed books using nomic-embed-text-v1.5."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    texts = build_embedding_texts(books)
    print(f"Encoding {len(texts)} texts (dim={dim})...")
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
    embeddings = embeddings[:, :dim]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    elapsed = time.time() - t0
    print(f"  Encoded in {elapsed:.1f}s ({len(texts)/elapsed:.0f} docs/sec)")
    return embeddings


def upsert_to_qdrant(books: list[dict], embeddings: np.ndarray, start_idx: int):
    """Upsert embedded books directly into Qdrant with dense + sparse vectors."""
    from qdrant_client import QdrantClient, models
    from sklearn.feature_extraction.text import TfidfVectorizer

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)

    # Ensure collection exists
    collections = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in collections:
        print(f"Creating collection '{QDRANT_COLLECTION}'...")
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={"dense": models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        client.create_payload_index(QDRANT_COLLECTION, "year", models.PayloadSchemaType.INTEGER)
        client.create_payload_index(QDRANT_COLLECTION, "tier", models.PayloadSchemaType.INTEGER)

    # Build TF-IDF sparse vectors for this batch
    corpus = []
    for b in books:
        parts = [b.get("title", ""), b.get("description", "")]
        if b.get("subjects"):
            parts.append(" ".join(b["subjects"][:10]))
        if b.get("authors"):
            authors = b["authors"] if isinstance(b["authors"], str) else ", ".join(b["authors"])
            parts.append(authors)
        corpus.append(" ".join(parts))

    vectorizer = TfidfVectorizer(max_features=30000, stop_words="english", sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Upload in batches
    UPLOAD_BATCH = 100
    total_uploaded = 0
    for batch_start in range(0, len(books), UPLOAD_BATCH):
        batch_end = min(batch_start + UPLOAD_BATCH, len(books))
        points = []

        for j in range(batch_start, batch_end):
            book = books[j]
            point_id = start_idx + j
            sparse_row = tfidf_matrix[j].tocsr()

            payload = {
                "id": point_id,
                "work_id": book.get("work_id", ""),
                "title": book.get("title", ""),
                "authors": book.get("authors", ""),
                "description": book.get("description", ""),
                "subjects": book.get("subjects", []),
                "first_publish_year": book.get("first_publish_year"),
                "year": book.get("first_publish_year"),
                "cover_id": book.get("cover_id"),
                "cover_url": f"https://covers.openlibrary.org/b/id/{book['cover_id']}-M.jpg" if book.get("cover_id") else None,
                "subject_places": book.get("subject_places", []),
                "subject_people": book.get("subject_people", []),
                "subject_times": book.get("subject_times", []),
                "tier": book.get("tier", 1),
                "description_source": book.get("description_source", "unknown"),
            }

            points.append(models.PointStruct(
                id=point_id,
                vector={
                    "dense": embeddings[j].tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_row.indices.tolist(),
                        values=sparse_row.data.astype(float).tolist(),
                    ),
                },
                payload=payload,
            ))

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        total_uploaded += len(points)

    print(f"  Upserted {total_uploaded} points to Qdrant (IDs {start_idx}–{start_idx + len(books) - 1})")


def process_message(queue_client, blob_client, message) -> bool:
    """Process a single queue message: download → embed → upsert → delete message."""
    try:
        payload = json.loads(message.content)
        batch_id = payload["batch_id"]
        blob_path = payload["blob_path"]
        start_idx = payload["start_idx"]
        end_idx = payload["end_idx"]

        print(f"\n{'='*60}")
        print(f"Processing batch: {batch_id} [{start_idx}–{end_idx})")
        print(f"{'='*60}")

        books = download_books_slice(blob_client, blob_path, start_idx, end_idx)
        if not books:
            print("  No tier-1 books in this slice, skipping.")
            queue_client.delete_message(message)
            return True

        embeddings = embed_books(books)
        upsert_to_qdrant(books, embeddings, start_idx)

        # Delete message on success
        queue_client.delete_message(message)
        print(f"✓ Batch {batch_id} complete.")
        return True

    except Exception as e:
        print(f"✗ Error processing message: {e}")
        import traceback
        traceback.print_exc()
        return False


def worker_loop(connection_string: str, loop: bool = False):
    """Main worker loop — process messages from queue."""
    queue_client, blob_client = get_storage_clients(connection_string)

    # Ensure queue exists
    try:
        queue_client.create_queue()
        print(f"Created queue '{QUEUE_NAME}'")
    except Exception:
        pass  # Already exists

    processed = 0
    while True:
        messages = queue_client.receive_messages(max_messages=1, visibility_timeout=600)
        msg_list = list(messages)

        if not msg_list:
            if loop:
                print("Queue empty — all batches processed.")
            else:
                print("No messages in queue.")
            break

        for msg in msg_list:
            success = process_message(queue_client, blob_client, msg)
            if success:
                processed += 1

        if not loop:
            break

    print(f"\nWorker finished. Processed {processed} batch(es).")


def enqueue_batches(connection_string: str, blob_path: str, total_books: int):
    """Enqueue batch tasks into the Storage Queue."""
    queue_client, _ = get_storage_clients(connection_string)

    try:
        queue_client.create_queue()
        print(f"Created queue '{QUEUE_NAME}'")
    except Exception:
        pass

    num_batches = math.ceil(total_books / BATCH_CHUNK_SIZE)
    print(f"Enqueueing {num_batches} batches ({total_books} books, {BATCH_CHUNK_SIZE}/batch)...")

    for i in range(num_batches):
        start = i * BATCH_CHUNK_SIZE
        end = min((i + 1) * BATCH_CHUNK_SIZE, total_books)
        message = json.dumps({
            "batch_id": f"batch-{i:04d}",
            "blob_path": blob_path,
            "start_idx": start,
            "end_idx": end,
        })
        queue_client.send_message(message)

    print(f"✓ Enqueued {num_batches} messages to '{QUEUE_NAME}'")


def main():
    parser = argparse.ArgumentParser(description="Event-driven embedding worker")
    parser.add_argument("--loop", action="store_true", help="Process all messages until queue is empty")
    parser.add_argument("--enqueue", action="store_true", help="Enqueue batch tasks (producer mode)")
    parser.add_argument("--blob-path", default="input/books_augmented.jsonl", help="Blob path for enqueue")
    parser.add_argument("--total-books", type=int, default=250811, help="Total lines in JSONL for enqueue")
    args = parser.parse_args()

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        print("ERROR: Set AZURE_STORAGE_CONNECTION_STRING environment variable")
        sys.exit(1)

    if args.enqueue:
        enqueue_batches(connection_string, args.blob_path, args.total_books)
    else:
        worker_loop(connection_string, loop=args.loop)


if __name__ == "__main__":
    main()
