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
    python -m src.indexing.worker              # Process one message then exit
    python -m src.indexing.worker --loop       # Process until queue is empty
    python -m src.indexing.worker --enqueue    # Enqueue batch tasks for all books
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from azure.storage.blob import BlobServiceClient
from azure.storage.queue import QueueServiceClient

from src.search.embed import MODEL_NAME, BATCH_SIZE, build_embedding_texts
from src.etl.clean_descriptions import clean_description

# Config from environment
QUEUE_NAME = os.getenv("QUEUE_NAME", "embed-tasks")
STORAGE_CONTAINER = os.getenv("STORAGE_CONTAINER", "embeddings")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "books")
EMBED_DIM = int(os.getenv("EMBED_DIM", "256"))
BATCH_CHUNK_SIZE = int(os.getenv("BATCH_CHUNK_SIZE", "500"))
# "blob"   -- write dense vectors as shards for offline assembly (bulk backfill).
# "qdrant" -- upsert directly. Only safe for incremental top-ups against an
#            existing collection, because the sparse vectors it builds are fit
#            per slice and are not comparable across slices or with the query
#            encoder's global vectorizer.
EMBED_OUTPUT_MODE = os.getenv("EMBED_OUTPUT_MODE", "blob").strip().lower()
SHARD_PREFIX = os.getenv("SHARD_PREFIX", "shards")
# The index-building default of 128 is tuned for a workstation. An ACA
# Consumption replica is capped at 4GiB, which the model plus a 128-wide batch
# overruns; the resulting OOMKill (exit 137) is a SIGKILL, so it cannot be
# caught or reported and the batch just vanishes.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", str(min(32, BATCH_SIZE))))
EMBED_MAX_SEQ_LEN = int(os.getenv("EMBED_MAX_SEQ_LEN", "1024"))

# The model is ~500MB to load and is reused across every message this replica
# handles. Loading it per message wasted about five seconds per batch.
_MODEL = None


def get_model():
    """Load the embedding model once per process."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        print(f"Loading model: {MODEL_NAME} (once per replica)...")
        t0 = time.time()
        _MODEL = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
        print(f"  Model loaded in {time.time() - t0:.1f}s")
    return _MODEL


def get_storage_clients(connection_string: str):
    """Create queue and blob clients."""
    queue_service = QueueServiceClient.from_connection_string(connection_string)
    queue_client = queue_service.get_queue_client(QUEUE_NAME)
    blob_service = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service.get_container_client(STORAGE_CONTAINER)
    return queue_client, blob_client


def download_books_slice(blob_client, blob_path: str, start_idx: int, end_idx: int) -> list[dict]:
    """Read one pre-cut slice blob and return its eligible books.

    Each slice is uploaded as its own small blob at enqueue time. The earlier
    design shipped the whole ~110MB corpus to every worker and had it seek to
    its range, which cost ~400MB of resident memory per replica on top of the
    ~1.5GB model. In a 4GiB Consumption replica that produced an OOMKill (exit
    137) -- a SIGKILL, so no traceback, no error blob, and the batch simply
    disappeared while the queue message stayed invisible for the full hour.
    Cutting slices up front also removes ~20GB of repeated egress across the run.
    """
    print(f"Reading slice {blob_path}...")
    raw = blob_client.download_blob(blob_path).readall().decode("utf-8")

    books = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        book = json.loads(line)
        if book.get("tier", 99) <= 1 and book.get("description"):
            book["description"] = clean_description(book["description"])
            books.append(book)

    print(f"  Loaded {len(books)} tier-1 books for indices [{start_idx}, {end_idx})")
    return books


def embed_books(books: list[dict], dim: int = EMBED_DIM) -> np.ndarray:
    """Embed books using nomic-embed-text-v1.5."""
    model = get_model()

    # nomic-embed accepts up to 8192 tokens, and sentence-transformers pads each
    # batch to its longest member, so the default sequence length lets one long
    # document balloon a whole batch. The longest embedding text in this corpus
    # is ~2.4K characters (~600 tokens), so 1024 truncates nothing here while
    # bounding worst-case activation memory.
    if model.max_seq_length is None or model.max_seq_length > EMBED_MAX_SEQ_LEN:
        model.max_seq_length = EMBED_MAX_SEQ_LEN

    texts = build_embedding_texts(books)
    print(f"Encoding {len(texts)} texts (dim={dim}, batch={EMBED_BATCH_SIZE}, msl={model.max_seq_length})...")
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False, normalize_embeddings=True)
    embeddings = embeddings[:, :dim]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    elapsed = time.time() - t0
    print(f"  Encoded in {elapsed:.1f}s ({len(texts)/elapsed:.1f} docs/sec)")
    return embeddings


def write_shard_to_blob(blob_client, batch_id: str, books: list[dict],
                        embeddings: np.ndarray, start_idx: int) -> None:
    """Persist one slice's dense vectors for offline assembly.

    Sparse vectors are deliberately not built here. TF-IDF has to be fit over
    the whole corpus so that document vectors share a vocabulary with the query
    encoder's pickled vectorizer; fitting it per slice would silently produce
    incompatible sparse vectors. Assembly and the global TF-IDF fit happen in
    src/qdrant/migrate.py.
    """
    import io

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        embeddings=embeddings.astype(np.float32),
        work_ids=np.array([b.get("work_id", "") for b in books], dtype=object),
        start_idx=np.array([start_idx]),
    )
    buf.seek(0)
    name = f"{SHARD_PREFIX}/{batch_id}.npz"
    blob_client.upload_blob(name=name, data=buf, overwrite=True)
    print(f"  Wrote shard {name} ({embeddings.shape[0]} vectors)")


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


def write_error_to_blob(blob_client, batch_id: str, detail: str) -> None:
    """Persist a failure traceback where it can be read without container logs.

    The ACA environment has no Log Analytics destination configured, so a
    container that fails leaves nothing behind. Writing the traceback to blob
    makes a failed backfill diagnosable after the fact.
    """
    try:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        name = f"errors/{batch_id}-{stamp}.txt"
        blob_client.upload_blob(name=name, data=detail.encode("utf-8"), overwrite=True)
        print(f"  wrote diagnostics to {name}")
    except Exception as exc:  # never let diagnostics-writing mask the real error
        print(f"  could not write diagnostics to blob: {exc}")


def process_message(queue_client, blob_client, message) -> bool:
    """Process a single queue message: download → embed → upsert → delete message."""
    batch_id = "unknown"
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
        if EMBED_OUTPUT_MODE == "qdrant":
            upsert_to_qdrant(books, embeddings, start_idx)
        else:
            write_shard_to_blob(blob_client, batch_id, books, embeddings, start_idx)

        # Delete message on success
        queue_client.delete_message(message)
        print(f"✓ Batch {batch_id} complete.")
        return True

    except Exception as e:
        import traceback

        detail = (
            f"batch_id={batch_id}\n"
            f"EMBED_OUTPUT_MODE={EMBED_OUTPUT_MODE}\n"
            f"error={type(e).__name__}: {e}\n\n"
            f"{traceback.format_exc()}"
        )
        print(f"✗ Error processing message: {e}")
        traceback.print_exc()
        write_error_to_blob(blob_client, batch_id, detail)
        return False


def worker_loop(connection_string: str, loop: bool = False) -> int:
    """Main worker loop — process messages from queue. Returns the failure count."""
    queue_client, blob_client = get_storage_clients(connection_string)

    # Ensure queue exists
    try:
        queue_client.create_queue()
        print(f"Created queue '{QUEUE_NAME}'")
    except Exception:
        pass  # Already exists

    # Load the model before touching the queue. A replica that cannot load the
    # model must not dequeue anything: receiving a message hides it for the
    # visibility timeout, so a broken replica would silently swallow work and
    # still exit cleanly. Failing here leaves the messages visible for a
    # healthy replica to pick up.
    try:
        get_model()
    except Exception as e:
        import traceback

        detail = f"startup: model load failed\nerror={type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        print(f"✗ FATAL: could not load embedding model: {e}")
        traceback.print_exc()
        write_error_to_blob(blob_client, "startup", detail)
        return 1

    processed = 0
    failed = 0
    while True:
        # Must exceed the time one batch takes to embed. If it does not, the
        # message becomes visible again mid-flight and a second replica redoes
        # the same slice.
        messages = queue_client.receive_messages(max_messages=1, visibility_timeout=3600)
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
            else:
                failed += 1

        if not loop:
            break

    print(f"\nWorker finished. Processed {processed} batch(es), {failed} failed.")
    return failed


def enqueue_batches(
    connection_string: str,
    corpus_path: str,
    slice_prefix: str = "inputs/slices",
    start_batch: int = 0,
    max_batches: int | None = None,
):
    """Cut the local corpus into per-batch slice blobs and enqueue one task each.

    Slicing happens here, once, rather than in every worker. Each worker then
    downloads only its own ~600KB slice instead of the full corpus, which is
    what keeps a replica inside the 4GiB Consumption memory cap.

    ``start_batch``/``max_batches`` bound which batches are enqueued. Batch ids
    stay tied to absolute corpus position, so a partial re-run re-enqueues the
    same ids and overwrites the same shards rather than shifting the numbering.
    """
    queue_client, blob_client = get_storage_clients(connection_string)

    try:
        queue_client.create_queue()
        print(f"Created queue '{QUEUE_NAME}'")
    except Exception:
        pass

    with open(corpus_path, encoding="utf-8") as f:
        lines = [ln for ln in (line.rstrip("\n") for line in f) if ln.strip()]

    total_books = len(lines)
    num_batches = math.ceil(total_books / BATCH_CHUNK_SIZE)
    last = num_batches if max_batches is None else min(num_batches, start_batch + max_batches)
    selected = range(start_batch, last)
    print(
        f"Corpus {total_books} books -> {num_batches} batches; "
        f"uploading batches {start_batch}..{last - 1} ({len(selected)} of them)"
    )

    for n, i in enumerate(selected, start=1):
        start = i * BATCH_CHUNK_SIZE
        end = min((i + 1) * BATCH_CHUNK_SIZE, total_books)
        batch_id = f"batch-{i:04d}"
        blob_path = f"{slice_prefix}/{batch_id}.jsonl"

        payload = ("\n".join(lines[start:end]) + "\n").encode("utf-8")
        blob_client.upload_blob(name=blob_path, data=payload, overwrite=True)

        queue_client.send_message(json.dumps({
            "batch_id": batch_id,
            "blob_path": blob_path,
            "start_idx": start,
            "end_idx": end,
        }))
        if n % 25 == 0 or n == len(selected):
            print(f"  {n}/{len(selected)} slices uploaded and enqueued")

    print(f"✓ Enqueued {len(selected)} messages to '{QUEUE_NAME}'")


def main():
    parser = argparse.ArgumentParser(description="Event-driven embedding worker")
    parser.add_argument("--loop", action="store_true", help="Process all messages until queue is empty")
    parser.add_argument("--enqueue", action="store_true", help="Enqueue batch tasks (producer mode)")
    parser.add_argument(
        "--corpus",
        default="data/processed/books_goodreads_v2.jsonl",
        help="Local corpus JSONL to slice and upload (enqueue mode)",
    )
    parser.add_argument("--slice-prefix", default="inputs/slices", help="Blob prefix for slice uploads")
    parser.add_argument("--start-batch", type=int, default=0, help="First batch index to enqueue")
    parser.add_argument("--max-batches", type=int, default=None, help="How many batches to enqueue")
    args = parser.parse_args()

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        print("ERROR: Set AZURE_STORAGE_CONNECTION_STRING environment variable")
        sys.exit(1)

    if args.enqueue:
        enqueue_batches(
            connection_string, args.corpus, args.slice_prefix, args.start_batch, args.max_batches
        )
    else:
        # Exit non-zero when any batch failed, so a broken backfill is not
        # reported as a successful job execution.
        failures = worker_loop(connection_string, loop=args.loop)
        if failures:
            sys.exit(1)


if __name__ == "__main__":
    main()
