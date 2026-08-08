"""Event-driven embedding worker — pulls slice tasks from a work queue, embeds
books, and either writes dense shards to the object store or upserts directly to
Qdrant.

Architecture (portable default):
    Kafka topic → worker (KEDA ScaledJob, scale 0→N on lag) → embed slice →
    write shard to S3/MinIO (or upsert to Qdrant) → commit offset

The queue and object store are pluggable (see ``src.indexing.backends``):

    QUEUE_BACKEND         kafka (default) | azure
    OBJECT_STORE_BACKEND  s3 (default)    | azure

so the same worker runs on Kafka + MinIO in a Kubernetes cluster or on Azure
Storage Queue + Blob, with no code change.

Message format (JSON):
    {
        "batch_id": "batch-0001",
        "blob_path": "inputs/slices/batch-0001.jsonl",
        "start_idx": 0,
        "end_idx": 1000
    }

Usage:
    python -m src.indexing.worker              # Process one message then exit
    python -m src.indexing.worker --loop       # Process until the queue drains
    python -m src.indexing.worker --enqueue    # Enqueue slice tasks (producer)
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from src.indexing.backends import get_object_store, get_queue
from src.indexing.embed import MODEL_NAME, BATCH_SIZE, build_embedding_texts
from src.etl.clean_descriptions import clean_description

# Config from environment
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
# The index-building default of 128 is tuned for a workstation. A memory-capped
# worker (e.g. a 4GiB pod / ACA replica) overruns the model plus a 128-wide
# batch; the resulting OOMKill (exit 137) is a SIGKILL, so it cannot be caught
# or reported and the batch just vanishes.
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


def download_books_slice(store, blob_path: str, start_idx: int, end_idx: int) -> list[dict]:
    """Read one pre-cut slice object and return its eligible books.

    Each slice is uploaded as its own small object at enqueue time. The earlier
    design shipped the whole ~110MB corpus to every worker and had it seek to
    its range, which cost ~400MB of resident memory per replica on top of the
    ~1.5GB model. In a 4GiB replica that produced an OOMKill (exit 137) -- a
    SIGKILL, so no traceback, no error object, and the batch simply disappeared
    while the queue message stayed in-flight for the full visibility window.
    Cutting slices up front also removes ~20GB of repeated egress across the run.
    """
    print(f"Reading slice {blob_path}...")
    raw = store.get_bytes(blob_path).decode("utf-8")

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


def write_shard(store, batch_id: str, books: list[dict],
                embeddings: np.ndarray, start_idx: int) -> None:
    """Persist one slice's dense vectors for offline assembly.

    Sparse vectors are deliberately not built here. TF-IDF has to be fit over
    the whole corpus so that document vectors share a vocabulary with the query
    encoder's pickled vectorizer; fitting it per slice would silently produce
    incompatible sparse vectors. Assembly and the global TF-IDF fit happen in
    src/indexing/load.py.
    """
    import io

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        embeddings=embeddings.astype(np.float32),
        work_ids=np.array([b.get("work_id", "") for b in books], dtype=object),
        start_idx=np.array([start_idx]),
    )
    name = f"{SHARD_PREFIX}/{batch_id}.npz"
    store.put_bytes(name, buf.getvalue())
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


def write_error(store, batch_id: str, detail: str) -> None:
    """Persist a failure traceback where it can be read without container logs.

    A worker that dies leaves little behind if the cluster has no log sink
    configured. Writing the traceback to the object store makes a failed
    backfill diagnosable after the fact, on any backend.
    """
    try:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        name = f"errors/{batch_id}-{stamp}.txt"
        store.put_bytes(name, detail.encode("utf-8"))
        print(f"  wrote diagnostics to {name}")
    except Exception as exc:  # never let diagnostics-writing mask the real error
        print(f"  could not write diagnostics to object store: {exc}")


def process_message(queue, store, message) -> bool:
    """Process a single task: download → embed → persist → ack the message."""
    batch_id = "unknown"
    try:
        payload = message.json()
        batch_id = payload["batch_id"]
        blob_path = payload["blob_path"]
        start_idx = payload["start_idx"]
        end_idx = payload["end_idx"]

        print(f"\n{'='*60}")
        print(f"Processing batch: {batch_id} [{start_idx}–{end_idx})")
        print(f"{'='*60}")

        books = download_books_slice(store, blob_path, start_idx, end_idx)
        if not books:
            print("  No tier-1 books in this slice, skipping.")
            queue.ack(message)
            return True

        embeddings = embed_books(books)
        if EMBED_OUTPUT_MODE == "qdrant":
            upsert_to_qdrant(books, embeddings, start_idx)
        else:
            write_shard(store, batch_id, books, embeddings, start_idx)

        # Ack (commit offset / delete message) only after the slice is durable.
        queue.ack(message)
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
        write_error(store, batch_id, detail)
        # Do NOT ack: leaving the offset uncommitted (Kafka) / the message
        # in-flight (Azure) lets a healthy replica retry the slice.
        return False


def worker_loop(loop: bool = False) -> int:
    """Main worker loop — process messages from the queue. Returns failure count."""
    queue = get_queue()
    store = get_object_store()
    queue.ensure()

    # Load the model before touching the queue. A replica that cannot load the
    # model must not consume anything: an in-flight message would be hidden for
    # its visibility window, so a broken replica would silently swallow work and
    # still exit cleanly. Failing here leaves the work for a healthy replica.
    try:
        get_model()
    except Exception as e:
        import traceback

        detail = f"startup: model load failed\nerror={type(e).__name__}: {e}\n\n{traceback.format_exc()}"
        print(f"✗ FATAL: could not load embedding model: {e}")
        traceback.print_exc()
        write_error(store, "startup", detail)
        return 1

    processed = 0
    failed = 0
    # Kafka is a streaming log with no definite "empty" signal, so the loop
    # keeps polling for ``idle_timeout`` seconds of continuous silence before
    # concluding the backfill is drained. The Azure queue reports empty
    # directly (idle_timeout == 0), so it exits on the first empty receive --
    # preserving the original single-pass behaviour.
    idle_deadline = None
    try:
        while True:
            messages = queue.receive(max_messages=1)

            if messages:
                idle_deadline = None
                for msg in messages:
                    if process_message(queue, store, msg):
                        processed += 1
                    else:
                        failed += 1
                if not loop:
                    break
                continue

            # No messages this poll.
            if not loop:
                break
            now = time.time()
            if idle_deadline is None:
                idle_deadline = now + queue.idle_timeout
            if now >= idle_deadline:
                print("Queue idle — all batches processed.")
                break
    finally:
        queue.close()

    print(f"\nWorker finished. Processed {processed} batch(es), {failed} failed.")
    return failed


def enqueue_batches(
    corpus_path: str,
    slice_prefix: str = "inputs/slices",
    start_batch: int = 0,
    max_batches: int | None = None,
):
    """Cut the local corpus into per-batch slice objects and enqueue one task each.

    Slicing happens here, once, rather than in every worker. Each worker then
    downloads only its own ~600KB slice instead of the full corpus, which is
    what keeps a replica inside its memory cap.

    ``start_batch``/``max_batches`` bound which batches are enqueued. Batch ids
    stay tied to absolute corpus position, so a partial re-run re-enqueues the
    same ids and overwrites the same shards rather than shifting the numbering.
    """
    queue = get_queue()
    store = get_object_store()
    queue.ensure()

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
        store.put_bytes(blob_path, payload)

        queue.send({
            "batch_id": batch_id,
            "blob_path": blob_path,
            "start_idx": start,
            "end_idx": end,
        })
        if n % 25 == 0 or n == len(selected):
            print(f"  {n}/{len(selected)} slices uploaded and enqueued")

    queue.close()
    print(f"✓ Enqueued {len(selected)} messages")


def main():
    parser = argparse.ArgumentParser(description="Event-driven embedding worker")
    parser.add_argument("--loop", action="store_true", help="Process all messages until the queue drains")
    parser.add_argument("--enqueue", action="store_true", help="Enqueue batch tasks (producer mode)")
    parser.add_argument(
        "--corpus",
        default="data/processed/books_goodreads_v2.jsonl",
        help="Local corpus JSONL to slice and upload (enqueue mode)",
    )
    parser.add_argument("--slice-prefix", default="inputs/slices", help="Object prefix for slice uploads")
    parser.add_argument("--start-batch", type=int, default=0, help="First batch index to enqueue")
    parser.add_argument("--max-batches", type=int, default=None, help="How many batches to enqueue")
    args = parser.parse_args()

    if args.enqueue:
        enqueue_batches(args.corpus, args.slice_prefix, args.start_batch, args.max_batches)
    else:
        # Exit non-zero when any batch failed, so a broken backfill is not
        # reported as a successful job execution.
        failures = worker_loop(loop=args.loop)
        if failures:
            sys.exit(1)


if __name__ == "__main__":
    main()
