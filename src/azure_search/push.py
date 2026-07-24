"""Push book data + embeddings to Azure AI Search index."""

import os
import json
import time
import numpy as np
import faiss
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
API_KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "books-v1")
API_VERSION = "2024-07-01"

HEADERS = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

BATCH_SIZE = 100  # AI Search limit is 1000 docs/batch, but smaller = safer


def load_data(index_dir: Path):
    """Load FAISS index and metadata."""
    # Load metadata
    metadata_path = index_dir / "metadata.jsonl"
    books = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            books.append(json.loads(line))

    # Load FAISS index to extract vectors
    faiss_path = index_dir / "faiss.index"
    index = faiss.read_index(str(faiss_path))
    n = index.ntotal
    dim = index.d

    # Extract all vectors from FAISS index
    vectors = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        vectors[i] = index.reconstruct(i)

    print(f"Loaded {len(books)} books, {n} vectors (dim={dim})")
    assert len(books) == n, f"Mismatch: {len(books)} books vs {n} vectors"
    return books, vectors


def make_document(book: dict, vector: np.ndarray) -> dict:
    """Convert a book + vector into an AI Search document."""
    # Clean work_id to make a valid document key (alphanumeric, underscores, dashes)
    doc_id = book["work_id"].replace("/works/", "").replace("/", "_")

    doc = {
        "@search.action": "upload",
        "id": doc_id,
        "title": book["title"],
        "authors": ", ".join(book.get("authors", [])),
        "description": book.get("description") or "",
        "subjects": book.get("subjects", [])[:15],
        "work_id": book["work_id"],
        "tier": book.get("tier", 1),
        "embedding": vector.tolist(),
    }

    # Optional fields
    year = book.get("first_publish_year")
    if year and isinstance(year, int) and 1000 <= year <= 2030:
        doc["year"] = year

    cover_url = book.get("cover_url")
    if cover_url:
        doc["cover_url"] = cover_url

    return doc


def upload_batch(documents: list[dict]) -> tuple[int, int]:
    """Upload a batch of documents. Returns (success_count, error_count)."""
    url = f"{ENDPOINT}/indexes/{INDEX_NAME}/docs/index?api-version={API_VERSION}"
    payload = {"value": documents}

    resp = requests.post(url, headers=HEADERS, json=payload)

    if resp.status_code == 200:
        results = resp.json()["value"]
        successes = sum(1 for r in results if r["status"])
        errors = sum(1 for r in results if not r["status"])
        return successes, errors
    elif resp.status_code == 207:
        # Partial success
        results = resp.json()["value"]
        successes = sum(1 for r in results if r["status"])
        errors = sum(1 for r in results if not r["status"])
        for r in results:
            if not r["status"]:
                print(f"  ✗ Doc {r['key']}: {r.get('errorMessage', 'unknown error')}")
        return successes, errors
    else:
        print(f"  ✗ Batch failed: {resp.status_code} - {resp.text[:200]}")
        return 0, len(documents)


def push_all(index_dir: Path = Path("data/index")):
    """Load data and push all documents to Azure AI Search."""
    books, vectors = load_data(index_dir)

    total_success = 0
    total_error = 0
    total = len(books)

    print(f"\nUploading {total} documents in batches of {BATCH_SIZE}...")
    start = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch_books = books[i : i + BATCH_SIZE]
        batch_vectors = vectors[i : i + BATCH_SIZE]

        documents = [
            make_document(book, vec)
            for book, vec in zip(batch_books, batch_vectors)
        ]

        successes, errors = upload_batch(documents)
        total_success += successes
        total_error += errors

        elapsed = time.time() - start
        rate = (i + len(batch_books)) / elapsed if elapsed > 0 else 0
        print(
            f"  [{i + len(batch_books):>6}/{total}] "
            f"+{successes} ok, {errors} err | "
            f"{rate:.0f} docs/s"
        )

        # Small delay to avoid throttling on free tier
        time.sleep(0.5)

    elapsed = time.time() - start
    print(f"\n{'='*50}")
    print(f"Done in {elapsed:.1f}s")
    print(f"  Uploaded: {total_success}")
    print(f"  Errors:   {total_error}")
    print(f"  Rate:     {total_success / elapsed:.0f} docs/s")


if __name__ == "__main__":
    push_all()
