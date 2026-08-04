"""
Batch embedding pipeline — embeds books from JSONL using nomic-embed-text-v1.5
and builds a FAISS index for local prototype search.
"""

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Matryoshka dimension — 256 is a trained checkpoint for nomic-embed-text-v1.5
# (supported: 768, 512, 256, 128, 64; 384 was non-standard)
EMBEDDING_DIM = 256
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
BATCH_SIZE = 128


def load_books(jsonl_path: Path, tier_filter: int | None = None) -> list[dict]:
    """Load books from JSONL, optionally filtering by tier."""
    books = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            book = json.loads(line)
            if tier_filter and book["tier"] > tier_filter:
                continue
            books.append(book)
    return books


def build_embedding_texts(books: list[dict]) -> list[str]:
    """Build embedding input strings with Nomic task prefix."""
    texts = []
    for b in books:
        author_str = ", ".join(b["authors"]) if b["authors"] else "Unknown"
        parts = [f"{b['title']} by {author_str}"]

        if b.get("description"):
            parts.append(b["description"][:2000])

        if b.get("subjects"):
            parts.append(", ".join(b["subjects"][:10]))

        if b.get("subject_people"):
            parts.append("People: " + ", ".join(b["subject_people"][:5]))
        if b.get("subject_places"):
            parts.append("Places: " + ", ".join(b["subject_places"][:5]))

        texts.append("search_document: " + ". ".join(parts))
    return texts


def embed_and_index(
    jsonl_path: Path,
    output_dir: Path,
    tier_filter: int | None = None,
    dim: int = EMBEDDING_DIM,
):
    """Full pipeline: load books → embed → build FAISS index → save."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load books
    print(f"Loading books from {jsonl_path}...")
    books = load_books(jsonl_path, tier_filter=tier_filter)
    print(f"  Loaded {len(books):,} books (tier_filter={tier_filter})")

    # 2. Build embedding texts
    texts = build_embedding_texts(books)
    print(f"  Built {len(texts):,} embedding texts")
    print(f"  Sample: {texts[0][:150]}...")

    # 3. Load model
    print(f"\nLoading model: {MODEL_NAME}...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # 4. Encode in batches
    print(f"\nEncoding {len(texts):,} texts (dim={dim}, batch_size={BATCH_SIZE})...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    # Truncate to Matryoshka dimension
    embeddings = embeddings[:, :dim]
    # Re-normalize after truncation
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    elapsed = time.time() - t0
    docs_per_sec = len(texts) / elapsed
    print(f"  Encoded in {elapsed:.1f}s ({docs_per_sec:.0f} docs/sec)")
    print(f"  Embeddings shape: {embeddings.shape}")

    # 5. Build FAISS index
    # Imported here, not at module scope: src/indexing/worker.py imports this
    # module for load_books/build_embedding_texts only, and its container image
    # (Dockerfile.embed) does not install faiss-cpu.
    import faiss

    print("\nBuilding FAISS index...")
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine sim with normalized vecs)
    index.add(embeddings.astype(np.float32))
    print(f"  Index size: {index.ntotal:,} vectors")

    # 6. Save everything
    index_path = output_dir / "faiss.index"
    meta_path = output_dir / "metadata.jsonl"

    faiss.write_index(index, str(index_path))
    print(f"  Saved FAISS index: {index_path}")

    with open(meta_path, "w", encoding="utf-8") as f:
        for book in books:
            f.write(json.dumps(book, ensure_ascii=False) + "\n")
    print(f"  Saved metadata: {meta_path}")

    size_mb = index_path.stat().st_size / 1024 / 1024
    print(f"\n  Index file size: {size_mb:.1f} MB")
    print(f"  Done! {len(books):,} books indexed.")

    return index, books


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed books and build FAISS index")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL path")
    parser.add_argument("--output", type=str, default="data/index", help="Output dir")
    parser.add_argument("--tier", type=int, default=None, help="Filter to this tier or above")
    parser.add_argument("--dim", type=int, default=EMBEDDING_DIM, help="Embedding dimension")
    args = parser.parse_args()

    embed_and_index(
        jsonl_path=Path(args.input),
        output_dir=Path(args.output),
        tier_filter=args.tier,
        dim=args.dim,
    )
