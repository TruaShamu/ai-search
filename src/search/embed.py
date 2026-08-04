"""
Corpus loading and embedding-text construction, shared by the indexing pipeline.

``src/indexing/worker.py`` imports ``MODEL_NAME``, ``BATCH_SIZE`` and
``build_embedding_texts`` from here so the distributed embedding job and any
local experiment build byte-identical inputs to the model.
"""

import json
from pathlib import Path

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
