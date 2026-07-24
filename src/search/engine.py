"""
FAISS search engine — loads index + metadata, handles queries.
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_DIM = 384
MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"


class BookSearchEngine:
    """Local FAISS-backed search engine for prototype."""

    def __init__(self, index_dir: Path, model: SentenceTransformer | None = None):
        index_path = index_dir / "faiss.index"
        meta_path = index_dir / "metadata.jsonl"

        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        print(f"  {self.index.ntotal:,} vectors loaded")

        print(f"Loading metadata from {meta_path}...")
        self.books = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                self.books.append(json.loads(line))
        print(f"  {len(self.books):,} books loaded")

        if model:
            self.model = model
        else:
            print(f"Loading model: {MODEL_NAME}...")
            self.model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

        self.dim = self.index.d

    def search(
        self,
        query: str,
        top_k: int = 10,
        tier_filter: int | None = None,
        year_min: int | None = None,
        year_max: int | None = None,
    ) -> list[dict]:
        """Search for books matching the query."""
        # Embed query with Nomic prefix
        query_text = f"search_query: {query}"
        query_vec = self.model.encode(
            [query_text], normalize_embeddings=True
        )
        query_vec = query_vec[:, :self.dim]
        # Re-normalize after Matryoshka truncation
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

        # Search more than top_k to allow for post-filtering
        fetch_k = top_k * 5 if (tier_filter or year_min or year_max) else top_k
        scores, indices = self.index.search(query_vec.astype(np.float32), fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            book = self.books[idx]

            # Post-filters
            if tier_filter and book["tier"] > tier_filter:
                continue
            if year_min and (not book.get("first_publish_year") or book["first_publish_year"] < year_min):
                continue
            if year_max and (not book.get("first_publish_year") or book["first_publish_year"] > year_max):
                continue

            result = {
                "rank": len(results) + 1,
                "score": float(score),
                "title": book["title"],
                "authors": book["authors"],
                "description": book.get("description"),
                "subjects": book["subjects"],
                "first_publish_year": book.get("first_publish_year"),
                "cover_url": f"https://covers.openlibrary.org/b/id/{book['cover_id']}-M.jpg"
                    if book.get("cover_id") else None,
                "work_id": book["work_id"],
                "tier": book["tier"],
            }
            results.append(result)
            if len(results) >= top_k:
                break

        return results

    def search_formatted(self, query: str, top_k: int = 10, **kwargs) -> str:
        """Search and return a formatted string for CLI display."""
        results = self.search(query, top_k=top_k, **kwargs)
        if not results:
            return f"No results for: {query}"

        lines = [f"Results for: \"{query}\"", ""]
        for r in results:
            authors = ", ".join(r["authors"]) if r["authors"] else "Unknown"
            year = f" ({r['first_publish_year']})" if r.get("first_publish_year") else ""
            subjects = ", ".join(r["subjects"][:5]) if r["subjects"] else ""
            lines.append(f"  [{r['rank']}] {r['title']}{year}  (score: {r['score']:.3f})")
            lines.append(f"      by {authors}")
            if subjects:
                lines.append(f"      Subjects: {subjects}")
            if r.get("description"):
                lines.append(f"      {r['description'][:150]}...")
            lines.append("")

        return "\n".join(lines)
