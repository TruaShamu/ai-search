"""Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

Takes (query, document) pairs and produces fine-grained relevance scores.
Unlike bi-encoders, the cross-encoder reads query and document together,
enabling much richer semantic matching at the cost of speed.

Usage:
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("romance in Scotland", candidates, top_k=10)
"""

import time
from dataclasses import dataclass

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankResult:
    """A single reranked result with original and reranked scores."""
    document: dict
    rerank_score: float
    original_score: float
    original_rank: int
    new_rank: int


class CrossEncoderReranker:
    """Reranks search candidates using a cross-encoder model."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        print(f"Loading cross-encoder: {model_name}...")
        self.model = CrossEncoder(model_name)
        print("Cross-encoder loaded.")

    def _build_passage(self, doc: dict) -> str:
        """Build a passage string from a document for reranking."""
        parts = []
        if doc.get("title"):
            parts.append(doc["title"])
        if doc.get("authors"):
            parts.append(f"by {doc['authors']}")
        if doc.get("description"):
            # Use first 300 chars of description for reranking
            desc = doc["description"][:300]
            parts.append(desc)
        if doc.get("subjects"):
            subjects = doc["subjects"]
            if isinstance(subjects, list):
                parts.append(f"Subjects: {', '.join(subjects[:5])}")
        return " | ".join(parts)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
    ) -> dict:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: The search query
            candidates: List of document dicts from retriever
            top_k: Number of results to return after reranking

        Returns:
            Dict with reranked results and metadata
        """
        if not candidates:
            return {"results": [], "latency_ms": 0, "candidates_scored": 0}

        start = time.time()

        # Build (query, passage) pairs
        pairs = [(query, self._build_passage(doc)) for doc in candidates]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Combine with original data and sort by rerank score
        scored = []
        for i, (doc, score) in enumerate(zip(candidates, scores)):
            scored.append(RerankResult(
                document=doc,
                rerank_score=float(score),
                original_score=doc.get("score", 0),
                original_rank=i + 1,
                new_rank=0,  # assigned after sort
            ))

        # Sort by rerank score (descending)
        scored.sort(key=lambda x: x.rerank_score, reverse=True)

        # Assign new ranks and take top_k
        for i, item in enumerate(scored):
            item.new_rank = i + 1

        top_results = scored[:top_k]
        latency_ms = (time.time() - start) * 1000

        # Format output
        results = []
        for item in top_results:
            result = dict(item.document)
            result["rerank_score"] = round(item.rerank_score, 4)
            result["original_rank"] = item.original_rank
            result["rank_change"] = item.original_rank - item.new_rank
            results.append(result)

        return {
            "results": results,
            "latency_ms": round(latency_ms, 1),
            "candidates_scored": len(candidates),
        }


if __name__ == "__main__":

    # Quick test with synthetic data
    reranker = CrossEncoderReranker()

    query = "romance set in Scotland"
    candidates = [
        {"title": "Desmond goes to Scotland", "authors": "Althea", "description": "A children's picture book about a bear visiting Scotland.", "subjects": ["Children's fiction"], "score": 24.7},
        {"title": "Seducing the Highlander", "authors": "Emma Wildes", "description": "Three stories of romance, adventure, and passion in the Scottish Highlands.", "subjects": ["Fiction, Romance, Historical"], "score": 0.80},
        {"title": "Computational Logic and Set Theory", "authors": "Jacob Schwartz", "description": "A technical book about mathematical logic.", "subjects": ["Mathematics"], "score": 19.5},
        {"title": "The Bride", "authors": "Julie Garwood", "description": "A Scottish laird must take an English bride. His choice was Jamie, a feisty beauty.", "subjects": ["Fiction, Romance, Historical", "Scotland In Fiction"], "score": 0.78},
    ]

    print(f"\nQuery: \"{query}\"")
    print(f"{'='*60}")
    print("\nBefore reranking (retriever order):")
    for i, c in enumerate(candidates, 1):
        print(f"  {i}. {c['title']} (score: {c['score']})")

    result = reranker.rerank(query, candidates, top_k=4)

    print(f"\nAfter reranking ({result['latency_ms']}ms, {result['candidates_scored']} scored):")
    for r in result["results"]:
        change = r["rank_change"]
        arrow = f"+{change}" if change > 0 else str(change) if change < 0 else "="
        print(f"  {r['rerank_score']:+.4f} | {r['title']} (was rank {r['original_rank']}, {arrow})")
