"""Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

Takes (query, document) pairs and produces fine-grained relevance scores.
Unlike bi-encoders, the cross-encoder reads query and document together,
enabling much richer semantic matching at the cost of speed.

Usage:
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank("romance in Scotland", candidates, top_k=10)
"""

import logging
import time
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from src.reranker.config import MAX_SEQUENCE_TOKENS  # noqa: F401
from src.reranker.passage import build_passage

logger = logging.getLogger(__name__)

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
        logger.info("Loading cross-encoder: %s", model_name)
        self.model = CrossEncoder(model_name)
        logger.info("Cross-encoder loaded.")

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
        pairs = [(query, build_passage(doc)) for doc in candidates]

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
