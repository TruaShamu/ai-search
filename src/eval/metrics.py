"""Evaluation metrics for information retrieval.

Implements standard IR metrics:
  - MRR@k (Mean Reciprocal Rank)
  - NDCG@k (Normalized Discounted Cumulative Gain)
  - Recall@k
  - Precision@k
"""

import math
from dataclasses import dataclass


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query: str
    mrr: float
    ndcg: float
    recall: float
    precision: float
    relevant_found: int
    relevant_total: int


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Reciprocal rank of the first relevant document in top-k."""
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def dcg(relevances: list[float], k: int = 10) -> float:
    """Discounted Cumulative Gain at k."""
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        if rel > 0:
            score += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    return score


def ndcg_at_k(
    retrieved_ids: list[str],
    relevance_map: dict[str, int],
    k: int = 10,
) -> float:
    """Normalized Discounted Cumulative Gain at k.
    
    relevance_map: {doc_id: relevance_score} where score is 0/1/2
    """
    # Actual DCG from retrieved order
    actual_relevances = [relevance_map.get(doc_id, 0) for doc_id in retrieved_ids[:k]]
    actual_dcg = dcg(actual_relevances, k)

    # Ideal DCG (best possible ranking)
    ideal_relevances = sorted(relevance_map.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal_relevances, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Fraction of relevant documents found in top-k."""
    if not relevant_ids:
        return 0.0
    found = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return found / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Fraction of top-k results that are relevant."""
    if k == 0:
        return 0.0
    found = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return found / k


def compute_query_metrics(
    query: str,
    retrieved_ids: list[str],
    relevance_map: dict[str, int],
    k: int = 10,
) -> QueryMetrics:
    """Compute all metrics for a single query."""
    relevant_ids = {doc_id for doc_id, rel in relevance_map.items() if rel > 0}

    return QueryMetrics(
        query=query,
        mrr=reciprocal_rank(retrieved_ids, relevant_ids, k),
        ndcg=ndcg_at_k(retrieved_ids, relevance_map, k),
        recall=recall_at_k(retrieved_ids, relevant_ids, k),
        precision=precision_at_k(retrieved_ids, relevant_ids, k),
        relevant_found=sum(1 for d in retrieved_ids[:k] if d in relevant_ids),
        relevant_total=len(relevant_ids),
    )
