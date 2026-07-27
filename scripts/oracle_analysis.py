"""Oracle reranking analysis.

For each query, compute the best possible reranking of the retrieved candidates
given the relevance judgments. This shows the theoretical ceiling — how much
headroom the reranker has. If oracle barely improves over RRF, no reranker can help.

Also computes recall@k curves to show if the issue is retrieval (candidates not
retrieved) vs ranking (candidates retrieved but poorly ordered).
"""

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.dataset import load_eval_dataset, get_eval_queries
from src.eval.metrics import compute_query_metrics

API = "https://booksearch-api.thankfulstone-e6f7cf40.eastus.azurecontainerapps.io"


def fetch(url: str, timeout: int = 30) -> dict:
    resp = urllib.request.urlopen(url, timeout=timeout)
    return json.loads(resp.read())


def oracle_rerank(retrieved_ids: list[str], relevance_map: dict[str, int]) -> list[str]:
    """Reorder retrieved_ids by relevance score (descending). Best possible reranking."""
    return sorted(retrieved_ids, key=lambda x: relevance_map.get(x, 0), reverse=True)


def main():
    # Load annotated queries
    annotated_path = Path("data/eval/queries_annotated.json")
    queries = load_eval_dataset(annotated_path) if annotated_path.exists() else get_eval_queries()
    queries = [q for q in queries if q.relevant]
    print(f"Oracle analysis: {len(queries)} queries with judgments\n")

    # Fetch hybrid results (our best mode) and compute oracle ceiling
    print("Fetching hybrid results from live API...")
    hybrid_metrics = []
    oracle_metrics = []
    recall_at_k = {5: [], 10: [], 20: [], 50: []}

    for eq in queries:
        relevance_map = {j.work_id: j.relevance for j in eq.relevant}

        # Fetch top-50 hybrid results (large pool for recall curve)
        url = f"{API}/search?q={urllib.parse.quote(eq.query)}&mode=hybrid&top_k=50&understand=false"
        try:
            data = fetch(url)
        except Exception as e:
            print(f"  ERROR: {eq.query}: {e}")
            continue

        retrieved_ids = [r["work_id"].replace("/works/", "") for r in data["results"]]

        # Actual ranking metrics at k=10
        actual_at_10 = compute_query_metrics(eq.query, retrieved_ids[:10], relevance_map, k=10)
        hybrid_metrics.append(actual_at_10)

        # Oracle: best possible reranking of the SAME candidates
        oracle_order = oracle_rerank(retrieved_ids[:10], relevance_map)
        oracle_at_10 = compute_query_metrics(eq.query, oracle_order, relevance_map, k=10)
        oracle_metrics.append(oracle_at_10)

        # Recall@k curve (how many relevant docs are in top-k candidates)
        for k in recall_at_k:
            pool = retrieved_ids[:k]
            found = sum(1 for doc_id in pool if relevance_map.get(doc_id, 0) > 0)
            total_rel = sum(1 for r in relevance_map.values() if r > 0)
            recall_at_k[k].append(found / total_rel if total_rel else 0)

    n = len(hybrid_metrics)
    if n == 0:
        print("No queries evaluated!")
        return

    def avg(metrics_list, attr):
        return sum(getattr(m, attr) for m in metrics_list) / len(metrics_list)

    print(f"\n{'='*60}")
    print(f"ORACLE ANALYSIS (n={n} queries)")
    print(f"{'='*60}\n")

    print("Hybrid RRF (actual) vs Oracle (best-possible reranking of same candidates):\n")
    print(f"  {'Metric':<15} {'Hybrid':>10} {'Oracle':>10} {'Headroom':>10} {'% Gain':>10}")
    print(f"  {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for attr, label in [("mrr", "MRR@10"), ("ndcg", "NDCG@10"), ("recall", "Recall@10")]:
        h = avg(hybrid_metrics, attr)
        o = avg(oracle_metrics, attr)
        headroom = o - h
        pct = (headroom / h * 100) if h > 0 else 0
        print(f"  {label:<15} {h:>10.4f} {o:>10.4f} {headroom:>+10.4f} {pct:>+9.1f}%")

    print("\n\nRecall@k curve (fraction of relevant docs retrieved):\n")
    print(f"  {'k':<5} {'Mean Recall':>12} {'Interpretation'}")
    print(f"  {'-'*5} {'-'*12} {'-'*40}")
    for k in sorted(recall_at_k.keys()):
        mean_r = sum(recall_at_k[k]) / len(recall_at_k[k])
        interp = ""
        if k == 10:
            interp = "<-- what user sees"
        elif k == 50:
            interp = "<-- what reranker could use"
        print(f"  {k:<5} {mean_r:>12.4f} {interp}")

    # Key insight
    h_ndcg = avg(hybrid_metrics, "ndcg")
    o_ndcg = avg(oracle_metrics, "ndcg")
    headroom_pct = (o_ndcg - h_ndcg) / h_ndcg * 100 if h_ndcg > 0 else 0

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    print(f"\n  NDCG headroom from perfect reranking: {o_ndcg - h_ndcg:+.4f} ({headroom_pct:+.1f}%)")

    recall_50 = sum(recall_at_k[50]) / len(recall_at_k[50])
    recall_10 = sum(recall_at_k[10]) / len(recall_at_k[10])

    if headroom_pct < 10:
        print("  → RRF ordering is near-optimal. Reranking has <10% room to improve.")
        print("  → The bottleneck is RETRIEVAL, not RANKING.")
    else:
        print(f"  → There IS room for a better reranker (+{headroom_pct:.0f}% possible).")
        print("  → But ms-marco-MiniLM fails to capture it (short book metadata).")

    if recall_50 > recall_10 * 1.3:
        print(f"\n  Recall@50 ({recall_50:.3f}) >> Recall@10 ({recall_10:.3f})")
        print("  → Retrieving more candidates WOULD help if reranker worked.")
    else:
        print(f"\n  Recall@50 ({recall_50:.3f}) ~ Recall@10 ({recall_10:.3f})")
        print("  → Deeper retrieval doesn't add many relevant docs.")


if __name__ == "__main__":
    main()
