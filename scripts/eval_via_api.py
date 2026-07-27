"""Run evaluation against the live API endpoint (no direct Qdrant access needed)."""
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


def main():
    # Load annotated queries
    annotated_path = Path("data/eval/queries_annotated.json")
    queries = load_eval_dataset(annotated_path) if annotated_path.exists() else get_eval_queries()
    queries = [q for q in queries if q.relevant]
    print(f"Evaluating {len(queries)} queries with judgments\n")

    strategies = ["keyword", "vector", "hybrid"]
    results = []

    for mode in strategies:
        metrics_list = []
        total_latency = 0
        for eq in queries:
            url = f"{API}/search?q={urllib.parse.quote(eq.query)}&mode={mode}&top_k=10&understand=false"
            try:
                data = fetch(url)
            except Exception as e:
                print(f"  ERROR on \"{eq.query}\": {e}")
                continue

            retrieved_ids = [r["work_id"].replace("/works/", "") for r in data["results"]]
            latency = data["latency_ms"]
            total_latency += latency

            relevance_map = {j.work_id: j.relevance for j in eq.relevant}
            m = compute_query_metrics(eq.query, retrieved_ids, relevance_map, k=10)
            metrics_list.append(m)

        n = len(metrics_list)
        if n == 0:
            continue

        def avg(attr, ml=metrics_list):
            return sum(getattr(m, attr) for m in ml) / len(ml)

        r = {
            "strategy": mode,
            "queries_evaluated": n,
            "mrr_at_10": round(avg("mrr"), 4),
            "ndcg_at_10": round(avg("ndcg"), 4),
            "recall_at_10": round(avg("recall"), 4),
            "precision_at_10": round(avg("precision"), 4),
            "avg_latency_ms": round(total_latency / n, 1),
        }
        results.append(r)
        print(f"{mode.upper():>12}: MRR={r['mrr_at_10']:.4f}  NDCG={r['ndcg_at_10']:.4f}  Recall={r['recall_at_10']:.4f}  Lat={r['avg_latency_ms']:.0f}ms")

    # hybrid + rerank
    print()
    metrics_list = []
    total_latency = 0
    for eq in queries:
        url = f"{API}/search?q={urllib.parse.quote(eq.query)}&mode=hybrid&top_k=10&understand=false&rerank=true"
        try:
            data = fetch(url, timeout=60)
        except Exception as e:
            print(f"  ERROR on \"{eq.query}\": {e}")
            continue
        retrieved_ids = [r["work_id"].replace("/works/", "") for r in data["results"]]
        latency = data["latency_ms"]
        total_latency += latency
        relevance_map = {j.work_id: j.relevance for j in eq.relevant}
        m = compute_query_metrics(eq.query, retrieved_ids, relevance_map, k=10)
        metrics_list.append(m)

    n = len(metrics_list)
    if n > 0:

        def avg(attr, ml=metrics_list):
            return sum(getattr(m, attr) for m in ml) / len(ml)

        r = {
            "strategy": "hybrid+rerank",
            "queries_evaluated": n,
            "mrr_at_10": round(avg("mrr"), 4),
            "ndcg_at_10": round(avg("ndcg"), 4),
            "recall_at_10": round(avg("recall"), 4),
            "precision_at_10": round(avg("precision"), 4),
            "avg_latency_ms": round(total_latency / n, 1),
        }
        results.append(r)
        print(f"HYBRID+RERANK: MRR={r['mrr_at_10']:.4f}  NDCG={r['ndcg_at_10']:.4f}  Recall={r['recall_at_10']:.4f}  Lat={r['avg_latency_ms']:.0f}ms")

    # Save
    Path("data/eval/results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to data/eval/results.json")


if __name__ == "__main__":
    main()
