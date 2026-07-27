"""Evaluate LLM query expansion impact on search quality.

Compares:
  A) Baseline: hybrid search with raw query
  B) Expanded: hybrid search with LLM-expanded query (BM25 gets extra terms)

Usage:
    python -m src.eval.eval_expansion
    python -m src.eval.eval_expansion --verbose
"""

import argparse
import json
from pathlib import Path

from src.eval.dataset import load_eval_dataset
from src.eval.metrics import compute_query_metrics
from src.azure_search.search import HybridSearchEngine
from src.query.expansion import QueryExpander

K = 10


def run_expansion_eval(verbose: bool = False):
    judged_path = Path("data/eval/queries_llm_judged.json")
    if not judged_path.exists():
        print("ERROR: LLM-judged dataset not found.")
        return

    queries = load_eval_dataset(judged_path)
    queries_with_judgments = [q for q in queries if q.relevant]
    print(f"Eval dataset: {len(queries_with_judgments)} queries with LLM judgments")
    print(f"k = {K}\n")

    engine = HybridSearchEngine()
    expander = QueryExpander()

    # --- Baseline: hybrid with raw query ---
    print("=== BASELINE (no expansion) ===")
    baseline_metrics = []
    baseline_latency = 0

    for eq in queries_with_judgments:
        relevance_map = {j.work_id: j.relevance for j in eq.relevant}
        result = engine.search(query=eq.query, top_k=K, mode="hybrid")
        if "error" in result:
            continue
        retrieved_ids = [r["id"] for r in result["results"]]
        baseline_latency += result["latency_ms"]
        m = compute_query_metrics(eq.query, retrieved_ids, relevance_map, k=K)
        baseline_metrics.append(m)

    # --- Expanded: hybrid with LLM-expanded query ---
    print("=== EXPANDED (LLM query expansion) ===")
    expanded_metrics = []
    expanded_latency = 0
    expansion_latency = 0
    expansions = []

    for eq in queries_with_judgments:
        relevance_map = {j.work_id: j.relevance for j in eq.relevant}

        # Expand query
        expanded_q, terms, exp_lat = expander.expand(eq.query)
        expansion_latency += exp_lat

        result = engine.search(query=expanded_q, top_k=K, mode="hybrid")
        if "error" in result:
            continue
        retrieved_ids = [r["id"] for r in result["results"]]
        expanded_latency += result["latency_ms"]

        m = compute_query_metrics(eq.query, retrieved_ids, relevance_map, k=K)
        expanded_metrics.append(m)

        expansions.append({
            "query": eq.query,
            "terms": terms,
            "expansion_latency_ms": round(exp_lat, 1),
            "baseline_ndcg": baseline_metrics[len(expanded_metrics) - 1].ndcg,
            "expanded_ndcg": m.ndcg,
            "delta": round(m.ndcg - baseline_metrics[len(expanded_metrics) - 1].ndcg, 4),
        })

    # --- Report ---
    n = len(baseline_metrics)
    b_ndcg = sum(m.ndcg for m in baseline_metrics) / n
    b_mrr = sum(m.mrr for m in baseline_metrics) / n
    b_recall = sum(m.recall for m in baseline_metrics) / n

    e_ndcg = sum(m.ndcg for m in expanded_metrics) / n
    e_mrr = sum(m.mrr for m in expanded_metrics) / n
    e_recall = sum(m.recall for m in expanded_metrics) / n

    avg_exp_latency = expansion_latency / n

    print(f"\n{'='*70}")
    print(f"{'Strategy':<30} {'NDCG@10':<10} {'MRR@10':<10} {'Recall@10':<12} {'Latency':<10}")
    print(f"{'-'*70}")
    print(f"{'Baseline (no expansion)':<30} {b_ndcg:<10.4f} {b_mrr:<10.4f} {b_recall:<12.4f} {baseline_latency/n:<10.0f}ms")
    print(f"{'LLM expanded':<30} {e_ndcg:<10.4f} {e_mrr:<10.4f} {e_recall:<12.4f} {expanded_latency/n:<10.0f}ms")
    print(f"{'-'*70}")

    delta_ndcg = e_ndcg - b_ndcg
    pct = (delta_ndcg / b_ndcg) * 100 if b_ndcg > 0 else 0
    sign = "+" if delta_ndcg >= 0 else ""
    print(f"Delta NDCG: {sign}{delta_ndcg:.4f} ({sign}{pct:.1f}%)")
    print(f"Avg expansion latency: {avg_exp_latency:.0f}ms")
    print(f"{'='*70}")

    # Count improvements vs regressions
    improved = sum(1 for e in expansions if e["delta"] > 0.01)
    regressed = sum(1 for e in expansions if e["delta"] < -0.01)
    neutral = n - improved - regressed
    print(f"\nImproved: {improved}/{n} | Neutral: {neutral}/{n} | Regressed: {regressed}/{n}")

    if verbose:
        print("\n--- Per-Query Deltas (sorted by impact) ---")
        sorted_exp = sorted(expansions, key=lambda x: x["delta"], reverse=True)
        for e in sorted_exp:
            if abs(e["delta"]) > 0.001:
                sign = "+" if e["delta"] > 0 else ""
                marker = "↑" if e["delta"] > 0.01 else ("↓" if e["delta"] < -0.01 else "=")
                print(
                    f"  [{marker}] {e['query'][:45]:<45} "
                    f"NDCG: {e['baseline_ndcg']:.3f}→{e['expanded_ndcg']:.3f} ({sign}{e['delta']:.3f})"
                    f"  terms: {e['terms'][:3]}"
                )

    # Save
    output = {
        "baseline_ndcg": round(b_ndcg, 4),
        "expanded_ndcg": round(e_ndcg, 4),
        "delta_ndcg": round(delta_ndcg, 4),
        "delta_pct": round(pct, 1),
        "avg_expansion_latency_ms": round(avg_exp_latency, 0),
        "improved": improved,
        "regressed": regressed,
        "neutral": neutral,
        "expansions": expansions,
    }
    out_path = Path("data/eval/expansion_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM query expansion")
    parser.add_argument("--verbose", action="store_true", help="Show per-query details")
    args = parser.parse_args()
    run_expansion_eval(verbose=args.verbose)
