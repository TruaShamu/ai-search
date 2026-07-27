"""Evaluate query-adaptive routing vs baseline (hybrid for all queries).

Compares:
  A) Baseline: all queries → hybrid mode (no query understanding)
  B) Routed:   each query → intent-based mode selection + spell correction

Usage:
    python -m src.eval.eval_routing
    python -m src.eval.eval_routing --verbose
"""

import argparse
import json
from pathlib import Path

from src.eval.dataset import load_eval_dataset
from src.eval.metrics import compute_query_metrics
from src.azure_search.search import HybridSearchEngine
from src.query.pipeline import QueryPipeline

K = 10


def run_eval_routing(verbose: bool = False):
    # Load LLM-judged queries
    judged_path = Path("data/eval/queries_llm_judged.json")
    if not judged_path.exists():
        print("ERROR: LLM-judged dataset not found. Run llm_judge.py first.")
        return

    queries = load_eval_dataset(judged_path)
    queries_with_judgments = [q for q in queries if q.relevant]
    print(f"Eval dataset: {len(queries_with_judgments)} queries with LLM judgments")
    print(f"k = {K}\n")

    engine = HybridSearchEngine()
    qp = QueryPipeline()

    # --- Baseline: hybrid for everything ---
    print("=== BASELINE (hybrid for all) ===")
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

    # --- Routed: query understanding → adaptive mode ---
    print("=== ROUTED (intent-adaptive) ===")
    routed_metrics = []
    routed_latency = 0
    routing_decisions = []

    for eq in queries_with_judgments:
        relevance_map = {j.work_id: j.relevance for j in eq.relevant}

        # Apply query understanding
        analysis = qp.process(eq.query)
        search_query = analysis.corrected
        search_mode = analysis.search_mode

        # For similar_to, use extracted title
        if analysis.intent.value == "similar_to" and analysis.extracted_title:
            search_query = f"novel titled {analysis.extracted_title}"

        # Apply detected year filters
        year_min = analysis.filters.get("year_min")
        year_max = analysis.filters.get("year_max")

        # Intent-detected filters are informational only (not applied as hard constraints)
        year_min = None
        year_max = None

        result = engine.search(
            query=search_query, top_k=K, mode=search_mode,
            year_min=year_min, year_max=year_max,
        )
        if "error" in result:
            continue
        retrieved_ids = [r["id"] for r in result["results"]]
        routed_latency += result["latency_ms"]

        m = compute_query_metrics(eq.query, retrieved_ids, relevance_map, k=K)
        routed_metrics.append(m)

        routing_decisions.append({
            "query": eq.query,
            "corrected": analysis.corrected,
            "intent": analysis.intent.value,
            "routed_mode": search_mode,
            "was_corrected": analysis.was_corrected,
            "baseline_ndcg": baseline_metrics[len(routed_metrics) - 1].ndcg,
            "routed_ndcg": m.ndcg,
            "delta": round(m.ndcg - baseline_metrics[len(routed_metrics) - 1].ndcg, 4),
        })

    # --- Report ---
    n = len(baseline_metrics)
    b_ndcg = sum(m.ndcg for m in baseline_metrics) / n
    b_mrr = sum(m.mrr for m in baseline_metrics) / n
    b_recall = sum(m.recall for m in baseline_metrics) / n

    r_ndcg = sum(m.ndcg for m in routed_metrics) / n
    r_mrr = sum(m.mrr for m in routed_metrics) / n
    r_recall = sum(m.recall for m in routed_metrics) / n

    print(f"\n{'='*70}")
    print(f"{'Strategy':<25} {'NDCG@10':<10} {'MRR@10':<10} {'Recall@10':<12} {'Latency':<10}")
    print(f"{'-'*70}")
    print(f"{'Baseline (hybrid)':<25} {b_ndcg:<10.4f} {b_mrr:<10.4f} {b_recall:<12.4f} {baseline_latency/n:<10.0f}ms")
    print(f"{'Routed (adaptive)':<25} {r_ndcg:<10.4f} {r_mrr:<10.4f} {r_recall:<12.4f} {routed_latency/n:<10.0f}ms")
    print(f"{'-'*70}")

    delta_ndcg = r_ndcg - b_ndcg
    pct = (delta_ndcg / b_ndcg) * 100 if b_ndcg > 0 else 0
    sign = "+" if delta_ndcg >= 0 else ""
    print(f"Delta NDCG: {sign}{delta_ndcg:.4f} ({sign}{pct:.1f}%)")
    print(f"{'='*70}")

    # Per-intent breakdown
    intent_groups = {}
    for d in routing_decisions:
        intent = d["intent"]
        if intent not in intent_groups:
            intent_groups[intent] = {"baseline": [], "routed": [], "queries": []}
        intent_groups[intent]["baseline"].append(d["baseline_ndcg"])
        intent_groups[intent]["routed"].append(d["routed_ndcg"])
        intent_groups[intent]["queries"].append(d["query"])

    print(f"\n{'Intent':<15} {'N':<5} {'Baseline NDCG':<15} {'Routed NDCG':<15} {'Delta':<10}")
    print(f"{'-'*60}")
    for intent, data in sorted(intent_groups.items(), key=lambda x: -len(x[1]["baseline"])):
        n_i = len(data["baseline"])
        b_i = sum(data["baseline"]) / n_i
        r_i = sum(data["routed"]) / n_i
        d_i = r_i - b_i
        sign = "+" if d_i >= 0 else ""
        print(f"{intent:<15} {n_i:<5} {b_i:<15.4f} {r_i:<15.4f} {sign}{d_i:.4f}")

    # Show per-query changes if verbose
    if verbose:
        print("\n--- Per-Query Deltas (sorted by impact) ---")
        sorted_decisions = sorted(routing_decisions, key=lambda x: x["delta"])
        for d in sorted_decisions:
            if d["delta"] != 0:
                sign = "+" if d["delta"] > 0 else ""
                marker = "✓" if d["delta"] > 0 else "✗"
                print(
                    f"  [{marker}] {d['query'][:50]:<50} "
                    f"intent={d['intent']:<12} mode={d['routed_mode']:<8} "
                    f"NDCG: {d['baseline_ndcg']:.3f}→{d['routed_ndcg']:.3f} ({sign}{d['delta']:.3f})"
                )

    # Save results
    output = {
        "baseline": {"ndcg": round(b_ndcg, 4), "mrr": round(b_mrr, 4), "recall": round(b_recall, 4)},
        "routed": {"ndcg": round(r_ndcg, 4), "mrr": round(r_mrr, 4), "recall": round(r_recall, 4)},
        "delta_ndcg": round(delta_ndcg, 4),
        "delta_pct": round(pct, 1),
        "per_intent": {k: {"n": len(v["baseline"]), "baseline_ndcg": round(sum(v["baseline"])/len(v["baseline"]), 4), "routed_ndcg": round(sum(v["routed"])/len(v["routed"]), 4)} for k, v in intent_groups.items()},
        "decisions": routing_decisions,
    }
    out_path = Path("data/eval/routing_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate query-adaptive routing")
    parser.add_argument("--verbose", action="store_true", help="Show per-query deltas")
    args = parser.parse_args()
    run_eval_routing(verbose=args.verbose)
