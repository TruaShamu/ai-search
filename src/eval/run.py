"""Evaluation harness — runs queries against all retrieval strategies and reports metrics.

Usage:
    python -m src.eval.run                    # Run full eval
    python -m src.eval.run --strategy hybrid  # Single strategy
    python -m src.eval.run --verbose          # Show per-query results
"""

import argparse
import time
import json
from dataclasses import asdict
from pathlib import Path

from src.eval.dataset import get_eval_queries, EvalQuery
from src.eval.metrics import compute_query_metrics, QueryMetrics
from src.azure_search.search import HybridSearchEngine


STRATEGIES = ["keyword", "vector", "hybrid"]
K = 10


def run_query(engine, query: str, mode: str, top_k: int = K, rerank: bool = False):
    """Run a single query and return list of retrieved doc IDs."""
    result = engine.search(query=query, top_k=top_k if not rerank else top_k * 5, mode=mode)

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return [], 0

    retrieved_ids = [r["id"] for r in result["results"]]
    latency = result["latency_ms"]

    if rerank:
        from src.reranker.onnx_reranker import OnnxReranker
        reranker = get_reranker()
        rerank_result = reranker.rerank(query=query, candidates=result["results"], top_k=top_k)
        retrieved_ids = [r["id"] for r in rerank_result["results"]]
        latency += rerank_result["latency_ms"]

    return retrieved_ids, latency


# Cache reranker instance
_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from src.reranker.onnx_reranker import OnnxReranker
        _reranker = OnnxReranker()
    return _reranker


def evaluate_strategy(
    engine,
    queries: list[EvalQuery],
    mode: str,
    rerank: bool = False,
    verbose: bool = False,
) -> dict:
    """Evaluate a single strategy across all queries with judgments."""
    all_metrics: list[QueryMetrics] = []
    total_latency = 0
    queries_with_judgments = [q for q in queries if q.relevant]

    if not queries_with_judgments:
        print("  No queries with relevance judgments!")
        return {}

    for eq in queries_with_judgments:
        # Build relevance map: work_id (without prefix) → relevance
        relevance_map = {j.work_id: j.relevance for j in eq.relevant}

        retrieved_ids, latency = run_query(engine, eq.query, mode, rerank=rerank)
        total_latency += latency

        metrics = compute_query_metrics(
            query=eq.query,
            retrieved_ids=retrieved_ids,
            relevance_map=relevance_map,
            k=K,
        )
        all_metrics.append(metrics)

        if verbose:
            found_marker = "+" if metrics.relevant_found > 0 else "-"
            print(
                f"  [{found_marker}] \"{eq.query}\" "
                f"MRR={metrics.mrr:.2f} NDCG={metrics.ndcg:.2f} "
                f"found={metrics.relevant_found}/{metrics.relevant_total}"
            )

    # Aggregate
    n = len(all_metrics)
    avg_mrr = sum(m.mrr for m in all_metrics) / n
    avg_ndcg = sum(m.ndcg for m in all_metrics) / n
    avg_recall = sum(m.recall for m in all_metrics) / n
    avg_precision = sum(m.precision for m in all_metrics) / n
    avg_latency = total_latency / n

    return {
        "strategy": f"{mode}+rerank" if rerank else mode,
        "queries_evaluated": n,
        "mrr_at_10": round(avg_mrr, 4),
        "ndcg_at_10": round(avg_ndcg, 4),
        "recall_at_10": round(avg_recall, 4),
        "precision_at_10": round(avg_precision, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "per_query": [asdict(m) for m in all_metrics],
    }


def run_eval(strategies: list[str] = None, rerank: bool = True, verbose: bool = False, annotated: bool = True, llm_judged: bool = False):
    """Run full evaluation across all strategies."""
    if strategies is None:
        strategies = STRATEGIES

    # Choose judgment source
    from src.eval.dataset import load_eval_dataset
    if llm_judged:
        judged_path = Path("data/eval/queries_llm_judged.json")
        if judged_path.exists():
            queries = load_eval_dataset(judged_path)
            print("[Using LLM-as-judge relevance judgments]")
        else:
            print("WARNING: LLM judgments not found, falling back to heuristic")
            queries = get_eval_queries()
    elif annotated:
        annotated_path = Path("data/eval/queries_annotated.json")
        if annotated_path.exists():
            queries = load_eval_dataset(annotated_path)
        else:
            queries = get_eval_queries()
    else:
        queries = get_eval_queries()

    queries_with_judgments = [q for q in queries if q.relevant]
    print(f"Evaluation dataset: {len(queries)} queries ({len(queries_with_judgments)} with judgments)")
    print(f"Strategies: {strategies}" + (" + rerank" if rerank else ""))
    print(f"k = {K}")
    print()

    engine = HybridSearchEngine()
    results = []

    for mode in strategies:
        print(f"--- {mode.upper()} ---")
        result = evaluate_strategy(engine, queries, mode, rerank=False, verbose=verbose)
        if result:
            results.append(result)
            print(
                f"  MRR@10={result['mrr_at_10']:.3f}  "
                f"NDCG@10={result['ndcg_at_10']:.3f}  "
                f"Recall@10={result['recall_at_10']:.3f}  "
                f"Latency={result['avg_latency_ms']:.0f}ms"
            )
        print()

    # Also run hybrid+rerank if requested
    if rerank:
        print("--- HYBRID+RERANK ---")
        result = evaluate_strategy(engine, queries, "hybrid", rerank=True, verbose=verbose)
        if result:
            results.append(result)
            print(
                f"  MRR@10={result['mrr_at_10']:.3f}  "
                f"NDCG@10={result['ndcg_at_10']:.3f}  "
                f"Recall@10={result['recall_at_10']:.3f}  "
                f"Latency={result['avg_latency_ms']:.0f}ms"
            )
        print()

    # Summary table
    print("=" * 70)
    print(f"{'Strategy':<20} {'MRR@10':<10} {'NDCG@10':<10} {'Recall@10':<12} {'Latency':<10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['strategy']:<20} "
            f"{r['mrr_at_10']:<10.4f} "
            f"{r['ndcg_at_10']:<10.4f} "
            f"{r['recall_at_10']:<12.4f} "
            f"{r['avg_latency_ms']:<10.0f}ms"
        )
    print("=" * 70)

    # Save results
    output_path = Path("data/eval/results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run search evaluation")
    parser.add_argument("--strategy", choices=STRATEGIES, help="Single strategy to evaluate")
    parser.add_argument("--no-rerank", action="store_true", help="Skip reranker evaluation")
    parser.add_argument("--verbose", action="store_true", help="Show per-query details")
    parser.add_argument("--llm-judged", action="store_true", help="Use LLM-as-judge annotations")
    args = parser.parse_args()

    strategies = [args.strategy] if args.strategy else None
    run_eval(strategies=strategies, rerank=not args.no_rerank, verbose=args.verbose, llm_judged=args.llm_judged)
