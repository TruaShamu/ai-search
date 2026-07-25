"""Evaluate spell correction impact on search quality.

Injects realistic typos into eval queries and measures NDCG with/without correction.

Usage:
    python -m src.eval.eval_spelling
    python -m src.eval.eval_spelling --verbose
"""

import argparse
import json
import random
import time
from pathlib import Path

from src.eval.dataset import load_eval_dataset, EvalQuery
from src.eval.metrics import compute_query_metrics, QueryMetrics
from src.azure_search.search import HybridSearchEngine
from src.query.spell import SpellCorrector

K = 10
random.seed(42)

# Realistic typo injection: swap adjacent chars, drop char, double char
def inject_typo(word: str) -> str:
    """Inject a single realistic typo into a word."""
    if len(word) < 4:
        return word
    
    typo_type = random.choice(["swap", "drop", "double", "nearby"])
    pos = random.randint(1, len(word) - 2)
    
    if typo_type == "swap" and pos < len(word) - 1:
        # Swap adjacent characters
        chars = list(word)
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        return "".join(chars)
    elif typo_type == "drop":
        # Drop a character
        return word[:pos] + word[pos + 1:]
    elif typo_type == "double":
        # Double a character
        return word[:pos] + word[pos] + word[pos:]
    else:
        # Nearby key substitution
        nearby = {
            'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sf', 'e': 'wr',
            'f': 'dg', 'g': 'fh', 'h': 'gj', 'i': 'uo', 'j': 'hk',
            'k': 'jl', 'l': 'ko', 'm': 'n', 'n': 'bm', 'o': 'ip',
            'p': 'ol', 'q': 'wa', 'r': 'et', 's': 'ad', 't': 'ry',
            'u': 'yi', 'v': 'cb', 'w': 'qe', 'x': 'zc', 'y': 'tu',
            'z': 'x',
        }
        char = word[pos].lower()
        if char in nearby:
            replacement = random.choice(nearby[char])
            return word[:pos] + replacement + word[pos + 1:]
        return word[:pos] + word[pos + 1:]


def inject_typos_in_query(query: str, n_typos: int = 1) -> str:
    """Inject typos into random words of a query."""
    words = query.split()
    # Only inject into words with 4+ chars (skip short words like "of", "the")
    candidates = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    
    if not candidates:
        return query
    
    # Pick n_typos words to corrupt
    targets = random.sample(candidates, min(n_typos, len(candidates)))
    for idx in targets:
        words[idx] = inject_typo(words[idx])
    
    return " ".join(words)


def run_spell_eval(verbose: bool = False):
    # Load LLM-judged queries
    judged_path = Path("data/eval/queries_llm_judged.json")
    if not judged_path.exists():
        print("ERROR: LLM-judged dataset not found.")
        return

    queries = load_eval_dataset(judged_path)
    queries_with_judgments = [q for q in queries if q.relevant]
    print(f"Eval dataset: {len(queries_with_judgments)} queries with LLM judgments")
    print(f"Injecting 1-2 typos per query, measuring NDCG@{K}\n")

    engine = HybridSearchEngine()
    corrector = SpellCorrector()

    # Generate typo variants
    typo_queries = []
    for eq in queries_with_judgments:
        n_typos = random.choice([1, 1, 1, 2])  # 75% one typo, 25% two typos
        corrupted = inject_typos_in_query(eq.query, n_typos=n_typos)
        typo_queries.append(corrupted)

    # --- A: Clean queries (baseline) ---
    print("=== CLEAN (original queries, no typos) ===")
    clean_metrics = []
    for eq in queries_with_judgments:
        relevance_map = {j.work_id: j.relevance for j in eq.relevant}
        result = engine.search(query=eq.query, top_k=K, mode="hybrid")
        if "error" in result:
            continue
        retrieved_ids = [r["id"] for r in result["results"]]
        m = compute_query_metrics(eq.query, retrieved_ids, relevance_map, k=K)
        clean_metrics.append(m)

    # --- B: Typo queries WITHOUT correction ---
    print("=== TYPOS (no correction) ===")
    typo_metrics = []
    for eq, typo_q in zip(queries_with_judgments, typo_queries):
        relevance_map = {j.work_id: j.relevance for j in eq.relevant}
        result = engine.search(query=typo_q, top_k=K, mode="hybrid")
        if "error" in result:
            continue
        retrieved_ids = [r["id"] for r in result["results"]]
        m = compute_query_metrics(eq.query, retrieved_ids, relevance_map, k=K)
        typo_metrics.append(m)

    # --- C: Typo queries WITH correction ---
    print("=== TYPOS + SPELL CORRECTION ===")
    corrected_metrics = []
    corrections = []
    for eq, typo_q in zip(queries_with_judgments, typo_queries):
        relevance_map = {j.work_id: j.relevance for j in eq.relevant}

        # Apply spell correction
        corrected_q, was_corrected = corrector.correct(typo_q)
        corrections.append({
            "original": eq.query,
            "typo": typo_q,
            "corrected": corrected_q,
            "exact_recovery": corrected_q.lower() == eq.query.lower(),
        })

        result = engine.search(query=corrected_q, top_k=K, mode="hybrid")
        if "error" in result:
            continue
        retrieved_ids = [r["id"] for r in result["results"]]
        m = compute_query_metrics(eq.query, retrieved_ids, relevance_map, k=K)
        corrected_metrics.append(m)

    # --- Report ---
    n = len(clean_metrics)
    clean_ndcg = sum(m.ndcg for m in clean_metrics) / n
    typo_ndcg = sum(m.ndcg for m in typo_metrics) / n
    corrected_ndcg = sum(m.ndcg for m in corrected_metrics) / n

    clean_mrr = sum(m.mrr for m in clean_metrics) / n
    typo_mrr = sum(m.mrr for m in typo_metrics) / n
    corrected_mrr = sum(m.mrr for m in corrected_metrics) / n

    # Recovery rate
    exact_recoveries = sum(1 for c in corrections if c["exact_recovery"])
    recovery_rate = exact_recoveries / len(corrections) * 100

    print(f"\n{'='*70}")
    print(f"{'Condition':<30} {'NDCG@10':<10} {'MRR@10':<10} {'vs Clean':<10}")
    print(f"{'-'*70}")
    print(f"{'Clean (no typos)':<30} {clean_ndcg:<10.4f} {clean_mrr:<10.4f} {'baseline':<10}")
    print(f"{'Typos (no correction)':<30} {typo_ndcg:<10.4f} {typo_mrr:<10.4f} {(typo_ndcg-clean_ndcg)/clean_ndcg*100:+.1f}%")
    print(f"{'Typos + spell correction':<30} {corrected_ndcg:<10.4f} {corrected_mrr:<10.4f} {(corrected_ndcg-clean_ndcg)/clean_ndcg*100:+.1f}%")
    print(f"{'='*70}")
    print(f"\nSpell recovery: {exact_recoveries}/{len(corrections)} queries exactly recovered ({recovery_rate:.0f}%)")
    print(f"NDCG recovered: {(corrected_ndcg - typo_ndcg):.4f} of {(clean_ndcg - typo_ndcg):.4f} lost ({(corrected_ndcg - typo_ndcg) / max(clean_ndcg - typo_ndcg, 0.001) * 100:.0f}%)")

    if verbose:
        print(f"\n--- Per-Query Details ---")
        for i, c in enumerate(corrections):
            delta = corrected_metrics[i].ndcg - typo_metrics[i].ndcg
            marker = "✓" if c["exact_recovery"] else "~" if delta > 0 else "✗"
            print(
                f"  [{marker}] \"{c['typo']}\" → \"{c['corrected']}\""
                f"  (want: \"{c['original']}\")"
                f"  NDCG: {typo_metrics[i].ndcg:.3f}→{corrected_metrics[i].ndcg:.3f}"
            )

    # Save results
    output = {
        "clean_ndcg": round(clean_ndcg, 4),
        "typo_ndcg": round(typo_ndcg, 4),
        "corrected_ndcg": round(corrected_ndcg, 4),
        "recovery_rate": round(recovery_rate, 1),
        "ndcg_recovery_pct": round((corrected_ndcg - typo_ndcg) / max(clean_ndcg - typo_ndcg, 0.001) * 100, 1),
        "corrections": corrections,
    }
    out_path = Path("data/eval/spelling_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate spell correction impact")
    parser.add_argument("--verbose", action="store_true", help="Show per-query details")
    args = parser.parse_args()
    run_spell_eval(verbose=args.verbose)
