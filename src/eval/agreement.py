from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Optional

MIN_NEGATIVE_RATE = 0.20


def grade_distribution(results: dict[str, list[dict]]) -> dict:
    """Compute grade distribution and return stats dict. Prints warnings."""
    all_grades = [
        j["grade"] for jlist in results.values() for j in jlist
        if j.get("grade") is not None
    ]
    total_pairs = sum(len(jlist) for jlist in results.values())
    none_count = sum(
        1 for jlist in results.values() for j in jlist if j.get("grade") is None
    )
    no_consensus = sum(
        1 for jlist in results.values() for j in jlist
        if j.get("reasoning") == "no_consensus"
    )
    low_conf = sum(
        1 for jlist in results.values() for j in jlist
        if j.get("low_confidence")
    )

    if not all_grades:
        print("WARNING: no valid grades found!")
        return {"total": 0, "distribution": {}, "negative_rate": 0.0}

    counts = Counter(all_grades)
    total = len(all_grades)
    dist = {g: counts.get(g, 0) for g in (0, 1, 2)}
    neg_rate = dist[0] / total if total else 0.0

    stats = {
        "total_judged": total,
        "total_pairs": total_pairs,
        "unjudged": none_count,
        "no_consensus": no_consensus,
        "low_confidence": low_conf,
        "distribution": dist,
        "distribution_pct": {g: round(c / total * 100, 1) for g, c in dist.items()},
        "negative_rate": round(neg_rate, 3),
        "mean_grade": round(statistics.mean(all_grades), 3),
    }

    print("\n=== Grade Distribution ===")
    for g in (0, 1, 2):
        bar = "#" * int(dist[g] / max(1, total) * 40)
        print(f"  Grade {g}: {dist[g]:>4}  ({dist[g]/total*100:5.1f}%)  {bar}")
    print(f"  Unjudged (None):  {none_count}")
    print(f"    no_consensus:   {no_consensus}")
    print(f"    other failures: {none_count - no_consensus}")
    print(f"  Low confidence:   {low_conf}")
    print(f"  Mean grade:       {stats['mean_grade']}")

    if neg_rate < MIN_NEGATIVE_RATE:
        print(
            f"\n  WARNING: Negative rate is {neg_rate:.1%} -- below the {MIN_NEGATIVE_RATE:.0%} "
            f"threshold. The judge may be too permissive! In the previous run, "
            f"0% negatives led to meaningless metrics."
        )

    if no_consensus > 0:
        nc_rate = no_consensus / total_pairs if total_pairs else 0
        print(
            f"\n  INFO: {no_consensus} pairs ({nc_rate:.1%}) had no consensus after"
            f" escalation. These are unjudged and routed to the audit CSV."
        )

    return stats


def self_consistency_report(results: dict[str, list[dict]]) -> dict:
    """Report per-pair and aggregate self-consistency across k samples."""
    agreements = []
    full_agreement = 0
    total = 0
    unstable: list[dict] = []
    partial_failure: list[dict] = []

    for query, jlist in results.items():
        for j in jlist:
            samps = j.get("samples", [])
            k_req = j.get("k_requested", len(samps))
            n_ok = j.get("n_samples_ok", len(samps))
            if k_req < 2:
                continue
            total += 1
            agr = j.get("agreement", 0.0)
            agreements.append(agr)
            if agr >= 1.0:
                full_agreement += 1
            if agr < 0.67:
                unstable.append({
                    "query": query,
                    "work_id": j.get("work_id", ""),
                    "title": j.get("title", ""),
                    "samples": samps,
                    "agreement": round(agr, 3),
                    "n_samples_ok": n_ok,
                    "k_requested": k_req,
                })
            if n_ok < k_req:
                partial_failure.append({
                    "query": query,
                    "work_id": j.get("work_id", ""),
                    "n_samples_ok": n_ok,
                    "k_requested": k_req,
                    "agreement": round(agr, 3),
                })

    if not agreements:
        print("No multi-sample judgments to assess consistency.")
        return {"total": 0}

    mean_agr = statistics.mean(agreements)
    low_conf_count = sum(
        1 for jlist in results.values() for j in jlist if j.get("low_confidence")
    )
    stats = {
        "total_pairs": total,
        "mean_agreement": round(mean_agr, 3),
        "full_agreement_rate": round(full_agreement / total, 3),
        "unstable_pairs": len(unstable),
        "partial_failure_pairs": len(partial_failure),
        "low_confidence_pairs": low_conf_count,
        "unstable_examples": unstable[:10],
    }

    print("\n=== Self-Consistency ===")
    print(f"  Pairs with k>=2 requested: {total}")
    print(f"  Mean agreement rate      : {mean_agr:.1%}")
    print(f"  Full agreement (100%)    : {full_agreement}/{total} ({full_agreement/total:.1%})")
    print(f"  Unstable (<67% agree)    : {len(unstable)}")
    print(f"  Partial sample failure   : {len(partial_failure)} (n_ok < k_requested)")
    print(f"  Low confidence (total)   : {low_conf_count}")
    if unstable:
        print("  Examples of unstable pairs:")
        for u in unstable[:5]:
            print(
                f"    query={u['query'][:40]}  doc={u['title'][:30]}"
                f"  samples={u['samples']}  ok={u['n_samples_ok']}/{u['k_requested']}"
            )

    return stats


def contradiction_report(results: dict[str, list[dict]]) -> list[dict]:
    """List judgments where the reasoning text contradicts the grade."""
    contras = []
    for query, jlist in results.items():
        for j in jlist:
            if j.get("contradiction"):
                contras.append({
                    "query": query,
                    "work_id": j.get("work_id", ""),
                    "title": j.get("title", ""),
                    "grade": j.get("grade"),
                    "reasoning": j.get("reasoning", ""),
                    "detail": j.get("contradiction_detail", ""),
                })

    print("\n=== Contradiction Detection ===")
    print(f"  Flagged: {len(contras)}")
    for c in contras[:5]:
        print(f"    [{c['grade']}] {c['query'][:40]} -- {c['detail']}")
    return contras


def cohens_kappa(labels_a: list[int], labels_b: list[int]) -> float:
    """Compute Cohen's kappa for two parallel label vectors."""
    assert len(labels_a) == len(labels_b), "label vectors must be same length"
    n = len(labels_a)
    if n == 0:
        return 0.0

    categories = sorted(set(labels_a) | set(labels_b))
    matrix: dict[tuple[int, int], int] = Counter(zip(labels_a, labels_b))
    p_o = sum(matrix.get((c, c), 0) for c in categories) / n

    p_e = 0.0
    for c in categories:
        row_c = sum(matrix.get((c, c2), 0) for c2 in categories)
        col_c = sum(matrix.get((c2, c), 0) for c2 in categories)
        p_e += (row_c / n) * (col_c / n)

    if abs(p_e - 1.0) < 1e-9:
        return 1.0

    return (p_o - p_e) / (1.0 - p_e)


def krippendorffs_alpha(labels_a: list[int], labels_b: list[int]) -> float:
    """Krippendorff's alpha for two coders on an ordinal 0/1/2 scale."""
    n = len(labels_a)
    if n < 2:
        return 0.0

    all_labels = labels_a + labels_b
    total = len(all_labels)
    freq = Counter(all_labels)
    d_o = sum(1 for a, b in zip(labels_a, labels_b) if a != b) / n
    d_e = 1.0 - sum(f * (f - 1) for f in freq.values()) / (total * (total - 1))

    if d_e == 0:
        return 1.0

    return 1.0 - d_o / d_e


def _read_grade(entry: dict) -> Optional[int]:
    """Read a grade from either the `grade` or `relevance` field."""
    g = entry.get("grade")
    if g is None:
        g = entry.get("relevance")
    return g


def compute_agreement(
    results_a: dict[str, list[dict]],
    results_b: dict[str, list[dict]],
) -> dict:
    """Compute inter-rater agreement between two judgment sets."""
    labels_a: list[int] = []
    labels_b: list[int] = []

    for query in results_a:
        if query not in results_b:
            continue
        map_b = {
            j["work_id"]: _read_grade(j)
            for j in results_b[query]
            if _read_grade(j) is not None
        }
        for j in results_a[query]:
            ga = _read_grade(j)
            gb = map_b.get(j.get("work_id"))
            if ga is not None and gb is not None:
                labels_a.append(ga)
                labels_b.append(gb)

    if len(labels_a) < 2:
        print("Too few overlapping judgments for agreement computation.")
        return {"n": len(labels_a), "kappa": None, "alpha": None}

    kappa = cohens_kappa(labels_a, labels_b)
    alpha = krippendorffs_alpha(labels_a, labels_b)
    conf: dict[tuple[int, int], int] = Counter(zip(labels_a, labels_b))

    stats = {
        "n": len(labels_a),
        "kappa": round(kappa, 4),
        "alpha": round(alpha, 4),
        "raw_agreement": round(sum(1 for a, b in zip(labels_a, labels_b) if a == b) / len(labels_a), 4),
        "confusion": {f"{a}v{b}": conf.get((a, b), 0) for a in (0, 1, 2) for b in (0, 1, 2)},
    }

    print("\n=== Inter-Rater Agreement ===")
    print(f"  Overlapping pairs : {stats['n']}")
    print(f"  Raw agreement     : {stats['raw_agreement']:.1%}")
    print(f"  Cohen's kappa     : {stats['kappa']:.4f}")
    print(f"  Krippendorff's alpha: {stats['alpha']:.4f}")
    print("  Confusion matrix (rows=A, cols=B):")
    print("         B=0  B=1  B=2")
    for a in (0, 1, 2):
        row = [conf.get((a, b), 0) for b in (0, 1, 2)]
        print(f"    A={a}  {row[0]:>4} {row[1]:>4} {row[2]:>4}")

    return stats


def gold_doc_agreement(
    results: dict[str, list[dict]],
    queries_path: Path,
) -> dict:
    """Check whether the judge grades gold (seed) documents as relevant."""
    if not queries_path.exists():
        print(f"  Gold queries file not found: {queries_path} -- skipping.")
        return {"available": False}

    try:
        queries = json.loads(queries_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  Could not load gold queries: {exc}")
        return {"available": False}

    if isinstance(queries, dict):
        queries = list(queries.values())

    gold_pairs = 0
    gold_grade_2 = 0
    gold_grade_gt0 = 0
    gold_grade_0 = 0
    misses: list[dict] = []

    for qobj in queries:
        query = qobj.get("query", "")
        gold_ids = set(qobj.get("gold_work_ids", []))
        if not gold_ids or query not in results:
            continue

        judged = {j["work_id"]: j for j in results[query]}
        for gid in gold_ids:
            j = judged.get(gid)
            if j is None:
                continue
            gold_pairs += 1
            g = j.get("grade")
            if g == 2:
                gold_grade_2 += 1
            if g is not None and g > 0:
                gold_grade_gt0 += 1
            if g == 0:
                gold_grade_0 += 1
                misses.append({
                    "query": query,
                    "work_id": gid,
                    "title": j.get("title", ""),
                    "grade": g,
                    "reasoning": j.get("reasoning", ""),
                })

    stats = {
        "available": True,
        "gold_pairs_in_pool": gold_pairs,
        "rate_grade_2": round(gold_grade_2 / gold_pairs, 3) if gold_pairs else 0.0,
        "rate_grade_gt0": round(gold_grade_gt0 / gold_pairs, 3) if gold_pairs else 0.0,
        "rate_grade_0": round(gold_grade_0 / gold_pairs, 3) if gold_pairs else 0.0,
        "misses": misses[:10],
    }

    print("\n=== Gold-Doc Agreement ===")
    if gold_pairs == 0:
        print("  No gold docs found in judged pool -- nothing to evaluate.")
    else:
        print(f"  Gold (query, doc) pairs in pool : {gold_pairs}")
        print(f"  Graded 2 (highly relevant)      : {gold_grade_2}/{gold_pairs} ({stats['rate_grade_2']:.1%})")
        print(f"  Graded >0 (at least partial)    : {gold_grade_gt0}/{gold_pairs} ({stats['rate_grade_gt0']:.1%})")
        print(f"  Graded 0 (missed - FALSE NEG)   : {gold_grade_0}/{gold_pairs} ({stats['rate_grade_0']:.1%})")
        if misses:
            print("  Example misses:")
            for m in misses[:3]:
                print(f"    query={m['query'][:40]}  title={m['title'][:30]}  reasoning={m['reasoning'][:60]}")

    return stats
