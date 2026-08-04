"""Threshold-sensitivity analysis for the graded relevance eval.

The headline MRR in the README counts a document as relevant when its graded
label is 1 or 2 ("partial" or "fully" relevant). That is a lenient threshold,
and on a pool where 24% of documents clear it, even a random ranking scores
well. This script recomputes the same metrics under a strict threshold that
only counts grade-2 documents, and reports a random-ranking baseline for both,
so the headline number can be read against something.

It also writes the raw per-mode ranked lists to ``rankings.json``. Those
lists are the only part of the eval that cannot be recovered offline -- the
pooled candidates and the judgments are stored, but the ranked order each mode
produced is not. Capturing them keeps every number below reproducible after the
underlying corpus changes.

Eval artifacts are corpus-coupled, so every path here is a flag rather than a
constant, and any path carrying an archived marker is refused as a write target.
The v1 OpenLibrary rankings and pooled candidates were deleted once the corpus
migrated, so the v1 threshold run is no longer re-derivable. What still carries a
``.v1-openlibrary`` suffix is the hand-labelled judge validation, which this
script must never overwrite -- those labels cannot be regenerated.

Usage:
    python scripts/threshold_sensitivity.py            # fetch + analyse
    python scripts/threshold_sensitivity.py --offline  # re-analyse saved ranks
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
V2 = REPO_ROOT / "data" / "eval" / "v2"
JUDGMENTS_FILE = V2 / "judgments.json"
POOLED_FILE = V2 / "pooled.json"
RANKINGS_FILE = V2 / "rankings.json"
OUT_FILE = V2 / "threshold_sensitivity.json"

DEFAULT_API = "https://booksearch-api.thankfulstone-e6f7cf40.eastus.azurecontainerapps.io"
API = os.environ.get("EVAL_API_URL", DEFAULT_API)

MODES = ["keyword", "vector", "hybrid", "hybrid+rerank"]
K = 10

# Markers identifying a frozen, superseded-corpus artifact. `.v1-openlibrary` is
# the suffix used when the corpus migrated; the `_v1.`/`.v1.` forms are kept so an
# artifact following the older naming cannot be waved through as a write target.
# The surviving v1 files are hand-labelled and cannot be regenerated, so this
# guard is the only thing standing between a stray --out and permanent loss.
_ARCHIVED_MARKERS = (".v1-openlibrary", "_v1.", ".v1.")


def _is_archived(path: Path) -> bool:
    return any(marker in path.name for marker in _ARCHIVED_MARKERS)


def _url(query: str, mode: str) -> str:
    q = urllib.parse.quote(query)
    if mode == "hybrid+rerank":
        return f"{API}/search?q={q}&mode=hybrid&top_k={K}&understand=false&rerank=true"
    return f"{API}/search?q={q}&mode={mode}&top_k={K}&understand=false"


def fetch_ranked_ids(query: str, mode: str, retries: int = 4) -> list[str] | None:
    """Fetch one ranked list, retrying transient network failures.

    A single dropped socket previously killed an entire eval run, so failures
    here retry with backoff and only give up after several attempts.
    """
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(_url(query, mode), timeout=90) as resp:
                data = json.loads(resp.read())
            return [
                r["work_id"].replace("/works/", "")
                for r in data.get("results", [])
                if r.get("work_id")
            ]
        except Exception as exc:  # noqa: BLE001 - report and retry any transport error
            if attempt == retries - 1:
                print(f"    FAILED [{mode}] {query[:40]!r}: {exc}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


def load_relevance_maps(judgments_file: Path = JUDGMENTS_FILE) -> dict[str, dict[str, int]]:
    judgments = json.loads(judgments_file.read_text(encoding="utf-8"))
    maps: dict[str, dict[str, int]] = {}
    for query, jlist in judgments.items():
        rmap = {j["work_id"]: j["relevance"] for j in jlist if j.get("relevance") is not None}
        if rmap:
            maps[query] = rmap
    return maps


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(ranked[:K], start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def recall(ranked: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in ranked[:K] if d in relevant) / len(relevant)


def random_baseline(pool_size: int, n_relevant: int, trials: int, rng: random.Random) -> float:
    """Expected MRR@10 when the pool is shuffled instead of ranked.

    The relevant set is a fixed identity; only the ranking order is shuffled.
    (Drawing the relevant set from the shuffled list itself would make position
    one relevant by construction and return a baseline of exactly 1.0.)
    """
    if n_relevant <= 0 or pool_size <= 0:
        return 0.0
    n_relevant = min(n_relevant, pool_size)
    relevant = set(range(n_relevant))
    order = list(range(pool_size))
    total = 0.0
    for _ in range(trials):
        rng.shuffle(order)
        for i, doc in enumerate(order[:K], start=1):
            if doc in relevant:
                total += 1.0 / i
                break
    return total / trials


def bootstrap_ci(values: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true",
                    help="reuse saved rankings instead of querying the API")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--trials", type=int, default=2000,
                    help="Monte Carlo trials for the random baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judgments", type=Path, default=JUDGMENTS_FILE,
                    help="graded relevance labels to score against")
    ap.add_argument("--pooled", type=Path, default=POOLED_FILE,
                    help="pooled candidates, used for the random baseline's pool size")
    ap.add_argument("--rankings", type=Path, default=RANKINGS_FILE,
                    help="where raw per-mode ranked lists are read/written")
    ap.add_argument("--out", type=Path, default=OUT_FILE,
                    help="report destination")
    args = ap.parse_args()

    rankings_file, out_file = args.rankings, args.out

    # A live fetch writes the ranked lists, so refuse to aim that write at an
    # archived run. Corpus-coupled artifacts are only useful while the corpus
    # that produced them is the one being queried.
    if not args.offline and _is_archived(rankings_file):
        print(f"ERROR: refusing to overwrite archived rankings {rankings_file.name} "
              f"with a live fetch. Pass --offline to re-analyse it instead.")
        return 1
    if not args.offline and _is_archived(out_file):
        print(f"ERROR: refusing to write a live-fetch report to archived {out_file.name}.")
        return 1

    relevance_maps = load_relevance_maps(args.judgments)
    # Match the published eval: a query is evaluated only if its pool holds at
    # least one document that clears the LENIENT threshold. Keeping this set
    # fixed across thresholds is what makes the two tables comparable.
    queries = sorted(q for q, m in relevance_maps.items() if any(r > 0 for r in m.values()))
    print(f"Queries evaluated: {len(queries)} (of {len(relevance_maps)} judged)")

    if args.offline:
        if not rankings_file.exists():
            print(f"ERROR: {rankings_file} not found; run without --offline first.")
            return 1
        rankings = json.loads(rankings_file.read_text(encoding="utf-8"))
        print(f"Loaded saved rankings from {rankings_file.name}")
    else:
        rankings = {}
        fetch_failures = []
        for mode in MODES:
            print(f"  Fetching {mode} ...")
            rankings[mode] = {}
            for i, query in enumerate(queries, start=1):
                ids = fetch_ranked_ids(query, mode)
                if ids is not None:
                    rankings[mode][query] = ids
                else:
                    fetch_failures.append((mode, query))
                if i % 25 == 0:
                    print(f"    {i}/{len(queries)}")
        # Abort before writing. A partial fetch silently changes the denominator
        # every threshold in this report is computed over, and the output still
        # looks like a finished run.
        if fetch_failures:
            modes_hit = sorted({m for m, _ in fetch_failures})
            print(
                f"\nABORTING: {len(fetch_failures)} request(s) failed after retries "
                f"(modes: {', '.join(modes_hit)}).\nNothing was written -- re-run once "
                f"the API is reachable."
            )
            return 1
        rankings_file.parent.mkdir(parents=True, exist_ok=True)
        rankings_file.write_text(
            json.dumps(rankings, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved raw ranked lists -> {rankings_file}")

    # Only keep queries every mode answered, so all columns share a denominator.
    common = set(queries)
    for mode in MODES:
        common &= set(rankings.get(mode, {}))
    common = sorted(common)
    missing = len(queries) - len(common)
    if missing:
        print(
            f"ERROR: {missing} of {len(queries)} queries are missing from at least one "
            f"mode.\nRefusing to publish a report whose columns cover different queries."
        )
        return 1
    print(f"Queries common to all modes: {len(common)}")

    pooled = json.loads(args.pooled.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)

    report: dict = {
        "n_queries": len(common),
        "k": K,
        "note": (
            "Lenient counts grade>=1 as relevant (this is what the headline "
            "README table uses). Strict counts only grade==2. Both are computed "
            "over the same query set so the columns are comparable."
        ),
        "thresholds": {},
    }

    for label, threshold in (("lenient_grade_ge_1", 1), ("strict_grade_eq_2", 2)):
        rel_sets = {
            q: {d for d, r in relevance_maps[q].items() if r >= threshold} for q in common
        }
        n_with_any = sum(1 for q in common if rel_sets[q])
        density = sum(len(rel_sets[q]) for q in common) / max(1, len(common))

        base_vals = [
            random_baseline(len(pooled.get(q, [])) or 50, len(rel_sets[q]), args.trials, rng)
            for q in common
        ]
        base_mrr = sum(base_vals) / max(1, len(base_vals))

        entry = {
            "threshold": f"grade >= {threshold}" if threshold == 1 else "grade == 2",
            "queries_with_at_least_one_relevant": n_with_any,
            "mean_relevant_docs_per_pool": round(density, 2),
            "random_baseline_mrr_at_10": round(base_mrr, 4),
            "modes": {},
        }

        print(f"\n=== {label} ===")
        print(f"  queries with >=1 qualifying doc : {n_with_any}/{len(common)}")
        print(f"  mean qualifying docs per pool   : {density:.2f}")
        print(f"  random-ranking MRR@10 baseline  : {base_mrr:.4f}")

        for mode in MODES:
            mrrs = [reciprocal_rank(rankings[mode][q], rel_sets[q]) for q in common]
            recs = [recall(rankings[mode][q], rel_sets[q]) for q in common]
            mrr = sum(mrrs) / len(mrrs)
            rec = sum(recs) / len(recs)
            lo, hi = bootstrap_ci(mrrs, args.bootstrap, args.seed)
            rlo, rhi = bootstrap_ci(recs, args.bootstrap, args.seed + 1)
            entry["modes"][mode] = {
                "mrr_at_10": round(mrr, 4),
                "mrr_ci_95": [round(lo, 4), round(hi, 4)],
                "recall_at_10": round(rec, 4),
                "recall_ci_95": [round(rlo, 4), round(rhi, 4)],
                "lift_over_random": round(mrr - base_mrr, 4),
            }
            print(f"  {mode:16s} MRR={mrr:.4f} [{lo:.4f}, {hi:.4f}]  "
                  f"Recall={rec:.4f}  lift={mrr - base_mrr:+.4f}")

        report["thresholds"][label] = entry

    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved -> {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
