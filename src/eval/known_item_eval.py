"""Known-item retrieval evaluation -- CI regression gate.

Searches exact book titles against the API and asserts the correct work_id
appears at rank 1.  Reports accuracy@1, accuracy@5, MRR, and per-mode
breakdown.  Gates on **hybrid mode only** by default (the shipped config);
keyword and vector are reported as diagnostics.

The gate is **baseline-relative**: it fails when hybrid accuracy@1 drops
more than ``--max-drop`` percentage points below a recorded baseline,
rather than asserting an absolute quality level.  Run ``--update-baseline``
after any intentional change to lock in new numbers.

Keyword (TF-IDF) accuracy is reported split by title-word class:
  - *distinctive*: titles containing at least one rare corpus word
    (TF-IDF has strong signal -- near-ceiling expected).
  - *common_words*: titles whose every word is high-frequency in the corpus
    (TF-IDF has little discriminating power -- lower scores expected).

Usage::

    python -m src.eval.known_item_eval                       # gate vs baseline
    python -m src.eval.known_item_eval --update-baseline     # record baseline
    python -m src.eval.known_item_eval --fast --json         # quick CI smoke
    python -m src.eval.known_item_eval --api-url http://localhost:8000

Environment::

    EVAL_API_URL -- override the default API URL.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from src.eval.known_items import load_hard_variants, load_known_items

_DEFAULT_API = (
    "https://booksearch-api.thankfulstone-e6f7cf40.eastus.azurecontainerapps.io"
)
_MODES = ("keyword", "vector", "hybrid")
_REQUEST_TIMEOUT = 20  # seconds
_DEFAULT_MAX_DROP_PP = 5  # percentage points

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_PATH = _REPO_ROOT / "data" / "eval" / "known_item_baseline.json"


# ---------------------------------------------------------------------------
# Baseline management
# ---------------------------------------------------------------------------

def _get_git_sha() -> str | None:
    """Return the current short git SHA, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(_REPO_ROOT),
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def load_baseline(path: Path | str | None = None) -> dict[str, Any] | None:
    """Load a previously recorded baseline, or None if it does not exist."""
    p = Path(path) if path else _BASELINE_PATH
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_baseline(
    results: dict[str, Any],
    api_url: str,
    item_count: int,
    seed: int | None = None,
    max_drop_pp: float = _DEFAULT_MAX_DROP_PP,
    gated_modes: list[str] | None = None,
    path: Path | str | None = None,
) -> Path:
    """Write a baseline file from current evaluation results."""
    p = Path(path) if path else _BASELINE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)

    baseline: dict[str, Any] = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": _get_git_sha(),
        "api_url": api_url,
        "item_count": item_count,
        "seed": seed,
        "max_drop_pp": max_drop_pp,
        "gated_modes": gated_modes or ["hybrid"],
        "modes": {},
    }
    for mode, r in results.items():
        baseline["modes"][mode] = {
            "accuracy_at_1": r["accuracy_at_1"],
            "accuracy_at_5": r["accuracy_at_5"],
            "mrr": r["mrr"],
        }

    with open(p, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)

    return p


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _parse_mode(mode: str) -> tuple[str, bool]:
    """Split a mode label into ``(api_mode, rerank)``.

    ``"hybrid+rerank"`` -> ``("hybrid", True)``. Encoding the reranker arm in
    the mode label keeps a paired comparison (same queries, same index, same
    run) in a single result dict, so the only variable between the two arms is
    the reranker itself.
    """
    if mode.endswith("+rerank"):
        return mode[: -len("+rerank")], True
    return mode, False


def _search(
    api_url: str,
    query: str,
    mode: str,
    top_k: int = 10,
    rerank: bool = False,
) -> dict:
    """Call the search API. Raises on HTTP / network errors."""
    url = (
        f"{api_url}/search"
        f"?q={urllib.parse.quote(query)}"
        f"&mode={mode}&top_k={top_k}&understand=false"
        f"&rerank={'true' if rerank else 'false'}"
    )
    resp = urllib.request.urlopen(url, timeout=_REQUEST_TIMEOUT)
    return json.loads(resp.read())


def _extract_ids(results: list[dict]) -> list[str]:
    """Normalise work_ids to bare OLxxxxxW form."""
    return [r["work_id"].replace("/works/", "") for r in results]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _eval_items(
    api_url: str,
    items: list[dict[str, Any]],
    modes: tuple[str, ...] = _MODES,
    top_k: int = 10,
) -> dict[str, Any]:
    """Run known-item probes and compute per-mode metrics.

    Returns a dict keyed by mode with accuracy@1, accuracy@5, MRR,
    plus a ``details`` list per query.
    """
    results: dict[str, Any] = {}

    for mode in modes:
        api_mode, rerank = _parse_mode(mode)
        reciprocal_ranks: list[float] = []
        hits_at_1 = 0
        hits_at_5 = 0
        details: list[dict[str, Any]] = []

        for item in items:
            query = item["query"]
            target_id = item["work_id"]
            entry: dict[str, Any] = {
                "query": query,
                "work_id": target_id,
                "title": item.get("title", ""),
                "variant_type": item.get("variant_type"),
                "title_word_class": item.get("title_word_class"),
            }

            try:
                data = _search(api_url, query, api_mode, top_k, rerank=rerank)
            except Exception as exc:
                entry.update(rank=None, error=str(exc))
                reciprocal_ranks.append(0.0)
                details.append(entry)
                continue

            ids = _extract_ids(data.get("results", []))

            if target_id in ids:
                rank = ids.index(target_id) + 1
                reciprocal_ranks.append(1.0 / rank)
                if rank == 1:
                    hits_at_1 += 1
                if rank <= 5:
                    hits_at_5 += 1
                entry["rank"] = rank
            else:
                reciprocal_ranks.append(0.0)
                entry["rank"] = None
                top1 = (
                    data["results"][0]["title"]
                    if data.get("results") else "NO RESULTS"
                )
                entry["got_top1"] = top1

            details.append(entry)

        n = len(reciprocal_ranks)
        results[mode] = {
            "accuracy_at_1": round(hits_at_1 / n, 4) if n else 0,
            "accuracy_at_5": round(hits_at_5 / n, 4) if n else 0,
            "mrr": round(sum(reciprocal_ranks) / n, 4) if n else 0,
            "hits_at_1": hits_at_1,
            "hits_at_5": hits_at_5,
            "total": n,
            "errors": sum(1 for d in details if d.get("error")),
            "details": details,
        }

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_keyword_split(results: dict[str, Any]) -> None:
    """Print keyword accuracy broken down by title-word class."""
    kw = results.get("keyword")
    if not kw:
        return

    details = kw.get("details", [])
    distinctive = [d for d in details if d.get("title_word_class") == "distinctive"]
    common = [d for d in details if d.get("title_word_class") == "common_words"]

    if not distinctive and not common:
        return

    print("\n  Keyword accuracy by title-word class:")

    for label, group in [("distinctive", distinctive), ("common_words", common)]:
        if not group:
            continue
        n = len(group)
        at1 = sum(1 for d in group if d.get("rank") == 1)
        at5 = sum(1 for d in group if d.get("rank") is not None and d["rank"] <= 5)
        print(
            f"    {label:>13}: Acc@1={at1/n:.0%} ({at1}/{n})  "
            f"Acc@5={at5/n:.0%} ({at5}/{n})"
        )

    print(
        "    Keyword matches dense retrieval on distinctive titles and loses"
    )
    print(
        "    accuracy on common-word titles where TF-IDF lacks signal."
    )


def _print_standard_report(
    results: dict[str, Any],
    baseline: dict[str, Any] | None,
    max_drop_pp: float,
    gated_modes: list[str],
    verbose: bool = False,
) -> bool:
    """Print the standard eval report.  Returns True if gate passes."""
    all_pass = True

    print("\n" + "=" * 64)
    print("  Known-Item (exact title) Results")
    print("=" * 64)
    print("  Exact-title lookup is near-ceiling for dense (vector) retrieval.")
    print("  Keyword (TF-IDF) degrades on titles composed of common words")
    print("  where term frequencies provide little discriminating signal.")
    print("  The pass/fail gate applies to hybrid (the shipped default) only;")
    print("  keyword and vector are reported as diagnostics.")

    if baseline:
        print(f"  Baseline: {baseline.get('created_at', '?')[:19]}  "
              f"SHA {baseline.get('git_sha', '?')}  "
              f"max-drop {max_drop_pp}pp")

    print("=" * 64)

    for mode, r in results.items():
        is_gated = mode in gated_modes
        status = ""

        if is_gated and baseline:
            bl_acc1 = baseline.get("modes", {}).get(mode, {}).get(
                "accuracy_at_1"
            )
            if bl_acc1 is not None:
                delta = r["accuracy_at_1"] - bl_acc1
                delta_pp = delta * 100
                passed = delta_pp >= -max_drop_pp
                delta_str = (
                    f"delta {delta_pp:+.1f}pp vs baseline {bl_acc1:.1%}"
                )
                status = f"  [{'PASS' if passed else 'FAIL'} {delta_str}]"
                if not passed:
                    all_pass = False
            else:
                status = "  [no baseline for mode]"
        elif is_gated and not baseline:
            status = "  [no baseline -- skipping gate]"
        else:
            status = "  (diagnostic)"

        print(
            f"  {mode:>8}: "
            f"Acc@1={r['accuracy_at_1']:.1%}  "
            f"Acc@5={r['accuracy_at_5']:.1%}  "
            f"MRR={r['mrr']:.4f}  "
            f"({r['hits_at_1']}/{r['total']} at rank 1)"
            f"{status}"
        )

        # A failed request is scored as a miss, so a client-side fault (bad
        # signature, wrong URL, timeout) is otherwise indistinguishable from
        # genuinely poor retrieval. Surface it unconditionally, not just -v.
        n_err = r.get("errors", 0)
        if n_err:
            print(
                f"           !! {n_err}/{r['total']} queries ERRORED and were "
                f"scored as misses -- metrics for {mode} are not "
                f"trustworthy (re-run with -v for details)"
            )

    # Keyword distinctive vs common split
    _print_keyword_split(results)

    if verbose:
        for mode, r in results.items():
            misses = [d for d in r["details"] if d.get("rank") != 1]
            if misses:
                print(f"\n  Misses [{mode}]:")
                for d in misses:
                    cls = f" [{d['title_word_class']}]" if d.get("title_word_class") else ""
                    if d.get("error"):
                        print(f"    ERROR  {d['query']!r}: {d['error']}")
                    elif d["rank"] is None:
                        print(
                            f"    MISS   {d['query']!r}{cls}"
                            f" -> got {d.get('got_top1', '?')!r}"
                        )
                    else:
                        print(f"    rank {d['rank']:>2}  {d['query']!r}{cls}")

    return all_pass


def _print_diagnostic_report(
    results: dict[str, Any],
    verbose: bool = False,
) -> None:
    """Print hard-variant diagnostics (never affects the gate)."""
    print("\n" + "=" * 64)
    print("  Hard Variants (diagnostic -- not part of pass/fail gate)")
    print("=" * 64)

    for mode, r in results.items():
        print(
            f"  {mode:>8}: "
            f"Acc@1={r['accuracy_at_1']:.1%}  "
            f"Acc@5={r['accuracy_at_5']:.1%}  "
            f"MRR={r['mrr']:.4f}  "
            f"({r['hits_at_1']}/{r['total']} at rank 1)"
        )

    if verbose:
        for mode, r in results.items():
            misses = [d for d in r["details"] if d.get("rank") != 1]
            if misses:
                print(f"\n  Misses [{mode}]:")
                for d in misses:
                    vt = f" ({d['variant_type']})" if d.get("variant_type") else ""
                    if d.get("error"):
                        print(f"    ERROR  {d['query']!r}: {d['error']}")
                    elif d["rank"] is None:
                        print(
                            f"    MISS   {d['query']!r}{vt}"
                            f" -> got {d.get('got_top1', '?')!r}"
                        )
                    else:
                        print(f"    rank {d['rank']:>2}  {d['query']!r}{vt}")


def _build_json_output(
    standard_results: dict[str, Any],
    hard_results: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
    max_drop_pp: float,
    gated_modes: list[str],
    passed: bool,
    elapsed: float,
) -> dict[str, Any]:
    """Build machine-readable JSON output."""

    def _summarise(res: dict[str, Any]) -> dict[str, Any]:
        return {
            mode: {k: v for k, v in r.items() if k != "details"}
            for mode, r in res.items()
        }

    out: dict[str, Any] = {
        "evaluation": "known_item",
        "passed": passed,
        "gated_modes": gated_modes,
        "max_drop_pp": max_drop_pp,
        "elapsed_seconds": round(elapsed, 2),
        "standard": _summarise(standard_results),
    }

    if baseline:
        deltas: dict[str, Any] = {}
        for mode in gated_modes:
            bl = baseline.get("modes", {}).get(mode, {})
            cur = standard_results.get(mode, {})
            if bl and cur:
                deltas[mode] = {
                    "baseline_acc1": bl.get("accuracy_at_1"),
                    "current_acc1": cur.get("accuracy_at_1"),
                        "delta_pp": round(
                            (cur["accuracy_at_1"] - bl["accuracy_at_1"]) * 100, 2
                    ),
                }
        out["baseline_comparison"] = deltas
        out["baseline_sha"] = baseline.get("git_sha")

    if hard_results:
        out["hard_variants"] = _summarise(hard_results)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Known-item retrieval eval -- CI regression gate.  "
            "Queries exact book titles and asserts the correct work_id "
            "appears at rank 1.  Gates on hybrid mode by default; "
            "keyword and vector are diagnostics."
        ),
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("EVAL_API_URL", _DEFAULT_API),
        help="API base URL (or set EVAL_API_URL env var)",
    )
    parser.add_argument(
        "--max-drop", type=float, default=None,
        help=(
            "Max allowed accuracy@1 drop in percentage points vs baseline "
            "(default: from baseline file, or 5pp)"
        ),
    )
    parser.add_argument(
        "--gated-modes", default=None,
        help="Comma-separated modes to gate on (default: from baseline, or 'hybrid')",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of results to retrieve per query (default: 10)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Use a small sample (first 10 items) for quick CI smoke tests",
    )
    parser.add_argument(
        "--update-baseline", action="store_true",
        help="Record current results as the new baseline",
    )
    parser.add_argument(
        "--no-hard-variants", action="store_true",
        help="Skip hard-variant diagnostics",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output machine-readable JSON to stdout",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-query miss details",
    )
    parser.add_argument(
        "--items-file", type=Path, default=None,
        help="Path to known_items.json (auto-detected from repo root)",
    )
    parser.add_argument(
        "--variants-file", type=Path, default=None,
        help="Path to known_item_hard_variants.json",
    )
    parser.add_argument(
        "--baseline-file", type=Path, default=None,
        help="Path to known_item_baseline.json",
    )
    parser.add_argument(
        "--modes", type=str, default=None,
        help=(
            "Comma-separated modes to evaluate (default: "
            f"{','.join(_MODES)}). Append '+rerank' to any mode to run it "
            "through the cross-encoder, e.g. 'hybrid,hybrid+rerank' for a "
            "paired reranker comparison."
        ),
    )
    args = parser.parse_args()

    if args.modes:
        eval_modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
    else:
        eval_modes = _MODES

    # Refuse to baseline on --fast (too much sampling noise)
    if args.update_baseline and args.fast:
        print(
            "ERROR: --update-baseline requires the full item set. "
            "Remove --fast.",
            file=sys.stderr,
        )
        return 1

    # Load items
    try:
        items = load_known_items(args.items_file)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.fast:
        items = items[:10]

    # Load baseline (if any)
    baseline = load_baseline(args.baseline_file)

    # Baselining a narrower mode set than the one already recorded would
    # silently shrink the gate: save_baseline writes exactly the modes it is
    # handed, so a dropped mode stops being checked without anything failing.
    # Adding modes is fine and is how the reranker arm was introduced.
    if args.update_baseline and baseline:
        dropped = set(baseline.get("modes", {})) - set(eval_modes)
        if dropped:
            print(
                f"ERROR: --update-baseline would drop {sorted(dropped)} from "
                f"the baseline, which silently removes them from the gate. "
                f"Include them in --modes, or delete the baseline file to "
                f"start over deliberately.",
                file=sys.stderr,
            )
            return 1

    # Resolve gated_modes and max_drop from baseline or CLI
    if args.gated_modes:
        gated_modes = [m.strip() for m in args.gated_modes.split(",")]
    elif baseline:
        gated_modes = baseline.get("gated_modes", ["hybrid"])
    else:
        gated_modes = ["hybrid"]

    # A gated mode that is not being evaluated can never fail, so the gate
    # would silently pass. Catch the mismatch instead of trusting it.
    ungated = [m for m in gated_modes if m not in eval_modes]
    if ungated:
        print(
            f"ERROR: gated modes {ungated} are not in the evaluated set "
            f"{list(eval_modes)}; the gate would pass without checking them.",
            file=sys.stderr,
        )
        return 1

    if args.max_drop is not None:
        max_drop_pp = args.max_drop
    elif baseline:
        max_drop_pp = baseline.get("max_drop_pp", _DEFAULT_MAX_DROP_PP)
    else:
        max_drop_pp = _DEFAULT_MAX_DROP_PP

    # Verify API is reachable
    try:
        health_url = f"{args.api_url}/health"
        resp = urllib.request.urlopen(health_url, timeout=10)
        health = json.loads(resp.read())
        if health.get("status") not in ("ok", "healthy"):
            print(
                f"WARNING: API health check returned: {health}",
                file=sys.stderr,
            )
    except Exception as exc:
        print(
            f"ERROR: API unreachable at {args.api_url}: {exc}",
            file=sys.stderr,
        )
        return 1

    sample_note = " (--fast)" if args.fast else ""
    print(f"Known-item eval: {len(items)} queries{sample_note}, k={args.top_k}")
    print(f"API: {args.api_url}")
    if baseline:
        print(f"Baseline: {baseline['created_at'][:19]}  "
              f"SHA {baseline.get('git_sha', '?')}")
        print(f"Gate: {', '.join(gated_modes)} acc@1 must not drop "
              f">{max_drop_pp}pp below baseline")
    else:
        print("No baseline recorded -- metrics only, gate skipped")
        print("Run with --update-baseline to establish one")

    t0 = time.time()

    # Standard eval
    standard_results = _eval_items(
        args.api_url, items, modes=eval_modes, top_k=args.top_k
    )
    passed = _print_standard_report(
        standard_results, baseline, max_drop_pp, gated_modes,
        verbose=args.verbose,
    )

    # If no baseline, don't fail the gate -- there's nothing to regress from
    if not baseline:
        passed = True

    # Hard variants (diagnostic only)
    hard_results = None
    if not args.no_hard_variants:
        try:
            hard_items = load_hard_variants(args.variants_file)
        except FileNotFoundError:
            hard_items = []

        if hard_items:
            hard_results = _eval_items(
                args.api_url, hard_items, modes=eval_modes, top_k=args.top_k
            )
            _print_diagnostic_report(hard_results, verbose=args.verbose)

    elapsed = time.time() - t0

    # Update baseline if requested
    if args.update_baseline:
        bl_path = save_baseline(
            standard_results,
            api_url=args.api_url,
            item_count=len(items),
            max_drop_pp=max_drop_pp,
            gated_modes=gated_modes,
            path=args.baseline_file,
        )
        print(f"\nBaseline written to {bl_path}")

    # Summary
    print(f"\nCompleted in {elapsed:.1f}s")
    if passed:
        print("PASSED")
    else:
        print("FAILED -- regression detected in gated mode(s)")

    # JSON output
    if args.json:
        out = _build_json_output(
            standard_results, hard_results,
            baseline, max_drop_pp, gated_modes,
            passed, elapsed,
        )
        print("\n--- JSON ---")
        print(json.dumps(out, indent=2))

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
