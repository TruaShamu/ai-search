"""Known-item retrieval eval — CI regression gate.

Tests that well-known books appear in top-k for their obvious queries.
Exits non-zero if MRR drops below threshold (default 0.7).

Usage:
    python -m src.eval.known_item_eval [--api-url URL] [--threshold 0.7]
"""

import argparse
import json
import sys
import urllib.parse
import urllib.request

from src.eval.known_items import KNOWN_ITEMS


def run_known_item_eval(api_url: str, top_k: int = 10, threshold: float = 0.7):
    modes = ["keyword", "vector", "hybrid"]
    results = {}

    for mode in modes:
        reciprocal_ranks = []
        failures = []

        for item in KNOWN_ITEMS:
            query = item["query"]
            target_id = item["work_id"]
            url = (
                f"{api_url}/search"
                f"?q={urllib.parse.quote(query)}"
                f"&mode={mode}&top_k={top_k}&understand=false"
            )

            try:
                resp = urllib.request.urlopen(url, timeout=15)
                data = json.loads(resp.read())
            except Exception as e:
                print(f"  ERROR [{mode}] \"{query}\": {e}")
                reciprocal_ranks.append(0.0)
                failures.append({"query": query, "error": str(e)})
                continue

            retrieved_ids = [
                r["work_id"].replace("/works/", "")
                for r in data["results"]
            ]

            if target_id in retrieved_ids:
                rank = retrieved_ids.index(target_id) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
                failures.append({
                    "query": query,
                    "expected": f"{item['title']} ({target_id})",
                    "got_top1": data["results"][0]["title"] if data["results"] else "NO RESULTS",
                })

        n = len(reciprocal_ranks)
        mrr = sum(reciprocal_ranks) / n if n else 0
        hits = sum(1 for rr in reciprocal_ranks if rr > 0)

        results[mode] = {
            "mrr": round(mrr, 4),
            "hit_rate": round(hits / n, 4) if n else 0,
            "hits": hits,
            "total": n,
            "failures": failures,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Known-item retrieval eval")
    parser.add_argument(
        "--api-url",
        default="https://booksearch-api.thankfulstone-e6f7cf40.eastus.azurecontainerapps.io",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Known-item eval: {len(KNOWN_ITEMS)} queries, k={args.top_k}")
    print(f"API: {args.api_url}")
    print(f"Threshold: MRR >= {args.threshold}\n")

    results = run_known_item_eval(args.api_url, args.top_k, args.threshold)

    # Report
    all_pass = True
    for mode, r in results.items():
        status = "PASS" if r["mrr"] >= args.threshold else "FAIL"
        if r["mrr"] < args.threshold:
            all_pass = False
        print(f"  {mode:>8}: MRR={r['mrr']:.4f}  Hit@{args.top_k}={r['hit_rate']:.0%} ({r['hits']}/{r['total']})  [{status}]")

    if args.verbose:
        print("\nFailures:")
        for mode, r in results.items():
            for f in r["failures"]:
                if "error" in f:
                    print(f"  [{mode}] \"{f['query']}\": {f['error']}")
                else:
                    print(f"  [{mode}] \"{f['query']}\": expected \"{f['expected']}\", got \"{f['got_top1']}\"")

    print()
    if all_pass:
        print("PASSED — all modes above threshold")
        return 0
    else:
        print("FAILED — one or more modes below threshold")
        return 1


if __name__ == "__main__":
    sys.exit(main())
