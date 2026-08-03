"""Full eval redesign: generate queries, pool from Qdrant, judge, compute CIs.

Usage:
    python scripts/eval_redesign.py --step generate   # Step 1: generate 100 queries
    python scripts/eval_redesign.py --step pool       # Step 2: pool top-50 from all modes (incl. rerank)
    python scripts/eval_redesign.py --step judge      # Step 3: LLM judge (keep zeros)
    python scripts/eval_redesign.py --step eval       # Step 4: compute metrics + paired bootstrap CIs
    python scripts/eval_redesign.py --step all        # Run all steps
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

DEFAULT_API = "https://booksearch-api.thankfulstone-e6f7cf40.eastus.azurecontainerapps.io"
API = os.environ.get("EVAL_API_URL", DEFAULT_API)
DATA_DIR = Path("data/eval/v2")
DATA_DIR.mkdir(parents=True, exist_ok=True)

QUERIES_FILE = DATA_DIR / "queries_grounded.json"
POOLED_FILE = DATA_DIR / "pooled.json"
JUDGMENTS_FILE = DATA_DIR / "judgments.json"
RESULTS_FILE = DATA_DIR / "results.json"

MAX_RETRIES = 4
RETRY_BACKOFF = [2, 5, 15, 30]

# Rate limiting is handled separately from network errors, and deliberately does
# NOT honour Retry-After.
#
# The history matters, because the obvious diagnosis was wrong. Judging ran at
# ~0.5 successful req/s no matter what -- 1, 4 and 8 workers all landed on the
# same number, which looks exactly like a client-side concurrency bug. It was
# not. The deployment itself was provisioned at capacity 30, and Azure turns
# that into a hard server-side quota, visible in ARM as:
#     rateLimits: [{key: "request", count: 30, renewalPeriod: 60}, ...]
# 30 requests per minute is 0.5/s. Every client-side knob was being measured
# against a ceiling none of them could move, which is why they all agreed.
#
# Two independent things were wrong, and only one is fixed here:
#   1. Capacity (fixed in Azure, not in code): raised 30 -> 500, so 500 RPM.
#      On GlobalStandard this is a rate cap, not a reservation -- billing stays
#      per-token, so the old value was pure self-inflicted throttling.
#   2. Backoff (fixed here): Azure answers a throttled call with
#      Retry-After: 60. Sleeping that parks a worker for a full minute over a
#      quota that refills continuously, so a single 429 cost ~60x what the
#      request itself did. Short jittered backoff re-probes as capacity returns;
#      the jitter stops workers resynchronising into a thundering herd.
#
# A 429 is cheap (~0.1s, no tokens billed), so attempts are generous.
RATE_LIMIT_MAX_ATTEMPTS = 12
RATE_LIMIT_BASE_WAIT = 0.6
RATE_LIMIT_MAX_WAIT = 8.0

# Concurrent judge calls. With the deployment at 500 RPM (see the rate-limit
# note above), throughput is bounded by whichever is smaller: the server quota,
# or workers / per-call latency. At ~1s per call, 8 workers offer ~480 RPM,
# which sits just under the quota and leaves the jittered backoff to absorb the
# occasional overshoot. Raising this further only converts successes into 429s.
# Check the deployment's actual `rateLimits` before tuning it: this number is
# only meaningful relative to the provisioned capacity.
JUDGE_WORKERS = 8

# Rerank candidate-pool multiplier. Imported from the reranker package so the
# oracle ceiling is measured over the exact candidate depth production uses —
# a hardcoded copy here silently drifts and inflates the reported headroom.
try:
    from src.reranker.config import RERANK_DEPTH_MULTIPLIER
except ImportError:  # running standalone without the package on sys.path
    RERANK_DEPTH_MULTIPLIER = 2.5


# ─── Step 1: Generate corpus-grounded queries ───────────────────────────────

def generate_queries():
    """Delegate to the corpus-grounded generator in src/eval/query_gen.py.

    This step used to ask the LLM to invent queries with no view of the corpus.
    That produced canonical titles ("1984", "The Great Gatsby", "Harry Potter")
    which are largely absent from this 26.5k OpenLibrary index, and carried no
    gold_work_ids -- so relevance rested entirely on the judge. Grounded
    generation instead seeds every query from a real indexed book, attaches
    gold_work_ids, and rejects queries that copy >=4 verbatim tokens from the
    seed description (which would hand lexical matching an artificial win).
    """
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable, "-m", "src.eval.query_gen",
        "--n", "100",
        "--max-verbatim-ngram", "4",
        "--out", str(QUERIES_FILE),
    ]
    print("Delegating to the corpus-grounded generator:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo_root))
    return json.loads(QUERIES_FILE.read_text(encoding="utf-8"))


# ─── Step 2: Pool from all modes (including rerank at depth 50) ──────────────

def pool_results(queries: list[dict]):
    """Pool top-20 from keyword/vector/hybrid + top-50 from hybrid (rerank candidate pool)."""
    pooled = {}

    print(f"Pooling results for {len(queries)} queries...")
    print("  Modes: keyword(20), vector(20), hybrid(20), hybrid(50 for rerank pool)")

    for i, q_obj in enumerate(queries):
        query = q_obj["query"]
        seen_ids = set()
        docs = []

        # Pool top-20 from each base mode
        for mode in ["keyword", "vector", "hybrid"]:
            url = f"{API}/search?q={urllib.parse.quote(query)}&mode={mode}&top_k=20&understand=false"
            try:
                resp = urllib.request.urlopen(url, timeout=20)
                data = json.loads(resp.read())
            except Exception as e:
                print(f"  ERROR [{mode}] \"{query}\": {e}")
                continue

            for r in data.get("results", []):
                work_id = r.get("work_id")
                if not work_id:
                    continue
                doc_id = work_id.replace("/works/", "")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    docs.append({
                        "work_id": doc_id,
                        "title": r.get("title", ""),
                        "authors": r.get("authors", ""),
                        "subjects": r.get("subjects", [])[:8],
                        "description": (r.get("description") or "")[:500],
                    })

        # Pool hybrid at depth 50 (reranker candidate pool)
        url = f"{API}/search?q={urllib.parse.quote(query)}&mode=hybrid&top_k=50&understand=false"
        try:
            resp = urllib.request.urlopen(url, timeout=20)
            data = json.loads(resp.read())
            for r in data.get("results", []):
                work_id = r.get("work_id")
                if not work_id:
                    continue
                doc_id = work_id.replace("/works/", "")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    docs.append({
                        "work_id": doc_id,
                        "title": r.get("title", ""),
                        "authors": r.get("authors", ""),
                        "subjects": r.get("subjects", [])[:8],
                        "description": (r.get("description") or "")[:500],
                    })
        except Exception as e:
            print(f"  ERROR [hybrid@50] \"{query}\": {e}")

        pooled[query] = docs

        if (i + 1) % 10 == 0:
            print(f"  Pooled {i+1}/{len(queries)} ({len(docs)} docs for last query)")

    # Stats
    total_docs = sum(len(d) for d in pooled.values())
    mean_docs = total_docs / len(pooled) if pooled else 0
    print(f"\nPooling complete: {total_docs} total docs, {mean_docs:.1f} mean/query")

    POOLED_FILE.write_text(json.dumps(pooled, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved to {POOLED_FILE}")
    return pooled


# ─── Step 3: LLM Judge (keep zeros, retry on error, never coerce) ────────────

JUDGE_PROMPT = """You are evaluating search result relevance for a book search engine.

Query: "{query}"
Document:
  Title: "{title}"
  Author(s): {authors}
  Subjects: {subjects}
  {description}

Rate the relevance of this document to the query on a scale of 0-2:
- 0: Not relevant (wrong topic, wrong book, no meaningful connection)
- 1: Partially relevant (related topic but not what user likely wants)
- 2: Highly relevant (exactly or very close to what user is searching for)

Respond with ONLY a single digit: 0, 1, or 2."""


def _is_content_filter(resp) -> bool:
    """True when a 4xx is Azure's content filter rejecting this specific input.

    A filtered document is a property of one book blurb, not of the run. The
    distinction matters because the two 400s need opposite handling: a bad
    deployment name or malformed payload repeats on every pair and should abort
    immediately, while a filtered blurb should cost exactly one unjudged pair.
    """
    try:
        err = resp.json().get("error") or {}
    except Exception:
        return False
    codes = " ".join(str(x) for x in (
        err.get("code", ""),
        (err.get("innererror") or {}).get("code", ""),
        err.get("message", ""),
    )).lower()
    return "content_filter" in codes or "responsibleaipolicyviolation" in codes


def _call_judge(client, url, headers, prompt) -> int | None:
    """Call LLM judge with retries. Returns relevance (0-2) or None on failure.

    Rate-limit retries and transient-failure retries are tracked separately: a
    429 says nothing about whether the request was malformed, so it must not
    consume the budget reserved for genuine network faults.
    """
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_completion_tokens": 5,
    }

    net_attempts = 0
    rate_attempts = 0

    while True:
        try:
            resp = client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content and content[0].isdigit():
                return min(2, max(0, int(content[0])))
            return None  # Non-digit response is unjudged, not "irrelevant"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                rate_attempts += 1
                if rate_attempts >= RATE_LIMIT_MAX_ATTEMPTS:
                    return None
                wait = min(RATE_LIMIT_MAX_WAIT,
                           RATE_LIMIT_BASE_WAIT * (1.7 ** (rate_attempts - 1)))
                time.sleep(wait * random.uniform(0.7, 1.3))
                continue
            # Fail fast on auth/permission errors (4xx other than rate-limit),
            # but never let one filtered document abort a 5,000-pair run.
            if 400 <= e.response.status_code < 500:
                if _is_content_filter(e.response):
                    return None
                print(f"  FATAL {e.response.status_code}: {e.response.text[:400]}")
                raise
            net_attempts += 1
            if net_attempts >= MAX_RETRIES:
                return None
            time.sleep(RETRY_BACKOFF[net_attempts - 1])
        except (httpx.TimeoutException, httpx.ConnectError):
            net_attempts += 1
            if net_attempts >= MAX_RETRIES:
                return None  # Exhausted retries — mark as unjudged
            time.sleep(RETRY_BACKOFF[net_attempts - 1])


def judge_documents(queries: list[dict], pooled: dict):
    """Judge each (query, doc) pair. KEEPS zeros. Uses None for failures (excluded from metrics)."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    key = os.getenv("AZURE_OPENAI_KEY", "")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-54-nano")

    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-12-01-preview"
    headers = {"api-key": key, "Content-Type": "application/json"}
    client = httpx.Client(timeout=30)

    # Load existing judgments if resuming
    judgments = {}
    if JUDGMENTS_FILE.exists():
        judgments = json.loads(JUDGMENTS_FILE.read_text(encoding="utf-8"))
        print(f"Resuming: {len(judgments)} queries already judged")

    total_pairs = sum(len(pooled.get(q["query"], [])) for q in queries)
    done_pairs = sum(
        sum(1 for j in judgments.get(q["query"], []) if j["relevance"] is not None)
        for q in queries if q["query"] in judgments
    )
    print(f"Judging {total_pairs - done_pairs} remaining pairs ({total_pairs} total)...")

    errors = 0
    for i, q_obj in enumerate(queries):
        query = q_obj["query"]

        if query in judgments:
            # Re-judge only pairs that previously failed (relevance is None)
            existing = judgments[query]
            failed_indices = [
                idx for idx, j in enumerate(existing)
                if j["relevance"] is None
            ]
            if not failed_indices:
                continue  # All pairs judged successfully
            retry_docs = pooled.get(query, [])
            for idx in failed_indices:
                if idx >= len(retry_docs):
                    continue
                doc = retry_docs[idx]
                if doc["work_id"] != existing[idx]["work_id"]:
                    continue  # Pool/judgment misalignment — skip
                prompt = JUDGE_PROMPT.format(
                    query=query,
                    title=doc["title"],
                    authors=doc.get("authors") or "Unknown",
                    subjects=", ".join(doc.get("subjects", [])[:5]) or "N/A",
                    description=f"Description: {doc['description']}" if doc.get("description") else "",
                )
                relevance = _call_judge(client, url, headers, prompt)
                if relevance is not None:
                    existing[idx]["relevance"] = relevance
                else:
                    errors += 1
                time.sleep(0.5)
            continue

        docs = pooled.get(query, [])
        if not docs:
            continue

        # Judge a query's documents concurrently. Order is preserved by index
        # because the retry path above realigns judgments to the pool by
        # position, so a reordered list would silently mismatch work_ids.
        query_judgments = [None] * len(docs)

        def _judge_one(idx_doc, _query=query):
            idx, doc = idx_doc
            prompt = JUDGE_PROMPT.format(
                query=_query,
                title=doc["title"],
                authors=doc.get("authors") or "Unknown",
                subjects=", ".join(doc.get("subjects", [])[:5]) or "N/A",
                description=f"Description: {doc['description']}" if doc.get("description") else "",
            )
            return idx, doc, _call_judge(client, url, headers, prompt)

        with ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool_exec:
            for idx, doc, relevance in pool_exec.map(_judge_one, enumerate(docs)):
                if relevance is None:
                    errors += 1
                # Store None as-is — excluded from metrics, not coerced to 0
                query_judgments[idx] = {
                    "work_id": doc["work_id"],
                    "title": doc["title"],
                    "relevance": relevance,
                }

        judgments[query] = query_judgments

        # Save incrementally every 5 queries
        if (i + 1) % 5 == 0:
            JUDGMENTS_FILE.write_text(json.dumps(judgments, indent=2, ensure_ascii=False), encoding="utf-8")
            judged_so_far = sum(len(v) for v in judgments.values())
            print(f"  Judged {i+1}/{len(queries)} queries ({judged_so_far} total pairs, {errors} errors)")

    # Final save
    JUDGMENTS_FILE.write_text(json.dumps(judgments, indent=2, ensure_ascii=False), encoding="utf-8")

    # Stats
    all_grades = [j["relevance"] for jlist in judgments.values() for j in jlist if j["relevance"] is not None]
    from collections import Counter
    dist = Counter(all_grades)
    null_count = sum(1 for jlist in judgments.values() for j in jlist if j["relevance"] is None)
    print(f"\nJudging complete: {len(all_grades)} pairs judged, {null_count} failures (excluded)")
    print(f"Grade distribution: {dict(sorted(dist.items()))}")
    return judgments


# ─── Step 4: Eval with paired bootstrap CIs ─────────────────────────────────

def _fetch_search(url: str, timeout: int = 30, attempts: int = 5):
    """GET a search URL, retrying transient network failures.

    A dropped connection mid-run used to be swallowed by `continue`, which
    silently shrank the evaluated set instead of failing: one flaky stretch
    produced a complete-looking results.json built from 43 of 98 queries, with
    every metric and confidence interval computed on the surviving subset. The
    numbers looked plausible, which is exactly what makes it dangerous. Retry
    hard, then let the caller abort rather than quietly publish a partial run.
    """
    last = None
    for attempt in range(attempts):
        try:
            resp = urllib.request.urlopen(url, timeout=timeout)
            return json.loads(resp.read())
        except Exception as e:  # noqa: BLE001 - network, DNS, and decode errors alike
            last = e
            if attempt < attempts - 1:
                time.sleep(RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)])
    raise RuntimeError(f"{url}: {last}")


def compute_metrics_with_cis(judgments: dict, n_bootstrap: int = 1000):
    """Compute MRR, NDCG, Recall with paired bootstrap 95% CIs and per-category breakdown."""
    import numpy as np
    from collections import Counter
    from src.eval.metrics import compute_query_metrics

    modes = ["keyword", "vector", "hybrid", "hybrid+rerank"]

    # Build relevance maps from judgments (including zeros, excluding None)
    relevance_maps = {}
    for query, jlist in judgments.items():
        rmap = {j["work_id"]: j["relevance"] for j in jlist if j["relevance"] is not None}
        if rmap:
            relevance_maps[query] = rmap

    queries_with_relevant = [
        q for q, rmap in relevance_maps.items()
        if any(r > 0 for r in rmap.values())
    ]

    dropped = len(judgments) - len(queries_with_relevant)
    print(f"Queries with at least one relevant doc: {len(queries_with_relevant)}/{len(judgments)} (dropped {dropped})")

    # Load query categories
    query_to_cat = {}
    if QUERIES_FILE.exists():
        all_queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
        query_to_cat = {q["query"]: q["category"] for q in all_queries}
        dropped_cats = Counter(
            query_to_cat.get(q, "unknown")
            for q in judgments
            if q not in queries_with_relevant
        )
        if dropped_cats:
            print(f"  Dropped by category: {dict(dropped_cats)}")

    # Collect per-query metrics for all modes (aligned by query)
    mode_scores = {mode: {} for mode in modes}
    fetch_failures = []

    for mode in modes:
        print(f"  Fetching {mode} results...")
        for query in queries_with_relevant:
            if mode == "hybrid+rerank":
                url = f"{API}/search?q={urllib.parse.quote(query)}&mode=hybrid&top_k=10&understand=false&rerank=true"
            else:
                url = f"{API}/search?q={urllib.parse.quote(query)}&mode={mode}&top_k=10&understand=false"

            try:
                data = _fetch_search(url)
            except RuntimeError as e:
                print(f"    ERROR [{mode}] \"{query[:30]}\": {e}")
                fetch_failures.append((mode, query))
                continue

            retrieved_ids = [
                r["work_id"].replace("/works/", "")
                for r in data.get("results", [])
                if r.get("work_id")
            ]
            rmap = relevance_maps[query]
            m = compute_query_metrics(query, retrieved_ids, rmap, k=10)
            mode_scores[mode][query] = m

    # Refuse to publish a partial run. Metrics are paired across modes, so a
    # query missing from one mode silently changes what every comparison is
    # averaging over.
    if fetch_failures:
        by_mode = Counter(mode for mode, _ in fetch_failures)
        raise SystemExit(
            f"\nABORTING: {len(fetch_failures)} search request(s) failed after retries "
            f"({dict(by_mode)}).\nResults were NOT written -- a partial run would produce "
            f"plausible-looking metrics over a subset of queries.\nCheck connectivity to "
            f"{API} and re-run."
        )

    # Align queries (only those present in ALL modes)
    common_queries = set(queries_with_relevant)
    for mode in modes:
        common_queries &= set(mode_scores[mode].keys())
    common_queries = sorted(common_queries)
    print(f"\nQueries evaluated across all modes: {len(common_queries)}")

    if not common_queries:
        print("ERROR: no queries have results in all modes. Cannot compute metrics.")
        return {"modes": [], "oracle": None, "per_category": {}}

    # Note achievable recall ceiling (recall@10 < 1.0 when pool has >10 relevant docs)
    recall_ceilings = []
    for q in common_queries:
        rmap = relevance_maps[q]
        n_relevant = sum(1 for v in rmap.values() if v > 0)
        if n_relevant > 0:
            recall_ceilings.append(min(1.0, 10 / n_relevant))
    if recall_ceilings:
        mean_ceiling = float(np.mean(recall_ceilings))
        print(f"  Mean achievable Recall@10 ceiling: {mean_ceiling:.4f} "
              f"(capped when >10 relevant docs in pool)")

    # --- Point estimates + bootstrap CIs ---
    def bootstrap_ci(arr, rng):
        n = len(arr)
        boots = [arr[rng.integers(0, n, n)].mean() for _ in range(n_bootstrap)]
        return np.percentile(boots, [2.5, 97.5])

    results = []

    for mi, mode in enumerate(modes):
        scores = [mode_scores[mode][q] for q in common_queries]
        n = len(scores)
        mrr_arr = np.array([m.mrr for m in scores])
        ndcg_arr = np.array([m.ndcg for m in scores])
        recall_arr = np.array([m.recall for m in scores])

        # Independent RNG per mode for reproducibility
        mode_rng = np.random.default_rng(42 + mi)

        mrr_ci = bootstrap_ci(mrr_arr, mode_rng)
        ndcg_ci = bootstrap_ci(ndcg_arr, mode_rng)
        recall_ci = bootstrap_ci(recall_arr, mode_rng)

        r = {
            "strategy": mode,
            "n_queries": n,
            "mrr_at_10": round(float(mrr_arr.mean()), 4),
            "mrr_ci_95": [round(float(mrr_ci[0]), 4), round(float(mrr_ci[1]), 4)],
            "ndcg_at_10": round(float(ndcg_arr.mean()), 4),
            "ndcg_ci_95": [round(float(ndcg_ci[0]), 4), round(float(ndcg_ci[1]), 4)],
            "recall_at_10": round(float(recall_arr.mean()), 4),
            "recall_ci_95": [round(float(recall_ci[0]), 4), round(float(recall_ci[1]), 4)],
        }
        results.append(r)
        print(f"  {mode:>14}: MRR={r['mrr_at_10']:.4f} [{r['mrr_ci_95'][0]:.4f}, {r['mrr_ci_95'][1]:.4f}]  "
              f"NDCG={r['ndcg_at_10']:.4f} [{r['ndcg_ci_95'][0]:.4f}, {r['ndcg_ci_95'][1]:.4f}]")

    # Paired comparisons (bootstrap on per-query deltas) — NDCG and MRR
    print("\n--- Paired comparisons (bootstrap on per-query deltas) ---")
    comparisons = [
        ("hybrid", "keyword"),
        ("hybrid+rerank", "keyword"),
        ("hybrid+rerank", "hybrid"),
    ]
    n_intervals = 0  # total delta-testing intervals for multiple-comparison disclosure

    for ci_idx, (mode, baseline) in enumerate(comparisons):
        pair_rng = np.random.default_rng(1000 + ci_idx)
        n_q = len(common_queries)
        # Shared bootstrap indices across metrics within this comparison
        boot_indices = pair_rng.integers(0, n_q, (n_bootstrap, n_q))

        for metric_name, attr in [("NDCG", "ndcg"), ("MRR", "mrr")]:
            mode_arr = np.array([getattr(mode_scores[mode][q], attr)
                                 for q in common_queries])
            base_arr = np.array([getattr(mode_scores[baseline][q], attr)
                                 for q in common_queries])
            deltas = mode_arr - base_arr

            boot_deltas = [deltas[boot_indices[i]].mean()
                           for i in range(n_bootstrap)]
            ci = np.percentile(boot_deltas, [2.5, 97.5])
            mean_delta = float(deltas.mean())
            sig = "significant" if (ci[0] > 0 or ci[1] < 0) else "NOT significant"

            # Fraction of replicates on the opposite side of zero
            if mean_delta > 0:
                p_opp = sum(1 for d in boot_deltas if d <= 0) / n_bootstrap
            elif mean_delta < 0:
                p_opp = sum(1 for d in boot_deltas if d >= 0) / n_bootstrap
            else:
                p_opp = 0.5

            n_intervals += 1

            print(f"  {mode} - {baseline}: {metric_name} delta={mean_delta:+.4f} "
                  f"[{ci[0]:+.4f}, {ci[1]:+.4f}] ({sig}, p_opp={p_opp:.3f})")

            for r in results:
                if r["strategy"] == mode:
                    r[f"{attr}_delta_vs_{baseline}"] = round(mean_delta, 4)
                    r[f"{attr}_delta_ci_vs_{baseline}"] = [round(float(ci[0]), 4), round(float(ci[1]), 4)]
                    r[f"{attr}_delta_p_opp_vs_{baseline}"] = round(p_opp, 4)

    # --- Per-category breakdown ---
    category_results = {}
    categories = sorted(set(query_to_cat.get(q, "unknown") for q in common_queries))
    if query_to_cat and len(categories) > 1:
        print("\n--- Per-category breakdown ---")
        for cat_idx, cat in enumerate(categories):
            cat_queries = [q for q in common_queries if query_to_cat.get(q, "unknown") == cat]
            if not cat_queries:
                continue
            n_cat = len(cat_queries)
            underpowered = " [low-n]" if n_cat < 30 else ""
            print(f"\n  {cat} (n={n_cat}):{underpowered}")
            cat_data = {"n_queries": n_cat, "modes": {}}
            for mode in modes:
                cat_mrr = float(np.mean([mode_scores[mode][q].mrr for q in cat_queries]))
                cat_ndcg = float(np.mean([mode_scores[mode][q].ndcg for q in cat_queries]))
                cat_recall = float(np.mean([mode_scores[mode][q].recall for q in cat_queries]))
                print(f"    {mode:>14}:  MRR={cat_mrr:.4f}  NDCG={cat_ndcg:.4f}  Recall={cat_recall:.4f}")
                cat_data["modes"][mode] = {
                    "mrr": round(cat_mrr, 4),
                    "ndcg": round(cat_ndcg, 4),
                    "recall": round(cat_recall, 4),
                }
            # Rerank delta with paired bootstrap CI
            hybrid_cat_arr = np.array([mode_scores["hybrid"][q].ndcg for q in cat_queries])
            rerank_cat_arr = np.array([mode_scores["hybrid+rerank"][q].ndcg for q in cat_queries])
            cat_deltas = rerank_cat_arr - hybrid_cat_arr
            delta = float(cat_deltas.mean())

            cat_rng = np.random.default_rng(2000 + cat_idx)
            boot_cat = [cat_deltas[cat_rng.integers(0, n_cat, n_cat)].mean()
                        for _ in range(n_bootstrap)]
            cat_ci = np.percentile(boot_cat, [2.5, 97.5])
            n_intervals += 1

            if delta > 0:
                p_opp = sum(1 for d in boot_cat if d <= 0) / n_bootstrap
            elif delta < 0:
                p_opp = sum(1 for d in boot_cat if d >= 0) / n_bootstrap
            else:
                p_opp = 0.5

            sig_cat = "significant" if (cat_ci[0] > 0 or cat_ci[1] < 0) else "NOT significant"
            print(f"    {'rerank delta':>14}: NDCG={delta:+.4f} "
                  f"[{cat_ci[0]:+.4f}, {cat_ci[1]:+.4f}] ({sig_cat}, p_opp={p_opp:.3f})")
            cat_data["rerank_ndcg_delta"] = round(delta, 4)
            cat_data["rerank_ndcg_delta_ci"] = [round(float(cat_ci[0]), 4), round(float(cat_ci[1]), 4)]
            cat_data["rerank_ndcg_delta_p_opp"] = round(p_opp, 4)
            category_results[cat] = cat_data

        # Summary table
        print("\n  --- Rerank impact summary ---")
        print(f"  {'Category':<14} {'N':>3}  {'hybrid':>8}  {'rerank':>8}  "
              f"{'delta':>8}  {'95% CI':>18}  {'p_opp':>6}")
        for cat in categories:
            if cat not in category_results:
                continue
            cd = category_results[cat]
            h = cd["modes"]["hybrid"]["ndcg"]
            rk = cd["modes"]["hybrid+rerank"]["ndcg"]
            d = cd["rerank_ndcg_delta"]
            ci_lo, ci_hi = cd["rerank_ndcg_delta_ci"]
            p = cd["rerank_ndcg_delta_p_opp"]
            flag = " [low-n]" if cd["n_queries"] < 30 else ""
            print(f"  {cat:<14} {cd['n_queries']:>3}  {h:>8.4f}  {rk:>8.4f}  "
                  f"{d:>+8.4f}  [{ci_lo:+.4f}, {ci_hi:+.4f}]  {p:>6.3f}{flag}")

    # Multiple-comparison disclosure
    print(f"\n  Note: {n_intervals} unadjusted 95% percentile-bootstrap intervals were computed.")
    print(f"  Under the null, ~{n_intervals / 20:.1f} of {n_intervals} would exclude zero by chance alone.")
    print("  p_opp = fraction of bootstrap replicates on the opposite side of zero from the estimate.")

    # --- Oracle analysis (best possible reranking of the full candidate pool) ---
    print("\n--- Oracle reranking (ceiling analysis) ---")
    oracle_depth = max(10, int(10 * RERANK_DEPTH_MULTIPLIER))
    oracle_scores = []
    oracle_queries = []
    oracle_failures = []

    for query in common_queries:
        rmap = relevance_maps[query]
        url = f"{API}/search?q={urllib.parse.quote(query)}&mode=hybrid&top_k={oracle_depth}&understand=false"
        try:
            data = _fetch_search(url)
            retrieved_ids = [
                r["work_id"].replace("/works/", "")
                for r in data.get("results", [])
                if r.get("work_id")
            ]
            # Oracle: sort ALL candidates by true relevance, best first
            oracle_order = sorted(retrieved_ids, key=lambda x: rmap.get(x, 0), reverse=True)
            m = compute_query_metrics(query, oracle_order, rmap, k=10)
            oracle_scores.append(m.ndcg)
            oracle_queries.append(query)
        except RuntimeError as e:
            # Skip — don't bias the ceiling with a fallback value
            print(f"    WARN: skipping oracle for \"{query[:30]}\": {e}")
            oracle_failures.append(query)

    # The ceiling is compared against hybrid on the same queries, so a partial
    # oracle is a different (and unstated) sample than the table above it.
    if oracle_failures:
        raise SystemExit(
            f"\nABORTING: oracle analysis failed for {len(oracle_failures)} of "
            f"{len(common_queries)} queries.\nResults were NOT written."
        )

    oracle_data = None
    if oracle_queries:
        oracle_ndcg = np.array(oracle_scores)
        hybrid_ndcg_aligned = np.array([mode_scores["hybrid"][q].ndcg for q in oracle_queries])
        headroom = float((oracle_ndcg - hybrid_ndcg_aligned).mean())
        hybrid_mean = float(hybrid_ndcg_aligned.mean())
        pct = f" ({headroom / hybrid_mean * 100:+.1f}%)" if hybrid_mean > 0 else ""
        print(f"  Oracle pool depth: {oracle_depth} candidates")
        print(f"  Hybrid NDCG:  {hybrid_mean:.4f} (n={len(oracle_queries)})")
        print(f"  Oracle NDCG:  {float(oracle_ndcg.mean()):.4f}")
        print(f"  Headroom:     {headroom:+.4f}{pct}")
        oracle_data = {
            "pool_depth": oracle_depth,
            "n_queries": len(oracle_queries),
            "hybrid_ndcg": round(hybrid_mean, 4),
            "oracle_ndcg": round(float(oracle_ndcg.mean()), 4),
            "headroom": round(headroom, 4),
        }
    else:
        print("  WARN: no queries available for oracle analysis")

    # Save all results
    output = {
        "modes": results,
        "oracle": oracle_data,
    }
    if category_results:
        output["per_category"] = category_results

    RESULTS_FILE.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved to {RESULTS_FILE}")
    return output


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full eval redesign pipeline")
    parser.add_argument("--step", choices=["generate", "pool", "judge", "eval", "all"], required=True)
    parser.add_argument("--api-url", default=None, help="Override API URL")
    args = parser.parse_args()

    global API
    if args.api_url:
        API = args.api_url

    if args.step in ("generate", "all"):
        queries = generate_queries()
    else:
        queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8")) if QUERIES_FILE.exists() else []

    if args.step in ("pool", "all"):
        if not queries:
            queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
        pooled = pool_results(queries)
    else:
        pooled = json.loads(POOLED_FILE.read_text(encoding="utf-8")) if POOLED_FILE.exists() else {}

    if args.step in ("judge", "all"):
        if not queries:
            queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
        if not pooled:
            pooled = json.loads(POOLED_FILE.read_text(encoding="utf-8"))
        judgments = judge_documents(queries, pooled)
    else:
        judgments = json.loads(JUDGMENTS_FILE.read_text(encoding="utf-8")) if JUDGMENTS_FILE.exists() else {}

    if args.step in ("eval", "all"):
        if not judgments:
            judgments = json.loads(JUDGMENTS_FILE.read_text(encoding="utf-8"))
        compute_metrics_with_cis(judgments)


if __name__ == "__main__":
    main()
