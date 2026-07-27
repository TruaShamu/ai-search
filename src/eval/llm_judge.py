"""LLM-as-Judge for search relevance evaluation.

Uses Azure OpenAI (gpt-5.4-nano) to score (query, document) relevance on a 0-2 scale,
replacing the keyword-overlap heuristic that biases toward BM25.

Usage:
    python -m src.eval.llm_judge                    # Judge all pooled results
    python -m src.eval.llm_judge --dry-run          # Preview without API calls
    python -m src.eval.llm_judge --compare-heuristic # Compare LLM vs heuristic judgments
"""

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import httpx
from dotenv import load_dotenv

from src.eval.dataset import EvalQuery, RelevanceJudgment, load_eval_dataset, get_eval_queries
from src.azure_search.search import HybridSearchEngine


ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
OUTPUT_PATH = Path("data/eval/queries_llm_judged.json")

SYSTEM_PROMPT = """You are a search relevance judge for a book search engine.
Given a user's search query and a book result, rate how well the book matches what the user is looking for.

Scoring scale:
0 = NOT RELEVANT — Wrong topic, misleading keyword match, or unrelated to the query intent.
1 = PARTIALLY RELEVANT — Related topic or theme, but not a direct match for what the user wants.
2 = HIGHLY RELEVANT — Directly matches the user's search intent. A user searching this query would be satisfied finding this book.

Focus on SEMANTIC relevance — whether the book's content matches the user's intent — not keyword overlap.

Respond in exactly this JSON format, nothing else:
{"relevance": <0|1|2>, "reasoning": "<one sentence>"}"""

USER_TEMPLATE = """Query: "{query}"

Book:
- Title: {title}
- Author: {authors}
- Description: {description}
- Subjects: {subjects}"""


class LLMJudge:
    """Scores (query, document) relevance using Azure OpenAI."""

    def __init__(self):
        load_dotenv(ENV_PATH)
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        self.key = os.getenv("AZURE_OPENAI_KEY", "")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-54-nano")
        self.api_version = "2024-10-21"

        if not self.endpoint or not self.key:
            raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be set in .env")

        self.url = (
            f"{self.endpoint}/openai/deployments/{self.deployment}"
            f"/chat/completions?api-version={self.api_version}"
        )
        self.headers = {"api-key": self.key, "Content-Type": "application/json"}
        self.client = httpx.Client(timeout=30)

        # Track token usage
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    def judge(self, query: str, doc: dict, retries: int = 3) -> tuple[int, str]:
        """Score a single (query, doc) pair. Returns (relevance, reasoning)."""
        title = doc.get("title", "Unknown")
        authors = doc.get("authors", "Unknown")
        description = doc.get("description", "No description available.")
        subjects = ", ".join(doc.get("subjects", [])[:10]) or "None"

        # Truncate long descriptions
        if len(description) > 500:
            description = description[:500] + "..."

        user_msg = USER_TEMPLATE.format(
            query=query,
            title=title,
            authors=authors,
            description=description,
            subjects=subjects,
        )

        body = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "max_completion_tokens": 100,
            "temperature": 0,
        }

        for attempt in range(retries):
            try:
                resp = self.client.post(self.url, json=body, headers=self.headers)

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 10))
                    print(f"    Rate limited, waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                resp.raise_for_status()
                data = resp.json()

                # Track usage
                usage = data.get("usage", {})
                self.total_input_tokens += usage.get("prompt_tokens", 0)
                self.total_output_tokens += usage.get("completion_tokens", 0)
                self.total_calls += 1

                content = data["choices"][0]["message"]["content"].strip()
                result = json.loads(content)
                relevance = int(result["relevance"])
                reasoning = result.get("reasoning", "")

                if relevance not in (0, 1, 2):
                    relevance = max(0, min(2, relevance))

                return relevance, reasoning

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt < retries - 1:
                    print(f"    Parse error ({e}), retrying...")
                    time.sleep(2)
                    continue
                print(f"    Failed to parse after {retries} attempts: {e}")
                return 0, f"parse_error: {e}"

            except httpx.HTTPStatusError as e:
                if attempt < retries - 1:
                    print(f"    HTTP {e.response.status_code}, retrying...")
                    time.sleep(5)
                    continue
                print(f"    HTTP error after {retries} attempts: {e}")
                return 0, f"http_error: {e.response.status_code}"

        return 0, "max_retries_exceeded"

    def cost_estimate(self) -> dict:
        """Estimate cost based on tracked token usage."""
        # gpt-5.4-nano pricing: $0.20/1M input, $1.25/1M output
        input_cost = (self.total_input_tokens / 1_000_000) * 0.20
        output_cost = (self.total_output_tokens / 1_000_000) * 1.25
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_calls": self.total_calls,
            "input_cost_usd": round(input_cost, 4),
            "output_cost_usd": round(output_cost, 4),
            "total_cost_usd": round(input_cost + output_cost, 4),
        }


def pool_results_for_judging(
    engine: HybridSearchEngine,
    queries: list[EvalQuery],
    top_k: int = 10,
) -> dict[str, list[dict]]:
    """Pool top results from all retrieval modes for each query."""
    pooled = {}
    modes = ["keyword", "vector", "hybrid"]

    for eq in queries:
        seen_ids = set()
        docs = []
        for mode in modes:
            result = engine.search(query=eq.query, top_k=top_k, mode=mode)
            if "error" in result:
                continue
            for r in result["results"]:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    docs.append(r)

        pooled[eq.query] = docs
        print(f"  Pooled {len(docs)} unique docs for: \"{eq.query}\"")

    return pooled


def run_llm_judging(dry_run: bool = False, compare_heuristic: bool = False):
    """Run LLM-as-judge on all pooled eval results."""
    queries = get_eval_queries()
    print(f"Loaded {len(queries)} eval queries\n")

    print("Pooling search results...")
    engine = HybridSearchEngine()
    pooled = pool_results_for_judging(engine, queries)

    total_pairs = sum(len(docs) for docs in pooled.values())
    print(f"\nTotal (query, doc) pairs to judge: {total_pairs}")

    if dry_run:
        est_input = total_pairs * 250  # ~250 tokens per prompt
        est_output = total_pairs * 30   # ~30 tokens per response
        cost_in = (est_input / 1_000_000) * 0.20
        cost_out = (est_output / 1_000_000) * 1.25
        print(f"Estimated cost: ~${cost_in + cost_out:.4f}")
        print("(Use without --dry-run to actually run)")
        return

    print("\nStarting LLM judging (gpt-5.4-nano)...\n")
    judge = LLMJudge()

    # Load existing heuristic judgments for comparison
    heuristic_judgments = {}
    if compare_heuristic:
        annotated_path = Path("data/eval/queries_annotated.json")
        if annotated_path.exists():
            heuristic_queries = load_eval_dataset(annotated_path)
            for q in heuristic_queries:
                heuristic_judgments[q.query] = {j.work_id: j.relevance for j in q.relevant}

    annotated_queries = []
    agreement_stats = {"agree": 0, "disagree": 0, "llm_higher": 0, "llm_lower": 0}

    for eq in queries:
        docs = pooled.get(eq.query, [])
        if not docs:
            annotated_queries.append(eq)
            continue

        print(f"Judging: \"{eq.query}\" ({len(docs)} docs)")
        judgments = []

        for doc in docs:
            relevance, reasoning = judge.judge(eq.query, doc)
            judgments.append(RelevanceJudgment(
                work_id=doc["id"],
                relevance=relevance,
                title=doc.get("title", ""),
            ))

            rel_marker = ["✗", "~", "✓"][relevance]
            print(f"  [{rel_marker}] {relevance} — {doc.get('title', '?')[:50]}")

            # Compare with heuristic
            if compare_heuristic and eq.query in heuristic_judgments:
                h_rel = heuristic_judgments[eq.query].get(doc["id"])
                if h_rel is not None:
                    if h_rel == relevance:
                        agreement_stats["agree"] += 1
                    else:
                        agreement_stats["disagree"] += 1
                        if relevance > h_rel:
                            agreement_stats["llm_higher"] += 1
                        else:
                            agreement_stats["llm_lower"] += 1

            # Rate limiting: be gentle with 1 TPM capacity
            time.sleep(2)

        annotated_queries.append(EvalQuery(
            query=eq.query,
            relevant=[j for j in judgments if j.relevance > 0],
            source="llm_judge",
            category=eq.category,
        ))
        print()

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(q) for q in annotated_queries]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Summary
    total_judgments = sum(len(q.relevant) for q in annotated_queries)
    cost = judge.cost_estimate()

    print("=" * 60)
    print("LLM-as-Judge Complete")
    print("=" * 60)
    print(f"  Queries judged:     {len(queries)}")
    print(f"  Total pairs:        {cost['total_calls']}")
    print(f"  Relevant found:     {total_judgments} (rel>0)")
    print(f"  Input tokens:       {cost['input_tokens']:,}")
    print(f"  Output tokens:      {cost['output_tokens']:,}")
    print(f"  Total cost:         ${cost['total_cost_usd']:.4f}")
    print(f"  Saved to:           {OUTPUT_PATH}")

    if compare_heuristic and agreement_stats["agree"] + agreement_stats["disagree"] > 0:
        total = agreement_stats["agree"] + agreement_stats["disagree"]
        pct = agreement_stats["agree"] / total * 100
        print("\n  Heuristic comparison:")
        print(f"    Agreement:        {pct:.0f}% ({agreement_stats['agree']}/{total})")
        print(f"    LLM rated higher: {agreement_stats['llm_higher']} (semantic relevance missed by heuristic)")
        print(f"    LLM rated lower:  {agreement_stats['llm_lower']} (keyword match != relevance)")

    return annotated_queries


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM-as-judge relevance annotation")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost without calling API")
    parser.add_argument("--compare-heuristic", action="store_true", help="Compare with existing heuristic judgments")
    args = parser.parse_args()

    run_llm_judging(dry_run=args.dry_run, compare_heuristic=args.compare_heuristic)
