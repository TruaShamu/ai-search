"""LLM-based query expansion — generates synonyms/related terms to improve TF-IDF recall.

Uses gpt-5.4-nano to produce 3-5 expansion terms per query.
Only applied to the keyword/TF-IDF component (vector search already handles synonyms via embeddings).

Design:
- Fast: single LLM call, ~200-500ms
- Cached: in-memory dict (same query = same expansion)
- Targeted: expansions are appended to TF-IDF search text, not the vector query
"""

import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)

EXPANSION_PROMPT = """Given this book search query, generate 3-5 related terms or synonyms that a relevant book might contain in its title, description, or subjects. These will be used to improve keyword search recall.

Rules:
- Output ONLY the terms, one per line
- No numbering, no explanations
- Include synonyms, related concepts, and alternate phrasings
- Think about what words would appear in a book's description or title

Query: "{query}"

Related terms:"""


class QueryExpander:
    """LLM-based query expansion using gpt-5.4-nano."""

    def __init__(self):
        self.endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
        self.api_key = os.environ["AZURE_OPENAI_KEY"]
        self.deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-54-nano")
        self.api_version = "2024-12-01-preview"
        self._cache: dict[str, tuple[str, list[str]]] = {}

    def expand(self, query: str) -> tuple[str, list[str], float]:
        """
        Expand a query with related terms.

        Returns:
            (expanded_query, expansion_terms, latency_ms)
            expanded_query = original + " " + expansion terms joined
        """
        # Cache hit
        if query.lower() in self._cache:
            expanded, terms = self._cache[query.lower()]
            return expanded, terms, 0.0

        url = (
            f"{self.endpoint}/openai/deployments/{self.deployment}"
            f"/chat/completions?api-version={self.api_version}"
        )

        start = time.time()
        try:
            resp = httpx.post(
                url,
                headers={"api-key": self.api_key, "Content-Type": "application/json"},
                json={
                    "messages": [
                        {"role": "user", "content": EXPANSION_PROMPT.format(query=query)},
                    ],
                    "max_completion_tokens": 80,
                    "temperature": 0.3,
                },
                timeout=5.0,
            )
            latency_ms = (time.time() - start) * 1000

            if resp.status_code != 200:
                return query, [], latency_ms

            content = resp.json()["choices"][0]["message"]["content"].strip()
            terms = [t.strip() for t in content.split("\n") if t.strip()]
            # Limit to 5 terms max
            terms = terms[:5]

            expanded = query + " " + " ".join(terms)
            self._cache[query.lower()] = (expanded, terms)
            return expanded, terms, latency_ms

        except Exception:
            latency_ms = (time.time() - start) * 1000
            return query, [], latency_ms


if __name__ == "__main__":
    """Quick test of query expansion."""
    expander = QueryExpander()

    test_queries = [
        "romance set in Scotland",
        "books about loneliness and isolation",
        "artificial intelligence and machine learning",
        "coming of age story set in India",
        "horror supernatural ghost story",
    ]

    for q in test_queries:
        expanded, terms, latency = expander.expand(q)
        print(f"Query: {q}")
        print(f"  Terms: {terms}")
        print(f"  Latency: {latency:.0f}ms")
        print()
