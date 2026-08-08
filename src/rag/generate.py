"""RAG pipeline — grounded answer generation with citation tracking.

Uses hybrid search to find relevant books, then generates a natural language
answer grounded in those sources with hallucination guardrails.
"""

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from dotenv import load_dotenv

from src.telemetry import span

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH)


SYSTEM_PROMPT = """You are a book recommendation assistant. Answer the user's question using ONLY the book information provided below.

Rules:
1. Cite books by their [number] (e.g., [1], [2]).
2. Only mention books that appear in the provided sources.
3. If the sources don't contain enough information to answer well, say so honestly.
4. Do not invent books, authors, or plot details not in the sources.
5. Be concise but helpful — 2-4 sentences for simple queries, up to a paragraph for complex ones.
6. If recommending, briefly explain WHY each book fits the query."""

CONTEXT_TEMPLATE = """[{rank}] "{title}" by {authors} ({year})
{description}
Subjects: {subjects}
"""


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict]
    citations_valid: bool
    hallucinated_titles: list[str] = field(default_factory=list)
    latency_ms: dict = field(default_factory=dict)
    token_usage: dict = field(default_factory=dict)
    model: str = ""


class RAGPipeline:
    """Retrieval-Augmented Generation for book questions."""

    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        self.key = os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY", "")
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

    def build_context(self, results: list[dict], max_sources: int = 5) -> tuple[str, list[dict]]:
        """Build context string and source list from search results."""
        sources = []
        context_parts = []

        for i, doc in enumerate(results[:max_sources], 1):
            title = doc.get("title", "Unknown")
            authors = doc.get("authors", "Unknown")
            year = doc.get("year") or "n.d."
            description = doc.get("description", "No description available.")
            subjects = ", ".join(doc.get("subjects", [])[:8]) or "None listed"

            context_parts.append(CONTEXT_TEMPLATE.format(
                rank=i,
                title=title,
                authors=authors,
                year=year,
                description=description[:500],
                subjects=subjects,
            ))

            sources.append({
                "rank": i,
                "title": title,
                "authors": authors,
                "year": year,
                "work_id": doc.get("work_id", doc.get("id", "")),
            })

        return "\n".join(context_parts), sources

    def generate(self, question: str, context: str) -> tuple[str, dict]:
        """Call Azure OpenAI to generate an answer."""
        body = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"},
            ],
            "max_completion_tokens": 500,
            "temperature": 0.3,
        }

        resp = self.client.post(self.url, json=body, headers=self.headers)
        resp.raise_for_status()
        data = resp.json()

        answer = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return answer, {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "model": data.get("model", self.deployment),
        }

    def validate_citations(self, answer: str, sources: list[dict]) -> tuple[bool, list[str]]:
        """Check that all book titles mentioned in the answer exist in sources."""
        source_titles = {s["title"].lower() for s in sources}
        hallucinated = []

        # Find quoted/formatted titles in the answer
        bold_titles = re.findall(r'\*\*(.+?)\*\*', answer)
        italic_titles = re.findall(r'(?<!\w)_([^_]+?)_(?!\w)', answer)
        quoted_titles = re.findall(r'"([^"]+)"', answer)
        candidate_titles = bold_titles + italic_titles + quoted_titles

        for title in candidate_titles:
            title_lower = title.lower().strip()
            # Strip citation markers like "[5] " prefix and markdown formatting
            title_lower = re.sub(r'^\[\d+\]\s*', '', title_lower)
            title_lower = title_lower.strip('_* ')
            if not title_lower or len(title_lower) < 3:
                continue
            # Check if it matches any source (fuzzy: substring match)
            if not any(title_lower in st or st in title_lower for st in source_titles):
                hallucinated.append(title)

        return len(hallucinated) == 0, hallucinated

    def ask(self, question: str, search_results: list[dict], max_sources: int = 5) -> RAGResponse:
        """Full RAG pipeline: context building → generation → validation."""
        # Build context
        t0 = time.time()
        with span("rag.build_context", **{"rag.max_sources": max_sources}):
            context, sources = self.build_context(search_results, max_sources)
        context_ms = (time.time() - t0) * 1000

        # Generate answer
        t1 = time.time()
        with span("rag.generate", **{"llm.model": self.deployment}):
            answer, usage = self.generate(question, context)
        generation_ms = (time.time() - t1) * 1000

        # Validate citations
        with span("rag.validate_citations"):
            citations_valid, hallucinated = self.validate_citations(answer, sources)

        return RAGResponse(
            answer=answer,
            sources=sources,
            citations_valid=citations_valid,
            hallucinated_titles=hallucinated,
            latency_ms={
                "context_build": round(context_ms, 1),
                "generation": round(generation_ms, 1),
                "total": round(context_ms + generation_ms, 1),
            },
            token_usage=usage,
            model=usage.get("model", self.deployment),
        )
