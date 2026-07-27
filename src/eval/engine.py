"""Unified search adapter for evaluation scripts.

Supports two backends:
  1. HTTP API — lightweight, no model/vectorizer loading required.
     Set via --api-url CLI flag or EVAL_API_URL environment variable.
  2. Local QdrantSearch — direct Qdrant access (needs embedding model + TF-IDF vectorizer).

Call signature matches the legacy HybridSearchEngine so eval script diffs stay small:
  .search(query, top_k, mode, year_min, year_max, tier, rerank) -> dict
  .compare(query, top_k) -> dict

NOTE: The original eval numbers were measured on a ~13K-book Azure AI Search index using
BM25 for keyword scoring. The Qdrant backend uses TF-IDF sparse vectors for "keyword" mode
and RRF fusion for "hybrid" mode, so historical numbers are not directly comparable.

DESIGN NOTE — why the HTTP backend *raises* on errors instead of returning empty results:
An eval harness must never silently convert a network hiccup (503, timeout, connection
reset) into MRR=0 / NDCG=0 / Recall=0 for a query.  That corrupts aggregate metrics in a
way that is invisible in the final table and not reproducible across runs.  By raising, we
force the caller (the eval script) to decide explicitly whether to skip or abort — the
adapter must not paper over infrastructure flakiness as poor retrieval quality.
"""

from __future__ import annotations

import os
import random
import time

import httpx


class EvalSearchEngine:
    """Adapter that routes eval search calls to an HTTP API or local Qdrant."""

    def __init__(self, *, api_url: str | None = None, qdrant_url: str | None = None):
        """Create an eval engine.

        Resolution order:
          1. Explicit *api_url* → use HTTP backend.
          2. Explicit *qdrant_url* → use local QdrantSearch.
          3. EVAL_API_URL env var → use HTTP backend.
          4. Fall back to local QdrantSearch (localhost:6333).
        """
        resolved_api = api_url or os.environ.get("EVAL_API_URL")

        if resolved_api:
            self._backend = _HttpBackend(resolved_api)
        elif qdrant_url:
            self._backend = _QdrantBackend(qdrant_url)
        else:
            # Try HTTP API first (lightweight); fall back to local Qdrant
            self._backend = _QdrantBackend()

    # --- public interface (drop-in for HybridSearchEngine) ---

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        year_min: int | None = None,
        year_max: int | None = None,
        tier: int | None = None,
        rerank: bool = False,
    ) -> dict:
        return self._backend.search(
            query=query,
            top_k=top_k,
            mode=mode,
            year_min=year_min,
            year_max=year_max,
            tier=tier,
            rerank=rerank,
        )

    def compare(self, query: str, top_k: int = 5) -> dict:
        return self._backend.compare(query=query, top_k=top_k)


# ---------------------------------------------------------------------------
# HTTP backend — talks to the live (or local) FastAPI service
# ---------------------------------------------------------------------------

class _HttpBackend:
    """Calls the deployed /search endpoint over HTTP."""

    MAX_RETRIES = 3
    BACKOFF_BASE = 1.0  # seconds: 1, 2, 4

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        self._client = httpx.Client(timeout=30)
        print(f"[EvalEngine] Using HTTP API: {self.api_url}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        year_min: int | None = None,
        year_max: int | None = None,
        tier: int | None = None,
        rerank: bool = False,
    ) -> dict:
        params: dict = {"q": query, "mode": mode, "top_k": top_k}
        if rerank:
            params["rerank"] = "true"
        if year_min is not None:
            params["year_min"] = year_min
        if year_max is not None:
            params["year_max"] = year_max
        if tier is not None:
            params["tier"] = tier

        last_exc: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            start = time.perf_counter()
            try:
                resp = self._client.get(f"{self.api_url}/search", params=params)
                latency_ms = round((time.perf_counter() - start) * 1000, 1)

                if resp.status_code >= 500:
                    # Retryable server error
                    last_exc = httpx.HTTPStatusError(
                        f"Server error {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                    self._backoff(attempt)
                    continue

                # 4xx → deterministic caller error, raise immediately
                resp.raise_for_status()

                data = resp.json()
                data.setdefault("latency_ms", latency_ms)
                data.setdefault("query", query)
                data.setdefault("mode", mode)
                return data

            except httpx.TransportError as exc:
                # Timeouts, connection resets, DNS failures — retryable
                last_exc = exc
                if attempt < self.MAX_RETRIES:
                    self._backoff(attempt)
                    continue

        raise RuntimeError(
            f"Search failed after {self.MAX_RETRIES} retries "
            f"(query={query!r}, mode={mode!r}): {last_exc}"
        ) from last_exc

    @staticmethod
    def _backoff(attempt: int) -> None:
        delay = _HttpBackend.BACKOFF_BASE * (2 ** (attempt - 1))
        jitter = random.uniform(0, delay * 0.25)
        time.sleep(delay + jitter)

    def compare(self, query: str, top_k: int = 5) -> dict:
        return {
            mode: self.search(query=query, top_k=top_k, mode=mode)
            for mode in ("keyword", "vector", "hybrid")
        }


# ---------------------------------------------------------------------------
# Qdrant backend — direct access via QdrantSearch
# ---------------------------------------------------------------------------

class _QdrantBackend:
    """Wraps src.qdrant.client.QdrantSearch (lazy-loaded)."""

    def __init__(self, url: str | None = None):
        from src.qdrant.client import QdrantSearch

        kwargs = {}
        if url:
            kwargs["url"] = url
        self._engine = QdrantSearch(**kwargs)
        print("[EvalEngine] Using local QdrantSearch")

    def search(self, **kwargs) -> dict:
        return self._engine.search(**kwargs)

    def compare(self, **kwargs) -> dict:
        return self._engine.compare(**kwargs)
