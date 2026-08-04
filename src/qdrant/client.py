from __future__ import annotations

import os
import pickle
import threading
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from src.reranker.config import rerank_fetch_k

load_dotenv()

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "books"
DEFAULT_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_DIM = 256
INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "index"
VECTORIZER_PATH = INDEX_DIR / "tfidf_vectorizer.pkl"

_UNSET = object()  # sentinel for lazy reranker init

# Qdrant's server-side RRF assigns each document sum(1/(k + rank)) over the lists
# it appears in, so any two documents sitting at the same rank in exactly one input
# list each get *bit-identical* scores. Ties are common in fusion (unlike raw dense
# scores, which are floats that effectively never collide) and Qdrant breaks them by
# segment-merge order, which is not stable across identical requests.
#
# Because the server truncated at exactly the requested limit, a tie straddling that
# boundary dropped a different document on each call: measured over 40 queries,
# back-to-back identical hybrid requests returned a different result *set* for 11 and
# a different rank-1 document for 8. That is also the source of the ~2pp run-to-run
# drift seen in the known-item benchmark.
#
# Fix: ask for a margin beyond the window we intend to return, then re-sort with an
# explicit tie-break and truncate client-side. The prefetch limits are deliberately
# left keyed to fetch_k so the fused ranking itself is unchanged — we are only
# declining to let the server make the cut for us.
TIE_BREAK_MARGIN = 20


class QdrantSearch:
    """Qdrant-backed hybrid search using dense, sparse, and RRF fusion."""

    def __init__(
        self,
        url: str = DEFAULT_QDRANT_URL,
        collection: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL_NAME,
        dim: int = DEFAULT_DIM,
    ):
        self.url = os.getenv("QDRANT_URL", url)
        self.collection = os.getenv("QDRANT_COLLECTION", collection)
        self.dim = dim

        print(f"Connecting to Qdrant at {self.url}...")
        if self.url.startswith("https://"):
            self.client = QdrantClient(url=self.url, port=443, https=True, timeout=60)
        elif "localhost" in self.url or "127.0.0.1" in self.url:
            self.client = QdrantClient(url=self.url, timeout=60)
        else:
            # ACA internal: ingress routes port 80 → targetPort 6333
            from urllib.parse import urlparse
            parsed = urlparse(self.url)
            port = parsed.port or 80
            host = parsed.hostname
            self.client = QdrantClient(host=host, port=port, timeout=60)

        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        print("Model loaded.")

        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                f"Missing TF-IDF vectorizer at {VECTORIZER_PATH}. "
                "Run `python -m src.indexing.load` first."
            )
        with VECTORIZER_PATH.open("rb") as f:
            self.vectorizer = pickle.load(f)

        self._reranker = _UNSET  # lazy — constructed on first access via .reranker
        self._reranker_lock = threading.Lock()

    @property
    def reranker(self):
        """Lazily load the reranker on first access (gracefully returns None)."""
        if self._reranker is not _UNSET:
            return self._reranker
        with self._reranker_lock:
            if self._reranker is _UNSET:
                try:
                    from src.reranker.onnx_reranker import OnnxReranker
                    self._reranker = OnnxReranker()
                except Exception as e:
                    print(f"Reranker not available: {e}")
                    self._reranker = None
        return self._reranker

    @property
    def reranker_state(self) -> str:
        """Report reranker status without triggering a load."""
        if self._reranker is _UNSET:
            return "not_loaded"
        return "unavailable" if self._reranker is None else "loaded"

    def embed_query(self, query: str) -> list[float]:
        """Embed a query using Nomic's search_query prefix convention."""
        prefixed = f"search_query: {query}"
        vec = self.model.encode([prefixed], normalize_embeddings=False)[0]
        vec = np.asarray(vec[: self.dim], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def _build_sparse_query(self, query: str) -> models.SparseVector:
        row = self.vectorizer.transform([query]).tocsr()[0]
        return models.SparseVector(
            indices=row.indices.tolist(),
            values=row.data.astype(float).tolist(),
        )

    def _build_filter(
        self,
        year_min: int | None = None,
        year_max: int | None = None,
        tier: int | None = None,
    ) -> models.Filter | None:
        conditions: list[models.FieldCondition] = []

        if year_min is not None or year_max is not None:
            range_kwargs = {}
            if year_min is not None:
                range_kwargs["gte"] = year_min
            if year_max is not None:
                range_kwargs["lte"] = year_max
            conditions.append(
                models.FieldCondition(
                    key="year",
                    range=models.Range(**range_kwargs),
                )
            )

        if tier is not None:
            conditions.append(
                models.FieldCondition(
                    key="tier",
                    match=models.MatchValue(value=tier),
                )
            )

        return models.Filter(must=conditions) if conditions else None

    @staticmethod
    def _format_point(point: models.ScoredPoint) -> dict:
        payload = point.payload or {}
        return {
            "id": str(payload.get("id", point.id)),
            "title": payload.get("title", ""),
            "authors": payload.get("authors", ""),
            "description": payload.get("description") or "",
            "subjects": payload.get("subjects", []),
            "year": payload.get("year"),
            "cover_url": payload.get("cover_url"),
            "work_id": payload.get("work_id"),
            "tier": payload.get("tier"),
            "score": float(point.score or 0.0),
        }

    @staticmethod
    def _stable_rank(results: list[dict]) -> list[dict]:
        """Order by score with a deterministic tie-break so equal scores never reshuffle."""
        return sorted(
            results,
            key=lambda r: (-r["score"], str(r.get("work_id") or r.get("id") or "")),
        )

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
        # Over-fetch when reranking to give the reranker more candidates
        fetch_k = rerank_fetch_k(top_k) if rerank and self.reranker else top_k
        # Extra headroom so the ranked page we return is not cut on a score tie.
        window = fetch_k + TIE_BREAK_MARGIN
        query_filter = self._build_filter(year_min=year_min, year_max=year_max, tier=tier)
        dense_vector = self.embed_query(query) if mode in {"vector", "hybrid"} else None
        sparse_vector = self._build_sparse_query(query) if mode in {"keyword", "hybrid"} else None

        start = time.perf_counter()
        if mode == "vector":
            response = self.client.query_points(
                collection_name=self.collection,
                query=dense_vector,
                using="dense",
                query_filter=query_filter,
                with_payload=True,
                limit=window,
            )
        elif mode == "keyword":
            response = self.client.query_points(
                collection_name=self.collection,
                query=sparse_vector,
                using="sparse",
                query_filter=query_filter,
                with_payload=True,
                limit=window,
            )
        elif mode == "hybrid":
            prefetch_limit = max(fetch_k * 2, fetch_k)
            response = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(query=dense_vector, using="dense", limit=prefetch_limit),
                    models.Prefetch(query=sparse_vector, using="sparse", limit=prefetch_limit),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                query_filter=query_filter,
                with_payload=True,
                limit=window,
            )
        else:
            raise ValueError(f"Unsupported search mode: {mode}")

        retrieval_latency_ms = round((time.perf_counter() - start) * 1000, 1)
        points = getattr(response, "points", response)
        results = [self._format_point(point) for point in points]
        # Make the cut ourselves, deterministically, rather than trusting the
        # server's tie ordering (see TIE_BREAK_MARGIN).
        results = self._stable_rank(results)[:fetch_k]

        # Rerank if requested and reranker is available
        reranked = False
        rerank_latency_ms = 0
        if rerank and self.reranker and results:
            rerank_result = self.reranker.rerank(query=query, candidates=results, top_k=top_k)
            results = rerank_result["results"]
            rerank_latency_ms = rerank_result["latency_ms"]
            reranked = True
        else:
            results = results[:top_k]

        total_latency = retrieval_latency_ms + rerank_latency_ms
        return {
            "query": query,
            "results": results,
            "total": len(results),
            "total_count": len(results),
            "mode": mode,
            "reranked": reranked,
            "latency_ms": round(total_latency, 1),
            "retrieval_latency_ms": retrieval_latency_ms,
            "rerank_latency_ms": rerank_latency_ms,
        }

    def compare(self, query: str, top_k: int = 5) -> dict:
        """Run the same query in all supported modes for comparison."""
        return {
            mode: self.search(query=query, top_k=top_k, mode=mode)
            for mode in ("keyword", "vector", "hybrid")
        }
