from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

load_dotenv()

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "books"
DEFAULT_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_DIM = 256
INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "index"
VECTORIZER_PATH = INDEX_DIR / "tfidf_vectorizer.pkl"


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
                "Run `python -m src.qdrant.migrate` first."
            )
        with VECTORIZER_PATH.open("rb") as f:
            self.vectorizer = pickle.load(f)

        # Load reranker (optional — gracefully skip if ONNX model missing)
        self._reranker = None
        try:
            from src.reranker.onnx_reranker import OnnxReranker
            self._reranker = OnnxReranker()
        except Exception as e:
            print(f"Reranker not available: {e}")

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
        fetch_k = top_k * 5 if rerank and self._reranker else top_k
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
                limit=fetch_k,
            )
        elif mode == "keyword":
            response = self.client.query_points(
                collection_name=self.collection,
                query=sparse_vector,
                using="sparse",
                query_filter=query_filter,
                with_payload=True,
                limit=fetch_k,
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
                limit=fetch_k,
            )
        else:
            raise ValueError(f"Unsupported search mode: {mode}")

        retrieval_latency_ms = round((time.perf_counter() - start) * 1000, 1)
        points = getattr(response, "points", response)
        results = [self._format_point(point) for point in points]

        # Rerank if requested and reranker is available
        reranked = False
        rerank_latency_ms = 0
        if rerank and self._reranker and results:
            rerank_result = self._reranker.rerank(query=query, candidates=results, top_k=top_k)
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
        }

    def compare(self, query: str, top_k: int = 5) -> dict:
        """Run the same query in all supported modes for comparison."""
        return {
            mode: self.search(query=query, top_k=top_k, mode=mode)
            for mode in ("keyword", "vector", "hybrid")
        }
