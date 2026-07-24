"""Hybrid search against Azure AI Search (BM25 + vector + RRF)."""

import os
import time
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
API_KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]
INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX", "books-v1")
API_VERSION = "2024-07-01"

HEADERS = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}


class HybridSearchEngine:
    """Azure AI Search hybrid engine: BM25 + vector + RRF fusion."""

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", dim: int = 384):
        self.dim = dim
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        print("Model loaded.")

    def embed_query(self, query: str) -> list[float]:
        """Embed a query using Nomic prefix convention."""
        prefixed = f"search_query: {query}"
        vec = self.model.encode([prefixed], normalize_embeddings=False)[0]
        # Matryoshka: truncate to target dim, re-normalize
        vec = vec[: self.dim]
        vec = vec / np.linalg.norm(vec)
        return vec.tolist()

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        year_min: int | None = None,
        year_max: int | None = None,
        tier: int | None = None,
    ) -> dict:
        """
        Execute search against Azure AI Search.

        mode: "hybrid" (BM25 + vector + RRF), "vector" (vector only), "keyword" (BM25 only)
        """
        url = f"{ENDPOINT}/indexes/{INDEX_NAME}/docs/search?api-version={API_VERSION}"

        # Build filter expression
        filters = []
        if year_min:
            filters.append(f"year ge {year_min}")
        if year_max:
            filters.append(f"year le {year_max}")
        if tier:
            filters.append(f"tier eq {tier}")
        filter_expr = " and ".join(filters) if filters else None

        # Build request body based on search mode
        body = {
            "top": top_k,
            "select": "id,title,authors,description,subjects,year,cover_url,work_id,tier",
            "count": True,
        }

        if filter_expr:
            body["filter"] = filter_expr

        if mode in ("hybrid", "keyword"):
            body["search"] = query
            body["queryType"] = "simple"

        if mode in ("hybrid", "vector"):
            vector = self.embed_query(query)
            body["vectorQueries"] = [
                {
                    "kind": "vector",
                    "vector": vector,
                    "fields": "embedding",
                    "k": top_k,
                    "exhaustive": True,
                }
            ]

        start = time.time()
        resp = requests.post(url, headers=HEADERS, json=body)
        latency_ms = (time.time() - start) * 1000

        if resp.status_code != 200:
            return {
                "error": f"Search failed: {resp.status_code}",
                "detail": resp.text[:300],
            }

        data = resp.json()
        results = []
        for doc in data.get("value", []):
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "authors": doc.get("authors", ""),
                "description": doc.get("description") or "",
                "subjects": doc.get("subjects", []),
                "year": doc.get("year"),
                "cover_url": doc.get("cover_url"),
                "work_id": doc.get("work_id"),
                "tier": doc.get("tier"),
                "score": doc.get("@search.score", 0),
            })

        return {
            "query": query,
            "mode": mode,
            "total_count": data.get("@odata.count", len(results)),
            "results": results,
            "latency_ms": round(latency_ms, 1),
        }

    def compare(self, query: str, top_k: int = 5) -> dict:
        """Run the same query in all 3 modes for comparison."""
        results = {}
        for mode in ["keyword", "vector", "hybrid"]:
            results[mode] = self.search(query, top_k=top_k, mode=mode)
        return results


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "romance set in Scotland"

    engine = HybridSearchEngine()

    print(f"\n{'='*60}")
    print(f"Query: \"{query}\"")
    print(f"{'='*60}")

    for mode in ["keyword", "vector", "hybrid"]:
        result = engine.search(query, top_k=5, mode=mode)
        print(f"\n--- {mode.upper()} ({result['latency_ms']}ms) ---")
        for r in result["results"]:
            year_str = f" ({r['year']})" if r.get("year") else ""
            print(f"  {r['score']:.4f} | {r['title']}{year_str}")
            if r["authors"]:
                print(f"           {r['authors']}")
