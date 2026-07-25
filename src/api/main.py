"""
FastAPI search API — hybrid search over Azure AI Search.
Supports BM25, vector, and hybrid (RRF) retrieval modes.
Falls back to local FAISS if Azure credentials are not configured.
"""

import os
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Book Search API",
    description="Hybrid semantic search over the OpenLibrary catalog",
    version="0.3.0",
)


class SearchMode(str, Enum):
    hybrid = "hybrid"
    vector = "vector"
    keyword = "keyword"


# Lazy-load search engine, reranker, and query pipeline
_engine = None
_reranker = None
_query_pipeline = None


def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    # Use Azure AI Search if configured, else fall back to local FAISS
    if os.environ.get("AZURE_SEARCH_ENDPOINT") and os.environ.get("AZURE_SEARCH_ADMIN_KEY"):
        from src.azure_search.search import HybridSearchEngine
        _engine = HybridSearchEngine()
    else:
        from src.search.engine import BookSearchEngine
        _engine = BookSearchEngine(Path("data/index"))

    return _engine


def get_reranker():
    global _reranker
    if _reranker is None:
        from src.reranker.onnx_reranker import OnnxReranker
        _reranker = OnnxReranker()
    return _reranker


def get_query_pipeline():
    global _query_pipeline
    if _query_pipeline is None:
        from src.query.pipeline import QueryPipeline
        _query_pipeline = QueryPipeline()
    return _query_pipeline


@app.get("/search")
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results"),
    mode: SearchMode = Query(SearchMode.hybrid, description="Retrieval mode"),
    rerank: bool = Query(False, description="Apply cross-encoder reranking"),
    tier: int | None = Query(None, ge=1, le=3, description="Tier filter"),
    year_min: int | None = Query(None, description="Min publication year"),
    year_max: int | None = Query(None, description="Max publication year"),
    explain: bool = Query(False, description="Include search metadata"),
    understand: bool = Query(True, description="Apply query understanding (spell + intent)"),
):
    """Search for books. Supports hybrid (BM25+vector+RRF), vector-only, or keyword-only.
    Query understanding (spell correction + intent classification) is on by default."""
    engine = get_engine()

    # Query understanding + adaptive routing
    query_info = None
    search_query = q
    search_mode = mode.value
    mode_was_default = (mode == SearchMode.hybrid)  # user didn't explicitly override

    if understand:
        qp = get_query_pipeline()
        analysis = qp.process(q)
        search_query = analysis.corrected

        # Adaptive routing: let intent override mode unless user explicitly set it
        if mode_was_default:
            search_mode = analysis.search_mode  # keyword, vector, or hybrid

        # Apply detected filters (only if user didn't explicitly set them)
        if analysis.filters.get("year_min") and not year_min:
            year_min = analysis.filters["year_min"]
        if analysis.filters.get("year_max") and not year_max:
            year_max = analysis.filters["year_max"]

        # For "similar_to" intent, search by title with context for embedding
        if analysis.intent.value == "similar_to" and analysis.extracted_title:
            search_query = f"novel titled {analysis.extracted_title}"

        query_info = {
            "original": analysis.original,
            "corrected": analysis.corrected,
            "was_corrected": analysis.was_corrected,
            "intent": analysis.intent.value,
            "confidence": analysis.confidence,
            "routed_mode": search_mode,
            "extracted_title": analysis.extracted_title or None,
            "filters_applied": analysis.filters or None,
        }

    # Fetch more candidates if reranking (need a larger pool to reorder)
    retrieve_k = top_k * 5 if rerank else top_k

    # Azure AI Search backend
    if hasattr(engine, "search") and hasattr(engine, "compare"):
        result = engine.search(
            query=search_query,
            top_k=retrieve_k,
            mode=search_mode,
            year_min=year_min,
            year_max=year_max,
            tier=tier,
        )

        if "error" in result:
            return JSONResponse(content=result, status_code=502)

        retrieval_latency = result["latency_ms"]
        results = result["results"]

        # Apply reranking if requested
        rerank_latency = 0
        if rerank and results:
            reranker = get_reranker()
            rerank_result = reranker.rerank(query=q, candidates=results, top_k=top_k)
            results = rerank_result["results"]
            rerank_latency = rerank_result["latency_ms"]

        response = {
            "query": q,
            "mode": search_mode,  # actual mode used (may differ from requested if routed)
            "reranked": rerank,
            "total_results": result["total_count"],
            "latency_ms": round(retrieval_latency + rerank_latency, 1),
            "retrieval_latency_ms": retrieval_latency,
            "results": results[:top_k],
        }

        if query_info:
            response["query_understanding"] = query_info

        if rerank:
            response["rerank_latency_ms"] = rerank_latency
            response["candidates_reranked"] = len(result["results"])

        if explain:
            response["explain"] = {
                "backend": "azure_ai_search",
                "index": os.environ.get("AZURE_SEARCH_INDEX", "books-v1"),
                "model": "nomic-ai/nomic-embed-text-v1.5",
                "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2" if rerank else None,
                "dimension": engine.dim,
                "requested_mode": mode.value,
                "routed_mode": search_mode,
                "retrieval": f"{search_mode} (BM25+vector+RRF)" if search_mode == "hybrid" else search_mode,
                "pipeline": "understand → retrieve → rerank → top_k" if rerank else "understand → retrieve → top_k",
            }

        return JSONResponse(content=response)

    # FAISS fallback (local mode)
    results = engine.search(
        query=q,
        top_k=top_k,
        tier_filter=tier,
        year_min=year_min,
        year_max=year_max,
    )
    return JSONResponse(content={
        "query": q,
        "mode": "vector",
        "total_results": len(results),
        "results": results,
    })


@app.get("/search/compare")
def compare(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=20, description="Results per mode"),
):
    """Compare results across all retrieval modes (keyword, vector, hybrid)."""
    engine = get_engine()

    if not hasattr(engine, "compare"):
        return JSONResponse(
            content={"error": "Compare only available with Azure AI Search backend"},
            status_code=501,
        )

    results = engine.compare(query=q, top_k=top_k)

    # Summarize for easy reading
    summary = {}
    for mode, data in results.items():
        summary[mode] = {
            "latency_ms": data["latency_ms"],
            "results": [
                {"title": r["title"], "score": r["score"], "year": r.get("year")}
                for r in data["results"]
            ],
        }

    return JSONResponse(content={"query": q, "modes": summary})


@app.get("/health")
def health():
    engine_type = "not_loaded"
    if _engine is not None:
        engine_type = "azure_ai_search" if hasattr(_engine, "compare") else "faiss_local"
    return {"status": "ok", "backend": engine_type}


@app.get("/stats")
def stats():
    engine = get_engine()
    if hasattr(engine, "compare"):
        # Azure backend
        from src.azure_search.index import get_index_stats
        idx_stats = get_index_stats()
        return {
            "backend": "azure_ai_search",
            "index": os.environ.get("AZURE_SEARCH_INDEX", "books-v1"),
            "documents": idx_stats["documentCount"] if idx_stats else "unknown",
            "storage_mb": round(idx_stats["storageSize"] / 1024 / 1024, 1) if idx_stats else "unknown",
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "dimension": engine.dim,
        }
    else:
        return {
            "backend": "faiss_local",
            "index_size": engine.index.ntotal,
            "dimension": engine.dim,
            "model": "nomic-ai/nomic-embed-text-v1.5",
        }


# --- RAG /ask endpoint ---

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500, description="Natural language question about books")
    max_sources: int = Field(5, ge=1, le=10, description="Max books to use as context")
    mode: SearchMode = Field(SearchMode.hybrid, description="Retrieval mode for finding sources")


_rag = None


def get_rag():
    global _rag
    if _rag is None:
        from src.rag.generate import RAGPipeline
        _rag = RAGPipeline()
    return _rag


@app.post("/ask")
def ask(req: AskRequest):
    """Ask a question and get a grounded answer with book citations.

    Uses hybrid search to find relevant books, then generates a natural language
    answer citing those sources. Hallucination guardrails validate all mentioned titles."""
    import time

    t0 = time.time()

    # Step 1: Retrieve relevant books
    engine = get_engine()
    if hasattr(engine, "search") and hasattr(engine, "compare"):
        result = engine.search(query=req.question, top_k=req.max_sources * 2, mode=req.mode.value)
        if "error" in result:
            return JSONResponse(content={"error": "Search failed", "detail": result["error"]}, status_code=502)
        search_results = result["results"]
        retrieval_ms = result["latency_ms"]
    else:
        search_results = engine.search(query=req.question, top_k=req.max_sources * 2)
        retrieval_ms = 0

    if not search_results:
        return JSONResponse(content={
            "answer": "I couldn't find any relevant books for your question.",
            "sources": [],
            "latency_ms": {"retrieval": retrieval_ms, "total": retrieval_ms},
        })

    # Step 2: RAG generation
    try:
        rag = get_rag()
        response = rag.ask(
            question=req.question,
            search_results=search_results,
            max_sources=req.max_sources,
        )
    except Exception as e:
        return JSONResponse(
            content={"error": "Generation failed", "detail": str(e)},
            status_code=502,
        )

    total_ms = (time.time() - t0) * 1000

    return JSONResponse(content={
        "question": req.question,
        "answer": response.answer,
        "sources": response.sources,
        "citations_valid": response.citations_valid,
        "hallucinated_titles": response.hallucinated_titles if not response.citations_valid else [],
        "latency_ms": {
            "retrieval": round(retrieval_ms, 1),
            "generation": response.latency_ms["generation"],
            "total": round(total_ms, 1),
        },
        "model": response.model,
        "token_usage": response.token_usage,
    })
