"""
FastAPI search API — hybrid search over book catalog.
Supports keyword (TF-IDF sparse), vector, and hybrid (RRF) retrieval modes.
Backend: Qdrant (required).
"""

import os
import threading
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.environ.get("DISABLE_WARMUP") and os.environ.get("QDRANT_URL"):
        threading.Thread(target=_warmup, name="warmup", daemon=True).start()
    yield


app = FastAPI(
    title="Book Search API",
    description="Hybrid semantic search over the OpenLibrary catalog",
    version="0.5.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://black-grass-0df1c7a0f.7.azurestaticapps.net",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchMode(str, Enum):
    hybrid = "hybrid"
    vector = "vector"
    keyword = "keyword"


# Lazy-load search engine and query pipeline.
# Locks matter: this container scales to zero, so a burst of requests can arrive
# while nothing is loaded. Without them each concurrent request builds its own
# SentenceTransformer, which spikes memory and can get the container OOM-killed.
_engine = None
_query_pipeline = None
_engine_lock = threading.Lock()
_query_pipeline_lock = threading.Lock()
_warmup_state = {"status": "cold", "error": None}


def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    qdrant_url = os.environ.get("QDRANT_URL")
    if not qdrant_url:
        raise RuntimeError(
            "QDRANT_URL environment variable is required but not set. "
            "Set it to your Qdrant instance URL (e.g., http://localhost:6333)."
        )

    with _engine_lock:
        if _engine is None:
            from src.qdrant.client import QdrantSearch
            _engine = QdrantSearch(
                url=qdrant_url,
                collection=os.environ.get("QDRANT_COLLECTION", "books"),
            )
    return _engine


def get_query_pipeline():
    global _query_pipeline
    if _query_pipeline is not None:
        return _query_pipeline
    with _query_pipeline_lock:
        if _query_pipeline is None:
            from src.query.pipeline import QueryPipeline
            _query_pipeline = QueryPipeline()
    return _query_pipeline


def _warmup() -> None:
    """Preload models so no user request pays the cold-start cost.

    Runs on a background thread: startup must not block, or the platform health
    probe fails before the container is ever marked ready. The cross-encoder is
    included deliberately — it is the slowest component to load, and the first
    `?rerank=true` request used to stall long enough to return a 500.
    """
    try:
        _warmup_state["status"] = "warming"
        engine = get_engine()
        engine.search(query="warmup", top_k=1, mode="hybrid")
        get_query_pipeline()
        if engine.reranker is not None:
            engine.search(query="warmup", top_k=1, mode="hybrid", rerank=True)
        _warmup_state["status"] = "ready"
        print("Warmup complete: engine, query pipeline, and reranker loaded.")
    except Exception as exc:  # warmup is best-effort; requests still lazy-load
        _warmup_state["status"] = "failed"
        _warmup_state["error"] = str(exc)
        print(f"Warmup failed (requests will lazy-load instead): {exc}")


@app.get("/search")
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results"),
    mode: SearchMode = Query(SearchMode.hybrid, description="Retrieval mode"),
    rerank: bool = Query(False, description="Apply cross-encoder reranking"),
    tier: int | None = Query(None, ge=1, le=3, description="Tier filter"),
    year_min: int | None = Query(None, description="Min publication year (NOTE: 'year' is unpopulated for the entire current index, so this filter matches nothing until the ETL backfills it)"),
    year_max: int | None = Query(None, description="Max publication year (see year_min caveat)"),
    explain: bool = Query(False, description="Include search metadata"),
    understand: bool = Query(True, description="Apply query understanding (spell + intent)"),
):
    """Search for books. Supports hybrid (sparse+vector+RRF), vector-only, or keyword-only.
    Query understanding (spell correction + intent classification) is on by default."""
    engine = get_engine()

    # Query understanding: spell correction + intent metadata (no routing)
    query_info = None
    search_query = q
    search_mode = mode.value

    if understand:
        qp = get_query_pipeline()
        analysis = qp.process(q)
        search_query = analysis.corrected

        query_info = {
            "original": analysis.original,
            "corrected": analysis.corrected,
            "was_corrected": analysis.was_corrected,
            "intent": analysis.intent.value,
            "confidence": analysis.confidence,
        }

    result = engine.search(
        query=search_query,
        top_k=top_k,
        mode=search_mode,
        rerank=rerank,
        tier=tier,
        year_min=year_min,
        year_max=year_max,
    )

    if "error" in result:
        return JSONResponse(content=result, status_code=502)

    retrieval_latency = result["retrieval_latency_ms"]
    results = result["results"]
    rerank_latency = result.get("rerank_latency_ms", 0)

    response = {
        "query": q,
        "mode": mode.value,
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
            "backend": "qdrant",
            "index": engine.collection,
            "model": "nomic-ai/nomic-embed-text-v1.5",
            "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2" if rerank else None,
            "dimension": engine.dim,
            "mode": mode.value,
            "retrieval": f"{mode.value} (TF-IDF+vector+RRF)" if mode == SearchMode.hybrid else mode.value,
            "pipeline": "spell_correct → retrieve → rerank → top_k" if rerank else "spell_correct → retrieve → top_k",
        }

    return JSONResponse(content=response)


@app.get("/search/compare")
def compare(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=20, description="Results per mode"),
):
    """Compare results across all retrieval modes (keyword, vector, hybrid)."""
    engine = get_engine()

    if not hasattr(engine, "compare"):
        return JSONResponse(
            content={"error": "Compare requires Azure AI Search or Qdrant backend"},
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
    # Never raise: this is the platform liveness probe. A 500 here gets a
    # perfectly healthy container restarted.
    try:
        engine = _engine
        return {
            "status": "healthy",
            "backend": "qdrant" if engine is not None else "not_loaded",
            "warmup": _warmup_state["status"],
            "reranker": getattr(engine, "reranker_state", "not_loaded"),
        }
    except Exception:
        return {"status": "healthy", "backend": "unknown", "warmup": "unknown"}


@app.get("/stats")
def stats():
    engine = get_engine()
    info = engine.client.get_collection(engine.collection)
    return {
        "backend": "qdrant",
        "collection": engine.collection,
        "documents": info.points_count,
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "dimension": engine.dim,
        "qdrant_url": engine.url,
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

    # Step 1: Retrieve relevant books (with reranker for better source quality)
    engine = get_engine()
    result = engine.search(query=req.question, top_k=req.max_sources * 2, mode=req.mode.value, rerank=True)
    if "error" in result:
        return JSONResponse(content={"error": "Search failed", "detail": result["error"]}, status_code=502)
    search_results = result["results"]
    retrieval_ms = result["latency_ms"]

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
