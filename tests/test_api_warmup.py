"""Cold-start behaviour: concurrent first requests must not load the model twice.

This container scales to zero, so a burst of traffic can arrive with nothing
loaded. Before the locks were added, each concurrent request constructed its own
SentenceTransformer, spiking memory. These tests pin that down.
"""
import threading

import pytest

import src.api.main as main


@pytest.fixture
def reset_engine(monkeypatch):
    monkeypatch.setattr(main, "_engine", None)
    monkeypatch.setattr(main, "_query_pipeline", None)
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    yield
    main._engine = None
    main._query_pipeline = None


def _hammer(fn, n_threads=16):
    """Call *fn* from n threads simultaneously; return collected results."""
    barrier = threading.Barrier(n_threads)
    results = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        value = fn()
        with lock:
            results.append(value)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return results


def test_get_engine_constructs_once_under_concurrency(reset_engine, monkeypatch):
    construction_count = {"n": 0}
    count_lock = threading.Lock()

    class FakeQdrantSearch:
        def __init__(self, **kwargs):
            with count_lock:
                construction_count["n"] += 1
            # Simulate slow model load so threads genuinely overlap
            threading.Event().wait(0.05)

    fake_module = type("m", (), {"QdrantSearch": FakeQdrantSearch})
    monkeypatch.setitem(__import__("sys").modules, "src.qdrant.client", fake_module)

    engines = _hammer(main.get_engine)

    assert construction_count["n"] == 1, (
        f"engine built {construction_count['n']}x under concurrency; lock is not working"
    )
    assert len({id(e) for e in engines}) == 1, "threads received different engine instances"


def test_get_engine_raises_without_qdrant_url(reset_engine, monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    with pytest.raises(RuntimeError, match="QDRANT_URL"):
        main.get_engine()


def test_health_reports_warmup_and_reranker_state(reset_engine):
    from fastapi.testclient import TestClient

    body = TestClient(main.app).get("/health").json()
    assert body["status"] == "healthy"
    assert "warmup" in body
    assert "reranker" in body
    assert body["reranker"] == "not_loaded"


@pytest.mark.parametrize(
    "state,expected_code,expected_ready",
    [
        ("cold", 503, False),
        ("warming", 503, False),
        ("failed", 503, False),
        ("ready", 200, True),
    ],
)
def test_ready_gates_on_warmup(monkeypatch, state, expected_code, expected_ready):
    """Ingress must not route to a replica whose models are still loading."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, "_warmup_state", {"status": state, "error": None})
    resp = TestClient(main.app).get("/ready")
    assert resp.status_code == expected_code
    assert resp.json()["ready"] is expected_ready
    assert resp.json()["warmup"] == state


def test_ready_and_health_disagree_while_warming(monkeypatch):
    """Liveness stays green while readiness is red.

    If /health went unhealthy during warmup the platform would restart the
    container mid-load, looping forever.
    """
    from fastapi.testclient import TestClient

    monkeypatch.setattr(main, "_warmup_state", {"status": "warming", "error": None})
    client = TestClient(main.app)
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503


def test_warmup_failure_does_not_raise(reset_engine, monkeypatch):
    """Warmup is best-effort — a failure must not crash the worker thread."""
    def boom():
        raise RuntimeError("simulated cold-start failure")

    monkeypatch.setattr(main, "get_engine", boom)
    monkeypatch.setattr(main, "_warmup_state", {"status": "cold", "error": None})

    main._warmup()

    assert main._warmup_state["status"] == "failed"
    assert "simulated cold-start failure" in main._warmup_state["error"]


def test_reranker_state_does_not_trigger_load():
    """Reading reranker_state must never construct the cross-encoder."""
    from src.qdrant.client import _UNSET, QdrantSearch

    obj = object.__new__(QdrantSearch)
    obj._reranker = _UNSET
    assert QdrantSearch.reranker_state.fget(obj) == "not_loaded"

    obj._reranker = None
    assert QdrantSearch.reranker_state.fget(obj) == "unavailable"

    obj._reranker = "sentinel"
    assert QdrantSearch.reranker_state.fget(obj) == "loaded"
