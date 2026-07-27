"""Smoke tests for the API — validates endpoints load and respond.

Tests that need a populated index are skipped when no Qdrant is reachable, so
the suite stays green in CI and on a laptop without local infra. Point them at
a real instance with QDRANT_URL to actually exercise retrieval.
"""
import os
import pytest
from fastapi.testclient import TestClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def _qdrant_reachable() -> bool:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{QDRANT_URL}/collections", timeout=2) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


needs_qdrant = pytest.mark.skipif(
    not _qdrant_reachable(),
    reason=f"No Qdrant reachable at {QDRANT_URL}; set QDRANT_URL to run retrieval tests",
)


@pytest.fixture
def client():
    os.environ.setdefault("QDRANT_URL", QDRANT_URL)
    from src.api.main import app
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


@needs_qdrant
def test_search_returns_results(client):
    resp = client.get("/search?q=python+programming&top_k=3&understand=false")
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert data["query"] == "python programming"


@needs_qdrant
def test_search_modes(client):
    for mode in ["hybrid", "keyword", "vector"]:
        resp = client.get(f"/search?q=cooking&mode={mode}&top_k=3&understand=false")
        assert resp.status_code == 200
        assert resp.json()["mode"] == mode


@needs_qdrant
def test_stats(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "backend" in data
