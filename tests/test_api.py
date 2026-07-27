"""Smoke tests for the API — validates endpoints load and respond."""
import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client. Skips if Qdrant is not available."""
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
    from src.api.main import app
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_search_returns_results(client):
    resp = client.get("/search?q=python+programming&top_k=3&understand=false")
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert data["query"] == "python programming"


def test_search_modes(client):
    for mode in ["hybrid", "keyword", "vector"]:
        resp = client.get(f"/search?q=cooking&mode={mode}&top_k=3&understand=false")
        assert resp.status_code == 200
        assert resp.json()["mode"] == mode


def test_stats(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "backend" in data
