"""
Integration tests for POST /search.
Mocks hybrid_search at the API boundary — no real Qdrant, BM25, or OpenAI calls.
"""

import pytest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.search import router as search_router
from src.core.dependencies import get_app_config, get_vector_store
from src.integrations.vector_db import VectorStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

APP_CONFIG = {
    "QDRANT": {"COLLECTION_NAME": "incidents"},
    "RETRIEVAL": {
        "K_MIN": 3, "K_DEFAULT": 10, "K_MAX": 20,
        "RRF_K": 60, "SCORE_DROPOFF_THRESHOLD": 0.15,
        "RESOLUTION_DEDUP_THRESHOLD": 0.95, "TOP_K_FINAL": 10,
    },
    "CACHE": {"QUERY_RESULT_TTL_SECONDS": 3600},
}


def _build_test_app(mock_store=None) -> FastAPI:
    store = mock_store or AsyncMock(spec=VectorStore)
    app = FastAPI()
    app.include_router(search_router)
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_app_config] = lambda: APP_CONFIG
    return app


def _mock_search_result(n_results: int = 3) -> dict:
    return {
        "query": "test query",
        "total_found": n_results,
        "results": [
            {
                "incident_id": f"INC-{i}",
                "ticket_id": f"TKT-{i}",
                "title": f"Test Incident {i}",
                "category": "Storage",
                "description": f"Description {i}",
                "resolution_notes": f"Fix {i}",
                "assigned_to": f"Server{i:02d}",
                "similarity_score": round(0.9 - i * 0.05, 2),
                "occurrence_count": 1,
            }
            for i in range(n_results)
        ],
        "resolution_options": [
            {
                "resolution_text": "Restart the service",
                "occurrence_count": 2,
                "avg_similarity": 0.85,
                "source_incident_ids": ["INC-0", "INC-1"],
            }
        ],
        "adaptive_k_used": 10,
        "retrieval_method": "hybrid",
        "cached": False,
        "latency_ms": 42.5,
    }


# ── Tests: valid requests ─────────────────────────────────────────────────────

class TestSearchEndpoint:

    @patch("src.api.search.hybrid_search", new_callable=AsyncMock)
    def test_valid_query_returns_200(self, mock_search):
        mock_search.return_value = _mock_search_result()
        client = TestClient(_build_test_app())
        resp = client.post("/search", json={"query": "disk space failure"})
        assert resp.status_code == 200

    @patch("src.api.search.hybrid_search", new_callable=AsyncMock)
    def test_response_has_all_fields(self, mock_search):
        mock_search.return_value = _mock_search_result(3)
        client = TestClient(_build_test_app())
        body = client.post("/search", json={"query": "storage issue"}).json()

        assert "results" in body
        assert "resolution_options" in body
        assert "adaptive_k_used" in body
        assert "retrieval_method" in body
        assert "cached" in body
        assert "latency_ms" in body
        assert "total_found" in body

    @patch("src.api.search.hybrid_search", new_callable=AsyncMock)
    def test_results_count_matches(self, mock_search):
        mock_search.return_value = _mock_search_result(5)
        client = TestClient(_build_test_app())
        body = client.post("/search", json={"query": "network latency"}).json()
        assert len(body["results"]) == 5

    @patch("src.api.search.hybrid_search", new_callable=AsyncMock)
    def test_filters_passed_to_pipeline(self, mock_search):
        mock_search.return_value = _mock_search_result(1)
        client = TestClient(_build_test_app())
        client.post("/search", json={
            "query": "database timeout",
            "filters": {"category": "Database"},
        })
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["filters"] == {"category": "Database"}

    @patch("src.api.search.hybrid_search", new_callable=AsyncMock)
    def test_top_k_override_passed(self, mock_search):
        mock_search.return_value = _mock_search_result(2)
        client = TestClient(_build_test_app())
        client.post("/search", json={"query": "memory leak", "top_k": 5})
        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs["top_k_override"] == 5

    @patch("src.api.search.hybrid_search", new_callable=AsyncMock)
    def test_cached_result_flag_preserved(self, mock_search):
        result = _mock_search_result()
        result["cached"] = True
        mock_search.return_value = result
        client = TestClient(_build_test_app())
        body = client.post("/search", json={"query": "disk space"}).json()
        assert body["cached"] is True


# ── Tests: validation errors ──────────────────────────────────────────────────

class TestSearchValidation:

    def test_query_too_short_returns_422(self):
        client = TestClient(_build_test_app())
        resp = client.post("/search", json={"query": "ab"})
        assert resp.status_code == 422

    def test_blank_query_returns_422(self):
        client = TestClient(_build_test_app())
        resp = client.post("/search", json={"query": "   "})
        assert resp.status_code == 422

    def test_top_k_zero_returns_422(self):
        client = TestClient(_build_test_app())
        resp = client.post("/search", json={"query": "valid query", "top_k": 0})
        assert resp.status_code == 422

    def test_top_k_too_large_returns_422(self):
        client = TestClient(_build_test_app())
        resp = client.post("/search", json={"query": "valid query", "top_k": 21})
        assert resp.status_code == 422

    def test_invalid_filter_priority_returns_422(self):
        client = TestClient(_build_test_app())
        resp = client.post("/search", json={
            "query": "valid query",
            "filters": {"priority": "INVALID"},
        })
        assert resp.status_code == 422


# ── Tests: error handling ─────────────────────────────────────────────────────

class TestSearchErrorHandling:

    @patch("src.api.search.hybrid_search", new_callable=AsyncMock)
    def test_retrieval_error_returns_503(self, mock_search):
        from src.exceptions.custom_exceptions import RetrievalError
        mock_search.side_effect = RetrievalError("Both search methods failed.")
        client = TestClient(_build_test_app())
        resp = client.post("/search", json={"query": "disk failure"})
        assert resp.status_code == 503

    @patch("src.api.search.hybrid_search", new_callable=AsyncMock)
    def test_unexpected_error_returns_500(self, mock_search):
        mock_search.side_effect = RuntimeError("unexpected")
        client = TestClient(_build_test_app())
        resp = client.post("/search", json={"query": "disk failure"})
        assert resp.status_code == 500

    @patch("src.api.search.hybrid_search", new_callable=AsyncMock)
    def test_no_results_returns_200_with_empty_list(self, mock_search):
        result = _mock_search_result(0)
        result["resolution_options"] = []
        mock_search.return_value = result
        client = TestClient(_build_test_app())
        resp = client.post("/search", json={"query": "obscure query"})
        assert resp.status_code == 200
        assert resp.json()["results"] == []
