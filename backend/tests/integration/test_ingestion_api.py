"""
Integration tests for POST /ingest and GET /ingest/status.
Uses FastAPI TestClient with mocked VectorStore and embed_batch so
no real Qdrant or OpenAI calls are made.
"""

import io
import os
import pytest
import pandas as pd
from unittest.mock import AsyncMock, patch, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.ingestion import router as ingestion_router
from src.core.dependencies import get_app_config, get_vector_store
from src.integrations.vector_db import VectorStore


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _build_test_app(mock_vector_store: VectorStore, app_config: dict) -> FastAPI:
    """Minimal FastAPI app wired with the ingestion router and mock deps."""
    app = FastAPI()
    app.include_router(ingestion_router)
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[get_app_config] = lambda: app_config
    return app


def _make_mock_vector_store() -> VectorStore:
    """Return an AsyncMock that satisfies the VectorStore interface."""
    store = AsyncMock(spec=VectorStore)
    store.upsert.return_value = 5
    return store


def _make_xlsx_bytes(rows: list[dict]) -> bytes:
    """Build an in-memory XLSX file and return raw bytes."""
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def _valid_rows(n: int = 5) -> list[dict]:
    return [
        {
            "Incident ID": f"INC-{1000 + i}",
            "Ticket ID": f"TKT-{2000 + i}",
            "Media Asset": f"MediaServer{i:02d}",
            "Category": "Storage",
            "Incident Details": f"Test Incident {i}",
            "Description": f"Test description for incident {i} with enough detail",
            "Solution": f"Test resolution for incident {i} — restart and reconfigure",
        }
        for i in range(n)
    ]


APP_CONFIG = {
    "QDRANT": {"COLLECTION_NAME": "incidents"},
}


# ── Tests: POST /ingest ───────────────────────────────────────────────────────


class TestIngestEndpoint:

    @patch("src.ingestion.pipeline.embed_batch", new_callable=AsyncMock)
    def test_ingest_valid_xlsx_returns_200(self, mock_embed):
        """Valid XLSX upload should succeed and return ingested count."""
        mock_embed.return_value = [[0.1] * 384] * 5  # 5 dummy vectors
        store = _make_mock_vector_store()
        client = TestClient(_build_test_app(store, APP_CONFIG))

        xlsx_bytes = _make_xlsx_bytes(_valid_rows(5))
        response = client.post(
            "/ingest",
            files={"file": ("incidents.xlsx", xlsx_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"
        assert body["ingested"] == 5
        assert body["skipped"] == 0

    @patch("src.ingestion.pipeline.embed_batch", new_callable=AsyncMock)
    def test_ingest_rejects_csv(self, mock_embed):
        """Non-XLSX file should be rejected with 422."""
        store = _make_mock_vector_store()
        client = TestClient(_build_test_app(store, APP_CONFIG))

        response = client.post(
            "/ingest",
            files={"file": ("data.csv", b"a,b,c\n1,2,3", "text/csv")},
        )
        assert response.status_code == 422

    @patch("src.ingestion.pipeline.embed_batch", new_callable=AsyncMock)
    def test_ingest_calls_vector_store_upsert(self, mock_embed):
        """Pipeline should call vector_store.upsert at least once."""
        mock_embed.return_value = [[0.0] * 384] * 3
        store = _make_mock_vector_store()
        client = TestClient(_build_test_app(store, APP_CONFIG))

        xlsx_bytes = _make_xlsx_bytes(_valid_rows(3))
        client.post(
            "/ingest",
            files={"file": ("test.xlsx", xlsx_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
        store.upsert.assert_called()

    @patch("src.ingestion.pipeline.embed_batch", new_callable=AsyncMock)
    def test_ingest_duration_ms_present(self, mock_embed):
        """Response should include a positive duration_ms."""
        mock_embed.return_value = [[0.1] * 384] * 2
        store = _make_mock_vector_store()
        client = TestClient(_build_test_app(store, APP_CONFIG))

        xlsx_bytes = _make_xlsx_bytes(_valid_rows(2))
        response = client.post(
            "/ingest",
            files={"file": ("test.xlsx", xlsx_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
        body = response.json()
        assert body["duration_ms"] >= 0

    @patch("src.ingestion.pipeline.embed_batch", new_callable=AsyncMock)
    def test_ingest_skips_blank_rows(self, mock_embed):
        """Rows with empty description should be counted in skipped."""
        mock_embed.return_value = [[0.1] * 384]
        store = _make_mock_vector_store()
        client = TestClient(_build_test_app(store, APP_CONFIG))

        rows = _valid_rows(1) + [
            {
                "Incident ID": "INC-9999",
                "Ticket ID": "TKT-9999",
                "Media Asset": "MS99",
                "Category": "Network",
                "Incident Details": "Bad row",
                "Description": "",   # ← blank
                "Solution": "Something",
            }
        ]
        xlsx_bytes = _make_xlsx_bytes(rows)
        response = client.post(
            "/ingest",
            files={"file": ("test.xlsx", xlsx_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["ingested"] == 1
        assert body["skipped"] == 1


# ── Tests: GET /ingest/status ─────────────────────────────────────────────────


class TestIngestStatusEndpoint:

    def test_status_returns_200(self):
        store = _make_mock_vector_store()
        client = TestClient(_build_test_app(store, APP_CONFIG))
        response = client.get("/ingest/status")
        assert response.status_code == 200

    def test_status_has_required_fields(self):
        store = _make_mock_vector_store()
        client = TestClient(_build_test_app(store, APP_CONFIG))
        body = client.get("/ingest/status").json()
        assert "status" in body
        assert "ingested" in body
        assert "skipped" in body
        assert "total" in body

    @patch("src.ingestion.pipeline.embed_batch", new_callable=AsyncMock)
    def test_status_completed_after_ingest(self, mock_embed):
        """After a successful ingest, status endpoint should show 'completed'."""
        mock_embed.return_value = [[0.1] * 384] * 2
        store = _make_mock_vector_store()
        client = TestClient(_build_test_app(store, APP_CONFIG))

        xlsx_bytes = _make_xlsx_bytes(_valid_rows(2))
        client.post(
            "/ingest",
            files={"file": ("test.xlsx", xlsx_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
        status_body = client.get("/ingest/status").json()
        assert status_body["status"] == "completed"
        assert status_body["ingested"] == 2
