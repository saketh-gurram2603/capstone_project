"""
Integration tests for the Triage API.

The LangGraph graph is mocked at the app.state level so no real LLM or DB
calls are made.  Tests verify:
  - Happy path L1 resolution
  - L2 escalation path
  - L3 escalation path with ticket_id
  - Validation errors (short description)
  - 503 when graph not initialised
  - GET /escalations endpoint
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_l1_state(confidence: float = 0.92) -> dict:
    return {
        "query": "Database connection pool exhausted",
        "impact": "High",
        "urgency": "High",
        "retrieved_incidents": [],
        "resolution_options": [],
        "l1_summary": "Increase connection pool size to resolve the exhaustion.",
        "l1_confidence": confidence,
        "l2_synthesis": "",
        "l2_confidence": 0.0,
        "escalation_reason": None,
        "escalation_ticket_id": None,
        "final_answer": "Increase connection pool size to resolve the exhaustion.",
        "escalation_level": "L1",
        "priority": "P1",
        "model_used": "gpt-4o-mini",
        "fallback_used": False,
    }


def _make_l2_state(l2_conf: float = 0.72) -> dict:
    state = _make_l1_state(confidence=0.50)
    state.update({
        "escalation_level": "L2",
        "l2_confidence": l2_conf,
        "l2_synthesis": "Add read replica and increase pool size.",
        "final_answer": "Add read replica and increase pool size.",
        "model_used": "gpt-4o",
        "web_search_results": [{"title": "DB pool", "url": "https://example.com", "content": "..."}],
    })
    return state


def _make_l3_state() -> dict:
    state = _make_l1_state(confidence=0.30)
    state.update({
        "escalation_level": "L3",
        "l2_confidence": 0.30,
        "escalation_ticket_id": "TKT-ABCD1234",
        "escalation_reason": "L2 confidence 0.30 below threshold 0.55.",
        "final_answer": "Escalated to specialist team. Ticket ID: TKT-ABCD1234",
        "model_used": "none",
    })
    return state


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def triage_client():
    """
    TestClient with triage_graph mocked on app.state.
    Patches run_triage so no real LangGraph execution occurs.
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from main import app

    client = TestClient(app, raise_server_exceptions=False)
    return client


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestTriageEndpoint:

    def test_l1_resolution_happy_path(self, sample_triage_payload):
        """L1 resolves with high confidence → escalation_level=L1."""
        from main import app
        mock_graph = MagicMock()

        with patch("src.api.triage.run_triage", new=AsyncMock(return_value=_make_l1_state())) as mock_run:
            app.state.triage_graph = mock_graph
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/api/v1/triage", json=sample_triage_payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["escalation_level"] == "L1"
        assert data["confidence"] == pytest.approx(0.92, abs=0.01)
        assert data["priority"] == "P1"
        assert data["fallback_used"] is False
        assert "latency_ms" in data

    def test_l2_escalation_path(self, sample_triage_payload):
        """L1 fails → L2 resolves → escalation_level=L2."""
        from main import app
        mock_graph = MagicMock()

        with patch("src.api.triage.run_triage", new=AsyncMock(return_value=_make_l2_state())):
            app.state.triage_graph = mock_graph
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/api/v1/triage", json=sample_triage_payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["escalation_level"] == "L2"
        assert data["l2_synthesis"] is not None
        assert data["model_used"] == "gpt-4o"

    def test_l3_escalation_path(self, sample_triage_payload):
        """L1 + L2 both fail → L3 creates ticket."""
        from main import app
        mock_graph = MagicMock()

        with patch("src.api.triage.run_triage", new=AsyncMock(return_value=_make_l3_state())):
            app.state.triage_graph = mock_graph
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/api/v1/triage", json=sample_triage_payload)

        assert resp.status_code == 200
        data = resp.json()
        assert data["escalation_level"] == "L3"
        assert data["escalation_ticket_id"] == "TKT-ABCD1234"
        assert data["escalation_reason"] is not None

    def test_description_too_short_returns_422(self):
        """Description shorter than min_length=10 must be rejected."""
        from main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/triage", json={"description": "short"})
        assert resp.status_code == 422

    def test_missing_description_returns_422(self):
        """Missing required description field must return 422."""
        from main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/api/v1/triage", json={"impact": "High"})
        assert resp.status_code == 422

    def test_graph_not_initialised_returns_503(self):
        """If triage_graph is None on app.state → 503."""
        from main import app
        original = getattr(app.state, "triage_graph", None)
        app.state.triage_graph = None
        try:
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/triage",
                json={
                    "description": "This is a valid description that is long enough",
                    "impact": "High",
                },
            )
            assert resp.status_code == 503
        finally:
            app.state.triage_graph = original

    def test_impact_urgency_optional(self):
        """Impact and urgency are optional — endpoint must accept payload without them."""
        from main import app
        mock_graph = MagicMock()

        with patch("src.api.triage.run_triage", new=AsyncMock(return_value=_make_l1_state())):
            app.state.triage_graph = mock_graph
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/api/v1/triage",
                json={"description": "Service keeps crashing with OOM error every hour"},
            )

        assert resp.status_code == 200

    def test_invalid_impact_enum_returns_422(self):
        """Impact must be one of the allowed enum values."""
        from main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post(
            "/api/v1/triage",
            json={
                "description": "Service keeps crashing with OOM error every hour",
                "impact": "CRITICAL",  # not in enum
            },
        )
        assert resp.status_code == 422

    def test_response_contains_latency_ms(self, sample_triage_payload):
        """Response must include latency_ms field."""
        from main import app
        mock_graph = MagicMock()

        with patch("src.api.triage.run_triage", new=AsyncMock(return_value=_make_l1_state())):
            app.state.triage_graph = mock_graph
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/api/v1/triage", json=sample_triage_payload)

        assert resp.status_code == 200
        assert resp.json()["latency_ms"] >= 0


class TestEscalationsEndpoint:

    def test_get_escalations_returns_list(self):
        """GET /escalations returns a valid EscalationListResponse."""
        from main import app

        mock_tickets = [
            {
                "ticket_id": "TKT-ABCD1234",
                "description": "OOM crash in payment service",
                "impact": "High",
                "urgency": "High",
                "l1_summary": "Memory leak detected",
                "l2_analysis": "Heap dump shows large object retention",
                "escalation_reason": "L2 confidence too low",
                "status": "OPEN",
                "created_at": "2026-05-22T10:00:00+00:00",
            }
        ]

        with patch(
            "src.api.triage.list_escalation_tickets",
            new=AsyncMock(return_value=mock_tickets),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/escalations")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["tickets"][0]["ticket_id"] == "TKT-ABCD1234"

    def test_get_escalations_empty(self):
        """GET /escalations with no tickets returns total=0."""
        from main import app

        with patch(
            "src.api.triage.list_escalation_tickets",
            new=AsyncMock(return_value=[]),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/escalations")

        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_invalid_status_filter_returns_422(self):
        """Status filter must match OPEN|IN_PROGRESS|RESOLVED pattern."""
        from main import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/v1/escalations?status=PENDING")
        assert resp.status_code == 422

    def test_pagination_params(self):
        """limit and offset parameters are accepted."""
        from main import app

        with patch(
            "src.api.triage.list_escalation_tickets",
            new=AsyncMock(return_value=[]),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/v1/escalations?limit=10&offset=5")

        assert resp.status_code == 200
