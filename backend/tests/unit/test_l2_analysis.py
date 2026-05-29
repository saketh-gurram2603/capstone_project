"""
Unit tests for L2 analysis helpers.

No real LLM or web calls — all external I/O is mocked.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.l2_analysis import (
    _format_l2_context,
    _parse_l2_response,
    _format_synthesis,
    make_l2_node,
)


# ── _parse_l2_response ─────────────────────────────────────────────────────────


class TestParseL2Response:

    def test_valid_json_parsed_correctly(self):
        payload = {
            "root_cause": "Memory leak in worker threads.",
            "resolution_steps": ["Restart service", "Increase heap"],
            "confidence": 0.85,
            "sources": ["KB", "Web"],
        }
        result = _parse_l2_response(json.dumps(payload))
        assert result["root_cause"] == "Memory leak in worker threads."
        assert result["confidence"] == pytest.approx(0.85)
        assert result["sources"] == ["KB", "Web"]

    def test_markdown_fence_stripped(self):
        payload = {"root_cause": "DB overload", "resolution_steps": [], "confidence": 0.7, "sources": []}
        raw = f"```json\n{json.dumps(payload)}\n```"
        result = _parse_l2_response(raw)
        assert result["confidence"] == pytest.approx(0.7)

    def test_confidence_clamped_to_one(self):
        payload = {"root_cause": "x", "resolution_steps": [], "confidence": 1.5, "sources": []}
        result = _parse_l2_response(json.dumps(payload))
        assert result["confidence"] == 1.0

    def test_confidence_clamped_to_zero(self):
        payload = {"root_cause": "x", "resolution_steps": [], "confidence": -0.5, "sources": []}
        result = _parse_l2_response(json.dumps(payload))
        assert result["confidence"] == 0.0

    def test_invalid_json_falls_back(self):
        result = _parse_l2_response("This is not JSON at all")
        assert "root_cause" in result
        assert 0.0 <= result["confidence"] <= 1.0

    def test_empty_string_falls_back(self):
        result = _parse_l2_response("")
        assert result["confidence"] >= 0.0

    def test_heuristic_confident_keyword_boosts(self):
        """'confident' keyword → heuristic confidence = 0.55."""
        result = _parse_l2_response("I am confident the solution is X")
        assert result["confidence"] == pytest.approx(0.55)

    def test_heuristic_no_keyword_low_confidence(self):
        result = _parse_l2_response("Unclear situation, unsure about root cause")
        assert result["confidence"] == pytest.approx(0.35)


# ── _format_synthesis ─────────────────────────────────────────────────────────


class TestFormatSynthesis:

    def test_output_contains_root_cause(self):
        parsed = {
            "root_cause": "Connection pool exhausted.",
            "resolution_steps": ["Step 1", "Step 2"],
            "confidence": 0.80,
            "sources": ["KB"],
        }
        text = _format_synthesis(parsed)
        assert "Connection pool exhausted" in text
        assert "Step 1" in text
        assert "80%" in text
        assert "KB" in text

    def test_empty_steps_handled(self):
        parsed = {"root_cause": "Unknown", "resolution_steps": [], "confidence": 0.3, "sources": []}
        text = _format_synthesis(parsed)
        assert "Root Cause" in text
        assert "Confidence" in text


# ── _format_l2_context ────────────────────────────────────────────────────────


class TestFormatL2Context:

    def test_query_present_in_context(self):
        text = _format_l2_context(
            query="DB crash on peak load",
            impact="High",
            urgency="High",
            l1_summary="Memory issue",
            retrieved=[],
            resolution_options=[],
            web_results=[],
            escalation_reason="",
        )
        assert "DB crash on peak load" in text

    def test_web_results_included(self):
        web = [{"title": "Fix OOM", "url": "https://x.com", "content": "Increase heap size"}]
        text = _format_l2_context(
            query="OOM error",
            impact="High",
            urgency="Low",
            l1_summary="",
            retrieved=[],
            resolution_options=[],
            web_results=web,
            escalation_reason="",
        )
        assert "Fix OOM" in text
        assert "Increase heap size" in text

    def test_no_web_results_noted(self):
        text = _format_l2_context(
            query="OOM error",
            impact="",
            urgency="",
            l1_summary="",
            retrieved=[],
            resolution_options=[],
            web_results=[],
            escalation_reason="",
        )
        assert "No web results" in text


# ── make_l2_node ──────────────────────────────────────────────────────────────


class TestMakeL2Node:

    @pytest.fixture
    def app_config(self):
        return {
            "LLM": {"L2_MODEL": "gpt-4o"},
            "RETRIEVAL": {"L2_CONFIDENCE_THRESHOLD": 0.55, "TAVILY_MAX_RESULTS": 3},
        }

    @patch("src.agents.l2_analysis.tavily_web_search", new_callable=AsyncMock)
    @patch("src.agents.l2_analysis.chat_completion", new_callable=AsyncMock)
    async def test_high_confidence_no_escalation(self, mock_llm, mock_tavily, app_config):
        """L2 produces confidence ≥ 0.55 → no escalation_reason."""
        mock_tavily.return_value = []
        payload = {
            "root_cause": "DB overload",
            "resolution_steps": ["Restart"],
            "confidence": 0.80,
            "sources": ["Web"],
        }
        mock_llm.return_value = (json.dumps(payload), False)

        node = make_l2_node(app_config)
        state = {
            "query": "Database OOM",
            "impact": "High",
            "urgency": "High",
            "l1_summary": "Memory issue",
            "retrieved_incidents": [],
            "resolution_options": [],
            "escalation_reason": None,
        }
        result = await node(state)

        assert result["escalation_reason"] is None
        assert result["l2_confidence"] == pytest.approx(0.80)
        assert result["escalation_level"] == "L2"

    @patch("src.agents.l2_analysis.tavily_web_search", new_callable=AsyncMock)
    @patch("src.agents.l2_analysis.chat_completion", new_callable=AsyncMock)
    async def test_low_confidence_sets_escalation_reason(self, mock_llm, mock_tavily, app_config):
        """L2 confidence < 0.55 → escalation_reason set."""
        mock_tavily.return_value = []
        payload = {
            "root_cause": "Unknown",
            "resolution_steps": [],
            "confidence": 0.30,
            "sources": [],
        }
        mock_llm.return_value = (json.dumps(payload), False)

        node = make_l2_node(app_config)
        state = {
            "query": "Weird obscure error",
            "impact": "",
            "urgency": "",
            "l1_summary": "",
            "retrieved_incidents": [],
            "resolution_options": [],
            "escalation_reason": None,
        }
        result = await node(state)

        assert result["escalation_reason"] is not None
        assert "Requires specialist" in result["escalation_reason"]

    @patch("src.agents.l2_analysis.tavily_web_search", new_callable=AsyncMock)
    @patch("src.agents.l2_analysis.chat_completion", new_callable=AsyncMock)
    async def test_llm_failure_caps_confidence(self, mock_llm, mock_tavily, app_config):
        """LLM exception → fallback_used=True, confidence ≤ 0.40."""
        mock_tavily.return_value = []
        mock_llm.side_effect = Exception("OpenAI timeout")

        node = make_l2_node(app_config)
        state = {
            "query": "Error",
            "impact": "",
            "urgency": "",
            "l1_summary": "",
            "retrieved_incidents": [],
            "resolution_options": [],
            "escalation_reason": None,
        }
        result = await node(state)

        assert result["fallback_used"] is True
        assert result["l2_confidence"] <= 0.40

    @patch("src.agents.l2_analysis.tavily_web_search", new_callable=AsyncMock)
    @patch("src.agents.l2_analysis.chat_completion", new_callable=AsyncMock)
    async def test_web_results_stored_in_state(self, mock_llm, mock_tavily, app_config):
        """Web search results must be stored in returned state."""
        web_results = [{"title": "Fix it", "url": "https://x.com", "content": "details"}]
        mock_tavily.return_value = web_results
        mock_llm.return_value = (
            json.dumps({"root_cause": "x", "resolution_steps": [], "confidence": 0.7, "sources": ["Web"]}),
            False,
        )

        node = make_l2_node(app_config)
        state = {
            "query": "DB error",
            "impact": "",
            "urgency": "",
            "l1_summary": "",
            "retrieved_incidents": [],
            "resolution_options": [],
            "escalation_reason": None,
        }
        result = await node(state)
        assert result["web_search_results"] == web_results
