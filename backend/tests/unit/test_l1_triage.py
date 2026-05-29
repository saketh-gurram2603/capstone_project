"""
Unit tests for L1 triage helpers.

No real LLM calls — chat_completion and search_kb_incidents are mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.l1_triage import (
    _calculate_confidence,
    _format_kb_context,
    make_l1_node,
)


# ── _calculate_confidence ─────────────────────────────────────────────────────


class TestCalculateConfidence:

    def test_empty_results_returns_zero(self):
        assert _calculate_confidence([]) == 0.0

    def test_single_result_returns_its_score(self):
        results = [{"similarity_score": 0.9}]
        conf = _calculate_confidence(results)
        # Only one result: weight=1, score=0.9 → 0.9*1/1 = 0.9
        assert abs(conf - 0.9) < 1e-6

    def test_top5_weighted_correctly(self):
        """Position 0 gets weight 5, position 4 gets weight 1."""
        results = [{"similarity_score": float(i) / 10} for i in range(1, 6)]
        # scores: [0.1, 0.2, 0.3, 0.4, 0.5]
        # weights: [5, 4, 3, 2, 1]
        # weighted: 0.5 + 0.8 + 0.9 + 0.8 + 0.5 = 3.5
        # sum weights: 15
        expected = (0.1 * 5 + 0.2 * 4 + 0.3 * 3 + 0.4 * 2 + 0.5 * 1) / 15
        assert abs(_calculate_confidence(results) - expected) < 1e-6

    def test_only_top5_used(self):
        """Results beyond index 4 must be ignored."""
        results = [{"similarity_score": 1.0}] * 10  # All score 1.0
        assert abs(_calculate_confidence(results) - 1.0) < 1e-6

    def test_missing_score_defaults_to_zero(self):
        results = [{"title": "No score"}]
        assert _calculate_confidence(results) == 0.0

    def test_confidence_in_range(self):
        import random
        random.seed(99)
        results = [{"similarity_score": random.random()} for _ in range(5)]
        conf = _calculate_confidence(results)
        assert 0.0 <= conf <= 1.0


# ── _format_kb_context ────────────────────────────────────────────────────────


class TestFormatKBContext:

    def test_empty_results_returns_no_incidents_message(self):
        text = _format_kb_context([], [])
        assert "No relevant incidents" in text

    def test_results_included_in_output(self):
        results = [
            {
                "title": "DB timeout",
                "similarity_score": 0.91,
                "description": "Database timed out during peak load",
                "resolution_notes": "Increased connection pool",
                "category": "Database",
            }
        ]
        text = _format_kb_context(results, [])
        assert "DB timeout" in text
        assert "0.91" in text
        assert "Increased connection pool" in text

    def test_resolution_options_included(self):
        results = [
            {
                "title": "T",
                "similarity_score": 0.8,
                "description": "desc",
                "resolution_notes": "fix",
                "category": "N/A",
            }
        ]
        opts = [{"resolution_text": "Restart the pod", "occurrence_count": 7}]
        text = _format_kb_context(results, opts)
        assert "Restart the pod" in text
        assert "7" in text

    def test_max_five_results_shown(self):
        results = [
            {
                "title": f"INC-{i}",
                "similarity_score": 0.9,
                "description": "desc",
                "resolution_notes": "fix",
                "category": "Cat",
            }
            for i in range(10)
        ]
        text = _format_kb_context(results, [])
        # Only INC-0 through INC-4 should appear
        assert "INC-4" in text
        assert "INC-5" not in text


# ── make_l1_node (async node) ─────────────────────────────────────────────────


class TestMakeL1Node:

    @pytest.fixture
    def mock_vector_store(self):
        return MagicMock()

    @pytest.fixture
    def app_config(self):
        return {
            "LLM": {"L1_MODEL": "gpt-4o-mini"},
            "RETRIEVAL": {"L1_CONFIDENCE_THRESHOLD": 0.80},
        }

    @patch("src.agents.l1_triage.search_kb_incidents", new_callable=AsyncMock)
    @patch("src.agents.l1_triage.chat_completion", new_callable=AsyncMock)
    async def test_high_confidence_no_escalation_reason(
        self, mock_llm, mock_search, mock_vector_store, app_config
    ):
        """High confidence → escalation_reason should be None."""
        mock_search.return_value = {
            "results": [{"similarity_score": 0.95, "title": "T", "description": "d",
                          "resolution_notes": "r", "category": "C"}],
            "resolution_options": [],
        }
        mock_llm.return_value = ("Restart the service.", False)

        node = make_l1_node(mock_vector_store, "incidents", app_config)
        state = {"query": "Database OOM", "impact": "High", "urgency": "High"}
        result = await node(state)

        assert result["escalation_reason"] is None
        assert result["l1_confidence"] > 0
        assert result["escalation_level"] == "L1"
        assert result["priority"] == "P1"
        assert result["fallback_used"] is False

    @patch("src.agents.l1_triage.search_kb_incidents", new_callable=AsyncMock)
    @patch("src.agents.l1_triage.chat_completion", new_callable=AsyncMock)
    async def test_low_confidence_sets_escalation_reason(
        self, mock_llm, mock_search, mock_vector_store, app_config
    ):
        """Low confidence → escalation_reason must be set."""
        mock_search.return_value = {
            "results": [{"similarity_score": 0.30, "title": "T", "description": "d",
                          "resolution_notes": "r", "category": "C"}],
            "resolution_options": [],
        }
        mock_llm.return_value = ("Unclear — needs investigation.", False)

        node = make_l1_node(mock_vector_store, "incidents", app_config)
        state = {"query": "Some weird error", "impact": "Low", "urgency": "Low"}
        result = await node(state)

        assert result["escalation_reason"] is not None
        assert "below threshold" in result["escalation_reason"]

    @patch("src.agents.l1_triage.search_kb_incidents", new_callable=AsyncMock)
    @patch("src.agents.l1_triage.chat_completion", new_callable=AsyncMock)
    async def test_llm_failure_forces_low_confidence(
        self, mock_llm, mock_search, mock_vector_store, app_config
    ):
        """LLM exception → confidence capped at 0.4, fallback_used=True."""
        mock_search.return_value = {
            "results": [{"similarity_score": 0.90, "title": "T", "description": "d",
                          "resolution_notes": "r", "category": "C"}],
            "resolution_options": [],
        }
        mock_llm.side_effect = Exception("OpenAI timeout")

        node = make_l1_node(mock_vector_store, "incidents", app_config)
        state = {"query": "Some error", "impact": "Medium", "urgency": "Medium"}
        result = await node(state)

        assert result["l1_confidence"] <= 0.4
        assert result["fallback_used"] is True

    @patch("src.agents.l1_triage.search_kb_incidents", new_callable=AsyncMock)
    @patch("src.agents.l1_triage.chat_completion", new_callable=AsyncMock)
    async def test_empty_kb_results_low_confidence(
        self, mock_llm, mock_search, mock_vector_store, app_config
    ):
        """No KB results → confidence=0.0 → must escalate."""
        mock_search.return_value = {"results": [], "resolution_options": []}
        mock_llm.return_value = ("No similar incidents found.", False)

        node = make_l1_node(mock_vector_store, "incidents", app_config)
        state = {"query": "Unknown exotic error", "impact": "High", "urgency": "Low"}
        result = await node(state)

        assert result["l1_confidence"] == 0.0
        assert result["escalation_reason"] is not None
