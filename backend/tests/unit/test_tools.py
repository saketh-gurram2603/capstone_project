"""
Unit tests for agent tools: classify_priority and search_kb_incidents.
tavily_web_search is not tested here (requires API key — covered separately).
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from src.agents.tools import classify_priority, search_kb_incidents


# ── classify_priority ─────────────────────────────────────────────────────────


class TestClassifyPriority:

    def test_high_high_is_p1(self):
        assert classify_priority("High", "High") == "P1"

    def test_high_medium_is_p2(self):
        assert classify_priority("High", "Medium") == "P2"

    def test_medium_medium_is_p3(self):
        # total = 2+2 = 4 → P3 (P2 requires total ≥ 5)
        assert classify_priority("Medium", "Medium") == "P3"

    def test_low_low_is_p4(self):
        assert classify_priority("Low", "Low") == "P4"

    def test_medium_low_is_p3(self):
        assert classify_priority("Medium", "Low") == "P3"

    def test_low_high_is_p3(self):
        assert classify_priority("Low", "High") == "P3"

    def test_unknown_values_default_to_medium(self):
        """Unknown impact/urgency defaults to score=2 (Medium). total=4 → P3."""
        result = classify_priority("Unknown", "Unknown")
        # Both default to 2 → total=4 → P3 (P2 requires total ≥ 5)
        assert result == "P3"

    def test_high_low_is_p3(self):
        assert classify_priority("High", "Low") == "P3"

    def test_empty_strings_default_to_medium(self):
        # Empty string → default score=2, total=4 → P3
        result = classify_priority("", "")
        assert result == "P3"


# ── search_kb_incidents ───────────────────────────────────────────────────────


class TestSearchKBIncidents:

    @patch("src.agents.tools.hybrid_search", new_callable=AsyncMock)
    async def test_returns_hybrid_search_result(self, mock_search):
        """Should pass through the hybrid_search result unchanged."""
        mock_result = {
            "results": [{"id": "INC-001", "similarity_score": 0.9}],
            "resolution_options": [{"resolution_text": "Restart", "occurrence_count": 3}],
        }
        mock_search.return_value = mock_result

        from unittest.mock import MagicMock
        vs = MagicMock()
        result = await search_kb_incidents(
            query="DB crash",
            vector_store=vs,
            collection="incidents",
            app_config={},
        )

        assert result == mock_result
        mock_search.assert_called_once()

    @patch("src.agents.tools.hybrid_search", new_callable=AsyncMock)
    async def test_returns_empty_dict_on_failure(self, mock_search):
        """hybrid_search exception → graceful empty result."""
        mock_search.side_effect = Exception("Qdrant down")

        from unittest.mock import MagicMock
        vs = MagicMock()
        result = await search_kb_incidents(
            query="DB crash",
            vector_store=vs,
            collection="incidents",
            app_config={},
        )

        assert result["results"] == []
        assert result["resolution_options"] == []
        assert result["retrieval_method"] == "failed"

    @patch("src.agents.tools.hybrid_search", new_callable=AsyncMock)
    async def test_filters_passed_through(self, mock_search):
        """Filters must be forwarded to hybrid_search."""
        mock_search.return_value = {"results": [], "resolution_options": []}

        from unittest.mock import MagicMock
        vs = MagicMock()
        filters = {"category": "Database"}
        await search_kb_incidents(
            query="slow query",
            vector_store=vs,
            collection="incidents",
            app_config={},
            filters=filters,
        )

        call_kwargs = mock_search.call_args.kwargs
        assert call_kwargs.get("filters") == filters
