"""
Unit tests for the cross-encoder reranker.
Loads the real model (cached locally) for a small batch to verify correctness.
Skips model-loading tests if the model is not cached.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.retrieval.reranker import (
    rerank,
    init_reranker,
    is_reranker_loaded,
    _sigmoid,
    _add_fallback_scores,
    _get_doc_text,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_candidates(texts: list[str]) -> list[dict]:
    return [
        {
            "id": str(i),
            "score": 0.02,
            "payload": {"search_text": t, "description": t},
        }
        for i, t in enumerate(texts)
    ]


# ── _sigmoid ──────────────────────────────────────────────────────────────────

class TestSigmoid:

    def test_zero_maps_to_half(self):
        assert abs(_sigmoid(0.0) - 0.5) < 1e-6

    def test_large_positive_approaches_one(self):
        assert _sigmoid(100.0) > 0.99

    def test_large_negative_approaches_zero(self):
        assert _sigmoid(-100.0) < 0.01

    def test_output_in_range(self):
        for x in [-10, -1, 0, 1, 10]:
            val = _sigmoid(float(x))
            assert 0.0 <= val <= 1.0


# ── _add_fallback_scores ──────────────────────────────────────────────────────

class TestFallbackScores:

    def test_assigns_rerank_score(self):
        candidates = [{"id": str(i), "score": 0.5} for i in range(3)]
        _add_fallback_scores(candidates)
        for c in candidates:
            assert "rerank_score" in c
            assert "similarity_score" in c

    def test_first_has_highest_similarity(self):
        candidates = [{"id": str(i), "score": 0.5} for i in range(5)]
        _add_fallback_scores(candidates)
        scores = [c["similarity_score"] for c in candidates]
        assert scores[0] == max(scores)

    def test_scores_in_valid_range(self):
        candidates = [{"id": str(i), "score": 0.5} for i in range(10)]
        _add_fallback_scores(candidates)
        for c in candidates:
            assert 0.0 <= c["similarity_score"] <= 1.0


# ── _get_doc_text ─────────────────────────────────────────────────────────────

class TestGetDocText:

    def test_returns_search_text_first(self):
        c = {"payload": {"search_text": "disk space alert", "description": "something else"}}
        assert _get_doc_text(c) == "disk space alert"

    def test_falls_back_to_description(self):
        c = {"payload": {"description": "memory leak"}}
        assert _get_doc_text(c) == "memory leak"

    def test_returns_empty_when_no_text(self):
        assert _get_doc_text({"payload": {}}) == ""

    def test_handles_missing_payload(self):
        assert _get_doc_text({}) == ""


# ── rerank — mocked cross-encoder ─────────────────────────────────────────────

class TestRerankMocked:

    def test_rerank_sorts_by_score_descending(self):
        """Mock the cross-encoder to return known scores and verify ordering."""
        with patch("src.retrieval.reranker._cross_encoder") as mock_ce:
            # Assign scores: candidate 2 gets highest, candidate 0 gets lowest
            mock_ce.predict.return_value = [1.0, 5.0, 10.0]

            candidates = _make_candidates(["low", "mid", "high"])
            result = rerank("test query", candidates)

            assert result[0]["payload"]["search_text"] == "high"
            assert result[1]["payload"]["search_text"] == "mid"
            assert result[2]["payload"]["search_text"] == "low"

    def test_rerank_adds_similarity_score(self):
        with patch("src.retrieval.reranker._cross_encoder") as mock_ce:
            mock_ce.predict.return_value = [0.0, 2.0]
            candidates = _make_candidates(["a", "b"])
            result = rerank("query", candidates)
            for c in result:
                assert "similarity_score" in c
                assert 0.0 <= c["similarity_score"] <= 1.0

    def test_rerank_empty_candidates_returns_empty(self):
        result = rerank("query", [])
        assert result == []

    def test_rerank_without_model_uses_fallback(self):
        """When model not loaded, candidates returned with fallback scores."""
        with patch("src.retrieval.reranker._cross_encoder", None):
            candidates = _make_candidates(["a", "b", "c"])
            result = rerank("query", candidates)
            assert len(result) == 3
            for c in result:
                assert "similarity_score" in c

    def test_rerank_calls_predict_with_pairs(self):
        with patch("src.retrieval.reranker._cross_encoder") as mock_ce:
            mock_ce.predict.return_value = [1.0, 2.0]
            candidates = _make_candidates(["doc one", "doc two"])
            rerank("my query", candidates)

            call_args = mock_ce.predict.call_args[0][0]
            assert call_args[0] == ("my query", "doc one")
            assert call_args[1] == ("my query", "doc two")


# ── rerank — real model (skipped if not cached) ───────────────────────────────

@pytest.mark.skipif(not is_reranker_loaded(), reason="Reranker not loaded in this env")
class TestRerankReal:

    def test_relevant_doc_scores_highest(self):
        candidates = _make_candidates([
            "disk space storage exceeded causing upload failures",
            "weather forecast sunny tomorrow",
            "disk quota full unable to write files",
        ])
        result = rerank("disk space full", candidates)
        top_text = result[0]["payload"]["search_text"]
        assert "disk" in top_text or "quota" in top_text
