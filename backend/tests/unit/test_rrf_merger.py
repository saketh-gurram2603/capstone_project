"""
Unit tests for the RRF merger.
Pure function — no I/O, no external services.
"""

import pytest
from src.retrieval.rrf_merger import reciprocal_rank_fusion


def _make_results(ids_and_payloads: list[tuple]) -> list[dict]:
    """Helper to build result dicts quickly."""
    return [
        {"id": str(id_), "score": float(i + 1), "payload": payload or {}}
        for i, (id_, payload) in enumerate(ids_and_payloads)
    ]


class TestReciprocalRankFusion:

    def test_doc_in_both_lists_scores_higher(self):
        """A doc ranked top in BOTH lists must score higher than one in only one."""
        bm25 = [
            {"id": "A", "score": 10.0, "payload": {}},  # rank 0
            {"id": "B", "score": 5.0,  "payload": {}},  # rank 1
        ]
        vector = [
            {"id": "A", "score": 0.95, "payload": {}},  # rank 0
            {"id": "C", "score": 0.90, "payload": {}},  # rank 1
        ]
        merged = reciprocal_rank_fusion(bm25, vector, k=60)
        ids = [r["id"] for r in merged]
        # A appears in both lists → should be #1
        assert ids[0] == "A"

    def test_all_unique_docs_are_present(self):
        bm25   = [{"id": "A", "score": 1.0, "payload": {}}]
        vector = [{"id": "B", "score": 1.0, "payload": {}}]
        merged = reciprocal_rank_fusion(bm25, vector, k=60)
        ids = {r["id"] for r in merged}
        assert ids == {"A", "B"}

    def test_empty_bm25_returns_vector_only(self):
        vector = [
            {"id": "X", "score": 0.9, "payload": {"description": "disk"}},
            {"id": "Y", "score": 0.8, "payload": {}},
        ]
        merged = reciprocal_rank_fusion([], vector, k=60)
        assert len(merged) == 2
        assert merged[0]["id"] == "X"

    def test_empty_vector_returns_bm25_only(self):
        bm25 = [
            {"id": "M", "score": 8.0, "payload": {}},
            {"id": "N", "score": 4.0, "payload": {}},
        ]
        merged = reciprocal_rank_fusion(bm25, [], k=60)
        assert len(merged) == 2
        assert merged[0]["id"] == "M"

    def test_both_empty_returns_empty(self):
        assert reciprocal_rank_fusion([], [], k=60) == []

    def test_rrf_scores_are_positive(self):
        bm25   = [{"id": "A", "score": 5.0, "payload": {}}]
        vector = [{"id": "A", "score": 0.9, "payload": {}}]
        merged = reciprocal_rank_fusion(bm25, vector, k=60)
        assert merged[0]["score"] > 0

    def test_rrf_score_formula_correct(self):
        """rank-0 doc in BM25 only: score = 1/(60+0+1) = 1/61 ≈ 0.01639"""
        bm25 = [{"id": "Z", "score": 99.0, "payload": {}}]
        merged = reciprocal_rank_fusion(bm25, [], k=60)
        expected = 1.0 / 61.0
        assert abs(merged[0]["score"] - expected) < 1e-9

    def test_higher_rank_gets_higher_score(self):
        """Doc ranked 0 should score higher than doc ranked 1."""
        bm25 = [
            {"id": "A", "score": 10.0, "payload": {}},  # rank 0
            {"id": "B", "score":  5.0, "payload": {}},  # rank 1
        ]
        merged = reciprocal_rank_fusion(bm25, [], k=60)
        assert merged[0]["score"] > merged[1]["score"]

    def test_payload_from_vector_preferred(self):
        """When a doc is in both lists, the vector payload (richer) is kept."""
        bm25   = [{"id": "A", "score": 5.0, "payload": {"source": "bm25"}}]
        vector = [{"id": "A", "score": 0.9, "payload": {"source": "vector", "description": "rich"}}]
        merged = reciprocal_rank_fusion(bm25, vector, k=60)
        assert merged[0]["payload"]["source"] == "vector"

    def test_sources_field_recorded(self):
        bm25   = [{"id": "A", "score": 5.0, "payload": {}}]
        vector = [{"id": "A", "score": 0.9, "payload": {}}]
        merged = reciprocal_rank_fusion(bm25, vector, k=60)
        assert "bm25" in merged[0]["sources"]
        assert "vector" in merged[0]["sources"]

    def test_output_sorted_descending(self):
        bm25 = [{"id": str(i), "score": float(10 - i), "payload": {}} for i in range(5)]
        merged = reciprocal_rank_fusion(bm25, [], k=60)
        scores = [r["score"] for r in merged]
        assert scores == sorted(scores, reverse=True)
