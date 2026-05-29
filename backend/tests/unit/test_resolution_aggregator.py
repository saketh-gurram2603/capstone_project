"""
Unit tests for the resolution aggregator.
embed_batch is mocked — no OpenAI/MiniLM calls needed.
"""

import pytest
from unittest.mock import AsyncMock, patch

from src.retrieval.resolution_aggregator import (
    aggregate_resolutions,
    _cosine_similarity,
    _greedy_cluster,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_candidate(incident_id: str, resolution: str, similarity: float = 0.8) -> dict:
    return {
        "id": incident_id,
        "score": 0.02,
        "rerank_score": 2.0,
        "similarity_score": similarity,
        "payload": {
            "incident_id": incident_id,
            "resolution_notes": resolution,
        },
    }


# ── _cosine_similarity ────────────────────────────────────────────────────────

class TestCosineSimilarity:

    def test_identical_vectors_similarity_one(self):
        v = [1.0, 0.0, 0.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors_similarity_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_similarity_in_range(self):
        import random
        random.seed(42)
        a = [random.random() for _ in range(10)]
        b = [random.random() for _ in range(10)]
        sim = _cosine_similarity(a, b)
        assert -1.0 <= sim <= 1.0


# ── _greedy_cluster ───────────────────────────────────────────────────────────

class TestGreedyCluster:

    def test_identical_vectors_merged(self):
        v = [1.0, 0.0, 0.0]
        vectors = [v, v, v]
        clusters = _greedy_cluster(vectors, threshold=0.95)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_orthogonal_vectors_separate_clusters(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        v3 = [0.0, 0.0, 1.0]
        clusters = _greedy_cluster([v1, v2, v3], threshold=0.95)
        assert len(clusters) == 3

    def test_similar_vectors_merged_dissimilar_kept(self):
        # v1 and v2 are nearly identical; v3 is orthogonal
        v1 = [1.0, 0.01, 0.0]
        v2 = [1.0, 0.02, 0.0]
        v3 = [0.0, 0.0, 1.0]
        clusters = _greedy_cluster([v1, v2, v3], threshold=0.95)
        assert len(clusters) == 2

    def test_all_indices_covered(self):
        vectors = [[float(i), 0.0] for i in range(5)]
        clusters = _greedy_cluster(vectors, threshold=0.95)
        all_indices = [idx for cluster in clusters for idx in cluster]
        assert sorted(all_indices) == list(range(5))


# ── aggregate_resolutions ─────────────────────────────────────────────────────

class TestAggregateResolutions:

    @patch("src.retrieval.resolution_aggregator.embed_batch", new_callable=AsyncMock)
    async def test_empty_input_returns_empty(self, mock_embed):
        result = await aggregate_resolutions([])
        assert result == []
        mock_embed.assert_not_called()

    @patch("src.retrieval.resolution_aggregator.embed_batch", new_callable=AsyncMock)
    async def test_single_incident_returns_one_option(self, mock_embed):
        mock_embed.return_value = [[1.0, 0.0, 0.0]]
        candidates = [_make_candidate("INC-001", "Restart the service")]
        result = await aggregate_resolutions(candidates, dedup_threshold=0.95)
        assert len(result) == 1
        assert result[0]["resolution_text"] == "Restart the service"
        assert result[0]["occurrence_count"] == 1

    @patch("src.retrieval.resolution_aggregator.embed_batch", new_callable=AsyncMock)
    async def test_identical_resolutions_deduplicated(self, mock_embed):
        # Both incidents have the same resolution → same vector → merged into 1 cluster
        same_vec = [1.0, 0.0, 0.0]
        mock_embed.return_value = [same_vec, same_vec]
        candidates = [
            _make_candidate("INC-001", "Restart the service"),
            _make_candidate("INC-002", "Restart the service"),
        ]
        result = await aggregate_resolutions(candidates, dedup_threshold=0.95)
        assert len(result) == 1
        assert result[0]["occurrence_count"] == 2

    @patch("src.retrieval.resolution_aggregator.embed_batch", new_callable=AsyncMock)
    async def test_different_resolutions_kept_separate(self, mock_embed):
        mock_embed.return_value = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        candidates = [
            _make_candidate("INC-001", "Restart the service"),
            _make_candidate("INC-002", "Increase connection pool size"),
        ]
        result = await aggregate_resolutions(candidates, dedup_threshold=0.95)
        assert len(result) == 2

    @patch("src.retrieval.resolution_aggregator.embed_batch", new_callable=AsyncMock)
    async def test_sorted_by_occurrence_times_similarity(self, mock_embed):
        """Fix used 3 times should appear before fix used 1 time."""
        same_vec = [1.0, 0.0, 0.0]
        other_vec = [0.0, 1.0, 0.0]
        mock_embed.return_value = [same_vec, same_vec, same_vec, other_vec]
        candidates = [
            _make_candidate("INC-001", "Restart service", similarity=0.9),
            _make_candidate("INC-002", "Restart service", similarity=0.85),
            _make_candidate("INC-003", "Restart service", similarity=0.80),
            _make_candidate("INC-004", "Add replica", similarity=0.95),
        ]
        result = await aggregate_resolutions(candidates, dedup_threshold=0.95)
        # "Restart service" cluster: count=3, avg=~0.85 → score=2.55
        # "Add replica" cluster: count=1, avg=0.95 → score=0.95
        assert result[0]["occurrence_count"] == 3
        assert result[1]["occurrence_count"] == 1

    @patch("src.retrieval.resolution_aggregator.embed_batch", new_callable=AsyncMock)
    async def test_source_incident_ids_present(self, mock_embed):
        same_vec = [1.0, 0.0, 0.0]
        mock_embed.return_value = [same_vec, same_vec]
        candidates = [
            _make_candidate("INC-A", "Fix A"),
            _make_candidate("INC-B", "Fix A"),
        ]
        result = await aggregate_resolutions(candidates, dedup_threshold=0.95)
        assert set(result[0]["source_incident_ids"]) == {"INC-A", "INC-B"}

    @patch("src.retrieval.resolution_aggregator.embed_batch", new_callable=AsyncMock)
    async def test_skips_empty_resolution_notes(self, mock_embed):
        mock_embed.return_value = [[1.0, 0.0], [0.0, 1.0]]
        candidates = [
            _make_candidate("INC-001", ""),           # empty → skip
            _make_candidate("INC-002", "Valid fix"),  # kept
        ]
        result = await aggregate_resolutions(candidates, dedup_threshold=0.95)
        assert len(result) == 1
        assert result[0]["resolution_text"] == "Valid fix"
