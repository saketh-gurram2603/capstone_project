"""
Unit tests for IR metrics — pure functions, no I/O.

Tests cover: ndcg_at_k, map_at_k, recall_at_k, precision_at_k,
             compute_all_metrics, and mathematical edge cases.
"""

from __future__ import annotations

import math
import pytest

from src.evaluation.ir_metrics import (
    compute_all_metrics,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# ── ndcg_at_k ─────────────────────────────────────────────────────────────────


class TestNdcgAtK:

    def test_perfect_retrieval_is_one(self):
        assert ndcg_at_k(["A", "B", "C"], ["A", "B", "C"], k=3) == pytest.approx(1.0)

    def test_no_relevant_in_retrieved_is_zero(self):
        assert ndcg_at_k(["X", "Y"], ["A", "B"], k=2) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert ndcg_at_k(["A", "B"], [], k=2) == 0.0

    def test_empty_retrieved_returns_zero(self):
        assert ndcg_at_k([], ["A", "B"], k=5) == 0.0

    def test_top_ranked_relevant_scores_higher(self):
        """Relevant doc at rank 0 must beat same doc at rank 2."""
        score_top = ndcg_at_k(["A", "X", "Y"], ["A"], k=3)
        score_low = ndcg_at_k(["X", "Y", "A"], ["A"], k=3)
        assert score_top > score_low

    def test_k_limits_retrieved_list(self):
        """Results beyond k must be ignored."""
        # Only "A" within top-1 → DCG = 1/log2(2)=1.0 = IDCG → score=1.0
        assert ndcg_at_k(["A", "B", "C"], ["A"], k=1) == pytest.approx(1.0)
        # "A" at rank 1 (index 1) within k=2 but not rank 0
        s1 = ndcg_at_k(["X", "A"], ["A"], k=2)
        assert 0.0 < s1 < 1.0

    def test_single_relevant_single_retrieved_match(self):
        # DCG = 1/log2(2) = 1.0; IDCG = 1/log2(2) = 1.0 → NDCG = 1.0
        assert ndcg_at_k(["A"], ["A"], k=1) == pytest.approx(1.0)

    def test_partial_retrieval_between_zero_and_one(self):
        score = ndcg_at_k(["A", "X"], ["A", "B"], k=2)
        assert 0.0 < score < 1.0

    def test_score_in_range(self):
        score = ndcg_at_k(["A", "B", "C", "D"], ["B", "D"], k=4)
        assert 0.0 <= score <= 1.0

    def test_formula_single_result_rank_0(self):
        """Single relevant doc at rank 0: DCG = IDCG = 1/log2(2) → NDCG = 1.0"""
        assert ndcg_at_k(["X"], ["X"], k=10) == pytest.approx(1.0)

    def test_formula_single_result_rank_1(self):
        """Relevant doc at rank 1 (index 1): DCG = 1/log2(3), IDCG = 1/log2(2)"""
        expected = (1 / math.log2(3)) / (1 / math.log2(2))
        assert ndcg_at_k(["Y", "X"], ["X"], k=2) == pytest.approx(expected, abs=1e-9)


# ── map_at_k ──────────────────────────────────────────────────────────────────


class TestMapAtK:

    def test_all_relevant_at_top(self):
        assert map_at_k(["A", "B", "C"], ["A", "B", "C"], k=3) == pytest.approx(1.0)

    def test_no_hits_returns_zero(self):
        assert map_at_k(["X", "Y"], ["A", "B"], k=2) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert map_at_k(["A"], [], k=5) == 0.0

    def test_empty_retrieved_returns_zero(self):
        assert map_at_k([], ["A"], k=5) == 0.0

    def test_single_relevant_at_rank_0(self):
        # Precision at rank 1 = 1/1 = 1.0; num_relevant=1 → MAP=1.0
        assert map_at_k(["A", "X"], ["A"], k=2) == pytest.approx(1.0)

    def test_single_relevant_at_rank_1(self):
        # Precision at rank 2 = 1/2 = 0.5; num_relevant=1 → MAP=0.5
        assert map_at_k(["X", "A"], ["A"], k=2) == pytest.approx(0.5)

    def test_score_in_range(self):
        score = map_at_k(["A", "B", "X", "C"], ["A", "C"], k=4)
        assert 0.0 <= score <= 1.0

    def test_k_truncates_beyond_limit(self):
        # A is at rank 4 (index 4) → beyond k=3 → not counted
        assert map_at_k(["X", "Y", "Z", "A"], ["A"], k=3) == 0.0


# ── recall_at_k ───────────────────────────────────────────────────────────────


class TestRecallAtK:

    def test_all_relevant_retrieved(self):
        assert recall_at_k(["A", "B", "C"], ["A", "B", "C"], k=3) == pytest.approx(1.0)

    def test_no_relevant_retrieved(self):
        assert recall_at_k(["X", "Y"], ["A", "B"], k=2) == 0.0

    def test_partial_recall(self):
        # Retrieved A and B, but C not retrieved → recall = 2/3
        assert recall_at_k(["A", "B", "X"], ["A", "B", "C"], k=3) == pytest.approx(2 / 3)

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k(["A"], [], k=5) == 0.0

    def test_empty_retrieved_returns_zero(self):
        assert recall_at_k([], ["A"], k=5) == 0.0

    def test_k_truncates_retrieved(self):
        # Relevant=A is at rank 3 (index 3) but k=2 → not retrieved
        assert recall_at_k(["X", "Y", "Z", "A"], ["A"], k=2) == 0.0

    def test_score_in_range(self):
        score = recall_at_k(["A", "B"], ["A", "B", "C"], k=5)
        assert 0.0 <= score <= 1.0


# ── precision_at_k ────────────────────────────────────────────────────────────


class TestPrecisionAtK:

    def test_perfect_precision(self):
        assert precision_at_k(["A", "B"], ["A", "B"], k=2) == pytest.approx(1.0)

    def test_zero_precision(self):
        assert precision_at_k(["X", "Y"], ["A", "B"], k=2) == 0.0

    def test_half_precision(self):
        assert precision_at_k(["A", "X"], ["A", "B"], k=2) == pytest.approx(0.5)

    def test_k_zero_returns_zero(self):
        assert precision_at_k(["A"], ["A"], k=0) == 0.0

    def test_empty_retrieved_returns_zero(self):
        assert precision_at_k([], ["A"], k=5) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert precision_at_k(["A", "B"], [], k=2) == 0.0

    def test_k_truncates(self):
        # k=1: only first doc "X" (irrelevant) → precision=0
        assert precision_at_k(["X", "A"], ["A"], k=1) == 0.0

    def test_score_in_range(self):
        score = precision_at_k(["A", "B", "X", "C"], ["A", "C"], k=4)
        assert 0.0 <= score <= 1.0


# ── compute_all_metrics ───────────────────────────────────────────────────────


class TestComputeAllMetrics:

    def test_returns_all_four_keys(self):
        result = compute_all_metrics(["A", "B"], ["A"], k=2)
        assert set(result.keys()) == {"ndcg_at_k", "map_at_k", "recall_at_k", "precision_at_k"}

    def test_perfect_retrieval_all_ones(self):
        result = compute_all_metrics(["A", "B", "C"], ["A", "B", "C"], k=3)
        for key, val in result.items():
            assert val == pytest.approx(1.0), f"{key} should be 1.0"

    def test_empty_returns_all_zeros(self):
        result = compute_all_metrics([], [], k=10)
        for val in result.values():
            assert val == 0.0

    def test_scores_in_range(self):
        result = compute_all_metrics(["A", "X", "B"], ["A", "B", "C"], k=3)
        for val in result.values():
            assert 0.0 <= val <= 1.0

    def test_consistent_with_individual_functions(self):
        retrieved = ["B", "A", "X", "C"]
        relevant = ["A", "B"]
        k = 4
        result = compute_all_metrics(retrieved, relevant, k)
        assert result["ndcg_at_k"] == pytest.approx(ndcg_at_k(retrieved, relevant, k))
        assert result["map_at_k"] == pytest.approx(map_at_k(retrieved, relevant, k))
        assert result["recall_at_k"] == pytest.approx(recall_at_k(retrieved, relevant, k))
        assert result["precision_at_k"] == pytest.approx(precision_at_k(retrieved, relevant, k))
