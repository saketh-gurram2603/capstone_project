"""
Unit tests for the Adaptive-K selector.
Pure functions — no I/O, no external services.
"""

import pytest
from src.retrieval.adaptive_k import compute_k, trim_by_score_dropoff


class TestComputeK:

    def test_very_short_query_returns_k_min(self):
        # "disk" → 1 token, complexity=1 < 5 → k_min
        assert compute_k("disk", k_min=3, k_default=10, k_max=20) == 3

    def test_medium_query_returns_k_default(self):
        # "database connection timeout error" → 4 tokens, complexity=4 < 5? No, 4 < 5 → k_min
        # "storage service crashed after upgrade" → 5 tokens, complexity=5 → boundary
        assert compute_k("storage service crashed after upgrade", k_min=3, k_default=10, k_max=20) == 10

    def test_long_query_returns_k_max(self):
        # 15-word query with no digits → complexity = 15 + 2 = 17 > 12 → k_max
        query = "service keeps throwing out of memory error every thirty minutes under peak load"
        assert compute_k(query, k_min=3, k_default=10, k_max=20) == 20

    def test_digit_tokens_increase_complexity(self):
        # "502 error" → 2 tokens + 3*1 digit-token = 5 → boundary case
        # "502 error http 404 timeout" → 5 tokens + 3*2 = 11 → k_default
        assert compute_k("502 error http 404 timeout", k_min=3, k_default=10, k_max=20) == 10

    def test_error_code_query_gets_high_k(self):
        # "OOM error 502 crash 404 500 retry" → 7 + 3*5=15 + 0 = 22 > 12 → k_max
        assert compute_k("OOM error 502 crash 404 500 retry", k_min=3, k_default=10, k_max=20) == 20

    def test_single_token_returns_k_min(self):
        assert compute_k("crash", k_min=3, k_default=10, k_max=20) == 3

    def test_empty_string_returns_k_min(self):
        assert compute_k("", k_min=3, k_default=10, k_max=20) == 3

    def test_respects_custom_k_values(self):
        assert compute_k("disk", k_min=5, k_default=15, k_max=25) == 5

    def test_boundary_complexity_5(self):
        # exactly 5 tokens, no digits → complexity=5 → k_default (not k_min)
        result = compute_k("a b c d e", k_min=3, k_default=10, k_max=20)
        assert result == 10

    def test_over_10_tokens_adds_bonus(self):
        # 11 tokens, no digits → complexity = 11 + 2 = 13 > 12 → k_max
        query = "a b c d e f g h i j k"
        assert compute_k(query, k_min=3, k_default=10, k_max=20) == 20


class TestTrimByScoreDropoff:

    def test_no_trim_when_scores_close(self):
        results = [
            {"id": "1", "score": 0.90},
            {"id": "2", "score": 0.88},
            {"id": "3", "score": 0.86},
        ]
        trimmed = trim_by_score_dropoff(results, threshold=0.15)
        assert len(trimmed) == 3

    def test_trims_at_large_score_drop(self):
        results = [
            {"id": "1", "score": 0.90},
            {"id": "2", "score": 0.88},
            {"id": "3", "score": 0.86},
            {"id": "4", "score": 0.50},   # drop: (0.86-0.50)/0.86 = 0.42 > 0.15 → STOP
            {"id": "5", "score": 0.48},
        ]
        trimmed = trim_by_score_dropoff(results, threshold=0.15)
        assert len(trimmed) == 3
        assert trimmed[-1]["id"] == "3"

    def test_always_keeps_first_result(self):
        results = [{"id": "1", "score": 0.9}]
        trimmed = trim_by_score_dropoff(results, threshold=0.15)
        assert len(trimmed) == 1

    def test_empty_list_returns_empty(self):
        assert trim_by_score_dropoff([], threshold=0.15) == []

    def test_all_zero_scores_returns_all(self):
        results = [{"id": str(i), "score": 0.0} for i in range(5)]
        trimmed = trim_by_score_dropoff(results, threshold=0.15)
        assert len(trimmed) == 5

    def test_trim_at_exact_threshold(self):
        # Drop = (0.80 - 0.68) / 0.80 = 0.15 — NOT > threshold → keep
        results = [
            {"id": "1", "score": 0.80},
            {"id": "2", "score": 0.68},
        ]
        trimmed = trim_by_score_dropoff(results, threshold=0.15)
        assert len(trimmed) == 2

    def test_trim_just_above_threshold(self):
        # Drop = (0.80 - 0.67) / 0.80 = 0.1625 > 0.15 → trim
        results = [
            {"id": "1", "score": 0.80},
            {"id": "2", "score": 0.67},
        ]
        trimmed = trim_by_score_dropoff(results, threshold=0.15)
        assert len(trimmed) == 1

    def test_returns_copy_not_original_list(self):
        results = [{"id": "1", "score": 0.9}, {"id": "2", "score": 0.8}]
        trimmed = trim_by_score_dropoff(results, threshold=0.15)
        assert trimmed is not results
