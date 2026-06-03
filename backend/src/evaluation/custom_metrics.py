"""
Custom evaluation metrics for the Incident KB system.

Metrics:
  resolution_time_mae   — Mean Absolute Error between predicted and actual
                          resolution time (hours).  Predicted = weighted
                          average of retrieved incidents' resolution_hours,
                          weighted by similarity_score.

  fix_accuracy          — Fraction of test cases where the top-ranked
                          resolution option contains at least one expected
                          keyword from the ground-truth dataset.

Both metrics are pure Python, zero external dependencies.
"""

from __future__ import annotations

from typing import Sequence


# ── Resolution Time Prediction ────────────────────────────────────────────────

def predict_resolution_hours(
    retrieved_results: list[dict],
    top_k: int = 5,
) -> float:
    """
    Predict resolution time for a query as the similarity-weighted average
    of the retrieved incidents' resolution_hours values.

    Parameters
    ----------
    retrieved_results : list[dict]
        Each item is an IncidentResponse-compatible dict with at least
        ``resolution_hours`` (float) and ``similarity_score`` (float) keys.
        These come directly from the hybrid_search response.
    top_k : int
        Number of top results to use for the prediction.

    Returns
    -------
    float  Predicted hours, or 0.0 if no results carry resolution_hours data.
    """
    weighted_sum  = 0.0
    weight_total  = 0.0

    for r in retrieved_results[:top_k]:
        hours  = float(r.get("resolution_hours") or 0.0)
        weight = float(r.get("similarity_score")  or 0.5)
        if hours > 0:
            weighted_sum  += hours * weight
            weight_total  += weight

    if weight_total == 0.0:
        return 0.0
    return round(weighted_sum / weight_total, 2)


def resolution_time_mae(
    predicted_hours_list: Sequence[float],
    actual_hours_list:    Sequence[float],
) -> float:
    """
    Mean Absolute Error between predicted and actual resolution times.

    Parameters
    ----------
    predicted_hours_list : Sequence[float]
        Predicted resolution hours, one per test case.
    actual_hours_list : Sequence[float]
        Actual resolution hours, one per test case (from ground truth).

    Returns
    -------
    float  MAE in hours.  Lower is better.  0.0 when list is empty.
    """
    pairs = [
        (p, a) for p, a in zip(predicted_hours_list, actual_hours_list)
        if a > 0   # skip test cases with no actual data
    ]
    if not pairs:
        return 0.0
    return round(sum(abs(p - a) for p, a in pairs) / len(pairs), 2)


# ── Fix Accuracy ─────────────────────────────────────────────────────────────

def fix_accuracy_score(
    top_resolution_text: str,
    expected_keywords:   Sequence[str],
) -> float:
    """
    Binary accuracy: 1.0 if the top resolution option contains at least one
    expected keyword (case-insensitive), 0.0 otherwise.

    Parameters
    ----------
    top_resolution_text : str
        The resolution_text of the highest-ranked resolution option.
    expected_keywords : Sequence[str]
        Keywords from the ground-truth dataset entry
        (e.g. ["disk quota", "log rotation", "archive policy"]).

    Returns
    -------
    float  1.0 or 0.0.
    """
    if not top_resolution_text or not expected_keywords:
        return 0.0
    text_lower = top_resolution_text.lower()
    return 1.0 if any(kw.lower() in text_lower for kw in expected_keywords) else 0.0


def mean_fix_accuracy(scores: Sequence[float]) -> float:
    """Average fix accuracy across all test cases."""
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)
