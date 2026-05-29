"""
Classical Information Retrieval metrics — pure Python, zero dependencies.

All functions operate on ranked lists and return float scores in [0.0, 1.0].

Exported functions:
  ndcg_at_k(retrieved_ids, relevant_ids, k)   → NDCG@k
  map_at_k(retrieved_ids, relevant_ids, k)    → MAP@k
  recall_at_k(retrieved_ids, relevant_ids, k) → Recall@k
  precision_at_k(retrieved_ids, relevant_ids, k) → P@k
  compute_all_metrics(retrieved_ids, relevant_ids, k) → dict of all four
"""

from __future__ import annotations

import math
from typing import Sequence


# ── NDCG@k ───────────────────────────────────────────────────────────────────


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int = 10,
) -> float:
    """
    Normalised Discounted Cumulative Gain at k.

    Uses binary relevance (1 if in relevant_ids, 0 otherwise).
    Returns 0.0 when relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    top_k = list(retrieved_ids)[:k]

    # DCG: sum of rel_i / log2(i+2) for i in 0..k-1
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, doc_id in enumerate(top_k)
        if doc_id in relevant_set
    )

    # IDCG: best possible DCG — all relevant docs ranked first
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ── MAP@k ─────────────────────────────────────────────────────────────────────


def map_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int = 10,
) -> float:
    """
    Mean Average Precision at k (single query).

    Computes the average of precision values at each rank position
    where a relevant document is found within the top-k results.
    Returns 0.0 when relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    top_k = list(retrieved_ids)[:k]

    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(top_k, 1):
        if doc_id in relevant_set:
            hits += 1
            precision_sum += hits / rank

    num_relevant = min(len(relevant_set), k)
    if num_relevant == 0:
        return 0.0
    return precision_sum / num_relevant


# ── Recall@k ──────────────────────────────────────────────────────────────────


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int = 10,
) -> float:
    """
    Fraction of relevant documents retrieved within the top-k results.
    Returns 0.0 when relevant_ids is empty.
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)
    top_k = set(list(retrieved_ids)[:k])
    hits = len(top_k & relevant_set)
    return hits / len(relevant_set)


# ── Precision@k ───────────────────────────────────────────────────────────────


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int = 10,
) -> float:
    """
    Fraction of top-k retrieved documents that are relevant.
    Returns 0.0 when k == 0.
    """
    if k == 0:
        return 0.0

    relevant_set = set(relevant_ids)
    top_k = list(retrieved_ids)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(top_k)


# ── Convenience wrapper ───────────────────────────────────────────────────────


def compute_all_metrics(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int = 10,
) -> dict[str, float]:
    """
    Compute NDCG@k, MAP@k, Recall@k, and Precision@k in one call.

    Returns:
        {
            "ndcg_at_k":      float,
            "map_at_k":       float,
            "recall_at_k":    float,
            "precision_at_k": float,
        }
    """
    return {
        "ndcg_at_k": ndcg_at_k(retrieved_ids, relevant_ids, k),
        "map_at_k": map_at_k(retrieved_ids, relevant_ids, k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
    }
