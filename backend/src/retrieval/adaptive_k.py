"""
Adaptive-K selector.

Decides how many candidates to fetch from BM25 and Qdrant based on query
complexity.  A simpler query gets a small k (fast path); a vague or
error-code-heavy query gets a large k to maximise recall.

compute_k  — choose k before retrieval
trim_by_score_dropoff — prune the fused ranked list by score discontinuity
"""

from __future__ import annotations

import re

from src.handlers.logger import get_logger

logger = get_logger("retrieval.adaptive_k")

# Default thresholds (overridden by values from app_config at runtime)
_COMPLEXITY_LOW = 5     # below → k_min
_COMPLEXITY_MID = 12    # below → k_default


# ── Public API ─────────────────────────────────────────────────────────────────


def compute_k(
    query: str,
    k_min: int = 3,
    k_default: int = 10,
    k_max: int = 20,
) -> int:
    """
    Return the number of candidates to retrieve for this query.

    Complexity scoring:
      + 1 per token
      + 3 per token that contains a digit (error codes: 502, OOM, etc.)
      + 2 flat bonus if token count > 10 (long/compound query)

    k_min     → simple, specific query (e.g. "disk full")
    k_default → typical medium query
    k_max     → vague, rare, or multi-concept query
    """
    tokens = query.strip().lower().split()
    token_count = len(tokens)

    # Tokens that contain at least one digit → likely error codes / metrics
    digit_tokens = sum(1 for t in tokens if re.search(r"\d", t))

    complexity = token_count + 3 * digit_tokens + (2 if token_count > 10 else 0)

    if complexity < _COMPLEXITY_LOW:
        k = k_min
    elif complexity < _COMPLEXITY_MID:
        k = k_default
    else:
        k = k_max

    logger.debug(
        "compute_k | query_len=%d digit_tokens=%d complexity=%d → k=%d",
        token_count,
        digit_tokens,
        complexity,
        k,
    )
    return k


def trim_by_score_dropoff(
    results: list[dict],
    threshold: float = 0.15,
) -> list[dict]:
    """
    Remove tail results where the relative score drop from the previous
    result exceeds ``threshold``.

    Example: scores [0.9, 0.85, 0.83, 0.60, 0.58]
      Drop at position 3: (0.83 - 0.60) / 0.83 = 0.277 > 0.15 → STOP
      Returns first 3 results.

    Rules:
      • Always keeps results[0] (the top hit).
      • If prev_score == 0, always keeps the next result.
      • Returns the full list unchanged if len(results) ≤ 1.
    """
    if len(results) <= 1:
        return list(results)

    trimmed = [results[0]]
    for result in results[1:]:
        prev_score = trimmed[-1].get("score", 0.0)
        curr_score = result.get("score", 0.0)

        if prev_score > 0:
            relative_drop = (prev_score - curr_score) / prev_score
            if relative_drop > threshold:
                break

        trimmed.append(result)

    logger.debug(
        "trim_by_score_dropoff | before=%d after=%d threshold=%.2f",
        len(results),
        len(trimmed),
        threshold,
    )
    return trimmed
