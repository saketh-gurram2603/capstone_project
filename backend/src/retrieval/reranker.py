"""
Cross-encoder reranker.

Stage-2 scoring: after BM25 + vector retrieval + RRF + trim, this module
re-scores each (query, document) pair with a cross-encoder.  Cross-encoders
attend to BOTH query and document jointly, giving much higher accuracy than
bi-encoder dot-product similarity.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  • ~22MB — fast on CPU (~5ms per pair)
  • Pre-trained for passage ranking on MS MARCO
  • Returns raw logit scores (unbounded) — we sigmoid-normalise to [0, 1]

The module exposes a singleton pattern identical to embeddings.py and llm.py.
Call init_reranker() once in the FastAPI lifespan; then call rerank() freely.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from src.handlers.logger import get_logger, log_info, log_warning

# sentence_transformers imported lazily inside init_reranker() so a
# torch/torchvision ABI mismatch never crashes the app at startup.

logger = get_logger("retrieval.reranker")

# ── Singleton state ───────────────────────────────────────────────────────────
_cross_encoder = None   # CrossEncoder — lazy-loaded
_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ── Startup initialisation ────────────────────────────────────────────────────

def init_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> None:
    """
    Load the cross-encoder into the module-level singleton.
    Must be called at startup (FastAPI lifespan) before any search request.
    """
    global _cross_encoder, _model_name
    _model_name = model_name
    log_info("Loading cross-encoder reranker '%s' ...", model_name)
    try:
        from sentence_transformers import CrossEncoder  # lazy import
        _cross_encoder = CrossEncoder(model_name)
        log_info("Cross-encoder reranker loaded successfully | model=%s", model_name)
    except Exception as exc:
        log_warning(
            "Cross-encoder reranker could not be loaded (torch/torchvision issue?) "
            "— hybrid search will use RRF order without reranking | error=%s",
            exc,
        )
        _cross_encoder = None


# ── Core reranking ────────────────────────────────────────────────────────────

def rerank(query: str, candidates: list[dict]) -> list[dict]:
    """
    Re-score ``candidates`` with the cross-encoder and return them sorted
    by descending rerank score.

    Each candidate dict is mutated in place with two new keys:
      ``rerank_score``      : raw logit (float, unbounded)
      ``similarity_score``  : sigmoid-normalised score in (0, 1)

    If the reranker is not loaded, candidates are returned unchanged.

    Parameters
    ----------
    query      : The user's natural-language query.
    candidates : list[dict] with 'payload' containing 'search_text'.
    """
    if not candidates:
        return candidates

    if _cross_encoder is None:
        log_warning("Reranker not loaded — returning candidates in original order")
        _add_fallback_scores(candidates)
        return candidates

    # Build (query, document_text) pairs — use search_text; fall back to description
    pairs = [
        (query, _get_doc_text(c))
        for c in candidates
    ]

    raw_scores = _cross_encoder.predict(pairs)
    raw_floats = [float(s) for s in raw_scores]

    # Assign raw rerank scores + an ABSOLUTE confidence.
    # rerank_confidence = sigmoid(raw_logit): batch-independent and calibrated by
    # the cross-encoder itself. Empirically ms-marco-MiniLM gives ~+3.5 for a
    # relevant (query, incident) pair (sigmoid≈0.97) and ~-11 for an irrelevant
    # one (sigmoid≈0.0), so this is a reliable gate signal for L1 triage — unlike
    # the batch min-max similarity_score below, which is for DISPLAY only.
    for candidate, raw in zip(candidates, raw_floats):
        candidate["rerank_score"] = raw
        candidate["rerank_confidence"] = _sigmoid(raw)

    # Sort by raw logit descending — this is the actual rerank
    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    # ── Similarity score: batch min-max → [0.30, 1.00] ───────────────────────
    # We use relative normalisation rather than raw sigmoid because ms-marco
    # logits for short incident text cluster around -5 to -7 (sigmoid ≈ 0.001),
    # making all scores display as "0%".  Since cross-encoder is used here for
    # *ranking* within the retrieved set (not threshold filtering), what matters
    # is relative order.  Map the batch linearly so best → 1.0, worst → 0.30.
    lo, hi = min(raw_floats), max(raw_floats)
    span = hi - lo
    for candidate in reranked:
        raw = candidate["rerank_score"]
        if span > 1e-6:
            t = (raw - lo) / span          # t ∈ [0, 1]
            semantic = round(0.30 + t * 0.70, 4)
        else:
            semantic = 1.0

        # ── Recency blend ─────────────────────────────────────────────────────
        # Blend recency into the DISPLAY score only.
        # rerank_confidence (L1 gate) stays as pure semantic sigmoid — untouched.
        # similarity_score (display + resolution_aggregator weighting) gets a
        # small recency boost: recent proven fixes rank slightly higher when
        # semantic relevance is otherwise equal.
        #   recency = exp(-days_since_resolved / 365)
        #   1 month ago  → 0.92   (fresh)
        #   6 months ago → 0.61   (moderate)
        #   2 years ago  → 0.17   (older)
        recency = _recency_score(candidate.get("payload", {}).get("resolved_at", ""))
        candidate["recency_score"]    = recency
        candidate["similarity_score"] = round(0.75 * semantic + 0.25 * recency, 4)

    # Re-sort with blended score so recency affects the final result order
    reranked = sorted(reranked, key=lambda c: c["similarity_score"], reverse=True)

    logger.debug(
        "Reranker scored %d candidates | top_score=%.4f top_similarity=%.4f top_recency=%.4f",
        len(reranked),
        reranked[0]["rerank_score"] if reranked else 0.0,
        reranked[0].get("similarity_score", 0.0) if reranked else 0.0,
        reranked[0].get("recency_score", 0.0) if reranked else 0.0,
    )
    return reranked


def is_reranker_loaded() -> bool:
    """Return True if the cross-encoder has been loaded."""
    return _cross_encoder is not None


# ── Private helpers ───────────────────────────────────────────────────────────

def _recency_score(resolved_at_iso: str) -> float:
    """
    Exponential decay based on days since the incident was resolved.
    Returns a value in (0, 1] — 1.0 for very recent, approaches 0 for old.
    Falls back to 0.5 (neutral) when the timestamp is absent or unparseable.
    """
    if not resolved_at_iso:
        return 0.5
    try:
        resolved = datetime.fromisoformat(resolved_at_iso.replace("Z", "+00:00"))
        days = max(0, (datetime.now(timezone.utc) - resolved).days)
        return round(math.exp(-days / 365.0), 4)
    except (ValueError, TypeError):
        return 0.5


def _get_doc_text(candidate: dict) -> str:
    """Extract the best text field from a candidate's payload."""
    payload = candidate.get("payload", {})
    return (
        payload.get("search_text")
        or payload.get("description")
        or ""
    )


def _sigmoid(x: float) -> float:
    """Map a raw logit to (0, 1). Handles extreme values safely."""
    try:
        return round(1.0 / (1.0 + math.exp(-x)), 6)
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _add_fallback_scores(candidates: list[dict]) -> None:
    """
    When reranker is unavailable, assign descending scores based on
    RRF rank position so downstream code always finds similarity_score.
    """
    n = len(candidates)
    for i, c in enumerate(candidates):
        c.setdefault("rerank_score", float(n - i))
        # Normalise linearly: 1st → 1.0, last → ~0.5 for up to 20 candidates
        score = round(1.0 - (i / (2 * n)), 4)
        c.setdefault("similarity_score", score)
        # No cross-encoder logit available — use the positional score as a
        # best-effort confidence so the L1 gate still has a value to read.
        c.setdefault("rerank_confidence", score)
