"""
Reciprocal Rank Fusion (RRF) merger.

RRF score for a document d:
    score(d) = Σ  1 / (k + rank_i(d))
               i

where rank_i(d) is the 1-based position of d in retrieval list i,
and k = 60 (standard value, controls sensitivity to top-ranks).

Properties:
  • Rank-only: individual scores from BM25/vector are ignored, only rank matters.
  • No score normalisation needed across heterogeneous retrieval methods.
  • A document ranked 1st in BOTH lists scores roughly 2 × a doc in one list.
  • Robust to score distribution differences between retrieval methods.
"""

from __future__ import annotations

from src.handlers.logger import get_logger

logger = get_logger("retrieval.rrf_merger")


def reciprocal_rank_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Fuse two ranked result lists using RRF.

    Parameters
    ----------
    bm25_results   : Ranked list from BM25Retriever  — items with 'id' + 'payload'.
    vector_results : Ranked list from VectorRetriever — items with 'id' + 'payload'.
    k              : RRF constant (default 60, per Robertson 2009).

    Returns
    -------
    list[dict]  Merged and re-ranked by RRF score (descending).
                Each item: {id, score (RRF), payload, sources (set)}
    """
    rrf_scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    sources: dict[str, set] = {}

    for rank, doc in enumerate(bm25_results):
        doc_id = str(doc["id"])
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        payloads.setdefault(doc_id, doc.get("payload", {}))
        sources.setdefault(doc_id, set()).add("bm25")

    for rank, doc in enumerate(vector_results):
        doc_id = str(doc["id"])
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        # Prefer vector payload (richer metadata from Qdrant)
        if doc.get("payload"):
            payloads[doc_id] = doc["payload"]
        else:
            payloads.setdefault(doc_id, {})
        sources.setdefault(doc_id, set()).add("vector")

    merged = [
        {
            "id": doc_id,
            "score": score,
            "payload": payloads[doc_id],
            "sources": sources[doc_id],
        }
        for doc_id, score in sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )
    ]

    logger.debug(
        "RRF fusion | bm25=%d vector=%d merged=%d",
        len(bm25_results),
        len(vector_results),
        len(merged),
    )
    return merged
