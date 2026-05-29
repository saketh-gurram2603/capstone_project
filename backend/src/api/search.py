"""
Search API — POST /search

Accepts a natural-language query with optional metadata filters and returns
ranked historical incidents plus deduplicated resolution options.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.core.dependencies import get_app_config, get_vector_store
from src.exceptions.custom_exceptions import RetrievalError
from src.handlers.logger import get_logger, log_error, log_info
from src.integrations.vector_db import VectorStore
from src.models.search import SearchQuery, SearchResponse
from src.retrieval.hybrid_search import hybrid_search

logger = get_logger("api.search")

router = APIRouter(tags=["Search"])


@router.post("/search", response_model=SearchResponse)
async def search_incidents(
    request: SearchQuery,
    vector_store: VectorStore = Depends(get_vector_store),
    app_config: dict = Depends(get_app_config),
) -> SearchResponse:
    """
    Hybrid semantic + keyword search over ingested incidents.

    Pipeline: Adaptive-K → BM25 + Qdrant (parallel) → RRF → trim →
              cross-encoder rerank → resolution aggregation

    Modes:
      - hybrid  (default): BM25 keyword + Qdrant vector, fused via RRF
      - semantic          : Qdrant vector-only (BM25 skipped)

    Graceful degradation:
      - Qdrant unavailable → BM25-only
      - BM25 index missing → vector-only
      - Both fail → 503
    """
    query = request.query.strip()
    log_info("POST /search | query='%s' top_k=%s", query[:80], request.top_k)

    # Convert Pydantic filter model to plain dict for the pipeline
    filters: dict | None = None
    if request.filters:
        raw = request.filters.model_dump(exclude_none=True)
        filters = {k: v.value if hasattr(v, "value") else v for k, v in raw.items()} or None

    collection = app_config.get("QDRANT", {}).get("COLLECTION_NAME", "incidents")

    try:
        result = await hybrid_search(
            query=query,
            vector_store=vector_store,
            collection=collection,
            filters=filters,
            top_k_override=request.top_k,
            app_config=app_config,
            mode=request.mode,
        )
    except RetrievalError as exc:
        raise HTTPException(status_code=503, detail=exc.message) from exc
    except Exception as exc:
        log_error("Unexpected search error | error=%s | type=%s", exc, type(exc).__name__)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {type(exc).__name__}: {exc}",
        ) from exc

    return SearchResponse(**result)
