"""
Vector retriever — thin async wrapper around QdrantVectorStore.search().

Converts raw Qdrant results to the canonical {id, score, rank, payload, source}
shape used throughout the retrieval pipeline.
"""

from __future__ import annotations

from typing import Optional

from src.exceptions.custom_exceptions import VectorDBUnavailableError
from src.handlers.logger import get_logger, log_warning
from src.integrations.vector_db import VectorStore

logger = get_logger("retrieval.vector_retriever")


class VectorRetriever:
    """
    Wraps a VectorStore instance for semantic search with optional
    metadata pre-filtering.

    Instantiated once per request (the vector_store comes from FastAPI
    Depends, so there is no startup singleton needed here).
    """

    def __init__(self, vector_store: VectorStore, collection: str) -> None:
        self._store = vector_store
        self._collection = collection

    async def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Perform semantic search.

        Parameters
        ----------
        query_vector : Embedded query (1536-dim for ada-002).
        top_k        : Maximum number of results to return.
        filters      : Optional {field: value} dict — passed to Qdrant filter DSL.

        Returns
        -------
        list[dict]  Each item: {id, score, rank, payload, source="vector"}

        Raises
        ------
        VectorDBUnavailableError  if Qdrant raises any exception.
        """
        try:
            raw = await self._store.search(
                collection=self._collection,
                query_vector=query_vector,
                top_k=top_k,
                filters=filters,
            )
        except Exception as exc:
            log_warning("Qdrant search failed | error=%s", exc)
            raise VectorDBUnavailableError(reason=str(exc)) from exc

        results = []
        for rank, hit in enumerate(raw):
            results.append(
                {
                    "id": hit["id"],
                    "score": float(hit["score"]),
                    "rank": rank,
                    "payload": hit.get("payload", {}),
                    "source": "vector",
                }
            )

        logger.debug(
            "Vector search | collection=%s top_k=%d returned=%d",
            self._collection,
            top_k,
            len(results),
        )
        return results
