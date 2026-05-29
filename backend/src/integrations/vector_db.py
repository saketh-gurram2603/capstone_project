"""
Vector database abstraction layer.
Defines a swap-able VectorStore interface — swap Qdrant for Milvus
by implementing the interface; no API or business logic changes needed.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.handlers.logger import get_logger, log_error, log_info, log_warning

logger = get_logger("integrations.vector_db")


# ── Abstract Interface ────────────────────────────────────────────────────────

class VectorStore(ABC):
    """Abstract interface for vector database operations.
    Implement this to swap the underlying vector DB without touching
    any retrieval or ingestion business logic.
    """

    @abstractmethod
    async def upsert(self, collection: str, points: list[dict]) -> int:
        """Insert or update vectors. Returns number of points upserted."""
        ...

    @abstractmethod
    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Semantic search. Returns list of {id, score, payload} dicts."""
        ...

    @abstractmethod
    async def collection_exists(self, collection: str) -> bool:
        """Return True if the collection exists and has points."""
        ...

    @abstractmethod
    async def create_collection(self, collection: str, vector_size: int) -> None:
        """Create collection if it does not exist."""
        ...

    @abstractmethod
    async def count(self, collection: str) -> int:
        """Return number of points in collection."""
        ...

    @abstractmethod
    async def scroll_all(self, collection: str) -> list[dict]:
        """Return all points (id + payload) in the collection. No vectors returned."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the database is reachable."""
        ...


# ── Qdrant Implementation ─────────────────────────────────────────────────────

class QdrantVectorStore(VectorStore):
    """
    Qdrant implementation of VectorStore.
    Uses AsyncQdrantClient for non-blocking I/O inside FastAPI.
    """

    def __init__(self, url: str, api_key: Optional[str] = None) -> None:
        self._url = url
        self._api_key = api_key or None
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self._client: AsyncQdrantClient = AsyncQdrantClient(
                url=url,
                api_key=self._api_key,
                timeout=30,
                prefer_grpc=False,   # Force REST-only; avoids port-6334 gRPC
                                     # failures on networks that block gRPC traffic
            )
        log_info("QdrantVectorStore initialised | url=%s", url)

    async def upsert(self, collection: str, points: list[dict]) -> int:
        """
        points: list of {id, vector, payload} dicts
        """
        try:
            qdrant_points = [
                PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload=p.get("payload", {}),
                )
                for p in points
            ]
            await self._client.upsert(collection_name=collection, points=qdrant_points)
            logger.info("Upserted %d points to collection '%s'", len(points), collection)
            return len(points)
        except Exception as exc:
            log_error("Qdrant upsert failed | collection=%s | error=%s", collection, exc)
            raise

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        filters: optional dict {field: value} — converted to Qdrant Filter DSL.
        Returns: list of {id, score, payload}
        """
        try:
            qdrant_filter = self._build_filter(filters) if filters else None
            results = await self._client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            return [
                {
                    "id": str(r.id),
                    "score": float(r.score),
                    "payload": r.payload or {},
                }
                for r in results
            ]
        except Exception as exc:
            log_error("Qdrant search failed | collection=%s | error=%s", collection, exc)
            raise

    async def collection_exists(self, collection: str) -> bool:
        try:
            collections = await self._client.get_collections()
            names = [c.name for c in collections.collections]
            return collection in names
        except Exception as exc:
            log_warning("collection_exists check failed | error=%s", exc)
            return False

    async def create_collection(self, collection: str, vector_size: int) -> None:
        try:
            exists = await self.collection_exists(collection)
            if not exists:
                await self._client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                log_info("Created Qdrant collection '%s' (dim=%d)", collection, vector_size)
            else:
                log_info("Qdrant collection '%s' already exists — skipping create", collection)
        except Exception as exc:
            log_error("create_collection failed | collection=%s | error=%s", collection, exc)
            raise

    async def count(self, collection: str) -> int:
        try:
            result = await self._client.count(collection_name=collection, exact=True)
            return result.count
        except Exception:
            return 0

    async def scroll_all(self, collection: str) -> list[dict]:
        """
        Page through the entire collection and return all points as
        {id, payload} dicts. Used once at startup to build the BM25 payload map.
        """
        all_points: list[dict] = []
        offset = None
        try:
            while True:
                results, next_offset = await self._client.scroll(
                    collection_name=collection,
                    limit=250,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for r in results:
                    all_points.append({"id": str(r.id), "payload": r.payload or {}})
                if next_offset is None:
                    break
                offset = next_offset
        except Exception as exc:
            log_error("scroll_all failed | collection=%s | error=%s", collection, exc)
        return all_points

    async def health_check(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception as exc:
            log_warning("Qdrant health check failed | error=%s", exc)
            return False

    @staticmethod
    def _build_filter(filters: dict) -> Filter:
        """Convert a flat {field: value} dict to a Qdrant Filter (must match all)."""
        conditions = [
            FieldCondition(key=field, match=MatchValue(value=value))
            for field, value in filters.items()
            if value is not None
        ]
        return Filter(must=conditions) if conditions else None
