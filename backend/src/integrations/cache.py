"""
cache.py — Redis removed. All functions are no-ops.
Kept as stubs so existing imports don't break.
"""

from typing import Any, Optional


def init_cache(*args, **kwargs) -> None:
    pass


def get_cache_client():
    return None


async def cache_get(key: str) -> Optional[Any]:
    return None


async def cache_set(key: str, value: Any, ttl_seconds: int) -> bool:
    return False


async def cache_delete(key: str) -> None:
    pass


async def health_check() -> bool:
    return False


def embedding_cache_key(text: str, model: str) -> str:
    import hashlib
    digest = hashlib.md5(f"{model}:{text}".encode()).hexdigest()
    return f"emb:{digest}"


def query_cache_key(query: str, filters: Optional[dict]) -> str:
    import hashlib, json
    payload = json.dumps({"q": query.lower().strip(), "f": filters or {}}, sort_keys=True)
    digest = hashlib.md5(payload.encode()).hexdigest()
    return f"search:{digest}"
