"""
Embedding integration.
Primary  : OpenAI text-embedding-ada-002
Fallback : sentence-transformers/all-MiniLM-L6-v2 (local, loaded at startup)

Redis cache removed — no per-request caching overhead.
"""

import asyncio
from typing import Optional

from openai import AsyncOpenAI

from src.handlers.logger import get_logger, log_error, log_info, log_warning

logger = get_logger("integrations.embeddings")

# ── Module-level state ────────────────────────────────────────────────────────
_openai_client: Optional[AsyncOpenAI] = None
_local_model = None          # SentenceTransformer — lazy-loaded at startup
_embedding_model_name: str = "text-embedding-ada-002"
_fallback_model_name: str = "all-MiniLM-L6-v2"


def init_embeddings(
    openai_api_key: str,
    embedding_model: str = "text-embedding-ada-002",
    fallback_model: str = "all-MiniLM-L6-v2",
    embedding_ttl: int = 86400,   # kept for API compat, unused
) -> None:
    """Load local fallback model and configure OpenAI client. Called at startup."""
    global _openai_client, _local_model, _embedding_model_name, _fallback_model_name

    _embedding_model_name = embedding_model
    _fallback_model_name = fallback_model

    _openai_client = AsyncOpenAI(api_key=openai_api_key)
    log_info("OpenAI embedding client initialised | model=%s", embedding_model)

    log_info("Loading local fallback embedding model '%s' ...", fallback_model)
    try:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(fallback_model)
        log_info("Local embedding model '%s' loaded successfully", fallback_model)
    except Exception as exc:
        log_warning(
            "Local embedding model '%s' could not be loaded — will rely on OpenAI only | error=%s",
            fallback_model, exc,
        )
        _local_model = None


# ── Primary embedding (OpenAI) ────────────────────────────────────────────────

async def embed_text(text: str) -> list[float]:
    """
    Embed a single text string via OpenAI ada-002.
    Falls back to local MiniLM if OpenAI fails.
    """
    if _openai_client:
        try:
            response = await asyncio.wait_for(
                _openai_client.embeddings.create(
                    model=_embedding_model_name,
                    input=text,
                ),
                timeout=30.0,
            )
            return response.data[0].embedding
        except Exception as exc:
            log_warning("OpenAI embed_text failed, falling back to local | error=%s", exc)

    return _embed_local(text)


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of non-empty texts in one OpenAI API call.
    Falls back to local model per-text if OpenAI fails.
    Caller is responsible for filtering out empty strings.
    """
    if not texts:
        return []

    if _openai_client:
        try:
            response = await asyncio.wait_for(
                _openai_client.embeddings.create(
                    model=_embedding_model_name,
                    input=texts,
                ),
                timeout=60.0,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            log_warning("OpenAI embed_batch failed, using local fallback | error=%s", exc)

    return [_embed_local(t) for t in texts]


# ── Local-only embedding (for clustering / non-search tasks) ──────────────────

def embed_local_batch_sync(texts: list[str]) -> list[list[float]]:
    """
    Synchronous local embedding using the loaded MiniLM model.
    Used by resolution_aggregator so it avoids a second OpenAI call.
    Returns empty list if local model is not available.
    """
    if not texts:
        return []
    if _local_model is None:
        raise RuntimeError(
            "Local embedding model not loaded. Cannot compute local embeddings."
        )
    vectors = _local_model.encode(texts, normalize_embeddings=True, batch_size=32)
    return [v.tolist() for v in vectors]


# ── Private ───────────────────────────────────────────────────────────────────

def _embed_local(text: str) -> list[float]:
    """Synchronous single-text local embedding."""
    if _local_model is None:
        raise RuntimeError(
            "Local embedding model unavailable. "
            "Check that torch and sentence-transformers are correctly installed."
        )
    vector = _local_model.encode(text, normalize_embeddings=True)
    return vector.tolist()
