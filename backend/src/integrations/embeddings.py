"""
Embedding integration — Azure OpenAI backend.
Primary  : Azure text-embedding-ada-002 (deployment: synapt-dev-text-embedding-ada-002)
Fallback : sentence-transformers/all-MiniLM-L6-v2 (local, loaded at startup)

Note: the embedding endpoint uses a *different* API version from the chat endpoint
(2024-05-01-preview vs 2025-01-01-preview). Two separate ``AsyncAzureOpenAI``
clients are therefore NOT needed — the api_version is set at construction time and
applies to every call made on that client. We configure the embedding client with
its own version at startup.
"""

import asyncio
from typing import Optional

from openai import AsyncAzureOpenAI

from src.handlers.logger import get_logger, log_error, log_info, log_warning

logger = get_logger("integrations.embeddings")

# ── Module-level state ────────────────────────────────────────────────────────
_openai_client: Optional[AsyncAzureOpenAI] = None
_local_model = None          # SentenceTransformer — lazy-loaded at startup
_embedding_model_name: str = "synapt-dev-text-embedding-ada-002"
_fallback_model_name: str = "all-MiniLM-L6-v2"
# Dimension of the Qdrant collection (ada-002 = 1536). The local fallback model
# (all-MiniLM-L6-v2) emits 384-dim vectors; any fallback vector sent to Qdrant
# MUST be fitted to this dimension or Qdrant rejects/corrupts the index.
_expected_dim: int = 1536


def init_embeddings(
    azure_api_key: str,
    azure_endpoint: str,
    azure_api_version: str,
    embedding_model: str = "synapt-dev-text-embedding-ada-002",
    fallback_model: str = "all-MiniLM-L6-v2",
    embedding_ttl: int = 86400,   # kept for API compat, unused
    expected_dim: int = 1536,
) -> None:
    """Load local fallback model and configure Azure OpenAI embedding client."""
    global _openai_client, _local_model, _embedding_model_name, _fallback_model_name
    global _expected_dim

    _embedding_model_name = embedding_model
    _fallback_model_name = fallback_model
    _expected_dim = expected_dim

    _openai_client = AsyncAzureOpenAI(
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        api_version=azure_api_version,
    )
    log_info(
        "Azure OpenAI embedding client initialised | endpoint=%s version=%s deployment=%s",
        azure_endpoint, azure_api_version, embedding_model,
    )

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
    """
    Synchronous single-text local embedding, fitted to the Qdrant collection
    dimension so a fallback vector can be safely searched/upserted.
    """
    if _local_model is None:
        raise RuntimeError(
            "Local embedding model unavailable. "
            "Check that torch and sentence-transformers are correctly installed."
        )
    vector = _local_model.encode(text, normalize_embeddings=True)
    return _fit_dim(vector.tolist())


def _fit_dim(vector: list[float]) -> list[float]:
    """
    Pad (with zeros) or truncate a vector to ``_expected_dim``.

    The local fallback model emits 384-dim vectors while the Qdrant collection
    is 1536-dim (ada-002). Without this, a fallback embedding makes Qdrant raise
    on query and corrupts the index on ingest. Zero-padding keeps a unit vector
    unit-norm; vector-search quality is degraded under fallback (logged as a
    warning) but the system stays operational instead of crashing.
    """
    n = len(vector)
    if n == _expected_dim:
        return vector
    log_warning(
        "Local fallback embedding dim=%d != collection dim=%d — fitting "
        "(vector search quality is degraded until OpenAI embeddings recover).",
        n, _expected_dim,
    )
    if n > _expected_dim:
        return vector[:_expected_dim]
    return vector + [0.0] * (_expected_dim - n)
