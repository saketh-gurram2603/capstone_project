"""
Embedding integration — Azure OpenAI backend.

Primary : Azure text-embedding-ada-002 (deployment: synapt-dev-text-embedding-ada-002)
          → used for all Qdrant search queries and ingest vectors.

Local   : sentence-transformers/all-MiniLM-L6-v2 (loaded at startup, 384-dim)
          → used ONLY by resolution_aggregator to cluster resolution texts against
            each other. It never touches Qdrant — dimensions don't need to match.

Fallback strategy for Qdrant paths (embed_text / embed_batch):
  Azure ada-002 fails → raise immediately.
  The caller (hybrid_search.py) catches the exception, sets query_vector=None,
  and falls back to BM25-only search — an honest degradation that returns real
  keyword results. This is far better than zero-padding a 384-dim MiniLM vector
  to 1536-dim and letting Qdrant compute garbage cosine similarities.

  For ingest (pipeline.py), the exception propagates to run_ingestion's
  try/except, which marks the job as failed with a clear error message — also
  correct, because storing bad vectors would silently corrupt the index.
"""

import asyncio
from typing import Optional

from openai import AsyncAzureOpenAI

from src.handlers.logger import get_logger, log_info, log_warning

logger = get_logger("integrations.embeddings")

# ── Module-level state ────────────────────────────────────────────────────────
_openai_client: Optional[AsyncAzureOpenAI] = None
_local_model    = None          # SentenceTransformer — for resolution clustering only
_embedding_model_name: str = "synapt-dev-text-embedding-ada-002"
_fallback_model_name:  str = "all-MiniLM-L6-v2"


def init_embeddings(
    azure_api_key: str,
    azure_endpoint: str,
    azure_api_version: str,
    embedding_model: str = "synapt-dev-text-embedding-ada-002",
    fallback_model: str  = "all-MiniLM-L6-v2",
    embedding_ttl: int   = 86400,   # kept for API compat, unused
) -> None:
    """
    Configure the Azure OpenAI embedding client and load the local MiniLM model.
    Called once from the FastAPI lifespan at startup.

    The local model is loaded here so resolution_aggregator never pays a
    cold-start penalty on the first request.
    """
    global _openai_client, _local_model, _embedding_model_name, _fallback_model_name

    _embedding_model_name = embedding_model
    _fallback_model_name  = fallback_model

    _openai_client = AsyncAzureOpenAI(
        api_key=azure_api_key,
        azure_endpoint=azure_endpoint,
        api_version=azure_api_version,
    )
    log_info(
        "Azure OpenAI embedding client initialised | endpoint=%s version=%s deployment=%s",
        azure_endpoint, azure_api_version, embedding_model,
    )

    # Load local model for resolution clustering (NOT for Qdrant queries)
    log_info("Loading local clustering model '%s' ...", fallback_model)
    try:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer(fallback_model)
        log_info("Local clustering model '%s' loaded successfully", fallback_model)
    except Exception as exc:
        log_warning(
            "Local clustering model '%s' could not be loaded — "
            "resolution deduplication will be unavailable | error=%s",
            fallback_model, exc,
        )
        _local_model = None


# ── Azure embedding (for Qdrant queries and ingest) ───────────────────────────

async def embed_text(text: str) -> list[float]:
    """
    Embed a single text string via Azure ada-002.

    Raises on failure — the caller (hybrid_search.py) handles this by
    setting query_vector=None and falling back to BM25-only search.
    """
    if _openai_client is None:
        raise RuntimeError("Embedding client not initialised. Call init_embeddings() at startup.")

    response = await asyncio.wait_for(
        _openai_client.embeddings.create(
            model=_embedding_model_name,
            input=text,
        ),
        timeout=30.0,
    )
    return response.data[0].embedding


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts in one Azure API call.

    Raises on failure — the caller (pipeline.py / ingest) propagates the
    error rather than storing zero-padded garbage vectors in Qdrant.
    Caller is responsible for filtering out empty strings before calling.
    """
    if not texts:
        return []

    if _openai_client is None:
        raise RuntimeError("Embedding client not initialised. Call init_embeddings() at startup.")

    response = await asyncio.wait_for(
        _openai_client.embeddings.create(
            model=_embedding_model_name,
            input=texts,
        ),
        timeout=60.0,
    )
    return [item.embedding for item in response.data]


# ── Local-only embedding (resolution clustering, never touches Qdrant) ─────────

def embed_local_batch_sync(texts: list[str]) -> list[list[float]]:
    """
    Synchronous local embedding using the loaded MiniLM model (384-dim).

    Used exclusively by resolution_aggregator to cluster resolution texts
    against each other via cosine similarity. These vectors are never stored
    in or queried against Qdrant, so the 384-dim output is perfectly valid.
    """
    if not texts:
        return []
    if _local_model is None:
        raise RuntimeError(
            "Local clustering model not loaded. Cannot deduplicate resolutions."
        )
    vectors = _local_model.encode(texts, normalize_embeddings=True, batch_size=32)
    return [v.tolist() for v in vectors]
