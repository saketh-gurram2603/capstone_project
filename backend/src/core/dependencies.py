"""
FastAPI dependency injection.
All services injected via Depends() so endpoints are testable.
"""

import os
from typing import Optional

from fastapi import Header, HTTPException, Request

from src.integrations.vector_db import VectorStore


def get_app_config(request: Request) -> dict:
    """Inject static app config (from app_config.json)."""
    return request.app.state.app_config


def get_env_config(request: Request) -> dict:
    """Inject environment config (from config.json[env])."""
    return request.app.state.env_config


def get_vector_store(request: Request) -> VectorStore:
    """Inject the VectorStore implementation (QdrantVectorStore)."""
    return request.app.state.vector_store


async def require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> None:
    """
    Optional API-key gate for expensive endpoints (/ingest, /evaluate).

    If the INTERNAL_API_KEY environment variable is set, every protected
    request must include a matching ``X-API-Key`` header.  When the variable
    is absent (default in development) the check is skipped entirely so local
    runs work without configuration.
    """
    expected = os.getenv("INTERNAL_API_KEY")
    if not expected:
        return  # dev mode — no key required
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
