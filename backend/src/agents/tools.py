"""
Agent tools — reusable async functions called by L1 and L2 nodes.

  search_kb_incidents   — run the full hybrid retrieval pipeline
  tavily_web_search     — Tavily API search (graceful if key missing)
  classify_priority     — rule-based P1-P4 from impact × urgency matrix
"""

from __future__ import annotations

import os
from typing import Optional

from src.handlers.logger import get_logger, log_info, log_warning
from src.integrations.vector_db import VectorStore
from src.retrieval.hybrid_search import hybrid_search

logger = get_logger("agents.tools")


# ── Knowledge base search ─────────────────────────────────────────────────────

async def search_kb_incidents(
    query: str,
    vector_store: VectorStore,
    collection: str,
    app_config: dict,
    filters: Optional[dict] = None,
) -> dict:
    """
    Run the full hybrid search pipeline against the incident KB.

    Returns the raw SearchResponse-compatible dict so L1 can access both
    ``results`` (ranked incidents) and ``resolution_options`` (unique fixes).
    Returns an empty result dict on any failure.
    """
    try:
        result = await hybrid_search(
            query=query,
            vector_store=vector_store,
            collection=collection,
            filters=filters,
            app_config=app_config,
        )
        log_info("KB search completed | results=%d", len(result.get("results", [])))
        return result
    except Exception as exc:
        log_warning("KB search failed | error=%s — returning empty result", exc)
        return {"results": [], "resolution_options": [], "adaptive_k_used": 0,
                "retrieval_method": "failed", "cached": False, "latency_ms": 0}


# ── Tavily web search ─────────────────────────────────────────────────────────

async def tavily_web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web via Tavily API.

    Returns a list of {title, url, content} dicts.
    Returns [] gracefully if the API key is absent or the call fails.
    """
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        log_warning("TAVILY_API_KEY not set — skipping web search")
        return []

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)

        # Tavily is synchronous; run in thread pool to avoid blocking the event loop
        import asyncio
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
            ),
        )
        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:500],  # cap content length
            }
            for r in response.get("results", [])
        ]
        log_info("Tavily search | query='%s' results=%d", query[:60], len(results))
        return results
    except Exception as exc:
        log_warning("Tavily search failed | error=%s — returning empty results", exc)
        return []


# ── Priority classification ───────────────────────────────────────────────────

_IMPACT_SCORE = {"High": 3, "Medium": 2, "Low": 1}
_URGENCY_SCORE = {"High": 3, "Medium": 2, "Low": 1}


def classify_priority(impact: str, urgency: str) -> str:
    """
    Derive P1–P4 from impact × urgency using a simple additive matrix.

    Impact  × Urgency  → Priority
    High    × High     → P1  (total 6)
    High    × Medium   → P2  (total 5)
    Medium  × Medium   → P2  (total 4)
    Low     × High     → P3  (total 4) [same total as above → P2 or P3 by impact]
    Low     × Low      → P4  (total 2)
    """
    i = _IMPACT_SCORE.get(impact, 2)
    u = _URGENCY_SCORE.get(urgency, 2)
    total = i + u

    if total >= 6:
        return "P1"
    if total >= 5:
        return "P2"
    if total >= 3:
        return "P3"
    return "P4"
