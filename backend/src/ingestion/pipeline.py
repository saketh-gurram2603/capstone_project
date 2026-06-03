"""
Ingestion pipeline — orchestrates the full ingest flow:
  XLSX parse → BM25 index build → batch embed → Qdrant upsert

Design choices:
  • Incidents are embedded in batches of ``batch_size`` (default 50).
  • asyncio.gather is used to fire the embed_batch calls concurrently.
  • A global ``_status`` dict lets GET /ingest/status return real-time progress.
  • Qdrant point IDs are stable 32-bit integers derived from incident_id via SHA-1.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, Optional

from src.exceptions.custom_exceptions import IngestionError
from src.handlers.logger import get_logger, log_error, log_info, log_warning
from src.ingestion.bm25_builder import build_bm25_index, save_bm25_index
from src.ingestion.preprocessor import preprocess_xlsx
from src.integrations.embeddings import embed_batch
from src.integrations.vector_db import VectorStore

logger = get_logger("ingestion.pipeline")

# ── Global status tracker ─────────────────────────────────────────────────────
# Shared across coroutines in the same process; reset at each ingest call.

_status: dict[str, Any] = {
    "status": "idle",          # idle | running | completed | failed
    "total": 0,
    "ingested": 0,
    "skipped": 0,
    "started_at": None,
    "completed_at": None,
    "duration_ms": None,
    "error": None,
}


def get_ingestion_status() -> dict:
    """Return a copy of the current ingestion status dict."""
    return dict(_status)


# ── Qdrant ID derivation ──────────────────────────────────────────────────────


def incident_to_qdrant_id(incident_id: str) -> int:
    """
    Convert a string incident ID (e.g. "INC-5001") to a stable positive
    32-bit integer suitable as a Qdrant point ID.
    Uses the first 8 hex chars of SHA-1 → guarantees no collisions for
    the 150-row dataset and is reproducible across restarts.
    """
    return int(hashlib.sha1(incident_id.encode()).hexdigest()[:8], 16)


# ── Public API ─────────────────────────────────────────────────────────────────


async def run_ingestion(
    file_path: str,
    vector_store: VectorStore,
    collection: str,
    batch_size: int = 50,
    index_dir: Optional[str] = None,
) -> dict:
    """
    Full ingestion pipeline for an XLSX incident dataset.

    Parameters
    ----------
    file_path     : Path to the .xlsx file to ingest.
    vector_store  : Initialised VectorStore (Qdrant) instance.
    collection    : Qdrant collection name.
    batch_size    : Number of incidents to embed + upsert per async batch.
    index_dir     : Directory for BM25 pickle file (defaults to "data/").

    Returns
    -------
    dict  Summary: {"ingested": N, "skipped": M, "duration_ms": X}
    """
    global _status
    _reset_status()
    start_ts = time.monotonic()

    try:
        # ── 1. Parse XLSX ─────────────────────────────────────────────────────
        log_info("Pipeline: parsing XLSX | path=%s", file_path)
        incidents, skipped_labels = preprocess_xlsx(file_path)

        _status["total"] = len(incidents)
        _status["skipped"] = len(skipped_labels)
        log_info("Pipeline: parsed %d incidents, %d skipped", len(incidents), len(skipped_labels))

        # ── 2. Build BM25 index ───────────────────────────────────────────────
        log_info("Pipeline: building BM25 index")
        bm25_index, corpus, ids = build_bm25_index(incidents)
        save_bm25_index(bm25_index, corpus, ids, index_dir=index_dir)
        log_info("Pipeline: BM25 index saved")

        # ── 3. Batch embed + upsert to Qdrant ─────────────────────────────────
        log_info("Pipeline: embedding and upserting in batches of %d", batch_size)
        total_ingested = await _embed_and_upsert(
            incidents, vector_store, collection, batch_size
        )

        duration_ms = (time.monotonic() - start_ts) * 1000

        _status.update(
            status="completed",
            ingested=total_ingested,
            duration_ms=round(duration_ms, 1),
            completed_at=_now_iso(),
        )
        log_info(
            "Pipeline: completed | ingested=%d skipped=%d duration_ms=%.1f",
            total_ingested,
            len(skipped_labels),
            duration_ms,
        )
        pii_masked_total = sum(inc.get("pii_masked_count", 0) for inc in incidents)
        return {
            "ingested": total_ingested,
            "skipped": len(skipped_labels),
            "duration_ms": round(duration_ms, 1),
            "pii_masked_total": pii_masked_total,
        }

    except Exception as exc:
        duration_ms = (time.monotonic() - start_ts) * 1000
        _status.update(
            status="failed",
            error=str(exc),
            duration_ms=round(duration_ms, 1),
            completed_at=_now_iso(),
        )
        log_error("Pipeline: ingestion failed | error=%s", exc)
        raise IngestionError(f"Ingestion failed: {exc}") from exc


# ── Private helpers ───────────────────────────────────────────────────────────


async def _embed_and_upsert(
    incidents: list[dict],
    vector_store: VectorStore,
    collection: str,
    batch_size: int,
) -> int:
    """Embed incidents in async batches and upsert each batch to Qdrant."""
    total = 0
    batches = [incidents[i: i + batch_size] for i in range(0, len(incidents), batch_size)]

    for batch_idx, batch in enumerate(batches):
        texts = [inc["search_text"] for inc in batch]

        # Embed entire batch in one API call (cache-aware, fallback-aware)
        vectors = await embed_batch(texts)

        # Build Qdrant point dicts
        points = []
        for inc, vec in zip(batch, vectors):
            points.append(
                {
                    "id": incident_to_qdrant_id(inc["incident_id"]),
                    "vector": vec,
                    "payload": {
                        "incident_id":      inc["incident_id"],
                        "ticket_id":        inc.get("ticket_id", ""),
                        "title":            inc.get("title", ""),
                        "category":         inc.get("category", ""),
                        "impact":           inc.get("impact", ""),
                        "urgency":          inc.get("urgency", ""),
                        "priority":         inc.get("priority", ""),
                        "description":      inc["description"],
                        "resolution_notes": inc["resolution_notes"],
                        "assigned_to":      inc.get("assigned_to", ""),
                        "search_text":      inc["search_text"],
                        "opened_at":        inc.get("opened_at", ""),
                        "resolved_at":      inc.get("resolved_at", ""),
                        "resolution_hours": inc.get("resolution_hours", 0.0),
                    },
                }
            )

        await vector_store.upsert(collection, points)
        total += len(batch)
        _status["ingested"] = total
        log_info(
            "Pipeline: batch %d/%d upserted | running_total=%d",
            batch_idx + 1,
            len(batches),
            total,
        )

    return total


def _reset_status() -> None:
    global _status
    _status = {
        "status": "running",
        "total": 0,
        "ingested": 0,
        "skipped": 0,
        "started_at": _now_iso(),
        "completed_at": None,
        "duration_ms": None,
        "error": None,
    }


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
