"""
Ingestion API
  POST /ingest        — upload an XLSX file, trigger ingestion pipeline
  GET  /ingest/status — poll current ingestion job status
"""

from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.core.dependencies import get_app_config, get_vector_store, require_api_key
from src.exceptions.custom_exceptions import (
    EmptyDatasetError,
    IngestionError,
    InvalidFileFormatError,
)
from src.handlers.logger import get_logger, log_error, log_info, log_warning
from src.ingestion.pipeline import get_ingestion_status, run_ingestion
from src.integrations.vector_db import VectorStore
from src.retrieval.bm25_retriever import load_bm25_retriever, set_bm25_payload_map

logger = get_logger("api.ingestion")

router = APIRouter(tags=["Ingestion"])


# ── Response schemas ──────────────────────────────────────────────────────────


class IngestResponse(BaseModel):
    status: str
    message: str
    ingested: int = 0
    skipped: int = 0
    duration_ms: float = 0.0
    pii_masked_total: int = 0


class IngestStatusResponse(BaseModel):
    status: str             # idle | running | completed | failed
    total: int
    ingested: int
    skipped: int
    started_at: str | None
    completed_at: str | None
    duration_ms: float | None
    error: str | None


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("/ingest", response_model=IngestResponse, status_code=200)
async def ingest_incidents(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="XLSX incident dataset to ingest"),
    vector_store: VectorStore = Depends(get_vector_store),
    app_config: dict = Depends(get_app_config),
    _: None = Depends(require_api_key),
) -> IngestResponse:
    """
    Upload an XLSX file and ingest all incident records into Qdrant + BM25 index.

    The ingestion runs synchronously in the request so callers know when it
    finishes.  For large datasets the caller can instead use
    ``background_tasks`` — swap to ``background_tasks.add_task(...)`` and
    return 202 Accepted.
    """
    filename = file.filename or "upload.xlsx"
    log_info("POST /ingest | filename=%s size=%s", filename, file.size)

    # ── Validate file extension ───────────────────────────────────────────────
    if not filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file format: '{filename}'. Only XLSX files are accepted.",
        )

    # ── Enforce file size limit (before reading content into memory) ─────────
    max_mb = app_config.get("INGESTION", {}).get("MAX_FILE_SIZE_MB", 50)
    if file.size and file.size > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {file.size / (1024 * 1024):.1f} MB. "
                f"Maximum allowed: {max_mb} MB."
            ),
        )

    # ── Save upload to temp file ──────────────────────────────────────────────
    try:
        content = await file.read()
        suffix = ".xlsx" if filename.lower().endswith(".xlsx") else ".xls"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as exc:
        log_error("Failed to save upload to temp file | error=%s", exc)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.") from exc

    # ── Run ingestion pipeline ────────────────────────────────────────────────
    try:
        collection = app_config.get("QDRANT", {}).get("COLLECTION_NAME", "incidents")
        result = await run_ingestion(
            file_path=tmp_path,
            vector_store=vector_store,
            collection=collection,
        )
        # ── Reload BM25 index + payload map so new data is immediately searchable ─
        try:
            load_bm25_retriever(index_dir="data")
            all_points = await vector_store.scroll_all(collection)
            payload_map = {
                p["payload"]["incident_id"]: p["payload"]
                for p in all_points
                if p.get("payload", {}).get("incident_id")
            }
            set_bm25_payload_map(payload_map)
            log_info("BM25 index reloaded after ingestion | docs=%d", len(payload_map))
        except Exception as _bm25_exc:
            log_warning(
                "BM25 reload after ingestion failed — keyword search may be stale "
                "until server restart. | error=%s", _bm25_exc,
            )

        return IngestResponse(
            status="completed",
            message=f"Ingestion finished. {result['ingested']} records upserted.",
            ingested=result["ingested"],
            skipped=result["skipped"],
            duration_ms=result["duration_ms"],
            pii_masked_total=result.get("pii_masked_total", 0),
        )
    except InvalidFileFormatError as exc:
        raise HTTPException(status_code=422, detail=exc.message) from exc
    except EmptyDatasetError as exc:
        raise HTTPException(status_code=422, detail=exc.message) from exc
    except IngestionError as exc:
        raise HTTPException(status_code=500, detail=exc.message) from exc
    except Exception as exc:
        log_error("Unexpected ingestion error | error=%s", exc)
        raise HTTPException(status_code=500, detail="Ingestion failed unexpectedly.") from exc
    finally:
        # Clean up temp file regardless of success/failure
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@router.get("/ingest/status", response_model=IngestStatusResponse)
async def get_ingest_status() -> IngestStatusResponse:
    """Return the current (or last completed) ingestion job status."""
    raw = get_ingestion_status()
    return IngestStatusResponse(
        status=raw.get("status", "idle"),
        total=raw.get("total", 0),
        ingested=raw.get("ingested", 0),
        skipped=raw.get("skipped", 0),
        started_at=raw.get("started_at"),
        completed_at=raw.get("completed_at"),
        duration_ms=raw.get("duration_ms"),
        error=raw.get("error"),
    )
