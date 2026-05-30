"""
Feedback store — captures user feedback from the guided chat and serves it to
the admin review page.

Feedback is recorded automatically by the chat agent:
  • "Issue resolved"            → positive feedback on the current fix
  • "This didn't work…"         → negative feedback on the current fix, with the
                                  user's message kept as the free-text reason

Persistence strategy (mirrors src.agents.l3_specialist):
  - Primary:  SQLite / Postgres via SQLAlchemy (FeedbackDB ORM model)
  - Fallback: in-memory list (_feedback) when the DB is not initialised

The admin "verify / dismiss" action only flips the record's status — no
downstream ranking or KB pipeline is wired (intentionally out of scope).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.handlers.logger import get_logger, log_info, log_warning

logger = get_logger("feedback.store")

# ── In-memory store (fallback when the DB is unavailable) ─────────────────────
_feedback: list[dict] = []


# ── Recording ─────────────────────────────────────────────────────────────────

async def record_feedback(
    *,
    session_id: Optional[str],
    query: str,
    sentiment: str,
    fix_index: int,
    fix_total: int,
    resolution_text: str = "",
    incident_ids: Optional[list[str]] = None,
    occurrence_count: int = 0,
    reason: Optional[str] = None,
    status: str = "PENDING",
    created_at: Optional[datetime] = None,
) -> dict:
    """
    Create and persist a feedback record. Returns the stored record as a dict.
    Never raises on persistence failure — the in-memory copy is always kept.
    """
    feedback_id = f"FB-{uuid.uuid4().hex[:8].upper()}"
    incident_ids = incident_ids or []
    created_at = created_at or datetime.now(timezone.utc)

    record = {
        "feedback_id": feedback_id,
        "session_id": session_id,
        "query": (query or "")[:2000],
        "sentiment": sentiment,
        "fix_index": fix_index,
        "fix_total": fix_total,
        "resolution_text": (resolution_text or "")[:2000] or None,
        "incident_ids": incident_ids,
        "occurrence_count": occurrence_count,
        "reason": (reason or "")[:2000] or None,
        "status": status,
        "admin_action": None,
        "created_at": created_at.isoformat(),
        "reviewed_at": None,
    }

    # Always keep an in-memory copy.
    _feedback.append(record)

    # Attempt DB persistence.
    try:
        from src.integrations.database import get_session
        from src.models.db_models import FeedbackDB

        async with get_session() as session:
            session.add(FeedbackDB(
                feedback_id=feedback_id,
                session_id=session_id,
                query=record["query"],
                sentiment=sentiment,
                fix_index=fix_index,
                fix_total=fix_total,
                resolution_text=record["resolution_text"],
                incident_ids=json.dumps(incident_ids),
                occurrence_count=occurrence_count,
                reason=record["reason"],
                status=status,
                created_at=created_at,
            ))
        log_info("Feedback persisted | id=%s sentiment=%s", feedback_id, sentiment)
    except RuntimeError:
        log_warning("DB not initialised — feedback stored in memory only | id=%s", feedback_id)
    except Exception as exc:
        log_warning("Feedback DB write failed, in-memory only | id=%s error=%s", feedback_id, exc)

    return record


# ── Querying ──────────────────────────────────────────────────────────────────

async def _all_records() -> list[dict]:
    """Return every feedback record, most recent first (DB-first, else memory)."""
    try:
        from sqlalchemy import desc, select

        from src.integrations.database import get_session
        from src.models.db_models import FeedbackDB

        async with get_session() as session:
            stmt = select(FeedbackDB).order_by(desc(FeedbackDB.created_at))
            result = await session.execute(stmt)
            return [row.to_dict() for row in result.scalars().all()]
    except RuntimeError:
        pass  # DB not initialised — use in-memory
    except Exception as exc:
        log_warning("Feedback DB query failed, using in-memory | error=%s", exc)

    return list(reversed(_feedback))


async def list_feedback(
    status: Optional[str] = None,
    sentiment: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """
    Return (page, total_matching) for the given filters.
    """
    records = await _all_records()
    if status:
        records = [r for r in records if r.get("status") == status]
    if sentiment:
        records = [r for r in records if r.get("sentiment") == sentiment]
    total = len(records)
    return records[offset: offset + limit], total


async def get_feedback_stats() -> dict:
    """Aggregate counts across all feedback for the admin dashboard cards."""
    records = await _all_records()
    return {
        "total":     len(records),
        "negative":  sum(1 for r in records if r.get("sentiment") == "negative"),
        "positive":  sum(1 for r in records if r.get("sentiment") == "positive"),
        "pending":   sum(1 for r in records if r.get("status") == "PENDING"),
        "verified":  sum(1 for r in records if r.get("status") == "VERIFIED"),
        "dismissed": sum(1 for r in records if r.get("status") == "DISMISSED"),
    }


# ── Admin review ────────────────────────────────────────────────────────────────

async def update_feedback(
    feedback_id: str,
    status: str,
    admin_action: Optional[str] = None,
) -> Optional[dict]:
    """
    Flip a feedback record's review status. Returns the updated record, or None
    if no record with that id exists.
    """
    reviewed_at = datetime.now(timezone.utc)

    # Update the in-memory copy if present.
    updated_mem: Optional[dict] = None
    for rec in _feedback:
        if rec["feedback_id"] == feedback_id:
            rec["status"] = status
            rec["admin_action"] = admin_action
            rec["reviewed_at"] = reviewed_at.isoformat()
            updated_mem = rec
            break

    # Update the DB row.
    try:
        from sqlalchemy import select

        from src.integrations.database import get_session
        from src.models.db_models import FeedbackDB

        async with get_session() as session:
            stmt = select(FeedbackDB).where(FeedbackDB.feedback_id == feedback_id)
            row = (await session.execute(stmt)).scalars().first()
            if row is not None:
                row.status = status
                row.admin_action = admin_action
                row.reviewed_at = reviewed_at
                log_info("Feedback reviewed | id=%s status=%s", feedback_id, status)
                return row.to_dict()
    except RuntimeError:
        pass  # DB not initialised — in-memory only
    except Exception as exc:
        log_warning("Feedback DB update failed | id=%s error=%s", feedback_id, exc)

    return updated_mem


# ── Demo seed ──────────────────────────────────────────────────────────────────

async def seed_sample_feedback() -> None:
    """
    Insert a few sample feedback rows so the admin page is never empty on stage.
    No-op if any feedback already exists.
    """
    existing, _ = await list_feedback(limit=1)
    if existing:
        return

    now = datetime.now(timezone.utc)
    samples = [
        {
            "session_id": "seed-001",
            "query": "Database connections are timing out after the latest deployment",
            "sentiment": "negative",
            "fix_index": 1,
            "fix_total": 3,
            "resolution_text": "Increase the connection pool size and restart the DB service.",
            "incident_ids": ["INC-5012", "INC-5044"],
            "occurrence_count": 9,
            "reason": "Bumped the pool size and restarted but the timeouts came back within minutes.",
            "status": "PENDING",
            "created_at": now - timedelta(minutes=6),
        },
        {
            "session_id": "seed-002",
            "query": "VPN keeps disconnecting for remote staff every 15 minutes",
            "sentiment": "negative",
            "fix_index": 2,
            "fix_total": 4,
            "resolution_text": "Disable IPv6 on the client adapter and re-establish the tunnel.",
            "incident_ids": ["INC-4821"],
            "occurrence_count": 4,
            "reason": "Disabling IPv6 made no difference — still drops on a timer.",
            "status": "VERIFIED",
            "created_at": now - timedelta(hours=2),
        },
        {
            "session_id": "seed-003",
            "query": "Storage volume at 95% capacity, alerts firing on MediaServer01",
            "sentiment": "positive",
            "fix_index": 1,
            "fix_total": 2,
            "resolution_text": "Enable log rotation and archive logs older than 30 days.",
            "incident_ids": ["INC-5001", "INC-5089"],
            "occurrence_count": 14,
            "reason": "Worked — log rotation cleared the volume immediately. Thanks!",
            "status": "PENDING",
            "created_at": now - timedelta(hours=5),
        },
    ]
    for s in samples:
        await record_feedback(**s)
    log_info("Seeded %d sample feedback records", len(samples))
