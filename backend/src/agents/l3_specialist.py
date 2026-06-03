"""
L3 Specialist node — pure routing with zero LLM calls.

When both L1 and L2 fail to reach sufficient confidence, L3:
  1. Creates an escalation ticket (persisted to Postgres when available)
  2. Returns the ticket ID and a clear escalation message
  3. Sets escalation_level = "L3" and final_answer to the ticket reference

Persistence strategy:
  - Primary:  Postgres via SQLAlchemy (EscalationTicketDB ORM model)
  - Fallback: In-memory list (_tickets) when Postgres is not configured
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable

from src.agents.state import IncidentState
from src.handlers.logger import get_logger, log_info, log_warning

logger = get_logger("agents.l3_specialist")

# ── In-memory ticket store (fallback when Postgres is unavailable) ────────────
_tickets: list[dict] = []


# ── Node factory ──────────────────────────────────────────────────────────────


async def create_escalation_ticket(
    description: str,
    impact: str = "",
    urgency: str = "",
    priority: str | None = None,
    l1_summary: str = "",
    l2_synthesis: str = "",
    escalation_reason: str = "All resolution options exhausted via chat.",
) -> dict:
    """
    Create and persist an escalation ticket.
    Reusable by both the LangGraph L3 node and the chat agent.

    Returns a dict with ticket_id, final_answer, escalation_level, and status.
    """
    log_info("Creating escalation ticket | query='%s'", description[:80])

    ticket_id = _generate_ticket_id()

    await _write_ticket(
        ticket_id=ticket_id,
        description=description,
        impact=impact,
        urgency=urgency,
        priority=priority,
        l1_summary=l1_summary,
        l2_analysis=l2_synthesis,
        escalation_reason=escalation_reason,
    )

    log_info("Escalation ticket created | ticket_id=%s", ticket_id)

    final_answer = (
        f"This incident has been escalated to the specialist team.\n"
        f"Ticket ID: {ticket_id}\n"
        f"Status: OPEN — assigned to IT-OPS escalation queue.\n\n"
        f"Escalation reason: {escalation_reason}"
    )

    return {
        "ticket_id": ticket_id,
        "final_answer": final_answer,
        "escalation_level": "L3",
        "status": "OPEN",
    }


def make_l3_node() -> Callable[[IncidentState], dict]:
    """
    Factory: returns an async L3 specialist node.
    No external config needed — in-memory write + return.
    """

    async def l3_node(state: IncidentState) -> dict:
        result = await create_escalation_ticket(
            description=state.get("query", ""),
            impact=state.get("impact", ""),
            urgency=state.get("urgency", ""),
            priority=state.get("priority"),
            l1_summary=state.get("l1_summary", ""),
            l2_synthesis=state.get("l2_synthesis", ""),
            escalation_reason=state.get("escalation_reason", "Escalated after L1 + L2 failed."),
        )
        return {
            "escalation_ticket_id": result["ticket_id"],
            "final_answer": result["final_answer"],
            "escalation_level": "L3",
            "model_used": "none",
            "fallback_used": False,
        }

    return l3_node


# ── Private helpers ───────────────────────────────────────────────────────────


def _generate_ticket_id() -> str:
    """Generate a unique ticket ID in the format TKT-<8 uppercase hex chars>."""
    return f"TKT-{uuid.uuid4().hex[:8].upper()}"


async def _write_ticket(
    ticket_id: str,
    description: str,
    impact: str,
    urgency: str,
    priority: str | None,
    l1_summary: str,
    l2_analysis: str,
    escalation_reason: str,
) -> None:
    """
    Persist the escalation ticket.
    Primary: Postgres (EscalationTicketDB).
    Fallback: in-memory _tickets list when Postgres is not initialised.
    """
    created_at = datetime.now(timezone.utc)
    ticket = {
        "ticket_id": ticket_id,
        "description": description[:2000],
        "impact": impact or None,
        "urgency": urgency or None,
        "priority": priority or None,
        "l1_summary": l1_summary[:2000] if l1_summary else None,
        "l2_analysis": l2_analysis[:2000] if l2_analysis else None,
        "escalation_reason": escalation_reason[:1000],
        "status": "OPEN",
        "created_at": created_at.isoformat(),
    }
    # Always keep an in-memory copy (fast list access for same-process queries)
    _tickets.append(ticket)

    # Attempt Postgres persistence
    try:
        from src.models.db_models import EscalationTicketDB
        from src.integrations.database import get_session

        async with get_session() as session:
            row = EscalationTicketDB(
                ticket_id=ticket_id,
                description=ticket["description"],
                impact=ticket["impact"],
                urgency=ticket["urgency"],
                priority=ticket["priority"],
                l1_summary=ticket["l1_summary"],
                l2_analysis=ticket["l2_analysis"],
                escalation_reason=ticket["escalation_reason"],
                status=ticket["status"],
                created_at=created_at,
            )
            session.add(row)
        log_info("Ticket persisted to Postgres | ticket_id=%s", ticket_id)
    except RuntimeError:
        # Database not initialised — in-memory fallback already done above
        log_warning("Postgres not initialised — ticket stored in memory only | ticket_id=%s", ticket_id)
    except Exception as exc:
        log_warning("Postgres ticket write failed, in-memory only | ticket_id=%s error=%s", ticket_id, exc)


async def resolve_ticket(
    ticket_id: str,
    resolution_steps: str,
    vector_store,
    collection: str,
) -> dict:
    """
    Mark an escalation ticket RESOLVED and ingest the IT team's resolution
    into the knowledge base (Qdrant + BM25) as a new searchable incident record.

    Returns a dict with ticket_id, new_incident_id, and ingested_to_kb flag.
    """
    import hashlib

    from src.handlers.logger import log_info, log_warning

    log_info("Resolving ticket | ticket_id=%s", ticket_id)

    # ── 1. Fetch ticket details ───────────────────────────────────────────────
    all_tickets = await list_escalation_tickets(limit=1000)
    ticket = next((t for t in all_tickets if t["ticket_id"] == ticket_id), None)
    if ticket is None:
        raise ValueError(f"Ticket not found: {ticket_id}")

    description = ticket.get("description", "")

    # ── 2. Mark RESOLVED in DB + memory ─────────────────────────────────────
    await _update_ticket_status(ticket_id, "RESOLVED")

    # ── 3. Generate a unique incident ID for the new KB record ──────────────
    new_incident_id = f"IT-{uuid.uuid4().hex[:8].upper()}"
    search_text = description

    # ── 4. Ingest into Qdrant + BM25 ────────────────────────────────────────
    ingested = False
    try:
        from src.integrations.embeddings import embed_text
        from src.retrieval.bm25_retriever import add_document_to_bm25

        vector = await embed_text(search_text)

        # Compute actual resolution time from ticket timestamps
        from datetime import datetime, timezone as _tz
        now_iso   = datetime.now(_tz.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        opened_iso = ticket.get("created_at") or ""
        res_hours  = 0.0
        try:
            if opened_iso:
                opened_dt  = datetime.fromisoformat(opened_iso.replace("Z", "+00:00"))
                res_hours  = round(
                    (datetime.now(_tz.utc) - opened_dt).total_seconds() / 3600, 2
                )
        except Exception:
            res_hours = 0.0

        payload = {
            "incident_id":      new_incident_id,
            "ticket_id":        ticket_id,
            "title":            description[:120],
            "category":         "IT Resolution",
            "description":      description,
            "resolution_notes": resolution_steps,
            "assigned_to":      "IT Operations",
            "search_text":      search_text,
            "impact":           ticket.get("impact")  or "",
            "urgency":          ticket.get("urgency") or "",
            "priority":         ticket.get("priority") or "",
            "opened_at":        opened_iso,
            "resolved_at":      now_iso,
            "resolution_hours": res_hours,
        }

        # Stable Qdrant point ID from the new incident ID
        qdrant_id = int(hashlib.sha1(new_incident_id.encode()).hexdigest()[:8], 16)
        await vector_store.upsert(collection, [{
            "id":      qdrant_id,
            "vector":  vector,
            "payload": payload,
        }])
        log_info("IT resolution upserted to Qdrant | incident_id=%s", new_incident_id)

        # Live BM25 update — no restart needed
        add_document_to_bm25(new_incident_id, search_text, payload)
        ingested = True

    except Exception as exc:
        log_warning(
            "KB ingestion failed for IT resolution | ticket=%s error=%s — "
            "ticket is still marked RESOLVED", ticket_id, exc,
        )

    return {
        "ticket_id":       ticket_id,
        "new_incident_id": new_incident_id,
        "status":          "RESOLVED",
        "ingested_to_kb":  ingested,
    }


async def _update_ticket_status(ticket_id: str, status: str) -> None:
    """Update ticket status in both the in-memory list and the DB."""
    # In-memory update
    for t in _tickets:
        if t["ticket_id"] == ticket_id:
            t["status"] = status
            break

    # DB update
    try:
        from src.models.db_models import EscalationTicketDB
        from src.integrations.database import get_session
        from sqlalchemy import select

        async with get_session() as session:
            stmt = select(EscalationTicketDB).where(
                EscalationTicketDB.ticket_id == ticket_id
            )
            row = (await session.execute(stmt)).scalars().first()
            if row:
                row.status = status
        log_info("Ticket status updated | ticket_id=%s status=%s", ticket_id, status)
    except RuntimeError:
        pass  # DB not initialised — in-memory only
    except Exception as exc:
        from src.handlers.logger import log_warning
        log_warning(
            "DB ticket status update failed | ticket_id=%s error=%s", ticket_id, exc
        )


async def list_escalation_tickets(
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """
    Return escalation tickets ordered by creation time (most recent first).
    Queries Postgres when available; falls back to the in-memory list.
    Optionally filter by status ("OPEN", "IN_PROGRESS", "RESOLVED").
    """
    # ── Try Postgres ──────────────────────────────────────────────────────────
    try:
        from src.models.db_models import EscalationTicketDB
        from src.integrations.database import get_session
        from sqlalchemy import select, desc

        async with get_session() as session:
            stmt = select(EscalationTicketDB).order_by(desc(EscalationTicketDB.created_at))
            if status:
                stmt = stmt.where(EscalationTicketDB.status == status)
            stmt = stmt.offset(offset).limit(limit)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [row.to_dict() for row in rows]
    except RuntimeError:
        pass  # DB not initialised — fall through to in-memory
    except Exception as exc:
        log_warning("Postgres ticket list failed, using in-memory | error=%s", exc)

    # ── Fallback: in-memory ───────────────────────────────────────────────────
    results = _tickets
    if status:
        results = [t for t in results if t["status"] == status]
    results = list(reversed(results))
    return results[offset: offset + limit]
