"""
Triage API endpoints.

POST /triage    — run the L1 → L2 → L3 agent graph for a new incident
GET  /escalations — list L3 escalation tickets from Postgres
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.agents.graph import run_triage, _final_confidence
from src.agents.l3_specialist import list_escalation_tickets, resolve_ticket
from src.core.dependencies import get_app_config, get_vector_store
from src.handlers.logger import get_logger, log_info, log_warning
from src.integrations.vector_db import VectorStore
from src.models.triage import (
    EscalationListResponse,
    EscalationTicket,
    ResolveTicketRequest,
    ResolveTicketResponse,
    TriageRequest,
    TriageResult,
)

logger = get_logger("api.triage")
router = APIRouter(tags=["Triage"])


@router.post("/triage", response_model=TriageResult, summary="Run L1→L2→L3 incident triage")
async def triage_incident(request: Request, body: TriageRequest) -> TriageResult:
    """
    Run the full incident triage pipeline.

    - **L1** (GPT-4o-mini): searches the incident knowledge base and attempts
      resolution. Returns immediately if confidence ≥ 0.80.
    - **L2** (GPT-4o): adds Tavily web search context and synthesises a deeper
      analysis. Returns if confidence ≥ 0.55.
    - **L3**: inserts a Postgres escalation ticket and returns the ticket ID.
    """
    start_ts = time.monotonic()

    # Retrieve the compiled graph from app state (built once at startup)
    graph = getattr(request.app.state, "triage_graph", None)
    if graph is None:
        log_warning("Triage graph not initialised — returning 503")
        raise HTTPException(
            status_code=503,
            detail="Triage service is not ready. Please retry shortly.",
        )

    impact = body.impact.value if body.impact else ""
    urgency = body.urgency.value if body.urgency else ""

    log_info(
        "POST /triage | description='%s' impact=%s urgency=%s",
        body.description[:80],
        impact,
        urgency,
    )

    final_state = await run_triage(
        graph=graph,
        query=body.description,
        impact=impact,
        urgency=urgency,
    )

    latency_ms = (time.monotonic() - start_ts) * 1000
    confidence = _final_confidence(final_state)

    # Map priority string to enum-safe value (PriorityEnum or None)
    priority_raw = final_state.get("priority")

    return TriageResult(
        escalation_level=final_state.get("escalation_level", "L3"),
        priority=priority_raw,
        confidence=round(confidence, 4),
        final_answer=final_state.get("final_answer", ""),
        l1_summary=final_state.get("l1_summary") or None,
        l2_synthesis=final_state.get("l2_synthesis") or None,
        escalation_reason=final_state.get("escalation_reason") or None,
        escalation_ticket_id=final_state.get("escalation_ticket_id") or None,
        model_used=final_state.get("model_used", ""),
        fallback_used=final_state.get("fallback_used", False),
        latency_ms=round(latency_ms, 1),
    )


@router.get(
    "/escalations",
    response_model=EscalationListResponse,
    summary="List L3 escalation tickets",
)
async def get_escalations(
    status: Optional[str] = Query(
        None,
        description="Filter by ticket status: OPEN | IN_PROGRESS | RESOLVED",
        pattern="^(OPEN|IN_PROGRESS|RESOLVED)$",
    ),
    limit: int = Query(50, ge=1, le=200, description="Maximum tickets to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> EscalationListResponse:
    """
    Retrieve L3 escalation tickets from Postgres.

    Supports optional status filtering and pagination.
    """
    log_info("GET /escalations | status=%s limit=%d offset=%d", status, limit, offset)

    tickets_raw = await list_escalation_tickets(status=status, limit=limit, offset=offset)

    tickets = [
        EscalationTicket(
            ticket_id=t["ticket_id"],
            description=t["description"],
            impact=t.get("impact"),
            urgency=t.get("urgency"),
            l1_summary=t.get("l1_summary"),
            l2_analysis=t.get("l2_analysis"),
            escalation_reason=t.get("escalation_reason", ""),
            status=t.get("status", "OPEN"),
            created_at=t.get("created_at", ""),
        )
        for t in tickets_raw
    ]

    return EscalationListResponse(total=len(tickets), tickets=tickets)


@router.post(
    "/escalations/{ticket_id}/resolve",
    response_model=ResolveTicketResponse,
    summary="Resolve an escalation ticket and ingest IT team's fix into the KB",
)
async def resolve_escalation(
    ticket_id: str,
    body: ResolveTicketRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    app_config: dict = Depends(get_app_config),
) -> ResolveTicketResponse:
    """
    Mark an L3 escalation ticket as RESOLVED and ingest the IT team's
    resolution steps into the knowledge base as a new searchable incident record.

    The new record is immediately available in both Qdrant (vector search)
    and BM25 (keyword search) — no restart required.
    """
    log_info("POST /escalations/%s/resolve", ticket_id)

    collection = app_config.get("QDRANT", {}).get("COLLECTION_NAME", "incidents")

    try:
        result = await resolve_ticket(
            ticket_id=ticket_id,
            resolution_steps=body.resolution_steps,
            vector_store=vector_store,
            collection=collection,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        log_warning("Resolve ticket failed | ticket_id=%s error=%s", ticket_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to resolve ticket: {exc}") from exc

    return ResolveTicketResponse(
        ticket_id=result["ticket_id"],
        new_incident_id=result["new_incident_id"],
        status=result["status"],
        ingested_to_kb=result["ingested_to_kb"],
        message=(
            f"Ticket {ticket_id} resolved. "
            + (f"Resolution ingested as {result['new_incident_id']} — now searchable in the KB."
               if result["ingested_to_kb"]
               else "Note: KB ingestion failed (check embedding service). Ticket is still marked RESOLVED.")
        ),
    )
