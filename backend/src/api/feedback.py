"""
Feedback API.

POST /feedback                     — user submits feedback from the chat thumbs UI
GET  /feedback                     — list feedback + aggregate stats (admin, auth required)
POST /feedback/{id}/review         — admin verifies or dismisses a feedback item (auth required)
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.core.dependencies import require_api_key
from src.feedback.feedback_store import (
    get_feedback_stats,
    list_feedback,
    update_feedback,
)
from src.handlers.logger import get_logger, log_info
from src.feedback.feedback_store import record_feedback
from src.models.feedback import (
    FeedbackItem,
    FeedbackListResponse,
    FeedbackReviewRequest,
    FeedbackStats,
    FeedbackSubmitRequest,
)

logger = get_logger("api.feedback")
router = APIRouter(tags=["Feedback"])


@router.post(
    "/feedback",
    response_model=FeedbackItem,
    summary="Submit feedback from the chat thumbs UI",
)
async def submit_feedback(body: FeedbackSubmitRequest) -> FeedbackItem:
    """
    Public endpoint — no auth required.
    The frontend sends session_id + fix_index + sentiment + optional reason.
    The backend looks up the session to retrieve resolution metadata so the
    admin queue has full context.
    """
    log_info("POST /feedback | session=%s fix=%d sentiment=%s", body.session_id, body.fix_index, body.sentiment)

    from src.chat.session_manager import session_manager

    session     = session_manager.get_session(body.session_id)
    query       = ""
    fix_total   = 0
    res_text    = ""
    inc_ids: list[str] = []
    occ_count   = 0

    if session:
        query     = session.incident_description
        fix_total = len(session.resolution_options)
        idx = body.fix_index - 1          # convert to 0-based
        if 0 <= idx < len(session.resolution_options):
            opt       = session.resolution_options[idx]
            res_text  = opt.get("resolution_text", "")
            inc_ids   = opt.get("source_incident_ids", [])
            occ_count = opt.get("occurrence_count", 1)

    record = await record_feedback(
        session_id=body.session_id,
        query=query,
        sentiment=body.sentiment,
        fix_index=body.fix_index,
        fix_total=fix_total,
        resolution_text=res_text,
        incident_ids=inc_ids,
        occurrence_count=occ_count,
        reason=body.reason,
    )
    return FeedbackItem(**record)


@router.get(
    "/feedback",
    response_model=FeedbackListResponse,
    summary="List user feedback for admin review",
)
async def get_feedback(
    _: str = Depends(require_api_key),
    status: Optional[str] = Query(
        None,
        description="Filter by review status",
        pattern="^(PENDING|VERIFIED|DISMISSED)$",
    ),
    sentiment: Optional[str] = Query(
        None,
        description="Filter by sentiment",
        pattern="^(positive|negative)$",
    ),
    limit: int = Query(200, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> FeedbackListResponse:
    """Return feedback records (most recent first) plus aggregate stats."""
    log_info("GET /feedback | status=%s sentiment=%s", status, sentiment)

    items, total = await list_feedback(
        status=status, sentiment=sentiment, limit=limit, offset=offset,
    )
    stats = await get_feedback_stats()

    return FeedbackListResponse(
        total=total,
        stats=FeedbackStats(**stats),
        items=[FeedbackItem(**item) for item in items],
    )


@router.post(
    "/feedback/{feedback_id}/review",
    response_model=FeedbackItem,
    summary="Verify or dismiss a feedback item",
)
async def review_feedback(
    feedback_id: str,
    body: FeedbackReviewRequest,
    _: str = Depends(require_api_key),
) -> FeedbackItem:
    """Admin action — flips the feedback status to VERIFIED or DISMISSED."""
    log_info("POST /feedback/%s/review | status=%s", feedback_id, body.status)

    updated = await update_feedback(
        feedback_id=feedback_id,
        status=body.status,
        admin_action=body.admin_action,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail=f"Feedback not found: {feedback_id}")

    return FeedbackItem(**updated)
