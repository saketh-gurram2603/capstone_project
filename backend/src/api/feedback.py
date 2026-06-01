"""
Feedback API.

GET  /feedback                     — list feedback + aggregate stats (admin page)
POST /feedback/{id}/review         — admin verifies or dismisses a feedback item

Feedback itself is captured automatically inside the chat flow
(see src.chat.chat_agent), so there is no public create endpoint here.
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
from src.models.feedback import (
    FeedbackItem,
    FeedbackListResponse,
    FeedbackReviewRequest,
    FeedbackStats,
)

logger = get_logger("api.feedback")
router = APIRouter(tags=["Feedback"])


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
