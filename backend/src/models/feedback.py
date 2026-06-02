"""Pydantic schemas for the user-feedback / admin-review feature."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class FeedbackItem(BaseModel):
    """A single feedback record as surfaced to the admin review page."""

    feedback_id: str
    session_id: Optional[str] = None
    query: str
    sentiment: Literal["positive", "negative"]
    fix_index: int = Field(..., description="1-based index of the fix the user reacted to")
    fix_total: int = Field(..., description="Total number of fixes offered in the session")
    resolution_text: Optional[str] = None
    incident_ids: list[str] = Field(default_factory=list)
    occurrence_count: int = 0
    reason: Optional[str] = Field(None, description="Free-text reason supplied by the user")
    status: Literal["PENDING", "VERIFIED", "DISMISSED"] = "PENDING"
    admin_action: Optional[str] = None
    created_at: Optional[str] = None
    reviewed_at: Optional[str] = None


class FeedbackStats(BaseModel):
    """Aggregate counts shown on the admin dashboard cards."""

    total: int = 0
    negative: int = 0
    positive: int = 0
    pending: int = 0
    verified: int = 0
    dismissed: int = 0


class FeedbackListResponse(BaseModel):
    """Response body for GET /feedback."""

    total: int = Field(..., description="Number of records matching the active filter")
    stats: FeedbackStats
    items: list[FeedbackItem] = Field(default_factory=list)


class FeedbackReviewRequest(BaseModel):
    """Request body for POST /feedback/{feedback_id}/review (admin action)."""

    status: Literal["VERIFIED", "DISMISSED"]
    admin_action: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional note describing what the admin did or decided.",
    )


class FeedbackSubmitRequest(BaseModel):
    """Request body for POST /feedback — submitted by the user from the chat UI."""

    session_id: str = Field(..., description="Active chat session ID")
    fix_index: int  = Field(..., description="1-based index of the fix being rated")
    sentiment: Literal["positive", "negative"]
    reason: Optional[str] = Field(None, max_length=1000)
