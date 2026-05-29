"""Pydantic models for the guided chat / troubleshooting feature."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        None,
        description="Omit (or null) to start a new session; supply an existing ID to continue.",
    )
    message: str = Field(..., min_length=3, max_length=2000)


class OptionProgress(BaseModel):
    current: int = Field(..., description="1-based index of the fix currently being presented")
    total: int = Field(..., description="Total number of resolution options found")


class ChatResponse(BaseModel):
    session_id: str
    message: str = Field(..., description="Markdown-formatted assistant reply")
    role: Literal["assistant"] = "assistant"
    option_progress: Optional[OptionProgress] = None
    is_escalated: bool = False
    escalation_ticket_id: Optional[str] = None
    all_options_exhausted: bool = False
    suggested_actions: list[str] = Field(
        default_factory=list,
        description="Button labels the UI should render for quick replies",
    )
