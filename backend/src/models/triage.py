"""Pydantic V2 schemas for Triage / Agent domain."""

from typing import Literal, Optional
from pydantic import BaseModel, Field

from src.models.incident import ImpactEnum, UrgencyEnum, PriorityEnum


class TriageRequest(BaseModel):
    """Request body for POST /triage."""

    description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Natural language description of the new incident",
    )
    impact: Optional[ImpactEnum] = Field(None, description="Reported impact level")
    urgency: Optional[UrgencyEnum] = Field(None, description="Reported urgency level")


class TriageResult(BaseModel):
    """Response body for POST /triage."""

    escalation_level: Literal["L1", "L2", "L3"] = Field(
        ..., description="Which agent tier produced the final answer"
    )
    priority: Optional[PriorityEnum] = Field(None, description="Classified priority")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence of the final answer")
    final_answer: str = Field(..., description="Synthesised resolution or escalation message")
    l1_summary: Optional[str] = Field(None, description="L1 retrieval summary (always populated)")
    l2_synthesis: Optional[str] = Field(None, description="L2 synthesis (populated if L2 ran)")
    escalation_reason: Optional[str] = Field(
        None, description="Why the incident was escalated beyond L1"
    )
    escalation_ticket_id: Optional[str] = Field(
        None, description="Postgres ticket ID if escalated to L3"
    )
    model_used: str = Field(..., description="Primary model used for final answer")
    fallback_used: bool = Field(default=False, description="True if local Flan-T5 was used")
    latency_ms: float = Field(..., description="Total triage latency in milliseconds")


class EscalationTicket(BaseModel):
    """Postgres record for an L3 escalation."""

    ticket_id: str
    description: str
    impact: Optional[str]
    urgency: Optional[str]
    l1_summary: Optional[str]
    l2_analysis: Optional[str]
    escalation_reason: str
    status: Literal["OPEN", "IN_PROGRESS", "RESOLVED"] = "OPEN"
    created_at: str


class EscalationListResponse(BaseModel):
    """Response for GET /escalations."""

    total: int
    tickets: list[EscalationTicket]
