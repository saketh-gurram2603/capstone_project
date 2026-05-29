"""Pydantic V2 schemas for Incident domain."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class PriorityEnum(str, Enum):
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"
    P4 = "P4"


class ImpactEnum(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class UrgencyEnum(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class IncidentStateEnum(str, Enum):
    OPEN = "Open"
    IN_PROGRESS = "In Progress"
    RESOLVED = "Resolved"
    CLOSED = "Closed"


class IncidentCreate(BaseModel):
    """Schema for creating/ingesting an incident record."""

    number: str = Field(..., description="Unique incident identifier (e.g. INC0010001)")
    state: Optional[IncidentStateEnum] = Field(None, description="Current incident state")
    impact: Optional[ImpactEnum] = Field(None, description="Business impact level")
    urgency: Optional[UrgencyEnum] = Field(None, description="Urgency level")
    priority: Optional[PriorityEnum] = Field(None, description="Combined priority (P1-P4)")
    category: Optional[str] = Field(None, description="Incident category")
    description: str = Field(..., min_length=5, description="Full incident description")
    resolution_notes: Optional[str] = Field(None, description="How the incident was resolved")
    assigned_to: Optional[str] = Field(None, description="Assigned team or engineer")
    opened_at: Optional[str] = Field(None, description="ISO timestamp when incident opened")
    resolved_at: Optional[str] = Field(None, description="ISO timestamp when resolved")

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Description cannot be blank.")
        return v.strip()


class IncidentResponse(BaseModel):
    """Schema returned for a single incident in search results."""

    incident_id: str = Field(..., description="Unique incident identifier (e.g. INC-5001)")
    ticket_id: Optional[str] = Field(None, description="Ticket reference (e.g. TKT-1001)")
    title: Optional[str] = Field(None, description="Short incident headline")
    priority: Optional[str] = Field(None, description="P1–P4")
    impact: Optional[str] = Field(None, description="Impact level")
    category: Optional[str] = Field(None, description="Incident category")
    description: str = Field(..., description="Incident description")
    resolution_notes: Optional[str] = Field(None, description="Resolution steps")
    assigned_to: Optional[str] = Field(None, description="Assigned asset or team")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Reranked similarity score")
    occurrence_count: int = Field(default=1, description="Times this resolution pattern appeared")
