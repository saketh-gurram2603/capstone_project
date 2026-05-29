"""
SQLAlchemy ORM models for Postgres persistence.

Tables:
  - escalation_tickets  (L3 specialist output)
  - eval_runs           (evaluation pipeline results)

Both tables are created via create_tables() in src.integrations.database.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.integrations.database import Base


class EscalationTicketDB(Base):
    """Persistent store for L3 escalation tickets."""

    __tablename__ = "escalation_tickets"

    ticket_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    description: Mapped[str] = mapped_column(Text)
    impact: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    urgency: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    priority: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    l1_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    l2_analysis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    escalation_reason: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="OPEN")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    def to_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "description": self.description,
            "impact": self.impact,
            "urgency": self.urgency,
            "priority": self.priority,
            "l1_summary": self.l1_summary,
            "l2_analysis": self.l2_analysis,
            "escalation_reason": self.escalation_reason,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class EvalRunDB(Base):
    """Persistent store for evaluation pipeline run results."""

    __tablename__ = "eval_runs"

    run_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    metrics_json: Mapped[str] = mapped_column(Text)   # JSON-serialised list[dict]
    overall_passed: Mapped[bool] = mapped_column(Boolean)
    num_test_cases: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[str] = mapped_column(String(50))

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "metrics": json.loads(self.metrics_json),
            "overall_passed": self.overall_passed,
            "num_test_cases": self.num_test_cases,
            "timestamp": self.timestamp,
        }
