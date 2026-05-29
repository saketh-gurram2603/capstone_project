"""
LangGraph agent state — single TypedDict that flows through L1 → L2 → L3.

Every node receives the full state and returns a partial dict of updated keys.
LangGraph merges those updates back into the state automatically.
"""

from __future__ import annotations

from typing import Literal, Optional, TypedDict


class IncidentState(TypedDict, total=False):
    """
    Shared state carried through the L1 → L2 → L3 agent graph.

    ``total=False`` means every key is optional at the TypedDict level so
    LangGraph can do partial updates without raising KeyError.  Runtime code
    must still guard against missing keys with .get().
    """

    # ── Input ─────────────────────────────────────────────────────────────────
    query: str
    impact: str
    urgency: str

    # ── L1 outputs ────────────────────────────────────────────────────────────
    retrieved_incidents: list[dict]
    resolution_options: list[dict]
    l1_summary: str
    l1_confidence: float
    escalation_reason: Optional[str]

    # ── L2 outputs ────────────────────────────────────────────────────────────
    web_search_results: list[dict]
    l2_synthesis: str
    l2_confidence: float

    # ── L3 outputs ────────────────────────────────────────────────────────────
    escalation_ticket_id: Optional[str]

    # ── Final ─────────────────────────────────────────────────────────────────
    final_answer: str
    escalation_level: Literal["L1", "L2", "L3"]
    priority: Optional[str]
    model_used: str
    fallback_used: bool


def initial_state(query: str, impact: str = "", urgency: str = "") -> IncidentState:
    """Return a blank state with only the input fields populated."""
    return IncidentState(
        query=query,
        impact=impact,
        urgency=urgency,
        retrieved_incidents=[],
        resolution_options=[],
        l1_summary="",
        l1_confidence=0.0,
        escalation_reason=None,
        web_search_results=[],
        l2_synthesis="",
        l2_confidence=0.0,
        escalation_ticket_id=None,
        final_answer="",
        escalation_level="L1",
        priority=None,
        model_used="",
        fallback_used=False,
    )
