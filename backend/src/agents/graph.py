"""
LangGraph StateGraph for the L1 → L2 → L3 incident triage pipeline.

Graph topology:
  START → l1_triage
         ├─ confidence ≥ threshold → END
         └─ confidence < threshold → l2_analysis
                                    ├─ confidence ≥ threshold → END
                                    └─ confidence < threshold → l3_specialist → END

Each node is created via its make_*_node() factory with injected dependencies.
The graph is compiled once at startup and reused across all requests.

Public API:
  build_triage_graph(vector_store, collection, app_config)  → CompiledGraph
  run_triage(graph, query, impact, urgency)                  → IncidentState
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.l1_triage import make_l1_node
from src.agents.l2_analysis import make_l2_node
from src.agents.l3_specialist import make_l3_node
from src.agents.state import IncidentState, initial_state
from src.handlers.logger import get_logger, log_info, log_warning
from src.integrations.vector_db import VectorStore

logger = get_logger("agents.graph")


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_triage_graph(
    vector_store: VectorStore,
    collection: str,
    app_config: dict,
):
    """
    Compile the LangGraph StateGraph with all three triage nodes.

    Called once at startup — the compiled graph is stored on app.state and
    reused for every POST /triage request.

    Returns a CompiledGraph (LangGraph's runnable graph object).
    """
    l1_threshold = app_config.get("RETRIEVAL", {}).get("L1_CONFIDENCE_THRESHOLD", 0.80)
    l2_threshold = app_config.get("RETRIEVAL", {}).get("L2_CONFIDENCE_THRESHOLD", 0.55)

    # Create node callables via factories
    l1_node = make_l1_node(vector_store, collection, app_config)
    l2_node = make_l2_node(app_config)
    l3_node = make_l3_node()

    # ── Routing functions (pure — read state, return next node name) ──────────

    def route_after_l1(state: IncidentState) -> str:
        confidence = state.get("l1_confidence", 0.0)
        if confidence >= l1_threshold:
            log_info("L1 routing → END (confidence=%.3f ≥ %.2f)", confidence, l1_threshold)
            return END
        log_info("L1 routing → L2 (confidence=%.3f < %.2f)", confidence, l1_threshold)
        return "l2_analysis"

    def route_after_l2(state: IncidentState) -> str:
        confidence = state.get("l2_confidence", 0.0)
        if confidence >= l2_threshold:
            log_info("L2 routing → END (confidence=%.3f ≥ %.2f)", confidence, l2_threshold)
            return END
        log_info("L2 routing → L3 (confidence=%.3f < %.2f)", confidence, l2_threshold)
        return "l3_specialist"

    # ── Build the graph ───────────────────────────────────────────────────────
    builder = StateGraph(IncidentState)

    builder.add_node("l1_triage", l1_node)
    builder.add_node("l2_analysis", l2_node)
    builder.add_node("l3_specialist", l3_node)

    builder.add_edge(START, "l1_triage")

    builder.add_conditional_edges(
        "l1_triage",
        route_after_l1,
        {END: END, "l2_analysis": "l2_analysis"},
    )

    builder.add_conditional_edges(
        "l2_analysis",
        route_after_l2,
        {END: END, "l3_specialist": "l3_specialist"},
    )

    builder.add_edge("l3_specialist", END)

    compiled = builder.compile()
    log_info(
        "Triage graph compiled | L1 threshold=%.2f L2 threshold=%.2f",
        l1_threshold,
        l2_threshold,
    )
    return compiled


# ── Runtime helper ────────────────────────────────────────────────────────────


async def run_triage(
    graph: Any,
    query: str,
    impact: str = "",
    urgency: str = "",
) -> IncidentState:
    """
    Execute the compiled triage graph for a single incident.

    Args:
        graph:   Compiled LangGraph StateGraph (from build_triage_graph).
        query:   Natural-language incident description.
        impact:  Reported impact level (High / Medium / Low).
        urgency: Reported urgency level (High / Medium / Low).

    Returns:
        Final IncidentState after the graph has run to completion.
    """
    start_ts = time.monotonic()
    state = initial_state(query=query, impact=impact, urgency=urgency)

    log_info("run_triage | query='%s' impact=%s urgency=%s", query[:80], impact, urgency)

    try:
        final_state: IncidentState = await graph.ainvoke(state)
    except Exception as exc:
        log_warning("Triage graph execution failed | error=%s", exc)
        # Return a graceful degraded state rather than surfacing a 500
        final_state = _degraded_state(state, str(exc))

    elapsed_ms = (time.monotonic() - start_ts) * 1000
    log_info(
        "run_triage complete | level=%s confidence=%.3f latency=%.0fms",
        final_state.get("escalation_level", "?"),
        _final_confidence(final_state),
        elapsed_ms,
    )
    return final_state


# ── Private helpers ───────────────────────────────────────────────────────────


def _final_confidence(state: IncidentState) -> float:
    """Return the most relevant confidence score from the completed state."""
    level = state.get("escalation_level", "L1")
    if level == "L1":
        return state.get("l1_confidence", 0.0)
    if level == "L2":
        return state.get("l2_confidence", 0.0)
    return 0.0  # L3 has no LLM confidence


def _degraded_state(base_state: IncidentState, error_msg: str) -> IncidentState:
    """Produce a safe fallback state when the graph itself crashes."""
    base_state = dict(base_state)  # type: ignore[assignment]
    base_state.update(
        {
            "final_answer": (
                "Automated triage failed due to a system error. "
                "Please review the incident manually or contact your IT support team."
            ),
            "escalation_level": "L3",
            "l1_confidence": 0.0,
            "l2_confidence": 0.0,
            "escalation_reason": f"Graph execution error: {error_msg}",
            "model_used": "none",
            "fallback_used": True,
        }
    )
    return base_state  # type: ignore[return-value]
