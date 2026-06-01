"""
L1 Triage node — first-line incident triage using the knowledge base.

Flow:
  1. Search incident KB via hybrid_search (full retrieval pipeline)
  2. Format top results + unique resolution options as LLM context
  3. Call GPT-4o-mini to summarise findings and recommend resolution
  4. Compute confidence from reranked similarity scores
  5. Classify priority from impact × urgency
  6. Gate: confidence ≥ threshold → END (L1 resolved)
            confidence < threshold → escalate to L2

The node is created via make_l1_node() factory so vector_store / app_config
are injected at graph-build time without polluting module globals.
"""

from __future__ import annotations

from typing import Callable

from src.agents.state import IncidentState
from src.agents.tools import classify_priority, search_kb_incidents
from src.handlers.logger import get_logger, log_info, log_warning
from src.integrations.llm import chat_completion
from src.integrations.vector_db import VectorStore

logger = get_logger("agents.l1_triage")

_L1_SYSTEM_PROMPT = """\
You are an expert IT incident triage engineer with access to a historical incident knowledge base.

Based on the retrieved similar incidents and their resolutions, provide:
1. A concise diagnosis of the likely root cause (1-2 sentences)
2. Recommended resolution steps (ordered by frequency of past success)
3. Expected resolution time estimate

Rules:
- Be direct and actionable — support engineers will follow these steps immediately
- Prioritise resolutions that have been used most often in similar incidents
- If the retrieved incidents are not relevant, clearly state that
- Write in plain text, no markdown formatting
"""


def make_l1_node(
    vector_store: VectorStore,
    collection: str,
    app_config: dict,
) -> Callable[[IncidentState], dict]:
    """
    Factory: returns an async L1 triage node with vector_store injected.
    Called once when the graph is compiled.
    """
    l1_model = app_config.get("LLM", {}).get("L1_MODEL", "synapt-dev-gpt-4o-mini")
    l1_threshold = app_config.get("RETRIEVAL", {}).get("L1_CONFIDENCE_THRESHOLD", 0.80)

    async def l1_node(state: IncidentState) -> dict:
        query = state.get("query", "")
        impact = state.get("impact", "")
        urgency = state.get("urgency", "")

        log_info("L1 node | query='%s'", query[:80])

        # ── 1. KB search ──────────────────────────────────────────────────────
        search_result = await search_kb_incidents(
            query=query,
            vector_store=vector_store,
            collection=collection,
            app_config=app_config,
        )

        results = search_result.get("results", [])
        resolution_options = search_result.get("resolution_options", [])

        # ── 2. Confidence from top similarity scores ───────────────────────────
        confidence = _calculate_confidence(results)
        log_info("L1 confidence=%.3f threshold=%.2f", confidence, l1_threshold)

        # ── 3. Build LLM context ──────────────────────────────────────────────
        context = _format_kb_context(results, resolution_options)

        messages = [
            {"role": "system", "content": _L1_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"New incident: {query}\n"
                    f"Impact: {impact or 'Unknown'} | Urgency: {urgency or 'Unknown'}\n\n"
                    f"{context}"
                ),
            },
        ]

        # ── 4. LLM call (GPT-4o-mini with Flan-T5 fallback) ──────────────────
        try:
            summary, fallback_used = await chat_completion(
                messages=messages,
                model=l1_model,
                temperature=0.2,
                max_tokens=512,
            )
        except Exception as exc:
            log_warning("L1 LLM call failed | error=%s", exc)
            summary = "Unable to generate automated summary. Please review the retrieved incidents manually."
            fallback_used = True
            confidence = min(confidence, 0.4)  # force escalation if LLM failed

        # ── 5. Priority classification ────────────────────────────────────────
        priority = classify_priority(impact, urgency)

        # ── 6. Escalation reason if confidence is low ──────────────────────────
        escalation_reason = None
        if confidence < l1_threshold:
            escalation_reason = (
                f"L1 confidence {confidence:.2f} below threshold {l1_threshold:.2f}. "
                f"Only {len(results)} similar incident(s) found in KB."
            )

        return {
            "retrieved_incidents": results,
            "resolution_options": resolution_options,
            "l1_summary": summary,
            "l1_confidence": round(confidence, 4),
            "priority": priority,
            "escalation_reason": escalation_reason,
            "escalation_level": "L1",
            "final_answer": summary,
            "model_used": l1_model if not fallback_used else "flan-t5-base",
            "fallback_used": fallback_used,
        }

    return l1_node


# ── Private helpers ───────────────────────────────────────────────────────────

# Fallback floor for the legacy similarity_score path (reranker not loaded).
_RERANKER_SCORE_FLOOR = 0.30


def _calculate_confidence(results: list[dict]) -> float:
    """
    L1 confidence = best ABSOLUTE rerank confidence across the retrieved set.

    The reranker stores ``rerank_confidence`` = sigmoid(cross-encoder logit),
    which is batch-independent: a genuinely relevant top incident scores ~0.97
    and an irrelevant one ~0.0. We take the max over the candidates so a single
    strong historical match lets L1 resolve, while a query with no relevant
    match yields low confidence and correctly escalates to L2.

    Falls back to the legacy rescaled similarity_score only when the reranker is
    unavailable (no absolute confidence was attached).
    """
    if not results:
        return 0.0

    abs_confidences = [
        r["rerank_confidence"] for r in results
        if r.get("rerank_confidence") is not None
    ]
    if abs_confidences:
        return max(abs_confidences)

    # ── Fallback: reranker not loaded → rescale min-max similarity_score ──────
    top = results[:5]
    floor = _RERANKER_SCORE_FLOOR
    span = 1.0 - floor
    scores = [
        max(0.0, (r.get("similarity_score", 0.0) - floor) / span)
        for r in top
    ]
    weights = list(range(len(scores), 0, -1))  # [5, 4, 3, 2, 1]
    weighted = sum(s * w for s, w in zip(scores, weights))
    return weighted / sum(weights)


def _format_kb_context(results: list[dict], resolution_options: list[dict]) -> str:
    """Format retrieved incidents and resolution options into a readable LLM prompt block."""
    lines: list[str] = []

    if not results:
        return "No relevant incidents found in the knowledge base."

    lines.append(f"Retrieved {len(results)} similar past incident(s):")
    for i, r in enumerate(results[:5], 1):
        title = r.get("title") or "Incident"
        score = r.get("similarity_score", 0.0)
        desc = (r.get("description") or "")[:150]
        resolution = (r.get("resolution_notes") or "No resolution recorded")[:200]
        category = r.get("category") or "N/A"
        lines.append(
            f"\n[{i}] {title} | Category: {category} | Similarity: {score:.2f}\n"
            f"    Issue: {desc}\n"
            f"    Resolution: {resolution}"
        )

    if resolution_options:
        lines.append("\nUnique resolution approaches (by frequency):")
        for i, opt in enumerate(resolution_options[:3], 1):
            count = opt.get("occurrence_count", 1)
            text = (opt.get("resolution_text") or "")[:300]
            lines.append(f"\n  Fix {i} (used {count}×): {text}")

    return "\n".join(lines)
