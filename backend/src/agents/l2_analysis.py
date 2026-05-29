"""
L2 Analysis node — web-augmented synthesis when L1 confidence is insufficient.

Flow:
  1. Run Tavily web search on the original query
  2. Combine L1 KB context + web results into a rich prompt
  3. Call GPT-4o to produce structured analysis: root_cause, resolution_steps,
     confidence, sources
  4. Parse confidence from the JSON response (fallback: keyword heuristic)
  5. Gate: confidence ≥ threshold → END (L2 resolved)
            confidence < threshold → escalate to L3

The node is created via make_l2_node() factory so app_config is injected at
graph-build time.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from src.agents.state import IncidentState
from src.agents.tools import tavily_web_search
from src.handlers.logger import get_logger, log_info, log_warning
from src.integrations.llm import chat_completion

logger = get_logger("agents.l2_analysis")

_L2_SYSTEM_PROMPT = """\
You are a senior IT incident analyst with access to both a historical incident \
knowledge base (already searched) and fresh web results.

Synthesise ALL available information to produce a structured JSON response ONLY — \
no prose outside the JSON object.

Required JSON schema:
{
  "root_cause": "<1-2 sentence likely root cause>",
  "resolution_steps": ["<step 1>", "<step 2>", "..."],
  "confidence": <float 0.0-1.0>,
  "sources": ["KB", "Web"] or subset
}

Guidance:
- confidence: 0.9+ if you are certain, 0.7-0.9 if likely, 0.5-0.7 if plausible, <0.5 if unclear
- resolution_steps: ordered from most to least impactful
- sources: include "KB" if KB results were relevant, "Web" if web results were relevant
- If neither source is helpful, set confidence below 0.50
"""


def make_l2_node(app_config: dict) -> Callable[[IncidentState], dict]:
    """
    Factory: returns an async L2 analysis node.
    Called once when the graph is compiled.
    """
    l2_model = app_config.get("LLM", {}).get("L2_MODEL", "gpt-4o")
    l2_threshold = app_config.get("RETRIEVAL", {}).get("L2_CONFIDENCE_THRESHOLD", 0.55)
    tavily_max = app_config.get("RETRIEVAL", {}).get("TAVILY_MAX_RESULTS", 5)

    async def l2_node(state: IncidentState) -> dict:
        query = state.get("query", "")
        impact = state.get("impact", "")
        urgency = state.get("urgency", "")
        l1_summary = state.get("l1_summary", "")
        retrieved = state.get("retrieved_incidents", [])
        resolution_options = state.get("resolution_options", [])
        escalation_reason = state.get("escalation_reason", "")

        log_info("L2 node | query='%s'", query[:80])

        # ── 1. Web search ──────────────────────────────────────────────────────
        web_results = await tavily_web_search(query, max_results=tavily_max)
        log_info("L2 web search | results=%d", len(web_results))

        # ── 2. Build combined context ──────────────────────────────────────────
        context = _format_l2_context(
            query=query,
            impact=impact,
            urgency=urgency,
            l1_summary=l1_summary,
            retrieved=retrieved,
            resolution_options=resolution_options,
            web_results=web_results,
            escalation_reason=escalation_reason,
        )

        messages = [
            {"role": "system", "content": _L2_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]

        # ── 3. LLM call (GPT-4o) ──────────────────────────────────────────────
        fallback_used = False
        raw_response = ""
        try:
            raw_response, fallback_used = await chat_completion(
                messages=messages,
                model=l2_model,
                temperature=0.2,
                max_tokens=768,
            )
        except Exception as exc:
            log_warning("L2 LLM call failed | error=%s", exc)
            raw_response = json.dumps({
                "root_cause": "Unable to determine — LLM unavailable.",
                "resolution_steps": ["Escalate to L3 specialist team."],
                "confidence": 0.0,
                "sources": [],
            })
            fallback_used = True

        # ── 4. Parse structured JSON response ─────────────────────────────────
        parsed = _parse_l2_response(raw_response)
        synthesis = _format_synthesis(parsed)
        l2_confidence = parsed.get("confidence", 0.0)

        if fallback_used:
            l2_confidence = min(l2_confidence, 0.40)

        log_info("L2 confidence=%.3f threshold=%.2f", l2_confidence, l2_threshold)

        # ── 5. Escalation reason if still insufficient ─────────────────────────
        new_escalation_reason = None
        if l2_confidence < l2_threshold:
            new_escalation_reason = (
                f"L2 confidence {l2_confidence:.2f} below threshold {l2_threshold:.2f}. "
                f"Web search returned {len(web_results)} result(s). Requires specialist."
            )

        return {
            "web_search_results": web_results,
            "l2_synthesis": synthesis,
            "l2_confidence": round(l2_confidence, 4),
            "final_answer": synthesis,
            "escalation_level": "L2",
            "escalation_reason": new_escalation_reason,
            "model_used": l2_model if not fallback_used else "flan-t5-base",
            "fallback_used": fallback_used,
        }

    return l2_node


# ── Private helpers ───────────────────────────────────────────────────────────


def _format_l2_context(
    query: str,
    impact: str,
    urgency: str,
    l1_summary: str,
    retrieved: list[dict],
    resolution_options: list[dict],
    web_results: list[dict],
    escalation_reason: str,
) -> str:
    """Build the full L2 user-turn context block."""
    lines: list[str] = []

    lines.append(f"Incident: {query}")
    lines.append(f"Impact: {impact or 'Unknown'} | Urgency: {urgency or 'Unknown'}")

    if escalation_reason:
        lines.append(f"Escalation reason: {escalation_reason}")

    # KB context (L1 summary + top incidents)
    lines.append("\n--- Knowledge Base Summary (L1) ---")
    if l1_summary:
        lines.append(l1_summary[:500])
    else:
        lines.append("No L1 summary available.")

    if retrieved:
        lines.append(f"\nTop {min(3, len(retrieved))} KB matches:")
        for i, r in enumerate(retrieved[:3], 1):
            title = r.get("title") or "Incident"
            resolution = (r.get("resolution_notes") or "")[:150]
            lines.append(f"  [{i}] {title}: {resolution}")

    if resolution_options:
        lines.append("\nUnique KB resolution options:")
        for i, opt in enumerate(resolution_options[:2], 1):
            count = opt.get("occurrence_count", 1)
            text = (opt.get("resolution_text") or "")[:200]
            lines.append(f"  Fix {i} (used {count}×): {text}")

    # Web results
    lines.append("\n--- Web Search Results ---")
    if web_results:
        for i, r in enumerate(web_results[:5], 1):
            title = r.get("title", "")
            content = (r.get("content") or "")[:250]
            url = r.get("url", "")
            lines.append(f"\n[Web {i}] {title}\n  {content}\n  Source: {url}")
    else:
        lines.append("No web results available.")

    return "\n".join(lines)


def _parse_l2_response(raw: str) -> dict:
    """
    Extract structured JSON from the LLM response.
    Handles cases where the model wraps JSON in ```json ... ``` fences.
    Falls back to a safe default on any parse failure.
    """
    # Strip markdown fences if present
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        data = json.loads(text)
        # Validate / coerce confidence to float in [0, 1]
        raw_conf = data.get("confidence", 0.5)
        data["confidence"] = max(0.0, min(1.0, float(raw_conf)))
        return data
    except (json.JSONDecodeError, ValueError, TypeError):
        log_warning("L2 JSON parse failed — using heuristic confidence")
        # Heuristic: presence of "confident" / "likely" → moderate confidence
        confidence = 0.55 if re.search(r"\b(confident|likely|probably)\b", raw, re.I) else 0.35
        return {
            "root_cause": raw[:300] if raw else "Unable to determine root cause.",
            "resolution_steps": ["Review the raw analysis text above."],
            "confidence": confidence,
            "sources": ["Web"] if "web" in raw.lower() else [],
        }


def _format_synthesis(parsed: dict) -> str:
    """Convert the parsed L2 JSON into a readable plain-text synthesis."""
    root_cause = parsed.get("root_cause", "Unknown")
    steps = parsed.get("resolution_steps", [])
    confidence = parsed.get("confidence", 0.0)
    sources = parsed.get("sources", [])

    lines = [
        f"Root Cause: {root_cause}",
        "",
        "Recommended Resolution Steps:",
    ]
    for i, step in enumerate(steps, 1):
        lines.append(f"  {i}. {step}")

    lines.append("")
    lines.append(
        f"Confidence: {confidence:.0%} | Sources: {', '.join(sources) if sources else 'None'}"
    )
    return "\n".join(lines)
