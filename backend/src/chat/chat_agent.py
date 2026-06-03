"""
ChatAgent — handles one guided troubleshooting turn.

Intent is classified per turn by a small LLM call (gpt-4o-mini) into
ADVANCE / RESOLVED / TROUBLESHOOT, with a deterministic fast-path for the
UI action buttons.  The assistant NEVER auto-advances on "it didn't work":
reporting difficulty keeps the user on the current fix and engages
troubleshooting.  Moving to the next fix is always an explicit user choice.

Flow:
  new session   → hybrid_search() → format Option 1 as numbered steps
  troubleshoot  → engage on the CURRENT fix, ask a follow-up (no index advance)
  advance       → explicit "next fix" only → advance current_index
  all exhausted → create L3 escalation ticket
  resolved      → confirmation message, session done
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.agents.l3_specialist import create_escalation_ticket
from src.chat.session_manager import ChatSession, SessionManager
from src.handlers.logger import get_logger, log_info, log_warning
from src.integrations.llm import chat_completion
from src.integrations.vector_db import VectorStore
from src.models.chat import ChatRequest, ChatResponse, ConversationMessage, OptionProgress
from src.retrieval.hybrid_search import hybrid_search

logger = get_logger("chat.agent")

# ── Intent classification ─────────────────────────────────────────────────────
# Free-text replies are classified by a small LLM call. Difficulty ("it's still
# not working") must NOT advance — it routes to TROUBLESHOOT so the assistant
# helps with the current fix. Only an explicit request advances.
_INTENT_SYSTEM_PROMPT = (
    "You classify a user's reply in a step-by-step IT troubleshooting chat. "
    "The user was given ONE specific fix to try. Reply with EXACTLY one word:\n"
    "ADVANCE — the user explicitly wants to stop this fix and try a different/next one "
    "(e.g. 'skip this', 'what else can I try', 'give me the next one').\n"
    "RESOLVED — the issue is now fixed/working (e.g. 'that worked', 'all good now').\n"
    "HELP — the user wants help getting THIS fix to work: they ask a question, report a "
    "specific error, or describe what happened / what they tried.\n"
    "UNCLEAR — the user only says it failed or they're stuck, with NO error, NO question, "
    "and NO details (e.g. 'this isn't working', 'nope', 'still broken', 'didn't help').\n"
    "Output only one word: ADVANCE, RESOLVED, HELP, or UNCLEAR."
)

# Exact UI action-button strings — matched deterministically (no LLM call).
_ACTION_NEXT = "This didn't work, try next fix"
_ACTION_RESOLVED = "Issue resolved"
_SUGGESTED_ACTIONS = [_ACTION_NEXT, _ACTION_RESOLVED]


class ChatAgent:
    """Stateless agent — all session state lives in ChatSession."""

    def __init__(self, vector_store: VectorStore, collection: str, app_config: dict) -> None:
        self._vector_store = vector_store
        self._collection = collection
        self._app_config = app_config

    # ── Public entry point ────────────────────────────────────────────────────

    async def handle_turn(
        self,
        session_manager: SessionManager,
        request: ChatRequest,
    ) -> ChatResponse:
        if request.session_id is None:
            return await self._handle_new_session(session_manager, request.message)

        session = session_manager.get_session(request.session_id)
        if session is None:
            raise KeyError(f"Session not found or expired: {request.session_id}")

        # Serialise concurrent turns on the same session to prevent the
        # current_index read-modify-write from interleaving across LLM awaits.
        async with session.lock:
            return await self._handle_continuation(session, request.message, session_manager)

    # ── New session ───────────────────────────────────────────────────────────

    async def _handle_new_session(
        self,
        session_manager: SessionManager,
        message: str,
    ) -> ChatResponse:
        log_info("Chat new session | query='%s'", message[:80])

        search_result = await hybrid_search(
            query=message,
            vector_store=self._vector_store,
            collection=self._collection,
            app_config=self._app_config,
        )
        resolution_options: list[dict] = search_result.get("resolution_options", [])

        session = session_manager.create_session(message, resolution_options)
        session.conversation_history.append(
            ConversationMessage(role="user", content=message)
        )

        if not resolution_options:
            log_warning("Chat: no resolution options found | query='%s'", message[:80])
            return await self._escalate(
                session,
                session_manager,
                escalation_reason="No matching resolution options found in the knowledge base.",
            )

        formatted = await self._format_option(
            option=resolution_options[0],
            option_index=0,
            total=len(resolution_options),
            incident_description=message,
        )
        session.conversation_history.append(
            ConversationMessage(role="assistant", content=formatted)
        )
        session_manager.save_session(session)

        return ChatResponse(
            session_id=session.session_id,
            message=formatted,
            option_progress=OptionProgress(current=1, total=len(resolution_options)),
            suggested_actions=_SUGGESTED_ACTIONS,
        )

    # ── Continuation ──────────────────────────────────────────────────────────

    async def _handle_continuation(
        self,
        session: ChatSession,
        message: str,
        session_manager: SessionManager,
    ) -> ChatResponse:
        session.conversation_history.append(
            ConversationMessage(role="user", content=message)
        )

        intent = await self._classify_intent(session, message)
        log_info("Chat continuation | intent=%s session=%s", intent, session.session_id)

        if intent == "RESOLVED":
            reply = (
                "Great — glad that resolved your incident! "
                "If the issue returns, feel free to start a new session. "
                "Stay safe out there."
            )
            session.conversation_history.append(
                ConversationMessage(role="assistant", content=reply)
            )
            session_manager.save_session(session)
            return ChatResponse(
                session_id=session.session_id,
                message=reply,
                suggested_actions=[],
            )

        if intent == "OFFER_CHOICE":
            return await self._offer_choice(session, session_manager)

        if intent == "TROUBLESHOOT":
            return await self._troubleshoot(session, message, session_manager)

        # NEXT_OPTION — advance to the next fix. Reached only on an explicit
        # request (the action button or a clear "next/skip"), never on a bare
        # "it didn't work". Feedback comes from the thumbs UI, not this button.
        next_index = session.current_index + 1
        if next_index >= len(session.resolution_options):
            return await self._escalate(
                session,
                session_manager,
                escalation_reason="All resolution options have been attempted without success.",
            )

        session.current_index = next_index
        option = session.resolution_options[next_index]
        formatted = await self._format_option(
            option=option,
            option_index=next_index,
            total=len(session.resolution_options),
            incident_description=session.incident_description,
        )
        session.conversation_history.append(
            ConversationMessage(role="assistant", content=formatted)
        )
        session_manager.save_session(session)

        return ChatResponse(
            session_id=session.session_id,
            message=formatted,
            option_progress=OptionProgress(
                current=next_index + 1,
                total=len(session.resolution_options),
            ),
            suggested_actions=_SUGGESTED_ACTIONS,
        )

    # ── Intent detection ──────────────────────────────────────────────────────

    async def _classify_intent(self, session: ChatSession, message: str) -> str:
        """
        Returns NEXT_OPTION | RESOLVED | TROUBLESHOOT | OFFER_CHOICE.

        UI action buttons are matched deterministically (no LLM call). Every
        other free-text reply is classified by a small gpt-4o-mini call:
          • HELP    (question / error / details)      → TROUBLESHOOT
          • UNCLEAR (bare "it didn't work")            → OFFER_CHOICE
        On any failure we default to OFFER_CHOICE — the assistant never
        auto-advances when intent is unclear, but it also doesn't presume the
        user wants a deep troubleshoot; it asks them which they want.
        """
        text = message.strip()
        low = text.lower()

        # ── Deterministic fast-path for the action buttons ─────────────────────
        if text == _ACTION_RESOLVED or low in ("issue resolved", "resolved"):
            return "RESOLVED"
        if text == _ACTION_NEXT or "try next fix" in low or low in ("next", "next fix", "skip"):
            return "NEXT_OPTION"

        # ── LLM classification for everything else ─────────────────────────────
        current = ""
        if session.resolution_options and session.current_index < len(session.resolution_options):
            current = session.resolution_options[session.current_index].get("resolution_text", "")[:300]

        try:
            l1_model = self._app_config.get("LLM", {}).get("L1_MODEL", "synapt-dev-gpt-4o-mini")
            reply, _ = await chat_completion(
                messages=[
                    {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Current fix being attempted:\n{current or 'N/A'}\n\n"
                        f"User reply:\n{text}"
                    )},
                ],
                model=l1_model,
                temperature=0,
                max_tokens=4,
            )
            label = reply.strip().upper()
        except Exception as exc:
            log_warning("Chat: intent classification failed, offering choice | error=%s", exc)
            return "OFFER_CHOICE"

        if "RESOLVED" in label:
            return "RESOLVED"
        if "ADVANCE" in label:
            return "NEXT_OPTION"
        if "HELP" in label:
            return "TROUBLESHOOT"
        return "OFFER_CHOICE"

    # ── Feedback capture ──────────────────────────────────────────────────────

    def _current_fix_meta(self, session: ChatSession) -> dict | None:
        """Metadata for the fix the user is currently reacting to (or None)."""
        options = session.resolution_options
        if not options or session.current_index >= len(options):
            return None
        option = options[session.current_index]
        return {
            "resolution_text": option.get("resolution_text", ""),
            "incident_ids": option.get("source_incident_ids", []),
            "occurrence_count": option.get("occurrence_count", 1),
            "fix_index": session.current_index + 1,
            "fix_total": len(options),
        }

    async def _safe_record_feedback(
        self,
        session: ChatSession,
        sentiment: str,
        reason: str,
    ) -> None:
        """
        Persist feedback on the current fix for admin review.
        Wrapped so a feedback-store failure never breaks the chat turn.
        """
        try:
            from src.feedback.feedback_store import record_feedback

            meta = self._current_fix_meta(session)
            if not meta:
                return
            await record_feedback(
                session_id=session.session_id,
                query=session.incident_description,
                sentiment=sentiment,
                fix_index=meta["fix_index"],
                fix_total=meta["fix_total"],
                resolution_text=meta["resolution_text"],
                incident_ids=meta["incident_ids"],
                occurrence_count=meta["occurrence_count"],
                reason=reason,
            )
        except Exception as exc:
            log_warning("Chat: feedback capture failed | error=%s", exc)

    # ── Format a resolution option as numbered steps ──────────────────────────

    async def _format_option(
        self,
        option: dict,
        option_index: int,
        total: int,
        incident_description: str,
    ) -> str:
        resolution_text = option.get("resolution_text", "")
        occurrence = option.get("occurrence_count", 1)

        system_prompt = (
            "You are a helpful IT support assistant. "
            "Format the provided resolution as clear, numbered troubleshooting steps. "
            "Each step should be concise and actionable. "
            "Do NOT add information that is not present in the resolution. "
            "Use markdown formatting (numbered list, bold for key actions)."
        )
        user_prompt = (
            f"Incident: {incident_description}\n\n"
            f"Resolution approach ({occurrence} historical occurrence{'s' if occurrence != 1 else ''}):\n"
            f"{resolution_text}"
        )

        try:
            l1_model = self._app_config.get("LLM", {}).get("L1_MODEL", "synapt-dev-gpt-4o-mini")
            response, _ = await chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=l1_model,
                temperature=0,
                max_tokens=600,
            )
            header = f"**Fix {option_index + 1} of {total}** _(verified {occurrence}× in KB)_\n\n"
            return header + response
        except Exception as exc:
            log_warning("Chat: LLM format_option failed, using raw text | error=%s", exc)
            header = f"**Fix {option_index + 1} of {total}**\n\n"
            return header + resolution_text

    # ── Bare failure → offer a choice (no LLM call, never presumes) ───────────

    async def _offer_choice(
        self,
        session: ChatSession,
        session_manager: SessionManager,
    ) -> ChatResponse:
        """
        The user reported failure with no error/question/detail. Don't presume:
        acknowledge briefly and let them pick — dig in, or move on. Templated so
        it works even when the LLM is unavailable.
        """
        idx = session.current_index
        has_next = (idx + 1) < len(session.resolution_options)

        if has_next:
            tail = (
                f"or tap **{_ACTION_NEXT}** and I'll show you the next one."
            )
        else:
            tail = (
                "or — since this was the last fix in the knowledge base — I can escalate "
                "it to a specialist. Just let me know."
            )
        reply = (
            f"Sorry, Fix {idx + 1} didn't do it. Want me to help dig into *why* it's not "
            f"working? If so, tell me what happened — any error message or where it got "
            f"stuck — {tail}"
        )

        session.conversation_history.append(
            ConversationMessage(role="assistant", content=reply)
        )
        session_manager.save_session(session)

        progress = None
        if session.resolution_options:
            progress = OptionProgress(
                current=idx + 1,
                total=len(session.resolution_options),
            )

        return ChatResponse(
            session_id=session.session_id,
            message=reply,
            option_progress=progress,
            suggested_actions=_SUGGESTED_ACTIONS,
        )

    # ── Troubleshoot the current fix (engage, do NOT advance) ─────────────────

    async def _troubleshoot(
        self,
        session: ChatSession,
        message: str,
        session_manager: SessionManager,
    ) -> ChatResponse:
        current_option = (
            session.resolution_options[session.current_index]
            if session.resolution_options
            else {}
        )
        total = len(session.resolution_options) or 1
        system_prompt = (
            "You are an IT support assistant helping a user work through ONE specific fix, "
            "step by step. The user is stuck, hit an error, or has a question about the fix "
            "they are currently trying. Work the problem WITH them: briefly acknowledge what "
            "they tried, explain what to check next or how to adapt the current step, and end "
            "with ONE focused follow-up question to keep diagnosing together. "
            "Do NOT tell them to give up or switch to a different fix — they can choose to move "
            "on themselves using the 'try next fix' action. Be concise and practical; use markdown."
        )
        user_prompt = (
            f"Incident: {session.incident_description}\n\n"
            f"Fix they are currently trying (step {session.current_index + 1} of {total}):\n"
            f"{current_option.get('resolution_text', 'N/A')}\n\n"
            f"Their message: {message}"
        )

        try:
            l1_model = self._app_config.get("LLM", {}).get("L1_MODEL", "synapt-dev-gpt-4o-mini")
            response, _ = await chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=l1_model,
                temperature=0,
                max_tokens=450,
            )
        except Exception as exc:
            log_warning("Chat: troubleshooting reply failed | error=%s", exc)
            response = (
                "I'm sorry, I couldn't process that just now. Can you tell me what happened "
                "when you tried the current step — any error message or where it got stuck?"
            )

        session.conversation_history.append(
            ConversationMessage(role="assistant", content=response)
        )
        session_manager.save_session(session)

        # Keep same option_progress so the user knows they haven't advanced
        progress = None
        if session.resolution_options:
            progress = OptionProgress(
                current=session.current_index + 1,
                total=len(session.resolution_options),
            )

        return ChatResponse(
            session_id=session.session_id,
            message=response,
            option_progress=progress,
            suggested_actions=_SUGGESTED_ACTIONS,
        )

    # ── Escalation ────────────────────────────────────────────────────────────

    async def _escalate(
        self,
        session: ChatSession,
        session_manager: SessionManager,
        escalation_reason: str = "All resolution options exhausted via chat.",
    ) -> ChatResponse:
        conversation_summary = "\n".join(
            f"{m.role}: {m.content[:200]}"
            for m in session.conversation_history[-6:]
        )

        try:
            result = await create_escalation_ticket(
                description=session.incident_description,
                l1_summary=conversation_summary,
                escalation_reason=escalation_reason,
            )
            ticket_id = result["ticket_id"]
        except Exception as exc:
            log_warning("Chat: escalation ticket creation failed | error=%s", exc)
            ticket_id = None

        session.is_escalated = True
        session.escalation_ticket_id = ticket_id
        session_manager.save_session(session)

        if ticket_id:
            reply = (
                f"I've exhausted all available resolution options from the knowledge base "
                f"without finding a fix for your incident.\n\n"
                f"**Your case has been escalated to the specialist team.**\n\n"
                f"- **Ticket ID:** `{ticket_id}`\n"
                f"- **Status:** OPEN — assigned to IT-OPS escalation queue\n"
                f"- **Reason:** {escalation_reason}\n\n"
                f"A specialist will review your case shortly."
            )
        else:
            reply = (
                "I've exhausted all available resolution options. "
                "Please contact your IT support team directly for further assistance."
            )

        session.conversation_history.append(
            ConversationMessage(role="assistant", content=reply)
        )

        return ChatResponse(
            session_id=session.session_id,
            message=reply,
            is_escalated=True,
            escalation_ticket_id=ticket_id,
            all_options_exhausted=True,
            suggested_actions=[],
        )
