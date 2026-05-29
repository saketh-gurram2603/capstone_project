"""
ChatAgent — handles one guided troubleshooting turn.

Flow:
  new session  → hybrid_search() → format Option 1 as numbered steps
  next fix     → advance current_index → format next option
  all exhausted → create L3 escalation ticket
  question     → freeform GPT-4o-mini answer with context (no index advance)
  resolved     → confirmation message, session done
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

# ── Intent keyword sets ───────────────────────────────────────────────────────
_NEXT_KEYWORDS = frozenset([
    "didn't work", "didnt work", "not working", "does not work", "doesn't work",
    "failed", "still broken", "no luck", "try next", "next fix", "next option",
    "not fixed", "still failing", "another", "else",
])
_RESOLVED_KEYWORDS = frozenset([
    "worked", "it worked", "resolved", "fixed", "thank", "thanks",
    "got it", "solved", "all good", "success", "done",
])

_SUGGESTED_ACTIONS = ["This didn't work, try next fix", "Issue resolved"]


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

        intent = self._detect_intent(message)
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

        if intent == "QUESTION":
            return await self._answer_question(session, message, session_manager)

        # NEXT_OPTION — advance the index
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

    def _detect_intent(self, message: str) -> str:
        """
        Returns NEXT_OPTION | RESOLVED | QUESTION.

        Rule-based keyword matching — fast and deterministic.
        Falls back to NEXT_OPTION for short ambiguous messages.
        """
        lower = message.lower()

        for kw in _RESOLVED_KEYWORDS:
            if kw in lower:
                return "RESOLVED"

        for kw in _NEXT_KEYWORDS:
            if kw in lower:
                return "NEXT_OPTION"

        # Longer messages with no keywords are likely clarifying questions
        if len(message.strip()) > 40:
            return "QUESTION"

        return "NEXT_OPTION"

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
            l1_model = self._app_config.get("LLM", {}).get("L1_MODEL", "gpt-4o-mini")
            response, _ = await chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=l1_model,
                temperature=0.2,
                max_tokens=600,
            )
            header = f"**Fix {option_index + 1} of {total}** _(verified {occurrence}× in KB)_\n\n"
            return header + response
        except Exception as exc:
            log_warning("Chat: LLM format_option failed, using raw text | error=%s", exc)
            header = f"**Fix {option_index + 1} of {total}**\n\n"
            return header + resolution_text

    # ── Answer a mid-session question ─────────────────────────────────────────

    async def _answer_question(
        self,
        session: ChatSession,
        question: str,
        session_manager: SessionManager,
    ) -> ChatResponse:
        current_option = (
            session.resolution_options[session.current_index]
            if session.resolution_options
            else {}
        )
        system_prompt = (
            "You are a helpful IT support assistant. "
            "Answer the user's question in the context of the incident they reported "
            "and the fix they are currently trying. Be concise."
        )
        user_prompt = (
            f"Incident: {session.incident_description}\n\n"
            f"Current fix being attempted:\n{current_option.get('resolution_text', 'N/A')}\n\n"
            f"User question: {question}"
        )

        try:
            l1_model = self._app_config.get("LLM", {}).get("L1_MODEL", "gpt-4o-mini")
            response, _ = await chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model=l1_model,
                temperature=0.3,
                max_tokens=400,
            )
        except Exception as exc:
            log_warning("Chat: question answering failed | error=%s", exc)
            response = "I'm sorry, I couldn't process your question right now. Please try again."

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
