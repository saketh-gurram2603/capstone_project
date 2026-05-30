"""
In-memory chat session store with TTL-based eviction.

Sessions are created when a user starts a new guided troubleshooting conversation
and are reused across subsequent turns via session_id.  A background asyncio task
evicts sessions that have been idle longer than _SESSION_TTL_SECONDS.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.models.chat import ConversationMessage
from src.handlers.logger import get_logger, log_info, log_warning

logger = get_logger("chat.session_manager")

_SESSION_TTL_SECONDS = 1800   # 30 minutes
_CLEANUP_INTERVAL_SECONDS = 300


@dataclass
class ChatSession:
    session_id: str
    incident_description: str
    resolution_options: list[dict]   # ordered list of ResolutionOption dicts
    current_index: int = 0           # which option is currently being presented
    conversation_history: list[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_escalated: bool = False
    escalation_ticket_id: Optional[str] = None
    # Serialises concurrent turns on the SAME session so the read-modify-write of
    # current_index across await points (LLM calls) cannot interleave. Excluded
    # from equality/repr — it is runtime-only state.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, compare=False, repr=False)


class SessionManager:
    """Thread-safe (asyncio) in-memory session registry."""

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._lock = asyncio.Lock()

    def create_session(
        self,
        incident_description: str,
        resolution_options: list[dict],
    ) -> ChatSession:
        session = ChatSession(
            session_id=str(uuid.uuid4()),
            incident_description=incident_description,
            resolution_options=resolution_options,
        )
        self._sessions[session.session_id] = session
        log_info(
            "Chat session created | id=%s options=%d",
            session.session_id,
            len(resolution_options),
        )
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self._sessions.get(session_id)

    def save_session(self, session: ChatSession) -> None:
        session.last_active = datetime.now(timezone.utc)
        self._sessions[session.session_id] = session

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def _cleanup_expired(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_active).total_seconds() > _SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del self._sessions[sid]
            log_info("Chat session evicted (TTL) | id=%s", sid)
        if expired:
            log_info("Session cleanup | evicted=%d remaining=%d", len(expired), len(self._sessions))

    async def start_cleanup_task(self) -> None:
        """Run background cleanup loop. Call once from lifespan."""
        while True:
            await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
            try:
                await self._cleanup_expired()
            except Exception as exc:
                log_warning("Session cleanup error | error=%s", exc)


# Module-level singleton shared by the chat router
session_manager = SessionManager()
