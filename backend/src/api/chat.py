"""
Chat API
  POST /chat — one conversational turn of guided troubleshooting
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from src.chat.chat_agent import ChatAgent
from src.chat.session_manager import session_manager
from src.core.dependencies import get_app_config, get_vector_store
from src.handlers.logger import get_logger, log_info
from src.integrations.vector_db import VectorStore
from src.models.chat import ChatRequest, ChatResponse

logger = get_logger("api.chat")

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse, summary="Guided troubleshooting chat")
async def chat_turn(
    body: ChatRequest,
    vector_store: VectorStore = Depends(get_vector_store),
    app_config: dict = Depends(get_app_config),
) -> ChatResponse:
    """
    Handle one conversational turn of guided troubleshooting.

    - Send with ``session_id=null`` to start a new session.
      The system searches the incident KB and presents the top-ranked fix as numbered steps.
    - Send with an existing ``session_id`` to continue the conversation.
      Saying *"this didn't work"* advances to the next resolution option;
      all options exhausted → automatic L3 escalation ticket.
    """
    log_info("POST /chat | session_id=%s msg_len=%d", body.session_id, len(body.message))

    collection = app_config.get("QDRANT", {}).get("COLLECTION_NAME", "incidents")
    agent = ChatAgent(
        vector_store=vector_store,
        collection=collection,
        app_config=app_config,
    )

    try:
        return await agent.handle_turn(session_manager, body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected chat error")
        raise HTTPException(status_code=500, detail="Chat request failed unexpectedly.") from exc
