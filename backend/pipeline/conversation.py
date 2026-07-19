"""
Node: CONVERSATION
Follow-up Q&A on saved ETF memory (+ optional web search).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

try:
    from agents.conversationAgent import ConversationAgent
    from schemas.chat import Message
except ImportError:
    from backend.agents.conversationAgent import ConversationAgent
    from backend.schemas.chat import Message

from .state import PipelineState

logger = logging.getLogger(__name__)


def _to_messages(state: PipelineState) -> List[Message]:
    raw = state.get("chat_messages") or []
    messages: List[Message] = []
    for item in raw:
        if isinstance(item, Message):
            messages.append(item)
        elif isinstance(item, dict) and "role" in item and "content" in item:
            messages.append(Message(role=str(item["role"]), content=str(item["content"])))

    if not messages:
        user_message = state.get("user_message") or state.get("clean_query") or ""
        if user_message:
            messages = [Message(role="user", content=user_message)]
    return messages


async def conversation_node(state: PipelineState) -> Dict[str, Any]:
    isin = state.get("isin", "N/A")
    logger.info("Executing conversation node for ISIN: %s", isin)
    messages = _to_messages(state)
    reply = await ConversationAgent().run(messages)
    return {"out_finale": reply}
