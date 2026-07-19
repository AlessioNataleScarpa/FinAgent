"""Follow-up agent: answers from saved ETF memory + optional web search."""

from __future__ import annotations

import logging
import re
from typing import List, Optional

try:
    from agents.base import BaseAgent
    from api.web_search import search_web
    from memory.store import get_memory_store
    from schemas.chat import Message
except ImportError:
    from backend.agents.base import BaseAgent
    from backend.api.web_search import search_web
    from backend.memory.store import get_memory_store
    from backend.schemas.chat import Message

logger = logging.getLogger(__name__)

ISIN_PATTERN = re.compile(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b", re.IGNORECASE)
NEEDS_WEB = re.compile(
    r"\b(notizie|news|oggi|recente|mercato|nvidia|apple|microsoft|tesla|"
    r"cerca|web|aggiorn|ultime|impatto|esce|lascia|rimuov)\b",
    re.IGNORECASE,
)


class ConversationAgent(BaseAgent):
    @property
    def model_id(self) -> str:
        return "Conversation Agent"

    @staticmethod
    def _extract_isin(text: str) -> Optional[str]:
        match = ISIN_PATTERN.search(text or "")
        return match.group(1).upper() if match else None

    def _resolve_memory(self, messages: List[Message]) -> tuple[Optional[str], str]:
        store = get_memory_store()
        latest_user = self.extract_latest_user_message(messages)
        isin = self._extract_isin(latest_user)

        if not isin:
            for message in reversed(messages):
                isin = self._extract_isin(message.content or "")
                if isin:
                    break

        analysis = store.get(isin) if isin else store.get_latest()
        if analysis and not isin:
            isin = analysis.get("isin")

        context = store.context_blob(isin) if isin else store.context_blob()
        return isin, context

    def _maybe_web_context(self, latest_user: str, isin: Optional[str]) -> str:
        if not NEEDS_WEB.search(latest_user or ""):
            return ""
        query = latest_user.strip()
        if isin and isin not in query.upper():
            query = f"{query} ETF {isin}"
        return search_web(query)

    async def run(self, messages: List[Message]) -> str:
        latest_user = self.extract_latest_user_message(messages)
        isin, context = self._resolve_memory(messages)

        if not context:
            return (
                "Non ho ancora un'analisi ETF in memoria.\n\n"
                "Chiedi prima a **gatewayAgent** di analizzare un ISIN "
                "(es. `Analizza IE00B4L5Y983`). Poi potrai continuare la conversazione."
            )

        web_block = self._maybe_web_context(latest_user, isin)
        web_section = (
            f"\n\n=== RICERCA WEB (opzionale, best-effort) ===\n{web_block}"
            if web_block
            else "\n\n=== RICERCA WEB ===\nNessun risultato web aggiuntivo."
        )

        system_prompt = (
            "Sei ConversationAgent, fase conversazionale dopo la pipeline di analisi ETF.\n"
            "L'utente ha già un report in memoria: rispondi in modo rapido e mirato.\n"
            "Priorità delle fonti:\n"
            "1) contesto in memoria (report, grafici, news della pipeline)\n"
            "2) eventuale ricerca web allegata\n"
            "Non inventare numeri assenti. Se i dati web e la memoria divergono, dillo.\n"
            "Rispondi SOLO con Markdown italiano finale leggibile.\n"
            "NON restituire JSON, liste Python, blocchi thinking/reasoning, né metadati.\n"
            "Se utile, puoi riproporre blocchi ```mermaid``` già presenti in memoria.\n"
            f"ISIN in focus: {isin or 'N/D'}\n\n"
            f"CONTESTO IN MEMORIA:\n{context}"
            f"{web_section}"
        )

        llm = self.create_llm(system_prompt=system_prompt)
        response = await llm.ainvoke(latest_user)
        return self.normalize_llm_content(
            response.content if hasattr(response, "content") else response
        )
