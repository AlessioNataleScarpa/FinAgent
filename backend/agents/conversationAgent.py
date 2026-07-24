"""Follow-up agent: answers from saved ETF memory + optional web search."""

from __future__ import annotations

import logging
import json
import re
from typing import Any, Dict, List, Optional

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

    def _resolve_memory(
        self,
        messages: List[Message],
    ) -> tuple[Optional[str], str, Optional[Dict[str, Any]]]:
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
        return isin, context, analysis

    @staticmethod
    def _memory_fallback(
        latest_user: str,
        isin: Optional[str],
        analysis: Dict[str, Any],
    ) -> str:
        """Answer from persisted artifacts when the conversational LLM times out."""
        lowered = (latest_user or "").lower()
        header = (
            f"## Dati salvati per `{isin or analysis.get('isin') or 'N/D'}`\n\n"
            "_Risposta diretta dalla memoria della pipeline._\n\n"
        )

        if re.search(r"previs|forecast|futur|intervall|incertezz|graf", lowered):
            forecast = analysis.get("tsfm_forecast") or {}
            mean = forecast.get("mean") or []
            lower = forecast.get("lower_bound") or []
            upper = forecast.get("upper_bound") or []
            if mean and lower and upper:
                horizon_rows = []
                for label, step in (
                    ("1 anno", 12),
                    ("5 anni", 60),
                    ("10 anni", 120),
                    ("20 anni", 240),
                ):
                    if len(mean) >= step and len(lower) >= step and len(upper) >= step:
                        index = step - 1
                        horizon_rows.append(
                            f"| {label} | {float(mean[index]):.2f} | "
                            f"{float(lower[index]):.2f} – "
                            f"{float(upper[index]):.2f} |"
                        )
                if not horizon_rows:
                    horizon_rows.append(
                        f"| {len(mean)} mesi | {float(mean[-1]):.2f} | "
                        f"{float(lower[-1]):.2f} – {float(upper[-1]):.2f} |"
                    )
                table = ""
                if horizon_rows:
                    table = (
                        "\n| Orizzonte | Scenario centrale | Intervallo 80% |\n"
                        "|---|---:|---:|\n"
                        + "\n".join(horizon_rows)
                        + "\n"
                    )
                summary = (
                    f"- **Modello:** `{forecast.get('model', 'N/D')}`\n"
                    f"- **Stato:** `{forecast.get('status', 'N/D')}`\n"
                    f"- **Orizzonte:** {forecast.get('horizon', len(mean))} mesi\n"
                    f"{table}\n"
                    "Lo scenario centrale è la traiettoria più rappresentativa del "
                    "modello; i due limiti descrivono l'incertezza. Più sono distanti, "
                    "meno è prudente attribuire forza al segnale direzionale.\n\n"
                )
            else:
                summary = (
                    "La previsione strutturata non è disponibile per questa analisi.\n\n"
                )
            return header + summary + (analysis.get("forecast_charts") or "")

        if re.search(r"news|notizi|sentiment|impatto", lowered):
            return header + (
                analysis.get("technical")
                or analysis.get("news_data")
                or "Nessun dato news salvato."
            )

        if re.search(r"compos|sett|region|alloc|present", lowered):
            return header + "\n\n".join(
                filter(
                    None,
                    [
                        analysis.get("presentation"),
                        analysis.get("composition_charts"),
                    ],
                )
            )

        if re.search(r"storic|andament|passat|prezzo", lowered):
            return header + "\n\n".join(
                filter(
                    None,
                    [
                        analysis.get("timeline_charts"),
                        analysis.get("info_storici"),
                    ],
                )
            )

        forecast_json = json.dumps(
            analysis.get("tsfm_forecast") or {},
            ensure_ascii=False,
        )
        return (
            header
            + (analysis.get("report") or "Report non disponibile.")
            + f"\n\n### Forecast strutturato\n\n`{forecast_json}`"
        )

    def _maybe_web_context(self, latest_user: str, isin: Optional[str]) -> str:
        if not NEEDS_WEB.search(latest_user or ""):
            return ""
        query = latest_user.strip()
        if isin and isin not in query.upper():
            query = f"{query} ETF {isin}"
        return search_web(query)

    async def run(self, messages: List[Message]) -> str:
        latest_user = self.extract_latest_user_message(messages)
        isin, context, analysis = self._resolve_memory(messages)

        if not context or not analysis:
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

        try:
            llm = self.create_llm(system_prompt=system_prompt)
            response = await llm.ainvoke(latest_user)
            return self.normalize_llm_content(
                response.content if hasattr(response, "content") else response
            )
        except Exception as exc:
            logger.warning(
                "Conversation LLM unavailable for %s; answering from memory: %s",
                isin or "N/D",
                exc,
            )
            return self._memory_fallback(latest_user, isin, analysis)
