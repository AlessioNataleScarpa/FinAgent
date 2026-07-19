import json
import logging
import re
import time
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

try:
    from agents.base import BaseAgent
    from memory.store import get_memory_store
    from prompts import build_gateway_system_prompt
    from schemas.chat import Message
    from schemas.routing import RouterIntentSchema
except ImportError:
    from backend.agents.base import BaseAgent
    from backend.memory.store import get_memory_store
    from backend.prompts import build_gateway_system_prompt
    from backend.schemas.chat import Message
    from backend.schemas.routing import RouterIntentSchema

logger = logging.getLogger(__name__)

ISIN_PATTERN = re.compile(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b", re.IGNORECASE)
REANALYZE_PATTERN = re.compile(
    r"\b(analizza|analisi completa|riesegui|ricalcola|nuova analisi|"
    r"rifai(?:\s+l['\u2019]?analisi)?|rianalizza|pipeline|partiamo da capo|"
    r"parlami|dimmi|raccont|info(?:rmazioni)?|presenta)\b",
    re.IGNORECASE,
)
REPORT_MARKERS = (
    "Analisi salvata in memoria",
    "```mermaid",
    "Report di analisi",
    "REPORT COMPLETO",
    "Executive summary",
)


class GatewayAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.structured_llm = self.create_structured_llm(RouterIntentSchema)

    @property
    def model_id(self) -> str:
        return "Gateway Agent"

    @staticmethod
    def _extract_router_state(content: str) -> Optional[dict]:
        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed_content, dict):
            return None

        if parsed_content.get("status") == "accepted" or "isin" in parsed_content:
            return {
                "status": parsed_content.get("status"),
                "intent": parsed_content.get("intent"),
                "isin": parsed_content.get("isin"),
                "clean_query": parsed_content.get("clean_query"),
            }

        return None

    @staticmethod
    def _extract_isin_fallback(text: str) -> Optional[str]:
        match = ISIN_PATTERN.search(text or "")
        return match.group(1).upper() if match else None

    def _extract_isin_from_history(self, messages: List[Message]) -> Optional[str]:
        for message in reversed(messages):
            isin = self._extract_isin_fallback(message.content or "")
            if isin:
                return isin
        return None

    @staticmethod
    def _chat_has_prior_report(messages: List[Message]) -> bool:
        for message in messages:
            if message.role != "assistant":
                continue
            content = message.content or ""
            if any(marker in content for marker in REPORT_MARKERS):
                return True
        return False

    def _build_system_prompt(self, previous_state_json: Optional[str]) -> str:
        return build_gateway_system_prompt(previous_state_json)

    def _should_use_conversation(
        self,
        *,
        messages: List[Message],
        latest_user_message: str,
        isin: Optional[str],
        memory: Optional[dict],
        isin_in_latest: Optional[str],
    ) -> bool:
        """Follow-up Q&A only when we already analyzed THIS ISIN in this chat."""
        if not memory:
            return False
        if not self._chat_has_prior_report(messages):
            return False

        memory_isin = (memory.get("isin") or "").upper() or None
        if isin and memory_isin and isin != memory_isin:
            return False

        # Explicit (re)analysis / "parlami di <ISIN>" with ISIN in this message
        # always goes to full pipeline if it's a fresh ask for that instrument.
        if isin_in_latest and REANALYZE_PATTERN.search(latest_user_message or ""):
            # If memory already has this ISIN and user just says "parlami" again
            # without asking for a re-run keyword like analizza/riesegui, prefer conversation.
            hard_rerun = re.search(
                r"\b(analizza|riesegui|ricalcola|nuova analisi|rianalizza|pipeline)\b",
                latest_user_message or "",
                re.IGNORECASE,
            )
            if hard_rerun:
                return False
            # "parlami di IE00..." on an ISIN already in memory → conversation (fast)
            return True

        if REANALYZE_PATTERN.search(latest_user_message or ""):
            hard_rerun = re.search(
                r"\b(analizza|riesegui|ricalcola|nuova analisi|rianalizza|pipeline)\b",
                latest_user_message or "",
                re.IGNORECASE,
            )
            if hard_rerun:
                return False

        return True

    @staticmethod
    def _serialize_messages(messages: List[Message]) -> list:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def _invoke_graph(
        self,
        *,
        mode: str,
        isin: Optional[str],
        clean_query: Optional[str],
        latest_user_message: str,
        messages: List[Message],
    ) -> str:
        try:
            from pipeline.graph import graph
        except ImportError:
            from backend.pipeline.graph import graph

        if mode == "full_analysis" and not isin:
            return (
                "Per avviare l'analisi completa mi serve un codice ISIN "
                "(es. `IE00B4L5Y983`)."
            )

        pipeline_input = {
            "mode": mode,
            "isin": isin or "",
            "clean_query": clean_query or latest_user_message,
            "user_message": latest_user_message,
            "chat_messages": self._serialize_messages(messages),
        }
        started = time.perf_counter()
        logger.info("Gateway: graph mode=%s ISIN=%s", mode, isin or "N/A")
        state = await graph.ainvoke(pipeline_input)
        logger.info(
            "Gateway: graph finished mode=%s in %.2fs",
            mode,
            time.perf_counter() - started,
        )
        if isinstance(state, dict) and state.get("out_finale"):
            return state["out_finale"]

        if mode == "conversation":
            return "Non sono riuscito a rispondere dalla memoria. Riprova o riesegui l'analisi."

        payload = {
            "status": "accepted",
            "intent": "etf",
            "isin": isin,
            "clean_query": clean_query or latest_user_message,
            "message": "Richiesta ETF accettata. Pipeline di analisi non ha restituito output.",
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    async def run(self, messages: List[Message]) -> str:
        latest_user_message = self.extract_latest_user_message(messages)
        store = get_memory_store()

        # ISIN written in THIS user message (highest priority).
        isin_in_latest = self._extract_isin_fallback(latest_user_message)
        isin = isin_in_latest or self._extract_isin_from_history(messages)

        memory = store.get(isin) if isin else store.get_latest()
        if memory and not isin:
            isin = memory.get("isin")

        # New ISIN in the current message and no saved analysis yet → full pipeline,
        # never touch the Gemini router (it can hang and freeze OpenWebUI).
        if isin_in_latest and not store.get(isin_in_latest):
            logger.info("Gateway: new ISIN %s → full_analysis (no router LLM)", isin_in_latest)
            return await self._invoke_graph(
                mode="full_analysis",
                isin=isin_in_latest,
                clean_query=latest_user_message,
                latest_user_message=latest_user_message,
                messages=messages,
            )

        if self._should_use_conversation(
            messages=messages,
            latest_user_message=latest_user_message,
            isin=isin,
            memory=memory,
            isin_in_latest=isin_in_latest,
        ):
            return await self._invoke_graph(
                mode="conversation",
                isin=isin,
                clean_query=latest_user_message,
                latest_user_message=latest_user_message,
                messages=messages,
            )

        # Any ISIN present in the current message → full analysis, skip router LLM.
        if isin_in_latest:
            logger.info("Gateway: ISIN in message %s → full_analysis (no router LLM)", isin_in_latest)
            return await self._invoke_graph(
                mode="full_analysis",
                isin=isin_in_latest,
                clean_query=latest_user_message,
                latest_user_message=latest_user_message,
                messages=messages,
            )

        previous_state_json = self.extract_previous_assistant_state(
            messages,
            self._extract_router_state,
        )
        if not previous_state_json and memory:
            previous_state_json = json.dumps(
                {
                    "status": "accepted",
                    "intent": "etf",
                    "isin": memory.get("isin"),
                    "clean_query": memory.get("clean_query"),
                },
                ensure_ascii=False,
            )

        # No ISIN in current message: lightweight heuristic before LLM router.
        lowered = (latest_user_message or "").lower()
        if any(token in lowered for token in ("ciao", "hey", "buongiorno", "buonasera")):
            return (
                "Ciao! Sono il gateway ETF. Inviami un ISIN "
                "(es. `parlami di IE00B4L5Y983`) per avviare l'analisi."
            )

        system_prompt = self._build_system_prompt(previous_state_json)
        langchain_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=latest_user_message),
        ]

        logger.info("Gateway: calling router LLM (no ISIN in latest message)")
        try:
            response = await self.structured_llm.ainvoke(langchain_messages)
        except Exception as exc:
            logger.warning("Gateway router LLM failed: %s", exc)
            if memory:
                return await self._invoke_graph(
                    mode="conversation",
                    isin=memory.get("isin"),
                    clean_query=latest_user_message,
                    latest_user_message=latest_user_message,
                    messages=messages,
                )
            return (
                "Non sono riuscito a classificare la richiesta. "
                "Invia un ISIN (es. `IE00B5BMR087`)."
            )

        if isinstance(response, RouterIntentSchema):
            intent_data = response.model_dump()
        elif isinstance(response, BaseModel):
            intent_data = response.model_dump()
        elif isinstance(response, dict):
            intent_data = response
        else:
            response_content = response.content if hasattr(response, "content") else response
            stripped_content = self.strip_code_fences(str(response_content))
            try:
                intent_data = RouterIntentSchema.model_validate_json(stripped_content).model_dump()
            except Exception:
                return "Errore interno del gateway: classificazione intent fallita."

        if not intent_data.get("is_routable", False):
            direct_response = intent_data.get("direct_response")
            if direct_response:
                return direct_response
            return (
                "Posso aiutarti solo con richieste relative a ETF e codici ISIN. "
                "Scrivi una domanda su un ETF o indica un ISIN."
            )

        isin = (
            intent_data.get("isin")
            or isin
            or self._extract_isin_fallback(latest_user_message)
            or self._extract_isin_from_history(messages)
        )
        if not isin and previous_state_json:
            try:
                previous_state = json.loads(previous_state_json)
                isin = previous_state.get("isin")
            except json.JSONDecodeError:
                pass
        if not isin and memory:
            isin = memory.get("isin")

        memory = store.get(isin) if isin else memory
        mode = (
            "conversation"
            if self._should_use_conversation(
                messages=messages,
                latest_user_message=latest_user_message,
                isin=isin,
                memory=memory,
                isin_in_latest=isin_in_latest,
            )
            else "full_analysis"
        )

        return await self._invoke_graph(
            mode=mode,
            isin=isin,
            clean_query=intent_data.get("clean_query"),
            latest_user_message=latest_user_message,
            messages=messages,
        )
