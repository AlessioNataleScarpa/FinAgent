import json
import re
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from agents.base import BaseAgent
from prompts import build_gateway_system_prompt
from schemas.chat import Message
from schemas.routing import RouterIntentSchema

try:
    from pipeline.graph import graph
except ImportError:
    from backend.pipeline.graph import graph

ISIN_PATTERN = re.compile(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b", re.IGNORECASE)


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

    def _build_system_prompt(self, previous_state_json: Optional[str]) -> str:
        return build_gateway_system_prompt(previous_state_json)

    async def _build_accepted_stub(
        self,
        clean_query: Optional[str],
        isin: Optional[str],
        latest_user_message: str,
    ) -> str:
        pipeline_input = {
            "isin": isin or "",
            "clean_query": clean_query or latest_user_message,
        }
        state = await graph.ainvoke(pipeline_input)
        if isinstance(state, dict) and state.get("out_finale"):
            return state["out_finale"]

        payload = {
            "status": "accepted",
            "intent": "etf",
            "isin": isin,
            "clean_query": clean_query or latest_user_message,
            "message": "Richiesta ETF accettata. Pipeline di analisi non ha restituito output.",
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    async def run(self, messages: List[Message]) -> str:
        previous_state_json = self.extract_previous_assistant_state(
            messages,
            self._extract_router_state,
        )
        latest_user_message = self.extract_latest_user_message(messages)

        system_prompt = self._build_system_prompt(previous_state_json)
        langchain_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=latest_user_message),
        ]

        response = await self.structured_llm.ainvoke(langchain_messages)

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

        isin = intent_data.get("isin") or self._extract_isin_fallback(latest_user_message)
        if not isin and previous_state_json:
            try:
                previous_state = json.loads(previous_state_json)
                isin = previous_state.get("isin")
            except json.JSONDecodeError:
                pass

        return await self._build_accepted_stub(
            clean_query=intent_data.get("clean_query"),
            isin=isin,
            latest_user_message=latest_user_message,
        )
