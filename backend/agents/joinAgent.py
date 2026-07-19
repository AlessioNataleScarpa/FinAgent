import logging
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from agents.base import BaseAgent
    from prompts.join import build_join_human_prompt, build_join_system_prompt
    from schemas.chat import Message
    from utils.flags import pipeline_use_llm
except ImportError:
    from backend.agents.base import BaseAgent
    from backend.prompts.join import build_join_human_prompt, build_join_system_prompt
    from backend.schemas.chat import Message
    from backend.utils.flags import pipeline_use_llm

logger = logging.getLogger(__name__)


class JoinAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    @property
    def model_id(self) -> str:
        return "Join Agent"

    def _build_messages(
        self,
        out1: str,
        out_tech: str,
        *,
        isin: str = "",
        composition_charts: str = "",
        timeline_charts: str = "",
        sentiment_charts: str = "",
    ) -> List:
        return [
            SystemMessage(content=build_join_system_prompt()),
            HumanMessage(
                content=build_join_human_prompt(
                    isin=isin,
                    out1=out1,
                    out_tech=out_tech,
                    composition_charts=composition_charts,
                    timeline_charts=timeline_charts,
                    sentiment_charts=sentiment_charts,
                )
            ),
        ]

    def _fallback_synthesis(
        self,
        out1: str,
        out_tech: str,
        *,
        isin: str = "",
        composition_charts: str = "",
        timeline_charts: str = "",
        sentiment_charts: str = "",
    ) -> str:
        isin_label = isin or "N/D"
        return (
            f"# Report di analisi ETF\n\n"
            f"**ISIN:** `{isin_label}`\n\n"
            f"## Executive summary\n\n"
            f"Report generato dalla pipeline LangGraph (presentazione, tecnica/news e grafici Mermaid).\n\n"
            f"## Presentazione strumento\n\n"
            f"{out1}\n\n"
            f"{composition_charts}\n\n"
            f"{timeline_charts}\n\n"
            f"## Analisi tecnica e confronto news\n\n"
            f"{out_tech}\n\n"
            f"{sentiment_charts}\n\n"
            f"## Note\n\n"
            f"Output in Markdown (non JSON). I diagrammi Mermaid sono in blocchi "
            f"```mermaid``` per il rendering in OpenWebUI.\n\n"
            f"---\n"
            f"*Analisi salvata in memoria. Puoi continuare a fare domande qui su "
            f"`gatewayAgent`: i follow-up usano automaticamente il nodo `conversation` "
            f"(memoria + eventuale ricerca web).*"
        )

    async def run_join(
        self,
        out1: str,
        out_tech: str,
        *,
        isin: str = "",
        composition_charts: str = "",
        timeline_charts: str = "",
        sentiment_charts: str = "",
    ) -> str:
        if not pipeline_use_llm():
            return self._fallback_synthesis(
                out1,
                out_tech,
                isin=isin,
                composition_charts=composition_charts,
                timeline_charts=timeline_charts,
                sentiment_charts=sentiment_charts,
            )

        messages = self._build_messages(
            out1,
            out_tech,
            isin=isin,
            composition_charts=composition_charts,
            timeline_charts=timeline_charts,
            sentiment_charts=sentiment_charts,
        )

        try:
            llm = self.create_llm()
            res = await llm.ainvoke(messages)
            content = res.content if hasattr(res, "content") else str(res)
            if not isinstance(content, str):
                content = str(content)
            content = self.normalize_llm_content(content).strip()
            if content.startswith("{") or content.startswith("["):
                return self._fallback_synthesis(
                    out1,
                    out_tech,
                    isin=isin,
                    composition_charts=composition_charts,
                    timeline_charts=timeline_charts,
                    sentiment_charts=sentiment_charts,
                )
            return content
        except Exception as e:
            logger.warning("JoinAgent LLM generation failed: %s. Falling back.", e)
            return self._fallback_synthesis(
                out1,
                out_tech,
                isin=isin,
                composition_charts=composition_charts,
                timeline_charts=timeline_charts,
                sentiment_charts=sentiment_charts,
            )

    def run_sync(
        self,
        out1: str,
        out_tech: str,
        *,
        isin: str = "",
        composition_charts: str = "",
        timeline_charts: str = "",
        sentiment_charts: str = "",
    ) -> str:
        if not pipeline_use_llm():
            return self._fallback_synthesis(
                out1,
                out_tech,
                isin=isin,
                composition_charts=composition_charts,
                timeline_charts=timeline_charts,
                sentiment_charts=sentiment_charts,
            )

        messages = self._build_messages(
            out1,
            out_tech,
            isin=isin,
            composition_charts=composition_charts,
            timeline_charts=timeline_charts,
            sentiment_charts=sentiment_charts,
        )

        try:
            llm = self.create_llm()
            res = llm.invoke(messages)
            content = res.content if hasattr(res, "content") else str(res)
            if not isinstance(content, str):
                content = str(content)
            content = self.normalize_llm_content(content).strip()
            if content.startswith("{") or content.startswith("["):
                return self._fallback_synthesis(
                    out1,
                    out_tech,
                    isin=isin,
                    composition_charts=composition_charts,
                    timeline_charts=timeline_charts,
                    sentiment_charts=sentiment_charts,
                )
            footer = (
                "\n\n---\n"
                "*Analisi salvata in memoria. Puoi continuare a fare domande qui su "
                "`gatewayAgent`: i follow-up usano automaticamente il nodo `conversation`.*"
            )
            if "Analisi salvata in memoria" not in content:
                content = content + footer
            return content
        except Exception as e:
            logger.warning("JoinAgent LLM sync generation failed: %s. Falling back.", e)
            return self._fallback_synthesis(
                out1,
                out_tech,
                isin=isin,
                composition_charts=composition_charts,
                timeline_charts=timeline_charts,
                sentiment_charts=sentiment_charts,
            )

    async def run(self, messages: List[Message]) -> str:
        latest_user_message = self.extract_latest_user_message(messages)
        return await self.run_join(latest_user_message, "")
