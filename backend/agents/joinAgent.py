import logging
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from agents.base import BaseAgent
    from schemas.chat import Message
except ImportError:
    from backend.agents.base import BaseAgent
    from backend.schemas.chat import Message

logger = logging.getLogger(__name__)


class JoinAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    @property
    def model_id(self) -> str:
        return "Join Agent"

    def _build_messages(self, out1: str, out_tech: str) -> List:
        system_prompt = (
            "Sei l'Agente Join / Sintesi, un assistente AI specializzato nella combinazione e sintesi di analisi finanziarie.\n"
            "Il tuo compito è sintetizzare le informazioni della presentazione dello strumento (OUT 1) e l'analisi tecnica e notizie (OUT TECNICA) "
            "in un unico REPORT FINALE completo, coerente e ben formattato in Markdown."
        )
        human_prompt = (
            f"Sintetizza i seguenti due moduli nel REPORT FINALE:\n\n"
            f"=== OUT 1 (PRESENTAZIONE STRUMENTO) ===\n{out1}\n\n"
            f"=== OUT TECNICA (ANALISI TECNICA E NEWS) ===\n{out_tech}\n"
        )
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

    def _fallback_synthesis(self, out1: str, out_tech: str) -> str:
        return (
            f"# 🏆 REPORT COMPLETO DI ANALISI STRUMENTO FINANZIARIO\n\n"
            f"{out1}\n\n"
            f"---\n\n"
            f"{out_tech}\n\n"
            f"---\n"
            f"*Report generato automaticamente dalla Pipeline LangGraph.*"
        )

    async def run_join(self, out1: str, out_tech: str) -> str:
        """
        Synthesizes out1 (Presentation) and out_tech (Technical & News) into OUT FINALE using LLM.
        """
        messages = self._build_messages(out1, out_tech)

        try:
            llm = self.create_llm()
            res = await llm.ainvoke(messages)
            content = res.content if hasattr(res, "content") else str(res)
            if not isinstance(content, str):
                content = str(content)
            return content.strip()
        except Exception as e:
            logger.warning("JoinAgent LLM generation failed: %s. Falling back to concatenation.", e)
            return self._fallback_synthesis(out1, out_tech)

    def run_sync(self, out1: str, out_tech: str) -> str:
        """
        Synchronous version of run_join for sync LangGraph nodes.
        """
        messages = self._build_messages(out1, out_tech)

        try:
            llm = self.create_llm()
            res = llm.invoke(messages)
            content = res.content if hasattr(res, "content") else str(res)
            if not isinstance(content, str):
                content = str(content)
            return content.strip()
        except Exception as e:
            logger.warning("JoinAgent LLM sync generation failed: %s. Falling back to concatenation.", e)
            return self._fallback_synthesis(out1, out_tech)

    async def run(self, messages: List[Message]) -> str:
        latest_user_message = self.extract_latest_user_message(messages)
        return await self.run_join(latest_user_message, "")

