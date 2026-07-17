import logging
from typing import List, Optional

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

    async def run_join(self, out1: str, out_tech: str) -> str:
        """
        Synthesizes out1 (Presentation) and out_tech (Technical & News) into OUT FINALE using LLM.
        """
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

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        try:
            llm = self.create_llm()
            res = await llm.ainvoke(messages)
            content = res.content if hasattr(res, "content") else str(res)
            if not isinstance(content, str):
                content = str(content)
            return content.strip()
        except Exception as e:
            logger.warning("JoinAgent LLM generation failed: %s. Falling back to concatenation.", e)
            return (
                f"# 🏆 REPORT COMPLETO DI ANALISI STRUMENTO FINANZIARIO\n\n"
                f"{out1}\n\n"
                f"---\n\n"
                f"{out_tech}\n\n"
                f"---\n"
                f"*Report generato automaticamente dalla Pipeline LangGraph.*"
            )

    def run_sync(self, out1: str, out_tech: str) -> str:
        """
        Synchronous version of run_join for sync LangGraph nodes.
        """
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

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        try:
            llm = self.create_llm()
            res = llm.invoke(messages)
            content = res.content if hasattr(res, "content") else str(res)
            if not isinstance(content, str):
                content = str(content)
            return content.strip()
        except Exception as e:
            logger.warning("JoinAgent LLM sync generation failed: %s. Falling back to concatenation.", e)
            return (
                f"# 🏆 REPORT COMPLETO DI ANALISI STRUMENTO FINANZIARIO\n\n"
                f"{out1}\n\n"
                f"---\n\n"
                f"{out_tech}\n\n"
                f"---\n"
                f"*Report generato automaticamente dalla Pipeline LangGraph.*"
            )

    async def run(self, messages: List[Message]) -> str:
        latest_user_message = self.extract_latest_user_message(messages)
        return await self.run_join(latest_user_message, "")

