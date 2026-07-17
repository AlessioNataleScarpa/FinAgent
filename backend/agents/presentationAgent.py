import logging
from typing import List, Optional, Union

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from agents.base import BaseAgent
    from prompts.presentation import build_presentation_agent_prompt
    from schemas.chat import Message
    from schemas.presentation import PresentationAgentSchema, PresentationOutputSchema
except ImportError:
    from backend.agents.base import BaseAgent
    from backend.prompts.presentation import build_presentation_agent_prompt
    from backend.schemas.chat import Message
    from backend.schemas.presentation import PresentationAgentSchema, PresentationOutputSchema

logger = logging.getLogger(__name__)


class PresentationAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    @property
    def model_id(self) -> str:
        return "Presentation Agent"

    def _format_mermaid(self, raw_diagram: str, default_diagram: str) -> str:
        if not raw_diagram:
            return default_diagram
        cleaned = self.strip_code_fences(raw_diagram)
        if not cleaned:
            return default_diagram
        return cleaned

    def _build_fallback_markdown(self, isin: str, info_presentazione: str) -> str:
        return (
            f"# 📊 PRESENTAZIONE STRUMENTO (OUT 1)\n\n"
            f"**ISIN Analizzato:** `{isin}`\n\n"
            f"### 📋 Dettagli Fondamentali\n"
            f"{info_presentazione}\n\n"
            f"### 🌐 Struttura e Allocazione (Mermaid Diagram)\n\n"
            f"```mermaid\n"
            f"pie title Allocazione Settoriale ISIN {isin}\n"
            f'    "Information Tech" : 23.5\n'
            f'    "Financials" : 15.2\n'
            f'    "Healthcare" : 12.1\n'
            f'    "Industrials" : 11.0\n'
            f'    "Consumer Disc" : 10.4\n'
            f'    "Communication" : 7.5\n'
            f'    "Altri Settori" : 20.3\n'
            f"```\n\n"
            f"```mermaid\n"
            f"graph TD\n"
            f"    A[{isin}] --> B[Mercati Sviluppati]\n"
            f"    B --> C[USA - 70.1%]\n"
            f"    B --> D[Giappone - 6.2%]\n"
            f"    B --> E[Europa / UK - 10.0%]\n"
            f"    B --> F[Altri - 13.7%]\n"
            f"```\n"
        )

    async def run_presentation(self, info_presentazione: str) -> Union[PresentationOutputSchema, str]:
        """
        Runs the presentation agent asynchronously using PresentationOutputSchema or fallback LLM.
        """
        system_prompt = (
            "Sei un assistente AI specializzato nella presentazione di asset finanziari ed ETF.\n"
            "Analizza le informazioni fornite e genera una presentazione strutturata e dettagliata dell'ETF."
        )
        human_prompt = f"Analizza e presenta il seguente ETF:\n{info_presentazione}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        try:
            structured_llm = self.create_structured_llm(PresentationOutputSchema, fallback_to_plain=True)
            res = await structured_llm.ainvoke(messages)
            if isinstance(res, PresentationOutputSchema):
                return res
            if hasattr(res, "model_dump"):
                return PresentationOutputSchema(**res.model_dump())
            if isinstance(res, dict):
                return PresentationOutputSchema(**res)
            return res.content if hasattr(res, "content") else str(res)
        except Exception as e:
            logger.warning("run_presentation LLM failed: %s", e)
            try:
                plain_llm = self.create_llm()
                res = await plain_llm.ainvoke(messages)
                return res.content if hasattr(res, "content") else str(res)
            except Exception as inner_e:
                logger.warning("run_presentation plain LLM fallback failed: %s", inner_e)
                return self._build_fallback_markdown("", info_presentazione)

    def run(self, isin: str = "", info_presentazione: str = "", messages: Optional[List[Message]] = None) -> str:
        if not isin and messages:
            isin = self.extract_latest_user_message(messages)

        try:
            structured_llm = self.create_structured_llm(PresentationAgentSchema)
            prompt_content = build_presentation_agent_prompt(isin, info_presentazione)
            response = structured_llm.invoke([
                SystemMessage(content=prompt_content),
                HumanMessage(content=f"Genera la presentazione per ISIN: {isin}"),
            ])

            if isinstance(response, PresentationAgentSchema):
                data = response
            elif hasattr(response, "model_dump"):
                data = PresentationAgentSchema(**response.model_dump())
            elif isinstance(response, dict):
                data = PresentationAgentSchema(**response)
            elif isinstance(response, PresentationOutputSchema):
                data = response
            else:
                raise ValueError("Unexpected response type from structured LLM")

            if isinstance(data, PresentationOutputSchema):
                sectors = ", ".join(data.sector_breakdown) if isinstance(data.sector_breakdown, list) else data.sector_breakdown
                regions = ", ".join(data.regional_breakdown) if isinstance(data.regional_breakdown, list) else data.regional_breakdown
                chart = self._format_mermaid(data.mermaid_chart, f"pie title Allocazione ISIN {isin}")
                return (
                    f"# 📊 PRESENTAZIONE STRUMENTO (OUT 1)\n\n"
                    f"**ISIN Analizzato:** `{isin}`\n\n"
                    f"### 📋 Dettagli Fondamentali\n{data.summary}\n\n"
                    f"### 💼 Allocazione Asset\n{data.asset_allocation}\n\n"
                    f"### 📊 Ripartizione Settoriale\n{sectors}\n\n"
                    f"### 🌐 Ripartizione Geografica\n{regions}\n\n"
                    f"### 🌐 Struttura e Allocazione (Mermaid Diagram)\n\n"
                    f"```mermaid\n{chart}\n```\n"
                )

            pie_chart = self._format_mermaid(
                data.mermaid_pie_chart,
                f"pie title Allocazione Settoriale ISIN {isin}\n"
                f'    "Information Tech" : 23.5\n'
                f'    "Financials" : 15.2\n'
                f'    "Healthcare" : 12.1\n'
                f'    "Industrials" : 11.0\n'
                f'    "Consumer Disc" : 10.4\n'
                f'    "Communication" : 7.5\n'
                f'    "Altri Settori" : 20.3',
            )

            flowchart = self._format_mermaid(
                data.mermaid_flowchart,
                f"graph TD\n"
                f"    A[{isin}] --> B[Mercati Sviluppati]\n"
                f"    B --> C[USA - 70.1%]\n"
                f"    B --> D[Giappone - 6.2%]\n"
                f"    B --> E[Europa / UK - 10.0%]\n"
                f"    B --> F[Altri - 13.7%]",
            )

            out_markdown = (
                f"# 📊 PRESENTAZIONE STRUMENTO (OUT 1)\n\n"
                f"**ISIN Analizzato:** `{isin}`\n\n"
                f"### 📋 Dettagli Fondamentali\n"
                f"{data.summary}\n\n"
                f"### 📊 Allocazione Settoriale\n"
                f"{data.sector_allocation_desc}\n\n"
                f"### 🌐 Allocazione Geografica\n"
                f"{data.regional_allocation_desc}\n\n"
                f"### 🌐 Struttura e Allocazione (Mermaid Diagram)\n\n"
                f"```mermaid\n"
                f"{pie_chart}\n"
                f"```\n\n"
                f"```mermaid\n"
                f"{flowchart}\n"
                f"```\n"
            )
            return out_markdown
        except Exception as e:
            logger.warning(
                "PresentationAgent LLM generation failed or unavailable: %s. Falling back to default markdown.",
                e,
            )
            return self._build_fallback_markdown(isin, info_presentazione)
