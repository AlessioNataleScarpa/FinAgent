import json
import logging
from typing import Any, Dict, List, Optional, Union

try:
    from agents.base import AwaitableString, BaseAgent
    from prompts.presentation import build_presentation_agent_prompt
    from schemas.chat import Message
    from schemas.presentation import PresentationOutputSchema
    from utils.flags import pipeline_use_llm
except ImportError:
    from backend.agents.base import AwaitableString, BaseAgent
    from backend.prompts.presentation import build_presentation_agent_prompt
    from backend.schemas.chat import Message
    from backend.schemas.presentation import PresentationOutputSchema
    from backend.utils.flags import pipeline_use_llm

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

    @staticmethod
    def _parse_profile(info_presentazione: str) -> Dict[str, Any]:
        try:
            payload = json.loads(info_presentazione or "{}")
        except json.JSONDecodeError:
            return {"raw": info_presentazione}
        profile = payload.get("Profile") if isinstance(payload, dict) else {}
        if not isinstance(profile, dict):
            profile = {}
        return {
            "isin": payload.get("ISIN") if isinstance(payload, dict) else None,
            "ticker": payload.get("Ticker") if isinstance(payload, dict) else None,
            "profile": profile,
            "raw": info_presentazione,
        }

    def _build_fallback_markdown(self, isin: str, info_presentazione: str) -> str:
        parsed = self._parse_profile(info_presentazione)
        profile = parsed.get("profile") or {}
        ticker = parsed.get("ticker") or "N/D"
        name = profile.get("name") or "ETF"
        description = profile.get("description") or "Descrizione non disponibile."
        if isinstance(description, str) and len(description) > 700:
            description = description[:700].rstrip() + "..."

        lines = [
            f"# Presentazione strumento",
            "",
            f"**ISIN:** `{isin}`  ",
            f"**Ticker:** `{ticker}`  ",
            f"**Nome:** {name}",
            "",
            "### Dettagli fondamentali",
            f"- Categoria: {profile.get('category', 'N/A')}",
            f"- Exchange: {profile.get('exchange', 'N/A')}",
            f"- Valuta: {profile.get('currency', 'N/A')}",
            f"- Previous close: {profile.get('previousClose', 'N/A')}",
            f"- AUM / Market cap: {profile.get('totalAssets', profile.get('marketCap', 'N/A'))}",
            f"- Yield: {profile.get('yield', 'N/A')}",
            "",
            "### Descrizione",
            description,
            "",
            "_I grafici di composizione sono generati dal nodo `composition_charts`._",
        ]
        return AwaitableString("\n".join(lines))

    async def run_presentation(self, info_presentazione: str) -> Union[PresentationOutputSchema, str]:
        if not pipeline_use_llm():
            return self._build_fallback_markdown("", info_presentazione)

        system_prompt = (
            "Sei un assistente AI specializzato nella presentazione di asset finanziari ed ETF.\n"
            "Analizza le informazioni fornite e genera una presentazione strutturata e dettagliata dell'ETF."
        )

        try:
            structured_llm = self.create_structured_llm(
                PresentationOutputSchema,
                system_prompt=system_prompt,
                fallback_to_plain=True,
            )
            res = await structured_llm.ainvoke(f"Analizza e presenta il seguente ETF:\n{info_presentazione}")
            parsed = self.parse_structured_output(res, PresentationOutputSchema)
            if parsed is not None:
                return parsed
            return res.content if hasattr(res, "content") else str(res)
        except Exception as e:
            logger.warning("run_presentation LLM failed: %s", e)
            return self._build_fallback_markdown("", info_presentazione)

    def run(self, isin: str = "", info_presentazione: str = "", messages: Optional[List[Message]] = None) -> str:
        if not isin and messages:
            isin = self.extract_latest_user_message(messages)

        if not pipeline_use_llm():
            return self._build_fallback_markdown(isin, info_presentazione)

        try:
            prompt_content = build_presentation_agent_prompt(isin, info_presentazione)
            structured_llm = self.create_structured_llm(
                PresentationOutputSchema,
                system_prompt=prompt_content,
            )
            response = structured_llm.invoke(f"Genera la presentazione per ISIN: {isin}")

            data = self.parse_structured_output(response, PresentationOutputSchema)
            if data is None:
                raise ValueError("Unexpected response type from structured LLM")

            sectors = ", ".join(data.sector_breakdown) if isinstance(data.sector_breakdown, list) else data.sector_breakdown
            regions = ", ".join(data.regional_breakdown) if isinstance(data.regional_breakdown, list) else data.regional_breakdown
            chart = self._format_mermaid(data.mermaid_chart, f"pie title Allocazione ISIN {isin}")
            return AwaitableString(
                f"# Presentazione strumento\n\n"
                f"**ISIN Analizzato:** `{isin}`\n\n"
                f"### Dettagli Fondamentali\n{data.summary}\n\n"
                f"### Allocazione Asset\n{data.asset_allocation}\n\n"
                f"### Ripartizione Settoriale\n{sectors}\n\n"
                f"### Ripartizione Geografica\n{regions}\n\n"
                f"### Struttura e Allocazione\n\n"
                f"```mermaid\n{chart}\n```\n"
            )
        except Exception as e:
            logger.warning("PresentationAgent LLM failed: %s. Using deterministic markdown.", e)
            return self._build_fallback_markdown(isin, info_presentazione)
