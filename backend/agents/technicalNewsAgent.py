import logging
from typing import List, Optional, Union

try:
    from agents.base import AwaitableString, BaseAgent
    from prompts.technical_news import build_technical_news_agent_prompt
    from schemas.chat import Message
    from schemas.technical_news import TechnicalNewsOutputSchema
    from utils.flags import pipeline_use_llm
except ImportError:
    from backend.agents.base import AwaitableString, BaseAgent
    from backend.prompts.technical_news import build_technical_news_agent_prompt
    from backend.schemas.chat import Message
    from backend.schemas.technical_news import TechnicalNewsOutputSchema
    from backend.utils.flags import pipeline_use_llm

logger = logging.getLogger(__name__)


class TechnicalNewsAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    @property
    def model_id(self) -> str:
        return "Technical News Agent"

    def _format_mermaid(self, raw_diagram: str, default_diagram: str) -> str:
        if not raw_diagram:
            return default_diagram
        cleaned = self.strip_code_fences(raw_diagram)
        if not cleaned:
            return default_diagram
        return cleaned

    def _build_fallback_markdown(self, isin: str, prediction: str, news: str) -> str:
        return AwaitableString(
            f"# Analisi tecnica e confronto news\n\n"
            f"**ISIN:** `{isin}`\n\n"
            f"### Previsione quantitativa\n"
            f"{prediction}\n\n"
            f"### Notizie di mercato\n"
            f"{news}\n\n"
            f"### Impatto delle news sulla previsione\n"
            f"- Le news macro e gli utili influenzano il bias di breve/medio termine.\n"
            f"- I rischi geopolitici restano un fattore di volatilità.\n"
            f"- La previsione quantitativa va interpretata insieme al sentiment.\n\n"
            f"_I grafici sentiment/impatto sono generati dal nodo `sentiment_charts`._\n"
        )

    async def run_technical_news(
        self,
        news_data: str,
        prediction_data: str,
    ) -> Union[TechnicalNewsOutputSchema, str]:
        if not pipeline_use_llm():
            return self._build_fallback_markdown("", prediction_data, news_data)

        system_prompt = (
            "Sei un assistente AI specializzato nell'analisi tecnica e nell'impatto delle notizie finanziarie sugli ETF.\n"
            "Analizza la previsione quantitativa e le notizie per fornire una sintesi tecnica ed un'analisi dell'impatto delle notizie."
        )
        user_prompt = (
            f"Previsione Quantitativa:\n{prediction_data}\n\n"
            f"Notizie di Mercato:\n{news_data}"
        )

        try:
            structured_llm = self.create_structured_llm(
                TechnicalNewsOutputSchema,
                system_prompt=system_prompt,
                fallback_to_plain=True,
            )
            res = await structured_llm.ainvoke(user_prompt)
            parsed = self.parse_structured_output(res, TechnicalNewsOutputSchema)
            if parsed is not None:
                return parsed
            return res.content if hasattr(res, "content") else str(res)
        except Exception as e:
            logger.warning("run_technical_news LLM failed: %s", e)
            return self._build_fallback_markdown("", prediction_data, news_data)

    def run(
        self,
        isin: str = "",
        prediction: str = "",
        news: str = "",
        messages: Optional[List[Message]] = None,
    ) -> str:
        if not isin and messages:
            isin = self.extract_latest_user_message(messages)

        if not pipeline_use_llm():
            return self._build_fallback_markdown(isin, prediction, news)

        try:
            prompt_content = build_technical_news_agent_prompt(isin, prediction, news)
            structured_llm = self.create_structured_llm(
                TechnicalNewsOutputSchema,
                system_prompt=prompt_content,
            )
            response = structured_llm.invoke(f"Genera l'analisi tecnica e news per ISIN: {isin}")

            data = self.parse_structured_output(response, TechnicalNewsOutputSchema)
            if data is None:
                raise ValueError("Unexpected response type from structured LLM")

            chart = self._format_mermaid(data.mermaid_chart, "graph LR\n    A --> B")
            return AwaitableString(
                f"# Analisi tecnica e confronto news\n\n"
                f"### Previsione quantitativa\n{data.technical_summary}\n\n"
                f"### Notizie e impatto\n"
                f"**Sentiment:** {data.sentiment_score}\n\n"
                f"{data.news_impact_analysis}\n\n"
                f"### Diagramma impatto\n\n"
                f"```mermaid\n{chart}\n```\n"
            )
        except Exception as e:
            logger.warning("TechnicalNewsAgent LLM failed: %s. Using deterministic markdown.", e)
            return self._build_fallback_markdown(isin, prediction, news)
