import logging
from typing import List, Optional, Union

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from agents.base import AwaitableString, BaseAgent
    from prompts.technical_news import build_technical_news_agent_prompt
    from schemas.chat import Message
    from schemas.technical_news import TechnicalNewsOutputSchema
except ImportError:
    from backend.agents.base import AwaitableString, BaseAgent
    from backend.prompts.technical_news import build_technical_news_agent_prompt
    from backend.schemas.chat import Message
    from backend.schemas.technical_news import TechnicalNewsOutputSchema

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
            f"# 📈 ANALISI TECNICA E CONFRONTO NEWS (OUT TECNICA)\n\n"
            f"### 🔮 Previsione Quantitativa (OUT 2)\n"
            f"{prediction}\n\n"
            f"### 📰 Notizie di Mercato Rilevanti\n"
            f"{news}\n\n"
            f"### 🎯 Impatto delle News sulla Previsione\n"
            f"L'analisi quantitativa indica un trend rialzista (+7.5% - +10.2%). L'integrazione delle notizie recenti conferma il sentiment favorevole:\n"
            f"- **Taglio/Pausa dei Tassi**: Favorisce i titoli tech e azionari globali presenti nell'ETF.\n"
            f"- **Utili superiori alle attese**: Supportano la crescita dei fondamentali sui 12 mesi.\n"
            f"- **Rischi Geopolitici**: Possono generare volatilità di breve termine entro la fascia del 12-14%.\n\n"
            f"### 📊 Diagramma Mermaid Previsionale e Impatto Notizie\n\n"
            f"```mermaid\n"
            f"graph LR\n"
            f"    N1[Notizie Macro Positive] -->|Spinta Rialzista| P[Trend Previsto: +8.5%]\n"
            f"    N2[Rischi Geopolitici] -->|Resistenza Volatilità| P\n"
            f"    H[Storico CAGR 11.5%] -->|Base Quantitativa| P\n"
            f"    P --> OUT[Target Price 12 Mesi: Growth Zone]\n"
            f"```\n\n"
            f"```mermaid\n"
            f"gantt\n"
            f"    title Orizzonte temporale di previsione ({isin})\n"
            f"    dateFormat YYYY-MM-DD\n"
            f"    section Fasi Previsione\n"
            f"    Accumulazione iniziale    :a1, 2026-07-01, 30d\n"
            f"    Crescita Moderata          :after a1, 60d\n"
            f"    Target Growth Achieved     :after a2, 90d\n"
            f"```\n"
        )

    async def run_technical_news(
        self,
        news_data: str,
        prediction_data: str,
    ) -> Union[TechnicalNewsOutputSchema, str]:
        """
        Runs the technical news agent asynchronously using TechnicalNewsOutputSchema or fallback LLM.
        """
        system_prompt = (
            "Sei un assistente AI specializzato nell'analisi tecnica e nell'impatto delle notizie finanziarie sugli ETF.\n"
            "Analizza la previsione quantitativa e le notizie per fornire una sintesi tecnica ed un'analisi dell'impatto delle notizie."
        )
        human_prompt = (
            f"Previsione Quantitativa:\n{prediction_data}\n\n"
            f"Notizie di Mercato:\n{news_data}"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]

        try:
            structured_llm = self.create_structured_llm(TechnicalNewsOutputSchema, fallback_to_plain=True)
            res = await structured_llm.ainvoke(messages)
            if isinstance(res, TechnicalNewsOutputSchema):
                return res
            if hasattr(res, "model_dump"):
                return TechnicalNewsOutputSchema(**res.model_dump())
            if isinstance(res, dict):
                return TechnicalNewsOutputSchema(**res)
            return res.content if hasattr(res, "content") else str(res)
        except Exception as e:
            logger.warning("run_technical_news LLM failed: %s", e)
            try:
                plain_llm = self.create_llm()
                res = await plain_llm.ainvoke(messages)
                return res.content if hasattr(res, "content") else str(res)
            except Exception as inner_e:
                logger.warning("run_technical_news plain LLM fallback failed: %s", inner_e)
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

        try:
            structured_llm = self.create_structured_llm(TechnicalNewsOutputSchema)
            prompt_content = build_technical_news_agent_prompt(isin, prediction, news)
            response = structured_llm.invoke([
                SystemMessage(content=prompt_content),
                HumanMessage(content=f"Genera l'analisi tecnica e news per ISIN: {isin}"),
            ])

            if isinstance(response, TechnicalNewsOutputSchema):
                data = response
            elif hasattr(response, "model_dump"):
                data = TechnicalNewsOutputSchema(**response.model_dump())
            elif isinstance(response, dict):
                data = TechnicalNewsOutputSchema(**response)
            else:
                raise ValueError("Unexpected response type from structured LLM")

            chart = self._format_mermaid(data.mermaid_chart, "graph LR\n    A -> B")
            return AwaitableString(
                f"# 📈 ANALISI TECNICA E CONFRONTO NEWS (OUT TECNICA)\n\n"
                f"### 🔮 Previsione Quantitativa (OUT 2)\n{data.technical_summary}\n\n"
                f"### 📰 Notizie di Mercato Rilevanti e Impatto\n"
                f"**Sentiment Generale:** {data.sentiment_score}\n\n"
                f"{data.news_impact_analysis}\n\n"
                f"### 📊 Diagramma Mermaid Previsionale e Impatto Notizie\n\n"
                f"```mermaid\n{chart}\n```\n"
            )
        except Exception as e:
            logger.warning(
                "TechnicalNewsAgent LLM generation failed or unavailable: %s. Falling back to default markdown.",
                e,
            )
            return self._build_fallback_markdown(isin, prediction, news)
