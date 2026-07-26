import json
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

    @staticmethod
    def _format_news(news: str) -> str:
        try:
            payload = json.loads(news or "")
        except (TypeError, json.JSONDecodeError):
            return news or "Nessuna notizia verificabile disponibile."
        if not isinstance(payload, dict):
            return news
        articles = payload.get("articles") or []
        if not isinstance(articles, list) or not articles:
            return "Nessuna notizia verificabile disponibile."

        lines = []
        for article in articles[:8]:
            if not isinstance(article, dict):
                continue
            title = str(article.get("title") or "Senza titolo").strip()
            summary = str(article.get("summary") or "").strip()
            publisher = str(article.get("publisher") or "").strip()
            url = str(article.get("url") or "").strip()
            label = f"[{title}]({url})" if url else f"**{title}**"
            detail = " — ".join(part for part in (publisher, summary) if part)
            lines.append(f"- {label}" + (f": {detail}" if detail else ""))
        return "\n".join(lines) or "Nessuna notizia verificabile disponibile."

    def _build_fallback_markdown(self, isin: str, prediction: str, news: str) -> str:
        news_markdown = self._format_news(news)
        lowered_prediction = (prediction or "").lower()
        if "stato: non disponibile" in lowered_prediction or "previsione non disponibile" in lowered_prediction:
            uncertainty = (
                "- **Affidabilità quantitativa: non disponibile.** Le news non "
                "devono essere usate come sostituto di una previsione mancante.\n"
            )
        elif "stima statistica di riserva" in lowered_prediction:
            uncertainty = (
                "- **Affidabilità quantitativa: ridotta.** È attivo il fallback "
                "statistico; il segnale resta esplorativo e conservativo.\n"
                "- Le news possono spiegare uno scenario, ma non trasformarlo in "
                "un'indicazione direzionale ad alta confidenza.\n"
            )
        else:
            uncertainty = (
                "- La forza del segnale va calibrata sull'ampiezza dell'intervallo "
                "di previsione: range ampio significa bassa confidenza.\n"
            )

        return AwaitableString(
            f"# Analisi tecnica e confronto news\n\n"
            f"**ISIN:** `{isin}`\n\n"
            f"### Previsione quantitativa\n"
            f"{prediction}\n\n"
            f"### Notizie di mercato\n"
            f"{news_markdown}\n\n"
            f"### Impatto delle news sulla previsione\n"
            f"{uncertainty}"
            f"- Le news macro e gli utili influenzano il bias di breve/medio termine.\n"
            f"- I rischi geopolitici restano un fattore di volatilità.\n"
            f"- Fatti riportati, inferenze e stime del modello vanno tenuti distinti.\n\n"
            f"_Analisi informativa, non consulenza finanziaria né rendimento garantito._\n"
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
            "Analizza la previsione quantitativa e le notizie per fornire una sintesi tecnica ed un'analisi dell'impatto delle notizie. "
            "Se il forecast usa un fallback o ha un intervallo ampio, dichiara l'incertezza e adotta un bias conservativo. "
            "Distingui sempre fatti, inferenze e output del modello; non formulare rendimenti garantiti."
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
