"""Agent specialized in Mermaid charts for news/sentiment overview."""

from __future__ import annotations

import logging
import re
from typing import Dict, List

try:
    from agents.base import BaseAgent
    from models.sentiment import analyze_sentiment
    from schemas.chat import Message
    from utils.mermaid import build_pie_chart, wrap_mermaid
except ImportError:
    from backend.agents.base import BaseAgent
    from backend.models.sentiment import analyze_sentiment
    from backend.schemas.chat import Message
    from backend.utils.mermaid import build_pie_chart, wrap_mermaid

logger = logging.getLogger(__name__)


class SentimentChartAgent(BaseAgent):
    @property
    def model_id(self) -> str:
        return "Sentiment Chart Agent"

    def build_markdown(
        self,
        isin: str,
        news_data: str = "",
        prediction: str = "",
        technical: str = "",
    ) -> str:
        sentiment_input = news_data if news_data.strip() else "\n".join([prediction, technical])
        result = analyze_sentiment(sentiment_input)

        slices = {
            "Positivo": max(0.1, round(result.positive_score * 100.0, 1)),
            "Neutro": max(0.1, round(result.neutral_score * 100.0, 1)),
            "Negativo": max(0.1, round(result.negative_score * 100.0, 1)),
        }
        pie = build_pie_chart(f"Distribuzione Sentiment {isin}", slices)

        impact_graph = (
            "graph LR\n"
            f"    N[Rassegna News e Macro] -->|Ponderazione Neurale| S[Sentiment FinBERT]\n"
            f"    P[Previsione Quantitativa Chronos] --> S\n"
            f"    S --> O[Outlook Complessivo {isin}]\n"
        )

        headlines_table = ""
        if result.headline_breakdown:
            rows = []
            for h in result.headline_breakdown[:6]:
                title_link = f"[{h.title[:65]}...]({h.url})" if h.url else f"{h.title[:70]}"
                rows.append(
                    f"| {title_link} | **{h.label}** | `{h.polarity:+.2f}` | `{h.confidence * 100:.1f}%` |"
                )
            headlines_table = (
                "\n#### Analisi Granulare dei Titoli (FinBERT)\n\n"
                "| Notizia | Polarità | Indice ($S$) | Confidenza |\n"
                "|---|---|---:|---:|\n"
                + "\n".join(rows)
                + "\n"
            )

        method_note = (
            "Modello Neurale Transformer: `ProsusAI/finbert` (Softmax inference)"
            if result.method == "finbert_neural"
            else "Modello di riserva analitico-lessicale calibrato"
        )

        return (
            f"## Sentiment e Analisi Notizie\n\n"
            f"{result.summary_text}\n\n"
            f"- **Metodologia:** {method_note}\n"
            f"- **Indice di Polarità Composito ($S$):** `{result.polarity_index:+.4f}` (intervallo $[-1.0, +1.0]$)\n\n"
            f"### Distribuzione Probabilistica del Sentiment\n\n"
            f"{wrap_mermaid(pie)}\n\n"
            f"{headlines_table}"
            f"### Catena di Trasmissione dell'Impatto\n\n"
            f"{wrap_mermaid(impact_graph)}\n"
        )

    async def run(self, messages: List[Message]) -> str:
        latest = self.extract_latest_user_message(messages)
        return self.build_markdown(isin=latest or "N/D")
