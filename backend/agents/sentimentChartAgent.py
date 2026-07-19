"""Agent specialized in Mermaid charts for news/sentiment overview."""

from __future__ import annotations

import logging
import re
from typing import Dict, List

try:
    from agents.base import BaseAgent
    from schemas.chat import Message
    from utils.mermaid import build_pie_chart, wrap_mermaid
except ImportError:
    from backend.agents.base import BaseAgent
    from backend.schemas.chat import Message
    from backend.utils.mermaid import build_pie_chart, wrap_mermaid

logger = logging.getLogger(__name__)


class SentimentChartAgent(BaseAgent):
    @property
    def model_id(self) -> str:
        return "Sentiment Chart Agent"

    @staticmethod
    def _score_text(text: str) -> Dict[str, float]:
        lowered = (text or "").lower()
        positive_hits = len(
            re.findall(
                r"bullish|positivo|growth|rialz|gain|inflow|beat|exceed|favorevole",
                lowered,
            )
        )
        negative_hits = len(
            re.findall(
                r"bearish|negativo|rischi|volatil|friction|drawdown|sell|outflow|geopolitic",
                lowered,
            )
        )
        neutral = max(1, 3 - abs(positive_hits - negative_hits))
        return {
            "Positivo": float(max(1, positive_hits + 2)),
            "Neutro": float(neutral + 1),
            "Negativo": float(max(1, negative_hits + 1)),
        }

    def build_markdown(
        self,
        isin: str,
        news_data: str = "",
        prediction: str = "",
        technical: str = "",
    ) -> str:
        scores = self._score_text("\n".join([news_data, prediction, technical]))
        pie = build_pie_chart(f"Sentiment di mercato {isin}", scores)

        impact_graph = (
            "graph LR\n"
            f"    N[News e macro] -->|Impatto| S[Sentiment]\n"
            f"    P[Previsione quantitativa] --> S\n"
            f"    S --> O[Outlook ETF {isin}]\n"
        )

        return (
            f"## Sentiment e impatto news\n\n"
            f"Sintesi visuale dell'agente **SentimentChartAgent**.\n\n"
            f"### Distribuzione sentiment\n\n"
            f"{wrap_mermaid(pie)}\n\n"
            f"### Catena di impatto\n\n"
            f"{wrap_mermaid(impact_graph)}\n"
        )

    async def run(self, messages: List[Message]) -> str:
        latest = self.extract_latest_user_message(messages)
        return self.build_markdown(isin=latest or "N/D")
