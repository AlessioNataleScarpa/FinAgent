"""
Node: SENTIMENT CHARTS
Builds Mermaid charts for news/sentiment overview.
"""

import logging
from typing import Any, Dict

try:
    from agents.sentimentChartAgent import SentimentChartAgent
except ImportError:
    from backend.agents.sentimentChartAgent import SentimentChartAgent

from .state import PipelineState

logger = logging.getLogger(__name__)


def sentiment_charts_node(state: PipelineState) -> Dict[str, Any]:
    isin = state.get("isin", "N/A")
    logger.info("Building sentiment Mermaid charts for ISIN: %s", isin)
    markdown = SentimentChartAgent().build_markdown(
        isin=isin,
        news_data=state.get("news_data", "") or "",
        prediction=state.get("prediction_out2", "") or "",
        technical=state.get("agent_2_out_tech", "") or "",
    )
    return {"sentiment_charts": markdown}
