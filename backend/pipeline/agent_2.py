"""
Node: AGENT 2
Takes OUT 2 and NEWS, generates Mermaid chart for prediction and explains how news impacts forecast (OUT TECNICA E CONFRONTO NEWS).
"""

import logging
from typing import Dict, Any

try:
    from agents.technicalNewsAgent import TechnicalNewsAgent
except ImportError:
    from backend.agents.technicalNewsAgent import TechnicalNewsAgent

from .state import PipelineState

logger = logging.getLogger(__name__)


def generate_agent_2_out(state: PipelineState) -> Dict[str, Any]:
    """
    Agent 2: Combines technical prediction (OUT 2) and current news sentiment (NEWS)
    to generate technical analysis and news comparison with Mermaid charts.
    """
    prediction = state.get("prediction_out2", "No prediction available.")
    news = state.get("news_data", "No news data available.")
    isin = state.get("isin", "N/A")

    logger.info("Generating Agent 2 technical analysis and news comparison for ISIN: %s", isin)

    out_tech = TechnicalNewsAgent().run(isin=isin, prediction=prediction, news=news)

    return {"agent_2_out_tech": out_tech}
