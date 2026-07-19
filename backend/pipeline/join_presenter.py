"""
Node: JOIN
Synthesizes presentation, technical analysis and Mermaid chart modules into OUT FINALE.
"""

import logging
from typing import Any, Dict

try:
    from agents.joinAgent import JoinAgent
except ImportError:
    from backend.agents.joinAgent import JoinAgent

from .state import PipelineState

logger = logging.getLogger(__name__)


def join_presenter_node(state: PipelineState) -> Dict[str, Any]:
    out1 = state.get("agent_1_out1", "") or ""
    out_tech = state.get("agent_2_out_tech", "") or ""
    isin = state.get("isin", "") or ""
    composition = state.get("composition_charts", "") or ""
    timeline = state.get("timeline_charts", "") or ""
    sentiment = state.get("sentiment_charts", "") or ""

    logger.info("Executing JOIN node for final Markdown report (ISIN: %s)", isin)

    out_finale = JoinAgent().run_sync(
        out1,
        out_tech,
        isin=isin,
        composition_charts=composition,
        timeline_charts=timeline,
        sentiment_charts=sentiment,
    )

    return {"out_finale": out_finale}
