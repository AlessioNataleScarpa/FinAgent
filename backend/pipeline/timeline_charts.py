"""
Node: TIMELINE CHARTS
Builds Mermaid XY charts for historical price trends.
"""

import logging
from typing import Any, Dict

try:
    from agents.timelineChartAgent import TimelineChartAgent
except ImportError:
    from backend.agents.timelineChartAgent import TimelineChartAgent

from .state import PipelineState

logger = logging.getLogger(__name__)


def timeline_charts_node(state: PipelineState) -> Dict[str, Any]:
    isin = state.get("isin", "N/A")
    info = state.get("info_storici", "")
    logger.info("Building timeline Mermaid charts for ISIN: %s", isin)
    markdown = TimelineChartAgent().build_markdown(isin=isin, info_storici=info or "")
    return {"timeline_charts": markdown}
