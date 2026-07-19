"""
Node: COMPOSITION CHARTS
Builds Mermaid pie charts for sector/asset allocation.
"""

import logging
from typing import Any, Dict

try:
    from agents.compositionChartAgent import CompositionChartAgent
except ImportError:
    from backend.agents.compositionChartAgent import CompositionChartAgent

from .state import PipelineState

logger = logging.getLogger(__name__)


def composition_charts_node(state: PipelineState) -> Dict[str, Any]:
    isin = state.get("isin", "N/A")
    info = state.get("info_presentazione", "")
    logger.info("Building composition Mermaid charts for ISIN: %s", isin)
    markdown = CompositionChartAgent().build_markdown(isin=isin, info_presentazione=info or "")
    return {"composition_charts": markdown}
