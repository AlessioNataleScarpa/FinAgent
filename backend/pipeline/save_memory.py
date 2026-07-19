"""
Node: SAVE MEMORY
Persists the completed ETF analysis for ConversationAgent follow-ups.
"""

import logging
from typing import Any, Dict

try:
    from memory.store import get_memory_store
except ImportError:
    from backend.memory.store import get_memory_store

from .state import PipelineState

logger = logging.getLogger(__name__)


def save_memory_node(state: PipelineState) -> Dict[str, Any]:
    isin = (state.get("isin") or "").strip().upper()
    report = state.get("out_finale") or ""

    if not isin or not report:
        logger.warning("Skipping memory save: missing isin or report")
        return {"memory_saved": False}

    store = get_memory_store()
    store.save_analysis(
        isin,
        report=report,
        presentation=state.get("agent_1_out1") or "",
        technical=state.get("agent_2_out_tech") or "",
        composition_charts=state.get("composition_charts") or "",
        timeline_charts=state.get("timeline_charts") or "",
        sentiment_charts=state.get("sentiment_charts") or "",
        info_presentazione=state.get("info_presentazione") or "",
        info_storici=state.get("info_storici") or "",
        news_data=state.get("news_data") or "",
        prediction=state.get("prediction_out2") or "",
        clean_query=state.get("clean_query") or "",
    )
    logger.info("Saved ETF analysis memory for ISIN: %s", isin)
    return {"memory_saved": True}
