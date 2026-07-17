"""
Node: JOIN
Concatenates OUT 1 and OUT TECNICA E CONFRONTO NEWS into OUT FINALE.
"""

import logging
from typing import Dict, Any
from .state import PipelineState

logger = logging.getLogger(__name__)


try:
    from agents.joinAgent import JoinAgent
except ImportError:
    from backend.agents.joinAgent import JoinAgent

from .state import PipelineState

logger = logging.getLogger(__name__)


def join_presenter_node(state: PipelineState) -> Dict[str, Any]:
    """
    Join Node: Concatenates presentation (OUT 1) and technical/news analysis (OUT TECNICA)
    into a comprehensive final output (OUT FINALE) using JoinAgent.
    """
    out1 = state.get("agent_1_out1", "")
    out_tech = state.get("agent_2_out_tech", "")
    isin = state.get("isin", "")

    logger.info("Executing JOIN node for final report output (ISIN: %s)", isin)

    out_finale = JoinAgent().run_sync(out1, out_tech)

    return {"out_finale": out_finale}

