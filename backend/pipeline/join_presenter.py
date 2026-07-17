"""
Node: JOIN
Concatenates OUT 1 and OUT TECNICA E CONFRONTO NEWS into OUT FINALE.
"""

import logging
from typing import Dict, Any
from .state import PipelineState

logger = logging.getLogger(__name__)


def join_presenter_node(state: PipelineState) -> Dict[str, Any]:
    """
    Join Node: Concatenates presentation (OUT 1) and technical/news analysis (OUT TECNICA)
    into a comprehensive final output (OUT FINALE).
    """
    out1 = state.get("agent_1_out1", "")
    out_tech = state.get("agent_2_out_tech", "")
    isin = state.get("isin", "")

    logger.info("Executing JOIN node for final report output (ISIN: %s)", isin)

    out_finale = (
        f"# 🏆 REPORT COMPLETO DI ANALISI STRUMENTO FINANZIARIO ({isin})\n\n"
        f"{out1}\n\n"
        f"---\n\n"
        f"{out_tech}\n\n"
        f"---\n"
        f"*Report generato automaticamente dalla Pipeline LangGraph.*"
    )

    return {"out_finale": out_finale}
