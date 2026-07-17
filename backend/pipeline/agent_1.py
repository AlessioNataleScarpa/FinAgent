"""
Node: AGENT 1
Formats presentation and generates Mermaid chart/diagram for the object (OUT 1).
"""

import logging
from typing import Dict, Any

try:
    from agents.presentationAgent import PresentationAgent
except ImportError:
    from backend.agents.presentationAgent import PresentationAgent

from .state import PipelineState

logger = logging.getLogger(__name__)


def generate_agent_1_out(state: PipelineState) -> Dict[str, Any]:
    """
    Agent 1: Formats asset presentation and generates a Mermaid diagram showing sector/region breakdown.
    """
    info = state.get("info_presentazione", "No presentation info available.")
    isin = state.get("isin", "N/A")

    logger.info("Generating Agent 1 presentation and Mermaid diagram for ISIN: %s", isin)

    out1 = PresentationAgent().run(isin=isin, info_presentazione=info)

    return {"agent_1_out1": out1}
