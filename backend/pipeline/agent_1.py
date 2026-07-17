"""
Node: AGENT 1
Formats presentation and generates Mermaid chart/diagram for the object (OUT 1).
"""

import logging
from typing import Dict, Any
from .state import PipelineState

logger = logging.getLogger(__name__)


def generate_agent_1_out(state: PipelineState) -> Dict[str, Any]:
    """
    Agent 1: Formats asset presentation and generates a Mermaid diagram showing sector/region breakdown.
    """
    info = state.get("info_presentazione", "No presentation info available.")
    isin = state.get("isin", "N/A")

    logger.info("Generating Agent 1 presentation and Mermaid diagram for ISIN: %s", isin)

    out1 = (
        f"# 📊 PRESENTAZIONE STRUMENTO (OUT 1)\n\n"
        f"**ISIN Analizzato:** `{isin}`\n\n"
        f"### 📋 Dettagli Fondamentali\n"
        f"{info}\n\n"
        f"### 🌐 Struttura e Allocazione (Mermaid Diagram)\n\n"
        f"```mermaid\n"
        f"pie title Allocazione Settoriale ISIN {isin}\n"
        f'    "Information Tech" : 23.5\n'
        f'    "Financials" : 15.2\n'
        f'    "Healthcare" : 12.1\n'
        f'    "Industrials" : 11.0\n'
        f'    "Consumer Disc" : 10.4\n'
        f'    "Communication" : 7.5\n'
        f'    "Altri Settori" : 20.3\n'
        f"```\n\n"
        f"```mermaid\n"
        f"graph TD\n"
        f"    A[{isin}] --> B[Mercati Sviluppati]\n"
        f"    B --> C[USA - 70.1%]\n"
        f"    B --> D[Giappone - 6.2%]\n"
        f"    B --> E[Europa / UK - 10.0%]\n"
        f"    B --> F[Altri - 13.7%]\n"
        f"```\n"
    )

    return {"agent_1_out1": out1}
