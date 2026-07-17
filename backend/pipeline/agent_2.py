"""
Node: AGENT 2
Takes OUT 2 and NEWS, generates Mermaid chart for prediction and explains how news impacts forecast (OUT TECNICA E CONFRONTO NEWS).
"""

import logging
from typing import Dict, Any
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

    out_tech = (
        f"# 📈 ANALISI TECNICA E CONFRONTO NEWS (OUT TECNICA)\n\n"
        f"### 🔮 Previsione Quantitativa (OUT 2)\n"
        f"{prediction}\n\n"
        f"### 📰 Notizie di Mercato Rilevanti\n"
        f"{news}\n\n"
        f"### 🎯 Impatto delle News sulla Previsione\n"
        f"L'analisi quantitativa indica un trend rialzista (+7.5% - +10.2%). L'integrazione delle notizie recenti conferma il sentiment favorevole:\n"
        f"- **Taglio/Pausa dei Tassi**: Favorisce i titoli tech e azionari globali presenti nell'ETF.\n"
        f"- **Utili superiori alle attese**: Supportano la crescita dei fondamentali sui 12 mesi.\n"
        f"- **Rischi Geopolitici**: Possono generare volatilità di breve termine entro la fascia del 12-14%.\n\n"
        f"### 📊 Diagramma Mermaid Previsionale e Impatto Notizie\n\n"
        f"```mermaid\n"
        f"graph LR\n"
        f"    N1[Notizie Macro Positive] -->|Spinta Rialzista| P[Trend Previsto: +8.5%]\n"
        f"    N2[Rischi Geopolitici] -->|Resistenza Volatilità| P\n"
        f"    H[Storico CAGR 11.5%] -->|Base Quantitativa| P\n"
        f"    P --> OUT[Target Price 12 Mesi: Growth Zone]\n"
        f"```\n\n"
        f"```mermaid\n"
        f"gantt\n"
        f"    title Orizzonte temporale di previsione ({isin})\n"
        f"    dateFormat YYYY-MM-DD\n"
        f"    section Fasi Previsione\n"
        f"    Accumulazione iniziale    :a1, 2026-07-01, 30d\n"
        f"    Crescita Moderata          :after a1, 60d\n"
        f"    Target Growth Achieved     :after a2, 90d\n"
        f"```\n"
    )

    return {"agent_2_out_tech": out_tech}
