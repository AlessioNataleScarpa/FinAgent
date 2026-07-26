"""
Node: NEWS
Fetches recent market news relevant to the ISIN.
"""

import json
import logging
from typing import Dict, Any

from api.openfigi import get_best_ticker
from api.yahoo_data import get_news
from .state import PipelineState

logger = logging.getLogger(__name__)


def fetch_news(state: PipelineState) -> Dict[str, Any]:
    """
    Fetches recent news articles through the Yahoo Finance MCP server.
    The fallback is explicitly labelled so synthetic context is never mistaken for live news.
    """
    isin = (state.get("isin") or "").strip()
    logger.info("Fetching market news for ISIN: %s", isin)

    ticker = get_best_ticker(isin)
    payload = get_news(ticker, limit=8, query=isin) if ticker else {}
    articles = payload.get("articles") if isinstance(payload, dict) else None

    if articles:
        news_text = json.dumps(
            {
                "source": "Yahoo Finance",
                "isin": isin,
                "ticker": ticker,
                "articles": articles,
            },
            ensure_ascii=False,
            indent=2,
        )
    else:
        error = payload.get("error", "ticker non disponibile") if isinstance(payload, dict) else "fonte esterna non raggiungibile"
        news_text = (
            f"Le notizie aggiornate non sono disponibili ({error}). In assenza di "
            "articoli verificabili, l'analisi considera soltanto i rischi generali: "
            "tassi, inflazione, liquidità, flussi di mercato, geopolitica e volatilità. "
            "Questi fattori non vengono trattati come notizie né usati per rafforzare "
            "artificialmente la previsione.\n"
        )

    return {"news_data": news_text}
