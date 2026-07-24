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
                "source": "Yahoo Finance via MCP",
                "isin": isin,
                "ticker": ticker,
                "articles": articles,
            },
            ensure_ascii=False,
            indent=2,
        )
    else:
        error = payload.get("error", "ticker non disponibile") if isinstance(payload, dict) else "errore MCP"
        news_text = (
            f"--- RECENT MARKET NEWS & SENTIMENT ({isin}) ---\n"
            f"[FALLBACK - LIVE NEWS UNAVAILABLE: {error}]\n"
            f"1. [MACRO]: Monitorare tassi, inflazione e liquidità globale.\n"
            f"2. [MARKET TRENDS]: Verificare flussi e performance del settore di riferimento.\n"
            f"3. [RISK FACTORS]: Geopolitica e volatilità restano fattori di rischio.\n"
        )

    return {"news_data": news_text}
