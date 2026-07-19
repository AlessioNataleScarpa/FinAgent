"""
Node: INFO ANDAMENTI STORICI
Fetches historical performance data using FMP
"""

import logging
import json
from typing import Dict, Any
from .state import PipelineState
from api.openfigi import get_best_ticker
from api.yahoo_data import get_historical_data

logger = logging.getLogger(__name__)

def fetch_info_andamenti_storici(state: PipelineState) -> Dict[str, Any]:
    """
    Recupera i dati storici tramite FMP usando il ticker ottenuto dall'ISIN.
    """
    isin = state.get("isin", "N/A")
    logger.info("Fetching historical info for ISIN: %s", isin)

    ticker = get_best_ticker(isin)
    if not ticker:
        logger.warning(f"Impossibile trovare un Ticker per l'ISIN {isin}. Uso fallback.")
        ticker = "SWDA.MI" if "IE00B4L5Y983" in isin else "AAPL"

    logger.info(f"Ticker convertito: {ticker}")
    
    # Recupera i dati storici (ultimo anno = "1y")
    historical_data = get_historical_data(ticker, period="1y")
    
    monthly = []
    if isinstance(historical_data, dict):
        monthly = historical_data.get("Monthly_Closes") or []

    # Solo serie mensile: evita payload enormi nello state / nei prompt.
    info_str = json.dumps(
        {
            "ISIN": isin,
            "Ticker": ticker,
            "Monthly_Closes": monthly,
        },
        indent=2,
    )

    return {"info_storici": info_str}
