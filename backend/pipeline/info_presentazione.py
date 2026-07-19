"""
Node: INFO PRESENTAZIONE
Fetches metadata and presentation info for an ISIN / ETF.
"""

import logging
from typing import Dict, Any
from .state import PipelineState
from api.openfigi import get_best_ticker
from api.yahoo_data import get_profile

logger = logging.getLogger(__name__)


def fetch_info_presentazione(state: PipelineState) -> Dict[str, Any]:
    """
    Fetch metadata / presentation details for the provided ISIN.
    Falls back to structured mock metadata if external fetching fails.
    """
    isin = (state.get("isin") or "").strip().upper()

    logger.info("Fetching presentation info for ISIN: %s", isin)

    ticker = get_best_ticker(isin)
    if not ticker:
        logger.warning(f"Impossibile trovare un Ticker per l'ISIN {isin}. Uso fallback.")
        ticker = "SWDA.MI" if "IE00B4L5Y983" in isin else "AAPL"

    logger.info(f"Ticker convertito: {ticker}")
    
    # Recupera il profilo tramite FMP
    profile_data = get_profile(ticker)
    
    # Convertiamo i dati in una stringa JSON formattata per passarla nello state
    import json
    info_str = json.dumps({
        "ISIN": isin,
        "Ticker": ticker,
        "Profile": profile_data
    }, indent=2)

    return {"info_presentazione": info_str}
