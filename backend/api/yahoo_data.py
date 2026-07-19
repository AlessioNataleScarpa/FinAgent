import logging
from typing import Any, Dict, List

import yfinance as yf

logger = logging.getLogger(__name__)


def _extract_sector_weights(info: Dict[str, Any], ticker_obj: yf.Ticker) -> Dict[str, float]:
    candidates = [
        info.get("sectorWeightings"),
        info.get("categoryWeightings"),
        info.get("sectorWeighting"),
    ]

    try:
        funds = getattr(ticker_obj, "funds_data", None)
        if funds is not None:
            sector = getattr(funds, "sector_weightings", None)
            if isinstance(sector, dict):
                candidates.append(sector)
    except Exception:
        pass

    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        weights: Dict[str, float] = {}
        for key, value in raw.items():
            try:
                if isinstance(value, dict):
                    numeric = float(next(iter(value.values())))
                else:
                    numeric = float(value)
                # yfinance sometimes returns 0-1 fractions
                if 0 < numeric <= 1:
                    numeric *= 100.0
                if numeric > 0:
                    weights[str(key)] = round(numeric, 2)
            except (TypeError, ValueError, StopIteration):
                continue
        if weights:
            return weights
    return {}


def _extract_asset_allocation(info: Dict[str, Any]) -> Dict[str, float]:
    raw = info.get("assetAllocation") or info.get("holdings")
    weights: Dict[str, float] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                numeric = float(value)
                if 0 < numeric <= 1:
                    numeric *= 100.0
                if numeric > 0:
                    weights[str(key)] = round(numeric, 2)
            except (TypeError, ValueError):
                continue
    quote_type = str(info.get("quoteType") or info.get("typeDisp") or "").upper()
    if not weights and ("ETF" in quote_type or "FUND" in quote_type or info.get("longBusinessSummary")):
        weights = {"Azioni / Equity": 98.0, "Liquidita / Altro": 2.0}
    return weights


def get_profile(ticker: str) -> Dict[str, Any]:
    """
    Recupera il profilo aziendale o dell'ETF tramite yfinance.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        if info:
            profile_data = {
                "name": info.get("shortName", info.get("longName")),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "description": info.get("longBusinessSummary", "N/A"),
                "marketCap": info.get("marketCap", "N/A"),
                "currency": info.get("currency", "N/A"),
                "exchange": info.get("exchange", "N/A"),
                "previousClose": info.get("previousClose", "N/A"),
                "yield": info.get("yield", "N/A"),
                "category": info.get("category", "N/A"),
                "totalAssets": info.get("totalAssets", "N/A"),
                "sectorWeightings": _extract_sector_weights(info, t),
                "assetAllocation": _extract_asset_allocation(info),
            }
            return profile_data

        logger.warning("Nessun profilo trovato su Yahoo Finance per il ticker %s", ticker)
        return {"error": "Profile not found"}
    except Exception as e:
        logger.error("Errore durante il recupero del profilo Yahoo Finance per %s: %s", ticker, e)
        return {"error": str(e)}


def get_historical_data(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """
    Recupera una serie mensile compatta (no daily dump) per i grafici Mermaid.
    """
    try:
        t = yf.Ticker(ticker)
        # Prefer monthly bars to avoid downloading ~250 daily rows.
        hist = t.history(period=period, interval="1mo")
        if hist.empty:
            hist = t.history(period=period)

        if hist.empty:
            logger.warning("Nessun dato storico trovato su Yahoo Finance per il ticker %s", ticker)
            return {"error": "Historical data not found"}

        hist = hist.copy()
        hist.reset_index(inplace=True)
        hist["Date"] = hist["Date"].astype(str)

        monthly = (
            hist.assign(month=hist["Date"].astype(str).str.slice(0, 7))
            .groupby("month", as_index=False)
            .agg(close=("Close", "last"))
        )
        monthly_closes: List[Dict[str, Any]] = [
            {"month": row["month"], "close": round(float(row["close"]), 4)}
            for _, row in monthly.iterrows()
        ]

        return {"Monthly_Closes": monthly_closes}
    except Exception as e:
        logger.error(
            "Errore durante il recupero dei dati storici Yahoo Finance per %s: %s",
            ticker,
            e,
        )
        return {"error": str(e)}
