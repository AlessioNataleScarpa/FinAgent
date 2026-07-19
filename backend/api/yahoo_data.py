import logging
import yfinance as yf
from typing import Dict, Any

logger = logging.getLogger(__name__)

def get_profile(ticker: str) -> Dict[str, Any]:
    """
    Recupera il profilo aziendale o dell'ETF tramite yfinance.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info
        if info:
            # Estraiamo i campi più rilevanti per non saturare l'LLM con dati inutili
            profile_data = {
                "name": info.get("shortName", info.get("longName")),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "description": info.get("longBusinessSummary", "N/A"),
                "marketCap": info.get("marketCap", "N/A"),
                "currency": info.get("currency", "N/A"),
                "exchange": info.get("exchange", "N/A"),
                "previousClose": info.get("previousClose", "N/A"),
                "yield": info.get("yield", "N/A")
            }
            return profile_data
        else:
            logger.warning(f"Nessun profilo trovato su Yahoo Finance per il ticker {ticker}")
            return {"error": "Profile not found"}
    except Exception as e:
        logger.error(f"Errore durante il recupero del profilo Yahoo Finance per {ticker}: {e}")
        return {"error": str(e)}

def get_historical_data(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """
    Recupera i dati storici tramite yfinance.
    period può essere: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    """
    try:
        t = yf.Ticker(ticker)
        # Scarica lo storico
        hist = t.history(period=period)
        
        if not hist.empty:
            # Per evitare un payload gigantesco (JSON con 250 giorni * 5 colonne),
            # possiamo restituire direttamente i record convertiti in JSON.
            # Reimpostiamo l'indice (Date) per averlo come stringa
            hist.reset_index(inplace=True)
            hist['Date'] = hist['Date'].astype(str)
            
            # Convertiamo il dataframe in una lista di dizionari
            records = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].to_dict(orient="records")
            return {"historical": records}
        else:
            logger.warning(f"Nessun dato storico trovato su Yahoo Finance per il ticker {ticker}")
            return {"error": "Historical data not found"}
    except Exception as e:
        logger.error(f"Errore durante il recupero dei dati storici Yahoo Finance per {ticker}: {e}")
        return {"error": str(e)}
