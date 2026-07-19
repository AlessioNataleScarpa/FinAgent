import logging
import httpx
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Avoid duplicate OpenFIGI calls across parallel LangGraph branches.
_TICKER_CACHE: Dict[str, Optional[str]] = {}

def isin_to_ticker(isin: str, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Converte un ISIN in una lista di Ticker corrispondenti utilizzando l'API di OpenFIGI.
    
    Args:
        isin: Il codice ISIN (es. "US0378331005")
        api_key: Chiave API opzionale per OpenFIGI (aumenta i limiti di rate).
        
    Returns:
        Una lista di dizionari con i risultati del mapping.
        Esempio: [{'figi': '...', 'name': 'APPLE INC', 'ticker': 'AAPL', 'exchCode': 'US', ...}]
    """
    url = "https://api.openfigi.com/v3/mapping"
    payload = [{"idType": "ID_ISIN", "idValue": isin}]
    headers = {'Content-Type': 'application/json'}
    
    if api_key:
        headers['X-OPENFIGI-APIKEY'] = api_key
        
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                # OpenFIGI restituisce un array di risposte, una per ogni query nell'array payload.
                if data and isinstance(data, list) and len(data) > 0:
                    first_mapping = data[0]
                    # I risultati effettivi sono dentro la chiave 'data'
                    if 'data' in first_mapping:
                        return first_mapping['data']
                    elif 'error' in first_mapping:
                        logger.error(f"OpenFIGI ha restituito un errore per {isin}: {first_mapping['error']}")
                        return []
            else:
                logger.error(f"Errore HTTP OpenFIGI: {response.status_code} {response.text}")
                return []
    except Exception as e:
        logger.error(f"Impossibile connettersi a OpenFIGI: {e}")
        
    return []

def get_best_ticker(isin: str, preferred_exchange: str = "US", api_key: Optional[str] = None) -> Optional[str]:
    """
    Estrae un ticker da un ISIN, privilegiando exchange liquidi per ETF (L/LN/MI/US).
    """
    key = (isin or "").strip().upper()
    if not key:
        return None
    if key in _TICKER_CACHE:
        return _TICKER_CACHE[key]

    results = isin_to_ticker(key, api_key)
    if not results:
        _TICKER_CACHE[key] = None
        return None

    preferred = [preferred_exchange, "L", "LN", "MI", "US", "GY", "NA"]
    ticker: Optional[str] = None
    for exch in preferred:
        if not exch:
            continue
        for res in results:
            if res.get("exchCode") == exch and res.get("ticker"):
                ticker = res.get("ticker")
                # Yahoo often needs a suffix for non-US listings.
                if exch in {"L", "LN"} and "." not in ticker:
                    ticker = f"{ticker}.L"
                elif exch == "MI" and "." not in ticker:
                    ticker = f"{ticker}.MI"
                break
        if ticker:
            break

    if not ticker:
        ticker = results[0].get("ticker")

    _TICKER_CACHE[key] = ticker
    return ticker
