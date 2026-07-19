import logging
import httpx
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

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
    Funzione di utilità per estrarre direttamente un singolo ticker da un ISIN.
    Si può specificare un exchange preferito (es. 'US', 'IT', 'LN').
    """
    results = isin_to_ticker(isin, api_key)
    if not results:
        return None
        
    # Cerca prima una corrispondenza esatta con l'exchange preferito
    if preferred_exchange:
        for res in results:
            if res.get('exchCode') == preferred_exchange:
                return res.get('ticker')
                
    # Se non trova l'exchange preferito, restituisce il primo ticker disponibile
    return results[0].get('ticker')
