# DL-2026

Progetto universitario (Deep Learning): agente intelligente per prevedere l'andamento a lungo termine di un ETF a partire dal codice ISIN, con spiegazione logica e confronto critico con notizie macroeconomiche.

## Architettura (scaffold attuale)

```
OpenWebUI (:3000) → backend FastAPI (:8000/v1) → gatewayAgent → Ollama / OpenAI
```

Gli agent sono esposti come modelli OpenAI-compatible. In OpenWebUI selezioni `gatewayAgent` dal dropdown.

Per ora c'è **un solo agent** (`gatewayAgent`): filtra il dominio ETF/ISIN e, se in-domain, restituisce uno stub JSON (pipeline di analisi non ancora collegata).

## Avvio

```bash
docker compose up --build
```

Poi (prima volta / se manca il modello):

```bash
docker exec dl2026_ollama ollama pull phi3
```

UI: http://localhost:3000  
API: http://localhost:8000/v1/models

## LLM (env)

| Variabile | Default | Esempio OpenAI |
|-----------|---------|----------------|
| `OPENAI_API_BASE` | `http://ollama:11434/v1` | `https://api.openai.com/v1` |
| `OPENAI_API_KEY` | `ollama` | la tua chiave |
| `OPENAI_MODEL` | `phi3` | `gpt-4o-mini` |

Esempio override:

```bash
OPENAI_MODEL=phi3 docker compose up --build
```

## Estendere con nuovi agent

1. Crea una classe che estende `BaseAgent` in `backend/agents/`
2. Registrala in `backend/agents/registry.py`
3. Appare subito in OpenWebUI come modello selezionabile
