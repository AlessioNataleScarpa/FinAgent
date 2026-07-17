# DL-2026

Progetto universitario (Deep Learning) per un sistema AI che analizza e presenta ETF a partire da un codice ISIN.
Il progetto integra un backend FastAPI, una pipeline LangGraph e agenti specializzati per generare presentazioni, analisi tecnica, confronto notizie e sintesi finale.

## Architettura

```
OpenWebUI (:3000) → backend FastAPI (:8000/v1) → agents / pipeline → Google Gemini / LLM
```

- `backend/main.py`: applicazione FastAPI.
- `backend/api/routes.py`: API compatibili OpenAI su `/v1/models` e `/v1/chat/completions`.
- `backend/agents/registry.py`: registro agent disponibili.
- `backend/agents/base.py`: classe base per tutti gli agent.
- `backend/agents/gatewayAgent.py`: router principale che identifica intent, validazione ISIN e invoca la pipeline.
- `backend/agents/presentationAgent.py`: agente per la presentazione dell'ETF (OUT 1).
- `backend/agents/technicalNewsAgent.py`: agente per analisi tecnica e confronto con le notizie (OUT TECNICA).
- `backend/agents/joinAgent.py`: agente di sintesi finale che unisce i risultati in un report.

## Agent disponibili

- `gatewayAgent`
- `presentationAgent`
- `technicalNewsAgent`
- `joinAgent`

## Pipeline LangGraph

La pipeline in `backend/pipeline/graph.py` esegue una composizione multi-stage:

1. `info_presentazione` → `agent_1`
2. `news` + `info_andamenti_storici` → `predict` → `agent_2`
3. `agent_1` + `agent_2` → `join_presenter`

### Nodi principali

- `backend/pipeline/info_presentazione.py`: recupera i metadati e le informazioni di presentazione ETF.
- `backend/pipeline/news.py`: genera un blocco di notizie e sentiment di mercato.
- `backend/pipeline/info_andamenti_storici.py`: recupera dati storici con fallback realistico.
- `backend/pipeline/predict.py`: genera una previsione sintetica (OUT 2).
- `backend/pipeline/agent_1.py`: invoca `PresentationAgent` e produce `agent_1_out1`.
- `backend/pipeline/agent_2.py`: invoca `TechnicalNewsAgent` e produce `agent_2_out_tech`.
- `backend/pipeline/join_presenter.py`: sintetizza i due output in un report finale.

## Avvio con Docker

```bash
docker compose up --build
```

### Servizi

- OpenWebUI: `http://localhost:3000`
- Backend FastAPI: `http://localhost:8000`

### Nota

La configurazione `docker-compose.yml` include:
- servizio `open-webui`
- servizio `backend`
- volume persistente `open-webui-data`

## Variabili d'ambiente

- `GOOGLE_API_KEY`: chiave API per Google Gemini.
- `GEMINI_MODEL`: modello LLM per gli agent. Default: `gemma-4-31b-it`.

> Il backend utilizza `langchain-google-genai` e `ChatGoogleGenerativeAI`.

## API

- `GET /v1/models`: restituisce la lista degli agent registrati.
- `POST /v1/chat/completions`: invia messaggi compatibili OpenAI e riceve la risposta dall'agent selezionato.

### Esempio richiesta

```json
{
  "model": "gatewayAgent",
  "messages": [
    {"role": "user", "content": "Analizza l'ETF con ISIN IE00B4L5Y983"}
  ]
}
```

## Requisiti

Dipendenze Python principali in `requirements.txt`:

- `fastapi`
- `uvicorn`
- `langchain-google-genai`
- `langchain-core`
- `langgraph`
- `pydantic`
- `pytest`
- `pytest-asyncio`
- `httpx`

## Come estendere il progetto

1. Crea una nuova classe in `backend/agents/` estendendo `BaseAgent`.
2. Implementa `model_id` e `run(...)`.
3. Registra l'agente in `backend/agents/registry.py`.
4. Aggiungi prompt dedicati in `backend/prompts/` se necessario.

## Stato corrente

- Architettura agent-based con pipeline multi-stage.
- Output testuali e Mermaid markdown per visualizzare grafici e analisi.
- Il gateway gestisce la classificazione dell'intento, l'estrazione ISIN e l'invocazione della pipeline.
- La pipeline è pronta per integrazioni più profonde con dati reali e modelli LLM.
