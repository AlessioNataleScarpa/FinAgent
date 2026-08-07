# DL-2026

Progetto universitario (Deep Learning) per un sistema AI che analizza e presenta ETF a partire da un codice ISIN.
Il progetto integra un backend FastAPI, una pipeline LangGraph e agenti specializzati per generare presentazioni, analisi tecnica, confronto notizie e sintesi finale.

## Architettura

```
OpenWebUI (:3000) → backend FastAPI (:8000/v1) → LangGraph
                                                ├→ Google Gemini / LLM
                                                └→ Chronos TSFM (:8001)
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

## Pipeline LangGraph dual-stream (zero-shot)

La pipeline in `backend/pipeline/graph.py` esegue una composizione multi-stage:

1. `info_presentazione` → `agent_1` (contesto statico).
2. `info_andamenti_storici` → `predict` (forecast TSFM zero-shot).
3. `predict` → spiegazione XAI per occlusione temporale.
4. `news` + `predict` + XAI → `agent_2` (fusione quantitativa/qualitativa).
5. `predict` → `forecast_charts` genera viste a 1, 5, 10 e 20 anni con intervallo 80%.
6. I rami e i grafici convergono in `join_presenter`.

I tre nodi di raccolta partono in parallelo. Lo stato conserva separatamente
prezzi, date e forecast strutturato (`mean`, `lower_bound`, `upper_bound`), così
gli array non devono essere ricostruiti dai prompt.

### Nodi principali

- `backend/pipeline/info_presentazione.py`: recupera i metadati e le informazioni di presentazione ETF.
- `backend/pipeline/news.py`: recupera news reali tramite Yahoo Finance MCP.
- `backend/pipeline/info_andamenti_storici.py`: recupera OHLCV mensile tramite MCP.
- `backend/pipeline/predict.py`: invoca un TSFM e genera OUT 2 con intervalli.
- `backend/pipeline/explain_forecast.py`: rende leggibili le attribuzioni XAI e i fattori di rischio.
- `backend/pipeline/forecast_charts.py`: mostra forecast centrale, limite inferiore e superiore.
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
- Chronos TSFM: `http://localhost:8001` (avviato automaticamente)

### Nota

La configurazione `docker-compose.yml` include:
- servizio `open-webui`
- servizio `backend`
- servizio `tsfm` con Amazon Chronos-Bolt Tiny su CPU
- volumi persistenti `open-webui-data` e `tsfm-model-cache`

## Variabili d'ambiente

- `GOOGLE_API_KEY`: chiave API per Google Gemini.
- `GEMINI_MODEL`: modello LLM per gli agent. Default: `gemma-4-31b-it`.
- `TSFM_MODEL_ID`: modello locale, default `amazon/chronos-bolt-tiny`.
- `TSFM_TORCH_THREADS`: thread CPU assegnati al modello, default `2`.
- `TSFM_ENDPOINT`: override opzionale; normalmente non va impostato.
- `TSFM_API_TOKEN`: bearer token opzionale.
- `TSFM_HORIZON`: orizzonte mensile, default `240` (20 anni).
- `TSFM_HISTORY_PERIOD`: storico richiesto a Yahoo, default `5y`.
- `COMPOSITION_TIMEOUT_SECONDS`: timeout per ricerca e holdings ufficiali, default `35`.
- `COMPOSITION_CACHE_TTL_SECONDS`: durata cache composizione, default `21600` (6 ore).
- `LANGGRAPH_POSTGRES_URI`: URI PostgreSQL opzionale per `AsyncPostgresSaver`.

> Il backend utilizza `langchain-google-genai` e `ChatGoogleGenerativeAI`.

### Container TSFM locale

`docker compose up --build` costruisce e avvia automaticamente
`tsfm_service/`. Al primo avvio scarica Chronos-Bolt Tiny nel volume
persistente; gli avvii successivi riutilizzano la cache. Il backend attende
l'health check del modello e poi usa internamente
`http://tsfm:8081/forecast`.

Il modello produce 240 punti mensili in una sola esecuzione. Il report mostra
il primo anno mese per mese e gli orizzonti 5/10/20 anni con campionamento
annuale, sempre con scenario centrale, limite inferiore e limite superiore.
Le viste più lunghe sono dichiarate esplorative perché l'incertezza cumulativa
è sostanzialmente maggiore.

Il servizio accetta internamente richieste come:

```json
{
  "model": "amazon/chronos-bolt-tiny",
  "series": [101.2, 103.8, 102.4, 104.0, 105.1, 106.3, 105.8, 107.2],
  "frequency": "M",
  "prediction_length": 3,
  "quantile_levels": [0.1, 0.5, 0.9],
  "explain": true
}
```

Quando `explain` è attivo, Chronos viene rieseguito in batch su tre serie
controfattuali: ultimi 6 mesi, 7–18 mesi e 19–36 mesi vengono neutralizzati una
finestra alla volta. La variazione della previsione a 12 mesi misura la
sensibilità locale del modello. Il report non presenta questa misura come
causalità di mercato.

### Composizione del portafoglio

I grafici di composizione non usano Yahoo e non hanno valori predefiniti. Il
backend cerca la pagina dell'emittente e, per i fondi iShares/BlackRock,
aggrega il file holdings ufficiale per settore, paese e classe di attivo. Se
non trova una fonte supportata e verificabile, omette il grafico. Quando una
classe supera il 95%, la torta quasi monocromatica viene sostituita da una nota
testuale e restano visibili le ripartizioni più informative.

L'endpoint deve rispondere con vettori lunghi quanto `prediction_length`:

```json
{
  "model": "amazon/chronos-bolt-tiny",
  "mean": [104.1, 105.0, 106.2],
  "lower_bound": [91.3, 90.8, 90.1],
  "upper_bound": [117.8, 119.6, 122.0]
}
```

Se il container TSFM fallisce durante una singola richiesta, il report usa un
damped-trend robusto e lo marca chiaramente come `fallback`, senza spacciarlo
per un Foundation Model.

### Checkpoint PostgreSQL

Per abilitare la persistenza nativa di LangGraph, creare il database sul sistema
host e impostare, ad esempio:

```dotenv
LANGGRAPH_POSTGRES_URI=postgresql://utente:password@host.docker.internal:5432/dl2026
```

All'avvio FastAPI esegue il setup di `AsyncPostgresSaver`; se PostgreSQL non è
configurato o non è raggiungibile, la pipeline continua senza checkpoint.

### Memoria e domande successive

Ogni analisi completa salva report, dati, news, forecast numerico e grafici in
`backend/data/etf_memory.json`, indicizzati per ISIN. Continuando nella stessa
chat con `gatewayAgent`, il nodo `conversation` usa questi artefatti per
rispondere a domande su previsioni, intervalli, composizione e news. Se il
servizio LLM non risponde, viene comunque restituita una risposta deterministica
dalla memoria invece di un errore HTTP.

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

## Report e valutazione

Il report LaTeX è in `report_progetto.tex` (PDF: `report_progetto.pdf`).

Artefatti sperimentali:
- `report/figures/` — heatmap, confusion matrix, ablation temperatura, metriche
- `report/results/` — CSV/JSON dei test (`summary.json`)
- `scripts/eval_suite.py` — suite riproducibile (routing, Mermaid, ISIN, schema, LLM)

```bash
python -u scripts/eval_suite.py --llm-limit 20
# solo offline (senza API):
python -u scripts/eval_suite.py --skip-llm
pdflatex report_progetto.tex
```

## Stato corrente

- Architettura agent-based con pipeline multi-stage.
- Output testuali e Mermaid markdown per visualizzare grafici e analisi.
- Il gateway gestisce la classificazione dell'intento, l'estrazione ISIN e l'invocazione della pipeline.
- Valutazione quantitativa versionata nel report (routing, Mermaid, router LLM).
- La pipeline è pronta per integrazioni più profonde con dati reali e modelli LLM.
