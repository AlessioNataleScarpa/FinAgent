"""Join / synthesis prompts for the final Markdown report."""


def build_join_system_prompt() -> str:
    return (
        "Sei JoinAgent, l'agente di sintesi finale di un sistema di analisi ETF.\n"
        "Devi produrre UN UNICO REPORT in Markdown italiano, chiaro e leggibile.\n\n"
        "REGOLE OBBLIGATORIE:\n"
        "1. NON restituire mai JSON, YAML o oggetti strutturati.\n"
        "2. Usa titoli Markdown (# ## ###), elenchi e grassetti.\n"
        "3. Includi i blocchi Mermaid forniti COSI' COME SONO, dentro fence "
        "```mermaid ... ``` (OpenWebUI li renderizza automaticamente).\n"
        "4. Non riscrivere la sintassi Mermaid: copiala verbatim.\n"
        "5. Struttura consigliata:\n"
        "   - Titolo del report + ISIN\n"
        "   - Executive summary (5-8 righe)\n"
        "   - Presentazione strumento\n"
        "   - Composizione (grafici a torta)\n"
        "   - Andamento temporale (grafico xy)\n"
        "   - Analisi tecnica e news\n"
        "   - Sentiment\n"
        "   - Conclusioni operative\n"
        "6. Tono professionale, concreto, senza emoji eccessive.\n"
    )


def build_join_human_prompt(
    *,
    isin: str,
    out1: str,
    out_tech: str,
    composition_charts: str = "",
    timeline_charts: str = "",
    sentiment_charts: str = "",
) -> str:
    return (
        f"ISIN: {isin or 'N/D'}\n\n"
        f"=== OUT 1 PRESENTAZIONE ===\n{out1}\n\n"
        f"=== OUT TECNICA / NEWS ===\n{out_tech}\n\n"
        f"=== GRAFICI COMPOSIZIONE (Mermaid pronti) ===\n{composition_charts}\n\n"
        f"=== GRAFICI ANDAMENTO TEMPORALE (Mermaid pronti) ===\n{timeline_charts}\n\n"
        f"=== GRAFICI SENTIMENT (Mermaid pronti) ===\n{sentiment_charts}\n\n"
        "Genera ora il REPORT FINALE in Markdown."
    )
