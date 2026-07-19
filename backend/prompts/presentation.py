def build_presentation_agent_prompt(isin: str, info_presentazione: str) -> str:
    return (
        f"You are Agent 1 (Presentation Agent), an AI assistant specialized in financial asset presentation for ETFs.\n"
        f"Your task is to analyze the ETF information for ISIN '{isin}' and structure a detailed presentation response.\n\n"
        f"ETF ISIN: {isin}\n"
        f"PRESENTATION INFO:\n{info_presentazione}\n\n"
        f"Output structured fields:\n"
        f"- summary: Overall summary of the ETF in Italian Markdown prose (no JSON).\n"
        f"- asset_allocation: Asset class breakdown description.\n"
        f"- sector_breakdown: List of sector exposures.\n"
        f"- regional_breakdown: List of geographic exposures.\n"
        f"- mermaid_chart: Valid Mermaid pie chart body only (start with 'pie title ...', no fences).\n"
        f"Do not wrap the whole answer as JSON in a code fence; use the structured schema fields.\n"
    )
