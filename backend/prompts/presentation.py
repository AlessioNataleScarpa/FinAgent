def build_presentation_agent_prompt(isin: str, info_presentazione: str) -> str:
    return (
        f"You are Agent 1 (Presentation Agent), an AI assistant specialized in financial asset presentation for ETFs.\n"
        f"Your task is to analyze the ETF information for ISIN '{isin}' and structure a detailed presentation response.\n\n"
        f"ETF ISIN: {isin}\n"
        f"PRESENTATION INFO:\n{info_presentazione}\n\n"
        f"Output structured fields:\n"
        f"- summary: Overall summary of the ETF.\n"
        f"- sector_allocation_desc: Sector allocation description.\n"
        f"- regional_allocation_desc: Regional allocation description.\n"
        f"- mermaid_pie_chart: Valid Mermaid pie chart code (must include 'pie title').\n"
        f"- mermaid_flowchart: Valid Mermaid flowchart diagram (must include 'graph TD').\n"
    )
