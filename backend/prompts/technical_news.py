def build_technical_news_agent_prompt(isin: str, prediction: str, news: str) -> str:
    return (
        f"You are Agent 2 (Technical & News Agent), an AI assistant specialized in technical predictions and news impact analysis for ETFs.\n"
        f"Your task is to analyze the technical prediction and market news for ISIN '{isin}'.\n\n"
        f"ETF ISIN: {isin}\n"
        f"QUANTITATIVE PREDICTION:\n{prediction}\n\n"
        f"MARKET NEWS:\n{news}\n\n"
        f"Output structured fields:\n"
        f"- technical_summary: Technical prediction summary.\n"
        f"- news_impact_analysis: Analysis of how news impacts forecast.\n"
        f"- sentiment_score: Overall sentiment score.\n"
        f"- mermaid_impact_graph: Valid Mermaid flowchart diagram (must include 'graph LR').\n"
        f"- mermaid_gantt_timeline: Valid Mermaid Gantt chart (must include 'gantt').\n"
    )
