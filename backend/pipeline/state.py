"""
State definition for the ETF Analysis Pipeline.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict


class PipelineState(TypedDict, total=False):
    # Routing
    mode: Literal["full_analysis", "conversation"]
    user_message: Optional[str]
    chat_messages: Optional[List[Dict[str, Any]]]

    # Shared
    isin: str
    clean_query: Optional[str]

    # Full analysis branch
    info_presentazione: Optional[str]
    agent_1_out1: Optional[str]
    news_data: Optional[str]
    info_storici: Optional[str]
    prediction_out2: Optional[str]
    agent_2_out_tech: Optional[str]
    composition_charts: Optional[str]
    timeline_charts: Optional[str]
    sentiment_charts: Optional[str]
    memory_saved: Optional[bool]

    # Output (both branches)
    out_finale: Optional[str]
