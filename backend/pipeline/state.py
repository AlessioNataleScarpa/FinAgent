"""
State definition for the ETF Analysis Pipeline.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict


class TSFMForecast(TypedDict, total=False):
    """Portable output contract shared by local and remote TSFM backends."""

    model: str
    frequency: str
    horizon: int
    confidence_level: float
    mean: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    status: Literal["ok", "fallback", "unavailable"]
    error: str
    explanation: Dict[str, Any]


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
    historical_prices: Optional[List[float]]
    historical_dates: Optional[List[str]]
    tsfm_forecast: Optional[TSFMForecast]
    prediction_out2: Optional[str]
    agent_2_out_tech: Optional[str]
    composition_charts: Optional[str]
    timeline_charts: Optional[str]
    forecast_charts: Optional[str]
    xai_analysis: Optional[str]
    sentiment_charts: Optional[str]
    memory_saved: Optional[bool]

    # Output (both branches)
    out_finale: Optional[str]
