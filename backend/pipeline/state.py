"""
State definition for the ETF Analysis Pipeline.
"""

from typing import TypedDict, Optional


class PipelineState(TypedDict, total=False):
    isin: str
    clean_query: Optional[str]
    info_presentazione: Optional[str]
    agent_1_out1: Optional[str]
    news_data: Optional[str]
    info_storici: Optional[str]
    prediction_out2: Optional[str]
    agent_2_out_tech: Optional[str]
    out_finale: Optional[str]
