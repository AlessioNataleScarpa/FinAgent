from pydantic import BaseModel, Field


class TechnicalNewsOutputSchema(BaseModel):
    technical_summary: str = Field(..., description="Summary of forecast/predict model output")
    news_impact_analysis: str = Field(..., description="Analysis of recent news impact on forecast")
    sentiment_score: str = Field(..., description="Market sentiment e.g. Bullish, Bearish, Neutral")
    mermaid_chart: str = Field(
        ...,
        description="Valid Mermaid flowchart/trend graph string representing prediction trend and news impact",
    )



