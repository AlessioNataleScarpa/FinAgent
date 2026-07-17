from typing import List

from pydantic import BaseModel, Field


class PresentationOutputSchema(BaseModel):
    summary: str = Field(..., description="Overview of ETF object, provider, index tracked")
    asset_allocation: str = Field(..., description="Asset class breakdown")
    sector_breakdown: List[str] = Field(..., description="Sector exposure list")
    regional_breakdown: List[str] = Field(..., description="Geographic exposure list")
    mermaid_chart: str = Field(
        ...,
        description="Valid Mermaid pie chart / graph string representing asset & sector allocation",
    )



