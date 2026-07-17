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


class PresentationAgentSchema(BaseModel):
    summary: str = Field(..., description="Overview of ETF object, provider, index tracked")
    sector_allocation_desc: str = Field(..., description="Sector allocation description")
    regional_allocation_desc: str = Field(..., description="Regional allocation description")
    mermaid_pie_chart: str = Field(..., description="Valid Mermaid pie chart string")
    mermaid_flowchart: str = Field(..., description="Valid Mermaid flowchart string")

