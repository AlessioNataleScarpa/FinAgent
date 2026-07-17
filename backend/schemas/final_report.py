from pydantic import BaseModel, Field


class FinalReportSchema(BaseModel):
    title: str = Field(..., description="Report title")
    presentation_section: str = Field(..., description="Presentation section content")
    technical_news_section: str = Field(..., description="Technical and news section content")
    executive_summary: str = Field(..., description="Executive summary")
