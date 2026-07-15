from typing import Literal, Optional

from pydantic import BaseModel, Field


class RouterIntentSchema(BaseModel):
    intent: Literal["etf", "chit_chat", "out_of_domain"]
    is_routable: bool
    direct_response: Optional[str] = None
    clean_query: Optional[str] = None
    isin: Optional[str] = Field(
        default=None,
        description="ISIN code if present in the user message (12 chars: 2 letters + 9 alphanumerics + check digit).",
    )
