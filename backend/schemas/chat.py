from typing import List, Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local-backend"
