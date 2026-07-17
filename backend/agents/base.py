import os
from abc import ABC, abstractmethod
import json
from typing import Callable, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from schemas.chat import Message
except ImportError:
    from backend.schemas.chat import Message


class AwaitableString(str):
    """String subclass that can be awaited in async contexts for API route compatibility."""

    def __await__(self):
        async def _coro():
            return str(self)

        return _coro().__await__()


class BaseAgent(ABC):
    @staticmethod
    def llm_model() -> str:
        return os.getenv("GEMINI_MODEL", "gemma-4-31b-it")

    @staticmethod
    def llm_api_key() -> Optional[str]:
        return os.getenv("GOOGLE_API_KEY")

    def create_llm(self) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self.llm_model(),
            google_api_key=self.llm_api_key(),
            temperature=0.0,
        )

    def create_structured_llm(self, schema, fallback_to_plain: bool = False):
        llm = self.create_llm()
        if not fallback_to_plain:
            return llm.with_structured_output(schema)

        try:
            return llm.with_structured_output(schema)
        except Exception:
            return llm

    @staticmethod
    def extract_latest_user_message(messages: List[Message]) -> str:
        return next(
            (message.content for message in reversed(messages) if message.role == "user"),
            "",
        )

    @staticmethod
    def strip_code_fences(content: str) -> str:
        normalized_content = content.strip()

        if "```json" in normalized_content:
            return normalized_content.split("```json", 1)[1].split("```", 1)[0].strip()

        if "```" in normalized_content:
            return normalized_content.split("```", 1)[1].split("```", 1)[0].strip()

        return normalized_content

    @staticmethod
    def extract_previous_assistant_state(
        messages: List[Message],
        extractor: Callable[[str], Optional[dict]],
    ) -> Optional[str]:
        for message in reversed(messages):
            if message.role != "assistant":
                continue
            previous_state = extractor(message.content)
            if previous_state is not None:
                return json.dumps(previous_state, indent=2, ensure_ascii=False)
        return None

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass

    @abstractmethod
    async def run(self, messages: List[Message]) -> str:
        pass
