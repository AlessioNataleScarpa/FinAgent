import os
from abc import ABC, abstractmethod
import ast
import json
import re
from typing import Any, Callable, List, Optional

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

    def create_llm(self, system_prompt: Optional[str] = None) -> ChatGoogleGenerativeAI:
        kwargs: Dict[str, Any] = {
            "model": self.llm_model(),
            "google_api_key": self.llm_api_key(),
            "temperature": 0.0,
            "timeout": 45.0,
            "max_retries": 1,
        }
        if system_prompt:
            kwargs["model_kwargs"] = {"system_instruction": system_prompt}

        return ChatGoogleGenerativeAI(**kwargs)

    def create_structured_llm(
        self,
        schema: Any,
        system_prompt: Optional[str] = None,
        fallback_to_plain: bool = False,
    ):
        llm = self.create_llm(system_prompt=system_prompt)
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
    def parse_structured_output(response: Any, schema_cls: Any) -> Optional[Any]:
        if isinstance(response, schema_cls):
            return response
        if hasattr(response, "model_dump"):
            try:
                return schema_cls(**response.model_dump())
            except Exception:
                pass
        if isinstance(response, dict):
            try:
                return schema_cls(**response)
            except Exception:
                pass
        return None

    @staticmethod
    def strip_code_fences(content: str) -> str:
        normalized_content = content.strip()

        if "```json" in normalized_content:
            return normalized_content.split("```json", 1)[1].split("```", 1)[0].strip()

        if "```" in normalized_content:
            return normalized_content.split("```", 1)[1].split("```", 1)[0].strip()

        return normalized_content

    @staticmethod
    def _text_from_part(part: Any) -> str:
        if part is None:
            return ""
        if isinstance(part, str):
            return part
        if isinstance(part, dict):
            part_type = str(part.get("type") or part.get("kind") or "").lower()
            if part_type in {"thinking", "thought", "reasoning"}:
                return ""
            for key in ("text", "content", "output_text"):
                value = part.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return ""
        for attr in ("text", "content"):
            value = getattr(part, attr, None)
            if isinstance(value, str) and value.strip():
                part_type = str(getattr(part, "type", "") or "").lower()
                if part_type in {"thinking", "thought", "reasoning"}:
                    return ""
                return value
        return ""

    @classmethod
    def normalize_llm_content(cls, content: Any) -> str:
        """
        Normalize Gemini/Gemma responses to clean Markdown text.
        Drops thinking/reasoning blocks that some models return as structured parts.
        """
        if content is None:
            return ""

        if isinstance(content, list):
            texts = [cls._text_from_part(part) for part in content]
            joined = "\n\n".join(t.strip() for t in texts if t and t.strip())
            return joined.strip()

        if not isinstance(content, str):
            if hasattr(content, "content"):
                return cls.normalize_llm_content(getattr(content, "content"))
            content = str(content)

        text = content.strip()
        if not text:
            return ""

        # Model sometimes stringifies a list of {type: thinking|text, ...}
        if text.startswith("[{") and ("'type'" in text or '"type"' in text):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return cls.normalize_llm_content(parsed)
            except (SyntaxError, ValueError, MemoryError):
                pass

        # Fallback regex extraction for stringified thinking payloads
        if "'type': 'thinking'" in text or '"type": "thinking"' in text:
            matches = re.findall(
                r"['\"]type['\"]\s*:\s*['\"]text['\"]\s*,\s*['\"]text['\"]\s*:\s*['\"](.*?)['\"]\s*\}",
                text,
                flags=re.DOTALL,
            )
            if matches:
                recovered = matches[-1]
                recovered = (
                    recovered.replace("\\n", "\n")
                    .replace("\\'", "'")
                    .replace('\\"', '"')
                )
                return recovered.strip()

        return text

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
