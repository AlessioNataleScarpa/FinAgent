import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Mock langchain_google_genai module before any agent modules are imported
if "langchain_google_genai" not in sys.modules:
    mock_google_genai = MagicMock()
    mock_chat_google = MagicMock()
    mock_google_genai.ChatGoogleGenerativeAI = mock_chat_google
    sys.modules["langchain_google_genai"] = mock_google_genai

# Ensure backend directory is in Python path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))


@pytest.fixture(autouse=True)
def mock_google_ai():
    """Mock ChatGoogleGenerativeAI so tests run fast without external API calls or real GOOGLE_API_KEY."""
    mock_instance = MagicMock()
    with patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_instance) as mock_class:
        yield mock_class
