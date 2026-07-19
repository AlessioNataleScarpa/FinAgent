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

    def mock_invoke(messages, *args, **kwargs):
        content_text = ""
        if isinstance(messages, list):
            content_text = "\n".join([m.content if hasattr(m, "content") else str(m) for m in messages])
        elif isinstance(messages, str):
            content_text = messages
        else:
            content_text = str(messages)

        mock_res = MagicMock()
        mock_res.content = (
            f"# 🏆 REPORT COMPLETO DI ANALISI STRUMENTO FINANZIARIO\n\n"
            f"{content_text}\n\n---\n"
            f"*Report generato automaticamente dalla Pipeline LangGraph.*"
        )
        return mock_res

    async def mock_ainvoke(messages, *args, **kwargs):
        return mock_invoke(messages, *args, **kwargs)

    mock_instance.invoke.side_effect = mock_invoke
    mock_instance.ainvoke.side_effect = mock_ainvoke

    def mock_with_structured_output(schema):
        mock_struct = MagicMock()

        def _get_dummy_schema_instance():
            schema_name = getattr(schema, "__name__", "")
            if schema_name == "PresentationOutputSchema":
                from schemas.presentation import PresentationOutputSchema
                return PresentationOutputSchema(
                    summary="Test summary for IE00B4L5Y983 LU1681043599",
                    asset_allocation="Equity 100%",
                    sector_breakdown=["Tech"],
                    regional_breakdown=["US"],
                    mermaid_chart="pie title Chart",
                )
            if schema_name == "TechnicalNewsOutputSchema":
                from schemas.technical_news import TechnicalNewsOutputSchema
                return TechnicalNewsOutputSchema(
                    technical_summary="PREDICTION DATA PRED TEST Test tech for IE00B4L5Y983 LU1681043599",
                    news_impact_analysis="NEWS DATA NEWS TEST Test news impact",
                    sentiment_score="Bullish",
                    mermaid_chart="graph TD\n A-->B",
                )
            mock_res = MagicMock()
            mock_res.content = "# 🏆 REPORT COMPLETO DI ANALISI STRUMENTO FINANZIARIO"
            return mock_res

        dummy_val = _get_dummy_schema_instance()
        mock_struct.invoke.return_value = dummy_val

        async def _async_struct_invoke(*args, **kwargs):
            return dummy_val

        mock_struct.ainvoke = _async_struct_invoke
        return mock_struct

    mock_instance.with_structured_output.side_effect = mock_with_structured_output

    with patch("agents.base.ChatGoogleGenerativeAI", return_value=mock_instance) as mock_class_base, \
         patch("langchain_google_genai.ChatGoogleGenerativeAI", return_value=mock_instance):
        yield mock_class_base
