import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from agents.base import BaseAgent
from agents.gatewayAgent import GatewayAgent
from agents.joinAgent import JoinAgent
from agents.presentationAgent import PresentationAgent
from agents.registry import AVAILABLE_AGENTS
from agents.technicalNewsAgent import TechnicalNewsAgent
from schemas.chat import Message
from schemas.presentation import PresentationOutputSchema
from schemas.routing import RouterIntentSchema
from schemas.technical_news import TechnicalNewsOutputSchema


class DummyAgent(BaseAgent):
    @property
    def model_id(self) -> str:
        return "Dummy Model"

    async def run(self, messages: list[Message]) -> str:
        return "dummy response"


class TestBaseAgent:
    def test_env_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            assert BaseAgent.llm_model() == "gemma-4-31b-it"
            assert BaseAgent.llm_api_key() is None

    def test_env_overrides(self):
        test_env = {
            "GEMINI_MODEL": "custom-gemini-model",
            "GOOGLE_API_KEY": "custom-google-key",
        }
        with patch.dict(os.environ, test_env, clear=True):
            assert BaseAgent.llm_model() == "custom-gemini-model"
            assert BaseAgent.llm_api_key() == "custom-google-key"

    @patch("agents.base.ChatGoogleGenerativeAI")
    def test_create_llm(self, mock_chat_google):
        test_env = {
            "GEMINI_MODEL": "custom-gemini-model",
            "GOOGLE_API_KEY": "custom-google-key",
        }
        with patch.dict(os.environ, test_env, clear=True):
            agent = DummyAgent()
            llm = agent.create_llm()
            mock_chat_google.assert_called_once_with(
                model="custom-gemini-model",
                google_api_key="custom-google-key",
                temperature=0.0,
            )
            assert llm == mock_chat_google.return_value

    @patch("agents.base.ChatGoogleGenerativeAI")
    def test_create_structured_llm_success(self, mock_chat_google):
        mock_llm_instance = MagicMock()
        mock_chat_google.return_value = mock_llm_instance
        mock_structured = MagicMock()
        mock_llm_instance.with_structured_output.return_value = mock_structured

        agent = DummyAgent()
        result = agent.create_structured_llm(RouterIntentSchema)
        mock_llm_instance.with_structured_output.assert_called_once_with(RouterIntentSchema)
        assert result == mock_structured

    @patch("agents.base.ChatGoogleGenerativeAI")
    def test_create_structured_llm_fallback(self, mock_chat_google):
        mock_llm_instance = MagicMock()
        mock_chat_google.return_value = mock_llm_instance
        mock_llm_instance.with_structured_output.side_effect = Exception("Not supported")

        agent = DummyAgent()

        # Without fallback -> exception is raised
        with pytest.raises(Exception):
            agent.create_structured_llm(RouterIntentSchema, fallback_to_plain=False)

        # With fallback -> returns raw llm
        result = agent.create_structured_llm(RouterIntentSchema, fallback_to_plain=True)
        assert result == mock_llm_instance

    def test_extract_latest_user_message(self):
        messages = [
            Message(role="system", content="System prompt"),
            Message(role="user", content="First user prompt"),
            Message(role="assistant", content="Assistant reply"),
            Message(role="user", content="Latest user prompt"),
        ]
        assert BaseAgent.extract_latest_user_message(messages) == "Latest user prompt"

        no_user_msgs = [Message(role="system", content="System prompt")]
        assert BaseAgent.extract_latest_user_message(no_user_msgs) == ""

    def test_strip_code_fences(self):
        json_fenced = "```json\n{\"key\": \"val\"}\n```"
        assert BaseAgent.strip_code_fences(json_fenced) == "{\"key\": \"val\"}"

        plain_fenced = "```\nhello world\n```"
        assert BaseAgent.strip_code_fences(plain_fenced) == "hello world"

        unfenced = "hello world"
        assert BaseAgent.strip_code_fences(unfenced) == "hello world"

    def test_extract_previous_assistant_state(self):
        def extractor(content: str):
            try:
                data = json.loads(content)
                if data.get("status") == "accepted":
                    return data
            except Exception:
                pass
            return None

        messages = [
            Message(role="user", content="Query 1"),
            Message(role="assistant", content=json.dumps({"status": "accepted", "isin": "IE00B4L5Y983"})),
            Message(role="user", content="Query 2"),
        ]

        extracted = BaseAgent.extract_previous_assistant_state(messages, extractor)
        assert extracted is not None
        parsed = json.loads(extracted)
        assert parsed["isin"] == "IE00B4L5Y983"

        no_state_messages = [Message(role="assistant", content="Not JSON")]
        assert BaseAgent.extract_previous_assistant_state(no_state_messages, extractor) is None


class TestGatewayAgent:
    @patch("agents.base.ChatGoogleGenerativeAI")
    def test_gateway_agent_initialization(self, mock_chat_google):
        agent = GatewayAgent()
        assert agent.model_id == "Gateway Agent"
        mock_chat_google.assert_called()

    def test_non_routable_with_direct_response(self):
        agent = GatewayAgent()
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke.return_value = RouterIntentSchema(
            intent="chit_chat",
            is_routable=False,
            direct_response="Ciao! Come posso aiutarti oggi?",
        )
        agent.structured_llm = mock_structured_llm

        messages = [Message(role="user", content="Ciao!")]
        result = asyncio.run(agent.run(messages))
        assert result == "Ciao! Come posso aiutarti oggi?"

    def test_non_routable_without_direct_response(self):
        agent = GatewayAgent()
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke.return_value = RouterIntentSchema(
            intent="out_of_domain",
            is_routable=False,
            direct_response=None,
        )
        agent.structured_llm = mock_structured_llm

        messages = [Message(role="user", content="Che tempo fa a Roma?")]
        result = asyncio.run(agent.run(messages))
        assert "Posso aiutarti solo con richieste relative a ETF e codici ISIN" in result

    def test_routable_etf_with_isin_in_llm_response(self):
        agent = GatewayAgent()
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke.return_value = RouterIntentSchema(
            intent="etf",
            is_routable=True,
            isin="IE00B4L5Y983",
            clean_query="Analisi iShares Core MSCI World",
        )
        agent.structured_llm = mock_structured_llm

        messages = [Message(role="user", content="Analizza l'ETF IE00B4L5Y983")]
        result = asyncio.run(agent.run(messages))
        assert "IE00B4L5Y983" in result
        assert "REPORT COMPLETO" in result

    def test_routable_etf_isin_regex_fallback(self):
        agent = GatewayAgent()
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke.return_value = RouterIntentSchema(
            intent="etf",
            is_routable=True,
            isin=None,
            clean_query="Informazioni su IE00B4L5Y983",
        )
        agent.structured_llm = mock_structured_llm

        messages = [Message(role="user", content="Mostrami i dati per IE00B4L5Y983 per favore")]
        result = asyncio.run(agent.run(messages))
        assert "IE00B4L5Y983" in result
        assert "REPORT COMPLETO" in result

    def test_routable_etf_isin_previous_state_fallback(self):
        agent = GatewayAgent()
        mock_structured_llm = AsyncMock()
        mock_structured_llm.ainvoke.return_value = RouterIntentSchema(
            intent="etf",
            is_routable=True,
            isin=None,
            clean_query="Quali sono i costi?",
        )
        agent.structured_llm = mock_structured_llm

        previous_assistant_msg = json.dumps({
            "status": "accepted",
            "intent": "etf",
            "isin": "LU1681043599",
            "clean_query": "ETF Amundi MSCI World",
        })

        messages = [
            Message(role="user", content="Parlami di LU1681043599"),
            Message(role="assistant", content=previous_assistant_msg),
            Message(role="user", content="Quali sono i costi?"),
        ]

        result = asyncio.run(agent.run(messages))
        assert "LU1681043599" in result
        assert "REPORT COMPLETO" in result

    def test_invalid_llm_string_response(self):
        agent = GatewayAgent()
        mock_structured_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Invalid non-JSON output"
        mock_structured_llm.ainvoke.return_value = mock_response
        agent.structured_llm = mock_structured_llm

        messages = [Message(role="user", content="Test prompt")]
        result = asyncio.run(agent.run(messages))
        assert "Errore interno del gateway" in result


class TestPresentationAgent:
    def test_presentation_agent_fallback(self):
        agent = PresentationAgent()
        with patch.object(agent, "create_structured_llm", side_effect=Exception("LLM unavailable")):
            out = agent.run(isin="IE00B4L5Y983", info_presentazione="Info ETF Test")
            assert "IE00B4L5Y983" in out
            assert "```mermaid" in out
            assert "Allocazione Settoriale" in out

    def test_presentation_agent_success(self):
        agent = PresentationAgent()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = PresentationOutputSchema(
            summary="ETF Summary Test",
            asset_allocation="Equity 100%",
            sector_breakdown=["Tech 30%"],
            regional_breakdown=["USA 80%"],
            mermaid_chart="pie title Settori\n    \"Tech\" : 30",
        )

        with patch.object(agent, "create_structured_llm", return_value=mock_structured):
            out = agent.run(isin="IE00B4L5Y983", info_presentazione="Info ETF Test")
            assert "ETF Summary Test" in out
            assert "Tech 30%" in out
            assert "USA 80%" in out
            assert "```mermaid" in out


class TestTechnicalNewsAgent:
    def test_technical_news_agent_fallback(self):
        agent = TechnicalNewsAgent()
        with patch.object(agent, "create_structured_llm", side_effect=Exception("LLM unavailable")):
            out = agent.run(isin="IE00B4L5Y983", prediction="PRED TEST", news="NEWS TEST")
            assert "IE00B4L5Y983" in out
            assert "PRED TEST" in out
            assert "NEWS TEST" in out
            assert "```mermaid" in out

    def test_technical_news_agent_success(self):
        agent = TechnicalNewsAgent()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = TechnicalNewsOutputSchema(
            technical_summary="Quantitative prediction +10%",
            news_impact_analysis="Positive Fed news",
            sentiment_score="Bullish",
            mermaid_chart="graph LR\n    A -> B",
        )

        with patch.object(agent, "create_structured_llm", return_value=mock_structured):
            out = agent.run(isin="IE00B4L5Y983", prediction="PRED TEST", news="NEWS TEST")
            assert "Quantitative prediction +10%" in out
            assert "Positive Fed news" in out
            assert "Bullish" in out
            assert "```mermaid" in out


class TestNewAgentMethodsAndJoinAgent:
    @pytest.mark.asyncio
    async def test_run_presentation_async(self):
        agent = PresentationAgent()
        mock_structured = AsyncMock()
        mock_structured.ainvoke.return_value = PresentationOutputSchema(
            summary="Async Presentation Summary",
            asset_allocation="Equity 100%",
            sector_breakdown=["Tech", "Healthcare"],
            regional_breakdown=["USA", "Europe"],
            mermaid_chart="pie title Assets\n \"Tech\" : 50",
        )
        with patch.object(agent, "create_structured_llm", return_value=mock_structured):
            result = await agent.run_presentation("Test info")
            assert isinstance(result, PresentationOutputSchema)
            assert result.summary == "Async Presentation Summary"

    @pytest.mark.asyncio
    async def test_run_technical_news_async(self):
        agent = TechnicalNewsAgent()
        mock_structured = AsyncMock()
        mock_structured.ainvoke.return_value = TechnicalNewsOutputSchema(
            technical_summary="Async Tech Summary",
            news_impact_analysis="Positive news impact",
            sentiment_score="Bullish",
            mermaid_chart="graph LR\n A -> B",
        )
        with patch.object(agent, "create_structured_llm", return_value=mock_structured):
            result = await agent.run_technical_news("News data", "Prediction data")
            assert isinstance(result, TechnicalNewsOutputSchema)
            assert result.technical_summary == "Async Tech Summary"

    @pytest.mark.asyncio
    async def test_join_agent_run_join_async(self):
        agent = JoinAgent()
        assert agent.model_id == "Join Agent"
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value.content = "# FINAL SYNTHESIS REPORT"
        with patch.object(agent, "create_llm", return_value=mock_llm):
            res = await agent.run_join("Out 1 text", "Out tech text")
            assert res == "# FINAL SYNTHESIS REPORT"

    @pytest.mark.asyncio
    async def test_join_agent_run_fallback(self):
        agent = JoinAgent()
        mock_llm = AsyncMock()
        mock_llm.ainvoke.side_effect = Exception("LLM connection failed")
        with patch.object(agent, "create_llm", return_value=mock_llm):
            res = await agent.run_join("Out 1 text", "Out tech text")
            assert "REPORT COMPLETO DI ANALISI STRUMENTO FINANZIARIO" in res
            assert "Out 1 text" in res
            assert "Out tech text" in res

    def test_registry_has_all_agents(self):
        assert "gatewayAgent" in AVAILABLE_AGENTS
        assert "presentationAgent" in AVAILABLE_AGENTS
        assert "technicalNewsAgent" in AVAILABLE_AGENTS
        assert "joinAgent" in AVAILABLE_AGENTS
        assert isinstance(AVAILABLE_AGENTS["joinAgent"], JoinAgent)



