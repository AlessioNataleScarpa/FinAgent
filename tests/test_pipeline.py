"""
Unit and integration tests for the ETF Analysis Pipeline in backend/pipeline.
Tests individual pipeline nodes, compiled StateGraph execution, and GatewayAgent end-to-end integration.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from pipeline import graph, app, PipelineState
from pipeline.info_presentazione import fetch_info_presentazione
from pipeline.agent_1 import generate_agent_1_out
from pipeline.news import fetch_news
from pipeline.info_andamenti_storici import fetch_info_andamenti_storici
from pipeline.predict import predict_node
from pipeline.agent_2 import generate_agent_2_out
from pipeline.join_presenter import join_presenter_node
from agents.gatewayAgent import GatewayAgent
from schemas.chat import Message
from schemas.routing import RouterIntentSchema


class TestPipelineNodes:
    """Tests each pipeline node individually."""

    def test_info_presentazione_node(self):
        state: PipelineState = {"isin": "IE00B4L5Y983"}
        res = fetch_info_presentazione(state)
        assert "info_presentazione" in res
        info = res["info_presentazione"]
        assert "IE00B4L5Y983" in info
        assert "iShares Core MSCI World" in info
        assert "TER:" in info

    def test_info_presentazione_node_empty_isin(self):
        state: PipelineState = {"isin": ""}
        res = fetch_info_presentazione(state)
        assert "info_presentazione" in res
        info = res["info_presentazione"]
        assert "IE00B4L5Y983" in info

    def test_agent_1_node(self):
        state: PipelineState = {
            "isin": "IE00B4L5Y983",
            "info_presentazione": "Sample presentation metadata",
        }
        res = generate_agent_1_out(state)
        assert "agent_1_out1" in res
        out1 = res["agent_1_out1"]
        assert "IE00B4L5Y983" in out1
        assert "```mermaid" in out1
        assert "Allocazione Settoriale" in out1

    def test_news_node(self):
        state: PipelineState = {"isin": "IE00B4L5Y983"}
        res = fetch_news(state)
        assert "news_data" in res
        news = res["news_data"]
        assert "IE00B4L5Y983" in news
        assert "MACRO" in news

    @patch("httpx.Client")
    def test_info_andamenti_storici_node_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        mock_client_cls.return_value.__enter__.return_value = mock_client

        state: PipelineState = {"isin": "IE00B4L5Y983"}
        res = fetch_info_andamenti_storici(state)
        assert "info_storici" in res
        info_storici = res["info_storici"]
        assert "Data retrieved from https://londonstrategicedge.com/: HTTP 200 OK" in info_storici
        assert "1-Year Return" in info_storici

    @patch("httpx.Client")
    def test_info_andamenti_storici_node_fallback(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.get.side_effect = Exception("Connection error")
        mock_client_cls.return_value.__enter__.return_value = mock_client

        state: PipelineState = {"isin": "IE00B4L5Y983"}
        res = fetch_info_andamenti_storici(state)
        assert "info_storici" in res
        info_storici = res["info_storici"]
        assert "Fallback Active" in info_storici
        assert "Annualized Volatility" in info_storici

    def test_predict_node(self):
        state: PipelineState = {
            "isin": "IE00B4L5Y983",
            "info_storici": "1-Year Return: +14.2%",
        }
        res = predict_node(state)
        assert "prediction_out2" in res
        pred = res["prediction_out2"]
        assert "IE00B4L5Y983" in pred
        assert "BULLISH" in pred

    def test_agent_2_node(self):
        state: PipelineState = {
            "isin": "IE00B4L5Y983",
            "prediction_out2": "PREDICTION DATA",
            "news_data": "NEWS DATA",
        }
        res = generate_agent_2_out(state)
        assert "agent_2_out_tech" in res
        out_tech = res["agent_2_out_tech"]
        assert "PREDICTION DATA" in out_tech
        assert "NEWS DATA" in out_tech
        assert "```mermaid" in out_tech

    def test_join_presenter_node(self):
        state: PipelineState = {
            "isin": "IE00B4L5Y983",
            "agent_1_out1": "PRESENTATION OUT 1",
            "agent_2_out_tech": "TECHNICAL OUT 2",
        }
        res = join_presenter_node(state)
        assert "out_finale" in res
        out_finale = res["out_finale"]
        assert "REPORT COMPLETO" in out_finale
        assert "PRESENTATION OUT 1" in out_finale
        assert "TECHNICAL OUT 2" in out_finale


class TestStateGraphExecution:
    """Tests execution of the compiled StateGraph."""

    def test_stategraph_sync_execution(self):
        initial_state: PipelineState = {
            "isin": "IE00B4L5Y983",
            "clean_query": "Analizza ETF IE00B4L5Y983",
        }

        result = graph.invoke(initial_state)

        assert "info_presentazione" in result
        assert "agent_1_out1" in result
        assert "news_data" in result
        assert "info_storici" in result
        assert "prediction_out2" in result
        assert "agent_2_out_tech" in result
        assert "out_finale" in result

        assert "IE00B4L5Y983" in result["out_finale"]
        assert "REPORT COMPLETO DI ANALISI STRUMENTO FINANZIARIO" in result["out_finale"]

    @pytest.mark.asyncio
    async def test_stategraph_async_execution(self):
        initial_state: PipelineState = {
            "isin": "LU1681043599",
            "clean_query": "Analizza ETF LU1681043599",
        }

        result = await app.ainvoke(initial_state)

        assert "out_finale" in result
        assert "LU1681043599" in result["out_finale"]


class TestGatewayAgentPipelineIntegration:
    """Tests GatewayAgent end-to-end integration with the pipeline."""

    @pytest.mark.asyncio
    async def test_gateway_agent_runs_pipeline_for_routable_etf(self):
        agent = GatewayAgent()
        mock_structured_llm = MagicMock()

        async def mock_ainvoke(messages):
            return RouterIntentSchema(
                intent="etf",
                is_routable=True,
                isin="IE00B4L5Y983",
                clean_query="Analizza ETF IE00B4L5Y983",
            )

        mock_structured_llm.ainvoke = mock_ainvoke
        agent.structured_llm = mock_structured_llm

        messages = [Message(role="user", content="Analizza ETF IE00B4L5Y983")]
        result = await agent.run(messages)

        assert "REPORT COMPLETO DI ANALISI STRUMENTO FINANZIARIO" in result
        assert "IE00B4L5Y983" in result
        assert "```mermaid" in result

    @pytest.mark.asyncio
    async def test_gateway_agent_pipeline_fallback_on_missing_out_finale(self):
        agent = GatewayAgent()
        mock_structured_llm = MagicMock()

        async def mock_ainvoke(messages):
            return RouterIntentSchema(
                intent="etf",
                is_routable=True,
                isin="IE00B4L5Y983",
                clean_query="Analizza ETF IE00B4L5Y983",
            )

        mock_structured_llm.ainvoke = mock_ainvoke
        agent.structured_llm = mock_structured_llm

        with patch.object(graph, "ainvoke", new_callable=AsyncMock, return_value={}):
            messages = [Message(role="user", content="Analizza ETF IE00B4L5Y983")]
            result = await agent.run(messages)
            parsed = json.loads(result)
            assert parsed["status"] == "accepted"
            assert parsed["isin"] == "IE00B4L5Y983"
