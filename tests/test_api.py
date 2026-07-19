from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from main import app
from agents.registry import AVAILABLE_AGENTS

client = TestClient(app)


class TestAPIRoutes:
    def test_get_models(self):
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)

        model_ids = [m["id"] for m in data["data"]]
        assert "gatewayAgent" in model_ids
        assert "conversationAgent" in model_ids

    @patch.dict(AVAILABLE_AGENTS)
    def test_chat_completions_success(self):
        mock_agent = AsyncMock()
        mock_agent.run.return_value = "Mocked agent response for ETF request"
        AVAILABLE_AGENTS["gatewayAgent"] = mock_agent

        payload = {
            "model": "gatewayAgent",
            "messages": [
                {"role": "user", "content": "Analizza ETF IE00B4L5Y983"}
            ],
        }

        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == "chatcmpl-gatewayAgent"
        assert data["object"] == "chat.completion"
        assert data["model"] == "gatewayAgent"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "Mocked agent response for ETF request"
        assert data["choices"][0]["finish_reason"] == "stop"

        mock_agent.run.assert_called_once()

    def test_chat_completions_model_not_found(self):
        payload = {
            "model": "unknown-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
        }

        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 404
        assert response.json()["detail"] == "Model not found in registry"

    def test_chat_completions_invalid_payload(self):
        payload = {
            "model": "gatewayAgent",
            # missing messages field
        }

        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 422
