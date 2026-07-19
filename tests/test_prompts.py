"""
Unit tests for prompt builder functions in backend/prompts.
"""

from prompts import (
    build_gateway_system_prompt,
    build_presentation_agent_prompt,
    build_technical_news_agent_prompt,
)


def test_build_gateway_system_prompt():
    prompt = build_gateway_system_prompt("{}")
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_presentation_agent_prompt():
    isin = "IE00B4L5Y983"
    info = "iShares Core MSCI World ETF USD (Acc)"
    prompt = build_presentation_agent_prompt(isin, info)

    assert isin in prompt
    assert info in prompt
    assert "Agent 1" in prompt
    assert "pie title" in prompt


def test_build_technical_news_agent_prompt():
    isin = "IE00B4L5Y983"
    pred = "BULLISH +8.5%"
    news = "Federal Reserve signals potential rate cuts."
    prompt = build_technical_news_agent_prompt(isin, pred, news)

    assert isin in prompt
    assert pred in prompt
    assert news in prompt
    assert "Agent 2" in prompt
    assert "graph LR" in prompt


