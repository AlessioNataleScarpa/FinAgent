"""
Unit tests for prompt builder functions in backend/prompts.
"""

from prompts import (
    build_gateway_system_prompt,
    build_presentation_agent_prompt,
    build_technical_news_agent_prompt,
    build_agent_1_system_prompt,
    build_agent_2_system_prompt,
    build_join_system_prompt,
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
    assert "graph TD" in prompt


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
    assert "gantt" in prompt


def test_build_agent_1_system_prompt():
    info = "iShares Core MSCI World ETF USD (Acc)"
    prompt = build_agent_1_system_prompt(info)

    assert isinstance(prompt, str)
    assert info in prompt
    assert "Agente 1" in prompt
    assert "Mermaid" in prompt


def test_build_agent_2_system_prompt():
    news = "Federal Reserve signals potential rate cuts."
    pred = "BULLISH +8.5%"
    prompt = build_agent_2_system_prompt(news_data=news, prediction_data=pred)

    assert isinstance(prompt, str)
    assert news in prompt
    assert pred in prompt
    assert "Agente 2" in prompt
    assert "Mermaid" in prompt


def test_build_join_system_prompt():
    out1 = "Presentation content for ETF"
    out_tech = "Technical and news analysis content"
    prompt = build_join_system_prompt(out1=out1, out_tech=out_tech)

    assert isinstance(prompt, str)
    assert out1 in prompt
    assert out_tech in prompt
    assert "Join Presenter Agent" in prompt
    assert "Markdown" in prompt
