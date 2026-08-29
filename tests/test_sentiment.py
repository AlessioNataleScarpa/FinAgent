"""
Unit tests for the Neural Sentiment Analysis module and SentimentChartAgent.
Compatible with both pytest and python -m unittest.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

os.environ.setdefault("GOOGLE_API_KEY", "mock_google_key_for_testing")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.models.sentiment import (
    BaseSentimentStrategy,
    HeadlineSentiment,
    LexicalFallbackSentimentStrategy,
    SentimentAnalysisService,
    SentimentResult,
    analyze_sentiment,
)
from backend.agents.sentimentChartAgent import SentimentChartAgent


class TestSentimentModels(unittest.TestCase):
    def test_headline_sentiment_schema(self):
        hs = HeadlineSentiment(
            title="ETF reaches record high inflows",
            publisher="Bloomberg",
            url="https://example.com/news1",
            positive=0.85,
            neutral=0.10,
            negative=0.05,
            polarity=0.80,
            confidence=0.85,
            label="Positivo",
        )
        self.assertEqual(hs.label, "Positivo")
        self.assertEqual(hs.positive, 0.85)
        self.assertEqual(hs.polarity, 0.80)

    def test_sentiment_result_schema(self):
        sr = SentimentResult(
            positive_score=0.70,
            neutral_score=0.20,
            negative_score=0.10,
            polarity_index=0.60,
            confidence=0.70,
            label="Positivo",
            method="finbert_neural",
            headline_breakdown=[],
            summary_text="Sentiment Positivo",
        )
        self.assertEqual(sr.label, "Positivo")
        self.assertEqual(sr.polarity_index, 0.60)
        self.assertEqual(sr.method, "finbert_neural")


class TestSentimentService(unittest.TestCase):
    def test_empty_news_input(self):
        result = analyze_sentiment("")
        self.assertEqual(result.label, "Neutro")
        self.assertEqual(result.polarity_index, 0.0)
        self.assertEqual(result.headline_breakdown, [])

    def test_json_articles_input(self):
        news_payload = json.dumps({
            "source": "Yahoo Finance",
            "articles": [
                {
                    "title": "Strong quarterly growth for semiconductor ETF",
                    "summary": "Record earnings and high investor inflows beat consensus expectations.",
                    "publisher": "Reuters",
                    "url": "https://example.com/art1",
                },
                {
                    "title": "Geopolitical risk and inflation concerns create volatility",
                    "summary": "Market experiences temporary pullback amid interest rate uncertainty.",
                    "publisher": "Financial Times",
                    "url": "https://example.com/art2",
                },
            ]
        })
        result = analyze_sentiment(news_payload)
        self.assertIsInstance(result, SentimentResult)
        self.assertEqual(len(result.headline_breakdown), 2)
        total_prob = result.positive_score + result.neutral_score + result.negative_score
        self.assertAlmostEqual(total_prob, 1.0, places=2)
        self.assertTrue(-1.0 <= result.polarity_index <= 1.0)

    def test_neural_strategy_mocked(self):
        mock_strategy = MagicMock(spec=BaseSentimentStrategy)
        mock_strategy.classify_text.return_value = {
            "positive": 0.90,
            "neutral": 0.08,
            "negative": 0.02,
        }

        service = SentimentAnalysisService(neural_strategy=mock_strategy)
        articles = [{"title": "Massive rally and record growth", "summary": "Unprecedented gains."}]
        res = service.analyze(articles)

        self.assertEqual(res.method, "finbert_neural")
        self.assertEqual(res.label, "Positivo")
        self.assertGreater(res.polarity_index, 0.8)
        self.assertEqual(len(res.headline_breakdown), 1)
        self.assertEqual(res.headline_breakdown[0].label, "Positivo")

    def test_fallback_when_neural_fails(self):
        failing_neural = MagicMock(spec=BaseSentimentStrategy)
        failing_neural.classify_text.side_effect = RuntimeError("GPU/Torch error")

        fallback_strat = LexicalFallbackSentimentStrategy()
        service = SentimentAnalysisService(neural_strategy=failing_neural, fallback_strategy=fallback_strat)

        res = service.analyze("Bearish selloff and recession crisis leads to severe losses.")
        self.assertEqual(res.method, "lexical_fallback")
        self.assertEqual(res.label, "Negativo")
        self.assertLess(res.polarity_index, 0.0)


class TestSentimentChartAgent(unittest.TestCase):
    def test_build_markdown_output(self):
        agent = SentimentChartAgent()
        sample_news = json.dumps({
            "articles": [
                {
                    "title": "Global Tech ETF surges past all-time high on AI momentum",
                    "summary": "Massive capital inflows drive valuations higher.",
                    "url": "https://example.com/news",
                }
            ]
        })
        md = agent.build_markdown(isin="IE00B4L5Y983", news_data=sample_news)

        self.assertIn("## Sentiment e Analisi Notizie", md)
        self.assertIn("```mermaid", md)
        self.assertIn("pie showData", md)
        self.assertIn("graph LR", md)
        self.assertIn("IE00B4L5Y983", md)
        self.assertIn("Indice di Polarità Composito", md)


if __name__ == "__main__":
    unittest.main()
