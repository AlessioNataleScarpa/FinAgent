"""
Neural and Machine Learning models package for FinAgent.
"""

from .sentiment import (
    HeadlineSentiment,
    SentimentResult,
    SentimentAnalysisService,
    analyze_sentiment,
    get_sentiment_service,
)
from .explainability import (
    WindowAttribution,
    TemporalExplanationResult,
    BaseTemporalExplainer,
    KernelShapTemporalExplainer,
    OcclusionTemporalExplainer,
    TemporalXAIService,
    get_xai_service,
)

__all__ = [
    "HeadlineSentiment",
    "SentimentResult",
    "SentimentAnalysisService",
    "analyze_sentiment",
    "get_sentiment_service",
    "WindowAttribution",
    "TemporalExplanationResult",
    "BaseTemporalExplainer",
    "KernelShapTemporalExplainer",
    "OcclusionTemporalExplainer",
    "TemporalXAIService",
    "get_xai_service",
]
