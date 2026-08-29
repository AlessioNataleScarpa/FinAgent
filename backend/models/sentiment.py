"""
Neural Sentiment Analysis Module for FinAgent.

Implements the Strategy and Protected Variations design patterns (GRASP / GoF)
for financial news sentiment classification using ProsusAI/finbert (Transformer)
with graceful fallback to calibrated lexical analysis.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

FINBERT_MODEL_ID = os.getenv("FINBERT_MODEL_ID", "ProsusAI/finbert")


class HeadlineSentiment(BaseModel):
    """Structured sentiment scores for a single financial headline/snippet."""

    title: str = Field(description="Titolo o testo della notizia")
    publisher: Optional[str] = Field(default="", description="Fonte o agenzia di stampa")
    url: Optional[str] = Field(default="", description="URL dell'articolo")
    positive: float = Field(ge=0.0, le=1.0, description="Probabilità Softmax classe Positiva")
    neutral: float = Field(ge=0.0, le=1.0, description="Probabilità Softmax classe Neutra")
    negative: float = Field(ge=0.0, le=1.0, description="Probabilità Softmax classe Negativa")
    polarity: float = Field(ge=-1.0, le=1.0, description="Indice di polarità continuo (Pos - Neg)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidenza della predizione (max probabilità)")
    label: Literal["Positivo", "Neutro", "Negativo"] = Field(description="Etichetta dominante")


class SentimentResult(BaseModel):
    """Consolidated aggregate sentiment result for an ETF's newsflow."""

    positive_score: float = Field(ge=0.0, le=1.0, description="Punteggio aggregato classe Positiva")
    neutral_score: float = Field(ge=0.0, le=1.0, description="Punteggio aggregato classe Neutra")
    negative_score: float = Field(ge=0.0, le=1.0, description="Punteggio aggregato classe Negativa")
    polarity_index: float = Field(ge=-1.0, le=1.0, description="Indice di polarità composito pesato [-1.0, +1.0]")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidenza media pesata")
    label: Literal["Positivo", "Neutro", "Negativo"] = Field(description="Sentiment complessivo dominante")
    method: Literal["finbert_neural", "lexical_fallback"] = Field(description="Metodo di inferenza utilizzato")
    headline_breakdown: List[HeadlineSentiment] = Field(default_factory=list, description="Dettaglio per singola notizia")
    summary_text: str = Field(default="", description="Sintesi discorsiva del sentiment per il report")


class BaseSentimentStrategy(ABC):
    """Abstract Strategy interface for financial sentiment classification (GRASP: Protected Variations)."""

    @abstractmethod
    def classify_text(self, text: str) -> Dict[str, float]:
        """Returns normalized probabilities {'positive': float, 'neutral': float, 'negative': float}."""
        pass


class FinBERTSentimentStrategy(BaseSentimentStrategy):
    """
    Concrete Strategy implementing Deep Learning inference with FinBERT (ProsusAI/finbert).
    Uses HuggingFace Transformers and PyTorch in inference mode.
    """

    def __init__(self, model_name: str = FINBERT_MODEL_ID):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._initialized = False

    def _lazy_init(self) -> bool:
        if self._initialized:
            return self._model is not None

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._torch = torch
            logger.info("Caricamento modello neurale FinBERT: %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            self._initialized = True
            logger.info("FinBERT caricato con successo per inferenza CPU")
            return True
        except Exception as exc:
            logger.warning("Inizializzazione FinBERT neurale non riuscita: %s. Attivazione fallback.", exc)
            self._initialized = True
            self._model = None
            return False

    def classify_text(self, text: str) -> Dict[str, float]:
        if not self._lazy_init() or self._model is None or self._tokenizer is None:
            raise RuntimeError("FinBERT non disponibile")

        inputs = self._tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with self._torch.inference_mode():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = self._torch.nn.functional.softmax(logits, dim=-1)[0].cpu().tolist()

        # ProsusAI/finbert output order: [positive, negative, neutral]
        return {
            "positive": round(float(probs[0]), 4),
            "negative": round(float(probs[1]), 4),
            "neutral": round(float(probs[2]), 4),
        }


class LexicalFallbackSentimentStrategy(BaseSentimentStrategy):
    """
    Calibrated analytical fallback strategy based on financial domain lexicon weighting.
    Used when neural dependencies are unavailable or in resource-constrained environments.
    """

    POSITIVE_WORDS = {
        "bullish": 2.0, "positivo": 1.5, "growth": 1.5, "crescita": 1.5, "rialzista": 2.0,
        "gain": 1.5, "profit": 1.5, "inflow": 1.8, "beat": 1.8, "exceed": 1.5, "favorevole": 1.2,
        "rally": 2.0, "outperform": 2.0, "dividend": 1.2, "upgrade": 1.8, "surge": 1.8,
        "high": 1.0, "record": 1.5, "strong": 1.5, "positive": 1.5, "opportunity": 1.3
    }

    NEGATIVE_WORDS = {
        "bearish": 2.0, "negativo": 1.5, "rischi": 1.5, "rischio": 1.5, "volatilità": 1.2,
        "volatil": 1.2, "friction": 1.2, "drawdown": 2.0, "sell": 1.5, "selloff": 2.0,
        "outflow": 1.8, "geopolitic": 1.5, "geopolitico": 1.5, "downturn": 1.8, "inflation": 1.3,
        "inflazione": 1.3, "recession": 2.0, "recessione": 2.0, "downgrade": 1.8, "drop": 1.5,
        "loss": 1.8, "crisis": 2.0, "war": 2.0, "weak": 1.5, "slump": 1.8, "debt": 1.2
    }

    def classify_text(self, text: str) -> Dict[str, float]:
        tokens = re.findall(r"\b\w+\b", (text or "").lower())
        pos_weight = sum(self.POSITIVE_WORDS.get(tok, 0.0) for tok in tokens)
        neg_weight = sum(self.NEGATIVE_WORDS.get(tok, 0.0) for tok in tokens)

        base_neutral = 1.0
        total = pos_weight + neg_weight + base_neutral

        p_pos = pos_weight / total
        p_neg = neg_weight / total
        p_neu = base_neutral / total

        return {
            "positive": round(p_pos, 4),
            "neutral": round(p_neu, 4),
            "negative": round(p_neg, 4),
        }


class SentimentAnalysisService:
    """
    Pure Fabrication / Information Expert orchestrating financial sentiment analysis.
    Manages strategy selection, multi-headline exponential weighting, and structured output generation.
    """

    def __init__(self, neural_strategy: Optional[BaseSentimentStrategy] = None, fallback_strategy: Optional[BaseSentimentStrategy] = None):
        self.neural_strategy = neural_strategy or FinBERTSentimentStrategy()
        self.fallback_strategy = fallback_strategy or LexicalFallbackSentimentStrategy()

    @staticmethod
    def _extract_articles(news_payload: Union[str, dict, list]) -> List[Dict[str, str]]:
        """Parses articles list from raw string, dictionary or list input."""
        if isinstance(news_payload, list):
            return [item for item in news_payload if isinstance(item, dict)]

        if isinstance(news_payload, dict):
            articles = news_payload.get("articles")
            if isinstance(articles, list):
                return [item for item in articles if isinstance(item, dict)]
            return [news_payload]

        if isinstance(news_payload, str):
            text = news_payload.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and isinstance(parsed.get("articles"), list):
                    return [item for item in parsed["articles"] if isinstance(item, dict)]
                if isinstance(parsed, list):
                    return [item for item in parsed if isinstance(item, dict)]
            except (json.JSONDecodeError, TypeError):
                pass
            lines = [line.strip("- *#") for line in text.splitlines() if len(line.strip("- *#")) > 10]
            if lines:
                return [{"title": line, "summary": ""} for line in lines[:8]]
            return [{"title": text[:300], "summary": ""}]

        return []

    def analyze(self, news_input: Union[str, dict, list]) -> SentimentResult:
        """
        Executes sentiment analysis over the provided financial news input.
        Returns calibrated probabilities, polarity index, and per-headline breakdowns.
        """
        articles = self._extract_articles(news_input)

        if not articles:
            return SentimentResult(
                positive_score=0.3333,
                neutral_score=0.3334,
                negative_score=0.3333,
                polarity_index=0.0,
                confidence=0.3334,
                label="Neutro",
                method="lexical_fallback",
                headline_breakdown=[],
                summary_text="Nessuna notizia specifica disponibile; sentiment assunto neutrale di base.",
            )

        headlines_results: List[HeadlineSentiment] = []
        method_used: Literal["finbert_neural", "lexical_fallback"] = "finbert_neural"

        for art in articles[:8]:
            title = str(art.get("title") or art.get("headline") or "Notizia finanziaria").strip()
            summary = str(art.get("summary") or art.get("description") or "").strip()
            publisher = str(art.get("publisher") or art.get("source") or "").strip()
            url = str(art.get("url") or art.get("link") or "").strip()

            combined_text = f"{title}. {summary}".strip() if summary else title

            probs = None
            try:
                probs = self.neural_strategy.classify_text(combined_text)
            except Exception:
                method_used = "lexical_fallback"
                probs = self.fallback_strategy.classify_text(combined_text)

            pos = probs["positive"]
            neu = probs["neutral"]
            neg = probs["negative"]
            polarity = round(pos - neg, 4)
            confidence = round(max(pos, neu, neg), 4)

            if pos > neg and pos > neu:
                label: Literal["Positivo", "Neutro", "Negativo"] = "Positivo"
            elif neg > pos and neg > neu:
                label = "Negativo"
            else:
                label = "Neutro"

            headlines_results.append(
                HeadlineSentiment(
                    title=title,
                    publisher=publisher,
                    url=url,
                    positive=pos,
                    neutral=neu,
                    negative=neg,
                    polarity=polarity,
                    confidence=confidence,
                    label=label,
                )
            )

        n = len(headlines_results)
        raw_weights = [math.exp(-0.25 * i) for i in range(n)]
        sum_weights = sum(raw_weights) or 1.0
        normalized_weights = [w / sum_weights for w in raw_weights]

        agg_pos = sum(h.positive * w for h, w in zip(headlines_results, normalized_weights))
        agg_neu = sum(h.neutral * w for h, w in zip(headlines_results, normalized_weights))
        agg_neg = sum(h.negative * w for h, w in zip(headlines_results, normalized_weights))

        total_agg = agg_pos + agg_neu + agg_neg or 1.0
        agg_pos /= total_agg
        agg_neu /= total_agg
        agg_neg /= total_agg

        polarity_index = round(agg_pos - agg_neg, 4)
        agg_confidence = round(sum(h.confidence * w for h, w in zip(headlines_results, normalized_weights)), 4)

        if agg_pos > agg_neg and agg_pos >= 0.40:
            agg_label: Literal["Positivo", "Neutro", "Negativo"] = "Positivo"
        elif agg_neg > agg_pos and agg_neg >= 0.40:
            agg_label = "Negativo"
        else:
            agg_label = "Neutro"

        method_label = "FinBERT (ProsusAI)" if method_used == "finbert_neural" else "Analitico Lessicale (Fallback)"
        summary_text = (
            f"Sentiment aggregato: **{agg_label}** (Indice di Polarità: `{polarity_index:+.2f}`, "
            f"Confidenza media: `{agg_confidence * 100:.1f}%`). "
            f"Inferenza condotta tramite `{method_label}` su {n} articoli di mercato."
        )

        return SentimentResult(
            positive_score=round(agg_pos, 4),
            neutral_score=round(agg_neu, 4),
            negative_score=round(agg_neg, 4),
            polarity_index=polarity_index,
            confidence=agg_confidence,
            label=agg_label,
            method=method_used,
            headline_breakdown=headlines_results,
            summary_text=summary_text,
        )


_service_singleton: Optional[SentimentAnalysisService] = None


def get_sentiment_service() -> SentimentAnalysisService:
    """Returns singleton instance of SentimentAnalysisService."""
    global _service_singleton
    if _service_singleton is None:
        _service_singleton = SentimentAnalysisService()
    return _service_singleton


def analyze_sentiment(news_input: Union[str, dict, list]) -> SentimentResult:
    """Convenience facade function for sentiment analysis."""
    return get_sentiment_service().analyze(news_input)
