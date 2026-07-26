"""Local CPU inference service for Amazon Chronos-Bolt."""

from __future__ import annotations

import logging
import math
import os
from contextlib import asynccontextmanager
from typing import Any, List

import torch
from chronos import BaseChronosPipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("tsfm")

MODEL_ID = os.getenv("TSFM_MODEL_ID", "amazon/chronos-bolt-tiny")
pipeline: BaseChronosPipeline | None = None


class ForecastRequest(BaseModel):
    model: str | None = None
    series: List[float] = Field(min_length=8, max_length=2048)
    frequency: str = "M"
    prediction_length: int = Field(default=240, ge=1, le=240)
    quantile_levels: List[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    explain: bool = True

    @field_validator("series")
    @classmethod
    def validate_series(cls, values: List[float]) -> List[float]:
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("series must contain finite positive values")
        return values

    @field_validator("quantile_levels")
    @classmethod
    def validate_quantiles(cls, values: List[float]) -> List[float]:
        if not values or any(not 0 < value < 1 for value in values):
            raise ValueError("quantile_levels must be between 0 and 1")
        return values


class ForecastResponse(BaseModel):
    model: str
    mean: List[float]
    lower_bound: List[float]
    upper_bound: List[float]
    quantiles: dict[str, List[float]]
    explanation: dict[str, Any] | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global pipeline
    threads = max(1, int(os.getenv("TORCH_NUM_THREADS", "2")))
    torch.set_num_threads(threads)
    logger.info("Loading %s on CPU with %d torch threads", MODEL_ID, threads)
    pipeline = BaseChronosPipeline.from_pretrained(
        MODEL_ID,
        device_map="cpu",
        dtype=torch.float32,
    )
    logger.info("Chronos model ready")
    yield
    pipeline = None


app = FastAPI(title="DL-2026 Local TSFM", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ready" if pipeline is not None else "loading",
        "model": MODEL_ID,
        "device": "cpu",
    }


def _nearest_quantile(levels: List[float], target: float) -> int:
    return min(range(len(levels)), key=lambda index: abs(levels[index] - target))


def _neutralise_window(series: List[float], start: int, end: int) -> List[float]:
    """Replace one temporal window with a locally neutral counterfactual."""
    result = list(series)
    left = series[start - 1] if start > 0 else series[end] if end < len(series) else series[0]
    right = series[end] if end < len(series) else left
    width = max(1, end - start)
    for offset, index in enumerate(range(start, end), start=1):
        fraction = offset / (width + 1)
        result[index] = left + (right - left) * fraction
    return result


def _temporal_occlusion_explanation(
    request: ForecastRequest,
    baseline_mean: List[float],
    levels: List[float],
) -> dict[str, Any] | None:
    """Measure how distinct historical windows affect the 12-month forecast.

    This is a model-agnostic perturbation explanation: each window is replaced
    by a neutral interpolation and Chronos is queried again.  The difference
    against the original forecast is reported in percentage points.
    """
    if pipeline is None or len(request.series) < 12 or not baseline_mean:
        return None

    size = len(request.series)
    definitions = [
        ("Ultimi 6 mesi", max(0, size - 6), size),
        ("Da 7 a 18 mesi fa", max(0, size - 18), max(0, size - 6)),
        ("Da 19 a 36 mesi fa", max(0, size - 36), max(0, size - 18)),
    ]
    definitions = [
        item for item in definitions if item[2] - item[1] >= 2
    ]
    if not definitions:
        return None

    target = min(12, request.prediction_length, len(baseline_mean))
    counterfactuals = [
        _neutralise_window(request.series, start, end)
        for _, start, end in definitions
    ]
    contexts = torch.tensor(counterfactuals, dtype=torch.float32)
    with torch.inference_mode():
        _, perturbed_means = pipeline.predict_quantiles(
            inputs=contexts,
            prediction_length=target,
            quantile_levels=levels,
        )

    baseline_target = float(baseline_mean[target - 1])
    last_price = float(request.series[-1])
    raw_windows = []
    for index, (label, start, end) in enumerate(definitions):
        perturbed_target = float(perturbed_means[index, target - 1].detach().cpu())
        impact = (baseline_target - perturbed_target) / last_price * 100.0
        window_start = request.series[start]
        window_end = request.series[end - 1]
        trend = (window_end / window_start - 1.0) * 100.0 if window_start else 0.0
        raw_windows.append(
            {
                "label": label,
                "trend_pct": round(trend, 3),
                "impact_pct_points": round(impact, 3),
            }
        )

    total_impact = sum(abs(item["impact_pct_points"]) for item in raw_windows)
    for item in raw_windows:
        item["importance_pct"] = round(
            abs(item["impact_pct_points"]) / total_impact * 100.0,
            2,
        ) if total_impact else 0.0

    return {
        "method": "temporal_occlusion",
        "model_agnostic": True,
        "target_horizon_months": target,
        "baseline_target": round(baseline_target, 6),
        "windows": raw_windows,
        "interpretation": (
            "Impatto positivo: quella finestra storica alza la previsione rispetto "
            "alla versione neutralizzata; impatto negativo: la riduce."
        ),
    }


@app.post("/forecast", response_model=ForecastResponse)
def forecast(request: ForecastRequest) -> ForecastResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Chronos model is still loading")

    levels = sorted(set(request.quantile_levels + [0.1, 0.5, 0.9]))
    context = torch.tensor(request.series, dtype=torch.float32)

    try:
        with torch.inference_mode():
            quantile_tensor, mean_tensor = pipeline.predict_quantiles(
                inputs=context,
                prediction_length=request.prediction_length,
                quantile_levels=levels,
            )
    except Exception as exc:
        logger.exception("Chronos inference failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Chronos returns [batch, horizon, quantile] and [batch, horizon].
    quantile_values = quantile_tensor[0].detach().cpu()
    mean_values = mean_tensor[0].detach().cpu().tolist()
    quantiles = {
        str(level): quantile_values[:, index].tolist()
        for index, level in enumerate(levels)
    }
    lower = quantile_values[:, _nearest_quantile(levels, 0.1)].tolist()
    upper = quantile_values[:, _nearest_quantile(levels, 0.9)].tolist()
    explanation = None
    if request.explain:
        try:
            explanation = _temporal_occlusion_explanation(
                request,
                [float(value) for value in mean_values],
                levels,
            )
        except Exception:
            logger.warning(
                "Temporal occlusion explanation unavailable; forecast remains valid",
                exc_info=True,
            )

    return ForecastResponse(
        model=MODEL_ID,
        mean=[round(float(value), 6) for value in mean_values],
        lower_bound=[round(float(value), 6) for value in lower],
        upper_bound=[round(float(value), 6) for value in upper],
        quantiles={
            level: [round(float(value), 6) for value in values]
            for level, values in quantiles.items()
        },
        explanation=explanation,
    )
