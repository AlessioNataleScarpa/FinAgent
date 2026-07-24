"""Local CPU inference service for Amazon Chronos-Bolt."""

from __future__ import annotations

import logging
import math
import os
from contextlib import asynccontextmanager
from typing import List

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

    return ForecastResponse(
        model=MODEL_ID,
        mean=[round(float(value), 6) for value in mean_values],
        lower_bound=[round(float(value), 6) for value in lower],
        upper_bound=[round(float(value), 6) for value in upper],
        quantiles={
            level: [round(float(value), 6) for value in values]
            for level, values in quantiles.items()
        },
    )
