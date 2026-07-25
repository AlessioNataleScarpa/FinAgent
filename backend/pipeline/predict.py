"""Zero-shot time-series forecasting node (OUT 2).

The node prefers a configured TSFM inference endpoint and falls back to a
deterministic statistical forecaster.  The fallback keeps the graph usable
offline, but is clearly identified in both state and report output.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import statistics
from typing import Any, Dict, List, Mapping, Sequence

import httpx

from .state import PipelineState, TSFMForecast

logger = logging.getLogger(__name__)

DEFAULT_HORIZON = 240
MIN_CONTEXT = 8


def _positive_finite(values: Sequence[Any]) -> List[float]:
    clean: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number > 0:
            clean.append(number)
    return clean


def _extract_prices(state: PipelineState) -> List[float]:
    direct = _positive_finite(state.get("historical_prices") or [])
    if direct:
        return direct

    raw = state.get("info_storici") or ""
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []

    if not isinstance(payload, dict):
        return []

    points = (
        payload.get("Weekly_Closes")
        or payload.get("weekly_closes")
        or payload.get("Monthly_Closes")
        or payload.get("monthly_closes")
        or []
    )
    if not isinstance(points, list):
        hist = payload.get("Historical_Prices") or {}
        if isinstance(hist, dict):
            points = hist.get("Weekly_Closes") or hist.get("historical") or hist.get("Monthly_Closes") or []
        elif isinstance(hist, list):
            points = hist

    if not isinstance(points, list):
        return []

    return _positive_finite(
        point.get("close") or point.get("Close") for point in points if isinstance(point, dict)
    )


def _horizon() -> int:
    try:
        return max(1, min(int(os.getenv("TSFM_HORIZON", DEFAULT_HORIZON)), 240))
    except ValueError:
        return DEFAULT_HORIZON


def _coerce_vector(
    value: Any,
    horizon: int,
    *,
    allow_zero: bool = False,
) -> List[float]:
    if not isinstance(value, list):
        return []
    clean: List[float] = []
    for item in value[:horizon]:
        try:
            number = float(item)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(number):
            return []
        if allow_zero:
            number = max(0.0, number)
        elif number <= 0:
            return []
        clean.append(number)
    return clean if len(clean) == horizon else []


def _remote_tsfm_forecast(prices: List[float], horizon: int) -> TSFMForecast:
    """Call a model-agnostic TSFM endpoint (Chronos, TimesFM, Lag-Llama, ...)."""
    endpoint = os.getenv("TSFM_ENDPOINT", "").strip()
    if not endpoint:
        raise RuntimeError("TSFM_ENDPOINT non configurato")

    timeout = float(os.getenv("TSFM_TIMEOUT_SECONDS", "45"))
    model = os.getenv("TSFM_MODEL_ID", "configured-tsfm")
    request = {
        "model": model,
        "series": prices,
        "frequency": "M",
        "prediction_length": horizon,
        "quantile_levels": [0.1, 0.5, 0.9],
    }
    headers = {"Content-Type": "application/json"}
    token = os.getenv("TSFM_API_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with httpx.Client(timeout=timeout) as client:
        response = client.post(endpoint, json=request, headers=headers)
        response.raise_for_status()
        body = response.json()

    # Accept either a flat response or {"forecast": {...}}.
    data = body.get("forecast", body) if isinstance(body, dict) else {}
    quantiles = data.get("quantiles", {}) if isinstance(data, dict) else {}
    mean = _coerce_vector(
        data.get("mean") or data.get("median") or data.get("prediction"),
        horizon,
    )
    lower = _coerce_vector(
        data.get("lower_bound")
        or data.get("lower")
        or (quantiles.get("0.1") if isinstance(quantiles, Mapping) else None),
        horizon,
        allow_zero=True,
    )
    upper = _coerce_vector(
        data.get("upper_bound")
        or data.get("upper")
        or (quantiles.get("0.9") if isinstance(quantiles, Mapping) else None),
        horizon,
    )
    if not (mean and lower and upper):
        raise ValueError(
            "Risposta TSFM non valida: servono mean, lower_bound e upper_bound"
        )

    return {
        "model": str(data.get("model") or model),
        "frequency": "monthly",
        "horizon": horizon,
        "confidence_level": 0.8,
        "mean": mean,
        "lower_bound": lower,
        "upper_bound": upper,
        "status": "ok",
    }


def _statistical_fallback(
    prices: List[float],
    horizon: int,
    error: str,
) -> TSFMForecast:
    """Robust damped-trend forecast used only when the TSFM is unavailable."""
    if len(prices) < MIN_CONTEXT:
        return {
            "model": "unavailable",
            "frequency": "monthly",
            "horizon": horizon,
            "confidence_level": 0.0,
            "mean": [],
            "lower_bound": [],
            "upper_bound": [],
            "status": "unavailable",
            "error": error or f"Servono almeno {MIN_CONTEXT} osservazioni",
        }

    window = prices[-min(len(prices), 36) :]
    log_returns = [
        math.log(current / previous)
        for previous, current in zip(window, window[1:])
        if previous > 0 and current > 0
    ]
    drift = statistics.median(log_returns) if log_returns else 0.0
    volatility = statistics.stdev(log_returns) if len(log_returns) > 1 else 0.0
    # Prevent one unusual period from creating an explosive long-horizon path.
    drift = max(-0.08, min(drift, 0.08))
    volatility = max(0.005, min(volatility, 0.35))

    mean: List[float] = []
    lower: List[float] = []
    upper: List[float] = []
    last = window[-1]
    cumulative_drift = 0.0
    z_score = 1.2816  # central 80% interval (q10–q90)
    for step in range(1, horizon + 1):
        cumulative_drift += drift * (0.9 ** (step - 1))
        center = last * math.exp(cumulative_drift)
        uncertainty = z_score * volatility * math.sqrt(step)
        mean.append(round(center, 4))
        lower.append(round(center * math.exp(-uncertainty), 4))
        upper.append(round(center * math.exp(uncertainty), 4))

    return {
        "model": "robust-damped-trend-fallback",
        "frequency": "monthly",
        "horizon": horizon,
        "confidence_level": 0.8,
        "mean": mean,
        "lower_bound": lower,
        "upper_bound": upper,
        "status": "fallback",
        "error": error,
    }


def _legacy_direction(raw: str) -> str:
    """Preserve a useful signal for old free-form historical inputs."""
    match = re.search(r"([+-]?\d+(?:[.,]\d+)?)\s*%", raw)
    if not match:
        return "UNAVAILABLE"
    value = float(match.group(1).replace(",", "."))
    if value > 1:
        return "BULLISH"
    if value < -1:
        return "BEARISH"
    return "NEUTRAL"


def _render_forecast(
    isin: str,
    prices: List[float],
    forecast: TSFMForecast,
    legacy_raw: str,
) -> str:
    mean = forecast.get("mean") or []
    lower = forecast.get("lower_bound") or []
    upper = forecast.get("upper_bound") or []

    horizon_rows: List[str] = []
    if prices and mean:
        direction_index = min(12, len(mean)) - 1
        directional_return = (mean[direction_index] / prices[-1] - 1.0) * 100
        if directional_return > 1:
            direction = "BULLISH"
        elif directional_return < -1:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        for label, step in (
            ("1 anno", 12),
            ("5 anni", 60),
            ("10 anni", 120),
            ("20 anni", 240),
        ):
            if len(mean) < step or len(lower) < step or len(upper) < step:
                continue
            index = step - 1
            expected_return = (mean[index] / prices[-1] - 1.0) * 100
            horizon_rows.append(
                f"| {label} | {mean[index]:.4f} | {expected_return:+.2f}% | "
                f"{lower[index]:.4f} – {upper[index]:.4f} |"
            )
        expected = f"{directional_return:+.2f}% a 12 mesi"
    else:
        direction = _legacy_direction(legacy_raw)
        expected = "non disponibile"

    status = forecast.get("status", "unavailable")
    warning = ""
    if status == "fallback":
        warning = (
            "\nATTENZIONE: il TSFM non era raggiungibile; risultato prodotto dal "
            "fallback statistico, non da un Foundation Model."
        )
    elif status == "unavailable":
        warning = f"\nPREVISIONE NON DISPONIBILE: {forecast.get('error', '')}"

    table = ""
    if horizon_rows:
        table = (
            "\n| Orizzonte | Scenario centrale | Variazione | Intervallo 80% |\n"
            "|---|---:|---:|---:|\n"
            + "\n".join(horizon_rows)
            + "\n"
        )
    return (
        "--- TSFM FORECAST OUTPUT (OUT 2) ---\n"
        f"Target ISIN: {isin}\n"
        f"Model: {forecast.get('model', 'N/A')}\n"
        f"Backend status: {status}\n"
        f"Context observations: {len(prices)}\n"
        f"Forecast horizon: {forecast.get('horizon', 0)} months\n"
        f"Trend Direction (12 months): {direction}\n"
        f"Projected return: {expected}\n"
        f"{table}"
        "Interpretation rule: a wide interval means high uncertainty; do not "
        "let news sentiment turn it into a high-confidence directional claim.\n"
        f"{warning}\n"
    )


def predict_node(state: PipelineState) -> Dict[str, Any]:
    """Run a zero-shot forecast over the historical close-price sequence."""
    isin = state.get("isin", "N/A")
    raw = state.get("info_storici", "") or ""
    prices = _extract_prices(state)
    horizon = _horizon()
    logger.info(
        "Generating zero-shot forecast for ISIN %s with %d observations",
        isin,
        len(prices),
    )

    if len(prices) < MIN_CONTEXT:
        forecast = _statistical_fallback(
            prices,
            horizon,
            f"Contesto insufficiente: {len(prices)} osservazioni, minimo {MIN_CONTEXT}",
        )
    else:
        try:
            forecast = _remote_tsfm_forecast(prices, horizon)
        except Exception as exc:
            logger.warning("TSFM inference unavailable for %s: %s", isin, exc)
            forecast = _statistical_fallback(prices, horizon, str(exc))

    return {
        "tsfm_forecast": forecast,
        "prediction_out2": _render_forecast(isin, prices, forecast, raw),
    }
