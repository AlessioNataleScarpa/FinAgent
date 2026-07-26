"""Visualise monthly point and interval forecasts at useful horizons."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

try:
    from utils.mermaid import build_xychart_lines, wrap_mermaid
except ImportError:
    from backend.utils.mermaid import build_xychart_lines, wrap_mermaid

from .state import PipelineState

logger = logging.getLogger(__name__)


def _future_months(horizon: int) -> List[str]:
    return [f"+{step}m" for step in range(1, horizon + 1)]


def _chart_specs(horizon: int) -> List[tuple[str, List[int]]]:
    specs: List[tuple[str, List[int]]] = [
        (
            "primo anno (dettaglio mensile)"
            if horizon >= 12
            else f"{horizon} mesi",
            list(range(min(horizon, 12))),
        )
    ]
    for years, months in ((5, 60), (10, 120), (20, 240)):
        if horizon >= months:
            specs.append(
                (
                    f"{years} anni (campionamento annuale)",
                    list(range(11, months, 12)),
                )
            )
    return specs


def forecast_charts_node(state: PipelineState) -> Dict[str, Any]:
    isin = state.get("isin", "N/D")
    forecast = state.get("tsfm_forecast") or {}
    mean = forecast.get("mean") or []
    lower = forecast.get("lower_bound") or []
    upper = forecast.get("upper_bound") or []
    prices = state.get("historical_prices") or []
    dates = state.get("historical_dates") or []

    if not (mean and lower and upper and prices):
        reason = forecast.get("error") or "serie o previsione non disponibile"
        return {
            "forecast_charts": (
                "## Previsione futura\n\n"
                f"Grafico non disponibile: {reason}.\n"
            )
        }

    horizon = min(len(mean), len(lower), len(upper))
    mean = [float(value) for value in mean[:horizon]]
    lower = [float(value) for value in lower[:horizon]]
    upper = [float(value) for value in upper[:horizon]]
    last_price = float(prices[-1])
    last_label = dates[-1] if dates else "T₀"
    future_labels = _future_months(horizon)
    status = forecast.get("status", "unavailable")
    model = forecast.get("model", "N/D")
    specs = _chart_specs(horizon)
    
    years_equiv = round(horizon / 12, 1)
    header = (
        f"## Previsione futura: {horizon} mesi ({years_equiv} anni)\n\n"
        if horizon >= 12
        else f"## Previsione futura: {horizon} mesi\n\n"
    )
    status_label = {
        "ok": "operativo",
        "fallback": "stima di riserva",
        "unavailable": "non disponibile",
    }.get(status, status)
    sections = [
        header,
        f"- **Modello:** `{model}` ({status_label})\n",
        "- **Frequenza:** mensile\n",
        "- **Intervallo di confidenza 80%:** banda tra limite inferiore e superiore\n\n",
        "La linea centrale rappresenta lo scenario mediano; le altre due delimitano "
        "il 10° e il 90° percentile.\n",
    ]

    for label, indices in specs:
        endpoint = indices[-1]
        expected_return = (mean[endpoint] / last_price - 1) * 100
        interval_width = (
            (upper[endpoint] - lower[endpoint]) / mean[endpoint] * 100
            if mean[endpoint]
            else 0
        )
        
        raw_x_labels = [last_label] + [future_labels[index] for index in indices]
        series_mean = [last_price] + [mean[index] for index in indices]
        series_lower = [last_price] + [lower[index] for index in indices]
        series_upper = [last_price] + [upper[index] for index in indices]

        chart = build_xychart_lines(
            title=f"Forecast {label} - {isin}",
            x_labels=raw_x_labels,
            series=[
                series_mean,
                series_lower,
                series_upper,
            ],
            y_axis_label="Prezzo",
        )
        sections.extend(
            [
                f"\n### Orizzonte {label}\n\n",
                f"- Scenario centrale finale (P50): **{mean[endpoint]:.2f}** "
                f"({expected_return:+.2f}%)\n",
                f"- Intervallo 80% finale (P10 - P90): **{lower[endpoint]:.2f} – "
                f"{upper[endpoint]:.2f}**\n",
                f"- Ampiezza relativa incertezza: **{interval_width:.2f}%**\n\n",
                f"{wrap_mermaid(chart)}\n",
            ]
        )

    sections.append(
        "\n_Le traiettorie sono probabilistiche e non rappresentano rendimenti garantiti._\n"
    )
    return {"forecast_charts": "".join(sections)}
