"""Node: FORECAST CHARTS — visualise TSFM point and interval forecasts."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

try:
    from utils.mermaid import build_xychart_lines, wrap_mermaid
except ImportError:
    from backend.utils.mermaid import build_xychart_lines, wrap_mermaid

from .state import PipelineState

logger = logging.getLogger(__name__)


def _future_months(last_date: str, horizon: int) -> List[str]:
    try:
        year, month = (int(part) for part in last_date[:7].split("-"))
    except (TypeError, ValueError):
        return [f"F+{step}" for step in range(1, horizon + 1)]

    labels: List[str] = []
    for _ in range(horizon):
        month += 1
        if month == 13:
            month = 1
            year += 1
        labels.append(f"{year:04d}-{month:02d}")
    return labels


def _chart_specs(horizon: int) -> List[tuple[str, List[int]]]:
    short_horizon = min(12, horizon)
    specs: List[tuple[str, List[int]]] = [
        (
            "1 anno (dettaglio mensile)"
            if horizon >= 12
            else f"{horizon} mesi (dettaglio mensile)",
            list(range(short_horizon)),
        )
    ]
    for years in (5, 10, 20):
        months = years * 12
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
    last_label = dates[-1] if dates else "Ultimo"
    future_labels = _future_months(last_label, horizon)
    status = forecast.get("status", "unavailable")
    model = forecast.get("model", "N/D")
    specs = _chart_specs(horizon)
    title_horizons = ", ".join(spec[0].split(" ")[0] for spec in specs)
    header = (
        f"## Previsioni future: {title_horizons}\n\n"
        if horizon >= 12
        else f"## Previsione futura: {horizon} mesi\n\n"
    )
    sections = [
        header,
        f"- **Modello:** `{model}` (`{status}`)\n",
        "- **Intervallo di previsione 80%:** banda tra limite inferiore e superiore\n\n",
        "In ogni grafico: **linea 1 = scenario centrale**, **linea 2 = limite "
        "inferiore**, **linea 3 = limite superiore**.\n",
    ]

    for label, indices in specs:
        endpoint = indices[-1]
        expected_return = (mean[endpoint] / last_price - 1) * 100
        interval_width = (
            (upper[endpoint] - lower[endpoint]) / mean[endpoint] * 100
            if mean[endpoint]
            else 0
        )
        chart = build_xychart_lines(
            title=f"Forecast {label} - {isin}",
            x_labels=[last_label] + [future_labels[index] for index in indices],
            series=[
                [last_price] + [mean[index] for index in indices],
                [last_price] + [lower[index] for index in indices],
                [last_price] + [upper[index] for index in indices],
            ],
            y_axis_label="Prezzo",
        )
        sections.extend(
            [
                f"\n### Orizzonte {label}\n\n",
                f"- Scenario centrale finale: **{mean[endpoint]:.2f}** "
                f"({expected_return:+.2f}%)\n",
                f"- Intervallo 80% finale: **{lower[endpoint]:.2f} – "
                f"{upper[endpoint]:.2f}**\n",
                f"- Ampiezza relativa: **{interval_width:.2f}%**\n\n",
                f"{wrap_mermaid(chart)}\n",
            ]
        )

    if horizon >= 120:
        sections.append(
            "\n> Gli orizzonti di 10 e 20 anni sono scenari esplorativi: errori, "
            "regimi di mercato e inflazione si accumulano, quindi la banda conta "
            "più della singola linea centrale.\n"
        )
    sections.append(
        "\n_Queste sono previsioni probabilistiche, non rendimenti garantiti._\n"
    )
    return {"forecast_charts": "".join(sections)}
