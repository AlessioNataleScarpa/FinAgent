"""Node: FORECAST CHARTS — visualise TSFM point and interval forecasts on weekly frequency."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

try:
    from utils.mermaid import build_xychart_lines, wrap_mermaid
except ImportError:
    from backend.utils.mermaid import build_xychart_lines, wrap_mermaid

from .state import PipelineState

logger = logging.getLogger(__name__)


def _future_weeks(horizon: int) -> List[str]:
    """Generate weekly horizon labels (+1w, +2w, ..., +Hw)."""
    return [f"+{step}w" for step in range(1, horizon + 1)]


def _chart_specs(horizon: int) -> List[tuple[str, List[int]]]:
    """
    Build chart specifications tailored to weekly frequency.
    For short-to-medium horizons, show full weekly resolution.
    For longer horizons (e.g. 52, 104, 156 weeks), offer weekly and yearly sampled views.
    """
    specs: List[tuple[str, List[int]]] = []

    if horizon <= 52:
        # Full weekly breakdown for up to 1 year (52 weeks)
        num_years = max(1, round(horizon / 52)) if horizon >= 26 else None
        label_str = f"{horizon} settimane (dettaglio settimanale)" if not num_years else f"{num_years} anno/i ({horizon} settimane)"
        specs.append((label_str, list(range(horizon))))
    else:
        # For multi-year weekly forecasts (> 52 weeks)
        specs.append((f"{horizon} settimane (dettaglio settimanale integrale)", list(range(horizon))))
        # Add yearly sampled view (every 52 weeks)
        yearly_indices = list(range(51, horizon, 52))
        if yearly_indices:
            years_count = len(yearly_indices)
            specs.append((f"{years_count} anni (campionamento annuale a step 52w)", yearly_indices))

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
    future_labels = _future_weeks(horizon)
    status = forecast.get("status", "unavailable")
    model = forecast.get("model", "N/D")
    specs = _chart_specs(horizon)
    
    weeks_horizon = horizon
    years_equiv = round(weeks_horizon / 52, 1)
    
    header = (
        f"## Previsione futura settimanale: {weeks_horizon} settimane ({years_equiv} anni)\n\n"
        if weeks_horizon >= 52
        else f"## Previsione futura settimanale: {weeks_horizon} settimane\n\n"
    )
    
    sections = [
        header,
        f"- **Modello:** `{model}` (`{status}`)\n",
        "- **Frequenza dati:** Settimanale (`1wk`)\n",
        "- **Intervallo di confidenza 80%:** banda tra limite inferiore e superiore\n\n",
        "In ogni grafico: **linea 1 = scenario centrale (P50)**, **linea 2 = limite "
        "inferiore (P10)**, **linea 3 = limite superiore (P90)**.\n",
    ]

    for label, indices in specs:
        endpoint = indices[-1]
        expected_return = (mean[endpoint] / last_price - 1) * 100
        interval_width = (
            (upper[endpoint] - lower[endpoint]) / mean[endpoint] * 100
            if mean[endpoint]
            else 0
        )
        
        # Subsample x_labels to avoid overcrowding if horizon > 26 weeks
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
        "\n_Queste sono previsioni probabilistiche settimanali, non rendimenti garantiti._\n"
    )
    return {"forecast_charts": "".join(sections)}
