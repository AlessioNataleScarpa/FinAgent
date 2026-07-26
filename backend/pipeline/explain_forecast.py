"""Human-readable, model-aware explanation of the forecast."""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List

from .state import PipelineState


def _returns(prices: List[float]) -> List[float]:
    return [
        math.log(current / previous)
        for previous, current in zip(prices, prices[1:])
        if previous > 0 and current > 0
    ]


def _risk_context(prices: List[float], forecast: Dict[str, Any]) -> str:
    returns = _returns(prices[-37:])
    volatility = (
        statistics.stdev(returns) * math.sqrt(12.0) * 100.0
        if len(returns) > 1
        else 0.0
    )
    peak = max(prices) if prices else 0.0
    drawdown = ((prices[-1] / peak) - 1.0) * 100.0 if peak else 0.0

    mean = forecast.get("mean") or []
    lower = forecast.get("lower_bound") or []
    upper = forecast.get("upper_bound") or []
    target = min(12, len(mean), len(lower), len(upper))
    if target:
        index = target - 1
        interval = (
            (float(upper[index]) - float(lower[index])) / float(mean[index]) * 100.0
            if float(mean[index])
            else 0.0
        )
        interval_text = f"ampiezza relativa dell'intervallo a {target} mesi: **{interval:.1f}%**"
    else:
        interval_text = "intervallo previsionale non disponibile"

    return (
        f"Volatilità storica annualizzata: **{volatility:.1f}%**; "
        f"drawdown dal massimo del campione: **{drawdown:.1f}%**; "
        f"{interval_text}."
    )


def _occlusion_markdown(explanation: Dict[str, Any]) -> str:
    windows = [
        item
        for item in explanation.get("windows") or []
        if isinstance(item, dict)
    ]
    if not windows:
        return ""

    rows = []
    for item in windows:
        impact = float(item.get("impact_pct_points") or 0.0)
        effect = "spinge in alto" if impact > 0.01 else "spinge in basso" if impact < -0.01 else "quasi neutro"
        rows.append(
            f"| {item.get('label', 'Periodo')} | "
            f"{float(item.get('trend_pct') or 0.0):+.2f}% | "
            f"{impact:+.2f} p.p. ({effect}) | "
            f"{float(item.get('importance_pct') or 0.0):.1f}% |"
        )

    strongest = max(
        windows,
        key=lambda item: abs(float(item.get("impact_pct_points") or 0.0)),
    )
    strongest_impact = float(strongest.get("impact_pct_points") or 0.0)
    direction = "rialzo" if strongest_impact > 0 else "ribasso" if strongest_impact < 0 else "neutralità"
    target = explanation.get("target_horizon_months") or 12
    return (
        "La spiegazione usa **occlusione temporale**: il modello viene rieseguito "
        "neutralizzando, una alla volta, diverse porzioni dello storico. La "
        "differenza rispetto alla previsione originale misura l'influenza locale "
        f"sullo scenario a {target} mesi.\n\n"
        "| Periodo osservato | Movimento nel periodo | Effetto sulla previsione | Importanza relativa |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n"
        f"Il tratto più influente è **{strongest.get('label', 'il periodo recente')}** "
        f"e orienta localmente il modello verso il **{direction}**. Questa è una "
        "spiegazione della singola previsione, non una relazione causale di mercato.\n"
    )


def _fallback_markdown(explanation: Dict[str, Any]) -> str:
    drift = float(explanation.get("median_monthly_return_pct") or 0.0)
    volatility = float(explanation.get("monthly_volatility_pct") or 0.0)
    direction = "positivo" if drift > 0.01 else "negativo" if drift < -0.01 else "neutro"
    return (
        "Il modello principale non era raggiungibile, quindi la motivazione riguarda "
        "la stima statistica di riserva. Il rendimento mensile mediano recente è "
        f"**{drift:+.2f}%** ({direction}) e determina la direzione centrale; la "
        f"volatilità mensile di **{volatility:.2f}%** determina soprattutto "
        "l'allargamento dell'intervallo. Il trend viene smorzato progressivamente "
        "per non proiettare all'infinito il ritmo recente.\n"
    )


def explain_forecast_node(state: PipelineState) -> Dict[str, Any]:
    forecast = state.get("tsfm_forecast") or {}
    prices = [
        float(value)
        for value in state.get("historical_prices") or []
        if isinstance(value, (int, float)) and float(value) > 0
    ]
    explanation = forecast.get("explanation") or {}
    status = forecast.get("status")

    sections = ["## Perché il modello produce questo scenario\n\n"]
    method = explanation.get("method")
    if method == "temporal_occlusion":
        sections.append(_occlusion_markdown(explanation))
    elif method == "analytical_fallback":
        sections.append(_fallback_markdown(explanation))
    elif status == "unavailable":
        sections.append(
            "Non ci sono dati sufficienti per attribuire la previsione a fattori "
            "specifici.\n"
        )
    else:
        sections.append(
            "L'endpoint di previsione non ha restituito le prove controfattuali "
            "necessarie per un'attribuzione locale. Sono quindi riportati solo i "
            "fattori di rischio osservabili, senza inventare importanze del modello.\n"
        )

    if prices:
        sections.extend(["\n### Rischio e affidabilità\n\n", _risk_context(prices, forecast), "\n"])
    sections.append(
        "\nLa spiegazione descrive la sensibilità del modello allo storico disponibile; "
        "non dimostra che quei periodi siano la causa dei movimenti futuri.\n"
    )
    return {"xai_analysis": "".join(sections)}
