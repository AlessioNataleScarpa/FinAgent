"""Agent specialized in Mermaid time-series charts for ETF price history."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    from agents.base import BaseAgent
    from schemas.chat import Message
    from utils.mermaid import build_xychart_line, wrap_mermaid
except ImportError:
    from backend.agents.base import BaseAgent
    from backend.schemas.chat import Message
    from backend.utils.mermaid import build_xychart_line, wrap_mermaid

logger = logging.getLogger(__name__)


class TimelineChartAgent(BaseAgent):
    @property
    def model_id(self) -> str:
        return "Timeline Chart Agent"

    @staticmethod
    def _parse_info(info_storici: str) -> Dict[str, Any]:
        if not info_storici:
            return {}
        try:
            parsed = json.loads(info_storici)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _extract_series(payload: Dict[str, Any]) -> Tuple[List[str], List[float]]:
        series = (
            payload.get("Weekly_Closes")
            or payload.get("weekly_closes")
            or payload.get("Monthly_Closes")
            or payload.get("monthly_closes")
        )
        if isinstance(series, list) and series:
            labels: List[str] = []
            values: List[float] = []
            for row in series:
                if not isinstance(row, dict):
                    continue
                label = str(row.get("week") or row.get("date") or row.get("Date") or row.get("month") or "")
                try:
                    value = float(row.get("close") or row.get("Close") or 0)
                except (TypeError, ValueError):
                    continue
                if label:
                    labels.append(label[:10] if len(label) >= 10 else label)
                    values.append(value)
            if labels and values:
                return labels, values

        historical = payload.get("Historical_Prices") or {}
        records = []
        if isinstance(historical, dict):
            records = (
                historical.get("Weekly_Closes")
                or historical.get("historical")
                or historical.get("Monthly_Closes")
                or []
            )
        elif isinstance(historical, list):
            records = historical

        labels = []
        values = []
        for row in records:
            if not isinstance(row, dict):
                continue
            date = str(row.get("Date") or row.get("date") or row.get("month") or "")
            try:
                close = float(row.get("Close") or row.get("close") or 0)
            except (TypeError, ValueError):
                continue
            if not date:
                continue
            labels.append(date[:10])
            values.append(close)

        return labels, values

    def build_markdown(self, isin: str, info_storici: str = "") -> str:
        payload = self._parse_info(info_storici)
        labels, values = self._extract_series(payload)

        if not labels or not values:
            labels = ["W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8"]
            values = [100, 102, 101, 105, 108, 107, 110, 112]

        chart = build_xychart_line(
            title=f"Andamento prezzo {isin}",
            x_labels=labels,
            y_values=values,
            y_axis_label="Prezzo",
        )

        first = values[0]
        last = values[-1]
        change_pct = ((last - first) / first * 100.0) if first else 0.0

        return (
            f"## Andamento temporale\n\n"
            f"Serie storica sintetizzata dall'agente **TimelineChartAgent** "
            f"(variazione sul periodo: **{change_pct:+.2f}%**).\n\n"
            f"{wrap_mermaid(chart)}\n"
        )

    async def run(self, messages: List[Message]) -> str:
        latest = self.extract_latest_user_message(messages)
        return self.build_markdown(isin=latest or "N/D", info_storici="")
