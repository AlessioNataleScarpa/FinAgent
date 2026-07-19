"""Agent specialized in Mermaid pie charts for asset/sector composition."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Mapping, Optional

try:
    from agents.base import BaseAgent
    from schemas.chat import Message
    from utils.mermaid import (
        build_pie_chart,
        default_asset_slices,
        default_sector_slices,
        wrap_mermaid,
    )
except ImportError:
    from backend.agents.base import BaseAgent
    from backend.schemas.chat import Message
    from backend.utils.mermaid import (
        build_pie_chart,
        default_asset_slices,
        default_sector_slices,
        wrap_mermaid,
    )

logger = logging.getLogger(__name__)


class CompositionChartAgent(BaseAgent):
    @property
    def model_id(self) -> str:
        return "Composition Chart Agent"

    @staticmethod
    def _parse_info(info_presentazione: str) -> Dict[str, Any]:
        if not info_presentazione:
            return {}
        try:
            parsed = json.loads(info_presentazione)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {"raw": info_presentazione}

    @staticmethod
    def _normalize_weights(raw: Any) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        if isinstance(raw, Mapping):
            for key, value in raw.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    match = re.search(r"[-+]?\d*\.?\d+", str(value))
                    if not match:
                        continue
                    numeric = float(match.group(0))
                if numeric > 0:
                    weights[str(key)] = numeric
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, Mapping):
                    continue
                label = item.get("sector") or item.get("name") or item.get("label")
                value = item.get("weight") or item.get("pct") or item.get("percentage")
                if label is None or value is None:
                    continue
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if numeric > 0:
                    weights[str(label)] = numeric
        return weights

    def build_markdown(self, isin: str, info_presentazione: str = "") -> str:
        payload = self._parse_info(info_presentazione)
        profile = payload.get("Profile") if isinstance(payload.get("Profile"), dict) else {}
        if not profile and isinstance(payload, dict):
            profile = payload

        sector_weights = self._normalize_weights(
            profile.get("sectorWeightings")
            or profile.get("sector_weights")
            or profile.get("categoryWeightings")
        )
        asset_weights = self._normalize_weights(
            profile.get("assetAllocation")
            or profile.get("asset_allocation")
            or profile.get("holdings")
        )

        if not sector_weights:
            sector_weights = default_sector_slices()
        if not asset_weights:
            asset_weights = default_asset_slices()

        sector_pie = build_pie_chart(f"Composizione settoriale {isin}", sector_weights)
        asset_pie = build_pie_chart(f"Composizione asset {isin}", asset_weights)

        return (
            f"## Composizione del portafoglio\n\n"
            f"Grafici Mermaid generati dall'agente specializzato **CompositionChartAgent** "
            f"sull'ISIN `{isin}`.\n\n"
            f"### Allocazione settoriale\n\n"
            f"{wrap_mermaid(sector_pie)}\n\n"
            f"### Allocazione per classe di asset\n\n"
            f"{wrap_mermaid(asset_pie)}\n"
        )

    async def run(self, messages: List[Message]) -> str:
        latest = self.extract_latest_user_message(messages)
        return self.build_markdown(isin=latest or "N/D", info_presentazione="")
