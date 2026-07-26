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
        wrap_mermaid,
    )
except ImportError:
    from backend.agents.base import BaseAgent
    from backend.schemas.chat import Message
    from backend.utils.mermaid import (
        build_pie_chart,
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

    @staticmethod
    def _as_percentages(weights: Mapping[str, float]) -> Dict[str, float]:
        total = sum(float(value) for value in weights.values() if float(value) > 0)
        if total <= 0:
            return {}
        return {
            label: round(float(value) / total * 100.0, 2)
            for label, value in weights.items()
            if float(value) > 0
        }

    @staticmethod
    def _is_informative(weights: Mapping[str, float]) -> bool:
        values = sorted((float(value) for value in weights.values() if value > 0), reverse=True)
        return len(values) >= 2 and values[0] < 95.0

    def build_markdown(self, isin: str, info_presentazione: str = "") -> str:
        payload = self._parse_info(info_presentazione)
        composition = (
            payload.get("Composition")
            if isinstance(payload.get("Composition"), dict)
            else {}
        )
        if composition.get("status") != "ok":
            return (
                "## Composizione del portafoglio\n\n"
                "La composizione non è disponibile da una fonte ufficiale verificabile. "
                "Il grafico non viene mostrato per evitare percentuali stimate o "
                "allocazioni predefinite.\n"
            )

        sectors = self._as_percentages(
            self._normalize_weights(composition.get("sector_weights"))
        )
        assets = self._as_percentages(
            self._normalize_weights(composition.get("asset_allocation"))
        )
        geography = self._as_percentages(
            self._normalize_weights(composition.get("geography_weights"))
        )

        sections = ["## Composizione del portafoglio\n\n"]
        source_name = composition.get("provider") or "emittente"
        source_url = composition.get("source_url") or ""
        as_of = composition.get("as_of") or "data non indicata"
        if source_url:
            sections.append(
                f"Dati delle posizioni pubblicati da [{source_name}]({source_url}), "
                f"aggiornati al **{as_of}**.\n\n"
            )
        else:
            sections.append(
                f"Dati delle posizioni pubblicati da **{source_name}**, "
                f"aggiornati al **{as_of}**.\n\n"
            )

        if self._is_informative(sectors):
            sections.extend(
                [
                    "### Ripartizione settoriale\n\n",
                    wrap_mermaid(
                        build_pie_chart(f"Settori - {isin}", sectors)
                    ),
                    "\n\n",
                ]
            )
        if self._is_informative(geography):
            sections.extend(
                [
                    "### Ripartizione geografica\n\n",
                    wrap_mermaid(
                        build_pie_chart(f"Aree geografiche - {isin}", geography)
                    ),
                    "\n\n",
                ]
            )
        if self._is_informative(assets):
            sections.extend(
                [
                    "### Ripartizione per classe di attivo\n\n",
                    wrap_mermaid(
                        build_pie_chart(f"Classi di attivo - {isin}", assets)
                    ),
                    "\n",
                ]
            )
        elif assets:
            dominant_label, dominant_weight = max(assets.items(), key=lambda item: item[1])
            sections.append(
                f"La classe di attivo è concentrata in **{dominant_label} "
                f"({dominant_weight:.2f}%)**. Una torta quasi monocromatica non "
                "aggiungerebbe informazione, quindi non viene disegnata.\n"
            )

        if len(sections) <= 2:
            sections.append(
                "Le posizioni sono state recuperate, ma non contengono almeno due "
                "categorie con pesi sufficienti per un grafico attendibile.\n"
            )
        return "".join(sections)

    async def run(self, messages: List[Message]) -> str:
        latest = self.extract_latest_user_message(messages)
        return self.build_markdown(isin=latest or "N/D", info_presentazione="")
