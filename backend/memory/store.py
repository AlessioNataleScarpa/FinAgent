"""Persistent ETF analysis memory for follow-up conversation agents."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "etf_memory.json"
_lock = threading.Lock()
_store_singleton: Optional["EtfMemoryStore"] = None


class EtfMemoryStore:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else _DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = {"latest_isin": None, "analyses": {}}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self._data["latest_isin"] = loaded.get("latest_isin")
                analyses = loaded.get("analyses") or {}
                if isinstance(analyses, dict):
                    self._data["analyses"] = analyses
        except Exception as exc:
            logger.warning("Unable to load ETF memory from %s: %s", self.path, exc)

    def _save(self) -> None:
        try:
            self.path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Unable to persist ETF memory to %s: %s", self.path, exc)

    def save_analysis(
        self,
        isin: str,
        *,
        report: str,
        presentation: str = "",
        technical: str = "",
        composition_charts: str = "",
        timeline_charts: str = "",
        sentiment_charts: str = "",
        info_presentazione: str = "",
        info_storici: str = "",
        news_data: str = "",
        prediction: str = "",
        clean_query: str = "",
    ) -> Dict[str, Any]:
        key = (isin or "").strip().upper()
        if not key:
            raise ValueError("ISIN is required to save analysis memory")

        payload = {
            "isin": key,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "clean_query": clean_query,
            "report": report,
            "presentation": presentation,
            "technical": technical,
            "composition_charts": composition_charts,
            "timeline_charts": timeline_charts,
            "sentiment_charts": sentiment_charts,
            "info_presentazione": info_presentazione,
            "info_storici": info_storici,
            "news_data": news_data,
            "prediction": prediction,
        }

        with _lock:
            self._data["analyses"][key] = payload
            self._data["latest_isin"] = key
            self._save()
        return payload

    def get(self, isin: str) -> Optional[Dict[str, Any]]:
        key = (isin or "").strip().upper()
        if not key:
            return None
        with _lock:
            analysis = self._data["analyses"].get(key)
            return dict(analysis) if analysis else None

    def get_latest(self) -> Optional[Dict[str, Any]]:
        with _lock:
            latest = self._data.get("latest_isin")
            if not latest:
                return None
            analysis = self._data["analyses"].get(latest)
            return dict(analysis) if analysis else None

    def context_blob(self, isin: Optional[str] = None, *, max_chars: int = 12000) -> str:
        analysis = self.get(isin) if isin else self.get_latest()
        if not analysis:
            return ""

        # Keep conversation prompts lean: report + short extras only.
        report = (analysis.get("report") or "")[:8000]
        news = (analysis.get("news_data") or "")[:1500]
        prediction = (analysis.get("prediction") or "")[:1000]

        sections = [
            f"ISIN: {analysis.get('isin')}",
            f"Aggiornato: {analysis.get('updated_at')}",
            "",
            "=== REPORT FINALE ===",
            report,
            "",
            "=== NEWS ===",
            news,
            "",
            "=== PREVISIONE ===",
            prediction,
        ]
        blob = "\n".join(sections).strip()
        if len(blob) > max_chars:
            return blob[:max_chars].rstrip() + "\n\n[...contesto troncato...]"
        return blob


def get_memory_store() -> EtfMemoryStore:
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = EtfMemoryStore()
    return _store_singleton
