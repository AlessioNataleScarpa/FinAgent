"""Lightweight web search helpers for ConversationAgent enrichment."""

from __future__ import annotations

import logging
from typing import List
from urllib.parse import quote_plus

import httpx

logger = logging.getLogger(__name__)


def search_web(query: str, *, max_results: int = 5) -> str:
    """
    Best-effort web search via DuckDuckGo Instant Answer API.
    Returns a compact Markdown block, or an empty string on failure.
    """
    q = (query or "").strip()
    if not q:
        return ""

    try:
        url = f"https://api.duckduckgo.com/?q={quote_plus(q)}&format=json&no_html=1&skip_disambig=1"
        with httpx.Client(timeout=8.0, follow_redirects=True) as client:
            response = client.get(url, headers={"User-Agent": "DL-2026-ETF-Agent/1.0"})
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning("Web search failed for %r: %s", q, exc)
        return ""

    lines: List[str] = [f"### Risultati web per: {q}"]

    abstract = (data.get("AbstractText") or "").strip()
    abstract_url = (data.get("AbstractURL") or "").strip()
    heading = (data.get("Heading") or "").strip()
    if abstract:
        title = heading or "Sintesi"
        if abstract_url:
            lines.append(f"- **{title}**: {abstract} ({abstract_url})")
        else:
            lines.append(f"- **{title}**: {abstract}")

    related = data.get("RelatedTopics") or []
    count = 0
    for item in related:
        if count >= max_results:
            break
        if not isinstance(item, dict):
            continue
        if "Topics" in item and isinstance(item["Topics"], list):
            for nested in item["Topics"]:
                if count >= max_results:
                    break
                if not isinstance(nested, dict):
                    continue
                text = (nested.get("Text") or "").strip()
                link = (nested.get("FirstURL") or "").strip()
                if text:
                    lines.append(f"- {text}" + (f" ({link})" if link else ""))
                    count += 1
            continue
        text = (item.get("Text") or "").strip()
        link = (item.get("FirstURL") or "").strip()
        if text:
            lines.append(f"- {text}" + (f" ({link})" if link else ""))
            count += 1

    if len(lines) == 1:
        return ""
    return "\n".join(lines)
