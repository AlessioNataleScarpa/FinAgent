"""Retrieve ETF composition from issuer data discovered through web search.

Composition deliberately does not use Yahoo Finance.  The first supported
provider is iShares/BlackRock, whose official holdings export contains sector,
asset-class, geography and portfolio weights.  Unsupported issuers fail closed:
the UI reports that composition is unavailable instead of drawing invented
allocations.
"""

from __future__ import annotations

import csv
import html
import io
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx

logger = logging.getLogger(__name__)

SEARCH_URL = "https://html.duckduckgo.com/html/"
BING_RSS_URL = "https://www.bing.com/search"
ISHARES_EXPORT_ID = "1506575576011.ajax"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_CACHE: Dict[str, tuple[float, Dict[str, Any]]] = {}


def _timeout() -> float:
    try:
        return max(2.0, min(float(os.getenv("COMPOSITION_TIMEOUT_SECONDS", "35")), 60.0))
    except ValueError:
        return 35.0


def _cache_ttl() -> float:
    try:
        return max(60.0, float(os.getenv("COMPOSITION_CACHE_TTL_SECONDS", "21600")))
    except ValueError:
        return 21600.0


def _decode_search_href(href: str) -> str:
    decoded = html.unescape(href or "")
    if decoded.startswith("//"):
        decoded = "https:" + decoded
    parsed = urlparse(decoded)
    redirect_target = parse_qs(parsed.query).get("uddg")
    return unquote(redirect_target[0]) if redirect_target else decoded


def _search_result_links(page: str) -> Iterable[str]:
    pattern = re.compile(
        r'class=["\']result__a["\'][^>]*href=["\']([^"\']+)',
        re.IGNORECASE,
    )
    for match in pattern.finditer(page or ""):
        url = _decode_search_href(match.group(1))
        if url.startswith(("http://", "https://")):
            yield url


def _rss_result_links(page: str) -> Iterable[str]:
    try:
        root = ET.fromstring(page)
    except ET.ParseError:
        return
    for element in root.findall(".//item/link"):
        url = (element.text or "").strip()
        if url.startswith(("http://", "https://")):
            yield url


def _discover_ishares_product(client: httpx.Client, isin: str) -> str:
    query = f"site:ishares.com {isin} iShares product"
    links = []
    try:
        response = client.get(
            SEARCH_URL,
            params={"q": query},
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        links.extend(_search_result_links(response.text))
    except Exception as exc:
        logger.info("Primary composition search unavailable: %s", exc)

    # The HTML endpoint can return an anti-bot page with HTTP 202 and no
    # results. RSS search is a compact fallback and does not require page
    # rendering or browser automation.
    if not links:
        response = client.get(
            BING_RSS_URL,
            params={"format": "rss", "q": query},
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        links.extend(_rss_result_links(response.text))

    candidates = []
    for url in links:
        parsed = urlparse(url)
        if "ishares.com" not in parsed.netloc.lower():
            continue
        if not re.search(r"/products/\d+/", parsed.path, re.IGNORECASE):
            continue
        score = 0
        if "/uk/individual/en/" in parsed.path:
            score += 4
        if "/professional/en/" in parsed.path:
            score += 2
        candidates.append((score, parsed._replace(query="", fragment="").geturl()))

    if not candidates:
        raise LookupError("nessuna pagina ufficiale iShares trovata dalla ricerca")

    candidates.sort(key=lambda item: item[0], reverse=True)
    product_url = candidates[0][1]
    product_page = client.get(
        product_url,
        params={"siteEntryPassthrough": "true", "switchLocale": "y"},
        headers={"User-Agent": USER_AGENT},
    )
    product_page.raise_for_status()
    if isin.upper() not in product_page.text.upper():
        raise ValueError("la pagina trovata non corrisponde all'ISIN richiesto")
    return str(product_page.url).split("?", 1)[0]


def _holdings_url(product_url: str) -> str:
    return (
        f"{product_url.rstrip('/')}/{ISHARES_EXPORT_ID}"
        "?fileType=csv&fileName=holdings&dataType=fund"
    )


def _csv_rows(raw: bytes) -> tuple[str, list[Dict[str, str]]]:
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "Ticker" in line
            and "Sector" in line
            and "Asset Class" in line
            and "Weight (%)" in line
        ),
        -1,
    )
    if header_index < 0:
        raise ValueError("intestazione del file holdings non riconosciuta")

    as_of = ""
    if header_index:
        preamble = next(csv.reader([lines[0]]), [])
        if len(preamble) > 1:
            as_of = preamble[1].strip()
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    return as_of, [dict(row) for row in reader]


def _number(value: Any) -> float:
    text = str(value or "").strip().replace("\xa0", "")
    if not text:
        raise ValueError("empty number")
    if "," in text and "." in text:
        text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    return float(text)


def _aggregate(rows: Iterable[Mapping[str, Any]], field: str) -> Dict[str, float]:
    totals: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        label = str(row.get(field) or "").strip()
        if not label or label in {"-", "N/A"}:
            label = "Altro / non classificato"
        try:
            weight = _number(row.get("Weight (%)"))
        except (TypeError, ValueError):
            continue
        if 0 < weight < 150:
            totals[label] += weight
    return {
        label: round(weight, 4)
        for label, weight in totals.items()
        if weight >= 0.005
    }


def _compact(weights: Mapping[str, float], *, limit: int) -> Dict[str, float]:
    ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    if len(ordered) <= limit:
        return dict(ordered)
    kept = ordered[:limit]
    remainder = sum(value for _, value in ordered[limit:])
    if remainder > 0:
        kept.append(("Altri", remainder))
    return {label: round(value, 4) for label, value in kept}


def _parse_ishares_holdings(
    raw: bytes,
    *,
    isin: str,
    product_url: str,
    data_url: str,
) -> Dict[str, Any]:
    as_of, rows = _csv_rows(raw)
    if not rows:
        raise ValueError("il file holdings non contiene posizioni")

    sectors = _compact(_aggregate(rows, "Sector"), limit=10)
    assets = _compact(_aggregate(rows, "Asset Class"), limit=8)
    geography = _compact(_aggregate(rows, "Location"), limit=10)
    if not sectors and not assets and not geography:
        raise ValueError("nessun peso di composizione valido nel file holdings")

    return {
        "status": "ok",
        "provider": "iShares / BlackRock",
        "source_type": "official_holdings",
        "source_url": product_url,
        "data_url": data_url,
        "as_of": as_of,
        "isin": isin,
        "holdings_count": len(rows),
        "sector_weights": sectors,
        "asset_allocation": assets,
        "geography_weights": geography,
    }


def get_fund_composition(isin: str) -> Dict[str, Any]:
    """Search the web and aggregate official holdings for an ETF ISIN."""
    key = (isin or "").strip().upper()
    if not re.fullmatch(r"[A-Z]{2}[A-Z0-9]{9}[0-9]", key):
        return {
            "status": "unavailable",
            "provider": "web search",
            "error": "ISIN non valido o mancante",
        }

    cached = _CACHE.get(key)
    if cached and time.monotonic() - cached[0] < _cache_ttl():
        return dict(cached[1])

    try:
        with httpx.Client(timeout=_timeout(), follow_redirects=True) as client:
            product_url = _discover_ishares_product(client, key)
            data_url = _holdings_url(product_url)
            response = client.get(data_url, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()
            result = _parse_ishares_holdings(
                response.content,
                isin=key,
                product_url=product_url,
                data_url=data_url,
            )
    except Exception as exc:
        logger.warning("Official composition lookup failed for %s: %s", key, exc)
        result = {
            "status": "unavailable",
            "provider": "official issuer search",
            "error": str(exc),
            "isin": key,
        }

    _CACHE[key] = (time.monotonic(), result)
    return dict(result)
