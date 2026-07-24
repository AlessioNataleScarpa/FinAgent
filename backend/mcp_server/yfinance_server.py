import logging
from typing import Any, Dict, List
import tempfile
import yfinance as yf
from mcp.server.fastmcp import FastMCP

try:
    yf.set_tz_cache_location(tempfile.gettempdir())
except Exception:
    pass

logger = logging.getLogger(__name__)

mcp = FastMCP("YahooFinanceServer")


def _extract_sector_weights(info: Dict[str, Any], ticker_obj: yf.Ticker) -> Dict[str, float]:
    candidates = [
        info.get("sectorWeightings"),
        info.get("categoryWeightings"),
        info.get("sectorWeighting"),
    ]

    try:
        funds = getattr(ticker_obj, "funds_data", None)
        if funds is not None:
            sector = getattr(funds, "sector_weightings", None)
            if isinstance(sector, dict):
                candidates.append(sector)
    except Exception:
        pass

    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        weights: Dict[str, float] = {}
        for key, value in raw.items():
            try:
                if isinstance(value, dict):
                    numeric = float(next(iter(value.values())))
                else:
                    numeric = float(value)
                if 0 < numeric <= 1:
                    numeric *= 100.0
                if numeric > 0:
                    weights[str(key)] = round(numeric, 2)
            except (TypeError, ValueError, StopIteration):
                continue
        if weights:
            return weights
    return {}


def _extract_asset_allocation(info: Dict[str, Any]) -> Dict[str, float]:
    raw = info.get("assetAllocation") or info.get("holdings")
    weights: Dict[str, float] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                numeric = float(value)
                if 0 < numeric <= 1:
                    numeric *= 100.0
                if numeric > 0:
                    weights[str(key)] = round(numeric, 2)
            except (TypeError, ValueError):
                continue
    quote_type = str(info.get("quoteType") or info.get("typeDisp") or "").upper()
    if not weights and ("ETF" in quote_type or "FUND" in quote_type or info.get("longBusinessSummary")):
        weights = {"Azioni / Equity": 98.0, "Liquidita / Altro": 2.0}
    return weights


@mcp.tool()
def get_profile(ticker: str) -> Dict[str, Any]:
    """Recupera il profilo aziendale o dell'ETF tramite yfinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        if info:
            return {
                "name": info.get("shortName", info.get("longName")),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "description": info.get("longBusinessSummary", "N/A"),
                "marketCap": info.get("marketCap", "N/A"),
                "currency": info.get("currency", "N/A"),
                "exchange": info.get("exchange", "N/A"),
                "previousClose": info.get("previousClose", "N/A"),
                "yield": info.get("yield", "N/A"),
                "category": info.get("category", "N/A"),
                "totalAssets": info.get("totalAssets", "N/A"),
                "sectorWeightings": _extract_sector_weights(info, t),
                "assetAllocation": _extract_asset_allocation(info),
            }
        return {"error": "Profile not found"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_historical_data(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """Recupera OHLCV mensile compatto, adatto a grafici e inferenza TSFM."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval="1mo")
        if hist.empty:
            hist = t.history(period=period)

        if hist.empty:
            return {"error": "Historical data not found"}

        hist = hist.copy()
        hist.reset_index(inplace=True)
        hist["Date"] = hist["Date"].astype(str)

        monthly = (
            hist.assign(month=hist["Date"].astype(str).str.slice(0, 7))
            .groupby("month", as_index=False)
            .agg(
                open=("Open", "first"),
                high=("High", "max"),
                low=("Low", "min"),
                close=("Close", "last"),
                volume=("Volume", "sum"),
            )
        )
        monthly_ohlcv: List[Dict[str, Any]] = [
            {
                "month": row["month"],
                "open": round(float(row["open"]), 4),
                "high": round(float(row["high"]), 4),
                "low": round(float(row["low"]), 4),
                "close": round(float(row["close"]), 4),
                "volume": int(row["volume"]),
            }
            for _, row in monthly.iterrows()
        ]

        return {
            "Monthly_OHLCV": monthly_ohlcv,
            # Backward-compatible projection consumed by existing chart code.
            "Monthly_Closes": [
                {"month": row["month"], "close": row["close"]}
                for row in monthly_ohlcv
            ],
        }
    except Exception as e:
        return {"error": str(e)}


def _normalise_news(raw_items: List[Any], limit: int) -> List[Dict[str, Any]]:
    articles: List[Dict[str, Any]] = []
    seen = set()
    for raw in raw_items:
        content = raw.get("content") if isinstance(raw, dict) else None
        item = content if isinstance(content, dict) else raw
        if not isinstance(item, dict):
            continue
        provider = item.get("provider")
        if isinstance(provider, dict):
            provider = provider.get("displayName")
        canonical = item.get("canonicalUrl") or item.get("clickThroughUrl")
        if isinstance(canonical, dict):
            canonical = canonical.get("url")
        title = str(item.get("title") or "Senza titolo").strip()
        url = canonical or item.get("link") or ""
        identity = (title.lower(), str(url))
        if identity in seen:
            continue
        seen.add(identity)
        articles.append(
            {
                "title": title,
                "summary": item.get("summary") or item.get("description") or "",
                "published_at": item.get("pubDate") or item.get("providerPublishTime"),
                "publisher": provider or item.get("publisher") or "",
                "url": url,
            }
        )
        if len(articles) >= limit:
            break
    return articles


@mcp.tool()
def get_news(ticker: str, limit: int = 8, query: str = "") -> Dict[str, Any]:
    """Recupera le notizie Yahoo Finance più recenti per lo strumento."""
    try:
        safe_limit = max(1, min(limit, 20))
        raw_items = yf.Ticker(ticker).news or []
        articles = _normalise_news(raw_items, safe_limit)

        # ETF ticker feeds are often empty. Yahoo Search uses a different
        # endpoint and can retrieve instrument/market news for ticker or ISIN.
        for search_query in dict.fromkeys([ticker, query]):
            if len(articles) >= safe_limit or not search_query:
                break
            try:
                search_items = yf.Search(
                    search_query,
                    news_count=safe_limit,
                ).news or []
                existing_keys = {
                    (existing["title"], existing["url"])
                    for existing in articles
                }
                articles.extend(
                    article
                    for article in _normalise_news(search_items, safe_limit)
                    if (article["title"], article["url"]) not in existing_keys
                )
                articles = articles[:safe_limit]
            except Exception as exc:
                logger.warning("Yahoo Search failed for %s: %s", search_query, exc)

        if not articles:
            return {"error": "News not found", "articles": []}
        return {"articles": articles, "search_fallback_used": not bool(raw_items)}
    except Exception as e:
        return {"error": str(e), "articles": []}


if __name__ == "__main__":
    mcp.run(transport="stdio")
