import json

from agents.compositionChartAgent import CompositionChartAgent
from api.composition_search import (
    _decode_search_href,
    _parse_ishares_holdings,
    _rss_result_links,
    _search_result_links,
)


SAMPLE_HOLDINGS = b"""Fund Holdings as of,"24/Jul/2026"

Ticker,Name,Sector,Asset Class,Market Value,Weight (%),Location
AAA,Alpha,Information Technology,Equity,1000,40.0,United States
BBB,Beta,Financials,Equity,800,30.0,United Kingdom
CCC,Gamma,Healthcare,Equity,700,25.0,Japan
-,Cash,Altro / non classificato,Cash,50,5.0,Ireland
"""


def test_search_redirect_is_decoded():
    encoded = (
        "//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.ishares.com%2Fuk%2F"
        "individual%2Fen%2Fproducts%2F12345%2Fexample"
    )
    assert _decode_search_href(encoded) == (
        "https://www.ishares.com/uk/individual/en/products/12345/example"
    )


def test_search_result_parser_only_returns_links():
    page = (
        '<a class="result__a" href="https://example.test/fund">Fund</a>'
        '<a class="other" href="https://ignored.test">Ignored</a>'
    )
    assert list(_search_result_links(page)) == ["https://example.test/fund"]


def test_rss_result_parser_returns_item_links():
    feed = (
        "<rss><channel><item><link>https://www.ishares.com/uk/individual/en/"
        "products/12345/example</link></item></channel></rss>"
    )
    assert list(_rss_result_links(feed)) == [
        "https://www.ishares.com/uk/individual/en/products/12345/example"
    ]


def test_official_holdings_are_aggregated_without_fixed_98_2():
    result = _parse_ishares_holdings(
        SAMPLE_HOLDINGS,
        isin="IE00TEST0001",
        product_url="https://www.ishares.com/products/12345/fund",
        data_url="https://www.ishares.com/products/12345/fund/holdings.csv",
    )
    assert result["status"] == "ok"
    assert result["sector_weights"]["Information Technology"] == 40.0
    assert result["asset_allocation"] == {"Equity": 95.0, "Cash": 5.0}
    assert result["geography_weights"]["United States"] == 40.0
    assert result["holdings_count"] == 4


def test_composition_charts_use_sector_and_geography_and_hide_dominant_asset_pie():
    info = json.dumps(
        {
            "Composition": {
                "status": "ok",
                "provider": "Issuer",
                "source_url": "https://issuer.test/fund",
                "as_of": "24/Jul/2026",
                "sector_weights": {
                    "Technology": 40,
                    "Financials": 30,
                    "Healthcare": 30,
                },
                "geography_weights": {
                    "United States": 55,
                    "Europe": 30,
                    "Japan": 15,
                },
                "asset_allocation": {"Equity": 98, "Cash": 2},
            }
        }
    )
    output = CompositionChartAgent().build_markdown("IE00TEST0001", info)
    assert "Ripartizione settoriale" in output
    assert "Ripartizione geografica" in output
    assert "quasi monocromatica" in output
    assert 'title Classi di attivo' not in output
    assert output.count("```mermaid") == 2


def test_missing_official_data_never_creates_placeholder_percentages():
    info = json.dumps(
        {"Composition": {"status": "unavailable", "error": "not found"}}
    )
    output = CompositionChartAgent().build_markdown("IE00TEST0001", info)
    assert "fonte ufficiale verificabile" in output
    assert "98" not in output
    assert "```mermaid" not in output
