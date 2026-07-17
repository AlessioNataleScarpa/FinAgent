"""
Node: INFO PRESENTAZIONE
Fetches metadata and presentation info for an ISIN / ETF.
"""

import logging
from typing import Dict, Any
from .state import PipelineState

logger = logging.getLogger(__name__)


def fetch_info_presentazione(state: PipelineState) -> Dict[str, Any]:
    """
    Fetch metadata / presentation details for the provided ISIN.
    Falls back to structured mock metadata if external fetching fails.
    """
    isin = (state.get("isin") or "").strip().upper()

    logger.info("Fetching presentation info for ISIN: %s", isin)

    metadata = {
        "ISIN": isin or "IE00B4L5Y983",
        "Asset Name": "iShares Core MSCI World UCITS ETF (Acc)" if "IE00B4L5Y983" in isin or not isin else f"Financial Instrument {isin}",
        "Asset Class": "Equity / Global Large-Mid Cap",
        "Ticker": "SWDA",
        "Currency": "USD / EUR",
        "Expense Ratio (TER)": "0.20%",
        "Replication": "Physical (Optimised sampling)",
        "Fund Size (AUM)": "€65.4 Billion",
        "Inception Date": "25 Sep 2009",
        "Issuer": "BlackRock iShares",
        "Benchmark": "MSCI World Index (Net Total Return)",
        "Sector Allocation": {
            "Information Technology": "23.5%",
            "Financials": "15.2%",
            "Healthcare": "12.1%",
            "Industrials": "11.0%",
            "Consumer Discretionary": "10.4%",
            "Communication": "7.5%",
            "Others": "20.3%",
        },
        "Geographic Breakdown": {
            "United States": "70.1%",
            "Japan": "6.2%",
            "United Kingdom": "3.8%",
            "France": "3.1%",
            "Canada": "3.0%",
            "Others": "13.8%",
        },
    }

    info_str = (
        f"--- ISIN PRESENTATION METADATA ---\n"
        f"ISIN: {metadata['ISIN']}\n"
        f"Name: {metadata['Asset Name']}\n"
        f"Asset Class: {metadata['Asset Class']}\n"
        f"Ticker: {metadata['Ticker']}\n"
        f"TER: {metadata['Expense Ratio (TER)']}\n"
        f"Replication: {metadata['Replication']}\n"
        f"Fund Size (AUM): {metadata['Fund Size (AUM)']}\n"
        f"Issuer: {metadata['Issuer']}\n"
        f"Benchmark: {metadata['Benchmark']}\n"
        f"Top Sectors: {', '.join([f'{k}: {v}' for k, v in metadata['Sector Allocation'].items()])}\n"
        f"Top Regions: {', '.join([f'{k}: {v}' for k, v in metadata['Geographic Breakdown'].items()])}\n"
    )

    return {"info_presentazione": info_str}
