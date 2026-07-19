import asyncio
import concurrent.futures
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

SERVER_SCRIPT = Path(__file__).parent.parent / "mcp_server" / "yfinance_server.py"


async def _call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Esegue una chiamata a un tool fornito dal server MCP Yahoo Finance.
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        env=None,
    )
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                if hasattr(result, "structuredContent") and result.structuredContent and "result" in result.structuredContent:
                    return result.structuredContent["result"]
                if result.content:
                    for content in result.content:
                        if hasattr(content, "text") and content.text:
                            try:
                                return json.loads(content.text)
                            except Exception:
                                return {"text": content.text}
                return {}
    except Exception as e:
        logger.error("Errore durante la chiamata MCP tool %s per %s: %s", tool_name, arguments, e)
        return {"error": str(e)}


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


def get_profile(ticker: str) -> Dict[str, Any]:
    """
    Recupera il profilo aziendale o dell'ETF delegando la chiamata al server MCP.
    """
    return _run_async(_call_mcp_tool("get_profile", {"ticker": ticker}))


def get_historical_data(ticker: str, period: str = "1y") -> Dict[str, Any]:
    """
    Recupera le serie storiche mensili delegando la chiamata al server MCP.
    """
    return _run_async(_call_mcp_tool("get_historical_data", {"ticker": ticker, "period": period}))
