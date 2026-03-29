"""
Trading strategy tools:
- PortfolioStateTool — reads the current paper portfolio state for the strategy agent
"""

import json
from typing import Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..portfolio_store import load_portfolio, get_portfolio_summary


class PortfolioStateInput(BaseModel):
    symbol: str = Field(description="Crypto symbol e.g. BTC, ETH")
    current_price: float = Field(
        default=0.0,
        description="Current market price in USD (used to mark-to-market any open position). "
                    "Pass 0 to skip mark-to-market.",
    )


def _fetch_current_price(symbol: str) -> float:
    """Fetch live price from Binance public API as fallback."""
    binance_symbol = symbol.upper()
    if not binance_symbol.endswith("USDT"):
        binance_symbol += "USDT"
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": binance_symbol},
            timeout=5,
        )
        data = resp.json()
        return float(data["price"]) if "price" in data else 0.0
    except Exception:
        return 0.0


class PortfolioStateTool(BaseTool):
    name: str = "get_portfolio_state"
    description: str = (
        "Read the current paper trading portfolio state for a given symbol. "
        "Returns: starting balance ($10,000), current total value, cash available, "
        "open position details (if any: direction, entry price, stop loss, take profit, "
        "size in USD and coins), last 3 closed trades with P&L, overall win rate, "
        "total P&L since inception, and the equity curve history. "
        "Use this before proposing a trade to understand available capital and "
        "whether an existing position is already open."
    )
    args_schema: Type[BaseModel] = PortfolioStateInput

    def _run(self, symbol: str, current_price: float = 0.0) -> str:
        if current_price <= 0:
            current_price = _fetch_current_price(symbol)

        portfolio = load_portfolio(symbol)
        summary   = get_portfolio_summary(portfolio, current_price)
        return json.dumps(summary, indent=2, default=str)
