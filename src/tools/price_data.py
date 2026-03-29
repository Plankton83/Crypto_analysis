"""
Price data tools using Binance public API and CoinGecko.
No API keys required for basic usage.
"""

import json
from datetime import datetime, timedelta
from typing import Optional, Type

import pandas as pd
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


BINANCE_BASE = "https://api.binance.com/api/v3"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"


def _symbol_to_coingecko_id(symbol: str) -> str:
    mapping = {
        "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
        "SOL": "solana", "XRP": "ripple", "ADA": "cardano",
        "DOGE": "dogecoin", "AVAX": "avalanche-2", "DOT": "polkadot",
        "MATIC": "matic-network", "LINK": "chainlink", "LTC": "litecoin",
        "UNI": "uniswap", "ATOM": "cosmos", "XLM": "stellar",
    }
    return mapping.get(symbol.upper(), symbol.lower())


class OHLCVInput(BaseModel):
    symbol: str = Field(description="Crypto symbol e.g. BTC, ETH")
    interval: str = Field(default="1d", description="Interval: 1m,5m,15m,1h,4h,1d,1w")
    lookback_days: int = Field(default=90, description="Number of days of historical data")


class OHLCVTool(BaseTool):
    name: str = "get_ohlcv_data"
    description: str = (
        "Fetch OHLCV (Open, High, Low, Close, Volume) historical price data "
        "from Binance for a cryptocurrency. Returns candlestick data as JSON."
    )
    args_schema: Type[BaseModel] = OHLCVInput

    def _run(self, symbol: str, interval: str = "1d", lookback_days: int = 90) -> str:
        symbol = symbol.upper()
        pair = f"{symbol}USDT"
        start_ms = int((datetime.utcnow() - timedelta(days=lookback_days)).timestamp() * 1000)

        params = {
            "symbol": pair,
            "interval": interval,
            "startTime": start_ms,
            "limit": 1000,
        }
        resp = requests.get(f"{BINANCE_BASE}/klines", params=params, timeout=10)
        resp.raise_for_status()

        cols = ["open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"]
        df = pd.DataFrame(resp.json(), columns=cols)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df[["open_time", "open", "high", "low", "close", "volume", "trades"]]
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df["trades"] = df["trades"].astype(int)

        summary = {
            "symbol": pair,
            "interval": interval,
            "records": len(df),
            "start": str(df["open_time"].iloc[0].date()),
            "end": str(df["open_time"].iloc[-1].date()),
            "current_price": df["close"].iloc[-1],
            "price_7d_ago": df["close"].iloc[-8] if len(df) > 8 else None,
            "price_30d_ago": df["close"].iloc[-31] if len(df) > 31 else None,
            "high_period": df["high"].max(),
            "low_period": df["low"].min(),
            "avg_volume": df["volume"].mean(),
            "recent_candles": df.tail(10)[["open_time", "open", "high", "low", "close", "volume"]].to_dict(orient="records"),
        }

        # Compute price change percentages
        if summary["price_7d_ago"]:
            summary["change_7d_pct"] = round(
                (summary["current_price"] - summary["price_7d_ago"]) / summary["price_7d_ago"] * 100, 2
            )
        if summary["price_30d_ago"]:
            summary["change_30d_pct"] = round(
                (summary["current_price"] - summary["price_30d_ago"]) / summary["price_30d_ago"] * 100, 2
            )

        return json.dumps(summary, default=str)


class MarketOverviewInput(BaseModel):
    symbol: str = Field(description="Crypto symbol e.g. BTC, ETH")


class MarketOverviewTool(BaseTool):
    name: str = "get_market_overview"
    description: str = (
        "Get current market overview for a cryptocurrency including market cap, "
        "dominance, circulating supply, and 24h stats from CoinGecko."
    )
    args_schema: Type[BaseModel] = MarketOverviewInput

    def _run(self, symbol: str) -> str:
        coin_id = _symbol_to_coingecko_id(symbol)
        params = {
            "ids": coin_id,
            "vs_currencies": "usd",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_24hr_change": "true",
            "include_last_updated_at": "true",
        }
        resp = requests.get(f"{COINGECKO_BASE}/simple/price", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json().get(coin_id, {})

        # Global market data
        global_resp = requests.get(f"{COINGECKO_BASE}/global", timeout=10)
        global_data = global_resp.json().get("data", {})

        result = {
            "symbol": symbol.upper(),
            "current_price_usd": data.get("usd"),
            "market_cap_usd": data.get("usd_market_cap"),
            "volume_24h_usd": data.get("usd_24h_vol"),
            "change_24h_pct": data.get("usd_24h_change"),
            "btc_dominance_pct": global_data.get("market_cap_percentage", {}).get("btc"),
            "eth_dominance_pct": global_data.get("market_cap_percentage", {}).get("eth"),
            "total_market_cap_usd": global_data.get("total_market_cap", {}).get("usd"),
            "total_volume_24h_usd": global_data.get("total_volume", {}).get("usd"),
            "active_cryptocurrencies": global_data.get("active_cryptocurrencies"),
        }

        # Stablecoin dominance — rising = capital on sidelines (bearish); falling = capital deploying (bullish)
        mcp = global_data.get("market_cap_percentage", {})
        stable_syms = ["usdt", "usdc", "busd", "dai", "tusd", "usdp", "gusd"]
        stable_dom = round(sum(mcp.get(s, 0) for s in stable_syms), 2)
        btc_dom = mcp.get("btc", 0) or 0
        result["stablecoin_dominance_pct"] = stable_dom
        result["stablecoin_signal"] = (
            "bearish - high stablecoin dominance (>12%), large capital on sidelines"
            if stable_dom > 12
            else "neutral - moderate stablecoin dominance (8-12%)"
            if stable_dom > 8
            else "bullish - low stablecoin dominance (<8%), capital deployed in crypto"
        )
        result["altcoin_season_indicator"] = (
            "altcoin season likely - BTC dominance low (<45%), risk appetite strong"
            if btc_dom < 45
            else "bitcoin dominance high (>60%) - capital concentrating in BTC"
            if btc_dom > 60
            else "transition phase - BTC dominance 45-60%"
        )

        return json.dumps(result, default=str)


class OrderBookInput(BaseModel):
    symbol: str = Field(description="Crypto symbol e.g. BTC, ETH")
    depth: int = Field(default=20, description="Order book depth (5, 10, 20, 50, 100, 500, 1000)")


class OrderBookTool(BaseTool):
    name: str = "get_order_book"
    description: str = (
        "Fetch current order book (bid/ask walls) from Binance. "
        "Useful for identifying support/resistance levels and buy/sell pressure."
    )
    args_schema: Type[BaseModel] = OrderBookInput

    def _run(self, symbol: str, depth: int = 20) -> str:
        pair = f"{symbol.upper()}USDT"
        resp = requests.get(
            f"{BINANCE_BASE}/depth",
            params={"symbol": pair, "limit": depth},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        bids = [[float(p), float(q)] for p, q in data["bids"]]
        asks = [[float(p), float(q)] for p, q in data["asks"]]

        total_bid_volume = sum(q for _, q in bids)
        total_ask_volume = sum(q for _, q in asks)
        buy_pressure = round(total_bid_volume / (total_bid_volume + total_ask_volume) * 100, 2)

        top_bid_wall = max(bids, key=lambda x: x[1])
        top_ask_wall = max(asks, key=lambda x: x[1])

        result = {
            "symbol": pair,
            "best_bid": bids[0][0] if bids else None,
            "best_ask": asks[0][0] if asks else None,
            "spread": round(asks[0][0] - bids[0][0], 4) if bids and asks else None,
            "buy_pressure_pct": buy_pressure,
            "sell_pressure_pct": round(100 - buy_pressure, 2),
            "largest_bid_wall": {"price": top_bid_wall[0], "quantity": top_bid_wall[1]},
            "largest_ask_wall": {"price": top_ask_wall[0], "quantity": top_ask_wall[1]},
            "top_5_bids": bids[:5],
            "top_5_asks": asks[:5],
        }
        return json.dumps(result)
