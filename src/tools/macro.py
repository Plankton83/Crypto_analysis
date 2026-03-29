"""
Macro market tools: S&P 500, Nasdaq, futures, VIX, oil, dollar, gold, yields, CME futures
All via yfinance (free, no API key).
"""

import json

import requests
import yfinance as yf
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


def _fetch_ticker_summary(ticker_sym: str, lookback_days: int) -> dict:
    """Shared helper: fetch OHLCV history and return a summary dict."""
    period_map = {7: "1mo", 14: "1mo", 30: "1mo", 60: "3mo", 90: "3mo"}
    period = period_map.get(lookback_days, "1mo")

    try:
        t = yf.Ticker(ticker_sym)
        hist = t.history(period=period)

        if hist.empty:
            return {"error": f"No data returned for {ticker_sym}"}

        # Drop rows where Close is NaN (can happen for futures near market open)
        closes = hist["Close"].dropna()
        if closes.empty:
            return {"error": f"All Close values are NaN for {ticker_sym}"}

        current = float(closes.iloc[-1])
        prev_close = float(closes.iloc[-2]) if len(closes) >= 2 else current
        week_ago = float(closes.iloc[-5]) if len(closes) >= 5 else None
        month_start = float(closes.iloc[0])

        # Use only non-NaN rows for high/low
        high = float(hist["High"].dropna().max())
        low = float(hist["Low"].dropna().min())

        summary = {
            "current": round(current, 4),
            "change_1d_pct": round((current - prev_close) / prev_close * 100, 2),
            "change_7d_pct": (
                round((current - week_ago) / week_ago * 100, 2) if week_ago else None
            ),
            "change_30d_pct": round((current - month_start) / month_start * 100, 2),
            "period_high": round(high, 4),
            "period_low": round(low, 4),
            "trend_30d": "up" if current > month_start else "down",
            "data_points": len(closes),
        }
        return summary

    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tool 1 - S&P 500 + E-mini Futures + VIX
# ---------------------------------------------------------------------------

class SP500Input(BaseModel):
    lookback_days: int = Field(
        default=30,
        description="Days of historical data to analyze (default 30)",
    )


class SP500Tool(BaseTool):
    name: str = "S&P 500 and VIX Tool"
    description: str = (
        "Fetches S&P 500 index (^GSPC) trend data, E-mini S&P 500 futures (ES=F) "
        "premium or discount vs spot, and the VIX volatility index (^VIX). "
        "Use this to assess US equity market momentum, near-term futures sentiment, "
        "and overall fear/greed levels in traditional markets."
    )
    args_schema: type[BaseModel] = SP500Input

    def _run(self, lookback_days: int = 30) -> str:
        result: dict = {}

        # --- S&P 500 spot ---
        spx_data = _fetch_ticker_summary("^GSPC", lookback_days)
        result["sp500_spot"] = spx_data
        spx_current = spx_data.get("current") if "error" not in spx_data else None

        # Simple EMA-based trend: compare last 5 closes vs previous 5
        try:
            hist = yf.Ticker("^GSPC").history(period="1mo")
            if not hist.empty and len(hist) >= 10:
                recent_avg = float(hist["Close"].iloc[-5:].mean())
                prior_avg = float(hist["Close"].iloc[-10:-5].mean())
                spx_data["momentum"] = "positive" if recent_avg > prior_avg else "negative"
                spx_data["avg_volume_vs_prev_5d"] = round(
                    hist["Volume"].iloc[-5:].mean() / hist["Volume"].iloc[-10:-5].mean(), 2
                )
        except Exception:
            pass

        # --- E-mini S&P 500 Futures ---
        es_data = _fetch_ticker_summary("ES=F", lookback_days)
        if "error" not in es_data and spx_current:
            futures_price = es_data["current"]
            diff_pts = round(futures_price - spx_current, 2)
            diff_pct = round(diff_pts / spx_current * 100, 4)
            es_data["basis_pts"] = diff_pts
            es_data["basis_pct"] = diff_pct
            es_data["signal"] = (
                "contango - futures above spot (mild bullish bias)"
                if diff_pts > 0
                else "backwardation - futures below spot (mild bearish bias)"
            )
        result["sp500_futures_ES"] = es_data

        # --- Nasdaq 100 ---
        ndx_data = _fetch_ticker_summary("^NDX", lookback_days)
        if "error" not in ndx_data:
            chg = ndx_data.get("change_30d_pct", 0) or 0
            ndx_data["crypto_implication"] = (
                "bearish - tech selloff often drags crypto"
                if chg < -5
                else "bullish - tech rally supports crypto risk appetite"
                if chg > 5
                else "neutral"
            )
        result["nasdaq100"] = ndx_data

        # --- VIX ---
        vix_data = _fetch_ticker_summary("^VIX", lookback_days)
        if "error" not in vix_data:
            vix = vix_data["current"]
            vix_data["classification"] = (
                "extreme fear (>30) - high crash risk, potential crypto selloff"
                if vix > 30
                else "elevated fear (20-30) - risk-off environment"
                if vix > 20
                else "neutral (15-20) - moderate uncertainty"
                if vix > 15
                else "complacency (<15) - risk-on, favorable for crypto"
            )
            vix_data["crypto_implication"] = (
                "bearish" if vix > 25 else "neutral" if vix > 17 else "bullish"
            )
        result["vix"] = vix_data

        return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 2 - Oil & Dollar Index
# ---------------------------------------------------------------------------

class CommoditiesInput(BaseModel):
    lookback_days: int = Field(
        default=30,
        description="Days of historical data to analyze (default 30)",
    )


class MacroCommoditiesTool(BaseTool):
    name: str = "Oil and Dollar Index Tool"
    description: str = (
        "Fetches WTI crude oil (CL=F), Brent crude oil (BZ=F), and the US Dollar Index "
        "(DX-Y.NYB). Rising oil signals inflationary pressure which can drive rate "
        "expectations and hurt risk assets. A strengthening dollar typically creates "
        "headwinds for crypto and commodities (inverse correlation)."
    )
    args_schema: type[BaseModel] = CommoditiesInput

    def _run(self, lookback_days: int = 30) -> str:
        tickers = {
            "wti_crude_oil": "CL=F",
            "brent_crude_oil": "BZ=F",
            "us_dollar_index": "DX-Y.NYB",
        }

        result: dict = {}

        for label, sym in tickers.items():
            data = _fetch_ticker_summary(sym, lookback_days)
            if "error" not in data:
                chg = data.get("change_30d_pct", 0)
                if label in ("wti_crude_oil", "brent_crude_oil"):
                    data["crypto_implication"] = (
                        "bearish - rising oil fuels inflation fears, tighter monetary policy"
                        if chg > 5
                        else "bullish - falling oil reduces inflation pressure"
                        if chg < -5
                        else "neutral"
                    )
                elif label == "us_dollar_index":
                    data["crypto_implication"] = (
                        "bearish - strong dollar compresses risk-asset valuations"
                        if chg > 2
                        else "bullish - weak dollar supports risk assets and crypto"
                        if chg < -2
                        else "neutral"
                    )
            result[label] = data

        # Combined macro sentiment signal
        signals = []
        wti = result.get("wti_crude_oil", {})
        dxy = result.get("us_dollar_index", {})

        if "error" not in wti:
            signals.append("oil_rising" if (wti.get("change_30d_pct") or 0) > 3 else "oil_falling")
        if "error" not in dxy:
            signals.append("dollar_strong" if (dxy.get("change_30d_pct") or 0) > 1 else "dollar_weak")

        bearish_count = sum(1 for s in signals if s in ("oil_rising", "dollar_strong"))
        result["combined_macro_signal"] = (
            "risk-off (bearish for crypto)" if bearish_count == 2
            else "risk-on (bullish for crypto)" if bearish_count == 0
            else "mixed"
        )

        return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Tool 3 - Rates, Gold & CME Bitcoin Futures
# ---------------------------------------------------------------------------

class MacroRatesGoldTool(BaseTool):
    name: str = "Macro Rates, Gold and CME Futures Tool"
    description: str = (
        "Fetches US 10-year Treasury yield (^TNX), Gold futures (GC=F), and "
        "CME Bitcoin futures (BTC=F) with their 30-day trends. Also computes the "
        "CME futures premium/discount vs Binance spot price. "
        "Rising yields = bearish for crypto (higher cost of capital, competing with bonds). "
        "Gold rallying = safe-haven demand, can be risk-off. "
        "CME premium > 0 = institutional Bitcoin demand (contango). "
        "CME discount = institutional hedging or bearish positioning (backwardation)."
    )
    args_schema: type[BaseModel] = CommoditiesInput

    def _run(self, lookback_days: int = 30) -> str:
        result: dict = {}

        # --- 10-Year Treasury Yield ---
        tnx = _fetch_ticker_summary("^TNX", lookback_days)
        if "error" not in tnx:
            chg = tnx.get("change_30d_pct", 0) or 0
            current_yield = tnx.get("current", 0) or 0
            tnx["label"] = f"{current_yield:.2f}% yield"
            tnx["crypto_implication"] = (
                "bearish - yields rising, increases cost of capital and bond competition"
                if chg > 5
                else "bullish - yields falling, reduces opportunity cost, supports risk assets"
                if chg < -5
                else "neutral"
            )
            tnx["regime"] = (
                "high yield environment (>4.5%) - structural headwind for crypto"
                if current_yield > 4.5
                else "moderate yield (3-4.5%) - watch direction"
                if current_yield > 3.0
                else "low yield (<3%) - favorable for risk assets"
            )
        result["us_10y_treasury_yield"] = tnx

        # --- Gold ---
        gold = _fetch_ticker_summary("GC=F", lookback_days)
        if "error" not in gold:
            chg = gold.get("change_30d_pct", 0) or 0
            gold["crypto_implication"] = (
                "mixed - gold rising can signal inflation hedge demand (crypto-positive) "
                "or risk-off flight to safety (crypto-negative)"
                if chg > 3
                else "neutral - gold falling, limited inflation concern"
                if chg < -3
                else "neutral"
            )
        result["gold"] = gold

        # --- CME Bitcoin Futures ---
        cme = _fetch_ticker_summary("BTC=F", lookback_days)
        result["cme_btc_futures"] = cme

        if "error" not in cme:
            cme_price = cme.get("current")
            try:
                spot_resp = requests.get(
                    "https://api.binance.com/api/v3/ticker/price",
                    params={"symbol": "BTCUSDT"},
                    timeout=5,
                )
                binance_spot = float(spot_resp.json()["price"])
                if cme_price and binance_spot:
                    premium = cme_price - binance_spot
                    premium_pct = round(premium / binance_spot * 100, 4)
                    result["cme_vs_binance_spot"] = {
                        "binance_spot_usd": round(binance_spot, 2),
                        "cme_futures_usd": round(cme_price, 2),
                        "premium_usd": round(premium, 2),
                        "premium_pct": premium_pct,
                        "signal": (
                            "contango - CME above spot, institutional Bitcoin demand / bullish"
                            if premium > 200
                            else "backwardation - CME below spot, institutional hedging / bearish"
                            if premium < -200
                            else "at parity - neutral institutional positioning"
                        ),
                    }
            except Exception:
                pass

        return json.dumps(result, indent=2)
