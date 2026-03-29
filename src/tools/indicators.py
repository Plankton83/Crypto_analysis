"""
Technical analysis indicators computed from OHLCV data using the `ta` library.
All indicators are calculated locally — no external API needed.
"""

import json
from datetime import datetime, timedelta
from typing import Type

import numpy as np
import pandas as pd
import requests
import ta
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


BINANCE_BASE = "https://api.binance.com/api/v3"


_INTERVAL_DAYS: dict[str, float] = {
    "1m": 1/1440, "3m": 3/1440, "5m": 5/1440, "15m": 15/1440,
    "30m": 0.5/24, "1h": 1/24, "2h": 2/24, "4h": 4/24, "6h": 6/24,
    "8h": 8/24, "12h": 12/24, "1d": 1, "3d": 3, "1w": 7, "1M": 30,
}


def _fetch_ohlcv(symbol: str, interval: str = "1d", lookback_days: int = 200) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Binance.

    lookback_days is a minimum hint for daily candles; for wider intervals (e.g. 1w)
    the lookback is automatically scaled up so we always fetch at least 250 candles,
    which is enough for EMA200 and all other indicators.
    """
    pair = f"{symbol.upper()}USDT"
    candle_width_days = _INTERVAL_DAYS.get(interval, 1)
    # Ensure at least 250 candles worth of history regardless of interval
    min_days = max(lookback_days, int(250 * candle_width_days) + 1)
    start_ms = int((datetime.utcnow() - timedelta(days=min_days)).timestamp() * 1000)
    params = {"symbol": pair, "interval": interval, "startTime": start_ms, "limit": 1000}
    resp = requests.get(f"{BINANCE_BASE}/klines", params=params, timeout=10)
    resp.raise_for_status()

    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(resp.json(), columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df.set_index("open_time")


class IndicatorsInput(BaseModel):
    symbol: str = Field(description="Crypto symbol e.g. BTC, ETH")
    interval: str = Field(default="1d", description="Candle interval: 1h, 4h, 1d")


class TechnicalIndicatorsTool(BaseTool):
    name: str = "get_technical_indicators"
    description: str = (
        "Calculate comprehensive technical indicators for a cryptocurrency: "
        "RSI, MACD, Bollinger Bands, EMA (20/50/200), Stochastic RSI, ATR, OBV, "
        "ADX, CCI, Williams %R, and trend signals. Returns current values and signal interpretations."
    )
    args_schema: Type[BaseModel] = IndicatorsInput

    def _run(self, symbol: str, interval: str = "1d") -> str:
        df = _fetch_ohlcv(symbol, interval, lookback_days=200)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # RSI
        rsi = ta.momentum.RSIIndicator(close=close, window=14).rsi().dropna().iloc[-1]

        # MACD
        macd_ind = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        macd_val = macd_ind.macd().dropna().iloc[-1]
        macd_signal = macd_ind.macd_signal().dropna().iloc[-1]
        macd_hist = macd_ind.macd_diff().dropna().iloc[-1]
        macd_hist_prev = macd_ind.macd_diff().dropna().iloc[-2]

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband().dropna().iloc[-1]
        bb_mid = bb.bollinger_mavg().dropna().iloc[-1]
        bb_lower = bb.bollinger_lband().dropna().iloc[-1]
        current_price = close.iloc[-1]
        bb_width = round((bb_upper - bb_lower) / bb_mid * 100, 2)
        bb_position = round((current_price - bb_lower) / (bb_upper - bb_lower) * 100, 2)

        # EMAs
        ema20 = ta.trend.EMAIndicator(close=close, window=20).ema_indicator().dropna().iloc[-1]
        ema50 = ta.trend.EMAIndicator(close=close, window=50).ema_indicator().dropna().iloc[-1]
        ema200_series = ta.trend.EMAIndicator(close=close, window=200).ema_indicator().dropna()
        ema200 = ema200_series.iloc[-1] if len(ema200_series) > 0 else None

        # Stochastic RSI
        stoch = ta.momentum.StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3)
        stoch_k = stoch.stochrsi_k().dropna().iloc[-1] * 100
        stoch_d = stoch.stochrsi_d().dropna().iloc[-1] * 100

        # ATR
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range().dropna().iloc[-1]
        atr_pct = round(atr / current_price * 100, 2)

        # OBV
        obv_series = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        obv_trend = "rising" if obv_series.iloc[-1] > obv_series.iloc[-8] else "falling"

        # ADX
        adx_ind = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14)
        adx = adx_ind.adx().dropna().iloc[-1]
        dmp = adx_ind.adx_pos().dropna().iloc[-1]
        dmn = adx_ind.adx_neg().dropna().iloc[-1]

        # CCI
        cci = ta.trend.CCIIndicator(high=high, low=low, close=close, window=20).cci().dropna().iloc[-1]

        # Williams %R
        willr = ta.momentum.WilliamsRIndicator(high=high, low=low, close=close, lbp=14).williams_r().dropna().iloc[-1]

        # Signal helpers
        def rsi_signal(v):
            if v >= 70: return "overbought"
            if v <= 30: return "oversold"
            if v >= 60: return "bullish"
            if v <= 40: return "bearish"
            return "neutral"

        def trend_signal(p, e20, e50, e200):
            signals = []
            signals.append("above EMA20" if p > e20 else "below EMA20")
            signals.append("above EMA50" if p > e50 else "below EMA50")
            if e200:
                signals.append("above EMA200 (bull market)" if p > e200 else "below EMA200 (bear market)")
                signals.append("EMA20>EMA50 (golden cross)" if e20 > e50 else "EMA20<EMA50 (death cross)")
            return ", ".join(signals)

        result = {
            "symbol": symbol.upper(),
            "interval": interval,
            "current_price": round(current_price, 4),
            "rsi_14": round(rsi, 2),
            "rsi_signal": rsi_signal(rsi),
            "macd": round(macd_val, 4),
            "macd_signal_line": round(macd_signal, 4),
            "macd_histogram": round(macd_hist, 4),
            "macd_interpretation": (
                f"{'bullish' if macd_val > macd_signal else 'bearish'} cross, "
                f"momentum {'strengthening' if abs(macd_hist) > abs(macd_hist_prev) else 'weakening'}"
            ),
            "bollinger_upper": round(bb_upper, 4),
            "bollinger_mid": round(bb_mid, 4),
            "bollinger_lower": round(bb_lower, 4),
            "bb_width_pct": bb_width,
            "bb_position_pct": bb_position,
            "bb_signal": (
                "near upper band (resistance)" if bb_position > 80
                else "near lower band (support)" if bb_position < 20
                else "mid-range"
            ),
            "ema_20": round(ema20, 4),
            "ema_50": round(ema50, 4),
            "ema_200": round(ema200, 4) if ema200 else "insufficient data",
            "trend_signal": trend_signal(current_price, ema20, ema50, ema200),
            "stoch_rsi_k": round(stoch_k, 2),
            "stoch_rsi_d": round(stoch_d, 2),
            "stoch_signal": "overbought" if stoch_k > 80 else "oversold" if stoch_k < 20 else "neutral",
            "atr_14": round(atr, 4),
            "atr_pct_of_price": atr_pct,
            "volatility": "high" if atr_pct > 5 else "moderate" if atr_pct > 2 else "low",
            "obv_trend": obv_trend,
            "adx_14": round(adx, 2),
            "trend_strength": "strong" if adx > 25 else "weak" if adx < 20 else "moderate",
            "dmi_plus": round(dmp, 2),
            "dmi_minus": round(dmn, 2),
            "dmi_signal": "bullish" if dmp > dmn else "bearish",
            "cci_20": round(cci, 2),
            "cci_signal": "overbought" if cci > 100 else "oversold" if cci < -100 else "neutral",
            "williams_r": round(willr, 2),
            "williams_signal": "overbought" if willr > -20 else "oversold" if willr < -80 else "neutral",
        }

        # Bias score (-10 to +10)
        score = 0
        score += 2 if rsi < 40 else (-2 if rsi > 70 else 0)
        score += 1 if macd_val > macd_signal else -1
        score += 1 if macd_hist > 0 else -1
        score += 1 if bb_position < 30 else (-1 if bb_position > 70 else 0)
        score += 1 if current_price > ema20 else -1
        score += 1 if current_price > ema50 else -1
        score += 1 if ema200 and current_price > ema200 else -1
        score += 1 if stoch_k < 20 else (-1 if stoch_k > 80 else 0)
        score += 1 if obv_trend == "rising" else -1
        score += 1 if dmp > dmn else -1

        result["overall_bias_score"] = score
        result["overall_bias"] = (
            "strongly bullish" if score >= 6
            else "bullish" if score >= 3
            else "strongly bearish" if score <= -6
            else "bearish" if score <= -3
            else "neutral"
        )

        return json.dumps(result, default=str)


# ── Volume Profile ────────────────────────────────────────────────────────────

class VolumeProfileInput(BaseModel):
    symbol: str = Field(description="Crypto symbol e.g. BTC, ETH")
    lookback_days: int = Field(default=90, description="Days of OHLCV data to build profile from")
    num_bins: int = Field(default=60, description="Number of price buckets (resolution of profile)")


class VolumeProfileTool(BaseTool):
    name: str = "get_volume_profile"
    description: str = (
        "Compute Volume Profile from OHLCV data: VPOC (Volume Point of Control — "
        "price level with the highest traded volume, strongest support/resistance), "
        "Value Area High/Low (range containing 70% of all volume = fair value zone), "
        "and High Volume Nodes (top price levels by volume, act as magnets). "
        "Price above VPOC = bullish structure. Price outside Value Area = extended."
    )
    args_schema: Type[BaseModel] = VolumeProfileInput

    def _run(self, symbol: str, lookback_days: int = 90, num_bins: int = 60) -> str:
        df = _fetch_ohlcv(symbol, "1d", lookback_days)
        current_price = float(df["close"].iloc[-1])
        price_min = float(df["low"].min())
        price_max = float(df["high"].max())

        bins = np.linspace(price_min, price_max, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        volume_by_level = np.zeros(num_bins)

        for _, row in df.iterrows():
            low  = float(row["low"])
            high = float(row["high"])
            vol  = float(row["volume"])
            if high <= low:
                idx = min(max(int((low - price_min) / (price_max - price_min) * num_bins), 0), num_bins - 1)
                volume_by_level[idx] += vol
            else:
                for i in range(num_bins):
                    overlap = max(0.0, min(high, bins[i + 1]) - max(low, bins[i]))
                    if overlap > 0:
                        volume_by_level[i] += vol * overlap / (high - low)

        # VPOC
        vpoc_idx = int(np.argmax(volume_by_level))
        vpoc = round(float(bin_centers[vpoc_idx]), 2)

        # Value Area — accumulate top bins by volume until 70% is covered
        total_vol = float(volume_by_level.sum())
        sorted_indices = list(np.argsort(volume_by_level)[::-1])
        accumulated = 0.0
        va_indices: list[int] = []
        for idx in sorted_indices:
            accumulated += float(volume_by_level[idx])
            va_indices.append(int(idx))
            if accumulated >= total_vol * 0.70:
                break

        vah = round(float(bin_centers[max(va_indices)]), 2)
        val = round(float(bin_centers[min(va_indices)]), 2)

        # High Volume Nodes (top 5)
        hvn = [
            {
                "price": round(float(bin_centers[int(i)]), 2),
                "relative_volume": round(float(volume_by_level[int(i)]) / float(volume_by_level.max()), 3),
            }
            for i in list(np.argsort(volume_by_level)[::-1])[:5]
        ]

        # Interpretation
        if val <= current_price <= vah:
            va_position = f"inside Value Area ({val} - {vah}) — at fair value"
        elif current_price > vah:
            va_position = f"above Value Area High ({vah}) — extended / overbought zone"
        else:
            va_position = f"below Value Area Low ({val}) — discounted / potential demand zone"

        result = {
            "symbol": symbol.upper(),
            "lookback_days": lookback_days,
            "current_price": round(current_price, 2),
            "vpoc": vpoc,
            "value_area_high": vah,
            "value_area_low": val,
            "value_area_width_pct": round((vah - val) / vpoc * 100, 2),
            "high_volume_nodes": hvn,
            "price_vs_value_area": va_position,
            "vpoc_signal": (
                f"price above VPOC ({vpoc}) — bullish market structure"
                if current_price > vpoc
                else f"price below VPOC ({vpoc}) — bearish market structure"
            ),
        }
        return json.dumps(result)


class SupportResistanceInput(BaseModel):
    symbol: str = Field(description="Crypto symbol e.g. BTC, ETH")
    interval: str = Field(default="1d", description="Candle interval")


class SupportResistanceTool(BaseTool):
    name: str = "get_support_resistance"
    description: str = (
        "Identify key support and resistance levels for a cryptocurrency "
        "using pivot points and recent swing highs/lows."
    )
    args_schema: Type[BaseModel] = SupportResistanceInput

    def _run(self, symbol: str, interval: str = "1d") -> str:
        df = _fetch_ohlcv(symbol, interval, lookback_days=90)
        close_price = df["close"].iloc[-1]
        high = df["high"]
        low = df["low"]

        # Classic Pivot Points from last complete candle
        prev = df.iloc[-2]
        pp = (prev["high"] + prev["low"] + prev["close"]) / 3
        r1 = 2 * pp - prev["low"]
        r2 = pp + (prev["high"] - prev["low"])
        r3 = prev["high"] + 2 * (pp - prev["low"])
        s1 = 2 * pp - prev["high"]
        s2 = pp - (prev["high"] - prev["low"])
        s3 = prev["low"] - 2 * (prev["high"] - pp)

        # Swing highs/lows
        window = 5
        swing_highs, swing_lows = [], []
        for i in range(window, len(df) - window):
            if high.iloc[i] == high.iloc[i - window:i + window + 1].max():
                swing_highs.append(round(high.iloc[i], 4))
            if low.iloc[i] == low.iloc[i - window:i + window + 1].min():
                swing_lows.append(round(low.iloc[i], 4))

        resistances = sorted([r for r in swing_highs if r > close_price], key=lambda x: x - close_price)[:5]
        supports = sorted([s for s in swing_lows if s < close_price], key=lambda x: close_price - x)[:5]

        result = {
            "symbol": symbol.upper(),
            "current_price": round(close_price, 4),
            "pivot_point": round(pp, 4),
            "resistances": {"R1": round(r1, 4), "R2": round(r2, 4), "R3": round(r3, 4), "swing_highs_nearby": resistances},
            "supports": {"S1": round(s1, 4), "S2": round(s2, 4), "S3": round(s3, 4), "swing_lows_nearby": supports},
            "nearest_resistance": resistances[0] if resistances else round(r1, 4),
            "nearest_support": supports[0] if supports else round(s1, 4),
            "distance_to_resistance_pct": round((resistances[0] - close_price) / close_price * 100, 2) if resistances else None,
            "distance_to_support_pct": round((close_price - supports[0]) / close_price * 100, 2) if supports else None,
        }
        return json.dumps(result)
