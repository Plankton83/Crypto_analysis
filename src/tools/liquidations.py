"""
Liquidation and derivatives data tools.
Sources:
- Coinglass v3 API (requires API key — register at coinglass.com, use COINGLASS_API_KEY env var)
- OKX public API (no key needed — ~8h of detailed liquidation events with price data)
- Gate.io public API (no key needed — additional exchange cross-validation)
- Binance Futures public API (no key needed — taker pressure ratios + open interest + funding)
"""

import json
import os
from datetime import datetime, timezone
from typing import Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


BINANCE_FUTURES_BASE = "https://fapi.binance.com/fapi/v1"
BINANCE_DATA_BASE = "https://fapi.binance.com/futures/data"
COINGLASS_V3_BASE = "https://open-api-v3.coinglass.com"
OKX_BASE = "https://www.okx.com/api/v5"
GATE_BASE = "https://api.gateio.ws/api/v4"


class LiquidationInput(BaseModel):
    symbol: str = Field(description="Crypto symbol e.g. BTC, ETH")
    interval: str = Field(default="1h", description="Time interval: 5m, 15m, 30m, 1h, 4h, 12h, 1d")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_okx_contract_size(inst_id: str) -> float:
    """Fetch the contract face value (ctVal) for an OKX SWAP instrument."""
    resp = requests.get(
        f"{OKX_BASE}/public/instruments",
        params={"instType": "SWAP", "instId": inst_id},
        timeout=10,
    )
    resp.raise_for_status()
    instruments = resp.json().get("data", [])
    return float(instruments[0].get("ctVal", 1)) if instruments else 1.0


def _bucket_size_for_price(price: float) -> float:
    """Return a sensible $ bucket width for the heatmap given the asset price."""
    if price >= 10_000:
        return 500.0
    if price >= 1_000:
        return 50.0
    if price >= 100:
        return 5.0
    return 1.0


def _build_price_heatmap(events: list[tuple[float, float, str]], bucket_size: float) -> list[dict]:
    """
    Build a price-level liquidation heatmap from raw events.

    events: list of (price, usd_value, side) where side is 'long' or 'short'
    Returns top-10 price buckets sorted by total USD volume.
    """
    buckets: dict[float, dict] = {}
    for price, usd_val, side in events:
        b = round(int(price / bucket_size) * bucket_size, 2)
        if b not in buckets:
            buckets[b] = {"long_usd": 0.0, "short_usd": 0.0}
        if side == "long":
            buckets[b]["long_usd"] += usd_val
        else:
            buckets[b]["short_usd"] += usd_val

    top = sorted(
        buckets.items(),
        key=lambda x: x[1]["long_usd"] + x[1]["short_usd"],
        reverse=True,
    )[:10]

    return [
        {
            "price_range": f"${b:,.0f}-${b + bucket_size:,.0f}",
            "long_liq_usd": round(v["long_usd"]),
            "short_liq_usd": round(v["short_usd"]),
            "total_usd": round(v["long_usd"] + v["short_usd"]),
            "dominant": "long liquidations" if v["long_usd"] >= v["short_usd"] else "short liquidations",
        }
        for b, v in top
    ]


def _build_hourly_timeline(events: list[tuple[float, float, str, int]]) -> list[dict]:
    """
    Aggregate events into hourly bins.
    events: list of (price, usd_value, side, timestamp_ms)
    Returns last 12 hours sorted chronologically.
    """
    bins: dict[str, dict] = {}
    for _, usd_val, side, ts_ms in events:
        hour = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:00 UTC")
        if hour not in bins:
            bins[hour] = {"long_usd": 0.0, "short_usd": 0.0}
        if side == "long":
            bins[hour]["long_usd"] += usd_val
        else:
            bins[hour]["short_usd"] += usd_val

    timeline = [
        {
            "hour": h,
            "long_liq_usd": round(v["long_usd"]),
            "short_liq_usd": round(v["short_usd"]),
            "total_usd": round(v["long_usd"] + v["short_usd"]),
        }
        for h, v in sorted(bins.items())
    ]
    return timeline[-12:]  # keep last 12 hours


# ── Exchange fetchers ─────────────────────────────────────────────────────────

def _fetch_okx_events(symbol: str) -> list[tuple[float, float, str, int]]:
    """
    Fetch raw liquidation events from OKX.
    Returns list of (price, usd_value, side, timestamp_ms).
    Covers roughly the last 8 hours (~1600 events).
    """
    sym = symbol.upper()
    try:
        ct_val = _get_okx_contract_size(f"{sym}-USDT-SWAP")
    except Exception:
        ct_val = 1.0

    resp = requests.get(
        f"{OKX_BASE}/public/liquidation-orders",
        params={"instType": "SWAP", "uly": f"{sym}-USDT", "state": "filled", "limit": 100},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "0":
        raise ValueError(f"OKX error: {data.get('msg')}")

    events = []
    for record in data.get("data", []):
        for d in record.get("details", []):
            sz = float(d.get("sz", 0))
            px = float(d.get("bkPx", 0))
            side = d.get("posSide", "")
            ts = int(d.get("ts", 0))
            if sz and px and side in ("long", "short"):
                events.append((px, sz * ct_val * px, side, ts))
    return events


def _fetch_gate_events(symbol: str) -> list[tuple[float, float, str, int]]:
    """
    Fetch raw liquidation events from Gate.io.
    Returns list of (price, usd_value, side, timestamp_ms).
    Gate.io BTC_USDT multiplier: 0.0001 BTC/contract.
    Positive size = long liquidated; negative = short liquidated.
    """
    gate_sym = f"{symbol.upper()}_USDT"
    try:
        resp = requests.get(
            f"{GATE_BASE}/futures/usdt/contracts/{gate_sym}",
            timeout=10,
        )
        resp.raise_for_status()
        multiplier = float(resp.json().get("quanto_multiplier", 0.0001))
    except Exception:
        multiplier = 0.0001

    resp = requests.get(
        f"{GATE_BASE}/futures/usdt/liq_orders",
        params={"contract": gate_sym, "limit": 100},
        timeout=10,
    )
    resp.raise_for_status()
    records = resp.json()
    if not isinstance(records, list):
        raise ValueError(f"Gate.io unexpected response: {records}")

    events = []
    for r in records:
        size = int(r.get("size", 0))
        px = float(r.get("fill_price", 0))
        ts_s = int(r.get("time", 0))
        if size and px:
            usd = abs(size) * px * multiplier
            side = "long" if size > 0 else "short"
            events.append((px, usd, side, ts_s * 1000))
    return events


def _fetch_binance_taker_ratio(symbol: str, periods: int = 12) -> list[dict]:
    """Fetch hourly taker buy/sell ratio from Binance (no key required)."""
    pair = f"{symbol.upper()}USDT"
    resp = requests.get(
        f"{BINANCE_DATA_BASE}/takerlongshortRatio",
        params={"symbol": pair, "period": "1h", "limit": periods},
        timeout=10,
    )
    if not resp.ok:
        return []
    rows = resp.json()
    result = []
    for r in rows:
        ratio = float(r.get("buySellRatio", 1))
        result.append({
            "hour": datetime.fromtimestamp(int(r["timestamp"]) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:00 UTC"),
            "buy_sell_ratio": round(ratio, 4),
            "pressure": (
                "strong buy pressure" if ratio > 1.2
                else "slight buy pressure" if ratio > 1.05
                else "strong sell pressure" if ratio < 0.83
                else "slight sell pressure" if ratio < 0.95
                else "balanced"
            ),
        })
    return result


# ── Tool ─────────────────────────────────────────────────────────────────────

class LiquidationHeatmapTool(BaseTool):
    name: str = "get_liquidation_data"
    description: str = (
        "Fetch a comprehensive liquidation heatmap from multiple exchanges. "
        "Shows: (1) total long vs short liquidation USD volumes, "
        "(2) price-level heatmap — which price buckets had the most liquidations, "
        "(3) hourly liquidation timeline over the past 8–12 hours, "
        "(4) Binance taker buy/sell pressure context. "
        "Uses Coinglass v3 if COINGLASS_API_KEY is set; otherwise combines "
        "OKX + Gate.io public data (no key required)."
    )
    args_schema: Type[BaseModel] = LiquidationInput

    def _run(self, symbol: str, interval: str = "1h") -> str:
        api_key = os.getenv("COINGLASS_API_KEY")

        # ── Try Coinglass v3 first (if key available) ──────────────────────
        if api_key:
            try:
                headers = {"CG-API-KEY": api_key}
                params = {"symbol": symbol.upper(), "interval": interval, "limit": 48}
                resp = requests.get(
                    f"{COINGLASS_V3_BASE}/api/futures/liquidation/history",
                    headers=headers, params=params, timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                if not data.get("success"):
                    raise ValueError(data.get("msg", "Coinglass v3 API error"))

                liq_data = data.get("data", [])
                total_long = sum(
                    d.get("longLiquidationUsd") or d.get("buyLiquidationUSD", 0) for d in liq_data
                )
                total_short = sum(
                    d.get("shortLiquidationUsd") or d.get("sellLiquidationUSD", 0) for d in liq_data
                )
                result = {
                    "source": "Coinglass v3",
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "periods_analyzed": len(liq_data),
                    "total_long_liquidations_usd": round(total_long, 2),
                    "total_short_liquidations_usd": round(total_short, 2),
                    "long_vs_short_ratio": round(total_long / total_short, 3) if total_short else None,
                    "dominant_side": (
                        "longs being liquidated (bearish pressure)" if total_long > total_short
                        else "shorts being liquidated (bullish squeeze)"
                    ),
                    "recent_periods": liq_data[-12:] if len(liq_data) >= 12 else liq_data,
                }
                return json.dumps(result, default=str)

            except Exception as e:
                coinglass_error = str(e)
        else:
            coinglass_error = "COINGLASS_API_KEY not set"

        # ── Multi-exchange fallback: OKX + Gate.io ─────────────────────────
        all_events: list[tuple[float, float, str, int]] = []
        sources_used: list[str] = []
        errors: list[str] = []

        try:
            okx_events = _fetch_okx_events(symbol)
            all_events.extend(okx_events)
            sources_used.append(f"OKX ({len(okx_events)} events)")
        except Exception as e:
            errors.append(f"OKX: {e}")

        try:
            gate_events = _fetch_gate_events(symbol)
            all_events.extend(gate_events)
            sources_used.append(f"Gate.io ({len(gate_events)} events)")
        except Exception as e:
            errors.append(f"Gate.io: {e}")

        if not all_events:
            return json.dumps({
                "error": f"All liquidation sources failed. Coinglass: {coinglass_error}. {'; '.join(errors)}",
            })

        # Aggregate totals
        total_long_usd = sum(usd for _, usd, side, _ in all_events if side == "long")
        total_short_usd = sum(usd for _, usd, side, _ in all_events if side == "short")

        # Time range
        ts_list = [ts for _, _, _, ts in all_events]
        time_from = datetime.fromtimestamp(min(ts_list) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        time_to = datetime.fromtimestamp(max(ts_list) / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        # Price heatmap
        avg_price = sum(px for px, _, _, _ in all_events) / len(all_events)
        bucket_sz = _bucket_size_for_price(avg_price)
        heatmap = _build_price_heatmap([(px, usd, side) for px, usd, side, _ in all_events], bucket_sz)
        peak_range = heatmap[0]["price_range"] if heatmap else "N/A"

        # Hourly timeline
        hourly = _build_hourly_timeline(all_events)
        peak_hour_entry = max(hourly, key=lambda x: x["total_usd"], default=None)

        # Binance taker pressure context
        taker_ratios: list[dict] = []
        try:
            taker_ratios = _fetch_binance_taker_ratio(symbol, periods=12)
        except Exception:
            pass

        result = {
            "sources": sources_used,
            "coinglass_skipped": coinglass_error,
            "time_range": f"{time_from} to {time_to}",
            "total_events": len(all_events),
            "total_long_liquidations_usd": round(total_long_usd),
            "total_short_liquidations_usd": round(total_short_usd),
            "long_vs_short_ratio": round(total_long_usd / total_short_usd, 3) if total_short_usd else None,
            "dominant_side": (
                "longs being liquidated (bearish pressure)" if total_long_usd > total_short_usd
                else "shorts being liquidated (bullish squeeze)"
            ),
            "price_heatmap": {
                "bucket_size_usd": bucket_sz,
                "note": "Price levels with highest liquidation USD volume — top 10 shown",
                "peak_liquidation_range": peak_range,
                "levels": heatmap,
            },
            "hourly_timeline": {
                "note": "Liquidation volume aggregated by hour (UTC)",
                "peak_hour": peak_hour_entry["hour"] if peak_hour_entry else None,
                "peak_hour_total_usd": peak_hour_entry["total_usd"] if peak_hour_entry else None,
                "hours": hourly,
            },
            "taker_pressure_12h": {
                "note": "Binance taker buy/sell ratio — >1 means more aggressive buyers than sellers",
                "latest": taker_ratios[-1] if taker_ratios else None,
                "history": taker_ratios,
            },
        }
        if errors:
            result["fetch_errors"] = errors

        return json.dumps(result, default=str)


class FundingRateTool(BaseTool):
    name: str = "get_funding_rate"
    description: str = (
        "Fetch current and historical perpetual futures funding rates from Binance. "
        "Positive funding = longs pay shorts (market is long-heavy, bearish signal). "
        "Negative funding = shorts pay longs (market is short-heavy, bullish signal). "
        "No API key required."
    )
    args_schema: Type[BaseModel] = LiquidationInput

    def _run(self, symbol: str, interval: str = "1h") -> str:
        pair = f"{symbol.upper()}USDT"

        # Current funding rate
        resp = requests.get(
            f"{BINANCE_FUTURES_BASE}/premiumIndex",
            params={"symbol": pair}, timeout=10,
        )
        resp.raise_for_status()
        current = resp.json()

        # Historical funding rates (last 30 periods)
        hist_resp = requests.get(
            f"{BINANCE_FUTURES_BASE}/fundingRate",
            params={"symbol": pair, "limit": 30}, timeout=10,
        )
        hist_resp.raise_for_status()
        history = hist_resp.json()

        rates = [float(h["fundingRate"]) * 100 for h in history]
        avg_rate = sum(rates) / len(rates) if rates else 0

        current_rate = float(current.get("lastFundingRate", 0)) * 100
        mark_price = float(current.get("markPrice", 0))
        index_price = float(current.get("indexPrice", 0))
        basis = round((mark_price - index_price) / index_price * 100, 4) if index_price else 0

        def funding_signal(r):
            if r > 0.1: return "extremely positive (market over-leveraged long, bearish)"
            if r > 0.03: return "positive (slight long bias)"
            if r < -0.1: return "extremely negative (market over-leveraged short, bullish)"
            if r < -0.03: return "negative (slight short bias)"
            return "neutral"

        result = {
            "symbol": pair,
            "current_funding_rate_pct": round(current_rate, 4),
            "funding_signal": funding_signal(current_rate),
            "mark_price": mark_price,
            "index_price": index_price,
            "basis_pct": basis,
            "basis_signal": "contango (longs expensive)" if basis > 0 else "backwardation (shorts expensive)",
            "avg_funding_30_periods": round(avg_rate, 4),
            "annualized_funding_rate_pct": round(current_rate * 3 * 365, 2),
            "funding_trend": "increasing" if current_rate > avg_rate else "decreasing",
            "recent_funding_rates": [round(r, 4) for r in rates[-10:]],
        }
        return json.dumps(result)


class OpenInterestInput(BaseModel):
    symbol: str = Field(description="Crypto symbol e.g. BTC, ETH")


class OpenInterestTool(BaseTool):
    name: str = "get_open_interest"
    description: str = (
        "Fetch open interest (total value of outstanding futures contracts) from Binance Futures. "
        "Rising OI with rising price = bullish. Falling OI with falling price = bearish. "
        "No API key required."
    )
    args_schema: Type[BaseModel] = OpenInterestInput

    def _run(self, symbol: str) -> str:
        pair = f"{symbol.upper()}USDT"

        # Current open interest
        oi_resp = requests.get(
            f"{BINANCE_FUTURES_BASE}/openInterest",
            params={"symbol": pair}, timeout=10,
        )
        oi_resp.raise_for_status()
        current_oi = oi_resp.json()

        # Historical OI (last 30 periods, 4h interval)
        hist_resp = requests.get(
            "https://fapi.binance.com/futures/data/openInterestHist",
            params={"symbol": pair, "period": "4h", "limit": 30}, timeout=10,
        )
        hist_resp.raise_for_status()
        history = hist_resp.json()

        oi_values = [float(h["sumOpenInterestValue"]) for h in history]
        current_oi_val = float(current_oi.get("openInterest", 0))
        prev_oi_val = oi_values[-2] if len(oi_values) >= 2 else current_oi_val
        oi_change = round((oi_values[-1] - oi_values[0]) / oi_values[0] * 100, 2) if oi_values[0] else 0

        # Long/short ratio from Binance
        ls_resp = requests.get(
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
            params={"symbol": pair, "period": "4h", "limit": 10}, timeout=10,
        )
        ls_data = ls_resp.json() if ls_resp.ok else []

        current_ls = float(ls_data[-1]["longShortRatio"]) if ls_data else None
        long_pct = float(ls_data[-1]["longAccount"]) * 100 if ls_data else None
        short_pct = float(ls_data[-1]["shortAccount"]) * 100 if ls_data else None

        result = {
            "symbol": pair,
            "open_interest_contracts": float(current_oi.get("openInterest", 0)),
            "oi_change_30_periods_pct": oi_change,
            "oi_trend": "increasing" if oi_change > 0 else "decreasing",
            "long_account_pct": round(long_pct, 2) if long_pct else None,
            "short_account_pct": round(short_pct, 2) if short_pct else None,
            "long_short_ratio": round(current_ls, 3) if current_ls else None,
            "positioning_signal": (
                "over-leveraged long (contrarian bearish)" if (long_pct or 0) > 65
                else "over-leveraged short (contrarian bullish)" if (long_pct or 0) < 35
                else "balanced positioning"
            ),
            "oi_history_values_usd": [round(v) for v in oi_values[-10:]],
        }
        return json.dumps(result, default=str)
