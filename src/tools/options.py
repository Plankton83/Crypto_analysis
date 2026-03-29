"""
Options market tools:
- Deribit BTC/ETH options (free public API, no key required)
  Provides: put/call ratio, max pain price, IV term structure, top OI strikes
"""

import json
from datetime import datetime, timezone

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


DERIBIT_BASE = "https://www.deribit.com/api/v2/public"


def _parse_instrument(name: str) -> dict | None:
    """Parse 'BTC-28MAR25-80000-C' into components."""
    parts = name.split("-")
    if len(parts) != 4:
        return None
    try:
        return {
            "expiry_str": parts[1],
            "strike": float(parts[2]),
            "option_type": parts[3],  # "C" or "P"
        }
    except (ValueError, IndexError):
        return None


def _expiry_to_dt(expiry_str: str) -> datetime | None:
    for fmt in ("%d%b%y", "%d%b%Y"):
        try:
            return datetime.strptime(expiry_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _compute_max_pain(contracts: list[dict]) -> float | None:
    """
    Find the strike price where total option payout is minimized
    (= maximum pain for option holders, options market makers profit most here).

    At a given expiry price P:
      total_payout = sum_calls(OI * max(0, P - K)) + sum_puts(OI * max(0, K - P))
    """
    strikes = sorted(set(c["strike"] for c in contracts))
    if len(strikes) < 3:
        return None

    min_payout = float("inf")
    max_pain = strikes[len(strikes) // 2]

    for test_price in strikes:
        payout = 0.0
        for c in contracts:
            oi = c.get("open_interest") or 0
            if c["option_type"] == "C":
                payout += oi * max(0.0, test_price - c["strike"])
            else:
                payout += oi * max(0.0, c["strike"] - test_price)
        if payout < min_payout:
            min_payout = payout
            max_pain = test_price

    return max_pain


class OptionsInput(BaseModel):
    symbol: str = Field(description="Crypto symbol: BTC or ETH (Deribit only supports these)")
    max_expiries: int = Field(default=3, description="Number of nearest expiries to analyze")


class DeribitOptionsTool(BaseTool):
    name: str = "get_deribit_options_data"
    description: str = (
        "Fetch Bitcoin or Ethereum options market data from Deribit (free, no API key). "
        "Returns: put/call ratio by open interest and volume (>1 = bearish hedging, <0.7 = bullish call buying), "
        "max pain price per expiry (price level that minimizes total option payout — acts as a gravity point), "
        "implied volatility term structure (rising IV = uncertainty ahead), "
        "and top open-interest strikes (price levels the market is most focused on). "
        "Options market often leads spot price by 24-72 hours."
    )
    args_schema: type[BaseModel] = OptionsInput

    def _run(self, symbol: str, max_expiries: int = 3) -> str:
        currency = symbol.upper()
        if currency not in ("BTC", "ETH"):
            return json.dumps({
                "error": f"Deribit options only available for BTC and ETH, not {currency}. "
                         "Skip this tool for other symbols."
            })

        try:
            resp = requests.get(
                f"{DERIBIT_BASE}/get_book_summary_by_currency",
                params={"currency": currency, "kind": "option"},
                timeout=15,
            )
            resp.raise_for_status()
            summaries = resp.json().get("result", [])
        except Exception as exc:
            return json.dumps({"symbol": currency, "error": f"Deribit API error: {exc}"})

        if not summaries:
            return json.dumps({"symbol": currency, "error": "No options data returned"})

        # Parse all valid, unexpired contracts
        now = datetime.now(tz=timezone.utc)
        contracts = []
        for s in summaries:
            parsed = _parse_instrument(s.get("instrument_name", ""))
            if not parsed:
                continue
            expiry_dt = _expiry_to_dt(parsed["expiry_str"])
            if not expiry_dt or expiry_dt <= now:
                continue
            contracts.append({
                **parsed,
                "days_to_expiry": (expiry_dt - now).days,
                "open_interest": s.get("open_interest") or 0,
                "volume_24h": s.get("volume") or 0,
                "mark_iv": s.get("mark_iv") or 0,
                "underlying_price": s.get("underlying_price") or 0,
            })

        if not contracts:
            return json.dumps({"symbol": currency, "error": "No valid unexpired options found"})

        underlying = contracts[0]["underlying_price"]

        # Sort and pick nearest expiries
        all_expiries = sorted(
            set(c["expiry_str"] for c in contracts),
            key=lambda e: _expiry_to_dt(e) or datetime.max.replace(tzinfo=timezone.utc),
        )
        target_expiries = all_expiries[:max_expiries]
        filtered = [c for c in contracts if c["expiry_str"] in target_expiries]

        # --- Put / Call Ratio ---
        calls_oi  = sum(c["open_interest"] for c in filtered if c["option_type"] == "C")
        puts_oi   = sum(c["open_interest"] for c in filtered if c["option_type"] == "P")
        calls_vol = sum(c["volume_24h"]    for c in filtered if c["option_type"] == "C")
        puts_vol  = sum(c["volume_24h"]    for c in filtered if c["option_type"] == "P")

        pc_oi  = round(puts_oi  / calls_oi,  3) if calls_oi  else None
        pc_vol = round(puts_vol / calls_vol, 3) if calls_vol else None

        def _pc_signal(ratio):
            if ratio is None:
                return "N/A"
            if ratio > 1.3:
                return "strongly bearish - heavy put buying, market hedging hard downside"
            if ratio > 1.0:
                return "mildly bearish - more puts than calls, cautious sentiment"
            if ratio < 0.6:
                return "strongly bullish - call-heavy positioning, market expects upside"
            if ratio < 0.8:
                return "mildly bullish - slight call bias"
            return "neutral (P/C ~1.0)"

        # --- IV Term Structure ---
        iv_by_expiry = {}
        for expiry in target_expiries:
            ec = [c for c in filtered if c["expiry_str"] == expiry]
            ivs = [c["mark_iv"] for c in ec if c["mark_iv"] > 0]
            days = next((c["days_to_expiry"] for c in ec), 0)
            iv_by_expiry[expiry] = {
                "avg_iv_pct": round(sum(ivs) / len(ivs), 1) if ivs else None,
                "days_to_expiry": days,
            }

        iv_values = [v["avg_iv_pct"] for v in iv_by_expiry.values() if v["avg_iv_pct"]]
        iv_structure = "N/A"
        if len(iv_values) >= 2:
            iv_structure = (
                "backwardation - near-term IV elevated, short-term uncertainty high"
                if iv_values[0] > iv_values[-1]
                else "contango - far-term IV higher, stable near-term outlook"
            )

        # --- Max Pain per Expiry ---
        max_pain_map = {}
        for expiry in target_expiries:
            ec = [c for c in filtered if c["expiry_str"] == expiry]
            mp = _compute_max_pain(ec)
            if mp is not None:
                max_pain_map[expiry] = mp

        nearest_max_pain = max_pain_map.get(target_expiries[0]) if target_expiries else None
        max_pain_signal = "N/A"
        if nearest_max_pain and underlying:
            gap_pct = (nearest_max_pain - underlying) / underlying * 100
            direction = "above" if gap_pct > 0 else "below"
            gravity = "gravity pulls price UP toward max pain" if gap_pct > 0 else "gravity pulls price DOWN toward max pain"
            max_pain_signal = (
                f"price is {abs(gap_pct):.1f}% {direction} max pain "
                f"({nearest_max_pain:,.0f}) — {gravity} near expiry"
            )

        # --- Top OI Strikes (magnetic levels) ---
        oi_by_strike: dict[float, dict] = {}
        for c in filtered:
            k = c["strike"]
            if k not in oi_by_strike:
                oi_by_strike[k] = {"calls": 0.0, "puts": 0.0}
            if c["option_type"] == "C":
                oi_by_strike[k]["calls"] += c["open_interest"]
            else:
                oi_by_strike[k]["puts"] += c["open_interest"]

        top_strikes = sorted(
            [
                {
                    "strike": k,
                    "total_oi": round(v["calls"] + v["puts"], 1),
                    "call_oi": round(v["calls"], 1),
                    "put_oi": round(v["puts"], 1),
                    "bias": "call wall" if v["calls"] > v["puts"] * 1.5 else
                            "put wall" if v["puts"] > v["calls"] * 1.5 else "balanced",
                }
                for k, v in oi_by_strike.items()
            ],
            key=lambda x: x["total_oi"],
            reverse=True,
        )[:8]

        result = {
            "symbol": currency,
            "underlying_price": underlying,
            "expiries_analyzed": target_expiries,
            "put_call_ratio": {
                "by_open_interest": pc_oi,
                "by_volume_24h": pc_vol,
                "signal": _pc_signal(pc_oi),
            },
            "max_pain": {
                "by_expiry": max_pain_map,
                "nearest_expiry_max_pain": nearest_max_pain,
                "signal": max_pain_signal,
            },
            "iv_term_structure": {
                "by_expiry": iv_by_expiry,
                "structure_shape": iv_structure,
            },
            "top_oi_strikes": top_strikes,
        }
        return json.dumps(result, indent=2)
