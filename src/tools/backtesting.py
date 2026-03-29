"""
Backtesting tools:
- BacktestAnalysisTool: evaluates past system predictions against actual outcomes.
- HistoricalPatternTool: scans 3 years of daily data for historical analogues that
  match the current technical profile and reports forward returns.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Type

import numpy as np
import pandas as pd
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from src.prediction_store import load_predictions


def _fetch_price_at_datetime(symbol: str, target_dt: datetime) -> float | None:
    """
    Fetch the closing price of a 1-hour candle at (or just after) the target datetime.
    Uses Binance public klines endpoint — no API key required.
    """
    binance_symbol = symbol.upper()
    if not binance_symbol.endswith("USDT"):
        binance_symbol += "USDT"

    # Convert to milliseconds; ensure UTC
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=timezone.utc)
    start_ms = int(target_dt.timestamp() * 1000)
    end_ms = start_ms + 2 * 3600 * 1000  # 2-hour window

    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": binance_symbol,
        "interval": "1h",
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": 1,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            return float(data[0][4])  # index 4 = close price
    except Exception:
        pass
    return None


class BacktestInput(BaseModel):
    symbol: str = Field(..., description="Crypto symbol to backtest, e.g. 'BTC'")
    num_predictions: int = Field(
        default=5,
        description="How many past predictions to evaluate (most recent first)",
    )


class BacktestAnalysisTool(BaseTool):
    name: str = "Backtesting Analysis Tool"
    description: str = (
        "Loads past predictions for a cryptocurrency from the predictions store and "
        "compares each prediction's signal (BUY/SELL/NEUTRAL) against what actually "
        "happened to the price 24 hours and 7 days later. "
        "Returns a structured accuracy report including hit rate, average price errors, "
        "and individual prediction outcomes. Use this to calibrate confidence in the "
        "current prediction based on historical track record."
    )
    args_schema: type[BaseModel] = BacktestInput

    def _run(self, symbol: str, num_predictions: int = 5) -> str:
        predictions = load_predictions(symbol, limit=num_predictions)

        if not predictions:
            return json.dumps(
                {
                    "status": "no_history",
                    "symbol": symbol.upper(),
                    "message": (
                        f"No past predictions found for {symbol}. "
                        "This is the first analysis run — no historical accuracy data available."
                    ),
                    "predictions_analyzed": 0,
                }
            )

        now = datetime.now(tz=timezone.utc)
        results = []

        for pred in predictions:
            pred_dt = datetime.fromisoformat(pred["timestamp"])
            if pred_dt.tzinfo is None:
                pred_dt = pred_dt.replace(tzinfo=timezone.utc)

            price_at_pred = pred.get("price_at_prediction")
            result: dict = {
                "id": pred["id"],
                "date": pred["timestamp"][:16].replace("T", " "),
                "signal": pred["signal"],
                "confidence": pred["confidence"],
                "price_at_prediction": price_at_pred,
            }

            for label, hours in [("24h", 24), ("7d", 168)]:
                target_dt = pred_dt + timedelta(hours=hours)

                if target_dt > now:
                    result[f"outcome_{label}"] = "pending"
                    result[f"price_{label}"] = None
                    result[f"change_{label}_pct"] = None
                    result[f"signal_correct_{label}"] = None
                    continue

                actual_price = _fetch_price_at_datetime(symbol, target_dt)
                result[f"price_{label}"] = actual_price

                if actual_price and price_at_pred:
                    change_pct = (actual_price - price_at_pred) / price_at_pred * 100
                    result[f"change_{label}_pct"] = round(change_pct, 2)

                    signal = pred["signal"]
                    if signal in ("BUY", "STRONG_BUY"):
                        correct = change_pct > 1.0
                    elif signal in ("SELL", "STRONG_SELL"):
                        correct = change_pct < -1.0
                    elif signal == "NEUTRAL":
                        correct = abs(change_pct) < 3.0
                    else:
                        correct = None

                    result[f"signal_correct_{label}"] = correct
                    result[f"outcome_{label}"] = "evaluated"
                else:
                    result[f"change_{label}_pct"] = None
                    result[f"signal_correct_{label}"] = None
                    result[f"outcome_{label}"] = "price_unavailable"

            results.append(result)

        # Aggregate stats
        evaluated_24h = [r for r in results if r.get("outcome_24h") == "evaluated"]
        evaluated_7d = [r for r in results if r.get("outcome_7d") == "evaluated"]
        correct_24h = [r for r in evaluated_24h if r.get("signal_correct_24h") is True]
        correct_7d = [r for r in evaluated_7d if r.get("signal_correct_7d") is True]

        def avg_change(rows, key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        summary = {
            "symbol": symbol.upper(),
            "predictions_loaded": len(predictions),
            "evaluated_24h": len(evaluated_24h),
            "evaluated_7d": len(evaluated_7d),
            "signal_accuracy_24h": (
                f"{len(correct_24h)}/{len(evaluated_24h)}" if evaluated_24h else "N/A"
            ),
            "signal_accuracy_7d": (
                f"{len(correct_7d)}/{len(evaluated_7d)}" if evaluated_7d else "N/A"
            ),
            "accuracy_rate_24h_pct": (
                round(len(correct_24h) / len(evaluated_24h) * 100, 1) if evaluated_24h else None
            ),
            "accuracy_rate_7d_pct": (
                round(len(correct_7d) / len(evaluated_7d) * 100, 1) if evaluated_7d else None
            ),
            "avg_price_change_24h_pct": avg_change(evaluated_24h, "change_24h_pct"),
            "avg_price_change_7d_pct": avg_change(evaluated_7d, "change_7d_pct"),
            "individual_results": results,
        }

        return json.dumps(summary, indent=2)


# ── Historical Pattern Matching ───────────────────────────────────────────────

def _fetch_daily_ohlcv(symbol: str, days: int = 1825) -> pd.DataFrame:
    """
    Fetch up to `days` daily candles from Binance (no API key required).
    Binance returns at most 1000 candles per call, so for >1000 days this
    function pages backwards in time until the requested depth is reached.
    Default 1825 days ≈ 5 years.
    """
    pair = f"{symbol.upper()}USDT"
    url  = "https://api.binance.com/api/v3/klines"
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]

    batches: list[list] = []
    end_ms: int | None = None   # None → fetch most recent batch first

    while sum(len(b) for b in batches) < days:
        params: dict = {"symbol": pair, "interval": "1d", "limit": 1000}
        if end_ms is not None:
            params["endTime"] = end_ms

        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        batch = resp.json()

        if not batch:
            break

        batches.insert(0, batch)          # prepend so result is chronological
        end_ms = int(batch[0][0]) - 1     # page before the oldest candle fetched

        if len(batch) < 1000:             # reached the start of available history
            break

    if not batches:
        raise ValueError(f"No OHLCV data returned for {pair}")

    # Flatten, keep most recent `days` candles
    all_rows = [row for b in batches for row in b][-days:]

    df = pd.DataFrame(all_rows, columns=cols)
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.strftime("%Y-%m-%d")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df.reset_index(drop=True)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    macd = _ema(series, fast) - _ema(series, slow)
    return macd - _ema(macd, sig)


def _bucket(val: float, edges: list[float]) -> int:
    """Return the index of the bucket that val falls into."""
    for i, edge in enumerate(edges):
        if val < edge:
            return i
    return len(edges)


def _profile(close: pd.Series, i: int,
             rsi_s: pd.Series, ema20_s: pd.Series, ema50_s: pd.Series,
             ema200_s: pd.Series, macd_s: pd.Series) -> dict:
    """Extract a comparable feature profile at position i."""
    px = close.iloc[i]

    mom7  = (px / close.iloc[i - 7]  - 1) * 100 if i >= 7  else None
    mom30 = (px / close.iloc[i - 30] - 1) * 100 if i >= 30 else None

    return {
        "rsi_bucket":      _bucket(rsi_s.iloc[i],  [30, 45, 55, 70]),   # 0-4
        "mom7_bucket":     _bucket(mom7 or 0,       [-5, -1, 1, 5]),     # 0-4
        "mom30_bucket":    _bucket(mom30 or 0,      [-15, -5, 5, 15]),   # 0-4
        "above_ema20":     int(px > ema20_s.iloc[i]),
        "above_ema50":     int(px > ema50_s.iloc[i]),
        "above_ema200":    int(px > ema200_s.iloc[i]),
        "macd_hist_pos":   int(macd_s.iloc[i] > 0),
        "macd_hist_up":    int(macd_s.iloc[i] > macd_s.iloc[i - 3]) if i >= 3 else 0,
    }


def _score(a: dict, b: dict) -> int:
    """Count how many profile features match between two windows."""
    keys = ["rsi_bucket", "mom7_bucket", "mom30_bucket",
            "above_ema20", "above_ema50", "above_ema200",
            "macd_hist_pos", "macd_hist_up"]
    return sum(a.get(k) == b.get(k) for k in keys)


class HistoricalPatternInput(BaseModel):
    symbol: str = Field(description="Crypto symbol e.g. BTC, ETH")
    min_match_score: int = Field(
        default=6,
        description="Minimum feature matches required (out of 8) to count as an analogue",
    )


class HistoricalPatternTool(BaseTool):
    name: str = "find_historical_analogues"
    description: str = (
        "Scan approximately 5 years of daily price history to find past periods whose "
        "technical conditions most closely match today's market. Matching features: "
        "RSI level, 7-day and 30-day momentum, price position vs EMA20/50/200, and "
        "MACD histogram direction. For each analogue, reports what happened to price "
        "7, 14, and 30 days later, plus aggregate statistics (median return, % bullish). "
        "Helps answer: 'When the market looked like this before, what happened next?'"
    )
    args_schema: Type[BaseModel] = HistoricalPatternInput

    def _run(self, symbol: str, min_match_score: int = 6) -> str:
        try:
            df = _fetch_daily_ohlcv(symbol, days=1825)
        except Exception as e:
            return json.dumps({"error": f"Failed to fetch OHLCV data: {e}"})

        close = df["close"]
        n = len(close)

        if n < 250:
            return json.dumps({"error": f"Not enough history: only {n} candles available"})

        # Vectorised indicators
        rsi_s    = _rsi(close)
        ema20_s  = _ema(close, 20)
        ema50_s  = _ema(close, 50)
        ema200_s = _ema(close, 200)
        macd_s   = _macd_hist(close)

        # Current profile (last row)
        current_i = n - 1
        current = _profile(close, current_i, rsi_s, ema20_s, ema50_s, ema200_s, macd_s)
        current_price = float(close.iloc[current_i])
        current_date  = df["date"].iloc[current_i]

        # Scan history — start at 210 (enough warmup) and stop at n-31 (need 30d forward)
        candidates: list[tuple[int, int]] = []  # (score, index)
        for i in range(210, n - 31):
            prof = _profile(close, i, rsi_s, ema20_s, ema50_s, ema200_s, macd_s)
            s = _score(current, prof)
            if s >= min_match_score:
                candidates.append((s, i))

        # Sort by score desc, then enforce 30-day spacing to avoid clustered analogues
        candidates.sort(key=lambda x: (-x[0], -x[1]))
        selected: list[tuple[int, int]] = []
        used_indices: list[int] = []
        for score, idx in candidates:
            if all(abs(idx - u) >= 30 for u in used_indices):
                selected.append((score, idx))
                used_indices.append(idx)
            if len(selected) >= 15:
                break

        if not selected:
            return json.dumps({
                "symbol": symbol.upper(),
                "current_date": current_date,
                "analogues_found": 0,
                "message": (
                    f"No historical periods found with {min_match_score}+ matching features. "
                    "Try lowering min_match_score."
                ),
                "current_profile": current,
            })

        # Forward returns for each analogue
        analogue_records: list[dict] = []
        fwd_7:  list[float] = []
        fwd_14: list[float] = []
        fwd_30: list[float] = []

        for score, idx in selected:
            px0 = float(close.iloc[idx])
            r7  = (float(close.iloc[idx + 7])  / px0 - 1) * 100
            r14 = (float(close.iloc[idx + 14]) / px0 - 1) * 100
            r30 = (float(close.iloc[idx + 30]) / px0 - 1) * 100
            fwd_7.append(r7)
            fwd_14.append(r14)
            fwd_30.append(r30)

            # Human-readable profile summary for this analogue
            rsi_val = float(rsi_s.iloc[idx])
            mom7_val  = (px0 / float(close.iloc[idx - 7])  - 1) * 100 if idx >= 7  else 0
            mom30_val = (px0 / float(close.iloc[idx - 30]) - 1) * 100 if idx >= 30 else 0

            analogue_records.append({
                "date":          df["date"].iloc[idx],
                "price":         round(px0, 2),
                "match_score":   f"{score}/8",
                "rsi":           round(rsi_val, 1),
                "mom_7d_pct":    round(mom7_val, 1),
                "mom_30d_pct":   round(mom30_val, 1),
                "fwd_7d_pct":    round(r7,  2),
                "fwd_14d_pct":   round(r14, 2),
                "fwd_30d_pct":   round(r30, 2),
                "outcome_7d":    "up" if r7  > 1 else ("down" if r7  < -1 else "flat"),
                "outcome_30d":   "up" if r30 > 3 else ("down" if r30 < -3 else "flat"),
            })

        def _stats(vals: list[float]) -> dict:
            arr = np.array(vals)
            return {
                "median_pct":   round(float(np.median(arr)), 2),
                "mean_pct":     round(float(np.mean(arr)),   2),
                "best_pct":     round(float(np.max(arr)),    2),
                "worst_pct":    round(float(np.min(arr)),    2),
                "pct_positive": round(float(np.mean(arr > 0)) * 100, 1),
            }

        # Current profile in human-readable form
        rsi_now   = float(rsi_s.iloc[current_i])
        mom7_now  = (current_price / float(close.iloc[current_i - 7])  - 1) * 100
        mom30_now = (current_price / float(close.iloc[current_i - 30]) - 1) * 100

        result = {
            "symbol":        symbol.upper(),
            "current_date":  current_date,
            "current_price": round(current_price, 2),
            "current_profile": {
                "rsi_14":          round(rsi_now,   1),
                "mom_7d_pct":      round(mom7_now,  1),
                "mom_30d_pct":     round(mom30_now, 1),
                "above_ema20":     bool(current["above_ema20"]),
                "above_ema50":     bool(current["above_ema50"]),
                "above_ema200":    bool(current["above_ema200"]),
                "macd_hist_pos":   bool(current["macd_hist_pos"]),
                "macd_hist_up":    bool(current["macd_hist_up"]),
            },
            "history_scanned": f"~{n} trading days (~{n // 365} years)",
            "analogues_found": len(selected),
            "min_match_score": f"{min_match_score}/8 features",
            "forward_return_stats": {
                "7d":  _stats(fwd_7),
                "14d": _stats(fwd_14),
                "30d": _stats(fwd_30),
            },
            "interpretation": _interpret(fwd_7, fwd_14, fwd_30, len(selected)),
            "top_analogues": analogue_records[:10],
        }
        return json.dumps(result, indent=2)


def _interpret(fwd_7: list[float], fwd_14: list[float], fwd_30: list[float], n: int) -> str:
    """Generate a plain-English summary of what analogues suggest."""
    med7  = float(np.median(fwd_7))
    med30 = float(np.median(fwd_30))
    pct_up_7  = float(np.mean(np.array(fwd_7)  > 0)) * 100
    pct_up_30 = float(np.mean(np.array(fwd_30) > 0)) * 100

    bias_7  = "bullish" if med7  > 1  else ("bearish" if med7  < -1  else "neutral")
    bias_30 = "bullish" if med30 > 3  else ("bearish" if med30 < -3  else "neutral")
    conf_7  = "high" if pct_up_7  > 65 or pct_up_7  < 35 else "low"
    conf_30 = "high" if pct_up_30 > 65 or pct_up_30 < 35 else "low"

    return (
        f"Across {n} historical analogues: "
        f"7-day outlook is {bias_7} (median {med7:+.1f}%, {pct_up_7:.0f}% ended higher, confidence {conf_7}). "
        f"30-day outlook is {bias_30} (median {med30:+.1f}%, {pct_up_30:.0f}% ended higher, confidence {conf_30})."
    )
