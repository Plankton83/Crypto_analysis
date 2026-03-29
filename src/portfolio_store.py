"""
Paper portfolio persistence for the trading strategy agent.

Each symbol has its own file at  portfolio/{SYMBOL}_portfolio.json.

The portfolio is updated twice per run:
  1. START of main.py  — previous open position evaluated and closed (SL/TP check).
  2. END of main.py    — new trade plan from the strategy agent recorded.
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

PORTFOLIO_DIR = Path(__file__).parent.parent / "portfolio"
INITIAL_BALANCE = 100_000.0

# Fraction of portfolio to RISK per trade (not position size — see sizing formula)
_RISK_PCT = {"HIGH": 0.03, "MEDIUM": 0.015, "LOW": 0.0}
# Hard cap: position notional cannot exceed this share of total portfolio
_MAX_POSITION_PCT = 0.30


# ── persistence ───────────────────────────────────────────────────────────────

def _ensure_dir() -> None:
    PORTFOLIO_DIR.mkdir(exist_ok=True)


def _portfolio_path(symbol: str) -> Path:
    return PORTFOLIO_DIR / f"{symbol.upper()}_portfolio.json"


def load_portfolio(symbol: str) -> dict:
    """Load portfolio state or initialise a fresh $100,000 paper account."""
    _ensure_dir()
    path = _portfolio_path(symbol)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    return {
        "symbol":         symbol.upper(),
        "initial_balance": INITIAL_BALANCE,
        "cash":            INITIAL_BALANCE,
        "open_position":   None,
        "trade_history":   [],
        "portfolio_value_history": [{"date": now, "value": INITIAL_BALANCE}],
    }


def save_portfolio(symbol: str, portfolio: dict) -> None:
    _ensure_dir()
    with open(_portfolio_path(symbol), "w") as f:
        json.dump(portfolio, f, indent=2)


# ── valuation ─────────────────────────────────────────────────────────────────

def portfolio_value(portfolio: dict, current_price: float) -> float:
    """Total portfolio value: cash + open position mark-to-market."""
    cash = portfolio.get("cash", 0.0)
    pos  = portfolio.get("open_position")
    if not pos or current_price <= 0:
        return cash

    direction   = pos.get("direction", "LONG")
    size_coins  = pos.get("size_coins", 0.0)
    entry_price = pos.get("entry_price", current_price)
    size_usd    = pos.get("size_usd", 0.0)

    if direction == "LONG":
        position_value = size_coins * current_price
    else:  # SHORT — collateral locked, P&L added on top
        pnl = (entry_price - current_price) * size_coins
        position_value = size_usd + pnl

    return cash + max(position_value, 0.0)  # floor at 0


# ── OHLCV helper for SL/TP evaluation ─────────────────────────────────────────

def _fetch_ohlcv_since(symbol: str, since_iso: str) -> list[dict]:
    """Daily candles from Binance from `since_iso` to now (up to 365 candles)."""
    binance_symbol = symbol.upper()
    if not binance_symbol.endswith("USDT"):
        binance_symbol += "USDT"
    try:
        start_ms = int(
            datetime.fromisoformat(since_iso.replace("Z", "+00:00")).timestamp() * 1000
        )
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": binance_symbol, "interval": "1d",
                    "startTime": start_ms, "limit": 365},
            timeout=10,
        )
        resp.raise_for_status()
        return [
            {"high": float(c[2]), "low": float(c[3]), "close": float(c[4])}
            for c in resp.json()
        ]
    except Exception as exc:
        logger.warning("OHLCV fetch for SL/TP check failed: %s", exc)
        return []


# ── position lifecycle ────────────────────────────────────────────────────────

def evaluate_and_close_position(
    portfolio: dict,
    symbol: str,
    current_price: float,
) -> Optional[dict]:
    """
    If there is an open position, scan Binance candles since it was opened
    to check whether stop-loss or take-profit was triggered.

    Conservative assumption: within each daily candle, the adverse direction
    is tested first (low before high for longs, high before low for shorts).

    Returns the closed trade dict, or None if no position was open.
    """
    pos = portfolio.get("open_position")
    if not pos:
        return None

    direction   = pos["direction"]
    entry_price = pos["entry_price"]
    stop_loss   = pos["stop_loss"]
    tp1         = pos["take_profit_1"]
    tp2         = pos.get("take_profit_2")
    size_usd    = pos["size_usd"]
    size_coins  = pos["size_coins"]
    opened_at   = pos["opened_at"]

    candles    = _fetch_ohlcv_since(symbol, opened_at)
    exit_price = current_price
    outcome    = "closed_at_run"

    for c in candles:
        high, low = c["high"], c["low"]
        if direction == "LONG":
            # Assume worst: low hit before high on the same candle
            if low <= stop_loss:
                exit_price, outcome = stop_loss, "sl_hit"
                break
            if tp2 and high >= tp2:
                exit_price, outcome = tp2, "tp2_hit"
                break
            if high >= tp1:
                exit_price, outcome = tp1, "tp1_hit"
                break
        else:  # SHORT
            # Assume worst: high hit before low
            if high >= stop_loss:
                exit_price, outcome = stop_loss, "sl_hit"
                break
            if tp2 and low <= tp2:
                exit_price, outcome = tp2, "tp2_hit"
                break
            if low <= tp1:
                exit_price, outcome = tp1, "tp1_hit"
                break

    # P&L calculation
    if direction == "LONG":
        pnl_usd = (exit_price - entry_price) * size_coins
    else:
        pnl_usd = (entry_price - exit_price) * size_coins

    pnl_pct = round(pnl_usd / size_usd * 100, 2) if size_usd else 0.0

    trade = {
        **pos,
        "exit_price": round(exit_price, 2),
        "pnl_usd":    round(pnl_usd, 2),
        "pnl_pct":    pnl_pct,
        "outcome":    outcome,
        "closed_at":  datetime.now(tz=timezone.utc).isoformat(),
    }

    # Return collateral + P&L to cash (floor at 0)
    portfolio["cash"] = max(portfolio.get("cash", 0.0) + size_usd + pnl_usd, 0.0)
    portfolio["open_position"] = None
    portfolio["trade_history"].append(trade)

    # Equity curve snapshot
    total_val = portfolio_value(portfolio, current_price)
    portfolio["portfolio_value_history"].append({
        "date":  datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        "value": round(total_val, 2),
    })

    return trade


def record_trade_plan(
    portfolio: dict,
    trade_plan: dict,
    current_price: float,
    prediction_id: str = "",
) -> Optional[dict]:
    """
    Open a new simulated position from the strategy agent's trade plan.

    Position sizing:
      risk_amount  = cash * risk_pct(confidence)
      size_usd     = risk_amount / sl_distance   [capped at 30% of portfolio]
      size_coins   = size_usd / entry_price

    Returns the new position dict, or None if the plan calls for no trade.
    """
    direction  = trade_plan.get("direction", "NO_TRADE").upper()
    confidence = trade_plan.get("confidence", "LOW").upper()
    signal     = trade_plan.get("signal", "NEUTRAL").upper()

    if direction == "NO_TRADE" or signal == "NEUTRAL" or confidence == "LOW":
        return None

    try:
        entry_price = float(trade_plan["entry_price"])
        stop_loss   = float(trade_plan["stop_loss"])
        tp1         = float(trade_plan["take_profit_1"])
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Invalid trade plan fields: %s — %s", trade_plan, exc)
        return None

    tp2_raw = trade_plan.get("take_profit_2")
    tp2     = float(tp2_raw) if tp2_raw else None

    sl_distance = abs(entry_price - stop_loss) / entry_price
    if sl_distance <= 0 or tp1 <= 0:
        logger.warning("Degenerate trade plan (zero SL distance or TP): %s", trade_plan)
        return None

    cash        = portfolio.get("cash", INITIAL_BALANCE)
    total_val   = portfolio_value(portfolio, current_price)
    risk_pct    = _RISK_PCT.get(confidence, 0.015)
    risk_amount = cash * risk_pct
    size_usd    = min(risk_amount / sl_distance, total_val * _MAX_POSITION_PCT)
    size_usd    = min(size_usd, cash)          # cannot spend more than available cash
    size_coins  = size_usd / entry_price

    if size_usd < 1.0:
        logger.info("Computed position size too small ($%.2f) — skipping trade", size_usd)
        return None

    position = {
        "prediction_id": prediction_id,
        "direction":     direction,
        "signal":        signal,
        "confidence":    confidence,
        "entry_price":   round(entry_price, 2),
        "stop_loss":     round(stop_loss, 2),
        "take_profit_1": round(tp1, 2),
        "take_profit_2": round(tp2, 2) if tp2 else None,
        "size_usd":      round(size_usd, 2),
        "size_coins":    round(size_coins, 6),
        "risk_pct":      round(risk_pct * 100, 1),
        "opened_at":     datetime.now(tz=timezone.utc).isoformat(),
    }

    portfolio["cash"]         -= size_usd
    portfolio["open_position"] = position
    return position


# ── parsing ───────────────────────────────────────────────────────────────────

def parse_trade_plan_from_text(text: str) -> Optional[dict]:
    """Extract the TRADE_PLAN_JSON sentinel from the strategy agent's output."""
    m = re.search(r"TRADE_PLAN_JSON:\s*(\{[^}]+\})", text)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse TRADE_PLAN_JSON: %s", exc)
        return None


# ── summary ───────────────────────────────────────────────────────────────────

def get_portfolio_summary(portfolio: dict, current_price: float) -> dict:
    """Human-readable portfolio snapshot for the agent tool and HTML report."""
    total_val     = portfolio_value(portfolio, current_price)
    initial       = portfolio.get("initial_balance", INITIAL_BALANCE)
    total_pnl     = round(total_val - initial, 2)
    total_pnl_pct = round(total_pnl / initial * 100, 2)

    history = portfolio.get("trade_history", [])
    wins    = sum(1 for t in history if t.get("pnl_usd", 0) > 0)
    total   = len(history)

    return {
        "symbol":           portfolio.get("symbol"),
        "initial_balance":  initial,
        "current_value":    round(total_val, 2),
        "total_pnl_usd":    total_pnl,
        "total_pnl_pct":    total_pnl_pct,
        "cash_available":   round(portfolio.get("cash", 0.0), 2),
        "open_position":    portfolio.get("open_position"),
        "trades_total":     total,
        "trades_won":       wins,
        "win_rate_pct":     round(wins / total * 100, 1) if total else None,
        "last_3_trades":    history[-3:] if history else [],
        "equity_curve":     portfolio.get("portfolio_value_history", []),
    }
