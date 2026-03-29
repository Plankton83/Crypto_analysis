#!/usr/bin/env python3
"""
Crypto Prediction CrewAI — entry point.

Usage:
    python main.py                              # Analyze BTC (default, OpenAI)
    python main.py ETH                          # Analyze ETH
    python main.py SOL --model gpt-4o           # Use specific OpenAI model
    python main.py BTC --ollama                 # Use local Ollama (llama3)
    python main.py BTC --gemini                 # Use Gemini (gemini-2.0-flash)
    python main.py BTC --gemini --gemini-model gemini-2.5-pro-exp-03-25
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crypto price prediction using CrewAI multi-agent analysis"
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default=os.getenv("DEFAULT_SYMBOL", "BTC"),
        help="Crypto symbol to analyze (default: BTC)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="OpenAI model name (default: gpt-5.4-mini)",
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Use local Ollama instead of OpenAI (model: llama3)",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3",
        help="Ollama model name (default: llama3)",
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Use Google Gemini instead of OpenAI (requires GEMINI_API_KEY in .env)",
    )
    parser.add_argument(
        "--gemini-model",
        default="gemini-3-flash-preview",
        help="Gemini model name (default: gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save report to file (e.g. --output report.txt)",
    )
    parser.add_argument(
        "--email",
        action="store_true",
        help="Send the HTML report by email (configure EMAIL_* vars in .env)",
    )
    return parser.parse_args()


def build_llm(args):
    """Build a CrewAI LLM instance (backed by LiteLLM — no extra packages needed)."""
    from crewai import LLM

    if args.ollama:
        print(f"[*] Using local Ollama model: {args.ollama_model}")
        return LLM(model=f"ollama/{args.ollama_model}", base_url="http://localhost:11434")

    if args.gemini:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[!] ERROR: GEMINI_API_KEY not set. Add it to your .env file.")
            print("    Get a free key at https://aistudio.google.com/apikey")
            sys.exit(1)
        model = f"gemini/{args.gemini_model}"
        print(f"[*] Using Gemini model: {model}")
        return LLM(model=model, api_key=api_key, temperature=0.1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[!] ERROR: OPENAI_API_KEY not set. Add it to your .env file.")
        print("    Alternatively, use --gemini or --ollama for other providers.")
        sys.exit(1)

    print(f"[*] Using OpenAI model: {args.model}")
    return LLM(model=args.model, api_key=api_key, temperature=0.1)


def fetch_current_price(symbol: str) -> float | None:
    """Fetch the current price from Binance public API (no key required)."""
    import requests

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
        return float(data["price"]) if "price" in data else None
    except Exception:
        return None


def print_header(symbol: str):
    width = 60
    print("=" * width)
    print(f"  CRYPTO PREDICTION CREW — {symbol}")
    print(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * width)
    print()
    print("Agents launching:")
    print("  [0] Prediction Accuracy Analyst  -> Historical accuracy calibration")
    print("  [1] Macro Market Analyst         -> S&P500 + Nasdaq + VIX + Yields + Gold + CME")
    print("  [2] On-Chain Network Analyst     -> Bitcoin mempool + network activity")
    print("  [3] Market Data Analyst          -> Binance OHLCV + CoinGecko + stablecoin dom.")
    print("  [4] Technical Analysis           -> Weekly/Daily/4h indicators + Volume Profile")
    print("  [5] Sentiment & On-Chain         -> Fear&Greed + News RSS + Reddit")
    print("  [6] Derivatives & Options        -> Funding + OI + Liquidations + Deribit options")
    print("  [7] Chief Strategist             -> Agent vote table + synthesis + prediction")
    print("  [8] Paper Trading Strategist     -> Trade plan + paper portfolio management")
    print("  [9] Report Formatter             -> HTML report with charts saved to reports/")
    print()


def main():
    args = parse_args()
    symbol = args.symbol.upper()

    print_header(symbol)

    llm = build_llm(args)

    from src.crew import run_crypto_analysis
    from src.prediction_store import save_prediction
    from src.portfolio_store import (
        load_portfolio, save_portfolio,
        evaluate_and_close_position,
        parse_trade_plan_from_text,
        record_trade_plan,
        portfolio_value,
    )

    # ── 1. Snapshot current price ──────────────────────────────────────────────
    current_price = fetch_current_price(symbol)
    if current_price:
        print(f"[*] Current {symbol} price: ${current_price:,.2f}")

    # ── 2. Evaluate & close previous position (before analysis starts) ─────────
    portfolio = load_portfolio(symbol)
    closed_trade = evaluate_and_close_position(portfolio, symbol, current_price or 0.0)
    if closed_trade:
        direction = closed_trade.get("direction", "?")
        pnl       = closed_trade.get("pnl_usd", 0.0)
        outcome   = closed_trade.get("outcome", "?")
        pnl_sign  = "+" if pnl >= 0 else ""
        print(f"[*] Previous {direction} position closed — "
              f"outcome: {outcome}, P&L: {pnl_sign}${pnl:,.2f}")
        save_portfolio(symbol, portfolio)
    total_val = portfolio_value(portfolio, current_price or 0.0)
    print(f"[*] Paper portfolio value: ${total_val:,.2f} "
          f"(cash: ${portfolio['cash']:,.2f})\n")

    # ── 3. Run the analysis crew ───────────────────────────────────────────────
    try:
        report, strategy_text, html_path = run_crypto_analysis(symbol, llm=llm)
    except KeyboardInterrupt:
        print("\n[!] Analysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] Error during analysis: {e}")
        raise

    print("\n" + "=" * 60)
    print("  FINAL PREDICTION REPORT")
    print("=" * 60)
    print(report)

    # ── 4. Persist prediction for backtesting ──────────────────────────────────
    prediction_id = save_prediction(symbol, report, price_at_prediction=current_price)
    print(f"\n[*] Prediction saved (id: {prediction_id}) → predictions/{symbol}_predictions.json")

    # ── 5. Record trade plan from strategy agent ───────────────────────────────
    trade_plan = parse_trade_plan_from_text(strategy_text)
    if trade_plan:
        new_position = record_trade_plan(
            portfolio, trade_plan,
            current_price=current_price or 0.0,
            prediction_id=prediction_id,
        )
        save_portfolio(symbol, portfolio)
        if new_position:
            direction = new_position["direction"]
            size_usd  = new_position["size_usd"]
            entry     = new_position["entry_price"]
            sl        = new_position["stop_loss"]
            tp1       = new_position["take_profit_1"]
            print(f"[*] New {direction} position opened — "
                  f"${size_usd:,.2f} @ ${entry:,.2f} | SL: ${sl:,.2f} | TP1: ${tp1:,.2f}")
        else:
            print("[*] Strategy: NO TRADE this run (NEUTRAL / LOW confidence)")
    else:
        print("[!] Could not parse trade plan from strategy output — portfolio unchanged")

    new_total = portfolio_value(portfolio, current_price or 0.0)
    print(f"[*] Paper portfolio after trade: ${new_total:,.2f}")

    # ── 6. HTML report ─────────────────────────────────────────────────────────
    if html_path:
        print(f"[*] HTML report saved → {html_path}")

    if args.email and html_path:
        from src.email_sender import send_report
        try:
            send_report(html_path, symbol)
            print(f"[*] HTML report emailed to {os.getenv('EMAIL_TO')}")
        except Exception as e:
            print(f"[!] Failed to send email: {e}")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            f.write(f"# Crypto Prediction Report — {symbol}\n")
            f.write(f"# Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
            f.write(report)
        print(f"\n[*] Report saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
