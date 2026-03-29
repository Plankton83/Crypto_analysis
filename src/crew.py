"""
Crypto prediction crew assembly and execution.
"""

from crewai import Crew, Process

from .agents import (
    create_backtesting_agent,
    create_macro_agent,
    create_onchain_agent,
    create_market_data_agent,
    create_technical_analysis_agent,
    create_sentiment_agent,
    create_derivatives_agent,
    create_prediction_agent,
    create_trading_strategy_agent,
    create_report_formatter_agent,
)
from .tasks import (
    create_backtesting_task,
    create_macro_task,
    create_onchain_task,
    create_market_data_task,
    create_technical_analysis_task,
    create_sentiment_task,
    create_derivatives_task,
    create_prediction_task,
    create_trading_strategy_task,
    create_report_formatter_task,
)


def run_crypto_analysis(symbol: str, llm=None) -> tuple[str, str, str]:
    """
    Run the full crypto prediction crew for the given symbol.

    Pipeline (sequential):
      0. Backtesting       — historical accuracy calibration
      1. Macro             — S&P500, Nasdaq, VIX, yields, gold, oil, DXY, CME futures
      2. On-Chain          — Bitcoin mempool, mining health, network activity, LN, CoinMetrics
      3. Market Data       — OHLCV, market cap, order book, stablecoin dominance
      4. Technical         — weekly/daily/4h indicators + Volume Profile
      5. Sentiment         — Fear&Greed, news RSS, Reddit, on-chain social
      6. Derivatives       — funding, OI, liquidations, Deribit options
      7. Prediction        — synthesises all above + agent vote table
      8. Trading Strategy  — converts prediction to trade plan + paper portfolio update
      9. Report Formatter  — renders HTML report to reports/ folder

    Args:
        symbol: Crypto symbol (e.g. 'BTC', 'ETH', 'SOL')
        llm: Optional custom LLM instance.

    Returns:
        Tuple of (prediction_text, strategy_text, html_report_path).
    """
    symbol = symbol.upper()

    # Instantiate agents
    backtesting_agent  = create_backtesting_agent(llm)
    macro_agent        = create_macro_agent(llm)
    onchain_agent      = create_onchain_agent(llm)
    market_agent       = create_market_data_agent(llm)
    technical_agent    = create_technical_analysis_agent(llm)
    sentiment_agent    = create_sentiment_agent(llm)
    derivatives_agent  = create_derivatives_agent(llm)
    prediction_agent   = create_prediction_agent(llm)
    strategy_agent     = create_trading_strategy_agent(llm)
    formatter_agent    = create_report_formatter_agent(llm)

    # Instantiate tasks
    backtesting_task = create_backtesting_task(backtesting_agent, symbol)
    macro_task       = create_macro_task(macro_agent,             symbol)
    onchain_task     = create_onchain_task(onchain_agent,         symbol)
    market_task      = create_market_data_task(market_agent,      symbol)
    technical_task   = create_technical_analysis_task(technical_agent, symbol)
    sentiment_task   = create_sentiment_task(sentiment_agent,     symbol)
    derivatives_task = create_derivatives_task(derivatives_agent, symbol)

    prediction_task = create_prediction_task(
        prediction_agent,
        symbol,
        context_tasks=[
            backtesting_task,
            macro_task,
            onchain_task,
            market_task,
            technical_task,
            sentiment_task,
            derivatives_task,
        ],
    )

    strategy_task = create_trading_strategy_task(strategy_agent, symbol, prediction_task)

    formatter_task = create_report_formatter_task(
        formatter_agent, symbol, prediction_task, strategy_task
    )

    crew = Crew(
        agents=[
            backtesting_agent, macro_agent, onchain_agent,
            market_agent, technical_agent, sentiment_agent,
            derivatives_agent, prediction_agent, strategy_agent, formatter_agent,
        ],
        tasks=[
            backtesting_task, macro_task, onchain_task,
            market_task, technical_task, sentiment_task,
            derivatives_task, prediction_task, strategy_task, formatter_task,
        ],
        process=Process.sequential,
        verbose=True,
        memory=False,
        embedder=None,
    )

    result = crew.kickoff()

    # tasks_output: [..., prediction(-3), strategy(-2), formatter(-1)]
    prediction_text = result.tasks_output[-3].raw
    strategy_text   = result.tasks_output[-2].raw
    html_path       = result.tasks_output[-1].raw.strip()
    return prediction_text, strategy_text, html_path
