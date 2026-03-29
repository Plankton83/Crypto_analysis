"""
CrewAI task definitions for the crypto prediction crew.

Every research task ends with a standardised AGENT VOTE block so the Chief
Strategist can tabulate all signals before synthesising the final prediction.

Vote format (last section of each research task output):
  AGENT VOTE:
  SIGNAL: [BUY | SELL | NEUTRAL] | Confidence: [HIGH | MEDIUM | LOW] | Reason: <one sentence>
"""

from crewai import Task

_VOTE_INSTRUCTION = """
**AGENT VOTE (required — must be the last section of your output):**
Summarise your entire analysis into a single standardised line:
SIGNAL: [BUY | SELL | NEUTRAL] | Confidence: [HIGH | MEDIUM | LOW] | Reason: <one concise sentence>
"""


# ── Backtesting ──────────────────────────────────────────────────────────────

def create_backtesting_task(agent, symbol: str) -> Task:
    return Task(
        description=f"""
        Produce a two-part historical analysis for {symbol}.

        PART 1 — Prediction track record:
        Use the backtesting tool to load the 5 most recent past predictions for {symbol}
        and compare each signal (BUY/SELL/NEUTRAL) against actual price changes at 24h
        and 7d after the prediction was made. Report:
        - Overall accuracy rates ("3/5 correct at 24h")
        - Systematic biases (consistently too bullish, wrong in downtrends, etc.)
        - Notable failures and the market condition that caused them
        - Calibration recommendation: adjust confidence up or down based on track record
        - If no history exists, clearly state this is the first run

        PART 2 — Historical analogue analysis:
        Use the historical pattern tool to scan ~3 years of daily price data for periods
        whose technical profile best matches today's market. The tool matches on 8 features:
        RSI level, 7-day momentum, 30-day momentum, price vs EMA20/50/200, and MACD histogram
        direction. For the analogues found, report:
        - How many historical matches were found and over what date range
        - The current profile (RSI, momentum, EMA structure, MACD direction)
        - Median and mean forward returns at 7, 14, and 30 days across all analogues
        - Percentage of analogues where price was higher at each horizon
        - The 5 most similar historical dates with their prices and actual outcomes
        - Whether the historical base rate agrees or conflicts with the current prediction
        - Any notable patterns (e.g. "most analogues occurred during bear-market bounces
          and the 30-day outcome was negative 70% of the time")

        Explicitly state whether the two parts reinforce each other or diverge,
        and give a final calibration recommendation for the Chief Strategist.
        """,
        expected_output=f"""
        A two-part historical analysis report for {symbol}:

        PART 1 — Prediction Track Record:
        - Predictions evaluated, accuracy at 24h and 7d
        - Systematic biases identified
        - Calibration recommendation

        PART 2 — Historical Analogues:
        - Current technical profile summary
        - Number of analogues found (sample size) and date range scanned
        - Forward return statistics: median/mean at 7d, 14d, 30d; % times price was higher
        - Top 5 most similar historical periods with dates, prices, and outcomes
        - Agreement or conflict between analogues and the current directional bias
        - Overall base-rate verdict: what history says about setups like this one
        """,
        agent=agent,
    )


# ── Macro ────────────────────────────────────────────────────────────────────

def create_macro_task(agent, symbol: str) -> Task:
    return Task(
        description=f"""
        Analyze the current state of traditional financial markets and determine
        what the macro environment implies for {symbol} and the broader crypto market.

        Use your tools to:
        1. Fetch S&P 500 + Nasdaq 100 trend, E-mini futures positioning, and VIX
        2. Fetch WTI crude oil, Brent crude oil, and the US Dollar Index
        3. Fetch 10-year Treasury yield, Gold, and CME Bitcoin futures vs Binance spot

        Assess and explain:
        - S&P 500 and Nasdaq 100: Are equities in a bull run, correction, or bear market?
          Nasdaq is more correlated to crypto — its direction matters more than S&P alone.
        - S&P 500 futures: premium/discount and session bias
        - VIX: current fear level — high VIX (>25) typically signals risk-off / crypto selloff
        - 10-year Treasury yield: rising yields = higher cost of capital = pressure on risk assets.
          What regime are we in (high/moderate/low yield)?
        - Gold: trending up (safe-haven/inflation concern) or down?
        - Oil: inflationary pressure or easing?
        - Dollar Index: strengthening (bearish for crypto) or weakening (bullish)?
        - CME Bitcoin futures: trading at premium or discount to Binance spot?
          Premium = institutional demand. Discount = institutional hedging.
        - Is crypto currently tracking macro closely or showing divergence?

        Conclude with a clear macro verdict:
        - Is the macro backdrop FAVORABLE, NEUTRAL, or UNFAVORABLE for {symbol}?
        - What is the single biggest macro risk or tailwind right now?
        {_VOTE_INSTRUCTION}
        """,
        expected_output=f"""
        A macro market analysis report for {symbol} containing:
        - S&P 500 + Nasdaq 100: trend, momentum, and outlook
        - S&P 500 futures: premium/discount and session bias
        - VIX: level, classification, crypto implication
        - 10-year Treasury yield: level, direction, yield regime, crypto implication
        - Gold: trend and safe-haven context
        - Oil (WTI + Brent): trend and inflation implication
        - Dollar Index: trend and crypto implication
        - CME BTC futures: premium/discount vs Binance spot and institutional signal
        - Macro verdict: FAVORABLE / NEUTRAL / UNFAVORABLE for {symbol}
        - Biggest macro risk and biggest macro tailwind

        AGENT VOTE:
        SIGNAL: [BUY | SELL | NEUTRAL] | Confidence: [HIGH | MEDIUM | LOW] | Reason: <one sentence>
        """,
        agent=agent,
    )


# ── On-Chain ─────────────────────────────────────────────────────────────────

def create_onchain_task(agent, symbol: str) -> Task:
    return Task(
        description=f"""
        Analyze Bitcoin network health across five on-chain layers and interpret
        what they collectively imply for the broader crypto market, including {symbol}.

        Use your tools to collect all five data layers:
        1. Bitcoin mempool stats (get_bitcoin_mempool_stats):
           - Pending transaction count, total mempool size, fee levels (fastest/30min/1h/economy)
           - Recent block utilisation (tx count, avg fee rate)

        2. Mining network health (get_mining_health):
           - Current hash rate (EH/s) and 30-day trend
           - Next difficulty adjustment: direction (%), ETA, and prior adjustment
           - Miner revenue split: block subsidy vs fee share

        3. Network activity / Blockchair stats (get_network_activity):
           - 24h transaction count and transfer volume
           - Coin Days Destroyed (CDD): how many dormant coins moved today?
           - Hodling addresses count (growing = more long-term holders = bullish)
           - Full node count, avg/median fee in USD

        4. Lightning Network stats (get_lightning_network_stats):
           - Channel count, total capacity (BTC), node count
           - Week-over-week changes in capacity and channel count

        5. CoinMetrics on-chain valuation (get_coinmetrics_onchain):
           - NVT ratio (30-day + 90-day smoothed): is the market overvalued vs on-chain usage?
           - Active addresses 30-day trend: is adoption growing or contracting?
           - Supply-cohort activity: what fraction of the 1-year-active supply moved in the last 30 days?

        Assess and explain each layer:

        LAYER 1 — Mempool demand:
        - Congestion level: >50k pending txs + fees >30 sat/vB = intense demand (bullish)
        - Cheap fees + thin mempool = bear market or low-interest regime
        - Are recent blocks full or half-empty?

        LAYER 2 — Mining conviction:
        - Rising hash rate = miners expanding capacity = long-term bullish conviction
        - Positive upcoming difficulty adjustment = network growing
        - High fee share (>20%) = fee-driven demand, not just subsidy-dependent
        - Falling hash rate + negative difficulty = miner capitulation (bearish)

        LAYER 3 — UTXO / holder behavior (Blockchair):
        - Coin Days Destroyed (CDD): HIGH CDD (>8M/day) = dormant long-term coins moving
          = possible distribution by long-term holders (bearish signal)
          LOW CDD (<2M/day) = coins are resting = accumulation regime (bullish)
        - Rising hodling addresses = growing holder base = adoption signal (bullish)
        - Transaction volume: high = active network; low = quiet/bear regime

        LAYER 4 — Lightning Network adoption:
        - Growing capacity during price decline = BTC locked, not sold = quiet accumulation
        - Shrinking capacity = channel closures, reducing exposure (bearish)

        LAYER 5 — On-chain valuation & supply cohorts (CoinMetrics):
        - NVT90 < 50 = undervalued vs on-chain throughput (bullish); >100 = overvalued (bearish)
        - Rising active addresses (30d trend > +5%) = growing adoption (bullish)
        - Supply-cohort: high 30d/1yr activity ratio = coins rotating to short-term holders
          = distribution pressure (bearish); low ratio = long-term holders holding (bullish)

        Synthesise all five layers:
        - Do they all agree? Convergence = higher conviction
        - Any notable divergences? (e.g. rising hash rate but high CDD = miners bullish, longs distributing)
        - What is the overall on-chain verdict for {symbol} and broader crypto?

        Note: all five tools are Bitcoin-specific but Bitcoin network health is a
        broad crypto market sentiment indicator.

        {_VOTE_INSTRUCTION}
        """,
        expected_output=f"""
        A five-layer on-chain network health report containing:

        LAYER 1 — Mempool:
        - Pending transaction count and congestion classification
        - Fee environment (sat/vB tiers) and demand signal
        - Recent block utilisation

        LAYER 2 — Mining health:
        - Hash rate (EH/s), 30-day trend direction
        - Upcoming difficulty adjustment (% change, ETA)
        - Miner revenue: fee share % and sustainability signal
        - Mining conviction signal: bullish / neutral / bearish

        LAYER 3 — Network activity & holder behavior (Blockchair):
        - 24h transactions and transfer volume (BTC + USD)
        - Coin Days Destroyed with interpretation (distribution vs accumulation)
        - Hodling addresses count and trend signal

        LAYER 4 — Lightning Network:
        - Channel count, capacity (BTC), node count
        - Week-over-week changes
        - Adoption signal: growing / stable / contracting

        LAYER 5 — On-chain valuation & supply cohorts (CoinMetrics):
        - NVT ratio (30d + 90d smoothed) with valuation signal
        - Active addresses count and 30-day trend
        - Supply-cohort breakdown (1yr / 180d / 30d active supply in BTC)
        - Cohort signal: accumulation / neutral / distribution

        SYNTHESIS:
        - Cross-layer convergence or divergence summary
        - Overall on-chain verdict and implication for {symbol}

        AGENT VOTE:
        SIGNAL: [BUY | SELL | NEUTRAL] | Confidence: [HIGH | MEDIUM | LOW] | Reason: <one sentence>
        """,
        agent=agent,
    )


# ── Market Data ───────────────────────────────────────────────────────────────

def create_market_data_task(agent, symbol: str) -> Task:
    return Task(
        description=f"""
        Collect comprehensive market data for {symbol} cryptocurrency.

        Use your tools to:
        1. Fetch 90 days of daily OHLCV data from Binance
        2. Get current market overview (market cap, dominance, 24h stats, stablecoin
           dominance, altcoin season indicator) from CoinGecko
        3. Analyze the current order book for buy/sell pressure and key price walls

        In your analysis, highlight:
        - Current price and key price change percentages (24h, 7d, 30d)
        - Volume trend (increasing or decreasing) and any unusual volume spikes
        - Any unusual order book imbalances (large bid/ask walls)
        - Market cap rank and dominance context
        - Stablecoin dominance: is capital sitting on the sidelines or deployed?
        - Altcoin season indicator: are we in BTC season or altcoin season?
        - Whether the market is in a trending or ranging phase based on volume

        {_VOTE_INSTRUCTION}
        """,
        expected_output=f"""
        A structured market data summary for {symbol} containing:
        - Current price, 24h/7d/30d performance
        - Volume analysis and trend
        - Order book analysis with key price walls
        - Market cap and dominance context
        - Stablecoin dominance level and signal
        - Altcoin season indicator
        - Overall market structure assessment (trending/ranging, bull/bear)

        AGENT VOTE:
        SIGNAL: [BUY | SELL | NEUTRAL] | Confidence: [HIGH | MEDIUM | LOW] | Reason: <one sentence>
        """,
        agent=agent,
    )


# ── Technical Analysis ────────────────────────────────────────────────────────

def create_technical_analysis_task(agent, symbol: str) -> Task:
    return Task(
        description=f"""
        Perform comprehensive technical analysis on {symbol} across three timeframes
        and compute the Volume Profile.

        Use your tools to:
        1. Calculate technical indicators on the WEEKLY (1w) timeframe — for macro trend context
        2. Calculate technical indicators on the DAILY (1d) timeframe — primary trend
        3. Calculate technical indicators on the 4-HOUR (4h) timeframe — short-term signals
        4. Compute the Volume Profile (VPOC, Value Area High/Low, High Volume Nodes)
           from 90 days of daily data
        5. Identify key support and resistance levels

        Analyze and interpret:
        - Weekly: What is the macro structure? Is the weekly trend bullish, bearish, or sideways?
          What do weekly RSI and MACD say about the longer-term momentum?
        - Daily RSI: overbought/oversold/divergences?
        - Daily MACD: histogram direction and strength?
        - Bollinger Bands: price near upper/lower band? Volatility expanding/contracting?
        - EMA alignment (20/50/200): golden cross, death cross, or mixed?
        - Volume indicators (OBV): is volume confirming the price move?
        - ADX: how strong is the current trend?
        - Stochastic RSI: short-term overbought/oversold?
        - Volume Profile:
          - Where is the VPOC? Is price above or below it (bullish/bearish structure)?
          - Is price inside the Value Area (fair value) or extended above/below?
          - Which High Volume Nodes are acting as support/resistance?
        - Support/Resistance: nearest key levels and risk/reward context

        Look for confluence across timeframes and between indicators and Volume Profile.

        {_VOTE_INSTRUCTION}
        """,
        expected_output=f"""
        A technical analysis report for {symbol} containing:
        - Weekly technical bias (macro structure and momentum)
        - Daily indicators summary: RSI, MACD, Bollinger Bands, EMAs, OBV, ADX, Stoch RSI
        - 4-hour indicators summary (short-term momentum and entry context)
        - Volume Profile: VPOC, Value Area High/Low, High Volume Nodes, price positioning
        - Key support and resistance levels with distances from current price
        - Multi-timeframe confluence analysis
        - Overall technical bias score and classification

        AGENT VOTE:
        SIGNAL: [BUY | SELL | NEUTRAL] | Confidence: [HIGH | MEDIUM | LOW] | Reason: <one sentence>
        """,
        agent=agent,
    )


# ── Sentiment ─────────────────────────────────────────────────────────────────

def create_sentiment_task(agent, symbol: str) -> Task:
    return Task(
        description=f"""
        Analyze market sentiment for {symbol} from multiple data sources.

        Use your tools to:
        1. Fetch the Fear & Greed Index (current value and 30-day trend)
        2. Fetch and analyze crypto news headlines from RSS feeds (CoinTelegraph, Decrypt,
           CoinDesk, Bitcoin Magazine) — filter for {symbol}-specific articles and run
           sentiment analysis on headlines
        3. Get Reddit community sentiment (hot posts from coin-specific and general crypto
           subreddits — look at VADER scores AND upvote ratios as community approval signals)
        4. Fetch on-chain and social metrics from CoinGecko

        Assess:
        - Is the overall market in fear, greed, or neutral territory?
        - What tone are crypto media outlets taking toward {symbol}?
          Are headlines positive, negative, or neutral? Any major stories dominating coverage?
        - What is the general crypto market mood from news (even non-{symbol} articles)?
        - What is the Reddit community's mood — are posts bullish/bearish, and are they
          receiving high or low upvote ratios?
        - Is developer activity increasing or declining?
        - Are there any contrarian signals (extreme sentiment opposite to recent price action)?
        - Identify any major news events or catalysts in recent headlines

        {_VOTE_INSTRUCTION}
        """,
        expected_output=f"""
        A sentiment analysis report for {symbol} containing:
        - Fear & Greed Index: current value, trend, and interpretation
        - News RSS sentiment: headline tone from 4 outlets, coin-specific vs general market mood
        - Reddit community sentiment: VADER scores, upvote ratios, top posts from relevant subs
        - On-chain/social metrics: developer activity, community size
        - Contrarian signal assessment (is sentiment extreme?)
        - Overall sentiment classification and strength

        AGENT VOTE:
        SIGNAL: [BUY | SELL | NEUTRAL] | Confidence: [HIGH | MEDIUM | LOW] | Reason: <one sentence>
        """,
        agent=agent,
    )


# ── Derivatives ───────────────────────────────────────────────────────────────

def create_derivatives_task(agent, symbol: str) -> Task:
    return Task(
        description=f"""
        Analyze the full derivatives landscape for {symbol}: perpetual futures AND options.

        Use your tools to:
        1. Fetch current and historical funding rates from Binance Futures
        2. Analyze open interest and long/short ratio data
        3. Get liquidation heatmap data from Coinglass (if API key is available)
        4. Fetch Deribit options data (BTC and ETH only) for put/call ratio,
           max pain price, implied volatility term structure, and top OI strikes

        For perpetual futures, investigate:
        - Funding rate direction and magnitude: Who is paying whom? Is leverage extreme?
        - Open interest trend: Is money flowing into or out of the futures market?
        - Long/short ratio: Is the market over-positioned on one side?
        - Liquidation risk: Which side (longs or shorts) is more vulnerable?

        For options (BTC/ETH only):
        - Put/Call ratio by OI: >1.2 = bearish hedging; <0.7 = call-heavy (bullish)
        - Max pain price: What strike minimises total option payout for nearest expiry?
          Is current spot above or below max pain? (Gravity pulls price toward max pain near expiry)
        - IV term structure: Is near-term IV elevated vs longer-dated? (backwardation = uncertainty)
        - Top OI strikes: Which price levels have the most open interest? (magnetic levels)

        {_VOTE_INSTRUCTION}
        """,
        expected_output=f"""
        A derivatives analysis report for {symbol} containing:
        - Funding rate: current value, interpretation, trend
        - Open interest trend and positioning analysis
        - Long/short ratio and liquidation risk assessment
        - Liquidation clusters at key price levels (if Coinglass available)
        - Options put/call ratio (OI + volume) and sentiment signal
        - Max pain price for nearest expiry and gravity direction
        - IV term structure and uncertainty reading
        - Top OI strikes acting as magnetic price levels
        - Overall derivatives bias (combining perps + options)

        AGENT VOTE:
        SIGNAL: [BUY | SELL | NEUTRAL] | Confidence: [HIGH | MEDIUM | LOW] | Reason: <one sentence>
        """,
        agent=agent,
    )


# ── Final Prediction ──────────────────────────────────────────────────────────

def create_prediction_task(agent, symbol: str, context_tasks: list) -> Task:
    return Task(
        description=f"""
        Synthesize all research to produce a comprehensive price prediction for {symbol}.

        You have received reports from:
        1. Prediction Accuracy Analyst — historical accuracy calibration
        2. Macro Market Analyst — S&P 500, Nasdaq, VIX, yields, gold, oil, dollar, CME futures
        3. On-Chain Network Analyst — Bitcoin mempool and network activity
        4. Market Data Analyst — price history, volume, order book, stablecoin dominance
        5. Technical Analysis Specialist — weekly/daily/4h indicators, Volume Profile, S/R levels
        6. Sentiment & On-Chain Analyst — Fear & Greed, news RSS, Reddit, social/dev metrics
        7. Derivatives & Options Specialist — funding, OI, liquidations, put/call, max pain, IV

        **STEP 1 — AGENT VOTE TABLE (required, do this first):**
        Create a table listing every agent's vote:

        | Agent | Signal | Confidence | Key Reason |
        |---|---|---|---|
        | Macro | ... | ... | ... |
        | On-Chain | ... | ... | ... |
        | Market Data | ... | ... | ... |
        | Technical | ... | ... | ... |
        | Sentiment | ... | ... | ... |
        | Derivatives | ... | ... | ... |

        Count the votes: X BUY, Y SELL, Z NEUTRAL.
        Note any significant disagreements between agents.

        **STEP 2 — MACRO CONTEXT:**
        How does the equity market / VIX / yields / dollar / oil backdrop
        support or threaten this prediction? Is crypto tracking macro or diverging?

        **STEP 3 — SHORT-TERM (24-48 hours):**
        - Price direction bias (bullish/bearish/neutral) with confidence %
        - Target price range
        - Key catalyst or trigger to watch
        - Invalidation level

        **STEP 4 — MEDIUM-TERM (7 days):**
        - Expected range with bull/base/bear scenarios and probabilities
        - Key factors that would shift the outlook

        **STEP 5 — LONGER-TERM (30 days):**
        - Macro trend assessment and key levels to watch

        **STEP 6 — RISK FACTORS:**
        - Top 3-5 risks to the prediction
        - Any conflicting signals between agents

        **STEP 7 — OVERALL SIGNAL:**
        - Single classification: STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL
        - Confidence: LOW / MEDIUM / HIGH
          (adjust downward if backtesting shows poor recent accuracy or agents strongly disagree)
        - Key conviction driver (the single most important signal)

        Be honest about uncertainty. Use ranges, not point predictions.
        Note any significant data gaps that affected the analysis.

        **FINAL LINE — machine-readable signal (required, must be last):**
        After your full narrative, output this exact line as the very last line of your response,
        replacing the placeholders with your chosen values:
        PREDICTION_SIGNAL_JSON:{{"signal":"<STRONG_BUY|BUY|NEUTRAL|SELL|STRONG_SELL>","confidence":"<HIGH|MEDIUM|LOW>"}}
        """,
        expected_output=f"""
        A comprehensive crypto prediction report for {symbol}:
        - Agent vote table (all 6 research agents)
        - Vote tally and disagreement summary
        - Macro context section
        - Short-term (24-48h): direction, targets, invalidation
        - Medium-term (7d): bull/base/bear scenarios with probabilities
        - Long-term (30d): macro trend assessment
        - Risk factors and conflicting signals
        - Overall signal: STRONG_BUY/BUY/NEUTRAL/SELL/STRONG_SELL with confidence
        - Data quality note
        - Last line: PREDICTION_SIGNAL_JSON:{{"signal":"...","confidence":"..."}}
        """,
        agent=agent,
        context=context_tasks,
    )


# ── Trading Strategy ─────────────────────────────────────────────────────────

def create_trading_strategy_task(agent, symbol: str, prediction_task) -> Task:
    return Task(
        description=f"""
        Convert the Chief Strategist's prediction for {symbol} into a concrete, risk-managed
        trade plan for the paper portfolio, then record it as a structured sentinel.

        STEP 1 — Read portfolio state:
        Call get_portfolio_state with symbol="{symbol}". Note:
        - Cash available (available capital for a new trade)
        - Any currently open position (direction, entry, SL, TP, size)
        - Last 3 trades with outcomes (win/loss context)
        - Total P&L and win rate since inception

        STEP 2 — Review the prediction:
        From the Chief Strategist's report, extract:
        - Overall signal: STRONG_BUY / BUY / NEUTRAL / SELL / STRONG_SELL
        - Confidence: HIGH / MEDIUM / LOW
        - Short-term price target (24-48h) → use as Take Profit 1
        - Medium-term price target (7d)   → use as Take Profit 2
        - Invalidation level              → use as Stop Loss
        - Current price (mentioned in the prediction header or market data section)

        STEP 3 — Decide the trade:

        Signal mapping:
        - STRONG_BUY or BUY  → direction = LONG
        - STRONG_SELL or SELL → direction = SHORT (simulated, no leverage)
        - NEUTRAL            → direction = NO_TRADE (close any open position if exists)
        - LOW confidence     → direction = NO_TRADE regardless of signal

        If there is already an open position in the SAME direction as the new signal:
        mention it but do NOT open a second position (one trade at a time).
        If it is in the OPPOSITE direction: note it will be closed at next run evaluation.

        STEP 4 — Define levels (for LONG or SHORT):
        - Entry price: current market price (assume market order)
        - Stop Loss: the prediction's invalidation level (key support for LONG,
          key resistance for SHORT). Must be < entry for LONG, > entry for SHORT.
        - Take Profit 1: conservative 24-48h target (closer to entry, ~1:1.5 R/R minimum)
        - Take Profit 2: aggressive 7-day target (further, higher R/R)
        - Calculate Risk/Reward ratio = (TP1 - entry) / (entry - SL) for LONG
          (or (entry - TP1) / (SL - entry) for SHORT)

        STEP 5 — Position sizing explanation:
        Position sizing is handled automatically by the system using:
          risk_amount = cash * risk_pct  (HIGH=3%, MEDIUM=1.5%)
          size_usd = risk_amount / sl_distance_pct  (capped at 30% of portfolio)
        State the estimated size in USD and coins based on this formula.
        State the dollar risk if stop-loss is hit.

        STEP 6 — Narrative:
        Write a clear 3-5 sentence rationale explaining:
        - Why this direction aligns with the prediction
        - What the key conviction driver is
        - What would invalidate the trade

        STEP 7 — Portfolio context:
        Summarise current portfolio status: total value, P&L since start, win rate.
        If this is the first trade, note the starting balance of $100,000.

        **FINAL LINE — machine-readable trade plan (required, must be the very last line):**
        TRADE_PLAN_JSON:{{"direction":"<LONG|SHORT|NO_TRADE>","entry_price":<float>,"stop_loss":<float>,"take_profit_1":<float>,"take_profit_2":<float or null>,"confidence":"<HIGH|MEDIUM|LOW>","signal":"<BUY|SELL|NEUTRAL|STRONG_BUY|STRONG_SELL>"}}
        """,
        expected_output=f"""
        A structured trade plan report for {symbol} containing:
        - Portfolio state summary (value, cash, any open position, recent trades)
        - Prediction signal and confidence extracted from the Chief Strategist
        - Trade decision: LONG / SHORT / NO_TRADE with justification
        - Entry price, Stop Loss, Take Profit 1, Take Profit 2
        - Risk/Reward ratio
        - Estimated position size in USD and coins
        - Dollar risk if SL is hit
        - Trade rationale (3-5 sentences)
        - Portfolio cumulative P&L and win rate summary
        - Last line: TRADE_PLAN_JSON:{{...}}
        """,
        agent=agent,
        context=[prediction_task],
    )


# ── Report Formatter ──────────────────────────────────────────────────────────

def create_report_formatter_task(agent, symbol: str, prediction_task, strategy_task=None) -> Task:
    context_tasks = [prediction_task]
    if strategy_task is not None:
        context_tasks.append(strategy_task)

    return Task(
        description=f"""
        Take the final prediction report and trade plan for {symbol} and generate
        a polished HTML report saved to the reports/ folder.

        You MUST call the html_report_generator tool with exactly these three arguments:
          - symbol: "{symbol}"
          - prediction_text: the full text of the prediction report from the Chief Strategist
          - strategy_text: the full text of the trade plan from the Trading Strategist

        Do not summarise, shorten, or modify either text — pass both verbatim.
        After the tool completes, return the file path it gave you.
        """,
        expected_output=f"""
        The absolute file path to the saved HTML report for {symbol},
        e.g. /home/user/project/reports/{symbol}_20250101_120000.html
        """,
        agent=agent,
        context=context_tasks,
    )
