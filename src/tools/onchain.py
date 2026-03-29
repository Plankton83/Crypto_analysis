"""
On-chain network tools:
- MempoolTool         — Bitcoin mempool stats (mempool.space — free, no key)
- MiningHealthTool    — Hash rate, difficulty adjustment, miner revenue (mempool.space — free, no key)
- NetworkActivityTool — Tx count, CDD, hodling addresses, node count (Blockchair — free, no key)
- LightningNetworkTool — LN channel count, capacity, node count + weekly trend (mempool.space — free, no key)
- CoinMetricsTool     — NVT ratio, active addresses, supply-cohort activity (CoinMetrics Community — free, no key)
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class OnchainInput(BaseModel):
    symbol: str = Field(
        default="BTC",
        description="Crypto symbol. All on-chain tools are Bitcoin-specific but relevant as broad crypto market signals.",
    )

# Keep old name as alias so existing imports don't break
MempoolInput = OnchainInput


class MempoolTool(BaseTool):
    name: str = "get_bitcoin_mempool_stats"
    description: str = (
        "Fetch Bitcoin mempool statistics from mempool.space: pending transaction count, "
        "total mempool size, and recommended fee levels (sat/vB). "
        "High congestion + high fees = intense on-chain activity, historically precedes or "
        "accompanies price moves. A clearing mempool after a spike can signal exhaustion. "
        "Even for non-BTC analysis, Bitcoin network health is a broad crypto market signal. "
        "No API key required."
    )
    args_schema: type[BaseModel] = MempoolInput

    def _run(self, symbol: str = "BTC") -> str:
        result: dict = {"note": "Bitcoin mempool data — relevant as a crypto-wide network health indicator."}

        # Mempool overview
        try:
            resp = requests.get("https://mempool.space/api/mempool", timeout=10)
            resp.raise_for_status()
            m = resp.json()
            result["pending_transactions"] = m.get("count", 0)
            result["mempool_size_vbytes"] = m.get("vsize", 0)
            result["total_pending_fees_sat"] = m.get("total_fee", 0)
        except Exception as exc:
            result["mempool_error"] = str(exc)

        # Recommended fees
        try:
            resp2 = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=10)
            resp2.raise_for_status()
            fees = resp2.json()
            result["fees_sat_per_vbyte"] = {
                "fastest_1_block": fees.get("fastestFee"),
                "30_min": fees.get("halfHourFee"),
                "1_hour": fees.get("hourFee"),
                "economy": fees.get("economyFee"),
                "minimum": fees.get("minimumFee"),
            }
        except Exception as exc:
            result["fees_error"] = str(exc)

        # Block stats for recent activity
        try:
            resp3 = requests.get("https://mempool.space/api/v1/blocks", timeout=10)
            if resp3.ok:
                blocks = resp3.json()[:5]
                result["recent_blocks"] = [
                    {
                        "height": b.get("height"),
                        "tx_count": b.get("tx_count"),
                        "size_kb": round(b.get("size", 0) / 1024, 1),
                        "avg_fee_rate": b.get("extras", {}).get("avgFeeRate"),
                    }
                    for b in blocks
                ]
        except Exception:
            pass

        # Interpret signals
        pending = result.get("pending_transactions", 0)
        fast_fee = (result.get("fees_sat_per_vbyte") or {}).get("fastest_1_block") or 0

        result["congestion"] = (
            "extreme (>100k pending txs)" if pending > 100_000
            else "high (50k-100k pending txs)" if pending > 50_000
            else "moderate (20k-50k pending txs)" if pending > 20_000
            else "low (<20k pending txs)"
        )
        result["fee_environment"] = (
            "very expensive (>100 sat/vB) - high demand" if fast_fee > 100
            else "expensive (50-100 sat/vB)" if fast_fee > 50
            else "moderate (10-50 sat/vB)" if fast_fee > 10
            else "cheap (<10 sat/vB) - low demand"
        )
        result["network_activity_signal"] = (
            "very high on-chain activity - strong bullish pressure" if pending > 80_000 and fast_fee > 40
            else "elevated activity - supportive of bullish move" if pending > 30_000
            else "normal activity - neutral"
            if pending > 10_000
            else "low activity - weak demand, typically bearish or bear market regime"
        )

        return json.dumps(result, indent=2)


# ── Mining Health ─────────────────────────────────────────────────────────────

class MiningHealthTool(BaseTool):
    name: str = "get_mining_health"
    description: str = (
        "Fetch Bitcoin mining network health from mempool.space: current hash rate, "
        "3-month hash rate trend, next difficulty adjustment (% change and ETA), "
        "and miner revenue breakdown (block subsidy vs fees). "
        "Rising hash rate during a price decline = miner conviction = bullish long-term signal. "
        "Positive upcoming difficulty adjustment = network is growing. "
        "No API key required."
    )
    args_schema: Type[BaseModel] = OnchainInput

    def _run(self, symbol: str = "BTC") -> str:
        result: dict = {}

        # Current hash rate + 3-month trend
        try:
            resp = requests.get("https://mempool.space/api/v1/mining/hashrate/3m", timeout=10)
            resp.raise_for_status()
            data = resp.json()

            current_hr = data.get("currentHashrate", 0)
            hashrates  = data.get("hashrates", [])

            # EH/s (exahashes per second) = raw / 1e18
            current_ehs = round(current_hr / 1e18, 2)

            # 30-day trend: compare latest vs 30 entries ago
            if len(hashrates) >= 30:
                old_hr = hashrates[-30].get("avgHashrate", current_hr)
                hr_change_30d = round((current_hr - old_hr) / old_hr * 100, 2) if old_hr else 0
            else:
                hr_change_30d = None

            result["hash_rate_ehs"]         = current_ehs
            result["hash_rate_change_30d_pct"] = hr_change_30d
            result["hash_rate_trend"]       = (
                "rising" if (hr_change_30d or 0) > 2
                else "falling" if (hr_change_30d or 0) < -2
                else "flat"
            )
        except Exception as exc:
            result["hash_rate_error"] = str(exc)

        # Next difficulty adjustment
        try:
            resp2 = requests.get("https://mempool.space/api/v1/difficulty-adjustment", timeout=10)
            resp2.raise_for_status()
            da = resp2.json()

            eta_ms   = da.get("estimatedRetargetDate", 0)
            eta_dt   = datetime.fromtimestamp(eta_ms / 1000, tz=timezone.utc)
            days_away = (eta_dt - datetime.now(tz=timezone.utc)).days

            result["difficulty_adjustment"] = {
                "change_pct":          round(da.get("difficultyChange", 0), 2),
                "direction":           "increase" if da.get("difficultyChange", 0) > 0 else "decrease",
                "remaining_blocks":    da.get("remainingBlocks"),
                "eta_days":            days_away,
                "eta_date":            eta_dt.strftime("%Y-%m-%d"),
                "previous_adjustment_pct": round(da.get("previousRetarget", 0), 2),
                "current_block_time_s":    round(da.get("timeAvg", 0) / 1000, 1),
                "target_block_time_s": 600,
                "progress_pct":        round(da.get("progressPercent", 0), 1),
            }
            result["difficulty_signal"] = (
                "network growing — bullish long-term" if da.get("difficultyChange", 0) > 3
                else "network shrinking — miners capitulating" if da.get("difficultyChange", 0) < -3
                else "network stable"
            )
        except Exception as exc:
            result["difficulty_error"] = str(exc)

        # Miner revenue: latest block subsidy vs fees
        try:
            resp3 = requests.get("https://mempool.space/api/v1/blocks", timeout=10)
            if resp3.ok:
                blocks = resp3.json()[:6]
                fee_revenues   = [b.get("extras", {}).get("totalFees", 0) for b in blocks]
                reward_per_block = 3_125_000_000  # 3.125 BTC in sats (post-halving)
                avg_fees_sat   = sum(fee_revenues) / len(fee_revenues) if fee_revenues else 0
                fee_share_pct  = round(avg_fees_sat / (reward_per_block + avg_fees_sat) * 100, 2)
                result["miner_revenue"] = {
                    "avg_fees_per_block_sat": round(avg_fees_sat),
                    "avg_fees_per_block_btc": round(avg_fees_sat / 1e8, 4),
                    "fee_share_of_reward_pct": fee_share_pct,
                    "subsidy_btc": 3.125,
                    "signal": (
                        "fee-driven revenue (high demand, sustainable)" if fee_share_pct > 20
                        else "subsidy-dependent (typical, low fee demand)"
                    ),
                }
        except Exception:
            pass

        # Composite signal
        hr_trend   = result.get("hash_rate_trend", "flat")
        diff_pct   = result.get("difficulty_adjustment", {}).get("change_pct", 0)
        result["mining_conviction_signal"] = (
            "strong bullish: rising hash rate + increasing difficulty — miners expanding capacity"
            if hr_trend == "rising" and diff_pct > 0
            else "bullish: hash rate holding despite difficulty increase"
            if hr_trend == "flat" and diff_pct > 3
            else "bearish: hash rate falling + difficulty decreasing — miner capitulation"
            if hr_trend == "falling" and diff_pct < -3
            else "neutral: mixed mining signals"
        )

        return json.dumps(result, indent=2)


# ── Network Activity (Blockchair) ─────────────────────────────────────────────

class NetworkActivityTool(BaseTool):
    name: str = "get_network_activity"
    description: str = (
        "Fetch Bitcoin network activity from Blockchair (no API key required): "
        "24h transaction count, transfer volume, Coin Days Destroyed (CDD), "
        "mempool transactions per second, total hodling addresses, and full node count. "
        "CDD is a UTXO-flavoured signal: high CDD = dormant long-term coins moving = "
        "potential distribution (bearish). Low CDD = coins resting = accumulation regime. "
        "Rising hodling addresses = growing holder base = bullish adoption signal."
    )
    args_schema: Type[BaseModel] = OnchainInput

    def _run(self, symbol: str = "BTC") -> str:
        try:
            resp = requests.get("https://api.blockchair.com/bitcoin/stats", timeout=15)
            resp.raise_for_status()
            d = resp.json().get("data", {})
        except Exception as exc:
            return json.dumps({"error": f"Blockchair fetch failed: {exc}"})

        # CDD interpretation thresholds (approximate historical context)
        cdd = float(d.get("cdd_24h", 0))
        cdd_signal = (
            "very high — significant dormant coin movement, possible long-term holder distribution (bearish)"
            if cdd > 20_000_000
            else "elevated — some long-term holder activity, watch for distribution"
            if cdd > 8_000_000
            else "normal — typical daily coin movement, no strong signal"
            if cdd > 2_000_000
            else "low — coins are resting, accumulation / dormant regime (bullish)"
        )

        txs_24h    = int(d.get("transactions_24h", 0))
        tx_signal  = (
            "high on-chain activity" if txs_24h > 500_000
            else "moderate activity" if txs_24h > 300_000
            else "low activity — quiet network"
        )

        volume_sat  = int(d.get("volume_24h", 0))
        volume_btc  = round(volume_sat / 1e8, 2)
        # Convert to USD using market price from the same endpoint
        price_usd   = float(d.get("market_price_usd", 0))
        volume_usd  = round(volume_btc * price_usd) if price_usd else None

        result = {
            "source": "Blockchair (raw on-chain, not entity-adjusted)",
            "transactions_24h":       txs_24h,
            "transaction_signal":     tx_signal,
            "transfer_volume_btc":    volume_btc,
            "transfer_volume_usd":    volume_usd,
            "coin_days_destroyed_24h": round(cdd, 0),
            "cdd_signal":             cdd_signal,
            "mempool_tps":            round(float(d.get("mempool_tps", 0)), 2),
            "mempool_transactions":   int(d.get("mempool_transactions", 0)),
            "mempool_total_fee_usd":  round(float(d.get("mempool_total_fee_usd", 0)), 2),
            "hodling_addresses":      int(d.get("hodling_addresses", 0)),
            "full_node_count":        int(d.get("nodes", 0)),
            "avg_fee_usd_24h":        round(float(d.get("average_transaction_fee_usd_24h", 0)), 4),
            "median_fee_usd_24h":     round(float(d.get("median_transaction_fee_usd_24h", 0)), 4),
            "blocks_mined_24h":       int(d.get("blocks_24h", 0)),
            "next_difficulty_estimate": d.get("next_difficulty_estimate"),
            "next_retarget_estimate": d.get("next_retarget_time_estimate"),
            "note": (
                "CDD and transaction volume are raw/unadjusted — they include exchange "
                "internal moves and consolidations. Use as directional signal, not precision metric."
            ),
        }
        return json.dumps(result, indent=2)


# ── Lightning Network ─────────────────────────────────────────────────────────

class LightningNetworkTool(BaseTool):
    name: str = "get_lightning_network_stats"
    description: str = (
        "Fetch Bitcoin Lightning Network statistics from mempool.space: current channel count, "
        "total capacity (BTC), node count, and week-over-week changes. "
        "Growing LN capacity during a price decline = BTC being locked into payment channels "
        "rather than sold = quiet accumulation / adoption signal (bullish). "
        "Shrinking capacity = participants closing channels, reducing exposure (bearish). "
        "No API key required."
    )
    args_schema: Type[BaseModel] = OnchainInput

    def _run(self, symbol: str = "BTC") -> str:
        result: dict = {}

        # Use the 1-month history endpoint which returns an array of weekly snapshots.
        # The /latest endpoint returns a flat object (no nested latest/previous keys).
        try:
            resp = requests.get(
                "https://mempool.space/api/v1/lightning/statistics/1m", timeout=10
            )
            resp.raise_for_status()
            rows = resp.json()

            if not rows or not isinstance(rows, list):
                raise ValueError("Unexpected response format from LN statistics endpoint")

            latest = rows[-1]
            prev   = rows[-2] if len(rows) >= 2 else {}

            cap_sat      = latest.get("total_capacity") or 0
            cap_btc      = round(cap_sat / 1e8, 2)
            prev_cap_sat = prev.get("total_capacity") or 0
            cap_change   = round((cap_sat - prev_cap_sat) / prev_cap_sat * 100, 2) if prev_cap_sat else None

            ch_now    = latest.get("channel_count") or 0
            ch_prev   = prev.get("channel_count") or 0
            ch_change = ch_now - ch_prev

            node_now    = latest.get("node_count") or 0
            node_prev   = prev.get("node_count") or 0
            node_change = node_now - node_prev

            result["channel_count"]               = ch_now
            result["channel_count_change_7d"]     = ch_change
            result["total_capacity_btc"]          = cap_btc
            result["total_capacity_change_7d_pct"] = cap_change
            result["node_count"]                  = node_now
            result["node_count_change_7d"]        = node_change
            result["avg_channel_capacity_btc"]    = round(
                (latest.get("avg_capacity") or 0) / 1e8, 6
            )
            result["median_channel_capacity_btc"] = round(
                (latest.get("med_capacity") or 0) / 1e8, 6
            )
            result["avg_fee_rate_ppm"] = latest.get("avg_fee_rate")
            result["last_updated"]     = (latest.get("added") or "")[:10]

        except Exception as exc:
            result["lightning_error"] = str(exc)
            return json.dumps(result, indent=2)

        # Trend signal
        cap_chg = result.get("total_capacity_change_7d_pct") or 0
        ch_chg  = result.get("channel_count_change_7d") or 0

        result["adoption_signal"] = (
            "strong growth — LN expanding rapidly, increasing BTC locked in payment layer (bullish)"
            if cap_chg > 2 and ch_chg > 0
            else "steady growth — gradual LN adoption (mildly bullish)"
            if cap_chg > 0 and ch_chg >= 0
            else "contraction — channels closing, capacity leaving the network (bearish)"
            if cap_chg < -2
            else "flat — LN capacity stable, no strong adoption signal"
        )

        return json.dumps(result, indent=2)


# ── CoinMetrics Community (NVT + active addresses + supply cohorts) ───────────

class CoinMetricsTool(BaseTool):
    name: str = "get_coinmetrics_onchain"
    description: str = (
        "Fetch Bitcoin on-chain valuation and holder-behavior metrics from the CoinMetrics "
        "Community API (free, no key required): NVT ratio (30-day + 90-day smoothed), "
        "active addresses trend, and supply-cohort activity (% of supply moved in 30d / 180d / 1yr). "
        "NVT (Network Value to Transactions) is the crypto equivalent of a P/E ratio: "
        "high NVT (>100) = network is overvalued relative to on-chain throughput (bearish); "
        "low NVT (<50) = undervalued relative to usage (bullish). "
        "Rising active addresses = growing adoption (bullish). "
        "Rising short-term supply activity vs dormant supply = coins moving from long-term "
        "holders to short-term speculators = possible distribution top (bearish)."
    )
    args_schema: Type[BaseModel] = OnchainInput

    _ASSET_MAP: dict = {
        "BTC": "btc", "ETH": "eth", "SOL": "sol", "BNB": "bnb",
        "ADA": "ada", "AVAX": "avax", "DOT": "dot", "MATIC": "matic",
        "LINK": "link", "XRP": "xrp", "LTC": "ltc", "DOGE": "doge",
    }

    # Metrics in priority order: try full set first, fall back to basics if 400
    _FULL_METRICS  = "AdrActCnt,TxCnt,NVTAdj,NVTAdj90,SplyAct1yr,SplyAct180d,SplyAct30d"
    _BASIC_METRICS = "AdrActCnt,TxCnt,FeeMeanUSD,FeeMedUSD"

    def _fetch_cm(self, asset: str, metrics: str) -> list[dict]:
        start = (datetime.utcnow() - timedelta(days=32)).strftime("%Y-%m-%dT00:00:00Z")
        resp = requests.get(
            "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
            params={
                "assets": asset,
                "metrics": metrics,
                "frequency": "1d",
                "start_time": start,
                "page_size": 35,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("data", [])

    def _run(self, symbol: str = "BTC") -> str:
        asset = self._ASSET_MAP.get(symbol.upper(), symbol.lower())

        rows: list[dict] = []
        metrics_used = self._FULL_METRICS
        try:
            rows = self._fetch_cm(asset, self._FULL_METRICS)
        except Exception as exc:
            # 400 often means one or more metrics are premium — retry with basics
            if "400" in str(exc) or "Client Error" in str(exc):
                try:
                    rows = self._fetch_cm(asset, self._BASIC_METRICS)
                    metrics_used = self._BASIC_METRICS
                except Exception as exc2:
                    return json.dumps({"error": f"CoinMetrics fetch failed: {exc2}"})
            else:
                return json.dumps({"error": f"CoinMetrics fetch failed: {exc}"})

        if not rows:
            return json.dumps({"error": "No data returned from CoinMetrics"})

        def _float(val):
            try:
                return float(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        latest = rows[-1]
        oldest = rows[0]

        nvt_now   = _float(latest.get("NVTAdj"))
        nvt_90    = _float(latest.get("NVTAdj90"))
        nvt_old   = _float(oldest.get("NVTAdj"))
        adr_now   = _float(latest.get("AdrActCnt"))
        adr_old   = _float(oldest.get("AdrActCnt"))
        tx_now    = _float(latest.get("TxCnt"))
        sply_1yr  = _float(latest.get("SplyAct1yr"))
        sply_180d = _float(latest.get("SplyAct180d"))
        sply_30d  = _float(latest.get("SplyAct30d"))

        nvt_change_30d = round((nvt_now - nvt_old) / nvt_old * 100, 1) if nvt_now and nvt_old else None
        adr_change_30d = round((adr_now - adr_old) / adr_old * 100, 1) if adr_now and adr_old else None

        if nvt_90 is not None:
            nvt_signal = (
                "severely overvalued (NVT90 > 150) - strong bearish on-chain signal" if nvt_90 > 150
                else "overvalued (NVT90 100-150) - bearish, throughput lags market cap" if nvt_90 > 100
                else "fair value (NVT90 50-100) - neutral, market cap aligned with usage" if nvt_90 > 50
                else "undervalued (NVT90 < 50) - bullish, high on-chain usage vs market cap"
            )
        else:
            nvt_signal = "unavailable"

        if adr_change_30d is not None:
            adr_signal = (
                "strong growth (>10%) - expanding user base, bullish adoption" if adr_change_30d > 10
                else "moderate growth (3-10%) - gradual adoption, mildly bullish" if adr_change_30d > 3
                else "declining (< -3%) - shrinking active user base, bearish" if adr_change_30d < -3
                else "stable - no strong adoption signal"
            )
        else:
            adr_signal = "unavailable"

        cohort_signal = "unavailable"
        if sply_1yr and sply_30d and sply_1yr > 0:
            short_vs_long = round(sply_30d / sply_1yr * 100, 1)
            cohort_signal = (
                f"high short-term activity ({short_vs_long}% of 1yr-active supply moved in 30d) "
                "- coins rotating to short-term holders, possible distribution (bearish)"
                if short_vs_long > 40
                else f"moderate short-term activity ({short_vs_long}%) - neutral"
                if short_vs_long > 20
                else f"low short-term activity ({short_vs_long}% of 1yr-active supply in 30d) "
                "- supply resting with long-term holders (bullish accumulation)"
            )

        result = {
            "source": "CoinMetrics Community API (free, no key)",
            "metrics_fetched": metrics_used,
            "asset": asset.upper(),
            "date": latest.get("time", "")[:10],
        }

        if nvt_now is not None:
            result["nvt_ratio_30d"] = round(nvt_now, 1)
        if nvt_90 is not None:
            result["nvt_ratio_90d_smoothed"] = round(nvt_90, 1)
        if nvt_change_30d is not None:
            result["nvt_change_30d_pct"] = nvt_change_30d
        result["nvt_signal"] = nvt_signal

        if adr_now is not None:
            result["active_addresses"] = int(adr_now)
        if adr_change_30d is not None:
            result["active_addresses_change_30d_pct"] = adr_change_30d
        result["active_address_signal"] = adr_signal

        if tx_now is not None:
            result["transactions_24h"] = int(tx_now)

        # Supply cohorts (only present when full metrics are available)
        if sply_1yr is not None:
            result["supply_active_1yr_btc"]  = round(sply_1yr, 2)
        if sply_180d is not None:
            result["supply_active_180d_btc"] = round(sply_180d, 2)
        if sply_30d is not None:
            result["supply_active_30d_btc"]  = round(sply_30d, 2)
        result["supply_cohort_signal"] = cohort_signal

        # Fee data when only basic metrics available
        fee_mean = _float(latest.get("FeeMeanUSD"))
        fee_med  = _float(latest.get("FeeMedUSD"))
        if fee_mean is not None:
            result["avg_fee_usd"] = round(fee_mean, 4)
        if fee_med is not None:
            result["median_fee_usd"] = round(fee_med, 4)

        result["note"] = (
            "NVT = market cap / on-chain tx volume (like a P/E ratio). "
            "Supply cohorts: high 30d/1yr ratio = short-term holders dominating "
            "(distribution risk); low = long-term holders holding (accumulation). "
            "If supply cohort fields are absent, the free tier did not return them."
        )
        return json.dumps(result, indent=2)
