"""
Sentiment analysis tools:
- Fear & Greed Index (alternative.me — free, no key)
- Stocktwits sentiment (free, no key — users tag messages Bullish/Bearish)
- Reddit community sentiment (public JSON API — no key)
- On-chain & social metrics (CoinGecko — no key)
"""

import json
import os
from datetime import datetime, timedelta
from typing import Optional, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_vader = SentimentIntensityAnalyzer()


def _classify_compound(score: float) -> str:
    if score >= 0.5: return "very_positive"
    if score >= 0.1: return "positive"
    if score <= -0.5: return "very_negative"
    if score <= -0.1: return "negative"
    return "neutral"


# ── Fear & Greed Index ────────────────────────────────────────────────────────

class FearGreedInput(BaseModel):
    limit: int = Field(default=30, description="Number of days of history to fetch")


class FearGreedTool(BaseTool):
    name: str = "get_fear_greed_index"
    description: str = (
        "Fetch the Crypto Fear & Greed Index from alternative.me. "
        "Returns current value (0=extreme fear, 100=extreme greed) and historical trend. "
        "No API key required."
    )
    args_schema: Type[BaseModel] = FearGreedInput

    def _run(self, limit: int = 30) -> str:
        resp = requests.get(
            f"https://api.alternative.me/fng/?limit={limit}&format=json",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()["data"]

        current = data[0]
        values = [int(d["value"]) for d in data]
        avg_7d = round(sum(values[:7]) / 7, 1)
        avg_30d = round(sum(values) / len(values), 1)

        # Detect trend
        recent_avg = sum(values[:7]) / 7
        older_avg = sum(values[7:14]) / 7 if len(values) >= 14 else recent_avg
        trend = "improving" if recent_avg > older_avg else "deteriorating"

        result = {
            "current_value": int(current["value"]),
            "current_classification": current["value_classification"],
            "updated": datetime.fromtimestamp(int(current["timestamp"])).strftime("%Y-%m-%d"),
            "7d_average": avg_7d,
            "30d_average": avg_30d,
            "sentiment_trend": trend,
            "history_30d": [{"date": datetime.fromtimestamp(int(d["timestamp"])).strftime("%Y-%m-%d"),
                              "value": int(d["value"]),
                              "classification": d["value_classification"]} for d in data[:30]],
        }
        return json.dumps(result)


# ── Crypto News RSS Sentiment ─────────────────────────────────────────────────

import xml.etree.ElementTree as _ET

# Free RSS feeds from major crypto news outlets — no API key required
_RSS_FEEDS = [
    ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ("Decrypt",       "https://decrypt.co/feed"),
    ("CoinDesk",      "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("Bitcoin Mag",   "https://bitcoinmagazine.com/feed"),
]

_COIN_KEYWORDS: dict[str, list[str]] = {
    "BTC":  ["bitcoin", "btc"],
    "ETH":  ["ethereum", "eth"],
    "SOL":  ["solana", "sol"],
    "BNB":  ["bnb", "binance coin"],
    "XRP":  ["xrp", "ripple"],
    "ADA":  ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
    "AVAX": ["avalanche", "avax"],
    "DOT":  ["polkadot", "dot"],
    "LINK": ["chainlink", "link"],
    "LTC":  ["litecoin", "ltc"],
}

_RSS_HEADERS = {"User-Agent": "Mozilla/5.0 (crypto-predictor sentiment bot)"}


def _sanitize(text: str) -> str:
    """Replace typographic punctuation with ASCII equivalents."""
    return (
        text.replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2013", "-").replace("\u2014", "-")
            .encode("ascii", "replace").decode("ascii")
    )


def _parse_rss_items(xml_bytes: bytes) -> list[dict]:
    """Extract title + description from an RSS or Atom feed."""
    items = []
    try:
        root = _ET.fromstring(xml_bytes)

        # RSS 2.0 format
        for item in root.iter("item"):
            title = _sanitize((item.findtext("title") or "").strip())
            desc  = _sanitize((item.findtext("description") or "").strip())
            pub   = (item.findtext("pubDate") or "")[:16]
            if title:
                items.append({"title": title, "description": desc[:200], "published": pub})

        # Atom format (fallback)
        if not items:
            for entry in root.iter("{http://www.w3.org/2005/Atom}entry"):
                title = _sanitize((entry.findtext("{http://www.w3.org/2005/Atom}title") or "").strip())
                pub   = (entry.findtext("{http://www.w3.org/2005/Atom}updated") or "")[:16]
                if title:
                    items.append({"title": title, "description": "", "published": pub})
    except Exception:
        pass
    return items


class CryptoNewsFeedInput(BaseModel):
    symbol: str = Field(description="Crypto symbol e.g. BTC, ETH")


class CryptoNewsFeedTool(BaseTool):
    name: str = "get_crypto_news_feed_sentiment"
    description: str = (
        "Fetches and analyzes recent headlines from four free crypto news RSS feeds: "
        "CoinTelegraph, Decrypt, CoinDesk, and Bitcoin Magazine. "
        "Filters for articles mentioning the target cryptocurrency and runs VADER "
        "sentiment analysis on headline text. No API key required."
    )
    args_schema: Type[BaseModel] = CryptoNewsFeedInput

    def _run(self, symbol: str) -> str:
        symbol = symbol.upper()
        keywords = _COIN_KEYWORDS.get(symbol, [symbol.lower()])
        # Always include generic crypto terms to capture market-wide mood
        generic = ["crypto", "bitcoin", "market"]

        all_articles: list[dict] = []

        for source_name, url in _RSS_FEEDS:
            try:
                resp = requests.get(url, timeout=10, headers=_RSS_HEADERS)
                resp.raise_for_status()
                items = _parse_rss_items(resp.content)
                for item in items:
                    item["source"] = source_name
                all_articles.extend(items)
            except Exception:
                continue

        if not all_articles:
            return json.dumps({"symbol": symbol, "error": "Could not fetch any RSS feeds"})

        # Split into coin-specific and general market articles
        coin_articles = []
        general_articles = []
        for art in all_articles:
            text_lower = (art["title"] + " " + art["description"]).lower()
            if any(kw in text_lower for kw in keywords):
                coin_articles.append(art)
            elif any(kw in text_lower for kw in generic):
                general_articles.append(art)

        def _score_articles(articles: list[dict]) -> tuple[list[float], list[dict]]:
            scores, samples = [], []
            for art in articles:
                s = _vader.polarity_scores(art["title"])
                scores.append(s["compound"])
                if len(samples) < 5:
                    samples.append({
                        "title": art["title"][:120],
                        "source": art["source"],
                        "published": art["published"],
                        "sentiment": _classify_compound(s["compound"]),
                        "compound": round(s["compound"], 3),
                    })
            return scores, samples

        coin_scores, coin_samples = _score_articles(coin_articles)
        gen_scores, _ = _score_articles(general_articles[:20])

        def _agg(scores: list[float]) -> dict:
            if not scores:
                return {}
            avg = sum(scores) / len(scores)
            pos = sum(1 for s in scores if s > 0.1)
            neg = sum(1 for s in scores if s < -0.1)
            return {
                "articles": len(scores),
                "avg_compound": round(avg, 3),
                "classification": _classify_compound(avg),
                "positive_pct": round(pos / len(scores) * 100, 1),
                "negative_pct": round(neg / len(scores) * 100, 1),
                "neutral_pct": round((len(scores) - pos - neg) / len(scores) * 100, 1),
            }

        result = {
            "symbol": symbol,
            "sources_fetched": len(_RSS_FEEDS),
            "total_articles_scanned": len(all_articles),
            "coin_specific": {**_agg(coin_scores), "sample_headlines": coin_samples},
            "general_market": _agg(gen_scores),
            "overall_sentiment": _classify_compound(
                sum(coin_scores) / len(coin_scores) if coin_scores else 0
            ),
        }
        return json.dumps(result)


# ── Reddit Community Sentiment ────────────────────────────────────────────────

# Subreddits to monitor per coin (coin-specific first, then general market)
_REDDIT_SUBS: dict[str, list[str]] = {
    "BTC":  ["Bitcoin", "CryptoCurrency"],
    "ETH":  ["ethereum", "CryptoCurrency"],
    "SOL":  ["solana", "CryptoCurrency"],
    "BNB":  ["binance", "CryptoCurrency"],
    "XRP":  ["Ripple", "CryptoCurrency"],
    "ADA":  ["cardano", "CryptoCurrency"],
    "DOGE": ["dogecoin", "CryptoCurrency"],
    "AVAX": ["Avax", "CryptoCurrency"],
    "DOT":  ["dot", "CryptoCurrency"],
    "LINK": ["Chainlink", "CryptoCurrency"],
    "LTC":  ["litecoin", "CryptoCurrency"],
}
_REDDIT_HEADERS = {"User-Agent": "crypto-predictor-sentiment/1.0"}


class RedditSentimentInput(BaseModel):
    symbol: str = Field(description="Crypto symbol e.g. BTC, ETH")


class RedditSentimentTool(BaseTool):
    name: str = "get_reddit_community_sentiment"
    description: str = (
        "Fetch and analyze hot posts from coin-specific and general crypto subreddits "
        "(e.g. r/Bitcoin, r/CryptoCurrency). Returns VADER sentiment scores on post titles, "
        "community upvote ratios, and top trending posts. No API key required."
    )
    args_schema: Type[BaseModel] = RedditSentimentInput

    def _run(self, symbol: str) -> str:
        symbol = symbol.upper()
        subs = _REDDIT_SUBS.get(symbol, ["CryptoCurrency"])

        all_posts: list[dict] = []
        subs_fetched: list[str] = []

        for sub in subs:
            try:
                resp = requests.get(
                    f"https://www.reddit.com/r/{sub}/hot.json",
                    params={"limit": 25},
                    headers=_REDDIT_HEADERS,
                    timeout=10,
                )
                resp.raise_for_status()
                children = resp.json()["data"]["children"]
                posts = [
                    c["data"] for c in children
                    if not c["data"].get("stickied")
                ]
                for p in posts:
                    p["_sub"] = sub
                all_posts.extend(posts)
                subs_fetched.append(sub)
            except Exception:
                continue

        if not all_posts:
            return json.dumps({"symbol": symbol, "error": "Could not fetch any Reddit data"})

        vader_scores: list[float] = []
        upvote_ratios: list[float] = []
        samples: list[dict] = []

        for p in all_posts:
            title = p.get("title", "")
            s = _vader.polarity_scores(title)
            compound = s["compound"]
            ratio = float(p.get("upvote_ratio", 0.5))
            vader_scores.append(compound)
            upvote_ratios.append(ratio)
            if len(samples) < 10:
                samples.append({
                    "title": title[:120],
                    "subreddit": p["_sub"],
                    "score": p.get("score", 0),
                    "upvote_ratio": ratio,
                    "num_comments": p.get("num_comments", 0),
                    "sentiment": _classify_compound(compound),
                    "compound": round(compound, 3),
                })

        avg_vader = sum(vader_scores) / len(vader_scores)
        avg_ratio = sum(upvote_ratios) / len(upvote_ratios)

        # Upvote ratio > 0.75 signals community approval/positivity
        community_mood = (
            "positive" if avg_ratio >= 0.75
            else "mixed" if avg_ratio >= 0.55
            else "negative"
        )

        pos_count = sum(1 for s in vader_scores if s > 0.1)
        neg_count = sum(1 for s in vader_scores if s < -0.1)
        total = len(vader_scores)

        result = {
            "source": "Reddit (public, no key required)",
            "symbol": symbol,
            "subreddits": subs_fetched,
            "posts_analyzed": total,
            "avg_vader_compound": round(avg_vader, 3),
            "vader_sentiment": _classify_compound(avg_vader),
            "positive_posts_pct": round(pos_count / total * 100, 1),
            "negative_posts_pct": round(neg_count / total * 100, 1),
            "neutral_posts_pct": round((total - pos_count - neg_count) / total * 100, 1),
            "avg_upvote_ratio": round(avg_ratio, 3),
            "community_mood": community_mood,
            "top_posts": samples,
        }
        return json.dumps(result)


# Keep alias for backwards compatibility
CryptoPanicTool = RedditSentimentTool


# ── On-Chain & Social Metrics (CoinGecko — no key) ───────────────────────────

class OnChainMetricsInput(BaseModel):
    symbol: str = Field(description="Crypto symbol e.g. BTC, ETH")


def _symbol_to_coingecko_id(symbol: str) -> str:
    mapping = {
        "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
        "SOL": "solana", "XRP": "ripple", "ADA": "cardano",
        "DOGE": "dogecoin", "AVAX": "avalanche-2", "DOT": "polkadot",
    }
    return mapping.get(symbol.upper(), symbol.lower())


class OnChainMetricsTool(BaseTool):
    name: str = "get_onchain_social_metrics"
    description: str = (
        "Fetch on-chain and social metrics from CoinGecko: developer activity, "
        "community stats (Twitter followers, Reddit subscribers, Telegram members), "
        "and public interest scores. No API key required."
    )
    args_schema: Type[BaseModel] = OnChainMetricsInput

    def _run(self, symbol: str) -> str:
        coin_id = _symbol_to_coingecko_id(symbol)
        try:
            resp = requests.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}",
                params={"localization": "false", "tickers": "false",
                        "market_data": "false", "community_data": "true",
                        "developer_data": "true", "sparkline": "false"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return json.dumps({"symbol": symbol.upper(), "error": f"CoinGecko fetch failed: {exc}"})

        community = data.get("community_data") or {}
        developer = data.get("developer_data") or {}
        sentiment = data.get("sentiment_votes_up_percentage")

        # Helper: return value only when non-null and non-zero, else omit key
        def _val(v):
            return v if v is not None else None

        # Build community block — skip fields that CoinGecko has deprecated/nulled
        community_block: dict = {}
        for label, raw_key in [
            ("reddit_subscribers",       "reddit_subscribers"),
            ("reddit_active_accounts_48h","reddit_accounts_active_48h"),
            ("twitter_followers",        "twitter_followers"),
            ("telegram_channel_users",   "telegram_channel_user_count"),
        ]:
            v = community.get(raw_key)
            if v is not None:
                community_block[label] = v
        if not community_block:
            community_block["note"] = "Community stats not available (CoinGecko deprecated several fields)"

        # Build developer block
        commits_4w = developer.get("commit_count_4_weeks")
        code_chg   = developer.get("code_additions_deletions_4_weeks") or {}
        developer_block: dict = {}
        for label, v in [
            ("github_stars",          developer.get("stars")),
            ("github_forks",          developer.get("forks")),
            ("github_issues_closed_4w", developer.get("closed_issues")),
            ("github_commits_4w",     commits_4w),
            ("code_additions_4w",     code_chg.get("additions")),
            ("code_deletions_4w",     code_chg.get("deletions")),
        ]:
            if v is not None:
                developer_block[label] = v
        if not developer_block:
            developer_block["note"] = "Developer stats not available"

        result: dict = {"symbol": symbol.upper()}

        if sentiment is not None:
            result["sentiment_votes_up_pct"]   = sentiment
            result["sentiment_votes_down_pct"] = round(100 - sentiment, 2)
            result["community_bullish_pct"]    = sentiment
        else:
            result["sentiment_votes"] = "unavailable"

        result["community_data"]  = community_block
        result["developer_data"]  = developer_block
        result["developer_activity"] = (
            "high"     if (commits_4w or 0) > 50
            else "moderate" if (commits_4w or 0) > 10
            else "low" if commits_4w is not None
            else "unknown"
        )
        return json.dumps(result, default=str)
