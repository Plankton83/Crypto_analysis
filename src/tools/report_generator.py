"""
HTML report generator tool for the crypto prediction crew.

Produces a self-contained, dark-themed HTML report from the final prediction
text. Uses Chart.js (CDN) for a 30-day price chart and an agent-vote donut
chart. Key figures and signal words are highlighted.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ── helpers ───────────────────────────────────────────────────────────────────

def _fetch_30d_price(symbol: str) -> list[dict]:
    """Return list of {date, price} dicts for the last 30 days from Binance."""
    binance_symbol = symbol.upper()
    if not binance_symbol.endswith("USDT"):
        binance_symbol += "USDT"
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": binance_symbol, "interval": "1d", "limit": 30},
            timeout=10,
        )
        data = resp.json()
        return [
            {
                "date": datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc)
                               .strftime("%b %d"),
                "price": float(c[4]),
            }
            for c in data
        ]
    except Exception:
        return []


def _parse_vote_table(text: str) -> list[dict]:
    """
    Extract agent votes from markdown table rows.
    Returns list of {agent, signal, confidence, reason}.
    """
    agents_order = ["Macro", "On-Chain", "Market Data", "Technical", "Sentiment", "Derivatives"]
    votes = []

    # Try markdown table rows  |Agent|Signal|Confidence|Reason|
    row_re = re.compile(
        r"\|\s*([^|]+?)\s*\|\s*(BUY|SELL|NEUTRAL)\s*\|\s*(HIGH|MEDIUM|LOW)\s*\|\s*([^|]+?)\s*\|",
        re.IGNORECASE,
    )
    for m in row_re.finditer(text):
        agent_raw = m.group(1).strip()
        if agent_raw.lower() in ("agent", "---"):
            continue
        votes.append({
            "agent": agent_raw,
            "signal": m.group(2).upper(),
            "confidence": m.group(3).upper(),
            "reason": m.group(4).strip(),
        })

    # Fallback: inline AGENT VOTE lines
    if not votes:
        line_re = re.compile(
            r"SIGNAL:\s*(BUY|SELL|NEUTRAL)\s*\|\s*Confidence:\s*(HIGH|MEDIUM|LOW)\s*\|\s*Reason:\s*(.+)",
            re.IGNORECASE,
        )
        for i, m in enumerate(line_re.finditer(text)):
            votes.append({
                "agent": agents_order[i] if i < len(agents_order) else f"Agent {i+1}",
                "signal": m.group(1).upper(),
                "confidence": m.group(2).upper(),
                "reason": m.group(3).strip(),
            })

    return votes


def _parse_overall_signal(text: str) -> str:
    """Extract the final STRONG_BUY/BUY/NEUTRAL/SELL/STRONG_SELL from the report."""
    m = re.search(
        r"\b(STRONG_BUY|STRONG_SELL|BUY|SELL|NEUTRAL)\b",
        text,
        re.IGNORECASE,
    )
    return m.group(1).upper() if m else "NEUTRAL"


def _signal_color(signal: str) -> str:
    mapping = {
        "STRONG_BUY": "#00e676",
        "BUY": "#69f0ae",
        "NEUTRAL": "#ffd740",
        "SELL": "#ff6e40",
        "STRONG_SELL": "#ff1744",
    }
    return mapping.get(signal.upper(), "#ffd740")


def _vote_color(signal: str) -> str:
    if signal == "BUY":
        return "#69f0ae"
    if signal == "SELL":
        return "#ff6e40"
    return "#ffd740"


def _md_to_html(text: str) -> str:
    """
    Convert a markdown-formatted prediction text to HTML.

    Handles: headings (#/##/###), bold (**), italic (*/_), inline code (`),
    unordered lists (- / *), ordered lists (1.), horizontal rules (---),
    and plain newlines.
    """
    # Escape HTML entities first (before we add real tags)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    lines = text.split("\n")
    out: list[str] = []
    in_ul = False
    in_ol = False

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    ul_re = re.compile(r"^[\*\-]\s+(.+)$")
    ol_re = re.compile(r"^\d+\.\s+(.+)$")
    h_re  = re.compile(r"^(#{1,6})\s+(.+)$")
    hr_re = re.compile(r"^[-*_]{3,}$")

    for line in lines:
        stripped = line.strip()

        # Horizontal rule
        if hr_re.match(stripped):
            close_lists()
            out.append("<hr>")
            continue

        # Headings
        hm = h_re.match(stripped)
        if hm:
            close_lists()
            level = min(len(hm.group(1)) + 2, 6)  # # -> h3, ## -> h4, ### -> h5
            out.append(f"<h{level}>{hm.group(2)}</h{level}>")
            continue

        # Unordered list item
        um = ul_re.match(stripped)
        if um:
            if in_ol:
                close_lists()
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{um.group(1)}</li>")
            continue

        # Ordered list item
        om = ol_re.match(stripped)
        if om:
            if in_ul:
                close_lists()
            if not in_ol:
                out.append("<ol>")
                in_ol = True
            out.append(f"<li>{om.group(1)}</li>")
            continue

        # Blank line → close lists, paragraph break
        if stripped == "":
            close_lists()
            out.append("<br>")
            continue

        # Regular text line
        close_lists()
        out.append(line)

    close_lists()
    result = "\n".join(out)

    # Inline: **bold** and __bold__
    result = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", result)
    result = re.sub(r"__(.+?)__",     r"<strong>\1</strong>", result)
    # Inline: *italic* and _italic_
    result = re.sub(r"\*(.+?)\*",     r"<em>\1</em>",         result)
    result = re.sub(r"_(.+?)_",       r"<em>\1</em>",         result)
    # Inline code
    result = re.sub(r"`(.+?)`",       r"<code>\1</code>",     result)

    return result


def _highlight_key_terms(text: str) -> str:
    """Wrap key financial terms and numbers in <strong> tags (applied after md conversion)."""
    terms = [
        r"\b(STRONG_BUY|STRONG_SELL|BUY|SELL|NEUTRAL)\b",
        r"\b(HIGH|MEDIUM|LOW)\s+confidence\b",
        r"\b(bullish|bearish|overbought|oversold|golden cross|death cross|breakout|breakdown)\b",
        r"\$[\d,]+(?:\.\d+)?(?:[KMB])?",   # dollar amounts
        r"\b\d+(?:\.\d+)?%",               # percentages
        r"\b(RSI|MACD|EMA|SMA|VPOC|VIX|OI|IV)\b",
    ]
    for pattern in terms:
        # Only highlight outside existing HTML tags
        text = re.sub(
            pattern,
            lambda m: f"<strong>{m.group(0)}</strong>",
            text,
            flags=re.IGNORECASE,
        )
    return text


def _section_html(title: str, content: str) -> str:
    content_html = _highlight_key_terms(_md_to_html(content))
    return f"""
    <div class="section">
      <h2>{title}</h2>
      <div class="section-body">{content_html}</div>
    </div>"""


# ── Portfolio panel helpers ───────────────────────────────────────────────────

def _outcome_label(outcome: str) -> str:
    return {
        "sl_hit": "Stop Loss Hit",
        "tp1_hit": "TP1 Hit",
        "tp2_hit": "TP2 Hit",
        "closed_at_run": "Closed at Run",
    }.get(outcome, outcome)


def _build_portfolio_html(summary: dict) -> str:
    """Build the portfolio panel HTML block from a portfolio summary dict."""
    if not summary:
        return ""

    initial   = summary.get("initial_balance", 10000)
    cur_val   = summary.get("current_value", initial)
    pnl_usd   = summary.get("total_pnl_usd", 0.0)
    pnl_pct   = summary.get("total_pnl_pct", 0.0)
    cash      = summary.get("cash_available", 0.0)
    wins      = summary.get("trades_won", 0)
    total_tr  = summary.get("trades_total", 0)
    win_rate  = summary.get("win_rate_pct")
    pos       = summary.get("open_position")

    pnl_color = "#69f0ae" if pnl_usd >= 0 else "#ff6e40"
    pnl_sign  = "+" if pnl_usd >= 0 else ""

    # ── stat cards ──
    win_rate_str = f"{win_rate:.1f}%" if win_rate is not None else f"{wins}/{total_tr}"
    stat_cards = f"""
    <div class="port-stats">
      <div class="stat-card">
        <div class="stat-label">Portfolio Value</div>
        <div class="stat-value">${cur_val:,.2f}</div>
        <div class="stat-sub" style="color:{pnl_color}">{pnl_sign}${pnl_usd:,.2f} ({pnl_sign}{pnl_pct:.2f}%)</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Cash Available</div>
        <div class="stat-value">${cash:,.2f}</div>
        <div class="stat-sub">{cash/cur_val*100:.1f}% of portfolio</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Win Rate</div>
        <div class="stat-value">{win_rate_str}</div>
        <div class="stat-sub">{total_tr} trade{'s' if total_tr != 1 else ''} closed</div>
      </div>
    </div>"""

    # ── open position card ──
    if pos:
        direction  = pos.get("direction", "?")
        dir_color  = "#69f0ae" if direction == "LONG" else "#ff6e40"
        entry      = pos.get("entry_price", 0)
        sl         = pos.get("stop_loss", 0)
        tp1        = pos.get("take_profit_1", 0)
        tp2        = pos.get("take_profit_2")
        size_usd   = pos.get("size_usd", 0)
        size_coins = pos.get("size_coins", 0)
        conf       = pos.get("confidence", "")
        opened     = (pos.get("opened_at") or "")[:10]
        tp2_str    = f"${tp2:,.2f}" if tp2 else "—"
        pos_html = f"""
      <div class="pos-card">
        <div class="pos-header">
          <span class="pos-dir" style="color:{dir_color}">{direction}</span>
          <span class="pos-meta">{conf} confidence &nbsp;|&nbsp; opened {opened}</span>
        </div>
        <div class="pos-levels">
          <span>Entry <strong>${entry:,.2f}</strong></span>
          <span>Stop Loss <strong style="color:#ff6e40">${sl:,.2f}</strong></span>
          <span>TP1 <strong style="color:#69f0ae">${tp1:,.2f}</strong></span>
          <span>TP2 <strong style="color:#69f0ae">{tp2_str}</strong></span>
          <span>Size <strong>${size_usd:,.2f}</strong> ({size_coins} coins)</span>
        </div>
      </div>"""
    else:
        pos_html = "<div class='pos-card pos-flat'>No open position — flat</div>"

    # ── last 3 trades ──
    trade_rows = ""
    for t in reversed(summary.get("last_3_trades", [])):
        t_dir     = t.get("direction", "?")
        t_pnl     = t.get("pnl_usd", 0.0)
        t_pct     = t.get("pnl_pct", 0.0)
        t_outcome = _outcome_label(t.get("outcome", ""))
        t_date    = (t.get("closed_at") or "")[:10]
        t_color   = "#69f0ae" if t_pnl >= 0 else "#ff6e40"
        t_sign    = "+" if t_pnl >= 0 else ""
        trade_rows += (
            f"<tr>"
            f"<td>{t_date}</td>"
            f"<td>{t_dir}</td>"
            f"<td style='color:{t_color};font-weight:700'>{t_sign}${t_pnl:,.2f} ({t_sign}{t_pct:.1f}%)</td>"
            f"<td>{t_outcome}</td>"
            f"</tr>"
        )
    if trade_rows:
        trades_html = f"""
      <table class='vote-table' style='margin-top:12px'>
        <thead><tr><th>Date</th><th>Direction</th><th>P&amp;L</th><th>Outcome</th></tr></thead>
        <tbody>{trade_rows}</tbody>
      </table>"""
    else:
        trades_html = "<p style='color:#8b949e;margin-top:12px'>No closed trades yet.</p>"

    return f"""
  <div class="section" id="portfolio-panel">
    <h2>Paper Portfolio — $100,000 Starting Balance</h2>
    {stat_cards}
    {pos_html}
    <div style="margin-top:16px">
      <strong style="font-size:0.9rem;color:#8b949e">RECENT TRADES</strong>
      {trades_html}
    </div>
    <div class="chart-wrap" style="margin-top:20px;height:200px">
      <canvas id="equityChart"></canvas>
    </div>
  </div>"""


def _split_sections(text: str) -> dict[str, str]:
    """
    Best-effort split of the prediction report into named sections.
    Returns an ordered dict of {section_title: body_text}.
    """
    headings = [
        ("vote_table",    r"(?:agent vote table|vote tally|step 1)",),
        ("macro",         r"(?:macro context|step 2)",),
        ("short_term",    r"(?:short.?term|24.48\s*hour|step 3)",),
        ("medium_term",   r"(?:medium.?term|7.?day|step 4)",),
        ("long_term",     r"(?:long.?term|30.?day|step 5)",),
        ("risks",         r"(?:risk factor|conflicting|step 6)",),
        ("overall",       r"(?:overall signal|step 7)",),
    ]
    sections: dict[str, str] = {}
    remaining = text

    for i, (key, pattern) in enumerate(headings):
        m = re.search(pattern, remaining, re.IGNORECASE)
        if not m:
            continue
        # Collect until next heading or end — snap boundary back to the start
        # of the next heading LINE so we don't include the "## STEP N —" prefix
        # of the following section in the current section's body.
        end = len(remaining)
        for _, next_pattern in headings[i + 1:]:
            nm = re.search(next_pattern, remaining[m.end():], re.IGNORECASE)
            if nm:
                abs_kw = m.end() + nm.start()
                line_start = remaining.rfind("\n", 0, abs_kw)
                end = (line_start + 1) if line_start >= 0 else abs_kw
                break
        sections[key] = remaining[m.end():end].strip()
        remaining = remaining[end:]

    if not sections:
        # Fallback: entire text as "full"
        sections["full"] = text
    return sections


# ── HTML template ─────────────────────────────────────────────────────────────

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Crypto Prediction - {symbol} - {date}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 15px;
    line-height: 1.6;
    padding: 0 16px 40px;
  }}
  a {{ color: #58a6ff; }}
  h1 {{ font-size: 2rem; font-weight: 700; color: #f0f6fc; }}
  h2 {{ font-size: 1.2rem; font-weight: 600; color: #e6edf3; margin-bottom: 10px; }}
  strong {{ color: #f0f6fc; font-weight: 600; }}

  /* header */
  .header {{
    max-width: 1100px; margin: 0 auto;
    padding: 32px 0 24px;
    border-bottom: 1px solid #21262d;
    display: flex; align-items: center; gap: 20px;
  }}
  .header-meta {{ color: #8b949e; font-size: 0.875rem; margin-top: 4px; }}

  /* signal badge */
  .badge {{
    display: inline-block;
    padding: 6px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.05em;
    background: {signal_bg}22;
    color: {signal_bg};
    border: 2px solid {signal_bg};
    margin-left: auto;
    white-space: nowrap;
  }}

  /* layout */
  .container {{ max-width: 1100px; margin: 0 auto; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
  @media (max-width: 700px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

  /* cards */
  .card {{
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 20px;
  }}

  /* chart canvases */
  .chart-wrap {{ position: relative; height: 260px; }}

  /* section */
  .section {{
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 20px 24px;
    margin: 16px 0;
  }}
  .section-body {{ color: #c9d1d9; font-size: 0.93rem; }}

  /* vote table */
  .vote-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; margin-top: 8px; }}
  .vote-table th {{
    background: #21262d; color: #8b949e;
    text-align: left; padding: 8px 12px;
    font-weight: 600; letter-spacing: 0.03em;
    border-bottom: 1px solid #30363d;
  }}
  .vote-table td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; }}
  .vote-table tr:last-child td {{ border-bottom: none; }}
  .sig-buy    {{ color: #69f0ae; font-weight: 700; }}
  .sig-sell   {{ color: #ff6e40; font-weight: 700; }}
  .sig-neutral {{ color: #ffd740; font-weight: 700; }}

  /* tally chips */
  .tally {{ display: flex; gap: 12px; margin-top: 16px; flex-wrap: wrap; }}
  .chip {{
    padding: 4px 14px; border-radius: 12px; font-size: 0.83rem; font-weight: 700;
  }}
  .chip-buy    {{ background: #69f0ae22; color: #69f0ae; border: 1px solid #69f0ae; }}
  .chip-sell   {{ background: #ff6e4022; color: #ff6e40; border: 1px solid #ff6e40; }}
  .chip-neutral {{ background: #ffd74022; color: #ffd740; border: 1px solid #ffd740; }}

  /* portfolio panel */
  .port-stats {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-bottom: 16px; }}
  @media (max-width: 600px) {{ .port-stats {{ grid-template-columns: 1fr; }} }}
  .stat-card {{
    background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 14px 16px;
  }}
  .stat-label {{ color: #8b949e; font-size: 0.78rem; letter-spacing: 0.04em; text-transform: uppercase; }}
  .stat-value {{ font-size: 1.35rem; font-weight: 700; color: #f0f6fc; margin: 4px 0 2px; }}
  .stat-sub   {{ font-size: 0.82rem; color: #8b949e; }}
  .pos-card {{
    background: #0d1117; border: 1px solid #30363d; border-radius: 6px;
    padding: 14px 16px; margin-bottom: 4px;
  }}
  .pos-flat {{ color: #8b949e; font-style: italic; }}
  .pos-header {{ display: flex; align-items: center; gap: 16px; margin-bottom: 10px; }}
  .pos-dir  {{ font-size: 1.1rem; font-weight: 700; letter-spacing: 0.05em; }}
  .pos-meta {{ color: #8b949e; font-size: 0.82rem; }}
  .pos-levels {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 0.88rem; color: #c9d1d9; }}

  /* footer */
  .footer {{ max-width: 1100px; margin: 32px auto 0;
    color: #484f58; font-size: 0.8rem; text-align: center;
    border-top: 1px solid #21262d; padding-top: 16px;
  }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>{symbol} Prediction Report</h1>
    <div class="header-meta">Generated {datetime} UTC &nbsp;|&nbsp; Crypto Prediction Crew</div>
  </div>
  <div class="badge">{overall_signal}</div>
</div>

<div class="container">

  <!-- Charts row -->
  <div class="grid-2">
    <div class="card">
      <h2>30-Day Price History (USD)</h2>
      <div class="chart-wrap">
        <canvas id="priceChart"></canvas>
      </div>
    </div>
    <div class="card">
      <h2>Agent Vote Distribution</h2>
      <div class="chart-wrap">
        <canvas id="voteChart"></canvas>
      </div>
    </div>
  </div>

  <!-- Vote table -->
  <div class="section">
    <h2>Agent Votes</h2>
    {vote_table_html}
    <div class="tally">{tally_html}</div>
  </div>

  {portfolio_html}

  {sections_html}

  {strategy_html}

</div>

<div class="footer">
  This report is generated by an AI multi-agent system and is for informational purposes only.
  It does not constitute financial advice. Always do your own research before making investment decisions.
</div>

<script>
// ── Price chart ──────────────────────────────────────────────────────────────
const priceLabels = {price_labels};
const priceData   = {price_data};

new Chart(document.getElementById('priceChart'), {{
  type: 'line',
  data: {{
    labels: priceLabels,
    datasets: [{{
      label: '{symbol} Close (USD)',
      data: priceData,
      borderColor: '#58a6ff',
      backgroundColor: 'rgba(88,166,255,0.08)',
      borderWidth: 2,
      pointRadius: 2,
      tension: 0.3,
      fill: true,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }}, tooltip: {{
      callbacks: {{
        label: ctx => ' $' + ctx.parsed.y.toLocaleString(undefined, {{maximumFractionDigits:2}})
      }}
    }} }},
    scales: {{
      x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 8 }}, grid: {{ color: '#21262d' }} }},
      y: {{
        ticks: {{ color: '#8b949e', callback: v => '$' + v.toLocaleString() }},
        grid: {{ color: '#21262d' }}
      }}
    }}
  }}
}});

// ── Equity curve chart ───────────────────────────────────────────────────────
const equityLabels = {equity_labels};
const equityData   = {equity_data};
const equityCanvas = document.getElementById('equityChart');
if (equityCanvas && equityLabels.length > 1) {{
  const startVal = equityData[0] || 10000;
  new Chart(equityCanvas, {{
    type: 'line',
    data: {{
      labels: equityLabels,
      datasets: [{{
        label: 'Portfolio Value ($)',
        data: equityData,
        borderColor: equityData[equityData.length-1] >= startVal ? '#69f0ae' : '#ff6e40',
        backgroundColor: equityData[equityData.length-1] >= startVal
          ? 'rgba(105,240,174,0.07)' : 'rgba(255,110,64,0.07)',
        borderWidth: 2,
        pointRadius: 3,
        tension: 0.3,
        fill: true,
      }},{{
        label: 'Baseline ($10k)',
        data: equityLabels.map(() => 10000),
        borderColor: '#484f58',
        borderWidth: 1,
        borderDash: [4,4],
        pointRadius: 0,
        fill: false,
      }}]
    }},
    options: {{
      responsive: true, maintainAspectRatio: false,
      plugins: {{ legend: {{ display: false }} }},
      scales: {{
        x: {{ ticks: {{ color: '#8b949e', maxTicksLimit: 6 }}, grid: {{ color: '#21262d' }} }},
        y: {{ ticks: {{ color: '#8b949e', callback: v => '$' + v.toLocaleString() }},
              grid: {{ color: '#21262d' }} }}
      }}
    }}
  }});
}}

// ── Vote donut chart ─────────────────────────────────────────────────────────
const voteCounts  = {vote_counts_json};
const voteColors  = ['#69f0ae', '#ff6e40', '#ffd740'];
const voteLabels  = ['BUY', 'SELL', 'NEUTRAL'];

new Chart(document.getElementById('voteChart'), {{
  type: 'doughnut',
  data: {{
    labels: voteLabels,
    datasets: [{{
      data: voteCounts,
      backgroundColor: voteColors.map(c => c + '99'),
      borderColor: voteColors,
      borderWidth: 2,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{
        position: 'bottom',
        labels: {{ color: '#c9d1d9', padding: 16, font: {{ size: 13 }} }}
      }},
      tooltip: {{ callbacks: {{
        label: ctx => ` ${{ctx.label}}: ${{ctx.parsed}} vote${{ctx.parsed !== 1 ? 's' : ''}}`
      }} }}
    }},
    cutout: '62%',
  }}
}});
</script>
</body>
</html>
"""


# ── Tool ─────────────────────────────────────────────────────────────────────

class HtmlReportInput(BaseModel):
    symbol: str = Field(..., description="Cryptocurrency symbol, e.g. BTC")
    prediction_text: str = Field(..., description="Full prediction report text from the Chief Strategist")
    strategy_text: Optional[str] = Field(default=None, description="Full trade plan text from the Trading Strategist")


class HtmlReportTool(BaseTool):
    name: str = "html_report_generator"
    description: str = (
        "Generates a visually rich HTML report from the prediction text and saves it to the "
        "reports/ folder. Returns the absolute path to the saved HTML file."
    )
    args_schema: Type[BaseModel] = HtmlReportInput

    def _run(self, symbol: str, prediction_text: str, strategy_text: Optional[str] = None) -> str:
        symbol = symbol.upper()

        # Strip machine-readable sentinel lines — must not appear in the HTML.
        # Truncate prediction_text AT the sentinel: everything after it is noise
        # (CrewAI context causes the strategy output to bleed in after this marker).
        _psj = re.search(r"PREDICTION_SIGNAL_JSON:\s*\{[^}]*\}", prediction_text, re.IGNORECASE)
        if _psj:
            prediction_text = prediction_text[:_psj.start()].strip()
        else:
            prediction_text = prediction_text.strip()
        # Belt-and-suspenders: also strip any stray TRADE_PLAN_JSON that bled in
        prediction_text = re.sub(
            r"TRADE_PLAN_JSON:\s*\{[^}]+\}", "", prediction_text, flags=re.IGNORECASE
        ).strip()

        if strategy_text:
            strategy_text = re.sub(
                r"TRADE_PLAN_JSON:\s*\{[^}]+\}",
                "",
                strategy_text,
                flags=re.IGNORECASE,
            ).strip()

        now = datetime.now(tz=timezone.utc)
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        date_str  = now.strftime("%Y-%m-%d")
        dt_str    = now.strftime("%Y-%m-%d %H:%M")

        # ── price data ──
        price_points = _fetch_30d_price(symbol)
        price_labels = json.dumps([p["date"] for p in price_points])
        price_data   = json.dumps([p["price"] for p in price_points])

        # ── vote data ──
        votes = _parse_vote_table(prediction_text)
        buy_count     = sum(1 for v in votes if v["signal"] == "BUY")
        sell_count    = sum(1 for v in votes if v["signal"] == "SELL")
        neutral_count = sum(1 for v in votes if v["signal"] == "NEUTRAL")
        vote_counts   = json.dumps([buy_count, sell_count, neutral_count])

        # ── vote table HTML ──
        if votes:
            rows = []
            for v in votes:
                sig_class = {"BUY": "sig-buy", "SELL": "sig-sell"}.get(v["signal"], "sig-neutral")
                conf_str  = v["confidence"]
                reason_esc = v["reason"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                rows.append(
                    f"<tr>"
                    f"<td>{v['agent']}</td>"
                    f"<td class='{sig_class}'>{v['signal']}</td>"
                    f"<td>{conf_str}</td>"
                    f"<td>{reason_esc}</td>"
                    f"</tr>"
                )
            vote_table_html = (
                "<table class='vote-table'>"
                "<thead><tr><th>Agent</th><th>Signal</th><th>Confidence</th><th>Key Reason</th></tr></thead>"
                "<tbody>" + "".join(rows) + "</tbody></table>"
            )
        else:
            vote_table_html = "<p style='color:#8b949e'>No structured vote data found in report.</p>"

        # ── tally chips ──
        tally_parts = []
        if buy_count:
            tally_parts.append(f"<span class='chip chip-buy'>{buy_count} BUY</span>")
        if sell_count:
            tally_parts.append(f"<span class='chip chip-sell'>{sell_count} SELL</span>")
        if neutral_count:
            tally_parts.append(f"<span class='chip chip-neutral'>{neutral_count} NEUTRAL</span>")
        tally_html = "".join(tally_parts) if tally_parts else ""

        # ── overall signal ──
        overall_signal = _parse_overall_signal(prediction_text)
        signal_bg      = _signal_color(overall_signal)

        # ── section blocks ──
        section_titles = {
            "vote_table": None,          # already rendered above
            "macro":      "Macro Context",
            "short_term": "Short-Term Outlook (24-48 hours)",
            "medium_term":"Medium-Term Outlook (7 Days)",
            "long_term":  "Longer-Term Outlook (30 Days)",
            "risks":      "Risk Factors & Conflicting Signals",
            "overall":    "Overall Signal",
            "full":       "Full Analysis",
        }
        parsed = _split_sections(prediction_text)
        sections_html_parts = []
        for key, title in section_titles.items():
            if key in parsed and title is not None:
                sections_html_parts.append(_section_html(title, parsed[key]))

        # Fallback: render entire text as a single section
        if not sections_html_parts:
            sections_html_parts.append(_section_html("Full Prediction Report", prediction_text))

        sections_html = "\n".join(sections_html_parts)

        # ── portfolio panel ──
        portfolio_html = ""
        equity_labels  = json.dumps([])
        equity_data    = json.dumps([])
        try:
            from ..portfolio_store import load_portfolio, get_portfolio_summary
            current_price_now = price_points[-1]["price"] if price_points else 0.0
            portfolio    = load_portfolio(symbol)
            port_summary = get_portfolio_summary(portfolio, current_price_now)
            portfolio_html = _build_portfolio_html(port_summary)
            curve = port_summary.get("equity_curve", [])
            equity_labels = json.dumps([pt["date"] for pt in curve])
            equity_data   = json.dumps([pt["value"] for pt in curve])
        except Exception:
            pass

        # ── strategy section ──
        if strategy_text and strategy_text.strip():
            strategy_html = _section_html("Portfolio Overview", strategy_text)
        else:
            strategy_html = ""

        # ── render ──
        html = _HTML_TEMPLATE.format(
            symbol=symbol,
            date=date_str,
            datetime=dt_str,
            overall_signal=overall_signal.replace("_", " "),
            signal_bg=signal_bg,
            price_labels=price_labels,
            price_data=price_data,
            vote_counts_json=vote_counts,
            vote_table_html=vote_table_html,
            tally_html=tally_html,
            portfolio_html=portfolio_html,
            strategy_html=strategy_html,
            equity_labels=equity_labels,
            equity_data=equity_data,
            sections_html=sections_html,
        )

        # ── save ──
        reports_dir = Path(__file__).resolve().parents[2] / "reports"
        reports_dir.mkdir(exist_ok=True)
        filename = f"{symbol}_{timestamp}.html"
        output_path = reports_dir / filename
        output_path.write_text(html, encoding="utf-8")

        return str(output_path.resolve())
