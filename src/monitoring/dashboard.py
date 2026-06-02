"""Trading dashboard — read-only Flask app for KRX and US market monitoring.

Routes:
    /               → Home: balance, open positions summary, today's PnL
    /positions      → Open positions detail
    /signals        → Recent signals (blocked + unblocked)
    /orders         → Recent orders
    /performance    → Daily & per-strategy breakdown
    /screener       → Active symbols list
    /system         → DB stats, candle freshness
    /api/status     → Liveness/readiness probe (JSON)

Run standalone::

    DB_PATH=/data/trading.db MARKET=KR python -m src.monitoring.dashboard

Environment variables:
    SQLITE_DB_PATH       Path to SQLite DB (default /data/trading.db)
    DASHBOARD_PORT       HTTP port (default 5000)
    TRADING_MODE         paper | live
    MARKET               KR (default) | US — switches currency and market-specific columns
    FALLBACK_BALANCE_KRW Available KRW balance fallback (MARKET=KR)
    FALLBACK_BALANCE_USD Available USD balance fallback (MARKET=US)
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import date, datetime, timedelta, timezone

from flask import Flask, jsonify, render_template_string

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

app = Flask(__name__)

_DB_PATH = os.environ.get("SQLITE_DB_PATH", "/data/trading.db")
_KST    = timezone(timedelta(hours=9))
_MARKET = os.environ.get("MARKET", "KR").upper()   # KR | US
_IS_US  = _MARKET == "US"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _q(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    conn = _get_conn()
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _q1(sql: str, params: tuple = ()) -> sqlite3.Row | None:
    conn = _get_conn()
    try:
        return conn.execute(sql, params).fetchone()
    finally:
        conn.close()


def _fmt(v) -> str:
    """Format monetary value in market currency (KRW 원 or USD $)."""
    if v is None:
        return "—"
    f = float(v)
    return f"${f:,.2f}" if _IS_US else f"{f:,.0f} 원"


def _pct(v) -> str:
    if v is None:
        return "—"
    return f"{float(v):+.2f}%"


def _currency_label() -> str:
    return "USD" if _IS_US else "KRW"


def _pos_currency_filter() -> str:
    """SQL fragment to filter positions by market currency."""
    return "AND currency='USD'" if _IS_US else "AND (currency='KRW' OR currency IS NULL)"


def _sig_currency_filter() -> str:
    """SQL fragment to filter signals by market currency."""
    return "AND (currency='USD')" if _IS_US else "AND (currency='KRW' OR currency IS NULL)"


def _ord_market_filter() -> str:
    """SQL fragment to filter orders by market (via symbols.excd)."""
    if _IS_US:
        return "AND symbol IN (SELECT symbol FROM symbols WHERE excd IS NOT NULL AND excd != '')"
    return "AND symbol NOT IN (SELECT symbol FROM symbols WHERE excd IS NOT NULL AND excd != '')"


def _candle_market_filter() -> str:
    """SQL fragment to filter klines by market (via symbols.excd)."""
    if _IS_US:
        return "WHERE symbol IN (SELECT symbol FROM symbols WHERE excd IS NOT NULL AND excd != '')"
    return "WHERE symbol NOT IN (SELECT symbol FROM symbols WHERE excd IS NOT NULL AND excd != '')"


# ---------------------------------------------------------------------------
# HTML shell
# ---------------------------------------------------------------------------

_NAV = """
<nav style="font-family:monospace;padding:8px 16px;background:#111;color:#aaa;display:flex;gap:16px;flex-wrap:wrap">
  <a href="/" style="color:#4af">Home</a>
  <a href="/positions" style="color:#4af">Positions</a>
  <a href="/signals" style="color:#4af">Signals</a>
  <a href="/orders" style="color:#4af">Orders</a>
  <a href="/performance" style="color:#4af">Performance</a>
  <a href="/screener" style="color:#4af">Screener</a>
  <a href="/system" style="color:#4af">System</a>
</nav>
"""

_STYLE = """
<style>
  body{background:#0d0d0d;color:#e0e0e0;font-family:monospace;padding:16px;margin:0}
  h2{color:#4af;margin-top:8px}
  h3{color:#7af;margin-top:20px}
  table{border-collapse:collapse;width:100%;margin-bottom:16px}
  th{background:#1a1a2e;color:#7af;text-align:left;padding:6px 12px;white-space:nowrap}
  td{padding:5px 12px;border-bottom:1px solid #1e1e1e;white-space:nowrap}
  tr:hover td{background:#131320}
  .ok{color:#4f4}
  .warn{color:#fa4}
  .danger{color:#f44}
  .muted{color:#666}
  .badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.82em;font-weight:bold}
  .badge-ok{background:#1a3d1a;color:#4f4}
  .badge-warn{background:#3d2a00;color:#fa4}
  .badge-danger{background:#3d0000;color:#f44}
  .badge-blue{background:#0a2040;color:#4af}
  .ts{color:#555;font-size:0.8em}
</style>
"""


def _page(title: str, body: str) -> str:
    now_kst = datetime.now(_KST).strftime("%Y-%m-%d %H:%M:%S KST")
    refresh = f'<p class="ts">Updated: {now_kst} &nbsp;<a href="" style="color:#4af">↺</a></p>'
    market_badge = (
        '<span class="badge badge-blue" style="font-size:0.75em;margin-left:8px">US</span>'
        if _IS_US else
        '<span class="badge badge-blue" style="font-size:0.75em;margin-left:8px">KR</span>'
    )
    return (
        f"<!DOCTYPE html><html><head>"
        f"<meta charset=utf-8><title>{_MARKET} Trading — {title}</title>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"{_STYLE}</head>"
        f"<body>{_NAV}<h2>{title}{market_badge}</h2>{body}{refresh}</body></html>"
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    trading_mode = os.environ.get("TRADING_MODE", "paper").upper()
    if _IS_US:
        balance = float(os.environ.get("FALLBACK_BALANCE_USD", "0") or 0)
    else:
        balance = float(os.environ.get("FALLBACK_BALANCE_KRW", "0") or 0)

    today_kst = datetime.now(_KST).date().isoformat()
    cf = _pos_currency_filter()

    today_stats = _q1(
        f"""SELECT COUNT(*) AS n,
                  SUM(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN 1 ELSE 0 END) AS wins,
                  SUM(CAST(realized_pnl AS REAL)) AS net
           FROM positions WHERE status='closed' AND DATE(closed_at) = ? {cf}""",
        (today_kst,),
    )
    open_count = _q1(f"SELECT COUNT(*) AS c FROM positions WHERE status='open' {cf}")
    scf = _sig_currency_filter()
    recent_signals = _q(
        f"""SELECT symbol, signal_type, strategy_name, strength_score, blocked, created_at
            FROM signals WHERE 1=1 {scf} ORDER BY created_at DESC LIMIT 8"""
    )

    n_open = (open_count["c"] if open_count else 0) or 0
    total_trades = (today_stats["n"] or 0) if today_stats else 0
    wins = (today_stats["wins"] or 0) if today_stats else 0
    today_pnl = float(today_stats["net"] or 0) if today_stats else 0.0
    win_rate = wins / total_trades * 100 if total_trades else 0.0

    mode_cls  = "warn" if trading_mode == "PAPER" else "danger"
    pnl_cls   = "ok" if today_pnl >= 0 else "danger"
    bal_label = f"Available Balance ({_currency_label()})"

    sig_rows = ""
    for s in recent_signals:
        b = bool(s["blocked"])
        b_cls = "muted" if b else ("ok" if s["signal_type"] == "long" else "warn")
        b_txt = '<span class="muted">[blocked]</span> ' if b else ""
        sig_rows += (
            f"<tr><td>{s['symbol']}</td>"
            f"<td class='{b_cls}'>{b_txt}{s['signal_type']}</td>"
            f"<td>{s['strategy_name'] or '—'}</td>"
            f"<td>{s['strength_score'] or '—'}</td>"
            f"<td class='ts'>{(s['created_at'] or '')[:16]}</td></tr>"
        )

    sig_table = (
        f"<table><tr><th>Symbol</th><th>Signal</th><th>Strategy</th><th>Score</th><th>Time</th></tr>"
        f"{sig_rows}</table>"
        if sig_rows else "<p class='muted'>No signals yet.</p>"
    )

    body = f"""
    <table style="width:auto;margin-bottom:20px">
      <tr><th>Mode</th><td><span class="badge badge-{mode_cls}">{trading_mode}</span></td></tr>
      <tr><th>{bal_label}</th><td><b>{_fmt(balance) if balance else '—'}</b></td></tr>
      <tr><th>Open Positions</th><td><b>{n_open}</b></td></tr>
      <tr><th>Today Closed Trades</th><td>{total_trades} &nbsp;({wins}W / {total_trades - wins}L) &nbsp;{win_rate:.1f}%</td></tr>
      <tr><th>Today Realized PnL</th><td class="{pnl_cls}"><b>{_fmt(today_pnl)}</b></td></tr>
    </table>
    <h3>Recent Signals</h3>
    {sig_table}
    """
    return _page("Dashboard", body)


@app.route("/positions")
def positions():
    cf = _pos_currency_filter()
    rows = _q(
        f"""SELECT symbol, side, quantity, entry_price, stop_loss, take_profit_1, take_profit_2,
                  realized_pnl, unrealized_pnl, strategy_name, trading_mode,
                  market, opened_at, t2_settle_date
           FROM positions WHERE status='open' {cf} ORDER BY opened_at DESC"""
    )
    if not rows:
        return _page("Open Positions", "<p class='muted'>No open positions.</p>")

    t2_col = "" if _IS_US else "<th>T+2</th>"
    market_col = "<th>Exchange</th>" if _IS_US else "<th>Market</th>"
    header = (
        f"<tr><th>Symbol</th>{market_col}<th>Qty</th><th>Entry</th>"
        f"<th>SL</th><th>TP1</th><th>TP2</th>"
        f"<th>uPnL</th><th>Strategy</th><th>Mode</th>{t2_col}<th>Opened</th></tr>"
    )
    trs = []
    for r in rows:
        entry = float(r["entry_price"] or 0)
        sl    = float(r["stop_loss"] or 0)
        upnl  = float(r["unrealized_pnl"] or 0)
        upnl_cls = "ok" if upnl >= 0 else "danger"
        sl_pct   = (entry - sl) / entry * 100 if entry and sl else 0
        sl_str   = f"{_fmt(sl)} ({sl_pct:.1f}%)" if sl else "—"
        mode_cls = "warn" if (r["trading_mode"] or "paper") == "paper" else "ok"
        market_val = r["market"] or ("US" if _IS_US else "KOSPI")
        t2_td = "" if _IS_US else f"<td class='ts'>{r['t2_settle_date'] or '—'}</td>"
        trs.append(
            f"<tr>"
            f"<td><b>{r['symbol']}</b></td>"
            f"<td>{market_val}</td>"
            f"<td>{r['quantity']}</td>"
            f"<td>{_fmt(entry)}</td>"
            f"<td class='warn'>{sl_str}</td>"
            f"<td>{_fmt(r['take_profit_1']) if r['take_profit_1'] else '—'}</td>"
            f"<td>{_fmt(r['take_profit_2']) if r['take_profit_2'] else '—'}</td>"
            f"<td class='{upnl_cls}'>{_fmt(upnl)}</td>"
            f"<td>{r['strategy_name'] or '—'}</td>"
            f"<td class='{mode_cls}'>{r['trading_mode'] or 'paper'}</td>"
            f"{t2_td}"
            f"<td class='ts'>{(r['opened_at'] or '')[:16]}</td>"
            f"</tr>"
        )
    body = f"<table>{header}{''.join(trs)}</table>"
    return _page("Open Positions", body)


@app.route("/signals")
def signals():
    scf = _sig_currency_filter()
    rows = _q(
        f"""SELECT symbol, signal_type, strategy_name, strength_score,
                  entry_price, sl_price, tp_price,
                  blocked, block_reason, created_at
           FROM signals WHERE 1=1 {scf} ORDER BY created_at DESC LIMIT 100"""
    )
    if not rows:
        return _page("Signals", "<p class='muted'>No signals recorded.</p>")

    blocked_count = sum(1 for r in rows if r["blocked"])
    fired_count = len(rows) - blocked_count

    summary = (
        f"<p>Showing last {len(rows)} signals — "
        f"<span class='ok'>{fired_count} fired</span> / "
        f"<span class='muted'>{blocked_count} blocked</span></p>"
    )

    header = (
        "<tr><th>Symbol</th><th>Type</th><th>Strategy</th><th>Score</th>"
        "<th>Entry</th><th>SL</th><th>TP</th>"
        "<th>Status</th><th>Block Reason</th><th>Time</th></tr>"
    )
    trs = []
    for r in rows:
        b = bool(r["blocked"])
        type_cls = "muted" if b else ("ok" if r["signal_type"] == "long" else "warn")
        status = '<span class="muted">blocked</span>' if b else '<span class="ok">fired</span>'
        trs.append(
            f"<tr>"
            f"<td><b>{r['symbol']}</b></td>"
            f"<td class='{type_cls}'>{r['signal_type']}</td>"
            f"<td>{r['strategy_name'] or '—'}</td>"
            f"<td>{r['strength_score'] or '—'}</td>"
            f"<td>{_fmt(r['entry_price']) if r['entry_price'] else '—'}</td>"
            f"<td>{_fmt(r['sl_price']) if r['sl_price'] else '—'}</td>"
            f"<td>{_fmt(r['tp_price']) if r['tp_price'] else '—'}</td>"
            f"<td>{status}</td>"
            f"<td class='muted'>{r['block_reason'] or ''}</td>"
            f"<td class='ts'>{(r['created_at'] or '')[:16]}</td>"
            f"</tr>"
        )
    body = summary + f"<table>{header}{''.join(trs)}</table>"
    return _page("Signals", body)


@app.route("/orders")
def orders():
    omf = _ord_market_filter()
    rows = _q(
        f"""SELECT symbol, side, order_type, quantity, price, avg_fill_price,
                  status, broker_order_id, fee, trading_mode, updated_at
           FROM orders WHERE 1=1 {omf} ORDER BY updated_at DESC LIMIT 100"""
    )
    if not rows:
        return _page("Orders", "<p class='muted'>No orders.</p>")

    header = (
        "<tr><th>Symbol</th><th>Side</th><th>Type</th><th>Qty</th>"
        "<th>Price</th><th>Fill Price</th><th>Status</th>"
        "<th>Fee</th><th>Mode</th><th>Time</th></tr>"
    )
    trs = []
    for r in rows:
        s = r["status"] or ""
        s_cls = "ok" if s == "filled" else ("warn" if s in ("open", "partial") else "danger")
        side_cls = "ok" if r["side"] == "buy" else "warn"
        trs.append(
            f"<tr>"
            f"<td><b>{r['symbol']}</b></td>"
            f"<td class='{side_cls}'>{r['side'].upper()}</td>"
            f"<td>{r['order_type']}</td>"
            f"<td>{r['quantity']}</td>"
            f"<td>{_fmt(r['price']) if r['price'] else '—'}</td>"
            f"<td>{_fmt(r['avg_fill_price']) if r['avg_fill_price'] else '—'}</td>"
            f"<td class='{s_cls}'>{s}</td>"
            f"<td>{_fmt(r['fee']) if r['fee'] and float(r['fee']) else '—'}</td>"
            f"<td>{r['trading_mode'] or 'paper'}</td>"
            f"<td class='ts'>{(r['updated_at'] or '')[:16]}</td>"
            f"</tr>"
        )
    body = f"<table>{header}{''.join(trs)}</table>"
    return _page("Orders", body)


@app.route("/performance")
def performance():
    today_kst = datetime.now(_KST).date()
    week_start = (today_kst - timedelta(days=today_kst.weekday())).isoformat()
    cf = _pos_currency_filter()

    week = _q1(
        f"""SELECT COUNT(*) AS t,
                  SUM(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN 1 ELSE 0 END) AS w,
                  SUM(CAST(realized_pnl AS REAL)) AS net
           FROM positions WHERE status='closed' AND DATE(closed_at) >= ? {cf}""",
        (week_start,),
    )

    daily_rows = _q(
        f"""SELECT DATE(closed_at) AS d,
                  COUNT(*) AS total,
                  SUM(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN 1 ELSE 0 END) AS wins,
                  SUM(CAST(realized_pnl AS REAL)) AS net,
                  SUM(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN CAST(realized_pnl AS REAL) ELSE 0 END) AS gross_p,
                  ABS(SUM(CASE WHEN CAST(realized_pnl AS REAL) < 0 THEN CAST(realized_pnl AS REAL) ELSE 0 END)) AS gross_l
           FROM positions WHERE status='closed' {cf}
           GROUP BY DATE(closed_at) ORDER BY d DESC LIMIT 30"""
    )

    strat_rows = _q(
        f"""SELECT strategy_name,
                  COUNT(*) AS total,
                  SUM(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN 1 ELSE 0 END) AS wins,
                  SUM(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN CAST(realized_pnl AS REAL) ELSE 0 END) AS gross_p,
                  ABS(SUM(CASE WHEN CAST(realized_pnl AS REAL) < 0 THEN CAST(realized_pnl AS REAL) ELSE 0 END)) AS gross_l,
                  SUM(CAST(realized_pnl AS REAL)) AS net
           FROM positions
           WHERE status='closed' AND realized_pnl IS NOT NULL AND realized_pnl != '0' {cf}
           GROUP BY strategy_name ORDER BY total DESC"""
    )

    week_t = (week["t"] or 0) if week else 0
    week_w = (week["w"] or 0) if week else 0
    week_net = float(week["net"] or 0) if week else 0.0
    week_wr = week_w / week_t * 100 if week_t else 0.0
    net_cls = "ok" if week_net >= 0 else "danger"

    body = f"""
    <h3>This Week ({week_start}~)</h3>
    <table style="width:auto">
      <tr><th>Trades</th><td>{week_t} ({week_w}W / {week_t - week_w}L)</td></tr>
      <tr><th>Win Rate</th><td>{week_wr:.1f}%</td></tr>
      <tr><th>Net PnL</th><td class="{net_cls}"><b>{_fmt(week_net)}</b></td></tr>
    </table>
    <h3>Daily History (last 30 days)</h3>
    """

    if daily_rows:
        header = "<tr><th>Date</th><th>Trades</th><th>Win%</th><th>Net PnL</th><th>Profit Factor</th></tr>"
        trs = []
        for r in daily_rows:
            t = r["total"] or 0
            w = r["wins"] or 0
            wr = w / t * 100 if t else 0.0
            net = float(r["net"] or 0)
            gp = float(r["gross_p"] or 0)
            gl = float(r["gross_l"] or 0)
            pf = gp / gl if gl > 0 else None
            pf_str = f"{pf:.2f}" if pf is not None else "∞"
            cls = "ok" if net >= 0 else "danger"
            trs.append(
                f"<tr><td>{r['d']}</td><td>{t} ({w}W/{t-w}L)</td>"
                f"<td>{wr:.1f}%</td><td class='{cls}'>{_fmt(net)}</td>"
                f"<td>{pf_str}</td></tr>"
            )
        body += f"<table>{header}{''.join(trs)}</table>"
    else:
        body += "<p class='muted'>No closed trades yet.</p>"

    body += "<h3>Strategy Breakdown (all-time)</h3>"
    if strat_rows:
        _KELLY_MIN = 6
        s_header = "<tr><th>Strategy</th><th>Trades</th><th>Win%</th><th>Net PnL</th><th>PF</th><th>Kelly</th></tr>"
        s_trs = []
        for r in strat_rows:
            t = r["total"] or 0
            w = r["wins"] or 0
            gp = float(r["gross_p"] or 0)
            gl = float(r["gross_l"] or 0)
            net = float(r["net"] or 0)
            wr = w / t * 100 if t else 0.0
            pf = gp / gl if gl > 0 else None
            pf_str = f"{pf:.2f}" if pf is not None else "∞"
            pf_cls = "ok" if (pf is None or pf >= 1.5) else ("warn" if pf >= 1.0 else "danger")
            net_cls = "ok" if net >= 0 else "danger"
            kelly = (f'<span class="ok">active ({t})</span>'
                     if t >= _KELLY_MIN else
                     f'<span class="warn">warming {t}/{_KELLY_MIN}</span>')
            s_trs.append(
                f"<tr><td>{r['strategy_name'] or 'unknown'}</td>"
                f"<td>{t}</td><td>{wr:.1f}%</td>"
                f"<td class='{net_cls}'>{_fmt(net)}</td>"
                f"<td class='{pf_cls}'>{pf_str}</td>"
                f"<td>{kelly}</td></tr>"
            )
        body += f"<table>{s_header}{''.join(s_trs)}</table>"
    else:
        body += "<p class='muted'>No closed trades yet.</p>"

    return _page("Performance", body)


@app.route("/screener")
def screener():
    if _IS_US:
        rows = _q(
            """SELECT s.symbol, s.is_active, s.excd, s.strategy, s.added_at,
                      (SELECT COUNT(*) FROM positions p WHERE p.symbol=s.symbol AND p.status='open') AS open_pos,
                      (SELECT COUNT(*) FROM klines k WHERE k.symbol=s.symbol AND k.interval_type='1d') AS candles
               FROM symbols s WHERE s.excd IS NOT NULL
               ORDER BY s.is_active DESC, s.symbol ASC"""
        )
    else:
        rows = _q(
            """SELECT s.symbol, s.is_active, s.market, s.sector, s.market_cap, s.strategy, s.added_at,
                      (SELECT COUNT(*) FROM positions p WHERE p.symbol=s.symbol AND p.status='open') AS open_pos,
                      (SELECT COUNT(*) FROM klines k WHERE k.symbol=s.symbol AND k.interval_type='1d') AS candles
               FROM symbols s WHERE s.excd IS NULL OR s.excd = ''
               ORDER BY s.is_active DESC, s.symbol ASC"""
        )
    if not rows:
        return _page("Screener", "<p class='muted'>No symbols in DB.</p>")

    active = sum(1 for r in rows if r["is_active"])
    if _IS_US:
        header = (
            "<tr><th>Symbol</th><th>Exchange</th><th>Strategy</th>"
            "<th>Open</th><th>Candles(1d)</th><th>Active</th><th>Added</th></tr>"
        )
    else:
        header = (
            "<tr><th>Symbol</th><th>Market</th><th>Sector</th>"
            "<th>시가총액</th><th>Strategy</th><th>Open</th>"
            "<th>Candles(1d)</th><th>Active</th><th>Added</th></tr>"
        )
    trs = []
    for r in rows:
        active_cls = "ok" if r["is_active"] else "muted"
        active_txt = "YES" if r["is_active"] else "no"
        open_cls   = "warn" if r["open_pos"] else ""
        if _IS_US:
            trs.append(
                f"<tr>"
                f"<td class='{active_cls}'><b>{r['symbol']}</b></td>"
                f"<td>{r['excd'] or '—'}</td>"
                f"<td>{r['strategy'] or '—'}</td>"
                f"<td class='{open_cls}'>{r['open_pos'] or 0}</td>"
                f"<td>{r['candles'] or 0}</td>"
                f"<td class='{active_cls}'>{active_txt}</td>"
                f"<td class='ts'>{(r['added_at'] or '')[:10]}</td>"
                f"</tr>"
            )
        else:
            trs.append(
                f"<tr>"
                f"<td class='{active_cls}'><b>{r['symbol']}</b></td>"
                f"<td>{r['market'] or '—'}</td>"
                f"<td>{r['sector'] or '—'}</td>"
                f"<td>{r['market_cap'] or '—'}</td>"
                f"<td>{r['strategy'] or '—'}</td>"
                f"<td class='{open_cls}'>{r['open_pos'] or 0}</td>"
                f"<td>{r['candles'] or 0}</td>"
                f"<td class='{active_cls}'>{active_txt}</td>"
                f"<td class='ts'>{(r['added_at'] or '')[:10]}</td>"
                f"</tr>"
            )
    body = (
        f"<p><span class='ok'>{active} active</span> / {len(rows)} total symbols</p>"
        f"<table>{header}{''.join(trs)}</table>"
    )
    return _page("Screener", body)


@app.route("/system")
def system():
    db_size_mb = 0.0
    try:
        db_size_mb = os.path.getsize(_DB_PATH) / 1024 / 1024
    except OSError:
        pass

    cmf = _candle_market_filter()
    candle_rows = _q(
        f"""SELECT symbol, interval_type, COUNT(*) AS cnt, MAX(open_time) AS latest
           FROM klines {cmf} GROUP BY symbol, interval_type ORDER BY symbol, interval_type"""
    )

    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    candle_table = ""
    if candle_rows:
        header = "<tr><th>Symbol</th><th>Interval</th><th>Count</th><th>Latest (ms)</th><th>Age</th></tr>"
        trs = []
        for r in candle_rows:
            latest_ms = int(r["latest"] or 0)
            age_s = (now_ms - latest_ms) / 1000 if latest_ms else None
            if age_s is None:
                age_str, age_cls = "—", ""
            elif age_s < 7200:  # <2h: fresh for daily candles
                age_str, age_cls = f"{age_s/3600:.1f}h", "ok"
            elif age_s < 90000:  # <25h: within one trading day
                age_str, age_cls = f"{age_s/3600:.1f}h", "warn"
            else:
                age_str, age_cls = f"{age_s/3600:.0f}h", "danger"
            trs.append(
                f"<tr><td>{r['symbol']}</td><td>{r['interval_type']}</td>"
                f"<td>{r['cnt']}</td><td class='ts'>{latest_ms}</td>"
                f"<td class='{age_cls}'>{age_str}</td></tr>"
            )
        candle_table = f"<table>{header}{''.join(trs)}</table>"
    else:
        candle_table = "<p class='muted'>No candle data.</p>"

    scf = _sig_currency_filter()
    pcf = _pos_currency_filter()
    omf = _ord_market_filter()
    sig_count   = _q1(f"SELECT COUNT(*) AS c FROM signals WHERE 1=1 {scf}")
    pos_count   = _q1(f"SELECT COUNT(*) AS c FROM positions WHERE 1=1 {pcf}")
    order_count = _q1(f"SELECT COUNT(*) AS c FROM orders WHERE 1=1 {omf}")

    body = f"""
    <h3>Database</h3>
    <table style="width:auto">
      <tr><th>Path</th><td>{_DB_PATH}</td></tr>
      <tr><th>Size</th><td>{db_size_mb:.2f} MB</td></tr>
      <tr><th>Signals</th><td>{sig_count['c'] if sig_count else 0}</td></tr>
      <tr><th>Positions</th><td>{pos_count['c'] if pos_count else 0}</td></tr>
      <tr><th>Orders</th><td>{order_count['c'] if order_count else 0}</td></tr>
    </table>
    <h3>Candle Freshness</h3>
    {candle_table}
    """
    return _page("System", body)


@app.route("/api/status")
def api_status():
    try:
        conn = _get_conn()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        db_ok = True
    except Exception:
        db_ok = False

    return jsonify({"status": "ok" if db_ok else "degraded", "db": db_ok}), (200 if db_ok else 503)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    port = int(os.environ.get("DASHBOARD_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
