"""Flask monitoring dashboard.

Serves read-only HTML/JSON views of the trading DB.

Routes:
    /                → main: P&L summary, safe mode, trading mode
    /positions       → open positions + liquidation distance
    /liquidation     → liquidation risk monitor
    /safety          → SafeMode / market shock status
    /orders          → recent order history
    /performance     → Sharpe, MDD, win rate, liquidation count
    /screener        → active symbol list
    /system          → DB stats, candle freshness
    /api/status      → K8s liveness/readiness probe (JSON)

Run standalone::

    DB_PATH=/data/trading.db python -m src.monitoring.dashboard

Environment variables:
    SQLITE_DB_PATH   Path to SQLite DB (default /data/trading.db)
    DASHBOARD_PORT   HTTP port (default 5000)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone

import ccxt
from flask import Flask, jsonify, render_template_string

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

app = Flask(__name__)

_DB_PATH = os.environ.get("SQLITE_DB_PATH", "/data/trading.db")


# ---------------------------------------------------------------------------
# Balance cache (avoid hammering exchange on every page load)
# ---------------------------------------------------------------------------

_balance_cache: dict = {}
_balance_cache_time: float = 0.0
_BALANCE_TTL = 60  # seconds
_balance_error: str = ""


def _get_exchange() -> ccxt.Exchange:
    trading_mode = os.environ.get("TRADING_MODE", "testnet").lower()
    if trading_mode == "testnet":
        api_key = os.environ.get("BINANCE_DEMO_API_KEY") or os.environ.get("BINANCE_API_KEY", "")
        api_secret = os.environ.get("BINANCE_DEMO_API_SECRET") or os.environ.get("BINANCE_API_SECRET", "")
    else:
        api_key = os.environ.get("BINANCE_API_KEY", "")
        api_secret = os.environ.get("BINANCE_API_SECRET", "")
    exchange = ccxt.binanceusdm({"apiKey": api_key, "secret": api_secret, "options": {"defaultType": "future"}})
    if trading_mode == "testnet":
        exchange.enable_demo_trading(True)
    return exchange


def _fetch_balance() -> dict:
    """Return {wallet, equity, free, used, error} from exchange. Cached 60s.

    wallet = walletBalance (realized only, matches Binance console "Wallet Balance")
    equity = marginBalance = wallet + unrealizedProfit
    """
    global _balance_cache, _balance_cache_time, _balance_error
    if time.time() - _balance_cache_time < _BALANCE_TTL and _balance_cache:
        return _balance_cache
    try:
        raw = _get_exchange().fetch_balance()
        wallet = None
        equity = None
        free = None

        # Try raw info.assets first — most precise field names
        if "info" in raw:
            assets = raw["info"].get("assets", [])
            for a in assets:
                if a.get("asset") == "USDT":
                    wallet = float(a.get("walletBalance") or 0)
                    equity = float(a.get("marginBalance") or wallet)
                    free = float(a.get("availableBalance") or 0)
                    break

        # Fallback: ccxt normalized (total = marginBalance in futures)
        if wallet is None:
            usdt = raw.get("USDT") or raw.get("usdt") or {}
            equity = float(usdt.get("total") or 0) if usdt else None
            free = float(usdt.get("free") or 0) if usdt else None
            wallet = equity  # walletBalance not available via ccxt normalized

        used = (equity - free) if (equity is not None and free is not None) else None
        _balance_cache = {
            "wallet": wallet, "equity": equity,
            "free": free, "used": used, "error": None,
        }
        _balance_error = ""
        _balance_cache_time = time.time()
    except Exception as exc:
        _log.exception("fetch_balance failed")
        _balance_error = str(exc)
        _balance_cache = {"wallet": None, "equity": None, "free": None, "used": None, "error": str(exc)}
    return _balance_cache


# ---------------------------------------------------------------------------
# DB helper
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


def _regime_info() -> dict:
    """Compute BTC ADX from klines and derive regime/limit/TP-scale info."""
    try:
        import numpy as np
        import talib

        rows = _q(
            "SELECT high, low, close FROM klines WHERE symbol='BTCUSDT' "
            "ORDER BY open_time DESC LIMIT 60"
        )
        if len(rows) < 30:
            raise ValueError("insufficient BTC candles")
        rows = list(reversed(rows))
        high  = np.array([float(r["high"])  for r in rows])
        low   = np.array([float(r["low"])   for r in rows])
        close = np.array([float(r["close"]) for r in rows])
        adx_arr = talib.ADX(high, low, close, timeperiod=14)
        adx = float(adx_arr[-1])
        if np.isnan(adx):
            raise ValueError("ADX is NaN")
    except Exception:
        return {"adx": None, "label": "unknown", "label_cls": "warn",
                "limit": "?", "base": "?", "tp_scale": 1.0}

    if adx >= 25:
        label, label_cls = "TRENDING", "ok"
    elif adx >= 20:
        label, label_cls = "WEAK", "warn"
    else:
        label, label_cls = "RANGING", "danger"

    try:
        from src.signal.signal_blocker import _dynamic_max_positions
        from src.utils.config import load_config
        base = load_config().max_positions
        limit = _dynamic_max_positions(adx, base)
    except Exception:
        base, limit = "?", "?"

    tp_scale = max(0.75, min(1.25, 1.0 + (adx - 25.0) * 0.01))
    return {"adx": adx, "label": label, "label_cls": label_cls,
            "limit": limit, "base": base, "tp_scale": tp_scale}


# ---------------------------------------------------------------------------
# Shared HTML shell
# ---------------------------------------------------------------------------

_NAV = """
<nav style="font-family:monospace;padding:8px;background:#111;color:#aaa">
  <a href="/" style="color:#4af;margin-right:12px">Home</a>
  <a href="/positions" style="color:#4af;margin-right:12px">Positions</a>
  <a href="/trades" style="color:#4af;margin-right:12px">Trades</a>
  <a href="/liquidation" style="color:#4af;margin-right:12px">Liquidation</a>
  <a href="/safety" style="color:#4af;margin-right:12px">Safety</a>
  <a href="/orders" style="color:#4af;margin-right:12px">Orders</a>
  <a href="/performance" style="color:#4af;margin-right:12px">Performance</a>
  <a href="/screener" style="color:#4af;margin-right:12px">Screener</a>
  <a href="/system" style="color:#4af">System</a>
</nav>
"""

_STYLE = """
<style>
  body{background:#0d0d0d;color:#e0e0e0;font-family:monospace;padding:16px}
  h2{color:#4af}
  table{border-collapse:collapse;width:100%}
  th{background:#1a1a2e;color:#7af;text-align:left;padding:6px 10px}
  td{padding:5px 10px;border-bottom:1px solid #222}
  tr:hover td{background:#1a1a1a}
  .ok{color:#4f4}
  .warn{color:#fa4}
  .danger{color:#f44}
  .badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.85em}
  .badge-ok{background:#1a3d1a;color:#4f4}
  .badge-warn{background:#3d2a00;color:#fa4}
  .badge-danger{background:#3d0000;color:#f44}
</style>
"""


def _page(title: str, body: str) -> str:
    return f"<!DOCTYPE html><html><head><title>{title}</title>{_STYLE}</head><body>{_NAV}<h2>{title}</h2>{body}</body></html>"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    today = date.today().isoformat()

    today_stats = _q1(
        """SELECT COUNT(*) AS n,
                  SUM(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN 1 ELSE 0 END) AS wins,
                  SUM(CAST(realized_pnl AS REAL)) AS net
           FROM positions WHERE status='closed' AND DATE(closed_at)=?""",
        (today,),
    )
    unrealized = _q1(
        "SELECT COALESCE(SUM(CAST(unrealized_pnl AS REAL)),0) AS u FROM positions WHERE status='open'"
    )
    open_count = _q1("SELECT COUNT(*) AS c FROM positions WHERE status='open'")
    safe_mode = _q1("SELECT action, reason, created_at FROM safe_mode_events ORDER BY rowid DESC LIMIT 1")
    trading_mode = os.environ.get("TRADING_MODE", "testnet").upper()

    total_trades = today_stats["n"] or 0 if today_stats else 0
    wins = today_stats["wins"] or 0 if today_stats else 0
    today_pnl = float(today_stats["net"] or 0) if today_stats else 0.0
    u_pnl = float(unrealized["u"] or 0) if unrealized else 0.0
    n_open = open_count["c"] if open_count else 0
    win_rate = (wins / total_trades * 100) if total_trades else 0.0

    regime = _regime_info()
    dir_counts = _q(
        "SELECT side, COUNT(*) AS c FROM positions WHERE status='open' GROUP BY side"
    )
    dir_map = {r["side"]: r["c"] for r in dir_counts}
    longs  = dir_map.get("long", 0)
    shorts = dir_map.get("short", 0)

    balance = _fetch_balance()
    sm_active = safe_mode and safe_mode["action"] == "activated"
    sm_reason_index = (safe_mode["reason"] or "") if safe_mode else ""
    sm_badge = '<span class="badge badge-danger">ACTIVE</span>' if sm_active else '<span class="badge badge-ok">OFF</span>'
    mode_badge = f'<span class="badge badge-{"warn" if trading_mode=="TESTNET" else "danger"}">{trading_mode}</span>'

    pnl_class = "ok" if today_pnl >= 0 else "danger"
    u_class = "ok" if u_pnl >= 0 else "danger"

    bal_wallet = f"{balance['wallet']:.2f} USDT" if balance["wallet"] is not None else "N/A"
    bal_equity = f"{balance['equity']:.2f} USDT" if balance["equity"] is not None else "N/A"
    bal_free   = f"{balance['free']:.2f} USDT"   if balance["free"]   is not None else "N/A"
    bal_used   = f"{balance['used']:.2f} USDT"   if balance["used"]   is not None else "N/A"
    bal_err    = balance.get("error") or ""
    bal_err_row = f'<tr><td colspan="2" class="danger" style="font-size:0.8em">⚠ {bal_err}</td></tr>' if bal_err else ""

    adx_str    = f"{regime['adx']:.1f}" if regime["adx"] is not None else "N/A"
    tp_str     = f"×{regime['tp_scale']:.2f}" if regime["adx"] is not None else "N/A"
    limit_str  = f"{regime['limit']} / {regime['base']}"
    lc         = regime["label_cls"]

    body = f"""
    <table>
      <tr><th>Trading Mode</th><td>{mode_badge}</td></tr>
      <tr><th>Safe Mode</th><td>{sm_badge} {(safe_mode["reason"] or "") if sm_active else ""}</td></tr>
      <tr><th colspan="2" style="color:#7af;padding-top:10px">📡 Regime (BTC ADX)</th></tr>
      <tr><th>BTC ADX</th><td class="{lc}"><b>{adx_str}</b> — {regime['label']}</td></tr>
      <tr><th>Position Limit</th><td>{limit_str}</td></tr>
      <tr><th>TP Scale</th><td>{tp_str}</td></tr>
      <tr><th>Direction Mix</th><td>
        <span class="ok">{longs}L</span> / <span class="danger">{shorts}S</span>
        &nbsp;({n_open} total)
      </td></tr>
      <tr><th colspan="2" style="color:#7af;padding-top:10px">💰 Balance</th></tr>
      {bal_err_row}
      <tr><th>Wallet Balance</th><td><b>{bal_wallet}</b></td></tr>
      <tr><th>Total Equity (incl. uPnL)</th><td>{bal_equity}</td></tr>
      <tr><th>Available (Free)</th><td class="ok">{bal_free}</td></tr>
      <tr><th>In Use (Margin)</th><td class="warn">{bal_used}</td></tr>
      <tr><th colspan="2" style="color:#7af;padding-top:10px">📊 P&amp;L</th></tr>
      <tr><th>Open Positions</th><td>{n_open}</td></tr>
      <tr><th>Today Realised PnL</th><td class="{pnl_class}">{today_pnl:+.2f} USDT</td></tr>
      <tr><th>Unrealised PnL</th><td class="{u_class}">{u_pnl:+.2f} USDT</td></tr>
      <tr><th>Today Win Rate</th><td>{win_rate:.1f}% ({wins}/{total_trades})</td></tr>
    </table>
    <p style="color:#555;font-size:0.8em">Refreshed: {datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S")} KST
    &nbsp;<a href="/" style="color:#4af">↺</a></p>
    """
    return _page("Dashboard", body)


@app.route("/positions")
def positions():
    rows = _q(
        """SELECT symbol, side, quantity, entry_price, liquidation_price,
                  stop_loss, take_profit_1, unrealized_pnl, leverage, opened_at
           FROM positions WHERE status='open' ORDER BY opened_at"""
    )
    if not rows:
        return _page("Open Positions", "<p>No open positions.</p>")

    header = "<tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Liq Price</th><th>SL</th><th>TP1</th><th>uPnL</th><th>Lev</th><th>Opened</th></tr>"
    trs = []
    for r in rows:
        entry = float(r["entry_price"] or 0)
        liq = float(r["liquidation_price"] or 0)
        pnl = float(r["unrealized_pnl"] or 0)
        dist_pct = abs((liq - entry) / entry * 100) if entry else 0
        dist_class = "danger" if dist_pct < 5 else ("warn" if dist_pct < 10 else "ok")
        pnl_class = "ok" if pnl >= 0 else "danger"
        trs.append(
            f"<tr><td>{r['symbol']}</td><td>{r['side'].upper()}</td>"
            f"<td>{r['quantity']}</td><td>{entry:.4f}</td>"
            f"<td class='{dist_class}'>{liq:.4f} ({dist_pct:.1f}%)</td>"
            f"<td>{r['stop_loss'] or '-'}</td><td>{r['take_profit_1'] or '-'}</td>"
            f"<td class='{pnl_class}'>{pnl:+.2f}</td>"
            f"<td>{r['leverage']}x</td><td>{r['opened_at']}</td></tr>"
        )
    body = f"<table>{header}{''.join(trs)}</table>"
    return _page("Open Positions", body)


@app.route("/trades")
def trades():
    rows = _q(
        """SELECT symbol, side, entry_price, exit_price, quantity, realized_pnl,
                  close_reason, opened_at, closed_at
           FROM positions WHERE status='closed'
           ORDER BY closed_at DESC LIMIT 100"""
    )
    if not rows:
        return _page("Trade History", "<p>No closed trades yet.</p>")

    total = len(rows)
    wins = sum(1 for r in rows if float(r["realized_pnl"] or 0) > 0)
    net = sum(float(r["realized_pnl"] or 0) for r in rows)
    gross_p = sum(float(r["realized_pnl"] or 0) for r in rows if float(r["realized_pnl"] or 0) > 0)
    gross_l = abs(sum(float(r["realized_pnl"] or 0) for r in rows if float(r["realized_pnl"] or 0) < 0))
    pf = gross_p / gross_l if gross_l > 0 else float("inf")
    wr = wins / total * 100 if total else 0.0
    net_cls = "ok" if net >= 0 else "danger"
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"

    summary = f"""
    <table style="margin-bottom:16px;width:auto">
      <tr><th>Trades (last 100)</th><td>{total}</td></tr>
      <tr><th>Win Rate</th><td>{wr:.1f}% ({wins}W / {total - wins}L)</td></tr>
      <tr><th>Net PnL</th><td class="{net_cls}">{net:+.4f} USDT</td></tr>
      <tr><th>Profit Factor</th><td>{pf_str}</td></tr>
    </table>
    """

    header = "<tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>PnL</th><th>Result</th><th>Reason</th><th>Duration</th><th>Closed</th></tr>"
    trs = []
    for r in rows:
        pnl = float(r["realized_pnl"] or 0)
        pnl_cls = "ok" if pnl > 0 else ("danger" if pnl < 0 else "")
        result = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BE")
        entry = float(r["entry_price"] or 0)
        exit_p = float(r["exit_price"] or 0)
        # Duration
        duration = "-"
        if r["opened_at"] and r["closed_at"]:
            try:
                from datetime import datetime, timezone
                fmt = "%Y-%m-%dT%H:%M:%S.%f%z"
                o = datetime.fromisoformat(r["opened_at"])
                c = datetime.fromisoformat(r["closed_at"])
                secs = int((c - o).total_seconds())
                h, rem = divmod(secs, 3600)
                m = rem // 60
                duration = f"{h}h{m:02d}m" if h else f"{m}m"
            except Exception:
                pass
        trs.append(
            f"<tr>"
            f"<td>{r['symbol']}</td>"
            f"<td>{r['side'].upper()}</td>"
            f"<td>{entry:.4f}</td>"
            f"<td>{exit_p:.4f}</td>"
            f"<td class='{pnl_cls}'>{pnl:+.4f}</td>"
            f"<td class='{pnl_cls}'><b>{result}</b></td>"
            f"<td>{r['close_reason'] or '-'}</td>"
            f"<td>{duration}</td>"
            f"<td>{(r['closed_at'] or '')[:19]}</td>"
            f"</tr>"
        )
    body = summary + f"<table>{header}{''.join(trs)}</table>"
    return _page("Trade History", body)


@app.route("/liquidation")
def liquidation():
    rows = _q(
        """SELECT symbol, side, entry_price, liquidation_price, quantity, leverage
           FROM positions WHERE status='open' ORDER BY
           ABS(CAST(liquidation_price AS REAL) - CAST(entry_price AS REAL)) /
           CAST(entry_price AS REAL) ASC"""
    )
    if not rows:
        return _page("Liquidation Risk", "<p>No open positions.</p>")

    header = "<tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Liq Price</th><th>Distance</th><th>Risk</th></tr>"
    trs = []
    for r in rows:
        entry = float(r["entry_price"] or 0)
        liq = float(r["liquidation_price"] or 0)
        dist_pct = abs((liq - entry) / entry * 100) if entry else 0
        if dist_pct < 3:
            risk, cls = "CRITICAL", "danger"
        elif dist_pct < 7:
            risk, cls = "HIGH", "warn"
        else:
            risk, cls = "OK", "ok"
        trs.append(
            f"<tr><td>{r['symbol']}</td><td>{r['side'].upper()}</td>"
            f"<td>{entry:.4f}</td><td>{liq:.4f}</td>"
            f"<td class='{cls}'>{dist_pct:.2f}%</td>"
            f"<td class='{cls}'><b>{risk}</b></td></tr>"
        )
    body = f"<table>{header}{''.join(trs)}</table>"
    return _page("Liquidation Risk", body)


@app.route("/safety")
def safety():
    sm = _q1("SELECT action, reason, created_at FROM safe_mode_events ORDER BY rowid DESC LIMIT 1")
    shocks = _q(
        """SELECT risk_level, risk_score, action_taken, created_at
           FROM market_shock_events ORDER BY created_at DESC LIMIT 20"""
    )

    sm_is_active = sm and sm["action"] == "activated"
    sm_status = "ACTIVE" if sm_is_active else "OFF"
    sm_cls = "danger" if sm_is_active else "ok"
    sm_reason = (sm["reason"] or "—") if sm_is_active else "—"
    sm_since = (sm["created_at"] or "—") if (sm and sm_is_active) else "—"

    sm_history = _q(
        "SELECT action, reason, by, created_at FROM safe_mode_events ORDER BY rowid DESC LIMIT 10"
    )

    body = f"""
    <h3>Safe Mode</h3>
    <table>
      <tr><th>Status</th><td class="{sm_cls}"><b>{sm_status}</b></td></tr>
      <tr><th>Reason</th><td>{sm_reason}</td></tr>
      <tr><th>Since</th><td>{sm_since}</td></tr>
    </table>
    <h3>Safe Mode History</h3>
    """
    if sm_history:
        h_header = "<tr><th>Action</th><th>Reason</th><th>By</th><th>Time</th></tr>"
        h_trs = []
        for e in sm_history:
            a_cls = "danger" if e["action"] == "activated" else "ok"
            h_trs.append(
                f"<tr><td class='{a_cls}'>{e['action']}</td>"
                f"<td>{e['reason'] or '-'}</td><td>{e['by'] or '-'}</td>"
                f"<td>{e['created_at']}</td></tr>"
            )
        body += f"<table>{h_header}{''.join(h_trs)}</table>"
    else:
        body += "<p>No safe mode events.</p>"

    body += "<h3>Recent Market Shock Events</h3>"
    if shocks:
        header = "<tr><th>Level</th><th>Score</th><th>Action</th><th>Time</th></tr>"
        trs = []
        for s in shocks:
            cls = "danger" if s["risk_level"] == "DANGER" else "warn"
            trs.append(
                f"<tr><td class='{cls}'>{s['risk_level']}</td>"
                f"<td>{s['risk_score']}</td>"
                f"<td>{s['action_taken'] or '-'}</td>"
                f"<td>{s['created_at']}</td></tr>"
            )
        body += f"<table>{header}{''.join(trs)}</table>"
    else:
        body += "<p>No shock events recorded.</p>"

    return _page("Safety Monitor", body)


@app.route("/orders")
def orders():
    rows = _q(
        """SELECT symbol, side, order_type, quantity, avg_fill_price, status,
                  fee, trading_mode, created_at
           FROM orders ORDER BY created_at DESC LIMIT 100"""
    )
    if not rows:
        return _page("Order History", "<p>No orders.</p>")

    header = "<tr><th>Symbol</th><th>Side</th><th>Type</th><th>Qty</th><th>Fill Price</th><th>Status</th><th>Fee</th><th>Mode</th><th>Time</th></tr>"
    trs = []
    for r in rows:
        status_cls = "ok" if r["status"] == "filled" else ("warn" if r["status"] == "open" else "danger")
        trs.append(
            f"<tr><td>{r['symbol']}</td><td>{r['side'].upper()}</td>"
            f"<td>{r['order_type']}</td><td>{r['quantity']}</td>"
            f"<td>{r['avg_fill_price'] or '-'}</td>"
            f"<td class='{status_cls}'>{r['status']}</td>"
            f"<td>{float(r['fee'] or 0):.4f}</td>"
            f"<td>{r['trading_mode']}</td><td>{r['created_at']}</td></tr>"
        )
    body = f"<table>{header}{''.join(trs)}</table>"
    return _page("Order History", body)


@app.route("/performance")
def performance():
    today = date.today()
    week_start = (today - timedelta(days=today.weekday())).isoformat()

    week = _q1(
        """SELECT COUNT(*) AS t,
                  SUM(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN 1 ELSE 0 END) AS w,
                  SUM(CAST(realized_pnl AS REAL)) AS net
           FROM positions WHERE status='closed' AND DATE(closed_at) >= ?""",
        (week_start,),
    )

    rows = _q(
        """SELECT DATE(closed_at) AS d,
                  COUNT(*) AS total,
                  SUM(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN 1 ELSE 0 END) AS wins,
                  SUM(CAST(realized_pnl AS REAL)) AS net,
                  SUM(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN CAST(realized_pnl AS REAL) ELSE 0 END) AS gross_p,
                  ABS(SUM(CASE WHEN CAST(realized_pnl AS REAL) < 0 THEN CAST(realized_pnl AS REAL) ELSE 0 END)) AS gross_l
           FROM positions WHERE status='closed'
           GROUP BY DATE(closed_at) ORDER BY d DESC LIMIT 30"""
    )

    strat_rows = _q(
        """SELECT strategy_name,
                  COUNT(*) AS total,
                  SUM(CASE WHEN CAST(realized_pnl AS REAL) > 0 THEN 1 ELSE 0 END) AS wins,
                  SUM(CASE WHEN CAST(realized_pnl AS REAL) > 0
                            THEN CAST(realized_pnl AS REAL) ELSE 0 END) AS gross_p,
                  ABS(SUM(CASE WHEN CAST(realized_pnl AS REAL) < 0
                               THEN CAST(realized_pnl AS REAL) ELSE 0 END)) AS gross_l,
                  SUM(CAST(realized_pnl AS REAL)) AS net,
                  AVG(CAST(slippage_bps AS REAL)) AS avg_slip
           FROM positions
           WHERE status='closed' AND realized_pnl IS NOT NULL AND realized_pnl != '0'
           GROUP BY strategy_name ORDER BY total DESC"""
    )
    liq_count = _q1("SELECT COUNT(*) AS c FROM positions WHERE close_reason='liquidation'")

    week_trades = week["t"] or 0 if week else 0
    week_wins = week["w"] or 0 if week else 0
    week_net = float(week["net"] or 0) if week else 0.0
    week_wr = (week_wins / week_trades * 100) if week_trades else 0.0
    liqs = liq_count["c"] if liq_count else 0

    net_cls = "ok" if week_net >= 0 else "danger"
    body = f"""
    <h3>This Week ({week_start}~)</h3>
    <table>
      <tr><th>Trades</th><td>{week_trades} ({week_wins}W / {week_trades - week_wins}L)</td></tr>
      <tr><th>Win Rate</th><td>{week_wr:.1f}%</td></tr>
      <tr><th>Net PnL</th><td class="{net_cls}">{week_net:+.2f} USDT</td></tr>
      <tr><th>Total Liquidations</th><td class="{"danger" if liqs>0 else "ok"}">{liqs}</td></tr>
    </table>
    <h3>Daily History (last 30 days)</h3>
    """
    if rows:
        header = "<tr><th>Date</th><th>Trades</th><th>Win%</th><th>Net PnL</th><th>Profit Factor</th></tr>"
        trs = []
        for r in rows:
            total = r["total"] or 0
            wins_d = r["wins"] or 0
            wr = (wins_d / total * 100) if total else 0.0
            net = float(r["net"] or 0)
            gross_p = float(r["gross_p"] or 0)
            gross_l = float(r["gross_l"] or 0)
            pf = gross_p / gross_l if gross_l > 0 else float("inf")
            pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
            cls = "ok" if net >= 0 else "danger"
            trs.append(
                f"<tr><td>{r['d']}</td><td>{total} ({wins_d}W/{total - wins_d}L)</td>"
                f"<td>{wr:.1f}%</td><td class='{cls}'>{net:+.2f}</td>"
                f"<td>{pf_str}</td></tr>"
            )
        body += f"<table>{header}{''.join(trs)}</table>"

    body += "<h3>Strategy Breakdown (all-time)</h3>"
    if strat_rows:
        _KELLY_MIN_TRADES = 6
        s_header = ("<tr><th>Strategy</th><th>Trades</th><th>Win%</th>"
                    "<th>Net PnL</th><th>PF</th><th>Avg Slip</th><th>Kelly</th></tr>")
        s_trs = []
        for r in strat_rows:
            t = r["total"] or 0
            w = r["wins"] or 0
            gp = float(r["gross_p"] or 0)
            gl = float(r["gross_l"] or 0)
            net = float(r["net"] or 0)
            wr = (w / t * 100) if t else 0.0
            pf = gp / gl if gl > 0 else float("inf")
            pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
            pf_cls = "ok" if pf >= 1.5 or pf == float("inf") else ("warn" if pf >= 1.0 else "danger")
            net_cls = "ok" if net >= 0 else "danger"
            slip = r["avg_slip"]
            slip_str = f"{float(slip):+.1f} bps" if slip is not None else "—"
            kelly_str = (f'<span class="ok">ACTIVE ({t})</span>'
                         if t >= _KELLY_MIN_TRADES
                         else f'<span class="warn">warming {t}/{_KELLY_MIN_TRADES}</span>')
            name = r["strategy_name"] or "unknown"
            s_trs.append(
                f"<tr><td>{name}</td><td>{t}</td>"
                f"<td>{wr:.1f}% ({w}W/{t-w}L)</td>"
                f"<td class='{net_cls}'>{net:+.2f}</td>"
                f"<td class='{pf_cls}'>{pf_str}</td>"
                f"<td>{slip_str}</td>"
                f"<td>{kelly_str}</td></tr>"
            )
        body += f"<table>{s_header}{''.join(s_trs)}</table>"
    else:
        body += "<p>No closed trades yet.</p>"

    return _page("Performance", body)


@app.route("/screener")
def screener():
    rows = _q(
        """SELECT symbol, base_asset, quote_asset, is_active, added_at, strategy
           FROM symbols ORDER BY is_active DESC, symbol ASC"""
    )
    if not rows:
        return _page("Screener", "<p>No symbols in DB.</p>")

    global_strategy = os.environ.get("ACTIVE_STRATEGY", "—")
    header = ("<tr><th>Symbol</th><th>Base</th><th>Quote</th>"
              "<th>Active</th><th>Strategy</th><th>Added</th></tr>")
    trs = []
    for r in rows:
        active_cls = "ok" if r["is_active"] else ""
        active_txt = "YES" if r["is_active"] else "no"
        strategy = r["strategy"] or global_strategy
        strat_cls = "warn" if r["strategy"] else ""  # highlight per-coin overrides
        trs.append(
            f"<tr><td>{r['symbol']}</td>"
            f"<td>{r['base_asset'] or '-'}</td>"
            f"<td>{r['quote_asset'] or '-'}</td>"
            f"<td class='{active_cls}'>{active_txt}</td>"
            f"<td class='{strat_cls}'>{strategy}</td>"
            f"<td>{r['added_at'] or '-'}</td></tr>"
        )
    body = (f"<p style='color:#aaa;font-size:0.85em'>Global strategy: "
            f"<b style='color:#4af'>{global_strategy}</b> — "
            f"per-coin overrides shown in <span class='warn'>yellow</span></p>"
            f"<table>{header}{''.join(trs)}</table>")
    return _page("Screener", body)


@app.route("/system")
def system():
    db_size_mb = 0.0
    try:
        db_size_mb = os.path.getsize(_DB_PATH) / 1024 / 1024
    except OSError:
        pass

    candle_freshness = _q(
        """SELECT symbol, interval_type, MAX(open_time) AS latest
           FROM klines GROUP BY symbol, interval_type ORDER BY symbol, latest DESC"""
    )
    # One row per symbol: keep the freshest interval row
    seen: set[str] = set()
    deduped = []
    for r in candle_freshness:
        if r["symbol"] not in seen:
            seen.add(r["symbol"])
            deduped.append(r)
    candle_freshness = deduped

    now_ms = datetime.now(timezone.utc).timestamp() * 1000
    body = f"""
    <h3>Database</h3>
    <table>
      <tr><th>Path</th><td>{_DB_PATH}</td></tr>
      <tr><th>Size</th><td>{db_size_mb:.2f} MB</td></tr>
    </table>
    <h3>Candle Freshness</h3>
    """
    if candle_freshness:
        header = "<tr><th>Symbol</th><th>Interval</th><th>Latest Candle</th><th>Age</th></tr>"
        trs = []
        for r in candle_freshness:
            latest_ms = int(r["latest"] or 0)
            age_s = (now_ms - latest_ms) / 1000
            age_cls = "ok" if age_s < 120 else ("warn" if age_s < 300 else "danger")
            age_str = f"{age_s:.0f}s"
            trs.append(
                f"<tr><td>{r['symbol']}</td><td>{r['interval_type']}</td>"
                f"<td>{latest_ms}</td>"
                f"<td class='{age_cls}'>{age_str}</td></tr>"
            )
        body += f"<table>{header}{''.join(trs)}</table>"
    else:
        body += "<p>No candle data.</p>"

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

    status = "ok" if db_ok else "degraded"
    return jsonify({"status": status, "db": db_ok}), (200 if db_ok else 503)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    port = int(os.environ.get("DASHBOARD_PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
