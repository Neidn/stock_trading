"""Monthly backtest report — all active strategies vs their best params.

For each strategy, fetches last 12 months of OHLCV for the reference symbol,
runs a full grid search, simulates current live params, and sends a combined
comparison report via Telegram. Human decides whether to apply changes.

Strategy → Reference Symbol (regime-matched):
    ema_pullback_rsi     → BTCUSDT  (strong trend, ADX ~38)
    ema_crossover        → ETHUSDT  (trending, diverse)
    macd_sma200_chartart → SOLUSDT  (strong sustained trend, ADX ~55)
    rsi_supertrend       → BNBUSDT  (transitioning/ranging, ADX ~28)
    supertrend           → SOLUSDT  (volatile + trending, ADX ~35)

Run:
    python -m src.jobs.backtest_report
"""

from __future__ import annotations

import html
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd

from src.backtest.tune import GRID as EPR_GRID, _precompute, _simulate, run_grid as epr_run_grid
from src.backtest.tune_strategies import (
    find_params_in_results,
    run_grid as st_run_grid,
    _STRATEGY_PARAM_KEYS,
)
from src.monitoring.telegram_bot import get_telegram_bot

logger = logging.getLogger(__name__)

LOOKBACK_DAYS       = 365
MIN_TRADES          = 30
PF_CHANGE_THRESHOLD = 0.10
DRIFT_THRESHOLD     = 0.20   # live PF below backtest PF by this much → drift alert
MIN_LIVE_TRADES     = 10     # min live closed trades to compute drift
INITIAL_BALANCE     = 100.0
RISK_PCT            = 0.01

# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------

STRATEGY_CONFIGS = [
    {
        "name":       "ema_pullback_rsi",
        "symbol":     "BTCUSDT",
        "env_key":    "STRATEGY_PARAMS_EMA_PULLBACK_RSI",
        "param_keys": ["adx_threshold", "rsi_low", "rsi_high", "sl_atr_mult", "tp1_atr_mult", "tp2_atr_mult"],
        "defaults":   {
            "adx_threshold": 30.0, "rsi_low": 45.0, "rsi_high": 55.0,
            "sl_atr_mult": 2.0, "tp1_atr_mult": 4.0, "tp2_atr_mult": 6.0,
        },
        "engine": "epr",   # uses tune.py
    },
    {
        "name":       "ema_crossover",
        "symbol":     "ETHUSDT",
        "env_key":    "STRATEGY_PARAMS_EMA_CROSSOVER",
        "param_keys": _STRATEGY_PARAM_KEYS["ema_crossover"],
        "defaults":   {"adx_threshold": 30.0, "sl_atr_mult": 2.0, "tp1_atr_mult": 2.5, "tp2_atr_mult": 6.0},
        "engine": "multi",
    },
    {
        "name":       "macd_sma200_chartart",
        "symbol":     "SOLUSDT",
        "env_key":    "STRATEGY_PARAMS_MACD_SMA200_CHARTART",
        "param_keys": _STRATEGY_PARAM_KEYS["macd_sma200_chartart"],
        "defaults":   {"sl_atr_mult": 1.5, "tp1_atr_mult": 3.0, "tp2_atr_mult": 6.0},
        "engine": "multi",
    },
    {
        "name":       "rsi_supertrend",
        "symbol":     "BNBUSDT",
        "env_key":    "STRATEGY_PARAMS_RSI_SUPERTREND",
        "param_keys": _STRATEGY_PARAM_KEYS["rsi_supertrend"],
        "defaults":   {"multiplier": 2.0, "rsi_threshold": 55.0, "sl_atr_mult": 2.5, "tp1_atr_mult": 4.0, "tp2_atr_mult": 6.0},
        "engine": "multi",
    },
    {
        "name":       "supertrend",
        "symbol":     "SOLUSDT",
        "env_key":    "STRATEGY_PARAMS_SUPERTREND",
        "param_keys": _STRATEGY_PARAM_KEYS["supertrend"],
        "defaults":   {"multiplier": 4.0, "sl_atr_mult": 2.5, "tp1_atr_mult": 3.0, "tp2_atr_mult": 6.0},
        "engine": "multi",
    },
]


# ---------------------------------------------------------------------------
# OHLCV fetch
# ---------------------------------------------------------------------------

def _fetch_ohlcv(symbol: str, since_ms: int, until_ms: int) -> pd.DataFrame:
    exchange = ccxt.binanceusdm({"options": {"defaultType": "future"}})
    all_rows: list = []
    limit = 1000
    current = since_ms
    while True:
        rows = exchange.fetch_ohlcv(symbol, "1h", since=current, limit=limit)
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        if last_ts >= until_ms or len(rows) < limit:
            break
        current = last_ts + 1
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df[df["timestamp"] <= until_ms].copy()
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Param loading
# ---------------------------------------------------------------------------

def _load_params(env_key: str, defaults: dict) -> dict:
    raw = os.environ.get(env_key, "").strip()
    overrides = {}
    if raw:
        try:
            overrides = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse %s", env_key)
    return {**defaults, **overrides}


# ---------------------------------------------------------------------------
# Per-strategy grid + current simulation
# ---------------------------------------------------------------------------

def _run_epr(df: pd.DataFrame, current_params: dict) -> tuple[dict, dict]:
    """Run ema_pullback_rsi grid. Returns (best_result, current_result)."""
    pre = _precompute(df)

    current_result = _simulate(
        pre,
        adx_threshold=current_params["adx_threshold"],
        rsi_low=current_params["rsi_low"],
        rsi_high=current_params["rsi_high"],
        sl_atr_mult=current_params["sl_atr_mult"],
        tp1_atr_mult=current_params["tp1_atr_mult"],
        tp2_atr_mult=current_params["tp2_atr_mult"],
        initial_balance=INITIAL_BALANCE,
        risk_pct=RISK_PCT,
    )

    top = epr_run_grid(df, initial_balance=INITIAL_BALANCE, risk_pct=RISK_PCT, top_n=1)
    best_result = top[0] if top else current_result
    return best_result, current_result


def _run_multi(cfg: dict, df: pd.DataFrame, current_params: dict) -> tuple[dict, dict]:
    """Run grid for non-epr strategies. Returns (best_result, current_result)."""
    all_results = st_run_grid(
        cfg["name"], df,
        initial_balance=INITIAL_BALANCE,
        risk_pct=RISK_PCT,
        top_n=9999,
    )
    best_result   = all_results[0] if all_results else None
    current_result = find_params_in_results(all_results, current_params, cfg["param_keys"])

    empty = {"n": 0, "pf": 0.0, "wr": 0.0, "net_pnl": 0.0, "max_dd": 0.0}
    return best_result or empty, current_result or empty


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _pf_str(r: dict) -> str:
    pf = r.get("pf", 0.0)
    return f"{pf:.2f}" if pf != float("inf") else "∞"


def _format_strategy_section(
    cfg: dict,
    period_start: str,
    period_end: str,
    current_params: dict,
    current_result: dict,
    best_params: dict,
    best_result: dict,
) -> str:
    pf_diff = best_result.get("pf", 0.0) - current_result.get("pf", 0.0)
    recommend = pf_diff >= PF_CHANGE_THRESHOLD and best_result.get("n", 0) >= MIN_TRADES

    param_keys = cfg["param_keys"]
    cur_param_str  = "  ".join(f"{k.split('_')[0][:4]}={current_params.get(k, '?')}" for k in param_keys)
    best_param_str = "  ".join(f"{k.split('_')[0][:4]}={best_params.get(k, '?')}" for k in param_keys)

    lines = [
        f"<b>▶ {cfg['name']}</b> ({cfg['symbol']})",
        f"현재: {cur_param_str}",
        f"거래:{current_result.get('n',0)}건  승률:{current_result.get('wr',0):.1%}  PF:{_pf_str(current_result)}  낙폭:{current_result.get('max_dd',0):.1f}%",
        f"최적: {best_param_str}",
        f"거래:{best_result.get('n',0)}건  승률:{best_result.get('wr',0):.1%}  PF:{_pf_str(best_result)}  낙폭:{best_result.get('max_dd',0):.1f}%",
    ]

    if recommend:
        env_val = json.dumps({k: best_params[k] for k in param_keys}, separators=(",", ":"))
        lines.append(f"⚠️ 변경 권장 (+{pf_diff:.2f}) → {cfg['env_key']}='{env_val}'")
    else:
        lines.append(f"✅ 유지 권장 (PF차이 {pf_diff:+.2f})")

    return "\n".join(lines)


def _format_full_report(sections: list[str], period_start: str, period_end: str) -> str:
    header = [
        "📊 <b>월간 백테스트 리포트</b>",
        f"기간: {period_start} ~ {period_end} (12개월)",
        f"전략 수: {len(sections)}개",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    return "\n".join(header + [""] + ["\n".join([s, "━━━━━━━━━━━━━━━━━━━━"]) for s in sections])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> str:
    now = datetime.now(timezone.utc)
    until_ms = int(now.timestamp() * 1000)
    since_ms = int((now - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)
    period_start = (now - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    period_end   = now.strftime("%Y-%m-%d")

    # Cache OHLCV per symbol (avoid re-fetching same symbol for multiple strategies)
    df_cache: dict[str, pd.DataFrame] = {}
    symbols_needed = list({cfg["symbol"] for cfg in STRATEGY_CONFIGS})
    for sym in symbols_needed:
        logger.info("Fetching %s 1h OHLCV (%s → %s)...", sym, period_start, period_end)
        df_cache[sym] = _fetch_ohlcv(sym, since_ms, until_ms)
        logger.info("  %s: %d candles", sym, len(df_cache[sym]))

    sections: list[str] = []
    for cfg in STRATEGY_CONFIGS:
        name = cfg["name"]
        logger.info("Running grid: %s (%s)...", name, cfg["symbol"])
        df = df_cache[cfg["symbol"]]
        current_params = _load_params(cfg["env_key"], cfg["defaults"])

        try:
            if cfg["engine"] == "epr":
                best_result, current_result = _run_epr(df, current_params)
                best_params = {k: best_result.get(k, current_params[k]) for k in cfg["param_keys"]}
            else:
                best_result, current_result = _run_multi(cfg, df, current_params)
                best_params = {k: best_result.get(k, current_params[k]) for k in cfg["param_keys"]}
        except Exception as exc:
            logger.warning("Grid failed for %s: %s", name, exc)
            sections.append(f"<b>▶ {name}</b> — ❌ 오류: {html.escape(str(exc))}")
            continue

        section = _format_strategy_section(
            cfg, period_start, period_end,
            current_params, current_result,
            best_params, best_result,
        )
        sections.append(section)
        logger.info("  %s done: current PF=%.2f  best PF=%.2f",
                    name, current_result.get("pf", 0), best_result.get("pf", 0))

    report = _format_full_report(sections, period_start, period_end)

    logger.info("Sending report to Telegram...")
    try:
        telegram = get_telegram_bot()
        telegram.send_info(report)
        logger.info("Telegram report sent.")
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)

    return report


# ---------------------------------------------------------------------------
# Live PF loader (for drift check)
# ---------------------------------------------------------------------------

def _load_live_pf(db_path: str, strategy_name: str, since_iso: str) -> tuple[float, int]:
    """Query closed positions for symbols currently assigned to strategy_name.

    Returns (profit_factor, trade_count). PF=0.0 when no trades or all losses.
    """
    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            """
            SELECT CAST(p.realized_pnl AS REAL)
            FROM positions p
            JOIN symbols s ON p.symbol = s.symbol
            WHERE s.strategy = ?
              AND p.status = 'closed'
              AND p.closed_at >= ?
              AND p.realized_pnl IS NOT NULL
              AND p.realized_pnl != ''
              AND p.realized_pnl != '0.0'
            """,
            (strategy_name, since_iso),
        ).fetchall()
        conn.close()
    except Exception as exc:
        logger.warning("DB query failed for %s: %s", strategy_name, exc)
        return 0.0, 0

    pnls = []
    for (raw,) in rows:
        try:
            pnls.append(float(raw))
        except (TypeError, ValueError):
            pass

    n = len(pnls)
    if n == 0:
        return 0.0, 0

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss   = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return pf, n


# ---------------------------------------------------------------------------
# Weekly checker
# ---------------------------------------------------------------------------

def check() -> list[dict]:
    """Weekly param + drift checker. Sends Telegram only when action needed.

    Returns list of triggered alerts (empty = all good, no message sent).
    """
    now       = datetime.now(timezone.utc)
    until_ms  = int(now.timestamp() * 1000)
    since_dt  = now - timedelta(days=LOOKBACK_DAYS)
    since_ms  = int(since_dt.timestamp() * 1000)
    since_iso = since_dt.isoformat()
    db_path   = os.environ.get("SQLITE_DB_PATH", "/data/trading.db")

    df_cache: dict[str, pd.DataFrame] = {}
    for cfg in STRATEGY_CONFIGS:
        sym = cfg["symbol"]
        if sym not in df_cache:
            logger.info("Fetching %s for check...", sym)
            df_cache[sym] = _fetch_ohlcv(sym, since_ms, until_ms)

    alerts: list[dict] = []

    for cfg in STRATEGY_CONFIGS:
        name           = cfg["name"]
        current_params = _load_params(cfg["env_key"], cfg["defaults"])
        df             = df_cache[cfg["symbol"]]

        # --- rank check (grid) ---
        try:
            if cfg["engine"] == "epr":
                best_result, current_result = _run_epr(df, current_params)
                best_params = {k: best_result.get(k, current_params[k]) for k in cfg["param_keys"]}
            else:
                best_result, current_result = _run_multi(cfg, df, current_params)
                best_params = {k: best_result.get(k, current_params[k]) for k in cfg["param_keys"]}
        except Exception as exc:
            logger.warning("Grid failed for %s: %s", name, exc)
            continue

        backtest_pf = current_result.get("pf", 0.0)
        best_pf     = best_result.get("pf", 0.0)
        pf_diff     = best_pf - backtest_pf
        rank_alert  = pf_diff >= PF_CHANGE_THRESHOLD and best_result.get("n", 0) >= MIN_TRADES

        # --- drift check (live DB) ---
        live_pf, live_n = _load_live_pf(db_path, name, since_iso)
        drift       = backtest_pf - live_pf   # positive = live underperforming backtest
        drift_alert = (
            live_n >= MIN_LIVE_TRADES
            and backtest_pf > 1.0            # only meaningful if backtest expected profitable
            and drift >= DRIFT_THRESHOLD
        )

        logger.info(
            "%s: backtest_pf=%.2f best_pf=%.2f diff=%.2f | live_pf=%.2f live_n=%d drift=%.2f",
            name, backtest_pf, best_pf, pf_diff, live_pf, live_n, drift,
        )

        if rank_alert or drift_alert:
            alerts.append({
                "name":           name,
                "rank_alert":     rank_alert,
                "drift_alert":    drift_alert,
                "backtest_pf":    backtest_pf,
                "best_pf":        best_pf,
                "pf_diff":        pf_diff,
                "live_pf":        live_pf,
                "live_n":         live_n,
                "drift":          drift,
                "best_params":    best_params,
                "current_params": current_params,
                "param_keys":     cfg["param_keys"],
                "env_key":        cfg["env_key"],
            })

    if not alerts:
        logger.info("check(): all strategies OK — no Telegram sent")
        return []

    # Build alert message
    lines = [
        "⚠️ <b>전략 파라미터 점검 알림</b>",
        f"({now.strftime('%Y-%m-%d')} 주간 체크)",
        "",
    ]
    for a in alerts:
        lines.append(f"<b>▶ {a['name']}</b>")
        if a["rank_alert"]:
            env_val = json.dumps(
                {k: a["best_params"][k] for k in a["param_keys"]},
                separators=(",", ":"),
            )
            lines.append(
                f"  📈 파라미터 교체 권장: 현재 PF {a['backtest_pf']:.2f} → 최적 PF {a['best_pf']:.2f} (+{a['pf_diff']:.2f})"
            )
            lines.append(f"  {a['env_key']}='{env_val}'")
        if a["drift_alert"]:
            lines.append(
                f"  🔻 라이브 드리프트: 백테스트 PF {a['backtest_pf']:.2f} vs 실거래 PF {a['live_pf']:.2f}"
                f" (차이 {a['drift']:.2f}, {a['live_n']}건)"
            )
            lines.append("  → 시장 레짐 변화 가능. 전략 교체 검토 권장.")
        lines.append("")

    msg = "\n".join(lines)
    logger.info("check(): %d alert(s) — sending Telegram", len(alerts))
    try:
        telegram = get_telegram_bot()
        telegram.send_warning(msg)
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)

    return alerts


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s - %(message)s")
    report = run()
    print(report)


def check_main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s - %(message)s")
    alerts = check()
    if alerts:
        print(f"{len(alerts)} alert(s) sent.")
        for a in alerts:
            print(f"  {a['name']}: rank={a['rank_alert']} drift={a['drift_alert']}")
    else:
        print("All strategies OK.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_main()
    else:
        main()
