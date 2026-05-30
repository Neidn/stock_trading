"""Walk-forward screener quality validation.

Compares old (volatility-only) vs new (volatility + suitability) universe selection.

Method:
  - Monthly rebalancing over 1 year of 1h data for 20 liquid futures
  - At each rebalance: score all coins → select top 10 per method
  - Backtest each selected coin with its assigned strategy + current params for next month
  - Aggregate PF, net PnL, win rate across all periods

Usage:
  python -m src.backtest.screener_validation
  python -m src.backtest.screener_validation --years 2 --top-n 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.signal.indicators import calc_adx, calc_atr, calc_bollinger, calc_ema, calc_rsi
from src.signal.strategies.supertrend import _calc_supertrend
from src.backtest.tune_strategies import (
    TAKER_FEE, _open_position, _check_position, _close_eod, _metrics,
)
from src.jobs.screener import _discover_strategies

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── config ──────────────────────────────────────────────────────────────────

COINS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT",
    "POLUSDT", "SANDUSDT", "AAVEUSDT", "TRXUSDT", "FILUSDT",
]

LOOKBACK  = 230   # bars for indicator computation
FORWARD   = 720   # bars per evaluation period (~1 month of 1h)
REBALANCE = 720

DATA_DIR    = Path("data/backtest")
INITIAL_BAL = 100.0
RISK_PCT    = 0.01

LIVE_PARAMS: dict[str, dict] = {
    "ema_pullback_rsi": {
        "ema_fast": 20, "ema_mid": 50, "ema_slow": 200,
        "rsi_period": 14, "rsi_low": 45.0, "rsi_high": 55.0,
        "adx_period": 14, "adx_threshold": 30.0,
        "sl_atr_mult": 2.0, "tp1_atr_mult": 4.0, "tp2_atr_mult": 6.0,
    },
    "rsi_supertrend": {
        "atr_period": 10, "rsi_period": 14, "multiplier": 2.0,
        "rsi_threshold": 55.0,
        "sl_atr_mult": 2.5, "tp1_atr_mult": 4.0, "tp2_atr_mult": 6.0,
    },
    "bb_rsi_chartart": {
        "rsi_period": 6, "rsi_level": 50.0, "bb_period": 50, "bb_std": 2.0,
        "sl_atr_mult": 2.0, "tp1_atr_mult": 3.0, "tp2_atr_mult": 5.0,
    },
    "ema_crossover": {
        "ema_fast": 20, "ema_slow": 50, "adx_period": 14, "adx_threshold": 25.0,
        "sl_atr_mult": 2.0, "tp1_atr_mult": 4.0, "tp2_atr_mult": 6.0,
    },
    "supertrend": {
        "atr_period": 10, "multiplier": 4.0,
        "sl_atr_mult": 2.5, "tp1_atr_mult": 3.0, "tp2_atr_mult": 6.0,
    },
    "macd_sma200_chartart": {
        "fast_period": 12, "slow_period": 26, "signal_period": 9,
        "sl_atr_mult": 1.5, "tp1_atr_mult": 3.0, "tp2_atr_mult": 6.0,
    },
}

# ── data loading ─────────────────────────────────────────────────────────────

def _fetch_ohlcv(symbol: str, since_ms: int, until_ms: int) -> pd.DataFrame:
    import ccxt
    exchange = ccxt.binanceusdm({"options": {"defaultType": "future"}})
    all_rows: list = []
    current = since_ms
    while True:
        rows = exchange.fetch_ohlcv(symbol, "1h", since=current, limit=1000)
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        if last_ts >= until_ms or len(rows) < 1000:
            break
        current = last_ts + 1
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df[df["timestamp"] <= until_ms].reset_index(drop=True)


def load_coin(symbol: str, years: int = 1) -> pd.DataFrame | None:
    until_ms  = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms  = until_ms - years * 365 * 24 * 3600 * 1000
    cache     = DATA_DIR / f"{symbol}_1h_{since_ms}_{until_ms}_sv.csv"
    if cache.exists():
        return pd.read_csv(cache)
    logger.info("Fetching %s ...", symbol)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df = _fetch_ohlcv(symbol, since_ms, until_ms)
        if len(df) < LOOKBACK + FORWARD:
            logger.warning("%s: too few candles (%d), skipping", symbol, len(df))
            return None
        df.to_csv(cache, index=False)
        return df
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", symbol, exc)
        return None

# ── indicator computation from raw arrays ────────────────────────────────────

def _indicators(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> dict:
    atr_arr = calc_atr(high, low, close, 14)
    adx_arr = calc_adx(high, low, close, 14)
    cur_close = float(close[-1])
    cur_atr   = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
    cur_adx   = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 25.0
    atr_pct   = min((cur_atr / cur_close * 100) if cur_close > 0 else 0.0, 50.0)

    sma_aligned = sma50_slope = 0.0
    above_sma200 = False
    if len(close) >= 200:
        sma20  = float(np.mean(close[-20:]))
        sma50  = float(np.mean(close[-50:]))
        sma200 = float(np.mean(close[-200:]))
        above_sma200 = bool(cur_close > sma200)
        sma_aligned  = bool((sma20 > sma50 > sma200) or (sma20 < sma50 < sma200))
        if len(close) >= 70:
            sma50_prev   = float(np.mean(close[-70:-20]))
            sma50_slope  = (sma50 - sma50_prev) / sma50_prev if sma50_prev > 0 else 0.0

    adx_change = 0.0
    if len(adx_arr) >= 6 and not np.isnan(adx_arr[-6]):
        adx_change = cur_adx - float(adx_arr[-6])

    return {
        "adx": cur_adx, "atr_pct": atr_pct, "above_sma200": above_sma200,
        "sma_aligned": sma_aligned, "sma50_slope": sma50_slope, "adx_change": adx_change,
    }

# ── per-strategy simulations (fixed params, 50/50 partial close) ─────────────

def _sim_epr(df: pd.DataFrame, p: dict) -> dict:
    close = df["close"].to_numpy(float)
    high  = df["high"].to_numpy(float)
    low   = df["low"].to_numpy(float)
    open_ = df["open"].to_numpy(float)
    n     = len(close)

    ema_f = calc_ema(close, int(p["ema_fast"]))
    ema_m = calc_ema(close, int(p["ema_mid"]))
    ema_s = calc_ema(close, int(p["ema_slow"]))
    rsi   = calc_rsi(close, int(p["rsi_period"]))
    adx   = calc_adx(high, low, close, int(p["adx_period"]))
    atr   = calc_atr(high, low, close, int(p["adx_period"]))
    min_i = int(p["ema_slow"]) * 2 + 10

    balance, open_pos, trades = INITIAL_BAL, None, []
    for i in range(min_i, n):
        if open_pos is not None:
            open_pos, trade = _check_position(open_pos, high[i], low[i], i)
            if trade:
                balance += trade["pnl"]; trades.append(trade)
        if np.isnan(adx[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]) or atr[i] <= 0:
            continue
        if adx[i] < p["adx_threshold"]: continue
        bull = ema_f[i] > ema_m[i] > ema_s[i]
        bear = ema_f[i] < ema_m[i] < ema_s[i]
        rsi_ok = p["rsi_low"] <= rsi[i] <= p["rsi_high"]
        if not rsi_ok or open_pos is not None: continue
        side = "long" if (bull and close[i] > open_[i]) else "short" if (bear and close[i] < open_[i]) else None
        if side is None or i + 1 >= n: continue
        ep = open_[i + 1]; sl_d = atr[i] * p["sl_atr_mult"]
        qty = (balance * RISK_PCT) / sl_d
        if qty <= 0: continue
        fee = qty * ep * TAKER_FEE; balance -= fee
        sl  = ep - sl_d if side == "long" else ep + sl_d
        tp1 = ep + atr[i] * p["tp1_atr_mult"] if side == "long" else ep - atr[i] * p["tp1_atr_mult"]
        tp2 = ep + atr[i] * p["tp2_atr_mult"] if side == "long" else ep - atr[i] * p["tp2_atr_mult"]
        open_pos = _open_position(side, ep, sl, tp1, tp2, qty, fee, i + 1)
    if open_pos is not None:
        t = _close_eod(open_pos, close[-1], n); balance += t["pnl"]; trades.append(t)
    return _metrics(trades, INITIAL_BAL, balance)


def _sim_rsi_supertrend(df: pd.DataFrame, p: dict) -> dict:
    close = df["close"].to_numpy(float)
    high  = df["high"].to_numpy(float)
    low   = df["low"].to_numpy(float)
    open_ = df["open"].to_numpy(float)
    n     = len(close)
    period = int(p["atr_period"]); mult = float(p["multiplier"])
    up, dn, trend = _calc_supertrend(high, low, close, period, mult)
    rsi = calc_rsi(close, int(p["rsi_period"]))
    atr = calc_atr(high, low, close, period)
    min_i = period * 3 + int(p["rsi_period"]) * 2 + 10

    balance, open_pos, trades = INITIAL_BAL, None, []
    for i in range(min_i, n):
        if open_pos is not None:
            open_pos, trade = _check_position(open_pos, high[i], low[i], i)
            if trade:
                balance += trade["pnl"]; trades.append(trade)
        if np.isnan(rsi[i]) or np.isnan(atr[i]) or atr[i] <= 0: continue
        flip_up   = trend[i] == 1 and trend[i - 1] == -1
        flip_down = trend[i] == -1 and trend[i - 1] == 1
        if not flip_up and not flip_down: continue
        if open_pos is not None: continue
        side = "long" if flip_up else "short"
        rsi_ok = rsi[i] > p["rsi_threshold"] if side == "long" else rsi[i] < (100 - p["rsi_threshold"])
        if not rsi_ok or i + 1 >= n: continue
        ep = open_[i + 1]; sl_d = atr[i] * p["sl_atr_mult"]
        qty = (balance * RISK_PCT) / sl_d
        if qty <= 0: continue
        fee = qty * ep * TAKER_FEE; balance -= fee
        band_sl = float(up[i]) if side == "long" else float(dn[i])
        sl  = max(band_sl, ep - sl_d) if side == "long" else min(band_sl, ep + sl_d)
        tp1 = ep + atr[i] * p["tp1_atr_mult"] if side == "long" else ep - atr[i] * p["tp1_atr_mult"]
        tp2 = ep + atr[i] * p["tp2_atr_mult"] if side == "long" else ep - atr[i] * p["tp2_atr_mult"]
        open_pos = _open_position(side, ep, sl, tp1, tp2, qty, fee, i + 1)
    if open_pos is not None:
        t = _close_eod(open_pos, close[-1], n); balance += t["pnl"]; trades.append(t)
    return _metrics(trades, INITIAL_BAL, balance)


def _sim_bb_rsi(df: pd.DataFrame, p: dict) -> dict:
    close = df["close"].to_numpy(float)
    high  = df["high"].to_numpy(float)
    low   = df["low"].to_numpy(float)
    open_ = df["open"].to_numpy(float)
    n     = len(close)
    rsi = calc_rsi(close, int(p["rsi_period"]))
    upper, mid, lower = calc_bollinger(close, int(p["bb_period"]), float(p["bb_std"]))
    atr = calc_atr(high, low, close, 14)
    min_i = int(p["bb_period"]) + 10

    balance, open_pos, trades = INITIAL_BAL, None, []
    for i in range(min_i, n):
        if open_pos is not None:
            open_pos, trade = _check_position(open_pos, high[i], low[i], i)
            if trade:
                balance += trade["pnl"]; trades.append(trade)
        if np.isnan(rsi[i]) or np.isnan(atr[i]) or atr[i] <= 0: continue
        if open_pos is not None: continue
        rsi_cross_up   = rsi[i - 1] <= p["rsi_level"] and rsi[i] > p["rsi_level"]
        rsi_cross_down = rsi[i - 1] >= p["rsi_level"] and rsi[i] < p["rsi_level"]
        bb_cross_up    = close[i - 1] <= lower[i - 1] and close[i] > lower[i]
        bb_cross_down  = close[i - 1] >= upper[i - 1] and close[i] < upper[i]
        side = "long" if (rsi_cross_up and bb_cross_up) else "short" if (rsi_cross_down and bb_cross_down) else None
        if side is None or i + 1 >= n: continue
        ep = open_[i + 1]; sl_d = atr[i] * p["sl_atr_mult"]
        qty = (balance * RISK_PCT) / sl_d
        if qty <= 0: continue
        fee = qty * ep * TAKER_FEE; balance -= fee
        sl  = ep - sl_d if side == "long" else ep + sl_d
        tp1 = ep + atr[i] * p["tp1_atr_mult"] if side == "long" else ep - atr[i] * p["tp1_atr_mult"]
        tp2 = ep + atr[i] * p["tp2_atr_mult"] if side == "long" else ep - atr[i] * p["tp2_atr_mult"]
        open_pos = _open_position(side, ep, sl, tp1, tp2, qty, fee, i + 1)
    if open_pos is not None:
        t = _close_eod(open_pos, close[-1], n); balance += t["pnl"]; trades.append(t)
    return _metrics(trades, INITIAL_BAL, balance)


def _sim_ema_crossover(df: pd.DataFrame, p: dict) -> dict:
    close = df["close"].to_numpy(float)
    high  = df["high"].to_numpy(float)
    low   = df["low"].to_numpy(float)
    open_ = df["open"].to_numpy(float)
    n     = len(close)
    ema_f = calc_ema(close, int(p["ema_fast"]))
    ema_s = calc_ema(close, int(p["ema_slow"]))
    adx   = calc_adx(high, low, close, int(p["adx_period"]))
    atr   = calc_atr(high, low, close, int(p["adx_period"]))
    min_i = int(p["ema_slow"]) * 2 + 10

    balance, open_pos, trades = INITIAL_BAL, None, []
    for i in range(min_i, n):
        if open_pos is not None:
            open_pos, trade = _check_position(open_pos, high[i], low[i], i)
            if trade:
                balance += trade["pnl"]; trades.append(trade)
        if np.isnan(adx[i]) or np.isnan(atr[i]) or atr[i] <= 0: continue
        if adx[i] < p["adx_threshold"]: continue
        cross_up   = ema_f[i - 1] <= ema_s[i - 1] and ema_f[i] > ema_s[i]
        cross_down = ema_f[i - 1] >= ema_s[i - 1] and ema_f[i] < ema_s[i]
        if not cross_up and not cross_down: continue
        if i + 1 >= n: continue
        # Reversal close + new entry both execute at next bar open
        if open_pos is not None:
            new_side = "long" if cross_up else "short"
            if open_pos["side"] != new_side:
                t = _close_eod(open_pos, open_[i + 1], i + 1); balance += t["pnl"]
                trades.append({**t, "reason": "reversal"}); open_pos = None
        if open_pos is not None: continue
        side = "long" if cross_up else "short"
        ep = open_[i + 1]; sl_d = atr[i] * p["sl_atr_mult"]
        qty = (balance * RISK_PCT) / sl_d
        if qty <= 0: continue
        fee = qty * ep * TAKER_FEE; balance -= fee
        sl  = ep - sl_d if side == "long" else ep + sl_d
        tp1 = ep + atr[i] * p["tp1_atr_mult"] if side == "long" else ep - atr[i] * p["tp1_atr_mult"]
        tp2 = ep + atr[i] * p["tp2_atr_mult"] if side == "long" else ep - atr[i] * p["tp2_atr_mult"]
        open_pos = _open_position(side, ep, sl, tp1, tp2, qty, fee, i + 1)
    if open_pos is not None:
        t = _close_eod(open_pos, close[-1], n); balance += t["pnl"]; trades.append(t)
    return _metrics(trades, INITIAL_BAL, balance)


def _simulate(strategy: str, df: pd.DataFrame) -> dict:
    if len(df) < 50:
        return {"n": 0, "pf": 0.0, "wr": 0.0, "net_pnl": 0.0}
    if strategy in ("bb_rsi_chartart", "bb_breakout"):
        return _sim_bb_rsi(df, LIVE_PARAMS["bb_rsi_chartart"])
    p = LIVE_PARAMS.get(strategy, LIVE_PARAMS["ema_pullback_rsi"])
    if strategy == "ema_pullback_rsi":
        return _sim_epr(df, p)
    if strategy == "rsi_supertrend":
        return _sim_rsi_supertrend(df, p)
    if strategy == "ema_crossover":
        return _sim_ema_crossover(df, p)
    # supertrend + macd_sma200 fall back to EPR
    return _sim_epr(df, LIVE_PARAMS["ema_pullback_rsi"])

# ── scoring ───────────────────────────────────────────────────────────────────

def _old_score(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> float:
    """Volatility-only proxy: ATR% (mimics old screener's primary signal)."""
    atr = calc_atr(high, low, close, 14)
    cur = float(close[-1])
    if cur <= 0 or np.isnan(atr[-1]):
        return 0.0
    return float(atr[-1]) / cur * 100


def _new_score(close: np.ndarray, high: np.ndarray, low: np.ndarray,
               strategies: list) -> float:
    vol = _old_score(close, high, low)
    ind = _indicators(close, high, low)
    suit = max(cls.suitability_score(ind) for cls in strategies)
    return vol + suit * 1.5

# ── walk-forward ──────────────────────────────────────────────────────────────

def _agg(results: list[dict]) -> dict:
    valid = [r for r in results if r["n"] > 0]
    if not valid:
        return {"periods": 0, "avg_pf": 0.0, "avg_wr": 0.0, "total_pnl": 0.0, "total_trades": 0}
    finite_pf = [r["pf"] for r in valid if r["pf"] != float("inf")]
    avg_pf = sum(finite_pf) / len(finite_pf) if finite_pf else float("inf")
    return {
        "periods":      len(results),
        "avg_pf":       avg_pf,
        "avg_wr":       sum(r["wr"] for r in valid) / len(valid) * 100,
        "total_pnl":    sum(r["net_pnl"] for r in valid),
        "total_trades": sum(r["n"] for r in valid),
    }


def run(top_n: int = 10, years: int = 1) -> None:
    logger.info("Loading data for %d coins (%d year(s))...", len(COINS), years)
    data: dict[str, pd.DataFrame] = {}
    for coin in COINS:
        df = load_coin(coin, years)
        if df is not None:
            data[coin] = df
    logger.info("Loaded %d coins", len(data))

    if len(data) < top_n:
        logger.error("Not enough coins loaded (%d < %d)", len(data), top_n)
        sys.exit(1)

    strategies = _discover_strategies()
    min_len = min(len(df) for df in data.values())
    rebalance_points = list(range(LOOKBACK, min_len - FORWARD, REBALANCE))
    logger.info("%d rebalance periods", len(rebalance_points))

    old_results: list[dict] = []
    new_results: list[dict] = []
    period_rows: list[dict] = []

    for period_idx, t in enumerate(rebalance_points):
        scores_old: dict[str, float] = {}
        scores_new: dict[str, float] = {}
        assignments: dict[str, str]  = {}

        for coin, df in data.items():
            sl = df.iloc[t - LOOKBACK:t]
            c = sl["close"].to_numpy(float)
            h = sl["high"].to_numpy(float)
            l = sl["low"].to_numpy(float)
            scores_old[coin] = _old_score(c, h, l)
            scores_new[coin] = _new_score(c, h, l, strategies)
            ind  = _indicators(c, h, l)
            best = max(strategies, key=lambda cls: cls.suitability_score(ind))
            assignments[coin] = best({}).get_name()

        old_universe = sorted(scores_old, key=scores_old.__getitem__, reverse=True)[:top_n]
        new_universe = sorted(scores_new, key=scores_new.__getitem__, reverse=True)[:top_n]
        overlap = len(set(old_universe) & set(new_universe))

        for coin in old_universe:
            fwd = data[coin].iloc[t:t + FORWARD]
            old_results.append(_simulate(assignments[coin], fwd))

        for coin in new_universe:
            fwd = data[coin].iloc[t:t + FORWARD]
            new_results.append(_simulate(assignments[coin], fwd))

        period_rows.append({
            "period":   period_idx + 1,
            "overlap":  overlap,
            "old_only": sorted(set(old_universe) - set(new_universe)),
            "new_only": sorted(set(new_universe) - set(old_universe)),
        })
        logger.info(
            "Period %d: overlap=%d/%d | old_only=%s | new_only=%s",
            period_idx + 1, overlap, top_n,
            period_rows[-1]["old_only"], period_rows[-1]["new_only"],
        )

    # ── results ──────────────────────────────────────────────────────────────
    old_agg = _agg(old_results)
    new_agg = _agg(new_results)

    print("\n" + "=" * 60)
    print("SCREENER VALIDATION — vol-only vs vol+suitability")
    print("=" * 60)
    print(f"Coins: {len(data)}  |  Top-N: {top_n}  |  Periods: {len(rebalance_points)}")
    print(f"Lookback: {LOOKBACK}h  |  Forward: {FORWARD}h (~{FORWARD//720}mo)")
    print()
    print(f"{'Metric':<20} {'OLD (vol-only)':>18} {'NEW (vol+suit)':>18} {'Delta':>10}")
    print("-" * 68)
    for key, label, fmt in [
        ("avg_pf",       "Avg PF",          "{:.3f}"),
        ("avg_wr",       "Avg Win Rate %",  "{:.1f}%"),
        ("total_pnl",    "Total PnL",       "{:.2f}"),
        ("total_trades", "Total Trades",    "{:d}"),
    ]:
        ov = old_agg[key]
        nv = new_agg[key]
        delta = nv - ov
        sign  = "+" if delta >= 0 else ""
        print(f"{label:<20} {fmt.format(ov):>18} {fmt.format(nv):>18} {sign}{fmt.format(delta):>9}")

    print()
    print("Period breakdown:")
    for row in period_rows:
        print(f"  Period {row['period']:2d}: overlap={row['overlap']}/{top_n}"
              f"  new_only={row['new_only']}  dropped={row['old_only']}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--years", type=int, default=1)
    args = parser.parse_args()
    run(top_n=args.top_n, years=args.years)
