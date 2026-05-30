"""Fast parameter grid search for four active strategies.

Precomputes strategy-independent indicators once, then replays each
parameter combination in O(n) per combo.

Strategies supported:
    ema_crossover       — EMA crossover + ADX filter
    macd_sma200_chartart — SMA-MACD + SMA200 trend filter
    rsi_supertrend      — SuperTrend flip + RSI confirmation
    supertrend          — SuperTrend flip only

All strategies use 50% close at TP1, remaining 50% at TP2 or SL
(matching the live order_manager behavior).

Usage:
    python -m src.backtest.tune_strategies \\
        --strategy ema_crossover \\
        --symbol BTCUSDT \\
        --start 2020-01-01 \\
        --end 2024-12-31 \\
        [--balance 100] [--risk-pct 0.01] [--top 15] [--no-cache]
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.signal.indicators import calc_atr, calc_adx, calc_ema, calc_rsi, calc_sma

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "backtest"
TAKER_FEE = 0.0004


# ---------------------------------------------------------------------------
# OHLCV fetch / cache (identical to tune.py)
# ---------------------------------------------------------------------------

def _load_ohlcv(symbol: str, since_ms: int, until_ms: int, no_cache: bool = False) -> pd.DataFrame:
    cache_file = DATA_DIR / f"{symbol}_1h_{since_ms}_{until_ms}.csv"
    if not no_cache and cache_file.exists():
        logger.info("Loading from cache: %s", cache_file.name)
        return pd.read_csv(cache_file)

    import ccxt
    exchange = ccxt.binanceusdm({"options": {"defaultType": "future"}})
    logger.info("Fetching %s 1h from Binance...", symbol)
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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    logger.info("Fetched %d candles", len(df))
    return df


# ---------------------------------------------------------------------------
# Position management — shared 50/50 partial-close logic
# ---------------------------------------------------------------------------

def _open_position(side, entry_price, sl, tp1, tp2, qty, entry_fee, entry_bar):
    return {
        "side": side, "entry_price": entry_price,
        "sl": sl, "tp1": tp1, "tp2": tp2, "qty": qty,
        "entry_fee": entry_fee, "entry_bar": entry_bar,
        "half_closed": False, "partial_pnl": 0.0,
    }


def _check_position(pos, h, l, i):
    """Return (new_pos, trade_or_None) after checking SL/TP on one bar."""
    side     = pos["side"]
    sl       = pos["sl"]
    tp1      = pos["tp1"]
    tp2      = pos["tp2"]
    half_qty = pos["qty"] / 2.0
    half_fee = pos["entry_fee"] / 2.0

    sl_hit  = (side == "long" and l <= sl)  or (side == "short" and h >= sl)
    tp1_hit = (side == "long" and h >= tp1) or (side == "short" and l <= tp1)
    tp2_hit = (side == "long" and h >= tp2) or (side == "short" and l <= tp2)

    if not pos["half_closed"]:
        if sl_hit:
            pnl_raw  = (sl - pos["entry_price"]) * pos["qty"] if side == "long" \
                       else (pos["entry_price"] - sl) * pos["qty"]
            exit_fee = pos["qty"] * sl * TAKER_FEE
            realized = pnl_raw - pos["entry_fee"] - exit_fee
            return None, {"pnl": realized, "reason": "sl", "bars": i - pos["entry_bar"]}
        if tp1_hit:
            pnl_raw  = (tp1 - pos["entry_price"]) * half_qty if side == "long" \
                       else (pos["entry_price"] - tp1) * half_qty
            exit_fee = half_qty * tp1 * TAKER_FEE
            partial  = pnl_raw - half_fee - exit_fee
            pos = {**pos, "half_closed": True, "partial_pnl": partial}
            return pos, None
    else:
        if sl_hit or tp2_hit:
            exit_price = sl if sl_hit else tp2
            reason     = "sl_after_tp1" if sl_hit else "tp2"
            pnl_raw    = (exit_price - pos["entry_price"]) * half_qty if side == "long" \
                         else (pos["entry_price"] - exit_price) * half_qty
            exit_fee   = half_qty * exit_price * TAKER_FEE
            second     = pnl_raw - half_fee - exit_fee
            total_pnl  = pos["partial_pnl"] + second
            return None, {"pnl": total_pnl, "reason": reason, "bars": i - pos["entry_bar"]}

    return pos, None


def _close_eod(pos, exit_price, n):
    """Force-close remaining position at end of data."""
    side = pos["side"]
    if pos["half_closed"]:
        remaining = pos["qty"] / 2.0
        half_fee  = pos["entry_fee"] / 2.0
        pnl_raw   = (exit_price - pos["entry_price"]) * remaining if side == "long" \
                    else (pos["entry_price"] - exit_price) * remaining
        exit_fee  = remaining * exit_price * TAKER_FEE
        realized  = pos["partial_pnl"] + pnl_raw - half_fee - exit_fee
    else:
        pnl_raw  = (exit_price - pos["entry_price"]) * pos["qty"] if side == "long" \
                   else (pos["entry_price"] - exit_price) * pos["qty"]
        exit_fee = pos["qty"] * exit_price * TAKER_FEE
        realized = pnl_raw - pos["entry_fee"] - exit_fee
    return {"pnl": realized, "reason": "end_of_data", "bars": n - 1 - pos["entry_bar"]}


def _metrics(trades: list, initial_balance: float, final_balance: float) -> dict:
    total = len(trades)
    if total == 0:
        return {"n": 0, "pf": 0.0, "wr": 0.0, "net_pnl": 0.0, "max_dd": 0.0, "avg_bars": 0.0, "final_bal": final_balance}
    wins   = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] < 0]
    gross_p = sum(wins)
    gross_l = abs(sum(losses))
    net_pnl = sum(t["pnl"] for t in trades)
    pf = gross_p / gross_l if gross_l > 0 else float("inf")

    eq = initial_balance
    peak = eq
    max_dd = 0.0
    for t in trades:
        eq += t["pnl"]
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    return {
        "n": total, "pf": pf,
        "wr": len(wins) / total,
        "net_pnl": net_pnl,
        "max_dd": max_dd,
        "avg_bars": sum(t["bars"] for t in trades) / total,
        "final_bal": final_balance,
    }


# ---------------------------------------------------------------------------
# EMA Crossover
# ---------------------------------------------------------------------------

_EMA_CROSS_FIXED = {"ema_fast": 20, "ema_slow": 50, "adx_period": 14}
_EMA_CROSS_GRID = {
    "adx_threshold": [20.0, 25.0, 30.0],
    "sl_atr_mult":   [1.5, 2.0, 2.5],
    "tp1_atr_mult":  [2.5, 3.0, 4.0],
    "tp2_atr_mult":  [4.0, 5.0, 6.0],
}


def _run_ema_crossover(df: pd.DataFrame, initial_balance: float, risk_pct: float, top_n: int) -> list[dict]:
    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    n     = len(close)

    f = _EMA_CROSS_FIXED
    ema_fast = calc_ema(close, f["ema_fast"])
    ema_slow = calc_ema(close, f["ema_slow"])
    adx_arr  = calc_adx(high, low, close, f["adx_period"])
    atr_arr  = calc_atr(high, low, close, f["adx_period"])

    min_i = f["ema_slow"] * 2 + 10
    keys  = list(_EMA_CROSS_GRID.keys())
    combos = list(product(*_EMA_CROSS_GRID.values()))
    print(f"  ema_crossover: {len(combos)} combos...")

    results = []
    for combo in combos:
        p = dict(zip(keys, combo))
        if p["tp1_atr_mult"] <= p["sl_atr_mult"]:
            continue
        if p["tp2_atr_mult"] <= p["tp1_atr_mult"]:
            continue

        balance = initial_balance
        open_pos = None
        trades: list = []

        for i in range(min_i, n):
            h, l = high[i], low[i]

            # Check SL/TP
            if open_pos is not None:
                open_pos, trade = _check_position(open_pos, h, l, i)
                if trade is not None:
                    balance += trade["pnl"]
                    trades.append(trade)

            # Signal: EMA crossover
            cur_adx = adx_arr[i]
            cur_atr = atr_arr[i]
            if np.isnan(cur_adx) or np.isnan(cur_atr) or cur_atr <= 0:
                continue
            if cur_adx < p["adx_threshold"]:
                continue

            cross_up   = ema_fast[i - 1] <= ema_slow[i - 1] and ema_fast[i] > ema_slow[i]
            cross_down = ema_fast[i - 1] >= ema_slow[i - 1] and ema_fast[i] < ema_slow[i]

            if not cross_up and not cross_down:
                continue
            if i + 1 >= n:
                continue

            signal_side = "long" if cross_up else "short"

            # Reversal close
            if open_pos is not None and open_pos["side"] != signal_side:
                rev = open_[i + 1]
                trade = _close_eod(open_pos, rev, i + 1)
                balance += trade["pnl"]
                trades.append({**trade, "reason": "reversal"})
                open_pos = None

            if open_pos is None:
                ep = open_[i + 1]
                sl_dist = cur_atr * p["sl_atr_mult"]
                qty = (balance * risk_pct) / sl_dist
                if qty <= 0:
                    continue
                entry_fee = qty * ep * TAKER_FEE
                balance -= entry_fee
                if signal_side == "long":
                    sl  = ep - sl_dist
                    tp1 = ep + cur_atr * p["tp1_atr_mult"]
                    tp2 = ep + cur_atr * p["tp2_atr_mult"]
                else:
                    sl  = ep + sl_dist
                    tp1 = ep - cur_atr * p["tp1_atr_mult"]
                    tp2 = ep - cur_atr * p["tp2_atr_mult"]
                open_pos = _open_position(signal_side, ep, sl, tp1, tp2, qty, entry_fee, i + 1)

        if open_pos is not None:
            trade = _close_eod(open_pos, close[-1], n)
            balance += trade["pnl"]
            trades.append(trade)

        r = _metrics(trades, initial_balance, balance)
        r.update(p)
        results.append(r)

    results.sort(key=lambda x: x["pf"] if x["n"] >= 30 else -1, reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# MACD + SMA200 (ChartArt)
# ---------------------------------------------------------------------------

_MACD_FIXED = {"fast": 12, "slow": 26, "signal": 9, "sma200": 200, "atr_period": 14}
_MACD_GRID = {
    "sl_atr_mult":  [1.5, 2.0, 2.5],
    "tp1_atr_mult": [2.5, 3.0, 4.0],
    "tp2_atr_mult": [4.0, 5.0, 6.0],
}


def _run_macd_sma200(df: pd.DataFrame, initial_balance: float, risk_pct: float, top_n: int) -> list[dict]:
    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    n     = len(close)

    f = _MACD_FIXED
    fast_sma   = calc_sma(close, f["fast"])
    slow_sma   = calc_sma(close, f["slow"])
    sma200     = calc_sma(close, f["sma200"])
    macd_line  = fast_sma - slow_sma
    signal_arr = calc_sma(macd_line, f["signal"])
    hist       = macd_line - signal_arr
    atr_arr    = calc_atr(high, low, close, f["atr_period"])

    min_i = f["sma200"] + f["signal"] + 10
    keys  = list(_MACD_GRID.keys())
    combos = list(product(*_MACD_GRID.values()))
    print(f"  macd_sma200_chartart: {len(combos)} combos...")

    results = []
    for combo in combos:
        p = dict(zip(keys, combo))
        if p["tp1_atr_mult"] <= p["sl_atr_mult"]:
            continue
        if p["tp2_atr_mult"] <= p["tp1_atr_mult"]:
            continue

        balance = initial_balance
        open_pos = None
        trades: list = []

        for i in range(min_i, n):
            h, l = high[i], low[i]

            if open_pos is not None:
                open_pos, trade = _check_position(open_pos, h, l, i)
                if trade is not None:
                    balance += trade["pnl"]
                    trades.append(trade)

            cur_hist  = hist[i]
            prev_hist = hist[i - 1]
            cur_atr   = atr_arr[i]
            if np.isnan(cur_hist) or np.isnan(prev_hist) or np.isnan(cur_atr) or cur_atr <= 0:
                continue

            hist_up   = prev_hist <= 0 and cur_hist > 0
            hist_down = prev_hist >= 0 and cur_hist < 0

            if not hist_up and not hist_down:
                continue
            if i + 1 >= n:
                continue

            cur_macd   = macd_line[i]
            cur_fast   = fast_sma[i]
            cur_slow   = slow_sma[i]
            cur_sma200 = sma200[i]
            close_lag  = close[i - f["slow"]]  # close[slow bars ago]

            if np.isnan(cur_macd) or np.isnan(cur_sma200):
                continue

            signal_side = None
            if (hist_up and cur_macd > 0 and cur_fast > cur_slow and close_lag > cur_sma200):
                signal_side = "long"
            elif (hist_down and cur_macd < 0 and cur_fast < cur_slow and close_lag < cur_sma200):
                signal_side = "short"

            if signal_side is None:
                continue

            # Reversal close
            if open_pos is not None and open_pos["side"] != signal_side:
                trade = _close_eod(open_pos, open_[i + 1], i + 1)
                balance += trade["pnl"]
                trades.append({**trade, "reason": "reversal"})
                open_pos = None

            if open_pos is None:
                ep = open_[i + 1]
                sl_dist = cur_atr * p["sl_atr_mult"]
                qty = (balance * risk_pct) / sl_dist
                if qty <= 0:
                    continue
                entry_fee = qty * ep * TAKER_FEE
                balance -= entry_fee
                if signal_side == "long":
                    sl  = ep - sl_dist
                    tp1 = ep + cur_atr * p["tp1_atr_mult"]
                    tp2 = ep + cur_atr * p["tp2_atr_mult"]
                else:
                    sl  = ep + sl_dist
                    tp1 = ep - cur_atr * p["tp1_atr_mult"]
                    tp2 = ep - cur_atr * p["tp2_atr_mult"]
                open_pos = _open_position(signal_side, ep, sl, tp1, tp2, qty, entry_fee, i + 1)

        if open_pos is not None:
            trade = _close_eod(open_pos, close[-1], n)
            balance += trade["pnl"]
            trades.append(trade)

        r = _metrics(trades, initial_balance, balance)
        r.update(p)
        results.append(r)

    results.sort(key=lambda x: x["pf"] if x["n"] >= 30 else -1, reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# SuperTrend (shared computation)
# ---------------------------------------------------------------------------

def _calc_supertrend(high, low, close, period, multiplier):
    """O(n) SuperTrend — up band, dn band, trend (+1/-1)."""
    from src.signal.strategies.supertrend import _calc_supertrend as _st
    return _st(high, low, close, period, multiplier)


# ---------------------------------------------------------------------------
# RSI + SuperTrend
# ---------------------------------------------------------------------------

_RSI_ST_FIXED = {"atr_period": 10, "rsi_period": 14}
_RSI_ST_GRID = {
    "multiplier":    [2.0, 3.0, 4.0],
    "rsi_threshold": [45.0, 50.0, 55.0],
    "sl_atr_mult":   [1.5, 2.0, 2.5],
    "tp1_atr_mult":  [2.5, 3.0, 4.0],
    "tp2_atr_mult":  [4.0, 5.0, 6.0],
}


def _run_rsi_supertrend(df: pd.DataFrame, initial_balance: float, risk_pct: float, top_n: int) -> list[dict]:
    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    n     = len(close)

    f = _RSI_ST_FIXED
    atr_arr = calc_atr(high, low, close, f["atr_period"])
    rsi_arr = calc_rsi(close, f["rsi_period"])

    # Precompute supertrend per multiplier value (3 times max)
    mult_values = _RSI_ST_GRID["multiplier"]
    st_cache: dict[float, tuple] = {}
    for mult in mult_values:
        st_cache[mult] = _calc_supertrend(high, low, close, f["atr_period"], mult)

    min_i = f["atr_period"] * 3 + f["rsi_period"] * 2 + 10
    keys  = list(_RSI_ST_GRID.keys())
    combos = list(product(*_RSI_ST_GRID.values()))
    print(f"  rsi_supertrend: {len(combos)} combos (supertrend precomputed x{len(mult_values)})...")

    results = []
    for combo in combos:
        p = dict(zip(keys, combo))
        if p["tp1_atr_mult"] <= p["sl_atr_mult"]:
            continue
        if p["tp2_atr_mult"] <= p["tp1_atr_mult"]:
            continue

        up, dn, trend = st_cache[p["multiplier"]]
        balance  = initial_balance
        open_pos = None
        trades: list = []

        for i in range(min_i, n):
            h, l = high[i], low[i]

            if open_pos is not None:
                open_pos, trade = _check_position(open_pos, h, l, i)
                if trade is not None:
                    balance += trade["pnl"]
                    trades.append(trade)

            cur_trend  = trend[i]
            prev_trend = trend[i - 1]
            cur_rsi    = rsi_arr[i]
            cur_atr    = atr_arr[i]

            if np.isnan(cur_rsi) or np.isnan(cur_atr) or cur_atr <= 0:
                continue

            flip_long  = cur_trend == 1  and prev_trend == -1
            flip_short = cur_trend == -1 and prev_trend == 1

            if not flip_long and not flip_short:
                continue
            if i + 1 >= n:
                continue

            rsi_thr = p["rsi_threshold"]
            signal_side = None
            if flip_long and cur_rsi > rsi_thr:
                signal_side = "long"
            elif flip_short and cur_rsi < (100.0 - rsi_thr):
                signal_side = "short"

            if signal_side is None:
                continue

            # Reversal close
            if open_pos is not None and open_pos["side"] != signal_side:
                trade = _close_eod(open_pos, open_[i + 1], i + 1)
                balance += trade["pnl"]
                trades.append({**trade, "reason": "reversal"})
                open_pos = None

            if open_pos is None:
                ep = open_[i + 1]
                sl_dist = cur_atr * p["sl_atr_mult"]
                qty = (balance * risk_pct) / sl_dist
                if qty <= 0:
                    continue
                entry_fee = qty * ep * TAKER_FEE
                balance -= entry_fee
                if signal_side == "long":
                    band_sl = float(up[i])
                    atr_sl  = ep - sl_dist
                    sl      = max(band_sl, atr_sl)
                    tp1     = ep + cur_atr * p["tp1_atr_mult"]
                    tp2     = ep + cur_atr * p["tp2_atr_mult"]
                else:
                    band_sl = float(dn[i])
                    atr_sl  = ep + sl_dist
                    sl      = min(band_sl, atr_sl)
                    tp1     = ep - cur_atr * p["tp1_atr_mult"]
                    tp2     = ep - cur_atr * p["tp2_atr_mult"]
                if sl <= 0:
                    balance += entry_fee  # refund fee, skip
                    continue
                open_pos = _open_position(signal_side, ep, sl, tp1, tp2, qty, entry_fee, i + 1)

        if open_pos is not None:
            trade = _close_eod(open_pos, close[-1], n)
            balance += trade["pnl"]
            trades.append(trade)

        r = _metrics(trades, initial_balance, balance)
        r.update(p)
        results.append(r)

    results.sort(key=lambda x: x["pf"] if x["n"] >= 30 else -1, reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# SuperTrend (plain)
# ---------------------------------------------------------------------------

_ST_FIXED = {"atr_period": 10}
_ST_GRID = {
    "multiplier":   [2.0, 3.0, 4.0],
    "sl_atr_mult":  [1.5, 2.0, 2.5],
    "tp1_atr_mult": [2.5, 3.0, 4.0],
    "tp2_atr_mult": [4.0, 5.0, 6.0],
}


def _run_supertrend(df: pd.DataFrame, initial_balance: float, risk_pct: float, top_n: int) -> list[dict]:
    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    n     = len(close)

    f = _ST_FIXED
    atr_arr = calc_atr(high, low, close, f["atr_period"])

    mult_values = _ST_GRID["multiplier"]
    st_cache: dict[float, tuple] = {}
    for mult in mult_values:
        st_cache[mult] = _calc_supertrend(high, low, close, f["atr_period"], mult)

    min_i = f["atr_period"] * 3 + 10
    keys  = list(_ST_GRID.keys())
    combos = list(product(*_ST_GRID.values()))
    print(f"  supertrend: {len(combos)} combos (supertrend precomputed x{len(mult_values)})...")

    results = []
    for combo in combos:
        p = dict(zip(keys, combo))
        if p["tp1_atr_mult"] <= p["sl_atr_mult"]:
            continue
        if p["tp2_atr_mult"] <= p["tp1_atr_mult"]:
            continue

        up, dn, trend = st_cache[p["multiplier"]]
        balance  = initial_balance
        open_pos = None
        trades: list = []

        for i in range(min_i, n):
            h, l = high[i], low[i]

            if open_pos is not None:
                open_pos, trade = _check_position(open_pos, h, l, i)
                if trade is not None:
                    balance += trade["pnl"]
                    trades.append(trade)

            cur_trend  = trend[i]
            prev_trend = trend[i - 1]
            cur_atr    = atr_arr[i]

            if np.isnan(cur_atr) or cur_atr <= 0:
                continue

            flip_long  = cur_trend == 1  and prev_trend == -1
            flip_short = cur_trend == -1 and prev_trend == 1

            if not flip_long and not flip_short:
                continue
            if i + 1 >= n:
                continue

            signal_side = "long" if flip_long else "short"

            # Reversal close
            if open_pos is not None and open_pos["side"] != signal_side:
                trade = _close_eod(open_pos, open_[i + 1], i + 1)
                balance += trade["pnl"]
                trades.append({**trade, "reason": "reversal"})
                open_pos = None

            if open_pos is None:
                ep = open_[i + 1]
                sl_dist = cur_atr * p["sl_atr_mult"]
                qty = (balance * risk_pct) / sl_dist
                if qty <= 0:
                    continue
                entry_fee = qty * ep * TAKER_FEE
                balance -= entry_fee
                if signal_side == "long":
                    band_sl = float(up[i])
                    atr_sl  = ep - sl_dist
                    sl      = max(band_sl, atr_sl)
                    tp1     = ep + cur_atr * p["tp1_atr_mult"]
                    tp2     = ep + cur_atr * p["tp2_atr_mult"]
                else:
                    band_sl = float(dn[i])
                    atr_sl  = ep + sl_dist
                    sl      = min(band_sl, atr_sl)
                    tp1     = ep - cur_atr * p["tp1_atr_mult"]
                    tp2     = ep - cur_atr * p["tp2_atr_mult"]
                if sl <= 0:
                    balance += entry_fee
                    continue
                open_pos = _open_position(signal_side, ep, sl, tp1, tp2, qty, entry_fee, i + 1)

        if open_pos is not None:
            trade = _close_eod(open_pos, close[-1], n)
            balance += trade["pnl"]
            trades.append(trade)

        r = _metrics(trades, initial_balance, balance)
        r.update(p)
        results.append(r)

    results.sort(key=lambda x: x["pf"] if x["n"] >= 30 else -1, reverse=True)
    return results[:top_n]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_STRATEGY_PARAM_KEYS = {
    "ema_crossover":       ["adx_threshold", "sl_atr_mult", "tp1_atr_mult", "tp2_atr_mult"],
    "macd_sma200_chartart": ["sl_atr_mult", "tp1_atr_mult", "tp2_atr_mult"],
    "rsi_supertrend":      ["multiplier", "rsi_threshold", "sl_atr_mult", "tp1_atr_mult", "tp2_atr_mult"],
    "supertrend":          ["multiplier", "sl_atr_mult", "tp1_atr_mult", "tp2_atr_mult"],
}

_STRATEGY_DEFAULTS = {
    "ema_crossover": {"adx_threshold": 25, "sl_atr_mult": 2.0, "tp1_atr_mult": 3.0, "tp2_atr_mult": 5.0},
    "macd_sma200_chartart": {"sl_atr_mult": 2.0, "tp1_atr_mult": 3.0, "tp2_atr_mult": 5.0},
    "rsi_supertrend": {"multiplier": 3.0, "rsi_threshold": 50, "sl_atr_mult": 2.0, "tp1_atr_mult": 3.0, "tp2_atr_mult": 5.0},
    "supertrend": {"multiplier": 3.0, "sl_atr_mult": 2.0, "tp1_atr_mult": 3.0, "tp2_atr_mult": 5.0},
}


def print_results(strategy: str, results: list[dict], top_n: int) -> None:
    param_keys = _STRATEGY_PARAM_KEYS[strategy]
    sep = "=" * 110
    print(f"\n{sep}")
    print(f"  Top {len(results)} {strategy} (sorted by PF, min 30 trades)")
    print(sep)

    param_header = "  ".join(f"{k[:6]:>6}" for k in param_keys)
    header = f"  {param_header}  {'N':>4}  {'WR':>6}  {'PF':>6}  {'NetPnL':>9}  {'MaxDD':>6}  {'AvgBars':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        pf = r["pf"]
        pf_str = f"{pf:.2f}" if pf != float("inf") else "  ∞"
        param_vals = "  ".join(f"{r[k]:>6.1f}" for k in param_keys)
        print(
            f"  {param_vals}  {r['n']:>4}  {r['wr']:>5.1%}  {pf_str:>6}"
            f"  {r['net_pnl']:>+9.4f}  {r['max_dd']:>5.1f}%  {r['avg_bars']:>7.1f}"
        )
    print(sep)

    if results:
        best = results[0]
        print(f"\nBest params ({strategy}):")
        for k in param_keys:
            print(f"  {k}={best[k]}")

        # Build env var JSON
        env_params = {k: best[k] for k in param_keys}
        import json
        env_str = json.dumps(env_params, separators=(",", ":"))
        env_key = f"STRATEGY_PARAMS_{strategy.upper()}"
        print(f"\nEXPORT env var:")
        print(f"  {env_key}='{env_str}'")
        print()


# ---------------------------------------------------------------------------
# Public API (used by backtest_report.py)
# ---------------------------------------------------------------------------

_RUNNERS = {
    "ema_crossover":        _run_ema_crossover,
    "macd_sma200_chartart": _run_macd_sma200,
    "rsi_supertrend":       _run_rsi_supertrend,
    "supertrend":           _run_supertrend,
}


def run_grid(
    strategy: str,
    df: pd.DataFrame,
    initial_balance: float = 100.0,
    risk_pct: float = 0.01,
    top_n: int = 15,
) -> list[dict]:
    """Run grid search for a strategy. Pass top_n=9999 to get all combos."""
    return _RUNNERS[strategy](df, initial_balance, risk_pct, top_n)


def find_params_in_results(results: list[dict], params: dict, param_keys: list[str]) -> dict | None:
    """Find a specific param combo in full grid results (exact match within 0.001)."""
    for r in results:
        if all(abs(float(r.get(k, float("nan"))) - float(params.get(k, float("nan")))) < 0.001
               for k in param_keys):
            return r
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_date(s: str) -> int:
    dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Grid-search params for active strategies")
    parser.add_argument("--strategy", required=True, choices=list(_RUNNERS.keys()))
    parser.add_argument("--symbol",   default="BTCUSDT")
    parser.add_argument("--start",    default="2020-01-01")
    parser.add_argument("--end",      default="2024-12-31")
    parser.add_argument("--balance",  type=float, default=100.0)
    parser.add_argument("--risk-pct", type=float, default=0.01)
    parser.add_argument("--top",      type=int,   default=15)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    since_ms = _parse_date(args.start)
    until_ms = _parse_date(args.end)

    df = _load_ohlcv(args.symbol, since_ms, until_ms, no_cache=args.no_cache)
    logger.info("Loaded %d candles for %s", len(df), args.symbol)

    runner = _RUNNERS[args.strategy]
    results = runner(df, initial_balance=args.balance, risk_pct=args.risk_pct, top_n=args.top)
    print_results(args.strategy, results, top_n=args.top)


if __name__ == "__main__":
    main()
