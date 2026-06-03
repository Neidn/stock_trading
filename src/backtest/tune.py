"""Fast parameter grid search for ema_pullback_rsi.

Precomputes all indicators once on the full dataset, then replays each
parameter combination in O(n) instead of the O(n²) of the generic engine.

Fetches KRX stock data via yfinance (e.g. '005930.KS' for Samsung Electronics).

Usage:
    python -m src.backtest.tune \\
        --symbol 005930.KS \\
        --start 2024-01-01 \\
        --end 2024-12-31 \\
        [--balance 100] \\
        [--top 15]
"""

from __future__ import annotations

import argparse
import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.signal.indicators import calc_adx, calc_atr, calc_ema, calc_rsi

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "backtest"
TAKER_FEE = 0.0004

FIXED_PARAMS = {
    "ema_fast":   20,
    "ema_mid":    50,
    "ema_slow":   200,
    "rsi_period": 14,
    "adx_period": 14,
}

GRID = {
    "adx_threshold": [20, 25, 30],
    "rsi_low":        [35, 40, 45],
    "rsi_high":       [55, 60, 65],
    "sl_atr_mult":    [1.5, 2.0, 2.5],
    "tp1_atr_mult":   [2.5, 3.0, 4.0],
    "tp2_atr_mult":   [4.0, 5.0, 6.0],
}


# ---------------------------------------------------------------------------
# Precompute
# ---------------------------------------------------------------------------

def _precompute(df: pd.DataFrame) -> dict:
    high  = df["high"].to_numpy(dtype=float)
    low   = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    open_ = df["open"].to_numpy(dtype=float)
    ts    = df["timestamp"].to_numpy(dtype=np.int64)

    ema_fast = calc_ema(close, FIXED_PARAMS["ema_fast"])
    ema_mid  = calc_ema(close, FIXED_PARAMS["ema_mid"])
    ema_slow = calc_ema(close, FIXED_PARAMS["ema_slow"])
    rsi      = calc_rsi(close, FIXED_PARAMS["rsi_period"])
    adx      = calc_adx(high, low, close, FIXED_PARAMS["adx_period"])
    atr      = calc_atr(high, low, close, FIXED_PARAMS["adx_period"])

    return dict(
        high=high, low=low, close=close, open=open_, ts=ts,
        ema_fast=ema_fast, ema_mid=ema_mid, ema_slow=ema_slow,
        rsi=rsi, adx=adx, atr=atr,
    )


# ---------------------------------------------------------------------------
# O(n) simulation
# ---------------------------------------------------------------------------

def _simulate(
    pre: dict,
    adx_threshold: float,
    rsi_low: float,
    rsi_high: float,
    sl_atr_mult: float,
    tp1_atr_mult: float,
    tp2_atr_mult: float,
    initial_balance: float,
    risk_pct: float,
) -> dict:
    min_candles = FIXED_PARAMS["ema_slow"] + FIXED_PARAMS["rsi_period"] + 10
    n = len(pre["close"])

    ema_fast = pre["ema_fast"]
    ema_mid  = pre["ema_mid"]
    ema_slow = pre["ema_slow"]
    rsi_arr  = pre["rsi"]
    adx_arr  = pre["adx"]
    atr_arr  = pre["atr"]
    close    = pre["close"]
    open_    = pre["open"]
    high     = pre["high"]
    low      = pre["low"]

    balance = initial_balance
    open_pos = None
    trades: list[dict] = []

    for i in range(min_candles, n):
        h = high[i]
        l = low[i]

        # 1. Check SL/TP — 50% close at tp1, remaining 50% at tp2 or SL
        if open_pos is not None:
            side      = open_pos["side"]
            sl        = open_pos["sl"]
            tp1       = open_pos["tp1"]
            tp2       = open_pos["tp2"]
            half_qty  = open_pos["qty"] / 2.0
            half_fee  = open_pos["entry_fee"] / 2.0

            sl_hit  = (side == "long" and l <= sl)  or (side == "short" and h >= sl)
            tp1_hit = (side == "long" and h >= tp1) or (side == "short" and l <= tp1)
            tp2_hit = (side == "long" and h >= tp2) or (side == "short" and l <= tp2)

            if not open_pos["half_closed"]:
                if sl_hit:
                    pnl_raw  = (sl - open_pos["entry_price"]) * open_pos["qty"] if side == "long" \
                               else (open_pos["entry_price"] - sl) * open_pos["qty"]
                    exit_fee = open_pos["qty"] * sl * TAKER_FEE
                    realized = pnl_raw - open_pos["entry_fee"] - exit_fee
                    balance += realized
                    trades.append({"pnl": realized, "reason": "sl", "bars": i - open_pos["entry_bar"]})
                    open_pos = None
                elif tp1_hit:
                    # Close first 50% at tp1
                    pnl_raw  = (tp1 - open_pos["entry_price"]) * half_qty if side == "long" \
                               else (open_pos["entry_price"] - tp1) * half_qty
                    exit_fee = half_qty * tp1 * TAKER_FEE
                    partial  = pnl_raw - half_fee - exit_fee
                    balance += partial
                    open_pos["half_closed"] = True
                    open_pos["partial_pnl"] = partial
            else:
                # Remaining 50%: check tp2 or SL
                if sl_hit or tp2_hit:
                    exit_price = sl if sl_hit else tp2
                    reason     = "sl_after_tp1" if sl_hit else "tp2"
                    pnl_raw    = (exit_price - open_pos["entry_price"]) * half_qty if side == "long" \
                                 else (open_pos["entry_price"] - exit_price) * half_qty
                    exit_fee   = half_qty * exit_price * TAKER_FEE
                    second     = pnl_raw - half_fee - exit_fee
                    balance   += second
                    total_pnl  = open_pos["partial_pnl"] + second
                    trades.append({"pnl": total_pnl, "reason": reason, "bars": i - open_pos["entry_bar"]})
                    open_pos = None

        # 2. Signal
        cur_adx      = adx_arr[i]
        cur_rsi      = rsi_arr[i]
        cur_atr      = atr_arr[i]
        cur_close    = close[i]
        cur_open     = open_[i]
        cur_ema_fast = ema_fast[i]
        cur_ema_mid  = ema_mid[i]
        cur_ema_slow = ema_slow[i]

        if np.isnan(cur_adx) or np.isnan(cur_rsi) or np.isnan(cur_atr):
            continue
        if cur_adx < adx_threshold:
            continue

        trend_up   = cur_ema_fast > cur_ema_mid > cur_ema_slow
        trend_down = cur_ema_fast < cur_ema_mid < cur_ema_slow
        rsi_ok     = rsi_low <= cur_rsi <= rsi_high

        signal_side = None
        if trend_up and rsi_ok and cur_close > cur_open:
            signal_side = "long"
        elif trend_down and rsi_ok and cur_close < cur_open:
            signal_side = "short"

        if signal_side is None:
            continue

        # Reversal close
        if open_pos is not None and open_pos["side"] != signal_side:
            if i + 1 < n:
                rev_price = open_[i + 1]
                side = open_pos["side"]
                pnl_raw = (
                    (rev_price - open_pos["entry_price"]) * open_pos["qty"]
                    if side == "long"
                    else (open_pos["entry_price"] - rev_price) * open_pos["qty"]
                )
                exit_fee = open_pos["qty"] * rev_price * TAKER_FEE
                realized = pnl_raw - open_pos["entry_fee"] - exit_fee
                balance += realized
                trades.append({"pnl": realized, "reason": "reversal", "bars": i + 1 - open_pos["entry_bar"]})
                open_pos = None

        # Entry
        if open_pos is None and i + 1 < n:
            entry_price = open_[i + 1]
            sl_dist = cur_atr * sl_atr_mult
            if sl_dist <= 0:
                continue
            qty = (balance * risk_pct) / sl_dist
            if qty <= 0:
                continue
            entry_fee = qty * entry_price * TAKER_FEE
            balance -= entry_fee

            if signal_side == "long":
                sl  = entry_price - sl_dist
                tp1 = entry_price + cur_atr * tp1_atr_mult
                tp2 = entry_price + cur_atr * tp2_atr_mult
            else:
                sl  = entry_price + sl_dist
                tp1 = entry_price - cur_atr * tp1_atr_mult
                tp2 = entry_price - cur_atr * tp2_atr_mult

            open_pos = {
                "side": signal_side, "entry_price": entry_price,
                "sl": sl, "tp1": tp1, "tp2": tp2, "qty": qty,
                "entry_fee": entry_fee, "entry_bar": i + 1,
                "half_closed": False, "partial_pnl": 0.0,
            }

    # Close at end — handle partial close state
    if open_pos is not None:
        exit_price = close[-1]
        side = open_pos["side"]
        if open_pos["half_closed"]:
            remaining_qty = open_pos["qty"] / 2.0
            half_fee = open_pos["entry_fee"] / 2.0
            pnl_raw  = (exit_price - open_pos["entry_price"]) * remaining_qty if side == "long" \
                       else (open_pos["entry_price"] - exit_price) * remaining_qty
            exit_fee = remaining_qty * exit_price * TAKER_FEE
            realized = open_pos["partial_pnl"] + pnl_raw - half_fee - exit_fee
        else:
            pnl_raw  = (exit_price - open_pos["entry_price"]) * open_pos["qty"] if side == "long" \
                       else (open_pos["entry_price"] - exit_price) * open_pos["qty"]
            exit_fee = open_pos["qty"] * exit_price * TAKER_FEE
            realized = pnl_raw - open_pos["entry_fee"] - exit_fee
        balance += realized
        trades.append({"pnl": realized, "reason": "end_of_data", "bars": n - 1 - open_pos["entry_bar"]})

    # Metrics
    total = len(trades)
    if total == 0:
        return {"n": 0, "pf": 0.0, "wr": 0.0, "net_pnl": 0.0, "max_dd": 0.0, "avg_bars": 0.0}

    wins   = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] < 0]
    gross_p = sum(wins)
    gross_l = abs(sum(losses))
    net_pnl = sum(t["pnl"] for t in trades)
    pf = gross_p / gross_l if gross_l > 0 else float("inf")

    # Max drawdown
    eq = initial_balance
    peak = eq
    max_dd = 0.0
    for t in trades:
        eq += t["pnl"]
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    return {
        "n":        total,
        "pf":       pf,
        "wr":       len(wins) / total,
        "net_pnl":  net_pnl,
        "max_dd":   max_dd,
        "avg_bars": sum(t["bars"] for t in trades) / total,
        "final_bal": balance,
    }


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_grid(
    df: pd.DataFrame,
    initial_balance: float = 100.0,
    risk_pct: float = 0.01,
    top_n: int = 15,
) -> list[dict]:
    print(f"Precomputing indicators on {len(df)} candles...")
    pre = _precompute(df)

    keys = list(GRID.keys())
    values = list(GRID.values())
    combos = list(product(*values))
    print(f"Running {len(combos)} parameter combinations...\n")

    results = []
    for combo in combos:
        p = dict(zip(keys, combo))
        if p["rsi_low"] >= p["rsi_high"]:
            continue
        if p["tp1_atr_mult"] <= p["sl_atr_mult"]:
            continue
        if p["tp2_atr_mult"] <= p["tp1_atr_mult"]:
            continue

        r = _simulate(
            pre,
            adx_threshold=p["adx_threshold"],
            rsi_low=p["rsi_low"],
            rsi_high=p["rsi_high"],
            sl_atr_mult=p["sl_atr_mult"],
            tp1_atr_mult=p["tp1_atr_mult"],
            tp2_atr_mult=p["tp2_atr_mult"],
            initial_balance=initial_balance,
            risk_pct=risk_pct,
        )
        r.update(p)
        results.append(r)

    # Sort by profit factor (require at least 30 trades for statistical validity)
    results.sort(key=lambda x: x["pf"] if x["n"] >= 30 else -1, reverse=True)
    return results[:top_n]


def print_grid_results(results: list[dict], top_n: int = 15) -> None:
    print(f"\n{'='*100}")
    print(f"  Top {len(results)} ema_pullback_rsi Parameter Combinations (sorted by PF, min 30 trades)")
    print(f"{'='*100}")
    header = (
        f"  {'ADX':>5}  {'RSI_L':>5}  {'RSI_H':>5}  {'SL':>4}  {'TP1':>4}  {'TP2':>4}"
        f"  {'N':>4}  {'WR':>6}  {'PF':>6}  {'NetPnL':>9}  {'MaxDD':>6}  {'AvgBars':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in results:
        pf = r["pf"]
        pf_str = f"{pf:.2f}" if pf != float("inf") else "  ∞"
        print(
            f"  {r['adx_threshold']:>5.0f}  {r['rsi_low']:>5.0f}  {r['rsi_high']:>5.0f}"
            f"  {r['sl_atr_mult']:>4.1f}  {r['tp1_atr_mult']:>4.1f}  {r['tp2_atr_mult']:>4.1f}"
            f"  {r['n']:>4}  {r['wr']:>5.1%}  {pf_str:>6}"
            f"  {r['net_pnl']:>+9.4f}  {r['max_dd']:>5.1f}%  {r['avg_bars']:>7.1f}"
        )
    print(f"{'='*100}\n")

    if results:
        best = results[0]
        print("Best params:")
        print(f"  adx_threshold={best['adx_threshold']}  rsi_low={best['rsi_low']}  rsi_high={best['rsi_high']}")
        print(f"  sl_atr_mult={best['sl_atr_mult']}  tp1_atr_mult={best['tp1_atr_mult']}  tp2_atr_mult={best['tp2_atr_mult']}")
        print(f"\nEXPORT env var:")
        print(
            f'  STRATEGY_PARAMS=\'{{"adx_threshold":{best["adx_threshold"]},'
            f'"rsi_low":{best["rsi_low"]},"rsi_high":{best["rsi_high"]},'
            f'"sl_atr_mult":{best["sl_atr_mult"]},"tp1_atr_mult":{best["tp1_atr_mult"]},'
            f'"tp2_atr_mult":{best["tp2_atr_mult"]}}}\''
        )
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Grid-search ema_pullback_rsi params")
    parser.add_argument("--symbol",  default="005930.KS",
                        help="KRX ticker with suffix, e.g. '005930.KS' (default: Samsung)")
    parser.add_argument("--start",   default="2024-01-01")
    parser.add_argument("--end",     default="2024-12-31")
    parser.add_argument("--balance", type=float, default=100.0)
    parser.add_argument("--risk-pct", type=float, default=0.01)
    parser.add_argument("--top",     type=int, default=15)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    since_ms = args.start
    until_ms = args.end

    # Try cache first
    cache_file = DATA_DIR / f"{args.symbol.replace('/', '_')}_1h_{args.start}_{args.end}.csv"
    if not args.no_cache and cache_file.exists():
        logger.info("Loading from cache: %s", cache_file.name)
        df = pd.read_csv(cache_file)
    else:
        import yfinance as yf
        ticker = yf.Ticker(args.symbol)
        df_raw = ticker.history(start=args.start, end=args.end, interval="1h", auto_adjust=True)
        if df_raw.empty:
            print("No data returned from yfinance")
            return
        df_raw = df_raw.reset_index()
        date_col = "Datetime" if "Datetime" in df_raw.columns else "Date"
        df_raw["timestamp"] = pd.to_datetime(df_raw[date_col]).astype("int64") // 10**6
        df_raw = df_raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        df = df_raw[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file, index=False)
        logger.info("Fetched %d candles", len(df))

    top = run_grid(df, initial_balance=args.balance, risk_pct=args.risk_pct, top_n=args.top)
    print_grid_results(top, top_n=args.top)


if __name__ == "__main__":
    main()
