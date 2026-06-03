"""Backtesting engine — replay historical OHLCV against any BaseStrategy.

Fetches KRX stock data via yfinance (e.g. '005930.KS' for Samsung Electronics).

Simulation rules:
  - Signal generated on candle i close → entry at candle i+1 open.
  - SL/TP checked each subsequent candle using candle high/low.
  - If both SL and TP hit in same candle → SL assumed first (conservative).
  - Reversal signal (opposite direction) → close at next open, enter immediately.
  - Fee: 0.2% round-trip (KRX brokerage + transaction tax).
  - One position per symbol at a time (spot only, long-only).

Usage:
    python -m src.backtest.engine \\
        --strategy ema_crossover \\
        --symbol 005930.KS \\
        --start 2024-01-01 \\
        --end 2024-12-31 \\
        [--balance 100] \\
        [--risk-pct 0.01] \\
        [--save-trades trades.csv] \\
        [--compare]          # run all strategies, print comparison table
"""

from __future__ import annotations

import argparse
import csv
import importlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.signal.base_strategy import BaseStrategy, SignalResult
from src.signal.indicators import calc_atr

logger = logging.getLogger(__name__)

TAKER_FEE = 0.002               # 0.2% round trip (KRX brokerage + transaction tax)
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "backtest"

_ALL_STRATEGIES = [
    "bb_breakout",
    "bb_rsi_chartart",
    "ema_crossover",
    "ema_pullback_rsi",
    "macd_sma200_chartart",
    "rsi_macd",
    "rsi_supertrend",
    "supertrend",
    "zscore_reversion",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    realized_pnl: float
    entry_fee: float
    exit_fee: float
    close_reason: str      # 'sl' | 'tp1' | 'reversal' | 'end_of_data' | 'liquidated'
    entry_bar: int
    exit_bar: int
    bars_held: int
    entry_ts: int = 0
    exit_ts: int = 0
    funding_fee: float = 0.0


@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    timeframe: str
    start_ts: int
    end_ts: int
    initial_balance: float
    final_balance: float
    trades: list[Trade] = field(default_factory=list)

    @property
    def net_pnl(self) -> float:
        return self.final_balance - self.initial_balance

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def winning_trades(self) -> int:
        return sum(1 for t in self.trades if t.realized_pnl > 0)

    @property
    def losing_trades(self) -> int:
        return sum(1 for t in self.trades if t.realized_pnl < 0)

    @property
    def win_rate(self) -> float:
        return self.winning_trades / self.total_trades if self.total_trades else 0.0

    @property
    def gross_profit(self) -> float:
        return sum(t.realized_pnl for t in self.trades if t.realized_pnl > 0)

    @property
    def gross_loss(self) -> float:
        return abs(sum(t.realized_pnl for t in self.trades if t.realized_pnl < 0))

    @property
    def profit_factor(self) -> float:
        return self.gross_profit / self.gross_loss if self.gross_loss > 0 else float("inf")

    @property
    def avg_pnl(self) -> float:
        return sum(t.realized_pnl for t in self.trades) / self.total_trades if self.total_trades else 0.0

    @property
    def best_trade(self) -> float:
        return max((t.realized_pnl for t in self.trades), default=0.0)

    @property
    def worst_trade(self) -> float:
        return min((t.realized_pnl for t in self.trades), default=0.0)

    @property
    def max_drawdown_pct(self) -> float:
        equity = self.initial_balance
        peak = equity
        max_dd = 0.0
        for t in self.trades:
            equity += t.realized_pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def close_reason_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for t in self.trades:
            counts[t.close_reason] = counts.get(t.close_reason, 0) + 1
        return counts

    @property
    def avg_bars_held(self) -> float:
        return (
            sum(t.bars_held for t in self.trades) / self.total_trades
            if self.total_trades else 0.0
        )

    @property
    def return_pct(self) -> float:
        return (self.final_balance - self.initial_balance) / self.initial_balance * 100

    @property
    def total_liquidations(self) -> int:
        return sum(1 for t in self.trades if t.close_reason == "liquidated")

    @property
    def total_funding_paid(self) -> float:
        return sum(t.funding_fee for t in self.trades)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Simulate a BaseStrategy on historical OHLCV data (KRX spot, long-only).

    Args:
        strategy:        Instantiated BaseStrategy subclass.
        symbol:          KRX ticker with suffix, e.g. '005930.KS'.
        initial_balance: Starting balance. Default 100.0.
        risk_pct:        Fraction of balance to risk per trade. Default 0.01 (1%).
        taker_fee:       Fee rate (round-trip). Default 0.002 (0.2% KRX).
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        symbol: str,
        initial_balance: float = 100.0,
        risk_pct: float = 0.01,
        taker_fee: float = TAKER_FEE,
    ) -> None:
        self._strategy = strategy
        self._symbol = symbol
        self._initial_balance = initial_balance
        self._risk_pct = risk_pct
        self._taker_fee = taker_fee

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        timeframe: str,
        start: str,
        end: str,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles via yfinance, using local CSV cache when available."""
        import yfinance as yf
        symbol = self._symbol
        cache_file = DATA_DIR / f"{symbol.replace('/', '_')}_{timeframe}_{start}_{end}.csv"
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        if use_cache and cache_file.exists():
            logger.info("Loading from cache: %s", cache_file.name)
            return pd.read_csv(cache_file)

        logger.info("Fetching %s %s from yfinance...", symbol, timeframe)
        ticker = yf.Ticker(symbol)
        df_raw = ticker.history(start=start, end=end, interval=timeframe, auto_adjust=True)
        if df_raw.empty:
            logger.warning("No data returned for %s %s %s→%s", symbol, timeframe, start, end)
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df_raw = df_raw.reset_index()
        date_col = "Datetime" if "Datetime" in df_raw.columns else "Date"
        df_raw["timestamp"] = pd.to_datetime(df_raw[date_col]).astype("int64") // 10**6
        df_raw = df_raw.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume",
        })
        df = df_raw[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
        df.to_csv(cache_file, index=False)
        logger.info("Fetched %d candles, cached to %s", len(df), cache_file.name)
        return df

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> BacktestResult:
        """Simulate strategy on historical data. No lookahead."""
        strategy = self._strategy
        symbol = self._symbol
        min_candles = strategy.get_min_candles()
        trades: list[Trade] = []
        balance = self._initial_balance

        open_pos: Optional[dict] = None
        n = len(df)

        for i in range(min_candles, n):
            candle = df.iloc[i]

            # --- 1. Check SL/TP on current candle ---
            if open_pos is not None:
                exit_info = self._check_sl_tp(candle, open_pos)
                if exit_info:
                    trade = self._close_position(open_pos, exit_info["price"],
                                                  exit_info["reason"], i,
                                                  int(candle["timestamp"]))
                    balance += trade.realized_pnl
                    trades.append(trade)
                    open_pos = None

            # --- 2. Generate signal on candles [0..i] ---
            df_slice = df.iloc[: i + 1]
            try:
                signal = strategy.generate_signal(df_slice, symbol)
            except Exception as exc:
                logger.debug("Signal error bar %d: %s", i, exc)
                continue

            if not signal.is_actionable() or signal.sl is None:
                continue

            new_side = signal.signal_type  # 'long' or 'short'

            # --- 3. Reversal: close open position ---
            if open_pos is not None and open_pos["side"] != new_side:
                if i + 1 < n:
                    next_open = float(df.iloc[i + 1]["open"])
                    next_ts = int(df.iloc[i + 1]["timestamp"])
                    trade = self._close_position(open_pos, next_open, "reversal", i + 1, next_ts)
                    balance += trade.realized_pnl
                    trades.append(trade)
                    open_pos = None
                else:
                    continue

            # --- 4. Enter new position if no open position ---
            if open_pos is None:
                if i + 1 >= n:
                    break
                next_candle = df.iloc[i + 1]
                entry_price = float(next_candle["open"])
                entry_ts = int(next_candle["timestamp"])

                sl_dist = abs(float(signal.entry_price or entry_price) - signal.sl)
                if sl_dist <= 0:
                    continue

                # Recompute ATR for sizing (on current slice)
                close_arr = df_slice["close"].to_numpy(dtype=float)
                high_arr = df_slice["high"].to_numpy(dtype=float)
                low_arr = df_slice["low"].to_numpy(dtype=float)
                atr_arr = calc_atr(high_arr, low_arr, close_arr)
                atr = float(atr_arr[-1])
                if math.isnan(atr) or atr <= 0:
                    continue

                risk_amount = balance * self._risk_pct
                qty = risk_amount / sl_dist
                if qty <= 0:
                    continue

                entry_notional = qty * entry_price
                entry_fee = entry_notional * self._taker_fee
                balance -= entry_fee

                # SL/TP anchored to actual entry price, same dollar distance as signal
                if signal.tp1 and signal.entry_price:
                    tp1_dist = abs(float(signal.entry_price) - signal.tp1)
                else:
                    tp1_dist = sl_dist * 1.5

                if new_side == "long":
                    sl = entry_price - sl_dist
                    tp1 = entry_price + tp1_dist
                else:
                    sl = entry_price + sl_dist
                    tp1 = entry_price - tp1_dist

                open_pos = {
                    "side": new_side,
                    "entry_price": entry_price,
                    "sl": sl,
                    "tp1": tp1,
                    "qty": qty,
                    "entry_fee": entry_fee,
                    "entry_bar": i + 1,
                    "entry_ts": entry_ts,
                    "accumulated_funding": 0.0,
                }

        # --- 5. Close any remaining position at end of data ---
        if open_pos is not None:
            last = df.iloc[-1]
            exit_price = float(last["close"])
            trade = self._close_position(
                open_pos, exit_price, "end_of_data",
                n - 1, int(last["timestamp"]),
            )
            balance += trade.realized_pnl
            trades.append(trade)

        return BacktestResult(
            strategy=strategy.get_name(),
            symbol=symbol,
            timeframe=strategy.get_timeframe(),
            start_ts=int(df.iloc[0]["timestamp"]),
            end_ts=int(df.iloc[-1]["timestamp"]),
            initial_balance=self._initial_balance,
            final_balance=balance,
            trades=trades,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_sl_tp(self, candle, pos: dict) -> Optional[dict]:
        side = pos["side"]
        sl = pos["sl"]
        tp1 = pos["tp1"]
        high = float(candle["high"])
        low = float(candle["low"])

        sl_hit = (side == "long" and low <= sl) or (side == "short" and high >= sl)
        tp_hit = (side == "long" and high >= tp1) or (side == "short" and low <= tp1)

        if sl_hit and tp_hit:
            return {"price": sl, "reason": "sl"}  # SL first (conservative)
        if sl_hit:
            return {"price": sl, "reason": "sl"}
        if tp_hit:
            return {"price": tp1, "reason": "tp1"}
        return None

    def _close_position(
        self,
        pos: dict,
        exit_price: float,
        reason: str,
        exit_bar: int,
        exit_ts: int,
    ) -> Trade:
        side = pos["side"]
        entry_price = pos["entry_price"]
        qty = pos["qty"]
        entry_fee = pos["entry_fee"]

        if side == "long":
            pnl_raw = (exit_price - entry_price) * qty
        else:
            pnl_raw = (entry_price - exit_price) * qty

        exit_fee = qty * exit_price * self._taker_fee
        funding_fee = pos.get("accumulated_funding", 0.0)
        realized_pnl = pnl_raw - entry_fee - exit_fee - funding_fee
        bars_held = max(0, exit_bar - pos["entry_bar"])

        return Trade(
            symbol=self._symbol,
            side=side,
            entry_price=entry_price,
            exit_price=exit_price,
            qty=qty,
            realized_pnl=realized_pnl,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            close_reason=reason,
            entry_bar=pos["entry_bar"],
            exit_bar=exit_bar,
            bars_held=bars_held,
            entry_ts=pos.get("entry_ts", 0),
            exit_ts=exit_ts,
            funding_fee=funding_fee,
        )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _ts_to_str(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def print_report(result: BacktestResult) -> None:
    pf = result.profit_factor
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"
    start = _ts_to_str(result.start_ts)
    end = _ts_to_str(result.end_ts)

    print(f"\n{'='*52}")
    print(f"  Backtest: {result.strategy} | {result.symbol} | spot")
    print(f"  Period:   {start} → {end}  ({result.timeframe})")
    print(f"{'='*52}")
    print(f"  Trades:          {result.total_trades}")
    if result.total_trades == 0:
        print("  No trades generated.")
        print(f"{'='*52}\n")
        return
    print(f"  Win rate:        {result.win_rate:.1%}  ({result.winning_trades}W / {result.losing_trades}L)")
    print(f"  Profit factor:   {pf_str}")
    print(f"  Net PnL:         {result.net_pnl:+.4f}  ({result.return_pct:+.2f}%)")
    print(f"  Gross profit:    {result.gross_profit:.4f}")
    print(f"  Gross loss:      {result.gross_loss:.4f}")
    print(f"  Avg PnL/trade:   {result.avg_pnl:+.4f}")
    print(f"  Best trade:      {result.best_trade:+.4f}")
    print(f"  Worst trade:     {result.worst_trade:+.4f}")
    print(f"  Max drawdown:    {result.max_drawdown_pct:.2f}%")
    print(f"  Avg bars held:   {result.avg_bars_held:.1f}")
    print(f"  Balance:         {result.initial_balance:.2f} → {result.final_balance:.4f}")
    print(f"  Exit reasons:    {result.close_reason_counts}")
    print(f"{'='*52}\n")


def print_trade_log(result: BacktestResult, max_rows: int = 20) -> None:
    if not result.trades:
        return
    print(f"\n  Trade log (last {min(max_rows, len(result.trades))}):")
    header = f"  {'#':>3}  {'Side':<6}  {'Entry':>10}  {'Exit':>10}  {'PnL':>8}  {'Reason':<14}  {'Bars':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, t in enumerate(result.trades[-max_rows:], 1):
        pnl_sign = "+" if t.realized_pnl >= 0 else ""
        print(
            f"  {i:>3}  {t.side:<6}  {t.entry_price:>10.6f}  {t.exit_price:>10.6f}"
            f"  {pnl_sign}{t.realized_pnl:>7.4f}  {t.close_reason:<14}  {t.bars_held:>5}"
        )
    print()


def save_trades_csv(result: BacktestResult, path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy", "symbol", "side", "entry_ts", "exit_ts",
            "entry_price", "exit_price", "qty",
            "entry_fee", "exit_fee", "funding_fee", "realized_pnl", "close_reason", "bars_held",
        ])
        for t in result.trades:
            writer.writerow([
                result.strategy, t.symbol, t.side,
                _ts_to_str(t.entry_ts) if t.entry_ts else "",
                _ts_to_str(t.exit_ts) if t.exit_ts else "",
                f"{t.entry_price:.6f}", f"{t.exit_price:.6f}",
                f"{t.qty:.6f}", f"{t.entry_fee:.6f}", f"{t.exit_fee:.6f}",
                f"{t.funding_fee:.6f}", f"{t.realized_pnl:.6f}", t.close_reason, t.bars_held,
            ])
    print(f"Trades saved to {path}")


def print_comparison_table(results: list[BacktestResult]) -> None:
    print(f"\n{'='*80}")
    print(f"  Strategy Comparison: {results[0].symbol if results else ''}")
    print(f"{'='*80}")
    header = f"  {'Strategy':<28}  {'N':>4}  {'WR':>6}  {'PF':>6}  {'NetPnL':>9}  {'MaxDD':>6}  {'Bars':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in sorted(results, key=lambda x: x.profit_factor, reverse=True):
        pf = r.profit_factor
        pf_str = f"{pf:.2f}" if pf != float("inf") else "  ∞"
        print(
            f"  {r.strategy:<28}  {r.total_trades:>4}  {r.win_rate:>5.1%}  "
            f"{pf_str:>6}  {r.net_pnl:>+9.4f}  {r.max_drawdown_pct:>5.1f}%  {r.avg_bars_held:>5.1f}"
        )
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Strategy loader
# ---------------------------------------------------------------------------

def _load_strategy(name: str, params: dict | None = None) -> Optional[BaseStrategy]:
    module_path = f"src.signal.strategies.{name}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        logger.warning("Strategy not found: %s", name)
        return None

    class_name = "".join(p.capitalize() for p in name.split("_")) + "Strategy"
    cls = getattr(module, class_name, None)
    if cls is None:
        logger.warning("Class %s not found in %s", class_name, module_path)
        return None
    try:
        return cls(params or {})
    except Exception as exc:
        logger.warning("Failed to instantiate %s: %s", class_name, exc)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Backtest a strategy on KRX historical data")
    parser.add_argument("--strategy", default="ema_crossover",
                        help="Strategy name (default: ema_crossover)")
    parser.add_argument("--symbol", default="005930.KS",
                        help="KRX ticker with suffix, e.g. '005930.KS' (default: Samsung)")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--balance", type=float, default=100.0,
                        help="Initial balance (default 100)")
    parser.add_argument("--risk-pct", type=float, default=0.01,
                        help="Risk per trade fraction (default 0.01 = 1%%)")
    parser.add_argument("--save-trades", metavar="FILE",
                        help="Save trade log to CSV file")
    parser.add_argument("--compare", action="store_true",
                        help="Run ALL strategies and print comparison table")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore local OHLCV cache, re-fetch from yfinance")
    args = parser.parse_args()

    if args.compare:
        results: list[BacktestResult] = []
        for strat_name in _ALL_STRATEGIES:
            strategy = _load_strategy(strat_name)
            if strategy is None:
                continue
            engine = BacktestEngine(
                strategy=strategy,
                symbol=args.symbol,
                initial_balance=args.balance,
                risk_pct=args.risk_pct,
            )
            timeframe = strategy.get_timeframe()
            try:
                df = engine.fetch_ohlcv(timeframe, args.start, args.end,
                                        use_cache=not args.no_cache)
                if len(df) < strategy.get_min_candles() + 2:
                    logger.warning("%s: insufficient data (%d candles)", strat_name, len(df))
                    continue
                result = engine.run(df)
                results.append(result)
                print(f"  {strat_name}: {result.total_trades} trades, PF={result.profit_factor:.2f}")
            except Exception as exc:
                logger.error("%s failed: %s", strat_name, exc)
        if results:
            print_comparison_table(results)
        return

    # Single strategy
    strategy = _load_strategy(args.strategy)
    if strategy is None:
        print(f"Strategy '{args.strategy}' not found.")
        return

    engine = BacktestEngine(
        strategy=strategy,
        symbol=args.symbol,
        initial_balance=args.balance,
        risk_pct=args.risk_pct,
    )
    timeframe = strategy.get_timeframe()
    df = engine.fetch_ohlcv(timeframe, args.start, args.end,
                             use_cache=not args.no_cache)

    if len(df) < strategy.get_min_candles() + 2:
        print(f"Insufficient data: {len(df)} candles (need {strategy.get_min_candles() + 2}+)")
        return

    result = engine.run(df)
    print_report(result)
    print_trade_log(result)

    if args.save_trades:
        save_trades_csv(result, args.save_trades)


if __name__ == "__main__":
    main()
