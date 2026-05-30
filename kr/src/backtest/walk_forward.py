"""Walk-forward optimization for all supported strategies.

For each rolling period:
  1. Optimize params on TRAIN_BARS via grid search (in-sample).
  2. Replay those exact params on TEST_BARS immediately after (out-of-sample).
  3. Step forward by STEP_BARS and repeat.

Reports per-period results and aggregate metrics vs the fixed (live) param baseline.

Default window sizes (1h bars):
  TRAIN_BARS = 2160   ~3 months
  TEST_BARS  = 720    ~1 month
  STEP_BARS  = 720    ~1 month  (non-overlapping test windows)

Usage:
    python -m src.backtest.walk_forward \\
        --strategy ema_crossover \\
        --symbol BTCUSDT \\
        [--years 2] [--train-bars 2160] [--test-bars 720] [--step-bars 720] \\
        [--balance 100] [--risk-pct 0.01] [--no-cache]
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from src.backtest.tune_strategies import (
    _STRATEGY_DEFAULTS,
    _STRATEGY_PARAM_KEYS,
    _load_ohlcv,
    find_params_in_results,
    run_grid,
)

logger = logging.getLogger(__name__)

_DEFAULT_TRAIN = 2160   # 3 months of 1h bars
_DEFAULT_TEST  = 720    # 1 month
_DEFAULT_STEP  = 720    # non-overlapping


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WFPeriod:
    period_idx: int
    train_start_ms: int
    train_end_ms: int
    test_start_ms: int
    test_end_ms: int
    best_params: dict
    train_pf: float
    train_n: int
    train_wr: float
    wf_pf: float          # out-of-sample PF with optimized params
    wf_n: int
    wf_wr: float
    fixed_pf: float       # out-of-sample PF with fixed live params
    fixed_n: int
    fixed_wr: float

    @property
    def wf_beats_fixed(self) -> bool:
        return self.wf_pf > self.fixed_pf


@dataclass
class WFResult:
    strategy: str
    symbol: str
    periods: list[WFPeriod] = field(default_factory=list)

    @property
    def n_periods(self) -> int:
        return len(self.periods)

    @property
    def avg_wf_pf(self) -> float:
        valid = [p.wf_pf for p in self.periods if p.wf_n > 0 and p.wf_pf != float("inf")]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def avg_fixed_pf(self) -> float:
        valid = [p.fixed_pf for p in self.periods if p.fixed_n > 0 and p.fixed_pf != float("inf")]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def wf_win_rate(self) -> float:
        # Only compare periods where both have finite PF and enough trades
        comparable = [p for p in self.periods
                      if p.wf_n > 0 and p.fixed_n > 0
                      and p.wf_pf != float("inf") and p.fixed_pf != float("inf")]
        if not comparable:
            return 0.0
        beats = sum(1 for p in comparable if p.wf_beats_fixed)
        return beats / len(comparable)

    @property
    def pf_edge(self) -> float:
        return self.avg_wf_pf - self.avg_fixed_pf


# ---------------------------------------------------------------------------
# Core walk-forward logic
# ---------------------------------------------------------------------------

def _ms_to_date(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def run(
    strategy: str,
    symbol: str,
    years: float = 2.0,
    initial_balance: float = 100.0,
    risk_pct: float = 0.01,
    train_bars: int = _DEFAULT_TRAIN,
    test_bars: int = _DEFAULT_TEST,
    step_bars: int = _DEFAULT_STEP,
    no_cache: bool = False,
) -> WFResult:
    """Run walk-forward optimization and return structured results.

    Args:
        strategy: One of the strategies supported by tune_strategies.run_grid.
        symbol: Trading pair, e.g. 'BTCUSDT'.
        years: How many years of 1h data to fetch.
        initial_balance: Starting balance for each sim window (USDT).
        risk_pct: Fraction of balance risked per trade.
        train_bars: In-sample window size (1h bars).
        test_bars: Out-of-sample window size (1h bars).
        step_bars: How far to advance each period.
        no_cache: Bypass CSV cache and re-fetch from exchange.
    """
    from datetime import timedelta
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = now_ms - int(years * 365 * 24 * 3600 * 1000)

    df = _load_ohlcv(symbol, since_ms, now_ms, no_cache=no_cache)
    logger.info("Loaded %d candles for %s", len(df), symbol)

    param_keys = _STRATEGY_PARAM_KEYS[strategy]
    fixed_params = _STRATEGY_DEFAULTS[strategy]
    result = WFResult(strategy=strategy, symbol=symbol)

    total_needed = train_bars + test_bars
    if len(df) < total_needed:
        logger.warning(
            "Insufficient data: %d bars, need at least %d (train+test). "
            "Fetch more history with --years.",
            len(df), total_needed,
        )
        return result

    period_idx = 0
    t = train_bars
    while t + test_bars <= len(df):
        train_df = df.iloc[t - train_bars:t].reset_index(drop=True)
        test_df  = df.iloc[t:t + test_bars].reset_index(drop=True)

        train_start_ms = int(df.iloc[t - train_bars]["timestamp"])
        train_end_ms   = int(df.iloc[t - 1]["timestamp"])
        test_start_ms  = int(df.iloc[t]["timestamp"])
        test_end_ms    = int(df.iloc[t + test_bars - 1]["timestamp"])

        logger.info(
            "Period %d: train=%s→%s test=%s→%s",
            period_idx + 1,
            _ms_to_date(train_start_ms), _ms_to_date(train_end_ms),
            _ms_to_date(test_start_ms), _ms_to_date(test_end_ms),
        )

        # 1. Optimize on training window
        train_top = run_grid(strategy, train_df, initial_balance, risk_pct, top_n=1)
        if not train_top:
            logger.warning("Period %d: no results from train grid — skipping", period_idx + 1)
            t += step_bars
            period_idx += 1
            continue

        best_params = {k: train_top[0][k] for k in param_keys}
        train_r = train_top[0]

        # 2. Evaluate on test window — run full grid once, then pick by params
        test_all = run_grid(strategy, test_df, initial_balance, risk_pct, top_n=9999)

        wf_r     = find_params_in_results(test_all, best_params, param_keys)
        fixed_r  = find_params_in_results(test_all, fixed_params, param_keys)

        result.periods.append(WFPeriod(
            period_idx   = period_idx,
            train_start_ms = train_start_ms,
            train_end_ms   = train_end_ms,
            test_start_ms  = test_start_ms,
            test_end_ms    = test_end_ms,
            best_params  = best_params,
            train_pf     = train_r["pf"],
            train_n      = train_r["n"],
            train_wr     = train_r["wr"],
            wf_pf        = wf_r["pf"]    if wf_r    else 0.0,
            wf_n         = wf_r["n"]     if wf_r    else 0,
            wf_wr        = wf_r["wr"]    if wf_r    else 0.0,
            fixed_pf     = fixed_r["pf"] if fixed_r else 0.0,
            fixed_n      = fixed_r["n"]  if fixed_r else 0,
            fixed_wr     = fixed_r["wr"] if fixed_r else 0.0,
        ))

        t += step_bars
        period_idx += 1

    return result


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(result: WFResult) -> None:
    """Print walk-forward results table and aggregate summary."""
    sep   = "═" * 108
    dash  = "─" * 108
    strat = result.strategy
    sym   = result.symbol
    param_keys = _STRATEGY_PARAM_KEYS.get(strat, [])

    print(f"\n{sep}")
    print(f"  Walk-Forward Optimization — {strat} on {sym}")
    print(f"  {len(result.periods)} periods | "
          f"avg WF PF: {result.avg_wf_pf:.2f} | "
          f"avg Fixed PF: {result.avg_fixed_pf:.2f} | "
          f"edge: {result.pf_edge:+.2f} | "
          f"WF>Fixed: {result.wf_win_rate:.0%} periods")
    print(sep)

    param_hdr = "  ".join(f"{k[:5]:>5}" for k in param_keys)
    hdr = (f"  {'#':>2}  {'Train':>10}  {'Test':>10}  "
           f"{param_hdr}  "
           f"{'TrN':>4}  {'TrPF':>6}  {'WFN':>4}  {'WFPF':>6}  "
           f"{'FxN':>4}  {'FxPF':>6}  {'△':>6}")
    print(hdr)
    print(dash)

    for p in result.periods:
        param_vals = "  ".join(f"{p.best_params[k]:>5.1f}" for k in param_keys)
        wf_pf_str    = f"{p.wf_pf:.2f}"    if p.wf_pf    != float("inf") else "  ∞"
        fixed_pf_str = f"{p.fixed_pf:.2f}" if p.fixed_pf != float("inf") else "  ∞"
        delta = p.wf_pf - p.fixed_pf
        beat  = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
        print(
            f"  {p.period_idx + 1:>2}  "
            f"{_ms_to_date(p.train_start_ms):>10}  "
            f"{_ms_to_date(p.test_start_ms):>10}  "
            f"{param_vals}  "
            f"{p.train_n:>4}  {p.train_pf:>6.2f}  "
            f"{p.wf_n:>4}  {wf_pf_str:>6}  "
            f"{p.fixed_n:>4}  {fixed_pf_str:>6}  "
            f"{beat}{abs(delta):>5.2f}"
        )

    print(sep)

    if result.periods:
        # Most-frequently chosen params across all periods
        from collections import Counter
        param_counter: Counter = Counter()
        for p in result.periods:
            param_counter[tuple(sorted(p.best_params.items()))] += 1
        top_entry = param_counter.most_common(1)[0]
        most_common_params = dict(top_entry[0])
        top_count = top_entry[1]
        print(f"\n  Most-chosen params ({top_count}/{len(result.periods)} periods):")
        for k, v in most_common_params.items():
            print(f"    {k} = {v}")

        if result.pf_edge > 0.05:
            import json
            env_key = f"STRATEGY_PARAMS_{strat.upper()}"
            env_val = json.dumps(most_common_params, separators=(",", ":"))
            print(f"\n  Suggested ConfigMap update (edge >{0.05:.0%} over fixed):")
            print(f"    {env_key}='{env_val}'")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

    from src.backtest.tune_strategies import _RUNNERS  # noqa: PLC0415
    parser = argparse.ArgumentParser(description="Walk-forward optimization for active strategies")
    parser.add_argument("--strategy",    required=True, choices=list(_RUNNERS.keys()))
    parser.add_argument("--symbol",      default="BTCUSDT")
    parser.add_argument("--years",       type=float, default=2.0)
    parser.add_argument("--train-bars",  type=int,   default=_DEFAULT_TRAIN)
    parser.add_argument("--test-bars",   type=int,   default=_DEFAULT_TEST)
    parser.add_argument("--step-bars",   type=int,   default=_DEFAULT_STEP)
    parser.add_argument("--balance",     type=float, default=100.0)
    parser.add_argument("--risk-pct",    type=float, default=0.01)
    parser.add_argument("--no-cache",    action="store_true")
    args = parser.parse_args()

    result = run(
        strategy      = args.strategy,
        symbol        = args.symbol,
        years         = args.years,
        initial_balance = args.balance,
        risk_pct      = args.risk_pct,
        train_bars    = args.train_bars,
        test_bars     = args.test_bars,
        step_bars     = args.step_bars,
        no_cache      = args.no_cache,
    )
    print_report(result)


if __name__ == "__main__":
    main()
