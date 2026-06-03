"""Performance tracker — backfills exit PnL and computes strategy stats.

Can be imported by other modules or run as a standalone script::

    python -m src.jobs.performance_tracker

Responsibilities:
  - Backfill ``exit_price`` / ``realized_pnl`` for closed positions that
    have no data (e.g., externally closed before fill_listener captured them).
  - Compute win rate, profit factor, avg PnL from closed positions.
  - Roll up aggregates into ``daily_performance``.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_performance_stats(conn: sqlite3.Connection, trading_mode: str | None = None) -> dict:
    """Compute strategy performance metrics from closed positions.

    Returns a dict with keys: total_trades, win_rate, net_pnl, profit_factor, etc.
    Returns ``{'total_trades': 0}`` when no data is available.
    """
    where_clauses = ["status='closed'", "realized_pnl IS NOT NULL", "realized_pnl != '0'"]
    params: list = []
    if trading_mode:
        where_clauses.append("trading_mode=?")
        params.append(trading_mode)

    where = " AND ".join(where_clauses)
    rows = conn.execute(
        f"SELECT realized_pnl, close_reason, symbol FROM positions WHERE {where}",
        params,
    ).fetchall()

    records: list[tuple[float, str, str]] = []
    for r in rows:
        try:
            pnl    = float(r["realized_pnl"] if hasattr(r, "keys") else r[0])
            reason = (r["close_reason"] if hasattr(r, "keys") else r[1]) or ""
            sym    = (r["symbol"] if hasattr(r, "keys") else r[2]) or ""
            records.append((pnl, reason, sym))
        except (TypeError, ValueError):
            pass

    if not records:
        return {"total_trades": 0}

    total = len(records)
    wins  = [p for p, _, _ in records if p > 0]
    losses = [p for p, _, _ in records if p < 0]
    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    net_pnl      = sum(p for p, _, _ in records)

    return {
        "total_trades":   total,
        "winning_trades": len(wins),
        "losing_trades":  len(losses),
        "win_rate":       len(wins) / total,
        "gross_profit":   gross_profit,
        "gross_loss":     gross_loss,
        "net_pnl":        net_pnl,
        "profit_factor":  gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "avg_pnl":        net_pnl / total,
        "best_trade":     max(p for p, _, _ in records),
        "worst_trade":    min(p for p, _, _ in records),
        "close_reasons":  {
            reason: sum(1 for _, r, _ in records if r == reason)
            for reason in {r for _, r, _ in records}
        },
    }


def roll_up_daily_performance(conn: sqlite3.Connection) -> None:
    """Aggregate today's closed positions into daily_performance.

    Upserts a row for today's date using all closed positions from today.
    """
    today = datetime.now(timezone.utc).date().isoformat()

    rows = conn.execute(
        "SELECT realized_pnl, close_reason, trading_mode"
        " FROM positions WHERE status='closed' AND date(closed_at)=?",
        (today,),
    ).fetchall()

    if not rows:
        return

    by_mode: dict[str, list] = {}
    for r in rows:
        mode   = (r["trading_mode"] if hasattr(r, "keys") else r[2]) or "live"
        pnl    = float((r["realized_pnl"] if hasattr(r, "keys") else r[0]) or 0)
        reason = (r["close_reason"] if hasattr(r, "keys") else r[1]) or ""
        by_mode.setdefault(mode, []).append((pnl, reason))

    for mode, records in by_mode.items():
        total      = len(records)
        wins       = sum(1 for p, _ in records if p > 0)
        losses     = sum(1 for p, _ in records if p < 0)
        liquidated = sum(1 for _, r in records if r == "liquidated")
        gross_p    = sum(p for p, _ in records if p > 0)
        gross_l    = abs(sum(p for p, _ in records if p < 0))
        net_pnl    = sum(p for p, _ in records)
        win_rate   = wins / total if total else 0

        existing = conn.execute(
            "SELECT 1 FROM daily_performance WHERE perf_date=? AND trading_mode=?",
            (today, mode),
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE daily_performance SET
                     total_trades=?, winning_trades=?, losing_trades=?, liquidated_trades=?,
                     gross_profit=?, gross_loss=?, net_pnl=?, win_rate=?
                   WHERE perf_date=? AND trading_mode=?""",
                (total, wins, losses, liquidated,
                 str(gross_p), str(gross_l), str(net_pnl), str(win_rate),
                 today, mode),
            )
        else:
            conn.execute(
                """INSERT INTO daily_performance
                   (perf_date, trading_mode, total_trades, winning_trades, losing_trades,
                    liquidated_trades, gross_profit, gross_loss, net_pnl, win_rate, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (today, mode, total, wins, losses, liquidated,
                 str(gross_p), str(gross_l), str(net_pnl), str(win_rate),
                 datetime.now(timezone.utc).isoformat()),
            )
    conn.commit()
    logger.info("daily_performance rolled up for %s", today)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(conn: sqlite3.Connection, trading_mode: str | None = None) -> None:
    """Print human-readable strategy performance to stdout."""
    stats = get_performance_stats(conn, trading_mode)
    if stats["total_trades"] == 0:
        print("No closed trades with PnL data. Run backfill first.")
        return

    pf = stats["profit_factor"]
    pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"

    print(f"\n{'='*42}")
    print(f"  Strategy Performance")
    if trading_mode:
        print(f"  Mode: {trading_mode}")
    print(f"{'='*42}")
    print(f"  Trades:          {stats['total_trades']}")
    print(f"  Win rate:        {stats['win_rate']:.1%}  ({stats['winning_trades']}W / {stats['losing_trades']}L)")
    print(f"  Net PnL:         {stats['net_pnl']:+.4f} USDT")
    print(f"  Gross profit:    {stats['gross_profit']:.4f} USDT")
    print(f"  Gross loss:      {stats['gross_loss']:.4f} USDT")
    print(f"  Profit factor:   {pf_str}")
    print(f"  Avg PnL/trade:   {stats['avg_pnl']:+.4f} USDT")
    print(f"  Best:            {stats['best_trade']:+.4f} USDT")
    print(f"  Worst:           {stats['worst_trade']:+.4f} USDT")
    print(f"  Close reasons:   {stats['close_reasons']}")
    print(f"{'='*42}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    db_path = os.environ.get("SQLITE_DB_PATH", "/data/trading.db")
    mode    = os.environ.get("TRADING_MODE", "live")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    roll_up_daily_performance(conn)

    print_report(conn, trading_mode=mode)


if __name__ == "__main__":
    main()
