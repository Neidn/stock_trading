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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backfill helpers
# ---------------------------------------------------------------------------

def _parse_ts_ms(ts_str: str | None) -> int:
    """Parse ISO datetime string → milliseconds epoch. Default: 24h ago."""
    if not ts_str:
        return int(datetime.now(timezone.utc).timestamp() * 1000) - 86_400_000
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return int(datetime.now(timezone.utc).timestamp() * 1000) - 86_400_000


def _fetch_realized_pnl(exchange, db_symbol: str, since_ms: int) -> float:
    """Fetch REALIZED_PNL income events from Binance for *db_symbol* since *since_ms*.

    Uses the raw Binance Futures API endpoint (fapiPrivateGetIncome) because
    ccxt does not expose fetch_income on binanceusdm.
    """
    try:
        events = exchange.fapiPrivateGetIncome({
            "incomeType": "REALIZED_PNL",
            "symbol": db_symbol,
            "startTime": since_ms,
            "limit": 100,
        })
        return sum(float(e.get("income", 0)) for e in events if e.get("symbol") == db_symbol)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fapiPrivateGetIncome failed [%s]: %s", db_symbol, exc)
        return 0.0


def backfill_closed_positions(exchange, conn: sqlite3.Connection) -> int:
    """Fetch realized_pnl from Binance for positions closed without PnL data.

    For each position WHERE status='closed' AND exit_price IS NULL:
      1. Calls Binance income API for REALIZED_PNL events.
      2. Derives approximate exit_price from PnL + entry math.
      3. Updates DB.

    Returns:
        Number of positions updated.
    """
    rows = conn.execute(
        "SELECT position_id, symbol, side, entry_price, quantity, opened_at, closed_at"
        " FROM positions WHERE status='closed' AND exit_price IS NULL"
    ).fetchall()

    if not rows:
        logger.info("backfill: nothing to update")
        return 0

    updated = 0
    for row in rows:
        if hasattr(row, "keys"):
            pos_id    = row["position_id"]
            symbol    = row["symbol"]
            side      = row["side"]
            entry_str = row["entry_price"]
            qty_str   = row["quantity"]
            closed_at = row["closed_at"]
        else:
            pos_id, symbol, side, entry_str, qty_str, _, closed_at = row

        # Fetch since 2h before close to catch the income event
        since_ms = _parse_ts_ms(closed_at) - 7_200_000

        realized_pnl = _fetch_realized_pnl(exchange, symbol, since_ms)

        exit_price: float | None = None
        try:
            entry = float(entry_str or 0)
            qty   = float(qty_str or 0)
            if realized_pnl != 0 and qty > 0 and entry > 0:
                # Derive approximate exit from PnL math
                if side == "long":
                    exit_price = entry + realized_pnl / qty
                else:
                    exit_price = entry - realized_pnl / qty
        except (TypeError, ValueError):
            pass

        if realized_pnl != 0 or exit_price is not None:
            conn.execute(
                "UPDATE positions SET exit_price=?, realized_pnl=? WHERE position_id=?",
                (
                    str(exit_price) if exit_price is not None else None,
                    str(realized_pnl),
                    pos_id,
                ),
            )
            updated += 1
            logger.info("Backfilled [%s]: exit=%s pnl=%s", symbol, exit_price, realized_pnl)

    if updated:
        conn.commit()

    return updated


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

    import ccxt
    exchange = ccxt.binanceusdm({
        "apiKey":  os.environ["BINANCE_API_KEY"],
        "secret":  os.environ["BINANCE_API_SECRET"],
        "options": {"defaultType": "future"},
    })
    if mode == "testnet":
        exchange.set_sandbox_mode(True)

    print(f"Backfilling closed positions from Binance...")
    n = backfill_closed_positions(exchange, conn)
    print(f"Updated {n} position(s).")

    roll_up_daily_performance(conn)

    print_report(conn, trading_mode=mode)


if __name__ == "__main__":
    main()
