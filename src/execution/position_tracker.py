"""Position lifecycle tracking — PnL updates, SL/TP hit detection, close handling.

All methods are static and accept an open SQLite connection so callers control
transaction boundaries.  No in-memory state is kept; the DB is the single source
of truth.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)


def _open_positions(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM positions WHERE status='open'"
    ).fetchall()


class PositionTracker:
    """Static utility class for position lifecycle operations.

    All methods take a ``conn`` argument rather than storing it as instance
    state, allowing the same instance to be reused across different connections
    (e.g. tests with in-memory DBs).
    """

    # ------------------------------------------------------------------
    # Unrealized PnL
    # ------------------------------------------------------------------

    @staticmethod
    def update_unrealized_pnl(
        conn: sqlite3.Connection,
        current_prices: dict[str, float],
    ) -> None:
        """Recalculate and persist unrealized PnL for every open position.

        PnL formula (KRX spot, long-only):
            long:  (current_price - entry_price) * quantity
            short: (entry_price - current_price) * quantity

        Positions whose symbol is not in *current_prices* are skipped.

        Args:
            conn: Open SQLite connection.
            current_prices: Mapping of symbol → latest mark price.
        """
        rows = _open_positions(conn)
        for row in rows:
            symbol = row["symbol"]
            price = current_prices.get(symbol)
            if price is None:
                continue

            entry = float(row["entry_price"])
            qty = float(row["quantity"])
            side = row["side"]

            if side == "long":
                upnl = (price - entry) * qty
            else:
                upnl = (entry - price) * qty

            conn.execute(
                "UPDATE positions SET unrealized_pnl=? WHERE position_id=?",
                (str(upnl), row["position_id"]),
            )

        conn.commit()
        logger.debug("Unrealized PnL updated for %d open positions", len(rows))

    # ------------------------------------------------------------------
    # SL / TP hit detection
    # ------------------------------------------------------------------

    @staticmethod
    def check_sl_tp_hit(
        conn: sqlite3.Connection,
        current_prices: dict[str, float],
    ) -> list[dict]:
        """Return open positions that have reached SL or TP.

        Hit conditions:
            Long:
                SL hit  → current_price <= stop_loss
                TP1 hit → current_price >= take_profit_1  (if set)
                TP2 hit → current_price >= take_profit_2  (if set)
            Short:
                SL hit  → current_price >= stop_loss
                TP1 hit → current_price <= take_profit_1  (if set)
                TP2 hit → current_price <= take_profit_2  (if set)

        Args:
            conn: Open SQLite connection.
            current_prices: Mapping of symbol → latest mark price.

        Returns:
            List of dicts, each with keys:
                ``position_id``, ``symbol``, ``side``, ``trigger``,
                ``current_price``, ``trigger_price``.
        """
        rows = _open_positions(conn)
        hits: list[dict] = []

        for row in rows:
            symbol = row["symbol"]
            price = current_prices.get(symbol)
            if price is None:
                continue

            side = row["side"]
            sl = float(row["stop_loss"])
            tp1 = float(row["take_profit_1"]) if row["take_profit_1"] else None
            tp2 = float(row["take_profit_2"]) if row["take_profit_2"] else None

            trigger: str | None = None
            trigger_price: float | None = None

            if side == "long":
                if price <= sl:
                    trigger, trigger_price = "sl", sl
                elif tp2 is not None and price >= tp2:
                    trigger, trigger_price = "tp2", tp2
                elif tp1 is not None and price >= tp1:
                    trigger, trigger_price = "tp1", tp1
            else:  # short
                if price >= sl:
                    trigger, trigger_price = "sl", sl
                elif tp2 is not None and price <= tp2:
                    trigger, trigger_price = "tp2", tp2
                elif tp1 is not None and price <= tp1:
                    trigger, trigger_price = "tp1", tp1

            if trigger is not None:
                hits.append({
                    "position_id": row["position_id"],
                    "symbol": symbol,
                    "side": side,
                    "trigger": trigger,
                    "current_price": price,
                    "trigger_price": trigger_price,
                })

        return hits

    # ------------------------------------------------------------------
    # Close position
    # ------------------------------------------------------------------

    @staticmethod
    def close_position(
        conn: sqlite3.Connection,
        position_id: str,
        exit_price: float,
        reason: str,
    ) -> None:
        """Mark a position as closed and update daily_performance aggregates.

        Realized PnL formula:
            long:  (exit_price - entry_price) * quantity
            short: (entry_price - exit_price) * quantity

        Daily performance updates (UPSERT):
            - total_trades + 1
            - winning_trades or losing_trades + 1
            - gross_profit or gross_loss accumulated
            - net_pnl accumulated

        Args:
            conn: Open SQLite connection.
            position_id: Primary key of the position to close.
            exit_price: Actual exit/fill price.
            reason: Human-readable close reason (e.g. ``'sl_hit'``, ``'tp1_hit'``).
        """
        row = conn.execute(
            "SELECT * FROM positions WHERE position_id=?", (position_id,)
        ).fetchone()
        if row is None:
            logger.warning("close_position: position_id not found: %s", position_id)
            return

        entry = float(row["entry_price"])
        qty = float(row["quantity"])
        side = row["side"]
        trading_mode = row["trading_mode"]

        if side == "long":
            realized_pnl = (exit_price - entry) * qty
        else:
            realized_pnl = (entry - exit_price) * qty

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            UPDATE positions
            SET status='closed',
                exit_price=?,
                realized_pnl=?,
                unrealized_pnl='0',
                close_reason=?,
                closed_at=?
            WHERE position_id=?
            """,
            (str(exit_price), str(realized_pnl), reason, now, position_id),
        )

        # Aggregate into daily_performance
        today = date.today().isoformat()
        is_win = realized_pnl > 0
        gross_profit_delta = realized_pnl if is_win else 0.0
        gross_loss_delta = abs(realized_pnl) if not is_win else 0.0

        conn.execute(
            """
            INSERT INTO daily_performance (perf_date, trading_mode,
                total_trades, winning_trades, losing_trades,
                gross_profit, gross_loss, net_pnl)
            VALUES (?, ?, 1, ?, ?, ?, ?, ?)
            ON CONFLICT(perf_date, trading_mode) DO UPDATE SET
                total_trades   = total_trades + 1,
                winning_trades = winning_trades + excluded.winning_trades,
                losing_trades  = losing_trades  + excluded.losing_trades,
                gross_profit   = CAST(gross_profit AS REAL) + excluded.gross_profit,
                gross_loss     = CAST(gross_loss   AS REAL) + excluded.gross_loss,
                net_pnl        = CAST(net_pnl      AS REAL) + excluded.net_pnl
            """,
            (
                today, trading_mode,
                1 if is_win else 0,
                0 if is_win else 1,
                str(gross_profit_delta),
                str(gross_loss_delta),
                str(realized_pnl),
            ),
        )
        conn.commit()
        logger.info(
            "Position closed: %s exit=%.2f pnl=%.4f reason=%s",
            position_id, exit_price, realized_pnl, reason,
        )

    # ------------------------------------------------------------------
    # Total exposure
    # ------------------------------------------------------------------

    @staticmethod
    def get_total_exposure(conn: sqlite3.Connection) -> float:
        """Return total notional value of all open positions (leverage included).

        Notional value = entry_price × quantity × leverage

        Args:
            conn: Open SQLite connection.

        Returns:
            Sum of notional values in USDT.
        """
        rows = _open_positions(conn)
        total = 0.0
        for row in rows:
            entry = float(row["entry_price"])
            qty = float(row["quantity"])
            lev = int(row["leverage"])
            total += entry * qty * lev
        return total
