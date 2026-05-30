"""Trailing stop manager — ATR-based step trailing with DB persistence."""

from __future__ import annotations

import sqlite3

from src.monitoring.logger import get_logger

logger = get_logger("trailing_stop")


class TrailingStopManager:
    """Manage ATR-based trailing stop updates for open positions.

    Stop movement rules (long example; short is mirrored):
        profit >= 1 ATR  →  new_sl = entry_price            (breakeven)
        profit >= 2 ATR  →  new_sl = entry_price + 0.5 ATR
        profit >= 3 ATR  →  new_sl = entry_price + 1.5 ATR

    Stop may only move in a *favourable* direction — never lower for long,
    never higher for short.

    Args:
        conn: Open SQLite connection.  May be ``None`` in unit tests that
              do not need DB writes.
    """

    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self._conn = conn
        self._activated: set[str] = set()

    # ------------------------------------------------------------------
    # Activation registry
    # ------------------------------------------------------------------

    def activate(self, position_id: str) -> None:
        """Mark trailing stop as active for *position_id*."""
        self._activated.add(position_id)

    def is_activated(self, position_id: str) -> bool:
        """Return True if trailing stop has been activated for *position_id*."""
        return position_id in self._activated

    # ------------------------------------------------------------------
    # Core update logic
    # ------------------------------------------------------------------

    def update(
        self,
        position: dict,
        current_price: float,
        atr: float,
    ) -> float | None:
        """Evaluate and possibly advance the trailing stop.

        Args:
            position: Dict with keys:
                ``position_id`` str
                ``side``         'long' | 'short'
                ``entry_price``  float | str
                ``stop_loss``    float | str
            current_price: Latest market price.
            atr: Current ATR(14) value in price units.

        Returns:
            New stop-loss price if the stop was moved, else ``None``.
        """
        position_id = position["position_id"]
        side = position["side"]
        entry_price = float(position["entry_price"])
        current_sl = float(position["stop_loss"])

        new_sl = self._calc_new_sl(side, entry_price, current_price, current_sl, atr)

        if new_sl is None:
            return None

        logger.info(
            "Trailing stop moved [%s] %s: %.4f → %.4f",
            position_id,
            side,
            current_sl,
            new_sl,
        )

        # Activate if not already active
        self.activate(position_id)

        # Persist to DB
        if self._conn is not None:
            self._conn.execute(
                "UPDATE positions SET stop_loss = ?, trailing_activated = 1 WHERE position_id = ?",
                (str(new_sl), position_id),
            )
            self._conn.commit()

        return new_sl

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calc_new_sl(
        self,
        side: str,
        entry_price: float,
        current_price: float,
        current_sl: float,
        atr: float,
    ) -> float | None:
        """Return new SL or None if no movement warranted."""
        if side == "long":
            return self._calc_long_sl(entry_price, current_price, current_sl, atr)
        elif side == "short":
            return self._calc_short_sl(entry_price, current_price, current_sl, atr)
        else:
            raise ValueError(f"side must be 'long' or 'short', got {side!r}")

    @staticmethod
    def _calc_long_sl(
        entry: float,
        current: float,
        current_sl: float,
        atr: float,
    ) -> float | None:
        profit = current - entry

        if profit >= 3 * atr:
            candidate = entry + 1.5 * atr
        elif profit >= 2 * atr:
            candidate = entry + 0.5 * atr
        elif profit >= 1 * atr:
            candidate = entry  # breakeven
        else:
            return None

        # Only move favourably (upward for long)
        if candidate <= current_sl:
            return None

        return candidate

    @staticmethod
    def _calc_short_sl(
        entry: float,
        current: float,
        current_sl: float,
        atr: float,
    ) -> float | None:
        profit = entry - current  # short profits when price falls

        if profit >= 3 * atr:
            candidate = entry - 1.5 * atr
        elif profit >= 2 * atr:
            candidate = entry - 0.5 * atr
        elif profit >= 1 * atr:
            candidate = entry  # breakeven
        else:
            return None

        # Only move favourably (downward for short)
        if candidate >= current_sl:
            return None

        return candidate
