"""Signal generation engine — main loop with per-symbol async processing.

SignalEngine ties together strategy execution, liquidation validation, and
DB persistence into a single runnable component.
"""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.db.models import get_klines, insert_signal
from src.execution.order_manager import round_to_tick
from src.monitoring.logger import get_logger
from src.signal.base_strategy import SignalResult

if TYPE_CHECKING:
    from src.execution.order_manager import OrderManager
    from src.ingest.kis_rest import KISRestClient
    from src.signal.strategy_runner import StrategyRunner

logger = get_logger("signal_engine")

# Seconds between strategy runs per timeframe label
_TIMEFRAME_INTERVALS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

_DEFAULT_CANDLE_LIMIT = 250  # macd_sma200_chartart needs 219; 250 gives headroom
_CORR_LOOKBACK = 22   # 22 closes → 21 returns; enough for a stable 1h correlation
_CORR_THRESHOLD = 0.7 # block same-direction entry when correlation >= this


def _rows_to_df(rows: list) -> pd.DataFrame:
    """Convert sqlite3.Row list from get_klines to a pandas DataFrame."""
    data = [
        {
            "open_time": row["open_time"],
            "open":      float(row["open"]),
            "high":      float(row["high"]),
            "low":       float(row["low"]),
            "close":     float(row["close"]),
            "volume":    float(row["volume"]),
        }
        for row in rows
    ]
    df = pd.DataFrame(data)
    if not df.empty:
        df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit="ms", utc=True)
        df.set_index("open_time", inplace=True)
    return df


class SignalEngine:
    """Orchestrates signal generation for all active KRX symbols.

    Args:
        conn: Open SQLite connection.
        strategy_runner: :class:`~src.signal.strategy_runner.StrategyRunner`
            with ``run(df, symbol) -> SignalResult``.
        order_manager: :class:`~src.execution.order_manager.OrderManager`
            (async KIS wrapper). None = signal-only mode (no orders placed).
        kis: KISRestClient for balance fetch. Falls back to DB cache when None.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        strategy_runner: "StrategyRunner",
        order_manager: "OrderManager | None" = None,
        kis: "KISRestClient | None" = None,
    ) -> None:
        self._conn = conn
        self._strategy_runner = strategy_runner
        self._order_manager = order_manager
        self._kis = kis

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_forever(self) -> None:
        """Block and run the main signal loop indefinitely.

        Schedules :meth:`process_all_symbols` on the timeframe interval.
        Catches and logs all exceptions to keep the loop alive.
        """
        asyncio.run(self._async_run_forever())

    async def process_all_symbols(self) -> list[SignalResult]:
        """Process all active symbols concurrently.

        Returns:
            List of :class:`SignalResult` objects (one per symbol that
            returned a result; symbols with no candles are excluded).
        """
        symbols = self._get_active_symbols()
        if not symbols:
            logger.warning("No active symbols found in DB")
            return []

        tasks = [self.process_symbol(s) for s in symbols]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        results: list[SignalResult] = []
        for symbol, outcome in zip(symbols, raw):
            if isinstance(outcome, Exception):
                logger.error("process_symbol error [%s]: %s", symbol, outcome, exc_info=False)
            elif outcome is not None:
                results.append(outcome)

        return results

    async def process_symbol(self, symbol: str) -> SignalResult | None:
        """Generate (and persist) a signal for a single symbol.

        Flow:
            1. Load recent daily candles from DB.
            2. Run strategy → SignalResult.
            3. Open position? Skip or exit on reversal.
            4. No position: check correlation, persist, execute entry.

        Returns:
            :class:`SignalResult` or ``None`` when no candles are available.
        """
        interval = self._get_timeframe()
        rows = get_klines(self._conn, symbol, interval, limit=_DEFAULT_CANDLE_LIMIT)
        if not rows:
            logger.debug("No candles for %s/%s — skipping", symbol, interval)
            return None

        df = _rows_to_df(rows)
        result: SignalResult = self._strategy_runner.run(df, symbol)

        open_pos = self._conn.execute(
            "SELECT side, quantity, entry_price FROM positions"
            " WHERE symbol=? AND status='open' LIMIT 1",
            (symbol,),
        ).fetchone()

        if open_pos is not None:
            pos_side = open_pos["side"] if hasattr(open_pos, "keys") else open_pos[0]
            if result.is_actionable() and result.signal_type == "long":
                if pos_side == "long":
                    logger.debug("Already long [%s] — skip", symbol)
                # KRX long-only: "short" signal from strategy = exit signal
            if result.is_actionable() and result.signal_type in ("short", "close"):
                await self._execute_exit(symbol, open_pos)
            return result

        # No open position — long signal only (KRX spot, no shorting)
        if result.is_actionable() and result.signal_type == "long":
            corr_blocked, corr_reason = self._is_correlated_with_open(symbol, "long")
            if corr_blocked:
                logger.warning("Signal blocked [%s]: %s", symbol, corr_reason)
                return result
            await self._persist_signal(symbol, result)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _async_run_forever(self) -> None:
        interval_secs = _TIMEFRAME_INTERVALS.get(self._get_timeframe(), 60)
        logger.info(
            "SignalEngine started — timeframe=%s interval=%ds",
            self._get_timeframe(),
            interval_secs,
        )
        while True:
            try:
                reload_fn = getattr(self._strategy_runner, "reload_if_changed", None)
                if callable(reload_fn):
                    await asyncio.to_thread(reload_fn)
                results = await self.process_all_symbols()
                actionable = sum(1 for r in results if r.is_actionable())
                logger.info(
                    "Cycle complete — %d symbols processed, %d actionable signals",
                    len(results),
                    actionable,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Unhandled error in signal loop: %s", exc, exc_info=True)
            await asyncio.sleep(interval_secs)

    def _get_timeframe(self) -> str:
        """Return timeframe string from strategy_runner if available, else '1m'."""
        getter = getattr(self._strategy_runner, "get_timeframe", None)
        if callable(getter):
            return getter()
        return "1m"

    def _get_active_symbols(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT symbol FROM symbols WHERE is_active = 1"
        ).fetchall()
        return [
            (r[0] if isinstance(r, (list, tuple)) else r["symbol"])
            for r in rows
        ]

    def _get_recent_closes(self, symbol: str) -> list[float]:
        """Return last _CORR_LOOKBACK 1h closes for *symbol*, oldest first."""
        rows = self._conn.execute(
            "SELECT close FROM klines WHERE symbol=? AND interval_type='1h'"
            " ORDER BY open_time DESC LIMIT ?",
            (symbol, _CORR_LOOKBACK),
        ).fetchall()
        return [float(r[0]) for r in reversed(rows)]

    def _is_correlated_with_open(self, symbol: str, signal_type: str) -> tuple[bool, str]:
        """Block if *symbol* is highly correlated with an open same-direction position.

        Computes Pearson correlation of 1h % returns over the last _CORR_LOOKBACK bars.
        Fails open (returns False) when klines are insufficient.
        Only blocks same-direction pairs (long↔long, short↔short); hedges are allowed.
        """
        open_positions = self._conn.execute(
            "SELECT symbol, side FROM positions WHERE status='open'",
        ).fetchall()
        if not open_positions:
            return False, ""

        new_closes = self._get_recent_closes(symbol)
        if len(new_closes) < _CORR_LOOKBACK - 1:
            return False, ""

        new_ret = np.diff(new_closes) / np.array(new_closes[:-1])

        for pos in open_positions:
            pos_sym  = pos["symbol"] if hasattr(pos, "keys") else pos[0]
            pos_side = pos["side"]   if hasattr(pos, "keys") else pos[1]
            if pos_sym == symbol:
                continue
            if pos_side != signal_type:
                continue  # opposite direction = hedge; allow

            pos_closes = self._get_recent_closes(pos_sym)
            if len(pos_closes) < _CORR_LOOKBACK - 1:
                continue  # not enough data; fail open

            pos_ret = np.diff(pos_closes) / np.array(pos_closes[:-1])
            n = min(len(new_ret), len(pos_ret))
            if n < 10:
                continue

            corr_matrix = np.corrcoef(new_ret[-n:], pos_ret[-n:])
            corr = float(corr_matrix[0, 1])
            if np.isnan(corr):
                continue

            if corr >= _CORR_THRESHOLD:
                return (
                    True,
                    f"correlation: {symbol}↔{pos_sym} r={corr:.2f}>={_CORR_THRESHOLD}"
                    f" (both {signal_type})",
                )

        return False, ""

    async def _persist_signal(self, symbol: str, result: SignalResult) -> None:
        """Validate, persist, and execute a long signal for KRX spot trading."""
        from src.utils.config import load_config  # lazy
        config = load_config()

        # Balance: KIS REST → available KRW
        balance_krw = 0.0
        if self._kis is not None:
            try:
                raw = await self._kis.fetch_account_balance()
                balance_krw = float(raw.get("availableBalance", 0) or 0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("KIS balance fetch failed: %s", exc)
        if balance_krw <= 0:
            from src.utils.startup_recovery import get_cached_balance  # lazy
            cached = get_cached_balance()
            balance_krw = float(cached.get("availableBalance", 0) or 0)

        strategy_name = getattr(self._strategy_runner, "get_symbol_strategy_name",
                                lambda s: None)(symbol) or \
                        getattr(self._strategy_runner, "get_active_strategy_name",
                                lambda: "unknown")()

        entry_price = result.entry_price or 0.0
        stop_loss = result.sl or 0.0

        # Block check: need valid entry + SL + available balance
        blocked = False
        block_reason: str | None = None
        if entry_price <= 0 or stop_loss <= 0:
            blocked, block_reason = True, "missing entry or SL price"
        elif entry_price <= stop_loss:
            blocked, block_reason = True, f"entry {entry_price} <= sl {stop_loss}"
        elif balance_krw <= 0:
            blocked, block_reason = True, "zero KRW balance"
        else:
            # Max positions guard
            open_count = self._conn.execute(
                "SELECT COUNT(*) FROM positions WHERE status='open'"
            ).fetchone()[0]
            if open_count >= config.max_positions:
                blocked, block_reason = True, f"max_positions={config.max_positions} reached"

        signal_record = {
            "symbol":          symbol,
            "signal_type":     result.signal_type,
            "strategy_name":   strategy_name,
            "strength_score":  result.strength_score,
            "entry_price":     str(entry_price) if entry_price else None,
            "tp_price":        str(result.tp1) if result.tp1 else None,
            "sl_price":        str(stop_loss) if stop_loss else None,
            "indicators_json": result.indicators,
            "blocked":         blocked,
            "block_reason":    block_reason,
        }
        try:
            insert_signal(self._conn, signal_record)
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to persist signal [%s]: %s", symbol, exc)
            return

        if blocked:
            logger.warning("Signal blocked [%s]: %s", symbol, block_reason)
            return

        logger.info("Signal saved [%s]: %s strength=%d", symbol, result.signal_type,
                    result.strength_score)
        await self._execute_signal(symbol, result, balance_krw, config, strategy_name)

    async def _execute_exit(self, symbol: str, position_row) -> None:
        """Close an open position on strategy exit signal (e.g. close/short).

        KRX spot: always a market sell. Safety monitor handles SL/TP;
        this handles strategy-driven exits (trend reversal, etc.).
        """
        if self._order_manager is None:
            logger.warning("No OrderManager — strategy exit skipped [%s]", symbol)
            return

        qty = int(float(
            position_row["quantity"] if hasattr(position_row, "keys") else position_row[1]
        ))
        if qty <= 0:
            return

        try:
            await self._order_manager.market_close(symbol, qty)
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "UPDATE positions SET status='closed', close_reason='strategy_exit',"
                " closed_at=? WHERE symbol=? AND status='open'",
                (now, symbol),
            )
            self._conn.commit()
            logger.info("strategy_exit [%s] qty=%d", symbol, qty)
        except Exception as exc:  # noqa: BLE001
            logger.error("strategy_exit failed [%s]: %s", symbol, exc)

    async def _execute_signal(
        self, symbol: str, result: SignalResult,
        balance_krw: float, config, strategy_name: str,
    ) -> None:
        """Size and submit a KRX buy order with SL/TP attached.

        No leverage, no futures-specific params. SL/TP enforcement delegated
        to SafetyMonitor price monitoring loop (kis_ws real-time prices).
        """
        if self._order_manager is None:
            logger.warning("No OrderManager — signal not executed [%s]", symbol)
            return

        from src.risk.position_sizer import kelly_risk_pct, krw_position_size  # lazy

        entry_price = round_to_tick(result.entry_price or 0)
        sl_price    = round_to_tick(result.sl or 0)
        tp1_price   = round_to_tick(result.tp1) if result.tp1 else None
        tp2_price   = round_to_tick(result.tp2) if result.tp2 else None

        if entry_price <= 0 or sl_price <= 0 or entry_price <= sl_price:
            logger.warning("Invalid SL/entry [%s]: entry=%d sl=%d", symbol, entry_price, sl_price)
            return

        risk_pct = kelly_risk_pct(self._conn, strategy_name, config.risk_per_trade)
        quantity = krw_position_size(balance_krw, risk_pct, entry_price, sl_price)
        if quantity <= 0:
            logger.warning("Position size zero [%s] — skipping", symbol)
            return

        logger.info(
            "signal.execute [%s] entry=%d sl=%d tp1=%s tp2=%s qty=%d risk_pct=%.4f",
            symbol, entry_price, sl_price, tp1_price, tp2_price, quantity, risk_pct,
        )

        try:
            order = {
                "symbol":        symbol,
                "side":          "buy",
                "quantity":      quantity,
                "price":         entry_price,   # limit order
                "sl":            sl_price,
                "tp1":           tp1_price,
                "tp2":           tp2_price,
                "strategy_name": strategy_name,
            }
            result_dict = await self._order_manager.submit_and_confirm(order)
            logger.info("signal.filled [%s]: position_id=%s",
                        symbol, result_dict.get("position_id", "?"))
        except Exception as exc:  # noqa: BLE001
            logger.error("signal.execute_failed [%s]: %s", symbol, exc)
