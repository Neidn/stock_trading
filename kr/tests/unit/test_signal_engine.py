"""Unit tests for KRX SignalEngine."""

from __future__ import annotations

import asyncio
import sqlite3
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from src.signal.base_strategy import SignalResult
from src.signal.signal_engine import SignalEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS symbols (
            symbol      TEXT PRIMARY KEY,
            base_asset  TEXT NOT NULL DEFAULT '',
            quote_asset TEXT NOT NULL DEFAULT '',
            is_active   INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS klines (
            id            TEXT PRIMARY KEY,
            symbol        TEXT NOT NULL,
            interval_type TEXT NOT NULL,
            open_time     TEXT NOT NULL,
            open          TEXT NOT NULL,
            high          TEXT NOT NULL,
            low           TEXT NOT NULL,
            close         TEXT NOT NULL,
            volume        TEXT NOT NULL,
            close_time    TEXT NOT NULL,
            UNIQUE (symbol, interval_type, open_time)
        );
        CREATE TABLE IF NOT EXISTS signals (
            signal_id       TEXT PRIMARY KEY,
            symbol          TEXT NOT NULL,
            signal_type     TEXT NOT NULL CHECK (signal_type IN ('long','short','close')),
            strategy_name   TEXT NOT NULL,
            strength_score  INTEGER NOT NULL,
            entry_price     TEXT,
            tp_price        TEXT,
            sl_price        TEXT,
            indicators_json TEXT,
            blocked         INTEGER NOT NULL DEFAULT 0,
            block_reason    TEXT,
            created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS positions (
            position_id       TEXT PRIMARY KEY,
            symbol            TEXT NOT NULL,
            side              TEXT NOT NULL,
            leverage          INTEGER NOT NULL DEFAULT 1,
            entry_price       TEXT NOT NULL DEFAULT '0',
            quantity          TEXT NOT NULL DEFAULT '0',
            liquidation_price TEXT NOT NULL DEFAULT '0',
            stop_loss         TEXT NOT NULL DEFAULT '0',
            initial_stop_loss TEXT NOT NULL DEFAULT '0',
            take_profit_1     TEXT,
            take_profit_2     TEXT,
            status            TEXT NOT NULL DEFAULT 'open',
            close_reason      TEXT,
            exit_price        TEXT,
            realized_pnl      TEXT DEFAULT '0',
            strategy_name     TEXT,
            trading_mode      TEXT NOT NULL DEFAULT 'paper',
            opened_at         TEXT NOT NULL DEFAULT (datetime('now')),
            closed_at         TEXT
        );
        """
    )
    conn.commit()
    return conn


def _insert_symbol(conn, symbol, is_active=1):
    conn.execute(
        "INSERT OR IGNORE INTO symbols (symbol, is_active) VALUES (?, ?)",
        (symbol, is_active),
    )
    conn.commit()


def _insert_candles_ms(conn, symbol, count=50, interval="1m"):
    """Insert candles with Unix-ms timestamps (required by _rows_to_df)."""
    base_ms = 1_704_067_200_000  # 2024-01-01 00:00 UTC
    for i in range(count):
        ts = str(base_ms + i * 60_000)
        row_id = f"{symbol}_{interval}_{ts}"
        conn.execute(
            "INSERT OR IGNORE INTO klines "
            "(id, symbol, interval_type, open_time, open, high, low, close, volume, close_time) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (row_id, symbol, interval, ts,
             "80000", "81000", "79000", "80100", "1000.0", str(int(ts) + 59_999)),
        )
    conn.commit()


def _insert_position(conn, symbol="005930", side="long", quantity="10", status="open") -> str:
    pid = str(__import__("uuid").uuid4())
    conn.execute(
        "INSERT INTO positions (position_id, symbol, side, quantity, entry_price, stop_loss, "
        "initial_stop_loss, status) VALUES (?,?,?,?,?,?,?,?)",
        (pid, symbol, side, quantity, "80000", "78000", "78000", status),
    )
    conn.commit()
    return pid


def _make_strategy_runner(
    signal: SignalResult | None = None,
    timeframe: str = "1m",
    strategy_name: str = "test_strategy",
) -> MagicMock:
    runner = MagicMock()
    runner.run.return_value = signal or SignalResult()
    runner.get_timeframe.return_value = timeframe
    runner.get_active_strategy_name.return_value = strategy_name
    runner.get_symbol_strategy_name.return_value = strategy_name
    return runner


def _make_engine(
    conn=None,
    signal: SignalResult | None = None,
) -> tuple[SignalEngine, sqlite3.Connection]:
    if conn is None:
        conn = _make_db()
    runner = _make_strategy_runner(signal)
    engine = SignalEngine(conn=conn, strategy_runner=runner)
    return engine, conn


# ---------------------------------------------------------------------------
# process_symbol: no candles
# ---------------------------------------------------------------------------

class TestProcessSymbolNoCandles(unittest.IsolatedAsyncioTestCase):

    async def test_returns_none_when_no_candles(self):
        conn = _make_db()
        _insert_symbol(conn, "005930")
        engine, _ = _make_engine(conn=conn)
        result = await engine.process_symbol("005930")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# process_symbol: non-actionable signal
# ---------------------------------------------------------------------------

class TestProcessSymbolNonActionable(unittest.IsolatedAsyncioTestCase):

    async def test_non_actionable_not_saved(self):
        conn = _make_db()
        _insert_symbol(conn, "005930")
        _insert_candles_ms(conn, "005930")

        non_action = SignalResult(signal_type="none", strength_score=0)
        engine, conn = _make_engine(conn=conn, signal=non_action)

        await engine.process_symbol("005930")

        count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        self.assertEqual(count, 0)

    async def test_low_strength_not_saved(self):
        conn = _make_db()
        _insert_symbol(conn, "005930")
        _insert_candles_ms(conn, "005930")

        weak = SignalResult(signal_type="long", strength_score=1)  # < 2 → not actionable
        engine, conn = _make_engine(conn=conn, signal=weak)

        await engine.process_symbol("005930")

        count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        self.assertEqual(count, 0)


# ---------------------------------------------------------------------------
# process_symbol: actionable signal → DB save
# ---------------------------------------------------------------------------

class TestProcessSymbolSavesSignal(unittest.IsolatedAsyncioTestCase):

    async def test_actionable_long_saved_not_blocked(self):
        conn = _make_db()
        _insert_symbol(conn, "005930")
        _insert_candles_ms(conn, "005930")

        good_signal = SignalResult(
            signal_type="long",
            strength_score=2,
            entry_price=80_000.0,
            sl=78_000.0,
            tp1=84_000.0,
            indicators={"rsi": 55.0},
        )
        engine, conn = _make_engine(conn=conn, signal=good_signal)

        with patch("src.utils.startup_recovery.get_cached_balance",
                   return_value={"availableBalance": 1_000_000.0}):
            result = await engine.process_symbol("005930")

        self.assertIsNotNone(result)
        row = conn.execute("SELECT * FROM signals LIMIT 1").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["signal_type"], "long")
        self.assertEqual(row["blocked"], 0)
        self.assertIsNone(row["block_reason"])

    async def test_actionable_short_not_executed_for_krx(self):
        # KRX long-only: "short" signal from strategy = exit signal, not new entry
        conn = _make_db()
        _insert_symbol(conn, "005930")
        _insert_candles_ms(conn, "005930")

        short_signal = SignalResult(
            signal_type="short",
            strength_score=3,
            entry_price=80_000.0,
            sl=82_000.0,
        )
        engine, conn = _make_engine(conn=conn, signal=short_signal)

        # No open position → short signal is ignored (KRX long-only)
        await engine.process_symbol("005930")

        count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        self.assertEqual(count, 0)


# ---------------------------------------------------------------------------
# process_symbol: blocked signal → blocked=1 saved
# ---------------------------------------------------------------------------

class TestProcessSymbolBlocked(unittest.IsolatedAsyncioTestCase):

    async def test_invalid_sl_saves_blocked(self):
        conn = _make_db()
        _insert_symbol(conn, "005930")
        _insert_candles_ms(conn, "005930")

        # entry <= sl → invalid → blocked
        sig = SignalResult(
            signal_type="long",
            strength_score=2,
            entry_price=78_000.0,
            sl=80_000.0,  # SL above entry → invalid
        )
        engine, conn = _make_engine(conn=conn, signal=sig)

        with patch("src.utils.startup_recovery.get_cached_balance",
                   return_value={"availableBalance": 1_000_000.0}):
            await engine.process_symbol("005930")

        row = conn.execute("SELECT blocked, block_reason FROM signals LIMIT 1").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["blocked"], 1)
        self.assertIn("entry", row["block_reason"].lower())

    async def test_blocked_signal_still_returned(self):
        conn = _make_db()
        _insert_symbol(conn, "005930")
        _insert_candles_ms(conn, "005930")

        sig = SignalResult(
            signal_type="long", strength_score=2,
            entry_price=78_000.0, sl=80_000.0,
        )
        engine, _ = _make_engine(conn=conn, signal=sig)

        with patch("src.utils.startup_recovery.get_cached_balance",
                   return_value={"availableBalance": 1_000_000.0}):
            result = await engine.process_symbol("005930")

        self.assertIsNotNone(result)
        self.assertEqual(result.signal_type, "long")


# ---------------------------------------------------------------------------
# process_all_symbols: multi-symbol concurrent
# ---------------------------------------------------------------------------

class TestProcessAllSymbols(unittest.IsolatedAsyncioTestCase):

    async def test_processes_multiple_symbols(self):
        conn = _make_db()
        for sym in ("005930", "000660", "035720"):
            _insert_symbol(conn, sym)
            _insert_candles_ms(conn, sym)

        sig = SignalResult(signal_type="long", strength_score=2,
                           entry_price=80_000.0, sl=78_000.0)
        engine, conn = _make_engine(conn=conn, signal=sig)

        with patch("src.utils.startup_recovery.get_cached_balance",
                   return_value={"availableBalance": 1_000_000.0}):
            results = await engine.process_all_symbols()

        self.assertEqual(len(results), 3)
        count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        self.assertEqual(count, 3)

    async def test_inactive_symbol_not_processed(self):
        conn = _make_db()
        _insert_symbol(conn, "005930", is_active=1)
        _insert_symbol(conn, "000660", is_active=0)
        _insert_candles_ms(conn, "005930")
        _insert_candles_ms(conn, "000660")

        sig = SignalResult(signal_type="long", strength_score=2,
                           entry_price=80_000.0, sl=78_000.0)
        engine, conn = _make_engine(conn=conn, signal=sig)

        with patch("src.utils.startup_recovery.get_cached_balance",
                   return_value={"availableBalance": 1_000_000.0}):
            results = await engine.process_all_symbols()

        self.assertEqual(len(results), 1)

    async def test_empty_symbols_returns_empty_list(self):
        conn = _make_db()
        engine, _ = _make_engine(conn=conn)
        results = await engine.process_all_symbols()
        self.assertEqual(results, [])

    async def test_one_symbol_error_does_not_abort_others(self):
        conn = _make_db()
        for sym in ("005930", "000660"):
            _insert_symbol(conn, sym)
            _insert_candles_ms(conn, sym)

        runner = MagicMock()
        runner.get_timeframe.return_value = "1m"
        runner.get_active_strategy_name.return_value = "test"
        runner.get_symbol_strategy_name.return_value = "test"
        good_sig = SignalResult(signal_type="long", strength_score=2,
                                entry_price=80_000.0, sl=78_000.0)
        runner.run.side_effect = [Exception("strategy crash"), good_sig]

        engine = SignalEngine(conn=conn, strategy_runner=runner)

        with patch("src.utils.startup_recovery.get_cached_balance",
                   return_value={"availableBalance": 1_000_000.0}):
            results = await engine.process_all_symbols()

        self.assertIsInstance(results, list)


# ---------------------------------------------------------------------------
# Timeframe resolution
# ---------------------------------------------------------------------------

class TestTimeframeResolution(unittest.TestCase):

    def test_uses_strategy_runner_timeframe(self):
        conn = _make_db()
        runner = _make_strategy_runner(timeframe="1h")
        engine = SignalEngine(conn=conn, strategy_runner=runner)
        self.assertEqual(engine._get_timeframe(), "1h")

    def test_defaults_to_1m_when_no_get_timeframe(self):
        conn = _make_db()
        runner = MagicMock(spec=[])
        engine = SignalEngine(conn=conn, strategy_runner=runner)
        self.assertEqual(engine._get_timeframe(), "1m")


# ---------------------------------------------------------------------------
# Strategy-driven exit: _execute_exit
# ---------------------------------------------------------------------------

class TestExecuteExit(unittest.IsolatedAsyncioTestCase):

    def _make_om(self):
        om = MagicMock()
        om.market_close = AsyncMock(return_value={"id": "order123"})
        om._telegram = None
        return om

    async def test_long_position_closed_on_exit(self):
        conn = _make_db()
        pid = _insert_position(conn, "005930", side="long", quantity="10")
        om = self._make_om()
        engine = SignalEngine(conn=conn,
                              strategy_runner=_make_strategy_runner(),
                              order_manager=om)

        pos_row = conn.execute(
            "SELECT side, quantity, entry_price FROM positions WHERE position_id=?", (pid,)
        ).fetchone()
        await engine._execute_exit("005930", pos_row)

        om.market_close.assert_called_once_with("005930", 10)
        status = conn.execute(
            "SELECT status, close_reason FROM positions WHERE position_id=?", (pid,)
        ).fetchone()
        self.assertEqual(status["status"], "closed")
        self.assertEqual(status["close_reason"], "strategy_exit")

    async def test_short_position_exit(self):
        conn = _make_db()
        pid = _insert_position(conn, "000660", side="short", quantity="15")
        om = self._make_om()
        engine = SignalEngine(conn=conn,
                              strategy_runner=_make_strategy_runner(),
                              order_manager=om)

        pos_row = conn.execute(
            "SELECT side, quantity, entry_price FROM positions WHERE position_id=?", (pid,)
        ).fetchone()
        await engine._execute_exit("000660", pos_row)

        om.market_close.assert_called_once_with("000660", 15)
        status = conn.execute("SELECT status FROM positions WHERE position_id=?", (pid,)).fetchone()
        self.assertEqual(status["status"], "closed")

    async def test_no_order_manager_skips_without_crash(self):
        conn = _make_db()
        pid = _insert_position(conn, "005930", side="long", quantity="5")
        engine = SignalEngine(conn=conn,
                              strategy_runner=_make_strategy_runner(),
                              order_manager=None)

        pos_row = conn.execute(
            "SELECT side, quantity, entry_price FROM positions WHERE position_id=?", (pid,)
        ).fetchone()
        await engine._execute_exit("005930", pos_row)  # must not raise

        status = conn.execute("SELECT status FROM positions WHERE position_id=?", (pid,)).fetchone()
        self.assertEqual(status["status"], "open")

    async def test_market_close_failure_leaves_db_open(self):
        conn = _make_db()
        pid = _insert_position(conn, "005930", side="long", quantity="5")
        om = self._make_om()
        om.market_close.side_effect = RuntimeError("KIS error")
        engine = SignalEngine(conn=conn,
                              strategy_runner=_make_strategy_runner(),
                              order_manager=om)

        pos_row = conn.execute(
            "SELECT side, quantity, entry_price FROM positions WHERE position_id=?", (pid,)
        ).fetchone()
        await engine._execute_exit("005930", pos_row)  # must not raise

        status = conn.execute("SELECT status FROM positions WHERE position_id=?", (pid,)).fetchone()
        self.assertEqual(status["status"], "open")


# ---------------------------------------------------------------------------
# process_symbol: open-position routing
# ---------------------------------------------------------------------------

class TestProcessSymbolWithOpenPosition(unittest.IsolatedAsyncioTestCase):

    async def test_reversal_signal_triggers_exit(self):
        conn = _make_db()
        _insert_symbol(conn, "005930")
        _insert_candles_ms(conn, "005930")
        _insert_position(conn, "005930", side="long", quantity="10")

        reversal = SignalResult(signal_type="short", strength_score=3,
                                entry_price=79_000.0, sl=81_000.0)
        runner = _make_strategy_runner(reversal)
        engine = SignalEngine(conn=conn, strategy_runner=runner)
        om = MagicMock()
        om.market_close = AsyncMock(return_value={"id": "x"})
        om._telegram = None
        engine._order_manager = om

        await engine.process_symbol("005930")

        om.market_close.assert_called_once()
        row = conn.execute(
            "SELECT status FROM positions WHERE symbol='005930'"
        ).fetchone()
        self.assertEqual(row["status"], "closed")
        count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        self.assertEqual(count, 0)

    async def test_same_direction_skips_entry(self):
        conn = _make_db()
        _insert_symbol(conn, "005930")
        _insert_candles_ms(conn, "005930")
        _insert_position(conn, "005930", side="long", quantity="10")

        same_dir = SignalResult(signal_type="long", strength_score=3,
                                entry_price=81_000.0, sl=79_000.0)
        runner = _make_strategy_runner(same_dir)
        engine = SignalEngine(conn=conn, strategy_runner=runner)
        om = MagicMock()
        om.market_close = AsyncMock()
        engine._order_manager = om

        await engine.process_symbol("005930")

        om.market_close.assert_not_called()
        count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        self.assertEqual(count, 0)

    async def test_no_open_position_normal_entry_flow(self):
        conn = _make_db()
        _insert_symbol(conn, "005930")
        _insert_candles_ms(conn, "005930")

        long_sig = SignalResult(signal_type="long", strength_score=3,
                                entry_price=80_000.0, sl=78_000.0)
        engine, conn = _make_engine(conn=conn, signal=long_sig)

        with patch("src.utils.startup_recovery.get_cached_balance",
                   return_value={"availableBalance": 1_000_000.0}):
            await engine.process_symbol("005930")

        count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        self.assertEqual(count, 1)


# ---------------------------------------------------------------------------
# _execute_signal: KRX position sizing
# ---------------------------------------------------------------------------

class TestExecuteSignal(unittest.IsolatedAsyncioTestCase):

    def _make_engine_with_om(self):
        conn = _make_db()
        om = MagicMock()
        om.submit_and_confirm = AsyncMock(return_value={
            "broker_order_id": "KIS001", "confirmed": True,
        })
        om._telegram = None
        runner = _make_strategy_runner()
        engine = SignalEngine(conn=conn, strategy_runner=runner, order_manager=om)
        return engine, conn, om

    async def test_executes_buy_order_for_valid_signal(self):
        engine, conn, om = self._make_engine_with_om()

        from src.utils.config import load_config
        config = load_config()

        result = SignalResult(
            signal_type="long", strength_score=3,
            entry_price=80_000.0, sl=78_000.0, tp1=84_000.0,
        )
        await engine._execute_signal("005930", result, 1_000_000.0, config, "rsi_macd")
        om.submit_and_confirm.assert_called_once()

    async def test_missing_entry_price_skips_execution(self):
        engine, conn, om = self._make_engine_with_om()

        from src.utils.config import load_config
        config = load_config()

        result = SignalResult(
            signal_type="long", strength_score=3,
            entry_price=0.0, sl=78_000.0,  # entry=0 → invalid
        )
        await engine._execute_signal("005930", result, 1_000_000.0, config, "rsi_macd")
        om.submit_and_confirm.assert_not_called()

    async def test_invalid_sl_skips_execution(self):
        engine, conn, om = self._make_engine_with_om()

        from src.utils.config import load_config
        config = load_config()

        result = SignalResult(
            signal_type="long", strength_score=3,
            entry_price=80_000.0, sl=82_000.0,  # sl > entry → invalid
        )
        await engine._execute_signal("005930", result, 1_000_000.0, config, "rsi_macd")
        om.submit_and_confirm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
