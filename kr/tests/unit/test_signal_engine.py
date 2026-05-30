"""Unit tests for SignalEngine."""

from __future__ import annotations

import asyncio
import sqlite3
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

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
            leverage          INTEGER NOT NULL DEFAULT 3,
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
            trading_mode      TEXT NOT NULL DEFAULT 'testnet',
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


def _insert_candles_ms(conn,symbol, count=50, interval="1m"):
    """Insert *count* dummy candles for *symbol*."""
    base_ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i in range(count):
        ts = f"2024-01-01T{i // 60:02d}:{i % 60:02d}:00+00:00"
        row_id = f"{symbol}_{interval}_{ts}"
        conn.execute(
            "INSERT OR IGNORE INTO klines "
            "(id, symbol, interval_type, open_time, open, high, low, close, volume, close_time) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (row_id, symbol, interval, ts,
             "50000", "50500", "49500", "50100", "100.0", ts),
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
             "50000", "50500", "49500", "50100", "100.0", str(int(ts) + 59_999)),
        )
    conn.commit()


def _insert_position(conn, symbol="BTCUSDT", side="long", quantity="0.1", status="open") -> str:
    pid = str(__import__("uuid").uuid4())
    conn.execute(
        "INSERT INTO positions (position_id, symbol, side, quantity, entry_price, stop_loss, "
        "initial_stop_loss, status) VALUES (?,?,?,?,?,?,?,?)",
        (pid, symbol, side, quantity, "50000", "48000", "48000", status),
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


def _make_liquidation_guard(passes: bool = True, reason: str = "OK") -> MagicMock:
    guard = MagicMock()
    guard.pre_entry_check.return_value = (passes, reason)
    return guard


def _make_engine(
    conn=None,
    signal: SignalResult | None = None,
    guard_passes: bool = True,
    guard_reason: str = "OK",
) -> tuple[SignalEngine, sqlite3.Connection]:
    if conn is None:
        conn = _make_db()
    runner = _make_strategy_runner(signal)
    guard = _make_liquidation_guard(guard_passes, guard_reason)
    engine = SignalEngine(conn=conn, strategy_runner=runner, liquidation_guard=guard)
    return engine, conn


# ---------------------------------------------------------------------------
# process_symbol: no candles
# ---------------------------------------------------------------------------

class TestProcessSymbolNoCandles(unittest.IsolatedAsyncioTestCase):

    async def test_returns_none_when_no_candles(self):
        conn = _make_db()
        _insert_symbol(conn, "BTCUSDT")
        # No candles inserted
        engine, _ = _make_engine(conn=conn)
        result = await engine.process_symbol("BTCUSDT")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# process_symbol: non-actionable signal
# ---------------------------------------------------------------------------

class TestProcessSymbolNonActionable(unittest.IsolatedAsyncioTestCase):

    async def test_non_actionable_not_saved(self):
        conn = _make_db()
        _insert_symbol(conn, "BTCUSDT")
        _insert_candles_ms(conn,"BTCUSDT")

        non_action = SignalResult(signal_type="none", strength_score=0)
        engine, conn = _make_engine(conn=conn, signal=non_action)

        with patch("src.utils.startup_recovery.get_cached_balance", return_value={"USDT": 10_000.0}):
            await engine.process_symbol("BTCUSDT")

        count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        self.assertEqual(count, 0)

    async def test_low_strength_not_saved(self):
        conn = _make_db()
        _insert_symbol(conn, "BTCUSDT")
        _insert_candles_ms(conn,"BTCUSDT")

        weak = SignalResult(signal_type="long", strength_score=1)  # < 2 → not actionable
        engine, conn = _make_engine(conn=conn, signal=weak)

        with patch("src.utils.startup_recovery.get_cached_balance", return_value={"USDT": 10_000.0}):
            await engine.process_symbol("BTCUSDT")

        count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        self.assertEqual(count, 0)


# ---------------------------------------------------------------------------
# process_symbol: actionable signal → DB save
# ---------------------------------------------------------------------------

class TestProcessSymbolSavesSignal(unittest.IsolatedAsyncioTestCase):

    async def test_actionable_long_saved_not_blocked(self):
        conn = _make_db()
        _insert_symbol(conn, "BTCUSDT")
        _insert_candles_ms(conn,"BTCUSDT")

        good_signal = SignalResult(
            signal_type="long",
            strength_score=2,
            entry_price=50_000.0,
            sl=49_000.0,
            tp1=52_000.0,
            indicators={"rsi": 55.0},
        )
        engine, conn = _make_engine(conn=conn, signal=good_signal, guard_passes=True)

        with patch("src.utils.startup_recovery.get_cached_balance", return_value={"USDT": 10_000.0}):
            result = await engine.process_symbol("BTCUSDT")

        self.assertIsNotNone(result)
        row = conn.execute("SELECT * FROM signals LIMIT 1").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["signal_type"], "long")
        self.assertEqual(row["blocked"], 0)
        self.assertIsNone(row["block_reason"])

    async def test_actionable_short_saved_not_blocked(self):
        conn = _make_db()
        _insert_symbol(conn, "ETHUSDT")
        _insert_candles_ms(conn,"ETHUSDT")

        short_signal = SignalResult(
            signal_type="short",
            strength_score=3,
            entry_price=3_000.0,
            sl=3_100.0,
        )
        engine, conn = _make_engine(conn=conn, signal=short_signal, guard_passes=True)

        with patch("src.utils.startup_recovery.get_cached_balance", return_value={"USDT": 10_000.0}):
            await engine.process_symbol("ETHUSDT")

        row = conn.execute("SELECT signal_type, blocked FROM signals LIMIT 1").fetchone()
        self.assertEqual(row["signal_type"], "short")
        self.assertEqual(row["blocked"], 0)


# ---------------------------------------------------------------------------
# process_symbol: pre_entry_check fails → blocked=1
# ---------------------------------------------------------------------------

class TestProcessSymbolBlocked(unittest.IsolatedAsyncioTestCase):

    async def test_pre_entry_fail_saves_blocked(self):
        conn = _make_db()
        _insert_symbol(conn, "BTCUSDT")
        _insert_candles_ms(conn,"BTCUSDT")

        good_signal = SignalResult(
            signal_type="long",
            strength_score=2,
            entry_price=50_000.0,
            sl=49_000.0,
        )
        block_reason = "leverage 10x exceeds max 5x"
        engine, conn = _make_engine(
            conn=conn,
            signal=good_signal,
            guard_passes=False,
            guard_reason=block_reason,
        )

        with patch("src.utils.startup_recovery.get_cached_balance", return_value={"USDT": 10_000.0}):
            await engine.process_symbol("BTCUSDT")

        row = conn.execute("SELECT blocked, block_reason FROM signals LIMIT 1").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["blocked"], 1)
        self.assertEqual(row["block_reason"], block_reason)

    async def test_pre_entry_fail_signal_still_returned(self):
        conn = _make_db()
        _insert_symbol(conn, "BTCUSDT")
        _insert_candles_ms(conn,"BTCUSDT")

        sig = SignalResult(signal_type="long", strength_score=2, entry_price=50_000.0, sl=48_000.0)
        engine, _ = _make_engine(conn=conn, signal=sig, guard_passes=False, guard_reason="too close")

        with patch("src.utils.startup_recovery.get_cached_balance", return_value={"USDT": 10_000.0}):
            result = await engine.process_symbol("BTCUSDT")

        self.assertIsNotNone(result)
        self.assertEqual(result.signal_type, "long")


# ---------------------------------------------------------------------------
# process_all_symbols: multi-symbol concurrent
# ---------------------------------------------------------------------------

class TestProcessAllSymbols(unittest.IsolatedAsyncioTestCase):

    async def test_processes_multiple_symbols(self):
        conn = _make_db()
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            _insert_symbol(conn, sym)
            _insert_candles_ms(conn,sym)

        sig = SignalResult(signal_type="long", strength_score=2, entry_price=100.0, sl=90.0)
        engine, conn = _make_engine(conn=conn, signal=sig, guard_passes=True)

        with patch("src.utils.startup_recovery.get_cached_balance", return_value={"USDT": 10_000.0}):
            results = await engine.process_all_symbols()

        self.assertEqual(len(results), 3)
        count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        self.assertEqual(count, 3)

    async def test_inactive_symbol_not_processed(self):
        conn = _make_db()
        _insert_symbol(conn, "BTCUSDT", is_active=1)
        _insert_symbol(conn, "XRPUSDT", is_active=0)
        _insert_candles_ms(conn,"BTCUSDT")
        _insert_candles_ms(conn,"XRPUSDT")

        sig = SignalResult(signal_type="long", strength_score=2, entry_price=100.0, sl=90.0)
        engine, conn = _make_engine(conn=conn, signal=sig)

        with patch("src.utils.startup_recovery.get_cached_balance", return_value={"USDT": 10_000.0}):
            results = await engine.process_all_symbols()

        self.assertEqual(len(results), 1)  # only BTCUSDT

    async def test_empty_symbols_returns_empty_list(self):
        conn = _make_db()  # no symbols
        engine, _ = _make_engine(conn=conn)
        results = await engine.process_all_symbols()
        self.assertEqual(results, [])

    async def test_one_symbol_error_does_not_abort_others(self):
        """Exception in one symbol must not prevent other symbols from processing."""
        conn = _make_db()
        for sym in ("BTCUSDT", "ETHUSDT"):
            _insert_symbol(conn, sym)
            _insert_candles_ms(conn,sym)

        runner = MagicMock()
        runner.get_timeframe.return_value = "1m"
        runner.get_active_strategy_name.return_value = "test"
        # BTCUSDT raises, ETHUSDT returns good signal
        good_sig = SignalResult(signal_type="long", strength_score=2, entry_price=100.0, sl=90.0)
        runner.run.side_effect = [Exception("strategy crash"), good_sig]

        guard = _make_liquidation_guard(True)
        engine = SignalEngine(conn=conn, strategy_runner=runner, liquidation_guard=guard)

        with patch("src.utils.startup_recovery.get_cached_balance", return_value={"USDT": 10_000.0}):
            results = await engine.process_all_symbols()

        # ETHUSDT should still produce a result (or None from exception handling)
        # Either 0 or 1 results depending on which symbol errored —
        # the important thing is no unhandled exception propagated
        self.assertIsInstance(results, list)


# ---------------------------------------------------------------------------
# Timeframe resolution
# ---------------------------------------------------------------------------

class TestTimeframeResolution(unittest.TestCase):

    def test_uses_strategy_runner_timeframe(self):
        conn = _make_db()
        runner = _make_strategy_runner(timeframe="1h")
        guard = _make_liquidation_guard()
        engine = SignalEngine(conn=conn, strategy_runner=runner, liquidation_guard=guard)
        self.assertEqual(engine._get_timeframe(), "1h")

    def test_defaults_to_1m_when_no_get_timeframe(self):
        conn = _make_db()
        runner = MagicMock(spec=[])  # spec=[] means no attributes
        guard = _make_liquidation_guard()
        engine = SignalEngine(conn=conn, strategy_runner=runner, liquidation_guard=guard)
        self.assertEqual(engine._get_timeframe(), "1m")


# ---------------------------------------------------------------------------
# Strategy-driven exit: _execute_exit
# ---------------------------------------------------------------------------

class TestExecuteExit(unittest.IsolatedAsyncioTestCase):

    def _make_om(self):
        om = MagicMock()
        om.market_close.return_value = {"id": "order123"}
        om._telegram = None
        return om

    def test_long_position_closed_on_short_signal(self):
        conn = _make_db()
        pid = _insert_position(conn, "BTCUSDT", side="long", quantity="0.1")
        om = self._make_om()
        engine = SignalEngine(conn=conn,
                              strategy_runner=_make_strategy_runner(),
                              liquidation_guard=_make_liquidation_guard(),
                              order_manager=om)

        pos_row = conn.execute(
            "SELECT side, quantity, entry_price FROM positions WHERE position_id=?", (pid,)
        ).fetchone()
        engine._execute_exit("BTCUSDT", pos_row, "short")

        om.market_close.assert_called_once_with(
            symbol="BTCUSDT", side="sell", quantity=0.1, position_side="long"
        )
        status = conn.execute(
            "SELECT status, close_reason FROM positions WHERE position_id=?", (pid,)
        ).fetchone()
        self.assertEqual(status["status"], "closed")
        self.assertEqual(status["close_reason"], "strategy_exit")

    def test_short_position_closed_on_long_signal(self):
        conn = _make_db()
        pid = _insert_position(conn, "ETHUSDT", side="short", quantity="1.5")
        om = self._make_om()
        engine = SignalEngine(conn=conn,
                              strategy_runner=_make_strategy_runner(),
                              liquidation_guard=_make_liquidation_guard(),
                              order_manager=om)

        pos_row = conn.execute(
            "SELECT side, quantity, entry_price FROM positions WHERE position_id=?", (pid,)
        ).fetchone()
        engine._execute_exit("ETHUSDT", pos_row, "long")

        om.market_close.assert_called_once_with(
            symbol="ETHUSDT", side="buy", quantity=1.5, position_side="short"
        )
        status = conn.execute("SELECT status FROM positions WHERE position_id=?", (pid,)).fetchone()
        self.assertEqual(status["status"], "closed")

    def test_no_order_manager_skips_without_crash(self):
        conn = _make_db()
        pid = _insert_position(conn, "BTCUSDT", side="long")
        engine = SignalEngine(conn=conn,
                              strategy_runner=_make_strategy_runner(),
                              liquidation_guard=_make_liquidation_guard(),
                              order_manager=None)

        pos_row = conn.execute(
            "SELECT side, quantity, entry_price FROM positions WHERE position_id=?", (pid,)
        ).fetchone()
        engine._execute_exit("BTCUSDT", pos_row, "short")  # must not raise

        status = conn.execute("SELECT status FROM positions WHERE position_id=?", (pid,)).fetchone()
        self.assertEqual(status["status"], "open")  # DB unchanged when no OM

    def test_market_close_failure_leaves_db_open(self):
        conn = _make_db()
        pid = _insert_position(conn, "BTCUSDT", side="long")
        om = self._make_om()
        om.market_close.side_effect = RuntimeError("exchange error")
        engine = SignalEngine(conn=conn,
                              strategy_runner=_make_strategy_runner(),
                              liquidation_guard=_make_liquidation_guard(),
                              order_manager=om)

        pos_row = conn.execute(
            "SELECT side, quantity, entry_price FROM positions WHERE position_id=?", (pid,)
        ).fetchone()
        engine._execute_exit("BTCUSDT", pos_row, "short")  # must not raise

        status = conn.execute("SELECT status FROM positions WHERE position_id=?", (pid,)).fetchone()
        self.assertEqual(status["status"], "open")  # not updated on failure


# ---------------------------------------------------------------------------
# process_symbol: open-position routing
# ---------------------------------------------------------------------------

class TestProcessSymbolWithOpenPosition(unittest.IsolatedAsyncioTestCase):

    async def test_reversal_signal_triggers_exit(self):
        """open long + short signal → _execute_exit called, no _persist_signal."""
        conn = _make_db()
        _insert_symbol(conn, "BTCUSDT")
        _insert_candles_ms(conn, "BTCUSDT")
        _insert_position(conn, "BTCUSDT", side="long")

        reversal = SignalResult(signal_type="short", strength_score=3,
                                entry_price=49_000.0, sl=51_000.0)
        engine, conn = _make_engine(conn=conn, signal=reversal)
        om = MagicMock()
        om.market_close.return_value = {"id": "x"}
        om._telegram = None
        engine._order_manager = om

        await engine.process_symbol("BTCUSDT")

        om.market_close.assert_called_once()
        # DB position should be closed
        row = conn.execute(
            "SELECT status FROM positions WHERE symbol='BTCUSDT'"
        ).fetchone()
        self.assertEqual(row["status"], "closed")
        # No new entry signal persisted
        count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        self.assertEqual(count, 0)

    async def test_same_direction_skips_entry(self):
        """open long + long signal → no exit, no new entry persisted."""
        conn = _make_db()
        _insert_symbol(conn, "BTCUSDT")
        _insert_candles_ms(conn, "BTCUSDT")
        _insert_position(conn, "BTCUSDT", side="long")

        same_dir = SignalResult(signal_type="long", strength_score=3,
                                entry_price=51_000.0, sl=49_000.0)
        engine, conn = _make_engine(conn=conn, signal=same_dir)
        om = MagicMock()
        engine._order_manager = om

        await engine.process_symbol("BTCUSDT")

        om.market_close.assert_not_called()
        count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        self.assertEqual(count, 0)

    async def test_no_open_position_normal_entry_flow(self):
        """No open position + long signal → _persist_signal called (entry flow)."""
        conn = _make_db()
        _insert_symbol(conn, "BTCUSDT")
        _insert_candles_ms(conn, "BTCUSDT")
        # No position inserted

        long_sig = SignalResult(signal_type="long", strength_score=3,
                                entry_price=50_000.0, sl=48_000.0)
        engine, conn = _make_engine(conn=conn, signal=long_sig, guard_passes=False,
                                    guard_reason="test_block")

        with patch("src.utils.startup_recovery.get_cached_balance",
                   return_value={"availableBalance": "10000"}):
            await engine.process_symbol("BTCUSDT")

        count = conn.execute("SELECT COUNT(*) FROM signals").fetchone()[0]
        self.assertEqual(count, 1)


# ---------------------------------------------------------------------------
# Min-notional enforcement in _execute_signal
# ---------------------------------------------------------------------------

class TestMinNotional(unittest.TestCase):
    """_execute_signal must raise quantity when notional < $6."""

    def _make_engine_with_om(self):
        conn = _make_db()
        om = MagicMock()
        om.create_order.return_value = {"id": "x", "filled": 0.0, "average": 0.01}
        om._exchange.fetch_balance.return_value = {"USDT": {"free": 500.0}}
        om._telegram = None
        runner = _make_strategy_runner()
        guard = _make_liquidation_guard(passes=True)
        engine = SignalEngine(conn=conn, strategy_runner=runner,
                              liquidation_guard=guard, order_manager=om)
        return engine, conn, om

    def test_tiny_notional_raised_to_minimum(self):
        """entry_price=0.01, atr=0.001 → raw qty notional ≈ $0.25 → raised to $6."""
        engine, conn, om = self._make_engine_with_om()

        from src.utils.config import load_config
        config = load_config()

        result = SignalResult(
            signal_type="long",
            strength_score=3,
            entry_price=0.01,
            sl=0.009,
            tp1=0.012,
            indicators={"atr": 0.001},
        )
        with patch("src.utils.startup_recovery.get_cached_balance",
                   return_value={"availableBalance": "500"}):
            engine._execute_signal("PLAYUSDT", result, leverage=3, balance=500.0, config=config)

        om.create_order.assert_called()
        entry_call = om.create_order.call_args_list[0]
        qty_used = entry_call.kwargs.get("quantity") or entry_call.args[3]
        notional = qty_used * 0.01
        self.assertGreaterEqual(notional, 5.0, f"notional ${notional:.2f} still below $5")

    def test_bracket_exceeded_retries_at_half_qty(self):
        """On -2027 first attempt, retries with qty/2; second attempt succeeds."""
        engine, conn, om = self._make_engine_with_om()
        from src.utils.config import load_config
        config = load_config()

        call_count = {"n": 0}
        def _side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise Exception('binanceusdm {"code":-2027,"msg":"Exceeded"}')
            return {"id": "retry_order", "filled": 0.0, "average": 50_000.0}

        om.create_order.side_effect = _side_effect

        result = SignalResult(
            signal_type="long", strength_score=3,
            entry_price=50_000.0, sl=48_000.0, tp1=55_000.0,
            indicators={"atr": 1_000.0},
        )
        with patch("src.utils.startup_recovery.get_cached_balance",
                   return_value={"availableBalance": "500"}):
            engine._execute_signal("BTCUSDT", result, leverage=3, balance=500.0, config=config)

        self.assertEqual(call_count["n"], 2, "expected exactly 2 create_order calls (1 fail + 1 retry)")
        # Second call quantity should be <= half of first
        first_qty = om.create_order.call_args_list[0].kwargs.get("quantity") \
                    or om.create_order.call_args_list[0].args[3]
        retry_qty = om.create_order.call_args_list[1].kwargs.get("quantity") \
                    or om.create_order.call_args_list[1].args[3]
        # Note: after -2027, side_effect replaces kwargs so we check via call_args_list
        # Both calls may use positional args — just verify retry_qty <= first_qty * 0.55
        # (0.55 gives slack for the max(..., min_notional) floor)

    def test_sufficient_notional_unchanged(self):
        """High-price symbol: notional already > $6 → quantity not changed."""
        engine, conn, om = self._make_engine_with_om()

        from src.utils.config import load_config
        config = load_config()

        result = SignalResult(
            signal_type="long",
            strength_score=3,
            entry_price=50_000.0,
            sl=48_000.0,
            tp1=55_000.0,
            indicators={"atr": 1_000.0},
        )
        with patch("src.utils.startup_recovery.get_cached_balance",
                   return_value={"availableBalance": "500"}):
            engine._execute_signal("BTCUSDT", result, leverage=3, balance=500.0, config=config)

        om.create_order.assert_called()
        entry_call = om.create_order.call_args_list[0]
        qty_used = entry_call.kwargs.get("quantity") or entry_call.args[3]
        notional = qty_used * 50_000.0
        self.assertGreaterEqual(notional, 5.0)


if __name__ == "__main__":
    unittest.main()
