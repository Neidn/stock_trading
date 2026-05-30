"""Unit tests for TelegramBot.

HTTP is never hit — urllib.request.urlopen is patched throughout.
DB: in-memory SQLite with the standard schema.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sqlite3
import unittest
from datetime import date
from unittest.mock import MagicMock, patch, call

from src.monitoring.telegram_bot import TelegramBot, get_telegram_bot
import src.monitoring.telegram_bot as _tb_module


# ---------------------------------------------------------------------------
# Schema / helpers (same as emergency_handler tests)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (symbol TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS positions (
    position_id     TEXT PRIMARY KEY,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL CHECK (side IN ('long','short')),
    leverage        INTEGER NOT NULL DEFAULT 5,
    entry_price     TEXT NOT NULL,
    exit_price      TEXT,
    quantity        TEXT NOT NULL,
    liquidation_price TEXT NOT NULL DEFAULT '40000',
    stop_loss       TEXT NOT NULL,
    take_profit_1   TEXT,
    take_profit_2   TEXT,
    initial_stop_loss TEXT NOT NULL DEFAULT '48000',
    trailing_activated INTEGER DEFAULT 0,
    realized_pnl    TEXT DEFAULT '0',
    unrealized_pnl  TEXT DEFAULT '0',
    status          TEXT NOT NULL DEFAULT 'open',
    close_reason    TEXT,
    trading_mode    TEXT NOT NULL DEFAULT 'testnet',
    opened_at       TEXT NOT NULL DEFAULT (datetime('now')),
    closed_at       TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    order_id        TEXT PRIMARY KEY,
    binance_order_id INTEGER,
    symbol          TEXT NOT NULL,
    side            TEXT NOT NULL,
    position_side   TEXT NOT NULL DEFAULT 'both',
    order_type      TEXT NOT NULL,
    price           TEXT,
    quantity        TEXT NOT NULL,
    filled_qty      TEXT NOT NULL DEFAULT '0',
    avg_fill_price  TEXT,
    status          TEXT NOT NULL DEFAULT 'open',
    fee             TEXT NOT NULL DEFAULT '0',
    fee_asset       TEXT,
    trading_mode    TEXT NOT NULL DEFAULT 'testnet',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT
);
CREATE TABLE IF NOT EXISTS daily_performance (
    perf_date       TEXT NOT NULL,
    trading_mode    TEXT NOT NULL DEFAULT 'testnet',
    total_trades    INTEGER DEFAULT 0,
    winning_trades  INTEGER DEFAULT 0,
    losing_trades   INTEGER DEFAULT 0,
    liquidated_trades INTEGER DEFAULT 0,
    gross_profit    TEXT DEFAULT '0',
    gross_loss      TEXT DEFAULT '0',
    net_pnl         TEXT DEFAULT '0',
    total_fees      TEXT DEFAULT '0',
    max_drawdown    TEXT DEFAULT '0',
    win_rate        TEXT DEFAULT '0',
    avg_liquidation_distance TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (perf_date, trading_mode)
);
CREATE TABLE IF NOT EXISTS safe_mode_events (
    event_id   TEXT PRIMARY KEY,
    action     TEXT NOT NULL CHECK (action IN ('activated','deactivated')),
    reason     TEXT NOT NULL,
    by         TEXT NOT NULL DEFAULT 'system',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _insert_position(conn, symbol="BTCUSDT", side="long", quantity=0.1,
                     unrealized_pnl="100.0", status="open") -> str:
    import uuid
    pid = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO positions
           (position_id, symbol, side, entry_price, quantity, stop_loss,
            unrealized_pnl, status)
           VALUES (?,?,?,?,?,?,?,?)""",
        (pid, symbol, side, "50000", str(quantity), "48000",
         unrealized_pnl, status),
    )
    conn.commit()
    return pid


def _insert_daily(conn, net_pnl=200.0, total_trades=4, wins=3,
                  gross_profit=300.0, gross_loss=100.0, fees=10.0,
                  perf_date=None) -> None:
    today = perf_date or date.today().isoformat()
    conn.execute(
        """INSERT OR REPLACE INTO daily_performance
           (perf_date, trading_mode, total_trades, winning_trades, losing_trades,
            net_pnl, gross_profit, gross_loss, total_fees)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (today, "testnet", total_trades, wins, total_trades - wins,
         str(net_pnl), str(gross_profit), str(gross_loss), str(fees)),
    )
    conn.commit()


def _make_bot(**kwargs) -> TelegramBot:
    defaults = {"token": "fake_token", "chat_id": "12345"}
    defaults.update(kwargs)
    return TelegramBot(**defaults)


# ---------------------------------------------------------------------------
# Send methods
# ---------------------------------------------------------------------------

class TestSendMethods(unittest.TestCase):

    def _urlopen_200(self):
        """Return a context-manager mock that looks like a 200 response."""
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        cm = MagicMock()
        cm.return_value = resp
        return cm

    @patch("urllib.request.urlopen")
    def test_send_info_prepends_emoji(self, mock_open):
        mock_open.side_effect = self._urlopen_200().side_effect
        mock_open.return_value.__enter__ = MagicMock(return_value=MagicMock(status=200))
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        bot = _make_bot()
        with patch.object(bot, "_send") as mock_send:
            bot.send_info("hello")
            mock_send.assert_called_once_with("ℹ️ hello")

    def test_send_warning_prepends_emoji(self):
        bot = _make_bot()
        with patch.object(bot, "_send") as mock_send:
            bot.send_warning("danger")
            mock_send.assert_called_once_with("⚠️ danger")

    def test_send_critical_prepends_emoji(self):
        bot = _make_bot()
        with patch.object(bot, "_send") as mock_send:
            bot.send_critical("emergency")
            mock_send.assert_called_once_with("🚨 emergency")

    def test_send_alert_passes_through(self):
        bot = _make_bot()
        with patch.object(bot, "_send") as mock_send:
            bot.send_alert("raw message")
            mock_send.assert_called_once_with("raw message")

    def test_send_no_token_skips_http(self):
        import os
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}):
            bot = TelegramBot(token="", chat_id="")
        with patch("urllib.request.urlopen") as mock_open:
            bot._send("test")
            mock_open.assert_not_called()

    def test_send_no_chat_id_skips_http(self):
        import os
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "tok", "TELEGRAM_CHAT_ID": ""}):
            bot = TelegramBot(token="tok", chat_id="")
        with patch("urllib.request.urlopen") as mock_open:
            bot._send("test")
            mock_open.assert_not_called()

    def test_send_network_error_does_not_raise(self):
        bot = _make_bot()
        with patch("urllib.request.urlopen", side_effect=OSError("network down")):
            # must not propagate
            bot._send("test")

    def test_send_posts_correct_url(self):
        bot = _make_bot(token="mytoken", chat_id="99")
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", fake_urlopen):
            bot._send("hi")

        assert "mytoken" in captured["url"]
        assert "sendMessage" in captured["url"]

    def test_send_payload_contains_chat_id_and_text(self):
        bot = _make_bot(token="tok", chat_id="42")
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["data"] = json.loads(req.data)
            resp = MagicMock()
            resp.status = 200
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        with patch("urllib.request.urlopen", fake_urlopen):
            bot._send("hello world")

        self.assertEqual(captured["data"]["chat_id"], "42")
        self.assertEqual(captured["data"]["text"], "hello world")


# ---------------------------------------------------------------------------
# Command: /status
# ---------------------------------------------------------------------------

class TestCmdStatus(unittest.TestCase):

    def test_no_safe_mode_shows_na(self):
        bot = _make_bot()
        reply = bot._cmd_status()
        self.assertIn("N/A", reply)

    def test_safe_mode_inactive_shows_green(self):
        sm = MagicMock()
        sm.is_active.return_value = False
        bot = _make_bot(safe_mode=sm)
        reply = bot._cmd_status()
        self.assertIn("🟢", reply)

    def test_safe_mode_active_shows_red_and_reason(self):
        sm = MagicMock()
        sm.is_active.return_value = True
        sm.reason = "drawdown limit"
        bot = _make_bot(safe_mode=sm)
        reply = bot._cmd_status()
        self.assertIn("🔴", reply)
        self.assertIn("drawdown limit", reply)


# ---------------------------------------------------------------------------
# Command: /positions
# ---------------------------------------------------------------------------

class TestCmdPositions(unittest.TestCase):

    def test_no_conn_returns_error(self):
        bot = _make_bot()
        reply = bot._cmd_positions()
        self.assertIn("DB 연결 없음", reply)

    def test_empty_db_returns_none_message(self):
        bot = _make_bot(conn=_make_conn())
        reply = bot._cmd_positions()
        self.assertIn("없음", reply)

    def test_open_position_listed(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT", side="long")
        bot = _make_bot(conn=conn)
        reply = bot._cmd_positions()
        self.assertIn("BTCUSDT", reply)
        self.assertIn("LONG", reply)

    def test_closed_position_not_listed(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT", status="closed")
        bot = _make_bot(conn=conn)
        reply = bot._cmd_positions()
        self.assertIn("없음", reply)

    def test_multiple_positions_all_shown(self):
        conn = _make_conn()
        _insert_position(conn, symbol="BTCUSDT")
        _insert_position(conn, symbol="ETHUSDT")
        bot = _make_bot(conn=conn)
        reply = bot._cmd_positions()
        self.assertIn("BTCUSDT", reply)
        self.assertIn("ETHUSDT", reply)


# ---------------------------------------------------------------------------
# Command: /balance
# ---------------------------------------------------------------------------

class TestCmdBalance(unittest.TestCase):

    def test_no_conn_returns_error(self):
        bot = _make_bot()
        reply = bot._cmd_balance()
        self.assertIn("DB 연결 없음", reply)

    def test_unrealized_pnl_shown(self):
        conn = _make_conn()
        _insert_position(conn, unrealized_pnl="250.5")
        bot = _make_bot(conn=conn)
        reply = bot._cmd_balance()
        self.assertIn("250.50", reply)

    def test_today_realized_pnl_shown(self):
        conn = _make_conn()
        _insert_daily(conn, net_pnl=150.0)
        bot = _make_bot(conn=conn)
        reply = bot._cmd_balance()
        self.assertIn("150.00", reply)

    def test_no_positions_shows_zero(self):
        conn = _make_conn()
        bot = _make_bot(conn=conn)
        reply = bot._cmd_balance()
        self.assertIn("0.00", reply)


# ---------------------------------------------------------------------------
# Command: /pause, /resume, /safemode_on, /safemode_off
# ---------------------------------------------------------------------------

class TestCmdSafeMode(unittest.TestCase):

    def test_pause_activates_safe_mode(self):
        sm = MagicMock()
        sm.is_active.return_value = False
        bot = _make_bot(safe_mode=sm)
        reply = bot._cmd_safemode_on("/pause")
        sm.activate.assert_called_once()
        self.assertIn("활성화", reply)

    def test_safemode_on_activates(self):
        sm = MagicMock()
        sm.is_active.return_value = False
        bot = _make_bot(safe_mode=sm)
        bot._cmd_safemode_on("/safemode_on")
        sm.activate.assert_called_once()

    def test_pause_already_active_returns_message(self):
        sm = MagicMock()
        sm.is_active.return_value = True
        sm.reason = "existing"
        bot = _make_bot(safe_mode=sm)
        reply = bot._cmd_safemode_on("/pause")
        sm.activate.assert_not_called()
        self.assertIn("이미 활성", reply)

    def test_resume_deactivates(self):
        sm = MagicMock()
        sm.is_active.return_value = True
        bot = _make_bot(safe_mode=sm)
        reply = bot._cmd_safemode_off("/resume")
        sm.deactivate.assert_called_once_with(by="telegram")
        self.assertIn("해제", reply)

    def test_resume_already_inactive_returns_message(self):
        sm = MagicMock()
        sm.is_active.return_value = False
        bot = _make_bot(safe_mode=sm)
        reply = bot._cmd_safemode_off("/resume")
        sm.deactivate.assert_not_called()
        self.assertIn("이미 비활성", reply)

    def test_no_safe_mode_returns_error(self):
        bot = _make_bot()
        reply = bot._cmd_safemode_on("/pause")
        self.assertIn("없음", reply)


# ---------------------------------------------------------------------------
# Command: /close_all, /close
# ---------------------------------------------------------------------------

class TestCmdClose(unittest.TestCase):

    def test_close_all_calls_emergency_handler(self):
        eh = MagicMock()
        eh.close_all_positions.return_value = {"closed": 2, "failed": 0}
        bot = _make_bot(emergency_handler=eh)
        reply = bot._cmd_close_all()
        eh.close_all_positions.assert_called_once()
        self.assertIn("2", reply)

    def test_close_all_no_handler_returns_error(self):
        bot = _make_bot()
        reply = bot._cmd_close_all()
        self.assertIn("없음", reply)

    def test_close_symbol_success(self):
        eh = MagicMock()
        eh.close_position.return_value = {"status": "closed", "error": None}
        bot = _make_bot(emergency_handler=eh)
        reply = bot._cmd_close("BTCUSDT")
        eh.close_position.assert_called_once_with("BTCUSDT", "텔레그램 /close")
        self.assertIn("✅", reply)

    def test_close_symbol_not_found(self):
        eh = MagicMock()
        eh.close_position.return_value = {"status": "not_found", "error": None}
        bot = _make_bot(emergency_handler=eh)
        reply = bot._cmd_close("BTCUSDT")
        self.assertIn("⚠️", reply)

    def test_close_symbol_failed(self):
        eh = MagicMock()
        eh.close_position.return_value = {"status": "failed", "error": "timeout"}
        bot = _make_bot(emergency_handler=eh)
        reply = bot._cmd_close("BTCUSDT")
        self.assertIn("❌", reply)
        self.assertIn("timeout", reply)

    def test_close_no_symbol_returns_usage(self):
        bot = _make_bot()
        reply = bot._cmd_close("")
        self.assertIn("사용법", reply)


# ---------------------------------------------------------------------------
# Command: /daily, /weekly
# ---------------------------------------------------------------------------

class TestCmdPerformance(unittest.TestCase):

    def test_daily_no_trades(self):
        conn = _make_conn()
        bot = _make_bot(conn=conn)
        reply = bot._cmd_daily()
        self.assertIn("거래 없음", reply)

    def test_daily_shows_stats(self):
        conn = _make_conn()
        _insert_daily(conn, net_pnl=200.0, total_trades=4, wins=3)
        bot = _make_bot(conn=conn)
        reply = bot._cmd_daily()
        self.assertIn("200.00", reply)
        self.assertIn("75.0%", reply)  # 3/4

    def test_daily_no_conn(self):
        bot = _make_bot()
        reply = bot._cmd_daily()
        self.assertIn("DB 연결 없음", reply)

    def test_weekly_no_trades(self):
        conn = _make_conn()
        bot = _make_bot(conn=conn)
        reply = bot._cmd_weekly()
        self.assertIn("거래 없음", reply)

    def test_weekly_aggregates_multiple_days(self):
        conn = _make_conn()
        from datetime import timedelta
        today = date.today()
        mon = today - timedelta(days=today.weekday())
        _insert_daily(conn, net_pnl=100.0, total_trades=2, wins=2,
                      perf_date=mon.isoformat())
        tue = mon + timedelta(days=1)
        _insert_daily(conn, net_pnl=50.0, total_trades=1, wins=0,
                      perf_date=tue.isoformat())
        bot = _make_bot(conn=conn)
        reply = bot._cmd_weekly()
        # 150 total net_pnl
        self.assertIn("150.00", reply)
        # 3 total trades, 2 wins → 66.7%
        self.assertIn("66.7%", reply)

    def test_weekly_no_conn(self):
        bot = _make_bot()
        reply = bot._cmd_weekly()
        self.assertIn("DB 연결 없음", reply)


# ---------------------------------------------------------------------------
# Routing: _handle_update
# ---------------------------------------------------------------------------

class TestHandleUpdate(unittest.IsolatedAsyncioTestCase):

    def _make_update(self, text: str, chat_id: str = "12345",
                     update_id: int = 1) -> dict:
        return {
            "update_id": update_id,
            "message": {
                "text": text,
                "chat": {"id": int(chat_id)},
            },
        }

    async def test_unknown_chat_ignored(self):
        bot = _make_bot(chat_id="12345")
        with patch.object(bot, "_send") as mock_send:
            await bot._handle_update(
                self._make_update("/status", chat_id="99999")
            )
            mock_send.assert_not_called()

    async def test_update_offset_advances(self):
        bot = _make_bot()
        with patch.object(bot, "_send"):
            await bot._handle_update(self._make_update("/status", update_id=42))
        self.assertEqual(bot._update_offset, 43)

    async def test_status_command_routed(self):
        bot = _make_bot()
        with patch.object(bot, "_cmd_status", return_value="ok") as mock_cmd:
            with patch.object(bot, "_send"):
                await bot._handle_update(self._make_update("/status"))
            mock_cmd.assert_called_once()

    async def test_positions_command_routed(self):
        bot = _make_bot()
        with patch.object(bot, "_cmd_positions", return_value="ok"):
            with patch.object(bot, "_send"):
                await bot._handle_update(self._make_update("/positions"))

    async def test_pause_command_routed(self):
        bot = _make_bot()
        with patch.object(bot, "_cmd_safemode_on", return_value="ok") as mock_cmd:
            with patch.object(bot, "_send"):
                await bot._handle_update(self._make_update("/pause"))
            mock_cmd.assert_called_once_with("/pause")

    async def test_resume_command_routed(self):
        bot = _make_bot()
        with patch.object(bot, "_cmd_safemode_off", return_value="ok") as mock_cmd:
            with patch.object(bot, "_send"):
                await bot._handle_update(self._make_update("/resume"))
            mock_cmd.assert_called_once_with("/resume")

    async def test_close_all_command_routed(self):
        bot = _make_bot()
        with patch.object(bot, "_cmd_close_all", return_value="ok"):
            with patch.object(bot, "_send"):
                await bot._handle_update(self._make_update("/close_all"))

    async def test_close_with_symbol_passed(self):
        bot = _make_bot()
        with patch.object(bot, "_cmd_close", return_value="ok") as mock_cmd:
            with patch.object(bot, "_send"):
                await bot._handle_update(self._make_update("/close BTCUSDT"))
        mock_cmd.assert_called_once_with("BTCUSDT")

    async def test_unknown_command_sends_help(self):
        bot = _make_bot()
        with patch.object(bot, "_send") as mock_send:
            await bot._handle_update(self._make_update("/unknown"))
        reply = mock_send.call_args[0][0]
        self.assertIn("알 수 없는 명령어", reply)

    async def test_non_command_message_ignored(self):
        bot = _make_bot()
        with patch.object(bot, "_send") as mock_send:
            await bot._handle_update(self._make_update("hello world"))
        mock_send.assert_not_called()

    async def test_bot_username_suffix_stripped(self):
        """'/status@MyBot' should route to /status."""
        bot = _make_bot()
        with patch.object(bot, "_cmd_status", return_value="ok") as mock_cmd:
            with patch.object(bot, "_send"):
                await bot._handle_update(self._make_update("/status@MyBot"))
        mock_cmd.assert_called_once()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton(unittest.TestCase):

    def setUp(self):
        # Reset singleton before each test
        _tb_module._bot_instance = None

    def tearDown(self):
        _tb_module._bot_instance = None

    def test_get_telegram_bot_returns_instance(self):
        bot = get_telegram_bot()
        self.assertIsInstance(bot, TelegramBot)

    def test_same_instance_on_second_call(self):
        bot1 = get_telegram_bot()
        bot2 = get_telegram_bot()
        self.assertIs(bot1, bot2)

    def test_dependencies_passed_on_first_call(self):
        sm = MagicMock()
        bot = get_telegram_bot(safe_mode=sm)
        self.assertIs(bot._safe_mode, sm)

    def test_second_call_ignores_new_dependencies(self):
        sm1 = MagicMock()
        sm2 = MagicMock()
        get_telegram_bot(safe_mode=sm1)
        bot2 = get_telegram_bot(safe_mode=sm2)
        self.assertIs(bot2._safe_mode, sm1)


if __name__ == "__main__":
    unittest.main()
