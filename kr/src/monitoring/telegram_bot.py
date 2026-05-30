"""Telegram notification and remote-control bot.

Responsibilities:
  - Send info / warning / critical alerts to a Telegram chat (fire-and-forget;
    failures are logged, never raised).
  - Expose ``send_alert(message)`` generic method used by SafeMode and
    EmergencyHandler.
  - Run an async polling loop that dispatches /command messages to handlers
    which wrap SafeMode and EmergencyHandler.

Environment variables:
  TELEGRAM_BOT_TOKEN   Bot API token from @BotFather.
  TELEGRAM_CHAT_ID     Target chat (user ID or group ID).

Usage::

    from src.monitoring.telegram_bot import get_telegram_bot

    bot = get_telegram_bot(safe_mode=sm, emergency_handler=eh, conn=db_conn)
    bot.start_polling()          # launches asyncio background task
    bot.send_critical("test")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import date, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)

_TELEGRAM_BASE = "https://api.telegram.org/bot"


class TelegramBot:
    """Telegram bot wrapper with fire-and-forget sends and async polling.

    Args:
        token: Bot API token. Falls back to ``TELEGRAM_BOT_TOKEN`` env var.
        chat_id: Target chat ID. Falls back to ``TELEGRAM_CHAT_ID`` env var.
        safe_mode: :class:`~src.safety.safe_mode.SafeMode` instance for
            /pause, /resume, /safemode_on, /safemode_off commands.
        emergency_handler: :class:`~src.safety.emergency_handler.EmergencyHandler`
            instance for /close_all and /close commands.
        conn: SQLite connection for /positions, /balance, /daily, /weekly queries.
    """

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        safe_mode=None,
        emergency_handler=None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        self._token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self._safe_mode = safe_mode
        self._emergency_handler = emergency_handler
        self._conn = conn
        self._update_offset: int = 0
        self._polling_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public send API
    # ------------------------------------------------------------------

    def send_info(self, message: str) -> None:
        """Send an ℹ️ informational message."""
        self._send(f"ℹ️ {message}")

    def send_warning(self, message: str) -> None:
        """Send a ⚠️ warning message."""
        self._send(f"⚠️ {message}")

    def send_critical(self, message: str) -> None:
        """Send a 🚨 critical / emergency message."""
        self._send(f"🚨 {message}")

    def send_alert(self, message: str) -> None:
        """Generic alert — used by SafeMode and EmergencyHandler."""
        self._send(message)

    # ------------------------------------------------------------------
    # Polling lifecycle
    # ------------------------------------------------------------------

    def start_polling(self) -> None:
        """Schedule the async polling loop as a background asyncio task.

        Safe to call multiple times — does nothing if already running.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if self._polling_task is None or self._polling_task.done():
            self._polling_task = loop.create_task(self._poll_loop())
            logger.info("Telegram polling task started")

    def stop_polling(self) -> None:
        """Cancel the polling task if running."""
        if self._polling_task and not self._polling_task.done():
            self._polling_task.cancel()
            logger.info("Telegram polling task cancelled")

    # ------------------------------------------------------------------
    # Polling internals
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        logger.info("Telegram command polling loop started")
        while True:
            try:
                updates = await asyncio.to_thread(self._get_updates)
                for update in updates:
                    await self._handle_update(update)
            except asyncio.CancelledError:
                logger.info("Telegram polling loop stopped")
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("Telegram polling error: %s", exc)
                await asyncio.sleep(5)
                continue
            await asyncio.sleep(1)

    def _get_updates(self) -> list[dict]:
        """Fetch new updates via long-polling getUpdates. Returns [] on error."""
        if not self._token:
            return []
        try:
            params = urllib.parse.urlencode({"offset": self._update_offset, "timeout": 30})
            url = f"{_TELEGRAM_BASE}{self._token}/getUpdates?{params}"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=35) as resp:
                body = json.loads(resp.read())
                if not body.get("ok"):
                    logger.warning("getUpdates not ok: %s", body)
                    return []
                return body.get("result", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("getUpdates failed: %s", exc)
            return []

    async def _handle_update(self, update: dict) -> None:
        """Route a single Telegram update to the appropriate command handler."""
        update_id = update.get("update_id", 0)
        self._update_offset = max(self._update_offset, update_id + 1)

        message = update.get("message") or update.get("edited_message")
        if not message:
            return

        text = (message.get("text") or "").strip()
        if not text.startswith("/"):
            return

        # Security: only respond to configured chat
        incoming_chat_id = str(message.get("chat", {}).get("id", ""))
        if self._chat_id and incoming_chat_id != self._chat_id:
            logger.warning("Ignoring command from unknown chat_id=%s", incoming_chat_id)
            return

        parts = text.split(maxsplit=1)
        # Strip @BotUsername suffix from command if present
        cmd = parts[0].split("@")[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        logger.info("Telegram command received: %s arg=%r", cmd, arg)

        if cmd == "/status":
            reply = self._cmd_status()
        elif cmd == "/positions":
            reply = self._cmd_positions()
        elif cmd == "/balance":
            reply = self._cmd_balance()
        elif cmd in ("/pause", "/safemode_on"):
            reply = self._cmd_safemode_on(cmd)
        elif cmd in ("/resume", "/safemode_off"):
            reply = self._cmd_safemode_off(cmd)
        elif cmd == "/close_all":
            reply = await asyncio.to_thread(self._cmd_close_all)
        elif cmd == "/close":
            symbol = arg.upper()
            reply = await asyncio.to_thread(self._cmd_close, symbol)
        elif cmd == "/daily":
            reply = self._cmd_daily()
        elif cmd == "/weekly":
            reply = self._cmd_weekly()
        else:
            reply = (
                f"알 수 없는 명령어: {cmd}\n"
                "사용 가능: /status /positions /balance /pause /resume "
                "/safemode_on /safemode_off /close_all /close /daily /weekly"
            )

        self._send(reply)

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _cmd_status(self) -> str:
        lines = ["📊 <b>시스템 상태</b>"]
        if self._safe_mode is not None:
            if self._safe_mode.is_active():
                lines.append(f"SafeMode: 🔴 활성 — {self._safe_mode.reason}")
            else:
                lines.append("SafeMode: 🟢 정상")
        else:
            lines.append("SafeMode: N/A")
        return "\n".join(lines)

    def _cmd_positions(self) -> str:
        if self._conn is None:
            return "DB 연결 없음"
        rows = self._conn.execute(
            """SELECT symbol, side, quantity, entry_price, unrealized_pnl
               FROM positions WHERE status='open'
               ORDER BY opened_at"""
        ).fetchall()
        if not rows:
            return "오픈 포지션 없음"
        lines = ["📋 <b>오픈 포지션</b>"]
        for r in rows:
            pnl = float(r["unrealized_pnl"] or 0)
            sign = "+" if pnl >= 0 else ""
            lines.append(
                f"• {r['symbol']} {r['side'].upper()} qty={r['quantity']} "
                f"entry={r['entry_price']} uPnL={sign}{pnl:.2f}"
            )
        return "\n".join(lines)

    def _cmd_balance(self) -> str:
        if self._conn is None:
            return "DB 연결 없음"
        row = self._conn.execute(
            """SELECT SUM(CAST(unrealized_pnl AS REAL)) AS unrealized
               FROM positions WHERE status='open'"""
        ).fetchone()
        unrealized = float(row["unrealized"] or 0) if row else 0.0
        today = date.today().isoformat()
        perf = self._conn.execute(
            "SELECT net_pnl FROM daily_performance WHERE perf_date=?",
            (today,),
        ).fetchone()
        today_pnl = float(perf["net_pnl"] or 0) if perf else 0.0
        return (
            "💰 <b>잔고 / P&amp;L</b>\n"
            f"미실현 PnL: {unrealized:+.2f} USDT\n"
            f"오늘 실현 PnL: {today_pnl:+.2f} USDT"
        )

    def _cmd_safemode_on(self, cmd: str) -> str:
        if self._safe_mode is None:
            return "SafeMode 인스턴스 없음"
        if self._safe_mode.is_active():
            return f"SafeMode 이미 활성 — 이유: {self._safe_mode.reason}"
        trigger = "pause" if cmd == "/pause" else "safemode_on"
        self._safe_mode.activate(reason=f"텔레그램 수동 ({trigger})")
        return "🔴 SafeMode 활성화됨"

    def _cmd_safemode_off(self, cmd: str) -> str:
        if self._safe_mode is None:
            return "SafeMode 인스턴스 없음"
        if not self._safe_mode.is_active():
            return "SafeMode 이미 비활성"
        self._safe_mode.deactivate(by="telegram")
        return "🟢 SafeMode 해제됨"

    def _cmd_close_all(self) -> str:
        if self._emergency_handler is None:
            return "EmergencyHandler 없음"
        result = self._emergency_handler.close_all_positions("텔레그램 /close_all")
        return (
            f"🚨 긴급 전량 청산 완료\n"
            f"성공: {result['closed']}건 / 실패: {result['failed']}건"
        )

    def _cmd_close(self, symbol: str) -> str:
        if not symbol:
            return "사용법: /close BTCUSDT"
        if self._emergency_handler is None:
            return "EmergencyHandler 없음"
        result = self._emergency_handler.close_position(symbol, "텔레그램 /close")
        if result["status"] == "closed":
            return f"✅ {symbol} 청산 완료"
        if result["status"] == "not_found":
            return f"⚠️ {symbol} 오픈 포지션 없음"
        return f"❌ {symbol} 청산 실패: {result.get('error')}"

    def _cmd_daily(self) -> str:
        if self._conn is None:
            return "DB 연결 없음"
        today = date.today().isoformat()
        row = self._conn.execute(
            """SELECT total_trades, winning_trades, losing_trades,
                      net_pnl, gross_profit, gross_loss, total_fees
               FROM daily_performance WHERE perf_date=?""",
            (today,),
        ).fetchone()
        if row is None:
            return f"오늘({today}) 거래 없음"
        total = row["total_trades"] or 0
        win_rate = (row["winning_trades"] / total * 100) if total else 0.0
        return (
            f"📊 <b>오늘 성과 ({today})</b>\n"
            f"총 거래: {total}건\n"
            f"승률: {win_rate:.1f}%\n"
            f"순손익: {float(row['net_pnl'] or 0):+.2f} USDT\n"
            f"총수익: {float(row['gross_profit'] or 0):+.2f} USDT\n"
            f"총손실: -{float(row['gross_loss'] or 0):.2f} USDT\n"
            f"수수료: {float(row['total_fees'] or 0):.2f} USDT"
        )

    def _cmd_weekly(self) -> str:
        if self._conn is None:
            return "DB 연결 없음"
        today = date.today()
        week_start = (today - timedelta(days=today.weekday())).isoformat()
        row = self._conn.execute(
            """SELECT SUM(total_trades)           AS trades,
                      SUM(winning_trades)         AS wins,
                      SUM(CAST(net_pnl AS REAL))  AS net,
                      SUM(CAST(total_fees AS REAL)) AS fees
               FROM daily_performance WHERE perf_date >= ?""",
            (week_start,),
        ).fetchone()
        if row is None or row["trades"] is None:
            return f"이번 주({week_start}~) 거래 없음"
        total = row["trades"] or 0
        win_rate = (row["wins"] / total * 100) if total else 0.0
        return (
            f"📈 <b>이번 주 성과 ({week_start}~)</b>\n"
            f"총 거래: {total}건\n"
            f"승률: {win_rate:.1f}%\n"
            f"순손익: {float(row['net'] or 0):+.2f} USDT\n"
            f"수수료: {float(row['fees'] or 0):.2f} USDT"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _send(self, text: str) -> None:
        """POST sendMessage to Telegram; silently ignores all failures."""
        if not self._token or not self._chat_id:
            logger.warning("Telegram not configured (token=%s, chat_id=%s); message skipped: %.80s",
                           bool(self._token), bool(self._chat_id), text)
            return
        import urllib.error
        try:
            url = f"{_TELEGRAM_BASE}{self._token}/sendMessage"
            payload = json.dumps(
                {"chat_id": self._chat_id, "text": text, "parse_mode": "HTML"}
            ).encode()
            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    logger.warning("Telegram sendMessage HTTP %d", resp.status)
                else:
                    logger.info("Telegram message sent OK (chat_id=%.8s, len=%d)", self._chat_id, len(text))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            logger.warning("Telegram send failed HTTP %d: %s", exc.code, body)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telegram send failed: %s", exc)


# ------------------------------------------------------------------
# Singleton
# ------------------------------------------------------------------

_bot_instance: TelegramBot | None = None


def get_telegram_bot(
    safe_mode=None,
    emergency_handler=None,
    conn: sqlite3.Connection | None = None,
) -> TelegramBot:
    """Return the global TelegramBot singleton.

    On first call creates the instance from env vars and supplied dependencies.
    Subsequent calls return the cached instance — dependency arguments are
    ignored after initialisation.

    Args:
        safe_mode: SafeMode instance for /pause, /resume, /safemode_* commands.
        emergency_handler: EmergencyHandler for /close_all, /close commands.
        conn: SQLite connection for query-based commands.

    Returns:
        The singleton :class:`TelegramBot` instance.
    """
    global _bot_instance
    if _bot_instance is None:
        _bot_instance = TelegramBot(
            safe_mode=safe_mode,
            emergency_handler=emergency_handler,
            conn=conn,
        )
    return _bot_instance
