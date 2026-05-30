"""Test all Telegram command handlers + send capability.

Run inside signal-engine or safety-monitor pod:
    kubectl exec -n trading <pod> -- python3 scripts/test_telegram_commands.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "/app")

from src.db.connection import get_connection
from src.monitoring.telegram_bot import TelegramBot

DB_PATH = os.getenv("SQLITE_DB_PATH", "/data/trading.db")

PASS = "PASS"
FAIL = "FAIL"

results: list[tuple[str, str, str]] = []  # (cmd, status, reply)


def check(cmd: str, reply: str) -> None:
    ok = bool(reply) and "오류" not in reply and "None" not in reply
    results.append((cmd, PASS if ok else FAIL, reply[:80].replace("\n", " ")))


def main() -> None:
    conn = get_connection(DB_PATH)
    conn.row_factory = __import__("sqlite3").Row

    bot = TelegramBot(conn=conn)  # no safe_mode / emergency_handler wired

    # --- test each read-only handler ---
    check("/status",    bot._cmd_status())
    check("/positions", bot._cmd_positions())
    check("/balance",   bot._cmd_balance())
    check("/daily",     bot._cmd_daily())
    check("/weekly",    bot._cmd_weekly())

    # --- safe_mode/emergency handlers without instances → should return graceful msg ---
    check("/pause",     bot._cmd_safemode_on("/pause"))
    check("/resume",    bot._cmd_safemode_off("/resume"))
    check("/close_all", bot._cmd_close_all())
    check("/close",     bot._cmd_close(""))  # empty symbol → usage hint

    # --- actual Telegram send test ---
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if token and chat_id:
        bot2 = TelegramBot(token=token, chat_id=chat_id)
        bot2.send_info("[TEST] Telegram command test — 시스템 정상")
        check("send_info", "sent OK")  # if no exception, passed
        print("[*] Telegram send_info dispatched — check your chat")
    else:
        print("[!] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — skip live send")

    # --- print results ---
    print()
    print(f"{'CMD':<15} {'STATUS':<6} REPLY")
    print("-" * 70)
    fail_count = 0
    for cmd, status, reply in results:
        print(f"{cmd:<15} {status:<6} {reply}")
        if status == FAIL:
            fail_count += 1

    print()
    total = len(results)
    print(f"Result: {total - fail_count}/{total} passed")
    if fail_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
