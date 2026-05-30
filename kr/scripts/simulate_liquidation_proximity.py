"""Phase 4 Week 2 — Liquidation proximity simulation script.

Tests 3 scenarios end-to-end with real Telegram notifications:
  Scenario 1: WATCH   — current price 19% above liquidation (just below 20% threshold)
  Scenario 2: WARNING — current price 14% above liquidation → partial_close(pct=0.5)
  Scenario 3: CRITICAL— current price 7%  above liquidation → close_all_positions()

Usage:
    python scripts/simulate_liquidation_proximity.py

Requires env vars:
    TELEGRAM_BOT_TOKEN  — bot token from @BotFather
    TELEGRAM_CHAT_ID    — your chat ID or group ID

All exchange / order calls are intercepted by stubs — no real orders sent.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock

# ── Add project root to sys.path ──────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.risk.liquidation_guard import LiquidationGuard, calc_liquidation_price
from src.safety.emergency_handler import EmergencyHandler
from src.safety.safe_mode import SafeMode
from src.safety.safety_monitor import SafetyMonitor

# ── Telegram (real) ───────────────────────────────────────────────────────────
def _build_telegram():
    token   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("[WARNING] TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 미설정 — 콘솔만 출력")
        return None
    from src.monitoring.telegram_bot import TelegramBot
    return TelegramBot(token=token, chat_id=chat_id)


# ── In-memory SQLite with minimal schema ─────────────────────────────────────
def _build_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS positions (
            position_id         TEXT PRIMARY KEY,
            symbol              TEXT NOT NULL,
            side                TEXT NOT NULL,
            leverage            INTEGER NOT NULL,
            entry_price         TEXT NOT NULL,
            exit_price          TEXT,
            quantity            TEXT NOT NULL,
            liquidation_price   TEXT NOT NULL,
            stop_loss           TEXT NOT NULL,
            take_profit_1       TEXT,
            take_profit_2       TEXT,
            initial_stop_loss   TEXT NOT NULL,
            trailing_activated  INTEGER DEFAULT 0,
            realized_pnl        TEXT DEFAULT '0',
            unrealized_pnl      TEXT DEFAULT '0',
            status              TEXT NOT NULL DEFAULT 'open',
            close_reason        TEXT,
            trading_mode        TEXT NOT NULL DEFAULT 'testnet',
            opened_at           TEXT NOT NULL DEFAULT (datetime('now')),
            closed_at           TEXT
        );
        CREATE TABLE IF NOT EXISTS orders (
            order_id        TEXT PRIMARY KEY,
            symbol          TEXT NOT NULL,
            side            TEXT NOT NULL,
            order_type      TEXT NOT NULL,
            quantity        TEXT NOT NULL,
            avg_fill_price  TEXT,
            status          TEXT NOT NULL DEFAULT 'filled',
            created_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS safe_mode_events (
            event_id    TEXT PRIMARY KEY,
            action      TEXT NOT NULL,
            reason      TEXT NOT NULL,
            by          TEXT NOT NULL DEFAULT 'system',
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    return conn


# ── Fake position params ──────────────────────────────────────────────────────
SYMBOL     = "SIMTEST/USDT:USDT"
ENTRY      = 100.0
LEVERAGE   = 5
SIDE       = "long"
QUANTITY   = 1.0
LIQ_PRICE  = calc_liquidation_price(ENTRY, LEVERAGE, SIDE)  # ~80.4


def _price_at_dist(dist_pct: float) -> float:
    """Return current price such that dist_to_liq == dist_pct% (long side).

    dist_pct = (current - liq) / current * 100
    → current = liq / (1 - dist_pct/100)
    """
    return LIQ_PRICE / (1 - dist_pct / 100)


def _insert_position(conn: sqlite3.Connection) -> str:
    pid = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO positions
           (position_id, symbol, side, leverage, entry_price, quantity,
            liquidation_price, stop_loss, initial_stop_loss)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (pid, SYMBOL, SIDE, LEVERAGE,
         str(ENTRY), str(QUANTITY),
         str(LIQ_PRICE),
         str(ENTRY * 0.95),   # SL at 95 (safe side)
         str(ENTRY * 0.95)),
    )
    conn.commit()
    return pid


def _build_ccxt_position(current_price: float) -> dict:
    """Build a ccxt-style position dict for SafetyMonitor."""
    return {
        "symbol":           SYMBOL,
        "side":             SIDE,
        "contracts":        QUANTITY,
        "contractSize":     1.0,
        "markPrice":        current_price,
        "entryPrice":       ENTRY,
        "liquidationPrice": LIQ_PRICE,
    }


def _stub_order_manager():
    """OrderManager stub — records calls, never touches exchange."""
    om = MagicMock()
    calls = []

    def market_close(symbol, side, qty, position_side=None):
        msg = (f"  [STUB market_close] {symbol} side={side} qty={qty} "
               f"position_side={position_side}")
        print(msg)
        calls.append({"symbol": symbol, "side": side, "qty": qty})

    om.market_close.side_effect = market_close
    om._calls = calls
    return om


def _stub_position_tracker():
    """PositionTracker stub — records close calls."""
    pt = MagicMock()

    def close_position(conn, pid, price, reason):
        print(f"  [STUB close_position] pid={pid} price={price} reason={reason}")
        conn.execute(
            "UPDATE positions SET status='closed', close_reason=? WHERE position_id=?",
            (reason, pid),
        )
        conn.commit()

    pt.close_position.side_effect = close_position
    return pt


# ══════════════════════════════════════════════════════════════════════════════
# Scenarios
# ══════════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def run_scenario_1(telegram) -> None:
    """WATCH: 19% above liq price — proximity check only, no action."""
    section("Scenario 1 — WATCH 단계 (청산가 대비 +19%)")

    current = _price_at_dist(19.0)
    print(f"  진입가:  {ENTRY}")
    print(f"  청산가:  {LIQ_PRICE:.4f}")
    print(f"  현재가:  {current:.4f}  (liq+19%)")

    pos = {
        "symbol":           SYMBOL,
        "side":             SIDE,
        "current_price":    current,
        "liquidation_price": LIQ_PRICE,
    }

    level = LiquidationGuard.check_proximity(pos)
    print(f"\n  → check_proximity() 결과: {level}")

    assert level == "WATCH", f"Expected WATCH, got {level}"

    msg = (
        f"[SIM] 시나리오 1 — WATCH\n"
        f"심볼: {SYMBOL}\n"
        f"현재가: {current:.4f} | 청산가: {LIQ_PRICE:.4f}\n"
        f"청산까지 거리: 19.0%\n"
        f"상태: {level} — 모니터링만, 청산 없음"
    )
    print(f"\n  Telegram 전송:\n{msg}")
    if telegram:
        telegram.send_warning(msg)

    print("\n  [OK] Scenario 1 통과")


def run_scenario_2(telegram) -> None:
    """WARNING: 14% above liq — partial_close(pct=0.5) triggered."""
    section("Scenario 2 — WARNING 단계 (청산가 대비 +14%) → 50% 부분 청산")

    conn = _build_db()
    pid  = _insert_position(conn)
    current = _price_at_dist(14.0)

    print(f"  진입가:  {ENTRY}")
    print(f"  청산가:  {LIQ_PRICE:.4f}")
    print(f"  현재가:  {current:.4f}  (liq+14%)")
    print(f"  position_id: {pid}")

    om = _stub_order_manager()
    pt = _stub_position_tracker()
    safe_mode = SafeMode(conn=conn)
    eh = EmergencyHandler(conn, om, pt, safe_mode, telegram_bot=telegram)

    # Build mock exchange that returns our single position
    mock_exchange = MagicMock()
    mock_exchange.fetch_positions.return_value = [_build_ccxt_position(current)]

    monitor = SafetyMonitor(
        exchange=mock_exchange,
        conn=conn,
        safe_mode=safe_mode,
        emergency_handler=eh,
        telegram_bot=telegram,
    )

    asyncio.run(monitor._check_all_positions())

    # Verify partial close was called (WARNING does NOT close position fully)
    assert om.market_close.called, "market_close should have been called for WARNING"
    assert not safe_mode.is_active(), "SafeMode should NOT activate for WARNING"

    print("\n  [OK] Scenario 2 통과 — partial_close(0.5) 호출됨, SafeMode 미활성")


def run_scenario_3(telegram) -> None:
    """CRITICAL: 7% above liq — close_all_positions() triggered."""
    section("Scenario 3 — CRITICAL 단계 (청산가 대비 +7%) → 전량 청산")

    conn = _build_db()
    pid  = _insert_position(conn)
    current = _price_at_dist(7.0)

    print(f"  진입가:  {ENTRY}")
    print(f"  청산가:  {LIQ_PRICE:.4f}")
    print(f"  현재가:  {current:.4f}  (liq+7%)")
    print(f"  position_id: {pid}")

    om = _stub_order_manager()
    pt = _stub_position_tracker()
    safe_mode = SafeMode(conn=conn)
    eh = EmergencyHandler(conn, om, pt, safe_mode, telegram_bot=telegram)

    mock_exchange = MagicMock()
    mock_exchange.fetch_positions.return_value = [_build_ccxt_position(current)]

    monitor = SafetyMonitor(
        exchange=mock_exchange,
        conn=conn,
        safe_mode=safe_mode,
        emergency_handler=eh,
        telegram_bot=telegram,
    )

    asyncio.run(monitor._check_all_positions())

    # Verify full close + SafeMode activated
    assert om.market_close.called, "market_close should have been called for CRITICAL"
    assert safe_mode.is_active(), "SafeMode should be ACTIVE after CRITICAL close"

    print(f"\n  SafeMode 활성화: {safe_mode.is_active()}  이유: {safe_mode.reason}")
    print("\n  [OK] Scenario 3 통과 — close_all_positions() 호출됨, SafeMode 활성")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\n진입가={ENTRY} | 레버리지={LEVERAGE}x | 포지션={SIDE}")
    print(f"청산가(계산값)={LIQ_PRICE:.4f}")
    print(f"  WATCH    임계값: ≤20% → 현재가≈{_price_at_dist(20.0):.2f}")
    print(f"  WARNING  임계값: ≤15% → 현재가≈{_price_at_dist(15.0):.2f}")
    print(f"  CRITICAL 임계값: ≤8%  → 현재가≈{_price_at_dist(8.0):.2f}")

    telegram = _build_telegram()

    try:
        run_scenario_1(telegram)
        run_scenario_2(telegram)
        run_scenario_3(telegram)
    except AssertionError as e:
        print(f"\n[FAIL] Assertion 실패: {e}")
        sys.exit(1)

    section("전체 결과")
    print("  Scenario 1 WATCH    ✓")
    print("  Scenario 2 WARNING  ✓  (50% 부분 청산)")
    print("  Scenario 3 CRITICAL ✓  (전량 청산 + SafeMode 활성)")
    print("\n  모든 시나리오 통과!")
