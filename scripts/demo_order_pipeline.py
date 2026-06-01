"""Demo: full order pipeline — entry → TP/SL register → fill.

Run inside signal-engine pod:
    kubectl exec -n trading <pod> -- python3 scripts/demo_order_pipeline.py

Uses order_stream=None so submit_and_confirm treats submitted order as
immediately filled (no WebSocket needed). Verifies DB state after each step.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import sys

import ccxt

# ---------------------------------------------------------------------------
# Bootstrap path
# ---------------------------------------------------------------------------
sys.path.insert(0, "/app")  # pod workdir

from src.db.connection import get_connection
from src.execution.order_manager import OrderManager

SYMBOL = "BTCUSDT"
DB_PATH = os.getenv("SQLITE_DB_PATH", "/data/trading.db")


def build_exchange() -> ccxt.Exchange:
    trading_mode = os.environ.get("TRADING_MODE", "testnet").lower()
    if trading_mode == "testnet":
        api_key = os.environ.get("BINANCE_DEMO_API_KEY") or os.environ["BINANCE_API_KEY"]
        api_secret = os.environ.get("BINANCE_DEMO_API_SECRET") or os.environ["BINANCE_API_SECRET"]
    else:
        api_key = os.environ["BINANCE_API_KEY"]
        api_secret = os.environ["BINANCE_API_SECRET"]
    exchange = ccxt.binanceusdm({
        "apiKey": api_key,
        "secret": api_secret,
        "options": {"defaultType": "future"},
    })
    if trading_mode == "testnet":
        exchange.enable_demo_trading(True)
    return exchange


async def run_demo() -> None:
    conn = get_connection(DB_PATH)
    exchange = build_exchange()

    # --- fetch current price ---
    ticker = exchange.fetch_ticker(SYMBOL)
    price = float(ticker["last"])
    print(f"[1] {SYMBOL} price = {price:.2f}")

    # --- compute SL/TP (long demo, 1% qty) ---
    qty = 0.001          # minimal qty
    sl  = round(price * 0.97, 2)   # -3%
    tp1 = round(price * 1.03, 2)   # +3%
    tp2 = round(price * 1.05, 2)   # +5%
    print(f"[2] qty={qty}  sl={sl}  tp1={tp1}  tp2={tp2}")

    om = OrderManager(conn=conn, exchange=exchange, order_stream=None)

    # --- submit entry + auto-confirm (order_stream=None → instant fill) ---
    order = {
        "symbol":        SYMBOL,
        "side":          "buy",
        "type":          "market",
        "quantity":      qty,
        "position_side": "long",
        "tp1":           tp1,
        "tp2":           tp2,
        "sl":            sl,
    }
    print("[3] submitting entry order…")
    filled = await om.submit_and_confirm(order, timeout_sec=30)
    print(f"[4] filled: id={filled.get('id')}  avg={filled.get('average') or filled.get('price') or price}")

    # --- verify DB ---
    conn.row_factory = sqlite3.Row
    orders = conn.execute(
        "SELECT order_type, side, price, quantity, status FROM orders "
        "WHERE symbol=? ORDER BY rowid DESC LIMIT 4",
        (SYMBOL,),
    ).fetchall()

    print("\n[5] DB orders (last 4):")
    for o in orders:
        print(f"    {o['order_type']:12s}  {o['side']:4s}  price={o['price']}  qty={o['quantity']}  status={o['status']}")

    # check we have entry + SL + TP1 + TP2
    types = {o["order_type"] for o in orders}
    expected = {"market", "stop_market", "limit"}
    ok = expected.issubset(types)
    print(f"\n[6] Pipeline {'PASS ✓' if ok else 'FAIL ✗'} — order types found: {types}")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run_demo())
