"""Position sizing utilities — ATR-based (Binance) and KRW fixed-stop (KRX)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)


def atr_position_size(
    account_balance: float,
    risk_pct: float,
    atr: float,
    atr_multiplier: float = 2.0,
    leverage: int = 1,
) -> float:
    """Return position size in contracts (base asset units).

    Formula:
        risk_amount  = account_balance * risk_pct
        stop_distance = atr * atr_multiplier
        position_size = risk_amount / stop_distance

    Leverage does not affect the contract count — it affects the required
    margin, which is computed separately via :func:`calc_position_margin`.

    Args:
        account_balance: Total USDT account balance.
        risk_pct: Fraction of balance to risk, e.g. 0.01 for 1%.
        atr: ATR(14) value in price units (e.g. USDT per BTC).
        atr_multiplier: Stop distance expressed as multiples of ATR.
        leverage: Informational; not used in size calculation.

    Returns:
        Number of contracts (base asset quantity).
    """
    if atr <= 0:
        raise ValueError(f"atr must be positive, got {atr}")
    if atr_multiplier <= 0:
        raise ValueError(f"atr_multiplier must be positive, got {atr_multiplier}")
    if leverage < 1:
        raise ValueError(f"leverage must be >= 1, got {leverage}")

    risk_amount = account_balance * risk_pct
    stop_distance = atr * atr_multiplier
    return risk_amount / stop_distance


def kelly_risk_pct(
    conn: "sqlite3.Connection",
    strategy_name: str,
    fallback: float,
    *,
    min_trades: int = 6,
    min_rpt: float = 0.002,
    max_rpt: float = 0.008,
) -> float:
    """Return Half-Kelly risk fraction derived from live trade history.

    Queries the last 30 closed trades for *strategy_name* from the positions
    table.  Falls back to *fallback* when:
    - fewer than *min_trades* are available,
    - all trades are wins or all are losses (can't compute R), or
    - Kelly fraction is non-positive (strategy edge is negative).

    Formula:
        W = win_rate, R = avg_win / avg_loss
        f* = W - (1-W)/R
        half_kelly = f* * 0.5
        result = clamp(half_kelly, min_rpt, max_rpt)

    Args:
        conn: SQLite connection with a ``positions`` table.
        strategy_name: Value stored in ``positions.strategy_name``.
        fallback: Risk fraction to return when insufficient data.
        min_trades: Minimum closed trades required to use Kelly.
        min_rpt: Floor for returned risk fraction.
        max_rpt: Ceiling for returned risk fraction.
    """
    try:
        rows = conn.execute(
            "SELECT realized_pnl FROM positions"
            " WHERE strategy_name=? AND status='closed'"
            " AND realized_pnl IS NOT NULL"
            " ORDER BY closed_at DESC LIMIT 30",
            (strategy_name,),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return fallback

    pnls: list[float] = []
    for row in rows:
        try:
            raw = row["realized_pnl"] if hasattr(row, "keys") else row[0]
            pnls.append(float(raw))
        except (TypeError, ValueError):
            continue

    if len(pnls) < min_trades:
        return fallback

    wins   = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]

    if not wins or not losses:
        return fallback

    win_rate        = len(wins) / len(pnls)
    avg_win         = sum(wins) / len(wins)
    avg_loss        = sum(losses) / len(losses)
    reward_to_risk  = avg_win / avg_loss
    kelly           = win_rate - (1 - win_rate) / reward_to_risk

    if kelly <= 0:
        return fallback

    half_kelly = kelly * 0.5
    result = max(min_rpt, min(max_rpt, half_kelly))
    logger.debug(
        "G3 Half-Kelly [%s]: W=%.2f R=%.2f f*=%.4f → half=%.4f → clamped=%.4f",
        strategy_name, win_rate, reward_to_risk, kelly, half_kelly, result,
    )
    return result


def krw_position_size(
    account_krw: float,
    risk_pct: float,
    entry_price: int,
    sl_price: int,
) -> int:
    """Integer share count for KRX spot long-only trading.

    Formula:
        risk_amount    = account_krw * risk_pct
        stop_distance  = entry_price - sl_price   (원)
        shares         = floor(risk_amount / stop_distance)

    Args:
        account_krw: Available KRW balance (원).
        risk_pct: Fraction of balance to risk per trade, e.g. 0.01 = 1%.
        entry_price: Intended entry price (원), already tick-rounded.
        sl_price: Stop-loss price (원), must be < entry_price.

    Returns:
        Integer share quantity ≥ 1, or 0 when stop_distance ≤ 0.
    """
    stop_distance = entry_price - sl_price
    if stop_distance <= 0:
        logger.warning(
            "krw_position_size: stop_distance <= 0 (entry=%d sl=%d) — returning 0",
            entry_price, sl_price,
        )
        return 0
    risk_amount = account_krw * risk_pct
    shares = int(risk_amount / stop_distance)
    return max(1, shares)


def calc_position_margin(
    position_size: float,
    entry_price: float,
    leverage: int,
) -> float:
    """Return required margin in USDT for a leveraged position.

    Formula:
        notional = position_size * entry_price
        margin   = notional / leverage

    Args:
        position_size: Contract quantity (base asset units).
        entry_price: Entry price in USDT.
        leverage: Applied leverage.

    Returns:
        Required margin in USDT.
    """
    if leverage < 1:
        raise ValueError(f"leverage must be >= 1, got {leverage}")

    notional = position_size * entry_price
    return notional / leverage
