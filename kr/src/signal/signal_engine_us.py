"""US signal engine — overrides balance fetch and price handling for USD.

Subclasses SignalEngine, changing only the two KR-specific methods:
  _persist_signal  — uses USD balance + FALLBACK_BALANCE_USD
  _execute_signal  — skips round_to_tick (US has float USD prices)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.signal.signal_engine import SignalEngine

if TYPE_CHECKING:
    from src.signal.base_strategy import SignalResult

_TIMEFRAME_INTERVALS_US: dict[str, int] = {
    "1d": 3600,   # daily strategy: check every hour during market hours
}


class SignalEngineUS(SignalEngine):
    """SignalEngine adapted for US overseas stocks (USD float prices).

    All symbol-iteration, correlation, and DB-persistence logic is inherited
    unchanged.  Only balance sourcing and price rounding differ.
    """

    async def _persist_signal(self, symbol: str, result: "SignalResult") -> None:
        """Validate, persist, and execute a long signal for US spot trading."""
        from src.utils.config import load_config  # lazy
        config = load_config()

        # Balance: paper mode → skip REST (VTS doesn't support US balance).
        # Live mode → fetch USD balance from KIS overseas API.
        is_paper = os.getenv("TRADING_MODE", "paper").strip().lower() != "live"
        balance_usd = 0.0
        if self._kis is not None and not is_paper:
            try:
                raw = await self._kis.fetch_account_balance_us()
                balance_usd = float(raw.get("availableBalance", 0) or 0)
            except Exception as exc:  # noqa: BLE001
                from src.monitoring.logger import get_logger
                get_logger("signal_engine_us").warning("KIS US balance fetch failed: %s", exc)
        if balance_usd <= 0:
            balance_usd = float(os.getenv("FALLBACK_BALANCE_USD", "10000"))

        strategy_name = getattr(self._strategy_runner, "get_symbol_strategy_name",
                                lambda s: None)(symbol) or \
                        getattr(self._strategy_runner, "get_active_strategy_name",
                                lambda: "unknown")()

        entry_price = result.entry_price or 0.0
        stop_loss   = result.sl or 0.0

        blocked = False
        block_reason: str | None = None
        if entry_price <= 0 or stop_loss <= 0:
            blocked, block_reason = True, "missing entry or SL price"
        elif entry_price <= stop_loss:
            blocked, block_reason = True, f"entry {entry_price:.4f} <= sl {stop_loss:.4f}"
        elif balance_usd <= 0:
            blocked, block_reason = True, "zero USD balance"
        else:
            open_count = self._conn.execute(
                "SELECT COUNT(*) FROM positions WHERE status='open'"
            ).fetchone()[0]
            if open_count >= config.max_positions:
                blocked, block_reason = True, f"max_positions={config.max_positions} reached"

        from src.db.models import insert_signal
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
        from src.monitoring.logger import get_logger
        _log = get_logger("signal_engine_us")
        try:
            insert_signal(self._conn, signal_record)
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001
            _log.error("Failed to persist US signal [%s]: %s", symbol, exc)
            return

        if blocked:
            _log.warning("US signal blocked [%s]: %s", symbol, block_reason)
            return

        _log.info("US signal saved [%s]: %s strength=%d", symbol, result.signal_type,
                  result.strength_score)
        await self._execute_signal_us(symbol, result, balance_usd, config, strategy_name)

    async def _execute_signal_us(
        self,
        symbol: str,
        result: "SignalResult",
        balance_usd: float,
        config,
        strategy_name: str,
    ) -> None:
        """Size and submit a US stock buy order (float USD prices, no tick rounding)."""
        from src.monitoring.logger import get_logger
        _log = get_logger("signal_engine_us")

        if self._order_manager is None:
            _log.warning("No OrderManager — US signal not executed [%s]", symbol)
            return

        from src.risk.position_sizer import kelly_risk_pct, usd_position_size

        entry_price = result.entry_price or 0.0
        sl_price    = result.sl or 0.0
        tp1_price   = result.tp1
        tp2_price   = result.tp2

        if entry_price <= 0 or sl_price <= 0 or entry_price <= sl_price:
            _log.warning(
                "Invalid US SL/entry [%s]: entry=%.4f sl=%.4f", symbol, entry_price, sl_price
            )
            return

        risk_pct = kelly_risk_pct(self._conn, strategy_name, config.risk_per_trade)
        quantity = usd_position_size(balance_usd, risk_pct, entry_price, sl_price)
        if quantity <= 0:
            _log.warning("US position size zero [%s] — skipping", symbol)
            return

        _log.info(
            "signal.us.execute [%s] entry=%.4f sl=%.4f tp1=%s tp2=%s qty=%d risk_pct=%.4f",
            symbol, entry_price, sl_price, tp1_price, tp2_price, quantity, risk_pct,
        )

        try:
            order = {
                "symbol":        symbol,
                "side":          "buy",
                "quantity":      quantity,
                "price":         entry_price,
                "sl":            sl_price,
                "tp1":           tp1_price,
                "tp2":           tp2_price,
                "strategy_name": strategy_name,
            }
            result_dict = await self._order_manager.submit_and_confirm(order)
            _log.info("signal.us.filled [%s]: position_id=%s",
                      symbol, result_dict.get("position_id", "?"))
        except Exception as exc:  # noqa: BLE001
            _log.error("signal.us.execute_failed [%s]: %s", symbol, exc)
