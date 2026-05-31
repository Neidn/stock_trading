"""StrategyRunner — dynamic strategy loader and execution wrapper.

Reads ACTIVE_STRATEGY and STRATEGY_PARAMS from environment variables and
dispatches signal generation to the appropriate strategy instance.  This
file must never be modified when strategies are added or swapped.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import sqlite3
import time
from typing import TYPE_CHECKING

from src.signal.base_strategy import BaseStrategy, SignalResult

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def _tp_scale_from_adx(btc_adx: float | None) -> float:
    """Regime-adaptive TP multiplier based on BTC ADX.

    ADX 25 = neutral (×1.0). Each ADX point above/below shifts by 1%.
    Clamped to [0.75, 1.25]: never more than ±25% from baseline.

    Trending (ADX>25): wider TP — ride the trend.
    Ranging  (ADX<25): tighter TP — exit before mean-reversion kills gains.
    """
    if btc_adx is None:
        return 1.0
    return max(0.75, min(1.25, 1.0 + (btc_adx - 25.0) * 0.01))


class StrategyRunner:
    """Loads a strategy by name and proxies signal generation.

    The active strategy is determined solely by the ``ACTIVE_STRATEGY``
    environment variable.  Swapping strategies at runtime (or via ConfigMap
    update + pod restart) requires no code changes.

    Environment variables:
        ACTIVE_STRATEGY: snake_case name matching a file under
            ``src/signal/strategies/`` (e.g. ``rsi_macd``).
        STRATEGY_PARAMS: JSON string of strategy-specific parameters.
            Falls back to ``{}`` on parse failure.

    Example:
        runner = StrategyRunner()
        signal = runner.run(df, "BTCUSDT")
        if signal.is_actionable():
            ...
    """

    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        """Load the active strategy from environment variables."""
        self._conn = conn
        self._strategy_name: str = os.environ.get("ACTIVE_STRATEGY", "").strip()
        if not self._strategy_name:
            raise ValueError(
                "ACTIVE_STRATEGY 환경변수가 설정되지 않았습니다. "
                "예: ACTIVE_STRATEGY=rsi_macd"
            )
        params = self._load_params(self._strategy_name)
        self._strategy: BaseStrategy = self._load_strategy(self._strategy_name, params)
        # Per-symbol cache: {strategy_name: (BaseStrategy, loaded_at_monotonic)}
        self._symbol_strategy_cache: dict[str, tuple[BaseStrategy, float]] = {}
        self._CACHE_TTL = 300.0  # re-check DB every 5 min

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reload_if_changed(self) -> None:
        """Reload global strategy if ACTIVE_STRATEGY env var changed since init.

        Also triggers G2 intraday regime refresh every 30 min: re-assigns
        per-symbol strategies when BTC ADX shifts >5 since last assignment.
        No-op in normal k8s (env vars fixed at pod start). Useful when env is
        updated live (e.g. exec into pod) or configmap mounted as a file.
        """
        current = os.environ.get("ACTIVE_STRATEGY", "").strip()
        if current and current != self._strategy_name:
            logger.info("ACTIVE_STRATEGY changed: %s → %s, reloading", self._strategy_name, current)
            self._strategy_name = current
            self._strategy = self._load_strategy(current, self._load_params(current))
            self._symbol_strategy_cache.clear()

        # KRX: regime refresh disabled (no BTC/futures reference in KRX spot universe)

    def _get_symbol_strategy(self, symbol: str) -> BaseStrategy:
        """Return per-symbol strategy if set in DB, else global strategy.

        Cache TTL = 5 min so DB changes reflect without pod restart.
        Per-symbol strategies use STRATEGY_PARAMS_<NAME> env var if set,
        else fall back to global STRATEGY_PARAMS.
        """
        if self._conn is None:
            return self._strategy
        try:
            row = self._conn.execute(
                "SELECT strategy FROM symbols WHERE symbol=? AND is_active=1 LIMIT 1",
                (symbol,),
            ).fetchone()
            override = row[0] if row and row[0] else None
        except Exception:  # noqa: BLE001
            return self._strategy

        if not override or override == self._strategy_name:
            return self._strategy

        import time
        now = time.monotonic()
        cached = self._symbol_strategy_cache.get(override)
        if cached and (now - cached[1]) < self._CACHE_TTL:
            return cached[0]

        try:
            params = self._load_params(override)
            instance = self._load_strategy(override, params)
            self._symbol_strategy_cache[override] = (instance, now)
            logger.info("Per-symbol strategy loaded [%s]: %s", symbol, override)
            return instance
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load per-symbol strategy %s: %s", override, exc)
            return self._strategy

    def run(self, df: "pd.DataFrame", symbol: str) -> SignalResult:
        """Run the active strategy and return a signal.

        Execution order:
        1. SignalBlocker check — if blocked, return 'none' immediately.
        2. Data sufficiency check — if too few candles, return 'none'.
        3. Strategy execution — exceptions are caught and logged.

        Per-symbol strategy override: if ``symbols.strategy`` is set for this
        symbol, that strategy is used instead of the global ``ACTIVE_STRATEGY``.

        Args:
            df: OHLCV DataFrame sorted ascending by time.
            symbol: Trading pair, e.g. ``'BTCUSDT'``.

        Returns:
            :class:`SignalResult` — never raises.
        """
        # 1. SignalBlocker (stub-safe: import only if implemented)
        blocked, block_reason = self._check_signal_blocker(symbol)
        if blocked:
            logger.info("신호 차단 [%s]: %s", symbol, block_reason)
            return SignalResult(signal_type="none", reason=block_reason)

        strategy = self._get_symbol_strategy(symbol)
        active_name = strategy.get_name()

        # 2. Data sufficiency
        if not strategy.is_data_sufficient(df):
            reason = (
                f"데이터 부족: {len(df)}개 캔들 "
                f"(최소 {strategy.get_min_candles()}개 필요)"
            )
            logger.info("신호 차단 [%s]: %s", symbol, reason)
            return SignalResult(signal_type="none", reason=reason)

        # 3. Strategy execution
        try:
            result = strategy.generate_signal(df, symbol)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "전략 실행 오류 [%s] %s: %s",
                symbol,
                active_name,
                exc,
                exc_info=True,
            )
            return SignalResult(signal_type="none", reason=f"전략 실행 오류: {exc}")

        # Directional concentration guard: needs direction, so runs post-signal
        if result.is_actionable():
            dir_blocked, dir_reason = self._check_directional_concentration(result.signal_type)
            if dir_blocked:
                logger.info("신호 차단 [%s]: %s", symbol, dir_reason)
                return SignalResult(signal_type="none", reason=dir_reason)

        # Regime-adaptive TP scaling (G8): scale tp1/tp2 with BTC ADX
        if result.is_actionable() and result.entry_price is not None:
            result = self._apply_regime_tp_scale(result)

        if result.is_actionable():
            logger.info(
                "신호 생성 [%s][%s]: %s 강도%d — %s",
                symbol,
                active_name,
                result.signal_type,
                result.strength_score,
                result.reason,
            )
        return result

    def reload(self, new_strategy_name: str, new_params: dict) -> None:
        """Hot-swap the active strategy without restarting the pod.

        Args:
            new_strategy_name: snake_case name of the new strategy.
            new_params: Parameter dict for the new strategy instance.
        """
        self._strategy = self._load_strategy(new_strategy_name, new_params)
        self._strategy_name = new_strategy_name
        logger.info("전략 교체 완료: %s → %s", self._strategy_name, new_strategy_name)

    def get_active_strategy_name(self) -> str:
        """Return the snake_case name of the global fallback strategy."""
        return self._strategy_name

    def get_symbol_strategy_name(self, symbol: str) -> str:
        """Return the strategy name that will be used for *symbol*.

        Reads ``symbols.strategy`` from DB (same logic as ``_get_symbol_strategy``).
        Falls back to the global ``ACTIVE_STRATEGY`` when the column is NULL or
        the DB is unavailable.
        """
        if self._conn is None:
            return self._strategy_name
        try:
            row = self._conn.execute(
                "SELECT strategy FROM symbols WHERE symbol=? AND is_active=1 LIMIT 1",
                (symbol,),
            ).fetchone()
            override = row[0] if row and row[0] else None
            return override if override else self._strategy_name
        except Exception:  # noqa: BLE001
            return self._strategy_name

    def get_timeframe(self) -> str:
        """Return the timeframe of the currently active strategy."""
        getter = getattr(self._strategy, "get_timeframe", None)
        if callable(getter):
            return getter()
        return "1m"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_strategy(self, name: str, params: dict) -> BaseStrategy:
        """Dynamically import and instantiate a strategy class.

        Args:
            name: snake_case strategy name (e.g. ``'rsi_macd'``).
            params: Parameter dict passed to the strategy constructor.

        Returns:
            An instantiated :class:`BaseStrategy` subclass.

        Raises:
            ValueError: If the strategy module does not exist.
            TypeError: If the class does not inherit from BaseStrategy.
        """
        module_path = f"src.signal.strategies.{name}"
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise ValueError(
                f"전략 파일을 찾을 수 없습니다: '{module_path}'\n"
                f"'src/signal/strategies/{name}.py' 파일이 존재하는지 확인하세요."
            ) from exc

        class_name = self._snake_to_class_name(name)
        cls = getattr(module, class_name, None)
        if cls is None:
            raise ValueError(
                f"'{module_path}' 모듈에서 '{class_name}' 클래스를 찾을 수 없습니다."
            )
        if not (isinstance(cls, type) and issubclass(cls, BaseStrategy)):
            raise TypeError(
                f"'{class_name}'은 BaseStrategy를 상속하지 않습니다."
            )

        instance: BaseStrategy = cls(params)
        logger.info("전략 로드 완료: %s (파라미터 %d개)", name, len(params))
        return instance

    @staticmethod
    def _snake_to_class_name(snake: str) -> str:
        """Convert snake_case strategy name to PascalCase class name.

        Appends the ``Strategy`` suffix automatically.

        Args:
            snake: e.g. ``'rsi_macd'``

        Returns:
            e.g. ``'RsiMacdStrategy'``

        Example:
            >>> StrategyRunner._snake_to_class_name("zscore_reversion")
            'ZscoreReversionStrategy'
        """
        return "".join(part.capitalize() for part in snake.split("_")) + "Strategy"

    @staticmethod
    def _load_params(strategy_name: str = "") -> dict:
        """Parse strategy params from environment variables.

        Lookup order:
        1. ``STRATEGY_PARAMS_<NAME>`` (e.g. ``STRATEGY_PARAMS_RSI_MACD``)
        2. ``STRATEGY_PARAMS`` (global fallback)

        Returns:
            Parsed dict, or ``{}`` if absent or invalid JSON.
        """
        per_strategy_key = f"STRATEGY_PARAMS_{strategy_name.upper()}" if strategy_name else ""
        raw = (
            os.environ.get(per_strategy_key, "").strip()
            if per_strategy_key
            else ""
        ) or os.environ.get("STRATEGY_PARAMS", "").strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("STRATEGY_PARAMS JSON 파싱 실패, 빈 dict로 폴백: %s", exc)
            return {}

    def _apply_regime_tp_scale(self, result: SignalResult) -> SignalResult:
        """Scale tp1/tp2 distances from entry based on current BTC ADX regime."""
        scale = _tp_scale_from_adx(self._last_btc_adx)
        if scale == 1.0:
            return result
        entry = result.entry_price
        is_long = result.signal_type == "long"
        if result.tp1 is not None:
            dist = result.tp1 - entry if is_long else entry - result.tp1
            result.tp1 = (entry + dist * scale) if is_long else (entry - dist * scale)
        if result.tp2 is not None:
            dist = result.tp2 - entry if is_long else entry - result.tp2
            result.tp2 = (entry + dist * scale) if is_long else (entry - dist * scale)
        logger.info(
            "Regime-TP [%s]: BTC ADX=%.1f → ×%.2f  tp1=%.4f  tp2=%.4f",
            result.signal_type,
            self._last_btc_adx if self._last_btc_adx is not None else 0.0,
            scale,
            result.tp1 or 0.0,
            result.tp2 or 0.0,
        )
        return result

    def _check_directional_concentration(self, direction: str) -> tuple[bool, str]:
        """Block if too many open positions share the same direction.

        Threshold = dynamic_limit - 1, ensuring at least one slot remains
        open for the opposite direction.  Reuses G2's ADX-adjusted limit.
        """
        if self._conn is None:
            return False, ""
        try:
            from src.signal.signal_blocker import _dynamic_max_positions  # noqa: PLC0415
            from src.utils.config import load_config  # noqa: PLC0415

            config = load_config()
            limit = _dynamic_max_positions(self._last_btc_adx, config.max_positions)
            max_same = max(1, limit - 1)
            row = self._conn.execute(
                "SELECT COUNT(*) FROM positions WHERE status='open' AND side=?",
                (direction,),
            ).fetchone()
            same_count = int(row[0] if row is not None else 0)
            if same_count >= max_same:
                adx_str = (
                    f" (BTC ADX={self._last_btc_adx:.1f})"
                    if self._last_btc_adx is not None
                    else ""
                )
                return (
                    True,
                    f"directional_concentration: {same_count} {direction} >= limit {max_same}{adx_str}",
                )
            return False, ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("Directional concentration check failed (ignored): %s", exc)
            return False, ""

    def _check_signal_blocker(self, symbol: str) -> tuple[bool, str]:
        """Delegate to SignalBlocker if available."""
        if self._conn is None:
            return False, ""
        try:
            from src.signal.signal_blocker import SignalBlocker  # noqa: PLC0415
            blocker = SignalBlocker(self._conn, gap_detector=None, btc_adx=self._last_btc_adx)
            return blocker.should_block(symbol)
        except (ImportError, AttributeError):
            return False, ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("SignalBlocker 오류 (무시): %s", exc)
            return False, ""
