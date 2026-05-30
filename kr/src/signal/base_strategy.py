"""Base strategy abstractions: SignalResult dataclass and BaseStrategy ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SignalResult:
    """Represents the output of a strategy's signal generation.

    Attributes:
        signal_type: Direction of the signal — 'long', 'short', or 'none'.
        strength_score: Number of conditions met, capped at 3. Signals with
            score < 2 are treated as non-actionable.
        entry_price: Suggested entry price; None when signal_type is 'none'.
        tp1: First take-profit price (typically ATR × 3.0).
        tp2: Second take-profit price (typically ATR × 5.0).
        sl: Stop-loss price (typically ATR × 2.0).
        indicators: Snapshot of indicator values at signal generation time,
            stored verbatim for logging and DB persistence.
        reason: Human-readable explanation of why this signal was generated.
    """

    signal_type: str = "none"
    strength_score: int = 0
    entry_price: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    sl: float | None = None
    indicators: dict = field(default_factory=dict)
    reason: str = ""

    def is_actionable(self) -> bool:
        """Return True if the signal should trigger an order.

        A signal is actionable when it has a direction and enough confluence
        (strength_score >= 2).
        """
        return self.signal_type != "none" and self.strength_score >= 2

    def to_dict(self) -> dict:
        """Serialize to a flat dict suitable for DB insertion."""
        return {
            "signal_type": self.signal_type,
            "strength_score": self.strength_score,
            "entry_price": self.entry_price,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "sl": self.sl,
            "indicators": self.indicators,
            "reason": self.reason,
        }


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    All concrete strategies must inherit from this class and implement the
    three abstract methods. Strategy selection is driven by the
    ``ACTIVE_STRATEGY`` environment variable; never hardcode a strategy name.

    Example:
        class MyStrategy(BaseStrategy):
            def get_name(self) -> str:
                return "my_strategy"

            def get_min_candles(self) -> int:
                return 100

            def generate_signal(self, df, symbol: str) -> SignalResult:
                ...
    """

    def __init__(self, params: dict) -> None:
        """Store strategy parameters and run validation.

        Args:
            params: Key-value configuration loaded from ``STRATEGY_PARAMS``
                env var. Concrete strategies read their defaults from here.
        """
        self.params = params
        self._validate_params()

    @abstractmethod
    def generate_signal(self, df, symbol: str) -> SignalResult:
        """Compute a trading signal from OHLCV data.

        Args:
            df: pandas DataFrame with columns [open, high, low, close, volume],
                sorted ascending by time.
            symbol: Trading pair identifier, e.g. ``'BTCUSDT'``.

        Returns:
            A :class:`SignalResult` describing the signal (or lack thereof).
        """

    @abstractmethod
    def get_name(self) -> str:
        """Return the snake_case strategy identifier.

        Must match the module filename under ``src/signal/strategies/``.
        """

    @abstractmethod
    def get_min_candles(self) -> int:
        """Return the minimum number of candles required to produce a signal."""

    @abstractmethod
    def get_timeframe(self) -> str:
        """Return the candle timeframe this strategy operates on. E.g. '1m', '5m', '1h'."""

    def _validate_params(self) -> None:
        """Validate strategy-specific parameters.

        Base implementation is a no-op. Override in subclasses to raise
        ``ValueError`` on invalid parameter combinations.
        """

    def is_data_sufficient(self, df) -> bool:
        """Return True if *df* contains enough candles to generate a signal.

        Args:
            df: pandas DataFrame of OHLCV candles.
        """
        return len(df) >= self.get_min_candles()

    @classmethod
    def primary_regimes(cls) -> frozenset[str]:
        """Declare which market regimes this strategy is designed for.

        The screener uses this as a hard eligibility filter before comparing
        ``suitability_score``.  A strategy is only considered for a symbol
        when the detected regime is in its declared set (or the set contains
        ``'any'``).

        Valid regime strings: ``'ranging'``, ``'trending'``, ``'volatile'``, ``'any'``.

        Override in concrete strategies.  Default ``{'any'}`` means eligible
        in all regimes (safe fallback for strategies that don't override).
        """
        return frozenset({"any"})

    @classmethod
    def suitability_score(cls, indicators: dict) -> float:
        """Return 0.0–1.0 score for how well-suited this strategy is to current market.

        Override in each concrete strategy.  The screener calls this for all
        discovered strategies and picks the highest scorer for each symbol.

        Args:
            indicators: Dict with at minimum:
                ``adx``        — ADX value (0–100, higher = stronger trend)
                ``atr_pct``    — ATR as % of price (higher = more volatile)
                ``above_sma200`` — bool, close > SMA200

        Returns:
            Float in [0.0, 1.0].  Default 0.5 (neutral — used as fallback
            when a strategy does not override this method).
        """
        return 0.5
