"""API guard utilities: custom exceptions and retry decorator.

Usage::

    from src.utils.api_guard import with_retry, RateLimitError, NetworkError

    @with_retry(max_retries=3, delay=1.0)
    async def fetch_something():
        ...
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable

logger = logging.getLogger("trading.api_guard")


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class TradingAPIError(Exception):
    """Base class for all API-level errors."""


class RateLimitError(TradingAPIError):
    """Raised when the exchange returns HTTP 429 (Too Many Requests)."""


class NetworkError(TradingAPIError):
    """Raised on connection-level failures (timeout, DNS, SSL)."""


class InvalidAPIKeyError(TradingAPIError):
    """Raised when the exchange returns HTTP 401 / -2014 / -2015."""


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------


def with_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    fatal_on_failure: bool = False,
) -> Callable:
    """Async retry decorator with exponential back-off.

    Retries on :class:`NetworkError` and :class:`RateLimitError`.
    Never retries on :class:`InvalidAPIKeyError` (auth errors are permanent).

    Args:
        max_retries: Maximum number of attempts (first try + retries).
        delay: Base delay in seconds between attempts (doubles each retry).
        fatal_on_failure: If True, re-raise the last exception after all
            retries are exhausted. If False, logs the error and re-raises
            (callers are responsible for handling).

    Raises:
        The last exception after all retries are exhausted, regardless of
        ``fatal_on_failure`` (the flag is reserved for caller signalling).
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if max_retries < 1:
                raise ValueError(f"max_retries must be >= 1, got {max_retries}")
            last_exc: Exception = RuntimeError("unreachable")
            wait = delay
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except InvalidAPIKeyError:
                    # Auth errors are permanent — no retry
                    raise
                except (RateLimitError, NetworkError) as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        logger.warning(
                            "%s attempt %d/%d failed: %s — retrying in %.1fs",
                            func.__qualname__, attempt, max_retries, exc, wait,
                        )
                        await asyncio.sleep(wait)
                        wait *= 2
                    else:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__qualname__, max_retries, exc,
                        )
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator
