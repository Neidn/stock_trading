"""Structured logging with dual output: stderr + SQLite system_events.

Usage::

    from src.monitoring.logger import get_logger
    logger = get_logger("data_ingest")
    logger.info("WebSocket 연결 시작")
    logger.error("재연결 실패", extra={"symbol": "BTCUSDT", "attempt": 3})

All loggers share one StreamHandler (stderr) and one SQLiteHandler.
The SQLiteHandler writes to ``system_events`` only for INFO and above;
it silently skips if the DB is not yet initialised.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from typing import Any

# ---------------------------------------------------------------------------
# Log format
# ---------------------------------------------------------------------------

_FMT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# SQLite handler
# ---------------------------------------------------------------------------


class SQLiteHandler(logging.Handler):
    """Append log records to the ``system_events`` table.

    The handler lazily resolves ``db_path`` from the ``SQLITE_DB_PATH``
    environment variable at emit-time so it works before ``cfg`` is available.
    Failures are silently swallowed (handler calls ``handleError`` which
    prints to stderr without raising).
    """

    # Mapping from Python log level names → system_events severity values
    _SEVERITY: dict[str, str] = {
        "DEBUG":    "info",
        "INFO":     "info",
        "WARNING":  "warning",
        "ERROR":    "error",
        "CRITICAL": "critical",
    }

    def emit(self, record: logging.LogRecord) -> None:
        try:
            db_path = os.getenv("SQLITE_DB_PATH", "")
            if not db_path:
                return

            # Gather extra fields that callers may pass via `extra=`
            # Built-in LogRecord attributes to exclude from metadata
            _BUILTIN = {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "asctime", "taskName",
            }
            metadata: dict[str, Any] = {}
            for k, v in record.__dict__.items():
                if k.startswith("_") or k in _BUILTIN:
                    continue
                try:
                    json.dumps(v)   # probe serializability
                    metadata[k] = v
                except (TypeError, ValueError):
                    metadata[k] = str(v)

            severity = self._SEVERITY.get(record.levelname, "info")
            msg = self.format(record)

            # Reuse the managed singleton (WAL + PRAGMA already applied).
            # Import here to avoid circular import at module level.
            from src.db.connection import get_connection  # noqa: PLC0415
            conn = get_connection(db_path)

            # Silently skip if schema isn't initialised yet (early startup).
            table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='system_events'"
            ).fetchone()
            if not table_exists:
                return

            conn.execute(
                """
                INSERT INTO system_events
                    (event_id, event_type, severity, module, message, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    "log",
                    severity,
                    record.name,
                    msg,
                    json.dumps(metadata) if metadata else None,
                ),
            )
            conn.commit()
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Root logger bootstrap (runs once)
# ---------------------------------------------------------------------------

def _bootstrap_root() -> None:
    """Configure the root 'trading' logger exactly once."""
    root = logging.getLogger("trading")
    if root.handlers:
        return  # already set up

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    root.setLevel(level)

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    # stderr stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # SQLite handler (only INFO and above — no DEBUG noise in the DB)
    sqlite_handler = SQLiteHandler(level=logging.INFO)
    sqlite_handler.setFormatter(formatter)
    root.addHandler(sqlite_handler)

    root.propagate = False


_bootstrap_root()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_logger(module_name: str) -> logging.Logger:
    """Return a child logger under the 'trading' namespace.

    Args:
        module_name: Short module identifier, e.g. ``'data_ingest'``.

    Returns:
        A :class:`logging.Logger` that writes to stderr and SQLite.
    """
    return logging.getLogger(f"trading.{module_name}")
