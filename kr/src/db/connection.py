"""SQLite connection manager with WAL mode and singleton reuse.

One connection per db_path is created and reused across the process lifetime.
All connections are configured with WAL journal mode for concurrent reads.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

_lock = threading.Lock()
_connections: dict[str, sqlite3.Connection] = {}


def get_connection(db_path: str) -> sqlite3.Connection:
    """Return a singleton SQLite connection for *db_path*.

    Creates the connection (and its parent directories) on first call.
    Subsequent calls with the same path return the cached connection.

    PRAGMA settings applied once at connection creation:
        - journal_mode       = WAL   — concurrent reads during writes
        - synchronous        = NORMAL — balanced durability / speed
        - cache_size         = -64000 — 64 MB page cache
        - foreign_keys       = ON    — enforce FK constraints
        - busy_timeout       = 30000 — wait for concurrent pod writers
        - wal_autocheckpoint = 1000  — flush WAL to main DB every ~4 MB

    Args:
        db_path: Filesystem path to the SQLite database file.

    Returns:
        A configured :class:`sqlite3.Connection` with ``row_factory`` set to
        :class:`sqlite3.Row`.
    """
    with _lock:
        if db_path not in _connections:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.executescript("""
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                PRAGMA cache_size=-64000;
                PRAGMA foreign_keys=ON;
                PRAGMA busy_timeout=30000;
                PRAGMA wal_autocheckpoint=1000;
            """)
            _connections[db_path] = conn
        return _connections[db_path]
