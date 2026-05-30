"""Flask health-check server for K8s liveness / readiness probes.

Runs in a daemon thread alongside the main asyncio event loop.

Endpoints:
    GET /health  — liveness probe (always 200 if process is alive)
    GET /ready   — readiness probe (200 = DB ok + WS active, 503 otherwise)
    GET /status  — dashboard detail (connections, candle age, DB size, etc.)

Usage::

    from src.monitoring.health import HealthServer, start_health_server

    # Option A: functional helper (background daemon thread)
    server = start_health_server("data-ingest")

    # Option B: manual lifecycle
    server = HealthServer("data-ingest")
    server.register_db(conn)
    server.register_ws_manager(ws_manager)
    server.start()          # non-blocking daemon thread
    ...
    server.stop()
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any

from flask import Flask, jsonify
from werkzeug.serving import make_server

from src.monitoring.logger import get_logger

logger = get_logger("health")

_DEFAULT_PORT = 8080


# ---------------------------------------------------------------------------
# HealthServer
# ---------------------------------------------------------------------------


class HealthServer:
    """Encapsulates the Flask app and its background thread.

    Args:
        module_name: Identifier returned in /health responses, e.g.
                     ``'data-ingest'``.
        port: TCP port to listen on. Defaults to 8080.
    """

    def __init__(self, module_name: str, port: int = _DEFAULT_PORT) -> None:
        self._module = module_name
        self._port = port
        self._start_time = time.time()

        # Optional references — set via register_* after construction
        self._db_conn: sqlite3.Connection | None = None
        self._ws_manager: Any | None = None  # BinanceWSManager

        self._app = self._build_app()
        self._server = make_server("0.0.0.0", self._port, self._app)
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def register_db(self, conn: sqlite3.Connection) -> None:
        """Attach a DB connection for readiness checks."""
        self._db_conn = conn

    def register_ws_manager(self, ws_manager: Any) -> None:
        """Attach a :class:`BinanceWSManager` for WS liveness checks."""
        self._ws_manager = ws_manager

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the Flask server in a background daemon thread."""
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"health-{self._module}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Health server started on port %d (module=%s)",
            self._port, self._module,
        )

    def stop(self) -> None:
        """Shut down the Flask server."""
        self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Flask app factory
    # ------------------------------------------------------------------

    def _build_app(self) -> Flask:
        app = Flask(__name__)
        # Suppress Werkzeug request logs (K8s probes hit /health every few sec)
        import logging as _logging
        _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

        server = self  # closure reference

        @app.get("/health")
        def health():
            return jsonify({
                "status": "ok",
                "module": server._module,
                "timestamp": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
            }), 200

        @app.get("/ready")
        def ready():
            checks: dict[str, Any] = {}
            all_ok = True

            # --- DB check ---
            try:
                if server._db_conn is None:
                    raise RuntimeError("no db registered")
                server._db_conn.execute("SELECT 1").fetchone()
                checks["db"] = "ok"
            except Exception as exc:
                checks["db"] = f"error: {exc}"
                all_ok = False

            # --- WebSocket check ---
            ws = server._ws_manager
            if ws is None:
                checks["websocket"] = "not registered"
                # Don't fail readiness just because WS manager isn't wired yet
            else:
                active = {
                    sym: t
                    for sym, t in ws._last_msg_time.items()
                    if time.monotonic() - t < 60  # active in last 60s
                }
                if active:
                    checks["websocket"] = "ok"
                    latest = max(active.values())
                    checks["last_candle_age_sec"] = round(
                        time.monotonic() - latest, 1
                    )
                else:
                    checks["websocket"] = "no active streams"
                    all_ok = False

            status = "ready" if all_ok else "not ready"
            code = 200 if all_ok else 503
            return jsonify({"status": status, "checks": checks}), code

        @app.get("/status")
        def status():
            uptime_sec = round(time.time() - server._start_time)

            # --- DB stats ---
            db_info: dict[str, Any] = {"connected": False}
            if server._db_conn is not None:
                try:
                    server._db_conn.execute("SELECT 1").fetchone()
                    db_info["connected"] = True

                    db_path = os.getenv("SQLITE_DB_PATH", "")
                    if db_path and os.path.exists(db_path):
                        db_info["size_mb"] = round(
                            os.path.getsize(db_path) / 1_048_576, 2
                        )

                    row = server._db_conn.execute(
                        "SELECT COUNT(*) AS n FROM klines"
                    ).fetchone()
                    db_info["total_klines"] = row["n"] if row else 0

                    row = server._db_conn.execute(
                        "SELECT COUNT(*) AS n FROM system_events"
                    ).fetchone()
                    db_info["total_events"] = row["n"] if row else 0
                except Exception as exc:
                    db_info["error"] = str(exc)

            # --- WebSocket stats ---
            ws_info: dict[str, Any] = {"manager": "not registered"}
            ws = server._ws_manager
            if ws is not None:
                now = time.monotonic()
                streams: dict[str, Any] = {}
                for sym, last_t in ws._last_msg_time.items():
                    age = round(now - last_t, 1)
                    streams[sym] = {
                        "last_msg_age_sec": age,
                        "connected": sym in ws._ws_connections
                        and not ws._ws_connections[sym].closed,
                    }
                ws_info = {
                    "active_streams": len(ws._tasks),
                    "streams": streams,
                }

            return jsonify({
                "module":     server._module,
                "uptime_sec": uptime_sec,
                "timestamp":  datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "db":         db_info,
                "websocket":  ws_info,
            }), 200

        return app


# ---------------------------------------------------------------------------
# Functional helper
# ---------------------------------------------------------------------------


def start_health_server(
    module_name: str,
    port: int = _DEFAULT_PORT,
    db_conn: sqlite3.Connection | None = None,
    ws_manager: Any | None = None,
) -> HealthServer:
    """Create, configure, and start a :class:`HealthServer` daemon thread.

    Args:
        module_name: Short identifier for this pod, e.g. ``'data-ingest'``.
        port: TCP port. Defaults to 8080.
        db_conn: Optional DB connection to register immediately.
        ws_manager: Optional :class:`BinanceWSManager` to register immediately.

    Returns:
        The running :class:`HealthServer` instance (call ``.stop()`` to shut
        down, though it's a daemon thread and will exit with the process).
    """
    server = HealthServer(module_name, port)
    if db_conn is not None:
        server.register_db(db_conn)
    if ws_manager is not None:
        server.register_ws_manager(ws_manager)
    server.start()
    return server
