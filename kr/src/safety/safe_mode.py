"""Safe mode — emergency halt that blocks all new order signals.

Activated automatically by protective subsystems (DrawdownGuard, LiquidationGuard,
etc.) or manually via the Telegram bot. When active, SignalBlocker will reject
every new signal until safe mode is deactivated or the auto-release timer fires.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)


class SafeMode:
    """Manages the safe-mode flag with DB persistence and Telegram notifications.

    Args:
        conn: Optional SQLite connection. State changes are persisted when provided.
        telegram_bot: Optional bot instance. Must expose ``send_alert(message: str)``.
            Implemented in Phase 5; pass ``None`` until then.
    """

    def __init__(
        self,
        conn: sqlite3.Connection | None = None,
        telegram_bot: object | None = None,
    ) -> None:
        self._conn = conn
        self._telegram = telegram_bot
        self._active: bool = False
        self._reason: str = ""
        self._activated_at: datetime | None = None
        self._auto_release_hours: float = 0.0
        self._restore_from_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def activate(self, reason: str, auto_release_hours: float = 0) -> None:
        """Activate safe mode.

        Args:
            reason: Human-readable description of why safe mode is being activated.
            auto_release_hours: If > 0, automatically deactivate after this many hours.
        """
        self._active = True
        self._reason = reason
        self._activated_at = datetime.now(timezone.utc)
        self._auto_release_hours = float(auto_release_hours)
        self._record_event("activated", reason, "system")
        self._notify(f"[SAFE MODE] ACTIVATED: {reason}")
        logger.warning("SafeMode activated: %s", reason)

    def deactivate(self, by: str = "auto") -> None:
        """Deactivate safe mode.

        Args:
            by: Who/what triggered deactivation. E.g. 'auto', 'telegram', 'manual'.
        """
        if not self._active:
            return
        reason_snapshot = self._reason
        self._active = False
        self._record_event("deactivated", reason_snapshot or "deactivated", by)
        self._notify(f"[SAFE MODE] DEACTIVATED by {by}")
        self._reason = ""
        self._activated_at = None
        self._auto_release_hours = 0.0
        logger.info("SafeMode deactivated by: %s", by)

    def is_active(self) -> bool:
        """Return ``True`` if safe mode is currently active."""
        return self._active

    @property
    def reason(self) -> str:
        """Return the reason safe mode was activated, or empty string."""
        return self._reason

    def check_auto_release(self) -> None:
        """Deactivate automatically if the auto-release timer has expired."""
        if not self._active:
            return
        if self._auto_release_hours <= 0:
            return
        if self._activated_at is None:
            return
        now = datetime.now(timezone.utc)
        elapsed_hours = (now - self._activated_at).total_seconds() / 3600.0
        if elapsed_hours >= self._auto_release_hours:
            logger.info(
                "SafeMode auto-release: %.2f h elapsed (limit %.2f h)",
                elapsed_hours,
                self._auto_release_hours,
            )
            self.deactivate(by="auto")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _restore_from_db(self) -> None:
        """Restore in-memory state from last DB event on startup."""
        if self._conn is None:
            return
        try:
            row = self._conn.execute(
                "SELECT action, reason, created_at FROM safe_mode_events "
                "ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            if row and row[0] == "activated":
                self._active = True
                self._reason = row[1] or ""
                try:
                    self._activated_at = datetime.fromisoformat(row[2]).replace(
                        tzinfo=timezone.utc
                    )
                except Exception:  # noqa: BLE001
                    self._activated_at = datetime.now(timezone.utc)
                logger.warning("SafeMode restored as ACTIVE from DB: %s", self._reason)
        except Exception as exc:  # noqa: BLE001
            logger.error("SafeMode DB restore failed: %s", exc)

    def _record_event(self, action: str, reason: str, by: str) -> None:
        if self._conn is None:
            return
        try:
            self._conn.execute(
                "INSERT INTO safe_mode_events (event_id, action, reason, by) VALUES (?, ?, ?, ?)",
                (str(uuid.uuid4()), action, reason, by),
            )
            self._conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.error("SafeMode DB write failed: %s", exc)

    def _notify(self, message: str) -> None:
        if self._telegram is None:
            return
        try:
            self._telegram.send_alert(message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SafeMode Telegram notify failed: %s", exc)
