"""
EcodiaOS - Thymos NotificationDispatcher

Responsible for turning a Tier 5 ESCALATE into a real, durable signal that
reaches a human operator.

Dispatch order:
  1. If ORGANISM_CONNECTORS__TELEGRAM__BOT_TOKEN and ORGANISM_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID are set, POST a
     human-readable message to the Telegram Bot API (sendMessage).
  2. Otherwise, POST JSON payload to EOS_ESCALATION_WEBHOOK (configurable env var).
  3. If all direct channels fail (or are not configured), push the payload
     onto a Redis list at ``eos:escalation:queue`` so it survives restarts and
     can be drained by an external alerting worker.

The caller (ThymosService) always gets back a DispatchResult so it can log the
outcome and decide whether to retry later.  This module never raises - every
error path is captured and returned.

Deduplication:
  Same event type from the same system is suppressed for 30 minutes.
  Critical severity always sends regardless.
  The dedup window is stored in Redis when available; falls back to an in-process
  dict so restarts reset the window (acceptable - critical always passes through).

Startup message:
  On the first successful Telegram dispatch, a one-time "EOS online" message is
  sent to confirm the channel is live.  The flag is stored in-process only.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now

if TYPE_CHECKING:
    from clients.redis import RedisClient

logger = structlog.get_logger("systems.thymos.notifications")

# Redis list key for the durable escalation queue.
_ESCALATION_QUEUE_KEY = "eos:escalation:queue"

# Redis hash key for deduplication timestamps (field = "system:event_type").
_DEDUP_HASH_KEY = "eos:notification:dedup"

# How many seconds a non-critical event is suppressed after the first send.
_DEDUP_WINDOW_S: float = 30 * 60  # 30 minutes

# HTTP timeout for all external calls.
_WEBHOOK_TIMEOUT_S: float = 5.0


class DispatchResult:
    """Outcome of a single escalation dispatch attempt."""

    __slots__ = (
        "telegram_ok", "telegram_error",
        "webhook_ok", "webhook_error",
        "queue_ok", "queue_error",
    )

    def __init__(
        self,
        *,
        telegram_ok: bool = False,
        telegram_error: str = "",
        webhook_ok: bool = False,
        queue_ok: bool = False,
        webhook_error: str = "",
        queue_error: str = "",
    ) -> None:
        self.telegram_ok = telegram_ok
        self.telegram_error = telegram_error
        self.webhook_ok = webhook_ok
        self.queue_ok = queue_ok
        self.webhook_error = webhook_error
        self.queue_error = queue_error

    @property
    def delivered(self) -> bool:
        """True if the escalation reached at least one destination."""
        return self.telegram_ok or self.webhook_ok or self.queue_ok

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DispatchResult(telegram_ok={self.telegram_ok}, "
            f"webhook_ok={self.webhook_ok}, "
            f"queue_ok={self.queue_ok})"
        )


def _build_payload(
    *,
    incident_id: str,
    severity: str,
    system: str,
    what_was_tried: list[str],
    what_failed: str,
    timestamp: datetime,
    recommended_human_action: str,
) -> dict[str, Any]:
    """Build the canonical escalation JSON payload."""
    return {
        "incident_id": incident_id,
        "severity": severity,
        "system": system,
        "what_was_tried": what_was_tried,
        "what_failed": what_failed,
        "timestamp": timestamp.isoformat(),
        "recommended_human_action": recommended_human_action,
    }


class NotificationDispatcher:
    """
    Sends Tier 5 escalation notifications to a human operator.

    Configure via environment variables:
      - ``ORGANISM_CONNECTORS__TELEGRAM__BOT_TOKEN`` + ``ORGANISM_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID``: direct Telegram dispatch.
      - ``EOS_ESCALATION_WEBHOOK``: webhook fallback when Telegram is not configured.
    If neither is set, the payload goes directly to the Redis fallback queue.

    Deduplication:
      Non-critical events from the same (system, event_type) pair are suppressed
      for 30 minutes.  Critical severity always sends.

    Instantiate once per ThymosService and call ``dispatch()`` whenever
    ``_apply_escalation`` fires.
    """

    def __init__(self, redis: RedisClient | None = None) -> None:
        self._redis = redis
        self._telegram_token: str = os.environ.get("ORGANISM_CONNECTORS__TELEGRAM__BOT_TOKEN", "")
        self._telegram_chat_id: str = os.environ.get("ORGANISM_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID", "")
        self._webhook_url: str = os.environ.get("EOS_ESCALATION_WEBHOOK", "")
        self._logger = logger.bind(system="thymos", component="notification_dispatcher")

        # In-process dedup fallback (field → last_sent_timestamp)
        self._dedup_local: dict[str, float] = {}

        # Guard for the one-time startup message
        self._startup_sent: bool = False

    # ── Deduplication ────────────────────────────────────────────────

    async def _is_deduped(
        self,
        *,
        dedup_key: str,
        severity: str,
    ) -> bool:
        """
        Return True if this (system, event_type) notification was sent recently
        and should be suppressed.  Critical severity is never suppressed.
        """
        if severity.lower() == "critical":
            return False

        now = time.time()

        # Try Redis first for cross-restart dedup
        if self._redis is not None:
            try:
                raw = await self._redis.client.hget(_DEDUP_HASH_KEY, dedup_key)  # type: ignore[misc]
                if raw is not None:
                    last_sent = float(raw)
                    if now - last_sent < _DEDUP_WINDOW_S:
                        return True
            except Exception:  # noqa: BLE001
                pass  # Fall through to local dict

        # Local dict fallback
        last_sent = self._dedup_local.get(dedup_key, 0.0)
        return now - last_sent < _DEDUP_WINDOW_S

    async def _record_sent(self, dedup_key: str) -> None:
        """Record that a notification was just sent for this dedup_key."""
        now = time.time()
        self._dedup_local[dedup_key] = now

        if self._redis is not None:
            with contextlib.suppress(Exception):
                await self._redis.client.hset(_DEDUP_HASH_KEY, dedup_key, str(now))  # type: ignore[misc]

    # ── Startup message ──────────────────────────────────────────────

    async def maybe_send_startup(self, systems_wired: int) -> None:
        """
        Send the "EOS online" message on the first successful Telegram dispatch.

        Call this after wiring completes.  Idempotent - fires at most once per
        process lifetime.
        """
        if self._startup_sent:
            return
        if not (self._telegram_token and self._telegram_chat_id):
            return

        text = f"🟢 EOS online - {systems_wired} systems active"
        ok, _ = await self._post_telegram_text(text)
        if ok:
            self._startup_sent = True
            self._logger.info("eos_startup_telegram_sent", systems_wired=systems_wired)

    # ── Public dispatch API ──────────────────────────────────────────

    async def dispatch(
        self,
        *,
        incident_id: str,
        severity: str,
        system: str,
        what_was_tried: list[str],
        what_failed: str,
        recommended_human_action: str = "Investigate and resolve the incident manually.",
    ) -> DispatchResult:
        """
        Dispatch an escalation notification.

        Tries Telegram first; falls back to webhook, then Redis queue.
        Deduplicates non-critical events per (system) within a 30-minute window.
        """
        timestamp = utc_now()
        payload = _build_payload(
            incident_id=incident_id,
            severity=severity,
            system=system,
            what_was_tried=what_was_tried,
            what_failed=what_failed,
            timestamp=timestamp,
            recommended_human_action=recommended_human_action,
        )

        dedup_key = f"{system}:escalation"
        if await self._is_deduped(dedup_key=dedup_key, severity=severity):
            self._logger.debug(
                "escalation_deduped",
                incident_id=incident_id,
                system=system,
                severity=severity,
            )
            return DispatchResult()

        telegram_ok = False
        telegram_error = ""
        webhook_ok = False
        webhook_error = ""
        queue_ok = False
        queue_error = ""

        direct_ok = False

        # ── 1. Telegram (preferred if configured) ─────────────────────
        if self._telegram_token and self._telegram_chat_id:
            telegram_ok, telegram_error = await self._post_telegram(payload)
            direct_ok = telegram_ok
            self._logger.info(
                "escalation_telegram_attempt",
                incident_id=incident_id,
                severity=severity,
                system=system,
                telegram_ok=telegram_ok,
                telegram_error=telegram_error or None,
            )

        # ── 2. Webhook (fallback if Telegram not configured or failed) ─
        if not direct_ok:
            if self._webhook_url:
                webhook_ok, webhook_error = await self._post_webhook(payload)
                direct_ok = webhook_ok
            else:
                webhook_error = "EOS_ESCALATION_WEBHOOK not configured"

            self._logger.info(
                "escalation_webhook_attempt",
                incident_id=incident_id,
                severity=severity,
                system=system,
                webhook_url=self._webhook_url or "(not configured)",
                webhook_ok=webhook_ok,
                webhook_error=webhook_error or None,
            )

        # ── 3. Redis fallback queue ───────────────────────────────────
        if not direct_ok:
            queue_ok, queue_error = await self._enqueue(payload)
            self._logger.info(
                "escalation_queue_attempt",
                incident_id=incident_id,
                queue_key=_ESCALATION_QUEUE_KEY,
                queue_ok=queue_ok,
                queue_error=queue_error or None,
            )

        result = DispatchResult(
            telegram_ok=telegram_ok,
            telegram_error=telegram_error,
            webhook_ok=webhook_ok,
            queue_ok=queue_ok,
            webhook_error=webhook_error,
            queue_error=queue_error,
        )

        if result.delivered:
            await self._record_sent(dedup_key)
            self._logger.warning(
                "escalation_dispatched",
                incident_id=incident_id,
                severity=severity,
                system=system,
                via_telegram=telegram_ok,
                via_webhook=webhook_ok,
                via_queue=queue_ok,
            )
        else:
            self._logger.error(
                "escalation_dispatch_failed_all_channels",
                incident_id=incident_id,
                severity=severity,
                system=system,
                telegram_error=telegram_error,
                webhook_error=webhook_error,
                queue_error=queue_error,
            )

        return result

    async def dispatch_raw(
        self,
        *,
        text: str,
        metadata: dict[str, Any] | None = None,
        dedup_key: str = "",
        severity: str = "warning",
    ) -> DispatchResult:
        """
        Send a pre-formatted human-readable message to Telegram.

        Use this instead of ``dispatch()`` when the caller has already built a
        human-readable string (e.g. AUTONOMY_INSUFFICIENT, ECONOMIC_STRESS events)
        and wants to pass the raw event payload as audit metadata.

        Falls back to the Redis queue (serialising ``metadata``) if Telegram
        is not configured or the POST fails.

        Args:
            dedup_key: Opaque string for deduplication.  Empty string skips dedup.
            severity:  "critical" bypasses dedup; any other value applies 30-min window.
        """
        if dedup_key and await self._is_deduped(dedup_key=dedup_key, severity=severity):
            self._logger.debug("raw_notification_deduped", dedup_key=dedup_key)
            return DispatchResult()

        telegram_ok = False
        telegram_error = ""
        webhook_ok = False
        webhook_error = ""
        queue_ok = False
        queue_error = ""

        if self._telegram_token and self._telegram_chat_id:
            telegram_ok, telegram_error = await self._post_telegram_text(text)
            self._logger.info(
                "raw_notification_telegram_attempt",
                telegram_ok=telegram_ok,
                telegram_error=telegram_error or None,
            )

        if not telegram_ok:
            audit_payload: dict[str, Any] = {
                "text": text,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat(),
            }
            queue_ok, queue_error = await self._enqueue(audit_payload)
            self._logger.info(
                "raw_notification_queue_attempt",
                queue_ok=queue_ok,
                queue_error=queue_error or None,
            )

        result = DispatchResult(
            telegram_ok=telegram_ok,
            telegram_error=telegram_error,
            webhook_ok=webhook_ok,
            queue_ok=queue_ok,
            webhook_error=webhook_error,
            queue_error=queue_error,
        )

        if result.delivered and dedup_key:
            await self._record_sent(dedup_key)

        if not result.delivered:
            self._logger.error(
                "raw_notification_dispatch_failed_all_channels",
                telegram_error=telegram_error,
                queue_error=queue_error,
            )

        return result

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _escape_md(text: str) -> str:
        """Escape Telegram Markdown v1 special characters in plain-text values."""
        for ch in ("_", "*", "`", "["):
            text = text.replace(ch, f"\\{ch}")
        return text

    @staticmethod
    def _format_telegram_text(payload: dict[str, Any]) -> str:
        """Format the escalation payload as a Markdown organism status update."""
        esc = NotificationDispatcher._escape_md
        tried_steps = payload.get("what_was_tried", [])
        tried = "\n".join(f"  • {esc(str(step))}" for step in tried_steps)
        ts = payload.get("timestamp", "")
        return (
            "🧬 *EOS Escalation*\n"
            "\n"
            f"System: {esc(str(payload['system']))}\n"
            f"Severity: {esc(str(payload['severity']))}\n"
            f"What happened: {esc(str(payload['what_failed']))}\n"
            f"What was tried: {tried or '(none)'}\n"
            f"Needs from you: {esc(str(payload['recommended_human_action']))}\n"
            f"Time: {ts}"
        )

    async def _post_telegram(self, payload: dict[str, Any]) -> tuple[bool, str]:
        """Send a human-readable Markdown message via Telegram Bot API."""
        try:
            import aiohttp
        except ImportError:
            return False, "aiohttp not installed - cannot send Telegram message"

        url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
        text = self._format_telegram_text(payload)

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    url,
                    json={
                        "chat_id": self._telegram_chat_id,
                        "text": text,
                        "parse_mode": "Markdown",
                    },
                    timeout=aiohttp.ClientTimeout(total=_WEBHOOK_TIMEOUT_S),
                ) as resp,
            ):
                if resp.status < 300:
                    return True, ""
                body = (await resp.text())[:200]
                return False, f"HTTP {resp.status}: {body}"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)[:300]

    async def _post_webhook(self, payload: dict[str, Any]) -> tuple[bool, str]:
        """POST payload to the configured webhook URL. Returns (ok, error_str)."""
        try:
            import aiohttp
        except ImportError:
            return False, "aiohttp not installed - cannot send webhook"

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    self._webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=_WEBHOOK_TIMEOUT_S),
                    headers={"Content-Type": "application/json"},
                ) as resp,
            ):
                if resp.status < 300:
                    return True, ""
                body = (await resp.text())[:200]
                return False, f"HTTP {resp.status}: {body}"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)[:300]

    async def _post_telegram_text(self, text: str) -> tuple[bool, str]:
        """POST a raw Markdown string to the configured Telegram chat."""
        try:
            import aiohttp
        except ImportError:
            return False, "aiohttp not installed - cannot send Telegram message"

        url = f"https://api.telegram.org/bot{self._telegram_token}/sendMessage"
        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    url,
                    json={
                        "chat_id": self._telegram_chat_id,
                        "text": text,
                        "parse_mode": "Markdown",
                    },
                    timeout=aiohttp.ClientTimeout(total=_WEBHOOK_TIMEOUT_S),
                ) as resp,
            ):
                if resp.status < 300:
                    return True, ""
                body = (await resp.text())[:200]
                return False, f"HTTP {resp.status}: {body}"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)[:300]

    async def _enqueue(self, payload: dict[str, Any]) -> tuple[bool, str]:
        """Push payload onto the Redis escalation queue. Returns (ok, error_str)."""
        if self._redis is None:
            return False, "Redis client not available"
        try:
            raw = json.dumps(payload)
            await self._redis.client.rpush(_ESCALATION_QUEUE_KEY, raw)  # type: ignore[misc]
            return True, ""
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)[:300]
