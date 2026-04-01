"""
EcodiaOS - Send Telegram Executor (Axon, action_type="send_telegram")

Sends a Telegram message to a specified chat_id via TelegramConnector.
Requires COLLABORATOR autonomy (level 2) - sending unsolicited messages to
an external channel is not reversible and has real-world effect.

Constitutional gating:
  - Equor gates every execution via the normal Axon pipeline (Stage 0).
  - Honesty drive: Voxis personality rendering recommended but not enforced
    here - this executor emits raw text; callers should pass rendered content.

RE training:
  - Emits RE_TRAINING_EXAMPLE on every send with outcome_quality and
    constitutional_alignment scores.

Env vars:
  ECODIAOS_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID - default chat_id fallback
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.executor import Executor
from systems.axon.types import ExecutionContext, ExecutionResult, RateLimit, ValidationResult
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger()

# Hard rate limit: 30 Telegram messages per hour to avoid API flooding.
_RATE_LIMIT = RateLimit.per_hour(30)

# Max text length matches Telegram's single-message limit (4096 UTF-8 chars).
_MAX_TEXT_LENGTH = 4096

_VALID_PARSE_MODES = frozenset({"Markdown", "MarkdownV2", "HTML", ""})


class SendTelegramExecutor(Executor):
    """
    Send a Telegram message to a chat via the organism's bot token.

    Required params:
      message (str): The text to send (max 4096 chars).

    Optional params:
      chat_id (int | str): Target chat or channel ID.
                           Falls back to ECODIAOS_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID
                           when omitted.
      parse_mode (str): "Markdown" | "MarkdownV2" | "HTML" | "".
                        Default: "Markdown".

    Emits:
      RE_TRAINING_EXAMPLE on each send.

    Level 2 (COLLABORATOR) - sending messages is not reversible.
    Rate limit: 30 / hour.
    """

    action_type = "send_telegram"
    description = "Send a message to a Telegram chat via the organism's bot"
    required_autonomy = 2
    reversible = False
    max_duration_ms = 15_000
    rate_limit = _RATE_LIMIT

    def __init__(
        self,
        telegram_connector: Any = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._connector = telegram_connector
        self._event_bus = event_bus
        self._logger = logger.bind(system="axon.executor.send_telegram")

    def set_telegram_connector(self, connector: Any) -> None:
        """Hot-wire a TelegramConnector after construction (e.g. from app.state)."""
        self._connector = connector

    def set_event_bus(self, event_bus: Any) -> None:
        self._event_bus = event_bus

    # ── Validation ────────────────────────────────────────────────────────

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        message = params.get("message", "")
        if not message or not isinstance(message, str):
            return ValidationResult.fail("'message' is required and must be a non-empty string")
        if len(message) > _MAX_TEXT_LENGTH:
            return ValidationResult.fail(
                f"'message' exceeds Telegram's {_MAX_TEXT_LENGTH}-char limit"
            )

        parse_mode = params.get("parse_mode", "Markdown")
        if parse_mode not in _VALID_PARSE_MODES:
            return ValidationResult.fail(
                f"'parse_mode' must be one of {sorted(_VALID_PARSE_MODES)!r}"
            )

        # chat_id is optional - validated at execution time when connector resolves default
        chat_id = params.get("chat_id")
        if chat_id is not None and not isinstance(chat_id, (int, str)):
            return ValidationResult.fail("'chat_id' must be an int or string if provided")

        return ValidationResult.ok()

    # ── Execution ─────────────────────────────────────────────────────────

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        import os

        message: str = params["message"]
        parse_mode: str = params.get("parse_mode", "Markdown")

        # Resolve chat_id - param → env fallback → error
        chat_id = params.get("chat_id")
        if chat_id is None:
            env_chat_id = os.environ.get("ECODIAOS_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID", "")
            if env_chat_id:
                chat_id = int(env_chat_id)
        if chat_id is None:
            self._logger.warning("send_telegram_no_chat_id")
            return ExecutionResult(
                success=False,
                error="No chat_id provided and ECODIAOS_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID not set",
            )

        if self._connector is None:
            self._logger.warning("send_telegram_no_connector")
            await self._emit_re_training(context, success=False, error="no_connector")
            return ExecutionResult(
                success=False,
                error="TelegramConnector not configured - set ECODIAOS_CONNECTORS__TELEGRAM__BOT_TOKEN",
            )

        t0 = time.monotonic()
        try:
            result = await self._connector.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=parse_mode,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            self._logger.info(
                "send_telegram_success",
                chat_id=chat_id,
                message_id=result.get("message_id"),
                elapsed_ms=round(elapsed_ms, 1),
            )
            await self._emit_re_training(context, success=True)
            return ExecutionResult(
                success=True,
                data={
                    "message_id": result.get("message_id"),
                    "chat_id": chat_id,
                    "elapsed_ms": round(elapsed_ms, 1),
                },
                side_effects=["telegram_message_sent"],
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            self._logger.error(
                "send_telegram_failed",
                chat_id=chat_id,
                error=str(exc),
                elapsed_ms=round(elapsed_ms, 1),
            )
            await self._emit_re_training(context, success=False, error=str(exc))
            return ExecutionResult(
                success=False,
                error=f"Telegram send failed: {exc}",
            )

    # ── RE training ───────────────────────────────────────────────────────

    async def _emit_re_training(
        self,
        context: ExecutionContext,
        success: bool,
        error: str = "",
    ) -> None:
        if self._event_bus is None:
            return
        try:
            event = SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="axon",
                data={
                    "action_type": self.action_type,
                    "execution_id": context.execution_id,
                    "intent_id": getattr(context, "intent_id", ""),
                    "outcome_quality": 1.0 if success else 0.0,
                    "success": success,
                    "error": error,
                    "category": "send_telegram",
                    "constitutional_alignment": {
                        "honesty": 1.0,    # only sends what the organism intends
                        "care": 0.8,       # respects recipient's attention
                        "growth": 0.6,     # communication is growth-adjacent
                        "coherence": 0.9,  # message coherence validated upstream
                    },
                },
            )
            await self._event_bus.emit(event)
        except Exception:
            pass  # RE training is best-effort
