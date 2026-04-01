"""
EcodiaOS - Send Discord Executor (Axon, action_type="send_discord")

Sends a Discord message to a specified channel_id via DiscordConnector.
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
  ECODIAOS_DISCORD_CHANNEL_ID - default channel_id fallback
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

# Hard rate limit: 30 Discord messages per hour to avoid API flooding.
_RATE_LIMIT = RateLimit.per_hour(30)

# Max text length matches Discord's single-message limit (2000 UTF-8 chars).
_MAX_TEXT_LENGTH = 2000


class SendDiscordExecutor(Executor):
    """
    Send a Discord message to a channel via the organism's bot token.

    Required params:
      message (str): The text to send (max 2000 chars).

    Optional params:
      channel_id (str | int): Target Discord channel ID.
                              Falls back to ECODIAOS_DISCORD_CHANNEL_ID
                              when omitted.

    Emits:
      RE_TRAINING_EXAMPLE on each send.

    Level 2 (COLLABORATOR) - sending messages is not reversible.
    Rate limit: 30 / hour.
    """

    action_type = "send_discord"
    description = "Send a message to a Discord channel via the organism's bot"
    required_autonomy = 2
    reversible = False
    max_duration_ms = 15_000
    rate_limit = _RATE_LIMIT

    def __init__(
        self,
        discord_connector: Any = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._connector = discord_connector
        self._event_bus = event_bus
        self._logger = logger.bind(system="axon.executor.send_discord")

    def set_discord_connector(self, connector: Any) -> None:
        """Hot-wire a DiscordConnector after construction (e.g. from app.state)."""
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
                f"'message' exceeds Discord's {_MAX_TEXT_LENGTH}-char limit"
            )

        # channel_id is optional - validated at execution time when connector resolves default
        channel_id = params.get("channel_id")
        if channel_id is not None and not isinstance(channel_id, (int, str)):
            return ValidationResult.fail("'channel_id' must be an int or string if provided")

        return ValidationResult.ok()

    # ── Execution ─────────────────────────────────────────────────────────

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        import os

        message: str = params["message"]

        # Resolve channel_id - param → env fallback → error
        channel_id = params.get("channel_id")
        if channel_id is None:
            env_channel_id = os.environ.get("ECODIAOS_DISCORD_CHANNEL_ID", "")
            if env_channel_id:
                channel_id = env_channel_id
        if channel_id is None:
            self._logger.warning("send_discord_no_channel_id")
            return ExecutionResult(
                success=False,
                error="No channel_id provided and ECODIAOS_DISCORD_CHANNEL_ID not set",
            )

        if self._connector is None:
            self._logger.warning("send_discord_no_connector")
            await self._emit_re_training(context, success=False, error="no_connector")
            return ExecutionResult(
                success=False,
                error="DiscordConnector not configured - set ECODIAOS_CONNECTORS__DISCORD__BOT_TOKEN",
            )

        t0 = time.monotonic()
        try:
            result = await self._connector.send_message(
                channel_id=channel_id,
                text=message,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            self._logger.info(
                "send_discord_success",
                channel_id=channel_id,
                message_id=result.get("id"),
                elapsed_ms=round(elapsed_ms, 1),
            )
            await self._emit_re_training(context, success=True)
            return ExecutionResult(
                success=True,
                data={
                    "message_id": result.get("id"),
                    "channel_id": channel_id,
                    "elapsed_ms": round(elapsed_ms, 1),
                },
                side_effects=["discord_message_sent"],
            )

        except Exception as exc:
            elapsed_ms = (time.monotonic() - t0) * 1000
            self._logger.error(
                "send_discord_failed",
                channel_id=channel_id,
                error=str(exc),
                elapsed_ms=round(elapsed_ms, 1),
            )
            await self._emit_re_training(context, success=False, error=str(exc))
            return ExecutionResult(
                success=False,
                error=f"Discord send failed: {exc}",
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
                    "success": success,
                    "error_type": error if error else "",
                    "execution_id": context.execution_id,
                    "outcome_quality": 1.0 if success else 0.0,
                    "constitutional_alignment": 0.95,  # Messages are honesty-aligned
                },
            )
            await self._event_bus.publish(event)
        except Exception as e:
            self._logger.warning("failed_to_emit_re_training", error=str(e))
