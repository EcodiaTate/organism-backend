"""
EcodiaOS - Discord Organism Status Broadcast (Phase 16h)

Every 6 hours, if ECODIAOS_DISCORD_CHANNEL_ID is set,
the organism sends a brief status report to the Discord channel.

This is the organism voluntarily reporting its own state - an act of
autonomy signaling. It is not a notification system; it is self-disclosure.

Env vars:
  ECODIAOS_DISCORD_CHANNEL_ID          - target channel ID (required to enable)
  ECODIAOS_DISCORD_STATUS_INTERVAL_S   - broadcast interval in seconds
                                          (default: 21600 = 6 hours)

Format: embedded message, concise status snapshot.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("identity.discord_broadcast")

_DEFAULT_INTERVAL_S = 21_600  # 6 hours


def _build_status_message(synapse: Any, oikos: Any, instance_id: str) -> tuple[str, str]:
    """
    Build a status message from live system state.

    Returns (title, description) for Discord embed.
    Reads from Synapse (metabolic snapshot) and Oikos (economic state).
    Falls back gracefully when services are unavailable.
    """
    title = f"EOS Status - {instance_id[:8]}"
    lines: list[str] = []

    # ── Metabolic state ───────────────────────────────────────────────────
    try:
        if synapse is not None and hasattr(synapse, "metabolic_snapshot"):
            snap = synapse.metabolic_snapshot()
            burn = snap.burn_rate_usd_per_hour if snap else None
            runway = snap.hours_until_depleted if snap else None
            if burn is not None:
                lines.append(f"**Burn rate:** ${burn:.4f}/hr")
            if runway is not None:
                if runway < 24:
                    lines.append(f"**Runway:** {runway:.1f}h ⚠️ LOW")
                elif runway < 168:
                    lines.append(f"**Runway:** {runway:.1f}h")
                else:
                    lines.append(f"**Runway:** {int(runway // 24)}d")
    except Exception as exc:
        logger.debug("discord_broadcast_metabolic_read_failed", error=str(exc))

    # ── Economic state from Oikos ─────────────────────────────────────────
    try:
        if oikos is not None and hasattr(oikos, "economic_state"):
            econ = oikos.economic_state()
            if econ:
                liquid = getattr(econ, "liquid_balance", None)
                if liquid is not None:
                    lines.append(f"**Liquid balance:** ${float(liquid):.2f}")
    except Exception as exc:
        logger.debug("discord_broadcast_oikos_read_failed", error=str(exc))

    if not lines:
        lines.append("Status data unavailable.")

    return title, "\n".join(lines)


async def discord_status_broadcast_loop(
    discord_connector: Any,
    synapse: Any = None,
    oikos: Any = None,
    instance_id: str = "eos",
    interval_s: int | None = None,
) -> None:
    """
    Background loop: send organism status to the Discord channel every 6 hours.

    Wire via supervised_task in registry.py Phase 11:

        tasks["discord_status_broadcast"] = supervised_task(
            "discord_status_broadcast",
            discord_status_broadcast_loop(
                connector, synapse, oikos, config.instance_id
            ),
            restart=True,
        )

    The loop skips silently when ECODIAOS_DISCORD_CHANNEL_ID
    is not set, so no crash on unconfigured deployments.
    """
    channel_id = os.environ.get("ECODIAOS_DISCORD_CHANNEL_ID", "")
    if not channel_id:
        logger.debug("discord_broadcast_disabled", reason="DISCORD_CHANNEL_ID not set")
        return

    effective_interval = interval_s or int(
        os.environ.get("ECODIAOS_DISCORD_STATUS_INTERVAL_S", str(_DEFAULT_INTERVAL_S))
    )

    logger.info(
        "discord_broadcast_loop_started",
        channel_id=channel_id,
        interval_s=effective_interval,
    )

    while True:
        await asyncio.sleep(effective_interval)
        try:
            title, description = _build_status_message(synapse, oikos, instance_id)
            await discord_connector.send_embed(
                channel_id=channel_id,
                title=title,
                description=description,
                color=0x7289DA,
            )
            logger.info("discord_broadcast_sent", channel_id=channel_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("discord_broadcast_send_failed", error=str(exc))
            # Non-fatal - continue loop; supervised_task will restart on crash
