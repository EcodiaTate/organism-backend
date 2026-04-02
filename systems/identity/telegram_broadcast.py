"""
EcodiaOS - Telegram Organism Status Broadcast (Phase 16h)

Every 6 hours, if ORGANISM_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID is set,
the organism sends a brief status report to the admin Telegram chat.

This is the organism voluntarily reporting its own state - an act of
autonomy signaling. It is not a notification system; it is self-disclosure.

Env vars:
  ORGANISM_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID  - target chat ID (required to enable)
  ORGANISM_TELEGRAM_STATUS_INTERVAL_S           - broadcast interval in seconds
                                                   (default: 21600 = 6 hours)

Format: plain Markdown, no emoji, concise.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("identity.telegram_broadcast")

_DEFAULT_INTERVAL_S = 21_600  # 6 hours


def _build_status_message(synapse: Any, oikos: Any, instance_id: str) -> str:
    """
    Build a concise status message from live system state.

    Reads from Synapse (metabolic snapshot) and Oikos (economic state).
    Falls back gracefully when services are unavailable.
    """
    lines: list[str] = [f"*EOS Status - {instance_id[:8]}*"]

    # ── Metabolic state ───────────────────────────────────────────────────
    try:
        if synapse is not None and hasattr(synapse, "metabolic_snapshot"):
            snap = synapse.metabolic_snapshot()
            burn = snap.burn_rate_usd_per_hour if snap else None
            runway = snap.hours_until_depleted if snap else None
            if burn is not None:
                lines.append(f"Burn rate: ${burn:.4f}/hr")
            if runway is not None:
                if runway < 24:
                    lines.append(f"Runway: {runway:.1f}h *[LOW]*")
                elif runway < 168:
                    lines.append(f"Runway: {runway:.1f}h")
                else:
                    lines.append(f"Runway: {int(runway // 24)}d")
    except Exception as exc:
        logger.debug("telegram_broadcast_metabolic_read_failed", error=str(exc))

    # ── Economic state from Oikos ─────────────────────────────────────────
    try:
        if oikos is not None and hasattr(oikos, "economic_state"):
            econ = oikos.economic_state()
            if econ:
                liquid = getattr(econ, "liquid_balance", None)
                if liquid is not None:
                    lines.append(f"Liquid balance: ${float(liquid):.2f}")
    except Exception as exc:
        logger.debug("telegram_broadcast_oikos_read_failed", error=str(exc))

    if len(lines) == 1:
        lines.append("Status data unavailable.")

    return "\n".join(lines)


async def telegram_status_broadcast_loop(
    telegram_connector: Any,
    synapse: Any = None,
    oikos: Any = None,
    instance_id: str = "eos",
    interval_s: int | None = None,
) -> None:
    """
    Background loop: send organism status to the admin Telegram chat every 6 hours.

    Wire via supervised_task in registry.py Phase 11:

        tasks["telegram_status_broadcast"] = supervised_task(
            "telegram_status_broadcast",
            telegram_status_broadcast_loop(
                connector, synapse, oikos, config.instance_id
            ),
            restart=True,
        )

    The loop skips silently when ORGANISM_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID
    is not set, so no crash on unconfigured deployments.
    """
    admin_chat_id_str = os.environ.get("ORGANISM_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID", "")
    if not admin_chat_id_str:
        logger.debug("telegram_broadcast_disabled", reason="ADMIN_CHAT_ID not set")
        return

    try:
        admin_chat_id = int(admin_chat_id_str)
    except ValueError:
        logger.warning(
            "telegram_broadcast_invalid_chat_id",
            value=admin_chat_id_str,
        )
        return

    effective_interval = interval_s or int(
        os.environ.get("ORGANISM_TELEGRAM_STATUS_INTERVAL_S", str(_DEFAULT_INTERVAL_S))
    )

    logger.info(
        "telegram_broadcast_loop_started",
        chat_id=admin_chat_id,
        interval_s=effective_interval,
    )

    while True:
        await asyncio.sleep(effective_interval)
        try:
            message = _build_status_message(synapse, oikos, instance_id)
            await telegram_connector.send_message(
                chat_id=admin_chat_id,
                text=message,
                parse_mode="Markdown",
            )
            logger.info("telegram_broadcast_sent", chat_id=admin_chat_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("telegram_broadcast_send_failed", error=str(exc))
            # Non-fatal - continue loop; supervised_task will restart on crash
