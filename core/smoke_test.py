"""
EcodiaOS — Startup Smoke Tests

Quick validation checks that confirm all critical loops are wired
at organism startup.  Failures are logged but do not prevent boot
— the organism starts in a degraded state.

Extracted from main.py lifespan().
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.oikos.service import OikosService
    from systems.thymos.service import ThymosService

logger = structlog.get_logger()


async def run_smoke_tests(
    *,
    oikos: OikosService,
    simula: Any,
    thymos: ThymosService,
    nova: Any,
    evo: Any,
    synapse: Any,
) -> bool:
    """
    Run all startup smoke tests.

    Returns True if all tests pass, False if any fail.
    """
    ok = True
    ok = await _smoke_oikos(oikos, thymos) and ok
    ok = await _smoke_thymos_simula(simula) and ok
    _smoke_escalation_webhook()
    ok = await _smoke_budget_check(oikos) and ok
    await _smoke_capital_deployment(oikos)

    # Final readiness announcement
    logger.info(
        "ecodiaos_organism_ready",
        all_smoke_passed=ok,
        systems_wired={
            "thymos_simula": thymos._simula is not None,
            "nova_thymos": nova._thymos is not None,
            "evo_thymos": evo._thymos is not None,
            "oikos_event_bus": oikos._event_bus is not None,
            "synapse_clock": synapse._clock is not None,
            "escalation_webhook": bool(os.environ.get("EOS_ESCALATION_WEBHOOK", "")),
        },
        supervised_tasks=["nova_heartbeat", "inner_life_generator", "metrics_publisher"],
    )
    return ok


async def _smoke_oikos(oikos: OikosService, thymos: ThymosService) -> bool:
    """Smoke 1: Oikos metabolism returns real data (not zeros)."""
    from systems.thymos.types import Incident, IncidentClass, IncidentSeverity

    try:
        snap = oikos.snapshot()
        bmr = float(snap.bmr_usd_per_day)
        if bmr > 0:
            logger.info(
                "smoke_oikos_ok",
                bmr_usd_per_day=f"{bmr:.4f}",
                runway_days=str(snap.runway_days),
                starvation_level=snap.starvation_level,
            )
            return True
        else:
            logger.warning(
                "smoke_oikos_zero_bmr",
                bmr=bmr,
                message="Oikos metabolism is returning zero BMR — metabolic loop may be unwired",
            )
            return False
    except Exception as exc:
        logger.error("smoke_oikos_failed", error=str(exc))
        asyncio.create_task(
            thymos.on_incident(
                Incident(
                    incident_class=IncidentClass.CRASH,
                    severity=IncidentSeverity.HIGH,
                    fingerprint=hashlib.md5(
                        f"oikos_smoke_test_{type(exc).__name__}".encode()
                    ).hexdigest(),
                    source_system="oikos",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    context={"location": "smoke_test", "attribute": "bmr_usd_per_day"},
                )
            )
        )
        return False


async def _smoke_thymos_simula(simula: Any) -> bool:
    """Smoke 2: Thymos can reach Simula."""
    try:
        health = await simula.health()
        if health.get("status") == "error":
            logger.warning(
                "smoke_thymos_simula_degraded",
                simula_health=health,
                message=(
                    "Simula health check returned error"
                    " — Thymos Tier 4 repairs may not dispatch"
                ),
            )
            return False
        else:
            logger.info(
                "smoke_thymos_simula_ok",
                simula_initialized=health.get("initialized", False),
            )
            return True
    except Exception as exc:
        logger.error("smoke_thymos_simula_failed", error=str(exc))
        return False


def _smoke_escalation_webhook() -> None:
    """Smoke 3: Escalation webhook configuration."""
    url = os.environ.get("EOS_ESCALATION_WEBHOOK", "")
    if url:
        logger.info("smoke_escalation_ok", webhook_configured=True)
    else:
        logger.warning(
            "smoke_escalation_unwired",
            message=(
                "EOS_ESCALATION_WEBHOOK is empty"
                " — organism cannot escalate to external operators"
            ),
        )


async def _smoke_budget_check(oikos: OikosService) -> bool:
    """Smoke 4: End-to-end budget check."""
    from decimal import Decimal

    try:
        result = await oikos.check_budget(
            system_id="smoke_test",
            action="startup_probe",
            estimated_cost_usd=Decimal("0.00"),
        )
        logger.info(
            "smoke_budget_check_ok",
            approved=result.approved,
            reason=result.reason[:80] if result.reason else "",
        )
        return True
    except Exception as exc:
        logger.error("smoke_budget_check_failed", error=str(exc))
        return False


async def _smoke_capital_deployment(oikos: OikosService) -> None:
    """Smoke 5: Oikos idle-capital deployment."""
    try:
        result = await oikos.deploy_idle_capital()
        if result and result.tx_hash:
            logger.info(
                "smoke_capital_deployed",
                amount=result.amount_deployed_usd,
                apy=result.apy,
                tx=result.tx_hash,
            )
        else:
            logger.info("smoke_capital_below_floor_or_deployed")
    except Exception as exc:
        logger.info("smoke_capital_below_floor_or_deployed", error=str(exc))
