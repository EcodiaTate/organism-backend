"""
EcodiaOS - Child-Side Health Reporter (Spec 26 §7 / HIGH #2)

When EOS runs as a child instance (IS_GENESIS_NODE=false), this module
starts a background loop that emits CHILD_HEALTH_REPORT every 10 minutes
to the parent's federation address.

The report carries:
  cpu_usage, memory_usage, episode_count, hypothesis_count,
  drive_alignment_scores, constitutional_drift_severity

Wire-up:
  reporter = ChildHealthReporter(config, synapse, soma, evo, equor)
  await reporter.start()          # non-blocking, starts background task
  await reporter.stop()           # on shutdown
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now

if TYPE_CHECKING:
    from systems.equor.service import EquorService
    from systems.evo.service import EvoService
    from systems.soma.service import SomaService
    from systems.synapse.service import SynapseService

logger = structlog.get_logger().bind(component="mitosis.health_reporter")

_REPORT_INTERVAL_S = 600  # 10 minutes


class ChildHealthReporter:
    """
    Emits CHILD_HEALTH_REPORT every 10 minutes when running as a child instance.

    Activated only when ORGANISM_IS_GENESIS_NODE != 'true'. Safe to instantiate
    on a genesis node - start() will no-op.

    Parameters
    ----------
    synapse : SynapseService
        Used to emit the health report event.
    soma : SomaService | None
        Source of cpu_usage, memory_usage, allostatic state.
    evo : EvoService | None
        Source of hypothesis_count.
    equor : EquorService | None
        Source of constitutional_drift_severity.
    instance_id : str
        This child's instance ID (defaults to ORGANISM_INSTANCE_ID env var).
    parent_instance_id : str
        Parent's instance ID (from ORGANISM_PARENT_INSTANCE_ID env var).
    """

    def __init__(
        self,
        synapse: SynapseService | None = None,
        soma: SomaService | None = None,
        evo: EvoService | None = None,
        equor: EquorService | None = None,
        instance_id: str = "",
        parent_instance_id: str = "",
        report_interval_s: float = _REPORT_INTERVAL_S,
    ) -> None:
        self._synapse = synapse
        self._soma = soma
        self._evo = evo
        self._equor = equor
        self._instance_id = instance_id or os.environ.get("ORGANISM_INSTANCE_ID", "")
        self._parent_instance_id = parent_instance_id or os.environ.get(
            "ORGANISM_PARENT_INSTANCE_ID", ""
        )
        self._is_child = os.environ.get("ORGANISM_IS_GENESIS_NODE", "true").lower() != "true"
        self._interval = report_interval_s
        self._task: asyncio.Task[None] | None = None
        self._log = logger.bind(
            instance_id=self._instance_id,
            parent_id=self._parent_instance_id,
        )

    async def start(self) -> None:
        """Start the health report background loop (no-op on genesis nodes)."""
        if not self._is_child:
            self._log.debug("health_reporter_genesis_node_skip")
            return
        if not self._parent_instance_id:
            self._log.warning("health_reporter_no_parent_id")
            return
        if self._task is not None:
            return

        # Run cert handshake first - emits CHILD_WALLET_REPORTED (triggers deferred
        # seed transfer) and FEDERATION_PEER_CONNECTED. Non-fatal: failure is logged
        # but does not prevent the health reporter from starting.
        await self._run_cert_handshake()

        # Problem 4: Subscribe to CHILD_HEALTH_REQUEST so parent probes trigger an
        # immediate CHILD_HEALTH_REPORT instead of waiting for the next 10-min cycle.
        if self._synapse is not None:
            event_bus = getattr(self._synapse, "event_bus", None)
            if event_bus is not None and hasattr(event_bus, "subscribe"):
                try:
                    from systems.synapse.types import SynapseEventType
                    event_bus.subscribe(
                        SynapseEventType.CHILD_HEALTH_REQUEST,
                        self._on_child_health_request,
                    )
                    self._log.info("child_health_request_subscription_registered")
                except Exception as exc:
                    self._log.warning(
                        "child_health_request_subscription_failed", error=str(exc)
                    )

        self._task = asyncio.create_task(
            self._report_loop(),
            name="child_health_reporter",
        )
        self._log.info("child_health_reporter_started", interval_s=self._interval)

    async def _run_cert_handshake(self) -> None:
        """Validate birth certificate and announce wallet to parent (Spec 26 §6)."""
        from systems.mitosis.cert_handshake import (
            CertificateValidationError,
            ChildCertHandshake,
        )

        # SynapseService is what the handshake needs for event_bus access
        handshake = ChildCertHandshake(
            synapse=self._synapse,
            instance_id=self._instance_id,
            parent_instance_id=self._parent_instance_id,
        )
        try:
            cert_info = await handshake.run()
            self._log.info(
                "child_cert_handshake_complete",
                serial=cert_info.get("serial", ""),
                ca_verified=cert_info.get("ca_verified", False),
            )
        except CertificateValidationError as exc:
            self._log.error(
                "child_cert_handshake_failed",
                error=str(exc),
                note="Seed capital transfer deferred until cert is valid",
            )
        except Exception as exc:
            self._log.warning("child_cert_handshake_error", error=str(exc))

    async def stop(self) -> None:
        """Cancel the background loop."""
        if self._task is not None:
            self._task.cancel()
            self._task = None
            self._log.info("child_health_reporter_stopped")

    # ── Internal ─────────────────────────────────────────────────

    async def _report_loop(self) -> None:
        """Emit health report every _interval seconds."""
        # Stagger first report by 30s to let the instance fully boot
        await asyncio.sleep(30)
        while True:
            try:
                await self._emit_report()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._log.error("health_report_failed", error=str(exc))
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break

    async def _on_child_health_request(self, event: Any) -> None:
        """Handle CHILD_HEALTH_REQUEST - emit an immediate CHILD_HEALTH_REPORT.

        Oikos emits CHILD_HEALTH_REQUEST every 10 minutes to probe each live child.
        This handler responds immediately rather than waiting for the next scheduled
        _report_loop cycle, closing the 10-minute visibility gap.

        Only responds if the request targets this child instance (child_instance_id
        matches self._instance_id) or if no specific target is given.
        """
        data = event.data if hasattr(event, "data") else {}
        target_id = str(data.get("child_instance_id", ""))
        if target_id and target_id != self._instance_id:
            # Request is for a different child - ignore
            return

        self._log.debug(
            "child_health_request_received",
            request_id=data.get("request_id", ""),
            parent_id=data.get("parent_instance_id", ""),
        )
        try:
            await self._emit_report()
        except Exception as exc:
            self._log.error("child_health_request_report_failed", error=str(exc))

    async def _emit_report(self) -> None:
        """Collect metrics and emit CHILD_HEALTH_REPORT."""
        if self._synapse is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        metrics = await self._collect_metrics()
        await self._synapse.event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.CHILD_HEALTH_REPORT,
            source_system=f"child.{self._instance_id}",
            data={
                "child_instance_id": self._instance_id,
                "parent_instance_id": self._parent_instance_id,
                "report_id": new_id(),
                "reported_at": utc_now().isoformat(),
                **metrics,
            },
        ))
        self._log.debug("health_report_emitted", metrics=metrics)

    async def _collect_metrics(self) -> dict[str, Any]:
        """Gather metrics from available services."""
        metrics: dict[str, Any] = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "episode_count": 0,
            "hypothesis_count": 0,
            "drive_alignment_scores": {},
            "constitutional_drift_severity": 0.0,
            # Financial health (populated if Oikos is available)
            "net_worth_usd": "0",
            "runway_days": 0.0,
            "efficiency": 0.0,
            "net_income_7d": "0",
            "consecutive_positive_days": 0,
        }

        # CPU + memory from Soma
        if self._soma is not None:
            try:
                soma_state = await _safe_call(self._soma.get_interoceptive_state)
                if soma_state is not None:
                    metrics["cpu_usage"] = float(getattr(soma_state, "cpu_load", 0.0))
                    metrics["memory_usage"] = float(getattr(soma_state, "memory_pressure", 0.0))
            except Exception:
                # Fallback to psutil if Soma unavailable
                try:
                    import psutil

                    metrics["cpu_usage"] = psutil.cpu_percent(interval=0.1) / 100.0
                    mem = psutil.virtual_memory()
                    metrics["memory_usage"] = mem.percent / 100.0
                except ImportError:
                    pass

        # Hypothesis count from Evo
        if self._evo is not None:
            try:
                hyp_count = await _safe_call(self._evo.get_hypothesis_count)
                if hyp_count is not None:
                    metrics["hypothesis_count"] = int(hyp_count)
            except Exception:
                pass

        # Drive alignment + drift severity from Equor
        if self._equor is not None:
            try:
                alignment = await _safe_call(self._equor.get_drive_alignment_scores)
                if alignment is not None:
                    metrics["drive_alignment_scores"] = dict(alignment)
                drift = await _safe_call(self._equor.get_constitutional_drift_severity)
                if drift is not None:
                    metrics["constitutional_drift_severity"] = float(drift)
            except Exception:
                pass

        return metrics


async def _safe_call(coro_fn: Any, *args: Any) -> Any:
    """Call a coroutine function and return None on any exception."""
    try:
        result = coro_fn(*args)
        if asyncio.iscoroutine(result):
            return await result
        return result
    except Exception:
        return None
