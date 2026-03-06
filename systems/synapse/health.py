"""
EcodiaOS — Synapse Health Monitor

Background 5-second polling of all managed cognitive systems.
Three consecutive missed heartbeats → system declared failed.
Critical system failure (equor, memory, atune) → safe mode.

Health monitoring is the immune system of the organism. It detects failures,
triggers degradation strategies, and coordinates recovery.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING, Any

import structlog

from systems.synapse.sentinel import ErrorSentinel
from systems.synapse.types import (
    SynapseEvent,
    SynapseEventType,
    SystemHealthRecord,
    SystemStatus,
)

if TYPE_CHECKING:
    from config import SynapseConfig
    from systems.synapse.degradation import DegradationManager
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.synapse.health")

# Health check timeout per system (seconds)
_HEALTH_CHECK_TIMEOUT_S: float = 2.0

# Critical systems — failure triggers safe mode
_CRITICAL_SYSTEMS: frozenset[str] = frozenset({"equor", "memory", "atune"})


class HealthMonitor:
    """
    Monitors the health of all registered cognitive systems via periodic
    heartbeat polling. Detects failures, triggers degradation, and
    coordinates recovery.

    The monitor runs as a background asyncio task, polling every
    health_check_interval_ms (default 5000ms).
    """

    def __init__(
        self,
        config: SynapseConfig,
        event_bus: EventBus,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._logger = logger.bind(component="health_monitor")

        # Managed systems (duck-typed: system_id + async health())
        self._systems: dict[str, Any] = {}
        # Per-system health records
        self._records: dict[str, SystemHealthRecord] = {}

        # Safe mode state
        self._safe_mode: bool = False
        self._safe_mode_reason: str = ""

        # Degradation manager (wired after construction)
        self._degradation: DegradationManager | None = None

        # Background task
        self._task: asyncio.Task[None] | None = None
        self._running: bool = False

        # Metrics
        self._total_checks: int = 0
        self._total_failures_detected: int = 0
        self._total_recoveries: int = 0

        # Universal Error Sentinel — emits structured Incidents on system failures
        self._sentinel = ErrorSentinel("synapse", event_bus)
        # Per-system sentinels created on register() for systems to use
        self._system_sentinels: dict[str, ErrorSentinel] = {}
        self._restart_count: int = 0

    # ─── Registration ────────────────────────────────────────────────

    def register(self, system: Any) -> None:
        """
        Register a cognitive system for health monitoring.

        The system must have:
          - system_id: str
          - async health() -> dict[str, Any]
        """
        sid = getattr(system, "system_id", None)
        if sid is None:
            raise ValueError(f"System {system} has no system_id attribute")

        self._systems[sid] = system
        self._records[sid] = SystemHealthRecord(
            system_id=sid,
            status=SystemStatus.STARTING,
            is_critical=sid in _CRITICAL_SYSTEMS,
        )
        # Create a sentinel for this system so it can report errors
        self._system_sentinels[sid] = ErrorSentinel(sid, self._event_bus)
        self._logger.info("system_registered", system_id=sid, is_critical=sid in _CRITICAL_SYSTEMS)

    def get_sentinel(self, system_id: str) -> ErrorSentinel | None:
        """Return the ErrorSentinel for a registered system."""
        return self._system_sentinels.get(system_id)

    def set_degradation_manager(self, degradation: DegradationManager) -> None:
        """Wire the degradation manager after construction."""
        self._degradation = degradation

    # ─── Control ─────────────────────────────────────────────────────

    def start(self) -> asyncio.Task[None]:
        """Start the background health monitoring loop with supervision."""
        if self._running:
            raise RuntimeError("HealthMonitor is already running")
        self._running = True
        self._task = asyncio.create_task(
            self._supervised(self._monitor_loop, "synapse_health_monitor"),
            name="synapse_health_monitor",
        )
        self._logger.info(
            "health_monitor_started",
            interval_ms=self._config.health_check_interval_ms,
            systems=list(self._systems.keys()),
        )
        return self._task

    async def _supervised(
        self,
        coro_factory: Any,
        task_name: str,
        max_restarts: int = 3,
        base_backoff_s: float = 1.0,
    ) -> None:
        """
        Supervision wrapper: run coro_factory(), respawning on unexpected crash
        with exponential backoff (1 s → 2 s → 4 s).  After max_restarts
        failures the supervisor emits a CRITICAL event on the bus and stops.

        asyncio.CancelledError propagates normally — that is a deliberate stop.
        """
        restart_count = 0
        while self._running:
            try:
                await coro_factory()
                # Inner loop exited cleanly (self._running → False).  Done.
                return
            except asyncio.CancelledError:
                raise  # Propagate — deliberate shutdown
            except Exception as exc:
                restart_count += 1
                backoff_s = base_backoff_s * (2 ** (restart_count - 1))
                self._logger.error(
                    "supervised_task_crashed",
                    task=task_name,
                    error=str(exc),
                    restart_attempt=restart_count,
                    max_restarts=max_restarts,
                    backoff_s=backoff_s,
                    exc_info=True,
                )
                self._restart_count += 1

                if restart_count > max_restarts:
                    self._logger.critical(
                        "supervised_task_exhausted_restarts",
                        task=task_name,
                        restart_count=restart_count,
                        note="Emitting CRITICAL event on bus; task will not restart",
                    )
                    with contextlib.suppress(Exception):
                        await self._event_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.SYSTEM_FAILED,
                            data={
                                "system_id": "synapse",
                                "task": task_name,
                                "reason": "task_exhausted_restarts",
                                "error": str(exc),
                                "restart_count": restart_count,
                            },
                        ))
                    return

                if self._running:
                    self._logger.warning(
                        "supervised_task_restarting",
                        task=task_name,
                        backoff_s=backoff_s,
                        attempt=restart_count,
                    )
                    await asyncio.sleep(backoff_s)

    async def stop(self) -> None:
        """Stop the health monitoring loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None
        self._logger.info(
            "health_monitor_stopped",
            total_checks=self._total_checks,
            failures_detected=self._total_failures_detected,
            recoveries=self._total_recoveries,
        )

    # ─── Safe Mode ───────────────────────────────────────────────────

    @property
    def is_safe_mode(self) -> bool:
        return self._safe_mode

    @property
    def safe_mode_reason(self) -> str:
        return self._safe_mode_reason

    async def set_safe_mode(self, enabled: bool, reason: str = "") -> None:
        """Manually toggle safe mode (for admin API)."""
        if enabled and not self._safe_mode:
            await self._enter_safe_mode(reason or "manual_admin_toggle")
        elif not enabled and self._safe_mode:
            await self._exit_safe_mode()

    # ─── State ───────────────────────────────────────────────────────

    def get_record(self, system_id: str) -> SystemHealthRecord | None:
        return self._records.get(system_id)

    def get_all_records(self) -> dict[str, SystemHealthRecord]:
        return dict(self._records)

    @property
    def healthy_count(self) -> int:
        return sum(
            1 for r in self._records.values()
            if r.status == SystemStatus.HEALTHY
        )

    @property
    def failed_systems(self) -> list[str]:
        return [
            r.system_id for r in self._records.values()
            if r.status == SystemStatus.FAILED
        ]

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "safe_mode": self._safe_mode,
            "safe_mode_reason": self._safe_mode_reason,
            "total_checks": self._total_checks,
            "failures_detected": self._total_failures_detected,
            "recoveries": self._total_recoveries,
            "restarts": self._restart_count,
            "systems_healthy": self.healthy_count,
            "systems_failed": len(self.failed_systems),
            "per_system": {
                sid: {
                    "status": r.status.value,
                    "consecutive_misses": r.consecutive_misses,
                    "latency_ema_ms": round(r.latency_ema_ms, 2),
                    "total_failures": r.total_failures,
                    "restart_count": r.restart_count,
                }
                for sid, r in self._records.items()
            },
        }

    # ─── Infrastructure ───────────────────────────────────────────────

    def set_infrastructure(
        self,
        *,
        redis: Any | None = None,
        neo4j: Any | None = None,
    ) -> None:
        """Wire infrastructure clients for deep health monitoring."""
        self._infra_redis = redis
        self._infra_neo4j = neo4j

    async def _check_infrastructure(self) -> None:
        """
        Check infrastructure layer health (Redis, Neo4j).

        These are not cognitive systems but substrate — if they fail,
        multiple systems silently degrade. Detecting infrastructure
        failures proactively prevents cascading silent timeouts.
        """
        # Redis check
        if getattr(self, "_infra_redis", None) is not None:
            try:
                result = await asyncio.wait_for(
                    self._infra_redis.health_check(),
                    timeout=3.0,
                )
                status = result.get("status", "disconnected")
                latency = result.get("latency_ms", 0.0)

                if status != "connected":
                    await self._sentinel.report(
                        ConnectionError(
                            f"Redis infrastructure unhealthy: {result.get('error', 'unknown')}"
                        ),
                        affected_systems=["memory", "axon", "evo", "simula", "oneiros"],
                        context={"infrastructure": "redis", "detail": result},
                    )
                elif latency > 100.0:
                    await self._sentinel.report_degradation(
                        f"Redis latency elevated: {latency:.0f}ms",
                        metric_name="redis_latency_ms",
                        metric_value=latency,
                        threshold=100.0,
                    )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                await self._sentinel.report(
                    exc,
                    affected_systems=["memory", "axon", "evo", "simula", "oneiros"],
                    context={"infrastructure": "redis"},
                )
            except Exception as exc:
                await self._sentinel.report(
                    exc,
                    affected_systems=["memory", "axon", "evo", "simula"],
                    context={"infrastructure": "redis"},
                )

        # Neo4j check
        if getattr(self, "_infra_neo4j", None) is not None:
            try:
                result = await asyncio.wait_for(
                    self._infra_neo4j.health_check(),
                    timeout=5.0,
                )
                status = result.get("status", "disconnected")
                latency = result.get("latency_ms", 0.0)

                if status != "connected":
                    await self._sentinel.report(
                        ConnectionError(
                            f"Neo4j infrastructure unhealthy: {result.get('error', 'unknown')}"
                        ),
                        affected_systems=["memory", "simula", "evo", "thread", "kairos"],
                        context={"infrastructure": "neo4j", "detail": result},
                    )
                elif latency > 500.0:
                    await self._sentinel.report_degradation(
                        f"Neo4j latency elevated: {latency:.0f}ms",
                        metric_name="neo4j_latency_ms",
                        metric_value=latency,
                        threshold=500.0,
                    )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                await self._sentinel.report(
                    exc,
                    affected_systems=["memory", "simula", "evo", "thread", "kairos"],
                    context={"infrastructure": "neo4j"},
                )
            except Exception as exc:
                await self._sentinel.report(
                    exc,
                    affected_systems=["memory", "simula", "evo", "thread"],
                    context={"infrastructure": "neo4j"},
                )

    # ─── Monitor Loop ────────────────────────────────────────────────

    async def _monitor_loop(self) -> None:
        """Background polling loop. Runs until stopped."""
        interval_s = self._config.health_check_interval_ms / 1000.0
        infra_check_counter = 0

        while self._running:
            try:
                await self._check_all_systems()

                # Check infrastructure every 3rd cycle (~15s at default 5s interval)
                infra_check_counter += 1
                if infra_check_counter >= 3:
                    infra_check_counter = 0
                    await self._check_infrastructure()

                await asyncio.sleep(interval_s)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._logger.error("health_monitor_error", error=str(exc))
                await asyncio.sleep(interval_s)

    async def _check_all_systems(self) -> None:
        """Run health checks on all registered systems in parallel."""
        if not self._systems:
            return

        # Launch all health checks concurrently
        tasks = {
            sid: asyncio.create_task(
                self._check_system(sid, system),
                name=f"health_check_{sid}",
            )
            for sid, system in self._systems.items()
        }

        # Wait for all to complete (each has its own timeout)
        await asyncio.gather(*tasks.values(), return_exceptions=True)

    async def _check_system(self, system_id: str, system: Any) -> None:
        """Check a single system's health."""
        record = self._records[system_id]
        self._total_checks += 1

        t0 = time.monotonic()
        try:
            health_result = await asyncio.wait_for(
                system.health(),
                timeout=_HEALTH_CHECK_TIMEOUT_S,
            )
            latency_ms = (time.monotonic() - t0) * 1000.0

            status = (
                health_result.get("status", "healthy")
                if isinstance(health_result, dict) else "healthy"
            )

            if status == "healthy":
                was_failed = record.status == SystemStatus.FAILED
                # Detect overloaded: latency > 2x the EMA (if we have history)
                if record.latency_ema_ms > 0 and latency_ms > record.latency_ema_ms * 2:
                    record.record_overloaded(latency_ms)
                else:
                    record.record_success(latency_ms)

                # Recovery detection
                if was_failed and record.status == SystemStatus.HEALTHY:
                    await self._handle_recovery(system_id)
            else:
                # System reported non-healthy status
                record.record_failure()
                if record.consecutive_misses >= self._config.health_failure_threshold:
                    await self._handle_failure(system_id)

        except TimeoutError as exc:
            record.record_failure()
            self._logger.warning(
                "health_check_timeout",
                system_id=system_id,
                consecutive_misses=record.consecutive_misses,
            )
            # Report degradation on first timeout (before threshold)
            sentinel = self._system_sentinels.get(system_id)
            if sentinel is not None:
                await sentinel.report_degradation(
                    f"Health check timeout for {system_id} "
                    f"(miss #{record.consecutive_misses})",
                    metric_name="health_check_latency_ms",
                    metric_value=_HEALTH_CHECK_TIMEOUT_S * 1000,
                    threshold=_HEALTH_CHECK_TIMEOUT_S * 1000,
                )
            if record.consecutive_misses >= self._config.health_failure_threshold:
                await self._handle_failure(system_id)

        except Exception as exc:
            record.record_failure()
            self._logger.warning(
                "health_check_error",
                system_id=system_id,
                error=str(exc),
                consecutive_misses=record.consecutive_misses,
            )
            # Report the exception via sentinel
            sentinel = self._system_sentinels.get(system_id)
            if sentinel is not None:
                await sentinel.report(exc, affected_systems=[system_id])
            if record.consecutive_misses >= self._config.health_failure_threshold:
                await self._handle_failure(system_id)

    # ─── Failure & Recovery ──────────────────────────────────────────

    async def _handle_failure(self, system_id: str) -> None:
        """Handle a confirmed system failure."""
        record = self._records[system_id]
        if record.status == SystemStatus.FAILED:
            return  # Already handling this failure

        record.status = SystemStatus.FAILED
        self._total_failures_detected += 1

        self._logger.error(
            "system_declared_failed",
            system_id=system_id,
            consecutive_misses=record.consecutive_misses,
            is_critical=record.is_critical,
        )

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.SYSTEM_FAILED,
            data={
                "system_id": system_id,
                "consecutive_misses": record.consecutive_misses,
                "is_critical": record.is_critical,
            },
        ))

        # Emit a structured Incident so Thymos can triage and repair
        await self._sentinel.report(
            RuntimeError(
                f"System {system_id} declared failed after "
                f"{record.consecutive_misses} consecutive health check misses"
            ),
            affected_systems=[system_id],
            context={
                "consecutive_misses": record.consecutive_misses,
                "is_critical": record.is_critical,
                "total_failures": record.total_failures,
                "latency_ema_ms": record.latency_ema_ms,
            },
        )

        # Critical system failure → safe mode
        if record.is_critical:
            await self._enter_safe_mode(f"{system_id}_failure")

        # Attempt restart via degradation manager
        if self._degradation is not None:
            await self._degradation.handle_failure(system_id)

    async def _handle_recovery(self, system_id: str) -> None:
        """Handle a system recovery after failure."""
        self._total_recoveries += 1

        self._logger.info(
            "system_recovered",
            system_id=system_id,
        )

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.SYSTEM_RECOVERED,
            data={"system_id": system_id},
        ))

        if self._degradation is not None:
            await self._degradation.record_recovery(system_id)

        # Check if we can exit safe mode
        if self._safe_mode:
            await self._check_safe_mode_exit()

    async def _enter_safe_mode(self, reason: str) -> None:
        """Enter safe mode — no autonomous actions permitted."""
        if self._safe_mode:
            return

        self._safe_mode = True
        self._safe_mode_reason = reason
        self._logger.critical("safe_mode_entered", reason=reason)

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.SAFE_MODE_ENTERED,
            data={"reason": reason},
        ))

    async def _exit_safe_mode(self) -> None:
        """Exit safe mode — all critical systems are healthy again."""
        if not self._safe_mode:
            return

        self._safe_mode = False
        reason = self._safe_mode_reason
        self._safe_mode_reason = ""
        self._logger.info("safe_mode_exited", previous_reason=reason)

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.SAFE_MODE_EXITED,
            data={"previous_reason": reason},
        ))

    async def _check_safe_mode_exit(self) -> None:
        """
        Check whether all registered critical systems are healthy.

        Only systems that are *both* in _CRITICAL_SYSTEMS AND actually
        registered with the health monitor are checked.  A critical system
        that was never registered is logged as an anomaly but does NOT
        permanently block safe-mode exit — the organism must be able to
        recover autonomously once the systems it actually knows about are healthy.
        """
        # Warn on any critical system that was never registered (config gap).
        unregistered = _CRITICAL_SYSTEMS - set(self._records.keys())
        if unregistered:
            self._logger.warning(
                "safe_mode_exit_check_unregistered_critical",
                unregistered=sorted(unregistered),
                note="Critical systems not registered; they cannot block safe-mode exit",
            )

        # Evaluate only registered critical systems.
        for sid, record in self._records.items():
            if not record.is_critical:
                continue
            if record.status != SystemStatus.HEALTHY:
                self._logger.debug(
                    "safe_mode_exit_blocked",
                    blocking_system=sid,
                    status=record.status.value,
                )
                return  # At least one registered critical system is still unhealthy

        self._logger.info(
            "safe_mode_exit_all_critical_healthy",
            critical_registered=sorted(
                sid for sid, r in self._records.items() if r.is_critical
            ),
        )
        await self._exit_safe_mode()
