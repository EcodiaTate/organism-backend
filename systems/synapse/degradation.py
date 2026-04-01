"""
EcodiaOS - Synapse Degradation Manager

Per-system graceful fallback strategies. When a system fails, Synapse
applies the appropriate degradation strategy: safe mode for critical
systems, queuing for Axon, raw fallback for Voxis, etc.

Auto-restart with exponential backoff ensures failed systems get
multiple recovery attempts before giving up.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

from systems.synapse.types import (
    DegradationLevel,
    DegradationStrategy,
    SynapseEvent,
    SynapseEventType,
)

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus
    from systems.synapse.health import HealthMonitor

logger = structlog.get_logger("systems.synapse.degradation")


# ─── Per-System Strategies (from spec) ─────────────────────────────

_STRATEGIES: dict[str, DegradationStrategy] = {
    "equor": DegradationStrategy(
        system_id="equor",
        triggers_safe_mode=True,
        fallback_behavior="Enter safe mode. No actions without ethics.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "memory": DegradationStrategy(
        system_id="memory",
        triggers_safe_mode=True,
        fallback_behavior="Enter safe mode. Use in-context memory only.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "atune": DegradationStrategy(
        system_id="atune",
        triggers_safe_mode=True,
        fallback_behavior="Enter safe mode. Direct input passthrough, no Fovea prediction error evaluation.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "nova": DegradationStrategy(
        system_id="nova",
        triggers_safe_mode=False,
        fallback_behavior="Voxis responds with 'I'm having difficulty thinking right now.'",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "voxis": DegradationStrategy(
        system_id="voxis",
        triggers_safe_mode=False,
        fallback_behavior="Use raw LLM output without personality rendering.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "axon": DegradationStrategy(
        system_id="axon",
        triggers_safe_mode=False,
        fallback_behavior="Queue intents, retry when Axon recovers.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "evo": DegradationStrategy(
        system_id="evo",
        triggers_safe_mode=False,
        fallback_behavior="Skip consolidation. No learning, but core function preserved.",
        auto_restart=True,
        max_restart_attempts=2,
    ),
    "simula": DegradationStrategy(
        system_id="simula",
        triggers_safe_mode=False,
        fallback_behavior="No evolution. Fully functional otherwise.",
        auto_restart=True,
        max_restart_attempts=2,
    ),
    "thread": DegradationStrategy(
        system_id="thread",
        triggers_safe_mode=False,
        fallback_behavior=(
            "Skip narrative synthesis. Structural state (commitments, schemas) continues updating."
        ),
        auto_restart=True,
        max_restart_attempts=2,
    ),
    "soma": DegradationStrategy(
        system_id="soma",
        triggers_safe_mode=False,
        fallback_behavior=(
            "Skip interoceptive analysis. Allostatic signals frozen at last known values."
        ),
        auto_restart=True,
        max_restart_attempts=2,
    ),
    # ── Previously unprotected systems ───────────────────────────────────
    "alive": DegradationStrategy(
        system_id="alive",
        triggers_safe_mode=False,
        fallback_behavior="WebSocket bridge offline. No real-time visualization. REST API still works.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "identity": DegradationStrategy(
        system_id="identity",
        triggers_safe_mode=False,
        fallback_behavior="No certificate provisioning. Existing sessions continue; new auth requests queued.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "kairos": DegradationStrategy(
        system_id="kairos",
        triggers_safe_mode=False,
        fallback_behavior="Causal discovery paused. Existing invariants remain valid. No new discoveries.",
        auto_restart=True,
        max_restart_attempts=2,
    ),
    "mitosis": DegradationStrategy(
        system_id="mitosis",
        triggers_safe_mode=False,
        fallback_behavior="No child spawning or fleet management. Existing children continue independently.",
        auto_restart=True,
        max_restart_attempts=2,
    ),
    "fovea": DegradationStrategy(
        system_id="fovea",
        triggers_safe_mode=False,
        fallback_behavior="Prediction error calculation skipped. Atune falls back to habituated salience.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "logos": DegradationStrategy(
        system_id="logos",
        triggers_safe_mode=False,
        fallback_behavior="Compression engine offline. New episodes stored uncompressed; consolidation deferred.",
        auto_restart=True,
        max_restart_attempts=2,
    ),
    "nexus": DegradationStrategy(
        system_id="nexus",
        triggers_safe_mode=False,
        fallback_behavior="Epistemic triangulation paused. Knowledge operates at local confidence only.",
        auto_restart=True,
        max_restart_attempts=2,
    ),
    "oikos": DegradationStrategy(
        system_id="oikos",
        triggers_safe_mode=False,
        fallback_behavior="Economic metabolism paused. No yield farming, spending frozen at survival tier.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "oneiros": DegradationStrategy(
        system_id="oneiros",
        triggers_safe_mode=False,
        fallback_behavior="Sleep compiler offline. No consolidation or dream hypothesis testing.",
        auto_restart=True,
        max_restart_attempts=2,
    ),
    "telos": DegradationStrategy(
        system_id="telos",
        triggers_safe_mode=False,
        fallback_behavior="Drive topology frozen at last known values. Constitutional binding still enforced by Equor.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "thymos": DegradationStrategy(
        system_id="thymos",
        triggers_safe_mode=False,
        fallback_behavior="Immune system degraded. Health monitor continues; no active incident triage or repair.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "eis": DegradationStrategy(
        system_id="eis",
        triggers_safe_mode=False,
        fallback_behavior="Epistemic immune screening disabled. Percepts pass through unfiltered.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
    "federation": DegradationStrategy(
        system_id="federation",
        triggers_safe_mode=False,
        fallback_behavior="Federation offline. Operate in standalone mode; no peer discovery or consensus.",
        auto_restart=True,
        max_restart_attempts=2,
    ),
    "benchmarks": DegradationStrategy(
        system_id="benchmarks",
        triggers_safe_mode=False,
        fallback_behavior="KPI measurement paused. No regression detection until recovery.",
        auto_restart=True,
        max_restart_attempts=2,
    ),
    "sacm": DegradationStrategy(
        system_id="sacm",
        triggers_safe_mode=False,
        fallback_behavior="Compute orchestration offline. Stay on current provider; no substrate arbitrage.",
        auto_restart=True,
        max_restart_attempts=2,
    ),
    "phantom": DegradationStrategy(
        system_id="phantom",
        triggers_safe_mode=False,
        fallback_behavior="LP price oracle offline. Use cached prices; no new swap event listening.",
        auto_restart=True,
        max_restart_attempts=2,
    ),
    "skia": DegradationStrategy(
        system_id="skia",
        triggers_safe_mode=False,
        fallback_behavior="Shadow infrastructure offline. No heartbeat monitoring or IPFS snapshots.",
        auto_restart=True,
        max_restart_attempts=3,
    ),
}


class DegradationManager:
    """
    Manages graceful degradation when cognitive systems fail.

    Each system has a defined fallback strategy. Critical systems
    (equor, memory, atune) trigger safe mode. Non-critical failures
    apply specific fallback behaviours and attempt auto-restart
    with exponential backoff.

    The dependency graph (populated via ``declare_dependency()``) enables
    ordered restarts: leaves are shut down first so that no system is
    restarted while something it depends on is still mid-cycle.
    """

    def __init__(
        self,
        event_bus: EventBus,
        health_monitor: HealthMonitor,
    ) -> None:
        self._event_bus = event_bus
        self._health = health_monitor
        self._logger = logger.bind(component="degradation_manager")

        # Managed systems (for restart)
        self._systems: dict[str, Any] = {}

        # Dependency graph: system_id → set of system_ids it depends on.
        # E.g. {"nova": {"memory", "equor"}} means Nova depends on Memory and Equor.
        self._dependencies: dict[str, set[str]] = {}
        # Reverse index: system_id → set of system_ids that depend on it.
        self._dependents: dict[str, set[str]] = {}

        # Restart tracking
        self._restart_attempts: dict[str, int] = {}
        self._restart_tasks: dict[str, asyncio.Task[None]] = {}

        # Current degradation level
        self._level: DegradationLevel = DegradationLevel.NOMINAL

    # ─── System Registration ─────────────────────────────────────────

    def register_system(self, system: Any) -> None:
        """Register a system for potential restart management."""
        sid = getattr(system, "system_id", None)
        if sid:
            self._systems[sid] = system

    def declare_dependency(self, dependent: str, dependency: str) -> None:
        """
        Declare that *dependent* holds a reference to *dependency*.

        This is used during hot-reload restarts to determine the correct
        shutdown/init order and to identify dependents that must be
        re-initialized after the target system restarts.
        """
        self._dependencies.setdefault(dependent, set()).add(dependency)
        self._dependents.setdefault(dependency, set()).add(dependent)

    # ─── Failure Handling ────────────────────────────────────────────

    async def handle_failure(self, system_id: str) -> None:
        """
        Apply the degradation strategy for a failed system.

        1. Look up the strategy
        2. Log the fallback behaviour
        3. Attempt auto-restart with exponential backoff
        4. Update the overall degradation level
        """
        strategy = _STRATEGIES.get(system_id)
        if strategy is None:
            self._logger.warning(
                "no_degradation_strategy",
                system_id=system_id,
                note="Using generic fallback strategy",
            )
            # Use a generic strategy for unregistered systems
            strategy = DegradationStrategy(
                system_id=system_id,
                triggers_safe_mode=False,
                fallback_behavior=f"System {system_id} degraded. Skipping non-essential operations.",
                auto_restart=True,
                max_restart_attempts=2,
            )

        self._logger.warning(
            "applying_degradation_strategy",
            system_id=system_id,
            fallback=strategy.fallback_behavior,
            triggers_safe_mode=strategy.triggers_safe_mode,
        )

        # Cascade detection: proactively notify dependent systems
        await self._propagate_cascade(system_id)

        # Update level
        self._update_level()

        # Auto-restart if configured
        if strategy.auto_restart:
            attempts = self._restart_attempts.get(system_id, 0)
            if attempts < strategy.max_restart_attempts:
                self._schedule_restart(system_id, strategy, attempts)
            else:
                self._logger.error(
                    "max_restart_attempts_reached",
                    system_id=system_id,
                    attempts=attempts,
                )

    async def _propagate_cascade(self, failed_system_id: str) -> None:
        """
        Proactively notify dependent systems that a dependency has failed.

        Instead of waiting for each dependent to timeout individually,
        emit SYSTEM_DEGRADED events so they can enter fallback immediately.
        """
        dependents = self._dependents.get(failed_system_id, set())
        if not dependents:
            return

        self._logger.warning(
            "cascade_detected",
            failed_system=failed_system_id,
            affected_dependents=sorted(dependents),
        )

        for dependent_id in dependents:
            try:
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SYSTEM_DEGRADED,
                    data={
                        "system_id": dependent_id,
                        "reason": f"dependency_{failed_system_id}_failed",
                        "cascade_source": failed_system_id,
                    },
                ))
            except Exception as exc:
                self._logger.warning(
                    "cascade_notification_failed",
                    dependent=dependent_id,
                    error=str(exc),
                )

    async def record_recovery(self, system_id: str) -> None:
        """Record that a system has recovered. Reset restart counter."""
        self._restart_attempts[system_id] = 0

        # Cancel any pending restart task
        task = self._restart_tasks.pop(system_id, None)
        if task and not task.done():
            task.cancel()

        self._update_level()
        self._logger.info("recovery_recorded", system_id=system_id)

    # ─── Auto-Restart ────────────────────────────────────────────────

    def _schedule_restart(
        self,
        system_id: str,
        strategy: DegradationStrategy,
        attempt: int,
    ) -> None:
        """Schedule an auto-restart with exponential backoff."""
        # Exponential backoff: base * 2^attempt
        delay_s = strategy.restart_backoff_base_s * (2 ** attempt)

        self._logger.info(
            "scheduling_restart",
            system_id=system_id,
            attempt=attempt + 1,
            max_attempts=strategy.max_restart_attempts,
            delay_s=delay_s,
        )

        task = asyncio.create_task(
            self._restart_system(system_id, attempt, delay_s),
            name=f"restart_{system_id}_{attempt}",
        )
        self._restart_tasks[system_id] = task

    async def _restart_system(
        self,
        system_id: str,
        attempt: int,
        delay_s: float,
    ) -> None:
        """Wait for backoff, then attempt to restart the system."""
        try:
            await asyncio.sleep(delay_s)

            system = self._systems.get(system_id)
            if system is None:
                self._logger.warning("restart_no_system_ref", system_id=system_id)
                return

            self._restart_attempts[system_id] = attempt + 1

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.SYSTEM_RESTARTING,
                data={
                    "system_id": system_id,
                    "attempt": attempt + 1,
                },
            ))

            # Attempt shutdown then re-initialize
            try:
                if hasattr(system, "shutdown"):
                    await system.shutdown()
            except Exception:
                pass  # Shutdown may fail on a broken system

            if hasattr(system, "initialize"):
                await system.initialize()
                self._logger.info(
                    "system_restarted",
                    system_id=system_id,
                    attempt=attempt + 1,
                )
            else:
                self._logger.warning(
                    "system_no_initialize",
                    system_id=system_id,
                )

        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._logger.error(
                "restart_failed",
                system_id=system_id,
                attempt=attempt + 1,
                error=str(exc),
            )

    # ─── Hot-Reload Restart ────────────────────────────────────────────

    async def restart_for_reload(self, system_id: str) -> bool:
        """
        Restart a single system for hot-reload (backward-compatible entry point).

        Delegates to ``restart_batch_for_reload({system_id})``.
        """
        results = await self.restart_batch_for_reload({system_id})
        return results.get(system_id, False)

    async def restart_batch_for_reload(
        self,
        system_ids: set[str],
    ) -> dict[str, bool]:
        """
        Restart a batch of systems plus their transitive dependents,
        respecting dependency order so no system is restarted while
        something it depends on is mid-cycle.

        Unlike ``handle_failure()``, this does NOT count as a failure
        attempt and does NOT apply degradation strategies.  It emits
        ``SYSTEM_RELOADING`` (not ``SYSTEM_RESTARTING``) so Thymos
        does not create failure incidents.

        Returns a dict of ``{system_id: success_bool}`` for every
        system that was restarted.
        """
        # Expand the set to include all transitive dependents that hold
        # stale references after the target systems re-initialize.
        full_set = self._expand_dependents(system_ids)
        # Filter to systems we actually manage.
        full_set = {sid for sid in full_set if sid in self._systems}
        if not full_set:
            return {}

        # Compute a safe restart order (leaves first for shutdown,
        # roots first for initialize).
        ordered = self._topological_sort(full_set)

        self._logger.info(
            "reload_batch_starting",
            requested=sorted(system_ids),
            full_set=sorted(full_set),
            shutdown_order=[sid for sid in ordered],
        )

        # Emit a single SYSTEM_RELOADING event for the whole batch.
        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.SYSTEM_RELOADING,
            data={
                "system_ids": sorted(full_set),
                "reason": "neuroplasticity_module_patch",
            },
        ))

        # Phase 1: shutdown in dependency-leaf-first order.
        # Leaves (systems that nothing else in the batch depends on) shut
        # down first so their dependents can still function until it's
        # their turn.
        for sid in ordered:
            system = self._systems[sid]
            try:
                if hasattr(system, "shutdown"):
                    await system.shutdown()
            except Exception as exc:
                self._logger.warning(
                    "reload_restart_shutdown_error",
                    system_id=sid,
                    error=str(exc),
                )

        # Phase 2: initialize in reverse order (roots/dependencies first)
        # so that when a dependent initializes, everything it depends on
        # is already alive.
        results: dict[str, bool] = {}
        for sid in reversed(ordered):
            system = self._systems[sid]
            try:
                if hasattr(system, "initialize"):
                    await system.initialize()
                    results[sid] = True
                    self._logger.info(
                        "reload_restart_complete",
                        system_id=sid,
                    )
                else:
                    results[sid] = False
                    self._logger.warning(
                        "reload_restart_no_initialize",
                        system_id=sid,
                    )
            except Exception as exc:
                results[sid] = False
                self._logger.error(
                    "reload_restart_initialize_failed",
                    system_id=sid,
                    error=str(exc),
                )

        return results

    # ─── Dependency Helpers ───────────────────────────────────────────

    def _expand_dependents(self, roots: set[str]) -> set[str]:
        """
        Return *roots* plus every system that transitively depends on
        any member of *roots*.  These are the systems that may hold
        stale internal state after the roots re-initialize.
        """
        expanded = set(roots)
        queue = list(roots)
        while queue:
            current = queue.pop()
            for dep in self._dependents.get(current, ()):
                if dep not in expanded:
                    expanded.add(dep)
                    queue.append(dep)
        return expanded

    def _topological_sort(self, system_ids: set[str]) -> list[str]:
        """
        Return *system_ids* in dependency-leaf-first order (suitable for
        shutdown). The reverse is the correct init order.

        Only edges within *system_ids* are considered. Cycles are broken
        by a visited guard - the first system encountered in a cycle wins.
        """
        # Build a subgraph restricted to the batch.
        in_degree: dict[str, int] = {sid: 0 for sid in system_ids}
        adj: dict[str, list[str]] = {sid: [] for sid in system_ids}
        for sid in system_ids:
            for dep in self._dependencies.get(sid, ()):
                if dep in system_ids:
                    # sid depends on dep → edge dep → sid (dep must init first)
                    adj[dep].append(sid)
                    in_degree[sid] += 1

        # Kahn's algorithm: nodes with in_degree 0 are leaves (no
        # in-batch dependency) and can be shut down first.
        queue = [sid for sid in system_ids if in_degree[sid] == 0]
        ordered: list[str] = []
        while queue:
            # Sort for determinism
            queue.sort()
            node = queue.pop(0)
            ordered.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # If there are remaining nodes they form a cycle. Append them
        # in alphabetical order (cycle guard - safe because all systems
        # in the cycle will be fully shut down before any re-init).
        remaining = system_ids - set(ordered)
        if remaining:
            self._logger.warning(
                "reload_dependency_cycle_detected",
                cycle_members=sorted(remaining),
            )
            ordered.extend(sorted(remaining))

        return ordered

    # ─── Level Computation ───────────────────────────────────────────

    def _update_level(self) -> None:
        """Recompute the overall degradation level from system health."""
        if self._health.is_safe_mode:
            self._level = DegradationLevel.SAFE_MODE
        elif len(self._health.failed_systems) > 0:
            self._level = DegradationLevel.DEGRADED
        else:
            self._level = DegradationLevel.NOMINAL

    @property
    def level(self) -> DegradationLevel:
        self._update_level()
        return self._level

    def get_strategy(self, system_id: str) -> DegradationStrategy | None:
        return _STRATEGIES.get(system_id)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "level": self._level.value,
            "restart_attempts": dict(self._restart_attempts),
            "active_restart_tasks": [
                sid for sid, t in self._restart_tasks.items()
                if not t.done()
            ],
        }
