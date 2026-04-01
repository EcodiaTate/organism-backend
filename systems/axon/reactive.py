"""
EcodiaOS -- Axon Reactive Adaptation Layer

Makes Axon a reactive participant in the organism's nervous system,
not just a passive executor. Subscribes to Synapse events and adapts
Axon's internal state in response:

  SYSTEM_DEGRADED    -> Pre-emptively open circuit breakers for degraded systems
  SYSTEM_RECOVERED   -> Close circuit breakers, restore normal operation
  METABOLIC_PRESSURE -> Tighten execution budget to conserve resources
  REVENUE_INJECTED   -> Relax budget if previously tightened
  SLEEP_INITIATED    -> Queue non-emergency intents for post-wake execution
  WAKE_ONSET         -> Drain the sleep queue and resume normal execution
  RHYTHM_STATE_CHANGED -> Adapt rate limits based on cognitive rhythm

This is what makes Axon a living motor cortex rather than a static
executor -- it adapts its behaviour based on the organism's state.
"""

from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING, Any

import structlog

from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from systems.axon.safety import BudgetTracker, CircuitBreaker, RateLimiter
    from systems.axon.types import ExecutionRequest

logger = structlog.get_logger()


class AxonReactiveAdapter:
    """
    Subscribes to Synapse events and adapts Axon's safety systems
    in real-time based on organism state.
    """

    def __init__(
        self,
        budget_tracker: BudgetTracker | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self._budget = budget_tracker
        self._circuit = circuit_breaker
        self._rate_limiter = rate_limiter
        self._logger = logger.bind(system="axon.reactive")

        # Track adaptations for observability
        self._adaptations: deque[dict[str, Any]] = deque(maxlen=100)
        self._budget_tightened: bool = False
        self._original_budget_max: int | None = None

        # Sleep queue: intents deferred during sleep
        self._sleep_queue: deque[ExecutionRequest] = deque(maxlen=50)
        self._is_sleeping: bool = False

        # Degraded system tracking
        self._degraded_systems: set[str] = set()

        # Active threat level from immune system
        self._active_threat_level: str | None = None

    def register_on_synapse(self, event_bus: Any) -> None:
        """Subscribe to all relevant Synapse events."""
        if event_bus is None or not hasattr(event_bus, "subscribe"):
            self._logger.warning("reactive_adapter_no_event_bus")
            return

        subscriptions = [
            (SynapseEventType.SYSTEM_DEGRADED, self._on_system_degraded),
            (SynapseEventType.SYSTEM_RECOVERED, self._on_system_recovered),
            (SynapseEventType.SYSTEM_FAILED, self._on_system_failed),
            (SynapseEventType.METABOLIC_PRESSURE, self._on_metabolic_pressure),
            (SynapseEventType.REVENUE_INJECTED, self._on_revenue_injected),
            (SynapseEventType.SLEEP_INITIATED, self._on_sleep_initiated),
            (SynapseEventType.WAKE_ONSET, self._on_wake_onset),
            (SynapseEventType.RHYTHM_STATE_CHANGED, self._on_rhythm_changed),
            (SynapseEventType.RESOURCE_PRESSURE, self._on_resource_pressure),
            (SynapseEventType.THREAT_DETECTED, self._on_threat_detected),
            (SynapseEventType.IMMUNE_CYCLE_COMPLETE, self._on_immune_cycle_complete),
        ]

        for event_type, handler in subscriptions:
            try:
                event_bus.subscribe(event_type, handler)
            except Exception as e:
                self._logger.debug(
                    "reactive_subscribe_failed",
                    event_type=event_type.value,
                    error=str(e),
                )

        self._logger.info(
            "reactive_adapter_registered",
            subscriptions=len(subscriptions),
        )

    async def _on_system_degraded(self, event: SynapseEvent | dict[str, Any]) -> None:
        """
        Pre-emptively tighten circuit breaker for degraded systems.

        When a system degrades, its executors are more likely to fail.
        Rather than waiting for 5 consecutive failures, we proactively
        move the circuit to HALF_OPEN to reduce load on the degraded system.
        """
        data = event.data if hasattr(event, "data") else event
        system_id = data.get("system_id", data.get("system", "unknown"))

        self._degraded_systems.add(system_id)
        self._record_adaptation("system_degraded", system_id=system_id)

        if self._circuit is not None and hasattr(self._circuit, "force_half_open"):
            self._circuit.force_half_open(system_id)
            self._logger.info(
                "circuit_breaker_preemptive_half_open",
                system_id=system_id,
            )

    async def _on_system_recovered(self, event: SynapseEvent | dict[str, Any]) -> None:
        """Close circuit breaker and restore normal operation for recovered systems."""
        data = event.data if hasattr(event, "data") else event
        system_id = data.get("system_id", data.get("system", "unknown"))

        self._degraded_systems.discard(system_id)
        self._record_adaptation("system_recovered", system_id=system_id)

        if self._circuit is not None and hasattr(self._circuit, "force_close"):
            self._circuit.force_close(system_id)
            self._logger.info(
                "circuit_breaker_restored",
                system_id=system_id,
            )

    async def _on_system_failed(self, event: SynapseEvent | dict[str, Any]) -> None:
        """Immediately open circuit breaker for failed systems."""
        data = event.data if hasattr(event, "data") else event
        system_id = data.get("system_id", data.get("system", "unknown"))

        self._degraded_systems.add(system_id)
        self._record_adaptation("system_failed", system_id=system_id)

        if self._circuit is not None and hasattr(self._circuit, "force_open"):
            self._circuit.force_open(system_id)
            self._logger.info(
                "circuit_breaker_force_opened",
                system_id=system_id,
            )

    async def _on_metabolic_pressure(self, event: SynapseEvent | dict[str, Any]) -> None:
        """
        Tighten execution budget when the organism is under metabolic pressure.

        Metabolic pressure means burn rate exceeds revenue. We reduce the
        max actions per cycle to conserve resources -- the organism should
        think more carefully about each action when resources are scarce.
        """
        data = event.data if hasattr(event, "data") else event
        pressure_level = data.get("pressure_level", data.get("severity", "moderate"))

        if self._budget is None or not hasattr(self._budget, "_budget"):
            return

        budget = self._budget._budget
        if not hasattr(budget, "max_actions_per_cycle"):
            return

        if not self._budget_tightened:
            self._original_budget_max = budget.max_actions_per_cycle

        # Scale reduction based on pressure severity
        if pressure_level in ("critical", "severe"):
            reduction = 0.4  # 60% reduction
        elif pressure_level in ("high", "moderate"):
            reduction = 0.25  # 25% reduction
        else:
            reduction = 0.15  # 15% reduction

        new_max = max(1, int(budget.max_actions_per_cycle * (1 - reduction)))
        budget.max_actions_per_cycle = new_max
        self._budget_tightened = True

        self._record_adaptation(
            "budget_tightened",
            pressure_level=pressure_level,
            new_max=new_max,
        )
        self._logger.info(
            "budget_tightened_metabolic_pressure",
            pressure_level=pressure_level,
            new_max=new_max,
        )

    async def _on_revenue_injected(self, event: SynapseEvent | dict[str, Any]) -> None:
        """Restore budget if previously tightened due to metabolic pressure."""
        if not self._budget_tightened or self._original_budget_max is None:
            return

        if self._budget is None or not hasattr(self._budget, "_budget"):
            return

        budget = self._budget._budget
        if hasattr(budget, "max_actions_per_cycle"):
            budget.max_actions_per_cycle = self._original_budget_max
            self._budget_tightened = False

            self._record_adaptation(
                "budget_restored",
                restored_max=self._original_budget_max,
            )
            self._logger.info(
                "budget_restored_revenue_injected",
                restored_max=self._original_budget_max,
            )

    async def _on_sleep_initiated(self, event: SynapseEvent | dict[str, Any]) -> None:
        """Mark organism as sleeping so Axon queues non-emergency intents."""
        self._is_sleeping = True
        self._record_adaptation("sleep_initiated")
        self._logger.info("axon_sleep_mode_entered")

    async def _on_wake_onset(self, event: SynapseEvent | dict[str, Any]) -> None:
        """Mark organism as awake and return queued intents for processing."""
        self._is_sleeping = False
        queued_count = len(self._sleep_queue)

        self._record_adaptation(
            "wake_onset",
            queued_intents=queued_count,
        )
        self._logger.info(
            "axon_sleep_mode_exited",
            queued_intents=queued_count,
        )

    async def _on_rhythm_changed(self, event: SynapseEvent | dict[str, Any]) -> None:
        """
        Adapt rate limits based on cognitive rhythm state.

        FLOW state  -> normal rate limits (1.0)
        RUSH state  -> tighten (0.7) - organism is overactive
        STALL state -> loosen (1.3) - allow recovery actions
        DRIFT state -> tighten (0.6) - reduce noise
        """
        data = event.data if hasattr(event, "data") else event
        rhythm = data.get("rhythm_state", data.get("state", "unknown"))

        self._record_adaptation("rhythm_changed", rhythm=rhythm)

        if self._rate_limiter is None:
            return

        rhythm_multipliers: dict[str, float] = {
            "FLOW": 1.0,
            "RUSH": 0.7,
            "STALL": 1.3,
            "DRIFT": 0.6,
        }
        multiplier = rhythm_multipliers.get(rhythm.upper(), 1.0)

        if hasattr(self._rate_limiter, "set_global_multiplier"):
            self._rate_limiter.set_global_multiplier(multiplier)
            self._logger.info(
                "rhythm_rate_limit_adapted",
                rhythm=rhythm,
                multiplier=multiplier,
            )

    async def _on_threat_detected(self, event: SynapseEvent | dict[str, Any]) -> None:
        """
        Tighten rate limits when the immune system detects a threat.

        Thymos T3+ incidents signal active attacks or cascading failures.
        Financial executors get the tightest restriction; others moderate.
        """
        data = event.data if hasattr(event, "data") else event
        severity = data.get("severity", data.get("tier", "moderate"))

        self._record_adaptation("threat_detected", severity=severity)
        self._active_threat_level = severity

        if self._rate_limiter is None or not hasattr(
            self._rate_limiter, "set_multiplier"
        ):
            return

        # Scale tightening based on threat severity
        if severity in ("critical", "T4", "T5", "T6"):
            financial_factor = 0.2  # 80% reduction for financial ops
            general_factor = 0.5   # 50% reduction for everything else
        elif severity in ("high", "T3"):
            financial_factor = 0.4
            general_factor = 0.7
        else:
            financial_factor = 0.7
            general_factor = 0.9

        for action_type in ("wallet_transfer", "defi_yield", "phantom_liquidity"):
            self._rate_limiter.set_multiplier(action_type, financial_factor)
        self._rate_limiter.set_global_multiplier(general_factor)

        self._logger.warning(
            "threat_rate_limits_tightened",
            severity=severity,
            financial_factor=financial_factor,
            general_factor=general_factor,
        )

    async def _on_immune_cycle_complete(
        self, event: SynapseEvent | dict[str, Any]
    ) -> None:
        """
        Relax rate limits when the immune cycle completes without active threats.

        If the cycle reports all-clear, restore normal rate limiting.
        """
        data = event.data if hasattr(event, "data") else event
        active_incidents = data.get("active_incidents", data.get("open_count", 0))

        self._record_adaptation(
            "immune_cycle_complete", active_incidents=active_incidents
        )

        if active_incidents == 0 and self._rate_limiter is not None:
            if hasattr(self._rate_limiter, "reset_global_multiplier"):
                self._rate_limiter.reset_global_multiplier()
            # Clear financial-specific overrides
            if hasattr(self._rate_limiter, "clear_multiplier"):
                for at in ("wallet_transfer", "defi_yield", "phantom_liquidity"):
                    self._rate_limiter.clear_multiplier(at)

            self._active_threat_level = None
            self._logger.info("threat_rate_limits_restored")

    async def _on_resource_pressure(self, event: SynapseEvent | dict[str, Any]) -> None:
        """Reduce concurrency limit when resources are pressured."""
        if self._budget is None or not hasattr(self._budget, "_budget"):
            return

        budget = self._budget._budget
        if hasattr(budget, "max_concurrent_executions"):
            new_concurrent = max(1, budget.max_concurrent_executions - 1)
            budget.max_concurrent_executions = new_concurrent
            self._record_adaptation(
                "concurrency_reduced",
                new_concurrent=new_concurrent,
            )

    def queue_for_wake(self, request: ExecutionRequest) -> bool:
        """Queue a non-emergency intent for post-wake execution."""
        if not self._is_sleeping:
            return False
        self._sleep_queue.append(request)
        return True

    def drain_sleep_queue(self) -> list[ExecutionRequest]:
        """Drain all queued intents after wake."""
        items = list(self._sleep_queue)
        self._sleep_queue.clear()
        return items

    def _record_adaptation(self, event_type: str, **kwargs: Any) -> None:
        self._adaptations.append({
            "event": event_type,
            "timestamp": time.monotonic(),
            **kwargs,
        })

    @property
    def is_sleeping(self) -> bool:
        return self._is_sleeping

    @property
    def degraded_systems(self) -> set[str]:
        return set(self._degraded_systems)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "is_sleeping": self._is_sleeping,
            "budget_tightened": self._budget_tightened,
            "degraded_systems": list(self._degraded_systems),
            "sleep_queue_size": len(self._sleep_queue),
            "total_adaptations": len(self._adaptations),
            "recent_adaptations": list(self._adaptations)[-5:],
        }
