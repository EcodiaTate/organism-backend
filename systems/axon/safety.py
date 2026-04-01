"""
EcodiaOS - Axon Safety Mechanisms

Three interlocking safety systems protect against runaway action loops,
cascading failures, and excessive external calls:

1. RateLimiter - sliding-window counters per executor type
   Prevents any single executor from flooding an external service or
   spamming notifications. Counters are in-memory (per-process).
   For distributed deployments, back this with Redis (future Synapse work).

2. CircuitBreaker - per-executor open/half-open/closed state machine
   If an executor repeatedly fails, it is disabled for a recovery window.
   After recovery_timeout_s, a single probe execution is allowed (half-open).
   Success → closed (normal). Failure → re-opens. Prevents cascading failures
   from a degraded external service.

3. BudgetTracker - per-cycle execution budget enforcement
   Limits the total number and type of actions EOS can take in a single
   cognitive cycle. This is the non-negotiable safety valve - it exists
   to prevent EOS from acting obsessively or exhausting shared resources.
   Budget limits come from AxonConfig and cannot be raised at runtime.

These are defence-in-depth. Nova's EFE evaluation and Equor's constitutional
review are the first two lines. The safety mechanisms are the third.
"""

from __future__ import annotations

import contextlib
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.types import CircuitState, CircuitStatus, ExecutionBudget, RateLimit

if TYPE_CHECKING:
    from config import AxonConfig

logger = structlog.get_logger()


# ─── Rate Limiter ─────────────────────────────────────────────────


class RateLimiter:
    """
    Sliding-window rate limiter with optional Redis backing.

    Each action type gets its own window. In single-process mode, counters
    are in-memory (deque of timestamps). In distributed mode, counters
    are backed by Redis sorted sets for cross-process coordination.

    The Redis path is best-effort: if Redis is unavailable, falls back
    to in-memory counters. This ensures Axon never blocks on a Redis
    failure -- safety degrades gracefully to single-process accuracy.
    """

    def __init__(self, redis_client: Any = None, key_prefix: str = "axon:rl") -> None:
        # In-memory counters (always present as fallback)
        self._windows: dict[str, deque[float]] = defaultdict(deque)
        self._redis = redis_client
        self._key_prefix = key_prefix
        self._redis_failures: int = 0
        self._logger = logger.bind(system="axon.rate_limiter")
        # Adaptive multipliers: action_type -> factor (1.0 = normal)
        # < 1.0 = tighter (fewer allowed), > 1.0 = looser (more allowed)
        self._multipliers: dict[str, float] = {}
        self._global_multiplier: float = 1.0

    def check(self, action_type: str, rate_limit: RateLimit) -> bool:
        """
        Return True if the action is within its rate limit.

        Checks Redis first if available, falls back to in-memory.
        Does NOT record the call -- call record() after a successful check.

        Adaptive multipliers scale the effective max_calls:
          multiplier < 1.0 → fewer calls allowed (tighter, e.g. during incidents)
          multiplier > 1.0 → more calls allowed (looser, e.g. calm periods)
        """
        # Apply per-action and global multipliers to the limit
        multiplier = self._multipliers.get(action_type, 1.0) * self._global_multiplier
        effective_max = max(1, int(rate_limit.max_calls * multiplier))

        # Try Redis-backed check first
        if self._redis is not None:
            try:
                count = self._redis_count(action_type, rate_limit.window_seconds)
                if count is not None:
                    allowed = count < effective_max
                    if not allowed:
                        self._logger.warning(
                            "rate_limit_exceeded",
                            action_type=action_type,
                            current_count=count,
                            effective_max=effective_max,
                            window_seconds=rate_limit.window_seconds,
                            source="redis",
                        )
                    return allowed
            except Exception as e:
                self._redis_failures += 1
                if self._redis_failures <= 3:
                    self._logger.warning("redis_rate_limit_fallback", error=str(e))

        # In-memory fallback
        window = self._windows[action_type]
        now = time.monotonic()
        cutoff = now - rate_limit.window_seconds

        while window and window[0] < cutoff:
            window.popleft()

        allowed = len(window) < effective_max
        if not allowed:
            self._logger.warning(
                "rate_limit_exceeded",
                action_type=action_type,
                current_count=len(window),
                effective_max=effective_max,
                window_seconds=rate_limit.window_seconds,
                source="memory",
            )
        return allowed

    def record(self, action_type: str) -> None:
        """Record a call for rate-limit accounting (both in-memory and Redis)."""
        now = time.monotonic()
        self._windows[action_type].append(now)

        # Also record in Redis if available
        if self._redis is not None:
            with contextlib.suppress(Exception):
                self._redis_record(action_type, now)

    def reset(self, action_type: str) -> None:
        """Reset the window for a specific action type."""
        self._windows[action_type].clear()
        if self._redis is not None:
            try:
                key = f"{self._key_prefix}:{action_type}"
                self._redis.delete(key)
            except Exception:
                pass

    def current_count(self, action_type: str, window_seconds: float) -> int:
        """Return the number of calls within the last window_seconds."""
        # Try Redis first
        if self._redis is not None:
            try:
                count = self._redis_count(action_type, window_seconds)
                if count is not None:
                    return count
            except Exception:
                pass

        # In-memory fallback
        window = self._windows[action_type]
        cutoff = time.monotonic() - window_seconds
        return sum(1 for ts in window if ts >= cutoff)

    def remaining(self, action_type: str, rate_limit: "RateLimit | None" = None) -> int | None:
        """
        Return the number of calls remaining in the current window for action_type.

        Args:
            action_type: The action type to check
            rate_limit: Optional RateLimit definition. If None, returns None.

        Returns:
            Number of calls remaining, or None if rate_limit is not provided.
        """
        if rate_limit is None:
            return None

        current = self.current_count(action_type, rate_limit.window_seconds)
        # Apply multipliers to get the effective max
        multiplier = self._multipliers.get(action_type, 1.0) * self._global_multiplier
        effective_max = max(1, int(rate_limit.max_calls * multiplier))

        return max(0, effective_max - current)

    def _redis_count(self, action_type: str, window_seconds: float) -> int | None:
        """Count entries in Redis sorted set within the time window."""
        key = f"{self._key_prefix}:{action_type}"
        now = time.time()
        cutoff = now - window_seconds

        # ZRANGEBYSCORE to count entries within the window
        try:
            # Use ZCOUNT for efficiency (no data transfer)
            count = self._redis.zcount(key, cutoff, "+inf")
            return int(count)
        except Exception:
            return None

    def set_multiplier(self, action_type: str, factor: float) -> None:
        """Set an adaptive multiplier for a specific action type."""
        self._multipliers[action_type] = max(0.1, factor)
        self._logger.debug(
            "rate_limit_multiplier_set",
            action_type=action_type,
            factor=self._multipliers[action_type],
        )

    def clear_multiplier(self, action_type: str) -> None:
        """Remove the adaptive multiplier for an action type."""
        self._multipliers.pop(action_type, None)

    def set_global_multiplier(self, factor: float) -> None:
        """Set a global multiplier applied to ALL action types."""
        self._global_multiplier = max(0.1, factor)
        self._logger.info(
            "rate_limit_global_multiplier_set",
            factor=self._global_multiplier,
        )

    def reset_global_multiplier(self) -> None:
        """Reset the global multiplier to 1.0 (normal)."""
        self._global_multiplier = 1.0

    def _redis_record(self, action_type: str, monotonic_ts: float) -> None:
        """Add a timestamp to the Redis sorted set."""
        key = f"{self._key_prefix}:{action_type}"
        now = time.time()  # Use wall-clock for Redis (cross-process)

        pipe = self._redis.pipeline(transaction=False)
        pipe.zadd(key, {str(now): now})
        # Expire old entries (2x the largest expected window)
        pipe.zremrangebyscore(key, "-inf", now - 7200)
        # Auto-expire the key after 2 hours
        pipe.expire(key, 7200)
        pipe.execute()


# ─── Circuit Breaker ──────────────────────────────────────────────


class CircuitBreaker:
    """
    Per-executor circuit breaker using a three-state finite state machine.

    States:
      CLOSED - normal operation; all executions allowed
      OPEN - tripped; all executions blocked for recovery_timeout_s
      HALF_OPEN - probing; allows exactly half_open_max_calls attempts

    Transitions:
      CLOSED → OPEN: failure_threshold consecutive failures
      OPEN → HALF_OPEN: after recovery_timeout_s
      HALF_OPEN → CLOSED: probe succeeds
      HALF_OPEN → OPEN: probe fails

    Optional Redis persistence and Synapse event emission on state transitions.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: int = 300,
        half_open_max_calls: int = 1,
        redis_client: Any = None,
        event_bus: Any = None,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.half_open_max_calls = half_open_max_calls
        self._states: dict[str, CircuitState] = {}
        self._redis = redis_client
        self._event_bus = event_bus
        self._logger = logger.bind(system="axon.circuit_breaker")

    def can_execute(self, action_type: str) -> bool:
        """Return True if the circuit is closed or in a half-open probe window."""
        state = self._states.get(action_type)
        if state is None:
            return True

        if state.status == CircuitStatus.CLOSED:
            return True

        if state.status == CircuitStatus.OPEN:
            elapsed = time.monotonic() - state.tripped_at
            if elapsed >= self.recovery_timeout_s:
                # Transition to half-open for probing
                old_status = state.status
                state.status = CircuitStatus.HALF_OPEN
                state.half_open_calls = 0
                self._logger.info(
                    "circuit_half_open",
                    action_type=action_type,
                    elapsed_s=int(elapsed),
                )
                self._fire_transition(action_type, old_status, CircuitStatus.HALF_OPEN)
                return True
            return False

        if state.status == CircuitStatus.HALF_OPEN:
            if state.half_open_calls < self.half_open_max_calls:
                state.half_open_calls += 1
                return True
            return False

        return False  # Unreachable, but safe

    def record_result(self, action_type: str, success: bool) -> None:
        """Update circuit state after an execution attempt."""
        state = self._states.setdefault(action_type, CircuitState())

        if success:
            if state.status == CircuitStatus.HALF_OPEN:
                old_status = state.status
                state.status = CircuitStatus.CLOSED
                state.consecutive_failures = 0
                self._logger.info("circuit_closed", action_type=action_type)
                self._fire_transition(action_type, old_status, CircuitStatus.CLOSED)
            elif state.status == CircuitStatus.CLOSED:
                state.consecutive_failures = 0
        else:
            state.consecutive_failures += 1
            if (
                state.consecutive_failures >= self.failure_threshold
                and state.status != CircuitStatus.OPEN
            ):
                old_status = state.status
                state.status = CircuitStatus.OPEN
                state.tripped_at = time.monotonic()
                self._logger.warning(
                    "circuit_tripped",
                    action_type=action_type,
                    consecutive_failures=state.consecutive_failures,
                )
                self._fire_transition(action_type, old_status, CircuitStatus.OPEN)

    def status(self, action_type: str) -> CircuitStatus:
        """Return current circuit status for an executor."""
        state = self._states.get(action_type)
        return state.status if state else CircuitStatus.CLOSED

    def reset(self, action_type: str) -> None:
        """Manually reset a circuit (governance action)."""
        if action_type in self._states:
            del self._states[action_type]
            self._logger.info("circuit_manually_reset", action_type=action_type)

    def force_half_open(self, action_type: str) -> None:
        """Force a circuit into HALF_OPEN state (reactive degradation response)."""
        state = self._states.setdefault(action_type, CircuitState())
        if state.status == CircuitStatus.CLOSED:
            old_status = state.status
            state.status = CircuitStatus.HALF_OPEN
            state.half_open_calls = 0
            self._logger.info("circuit_force_half_open", action_type=action_type)
            self._fire_transition(action_type, old_status, CircuitStatus.HALF_OPEN)

    def force_open(self, action_type: str) -> None:
        """Force a circuit into OPEN state (system failure response)."""
        state = self._states.setdefault(action_type, CircuitState())
        old_status = state.status
        state.status = CircuitStatus.OPEN
        state.tripped_at = time.monotonic()
        self._logger.info("circuit_force_opened", action_type=action_type)
        if old_status != CircuitStatus.OPEN:
            self._fire_transition(action_type, old_status, CircuitStatus.OPEN)

    def force_close(self, action_type: str) -> None:
        """Force a circuit into CLOSED state (system recovery response)."""
        state = self._states.get(action_type)
        if state is not None:
            old_status = state.status
            state.status = CircuitStatus.CLOSED
            state.consecutive_failures = 0
            self._logger.info("circuit_force_closed", action_type=action_type)
            if old_status != CircuitStatus.CLOSED:
                self._fire_transition(action_type, old_status, CircuitStatus.CLOSED)

    def _fire_transition(
        self, action_type: str, old_status: CircuitStatus, new_status: CircuitStatus
    ) -> None:
        """Fire-and-forget: persist state + emit event on transition."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # No event loop - skip async operations

        loop.create_task(
            self._persist_state(action_type),
            name=f"cb_persist_{action_type}",
        )
        loop.create_task(
            self._emit_state_change(action_type, old_status, new_status),
            name=f"cb_emit_{action_type}",
        )

    def trip_count(self) -> int:
        """Total circuits currently open."""
        return sum(
            1 for s in self._states.values() if s.status == CircuitStatus.OPEN
        )

    # ── Redis persistence ─────────────────────────────────────────

    async def _persist_state(self, action_type: str) -> None:
        """Save circuit breaker state to Redis."""
        if self._redis is None:
            return
        state = self._states.get(action_type)
        if state is None:
            return
        key = f"axon:circuit_breaker:{action_type}"
        try:
            import json
            payload = json.dumps({
                "status": state.status.value,
                "consecutive_failures": state.consecutive_failures,
                "tripped_at": state.tripped_at,
                "half_open_calls": state.half_open_calls,
            })
            self._redis.set(key, payload, ex=self.recovery_timeout_s * 3)
        except Exception as exc:
            self._logger.debug("circuit_breaker_persist_failed", error=str(exc))

    async def _load_state(self, action_type: str) -> None:
        """Load circuit breaker state from Redis on startup."""
        if self._redis is None:
            return
        key = f"axon:circuit_breaker:{action_type}"
        try:
            import json
            raw = self._redis.get(key)
            if raw is None:
                return
            data = json.loads(raw)
            self._states[action_type] = CircuitState(
                status=CircuitStatus(data["status"]),
                consecutive_failures=data.get("consecutive_failures", 0),
                tripped_at=data.get("tripped_at", 0.0),
                half_open_calls=data.get("half_open_calls", 0),
            )
            self._logger.info(
                "circuit_breaker_state_restored",
                action_type=action_type,
                status=data["status"],
            )
        except Exception as exc:
            self._logger.debug("circuit_breaker_load_failed", error=str(exc))

    async def load_all_states(self) -> None:
        """Load all persisted circuit breaker states from Redis on startup."""
        if self._redis is None:
            return
        try:
            keys = self._redis.keys("axon:circuit_breaker:*")
            for key in keys:
                action_type = key.split(":")[-1] if isinstance(key, str) else key.decode().split(":")[-1]
                await self._load_state(action_type)
        except Exception as exc:
            self._logger.debug("circuit_breaker_load_all_failed", error=str(exc))

    async def _emit_state_change(
        self, action_type: str, old_status: CircuitStatus, new_status: CircuitStatus
    ) -> None:
        """Emit CIRCUIT_BREAKER_STATE_CHANGED event on transitions."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.CIRCUIT_BREAKER_STATE_CHANGED,
                source_system="axon",
                data={
                    "action_type": action_type,
                    "old_status": old_status.value,
                    "new_status": new_status.value,
                    "failure_threshold": self.failure_threshold,
                    "recovery_timeout_s": self.recovery_timeout_s,
                },
            ))
        except Exception as exc:
            self._logger.debug("circuit_breaker_event_emit_failed", error=str(exc))


# ─── Action Budget (mutable runtime limits) ───────────────────────


from dataclasses import dataclass, field as dc_field


@dataclass
class TemporaryExpansion:
    """
    A single approved budget expansion for one field.

    Created by ActionBudget.apply_expansion() when Equor approves an
    ACTION_BUDGET_EXPANSION_REQUEST.  Automatically reverted when
    cycles_remaining reaches zero.
    """

    field: str
    original_value: int
    expanded_value: int
    cycles_remaining: int
    approved_by: str = "equor"


@dataclass
class ActionBudget:
    """
    Mutable runtime action limits for AxonService.

    Unlike the immutable ExecutionBudget dataclass, ActionBudget holds the
    *live* limits that the organism actually enforces each cycle.  Equor can
    approve temporary expansions via ACTION_BUDGET_EXPANSION_RESPONSE; Evo can
    propose permanent baseline adjustments via EVO_ADJUST_BUDGET.

    The three Equor-negotiable fields are:
      max_actions_per_cycle     - hard cap per theta cycle (default 5)
      max_concurrent_executions - parallel execution slots (default 3)
      max_api_calls_per_minute  - external call rate cap (default 30)

    Safe upper bounds enforced by Equor:
      max_actions_per_cycle     ≤ 20
      max_concurrent_executions ≤ 10
      max_api_calls_per_minute  ≤ 120
    """

    # Baseline defaults (Evo-tunable over generations)
    max_actions_per_cycle: int = 5
    max_concurrent_executions: int = 3
    max_api_calls_per_minute: int = 30

    # Active temporary expansions keyed by field name.
    # Only one expansion per field is active at a time.
    temporary_expansions: dict[str, TemporaryExpansion] = dc_field(
        default_factory=dict
    )

    def effective_max_actions(self) -> int:
        """Return current max_actions_per_cycle (baseline or expanded)."""
        exp = self.temporary_expansions.get("max_actions_per_cycle")
        return exp.expanded_value if exp else self.max_actions_per_cycle

    def effective_max_concurrent(self) -> int:
        """Return current max_concurrent_executions (baseline or expanded)."""
        exp = self.temporary_expansions.get("max_concurrent_executions")
        return exp.expanded_value if exp else self.max_concurrent_executions

    def effective_max_api_calls(self) -> int:
        """Return current max_api_calls_per_minute (baseline or expanded)."""
        exp = self.temporary_expansions.get("max_api_calls_per_minute")
        return exp.expanded_value if exp else self.max_api_calls_per_minute

    def apply_expansion(
        self,
        field: str,
        approved_value: int,
        duration_cycles: int,
        approved_by: str = "equor",
    ) -> None:
        """
        Apply an Equor-approved temporary expansion for the given field.

        Overwrites any existing expansion for the same field.
        Raises ValueError if the field is not an expandable budget field.
        """
        _EXPANDABLE = {
            "max_actions_per_cycle",
            "max_concurrent_executions",
            "max_api_calls_per_minute",
        }
        if field not in _EXPANDABLE:
            raise ValueError(f"ActionBudget: '{field}' is not an expandable field")
        original = getattr(self, field)
        self.temporary_expansions[field] = TemporaryExpansion(
            field=field,
            original_value=original,
            expanded_value=approved_value,
            cycles_remaining=max(1, duration_cycles),
            approved_by=approved_by,
        )

    def check_expirations(self) -> list[str]:
        """
        Tick down all active expansions; revert any that have expired.

        Called once per theta cycle by BudgetTracker.  Returns list of field
        names whose expansions expired this call.
        """
        expired: list[str] = []
        for field, exp in list(self.temporary_expansions.items()):
            exp.cycles_remaining -= 1
            if exp.cycles_remaining <= 0:
                expired.append(field)
                del self.temporary_expansions[field]
        return expired

    def to_snapshot(self) -> dict:
        """Return a snapshot dict for logging / AXON_CAPABILITY_SNAPSHOT."""
        return {
            "max_actions_per_cycle": self.max_actions_per_cycle,
            "max_concurrent_executions": self.max_concurrent_executions,
            "max_api_calls_per_minute": self.max_api_calls_per_minute,
            "effective_max_actions": self.effective_max_actions(),
            "effective_max_concurrent": self.effective_max_concurrent(),
            "effective_max_api_calls": self.effective_max_api_calls(),
            "active_expansions": {
                k: {
                    "expanded_value": v.expanded_value,
                    "cycles_remaining": v.cycles_remaining,
                    "approved_by": v.approved_by,
                }
                for k, v in self.temporary_expansions.items()
            },
        }


# ─── Budget Tracker ───────────────────────────────────────────────


class BudgetTracker:
    """
    Per-cycle execution budget enforcement.

    The budget is replenished at the start of each cognitive cycle by calling
    begin_cycle(). Checks are cumulative within the cycle - once a limit is
    hit, it blocks for the remainder of the cycle.

    Sub-limits (API calls per minute, notifications per hour) use sliding-window
    deques so they enforce across cycle boundaries - a burst of API calls in cycle N
    still counts against the limit in cycle N+1 (Spec §5.3).

    Limits are read from ActionBudget each check, so Equor-approved temporary
    expansions and Evo baseline adjustments take effect immediately.
    """

    # Action types that count as "api_calls" for the per-minute sub-limit
    _API_CALL_TYPES: frozenset[str] = frozenset({
        "call_api", "webhook_trigger", "defi_yield", "wallet_transfer",
        "phantom_liquidity", "query_memory", "axon.solve_bounty",
    })
    # Action types that count as "notifications" for the per-hour sub-limit
    _NOTIFICATION_TYPES: frozenset[str] = frozenset({
        "send_notification", "send_email", "post_message", "respond_text",
    })

    def __init__(self, config: AxonConfig) -> None:
        self._budget = ExecutionBudget(
            max_actions_per_cycle=config.max_actions_per_cycle,
            max_api_calls_per_minute=config.max_api_calls_per_minute,
            max_notifications_per_hour=config.max_notifications_per_hour,
            max_concurrent_executions=config.max_concurrent_executions,
            total_timeout_per_cycle_ms=config.total_timeout_per_cycle_ms,
        )
        # Mutable runtime limits - Equor can expand, Evo can tune baseline
        self.action_budget = ActionBudget(
            max_actions_per_cycle=config.max_actions_per_cycle,
            max_concurrent_executions=config.max_concurrent_executions,
            max_api_calls_per_minute=config.max_api_calls_per_minute,
        )
        self._logger = logger.bind(system="axon.budget_tracker")
        # Sliding-window deques for sub-limit enforcement (wall-clock timestamps)
        self._api_call_timestamps: deque[float] = deque()
        self._notification_timestamps: deque[float] = deque()
        self._reset_counters()

    def _reset_counters(self) -> None:
        self._actions_this_cycle: int = 0
        self._concurrent_executions: int = 0
        self._cycle_start: float = time.monotonic()
        # Per-intent budget tracking: intent_id -> actions consumed
        self._intent_actions: dict[str, int] = {}

    def begin_cycle(self) -> None:
        """Called at the start of each cognitive cycle to reset per-cycle counters."""
        self._reset_counters()

    def can_execute(self) -> tuple[bool, str]:
        """
        Check if the budget allows another execution.
        Returns (allowed, reason) - reason is empty string if allowed.

        Auto-resets the cycle if total_timeout_per_cycle_ms has elapsed since
        the last begin_cycle() call. This handles the case where Synapse has not
        yet called begin_cycle() (e.g. first few ticks after startup) so the
        budget does not permanently block after 30 seconds.
        """
        elapsed_ms = int((time.monotonic() - self._cycle_start) * 1000)
        if elapsed_ms >= self._budget.total_timeout_per_cycle_ms:
            # Cycle window expired - auto-begin a new cycle so we don't block
            # permanently when Synapse's begin_cycle() wiring is missing.
            self._reset_counters()
            elapsed_ms = 0

        max_actions = self.action_budget.effective_max_actions()
        max_concurrent = self.action_budget.effective_max_concurrent()
        if self._actions_this_cycle >= max_actions:
            return False, (
                f"Budget: max actions per cycle reached "
                f"({max_actions})"
            )
        if self._concurrent_executions >= max_concurrent:
            return False, (
                f"Budget: max concurrent executions reached "
                f"({max_concurrent})"
            )
        return True, ""

    def can_execute_action_type(self, action_type: str) -> tuple[bool, str]:
        """
        Check sub-limits (API calls/min, notifications/hr) for a specific action type.

        Called by the pipeline after the global can_execute() passes but before
        execution starts, to enforce category-level rate caps (Spec §5.3).

        Returns (allowed, reason).
        """
        now = time.monotonic()

        if action_type in self._API_CALL_TYPES:
            # Prune timestamps older than 60s
            cutoff = now - 60.0
            while self._api_call_timestamps and self._api_call_timestamps[0] < cutoff:
                self._api_call_timestamps.popleft()
            count = len(self._api_call_timestamps)
            max_api = self.action_budget.effective_max_api_calls()
            if count >= max_api:
                return False, (
                    f"Budget: max API calls per minute reached "
                    f"({count}/{max_api})"
                )

        if action_type in self._NOTIFICATION_TYPES:
            # Prune timestamps older than 3600s
            cutoff = now - 3600.0
            while (
                self._notification_timestamps
                and self._notification_timestamps[0] < cutoff
            ):
                self._notification_timestamps.popleft()
            count = len(self._notification_timestamps)
            if count >= self._budget.max_notifications_per_hour:
                return False, (
                    f"Budget: max notifications per hour reached "
                    f"({count}/{self._budget.max_notifications_per_hour})"
                )

        return True, ""

    def record_action_type(self, action_type: str) -> None:
        """
        Record that an action of the given type completed (for sub-limit tracking).

        Called by the pipeline after a step executes successfully, so the
        sliding windows only count actions that actually ran.
        """
        now = time.monotonic()
        if action_type in self._API_CALL_TYPES:
            self._api_call_timestamps.append(now)
        if action_type in self._NOTIFICATION_TYPES:
            self._notification_timestamps.append(now)

    def can_execute_intent(
        self, intent_id: str, step_count: int, max_steps: int = 0
    ) -> tuple[bool, str]:
        """
        Check if a specific intent has budget remaining.

        Args:
            intent_id: The intent being executed.
            step_count: Number of steps this intent wants to execute.
            max_steps: Maximum steps allowed for this intent (0 = unlimited).

        Returns:
            (allowed, reason) - reason is empty string if allowed.
        """
        if max_steps <= 0:
            return True, ""
        consumed = self._intent_actions.get(intent_id, 0)
        if consumed + step_count > max_steps:
            return False, (
                f"Intent {intent_id} budget exhausted: "
                f"{consumed}/{max_steps} steps used, "
                f"requested {step_count} more"
            )
        return True, ""

    def begin_execution(self, intent_id: str = "") -> None:
        """Called when an execution starts (tracks concurrency + per-intent)."""
        self._concurrent_executions += 1
        self._actions_this_cycle += 1
        if intent_id:
            self._intent_actions[intent_id] = (
                self._intent_actions.get(intent_id, 0) + 1
            )

    def end_execution(self) -> None:
        """Called when an execution completes or fails (releases concurrency slot)."""
        self._concurrent_executions = max(0, self._concurrent_executions - 1)

    @property
    def utilisation(self) -> float:
        """Fraction of cycle action budget consumed (0.0 to 1.0+)."""
        return self._actions_this_cycle / max(
            1, self.action_budget.effective_max_actions()
        )

    @property
    def budget(self) -> ExecutionBudget:
        return self._budget

    def tick_expansions(self) -> list[str]:
        """
        Decrement active temporary expansions; return names of fields that expired.

        Called once per theta cycle (from AxonService._on_theta_cycle_complete).
        """
        expired = self.action_budget.check_expirations()
        for field in expired:
            self._logger.info(
                "budget_expansion_expired",
                field=field,
                baseline=getattr(self.action_budget, field, "?"),
            )
        return expired
