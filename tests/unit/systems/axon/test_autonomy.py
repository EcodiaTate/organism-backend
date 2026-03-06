"""
Unit tests for Axon autonomy features.

Tests:
  - ExecutorRegistry: capabilities(), find_by_autonomy(), deregister()
  - RateLimiter: adaptive multipliers (global + per-action)
  - BudgetTracker: per-intent execution budgets
  - AxonReactiveAdapter: threat/immune rate limit adaptation
  - AxonIntrospector: degradation detection for self-healing
"""

from __future__ import annotations

import asyncio

import pytest

from config import AxonConfig
from systems.axon.introspection import AxonIntrospector, ExecutorProfile
from systems.axon.reactive import AxonReactiveAdapter
from systems.axon.registry import ExecutorRegistry
from systems.axon.safety import BudgetTracker, CircuitBreaker, RateLimiter
from systems.axon.types import RateLimit, ValidationResult, ExecutionResult, ExecutionContext
from systems.axon.executor import Executor


# ─── Fixtures ─────────────────────────────────────────────────────


class _TestExecutor(Executor):
    action_type = "test_exec"
    description = "Test executor"
    required_autonomy = 2
    reversible = True
    max_duration_ms = 500
    rate_limit = RateLimit.per_minute(10)

    async def validate_params(self, params: dict) -> ValidationResult:
        return ValidationResult.ok()

    async def execute(self, params: dict, context: ExecutionContext) -> ExecutionResult:
        return ExecutionResult(success=True, data={})


class _HighAutonomyExecutor(_TestExecutor):
    action_type = "high_auto"
    required_autonomy = 5
    reversible = False
    rate_limit = RateLimit.per_minute(3)


class _LowAutonomyExecutor(_TestExecutor):
    action_type = "low_auto"
    required_autonomy = 1
    rate_limit = RateLimit.per_minute(20)


def _make_config(**kwargs) -> AxonConfig:
    defaults = {
        "max_actions_per_cycle": 10,
        "max_api_calls_per_minute": 30,
        "max_notifications_per_hour": 10,
        "max_concurrent_executions": 3,
        "total_timeout_per_cycle_ms": 30_000,
    }
    return AxonConfig(**{**defaults, **kwargs})


def _make_registry() -> ExecutorRegistry:
    reg = ExecutorRegistry()
    reg.register(_TestExecutor())
    reg.register(_HighAutonomyExecutor())
    reg.register(_LowAutonomyExecutor())
    return reg


# ─── Tests: ExecutorRegistry — capabilities() ────────────────────


class TestRegistryCapabilities:
    def test_capabilities_returns_all_executors(self) -> None:
        reg = _make_registry()
        caps = reg.capabilities()
        assert len(caps) == 3
        action_types = {c["action_type"] for c in caps}
        assert action_types == {"test_exec", "high_auto", "low_auto"}

    def test_capabilities_include_rate_limit(self) -> None:
        reg = _make_registry()
        caps = reg.capabilities()
        for cap in caps:
            assert "rate_limit" in cap
            rl = cap["rate_limit"]
            assert "max_calls" in rl
            assert "window_seconds" in rl

    def test_capabilities_sorted_by_action_type(self) -> None:
        reg = _make_registry()
        caps = reg.capabilities()
        action_types = [c["action_type"] for c in caps]
        assert action_types == sorted(action_types)


# ─── Tests: ExecutorRegistry — find_by_autonomy() ────────────────


class TestRegistryFindByAutonomy:
    def test_find_all_at_max_level(self) -> None:
        reg = _make_registry()
        result = reg.find_by_autonomy(10)
        assert len(result) == 3

    def test_find_filters_by_level(self) -> None:
        reg = _make_registry()
        # Only low_auto (1) and test_exec (2) should match
        result = reg.find_by_autonomy(2)
        assert "low_auto" in result
        assert "test_exec" in result
        assert "high_auto" not in result

    def test_find_none_at_zero(self) -> None:
        reg = _make_registry()
        result = reg.find_by_autonomy(0)
        assert result == []


# ─── Tests: ExecutorRegistry — deregister() ──────────────────────


class TestRegistryDeregister:
    def test_deregister_removes_executor(self) -> None:
        reg = _make_registry()
        assert reg.deregister("test_exec") is True
        assert reg.get("test_exec") is None
        assert len(reg) == 2

    def test_deregister_nonexistent_returns_false(self) -> None:
        reg = _make_registry()
        assert reg.deregister("nonexistent") is False

    def test_deregister_with_alias(self) -> None:
        """Aliases should normalise to canonical name for deregistration."""
        reg = ExecutorRegistry()

        class _ObsExec(_TestExecutor):
            action_type = "observe"

        reg.register(_ObsExec())
        # Deregister via alias
        assert reg.deregister("executor.observe") is True
        assert reg.get("observe") is None


# ─── Tests: RateLimiter — Adaptive Multipliers ───────────────────


class TestRateLimiterMultipliers:
    def test_global_multiplier_tightens(self) -> None:
        limiter = RateLimiter()
        limit = RateLimit.per_minute(10)
        limiter.set_global_multiplier(0.5)  # Halve the effective limit

        for _ in range(5):
            limiter.record("test")

        # 5 calls against effective max of 5 (10 * 0.5) — should block
        assert limiter.check("test", limit) is False

    def test_global_multiplier_loosens(self) -> None:
        limiter = RateLimiter()
        limit = RateLimit.per_minute(10)
        limiter.set_global_multiplier(2.0)  # Double the effective limit

        for _ in range(15):
            limiter.record("test")

        # 15 calls against effective max of 20 (10 * 2.0) — should allow
        assert limiter.check("test", limit) is True

    def test_per_action_multiplier(self) -> None:
        limiter = RateLimiter()
        limit = RateLimit.per_minute(10)
        limiter.set_multiplier("tight_action", 0.3)  # 3 effective max

        for _ in range(3):
            limiter.record("tight_action")

        assert limiter.check("tight_action", limit) is False
        # Other actions unaffected
        for _ in range(9):
            limiter.record("normal_action")
        assert limiter.check("normal_action", limit) is True

    def test_multiplier_combines_global_and_per_action(self) -> None:
        limiter = RateLimiter()
        limit = RateLimit.per_minute(100)
        limiter.set_global_multiplier(0.5)  # 50 effective
        limiter.set_multiplier("special", 0.5)  # 25 effective (100 * 0.5 * 0.5)

        for _ in range(25):
            limiter.record("special")
        assert limiter.check("special", limit) is False

    def test_clear_multiplier(self) -> None:
        limiter = RateLimiter()
        limit = RateLimit.per_minute(10)
        limiter.set_multiplier("test", 0.1)  # 1 effective max

        limiter.record("test")
        assert limiter.check("test", limit) is False

        limiter.clear_multiplier("test")
        assert limiter.check("test", limit) is True  # Back to 10 effective

    def test_reset_global_multiplier(self) -> None:
        limiter = RateLimiter()
        limit = RateLimit.per_minute(10)
        limiter.set_global_multiplier(0.1)  # 1 effective

        limiter.record("test")
        assert limiter.check("test", limit) is False

        limiter.reset_global_multiplier()
        assert limiter.check("test", limit) is True

    def test_minimum_effective_max_is_one(self) -> None:
        """Even with extreme tightening, at least 1 call should be allowed."""
        limiter = RateLimiter()
        limit = RateLimit.per_minute(1)
        limiter.set_global_multiplier(0.001)  # Would be 0.001, clamped to 0.1

        assert limiter.check("test", limit) is True  # At least 1 allowed
        limiter.record("test")
        assert limiter.check("test", limit) is False


# ─── Tests: BudgetTracker — Per-Intent Budgets ───────────────────


class TestPerIntentBudget:
    def test_per_intent_tracking(self) -> None:
        tracker = BudgetTracker(_make_config())
        tracker.begin_execution(intent_id="intent-1")
        tracker.begin_execution(intent_id="intent-1")
        tracker.begin_execution(intent_id="intent-2")

        # intent-1 used 2 steps, intent-2 used 1
        allowed, _ = tracker.can_execute_intent("intent-1", step_count=1, max_steps=3)
        assert allowed is True  # 2 + 1 <= 3

        allowed, reason = tracker.can_execute_intent("intent-1", step_count=2, max_steps=3)
        assert allowed is False  # 2 + 2 > 3
        assert "exhausted" in reason

    def test_zero_max_steps_means_unlimited(self) -> None:
        tracker = BudgetTracker(_make_config())
        allowed, _ = tracker.can_execute_intent("any", step_count=100, max_steps=0)
        assert allowed is True

    def test_begin_cycle_resets_intent_tracking(self) -> None:
        tracker = BudgetTracker(_make_config())
        tracker.begin_execution(intent_id="intent-1")
        tracker.begin_execution(intent_id="intent-1")
        tracker.begin_cycle()

        allowed, _ = tracker.can_execute_intent("intent-1", step_count=1, max_steps=1)
        assert allowed is True

    def test_backward_compatible_no_intent_id(self) -> None:
        """begin_execution() still works without intent_id."""
        tracker = BudgetTracker(_make_config(max_actions_per_cycle=3))
        tracker.begin_execution()
        tracker.begin_execution()
        tracker.begin_execution()
        allowed, _ = tracker.can_execute()
        assert allowed is False


# ─── Tests: AxonReactiveAdapter — Threat Adaptation ──────────────


class TestReactiveThreatAdaptation:
    @pytest.mark.asyncio
    async def test_threat_tightens_rate_limits(self) -> None:
        limiter = RateLimiter()
        adapter = AxonReactiveAdapter(rate_limiter=limiter)

        event = {"severity": "T4", "threat_type": "sandwich_attack"}
        await adapter._on_threat_detected(event)

        # Global multiplier should be tightened
        assert limiter._global_multiplier < 1.0
        # Financial actions should be extra tight
        assert "wallet_transfer" in limiter._multipliers
        assert limiter._multipliers["wallet_transfer"] < limiter._global_multiplier

    @pytest.mark.asyncio
    async def test_immune_cycle_restores_rates(self) -> None:
        limiter = RateLimiter()
        adapter = AxonReactiveAdapter(rate_limiter=limiter)

        # First tighten
        await adapter._on_threat_detected({"severity": "critical"})
        assert limiter._global_multiplier < 1.0

        # Then immune cycle completes with no active incidents
        await adapter._on_immune_cycle_complete({"active_incidents": 0})
        assert limiter._global_multiplier == 1.0
        assert "wallet_transfer" not in limiter._multipliers

    @pytest.mark.asyncio
    async def test_immune_cycle_keeps_tight_if_incidents_active(self) -> None:
        limiter = RateLimiter()
        adapter = AxonReactiveAdapter(rate_limiter=limiter)

        await adapter._on_threat_detected({"severity": "T3"})
        saved_multiplier = limiter._global_multiplier

        await adapter._on_immune_cycle_complete({"active_incidents": 2})
        # Should NOT restore — incidents still active
        assert limiter._global_multiplier == saved_multiplier

    @pytest.mark.asyncio
    async def test_rhythm_adapts_rate_limits(self) -> None:
        limiter = RateLimiter()
        adapter = AxonReactiveAdapter(rate_limiter=limiter)

        await adapter._on_rhythm_changed({"rhythm_state": "RUSH"})
        assert limiter._global_multiplier == pytest.approx(0.7)

        await adapter._on_rhythm_changed({"rhythm_state": "STALL"})
        assert limiter._global_multiplier == pytest.approx(1.3)

        await adapter._on_rhythm_changed({"rhythm_state": "FLOW"})
        assert limiter._global_multiplier == pytest.approx(1.0)


# ─── Tests: AxonIntrospector — Degradation Detection ─────────────


class TestIntrospectorDegradation:
    def test_profile_degradation_detection(self) -> None:
        profile = ExecutorProfile("failing_exec")
        # 5 executions, all failures
        for _ in range(5):
            profile.record_failure(100, "timeout")
        assert profile.is_degrading is True
        assert profile.consecutive_failures == 5

    def test_profile_not_degrading_with_mixed_results(self) -> None:
        profile = ExecutorProfile("mixed_exec")
        for _ in range(3):
            profile.record_success(50)
        for _ in range(2):
            profile.record_failure(100, "error")
        # 60% success rate, 2 consecutive — not degrading
        assert profile.is_degrading is False

    def test_introspector_generates_recommendations(self) -> None:
        """Stub outcome to exercise recommendation generation."""
        introspector = AxonIntrospector()

        # Manually create a degrading profile
        profile = ExecutorProfile("bad_exec")
        for _ in range(6):
            profile.record_failure(100, "crash")
        introspector._profiles["bad_exec"] = profile

        introspector._check_degradation()
        recs = introspector.drain_recommendations()
        assert len(recs) >= 1
        assert recs[0]["type"] == "executor_degrading"
        assert recs[0]["action_type"] == "bad_exec"

    def test_get_degrading_executors(self) -> None:
        introspector = AxonIntrospector()
        profile = ExecutorProfile("dying_exec")
        for _ in range(5):
            profile.record_failure(200, "oom")
        introspector._profiles["dying_exec"] = profile

        degrading = introspector.get_degrading_executors()
        assert len(degrading) == 1
        assert degrading[0]["action_type"] == "dying_exec"
        assert degrading[0]["is_degrading"] is True
