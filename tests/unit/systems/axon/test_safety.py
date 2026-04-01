"""
Unit tests for Axon safety mechanisms.

Tests RateLimiter, CircuitBreaker, and BudgetTracker.
"""

from __future__ import annotations

import time

import pytest

from config import AxonConfig
from systems.axon.safety import BudgetTracker, CircuitBreaker, RateLimiter
from systems.axon.types import CircuitStatus, RateLimit

# ─── Fixtures ─────────────────────────────────────────────────────


def make_config(**kwargs) -> AxonConfig:
    defaults = {
        "max_actions_per_cycle": 5,
        "max_api_calls_per_minute": 30,
        "max_notifications_per_hour": 10,
        "max_concurrent_executions": 3,
        "total_timeout_per_cycle_ms": 30_000,
    }
    return AxonConfig(**{**defaults, **kwargs})


# ─── Tests: RateLimiter ───────────────────────────────────────────


class TestRateLimiter:
    def test_allows_within_limit(self):
        limiter = RateLimiter()
        limit = RateLimit.per_minute(5)
        for _ in range(5):
            assert limiter.check("test", limit) is True
            limiter.record("test")

    def test_blocks_when_limit_exceeded(self):
        limiter = RateLimiter()
        limit = RateLimit.per_minute(3)
        for _ in range(3):
            limiter.record("test")
        assert limiter.check("test", limit) is False

    def test_different_action_types_are_independent(self):
        limiter = RateLimiter()
        limit = RateLimit.per_minute(1)
        limiter.record("action_a")
        # action_a is at limit but action_b should still be allowed
        assert limiter.check("action_a", limit) is False
        assert limiter.check("action_b", limit) is True

    def test_unlimited_never_blocks(self):
        limiter = RateLimiter()
        limit = RateLimit.unlimited()
        for _ in range(10_000):
            assert limiter.check("test", limit) is True
            limiter.record("test")

    def test_check_does_not_record(self):
        """check() should not consume a rate limit slot."""
        limiter = RateLimiter()
        limit = RateLimit.per_minute(1)
        # Check multiple times without recording - all should pass
        for _ in range(5):
            assert limiter.check("test", limit) is True
        # Now record once - should still have 0 used after that
        limiter.record("test")
        assert limiter.check("test", limit) is False

    def test_reset_clears_window(self):
        limiter = RateLimiter()
        limit = RateLimit.per_minute(1)
        limiter.record("test")
        assert limiter.check("test", limit) is False
        limiter.reset("test")
        assert limiter.check("test", limit) is True

    def test_current_count(self):
        limiter = RateLimiter()
        limiter.record("test")
        limiter.record("test")
        assert limiter.current_count("test", window_seconds=60) == 2
        assert limiter.current_count("test", window_seconds=0) == 0


# ─── Tests: CircuitBreaker ────────────────────────────────────────


class TestCircuitBreaker:
    def make_breaker(self, threshold=3, recovery_s=300) -> CircuitBreaker:
        return CircuitBreaker(
            failure_threshold=threshold,
            recovery_timeout_s=recovery_s,
            half_open_max_calls=1,
        )

    def test_starts_closed(self):
        breaker = self.make_breaker()
        assert breaker.can_execute("test") is True
        assert breaker.status("test") == CircuitStatus.CLOSED

    def test_trips_after_threshold_failures(self):
        breaker = self.make_breaker(threshold=3)
        for _ in range(3):
            breaker.record_result("test", success=False)
        assert breaker.status("test") == CircuitStatus.OPEN
        assert breaker.can_execute("test") is False

    def test_success_resets_failure_count(self):
        breaker = self.make_breaker(threshold=3)
        for _ in range(2):
            breaker.record_result("test", success=False)
        breaker.record_result("test", success=True)
        assert breaker.status("test") == CircuitStatus.CLOSED
        # Failure count reset - need 3 more to trip
        for _ in range(2):
            breaker.record_result("test", success=False)
        assert breaker.status("test") == CircuitStatus.CLOSED

    def test_different_executors_independent(self):
        breaker = self.make_breaker(threshold=2)
        breaker.record_result("action_a", success=False)
        breaker.record_result("action_a", success=False)
        assert breaker.status("action_a") == CircuitStatus.OPEN
        assert breaker.can_execute("action_b") is True  # Unaffected

    def test_half_open_after_recovery_timeout(self):
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout_s=0,  # Immediate recovery for testing
            half_open_max_calls=1,
        )
        breaker.record_result("test", success=False)
        assert breaker.status("test") == CircuitStatus.OPEN

        # With recovery_timeout_s=0, recovery happens immediately
        # Simulate time passage by directly checking
        time.sleep(0.01)
        assert breaker.can_execute("test") is True
        assert breaker.status("test") == CircuitStatus.HALF_OPEN

    def test_closes_on_half_open_success(self):
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout_s=0,
            half_open_max_calls=1,
        )
        breaker.record_result("test", success=False)
        time.sleep(0.01)
        breaker.can_execute("test")  # Transitions to HALF_OPEN
        breaker.record_result("test", success=True)
        assert breaker.status("test") == CircuitStatus.CLOSED

    def test_reopens_on_half_open_failure(self):
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout_s=0,
            half_open_max_calls=1,
        )
        breaker.record_result("test", success=False)
        time.sleep(0.01)
        breaker.can_execute("test")  # Transitions to HALF_OPEN
        breaker.record_result("test", success=False)
        assert breaker.status("test") == CircuitStatus.OPEN

    def test_manual_reset(self):
        breaker = self.make_breaker(threshold=1)
        breaker.record_result("test", success=False)
        assert breaker.status("test") == CircuitStatus.OPEN
        breaker.reset("test")
        assert breaker.status("test") == CircuitStatus.CLOSED
        assert breaker.can_execute("test") is True

    def test_trip_count(self):
        breaker = self.make_breaker(threshold=1)
        assert breaker.trip_count() == 0
        breaker.record_result("action_a", success=False)
        assert breaker.trip_count() == 1
        breaker.record_result("action_b", success=False)
        assert breaker.trip_count() == 2


# ─── Tests: BudgetTracker ─────────────────────────────────────────


class TestBudgetTracker:
    def test_allows_within_budget(self):
        tracker = BudgetTracker(make_config(max_actions_per_cycle=5))
        allowed, reason = tracker.can_execute()
        assert allowed is True
        assert reason == ""

    def test_blocks_when_action_limit_exceeded(self):
        tracker = BudgetTracker(make_config(max_actions_per_cycle=2))
        tracker.begin_execution()
        tracker.end_execution()
        tracker.begin_execution()
        tracker.end_execution()
        allowed, reason = tracker.can_execute()
        assert allowed is False
        assert "max actions" in reason

    def test_blocks_when_concurrent_limit_exceeded(self):
        tracker = BudgetTracker(make_config(max_concurrent_executions=2))
        tracker.begin_execution()
        tracker.begin_execution()
        allowed, reason = tracker.can_execute()
        assert allowed is False
        assert "concurrent" in reason

    def test_concurrency_released_after_end(self):
        tracker = BudgetTracker(make_config(max_concurrent_executions=1))
        tracker.begin_execution()
        assert tracker.can_execute()[0] is False
        tracker.end_execution()
        assert tracker.can_execute()[0] is True

    def test_begin_cycle_resets_counters(self):
        tracker = BudgetTracker(make_config(max_actions_per_cycle=1))
        tracker.begin_execution()
        tracker.end_execution()
        assert tracker.can_execute()[0] is False
        tracker.begin_cycle()
        assert tracker.can_execute()[0] is True

    def test_utilisation_fraction(self):
        tracker = BudgetTracker(make_config(max_actions_per_cycle=4))
        assert tracker.utilisation == 0.0
        tracker.begin_execution()
        assert tracker.utilisation == pytest.approx(0.25)
        tracker.begin_execution()
        assert tracker.utilisation == pytest.approx(0.50)

    def test_concurrent_never_goes_negative(self):
        tracker = BudgetTracker(make_config())
        tracker.end_execution()  # Without begin - should not go below 0
        assert tracker._concurrent_executions == 0
