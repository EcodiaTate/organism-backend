"""
EcodiaOS -- Axon Execution Introspection & Autonomous Learning

Axon learns from its own execution patterns to become more effective
over time. This is not external learning (Evo's domain) -- this is
the motor cortex optimising its own execution machinery:

1. Executor Performance Tracking
   - Per-executor success rate, latency percentiles, failure modes
   - Detects degrading executors before circuit breaker trips

2. Outcome Pattern Detection
   - Identifies recurring failure sequences
   - Detects action patterns that consistently succeed
   - Feeds patterns to Evo as ACTION_COMPLETED enrichment

3. Adaptive Safety Tuning
   - Rate limit adjustment based on actual usage patterns
   - Budget utilisation smoothing across cycles
   - Circuit breaker threshold adaptation

This is what makes Axon a learning motor system rather than a
static executor -- it improves its own execution efficiency
without requiring external intervention.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.axon.types import AxonOutcome

logger = structlog.get_logger()


class ExecutorProfile:
    """Performance profile for a single executor type."""

    __slots__ = (
        "action_type",
        "total_executions",
        "successes",
        "failures",
        "latencies_ms",
        "failure_reasons",
        "last_execution_time",
        "_consecutive_failures",
    )

    def __init__(self, action_type: str) -> None:
        self.action_type = action_type
        self.total_executions: int = 0
        self.successes: int = 0
        self.failures: int = 0
        self.latencies_ms: deque[int] = deque(maxlen=100)
        self.failure_reasons: deque[str] = deque(maxlen=20)
        self.last_execution_time: float = 0.0
        self._consecutive_failures: int = 0

    def record_success(self, latency_ms: int) -> None:
        self.total_executions += 1
        self.successes += 1
        self.latencies_ms.append(latency_ms)
        self.last_execution_time = time.monotonic()
        self._consecutive_failures = 0

    def record_failure(self, latency_ms: int, reason: str) -> None:
        self.total_executions += 1
        self.failures += 1
        self.latencies_ms.append(latency_ms)
        self.failure_reasons.append(reason)
        self.last_execution_time = time.monotonic()
        self._consecutive_failures += 1

    @property
    def success_rate(self) -> float:
        if self.total_executions == 0:
            return 1.0
        return self.successes / self.total_executions

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return sum(self.latencies_ms) / len(self.latencies_ms)

    @property
    def p95_latency_ms(self) -> int:
        if not self.latencies_ms:
            return 0
        sorted_l = sorted(self.latencies_ms)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def is_degrading(self) -> bool:
        """True if the executor shows signs of degradation."""
        if self.total_executions < 5:
            return False
        # Recent success rate below 50% with at least 3 consecutive failures
        return self.success_rate < 0.5 and self._consecutive_failures >= 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "total_executions": self.total_executions,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p95_latency_ms": self.p95_latency_ms,
            "consecutive_failures": self._consecutive_failures,
            "is_degrading": self.is_degrading,
            "recent_failures": list(self.failure_reasons)[-5:],
        }


class OutcomePattern:
    """Tracks recurring execution patterns (success/failure sequences)."""

    def __init__(self, max_patterns: int = 50) -> None:
        # Pattern key -> (success_count, failure_count, last_seen)
        self._patterns: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        self._max_patterns = max_patterns

    def record(self, action_types: list[str], success: bool) -> None:
        """Record a multi-step execution pattern."""
        if not action_types:
            return
        key = "->".join(action_types)
        pattern = self._patterns[key]
        if success:
            pattern[0] += 1
        else:
            pattern[1] += 1
        pattern[2] = int(time.monotonic())

        # Evict oldest patterns if over limit
        if len(self._patterns) > self._max_patterns:
            oldest_key = min(
                self._patterns,
                key=lambda k: self._patterns[k][2],
            )
            del self._patterns[oldest_key]

    def get_reliable_patterns(self, min_executions: int = 3) -> list[dict[str, Any]]:
        """Return patterns with enough data to be meaningful."""
        reliable = []
        for key, (successes, failures, _last_seen) in self._patterns.items():
            total = successes + failures
            if total < min_executions:
                continue
            rate = successes / total if total > 0 else 0.0
            reliable.append({
                "pattern": key,
                "total_executions": total,
                "success_rate": round(rate, 4),
                "successes": successes,
                "failures": failures,
            })
        return sorted(reliable, key=lambda p: p["total_executions"], reverse=True)

    def get_failure_hotspots(self, threshold: float = 0.5) -> list[dict[str, Any]]:
        """Return patterns with failure rate above threshold."""
        return [
            p for p in self.get_reliable_patterns()
            if p["success_rate"] < threshold
        ]


class AxonIntrospector:
    """
    Axon's self-awareness layer. Tracks execution patterns, detects
    degradation, and provides recommendations for adaptive tuning.

    Integrated into AxonService.execute() to record every outcome.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, ExecutorProfile] = {}
        self._patterns = OutcomePattern()
        self._cycle_utilizations: deque[float] = deque(maxlen=100)
        self._logger = logger.bind(system="axon.introspection")

        # Adaptation recommendations
        self._pending_recommendations: deque[dict[str, Any]] = deque(maxlen=20)

    def record_outcome(self, outcome: AxonOutcome) -> None:
        """
        Record an execution outcome for introspection.

        Call this after every AxonService.execute() to feed the learning loop.
        """
        # Per-step profiles
        for step in outcome.step_outcomes:
            profile = self._profiles.get(step.action_type)
            if profile is None:
                profile = ExecutorProfile(step.action_type)
                self._profiles[step.action_type] = profile

            latency = step.duration_ms if hasattr(step, "duration_ms") else 0
            if step.result.success:
                profile.record_success(latency)
            else:
                reason = step.result.error or "unknown"
                profile.record_failure(latency, reason)

        # Outcome patterns (multi-step)
        action_types = [s.action_type for s in outcome.step_outcomes]
        self._patterns.record(action_types, outcome.success)

        # Check for degradation and generate recommendations
        self._check_degradation()

    def record_cycle_utilization(self, utilization: float) -> None:
        """Record per-cycle budget utilization for smoothing analysis."""
        self._cycle_utilizations.append(utilization)

    def _check_degradation(self) -> None:
        """Detect degrading executors and generate recommendations."""
        for action_type, profile in self._profiles.items():
            if profile.is_degrading:
                recommendation = {
                    "type": "executor_degrading",
                    "action_type": action_type,
                    "success_rate": profile.success_rate,
                    "consecutive_failures": profile.consecutive_failures,
                    "recent_failures": list(profile.failure_reasons)[-3:],
                    "recommendation": (
                        f"Executor '{action_type}' is degrading "
                        f"(success rate {profile.success_rate:.0%}, "
                        f"{profile.consecutive_failures} consecutive failures). "
                        f"Consider pre-emptive circuit break or repair."
                    ),
                    "timestamp": time.monotonic(),
                }

                # Don't duplicate recommendations for the same executor
                existing = any(
                    r.get("action_type") == action_type
                    and r.get("type") == "executor_degrading"
                    for r in self._pending_recommendations
                )
                if not existing:
                    self._pending_recommendations.append(recommendation)
                    self._logger.warning(
                        "executor_degradation_detected",
                        action_type=action_type,
                        success_rate=round(profile.success_rate, 3),
                        consecutive_failures=profile.consecutive_failures,
                    )

        # Check for failure hotspot patterns
        hotspots = self._patterns.get_failure_hotspots()
        for hotspot in hotspots[:3]:
            recommendation = {
                "type": "pattern_failure_hotspot",
                "pattern": hotspot["pattern"],
                "success_rate": hotspot["success_rate"],
                "total_executions": hotspot["total_executions"],
                "recommendation": (
                    f"Action sequence '{hotspot['pattern']}' fails "
                    f"{1 - hotspot['success_rate']:.0%} of the time. "
                    f"Nova should consider alternative action plans."
                ),
                "timestamp": time.monotonic(),
            }
            existing = any(
                r.get("pattern") == hotspot["pattern"]
                and r.get("type") == "pattern_failure_hotspot"
                for r in self._pending_recommendations
            )
            if not existing:
                self._pending_recommendations.append(recommendation)

    def get_degrading_executors(self) -> list[dict[str, Any]]:
        """Return profiles of currently degrading executors."""
        return [
            p.to_dict()
            for p in self._profiles.values()
            if p.is_degrading
        ]

    def drain_recommendations(self) -> list[dict[str, Any]]:
        """Drain and return all pending adaptation recommendations."""
        items = list(self._pending_recommendations)
        self._pending_recommendations.clear()
        return items

    def get_executor_profile(self, action_type: str) -> dict[str, Any] | None:
        """Return the performance profile for a specific executor."""
        profile = self._profiles.get(action_type)
        return profile.to_dict() if profile is not None else None

    @property
    def stats(self) -> dict[str, Any]:
        total_execs = sum(p.total_executions for p in self._profiles.values())
        total_success = sum(p.successes for p in self._profiles.values())
        degrading = [p.action_type for p in self._profiles.values() if p.is_degrading]

        avg_utilization = 0.0
        if self._cycle_utilizations:
            avg_utilization = sum(self._cycle_utilizations) / len(self._cycle_utilizations)

        return {
            "tracked_executors": len(self._profiles),
            "total_executions_tracked": total_execs,
            "overall_success_rate": round(
                total_success / total_execs if total_execs > 0 else 1.0, 4
            ),
            "degrading_executors": degrading,
            "reliable_patterns": len(self._patterns.get_reliable_patterns()),
            "failure_hotspots": len(self._patterns.get_failure_hotspots()),
            "pending_recommendations": len(self._pending_recommendations),
            "avg_cycle_utilization": round(avg_utilization, 4),
        }

    @property
    def full_report(self) -> dict[str, Any]:
        """Comprehensive introspection report for diagnostics."""
        return {
            "executor_profiles": {
                at: p.to_dict() for at, p in self._profiles.items()
            },
            "reliable_patterns": self._patterns.get_reliable_patterns(),
            "failure_hotspots": self._patterns.get_failure_hotspots(),
            "recommendations": list(self._pending_recommendations),
            "stats": self.stats,
        }
