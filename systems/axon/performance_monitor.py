"""
EcodiaOS - Axon Action Performance Monitor

Tracks rolling action success rate and detects motor degradation.
When degradation is detected, emits MOTOR_DEGRADATION_DETECTED so Nova
can replan with alternative strategies.

Closure Loop 2: Axon → Nova (Motor Degradation → Replanning)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import structlog

logger = structlog.get_logger("systems.axon.performance_monitor")

# Cooldown between consecutive MOTOR_DEGRADATION_DETECTED emissions (seconds)
_DEGRADATION_EMIT_COOLDOWN_S: float = 60.0


class ActionPerformanceMonitor:
    """Tracks rolling action success rate and detects degradation."""

    def __init__(
        self,
        window_size: int = 20,
        degradation_threshold: float = 0.5,
    ):
        self._results: deque[bool] = deque(maxlen=window_size)
        self._recent_errors: deque[str] = deque(maxlen=5)
        self._affected_executors: deque[str] = deque(maxlen=10)
        self._degradation_threshold = degradation_threshold
        self._window_size = window_size
        self._last_alert_time: float = 0.0

    def record(self, success: bool, error: str = "", executor_type: str = "") -> bool:
        """Record result. Returns True if degradation alert should fire."""
        self._results.append(success)
        if not success:
            if error:
                self._recent_errors.append(error[:200])
            if executor_type:
                self._affected_executors.append(executor_type)

        if len(self._results) < 5:
            return False

        rate = sum(self._results) / len(self._results)
        if rate < self._degradation_threshold:
            now = time.monotonic()
            if now - self._last_alert_time < _DEGRADATION_EMIT_COOLDOWN_S:
                return False
            self._last_alert_time = now
            return True
        return False

    @property
    def success_rate(self) -> float:
        if not self._results:
            return 1.0
        return sum(self._results) / len(self._results)

    def build_event_data(self) -> dict[str, Any]:
        """Build the event payload for MOTOR_DEGRADATION_DETECTED."""
        return {
            "success_rate": round(self.success_rate, 3),
            "window_size": len(self._results),
            "recent_errors": list(self._recent_errors),
            "affected_executors": list(set(self._affected_executors)),
        }
