"""
EcodiaOS — Simula Hardware Timing Profiler

Collects nanosecond-precision timing data for execution phases.
Analyzes timing variance and identifies side-channel information leakage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any

import structlog

logger = structlog.get_logger().bind(system="simula.sidechannel.timing")


@dataclass
class TimingRecord:
    """Single timing measurement for an operation."""
    operation: str
    duration_ns: int
    wall_clock_ms: float
    cpu_time_ns: int = 0
    memory_peak_mb: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)
    success: bool = True


@dataclass
class WorkloadClassification:
    """Classification of operation's resource usage pattern."""
    is_io_bound: bool = False
    is_cpu_bound: bool = False
    is_memory_bound: bool = False
    estimated_io_ratio: float = 0.0
    estimated_cpu_ratio: float = 0.0
    classification_confidence: float = 0.5


class TimingProfiler:
    """
    Records detailed timing for operations in proposal execution.

    Provides:
      - Nanosecond-precision timing per operation
      - Correlation analysis with state variables
      - Timing variance quantification
      - Workload classification (I/O vs CPU vs memory)
    """

    def __init__(self, proposal_id: str) -> None:
        self.proposal_id = proposal_id
        self.records: list[TimingRecord] = []
        self._operation_stack: list[tuple[str, int]] = []  # Stack of (name, start_ns)
        self._log = logger

    def start_operation(self, operation_name: str, context: dict[str, Any] | None = None) -> None:
        """Begin timing an operation."""
        start_ns = time.perf_counter_ns()
        self._operation_stack.append((operation_name, start_ns))

        self._log.debug(
            "timing_operation_start",
            proposal_id=self.proposal_id,
            operation=operation_name,
        )

    def end_operation(
        self,
        context: dict[str, Any] | None = None,
        success: bool = True,
    ) -> TimingRecord | None:
        """End timing for the current operation."""
        if not self._operation_stack:
            self._log.warning("timing_end_without_start", proposal_id=self.proposal_id)
            return None

        operation_name, start_ns = self._operation_stack.pop()
        end_ns = time.perf_counter_ns()
        duration_ns = end_ns - start_ns
        duration_ms = duration_ns / 1_000_000

        record = TimingRecord(
            operation=operation_name,
            duration_ns=duration_ns,
            wall_clock_ms=duration_ms,
            context=context or {},
            success=success,
        )
        self.records.append(record)

        self._log.debug(
            "timing_operation_end",
            proposal_id=self.proposal_id,
            operation=operation_name,
            duration_ms=round(duration_ms, 2),
        )

        return record

    def correlate_with_state(self, state_vars: dict[str, Any]) -> dict[str, float]:
        """
        Analyze correlation between operation timing and state variables.

        Returns correlation coefficients for each state variable.
        """
        if len(self.records) < 3:
            return {}

        correlations: dict[str, float] = {}

        for var_name, var_values in state_vars.items():
            if not isinstance(var_values, list) or len(var_values) != len(self.records):
                continue

            # Simple Pearson correlation
            timing_values = [r.duration_ns for r in self.records]
            correlation = self._pearson_correlation(timing_values, var_values)
            correlations[var_name] = correlation

        return correlations

    def classify_workload(self) -> WorkloadClassification:
        """
        Classify the overall workload based on timing patterns.

        Simple heuristic: if variance is high, likely I/O bound.
        If consistently fast, likely CPU bound or memory bound.
        """
        if len(self.records) < 2:
            return WorkloadClassification()

        durations = [r.duration_ns for r in self.records]
        mean_duration = mean(durations)
        variance = stdev(durations) if len(durations) > 1 else 0

        # Heuristic classification
        cv = (variance / mean_duration) if mean_duration > 0 else 0
        io_ratio = min(1.0, cv / 0.3)  # High CV suggests I/O variability
        cpu_ratio = 1.0 - io_ratio

        return WorkloadClassification(
            is_io_bound=io_ratio > 0.5,
            is_cpu_bound=cpu_ratio > 0.5,
            is_memory_bound=False,  # Would need memory profiling
            estimated_io_ratio=io_ratio,
            estimated_cpu_ratio=cpu_ratio,
            classification_confidence=min(0.9, abs(io_ratio - 0.5) * 2),
        )

    def get_variance_analysis(self) -> dict[str, float]:
        """Analyze timing variance across all records."""
        if len(self.records) < 2:
            return {"variance_percent": 0.0}

        durations = [r.duration_ns for r in self.records]
        mean_duration = mean(durations)
        variance = stdev(durations) if len(durations) > 1 else 0

        variance_percent = (variance / mean_duration * 100) if mean_duration > 0 else 0.0

        return {
            "mean_duration_ns": mean_duration,
            "stdev_duration_ns": variance,
            "variance_percent": variance_percent,
            "min_duration_ns": min(durations),
            "max_duration_ns": max(durations),
            "range_ns": max(durations) - min(durations),
        }

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive timing summary."""
        variance_analysis = self.get_variance_analysis()
        workload = self.classify_workload()
        total_duration_ns = sum(r.duration_ns for r in self.records)

        return {
            "proposal_id": self.proposal_id,
            "total_operations": len(self.records),
            "total_duration_ns": total_duration_ns,
            "total_duration_ms": total_duration_ns / 1_000_000,
            "variance_analysis": variance_analysis,
            "workload_classification": {
                "is_io_bound": workload.is_io_bound,
                "is_cpu_bound": workload.is_cpu_bound,
                "io_ratio": workload.estimated_io_ratio,
                "cpu_ratio": workload.estimated_cpu_ratio,
            },
            "operations": [
                {
                    "name": r.operation,
                    "duration_ns": r.duration_ns,
                    "duration_ms": r.duration_ns / 1_000_000,
                    "success": r.success,
                }
                for r in self.records
            ],
        }

    @staticmethod
    def _pearson_correlation(x: list[float], y: list[float]) -> float:
        """Compute Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        x_mean = mean(x)
        y_mean = mean(y)

        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        x_var = sum((xi - x_mean) ** 2 for xi in x)
        y_var = sum((yi - y_mean) ** 2 for yi in y)

        denominator = (x_var * y_var) ** 0.5
        if denominator == 0:
            return 0.0

        return numerator / denominator
