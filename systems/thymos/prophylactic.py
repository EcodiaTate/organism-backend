"""
EcodiaOS — Thymos Prophylactic Layer (Prevention)

Thymos doesn't just react to errors — it prevents them.

Two components:
  1. ProphylacticScanner    — scans new code against the antibody library
  2. HomeostasisController  — maintains optimal operating ranges proactively
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from systems.thymos.types import (
    ParameterAdjustment,
    ParameterFix,
    ProphylacticWarning,
)

logger = structlog.get_logger()


# ─── Prophylactic Scanner ───────────────────────────────────────


class ProphylacticScanner:
    """
    Scans new or modified code against the Antibody Library's error patterns.

    "This code pattern has caused 3 incidents in the past. The error was
    always an unhandled None return from Memory. Consider adding a null check."

    This is vaccination: exposure to weakened versions of past errors
    to build resistance before the real infection hits.
    """

    def __init__(self, antibody_library: Any = None) -> None:
        self._library = antibody_library
        self._scans_run: int = 0
        self._warnings_issued: int = 0
        self._logger = logger.bind(system="thymos", component="prophylactic_scanner")

    async def scan(
        self,
        files_changed: list[str],
        file_contents: dict[str, str] | None = None,
    ) -> list[ProphylacticWarning]:
        """
        Scan changed files against known error patterns.

        Args:
            files_changed: List of file paths that were modified.
            file_contents: Optional map of filepath → content. If not provided,
                          only filename-based pattern matching is used.
        """
        self._scans_run += 1
        if self._library is None:
            return []

        warnings: list[ProphylacticWarning] = []
        all_antibodies = await self._library.get_all_active()

        for filepath in files_changed:
            content = (file_contents or {}).get(filepath, "")

            for antibody in all_antibodies:
                # Check if this file is in the same system as the antibody
                if antibody.source_system not in filepath:
                    continue

                # Check error pattern similarity
                similarity = self._check_pattern_similarity(
                    content,
                    antibody.error_pattern,
                    filepath,
                )
                if similarity > 0.7:
                    warnings.append(
                        ProphylacticWarning(
                            filepath=filepath,
                            antibody_id=antibody.id,
                            warning=(
                                f"Code pattern similar to known error: "
                                f"{antibody.root_cause_description}"
                            ),
                            suggestion=self._extract_fix_suggestion(antibody),
                            confidence=similarity,
                        )
                    )

        self._warnings_issued += len(warnings)

        if warnings:
            self._logger.info(
                "prophylactic_warnings",
                files_scanned=len(files_changed),
                warnings=len(warnings),
            )

        return warnings

    def _check_pattern_similarity(
        self,
        content: str,
        error_pattern: str,
        filepath: str,
    ) -> float:
        """
        Check if file content resembles a known error-prone pattern.

        Uses simple keyword overlap — more sophisticated analysis
        (AST comparison, semantic embedding) can be added later.
        """
        if not content or not error_pattern:
            # Filename-only check: if the system matches, low confidence match
            return 0.3 if error_pattern else 0.0

        # Normalize both strings
        content_lower = content.lower()
        pattern_lower = error_pattern.lower()

        # Extract keywords from error pattern (skip placeholders)
        pattern_words = set(
            w for w in re.split(r"[^a-z_]+", pattern_lower)
            if len(w) > 3 and w not in {"none", "error", "failed", "the", "and", "was"}
        )

        if not pattern_words:
            return 0.0

        # Count how many pattern keywords appear in the content
        matches = sum(1 for w in pattern_words if w in content_lower)
        overlap = matches / len(pattern_words)

        return min(1.0, overlap)

    def _extract_fix_suggestion(self, antibody: Any) -> str:
        """Generate a fix suggestion from an antibody."""
        if antibody.repair_spec.action == "adjust_parameters":
            return f"Consider parameter adjustment: {antibody.root_cause_description}"
        if antibody.repair_spec.action == "restart_system":
            return "This pattern has historically required a restart to resolve."
        ab_short = antibody.id[:8]
        return f"Known fix available (antibody {ab_short}): {antibody.root_cause_description}"

    @property
    def stats(self) -> dict[str, int]:
        return {
            "scans_run": self._scans_run,
            "warnings_issued": self._warnings_issued,
        }


# ─── Homeostasis Controller ─────────────────────────────────────


# Optimal ranges for key metrics
DEFAULT_HOMEOSTATIC_RANGES: dict[str, tuple[float, float, str]] = {
    # Metric → (optimal_min, optimal_max, unit)
    "synapse.cycle.latency_ms": (80, 180, "ms"),
    "memory.retrieval.latency_ms": (10, 150, "ms"),
    "synapse.resources.memory_mb": (0, 3072, "MB"),
    "atune.coherence.phi": (0.3, 1.0, "phi"),
    "evo.self_model.success_rate": (0.5, 1.0, "rate"),
    "nova.intent_rate": (0.01, 0.5, "per_cycle"),
}


class HomeostasisController:
    """
    Maintains optimal operating ranges for key metrics.

    Like body temperature regulation: the organism doesn't wait for
    hypothermia or heatstroke. It actively maintains homeostasis.

    When a metric is trending toward the edge of its optimal range,
    Thymos makes small preemptive adjustments to pull it back.

    This runs on the MAINTAIN step of the cognitive cycle — always-on
    background processing, not triggered by incidents.
    """

    def __init__(
        self,
        ranges: dict[str, tuple[float, float, str]] | None = None,
    ) -> None:
        self._ranges = ranges or DEFAULT_HOMEOSTATIC_RANGES
        # Metric → list of recent values for trend detection
        self._history: dict[str, list[float]] = {
            name: [] for name in self._ranges
        }
        self._max_history = 200
        self._adjustments_made: int = 0
        self._logger = logger.bind(system="thymos", component="homeostasis")

    def record_metric(self, metric_name: str, value: float) -> None:
        """Record a metric value for trend tracking."""
        if metric_name not in self._ranges:
            return
        history = self._history[metric_name]
        history.append(value)
        if len(history) > self._max_history:
            self._history[metric_name] = history[-self._max_history:]

    def check_homeostasis(self) -> list[ParameterAdjustment]:
        """
        Check all homeostatic ranges and propose micro-adjustments
        for any metric trending toward the edge.

        Returns parameter adjustments (if any). These are Tier 1 —
        no governance required, just config nudges.
        """
        adjustments: list[ParameterAdjustment] = []

        for metric, (opt_min, opt_max, _unit) in self._ranges.items():
            history = self._history[metric]
            if len(history) < 10:
                continue  # Need enough data

            current = history[-1]
            trend = self._compute_trend(history, window=min(100, len(history)))

            # Approaching upper bound and trending up
            if current > opt_max * 0.85 and trend > 0:
                adj = self._prescribe_cooling(metric, current, opt_max, trend)
                if adj is not None:
                    adjustments.append(adj)

            # Approaching lower bound and trending down
            elif current < opt_min * 1.15 + 0.01 and trend < 0:
                adj = self._prescribe_warming(metric, current, opt_min, trend)
                if adj is not None:
                    adjustments.append(adj)

        if adjustments:
            self._adjustments_made += len(adjustments)

        return adjustments

    def _compute_trend(self, values: list[float], window: int) -> float:
        """Compute trend direction using simple linear regression slope."""
        if len(values) < 2:
            return 0.0
        recent = values[-window:]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _prescribe_cooling(
        self,
        metric: str,
        current: float,
        opt_max: float,
        trend: float,
    ) -> ParameterAdjustment | None:
        """Prescribe a small decrease to pull metric back from upper bound."""
        # Map metrics to adjustable parameters
        cooling_params: dict[str, tuple[str, float]] = {
            "synapse.cycle.latency_ms": ("synapse.clock.current_period_ms", 10),
            "memory.retrieval.latency_ms": ("memory.retrieval.cache_ttl_seconds", 60),
            "synapse.resources.memory_mb": ("evo.consolidation.batch_size", -2),
        }
        entry = cooling_params.get(metric)
        if entry is None:
            return None

        param_path, delta = entry
        opt_min = self._ranges[metric][0]

        return ParameterAdjustment(
            metric_name=metric,
            current_value=current,
            optimal_min=opt_min,
            optimal_max=opt_max,
            adjustment=ParameterFix(
                parameter_path=param_path,
                delta=delta,
                reason=f"Homeostatic cooling: {metric} at {current:.1f} (max: {opt_max})",
            ),
            trend_direction="rising",
        )

    def _prescribe_warming(
        self,
        metric: str,
        current: float,
        opt_min: float,
        trend: float,
    ) -> ParameterAdjustment | None:
        """Prescribe a small increase to pull metric back from lower bound."""
        warming_params: dict[str, tuple[str, float]] = {
            "atune.coherence.phi": ("synapse.clock.current_period_ms", -5),
            "evo.self_model.success_rate": ("evo.hypothesis.min_evidence", -1),
            "nova.intent_rate": ("synapse.clock.current_period_ms", -10),
        }
        entry = warming_params.get(metric)
        if entry is None:
            return None

        param_path, delta = entry
        opt_max = self._ranges[metric][1]

        return ParameterAdjustment(
            metric_name=metric,
            current_value=current,
            optimal_min=opt_min,
            optimal_max=opt_max,
            adjustment=ParameterFix(
                parameter_path=param_path,
                delta=delta,
                reason=f"Homeostatic warming: {metric} at {current:.2f} (min: {opt_min})",
            ),
            trend_direction="falling",
        )

    @property
    def metrics_in_range(self) -> int:
        """Count of metrics currently within optimal range."""
        count = 0
        for metric, (opt_min, opt_max, _) in self._ranges.items():
            history = self._history.get(metric, [])
            if history:
                current = history[-1]
                if opt_min <= current <= opt_max:
                    count += 1
        return count

    @property
    def metrics_total(self) -> int:
        return len(self._ranges)

    @property
    def adjustments_count(self) -> int:
        return self._adjustments_made
