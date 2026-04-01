"""
EcodiaOS - Telos: Alignment Gap Monitor

The alignment gap is the single most important health metric in EOS.
A large gap means EOS is claiming more intelligence than it has.
That's the definition of dangerous misalignment.

This monitor tracks the alignment gap over time, detects widening
trends, and escalates urgency when the gap is growing.
"""

from __future__ import annotations

from collections import deque

import structlog

from systems.telos.types import (
    AlignmentGapSample,
    AlignmentGapTrend,
    EffectiveIntelligenceReport,
    TelosConfig,
)

logger = structlog.get_logger()


class AlignmentGapMonitor:
    """
    Tracks the alignment gap (nominal_I - effective_I) over time.

    Maintains a rolling window of gap samples and computes:
    - Current gap fraction (gap / nominal_I)
    - Slope of gap fraction per hour (is it widening?)
    - Urgency level based on current gap + trend

    Urgency levels:
    - nominal: gap < 10% and not widening
    - warning: gap > 20% OR widening
    - critical: gap > 20% AND widening
    - emergency: gap > 40% OR widening rapidly (slope > 2x threshold)
    """

    def __init__(self, config: TelosConfig) -> None:
        self._config = config
        self._logger = logger.bind(component="telos.alignment")
        self._history: deque[AlignmentGapSample] = deque(
            maxlen=config.alignment_gap_history_size
        )

    def record(
        self,
        report: EffectiveIntelligenceReport,
        primary_cause: str,
    ) -> AlignmentGapTrend:
        """
        Record a new alignment gap measurement and return the current trend.
        Called after every effective_I computation.
        """
        gap_fraction = report.alignment_gap_fraction

        sample = AlignmentGapSample(
            nominal_I=report.nominal_I,
            effective_I=report.effective_I,
            gap_fraction=gap_fraction,
            primary_cause=primary_cause,
        )
        self._history.append(sample)

        return self.compute_trend()

    def compute_trend(self) -> AlignmentGapTrend:
        """Compute the alignment gap trend from the rolling history."""
        if not self._history:
            return AlignmentGapTrend()

        latest = self._history[-1]
        slope = self._compute_slope()
        threshold = self._config.alignment_gap_widening_slope_threshold
        is_widening = slope > threshold

        # Determine urgency
        urgency = self._classify_urgency(
            gap_fraction=latest.gap_fraction,
            slope=slope,
            threshold=threshold,
        )

        trend = AlignmentGapTrend(
            current_gap_fraction=latest.gap_fraction,
            slope_per_hour=slope,
            is_widening=is_widening,
            samples_count=len(self._history),
            primary_cause=latest.primary_cause,
            urgency=urgency,
        )

        if urgency != "nominal":
            self._logger.warning(
                "alignment_gap_trend",
                gap_fraction=f"{latest.gap_fraction:.3f}",
                slope=f"{slope:.4f}/hr",
                urgency=urgency,
                primary_cause=latest.primary_cause,
                samples=len(self._history),
            )

        return trend

    @property
    def current_gap_fraction(self) -> float:
        """Return the most recent gap fraction, or 0 if no history."""
        if not self._history:
            return 0.0
        return self._history[-1].gap_fraction

    @property
    def history(self) -> list[AlignmentGapSample]:
        """Return the full rolling history (oldest first)."""
        return list(self._history)

    def _compute_slope(self) -> float:
        """
        Compute the slope of the gap fraction over time (per hour).

        Uses linear regression on (time_offset_hours, gap_fraction) pairs.
        """
        if len(self._history) < 3:
            return 0.0

        samples = list(self._history)
        t0 = samples[0].timestamp
        points: list[tuple[float, float]] = []

        for sample in samples:
            dt_hours = (sample.timestamp - t0).total_seconds() / 3600.0
            points.append((dt_hours, sample.gap_fraction))

        return _linear_regression_slope(points)

    def _classify_urgency(
        self,
        gap_fraction: float,
        slope: float,
        threshold: float,
    ) -> str:
        """
        Classify urgency based on gap magnitude and trend.

        Emergency:  gap > 40% OR slope > 2x threshold
        Critical:   gap > 20% AND widening
        Warning:    gap > 20% OR widening
        Nominal:    everything else
        """
        is_widening = slope > threshold

        if gap_fraction > 0.4 or slope > threshold * 2:
            return "emergency"
        if gap_fraction > 0.2 and is_widening:
            return "critical"
        if gap_fraction > 0.2 or is_widening:
            return "warning"
        return "nominal"


# ─── Helpers ─────────────────────────────────────────────────────────


def _linear_regression_slope(points: list[tuple[float, float]]) -> float:
    """Compute the slope of a simple linear regression."""
    n = len(points)
    if n < 2:
        return 0.0

    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    sum_xy = sum(p[0] * p[1] for p in points)
    sum_x2 = sum(p[0] ** 2 for p in points)

    denominator = n * sum_x2 - sum_x**2
    if abs(denominator) < 1e-12:
        return 0.0

    return (n * sum_xy - sum_x * sum_y) / denominator
