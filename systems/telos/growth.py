"""
EcodiaOS - Telos: Growth Topology Engine

Growth is not accumulation. Growth is the drive that keeps dI/dt positive -
the temporal dimension of intelligence topology. A model with high Growth
continuously identifies the frontier of its own understanding, seeks
high-information-content deltas, and never allows cognitive pressure to reach
zero.

Growth doesn't multiply I - it multiplies dI/dt. It determines how fast
effective intelligence grows, not its current value.
"""

from __future__ import annotations

import structlog

from systems.telos.types import (
    FoveaMetrics,
    GrowthDirective,
    GrowthMetrics,
    LogosMetrics,
    TelosConfig,
    TimestampedValue,
)

logger = structlog.get_logger()


class GrowthTopologyEngine:
    """
    Implements Growth as the temporal dimension of intelligence topology.

    Computes:
    - dI/dt: first derivative of the intelligence ratio (velocity)
    - d2I/dt2: second derivative (acceleration)
    - frontier_domains: where coverage is weakest (< 0.4)
    - novel_domain_fraction: what fraction of recent experiences were novel
    - compression_rate: compression events per hour
    - growth_score: composite metric in [-1, 1]
    - growth_pressure_needed: True when dI/dt falls below minimum
    """

    def __init__(self, config: TelosConfig) -> None:
        self._config = config
        self._logger = logger.bind(component="telos.growth")

    async def compute_growth_rate(
        self,
        logos: LogosMetrics,
        fovea: FoveaMetrics,
    ) -> GrowthMetrics:
        """
        Compute growth metrics from intelligence ratio history and exploration patterns.

        Steps:
        1. Get I history from Logos over the configured window
        2. Compute dI/dt (first derivative) and d2I/dt2 (second derivative)
        3. Get domain coverage to identify frontier domains
        4. Get recent high-error experiences to measure novel domain fraction
        5. Get recent compression events to measure compression rate
        6. Compose into a growth_score
        """
        window_hours = self._config.growth_window_hours

        # Intelligence ratio over time
        I_history = await logos.get_I_history(window_hours=window_hours)
        dI_dt = self._compute_derivative(I_history)
        d2I_dt2 = self._compute_second_derivative(I_history)

        # Domain frontier: where is the world model weakest?
        domain_coverage = await logos.get_domain_coverage_map()
        frontier_domains = sorted(
            [d for d, cov in domain_coverage.items() if cov < 0.4],
            key=lambda d: domain_coverage.get(d, 0.0),
        )

        # Exploration rate: what fraction of recent experiences were in novel domains?
        recent_experiences = await fovea.get_recent_high_error_experiences(
            window_hours=window_hours
        )
        novel_domain_fraction = self._compute_novel_domain_fraction(
            recent_experiences, domain_coverage
        )

        # Compression rate: events per hour
        compression_events = await logos.get_recent_compression_events(
            window_hours=window_hours
        )
        compression_rate = (
            len(compression_events) / max(window_hours, 0.001)
        )

        # Composite growth score
        growth_score = self._composite_growth_score(
            dI_dt=dI_dt,
            d2I_dt2=d2I_dt2,
            novel_domain_fraction=novel_domain_fraction,
            compression_rate=compression_rate,
        )

        # Is growth stagnating?
        growth_pressure_needed = dI_dt < self._config.minimum_growth_rate

        metrics = GrowthMetrics(
            dI_dt=dI_dt,
            d2I_dt2=d2I_dt2,
            frontier_domains=frontier_domains[:10],  # Top 10 weakest
            novel_domain_fraction=novel_domain_fraction,
            compression_rate=compression_rate,
            growth_score=growth_score,
            growth_pressure_needed=growth_pressure_needed,
        )

        self._logger.debug(
            "growth_rate_computed",
            dI_dt=f"{dI_dt:.4f}",
            d2I_dt2=f"{d2I_dt2:.4f}",
            frontier_count=len(frontier_domains),
            novel_fraction=f"{novel_domain_fraction:.3f}",
            compression_rate=f"{compression_rate:.2f}",
            score=f"{growth_score:.3f}",
            stagnating=growth_pressure_needed,
        )

        return metrics

    def generate_growth_directive(
        self, growth_metrics: GrowthMetrics
    ) -> GrowthDirective | None:
        """
        When growth rate falls below minimum, generate a directive.

        The urgency is proportional to how far below the minimum the growth
        rate has fallen. The directive targets the top frontier domains.
        """
        if not growth_metrics.growth_pressure_needed:
            return None

        min_rate = self._config.minimum_growth_rate
        # Urgency: 0 when dI/dt == min_rate, 1 when dI/dt <= 0
        urgency = max(0.0, min(1.0, 1.0 - (growth_metrics.dI_dt / max(min_rate, 1e-9))))

        directive = GrowthDirective(
            urgency=urgency,
            frontier_domains=growth_metrics.frontier_domains[:3],
            directive="explore_frontier",
        )

        self._logger.info(
            "growth_directive_generated",
            urgency=f"{urgency:.3f}",
            frontier_targets=directive.frontier_domains,
        )

        return directive

    def _compute_derivative(self, history: list[TimestampedValue]) -> float:
        """
        Compute dI/dt as the slope of a linear regression over the history.

        Uses simple least-squares regression on (time_offset, value) pairs.
        Returns the slope in units of I per hour.
        """
        if len(history) < 2:
            return 0.0

        # Convert timestamps to hours relative to the first point
        t0 = history[0].timestamp
        points: list[tuple[float, float]] = []
        for entry in history:
            dt_hours = (entry.timestamp - t0).total_seconds() / 3600.0
            points.append((dt_hours, entry.value))

        return _linear_regression_slope(points)

    def _compute_second_derivative(self, history: list[TimestampedValue]) -> float:
        """
        Compute d2I/dt2: the acceleration of intelligence growth.

        Split the history in half and compute dI/dt for each half,
        then take the difference.
        """
        if len(history) < 4:
            return 0.0

        mid = len(history) // 2
        first_half = history[:mid]
        second_half = history[mid:]

        dI_dt_first = self._compute_derivative(first_half)
        dI_dt_second = self._compute_derivative(second_half)

        # Time span of each half (in hours)
        t0 = first_half[0].timestamp
        t_end = second_half[-1].timestamp
        dt_hours = (t_end - t0).total_seconds() / 3600.0

        if dt_hours <= 0:
            return 0.0

        return (dI_dt_second - dI_dt_first) / dt_hours

    def _compute_novel_domain_fraction(
        self,
        recent_experiences: list,  # list[HighErrorExperience]
        domain_coverage: dict[str, float],
    ) -> float:
        """
        Compute the fraction of recent experiences that were in novel domains.

        A domain is "novel" if its coverage is below 0.2 (barely explored).
        """
        if not recent_experiences:
            return 0.0

        novel_count = sum(
            1
            for exp in recent_experiences
            if domain_coverage.get(exp.domain, 0.0) < 0.2 or exp.was_novel_domain
        )

        return novel_count / len(recent_experiences)

    def _composite_growth_score(
        self,
        dI_dt: float,
        d2I_dt2: float,
        novel_domain_fraction: float,
        compression_rate: float,
    ) -> float:
        """
        Composite growth score in [-1, 1].

        Weights:
        - dI/dt velocity: 0.35 (is intelligence growing?)
        - d2I/dt2 acceleration: 0.20 (is growth accelerating?)
        - novel_domain_fraction: 0.25 (is the model exploring?)
        - compression_rate: 0.20 (is the model compressing?)

        Each component is normalized to [-1, 1] before weighting.
        """
        # Normalize dI/dt: positive is good, negative is bad
        # Saturate at +/-0.1 per hour
        velocity_norm = max(-1.0, min(1.0, dI_dt / 0.1))

        # Normalize d2I/dt2: positive acceleration is good
        accel_norm = max(-1.0, min(1.0, d2I_dt2 / 0.01))

        # Novel domain fraction is already in [0, 1]; map to [-1, 1]
        # 0% novel = -1 (no exploration), 100% novel = 1
        exploration_norm = novel_domain_fraction * 2.0 - 1.0

        # Compression rate: normalize against expected baseline (1/hour)
        compression_norm = max(-1.0, min(1.0, (compression_rate - 0.5) / 0.5))

        score = (
            0.35 * velocity_norm
            + 0.20 * accel_norm
            + 0.25 * exploration_norm
            + 0.20 * compression_norm
        )

        return max(-1.0, min(1.0, score))


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
