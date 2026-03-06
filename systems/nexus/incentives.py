"""
EcodiaOS — Nexus: Divergence Incentive Engine

Computes triangulation weights and divergence pressure signals.

When an instance is too similar to the federation (triangulation_weight < 0.4),
Nexus pushes it toward frontier domains and away from saturated ones.
This pressure is routed to Thymos as a GROWTH drive signal.

Key operations:
  compute_triangulation_weight: proportional to average divergence from
    all other instances. Near-duplicate = near-zero weight.
  compute_divergence_pressure: when weight is low, generate a pressure
    signal with frontier/saturated domain recommendations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from primitives.common import utc_now
from systems.nexus.types import (
    DivergencePressure,
    DivergenceScore,
    InstanceDivergenceProfile,
)

if TYPE_CHECKING:
    from systems.nexus.protocols import ThymosDriveSinkProtocol

logger = structlog.get_logger("nexus.incentives")

_PRESSURE_THRESHOLD = 0.4


class DivergenceIncentiveEngine:
    """
    Computes triangulation weights and generates divergence pressure.

    Triangulation weight: how valuable an instance's contributions are
    to the federation. Proportional to average divergence from all peers.
    Near-duplicate instances have near-zero weight.

    Divergence pressure: when weight < 0.4, generates a GROWTH drive
    signal pushing toward frontier (under-explored) domains and away
    from saturated (over-covered) domains.
    """

    def __init__(
        self,
        *,
        thymos: ThymosDriveSinkProtocol | None = None,
        local_instance_id: str = "",
    ) -> None:
        self._thymos = thymos
        self._local_instance_id = local_instance_id
        self._divergence_cache: dict[str, DivergenceScore] = {}
        self._federation_domain_counts: dict[str, int] = {}

    def update_divergence(
        self, remote_instance_id: str, score: DivergenceScore
    ) -> None:
        """Update cached divergence score for a remote instance."""
        self._divergence_cache[remote_instance_id] = score

    def update_federation_domains(
        self, profiles: list[InstanceDivergenceProfile]
    ) -> None:
        """
        Update federation domain coverage counts from all known profiles.
        Used to identify frontier vs saturated domains.
        """
        self._federation_domain_counts.clear()
        for profile in profiles:
            for domain in profile.domain_coverage:
                self._federation_domain_counts[domain] = (
                    self._federation_domain_counts.get(domain, 0) + 1
                )

    def compute_triangulation_weight(self) -> float:
        """
        Compute how much this instance's contributions are worth.

        Proportional to average divergence from all other instances.
        Near-duplicate = near-zero weight. Unique = full weight (1.0).
        """
        if not self._divergence_cache:
            return 1.0  # Solo instance = full weight

        total_divergence = sum(
            score.overall for score in self._divergence_cache.values()
        )
        avg_divergence = total_divergence / len(self._divergence_cache)
        weight = min(avg_divergence, 1.0)

        logger.debug(
            "triangulation_weight_computed",
            instance=self._local_instance_id,
            weight=weight,
            peer_count=len(self._divergence_cache),
            avg_divergence=avg_divergence,
        )

        return weight

    def compute_divergence_pressure(
        self,
        local_profile: InstanceDivergenceProfile,
    ) -> DivergencePressure | None:
        """
        Compute divergence pressure when triangulation weight is too low.

        Returns a DivergencePressure signal if weight < 0.4, else None.
        """
        weight = self.compute_triangulation_weight()

        if weight >= _PRESSURE_THRESHOLD:
            return None

        # Pressure magnitude: inversely proportional to weight
        pressure_magnitude = 1.0 - (weight / _PRESSURE_THRESHOLD)

        # Frontier domains: covered by 0-1 instances in the federation
        frontier_domains = [
            domain
            for domain, count in self._federation_domain_counts.items()
            if count <= 1
        ]

        # Saturated domains: covered by 3+ instances
        saturated_domains = [
            domain
            for domain, count in self._federation_domain_counts.items()
            if count >= 3
        ]

        # Find unexplored frontiers for this instance
        local_domains = set(local_profile.domain_coverage)
        unexplored_frontiers = [
            d for d in frontier_domains if d not in local_domains
        ]

        if unexplored_frontiers:
            direction = (
                f"Explore frontier domains: {', '.join(unexplored_frontiers[:3])}. "
                f"Avoid saturated domains: {', '.join(saturated_domains[:3])}."
            )
        elif frontier_domains:
            direction = (
                f"Deepen frontier domain coverage: {', '.join(frontier_domains[:3])}."
            )
        else:
            direction = "Seek novel domains not yet covered by the federation."

        pressure = DivergencePressure(
            instance_id=self._local_instance_id,
            triangulation_weight=weight,
            pressure_magnitude=pressure_magnitude,
            frontier_domains=frontier_domains,
            saturated_domains=saturated_domains,
            recommended_direction=direction,
            generated_at=utc_now(),
        )

        logger.info(
            "divergence_pressure_generated",
            instance=self._local_instance_id,
            weight=weight,
            pressure=pressure_magnitude,
            frontier_count=len(frontier_domains),
            saturated_count=len(saturated_domains),
        )

        return pressure

    def apply_pressure_to_thymos(
        self, pressure: DivergencePressure
    ) -> bool:
        """
        Route divergence pressure to Thymos as a GROWTH drive signal.
        Returns True if delivered, False if Thymos is not wired.
        """
        if self._thymos is None:
            logger.warning(
                "thymos_not_wired",
                instance=self._local_instance_id,
                pressure=pressure.pressure_magnitude,
            )
            return False

        self._thymos.receive_divergence_pressure(pressure)

        logger.info(
            "divergence_pressure_routed_to_thymos",
            instance=self._local_instance_id,
            pressure=pressure.pressure_magnitude,
            weight=pressure.triangulation_weight,
        )

        return True
