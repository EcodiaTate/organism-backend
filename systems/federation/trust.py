"""
EcodiaOS — Federation Trust Model

Trust between EOS instances starts at zero and builds through successful
interaction. Violations cost 3x; a privacy breach resets trust to zero
immediately.

The trust model implements graduated trust levels (NONE → ACQUAINTANCE →
COLLEAGUE → PARTNER → ALLY) with score thresholds. Trust also decays over
time for inactive links, preventing stale trust from persisting.

This is not a reputation system — it is a direct relationship model.
Each link tracks its own trust independently. Trust is earned by this
specific pair of instances, not inherited from the network.
"""

from __future__ import annotations

from typing import Any

import structlog

from primitives.common import utc_now
from primitives.federation import (
    TRUST_THRESHOLDS,
    VIOLATION_MULTIPLIER,
    FederationInteraction,
    FederationLink,
    InteractionOutcome,
    TrustLevel,
    ViolationType,
)

logger = structlog.get_logger("systems.federation.trust")


class TrustManager:
    """
    Manages trust scoring and level transitions for federation links.

    Trust is a float score that maps to discrete TrustLevel values via
    thresholds. Successful interactions increment the score; failures
    decrement at a multiplied rate. Privacy breaches are catastrophic
    — immediate reset to zero.

    Trust also decays over time for inactive links, modelling the
    realistic principle that trust requires ongoing interaction.
    """

    def __init__(
        self,
        trust_decay_enabled: bool = True,
        trust_decay_rate_per_day: float = 0.1,
        max_trust_level: TrustLevel = TrustLevel.ALLY,
    ) -> None:
        self._trust_decay_enabled = trust_decay_enabled
        self._trust_decay_rate_per_day = trust_decay_rate_per_day
        self._max_trust_level = max_trust_level
        self._logger = logger.bind(component="trust_manager")

    # ─── Trust Updates ──────────────────────────────────────────────

    def update_trust(
        self,
        link: FederationLink,
        interaction: FederationInteraction,
    ) -> TrustLevel:
        """
        Update trust score and level based on an interaction outcome.

        Rules (from the spec):
          - Successful interactions: +trust_value
          - Failed interactions: -trust_value (1x)
          - Violations: -trust_value * 3x
          - Privacy breach violations: instant reset to 0, level to NONE

        Returns the new TrustLevel.
        """
        previous_level = link.trust_level
        previous_score = link.trust_score

        if interaction.outcome == InteractionOutcome.SUCCESSFUL:
            link.trust_score += interaction.trust_value
            link.successful_interactions += 1

        elif interaction.outcome == InteractionOutcome.FAILED:
            link.trust_score = max(0.0, link.trust_score - interaction.trust_value)
            link.failed_interactions += 1

        elif interaction.outcome == InteractionOutcome.VIOLATION:
            link.violation_count += 1
            link.failed_interactions += 1

            if interaction.violation_type == ViolationType.PRIVACY_BREACH:
                # Privacy breaches are catastrophic — instant zero
                link.trust_score = 0.0
                link.trust_level = TrustLevel.NONE
                self._logger.warning(
                    "trust_privacy_breach_reset",
                    link_id=link.id,
                    remote_id=link.remote_instance_id,
                )
                return TrustLevel.NONE
            else:
                # Other violations cost 3x
                penalty = interaction.trust_value * VIOLATION_MULTIPLIER
                link.trust_score = max(0.0, link.trust_score - penalty)

        elif interaction.outcome == InteractionOutcome.TIMEOUT:
            # Timeouts are mild — half penalty
            link.trust_score = max(
                0.0, link.trust_score - interaction.trust_value * 0.5
            )
            link.failed_interactions += 1

        # Recompute trust level from score
        link.trust_level = self._score_to_level(link.trust_score)

        # Cap at max allowed level
        if link.trust_level.value > self._max_trust_level.value:
            link.trust_level = self._max_trust_level

        # Update communication timestamp
        link.last_communication = utc_now()

        if link.trust_level != previous_level:
            self._logger.info(
                "trust_level_changed",
                link_id=link.id,
                remote_id=link.remote_instance_id,
                previous=previous_level.name,
                new=link.trust_level.name,
                score=round(link.trust_score, 2),
                delta=round(link.trust_score - previous_score, 2),
            )

        return link.trust_level

    # ─── Trust Decay ────────────────────────────────────────────────

    def apply_decay(self, link: FederationLink) -> None:
        """
        Apply time-based trust decay for inactive links.

        Trust decays linearly based on days since last communication.
        This prevents stale trust from persisting and encourages
        ongoing interaction.
        """
        if not self._trust_decay_enabled:
            return

        if link.last_communication is None:
            return

        now = utc_now()
        delta = now - link.last_communication
        days_inactive = delta.total_seconds() / 86400.0

        if days_inactive <= 1.0:
            return  # No decay within 24 hours

        decay = days_inactive * self._trust_decay_rate_per_day
        previous_score = link.trust_score
        link.trust_score = max(0.0, link.trust_score - decay)
        link.trust_level = self._score_to_level(link.trust_score)

        if link.trust_level.value < self._score_to_level(previous_score).value:
            self._logger.info(
                "trust_decayed",
                link_id=link.id,
                remote_id=link.remote_instance_id,
                days_inactive=round(days_inactive, 1),
                decay_applied=round(decay, 2),
                new_level=link.trust_level.name,
            )

    # ─── Trust Queries ──────────────────────────────────────────────

    def can_share_knowledge_type(
        self, link: FederationLink, knowledge_type: str
    ) -> bool:
        """Check if the current trust level permits sharing a knowledge type."""
        from primitives.federation import SHARING_PERMISSIONS, KnowledgeType

        try:
            kt = KnowledgeType(knowledge_type)
        except ValueError:
            return False

        permitted = SHARING_PERMISSIONS.get(link.trust_level, [])
        return kt in permitted

    def can_coordinate(self, link: FederationLink) -> bool:
        """Check if the link has sufficient trust for coordinated action."""
        return link.trust_level.value >= TrustLevel.COLLEAGUE.value

    def mean_trust(self, links: list[FederationLink]) -> float:
        """Compute mean trust score across all active links."""
        if not links:
            return 0.0
        return sum(lnk.trust_score for lnk in links) / len(links)

    # ─── Internal ───────────────────────────────────────────────────

    @staticmethod
    def _score_to_level(score: float) -> TrustLevel:
        """
        Map a trust score to the highest applicable TrustLevel.

        Thresholds (from spec):
          ACQUAINTANCE: 5
          COLLEAGUE: 20
          PARTNER: 50
          ALLY: 100
        """
        # Iterate from highest to lowest threshold
        for level in sorted(TRUST_THRESHOLDS.keys(), key=lambda lvl: lvl.value, reverse=True):
            if score >= TRUST_THRESHOLDS[level]:
                return level
        return TrustLevel.NONE

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "decay_enabled": self._trust_decay_enabled,
            "decay_rate_per_day": self._trust_decay_rate_per_day,
            "max_trust_level": self._max_trust_level.name,
        }
