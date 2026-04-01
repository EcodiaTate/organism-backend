"""
EcodiaOS - Nexus: Ground Truth Promotion Pipeline

Phase D of epistemic triangulation. Fragments climb through four
epistemic levels, each requiring strictly more evidence:

Level 0 - HYPOTHESIS: Single instance. Could be experience path artifact.
Level 1 - CORROBORATED: independent_source_count >= 2.
Level 2 - TRIANGULATED: confidence > 0.75, diversity > 0.5, sources >= 3.
Level 3 - GROUND_TRUTH_CANDIDATE: confidence > 0.9, diversity > 0.7,
          sources >= 5, survived speciation bridge exchange.
          Anchor memory status. Near-zero decay rate.
Level 4 - EMPIRICAL_INVARIANT: Level 3 + survived Oneiros adversarial
          simulation + survived Evo hypothesis competition.
          Constitutional-level protection via Equor.

The pipeline evaluates fragments and promotes them one level at a time.
Level 3→4 promotion requires external system validation (Oneiros + Evo)
that Nexus orchestrates via dependency-injected protocols.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from primitives.common import utc_now
from systems.nexus.types import (
    EpistemicLevel,
    NexusConfig,
    PromotionDecision,
    ShareableWorldModelFragment,
)

if TYPE_CHECKING:
    from systems.nexus.protocols import (
        EquorProtectionProtocol,
        EvoCompetitionProtocol,
        OneirosAdversarialProtocol,
    )
    from systems.nexus.speciation import SpeciationRegistry

logger = structlog.get_logger("nexus.ground_truth")


class GroundTruthPromotionPipeline:
    """
    Evaluates fragments for epistemic level promotion.

    Each level requires strictly more evidence. The pipeline checks
    the current level and tests promotion criteria for the next level.
    Promotion is always one level at a time - no skipping.
    """

    def __init__(
        self,
        *,
        config: NexusConfig | None = None,
        speciation_registry: SpeciationRegistry | None = None,
        oneiros: OneirosAdversarialProtocol | None = None,
        evo: EvoCompetitionProtocol | None = None,
        equor: EquorProtectionProtocol | None = None,
    ) -> None:
        self._config = config or NexusConfig()
        self._registry = speciation_registry
        self._oneiros = oneiros
        self._evo = evo
        self._equor = equor

        # Track epistemic levels for fragments (fragment_id → level)
        self._fragment_levels: dict[str, EpistemicLevel] = {}

        # Track which fragments have survived bridge exchange
        self._bridge_survivors: set[str] = set()

    def get_level(self, fragment_id: str) -> EpistemicLevel:
        """Return the current epistemic level of a fragment."""
        return self._fragment_levels.get(fragment_id, EpistemicLevel.HYPOTHESIS)

    def set_level(self, fragment_id: str, level: EpistemicLevel) -> None:
        """Manually set a fragment's epistemic level."""
        self._fragment_levels[fragment_id] = level

    def mark_bridge_survivor(self, fragment_id: str) -> None:
        """Mark a fragment as having survived speciation bridge exchange."""
        self._bridge_survivors.add(fragment_id)

    async def evaluate_for_promotion(
        self,
        fragment: ShareableWorldModelFragment,
    ) -> PromotionDecision:
        """
        Evaluate a fragment for promotion to the next epistemic level.

        Checks the current level and tests criteria for level + 1.
        Returns a PromotionDecision with the result and evidence.
        """
        current = self.get_level(fragment.fragment_id)
        tri = fragment.triangulation

        decision = PromotionDecision(
            fragment_id=fragment.fragment_id,
            current_level=current,
            independent_source_count=tri.independent_source_count,
            triangulation_confidence=tri.triangulation_confidence,
            source_diversity=tri.source_diversity_score,
            survived_speciation_bridge=fragment.fragment_id in self._bridge_survivors,
            evaluated_at=utc_now(),
        )

        if current == EpistemicLevel.HYPOTHESIS:
            return self._evaluate_to_corroborated(fragment, decision)

        if current == EpistemicLevel.CORROBORATED:
            return self._evaluate_to_triangulated(fragment, decision)

        if current == EpistemicLevel.TRIANGULATED:
            return self._evaluate_to_ground_truth(fragment, decision)

        if current == EpistemicLevel.GROUND_TRUTH_CANDIDATE:
            return await self._evaluate_to_empirical(fragment, decision)

        # Already at EMPIRICAL_INVARIANT - no further promotion
        decision.reason = "Already at maximum epistemic level."
        return decision

    def _evaluate_to_corroborated(
        self,
        fragment: ShareableWorldModelFragment,
        decision: PromotionDecision,
    ) -> PromotionDecision:
        """Level 0 → 1: Requires independent_source_count >= 2."""
        tri = fragment.triangulation
        required = self._config.level_1_min_sources

        if tri.independent_source_count >= required:
            decision.proposed_level = EpistemicLevel.CORROBORATED
            decision.promoted = True
            decision.reason = (
                f"Corroborated by {tri.independent_source_count} "
                f"independent sources (min: {required})."
            )
            self._fragment_levels[fragment.fragment_id] = EpistemicLevel.CORROBORATED
            logger.info(
                "fragment_promoted",
                fragment_id=fragment.fragment_id,
                from_level=0,
                to_level=1,
                sources=tri.independent_source_count,
            )
        else:
            decision.reason = (
                f"Needs {required} independent sources, "
                f"has {tri.independent_source_count}."
            )

        return decision

    def _evaluate_to_triangulated(
        self,
        fragment: ShareableWorldModelFragment,
        decision: PromotionDecision,
    ) -> PromotionDecision:
        """
        Level 1 → 2: Requires confidence > 0.75, diversity > 0.5,
        sources >= 3.
        """
        tri = fragment.triangulation
        cfg = self._config

        sources_ok = tri.independent_source_count >= cfg.level_2_min_sources
        confidence_ok = tri.triangulation_confidence > cfg.level_2_min_confidence
        diversity_ok = tri.source_diversity_score > cfg.level_2_min_diversity

        if sources_ok and confidence_ok and diversity_ok:
            decision.proposed_level = EpistemicLevel.TRIANGULATED
            decision.promoted = True
            decision.reason = (
                f"Triangulated: {tri.independent_source_count} sources, "
                f"confidence={tri.triangulation_confidence:.3f}, "
                f"diversity={tri.source_diversity_score:.3f}."
            )
            self._fragment_levels[fragment.fragment_id] = EpistemicLevel.TRIANGULATED
            logger.info(
                "fragment_promoted",
                fragment_id=fragment.fragment_id,
                from_level=1,
                to_level=2,
                sources=tri.independent_source_count,
                confidence=tri.triangulation_confidence,
                diversity=tri.source_diversity_score,
            )
        else:
            reasons: list[str] = []
            if not sources_ok:
                reasons.append(
                    f"sources {tri.independent_source_count} < {cfg.level_2_min_sources}"
                )
            if not confidence_ok:
                reasons.append(
                    f"confidence {tri.triangulation_confidence:.3f} <= {cfg.level_2_min_confidence}"
                )
            if not diversity_ok:
                reasons.append(
                    f"diversity {tri.source_diversity_score:.3f} <= {cfg.level_2_min_diversity}"
                )
            decision.reason = f"Not yet triangulated: {'; '.join(reasons)}."

        return decision

    def _evaluate_to_ground_truth(
        self,
        fragment: ShareableWorldModelFragment,
        decision: PromotionDecision,
    ) -> PromotionDecision:
        """
        Level 2 → 3: Requires confidence > 0.9, diversity > 0.7,
        sources >= 5, survived speciation bridge exchange.
        """
        tri = fragment.triangulation
        cfg = self._config

        sources_ok = tri.independent_source_count >= cfg.level_3_min_sources
        confidence_ok = tri.triangulation_confidence > cfg.level_3_min_confidence
        diversity_ok = tri.source_diversity_score > cfg.level_3_min_diversity
        bridge_ok = fragment.fragment_id in self._bridge_survivors

        if sources_ok and confidence_ok and diversity_ok and bridge_ok:
            decision.proposed_level = EpistemicLevel.GROUND_TRUTH_CANDIDATE
            decision.promoted = True
            decision.survived_speciation_bridge = True
            decision.reason = (
                f"Ground truth candidate: {tri.independent_source_count} sources, "
                f"confidence={tri.triangulation_confidence:.3f}, "
                f"diversity={tri.source_diversity_score:.3f}, "
                f"survived bridge exchange."
            )
            self._fragment_levels[fragment.fragment_id] = (
                EpistemicLevel.GROUND_TRUTH_CANDIDATE
            )
            logger.info(
                "fragment_promoted",
                fragment_id=fragment.fragment_id,
                from_level=2,
                to_level=3,
                sources=tri.independent_source_count,
                confidence=tri.triangulation_confidence,
                diversity=tri.source_diversity_score,
            )
        else:
            reasons: list[str] = []
            if not sources_ok:
                reasons.append(
                    f"sources {tri.independent_source_count} < {cfg.level_3_min_sources}"
                )
            if not confidence_ok:
                reasons.append(
                    f"confidence {tri.triangulation_confidence:.3f} <= {cfg.level_3_min_confidence}"
                )
            if not diversity_ok:
                reasons.append(
                    f"diversity {tri.source_diversity_score:.3f} <= {cfg.level_3_min_diversity}"
                )
            if not bridge_ok:
                reasons.append("has not survived speciation bridge exchange")
            decision.reason = (
                f"Not yet ground truth candidate: {'; '.join(reasons)}."
            )

        return decision

    async def _evaluate_to_empirical(
        self,
        fragment: ShareableWorldModelFragment,
        decision: PromotionDecision,
    ) -> PromotionDecision:
        """
        Level 3 → 4: Requires survived Oneiros adversarial test AND
        survived Evo hypothesis competition.

        This is the highest epistemic level. Once confirmed, the
        fragment is routed to Equor for constitutional protection.
        """
        cfg = self._config

        # Adversarial test via Oneiros
        survived_adversarial = False
        if cfg.level_4_adversarial_required:
            if self._oneiros is not None:
                try:
                    survived_adversarial = (
                        await self._oneiros.run_adversarial_test(fragment)
                    )
                except Exception:
                    logger.exception(
                        "adversarial_test_failed",
                        fragment_id=fragment.fragment_id,
                    )
            else:
                logger.warning(
                    "oneiros_not_wired_for_adversarial",
                    fragment_id=fragment.fragment_id,
                )
        else:
            survived_adversarial = True  # Disabled in config

        # Hypothesis competition via Evo
        survived_competition = False
        if cfg.level_4_competition_required:
            if self._evo is not None:
                try:
                    survived_competition = (
                        await self._evo.run_hypothesis_competition(fragment)
                    )
                except Exception:
                    logger.exception(
                        "hypothesis_competition_failed",
                        fragment_id=fragment.fragment_id,
                    )
            else:
                logger.warning(
                    "evo_not_wired_for_competition",
                    fragment_id=fragment.fragment_id,
                )
        else:
            survived_competition = True  # Disabled in config

        decision.survived_adversarial_test = survived_adversarial
        decision.survived_hypothesis_competition = survived_competition

        if survived_adversarial and survived_competition:
            decision.proposed_level = EpistemicLevel.EMPIRICAL_INVARIANT
            decision.promoted = True
            decision.reason = (
                "Empirical invariant confirmed: survived adversarial "
                "simulation and hypothesis competition."
            )
            self._fragment_levels[fragment.fragment_id] = (
                EpistemicLevel.EMPIRICAL_INVARIANT
            )

            logger.info(
                "fragment_promoted_to_empirical_invariant",
                fragment_id=fragment.fragment_id,
                from_level=3,
                to_level=4,
            )

            # Route to Equor for constitutional protection
            await self._request_equor_protection(fragment)
        else:
            reasons: list[str] = []
            if not survived_adversarial:
                reasons.append("failed Oneiros adversarial test")
            if not survived_competition:
                reasons.append("failed Evo hypothesis competition")
            decision.reason = (
                f"Not yet empirical invariant: {'; '.join(reasons)}."
            )

        return decision

    async def _request_equor_protection(
        self,
        fragment: ShareableWorldModelFragment,
    ) -> None:
        """
        Route a Level 4 EMPIRICAL_INVARIANT to Equor for
        constitutional-level protection.
        """
        if self._equor is None:
            logger.warning(
                "equor_not_wired_for_protection",
                fragment_id=fragment.fragment_id,
            )
            return

        evidence = {
            "triangulation_confidence": (
                fragment.triangulation.triangulation_confidence
            ),
            "source_diversity": fragment.triangulation.source_diversity_score,
            "source_count": fragment.triangulation.independent_source_count,
            "survived_adversarial": True,
            "survived_competition": True,
            "epistemic_level": EpistemicLevel.EMPIRICAL_INVARIANT.value,
        }

        try:
            accepted = await self._equor.protect_invariant(fragment, evidence)
            if accepted:
                logger.info(
                    "equor_protection_granted",
                    fragment_id=fragment.fragment_id,
                )
            else:
                logger.warning(
                    "equor_protection_denied",
                    fragment_id=fragment.fragment_id,
                )
        except Exception:
            logger.exception(
                "equor_protection_request_failed",
                fragment_id=fragment.fragment_id,
            )
