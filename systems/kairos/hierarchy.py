"""
EcodiaOS -- Kairos: Causal Hierarchy

Three tiers of causal knowledge, ordered by generative power:

TIER 1: DOMAIN INVARIANTS
    Hold within a single broad domain. High confidence, limited transfer.

TIER 2: CROSS-DOMAIN INVARIANTS
    Hold across multiple distinct domains. High transfer value.

TIER 3: SUBSTRATE-INDEPENDENT INVARIANTS
    Hold regardless of substrate. The deepest layer.
    Finding one is an architectural event for the world model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import structlog

from systems.kairos.types import (
    ApplicableDomain,
    CausalInvariant,
    CausalInvariantTier,
    CausalRule,
    InvarianceTestResult,
    KairosConfig,
)

logger = structlog.get_logger("kairos.hierarchy")


class CausalHierarchy:
    """
    Manages the three-tier causal invariant hierarchy.

    Invariants are promoted based on domain coverage and substrate diversity:
    - 1 domain: Tier 1
    - 2-3 domains: Tier 2
    - 4+ domains spanning 3+ substrates: Tier 3
    """

    def __init__(self, config: KairosConfig | None = None) -> None:
        self._config = config or KairosConfig()
        self._tier1: list[CausalInvariant] = []
        self._tier2: list[CausalInvariant] = []
        self._tier3: list[CausalInvariant] = []
        self._on_tier3_callbacks: list[Callable[[CausalInvariant], Any]] = []

    def on_tier3_discovered(self, callback: Callable[[CausalInvariant], Any]) -> None:
        """Register a callback for Tier 3 promotion events."""
        self._on_tier3_callbacks.append(callback)

    @property
    def tier1_invariants(self) -> list[CausalInvariant]:
        return list(self._tier1)

    @property
    def tier2_invariants(self) -> list[CausalInvariant]:
        return list(self._tier2)

    @property
    def tier3_invariants(self) -> list[CausalInvariant]:
        return list(self._tier3)

    @property
    def total_count(self) -> int:
        return len(self._tier1) + len(self._tier2) + len(self._tier3)

    def create_from_rule(
        self,
        rule: CausalRule,
        invariance_result: InvarianceTestResult,
        domain: str = "",
        substrate: str = "",
    ) -> CausalInvariant:
        """
        Create a new causal invariant from a confirmed rule and its invariance test.
        Starts at Tier 1 and may be promoted later.
        """
        applicable_domains = [
            ApplicableDomain(
                domain=domain or rule.domain or "general",
                substrate=substrate,
                hold_rate=invariance_result.hold_rate,
                observation_count=rule.observation_count,
            )
        ]

        invariant = CausalInvariant(
            tier=CausalInvariantTier.TIER_1_DOMAIN,
            abstract_form=f"{rule.cause_variable} causes {rule.effect_variable}",
            concrete_instances=[rule.id],
            applicable_domains=applicable_domains,
            invariance_hold_rate=invariance_result.hold_rate,
            scope_conditions=invariance_result.scope_conditions,
            source_rule_id=rule.id,
        )

        self._place(invariant)

        logger.info(
            "invariant_created",
            invariant_id=invariant.id,
            tier=invariant.tier,
            cause=rule.cause_variable,
            effect=rule.effect_variable,
            hold_rate=round(invariance_result.hold_rate, 3),
        )

        return invariant

    def add_domain(
        self,
        invariant_id: str,
        domain: str,
        substrate: str,
        hold_rate: float,
        observation_count: int = 0,
    ) -> CausalInvariant | None:
        """
        Register a new domain where an existing invariant holds.
        May trigger tier promotion.
        """
        invariant = self._find(invariant_id)
        if invariant is None:
            return None

        existing_domains = {d.domain for d in invariant.applicable_domains}
        if domain in existing_domains:
            return invariant

        invariant.applicable_domains.append(
            ApplicableDomain(
                domain=domain,
                substrate=substrate,
                hold_rate=hold_rate,
                observation_count=observation_count,
            )
        )

        old_tier = invariant.tier
        new_tier = self._compute_tier(invariant)

        if new_tier != old_tier:
            self._remove(invariant)
            invariant.tier = new_tier
            self._place(invariant)

            logger.info(
                "invariant_promoted",
                invariant_id=invariant.id,
                from_tier=old_tier,
                to_tier=new_tier,
                domain_count=invariant.domain_count,
                substrate_count=invariant.substrate_count,
            )

            if new_tier == CausalInvariantTier.TIER_3_SUBSTRATE:
                self._fire_tier3_callbacks(invariant)

        return invariant

    def promote_if_eligible(self, invariant_id: str) -> CausalInvariantTier | None:
        """Re-evaluate tier placement for an invariant. Returns new tier or None."""
        invariant = self._find(invariant_id)
        if invariant is None:
            return None

        new_tier = self._compute_tier(invariant)
        if new_tier != invariant.tier:
            self._remove(invariant)
            invariant.tier = new_tier
            self._place(invariant)

            if new_tier == CausalInvariantTier.TIER_3_SUBSTRATE:
                self._fire_tier3_callbacks(invariant)

            return new_tier

        return None

    def get_all(self) -> list[CausalInvariant]:
        """Return all invariants across all tiers."""
        return self._tier1 + self._tier2 + self._tier3

    def get_by_tier(self, tier: CausalInvariantTier) -> list[CausalInvariant]:
        """Return invariants for a specific tier."""
        if tier == CausalInvariantTier.TIER_1_DOMAIN:
            return list(self._tier1)
        if tier == CausalInvariantTier.TIER_2_CROSS_DOMAIN:
            return list(self._tier2)
        if tier == CausalInvariantTier.TIER_3_SUBSTRATE:
            return list(self._tier3)
        return []

    def summary(self) -> dict[str, Any]:
        """Return a summary of the hierarchy state."""
        return {
            "tier1_count": len(self._tier1),
            "tier2_count": len(self._tier2),
            "tier3_count": len(self._tier3),
            "total": self.total_count,
            "tier3_invariants": [
                {
                    "id": inv.id,
                    "abstract_form": inv.abstract_form,
                    "domain_count": inv.domain_count,
                    "substrate_count": inv.substrate_count,
                    "hold_rate": inv.invariance_hold_rate,
                }
                for inv in self._tier3
            ],
        }

    # --- Internal ---

    def _compute_tier(self, invariant: CausalInvariant) -> CausalInvariantTier:
        """
        Determine the correct tier based on domain/substrate coverage
        and Phase C distillation requirements.

        Tier 3 (Phase C) requires:
        - domain_count >= tier3_min_domains (4)
        - substrate_count >= tier3_min_substrates (3)
        - hold_rate > 0.95
        - distilled AND minimal AND not tautological
        - 5+ contexts
        """
        domain_count = invariant.domain_count
        substrate_count = invariant.substrate_count

        tier3_domain_ok = (
            domain_count >= self._config.tier3_min_domains
            and substrate_count >= self._config.tier3_min_substrates
        )
        tier3_distillation_ok = (
            invariant.distilled
            and invariant.is_minimal
            and not invariant.is_tautological
            and invariant.invariance_hold_rate >= 0.95
        )

        if tier3_domain_ok and tier3_distillation_ok:
            return CausalInvariantTier.TIER_3_SUBSTRATE
        if domain_count >= self._config.tier2_min_domains:
            return CausalInvariantTier.TIER_2_CROSS_DOMAIN
        return CausalInvariantTier.TIER_1_DOMAIN

    def _fire_tier3_callbacks(self, invariant: CausalInvariant) -> None:
        """Fire all registered Tier 3 discovery callbacks."""
        logger.info(
            "tier3_invariant_discovered",
            invariant_id=invariant.id,
            abstract_form=invariant.abstract_form[:80],
            domain_count=invariant.domain_count,
            substrate_count=invariant.substrate_count,
        )
        for callback in self._on_tier3_callbacks:
            try:
                callback(invariant)
            except Exception:
                logger.exception(
                    "tier3_callback_failed",
                    invariant_id=invariant.id,
                )

    def _place(self, invariant: CausalInvariant) -> None:
        """Place an invariant in the correct tier list."""
        if invariant.tier == CausalInvariantTier.TIER_1_DOMAIN:
            self._tier1.append(invariant)
        elif invariant.tier == CausalInvariantTier.TIER_2_CROSS_DOMAIN:
            self._tier2.append(invariant)
        elif invariant.tier == CausalInvariantTier.TIER_3_SUBSTRATE:
            self._tier3.append(invariant)

    def _remove(self, invariant: CausalInvariant) -> None:
        """Remove an invariant from its current tier list."""
        for tier_list in (self._tier1, self._tier2, self._tier3):
            tier_list[:] = [inv for inv in tier_list if inv.id != invariant.id]

    def _find(self, invariant_id: str) -> CausalInvariant | None:
        """Find an invariant by ID across all tiers."""
        for inv in self._tier1 + self._tier2 + self._tier3:
            if inv.id == invariant_id:
                return inv
        return None
