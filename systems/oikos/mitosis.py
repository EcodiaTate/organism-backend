"""
EcodiaOS - Mitosis Engine (Phase 16e: Speciation & Fleet Growth)

Reproduction is not a quantity play. Each child instance exists because there
is a SPECIFIC ECOLOGICAL NICHE it is adapted to fill.

The MitosisEngine:
  1. Evaluates parent reproductive fitness (capital surplus × metabolic
     efficiency × runway headroom).
  2. Identifies underserved niches via a hot-swappable BaseMitosisStrategy.
  3. Constructs a SeedConfiguration for the highest-scoring viable niche.
  4. Tracks child health, dividend payments, rescue events, and independence.

Architecture:
  - The engine is stateless logic - all state lives in OikosService's
    EconomicState (child_instances list) and DividendRecord history.
  - The niche-identification strategy is neuroplastic: Evo can hot-swap
    BaseMitosisStrategy subclasses at runtime via NeuroplasticityBus.
  - Actual child spawning is orchestrated by SpawnChildExecutor in Axon.

Constraints (from spec Section VII):
  - min_parent_runway_days: 180 (6 months survival assured)
  - min_seed_capital: $50.00
  - max_seed_pct_of_net_worth: 20%
  - min_parent_efficiency: 1.5 (50% profit margin)
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from systems.oikos.base import BaseMitosisStrategy
from systems.oikos.models import (
    ChildPosition,
    ChildStatus,
    DividendRecord,
    EcologicalNiche,
    EconomicState,
    SeedConfiguration,
)

if TYPE_CHECKING:
    from config import OikosConfig
    from systems.identity.manager import CertificateManager

logger = structlog.get_logger("oikos.mitosis")


# ─── Fitness Check Result ────────────────────────────────────────


class ReproductiveFitness:
    """Result of evaluating whether the parent can reproduce."""

    __slots__ = ("fit", "reasons", "runway_days", "efficiency", "net_worth", "active_children")

    def __init__(
        self,
        *,
        fit: bool,
        reasons: list[str],
        runway_days: Decimal,
        efficiency: Decimal,
        net_worth: Decimal,
        active_children: int,
    ) -> None:
        self.fit = fit
        self.reasons = reasons
        self.runway_days = runway_days
        self.efficiency = efficiency
        self.net_worth = net_worth
        self.active_children = active_children

    def __repr__(self) -> str:
        return (
            f"<ReproductiveFitness fit={self.fit} "
            f"runway={self.runway_days}d efficiency={self.efficiency} "
            f"children={self.active_children}>"
        )


# ─── Default Mitosis Strategy ───────────────────────────────────


class DefaultMitosisStrategy(BaseMitosisStrategy):
    """
    Default niche-identification and scoring strategy.

    Uses simple heuristic scoring: weighted combination of estimated
    profitability, competitive gap, and capability alignment. Evolved
    strategies can replace this with market-data-driven or embedding-based
    approaches via the NeuroplasticityBus.
    """

    @property
    def strategy_name(self) -> str:
        return "default_heuristic"

    def identify_niches(
        self,
        state: EconomicState,
        parent_capabilities: list[str],
    ) -> list[EcologicalNiche]:
        """
        Default: return an empty list. Real niche identification requires
        market scanning (bounty platform analysis, competitor gaps, etc.)
        which will be wired in via Evo hypothesis integration.

        Callers should provide pre-identified niches for scoring instead.
        """
        return []

    def score_niche(
        self,
        niche: EcologicalNiche,
        state: EconomicState,
    ) -> Decimal:
        """
        Score = 0.4*profitability + 0.3*(1-competition) + 0.2*alignment + 0.1*confidence
        """
        # Profitability: estimated_efficiency capped at 3.0, normalised to 0..1
        eff = min(niche.estimated_efficiency, Decimal("3"))
        profitability = (eff / Decimal("3")).quantize(Decimal("0.001"))

        # Competition gap: lower density = better opportunity
        competition_gap = (Decimal("1") - niche.competitive_density).quantize(Decimal("0.001"))

        # Alignment and confidence are already 0..1
        alignment = min(niche.capability_alignment, Decimal("1"))
        confidence = min(niche.confidence, Decimal("1"))

        score = (
            Decimal("0.4") * profitability
            + Decimal("0.3") * competition_gap
            + Decimal("0.2") * alignment
            + Decimal("0.1") * confidence
        ).quantize(Decimal("0.001"))

        return max(Decimal("0"), min(score, Decimal("1")))

    def compute_seed_capital(
        self,
        niche: EcologicalNiche,
        state: EconomicState,
        min_seed: Decimal,
        max_seed_pct: Decimal,
    ) -> Decimal:
        """
        Seed capital = max(min_seed, 3x estimated monthly cost), capped at
        max_seed_pct x net_worth. This gives the child ~3 months of runway
        at projected burn rate.
        """
        three_months_cost = niche.estimated_monthly_cost_usd * Decimal("3")
        ideal = max(min_seed, three_months_cost)

        # Cap at max % of parent net worth
        cap = (state.total_net_worth * max_seed_pct).quantize(Decimal("0.01"))
        return min(ideal, cap).quantize(Decimal("0.01"))


# ─── Mitosis Engine ──────────────────────────────────────────────


class MitosisEngine:
    """
    Evaluates reproductive fitness, scores niches, and builds seed configs.

    This is pure logic - no I/O, no wallet calls, no container orchestration.
    The actual spawning is handled by SpawnChildExecutor in Axon.
    """

    def __init__(self, config: OikosConfig) -> None:
        self._config = config
        self._strategy: BaseMitosisStrategy = DefaultMitosisStrategy()
        self._logger = logger.bind(component="mitosis_engine")
        self._dividend_history: list[DividendRecord] = []
        self._certificate_manager: CertificateManager | None = None

    # --- Phase 16g: Certificate Manager wiring ---

    def set_certificate_manager(self, cert_mgr: CertificateManager) -> None:
        """Wire CertificateManager for birth certificate issuance."""
        self._certificate_manager = cert_mgr
        self._logger.info("certificate_manager_wired_to_mitosis")

    # ─── Strategy hot-swap (neuroplastic) ──────────────────────

    @property
    def strategy(self) -> BaseMitosisStrategy:
        return self._strategy

    def set_strategy(self, strategy: BaseMitosisStrategy) -> None:
        old = self._strategy.strategy_name
        self._strategy = strategy
        self._logger.info("mitosis_strategy_evolved", old=old, new=strategy.strategy_name)

    # ─── Reproductive Fitness Check ────────────────────────────

    def evaluate_fitness(
        self,
        state: EconomicState,
        max_children_override: int | None = None,
    ) -> ReproductiveFitness:
        """
        Check all preconditions for reproduction. ALL must pass.

        Conditions:
          1. Parent runway >= min_parent_runway_days (180)
          2. Parent metabolic_efficiency >= min_parent_efficiency (1.5)
          3. Parent is metabolically positive (net_income_7d > 0)
          4. Active children < max_children (dynamic: max(5, floor(net_worth/1000)))
          5. Net worth > min_seed_capital (can actually afford to seed)

        Parameters
        ----------
        max_children_override
            If provided, overrides the configured max_children with a dynamic
            cap (e.g. from ``MitosisFleetService.compute_dynamic_max_children``).
        """
        reasons: list[str] = []
        cfg = self._config

        min_runway = Decimal(str(cfg.mitosis_min_parent_runway_days))
        min_efficiency = Decimal(str(cfg.mitosis_min_parent_efficiency))
        min_seed = Decimal(str(cfg.mitosis_min_seed_capital))
        max_children = max_children_override if max_children_override is not None else cfg.mitosis_max_children

        # Count active (non-dead, non-independent) children
        active_children = sum(
            1 for c in state.child_instances
            if c.status not in (ChildStatus.DEAD, ChildStatus.INDEPENDENT)
        )

        if state.runway_days < min_runway:
            reasons.append(
                f"Runway {state.runway_days}d < required {min_runway}d"
            )

        if state.metabolic_efficiency < min_efficiency:
            reasons.append(
                f"Efficiency {state.metabolic_efficiency} < required {min_efficiency}"
            )

        if not state.is_metabolically_positive:
            reasons.append("Not metabolically positive (net_income_7d <= 0)")

        if active_children >= max_children:
            reasons.append(
                f"Active children ({active_children}) >= max ({max_children})"
            )

        if state.total_net_worth < min_seed:
            reasons.append(
                f"Net worth ${state.total_net_worth} < min seed ${min_seed}"
            )

        fit = len(reasons) == 0

        self._logger.debug(
            "reproductive_fitness_evaluated",
            fit=fit,
            runway_days=str(state.runway_days),
            efficiency=str(state.metabolic_efficiency),
            active_children=active_children,
            rejection_reasons=reasons or None,
        )

        return ReproductiveFitness(
            fit=fit,
            reasons=reasons,
            runway_days=state.runway_days,
            efficiency=state.metabolic_efficiency,
            net_worth=state.total_net_worth,
            active_children=active_children,
        )

    # ─── Niche Selection ───────────────────────────────────────

    def select_best_niche(
        self,
        state: EconomicState,
        candidate_niches: list[EcologicalNiche] | None = None,
        parent_capabilities: list[str] | None = None,
    ) -> EcologicalNiche | None:
        """
        Identify and return the highest-scoring viable niche, or None.

        If candidate_niches is provided, scores those directly. Otherwise
        delegates to the active strategy's identify_niches().
        """
        min_score = Decimal(str(self._config.mitosis_min_niche_score))

        niches = candidate_niches or self._strategy.identify_niches(
            state=state,
            parent_capabilities=parent_capabilities or [],
        )

        if not niches:
            self._logger.debug("no_candidate_niches")
            return None

        # Score and rank
        scored: list[tuple[Decimal, EcologicalNiche]] = []
        for niche in niches:
            score = self._strategy.score_niche(niche, state)
            if score >= min_score:
                scored.append((score, niche))

        if not scored:
            self._logger.debug(
                "no_viable_niches",
                candidates=len(niches),
                min_score=str(min_score),
            )
            return None

        # Sort descending by score
        scored.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best_niche = scored[0]

        self._logger.info(
            "niche_selected",
            niche=best_niche.name,
            score=str(best_score),
            candidates=len(niches),
            viable=len(scored),
        )
        return best_niche

    # ─── Seed Configuration Assembly ───────────────────────────

    def build_seed_config(
        self,
        state: EconomicState,
        niche: EcologicalNiche,
    ) -> SeedConfiguration | None:
        """
        Construct the complete birth-packet for a child in the given niche.

        Returns None if the computed seed capital doesn't meet the minimum
        or would exceed the maximum percentage of parent net worth.
        """
        cfg = self._config
        min_seed = Decimal(str(cfg.mitosis_min_seed_capital))
        max_seed_pct = Decimal(str(cfg.mitosis_max_seed_pct_of_net_worth))
        dividend_rate = Decimal(str(cfg.mitosis_default_dividend_rate))

        seed_capital = self._strategy.compute_seed_capital(
            niche=niche,
            state=state,
            min_seed=min_seed,
            max_seed_pct=max_seed_pct,
        )

        # Validate bounds
        if seed_capital < min_seed:
            self._logger.warning(
                "seed_capital_below_minimum",
                computed=str(seed_capital),
                min_seed=str(min_seed),
            )
            return None

        max_amount = (state.total_net_worth * max_seed_pct).quantize(Decimal("0.01"))
        if seed_capital > max_amount:
            self._logger.warning(
                "seed_capital_exceeds_cap",
                computed=str(seed_capital),
                cap=str(max_amount),
                net_worth=str(state.total_net_worth),
            )
            return None

        # Ensure parent can actually afford this without going below min runway
        post_seed_liquid = state.liquid_balance - seed_capital
        if state.basal_metabolic_rate.usd_per_day > Decimal("0"):
            post_seed_runway = post_seed_liquid / state.basal_metabolic_rate.usd_per_day
        else:
            post_seed_runway = Decimal("Infinity")

        min_runway = Decimal(str(cfg.mitosis_min_parent_runway_days))
        if post_seed_runway < min_runway:
            self._logger.warning(
                "seed_would_breach_runway",
                post_seed_runway=str(post_seed_runway),
                min_runway=str(min_runway),
            )
            return None

        config = SeedConfiguration(
            parent_instance_id=state.instance_id,
            niche=niche,
            seed_capital_usd=seed_capital,
            dividend_rate=dividend_rate,
            child_config_overrides={
                "specialisation": niche.name,
                "niche_description": niche.description,
            },
        )

        # Phase 16g: Sign a birth certificate for the child
        if self._certificate_manager is not None:
            birth_validity = getattr(
                self._config, "certificate_birth_validity_days", 7,
            )
            birth_cert = self._certificate_manager.issue_birth_certificate(
                child_instance_id=config.child_instance_id,
                validity_days=birth_validity,
            )
            if birth_cert is not None:
                config.birth_certificate_json = birth_cert.model_dump_json()
                self._logger.info(
                    "birth_certificate_attached",
                    child_id=config.child_instance_id,
                    cert_id=birth_cert.certificate_id,
                    validity_days=birth_validity,
                )
            else:
                self._logger.warning(
                    "birth_certificate_skipped",
                    child_id=config.child_instance_id,
                    reason="certificate manager could not issue",
                )

        self._logger.info(
            "seed_config_built",
            child_id=config.child_instance_id,
            niche=niche.name,
            seed_capital=str(seed_capital),
            dividend_rate=str(dividend_rate),
            has_birth_cert=bool(config.birth_certificate_json),
        )
        return config

    # ─── Dividend Tracking ─────────────────────────────────────

    def record_dividend(self, record: DividendRecord) -> None:
        """Record a dividend payment from a child."""
        self._dividend_history.append(record)
        self._logger.info(
            "dividend_recorded",
            child=record.child_instance_id,
            amount=str(record.amount_usd),
            rate=str(record.dividend_rate_applied),
        )

    def total_dividends_from(self, child_instance_id: str) -> Decimal:
        """Sum of all dividends received from a specific child."""
        return sum(
            (r.amount_usd for r in self._dividend_history if r.child_instance_id == child_instance_id),
            Decimal("0"),
        )

    @property
    def dividend_history(self) -> list[DividendRecord]:
        return list(self._dividend_history)

    # ─── Child Health Evaluation ───────────────────────────────

    def evaluate_child_health(
        self,
        child: ChildPosition,
    ) -> ChildStatus:
        """
        Determine the current lifecycle status of a child based on its metrics.

        Returns the status the child SHOULD be in (caller updates the model).
        """
        cfg = self._config

        if child.status == ChildStatus.DEAD:
            return ChildStatus.DEAD

        # Check independence
        if child.is_independent:
            return ChildStatus.INDEPENDENT

        # Check struggling
        struggling_threshold = Decimal(str(cfg.mitosis_child_struggling_runway_days))
        if child.current_runway_days < struggling_threshold:
            if child.is_rescuable:
                return ChildStatus.STRUGGLING
            return ChildStatus.DEAD  # Unrescuable - graceful death

        return ChildStatus.ALIVE

    # ─── Full Mitosis Evaluation (Convenience) ─────────────────

    def evaluate(
        self,
        state: EconomicState,
        candidate_niches: list[EcologicalNiche] | None = None,
        parent_capabilities: list[str] | None = None,
    ) -> SeedConfiguration | None:
        """
        Run the full mitosis evaluation pipeline:
          1. Check reproductive fitness
          2. Select best niche
          3. Build seed configuration

        Returns a SeedConfiguration if reproduction is viable, else None.
        """
        fitness = self.evaluate_fitness(state)
        if not fitness.fit:
            self._logger.info(
                "reproduction_not_viable",
                reasons=fitness.reasons,
            )
            return None

        niche = self.select_best_niche(
            state=state,
            candidate_niches=candidate_niches,
            parent_capabilities=parent_capabilities,
        )
        if niche is None:
            self._logger.info("reproduction_blocked_no_niche")
            return None

        return self.build_seed_config(state=state, niche=niche)
