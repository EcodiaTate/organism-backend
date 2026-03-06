"""
EcodiaOS -- Architecture-Level EFE Scorer

Applies active inference principles to structural evolution proposals.
Each proposed change gets an Expected Free Energy score measuring:

  EFE = -(pragmatic_value + epistemic_value) + complexity_penalty

Where:
  - pragmatic_value:  fraction of historical failures the change addresses
  - epistemic_value:  mutual information between the proposed change and
                      historical episode outcomes (entropy reduction)
  - complexity_penalty: cost of added parameters, code paths, state space

Simula generates multiple proposals per consolidation cycle. This scorer
ranks them so the organism prioritises high-value changes the same way
Nova prioritises actions: pick most-negative EFE first.

Design choices:
  - Zero LLM tokens for scoring (pure analytics from Neo4j history)
  - Complexity estimated from ChangeSpec fields (parameter count, system count)
  - Calibration via feedback loop: predicted EFE vs actual improvement
  - Penalty weight is Evo-tunable via TUNABLE_PARAMETERS

Integration:
  - Evo Phase 8 → score proposals before submitting to Simula
  - SimulaService.process_proposal() → attach EFE before pipeline
  - Equor review → consume ranked queue, auto-approve high-EFE proposals
  - Neo4j → persist ProposalEFE nodes for calibration feedback
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import structlog

from systems.simula.evolution_types import (
    ChangeCategory,
    EFECalibrationRecord,
    EvolutionProposal,
    ProposalEFEBreakdown,
    RankedProposalQueue,
)

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from systems.simula.analytics import EvolutionAnalyticsEngine
    from systems.simula.history import EvolutionHistoryManager

logger = structlog.get_logger().bind(system="simula.architecture_efe")

# Default penalty weight for complexity (Evo-tunable)
_DEFAULT_COMPLEXITY_PENALTY_WEIGHT: float = 0.15

# EFE confidence threshold for Equor auto-approval
_AUTO_APPROVE_EFE_THRESHOLD: float = -0.35

# Minimum proposals needed for meaningful mutual information estimate
_MIN_EPISODES_FOR_MI: int = 5

# Category-level complexity heuristics (base complexity cost 0.0-1.0)
_CATEGORY_BASE_COMPLEXITY: dict[ChangeCategory, float] = {
    ChangeCategory.ADJUST_BUDGET: 0.05,
    ChangeCategory.ADD_PATTERN_DETECTOR: 0.20,
    ChangeCategory.ADD_INPUT_CHANNEL: 0.25,
    ChangeCategory.ADD_EXECUTOR: 0.30,
    ChangeCategory.MODIFY_CONTRACT: 0.45,
    ChangeCategory.MODIFY_CYCLE_TIMING: 0.35,
    ChangeCategory.CHANGE_CONSOLIDATION: 0.40,
    ChangeCategory.ADD_SYSTEM_CAPABILITY: 0.60,
}


class ArchitectureEFEScorer:
    """
    Scores evolution proposals using architecture-level Expected Free Energy.

    Computes EFE from historical data (Neo4j evolution records + analytics)
    without LLM calls. The three components mirror Nova's EFE decomposition
    but operate on structural changes rather than action policies.

    Calibration feedback: after a proposal is applied, compare predicted
    EFE with actual outcome to reduce future prediction error.
    """

    def __init__(
        self,
        history: EvolutionHistoryManager | None = None,
        analytics: EvolutionAnalyticsEngine | None = None,
        complexity_penalty_weight: float = _DEFAULT_COMPLEXITY_PENALTY_WEIGHT,
    ) -> None:
        self._history = history
        self._analytics = analytics
        self._penalty_weight = complexity_penalty_weight
        self._log = logger

        # Calibration state: running mean of EFE prediction errors
        self._calibration_bias: float = 0.0
        self._calibration_count: int = 0
        # Neo4j client for parent genome calibration loading
        self._neo4j: Neo4jClient | None = None

    @property
    def complexity_penalty_weight(self) -> float:
        return self._penalty_weight

    def update_penalty_weight(self, new_weight: float) -> None:
        """Called by Evo when learning adjusts the complexity penalty."""
        self._penalty_weight = max(0.0, min(1.0, new_weight))
        self._log.info("penalty_weight_updated", new_weight=self._penalty_weight)

    # ─── Core Scoring ────────────────────────────────────────────────────────

    async def score_proposal(
        self, proposal: EvolutionProposal,
    ) -> ProposalEFEBreakdown:
        """
        Compute architecture-level EFE for a single proposal.

        EFE = -(pragmatic_value + epistemic_value) + (complexity * penalty_weight)

        Lower (more negative) = better expected value for the organism.
        """
        pragmatic = await self._compute_pragmatic_value(proposal)
        epistemic = await self._compute_epistemic_value(proposal)
        complexity = self._compute_complexity(proposal)

        efe_raw = -(pragmatic + epistemic)
        efe_penalised = efe_raw + (complexity * self._penalty_weight)

        # Apply calibration bias correction
        efe_penalised -= self._calibration_bias

        confidence = self._estimate_confidence(proposal, pragmatic, epistemic)

        breakdown = ProposalEFEBreakdown(
            pragmatic_value=round(pragmatic, 4),
            epistemic_value=round(epistemic, 4),
            complexity_penalty=round(complexity, 4),
            efe_raw=round(efe_raw, 4),
            efe_penalised=round(efe_penalised, 4),
            confidence=round(confidence, 3),
            reasoning=(
                f"pragmatic={pragmatic:.3f} (failures addressed), "
                f"epistemic={epistemic:.3f} (entropy reduction), "
                f"complexity={complexity:.3f} (penalty_w={self._penalty_weight:.2f}), "
                f"calibration_bias={self._calibration_bias:.4f} → "
                f"EFE={efe_penalised:.3f}"
            ),
        )

        self._log.info(
            "proposal_efe_scored",
            proposal_id=proposal.id,
            category=proposal.category.value,
            efe=efe_penalised,
            pragmatic=pragmatic,
            epistemic=epistemic,
            complexity=complexity,
            confidence=confidence,
        )

        return breakdown

    async def score_and_rank(
        self, proposals: list[EvolutionProposal],
    ) -> RankedProposalQueue:
        """
        Score all proposals and return a ranked queue (most negative EFE first).

        Attaches efe_score to each proposal and returns them sorted.
        """
        if not proposals:
            return RankedProposalQueue()

        breakdowns: list[ProposalEFEBreakdown] = []
        for proposal in proposals:
            breakdown = await self.score_proposal(proposal)
            proposal.efe_score = breakdown.efe_penalised
            breakdowns.append(breakdown)

        # Sort by EFE ascending (most negative = best)
        paired = list(zip(proposals, breakdowns, strict=True))
        paired.sort(key=lambda x: x[1].efe_penalised)

        sorted_proposals = [p for p, _ in paired]
        sorted_breakdowns = [b for _, b in paired]
        sorted_scores = [b.efe_penalised for b in sorted_breakdowns]

        queue = RankedProposalQueue(
            proposals=sorted_proposals,
            scores=sorted_scores,
            breakdowns=sorted_breakdowns,
            recommended=sorted_proposals[0] if sorted_proposals else None,
        )

        self._log.info(
            "proposals_ranked_by_efe",
            count=len(proposals),
            top_efe=sorted_scores[0] if sorted_scores else None,
            bottom_efe=sorted_scores[-1] if sorted_scores else None,
        )

        return queue

    # ─── Pragmatic Value ─────────────────────────────────────────────────────

    async def _compute_pragmatic_value(
        self, proposal: EvolutionProposal,
    ) -> float:
        """
        Pragmatic value = fraction of historical failures this change addresses.

        Uses evolution history to find:
        1. Total rolled-back / failed proposals in same category
        2. Whether this proposal's description addresses known failure patterns
        3. Evidence strength from supporting hypotheses

        Returns 0.0-1.0 scale.
        """
        if self._analytics is None:
            return self._pragmatic_from_evidence(proposal)

        try:
            analytics = await self._analytics.compute_analytics()
        except Exception:
            return self._pragmatic_from_evidence(proposal)

        category_key = proposal.category.value
        category_rate = analytics.category_rates.get(category_key)

        # Component 1: Category success rate inversion
        # High rollback rate in this category → higher pragmatic value for fixes
        if category_rate is not None and category_rate.total >= 3:
            rollback_rate = category_rate.rollback_rate
            # A change in a high-failure category has more pragmatic value
            category_signal = min(0.4, rollback_rate * 0.8)
        else:
            category_signal = 0.1

        # Component 2: Evidence strength from supporting hypotheses
        evidence_signal = self._pragmatic_from_evidence(proposal)

        # Component 3: Semantic similarity to past failures (if history available)
        similarity_signal = await self._failure_similarity_signal(proposal)

        # Weighted combination, capped at 1.0
        pragmatic = min(1.0, category_signal + evidence_signal * 0.5 + similarity_signal)
        return pragmatic

    def _pragmatic_from_evidence(self, proposal: EvolutionProposal) -> float:
        """Estimate pragmatic value from evidence count (logarithmic scaling)."""
        count = len(proposal.evidence)
        if count == 0:
            return 0.1
        # 1 item = 0.2, 5 items = 0.45, 10+ items = 0.6
        return min(0.7, 0.1 + 0.2 * math.log1p(count))

    async def _failure_similarity_signal(
        self, proposal: EvolutionProposal,
    ) -> float:
        """
        Check if this proposal addresses patterns similar to past rollbacks.
        Uses semantic similarity search on evolution records.
        Returns 0.0-0.3 scale.
        """
        if self._history is None:
            return 0.0

        try:
            similar = await self._history.find_similar_records(
                description=proposal.description,
                top_k=5,
                min_score=0.6,
            )
        except Exception:
            return 0.0

        if not similar:
            return 0.0

        # Count how many similar past proposals were rolled back
        rolled_back_similar = sum(
            1 for record, _score in similar if record.rolled_back
        )
        total_similar = len(similar)

        if total_similar == 0:
            return 0.0

        # Higher rollback rate among similar proposals → this change is more needed
        failure_fraction = rolled_back_similar / total_similar
        return min(0.3, failure_fraction * 0.3)

    # ─── Epistemic Value ─────────────────────────────────────────────────────

    async def _compute_epistemic_value(
        self, proposal: EvolutionProposal,
    ) -> float:
        """
        Epistemic value = entropy reduction from the proposed change.

        Approximated via mutual information between the proposed change
        and historical episode outcomes. A change that enables better
        discrimination between success/failure states has higher
        epistemic value.

        Uses category-level statistics as a proxy for full MI computation:
        - Categories with high variance in outcomes → more to learn → higher MI
        - Categories with consistent outcomes → less to learn → lower MI

        Returns 0.0-1.0 scale.
        """
        if self._analytics is None:
            return self._epistemic_from_category(proposal)

        try:
            analytics = await self._analytics.compute_analytics()
        except Exception:
            return self._epistemic_from_category(proposal)

        category_key = proposal.category.value
        category_rate = analytics.category_rates.get(category_key)

        if category_rate is None or category_rate.total < _MIN_EPISODES_FOR_MI:
            return self._epistemic_from_category(proposal)

        # Compute entropy of outcomes for this category
        # Binary: success vs rollback
        p_success = category_rate.success_rate
        p_failure = category_rate.rollback_rate
        # Clamp to avoid log(0)
        p_success = max(0.01, min(0.99, p_success))
        p_failure = max(0.01, 1.0 - p_success)

        # Shannon entropy H = -sum(p * log2(p))
        entropy = -(p_success * math.log2(p_success) + p_failure * math.log2(p_failure))

        # Higher entropy = more uncertainty = more epistemic value from any change
        # Normalise: max binary entropy is 1.0 (at p=0.5)
        epistemic = entropy * 0.5  # Scale to 0.0-0.5

        # Bonus: proposals with more affected systems reveal more about system interactions
        system_count = len(proposal.change_spec.affected_systems)
        interaction_bonus = min(0.2, system_count * 0.05)

        # Bonus: novel categories (few past proposals) have higher epistemic value
        novelty_bonus = 0.0
        if category_rate.total < 10:
            novelty_bonus = 0.15 * (1.0 - category_rate.total / 10.0)

        return min(1.0, epistemic + interaction_bonus + novelty_bonus)

    def _epistemic_from_category(self, proposal: EvolutionProposal) -> float:
        """Heuristic epistemic value when no analytics available."""
        # Higher-impact categories have more epistemic value
        category_epistemic: dict[ChangeCategory, float] = {
            ChangeCategory.ADJUST_BUDGET: 0.15,
            ChangeCategory.ADD_PATTERN_DETECTOR: 0.35,
            ChangeCategory.ADD_INPUT_CHANNEL: 0.30,
            ChangeCategory.ADD_EXECUTOR: 0.25,
            ChangeCategory.MODIFY_CONTRACT: 0.40,
            ChangeCategory.MODIFY_CYCLE_TIMING: 0.20,
            ChangeCategory.CHANGE_CONSOLIDATION: 0.25,
            ChangeCategory.ADD_SYSTEM_CAPABILITY: 0.45,
        }
        return category_epistemic.get(proposal.category, 0.25)

    # ─── Complexity Penalty ──────────────────────────────────────────────────

    def _compute_complexity(self, proposal: EvolutionProposal) -> float:
        """
        Estimate complexity increase from the proposed change.

        Three signals:
        1. Category base complexity (heuristic)
        2. Number of affected systems (cross-cutting complexity)
        3. ChangeSpec field density (more fields = more state to manage)

        Returns 0.0-1.0 scale.
        """
        # Base complexity by category
        base = _CATEGORY_BASE_COMPLEXITY.get(proposal.category, 0.30)

        # Cross-cutting: each additional affected system adds complexity
        systems = proposal.change_spec.affected_systems
        cross_cutting = min(0.3, len(systems) * 0.08) if systems else 0.0

        # Field density: count non-None, non-default fields in ChangeSpec
        spec = proposal.change_spec
        filled_fields = 0
        for field_name in [
            "executor_name", "executor_description", "executor_action_type",
            "channel_name", "channel_type", "channel_description",
            "detector_name", "detector_description", "detector_pattern_type",
            "budget_parameter", "capability_description",
            "timing_parameter", "consolidation_schedule",
        ]:
            if getattr(spec, field_name, None) is not None:
                filled_fields += 1

        # More specified fields → more complex change
        field_signal = min(0.2, filled_fields * 0.03)

        # Contract changes add complexity per item
        contract_signal = min(0.15, len(spec.contract_changes) * 0.05)

        complexity = min(1.0, base + cross_cutting + field_signal + contract_signal)
        return complexity

    # ─── Confidence Estimation ───────────────────────────────────────────────

    def _estimate_confidence(
        self,
        proposal: EvolutionProposal,
        pragmatic: float,
        epistemic: float,
    ) -> float:
        """
        Estimate confidence in the EFE score.

        Higher confidence when:
        - More evidence supporting the proposal
        - Analytics data is available for the category
        - Calibration has been updated (lower bias)
        """
        base = 0.3

        # Evidence count boosts confidence
        evidence_count = len(proposal.evidence)
        if evidence_count >= 10:
            base += 0.25
        elif evidence_count >= 5:
            base += 0.15
        elif evidence_count >= 1:
            base += 0.05

        # Analytics availability
        if self._analytics is not None:
            base += 0.15

        # Calibration reduces prediction error over time
        if self._calibration_count >= 10:
            base += 0.15
        elif self._calibration_count >= 3:
            base += 0.08

        return min(0.95, base)

    # ─── Calibration Feedback Loop ───────────────────────────────────────────

    def record_calibration(self, record: EFECalibrationRecord) -> None:
        """
        Update calibration from a feedback record.

        Uses exponential moving average to track prediction bias:
          bias_new = α * error + (1-α) * bias_old

        The bias is subtracted from future EFE scores to correct
        systematic over/under-estimation.
        """
        alpha = 0.2  # Learning rate for calibration
        self._calibration_bias = (
            alpha * record.efe_error
            + (1.0 - alpha) * self._calibration_bias
        )
        self._calibration_count += 1

        self._log.info(
            "efe_calibration_updated",
            proposal_id=record.proposal_id,
            predicted_efe=record.predicted_efe,
            actual_improvement=record.actual_improvement,
            efe_error=record.efe_error,
            new_bias=round(self._calibration_bias, 4),
            calibration_count=self._calibration_count,
        )

    async def compute_calibration_from_history(self) -> None:
        """
        Bootstrap calibration from historical proposal outcomes.

        Compares evolution records that were applied (not rolled back)
        with their predicted EFE scores to compute initial bias.
        Called during SimulaService.initialize().
        """
        if self._history is None:
            return

        try:
            records = await self._history.get_history(limit=100)
        except Exception:
            return

        if len(records) < 3:
            return

        # Use counterfactual regression rate as proxy for actual improvement
        errors: list[float] = []
        for record in records:
            if record.rolled_back:
                continue
            # If proposal was applied and NOT rolled back, it presumably improved things.
            # The regression rate tells us how much damage it DIDN'T do.
            actual = 1.0 - record.counterfactual_regression_rate
            # We don't have the original predicted EFE, so use constitutional_alignment
            # as a rough proxy for predicted value
            predicted = -record.constitutional_alignment
            errors.append(actual - predicted)

        if errors:
            self._calibration_bias = sum(errors) / len(errors)
            self._calibration_count = len(errors)
            self._log.info(
                "efe_calibration_bootstrapped",
                records_used=len(errors),
                initial_bias=round(self._calibration_bias, 4),
            )

    async def load_parent_genome_calibration(
        self,
        neo4j: Neo4jClient | None = None,
    ) -> None:
        """
        Load EFE calibration seeded from a parent genome.

        During Mitosis, the parent's calibration data is written as
        (:EFECalibration {source: "parent_genome"}) nodes. If the child
        has no history of its own (calibration_count < 3), apply the
        parent's calibration as a warm start so the child doesn't score
        proposals naively.
        """
        client = neo4j or self._neo4j
        if client is None:
            return

        # Only warm-start if we don't have enough local calibration
        if self._calibration_count >= 3:
            self._log.debug(
                "efe_parent_calibration_skipped",
                reason="sufficient_local_calibration",
                calibration_count=self._calibration_count,
            )
            return

        try:
            rows = await client.execute_read(
                """
                MATCH (c:EFECalibration)
                WHERE c.source = "parent_genome"
                RETURN c.efe_error AS error
                ORDER BY c.measured_at DESC
                LIMIT 50
                """,
            )

            if not rows:
                return

            parent_errors = [float(r["error"]) for r in rows if r.get("error") is not None]
            if not parent_errors:
                return

            # Apply parent calibration with discounted confidence (0.7x)
            parent_bias = sum(parent_errors) / len(parent_errors)
            discount = 0.7
            self._calibration_bias = parent_bias * discount
            self._calibration_count = max(self._calibration_count, 1)

            self._log.info(
                "efe_parent_calibration_loaded",
                parent_samples=len(parent_errors),
                parent_bias=round(parent_bias, 4),
                discounted_bias=round(self._calibration_bias, 4),
            )
        except Exception as exc:
            self._log.warning("efe_parent_calibration_error", error=str(exc))

    # ─── Equor Integration Helpers ───────────────────────────────────────────

    @staticmethod
    def should_auto_approve(breakdown: ProposalEFEBreakdown) -> bool:
        """
        Returns True if the EFE score is strong enough for Equor auto-approval.

        Criteria:
        - EFE is below the auto-approve threshold (strongly negative)
        - Confidence is at least 0.6
        """
        return (
            breakdown.efe_penalised <= _AUTO_APPROVE_EFE_THRESHOLD
            and breakdown.confidence >= 0.6
        )

    # ─── Stats ───────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, object]:
        return {
            "penalty_weight": self._penalty_weight,
            "calibration_bias": round(self._calibration_bias, 4),
            "calibration_count": self._calibration_count,
            "auto_approve_threshold": _AUTO_APPROVE_EFE_THRESHOLD,
        }
