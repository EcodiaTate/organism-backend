"""
EcodiaOS - Curiosity-Driven Exploration Engine

The difference between a database that learns and an organism that WANTS to know.

CuriosityEngine maintains an epistemic value map: for each active hypothesis,
it computes the Expected Information Gain (EIG) of possible observations using
KL-divergence between current posterior and expected posterior given observation.

It generates EpistemicIntents - actions the organism takes purely to gather
evidence (not to achieve user goals). These are ranked by:
    priority = EIG × hypothesis_importance × (1 / cost_estimate)

Integration:
  - Fed to Nova via Synapse event EVO_EPISTEMIC_INTENT_PROPOSED
  - Nova integrates them into EFE minimization alongside pragmatic goals
  - Soma curiosity drive amplifies epistemic intent priority
  - Tracks exploration/exploitation ratio: target ~20% epistemic / 80% pragmatic

Active Hypothesis Seeking:
  - When a hypothesis has been TESTING for >48h with <3 evidence episodes,
    the engine actively constructs scenarios that would confirm or refute it.
  - These probe actions bypass normal pattern detection - they are DESIGNED experiments.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now
from systems.evo.types import (
    CuriosityState,
    EpistemicIntent,
    Hypothesis,
    HypothesisCategory,
    HypothesisStatus,
)

if TYPE_CHECKING:
    from systems.memory.service import MemoryService

logger = structlog.get_logger()

# EIG computation constants
_MIN_EIG_THRESHOLD: float = 0.01          # Minimum EIG to emit an intent
_MAX_INTENTS_PER_CYCLE: int = 5            # Cap epistemic intents per cycle
_STALE_HYPOTHESIS_HOURS: float = 48.0      # Time before active seeking kicks in
_STALE_EVIDENCE_THRESHOLD: int = 3         # Min evidence episodes before considered stale
_TARGET_EXPLORATION_RATIO: float = 0.20    # 20% epistemic / 80% pragmatic
_EXPLORATION_RATIO_TOLERANCE: float = 0.05 # Deadband around target


class CuriosityEngine:
    """
    Maintains an epistemic value map and generates EpistemicIntents.

    The engine computes Expected Information Gain for each active hypothesis
    and proposes probe actions that would maximally reduce uncertainty.
    """

    def __init__(
        self,
        memory: MemoryService | None = None,
    ) -> None:
        self._memory = memory
        self._logger = logger.bind(system="evo.curiosity")

        # Counters
        self._total_intents_proposed: int = 0
        self._total_intents_executed: int = 0

        # Rolling window for exploration/exploitation ratio tracking
        self._recent_actions: list[bool] = []  # True=epistemic, False=pragmatic
        self._max_window: int = 100

        # Soma curiosity multiplier (wired externally)
        self._soma_curiosity_multiplier: float = 1.0

        # Per-hypothesis evidence velocity: hypothesis_id → evidence count at last check
        self._evidence_velocity: dict[str, int] = {}

        # Domain coverage: domain → last epistemic probe timestamp
        self._domain_last_probed: dict[str, float] = defaultdict(float)

    # ─── Core: Generate Epistemic Intents ────────────────────────────────────

    async def generate_epistemic_intents(
        self,
        active_hypotheses: list[Hypothesis],
        soma_curiosity: float = 0.5,
    ) -> list[EpistemicIntent]:
        """
        Generate EpistemicIntents for hypotheses that would benefit from
        active exploration.

        Steps:
          1. Compute EIG for each TESTING hypothesis
          2. Identify stale hypotheses needing active seeking
          3. Rank by priority = EIG × importance × (1/cost) × curiosity_multiplier
          4. Emit top-N intents respecting exploration ratio bounds

        Returns a list of EpistemicIntents ready for Nova integration.
        """
        # Update Soma curiosity multiplier
        # Scale: 0.5 drive = 1.0×, 1.0 drive = 2.0× (more exploration)
        self._soma_curiosity_multiplier = 0.5 + soma_curiosity * 1.5

        # Filter to TESTING hypotheses only
        testing = [
            h for h in active_hypotheses
            if h.status in (HypothesisStatus.PROPOSED, HypothesisStatus.TESTING)
        ]
        if not testing:
            return []

        # Check exploration ratio - throttle if we're over target
        current_ratio = self._compute_exploration_ratio()
        ratio_multiplier = 1.0
        if current_ratio > _TARGET_EXPLORATION_RATIO + _EXPLORATION_RATIO_TOLERANCE:
            # Over-exploring - reduce priority
            ratio_multiplier = 0.3
        elif current_ratio < _TARGET_EXPLORATION_RATIO - _EXPLORATION_RATIO_TOLERANCE:
            # Under-exploring - boost priority
            ratio_multiplier = 1.5

        intents: list[EpistemicIntent] = []

        for h in testing:
            eig = self._compute_eig(h)
            if eig < _MIN_EIG_THRESHOLD:
                continue

            importance = self._compute_importance(h)
            cost = self._estimate_cost(h)
            action = self._design_probe_action(h)

            priority = (
                eig * importance * (1.0 / max(0.1, cost))
                * self._soma_curiosity_multiplier
                * ratio_multiplier
            )

            intent = EpistemicIntent(
                hypothesis_id=h.id,
                hypothesis_statement=h.statement[:200],
                expected_information_gain=round(eig, 4),
                hypothesis_importance=round(importance, 4),
                cost_estimate=round(cost, 4),
                priority=round(priority, 4),
                proposed_action=action,
                target_domain=h.category.value,
            )
            intents.append(intent)

        # Sort by priority descending, take top N
        intents.sort(key=lambda i: i.priority, reverse=True)
        top_intents = intents[:_MAX_INTENTS_PER_CYCLE]

        # Track for ratio computation
        for _ in top_intents:
            self._record_action(epistemic=True)
        self._total_intents_proposed += len(top_intents)

        if top_intents:
            self._logger.info(
                "epistemic_intents_generated",
                count=len(top_intents),
                top_priority=top_intents[0].priority if top_intents else 0,
                exploration_ratio=round(current_ratio, 3),
                soma_curiosity=round(soma_curiosity, 3),
            )

        return top_intents

    # ─── Active Hypothesis Seeking ───────────────────────────────────────────

    async def identify_stale_hypotheses(
        self,
        active_hypotheses: list[Hypothesis],
    ) -> list[Hypothesis]:
        """
        Find hypotheses that have been TESTING for >48h with <3 evidence episodes.
        These need active scenario construction to confirm or refute them.
        """
        now = utc_now()
        stale: list[Hypothesis] = []

        for h in active_hypotheses:
            if h.status != HypothesisStatus.TESTING:
                continue
            age_hours = (now - h.created_at).total_seconds() / 3600.0
            evidence_count = len(h.supporting_episodes) + len(h.contradicting_episodes)

            if age_hours > _STALE_HYPOTHESIS_HOURS and evidence_count < _STALE_EVIDENCE_THRESHOLD:
                stale.append(h)

        return stale

    async def construct_evidence_seeking_scenarios(
        self,
        stale_hypotheses: list[Hypothesis],
    ) -> list[EpistemicIntent]:
        """
        For hypotheses stuck in TESTING with insufficient evidence, construct
        specific scenarios that would definitively confirm or refute them.

        These are higher-priority than normal epistemic intents because the
        organism has been uncertain for too long.
        """
        intents: list[EpistemicIntent] = []

        for h in stale_hypotheses[:3]:  # Cap at 3 active-seeking scenarios per cycle
            # Design a scenario that would maximally disambiguate
            confirming_action = (
                f"Test hypothesis '{h.statement[:100]}' by: "
                f"seeking observations where the formal test '{h.formal_test[:100]}' "
                f"would produce a clear positive or negative result"
            )

            # Stale-seeking intents get a priority boost (1.5×)
            age_hours = (utc_now() - h.created_at).total_seconds() / 3600.0
            urgency_boost = min(2.0, 1.0 + (age_hours - _STALE_HYPOTHESIS_HOURS) / 48.0)

            intent = EpistemicIntent(
                hypothesis_id=h.id,
                hypothesis_statement=h.statement[:200],
                expected_information_gain=0.5,  # High EIG - this is a designed experiment
                hypothesis_importance=0.8,
                cost_estimate=1.5,              # Slightly higher cost - active seeking
                priority=round(
                    0.5 * 0.8 * (1.0 / 1.5) * urgency_boost
                    * self._soma_curiosity_multiplier, 4,
                ),
                proposed_action=confirming_action,
                target_domain=h.category.value,
            )
            intents.append(intent)

        if intents:
            self._logger.info(
                "evidence_seeking_scenarios_constructed",
                count=len(intents),
                stale_hypotheses=len(stale_hypotheses),
            )

        return intents

    # ─── Pragmatic Action Recording ──────────────────────────────────────────

    def record_pragmatic_action(self) -> None:
        """Record that a pragmatic (non-epistemic) action was taken."""
        self._record_action(epistemic=False)

    def record_epistemic_execution(self, hypothesis_id: str) -> None:
        """Record that an epistemic intent was actually executed."""
        self._total_intents_executed += 1
        self._record_action(epistemic=True)

    # ─── EIG Computation ─────────────────────────────────────────────────────

    def _compute_eig(self, hypothesis: Hypothesis) -> float:
        """
        Compute Expected Information Gain for a hypothesis.

        EIG = KL(posterior_expected || posterior_current)

        Approximated using the hypothesis evidence score and supporting/contradicting
        episode counts as a proxy for the posterior distribution.

        High EIG means:
          - Evidence score is near zero (maximum uncertainty)
          - Few supporting AND few contradicting episodes (little data)
          - The hypothesis covers a domain with sparse knowledge
        """
        score = hypothesis.evidence_score
        supporting = len(hypothesis.supporting_episodes)
        contradicting = len(hypothesis.contradicting_episodes)
        total_evidence = supporting + contradicting

        # Uncertainty component: maximum at score=0, decays as |score| increases
        # Uses a Gaussian-like decay centered at 0
        uncertainty = math.exp(-0.5 * (score ** 2))

        # Data sparsity component: high when few observations
        # Uses inverse log to avoid division by zero
        sparsity = 1.0 / (1.0 + math.log1p(total_evidence))

        # Conflict component: if supporting ≈ contradicting, there's high uncertainty
        if total_evidence > 0:
            balance = 1.0 - abs(supporting - contradicting) / total_evidence
        else:
            balance = 1.0

        # EIG = uncertainty × sparsity × balance
        # Normalised to [0, 1]
        eig = uncertainty * sparsity * balance

        # Evidence velocity: how fast is this hypothesis accumulating data?
        prev_count = self._evidence_velocity.get(hypothesis.id, 0)
        velocity = total_evidence - prev_count
        self._evidence_velocity[hypothesis.id] = total_evidence

        # If evidence is flowing in fast, EIG is lower (we're already learning)
        if velocity > 2:
            eig *= 0.5

        return min(1.0, eig)

    def _compute_importance(self, hypothesis: Hypothesis) -> float:
        """
        Compute hypothesis importance based on category and age.

        WORLD_MODEL and PARAMETER hypotheses are more important because they
        affect more downstream systems. Older hypotheses with unresolved
        uncertainty are more important to resolve.
        """
        category_weights: dict[HypothesisCategory, float] = {
            HypothesisCategory.WORLD_MODEL: 0.8,
            HypothesisCategory.PARAMETER: 0.7,
            HypothesisCategory.PROCEDURAL: 0.5,
            HypothesisCategory.SELF_MODEL: 0.6,
            HypothesisCategory.SOCIAL: 0.4,
        }
        base = category_weights.get(hypothesis.category, 0.5)

        # Age boost: older unresolved hypotheses are more important to resolve
        age_hours = (utc_now() - hypothesis.created_at).total_seconds() / 3600.0
        age_factor = min(1.5, 1.0 + age_hours / 168.0)  # Caps at +50% after 1 week

        return min(1.0, base * age_factor)

    def _estimate_cost(self, hypothesis: Hypothesis) -> float:
        """
        Estimate the cost of an epistemic probe for this hypothesis.

        PARAMETER hypotheses are cheap (just run an experiment).
        WORLD_MODEL hypotheses are expensive (need external data).
        SOCIAL hypotheses are moderate (need interaction data).
        """
        category_costs: dict[HypothesisCategory, float] = {
            HypothesisCategory.PARAMETER: 0.5,
            HypothesisCategory.SELF_MODEL: 0.7,
            HypothesisCategory.PROCEDURAL: 0.8,
            HypothesisCategory.SOCIAL: 1.0,
            HypothesisCategory.WORLD_MODEL: 1.2,
        }
        return category_costs.get(hypothesis.category, 1.0)

    def _design_probe_action(self, hypothesis: Hypothesis) -> str:
        """
        Design a specific probe action that would maximally inform this hypothesis.

        Returns a natural-language description that Nova can interpret.
        """
        if hypothesis.category == HypothesisCategory.PARAMETER:
            return (
                f"Observe system behaviour with current parameters to test: "
                f"{hypothesis.formal_test[:150]}"
            )
        elif hypothesis.category == HypothesisCategory.WORLD_MODEL:
            return (
                f"Seek external evidence about: {hypothesis.statement[:150]}. "
                f"Falsification condition: {hypothesis.formal_test[:100]}"
            )
        elif hypothesis.category == HypothesisCategory.SELF_MODEL:
            return (
                f"Introspect on own capability re: {hypothesis.statement[:150]}. "
                f"Measure actual vs predicted performance."
            )
        elif hypothesis.category == HypothesisCategory.PROCEDURAL:
            return (
                f"Execute the procedure described and observe outcomes: "
                f"{hypothesis.formal_test[:150]}"
            )
        else:
            return (
                f"Gather observations about: {hypothesis.statement[:150]}"
            )

    # ─── Exploration Ratio ───────────────────────────────────────────────────

    def _record_action(self, epistemic: bool) -> None:
        """Record an action for exploration/exploitation ratio tracking."""
        self._recent_actions.append(epistemic)
        if len(self._recent_actions) > self._max_window:
            self._recent_actions = self._recent_actions[-self._max_window:]

    def _compute_exploration_ratio(self) -> float:
        """Compute the current exploration ratio from the rolling window."""
        if not self._recent_actions:
            return _TARGET_EXPLORATION_RATIO  # No data → assume at target
        epistemic_count = sum(1 for a in self._recent_actions if a)
        return epistemic_count / len(self._recent_actions)

    # ─── State Query ─────────────────────────────────────────────────────────

    def get_state(self) -> CuriosityState:
        """Return a snapshot of the curiosity engine's state."""
        return CuriosityState(
            total_intents_proposed=self._total_intents_proposed,
            total_intents_executed=self._total_intents_executed,
            exploration_ratio=round(self._compute_exploration_ratio(), 4),
            target_exploration_ratio=_TARGET_EXPLORATION_RATIO,
            active_hypotheses_seeking=0,  # Updated by caller
            soma_curiosity_multiplier=round(self._soma_curiosity_multiplier, 4),
        )

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_intents_proposed": self._total_intents_proposed,
            "total_intents_executed": self._total_intents_executed,
            "exploration_ratio": round(self._compute_exploration_ratio(), 4),
            "soma_curiosity_multiplier": round(self._soma_curiosity_multiplier, 4),
        }
