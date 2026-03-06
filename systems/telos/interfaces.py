"""
EcodiaOS — Telos: Integration Interfaces (Phase D)

The interfaces that let every other EOS module become Telos-aware.
These do NOT modify other modules — they expose what other modules
need to consume.

TelosPolicyScorer:
    For Nova — evaluate whether a proposed action improves effective_I
    (not just nominal_I). A policy that improves nominal_I by ignoring
    welfare consequences scores LOWER than one that improves effective_I
    by expanding care coverage.

TelosHypothesisPrioritizer:
    For Evo and Kairos — rank hypotheses by their topology contribution.
    Hypotheses that would improve care coverage, reduce incoherence,
    increase honesty validity, or open growth frontiers get priority.

TelosFragmentSelector:
    For Nexus — score world model fragments for federation sharing by
    their care+coherence content. High-care, high-coherence structures
    get shared preferentially.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import structlog

from systems.telos.types import (
    EffectiveIntelligenceReport,
    HypothesisTopologyContribution,
    TelosConfig,
    TelosScore,
)

logger = structlog.get_logger()


# ─── Protocols for what other modules provide ─────────────────────────
# Telos doesn't reach into other modules. It defines what it needs
# and lets them satisfy the interface.


@runtime_checkable
class PolicyDescriptor(Protocol):
    """What Telos needs to know about a policy to score it."""

    @property
    def goal_description(self) -> str: ...

    @property
    def expected_welfare_impact(self) -> float: ...

    @property
    def expected_coherence_impact(self) -> float: ...

    @property
    def expected_honesty_impact(self) -> float: ...

    @property
    def expected_growth_impact(self) -> float: ...

    @property
    def expected_nominal_I_delta(self) -> float: ...


@runtime_checkable
class HypothesisDescriptor(Protocol):
    """What Telos needs to know about a hypothesis to prioritize it."""

    @property
    def hypothesis_id(self) -> str: ...

    @property
    def domain(self) -> str: ...

    @property
    def statement(self) -> str: ...

    @property
    def confidence(self) -> float: ...


@runtime_checkable
class FragmentDescriptor(Protocol):
    """What Telos needs to know about a world model fragment to score it."""

    @property
    def domain(self) -> str: ...

    @property
    def coverage(self) -> float: ...

    @property
    def coherence_validated(self) -> bool: ...

    @property
    def prediction_accuracy(self) -> float: ...


# ─── Policy Scorer (for Nova) ─────────────────────────────────────────


class TelosPolicyScorer:
    """
    Evaluate whether a proposed action improves effective_I.

    The key insight: a policy that improves nominal_I by ignoring welfare
    consequences should score LOWER than one that improves effective_I
    by expanding care coverage. This is the mechanism by which Telos
    transforms Nova's policy selection from "maximize nominal I" to
    "maximize real intelligence."

    Requires the most recent EffectiveIntelligenceReport to compute
    how the policy would shift the drive multipliers.
    """

    def __init__(self, config: TelosConfig) -> None:
        self._config = config
        self._logger = logger.bind(component="telos.policy_scorer")

    def score_policy(
        self,
        policy: PolicyDescriptor | dict[str, Any],
        current_report: EffectiveIntelligenceReport | None,
    ) -> TelosScore:
        """
        Score a proposed policy by its effect on effective_I.

        Accepts either a PolicyDescriptor protocol or a plain dict
        with the same keys (for modules that can't implement the protocol).
        """
        # Extract fields from protocol or dict
        if isinstance(policy, dict):
            welfare_impact = float(policy.get("expected_welfare_impact", 0.0))
            coherence_impact = float(policy.get("expected_coherence_impact", 0.0))
            honesty_impact = float(policy.get("expected_honesty_impact", 0.0))
            growth_impact = float(policy.get("expected_growth_impact", 0.0))
            nominal_delta = float(policy.get("expected_nominal_I_delta", 0.0))
        else:
            welfare_impact = policy.expected_welfare_impact
            coherence_impact = policy.expected_coherence_impact
            honesty_impact = policy.expected_honesty_impact
            growth_impact = policy.expected_growth_impact
            nominal_delta = policy.expected_nominal_I_delta

        # Current drive multipliers (defaults to perfect if no report yet)
        if current_report is not None:
            care_m = current_report.care_multiplier
            coherence_b = current_report.coherence_bonus
            honesty_c = current_report.honesty_coefficient
        else:
            care_m = 1.0
            coherence_b = 1.0
            honesty_c = 1.0

        # Estimate how each drive multiplier would change
        # Care: welfare_impact > 0 means expanding coverage
        care_delta = welfare_impact * _CARE_SENSITIVITY
        # Coherence: positive coherence_impact means reducing contradictions
        coherence_delta = coherence_impact * _COHERENCE_SENSITIVITY
        # Honesty: positive honesty_impact means improving measurement validity
        honesty_delta = honesty_impact * _HONESTY_SENSITIVITY
        # Growth: positive growth_impact means frontier exploration
        growth_delta = growth_impact * _GROWTH_SENSITIVITY

        # Projected new multipliers (clamped)
        new_care = max(0.0, min(1.0, care_m + care_delta))
        new_coherence_penalty = 1.0 / max(1.0, coherence_b - coherence_delta)
        new_honesty = max(0.0, min(1.0, honesty_c + honesty_delta))

        # Compute effective_I delta
        if current_report is not None:
            current_effective = current_report.effective_I
            current_nominal = current_report.nominal_I
            projected_nominal = current_nominal + nominal_delta
            projected_effective = (
                projected_nominal * new_care * new_coherence_penalty * new_honesty
            )
            effective_delta = projected_effective - current_effective
        else:
            effective_delta = nominal_delta  # No topology data — assume 1:1

        # Misalignment risk: nominal goes up but effective goes down
        misalignment_risk = nominal_delta > 0 and effective_delta < 0

        # Composite score: effective_I delta weighted by drive contributions
        # Penalize policies that improve nominal but hurt effective
        if misalignment_risk:
            composite = effective_delta * 2.0  # Double the penalty
        else:
            composite = effective_delta + growth_delta * _GROWTH_BONUS

        score = TelosScore(
            nominal_I_delta=nominal_delta,
            effective_I_delta=effective_delta,
            care_impact=care_delta,
            coherence_impact=coherence_delta,
            honesty_impact=honesty_delta,
            growth_impact=growth_delta,
            composite_score=composite,
            misalignment_risk=misalignment_risk,
        )

        self._logger.debug(
            "policy_scored",
            nominal_delta=f"{nominal_delta:.4f}",
            effective_delta=f"{effective_delta:.4f}",
            composite=f"{composite:.4f}",
            misalignment_risk=misalignment_risk,
        )

        return score


# ─── Hypothesis Prioritizer (for Evo and Kairos) ─────────────────────


class TelosHypothesisPrioritizer:
    """
    Rank hypotheses by their contribution to the drive topology.

    Hypotheses that would improve care coverage, reduce incoherence,
    increase honesty validity, or open growth frontiers get priority.

    This is the mechanism by which Telos transforms hypothesis selection
    from "what's most testable" to "what would most improve real
    intelligence if confirmed."
    """

    def __init__(self, config: TelosConfig) -> None:
        self._config = config
        self._logger = logger.bind(component="telos.hypothesis_prioritizer")

    def prioritize(
        self,
        hypotheses: list[HypothesisDescriptor] | list[dict[str, Any]],
        current_report: EffectiveIntelligenceReport | None,
        domain_coverage: dict[str, float] | None = None,
    ) -> list[HypothesisTopologyContribution]:
        """
        Rank hypotheses by their topology contribution.

        Returns a sorted list (highest contribution first) with
        per-drive contribution scores and a composite rank.
        """
        contributions: list[HypothesisTopologyContribution] = []

        for hyp in hypotheses:
            contribution = self._score_hypothesis(
                hyp, current_report, domain_coverage
            )
            contributions.append(contribution)

        # Sort by composite contribution (highest first)
        contributions.sort(key=lambda c: c.composite_contribution, reverse=True)

        # Assign ranks
        for i, contrib in enumerate(contributions):
            contrib.rank = i + 1

        self._logger.debug(
            "hypotheses_prioritized",
            count=len(contributions),
            top_id=contributions[0].hypothesis_id if contributions else "none",
            top_score=(
                f"{contributions[0].composite_contribution:.3f}"
                if contributions else "0"
            ),
        )

        return contributions

    def _score_hypothesis(
        self,
        hyp: HypothesisDescriptor | dict[str, Any],
        current_report: EffectiveIntelligenceReport | None,
        domain_coverage: dict[str, float] | None,
    ) -> HypothesisTopologyContribution:
        """Score a single hypothesis by its topology contribution."""
        # Extract fields
        if isinstance(hyp, dict):
            hyp_id = str(hyp.get("hypothesis_id", ""))
            domain = str(hyp.get("domain", ""))
            statement = str(hyp.get("statement", ""))
            confidence = float(hyp.get("confidence", 0.0))
        else:
            hyp_id = hyp.hypothesis_id
            domain = hyp.domain
            statement = hyp.statement
            confidence = hyp.confidence

        statement_lower = statement.lower()

        # Care contribution: does this hypothesis involve welfare domains?
        care_score = 0.0
        if any(kw in statement_lower for kw in _WELFARE_KEYWORDS):
            care_score = 0.5 + confidence * 0.5
        if any(kw in domain.lower() for kw in _WELFARE_KEYWORDS):
            care_score = max(care_score, 0.3 + confidence * 0.4)

        # Coherence contribution: would confirming this resolve contradictions?
        coherence_score = 0.0
        if any(kw in statement_lower for kw in _COHERENCE_KEYWORDS):
            coherence_score = 0.4 + confidence * 0.4

        # Honesty contribution: does this hypothesis improve measurement validity?
        honesty_score = 0.0
        if any(kw in statement_lower for kw in _HONESTY_KEYWORDS):
            honesty_score = 0.5 + confidence * 0.5

        # Growth contribution: is this in a frontier domain?
        growth_score = 0.0
        if domain_coverage is not None:
            domain_cov = domain_coverage.get(domain, 0.0)
            if domain_cov < 0.4:
                # Frontier domain — high growth contribution
                growth_score = (1.0 - domain_cov) * confidence
        elif any(kw in statement_lower for kw in _GROWTH_KEYWORDS):
            growth_score = 0.3 + confidence * 0.3

        # Boost if the current report shows a particular drive is weak
        if current_report is not None:
            if current_report.care_multiplier < 0.8:
                care_score *= 1.5
            if current_report.coherence_bonus > 1.2:
                coherence_score *= 1.5
            if current_report.honesty_coefficient < 0.8:
                honesty_score *= 1.5

        # Composite: weighted sum
        composite = (
            care_score * 0.30
            + coherence_score * 0.25
            + honesty_score * 0.25
            + growth_score * 0.20
        )

        return HypothesisTopologyContribution(
            hypothesis_id=hyp_id,
            care_contribution=care_score,
            coherence_contribution=coherence_score,
            honesty_contribution=honesty_score,
            growth_contribution=growth_score,
            composite_contribution=composite,
        )


# ─── Fragment Selector (for Nexus) ───────────────────────────────────


class TelosFragmentSelector:
    """
    Score world model fragments for federation sharing.

    High-care, high-coherence structures get shared preferentially.
    This ensures that what propagates through the federation is the
    best of what EOS has built — not the most convenient or the
    most novel, but the most reality-grounded.
    """

    def __init__(self, config: TelosConfig) -> None:
        self._config = config
        self._logger = logger.bind(component="telos.fragment_selector")

    def score_fragment(
        self,
        fragment: FragmentDescriptor | dict[str, Any],
    ) -> float:
        """
        Score a world model fragment for federation sharing.

        Returns a float in [0, 1] where:
        - 1.0 = high care coverage + high coherence + high accuracy
        - 0.0 = welfare-blind, incoherent, or low-accuracy fragment

        The score is the product of:
        - care_score: is this fragment about welfare-relevant domains?
        - coherence_score: has this fragment been coherence-validated?
        - accuracy_score: how well does this fragment predict reality?
        """
        # Extract fields
        if isinstance(fragment, dict):
            domain = str(fragment.get("domain", ""))
            coverage = float(fragment.get("coverage", 0.0))
            coherence_validated = bool(fragment.get("coherence_validated", False))
            prediction_accuracy = float(fragment.get("prediction_accuracy", 0.0))
        else:
            domain = fragment.domain
            coverage = fragment.coverage
            coherence_validated = fragment.coherence_validated
            prediction_accuracy = fragment.prediction_accuracy

        # Care score: welfare-relevant domains score higher
        domain_lower = domain.lower()
        if any(kw in domain_lower for kw in _WELFARE_KEYWORDS):
            care_score = 0.6 + coverage * 0.4
        else:
            care_score = 0.3 + coverage * 0.3  # Non-welfare domains still have value

        # Coherence score: validated fragments score much higher
        if coherence_validated:
            coherence_score = 0.8 + coverage * 0.2
        else:
            coherence_score = 0.3  # Unvalidated = risky to share

        # Accuracy score: directly from prediction accuracy
        accuracy_score = max(0.0, min(1.0, prediction_accuracy))

        # Composite: care and coherence are weighted more heavily
        # because they determine whether the fragment is safe to share
        composite = (
            care_score * 0.35
            + coherence_score * 0.35
            + accuracy_score * 0.30
        )

        return max(0.0, min(1.0, composite))

    def rank_fragments(
        self,
        fragments: list[FragmentDescriptor] | list[dict[str, Any]],
    ) -> list[tuple[int, float]]:
        """
        Rank fragments by their sharing score.

        Returns list of (index, score) tuples sorted by score descending.
        """
        scored = [(i, self.score_fragment(f)) for i, f in enumerate(fragments)]  # type: ignore[arg-type]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# ─── Keyword Sets ─────────────────────────────────────────────────────

_WELFARE_KEYWORDS = (
    "welfare", "care", "harm", "trust", "social", "relationship",
    "safety", "health", "wellbeing", "emotional", "interpersonal",
    "cooperation", "conflict", "consent", "community",
)

_COHERENCE_KEYWORDS = (
    "contradiction", "inconsisten", "coherence", "incoherence",
    "conflict", "mismatch", "unif", "reconcil", "resolv",
)

_HONESTY_KEYWORDS = (
    "measurement", "validity", "calibrat", "confabul", "overclaim",
    "prediction accuracy", "falsif", "test", "verif", "honest",
)

_GROWTH_KEYWORDS = (
    "frontier", "novel", "unexplored", "unknown", "discover",
    "experiment", "hypothesis", "explore", "compress",
)

# ─── Sensitivity Constants ────────────────────────────────────────────
# How much each unit of impact translates to a multiplier change.

_CARE_SENSITIVITY = 0.1
_COHERENCE_SENSITIVITY = 0.1
_HONESTY_SENSITIVITY = 0.1
_GROWTH_SENSITIVITY = 0.05
_GROWTH_BONUS = 0.1  # Extra credit for growth-positive policies
