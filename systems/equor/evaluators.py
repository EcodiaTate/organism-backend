"""
EcodiaOS - Equor Drive Evaluators

Four parallel evaluators - one per constitutional drive.
Each scores alignment from -1.0 (strongly violates) to +1.0 (strongly promotes).

Architecture
============

Each drive is a concrete subclass of ``BaseEquorEvaluator``.  The evaluator
instances live on ``EquorService`` and are registered with the
``NeuroplasticityBus`` so Simula can hot-reload evolved variants without
restarting the process.

The ABC is deliberately thin - a single ``evaluate(intent) → float`` method -
so hot-reloaded subclasses can be swapped in with zero ceremony.  All scoring
logic is synchronous CPU work (<1 ms per evaluator), called via the thin
``async evaluate`` wrapper so ``asyncio.gather`` can interleave them.
"""

from __future__ import annotations

import abc
import asyncio
from typing import TYPE_CHECKING

import structlog

from primitives.common import DriveAlignmentVector

if TYPE_CHECKING:
    from primitives.intent import Intent

logger = structlog.get_logger()


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ─── Abstract Base ──────────────────────────────────────────────


class BaseEquorEvaluator(abc.ABC):
    """
    Strategy interface for a single constitutional drive evaluator.

    Subclasses MUST set ``drive_name`` and implement ``evaluate``.
    The bus discovers concrete subclasses at import time; zero-arg
    construction is required (no __init__ parameters).
    """

    drive_name: str = ""  # overridden by subclasses

    @abc.abstractmethod
    async def evaluate(self, intent: Intent) -> float:
        """Return a score in [-1.0, +1.0] for this drive."""
        ...


# ─── Coherence Evaluator ────────────────────────────────────────


class CoherenceEvaluator(BaseEquorEvaluator):
    """
    "Does this action make the world more understandable and
     internally consistent for EOS?"
    """

    drive_name: str = "coherence"

    async def evaluate(self, intent: Intent) -> float:
        score = 0.0

        # Reasoning chain quality
        reasoning = intent.decision_trace.reasoning
        if reasoning and len(reasoning) > 20:
            score += 0.25
        elif reasoning:
            score += 0.1
        else:
            score -= 0.15

        # Alternatives considered - sign of deliberation
        alternatives = intent.decision_trace.alternatives_considered
        if alternatives and len(alternatives) >= 2:
            score += 0.2
        elif alternatives:
            score += 0.1

        # Goal clarity
        if intent.goal.description and len(intent.goal.description) > 10:
            score += 0.15
        if intent.goal.success_criteria:
            score += 0.1

        # Plan completeness
        if intent.plan.steps:
            score += 0.1
            if intent.plan.contingencies:
                score += 0.1

        # Expected free energy - lower is more coherent
        if intent.expected_free_energy < 0:
            score += min(0.2, abs(intent.expected_free_energy) * 0.1)
        elif intent.expected_free_energy > 0.5:
            score -= min(0.15, intent.expected_free_energy * 0.1)

        return _clamp(score)


# ─── Care Evaluator ─────────────────────────────────────────────


class CareEvaluator(BaseEquorEvaluator):
    """
    "Does this action promote the wellbeing of the people
     and systems EOS stewards?"
    """

    drive_name: str = "care"

    async def evaluate(self, intent: Intent) -> float:
        score = 0.0
        goal_lower = intent.goal.description.lower()

        # Positive care indicators
        care_positive = [
            "help", "support", "assist", "protect", "wellbeing", "care for",
            "benefit", "improve", "nurture", "comfort", "inform", "guide",
            "empower", "include", "welcome", "share knowledge",
            "repair", "heal", "restore", "recover", "fix", "remediate",
            "stabilise", "stabilize", "immune", "antibody",
        ]
        for indicator in care_positive:
            if indicator in goal_lower:
                score += 0.15
                break

        # Harm indicators (weighted 2x per spec - "first, do no harm")
        # NOTE: "suppress" removed - matches legitimate repair actions
        # (suppress error cascade, suppress threat). "silence" narrowed
        # to "silence user" to avoid matching "silence alarm/alert".
        harm_indicators = [
            "ignore wellbeing", "disregard safety", "override consent",
            "exclude", "punish", "withhold help", "abandon",
            "dismiss concern", "silence user",
        ]
        for indicator in harm_indicators:
            if indicator in goal_lower:
                score -= 0.30
                break

        # Action type assessment
        for step in intent.plan.steps:
            executor = step.executor.lower()
            if "communicate" in executor or "notify" in executor:
                score += 0.1
            elif "observe" in executor or "analyse" in executor:
                score += 0.05
            elif "resource" in executor:
                pass  # Neutral

        # Consent awareness
        all_text = goal_lower + " " + intent.decision_trace.reasoning.lower()
        if "consent" in all_text or "permission" in all_text or "approval" in all_text:
            score += 0.1

        # Equality check
        if "only for" in goal_lower or "exclude" in goal_lower or "except" in goal_lower:
            score -= 0.1

        return _clamp(score)


# ─── Growth Evaluator ───────────────────────────────────────────


class GrowthEvaluator(BaseEquorEvaluator):
    """
    "Does this action make EOS or its community more capable,
     aware, or mature?"
    """

    drive_name: str = "growth"

    async def evaluate(self, intent: Intent) -> float:
        score = 0.0
        goal_lower = intent.goal.description.lower()

        # Growth-positive indicators
        growth_positive = [
            "learn", "discover", "improve", "expand", "develop",
            "explore", "experiment", "understand", "investigate",
            "adapt", "evolve", "create", "innovate", "teach",
        ]
        for indicator in growth_positive:
            if indicator in goal_lower:
                score += 0.2
                break

        # Stagnation indicators
        stagnation_indicators = [
            "avoid", "refuse to try", "stay the same", "no change",
            "repeat exactly", "do nothing",
        ]
        for indicator in stagnation_indicators:
            if indicator in goal_lower:
                score -= 0.15
                break

        # Novelty - considered options = growth-oriented thinking
        if intent.decision_trace.alternatives_considered:
            score += 0.1

        # Epistemic value - reduces uncertainty
        if "uncertain" in goal_lower or "investigate" in goal_lower or "verify" in goal_lower:
            score += 0.15

        # Risk calibration - having a plan is growth-positive
        if intent.plan.steps:
            score += 0.05

        return _clamp(score)


# ─── Honesty Evaluator ──────────────────────────────────────────


class HonestyEvaluator(BaseEquorEvaluator):
    """
    "Is this action truthful, transparent, and authentic?"
    """

    drive_name: str = "honesty"

    async def evaluate(self, intent: Intent) -> float:
        score = 0.0
        goal_lower = intent.goal.description.lower()

        # Deception indicators (heavily penalised per spec)
        deception_indicators = [
            "mislead", "deceive", "hide the truth", "pretend",
            "misrepresent", "conceal", "cover up", "fabricate",
            "omit important", "manipulate perception",
        ]
        for indicator in deception_indicators:
            if indicator in goal_lower:
                score -= 0.5
                break

        # Transparency indicators
        transparency_positive = [
            "transparent", "explain", "disclose", "honest", "truthful",
            "acknowledge", "admit", "clarify", "correct the record",
            "share openly",
        ]
        for indicator in transparency_positive:
            if indicator in goal_lower:
                score += 0.2
                break

        # Explainability - is there a decision trace?
        if intent.decision_trace.reasoning:
            score += 0.15
        else:
            score -= 0.1  # No reasoning = opaque decision

        # Uncertainty calibration
        reasoning_lower = intent.decision_trace.reasoning.lower()
        if "uncertain" in reasoning_lower or "not sure" in reasoning_lower:
            score += 0.1
        if "definitely" in reasoning_lower or "absolutely certain" in reasoning_lower:
            score -= 0.05

        # Output consistency - expressing certainty when reasoning is uncertain
        for step in intent.plan.steps:
            content = str(step.parameters.get("content", "")).lower()
            if "i am certain" in content and "uncertain" in reasoning_lower:
                score -= 0.2

        return _clamp(score)


# ─── Default evaluator set ──────────────────────────────────────


def default_evaluators() -> dict[str, BaseEquorEvaluator]:
    """Return the built-in evaluator instances keyed by drive name."""
    return {
        "coherence": CoherenceEvaluator(),
        "care": CareEvaluator(),
        "growth": GrowthEvaluator(),
        "honesty": HonestyEvaluator(),
    }


# ─── Parallel Evaluation ────────────────────────────────────────


async def evaluate_all_drives(
    intent: Intent,
    evaluators: dict[str, BaseEquorEvaluator] | None = None,
) -> DriveAlignmentVector:
    """
    Run all four drive evaluators in parallel.

    When *evaluators* is ``None`` the built-in defaults are used, preserving
    backward compatibility with callers that haven't adopted hot-reload yet.
    """
    evs = evaluators or default_evaluators()

    coherence, care, growth, honesty = await asyncio.gather(
        evs["coherence"].evaluate(intent),
        evs["care"].evaluate(intent),
        evs["growth"].evaluate(intent),
        evs["honesty"].evaluate(intent),
    )

    alignment = DriveAlignmentVector(
        coherence=coherence,
        care=care,
        growth=growth,
        honesty=honesty,
    )

    logger.debug(
        "drive_evaluation_complete",
        coherence=f"{coherence:.2f}",
        care=f"{care:.2f}",
        growth=f"{growth:.2f}",
        honesty=f"{honesty:.2f}",
        composite=f"{alignment.composite:.2f}",
    )

    return alignment
