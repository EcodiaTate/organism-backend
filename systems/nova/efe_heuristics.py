"""
EcodiaOS — Nova EFE Heuristic Fallbacks

Fast approximations for LLM-based components when:
- Budget is exhausted (Red/Yellow tier)
- Latency exceeds timeout
- LLM provider unavailable

All heuristics are deterministic and fast (<10ms).
Designed to maintain coherent behavior without LLM calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from systems.nova.types import BeliefState, Goal, Policy

logger = structlog.get_logger()


class EFEHeuristics:
    """Fast approximations for Nova's EFE components."""

    @staticmethod
    def estimate_pragmatic_value_heuristic(
        policy: Policy,
        goal: Goal,
        beliefs: BeliefState,
    ) -> float:
        """
        Heuristic pragmatic value: probability of goal satisfaction.

        LLM alternative: Asks model to estimate goal probability under policy.
        Heuristic: Use recent history of similar policies.

        Returns:
            Float in [0.0, 1.0], where 1.0 = certain goal satisfaction
        """
        # Base case: do-nothing is low probability for goal achievement
        if policy.type == "do_nothing":
            return 0.1

        # Heuristic: actionable policies are more likely to achieve goals
        action_types_to_weight = {
            "deliberate": 0.7,
            "express": 0.6,
            "observe": 0.4,
            "defer": 0.2,
        }

        base_score = action_types_to_weight.get(policy.type, 0.5)

        # Discount if policy conflicts with recent goals
        # (Simple: if policy name contains opposite keywords)
        if goal.description and policy.description:
            goal_lower = goal.description.lower()
            policy_lower = policy.description.lower()

            # Check for semantic opposition (crude but fast)
            opposing_pairs = [
                ("increase", "decrease"),
                ("approach", "avoid"),
                ("clarify", "obfuscate"),
            ]
            for pos, neg in opposing_pairs:
                if (
                    pos in goal_lower and neg in policy_lower
                    or neg in goal_lower and pos in policy_lower
                ):
                    base_score *= 0.3

        return min(1.0, max(0.0, base_score))

    @staticmethod
    def estimate_epistemic_value_heuristic(
        policy: Policy,
        beliefs: BeliefState,
    ) -> float:
        """
        Heuristic epistemic value: expected information gain.

        LLM alternative: Asks model to estimate uncertainty reduction.
        Heuristic: Information-seeking policies (observe, ask) gain more.

        Returns:
            Float in [0.0, 1.0], where 1.0 = maximum information gain
        """
        # Observation and expression policies are epistemic
        epistemic_policies = {"observe", "ask", "clarify", "explore"}

        base_score = 0.5 if policy.type in epistemic_policies else 0.2

        # Belief entropy as proxy for information need
        # (If beliefs are very certain, less room for info gain)
        # Discount epistemic value if already very certain
        # BeliefState has overall_confidence (0=uncertain, 1=certain)
        certainty = getattr(beliefs, "overall_confidence", None)
        if certainty is not None:
            base_score *= (1.0 - certainty)

        return min(1.0, max(0.0, base_score))

    @staticmethod
    def estimate_feasibility_heuristic(policy: Policy) -> float:
        """
        Heuristic feasibility: can we actually execute this policy?

        LLM alternative: Would ask model to assess resource/capability fit.
        Heuristic: Simpler policies are more feasible.

        Returns:
            Float in [0.0, 1.0], where 1.0 = certain feasibility
        """
        # Simpler policies are more feasible
        complexity_estimate = len(policy.description.split()) / 10.0 if policy.description else 0.5
        complexity_estimate = min(1.0, max(0.0, complexity_estimate))

        # Simple actions: high feasibility
        simple_policies = {"do_nothing", "observe", "wait", "express"}
        if policy.type in simple_policies:
            return 0.9

        # Complex / risky actions: lower feasibility
        risky_policies = {"external_api", "federate", "irreversible"}
        if policy.type in risky_policies:
            return max(0.2, 1.0 - complexity_estimate)

        # Standard actions: moderate feasibility
        return max(0.5, 1.0 - complexity_estimate * 0.3)

    @staticmethod
    def estimate_risk_heuristic(policy: Policy) -> float:
        """
        Heuristic risk: expected harm if policy goes wrong.

        LLM alternative: Would ask model to identify downsides.
        Heuristic: Certain action types carry known risks.

        Returns:
            Float in [0.0, 1.0], where 1.0 = maximum risk
        """
        # High-risk action types
        high_risk_types = {"irreversible", "external_api", "federate"}
        if policy.type in high_risk_types:
            return 0.7

        # Low-risk action types
        low_risk_types = {"observe", "wait", "defer"}
        if policy.type in low_risk_types:
            return 0.1

        # Expression carries moderate risk (social)
        if policy.type == "express":
            return 0.3

        # Default: low-moderate risk
        return 0.2

    @staticmethod
    def estimate_constitutional_alignment_heuristic(
        policy: Policy,
        drive_weights: dict[str, float],
    ) -> float:
        """
        Heuristic constitutional fit: alignment with drives (Coherence, Care, Growth, Honesty).

        LLM alternative: Would ask model to evaluate.
        Heuristic: Certain policy types align with known drives.

        Returns:
            Float in [0.0, 1.0], where 1.0 = perfect alignment
        """
        # Policy-to-drive affinities (simple lookup)
        affinities = {
            "observe": {"growth": 0.8, "coherence": 0.7},
            "express": {"care": 0.8, "coherence": 0.7, "honesty": 0.8},
            "clarify": {"coherence": 0.9, "honesty": 0.8},
            "defer": {"care": 0.5},
            "do_nothing": {"honesty": 0.3},
        }

        policy_affinities = affinities.get(policy.type, {})

        # Weighted sum of drive alignments
        total_alignment = 0.0
        total_weight = 0.0

        for drive, affinity in policy_affinities.items():
            weight = drive_weights.get(drive, 0.0)
            total_alignment += affinity * weight
            total_weight += weight

        if total_weight == 0:
            return 0.5  # Neutral if no drive weights

        return min(1.0, max(0.0, total_alignment / total_weight))

    @staticmethod
    def log_heuristic_fallback(
        system: str,
        reason: str,
        policy_type: str,
    ) -> None:
        """Log when a heuristic is used instead of LLM."""
        logger.info(
            "heuristic_fallback",
            system=system,
            reason=reason,
            policy_type=policy_type,
        )
