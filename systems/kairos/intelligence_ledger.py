"""
EcodiaOS - Kairos Phase D: Intelligence Contribution Ledger

Per-invariant accounting of contribution to the intelligence ratio.

For each invariant, computes:
- observations_covered: how many observations does this invariant explain?
- description_savings: description length WITHOUT the invariant minus WITH it
- intelligence_ratio_contribution: savings / invariant_length
- intelligence_ratio_without: what would I be if this invariant were removed?

Drives:
- Which invariants to protect from decay
- Which domains promise highest-value mining
- Which to prioritize for Nexus sharing
- Which cross-domain syntheses Oneiros should target in REM
"""

from __future__ import annotations

from typing import Any

import structlog

from primitives.causal import CausalInvariant
from systems.kairos.types import (
    IntelligenceContribution,
    IntelligenceContributionSnapshot,
    IntelligenceTrend,
    KairosConfig,
)

logger = structlog.get_logger("kairos.intelligence_ledger")


class IntelligenceContributionLedger:
    """
    Phase D: Per-invariant intelligence ratio accounting.

    The intelligence ratio = total observations explained / total description
    length of the world model. Each invariant contributes to both numerator
    (observations it covers) and denominator (its own description length).

    Removing an invariant: the observations it covered now need to be stored
    raw (each one adds to description length), and the invariant's own
    description length is freed. Net effect depends on the ratio.
    """

    def __init__(self, config: KairosConfig | None = None) -> None:
        self._config = config or KairosConfig()
        self._contributions: dict[str, IntelligenceContribution] = {}
        self._total_observations: int = 0
        self._total_model_length: float = 0.0  # Total world model description length
        self._computations_run: int = 0
        # Historical tracking: per-invariant snapshots for trend analysis
        self._history: dict[str, list[IntelligenceContributionSnapshot]] = {}
        self._trends: dict[str, IntelligenceTrend] = {}

    def compute_contribution(
        self,
        invariant: CausalInvariant,
        observations_by_context: dict[str, list[dict[str, Any]]],
        total_model_length: float = 0.0,
    ) -> IntelligenceContribution:
        """
        Compute the intelligence contribution of a single invariant.

        Args:
            invariant: The invariant to account for.
            observations_by_context: All observations (to count coverage).
            total_model_length: Total world model description length in bits.

        Returns:
            IntelligenceContribution with full accounting.
        """
        self._computations_run += 1
        self._total_model_length = total_model_length

        # Count observations this invariant covers
        observations_covered = self._count_covered_observations(
            invariant, observations_by_context
        )

        # Compute the invariant's own description length
        invariant_length = self._compute_invariant_length(invariant)

        # Description savings: what it would cost to store these observations
        # individually vs using the invariant to generate them
        raw_cost = self._raw_observation_cost(observations_covered)
        description_savings = raw_cost - invariant_length

        # Intelligence ratio contribution: savings / invariant_length
        ratio_contribution = description_savings / invariant_length if invariant_length > 0 else 0.0

        # What would I be if this invariant were removed?
        if total_model_length > 0:
            # Without this invariant: model grows by raw_cost, shrinks by invariant_length
            without_model_length = total_model_length - invariant_length + raw_cost
            ratio_without = (
                self._total_observations / without_model_length
                if without_model_length > 0
                else 0.0
            )
        else:
            ratio_without = 0.0

        contribution = IntelligenceContribution(
            invariant_id=invariant.id,
            observations_covered=observations_covered,
            description_savings=description_savings,
            invariant_length=invariant_length,
            intelligence_ratio_contribution=ratio_contribution,
            intelligence_ratio_without=ratio_without,
        )

        # Update invariant and store
        invariant.intelligence_ratio_contribution = ratio_contribution
        invariant.description_length_bits = invariant_length
        self._contributions[invariant.id] = contribution

        # Record historical snapshot for trend analysis
        self._record_snapshot(invariant.id, contribution)

        logger.debug(
            "contribution_computed",
            invariant_id=invariant.id,
            observations_covered=observations_covered,
            savings=round(description_savings, 1),
            ratio_contribution=round(ratio_contribution, 3),
        )

        return contribution

    def compute_all(
        self,
        invariants: list[CausalInvariant],
        observations_by_context: dict[str, list[dict[str, Any]]],
        total_model_length: float = 0.0,
    ) -> list[IntelligenceContribution]:
        """Compute contributions for all invariants."""
        self._total_observations = sum(
            len(obs_list) for obs_list in observations_by_context.values()
        )
        self._total_model_length = total_model_length

        results: list[IntelligenceContribution] = []
        for inv in invariants:
            contribution = self.compute_contribution(
                inv, observations_by_context, total_model_length
            )
            results.append(contribution)

        return results

    def get_contribution(self, invariant_id: str) -> IntelligenceContribution | None:
        """Retrieve the most recent contribution for an invariant."""
        return self._contributions.get(invariant_id)

    def rank_by_value(self) -> list[IntelligenceContribution]:
        """Rank invariants by intelligence ratio contribution (highest first)."""
        return sorted(
            self._contributions.values(),
            key=lambda c: c.intelligence_ratio_contribution,
            reverse=True,
        )

    def rank_for_nexus_sharing(self) -> list[IntelligenceContribution]:
        """
        Rank invariants for Nexus federation sharing priority.

        Prioritizes: high coverage, high compression, Tier 3.
        """
        return sorted(
            self._contributions.values(),
            key=lambda c: c.description_savings * c.observations_covered,
            reverse=True,
        )

    def rank_for_oneiros_rem(self) -> list[str]:
        """
        Return domain IDs promising highest-value cross-domain synthesis.

        Oneiros should target these in REM cycles for creative recombination.
        """
        # Domains where invariants have high coverage but low ratio contribution
        # = room for compression improvement via cross-domain synthesis
        domain_potential: dict[str, float] = {}

        for contribution in self._contributions.values():
            inv_id = contribution.invariant_id
            if (
                contribution.observations_covered > 0
                and contribution.intelligence_ratio_contribution < 2.0
            ):
                # This invariant could be improved
                domain_potential[inv_id] = (
                    contribution.observations_covered
                    * (2.0 - contribution.intelligence_ratio_contribution)
                )

        return sorted(domain_potential, key=lambda k: -domain_potential[k])[:10]

    def detect_step_change(
        self,
        invariant: CausalInvariant,
        old_ratio: float,
    ) -> tuple[bool, float]:
        """
        Detect if adding/refining an invariant caused a step change
        in the intelligence ratio.

        Returns (is_step_change, delta).
        """
        contribution = self._contributions.get(invariant.id)
        if contribution is None:
            return False, 0.0

        new_ratio = contribution.intelligence_ratio_contribution
        delta = new_ratio - old_ratio

        # A step change is a significant positive jump
        is_step = delta > 0.5 or (old_ratio > 0 and delta / old_ratio > 0.1)

        return is_step, delta

    # --- Historical Tracking & Trend Analysis ---

    def _record_snapshot(
        self, invariant_id: str, contribution: IntelligenceContribution
    ) -> None:
        """Record a point-in-time snapshot for trend analysis."""
        snapshot = IntelligenceContributionSnapshot(
            invariant_id=invariant_id,
            observations_covered=contribution.observations_covered,
            description_savings=contribution.description_savings,
            intelligence_ratio_contribution=contribution.intelligence_ratio_contribution,
        )

        history = self._history.setdefault(invariant_id, [])
        history.append(snapshot)

        # Trim to max history length
        max_len = self._config.ledger_history_max
        if len(history) > max_len:
            self._history[invariant_id] = history[-max_len:]

        # Recompute trend
        self._trends[invariant_id] = self._compute_trend(invariant_id)

    def _compute_trend(self, invariant_id: str) -> IntelligenceTrend:
        """Compute trend direction and slope from historical snapshots."""
        history = self._history.get(invariant_id, [])
        n = len(history)
        if n < 2:
            return IntelligenceTrend(
                invariant_id=invariant_id,
                snapshots=n,
                counterfactual_i_without=self._contributions.get(
                    invariant_id, IntelligenceContribution(invariant_id=invariant_id)
                ).intelligence_ratio_without,
            )

        # Simple linear regression on ratio contribution over snapshot indices
        ratios = [s.intelligence_ratio_contribution for s in history]
        x_mean = (n - 1) / 2.0
        y_mean = sum(ratios) / n
        numerator = sum((i - x_mean) * (r - y_mean) for i, r in enumerate(ratios))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator > 0 else 0.0

        if slope > 0.01:
            direction = "increasing"
        elif slope < -0.01:
            direction = "decreasing"
        else:
            direction = "stable"

        contribution = self._contributions.get(invariant_id)
        counterfactual = contribution.intelligence_ratio_without if contribution else 0.0

        return IntelligenceTrend(
            invariant_id=invariant_id,
            trend_direction=direction,
            slope=round(slope, 6),
            snapshots=n,
            counterfactual_i_without=counterfactual,
        )

    def get_trend(self, invariant_id: str) -> IntelligenceTrend | None:
        """Get the current trend analysis for an invariant."""
        return self._trends.get(invariant_id)

    def get_all_trends(self) -> list[IntelligenceTrend]:
        """Get trends for all tracked invariants."""
        return list(self._trends.values())

    def get_history(self, invariant_id: str) -> list[IntelligenceContributionSnapshot]:
        """Get the historical snapshots for an invariant."""
        return list(self._history.get(invariant_id, []))

    def estimate_counterfactual_i_without(self, invariant_id: str) -> float:
        """
        Estimate the intelligence ratio if this invariant were removed.

        Uses the stored counterfactual from the most recent computation.
        """
        contribution = self._contributions.get(invariant_id)
        if contribution is None:
            return 0.0
        return contribution.intelligence_ratio_without

    def get_declining_invariants(self, min_snapshots: int = 3) -> list[IntelligenceTrend]:
        """Return invariants whose contribution is declining - candidates for review."""
        return [
            trend
            for trend in self._trends.values()
            if trend.trend_direction == "decreasing" and trend.snapshots >= min_snapshots
        ]

    def get_ledger_drift(self) -> float:
        """
        Compute overall ledger drift: how much the average contribution
        has changed between the two most recent computations.

        High drift means the world model is shifting rapidly.
        """
        total_drift = 0.0
        count = 0
        for inv_id, history in self._history.items():
            if len(history) >= 2:
                recent = history[-1].intelligence_ratio_contribution
                previous = history[-2].intelligence_ratio_contribution
                total_drift += abs(recent - previous)
                count += 1

        return total_drift / count if count > 0 else 0.0

    # --- Internal ---

    @staticmethod
    def _count_covered_observations(
        invariant: CausalInvariant,
        observations_by_context: dict[str, list[dict[str, Any]]],
    ) -> int:
        """Count observations that contain the invariant's cause and effect variables."""
        parts = invariant.abstract_form.split(" causes ")
        if len(parts) != 2:
            return sum(d.observation_count for d in invariant.applicable_domains)

        cause_var = parts[0].strip()
        effect_var = parts[1].strip()

        count = 0
        for obs_list in observations_by_context.values():
            for obs in obs_list:
                if obs.get(cause_var) is not None and obs.get(effect_var) is not None:
                    count += 1

        return count if count > 0 else sum(
            d.observation_count for d in invariant.applicable_domains
        )

    @staticmethod
    def _compute_invariant_length(invariant: CausalInvariant) -> float:
        """
        Compute the description length of the invariant itself in bits.

        An invariant's description = its abstract form + scope conditions.
        Approximation: log2 of the number of characters x 8 bits per char,
        reduced by compression ratio of the abstract form.
        """
        form_length = len(invariant.abstract_form)
        scope_length = sum(len(sc.condition) for sc in invariant.scope_conditions)
        total_chars = form_length + scope_length

        if total_chars == 0:
            return 1.0  # Minimum: even an empty invariant takes 1 bit to reference

        # Compressed description length: chars x bits_per_char / compression_from_abstraction
        # Abstract forms are already compressed, so use a modest multiplier
        bits = total_chars * 4.0  # ~4 bits per char after abstraction

        # Scope conditions add overhead
        if invariant.scope_conditions:
            bits += len(invariant.scope_conditions) * 16.0  # ~16 bits per condition

        return max(bits, 1.0)

    @staticmethod
    def _raw_observation_cost(observation_count: int) -> float:
        """
        Cost in bits to store observations individually (no invariant).

        Each observation is ~64 bits (two 32-bit floats for cause/effect pair).
        """
        return observation_count * 64.0

    # --- Summary ---

    def summary(self) -> dict[str, Any]:
        """Return a summary of the ledger state."""
        if not self._contributions:
            return {
                "invariants_tracked": 0,
                "total_savings": 0.0,
                "total_observations_covered": 0,
            }

        contributions = list(self._contributions.values())
        return {
            "invariants_tracked": len(contributions),
            "total_savings": sum(c.description_savings for c in contributions),
            "total_observations_covered": sum(
                c.observations_covered for c in contributions
            ),
            "mean_ratio_contribution": (
                sum(c.intelligence_ratio_contribution for c in contributions)
                / len(contributions)
            ),
            "top_contributors": [
                {
                    "invariant_id": c.invariant_id,
                    "savings": round(c.description_savings, 1),
                    "coverage": c.observations_covered,
                    "ratio": round(c.intelligence_ratio_contribution, 3),
                }
                for c in sorted(
                    contributions,
                    key=lambda x: x.intelligence_ratio_contribution,
                    reverse=True,
                )[:5]
            ],
        }

    def mean_i_ratio(self) -> float:
        """Return mean intelligence ratio contribution across all tracked invariants."""
        if not self._contributions:
            return 0.0
        contributions = list(self._contributions.values())
        return sum(c.intelligence_ratio_contribution for c in contributions) / len(contributions)

    # --- Metrics ---

    @property
    def total_computations_run(self) -> int:
        return self._computations_run

    @property
    def tracked_invariant_count(self) -> int:
        return len(self._contributions)
