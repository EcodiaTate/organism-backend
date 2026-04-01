"""
EcodiaOS - Kairos Phase D: Counter-Invariant Detection

Actively searches for violations of accepted invariants.

For each invariant:
1. Scan observations in applicable_domains for violations
2. Record InvariantViolation with context and distinguishing features
3. Cluster violations by context -> find common distinguishing feature
4. If found: refine invariant scope (it holds everywhere EXCEPT contexts
   with this feature)
5. Check if distinguishing feature is a significant modulator (not noise)

Refined invariants are more precise, not weaker - knowing the boundary
condition is itself knowledge.
"""

from __future__ import annotations

import math
from typing import Any

import structlog

from primitives.common import utc_now
from primitives.causal import CausalInvariant
from systems.kairos.types import (
    InvariantViolation,
    KairosConfig,
    RefinedScope,
    ViolationCluster,
)

logger = structlog.get_logger("kairos.counter_invariant")


class CounterInvariantDetector:
    """
    Phase D: Actively searches for violations of accepted invariants.

    Finding a boundary condition is knowledge - a refined invariant is
    more precise, not weaker. The invariant "X causes Y everywhere except
    when Z > threshold" is strictly more informative than "X causes Y".
    """

    def __init__(self, config: KairosConfig | None = None) -> None:
        self._config = config or KairosConfig()
        self._scans_run: int = 0
        self._violations_found: int = 0
        self._refinements_made: int = 0

    async def scan_for_violations(
        self,
        invariant: CausalInvariant,
        observations_by_context: dict[str, list[dict[str, Any]]],
    ) -> list[InvariantViolation]:
        """
        Scan observations across all applicable domains for violations.

        A violation occurs when the invariant predicts a correlation
        direction but the observed data shows the opposite or no correlation.

        Args:
            invariant: The accepted invariant to test.
            observations_by_context: context_id to observation lists.

        Returns:
            List of InvariantViolation for each violating context.
        """
        self._scans_run += 1
        violations: list[InvariantViolation] = []

        # Parse cause/effect from abstract form
        parts = invariant.abstract_form.split(" causes ")
        if len(parts) != 2:
            return violations
        cause_var = parts[0].strip()
        effect_var = parts[1].strip()

        for ctx_id, obs_list in observations_by_context.items():
            # Extract paired values
            cause_vals: list[float] = []
            effect_vals: list[float] = []
            for obs in obs_list:
                c = obs.get(cause_var)
                e = obs.get(effect_var)
                if isinstance(c, (int, float)) and isinstance(e, (int, float)):
                    cause_vals.append(float(c))
                    effect_vals.append(float(e))

            if len(cause_vals) < 3:
                continue

            # Compute correlation in this context
            r = _pearson(cause_vals, effect_vals)
            if r is None:
                continue

            # Use the stored causal direction to determine expected sign.
            # direction="" means unknown - fall back to positive assumption.
            expected_positive = invariant.direction != "negative"
            is_violation = False
            if expected_positive and r < -0.1:
                is_violation = True
            elif not expected_positive and r > 0.1:
                is_violation = True

            if is_violation:
                # Compute distinguishing features of this context
                features = self._compute_context_features(ctx_id, obs_list)

                violation = InvariantViolation(
                    invariant_id=invariant.id,
                    violating_observation={"context_id": ctx_id, "correlation": r},
                    violation_context=ctx_id,
                    expected_direction="positive" if expected_positive else "negative",
                    observed_value=r,
                    distinguishing_features=features,
                )
                violations.append(violation)

        self._violations_found += len(violations)
        invariant.violation_count = len(violations)
        invariant.last_violation_check = utc_now()

        if violations:
            logger.info(
                "violations_found",
                invariant_id=invariant.id,
                violation_count=len(violations),
                contexts_scanned=len(observations_by_context),
            )

        return violations

    async def cluster_violations(
        self,
        invariant: CausalInvariant,
        violations: list[InvariantViolation],
    ) -> list[ViolationCluster]:
        """
        Cluster violations by shared distinguishing features.

        Finds the common feature that distinguishes violating contexts
        from holding contexts. If a significant modulator is found,
        the invariant's scope can be refined.
        """
        if len(violations) < self._config.min_violations_for_cluster:
            return []

        # Collect all features across violations
        feature_sums: dict[str, float] = {}
        feature_counts: dict[str, int] = {}

        for v in violations:
            for feat, val in v.distinguishing_features.items():
                feature_sums[feat] = feature_sums.get(feat, 0.0) + val
                feature_counts[feat] = feature_counts.get(feat, 0) + 1

        clusters: list[ViolationCluster] = []

        for feat in feature_sums:
            count = feature_counts[feat]
            if count < self._config.min_cluster_size_for_refinement:
                continue

            mean_val = feature_sums[feat] / count

            # Check if this feature is consistent across violations
            vals = [
                v.distinguishing_features.get(feat, 0.0)
                for v in violations
                if feat in v.distinguishing_features
            ]
            variance = _variance(vals)

            # Low variance across violations = consistent distinguishing feature
            is_significant = (
                variance < 0.5
                and abs(mean_val) > self._config.violation_significance_threshold
            )

            cluster = ViolationCluster(
                invariant_id=invariant.id,
                violations=[v.id for v in violations if feat in v.distinguishing_features],
                common_feature=feat,
                feature_threshold=mean_val,
                is_significant=is_significant,
                cluster_size=count,
            )
            clusters.append(cluster)

        # Sort by cluster size (largest first)
        clusters.sort(key=lambda c: -c.cluster_size)

        if clusters:
            logger.info(
                "violation_clusters_found",
                invariant_id=invariant.id,
                cluster_count=len(clusters),
                significant_count=sum(1 for c in clusters if c.is_significant),
            )

        return clusters

    async def refine_scope(
        self,
        invariant: CausalInvariant,
        clusters: list[ViolationCluster],
        observations_by_context: dict[str, list[dict[str, Any]]],
    ) -> RefinedScope | None:
        """
        Refine the invariant's scope based on violation clusters.

        Produces a boundary condition: "holds everywhere EXCEPT contexts
        where feature X > threshold".

        Returns None if no significant cluster was found.
        """
        significant = [c for c in clusters if c.is_significant]
        if not significant:
            return None

        # Use the largest significant cluster
        best = significant[0]

        # Count contexts where invariant holds excluding the cluster
        total_contexts = len(observations_by_context)
        excluded_contexts = best.cluster_size
        remaining_contexts = total_contexts - excluded_contexts
        if remaining_contexts <= 0:
            return None

        # Recompute hold rate: original holding contexts / remaining contexts
        original_holding = int(invariant.invariance_hold_rate * total_contexts)
        refined_hold_rate = min(original_holding / remaining_contexts, 1.0)

        boundary = (
            f"holds everywhere EXCEPT contexts where "
            f"{best.common_feature} > {best.feature_threshold:.3f}"
        )

        refined = RefinedScope(
            invariant_id=invariant.id,
            original_hold_rate=invariant.invariance_hold_rate,
            refined_hold_rate=refined_hold_rate,
            boundary_condition=boundary,
            excluded_feature=best.common_feature,
            excluded_threshold=best.feature_threshold,
            contexts_excluded=excluded_contexts,
        )

        # Update invariant
        invariant.refined_scope = boundary
        self._refinements_made += 1

        logger.info(
            "invariant_scope_refined",
            invariant_id=invariant.id,
            boundary=boundary,
            original_hold_rate=round(invariant.invariance_hold_rate, 3),
            refined_hold_rate=round(refined_hold_rate, 3),
            excluded_contexts=excluded_contexts,
        )

        return refined

    # --- Helpers ---

    @staticmethod
    def _compute_context_features(
        context_id: str,
        observations: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Compute mean of each numeric variable in a context as features."""
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}

        for obs in observations:
            for k, v in obs.items():
                if isinstance(v, (int, float)) and not math.isnan(v):
                    sums[k] = sums.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1

        return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}

    # --- Metrics ---

    @property
    def total_scans_run(self) -> int:
        return self._scans_run

    @property
    def total_violations_found(self) -> int:
        return self._violations_found

    @property
    def total_refinements_made(self) -> int:
        return self._refinements_made


# --- Module-level helpers ---


def _pearson(x: list[float], y: list[float]) -> float | None:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n < 3:
        return None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True)) / n
    var_x = sum((xi - mean_x) ** 2 for xi in x) / n
    var_y = sum((yi - mean_y) ** 2 for yi in y) / n

    if var_x < 1e-12 or var_y < 1e-12:
        return None

    return cov / math.sqrt(var_x * var_y)


def _variance(values: list[float]) -> float:
    """Compute population variance."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)
