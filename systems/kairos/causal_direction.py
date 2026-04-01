"""
EcodiaOS - Kairos Stage 2: Causal Direction Testing

Three independent tests. Agreement determines accepted direction.

Test 1 (Phase A): TEMPORAL PRECEDENCE
  Causes precede effects. If A consistently changes before B, A->B is supported.

Test 2 (Phase B): INTERVENTION ASYMMETRY
  Scan Axon execution logs. When EOS intervened on A, did B change?
  Asymmetric response = causal direction evidence.

Test 3 (Phase B): ADDITIVE NOISE MODEL
  For X->Y, residuals of regressing Y on X are independent of X.
  Wrong direction produces dependent residuals.
"""

from __future__ import annotations

import math
from typing import Any

import structlog

from systems.kairos.types import (
    AdditiveNoiseResult,
    CausalDirection,
    CausalDirectionResult,
    CorrelationCandidate,
    DirectionTestMethod,
    DirectionTestResult,
    InterventionEvidence,
    KairosConfig,
    TemporalEvidence,
)

logger = structlog.get_logger("kairos.causal_direction")


class CausalDirectionTester:
    """
    Stage 2 of the Kairos pipeline.

    Tests three independent signals for causal direction.
    Phase A implements temporal precedence only.
    Phase B adds intervention asymmetry and additive noise model.
    """

    def __init__(self, config: KairosConfig | None = None) -> None:
        self._config = config or KairosConfig()
        self._tests_run: int = 0
        self._accepted: int = 0

    async def test_direction(
        self,
        candidate: CorrelationCandidate,
        temporal_events: list[dict[str, Any]],
        axon_logs: list[dict[str, Any]] | None = None,
        observation_pairs: list[tuple[float, float]] | None = None,
    ) -> CausalDirectionResult:
        """
        Run all available direction tests on a correlation candidate.

        Args:
            candidate: The correlation candidate from Stage 1.
            temporal_events: List of dicts with at least {variable, value, timestamp_ms}.
            axon_logs: Axon execution logs for intervention asymmetry (Phase B).
            observation_pairs: Paired (a, b) values for additive noise model (Phase B).

        Returns:
            CausalDirectionResult with combined verdict.
        """
        self._tests_run += 1
        test_results: list[DirectionTestResult] = []

        # Test 1: Temporal precedence (Phase A)
        temporal_evidence = self._test_temporal_precedence(
            candidate.variable_a, candidate.variable_b, temporal_events
        )
        temporal_result = self._temporal_to_direction_result(temporal_evidence)
        test_results.append(temporal_result)

        # Test 2: Intervention asymmetry (Phase B)
        intervention_evidence: InterventionEvidence | None = None
        if axon_logs is not None:
            intervention_evidence = self._test_intervention_asymmetry(
                candidate.variable_a, candidate.variable_b, axon_logs
            )
            intervention_result = self._intervention_to_direction_result(intervention_evidence)
            test_results.append(intervention_result)

        # Test 3: Additive noise model (Phase B)
        noise_result_data: AdditiveNoiseResult | None = None
        if observation_pairs is not None and len(observation_pairs) >= 10:
            noise_result_data = self._test_additive_noise_model(observation_pairs)
            anm_result = self._noise_to_direction_result(noise_result_data)
            test_results.append(anm_result)

        # Vote aggregation: weighted by confidence
        votes: dict[str, float] = {
            CausalDirection.A_CAUSES_B: 0.0,
            CausalDirection.B_CAUSES_A: 0.0,
            CausalDirection.BIDIRECTIONAL: 0.0,
            CausalDirection.NO_CAUSAL: 0.0,
        }
        for tr in test_results:
            votes[tr.direction] += tr.confidence

        total_confidence = sum(votes.values())
        if total_confidence < 1e-12:
            winning = CausalDirection.NO_CAUSAL
            confidence = 0.0
        else:
            winning = CausalDirection(max(votes, key=lambda k: votes[k]))
            confidence = votes[winning] / total_confidence

        # Acceptance: confidence > threshold AND methods must agree
        methods_agreeing = sum(
            1 for tr in test_results if tr.direction == winning and tr.confidence > 0.1
        )
        min_agreement = min(2, len(test_results))
        accepted = (
            confidence > self._config.min_direction_confidence
            and methods_agreeing >= min_agreement
            and winning != CausalDirection.NO_CAUSAL
        )

        if accepted:
            self._accepted += 1

        result = CausalDirectionResult(
            candidate=candidate,
            direction=winning,
            confidence=confidence,
            accepted=accepted,
            test_results=test_results,
            temporal_evidence=temporal_evidence,
            intervention_evidence=intervention_evidence,
            noise_model_result=noise_result_data,
        )

        logger.info(
            "direction_test_complete",
            candidate_id=candidate.id,
            direction=winning.value,
            confidence=round(confidence, 3),
            accepted=accepted,
            methods_run=len(test_results),
            methods_agreeing=methods_agreeing,
        )

        return result

    # --- Test 1: Temporal Precedence (Phase A) ---

    def _test_temporal_precedence(
        self,
        var_a: str,
        var_b: str,
        events: list[dict[str, Any]],
    ) -> TemporalEvidence:
        """
        Count how often A changes before B vs B before A.

        Events must have: {variable: str, value: float, timestamp_ms: int|float}.
        """
        a_changes: list[float] = []
        b_changes: list[float] = []
        prev_a: float | None = None
        prev_b: float | None = None

        sorted_events = sorted(events, key=lambda e: e.get("timestamp_ms", 0))

        for event in sorted_events:
            var = event.get("variable", "")
            val = event.get("value")
            ts = event.get("timestamp_ms", 0)

            if not isinstance(val, (int, float)):
                continue

            if var == var_a:
                if prev_a is not None and abs(val - prev_a) > 1e-9:
                    a_changes.append(float(ts))
                prev_a = float(val)
            elif var == var_b:
                if prev_b is not None and abs(val - prev_b) > 1e-9:
                    b_changes.append(float(ts))
                prev_b = float(val)

        # For each B change, find the nearest preceding A change
        a_precedes_b = 0
        b_precedes_a = 0
        simultaneous = 0
        lag_sum = 0.0
        simultaneity_window_ms = 100.0

        for b_ts in b_changes:
            nearest_a_ts = self._find_nearest(a_changes, b_ts)
            if nearest_a_ts is None:
                continue
            lag = b_ts - nearest_a_ts
            if abs(lag) < simultaneity_window_ms:
                simultaneous += 1
            elif lag > 0:
                a_precedes_b += 1
                lag_sum += lag
            else:
                b_precedes_a += 1

        for a_ts in a_changes:
            nearest_b_ts = self._find_nearest(b_changes, a_ts)
            if nearest_b_ts is None:
                continue
            lag = a_ts - nearest_b_ts
            if abs(lag) < simultaneity_window_ms:
                pass  # Already counted
            elif lag > 0:
                b_precedes_a += 1
                lag_sum -= lag

        mean_lag = lag_sum / max(a_precedes_b, 1)

        return TemporalEvidence(
            a_precedes_b_count=a_precedes_b,
            b_precedes_a_count=b_precedes_a,
            simultaneous_count=simultaneous,
            mean_lag_ms=mean_lag,
        )

    def _temporal_to_direction_result(self, evidence: TemporalEvidence) -> DirectionTestResult:
        """Convert temporal evidence into a direction test result."""
        threshold = self._config.temporal_precedence_threshold

        if evidence.total < self._config.min_temporal_evidence_count:
            return DirectionTestResult(
                method=DirectionTestMethod.TEMPORAL_PRECEDENCE,
                direction=CausalDirection.NO_CAUSAL,
                confidence=0.0,
                evidence_count=evidence.total,
            )

        a_ratio = evidence.a_precedes_b_ratio
        b_ratio = evidence.b_precedes_a_ratio

        if a_ratio > threshold:
            direction = CausalDirection.A_CAUSES_B
            confidence = a_ratio
        elif b_ratio > threshold:
            direction = CausalDirection.B_CAUSES_A
            confidence = b_ratio
        elif a_ratio > 0.3 and b_ratio > 0.3:
            direction = CausalDirection.BIDIRECTIONAL
            confidence = min(a_ratio, b_ratio)
        else:
            direction = CausalDirection.NO_CAUSAL
            confidence = 0.0

        return DirectionTestResult(
            method=DirectionTestMethod.TEMPORAL_PRECEDENCE,
            direction=direction,
            confidence=confidence,
            evidence_count=evidence.total,
            details={
                "a_precedes_b_ratio": round(a_ratio, 3),
                "b_precedes_a_ratio": round(b_ratio, 3),
                "mean_lag_ms": round(evidence.mean_lag_ms, 1),
            },
        )

    # --- Test 2: Intervention Asymmetry (Phase B) ---

    def _test_intervention_asymmetry(
        self,
        var_a: str,
        var_b: str,
        axon_logs: list[dict[str, Any]],
    ) -> InterventionEvidence:
        """
        Scan Axon execution logs for intervention asymmetry.

        Expected log format: {
            action_type: str,
            target_variable: str,
            outcome_changes: list[str],
            timestamp_ms: float,
        }
        """
        interventions_on_a = 0
        b_changed_after_a = 0
        interventions_on_b = 0
        a_changed_after_b = 0

        for log_entry in axon_logs:
            target = log_entry.get("target_variable", "")
            outcome_changes = log_entry.get("outcome_changes", [])

            if target == var_a:
                interventions_on_a += 1
                if var_b in outcome_changes:
                    b_changed_after_a += 1
            elif target == var_b:
                interventions_on_b += 1
                if var_a in outcome_changes:
                    a_changed_after_b += 1

        return InterventionEvidence(
            interventions_on_a=interventions_on_a,
            b_changed_after_a_intervention=b_changed_after_a,
            interventions_on_b=interventions_on_b,
            a_changed_after_b_intervention=a_changed_after_b,
        )

    def _intervention_to_direction_result(
        self, evidence: InterventionEvidence
    ) -> DirectionTestResult:
        """Convert intervention evidence into a direction test result."""
        a_to_b = evidence.a_causes_b_score
        b_to_a = evidence.b_causes_a_score
        total_interventions = evidence.interventions_on_a + evidence.interventions_on_b

        if total_interventions < 3:
            return DirectionTestResult(
                method=DirectionTestMethod.INTERVENTION_ASYMMETRY,
                direction=CausalDirection.NO_CAUSAL,
                confidence=0.0,
                evidence_count=total_interventions,
            )

        asymmetry = abs(a_to_b - b_to_a)
        if asymmetry < 0.2:
            if a_to_b > 0.5 and b_to_a > 0.5:
                direction = CausalDirection.BIDIRECTIONAL
                confidence = min(a_to_b, b_to_a) * 0.7
            else:
                direction = CausalDirection.NO_CAUSAL
                confidence = 0.0
        elif a_to_b > b_to_a:
            direction = CausalDirection.A_CAUSES_B
            confidence = asymmetry * min(a_to_b, 1.0)
        else:
            direction = CausalDirection.B_CAUSES_A
            confidence = asymmetry * min(b_to_a, 1.0)

        return DirectionTestResult(
            method=DirectionTestMethod.INTERVENTION_ASYMMETRY,
            direction=direction,
            confidence=confidence,
            evidence_count=total_interventions,
            details={
                "a_causes_b_score": round(a_to_b, 3),
                "b_causes_a_score": round(b_to_a, 3),
                "asymmetry": round(asymmetry, 3),
            },
        )

    # --- Test 3: Additive Noise Model (Phase B) ---

    def _test_additive_noise_model(
        self,
        pairs: list[tuple[float, float]],
    ) -> AdditiveNoiseResult:
        """
        Additive Noise Model test.

        For the correct direction X->Y: Y = f(X) + noise, where noise is
        independent of X. For the wrong direction, noise depends on Y.
        """
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]

        residuals_a_to_b = self._linear_residuals(xs, ys)
        independence_a_to_b = self._residual_independence(xs, residuals_a_to_b)

        residuals_b_to_a = self._linear_residuals(ys, xs)
        independence_b_to_a = self._residual_independence(ys, residuals_b_to_a)

        return AdditiveNoiseResult(
            residual_independence_a_to_b=independence_a_to_b,
            residual_independence_b_to_a=independence_b_to_a,
        )

    def _noise_to_direction_result(self, result: AdditiveNoiseResult) -> DirectionTestResult:
        """Convert ANM result into a direction test result."""
        a_to_b = result.residual_independence_a_to_b
        b_to_a = result.residual_independence_b_to_a

        diff = a_to_b - b_to_a

        if abs(diff) < 0.1:
            direction = CausalDirection.NO_CAUSAL
            confidence = 0.0
        elif diff > 0:
            direction = CausalDirection.A_CAUSES_B
            confidence = min(diff * 2, 1.0)
        else:
            direction = CausalDirection.B_CAUSES_A
            confidence = min(abs(diff) * 2, 1.0)

        return DirectionTestResult(
            method=DirectionTestMethod.ADDITIVE_NOISE_MODEL,
            direction=direction,
            confidence=confidence,
            evidence_count=1,
            details={
                "independence_a_to_b": round(a_to_b, 4),
                "independence_b_to_a": round(b_to_a, 4),
            },
        )

    # --- Statistical helpers ---

    @staticmethod
    def _find_nearest(timestamps: list[float], target: float) -> float | None:
        """Find the timestamp in the list nearest to target."""
        if not timestamps:
            return None
        return min(timestamps, key=lambda t: abs(t - target))

    @staticmethod
    def _linear_residuals(x: list[float], y: list[float]) -> list[float]:
        """Fit y = a*x + b and return residuals."""
        n = len(x)
        if n < 2:
            return [0.0] * n

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        ss_xx = sum((xi - mean_x) ** 2 for xi in x)
        if ss_xx < 1e-12:
            return [yi - mean_y for yi in y]

        ss_xy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y, strict=True))
        slope = ss_xy / ss_xx
        intercept = mean_y - slope * mean_x

        return [yi - (slope * xi + intercept) for xi, yi in zip(x, y, strict=True)]

    @staticmethod
    def _residual_independence(cause: list[float], residuals: list[float]) -> float:
        """
        Measure independence of residuals from cause.
        Returns 1.0 for perfect independence, 0.0 for perfect dependence.
        """
        n = len(cause)
        if n < 3:
            return 0.5

        mean_c = sum(cause) / n
        mean_r = sum(residuals) / n

        cov = sum((c - mean_c) * (r - mean_r) for c, r in zip(cause, residuals, strict=True)) / n
        var_c = sum((c - mean_c) ** 2 for c in cause) / n
        var_r = sum((r - mean_r) ** 2 for r in residuals) / n

        if var_c < 1e-12 or var_r < 1e-12:
            return 1.0

        correlation = cov / math.sqrt(var_c * var_r)
        return 1.0 - abs(correlation)

    # --- Metrics ---

    @property
    def total_tests_run(self) -> int:
        return self._tests_run

    @property
    def total_accepted(self) -> int:
        return self._accepted
