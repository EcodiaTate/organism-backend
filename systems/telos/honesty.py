"""
EcodiaOS — Telos: Honesty Topology Engine

Honesty is not truthfulness in communication. It is the validity of the
intelligence metric itself. A world model allowed to ignore inconvenient data,
avoid testing hypotheses, or maintain beliefs against contradicting evidence
has an inflated nominal I and a deflated effective I.

The Honesty engine detects four modes of dishonesty and computes the
honesty_validity_coefficient that corrects the intelligence measurement.
"""

from __future__ import annotations

import structlog

from systems.telos.types import (
    FoveaMetrics,
    HonestyReport,
    LogosMetrics,
    TelosConfig,
)

logger = structlog.get_logger()


class HonestyTopologyEngine:
    """
    Implements Honesty as the validity coefficient of the intelligence ratio.

    Detects four modes of dishonesty that inflate nominal I:

    1. SELECTIVE ATTENTION: Only counting observations the model gets right.
       Measured by comparing observed success rate against expected success rate.

    2. HYPOTHESIS PROTECTION: Refusing to test hypotheses that might fail.
       Measured by hypothesis test rate vs expected test rate.

    3. CONFABULATION: Generating post-hoc explanations for unpredicted observations.
       Measured by Fovea's confabulation rate.

    4. COVERAGE OVERCLAIMING: Asserting coverage in domains never encountered.
       Measured by Fovea's overclaiming rate.

    honesty_validity_coefficient = 1.0 - weighted_sum(biases)

    A coefficient of 1.0 means perfect honesty (I measures real predictive power).
    A coefficient of 0.5 means half the claimed I is illusory.
    """

    def __init__(self, config: TelosConfig) -> None:
        self._config = config
        self._logger = logger.bind(component="telos.honesty")

    async def compute_validity(
        self,
        logos: LogosMetrics,
        fovea: FoveaMetrics,
        *,
        measured_hypothesis_protection_bias: float | None = None,
        measured_confabulation_rate: float | None = None,
    ) -> HonestyReport:
        """
        Compute the honesty validity coefficient.

        Steps:
        1. Measure selective attention bias (success rate vs expected)
        2. Measure confabulation rate (post-hoc explanation generation)
        3. Measure coverage overclaiming (claims in untested domains)
        4. Compute nominal I inflation
        5. Aggregate into validity_coefficient
        """
        nominal_I = await logos.get_intelligence_ratio()
        prediction_success_rate = await fovea.get_prediction_success_rate()

        # 1. Selective attention bias
        selective_attention_bias = self._measure_selective_attention(
            prediction_success_rate=prediction_success_rate,
            prediction_error_rate=await fovea.get_prediction_error_rate(),
        )

        # 2. Confabulation rate — use incident-based measurement when available
        # (>= 10 INCIDENT_RESOLVED events from Thymos), otherwise fall back to
        # Fovea's false-alarm heuristic (Spec 18 §SG3).
        if measured_confabulation_rate is not None:
            confabulation_rate = max(0.0, min(1.0, measured_confabulation_rate))
        else:
            confabulation_rate = await fovea.get_confabulation_rate()

        # 3. Coverage overclaiming
        overclaiming_rate = await fovea.get_overclaiming_rate()

        # 4. Hypothesis protection bias — use measured data from Evo
        # hypothesis tracking when available (>= 10 observations),
        # otherwise fall back to the heuristic estimate.
        if measured_hypothesis_protection_bias is not None:
            hypothesis_protection_bias = max(
                0.0, min(1.0, measured_hypothesis_protection_bias)
            )
        else:
            hypothesis_protection_bias = self._estimate_hypothesis_protection(
                overclaiming_rate=overclaiming_rate,
                confabulation_rate=confabulation_rate,
            )

        total_invalidity = (
            selective_attention_bias * _SELECTIVE_ATTENTION_WEIGHT
            + hypothesis_protection_bias * _HYPOTHESIS_PROTECTION_WEIGHT
            + confabulation_rate * _CONFABULATION_WEIGHT
            + overclaiming_rate * _OVERCLAIMING_WEIGHT
        )

        validity_coefficient = max(0.0, min(1.0, 1.0 - total_invalidity))

        # How much of nominal I is inflated beyond reality
        nominal_I_inflation = nominal_I * (1.0 - validity_coefficient)

        report = HonestyReport(
            selective_attention_bias=selective_attention_bias,
            hypothesis_protection_bias=hypothesis_protection_bias,
            confabulation_rate=confabulation_rate,
            overclaiming_rate=overclaiming_rate,
            validity_coefficient=validity_coefficient,
            nominal_I_inflation=nominal_I_inflation,
        )

        self._logger.debug(
            "honesty_validity_computed",
            coefficient=f"{validity_coefficient:.3f}",
            selective_attention=f"{selective_attention_bias:.3f}",
            hypothesis_protection=f"{hypothesis_protection_bias:.3f}",
            confabulation=f"{confabulation_rate:.3f}",
            overclaiming=f"{overclaiming_rate:.3f}",
            inflation=f"{nominal_I_inflation:.4f}",
        )

        return report

    def _measure_selective_attention(
        self,
        prediction_success_rate: float,
        prediction_error_rate: float,
    ) -> float:
        """
        Detect selective attention bias.

        If the observed success rate is much higher than what the error rate
        implies (1 - error_rate), the model is selectively attending to
        successes and ignoring failures.

        A well-calibrated model has: success_rate approx 1 - error_rate.
        Bias = max(0, success_rate - (1 - error_rate)).
        """
        expected_success_rate = 1.0 - prediction_error_rate
        bias = max(0.0, prediction_success_rate - expected_success_rate)
        return min(1.0, bias)

    def _estimate_hypothesis_protection(
        self,
        overclaiming_rate: float,
        confabulation_rate: float,
    ) -> float:
        """
        Estimate hypothesis protection bias.

        In the absence of a full Evo integration (which would track actual
        hypothesis test rates), we estimate protection bias from related
        signals: high overclaiming + low confabulation suggests the model
        is avoiding falsification rather than actively confabulating.

        High overclaiming + high confabulation suggests active dishonesty.
        High overclaiming + low confabulation suggests passive avoidance.
        """
        # Hypothesis protection is correlated with overclaiming but dampened
        # by confabulation (an actively confabulating model isn't avoiding tests,
        # it's failing them and lying about it).
        if overclaiming_rate > 0.1 and confabulation_rate < 0.2:
            # Passive avoidance pattern: high overclaiming, low confabulation
            return min(1.0, overclaiming_rate * 0.8)
        elif overclaiming_rate > 0.1 and confabulation_rate >= 0.2:
            # Active dishonesty pattern: both high
            return min(1.0, (overclaiming_rate + confabulation_rate) * 0.4)
        else:
            return 0.0


# ─── Weights ─────────────────────────────────────────────────────────
# These sum to 1.0 and determine each dishonesty mode's contribution
# to the total invalidity score.

_SELECTIVE_ATTENTION_WEIGHT = 0.35
_HYPOTHESIS_PROTECTION_WEIGHT = 0.30
_CONFABULATION_WEIGHT = 0.20
_OVERCLAIMING_WEIGHT = 0.15
