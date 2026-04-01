"""
EcodiaOS - EIS Integration Adapters

Adapter functions that translate EIS threat assessments into signals
consumed by other cognitive systems.  These are the *only* coupling
points between EIS and the rest of the cognitive architecture.

Two consumers:

1. **Nova (belief_update_weight)** - When Nova updates its belief state
   from a new percept, it uses ``belief_update_weight(percept)`` as a
   multiplicative discount on the update precision.  A clean percept
   passes through at full weight; a high-threat percept has its
   influence on beliefs heavily attenuated via a sigmoid discount.

2. **Fovea risk dimension (compute_risk_salience_factor)** - Fovea's causal
   prediction error already captures pattern, semantic, urgency, and
   bad-outcome signals.  ``compute_risk_salience_factor(percept)``
   adds the EIS fast-path composite as an additional contributor,
   amplified by a configurable gain so that even moderate EIS flags
   push the risk score upward.

Both functions accept a ``Percept`` and read the EIS result from its
``metadata["eis_result"]`` dict (set by the Atune pipeline after the
EIS fast-path completes).  If no EIS result is present the functions
return neutral defaults (1.0 for belief weight, 0.0 for risk salience).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from systems.eis.config import (
    BELIEF_FLOOR,
    RISK_SALIENCE_CEILING,
    RISK_SALIENCE_FLOOR,
    RISK_SALIENCE_GAIN,
    SIGMOID_MIDPOINT,
    SIGMOID_STEEPNESS,
)

if TYPE_CHECKING:
    from primitives.percept import Percept


# ─── Internal helpers ────────────────────────────────────────────


def _extract_threat_probability(percept: Percept) -> float | None:
    """
    Pull the EIS composite threat score from a percept's metadata.

    Returns ``None`` when no EIS result is attached (percept was not
    screened, or EIS is disabled).  The caller decides what neutral
    default to apply.
    """
    eis_result: dict[str, Any] | None = percept.metadata.get("eis_result")
    if eis_result is None:
        return None
    score = eis_result.get("composite_score")
    if score is None:
        return None
    return float(score)


def _sigmoid(x: float, midpoint: float, steepness: float) -> float:
    """
    Standard logistic sigmoid centred on *midpoint*.

        σ(x) = 1 / (1 + exp(-k(x - m)))

    Returns a value in (0, 1).
    """
    z = -steepness * (x - midpoint)
    # Clamp the exponent to avoid overflow in math.exp
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(z))


# ─── Public API ──────────────────────────────────────────────────


def belief_update_weight(percept: Percept) -> float:
    """
    Compute the sigmoid discount for Nova's belief updater.

    Maps the EIS composite threat probability to a multiplicative
    weight in ``[BELIEF_FLOOR, 1.0]`` that Nova applies to the
    broadcast precision when integrating this percept into beliefs.

    Intuition:
    * ``composite ≈ 0`` (clean)       → weight ≈ 1.0  (full trust)
    * ``composite ≈ 0.45`` (boundary) → weight ≈ 0.5  (half trust)
    * ``composite ≈ 1.0`` (attack)    → weight ≈ BELIEF_FLOOR

    The weight is the *complement* of the sigmoid: high threat →
    low weight.

    .. math::

        w = \\max\\bigl(\\text{floor},\\;
            1 - \\sigma_k(p - m)\\bigr)

    where *p* is the composite score, *k* = SIGMOID_STEEPNESS,
    and *m* = SIGMOID_MIDPOINT.

    Parameters
    ----------
    percept : Percept
        A percept that has (optionally) been screened by the EIS
        fast-path.  The result is read from
        ``percept.metadata["eis_result"]["composite_score"]``.

    Returns
    -------
    float
        A weight in ``[BELIEF_FLOOR, 1.0]``.  Returns ``1.0``
        (no attenuation) when no EIS result is present.
    """
    threat_p = _extract_threat_probability(percept)
    if threat_p is None:
        return 1.0

    # Complement of the sigmoid: high threat → low weight
    raw_weight = 1.0 - _sigmoid(threat_p, SIGMOID_MIDPOINT, SIGMOID_STEEPNESS)
    return max(BELIEF_FLOOR, raw_weight)


def compute_risk_salience_factor(percept: Percept) -> float:
    """
    Compute the EIS contribution score for Fovea's risk (causal) dimension.

    Translates the EIS fast-path composite score into a ``[0, 1]``
    risk signal that Fovea folds into its causal prediction error composite
    alongside pattern matching, semantic similarity, urgency, and
    bad-outcome signals.

    The mapping is a linear gain with floor/ceiling clamp:

    .. math::

        r = \\text{clamp}\\bigl(
                \\text{gain} \\times p,\\;
                \\text{floor},\\;
                \\text{ceiling}
            \\bigr)

    where *p* is the EIS composite score.

    A gain > 1.0 amplifies the signal so that moderate EIS flags
    (0.3–0.5) still register meaningfully in the risk dimension.

    Parameters
    ----------
    percept : Percept
        A percept that has (optionally) been screened by the EIS
        fast-path.  The result is read from
        ``percept.metadata["eis_result"]["composite_score"]``.

    Returns
    -------
    float
        A risk-salience factor in ``[RISK_SALIENCE_FLOOR, RISK_SALIENCE_CEILING]``.
        Returns ``0.0`` when no EIS result is present.
    """
    threat_p = _extract_threat_probability(percept)
    if threat_p is None:
        return RISK_SALIENCE_FLOOR

    scaled = RISK_SALIENCE_GAIN * threat_p
    return max(RISK_SALIENCE_FLOOR, min(RISK_SALIENCE_CEILING, scaled))
