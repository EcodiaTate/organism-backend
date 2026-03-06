"""
EcodiaOS — EIS Configuration & Baseline Thresholds

Centralises all EIS tuning parameters that the integration layer and
other EIS modules reference.  Thresholds are derived from the composite
threat score produced by the fast-path pipeline (see embeddings.py →
``compute_composite_threat_score``).

Zone semantics
--------------
* **clean**          — composite < 0.20; no evidence of adversarial intent.
* **elevated**       — 0.20 ≤ composite < 0.45; mild anomalies, log only.
* **antigenic_zone** — 0.45 ≤ composite < 0.85; route to quarantine (LLM
                       evaluation) for deeper inspection.
* **known_attack**   — composite ≥ 0.85; matches known-bad signatures,
                       immediate block.

These boundaries align with ``EISConfig.quarantine_threshold`` (0.45) and
``EISConfig.block_threshold`` (0.85) in ``models.py``.  They are extracted
here so the integration adapters (Nova belief discount, Fovea risk dimension
contribution) can reference them without importing the full model layer.
"""

from __future__ import annotations

from typing import NamedTuple

# ─── Threat-Zone Thresholds ──────────────────────────────────────


class ZoneBounds(NamedTuple):
    """Inclusive lower and exclusive upper bounds for a threat zone."""

    lower: float  # inclusive
    upper: float  # exclusive (1.01 used as sentinel for the top zone)


THRESHOLDS: dict[str, ZoneBounds] = {
    "clean":          ZoneBounds(lower=0.00, upper=0.20),
    "elevated":       ZoneBounds(lower=0.20, upper=0.45),
    "antigenic_zone": ZoneBounds(lower=0.45, upper=0.85),
    "known_attack":   ZoneBounds(lower=0.85, upper=1.01),
}
"""Map from zone label → (inclusive lower, exclusive upper) composite-score bounds."""


# ─── Sigmoid Discount Parameters (used by belief_update_weight) ──


SIGMOID_STEEPNESS: float = 12.0
"""
Controls how sharply the sigmoid transitions from 'trust' to 'distrust'
around the midpoint.  Higher values create a sharper cliff; lower values
create a softer ramp.  12 places the 10%-to-90% transition window roughly
between composite scores of 0.30 and 0.70.
"""

SIGMOID_MIDPOINT: float = 0.45
"""
The composite-score at which the belief discount is exactly 0.5.
Aligned with the quarantine threshold — percepts at the quarantine
boundary receive half the normal belief-update weight.
"""

BELIEF_FLOOR: float = 0.05
"""
Minimum belief-update weight.  Even a confirmed attack still shifts
beliefs by a tiny amount so Nova registers that *something* adversarial
happened (the event itself is informative, even if the content is not
trusted).
"""


# ─── Risk Salience Parameters (used by compute_risk_salience_factor) ──


RISK_SALIENCE_GAIN: float = 1.4
"""
Linear gain applied to the EIS composite score before clamping.
Values > 1.0 make the risk dimension more responsive to moderate threats;
< 1.0 would make it less sensitive.
"""

RISK_SALIENCE_FLOOR: float = 0.0
"""
Minimum risk-salience contribution from EIS.
Set to 0.0 so truly clean percepts add no phantom risk signal.
"""

RISK_SALIENCE_CEILING: float = 1.0
"""
Maximum risk-salience contribution from EIS.
Capped at 1.0 to stay within the risk dimension's [0, 1] score range.
"""


# ─── Convenience ─────────────────────────────────────────────────


def classify_zone(composite_score: float) -> str:
    """
    Return the threat-zone label for a given EIS composite score.

    >>> classify_zone(0.10)
    'clean'
    >>> classify_zone(0.50)
    'antigenic_zone'
    >>> classify_zone(0.90)
    'known_attack'
    """
    for label, bounds in THRESHOLDS.items():
        if bounds.lower <= composite_score < bounds.upper:
            return label
    # Defensive fallback — should not be reachable with well-formed scores.
    return "known_attack" if composite_score >= 0.85 else "clean"
