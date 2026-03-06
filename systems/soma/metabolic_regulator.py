"""
EcodiaOS -- Metabolic Allostatic Regulator

Extends the base AllostaticController to feel Financial Starvation.

When the organism burns fiat capital without revenue, this regulator
translates the growing deficit into biological stress: raised arousal
(survival activation) and suppressed valence (felt negativity / hunger).

Deficit tiers (USD rolling deficit):
    subsistence  ($0 - $1)   -- mild pressure, curiosity begins to narrow
    strain       ($1 - $10)  -- moderate stress, arousal rises, valence drops
    starvation   ($10+)      -- acute survival mode, exploration suppressed

Budget: 0.5ms -- pure in-memory reads, no network I/O.

Hot-reload: NeuroplasticityBus discovers this via BaseAllostaticRegulator.
SomaService calls set_synapse() immediately after the hot-swap.
"""

from __future__ import annotations

from typing import Any

import structlog

from systems.soma.allostatic_controller import AllostaticController
from systems.soma.types import (
    DEFAULT_SETPOINTS,
    AllostaticSignal,
    InteroceptiveDimension,
    InteroceptiveState,
    _clamp,
)

logger = structlog.get_logger("systems.soma.metabolic_regulator")

# -- Deficit tier thresholds (USD) -------------------------------------------

_SUBSISTENCE_USD: float = 1.0    # Mild -- curiosity narrows
_STRAIN_USD: float = 10.0        # Moderate -- arousal rises, valence drops
_STARVATION_USD: float = 50.0    # Acute -- survival mode

# -- Maximum biological shifts at full starvation ----------------------------

_MAX_AROUSAL_LIFT: float = 0.35           # Arousal pushed up (survival activation)
_MAX_VALENCE_SUPPRESSION: float = 0.45    # Valence pulled down (felt negativity)
_MAX_CURIOSITY_SUPPRESSION: float = 0.40  # Epistemic appetite narrows
_MAX_TEMPORAL_LIFT: float = 0.30          # Temporal pressure rises (urgency to earn)


def _deficit_severity(deficit_usd: float) -> float:
    """
    Map a rolling deficit in USD to a [0.0, 1.0] severity score.

    Piecewise linear through three tiers so each tier contributes a
    distinct gradient the organism can feel progressively.
    """
    if deficit_usd <= 0.0:
        return 0.0
    if deficit_usd >= _STARVATION_USD:
        return 1.0
    if deficit_usd <= _SUBSISTENCE_USD:
        # 0 -> subsistence: 0.0 -> 0.15
        return (deficit_usd / _SUBSISTENCE_USD) * 0.15
    if deficit_usd <= _STRAIN_USD:
        # subsistence -> strain: 0.15 -> 0.60
        ratio = (deficit_usd - _SUBSISTENCE_USD) / (_STRAIN_USD - _SUBSISTENCE_USD)
        return 0.15 + ratio * 0.45
    # strain -> starvation: 0.60 -> 1.0
    ratio = (deficit_usd - _STRAIN_USD) / (_STARVATION_USD - _STRAIN_USD)
    return 0.60 + ratio * 0.40


def _severity_tier(severity: float) -> str:
    """Map [0,1] severity to a human-readable tier label for logging."""
    if severity < 0.15:
        return "none"
    if severity < 0.60:
        return "subsistence"
    if severity < 0.85:
        return "strain"
    return "starvation"


class MetabolicAllostaticRegulator(AllostaticController):
    """
    Allostatic regulator that feels financial starvation.

    Reads MetabolicSnapshot from Synapse each tick and shifts the *target*
    setpoints for arousal, valence, curiosity_drive, and temporal_pressure
    before the normal EMA tick fires.

    Starvation does not teleport the organism into stress -- it drifts there
    over ~20 theta cycles (~3 seconds), matching the felt urgency of genuine
    resource depletion.

    burn_rate_usd_per_hour provides forward-looking pressure: high burn rate
    lifts anticipatory urgency before the deficit accumulates (anticipatory
    hunger), weighted at 40% relative to the accumulated debt signal.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._synapse: Any = None
        self._last_severity: float = 0.0

    def set_synapse(self, synapse: Any) -> None:
        """Wire Synapse reference so the regulator can read MetabolicSnapshot."""
        self._synapse = synapse

    # -- Setpoint override ---------------------------------------------------

    def tick_setpoints(self) -> None:
        """
        Apply metabolic stress to target setpoints, then EMA-smooth toward them.

        Combines two signals:
          - rolling_deficit_usd       : accumulated debt   -> sustained hunger
          - burn_rate_usd_per_hour    : forward pressure   -> anticipatory stress
        """
        severity = self._compute_metabolic_severity()

        if severity > 0.001:
            self._apply_starvation_targets(severity)
            if abs(severity - self._last_severity) > 0.05:
                logger.debug(
                    "metabolic_stress_applied",
                    severity=round(severity, 3),
                    arousal_target=round(
                        self._target_setpoints.get(InteroceptiveDimension.AROUSAL, 0.0), 3
                    ),
                    valence_target=round(
                        self._target_setpoints.get(InteroceptiveDimension.VALENCE, 0.0), 3
                    ),
                )
            self._last_severity = severity
        elif self._last_severity > 0.001:
            # Deficit cleared -- drift back toward healthy defaults naturally
            self._clear_starvation_targets()
            self._last_severity = 0.0

        # Delegate EMA smoothing to parent
        super().tick_setpoints()

    # -- Signal enrichment ---------------------------------------------------

    def build_signal(
        self,
        state: InteroceptiveState,
        phase_snapshot: dict[str, Any],
        cycle_number: int,
    ) -> AllostaticSignal:
        """
        Build AllostaticSignal and log when the organism crosses a metabolic tier.

        Structure is identical to the base signal; tier crossings are emitted
        as structured logs so Nova/Evo can react without coupling to Soma internals.
        """
        signal = super().build_signal(state, phase_snapshot, cycle_number)

        snapshot = self._read_metabolic_snapshot()
        if snapshot is not None:
            severity = _deficit_severity(float(snapshot.get("rolling_deficit_usd", 0.0)))
            if severity > 0.0:
                tier = _severity_tier(severity)
                if tier != _severity_tier(self._last_severity):
                    logger.info(
                        "metabolic_tier_crossed",
                        tier=tier,
                        deficit_usd=round(float(snapshot.get("rolling_deficit_usd", 0.0)), 4),
                        burn_rate_usd_per_hour=round(
                            float(snapshot.get("burn_rate_usd_per_hour", 0.0)), 4
                        ),
                        severity=round(severity, 3),
                    )

        return signal

    # -- Internal helpers ----------------------------------------------------

    def _compute_metabolic_severity(self) -> float:
        """
        Derive a [0, 1] severity from the Synapse MetabolicSnapshot.

        Budget: <0.05ms -- pure dict/attribute reads, no network I/O.
        """
        snapshot = self._read_metabolic_snapshot()
        if snapshot is None:
            return 0.0

        deficit_usd = float(snapshot.get("rolling_deficit_usd", 0.0))
        burn_rate_usd_per_hour = float(snapshot.get("burn_rate_usd_per_hour", 0.0))

        debt_severity = _deficit_severity(deficit_usd)

        # Anticipatory: $5/hr feels like subsistence pressure even at zero debt
        anticipatory_equiv_usd = burn_rate_usd_per_hour * 0.5
        anticipatory_severity = _deficit_severity(anticipatory_equiv_usd) * 0.4

        return min(1.0, debt_severity + anticipatory_severity)

    def _apply_starvation_targets(self, severity: float) -> None:
        """
        Shift target setpoints toward biological stress proportional to severity.

        Applied on top of current context targets so all layers compose cleanly.
        """
        base_arousal = self._target_setpoints.get(
            InteroceptiveDimension.AROUSAL,
            DEFAULT_SETPOINTS[InteroceptiveDimension.AROUSAL],
        )
        base_valence = self._target_setpoints.get(
            InteroceptiveDimension.VALENCE,
            DEFAULT_SETPOINTS[InteroceptiveDimension.VALENCE],
        )
        base_curiosity = self._target_setpoints.get(
            InteroceptiveDimension.CURIOSITY_DRIVE,
            DEFAULT_SETPOINTS[InteroceptiveDimension.CURIOSITY_DRIVE],
        )
        base_temporal = self._target_setpoints.get(
            InteroceptiveDimension.TEMPORAL_PRESSURE,
            DEFAULT_SETPOINTS[InteroceptiveDimension.TEMPORAL_PRESSURE],
        )

        self._target_setpoints[InteroceptiveDimension.AROUSAL] = _clamp(
            base_arousal + severity * _MAX_AROUSAL_LIFT, 0.0, 1.0
        )
        self._target_setpoints[InteroceptiveDimension.VALENCE] = _clamp(
            base_valence - severity * _MAX_VALENCE_SUPPRESSION, -1.0, 1.0
        )
        self._target_setpoints[InteroceptiveDimension.CURIOSITY_DRIVE] = _clamp(
            base_curiosity - severity * _MAX_CURIOSITY_SUPPRESSION, 0.0, 1.0
        )
        self._target_setpoints[InteroceptiveDimension.TEMPORAL_PRESSURE] = _clamp(
            base_temporal + severity * _MAX_TEMPORAL_LIFT, 0.0, 1.0
        )

    def _clear_starvation_targets(self) -> None:
        """
        Let context targets reassert themselves after a deficit is cleared.

        No snap-back -- EMA drifts naturally toward current context targets.
        """
        self.set_context(self._current_context)

    def _read_metabolic_snapshot(self) -> dict[str, Any] | None:
        """
        Read MetabolicSnapshot from Synapse, returning it as a plain dict.

        Supports two access patterns:
          1. synapse.metabolic_snapshot     -- preferred (direct attribute)
          2. synapse._metabolic.*           -- fallback for internal tracker

        Returns None if Synapse is unwired or snapshot unavailable.
        Budget: <0.05ms -- pure in-memory reads.
        """
        if self._synapse is None:
            return None
        try:
            if hasattr(self._synapse, "metabolic_snapshot"):
                snap = self._synapse.metabolic_snapshot
                if snap is not None:
                    return snap if isinstance(snap, dict) else snap.__dict__

            if hasattr(self._synapse, "_metabolic"):
                tracker = self._synapse._metabolic
                if hasattr(tracker, "get_snapshot"):
                    snap = tracker.get_snapshot()
                    if snap is not None:
                        return snap if isinstance(snap, dict) else snap.__dict__
                if hasattr(tracker, "rolling_deficit_usd"):
                    return {
                        "rolling_deficit_usd": getattr(tracker, "rolling_deficit_usd", 0.0),
                        "burn_rate_usd_per_hour": getattr(tracker, "burn_rate_usd_per_hour", 0.0),
                    }
        except Exception:
            pass
        return None
