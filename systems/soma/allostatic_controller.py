"""
EcodiaOS — Soma Allostatic Controller

Manages setpoints, constructs the AllostaticSignal, and computes urgency.

Setpoints are the organism's targets — where it "wants" each dimension to be.
They adapt based on context (conversation, deep processing, recovery, exploration)
using EMA smoothing to prevent oscillation.

Urgency = max(|errors|) * max(|error_rates|) — a composite need-to-act signal.
When urgency > threshold, Nova triggers allostatic deliberation.
"""

from __future__ import annotations

from typing import Any

import structlog

from systems.soma.base import BaseAllostaticRegulator
from systems.soma.types import (
    ALL_DIMENSIONS,
    DEFAULT_SETPOINTS,
    SETPOINT_CONTEXTS,
    AllostaticSignal,
    InteroceptiveDimension,
    InteroceptiveState,
)

logger = structlog.get_logger("systems.soma.allostatic_controller")


class AllostaticController(BaseAllostaticRegulator):
    """
    Manages allostatic setpoints and constructs the output signal.

    Responsibilities:
    - Maintain current setpoints (context-adaptive, EMA-smoothed)
    - Compute precision weights from prediction confidence
    - Build the AllostaticSignal from interoceptive state + phase-space
    - Compute urgency as the drive for allostatic regulation
    """

    def __init__(
        self,
        default_setpoints: dict[InteroceptiveDimension, float] | None = None,
        adaptation_alpha: float = 0.05,
        urgency_threshold: float = 0.3,
    ) -> None:
        self._setpoints: dict[InteroceptiveDimension, float] = dict(
            default_setpoints or DEFAULT_SETPOINTS
        )
        self._target_setpoints: dict[InteroceptiveDimension, float] = dict(self._setpoints)
        self._adaptation_alpha = adaptation_alpha
        self._urgency_threshold = urgency_threshold
        self._current_context: str = "default"

        # Energy burn rate tracking
        self._energy_burn_samples: list[float] = []
        self._max_burn_samples = 20

    @property
    def setpoints(self) -> dict[InteroceptiveDimension, float]:
        return dict(self._setpoints)

    @property
    def urgency_threshold(self) -> float:
        return self._urgency_threshold

    def set_context(self, context: str) -> None:
        """
        Switch allostatic context, updating target setpoints.

        Contexts: "conversation", "deep_processing", "recovery", "exploration", "default"
        Transitions are EMA-smoothed (alpha=0.05) over subsequent ticks to prevent oscillation.
        """
        if context == self._current_context:
            return

        self._current_context = context

        if context in SETPOINT_CONTEXTS:
            # Start from defaults, overlay context overrides
            new_targets = dict(DEFAULT_SETPOINTS)
            for dim, val in SETPOINT_CONTEXTS[context].items():
                new_targets[dim] = val
            self._target_setpoints = new_targets
        else:
            self._target_setpoints = dict(DEFAULT_SETPOINTS)

    def tick_setpoints(self) -> None:
        """
        Smooth setpoints toward targets using EMA.
        Called every theta cycle. Alpha=0.05 means ~20 cycles to converge.
        """
        alpha = self._adaptation_alpha
        for dim in ALL_DIMENSIONS:
            current = self._setpoints.get(dim, 0.0)
            target = self._target_setpoints.get(dim, current)
            self._setpoints[dim] = current + alpha * (target - current)

    def apply_learned_setpoints(
        self,
        learned: dict[InteroceptiveDimension, float],
    ) -> None:
        """Apply setpoints learned by AdaptiveSetpointLearner as new targets.

        These override context targets for dimensions where the organism has
        learned stable operating points in a given attractor basin.
        """
        for dim, val in learned.items():
            self._target_setpoints[dim] = val

    def compute_precision_weights(
        self,
        state: InteroceptiveState,
    ) -> dict[InteroceptiveDimension, float]:
        """
        Compute precision (gain) weights for each dimension.

        Precision = inverse variance = how much the organism should attend
        to each dimension. Higher error → higher precision (pay more attention
        to what's wrong).
        """
        weights: dict[InteroceptiveDimension, float] = {}
        moment_errors = state.errors.get("moment", {})

        for dim in ALL_DIMENSIONS:
            error = abs(moment_errors.get(dim, 0.0))
            rate = abs(state.error_rates.get(dim, 0.0))
            # Precision increases with error magnitude and error rate
            # Base of 1.0 + up to 2.0 from error + up to 1.0 from rate
            precision = 1.0 + min(error * 3.0, 2.0) + min(rate * 5.0, 1.0)
            weights[dim] = precision

        # Normalize so mean = 1.0 (relative weighting)
        total = sum(weights.values())
        if total > 0:
            mean = total / len(weights)
            if mean > 0:
                weights = {d: w / mean for d, w in weights.items()}

        return weights

    def compute_urgency(
        self,
        errors: dict[str, dict[InteroceptiveDimension, float]],
        error_rates: dict[InteroceptiveDimension, float],
    ) -> float:
        """
        Urgency = max(|errors at moment|) * max(|error_rates|).

        A composite need-to-act signal. When > urgency_threshold,
        triggers Nova allostatic deliberation.
        """
        moment_errors = errors.get("moment", {})
        if not moment_errors:
            return 0.0

        max_error = max(abs(e) for e in moment_errors.values()) if moment_errors else 0.0
        max_rate = max(abs(r) for r in error_rates.values()) if error_rates else 0.0

        # Urgency = max(|errors|) * max(|error_rates|), clamped to [0, 1].
        # No floor on max_rate — urgency is zero when errors are stable.
        urgency = max_error * max_rate
        return max(0.0, min(1.0, urgency))

    def find_dominant_error(
        self,
        errors: dict[str, dict[InteroceptiveDimension, float]],
    ) -> tuple[InteroceptiveDimension, float]:
        """Find the dimension with the largest error at moment horizon."""
        moment_errors = errors.get("moment", {})
        if not moment_errors:
            return InteroceptiveDimension.ENERGY, 0.0

        dominant_dim = InteroceptiveDimension.ENERGY
        max_mag = 0.0

        for dim, err in moment_errors.items():
            mag = abs(err)
            if mag > max_mag:
                max_mag = mag
                dominant_dim = dim

        return dominant_dim, max_mag

    def track_energy_burn(self, energy: float) -> tuple[float, float | None]:
        """
        Track energy consumption rate and predict exhaustion.

        Returns: (burn_rate, predicted_exhaustion_seconds_or_None)
        """
        self._energy_burn_samples.append(energy)
        if len(self._energy_burn_samples) > self._max_burn_samples:
            self._energy_burn_samples = self._energy_burn_samples[-self._max_burn_samples:]

        if len(self._energy_burn_samples) < 2:
            return 0.0, None

        # Burn rate = average energy delta per sample (negative = burning)
        deltas = [
            self._energy_burn_samples[i] - self._energy_burn_samples[i - 1]
            for i in range(1, len(self._energy_burn_samples))
        ]
        burn_rate = sum(deltas) / len(deltas)  # Per tick

        if burn_rate >= 0:
            return burn_rate, None  # Not burning, or regenerating

        # At 150ms/tick, burn_rate is per tick
        ticks_per_second = 1.0 / 0.15
        burn_per_second = burn_rate * ticks_per_second

        if burn_per_second >= 0:
            return burn_per_second, None

        # Predict when energy hits critical tier (0.1)
        remaining = energy - 0.1
        if remaining <= 0:
            return burn_per_second, 0.0

        seconds_to_exhaustion = remaining / abs(burn_per_second)
        return burn_per_second, seconds_to_exhaustion

    def build_signal(
        self,
        state: InteroceptiveState,
        phase_snapshot: dict[str, Any],
        cycle_number: int,
    ) -> AllostaticSignal:
        """
        Construct the AllostaticSignal from the interoceptive state and phase-space.

        This is the primary output that other systems consume.
        """
        precision_weights = self.compute_precision_weights(state)
        urgency = self.compute_urgency(state.errors, state.error_rates)
        dominant_dim, dominant_mag = self.find_dominant_error(state.errors)
        dominant_rate = state.error_rates.get(dominant_dim, 0.0)

        # Temporal dissonance
        max_td = 0.0
        td_dim: InteroceptiveDimension | None = None
        for dim, td in state.temporal_dissonance.items():
            if abs(td) > abs(max_td):
                max_td = td
                td_dim = dim

        # Energy tracking
        energy = state.sensed.get(InteroceptiveDimension.ENERGY, 0.5)
        burn_rate, exhaustion_s = self.track_energy_burn(energy)

        return AllostaticSignal(
            state=state,
            urgency=urgency,
            dominant_error=dominant_dim,
            dominant_error_magnitude=dominant_mag,
            dominant_error_rate=dominant_rate,
            precision_weights=precision_weights,
            max_temporal_dissonance=max_td,
            dissonant_dimension=td_dim,
            nearest_attractor=phase_snapshot.get("nearest_attractor"),
            distance_to_bifurcation=phase_snapshot.get("distance_to_nearest_bifurcation"),
            trajectory_heading=phase_snapshot.get("trajectory_heading", "transient"),
            energy_burn_rate=burn_rate,
            predicted_energy_exhaustion_s=exhaustion_s,
            cycle_number=cycle_number,
        )
