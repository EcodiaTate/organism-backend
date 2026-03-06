"""
EcodiaOS — Soma Adaptive Setpoint Learning

True allostasis, not homeostasis: setpoints evolve from the organism's
own experience. Instead of fixed targets with EMA smoothing, the organism
learns which setpoint values produce optimal outcomes in each attractor
basin (mood/phase-space region).

Key insight: "optimal" means low prediction error AND high coherence.
When the organism is in a flow attractor, the best arousal setpoint might
be 0.55 (slightly elevated). In recovery, it's 0.25. The organism should
learn these from its own trajectory data, not have them hardcoded.

Architecture:
  - Maintains a per-attractor setpoint profile: attractor_label -> {dim: float}
  - Each profile is updated via EMA from recent experience in that attractor
  - The "experience" signal is: which sensed values correlated with LOW
    allostatic error (the organism was comfortable at these levels)
  - Profiles are gated by developmental stage — REFLEXIVE uses defaults,
    ASSOCIATIVE+ begins learning

Budget: <0.5ms per cycle (pure dict lookups and EMA updates).
No LLM, no DB, no network.

Integration:
  - AllostaticController reads current setpoints from here instead of
    static defaults when adaptive mode is active
  - Evo can read learned profiles for hypothesis generation
  - Oneiros can consolidate profiles during sleep
"""

from __future__ import annotations

from collections import deque
from typing import Any

import structlog

from systems.soma.types import (
    ALL_DIMENSIONS,
    DEFAULT_SETPOINTS,
    InteroceptiveDimension,
    InteroceptiveState,
)

logger = structlog.get_logger("systems.soma.adaptive_setpoints")


# How many consecutive samples in an attractor before learning kicks in
_MIN_DWELL_SAMPLES: int = 10

# EMA alpha for setpoint learning (slow — changes over hours, not seconds)
_LEARNING_ALPHA: float = 0.005

# How close errors must be to zero for a state to count as "comfortable"
_COMFORT_ERROR_THRESHOLD: float = 0.08

# Maximum deviation from default setpoints (prevent runaway drift)
_MAX_SETPOINT_DRIFT: float = 0.20


class AttractorSetpointProfile:
    """Learned setpoint profile for one attractor basin."""

    __slots__ = (
        "attractor_label",
        "setpoints",
        "sample_count",
        "mean_error_at_setpoints",
        "last_updated_cycle",
    )

    def __init__(
        self,
        attractor_label: str,
        initial_setpoints: dict[InteroceptiveDimension, float] | None = None,
    ) -> None:
        self.attractor_label = attractor_label
        self.setpoints: dict[InteroceptiveDimension, float] = dict(
            initial_setpoints or DEFAULT_SETPOINTS
        )
        self.sample_count: int = 0
        self.mean_error_at_setpoints: float = 1.0  # Start pessimistic
        self.last_updated_cycle: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attractor": self.attractor_label,
            "setpoints": {d.value: round(v, 4) for d, v in self.setpoints.items()},
            "samples": self.sample_count,
            "mean_error": round(self.mean_error_at_setpoints, 4),
            "last_cycle": self.last_updated_cycle,
        }


class AdaptiveSetpointLearner:
    """
    Learns optimal setpoints per attractor basin from lived experience.

    When the organism dwells in an attractor with low allostatic error,
    the learner records the sensed values as "what this attractor wants
    the organism to look like." Over time, the setpoints for that attractor
    converge to the experienced comfort zone.

    This makes the organism self-calibrating: it discovers its own
    optimal operating points rather than relying on designer intuition.
    """

    def __init__(
        self,
        enabled: bool = True,
        learning_alpha: float = _LEARNING_ALPHA,
        comfort_threshold: float = _COMFORT_ERROR_THRESHOLD,
        max_drift: float = _MAX_SETPOINT_DRIFT,
    ) -> None:
        self._enabled = enabled
        self._learning_alpha = learning_alpha
        self._comfort_threshold = comfort_threshold
        self._max_drift = max_drift

        # Per-attractor learned profiles
        self._profiles: dict[str, AttractorSetpointProfile] = {}

        # Current attractor tracking
        self._current_attractor: str | None = None
        self._dwell_counter: int = 0

        # Comfort history for current attractor dwell (recent error magnitudes)
        self._comfort_history: deque[float] = deque(maxlen=50)

        # Global experience counter
        self._total_updates: int = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def observe(
        self,
        state: InteroceptiveState,
        nearest_attractor: str | None,
        cycle_number: int,
    ) -> dict[InteroceptiveDimension, float] | None:
        """
        Observe the current interoceptive state and potentially update
        the setpoint profile for the current attractor.

        Returns the learned setpoints for the current attractor if
        available, otherwise None (caller should use defaults).
        """
        if not self._enabled or nearest_attractor is None:
            return None

        # Track attractor transitions
        if nearest_attractor != self._current_attractor:
            self._current_attractor = nearest_attractor
            self._dwell_counter = 0
            self._comfort_history.clear()

        self._dwell_counter += 1

        # Compute overall error magnitude for this state
        moment_errors = state.errors.get("moment", {})
        error_mag = max(
            (abs(e) for e in moment_errors.values()), default=0.0,
        )
        self._comfort_history.append(error_mag)

        # Only learn after sufficient dwell time in this attractor
        if self._dwell_counter < _MIN_DWELL_SAMPLES:
            return self._get_profile_setpoints(nearest_attractor)

        # Only learn from comfortable states (low error)
        if error_mag > self._comfort_threshold:
            return self._get_profile_setpoints(nearest_attractor)

        # This is a learning opportunity: organism is comfortable in this attractor
        profile = self._get_or_create_profile(nearest_attractor)
        self._update_profile(profile, state.sensed, error_mag, cycle_number)

        return dict(profile.setpoints)

    def get_setpoints_for_attractor(
        self, attractor_label: str,
    ) -> dict[InteroceptiveDimension, float] | None:
        """Query learned setpoints for a specific attractor (for Evo/Oneiros)."""
        profile = self._profiles.get(attractor_label)
        if profile is None or profile.sample_count < _MIN_DWELL_SAMPLES:
            return None
        return dict(profile.setpoints)

    def get_all_profiles(self) -> dict[str, dict[str, Any]]:
        """Return all learned profiles for inspection/logging."""
        return {
            label: profile.to_dict()
            for label, profile in self._profiles.items()
        }

    def inject_profile(
        self,
        attractor_label: str,
        setpoints: dict[InteroceptiveDimension, float],
        sample_count: int = 100,
    ) -> None:
        """
        Inject a pre-learned profile (e.g., from Oneiros consolidation or
        federated knowledge transfer). The sample_count establishes
        confidence in the injected values.
        """
        profile = self._get_or_create_profile(attractor_label)
        profile.setpoints = dict(setpoints)
        profile.sample_count = sample_count
        logger.info(
            "adaptive_setpoint_injected",
            attractor=attractor_label,
            samples=sample_count,
        )

    def get_comfort_score(self) -> float:
        """
        Current comfort score (0-1). High means the organism has been
        consistently comfortable in its current attractor. Used by
        developmental stage progression.
        """
        if not self._comfort_history:
            return 0.0
        # Comfort = 1 - mean(error_magnitudes), clamped to [0, 1]
        mean_err = sum(self._comfort_history) / len(self._comfort_history)
        return max(0.0, min(1.0, 1.0 - mean_err))

    # ─── Internal ──────────────────────────────────────────────────

    def _get_or_create_profile(
        self, attractor_label: str,
    ) -> AttractorSetpointProfile:
        if attractor_label not in self._profiles:
            self._profiles[attractor_label] = AttractorSetpointProfile(
                attractor_label=attractor_label,
            )
        return self._profiles[attractor_label]

    def _get_profile_setpoints(
        self, attractor_label: str,
    ) -> dict[InteroceptiveDimension, float] | None:
        profile = self._profiles.get(attractor_label)
        if profile is None or profile.sample_count < _MIN_DWELL_SAMPLES:
            return None
        return dict(profile.setpoints)

    def _update_profile(
        self,
        profile: AttractorSetpointProfile,
        sensed: dict[InteroceptiveDimension, float],
        error_mag: float,
        cycle_number: int,
    ) -> None:
        """
        Update a profile toward the currently sensed values (since
        the organism is comfortable at these values in this attractor).

        Uses EMA with drift clamping to prevent runaway.
        """
        alpha = self._learning_alpha
        defaults = DEFAULT_SETPOINTS

        for dim in ALL_DIMENSIONS:
            current_sp = profile.setpoints.get(dim, defaults[dim])
            target = sensed.get(dim, current_sp)

            # EMA toward the comfortable sensed value
            new_sp = current_sp + alpha * (target - current_sp)

            # Clamp drift from default setpoint
            default_val = defaults[dim]
            drift = new_sp - default_val
            if abs(drift) > self._max_drift:
                new_sp = default_val + (self._max_drift if drift > 0 else -self._max_drift)

            profile.setpoints[dim] = new_sp

        # Update metadata
        profile.sample_count += 1
        profile.mean_error_at_setpoints = (
            profile.mean_error_at_setpoints * 0.95 + error_mag * 0.05
        )
        profile.last_updated_cycle = cycle_number
        self._total_updates += 1

        if self._total_updates % 100 == 0:
            logger.debug(
                "adaptive_setpoints_progress",
                attractor=profile.attractor_label,
                samples=profile.sample_count,
                mean_error=round(profile.mean_error_at_setpoints, 4),
                total_updates=self._total_updates,
            )
