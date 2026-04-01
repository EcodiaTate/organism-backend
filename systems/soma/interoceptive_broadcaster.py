"""
EcodiaOS - Soma Interoceptive Broadcaster

Mirrors Atune's broadcast architecture for internal percepts. Composes
all Phase A analysis results into InteroceptivePercepts - internal
sensations that enter the Global Workspace alongside external percepts.

Percepts compete for attention normally. A minor internal anomaly during
intense external activity may not reach consciousness. A severe anomaly
will dominate the workspace.

The broadcaster examines DerivativeSnapshots and OrganismStateVectors,
decides whether any thresholds are exceeded, composes a natural-language
description of what the organism feels, and returns an InteroceptivePercept
or None (healthy).

Pure numerical computation + string formatting. No I/O. Budget: <0.5ms.
"""

from __future__ import annotations

import time

import structlog

from systems.soma.types import (
    DerivativeSnapshot,
    InteroceptiveAction,
    InteroceptivePercept,
    OrganismStateVector,
    SensationType,
)

logger = structlog.get_logger("systems.soma.interoceptive_broadcaster")


# ─── Threshold Configuration ────────────────────────────────────────


class BroadcasterThresholds:
    """Adaptive thresholds for generating interoceptive percepts.

    These start at conservative defaults and can be tuned via config
    or evolved by Evo.
    """

    def __init__(
        self,
        velocity_norm_threshold: float = 5.0,
        acceleration_norm_threshold: float = 10.0,
        jerk_norm_threshold: float = 20.0,
        error_rate_threshold: float = 0.3,
        entropy_divergence_threshold: float = 2.0,
        fast_slow_divergence_threshold: float = 3.0,
    ) -> None:
        # Derivative norms above these trigger percepts
        self.velocity_norm = velocity_norm_threshold
        self.acceleration_norm = acceleration_norm_threshold
        self.jerk_norm = jerk_norm_threshold
        # System-level error rate above this triggers percepts
        self.error_rate = error_rate_threshold
        # Entropy divergence from baseline triggers percepts
        self.entropy_divergence = entropy_divergence_threshold
        # Fast vs slow derivative divergence (multi-scale dissonance)
        self.fast_slow_divergence = fast_slow_divergence_threshold


# ─── Sensation Descriptions ─────────────────────────────────────────

_SENSATION_TEMPLATES: dict[str, str] = {
    "velocity_spike": (
        "I feel a rapid change in {system} - {feature} is moving at "
        "{rate:.2f} units/cycle, {scale_label} dynamics are unstable"
    ),
    "acceleration_spike": (
        "I feel accelerating change in {system} - the rate of change itself "
        "is changing rapidly ({rate:.2f}), suggesting an exponential trend"
    ),
    "jerk_spike": (
        "I feel a sudden jolt in {system} - the trajectory has become "
        "rough and unpredictable (jerk norm {rate:.2f})"
    ),
    "error_surge": (
        "I feel pain in {system} - error rate has surged to "
        "{rate:.1%}, something is failing"
    ),
    "fast_slow_divergence": (
        "I feel fragmented - my fast dynamics ({fast_system}) diverge from "
        "my slow dynamics ({slow_system}). Short-term and long-term "
        "trajectories are misaligned"
    ),
    "entropy_anomaly": (
        "I feel {system} behaving unusually - event diversity has shifted "
        "to {entropy:.2f} bits, suggesting a change in operating mode"
    ),
}


# ─── Broadcaster ────────────────────────────────────────────────────


class InteroceptiveBroadcaster:
    """Composes interoceptive percepts from Phase A analysis results.

    Examines DerivativeSnapshots and OrganismStateVectors, and returns
    an InteroceptivePercept when any signal exceeds threshold. Returns
    None when the organism feels healthy.
    """

    def __init__(
        self,
        thresholds: BroadcasterThresholds | None = None,
    ) -> None:
        self._thresholds = thresholds or BroadcasterThresholds()

    def compose_percept(
        self,
        derivatives: DerivativeSnapshot,
        state_vector: OrganismStateVector | None = None,
    ) -> InteroceptivePercept | None:
        """Examine analysis results and compose a percept if warranted.

        Checks in priority order:
          1. Error rate surge (from state vector)
          2. Fast/slow derivative divergence
          3. Jerk spike (roughest trajectory = most urgent)
          4. Acceleration spike
          5. Velocity spike
          6. Entropy anomaly

        Returns the highest-urgency percept, or None if all clear.
        """
        candidates: list[InteroceptivePercept] = []

        # 1. Check error rate surge from state vector
        if state_vector is not None:
            percept = self._check_error_rates(state_vector)
            if percept is not None:
                candidates.append(percept)

        # 2. Check fast/slow divergence
        percept = self._check_fast_slow_divergence(derivatives)
        if percept is not None:
            candidates.append(percept)

        # 3. Check derivative norms at each scale
        for scale in ("fast", "medium", "slow"):
            percept = self._check_derivative_norms(derivatives, scale)
            if percept is not None:
                candidates.append(percept)

        # 4. Check entropy anomalies
        if state_vector is not None:
            percept = self._check_entropy(state_vector)
            if percept is not None:
                candidates.append(percept)

        if not candidates:
            return None

        # Return the highest-urgency percept
        candidates.sort(key=lambda p: p.urgency, reverse=True)
        return candidates[0]

    def _check_derivative_norms(
        self,
        deriv: DerivativeSnapshot,
        scale: str,
    ) -> InteroceptivePercept | None:
        """Check if velocity/acceleration/jerk norms exceed thresholds."""
        vel_norm = deriv.organism_velocity_norm.get(scale, 0.0)
        acc_norm = deriv.organism_acceleration_norm.get(scale, 0.0)
        jrk_norm = deriv.organism_jerk_norm.get(scale, 0.0)

        scale_labels = {
            "fast": "fast (~1s)",
            "medium": "medium (~10s)",
            "slow": "slow (~100s)",
        }
        scale_label = scale_labels.get(scale, scale)

        # Jerk is the most severe (trajectory roughness)
        if jrk_norm > self._thresholds.jerk_norm:
            sys_id = deriv.dominant_system_jerk.get(scale, "unknown")
            urgency = min(1.0, jrk_norm / (self._thresholds.jerk_norm * 3))
            return InteroceptivePercept(
                timestamp=time.monotonic(),
                urgency=urgency,
                sensation_type=SensationType.DERIVATIVE_SPIKE,
                description=_SENSATION_TEMPLATES["jerk_spike"].format(
                    system=sys_id, rate=jrk_norm,
                ),
                epicenter_system=sys_id,
                affected_systems=self._affected_from_deriv(
                    deriv.jerk.get(scale, {}), sys_id,
                ),
                recommended_action=self._action_for_urgency(urgency),
                derivative_snapshot=deriv,
            )

        # Acceleration
        if acc_norm > self._thresholds.acceleration_norm:
            sys_id = deriv.dominant_system_acceleration.get(scale, "unknown")
            urgency = min(1.0, acc_norm / (self._thresholds.acceleration_norm * 3))
            return InteroceptivePercept(
                timestamp=time.monotonic(),
                urgency=urgency,
                sensation_type=SensationType.DERIVATIVE_SPIKE,
                description=_SENSATION_TEMPLATES["acceleration_spike"].format(
                    system=sys_id, rate=acc_norm,
                ),
                epicenter_system=sys_id,
                affected_systems=self._affected_from_deriv(
                    deriv.acceleration.get(scale, {}), sys_id,
                ),
                recommended_action=self._action_for_urgency(urgency),
                derivative_snapshot=deriv,
            )

        # Velocity
        if vel_norm > self._thresholds.velocity_norm:
            sys_id = deriv.dominant_system_velocity.get(scale, "unknown")
            urgency = min(1.0, vel_norm / (self._thresholds.velocity_norm * 3))
            return InteroceptivePercept(
                timestamp=time.monotonic(),
                urgency=urgency,
                sensation_type=SensationType.DERIVATIVE_SPIKE,
                description=_SENSATION_TEMPLATES["velocity_spike"].format(
                    system=sys_id,
                    feature="state trajectory",
                    rate=vel_norm,
                    scale_label=scale_label,
                ),
                epicenter_system=sys_id,
                affected_systems=self._affected_from_deriv(
                    deriv.velocity.get(scale, {}), sys_id,
                ),
                recommended_action=self._action_for_urgency(urgency),
                derivative_snapshot=deriv,
            )

        return None

    def _check_fast_slow_divergence(
        self,
        deriv: DerivativeSnapshot,
    ) -> InteroceptivePercept | None:
        """Detect when fast-scale dynamics diverge from slow-scale dynamics.

        This is the multi-scale dissonance signal - short-term behavior
        doesn't match long-term trends.
        """
        fast_norm = deriv.organism_velocity_norm.get("fast", 0.0)
        slow_norm = deriv.organism_velocity_norm.get("slow", 0.0)

        if slow_norm < 0.001:
            return None

        ratio = fast_norm / slow_norm
        if ratio < self._thresholds.fast_slow_divergence:
            return None

        fast_sys = deriv.dominant_system_velocity.get("fast", "unknown")
        slow_sys = deriv.dominant_system_velocity.get("slow", "unknown")
        urgency = min(1.0, ratio / (self._thresholds.fast_slow_divergence * 3))

        return InteroceptivePercept(
            timestamp=time.monotonic(),
            urgency=urgency,
            sensation_type=SensationType.DERIVATIVE_SPIKE,
            description=_SENSATION_TEMPLATES["fast_slow_divergence"].format(
                fast_system=fast_sys, slow_system=slow_sys,
            ),
            epicenter_system=fast_sys,
            affected_systems=[slow_sys] if slow_sys != fast_sys else [],
            recommended_action=InteroceptiveAction.ATTEND_INWARD,
            derivative_snapshot=deriv,
        )

    def _check_error_rates(
        self,
        sv: OrganismStateVector,
    ) -> InteroceptivePercept | None:
        """Check if any system's error rate exceeds threshold."""
        worst_system = ""
        worst_rate = 0.0

        for sid, slc in sv.systems.items():
            if slc.error_rate > worst_rate:
                worst_rate = slc.error_rate
                worst_system = sid

        if worst_rate <= self._thresholds.error_rate:
            return None

        urgency = min(1.0, worst_rate / 0.8)
        affected = [
            sid
            for sid, slc in sv.systems.items()
            if slc.error_rate > self._thresholds.error_rate * 0.5
            and sid != worst_system
        ]

        return InteroceptivePercept(
            timestamp=time.monotonic(),
            urgency=urgency,
            sensation_type=SensationType.ERROR_RATE_SURGE,
            description=_SENSATION_TEMPLATES["error_surge"].format(
                system=worst_system, rate=worst_rate,
            ),
            epicenter_system=worst_system,
            affected_systems=affected,
            recommended_action=self._action_for_urgency(urgency),
        )

    def _check_entropy(
        self,
        sv: OrganismStateVector,
    ) -> InteroceptivePercept | None:
        """Check for entropy anomalies (unusually high event diversity)."""
        worst_system = ""
        worst_entropy = 0.0

        for sid, slc in sv.systems.items():
            if slc.event_entropy > worst_entropy:
                worst_entropy = slc.event_entropy
                worst_system = sid

        if worst_entropy <= self._thresholds.entropy_divergence:
            return None

        urgency = min(0.5, worst_entropy / (self._thresholds.entropy_divergence * 3))

        return InteroceptivePercept(
            timestamp=time.monotonic(),
            urgency=urgency,
            sensation_type=SensationType.ENTROPY_ANOMALY,
            description=_SENSATION_TEMPLATES["entropy_anomaly"].format(
                system=worst_system, entropy=worst_entropy,
            ),
            epicenter_system=worst_system,
            affected_systems=[],
            recommended_action=InteroceptiveAction.ATTEND_INWARD,
        )

    @staticmethod
    def _affected_from_deriv(
        by_system: dict[str, list[float]],
        epicenter: str,
    ) -> list[str]:
        """Find systems with significant derivative magnitude, excluding epicenter."""
        if not by_system:
            return []

        norms: list[tuple[str, float]] = []
        for sid, vec in by_system.items():
            if sid == epicenter:
                continue
            norm = sum(v * v for v in vec) ** 0.5
            norms.append((sid, norm))

        if not norms:
            return []

        epicenter_vec = by_system.get(epicenter, [])
        epicenter_norm = (
            sum(v * v for v in epicenter_vec) ** 0.5 if epicenter_vec else 1.0
        )
        threshold = epicenter_norm * 0.3

        return [sid for sid, norm in norms if norm > threshold]

    @staticmethod
    def _action_for_urgency(urgency: float) -> InteroceptiveAction:
        """Map urgency to recommended action."""
        if urgency >= 1.0:
            return InteroceptiveAction.CEASE_OPERATION
        if urgency >= 0.9:
            return InteroceptiveAction.EMERGENCY_SAFE_MODE
        if urgency >= 0.7:
            return InteroceptiveAction.TRIGGER_REPAIR
        if urgency >= 0.5:
            return InteroceptiveAction.INHIBIT_GROWTH
        if urgency >= 0.3:
            return InteroceptiveAction.MODULATE_DRIVES
        return InteroceptiveAction.ATTEND_INWARD
