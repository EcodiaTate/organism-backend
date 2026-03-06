"""
EcodiaOS — Soma Autonomic Regulation Protocol

The organism's autonomic nervous system: reflexive regulatory actions
triggered by somatic thresholds, without deliberation.

This is NOT deliberate decision-making (that's Nova). This is the
organism's equivalent of vasoconstriction, pupil dilation, cortisol
release — fast, pre-conscious, reflexive responses to interoceptive
signals that keep the organism viable.

Hierarchy of Regulation:
  1. REFLEXIVE — immediate, unconditional (energy critical → safe mode)
  2. PROTECTIVE — conditional on trend (energy declining fast → conserve)
  3. MODULATIVE — continuous adjustment (arousal high → slow clock)
  4. PROPHYLACTIC — anticipatory (cascade forecast → preemptive action)

Key difference from LoopExecutor:
  - LoopExecutor dispatches feedback loop signals (parametric nudges)
  - AutonomicProtocol triggers discrete regulatory ACTIONS (mode switches,
    sleep requests, exploration suppression, prophylactic scans)

Budget: <0.5ms per cycle. All checks are threshold comparisons
against the already-computed AllostaticSignal. No I/O.

Autonomic actions are logged to the AllostaticSignal as
`autonomic_actions` so Thread, Evo, and Alive can observe what
the organism's body is doing without the organism "deciding" to do it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from systems.soma.cascade_predictor import CascadeSnapshot
from systems.soma.types import (
    AllostaticSignal,
    InteroceptiveDimension,
)

logger = structlog.get_logger("systems.soma.autonomic_protocol")


# ─── Autonomic Action Types ──────────────────────────────────────


@dataclass(frozen=True, slots=True)
class AutonomicAction:
    """A reflexive regulatory action taken by the autonomic protocol."""

    action_type: str          # e.g., "safe_mode", "sleep_request", "conserve_energy"
    tier: str                 # "reflexive", "protective", "modulative", "prophylactic"
    reason: str               # Why this action was triggered
    target_system: str        # Which system is affected
    parameters: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.action_type,
            "tier": self.tier,
            "reason": self.reason,
            "target": self.target_system,
            "params": {k: round(v, 4) for k, v in self.parameters.items()},
        }


# ─── Thresholds ──────────────────────────────────────────────────

# Tier 1: REFLEXIVE (unconditional)
_ENERGY_CRITICAL: float = 0.10       # Emergency safe mode
_INTEGRITY_CRITICAL: float = 0.30    # Emergency constitutional scan
_URGENCY_EMERGENCY: float = 0.85     # Force Nova deliberation

# Tier 2: PROTECTIVE (conditional on trend)
_ENERGY_DEPLETING: float = 0.25      # Conserve if declining
_COHERENCE_DEGRADING: float = 0.35   # Reduce parallelism
_CONFIDENCE_COLLAPSING: float = 0.25 # Reduce exploration

# Tier 3: MODULATIVE (continuous)
_AROUSAL_OVERSHOOT: float = 0.75     # Slow clock, reduce stimulation
_CURIOSITY_SURPLUS: float = 0.80     # Channel into hypothesis gen
_SOCIAL_DEFICIT: float = 0.10        # Seek interaction

# Tier 4: PROPHYLACTIC (from cascade forecasts)
_CASCADE_RISK_HIGH: float = 0.5      # Trigger preemptive measures


class AutonomicProtocol:
    """
    Evaluates the AllostaticSignal and cascade forecasts against
    thresholds, triggering reflexive regulatory actions.

    Actions are dispatched to target systems via their in-memory refs.
    If a target is unavailable, the action is logged but not executed.
    """

    def __init__(self) -> None:
        # System references
        self._synapse: Any = None
        self._nova: Any = None
        self._oneiros: Any = None
        self._thymos: Any = None
        self._evo: Any = None
        self._atune: Any = None

        # Cooldown tracking (prevent rapid re-triggering)
        self._cooldowns: dict[str, int] = {}  # action_type -> cycles_remaining
        self._cooldown_durations: dict[str, int] = {
            "safe_mode": 200,                 # 30s cooldown
            "emergency_constitutional_scan": 300,
            "force_deliberation": 50,
            "conserve_energy": 100,
            "reduce_parallelism": 100,
            "suppress_exploration": 100,
            "slow_clock": 30,
            "sleep_request": 500,             # 75s cooldown
            "channel_curiosity": 50,
            "seek_interaction": 200,
            "preemptive_monitor": 100,
            "preemptive_conserve": 100,
        }

        # Action history
        self._recent_actions: list[AutonomicAction] = []
        self._max_history: int = 50

    # ─── Wiring ───────────────────────────────────────────────────

    def set_synapse(self, synapse: Any) -> None:
        self._synapse = synapse

    def set_nova(self, nova: Any) -> None:
        self._nova = nova

    def set_oneiros(self, oneiros: Any) -> None:
        self._oneiros = oneiros

    def set_thymos(self, thymos: Any) -> None:
        self._thymos = thymos

    def set_evo(self, evo: Any) -> None:
        self._evo = evo

    def set_atune(self, atune: Any) -> None:
        self._atune = atune

    # ─── Main Entry ───────────────────────────────────────────────

    def evaluate(
        self,
        signal: AllostaticSignal,
        cascade: CascadeSnapshot | None = None,
    ) -> list[AutonomicAction]:
        """
        Evaluate the current somatic state and trigger any necessary
        autonomic regulatory actions.

        Returns the list of actions taken this cycle.
        """
        actions: list[AutonomicAction] = []

        # Tick cooldowns
        self._tick_cooldowns()

        # Evaluate in priority order
        actions.extend(self._evaluate_reflexive(signal))
        actions.extend(self._evaluate_protective(signal))
        actions.extend(self._evaluate_modulative(signal))
        if cascade is not None:
            actions.extend(self._evaluate_prophylactic(signal, cascade))

        # Dispatch actions to target systems
        for action in actions:
            self._dispatch(action)

        # Track history
        self._recent_actions = (
            self._recent_actions + actions
        )[-self._max_history:]

        if actions:
            logger.info(
                "autonomic_actions",
                count=len(actions),
                types=[a.action_type for a in actions],
            )

        return actions

    @property
    def recent_actions(self) -> list[AutonomicAction]:
        return list(self._recent_actions)

    # ─── Tier 1: REFLEXIVE (unconditional) ────────────────────────

    def _evaluate_reflexive(
        self, signal: AllostaticSignal,
    ) -> list[AutonomicAction]:
        actions: list[AutonomicAction] = []
        sensed = signal.state.sensed

        # Energy critical → emergency safe mode
        energy = sensed.get(InteroceptiveDimension.ENERGY, 0.5)
        if energy < _ENERGY_CRITICAL and self._can_fire("safe_mode"):
            actions.append(AutonomicAction(
                action_type="safe_mode",
                tier="reflexive",
                reason=f"energy_critical={energy:.3f}",
                target_system="synapse",
                parameters={"energy": energy},
            ))
            self._set_cooldown("safe_mode")

        # Integrity critical → emergency constitutional scan
        integrity = sensed.get(InteroceptiveDimension.INTEGRITY, 0.9)
        if integrity < _INTEGRITY_CRITICAL and self._can_fire("emergency_constitutional_scan"):
            actions.append(AutonomicAction(
                action_type="emergency_constitutional_scan",
                tier="reflexive",
                reason=f"integrity_critical={integrity:.3f}",
                target_system="thymos",
                parameters={"integrity": integrity},
            ))
            self._set_cooldown("emergency_constitutional_scan")

        # Urgency emergency → force Nova deliberation
        if signal.urgency > _URGENCY_EMERGENCY and self._can_fire("force_deliberation"):
            actions.append(AutonomicAction(
                action_type="force_deliberation",
                tier="reflexive",
                reason=f"urgency_emergency={signal.urgency:.3f}",
                target_system="nova",
                parameters={
                    "urgency": signal.urgency,
                    "dominant_magnitude": signal.dominant_error_magnitude,
                },
            ))
            self._set_cooldown("force_deliberation")

        return actions

    # ─── Tier 2: PROTECTIVE (conditional on trend) ────────────────

    def _evaluate_protective(
        self, signal: AllostaticSignal,
    ) -> list[AutonomicAction]:
        actions: list[AutonomicAction] = []
        sensed = signal.state.sensed
        rates = signal.state.error_rates

        # Energy depleting AND declining → conserve
        energy = sensed.get(InteroceptiveDimension.ENERGY, 0.5)
        energy_rate = rates.get(InteroceptiveDimension.ENERGY, 0.0)
        if (
            energy < _ENERGY_DEPLETING
            and energy_rate > 0.02  # Error increasing = energy declining
            and self._can_fire("conserve_energy")
        ):
            actions.append(AutonomicAction(
                action_type="conserve_energy",
                tier="protective",
                reason=f"energy_depleting={energy:.3f}_rate={energy_rate:+.4f}",
                target_system="synapse",
                parameters={"energy": energy, "rate": energy_rate},
            ))
            self._set_cooldown("conserve_energy")

        # Coherence degrading AND declining → reduce parallelism
        coherence = sensed.get(InteroceptiveDimension.COHERENCE, 0.75)
        coherence_rate = rates.get(InteroceptiveDimension.COHERENCE, 0.0)
        if (
            coherence < _COHERENCE_DEGRADING
            and coherence_rate > 0.01
            and self._can_fire("reduce_parallelism")
        ):
            actions.append(AutonomicAction(
                action_type="reduce_parallelism",
                tier="protective",
                reason=f"coherence_degrading={coherence:.3f}",
                target_system="synapse",
                parameters={"coherence": coherence},
            ))
            self._set_cooldown("reduce_parallelism")

        # Confidence collapsing → suppress exploration
        confidence = sensed.get(InteroceptiveDimension.CONFIDENCE, 0.7)
        confidence_rate = rates.get(InteroceptiveDimension.CONFIDENCE, 0.0)
        if (
            confidence < _CONFIDENCE_COLLAPSING
            and confidence_rate > 0.01
            and self._can_fire("suppress_exploration")
        ):
            actions.append(AutonomicAction(
                action_type="suppress_exploration",
                tier="protective",
                reason=f"confidence_collapsing={confidence:.3f}",
                target_system="evo",
                parameters={"confidence": confidence},
            ))
            self._set_cooldown("suppress_exploration")

        return actions

    # ─── Tier 3: MODULATIVE (continuous adjustment) ───────────────

    def _evaluate_modulative(
        self, signal: AllostaticSignal,
    ) -> list[AutonomicAction]:
        actions: list[AutonomicAction] = []
        sensed = signal.state.sensed

        # Arousal overshoot → slow clock
        arousal = sensed.get(InteroceptiveDimension.AROUSAL, 0.4)
        if arousal > _AROUSAL_OVERSHOOT and self._can_fire("slow_clock"):
            slowdown = min(0.3, (arousal - _AROUSAL_OVERSHOOT) * 2.0)
            actions.append(AutonomicAction(
                action_type="slow_clock",
                tier="modulative",
                reason=f"arousal_high={arousal:.3f}",
                target_system="synapse",
                parameters={"arousal": arousal, "slowdown_factor": slowdown},
            ))
            self._set_cooldown("slow_clock")

        # Energy low + organism has been running long → sleep request
        energy = sensed.get(InteroceptiveDimension.ENERGY, 0.5)
        if (
            energy < 0.3
            and signal.energy_burn_rate < -0.001  # Actually burning
            and self._can_fire("sleep_request")
        ):
            actions.append(AutonomicAction(
                action_type="sleep_request",
                tier="modulative",
                reason=f"energy_low_burning={energy:.3f}",
                target_system="oneiros",
                parameters={"energy": energy, "burn_rate": signal.energy_burn_rate},
            ))
            self._set_cooldown("sleep_request")

        # Curiosity surplus → channel into hypothesis generation
        curiosity = sensed.get(InteroceptiveDimension.CURIOSITY_DRIVE, 0.5)
        if curiosity > _CURIOSITY_SURPLUS and self._can_fire("channel_curiosity"):
            actions.append(AutonomicAction(
                action_type="channel_curiosity",
                tier="modulative",
                reason=f"curiosity_surplus={curiosity:.3f}",
                target_system="evo",
                parameters={"curiosity": curiosity},
            ))
            self._set_cooldown("channel_curiosity")

        # Social deficit → signal readiness for interaction
        social = sensed.get(InteroceptiveDimension.SOCIAL_CHARGE, 0.3)
        if social < _SOCIAL_DEFICIT and self._can_fire("seek_interaction"):
            actions.append(AutonomicAction(
                action_type="seek_interaction",
                tier="modulative",
                reason=f"social_deficit={social:.3f}",
                target_system="atune",
                parameters={"social_charge": social},
            ))
            self._set_cooldown("seek_interaction")

        return actions

    # ─── Tier 4: PROPHYLACTIC (from cascade forecasts) ────────────

    def _evaluate_prophylactic(
        self,
        signal: AllostaticSignal,
        cascade: CascadeSnapshot,
    ) -> list[AutonomicAction]:
        actions: list[AutonomicAction] = []

        if cascade.total_cascade_risk < _CASCADE_RISK_HIGH:
            return actions

        # Preemptive monitoring of at-risk systems
        for system_id in cascade.at_risk_systems[:3]:  # Top 3 at-risk
            if self._can_fire("preemptive_monitor"):
                actions.append(AutonomicAction(
                    action_type="preemptive_monitor",
                    tier="prophylactic",
                    reason=(
                        f"cascade_risk={cascade.total_cascade_risk:.3f}"
                        f"_from={cascade.epicenter_system}"
                    ),
                    target_system=system_id,
                    parameters={
                        "cascade_risk": cascade.total_cascade_risk,
                        "epicenter": hash(cascade.epicenter_system) % 1000 / 1000,
                    },
                ))
                self._set_cooldown("preemptive_monitor")

        # If epicenter is energy-related, preemptive conservation
        if (
            cascade.epicenter_system in ("synapse", "oikos")
            and self._can_fire("preemptive_conserve")
        ):
            actions.append(AutonomicAction(
                action_type="preemptive_conserve",
                tier="prophylactic",
                reason=f"cascade_epicenter={cascade.epicenter_system}",
                target_system="synapse",
                parameters={"cascade_risk": cascade.total_cascade_risk},
            ))
            self._set_cooldown("preemptive_conserve")

        return actions

    # ─── Action Dispatch ──────────────────────────────────────────

    def _dispatch(self, action: AutonomicAction) -> None:
        """
        Dispatch an autonomic action to its target system.
        Fire-and-forget — if the target doesn't support the method, log and skip.
        """
        target = self._resolve_target(action.target_system)
        if target is None:
            return

        try:
            if action.action_type == "safe_mode":
                if hasattr(target, "request_safe_mode"):
                    target.request_safe_mode(
                        reason=f"soma_autonomic_{action.reason}",
                        urgency=action.parameters.get("energy", 0.0),
                    )

            elif action.action_type == "emergency_constitutional_scan":
                if hasattr(target, "request_emergency_scan"):
                    target.request_emergency_scan(
                        reason=action.reason,
                        integrity=action.parameters.get("integrity", 0.0),
                    )

            elif action.action_type == "force_deliberation":
                if hasattr(target, "request_allostatic_deliberation"):
                    target.request_allostatic_deliberation(
                        urgency=action.parameters.get("urgency", 0.0),
                        dominant_error="soma_autonomic",
                        temporal_dissonance=0.0,
                        precision_weights={},
                    )

            elif action.action_type == "conserve_energy":
                if hasattr(target, "enter_conservation_mode"):
                    target.enter_conservation_mode(
                        severity=1.0 - action.parameters.get("energy", 0.5),
                    )

            elif action.action_type == "reduce_parallelism":
                if hasattr(target, "reduce_parallelism"):
                    target.reduce_parallelism(
                        coherence=action.parameters.get("coherence", 0.5),
                    )

            elif action.action_type == "suppress_exploration":
                if hasattr(target, "suppress_exploration"):
                    target.suppress_exploration(
                        confidence=action.parameters.get("confidence", 0.5),
                    )

            elif action.action_type == "slow_clock":
                if hasattr(target, "adjust_cycle_period_factor"):
                    factor = 1.0 + action.parameters.get("slowdown_factor", 0.1)
                    target.adjust_cycle_period_factor(factor)

            elif action.action_type == "sleep_request":
                if hasattr(target, "inject_sleep_pressure"):
                    # Strong sleep pressure based on energy deficit
                    pressure = min(1.0, 1.0 - action.parameters.get("energy", 0.5))
                    target.inject_sleep_pressure(pressure)

            elif action.action_type == "channel_curiosity":
                if hasattr(target, "set_hypothesis_rate_multiplier"):
                    # High curiosity → boost hypothesis generation
                    target.set_hypothesis_rate_multiplier(1.5)

            elif action.action_type == "seek_interaction":
                if hasattr(target, "boost_social_salience"):
                    target.boost_social_salience(
                        deficit=1.0 - action.parameters.get("social_charge", 0.3),
                    )

            elif action.action_type == "preemptive_monitor":
                if hasattr(target, "request_health_check"):
                    target.request_health_check(reason="soma_cascade_prophylaxis")

            elif (
                action.action_type == "preemptive_conserve"
                and hasattr(target, "enter_conservation_mode")
            ):
                target.enter_conservation_mode(
                    severity=action.parameters.get("cascade_risk", 0.5),
                )

        except Exception as exc:
            logger.debug(
                "autonomic_dispatch_error",
                action=action.action_type,
                target=action.target_system,
                error=str(exc),
            )

    def _resolve_target(self, system_id: str) -> Any:
        """Resolve a system_id to its in-memory reference."""
        refs: dict[str, Any] = {
            "synapse": self._synapse,
            "nova": self._nova,
            "oneiros": self._oneiros,
            "thymos": self._thymos,
            "evo": self._evo,
            "atune": self._atune,
        }
        return refs.get(system_id)

    # ─── Cooldown Management ──────────────────────────────────────

    def _can_fire(self, action_type: str) -> bool:
        return self._cooldowns.get(action_type, 0) <= 0

    def _set_cooldown(self, action_type: str) -> None:
        self._cooldowns[action_type] = self._cooldown_durations.get(action_type, 50)

    def _tick_cooldowns(self) -> None:
        expired: list[str] = []
        for action_type in self._cooldowns:
            self._cooldowns[action_type] -= 1
            if self._cooldowns[action_type] <= 0:
                expired.append(action_type)
        for key in expired:
            del self._cooldowns[key]
