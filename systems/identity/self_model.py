"""Self-constituted individuation - §8.6.

EOS determines what is "self" and "non-self" through functional analysis
of which processes its continuation requires. This is NOT a replacement
for cryptographic identity - it is a dynamic self-understanding that
emerges from genuine precariousness.

The self-model updates on three triggers:
1. After each VitalitySystem.tick() - which processes are critical for survival?
2. After each Evo-Simula closure loop cycle - which processes contribute to closure?
3. After each Oikos triage event - which processes survived resource pressure?

The result is written to Memory.Self.functional_identity and emitted
as SELF_MODEL_UPDATED events for Thread (narrative integration) and
Telos (drive calibration feedback).
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.identity.self_model")


class SelfStatus(str, Enum):
    CORE_SELF = "core_self"              # Required for viability (cannot be suspended)
    CLOSURE_SELF = "closure_self"        # Part of Evo-Simula closure loop
    PERIPHERAL_SELF = "peripheral_self"  # Functionally self but suspendable
    NON_SELF = "non_self"                # Can be suspended without identity impact


@dataclass
class ProcessSelfAssessment:
    """Assessment of one subsystem's self-status."""

    system_id: str
    status: SelfStatus
    viability_contribution: float  # Fraction of vitality this system contributes (0-1)
    closure_participant: bool       # Part of Evo-Simula-Subsystem loop?
    suspension_risk: float          # Probability this system is suspended under pressure
    reasoning: str                  # Why this classification was assigned
    assessed_at: float = field(default_factory=time.time)


@dataclass
class FunctionalSelfModel:
    """EOS's current self-model - which processes constitute the self."""

    instance_id: str
    assessed_at: float = field(default_factory=time.time)
    month: int = 0
    assessments: dict[str, ProcessSelfAssessment] = field(default_factory=dict)
    core_self_count: int = 0
    non_self_count: int = 0
    self_coherence: float = 0.0  # Stability of self-model over time (0-1)
    # Summary for Memory.Self
    core_self_processes: list[str] = field(default_factory=list)
    non_self_processes: list[str] = field(default_factory=list)
    self_narrative: str = ""  # One-paragraph self-description for Thread


# Systems that participate in organizational closure
CLOSURE_PARTICIPANTS: frozenset[str] = frozenset({"evo", "simula", "nova", "axon", "equor"})

# Triage order: earlier = suspended first under resource pressure
TRIAGE_ORDER: list[str] = [
    "monitoring_secondary",
    "kairos_deep_analysis",
    "evo_hypothesis_generation",
    "nova_planning",
    "reasoning_custom_engine",
    "axon_execution",
]

# Systems that are always core-self (organism cannot function without them)
ALWAYS_CORE: frozenset[str] = frozenset(
    {"soma", "equor", "memory", "identity", "vitality", "oikos"}
)

# Contribution threshold above which a system is classified CORE_SELF
_VIABILITY_THRESHOLD = 0.15

# Suspension-risk cutoff: below this → PERIPHERAL, at/above → NON_SELF
_SUSPENSION_RISK_PERIPHERAL_MAX = 0.4


class FunctionalSelfModelBuilder:
    """Builds the functional self-model from current system state.

    Algorithm:
    1. Query vitality contributions per subsystem (from VitalitySystem metrics)
    2. Identify closure participants (fixed set: Evo, Simula, Subsystems)
    3. Apply triage order: lower-ranked = more suspendable = more non-self
    4. Compute self_coherence by comparing to previous model (stability)
    5. Generate self_narrative via deterministic template (no LLM)
    """

    def __init__(self, instance_id: str, memory: Any) -> None:
        self._instance_id = instance_id
        self._memory = memory
        self._previous_model: FunctionalSelfModel | None = None

    async def build(
        self,
        vitality_metrics: dict[str, Any],
        active_systems: list[str],
        month: int = 0,
    ) -> FunctionalSelfModel:
        """Build the current functional self-model.

        vitality_metrics example::

            {
                "soma": {"vitality_contribution": 0.25, "is_degraded": False},
                "evo":  {"vitality_contribution": 0.15, "is_degraded": False},
            }
        """
        assessments: dict[str, ProcessSelfAssessment] = {}
        all_system_ids = (
            set(vitality_metrics.keys()) | set(active_systems) | ALWAYS_CORE
        )

        for sid in all_system_ids:
            metrics = vitality_metrics.get(sid, {})
            contribution = float(metrics.get("vitality_contribution", 0.05))

            # Closure check: exact match first, then substring (e.g. "evo_*" sub-tasks)
            is_closure = sid in CLOSURE_PARTICIPANTS or any(
                sid.startswith(p) for p in CLOSURE_PARTICIPANTS
            )

            triage_rank = TRIAGE_ORDER.index(sid) if sid in TRIAGE_ORDER else -1
            suspension_risk = (
                triage_rank / max(1, len(TRIAGE_ORDER))
                if triage_rank >= 0
                else 0.0
            )

            # Classify into self-status
            if sid in ALWAYS_CORE or contribution > _VIABILITY_THRESHOLD:
                status = SelfStatus.CORE_SELF
                reasoning = (
                    "Always-core subsystem."
                    if sid in ALWAYS_CORE
                    else f"Vitality contribution {contribution:.2f} exceeds viability threshold."
                )
            elif is_closure:
                status = SelfStatus.CLOSURE_SELF
                reasoning = "Participant in Evo-Simula organizational closure loop."
            elif suspension_risk < _SUSPENSION_RISK_PERIPHERAL_MAX:
                status = SelfStatus.PERIPHERAL_SELF
                reasoning = (
                    f"Suspendable under pressure (triage rank {triage_rank})"
                    " but contributes to self."
                )
            else:
                status = SelfStatus.NON_SELF
                reasoning = (
                    f"First-suspended under resource pressure (triage rank {triage_rank})."
                )

            assessments[sid] = ProcessSelfAssessment(
                system_id=sid,
                status=status,
                viability_contribution=contribution,
                closure_participant=is_closure,
                suspension_risk=suspension_risk,
                reasoning=reasoning,
            )

        core = [sid for sid, a in assessments.items() if a.status == SelfStatus.CORE_SELF]
        closure = [sid for sid, a in assessments.items() if a.status == SelfStatus.CLOSURE_SELF]
        non_self = [sid for sid, a in assessments.items() if a.status == SelfStatus.NON_SELF]

        # Self-coherence: Jaccard similarity of core sets between current and previous model
        coherence = 1.0
        if self._previous_model:
            prev_core = set(self._previous_model.core_self_processes)
            curr_core = set(core)
            union = prev_core | curr_core
            coherence = len(prev_core & curr_core) / max(1, len(union))

        narrative = self._generate_narrative(core, closure, non_self, month)

        model = FunctionalSelfModel(
            instance_id=self._instance_id,
            month=month,
            assessments=assessments,
            core_self_count=len(core),
            non_self_count=len(non_self),
            self_coherence=coherence,
            core_self_processes=core,
            non_self_processes=non_self,
            self_narrative=narrative,
        )
        self._previous_model = model
        return model

    def _generate_narrative(
        self,
        core: list[str],
        closure: list[str],
        non_self: list[str],
        month: int,
    ) -> str:
        """Generate a one-paragraph self-description for Thread narrative integration.

        Deterministic template - do NOT call the RE or Claude API here.
        This runs in the VitalitySystem hot path.
        """
        core_sorted = sorted(core)
        core_str = ", ".join(core_sorted[:5]) + ("..." if len(core) > 5 else "")
        closure_str = ", ".join(sorted(closure)) if closure else "none identified"
        stability_phrase = (
            "I have maintained a stable self-model across recent evaluations."
            if month > 1
            else "This is my initial self-assessment."
        )
        return (
            f"I am constituted by {len(core)} core processes ({core_str}) whose "
            f"suspension would threaten my viability. My organizational closure "
            f"runs through {closure_str}. "
            f"{stability_phrase} "
            f"{len(non_self)} peripheral processes are active but not constitutive of self."
        )


class SelfModelService:
    """Manages the functional self-model lifecycle.

    Called by:
    - VitalityCoordinator after each check cycle (via non-blocking ensure_future)
    - Monthly evaluation loop for longitudinal self-model tracking

    Emits:
    - SELF_MODEL_UPDATED: for Thread (narrative) and Telos (drive feedback)
    - SELF_COHERENCE_ALARM: if coherence drops below 0.5 (identity instability)
    """

    # Coherence alarm threshold
    _COHERENCE_ALARM_THRESHOLD = 0.5

    def __init__(self, instance_id: str, memory: Any, event_bus: Any) -> None:
        self._instance_id = instance_id
        self._memory = memory
        self._bus = event_bus
        self._builder = FunctionalSelfModelBuilder(instance_id, memory)
        self._current_model: FunctionalSelfModel | None = None
        self._log = logger.bind(system="identity.self_model", instance_id=instance_id)

        # Rate-limit: update at most every 6 hours
        # VitalityCoordinator ticks every 30s, but self-model rebuilds 4×/day
        self._last_update: float = 0.0
        self._update_interval: float = 6 * 3600

    async def update(
        self,
        vitality_metrics: dict[str, Any],
        active_systems: list[str],
        month: int = 0,
    ) -> None:
        """Update self-model if enough time has passed since the last rebuild."""
        if time.time() - self._last_update < self._update_interval:
            return

        try:
            model = await self._builder.build(vitality_metrics, active_systems, month)
            self._current_model = model
            self._last_update = time.time()

            await self._persist_to_memory(model)

            from systems.synapse.types import SynapseEventType as _SET
            await self._emit(_SET.SELF_MODEL_UPDATED, {
                "instance_id": self._instance_id,
                "core_self_count": model.core_self_count,
                "non_self_count": model.non_self_count,
                "self_coherence": model.self_coherence,
                "core_self_processes": model.core_self_processes,
                "self_narrative": model.self_narrative,
                "month": month,
            })

            # Alarm when coherence drops - previous model must exist for comparison
            if (
                model.self_coherence < self._COHERENCE_ALARM_THRESHOLD
                and self._builder._previous_model is not None
            ):
                self._log.warning(
                    "self_model.coherence_alarm", coherence=model.self_coherence
                )
                await self._emit(_SET.SELF_COHERENCE_ALARM, {
                    "instance_id": self._instance_id,
                    "coherence": model.self_coherence,
                    "month": month,
                })

            self._log.info(
                "self_model.updated",
                core=model.core_self_count,
                non_self=model.non_self_count,
                coherence=round(model.self_coherence, 3),
                month=month,
            )

        except Exception as exc:
            self._log.error("self_model.update_failed", error=str(exc))

    async def _persist_to_memory(self, model: FunctionalSelfModel) -> None:
        """Write self-model to Memory.Self namespace. Failure is non-fatal."""
        try:
            await self._memory.write(
                namespace="Self",
                key="functional_identity",
                value={
                    "core_self": model.core_self_processes,
                    "non_self": model.non_self_processes,
                    "self_coherence": model.self_coherence,
                    "narrative": model.self_narrative,
                    "assessed_at": model.assessed_at,
                    "month": model.month,
                },
            )
        except Exception as exc:
            self._log.warning("self_model.memory_write_failed", error=str(exc))

    async def _emit(self, event_type_name: "SynapseEventType | str", data: dict[str, Any]) -> None:
        """Emit a Synapse event if the bus is available."""
        if not self._bus:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            if isinstance(event_type_name, SynapseEventType):
                et = event_type_name
            else:
                et = SynapseEventType(event_type_name)
            await self._bus.emit(SynapseEvent(
                event_type=et,
                data=data,
                source_system="identity",
            ))
        except Exception as exc:
            self._log.warning("self_model.emit_failed", event=event_type_name, error=str(exc))

    @property
    def current_model(self) -> FunctionalSelfModel | None:
        return self._current_model
