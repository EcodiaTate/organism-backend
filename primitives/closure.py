"""
EcodiaOS - Closure Loop Contracts

Declares the feedback loops that make the organism a closed causal system.
Each loop defines a trigger event, a response event, and the systems
involved. Downstream implementation chats wire the actual handlers.

Without closure, the organism is a collection of open-loop controllers.
With closure, it becomes a self-regulating system capable of homeostasis.
"""

from __future__ import annotations

from pydantic import Field

from primitives.common import EOSBaseModel, SystemID

# Import will be available after synapse/types.py is updated with new event types.
# Using string literals for SynapseEventType references to avoid circular imports
# at definition time - the actual enum values are validated at runtime by the
# systems that consume these constants.


class ClosureLoopDefinition(EOSBaseModel):
    """
    Declares a feedback loop between two systems.

    The trigger_event causes the source_system to emit a signal.
    The sink_system subscribes and must emit the response_event
    within timeout_ms. If is_critical is True, failure to close
    the loop triggers a Thymos incident.
    """

    loop_id: str
    name: str
    description: str = ""
    source_system: SystemID
    sink_system: SystemID
    trigger_event: str
    response_event: str
    timeout_ms: int = 30_000
    is_critical: bool = False


# ─── Canonical Closure Loops ────────────────────────────────────────

EQUOR_THYMOS_DRIFT = ClosureLoopDefinition(
    loop_id="equor_thymos_drift",
    name="Constitutional Drift → Immune Response",
    description=(
        "Equor detects constitutional drift (alignment gap widening, "
        "drive weight modification attempts) and emits CONSTITUTIONAL_DRIFT_DETECTED. "
        "Thymos subscribes and initiates a healing pipeline to correct the drift."
    ),
    source_system=SystemID.EQUOR,
    sink_system=SystemID.THYMOS,
    trigger_event="constitutional_drift_detected",
    response_event="repair_completed",
    timeout_ms=60_000,
    is_critical=True,
)

AXON_NOVA_REPAIR = ClosureLoopDefinition(
    loop_id="axon_nova_repair",
    name="Motor Degradation → Replanning",
    description=(
        "Axon detects motor execution degradation (executor failures, "
        "latency spikes, repeated action failures) and emits "
        "MOTOR_DEGRADATION_DETECTED. Nova subscribes and replans the "
        "current goal with alternative action sequences."
    ),
    source_system=SystemID.AXON,
    sink_system=SystemID.NOVA,
    trigger_event="motor_degradation_detected",
    response_event="policy_selected",
    timeout_ms=30_000,
    is_critical=False,
)

SIMULA_STAKES = ClosureLoopDefinition(
    loop_id="simula_stakes",
    name="Rollback → Metabolic Penalty",
    description=(
        "When Simula rolls back a mutation, the rollback must carry genuine "
        "metabolic cost - not just a log entry. SIMULA_ROLLBACK_PENALTY is "
        "emitted to Oikos, which deducts a configurable penalty from the "
        "organism's liquid balance. This makes evolution genuinely risky."
    ),
    source_system=SystemID.SIMULA,
    sink_system=SystemID.OIKOS,
    trigger_event="simula_rollback_penalty",
    response_event="metabolic_pressure",
    timeout_ms=10_000,
    is_critical=True,
)

THYMOS_SIMULA_IMMUNE = ClosureLoopDefinition(
    loop_id="thymos_simula_immune",
    name="Antibody Learning → Mutation Avoidance",
    description=(
        "When Thymos crystallises an antibody from a successful repair, "
        "it emits IMMUNE_PATTERN_ADVISORY. Simula subscribes and adds the "
        "pattern to its negative-example library, preventing future mutations "
        "that would re-introduce known-bad patterns."
    ),
    source_system=SystemID.THYMOS,
    sink_system=SystemID.SIMULA,
    trigger_event="immune_pattern_advisory",
    response_event="simula_calibration_degraded",
    timeout_ms=30_000,
    is_critical=False,
)

SOMA_DOWNSTREAM_MODULATION = ClosureLoopDefinition(
    loop_id="soma_downstream_modulation",
    name="Felt-Sense → Behavioral Modulation",
    description=(
        "Soma's interoceptive felt-sense (arousal, fatigue, metabolic stress) "
        "modulates downstream behaviour. SOMATIC_MODULATION_SIGNAL is emitted "
        "to Nova (policy urgency), Voxis (expression tone), and Equor "
        "(alignment threshold adjustment). This is the organism's gut feeling "
        "influencing its cognition."
    ),
    source_system=SystemID.SOMA,
    sink_system=SystemID.NOVA,
    trigger_event="somatic_modulation_signal",
    response_event="policy_selected",
    timeout_ms=15_000,
    is_critical=False,
)

EVO_BENCHMARKS_FITNESS = ClosureLoopDefinition(
    loop_id="evo_benchmarks_fitness",
    name="Learning Signals → Population Fitness",
    description=(
        "Evo emits learning signals (hypothesis confirmations, mutation outcomes, "
        "capability changes) as FITNESS_OBSERVABLE_BATCH. Benchmarks subscribes "
        "and updates population-level evolutionary activity statistics, feeding "
        "back into Evo's exploration/exploitation balance."
    ),
    source_system=SystemID.EVO,
    sink_system=SystemID.API,  # Benchmarks doesn't have its own SystemID; uses API
    trigger_event="fitness_observable_batch",
    response_event="benchmark_regression",
    timeout_ms=60_000,
    is_critical=False,
)

# All canonical loops for easy iteration
ALL_CLOSURE_LOOPS: list[ClosureLoopDefinition] = [
    EQUOR_THYMOS_DRIFT,
    AXON_NOVA_REPAIR,
    SIMULA_STAKES,
    THYMOS_SIMULA_IMMUNE,
    SOMA_DOWNSTREAM_MODULATION,
    EVO_BENCHMARKS_FITNESS,
]
