"""
EcodiaOS — Simula Reality Map Corpus

Phase 1 foundational data model for observing behavior across four dimensions:
  1. Execution Behaviour — instruction path variance
  2. Protocol State Behaviour — session evolution + state transitions
  3. Identity & Interaction Patterns — trust signal flows
  4. Hardware-Level Variance — timing + resource observability

A RealityMapRecord captures a single proposal execution across all dimensions.
RealityMapCorpus accumulates records for statistical analysis + pattern discovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger().bind(system="simula.reality_map")


# ─── Area 1: Execution Behaviour ──────────────────────────────────────────


class DecisionPointKind(StrEnum):
    """Type of decision point in code agent execution."""
    TOOL_SELECTION = "tool_selection"  # Which tool to call next
    PARAMETER_CHOICE = "parameter_choice"  # How to parameterize a tool
    STRATEGY_SELECTION = "strategy_selection"  # Iterative vs recursive, etc
    EARLY_EXIT = "early_exit"  # When to stop looping
    UNKNOWN = "unknown"


@dataclass
class DecisionPoint:
    """A point where code agent made a choice."""
    kind: DecisionPointKind
    description: str
    alternatives: list[str] = field(default_factory=list)
    chosen: str = ""
    confidence: float = 0.5


@dataclass
class ToolCallRecord:
    """Single tool invocation in code agent agentic loop."""
    tool_name: str
    input_spec: dict[str, Any]
    output_summary: str  # First 200 chars of output
    duration_ms: float
    tokens_used: int
    success: bool
    error_message: str = ""


@dataclass
class ExecutionVariant:
    """Area 1: How did execution differ from alternatives?"""
    tool_call_sequence: list[ToolCallRecord] = field(default_factory=list)
    decision_points: list[DecisionPoint] = field(default_factory=list)
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    code_fingerprint: str = ""  # Semantic signature of generated code
    implementation_variance: float = 0.0  # 0.0 = identical, 1.0 = completely different
    loop_iterations: int = 0
    max_turns_available: int = 32


# ─── Area 2: Protocol State Behaviour ─────────────────────────────────────


class ProposalStatusPhase(StrEnum):
    """Stages in Simula protocol."""
    INITIAL = "initial"
    DEDUPLICATED = "deduplicated"
    VALIDATED = "validated"
    SIMULATED = "simulated"
    GATED = "gated"
    APPLIED = "applied"
    VERIFIED = "verified"
    RECORDED = "recorded"


@dataclass
class StateObservation:
    """What a particular module observes at a state boundary."""
    phase: ProposalStatusPhase
    module: str  # "simulator", "applicator", "verifier", etc.
    observed_state: dict[str, Any]  # What this module sees
    duration_from_previous_ms: float = 0.0


@dataclass
class StateTransitionPath:
    """Area 2: How did proposal evolve through state machine?"""
    observations: list[StateObservation] = field(default_factory=list)
    state_interpretation_divergences: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (phase, module_a, module_b) where they saw different things
    total_phase_duration_ms: float = 0.0
    rollback_occurred: bool = False
    rollback_reason: str = ""


# ─── Area 3: Identity & Interaction Patterns ──────────────────────────────


@dataclass
class TrustAssumption:
    """An implicit trust point in the system."""
    name: str
    location: str  # "code_agent.py:420" or similar
    description: str
    risk_level: float  # 0.0 = safe, 1.0 = critical
    assumption_text: str


@dataclass
class IdentityContext:
    """Who initiated this action?"""
    principal_id: str  # "evo", "external_api", "admin", etc.
    role: str  # Role in the system
    delegation_chain: list[str] = field(default_factory=list)
    credential_strength: float = 1.0  # 0.0 (untrusted) - 1.0 (fully trusted)
    self_applicable: bool = False  # Can skip governance gate
    can_modify_self: bool = False  # Can modify Simula itself


@dataclass
class IdentityBranchingPoint:
    """A decision that depends on identity context."""
    phase: ProposalStatusPhase
    decision: str  # What was decided
    triggered_by_identity: str  # Which identity attribute triggered it
    alternative_if_different_identity: str  # What would happen otherwise


@dataclass
class IdentityAndTrustPattern:
    """Area 3: How does identity affect behavior?"""
    initiator_context: IdentityContext
    branching_points: list[IdentityBranchingPoint] = field(default_factory=list)
    trust_assumptions_relied_upon: list[TrustAssumption] = field(
        default_factory=list
    )
    privilege_gradient: list[tuple[str, float]] = field(
        default_factory=list
    )  # (phase, privilege_level)
    credential_flows: list[tuple[str, str]] = field(
        default_factory=list
    )  # (source, sink) credential paths
    total_governance_gates_passed: int = 0
    gates_skipped_due_to_self_applicable: int = 0


# ─── Area 4: Hardware-Level Variance ──────────────────────────────────────


@dataclass
class TimingRecord:
    """Nanosecond-precision timing for an operation."""
    operation: str
    duration_ns: int
    cpu_time_ns: int = 0
    wall_clock_ms: float = 0.0
    memory_peak_mb: float = 0.0
    cache_misses: int = 0
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkloadClassification:
    """How is this operation consuming resources?"""
    is_io_bound: bool
    is_cpu_bound: bool
    is_memory_bound: bool
    estimated_io_ratio: float = 0.0
    estimated_cpu_ratio: float = 0.0


@dataclass
class InformationLeakageSignature:
    """Does this operation leak information via side-channels?"""
    mutual_information_bits: float  # Bits leaked about secret
    timing_variance_correlated: bool
    resource_usage_correlated: bool
    risk_assessment: str  # "low", "medium", "high", "critical"


@dataclass
class HardwareVarianceProfile:
    """Area 4: Observable hardware behavior."""
    timing_records: list[TimingRecord] = field(default_factory=list)
    workload_classification: dict[str, WorkloadClassification] = field(
        default_factory=dict
    )
    timing_variance_percent: float = 0.0
    leakage_signatures: list[InformationLeakageSignature] = field(
        default_factory=list
    )
    total_duration_ms: float = 0.0


# ─── Unified Reality Map ──────────────────────────────────────────────────


@dataclass
class RealityMapRecord:
    """Single observation across all four dimensions."""
    proposal_id: str
    execution_variant: ExecutionVariant
    protocol_state_path: StateTransitionPath
    identity_and_trust: IdentityAndTrustPattern
    hardware_variance: HardwareVarianceProfile
    timestamp_utc: datetime = field(default_factory=datetime.utcnow)
    success: bool = True
    error_summary: str = ""


class RealityMapCorpus:
    """Accumulating dataset of behavior observations."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self._records: list[RealityMapRecord] = []
        self._storage_dir = storage_dir or Path(".reality_map_corpus")
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._log = logger

    def add_record(self, record: RealityMapRecord) -> None:
        """Store observation."""
        self._records.append(record)
        self._log.info(
            "reality_map_record_added",
            proposal_id=record.proposal_id,
            success=record.success,
            total_records=len(self._records),
        )

    def query_by_proposal_id(self, proposal_id: str) -> list[RealityMapRecord]:
        """Find observations for a specific proposal."""
        return [r for r in self._records if r.proposal_id == proposal_id]

    def query_by_success(self, success: bool) -> list[RealityMapRecord]:
        """Filter by success/failure."""
        return [r for r in self._records if r.success == success]

    def query_by_identity(self, principal_id: str) -> list[RealityMapRecord]:
        """Filter by initiator identity."""
        return [
            r for r in self._records
            if r.identity_and_trust.initiator_context.principal_id == principal_id
        ]

    def get_execution_variance_stats(self) -> dict[str, float]:
        """Statistical analysis of execution variance."""
        if not self._records:
            return {}

        implementation_variances = [
            r.execution_variant.implementation_variance
            for r in self._records
        ]
        tool_call_counts = [
            len(r.execution_variant.tool_call_sequence)
            for r in self._records
        ]

        return {
            "mean_implementation_variance": sum(implementation_variances) / len(implementation_variances),
            "max_implementation_variance": max(implementation_variances),
            "min_implementation_variance": min(implementation_variances),
            "mean_tool_calls": sum(tool_call_counts) / len(tool_call_counts),
            "max_tool_calls": max(tool_call_counts),
        }

    def get_protocol_state_stats(self) -> dict[str, Any]:
        """Statistical analysis of protocol state behavior."""
        if not self._records:
            return {}

        rollback_count = sum(
            1 for r in self._records if r.protocol_state_path.rollback_occurred
        )
        interpretation_divergence_count = sum(
            len(r.protocol_state_path.state_interpretation_divergences)
            for r in self._records
        )

        return {
            "total_records": len(self._records),
            "rollback_count": rollback_count,
            "rollback_rate": rollback_count / len(self._records) if self._records else 0,
            "interpretation_divergence_count": interpretation_divergence_count,
            "mean_phase_duration_ms": sum(
                r.protocol_state_path.total_phase_duration_ms for r in self._records
            ) / len(self._records) if self._records else 0,
        }

    def get_identity_trust_stats(self) -> dict[str, Any]:
        """Statistical analysis of identity and trust patterns."""
        if not self._records:
            return {}

        governance_gates_total = sum(
            r.identity_and_trust.total_governance_gates_passed
            for r in self._records
        )
        gates_skipped_total = sum(
            r.identity_and_trust.gates_skipped_due_to_self_applicable
            for r in self._records
        )

        self_applicable_count = sum(
            1 for r in self._records
            if r.identity_and_trust.initiator_context.self_applicable
        )

        return {
            "total_records": len(self._records),
            "self_applicable_count": self_applicable_count,
            "governance_gates_total": governance_gates_total,
            "gates_skipped_total": gates_skipped_total,
            "skip_rate": gates_skipped_total / (governance_gates_total or 1),
            "unique_principals": len(set(
                r.identity_and_trust.initiator_context.principal_id
                for r in self._records
            )),
        }

    def get_hardware_variance_stats(self) -> dict[str, float]:
        """Statistical analysis of hardware variance."""
        if not self._records:
            return {}

        timing_variances = [
            r.hardware_variance.timing_variance_percent
            for r in self._records
        ]
        durations = [
            r.hardware_variance.total_duration_ms
            for r in self._records
        ]

        return {
            "mean_timing_variance_percent": sum(timing_variances) / len(timing_variances),
            "max_timing_variance_percent": max(timing_variances),
            "mean_duration_ms": sum(durations) / len(durations),
            "max_duration_ms": max(durations),
            "min_duration_ms": min(durations),
        }

    def export_for_analysis(self) -> dict[str, Any]:
        """Export full corpus for external analysis."""
        return {
            "total_records": len(self._records),
            "execution_variance_stats": self.get_execution_variance_stats(),
            "protocol_state_stats": self.get_protocol_state_stats(),
            "identity_trust_stats": self.get_identity_trust_stats(),
            "hardware_variance_stats": self.get_hardware_variance_stats(),
            "records": [self._record_to_dict(r) for r in self._records],
        }

    @staticmethod
    def _record_to_dict(record: RealityMapRecord) -> dict[str, Any]:
        """Convert record to dict for serialization."""
        return {
            "proposal_id": record.proposal_id,
            "timestamp_utc": record.timestamp_utc.isoformat(),
            "success": record.success,
            "execution_variant": {
                "total_duration_ms": record.execution_variant.total_duration_ms,
                "total_tokens": record.execution_variant.total_tokens,
                "tool_call_count": len(record.execution_variant.tool_call_sequence),
                "decision_point_count": len(record.execution_variant.decision_points),
                "implementation_variance": record.execution_variant.implementation_variance,
            },
            "protocol_state_path": {
                "total_phase_duration_ms": record.protocol_state_path.total_phase_duration_ms,
                "rollback_occurred": record.protocol_state_path.rollback_occurred,
                "interpretation_divergences": len(
                    record.protocol_state_path.state_interpretation_divergences
                ),
            },
            "identity_and_trust": {
                "principal_id": record.identity_and_trust.initiator_context.principal_id,
                "self_applicable": record.identity_and_trust.initiator_context.self_applicable,
                "governance_gates_passed": record.identity_and_trust.total_governance_gates_passed,
            },
            "hardware_variance": {
                "total_duration_ms": record.hardware_variance.total_duration_ms,
                "timing_variance_percent": record.hardware_variance.timing_variance_percent,
            },
        }

    def list_all_records(self) -> list[RealityMapRecord]:
        """Return all records."""
        return self._records.copy()
