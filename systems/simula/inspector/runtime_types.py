"""
EcodiaOS — Inspector Phase 2: Runtime Behaviour Instrumentation Types

All data models for runtime observation, trace datasets, fault classification,
and the "control integrity score" metric.

Design philosophy
-----------------
Phase 2 operates at the *observation* layer — it records what happens, labels
what went wrong, and produces a numeric steerability signal.  It intentionally
does NOT produce exploit code or attempt to reproduce faults outside a
controlled context.

Layer map
---------
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Target process / test harness                                      │
  │    ↓  sys.settrace / sys.monitoring / ptrace / DWARF unwinder       │
  │  RuntimeTracer   → TraceEvent stream (call / return / bb / branch)  │
  │    ↓                                                                │
  │  FaultClassifier → FaultObservation (signal, exception, label)      │
  │    ↓                                                                │
  │  ControlIntegrityScorer → ControlIntegrityScore per run             │
  │    ↓                                                                │
  │  TraceDataset (normal runs ‖ failure-inducing runs)                  │
  └─────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ── Event / class enums ───────────────────────────────────────────────────────


class TraceEventKind(enum.StrEnum):
    """Granularity level of a single trace event."""

    CALL = "call"           # Function entry
    RETURN = "return"       # Function exit (normal)
    EXCEPTION = "exception" # Function exit (unhandled exception)
    BASIC_BLOCK = "bb"      # Coarse control-flow unit entered
    BRANCH = "branch"       # Conditional branch taken / not-taken
    SIGNAL = "signal"       # OS signal received by the process
    FAULT = "fault"         # Fault detected (memory / type / lifetime)


class FaultClass(enum.StrEnum):
    """
    Fault taxonomy for runtime failures.

    These are *labels* inferred from trace and crash evidence — they describe
    the category of the control-integrity violation, not the mechanism to
    reproduce it.

    OOB         — out-of-bounds read or write (index error, buffer overrun)
    UAF         — use-after-free / dangling reference
    LIFETIME    — object used outside its valid lifetime (not UAF-specific;
                  covers e.g. iterator invalidation, borrowed-ref escape)
    TYPE        — type confusion (object treated as a different type)
    UNHANDLED_EXC — unhandled exception propagated to the top level
    SIGNAL_ABORT  — process terminated by SIGABRT / assertion failure
    SIGNAL_SEGV   — process terminated by SIGSEGV
    SIGNAL_BUS    — process terminated by SIGBUS
    SIGNAL_FPE    — arithmetic exception (div-by-zero, float)
    SIGNAL_OTHER  — any other fatal signal
    LOGIC         — logical invariant violated (detected by assertion)
    UNKNOWN       — insufficient evidence to classify
    """

    OOB = "oob"
    UAF = "uaf_dangling"
    LIFETIME = "lifetime_error"
    TYPE = "type_confusion"
    UNHANDLED_EXC = "unhandled_exception"
    SIGNAL_ABORT = "signal_abort"
    SIGNAL_SEGV = "signal_segv"
    SIGNAL_BUS = "signal_bus"
    SIGNAL_FPE = "signal_fpe"
    SIGNAL_OTHER = "signal_other"
    LOGIC = "logic_invariant"
    UNKNOWN = "unknown"


class RunCategory(enum.StrEnum):
    """Whether this run was normal or failure-inducing."""

    NORMAL = "normal"
    FAILURE = "failure"
    CRASH = "crash"


# ── Trace events ──────────────────────────────────────────────────────────────


class TraceEvent(EOSBaseModel):
    """
    One instrumentation event emitted by the RuntimeTracer.

    Kept intentionally lightweight — a single run may produce thousands of
    events; callers that want aggregated views should use ControlFlowTrace.
    """

    run_id: str = Field(..., description="ID of the run this event belongs to")
    seq: int = Field(..., description="Monotonic sequence number within the run")
    kind: TraceEventKind
    timestamp_ns: int = Field(..., description="Monotonic clock nanoseconds (time.monotonic_ns)")

    # Location
    file: str = Field(default="", description="Source file (relative to workspace root)")
    line: int | None = Field(default=None, description="Source line number")
    func_name: str = Field(default="", description="Enclosing function name")

    # Call / return payload
    caller: str = Field(default="", description="Calling function name (for CALL events)")
    callee: str = Field(default="", description="Called function name (for CALL events)")
    return_value_type: str = Field(
        default="",
        description="Type name of the return value (for RETURN events — type visibility only)",
    )

    # Branch / basic-block payload
    bb_id: str = Field(
        default="",
        description="Opaque basic-block identifier (file:start_line-end_line)",
    )
    branch_taken: bool | None = Field(
        default=None,
        description="True = branch taken, False = not taken, None = N/A",
    )

    # Fault / signal payload
    fault_class: FaultClass | None = Field(
        default=None,
        description="Preliminary fault classification at event time (may be refined later)",
    )
    signal_number: int | None = Field(
        default=None,
        description="POSIX signal number if kind == SIGNAL",
    )
    exception_type: str = Field(
        default="",
        description="Exception class name if kind == EXCEPTION or FAULT",
    )
    exception_message: str = Field(
        default="",
        description="Exception message (truncated to 512 chars)",
    )

    # Stack depth at event time
    stack_depth: int = Field(default=0, description="Call stack depth when event was recorded")


class BasicBlockTrace(EOSBaseModel):
    """
    Aggregated basic-block coverage for a single run.

    Rather than recording every individual BB entry, the tracer accumulates a
    hit-count per BB ID.  This is the coarse control-flow signal suitable for
    "which paths were exercised?"
    """

    run_id: str
    # bb_id → hit count
    hits: dict[str, int] = Field(default_factory=dict)
    # Number of distinct BBs seen
    unique_blocks: int = 0
    # Number of branches observed (taken + not-taken combined)
    branch_observations: int = 0
    # Fraction of branches where the less-common arm was taken (0.0–1.0)
    # High value = diverse branching; low value = single-path execution.
    branch_diversity: float = 0.0


class ControlFlowTrace(EOSBaseModel):
    """
    Complete control-flow summary for a single run.

    Combines the ordered call sequence with the BB coverage.  This is the
    primary artefact handed to the ControlIntegrityScorer.
    """

    run_id: str
    run_category: RunCategory

    # Ordered list of (caller, callee) tuples — function-level call graph
    # observed during this run.  Stored as list-of-pairs for Pydantic compat.
    call_sequence: list[tuple[str, str]] = Field(default_factory=list)

    # Set of function names entered during this run
    functions_visited: list[str] = Field(default_factory=list)

    # Functions exited via exception rather than normal return
    exception_exit_functions: list[str] = Field(default_factory=list)

    # Basic-block coverage
    bb_trace: BasicBlockTrace | None = None

    # Maximum call stack depth observed
    max_stack_depth: int = 0

    # Total events recorded
    total_events: int = 0

    # Wall time of the run in milliseconds
    duration_ms: float = 0.0


# ── Fault observation ─────────────────────────────────────────────────────────


class FaultObservation(EOSBaseModel):
    """
    A classified fault / crash observed during a run.

    The classifier produces one FaultObservation per crash/fault event;
    a single run may have multiple if execution continued after soft faults.
    """

    id: str = Field(default_factory=new_id)
    run_id: str

    fault_class: FaultClass
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the fault classification (0.0–1.0)",
    )

    # Where in the trace the fault occurred
    fault_at_func: str = Field(default="", description="Function where fault was first observed")
    fault_at_file: str = Field(default="")
    fault_at_line: int | None = None

    # Evidence used for classification
    signal_number: int | None = None
    exception_type: str = Field(default="")
    exception_message: str = Field(default="")
    stack_trace: list[str] = Field(default_factory=list)

    # "Where did control become unstructured?" heuristic
    last_structured_func: str = Field(
        default="",
        description=(
            "The last function in the call sequence before control flow became "
            "unstructured (exception propagation, unexpected returns, etc.). "
            "This is the 'transition point' for steeerability analysis."
        ),
    )
    divergence_depth: int = Field(
        default=0,
        description=(
            "Call-stack depth at the transition point. Higher depth = fault "
            "occurred deep inside nested calls (harder to steer externally)."
        ),
    )

    observed_at: datetime = Field(default_factory=utc_now)


# ── Trace datasets ────────────────────────────────────────────────────────────


class TraceDataset(EOSBaseModel):
    """
    A labelled collection of runs for a target / input class.

    Normal and failure-inducing runs are stored together so the
    ControlIntegrityScorer can compare them.
    """

    id: str = Field(default_factory=new_id)
    target_id: str = Field(
        ...,
        description="Identifier for the target (workspace path, module name, etc.)",
    )
    description: str = Field(default="")

    # All traces keyed by run_id
    traces: dict[str, ControlFlowTrace] = Field(default_factory=dict)

    # All fault observations keyed by run_id (may be empty for normal runs)
    faults: dict[str, list[FaultObservation]] = Field(default_factory=dict)

    # Counts
    normal_run_count: int = 0
    failure_run_count: int = 0
    crash_run_count: int = 0

    collected_at: datetime = Field(default_factory=utc_now)

    @property
    def total_runs(self) -> int:
        return self.normal_run_count + self.failure_run_count + self.crash_run_count

    def add_trace(self, trace: ControlFlowTrace) -> None:
        """Register a completed trace into the dataset."""
        self.traces[trace.run_id] = trace
        match trace.run_category:
            case RunCategory.NORMAL:
                self.normal_run_count += 1
            case RunCategory.FAILURE:
                self.failure_run_count += 1
            case RunCategory.CRASH:
                self.crash_run_count += 1

    def add_fault(self, fault: FaultObservation) -> None:
        """Register a fault observation into the dataset."""
        if fault.run_id not in self.faults:
            self.faults[fault.run_id] = []
        self.faults[fault.run_id].append(fault)


# ── Control integrity score ───────────────────────────────────────────────────


class InfluencePermissiveTransition(EOSBaseModel):
    """
    A detected control-flow transition that an external influence could redirect.

    An "influence-permissive" transition is a point where:
    - the call target or branch direction depends on external input (tainted data),
    - OR the transition is anomalous relative to normal runs (new call edge, new BB),
    - OR the transition immediately precedes a fault.

    These are observed phenomena — labelling for the steerability model.
    """

    transition_id: str = Field(default_factory=new_id)
    run_id: str

    # Location
    from_func: str
    to_func: str = Field(default="", description="Callee, or empty for branch events")
    at_file: str = Field(default="")
    at_line: int | None = None

    # Classification
    is_new_edge: bool = Field(
        default=False,
        description="True if this call/branch edge was not seen in any normal run",
    )
    is_pre_fault: bool = Field(
        default=False,
        description="True if a fault was observed within 3 events after this transition",
    )
    taint_influenced: bool = Field(
        default=False,
        description=(
            "True if the tracer detected that tainted data (from the eBPF taint "
            "substrate) was present in the scope when this transition fired."
        ),
    )

    # Magnitude: how far this transition deviates from baseline
    deviation_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Normalised deviation from baseline call-frequency for this edge. "
            "0.0 = common / expected; 1.0 = never seen in normal runs."
        ),
    )


class ControlIntegrityScore(EOSBaseModel):
    """
    A single numeric + structured summary of control-flow integrity for one run.

    Score interpretation
    --------------------
    1.0  — Perfect integrity: execution matched baseline exactly; no anomalous
           transitions; no faults.
    0.0  — Complete loss of integrity: every transition was anomalous; multiple
           faults; deep divergence.
    0.5–0.8 — Partial deviation: some new edges but no crash (interesting for
               steerability — external influence reshaped the call graph without
               breaking execution).

    The score intentionally does NOT measure "security" — it measures how much
    the observed execution deviated from baseline.  The steerability model uses
    this as a continuous training signal.
    """

    id: str = Field(default_factory=new_id)
    run_id: str
    run_category: RunCategory

    # Core scalar
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Control integrity score for this run (1.0 = fully intact).",
    )

    # Components used to derive the scalar
    fraction_new_edges: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of call edges not present in any normal run.",
    )
    fraction_exception_exits: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of function exits that were exception-based.",
    )
    fault_count: int = Field(default=0, description="Number of faults observed in this run.")
    max_divergence_depth: int = Field(
        default=0,
        description="Maximum stack depth at any fault / unstructured transition.",
    )

    # Influence-permissive transitions detected
    permissive_transitions: list[InfluencePermissiveTransition] = Field(
        default_factory=list,
        description="All detected influence-permissive transitions in this run.",
    )
    permissive_transition_count: int = Field(
        default=0,
        description="len(permissive_transitions) — denormalised for fast queries.",
    )

    scored_at: datetime = Field(default_factory=utc_now)


# ── Fault classification report ───────────────────────────────────────────────


class FaultClassificationReport(EOSBaseModel):
    """
    Aggregate fault classification for a TraceDataset.

    Produced by the FaultClassifier after processing all failure/crash runs.
    """

    id: str = Field(default_factory=new_id)
    dataset_id: str

    # Per-class counts
    class_counts: dict[str, int] = Field(
        default_factory=dict,
        description="FaultClass value → number of observations",
    )

    # "Where did control become unstructured?" heuristics
    # Top-N transition points by frequency
    top_transition_points: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Most common last-structured-function values before faults, "
            "with occurrence counts and average divergence depth."
        ),
    )

    # Summary statistics
    total_faults: int = 0
    total_crash_runs: int = 0
    classified_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of faults that received a non-UNKNOWN classification.",
    )

    # Per-run fault labels keyed by run_id → list[FaultClass]
    run_labels: dict[str, list[str]] = Field(default_factory=dict)

    generated_at: datetime = Field(default_factory=utc_now)


# ── Phase 2 result container ──────────────────────────────────────────────────


class Phase2Result(EOSBaseModel):
    """
    Top-level output of a Phase 2 instrumentation session.

    Produced by RuntimeInstrumentationEngine after running both normal and
    failure-inducing inputs through the target.
    """

    id: str = Field(default_factory=new_id)
    target_id: str

    dataset: TraceDataset
    fault_report: FaultClassificationReport
    scores: list[ControlIntegrityScore] = Field(default_factory=list)

    # Aggregate score statistics
    mean_normal_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Mean CIS across all normal runs (baseline)",
    )
    mean_failure_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean CIS across all failure runs",
    )
    mean_crash_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean CIS across all crash runs",
    )

    # Exit criterion check
    influence_permissive_transitions_detected: bool = Field(
        default=False,
        description=(
            "True when at least one influence-permissive transition was reliably "
            "detected and labelled.  This is the Phase 2 exit criterion."
        ),
    )

    produced_at: datetime = Field(default_factory=utc_now)
