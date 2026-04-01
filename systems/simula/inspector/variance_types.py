"""
EcodiaOS - Inspector Phase 7: Physical Execution Variance Types

All domain models for side-channel-style execution distinguishability analysis.

Design philosophy
-----------------
Phase 7 answers the question: "does execution behaviour measurably vary with
sensitive internal values or decisions?"

It treats the system as a black box that emits observable timing signals.  A
timing *channel* exists when the distribution of execution latencies for
operation A (processing a "sensitive" value X) differs significantly from
operation B (processing a "neutral" value Y).  If an observer can classify
which code path was taken from timing alone, the channel is *distinguishable*.

Phase 7 does not require kernel-level access or hardware counters.  It models
the full spectrum from coarse wall-clock observations to fine-grained
microarchitectural profiles, and applies a hierarchy of statistical tests to
determine distinguishability under realistic noise conditions.

Layer map
---------
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Phase 6 (Phase6Result - protocol boundary failures + FSM coverage)      │
  │  Phase 5 (Phase5Result - trust corridors + privilege gradient paths)     │
  │  Phase 4 (Phase4Result - control-flow regions + steerable branches)      │
  │    ↓  WorkloadProfile                                                    │
  │  MeasurementPlan - what to measure, isolation strategy, trial count      │
  │    ↓  VarianceMeasurer                                                   │
  │  TrialBatch - raw latency + microarch counter observations per operation │
  │    ↓  VarianceProfiler                                                   │
  │  VarianceProfile - distribution stats + noise model per (target, op)     │
  │    ↓  DistinguishabilityAnalyzer                                         │
  │  DistinguishabilityResult - "distinguishable vs not" with evidence       │
  │    ↓                                                                     │
  │  ChannelSignature - mapping: variance signature → higher-layer event     │
  │  Phase7Result - top-level output with exit criterion                     │
  └──────────────────────────────────────────────────────────────────────────┘

Exit criterion
--------------
Phase7Result.exit_criterion_met = True when:
- ≥1 DistinguishabilityResult.is_distinguishable = True under defined noise,
- ≥1 ChannelSignature maps a variance pattern to a higher-layer event, AND
- The claim "execution varies based on sensitive internal values" is supported
  (or falsified) with data.

Key concepts
------------
OperationClass       - the class of operation being measured (crypto, branch, etc.)
IsolationStrategy    - scheduling isolation approach (affinity, quiesce, perf-event)
StatisticalTest      - the statistical test applied (Welch, Mann-Whitney, KS, etc.)
ChannelKind          - the physical side channel category (timing, cache, etc.)
NoiseLevel           - characterisation of measurement environment noise
DistinguishabilityResult - core finding: can an observer classify code path from signal?
ChannelSignature     - mapping: variance profile → protocol/trust-layer event
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ── Enumerations ───────────────────────────────────────────────────────────────


class OperationClass(enum.StrEnum):
    """
    The class of operation whose timing is being profiled.

    CRYPTO_BRANCH     - constant-time vs variable-time crypto decision
    AUTH_DECISION     - accept/reject authentication comparison
    PROTOCOL_GUARD    - guard predicate at a protocol FSM boundary state
    MEMORY_ACCESS     - cache-sensitive data load/store
    CONTROL_FLOW      - branch taken vs not-taken (general)
    SERIALISATION     - variable-length encode/decode
    NETWORK_IO        - syscall latency variation
    CUSTOM            - target-specific operation class
    """

    CRYPTO_BRANCH   = "crypto_branch"
    AUTH_DECISION   = "auth_decision"
    PROTOCOL_GUARD  = "protocol_guard"
    MEMORY_ACCESS   = "memory_access"
    CONTROL_FLOW    = "control_flow"
    SERIALISATION   = "serialisation"
    NETWORK_IO      = "network_io"
    CUSTOM          = "custom"


class ChannelKind(enum.StrEnum):
    """
    Physical side-channel category.

    TIMING_COARSE    - wall-clock, millisecond precision (system-call, HTTP latency)
    TIMING_FINE      - nanosecond-precision via monotonic clock + spin-wait
    CACHE_L1         - L1-cache load-miss distinguishable via Flush+Reload / Prime+Probe
    CACHE_LLC        - Last-level cache cross-core leakage (requires perf events)
    BRANCH_PREDICTOR - speculative execution residue via branch-miss rate
    MEMORY_BUS       - DRAM access pattern via memory-bus contention (coarse)
    POWER_PROXY      - CPU frequency scaling or RAPL energy as power proxy
    UNKNOWN          - unclassified channel
    """

    TIMING_COARSE    = "timing_coarse"
    TIMING_FINE      = "timing_fine"
    CACHE_L1         = "cache_l1"
    CACHE_LLC        = "cache_llc"
    BRANCH_PREDICTOR = "branch_predictor"
    MEMORY_BUS       = "memory_bus"
    POWER_PROXY      = "power_proxy"
    UNKNOWN          = "unknown"


class IsolationStrategy(enum.StrEnum):
    """
    Scheduling isolation approach to reduce noise during measurement.

    NONE             - no special isolation; baseline noise only
    CPU_AFFINITY     - pin measurer and target to same/different core
    QUIESCE_SYSTEM   - disable background services before measurement window
    PERF_EVENT_GUARD - use Linux perf_event_open to gate measurement
    REALTIME_SCHED   - elevate measurer to SCHED_FIFO / SCHED_RR
    CGROUP_ISOLATION - isolate target in dedicated cgroup
    COMBINED         - multiple strategies applied together
    """

    NONE             = "none"
    CPU_AFFINITY     = "cpu_affinity"
    QUIESCE_SYSTEM   = "quiesce_system"
    PERF_EVENT_GUARD = "perf_event_guard"
    REALTIME_SCHED   = "realtime_sched"
    CGROUP_ISOLATION = "cgroup_isolation"
    COMBINED         = "combined"


class StatisticalTest(enum.StrEnum):
    """
    The statistical test applied to determine distributional difference.

    WELCH_T          - Welch's t-test (unequal variance, parametric)
    MANN_WHITNEY_U   - Mann-Whitney U (non-parametric rank test)
    KS_TWO_SAMPLE    - Kolmogorov-Smirnov two-sample test (distribution shape)
    EFFECT_SIZE_D    - Cohen's d effect size (practical significance)
    LEAKAGE_RATIO    - simple ratio of means (coarse operational measure)
    MUTUAL_INFO      - mutual information estimate between class label and latency
    """

    WELCH_T        = "welch_t"
    MANN_WHITNEY_U = "mann_whitney_u"
    KS_TWO_SAMPLE  = "ks_two_sample"
    EFFECT_SIZE_D  = "effect_size_d"
    LEAKAGE_RATIO  = "leakage_ratio"
    MUTUAL_INFO    = "mutual_info"


class NoiseLevel(enum.StrEnum):
    """
    Characterisation of the noise environment during measurement.

    LOW     - controlled environment (isolated core, quiesced system, perf-event-gated)
    MEDIUM  - semi-controlled (CPU affinity only, background services present)
    HIGH    - uncontrolled (production-like; shared CPU, OS scheduler noise)
    UNKNOWN - noise characterisation not performed
    """

    LOW     = "low"
    MEDIUM  = "medium"
    HIGH    = "high"
    UNKNOWN = "unknown"


class DistinguishabilityVerdict(enum.StrEnum):
    """
    Outcome of a distinguishability test for one (operation_a, operation_b) pair.

    DISTINGUISHABLE     - statistically significant difference with sufficient effect size
    NOT_DISTINGUISHABLE - no significant difference under the noise conditions tested
    MARGINAL            - significant difference but small effect size; may need tighter isolation
    INCONCLUSIVE        - insufficient trials or high variance; cannot determine
    ERROR               - measurement infrastructure failure
    """

    DISTINGUISHABLE     = "distinguishable"
    NOT_DISTINGUISHABLE = "not_distinguishable"
    MARGINAL            = "marginal"
    INCONCLUSIVE        = "inconclusive"
    ERROR               = "error"


# ── Measurement plan ───────────────────────────────────────────────────────────


class OperationSpec(EOSBaseModel):
    """
    Specification of a single operation to measure.

    An operation is a (target_function, sensitive_value_class) pair.
    The measurer will invoke this operation with two input classes:
      - class A: "sensitive" inputs (e.g. incorrect auth token, boundary counter)
      - class B: "neutral" inputs (e.g. correct auth token, nominal counter value)
    """

    op_id: str = Field(default_factory=new_id)
    name: str = Field(..., description="e.g. 'hmac_verify_correct_vs_incorrect'")
    operation_class: OperationClass
    channel_kind: ChannelKind = ChannelKind.TIMING_COARSE

    # Link back to Phase 4/5/6 artifacts that motivated this measurement
    source_region_id: str = Field(
        default="",
        description="Phase 4 ControlFlowRegion ID that suggested this operation",
    )
    source_fsm_state_id: str = Field(
        default="",
        description="Phase 6 ProtocolFsmState ID of the boundary being measured",
    )
    source_corridor_id: str = Field(
        default="",
        description="Phase 5 ExpansionCorridor ID linking trust edge to this operation",
    )

    # Class A = the class believed to activate a different code path
    class_a_label: str = Field(default="sensitive", description="Label for class A inputs")
    class_b_label: str = Field(default="neutral",   description="Label for class B inputs")

    # Human-readable description of the distinguishability hypothesis
    hypothesis: str = Field(
        default="",
        description=(
            "One-sentence hypothesis: 'HMAC verification runs longer for incorrect "
            "tokens due to variable-time string comparison.'"
        ),
    )

    spec_reference: str = Field(default="")
    metadata: dict[str, Any] = Field(default_factory=dict)


class MeasurementPlan(EOSBaseModel):
    """
    A complete measurement plan for one analysis target.

    Specifies what to measure, how many trials to run, which isolation
    strategy to use, and which statistical tests to apply.
    """

    plan_id: str = Field(default_factory=new_id)
    target_id: str

    operations: list[OperationSpec] = Field(default_factory=list)

    # Trial budget
    trials_per_class: int = Field(
        default=1000,
        description="Number of measurements per input class (A and B) per operation",
    )
    warmup_trials: int = Field(
        default=50,
        description="Discarded warm-up trials before measurement begins",
    )

    # Isolation
    isolation_strategy: IsolationStrategy = IsolationStrategy.CPU_AFFINITY
    target_cpu_core: int | None = None
    measurer_cpu_core: int | None = None

    # Statistical configuration
    significance_level: float = Field(
        default=0.01,
        description="p-value threshold for significance (α)",
    )
    min_effect_size_d: float = Field(
        default=0.2,
        description="Minimum Cohen's d for practical significance",
    )
    tests: list[StatisticalTest] = Field(
        default_factory=lambda: [
            StatisticalTest.WELCH_T,
            StatisticalTest.MANN_WHITNEY_U,
            StatisticalTest.KS_TWO_SAMPLE,
            StatisticalTest.EFFECT_SIZE_D,
        ]
    )

    # Noise characterisation
    noise_characterisation_trials: int = Field(
        default=200,
        description="Trials to characterise baseline noise (same input, repeated)",
    )

    built_at: datetime = Field(default_factory=utc_now)


# ── Raw measurement ────────────────────────────────────────────────────────────


class MicroarchCounters(EOSBaseModel):
    """
    Microarchitectural performance counters for a single trial.

    All fields are optional - only those available on the current platform
    and with the current isolation strategy will be populated.
    """

    cycles: int | None = None
    instructions: int | None = None
    cache_misses: int | None = None
    cache_references: int | None = None
    branch_misses: int | None = None
    branch_instructions: int | None = None
    llc_load_misses: int | None = None
    llc_load_references: int | None = None
    context_switches: int | None = None
    cpu_migrations: int | None = None

    # Derived ratios (populated by profiler if raw counts available)
    ipc: float | None = Field(default=None, description="Instructions per cycle")
    cache_miss_rate: float | None = None
    branch_miss_rate: float | None = None
    llc_miss_rate: float | None = None


class TrialObservation(EOSBaseModel):
    """
    A single timing observation for one invocation of an operation.

    Carries wall-clock latency (primary observable) and optional
    microarchitectural counters (secondary, platform-dependent).
    """

    trial_id: str = Field(default_factory=new_id)
    op_id: str
    input_class: str = Field(..., description="'sensitive' or 'neutral' (class A or B)")
    trial_index: int

    # Primary observable: wall-clock latency in nanoseconds
    latency_ns: int

    # Secondary observables
    counters: MicroarchCounters = Field(default_factory=MicroarchCounters)

    # Metadata
    cpu_core: int | None = None
    os_context_switches_during: int = 0
    was_warmed_up: bool = False

    measured_at: datetime = Field(default_factory=utc_now)


class TrialBatch(EOSBaseModel):
    """
    All raw observations for a single OperationSpec.

    Contains trials for both input classes (A=sensitive, B=neutral),
    organised for downstream statistical analysis.
    """

    batch_id: str = Field(default_factory=new_id)
    op_id: str
    target_id: str

    class_a_observations: list[TrialObservation] = Field(default_factory=list)
    class_b_observations: list[TrialObservation] = Field(default_factory=list)

    # Baseline noise characterisation (same input class, repeated)
    noise_baseline_observations: list[TrialObservation] = Field(default_factory=list)

    isolation_strategy: IsolationStrategy = IsolationStrategy.NONE
    noise_level: NoiseLevel = NoiseLevel.UNKNOWN

    collected_at: datetime = Field(default_factory=utc_now)


# ── Variance profile ───────────────────────────────────────────────────────────


class DistributionStats(EOSBaseModel):
    """
    Descriptive statistics for one sample (class A or B, or baseline noise).
    All latency values in nanoseconds.
    """

    n: int = 0
    mean_ns: float = 0.0
    median_ns: float = 0.0
    std_ns: float = 0.0
    min_ns: float = 0.0
    max_ns: float = 0.0
    p5_ns: float = 0.0
    p25_ns: float = 0.0
    p75_ns: float = 0.0
    p95_ns: float = 0.0
    p99_ns: float = 0.0

    # Outlier-trimmed stats (10% trimmed mean)
    trimmed_mean_ns: float = 0.0
    trimmed_std_ns: float = 0.0

    # Coefficient of variation (std / mean)
    cv: float = 0.0


class StatTestResult(EOSBaseModel):
    """
    Result of a single statistical test comparing class A vs class B.
    """

    test: StatisticalTest
    statistic: float = Field(..., description="Test statistic value (t, U, D, d, etc.)")
    p_value: float | None = None
    effect_size: float | None = Field(
        default=None,
        description="Effect size (Cohen's d, or normalised variant for non-parametric tests)",
    )
    is_significant: bool = False
    detail: str = Field(default="", description="Human-readable interpretation")


class NoiseModel(EOSBaseModel):
    """
    Characterisation of measurement noise for this (target, operation) pair.

    Derived from noise_baseline_observations: repeated measurements of the
    same input to quantify scheduling jitter, cache warm/cold variation, etc.
    """

    baseline_stats: DistributionStats = Field(default_factory=DistributionStats)

    # Noise level classification
    noise_level: NoiseLevel = NoiseLevel.UNKNOWN

    # Coefficient of variation of the baseline (low CV → low noise)
    baseline_cv: float = 0.0

    # Minimum detectable effect size given this noise level
    # (effect size below which results are unreliable)
    min_detectable_effect_d: float = 0.0

    # Whether this noise model suggests the channel is viable for analysis
    is_viable: bool = False
    viability_reason: str = Field(default="")


class VarianceProfile(EOSBaseModel):
    """
    Complete variance characterisation for one (target, operation) pair.

    This is the primary Phase 7 deliverable per operation: it records
    both the statistical evidence of distributional difference and the
    noise model that contextualises that evidence.
    """

    profile_id: str = Field(default_factory=new_id)
    op_id: str
    target_id: str
    op_name: str
    operation_class: OperationClass
    channel_kind: ChannelKind

    # Distribution statistics per class
    class_a_stats: DistributionStats = Field(default_factory=DistributionStats)
    class_b_stats: DistributionStats = Field(default_factory=DistributionStats)

    # Noise model
    noise_model: NoiseModel = Field(default_factory=NoiseModel)

    # All statistical test results
    test_results: list[StatTestResult] = Field(default_factory=list)

    # Mean latency delta: class_a_mean - class_b_mean
    mean_delta_ns: float = 0.0
    mean_delta_relative: float = Field(
        default=0.0, description="mean_delta / class_b_mean (fractional)"
    )

    # Maximum effect size observed across all tests
    max_effect_size: float = 0.0

    # Overall verdict
    verdict: DistinguishabilityVerdict = DistinguishabilityVerdict.INCONCLUSIVE

    # Link to Phase 4/5/6 artifacts that motivated this measurement
    source_region_id: str = Field(default="")
    source_fsm_state_id: str = Field(default="")
    source_corridor_id: str = Field(default="")

    profiled_at: datetime = Field(default_factory=utc_now)


# ── Distinguishability result ──────────────────────────────────────────────────


class DistinguishabilityResult(EOSBaseModel):
    """
    The core Phase 7 finding: can an observer classify which code path
    (class A vs class B) was taken, from timing observations alone?

    This is the deliverable used by Phase 8 cross-layer correlation:
    it provides a concrete evidence anchor for "execution varies with
    sensitive internal values."
    """

    result_id: str = Field(default_factory=new_id)
    target_id: str
    op_id: str
    op_name: str
    operation_class: OperationClass
    channel_kind: ChannelKind

    # Core verdict
    verdict: DistinguishabilityVerdict
    is_distinguishable: bool = False

    # Supporting evidence
    profile: VarianceProfile

    # Which statistical tests agree (all that reported significant)
    confirming_tests: list[StatisticalTest] = Field(default_factory=list)
    # Which tests reported non-significant (do not falsify if confirming_tests non-empty)
    non_significant_tests: list[StatisticalTest] = Field(default_factory=list)

    # Practical significance
    mean_delta_ns: float = 0.0
    max_effect_size_d: float = 0.0

    # Noise level under which the result was obtained
    noise_level: NoiseLevel = NoiseLevel.UNKNOWN

    # Human-readable evidence summary
    evidence_summary: str = Field(
        default="",
        description=(
            "One-paragraph narrative: 'Under CPU_AFFINITY isolation (medium noise), "
            "HMAC verification for incorrect tokens was 23% slower (mean=4.3μs vs 3.5μs, "
            "Cohen's d=0.87, p<0.001). The channel is distinguishable under medium noise.'"
        ),
    )

    # Whether the result is security-relevant
    is_security_relevant: bool = False
    security_relevance_reason: str = Field(default="")

    # Link to higher-layer events (populated by ChannelSignature mapping)
    linked_event_ids: list[str] = Field(default_factory=list)

    produced_at: datetime = Field(default_factory=utc_now)


# ── Channel signature ──────────────────────────────────────────────────────────


class HigherLayerEvent(EOSBaseModel):
    """
    A higher-layer event (protocol decision, trust edge, auth outcome)
    that a channel signature maps to a variance profile.

    Bridges the raw timing observable to meaningful system semantics.
    """

    event_id: str = Field(default_factory=new_id)

    # Where this event lives in the stack
    layer: str = Field(
        ...,
        description="protocol | trust | auth | control_flow | state_machine",
    )

    # Cross-reference to Phase 4/5/6
    phase4_region_id: str = Field(default="")
    phase5_corridor_id: str = Field(default="")
    phase6_boundary_failure_id: str = Field(default="")
    phase6_fsm_state_id: str = Field(default="")

    # Human-readable description
    description: str = Field(
        ...,
        description=(
            "e.g. 'Sequence counter overflow at TCP DATA_TRANSFER state', "
            "'Auth window expiry during credential re-issuance'"
        ),
    )

    # Estimated sensitivity (0=not sensitive, 100=maximally sensitive)
    sensitivity: int = Field(default=50, ge=0, le=100)


class ChannelSignature(EOSBaseModel):
    """
    A mapping between a variance signature and one or more higher-layer events.

    This is the final Phase 7 deliverable: it answers "what does this timing
    variation *mean*?" by linking the observable signal to the system behaviour
    that causes it.

    Multiple DistinguishabilityResults may share a ChannelSignature if they
    exhibit structurally similar variance patterns.
    """

    signature_id: str = Field(default_factory=new_id)
    target_id: str

    # The variance pattern being described
    channel_kind: ChannelKind
    operation_class: OperationClass

    # All DistinguishabilityResults that match this signature
    distinguishability_result_ids: list[str] = Field(default_factory=list)

    # The higher-layer events this signature maps to
    higher_layer_events: list[HigherLayerEvent] = Field(default_factory=list)

    # Signature characteristics
    mean_delta_ns: float = 0.0
    mean_effect_size_d: float = 0.0
    noise_level: NoiseLevel = NoiseLevel.UNKNOWN

    # Confidence that this signature is real and reproducible (not noise)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Human-readable mapping narrative
    signature_narrative: str = Field(
        default="",
        description=(
            "End-to-end narrative: 'A ~23% latency increase in HMAC verification "
            "correlates with Phase 5 trust corridor TC-07 (auth-window re-issue) "
            "and Phase 6 boundary failure BF-12 (auth window expiry mid-flight). "
            "The channel signature is observable under medium noise conditions.'"
        ),
    )

    # Security relevance
    is_security_relevant: bool = False

    recorded_at: datetime = Field(default_factory=utc_now)


# ── Phase 7 result ─────────────────────────────────────────────────────────────


class Phase7Result(EOSBaseModel):
    """
    Top-level output of a Phase 7 execution variance analysis session.

    Wraps the MeasurementPlan, all VarianceProfiles, all
    DistinguishabilityResults, and all ChannelSignatures, with aggregate
    statistics and the exit criterion flag.

    Exit criterion
    --------------
    exit_criterion_met = True when:
    - ≥1 DistinguishabilityResult.is_distinguishable = True under defined noise,
    - ≥1 ChannelSignature maps a variance pattern to a higher-layer event, AND
    - The claim "execution varies based on sensitive internal values" is
      either supported or falsified with data.
    """

    id: str = Field(default_factory=new_id)
    target_id: str

    # Core artifacts
    measurement_plan: MeasurementPlan
    variance_profiles: list[VarianceProfile] = Field(default_factory=list)
    distinguishability_results: list[DistinguishabilityResult] = Field(default_factory=list)
    channel_signatures: list[ChannelSignature] = Field(default_factory=list)

    # Aggregate statistics
    total_operations_measured: int = 0
    total_distinguishable: int = 0
    total_not_distinguishable: int = 0
    total_marginal: int = 0
    total_inconclusive: int = 0
    total_channel_signatures: int = 0
    security_relevant_channels: int = 0

    # Cross-layer coverage: how many Phase 4/5/6 artifacts were linked to a channel
    phase4_regions_linked: int = 0
    phase5_corridors_linked: int = 0
    phase6_failures_linked: int = 0

    # Dominant noise level across measurements
    dominant_noise_level: NoiseLevel = NoiseLevel.UNKNOWN

    # Overall claim verdict
    claim_supported: bool | None = Field(
        default=None,
        description=(
            "True = execution varies with sensitive values (distinguishable channels found). "
            "False = execution does not measurably vary (no distinguishable channels). "
            "None = inconclusive (insufficient data or all results marginal)."
        ),
    )
    claim_evidence_summary: str = Field(
        default="",
        description="Paragraph summarising support or falsification of the claim",
    )

    # Exit criterion
    exit_criterion_met: bool = Field(
        default=False,
        description=(
            "True when: ≥1 distinguishable channel found, ≥1 channel signature "
            "maps variance to higher-layer event, and claim is supported or falsified."
        ),
    )

    produced_at: datetime = Field(default_factory=utc_now)
