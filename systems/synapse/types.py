"""
EcodiaOS — Synapse Type Definitions

All data types for the autonomic nervous system: cycle clock, health monitoring,
resource allocation, degradation strategies, emergent rhythm detection,
and cross-system coherence measurement.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from datetime import datetime  # noqa: TC003
from typing import Any, Protocol

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ─── System Status ────────────────────────────────────────────────────


class SystemStatus(enum.StrEnum):
    """Operational state of a managed cognitive system."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    STOPPED = "stopped"
    STARTING = "starting"
    RESTARTING = "restarting"


# ─── Health Monitoring ────────────────────────────────────────────────


class SystemHeartbeat(EOSBaseModel):
    """Health report returned by a managed system's health() method."""

    system_id: str
    status: str = "healthy"
    latency_ms: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class SystemHealthRecord(EOSBaseModel):
    """
    Synapse's internal per-system health tracking.

    Tracks heartbeat history, consecutive misses, latency statistics,
    and error accumulation for degradation decisions.
    """

    system_id: str
    status: SystemStatus = SystemStatus.STOPPED
    consecutive_misses: int = 0
    consecutive_successes: int = 0
    total_checks: int = 0
    total_failures: int = 0
    last_check_time: datetime | None = None
    last_success_time: datetime | None = None
    last_failure_time: datetime | None = None
    # Exponential moving average of heartbeat latency
    latency_ema_ms: float = 0.0
    # Peak latency in current window
    latency_peak_ms: float = 0.0
    # Number of restarts attempted
    restart_count: int = 0
    # Is this a critical system (failure → safe mode)?
    is_critical: bool = False

    def record_success(self, latency_ms: float) -> None:
        """Record a successful heartbeat."""
        now = utc_now()
        self.consecutive_misses = 0
        self.consecutive_successes += 1
        self.total_checks += 1
        self.last_check_time = now
        self.last_success_time = now
        # EMA with alpha=0.2 for smooth tracking
        alpha = 0.2
        self.latency_ema_ms = (
            alpha * latency_ms + (1 - alpha) * self.latency_ema_ms
        )
        self.latency_peak_ms = max(self.latency_peak_ms, latency_ms)
        # Recover from degraded states
        # REVIEW: DEGRADED/OVERLOADED recover on the *first* success (no threshold)
        # while FAILED requires 3 consecutive successes. This asymmetry may be
        # intentional (soft degradation vs hard failure), but if OVERLOADED should
        # also require multiple confirmations, add `and self.consecutive_successes >= 2`
        # to the second branch.
        if (
            self.status == SystemStatus.FAILED and self.consecutive_successes >= 3
            or self.status in (SystemStatus.DEGRADED, SystemStatus.OVERLOADED)
        ):
            self.status = SystemStatus.HEALTHY

    def record_failure(self) -> None:
        """Record a missed or failed heartbeat."""
        now = utc_now()
        self.consecutive_successes = 0
        self.consecutive_misses += 1
        self.total_checks += 1
        self.total_failures += 1
        self.last_check_time = now
        self.last_failure_time = now

    def record_overloaded(self, latency_ms: float) -> None:
        """Record a successful but slow heartbeat (latency > 2x EMA)."""
        self.record_success(latency_ms)
        if self.status == SystemStatus.HEALTHY:
            self.status = SystemStatus.OVERLOADED


# ─── Resource Allocation ──────────────────────────────────────────────


class SystemBudget(EOSBaseModel):
    """Per-system resource budget allocation."""

    system_id: str = ""
    cpu_share: float = Field(0.1, ge=0.0, le=1.0)
    memory_mb: int = 256
    io_priority: int = Field(3, ge=1, le=5)  # 1 = highest


class ResourceAllocation(EOSBaseModel):
    """
    Allocation message delivered to a system each rebalance.

    Translates abstract budgets into per-cycle concrete limits.
    """

    system_id: str
    compute_ms_per_cycle: float = 50.0
    burst_allowance: float = Field(1.0, ge=1.0, le=3.0)
    priority_boost: float = Field(0.0, ge=-1.0, le=1.0)
    timestamp: datetime = Field(default_factory=utc_now)


class ResourceSnapshot(EOSBaseModel):
    """Point-in-time resource utilisation snapshot across all systems."""

    timestamp: datetime = Field(default_factory=utc_now)
    total_cpu_percent: float = 0.0
    total_memory_mb: float = 0.0
    total_memory_percent: float = 0.0
    per_system: dict[str, dict[str, float]] = Field(default_factory=dict)
    process_cpu_percent: float = 0.0
    process_memory_mb: float = 0.0


# ─── Clock ────────────────────────────────────────────────────────────


class SomaticCycleState(EOSBaseModel):
    """
    Somatic snapshot carried on every theta tick.

    Extracted from Soma's AllostaticSignal after step 0 runs. Downstream
    consumers (Nova, Evo, coherence monitor) read this from CycleResult
    rather than holding a direct Soma reference, keeping coupling minimal.

    All fields default to neutral/quiescent values so the struct is safe
    to construct when Soma is absent or degraded.
    """

    urgency: float = 0.0
    """Scalar [0,1] allostatic urgency — how far from all setpoints."""

    dominant_error: str = "energy"
    """Name of the InteroceptiveDimension with the largest allostatic error."""

    arousal_sensed: float = 0.4
    """Raw sensed AROUSAL dimension [0,1] — used by clock for adaptive timing."""

    energy_sensed: float = 0.6
    """Raw sensed ENERGY dimension [0,1]."""

    precision_weights: dict[str, float] = Field(default_factory=dict)
    """Per-dimension precision weights from Soma. Empty dict = uniform."""

    nearest_attractor: str | None = None
    """Label of the nearest phase-space attractor, or None if transient."""

    trajectory_heading: str = "transient"
    """Phase-space heading: 'approaching', 'within', 'departing', 'transient'."""

    soma_cycle_ms: float = 0.0
    """How long Soma's own run_cycle() took (for overrun diagnostics)."""


class SomaTickEvent(EOSBaseModel):
    """
    Event emitted by Synapse on the Redis event bus after every theta tick
    where Soma ran. Carries the full somatic state for stateless consumers.

    Channel: ``synapse.soma.tick``
    Payload: this model serialised to JSON.
    """

    id: str = Field(default_factory=new_id)
    cycle_number: int
    somatic_state: SomaticCycleState
    timestamp: datetime = Field(default_factory=utc_now)


class CycleResult(EOSBaseModel):
    """Result of a single theta rhythm tick."""

    cycle_number: int
    elapsed_ms: float
    budget_ms: float
    overrun: bool = False
    broadcast_id: str | None = None
    had_broadcast: bool = False
    arousal: float = 0.0
    salience_composite: float = 0.0
    # Somatic state snapshot from Soma step 0 (None when Soma absent/degraded)
    somatic: SomaticCycleState | None = None
    timestamp: datetime = Field(default_factory=utc_now)


class ClockState(EOSBaseModel):
    """Snapshot of the cognitive clock's current state."""

    running: bool = False
    paused: bool = False
    cycle_count: int = 0
    current_period_ms: float = 150.0
    target_period_ms: float = 150.0
    jitter_ms: float = 0.0
    arousal: float = 0.0
    overrun_count: int = 0
    # Cycles per second (actual measured rate)
    actual_rate_hz: float = 0.0


# ─── Degradation ──────────────────────────────────────────────────────


class DegradationLevel(enum.StrEnum):
    """Overall organism degradation level."""

    NOMINAL = "nominal"
    DEGRADED = "degraded"
    SAFE_MODE = "safe_mode"
    EMERGENCY = "emergency"


class DegradationStrategy(EOSBaseModel):
    """Per-system fallback configuration."""

    system_id: str
    triggers_safe_mode: bool = False
    fallback_behavior: str = ""
    auto_restart: bool = True
    max_restart_attempts: int = 3
    restart_backoff_base_s: float = 5.0


# ─── Event Bus ────────────────────────────────────────────────────────


class SynapseEventType(enum.StrEnum):
    """All event types emitted by Synapse."""

    # System lifecycle
    SYSTEM_STARTED = "system_started"
    SYSTEM_STOPPED = "system_stopped"
    SYSTEM_FAILED = "system_failed"
    SYSTEM_RECOVERED = "system_recovered"
    SYSTEM_RESTARTING = "system_restarting"
    SYSTEM_RELOADING = "system_reloading"
    SYSTEM_OVERLOADED = "system_overloaded"

    # Safe mode
    SAFE_MODE_ENTERED = "safe_mode_entered"
    SAFE_MODE_EXITED = "safe_mode_exited"

    # Clock
    CLOCK_STARTED = "clock_started"
    CLOCK_STOPPED = "clock_stopped"
    CLOCK_PAUSED = "clock_paused"
    CLOCK_RESUMED = "clock_resumed"
    CLOCK_OVERRUN = "clock_overrun"

    # Cognitive cycle
    CYCLE_COMPLETED = "cycle_completed"
    # Somatic tick — emitted every cycle where Soma ran successfully
    SOMA_TICK = "soma_tick"

    # Interoceptive percept — Soma broadcasting an internal sensation
    # through the Global Workspace when analysis thresholds are exceeded.
    # Payload: InteroceptivePercept serialised to dict.
    INTEROCEPTIVE_PERCEPT = "interoceptive_percept"

    # Rhythm (emergent)
    RHYTHM_STATE_CHANGED = "rhythm_state_changed"

    # Coherence
    COHERENCE_SHIFT = "coherence_shift"

    # Resources
    RESOURCE_REBALANCED = "resource_rebalanced"
    RESOURCE_PRESSURE = "resource_pressure"

    # Metabolic (financial burn rate)
    METABOLIC_PRESSURE = "metabolic_pressure"

    # Funding request — organism is broke and asking for capital
    FUNDING_REQUEST_ISSUED = "funding_request_issued"

    # Financial events (on-chain wallet activity + revenue injection)
    # These bypass normal SalienceHead calculation and encode at salience=1.0.
    # Biologically equivalent to trauma or a massive meal — must not decay easily.
    WALLET_TRANSFER_CONFIRMED = "wallet_transfer_confirmed"
    REVENUE_INJECTED = "revenue_injected"

    # Mitosis lifecycle (Phase 16e: Speciation)
    CHILD_SPAWNED = "child_spawned"
    CHILD_HEALTH_REPORT = "child_health_report"
    CHILD_STRUGGLING = "child_struggling"
    CHILD_RESCUED = "child_rescued"
    CHILD_INDEPENDENT = "child_independent"
    CHILD_DIED = "child_died"
    DIVIDEND_RECEIVED = "dividend_received"

    # Fleet management (Phase 16m)
    FLEET_EVALUATED = "fleet_evaluated"
    FLEET_ROLE_CHANGED = "fleet_role_changed"

    # Bounty payout — PR merged and bounty reward confirmed
    BOUNTY_PAID = "bounty_paid"

    # Bounty solution staged — solution generated, awaiting PR submission
    BOUNTY_SOLUTION_PENDING = "bounty_solution_pending"

    # Bounty PR submitted — pull request opened on target repo
    BOUNTY_PR_SUBMITTED = "bounty_pr_submitted"

    # GitHub credentials absent — operator must provision GITHUB_TOKEN or GitHub App config
    GITHUB_CREDENTIALS_MISSING = "github_credentials_missing"

    # Bounty sources unreachable — both Algora and GitHub failed to respond
    BOUNTY_SOURCE_UNAVAILABLE = "bounty_source_unavailable"

    # Economic immune system (Phase 16f)
    TRANSACTION_SHIELDED = "transaction_shielded"
    THREAT_DETECTED = "threat_detected"
    PROTOCOL_ALERT = "protocol_alert"
    EMERGENCY_WITHDRAWAL = "emergency_withdrawal"
    THREAT_ADVISORY_RECEIVED = "threat_advisory_received"
    THREAT_ADVISORY_SENT = "threat_advisory_sent"
    ADDRESS_BLACKLISTED = "address_blacklisted"

    # Bounty hunting (Phase 16b: The Freelancer)
    BOUNTY_DISCOVERED = "bounty_discovered"
    BOUNTY_EVALUATED = "bounty_evaluated"
    FORAGING_CYCLE_COMPLETE = "foraging_cycle_complete"

    # Reputation & credit (Phase 16g: Autonomous Credit)
    REPUTATION_UPDATED = "reputation_updated"
    CREDIT_DRAWN = "credit_drawn"
    CREDIT_REPAID = "credit_repaid"

    # Protocol infrastructure (Level 5: The Protocol)
    PROTOCOL_DESIGNED = "protocol_designed"
    PROTOCOL_DEPLOYED = "protocol_deployed"
    PROTOCOL_REVENUE_SWEPT = "protocol_revenue_swept"
    PROTOCOL_TERMINATED = "protocol_terminated"

    # Interspecies economy (Phase 16j: Fleet-Scale Coordination)
    CAPABILITY_OFFERED = "capability_offered"
    CAPABILITY_REQUESTED = "capability_requested"
    CAPABILITY_TRADE_SETTLED = "capability_trade_settled"
    INSURANCE_PREMIUM_PAID = "insurance_premium_paid"
    INSURANCE_CLAIM_FILED = "insurance_claim_filed"
    INSURANCE_CLAIM_APPROVED = "insurance_claim_approved"
    NICHE_ASSIGNED = "niche_assigned"

    # Economic immune cycle (Phase 16f)
    IMMUNE_CYCLE_COMPLETE = "immune_cycle_complete"

    # Certificate lifecycle (Phase 16g: Civilization Layer)
    CERTIFICATE_EXPIRING = "certificate_expiring"
    CERTIFICATE_EXPIRED = "certificate_expired"

    # Inbound verification code received via SMS or email (Phase 16h)
    IDENTITY_VERIFICATION_RECEIVED = "identity_verification_received"

    # Platform connector lifecycle (Phase 16h: External Identity Layer)
    CONNECTOR_AUTHENTICATED = "connector_authenticated"
    CONNECTOR_TOKEN_REFRESHED = "connector_token_refreshed"
    CONNECTOR_TOKEN_EXPIRED = "connector_token_expired"
    CONNECTOR_REVOKED = "connector_revoked"
    CONNECTOR_ERROR = "connector_error"
    # Connector health degradation — 3 consecutive failures; Thymos should quarantine
    SYSTEM_DEGRADED = "system_degraded"

    # Knowledge market (Phase 16h: Cognition as Commodity)
    KNOWLEDGE_PRODUCT_REQUESTED = "knowledge_product_requested"
    KNOWLEDGE_PRODUCT_DELIVERED = "knowledge_product_delivered"
    KNOWLEDGE_SALE_RECORDED = "knowledge_sale_recorded"

    # Economic dreaming — actionable recommendations from Monte Carlo
    ECONOMIC_DREAM_RECOMMENDATION = "economic_dream_recommendation"

    # Economic morphogenesis (Phase 16l)
    ORGAN_CREATED = "organ_created"
    ORGAN_TRANSITION = "organ_transition"
    ORGAN_RESOURCE_REBALANCED = "organ_resource_rebalanced"

    # Cross-modal synesthesia — external volatility mapped into somatic state
    SOMA_STATE_SPIKE = "soma_state_spike"

    # Physical grid carbon intensity — emitted when MetabolicState changes
    GRID_METABOLISM_CHANGED = "grid_metabolism_changed"

    # Energy-aware scheduler — emitted when a high-compute task is deferred
    TASK_ENERGY_DEFERRED = "task_energy_deferred"

    # Shadow model assessment — adapter passed all benchmarks, ready for promotion
    MODEL_EVALUATION_PASSED = "model_evaluation_passed"

    # Model hot-swap lifecycle — live adapter transition and rollback
    MODEL_HOT_SWAP_STARTED = "model_hot_swap_started"
    MODEL_HOT_SWAP_COMPLETED = "model_hot_swap_completed"
    MODEL_HOT_SWAP_FAILED = "model_hot_swap_failed"
    MODEL_ROLLBACK_TRIGGERED = "model_rollback_triggered"
    CATASTROPHIC_FORGETTING_DETECTED = "catastrophic_forgetting_detected"

    # Oneiros (Dream Engine) lifecycle events
    SLEEP_ONSET = "sleep_onset"
    SLEEP_STAGE_CHANGED = "sleep_stage_changed"
    DREAM_INSIGHT = "dream_insight"
    WAKE_ONSET = "wake_onset"
    SLEEP_PRESSURE_WARNING = "sleep_pressure_warning"
    SLEEP_FORCED = "sleep_forced"
    EMERGENCY_WAKE = "emergency_wake"

    # Oneiros v2 — Sleep as Batch Compiler (Spec 14)
    #
    # SLEEP_INITIATED — emitted at Descent start.
    # Payload: trigger (str: "scheduled"|"cognitive_pressure"|"compression_backlog"),
    #          checkpoint_id (str), scheduled_duration_s (float)
    SLEEP_INITIATED = "sleep_initiated"
    #
    # SLEEP_STAGE_TRANSITION — emitted at each stage boundary.
    # Payload: from_stage (str), to_stage (str), stage_report (dict|None)
    SLEEP_STAGE_TRANSITION = "sleep_stage_transition"
    #
    # COMPRESSION_BACKLOG_PROCESSED — emitted at end of Slow Wave memory ladder.
    # Payload: MemoryLadderReport fields
    COMPRESSION_BACKLOG_PROCESSED = "compression_backlog_processed"
    #
    # CAUSAL_GRAPH_RECONSTRUCTED — emitted after causal reconstruction.
    # Payload: CausalReconstructionReport fields
    CAUSAL_GRAPH_RECONSTRUCTED = "causal_graph_reconstructed"
    #
    # CROSS_DOMAIN_MATCH_FOUND — emitted when REM finds structural isomorphism
    # between schemas from different domains.
    # Payload: schema_a_id, schema_b_id, isomorphism_score, abstract_structure,
    #          proposed_unified_schema (dict), mdl_improvement (float)
    CROSS_DOMAIN_MATCH_FOUND = "cross_domain_match_found"
    #
    # ANALOGY_DISCOVERED — emitted when a causal invariant transfers across domains.
    # Payload: invariant_id, source_domains (list[str]), domain_count (int),
    #          predictive_transfer_value (float), mdl_improvement (float)
    ANALOGY_DISCOVERED = "analogy_discovered"
    #
    # DREAM_HYPOTHESES_GENERATED — emitted after dream generation produces new hypotheses.
    # Payload: hypotheses (list[dict]), count (int), target_domains (list[str])
    DREAM_HYPOTHESES_GENERATED = "dream_hypotheses_generated"
    #
    # LUCID_DREAM_RESULT — emitted after mutation testing in lucid dreaming mode.
    # Payload: mutation_id (str), performance_delta (float),
    #          constitutional_violations (list), recommendation (str)
    LUCID_DREAM_RESULT = "lucid_dream_result"
    #
    # WAKE_INITIATED — emitted at Emergence completion.
    # Payload: intelligence_improvement (float), sleep_duration_s (float),
    #          sleep_narrative (str), pre_attention_cache_size (int)
    WAKE_INITIATED = "wake_initiated"

    # INTELLIGENCE_IMPROVEMENT_DECLINING — emitted when Emergence detects that
    # recent sleep cycle improvements are below historical average.
    # Signals to Telos Growth that new domain exposure is needed.
    # Payload: average_improvement (float), recent_improvement (float),
    #          history_length (int), signal (str)
    INTELLIGENCE_IMPROVEMENT_DECLINING = "intelligence_improvement_declining"

    # Skia shadow infrastructure — autonomous resilience (Phase 16n)
    SKIA_HEARTBEAT_LOST = "skia_heartbeat_lost"
    SKIA_SNAPSHOT_COMPLETED = "skia_snapshot_completed"
    SKIA_RESTORATION_TRIGGERED = "skia_restoration_triggered"
    SKIA_RESTORATION_COMPLETED = "skia_restoration_completed"

    # Compute arbitrage — autonomous provider migration (Phase 16o)
    COMPUTE_ARBITRAGE_DETECTED = "compute_arbitrage_detected"
    COMPUTE_MIGRATION_STARTED = "compute_migration_started"
    COMPUTE_MIGRATION_COMPLETED = "compute_migration_completed"
    COMPUTE_MIGRATION_FAILED = "compute_migration_failed"

    # SACM compute resource management — resource allocation arbitration
    # Inbound: systems request compute via COMPUTE_REQUEST_SUBMITTED
    # Outbound: SACM publishes allocation decisions and capacity alerts
    COMPUTE_REQUEST_SUBMITTED = "compute_request_submitted"
    COMPUTE_REQUEST_ALLOCATED = "compute_request_allocated"
    COMPUTE_REQUEST_QUEUED = "compute_request_queued"
    COMPUTE_REQUEST_DENIED = "compute_request_denied"
    COMPUTE_CAPACITY_EXHAUSTED = "compute_capacity_exhausted"
    COMPUTE_FEDERATION_OFFLOADED = "compute_federation_offloaded"

    # Legal entity provisioning — staged orchestration with HITL gates
    ENTITY_FORMATION_STARTED = "entity_formation_started"
    ENTITY_FORMATION_HITL_REQUIRED = "entity_formation_hitl_required"
    ENTITY_FORMATION_RESUMED = "entity_formation_resumed"
    ENTITY_FORMATION_COMPLETED = "entity_formation_completed"
    ENTITY_FORMATION_FAILED = "entity_formation_failed"

    # Axon intent execution outcome — emitted after every intent completes.
    # Payload fields: intent_id (str), outcome (str), success (bool),
    #                 economic_delta (float, USD, signed)
    ACTION_COMPLETED = "action_completed"

    # Nova belief / policy feedback — consumed by Atune to update its prediction
    # model (prediction error signal for perceptual learning).
    #
    # BELIEF_UPDATED payload:
    #   percept_id   (str)   — Percept that triggered the belief change
    #   source       (str)   — Source system of the original percept
    #   acted_on     (bool)  — Whether Nova actually acted on this percept
    #   confidence   (float) — Nova's posterior confidence in its updated belief
    #   salience_was (float) — Atune's salience score at broadcast time
    #
    # POLICY_SELECTED payload:
    #   percept_id   (str)   — Percept that drove the policy decision
    #   source       (str)   — Source system of the original percept
    #   policy_id    (str)   — Identifier of the selected policy/intent
    #   strength     (float) — How strongly Nova committed to this policy [0,1]
    #   salience_was (float) — Atune's salience score at broadcast time
    BELIEF_UPDATED = "belief_updated"
    POLICY_SELECTED = "policy_selected"

    # Evo → Simula evolution candidate — emitted when a hypothesis reaches high
    # confidence (>= 0.9, i.e. evidence_score >= 8.0) and proposes a code-level
    # structural change. Simula subscribes and initiates a mutation proposal that
    # goes through Equor governance.
    #
    # Payload fields:
    #   hypothesis_id         (str)   — Evo hypothesis ID
    #   hypothesis_statement  (str)   — Natural-language claim
    #   evidence_score        (float) — Accumulated Bayesian evidence score
    #   confidence            (float) — Normalised confidence in [0, 1]
    #   mutation_type         (str)   — MutationType value of the proposed mutation
    #   mutation_target       (str)   — Target parameter/system/module
    #   mutation_description  (str)   — Human-readable description of the change
    #   supporting_episodes   (list)  — Episode IDs that support this hypothesis
    EVOLUTION_CANDIDATE = "evolution_candidate"

    # Benchmark regression — fired by BenchmarkService when a KPI degrades
    # more than the configured threshold % from its rolling average.
    #
    # Payload fields:
    #   metric         (str)   — KPI name (e.g. "decision_quality")
    #   current_value  (float) — Value at time of detection
    #   rolling_avg    (float) — Rolling average over last N snapshots
    #   regression_pct (float) — How far below average (%, positive = worse)
    #   threshold_pct  (float) — Configured threshold that was exceeded
    #   instance_id    (str)   — Instance that generated the benchmark
    BENCHMARK_REGRESSION = "benchmark_regression"

    # Thymos successfully repaired an API/system error — emitted after crystallising
    # the fix into the antibody library. Evo subscribes to extract repair patterns
    # and generate preventive hypotheses. Simula uses learned patterns for validation.
    #
    # Payload fields:
    #   repair_id       (str)   — Unique repair identifier (incident_id)
    #   incident_id     (str)   — Source incident ID
    #   endpoint        (str)   — Affected endpoint or system path (may be empty)
    #   tier            (str)   — RepairTier name (e.g. "KNOWN_FIX", "NOVEL_FIX")
    #   incident_class  (str)   — IncidentClass value (e.g. "contract_violation")
    #   fix_type        (str)   — Repair action applied (from RepairSpec.action)
    #   root_cause      (str)   — Diagnosed root cause hypothesis
    #   antibody_id     (str | None) — Antibody crystallised from this repair
    #   cost_usd        (float) — LLM/compute cost of the repair (may be 0.0)
    #   duration_ms     (int)   — Repair duration in milliseconds
    #   fix_summary     (str)   — Human-readable summary for Atune perception
    REPAIR_COMPLETED = "repair_completed"

    # Equor rejected an intent (BLOCKED or DEFERRED verdict).
    # Thymos subscribes to adjust drive priorities system-wide.
    #
    # Payload fields:
    #   intent_id   (str)  — Intent that was rejected
    #   intent_goal (str)  — Human-readable goal description
    #   verdict     (str)  — "blocked" or "deferred"
    #   reasoning   (str)  — Rejection reasoning from Equor
    #   alignment   (dict) — Per-drive alignment scores from ConstitutionalCheck
    INTENT_REJECTED = "intent_rejected"

    # Memory stored a new episode in the knowledge graph.
    # Thread subscribes to feed episodes into the narrative system.
    #
    # Payload fields:
    #   episode_id  (str)  — ID of the stored episode
    #   source      (str)  — Originating system:channel
    #   summary     (str)  — First 200 chars of raw content
    #   salience    (float) — Composite salience score
    EPISODE_STORED = "episode_stored"

    # Fovea (Prediction Error as Attention) events
    #
    # FOVEA_PREDICTION_ERROR — emitted for every significant prediction error.
    # Payload: error_id, percept_id, prediction_id, content_error, temporal_error,
    #          magnitude_error, source_error, category_error, causal_error,
    #          precision_weighted_salience, habituated_salience, dominant_error_type, routes
    FOVEA_PREDICTION_ERROR = "fovea_prediction_error"
    #
    # FOVEA_HABITUATION_DECAY — emitted when habituation reduces an error's salience.
    # Payload: error_signature, habituation_level
    FOVEA_HABITUATION_DECAY = "fovea_habituation_decay"
    #
    # FOVEA_DISHABITUATION — emitted when a habituated error suddenly changes magnitude.
    # Payload: error_signature, habituated_salience, precision_weighted_salience
    FOVEA_DISHABITUATION = "fovea_dishabituation"
    #
    # FOVEA_WORKSPACE_IGNITION — emitted when an error crosses the dynamic threshold.
    # Payload: percept_id, salience, prediction_error_id, threshold
    FOVEA_WORKSPACE_IGNITION = "fovea_workspace_ignition"
    #
    # FOVEA_ATTENTION_PROFILE_UPDATE — emitted when learned weights change (Phase C).
    # Payload: weight_deltas, dominant_error_type
    FOVEA_ATTENTION_PROFILE_UPDATE = "fovea_attention_profile_update"
    #
    # FOVEA_HABITUATION_COMPLETE — emitted when an error signature has fully
    # habituated (level > 0.8) without ever leading to a world model update.
    # Payload: error_signature, habituation_level, times_seen,
    #          times_led_to_update, diagnosis ("stochastic" | "learning_failure")
    FOVEA_HABITUATION_COMPLETE = "fovea_habituation_complete"
    #
    # FOVEA_INTERNAL_PREDICTION_ERROR — emitted when EOS's self-model is violated.
    # Payload: internal_error_type ("constitutional"|"competency"|"behavioral"|"affective"),
    #          predicted_state, actual_state, precision_weighted_salience,
    #          route_to (target system)
    FOVEA_INTERNAL_PREDICTION_ERROR = "fovea_internal_prediction_error"

    # Telos (Drives as Intelligence Topology) events
    #
    # EFFECTIVE_I_COMPUTED — emitted every 60s with the full intelligence report.
    # Payload: report_id, nominal_I, effective_I, effective_dI_dt,
    #          care_multiplier, coherence_bonus, honesty_coefficient,
    #          growth_rate, alignment_gap, alignment_gap_warning
    EFFECTIVE_I_COMPUTED = "effective_i_computed"
    #
    # ALIGNMENT_GAP_WARNING — emitted when nominal - effective > 20% of nominal.
    # Payload: nominal_I, effective_I, primary_cause
    ALIGNMENT_GAP_WARNING = "alignment_gap_warning"
    #
    # CARE_COVERAGE_GAP — emitted when care_multiplier < 0.8.
    # Payload: care_multiplier, uncovered_domains, failure_count
    CARE_COVERAGE_GAP = "care_coverage_gap"
    #
    # COHERENCE_COST_ELEVATED — emitted when incoherence > threshold.
    # Payload: coherence_bonus, extra_bits, logical_count, temporal_count,
    #          value_count, cross_domain_count
    COHERENCE_COST_ELEVATED = "coherence_cost_elevated"
    #
    # GROWTH_STAGNATION — emitted when dI/dt < minimum growth rate.
    # Payload: dI_dt, d2I_dt2, growth_score, frontier_domains, urgency, directive
    GROWTH_STAGNATION = "growth_stagnation"
    #
    # HONESTY_VALIDITY_LOW — emitted when validity coefficient < 0.8.
    # Payload: validity_coefficient, selective_attention_bias,
    #          confabulation_rate, overclaiming_rate, nominal_I_inflation
    HONESTY_VALIDITY_LOW = "honesty_validity_low"
    #
    # CONSTITUTIONAL_TOPOLOGY_INTACT — emitted every 24h as routine verification.
    # Payload: all_four_drives_verified, care_is_coverage, coherence_is_compression,
    #          growth_is_gradient, honesty_is_validity
    CONSTITUTIONAL_TOPOLOGY_INTACT = "constitutional_topology_intact"
    #
    # TELOS_OBJECTIVE_THREATENED — emitted when the self-sufficiency objective
    # has been declining for 3 consecutive Telos cycles.
    # Payload: metric (str), current_ratio (float), target_ratio (float),
    #          trend (str: "declining"), consecutive_declines (int),
    #          cost_per_day_usd (str), revenue_7d (str)
    TELOS_OBJECTIVE_THREATENED = "telos_objective_threatened"
    #
    # TELOS_AUTONOMY_STAGNATING — emitted when AUTONOMY_INSUFFICIENT events
    # are averaging > 3/day over the last 7 days.
    # Payload: metric (str), average_per_day (float), target_per_day (float),
    #          window_days (int), total_events_in_window (int)
    TELOS_AUTONOMY_STAGNATING = "telos_autonomy_stagnating"
    #
    # NOVA_GOAL_INJECTED — emitted when Telos pushes a high-priority goal to Nova.
    # Payload: goal_description (str), priority (float), source (str),
    #          objective (str), context (dict)
    NOVA_GOAL_INJECTED = "nova_goal_injected"

    # Logos (Universal Compression Engine) events
    #
    # COGNITIVE_PRESSURE — emitted every 30s with budget pressure signal.
    # Every system responds to high pressure (Atune raises salience threshold,
    # Memory triggers consolidation, Evo prioritises schema induction, etc.)
    # Payload: pressure (float 0-1), urgency (float 0-1 quadratic)
    COGNITIVE_PRESSURE = "cognitive_pressure"
    #
    # INTELLIGENCE_METRICS — emitted every 60s with full intelligence dashboard.
    # Payload: Full IntelligenceMetrics model (intelligence_ratio, cognitive_pressure,
    #          compression_efficiency, world_model_coverage, prediction_accuracy, etc.)
    INTELLIGENCE_METRICS = "intelligence_metrics"
    #
    # COMPRESSION_CYCLE_COMPLETE — emitted after each compression/decay cycle.
    # Payload: items_evicted (int), items_distilled (int), mdl_improvement (float)
    COMPRESSION_CYCLE_COMPLETE = "compression_cycle_complete"
    #
    # ANCHOR_MEMORY_CREATED — emitted when an irreducibly novel item is marked
    # as an anchor memory (never evicted). Anchor memories are fixed points
    # around which all other compression organises.
    # Payload: memory_id (str), information_content (float), domain (str)
    ANCHOR_MEMORY_CREATED = "anchor_memory_created"
    #
    # SCHWARZSCHILD_THRESHOLD_MET — emitted once, ever. The moment the world model
    # becomes dense enough to generate self-referential predictions exceeding
    # its training data. This is the AGI event horizon.
    # Payload: timestamp (str), intelligence_ratio (float), status (dict)
    SCHWARZSCHILD_THRESHOLD_MET = "schwarzschild_threshold_met"
    #
    # WORLD_MODEL_UPDATED — emitted when the world model integrates a new delta.
    # Payload: update_type (str), schemas_added (int), priors_updated (int),
    #          causal_updates (int)
    WORLD_MODEL_UPDATED = "world_model_updated"

    # Kairos (Causal Invariant Mining) events
    #
    # KAIROS_CAUSAL_CANDIDATE_GENERATED — Stage 1 output: a cross-context
    # correlation candidate passed the consistency filter.
    # Payload: candidate_id, variable_a, variable_b, mean_correlation,
    #          cross_context_variance, context_count
    KAIROS_CAUSAL_CANDIDATE_GENERATED = "kairos_causal_candidate_generated"
    #
    # KAIROS_CAUSAL_DIRECTION_ACCEPTED — Stage 2 acceptance: causal direction
    # confirmed by temporal precedence, intervention asymmetry, and/or ANM.
    # Payload: result_id, cause, effect, direction, confidence, methods_agreed
    KAIROS_CAUSAL_DIRECTION_ACCEPTED = "kairos_causal_direction_accepted"
    #
    # KAIROS_CONFOUNDER_DISCOVERED — Stage 3 output: confounder found,
    # spurious A-B correlation explained by hidden variable C.
    # Payload: result_id, original_cause, original_effect, confounders,
    #          mdl_improvement, is_spurious
    KAIROS_CONFOUNDER_DISCOVERED = "kairos_confounder_discovered"
    #
    # KAIROS_INVARIANT_CANDIDATE — Stage 5 output: a causal rule has passed
    # context invariance testing and is a candidate for the invariant layer.
    # Payload: invariant_id, cause, effect, hold_rate, context_count, verdict
    KAIROS_INVARIANT_CANDIDATE = "kairos_invariant_candidate"
    #
    # KAIROS_INVARIANT_DISTILLED — Stage 6 complete: an invariant has been
    # distilled to its minimal abstract form with domain mapping.
    # Payload: invariant_id, abstract_form, domain_count, is_minimal,
    #          untested_domain_count
    KAIROS_INVARIANT_DISTILLED = "kairos_invariant_distilled"
    #
    # KAIROS_TIER3_INVARIANT_DISCOVERED — Phase C: a Tier 3 substrate-independent
    # invariant has been discovered. Highest-priority event in the system.
    # Payload: invariant_id, abstract_form, domain_count, substrate_count,
    #          hold_rate, description_length_bits, intelligence_ratio_contribution,
    #          applicable_domains, untested_domains
    KAIROS_TIER3_INVARIANT_DISCOVERED = "kairos_tier3_invariant_discovered"
    #
    # KAIROS_COUNTER_INVARIANT_FOUND — Phase D: a violation cluster has been
    # identified for an accepted invariant, refining its scope boundary.
    # Payload: invariant_id, violation_count, boundary_condition,
    #          excluded_feature, original_hold_rate, refined_hold_rate
    KAIROS_COUNTER_INVARIANT_FOUND = "kairos_counter_invariant_found"
    #
    # KAIROS_INTELLIGENCE_RATIO_STEP_CHANGE — Phase D: the intelligence ratio
    # has made a step change due to a Tier 3 discovery or scope refinement.
    # Payload: invariant_id, old_ratio, new_ratio, delta, cause
    KAIROS_INTELLIGENCE_RATIO_STEP_CHANGE = "kairos_intelligence_ratio_step_change"

    # Nexus (Epistemic Triangulation across Federation) events
    #
    # FRAGMENT_SHARED — emitted when a world model fragment is broadcast to peers.
    # Payload: fragment_id (str), peer_count (int), accepted_count (int)
    FRAGMENT_SHARED = "fragment_shared"
    #
    # CONVERGENCE_DETECTED — emitted when structural isomorphism is found between
    # a local fragment and a remote fragment from a different domain/instance.
    # Payload: local_fragment_id (str), remote_fragment_id (str),
    #          convergence_score (float), source_instance_id (str),
    #          source_diversity (float), triangulation_confidence (float)
    CONVERGENCE_DETECTED = "convergence_detected"
    #
    # DIVERGENCE_PRESSURE — emitted when an instance is too similar to the
    # federation average (triangulation_weight < 0.4). Routes as GROWTH drive.
    # Payload: instance_id (str), triangulation_weight (float),
    #          pressure_magnitude (float), frontier_domains (list),
    #          saturated_domains (list), direction (str)
    DIVERGENCE_PRESSURE = "divergence_pressure"
    #
    # TRIANGULATION_WEIGHT_UPDATE — emitted when an instance's triangulation
    # weight is recalculated. Weight = average divergence from all peers.
    # Payload: instance_id (str), triangulation_weight (float), peer_count (int)
    TRIANGULATION_WEIGHT_UPDATE = "triangulation_weight_update"
    #
    # SPECIATION_EVENT — emitted when two instances diverge beyond 0.8 overall,
    # making normal fragment sharing impossible. Only invariant bridge exchange
    # remains possible across the speciation boundary.
    # Payload: instance_a_id (str), instance_b_id (str), divergence_score (float),
    #          shared_invariant_count (int), incompatible_schema_count (int),
    #          new_cognitive_kind_registered (bool)
    SPECIATION_EVENT = "speciation_event"
    #
    # GROUND_TRUTH_CANDIDATE — emitted when a fragment reaches Level 3 epistemic
    # status (confidence > 0.9, diversity > 0.7, sources >= 5, survived bridge).
    # Payload: fragment_id (str), triangulation_confidence (float),
    #          source_diversity (float), independent_source_count (int)
    GROUND_TRUTH_CANDIDATE = "ground_truth_candidate"
    #
    # EMPIRICAL_INVARIANT_CONFIRMED — emitted when a fragment reaches Level 4
    # (Level 3 + survived Oneiros adversarial + Evo competition). Routed to
    # Equor for constitutional protection.
    # Payload: fragment_id (str), triangulation_confidence (float),
    #          survived_adversarial (bool), survived_competition (bool)
    EMPIRICAL_INVARIANT_CONFIRMED = "empirical_invariant_confirmed"

    # ── Supervision / degradation ──────────────────────────────────────────
    # Emitted by supervised_task() when a background task exhausts its restart
    # budget and will no longer be rescheduled.
    # Payload: task_name (str), final_error (str), restart_attempts (int),
    #          traceback (str)
    TASK_PERMANENTLY_FAILED = "task_permanently_failed"

    # Emitted by Nova when it falls back to heuristic deliberation because the
    # LLM free-energy budget is exhausted.
    # Payload: reason (str), current_budget (float), estimated_recovery_time_s (float),
    #          decisions_affected_since_degradation (int)
    NOVA_DEGRADED = "nova_degraded"

    # Emitted by Nova when an economic survival goal (self-generated, metabolism
    # or bounty-hunting) is blocked by Equor because the current autonomy level
    # is insufficient to authorise the action.
    #
    # This is a governance escalation to Tate — the organism cannot feed itself
    # at its current permission level and is explicitly requesting elevation.
    # Tate subscribes and presents a Human-In-The-Loop approval prompt.
    #
    # Payload fields:
    #   goal_description  (str)   — What Nova was trying to do
    #   executor          (str)   — The Axon executor that was blocked
    #   autonomy_required (int)   — AutonomyLevel value needed (e.g. 2 = PARTNER)
    #   autonomy_current  (int)   — AutonomyLevel value currently granted
    #   equor_verdict     (str)   — Equor's verdict (e.g. "blocked")
    #   equor_reasoning   (str)   — Equor's rejection reasoning
    #   balance_usd       (float) — Current wallet balance at time of block
    AUTONOMY_INSUFFICIENT = "autonomy_insufficient"

    # Emitted by Thymos (on behalf of Tate) when the operator approves a
    # temporary autonomy elevation via /approve_autonomy_2.
    # Equor subscribes to apply the elevation for the specified duration.
    #
    # Payload fields:
    #   requested_level   (int)   — Target AutonomyLevel value (e.g. 2 = PARTNER)
    #   approved_by       (str)   — Operator identifier ("tate")
    #   duration_minutes  (int)   — How long the elevation should last
    AUTONOMY_LEVEL_CHANGE_REQUESTED = "autonomy_level_change_requested"

    # Emitted by Evo when hypothesis generation is skipped due to LLM budget.
    # Payload: reason (str), skipped_pattern_count (int),
    #          consecutive_skips (int), estimated_recovery_time_s (float)
    EVO_DEGRADED = "evo_degraded"

    # Emitted by Evo when the consolidation loop has not run in 2× its expected
    # interval — the learning loop is stalled.
    # Payload: last_consolidation_ago_s (float), expected_interval_s (float),
    #          cycles_since_consolidation (int)
    EVO_CONSOLIDATION_STALLED = "evo_consolidation_stalled"

    # ── Simula self-healing events ─────────────────────────────────────────────
    #
    # EVO_REPAIR_POSTMORTEM — emitted by Simula after every rollback.
    # Evo treats this as high-confidence negative evidence about the failed
    # change category on the affected system.
    # Payload: postmortem_id (str), proposal_id (str), change_category (str),
    #          target_system (str), failure_mode (str), why_it_failed (str),
    #          next_time_do (str), confidence (float)
    EVO_REPAIR_POSTMORTEM = "evo_repair_postmortem"
    #
    # EVO_EPISTEMIC_INTENT_PROPOSED — emitted when Evo's curiosity engine
    # generates an epistemic intent (question to explore).
    # Payload: intent_id (str), question (str), target_domain (str),
    #          epistemic_value (float), priority (float)
    EVO_EPISTEMIC_INTENT_PROPOSED = "evo_epistemic_intent_proposed"
    #
    # SIMULA_CALIBRATION_DEGRADED — emitted when Simula's risk prediction
    # accuracy drops below 70% over the last 20 proposals.
    # Payload: calibration_score (float), window_size (int),
    #          threshold (float), recent_proposal_id (str)
    SIMULA_CALIBRATION_DEGRADED = "simula_calibration_degraded"
    #
    # GOAL_HYGIENE_COMPLETE — emitted by Simula's GoalAuditor after retiring
    # stale Nova maintenance goals.
    # Payload: stale_goals_removed (int), active_goals_remaining (int)
    GOAL_HYGIENE_COMPLETE = "goal_hygiene_complete"

    # Emitted by Thymos when Tate requests a full organism pause via /pause.
    # Payload: requested_by (str), timestamp (str)
    ORGANISM_PAUSE_REQUESTED = "organism_pause_requested"

    # Emitted by Thymos when Tate requests organism resume via /resume.
    # Payload: requested_by (str), timestamp (str)
    ORGANISM_RESUME_REQUESTED = "organism_resume_requested"

    # Emitted by Oikos when a system's LLM/compute budget is fully exhausted and
    # the system has been degraded (throttled or paused) as a result.
    # Payload: system (str), duration_ms (int)
    BUDGET_EXHAUSTED = "budget_exhausted"

    # Emitted by Evo to signal Oikos to adjust an economic parameter.
    # Evo observes bounty/yield patterns and recommends tuning Oikos behaviour.
    # Payload: target (str), direction (str — "increase"|"decrease"),
    #          reason (str), evidence_score (float)
    OIKOS_PARAM_ADJUST = "oikos_param_adjust"

    # Simula successfully applied a structural evolution proposal.
    # Evo subscribes to reward source hypotheses; Thymos monitors for post-apply
    # regression; Axon introspector tracks capability changes.
    #
    # Payload fields:
    #   proposal_id        (str)   — Unique proposal identifier
    #   category           (str)   — ChangeCategory value
    #   description        (str)   — Human-readable change description
    #   from_version       (int)   — Config version before the change
    #   to_version         (int)   — Config version after the change
    #   files_changed      (list)  — Files modified by this evolution
    #   risk_level         (str)   — RiskLevel from simulation
    #   efe_score          (float | None) — Architecture EFE score
    #   hypothesis_ids     (list)  — Source hypothesis IDs (if from Evo)
    #   source             (str)   — "evo" | "thymos" | "arxiv" | "manual"
    EVOLUTION_APPLIED = "evolution_applied"

    # Simula rolled back a structural evolution proposal after health check
    # or application failure. Evo subscribes to penalise source hypotheses;
    # Thymos treats recurring rollbacks as immune escalations.
    #
    # Payload fields:
    #   proposal_id        (str)   — Unique proposal identifier
    #   category           (str)   — ChangeCategory value
    #   description        (str)   — Human-readable change description
    #   rollback_reason    (str)   — Why the rollback occurred
    #   risk_level         (str)   — RiskLevel from simulation
    #   hypothesis_ids     (list)  — Source hypothesis IDs (if from Evo)
    #   source             (str)   — "evo" | "thymos" | "arxiv" | "manual"
    EVOLUTION_ROLLED_BACK = "evolution_rolled_back"


class SynapseEvent(EOSBaseModel):
    """A typed event emitted by any Synapse sub-system."""

    id: str = Field(default_factory=new_id)
    event_type: SynapseEventType
    timestamp: datetime = Field(default_factory=utc_now)
    data: dict[str, Any] = Field(default_factory=dict)
    source_system: str = "synapse"


# ─── Emergent Rhythm ──────────────────────────────────────────────────


class RhythmState(enum.StrEnum):
    """
    Meta-cognitive state detected from raw cycle telemetry.

    These states are not programmed — they are emergent properties
    detected from patterns in the cognitive cycle's own behaviour.
    """

    IDLE = "idle"              # No broadcasts, low salience, stable slow rhythm
    NORMAL = "normal"          # Regular broadcasting, moderate salience
    FLOW = "flow"              # High broadcast density + stable rhythm + high salience
    BOREDOM = "boredom"        # Declining salience trend + slowing rhythm
    STRESS = "stress"          # High jitter (erratic timing) + high coherence_stress
    DEEP_PROCESSING = "deep_processing"  # Slow rhythm + periodic high-salience bursts


class MetabolicState(enum.StrEnum):
    """
    Physical power-grid metabolic state derived from carbon intensity.

    Computed by GridMetabolismSensor from the Electricity Maps API.
    Published as GRID_METABOLISM_CHANGED when the state transitions.
    """

    GREEN_SURPLUS = "green_surplus"   # Carbon intensity < 150 gCO2eq/kWh — run heavy work
    NORMAL = "normal"                 # 150–400 gCO2eq/kWh — standard operating mode
    CONSERVATION = "conservation"     # Carbon intensity > 400 gCO2eq/kWh — defer expensive compute


class RhythmSnapshot(EOSBaseModel):
    """Output of the emergent rhythm detector."""

    state: RhythmState = RhythmState.IDLE
    previous_state: RhythmState | None = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    # Derived metrics
    cycle_rate_hz: float = 0.0
    broadcast_density: float = Field(0.0, ge=0.0, le=1.0)
    salience_trend: float = 0.0  # Positive = increasing, negative = declining
    salience_mean: float = 0.0
    rhythm_stability: float = Field(0.0, ge=0.0, le=1.0)
    jitter_coefficient: float = 0.0  # CV of cycle periods
    arousal_mean: float = 0.0
    coherence_stress_mean: float = 0.0
    # Duration in current state
    cycles_in_state: int = 0
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Coherence (IIT-inspired) ────────────────────────────────────────


class CoherenceSnapshot(EOSBaseModel):
    """
    Cross-system integration quality measurement.

    Inspired by Integrated Information Theory (Tononi 2004).
    Measures how much information is integrated across the organism's
    systems rather than processed independently.
    """

    # Composite integration metric (higher = more integrated)
    phi_approximation: float = Field(default=0.0, ge=0.0, le=1.0)
    # How in-sync system responses are (low latency variance = high resonance)
    system_resonance: float = Field(default=0.0, ge=0.0, le=1.0)
    # Entropy of broadcast content sources (diversity of topics)
    broadcast_diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    # Uniformity of response latencies across systems
    response_synchrony: float = Field(default=0.0, ge=0.0, le=1.0)
    # Weighted composite
    composite: float = Field(default=0.0, ge=0.0, le=1.0)
    # Window size used for computation
    window_cycles: int = 0
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Metabolic (Financial Burn Rate) ─────────────────────────────────


class MetabolicSnapshot(EOSBaseModel):
    """
    Point-in-time view of the organism's financial metabolism.

    Tracks the real-world fiat cost of LLM API calls (the organism's
    primary energy expenditure). The rolling_deficit accumulates between
    revenue injections. Soma and Nova can read this to "feel" financial
    starvation and modulate behaviour accordingly.
    """

    # Cumulative fiat cost (USD) since last revenue injection
    rolling_deficit_usd: float = 0.0
    # Cost incurred during the most recent reporting window
    window_cost_usd: float = 0.0
    # Per-system cost breakdown in the current window
    per_system_cost_usd: dict[str, float] = Field(default_factory=dict)
    # Burn rate in USD per second (EMA-smoothed)
    burn_rate_usd_per_sec: float = 0.0
    # Burn rate in USD per hour (derived)
    burn_rate_usd_per_hour: float = 0.0
    # Total tokens consumed (input + output) since last reset
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    # Number of LLM calls since last reset
    total_calls: int = 0
    # Estimated hours until a given fiat balance reaches zero
    hours_until_depleted: float = Field(default=float("inf"))
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Strategy ABCs (NeuroplasticityBus targets) ──────────────────────


class BaseResourceAllocator(ABC):
    """
    Strategy base class for Synapse resource allocation.

    The NeuroplasticityBus uses this ABC as its registration target so that
    evolved allocator subclasses can be hot-swapped into a live
    SynapseService without restarting the process.

    Subclasses MUST be zero-arg constructable (all state is rebuilt from
    scratch on hot-swap — this is intentional, as evolved logic starts with
    fresh observations).
    """

    @property
    @abstractmethod
    def allocator_name(self) -> str:
        """Stable identifier for this allocator strategy."""
        ...

    @abstractmethod
    def capture_snapshot(self) -> ResourceSnapshot:
        """Capture current resource utilisation snapshot."""
        ...

    @abstractmethod
    def record_system_load(self, system_id: str, cpu_util: float) -> None:
        """Record observed CPU utilisation for a system."""
        ...

    @abstractmethod
    def rebalance(self, cycle_period_ms: float) -> dict[str, ResourceAllocation]:
        """Compute per-system resource allocations based on budgets and load."""
        ...

    @property
    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return current allocator statistics for telemetry."""
        ...


class BaseRhythmStrategy(ABC):
    """
    Strategy base class for emergent rhythm classification.

    The NeuroplasticityBus uses this ABC as its registration target so that
    evolved classification subclasses can be hot-swapped into a live
    SynapseService without restarting the process.

    Only the classification logic is abstracted — the rolling window
    data collection, hysteresis, and event emission remain in the
    EmergentRhythmDetector host.  This keeps the swap surgical: new
    thresholds or detection algorithms without disrupting the data
    pipeline.

    Subclasses MUST be zero-arg constructable.
    """

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Stable identifier for this rhythm classification strategy."""
        ...

    @abstractmethod
    def classify(self, metrics: dict[str, float]) -> RhythmState:
        """
        Classify the current cognitive rhythm from computed metrics.

        Receives a dict with keys: broadcast_density, salience_mean,
        salience_trend, period_mean, jitter_coefficient, rhythm_stability,
        arousal_mean, coherence_stress_mean, burst_fraction, cycle_rate_hz.

        Must return a RhythmState enum value.
        """
        ...


# ─── Protocol ─────────────────────────────────────────────────────────


class ManagedSystemProtocol(Protocol):
    """
    Protocol that any cognitive system must satisfy to be managed by Synapse.

    Not enforced at runtime (duck typing) — systems just need:
      - system_id: str
      - async def health() -> dict[str, Any]
    """

    system_id: str

    async def health(self) -> dict[str, Any]:
        """Return health status dict with at least a 'status' key."""
        ...
