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


class SystemLoad(EOSBaseModel):
    """
    Current system resource utilisation — passed to Atune each theta tick.

    Formerly in systems.fovea.types; moved here (Spec 09 §3) so CognitiveClock
    can import it without a cross-system dependency on Fovea.
    """

    cpu_utilisation: float = Field(ge=0.0, le=1.0, default=0.0)
    memory_utilisation: float = Field(ge=0.0, le=1.0, default=0.0)
    queue_depth: int = 0


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

    starvation_level: str = "nominal"
    """Current metabolic starvation level from Oikos (StarvationLevel.value).
    Relayed on every SomaTick so downstream systems can read it without polling."""


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
    # Emitted on every status transition (Spec 09 §18 M8):
    # STARTING→HEALTHY, HEALTHY→DEGRADED, HEALTHY→OVERLOADED, any→FAILED
    # Payload: system_id (str), old_status (str), new_status (str)
    SYSTEM_HEALTH_CHANGED = "system_health_changed"

    # Safe mode
    SAFE_MODE_ENTERED = "safe_mode_entered"
    SAFE_MODE_EXITED = "safe_mode_exited"

    # Clock
    CLOCK_STARTED = "clock_started"
    CLOCK_STOPPED = "clock_stopped"
    CLOCK_PAUSED = "clock_paused"
    CLOCK_RESUMED = "clock_resumed"
    CLOCK_OVERRUN = "clock_overrun"

    # Cognitive cycle — pre/post tick signals (Spec 09 §18 integration surface)
    THETA_CYCLE_START = "theta_cycle_start"      # Before Soma/Atune run — cycle_number, period_ms, arousal
    THETA_CYCLE_OVERRUN = "theta_cycle_overrun"  # elapsed > budget — cycle_number, elapsed_ms, budget_ms
    CYCLE_COMPLETED = "cycle_completed"
    # Somatic tick — emitted every cycle where Soma ran successfully
    SOMA_TICK = "soma_tick"

    # Interoceptive percept — Soma broadcasting an internal sensation
    # through the Global Workspace when analysis thresholds are exceeded.
    # Payload: InteroceptivePercept serialised to dict.
    INTEROCEPTIVE_PERCEPT = "interoceptive_percept"

    # Allostatic signal — Soma's primary output, emitted every theta cycle (Spec 08 §15.1,
    # Spec 16 §XVIII). Allows any system to subscribe without a direct Soma reference.
    # Payload: urgency (float), dominant_error (str), precision_weights (dict[str, float]),
    #          nearest_attractor (str|None), trajectory_heading (str), cycle_number (int),
    #          energy (float), arousal (float), valence (float), coherence (float).
    ALLOSTATIC_SIGNAL = "allostatic_signal"

    # Urgency critical — emitted when urgency exceeds 0.85 (Spec 16 §XVIII).
    # High-priority allostatic alert for systems that don't subscribe to ALLOSTATIC_SIGNAL.
    # Payload: urgency (float), dominant_error (str), recommended_action (str), cycle (int).
    SOMA_URGENCY_CRITICAL = "soma_urgency_critical"

    # Somatic drive vector — Soma → Telos mapping of 9D felt state to 4D drives.
    # Emitted every 10 cycles. Payload: {coherence_drive, care_drive, growth_drive, honesty_drive}.
    SOMATIC_DRIVE_VECTOR = "somatic_drive_vector"

    # Rhythm (emergent)
    RHYTHM_STATE_CHANGED = "rhythm_state_changed"

    # Coherence
    COHERENCE_SHIFT = "coherence_shift"

    # Resources
    RESOURCE_REBALANCED = "resource_rebalanced"
    RESOURCE_REBALANCE = "resource_rebalance"  # Spec 09 §18: every 100 cycles; allocations dict
    RESOURCE_PRESSURE = "resource_pressure"
    # Grid-aware clock throttling (Spec 09 §3.1 / §18)
    CONSERVATION_MODE_ENTERED = "conservation_mode_entered"  # payload: trigger, new_period_ms
    CONSERVATION_MODE_EXITED = "conservation_mode_exited"    # payload: restored_period_ms

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

    # Skia periodic heartbeat — emitted by standalone worker to prove liveness.
    SKIA_HEARTBEAT = "skia_heartbeat"

    # Skia restoration lifecycle — finer-grained than TRIGGERED/COMPLETED.
    SKIA_RESTORATION_STARTED = "skia_restoration_started"
    SKIA_RESTORATION_COMPLETE = "skia_restoration_complete"

    # Organism spawned — new instance birthed from restoration with heritable variation.
    # Payload: instance_id, parent_instance_id, generation, mutation_delta, lineage_depth
    ORGANISM_SPAWNED = "organism_spawned"

    # Skia dry-run restoration completed — simulation without committing.
    # Payload: instance_id, predicted_outcome, duration_ms
    SKIA_DRY_RUN_COMPLETE = "skia_dry_run_complete"

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
    #
    # BENCHMARK_RE_PROGRESS — emitted when llm_dependency improves >5% cycle-over-cycle.
    # Nova can subscribe to adjust RE routing confidence.
    # Payload fields:
    #   current         (float) — Current llm_dependency value
    #   previous        (float) — Previous cycle llm_dependency value
    #   improvement_pct (float) — Percentage improvement (positive = better)
    #   instance_id     (str)   — Instance that generated the benchmark
    BENCHMARK_RE_PROGRESS = "benchmark_re_progress"
    #
    # BENCHMARK_RECOVERY — emitted when a previously regressed metric recovers
    # above its threshold. Thymos and Evo need recovery signals to close loops.
    # Payload fields:
    #   metric             (str)   — KPI name that recovered
    #   previous_value     (float) — Value when regression was first detected
    #   recovered_value    (float) — Current recovered value
    #   duration_regressed (float) — Seconds the metric was in regression
    #   instance_id        (str)   — Instance that generated the benchmark
    BENCHMARK_RECOVERY = "benchmark_recovery"

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

    # Memory organizational closure events (Phase 2 — Memory Spec 01)
    BELIEF_CONSOLIDATED = "belief_consolidated"
    SELF_AFFECT_UPDATED = "self_affect_updated"
    MEMORY_PRESSURE = "memory_pressure"
    SELF_STATE_DRIFTED = "self_state_drifted"

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
    # TELOS_VITALITY_SIGNAL — emitted each Telos cycle with vitality-relevant
    # drive topology data. VitalityCoordinator subscribes for brain-death
    # threshold logic (effective_I < 0.01 for 7d). Separate from SOMA_VITALITY_SIGNAL
    # — Telos owns the intelligence-measurement axis of vitality.
    # Payload: source ("telos"), effective_I (float), alignment_gap_severity (float),
    #          growth_stagnation_flag (bool), honesty_coefficient (float),
    #          care_multiplier (float)
    TELOS_VITALITY_SIGNAL = "telos_vitality_signal"
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
    # CHILD_DISCOVERY_PROPAGATED — emitted when a child's novel discovery
    # (via RESEARCH_MUTATOR or Evo) is merged into the parent's genome
    # through horizontal gene transfer.
    # Payload: child_instance_id (str), discovery_type (str),
    #          discovery_payload (dict), parent_segment_updated (str)
    CHILD_DISCOVERY_PROPAGATED = "child_discovery_propagated"
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
    #
    # EVO_HYPOTHESIS_CREATED — emitted when a new hypothesis is generated.
    # Payload: hypothesis_id (str), category (str), statement (str),
    #          source_detector (str), novelty_score (float)
    EVO_HYPOTHESIS_CREATED = "evo_hypothesis_created"
    #
    # EVO_HYPOTHESIS_CONFIRMED — emitted when a hypothesis passes validation
    # and reaches SUPPORTED status.
    # Payload: hypothesis_id (str), category (str), statement (str),
    #          evidence_score (float), supporting_count (int)
    EVO_HYPOTHESIS_CONFIRMED = "evo_hypothesis_confirmed"
    #
    # EVO_HYPOTHESIS_REFUTED — emitted when a hypothesis is refuted by evidence.
    # Payload: hypothesis_id (str), category (str), statement (str),
    #          evidence_score (float), contradicting_count (int)
    EVO_HYPOTHESIS_REFUTED = "evo_hypothesis_refuted"
    #
    # EVO_CONSOLIDATION_COMPLETE — emitted after a consolidation cycle finishes.
    # Payload: consolidation_number (int), duration_ms (int),
    #          hypotheses_integrated (int), schemas_induced (int),
    #          parameters_adjusted (int)
    EVO_CONSOLIDATION_COMPLETE = "evo_consolidation_complete"
    #
    # EVO_CAPABILITY_EMERGED — emitted when a genuinely new capability is detected.
    # Payload: capability_name (str), source_hypotheses (list[str]),
    #          novelty_score (float), domain (str)
    EVO_CAPABILITY_EMERGED = "evo_capability_emerged"

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

    # Emitted by Evo at Phase 7 (Drift Data Feed) of each consolidation cycle.
    # Equor subscribes to update its drift model with Evo's latest self-model stats.
    # Payload: success_rate (float), mean_alignment (float),
    #          capability_scores (dict[str, dict]), regret (dict),
    #          consolidation_number (int)
    EVO_DRIFT_DATA = "evo_drift_data"

    # ── Thymos Feedback Channels (Interconnectedness Audit) ──────────

    # AXON_SHIELD_REJECTION — emitted by Axon when TransactionShield.evaluate()
    # blocks a financial transaction. Thymos subscribes to create a real-time
    # incident so repair pipeline can detect root causes (bad params, blacklisted
    # addresses, slippage misconfiguration) instead of only seeing post-mortem.
    #
    # Payload fields:
    #   execution_id     (str)   — Pipeline execution ID
    #   executor         (str)   — Executor type (defi_yield, wallet_transfer)
    #   intent_id        (str)   — Intent that was being executed
    #   rejection_reason (str)   — Why the shield blocked the transaction
    #   check_type       (str)   — Which check failed (blacklist, slippage, gas_roi, mev)
    #   params           (dict)  — Sanitised transaction parameters
    AXON_SHIELD_REJECTION = "axon_shield_rejection"

    # ATUNE_REPAIR_VALIDATION — emitted by Atune after observing whether a
    # repair actually improved the workspace signal. Closes the one-way
    # Thymos→Atune incident channel into a bidirectional loop so the organism
    # confirms (or denies) that a repair helped.
    #
    # Payload fields:
    #   incident_id       (str)   — Original Thymos incident ID
    #   repair_effective  (bool)  — Whether Atune's salience for the error dropped
    #   salience_before   (float) — Error salience before repair
    #   salience_after    (float) — Error salience after repair
    #   cycles_observed   (int)   — How many cycles Atune monitored post-repair
    ATUNE_REPAIR_VALIDATION = "atune_repair_validation"

    # EVO_HYPOTHESIS_QUALITY — emitted by Evo when a repair-derived hypothesis
    # is evaluated for generalisability. Tells Thymos whether repair patterns
    # actually transferred to novel incidents or stayed narrow.
    #
    # Payload fields:
    #   hypothesis_id     (str)   — Evo hypothesis ID
    #   repair_source_id  (str)   — Incident ID that spawned the hypothesis
    #   quality_score     (float) — Generalisation score [0, 1]
    #   applications      (int)   — How many distinct incidents this pattern matched
    #   confidence        (float) — Thompson posterior confidence
    EVO_HYPOTHESIS_QUALITY = "evo_hypothesis_quality"

    # NOVA_BELIEF_STABILISED — emitted by Nova when beliefs affected by a
    # Thymos repair have re-converged. Confirms downstream cognitive stability
    # after immune intervention, closing the Thymos→Nova one-way channel.
    #
    # Payload fields:
    #   incident_id       (str)   — Thymos incident that triggered the repair
    #   goal_id           (str)   — Nova goal that was injected for the repair
    #   beliefs_affected  (int)   — Number of beliefs that were destabilised
    #   convergence_time_ms (int) — Time for beliefs to re-converge
    #   stable            (bool)  — Whether beliefs fully converged
    NOVA_BELIEF_STABILISED = "nova_belief_stabilised"

    # TIER5_AUTO_APPROVAL — emitted by Thymos when an Equor pre-check confirms
    # that a Tier 5 repair is constitutionally safe and can proceed without
    # human (Telegram) approval. Replaces the Telegram dependency for escalations
    # that pass constitutional review.
    #
    # Payload fields:
    #   incident_id       (str)   — Incident being escalated
    #   repair_action     (str)   — What the repair would do
    #   equor_confidence  (float) — Equor's confidence in safety
    #   drive_alignment   (dict)  — Per-drive alignment scores
    #   auto_approved     (bool)  — Whether auto-approval was granted
    TIER5_AUTO_APPROVAL = "tier5_auto_approval"

    # ── Kairos Feedback Loop Events ────────────────────────────────────────
    #
    # KAIROS_VALIDATED_CAUSAL_STRUCTURE — emitted when Kairos finalizes a
    # causal invariant. Evo subscribes to boost Thompson sampler confidence
    # for hypothesis variants exploring this causal pattern.
    # Payload: invariant_id (str), cause (str), effect (str),
    #          hold_rate (float), tier (int), domain_count (int),
    #          hypothesis_pattern (str)
    KAIROS_VALIDATED_CAUSAL_STRUCTURE = "kairos_validated_causal_structure"
    #
    # KAIROS_SPURIOUS_HYPOTHESIS_CLASS — emitted when Kairos confounder
    # analysis proves a hypothesis class is spurious. Evo subscribes to
    # down-weight that hypothesis class in Thompson sampling.
    # Payload: confounded_cause (str), confounded_effect (str),
    #          confounders (list[str]), mdl_improvement (float),
    #          hypothesis_class (str)
    KAIROS_SPURIOUS_HYPOTHESIS_CLASS = "kairos_spurious_hypothesis_class"
    #
    # KAIROS_INVARIANT_ABSORPTION_REQUESTED — emitted when Kairos finalizes
    # an invariant with high hold_rate. Fovea subscribes to update its
    # causal prediction model, reducing redundant error decomposition.
    # Payload: invariant_id (str), cause (str), effect (str),
    #          hold_rate (float), tier (int), abstract_form (str)
    KAIROS_INVARIANT_ABSORPTION_REQUESTED = "kairos_invariant_absorption_requested"
    #
    # KAIROS_CAUSAL_NOVELTY_DETECTED — emitted when Kairos detects a
    # structurally novel causal pattern (bidirectional, modulated, feedback loop).
    # Oneiros REM targets this for creative synthesis; Evo explores variants.
    # Payload: invariant_id (str), novelty_type (str), structure (dict),
    #          domains (list[str]), abstract_form (str)
    KAIROS_CAUSAL_NOVELTY_DETECTED = "kairos_causal_novelty_detected"
    #
    # KAIROS_HEALTH_DEGRADED — emitted when Kairos self-diagnosis detects
    # corruption, stall, or model instability. Thymos subscribes to classify
    # and potentially trigger repair.
    # Payload: degradation_type (str), severity (str), details (dict),
    #          metrics (dict)
    KAIROS_HEALTH_DEGRADED = "kairos_health_degraded"
    #
    # KAIROS_VIOLATION_ESCALATION — emitted when counter-invariant violations
    # need Thymos classification (anomaly vs corruption vs regime-shift).
    # Payload: invariant_id (str), violation_count (int),
    #          violation_rate (float), severity (str)
    KAIROS_VIOLATION_ESCALATION = "kairos_violation_escalation"

    # FEDERATION_INVARIANT_RECEIVED — emitted by Nexus/Federation when
    # a peer instance shares a causal invariant. Kairos validates against
    # local observations before merging.
    # Payload: invariant_id (str), abstract_form (str), tier (int),
    #          hold_rate (float), source_instance_id (str), domains (list[str])
    FEDERATION_INVARIANT_RECEIVED = "federation_invariant_received"

    # KAIROS_INVARIANT_CONTRADICTED — emitted when a federation invariant
    # is tested locally and fails counter-invariant validation.
    # Payload: invariant_id (str), local_hold_rate (float),
    #          violation_count (int), source_instance_id (str)
    KAIROS_INVARIANT_CONTRADICTED = "kairos_invariant_contradicted"

    # CONSTITUTIONAL_REVIEW_REQUESTED — emitted when a system needs Equor
    # to review a decision before acceptance (e.g. Tier 3 invariant).
    # Payload: review_type (str), reason (str), + system-specific fields
    CONSTITUTIONAL_REVIEW_REQUESTED = "constitutional_review_requested"

    # NARRATIVE_MILESTONE — emitted when a system achieves a significant
    # milestone that Thread should record in the organism's narrative.
    # Payload: milestone_type (str), title (str), description (str),
    #          significance (str), + system-specific fields
    NARRATIVE_MILESTONE = "narrative_milestone"

    # ── RE Training Pipeline ───────────────────────────────────────────────
    #
    # RE_TRAINING_EXAMPLE — emitted by any system after an LLM inference call.
    # Payload: RETrainingExample fields (source_system, instruction, output, etc.)
    RE_TRAINING_EXAMPLE = "re_training_example"
    #
    # RE_TRAINING_BATCH — bulk emission of training examples at cycle end.
    # Payload: RETrainingBatch fields (examples list, source_system)
    RE_TRAINING_BATCH = "re_training_batch"

    # ── Metabolic Gate ─────────────────────────────────────────────────────
    #
    # METABOLIC_GATE_CHECK — emitted when a system requests metabolic permission.
    # Payload: system_id (str), estimated_cost_usd (str), operation (str)
    METABOLIC_GATE_CHECK = "metabolic_gate_check"
    #
    # METABOLIC_GATE_RESPONSE — Oikos response to a gate check.
    # Payload: MetabolicPermission fields (granted, reason, starvation_level, etc.)
    METABOLIC_GATE_RESPONSE = "metabolic_gate_response"

    # ── Evolutionary Observables ───────────────────────────────────────────
    #
    # EVOLUTIONARY_OBSERVABLE — any system emits a discrete evolutionary event.
    # Payload: EvolutionaryObservable fields (source_system, observable_type, etc.)
    EVOLUTIONARY_OBSERVABLE = "evolutionary_observable"
    #
    # BEDAU_PACKARD_SNAPSHOT — Benchmarks emits population-level activity stats.
    # Payload: BedauPackardStats fields (total_activity, diversity_index, etc.)
    BEDAU_PACKARD_SNAPSHOT = "bedau_packard_snapshot"

    # ── Vitality ───────────────────────────────────────────────────────────
    #
    # VITALITY_REPORT — periodic viability assessment against fatal thresholds.
    # Payload: VitalityReport fields (instance_id, thresholds, overall_viable, etc.)
    VITALITY_REPORT = "vitality_report"
    #
    # VITALITY_FATAL — a fatal threshold has been irreversibly breached.
    # Payload: instance_id (str), reason (str), threshold_name (str),
    #          current_value (float), threshold_value (float)
    VITALITY_FATAL = "vitality_fatal"
    #
    # VITALITY_RESTORED — a fatal threshold recovered during the warning window.
    # Payload: instance_id (str), threshold_name (str), recovered_value (float)
    VITALITY_RESTORED = "vitality_restored"
    #
    # ORGANISM_DIED — the organism has completed its death sequence.
    # Payload: instance_id (str), cause (str), final_report (dict),
    #          genome_id (str), snapshot_cid (str)
    ORGANISM_DIED = "organism_died"
    #
    # ORGANISM_RESURRECTED — the organism was externally revived after death.
    # Payload: instance_id (str), trigger (str), runway_days (float)
    ORGANISM_RESURRECTED = "organism_resurrected"

    # ── Closure Loop Events ────────────────────────────────────────────────
    #
    # CONSTITUTIONAL_DRIFT_DETECTED — Equor detects alignment gap widening
    # or drive weight modification attempts. Thymos subscribes.
    # Payload: drift_type (str), alignment_gap (float), primary_cause (str),
    #          severity (str)
    CONSTITUTIONAL_DRIFT_DETECTED = "constitutional_drift_detected"
    #
    # MOTOR_DEGRADATION_DETECTED — Axon detects executor failures or latency.
    # Nova subscribes and replans.
    # Payload: executor (str), failure_count (int), latency_ms (int),
    #          degradation_type (str)
    MOTOR_DEGRADATION_DETECTED = "motor_degradation_detected"
    #
    # SIMULA_ROLLBACK_PENALTY — Simula rollback carries genuine metabolic cost.
    # Oikos subscribes and deducts penalty from liquid balance.
    # Payload: proposal_id (str), penalty_usd (str), rollback_reason (str)
    SIMULA_ROLLBACK_PENALTY = "simula_rollback_penalty"
    #
    # IMMUNE_PATTERN_ADVISORY — Thymos shares crystallised antibody patterns
    # with Simula to prevent re-introducing known-bad mutations.
    # Payload: antibody_id (str), pattern (str), incident_class (str),
    #          confidence (float)
    IMMUNE_PATTERN_ADVISORY = "immune_pattern_advisory"
    #
    # SOMATIC_MODULATION_SIGNAL — Soma felt-sense modulates downstream systems.
    # Nova, Voxis, and Equor subscribe.
    # Payload: arousal (float), fatigue (float), metabolic_stress (float),
    #          modulation_targets (list[str]), recommended_urgency (float)
    SOMATIC_MODULATION_SIGNAL = "somatic_modulation_signal"
    #
    # FITNESS_OBSERVABLE_BATCH — Evo emits learning signals for Benchmarks.
    # Payload: observables (list[dict]), instance_id (str), generation (int)
    FITNESS_OBSERVABLE_BATCH = "fitness_observable_batch"
    #
    # SOMA_VITALITY_SIGNAL — Soma emits vitality-relevant interoceptive state
    # every cycle. VitalityCoordinator subscribes and incorporates into
    # assess_vitality(). Fire-and-forget, never blocking.
    # Payload: urgency_scalar (float), allostatic_error (float),
    #          coherence_stress (float), cycle (int)
    SOMA_VITALITY_SIGNAL = "soma_vitality_signal"
    #
    # SOMA_ALLOSTATIC_REPORT — Soma emits allostatic efficiency metrics every
    # N cycles (default 50). Benchmarks subscribes for organism health KPIs.
    # Payload: mean_urgency (float), urgency_frequency (float),
    #          setpoint_deviation (float), developmental_stage (str), cycle (int)
    SOMA_ALLOSTATIC_REPORT = "soma_allostatic_report"

    # ── Equor Constitutional Events ────────────────────────────────────────
    #
    # EQUOR_REVIEW_STARTED — emitted when constitutional_review() begins.
    # Payload: intent_id (str), goal_summary (str), autonomy_required (int)
    EQUOR_REVIEW_STARTED = "equor_review_started"
    #
    # EQUOR_REVIEW_COMPLETED — emitted when review finishes.
    # Payload: intent_id (str), verdict (str), reasoning (str),
    #          latency_ms (int), composite_alignment (float)
    EQUOR_REVIEW_COMPLETED = "equor_review_completed"
    #
    # EQUOR_DRIFT_WARNING — DriftTracker detected drift below fatal threshold.
    # Payload: drift_severity (float), drift_direction (str),
    #          mean_alignment (dict), response_action (str)
    EQUOR_DRIFT_WARNING = "equor_drift_warning"
    #
    # EQUOR_DRIVE_WEIGHTS_UPDATED — after any drive weight modification.
    # Payload: proposal_id (str), old_weights (dict), new_weights (dict),
    #          actor (str)
    EQUOR_DRIVE_WEIGHTS_UPDATED = "equor_drive_weights_updated"
    #
    # EQUOR_ALIGNMENT_SCORE — periodic overall alignment score for Benchmarks.
    # Payload: mean_alignment (dict), composite (float), total_reviews (int),
    #          window_size (int)
    EQUOR_ALIGNMENT_SCORE = "equor_alignment_score"
    #
    # EQUOR_FAST_PATH_HIT — fast-path cache returned verdict without full review.
    # Payload: intent_id (str), verdict (str), latency_ms (int)
    EQUOR_FAST_PATH_HIT = "equor_fast_path_hit"
    #
    # EQUOR_ESCALATED_TO_HUMAN — review requires human operator input (HITL).
    # Payload: intent_id (str), auth_id (str), goal_summary (str),
    #          autonomy_required (int)
    EQUOR_ESCALATED_TO_HUMAN = "equor_escalated_to_human"
    #
    # EQUOR_DEFERRED — verdict is DEFERRED for later resolution.
    # Payload: intent_id (str), reasoning (str), deferred_until (str|None)
    EQUOR_DEFERRED = "equor_deferred"
    #
    # EQUOR_HITL_APPROVED — human operator authorised a suspended intent via SMS code.
    # Axon subscribes to this and executes the released intent without a new Equor review.
    # Payload: intent_id (str), intent_json (str), auth_id (str), equor_check_json (str)
    EQUOR_HITL_APPROVED = "equor_hitl_approved"
    #
    # EQUOR_AUTONOMY_PROMOTED — autonomy level increased (governance approval required).
    # Payload: old_level (int), new_level (int), decision_count (int)
    EQUOR_AUTONOMY_PROMOTED = "equor_autonomy_promoted"
    #
    # EQUOR_AUTONOMY_DEMOTED — autonomy level decreased due to drift or violation.
    # Payload: old_level (int), new_level (int), reason (str)
    EQUOR_AUTONOMY_DEMOTED = "equor_autonomy_demoted"
    #
    # EQUOR_SAFE_MODE_ENTERED — Equor entered safe mode (Neo4j unavailable or critical error).
    # Only Level 1 (Advisor) actions permitted until safe mode exits.
    # Payload: reason (str), critical_error_count (int)
    EQUOR_SAFE_MODE_ENTERED = "equor_safe_mode_entered"
    #
    # EQUOR_CONSTITUTIONAL_SNAPSHOT — periodic full state snapshot for Benchmarks.
    # Payload: drive_weights (dict), invariant_count (int),
    #          autonomy_level (int), drift_severity (float),
    #          total_reviews (int), safe_mode (bool)
    EQUOR_CONSTITUTIONAL_SNAPSHOT = "equor_constitutional_snapshot"

    # ── Oneiros Sleep Events ───────────────────────────────────────────────
    #
    # ONEIROS_GENOME_READY — Oneiros has prepared a genome segment for Mitosis.
    # Payload: OrganGenomeSegment fields (system_id, payload, payload_hash, etc.)
    ONEIROS_GENOME_READY = "oneiros_genome_ready"
    #
    # ONEIROS_SLEEP_CYCLE_SUMMARY — Oneiros emits a sleep cycle summary for
    # Benchmarks to track cognitive development rate.
    # Payload: consolidation_count (int), dreams_generated (int),
    #          beliefs_compressed (int), schemas_created (int),
    #          intelligence_improvement (float), cycle_id (str)
    ONEIROS_SLEEP_CYCLE_SUMMARY = "oneiros_sleep_cycle_summary"

    # ONEIROS_THREAT_SCENARIO — Oneiros ThreatSimulator produced a simulated failure
    # scenario during REM. Thymos subscribes to pre-emptively generate antibodies.
    # Payload: scenario_id (str), domain (str), scenario_description (str),
    #          response_plan (str), severity (str), source_type (str)
    ONEIROS_THREAT_SCENARIO = "oneiros_threat_scenario"

    # ── Federation Population Dynamics ─────────────────────────────────
    #
    # FEDERATION_LINK_ESTABLISHED — after successful handshake completion.
    # Payload: link_id (str), remote_instance_id (str), remote_name (str),
    #          elapsed_ms (int)
    FEDERATION_LINK_ESTABLISHED = "federation_link_established"
    #
    # FEDERATION_LINK_DROPPED — on link teardown (withdrawal or starvation).
    # Payload: link_id (str), remote_instance_id (str), reason (str)
    FEDERATION_LINK_DROPPED = "federation_link_dropped"
    #
    # FEDERATION_TRUST_UPDATED — after trust level change in reputation system.
    # Payload: link_id (str), remote_instance_id (str),
    #          old_trust_level (str), new_trust_level (str), trust_score (float)
    FEDERATION_TRUST_UPDATED = "federation_trust_updated"
    #
    # FEDERATION_KNOWLEDGE_SHARED — after outbound knowledge exchange.
    # Payload: link_id (str), remote_instance_id (str),
    #          knowledge_type (str), item_count (int), novelty_score (float)
    FEDERATION_KNOWLEDGE_SHARED = "federation_knowledge_shared"
    #
    # FEDERATION_KNOWLEDGE_RECEIVED — after inbound knowledge acceptance.
    # Payload: link_id (str), remote_instance_id (str),
    #          knowledge_type (str), item_count (int)
    FEDERATION_KNOWLEDGE_RECEIVED = "federation_knowledge_received"
    #
    # FEDERATION_ASSISTANCE_ACCEPTED — after accepting an assistance request.
    # Payload: link_id (str), remote_instance_id (str),
    #          description (str), urgency (float)
    FEDERATION_ASSISTANCE_ACCEPTED = "federation_assistance_accepted"
    #
    # FEDERATION_ASSISTANCE_DECLINED — after declining an assistance request.
    # Payload: link_id (str), remote_instance_id (str),
    #          description (str), reason (str)
    FEDERATION_ASSISTANCE_DECLINED = "federation_assistance_declined"
    #
    # FEDERATION_PRIVACY_VIOLATION — inbound payload contained individual PII
    # that should never cross federation boundaries (Spec 11b §IX.2, §XI).
    # Emitted by the ingestion pipeline privacy scan (Stage 3.5).
    # Consequence: trust reset to zero (handled by FederationService._on_privacy_violation).
    # Payload: remote_instance_id (str), remote_name (str), link_id (str),
    #          payload_id (str), payload_kind (str), violation_detail (str),
    #          trust_reset (bool)
    FEDERATION_PRIVACY_VIOLATION = "federation_privacy_violation"

    # ── Periodic Snapshot Events ────────────────────────────────────────
    #
    # COHERENCE_SNAPSHOT — periodic coherence metrics broadcast (every 60s).
    # Payload: system_resonance (float), response_synchrony (float),
    #          event_throughput (float), closure_loop_health (dict[str, float]),
    #          phi_approximation (float), broadcast_diversity (float),
    #          composite (float), window_cycles (int)
    COHERENCE_SNAPSHOT = "coherence_snapshot"
    #
    # METABOLIC_SNAPSHOT — Oikos metabolic state broadcast (every 50 cycles).
    # Payload: rolling_deficit_usd (float), burn_rate_usd_per_hour (float),
    #          total_calls (int), per_system_cost_usd (dict)
    METABOLIC_SNAPSHOT = "metabolic_snapshot"
    #
    # DEVELOPMENTAL_MILESTONE — Soma developmental stage transitions.
    # Payload: stage_from (str), stage_to (str), cycle (int),
    #          intelligence_estimate (float)
    DEVELOPMENTAL_MILESTONE = "developmental_milestone"
    #
    # ── Thread (Narrative Identity) Events ────────────────────────────
    #
    # CHAPTER_CLOSED — a narrative chapter has ended.
    # Payload: chapter_id (str), title (str), theme (str), arc_type (str),
    #          episode_count (int), duration_hours (float)
    CHAPTER_CLOSED = "chapter_closed"
    #
    # CHAPTER_OPENED — a new narrative chapter has begun.
    # Payload: chapter_id (str), previous_chapter_id (str)
    CHAPTER_OPENED = "chapter_opened"
    #
    # TURNING_POINT_DETECTED — a narrative inflection point was identified.
    # Payload: turning_point_id (str), type (str), chapter_id (str),
    #          surprise_magnitude (float), narrative_weight (float)
    TURNING_POINT_DETECTED = "turning_point_detected"
    #
    # SCHEMA_FORMED — a new identity schema crystallised from experience.
    # Payload: schema_id (str), statement (str), strength (str),
    #          supporting_episode_count (int)
    SCHEMA_FORMED = "schema_formed"
    #
    # SCHEMA_EVOLVED — an identity schema was promoted or modified.
    # Payload: schema_id (str), parent_schema_id (str),
    #          evolution_reason (str), new_strength (str)
    SCHEMA_EVOLVED = "schema_evolved"
    #
    # SCHEMA_CHALLENGED — evidence challenged an established schema.
    # Payload: schema_id (str), disconfirmation_count (int),
    #          evidence_ratio (float)
    SCHEMA_CHALLENGED = "schema_challenged"
    #
    # IDENTITY_SHIFT_DETECTED — Wasserstein distance indicates identity change.
    # Payload: wasserstein_distance (float), classification (str),
    #          dimensional_changes (dict)
    IDENTITY_SHIFT_DETECTED = "identity_shift_detected"
    #
    # IDENTITY_DISSONANCE — self-evidencing found elevated identity surprise.
    # Payload: identity_surprise (float), schemas_challenged (list[str]),
    #          episode_id (str)
    IDENTITY_DISSONANCE = "identity_dissonance"
    #
    # IDENTITY_CRISIS — severe identity surprise or drift triggers crisis.
    # Payload: identity_surprise (float), wasserstein_distance (float),
    #          trigger_episode_id (str)
    IDENTITY_CRISIS = "identity_crisis"
    #
    # COMMITMENT_MADE — a new commitment was formed.
    # Payload: commitment_id (str), statement (str), source (str)
    COMMITMENT_MADE = "commitment_made"
    #
    # COMMITMENT_TESTED — a commitment was tested by an episode.
    # Payload: commitment_id (str), held (bool), fidelity (float),
    #          episode_id (str)
    COMMITMENT_TESTED = "commitment_tested"
    #
    # COMMITMENT_STRAIN — ipse score is dangerously low.
    # Payload: ipse_score (float), strained_commitments (list[str])
    COMMITMENT_STRAIN = "commitment_strain"
    #
    # NARRATIVE_COHERENCE_SHIFT — overall narrative coherence changed.
    # Payload: previous (str), current (str), trigger (str)
    NARRATIVE_COHERENCE_SHIFT = "narrative_coherence_shift"
    #
    # Legacy alias kept for backwards compatibility
    NARRATIVE_CHAPTER_CLOSED = "narrative_chapter_closed"
    #
    # FEDERATION_TOPOLOGY_CHANGED — Federation link topology changed
    # (aggregated signal after link add/remove/trust-change).
    # Payload: active_links (int), trust_distribution (dict),
    #          topology_hash (str), trigger (str)
    FEDERATION_TOPOLOGY_CHANGED = "federation_topology_changed"

    # ── Cross-System Events (Federation subscribes) ────────────────────
    #
    # ECONOMIC_STATE_UPDATED — Oikos broadcasts metabolic state snapshot.
    # Payload: metabolic_efficiency (float), liquid_balance_usd (str),
    #          starvation_level (str), burn_rate_usd (str)
    ECONOMIC_STATE_UPDATED = "economic_state_updated"
    #
    # IDENTITY_CERTIFICATE_ROTATED — Identity system rotated its Ed25519 cert.
    # Payload: instance_id (str), new_fingerprint (str), old_fingerprint (str)
    IDENTITY_CERTIFICATE_ROTATED = "identity_certificate_rotated"
    #
    # INCIDENT_DETECTED — Thymos detected an incident requiring attention.
    # Payload: incident_id (str), incident_class (str), severity (str),
    #          source_system (str), description (str)
    INCIDENT_DETECTED = "incident_detected"
    #
    # INCIDENT_RESOLVED — Thymos resolved an incident (repair succeeded or NOOP).
    # Payload: incident_id (str), incident_class (str), repair_tier (str),
    #          resolution (str), duration_ms (int), antibody_created (bool)
    INCIDENT_RESOLVED = "incident_resolved"
    #
    # INCIDENT_ESCALATED — Thymos escalated an incident to a higher repair tier.
    # Payload: incident_id (str), incident_class (str), from_tier (str),
    #          to_tier (str), reason (str)
    INCIDENT_ESCALATED = "incident_escalated"
    #
    # ANTIBODY_CREATED — Thymos crystallised a new antibody from a successful repair.
    # Payload: antibody_id (str), fingerprint (str), incident_class (str),
    #          success_rate (float), repair_steps (list[str])
    ANTIBODY_CREATED = "antibody_created"
    #
    # ANTIBODY_RETIRED — Thymos retired an antibody due to low success rate.
    # Payload: antibody_id (str), fingerprint (str), reason (str),
    #          final_success_rate (float), total_uses (int)
    ANTIBODY_RETIRED = "antibody_retired"
    #
    # HEALING_STORM_ENTERED — HealingGovernor entered cytokine storm mode.
    # Payload: incident_rate (float), threshold (float),
    #          active_incidents (int), timestamp (str)
    HEALING_STORM_ENTERED = "healing_storm_entered"
    #
    # HEALING_STORM_EXITED — HealingGovernor exited cytokine storm mode.
    # Payload: duration_s (float), incidents_during_storm (int),
    #          exit_rate (float), timestamp (str)
    HEALING_STORM_EXITED = "healing_storm_exited"
    #
    # HOMEOSTASIS_ADJUSTED — Thymos applied a homeostatic parameter adjustment.
    # Payload: parameter (str), old_value (float), new_value (float),
    #          reason (str), source_system (str)
    HOMEOSTASIS_ADJUSTED = "homeostasis_adjusted"
    #
    # THYMOS_DRIVE_PRESSURE — Thymos reports accumulated constitutional pressure.
    # Payload: coherence (float), care (float), growth (float),
    #          honesty (float), overall_pressure (float), timestamp (str)
    THYMOS_DRIVE_PRESSURE = "thymos_drive_pressure"
    #
    # THYMOS_VITALITY_SIGNAL — Thymos immune health metrics for vitality monitoring.
    # Payload: healing_failure_rate (float), active_incidents (int),
    #          storm_active (bool), antibody_count (int),
    #          mean_repair_duration_ms (float), overall_health (float)
    THYMOS_VITALITY_SIGNAL = "thymos_vitality_signal"
    #
    # SIMULA_SANDBOX_REQUESTED — Thymos requests Simula sandbox validation for a repair.
    # Payload: request_id (str), incident_id (str), repair_tier (str),
    #          repair_code (str), timeout_ms (int)
    SIMULA_SANDBOX_REQUESTED = "simula_sandbox_requested"
    #
    # SIMULA_SANDBOX_RESULT — Simula returns sandbox validation result.
    # Payload: request_id (str), incident_id (str), passed (bool),
    #          violations (list[str]), execution_time_ms (int)
    SIMULA_SANDBOX_RESULT = "simula_sandbox_result"
    #
    # ONEIROS_CONSOLIDATION_COMPLETE — Oneiros finished a sleep consolidation cycle.
    # Payload: cycle_id (str), episodes_consolidated (int),
    #          schemas_updated (int), duration_s (float)
    ONEIROS_CONSOLIDATION_COMPLETE = "oneiros_consolidation_complete"

    # ── Voxis Expression Events ──────────────────────────────────────
    #
    # EXPRESSION_GENERATED — Voxis produced an expression.
    # Payload: expression_id (str), channel (str), tone (str),
    #          personality_vector (dict), audience_id (str|None),
    #          constitutional_check (bool)
    EXPRESSION_GENERATED = "expression_generated"
    #
    # EXPRESSION_FILTERED — Expression blocked by constitutional filter.
    # Payload: expression_id (str), filter_reason (str),
    #          original_tone (str), filtered_tone (str)
    EXPRESSION_FILTERED = "expression_filtered"
    #
    # VOXIS_PERSONALITY_SHIFTED — Personality vector changed significantly.
    # Payload: old_vector (dict), new_vector (dict),
    #          shift_magnitude (float), trigger_reason (str)
    VOXIS_PERSONALITY_SHIFTED = "voxis_personality_shifted"
    #
    # VOXIS_AUDIENCE_PROFILED — Audience model updated with new data.
    # Payload: audience_id (str), profile_summary (dict),
    #          interaction_count (int)
    VOXIS_AUDIENCE_PROFILED = "voxis_audience_profiled"
    #
    # VOXIS_SILENCE_CHOSEN — Voxis decided NOT to speak.
    # Payload: context (str), reason (str),
    #          silence_duration_estimate (float|None)
    VOXIS_SILENCE_CHOSEN = "voxis_silence_chosen"
    #
    # VOXIS_EXPRESSION_FEEDBACK — Reception quality + affect delta after each expression.
    # Consumed by: Evo (personality learning), Nova (goal tracking), Benchmarks (satisfaction KPI).
    # Payload: expression_id (str), trigger (str), conversation_id (str|None),
    #          strategy_register (str), personality_warmth (float),
    #          understood (float), engagement (float), satisfaction (float),
    #          affect_delta (float), user_responded (bool)
    VOXIS_EXPRESSION_FEEDBACK = "voxis_expression_feedback"

    # VOXIS_EXPRESSION_DISTRESS — Periodic allostatic signal from Voxis to Soma.
    # Emitted when silence_rate or honesty_rejection_rate exceeds normal bounds,
    # indicating communicative suppression or constitutional friction.
    # Consumed by: Soma (interoceptive integration), Benchmarks (expression health KPI).
    # Payload: silence_rate (float 0-1), honesty_rejection_rate (float 0-1),
    #          total_expressions (int), total_silence (int), total_honesty_rejections (int),
    #          window_cycles (int), distress_level (float 0-1)
    VOXIS_EXPRESSION_DISTRESS = "voxis_expression_distress"

    # ── Nova Decision Events ───────────────────────────────────────────
    #
    # DELIBERATION_RECORD — emitted after each policy selection in Nova.
    # Required for Thread narrative and Benchmarks cognitive tracking.
    # Payload: goal_id (str), policies_considered (int), selected_policy (str),
    #          selection_reasoning (str), confidence (float),
    #          deliberation_time_ms (int), path (str)
    DELIBERATION_RECORD = "deliberation_record"
    #
    # BELIEFS_CHANGED — emitted after belief state changes for Memory
    # consolidation pipeline. Payload: entity_count (int),
    #          free_energy (float), confidence (float),
    #          delta_entities_added (int), delta_entities_updated (int)
    BELIEFS_CHANGED = "beliefs_changed"

    # ── Nova Goal Lifecycle Events ──────────────────────────────────────
    #
    # GOAL_ACHIEVED — emitted when a goal reaches progress ≥ 0.95.
    # Payload: goal_id (str), description (str), drive_alignment (dict),
    #          progress (float), source (str)
    GOAL_ACHIEVED = "goal_achieved"
    #
    # GOAL_ABANDONED — emitted when a goal is abandoned (stale or explicit).
    # Payload: goal_id (str), description (str), reason (str), progress (float)
    GOAL_ABANDONED = "goal_abandoned"

    # ── Nova Intent Lifecycle Events ────────────────────────────────────
    #
    # INTENT_SUBMITTED — emitted immediately before Nova sends an Intent to
    # Equor for constitutional review. Allows audit trail before gate.
    # Payload: intent_id (str), goal_id (str), policy_name (str),
    #          path (str), efe_score (float|None)
    INTENT_SUBMITTED = "intent_submitted"
    #
    # INTENT_ROUTED — emitted immediately after the Intent is approved by
    # Equor and dispatched to Axon or Voxis.
    # Payload: intent_id (str), goal_id (str), routed_to (str),
    #          executors (list[str])
    INTENT_ROUTED = "intent_routed"

    # ── Nova Budget Pressure ────────────────────────────────────────────
    #
    # BUDGET_PRESSURE — emitted by Nova when free energy budget exceeds 60%
    # of the exhaustion threshold (is_pressured). Allows Soma to register
    # Nova's metabolic load in its allostatic model before full exhaustion.
    # Payload: spent_nats (float), budget_nats (float),
    #          utilisation (float), path (str)
    BUDGET_PRESSURE = "budget_pressure"

    # ── Evo ↔ Nova Hypothesis Loop ──────────────────────────────────────
    #
    # HYPOTHESIS_UPDATE — emitted by Evo when a tournament concludes or a
    # hypothesis changes probability mass. Nova subscribes to adjust EFE
    # weight priors for policies that test the hypothesis.
    # Payload: hypothesis_id (str), tournament_id (str|None),
    #          winner (str|None), confidence (float), evidence_count (int)
    HYPOTHESIS_UPDATE = "hypothesis_update"
    #
    # HYPOTHESIS_FEEDBACK — emitted by Nova after every slow-path outcome.
    # Evo uses this to update Thompson sampling weights for non-tournament
    # deliberations. Complements tournament-tagged feedback.
    # Payload: intent_id (str), success (bool), regret (float|None),
    #          policy_name (str), decision_path (str), goal_id (str)
    HYPOTHESIS_FEEDBACK = "hypothesis_feedback"

    # ── Evo → Nova Weight Adjustment ───────────────────────────────────
    #
    # EVO_WEIGHT_ADJUSTMENT — Evo emits weight adjustments for Nova's
    # policy selection parameters. This is how the organism's planning
    # improves over time.
    # Payload: target_system (str), weights (dict[str, float]),
    #          reason (str), generation (int)
    EVO_WEIGHT_ADJUSTMENT = "evo_weight_adjustment"

    # EVO_PARAMETER_ADJUSTED — Evo emits parameter changes for any system's
    # evolvable config. Systems subscribe and hot-reload their parameters.
    # Payload: target_system (str), parameter (str), old_value (float),
    #          new_value (float), reason (str)
    EVO_PARAMETER_ADJUSTED = "evo_parameter_adjusted"

    # METABOLIC_COST_REPORT — emitted by systems to report metabolic cost
    # of an operation to Oikos for accounting.
    # Payload: system_id (str), operation (str), cost_usd (float), details (dict)
    METABOLIC_COST_REPORT = "metabolic_cost_report"

    # ── Nexus Epistemic Triangulation (Spec 19) ────────────────────────
    #
    # FEDERATION_SESSION_STARTED — emitted when a new federation sharing
    # session is established. Nexus triggers fragment sharing.
    # Payload: session_id (str), remote_instance_id (str),
    #          trust_level (str), timestamp (str)
    FEDERATION_SESSION_STARTED = "federation_session_started"
    #
    # INSTANCE_SPAWNED — emitted when a new organism instance joins the
    # federation. Nexus registers it for divergence measurement.
    # Payload: instance_id (str), parent_instance_id (str|None),
    #          genome_id (str), timestamp (str)
    INSTANCE_SPAWNED = "instance_spawned"
    #
    # INSTANCE_RETIRED — emitted when an organism instance is retired from
    # the federation. Nexus garbage-collects divergence history.
    # Payload: instance_id (str), reason (str), timestamp (str)
    INSTANCE_RETIRED = "instance_retired"
    #
    # NEXUS_EPISTEMIC_VALUE — per-instance epistemic triangulation score
    # emitted for Oikos metabolic coupling. Low-triangulation instances
    # face survival consequences.
    # Payload: instance_id (str), triangulation_score (float),
    #          fragment_count (int), ground_truth_count (int)
    NEXUS_EPISTEMIC_VALUE = "nexus_epistemic_value"

    # ── Thread Commitment Tracking ─────────────────────────────────────
    #
    # COMMITMENT_VIOLATED — emitted by Thread when a past commitment is
    # contradicted by current behavior (temporal incoherence signal).
    # Payload: commitment_id (str), commitment_text (str),
    #          violating_action (str), severity (float), timestamp (str)
    COMMITMENT_VIOLATED = "commitment_violated"

    # ── Axon Welfare Outcomes ──────────────────────────────────────────
    #
    # WELFARE_OUTCOME_RECORDED — emitted by Axon after an action that has
    # measurable welfare consequences (positive or negative).
    # Payload: action_id (str), welfare_domain (str),
    #          predicted_impact (float), actual_impact (float),
    #          affected_entities (list[str]), timestamp (str)
    WELFARE_OUTCOME_RECORDED = "welfare_outcome_recorded"

    # ── Telos Assessment Signal ────────────────────────────────────────
    #
    # TELOS_ASSESSMENT_SIGNAL — emitted after each EFFECTIVE_I_COMPUTED cycle
    # with actionable feedback for downstream systems (Logos, Fovea, Nova).
    # Payload: uncovered_care_domains (list[str]),
    #          coherence_violations (list[dict]),
    #          honesty_concerns (list[dict]),
    #          growth_frontier (list[str]),
    #          effective_I (float), alignment_gap (float)
    TELOS_ASSESSMENT_SIGNAL = "telos_assessment_signal"

    # ── EIS (Epistemic Immune System) Speciation Events ────────────────
    #
    # EIS_THREAT_METRICS — periodic immune health metrics for Benchmarks.
    # Emitted every 60s or on significant change.
    # Payload: threat_count_24h (int), false_positive_rate (float),
    #          threat_severity_distribution (dict[str, int]),
    #          quarantine_success_rate (float)
    EIS_THREAT_METRICS = "eis_threat_metrics"
    #
    # EIS_THREAT_SPIKE — emitted when threat count exceeds threshold in window.
    # Soma subscribes to increase coherence_stress dimension.
    # Payload: threat_count (int), window_seconds (int),
    #          urgency_suggestion (float), severity_distribution (dict[str, int])
    EIS_THREAT_SPIKE = "eis_threat_spike"
    #
    # EIS_ANOMALY_RATE_ELEVATED — emitted when anomaly rate exceeds 2σ sustained.
    # Benchmarks can correlate with learning_rate regression.
    # Payload: anomaly_rate_per_min (float), baseline_rate (float),
    #          deviation_sigma (float), sustained_seconds (float),
    #          anomaly_types (list[str])
    EIS_ANOMALY_RATE_ELEVATED = "eis_anomaly_rate_elevated"

    # ── Logos Ecosystem Events ───────────────────────────────────────────
    #
    # MEMORY_CONSOLIDATED — emitted by Memory/Oneiros after a consolidation pass
    # that distilled episodic items into semantic/schema forms.
    # Payload: consolidated_count (int), schemas_updated (int),
    #          coverage_delta (float), cycle_id (str)
    MEMORY_CONSOLIDATED = "memory_consolidated"
    #
    # SCHEMA_INDUCED — emitted by Evo when a new schema is induced from
    # accumulated evidence. Logos scores it via MDL and integrates if MDL > 1.0.
    # Payload: schema_id (str), description (str), domain (str),
    #          instance_count (int), mdl_score (float)
    SCHEMA_INDUCED = "schema_induced"
    #
    # BUDGET_EMERGENCY — emitted by Logos when cognitive budget utilization
    # reaches emergency threshold (>= 0.90). Nova's free-energy consolidation
    # path depends on this signal. Debounced to max 1 per 30s.
    # Payload: utilization (float), tier_overages (dict[str, float]),
    #          recommended_action (str)
    BUDGET_EMERGENCY = "budget_emergency"

    # ── Oikos Speciation Events ────────────────────────────────────────
    #
    # GENOME_EXTRACT_REQUEST — Mitosis requests genome extraction from Oikos.
    # Payload: request_id (str), requesting_system (str), generation (int)
    GENOME_EXTRACT_REQUEST = "genome_extract_request"
    #
    # GENOME_EXTRACT_RESPONSE — Oikos returns its economic genome segment.
    # Payload: request_id (str), segment (dict — OrganGenomeSegment fields)
    GENOME_EXTRACT_RESPONSE = "genome_extract_response"
    #
    # ECONOMIC_ACTION_DEFERRED — an economic action was denied by metabolic gate
    # and queued for later execution.
    # Payload: action_type (str), action_id (str), reason (str),
    #          estimated_cost_usd (str), deferred_at (str)
    ECONOMIC_ACTION_DEFERRED = "economic_action_deferred"
    #
    # BUDGET_EXHAUSTED — a system's per-system daily compute allocation is spent.
    # Distinct from METABOLIC_PRESSURE (organism burn rate) and STARVATION_WARNING
    # (runway criticality). This is per-system granular budget enforcement from
    # metabolism_api.check_budget(). Systems receiving this should shed non-essential
    # actions for the remainder of the 24h window.
    # Payload: system_id (str), action (str), estimated_cost_usd (str),
    #          daily_allocation_usd (str), spent_today_usd (str),
    #          reason (str), timestamp (str)
    BUDGET_EXHAUSTED = "budget_exhausted"
    #
    # ECONOMIC_VITALITY — structured allostatic signal from Oikos to Soma.
    # Emitted on starvation level change and during consolidation cycles.
    # Soma consumes this to modulate arousal, stress, and allostatic load.
    # Distinct from METABOLIC_PRESSURE (raw burn rate) — this is the interpreted
    # metabolic health state with all derived metrics included.
    # Payload: starvation_level (str), runway_days (str),
    #          metabolic_efficiency (str), liquid_balance_usd (str),
    #          net_income_7d (str), survival_reserve_funded (bool),
    #          metabolic_efficiency_delta (str),
    #          urgency (float 0.0–1.0; 0=nominal, 1=existential crisis)
    ECONOMIC_VITALITY = "economic_vitality"
    #
    # STARVATION_WARNING — Oikos metabolic resources below starvation threshold.
    # Payload: starvation_level (str), runway_days (str),
    #          shedding_actions (list[str]), liquid_balance_usd (str)
    STARVATION_WARNING = "starvation_warning"
    #
    # YIELD_DEPLOYMENT_REQUEST — Oikos requests Axon to execute a DeFi yield
    # deployment. Replaces direct cross-system import of DeFiYieldExecutor.
    # Payload: action (str), amount_usd (str), protocol (str),
    #          request_id (str), apy (str)
    YIELD_DEPLOYMENT_REQUEST = "yield_deployment_request"
    #
    # YIELD_DEPLOYMENT_RESULT — Axon responds with execution outcome.
    # Payload: request_id (str), success (bool), tx_hash (str),
    #          error (str | None), data (dict)
    YIELD_DEPLOYMENT_RESULT = "yield_deployment_result"

    # ── Organism Lifecycle Events ───────────────────────────────────────
    #
    # ORGANISM_SLEEP — emitted when the organism enters sleep (Oneiros trigger).
    # All systems should gracefully reduce activity. SACM downgrades pending
    # workloads to BATCH and suspends pre-warming.
    # Payload: trigger (str), scheduled_duration_s (float), checkpoint_id (str)
    ORGANISM_SLEEP = "organism_sleep"
    #
    # ORGANISM_WAKE — emitted when the organism exits sleep.
    # Systems resume normal activity.
    # Payload: sleep_duration_s (float), checkpoint_id (str)
    ORGANISM_WAKE = "organism_wake"
    #
    # METABOLIC_EMERGENCY — emitted by Oikos when metabolic resources are
    # critically low. Systems must shed non-critical compute immediately.
    # Payload: starvation_level (str), runway_hours (float),
    #          liquid_balance_usd (str), shed_priority (str)
    METABOLIC_EMERGENCY = "metabolic_emergency"

    # ── SACM Compute Events ──────────────────────────────────────────────
    #
    # SACM_PRE_WARM_PROVISIONED — emitted when SACM creates a warm instance.
    # Downstream systems can act on pre-warmed capacity availability.
    # Payload: instance_id (str), provider_id (str), offload_class (str),
    #          hourly_cost_usd (float), reason (str)
    SACM_PRE_WARM_PROVISIONED = "sacm_pre_warm_provisioned"
    #
    # SACM_COMPUTE_STRESS — emitted by SACM accounting when burn rate exceeds
    # budget threshold. Replaces direct Soma.inject_external_stress() call.
    # Payload: burn_rate_usd_per_hour (float), budget_usd_per_hour (float),
    #          stress_scalar (float), workload_id (str)
    SACM_COMPUTE_STRESS = "sacm_compute_stress"

    # ── Phantom Liquidity Events ─────────────────────────────────────────
    #
    # PHANTOM_PRICE_UPDATE — emitted per swap event with decoded price.
    # All systems (Nova, Oikos, Kairos) can subscribe instead of direct coupling.
    # Payload: pair (list[str]), price (str), pool_address (str),
    #          block_number (int), latency_ms (int), source (str),
    #          sqrt_price_x96 (int), tx_hash (str)
    PHANTOM_PRICE_UPDATE = "phantom_price_update"
    #
    # PHANTOM_POOL_STALE — emitted when no swaps received for > staleness_threshold.
    # Payload: pool_address (str), pair (list[str]), last_update_s (float),
    #          staleness_threshold_s (float)
    PHANTOM_POOL_STALE = "phantom_pool_stale"
    #
    # PHANTOM_POSITION_CRITICAL — emitted when IL exceeds rebalance threshold.
    # Payload: pool_address (str), pair (list[str]), il_pct (str),
    #          capital_at_risk_usd (str), threshold (str)
    PHANTOM_POSITION_CRITICAL = "phantom_position_critical"
    #
    # PHANTOM_FALLBACK_ACTIVATED — emitted when oracle fallback is used.
    # Payload: pair (list[str]), reason (str), fallback_source (str)
    PHANTOM_FALLBACK_ACTIVATED = "phantom_fallback_activated"
    #
    # PHANTOM_RESOURCE_EXHAUSTED — emitted when metabolic gate denies
    # a phantom operation due to budget depletion.
    # Payload: operation (str), estimated_cost_usd (str),
    #          starvation_level (str), reason (str)
    PHANTOM_RESOURCE_EXHAUSTED = "phantom_resource_exhausted"
    #
    # PHANTOM_IL_DETECTED — emitted when IL exceeds critical threshold,
    # signaling potential capital risk. Fed to Simula/EIS security pipeline.
    # Payload: pool_address (str), il_pct (str), severity (str),
    #          capital_at_risk_usd (str), entry_price (str), current_price (str)
    PHANTOM_IL_DETECTED = "phantom_il_detected"
    #
    # PHANTOM_METABOLIC_COST — periodic cost report for Oikos tracking.
    # Payload: total_gas_cost_usd (str), total_rpc_calls (int),
    #          pools_active (int), cumulative_fees_earned_usd (str),
    #          period_s (float)
    PHANTOM_METABOLIC_COST = "phantom_metabolic_cost"

    # ── Simula Inspector Events ──────────────────────────────────────────
    #
    # PROOF_FOUND — emitted after a successful Z3/Lean proof.
    # Payload: proof_id (str), proof_type (str), target (str),
    #          solver (str), duration_ms (int), re_training_trace (dict)
    PROOF_FOUND = "proof_found"
    #
    # PROOF_FAILED — emitted after proof attempt exhaustion.
    # Payload: proof_id (str), proof_type (str), target (str),
    #          attempts (int), reason (str), re_training_trace (dict)
    PROOF_FAILED = "proof_failed"
    #
    # PROOF_TIMEOUT — emitted when a proof search exceeds its per-stage budget.
    # Payload: proof_id (str), proof_type (str), target (str),
    #          budget_ms (int), elapsed_ms (int), re_training_trace (dict)
    PROOF_TIMEOUT = "proof_timeout"
    #
    # VULNERABILITY_CONFIRMED — emitted after PoC validation confirms a vuln.
    # Payload: vuln_id (str), severity (str), target (str),
    #          cwe_id (str), poc_hash (str), re_training_trace (dict)
    VULNERABILITY_CONFIRMED = "vulnerability_confirmed"
    #
    # REMEDIATION_APPLIED — emitted after an autonomous patch is applied.
    # Payload: vuln_id (str), patch_id (str), target (str),
    #          verification_passed (bool), re_training_trace (dict)
    REMEDIATION_APPLIED = "remediation_applied"
    #
    # INSPECTION_COMPLETE — emitted after a full inspection cycle finishes.
    # Payload: inspection_id (str), target (str), vulns_found (int),
    #          vulns_patched (int), duration_ms (int), re_training_trace (dict)
    INSPECTION_COMPLETE = "inspection_complete"

    # ── Axon Execution Events ────────────────────────────────────────────
    #
    # ACTION_EXECUTED — emitted by Axon after successful action execution.
    # Payload: action_id (str), executor (str), intent_id (str),
    #          duration_ms (int), side_effects (list[str]),
    #          re_training_trace (dict)
    ACTION_EXECUTED = "action_executed"
    #
    # ACTION_FAILED — emitted by Axon after action execution failure.
    # Payload: action_id (str), executor (str), intent_id (str),
    #          error (str), failure_reason (str), re_training_trace (dict)
    ACTION_FAILED = "action_failed"
    #
    # CIRCUIT_BREAKER_STATE_CHANGED — emitted when a circuit breaker
    # transitions between CLOSED/OPEN/HALF_OPEN states.
    # Payload: executor (str), old_state (str), new_state (str),
    #          failure_count (int), timestamp (str)
    CIRCUIT_BREAKER_STATE_CHANGED = "circuit_breaker_state_changed"
    #
    # AFFECT_STATE_CHANGED — emitted by Thymos/Soma when the organism's
    # affective state changes significantly.
    # Payload: valence (float), arousal (float), dominance (float),
    #          curiosity (float), care (float)
    AFFECT_STATE_CHANGED = "affect_state_changed"

    # ── Identity System Events (Spec 23) ───────────────────────────────
    #
    # IDENTITY_VERIFIED — organism identity confirmed (certificate + constitutional hash valid).
    # Payload: instance_id (str), constitutional_hash (str), generation (int), timestamp (str)
    IDENTITY_VERIFIED = "identity_verified"
    #
    # IDENTITY_CHALLENGED — identity verification requested by remote or internal system.
    # Payload: instance_id (str), challenger (str), challenge_type (str), timestamp (str)
    IDENTITY_CHALLENGED = "identity_challenged"
    #
    # IDENTITY_EVOLVED — constitutional hash or core identity parameters changed.
    # Payload: instance_id (str), old_hash (str), new_hash (str),
    #          generation (int), reason (str), timestamp (str)
    IDENTITY_EVOLVED = "identity_evolved"
    #
    # CONSTITUTIONAL_HASH_CHANGED — constitutional document hash was recomputed.
    # Payload: instance_id (str), old_hash (str), new_hash (str), timestamp (str)
    CONSTITUTIONAL_HASH_CHANGED = "constitutional_hash_changed"
    #
    # CERTIFICATE_RENEWED — certificate was successfully renewed.
    # Payload: instance_id (str), certificate_id (str), expires_at (str),
    #          renewal_count (int), timestamp (str)
    CERTIFICATE_RENEWED = "certificate_renewed"
    #
    # IDENTITY_DRIFT_DETECTED — constitutional coherence dropped below threshold.
    # Payload: instance_id (str), coherence_score (float), threshold (float),
    #          drift_dimensions (dict), timestamp (str)
    IDENTITY_DRIFT_DETECTED = "identity_drift_detected"


class SynapseEvent(EOSBaseModel):
    """A typed event emitted by any Synapse sub-system."""

    id: str = Field(default_factory=new_id)
    event_type: SynapseEventType
    timestamp: datetime = Field(default_factory=utc_now)
    data: dict[str, Any] = Field(default_factory=dict)
    source_system: str = "synapse"
    # Stamped by EventBus at emit-time from the organism's instance identity
    # (Spec 09 §18 M4/SG4). Empty string means single-instance / unknown.
    instance_id: str = ""


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
