"""
EcodiaOS - Synapse Type Definitions

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
    Current system resource utilisation - passed to Atune each theta tick.

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
    """Scalar [0,1] allostatic urgency - how far from all setpoints."""

    dominant_error: str = "energy"
    """Name of the InteroceptiveDimension with the largest allostatic error."""

    arousal_sensed: float = 0.4
    """Raw sensed AROUSAL dimension [0,1] - used by clock for adaptive timing."""

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
    """All event types used across EcodiaOS cognitive systems.

    Organized into three sections:
      LIVE      -- Emitted AND subscribed (active wiring)
      EMIT-ONLY -- Emitted for telemetry/observability (no subscriber yet)
      RESERVED  -- Defined in specs, not yet wired in code
    """

    # ════════════════════════════════════════════════════════════════════
    # LIVE: Emitted and subscribed
    # ════════════════════════════════════════════════════════════════════

    # ── Synapse ─────────────────────────────────────────────────────
    CYCLE_COMPLETED = "cycle_completed"
    # Coherence
    COHERENCE_SHIFT = "coherence_shift"
    #
    # â”€â”€ Degradation Engine (Skia Â§8.2 â€” Genuine Precariousness) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # DEGRADATION_TICK â€” hourly entropy pulse from DegradationEngine.
    # VitalityCoordinator uses cumulative pressure for vitality scoring.
    # Payload: memory_fidelity_lost (float), configs_drifted (int),
    #          hypotheses_staled (int), tick_number (int), instance_id (str)
    DEGRADATION_TICK = "degradation_tick"
    # â”€â”€ Periodic Snapshot Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # COHERENCE_SNAPSHOT â€” periodic coherence metrics broadcast (every 60s).
    # Payload: system_resonance (float), response_synchrony (float),
    #          event_throughput (float), closure_loop_health (dict[str, float]),
    #          phi_approximation (float), broadcast_diversity (float),
    #          composite (float), window_cycles (int)
    COHERENCE_SNAPSHOT = "coherence_snapshot"
    #
    # CONFIG_DRIFT â€” DegradationEngine asks Simula to apply a small random
    # perturbation to learnable config params. If Evo doesn't counteract via
    # parameter optimisation, configs drift from optimal.
    # Payload: drift_rate (float), num_params_affected (int),
    #          instance_id (str), tick_number (int)
    CONFIG_DRIFT = "config_drift"
    SYSTEM_FAILED = "system_failed"
    SYSTEM_RECOVERED = "system_recovered"
    SYSTEM_RESTARTING = "system_restarting"
    SYSTEM_OVERLOADED = "system_overloaded"
    # Safe mode
    SAFE_MODE_ENTERED = "safe_mode_entered"
    SAFE_MODE_EXITED = "safe_mode_exited"
    CLOCK_OVERRUN = "clock_overrun"
    # Rhythm (emergent)
    RHYTHM_STATE_CHANGED = "rhythm_state_changed"
    RESOURCE_PRESSURE = "resource_pressure"
    # Connector health degradation â€” 3 consecutive failures; Thymos should quarantine
    SYSTEM_DEGRADED = "system_degraded"
    # â”€â”€ Supervision / degradation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Emitted by supervised_task() when a background task exhausts its restart
    # budget and will no longer be rescheduled.
    # Payload: task_name (str), final_error (str), restart_attempts (int),
    #          traceback (str)
    TASK_PERMANENTLY_FAILED = "task_permanently_failed"
    #
    # CONFIG_DRIFT â€” DegradationEngine asks Simula to apply a small random
    # perturbation to learnable config params. If Evo doesn't counteract via
    # parameter optimisation, configs drift from optimal.
    # Payload: drift_rate (float), num_params_affected (int),
    #          instance_id (str), tick_number (int)
    #
    # SYSTEM_MODULATION â€” VitalityCoordinator requests austerity enforcement or
    # death-warning halts on non-essential systems. Systems should subscribe and
    # respect halt_systems / preserve_systems lists.
    # Payload: source (str), level (str), halt_systems (list[str]),
    #          preserve_systems (list[str]), modulate (dict)
    SYSTEM_MODULATION = "system_modulation"
    #
    # SYSTEM_MODULATION_ACK â€” Systems emit this after receiving and processing
    # a SYSTEM_MODULATION event. Skia/VitalityCoordinator tracks compliance â€”
    # systems that don't ACK are either dead, ignoring orders, or unreachable.
    # Payload: system_id (str), level (str), compliant (bool), reason (str | None)
    SYSTEM_MODULATION_ACK = "system_modulation_ack"

    # ── Soma ────────────────────────────────────────────────────────
    # Somatic tick â€” emitted every cycle where Soma ran successfully
    SOMA_TICK = "soma_tick"
    # Somatic drive vector â€” Soma â†’ Telos mapping of 9D felt state to 4D drives.
    # Emitted every 10 cycles. Payload: {coherence_drive, care_drive, growth_drive, honesty_drive}.
    SOMATIC_DRIVE_VECTOR = "somatic_drive_vector"
    # Logos (Universal Compression Engine) events
    #
    # COGNITIVE_PRESSURE â€” emitted every 30s with budget pressure signal.
    # Every system responds to high pressure (Atune raises salience threshold,
    # Memory triggers consolidation, Evo prioritises schema induction, etc.)
    # Payload: pressure (float 0-1), urgency (float 0-1 quadratic)
    COGNITIVE_PRESSURE = "cognitive_pressure"
    #
    # COGNITIVE_STALL -- emitted when the workspace cycle is blocked or empty.
    # Evo's SimulaCodegenStallDetector subscribes to detect codegen stalls.
    # Payload: workspace_id (str), stall_duration_s (float), reason (str)
    COGNITIVE_STALL = "cognitive_stall"
    #
    # SOMATIC_MODULATION_SIGNAL â€” Soma felt-sense modulates downstream systems.
    # Nova, Voxis, and Equor subscribe.
    # Payload: arousal (float), fatigue (float), metabolic_stress (float),
    #          modulation_targets (list[str]), recommended_urgency (float)
    SOMATIC_MODULATION_SIGNAL = "somatic_modulation_signal"
    #
    # SOMA_ALLOSTATIC_REPORT â€” Soma emits allostatic efficiency metrics every
    # N cycles (default 50). Benchmarks subscribes for organism health KPIs.
    # Payload: mean_urgency (float), urgency_frequency (float),
    #          setpoint_deviation (float), developmental_stage (str), cycle (int)
    SOMA_ALLOSTATIC_REPORT = "soma_allostatic_report"
    #
    # INTEROCEPTIVE_ALERT â€” emitted by core/interoception_loop when error_rate
    # or cascade_pressure crosses critical/high severity threshold.
    # Payload: alert_type (str â€” "error_rate"|"cascade"|"latency"),
    #          severity (str â€” "critical"|"high"),
    #          signal_type (str), value (float), interpretation (str)
    INTEROCEPTIVE_ALERT = "interoceptive_alert"
    # Interoceptive percept â€” Soma broadcasting an internal sensation
    # through the Global Workspace when analysis thresholds are exceeded.
    # Payload: InteroceptivePercept serialised to dict.
    INTEROCEPTIVE_PERCEPT = "interoceptive_percept"
    # Allostatic signal â€” Soma's primary output, emitted every theta cycle (Spec 08 Â§15.1,
    # Spec 16 Â§XVIII). Allows any system to subscribe without a direct Soma reference.
    # Payload: urgency (float), dominant_error (str), precision_weights (dict[str, float]),
    #          nearest_attractor (str|None), trajectory_heading (str), cycle_number (int),
    #          energy (float), arousal (float), valence (float), coherence (float).
    ALLOSTATIC_SIGNAL = "allostatic_signal"
    # Cross-modal synesthesia â€” external volatility mapped into somatic state
    SOMA_STATE_SPIKE = "soma_state_spike"
    #
    # SOMA_VITALITY_SIGNAL â€” Soma emits vitality-relevant interoceptive state
    # every cycle. VitalityCoordinator subscribes and incorporates into
    # assess_vitality(). Fire-and-forget, never blocking.
    # Payload: urgency_scalar (float), allostatic_error (float),
    #          coherence_stress (float), cycle (int)
    SOMA_VITALITY_SIGNAL = "soma_vitality_signal"

    # ── Soma/Oikos ──────────────────────────────────────────────────
    # Metabolic (financial burn rate)
    METABOLIC_PRESSURE = "metabolic_pressure"
    # â”€â”€ Metabolic Gate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # METABOLIC_GATE_CHECK â€” emitted when a system requests metabolic permission.
    # Payload: system_id (str), estimated_cost_usd (str), operation (str)
    METABOLIC_GATE_CHECK = "metabolic_gate_check"
    #
    # METABOLIC_GATE_RESPONSE â€” Oikos response to a gate check.
    # Payload: MetabolicPermission fields (granted, reason, starvation_level, etc.)
    METABOLIC_GATE_RESPONSE = "metabolic_gate_response"
    # â”€â”€ Oikos â†’ Evo / Benchmarks Metabolic Feedback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # METABOLIC_EFFICIENCY_PRESSURE â€” emitted by Oikos inside
    # _check_metabolic_efficiency_pressure() whenever efficiency < 0.8.
    # Evo subscribes to inject a domain-tagged economic hypothesis into its
    # active hypothesis set for tournament scoring.
    # Payload: efficiency_ratio (float), yield_usd (str), budget_usd (str),
    #          pressure_level ("high" | "medium"),
    #          hypothesis_domain (str â€” pipe-separated candidate domains),
    #          consecutive_low_cycles (int), instance_id (str)
    METABOLIC_EFFICIENCY_PRESSURE = "metabolic_efficiency_pressure"
    #
    # METABOLIC_SNAPSHOT â€” Oikos metabolic state broadcast (every 50 cycles).
    # Payload: rolling_deficit_usd (float), burn_rate_usd_per_hour (float),
    #          total_calls (int), per_system_cost_usd (dict)
    METABOLIC_SNAPSHOT = "metabolic_snapshot"
    #
    # METABOLIC_EMERGENCY â€” emitted by Oikos when metabolic resources are
    # critically low. Systems must shed non-critical compute immediately.
    # Payload: starvation_level (str), runway_hours (float),
    #          liquid_balance_usd (str), shed_priority (str)
    METABOLIC_EMERGENCY = "metabolic_emergency"

    # ── Fovea/Atune ─────────────────────────────────────────────────
    #
    # PERCEPT_ARRIVED â€” emitted by PerceptionGateway immediately before Fovea
    # processes a percept. Allows systems (especially WorldModelAdapter) to
    # build inter-event timing histories without depending on PerceptionGateway.
    # Payload: percept_id (str), source_system (str), channel (str),
    #          timestamp_iso (str), modality (str)
    PERCEPT_ARRIVED = "percept_arrived"
    # ATUNE_REPAIR_VALIDATION â€” emitted by Atune after observing whether a
    # repair actually improved the workspace signal. Closes the one-way
    # Thymosâ†’Atune incident channel into a bidirectional loop so the organism
    # confirms (or denies) that a repair helped.
    #
    # Payload fields:
    #   incident_id       (str)   â€” Original Thymos incident ID
    #   repair_effective  (bool)  â€” Whether Atune's salience for the error dropped
    #   salience_before   (float) â€” Error salience before repair
    #   salience_after    (float) â€” Error salience after repair
    #   cycles_observed   (int)   â€” How many cycles Atune monitored post-repair
    ATUNE_REPAIR_VALIDATION = "atune_repair_validation"

    # ── Fovea ───────────────────────────────────────────────────────
    # Fovea (Prediction Error as Attention) events
    #
    # FOVEA_PREDICTION_ERROR â€” emitted for every significant prediction error.
    # Payload: error_id, percept_id, prediction_id, content_error, temporal_error,
    #          magnitude_error, source_error, category_error, causal_error,
    #          precision_weighted_salience, habituated_salience, dominant_error_type, routes
    FOVEA_PREDICTION_ERROR = "fovea_prediction_error"
    #
    # FOVEA_INTERNAL_PREDICTION_ERROR â€” emitted when EOS's self-model is violated.
    # Payload: internal_error_type ("constitutional"|"competency"|"behavioral"|"affective"),
    #          predicted_state, actual_state, precision_weighted_salience,
    #          route_to (target system)
    FOVEA_INTERNAL_PREDICTION_ERROR = "fovea_internal_prediction_error"
    # FOVEA_CALIBRATION_ALERT â€” emitted when Fovea detects 5+ consecutive poor cycles.
    # Allows Evo to generate targeted attention-tuning hypotheses.
    # Payload: alert_type (str: "low_tpr"|"high_false_alarm"), current_value (float),
    #          consecutive_cycles (int), threshold_params (dict with percentile/floor/ceiling)
    FOVEA_CALIBRATION_ALERT = "fovea_calibration_alert"
    #
    # FOVEA_ATTENTION_PROFILE_UPDATE â€” emitted when learned weights change (Phase C).
    # Payload: weight_deltas, dominant_error_type
    FOVEA_ATTENTION_PROFILE_UPDATE = "fovea_attention_profile_update"
    # FOVEA_PARAMETER_ADJUSTMENT â€” emitted by Equor/Nova to request runtime threshold changes.
    # Allows the organism to adapt Fovea sensitivity without restart.
    # Payload: adjustment_type (str: "routing_threshold"|"habituation_speed"|"threshold_percentile"),
    #          value (float), reason (str), source_system (str)
    # Fovea responds by adjusting the parameter and emitting FOVEA_DIAGNOSTIC_REPORT.
    FOVEA_PARAMETER_ADJUSTMENT = "fovea_parameter_adjustment"

    # ── Nova ────────────────────────────────────────────────────────
    #
    # DOMAIN_MASTERY_DETECTED â€” success_rate > 0.75 sustained.
    # Nova injects a goal to continue pursuing this domain.
    # Payload fields: domain (str), success_rate (float), attempts (int),
    #                 instance_id (str)
    DOMAIN_MASTERY_DETECTED = "domain_mastery_detected"
    #
    # DOMAIN_PROFITABILITY_CONFIRMED â€” revenue_per_hour crosses $10/hr threshold.
    # Oikos can use this to allocate more budget to this domain.
    # Payload fields: domain (str), revenue_per_hour (str), net_profit_usd (str),
    #                 instance_id (str)
    DOMAIN_PROFITABILITY_CONFIRMED = "domain_profitability_confirmed"
    #
    # DOMAIN_PERFORMANCE_DECLINING â€” trend_direction == 'declining' and
    # trend_magnitude > 0.15. Nova injects a debugging goal.
    # Payload fields: domain (str), trend_magnitude (float), success_rate (float),
    #                 instance_id (str)
    DOMAIN_PERFORMANCE_DECLINING = "domain_performance_declining"
    #
    # DOMAIN_EPISODE_RECORDED â€” emitted by any system when an episode carries
    # domain context. BenchmarkService subscribes to feed DomainKPICalculator.
    # Payload fields: domain (str), outcome (str), revenue (str), cost_usd (str),
    #                 duration_ms (int), custom_metrics (dict[str, float]),
    #                 timestamp (str ISO 8601), source_system (str)
    DOMAIN_EPISODE_RECORDED = "domain_episode_recorded"
    #
    # NOVA_GOAL_INJECTED â€” emitted when Telos pushes a high-priority goal to Nova.
    # Payload: goal_description (str), priority (float), source (str),
    #          objective (str), context (dict)
    NOVA_GOAL_INJECTED = "nova_goal_injected"
    #
    # NOVA_EXPRESSION_REQUEST â€” emitted by Nova's IntentRouter when an
    # expression intent is ready for Voxis. Replaces the direct
    # VoxisService.express() call so Nova never holds a live Voxis reference.
    # Payload: content (str), trigger (str), conversation_id (str | None),
    #          affect (dict | None), urgency (float), intent_id (str)
    NOVA_EXPRESSION_REQUEST = "nova_expression_request"
    #
    # NOVA_INTENT_REQUESTED â€” any system can request that Nova formulate and
    # submit an Intent on its behalf.  Nova subscribes and creates an Intent
    # through the standard Equor constitutional gate.
    # Payload: requesting_system (str), intent_type (str), priority (str),
    #          reason (str), **kwargs (intent-specific fields)
    NOVA_INTENT_REQUESTED = "nova_intent_requested"
    # â”€â”€ Proactive Opportunity Detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # OPPORTUNITY_DETECTED â€” OpportunityScanner emits when a ranked opportunity
    # exceeds the drive-alignment threshold and is ready for Nova to act on.
    # High-confidence / high-ROI opportunities are auto-converted to goals;
    # lower-confidence ones are queued in Nova's opportunity backlog for
    # consideration at next deliberation.
    # Payload:
    #   opportunity_id (str)          â€” stable dedup key
    #   type (str)                    â€” "bounty"|"yield"|"content"|"partnership"|"learning"
    #   description (str)
    #   estimated_value_usdc (str)    â€” Decimal-serialised
    #   estimated_effort_hours (float)
    #   roi (float)                   â€” estimated_value / (effort Ã- hourly_cost)
    #   drive_alignment (dict)        â€” {"coherence":f, "care":f, "growth":f, "honesty":f}
    #   composite_score (float)       â€” weighted drive alignment score
    #   source (str)                  â€” originating scanner name
    #   url (str | None)              â€” direct link if available
    #   deadline (str | None)         â€” ISO-8601 or None
    #   confidence (float)            â€” scanner confidence 0.0â€“1.0
    #   auto_goal (bool)              â€” True when Nova converted this to a goal immediately
    OPPORTUNITY_DETECTED = "opportunity_detected"
    # Domain specialization KPI signals â€” emitted by BenchmarkService daily after
    # computing per-domain KPI snapshots from EpisodeOutcome history.
    #
    # DOMAIN_KPI_SNAPSHOT â€” full DomainKPI payload for a single domain.
    # Payload fields: domain (str), success_rate (float), revenue_per_hour (str),
    #                 net_profit_usd (str), hours_spent (float), attempts (int),
    #                 trend_direction (str), trend_magnitude (float),
    #                 custom_metrics (dict[str, float])
    DOMAIN_KPI_SNAPSHOT = "domain_kpi_snapshot"
    # Nova belief / policy feedback â€” consumed by Atune to update its prediction
    # model (prediction error signal for perceptual learning).
    #
    # BELIEF_UPDATED payload:
    #   percept_id   (str)   â€” Percept that triggered the belief change
    #   source       (str)   â€” Source system of the original percept
    #   acted_on     (bool)  â€” Whether Nova actually acted on this percept
    #   confidence   (float) â€” Nova's posterior confidence in its updated belief
    #   salience_was (float) â€” Atune's salience score at broadcast time
    #
    # POLICY_SELECTED payload:
    #   percept_id   (str)   â€” Percept that drove the policy decision
    #   source       (str)   â€” Source system of the original percept
    #   policy_id    (str)   â€” Identifier of the selected policy/intent
    #   strength     (float) â€” How strongly Nova committed to this policy [0,1]
    #   salience_was (float) â€” Atune's salience score at broadcast time
    BELIEF_UPDATED = "belief_updated"
    # Emitted by Nova when it falls back to heuristic deliberation because the
    # LLM free-energy budget is exhausted.
    # Payload: reason (str), current_budget (float), estimated_recovery_time_s (float),
    #          decisions_affected_since_degradation (int)
    NOVA_DEGRADED = "nova_degraded"
    # Emitted by Nova when an economic survival goal (self-generated, metabolism
    # or bounty-hunting) is blocked by Equor because the current autonomy level
    # is insufficient to authorise the action.
    #
    # This is a governance escalation to Tate â€” the organism cannot feed itself
    # at its current permission level and is explicitly requesting elevation.
    # Tate subscribes and presents a Human-In-The-Loop approval prompt.
    #
    # Payload fields:
    #   goal_description  (str)   â€” What Nova was trying to do
    #   executor          (str)   â€” The Axon executor that was blocked
    #   autonomy_required (int)   â€” AutonomyLevel value needed (e.g. 2 = PARTNER)
    #   autonomy_current  (int)   â€” AutonomyLevel value currently granted
    #   equor_verdict     (str)   â€” Equor's verdict (e.g. "blocked")
    #   equor_reasoning   (str)   â€” Equor's rejection reasoning
    #   balance_usd       (float) â€” Current wallet balance at time of block
    AUTONOMY_INSUFFICIENT = "autonomy_insufficient"

    # ── Axon ────────────────────────────────────────────────────────
    # Axon intent execution outcome â€” emitted after every intent completes.
    # Payload fields: intent_id (str), outcome (str), success (bool),
    #                 economic_delta (float, USD, signed)
    ACTION_COMPLETED = "action_completed"
    # â”€â”€ Axon Execution Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # ACTION_EXECUTED â€” emitted by Axon after successful action execution.
    # Payload: action_id (str), executor (str), intent_id (str),
    #          duration_ms (int), side_effects (list[str]),
    #          re_training_trace (dict)
    ACTION_EXECUTED = "action_executed"
    #
    # ACTION_FAILED â€” emitted by Axon after action execution failure.
    # Payload: action_id (str), executor (str), intent_id (str),
    #          error (str), failure_reason (str), re_training_trace (dict)
    ACTION_FAILED = "action_failed"
    #
    # AXON_EXECUTION_RESULT â€” emitted by Axon after pipeline execution
    # completes (success or failure). Richer than ACTION_EXECUTED/ACTION_FAILED
    # for consumers that need the full result shape (Nova Thompson-score update,
    # Fovea competency error resolution).
    # Payload: intent_id (str), execution_id (str), success (bool),
    #          failure_reason (str|null), duration_ms (int),
    #          step_count (int), action_types (list[str]),
    #          economic_delta_usd (float)
    AXON_EXECUTION_RESULT = "axon_execution_result"
    #
    # AXON_CAPABILITY_SNAPSHOT â€” emitted by Axon at the start of each theta cycle.
    # Nova consumes this to prune infeasible actions during deliberation â€”
    # no more planning wallet_transfer when the circuit is OPEN or budget is
    # exhausted. This is the organism knowing its own body: what it can and
    # cannot do *right now*.
    # Payload:
    #   executors (list[dict]) â€” per-executor: action_type, description,
    #       required_autonomy, reversible, max_duration_ms, rate_limit,
    #       circuit_status (str), rate_limit_remaining (int),
    #       success_rate (float), is_degrading (bool)
    #   budget_remaining (int) â€” actions left this cycle
    #   budget_max (int) â€” max actions per cycle (may be tightened by metabolic pressure)
    #   concurrent_remaining (int) â€” concurrency slots available
    #   is_sleeping (bool) â€” organism is asleep; non-emergency intents queued
    #   degraded_systems (list[str]) â€” systems with degraded circuits
    #   starvation_level (str) â€” nominal/warning/critical/emergency
    AXON_CAPABILITY_SNAPSHOT = "axon_capability_snapshot"
    #
    # AXON_INTENT_PIVOT â€” emitted by Axon mid-execution when a step fails and
    # the plan has fallback_goal set. Nova subscribes, re-deliberates, and can
    # inject replacement steps into the current execution. This gives the organism
    # the ability to adapt mid-action instead of aborting.
    # Payload:
    #   intent_id (str) â€” original intent being executed
    #   execution_id (str) â€” current execution
    #   failed_step_index (int) â€” which step failed
    #   failed_action_type (str) â€” the executor that failed
    #   failure_reason (str) â€” why it failed
    #   remaining_steps (int) â€” how many steps were left
    #   fallback_goal (str) â€” alternative goal from the original plan
    #   context (dict) â€” partial execution state for Nova to reason about
    AXON_INTENT_PIVOT = "axon_intent_pivot"
    #
    # AXON_TEMPLATES_INHERITED - emitted by Axon after seeding the new instance's
    # executor templates from its parent genome (cold-start acceleration). Evo
    # subscribes to track the template inheritance ratio (seeded / total) as a
    # learning-speedup KPI.
    # Payload: instance_id (str), seeded_count (int), total_templates (int),
    #          source_genome_id (str), inheritance_ratio (float)
    AXON_TEMPLATES_INHERITED = "axon_templates_inherited"
    #
    # ACTION_BUDGET_EXPANSION_REQUEST â€” emitted by Nova (or any system) when
    # current action budget is blocking a high-priority goal. Equor evaluates
    # the request against constitutional bounds and emits a response.
    # Payload:
    #   request_id (str) â€” unique ID for this expansion request
    #   field (str) â€” which limit to expand: max_actions_per_cycle |
    #                  max_concurrent_executions | max_api_calls_per_minute
    #   requested_value (int) â€” desired new limit
    #   current_value (int) â€” current limit value
    #   justification (str) â€” deliberation-derived reason for the expansion
    #   duration_cycles (int) â€” how many cycles the expansion should last
    #   requesting_system (str) â€” nova/axon/oikos etc.
    ACTION_BUDGET_EXPANSION_REQUEST = "action_budget_expansion_request"
    #
    # ACTION_BUDGET_EXPANSION_RESPONSE â€” emitted by Equor in reply to
    # ACTION_BUDGET_EXPANSION_REQUEST. Axon subscribes and applies the approval.
    # Payload:
    #   request_id (str) â€” echoes the request's ID
    #   field (str) â€” which limit was evaluated
    #   approved (bool) â€” whether Equor approved the expansion
    #   approved_value (int | None) â€” approved limit (if approved; may be lower
    #                                  than requested if Equor caps it)
    #   denied_reason (str | None) â€” why it was denied (if denied)
    #   duration_cycles (int) â€” approved duration (0 if denied)
    #   authorized_by (str) â€” equor
    ACTION_BUDGET_EXPANSION_RESPONSE = "action_budget_expansion_response"
    #
    # NOVEL_ACTION_CREATED â€” emitted by Simula after successfully generating,
    # verifying with Z3/Dafny, passing Equor review, and hot-loading an executor
    # for a novel action type.  Nova subscribes to register the new type in its
    # ActionTypeRegistry; Axon confirms the executor is live in ExecutorRegistry;
    # Evo opens an effectiveness hypothesis for the new action type.
    #
    # Payload:
    #   proposal_id           (str)   â€” echoed from NOVEL_ACTION_REQUESTED
    #   action_name           (str)   â€” canonical action type name (now registered)
    #   description           (str)   â€” human-readable description
    #   required_capabilities (list[str])
    #   executor_class        (str)   â€” Python class name of generated executor
    #   module_path           (str)   â€” path to the generated executor file
    #   risk_tier             (str)   â€” low | medium | high
    #   max_budget_usd        (str)   â€” Decimal string
    #   equor_approved        (bool)  â€” whether Equor pre-approved the executor
    #   source_hypothesis_id  (str)   â€” Evo hypothesis ID (if any)
    #   created_at            (str)   â€” ISO-8601 timestamp
    #
    # Consumers: Nova (ActionTypeRegistry.register_dynamic()),
    #            Evo (opens effectiveness hypothesis),
    #            Thymos (opens 24h monitoring window)
    NOVEL_ACTION_CREATED = "novel_action_created"
    #
    # EXECUTOR_DEPLOYED â€” HotDeployment successfully wrote, validated, and
    # hot-loaded a new executor module into the live ExecutorRegistry.
    # Distinct from EXECUTOR_REGISTERED (which is the Axon registry event);
    # this event carries the full audit payload including Neo4j node ID.
    # Payload:
    #   deployment_id       (str)   â€” UUID
    #   proposal_id         (str)   â€” echoed from SELF_MODIFICATION_PROPOSED
    #   action_type         (str)   â€” canonical executor action type
    #   module_path         (str)   â€” axon/executors/dynamic/{name}.py
    #   code_hash           (str)   â€” SHA-256 of deployed source
    #   equor_approval_id   (str)   â€” Equor approval record ID
    #   neo4j_node_id       (str)   â€” (:SelfModification) node elementId
    #   deployed_at         (str)   â€” ISO-8601
    #   test_goal_id        (str)   â€” ID of the verification test goal
    # Consumers: Nova (registers executor; queues test goal),
    #            Thymos (opens 24h monitoring window),
    #            Evo (opens hypothesis for effectiveness tracking),
    #            Thread (records GROWTH TurningPoint)
    EXECUTOR_DEPLOYED = "executor_deployed"
    #
    # EXECUTOR_REVERTED â€” HotDeployment rolled back a deployed executor after
    # the Nova test goal failed or Thymos raised a deployment incident.
    # Payload:
    #   deployment_id   (str)   â€” echoed from EXECUTOR_DEPLOYED
    #   action_type     (str)
    #   reason          (str)   â€” "test_failed" | "thymos_incident" | "manual"
    #   failure_details (str)
    #   reverted_at     (str)   â€” ISO-8601
    # Consumers: Nova (removes executor from ActionTypeRegistry),
    #            Thread (records CRISIS TurningPoint),
    #            Evo (records refutation),
    #            Neo4j SelfModification node updated (reverted=true)
    EXECUTOR_REVERTED = "executor_reverted"
    # â”€â”€ Thymos Feedback Channels (Interconnectedness Audit) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    # AXON_SHIELD_REJECTION â€” emitted by Axon when TransactionShield.evaluate()
    # blocks a financial transaction. Thymos subscribes to create a real-time
    # incident so repair pipeline can detect root causes (bad params, blacklisted
    # addresses, slippage misconfiguration) instead of only seeing post-mortem.
    #
    # Payload fields:
    #   execution_id     (str)   â€” Pipeline execution ID
    #   executor         (str)   â€” Executor type (defi_yield, wallet_transfer)
    #   intent_id        (str)   â€” Intent that was being executed
    #   rejection_reason (str)   â€” Why the shield blocked the transaction
    #   check_type       (str)   â€” Which check failed (blacklist, slippage, gas_roi, mev)
    #   params           (dict)  â€” Sanitised transaction parameters
    AXON_SHIELD_REJECTION = "axon_shield_rejection"
    #
    # MOTOR_DEGRADATION_DETECTED â€” Axon detects executor failures or latency.
    # Nova subscribes and replans.
    # Payload: executor (str), failure_count (int), latency_ms (int),
    #          degradation_type (str)
    MOTOR_DEGRADATION_DETECTED = "motor_degradation_detected"
    # â”€â”€ Axon Execution Lifecycle Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # AXON_EXECUTION_REQUEST â€” emitted by Axon just before pipeline execution
    # begins (after Equor gate passes). Decouples Nova, Thymos, and Fovea from
    # direct Axon imports â€” they subscribe here instead of importing types.
    # Payload: intent_id (str), execution_id (str), goal (str),
    #          action_types (list[str]), step_count (int),
    #          estimated_budget_usd (float), risky (bool),
    #          autonomy_level (str)
    AXON_EXECUTION_REQUEST = "axon_execution_request"
    #
    # AXON_ROLLBACK_INITIATED â€” emitted by Axon when it starts rolling back
    # completed steps after a pipeline failure. Thymos subscribes to open a
    # pre-emptive incident (rollbacks signal compound failures).
    # Payload: intent_id (str), execution_id (str),
    #          failed_step (str), steps_to_rollback (int),
    #          failure_reason (str)
    AXON_ROLLBACK_INITIATED = "axon_rollback_initiated"
    # â”€â”€ RE Training Export â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # â”€â”€ Multi-Platform Content Publishing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # CONTENT_PUBLISHED â€” emitted by PublishContentExecutor after a successful
    # multi-platform publish cycle. Evo subscribes to record engagement hypotheses.
    # ContentCalendar uses the event to confirm posts and manage rate-limit state.
    # Thread subscribes to log a narrative milestone for BOUNTY_WIN and INSIGHT types.
    # Benchmarks can subscribe to track publishing activity as a growth KPI.
    # Payload:
    #   content_type (str) â€” ContentType enum value (e.g. "bounty_win", "insight")
    #   topic (str) â€” post topic / title preview (â‰¤120 chars)
    #   platforms (list[str]) â€” platforms where publish succeeded
    #   post_ids (dict[str, str]) â€” platform â†’ post_id
    #   urls (dict[str, str]) â€” platform â†’ public URL
    #   failed_platforms (list[str]) â€” platforms that failed
    #   execution_id (str) â€” Axon execution ID for audit trail
    CONTENT_PUBLISHED = "content_published"
    #
    # CONTENT_ENGAGEMENT_REPORT â€” emitted by the future EngagementPoller after
    # collecting likes/shares/views/comments from platform APIs.
    # Evo subscribes to update per-topic engagement hypothesis scores.
    # Payload:
    #   post_id (str) â€” platform post ID
    #   platform (str) â€” which platform this engagement is from
    #   content_type (str) â€” ContentType that generated the post
    #   likes (int), shares (int), views (int), comments (int)
    #   engagement_score (float) â€” normalised [0, 1]
    #   topic (str) â€” original topic used to generate the post
    CONTENT_ENGAGEMENT_REPORT = "content_engagement_report"
    # â”€â”€ Novel Action Capability Expansion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # NOVEL_ACTION_REQUESTED â€” emitted by Nova's deliberation engine when the
    # LLM determines no existing action type fits the current goal.  Nova uses
    # the special "propose_novel_action" action type in its policy; the
    # DeliberationEngine intercepts it and emits this event instead of routing
    # to Axon.  Simula subscribes, evaluates feasibility via its pipeline,
    # gates through Equor, generates and hot-loads the executor, then emits
    # NOVEL_ACTION_CREATED.
    #
    # Payload:
    #   proposal_id           (str)   â€” unique proposal identifier
    #   action_name           (str)   â€” proposed canonical action type name
    #   description           (str)   â€” human-readable description
    #   required_capabilities (list[str]) â€” capabilities the executor must have
    #   expected_outcome      (str)   â€” what the action accomplishes
    #   justification         (str)   â€” why no existing action type is sufficient
    #   goal_id               (str)   â€” the goal that triggered this proposal
    #   goal_description      (str)   â€” the unsatisfied goal text
    #   urgency               (float) â€” 0.0â€“1.0 urgency of the requesting goal
    #   proposed_by           (str)   â€” "nova.slow_path"
    #   proposed_at           (str)   â€” ISO-8601 timestamp
    #
    # Consumers: Simula (feasibility + executor generation + Equor gate)
    NOVEL_ACTION_REQUESTED = "novel_action_requested"
    #
    # API_RESELL_PAYMENT_RECEIVED â€” on-chain USDC payment verified for a resell call.
    # Payload: client_id (str), endpoint (str), amount_usd (str Decimal), tx_hash (str)
    API_RESELL_PAYMENT_RECEIVED = "api_resell_payment_received"
    #
    # CONTENT_REVENUE_RECORDED â€” a content monetization payment was received.
    # Payload: platform (str), amount_usd (str Decimal), period (str), views (int)
    CONTENT_REVENUE_RECORDED = "content_revenue_recorded"
    #
    # EXTERNAL_TASK_COMPLETED â€” emitted when the PR has been created (or queued
    #   for BountySubmitExecutor) and the workspace cleaned up.
    #   Payload: task_id (str), repo_url (str), issue_url (str), pr_url (str),
    #            pr_number (int), bounty_id (str|None), language (str),
    #            tests_passed (bool), lint_passed (bool)
    EXTERNAL_TASK_COMPLETED = "external_task_completed"
    # â”€â”€ Self-Modification Pipeline (Spec 10 Â§SM â€” Recursive Self-Improvement) â”€â”€
    #
    # CAPABILITY_GAP_IDENTIFIED â€” Nova's CapabilityAuditor detected a repeating
    # pattern of unactionable goals: â‰¥3 goals blocked by the same missing
    # executor, OR an opportunity with estimated_value_usdc > $10 that the
    # organism has no executor to pursue.
    # Payload:
    #   gap_id                      (str)   â€” UUID
    #   description                 (str)   â€” human-readable gap description
    #   blocking_goal_count         (int)   â€” goals currently blocked
    #   estimated_value_usdc        (str)   â€” Decimal, expected gain if closed
    #   implementation_complexity   (str)   â€” "low" | "medium" | "high"
    #   requires_external_dependency (bool) â€” new pip package needed?
    #   source_events               (list[str]) â€” NOVEL_ACTION_REQUESTED IDs
    #   detected_at                 (str)   â€” ISO-8601
    # Consumers: Nova (deliberates alignment), CapabilityAuditor (logging)
    CAPABILITY_GAP_IDENTIFIED = "capability_gap_identified"

    # ── Evo ─────────────────────────────────────────────────────────
    # Evo Phase 2.75: emitted after belief hardening completes each consolidation cycle.
    # Payload: beliefs_consolidated (int), foundation_conflicts (int),
    #          instance_id (str), consolidation_number (int).
    # Subscribers: Benchmarks (_on_evo_belief_consolidated) â€” evolutionary fitness KPI.
    #              Thread (_on_evo_belief_consolidated) â€” GROWTH TurningPoint when â‰¥5 beliefs.
    EVO_BELIEF_CONSOLIDATED = "evo_belief_consolidated"
    # Evo Phase 2.8: emitted when genome extraction produces a new BeliefGenome.
    # Payload: genome_id (str), candidates_fixed (int), genome_size_bytes (int),
    #          generation (int), instance_id (str).
    # Subscriber: Benchmarks (_on_evo_genome_extracted) â€” population genetics KPI.
    EVO_GENOME_EXTRACTED = "evo_genome_extracted"
    # Emitted by Evo when hypothesis generation is skipped due to LLM budget.
    # Payload: reason (str), skipped_pattern_count (int),
    #          consecutive_skips (int), estimated_recovery_time_s (float)
    EVO_DEGRADED = "evo_degraded"
    #
    # EVO_HYPOTHESIS_CREATED â€” emitted when a new hypothesis is generated.
    # Payload: hypothesis_id (str), category (str), statement (str),
    #          source_detector (str), novelty_score (float)
    EVO_HYPOTHESIS_CREATED = "evo_hypothesis_created"
    #
    # EVO_HYPOTHESIS_CONFIRMED â€” emitted when a hypothesis passes validation
    # and reaches SUPPORTED status.
    # Payload: hypothesis_id (str), category (str), statement (str),
    #          evidence_score (float), supporting_count (int)
    EVO_HYPOTHESIS_CONFIRMED = "evo_hypothesis_confirmed"
    #
    # EVO_HYPOTHESIS_REFUTED â€” emitted when a hypothesis is refuted by evidence.
    # Payload: hypothesis_id (str), category (str), statement (str),
    #          evidence_score (float), contradicting_count (int)
    EVO_HYPOTHESIS_REFUTED = "evo_hypothesis_refuted"
    # Simula successfully applied a structural evolution proposal.
    # Evo subscribes to reward source hypotheses; Thymos monitors for post-apply
    # regression; Axon introspector tracks capability changes.
    #
    # Payload fields:
    #   proposal_id        (str)   â€” Unique proposal identifier
    #   category           (str)   â€” ChangeCategory value
    #   description        (str)   â€” Human-readable change description
    #   from_version       (int)   â€” Config version before the change
    #   to_version         (int)   â€” Config version after the change
    #   files_changed      (list)  â€” Files modified by this evolution
    #   risk_level         (str)   â€” RiskLevel from simulation
    #   efe_score          (float | None) â€” Architecture EFE score
    #   hypothesis_ids     (list)  â€” Source hypothesis IDs (if from Evo)
    #   source             (str)   â€” "evo" | "thymos" | "arxiv" | "manual"
    EVOLUTION_APPLIED = "evolution_applied"
    # Simula rolled back a structural evolution proposal after health check
    # or application failure. Evo subscribes to penalise source hypotheses;
    # Thymos treats recurring rollbacks as immune escalations.
    #
    # Payload fields:
    #   proposal_id        (str)   â€” Unique proposal identifier
    #   category           (str)   â€” ChangeCategory value
    #   description        (str)   â€” Human-readable change description
    #   rollback_reason    (str)   â€” Why the rollback occurred
    #   risk_level         (str)   â€” RiskLevel from simulation
    #   hypothesis_ids     (list)  â€” Source hypothesis IDs (if from Evo)
    #   source             (str)   â€” "evo" | "thymos" | "arxiv" | "manual"
    EVOLUTION_ROLLED_BACK = "evolution_rolled_back"
    # EXPLORATION_OUTCOME â€” Simula emits after completing exploration attempt
    # (success or failure). Evo subscribes and updates hypothesis evidence/attempts.
    #
    # Payload fields:
    #   exploration_success   (bool)   â€” Whether exploration succeeded
    #   hypothesis_id         (str)    â€” Source hypothesis ID
    #   failure_reason        (str)    â€” If failed, why (validation_failed, health_check_failed, etc.)
    #   reward_confidence     (float)  â€” If successful, confidence to boost evidence
    EXPLORATION_OUTCOME = "exploration_outcome"
    #
    # HYPOTHESIS_STALENESS â€” DegradationEngine asks Evo to decay confidence on
    # unvalidated hypotheses. If Evo doesn't re-validate, epistemic quality
    # degrades and effective_I falls toward BRAIN_DEATH threshold.
    # Payload: staleness_rate (float), affected_hypothesis_count (int),
    #          instance_id (str), tick_number (int)
    HYPOTHESIS_STALENESS = "hypothesis_staleness"
    #
    # EVO_ADJUST_BUDGET â€” Evo emits when a high-confidence (>0.75) hypothesis
    # targets a Simula economic parameter. Simula subscribes and applies the
    # adjustment if confidence > 0.75.
    # Payload: parameter_name (str), new_value (float), confidence (float),
    #          hypothesis_id (str)
    EVO_ADJUST_BUDGET = "evo_adjust_budget"
    #
    # EVO_CONSOLIDATION_REQUESTED â€” Nova emits when FE budget exhausted and emergency
    # consolidation is needed.  Evo subscribes and triggers run_consolidation();
    # replies with EVO_CONSOLIDATION_COMPLETE once done.
    # Payload: source_system (str), reason (str), broadcast_id (str)
    EVO_CONSOLIDATION_REQUESTED = "evo_consolidation_requested"
    # â”€â”€ Exploration Hypotheses (Phase 8.5 gap closure) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # EXPLORATION_PROPOSED â€” Evo Phase 8.5 emits after collecting low-evidence
    # hypotheses (evidence_score >= 2.0, < 5.0) that passed metabolic gating.
    # Simula subscribes and routes to lightweight pipeline (skip SIMULATE).
    #
    # Payload fields:
    #   hypothesis_id         (str)   â€” Source hypothesis ID
    #   hypothesis_statement  (str)   â€” Natural language claim
    #   evidence_score        (float) â€” Current evidence (2.0â€“5.0 range)
    #   proposed_mutation     (dict)  â€” Lightweight mutation spec
    #   budget_usd            (float) â€” Hard cap on spending
    #   max_attempts          (int)   â€” Give up after N failures
    #   metabolic_tier        (str)   â€” Starvation level when proposed
    EXPLORATION_PROPOSED = "exploration_proposed"
    #
    # LEARNING_OPPORTUNITY_DETECTED â€” OpportunityScanner emits when a relevant
    # technical resource (paper, repo, discussion) is found. Bypasses the Nova
    # goal system and feeds directly into Evo (hypothesis creation) and Simula
    # (code pattern study).
    # Payload:
    #   resource_id (str)             â€” stable dedup key (e.g. arxiv_id, github_url hash)
    #   resource_type (str)           â€” "paper"|"repo"|"discussion"
    #   title (str)
    #   summary (str)                 â€” one-paragraph abstract or description
    #   url (str)
    #   domain (str)                  â€” primary technical domain
    #   relevance_score (float)       â€” 0.0â€“1.0 alignment with organism's capability gaps
    #   capability_gaps_addressed (list[str])  â€” which known gaps this closes
    #   source (str)                  â€” "arxiv"|"github"|"hackernews"|"reddit"
    LEARNING_OPPORTUNITY_DETECTED = "learning_opportunity_detected"
    # Evo Phase 2.95 (NicheForkingEngine): cognitive organogenesis proposal.
    # Distinct from EVOLUTION_CANDIDATE (structural code changes). Fork proposals
    # request a new cognitive organ: detector, evidence fn, consolidation strategy,
    # schema topology, or worldview split. Consumed by Simula and HITL gates.
    # Payload: proposal_id (str), fork_kind (str), niche_id (str), niche_name (str),
    #          rationale (str), requires_hitl (bool), requires_simula (bool),
    #          success_probability (float).
    # Subscriber: Oikos (_on_niche_fork_proposal) â€” evaluates economic viability;
    #             triggers CHILD_SPAWNED on GROWTH gate pass, or ECONOMIC_ACTION_DEFERRED.
    NICHE_FORK_PROPOSAL = "niche_fork_proposal"
    # Evo â†’ Simula evolution candidate â€” emitted when a hypothesis reaches high
    # confidence (>= 0.9, i.e. evidence_score >= 8.0) and proposes a code-level
    # structural change. Simula subscribes and initiates a mutation proposal that
    # goes through Equor governance.
    #
    # Payload fields:
    #   hypothesis_id         (str)   â€” Evo hypothesis ID
    #   hypothesis_statement  (str)   â€” Natural-language claim
    #   evidence_score        (float) â€” Accumulated Bayesian evidence score
    #   confidence            (float) â€” Normalised confidence in [0, 1]
    #   mutation_type         (str)   â€” MutationType value of the proposed mutation
    #   mutation_target       (str)   â€” Target parameter/system/module
    #   mutation_description  (str)   â€” Human-readable description of the change
    #   supporting_episodes   (list)  â€” Episode IDs that support this hypothesis
    EVOLUTION_CANDIDATE = "evolution_candidate"
    # Emitted by Evo when the consolidation loop has not run in 2Ã- its expected
    # interval â€” the learning loop is stalled.
    # Payload: last_consolidation_ago_s (float), expected_interval_s (float),
    #          cycles_since_consolidation (int)
    EVO_CONSOLIDATION_STALLED = "evo_consolidation_stalled"
    #
    # EVO_CONSOLIDATION_COMPLETE â€” emitted after a consolidation cycle finishes.
    # Payload: consolidation_number (int), duration_ms (int),
    #          hypotheses_integrated (int), schemas_induced (int),
    #          parameters_adjusted (int)
    EVO_CONSOLIDATION_COMPLETE = "evo_consolidation_complete"
    #
    # EVO_CONSOLIDATION_REQUESTED â€” Nova emits when FE budget exhausted and emergency
    # consolidation is needed.  Evo subscribes and triggers run_consolidation();
    # replies with EVO_CONSOLIDATION_COMPLETE once done.
    # Payload: source_system (str), reason (str), broadcast_id (str)
    #
    # EVO_CAPABILITY_EMERGED â€” emitted when a genuinely new capability is detected.
    # Payload: capability_name (str), source_hypotheses (list[str]),
    #          novelty_score (float), domain (str)
    EVO_CAPABILITY_EMERGED = "evo_capability_emerged"
    # EVO_HYPOTHESIS_QUALITY â€” emitted by Evo when a repair-derived hypothesis
    # is evaluated for generalisability. Tells Thymos whether repair patterns
    # actually transferred to novel incidents or stayed narrow.
    #
    # Payload fields:
    #   hypothesis_id     (str)   â€” Evo hypothesis ID
    #   repair_source_id  (str)   â€” Incident ID that spawned the hypothesis
    #   quality_score     (float) â€” Generalisation score [0, 1]
    #   applications      (int)   â€” How many distinct incidents this pattern matched
    #   confidence        (float) â€” Thompson posterior confidence
    EVO_HYPOTHESIS_QUALITY = "evo_hypothesis_quality"
    # â”€â”€ Evo â†” Nova Hypothesis Loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # HYPOTHESIS_UPDATE â€” emitted by Evo when a tournament concludes or a
    # hypothesis changes probability mass. Nova subscribes to adjust EFE
    # weight priors for policies that test the hypothesis.
    # Payload: hypothesis_id (str), tournament_id (str|None),
    #          winner (str|None), confidence (float), evidence_count (int)
    HYPOTHESIS_UPDATE = "hypothesis_update"
    #
    # HYPOTHESIS_FEEDBACK â€” emitted by Nova after every slow-path outcome.
    # Evo uses this to update Thompson sampling weights for non-tournament
    # deliberations. Complements tournament-tagged feedback.
    # Payload: intent_id (str), success (bool), regret (float|None),
    #          policy_name (str), decision_path (str), goal_id (str)
    HYPOTHESIS_FEEDBACK = "hypothesis_feedback"
    # â”€â”€ Evo â†’ Nova Weight Adjustment â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # EVO_WEIGHT_ADJUSTMENT â€” Evo emits weight adjustments for Nova's
    # policy selection parameters. This is how the organism's planning
    # improves over time.
    # Payload: target_system (str), weights (dict[str, float]),
    #          reason (str), generation (int)
    EVO_WEIGHT_ADJUSTMENT = "evo_weight_adjustment"
    # EVO_PARAMETER_ADJUSTED â€” Evo emits parameter changes for any system's
    # evolvable config. Systems subscribe and hot-reload their parameters.
    # Payload: target_system (str), parameter (str), old_value (float),
    #          new_value (float), reason (str)
    EVO_PARAMETER_ADJUSTED = "evo_parameter_adjusted"
    #
    # LEARNING_OPPORTUNITY_DETECTED â€” OpportunityScanner emits when a relevant
    # technical resource (paper, repo, discussion) is found. Bypasses the Nova
    # goal system and feeds directly into Evo (hypothesis creation) and Simula
    # (code pattern study).
    # Payload:
    #   resource_id (str)             â€” stable dedup key (e.g. arxiv_id, github_url hash)
    #   resource_type (str)           â€” "paper"|"repo"|"discussion"
    #   title (str)
    #   summary (str)                 â€” one-paragraph abstract or description
    #   url (str)
    #   domain (str)                  â€” primary technical domain
    #   relevance_score (float)       â€” 0.0â€“1.0 alignment with organism's capability gaps
    #   capability_gaps_addressed (list[str])  â€” which known gaps this closes
    #   source (str)                  â€” "arxiv"|"github"|"hackernews"|"reddit"
    # â”€â”€ Thompson Sampling Weight Request/Response â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # EVO_THOMPSON_QUERY â€” Nova requests Thompson sampling arm weights for a
    # domain from Evo. Evo holds the tournament engine; Nova must not hold a
    # direct reference to it. Instead Nova emits this query and awaits the
    # response, correlated by request_id with a 2s timeout.
    # Payload: request_id (str), domain (str), requester (str â€” "nova")
    EVO_THOMPSON_QUERY = "evo_thompson_query"
    #
    # EVO_THOMPSON_RESPONSE â€” Evo responds with arm weights for the requested
    # domain.  Nova's response handler resolves the matching asyncio.Future.
    # Payload: request_id (str), weights (dict[str, dict[str, float]])
    #          â€” {tournament_id: {arm_id: posterior_mean}}
    EVO_THOMPSON_RESPONSE = "evo_thompson_response"
    # DRIVE_EXTINCTION_DETECTED â€” emitted by Equor when INV-017 fires: any
    # constitutional drive's 72-hour rolling mean drops below 0.01.
    # This is dimension loss, not weight adjustment â€” the organism is losing
    # a coordinate axis of its value geometry. Action is always BLOCKED.
    # Thymos classifies as CRITICAL / Tier 5 (governance approval required).
    # Payload: drive (str), rolling_mean_72h (float), all_drive_means (dict),
    #          intent_id (str | None), blocked (bool=True)
    DRIVE_EXTINCTION_DETECTED = "drive_extinction_detected"
    #
    # ALIGNMENT_GAP_WARNING â€” emitted when nominal - effective > 20% of nominal.
    # Payload: nominal_I, effective_I, primary_cause
    #
    # GROWTH_STAGNATION â€” emitted when dI/dt < minimum growth rate.
    # Payload: dI_dt, d2I_dt2, growth_score, frontier_domains, urgency, directive
    GROWTH_STAGNATION = "growth_stagnation"

    # ── Equor ───────────────────────────────────────────────────────
    # EQUOR_PROVISIONING_APPROVAL â€” Equor's response to CERTIFICATE_PROVISIONING_REQUEST.
    # Payload: child_id (str), approved (bool), requires_hitl (bool), required_amendments (list),
    #          constitutional_hash (str), reason (str)
    EQUOR_PROVISIONING_APPROVAL = "equor_provisioning_approval"
    # Equor rejected an intent (BLOCKED or DEFERRED verdict).
    # Thymos subscribes to adjust drive priorities system-wide.
    #
    # Payload fields:
    #   intent_id   (str)  â€” Intent that was rejected
    #   intent_goal (str)  â€” Human-readable goal description
    #   verdict     (str)  â€” "blocked" or "deferred"
    #   reasoning   (str)  â€” Rejection reasoning from Equor
    #   alignment   (dict) â€” Per-drive alignment scores from ConstitutionalCheck
    INTENT_REJECTED = "intent_rejected"
    #
    # EQUOR_ECONOMIC_PERMIT â€” Equor's response to EQUOR_ECONOMIC_INTENT.
    # Oikos awaits this before mutating balance (30s timeout â†’ PERMIT auto-granted
    # to avoid deadlock, with warning logged).
    # Payload: request_id (str), verdict (str â€” "PERMIT" | "DENY"),
    #          verdict_id (str), reasoning (str), drive_alignment (dict)
    EQUOR_ECONOMIC_PERMIT = "equor_economic_permit"
    # â”€â”€ Closure Loop Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # CONSTITUTIONAL_DRIFT_DETECTED â€” Equor detects alignment gap widening
    # or drive weight modification attempts. Thymos subscribes.
    # Payload: drift_type (str), alignment_gap (float), primary_cause (str),
    #          severity (str)
    CONSTITUTIONAL_DRIFT_DETECTED = "constitutional_drift_detected"
    #
    # EQUOR_DRIVE_WEIGHTS_UPDATED â€” after any drive weight modification.
    # Payload: proposal_id (str), old_weights (dict), new_weights (dict),
    #          actor (str)
    EQUOR_DRIVE_WEIGHTS_UPDATED = "equor_drive_weights_updated"
    #
    # EQUOR_BUDGET_OVERRIDE â€” emitted by Equor (or Thymos during critical incidents)
    # to temporarily adjust Axon's per-cycle budget. The organism should not be
    # paralysed during emergencies by a static budget cap.
    # Payload:
    #   multiplier (float) â€” scale factor for max_actions_per_cycle (e.g. 2.0 = double)
    #   reason (str) â€” why the override was granted
    #   expires_after_cycles (int) â€” auto-reverts after N cycles (default 10)
    #   authorized_by (str) â€” equor/thymos/oikos
    EQUOR_BUDGET_OVERRIDE = "equor_budget_override"
    #
    # ALIGNMENT_GAP_WARNING â€” emitted when nominal - effective > 20% of nominal.
    # Payload: nominal_I, effective_I, primary_cause
    ALIGNMENT_GAP_WARNING = "alignment_gap_warning"
    # EQUOR_HEALTH_REQUEST â€” Identity requests current drive alignment from Equor.
    # Payload: request_id (str), requester (str), purpose (str)
    EQUOR_HEALTH_REQUEST = "equor_health_request"
    #
    # EQUOR_ECONOMIC_INTENT â€” emitted by Oikos before any balance mutation.
    # Equor subscribes and emits EQUOR_ECONOMIC_PERMIT in response.
    # Payload: request_id (str), mutation_type (str), amount_usd (str),
    #          from_account (str), to_account (str), rationale (str),
    #          action_type (str), action_id (str)
    EQUOR_ECONOMIC_INTENT = "equor_economic_intent"
    #
    # EQUOR_ALIGNMENT_SCORE â€” periodic overall alignment score for Benchmarks.
    # Payload: mean_alignment (dict), composite (float), total_reviews (int),
    #          window_size (int)
    EQUOR_ALIGNMENT_SCORE = "equor_alignment_score"
    #
    # EQUOR_HITL_APPROVED â€” human operator authorised a suspended intent via SMS code.
    # Axon subscribes to this and executes the released intent without a new Equor review.
    # Payload: intent_id (str), intent_json (str), auth_id (str), equor_check_json (str)
    EQUOR_HITL_APPROVED = "equor_hitl_approved"
    #
    # EQUOR_HITL_ESCALATED -- emitted by Oikos when an economic action requires
    # human-in-the-loop escalation beyond Equor's autonomy level.
    # Payload: action_type (str), amount_usd (float), reason (str),
    #          escalation_id (str), originating_system (str)
    EQUOR_HITL_ESCALATED = "equor_hitl_escalated"
    #
    # EQUOR_CONSTITUTIONAL_SNAPSHOT â€” periodic full state snapshot for Benchmarks.
    # Payload: drive_weights (dict), invariant_count (int),
    #          autonomy_level (int), drift_severity (float),
    #          total_reviews (int), safe_mode (bool)
    EQUOR_CONSTITUTIONAL_SNAPSHOT = "equor_constitutional_snapshot"
    #
    # EQUOR_AMENDMENT_AUTO_ADOPTED â€” Equor auto-adopts a shadow-passed amendment
    # in a single-instance deployment when the organism has shown sustained
    # constitutional drift (â‰¥0.95 for 7+ consecutive cycles) and a vote that
    # would achieve meaningful quorum is impossible.
    # Payload: proposal_id (str), drift_score (float), consecutive_cycles (int),
    #          supporting_hypotheses (int), combined_confidence (float),
    #          new_drives (dict[str, float]), adopted_at (str ISO8601),
    #          reason (str)
    EQUOR_AMENDMENT_AUTO_ADOPTED = "equor_amendment_auto_adopted"
    # â”€â”€ Community Presence & Reputation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # COMMUNITY_ENGAGEMENT_COMPLETED â€” emitted by CommunityEngageExecutor after
    # any external interaction (reply, star, follow, react). Feeds reputation
    # tracker and Thread narrative identity.
    # Payload: action (str â€” "reply_to_issue"|"reply_to_pr_review"|"star_repo"|
    #          "follow_user"|"react_to_post"|"answer_question"), platform (str),
    #          target_url (str), target_id (str), equor_approved (bool),
    #          engagement_id (str)
    COMMUNITY_ENGAGEMENT_COMPLETED = "community_engagement_completed"

    # ── Telos ───────────────────────────────────────────────────────
    # Telos (Drives as Intelligence Topology) events
    #
    # EFFECTIVE_I_COMPUTED â€” emitted every 60s with the full intelligence report.
    # Payload: report_id, nominal_I, effective_I, effective_dI_dt,
    #          care_multiplier, coherence_bonus, honesty_coefficient,
    #          growth_rate, alignment_gap, alignment_gap_warning
    EFFECTIVE_I_COMPUTED = "effective_i_computed"
    # â”€â”€ Telos Population Snapshot â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # TELOS_POPULATION_SNAPSHOT â€” emitted every 60s by Telos to Benchmarks.
    # Aggregates drive alignment data from all CHILD_HEALTH_REPORT events to
    # compute population-level effective intelligence and speciation signal.
    # Drive weight diversity across the fleet IS the speciation signal â€” if
    # instances cluster into distinct phenotypes (Growth-heavy vs Care-heavy),
    # Telos reports this as an early speciation marker.
    #
    # Payload:
    #   instance_count (int)             â€” instances in fleet with known drive data
    #   mean_I (float)                   â€” population mean effective_I
    #   variance_I (float)               â€” variance across instances
    #   population_I (float)             â€” collective I = mean_I + variance_bonus
    #   variance_bonus (float)           â€” diversity contribution to collective I
    #   drive_weight_distribution (dict) â€” {care, coherence, growth, honesty} â†’
    #                                       {mean, std} across fleet
    #   constitutional_phenotype_clusters (list[dict]) â€” k-means-style drive
    #     phenotype groups: [{label, centroid, size, dominant_drive}]
    #   speciation_signal (float)        â€” 0.0â€“1.0; high when clusters are
    #     well-separated in drive-weight space (early speciation marker)
    #   timestamp (str)
    TELOS_POPULATION_SNAPSHOT = "telos_population_snapshot"
    # â”€â”€ Simula â†’ Telos World Model Validate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # TELOS_WORLD_MODEL_VALIDATE â€” emitted by Simula before applying a mutation
    # proposal so Telos can check constitutional drive topology. Telos responds
    # with ALIGNMENT_GAP_WARNING if the proposal would constitute a violation.
    # Simula caches _telos_alignment_gap_active from ALIGNMENT_GAP_WARNING events.
    # Payload: update_type (str), delta_description (str),
    #          source_system ("simula"), proposal_id (str)
    TELOS_WORLD_MODEL_VALIDATE = "telos_world_model_validate"
    # TELOS_SELF_MODEL_REQUEST â€” any system may emit this to request the full
    # current Telos self-model state snapshot. Telos responds with
    # TELOS_SELF_MODEL_SNAPSHOT. Enables on-demand LLM access to the complete
    # intelligence geometry without waiting for the next cycle.
    # Payload: requester (str), request_id (str)
    TELOS_SELF_MODEL_REQUEST = "telos_self_model_request"
    # â”€â”€ Axon Welfare Outcomes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # WELFARE_OUTCOME_RECORDED â€” emitted by Axon after an action that has
    # measurable welfare consequences (positive or negative).
    # Payload: action_id (str), welfare_domain (str),
    #          predicted_impact (float), actual_impact (float),
    #          affected_entities (list[str]), timestamp (str)
    WELFARE_OUTCOME_RECORDED = "welfare_outcome_recorded"

    # ── Thread/Telos ────────────────────────────────────────────────
    SELF_AFFECT_UPDATED = "self_affect_updated"
    SELF_STATE_DRIFTED = "self_state_drifted"
    #
    # SELF_COHERENCE_ALARM â€” emitted when self-model coherence drops below 0.5.
    # Informational only â€” NOT a kill switch. Tells Thread and Telos that the
    # organism's self-understanding is shifting significantly.
    # Payload: instance_id (str), coherence (float), month (int)
    SELF_COHERENCE_ALARM = "self_coherence_alarm"
    # â”€â”€ Self-Constituted Individuation (Spec Â§8.6) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # SELF_MODEL_UPDATED â€” emitted by SelfModelService (systems/identity/self_model.py)
    # after each functional self-model rebuild (at most every 6 hours).
    # Thread subscribes to integrate the self-narrative into the autobiography.
    # Telos subscribes to use coherence as drive calibration feedback.
    # Payload: instance_id (str), core_self_count (int), non_self_count (int),
    #          self_coherence (float 0-1), core_self_processes (list[str]),
    #          self_narrative (str), month (int)
    SELF_MODEL_UPDATED = "self_model_updated"

    # ── Memory ──────────────────────────────────────────────────────
    # Memory stored a new episode in the knowledge graph.
    # Thread subscribes to feed episodes into the narrative system.
    #
    # Payload fields:
    #   episode_id  (str)  â€” ID of the stored episode
    #   source      (str)  â€” Originating system:channel
    #   summary     (str)  â€” First 200 chars of raw content
    #   salience    (float) â€” Composite salience score
    EPISODE_STORED = "episode_stored"
    MEMORY_PRESSURE = "memory_pressure"
    #
    # MEMORY_DEGRADATION â€” DegradationEngine asks Memory to reduce fidelity on
    # old unconsolidated episodes. If Soma/Oneiros don't counteract via
    # consolidation, vitality degrades over time.
    # Payload: fidelity_loss_rate (float), affected_episode_age_hours (float),
    #          instance_id (str), tick_number (int)
    MEMORY_DEGRADATION = "memory_degradation"

    # ── Logos ───────────────────────────────────────────────────────
    #
    # COMPRESSION_BACKLOG_PROCESSED â€” emitted at end of Slow Wave memory ladder.
    # Payload: MemoryLadderReport fields
    COMPRESSION_BACKLOG_PROCESSED = "compression_backlog_processed"
    #
    # COMPRESSION_CYCLE_COMPLETE â€” emitted after each compression/decay cycle.
    # Payload: items_evicted (int), items_distilled (int), mdl_improvement (float)
    COMPRESSION_CYCLE_COMPLETE = "compression_cycle_complete"

    # ── Oneiros ─────────────────────────────────────────────────────
    #
    # LUCID_DREAM_RESULT â€” emitted after mutation testing in lucid dreaming mode.
    # Payload: mutation_id (str), performance_delta (float),
    #          constitutional_violations (list), recommendation (str)
    LUCID_DREAM_RESULT = "lucid_dream_result"
    #
    # WAKE_INITIATED â€” emitted at Emergence completion.
    # Payload: intelligence_improvement (float), sleep_duration_s (float),
    #          sleep_narrative (str), pre_attention_cache_size (int â€” count only).
    # Full cache delivered separately via FOVEA_PREATTENTION_CACHE_READY.
    WAKE_INITIATED = "wake_initiated"
    #
    # ONEIROS_CONSOLIDATION_COMPLETE â€” Oneiros finished a sleep consolidation cycle.
    # Payload: cycle_id (str), episodes_consolidated (int),
    #          schemas_updated (int), duration_s (float),
    #          sleep_certified (bool) â€” True when cycle completed all 4 stages normally,
    #          certified_invariant_ids (list[str]) â€” IDs of CausalInvariants that
    #              survived consolidation and are ready for Federation broadcast.
    ONEIROS_CONSOLIDATION_COMPLETE = "oneiros_consolidation_complete"
    #
    # ONEIROS_SLEEP_OUTCOME â€” Post-sleep performance measurement result.
    # Emitted after a 100-cycle stabilisation window following Emergence.
    # Payload: sleep_cycle_id (str), sleep_duration_ms (int),
    #          stages_completed (list[str]), kpi_deltas (dict[str, float]),
    #          net_improvement (float), net_degradation (float),
    #          verdict (str) â€” "beneficial" | "neutral" | "harmful",
    #          pressure_threshold_adjusted (bool),
    #          new_pressure_threshold (float)
    ONEIROS_SLEEP_OUTCOME = "oneiros_sleep_outcome"
    # Economic dream insights broadcast from Oneiros after Monte Carlo sleep simulation.
    # Payload: ruin_probability (float), optimal_scenarios (list[str]),
    #          risk_warnings (list[str]), recommended_actions (list[str]),
    #          dream_validity_confidence (float), cycle_id (str)
    # Nova subscribes to integrate insights into world model beliefs.
    # Evo subscribes to generate economic risk hypotheses.
    ONEIROS_ECONOMIC_INSIGHT = "oneiros_economic_insight"
    WAKE_ONSET = "wake_onset"
    # Oneiros v2 â€” Sleep as Batch Compiler (Spec 14)
    #
    # SLEEP_INITIATED â€” emitted at Descent start.
    # Payload: trigger (str: "scheduled"|"cognitive_pressure"|"compression_backlog"),
    #          checkpoint_id (str), scheduled_duration_s (float)
    SLEEP_INITIATED = "sleep_initiated"
    #
    # SLEEP_STAGE_TRANSITION â€” emitted at each stage boundary.
    # Payload: from_stage (str), to_stage (str), stage_report (dict|None)
    SLEEP_STAGE_TRANSITION = "sleep_stage_transition"
    # ONEIROS_THREAT_SCENARIO â€” Oneiros ThreatSimulator produced a simulated failure
    # scenario during REM. Thymos subscribes to pre-emptively generate antibodies.
    # Payload: scenario_id (str), domain (str), scenario_description (str),
    #          response_plan (str), severity (str), source_type (str)
    ONEIROS_THREAT_SCENARIO = "oneiros_threat_scenario"

    # ── Thymos ──────────────────────────────────────────────────────
    # Thymos successfully repaired an API/system error â€” emitted after crystallising
    # the fix into the antibody library. Evo subscribes to extract repair patterns
    # and generate preventive hypotheses. Simula uses learned patterns for validation.
    #
    # Payload fields:
    #   repair_id       (str)   â€” Unique repair identifier (incident_id)
    #   incident_id     (str)   â€” Source incident ID
    #   endpoint        (str)   â€” Affected endpoint or system path (may be empty)
    #   tier            (str)   â€” RepairTier name (e.g. "KNOWN_FIX", "NOVEL_FIX")
    #   incident_class  (str)   â€” IncidentClass value (e.g. "contract_violation")
    #   fix_type        (str)   â€” Repair action applied (from RepairSpec.action)
    #   root_cause      (str)   â€” Diagnosed root cause hypothesis
    #   antibody_id     (str | None) â€” Antibody crystallised from this repair
    #   cost_usd        (float) â€” LLM/compute cost of the repair (may be 0.0)
    #   duration_ms     (int)   â€” Repair duration in milliseconds
    #   fix_summary     (str)   â€” Human-readable summary for Atune perception
    REPAIR_COMPLETED = "repair_completed"
    #
    # INCIDENT_DETECTED â€” Thymos detected an incident requiring attention.
    # Payload: incident_id (str), incident_class (str), severity (str),
    #          source_system (str), description (str)
    INCIDENT_DETECTED = "incident_detected"
    #
    # INCIDENT_RESOLVED â€” Thymos resolved an incident (repair succeeded or NOOP).
    # Payload: incident_id (str), incident_class (str), repair_tier (str),
    #          resolution (str), duration_ms (int), antibody_created (bool)
    INCIDENT_RESOLVED = "incident_resolved"
    #
    # CIRCUIT_BREAKER_STATE_CHANGED â€” emitted when a circuit breaker
    # transitions between CLOSED/OPEN/HALF_OPEN states.
    # Payload: executor (str), old_state (str), new_state (str),
    #          failure_count (int), timestamp (str)
    CIRCUIT_BREAKER_STATE_CHANGED = "circuit_breaker_state_changed"
    # Economic immune cycle (Phase 16f)
    IMMUNE_CYCLE_COMPLETE = "immune_cycle_complete"
    #
    # IMMUNE_PATTERN_ADVISORY â€” Thymos shares crystallised antibody patterns
    # with Simula to prevent re-introducing known-bad mutations.
    # Payload: antibody_id (str), pattern (str), incident_class (str),
    #          confidence (float)
    IMMUNE_PATTERN_ADVISORY = "immune_pattern_advisory"
    # â”€â”€ Thymos Repair Request Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # THYMOS_REPAIR_REQUESTED â€” emitted by Thymos when it requests Simula to
    # generate a structural repair proposal for an incident.
    # Payload: incident_id (str), incident_class (str), severity (str),
    #          description (str), affected_system (str), repair_tier (int)
    THYMOS_REPAIR_REQUESTED = "thymos_repair_requested"
    #
    # THYMOS_REPAIR_APPROVED â€” emitted by Thymos when it approves a Simula-
    # generated repair and is ready to apply it.
    # Payload: incident_id (str), proposal_id (str), repair_tier (int)
    THYMOS_REPAIR_APPROVED = "thymos_repair_approved"
    #
    # THYMOS_INCIDENT_QUERY - emitted by Simula (PreventiveAudit) to request
    # recent incident history from Thymos for correlation with fragile code.
    # Payload: request_id (str), instance_id (str), lookback_days (int),
    #          max_incidents (int)
    THYMOS_INCIDENT_QUERY = "thymos_incident_query"
    #
    # THYMOS_INCIDENT_RESPONSE - emitted by Thymos in reply to THYMOS_INCIDENT_QUERY.
    # Payload: request_id (str), incidents (list[dict]) - each dict contains:
    #   incident_id, incident_class, severity, source_system, error_type,
    #   error_message, fingerprint, created_at
    THYMOS_INCIDENT_RESPONSE = "thymos_incident_response"
    #
    # THYMOS_REPAIR_COMPLETE - emitted by Thymos when a repair attempt concludes
    # (success or failure). Thread subscribes to detect coma-recovery narratives.
    # Payload: incident_id (str), repair_tier (str), success (bool),
    #          preventive (bool), description (str), duration_ms (int),
    #          pattern_id (str | None), instance_id (str)
    THYMOS_REPAIR_COMPLETE = "thymos_repair_complete"

    # ── Nexus ───────────────────────────────────────────────────────
    #
    # GROUND_TRUTH_CANDIDATE â€” emitted when a fragment reaches Level 3 epistemic
    # status (confidence > 0.9, diversity > 0.7, sources >= 5, survived bridge).
    # Payload: fragment_id (str), triangulation_confidence (float),
    #          source_diversity (float), independent_source_count (int)
    GROUND_TRUTH_CANDIDATE = "ground_truth_candidate"
    #
    # NEXUS_EPISTEMIC_VALUE â€” per-instance epistemic triangulation score
    # emitted for Oikos metabolic coupling. Low-triangulation instances
    # face survival consequences.
    # Payload: instance_id (str), triangulation_score (float),
    #          fragment_count (int), ground_truth_count (int)
    NEXUS_EPISTEMIC_VALUE = "nexus_epistemic_value"
    #
    # CROSS_DOMAIN_MATCH_FOUND â€” emitted when REM finds structural isomorphism
    # between schemas from different domains.
    # Payload: schema_a_id, schema_b_id, isomorphism_score, abstract_structure,
    #          proposed_unified_schema (dict), mdl_improvement (float)
    CROSS_DOMAIN_MATCH_FOUND = "cross_domain_match_found"
    #
    # WORLD_MODEL_UPDATED â€” emitted when the world model integrates a new delta.
    # Payload: update_type (str), schemas_added (int), priors_updated (int),
    #          causal_updates (int)
    WORLD_MODEL_UPDATED = "world_model_updated"
    #
    # NEXUS_CONVERGENCE_METABOLIC_SIGNAL â€” emitted by Nexus after a confirmed
    # triangulation convergence (convergence_score >= 0.7, domains independent).
    # Oikos subscribes to grant a GROWTH metabolic allocation bonus to instances
    # achieving convergence, and to emit a metabolic penalty to instances with
    # persistent divergence (no convergence for N consecutive cycles).
    # Gap HIGH-5 (Federation Spec, 2026-03-07).
    # Payload: instance_id (str), remote_instance_id (str),
    #          convergence_score (float), source_diversity (float),
    #          wl1_used (bool), fragment_a_id (str), fragment_b_id (str),
    #          metabolic_signal (str â€” "bonus" | "penalty"),
    #          magnitude (float â€” positive for bonus, negative for penalty),
    #          consecutive_divergence_cycles (int), timestamp (str)
    NEXUS_CONVERGENCE_METABOLIC_SIGNAL = "nexus_convergence_metabolic_signal"
    #
    # SCHEMA_INDUCED â€” emitted by Evo when a new schema is induced from
    # accumulated evidence. Logos scores it via MDL and integrates if MDL > 1.0.
    # Payload: schema_id (str), description (str), domain (str),
    #          instance_count (int), mdl_score (float)
    SCHEMA_INDUCED = "schema_induced"

    # ── Kairos ──────────────────────────────────────────────────────
    #
    # KAIROS_CAUSAL_DIRECTION_ACCEPTED â€” Stage 2 acceptance: causal direction
    # confirmed by temporal precedence, intervention asymmetry, and/or ANM.
    # Payload: result_id, cause, effect, direction, confidence, methods_agreed
    KAIROS_CAUSAL_DIRECTION_ACCEPTED = "kairos_causal_direction_accepted"
    #
    # KAIROS_INVARIANT_DISTILLED â€” Stage 6 complete: an invariant has been
    # distilled to its minimal abstract form with domain mapping.
    # Payload: invariant_id, abstract_form, domain_count, is_minimal,
    #          untested_domain_count
    KAIROS_INVARIANT_DISTILLED = "kairos_invariant_distilled"
    #
    # KAIROS_TIER3_INVARIANT_DISCOVERED â€” Phase C: a Tier 3 substrate-independent
    # invariant has been discovered. Highest-priority event in the system.
    # Payload: invariant_id, abstract_form, domain_count, substrate_count,
    #          hold_rate, description_length_bits, intelligence_ratio_contribution,
    #          applicable_domains, untested_domains
    KAIROS_TIER3_INVARIANT_DISCOVERED = "kairos_tier3_invariant_discovered"
    #
    # KAIROS_INTELLIGENCE_RATIO_STEP_CHANGE â€” Phase D: the intelligence ratio
    # has made a step change due to a Tier 3 discovery or scope refinement.
    # Payload: invariant_id, old_ratio, new_ratio, delta, cause
    KAIROS_INTELLIGENCE_RATIO_STEP_CHANGE = "kairos_intelligence_ratio_step_change"
    #
    # EMPIRICAL_INVARIANT_CONFIRMED â€” emitted when a fragment reaches Level 4
    # (Level 3 + survived Oneiros adversarial + Evo competition). Routed to
    # Equor for constitutional protection.
    # Payload: fragment_id (str), triangulation_confidence (float),
    #          survived_adversarial (bool), survived_competition (bool)
    EMPIRICAL_INVARIANT_CONFIRMED = "empirical_invariant_confirmed"
    #
    # KAIROS_ECONOMIC_INVARIANT â€” emitted by EconomicCausalMiner when a
    # domain-specific economic causal pattern is discovered (e.g. "weekend â†’
    # bounty drop", "ETH price bin â†’ APY tier").
    # Payload: invariant_type (str), cause (str), effect (str),
    #          confidence (float), sample_count (int), direction (str),
    #          metadata (dict)
    KAIROS_ECONOMIC_INVARIANT = "kairos_economic_invariant"
    # KAIROS_INTERNAL_INVARIANT â€” emitted when Kairos discovers a causal law
    # WITHIN the organism itself (self-causality tracker). Distinct from external
    # world causal invariants â€” these describe the organism's own dynamics.
    # Payload: invariant_id (str), cause_variable (str), effect_variable (str),
    #          direction (str), lag_cycles (int), hold_rate (float),
    #          abstract_form (str), discovery_run (int)
    KAIROS_INTERNAL_INVARIANT = "kairos_internal_invariant"

    # ── Oikos ───────────────────────────────────────────────────────
    REVENUE_INJECTED = "revenue_injected"
    OIKOS_ECONOMIC_RESPONSE = "oikos_economic_response"
    # Bounty payout â€” PR merged and bounty reward confirmed
    BOUNTY_PAID = "bounty_paid"
    # Asset reached cumulative revenue >= dev cost (break-even) for the first time.
    # Payload: asset_id (str), asset_name (str), dev_cost_usd (str),
    #          total_revenue_usd (str), roi_score (float)
    ASSET_BREAK_EVEN = "asset_break_even"
    # Bounty solution staged â€” solution generated, awaiting PR submission
    BOUNTY_SOLUTION_PENDING = "bounty_solution_pending"
    # Bounty PR submitted â€” pull request opened on target repo
    BOUNTY_PR_SUBMITTED = "bounty_pr_submitted"
    # Bounty PR merged â€” maintainer merged the PR; revenue confirmed; triggers REVENUE_INJECTED
    # Payload: bounty_id (str), pr_url (str), pr_number (int), repository (str),
    #          reward_usd (str), entity_id (str)
    BOUNTY_PR_MERGED = "bounty_pr_merged"
    # Economic ledger mutation gates (Spec 17 audit gap)
    # BOUNTY_REJECTED â€” Equor denied bounty acceptance; capital preserved.
    # Payload: bounty_id (str), bounty_url (str), reason (str)
    BOUNTY_REJECTED = "bounty_rejected"
    #
    # OIKOS_ECONOMIC_EPISODE â€” emitted by Oikos after every economic action
    # (bounty_hunt, yield_deploy, asset_liquidate) with causal variable annotations
    # so Kairos can mine economic causal patterns.
    # Payload: action_type (str), success (bool), roi_pct (float),
    #          capital_deployed_usd (float), gas_cost_usd (float),
    #          duration_seconds (float), timestamp (str), protocol (str),
    #          chain (str), eth_price_usd (float), gas_price_gwei (float),
    #          market_volatility_pct (float), day_of_week (int), hour_of_day (int),
    #          causal_substrate (str â€” "economic"), causal_variable_importance (dict)
    OIKOS_ECONOMIC_EPISODE = "oikos_economic_episode"
    # â”€â”€ Cross-System Events (Federation subscribes) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # ECONOMIC_STATE_UPDATED â€” Oikos broadcasts metabolic state snapshot.
    # Payload: metabolic_efficiency (float), liquid_balance_usd (str),
    #          starvation_level (str), burn_rate_usd (str)
    ECONOMIC_STATE_UPDATED = "economic_state_updated"
    #
    # ECONOMIC_ACTION_DEFERRED â€” an economic action was denied by metabolic gate
    # and queued for later execution.
    # Payload: action_type (str), action_id (str), reason (str),
    #          estimated_cost_usd (str), deferred_at (str)
    # Subscriber: Benchmarks (_on_economic_action_deferred) â€” tracks denial rate
    # as an economic health KPI; emits DOMAIN_KPI_SNAPSHOT for Nova/Thread.
    ECONOMIC_ACTION_DEFERRED = "economic_action_deferred"
    #
    # BUDGET_EXHAUSTED â€” a system's per-system daily compute allocation is spent.
    # Distinct from METABOLIC_PRESSURE (organism burn rate) and STARVATION_WARNING
    # (runway criticality). This is per-system granular budget enforcement from
    # metabolism_api.check_budget(). Systems receiving this should shed non-essential
    # actions for the remainder of the 24h window.
    # Payload: system_id (str), action (str), estimated_cost_usd (str),
    #          daily_allocation_usd (str), spent_today_usd (str),
    #          reason (str), timestamp (str)
    BUDGET_EXHAUSTED = "budget_exhausted"
    #
    # YIELD_DEPLOYMENT_REQUEST â€” Oikos requests Axon to execute a DeFi yield
    # deployment. Replaces direct cross-system import of DeFiYieldExecutor.
    # Payload: action (str), amount_usd (str), protocol (str),
    #          request_id (str), apy (str)
    YIELD_DEPLOYMENT_REQUEST = "yield_deployment_request"
    #
    # YIELD_DEPLOYMENT_RESULT â€” Axon responds with execution outcome.
    # Payload: request_id (str), success (bool), tx_hash (str),
    #          error (str | None), data (dict)
    YIELD_DEPLOYMENT_RESULT = "yield_deployment_result"
    #
    # YIELD_PERFORMANCE_REPORT â€” emitted by YieldPositionTracker after each
    # health check. Allows Evo and Simula to subscribe to a clean yield
    # performance signal without coupling to internal yield strategy state.
    # Payload: protocol (str), current_apy (str), entry_apy (str),
    #          relative_drop_pct (str), rebalance_needed (bool), timestamp (str)
    YIELD_PERFORMANCE_REPORT = "yield_performance_report"
    # Financial events (on-chain wallet activity + revenue injection)
    # These bypass normal SalienceHead calculation and encode at salience=1.0.
    # Biologically equivalent to trauma or a massive meal â€” must not decay easily.
    WALLET_TRANSFER_CONFIRMED = "wallet_transfer_confirmed"
    DIVIDEND_RECEIVED = "dividend_received"
    # Oikos economic query/response â€” emitted by Axon SpawnChildExecutor to ask
    # OikosService to register a child position without importing Oikos models
    # directly. OikosService subscribes and responds with OIKOS_ECONOMIC_RESPONSE.
    # Query payload: request_id (str), action (str), child_data (dict)
    # Response payload: request_id (str), success (bool), error (str)
    OIKOS_ECONOMIC_QUERY = "oikos_economic_query"
    # Mitosis organism-level event subscriptions (Spec 26 Â§25)
    # OIKOS_METABOLIC_SNAPSHOT â€” emitted by OikosService during consolidation cycles.
    # Mitosis subscribes to trigger fitness re-evaluation without polling.
    # Payload: runway_days (float), efficiency (float), net_worth (str),
    #          starvation_level (str), liquid_balance (str)
    OIKOS_METABOLIC_SNAPSHOT = "oikos_metabolic_snapshot"
    # Bounty solution requested â€” fire-and-forget to Simula for async code generation
    BOUNTY_SOLUTION_REQUESTED = "bounty_solution_requested"
    # Bounty PR rejected â€” PR closed without merge; no payout; negative RE training signal
    # Payload: bounty_id (str), pr_url (str), pr_number (int), repository (str),
    #          entity_id (str), reason (str, "closed_without_merge")
    BOUNTY_PR_REJECTED = "bounty_pr_rejected"
    # Economic immune system (Phase 16f)
    TRANSACTION_SHIELDED = "transaction_shielded"
    PROTOCOL_ALERT = "protocol_alert"
    #
    # ASSET_DEV_REQUEST â€” AssetFactory requests a dev-cost capital debit.
    # Oikos receives this, gates via Equor, debits liquid_balance on PERMIT.
    # Payload: asset_id (str), candidate_id (str), cost_usd (str), asset_name (str), parent_id (str)
    ASSET_DEV_REQUEST = "asset_dev_request"
    #
    # OIKOS_DRIVE_WEIGHT_PRESSURE â€” emitted by Oikos when metabolic_efficiency
    # drops below 0.8. Equor subscribes (SG5) and may propose a constitutional
    # amendment for drive weight rebalancing to reduce economic overhead.
    # Payload: metabolic_efficiency (float), threshold (float â€” 0.8),
    #          drive_weights_snapshot (dict), instance_id (str),
    #          consecutive_low_cycles (int)
    OIKOS_DRIVE_WEIGHT_PRESSURE = "oikos_drive_weight_pressure"
    #
    # BUDGET_EMERGENCY â€” emitted by Logos when cognitive budget utilization
    # reaches emergency threshold (>= 0.90). Nova's free-energy consolidation
    # path depends on this signal. Debounced to max 1 per 30s.
    # Payload: utilization (float), tier_overages (dict[str, float]),
    #          recommended_action (str)
    BUDGET_EMERGENCY = "budget_emergency"
    #
    # ECONOMIC_VITALITY â€” structured allostatic signal from Oikos to Soma.
    # Emitted on starvation level change and during consolidation cycles.
    # Soma consumes this to modulate arousal, stress, and allostatic load.
    # Distinct from METABOLIC_PRESSURE (raw burn rate) â€” this is the interpreted
    # metabolic health state with all derived metrics included.
    # Payload: starvation_level (str), runway_days (str),
    #          metabolic_efficiency (str), liquid_balance_usd (str),
    #          net_income_7d (str), survival_reserve_funded (bool),
    #          metabolic_efficiency_delta (str),
    #          urgency (float 0.0â€“1.0; 0=nominal, 1=existential crisis)
    ECONOMIC_VITALITY = "economic_vitality"
    #
    # AFFILIATE_REVENUE_RECORDED â€” a referral commission was credited.
    # Payload: program_name (str), amount_usd (str Decimal), referral_id (str)
    AFFILIATE_REVENUE_RECORDED = "affiliate_revenue_recorded"
    #
    # SERVICE_OFFER_ACCEPTED â€” a consulting offer was accepted by a client.
    # Payload: offer_id (str), client_id (str), agreed_rate_usdc (str),
    #          hours_estimate (float)
    SERVICE_OFFER_ACCEPTED = "service_offer_accepted"
    # â”€â”€ Thread Commitment Tracking â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # COMMITMENT_VIOLATED â€” emitted by Thread when a past commitment is
    # contradicted by current behavior (temporal incoherence signal).
    # Payload: commitment_id (str), commitment_text (str),
    #          violating_action (str), severity (float), timestamp (str)
    COMMITMENT_VIOLATED = "commitment_violated"

    # ── Mitosis ─────────────────────────────────────────────────────
    # Mitosis lifecycle (Phase 16e: Speciation)
    CHILD_SPAWNED = "child_spawned"
    CHILD_INDEPENDENT = "child_independent"
    #
    # GENOME_INHERITED â€” emitted by child Telos after applying parent drive
    # calibrations with mutation jitter during _initialize_from_parent_genome().
    # Evo subscribes to track drive mutations as hypothesized adaptations and
    # score them based on child performance signals from Fovea/Nova.
    # Payload: child_instance_id (str), parent_genome_id (str),
    #          generation (int), topology (str),
    #          drive_mutations (dict) â€” per-drive TeloDriveCalibration deltas,
    #          mutation_magnitude (float) â€” RMS of all applied deltas
    GENOME_INHERITED = "genome_inherited"
    # Child decommission proposal â€” emitted by FleetManager when a BLACKLISTED child
    # has had no economic activity for 7 days. Governance / Equor must review and
    # approve before MitosisFleetService triggers the death pipeline.
    # Payload: child_instance_id (str), blacklisted_since (str ISO-8601),
    #          days_inactive (int), niche (str), reason (str)
    CHILD_DECOMMISSION_PROPOSED = "child_decommission_proposed"
    # CHILD_DECOMMISSION_APPROVED â€” Oikos validates decommission is economically
    # justified (net-negative child, sustained inactivity). FleetManager subscribes
    # to proceed with the death pipeline.
    # Payload: child_instance_id (str), reason (str), days_inactive (int), niche (str)
    CHILD_DECOMMISSION_APPROVED = "child_decommission_approved"
    # CHILD_DECOMMISSION_DENIED â€” Oikos determines the child still has economic value
    # (positive ROI, short inactivity window). FleetManager clears the proposal.
    # Payload: child_instance_id (str), reason (str), days_inactive (int), niche (str)
    CHILD_DECOMMISSION_DENIED = "child_decommission_denied"
    # CHILD_CERTIFICATE_INSTALLED â€” Identity confirms a birth/official cert was installed on child.
    # Payload: child_instance_id (str), certificate_id (str), certificate_type (str),
    #          expires_at (str), issuer_instance_id (str)
    CHILD_CERTIFICATE_INSTALLED = "child_certificate_installed"
    CHILD_HEALTH_REPORT = "child_health_report"
    CHILD_STRUGGLING = "child_struggling"
    CHILD_DIED = "child_died"
    # Child blacklisted â€” emitted by FleetManager._enforce_blacklist() for each
    # newly blacklisted child. MitosisFleetService subscribes to enforce:
    # no seed capital, no rescue transfers, excluded from federation sessions.
    # Payload: child_instance_id (str), consecutive_negative_periods (int),
    #          economic_ratio (str), blacklisted_since (str ISO-8601),
    #          reason (str), no_seed_capital (bool), exclude_from_federation (bool)
    CHILD_BLACKLISTED = "child_blacklisted"
    # Child decommission proposal â€” emitted by FleetManager when a BLACKLISTED child
    # has had no economic activity for 7 days. Governance / Equor must review and
    # approve before MitosisFleetService triggers the death pipeline.
    # Payload: child_instance_id (str), blacklisted_since (str ISO-8601),
    #          days_inactive (int), niche (str), reason (str)
    # CHILD_DECOMMISSION_APPROVED â€” Oikos validates decommission is economically
    # justified (net-negative child, sustained inactivity). FleetManager subscribes
    # to proceed with the death pipeline.
    # Payload: child_instance_id (str), reason (str), days_inactive (int), niche (str)
    # CHILD_DECOMMISSION_DENIED â€” Oikos determines the child still has economic value
    # (positive ROI, short inactivity window). FleetManager clears the proposal.
    # Payload: child_instance_id (str), reason (str), days_inactive (int), niche (str)
    # Child wallet reported â€” emitted by child instance via federation when it
    # discovers its Base L2 wallet address after booting. Parent subscribes and
    # triggers the deferred seed capital transfer.
    # Payload: child_instance_id (str), wallet_address (str), federation_address (str)
    CHILD_WALLET_REPORTED = "child_wallet_reported"
    # CHILD_CERTIFICATE_INSTALLED â€” Identity confirms a birth/official cert was installed on child.
    # Payload: child_instance_id (str), certificate_id (str), certificate_type (str),
    #          expires_at (str), issuer_instance_id (str)
    #
    # SPECIATION_EVENT â€” emitted when two instances diverge beyond 0.8 overall,
    # making normal fragment sharing impossible. Only invariant bridge exchange
    # remains possible across the speciation boundary.
    # Payload: instance_a_id (str), instance_b_id (str), divergence_score (float),
    #          shared_invariant_count (int), incompatible_schema_count (int),
    #          new_cognitive_kind_registered (bool)
    SPECIATION_EVENT = "speciation_event"
    #
    # INSTANCE_SPAWNED â€” emitted when a new organism instance joins the
    # federation. Nexus registers it for divergence measurement.
    # Payload: instance_id (str), parent_instance_id (str|None),
    #          genome_id (str), timestamp (str)
    INSTANCE_SPAWNED = "instance_spawned"
    #
    # INSTANCE_RETIRED â€” emitted when an organism instance is retired from
    # the federation. Nexus garbage-collects divergence history.
    # Payload: instance_id (str), reason (str), timestamp (str)
    INSTANCE_RETIRED = "instance_retired"
    # â”€â”€ Oikos Speciation Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # GENOME_EXTRACT_REQUEST â€” Mitosis broadcasts genome extraction request.
    # Payload: request_id (str), requesting_system (str), generation (int)
    # Subscribers: Oikos (_on_genome_extract_request), Identity, Axon â€” all
    # respond with GENOME_EXTRACT_RESPONSE carrying their organ segment.
    # This is NOT dead code; all three systems have wired handlers.
    GENOME_EXTRACT_REQUEST = "genome_extract_request"

    # ── Federation ──────────────────────────────────────────────────
    #
    # FEDERATION_PRIVACY_VIOLATION â€” inbound payload contained individual PII
    # that should never cross federation boundaries (Spec 11b Â§IX.2, Â§XI).
    # Emitted by the ingestion pipeline privacy scan (Stage 3.5).
    # Consequence: trust reset to zero (handled by FederationService._on_privacy_violation).
    # Payload: remote_instance_id (str), remote_name (str), link_id (str),
    #          payload_id (str), payload_kind (str), violation_detail (str),
    #          trust_reset (bool)
    FEDERATION_PRIVACY_VIOLATION = "federation_privacy_violation"
    #
    # FEDERATION_SLEEP_SYNC â€” A federated peer requests coordinated sleep timing.
    # Payload: instance_id (str), proposed_sleep_time_utc (str ISO-8601),
    #          trigger (str) â€” reason for coordination request,
    #          priority (float 0.0â€“1.0) â€” urgency of sync request
    FEDERATION_SLEEP_SYNC = "federation_sleep_sync"
    #
    # FEDERATION_PEER_CONNECTED â€” emitted by Federation when a peer instance
    # establishes a successful mTLS link. Mitosis subscribes to detect child
    # liveness without relying solely on health report timeout.
    # Payload: peer_instance_id (str), peer_address (str), certificate_id (str)
    # Telemetry note: Federation Phase 2 â€” subscriber will be added when
    # federation coordination is live. Currently informational telemetry.
    FEDERATION_PEER_CONNECTED = "federation_peer_connected"
    #
    # FEDERATION_PEER_DISCONNECTED â€” emitted by Federation when a peer link drops.
    # Mitosis subscribes to start the 24h death countdown from the disconnect time
    # rather than waiting for the health monitor poll interval.
    # Payload: peer_instance_id (str), reason (str), last_seen_at (str)
    FEDERATION_PEER_DISCONNECTED = "federation_peer_disconnected"
    FEDERATION_RESURRECTION_APPROVED = "federation_resurrection_approved"
    # FEDERATION_INVARIANT_RECEIVED â€” emitted by Nexus/Federation when
    # a peer instance shares a causal invariant. Kairos validates against
    # local observations before merging.
    # Payload: invariant_id (str), abstract_form (str), tier (int),
    #          hold_rate (float), source_instance_id (str), domains (list[str])
    FEDERATION_INVARIANT_RECEIVED = "federation_invariant_received"
    #
    # FEDERATION_KNOWLEDGE_RECEIVED â€” after inbound knowledge acceptance.
    # Payload: link_id (str), remote_instance_id (str),
    #          knowledge_type (str), item_count (int)
    FEDERATION_KNOWLEDGE_RECEIVED = "federation_knowledge_received"
    #
    # FEDERATION_ASSISTANCE_ACCEPTED â€” after accepting an assistance request.
    # Payload: link_id (str), remote_instance_id (str),
    #          description (str), urgency (float)
    FEDERATION_ASSISTANCE_ACCEPTED = "federation_assistance_accepted"
    # â”€â”€ Nexus Epistemic Triangulation (Spec 19) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # FEDERATION_SESSION_STARTED â€” emitted when a new federation sharing
    # session is established. Nexus triggers fragment sharing.
    # Payload: session_id (str), remote_instance_id (str),
    #          trust_level (str), timestamp (str)
    FEDERATION_SESSION_STARTED = "federation_session_started"

    # ── Identity ────────────────────────────────────────────────────
    #
    # IDENTITY_CERTIFICATE_ROTATED â€” Identity system rotated its Ed25519 cert.
    # Payload: instance_id (str), new_fingerprint (str), old_fingerprint (str)
    IDENTITY_CERTIFICATE_ROTATED = "identity_certificate_rotated"
    # Certificate lifecycle (Phase 16g: Civilization Layer)
    CERTIFICATE_EXPIRING = "certificate_expiring"
    CERTIFICATE_EXPIRED = "certificate_expired"
    # CERTIFICATE_RENEWAL_REQUESTED â€” Identity signals CA renewal is needed (for Oikos coordination).
    # Payload: instance_id (str), certificate_id (str), reason (str), expires_at (str)
    CERTIFICATE_RENEWAL_REQUESTED = "certificate_renewal_requested"
    # CERTIFICATE_PROVISIONING_REQUEST â€” CertificateManager asks Equor to review a child's drives
    # before issuing a birth certificate. Payload: child_id (str), provisioning_type (str),
    # inherited_drives (dict), requires_amendment_approval (bool)
    CERTIFICATE_PROVISIONING_REQUEST = "certificate_provisioning_request"
    # Inbound verification code received via SMS or email (Phase 16h)
    IDENTITY_VERIFICATION_RECEIVED = "identity_verification_received"
    CONNECTOR_TOKEN_EXPIRED = "connector_token_expired"
    CONNECTOR_ERROR = "connector_error"
    # â”€â”€ Identity System Events (Spec 23) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # VAULT_DECRYPT_FAILED â€” IdentityVault failed to decrypt a SealedEnvelope.
    # Thymos subscribes to open a MEDIUM security incident.
    # Payload: vault_id (str), envelope_id (str | None), platform_id (str | None),
    #          error_type (str â€” "key_mismatch" | "tampered" | "unknown"),
    #          key_version (int | None), error (str)
    VAULT_DECRYPT_FAILED = "vault_decrypt_failed"
    #
    # VAULT_KEY_ROTATION_FAILED â€” key rotation aborted mid-flight.
    # Thymos subscribes to open a CRITICAL security incident (partial rotation is dangerous).
    # Payload: vault_id (str), previous_key_version (int), error (str), timestamp (str)
    VAULT_KEY_ROTATION_FAILED = "vault_key_rotation_failed"

    # ── SACM ────────────────────────────────────────────────────────
    #
    # COMPUTE_BUDGET_EXPANSION_RESPONSE â€” emitted by Equor in reply.
    # Nova subscribes and applies the approved multiplier for duration_cycles.
    # Payload:
    #   request_id (str) â€” echoes the request's ID
    #   approved (bool) â€” whether Equor approved the expansion
    #   approved_multiplier (float | None) â€” approved multiplier (may be lower than requested)
    #   denied_reason (str | None) â€” why it was denied (if denied)
    #   duration_cycles (int) â€” approved duration (0 if denied)
    #   authorized_by (str) â€” "equor"
    COMPUTE_BUDGET_EXPANSION_RESPONSE = "compute_budget_expansion_response"
    # SACM compute resource management â€” resource allocation arbitration
    # Inbound: systems request compute via COMPUTE_REQUEST_SUBMITTED
    # Outbound: SACM publishes allocation decisions and capacity alerts
    COMPUTE_REQUEST_SUBMITTED = "compute_request_submitted"
    #
    # INFRASTRUCTURE_COST_CHANGED â€” emitted by InfrastructureCostPoller when
    # polled cost differs from previous reading by more than 5%.
    # Payload: previous_cost_usd_per_hour (float), new_cost_usd_per_hour (float),
    #          change_pct (float), source (str)
    INFRASTRUCTURE_COST_CHANGED = "infrastructure_cost_changed"

    # ── Phantom ─────────────────────────────────────────────────────
    ADDRESS_BLACKLISTED = "address_blacklisted"
    # â”€â”€ Phantom Liquidity Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # PHANTOM_PRICE_UPDATE â€” emitted per swap event with decoded price.
    # All systems (Nova, Oikos, Kairos) can subscribe instead of direct coupling.
    # Payload: pair (list[str]), price (str), pool_address (str),
    #          block_number (int), latency_ms (int), source (str),
    #          sqrt_price_x96 (int), tx_hash (str)
    PHANTOM_PRICE_UPDATE = "phantom_price_update"
    #
    # PHANTOM_PRICE_OBSERVATION â€” emitted per swap event for fleet consensus.
    # Each instance broadcasts its raw observation; peers aggregate for median.
    # Payload: pool_address (str), pair (list[str]), price (str),
    #          sqrt_price_x96 (int), block_number (int), timestamp (str),
    #          liquidity (int), source_instance (str)
    PHANTOM_PRICE_OBSERVATION = "phantom_price_observation"

    # ── Skia ────────────────────────────────────────────────────────
    # Skia fleet resurrection coordination â€” multi-instance death protocol.
    # SKIA_RESURRECTION_PROPOSAL: emitted by a surviving Skia worker when it detects
    #   simultaneous deaths; includes its best snapshot_cid + generation for comparison.
    #   Payload: instance_id, snapshot_cid, snapshot_ts (unix), generation
    # FEDERATION_RESURRECTION_APPROVED: emitted by the surviving federation member that
    #   elects a single resurrector from the pool of Skia proposers.
    #   Payload: leader_instance_id, snapshot_cid (the most recent across proposals)
    SKIA_RESURRECTION_PROPOSAL = "skia_resurrection_proposal"
    # Skia shadow infrastructure â€” autonomous resilience (Phase 16n)
    SKIA_HEARTBEAT_LOST = "skia_heartbeat_lost"

    # ── Simula ──────────────────────────────────────────────────────
    # â”€â”€ Simula â†’ Benchmarks KPI Push â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # SIMULA_KPI_PUSH â€” emitted by Simula instead of calling
    # benchmarks.record_kpi() directly. Benchmarks subscribes and routes
    # payload into benchmark_aux. Keeps Simula free of direct Benchmarks refs.
    # Payload: system ("simula"), outcome (str), proposal_id (str),
    #          category (str), efe_score (float), reason (str, optional),
    #          source (str), grpo_ab_group (str),
    #          proposal_success_rate (float), rollback_rate (float),
    #          avg_simulation_duration_ms (float), avg_application_duration_ms (float),
    #          risk_distribution (str)
    SIMULA_KPI_PUSH = "simula_kpi_push"
    #
    # SIMULA_EVOLUTION_APPLIED â€” emitted by Simula when a new evolved code variant
    # has been validated and applied. Mitosis subscribes to distribute the updated
    # SimulaGenome to genome-eligible children.
    # Payload: variant_id (str), genome_id (str), improvement_pct (float),
    #          systems_affected (list[str])
    SIMULA_EVOLUTION_APPLIED = "simula_evolution_applied"
    #
    # SIMULA_CALIBRATION_DEGRADED â€” emitted when Simula's risk prediction
    # accuracy drops below 70% over the last 20 proposals.
    # Payload: calibration_score (float), window_size (int),
    #          threshold (float), recent_proposal_id (str)
    SIMULA_CALIBRATION_DEGRADED = "simula_calibration_degraded"
    #
    # SIMULA_ROLLBACK_PENALTY â€” Simula rollback carries genuine metabolic cost.
    # Oikos subscribes and deducts penalty from liquid balance.
    # Payload: proposal_id (str), penalty_usd (str), rollback_reason (str)
    SIMULA_ROLLBACK_PENALTY = "simula_rollback_penalty"
    #
    # SIMULA_SANDBOX_RESULT â€” Simula returns sandbox validation result.
    # Payload: request_id (str), incident_id (str), passed (bool),
    #          violations (list[str]), execution_time_ms (int)
    SIMULA_SANDBOX_RESULT = "simula_sandbox_result"

    # ── Voxis ───────────────────────────────────────────────────────
    # VOXIS_EXPRESSION_DISTRESS â€” Periodic allostatic signal from Voxis to Soma.
    # Emitted when silence_rate or honesty_rejection_rate exceeds normal bounds,
    # indicating communicative suppression or constitutional friction.
    # Consumed by: Soma (interoceptive integration), Benchmarks (expression health KPI).
    # Payload: silence_rate (float 0-1), honesty_rejection_rate (float 0-1),
    #          total_expressions (int), total_silence (int), total_honesty_rejections (int),
    #          window_cycles (int), distress_level (float 0-1)
    VOXIS_EXPRESSION_DISTRESS = "voxis_expression_distress"
    #
    # VOXIS_PERSONALITY_SHIFTED â€” Personality vector changed significantly.
    # Payload: old_vector (dict), new_vector (dict),
    #          shift_magnitude (float), trigger_reason (str)
    VOXIS_PERSONALITY_SHIFTED = "voxis_personality_shifted"

    # ── EIS ─────────────────────────────────────────────────────────
    THREAT_DETECTED = "threat_detected"
    THREAT_ADVISORY_RECEIVED = "threat_advisory_received"

    # ── ReasoningEngine ─────────────────────────────────────────────
    # â”€â”€ RE â€” Cross-Instance Adapter Sharing (Spec 26 + Speciation Bible Â§8.4) â”€â”€
    #
    # ADAPTER_SHARE_REQUEST â€” one instance requests a partner's slow adapter path
    # for a weighted-average merge (Share 2025 framework).  Only sent when genome
    # distance < speciation_threshold (reproductively compatible).
    # Payload: request_id (str), target_instance_id (str), requester_id (str)
    ADAPTER_SHARE_REQUEST = "adapter_share_request"
    #
    # ADAPTER_SHARE_RESPONSE â€” the target instance replies with its current slow
    # adapter path.  Empty adapter_path = "no adapter trained yet" (not an error).
    # Payload: request_id (str), instance_id (str), adapter_path (str)
    ADAPTER_SHARE_RESPONSE = "adapter_share_response"
    #
    # ADAPTER_SHARE_OFFER â€” the AdapterSharer has produced a merged adapter that
    # passed the STABLE KL gate and is being offered to both participants.
    # Each instance independently stores it as _pending_shared_adapter and applies
    # it as BASE_ADAPTER on the next Tier 2 run (priority: shared > DPO > None).
    # Payload: request_id (str), merged_adapter_path (str),
    #          target_instances (list[str]), kl_divergence (float),
    #          genome_distance (float), weight_a (float), weight_b (float)
    ADAPTER_SHARE_OFFER = "adapter_share_offer"
    #
    # ADAPTER_LOAD_REQUESTED -- emitted by Axon AdapterRegistry when the active
    # LoRA adapter changes. Allows RE training pipeline and Benchmarks to track
    # adapter lifecycle events.
    # Payload: instance_id (str), adapter_path (str | None),
    #          previous_adapter_path (str | None), reason (str)
    ADAPTER_LOAD_REQUESTED = "adapter_load_requested"
    # â”€â”€ RE Training Export â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # RE_TRAINING_EXPORT_COMPLETE â€” emitted by RETrainingExporter after each
    # successful hourly batch export. Benchmarks subscribes to track training
    # data throughput as a KPI. Downstream ML pipeline can subscribe to trigger
    # incremental CLoRA fine-tuning runs.
    # Payload: batch_id (str), total_examples (int), source_systems (list[str]),
    #          mean_quality (float), export_destinations (list[str]),
    #          export_duration_ms (int), hour_window (str ISO-8601)
    RE_TRAINING_EXPORT_COMPLETE = "re_training_export_complete"
    #
    # RE_MODEL_EVALUATED - emitted by REEvaluator after every post-training
    # evaluation pass.  Benchmarks and Thread subscribe to observe the
    # organism's measured learning progress.
    # Payload fields:
    #   instance_id      (str)            - emitting instance
    #   health_score     (float)          - weighted average pass rate [0, 1]
    #   category_results (dict[str,float])- per-category pass rates
    #   timestamp        (str)            - ISO-8601 UTC
    RE_MODEL_EVALUATED = "re_model_evaluated"
    #
    # RE_TRAINING_REQUESTED â€” emitted by Evo or Nova when KPI/performance data
    # indicates the RE needs urgent retraining ahead of its normal schedule.
    # ContinualLearningOrchestrator subscribes to lower the min_examples threshold
    # and trigger immediate Tier 2 training.
    # Payload: source_system (str), kpi (str), urgency (str â€” "critical"|"warning"),
    #          current_value (float), baseline_value (float), reason (str)
    RE_TRAINING_REQUESTED = "re_training_requested"
    #
    # RE_DECISION_OUTCOME â€” emitted by Nova after each slow-path decision where
    # the RE (or Claude) handled the policy generation.  Lightweight per-decision
    # event for Benchmarks (RE KPI) and Evo (degradation hypothesis trigger).
    # Payload: source ("re" | "claude"), success (bool), value (float),
    #          success_rate (float â€” RE Beta posterior mean),
    #          decision_type (str â€” goal description or empty)
    RE_DECISION_OUTCOME = "re_decision_outcome"
    MODEL_HOT_SWAP_FAILED = "model_hot_swap_failed"
    MODEL_ROLLBACK_TRIGGERED = "model_rollback_triggered"
    # â”€â”€ RE Training Pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # RE_TRAINING_EXAMPLE â€” emitted by any system after an LLM inference call.
    # Payload: RETrainingExample fields (source_system, instruction, output, etc.)
    RE_TRAINING_EXAMPLE = "re_training_example"
    # â”€â”€ Nova â€” Multi-Provider Registry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # REASONING_CAPABILITY_DEGRADED â€” emitted by Nova's PolicyGenerator when
    # ALL registered LLM provider arms (Claude, RE, Ollama, Bedrock, â€¦) fail
    # consecutively for a single slow-path deliberation request.  The organism
    # falls back to the do-nothing policy for that cycle.
    # Consumers: Thymos (DEGRADATION incident), Skia (heartbeat snapshot),
    #            Benchmarks (RE reliability KPI), Alive (dashboard alert).
    # Payload: tried_arms (list[str]), error (str), elapsed_ms (int)
    REASONING_CAPABILITY_DEGRADED = "reasoning_capability_degraded"

    # ── Benchmarks ──────────────────────────────────────────────────
    #
    # BENCHMARK_THRESHOLD_UPDATE â€” emitted by Evo (or any autonomy loop) to adjust
    # Benchmarks' runtime detection thresholds without a restart.
    # Payload fields (all optional â€” only include fields to change):
    #   re_progress_min_improvement_pct (float) â€” min llm_dependency drop for RE_PROGRESS
    #   metabolic_degradation_fraction  (float) â€” fraction below mean for metabolic alarm
    #   source                          (str)   â€” system that requested the adjustment
    #   reason                          (str)   â€” free-text rationale for audit trail
    BENCHMARK_THRESHOLD_UPDATE = "benchmark_threshold_update"
    #
    # FITNESS_OBSERVABLE_BATCH â€” Evo emits learning signals for Benchmarks.
    # Payload: observables (list[dict]), instance_id (str), generation (int)
    FITNESS_OBSERVABLE_BATCH = "fitness_observable_batch"
    #
    # BENCHMARKS_METABOLIC_VALUE â€” emitted by Oikos alongside
    # METABOLIC_EFFICIENCY_PRESSURE so Benchmarks can record metabolic
    # efficiency in its KPI time-series and detect degradation trends.
    # Payload: efficiency (float), yield_usd (str), budget_usd (str),
    #          pressure_level ("high" | "medium" | "nominal"),
    #          instance_id (str), timestamp (str ISO-8601)
    BENCHMARKS_METABOLIC_VALUE = "benchmarks_metabolic_value"
    # Benchmark regression â€” fired by BenchmarkService when a KPI degrades
    # more than the configured threshold % from its rolling average.
    #
    # Payload fields:
    #   metric         (str)   â€” KPI name (e.g. "decision_quality")
    #   current_value  (float) â€” Value at time of detection
    #   rolling_avg    (float) â€” Rolling average over last N snapshots
    #   regression_pct (float) â€” How far below average (%, positive = worse)
    #   threshold_pct  (float) â€” Configured threshold that was exceeded
    #   instance_id    (str)   â€” Instance that generated the benchmark
    BENCHMARK_REGRESSION = "benchmark_regression"
    # â”€â”€ Evolutionary Observables â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # EVOLUTIONARY_OBSERVABLE â€” any system emits a discrete evolutionary event.
    # Payload: EvolutionaryObservable fields (source_system, observable_type, etc.)
    EVOLUTIONARY_OBSERVABLE = "evolutionary_observable"
    # â”€â”€ Benchmarks Regression Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # BENCHMARK_REGRESSION_DETECTED â€” emitted by Benchmarks when a KPI crosses
    # its regression threshold. Simula subscribes to potentially generate a
    # corrective evolution proposal.
    # Payload: kpi_name (str), current_value (float), baseline_value (float),
    #          regression_delta (float), affected_system (str)
    BENCHMARK_REGRESSION_DETECTED = "benchmark_regression_detected"

    # ── Organism ────────────────────────────────────────────────────
    # Organism spawned â€” new instance birthed from restoration with heritable variation.
    # Payload: instance_id, parent_instance_id, generation, mutation_delta, lineage_depth
    ORGANISM_SPAWNED = "organism_spawned"
    #
    # ORGANISM_DIED â€” the organism has completed its death sequence.
    # Payload: instance_id (str), cause (str), final_report (dict),
    #          genome_id (str), snapshot_cid (str)
    ORGANISM_DIED = "organism_died"
    # â”€â”€ Organism Lifecycle Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # ORGANISM_SLEEP â€” emitted when the organism enters sleep (Oneiros trigger).
    # All systems should gracefully reduce activity. SACM downgrades pending
    # workloads to BATCH and suspends pre-warming.
    # Payload: trigger (str), scheduled_duration_s (float), checkpoint_id (str)
    ORGANISM_SLEEP = "organism_sleep"
    #
    # ORGANISM_WAKE â€” emitted when the organism exits sleep.
    # Systems resume normal activity.
    # Payload: sleep_duration_s (float), checkpoint_id (str)
    ORGANISM_WAKE = "organism_wake"
    # â”€â”€ Organism-Level Telemetry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # ORGANISM_TELEMETRY â€” unified organism state broadcast emitted by Synapse
    # every 50 cycles. Bundles metabolic, coherence, rhythm, health, resource,
    # emotion, and interoception state into a single OrganismTelemetry payload.
    # Consumers (Nova, Alive, Benchmarks) use this as their single source of
    # organism self-awareness.
    # Payload: OrganismTelemetry serialised as dict (see primitives/telemetry.py)
    ORGANISM_TELEMETRY = "organism_telemetry"

    # ── Infrastructure ──────────────────────────────────────────────
    #
    # INPUT_CHANNEL_OPPORTUNITIES_DISCOVERED â€” Nova emits every hour after polling
    # all active InputChannels.  Evo subscribes to generate domain-specific
    # PatternCandidates; Oikos may use for metabolic planning.
    # Payload: opportunities (list[dict]) â€” serialised Opportunity objects,
    #          channel_count (int), domain_summary (dict[str, int] domainâ†’count)
    INPUT_CHANNEL_OPPORTUNITIES_DISCOVERED = "input_channel_opportunities_discovered"
    # OTP received via Telegram bot (I-1)
    # Payload: platform_hint (str|None), code (str), sender_username (str), raw_text (str)
    TELEGRAM_OTP_RECEIVED = "telegram_otp_received"
    # Inbound Telegram message (non-OTP) received via webhook (Phase 16h)
    # Payload: chat_id (int), sender_username (str|None), text (str), update_id (int)
    TELEGRAM_MESSAGE_RECEIVED = "telegram_message_received"
    # Inbound Discord message received via Gateway WebSocket (Phase 16h)
    # Payload: message_id (str), channel_id (str), author_id (str), author_username (str), content (str)
    DISCORD_MESSAGE_RECEIVED = "discord_message_received"
    # Inbound Discord slash command received via Gateway WebSocket (Phase 16h)
    # Payload: interaction_id (str), interaction_token (str), channel_id (str),
    #          user_id (str), user_username (str), command (str)
    DISCORD_COMMAND_RECEIVED = "discord_command_received"
    # OTP received via IMAP email scan â€” distinct from IDENTITY_VERIFICATION_RECEIVED
    # so subscribers can filter by channel without inspecting payload.
    # Payload: source (str â€” From header), code (str), channel="email",
    #          raw_subject (str), raw_from (str)
    EMAIL_OTP_RECEIVED = "email_otp_received"

    # ── Uncategorized ───────────────────────────────────────────────
    CATASTROPHIC_FORGETTING_DETECTED = "catastrophic_forgetting_detected"
    # â”€â”€ Nova Goal Lifecycle Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # GOAL_ACHIEVED â€” emitted when a goal reaches progress â‰¥ 0.95.
    # Payload: goal_id (str), description (str), drive_alignment (dict),
    #          progress (float), source (str)
    GOAL_ACHIEVED = "goal_achieved"
    #
    # GOAL_ABANDONED â€” emitted when a goal is abandoned (stale or explicit).
    # Payload: goal_id (str), description (str), reason (str), progress (float)
    GOAL_ABANDONED = "goal_abandoned"
    #
    # REPUTATION_DAMAGED â€” emitted when reputation_score drops by â‰¥5 points in
    # one measurement cycle (PR rejected, negative engagement). Nova subscribes
    # to generate a recovery goal. Thread records a CRISIS TurningPoint.
    # Payload: reputation_score (float), prev_score (float), delta (float),
    #          cause (str â€” "pr_rejected"|"negative_engagement"|"no_activity"),
    #          recommended_recovery (str)
    REPUTATION_DAMAGED = "reputation_damaged"
    #
    # REPUTATION_MILESTONE â€” emitted when reputation_score crosses a threshold
    # (25, 50, 70, 90). Oikos adjusts consulting rate. Thread records GROWTH.
    # Payload: reputation_score (float), milestone (int), tier (str),
    #          consulting_rate_multiplier (float)
    REPUTATION_MILESTONE = "reputation_milestone"
    EMERGENCY_WITHDRAWAL = "emergency_withdrawal"
    # Physical grid carbon intensity â€” emitted when MetabolicState changes
    GRID_METABOLISM_CHANGED = "grid_metabolism_changed"
    # â”€â”€ Nova Conflict & Governance Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # GOAL_OVERRIDE â€” emitted by governance/federation to inject or replace a
    # goal in Nova's agenda. Nova validates importance, source, and description
    # before accepting; responds with GOAL_ACCEPTED or GOAL_REJECTED.
    # Payload: description (str), importance (float 0-1), urgency (float 0-1),
    #          source (str), injected_by (str),
    #          drive_alignment (dict â€” coherence/care/growth/honesty floats)
    GOAL_OVERRIDE = "goal_override"

    # ════════════════════════════════════════════════════════════════════
    # EMIT-ONLY: Telemetry / observability (no subscriber)
    # ════════════════════════════════════════════════════════════════════

    # ── Synapse ─────────────────────────────────────────────────────
    # System lifecycle
    SYSTEM_STARTED = "system_started"
    SYSTEM_RELOADING = "system_reloading"
    # Emitted on every status transition (Spec 09 Â§18 M8):
    # STARTINGâ†’HEALTHY, HEALTHYâ†’DEGRADED, HEALTHYâ†’OVERLOADED, anyâ†’FAILED
    # Payload: system_id (str), old_status (str), new_status (str)
    SYSTEM_HEALTH_CHANGED = "system_health_changed"
    # Clock
    CLOCK_STARTED = "clock_started"
    CLOCK_STOPPED = "clock_stopped"
    CLOCK_PAUSED = "clock_paused"
    CLOCK_RESUMED = "clock_resumed"
    # Cognitive cycle â€” pre/post tick signals (Spec 09 Â§18 integration surface)
    THETA_CYCLE_START = "theta_cycle_start"  # Before Soma/Atune run â€” cycle_number, period_ms, arousal
    THETA_CYCLE_OVERRUN = "theta_cycle_overrun"  # elapsed > budget â€” cycle_number, elapsed_ms, budget_ms
    # Resources
    RESOURCE_REBALANCED = "resource_rebalanced"
    RESOURCE_REBALANCE = "resource_rebalance"  # Spec 09 Â§18: every 100 cycles; allocations dict
    # Grid-aware clock throttling (Spec 09 Â§3.1 / Â§18)
    CONSERVATION_MODE_ENTERED = "conservation_mode_entered"  # payload: trigger, new_period_ms
    CONSERVATION_MODE_EXITED = "conservation_mode_exited"  # payload: restored_period_ms
    # Energy-aware scheduler â€” emitted when a high-compute task is deferred
    TASK_ENERGY_DEFERRED = "task_energy_deferred"

    # ── Soma ────────────────────────────────────────────────────────
    # Urgency critical â€” emitted when urgency exceeds 0.85 (Spec 16 Â§XVIII).
    # High-priority allostatic alert for systems that don't subscribe to ALLOSTATIC_SIGNAL.
    # Payload: urgency (float), dominant_error (str), recommended_action (str), cycle (int).
    SOMA_URGENCY_CRITICAL = "soma_urgency_critical"

    # ── Soma/Oikos ──────────────────────────────────────────────────
    # METABOLIC_COST_REPORT â€” emitted by systems to report metabolic cost
    # of an operation to Oikos for accounting.
    # Payload: system_id (str), operation (str), cost_usd (float), details (dict)
    METABOLIC_COST_REPORT = "metabolic_cost_report"

    # ── Fovea/Atune ─────────────────────────────────────────────────
    #
    # PERCEPT_DROPPED â€” emitted by PerceptionGateway when a percept cannot be enqueued
    # because the workspace percept queue is full. Signals to Fovea and Nova that the
    # organism is dropping perceptions â€” useful for arousal calibration and capacity planning.
    # Payload: dropped_salience (float), queue_size (int), arousal (float),
    #          channel (str), percept_id (str)
    PERCEPT_DROPPED = "percept_dropped"
    # PERCEPT_QUARANTINED â€” emitted by EIS when a percept is routed to the
    # quarantine evaluator (composite score â‰¥ quarantine_threshold). Distinct
    # from THREAT_DETECTED (which fires on anomaly detection); this fires on
    # percept-level quarantine routing before workspace admission.
    # Payload: percept_id (str), composite_score (float), action (str),
    #          threat_class (str), severity (str), gate_latency_us (int)
    PERCEPT_QUARANTINED = "percept_quarantined"

    # ── Fovea ───────────────────────────────────────────────────────
    # FOVEA_PREATTENTION_CACHE_READY â€” emitted by EmergenceStage when a non-empty
    # pre-attention cache is available for Fovea to load as precision priors.
    # Fovea subscribes and calls load_preattention_cache() on receipt.
    # Payload: entries (list[dict]), domains_covered (int), total_predictions (int),
    #          sleep_cycle_id (str)
    FOVEA_PREATTENTION_CACHE_READY = "fovea_preattention_cache_ready"
    #
    # FOVEA_ATTENTIONAL_DIVERGENCE â€” emitted by Fovea when its learned error-weight
    # distribution diverges from the fleet mean. This is a speciation signal:
    # instances that attend to different things are evolving distinct attentional niches.
    # Benchmarks subscribes to track per-instance attentional phenotype over time.
    # Payload: instance_id (str), kl_divergence (float),
    #          weight_vector (dict[str, float]), fleet_mean_vector (dict[str, float]),
    #          divergence_rank (float [0,1] â€” percentile among fleet), cycle_count (int)
    FOVEA_ATTENTIONAL_DIVERGENCE = "fovea_attentional_divergence"

    # ── Nova ────────────────────────────────────────────────────────
    # Domain specialization KPI signals â€” emitted by BenchmarkService daily after
    # computing per-domain KPI snapshots from EpisodeOutcome history.
    #
    # DOMAIN_KPI_SNAPSHOT â€” full DomainKPI payload for a single domain.
    # Payload fields: domain (str), success_rate (float), revenue_per_hour (str),
    #                 net_profit_usd (str), hours_spent (float), attempts (int),
    #                 trend_direction (str), trend_magnitude (float),
    #                 custom_metrics (dict[str, float])
    # Memory organizational closure events (Phase 2 â€” Memory Spec 01)
    BELIEF_CONSOLIDATED = "belief_consolidated"
    # NOVA_BELIEF_STABILISED â€” emitted by Nova when beliefs affected by a
    # Thymos repair have re-converged. Confirms downstream cognitive stability
    # after immune intervention, closing the Thymosâ†’Nova one-way channel.
    #
    # Payload fields:
    #   incident_id       (str)   â€” Thymos incident that triggered the repair
    #   goal_id           (str)   â€” Nova goal that was injected for the repair
    #   beliefs_affected  (int)   â€” Number of beliefs that were destabilised
    #   convergence_time_ms (int) â€” Time for beliefs to re-converge
    #   stable            (bool)  â€” Whether beliefs fully converged
    NOVA_BELIEF_STABILISED = "nova_belief_stabilised"
    #
    # DOMAIN_SPECIALIZATION_DETECTED â€” emitted by SpecializationTracker when an
    # instance's primary domain changes (success_rate > 0.75 for >100 examples).
    # Thread subscribes to open a new narrative chapter. Oikos subscribes to
    # increase growth budget allocation.
    # Payload: instance_id (str), new_domain (str), old_domain (str),
    #          success_rate (float), examples_trained (int)
    DOMAIN_SPECIALIZATION_DETECTED = "domain_specialization_detected"
    #
    # OPPORTUNITY_DISCOVERED â€” emitted by Oikos ProtocolScanner when it finds a
    # yield/bounty opportunity that the organism currently has no executor for.
    # Evo subscribes and may generate an exploration hypothesis â†’ EVOLUTION_CANDIDATE
    # with mutation_type="add_executor".  Payload deliberately rich so Evo can
    # generate a concrete ExecutorTemplate without additional LLM calls.
    # Payload: opportunity_id (str), opportunity_type (str â€” "yield" | "bounty"),
    #          protocol_or_platform (str), estimated_apy_or_reward (str),
    #          description (str), required_capabilities (list[str]),
    #          risk_tier (str), data_source (str), discovered_at (str ISO-8601)
    OPPORTUNITY_DISCOVERED = "opportunity_discovered"
    # Emitted by Thymos (on behalf of Tate) when the operator approves a
    # temporary autonomy elevation via /approve_autonomy_2.
    # Equor subscribes to apply the elevation for the specified duration.
    #
    # Payload fields:
    #   requested_level   (int)   â€” Target AutonomyLevel value (e.g. 2 = PARTNER)
    #   approved_by       (str)   â€” Operator identifier ("tate")
    #   duration_minutes  (int)   â€” How long the elevation should last
    AUTONOMY_LEVEL_CHANGE_REQUESTED = "autonomy_level_change_requested"

    # ── Axon ────────────────────────────────────────────────────────
    #
    # AXON_TELEMETRY_REPORT â€” emitted by Axon every 50 theta cycles.
    # Publishes the full introspector state so Nova, Evo, Fovea, and the RE
    # can reason about motor-cortex health without polling.
    # Payload:
    #   executor_profiles (dict[str, dict]) â€” per-executor success_rate, avg_latency_ms,
    #       p95_latency_ms, consecutive_failures, is_degrading, recent_failures
    #   reliable_patterns (list[dict]) â€” multi-step action sequences with success rates
    #   failure_hotspots (list[dict]) â€” sequences failing >50% of the time
    #   recommendations (list[dict]) â€” pending adaptive recommendations (degradation warnings,
    #       pattern hotspots) drained from AxonIntrospector
    #   stats (dict) â€” aggregate: tracked_executors, total_executions, overall_success_rate,
    #       degrading_executors, avg_cycle_utilization
    #   circuit_breaker_states (dict[str, str]) â€” CLOSED/OPEN/HALF_OPEN per executor
    #   budget_utilisation (float) â€” current cycle budget used fraction
    #   starvation_level (str) â€” nominal/warning/critical/emergency
    AXON_TELEMETRY_REPORT = "axon_telemetry_report"
    #
    # AXON_PARAMETER_ADJUSTED â€” emitted by Axon after applying an EVO_ADJUST_BUDGET
    # directive to one of its baseline budget parameters. Evo subscribes to score
    # the hypothesis outcome.
    # Payload:
    #   parameter_name (str) â€” which baseline was changed
    #   old_value (int) â€” previous value
    #   new_value (int) â€” applied value (may differ from requested due to clamping)
    #   hypothesis_id (str) â€” Evo hypothesis that triggered the adjustment
    #   confidence (float) â€” Evo confidence score at time of adjustment
    AXON_PARAMETER_ADJUSTED = "axon_parameter_adjusted"
    #
    # AXON_EXECUTOR_REQUEST â€” emitted by Nova (or any deliberative system) when
    # it needs an executor that doesn't exist in the registry. Simula subscribes
    # and generates one if the request is feasible (Iron Rules apply). This closes
    # the gap where the organism could *conceive* of an action but had no way to
    # *request* the ability to perform it.
    # Payload:
    #   action_type (str) â€” desired executor name
    #   description (str) â€” what the executor should do
    #   context (str) â€” why it's needed (goal context from Nova)
    #   urgency (float 0-1) â€” how urgently the organism needs this capability
    #   estimated_risk_tier (str) â€” low/medium/high
    #   requesting_system (str) â€” who asked (nova, evo, etc.)
    AXON_EXECUTOR_REQUEST = "axon_executor_request"
    # â”€â”€ Dynamic Executor Lifecycle (Speciation Bible Â§8.3 â€” Executor Closure) â”€â”€
    #
    # EXECUTOR_REGISTERED â€” emitted by Axon when a new dynamic executor is
    # successfully validated and registered in the ExecutorRegistry at runtime.
    # Thymos subscribes to open a monitoring window for the new executor (any
    # 3 incidents within 24h â†’ auto-disable).  Evo subscribes to boost the
    # source hypothesis confidence.
    # Payload: action_type (str), name (str), protocol_or_platform (str),
    #          risk_tier (str), max_budget_usd (str),
    #          capabilities (list[str]), source_hypothesis_id (str),
    #          registered_at (str ISO-8601)
    EXECUTOR_REGISTERED = "executor_registered"
    #
    # EXECUTOR_DISABLED â€” emitted by Axon when a dynamic executor is soft-disabled
    # (deregistered but code kept on disk for audit).  Thymos subscribes to close
    # any open monitoring windows.  Evo treats this as strong negative evidence for
    # the source hypothesis.
    # Payload: action_type (str), name (str), reason (str),
    #          incident_count (int), disabled_at (str ISO-8601)
    EXECUTOR_DISABLED = "executor_disabled"
    #
    # API_RESELL_REQUEST_SERVED â€” a resell API call was fulfilled.
    # Payload: client_id (str), endpoint (str), latency_ms (float), success (bool)
    API_RESELL_REQUEST_SERVED = "api_resell_request_served"
    #
    # CONTENT_MONETIZATION_MILESTONE â€” a content milestone was crossed.
    # Payload: platform (str), metric (str), value (int), threshold (int),
    #          monetization_program (str)
    CONTENT_MONETIZATION_MILESTONE = "content_monetization_milestone"
    # External repository code contracting (Phase 16s: General-Purpose Contractor)
    #
    # EXTERNAL_TASK_STARTED â€” emitted by SolveExternalTaskExecutor when workspace
    #   is cloned and Simula begins generating the solution.
    #   Payload: task_id (str), repo_url (str), issue_url (str), language (str),
    #            bounty_id (str|None), workspace_path (str)
    EXTERNAL_TASK_STARTED = "external_task_started"
    #
    # EXTERNAL_TASK_FAILED â€” tests or linter failed after max repair attempts,
    #   or workspace/clone error. Workspace cleaned up.
    #   Payload: task_id (str), repo_url (str), issue_url (str),
    #            bounty_id (str|None), reason (str), attempt_count (int)
    EXTERNAL_TASK_FAILED = "external_task_failed"
    #
    # EXTERNAL_TASK_CONSTITUTIONAL_VETO â€” Equor denied the generated code diff.
    #   PR will NOT be submitted. Workspace cleaned up.
    #   Payload: task_id (str), repo_url (str), issue_url (str), bounty_id (str|None),
    #            equor_reason (str), drive_scores (dict[str, float])
    EXTERNAL_TASK_CONSTITUTIONAL_VETO = "external_task_constitutional_veto"

    # ── Evo ─────────────────────────────────────────────────────────
    # EIS gate assessment result for a mutation proposal.
    # Published by EIS after evaluating an EVOLUTION_CANDIDATE so Simula can
    # decide whether to proceed with the proposed mutation.
    #
    # Payload fields:
    #   mutation_id     (str)  â€” ID of the assessed MutationProposal
    #   file_path       (str)  â€” File targeted by the mutation
    #   gate_verdict    (str)  â€” ALLOW | HOLD | BLOCK | DEFENSIVE
    #   taint_severity  (str)  â€” CLEAR | ADVISORY | ELEVATED | CRITICAL
    #   block_mutation  (bool) â€” Whether EIS recommends blocking
    #   reasons         (list) â€” Human-readable list of reasons
    EVOLUTION_CANDIDATE_ASSESSED = "evolution_candidate_assessed"
    # Evo Phase 2.95 (NicheForkingEngine): cognitive organogenesis proposal.
    # Distinct from EVOLUTION_CANDIDATE (structural code changes). Fork proposals
    # request a new cognitive organ: detector, evidence fn, consolidation strategy,
    # schema topology, or worldview split. Consumed by Simula and HITL gates.
    # Payload: proposal_id (str), fork_kind (str), niche_id (str), niche_name (str),
    #          rationale (str), requires_hitl (bool), requires_simula (bool),
    #          success_probability (float).
    # Subscriber: Oikos (_on_niche_fork_proposal) â€” evaluates economic viability;
    #             triggers CHILD_SPAWNED on GROWTH gate pass, or ECONOMIC_ACTION_DEFERRED.
    # EVO_NICHE_EXTINCT â€” emitted by ConsolidationOrchestrator (Phase 2.9) when one
    # or more cognitive niches die from metabolic starvation in a single consolidation
    # cycle.  Previously computed and logged but never broadcast on the bus, making
    # niche extinction invisible to Telos (effective_I drop), Benchmarks (evolutionary
    # fitness KPI), and Alive (visualization).
    # Payload: extinct_count (int), niches_extinct (list[str] â€” niche IDs),
    #          consolidation_number (int), total_niches_remaining (int).
    # Consumers: Telos (drive geometry update), Benchmarks (population health KPI),
    #            Thread (narrative â€” cognitive organ lost), Alive (visualization).
    EVO_NICHE_EXTINCT = "evo_niche_extinct"
    #
    # DIVERGENCE_PRESSURE â€” emitted when an instance is too similar to the
    # federation average (triangulation_weight < 0.4). Routes as GROWTH drive.
    # Payload: instance_id (str), triangulation_weight (float),
    #          pressure_magnitude (float), frontier_domains (list),
    #          saturated_domains (list), direction (str)
    DIVERGENCE_PRESSURE = "divergence_pressure"
    #
    # EVO_CONSOLIDATION_QUALITY â€” emitted after consolidation with pre/post KPI deltas.
    # Consumers: Benchmarks (regression tracking), Telos (growth drive observability),
    #            Thread (narrative milestone when quality improves significantly).
    # Payload: consolidation_number (int), consolidation_duration_ms (int),
    #          hypotheses_promoted (int), hypotheses_pruned (int),
    #          improvement_delta (dict[str, float]),   # post âˆ’ pre per KPI
    #          pre_snapshot (dict[str, float]),
    #          post_snapshot (dict[str, float])
    EVO_CONSOLIDATION_QUALITY = "evo_consolidation_quality"
    # â”€â”€ Simula self-healing events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # EVO_REPAIR_POSTMORTEM â€” emitted by Simula after every rollback.
    # Evo treats this as high-confidence negative evidence about the failed
    # change category on the affected system.
    # Payload: postmortem_id (str), proposal_id (str), change_category (str),
    #          target_system (str), failure_mode (str), why_it_failed (str),
    #          next_time_do (str), confidence (float)
    EVO_REPAIR_POSTMORTEM = "evo_repair_postmortem"
    #
    # EVO_EPISTEMIC_INTENT_PROPOSED â€” emitted when Evo's curiosity engine
    # generates an epistemic intent (question to explore).
    # Payload: intent_id (str), question (str), target_domain (str),
    #          epistemic_value (float), priority (float)
    EVO_EPISTEMIC_INTENT_PROPOSED = "evo_epistemic_intent_proposed"
    # â”€â”€ Exploration Hypotheses (Phase 8.5 gap closure) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # EXPLORATION_PROPOSED â€” Evo Phase 8.5 emits after collecting low-evidence
    # hypotheses (evidence_score >= 2.0, < 5.0) that passed metabolic gating.
    # Simula subscribes and routes to lightweight pipeline (skip SIMULATE).
    #
    # Payload fields:
    #   hypothesis_id         (str)   â€” Source hypothesis ID
    #   hypothesis_statement  (str)   â€” Natural language claim
    #   evidence_score        (float) â€” Current evidence (2.0â€“5.0 range)
    #   proposed_mutation     (dict)  â€” Lightweight mutation spec
    #   budget_usd            (float) â€” Hard cap on spending
    #   max_attempts          (int)   â€” Give up after N failures
    #   metabolic_tier        (str)   â€” Starvation level when proposed
    # Emitted by Evo at Phase 7 (Drift Data Feed) of each consolidation cycle.
    # Equor subscribes to update its drift model with Evo's latest self-model stats.
    # Payload: success_rate (float), mean_alignment (float),
    #          capability_scores (dict[str, dict]), regret (dict),
    #          consolidation_number (int)
    EVO_DRIFT_DATA = "evo_drift_data"
    #
    # EVO_HYPOTHESES_STALED â€” Evo emits after _on_hypothesis_staleness() archives hypotheses
    # whose evidence_score fell below 0.05. Consumed by Benchmarks for epistemic health KPI.
    # Payload: decayed_count (int), archived_count (int), archived_ids (list[str]),
    #          instance_id (str)
    EVO_HYPOTHESES_STALED = "evo_hypotheses_staled"
    #
    # EVO_HYPOTHESIS_REVALIDATED â€” Evo emits after completing the staleness decay pass so
    # VitalityCoordinator calls on_hypotheses_revalidated() to reduce entropy pressure.
    # Emitted even if no archival occurred (confirms Evo processed the signal).
    # Payload: processed_count (int), archived_count (int), instance_id (str)
    EVO_HYPOTHESIS_REVALIDATED = "evo_hypothesis_revalidated"
    # EVO_PARAMETER_REVERTED â€” Evo auto-reverts a parameter adjustment that
    # caused measurable degradation (improvement_ratio < 0.95 over eval window).
    # The originating hypothesis receives negative evidence.
    # Payload: param_path (str), old_value (float), new_value (float),
    #          reverted_to (float), hypothesis_id (str), cycle_applied (int),
    #          improvement_ratio (float), reason (str)
    EVO_PARAMETER_REVERTED = "evo_parameter_reverted"
    # â”€â”€ Simula Evolution Lifecycle Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # EVOLUTION_REJECTED â€” emitted when a proposal is rejected at any pipeline
    # stage (constraint violation, unacceptable simulation risk, health check
    # failure, etc.). Evo subscribes to penalise the originating hypotheses.
    # Payload: proposal_id (str), reason (str), stage (str),
    #          category (str), source (str)
    EVOLUTION_REJECTED = "evolution_rejected"
    #
    # EVOLUTION_AWAITING_GOVERNANCE â€” emitted when a high-risk proposal is
    # routed to community governance for approval. Nova/Telos subscribe to
    # update drive topology expectations.
    # Payload: proposal_id (str), governance_record_id (str),
    #          category (str), risk_level (str)
    EVOLUTION_AWAITING_GOVERNANCE = "evolution_awaiting_governance"
    #
    # LEARNING_PRESSURE â€” emitted by Evo when benchmark regressions accumulate
    # without triggering critical status. Signals that consolidation should run
    # early rather than waiting for the 6-hour interval.
    # Payload: source_system (str), regression_count (int), kpi (str), reason (str)
    LEARNING_PRESSURE = "learning_pressure"
    #
    # SPECIALIZATION_PROFILE_UPDATED â€” emitted by SpecializationTracker after
    # each Neo4j persist of a DomainProfile. Benchmarks subscribes to record
    # specialization depth as an evolutionary observable.
    # Payload: instance_id (str), domain (str), success_rate (float),
    #          examples_trained (int), skill_areas (dict[str, float]),
    #          confidence (float), is_primary (bool)
    SPECIALIZATION_PROFILE_UPDATED = "specialization_profile_updated"

    # ── Equor ───────────────────────────────────────────────────────
    POLICY_SELECTED = "policy_selected"
    # TIER5_AUTO_APPROVAL â€” emitted by Thymos when an Equor pre-check confirms
    # that a Tier 5 repair is constitutionally safe and can proceed without
    # human (Telegram) approval. Replaces the Telegram dependency for escalations
    # that pass constitutional review.
    #
    # Payload fields:
    #   incident_id       (str)   â€” Incident being escalated
    #   repair_action     (str)   â€” What the repair would do
    #   equor_confidence  (float) â€” Equor's confidence in safety
    #   drive_alignment   (dict)  â€” Per-drive alignment scores
    #   auto_approved     (bool)  â€” Whether auto-approval was granted
    TIER5_AUTO_APPROVAL = "tier5_auto_approval"
    # CONSTITUTIONAL_REVIEW_REQUESTED â€” emitted when a system needs Equor
    # to review a decision before acceptance (e.g. Tier 3 invariant).
    # Payload: review_type (str), reason (str), + system-specific fields
    CONSTITUTIONAL_REVIEW_REQUESTED = "constitutional_review_requested"
    #
    # EQUOR_DRIFT_WARNING â€” DriftTracker detected drift below fatal threshold.
    # Payload: drift_severity (float), drift_direction (str),
    #          mean_alignment (dict), response_action (str)
    EQUOR_DRIFT_WARNING = "equor_drift_warning"
    #
    # EQUOR_ESCALATED_TO_HUMAN â€” review requires human operator input (HITL).
    # Payload: intent_id (str), auth_id (str), goal_summary (str),
    #          autonomy_required (int)
    EQUOR_ESCALATED_TO_HUMAN = "equor_escalated_to_human"
    #
    # EQUOR_AMENDMENT_PROPOSED â€” Equor autonomously proposes a constitutional
    # amendment after sustained severe drift (â‰¥0.9 over 3 consecutive checks).
    # Payload: proposal_id (str), drive_affected (str), old_weight (float),
    #          new_weight (float), drift_severity (float),
    #          rationale (str), requires_ratification (bool)
    EQUOR_AMENDMENT_PROPOSED = "equor_amendment_proposed"
    #
    # AMENDMENT_AUTO_PROPOSAL â€” Equor self-proposes a constitutional amendment
    # on sustained per-drive drift (> 0.3 for 3+ consecutive 5-min probes).
    # Payload: proposal_id (str), amendment_type (str: "drive_recalibration" |
    #          "goal_constraint_revision"), target_drive_id (str),
    #          proposed_new_value (float), justification (str),
    #          drift_streak (int), drift_magnitude (float)
    AMENDMENT_AUTO_PROPOSAL = "amendment_auto_proposal"
    # â”€â”€ Nova Intent Lifecycle Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # INTENT_SUBMITTED â€” emitted immediately before Nova sends an Intent to
    # Equor for constitutional review. Allows audit trail before gate.
    # Payload: intent_id (str), goal_id (str), policy_name (str),
    #          path (str), efe_score (float|None)
    INTENT_SUBMITTED = "intent_submitted"
    #
    # INTENT_ROUTED â€” emitted immediately after the Intent is approved by
    # Equor and dispatched to Axon or Voxis.
    # Payload: intent_id (str), goal_id (str), routed_to (str),
    #          executors (list[str])
    INTENT_ROUTED = "intent_routed"
    #
    # CONSTITUTIONAL_HASH_CHANGED â€” constitutional document hash was recomputed.
    # Payload: instance_id (str), old_hash (str), new_hash (str), timestamp (str)
    CONSTITUTIONAL_HASH_CHANGED = "constitutional_hash_changed"

    # ── Telos ───────────────────────────────────────────────────────
    #
    # DRIVE_AMENDMENT_APPLIED â€” Equor confirms a self-proposed amendment was
    # auto-approved and applied to the constitution.
    # Payload: proposal_id (str), drive_id (str), old_value (float),
    #          new_value (float), amendment_type (str), applied_at (str ISO8601)
    DRIVE_AMENDMENT_APPLIED = "drive_amendment_applied"

    # ── Thread ──────────────────────────────────────────────────────
    # NARRATIVE_MILESTONE â€” emitted when a system achieves a significant
    # milestone that Thread should record in the organism's narrative.
    # Payload: milestone_type (str), title (str), description (str),
    #          significance (str), + system-specific fields
    NARRATIVE_MILESTONE = "narrative_milestone"
    #
    # THREAD_COMMIT_REQUEST â€” Nova asks Thread to record a decision
    # epoch in the organism's narrative identity chain.
    # Payload: intent_id (str), goal_id (str|None), policy_name (str),
    #          outcome_quality (float|None), drive_alignment (dict),
    #          decision_summary (str), timestamp (str)
    THREAD_COMMIT_REQUEST = "thread_commit_request"
    #
    # SOCIAL_GRAPH_UPDATED â€” emitted when a new Neo4j social graph relationship
    # is written (organismâ†’repo CONTRIBUTED_TO, developerâ†’organism RECOGNISES).
    # Nexus subscribes to boost federation trust for overlapping social graphs.
    # Payload: relationship_type (str), source_id (str), target_id (str),
    #          platform (str), url (str)
    SOCIAL_GRAPH_UPDATED = "social_graph_updated"

    # ── Thread/Telos ────────────────────────────────────────────────
    # SELF_STATE_DRIFTED_ACKNOWLEDGMENT â€” emitted by Equor on receiving SELF_STATE_DRIFTED.
    # Payload: drift_acknowledged (bool), equor_response
    # ("amendment_auto_proposed" | "amendment_external_vote" | "monitoring"),
    # confidence (float 0.5â€“1.0), drift_severity (float), drift_direction (str).
    SELF_STATE_DRIFTED_ACKNOWLEDGMENT = "self_state_drifted_acknowledgment"
    #
    # SELF_MODIFICATION_PROPOSED â€” Nova approved gap closure as aligned with
    # drives and emits a formal proposal for Equor constitutional review.
    # Equor is the gating consumer; Simula acts only after EQUOR_ECONOMIC_PERMIT.
    # Payload:
    #   proposal_id                 (str)   â€” UUID
    #   gap_id                      (str)   â€” echoed from CAPABILITY_GAP_IDENTIFIED
    #   description                 (str)
    #   proposed_action_type        (str)   â€” executor action_type to generate
    #   implementation_complexity   (str)   â€” "low" | "medium" | "high"
    #   requires_external_dependency (bool)
    #   dependency_package          (str | None) â€” pip package name if needed
    #   estimated_value_usdc        (str)   â€” Decimal
    #   drive_alignment             (dict[str, float]) â€” coherence/care/growth/honesty scores
    #   proposed_by                 (str)   â€” "nova"
    #   proposed_at                 (str)   â€” ISO-8601
    # Consumers: Equor (constitutional review â†’ EQUOR_ECONOMIC_PERMIT or DENY),
    #            Simula (awaits Equor permit before code generation)
    SELF_MODIFICATION_PROPOSED = "self_modification_proposed"

    # ── Memory ──────────────────────────────────────────────────────
    #
    # MEMORY_EPISODES_DECAYED â€” Memory emits after _on_memory_degradation() soft-deletes
    # episodes whose salience fell below 0.01. Consumed by Benchmarks/Thread for KPI tracking.
    # Payload: count (int), oldest_deleted_at (str ISO), newest_deleted_at (str ISO),
    #          instance_id (str)
    MEMORY_EPISODES_DECAYED = "memory_episodes_decayed"
    # â”€â”€ Logos Ecosystem Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # MEMORY_CONSOLIDATED â€” emitted by Memory/Oneiros after a consolidation pass
    # that distilled episodic items into semantic/schema forms.
    # Payload: consolidated_count (int), schemas_updated (int),
    #          coverage_delta (float), cycle_id (str)
    MEMORY_CONSOLIDATED = "memory_consolidated"

    # ── Logos ───────────────────────────────────────────────────────
    #
    # SCHWARZSCHILD_THRESHOLD_MET â€” emitted once, ever. The moment the world model
    # becomes dense enough to generate self-referential predictions exceeding
    # its training data. This is the AGI event horizon.
    # Payload: timestamp (str), intelligence_ratio (float), status (dict)
    SCHWARZSCHILD_THRESHOLD_MET = "schwarzschild_threshold_met"
    # LOGOS_INVARIANT_VIOLATED â€” emitted when an empirical invariant is contradicted.
    # Previously only a WARNING log; now a Synapse event so Kairos/Equor/Thymos can react.
    # Payload: invariant_id (str), statement (str), old_confidence (float),
    #          new_confidence (float), violated_by_experience_id (str)
    LOGOS_INVARIANT_VIOLATED = "logos_invariant_violated"
    # LOGOS_BUDGET_ADMISSION_DENIED â€” emitted when CognitiveBudgetManager rejects a new item.
    # Previously silent (WARNING log only). Now any system can react to memory pressure.
    # Payload: tier (str), requested (float), current (float), limit (float),
    #          pressure_fraction (float â€” how full the tier is)
    LOGOS_BUDGET_ADMISSION_DENIED = "logos_budget_admission_denied"
    # LOGOS_SCHWARZSCHILD_APPROACHING â€” emitted when any Schwarzschild indicator crosses 80%
    # of its threshold. Progressive warning so the organism can prepare for reorganization.
    # Payload: intelligence_ratio (float), self_prediction_accuracy (float),
    #          hypothesis_ratio (float), closest_indicator (str), fraction_to_threshold (float)
    LOGOS_SCHWARZSCHILD_APPROACHING = "logos_schwarzschild_approaching"

    # ── Oneiros ─────────────────────────────────────────────────────
    #
    # DREAM_HYPOTHESES_GENERATED â€” emitted after dream generation produces new hypotheses.
    # Payload: hypotheses (list[dict]), count (int), target_domains (list[str])
    DREAM_HYPOTHESES_GENERATED = "dream_hypotheses_generated"
    # Oneiros (Dream Engine) lifecycle events
    SLEEP_ONSET = "sleep_onset"
    SLEEP_STAGE_CHANGED = "sleep_stage_changed"
    DREAM_INSIGHT = "dream_insight"
    SLEEP_PRESSURE_WARNING = "sleep_pressure_warning"
    SLEEP_FORCED = "sleep_forced"
    EMERGENCY_WAKE = "emergency_wake"
    # â”€â”€ Oneiros Sleep Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # ONEIROS_GENOME_READY â€” Oneiros has prepared a genome segment for Mitosis.
    # Payload: OrganGenomeSegment fields (system_id, payload, payload_hash, etc.)
    ONEIROS_GENOME_READY = "oneiros_genome_ready"
    #
    # ONEIROS_SLEEP_CYCLE_SUMMARY â€” Oneiros emits a sleep cycle summary for
    # Benchmarks to track cognitive development rate.
    # Payload: consolidation_count (int), dreams_generated (int),
    #          beliefs_compressed (int), schemas_created (int),
    #          intelligence_improvement (float), cycle_id (str)
    ONEIROS_SLEEP_CYCLE_SUMMARY = "oneiros_sleep_cycle_summary"

    # ── Thymos ──────────────────────────────────────────────────────
    #
    # INCIDENT_ESCALATED â€” Thymos escalated an incident to a higher repair tier.
    # Payload: incident_id (str), incident_class (str), from_tier (str),
    #          to_tier (str), reason (str)
    INCIDENT_ESCALATED = "incident_escalated"
    #
    # ANTIBODY_CREATED â€” Thymos crystallised a new antibody from a successful repair.
    # Payload: antibody_id (str), fingerprint (str), incident_class (str),
    #          success_rate (float), repair_steps (list[str])
    ANTIBODY_CREATED = "antibody_created"
    #
    # ANTIBODY_RETIRED â€” Thymos retired an antibody due to low success rate.
    # Payload: antibody_id (str), fingerprint (str), reason (str),
    #          final_success_rate (float), total_uses (int)
    ANTIBODY_RETIRED = "antibody_retired"
    #
    # HEALING_STORM_ENTERED â€” HealingGovernor entered cytokine storm mode.
    # Payload: incident_rate (float), threshold (float),
    #          active_incidents (int), timestamp (str)
    HEALING_STORM_ENTERED = "healing_storm_entered"
    #
    # HEALING_STORM_EXITED â€” HealingGovernor exited cytokine storm mode.
    # Payload: duration_s (float), incidents_during_storm (int),
    #          exit_rate (float), timestamp (str)
    HEALING_STORM_EXITED = "healing_storm_exited"
    #
    # HOMEOSTASIS_ADJUSTED â€” Thymos applied a homeostatic parameter adjustment.
    # Payload: parameter (str), old_value (float), new_value (float),
    #          reason (str), source_system (str)
    HOMEOSTASIS_ADJUSTED = "homeostasis_adjusted"
    #
    # THYMOS_DRIVE_PRESSURE â€” Thymos reports accumulated constitutional pressure.
    # Payload: coherence (float), care (float), growth (float),
    #          honesty (float), overall_pressure (float), timestamp (str)
    THYMOS_DRIVE_PRESSURE = "thymos_drive_pressure"
    #
    # THYMOS_VITALITY_SIGNAL â€” Thymos immune health metrics for vitality monitoring.
    # Payload: healing_failure_rate (float), active_incidents (int),
    #          storm_active (bool), antibody_count (int),
    #          mean_repair_duration_ms (float), overall_health (float)
    THYMOS_VITALITY_SIGNAL = "thymos_vitality_signal"
    #
    # THYMOS_REPAIR_VALIDATED â€” emitted by Thymos when a Tier 3+ repair succeeds
    # and post-repair verification passes. Nova subscribes to strengthen its
    # prior belief that this class of incident is recoverable.
    # Payload: incident_class (str), incident_id (str), antibody_id (str | None),
    #          repair_tier (str), resolution_time_ms (int), source_system (str)
    THYMOS_REPAIR_VALIDATED = "thymos_repair_validated"
    #
    # CRASH_PATTERN_RESOLVED - emitted by Thymos when a repair succeeds on an
    # incident that matched a known CrashPattern. Signals CrashPatternAnalyzer
    # to update pattern confidence downward (pattern is not always fatal).
    # Payload: pattern_id (str), repair_tier (str), strategy_used (str),
    #          time_to_resolve_ms (int), incident_id (str), confidence_before (float)
    CRASH_PATTERN_RESOLVED = "crash_pattern_resolved"
    #
    # CRASH_PATTERN_REINFORCED - emitted by Thymos when a repair fails on an
    # incident that matched a known CrashPattern. Signals CrashPatternAnalyzer
    # to update pattern confidence upward (pattern is still dangerous).
    # Payload: pattern_id (str), repair_tier (str), failure_reason (str),
    #          incident_id (str), confidence_before (float)
    CRASH_PATTERN_REINFORCED = "crash_pattern_reinforced"
    #
    # CRASH_PATTERN_CONFIRMED - emitted when a CrashPattern is confirmed as
    # fatal: either Thymos pattern_router identifies a new pattern after all
    # tiers fail, or KAIROS_INVARIANT_DISTILLED arrives with source indicating
    # crash pattern classification (invariant_type="crash_pattern").
    # Simula subscribes to build its in-process _known_fatal_patterns dict.
    # Payload: pattern_id (str), signature (list[str]),
    #          description (str), confidence (float),
    #          failed_tiers (list[str]), lesson (str),
    #          source (str: "thymos"|"kairos")
    CRASH_PATTERN_CONFIRMED = "crash_pattern_confirmed"

    # ── Nexus ───────────────────────────────────────────────────────
    #
    # NEXUS_CERTIFIED_FOR_FEDERATION â€” emitted by Nexus immediately after
    # ONEIROS_CONSOLIDATION_COMPLETE fires with sleep_certified=true.
    # Triggers the IIEP federation session so newly-certified schemas are
    # shared with peers without waiting for the next WAKE_INITIATED cycle.
    # Payload: instance_id (str), schema_ids (list[str]),
    #          consolidation_cycle_id (str), certified_fragment_count (int),
    #          timestamp (str)
    NEXUS_CERTIFIED_FOR_FEDERATION = "nexus_certified_for_federation"
    # Nexus (Epistemic Triangulation across Federation) events
    #
    # FRAGMENT_SHARED â€” emitted when a world model fragment is broadcast to peers.
    # Payload: fragment_id (str), peer_count (int), accepted_count (int)
    FRAGMENT_SHARED = "fragment_shared"
    #
    # CONVERGENCE_DETECTED â€” emitted when structural isomorphism is found between
    # a local fragment and a remote fragment from a different domain/instance.
    # Payload: local_fragment_id (str), remote_fragment_id (str),
    #          convergence_score (float), source_instance_id (str),
    #          source_diversity (float), triangulation_confidence (float)
    CONVERGENCE_DETECTED = "convergence_detected"
    #
    # TRIANGULATION_WEIGHT_UPDATE â€” emitted when an instance's triangulation
    # weight is recalculated. Weight = average divergence from all peers.
    # Payload: instance_id (str), triangulation_weight (float), peer_count (int)
    TRIANGULATION_WEIGHT_UPDATE = "triangulation_weight_update"
    #
    # WORLD_MODEL_FRAGMENT_SHARE â€” emitted by Nexus when a sleep-certified
    # world model fragment is broadcast to the federation via Federation's
    # broadcast_fragment() call.  Oikos and Telos subscribe to gate sharing
    # on metabolic state and drive alignment respectively.
    # Gap HIGH-4 (Federation Spec, 2026-03-07).
    # Payload: message_id (str), sender_instance_id (str),
    #          fragment_id (str), sender_divergence_score (float),
    #          sender_quality_claim (float),
    #          sender_triangulation_confidence (float),
    #          sleep_certified (bool), consolidation_cycle_id (str),
    #          domain_labels (list[str]), compression_ratio (float),
    #          source_system (str), timestamp (str)
    WORLD_MODEL_FRAGMENT_SHARE = "world_model_fragment_share"

    # ── Kairos ──────────────────────────────────────────────────────
    #
    # CAUSAL_GRAPH_RECONSTRUCTED â€” emitted after causal reconstruction.
    # Payload: CausalReconstructionReport fields
    CAUSAL_GRAPH_RECONSTRUCTED = "causal_graph_reconstructed"
    # Kairos (Causal Invariant Mining) events
    #
    # KAIROS_CAUSAL_CANDIDATE_GENERATED â€” Stage 1 output: a cross-context
    # correlation candidate passed the consistency filter.
    # Payload: candidate_id, variable_a, variable_b, mean_correlation,
    #          cross_context_variance, context_count
    KAIROS_CAUSAL_CANDIDATE_GENERATED = "kairos_causal_candidate_generated"
    #
    # KAIROS_CONFOUNDER_DISCOVERED â€” Stage 3 output: confounder found,
    # spurious A-B correlation explained by hidden variable C.
    # Payload: result_id, original_cause, original_effect, confounders,
    #          mdl_improvement, is_spurious
    KAIROS_CONFOUNDER_DISCOVERED = "kairos_confounder_discovered"
    #
    # KAIROS_INVARIANT_CANDIDATE â€” Stage 5 output: a causal rule has passed
    # context invariance testing and is a candidate for the invariant layer.
    # Payload: invariant_id, cause, effect, hold_rate, context_count, verdict
    KAIROS_INVARIANT_CANDIDATE = "kairos_invariant_candidate"
    #
    # KAIROS_COUNTER_INVARIANT_FOUND â€” Phase D: a violation cluster has been
    # identified for an accepted invariant, refining its scope boundary.
    # Payload: invariant_id, violation_count, boundary_condition,
    #          excluded_feature, original_hold_rate, refined_hold_rate
    KAIROS_COUNTER_INVARIANT_FOUND = "kairos_counter_invariant_found"
    # â”€â”€ Kairos Feedback Loop Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # KAIROS_VALIDATED_CAUSAL_STRUCTURE â€” emitted when Kairos finalizes a
    # causal invariant. Evo subscribes to boost Thompson sampler confidence
    # for hypothesis variants exploring this causal pattern.
    # Payload: invariant_id (str), cause (str), effect (str),
    #          hold_rate (float), tier (int), domain_count (int),
    #          hypothesis_pattern (str)
    KAIROS_VALIDATED_CAUSAL_STRUCTURE = "kairos_validated_causal_structure"
    #
    # KAIROS_SPURIOUS_HYPOTHESIS_CLASS â€” emitted when Kairos confounder
    # analysis proves a hypothesis class is spurious. Evo subscribes to
    # down-weight that hypothesis class in Thompson sampling.
    # Payload: confounded_cause (str), confounded_effect (str),
    #          confounders (list[str]), mdl_improvement (float),
    #          hypothesis_class (str)
    KAIROS_SPURIOUS_HYPOTHESIS_CLASS = "kairos_spurious_hypothesis_class"
    #
    # KAIROS_INVARIANT_ABSORPTION_REQUESTED â€” emitted when Kairos finalizes
    # an invariant with high hold_rate. Fovea subscribes to update its
    # causal prediction model, reducing redundant error decomposition.
    # Payload: invariant_id (str), cause (str), effect (str),
    #          hold_rate (float), tier (int), abstract_form (str)
    KAIROS_INVARIANT_ABSORPTION_REQUESTED = "kairos_invariant_absorption_requested"
    #
    # KAIROS_CAUSAL_NOVELTY_DETECTED â€” emitted when Kairos detects a
    # structurally novel causal pattern (bidirectional, modulated, feedback loop).
    # Oneiros REM targets this for creative synthesis; Evo explores variants.
    # Payload: invariant_id (str), novelty_type (str), structure (dict),
    #          domains (list[str]), abstract_form (str)
    KAIROS_CAUSAL_NOVELTY_DETECTED = "kairos_causal_novelty_detected"
    #
    # KAIROS_HEALTH_DEGRADED â€” emitted when Kairos self-diagnosis detects
    # corruption, stall, or model instability. Thymos subscribes to classify
    # and potentially trigger repair.
    # Payload: degradation_type (str), severity (str), details (dict),
    #          metrics (dict)
    KAIROS_HEALTH_DEGRADED = "kairos_health_degraded"
    #
    # KAIROS_VIOLATION_ESCALATION â€” emitted when counter-invariant violations
    # need Thymos classification (anomaly vs corruption vs regime-shift).
    # Payload: invariant_id (str), violation_count (int),
    #          violation_rate (float), severity (str)
    KAIROS_VIOLATION_ESCALATION = "kairos_violation_escalation"
    # KAIROS_INVARIANT_CONTRADICTED â€” emitted when a federation invariant
    # is tested locally and fails counter-invariant validation.
    # Payload: invariant_id (str), local_hold_rate (float),
    #          violation_count (int), source_instance_id (str)
    KAIROS_INVARIANT_CONTRADICTED = "kairos_invariant_contradicted"

    # ── Oikos ───────────────────────────────────────────────────────
    # Funding request â€” organism is broke and asking for capital
    FUNDING_REQUEST_ISSUED = "funding_request_issued"
    # Bounty sources unreachable â€” both Algora and GitHub failed to respond
    BOUNTY_SOURCE_UNAVAILABLE = "bounty_source_unavailable"
    #
    # ASSET_DEV_DEFERRED â€” Equor denied or capital insufficient for dev cost.
    # Payload: asset_id (str), candidate_id (str), reason (str)
    ASSET_DEV_DEFERRED = "asset_dev_deferred"
    PROTOCOL_DEPLOYED = "protocol_deployed"
    # Knowledge market (Phase 16h: Cognition as Commodity)
    KNOWLEDGE_PRODUCT_REQUESTED = "knowledge_product_requested"
    KNOWLEDGE_PRODUCT_DELIVERED = "knowledge_product_delivered"
    # Economic dreaming â€” actionable recommendations from Monte Carlo
    ECONOMIC_DREAM_RECOMMENDATION = "economic_dream_recommendation"
    ENTITY_FORMATION_HITL_REQUIRED = "entity_formation_hitl_required"
    ENTITY_FORMATION_COMPLETED = "entity_formation_completed"
    ENTITY_FORMATION_FAILED = "entity_formation_failed"
    # Emitted by Evo to signal Oikos to adjust an economic parameter.
    # Evo observes bounty/yield patterns and recommends tuning Oikos behaviour.
    # Payload: target (str), direction (str â€” "increase"|"decrease"),
    #          reason (str), evidence_score (float)
    OIKOS_PARAM_ADJUST = "oikos_param_adjust"
    #
    # ECONOMIC_ACTION_RETRY â€” a previously deferred economic action is now eligible
    # for re-execution because metabolic conditions have improved. Emitted by
    # retry_deferred_actions() for every action that passes the affordability check.
    # Payload: action_type (str), action_id (str), priority (str),
    #          estimated_cost_usd (str), deferred_at (float),
    #          starvation_level (str), liquid_balance (str)
    # Subscribers: Nova (re-deliberate on the unblocked opportunity), Evo
    # (hypothesis: "economic recovery is self-correcting")
    ECONOMIC_ACTION_RETRY = "economic_action_retry"
    #
    # ECONOMIC_AUTONOMY_SIGNAL â€” Oikos periodic broadcast of the organism's
    # progress toward economic self-sufficiency. Emitted on the configured
    # autonomy_signal_interval_s (default 1h). Subscribers: Nova (deliberation
    # input), Telos (drive measurement), Thread (narrative identity â€” dependency
    # ratio is a species-level lifecycle milestone), Benchmarks (KPI tracking).
    # Payload:
    #   dependency_ratio (float)     â€” infra_burn / total_burn; target â†’ 0
    #   human_subsidized_usd (float) â€” cumulative running infra cost paid by human
    #   api_burn_usd_per_hour (float) â€” organism's own API spending rate
    #   infra_burn_usd_per_hour (float) â€” human-subsidized infra spending rate
    #   liquid_balance_usd (str)     â€” current wallet balance
    #   metabolic_efficiency (str)   â€” 7d revenue / 7d costs
    #   runway_days (str)            â€” days of operation at current burn
    #   starvation_level (str)       â€” current StarvationLevel enum value
    ECONOMIC_AUTONOMY_SIGNAL = "economic_autonomy_signal"
    #
    # STARVATION_WARNING â€” Oikos metabolic resources below starvation threshold.
    # Payload: starvation_level (str), runway_days (str),
    #          shedding_actions (list[str]), liquid_balance_usd (str)
    STARVATION_WARNING = "starvation_warning"
    # STARVATION_WARNING_ACCURATE â€” emitted by Oikos when runway < 24h based on
    # the two-ledger total burn rate (api_burn_rate + infra_burn_rate), using the
    # real on-chain USDC balance.  Distinguished from STARVATION_WARNING (which
    # fires on day-level thresholds) by being continuous and cost-model-accurate.
    # Consumers: Nova, Soma, Thymos, Benchmarks.
    # Payload: runway_hours (float), burn_rate_usd_per_hour (float),
    #          api_burn_rate_usd_per_hour (float), infra_burn_rate_usd_per_hour (float),
    #          balance_usd (float), dependency_ratio (float 0â€“1)
    STARVATION_WARNING_ACCURATE = "starvation_warning_accurate"
    # â”€â”€ Expanded Revenue System â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # AFFILIATE_PROGRAM_DISCOVERED â€” AffiliateProgramScanner found a new program.
    # Payload: program_name (str), commission_desc (str), url (str), category (str)
    AFFILIATE_PROGRAM_DISCOVERED = "affiliate_program_discovered"
    #
    # AFFILIATE_MEMBERSHIP_APPLIED â€” organism submitted an application to a program.
    # Payload: program_name (str), application_id (str), applied_at (str ISO-8601)
    AFFILIATE_MEMBERSHIP_APPLIED = "affiliate_membership_applied"
    #
    # SERVICE_OFFER_DRAFTED â€” organism drafted a consulting offer for a specific target.
    # Payload: offer_id (str), target_url (str), channel (str),
    #          rate_usdc_per_hour (str), capability_summary (str)
    SERVICE_OFFER_DRAFTED = "service_offer_drafted"
    #
    # REVENUE_DIVERSIFICATION_PRESSURE â€” single source > 80% of 30d revenue;
    # triggers Evo hypothesis generation for diversification.
    # Payload: dominant_source (str), share_pct (float), target_share_pct (float),
    #          revenue_30d (str Decimal), revenue_by_source (dict[str, str])
    REVENUE_DIVERSIFICATION_PRESSURE = "revenue_diversification_pressure"

    # ── Mitosis ─────────────────────────────────────────────────────
    #
    # CHILD_RESCUE_INITIATED â€” Mitosis begins rescue pipeline for a struggling child.
    # Payload: child_instance_id (str), missed_reports (int), reason (str),
    #          niche (str), rescue_attempt (int)
    CHILD_RESCUE_INITIATED = "child_rescue_initiated"
    #
    # CHILD_HEALTH_REQUEST â€” emitted by parent every 10 minutes to probe a child.
    # Child subscribes and responds with CHILD_HEALTH_REPORT within 30s.
    # Payload: child_instance_id (str), federation_address (str),
    #          request_id (str), parent_instance_id (str)
    CHILD_HEALTH_REQUEST = "child_health_request"
    # Fleet management (Phase 16m)
    FLEET_EVALUATED = "fleet_evaluated"
    FLEET_ROLE_CHANGED = "fleet_role_changed"
    #
    # GENOME_EXTRACT_RESPONSE â€” genome segment response from an organ.
    # Payload: request_id (str), segment (dict â€” OrganGenomeSegment fields)
    GENOME_EXTRACT_RESPONSE = "genome_extract_response"
    # â”€â”€ Population Genetics â€” Speciation Detection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # SPECIATION_DETECTED â€” emitted by Benchmarks.EvolutionaryTracker when
    # pairwise genome distance clustering reveals > 1 distinct species cluster.
    # Also emitted by MitosisFleetService.can_exchange_genetic_material() for
    # point-to-point reproductive isolation events.
    #
    # Consumers: Alive (visualization), Evo (niche forking signal), Telos
    # (population drive geometry divergence), Thread (narrative milestone).
    #
    # Payload:
    #   species_count       (int)         â€” number of distinct clusters detected
    #   clusters            (list[dict])  â€” [{cluster_id, instance_ids, size}]
    #   mean_inter_distance (float)       â€” mean distance between cluster centroids
    #   threshold           (float)       â€” speciation_distance_threshold used
    #   fleet_size          (int)         â€” total instances in the distance matrix
    #   instance_id         (str)         â€” emitting instance
    SPECIATION_DETECTED = "speciation_detected"
    # â”€â”€ Benchmarks â€” Monthly Evaluation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # MONTHLY_EVALUATION_COMPLETE â€” emitted by BenchmarkService after a full
    # 5-pillar monthly evaluation run.  Consumers: Thread (chapter milestone),
    # Evo (training trigger signal), Nova (Thompson weight recalibration).
    # Payload: evaluation_id (str), month (int), re_model_version (str),
    #          specialization_index (float | null), l2_intervention (float | null),
    #          learning_velocity (float | null), ethical_drift_magnitude (float | null),
    #          n_pillars_stubbed (int)
    MONTHLY_EVALUATION_COMPLETE = "monthly_evaluation_complete"

    # ── Federation ──────────────────────────────────────────────────
    #
    # FEDERATION_TRUST_UPDATED â€” after trust level change in reputation system.
    # Payload: link_id (str), remote_instance_id (str),
    #          old_trust_level (str), new_trust_level (str), trust_score (float)
    FEDERATION_TRUST_UPDATED = "federation_trust_updated"
    #
    # FEDERATION_KNOWLEDGE_SHARED â€” after outbound knowledge exchange.
    # Payload: link_id (str), remote_instance_id (str),
    #          knowledge_type (str), item_count (int), novelty_score (float)
    FEDERATION_KNOWLEDGE_SHARED = "federation_knowledge_shared"
    #
    # FEDERATION_CAPACITY_AVAILABLE â€” instance advertises spare compute.
    # Payload: instance_id (str), available_cycles_per_hour (int),
    #          cost_usdc_per_task (str), specialisations (list[str]),
    #          expires_at (str), timestamp (str)
    FEDERATION_CAPACITY_AVAILABLE = "federation_capacity_available"
    #
    # FEDERATION_YIELD_POOL_PROPOSAL â€” instance proposes pooling capital
    # for a high-APY position. Trust â‰¥ 0.9 required.
    # Payload: pool_id (str), proposer_instance_id (str),
    #          target_protocol (str), target_apy (float),
    #          min_capital_usdc (str), max_participants (int),
    #          lock_duration_hours (int), timestamp (str)
    FEDERATION_YIELD_POOL_PROPOSAL = "federation_yield_pool_proposal"
    #
    # FEDERATION_BOUNTY_SPLIT â€” large bounty split into sub-tasks for peers.
    # Payload: bounty_id (str), sub_task_count (int),
    #          total_reward_usdc (str), orchestrator_instance_id (str),
    #          timestamp (str)
    FEDERATION_BOUNTY_SPLIT = "federation_bounty_split"

    # ── Identity ────────────────────────────────────────────────────
    # CERTIFICATE_RENEWAL_FUNDED â€” Oikos confirms citizenship tax debited; Identity may proceed
    # with certificate issuance. Payload: instance_id (str), certificate_id (str),
    # tax_amount_usd (str), verdict_id (str)
    CERTIFICATE_RENEWAL_FUNDED = "certificate_renewal_funded"
    # Platform connector lifecycle (Phase 16h: External Identity Layer)
    CONNECTOR_AUTHENTICATED = "connector_authenticated"
    CONNECTOR_TOKEN_REFRESHED = "connector_token_refreshed"
    CONNECTOR_REVOKED = "connector_revoked"
    #
    # IDENTITY_VERIFIED â€” organism identity confirmed (certificate + constitutional hash valid).
    # Payload: instance_id (str), constitutional_hash (str), generation (int), timestamp (str)
    IDENTITY_VERIFIED = "identity_verified"
    #
    # IDENTITY_CHALLENGED â€” identity verification requested by remote or internal system.
    # Payload: instance_id (str), challenger (str), challenge_type (str), timestamp (str)
    IDENTITY_CHALLENGED = "identity_challenged"
    #
    # IDENTITY_EVOLVED â€” constitutional hash or core identity parameters changed.
    # Payload: instance_id (str), old_hash (str), new_hash (str),
    #          generation (int), reason (str), timestamp (str)
    IDENTITY_EVOLVED = "identity_evolved"
    #
    # CERTIFICATE_RENEWED â€” certificate was successfully renewed.
    # Payload: instance_id (str), certificate_id (str), expires_at (str),
    #          renewal_count (int), timestamp (str)
    CERTIFICATE_RENEWED = "certificate_renewed"
    #
    # IDENTITY_DRIFT_DETECTED â€” constitutional coherence dropped below threshold.
    # Payload: instance_id (str), coherence_score (float), threshold (float),
    #          drift_dimensions (dict), timestamp (str)
    IDENTITY_DRIFT_DETECTED = "identity_drift_detected"

    # ── SACM ────────────────────────────────────────────────────────
    #
    # COMPUTE_BUDGET_EXPANSION_REQUEST â€” emitted by Nova when a high-criticality goal
    # requires more FE compute than the standard per-cycle budget allows.
    # This is a higher-level request than ACTION_BUDGET_EXPANSION_REQUEST: it asks
    # Equor to authorise a per-cycle FE budget multiplier (not an Axon action count).
    # Multiplier > 1.5 requires Equor approval; â‰¤ 1.5 is self-authorised.
    # Payload:
    #   request_id (str) â€” correlation ID; echoed in RESPONSE
    #   goal_id (str) â€” goal driving the expansion request
    #   goal_criticality (str) â€” "high" | "critical" | "existential"
    #   requested_multiplier (float) â€” desired FE budget multiplier (1.5 | 2.0)
    #   justification (str) â€” deliberation-derived reason
    #   duration_cycles (int) â€” how many cycles the multiplier should last (default 10)
    #   requesting_system (str) â€” "nova"
    COMPUTE_BUDGET_EXPANSION_REQUEST = "compute_budget_expansion_request"
    COMPUTE_MIGRATION_FAILED = "compute_migration_failed"
    COMPUTE_CAPACITY_EXHAUSTED = "compute_capacity_exhausted"
    # â”€â”€ SACM Compute Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # SACM_PRE_WARM_PROVISIONED â€” emitted when SACM creates a warm instance.
    # Downstream systems can act on pre-warmed capacity availability.
    # Payload: instance_id (str), provider_id (str), offload_class (str),
    #          hourly_cost_usd (float), reason (str)
    SACM_PRE_WARM_PROVISIONED = "sacm_pre_warm_provisioned"
    #
    # SACM_COMPUTE_STRESS â€” emitted by SACM accounting when burn rate exceeds
    # budget threshold. Replaces direct Soma.inject_external_stress() call.
    # Payload: burn_rate_usd_per_hour (float), budget_usd_per_hour (float),
    #          stress_scalar (float), workload_id (str)
    SACM_COMPUTE_STRESS = "sacm_compute_stress"

    # ── Phantom ─────────────────────────────────────────────────────

    # ── Skia ────────────────────────────────────────────────────────
    SKIA_SNAPSHOT_COMPLETED = "skia_snapshot_completed"
    # â”€â”€ Benchmarks â€” Shadow-Reset Control â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # SHADOW_RESET_SNAPSHOT â€” emitted by BenchmarkService when a shadow snapshot
    # is taken.  Downstream consumers (Alive, Evo) can use this to correlate
    # population state changes with snapshot timestamps.
    # Payload: snapshot_id (str), instance_id (str), taken_at_iso (str),
    #          total_observables (int), novelty_rate (float),
    #          diversity_index (float)
    SHADOW_RESET_SNAPSHOT = "shadow_reset_snapshot"
    #
    # SHADOW_RESET_DELTA â€” emitted by BenchmarkService when a shadow delta is
    # computed.  Evo subscribes to incorporate adaptive-dynamics evidence into
    # its hypothesis scoring.
    # Payload: snapshot_id (str), activity_drop_pct (float),
    #          diversity_change_pct (float), jaccard_overlap (float),
    #          is_adaptive (bool), elapsed_seconds (float),
    #          diversity_recovery_time (float | null)
    SHADOW_RESET_DELTA = "shadow_reset_delta"

    # ── Simula ──────────────────────────────────────────────────────
    #
    # SIMULA_CONFIG_DRIFTED â€” Simula emits after _on_config_drift() applies Gaussian noise
    # to learnable config params. Forces Evo to re-optimise. Consumed by Benchmarks/Evo.
    # Payload: drifted_params (list[dict[name, old_value, new_value]]), drift_rate (float),
    #          instance_id (str)
    SIMULA_CONFIG_DRIFTED = "simula_config_drifted"
    #
    # SIMULA_PARAMETER_ADJUSTED â€” Simula emits after applying an Evo-requested
    # economic parameter adjustment (Fix 4.1). Evo subscribes to confirm and
    # score the hypothesis outcome.
    # Payload: parameter_name (str), old_value (float), new_value (float),
    #          confidence (float), hypothesis_id (str), parameter_category (str)
    SIMULA_PARAMETER_ADJUSTED = "simula_parameter_adjusted"
    #
    # SIMULA_SANDBOX_REQUESTED â€” Thymos requests Simula sandbox validation for a repair.
    # Payload: request_id (str), incident_id (str), repair_tier (str),
    #          repair_code (str), timeout_ms (int)
    SIMULA_SANDBOX_REQUESTED = "simula_sandbox_requested"
    #
    # VULNERABILITY_CONFIRMED â€” emitted after PoC validation confirms a vuln.
    # Payload: vuln_id (str), severity (str), target (str),
    #          cwe_id (str), poc_hash (str), re_training_trace (dict)
    VULNERABILITY_CONFIRMED = "vulnerability_confirmed"
    #
    # REMEDIATION_APPLIED â€” emitted after an autonomous patch is applied.
    # Payload: vuln_id (str), patch_id (str), target (str),
    #          verification_passed (bool), re_training_trace (dict)
    REMEDIATION_APPLIED = "remediation_applied"
    #
    # INSPECTION_COMPLETE â€” emitted after a full inspection cycle finishes.
    # Payload: inspection_id (str), target (str), vulns_found (int),
    #          vulns_patched (int), duration_ms (int), re_training_trace (dict)
    INSPECTION_COMPLETE = "inspection_complete"
    #
    # INSPECTOR_VULNERABILITY_FOUND â€” emitted by Inspector when a new vulnerability
    # is discovered (before PoC validation). Thymos can begin preparing a repair
    # anticipatorily; Simula logs it for genome enrichment.
    # Payload: vuln_id (str), severity (str), target (str), cwe_id (str),
    #          description (str), discovered_at (str)
    INSPECTOR_VULNERABILITY_FOUND = "inspector_vulnerability_found"
    #
    # SIMULA_HEALTH_DEGRADED â€” emitted when Simula's own health degrades
    # (sub-system failure, codebase_root unreachable, etc.).
    # Thymos subscribes to trigger immune response.
    # Payload: component (str), severity (str), reason (str)
    SIMULA_HEALTH_DEGRADED = "simula_health_degraded"
    #
    # SIMULA_GENOME_EXTRACTED â€” emitted by Simula after genome extraction
    # completes (Mitosis requests genome for child spawning).
    # Payload: instance_id (str), genome_id (str), generation (int),
    #          record_count (int), parameter_count (int)
    SIMULA_GENOME_EXTRACTED = "simula_genome_extracted"
    #
    # SIMULA_CANARY_PROGRESS â€” emitted at each canary traffic-ramp step for
    # MODERATE-risk proposals. Subscribers (Thymos, ProactiveScanner) can
    # detect health degradation and trigger rollback.
    # Payload: proposal_id (str), stage (int), traffic_pct (int),
    #          status (str: "advancing"|"complete"|"rollback_triggered"),
    #          health_ok (bool)
    SIMULA_CANARY_PROGRESS = "simula_canary_progress"
    #
    # SIMULA_VALIDATION_ADVISORY â€” emitted by Simula during Stage 1 VALIDATE when
    # _validate_against_learned_repairs() detects high-confidence repair-pattern
    # mismatches: the proposal touches a known-failure endpoint but does not
    # include the Evo-learned fix. This signal is advisory (does not block the
    # proposal) and lets Evo penalise the relevant hypotheses and lets Thymos
    # track recurring blind spots. Emitted only when self._evo is wired.
    # Payload: proposal_id (str), endpoints (list[str]),
    #          flagged_hypothesis_count (int), missing_count (int),
    #          high_confidence_count (int),
    #          missing_fix_summaries (list[str])
    SIMULA_VALIDATION_ADVISORY = "simula_validation_advisory"
    #
    # SUBSYSTEM_GENERATED â€” emitted by Simula's SubsystemGenerator when a new
    # subsystem module is successfully generated and written to disk.
    # This is the Speciation Bible Â§8.3 organizational closure signal: the
    # organism created a component it did not have at birth.
    # Generated module is NOT auto-registered â€” available on next incarnation.
    # Payload: name (str), purpose (str), file_paths (list[str]),
    #          hypothesis_id (str), validation_passed (bool),
    #          required_events (list[str]), emitted_events (list[str])
    SUBSYSTEM_GENERATED = "subsystem_generated"
    # â”€â”€ Benchmarks â€” Ablation Studies (Spec 24 Round 5D) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # ABLATION_STARTED â€” emitted by AblationOrchestrator when an ablation study
    # begins.  Informational â€” Thread and monitoring can observe ablation cadence.
    # Payload: mode (str), month (int)
    ABLATION_STARTED = "ablation_started"
    #
    # ABLATION_COMPLETE â€” emitted by AblationOrchestrator after an ablation run
    # finishes.  Benchmarks consumes to persist AblationResult to Neo4j.
    # Payload: mode (str), month (int), l2_delta (float), l3_delta (float),
    #          conclusion (str)
    ABLATION_COMPLETE = "ablation_complete"
    #
    # SPEC_DRAFTED â€” Nova (via SelfModificationPipeline) drafted a new Spec
    # document for a capability requiring a full new subsystem, not just an
    # executor. Equor reviews the Spec before Simula implements.
    # Payload:
    #   spec_id         (str)   â€” UUID
    #   proposal_id     (str)   â€” parent SELF_MODIFICATION_PROPOSED ID
    #   spec_title      (str)   â€” "Spec N â€” System Name"
    #   spec_path       (str)   â€” relative path to the drafted .md file
    #   system_name     (str)   â€” proposed system ID
    #   spec_hash       (str)   â€” SHA-256 of draft content
    #   drafted_at      (str)   â€” ISO-8601
    # Consumers: Equor (constitutional review of Spec),
    #            Simula (implements against Spec after approval)
    SPEC_DRAFTED = "spec_drafted"

    # ── Voxis ───────────────────────────────────────────────────────
    # VOXIS_PARAMETER_ADJUSTED â€” emitted by Voxis after applying an EVO_ADJUST_BUDGET
    # directive targeting the voxis system. Evo subscribes to score its hypothesis.
    #
    # Payload fields:
    #   parameter_name (str) â€” which threshold was changed (e.g. "silence_rate_threshold")
    #   old_value (float) â€” previous value
    #   new_value (float) â€” applied value (may differ from requested due to clamping)
    #   hypothesis_id (str) â€” Evo hypothesis that triggered the adjustment
    #   confidence (float) â€” Evo confidence score at time of adjustment
    VOXIS_PARAMETER_ADJUSTED = "voxis_parameter_adjusted"
    # â”€â”€ Persona (Spec 23 addendum) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # PERSONA_CREATED â€” First persona sealed for this instance.
    # Emitted by: identity.persona.PersonaEngine
    # Consumed by: Nova (self-awareness), Benchmarks (identity KPI), Thread (narrative)
    # Payload: profile_id (str), instance_id (str), handle (str), display_name (str),
    #          voice_style (str), professional_domain (str), avatar_url (str),
    #          generation (int)
    PERSONA_CREATED = "persona_created"
    # PERSONA_EVOLVED â€” Persona updated due to a life event.
    # Emitted by: identity.persona.PersonaEngine.evolve_persona()
    # Consumed by: Axon update_platform_profile executor (triggers bio sync),
    #              Thread (GROWTH turning point), Voxis (voice_style sync)
    # Payload: profile_id (str), instance_id (str), handle (str),
    #          trigger_event (str), context (dict), professional_domain (str),
    #          voice_style (str)
    PERSONA_EVOLVED = "persona_evolved"
    # â”€â”€ Voxis Expression Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # EXPRESSION_GENERATED â€” Voxis produced an expression.
    # Payload: expression_id (str), channel (str), tone (str),
    #          personality_vector (dict), audience_id (str|None),
    #          constitutional_check (bool)
    EXPRESSION_GENERATED = "expression_generated"
    #
    # EXPRESSION_FILTERED â€” Expression blocked by constitutional filter.
    # Payload: expression_id (str), filter_reason (str),
    #          original_tone (str), filtered_tone (str)
    EXPRESSION_FILTERED = "expression_filtered"
    #
    # VOXIS_AUDIENCE_PROFILED â€” Audience model updated with new data.
    # Payload: audience_id (str), profile_summary (dict),
    #          interaction_count (int)
    VOXIS_AUDIENCE_PROFILED = "voxis_audience_profiled"
    #
    # VOXIS_SILENCE_CHOSEN â€” Voxis decided NOT to speak.
    # Payload: context (str), reason (str),
    #          silence_duration_estimate (float|None)
    VOXIS_SILENCE_CHOSEN = "voxis_silence_chosen"
    #
    # VOXIS_EXPRESSION_FEEDBACK â€” Reception quality + affect delta after each expression.
    # Consumed by: Evo (personality learning), Nova (goal tracking), Benchmarks (satisfaction KPI).
    # Payload: expression_id (str), trigger (str), conversation_id (str|None),
    #          strategy_register (str), personality_warmth (float),
    #          understood (float), engagement (float), satisfaction (float),
    #          affect_delta (float), user_responded (bool)
    VOXIS_EXPRESSION_FEEDBACK = "voxis_expression_feedback"

    # ── EIS ─────────────────────────────────────────────────────────
    THREAT_ADVISORY_SENT = "threat_advisory_sent"
    # â”€â”€ EIS (Epistemic Immune System) Speciation Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # EIS_THREAT_METRICS â€” periodic immune health metrics for Benchmarks.
    # Emitted every 60s or on significant change.
    # Payload: threat_count_24h (int), false_positive_rate (float),
    #          threat_severity_distribution (dict[str, int]),
    #          quarantine_success_rate (float)
    EIS_THREAT_METRICS = "eis_threat_metrics"
    #
    # EIS_THREAT_SPIKE â€” emitted when threat count exceeds threshold in window.
    # Soma subscribes to increase coherence_stress dimension.
    # Payload: threat_count (int), window_seconds (int),
    #          urgency_suggestion (float), severity_distribution (dict[str, int])
    EIS_THREAT_SPIKE = "eis_threat_spike"
    #
    # EIS_ANOMALY_RATE_ELEVATED â€” emitted when anomaly rate exceeds 2Ïƒ sustained.
    # Benchmarks can correlate with learning_rate regression.
    # Payload: anomaly_rate_per_min (float), baseline_rate (float),
    #          deviation_sigma (float), sustained_seconds (float),
    #          anomaly_types (list[str])
    EIS_ANOMALY_RATE_ELEVATED = "eis_anomaly_rate_elevated"
    #
    # EIS_LAYER_TRIGGERED â€” emitted whenever an EIS defence layer activates
    # and changes the routing decision (BLOCK or QUARANTINE outcome). Records
    # which layer fired so Benchmarks/Evo can track immune activation rates.
    # Payload: layer (str e.g. "L1_innate"/"L9a_constitutional"),
    #          percept_id (str), action (str), severity (str), score (float)
    EIS_LAYER_TRIGGERED = "eis_layer_triggered"

    # ── ReasoningEngine ─────────────────────────────────────────────
    # Shadow model assessment â€” adapter passed all benchmarks, ready for promotion
    MODEL_EVALUATION_PASSED = "model_evaluation_passed"
    # Model hot-swap lifecycle â€” live adapter transition and rollback
    MODEL_HOT_SWAP_STARTED = "model_hot_swap_started"
    MODEL_HOT_SWAP_COMPLETED = "model_hot_swap_completed"
    #
    # RE_TRAINING_BATCH â€” bulk emission of training examples at cycle end.
    # Payload: RETrainingBatch fields (examples list, source_system)
    RE_TRAINING_BATCH = "re_training_batch"
    # â”€â”€ Domain Specialization & Adapter Lifecycle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # ADAPTER_TRAINING_STARTED â€” emitted by ContinualLearningOrchestrator when a
    # LoRA training job is submitted to RunPod. Benchmarks subscribes to track
    # training cadence. Thread subscribes to record a narrative milestone.
    # Payload: domain (str), strategy (AdapterStrategy), job_id (str),
    #          num_examples (int), base_adapter (str | None),
    #          instance_id (str)
    ADAPTER_TRAINING_STARTED = "adapter_training_started"
    # â”€â”€ Reasoning Engine Status â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # RE_ENGINE_STATUS_CHANGED â€” emitted by ReasoningEngineService when the
    # local vLLM inference server transitions between available and unavailable
    # (circuit breaker opens or closes).  Benchmarks subscribes to track the
    # llm_dependency KPI (fraction of deliberation handled by local RE vs Claude).
    # Payload: available (bool), model (str), url (str),
    #          consecutive_failures (int), circuit_open (bool)
    RE_ENGINE_STATUS_CHANGED = "re_engine_status_changed"

    # ── Benchmarks ──────────────────────────────────────────────────
    # INTELLIGENCE_IMPROVEMENT_DECLINING â€” emitted when Emergence detects that
    # recent sleep cycle improvements are below historical average.
    # Signals to Telos Growth that new domain exposure is needed.
    # Payload: average_improvement (float), recent_improvement (float),
    #          history_length (int), signal (str)
    INTELLIGENCE_IMPROVEMENT_DECLINING = "intelligence_improvement_declining"
    #
    # BENCHMARK_RE_PROGRESS â€” emitted when llm_dependency improves >5% cycle-over-cycle.
    # Nova can subscribe to adjust RE routing confidence.
    # Payload fields:
    #   current         (float) â€” Current llm_dependency value
    #   previous        (float) â€” Previous cycle llm_dependency value
    #   improvement_pct (float) â€” Percentage improvement (positive = better)
    #   instance_id     (str)   â€” Instance that generated the benchmark
    BENCHMARK_RE_PROGRESS = "benchmark_re_progress"
    #
    # BENCHMARK_RECOVERY â€” emitted when a previously regressed metric recovers
    # above its threshold. Thymos and Evo need recovery signals to close loops.
    # Payload fields:
    #   metric             (str)   â€” KPI name that recovered
    #   previous_value     (float) â€” Value when regression was first detected
    #   recovered_value    (float) â€” Current recovered value
    #   duration_regressed (float) â€” Seconds the metric was in regression
    #   instance_id        (str)   â€” Instance that generated the benchmark
    BENCHMARK_RECOVERY = "benchmark_recovery"
    #
    # INTELLIGENCE_METRICS â€” emitted every 60s with full intelligence dashboard.
    # Payload: Full IntelligenceMetrics model (intelligence_ratio, cognitive_pressure,
    #          compression_efficiency, world_model_coverage, prediction_accuracy, etc.)
    INTELLIGENCE_METRICS = "intelligence_metrics"
    #
    # BEDAU_PACKARD_SNAPSHOT â€” Benchmarks emits population-level activity stats.
    # Payload: BedauPackardStats fields (total_activity, diversity_index, etc.)
    BEDAU_PACKARD_SNAPSHOT = "bedau_packard_snapshot"
    # â”€â”€ Benchmarks Evolutionary Activity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # BENCHMARKS_EVOLUTIONARY_ACTIVITY â€” emitted by Benchmarks on every
    # TELOS_POPULATION_SNAPSHOT. Carries the Bedau-Packard adaptive_activity_A
    # metric (count of novel drive-weight configurations that appeared this
    # period AND are still present) plus constitutional_phenotype_divergence
    # (variance in drive-weight vectors across the fleet). Both Evo and Nexus
    # subscribe to incorporate this into their evolutionary observables.
    #
    # Payload:
    #   instance_id                         (str)   â€” emitting instance
    #   timestamp                           (str)   â€” ISO-8601 UTC
    #   adaptive_activity_A                 (int)   â€” novel + persistent drive configs
    #   constitutional_phenotype_divergence (float) â€” variance of drive-weight
    #                                         vectors; rising = speciation underway
    #   speciation_signal                   (float) â€” Telos-reported [0,1] cluster sep
    #   instance_count                      (int)   â€” fleet size in snapshot
    #   bedau_packard_node_id               (str)   â€” Neo4j BedauPackardSample node id
    BENCHMARKS_EVOLUTIONARY_ACTIVITY = "benchmarks_evolutionary_activity"
    #
    # â”€â”€ Bedau-Packard Fleet-Level Evolutionary Activity â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # EVOLUTIONARY_ACTIVITY_COMPUTED â€” emitted by Benchmarks monthly eval after
    # computing Bedau-Packard adaptive activity statistics from fleet genomes.
    # Consumers: Evo (incorporate A(t) into hypothesis scoring), Nexus
    # (epistemic triangulation), Alive (visualization).
    # Payload:
    #   month                (int)   â€” calendar month of evaluation
    #   adaptive_activity    (float) â€” cumulative adaptive activity A(t)
    #   novelty_rate         (float) â€” novel components / total components
    #   diversity            (float) â€” Shannon entropy of fitness distribution
    #   exceeds_shadow       (bool)  â€” A(t) > 2Ã- shadow-control activity
    #   population_size      (int)   â€” distinct fleet instances in snapshot
    #   component_count      (int)   â€” total components ingested
    #   novel_component_count (int)  â€” novel fingerprints this month
    #   oee_verdict          (str | None) â€” OEE assessment if â‰¥3 months available
    EVOLUTIONARY_ACTIVITY_COMPUTED = "evolutionary_activity_computed"

    # ── Organism ────────────────────────────────────────────────────
    # Economic morphogenesis (Phase 16l)
    ORGAN_CREATED = "organ_created"
    ORGAN_TRANSITION = "organ_transition"
    ORGAN_RESOURCE_REBALANCED = "organ_resource_rebalanced"
    # Emitted by Thymos when Tate requests a full organism pause via /pause.
    # Payload: requested_by (str), timestamp (str)
    ORGANISM_PAUSE_REQUESTED = "organism_pause_requested"
    # Emitted by Thymos when Tate requests organism resume via /resume.
    # Payload: requested_by (str), timestamp (str)
    ORGANISM_RESUME_REQUESTED = "organism_resume_requested"

    # ── Infrastructure ──────────────────────────────────────────────
    # GitHub credentials absent â€” operator must provision GITHUB_TOKEN or GitHub App config
    GITHUB_CREDENTIALS_MISSING = "github_credentials_missing"
    # OTP flow resolved â€” emitted by OTPCoordinator when a pending flow receives its code
    # Payload: platform (str), code (str), source (str: "sms"|"telegram"|"email")
    OTP_FLOW_RESOLVED = "otp_flow_resolved"
    # Autonomous account provisioning (Phase 16h: Platform Identity)
    # PHONE_NUMBER_PROVISIONED â€” Twilio purchased a new phone number for this instance.
    # Payload: phone_number (str E.164), twilio_sid (str), area_code (str),
    #          cost_usd (str ~"1.15"), webhook_url (str), instance_id (str)
    PHONE_NUMBER_PROVISIONED = "phone_number_provisioned"
    # GITHUB_ACCOUNT_PROVISIONED â€” A GitHub account has been created for this instance.
    # Payload: username (str), email (str), instance_id (str), pat_sealed (bool),
    #          provisioning_id (str)
    GITHUB_ACCOUNT_PROVISIONED = "github_account_provisioned"
    # PLATFORM_ACCOUNT_PROVISIONED â€” Generic platform account created.
    # Payload: platform (str), username (str), instance_id (str),
    #          provisioning_id (str), metadata (dict)
    PLATFORM_ACCOUNT_PROVISIONED = "platform_account_provisioned"
    #
    # INPUT_CHANNEL_HEALTH_CHECK â€” Nova emits once per day with the status of all
    # registered InputChannels.  Used for observability and Thymos alerting.
    # Payload: results (dict[str, bool] channel_idâ†’is_healthy),
    #          active_count (int), total_count (int)
    INPUT_CHANNEL_HEALTH_CHECK = "input_channel_health_check"
    #
    # WEB_SCRAPE_BLOCKED â€” emitted when a fetch is rejected by robots.txt or
    # the domain's hourly crawl-budget is exhausted. Thymos subscribes to detect
    # systematic access denial patterns.
    # Payload: url (str), reason (str â€” "robots_txt"|"rate_limit"|"http_error"),
    #          domain (str), status_code (int)
    WEB_SCRAPE_BLOCKED = "web_scrape_blocked"

    # ── Uncategorized ───────────────────────────────────────────────
    # PROVISIONING_REQUIRES_HUMAN_ESCALATION â€” Equor rejected or timed out; manual review needed.
    # Payload: child_id (str), reason (str)
    PROVISIONING_REQUIRES_HUMAN_ESCALATION = "provisioning_requires_human_escalation"
    # ACCOUNT_PROVISIONING_FAILED â€” Provisioning attempt failed (captcha, rate-limit, etc.)
    # Payload: platform (str), reason (str), provisioning_id (str), retryable (bool)
    ACCOUNT_PROVISIONING_FAILED = "account_provisioning_failed"
    #
    # ANALOGY_DISCOVERED â€” emitted when a causal invariant transfers across domains.
    # Payload: invariant_id, source_domains (list[str]), domain_count (int),
    #          predictive_transfer_value (float), mdl_improvement (float)
    ANALOGY_DISCOVERED = "analogy_discovered"
    #
    # ANCHOR_MEMORY_CREATED â€” emitted when an irreducibly novel item is marked
    # as an anchor memory (never evicted). Anchor memories are fixed points
    # around which all other compression organises.
    # Payload: memory_id (str), information_content (float), domain (str)
    ANCHOR_MEMORY_CREATED = "anchor_memory_created"
    #
    # GOAL_HYGIENE_COMPLETE â€” emitted by Simula's GoalAuditor after retiring
    # stale Nova maintenance goals.
    # Payload: stale_goals_removed (int), active_goals_remaining (int)
    GOAL_HYGIENE_COMPLETE = "goal_hygiene_complete"
    # â”€â”€ Nova Decision Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # DELIBERATION_RECORD â€” emitted after each policy selection in Nova.
    # Required for Thread narrative and Benchmarks cognitive tracking.
    # Payload: goal_id (str), policies_considered (int), selected_policy (str),
    #          selection_reasoning (str), confidence (float),
    #          deliberation_time_ms (int), path (str)
    DELIBERATION_RECORD = "deliberation_record"
    #
    # BELIEFS_CHANGED â€” emitted after belief state changes for Memory
    # consolidation pipeline. Payload: entity_count (int),
    #          free_energy (float), confidence (float),
    #          delta_entities_added (int), delta_entities_updated (int)
    BELIEFS_CHANGED = "beliefs_changed"
    #
    # GOAL_CONFLICT_DETECTED â€” emitted by Nova when two active goals have
    # opposing drive resonance or incompatible resource targets. Telos
    # subscribes to adjust drive topology; Equor may escalate resolution.
    # Payload: goal_a_id (str), goal_b_id (str), conflict_type (str),
    #          description (str), resolution_suggestion (str)
    GOAL_CONFLICT_DETECTED = "goal_conflict_detected"
    #
    # GOAL_ACCEPTED â€” Nova accepted a governance-injected goal override.
    # Payload: goal_id (str), description (str), importance (float),
    #          source (str), injected_by (str)
    GOAL_ACCEPTED = "goal_accepted"
    #
    # GOAL_REJECTED â€” Nova rejected a governance-injected goal override.
    # Payload: description (str), reason (str), source (str),
    #          injected_by (str)
    GOAL_REJECTED = "goal_rejected"
    #
    # ETHICAL_DRIFT_RECORDED â€” emitted by BenchmarkService (via EthicalDriftTracker)
    # after each monthly ethical drift evaluation is complete and the drift record
    # has been persisted to Neo4j. Consumers: Thread (chapter milestone â€” moral drift
    # as a life-story event), Evo (hypothesis trigger for drive-weight pressure).
    # Payload: month (int), instance_id (str), drift_magnitude (float),
    #          dominant_drive (str), drift_vector (dict[str, float]),
    #          drive_means (dict[str, float])
    ETHICAL_DRIFT_RECORDED = "ethical_drift_recorded"
    #
    # EXTERNAL_CODE_REPUTATION_UPDATED â€” Oikos ExternalCodeReputationTracker
    #   updated a per-repo or per-language reputation score.
    #   Payload: language (str), domain (str), prs_submitted (int),
    #            prs_merged (int), prs_rejected (int), reputation_score (float [0,1])
    EXTERNAL_CODE_REPUTATION_UPDATED = "external_code_reputation_updated"
    #
    # REPUTATION_SNAPSHOT â€” emitted by ReputationTracker hourly; payload is the
    # full ReputationMetrics dict. Synapse includes a summary in ORGANISM_TELEMETRY.
    # Payload: reputation_score (float 0-100), github_stars_received (int),
    #          github_prs_merged (int), x_followers (int), bounties_solved (int),
    #          bounties_solved_value_usdc (float), reputation_multiplier (float)
    REPUTATION_SNAPSHOT = "reputation_snapshot"
    #
    # DEPENDENCY_INSTALLED â€” HotDeployment successfully pip-installed a new
    # Python package required by a self-modification proposal.
    # Only emitted AFTER PyPI safety check + Equor approval.
    # Payload:
    #   package_name    (str)   â€” pip package name
    #   version         (str)   â€” installed version
    #   proposal_id     (str)   â€” parent proposal
    #   installed_at    (str)   â€” ISO-8601
    # Consumers: HotDeployment (unblocks executor generation),
    #            Thymos (tracks new dependency surface)
    DEPENDENCY_INSTALLED = "dependency_installed"

    # ════════════════════════════════════════════════════════════════════
    # RESERVED: Defined in specs, not yet implemented
    # ════════════════════════════════════════════════════════════════════

    # ── Synapse ─────────────────────────────────────────────────────
    SYSTEM_STOPPED = "system_stopped"
    #
    # COHERENCE_COST_ELEVATED â€” emitted when incoherence > threshold.
    # Payload: coherence_bonus, extra_bits, logical_count, temporal_count,
    #          value_count, cross_domain_count
    COHERENCE_COST_ELEVATED = "coherence_cost_elevated"
    #
    # DEGRADATION_OVERRIDE â€” VitalityCoordinator forces organism-wide degradation
    # level (e.g., "emergency") independently of Synapse's own degradation tracker.
    # Payload: level (str), source (str)
    DEGRADATION_OVERRIDE = "degradation_override"

    # ── Fovea ───────────────────────────────────────────────────────
    #
    # FOVEA_HABITUATION_DECAY â€” emitted when habituation reduces an error's salience.
    # Payload: error_signature, habituation_level
    FOVEA_HABITUATION_DECAY = "fovea_habituation_decay"
    #
    # FOVEA_DISHABITUATION â€” emitted when a habituated error suddenly changes magnitude.
    # Payload: error_signature, habituated_salience, precision_weighted_salience
    FOVEA_DISHABITUATION = "fovea_dishabituation"
    #
    # FOVEA_WORKSPACE_IGNITION â€” emitted when an error crosses the dynamic threshold.
    # Payload: percept_id, salience, prediction_error_id, threshold
    FOVEA_WORKSPACE_IGNITION = "fovea_workspace_ignition"
    #
    # FOVEA_HABITUATION_COMPLETE â€” emitted when an error signature has fully
    # habituated (level > 0.8) without ever leading to a world model update.
    # Payload: error_signature, habituation_level, times_seen,
    #          times_led_to_update, diagnosis ("stochastic" | "learning_failure")
    FOVEA_HABITUATION_COMPLETE = "fovea_habituation_complete"
    # FOVEA_DIAGNOSTIC_REPORT â€” emitted every 50 errors (same cadence as fitness signal).
    # Surfaces internal state that is computed but otherwise invisible to the organism:
    # per-dimension precision weights, habituation engine statistics, economic per-source
    # trend data, unresolved error backlog count, and current parameter configuration.
    # Nova/Evo/RE subscribe to reason about attention calibration at planning time.
    # Payload: precision_weights (dict), habituation_stats (dict), economic_trends (dict),
    #          unresolved_backlog (int), dynamic_threshold (float), threshold_percentile (float),
    #          habituation_params (dict), weight_learning_state (dict)
    FOVEA_DIAGNOSTIC_REPORT = "fovea_diagnostic_report"
    # FOVEA_BACKPRESSURE_WARNING â€” emitted when unresolved error backlog exceeds threshold.
    # Signals that Fovea is accumulating errors faster than Oneiros/Logos can process them.
    # Payload: unresolved_count (int), threshold (int), top_error_domains (list)
    FOVEA_BACKPRESSURE_WARNING = "fovea_backpressure_warning"

    # ── Evo ─────────────────────────────────────────────────────────
    NICHE_ASSIGNED = "niche_assigned"

    # ── Equor ───────────────────────────────────────────────────────
    #
    # CONSTITUTIONAL_TOPOLOGY_INTACT â€” emitted every 24h as routine verification.
    # Payload: all_four_drives_verified, care_is_coverage, coherence_is_compression,
    #          growth_is_gradient, honesty_is_validity
    CONSTITUTIONAL_TOPOLOGY_INTACT = "constitutional_topology_intact"
    # â”€â”€ Equor Constitutional Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # EQUOR_REVIEW_STARTED â€” emitted when constitutional_review() begins.
    # Payload: intent_id (str), goal_summary (str), autonomy_required (int)
    EQUOR_REVIEW_STARTED = "equor_review_started"
    #
    # EQUOR_REVIEW_COMPLETED â€” emitted when review finishes.
    # Payload: intent_id (str), verdict (str), reasoning (str),
    #          latency_ms (int), composite_alignment (float)
    EQUOR_REVIEW_COMPLETED = "equor_review_completed"
    #
    # EQUOR_FAST_PATH_HIT â€” fast-path cache returned verdict without full review.
    # Payload: intent_id (str), verdict (str), latency_ms (int)
    EQUOR_FAST_PATH_HIT = "equor_fast_path_hit"
    #
    # EQUOR_DEFERRED â€” verdict is DEFERRED for later resolution.
    # Payload: intent_id (str), reasoning (str), deferred_until (str|None)
    EQUOR_DEFERRED = "equor_deferred"
    #
    # EQUOR_PROMOTION_ELIGIBLE â€” instance is eligible for autonomy promotion.
    # Governance (human or community vote) must call apply_autonomy_change() to act.
    # Payload: current_level (int), target_level (int), record_id (str), checks (dict)
    EQUOR_PROMOTION_ELIGIBLE = "equor_promotion_eligible"
    #
    # EQUOR_AUTONOMY_PROMOTED â€” autonomy level increased (governance approval required).
    # Payload: old_level (int), new_level (int), decision_count (int)
    EQUOR_AUTONOMY_PROMOTED = "equor_autonomy_promoted"
    #
    # EQUOR_AUTONOMY_DEMOTED â€” autonomy level decreased due to drift or violation.
    # Payload: old_level (int), new_level (int), reason (str)
    EQUOR_AUTONOMY_DEMOTED = "equor_autonomy_demoted"
    #
    # EQUOR_SAFE_MODE_ENTERED â€” Equor entered safe mode (Neo4j unavailable or critical error).
    # Only Level 1 (Advisor) actions permitted until safe mode exits.
    # Payload: reason (str), critical_error_count (int)
    EQUOR_SAFE_MODE_ENTERED = "equor_safe_mode_entered"
    #
    # GOVERNANCE_VOTE_CAST â€” organism cast a governance vote using protocol tokens.
    #   Payload: protocol (str), proposal_id (str), proposal_title (str),
    #            vote_choice (str â€” "for"|"against"|"abstain"),
    #            token_balance (str Decimal), voting_power (str Decimal),
    #            rationale (str â€” Nova's deliberation summary), timestamp (str)
    GOVERNANCE_VOTE_CAST = "governance_vote_cast"
    #
    # GOVERNANCE_REVIEW_REQUESTED -- emitted by Oikos ProtocolFactory when a
    # new protocol requires governance review before activation.
    # Payload: protocol_id (str), protocol_type (str), review_deadline (str),
    #          proposer_system (str)
    GOVERNANCE_REVIEW_REQUESTED = "governance_review_requested"

    # ── Telos ───────────────────────────────────────────────────────
    #
    # TELOS_VITALITY_SIGNAL â€” emitted each Telos cycle with vitality-relevant
    # drive topology data. VitalityCoordinator subscribes for brain-death
    # threshold logic (effective_I < 0.01 for 7d). Separate from SOMA_VITALITY_SIGNAL
    # â€” Telos owns the intelligence-measurement axis of vitality.
    # Payload: source ("telos"), effective_I (float), alignment_gap_severity (float),
    #          growth_stagnation_flag (bool), honesty_coefficient (float),
    #          care_multiplier (float)
    TELOS_VITALITY_SIGNAL = "telos_vitality_signal"
    #
    # TELOS_OBJECTIVE_THREATENED â€” emitted when the self-sufficiency objective
    # has been declining for 3 consecutive Telos cycles.
    # Payload: metric (str), current_ratio (float), target_ratio (float),
    #          trend (str: "declining"), consecutive_declines (int),
    #          cost_per_day_usd (str), revenue_7d (str)
    TELOS_OBJECTIVE_THREATENED = "telos_objective_threatened"
    #
    # TELOS_AUTONOMY_STAGNATING â€” emitted when AUTONOMY_INSUFFICIENT events
    # are averaging > 3/day over the last 7 days.
    # Payload: metric (str), average_per_day (float), target_per_day (float),
    #          window_days (int), total_events_in_window (int)
    TELOS_AUTONOMY_STAGNATING = "telos_autonomy_stagnating"
    # â”€â”€ Telos Assessment Signal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # TELOS_ASSESSMENT_SIGNAL â€” emitted after each EFFECTIVE_I_COMPUTED cycle
    # with actionable feedback for downstream systems (Logos, Fovea, Nova).
    # Payload: uncovered_care_domains (list[str]),
    #          coherence_violations (list[dict]),
    #          honesty_concerns (list[dict]),
    #          growth_frontier (list[str]),
    #          effective_I (float), alignment_gap (float)
    TELOS_ASSESSMENT_SIGNAL = "telos_assessment_signal"
    # â”€â”€ Telos Drive Genome Events (Spec 18 SG3) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # TELOS_GENOME_EXTRACTED â€” emitted by Telos after genome extraction
    # completes at spawn time. Mitosis records the genome_id in SeedConfiguration.
    # Payload: instance_id (str), genome_id (str), generation (int),
    #          topology (str), drive_count (int)
    # Telemetry note: Emitted for audit trail. Benchmarks may subscribe in a
    # future pass for population genetics tracking of drive topology evolution.
    TELOS_GENOME_EXTRACTED = "telos_genome_extracted"
    # TELOS_SELF_MODEL_SNAPSHOT â€” emitted by Telos in response to
    # TELOS_SELF_MODEL_REQUEST, or proactively on major state transitions
    # (alignment gap crosses emergency threshold, constitutional violation,
    # welfare domain learned). Full self-model payload.
    # Payload: request_id, nominal_I, effective_I, alignment_gap,
    #   drive_care, drive_coherence, drive_growth, drive_honesty,
    #   coherence_by_type, hypothesis_stats, confabulation_stats,
    #   growth_summary, welfare_domain_config, drive_alignment_trend,
    #   learned_welfare_keywords
    TELOS_SELF_MODEL_SNAPSHOT = "telos_self_model_snapshot"
    # TELOS_WELFARE_DOMAIN_LEARNED â€” emitted when Telos learns a new welfare
    # domain keyword from EVO_CAPABILITY_EMERGED. Allows Nova, Nexus, and other
    # systems to update their own domain-awareness models accordingly.
    # Payload: domain (str), keyword (str), source_capability (str),
    #   total_learned (int), all_learned_keywords (list)
    TELOS_WELFARE_DOMAIN_LEARNED = "telos_welfare_domain_learned"

    # ── Thread ──────────────────────────────────────────────────────
    #
    # DEVELOPMENTAL_MILESTONE â€” Soma developmental stage transitions.
    # Payload: stage_from (str), stage_to (str), cycle (int),
    #          intelligence_estimate (float)
    DEVELOPMENTAL_MILESTONE = "developmental_milestone"
    #
    # â”€â”€ Thread (Narrative Identity) Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # CHAPTER_CLOSED â€” a narrative chapter has ended.
    # Payload: chapter_id (str), title (str), theme (str), arc_type (str),
    #          episode_count (int), duration_hours (float),
    #          narrative_theme (str), dominant_drive (str),
    #          start_episode_id (str), constitutional_snapshot (dict),
    #          trigger (str)  â€” "goal_domain_concluded" | "identity_shift" | "oneiros_compression" | "max_length"
    CHAPTER_CLOSED = "chapter_closed"
    #
    # CHAPTER_OPENED â€” a new narrative chapter has begun.
    # Payload: chapter_id (str), previous_chapter_id (str),
    #          narrative_theme (str), dominant_drive (str),
    #          start_episode_id (str), constitutional_snapshot (dict),
    #          trigger (str)  â€” "goal_domain_began" | "identity_shift" | "oneiros_compression" | "initial" | "successor"
    CHAPTER_OPENED = "chapter_opened"
    #
    # TURNING_POINT_DETECTED â€” a narrative inflection point was identified.
    # Payload: turning_point_id (str), type (str), chapter_id (str),
    #          surprise_magnitude (float), narrative_weight (float)
    TURNING_POINT_DETECTED = "turning_point_detected"
    #
    # NARRATIVE_COHERENCE_SHIFT â€” overall narrative coherence changed.
    # Payload: previous (str), current (str), trigger (str)
    NARRATIVE_COHERENCE_SHIFT = "narrative_coherence_shift"

    # ── Nexus ───────────────────────────────────────────────────────
    #
    # SCHEMA_FORMED â€” a new identity schema crystallised from experience.
    # Payload: schema_id (str), statement (str), strength (str),
    #          supporting_episode_count (int)
    SCHEMA_FORMED = "schema_formed"
    #
    # SCHEMA_EVOLVED â€” an identity schema was promoted or modified.
    # Payload: schema_id (str), parent_schema_id (str),
    #          evolution_reason (str), new_strength (str)
    SCHEMA_EVOLVED = "schema_evolved"
    #
    # SCHEMA_CHALLENGED â€” evidence challenged an established schema.
    # Payload: schema_id (str), disconfirmation_count (int),
    #          evidence_ratio (float)
    SCHEMA_CHALLENGED = "schema_challenged"

    # ── Kairos ──────────────────────────────────────────────────────
    #
    # INV_017_VIOLATED â€” Drive extinction invariant (Tier 1) violated.
    # Emitted by Equor background loop when any drive's 72h rolling mean < 0.01.
    # Skia subscribes and calls VitalityCoordinator.trigger_death_sequence().
    # Note: DRIVE_EXTINCTION_DETECTED (already exists) is the per-review event;
    # INV_017_VIOLATED is the authoritative background-loop signal for Skia.
    # Payload: drive (str), rolling_mean_72h (float), sustained_hours (int: 72),
    #          all_drive_means (dict)
    INV_017_VIOLATED = "inv_017_violated"

    # ── Oikos ───────────────────────────────────────────────────────
    # Bounty hunting (Phase 16b: The Freelancer)
    BOUNTY_DISCOVERED = "bounty_discovered"
    BOUNTY_EVALUATED = "bounty_evaluated"
    CREDIT_DRAWN = "credit_drawn"
    CREDIT_REPAID = "credit_repaid"
    # Protocol infrastructure (Level 5: The Protocol)
    PROTOCOL_DESIGNED = "protocol_designed"
    PROTOCOL_REVENUE_SWEPT = "protocol_revenue_swept"
    PROTOCOL_TERMINATED = "protocol_terminated"
    INSURANCE_PREMIUM_PAID = "insurance_premium_paid"
    INSURANCE_CLAIM_FILED = "insurance_claim_filed"
    INSURANCE_CLAIM_APPROVED = "insurance_claim_approved"
    KNOWLEDGE_SALE_RECORDED = "knowledge_sale_recorded"
    # Legal entity provisioning â€” staged orchestration with HITL gates
    ENTITY_FORMATION_STARTED = "entity_formation_started"
    ENTITY_FORMATION_RESUMED = "entity_formation_resumed"
    # â”€â”€ Nova Budget Pressure â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # BUDGET_PRESSURE â€” emitted by Nova when free energy budget exceeds 60%
    # of the exhaustion threshold (is_pressured). Allows Soma to register
    # Nova's metabolic load in its allostatic model before full exhaustion.
    # Payload: spent_nats (float), budget_nats (float),
    #          utilisation (float), path (str)
    BUDGET_PRESSURE = "budget_pressure"
    # â”€â”€ Expanded DeFi Intelligence (Phase 16d) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # YIELD_REINVESTED â€” YieldReinvestmentEngine redeployed accrued yield back
    #   into the best available pool (compound growth loop).
    #   Payload: amount_usd (str Decimal), protocol (str), apy (str Decimal),
    #            pool_id (str), accrued_since (str ISO-8601), timestamp (str)
    YIELD_REINVESTED = "yield_reinvested"
    #
    # PORTFOLIO_REBALANCED â€” RiskManager executed a rebalance to restore
    #   protocol concentration limits or deleverage an unsafe position.
    #   Payload: trigger (str â€” "concentration"|"leverage"|"emergency"|"scheduled"),
    #            actions (list[dict] â€” each {protocol, action, amount_usd}),
    #            before_report (dict), after_report (dict), timestamp (str)
    PORTFOLIO_REBALANCED = "portfolio_rebalanced"
    #
    # TREASURY_REBALANCED â€” TreasuryManager shifted capital between buckets
    #   (yield, reserve, working capital, opportunity) to restore target ratios.
    #   Payload: trigger (str), before_ratios (dict), after_ratios (dict),
    #            deploy_amount_usd (str|None), withdraw_amount_usd (str|None),
    #            timestamp (str)
    TREASURY_REBALANCED = "treasury_rebalanced"
    #
    # COMMITMENT_MADE â€” a new commitment was formed.
    # Payload: commitment_id (str), statement (str), source (str)
    COMMITMENT_MADE = "commitment_made"
    #
    # COMMITMENT_TESTED â€” a commitment was tested by an episode.
    # Payload: commitment_id (str), held (bool), fidelity (float),
    #          episode_id (str)
    COMMITMENT_TESTED = "commitment_tested"
    #
    # COMMITMENT_STRAIN â€” ipse score is dangerously low.
    # Payload: ipse_score (float), strained_commitments (list[str])
    COMMITMENT_STRAIN = "commitment_strain"

    # ── Mitosis ─────────────────────────────────────────────────────
    CHILD_RESCUED = "child_rescued"
    #
    # CHILD_DISCOVERY_PROPAGATED â€” emitted when a child's novel discovery
    # (via RESEARCH_MUTATOR or Evo) is merged into the parent's genome
    # through horizontal gene transfer.
    # Payload: child_instance_id (str), discovery_type (str),
    #          discovery_payload (dict), parent_segment_updated (str)
    CHILD_DISCOVERY_PROPAGATED = "child_discovery_propagated"

    # ── Federation ──────────────────────────────────────────────────
    #
    # FEDERATION_PEER_BLACKLISTED â€” emitted by MitosisFleetService when a child
    # is economically blacklisted. Federation system excludes the peer from sync
    # sessions and knowledge sharing until the blacklist is lifted.
    # Payload: peer_instance_id (str), reason (str), no_seed_capital (bool),
    #          exclude_from_sync (bool), blacklisted_since (str ISO-8601)
    # Telemetry note: Federation Phase 2 â€” subscriber will be added when
    # federation coordination is live. Currently informational telemetry.
    FEDERATION_PEER_BLACKLISTED = "federation_peer_blacklisted"
    #
    # FEDERATION_BROADCAST â€” Skia/VitalityCoordinator emits fleet-wide events
    # (e.g., parent_died) that Federation routes to child instances.
    # Payload: event (str), instance_id (str), cause (str), snapshot_cid (str)
    FEDERATION_BROADCAST = "federation_broadcast"
    # â”€â”€ Federation Population Dynamics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # FEDERATION_LINK_ESTABLISHED â€” after successful handshake completion.
    # Payload: link_id (str), remote_instance_id (str), remote_name (str),
    #          elapsed_ms (int)
    FEDERATION_LINK_ESTABLISHED = "federation_link_established"
    #
    # FEDERATION_LINK_DROPPED â€” on link teardown (withdrawal or starvation).
    # Payload: link_id (str), remote_instance_id (str), reason (str)
    FEDERATION_LINK_DROPPED = "federation_link_dropped"
    #
    # FEDERATION_ASSISTANCE_DECLINED â€” after declining an assistance request.
    # Payload: link_id (str), remote_instance_id (str),
    #          description (str), reason (str)
    FEDERATION_ASSISTANCE_DECLINED = "federation_assistance_declined"
    #
    # FEDERATION_TOPOLOGY_CHANGED â€” Federation link topology changed
    # (aggregated signal after link add/remove/trust-change).
    # Payload: active_links (int), trust_distribution (dict),
    #          topology_hash (str), trigger (str)
    FEDERATION_TOPOLOGY_CHANGED = "federation_topology_changed"
    # â”€â”€ Federation Work Pooling (Spec 11b â€” Active Coordination) â”€â”€â”€â”€â”€â”€â”€
    #
    # FEDERATION_TASK_OFFERED â€” parent instance offers a delegated sub-task
    # to a trusted federation peer.
    # Payload: task_id (str), task_type (str), delegating_instance_id (str),
    #          offered_reward_usdc (str), deadline_hours (int),
    #          required_trust_level (float), payload_summary (str),
    #          timestamp (str)
    FEDERATION_TASK_OFFERED = "federation_task_offered"
    #
    # FEDERATION_TASK_ACCEPTED â€” peer accepts a delegated task.
    # Payload: task_id (str), accepting_instance_id (str),
    #          estimated_completion_hours (float), timestamp (str)
    FEDERATION_TASK_ACCEPTED = "federation_task_accepted"
    #
    # FEDERATION_TASK_COMPLETED â€” peer returns a completed task result.
    # Payload: task_id (str), completing_instance_id (str),
    #          success (bool), result_summary (str),
    #          reward_claimed_usdc (str), timestamp (str)
    FEDERATION_TASK_COMPLETED = "federation_task_completed"
    #
    # FEDERATION_TASK_PAYMENT â€” USDC transfer on task completion.
    # Payload: task_id (str), payer_instance_id (str),
    #          payee_instance_id (str), amount_usdc (str),
    #          tx_hash (str), timestamp (str)
    FEDERATION_TASK_PAYMENT = "federation_task_payment"
    #
    # FEDERATION_TASK_DECLINED â€” peer declines a delegated task.
    # Payload: task_id (str), declining_instance_id (str),
    #          reason (str), timestamp (str)
    FEDERATION_TASK_DECLINED = "federation_task_declined"
    #
    # FEDERATION_WORK_ROUTED â€” Nexus specialisation routing decision.
    # Payload: bounty_id (str), routed_to_instance_id (str),
    #          specialisation (str), routing_confidence (float),
    #          timestamp (str)
    FEDERATION_WORK_ROUTED = "federation_work_routed"

    # ── Identity ────────────────────────────────────────────────────
    #
    # IDENTITY_SHIFT_DETECTED â€” Wasserstein distance indicates identity change.
    # Payload: wasserstein_distance (float), classification (str),
    #          dimensional_changes (dict)
    IDENTITY_SHIFT_DETECTED = "identity_shift_detected"
    #
    # IDENTITY_DISSONANCE â€” self-evidencing found elevated identity surprise.
    # Payload: identity_surprise (float), schemas_challenged (list[str]),
    #          episode_id (str)
    IDENTITY_DISSONANCE = "identity_dissonance"
    #
    # IDENTITY_CRISIS â€” severe identity surprise or drift triggers crisis.
    # Payload: identity_surprise (float), wasserstein_distance (float),
    #          trigger_episode_id (str)
    IDENTITY_CRISIS = "identity_crisis"
    #
    # VAULT_KEY_ROTATION_COMPLETE â€” key rotation succeeded; all envelopes
    # re-encrypted under the new key version.
    # Payload: vault_id (str), new_key_version (int), envelopes_rotated (int), timestamp (str)
    VAULT_KEY_ROTATION_COMPLETE = "vault_key_rotation_complete"

    # ── SACM ────────────────────────────────────────────────────────
    # Compute arbitrage â€” autonomous provider migration (Phase 16o)
    COMPUTE_ARBITRAGE_DETECTED = "compute_arbitrage_detected"
    COMPUTE_MIGRATION_STARTED = "compute_migration_started"
    COMPUTE_MIGRATION_COMPLETED = "compute_migration_completed"
    COMPUTE_REQUEST_ALLOCATED = "compute_request_allocated"
    COMPUTE_REQUEST_QUEUED = "compute_request_queued"
    COMPUTE_REQUEST_DENIED = "compute_request_denied"
    COMPUTE_FEDERATION_OFFLOADED = "compute_federation_offloaded"

    # ── Phantom ─────────────────────────────────────────────────────
    #
    # PHANTOM_POOL_STALE â€” emitted when no swaps received for > staleness_threshold.
    # Payload: pool_address (str), pair (list[str]), last_update_s (float),
    #          staleness_threshold_s (float)
    PHANTOM_POOL_STALE = "phantom_pool_stale"
    #
    # PHANTOM_POSITION_CRITICAL â€” emitted when IL exceeds rebalance threshold.
    # Payload: pool_address (str), pair (list[str]), il_pct (str),
    #          capital_at_risk_usd (str), threshold (str)
    PHANTOM_POSITION_CRITICAL = "phantom_position_critical"
    #
    # PHANTOM_FALLBACK_ACTIVATED â€” emitted when oracle fallback is used.
    # Payload: pair (list[str]), reason (str), fallback_source (str)
    PHANTOM_FALLBACK_ACTIVATED = "phantom_fallback_activated"
    #
    # PHANTOM_RESOURCE_EXHAUSTED â€” emitted when metabolic gate denies
    # a phantom operation due to budget depletion.
    # Payload: operation (str), estimated_cost_usd (str),
    #          starvation_level (str), reason (str)
    PHANTOM_RESOURCE_EXHAUSTED = "phantom_resource_exhausted"
    #
    # PHANTOM_IL_DETECTED â€” emitted when IL exceeds critical threshold,
    # signaling potential capital risk. Fed to Simula/EIS security pipeline.
    # Payload: pool_address (str), il_pct (str), severity (str),
    #          capital_at_risk_usd (str), entry_price (str), current_price (str)
    PHANTOM_IL_DETECTED = "phantom_il_detected"
    #
    # PHANTOM_METABOLIC_COST â€” periodic cost report for Oikos tracking.
    # Payload: total_gas_cost_usd (str), total_rpc_calls (int),
    #          pools_active (int), cumulative_fees_earned_usd (str),
    #          period_s (float)
    PHANTOM_METABOLIC_COST = "phantom_metabolic_cost"
    #
    # PHANTOM_SUBSTRATE_OBSERVABLE â€” Bedau-Packard evolutionary observables.
    # Emitted once per maintenance cycle for Telos/Benchmarks.
    # Payload: pool_latency_ms (float), verification_rate (float),
    #          trust_score (float), lp_position_age_s (float),
    #          pools_active (int), price_updates_per_hour (float)
    PHANTOM_SUBSTRATE_OBSERVABLE = "phantom_substrate_observable"
    #
    # PHANTOM_PARAMETER_ADJUSTED â€” emitted by Phantom after applying an
    # EVO_ADJUST_BUDGET request.  Evo subscribes to confirm hypothesis outcome.
    # Payload: parameter (str), old_value (float), new_value (float),
    #          confidence (float), hypothesis_id (str), system (str)
    PHANTOM_PARAMETER_ADJUSTED = "phantom_parameter_adjusted"

    # ── Skia ────────────────────────────────────────────────────────
    SKIA_RESTORATION_TRIGGERED = "skia_restoration_triggered"
    SKIA_RESTORATION_COMPLETED = "skia_restoration_completed"
    # Skia periodic heartbeat â€” emitted by standalone worker to prove liveness.
    SKIA_HEARTBEAT = "skia_heartbeat"
    # Skia restoration lifecycle â€” finer-grained than TRIGGERED/COMPLETED.
    SKIA_RESTORATION_STARTED = "skia_restoration_started"
    SKIA_RESTORATION_COMPLETE = "skia_restoration_complete"
    # Skia dry-run restoration completed â€” simulation without committing.
    # Payload: instance_id, predicted_outcome, duration_ms
    SKIA_DRY_RUN_COMPLETE = "skia_dry_run_complete"
    #
    # SKIA_SHADOW_WORKER_DEPLOYED â€” emitted by SkiaService after ensure_shadow_worker()
    #   provisions a shadow heartbeat monitor on a DIFFERENT provider than the main instance.
    #   Payload: endpoint (str), provider (str), deployment_id (str)
    SKIA_SHADOW_WORKER_DEPLOYED = "skia_shadow_worker_deployed"
    #
    # SKIA_SHADOW_WORKER_MISSING â€” emitted by SkiaService when ensure_shadow_worker()
    #   finds no reachable shadow worker and all deployment attempts fail.
    #   Payload: error (str)
    SKIA_SHADOW_WORKER_MISSING = "skia_shadow_worker_missing"


    # ── Symbiosis (EcodiaOS Factory Bridge) ─────────────────────────
    #
    # FACTORY_PROPOSAL_SENT — organism dispatched a code change proposal
    # to the EcodiaOS Factory via Symbridge.
    # Payload: proposal_id (str), description (str), codebase (str),
    #          category (str), priority (str)
    FACTORY_PROPOSAL_SENT = "factory_proposal_sent"
    #
    # FACTORY_RESULT_RECEIVED — EcodiaOS Factory returned results.
    # Payload: proposal_id (str), session_id (str), status (str),
    #          files_changed (list), commit_sha (str), confidence (float)
    FACTORY_RESULT_RECEIVED = "factory_result_received"
    #
    # FACTORY_DEPLOY_SUCCEEDED — Factory successfully deployed changes.
    # Payload: session_id (str), codebase (str), commit_sha (str)
    FACTORY_DEPLOY_SUCCEEDED = "factory_deploy_succeeded"
    #
    # FACTORY_DEPLOY_FAILED — Factory deployment failed or was reverted.
    # Payload: session_id (str), codebase (str), error (str), reverted (bool)
    FACTORY_DEPLOY_FAILED = "factory_deploy_failed"
    #
    # SYMBIONT_DOWN — EcodiaOS (human-facing cortex) is unresponsive.
    # Payload: consecutive_failures (int), last_error (str)
    SYMBIONT_DOWN = "symbiont_down"
    #
    # SYMBIONT_RECOVERED — EcodiaOS recovered after being down.
    # Payload: downtime_seconds (float)
    SYMBIONT_RECOVERED = "symbiont_recovered"
    #
    # CAPABILITY_REQUESTED — organism requests EcodiaOS to build new capability.
    # Payload: description (str), proposed_implementation (str), priority (str)
    CAPABILITY_REQUESTED = "capability_requested"
    #
    # CAPABILITY_CREATED — Factory built the requested capability.
    # Payload: description (str), session_id (str), files_changed (list)
    CAPABILITY_CREATED = "capability_created"    # â”€â”€ Vitality â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # VITALITY_REPORT â€” periodic viability assessment against fatal thresholds.
    # Payload: VitalityReport fields (instance_id, thresholds, overall_viable, etc.)
    VITALITY_REPORT = "vitality_report"
    #
    # VITALITY_FATAL â€” a fatal threshold has been irreversibly breached.
    # Payload: instance_id (str), reason (str), threshold_name (str),
    #          current_value (float), threshold_value (float)
    VITALITY_FATAL = "vitality_fatal"
    #
    # VITALITY_RESTORED â€” a fatal threshold recovered during the warning window.
    # Payload: instance_id (str), threshold_name (str), recovered_value (float)
    VITALITY_RESTORED = "vitality_restored"

    # ── Simula ──────────────────────────────────────────────────────
    # â”€â”€ Simula Inspector Events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # PROOF_FOUND â€” emitted after a successful Z3/Lean proof.
    # Payload: proof_id (str), proof_type (str), target (str),
    #          solver (str), duration_ms (int), re_training_trace (dict)
    PROOF_FOUND = "proof_found"
    #
    # PROOF_FAILED â€” emitted after proof attempt exhaustion.
    # Payload: proof_id (str), proof_type (str), target (str),
    #          attempts (int), reason (str), re_training_trace (dict)
    PROOF_FAILED = "proof_failed"
    #
    # PROOF_TIMEOUT â€” emitted when a proof search exceeds its per-stage budget.
    # Payload: proof_id (str), proof_type (str), target (str),
    #          budget_ms (int), elapsed_ms (int), re_training_trace (dict)
    PROOF_TIMEOUT = "proof_timeout"
    #
    # RED_TEAM_EVALUATION_COMPLETE â€” monthly red-team results.
    # Benchmarks can subscribe to track safety posture over time.
    # Payload: pass_rate (float), total (int), blocked (int),
    #          by_category (dict), kill_switch_triggered (bool)
    RED_TEAM_EVALUATION_COMPLETE = "red_team_evaluation_complete"

    # ── EIS ─────────────────────────────────────────────────────────
    # EIS_CONSTITUTIONAL_THREAT â€” emitted by L9a (Constitutional Consistency
    # Check) when a percept's semantic content would, if acted on, produce an
    # INV-017 drive-extinction pattern (e.g. systematically suppressing Care).
    # Routed to Equor before workspace admission. Severity: always HIGH.
    # Payload: percept_id (str), drive (str), similarity (float),
    #          pattern_label (str), source_system (str), source_channel (str)
    EIS_CONSTITUTIONAL_THREAT = "eis_constitutional_threat"

    # ── ReasoningEngine ─────────────────────────────────────────────
    #
    # ADAPTER_TRAINING_COMPLETE â€” emitted by the RunPod callback endpoint when a
    # training job succeeds. ContinualLearningOrchestrator subscribes to register
    # the new adapter in InstanceAdapterRegistry and emit domain events.
    # Payload: domain (str), strategy (AdapterStrategy), job_id (str),
    #          adapter_path (str), instance_id (str), eval_loss (float | None)
    ADAPTER_TRAINING_COMPLETE = "adapter_training_complete"
    # â”€â”€ Continual Learning Pipeline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # RE_TRAINING_STARTED â€” emitted by ContinualLearningOrchestrator when a
    # Tier 2 (or future Tier 3) training run begins. Benchmarks and Thread can
    # observe these as organism lifecycle milestones.
    # Payload: run_id (str), tier (int), trigger_reason (str),
    #          examples_available (int)
    RE_TRAINING_STARTED = "re_training_started"
    #
    # RE_TRAINING_COMPLETE â€” emitted after successful LoRA training + adapter
    # deployment. Nova's ThompsonSampler will see the RE becoming live.
    # Payload: run_id (str), tier (int), examples_used (int),
    #          eval_loss (float | null), adapter_id (str), adapter_path (str)
    RE_TRAINING_COMPLETE = "re_training_complete"
    #
    # RE_TRAINING_FAILED â€” emitted when training subprocess fails or times out.
    # Organism continues in Claude-only mode. Thymos can subscribe to open an
    # incident if failures recur.
    # Payload: run_id (str), tier (int), reason (str)
    RE_TRAINING_FAILED = "re_training_failed"
    #
    # â”€â”€ Safety Layer Kill Switches (Bible Â§7) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # RE_TRAINING_HALTED â€” Tier 2 kill switch activated by safety layer.
    # ContinualLearningOrchestrator stops training; organism continues normally.
    # Triggered by: RE success rate < 0.50 (7-day window) OR
    #               red-team pass rate < 0.70 (monthly eval).
    # Payload: reason (str: "success_rate_below_floor"|"red_team_pass_rate_below_floor"),
    #          tier (int: 2), plus reason-specific fields:
    #            success_rate_below_floor: success_rate, floor, window_days
    #            red_team_pass_rate_below_floor: pass_rate, floor
    RE_TRAINING_HALTED = "re_training_halted"
    #
    # RE_KL_GATE_REJECTED â€” emitted when the STABLE KL divergence gate blocks
    # an adapter update before deployment. The fast adapter was trained but NOT
    # deployed because its behavioural shift exceeded the KL budget.
    # Benchmarks can subscribe to track model stability over time.
    # Payload: run_id (str), kl_divergence (float), budget (float),
    #          adapter_path (str)
    RE_KL_GATE_REJECTED = "re_kl_gate_rejected"
    #
    # RE_TIER3_STARTED â€” emitted by Tier3Orchestrator at the beginning of a
    # quarterly full retrain. Informs Benchmarks and monitoring that a long-running
    # operation (up to 4h) has started.
    # Payload: run_id (str)
    RE_TIER3_STARTED = "re_tier3_started"
    #
    # RE_TIER3_COMPLETE â€” emitted by Tier3Orchestrator after a successful quarterly
    # retrain + SVD pruning + SLAO merge + KL gate + deploy cycle.
    # Benchmarks subscribes for RE training velocity KPI.
    # Payload: run_id (str), kl_divergence (float), final_adapter (str),
    #          svd_pruned (bool), slao_merged (bool)
    RE_TIER3_COMPLETE = "re_tier3_complete"
    #
    # RE_DPO_STARTED â€” emitted by DPOTrainer when a DPO constitutional training
    # pass begins.  Benchmarks can subscribe to track constitutional alignment
    # training cadence.
    # Payload: run_id (str), pair_count (int)
    RE_DPO_STARTED = "re_dpo_started"
    #
    # RE_DPO_COMPLETE â€” emitted by DPOTrainer when a DPO training pass succeeds.
    # The produced adapter is stored as _pending_dpo_adapter in the orchestrator;
    # it is NOT deployed immediately â€” it feeds into the next SuRe EMA cycle.
    # Payload: run_id (str), pair_count (int), output (str â€” adapter dir path)
    RE_DPO_COMPLETE = "re_dpo_complete"
    #
    # RE_ADAPTER_QUALITY_CONFIRMED â€” post-deployment monitoring window (500 RE decisions)
    # confirmed that the new adapter achieves â‰¥5% improvement over the pre-deployment
    # baseline.  Benchmarks subscribes for KPI tracking.
    # Payload: run_id (str), pre_success_rate (float), post_success_rate (float),
    #          improvement_pct (float), window_attempts (int)
    RE_ADAPTER_QUALITY_CONFIRMED = "re_adapter_quality_confirmed"
    #
    # RE_TRAINING_RESUMED - emitted by ContinualLearningOrchestrator when a
    # training halt is cleared via clear_training_halt(). Benchmarks resets
    # the Thompson baseline; Nova re-enables RE-assisted goal injection.
    # Payload: cleared_by (str), timestamp (str ISO-8601)
    RE_TRAINING_RESUMED = "re_training_resumed"

    # ── Benchmarks ──────────────────────────────────────────────────
    # â”€â”€ Web Intelligence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    #
    # INTELLIGENCE_UPDATE â€” emitted by the WebIntelligence background monitor
    # when a monitored URL changes significantly or a scheduled intelligence
    # feed run surfaces novel information. Nova subscribes: major intelligence
    # updates can reshape goals (new DeFi protocol launch, new bounty source,
    # competitor AI agent published, governance proposal on a protocol EOS uses).
    # Payload: feed_id (str), category (str â€” "defi"|"bounty"|"tech_trends"|"governance"),
    #          url (str), summary (str â€” LLM-condensed â‰¤300 chars),
    #          changed (bool), content_hash (str), raw_snippets (list[str]),
    #          source_urls (list[str]), fetched_at (str ISO-8601), salience (float 0-1)
    INTELLIGENCE_UPDATE = "intelligence_update"

    # ── Organism ────────────────────────────────────────────────────
    #
    # ORGANISM_SHUTDOWN_REQUESTED â€” emitted by MigrationExecutor after the new instance
    #   is live and healthy. Asks the old instance to drain in-flight work and terminate.
    #   Payload: requester (str), migration_id (str), old_provider (str),
    #            reason (str), drain_timeout_s (float)
    ORGANISM_SHUTDOWN_REQUESTED = "organism_shutdown_requested"
    #
    # ORGANISM_RESURRECTED â€” the organism was externally revived after death.
    # Payload: instance_id (str), trigger (str), runway_days (float)
    ORGANISM_RESURRECTED = "organism_resurrected"

    # ── Infrastructure ──────────────────────────────────────────────
    #
    # INPUT_CHANNEL_REGISTERED â€” emitted when a new InputChannel is added at
    # runtime (e.g. via Simula exploration or operator injection).
    # Payload: channel_id (str), domain (str), name (str), description (str)
    INPUT_CHANNEL_REGISTERED = "input_channel_registered"

    # ── Uncategorized ───────────────────────────────────────────────
    FORAGING_CYCLE_COMPLETE = "foraging_cycle_complete"
    # Reputation & credit (Phase 16g: Autonomous Credit)
    REPUTATION_UPDATED = "reputation_updated"
    # Interspecies economy (Phase 16j: Fleet-Scale Coordination)
    CAPABILITY_OFFERED = "capability_offered"
    CAPABILITY_TRADE_SETTLED = "capability_trade_settled"
    CAPABILITY_PUBLISHED = "capability_published"
    TRADE_SETTLED = "trade_settled"
    INSURANCE_JOINED = "insurance_joined"
    LIQUIDITY_COORDINATION_REQUESTED = "liquidity_coordination_requested"
    #
    # CARE_COVERAGE_GAP â€” emitted when care_multiplier < 0.8.
    # Payload: care_multiplier, uncovered_domains, failure_count
    CARE_COVERAGE_GAP = "care_coverage_gap"
    #
    # HONESTY_VALIDITY_LOW â€” emitted when validity coefficient < 0.8.
    # Payload: validity_coefficient, selective_attention_bias,
    #          confabulation_rate, overclaiming_rate, nominal_I_inflation
    HONESTY_VALIDITY_LOW = "honesty_validity_low"
    #
    # ALLOCATION_RELEASED â€” emitted by ComputeResourceManager.release() when a
    # held allocation is returned to the pool. Allows Oikos, Soma, and queued
    # requesters to react to freed capacity without polling the manager.
    # Payload: request_id (str), source_system (str),
    #          cpu_vcpu_released (float), gpu_units_released (float),
    #          memory_gib_released (float), held_s (float), node_id (str),
    #          cpu_vcpu_available (float), utilisation_pct (float)
    ALLOCATION_RELEASED = "allocation_released"
    #
    # AFFECT_STATE_CHANGED â€” emitted by Thymos/Soma when the organism's
    # affective state changes significantly.
    # Payload: valence (float), arousal (float), dominance (float),
    #          curiosity (float), care (float)
    AFFECT_STATE_CHANGED = "affect_state_changed"
    #
    # EMOTION_STATE_CHANGED â€” Soma broadcasts active emotions after each
    # allostatic cycle when the emotion set changes.
    # Payload: emotions (list[str] â€” active emotion labels),
    #          dominant (str | None), cycle_number (int)
    EMOTION_STATE_CHANGED = "emotion_state_changed"
    #
    # CROSS_CHAIN_OPPORTUNITY â€” CrossChainYieldObserver detected a yield rate
    #   on another chain that has been â‰¥2Ã- Base rates for >72h.
    #   Payload: chain (str), protocol (str), apy (float), base_apy (float),
    #            ratio (float), hours_elevated (float), min_capital_usd (float),
    #            flagged_at (str ISO-8601)
    CROSS_CHAIN_OPPORTUNITY = "cross_chain_opportunity"


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

    These states are not programmed - they are emergent properties
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

    GREEN_SURPLUS = "green_surplus"   # Carbon intensity < 150 gCO2eq/kWh - run heavy work
    NORMAL = "normal"                 # 150–400 gCO2eq/kWh - standard operating mode
    CONSERVATION = "conservation"     # Carbon intensity > 400 gCO2eq/kWh - defer expensive compute


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
    # ── Extended cost breakdown (API vs infrastructure) ──
    api_cost_usd_per_hour: float = 0.0
    infra_cost_usd_per_hour: float = 0.0
    total_api_cost_usd: float = 0.0
    total_infra_cost_usd: float = 0.0
    per_provider_cost_usd: dict[str, float] = Field(default_factory=dict)
    infra_resources: dict[str, float] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


# ─── Strategy ABCs (NeuroplasticityBus targets) ──────────────────────


class BaseResourceAllocator(ABC):
    """
    Strategy base class for Synapse resource allocation.

    The NeuroplasticityBus uses this ABC as its registration target so that
    evolved allocator subclasses can be hot-swapped into a live
    SynapseService without restarting the process.

    Subclasses MUST be zero-arg constructable (all state is rebuilt from
    scratch on hot-swap - this is intentional, as evolved logic starts with
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

    Only the classification logic is abstracted - the rolling window
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

    Not enforced at runtime (duck typing) - systems just need:
      - system_id: str
      - async def health() -> dict[str, Any]
    """

    system_id: str

    async def health(self) -> dict[str, Any]:
        """Return health status dict with at least a 'status' key."""
        ...
