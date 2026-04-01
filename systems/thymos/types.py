"""
EcodiaOS - Thymos Type Definitions

All data types for the immune system: incidents, antibodies, repairs,
diagnoses, sentinels, and healing governance.

Every error, anomaly, and violation in EOS becomes an Incident - a
first-class primitive alongside Percept, Belief, and Intent.
"""

from __future__ import annotations

import enum
from datetime import datetime
import hashlib
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now
from primitives.incident import IncidentClass, IncidentSeverity

# ─── Enums ────────────────────────────────────────────────────────

# IncidentClass and IncidentSeverity are canonical primitives imported above.
# They are re-exported here for backwards compatibility with existing thymos
# internal code that does `from systems.thymos.types import IncidentClass`.
__all__ = ["IncidentClass", "IncidentSeverity"]


class RepairTier(int, enum.Enum):
    """Escalation ladder - least invasive first."""

    NOOP = 0  # Transient, already resolved
    PARAMETER = 1  # Adjust a configuration value
    RESTART = 2  # Restart the affected system
    KNOWN_FIX = 3  # Apply an antibody from the library
    NOVEL_FIX = 4  # Generate a new fix via Simula Code Agent
    ESCALATE = 5  # Human operator intervention required


class RepairStatus(enum.StrEnum):
    """Lifecycle of an incident repair."""

    PENDING = "pending"
    DIAGNOSING = "diagnosing"
    PRESCRIBING = "prescribing"
    VALIDATING = "validating"
    APPLYING = "applying"
    VERIFYING = "verifying"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    ACCEPTED = "accepted"  # Transient or INFO, no repair needed
    ROLLED_BACK = "rolled_back"


class HealingMode(enum.StrEnum):
    """Organism-wide healing state."""

    NOMINAL = "nominal"  # Normal operation
    HEALING = "healing"  # Active repair in progress
    STORM = "storm"  # Cytokine storm - focus on root cause only
    DEGRADED = "degraded"  # Repair budget exhausted - reduced healing capacity


# ─── Sentinel Types ──────────────────────────────────────────────


class ContractSLA(EOSBaseModel):
    """SLA definition for an inter-system contract."""

    source: str
    target: str
    operation: str
    max_latency_ms: float


class FeedbackLoop(EOSBaseModel):
    """Definition of a feedback loop that should be actively transmitting.

    ``active`` controls how the sentinel treats an unobserved loop at startup:
      - ``True``   - loop is expected to be running; MISSING → HIGH incident.
      - ``False``  - loop is explicitly disabled; never generates incidents.
      - ``None``   - loop's existence is unknown (not yet implemented in code);
                     generates LOW/INFO incidents so the flood doesn't drown real alerts.
    """

    name: str
    source: str
    target: str
    signal: str
    check: str  # Descriptive check expression
    description: str
    active: bool | None = None  # None = unknown/not yet wired in code


class DriftConfig(EOSBaseModel):
    """Configuration for statistical drift detection on a metric."""

    window: int = 500  # Number of samples in the rolling baseline
    sigma_threshold: float = 2.5  # Standard deviations before flagging
    direction: str | None = None  # "above", "below", or None (both)


class StallConfig(EOSBaseModel):
    """Threshold for cognitive stall detection."""

    min_value: float  # Rate must be above this
    window_cycles: int  # Number of cycles to observe


# ─── Economic Immune Types ───────────────────────────────────────


class ThreatType(enum.StrEnum):
    """Categories of economic threats the immune system detects."""

    FLASH_LOAN_ATTACK = "flash_loan_attack"
    PRICE_MANIPULATION = "price_manipulation"
    SUSPICIOUS_CONTRACT = "suspicious_contract"
    MEMPOOL_POISONING = "mempool_poisoning"
    RUG_PULL = "rug_pull"
    ORACLE_MANIPULATION = "oracle_manipulation"
    GOVERNANCE_ATTACK = "governance_attack"


class ThreatSeverity(enum.StrEnum):
    """Severity classification for economic threats."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ProtocolAlert(EOSBaseModel):
    """Alert raised when a DeFi protocol's health degrades."""

    protocol: str
    alert_type: str  # "tvl_drop" | "oracle_deviation" | "contract_paused" | "governance_anomaly"
    current_value: float
    threshold_value: float
    deviation_percent: float
    requires_withdrawal: bool = False


class ThreatPattern(EOSBaseModel):
    """A detection pattern for recognising on-chain threats."""

    pattern_id: str = Field(default_factory=new_id)
    threat_type: ThreatType
    description: str
    detection_rule: str  # Human-readable rule description
    severity: ThreatSeverity
    confidence: float = Field(0.8, ge=0.0, le=1.0)
    false_positive_rate: float = Field(0.05, ge=0.0, le=1.0)


class AddressBlacklistEntry(EOSBaseModel):
    """A blacklisted on-chain address with provenance tracking."""

    address: str
    chain_id: int = 8453  # Base L2
    reason: str
    threat_type: ThreatType
    source: str = "local"  # "local" | "federation" | "external"
    source_instance_id: str = ""
    confirmed: bool = False


class SimulationResult(EOSBaseModel):
    """Result of pre-simulating a transaction before broadcast."""

    passed: bool = True
    revert_reason: str = ""
    gas_used: int = 0
    value_delta_usd: float = 0.0
    slippage_bps: int = 0
    mev_risk_detected: bool = False
    warnings: list[str] = Field(default_factory=list)


# ─── API Error Context ────────────────────────────────────────────


class ApiErrorContext(EOSBaseModel):
    """
    Structured context for incidents originating from the API layer.

    Attached to Incidents reported via report_exception() when the caller
    supplies HTTP request metadata. Enables endpoint-aware deduplication,
    severity routing, and repair hints.
    """

    endpoint: str  # API path, e.g. "/api/v1/logos/health"
    method: str  # HTTP method: GET, POST, etc.
    status_code: int  # HTTP response status
    request_id: str  # Trace/correlation ID
    remote_addr: str  # Client IP address
    latency_ms: float  # Request processing time in milliseconds
    user_agent: str = ""  # Optional User-Agent header
    request_body_summary: str = ""  # First 200 chars, secrets redacted


# ─── Incident ────────────────────────────────────────────────────


class Incident(EOSBaseModel):
    """
    The fundamental immune primitive.

    Every error, anomaly, and violation becomes an Incident.
    Incidents are also Percepts - the organism perceives its own
    failures through the normal workspace broadcast cycle.
    """

    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)

    # ── Classification ──
    incident_class: IncidentClass
    severity: IncidentSeverity
    fingerprint: str  # Hash of (class, system, error_signature)

    # ── Source ──
    source_system: str
    error_type: str  # Exception class name or anomaly type
    error_message: str
    stack_trace: str | None = None
    context: dict[str, Any] | ApiErrorContext = Field(default_factory=dict)

    # ── Impact Assessment ──
    affected_systems: list[str] = Field(default_factory=list)
    blast_radius: float = Field(default=0.0, ge=0.0, le=1.0)
    user_visible: bool = False
    constitutional_impact: dict[str, float] = Field(
        default_factory=lambda: {
            "coherence": 0.0,
            "care": 0.0,
            "growth": 0.0,
            "honesty": 0.0,
        }
    )

    # ── Deduplication ──
    occurrence_count: int = 1
    first_seen: datetime | None = None

    # ── Diagnosis ──
    root_cause_hypothesis: str | None = None
    diagnostic_confidence: float = 0.0
    causal_chain: list[str] | None = None

    # ── Repair ──
    repair_tier: RepairTier | None = None
    repair_status: RepairStatus = RepairStatus.PENDING
    antibody_id: str | None = None
    repair_history: list[RepairAttempt] = Field(default_factory=list)

    # ── Pattern Match ──
    # Set by PatternAwareRouter before process_incident() runs.
    matched_pattern_id: str | None = None       # Redis key suffix of the matched CrashPattern
    pattern_confidence: float = 0.0             # Confidence of the best-matched pattern (0–1)
    tier_skip_reason: str | None = None         # Human-readable reason tiers were skipped

    # ── Learning ──
    resolution_time_ms: int | None = None
    repair_successful: bool | None = None

    def highest_attempted_tier(self) -> RepairTier | None:
        """Return the highest tier that was previously attempted, or None."""
        if not self.repair_history:
            return None
        return max(self.repair_history, key=lambda a: a.tier.value).tier

    def tier_was_attempted(self, tier: RepairTier) -> bool:
        """Check if a specific tier was already attempted for this incident."""
        return any(a.tier == tier for a in self.repair_history)

    def next_escalation_tier(self, current_tier: RepairTier) -> RepairTier:
        """Return the next tier to try based on what's already been attempted.

        Skips tiers that already failed. If all tiers up to NOVEL_FIX have
        been tried, returns ESCALATE.
        """
        tier_order = [
            RepairTier.NOOP,
            RepairTier.PARAMETER,
            RepairTier.RESTART,
            RepairTier.KNOWN_FIX,
            RepairTier.NOVEL_FIX,
            RepairTier.ESCALATE,
        ]
        failed_tiers = {
            a.tier for a in self.repair_history
            if a.outcome in ("failed", "rolled_back")
        }
        # Start from the tier after current_tier
        start_idx = next(
            (i for i, t in enumerate(tier_order) if t == current_tier), 0
        ) + 1
        for tier in tier_order[start_idx:]:
            if tier not in failed_tiers:
                return tier
        return RepairTier.ESCALATE


# ─── Diagnosis Types ─────────────────────────────────────────────


class CausalChain(EOSBaseModel):
    """Result of tracing error causality through the system graph."""

    root_system: str
    chain: list[str]  # System A → System B → failure
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: str = ""


class TemporalCorrelation(EOSBaseModel):
    """Something that changed in the window before the incident."""

    type: str  # "metric_anomaly" | "system_event"
    timestamp: datetime
    description: str
    time_delta_ms: int  # How many ms before the incident


class DiagnosticHypothesis(EOSBaseModel):
    """A testable hypothesis about what caused an incident."""

    id: str = Field(default_factory=new_id)
    statement: str
    diagnostic_test: str  # Name of the test to run
    diagnostic_test_params: dict[str, Any] = Field(default_factory=dict)
    suggested_repair_tier: RepairTier = RepairTier.PARAMETER
    confidence_prior: float = Field(0.5, ge=0.0, le=1.0)


class DiagnosticTestResult(EOSBaseModel):
    """Result of running a diagnostic test."""

    test_name: str
    passed: bool
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    reasoning: str = ""
    raw_value: Any = None


class DiagnosticEvidence(EOSBaseModel):
    """All evidence gathered for diagnosing an incident."""

    incident: Incident
    causal_chain: CausalChain
    temporal_correlations: list[TemporalCorrelation] = Field(default_factory=list)
    recent_similar: list[Incident] = Field(default_factory=list)
    system_health_history: dict[str, Any] = Field(default_factory=dict)


class Diagnosis(EOSBaseModel):
    """Final diagnosis of an incident's root cause."""

    root_cause: str
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    repair_tier: RepairTier = RepairTier.PARAMETER
    antibody_id: str | None = None
    all_hypotheses: list[DiagnosticHypothesis] = Field(default_factory=list)
    test_results: list[DiagnosticTestResult] = Field(default_factory=list)
    reasoning: str = ""


# ─── Repair Types ────────────────────────────────────────────────


class ParameterFix(EOSBaseModel):
    """A single parameter adjustment."""

    parameter_path: str  # e.g., "synapse.clock.current_period_ms"
    delta: float  # Change amount (can be negative)
    reason: str = ""


class RepairSpec(EOSBaseModel):
    """Specification for a repair action."""

    # Stable, deterministic ID for this repair pattern.
    # Format: "repair:{incident_class}:{fingerprint_hash[:12]}"
    # Built via RepairSpec.make_id(); empty string for legacy/unknown specs.
    repair_spec_id: str = ""

    tier: RepairTier
    action: str  # e.g., "log_and_monitor", "restart_system", "apply_antibody"
    target_system: str | None = None
    antibody_id: str | None = None
    parameter_changes: list[dict[str, Any]] = Field(default_factory=list)
    code_changes: dict[str, Any] | None = None
    evolution_proposal_id: str | None = None
    reason: str = ""

    @staticmethod
    def make_id(incident_class: str, fix_type: str) -> str:
        """
        Deterministic repair-pattern ID from incident class + fix type.

        The same repair pattern (same class, same fix) always maps to the
        same ID regardless of which incident triggered it.  This lets the
        hypothesis engine deduplicate correctly across incidents that share
        the same root cause.
        """
        fingerprint = hashlib.sha256(
            f"{incident_class}::{fix_type}".encode()
        ).hexdigest()
        return f"repair:{incident_class}:{fingerprint[:12]}"


class RepairAttempt(EOSBaseModel):
    """Record of a single repair attempt for an incident."""

    tier: RepairTier
    action: str
    outcome: str  # "success" | "failed" | "rejected" | "rolled_back"
    reason: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


class ValidationResult(EOSBaseModel):
    """Result of the repair validation gate."""

    approved: bool
    reason: str = ""
    escalate_to: RepairTier | None = None
    modifications: dict[str, Any] | None = None


# ─── Antibody Types ──────────────────────────────────────────────


class Antibody(EOSBaseModel):
    """
    A crystallized successful repair.

    When a repair succeeds, it becomes an Antibody. The next time an
    incident with the same fingerprint appears, the antibody is applied
    instantly - no diagnosis needed.

    This is genuine adaptive immunity: the organism gets harder to break
    over time.
    """

    id: str = Field(default_factory=new_id)

    # ── Matching ──
    fingerprint: str
    incident_class: IncidentClass
    source_system: str
    error_pattern: str  # Regex or fragment for matching error_message

    # ── Repair ──
    repair_tier: RepairTier
    repair_spec: RepairSpec
    root_cause_description: str

    # ── Effectiveness ──
    application_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    effectiveness: float = 1.0  # success / (success + failure)

    # ── Lifecycle ──
    created_at: datetime = Field(default_factory=utc_now)
    last_applied: datetime | None = None
    source_incident_id: str = ""
    retired: bool = False

    # ── Lineage ──
    generation: int = 1
    parent_antibody_id: str | None = None


# ─── Prophylactic Types ──────────────────────────────────────────


class ProphylacticWarning(EOSBaseModel):
    """Warning issued by the prophylactic scanner."""

    filepath: str
    antibody_id: str
    warning: str
    suggestion: str = ""
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class ParameterAdjustment(EOSBaseModel):
    """A homeostatic parameter nudge - Tier 1, no governance."""

    metric_name: str
    current_value: float
    optimal_min: float
    optimal_max: float
    adjustment: ParameterFix
    trend_direction: str  # "rising" | "falling"


# ─── Governor Types ──────────────────────────────────────────────


class HealingBudgetState(EOSBaseModel):
    """Current state of the healing budget."""

    repairs_this_hour: int = 0
    novel_repairs_today: int = 0
    max_repairs_per_hour: int = 50
    max_novel_repairs_per_day: int = 20
    active_diagnoses: int = 0
    max_concurrent_diagnoses: int = 5
    active_codegen: int = 0
    max_concurrent_codegen: int = 3
    # T4 (Tier 4 = novel repair via Simula) budget for cytokine storm prevention
    t4_proposals_this_hour: int = 0
    max_t4_proposals_per_hour: int = 20
    active_t4_proposals: int = 0
    max_concurrent_t4_proposals: int = 3
    storm_mode: bool = False
    storm_focus_system: str | None = None
    cpu_budget_fraction: float = 0.10


# ─── Health Snapshot ─────────────────────────────────────────────


class ThymosHealthSnapshot(EOSBaseModel):
    """Thymos system health and observability."""

    status: str = "healthy"
    healing_mode: HealingMode = HealingMode.NOMINAL

    # Incident metrics
    total_incidents_created: int = 0
    active_incidents: int = 0
    mean_resolution_ms: float = 0.0
    incidents_by_severity: dict[str, int] = Field(default_factory=dict)
    incidents_by_class: dict[str, int] = Field(default_factory=dict)

    # Antibody metrics
    total_antibodies: int = 0
    mean_antibody_effectiveness: float = 0.0
    antibodies_applied: int = 0
    antibodies_created: int = 0
    antibodies_retired: int = 0

    # Repair metrics
    repairs_attempted: int = 0
    repairs_succeeded: int = 0
    repairs_failed: int = 0
    repairs_rolled_back: int = 0
    repairs_by_tier: dict[str, int] = Field(default_factory=dict)

    # Diagnosis metrics
    diagnoses_run: int = 0
    mean_diagnosis_confidence: float = 0.0
    mean_diagnosis_latency_ms: float = 0.0

    # Homeostasis metrics
    homeostatic_adjustments: int = 0
    metrics_in_range: int = 0
    metrics_total: int = 0

    # Storm metrics
    storm_activations: int = 0

    # Prophylactic metrics
    prophylactic_scans: int = 0
    prophylactic_warnings: int = 0

    # Budget
    budget: HealingBudgetState = Field(default_factory=HealingBudgetState)
    budget_exhausted: bool = False
    degraded_reason: str | None = None

    timestamp: datetime = Field(default_factory=utc_now)
