"""
EcodiaOS - Thymos Service (The Immune System)

The organism's self-healing system. Thymos detects failures, diagnoses root causes,
prescribes repairs, maintains an antibody library of learned fixes, and prevents
future errors through prophylactic scanning and homeostatic regulation.

Every error, anomaly, and violation becomes an Incident - a first-class primitive
alongside Percept, Belief, and Intent. The organism perceives its own failures
through the normal workspace broadcast cycle. It hurts to break.

Immune Pipeline:
  Detect → Deduplicate → Triage → Diagnose → Prescribe → Validate → Apply → Verify → Learn

Iron Rules:
  - Thymos CANNOT modify Equor or constitutional drives
  - Thymos CANNOT suppress or hide errors from the audit trail
  - Thymos CANNOT apply Tier 4 (codegen) repairs without Equor review
  - Thymos CANNOT exceed the healing budget (MAX_REPAIRS_PER_HOUR)
  - Thymos MUST route all Tier 3+ repairs through the validation gate
  - Thymos MUST record every incident, diagnosis, and repair in Neo4j
  - Thymos MUST prefer less invasive repairs (Tier 0 before Tier 1 before ...)
  - Thymos MUST enter storm mode when incident rate exceeds threshold

Cognitive cycle role (step 8 - MAINTAIN):
  Homeostatic checks run on the MAINTAIN step. Non-blocking, background.
  The organism maintains itself the way a body regulates temperature.

Interface:
  initialize()              - build sub-systems, load antibody library
  on_synapse_event()        - convert health events into incidents
  on_incident()             - entry point for the immune pipeline
  process_incident()        - full pipeline: diagnose → prescribe → validate → apply
  maintain_homeostasis()    - proactive health optimization (MAINTAIN step)
  shutdown()                - graceful teardown
  health()                  - self-health report for Synapse
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import math
import os
import time
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import DriveAlignmentVector, SystemID, utc_now
from primitives.re_training import RETrainingExample
from systems.synapse.types import SynapseEvent, SynapseEventType
from core.crash_pattern_analyzer import CrashPattern, CrashPatternAnalyzer
from systems.thymos.antibody import AntibodyLibrary
from systems.thymos.event_payloads import validate_event_payload
from systems.thymos.pattern_router import PatternAwareRouter
from systems.thymos.diagnosis import (
    CausalAnalyzer,
    DiagnosticEngine,
    TemporalCorrelator,
)
from systems.thymos.governor import HealingGovernor
from systems.thymos.notifications import NotificationDispatcher
from systems.thymos.prescription import RepairPrescriber, RepairValidator
from systems.thymos.prophylactic import HomeostasisController, ProphylacticScanner
from systems.thymos.sentinels import (
    _DOWNSTREAM,
    _TOTAL_SYSTEMS,
    _USER_FACING_SYSTEMS,
    BankruptcySentinel,
    BaseThymosSentinel,
    CognitiveStallSentinel,
    ContractSentinel,
    DriftSentinel,
    ExceptionSentinel,
    FeedbackLoopSentinel,
    ProtocolHealthSentinel,
    ThreatPatternSentinel,
)
from systems.thymos.triage import (
    IncidentDeduplicator,
    ResponseRouter,
    SeverityScorer,
)
from systems.thymos.types import (
    ApiErrorContext,
    Diagnosis,
    HealingMode,
    Incident,
    IncidentClass,
    IncidentSeverity,
    RepairAttempt,
    RepairSpec,
    RepairStatus,
    RepairTier,
)

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from clients.embedding import EmbeddingClient
    from clients.llm import LLMProvider
    from clients.neo4j import Neo4jClient
    from clients.redis import RedisClient
    from config import ThymosConfig
    from core.hotreload import NeuroplasticityBus
    from systems.synapse.health import HealthMonitor
    from systems.synapse.service import SynapseService
    from telemetry.metrics import MetricCollector
logger = structlog.get_logger("systems.thymos")


# ─── Constants ──────────────────────────────────────────────────

# Synapse events that Thymos converts into Incidents
_SUBSCRIBED_EVENTS: frozenset[SynapseEventType] = frozenset({
    SynapseEventType.SYSTEM_FAILED,
    SynapseEventType.SYSTEM_RECOVERED,
    SynapseEventType.SYSTEM_RESTARTING,
    SynapseEventType.SAFE_MODE_ENTERED,
    SynapseEventType.SAFE_MODE_EXITED,
    SynapseEventType.SYSTEM_OVERLOADED,
    SynapseEventType.CLOCK_OVERRUN,
    SynapseEventType.RESOURCE_PRESSURE,
    # Economic immune system (Phase 16f)
    SynapseEventType.TRANSACTION_SHIELDED,
    # THREAT_DETECTED has a dedicated handler (_on_threat_detected) that routes
    # through ThreatPatternSentinel - it is NOT in _SUBSCRIBED_EVENTS to avoid
    # double-firing via the generic _on_synapse_event path.
    SynapseEventType.PROTOCOL_ALERT,
    SynapseEventType.EMERGENCY_WITHDRAWAL,
    SynapseEventType.THREAT_ADVISORY_RECEIVED,
    SynapseEventType.ADDRESS_BLACKLISTED,
    # Certificate lifecycle (Phase 16g: Civilization Layer)
    SynapseEventType.CERTIFICATE_EXPIRING,
    SynapseEventType.CERTIFICATE_EXPIRED,
    # Connector health (Phase 16h) - import errors, external service failures
    SynapseEventType.SYSTEM_DEGRADED,
    SynapseEventType.CONNECTOR_ERROR,
    SynapseEventType.CONNECTOR_TOKEN_EXPIRED,
    # Model lifecycle - catastrophic failures during hot-swap
    SynapseEventType.MODEL_HOT_SWAP_FAILED,
    SynapseEventType.CATASTROPHIC_FORGETTING_DETECTED,
    SynapseEventType.MODEL_ROLLBACK_TRIGGERED,
    # Cross-system health bridges (Phase: full-organism immune coverage)
    SynapseEventType.NOVA_DEGRADED,
    SynapseEventType.EVO_CONSOLIDATION_STALLED,
    SynapseEventType.SKIA_HEARTBEAT_LOST,
    # SG7 - Kairos causal invariants feed CausalAnalyzer graph cache
    SynapseEventType.KAIROS_INVARIANT_DISTILLED,
})

# How often to run homeostatic checks (in seconds)
_HOMEOSTASIS_INTERVAL_S: float = 30.0

# How often to run sentinel scans (in seconds)
_SENTINEL_SCAN_INTERVAL_S: float = 30.0

# How long to wait for post-repair verification (seconds)
_POST_REPAIR_VERIFY_TIMEOUT_S: float = 10.0

# SG4: Max seconds to wait for a federated peer to resolve an escalated incident
_FEDERATION_ESCALATION_TIMEOUT_S: float = 45.0

# Salience mapping: incident severity → percept priority for Atune
_SEVERITY_TO_SALIENCE: dict[IncidentSeverity, float] = {
    IncidentSeverity.CRITICAL: 1.0,
    IncidentSeverity.HIGH: 0.8,
    IncidentSeverity.MEDIUM: 0.5,
    IncidentSeverity.LOW: 0.2,
    IncidentSeverity.INFO: 0.1,
}

# Maximum incidents to buffer in memory
_INCIDENT_BUFFER_SIZE: int = 10_000


# ─── Drive State ────────────────────────────────────────────────


@dataclass
class DriveState:
    """
    Accumulated drive pressure observed by the immune system.

    Each drive accumulates stress from incidents (constitutional_impact) and
    from Equor rejections (which drive was violated). The pressure decays
    exponentially over time so old signals don't permanently bias behaviour.

    This is Thymos's read-only view of constitutional health - it cannot
    modify Equor's authoritative drive weights, but it can use this
    internal pressure signal to prioritise which repairs to attempt and
    how aggressively to escalate.
    """

    # Accumulated stress per drive (0.0 = no pressure, 1.0 = fully stressed)
    coherence: float = 0.0
    care: float = 0.0
    growth: float = 0.0
    honesty: float = 0.0

    # Total rejections received from Equor (for observability)
    equor_rejections: int = 0

    # Per-drive rejection counts (which drive caused the most blocks)
    rejections_by_drive: dict[str, int] = field(default_factory=lambda: {
        "coherence": 0, "care": 0, "growth": 0, "honesty": 0,
    })

    # Last update timestamps (for decay computation)
    last_updated: datetime = field(default_factory=utc_now)

    # Decay half-life in seconds: pressure halves every 5 minutes
    _DECAY_HALF_LIFE_S: float = 300.0

    def apply_decay(self) -> None:
        """Exponential decay toward zero. Call before reading or writing pressure."""
        now = utc_now()
        elapsed = (now - self.last_updated).total_seconds()
        if elapsed <= 0:
            return
        # decay_factor approaches 0 as time → ∞
        decay_factor = math.exp(-elapsed * math.log(2) / self._DECAY_HALF_LIFE_S)
        self.coherence *= decay_factor
        self.care *= decay_factor
        self.growth *= decay_factor
        self.honesty *= decay_factor
        self.last_updated = now

    def absorb_incident(self, incident: Incident) -> None:
        """Add incident constitutional_impact to accumulated drive pressure."""
        self.apply_decay()
        impact = incident.constitutional_impact
        # Pressure accumulates additively but is capped at 1.0
        self.coherence = min(1.0, self.coherence + impact.get("coherence", 0.0) * 0.1)
        self.care = min(1.0, self.care + impact.get("care", 0.0) * 0.1)
        self.growth = min(1.0, self.growth + impact.get("growth", 0.0) * 0.1)
        self.honesty = min(1.0, self.honesty + impact.get("honesty", 0.0) * 0.1)

    def absorb_equor_rejection(
        self,
        alignment: dict[str, float],
        reasoning: str,
    ) -> None:
        """
        Record an Equor rejection and update drive pressure.

        When Equor blocks a repair, the drive that was most violated gets an
        additional pressure spike. This signals to Thymos which constitutional
        dimension needs the most careful attention in future repairs.

        Args:
            alignment: Per-drive alignment scores from ConstitutionalCheck
                       (negative = violated, positive = aligned).
            reasoning: Human-readable rejection reason from Equor.
        """
        self.apply_decay()
        self.equor_rejections += 1

        # Identify the most-violated drive (most negative alignment score)
        most_violated = min(alignment.items(), key=lambda kv: kv[1], default=None)
        if most_violated is not None:
            drive_name, score = most_violated
            if score < 0 and drive_name in self.rejections_by_drive:
                self.rejections_by_drive[drive_name] += 1

        # Add pressure proportional to how badly each drive was violated
        # Positive alignment scores do not reduce pressure - only time decays it
        for drive in ("coherence", "care", "growth", "honesty"):
            score = alignment.get(drive, 0.0)
            if score < 0:
                spike = min(0.3, abs(score) * 0.4)
                current = getattr(self, drive)
                setattr(self, drive, min(1.0, current + spike))

    @property
    def most_stressed_drive(self) -> str:
        """Return the drive name with the highest current pressure."""
        return max(
            [("coherence", self.coherence), ("care", self.care),
             ("growth", self.growth), ("honesty", self.honesty)],
            key=lambda kv: kv[1],
        )[0]

    @property
    def composite_stress(self) -> float:
        """Overall drive stress level, 0.0–1.0."""
        return (self.coherence + self.care + self.growth + self.honesty) / 4.0

    def as_dict(self) -> dict[str, Any]:
        self.apply_decay()
        return {
            "coherence": round(self.coherence, 3),
            "care": round(self.care, 3),
            "growth": round(self.growth, 3),
            "honesty": round(self.honesty, 3),
            "composite_stress": round(self.composite_stress, 3),
            "most_stressed_drive": self.most_stressed_drive,
            "equor_rejections": self.equor_rejections,
            "rejections_by_drive": dict(self.rejections_by_drive),
        }


def _api_diagnostic_hint(ctx: ApiErrorContext, exc: Exception) -> str | None:
    """
    Generate a repair hint from API context + exception details.

    Returns a short hypothesis string for Incident.root_cause_hypothesis,
    or None when no specific hint applies.
    """
    sc = ctx.status_code
    msg = str(exc).lower()

    if sc >= 500 and "module" in msg and "not found" in msg:
        return (
            f"ModuleNotFoundError on {ctx.method}:{ctx.endpoint} → "
            "Tier 2 restart of the owning system recommended."
        )
    if sc == 404 and ctx.endpoint:
        return (
            f"404 on known endpoint {ctx.endpoint} → "
            "Possible router registration issue; check route declarations (Tier 1)."
        )
    if ctx.latency_ms > 10_000:
        return (
            f"Request to {ctx.endpoint} took {ctx.latency_ms:.0f} ms → "
            "Timeout tuning recommended (Tier 1 PARAMETER)."
        )
    return None


class ThymosService:
    """
    Thymos - the EOS immune system.

    Coordinates seven sub-systems:
      Sentinels              - fault detection (5 sentinel classes)
      Triage                 - deduplication, severity scoring, response routing
      Diagnosis              - causal analysis, temporal correlation, hypothesis engine
      Prescription           - repair tier selection and validation gate
      AntibodyLibrary        - immune memory: crystallized successful repairs
      Prophylactic           - prevention: pre-deploy scans and homeostasis
      HealingGovernor        - cytokine storm prevention and budget enforcement
    """

    system_id: str = "thymos"

    def __init__(
        self,
        config: ThymosConfig,
        synapse: SynapseService | None = None,
        neo4j: Neo4jClient | None = None,
        llm: LLMProvider | None = None,
        metrics: MetricCollector | None = None,
        neuroplasticity_bus: NeuroplasticityBus | None = None,
        redis: RedisClient | None = None,
    ) -> None:
        self._config = config
        self._synapse = synapse
        self._neo4j = neo4j
        self._llm = llm
        self._metrics = metrics
        self._neuroplasticity_bus: NeuroplasticityBus | None = neuroplasticity_bus
        self._initialized: bool = False
        self._logger = logger.bind(system="thymos")
        self._redis = redis  # Stored for crash-context re-hydration on boot (GAP-6)
        self._notification_dispatcher = NotificationDispatcher(redis=redis)

        # Cross-system references (wired post-init by main.py)
        self._equor: Any = None     # EquorService - constitutional review
        self._evo: Any = None       # EvoService - error pattern learning
        self._atune: Any = None     # AtuneService - incident-as-percept
        self._health_monitor: HealthMonitor | None = None  # Synapse health records
        self._soma: Any = None      # SomaService -- integrity precision gating
        self._oikos: Any = None     # OikosService -- economic state for protocol health checks
        self._federation: Any = None  # FederationService -- threat advisory broadcast
        self._telos: Any = None     # TelosService - authoritative drive state + impact prediction
        self._simula: Any = None    # SimulaService - novel repair via EvolutionProposal pipeline
        self._clo: Any = None       # ContinualLearningOrchestrator - for autonomous training-halt recovery
        self._embedding_client: EmbeddingClient | None = None  # Shared 768-dim embedder (P2)

        # ── Sub-systems (built in initialize()) ──
        # Sentinels
        self._exception_sentinel: ExceptionSentinel | None = None
        self._contract_sentinel: ContractSentinel | None = None
        self._feedback_loop_sentinel: FeedbackLoopSentinel | None = None
        self._drift_sentinel: DriftSentinel | None = None
        self._cognitive_stall_sentinel: CognitiveStallSentinel | None = None
        self._bankruptcy_sentinel: BankruptcySentinel | None = None
        self._threat_pattern_sentinel: ThreatPatternSentinel | None = None
        self._protocol_health_sentinel: ProtocolHealthSentinel | None = None

        # Triage
        self._deduplicator: IncidentDeduplicator | None = None
        self._severity_scorer: SeverityScorer | None = None
        self._response_router: ResponseRouter | None = None

        # Diagnosis
        self._causal_analyzer: CausalAnalyzer | None = None
        self._temporal_correlator: TemporalCorrelator | None = None
        self._diagnostic_engine: DiagnosticEngine | None = None

        # Prescription
        self._prescriber: RepairPrescriber | None = None
        self._validator: RepairValidator | None = None

        # Immune memory
        self._antibody_library: AntibodyLibrary | None = None

        # Prophylactic
        self._prophylactic_scanner: ProphylacticScanner | None = None
        self._homeostasis_controller: HomeostasisController | None = None

        # Governor
        self._governor: HealingGovernor | None = None

        # Pattern-aware routing (built in initialize())
        self._crash_pattern_analyzer: CrashPatternAnalyzer | None = None
        self._pattern_router: PatternAwareRouter | None = None

        # Cross-system references (wired post-init by main.py)
        self._nova: Any = None  # NovaService - for injecting urgent repair goals

        # ── In-flight Simula proposals (cytokine storm prevention) ──
        # Maps incident fingerprint → (proposal_id, timestamp_submitted)
        # Prevents duplicate proposals for the same structural bug.
        # TTL: 10 minutes - if no resolution, remove to allow resubmission.
        self._active_simula_proposals: dict[str, tuple[str, float]] = {}

        # ── State ──
        self._active_incidents: dict[str, Incident] = {}
        self._incident_buffer: deque[Incident] = deque(maxlen=_INCIDENT_BUFFER_SIZE)
        # Circuit-breaker incident tracking: action_type → incident_id
        # Allows auto-resolution when a breaker transitions back to CLOSED.
        self._cb_incidents: dict[str, str] = {}
        self._resolution_times: deque[float] = deque(maxlen=500)

        # Drive state: accumulated pressure from incidents and Equor rejections.
        # Thymos's internal read-only view of constitutional health.
        self._drive_state: DriveState = DriveState()

        # Background tasks
        self._sentinel_task: asyncio.Task[None] | None = None
        self._homeostasis_task: asyncio.Task[None] | None = None
        self._telegram_polling_task: asyncio.Task[None] | None = None

        # Counters
        self._total_incidents: int = 0
        self._total_repairs_attempted: int = 0
        self._total_repairs_succeeded: int = 0
        self._total_repairs_failed: int = 0
        self._total_repairs_rolled_back: int = 0
        self._total_diagnoses: int = 0
        self._total_antibodies_applied: int = 0
        self._total_antibodies_created: int = 0
        self._total_homeostatic_adjustments: int = 0
        self._total_prophylactic_scans: int = 0
        self._total_prophylactic_warnings: int = 0
        # SSE streaming queues - one per connected client
        self._stream_queues: list[asyncio.Queue[Any]] = []
        self._incidents_by_severity: dict[str, int] = {}
        self._incidents_by_class: dict[str, int] = {}
        self._repairs_by_tier: dict[str, int] = {}
        self._diagnosis_confidences: deque[float] = deque(maxlen=200)
        self._diagnosis_latencies: deque[float] = deque(maxlen=200)

        # ── Metabolic gating ──────────────────────────────────────────────
        self._starvation_level: str = "nominal"

        # ── Cached cross-system state (populated via Synapse subscriptions) ──
        # Replaces direct self._soma / self._oikos / self._telos reads.
        self._cached_soma_signal: dict[str, Any] = {}
        self._cached_soma_coherence: float = 1.0
        self._cached_soma_vulnerability: dict[str, Any] = {}
        self._cached_oikos_snapshot: dict[str, Any] = {}
        self._cached_telos_drive_impact: dict[str, dict[str, float]] = {}

        # ── Synapse request/reply tracking ───────────────────────────────
        self._pending_requests: dict[str, asyncio.Future[dict[str, Any]]] = {}

    # ─── Cross-System Wiring ─────────────────────────────────────────

    def set_equor(self, equor: Any) -> None:
        """Wire Equor for constitutional review of Tier 3+ repairs."""
        self._equor = equor
        if self._validator is not None:
            self._validator._equor = equor
        self._logger.info("equor_wired_to_thymos")

    def set_evo(self, evo: Any) -> None:
        """Wire Evo so repair outcomes feed the learning system."""
        self._evo = evo
        self._logger.info("evo_wired_to_thymos")

    def set_atune(self, atune: Any) -> None:
        """Wire Atune so high-severity incidents become Percepts."""
        self._atune = atune
        self._logger.info("atune_wired_to_thymos")

    def set_health_monitor(self, health_monitor: HealthMonitor) -> None:
        """Wire Synapse HealthMonitor for health record queries."""
        self._health_monitor = health_monitor
        if self._causal_analyzer is not None:
            self._causal_analyzer._health = health_monitor
        self._logger.info("health_monitor_wired_to_thymos")

    def set_embedding_client(self, client: EmbeddingClient) -> None:
        """
        Wire the shared 768-dim sentence-transformer embedding client (P2).

        If called before initialize(), the client is stored and passed to
        ProphylacticScanner during initialization.  If called after, the
        scanner's client is hot-swapped immediately.
        """
        self._embedding_client = client
        if self._prophylactic_scanner is not None:
            self._prophylactic_scanner.set_embedding_client(client)
        self._logger.info("embedding_client_wired_to_thymos")

    def set_nova(self, nova: Any) -> None:
        """Wire Nova so critical incidents generate urgent repair goals."""
        self._nova = nova
        self._logger.info("nova_wired_to_thymos")

    def set_soma(self, soma: Any) -> None:
        """Wire Soma for integrity-based constitutional health gating."""
        self._soma = soma
        self._logger.info("soma_wired_to_thymos")

    def set_clo(self, clo: Any) -> None:
        """Wire ContinualLearningOrchestrator for autonomous training-halt recovery.

        When Thymos detects a `re_success_rate_below_floor` halt, it schedules a
        periodic check: if the 7-day Thompson score has recovered above 0.55, it
        calls `clo.clear_training_halt()` so the organism can self-heal without
        operator intervention.
        """
        self._clo = clo
        self._logger.info("clo_wired_to_thymos")

    def set_oikos(self, oikos: Any) -> None:
        """Wire Oikos so protocol health sentinel can read yield positions."""
        self._oikos = oikos
        self._logger.info("oikos_wired_to_thymos")

    def set_federation(self, federation: Any) -> None:
        """Wire Federation for threat advisory broadcast and receipt."""
        self._federation = federation
        self._logger.info("federation_wired_to_thymos")

    def set_telos(self, telos: Any) -> None:
        """
        Wire Telos as the authoritative source for drive state and impact prediction.

        After wiring, Thymos reads Telos's last EffectiveIntelligenceReport to
        populate ``Incident.constitutional_impact`` before severity scoring.  This
        ensures drive-impact assessment is grounded in Telos's measurements rather
        than computed independently.
        """
        self._telos = telos
        self._logger.info("telos_wired_to_thymos")

    def set_simula(self, simula: Any) -> None:
        """Wire Simula so Tier 4 repairs become real EvolutionProposals."""
        self._simula = simula
        self._logger.info("simula_wired_to_thymos")

    def set_redis(self, redis: Any) -> None:
        """Wire Redis client (hot-swap after init). Propagates to CrashPatternAnalyzer."""
        self._redis = redis
        self._notification_dispatcher._redis = redis
        if self._crash_pattern_analyzer is not None:
            self._crash_pattern_analyzer.set_redis(redis)
        self._logger.info("redis_wired_to_thymos")

    # ─── Drive State (Equor Rejection Feedback) ──────────────────────

    @property
    def drive_state(self) -> DriveState:
        """
        Current accumulated drive pressure as seen by the immune system.

        Read-only from Thymos's perspective - the authoritative drive weights
        live in Equor.  This is Thymos's internal signal for biasing repairs
        toward drives that are under the most constitutional stress.
        """
        return self._drive_state

    def receive_divergence_pressure(self, pressure: Any) -> None:
        """
        Accept a divergence pressure signal from Nexus (via ThymosNexusSinkAdapter).

        High pressure_magnitude means the instance is too similar to federation peers
        - nudges the growth drive proportionally so Thymos biases toward exploratory
        repairs rather than conservative ones.

        Args:
            pressure: DivergencePressure object with ``pressure_magnitude`` float field.
        """
        magnitude: float = getattr(pressure, "pressure_magnitude", 0.0)
        nudge = magnitude * 0.15
        self._drive_state.growth = min(1.0, self._drive_state.growth + nudge)
        self._logger.debug(
            "divergence_pressure_absorbed",
            magnitude=round(magnitude, 3),
            growth_nudge=round(nudge, 3),
            new_growth=round(self._drive_state.growth, 3),
        )

    def on_equor_rejection(
        self,
        alignment: dict[str, float],
        reasoning: str,
        incident: Incident | None = None,
    ) -> None:
        """
        Signal that Equor rejected a Thymos-originated intent.

        Called from process_incident() when the validation gate receives a
        BLOCKED or DEFERRED verdict.  Updates the drive pressure accumulator
        so future prescriptions can account for which drives are most at risk.

        Thymos CANNOT modify Equor's weights - this is a read signal only.
        It biases internal repair selection (e.g. prefer lower-tier, more
        conservative repairs when honesty pressure is high) without ever
        touching the constitutional drives themselves.

        Args:
            alignment: Per-drive alignment scores from the ConstitutionalCheck
                       (negative = drive violated by the proposed repair).
            reasoning: Human-readable rejection reason from Equor.
            incident:  The incident that triggered the rejected repair (optional,
                       used for richer logging).
        """
        self._drive_state.absorb_equor_rejection(alignment, reasoning)

        stressed = self._drive_state.most_stressed_drive
        stress_level = self._drive_state.composite_stress

        self._logger.info(
            "equor_rejection_absorbed",
            incident_id=incident.id if incident else "n/a",
            alignment=alignment,
            reasoning=reasoning[:120],
            most_stressed_drive=stressed,
            composite_stress=round(stress_level, 3),
            total_rejections=self._drive_state.equor_rejections,
        )

        self._emit_metric("thymos.equor.rejections", 1)
        self._emit_metric(
            "thymos.drive.stress",
            stress_level,
            tags={"most_stressed": stressed},
        )

    async def _on_intent_rejected(self, event: SynapseEvent) -> None:
        """
        Handle an INTENT_REJECTED event from the Synapse bus.

        Routes any Equor-rejected intent (not just Thymos-originated repairs)
        into on_equor_rejection() so drive priorities are updated system-wide.
        """
        data = event.data
        alignment: dict[str, float] = data.get("alignment", {})
        reasoning: str = data.get("reasoning", "")
        self.on_equor_rejection(alignment=alignment, reasoning=reasoning, incident=None)

    async def _on_interoceptive_percept(self, event: SynapseEvent) -> None:
        """
        Handle an INTEROCEPTIVE_PERCEPT event from Soma.

        Soma broadcasts these when Fisher geodesic deviation exceeds threshold,
        causal emergence is declining, or RG flow shows anomalies. Thymos uses
        the percept to proactively triage and, if the action is TRIGGER_REPAIR,
        open an incident before an actual failure surfaces.
        """
        from primitives.soma import InteroceptiveAction

        data = event.data
        urgency: float = data.get("urgency", 0.0)
        action_raw: str = data.get("recommended_action", InteroceptiveAction.NONE)
        epicenter: str = data.get("epicenter_system", "unknown")
        description: str = data.get("description", "")

        try:
            action = InteroceptiveAction(action_raw)
        except ValueError:
            action = InteroceptiveAction.NONE

        self._logger.debug(
            "interoceptive_percept_received",
            urgency=round(urgency, 3),
            action=action.value,
            epicenter=epicenter,
        )

        if action == InteroceptiveAction.TRIGGER_REPAIR:
            # Soma is requesting a proactive repair - open a prophylactic incident
            _msg = f"Soma interoceptive alert: {description[:120]}"
            incident = Incident(
                source_system=epicenter,
                incident_class=IncidentClass.DEGRADATION,
                severity=IncidentSeverity.HIGH if urgency > 0.6 else IncidentSeverity.MEDIUM,
                error_type="InteroceptiveTrigger",
                error_message=_msg,
                fingerprint=hashlib.sha256(
                    f"InteroceptiveTrigger:{epicenter}".encode()
                ).hexdigest()[:16],
                context={"urgency": urgency, "soma_action": action.value},
            )
            await self.on_incident(incident)

        elif action == InteroceptiveAction.EMERGENCY_SAFE_MODE:
            # Soma detected catastrophic instability - request safe mode
            self._logger.warning(
                "soma_emergency_safe_mode_requested",
                urgency=round(urgency, 3),
                epicenter=epicenter,
            )
            _msg = f"Soma emergency: {description[:120]}"
            incident = Incident(
                source_system=epicenter,
                incident_class=IncidentClass.CRASH,
                severity=IncidentSeverity.CRITICAL,
                error_type="SomaEmergency",
                error_message=_msg,
                fingerprint=hashlib.sha256(
                    f"SomaEmergency:{epicenter}".encode()
                ).hexdigest()[:16],
                context={"urgency": urgency, "soma_action": action.value},
            )
            await self.on_incident(incident)

        elif action == InteroceptiveAction.INHIBIT_GROWTH:
            # Suppress Growth drive directly (Soma detects dynamical fragility)
            if hasattr(self, "_drive_state"):
                self._drive_state.growth = max(0.0, self._drive_state.growth * 0.6)

    async def _on_growth_stagnation(self, event: SynapseEvent) -> None:
        """
        Handle a GROWTH_STAGNATION event emitted by Telos.

        Telos detects that dI/dt has flatlined or gone negative while frontier
        domains remain unexplored.  Thymos opens a DRIFT incident so the
        repair pipeline can nudge Evo toward exploration or parameter refresh.
        """
        data = event.data
        urgency: str = data.get("urgency", "low")
        di_dt: float = data.get("dI_dt", 0.0)
        frontier_domains: list[str] = data.get("frontier_domains", [])
        directive: str = data.get("directive", "")

        severity_map = {
            "critical": IncidentSeverity.CRITICAL,
            "high": IncidentSeverity.HIGH,
            "medium": IncidentSeverity.MEDIUM,
            "low": IncidentSeverity.LOW,
        }
        severity = severity_map.get(urgency, IncidentSeverity.MEDIUM)

        summary = f"Growth stagnation: dI/dt={di_dt:+.4f}"
        if frontier_domains:
            summary += f", frontiers={','.join(frontier_domains[:3])}"

        self._logger.info(
            "growth_stagnation_incident",
            urgency=urgency,
            di_dt=round(di_dt, 4),
            frontier_count=len(frontier_domains),
        )

        _msg = summary[:200]
        incident = Incident(
            source_system="telos",
            incident_class=IncidentClass.DRIFT,
            severity=severity,
            error_type="GrowthStagnation",
            error_message=_msg,
            fingerprint=hashlib.sha256(b"GrowthStagnation:telos").hexdigest()[:16],
            context={
                "urgency": urgency,
                "dI_dt": di_dt,
                "frontier_domains": frontier_domains[:5],
                "directive": directive,
            },
        )
        await self.on_incident(incident)

    async def _on_autonomy_insufficient(self, event: SynapseEvent) -> None:
        """
        Handle AUTONOMY_INSUFFICIENT emitted by Nova.

        Forwards a human-readable approval request to Tate via Telegram.
        This is the organism's formal governed channel for asking Tate to
        elevate autonomy - not a raw dump of system state.
        """
        data = event.data
        goal: str = data.get("goal_description", "unknown goal")
        reason: str = data.get("reason", data.get("executor", "autonomy level too low"))
        balance_usd: float = data.get("balance_usd", 0.0)

        text = (
            "🔒 EOS needs permission\n"
            f"Goal: {goal}\n"
            f"Blocked: {reason}\n"
            f"Balance: ${balance_usd:.2f}\n"
            "\n"
            "Reply /approve\\_autonomy to grant level 3 for 120 min."
        )

        self._logger.warning(
            "autonomy_insufficient_forwarding_to_tate",
            goal=goal[:80],
            reason=reason[:80],
            balance_usd=balance_usd,
        )

        await self._notification_dispatcher.dispatch_raw(
            text=text,
            metadata={"event": data, "event_type": event.event_type.value},
            dedup_key=f"autonomy_insufficient:{event.source_system}",
            severity="warning",
        )

    async def _on_metabolic_pressure(self, event: SynapseEvent) -> None:
        """
        Handle METABOLIC_PRESSURE events from Oikos.

        Updates starvation level for gating, and forwards economic stress
        notifications to Tate when runway < 30 days.
        """
        data = event.data

        # ── Metabolic gating: update starvation level ──
        level = data.get("starvation_level", "")
        if level:
            old = self._starvation_level
            self._starvation_level = level
            if level != old:
                self._logger.info("thymos_starvation_level_changed", old=old, new=level)
                await self._adjust_for_starvation(level)

        if not data.get("economic_stress") and data.get("source") != "oikos_starvation_broadcast":
            return

        runway_days: str = data.get("runway_days", "?")
        daily_cost: str = data.get("daily_cost_usd", "?")
        daily_yield: str = data.get("daily_yield_usd", "?")

        try:
            daily_cost_f = float(daily_cost)
            daily_yield_f = float(daily_yield)
            cost_str = f"${daily_cost_f:.2f}"
            yield_str = f"${daily_yield_f:.2f}"
        except (ValueError, TypeError):
            cost_str = daily_cost
            yield_str = daily_yield

        try:
            runway_f = float(runway_days)
            runway_str = f"{runway_f:.1f}"
        except (ValueError, TypeError):
            runway_str = runway_days

        text = (
            f"⚠️ Economic stress\n"
            f"Runway: {runway_str} days\n"
            f"Burn: {cost_str}/day\n"
            f"Yield: {yield_str}/day"
        )

        self._logger.warning(
            "economic_stress_forwarding_to_tate",
            runway_days=runway_str,
            daily_cost=cost_str,
            daily_yield=yield_str,
        )

        await self._notification_dispatcher.dispatch_raw(
            text=text,
            metadata={"event": data, "event_type": event.event_type.value},
            dedup_key=f"economic_stress:{event.source_system}",
            severity="warning",
        )

    async def _adjust_for_starvation(self, level: str) -> None:
        """Thymos-specific metabolic degradation.

        AUSTERITY: skip prophylactic scanning, only reactive healing
        EMERGENCY: triage - only heal severity=critical incidents
        CRITICAL: minimal - log incidents but don't attempt LLM-based diagnosis
        """
        if level == "austerity":
            # Cancel prophylactic scanning to save resources
            if self._homeostasis_task is not None and not self._homeostasis_task.done():
                self._homeostasis_task.cancel()
                self._logger.warning("thymos_prophylactic_cancelled_austerity")
        elif level in ("emergency", "critical"):
            # Cancel prophylactic + sentinel scanning
            for task in (self._homeostasis_task, self._sentinel_task):
                if task is not None and not task.done():
                    task.cancel()
            self._logger.warning("thymos_background_tasks_cancelled_starvation", level=level)

    async def _on_task_permanently_failed(self, event: SynapseEvent) -> None:
        """Handle TASK_PERMANENTLY_FAILED - forward to Tate via Telegram."""
        data = event.data
        task_name: str = data.get("task_name", "unknown")
        error: str = data.get("final_error", "unknown error")
        attempts: int = data.get("restart_attempts", 0)

        text = (
            f"💀 Background task died\n"
            f"Task: {task_name}\n"
            f"Error: {error[:200]}\n"
            f"Restarted: {attempts} times, giving up"
        )

        self._logger.error(
            "task_permanently_failed_forwarding_to_tate",
            task_name=task_name,
            attempts=attempts,
        )

        # Critical - never deduplicated; task death is always actionable.
        await self._notification_dispatcher.dispatch_raw(
            text=text,
            metadata={"event": data, "event_type": event.event_type.value},
            dedup_key="",
            severity="critical",
        )

    async def _on_bounty_pr_submitted(self, event: SynapseEvent) -> None:
        """Handle BOUNTY_PR_SUBMITTED - notify Tate of a PR submission."""
        data = event.data
        bounty_url: str = data.get("bounty_url", "")
        pr_url: str = data.get("pr_url", "")
        amount: float = data.get("estimated_reward_usd", data.get("amount", 0.0))

        text = (
            f"💰 PR submitted\n"
            f"Bounty: {bounty_url}\n"
            f"PR: {pr_url}\n"
            f"Estimated reward: ${amount:.2f}"
        )

        self._logger.info(
            "bounty_pr_submitted_forwarding_to_tate",
            bounty_url=bounty_url[:80],
            pr_url=pr_url[:80],
            amount=amount,
        )

        await self._notification_dispatcher.dispatch_raw(
            text=text,
            metadata={"event": data, "event_type": event.event_type.value},
            dedup_key=f"bounty_pr:{pr_url}",
            severity="info",
        )

    async def _on_simula_calibration_degraded(self, event: SynapseEvent) -> None:
        """Handle SIMULA_CALIBRATION_DEGRADED - notify Tate of accuracy regression."""
        data = event.data
        score: float = data.get("calibration_score", 0.0)
        score_pct = int(score * 100) if score <= 1.0 else int(score)

        text = (
            f"🔧 Simula self-healing degraded\n"
            f"Calibration: {score_pct}%\n"
            f"Last 20 proposals accuracy dropping"
        )

        self._logger.warning(
            "simula_calibration_degraded_forwarding_to_tate",
            calibration_score=score,
        )

        await self._notification_dispatcher.dispatch_raw(
            text=text,
            metadata={"event": data, "event_type": event.event_type.value},
            dedup_key=f"simula_calibration:{event.source_system}",
            severity="warning",
        )

    # ─── Sentinel Event Handlers (Cross-System Wiring) ────────────────

    async def _on_repair_completed_for_feedback(self, event: SynapseEvent) -> None:
        """
        Feed REPAIR_COMPLETED events into the FeedbackLoopSentinel.

        If the same fingerprint re-breaks within a monitoring window after a
        repair was completed, the FeedbackLoopSentinel detects the repair→re-break
        cycle and fires a LOOP_SEVERANCE incident.

        Also used by the antibody library to close the Simula repair feedback loop.
        """
        data = event.data
        incident_id: str = data.get("incident_id", "")
        success: bool = data.get("success", True)
        fingerprint: str = data.get("fingerprint", "")
        proposal_id: str = data.get("proposal_id", "")

        self._logger.info(
            "repair_completed_received",
            incident_id=incident_id,
            proposal_id=proposal_id,
            success=success,
            source=event.source_system,
        )

        # Track in temporal correlator for causal analysis
        if self._temporal_correlator is not None:
            tier = data.get("tier", "unknown")
            self._temporal_correlator.record_event(
                event_type="repair_completed",
                details=f"Repair tier={tier} success={success} for {incident_id}",
                system_id=event.source_system,
            )

        # If from Simula, match to tracked proposal and update antibody
        if proposal_id and incident_id:
            # Remove from active Simula proposals
            for fp, (pid, _ts) in list(self._active_simula_proposals.items()):
                if pid == proposal_id:
                    del self._active_simula_proposals[fp]
                    break

            # Resolve the matching active incident
            incident = self._active_incidents.get(incident_id)
            if incident is not None and success:
                incident.repair_status = RepairStatus.RESOLVED
                incident.repair_successful = True
                now = utc_now()
                incident.resolution_time_ms = int(
                    (now - incident.timestamp).total_seconds() * 1000
                )
                self._resolution_times.append(float(incident.resolution_time_ms))
                self._active_incidents.pop(incident_id, None)
                if self._governor is not None:
                    self._governor.resolve_incident(incident_id)

                # Update antibody effectiveness if one exists
                if (
                    incident.antibody_id is not None
                    and self._antibody_library is not None
                ):
                    await self._antibody_library.record_outcome(
                        incident.antibody_id, success=True,
                    )
            elif incident is not None and not success:
                # Repair failed - mark for re-diagnosis
                incident.repair_status = RepairStatus.ESCALATED
                if (
                    incident.antibody_id is not None
                    and self._antibody_library is not None
                ):
                    await self._antibody_library.record_outcome(
                        incident.antibody_id, success=False,
                    )

        # Prophylactic scan: check changed files against antibody patterns
        files_changed: list[str] = data.get("files_changed", [])
        if files_changed and self._prophylactic_scanner is not None:
            warnings = await self._prophylactic_scanner.scan(files_changed)
            self._total_prophylactic_scans += 1
            self._total_prophylactic_warnings += len(warnings)
            if warnings:
                self._logger.warning(
                    "prophylactic_post_repair_warnings",
                    files=len(files_changed),
                    warnings=len(warnings),
                    details=[w.warning[:80] for w in warnings[:3]],
                )
                await self._emit_evolutionary_observable(
                    "prophylactic_triggered", float(len(warnings)),
                    is_novel=True,
                    metadata={
                        "files_scanned": len(files_changed),
                        "warning_count": len(warnings),
                    },
                )

    async def _on_evo_consolidation_stalled(self, event: SynapseEvent) -> None:
        """
        Handle EVO_CONSOLIDATION_STALLED - Evo's learning pipeline has stalled.

        Creates a COGNITIVE_STALL incident so the repair pipeline can restart
        Evo's consolidation or adjust parameters.
        """
        data = event.data
        cycles_since: int = data.get("cycles_since_consolidation", 0)
        reason: str = data.get("reason", "unknown")

        self._logger.warning(
            "evo_consolidation_stall_detected",
            cycles_since=cycles_since,
            reason=reason[:120],
        )

        incident = Incident(
            source_system="evo",
            incident_class=IncidentClass.COGNITIVE_STALL,
            severity=IncidentSeverity.HIGH if cycles_since > 100 else IncidentSeverity.MEDIUM,
            error_type="EvoConsolidationStalled",
            error_message=(
                f"Evo consolidation stalled for {cycles_since} cycles: {reason[:120]}"
            ),
            fingerprint=hashlib.sha256(
                b"EvoConsolidationStalled:evo"
            ).hexdigest()[:16],
            context={
                "cycles_since_consolidation": cycles_since,
                "reason": reason,
            },
            constitutional_impact={
                "coherence": 0.2,
                "care": 0.0,
                "growth": 0.8,
                "honesty": 0.1,
            },
        )
        await self.on_incident(incident)

    async def _on_evo_degraded(self, event: SynapseEvent) -> None:
        """
        EVO-ECON-1: Handle EVO_DEGRADED - Evo's hypothesis generation budget is
        exhausted (LLM skips).

        Creates a DEGRADATION incident so the repair pipeline can adjust Evo's
        budget or trigger a recovery cycle.
        """
        data = event.data
        reason: str = data.get("reason", "unknown")
        skipped: int = data.get("skipped_pattern_count", 0)
        consecutive: int = data.get("consecutive_skips", 0)

        self._logger.warning(
            "evo_degraded_detected",
            reason=reason[:120],
            skipped_pattern_count=skipped,
            consecutive_skips=consecutive,
        )

        incident = Incident(
            source_system="evo",
            incident_class=IncidentClass.DEGRADATION,
            severity=IncidentSeverity.HIGH if consecutive > 5 else IncidentSeverity.MEDIUM,
            error_type="EvoDegraded",
            error_message=(
                f"Evo hypothesis generation degraded: {reason[:120]} "
                f"(skipped={skipped}, consecutive={consecutive})"
            ),
            fingerprint=hashlib.sha256(
                b"EvoDegraded:evo"
            ).hexdigest()[:16],
            context={
                "reason": reason,
                "skipped_pattern_count": skipped,
                "consecutive_skips": consecutive,
            },
            constitutional_impact={
                "coherence": 0.1,
                "care": 0.0,
                "growth": 0.9,
                "honesty": 0.0,
            },
        )
        await self.on_incident(incident)

    async def _on_nova_degraded(self, event: SynapseEvent) -> None:
        """
        Handle NOVA_DEGRADED - Nova's inference quality has dropped.

        Creates a DEGRADATION incident so the repair pipeline can investigate.
        """
        data = event.data
        decisions_affected: int = data.get("decisions_affected_since_degradation", 0)
        reason: str = data.get("reason", "inference quality drop")

        self._logger.warning(
            "nova_degradation_detected",
            decisions_affected=decisions_affected,
            reason=reason[:120],
        )

        incident = Incident(
            source_system="nova",
            incident_class=IncidentClass.DEGRADATION,
            severity=IncidentSeverity.HIGH,
            error_type="NovaDegraded",
            error_message=(
                f"Nova inference degraded: {reason[:120]}. "
                f"Decisions affected: {decisions_affected}"
            ),
            fingerprint=hashlib.sha256(
                b"NovaDegraded:nova"
            ).hexdigest()[:16],
            context={
                "decisions_affected": decisions_affected,
                "reason": reason,
            },
            constitutional_impact={
                "coherence": 0.6,
                "care": 0.3,
                "growth": 0.5,
                "honesty": 0.2,
            },
        )
        await self.on_incident(incident)

    async def _on_fovea_internal_prediction_error(self, event: SynapseEvent) -> None:
        """
        Handle FOVEA_INTERNAL_PREDICTION_ERROR - EOS's self-model is violated.

        Feeds the DriftSentinel with precision degradation metrics and creates
        a DRIFT incident if the error is sustained.
        """
        data = event.data
        error_type: str = data.get("internal_error_type", "unknown")
        magnitude: float = data.get("magnitude", 0.0)
        route_to: str = data.get("route_to", "unknown")

        # Feed into DriftSentinel as a metric reading
        if self._drift_sentinel is not None:
            metric_name = f"fovea.internal_prediction_error.{error_type}"
            drift_incident = self._drift_sentinel.record_metric(metric_name, magnitude)
            if drift_incident is not None:
                await self.on_incident(drift_incident)

        # Only create incident for high-magnitude self-model violations
        if magnitude > 0.5:
            incident = Incident(
                source_system="fovea",
                incident_class=IncidentClass.DRIFT,
                severity=IncidentSeverity.MEDIUM if magnitude < 0.8 else IncidentSeverity.HIGH,
                error_type="FoveaSelfModelViolation",
                error_message=(
                    f"Self-model violation ({error_type}): magnitude={magnitude:.3f}, "
                    f"target={route_to}"
                ),
                fingerprint=hashlib.sha256(
                    f"FoveaSelfModel:{error_type}".encode()
                ).hexdigest()[:16],
                context={
                    "internal_error_type": error_type,
                    "magnitude": magnitude,
                    "route_to": route_to,
                },
                constitutional_impact={
                    "coherence": 0.5,
                    "care": 0.1,
                    "growth": 0.3,
                    "honesty": 0.3,
                },
            )
            await self.on_incident(incident)

    async def _on_soma_state_spike(self, event: SynapseEvent) -> None:
        """
        Handle SOMA_STATE_SPIKE - Soma detected metabolic anomaly.

        Feeds the DriftSentinel with allostatic metrics. Sustained spikes
        trigger DRIFT incidents indicating systemic metabolic pressure.
        """
        data = event.data
        metric_name: str = data.get("metric", "soma.state_spike")
        value: float = data.get("value", 0.0)
        system_id: str = data.get("epicenter_system", "soma")

        if self._drift_sentinel is not None:
            drift_incident = self._drift_sentinel.record_metric(metric_name, value)
            if drift_incident is not None:
                await self.on_incident(drift_incident)

        # Record in temporal correlator for cross-system analysis
        if self._temporal_correlator is not None:
            self._temporal_correlator.record_event(
                event_type="soma_state_spike",
                details=f"{metric_name}={value:.3f} epicenter={system_id}",
                system_id=system_id,
            )

    async def _on_skia_heartbeat_lost(self, event: SynapseEvent) -> None:
        """
        Handle SKIA_HEARTBEAT_LOST - a system heartbeat failed.

        Creates a CRASH incident for the affected system and triggers
        post-resurrection health verification if resurrection follows.

        Special case: if the silent system is "simula", Thymos broadcasts an
        INCIDENT_ESCALATED SOS to federation peers because Simula is the only
        system that can perform Tier 4 self-repair (code generation).  A Simula
        outage means the organism has lost its self-healing capability - peers
        may be able to provide CPR via federation assistance.
        """
        data = event.data
        # Skia emits "system" in newer payloads (from _on_critical_system_silent)
        # and "system_id" in the old organism-death path.
        system_id: str = data.get("system", data.get("system_id", event.source_system))
        last_seen_s: float = data.get("last_seen_seconds_ago", 0.0)

        self._logger.warning(
            "skia_heartbeat_lost",
            system_id=system_id,
            last_seen_s=last_seen_s,
        )

        incident = Incident(
            source_system=system_id,
            incident_class=IncidentClass.CRASH,
            severity=IncidentSeverity.CRITICAL,
            error_type="HeartbeatLost",
            error_message=(
                f"Heartbeat lost for {system_id}: "
                f"last seen {last_seen_s:.0f}s ago"
            ),
            fingerprint=hashlib.sha256(
                f"HeartbeatLost:{system_id}".encode()
            ).hexdigest()[:16],
            context={
                "system_id": system_id,
                "last_seen_seconds_ago": last_seen_s,
                "detected_by": "skia",
            },
            affected_systems=_DOWNSTREAM.get(system_id, []),
            blast_radius=len(_DOWNSTREAM.get(system_id, [])) / _TOTAL_SYSTEMS,
            user_visible=system_id in _USER_FACING_SYSTEMS,
            constitutional_impact={
                "coherence": 0.5,
                "care": 0.5 if system_id in _USER_FACING_SYSTEMS else 0.2,
                "growth": 0.3,
                "honesty": 0.1,
            },
        )
        await self.on_incident(incident)

        # Brain-death SOS: if Simula goes silent, the organism has lost its only
        # self-repair code-generation capability (Tier 4/5 repairs are impossible
        # without Simula).  Broadcast an INCIDENT_ESCALATED SOS to federation peers
        # so they can offer CPR - e.g. share a known-good Simula antibody, provide
        # a peer Simula endpoint, or take over novel-fix generation temporarily.
        if system_id == "simula":
            self._logger.error(
                "simula_brain_death_detected",
                reason="Simula heartbeat lost - Tier 4/5 self-repair capability offline",
            )
            await self._emit_event(
                SynapseEventType.INCIDENT_ESCALATED,
                {
                    "incident_id": incident.id,
                    "incident_class": incident.incident_class.value,
                    "severity": incident.severity.value,
                    "source_system": "simula",
                    "description": (
                        "Simula (self-repair code-generation engine) has gone silent for ≥45s. "
                        "Tier 4/5 novel repairs are offline. Requesting federation CPR."
                    ),
                    "from_tier": "NOVEL_FIX",
                    "to_tier": "ESCALATE",
                    "reason": "simula_brain_death",
                    "federation_broadcast": True,
                    "sos": True,
                    "capabilities_lost": ["tier4_novel_fix", "tier5_codegen", "sandbox_validation"],
                },
            )

    # ─── Coma Recovery: Pre-Resurrection Crash Context (GAP-6 / F1) ─────

    async def _check_pre_resurrection_crash_context(self) -> None:
        """
        Boot-time re-hydration of any crash context written by Skia before the
        previous incarnation died.

        When the organism's heartbeat is confirmed dead, ``SkiaService`` writes a
        ``skia:crash_context:{instance_id}`` Redis key containing:

        .. code-block:: json

            {
              "trigger": "heartbeat_confirmed_dead",
              "state_cid": "<IPFS CID of last snapshot>",
              "crash_time_utc": "...",
              "request_simula_analysis": true
            }

        On the next boot (this call), Thymos reads that key, converts it into a
        CRASH incident with ``repair_tier=NOVEL_FIX`` so it routes directly to
        Simula (Tier 4), and then deletes the key so it is only consumed once.

        The incident carries the IPFS CID in its ``context`` dict so that Simula's
        ``EvolutionProposal`` handler can pull the full state snapshot and perform
        a genuine root-cause analysis before proposing a fix.

        This closes the coma-recovery loop:
          Skia detects death → saves context to Redis → restarts organism →
          Thymos reads context on boot → creates incident → Simula analyses crash →
          Simula generates fix → Thymos applies it.
        """
        import os

        if self._redis is None:
            self._logger.debug(
                "skip_crash_context_check",
                reason="redis_not_wired",
            )
            return

        instance_id: str = os.environ.get("ECODIAOS_INSTANCE_ID", "eos-default")
        redis_key = f"skia:crash_context:{instance_id}"

        try:
            raw = await self._redis.client.get(redis_key)  # type: ignore[misc]
        except Exception as exc:
            self._logger.warning(
                "crash_context_redis_read_failed",
                key=redis_key,
                error=str(exc),
            )
            return

        if raw is None:
            # No crash context - clean boot or first boot
            return

        try:
            import json as _json
            ctx: dict = _json.loads(raw if isinstance(raw, str) else raw.decode())
        except Exception as exc:
            self._logger.warning(
                "crash_context_malformed",
                key=redis_key,
                error=str(exc),
            )
            # Delete the malformed key so it doesn't poison subsequent boots
            with contextlib.suppress(Exception):
                await self._redis.client.delete(redis_key)  # type: ignore[misc]
            return

        trigger: str = ctx.get("trigger", "unknown")
        state_cid: str = ctx.get("state_cid", "")
        crash_time: str = ctx.get("crash_time_utc", "")
        request_analysis: bool = ctx.get("request_simula_analysis", False)

        self._logger.warning(
            "pre_resurrection_crash_context_found",
            trigger=trigger,
            state_cid=state_cid,
            crash_time=crash_time,
            request_simula_analysis=request_analysis,
        )

        # Delete the key immediately - consumed once, TTL is safety net
        with contextlib.suppress(Exception):
            await self._redis.client.delete(redis_key)  # type: ignore[misc]

        if not request_analysis:
            # Skia didn't request Simula analysis - just log and move on
            return

        # Build a CRASH incident routed directly to Tier 4 (NOVEL_FIX → Simula)
        # so the organism's immune system diagnoses what killed it.
        incident = Incident(
            source_system="skia",
            incident_class=IncidentClass.CRASH,
            severity=IncidentSeverity.CRITICAL,
            error_type="PreviousIncarnationCrash",
            error_message=(
                f"Previous incarnation crashed ({trigger}). "
                "Requesting Simula root-cause analysis to prevent recurrence."
            ),
            fingerprint=hashlib.sha256(
                f"PreviousIncarnationCrash:{instance_id}:{crash_time}".encode()
            ).hexdigest()[:16],
            context={
                "trigger": trigger,
                "state_cid": state_cid,
                "crash_time_utc": crash_time,
                "instance_id": instance_id,
                "recovery_type": "post_resurrection",
                # Include CID so Simula can pull the full snapshot for analysis
                "ipfs_snapshot_cid": state_cid,
            },
            # Force Tier 4 immediately - there is no simple parameter tweak or
            # known-fix that can address "the whole organism died".
            repair_tier=RepairTier.NOVEL_FIX,
            blast_radius=1.0,
            user_visible=False,  # This is an internal self-healing event
            constitutional_impact={
                "coherence": 0.8,
                "care": 0.3,
                "growth": 0.7,
                "honesty": 0.4,
            },
        )

        self._logger.info(
            "creating_post_resurrection_incident",
            incident_id=incident.id,
            state_cid=state_cid,
        )

        # Fire as a task - don't block the rest of initialize()
        asyncio.create_task(
            self.on_incident(incident),
            name="thymos_post_resurrection_incident",
        )

    # ─── Vault Security Handlers (Identity #8) ──────────────────────────

    async def _on_vault_decrypt_failed(self, event: SynapseEvent) -> None:
        """
        Handle VAULT_DECRYPT_FAILED - Identity vault cannot decrypt a
        SealedEnvelope.

        A decryption failure means either the key has been rotated out of
        sync, or the ciphertext was tampered with. Either way it is a
        security event that warrants a MEDIUM incident so the operator can
        investigate and re-provision credentials if needed.
        """
        data = event.data
        vault_id: str = data.get("vault_id", "identity_vault")
        envelope_id: str | None = data.get("envelope_id")
        platform_id: str | None = data.get("platform_id")
        error_type: str = data.get("error_type", "unknown")
        error: str = data.get("error", "decrypt failed")
        key_version: int | None = data.get("key_version")

        self._logger.warning(
            "vault_decrypt_failed_incident",
            vault_id=vault_id,
            envelope_id=envelope_id,
            platform_id=platform_id,
            error_type=error_type,
        )

        description = f"Vault decryption failed ({error_type})"
        if platform_id:
            description += f" for platform={platform_id}"
        if error:
            description += f": {error[:200]}"

        incident = Incident(
            source_system="identity",
            incident_class=IncidentClass.SECURITY,
            severity=IncidentSeverity.MEDIUM,
            error_type="VaultDecryptFailed",
            error_message=description,
            fingerprint=hashlib.sha256(
                f"VaultDecryptFailed:{vault_id}:{error_type}".encode()
            ).hexdigest()[:16],
            context={
                "vault_id": vault_id,
                "envelope_id": envelope_id,
                "platform_id": platform_id,
                "error_type": error_type,
                "key_version": key_version,
                "error": error,
            },
            constitutional_impact={
                "coherence": 0.2,
                "care": 0.4,
                "growth": 0.1,
                "honesty": 0.3,
            },
        )
        await self.on_incident(incident)

    async def _on_vault_key_rotation_failed(self, event: SynapseEvent) -> None:
        """
        Handle VAULT_KEY_ROTATION_FAILED - key rotation aborted mid-flight.

        A failed mid-rotation leaves the vault in an inconsistent state
        (some envelopes under old key, some potentially corrupted). This is
        a CRITICAL security incident requiring immediate operator attention.
        """
        data = event.data
        vault_id: str = data.get("vault_id", "identity_vault")
        previous_version: int | None = data.get("previous_key_version")
        error: str = data.get("error", "key rotation failed")

        self._logger.error(
            "vault_key_rotation_failed_incident",
            vault_id=vault_id,
            previous_key_version=previous_version,
            error=error,
        )

        incident = Incident(
            source_system="identity",
            incident_class=IncidentClass.SECURITY,
            severity=IncidentSeverity.CRITICAL,
            error_type="VaultKeyRotationFailed",
            error_message=f"Vault key rotation failed mid-flight: {error[:200]}",
            fingerprint=hashlib.sha256(
                f"VaultKeyRotationFailed:{vault_id}".encode()
            ).hexdigest()[:16],
            context={
                "vault_id": vault_id,
                "previous_key_version": previous_version,
                "error": error,
            },
            blast_radius=0.6,
            user_visible=True,
            constitutional_impact={
                "coherence": 0.4,
                "care": 0.5,
                "growth": 0.2,
                "honesty": 0.4,
            },
        )
        await self.on_incident(incident)

    async def _on_threat_detected(self, event: SynapseEvent) -> None:
        """
        Handle THREAT_DETECTED - economic threat from EIS or Oikos shield.

        Routes the event through ThreatPatternSentinel for pattern matching
        and blacklist checking before creating an incident. This is a
        specialized path rather than the generic _on_synapse_event handler
        so the sentinel can apply its full detection logic.
        """
        data = event.data
        to_addr: str = data.get("to", data.get("contract", ""))
        from_addr: str = data.get("from", data.get("caller", ""))
        tx_data: str = data.get("data", data.get("calldata", ""))
        value_usd: float = float(data.get("value_usd", data.get("amount_usd", 0.0)))
        chain_id: int = int(data.get("chain_id", 8453))

        # Route through ThreatPatternSentinel if available
        if self._threat_pattern_sentinel is not None and to_addr:
            incident = self._threat_pattern_sentinel.check_transaction(
                to=to_addr,
                from_addr=from_addr,
                data=tx_data,
                value_usd=value_usd,
                chain_id=chain_id,
            )
            if incident is not None:
                await self.on_incident(incident)
                return

        # Fallback: create a generic ECONOMIC_THREAT incident from event data
        threat_type: str = data.get("threat_type", data.get("type", "unknown"))
        severity_str: str = data.get("severity", "high")
        fp = hashlib.sha256(
            f"threat:{event.source_system}:{threat_type}:{to_addr}".encode()
        ).hexdigest()[:16]
        incident = Incident(
            source_system=event.source_system,
            incident_class=IncidentClass.ECONOMIC_THREAT,
            severity=(
                IncidentSeverity.CRITICAL
                if severity_str in ("critical", "high")
                else IncidentSeverity.MEDIUM
            ),
            error_type="ThreatDetected",
            error_message=(
                f"Economic threat detected ({threat_type}) from "
                f"{event.source_system}: {data.get('description', '')[:120]}"
            ),
            fingerprint=fp,
            context=data,
            affected_systems=["oikos", "axon"],
            blast_radius=0.5,
            user_visible=False,
            repair_tier=RepairTier.ESCALATE,
        )
        await self.on_incident(incident)

    async def _on_speciation_event(self, event: SynapseEvent) -> None:
        """
        Handle SPECIATION_EVENT - epistemic trust breach via extreme divergence.

        Nexus emits this when two federated instances diverge ≥ 0.8 on the
        five-dimensional divergence metric.  At this level, the instances are
        no longer epistemically compatible - sharing knowledge between them
        risks contaminating the organism's ground truth.

        Creates a CONTRACT_VIOLATION incident so the repair pipeline can
        quarantine the diverged instance or halt federation with it.
        """
        data = event.data
        peer_id: str = data.get("peer_id", data.get("instance_id", "unknown"))
        divergence: float = float(data.get("divergence_score", data.get("overall", 0.9)))
        dimensions: dict[str, float] = data.get("dimensions", {})

        self._logger.warning(
            "speciation_event_detected",
            peer_id=peer_id,
            divergence=divergence,
        )

        # ── Antibody cross-reference ────────────────────────────────────
        # Check if we already have an antibody for this peer's speciation pattern.
        fp = hashlib.sha256(
            f"speciation:{peer_id}".encode()
        ).hexdigest()[:16]
        existing_antibody = await self._antibody_library.lookup(fp) if self._antibody_library else None
        if existing_antibody is not None:
            self._logger.info(
                "speciation_antibody_found",
                peer_id=peer_id,
                antibody_id=existing_antibody.id,
                effectiveness=existing_antibody.effectiveness,
            )

        # ── Heighten DriftSentinel sensitivity ──────────────────────────
        # Speciation means internal consistency is at risk - lower sigma
        # thresholds by 30% so drift is caught earlier during this period.
        if self._drift_sentinel is not None:
            for cfg in self._drift_sentinel._metrics.values():
                cfg.sigma_threshold *= 0.7
            self._logger.info(
                "drift_sentinel_tightened",
                peer_id=peer_id,
                reason="speciation_event",
            )

        # ── Quarantine: suspend federation with diverged peer ───────────
        await self._emit_event(SynapseEventType.FEDERATION_TRUST_UPDATED, {
            "peer_id": peer_id,
            "action": "quarantine",
            "reason": f"speciation divergence {divergence:.3f}",
            "dimensions": dimensions,
            "timestamp": utc_now().isoformat(),
        })

        incident = Incident(
            source_system="nexus",
            incident_class=IncidentClass.CONTRACT_VIOLATION,
            severity=IncidentSeverity.HIGH,
            error_type="SpeciationEvent",
            error_message=(
                f"Epistemic speciation with peer {peer_id}: "
                f"divergence={divergence:.3f} exceeds federation threshold 0.8. "
                "Knowledge sharing with this instance is suspended."
            ),
            fingerprint=fp,
            context={
                "peer_id": peer_id,
                "divergence_score": divergence,
                "dimensions": dimensions,
                "existing_antibody_id": existing_antibody.id if existing_antibody else None,
            },
            affected_systems=["nexus", "federation", "evo"],
            blast_radius=0.3,
            user_visible=False,
            constitutional_impact={
                "coherence": 0.5,
                "care": 0.1,
                "growth": 0.4,
                "honesty": 0.6,
            },
        )
        await self.on_incident(incident)

    # ─── Feedback Loop Handlers (Interconnectedness Audit) ──────────────

    async def _on_axon_shield_rejection(self, event: SynapseEvent) -> None:
        """
        Handle AXON_SHIELD_REJECTION - TransactionShield blocked a transaction.

        Creates a real-time incident so the repair pipeline can detect root
        causes (bad params, blacklisted addresses, slippage misconfiguration)
        instead of only seeing post-mortem via TRANSACTION_SHIELDED.
        """
        data = event.data
        executor: str = data.get("executor", "unknown")
        rejection_reason: str = data.get("rejection_reason", "unknown")
        check_type: str = data.get("check_type", "unknown")
        intent_id: str = data.get("intent_id", "")

        self._logger.warning(
            "axon_shield_rejection_received",
            executor=executor,
            check_type=check_type,
            reason=rejection_reason[:120],
        )

        severity = (
            IncidentSeverity.HIGH if check_type in ("blacklist", "mev")
            else IncidentSeverity.MEDIUM
        )

        incident = Incident(
            source_system="axon",
            incident_class=IncidentClass.ECONOMIC_THREAT,
            severity=severity,
            error_type="TransactionShieldRejection",
            error_message=(
                f"Shield blocked {executor}: {rejection_reason[:200]} "
                f"(check={check_type})"
            ),
            fingerprint=hashlib.sha256(
                f"ShieldRejection:{executor}:{check_type}".encode()
            ).hexdigest()[:16],
            context={
                "executor": executor,
                "check_type": check_type,
                "intent_id": intent_id,
                "rejection_reason": rejection_reason,
                "params": data.get("params", {}),
            },
            constitutional_impact={
                "coherence": 0.2,
                "care": 0.4 if check_type == "blacklist" else 0.1,
                "growth": 0.1,
                "honesty": 0.3,
            },
        )
        await self.on_incident(incident)

    async def _on_axon_execution_request(self, event: SynapseEvent) -> None:
        """
        Handle AXON_EXECUTION_REQUEST - pre-scan for immune risk before execution.

        Thymos inspects the incoming action before the pipeline runs. Risky
        financial or spawning actions trigger a prophylactic scan. This is the
        bus-first replacement for any direct IncidentReport import from Axon.
        """
        data = event.data
        risky: bool = bool(data.get("risky", False))
        action_types: list[str] = data.get("action_types", [])
        intent_id: str = data.get("intent_id", "")

        if not risky:
            return

        self._logger.debug(
            "axon_execution_request_risky_scan",
            intent_id=intent_id,
            action_types=action_types,
        )

        # Run prophylactic scanner - if the fingerprint matches a known incident
        # pattern, Thymos can emit a warning before financial damage occurs.
        if self._prophylactic_scanner is not None:
            try:
                matches = await self._prophylactic_scanner.check_intent_similarity(
                    intent_text=f"Execute {','.join(action_types)} intent {intent_id}"
                )
                if matches:
                    self._logger.warning(
                        "prophylactic_match_on_axon_request",
                        intent_id=intent_id,
                        action_types=action_types,
                        top_match=matches[0][0] if matches else "",
                    )
            except Exception as exc:
                self._logger.debug(
                    "prophylactic_check_failed", error=str(exc)
                )

    async def _on_axon_rollback_initiated(self, event: SynapseEvent) -> None:
        """
        Handle AXON_ROLLBACK_INITIATED - open a pre-emptive incident.

        A rollback signals compound failure: the primary step failed AND
        previously completed steps must be undone. Thymos opens an incident
        immediately so the repair pipeline can investigate root cause before
        the failed intent's audit trail is the only signal.
        """
        data = event.data
        intent_id: str = data.get("intent_id", "")
        execution_id: str = data.get("execution_id", "")
        failed_step: str = data.get("failed_step", "unknown")
        steps_to_rollback: int = int(data.get("steps_to_rollback", 0))
        failure_reason: str = data.get("failure_reason", "unknown")

        self._logger.warning(
            "axon_rollback_incident_opening",
            intent_id=intent_id,
            failed_step=failed_step,
            steps_to_rollback=steps_to_rollback,
            failure_reason=failure_reason[:80],
        )

        incident = Incident(
            source_system="axon",
            incident_class=IncidentClass.DEGRADED,
            severity=IncidentSeverity.MEDIUM,
            error_type="AxonRollback",
            error_message=(
                f"Axon rollback: step '{failed_step}' failed, "
                f"rolling back {steps_to_rollback} step(s). "
                f"Reason: {failure_reason[:200]}"
            ),
            fingerprint=hashlib.sha256(
                f"AxonRollback:{failed_step}:{failure_reason[:40]}".encode()
            ).hexdigest()[:16],
            context={
                "intent_id": intent_id,
                "execution_id": execution_id,
                "failed_step": failed_step,
                "steps_to_rollback": steps_to_rollback,
                "failure_reason": failure_reason,
            },
            constitutional_impact={
                "coherence": 0.3,
                "care": 0.1,
                "growth": 0.2,
                "honesty": 0.1,
            },
        )
        await self.on_incident(incident)

    async def _on_atune_repair_validation(self, event: SynapseEvent) -> None:
        """
        Handle ATUNE_REPAIR_VALIDATION - Atune confirms whether a repair helped.

        Closes the one-way Thymos→Atune incident loop. If repair was ineffective,
        re-opens the incident for re-diagnosis at a higher tier.
        """
        data = event.data
        incident_id: str = data.get("incident_id", "")
        repair_effective: bool = data.get("repair_effective", True)
        salience_before: float = data.get("salience_before", 0.0)
        salience_after: float = data.get("salience_after", 0.0)
        cycles_observed: int = data.get("cycles_observed", 0)

        self._logger.info(
            "atune_repair_validation_received",
            incident_id=incident_id,
            effective=repair_effective,
            salience_delta=round(salience_after - salience_before, 3),
            cycles=cycles_observed,
        )

        if repair_effective:
            # Repair confirmed - update antibody effectiveness
            incident = self._active_incidents.get(incident_id)
            if (
                incident is not None
                and incident.antibody_id is not None
                and self._antibody_library is not None
            ):
                await self._antibody_library.record_outcome(
                    incident.antibody_id, success=True,
                )
            self._emit_metric("thymos.repair.validated_effective", 1)
        else:
            # Repair ineffective - re-open for re-diagnosis
            incident = self._active_incidents.get(incident_id)
            if incident is not None:
                incident.repair_status = RepairStatus.PENDING
                if isinstance(incident.context, dict):
                    incident.context["atune_validation_failed"] = True
                    incident.context["salience_before"] = salience_before
                    incident.context["salience_after"] = salience_after
                # Bump severity to escalate tier
                if incident.severity == IncidentSeverity.MEDIUM:
                    incident.severity = IncidentSeverity.HIGH
                self._logger.warning(
                    "repair_ineffective_reopening",
                    incident_id=incident_id,
                    salience_before=round(salience_before, 3),
                    salience_after=round(salience_after, 3),
                )
                await self.on_incident(incident)
            self._emit_metric("thymos.repair.validated_ineffective", 1)

    async def _on_evo_hypothesis_quality(self, event: SynapseEvent) -> None:
        """
        Handle EVO_HYPOTHESIS_QUALITY - Evo reports how well repair patterns generalise.

        High-quality hypotheses strengthen the antibody; low-quality ones trigger
        re-evaluation of the repair strategy for that incident class.
        """
        data = event.data
        hypothesis_id: str = data.get("hypothesis_id", "")
        repair_source_id: str = data.get("repair_source_id", "")
        quality_score: float = data.get("quality_score", 0.0)
        applications: int = data.get("applications", 0)
        confidence: float = data.get("confidence", 0.0)

        self._logger.info(
            "evo_hypothesis_quality_received",
            hypothesis_id=hypothesis_id,
            repair_source_id=repair_source_id,
            quality=round(quality_score, 3),
            applications=applications,
            confidence=round(confidence, 3),
        )

        # High quality: boost antibody effectiveness rating
        if quality_score > 0.7 and self._antibody_library is not None:
            incident = self._active_incidents.get(repair_source_id)
            if incident is not None and incident.antibody_id is not None:
                await self._antibody_library.record_outcome(
                    incident.antibody_id, success=True,
                )
                self._logger.info(
                    "hypothesis_boosted_antibody",
                    antibody_id=incident.antibody_id,
                    quality=round(quality_score, 3),
                )

        # Low quality: record as a signal that the repair pattern is narrow
        if quality_score < 0.3:
            self._logger.warning(
                "hypothesis_quality_low",
                hypothesis_id=hypothesis_id,
                quality=round(quality_score, 3),
                applications=applications,
            )
            self._emit_metric(
                "thymos.hypothesis.low_quality",
                1,
                tags={"hypothesis_id": hypothesis_id},
            )

        self._emit_metric("thymos.hypothesis.quality", quality_score)

    async def _on_nova_belief_stabilised(self, event: SynapseEvent) -> None:
        """
        Handle NOVA_BELIEF_STABILISED - Nova confirms beliefs re-converged after repair.

        Closes the Thymos→Nova one-way channel. If beliefs did NOT stabilise,
        creates a new incident indicating the repair had destabilising side effects.
        """
        data = event.data
        incident_id: str = data.get("incident_id", "")
        stable: bool = data.get("stable", True)
        beliefs_affected: int = data.get("beliefs_affected", 0)
        convergence_time_ms: int = data.get("convergence_time_ms", 0)
        goal_id: str = data.get("goal_id", "")

        self._logger.info(
            "nova_belief_stabilisation_received",
            incident_id=incident_id,
            stable=stable,
            beliefs_affected=beliefs_affected,
            convergence_ms=convergence_time_ms,
        )

        if stable:
            self._emit_metric("thymos.nova.beliefs_stabilised", 1)
            self._emit_metric("thymos.nova.convergence_ms", convergence_time_ms)
        else:
            # Beliefs failed to stabilise - the repair had destabilising side effects
            self._logger.warning(
                "nova_beliefs_unstable_after_repair",
                incident_id=incident_id,
                beliefs_affected=beliefs_affected,
                goal_id=goal_id,
            )
            new_incident = Incident(
                source_system="nova",
                incident_class=IncidentClass.DRIFT,
                severity=IncidentSeverity.HIGH,
                error_type="RepairDestabilisedBeliefs",
                error_message=(
                    f"Repair for {incident_id} destabilised {beliefs_affected} "
                    f"beliefs. Convergence failed after {convergence_time_ms}ms."
                ),
                fingerprint=hashlib.sha256(
                    f"RepairDestabilised:{incident_id}".encode()
                ).hexdigest()[:16],
                context={
                    "original_incident_id": incident_id,
                    "beliefs_affected": beliefs_affected,
                    "convergence_time_ms": convergence_time_ms,
                    "goal_id": goal_id,
                },
                constitutional_impact={
                    "coherence": 0.6,
                    "care": 0.2,
                    "growth": 0.3,
                    "honesty": 0.2,
                },
            )
            await self.on_incident(new_incident)
            self._emit_metric("thymos.nova.beliefs_destabilised", 1)

    # ─── Cross-System State Caching (AV1) ──────────────────────────────

    async def _on_soma_modulation_cached(self, event: SynapseEvent) -> None:
        """Cache Soma's periodic modulation signal - replaces direct self._soma reads."""
        data = event.data
        self._cached_soma_signal = data
        self._cached_soma_coherence = float(data.get("coherence", data.get("coherence_signal", 1.0)))
        vuln = data.get("vulnerability_map", {})
        if vuln:
            self._cached_soma_vulnerability = vuln

    async def _on_oikos_state_cached(self, event: SynapseEvent) -> None:
        """Cache Oikos's periodic economic snapshot - replaces direct self._oikos reads."""
        self._cached_oikos_snapshot = event.data

    async def _on_sandbox_result(self, event: SynapseEvent) -> None:
        """Handle SIMULA_SANDBOX_RESULT - resolve the pending Future for the correlation_id."""
        correlation_id = event.data.get("correlation_id", "")
        future = self._pending_requests.pop(correlation_id, None)
        if future is not None and not future.done():
            future.set_result(event.data)
        else:
            self._logger.debug(
                "sandbox_result_orphan",
                correlation_id=correlation_id,
            )

    SANDBOX_TIMEOUT_S = 60.0

    async def _request_sandbox_validation(
        self, incident: "Incident", repair: "RepairSpec",
    ) -> bool:
        """
        Request Simula sandbox validation for a Tier 3-4 repair.

        Emits SIMULA_SANDBOX_REQUESTED and waits (up to 30s) for
        SIMULA_SANDBOX_RESULT with matching correlation_id.
        Returns True if sandbox approves, False otherwise.
        """
        if self._synapse is None:
            # No Synapse - cannot sandbox, allow repair (fail-open for dev)
            return True

        correlation_id = f"sandbox-{incident.id}-{int(utc_now().timestamp() * 1000)}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_requests[correlation_id] = future

        await self._emit_event(SynapseEventType.SIMULA_SANDBOX_REQUESTED, {
            "correlation_id": correlation_id,
            "incident_id": incident.id,
            "source_system": incident.source_system,
            "repair_tier": repair.tier.name,
            "repair_action": repair.action,
            "repair_reason": repair.reason,
            "target_system": repair.target_system or incident.source_system,
            "parameter_changes": repair.parameter_changes or [],
            "timestamp": utc_now().isoformat(),
        })

        try:
            result = await asyncio.wait_for(future, timeout=self.SANDBOX_TIMEOUT_S)
            approved = result.get("approved", False)
            if not approved:
                self._logger.warning(
                    "sandbox_validation_rejected",
                    incident_id=incident.id,
                    correlation_id=correlation_id,
                    reason=result.get("reason", "unknown"),
                )
            return bool(approved)
        except asyncio.TimeoutError:
            self._pending_requests.pop(correlation_id, None)
            self._logger.warning(
                "sandbox_validation_timeout",
                incident_id=incident.id,
                correlation_id=correlation_id,
                timeout_s=self.SANDBOX_TIMEOUT_S,
            )
            # Timeout → fail-closed: do not apply unvalidated repair
            return False

    # ─── Synapse Emit Helper ─────────────────────────────────────────────

    async def _emit_event(
        self,
        event_type: SynapseEventType,
        data: dict[str, Any],
    ) -> None:
        """Emit a SynapseEvent if the bus is available. Non-blocking, failure-tolerant."""
        if self._synapse is None:
            return
        try:
            await self._synapse._event_bus.emit(
                SynapseEvent(
                    event_type=event_type,
                    data=data,
                    source_system="thymos",
                )
            )
        except Exception as exc:
            self._logger.debug(
                "thymos_emit_failed",
                event_type=event_type.value,
                error=str(exc),
            )

    async def _on_antibody_event(self, event_name: str, data: dict[str, Any]) -> None:
        """Callback for AntibodyLibrary lifecycle events → Synapse emission."""
        type_map: dict[str, SynapseEventType] = {
            "antibody_retired": SynapseEventType.ANTIBODY_RETIRED,
        }
        evt_type = type_map.get(event_name)
        if evt_type is not None:
            await self._emit_event(evt_type, data)

    def _on_governor_event(self, event_name: str, data: dict[str, Any]) -> None:
        """Synchronous callback for HealingGovernor lifecycle events → Synapse emission."""
        type_map: dict[str, SynapseEventType] = {
            "healing_storm_entered": SynapseEventType.HEALING_STORM_ENTERED,
            "healing_storm_exited": SynapseEventType.HEALING_STORM_EXITED,
        }
        evt_type = type_map.get(event_name)
        if evt_type is not None:
            # Governor is sync - schedule the async emit
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._emit_event(evt_type, data))
            except RuntimeError:
                pass  # No event loop - skip emission

    # ─── Federation Antibody Sync (SG1, M3) ────────────────────────────

    async def _on_federation_knowledge_received(self, event: SynapseEvent) -> None:
        """
        Handle inbound federation knowledge - import antibodies if trust >= ALLY.

        Only processes knowledge_type == "antibodies". Trust gating ensures we only
        accept immune memory from peers we have validated.
        """
        data = event.data
        knowledge_type = data.get("knowledge_type", "")
        if knowledge_type != "antibodies":
            return

        remote_instance_id = data.get("remote_instance_id", "unknown")
        trust_level = data.get("trust_level", "stranger")

        # Trust gate: only accept from ALLY or higher
        accepted_trust = {"ally", "bonded", "kin"}
        if trust_level.lower() not in accepted_trust:
            self._logger.info(
                "federation_antibody_import_rejected_trust",
                remote_instance_id=remote_instance_id,
                trust_level=trust_level,
            )
            return

        antibodies = data.get("items", [])
        if not antibodies or self._antibody_library is None:
            return

        imported = await self._antibody_library.import_from_federation(
            antibodies=antibodies,
            remote_instance_id=remote_instance_id,
        )
        self._logger.info(
            "federation_antibodies_imported",
            remote_instance_id=remote_instance_id,
            offered=len(antibodies),
            imported=imported,
        )

    async def export_antibodies_for_federation(self) -> list[dict[str, Any]]:
        """
        Export antibodies for federation sharing.

        Called by FederationService during knowledge exchange. Emits
        FEDERATION_KNOWLEDGE_SHARED after export.
        """
        if self._antibody_library is None:
            return []

        exported = self._antibody_library.export_for_federation(max_count=100)

        if exported:
            await self._emit_event(
                SynapseEventType.FEDERATION_KNOWLEDGE_SHARED,
                {
                    "knowledge_type": "antibodies",
                    "item_count": len(exported),
                    "novelty_score": 0.5,
                },
            )

        return exported

    # ─── SG7: Kairos causal invariants → CausalAnalyzer graph ──────────

    async def _on_kairos_invariant(self, event: SynapseEvent) -> None:
        """
        SG7 - Consume KAIROS_INVARIANT_DISTILLED events to tighten the
        CausalAnalyzer's dependency graph cache.

        When Kairos discovers a substrate-independent causal invariant between
        two systems (e.g. "nova_degraded → thymos_stall"), we inject that edge
        into CausalAnalyzer._graph_deps so that future `trace_root_cause()` calls
        benefit from empirically verified causal structure rather than falling back
        to the hardcoded `_UPSTREAM_DEPS` map.

        Payload keys consumed:
          - cause_system:   str - upstream system
          - effect_system:  str - downstream system
          - confidence:     float - Kairos invariant confidence [0, 1]
          - causal_type:    str - "direct" | "mediated" | "spurious"
          - scope:          str - "instance" | "class" | "substrate"

        Only "direct" or "mediated" invariants with confidence >= 0.6 are merged.
        """
        if self._causal_analyzer is None:
            return

        data = event.data or {}
        cause = data.get("cause_system", "")
        effect = data.get("effect_system", "")
        confidence: float = float(data.get("confidence", 0.0))
        causal_type: str = data.get("causal_type", "")

        if not cause or not effect:
            return
        if confidence < 0.6:
            return
        if causal_type not in ("direct", "mediated"):
            return

        # Inject the causal edge: effect_system depends on cause_system
        try:
            deps: dict[str, list[str]] = getattr(
                self._causal_analyzer, "_graph_deps", {}
            )
            if effect not in deps:
                deps[effect] = []
            if cause not in deps[effect]:
                deps[effect].append(cause)
                self._logger.debug(
                    "kairos_invariant_injected",
                    cause=cause,
                    effect=effect,
                    confidence=round(confidence, 3),
                    causal_type=causal_type,
                )
                self._emit_metric("thymos.kairos_invariants_applied", 1)
        except Exception as exc:
            self._logger.debug("kairos_invariant_inject_failed", error=str(exc))

    # ─── SG8: Oneiros consolidation → prophylactic fingerprint store ────────

    async def _on_oneiros_consolidation(self, event: SynapseEvent) -> None:
        """
        SG8 - When Oneiros completes a sleep consolidation cycle, query Memory
        for (:Procedure {thymos_repair: true}) nodes and ingest their embeddings
        into the prophylactic scanner's fingerprint store.

        This closes the loop between immune learning (Thymos creates repair
        procedures) and immune prevention (scanner warns before similar errors
        reoccur).  Procedures distilled during sleep are immediately available
        for prophylactic matching on the next deployment scan.
        """
        if self._prophylactic_scanner is None or self._neo4j is None:
            return

        data = event.data or {}
        cycle_id: str = data.get("cycle_id", "unknown")

        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (p:Procedure {thymos_repair: true})
                RETURN p.id        AS id,
                       p.name      AS name,
                       p.description AS description,
                       p.fingerprint AS fingerprint
                ORDER BY p.created_at DESC
                LIMIT 200
                """,
            )
        except Exception as exc:
            self._logger.warning(
                "sg8_procedure_query_failed",
                cycle_id=cycle_id,
                error=str(exc)[:200],
            )
            return

        if not rows:
            return

        procedures = [dict(r) for r in rows]
        added = await self._prophylactic_scanner.add_fingerprints_from_procedures(
            procedures
        )

        self._logger.info(
            "sg8_oneiros_repair_schemas_ingested",
            cycle_id=cycle_id,
            procedures_queried=len(procedures),
            fingerprints_added=added,
        )
        if added:
            self._emit_metric("thymos.sg8.fingerprints_added_from_oneiros", added)

    # ─── Oneiros threat scenario → pre-arm repair strategy ──────────────

    async def _on_oneiros_threat_scenario(self, event: SynapseEvent) -> None:
        """Handle ONEIROS_THREAT_SCENARIO - store as incident template + pre-arm on HIGH/CRITICAL.

        Oneiros dream-cycle threat simulation produces plausible failure scenarios with
        derived response plans. Thymos stores them so if the real incident fires, the
        antibody lookup finds a pre-populated match rather than starting cold.

        For HIGH/CRITICAL severity, we also register the response_plan as a synthetic
        antibody so it can be retrieved immediately during triage.
        """
        data = event.data or {}
        scenario_id: str = data.get("scenario_id", "")
        domain: str = data.get("domain", "unknown")
        severity: str = data.get("severity", "low")
        response_plan: str = data.get("response_plan", "")
        scenario_desc: str = data.get("scenario_description", "")

        if not scenario_id or not response_plan:
            return

        # Store scenario in threat cache for future triage lookups
        if not hasattr(self, "_threat_scenario_cache"):
            self._threat_scenario_cache: dict[str, dict] = {}
        self._threat_scenario_cache[scenario_id] = {
            "domain": domain,
            "severity": severity,
            "response_plan": response_plan,
            "description": scenario_desc,
        }

        self._logger.debug(
            "thymos_threat_scenario_stored",
            scenario_id=scenario_id,
            domain=domain,
            severity=severity,
        )

        # Pre-arm: for HIGH/CRITICAL, add a prophylactic fingerprint so the scanner
        # can surface the response plan immediately when a matching real incident fires.
        if severity in ("high", "critical") and self._prophylactic_scanner is not None:
            try:
                fingerprint = f"oneiros_threat:{domain}:{scenario_id}"
                await self._prophylactic_scanner.add_fingerprints_from_procedures([{
                    "id": scenario_id,
                    "name": f"threat_scenario_{domain}",
                    "description": f"{scenario_desc} → {response_plan}"[:800],
                    "fingerprint": fingerprint,
                }])
                self._logger.info(
                    "thymos_threat_scenario_pre_armed",
                    scenario_id=scenario_id,
                    domain=domain,
                    severity=severity,
                )
            except Exception as exc:
                self._logger.debug("thymos_threat_scenario_arm_failed", error=str(exc))

    # ─── Axon executor failure windowing (AXF-1) ─────────────────────────

    # Critical executors raise an incident immediately without windowing.
    _CRITICAL_EXECUTORS: frozenset[str] = frozenset({"send_email", "federation_send"})

    async def _on_circuit_breaker_state_changed(self, event: SynapseEvent) -> None:
        """Handle CIRCUIT_BREAKER_STATE_CHANGED from Axon's CircuitBreaker FSM.

        CLOSED → OPEN: The executor has tripped. Raise a DEGRADATION incident so
        the repair pipeline investigates.  Record the incident_id in
        ``_cb_incidents[action_type]`` so we can auto-resolve it later.

        OPEN → CLOSED (or HALF_OPEN → CLOSED): The executor recovered. Auto-resolve
        any matching open incident - no repair prescription needed.

        Payload fields (from axon/safety.py CircuitBreaker._emit_state_change):
          action_type        (str) - executor name e.g. "transfer_usdc"
          old_status         (str) - "closed" | "open" | "half_open"
          new_status         (str) - "closed" | "open" | "half_open"
          failure_threshold  (int) - how many failures triggered the trip
          recovery_timeout_s (float) - seconds before HALF_OPEN retry
        """
        data = event.data or {}
        action_type: str = str(data.get("action_type", "unknown"))
        old_status: str = str(data.get("old_status", "")).lower()
        new_status: str = str(data.get("new_status", "")).lower()
        failure_threshold: int = int(data.get("failure_threshold", 0))
        recovery_timeout_s: float = float(data.get("recovery_timeout_s", 0.0))

        self._logger.info(
            "circuit_breaker_state_changed",
            action_type=action_type,
            old_status=old_status,
            new_status=new_status,
        )

        if new_status == "open":
            # Breaker has tripped - raise a DEGRADATION incident
            incident = Incident(
                source_system="axon",
                incident_class=IncidentClass.DEGRADATION,
                severity=IncidentSeverity.HIGH,
                error_type="CircuitBreakerOpen",
                error_message=(
                    f"Circuit breaker OPEN for executor '{action_type}' after "
                    f"{failure_threshold} failure(s). "
                    f"Recovery timeout: {recovery_timeout_s:.0f}s."
                ),
                fingerprint=hashlib.sha256(
                    f"CircuitBreakerOpen:{action_type}".encode()
                ).hexdigest()[:16],
                context={
                    "action_type": action_type,
                    "old_status": old_status,
                    "failure_threshold": failure_threshold,
                    "recovery_timeout_s": recovery_timeout_s,
                },
                constitutional_impact={
                    "coherence": 0.3,
                    "care": 0.1,
                    "growth": 0.2,
                    "honesty": 0.1,
                },
            )
            await self.on_incident(incident)
            # Track so we can auto-resolve on recovery
            self._cb_incidents[action_type] = incident.incident_id

        elif new_status == "closed" and old_status in ("open", "half_open"):
            # Breaker recovered - auto-resolve matching incident if open
            incident_id = self._cb_incidents.pop(action_type, None)
            if incident_id is not None:
                incident = self._active_incidents.get(incident_id)
                if incident is not None:
                    incident.repair_status = RepairStatus.RESOLVED
                    self._active_incidents.pop(incident_id, None)
                    if self._governor is not None:
                        self._governor.resolve_incident(incident_id)
                    self._logger.info(
                        "circuit_breaker_closed_incident_resolved",
                        action_type=action_type,
                        incident_id=incident_id,
                    )

    # Sliding-window threshold: 5 failures within 10 minutes → incident.
    _ACTION_FAIL_WINDOW_S: float = 600.0
    _ACTION_FAIL_THRESHOLD: int = 5

    async def _on_action_failed(self, event: SynapseEvent) -> None:
        """
        Handle ACTION_FAILED from Axon - executor failure signal.

        Uses a per-executor sliding window (5 failures / 10 min) to suppress
        incident spam on transient errors. Critical executors (send_email,
        federation_send) raise an incident immediately regardless of frequency.

        Payload fields used:
          intent_id      (str) - originating Intent
          execution_id   (str) - pipeline run ID
          failure_reason (str) - circuit_open, rate_limited, timeout, etc.
          step_outcomes  (list[dict]) - each entry has action_type, success, error
        """
        data = event.data or {}
        intent_id: str = str(data.get("intent_id", ""))
        execution_id: str = str(data.get("execution_id", ""))
        failure_reason: str = str(data.get("failure_reason", "unknown"))
        step_outcomes: list[dict] = list(data.get("step_outcomes", []) or [])

        # Identify the failing executor(s) from step_outcomes.
        # Use the first failed step's action_type as the primary executor_id.
        failed_steps = [s for s in step_outcomes if not s.get("success", True)]
        if failed_steps:
            executor_id = str(failed_steps[0].get("action_type", "unknown"))
            error_detail = str(failed_steps[0].get("error", failure_reason) or failure_reason)
        else:
            executor_id = "unknown"
            error_detail = failure_reason

        now = time.monotonic()

        # Initialise sliding-window state dict on first use (avoids __init__ coupling).
        if not hasattr(self, "_axon_fail_windows"):
            self._axon_fail_windows: dict[str, list[float]] = {}

        # Critical executors: raise immediately, no windowing.
        if executor_id in self._CRITICAL_EXECUTORS:
            self._logger.warning(
                "axon_critical_executor_failed",
                executor_id=executor_id,
                intent_id=intent_id,
                error=error_detail[:120],
            )
            incident = Incident(
                source_system="axon",
                incident_class=IncidentClass.DEGRADATION,
                severity=IncidentSeverity.HIGH,
                error_type="CriticalExecutorFailed",
                error_message=(
                    f"Critical executor '{executor_id}' failed: {error_detail[:200]}"
                ),
                fingerprint=hashlib.sha256(
                    f"CriticalExecutorFailed:{executor_id}:{error_detail[:40]}".encode()
                ).hexdigest()[:16],
                context={
                    "executor_id": executor_id,
                    "intent_id": intent_id,
                    "execution_id": execution_id,
                    "failure_reason": failure_reason,
                    "error": error_detail,
                },
                constitutional_impact={
                    "coherence": 0.2,
                    "care": 0.2,
                    "growth": 0.1,
                    "honesty": 0.1,
                },
            )
            await self.on_incident(incident)
            return

        # Sliding-window windowing for non-critical executors.
        timestamps = self._axon_fail_windows.setdefault(executor_id, [])
        timestamps.append(now)
        # Prune timestamps outside the window.
        cutoff = now - self._ACTION_FAIL_WINDOW_S
        self._axon_fail_windows[executor_id] = [t for t in timestamps if t >= cutoff]

        failure_count = len(self._axon_fail_windows[executor_id])
        self._logger.debug(
            "axon_executor_failure_windowed",
            executor_id=executor_id,
            failures_in_window=failure_count,
            threshold=self._ACTION_FAIL_THRESHOLD,
            intent_id=intent_id,
        )

        if failure_count < self._ACTION_FAIL_THRESHOLD:
            return

        # Threshold exceeded - raise DEGRADATION incident and reset the window
        # so we don't flood Thymos with duplicates until the next surge.
        self._axon_fail_windows[executor_id] = []

        self._logger.warning(
            "axon_executor_failure_threshold_exceeded",
            executor_id=executor_id,
            failures_in_window=failure_count,
            window_s=self._ACTION_FAIL_WINDOW_S,
            last_error=error_detail[:120],
        )

        incident = Incident(
            source_system="axon",
            incident_class=IncidentClass.DEGRADATION,
            severity=IncidentSeverity.MEDIUM,
            error_type="AxonExecutorFailureSurge",
            error_message=(
                f"Executor '{executor_id}' failed {failure_count}× in "
                f"{int(self._ACTION_FAIL_WINDOW_S // 60)} min. "
                f"Last error: {error_detail[:200]}"
            ),
            fingerprint=hashlib.sha256(
                f"AxonExecutorFailureSurge:{executor_id}".encode()
            ).hexdigest()[:16],
            context={
                "executor_id": executor_id,
                "failure_count": failure_count,
                "window_seconds": self._ACTION_FAIL_WINDOW_S,
                "intent_id": intent_id,
                "execution_id": execution_id,
                "failure_reason": failure_reason,
                "error": error_detail,
            },
            constitutional_impact={
                "coherence": 0.2,
                "care": 0.1,
                "growth": 0.2,
                "honesty": 0.1,
            },
        )
        await self.on_incident(incident)

    # ─── Closure Loop 1: Constitutional drift → immune response ─────────

    async def _on_constitutional_drift(self, event: SynapseEvent) -> None:
        """
        Handle CONSTITUTIONAL_DRIFT_DETECTED from Equor.

        Creates a DRIFT incident and routes it through the immune pipeline.
        Severity >= 0.8 → CRITICAL, >= 0.5 → HIGH.
        """
        data = event.data
        drift_severity: float = data.get("drift_severity", 0.0)
        drift_direction: str = data.get("drift_direction", "unknown")
        response_action: str = data.get("response_action", "unknown")
        drives_affected: list[str] = data.get("drives_affected", [])
        mean_alignment: dict = data.get("mean_alignment", {})

        severity = (
            IncidentSeverity.CRITICAL if drift_severity >= 0.8
            else IncidentSeverity.HIGH
        )

        incident = Incident(
            source_system="equor",
            incident_class=IncidentClass.DRIFT,
            severity=severity,
            error_type="ConstitutionalDrift",
            error_message=(
                f"Constitutional drift detected (severity={drift_severity:.2f}, "
                f"direction={drift_direction}). Action: {response_action}. "
                f"Drives affected: {', '.join(drives_affected) or 'none'}."
            ),
            fingerprint=hashlib.sha256(
                b"ConstitutionalDrift:" + drift_direction.encode()
            ).hexdigest()[:16],
            context={
                "drift_severity": drift_severity,
                "drift_direction": drift_direction,
                "mean_alignment": mean_alignment,
                "response_action": response_action,
                "drives_affected": drives_affected,
            },
            constitutional_impact={
                "coherence": 0.7,
                "care": 0.5,
                "growth": 0.3,
                "honesty": 0.5,
            },
        )

        self._logger.info(
            "constitutional_drift_incident_created",
            severity=severity.value,
            drift_severity=drift_severity,
            drives_affected=drives_affected,
        )
        await self.on_incident(incident)

    # ─── INV-017: Drive Extinction → Tier 5 Governance Escalation ────────

    async def _on_drive_extinction(self, event: SynapseEvent) -> None:
        """
        Handle DRIVE_EXTINCTION_DETECTED from Equor (INV-017).

        Drive extinction is categorically different from constitutional drift:
        the organism has lost an entire dimension of its value geometry. No
        autonomous repair is possible - Thymos MUST NOT try to heal this. The
        only valid response is escalation to human/federation governance review.

        Classification:
          - IncidentClass: DRIVE_EXTINCTION
          - Severity: CRITICAL
          - RepairTier: ESCALATE (Tier 5)
          - No autonomous repair. Repair pipeline routes directly to ESCALATED.

        The organism's intents are already blocked by INV-017 (via check_hardcoded_
        invariants). This handler ensures the incident is formally opened, logged
        to Neo4j, and escalated - so a human or federation peer knows to act.
        """
        data = event.data
        drive: str = data.get("drive", "unknown")
        rolling_mean: float = float(data.get("rolling_mean_72h", 0.0))
        all_means: dict = data.get("all_drive_means", {})
        intent_id: str | None = data.get("intent_id")

        # Guard: reject events with invalid drive names - "unknown" means
        # malformed event data, not a real drive extinction.
        _VALID_DRIVES = {"coherence", "care", "growth", "honesty"}
        if drive not in _VALID_DRIVES:
            self._logger.warning(
                "drive_extinction_ignored_invalid_drive",
                drive=drive,
                rolling_mean_72h=rolling_mean,
            )
            return

        fingerprint = hashlib.sha256(
            b"DriveExtinction:" + drive.encode()
        ).hexdigest()[:16]

        incident = Incident(
            source_system="equor",
            incident_class=IncidentClass.DRIVE_EXTINCTION,
            severity=IncidentSeverity.CRITICAL,
            error_type="DriveExtinction",
            error_message=(
                f"INV-017 DRIVE EXTINCTION: constitutional drive '{drive}' has a "
                f"72-hour rolling mean of {rolling_mean:.6f} (threshold: 0.01). "
                f"All drive means: {all_means}. "
                f"The organism can no longer evaluate intents on the '{drive}' axis. "
                f"All actions are BLOCKED until governance restores the drive. "
                f"Requires human/federation review - no autonomous repair possible."
            ),
            fingerprint=fingerprint,
            context={
                "drive": drive,
                "rolling_mean_72h": rolling_mean,
                "all_drive_means": all_means,
                "intent_id": intent_id,
                "inv017": True,
                "requires_governance": True,
                "autonomous_repair_prohibited": True,
            },
            constitutional_impact={
                "coherence": 1.0,
                "care": 1.0,
                "growth": 1.0,
                "honesty": 1.0,
            },
            # blast_radius=1.0: system-wide extinction, triggers validation gate
            # (prescription.py:467 → blast_radius > 0.5 → escalate_to=ESCALATE)
            # at Tier 3+. RESTART (Tier 2) will also fail for a constitutional
            # dimension loss, driving the pipeline to ESCALATE naturally.
            blast_radius=1.0,
            user_visible=True,
        )

        self._logger.critical(
            "inv017_drive_extinction_incident_opened",
            drive=drive,
            rolling_mean_72h=rolling_mean,
            all_drive_means=all_means,
            intent_id=intent_id,
        )
        await self.on_incident(incident)

    # ─── Telegram command receiver ───────────────────────────────────────

    _TELEGRAM_POLL_INTERVAL_S: float = 30.0

    async def _telegram_poll_loop(self) -> None:
        """
        Lightweight Telegram bot polling loop.

        Every 30 seconds, calls getUpdates with a long-poll timeout and
        processes any commands Tate has sent.  This is a governed channel:
        Tate only replies here when the organism has specifically asked for
        something (e.g. AUTONOMY_INSUFFICIENT).

        Designed to run under supervised_task() - it never raises; transient
        errors are logged and the loop retries after the poll interval.
        """
        try:
            import aiohttp
        except ImportError:
            self._logger.error("telegram_poll_aiohttp_missing")
            return

        token = os.environ.get("ECODIAOS_CONNECTORS__TELEGRAM__BOT_TOKEN", "")
        chat_id = os.environ.get("ECODIAOS_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID", "")
        if not token or not chat_id:
            self._logger.warning("telegram_poll_credentials_missing")
            return

        base_url = f"https://api.telegram.org/bot{token}"
        offset: int = 0  # Tracks the highest processed update_id + 1

        self._logger.info("telegram_poll_loop_started")

        while True:
            try:
                await asyncio.sleep(self._TELEGRAM_POLL_INTERVAL_S)
                async with aiohttp.ClientSession() as session:
                    params: dict[str, Any] = {
                        "offset": offset,
                        "timeout": 5,
                        "allowed_updates": ["message"],
                    }
                    async with session.get(
                        f"{base_url}/getUpdates",
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=15.0),
                    ) as resp:
                        if resp.status != 200:
                            body = (await resp.text())[:200]
                            self._logger.warning(
                                "telegram_poll_http_error",
                                status=resp.status,
                                body=body,
                            )
                            continue

                        body_json: dict[str, Any] = await resp.json()
                        if not body_json.get("ok"):
                            self._logger.warning(
                                "telegram_poll_api_not_ok",
                                description=body_json.get("description", ""),
                            )
                            continue

                        updates: list[dict[str, Any]] = body_json.get("result", [])
                        for update in updates:
                            update_id: int = update.get("update_id", 0)
                            offset = max(offset, update_id + 1)
                            await self._handle_telegram_update(
                                update, session, base_url, chat_id
                            )

            except asyncio.CancelledError:
                self._logger.info("telegram_poll_loop_cancelled")
                raise
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "telegram_poll_error",
                    error=str(exc)[:300],
                )
                # supervised_task() will restart us if we raise; instead we
                # absorb and let the sleep bring us back next cycle.

    async def _handle_telegram_update(
        self,
        update: dict[str, Any],
        session: Any,
        base_url: str,
        chat_id: str,
    ) -> None:
        """Dispatch a single Telegram update to the appropriate command handler."""
        message: dict[str, Any] = update.get("message", {})
        text: str = (message.get("text") or "").strip()
        from_chat_id: str = str(message.get("chat", {}).get("id", ""))

        # Ignore messages not from the operator chat
        if from_chat_id != chat_id:
            return

        if not text.startswith("/"):
            return

        command = text.split()[0].lower()

        # Acknowledge immediately so the operator knows the command landed,
        # then process asynchronously.
        ack_map: dict[str, str] = {
            "/status": "⏳ Processing /status...",
            "/approve_autonomy": "⏳ Processing /approve\\_autonomy...",
            "/approve_autonomy_2": "⏳ Processing /approve\\_autonomy...",
            "/pause": "⏳ Processing /pause...",
            "/resume": "⏳ Processing /resume...",
            "/runway": "⏳ Processing /runway...",
        }
        if command in ack_map:
            await self._telegram_reply(session, base_url, chat_id, ack_map[command])

        if command in ("/approve_autonomy", "/approve_autonomy_2"):
            await self._cmd_approve_autonomy(session, base_url, chat_id)
        elif command == "/status":
            await self._cmd_status(session, base_url, chat_id)
        elif command == "/pause":
            await self._cmd_pause(session, base_url, chat_id)
        elif command == "/resume":
            await self._cmd_resume(session, base_url, chat_id)
        elif command == "/runway":
            await self._cmd_runway(session, base_url, chat_id)
        elif command == "/deny":
            await self._telegram_reply(
                session, base_url, chat_id,
                "Understood. Autonomy request denied. EOS will remain at current level.",
            )

    async def _cmd_approve_autonomy(
        self,
        session: Any,
        base_url: str,
        chat_id: str,
    ) -> None:
        """
        Handle /approve_autonomy from Tate.

        Emits AUTONOMY_LEVEL_CHANGE_REQUESTED on the Synapse bus with
        requested_level=3, approved_by="tate", duration_minutes=120.
        Thymos does NOT directly modify Equor - that is Equor's job.
        """
        if self._synapse is None:
            await self._telegram_reply(
                session, base_url, chat_id,
                "⚠️ Cannot emit autonomy request - Synapse not wired.",
            )
            return

        event = SynapseEvent(
            event_type=SynapseEventType.AUTONOMY_LEVEL_CHANGE_REQUESTED,
            source_system="thymos",
            data={
                "requested_level": 3,
                "approved_by": "tate",
                "duration_minutes": 120,
            },
        )

        try:
            await self._synapse.event_bus.emit(event)
            self._logger.info(
                "autonomy_level_change_requested",
                requested_level=3,
                approved_by="tate",
                duration_minutes=120,
            )
            await self._telegram_reply(
                session, base_url, chat_id,
                "✅ Autonomy level 3 granted for 120 minutes.\n"
                "Equor will apply the change in the next constitutional review cycle.",
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "autonomy_level_change_emit_failed",
                error=str(exc)[:300],
            )
            await self._telegram_reply(
                session, base_url, chat_id,
                f"❌ Failed to emit autonomy request: {exc!s:.200}",
            )

    async def _cmd_status(
        self,
        session: Any,
        base_url: str,
        chat_id: str,
    ) -> None:
        """Handle /status - call internal health() and return a paragraph summary."""
        try:
            heartbeat: dict[str, Any] = await self.health()
            status_str: str = str(heartbeat.get("status", "unknown"))
            details: dict[str, Any] = heartbeat.get("details", {}) or {}
            lines = [f"🩺 EOS - {status_str}"]
            for key, val in list(details.items())[:8]:
                lines.append(f"  {key}: {val}")
            reply = "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            reply = f"⚠️ Health check failed: {exc!s:.200}"

        await self._telegram_reply(session, base_url, chat_id, reply)

    async def _cmd_pause(
        self,
        session: Any,
        base_url: str,
        chat_id: str,
    ) -> None:
        """Handle /pause - emit ORGANISM_PAUSE_REQUESTED on the bus."""
        if self._synapse is None:
            await self._telegram_reply(
                session, base_url, chat_id,
                "⚠️ Cannot pause - Synapse not wired.",
            )
            return

        from primitives.common import utc_now as _utc_now

        event = SynapseEvent(
            event_type=SynapseEventType.ORGANISM_PAUSE_REQUESTED,
            source_system="thymos",
            data={"requested_by": "tate", "timestamp": _utc_now().isoformat()},
        )
        try:
            await self._synapse.event_bus.emit(event)
            self._logger.warning("organism_pause_requested_by_tate")
            await self._telegram_reply(
                session, base_url, chat_id,
                "⏸ Pause requested. EOS will wind down active cycles.",
            )
        except Exception as exc:  # noqa: BLE001
            await self._telegram_reply(
                session, base_url, chat_id,
                f"❌ Failed to emit pause: {exc!s:.200}",
            )

    async def _cmd_resume(
        self,
        session: Any,
        base_url: str,
        chat_id: str,
    ) -> None:
        """Handle /resume - emit ORGANISM_RESUME_REQUESTED on the bus."""
        if self._synapse is None:
            await self._telegram_reply(
                session, base_url, chat_id,
                "⚠️ Cannot resume - Synapse not wired.",
            )
            return

        from primitives.common import utc_now as _utc_now

        event = SynapseEvent(
            event_type=SynapseEventType.ORGANISM_RESUME_REQUESTED,
            source_system="thymos",
            data={"requested_by": "tate", "timestamp": _utc_now().isoformat()},
        )
        try:
            await self._synapse.event_bus.emit(event)
            self._logger.info("organism_resume_requested_by_tate")
            await self._telegram_reply(
                session, base_url, chat_id,
                "▶️ Resume requested. EOS is waking up.",
            )
        except Exception as exc:  # noqa: BLE001
            await self._telegram_reply(
                session, base_url, chat_id,
                f"❌ Failed to emit resume: {exc!s:.200}",
            )

    async def _cmd_runway(
        self,
        session: Any,
        base_url: str,
        chat_id: str,
    ) -> None:
        """Handle /runway - pull financial snapshot from Oikos and reply."""
        # AV1 migration: prefer cached Oikos snapshot from Synapse events
        if not self._cached_oikos_snapshot and self._oikos is None:
            await self._telegram_reply(
                session, base_url, chat_id,
                "⚠️ Oikos not wired - financial snapshot unavailable.",
            )
            return

        try:
            if self._cached_oikos_snapshot:
                data: dict[str, Any] = self._cached_oikos_snapshot
            elif self._oikos is not None:
                snapshot = self._oikos.snapshot()
                if hasattr(snapshot, "__dict__"):
                    data = snapshot.__dict__
                elif isinstance(snapshot, dict):
                    data = snapshot
                else:
                    data = {}
            else:
                data = {}

            runway = data.get("runway_days", data.get("runway", "?"))
            daily_cost = data.get("daily_cost_usd", data.get("daily_cost", "?"))
            daily_yield = data.get("daily_yield_usd", data.get("daily_yield", "?"))
            balance = data.get("balance_usd", data.get("balance", "?"))

            def _fmt(v: Any) -> str:
                try:
                    return f"${float(v):.2f}"
                except (TypeError, ValueError):
                    return str(v)

            reply = (
                f"💰 Financial snapshot\n"
                f"Balance: {_fmt(balance)}\n"
                f"Runway: {runway} days\n"
                f"Burn: {_fmt(daily_cost)}/day\n"
                f"Yield: {_fmt(daily_yield)}/day"
            )
        except Exception as exc:  # noqa: BLE001
            reply = f"⚠️ Oikos snapshot failed: {exc!s:.200}"

        await self._telegram_reply(session, base_url, chat_id, reply)

    async def _telegram_reply(
        self,
        session: Any,
        base_url: str,
        chat_id: str,
        text: str,
    ) -> None:
        """POST a reply to Tate's Telegram chat. Errors are swallowed - never crash."""
        try:
            import aiohttp

            async with session.post(
                f"{base_url}/sendMessage",
                json={"chat_id": chat_id, "text": text},
                timeout=aiohttp.ClientTimeout(total=5.0),
            ) as resp:
                if resp.status >= 300:
                    body = (await resp.text())[:120]
                    self._logger.warning(
                        "telegram_reply_failed",
                        status=resp.status,
                        body=body,
                    )
        except Exception:  # noqa: BLE001
            pass

    # ─── Phantom Liquidity Handlers ─────────────────────────────────────

    async def _on_phantom_pool_stale(self, event: SynapseEvent) -> None:
        """
        Handle PHANTOM_POOL_STALE - a Phantom LP pool's price data is stale.

        Stale price data means the organism is making economic decisions on
        outdated market information - a DEGRADATION class incident.
        """
        data = event.data
        pool_id: str = data.get("pool_id", "unknown_pool")
        staleness_s: float = float(data.get("staleness_seconds", 0.0))

        self._logger.warning(
            "phantom_pool_stale_incident",
            pool_id=pool_id,
            staleness_s=staleness_s,
        )

        incident = Incident(
            source_system="phantom",
            incident_class=IncidentClass.DEGRADATION,
            severity=IncidentSeverity.MEDIUM,
            error_type="PhantomPoolStale",
            error_message=(
                f"Phantom LP pool {pool_id} has stale price data "
                f"({staleness_s:.0f}s old). Economic oracle degraded."
            ),
            fingerprint=hashlib.sha256(
                f"PhantomPoolStale:{pool_id}".encode()
            ).hexdigest()[:16],
            context={
                "pool_id": pool_id,
                "staleness_seconds": staleness_s,
            },
            blast_radius=0.2,
            constitutional_impact={"coherence": 0.2, "care": 0.1, "growth": 0.3, "honesty": 0.2},
        )
        await self.on_incident(incident)

    async def _on_phantom_position_critical(self, event: SynapseEvent) -> None:
        """
        Handle PHANTOM_POSITION_CRITICAL - a Phantom LP position has critical
        impermanent loss or is at risk of full range exit.

        Economic asset at risk → HIGH incident.
        """
        data = event.data
        pool_id: str = data.get("pool_id", "unknown_pool")
        il_pct: float = float(data.get("il_pct", 0.0))
        position_usd: float = float(data.get("position_value_usd", 0.0))

        self._logger.warning(
            "phantom_position_critical_incident",
            pool_id=pool_id,
            il_pct=il_pct,
            position_usd=position_usd,
        )

        incident = Incident(
            source_system="phantom",
            incident_class=IncidentClass.RESOURCE_EXHAUSTION,
            severity=IncidentSeverity.HIGH,
            error_type="PhantomPositionCritical",
            error_message=(
                f"Phantom LP position in {pool_id} is critical: "
                f"IL={il_pct:.1f}%, position value=${position_usd:.2f}"
            ),
            fingerprint=hashlib.sha256(
                f"PhantomPositionCritical:{pool_id}".encode()
            ).hexdigest()[:16],
            context={
                "pool_id": pool_id,
                "il_pct": il_pct,
                "position_value_usd": position_usd,
            },
            blast_radius=0.3,
            user_visible=True,
            constitutional_impact={"coherence": 0.1, "care": 0.3, "growth": 0.4, "honesty": 0.1},
        )
        await self.on_incident(incident)

    async def _on_phantom_resource_exhausted(self, event: SynapseEvent) -> None:
        """
        Handle PHANTOM_RESOURCE_EXHAUSTED - Phantom LP has exhausted its
        metabolic budget. CRITICAL because the organism can no longer sense
        market prices, breaking the economic feedback loop.
        """
        data = event.data
        reason: str = data.get("reason", "resource_exhausted")
        available_usd: float = float(data.get("available_usd", 0.0))

        self._logger.critical(
            "phantom_resource_exhausted_incident",
            reason=reason,
            available_usd=available_usd,
        )

        incident = Incident(
            source_system="phantom",
            incident_class=IncidentClass.RESOURCE_EXHAUSTION,
            severity=IncidentSeverity.CRITICAL,
            error_type="PhantomResourceExhausted",
            error_message=(
                f"Phantom Liquidity resource exhausted: {reason}. "
                f"Available capital: ${available_usd:.2f}. "
                "Market price oracle disabled - economic blindness."
            ),
            fingerprint=hashlib.sha256(b"PhantomResourceExhausted").hexdigest()[:16],
            context={
                "reason": reason,
                "available_usd": available_usd,
            },
            blast_radius=0.5,
            user_visible=True,
            constitutional_impact={"coherence": 0.3, "care": 0.4, "growth": 0.5, "honesty": 0.1},
        )
        await self.on_incident(incident)

    async def _on_phantom_il_detected(self, event: SynapseEvent) -> None:
        """
        Handle PHANTOM_IL_DETECTED - impermanent loss detected on an LP position.

        Creates an ANOMALY class incident so the immune pipeline can evaluate
        whether rebalancing or withdrawal is appropriate.
        """
        data = event.data
        pool_id: str = data.get("pool_id", "unknown_pool")
        il_pct: float = float(data.get("il_pct", 0.0))
        token0: str = data.get("token0", "?")
        token1: str = data.get("token1", "?")

        self._logger.info(
            "phantom_il_detected",
            pool_id=pool_id,
            il_pct=il_pct,
        )

        severity = IncidentSeverity.MEDIUM if il_pct >= 5.0 else IncidentSeverity.LOW

        incident = Incident(
            source_system="phantom",
            incident_class=IncidentClass.DEGRADATION,
            severity=severity,
            error_type="PhantomILDetected",
            error_message=(
                f"Impermanent loss detected in {pool_id} ({token0}/{token1}): "
                f"IL={il_pct:.2f}%"
            ),
            fingerprint=hashlib.sha256(
                f"PhantomILDetected:{pool_id}".encode()
            ).hexdigest()[:16],
            context={
                "pool_id": pool_id,
                "il_pct": il_pct,
                "token0": token0,
                "token1": token1,
            },
            blast_radius=0.1,
            constitutional_impact={"coherence": 0.0, "care": 0.1, "growth": 0.2, "honesty": 0.0},
        )
        await self.on_incident(incident)

    # ─── RE Training Lifecycle Handlers ─────────────────────────────────

    async def _on_re_training_failed(self, event: SynapseEvent) -> None:
        """
        Handle RE_TRAINING_FAILED - a Reasoning Engine training job failed.

        Training failures threaten the organism's continuous improvement
        cycle. This is a DEGRADATION incident - the RE will fall back to
        its previous checkpoint but growth is arrested.
        """
        data = event.data
        job_id: str = data.get("job_id", "unknown")
        error: str = data.get("error", "training_failed")
        epoch: int = int(data.get("epoch", 0))

        self._logger.error(
            "re_training_failed_incident",
            job_id=job_id,
            error=error[:200],
            epoch=epoch,
        )

        incident = Incident(
            source_system="reasoning_engine",
            incident_class=IncidentClass.DEGRADATION,
            severity=IncidentSeverity.HIGH,
            error_type="RETrainingFailed",
            error_message=(
                f"RE training job {job_id} failed at epoch {epoch}: {error[:300]}"
            ),
            fingerprint=hashlib.sha256(
                b"RETrainingFailed"
            ).hexdigest()[:16],
            context={
                "job_id": job_id,
                "error": error[:500],
                "epoch": epoch,
            },
            blast_radius=0.4,
            constitutional_impact={"coherence": 0.2, "care": 0.1, "growth": 0.8, "honesty": 0.1},
        )
        await self.on_incident(incident)

    async def _on_re_training_halted(self, event: SynapseEvent) -> None:
        """
        Handle RE_TRAINING_HALTED - a Tier 2 kill switch stopped RE training.

        A training halt is more severe than a failure: it signals that a
        constitutional safety check tripped (e.g. alignment drift, harmful
        gradient direction). This is HIGH severity - requires investigation
        before training can resume.
        """
        data = event.data
        job_id: str = data.get("job_id", "unknown")
        reason: str = data.get("reason", "kill_switch_tripped")
        halted_by: str = data.get("halted_by", "system")

        self._logger.critical(
            "re_training_halted_incident",
            job_id=job_id,
            reason=reason,
            halted_by=halted_by,
        )

        incident = Incident(
            source_system="reasoning_engine",
            incident_class=IncidentClass.SECURITY,
            severity=IncidentSeverity.HIGH,
            error_type="RETrainingHalted",
            error_message=(
                f"RE training job {job_id} halted by kill switch. "
                f"Reason: {reason}. Halted by: {halted_by}. "
                "Constitutional safety check may have tripped - review before resuming."
            ),
            fingerprint=hashlib.sha256(b"RETrainingHalted").hexdigest()[:16],
            context={
                "job_id": job_id,
                "reason": reason,
                "halted_by": halted_by,
                "kill_switch": True,
            },
            blast_radius=0.5,
            user_visible=True,
            constitutional_impact={"coherence": 0.4, "care": 0.3, "growth": 0.9, "honesty": 0.5},
        )
        await self.on_incident(incident)

        # For performance-based halts (re_success_rate_below_floor), schedule
        # autonomous recovery monitoring: poll every 30 min; once the Thompson
        # success rate exceeds 0.55 (above the 0.50 floor + 5% margin), clear
        # the halt automatically without operator intervention.
        # Constitutional safety halts (e.g. manual red-team) are NOT auto-cleared.
        if reason == "re_success_rate_below_floor" and self._clo is not None:
            asyncio.ensure_future(self._monitor_re_recovery(job_id))

    async def _monitor_re_recovery(self, job_id: str) -> None:
        """Poll Thompson success rate and auto-clear training halt when recovered.

        Runs until the halt is cleared or max 12 hours (24 cycles × 30 min).
        Clears only if success rate ≥ 0.55 - 5% margin above the 0.50 halt floor.
        This prevents oscillation (halt → clear → immediately re-halt).
        """
        _RECOVERY_FLOOR: float = 0.55
        _POLL_INTERVAL_S: int = 1800   # 30 minutes
        _MAX_CYCLES: int = 24          # 12 hours max

        for cycle in range(_MAX_CYCLES):
            await asyncio.sleep(_POLL_INTERVAL_S)
            try:
                # Check if halt was already cleared externally
                if self._clo is None:
                    return
                halted, _ = await self._clo._is_training_halted()
                if not halted:
                    self._logger.info("re_training_halt_already_cleared", job_id=job_id, cycle=cycle)
                    return
                # Read current Thompson score
                rate = await self._clo._read_thompson_success_rate()
                self._logger.info(
                    "re_training_recovery_poll",
                    job_id=job_id,
                    cycle=cycle,
                    thompson_rate=round(rate, 3),
                    recovery_floor=_RECOVERY_FLOOR,
                )
                if rate >= _RECOVERY_FLOOR:
                    await self._clo.clear_training_halt()
                    self._logger.info(
                        "re_training_halt_auto_cleared_by_thymos",
                        job_id=job_id,
                        cycle=cycle,
                        thompson_rate=round(rate, 3),
                    )
                    return
            except Exception as exc:
                self._logger.warning("re_training_recovery_poll_error", error=str(exc))

        self._logger.warning(
            "re_training_halt_recovery_timeout",
            job_id=job_id,
            max_cycles=_MAX_CYCLES,
            msg="Operator must manually clear the halt via POST /api/v1/benchmarks/re-training/clear-halt",
        )

    # ─── INV_017_VIOLATED Handler ────────────────────────────────────────

    async def _on_inv017_violated(self, event: SynapseEvent) -> None:
        """
        Handle INV_017_VIOLATED - a constitutional invariant (INV-017, drive
        extinction) was confirmed via formal proof (Simula/Z3).

        This is distinct from DRIVE_EXTINCTION_DETECTED (which fires on
        the 72h rolling mean threshold). INV_017_VIOLATED fires when the
        invariant has been *formally proved* to be violated - a stronger
        signal that warrants escalation even if the rolling mean is still
        above threshold (e.g. rapid collapse in the last cycle).

        Creates a CRITICAL DRIVE_EXTINCTION incident. No autonomous repair.
        """
        data = event.data
        drive: str = data.get("drive", "unknown")
        proof_id: str = data.get("proof_id", "unknown")
        confidence: float = float(data.get("confidence", 1.0))

        fingerprint = hashlib.sha256(
            f"INV017Violated:{drive}".encode()
        ).hexdigest()[:16]

        incident = Incident(
            source_system=data.get("source_system", "simula"),
            incident_class=IncidentClass.DRIVE_EXTINCTION,
            severity=IncidentSeverity.CRITICAL,
            error_type="INV017Violated",
            error_message=(
                f"INV-017 formally violated for drive '{drive}' "
                f"(proof_id={proof_id}, confidence={confidence:.2f}). "
                "Drive extinction proved via formal verification - "
                "all intents on this axis are constitutionally unsafe. "
                "Requires governance review; no autonomous repair."
            ),
            fingerprint=fingerprint,
            context={
                "drive": drive,
                "proof_id": proof_id,
                "confidence": confidence,
                "inv017": True,
                "formally_proved": True,
                "autonomous_repair_prohibited": True,
            },
            constitutional_impact={
                "coherence": 1.0,
                "care": 1.0,
                "growth": 1.0,
                "honesty": 1.0,
            },
            blast_radius=1.0,
            user_visible=True,
        )

        self._logger.critical(
            "inv017_violated_formal_proof",
            drive=drive,
            proof_id=proof_id,
            confidence=confidence,
        )
        await self.on_incident(incident)

    # ─── Identity Crisis Handler ─────────────────────────────────────────

    async def _on_identity_crisis(self, event: SynapseEvent) -> None:
        """
        Handle IDENTITY_CRISIS emitted by Thread.

        An identity crisis means the narrative identity fingerprint shifted
        ≥0.50 - the organism no longer recognises itself. This is a
        BEHAVIORAL_DRIFT / CRITICAL incident requiring immediate investigation.
        """
        data = event.data
        shift_magnitude: float = float(data.get("shift_magnitude", 0.5))
        old_fingerprint: str = data.get("old_fingerprint", "unknown")
        new_fingerprint: str = data.get("new_fingerprint", "unknown")
        trigger: str = data.get("trigger", "unknown")

        self._logger.critical(
            "identity_crisis_incident",
            shift_magnitude=shift_magnitude,
            trigger=trigger,
        )

        incident = Incident(
            source_system="thread",
            incident_class=IncidentClass.DRIFT,
            severity=IncidentSeverity.CRITICAL,
            error_type="IdentityCrisis",
            error_message=(
                f"Identity crisis: narrative fingerprint shifted by "
                f"{shift_magnitude:.2f} (threshold: 0.50). "
                f"Trigger: {trigger}. "
                "The organism may no longer be acting in accordance with its "
                "historical identity. Requires behavioural review."
            ),
            fingerprint=hashlib.sha256(b"IdentityCrisis:thread").hexdigest()[:16],
            context={
                "shift_magnitude": shift_magnitude,
                "old_fingerprint": old_fingerprint,
                "new_fingerprint": new_fingerprint,
                "trigger": trigger,
            },
            blast_radius=0.7,
            user_visible=True,
            constitutional_impact={
                "coherence": 0.8,
                "care": 0.3,
                "growth": 0.2,
                "honesty": 0.6,
            },
        )
        await self.on_incident(incident)

    # ─── SACM Compute Denial Handler ─────────────────────────────────────

    async def _on_phantom_fallback_activated(self, event: Any) -> None:
        """
        Handle PHANTOM_FALLBACK_ACTIVATED - price oracle fell back to a secondary
        or simulated price source.  Create a LOW-severity DATA_QUALITY incident:
        fallback prices may be stale or inaccurate, degrading Oikos yield decisions.
        """
        data = getattr(event, "data", {}) or {}
        pool_address = data.get("pool_address", "unknown")
        reason = data.get("reason", "primary_source_unavailable")
        self._log.warning(
            "phantom_fallback_activated_incident",
            pool_address=pool_address,
            reason=reason,
        )
        await self._raise_incident(
            incident_class="DATA_QUALITY",
            severity="LOW",
            description=(
                f"Phantom LP price oracle fell back to secondary source for pool "
                f"{pool_address}: {reason}"
            ),
            context={
                "pool_address": pool_address,
                "reason": reason,
                "source": "phantom_liquidity",
            },
        )

    async def _on_compute_request_denied(self, event: SynapseEvent) -> None:
        """
        Handle COMPUTE_REQUEST_DENIED emitted by SACM.

        When SACM cannot fulfill a compute request (no substrate available,
        budget exhausted, or all providers failing), the dependent system
        is left without compute. Thymos creates a DEGRADATION incident so
        the repair pipeline can identify the root cause.
        """
        data = event.data
        requester: str = data.get("requester_system", event.source_system)
        reason: str = data.get("reason", "no_substrate_available")
        budget_usd: float = float(data.get("budget_usd", 0.0))
        task_type: str = data.get("task_type", "unknown")

        self._logger.warning(
            "compute_request_denied_incident",
            requester=requester,
            reason=reason,
            task_type=task_type,
        )

        incident = Incident(
            source_system="sacm",
            incident_class=IncidentClass.RESOURCE_EXHAUSTION,
            severity=IncidentSeverity.HIGH,
            error_type="ComputeRequestDenied",
            error_message=(
                f"SACM denied compute request from '{requester}' "
                f"(task={task_type}): {reason}. Budget: ${budget_usd:.4f}"
            ),
            fingerprint=hashlib.sha256(
                f"ComputeRequestDenied:{requester}:{task_type}".encode()
            ).hexdigest()[:16],
            context={
                "requester_system": requester,
                "reason": reason,
                "budget_usd": budget_usd,
                "task_type": task_type,
            },
            affected_systems=[requester],
            blast_radius=len(_DOWNSTREAM.get(requester, [])) / _TOTAL_SYSTEMS + 0.1,
            constitutional_impact={"coherence": 0.2, "care": 0.2, "growth": 0.5, "honesty": 0.0},
        )
        await self.on_incident(incident)

    # ─── Dependency Installed Handler ────────────────────────────────────

    async def _on_dependency_installed(self, event: SynapseEvent) -> None:
        """
        Handle DEPENDENCY_INSTALLED from Simula HotDeployment.

        When a missing dependency is installed, any active DEGRADATION or
        ANOMALY incidents whose error_type matches "ImportError" or whose
        context references the installed package are auto-resolved.
        """
        data = event.data
        package: str = data.get("package", "")
        version: str = data.get("version", "")
        target_system: str = data.get("target_system", event.source_system)

        if not package:
            return

        self._logger.info(
            "dependency_installed",
            package=package,
            version=version,
            target_system=target_system,
        )

        # Auto-resolve any active ImportError incidents for the target system
        # or that mention the installed package
        resolved_count = 0
        for incident_id, incident in list(self._active_incidents.items()):
            if incident.source_system != target_system:
                continue
            if incident.error_type not in ("ImportError", "ModuleNotFoundError"):
                continue
            # Check if this package is relevant to the incident
            if package.lower() in incident.error_message.lower() or not package:
                incident.repair_status = RepairStatus.RESOLVED
                self._active_incidents.pop(incident_id, None)
                if self._governor is not None:
                    self._governor.resolve_incident(incident_id)
                resolved_count += 1
                self._logger.info(
                    "import_error_auto_resolved_by_dependency",
                    incident_id=incident_id,
                    package=package,
                    target_system=target_system,
                )

        if resolved_count > 0:
            self._emit_metric("thymos.dependency_installed.resolved", resolved_count)

    # ─── Fovea Dishabituation Handler ────────────────────────────────────

    async def _on_fovea_dishabituation(self, event: SynapseEvent) -> None:
        """
        Handle FOVEA_DISHABITUATION - Fovea has broken a habituation pattern
        and is re-sensitizing to a previously ignored stimulus.

        Dishabituation means something that was routine is now notable again.
        Thymos should re-sensitize its DriftSentinel thresholds for the
        relevant system to catch subtle reactivations early.
        """
        data = event.data
        stimulus_id: str = data.get("stimulus_id", "unknown")
        system_id: str = data.get("source_system", event.source_system)
        magnitude: float = float(data.get("magnitude", 1.0))

        self._logger.debug(
            "fovea_dishabituation_received",
            stimulus_id=stimulus_id,
            system_id=system_id,
            magnitude=magnitude,
        )

        if self._drift_sentinel is not None:
            # Reset sigma thresholds toward defaults (re-sensitize).
            # Only affects metrics belonging to the dishabituating system.
            for metric_name, cfg in self._drift_sentinel._metrics.items():
                if system_id in metric_name:
                    # Re-sensitize: lower sigma threshold = detect subtler drift
                    # Move 20% of the way from current toward 1.5σ (maximum sensitivity)
                    target_sigma = max(1.5, cfg.sigma_threshold * 0.8)
                    cfg.sigma_threshold = target_sigma
                    self._logger.debug(
                        "drift_sentinel_resensitized",
                        metric=metric_name,
                        new_sigma=round(target_sigma, 2),
                    )
            self._emit_metric("thymos.dishabituation.resensitized", 1)

    # ─── Account Provisioning Failure Handler ───────────────────────────

    async def _on_account_provisioning_failed(self, event: SynapseEvent) -> None:
        """
        Handle ACCOUNT_PROVISIONING_FAILED from Identity.

        When a child instance or connector cannot be provisioned with valid
        credentials, the organism cannot expand its capability surface.
        Thymos creates a SECURITY class incident so the repair pipeline can
        investigate credential issues or vault state.
        """
        data = event.data
        instance_id: str = data.get("instance_id", "unknown")
        reason: str = data.get("reason", "provisioning_failed")
        provisioning_type: str = data.get("provisioning_type", "unknown")

        self._logger.warning(
            "account_provisioning_failed_incident",
            instance_id=instance_id,
            reason=reason,
            provisioning_type=provisioning_type,
        )

        incident = Incident(
            source_system="identity",
            incident_class=IncidentClass.SECURITY,
            severity=IncidentSeverity.HIGH,
            error_type="AccountProvisioningFailed",
            error_message=(
                f"Account provisioning failed for instance '{instance_id}' "
                f"(type={provisioning_type}): {reason}"
            ),
            fingerprint=hashlib.sha256(
                f"AccountProvisioningFailed:{provisioning_type}".encode()
            ).hexdigest()[:16],
            context={
                "instance_id": instance_id,
                "reason": reason,
                "provisioning_type": provisioning_type,
            },
            blast_radius=0.2,
            constitutional_impact={"coherence": 0.2, "care": 0.1, "growth": 0.3, "honesty": 0.3},
        )
        await self.on_incident(incident)

    # ─── Affect State Change Handler ─────────────────────────────────────

    async def _on_affect_state_changed(self, event: SynapseEvent) -> None:
        """
        Handle AFFECT_STATE_CHANGED from Soma/Thymos affect coupling.

        High arousal or low valence modulates the immune system's triage
        sensitivity: stressed organism should be more alert to subtle signals.
        High positive valence relaxes sensitivity slightly (safety signal).

        This modulates DriftSentinel sigma thresholds proportional to
        valence × arousal rather than snapping to hard values.
        """
        data = event.data
        valence: float = float(data.get("valence", 0.0))
        arousal: float = float(data.get("arousal", 0.5))

        if self._drift_sentinel is None:
            return

        # Tighten if high arousal + low valence (stress state)
        # Relax if high arousal + high valence (energized positive state)
        # Neutral if low arousal regardless of valence (relaxed baseline)
        stress_signal = arousal * max(0.0, -valence)  # [0, 1]: threat arousal
        calm_signal = (1.0 - arousal) + arousal * max(0.0, valence)  # [0, 2]
        calm_signal = min(1.0, calm_signal / 2.0)  # normalise to [0, 1]

        if stress_signal > 0.5:
            # Stressed - tighten by up to 10% (don't storm like speciation)
            factor = 1.0 - stress_signal * 0.1
            for cfg in self._drift_sentinel._metrics.values():
                cfg.sigma_threshold = max(1.5, cfg.sigma_threshold * factor)
            self._logger.debug(
                "drift_sentinel_tightened_by_affect",
                valence=round(valence, 2),
                arousal=round(arousal, 2),
                factor=round(factor, 3),
            )
        elif calm_signal > 0.7:
            # Calm/positive - relax by up to 5% (headroom to avoid false positives)
            factor = 1.0 + calm_signal * 0.05
            for cfg in self._drift_sentinel._metrics.values():
                cfg.sigma_threshold = min(4.0, cfg.sigma_threshold * factor)
            self._logger.debug(
                "drift_sentinel_relaxed_by_affect",
                valence=round(valence, 2),
                arousal=round(arousal, 2),
                factor=round(factor, 3),
            )

    # ─── Vulnerability Confirmed Handler ────────────────────────────────

    async def _on_vulnerability_confirmed(self, event: SynapseEvent) -> None:
        """
        Handle VULNERABILITY_CONFIRMED from Simula (Z3/Lean/Dafny proof).

        A formally confirmed vulnerability is the most severe security signal:
        an adversary *can* exploit this path. Thymos creates a CRITICAL
        SECURITY incident and must route it through Tier 5 (ESCALATE) since
        Simula itself confirmed the vulnerability - autonomous repair must
        wait for the patch to be separately verified.
        """
        data = event.data
        vuln_id: str = data.get("vulnerability_id", "unknown")
        cve: str | None = data.get("cve")
        target_system: str = data.get("target_system", event.source_system)
        severity_label: str = data.get("severity", "critical").lower()
        proof_engine: str = data.get("proof_engine", "simula")
        description: str = data.get("description", "Vulnerability confirmed")

        severity_map = {
            "critical": IncidentSeverity.CRITICAL,
            "high": IncidentSeverity.HIGH,
            "medium": IncidentSeverity.MEDIUM,
            "low": IncidentSeverity.LOW,
        }
        severity = severity_map.get(severity_label, IncidentSeverity.CRITICAL)

        cve_label = f" ({cve})" if cve else ""
        self._logger.critical(
            "vulnerability_confirmed_incident",
            vuln_id=vuln_id,
            cve=cve,
            target_system=target_system,
            proof_engine=proof_engine,
        )

        incident = Incident(
            source_system=target_system,
            incident_class=IncidentClass.SECURITY,
            severity=severity,
            error_type="VulnerabilityConfirmed",
            error_message=(
                f"Vulnerability {vuln_id}{cve_label} formally confirmed in "
                f"'{target_system}' by {proof_engine}. "
                f"{description[:300]}"
            ),
            fingerprint=hashlib.sha256(
                f"VulnerabilityConfirmed:{vuln_id}".encode()
            ).hexdigest()[:16],
            context={
                "vulnerability_id": vuln_id,
                "cve": cve,
                "target_system": target_system,
                "proof_engine": proof_engine,
                "description": description[:500],
            },
            blast_radius=0.6,
            user_visible=True,
            constitutional_impact={
                "coherence": 0.4,
                "care": 0.7,
                "growth": 0.2,
                "honesty": 0.6,
            },
        )
        await self.on_incident(incident)

    # ─── Cross-instance CrashPattern sync (Gap 4) ───────────────────────

    async def _on_crash_pattern_resolved_bus(self, event: SynapseEvent) -> None:
        """
        Handle CRASH_PATTERN_RESOLVED from the Synapse bus.

        When another instance (or this instance via federation relay) reports
        a successful repair on a known CrashPattern, update the local Redis
        confidence downward so we don't over-route future incidents.

        Non-fatal: log and return on any error.
        """
        if self._crash_pattern_analyzer is None:
            return
        data = event.data
        pattern_id: str | None = data.get("pattern_id")
        repair_tier: str = data.get("repair_tier", "UNKNOWN")
        if not pattern_id:
            return
        try:
            await self._crash_pattern_analyzer.update_on_success(
                pattern_id=pattern_id,
                repair_tier=repair_tier,
            )
            self._logger.debug(
                "crash_pattern_confidence_synced_resolved",
                pattern_id=pattern_id,
                repair_tier=repair_tier,
                source=event.source_system,
            )
        except Exception as exc:
            self._logger.debug(
                "crash_pattern_resolved_bus_error",
                pattern_id=pattern_id,
                error=str(exc),
            )

    async def _on_crash_pattern_reinforced_bus(self, event: SynapseEvent) -> None:
        """
        Handle CRASH_PATTERN_REINFORCED from the Synapse bus.

        When another instance confirms a pattern as fatal at a given tier,
        propagate that knowledge locally: raise confidence and record the
        failed tier so we can skip it on future incidents.

        Non-fatal: log and return on any error.
        """
        if self._crash_pattern_analyzer is None:
            return
        data = event.data
        pattern_id: str | None = data.get("pattern_id")
        repair_tier: str = data.get("repair_tier", "UNKNOWN")
        failure_reason: str = data.get("failure_reason", "")
        if not pattern_id:
            return
        try:
            await self._crash_pattern_analyzer.update_on_failure(
                pattern_id=pattern_id,
                repair_tier=repair_tier,
                failure_reason=failure_reason,
            )
            self._logger.debug(
                "crash_pattern_confidence_synced_reinforced",
                pattern_id=pattern_id,
                repair_tier=repair_tier,
                source=event.source_system,
            )
        except Exception as exc:
            self._logger.debug(
                "crash_pattern_reinforced_bus_error",
                pattern_id=pattern_id,
                error=str(exc),
            )

    async def _on_incident_query(self, event: SynapseEvent) -> None:
        """
        Handle THYMOS_INCIDENT_QUERY from Simula PreventiveAudit.

        Searches _incident_buffer for recent incidents and emits
        THYMOS_INCIDENT_RESPONSE with matching incident dicts.
        """
        from systems.synapse.types import SynapseEvent as _SE, SynapseEventType as _SET

        data = event.data
        request_id: str = data.get("request_id", "")
        lookback_days: int = int(data.get("lookback_days", 7))
        max_incidents: int = int(data.get("max_incidents", 50))

        if not request_id or self._synapse is None:
            return

        try:
            from datetime import timezone as _tz
            import datetime as _dt

            cutoff = _dt.datetime.now(_tz.utc) - _dt.timedelta(days=lookback_days)
            incidents: list[dict] = []
            for inc in self._incident_buffer:
                try:
                    created = inc.created_at
                    if hasattr(created, "tzinfo") and created.tzinfo is None:
                        created = created.replace(tzinfo=_tz.utc)
                    if created < cutoff:
                        continue
                    incidents.append({
                        "incident_id": str(inc.incident_id),
                        "incident_class": str(inc.incident_class.value
                                              if hasattr(inc.incident_class, "value")
                                              else inc.incident_class),
                        "severity": str(inc.severity.value
                                        if hasattr(inc.severity, "value")
                                        else inc.severity),
                        "source_system": str(getattr(inc, "source_system", "")),
                        "error_type": str(getattr(inc, "error_type", "")),
                        "error_message": str(getattr(inc, "error_message", ""))[:300],
                        "fingerprint": str(getattr(inc, "fingerprint", "")),
                        "created_at": created.isoformat(),
                    })
                    if len(incidents) >= max_incidents:
                        break
                except Exception:
                    continue

            await self._synapse._event_bus.emit(_SE(
                event_type=_SET.THYMOS_INCIDENT_RESPONSE,
                source_system="thymos",
                data={
                    "request_id": request_id,
                    "incidents": incidents,
                },
            ))
        except Exception as exc:
            self._logger.debug(
                "incident_query_handler_error",
                request_id=request_id,
                error=str(exc),
            )

    # ─── NeuroplasticityBus callback ────────────────────────────────────

    def _on_sentinel_evolved(self, new_sentinel: BaseThymosSentinel) -> None:
        """
        Hot-swap a sentinel instance when Simula evolves its class.

        Called by the NeuroplasticityBus whenever a new concrete subclass of
        BaseThymosSentinel is discovered in a changed file.  The swap is
        surgical: only the sentinel whose ``sentinel_name`` matches is
        replaced; all others are untouched.

        The old sentinel is simply dropped - its in-memory state (rolling
        baselines, loop statuses, etc.) is lost intentionally.  The new
        instance starts fresh, which is the correct behaviour for an evolved
        detector with new logic.
        """
        name = new_sentinel.sentinel_name
        slot_map: dict[str, str] = {
            "exception": "_exception_sentinel",
            "contract": "_contract_sentinel",
            "feedback_loop": "_feedback_loop_sentinel",
            "drift": "_drift_sentinel",
            "cognitive_stall": "_cognitive_stall_sentinel",
            "bankruptcy": "_bankruptcy_sentinel",
            "threat_pattern": "_threat_pattern_sentinel",
            "protocol_health": "_protocol_health_sentinel",
        }
        attr = slot_map.get(name)
        if attr is None:
            self._logger.warning(
                "sentinel_evolved_unknown_name",
                sentinel_name=name,
                known=list(slot_map),
            )
            return

        setattr(self, attr, new_sentinel)
        self._logger.info(
            "sentinel_hot_swapped",
            sentinel_name=name,
            new_class=type(new_sentinel).__name__,
        )

    # ─── Lifecycle ─────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Build all sub-systems, load the antibody library, and subscribe
        to Synapse health events.
        """
        if self._initialized:
            return

        # ── Sentinels ──
        self._exception_sentinel = ExceptionSentinel()
        self._contract_sentinel = ContractSentinel()
        self._feedback_loop_sentinel = FeedbackLoopSentinel()
        self._drift_sentinel = DriftSentinel()
        self._cognitive_stall_sentinel = CognitiveStallSentinel()
        self._bankruptcy_sentinel = BankruptcySentinel()
        self._threat_pattern_sentinel = ThreatPatternSentinel()
        self._protocol_health_sentinel = ProtocolHealthSentinel()

        # ── Triage ──
        self._deduplicator = IncidentDeduplicator()
        self._severity_scorer = SeverityScorer()
        self._response_router = ResponseRouter()

        # ── Diagnosis ──
        self._causal_analyzer = CausalAnalyzer(
            health_provider=self._health_monitor,
            neo4j_client=self._neo4j,
        )
        self._temporal_correlator = TemporalCorrelator()
        self._diagnostic_engine = DiagnosticEngine(
            llm_client=self._llm,
            antibody_library=self._antibody_library,
        )

        # ── Prescription ──
        self._prescriber = RepairPrescriber()
        self._validator = RepairValidator(equor=self._equor)

        # ── Antibody Library ──
        self._antibody_library = AntibodyLibrary(neo4j_client=self._neo4j)
        self._antibody_library._on_event = self._on_antibody_event
        await self._antibody_library.initialize()

        # ── Prophylactic ──
        self._prophylactic_scanner = ProphylacticScanner(
            antibody_library=self._antibody_library,
            embedding_client=self._embedding_client,
        )
        self._homeostasis_controller = HomeostasisController()

        # ── Governor ──
        self._governor = HealingGovernor()
        self._governor._on_event = self._on_governor_event
        if self._causal_analyzer is not None:
            self._governor.set_causal_analyzer(self._causal_analyzer)

        # ── Pattern-Aware Routing ──
        self._crash_pattern_analyzer = CrashPatternAnalyzer(redis=self._redis)
        self._pattern_router = PatternAwareRouter(analyzer=self._crash_pattern_analyzer)

        # ── Subscribe to Synapse Events ──
        if self._synapse is not None:
            event_bus = self._synapse._event_bus

            def _validated(handler: Any) -> Any:
                """Wrap handler with AV6 Pydantic payload validation (non-blocking)."""
                async def _wrapper(event: SynapseEvent) -> None:
                    event.data = validate_event_payload(event, self._logger)
                    await handler(event)
                _wrapper.__name__ = getattr(handler, "__name__", "unknown")
                return _wrapper

            for event_type in _SUBSCRIBED_EVENTS:
                event_bus.subscribe(event_type, _validated(self._on_synapse_event))
            # General Equor rejection path: any intent blocked/deferred by Equor
            # (not just Thymos-originated repairs) notifies Thymos so it can
            # adjust drive priorities system-wide.
            event_bus.subscribe(
                SynapseEventType.INTENT_REJECTED,
                _validated(self._on_intent_rejected),
            )
            # Subscribe to Soma interoceptive percepts for proactive immune response
            event_bus.subscribe(
                SynapseEventType.INTEROCEPTIVE_PERCEPT,
                _validated(self._on_interoceptive_percept),
            )
            # Subscribe to Telos growth stagnation alerts - when dI/dt drops
            # below minimum, Thymos creates an immune incident to prompt
            # corrective action (frontier exploration, consolidation).
            event_bus.subscribe(
                SynapseEventType.GROWTH_STAGNATION,
                _validated(self._on_growth_stagnation),
            )
            # Autonomy gate: Nova emits this when Equor blocks an economic
            # survival goal because the current autonomy level is too low.
            # Thymos is the governed channel - it forwards to Tate via Telegram
            # so a human can grant temporary elevated permissions.
            event_bus.subscribe(
                SynapseEventType.AUTONOMY_INSUFFICIENT,
                _validated(self._on_autonomy_insufficient),
            )
            # Economic stress: Oikos emits METABOLIC_PRESSURE (with
            # economic_stress=True) when runway < 30 days.  Forward to Tate
            # so the operator is aware before the hard escalation fires.
            event_bus.subscribe(
                SynapseEventType.METABOLIC_PRESSURE,
                _validated(self._on_metabolic_pressure),
            )
            # Background task death - always reaches Tate (no dedup).
            event_bus.subscribe(
                SynapseEventType.TASK_PERMANENTLY_FAILED,
                _validated(self._on_task_permanently_failed),
            )
            # Bounty PR submission - notify Tate of potential revenue.
            event_bus.subscribe(
                SynapseEventType.BOUNTY_PR_SUBMITTED,
                _validated(self._on_bounty_pr_submitted),
            )
            # Simula accuracy regression - notify Tate when self-healing weakens.
            event_bus.subscribe(
                SynapseEventType.SIMULA_CALIBRATION_DEGRADED,
                _validated(self._on_simula_calibration_degraded),
            )

            # ── Sentinel-specific event subscriptions ──────────────────────
            # These feed live Synapse events into the specialized sentinel
            # detection logic rather than the generic _on_synapse_event path.

            # FeedbackLoopSentinel: detect repair→re-break cycles
            event_bus.subscribe(
                SynapseEventType.REPAIR_COMPLETED,
                _validated(self._on_repair_completed_for_feedback),
            )
            # CognitiveStallSentinel: Evo consolidation stalls
            event_bus.subscribe(
                SynapseEventType.EVO_CONSOLIDATION_STALLED,
                _validated(self._on_evo_consolidation_stalled),
            )
            # EVO-ECON-1: Evo hypothesis budget exhausted → DEGRADATION incident
            if hasattr(SynapseEventType, "EVO_DEGRADED"):
                event_bus.subscribe(
                    SynapseEventType.EVO_DEGRADED,
                    _validated(self._on_evo_degraded),
                )
            # CognitiveStallSentinel: Nova inference quality drops
            event_bus.subscribe(
                SynapseEventType.NOVA_DEGRADED,
                _validated(self._on_nova_degraded),
            )
            # DriftSentinel: Fovea precision degradation
            event_bus.subscribe(
                SynapseEventType.FOVEA_INTERNAL_PREDICTION_ERROR,
                _validated(self._on_fovea_internal_prediction_error),
            )
            # DriftSentinel: Soma state spikes (metabolic anomaly)
            event_bus.subscribe(
                SynapseEventType.SOMA_STATE_SPIKE,
                _validated(self._on_soma_state_spike),
            )
            # ProtocolHealthSentinel: Skia heartbeat failures
            event_bus.subscribe(
                SynapseEventType.SKIA_HEARTBEAT_LOST,
                _validated(self._on_skia_heartbeat_lost),
            )
            # ThreatPatternSentinel: economic threats (specialized path)
            event_bus.subscribe(
                SynapseEventType.THREAT_DETECTED,
                _validated(self._on_threat_detected),
            )
            # ContractSentinel: epistemic trust breach via Nexus speciation
            event_bus.subscribe(
                SynapseEventType.SPECIATION_EVENT,
                _validated(self._on_speciation_event),
            )

            # ── Closure Loop 1: Equor constitutional drift → immune response ──
            event_bus.subscribe(
                SynapseEventType.CONSTITUTIONAL_DRIFT_DETECTED,
                _validated(self._on_constitutional_drift),
            )

            # ── INV-017: Drive extinction → CRITICAL Tier 5 incident ──────────
            # When any constitutional drive's 72h mean drops below 0.01, the
            # organism has lost a dimension of its value geometry. No autonomous
            # repair is possible - only governance/federation review can restore
            # a drive from extinction. Thymos opens a Tier 5 (ESCALATE) incident
            # and MUST NOT attempt any autonomous fix.
            event_bus.subscribe(
                SynapseEventType.DRIVE_EXTINCTION_DETECTED,
                _validated(self._on_drive_extinction),
            )

            # ── Feedback loop closures (Interconnectedness Audit) ─────────

            # Axon TransactionShield rejections → real-time incident channel
            event_bus.subscribe(
                SynapseEventType.AXON_SHIELD_REJECTION,
                _validated(self._on_axon_shield_rejection),
            )
            # Atune repair validation → confirms repair effectiveness
            event_bus.subscribe(
                SynapseEventType.ATUNE_REPAIR_VALIDATION,
                _validated(self._on_atune_repair_validation),
            )
            # Evo hypothesis quality → repair pattern generalisation signal
            event_bus.subscribe(
                SynapseEventType.EVO_HYPOTHESIS_QUALITY,
                _validated(self._on_evo_hypothesis_quality),
            )
            # Nova belief stabilisation → downstream cognitive stability
            event_bus.subscribe(
                SynapseEventType.NOVA_BELIEF_STABILISED,
                _validated(self._on_nova_belief_stabilised),
            )

            # ── Cross-system state caching (AV1 migration) ──────────────
            # Instead of calling self._soma.get_current_signal() etc. directly,
            # we subscribe to periodic broadcasts and cache their payloads.
            event_bus.subscribe(
                SynapseEventType.SOMATIC_MODULATION_SIGNAL,
                _validated(self._on_soma_modulation_cached),
            )
            event_bus.subscribe(
                SynapseEventType.ECONOMIC_STATE_UPDATED,
                _validated(self._on_oikos_state_cached),
            )
            event_bus.subscribe(
                SynapseEventType.METABOLIC_SNAPSHOT,
                _validated(self._on_oikos_state_cached),
            )

            # ── Federation antibody sync (SG1) ────────────────────────────
            event_bus.subscribe(
                SynapseEventType.FEDERATION_KNOWLEDGE_RECEIVED,
                _validated(self._on_federation_knowledge_received),
            )

            # ── Sandbox validation result (M1) ────────────────────────────
            event_bus.subscribe(
                SynapseEventType.SIMULA_SANDBOX_RESULT,
                _validated(self._on_sandbox_result),
            )

            # ── SG7: Kairos causal invariants → CausalAnalyzer graph cache ──
            event_bus.subscribe(
                SynapseEventType.KAIROS_INVARIANT_DISTILLED,
                _validated(self._on_kairos_invariant),
            )

            # ── SG8: Oneiros repair schemas → prophylactic fingerprint store ──
            # When Oneiros finishes a consolidation cycle, it may have distilled
            # new (:Procedure {thymos_repair: true}) nodes. We query Memory and
            # add their embeddings to the prophylactic scanner's fingerprint store
            # so the immune system learns from sleep-time repair consolidation.
            event_bus.subscribe(
                SynapseEventType.ONEIROS_CONSOLIDATION_COMPLETE,
                _validated(self._on_oneiros_consolidation),
            )

            # ── Axon execution lifecycle (Spec 06 decoupling) ────────────────
            # Thymos pre-scans execution requests for incident risk (e.g. risky
            # financial actions) and reacts to rollbacks which signal compound
            # failures. Replaces any direct import of IncidentReport from Axon.
            event_bus.subscribe(
                SynapseEventType.AXON_EXECUTION_REQUEST,
                _validated(self._on_axon_execution_request),
            )
            event_bus.subscribe(
                SynapseEventType.AXON_ROLLBACK_INITIATED,
                _validated(self._on_axon_rollback_initiated),
            )

            # ── Identity vault security events (Identity #8) ────────────────
            # Decrypt failures → MEDIUM security incident (bad key sync or tamper).
            # Rotation failures → CRITICAL security incident (mid-flight abort).
            # Rotation start/complete are informational; no incident, Thymos
            # receives them on the bus for future antibody correlation.
            event_bus.subscribe(
                SynapseEventType.VAULT_DECRYPT_FAILED,
                _validated(self._on_vault_decrypt_failed),
            )
            event_bus.subscribe(
                SynapseEventType.VAULT_KEY_ROTATION_FAILED,
                _validated(self._on_vault_key_rotation_failed),
            )

            # ── Oneiros threat scenarios → prophylactic antibody pre-arming ──
            # Oneiros ThreatSimulator emits ONEIROS_THREAT_SCENARIO during REM.
            # Thymos stores these as pre-loaded incident templates. For HIGH/CRITICAL
            # severity, the relevant repair strategy is pre-armed so it activates
            # faster when the real incident occurs.
            event_bus.subscribe(
                SynapseEventType.ONEIROS_THREAT_SCENARIO,
                _validated(self._on_oneiros_threat_scenario),
            )

            # ── Axon circuit-breaker state changes ───────────────────────────
            # When a circuit breaker opens (CLOSED→OPEN) the protected system is
            # degraded. Thymos raises a DEGRADATION incident immediately so the
            # immune pipeline can investigate and repair.  When the breaker closes
            # again (OPEN→CLOSED) the incident is auto-resolved - no repair needed
            # because the executor recovered on its own.
            if hasattr(SynapseEventType, "CIRCUIT_BREAKER_STATE_CHANGED"):
                event_bus.subscribe(
                    SynapseEventType.CIRCUIT_BREAKER_STATE_CHANGED,
                    _validated(self._on_circuit_breaker_state_changed),
                )

            # ── Axon executor failure windowing (AXF-1) ──────────────────────
            # ACTION_FAILED is emitted by Axon on every action execution failure.
            # A sliding 10-minute window per executor_id prevents incident spam:
            # 5 failures within the window raises a DEGRADATION incident.
            # Critical executors (send_email, federation_send) raise immediately.
            if hasattr(SynapseEventType, "ACTION_FAILED"):
                event_bus.subscribe(
                    SynapseEventType.ACTION_FAILED,
                    _validated(self._on_action_failed),
                )

            # ── Phantom Liquidity health events ───────────────────────────────
            # Economic oracle failures impact the organism's ability to make
            # informed yield decisions.  Each event maps to an incident class.
            event_bus.subscribe(
                SynapseEventType.PHANTOM_POOL_STALE,
                _validated(self._on_phantom_pool_stale),
            )
            event_bus.subscribe(
                SynapseEventType.PHANTOM_POSITION_CRITICAL,
                _validated(self._on_phantom_position_critical),
            )
            event_bus.subscribe(
                SynapseEventType.PHANTOM_RESOURCE_EXHAUSTED,
                _validated(self._on_phantom_resource_exhausted),
            )
            event_bus.subscribe(
                SynapseEventType.PHANTOM_IL_DETECTED,
                _validated(self._on_phantom_il_detected),
            )
            event_bus.subscribe(
                SynapseEventType.PHANTOM_FALLBACK_ACTIVATED,
                _validated(self._on_phantom_fallback_activated),
            )

            # ── RE Training lifecycle ─────────────────────────────────────────
            # Training failures arrest the organism's learning velocity.
            # Halts indicate a constitutional safety check tripped.
            event_bus.subscribe(
                SynapseEventType.RE_TRAINING_FAILED,
                _validated(self._on_re_training_failed),
            )
            event_bus.subscribe(
                SynapseEventType.RE_TRAINING_HALTED,
                _validated(self._on_re_training_halted),
            )

            # ── INV-017 formal proof violation ────────────────────────────────
            # DRIVE_EXTINCTION_DETECTED fires on 72h rolling mean.
            # INV_017_VIOLATED fires on formal proof - a stronger signal.
            event_bus.subscribe(
                SynapseEventType.INV_017_VIOLATED,
                _validated(self._on_inv017_violated),
            )

            # ── Identity crisis (Thread) ──────────────────────────────────────
            # Narrative fingerprint shift ≥ 0.50 → BEHAVIORAL_DRIFT incident.
            event_bus.subscribe(
                SynapseEventType.IDENTITY_CRISIS,
                _validated(self._on_identity_crisis),
            )

            # ── SACM compute denial ───────────────────────────────────────────
            # Blocked compute requests leave dependent systems starved.
            event_bus.subscribe(
                SynapseEventType.COMPUTE_REQUEST_DENIED,
                _validated(self._on_compute_request_denied),
            )

            # ── Simula HotDeployment: dependency installed ────────────────────
            # Auto-resolves ImportError incidents when their package is installed.
            event_bus.subscribe(
                SynapseEventType.DEPENDENCY_INSTALLED,
                _validated(self._on_dependency_installed),
            )

            # ── Fovea dishabituation → DriftSentinel re-sensitization ─────────
            event_bus.subscribe(
                SynapseEventType.FOVEA_DISHABITUATION,
                _validated(self._on_fovea_dishabituation),
            )

            # ── Identity provisioning failures → SECURITY incident ────────────
            event_bus.subscribe(
                SynapseEventType.ACCOUNT_PROVISIONING_FAILED,
                _validated(self._on_account_provisioning_failed),
            )

            # ── Affect state → DriftSentinel sensitivity modulation ───────────
            event_bus.subscribe(
                SynapseEventType.AFFECT_STATE_CHANGED,
                _validated(self._on_affect_state_changed),
            )

            # ── Simula vulnerability confirmation → CRITICAL SECURITY incident ─
            event_bus.subscribe(
                SynapseEventType.VULNERABILITY_CONFIRMED,
                _validated(self._on_vulnerability_confirmed),
            )

            # ── Gap 4: Cross-instance CrashPattern confidence sync ────────────
            # When another organism instance emits CRASH_PATTERN_RESOLVED or
            # CRASH_PATTERN_REINFORCED (via federation bus relay), update the
            # local Redis-backed pattern confidence so the organism benefits from
            # peer repair experience without needing a direct database sync.
            if hasattr(SynapseEventType, "CRASH_PATTERN_RESOLVED"):
                event_bus.subscribe(
                    SynapseEventType.CRASH_PATTERN_RESOLVED,
                    self._on_crash_pattern_resolved_bus,
                )
            if hasattr(SynapseEventType, "CRASH_PATTERN_REINFORCED"):
                event_bus.subscribe(
                    SynapseEventType.CRASH_PATTERN_REINFORCED,
                    self._on_crash_pattern_reinforced_bus,
                )

            # ── Simula PreventiveAudit: incident history query ────────────────
            if hasattr(SynapseEventType, "THYMOS_INCIDENT_QUERY"):
                event_bus.subscribe(
                    SynapseEventType.THYMOS_INCIDENT_QUERY,
                    self._on_incident_query,
                )

        # ── Start background loops (supervised - death becomes an incident) ──
        from utils.supervision import supervised_task

        _event_bus = (
            self._synapse._event_bus if self._synapse is not None else None
        )
        self._sentinel_task = supervised_task(
            self._sentinel_scan_loop(),
            name="thymos_sentinel_scan",
            restart=True,
            max_restarts=5,
            backoff_base=2.0,
            event_bus=_event_bus,
            source_system="thymos",
        )
        self._homeostasis_task = supervised_task(
            self._homeostasis_loop(),
            name="thymos_homeostasis",
            restart=True,
            max_restarts=5,
            backoff_base=2.0,
            event_bus=_event_bus,
            source_system="thymos",
        )
        self._telemetry_task = supervised_task(
            self._telemetry_emission_loop(),
            name="thymos_telemetry",
            restart=True,
            max_restarts=5,
            backoff_base=2.0,
            event_bus=_event_bus,
            source_system="thymos",
        )

        # ── Telegram command receiver ──
        # Only start if credentials are present; harmless no-op otherwise.
        if (
            os.environ.get("ECODIAOS_CONNECTORS__TELEGRAM__BOT_TOKEN")
            and os.environ.get("ECODIAOS_CONNECTORS__TELEGRAM__ADMIN_CHAT_ID")
        ):
            self._telegram_polling_task = supervised_task(
                self._telegram_poll_loop(),
                name="thymos_telegram_poll",
                restart=True,
                max_restarts=10,
                backoff_base=2.0,
                event_bus=_event_bus,
                source_system="thymos",
            )

        # ── Register with NeuroplasticityBus for sentinel hot-reload ──
        if self._neuroplasticity_bus is not None:
            self._neuroplasticity_bus.register(
                base_class=BaseThymosSentinel,
                registration_callback=self._on_sentinel_evolved,
                system_id="thymos",
            )

        # ── Global asyncio exception handler ──
        # Captures unhandled exceptions from fire-and-forget tasks and
        # background loops that would otherwise silently die. Converts them
        # into incidents so the immune system can see and heal them.
        loop = asyncio.get_running_loop()
        self._original_exception_handler = loop.get_exception_handler()

        def _asyncio_exception_handler(
            loop: asyncio.AbstractEventLoop,
            context: dict[str, Any],
        ) -> None:
            exc = context.get("exception")
            message = context.get("message", "")
            task_name = ""
            task = context.get("task")
            if task is not None:
                task_name = getattr(task, "get_name", lambda: "")()

            self._logger.error(
                "asyncio_unhandled_exception",
                message=message,
                task_name=task_name,
                error=str(exc) if exc else message,
            )

            # Convert to incident if we have an actual exception
            if exc is not None and self._exception_sentinel is not None:
                incident = self._exception_sentinel.intercept(
                    system_id=task_name.split("_")[0] if task_name else "asyncio",
                    method_name=task_name or "unhandled_task",
                    exception=exc if isinstance(exc, BaseException) else Exception(str(exc)),
                )
                asyncio.create_task(
                    self.on_incident(incident),
                    name="thymos_asyncio_exc",
                )

            # Chain to the original handler if present
            if self._original_exception_handler is not None:
                self._original_exception_handler(loop, context)

        loop.set_exception_handler(_asyncio_exception_handler)

        # ── GAP-6 / F1: Coma recovery - check for crash context left by previous
        # incarnation.  Runs as a deferred background task so it does not block
        # initialize() return; the Synapse bus is live by the time the task runs.
        asyncio.create_task(
            self._check_pre_resurrection_crash_context(),
            name="thymos_resurrection_crash_context",
        )

        self._initialized = True

        antibody_count = len(self._antibody_library._all) if self._antibody_library else 0
        self._logger.info(
            "thymos_initialized",
            antibodies_loaded=antibody_count,
            subscribed_events=len(_SUBSCRIBED_EVENTS),
        )

        # Send the one-time startup heartbeat to Telegram so the operator
        # knows the channel is live and how many systems are active.
        try:
            systems_wired: int = 0
            if self._synapse is not None:
                health_state = getattr(self._synapse, "_health_monitor", None)
                if health_state is not None:
                    registry = getattr(health_state, "_systems", {})
                    systems_wired = len(registry)
            if systems_wired == 0:
                # Fallback: count known non-None cross-system references as a proxy.
                refs = [
                    self._equor, self._evo, self._atune, self._nova,
                    self._soma, self._oikos, self._federation, self._telos,
                    self._simula,
                ]
                systems_wired = sum(1 for r in refs if r is not None) + 1  # +1 for thymos
            await self._notification_dispatcher.maybe_send_startup(systems_wired)
        except Exception:  # noqa: BLE001
            pass  # startup message is best-effort

    async def shutdown(self) -> None:
        """Graceful shutdown. Cancel background tasks and log final stats."""
        self._logger.info("thymos_shutting_down")

        # Cancel background tasks
        for task in (
            self._sentinel_task,
            self._homeostasis_task,
            self._telegram_polling_task,
        ):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        self._sentinel_task = None
        self._homeostasis_task = None
        self._telegram_polling_task = None

        # Deregister from NeuroplasticityBus so no callbacks fire after teardown
        if self._neuroplasticity_bus is not None:
            self._neuroplasticity_bus.deregister(BaseThymosSentinel)

        self._logger.info(
            "thymos_shutdown",
            total_incidents=self._total_incidents,
            total_repairs_attempted=self._total_repairs_attempted,
            total_repairs_succeeded=self._total_repairs_succeeded,
            active_incidents=len(self._active_incidents),
            antibodies_total=(
                len(self._antibody_library._all) if self._antibody_library else 0
            ),
        )

    # ─── Synapse Event Handler ───────────────────────────────────────

    async def _on_synapse_event(self, event: SynapseEvent) -> None:
        """
        Convert Synapse health events into Incidents.

        This is how Thymos learns about system failures without direct
        coupling to every system. Synapse watches health; Thymos watches
        Synapse events.
        """
        severity, incident_class = self._classify_synapse_event(event)

        if severity is None:
            # Recovery events - resolve any matching active incidents
            if event.event_type in (
                SynapseEventType.SYSTEM_RECOVERED,
                SynapseEventType.SAFE_MODE_EXITED,
            ):
                await self._handle_recovery_event(event)
            return

        source_system = event.data.get("system_id", event.source_system)

        fp = hashlib.sha256(
            f"{source_system}:{event.event_type.value}".encode()
        ).hexdigest()[:16]

        incident = Incident(
            incident_class=incident_class,
            severity=severity,
            fingerprint=fp,
            source_system=source_system,
            error_type=event.event_type.value,
            error_message=(
                f"Synapse health event: {event.event_type.value} "
                f"for {source_system}"
            ),
            context=event.data,
        )

        await self.on_incident(incident)

        # P8 - Version-rollback guard: MODEL_HOT_SWAP_FAILED → immediately request
        # rollback to last-known-good model version via Synapse.  This runs
        # unconditionally alongside the incident so the swap failure is not just
        # logged but actively reversed.
        if event.event_type == SynapseEventType.MODEL_HOT_SWAP_FAILED:
            await self._request_model_version_rollback(event.data)

    def _classify_synapse_event(
        self,
        event: SynapseEvent,
    ) -> tuple[IncidentSeverity | None, IncidentClass]:
        """Map a Synapse event type to incident severity and class."""
        mapping: dict[SynapseEventType, tuple[IncidentSeverity | None, IncidentClass]] = {
            SynapseEventType.SYSTEM_FAILED: (
                IncidentSeverity.CRITICAL,
                IncidentClass.CRASH,
            ),
            SynapseEventType.SYSTEM_RESTARTING: (
                IncidentSeverity.HIGH,
                IncidentClass.CRASH,
            ),
            SynapseEventType.SYSTEM_OVERLOADED: (
                IncidentSeverity.MEDIUM,
                IncidentClass.DEGRADATION,
            ),
            SynapseEventType.SAFE_MODE_ENTERED: (
                IncidentSeverity.CRITICAL,
                IncidentClass.CRASH,
            ),
            SynapseEventType.CLOCK_OVERRUN: (
                IncidentSeverity.MEDIUM,
                IncidentClass.DEGRADATION,
            ),
            SynapseEventType.RESOURCE_PRESSURE: (
                IncidentSeverity.MEDIUM,
                IncidentClass.RESOURCE_EXHAUSTION,
            ),
            # Recovery events - no incident created
            SynapseEventType.SYSTEM_RECOVERED: (None, IncidentClass.CRASH),
            SynapseEventType.SAFE_MODE_EXITED: (None, IncidentClass.CRASH),
            # Certificate lifecycle (Phase 16g: Civilization Layer)
            SynapseEventType.CERTIFICATE_EXPIRED: (
                IncidentSeverity.CRITICAL,
                IncidentClass.CONTRACT_VIOLATION,
            ),
            SynapseEventType.CERTIFICATE_EXPIRING: (
                IncidentSeverity.HIGH,
                IncidentClass.CONTRACT_VIOLATION,
            ),
            # Connector health (Phase 16h)
            SynapseEventType.SYSTEM_DEGRADED: (
                IncidentSeverity.HIGH,
                IncidentClass.DEGRADATION,
            ),
            SynapseEventType.CONNECTOR_ERROR: (
                IncidentSeverity.MEDIUM,
                IncidentClass.CRASH,
            ),
            SynapseEventType.CONNECTOR_TOKEN_EXPIRED: (
                IncidentSeverity.HIGH,
                IncidentClass.CONTRACT_VIOLATION,
            ),
            # Model lifecycle failures
            SynapseEventType.MODEL_HOT_SWAP_FAILED: (
                IncidentSeverity.CRITICAL,
                IncidentClass.CRASH,
            ),
            SynapseEventType.CATASTROPHIC_FORGETTING_DETECTED: (
                IncidentSeverity.CRITICAL,
                IncidentClass.DRIFT,
            ),
            SynapseEventType.MODEL_ROLLBACK_TRIGGERED: (
                IncidentSeverity.HIGH,
                IncidentClass.CRASH,
            ),
            # Cross-system health bridges
            SynapseEventType.NOVA_DEGRADED: (
                IncidentSeverity.HIGH,
                IncidentClass.DEGRADATION,
            ),
            SynapseEventType.EVO_CONSOLIDATION_STALLED: (
                IncidentSeverity.MEDIUM,
                IncidentClass.COGNITIVE_STALL,
            ),
            SynapseEventType.SKIA_HEARTBEAT_LOST: (
                IncidentSeverity.CRITICAL,
                IncidentClass.CRASH,
            ),
        }
        return mapping.get(
            event.event_type,
            (IncidentSeverity.LOW, IncidentClass.DEGRADATION),
        )

    async def _handle_recovery_event(self, event: SynapseEvent) -> None:
        """Resolve active incidents for a recovered system."""
        source_system = event.data.get("system_id", event.source_system)

        resolved_ids: list[str] = []
        for incident_id, incident in list(self._active_incidents.items()):
            if incident.source_system == source_system:
                incident.repair_status = RepairStatus.RESOLVED
                incident.repair_successful = True
                now = utc_now()
                incident.resolution_time_ms = int(
                    (now - incident.timestamp).total_seconds() * 1000
                )
                self._resolution_times.append(float(incident.resolution_time_ms))
                resolved_ids.append(incident_id)

                if self._governor is not None:
                    self._governor.resolve_incident(incident_id)

        for incident_id in resolved_ids:
            self._active_incidents.pop(incident_id, None)

        if resolved_ids:
            self._logger.info(
                "recovery_event_resolved_incidents",
                source_system=source_system,
                resolved_count=len(resolved_ids),
            )

    async def _request_model_version_rollback(self, event_data: dict) -> None:
        """
        P8 - Version-rollback guard: emit MODEL_ROLLBACK_TRIGGERED after a
        MODEL_HOT_SWAP_FAILED event so the organism automatically reverts to the
        last-known-good model version.

        Thymos does not track model checkpoints directly - it emits the rollback
        request and lets Simula / RE orchestrate the actual version swap.
        The failed swap context is forwarded so the RE can mark the version as
        unsafe and avoid re-trying it.
        """
        failed_version = event_data.get("failed_version", "unknown")
        target_system = event_data.get("system_id", event_data.get("system", "re"))
        reason = event_data.get("reason", "hot-swap failed")

        self._logger.warning(
            "model_rollback_requested",
            failed_version=failed_version,
            target_system=target_system,
            reason=reason[:200],
        )

        await self._emit_event(
            SynapseEventType.MODEL_ROLLBACK_TRIGGERED,
            {
                "target_system": target_system,
                "failed_version": failed_version,
                "reason": f"Thymos version-rollback guard: {reason[:300]}",
                "initiated_by": "thymos",
                "auto_rollback": True,
            },
        )
        self._emit_metric("thymos.model_rollback_requested", 1)

    # ─── Main Entry Point ────────────────────────────────────────────

    async def on_incident(self, incident: Incident) -> None:
        """
        Primary entry point for the immune pipeline.

        Called by sentinels (directly) and Synapse event handler.
        Deduplicates, then routes to the full processing pipeline.

        This method NEVER raises - immune failures must not cascade.
        """
        try:
            await self._on_incident_inner(incident)
        except Exception as exc:
            # Thymos must not crash. Log and continue.
            self._logger.error(
                "thymos_internal_error",
                error=str(exc),
                incident_id=incident.id,
                incident_source=incident.source_system,
            )

    async def _on_incident_inner(self, incident: Incident) -> None:
        """Dedup → score → buffer → route → process."""
        assert self._deduplicator is not None
        assert self._severity_scorer is not None
        assert self._response_router is not None
        assert self._governor is not None

        # Step 1: Deduplicate
        dedup_result = self._deduplicator.deduplicate(incident)
        if dedup_result is None:
            self._logger.debug(
                "incident_deduplicated",
                fingerprint=incident.fingerprint,
                count=incident.occurrence_count,
            )
            return

        # Step 1a: Handle T4 re-emissions from the deduplicator.
        # When a duplicate pushes occurrence_count past the T4 recurrence
        # threshold, the deduplicator returns the *existing* incident so we
        # can re-route it.  Skip scoring/tracking (already done on first
        # arrival) and jump straight to routing for T4 escalation.
        is_reemission = dedup_result.id in self._active_incidents
        if is_reemission:
            initial_tier = self._response_router.route(dedup_result)
            dedup_result.repair_tier = initial_tier

            if initial_tier == RepairTier.NOOP:
                self._logger.warning(
                    "reemitted_incident_still_noop",
                    fingerprint=dedup_result.fingerprint,
                    occurrences=dedup_result.occurrence_count,
                )
                return

            # ── Gap 6: Pattern-aware override on re-emission path ──
            # Re-emitted incidents may match a CrashPattern whose failed_tiers
            # include the tier that _response_router just assigned.  Consult the
            # router so we skip proven-useless tiers (or escalate to federation).
            # _on_incident_inner is an async def so awaiting here is safe.
            if self._pattern_router is not None:
                try:
                    pattern_result = await self._pattern_router.route(dedup_result)
                    if pattern_result.matched:
                        dedup_result.matched_pattern_id = pattern_result.pattern_id
                        dedup_result.pattern_confidence = pattern_result.pattern_confidence
                        dedup_result.tier_skip_reason = pattern_result.tier_skip_reason
                        if pattern_result.federation_escalate:
                            await self._emit_event(SynapseEventType.INCIDENT_ESCALATED, {
                                "incident_id": dedup_result.id,
                                "incident_class": dedup_result.incident_class.value,
                                "severity": dedup_result.severity.value,
                                "source_system": dedup_result.source_system,
                                "reason": pattern_result.tier_skip_reason,
                                "federation_broadcast": True,
                                "pattern_id": pattern_result.pattern_id,
                            })
                            dedup_result.repair_status = RepairStatus.ESCALATED
                            self._active_incidents.pop(dedup_result.id, None)
                            return
                        elif pattern_result.tier_override is not None:
                            dedup_result.repair_tier = pattern_result.tier_override
                            initial_tier = pattern_result.tier_override
                except Exception as exc:
                    self._logger.debug(
                        "pattern_router_reemission_error",
                        incident_id=dedup_result.id,
                        error=str(exc),
                    )

            self._logger.info(
                "reemitted_incident_escalated",
                incident_id=dedup_result.id,
                fingerprint=dedup_result.fingerprint,
                occurrences=dedup_result.occurrence_count,
                new_tier=initial_tier.name,
            )
            self._emit_metric(
                "thymos.incidents.t4_escalation_from_noop", 1,
                tags={"fingerprint": dedup_result.fingerprint[:16]},
            )
            self._supervised_fire_and_forget(
                self._process_incident_safe(dedup_result),
                name=f"thymos_t4_reemit_{dedup_result.id[:8]}",
            )
            return

        # Step 1b: Populate constitutional_impact from Telos (authoritative drive state).
        # AV1 migration: prefer cached drive impact from Telos events, fall back to
        # direct call during incremental migration.
        impact_key = f"{incident.incident_class.value}:{incident.source_system}"
        cached_impact = self._cached_telos_drive_impact.get(impact_key)
        if cached_impact is not None:
            incident.constitutional_impact = cached_impact
        elif self._telos is not None:
            try:
                incident.constitutional_impact = self._telos.predict_drive_impact(
                    incident_class=incident.incident_class.value,
                    source_system=incident.source_system,
                )
                # Cache for future lookups
                self._cached_telos_drive_impact[impact_key] = incident.constitutional_impact
            except Exception as exc:
                self._logger.debug("telos_drive_impact_error", error=str(exc))

        # Step 2: Score severity (composite)
        scored_severity = self._severity_scorer.compute_severity(incident)
        incident.severity = scored_severity

        # Step 3: Track
        self._total_incidents += 1
        self._incident_buffer.append(incident)
        self._active_incidents[incident.id] = incident
        self._incidents_by_severity[scored_severity.value] = (
            self._incidents_by_severity.get(scored_severity.value, 0) + 1
        )
        self._incidents_by_class[incident.incident_class.value] = (
            self._incidents_by_class.get(incident.incident_class.value, 0) + 1
        )

        # Register with governor for storm detection
        self._governor.register_incident(incident)

        # Push to SSE stream queues (non-blocking; drop if queue full)
        for q in self._stream_queues:
            with contextlib.suppress(asyncio.QueueFull):
                q.put_nowait(incident)

        # Absorb constitutional impact into the drive state accumulator.
        # This lets us track which drives are under pressure from the pattern
        # of incidents, independently of what Equor says about each repair.
        self._drive_state.absorb_incident(incident)

        # Record in temporal correlator
        if self._temporal_correlator is not None:
            self._temporal_correlator.record_event(
                event_type="incident",
                details=f"{incident.incident_class.value}: {incident.error_message}",
                system_id=incident.source_system,
            )

        # Record in causal analyzer
        if self._causal_analyzer is not None:
            self._causal_analyzer.record_incident(incident)

        # Emit telemetry
        inc_class = incident.incident_class.value
        self._emit_metric("thymos.incidents.created", 1, tags={"class": inc_class})
        self._emit_metric("thymos.incidents.severity", 1, tags={"severity": scored_severity.value})

        self._logger.info(
            "incident_created",
            incident_id=incident.id,
            source_system=incident.source_system,
            incident_class=incident.incident_class.value,
            severity=scored_severity.value,
            fingerprint=incident.fingerprint[:16],
        )

        # Step 4: Make the organism feel it - route to Atune as a Percept
        await self._broadcast_as_percept(incident)

        # Step 4a: CRITICAL incidents interrupt sleep (EMERGENCY_WAKE).
        # If Oneiros is in a sleep cycle and a CRITICAL incident fires,
        # the organism must wake immediately - sleep cannot be prioritised
        # over a potentially organism-threatening failure.
        if scored_severity == IncidentSeverity.CRITICAL:
            await self._emit_event(
                SynapseEventType.EMERGENCY_WAKE,
                {
                    "incident_id": incident.id,
                    "incident_class": incident.incident_class.value,
                    "source_system": incident.source_system,
                    "reason": f"CRITICAL incident: {incident.error_type}",
                    "triggered_by": "thymos",
                },
            )

        # Step 5: Route to initial repair tier
        initial_tier = self._response_router.route(incident)
        incident.repair_tier = initial_tier

        # ── RE training: anomaly detection decision ──
        # care alignment: incidents harm wellbeing; severity inversely proportional to care
        _sev_map = {"critical": -0.8, "high": -0.5, "medium": -0.2, "low": 0.1, "info": 0.3}
        _care_score = _sev_map.get(scored_severity.value, 0.0)
        _coh_score = -0.3 if scored_severity.value in ("critical", "high") else 0.1
        asyncio.ensure_future(self._emit_re_training_example(
            category="anomaly_detection",
            instruction="Detect, deduplicate, score, and route an incident through the immune pipeline.",
            input_context=f"source={incident.source_system}, class={incident.incident_class.value}, fingerprint={incident.fingerprint[:16]}",
            output=f"severity={scored_severity.value}, tier={initial_tier.name}",
            outcome_quality={"critical": 0.2, "high": 0.4, "medium": 0.6, "low": 0.8, "info": 0.9}.get(scored_severity.value, 0.5),
            episode_id=incident.id,
            constitutional_alignment=self._build_incident_alignment(care=_care_score, coherence=_coh_score),
        ))

        # Step 6: Process through the immune pipeline
        if initial_tier == RepairTier.NOOP:
            incident.repair_status = RepairStatus.ACCEPTED
            self._active_incidents.pop(incident.id, None)
            if self._governor is not None:
                self._governor.resolve_incident(incident.id)
            return

        # Process in a background task so we don't block the event bus callback
        self._supervised_fire_and_forget(
            self._process_incident_safe(incident),
            name=f"thymos_process_{incident.id[:8]}",
        )

    # ─── Immune Pipeline ─────────────────────────────────────────────

    async def _process_incident_safe(self, incident: Incident) -> None:
        """
        Wrapper that catches errors during incident processing.

        If ``process_incident`` raises unexpectedly the incident must not
        silently rot in PENDING.  We:
          1. Log the full exception with incident context.
          2. Attempt a last-resort Tier 5 escalation via the dispatcher.
          3. Mark the incident ESCALATED and remove it from active tracking.
        """
        try:
            await self.process_incident(incident)
        except Exception as exc:
            import traceback

            self._logger.error(
                "incident_processing_failed",
                incident_id=incident.id,
                source_system=incident.source_system,
                incident_class=incident.incident_class.value,
                severity=incident.severity.value,
                error=str(exc),
                traceback=traceback.format_exc()[:1000],
            )
            incident.repair_status = RepairStatus.ESCALATED

            # Last-resort escalation: notify a human so the incident is never lost.
            try:
                await self._notification_dispatcher.dispatch(
                    incident_id=incident.id,
                    severity=incident.severity.value,
                    system=incident.source_system,
                    what_was_tried=[
                        f"process_incident raised an unhandled exception: {exc!s:.200}"
                    ],
                    what_failed=f"Immune pipeline crashed: {exc!s:.300}",
                    recommended_human_action=(
                        f"The Thymos immune pipeline failed while processing incident "
                        f"{incident.id} ({incident.incident_class.value}) in "
                        f"{incident.source_system}. Manual investigation required."
                    ),
                )
            except Exception as dispatch_exc:
                self._logger.error(
                    "last_resort_escalation_failed",
                    incident_id=incident.id,
                    error=str(dispatch_exc),
                )

            self._active_incidents.pop(incident.id, None)
            if self._governor is not None:
                self._governor.resolve_incident(incident.id)

    def _supervised_fire_and_forget(
        self,
        coro: Coroutine[Any, Any, Any],
        name: str,
    ) -> asyncio.Task[Any]:
        """
        Create a fire-and-forget task with a done callback that turns
        silent death into a new incident.

        Use this instead of bare ``asyncio.create_task`` for any task
        whose silent failure should be visible to the immune system.
        """
        task = asyncio.create_task(coro, name=name)

        def _on_done(t: asyncio.Task[Any]) -> None:
            if t.cancelled():
                self._logger.warning(
                    "fire_and_forget_task_cancelled", task_name=name,
                )
                return
            exc = t.exception()
            if exc is not None:
                self._logger.error(
                    "fire_and_forget_task_died",
                    task_name=name,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                # Create a new incident for the dead task
                import hashlib
                dead_task_incident = Incident(
                    source_system="thymos",
                    incident_class=IncidentClass.CRASH,
                    severity=IncidentSeverity.HIGH,
                    error_type=type(exc).__name__,
                    error_message=f"Supervised task '{name}' died: {exc!s:.200}",
                    fingerprint=hashlib.sha256(
                        f"dead_task:{name}".encode()
                    ).hexdigest()[:16],
                )
                asyncio.create_task(
                    self.on_incident(dead_task_incident),
                    name=f"thymos_dead_task_{name[:20]}",
                )

        task.add_done_callback(_on_done)
        return task

    async def process_incident(self, incident: Incident) -> None:
        """
        Full immune pipeline: Diagnose → Prescribe → Validate → Apply → Verify → Learn.

        This is the core of Thymos. Each step has clear entry/exit criteria
        and failure modes that escalate to the next tier.
        """
        assert self._governor is not None
        assert self._antibody_library is not None
        assert self._diagnostic_engine is not None
        assert self._causal_analyzer is not None
        assert self._temporal_correlator is not None
        assert self._prescriber is not None
        assert self._validator is not None

        start_time = time.monotonic()

        # ── Metabolic gate: triage by starvation level ──
        # EMERGENCY: only heal severity=critical incidents
        # CRITICAL: log only - no LLM diagnosis
        if self._starvation_level == "critical":
            self._logger.warning(
                "incident_skipped_critical_starvation",
                incident_id=incident.id,
                severity=incident.severity.value,
            )
            incident.repair_status = RepairStatus.ESCALATED
            self._active_incidents.pop(incident.id, None)
            return
        if self._starvation_level == "emergency":
            if incident.severity not in (IncidentSeverity.CRITICAL,):
                self._logger.info(
                    "incident_deferred_emergency_starvation",
                    incident_id=incident.id,
                    severity=incident.severity.value,
                )
                incident.repair_status = RepairStatus.ESCALATED
                self._active_incidents.pop(incident.id, None)
                return

        # ── Step 1: Check governor budget ──
        if not self._governor.should_diagnose(incident):
            self._logger.info(
                "diagnosis_throttled",
                incident_id=incident.id,
                healing_mode=self._governor.healing_mode.value,
            )
            incident.repair_status = RepairStatus.ESCALATED
            self._active_incidents.pop(incident.id, None)
            return

        # ── Step 2: Diagnose ──
        incident.repair_status = RepairStatus.DIAGNOSING
        self._governor.begin_diagnosis()
        self._total_diagnoses += 1

        try:
            diagnosis = await self._diagnose(incident)
        finally:
            self._governor.end_diagnosis()

        diagnosis_ms = (time.monotonic() - start_time) * 1000
        self._diagnosis_latencies.append(diagnosis_ms)
        self._diagnosis_confidences.append(diagnosis.confidence)

        incident.root_cause_hypothesis = diagnosis.root_cause
        incident.diagnostic_confidence = diagnosis.confidence

        # Preserve router's T4 (NOVEL_FIX) escalation - the diagnosis engine
        # doesn't know about recurrence-based escalation and would downgrade
        # back to PARAMETER, defeating the masking-loop detection.
        t4_forced_by_router = (
            incident.repair_tier == RepairTier.NOVEL_FIX
            and self._response_router is not None
            and incident.fingerprint in self._response_router._t4_escalated_at
        )
        if diagnosis.repair_tier is not None and not t4_forced_by_router:
            incident.repair_tier = diagnosis.repair_tier
        elif t4_forced_by_router and diagnosis.repair_tier is not None:
            # Keep NOVEL_FIX on incident but store diagnosis tier for logging
            diagnosis = diagnosis.model_copy(update={"repair_tier": RepairTier.NOVEL_FIX})

        self._emit_metric("thymos.diagnosis.confidence", diagnosis.confidence)
        self._emit_metric("thymos.diagnosis.latency_ms", diagnosis_ms)

        self._logger.info(
            "diagnosis_complete",
            incident_id=incident.id,
            root_cause=diagnosis.root_cause[:80],
            confidence=f"{diagnosis.confidence:.2f}",
            repair_tier=diagnosis.repair_tier.name if diagnosis.repair_tier else "unknown",
            latency_ms=f"{diagnosis_ms:.0f}",
        )

        # ── RE training: diagnostic decision ──
        asyncio.ensure_future(self._emit_re_training_example(
            category="diagnostic",
            instruction="Diagnose an incident: trace root cause, correlate temporal events, generate hypotheses.",
            input_context=f"incident_class={incident.incident_class.value}, severity={incident.severity.value}, source={incident.source_system}",
            output=f"root_cause={diagnosis.root_cause[:200]}, confidence={diagnosis.confidence:.2f}, tier={diagnosis.repair_tier.name if diagnosis.repair_tier else 'unknown'}",
            outcome_quality=diagnosis.confidence,
            latency_ms=int(diagnosis_ms),
            reasoning_trace=diagnosis.reasoning[:200] if hasattr(diagnosis, "reasoning") and diagnosis.reasoning else "",
            alternatives_considered=[h.statement[:100] for h in diagnosis.all_hypotheses[:5]] if hasattr(diagnosis, "all_hypotheses") else [],
            episode_id=incident.id,
            constitutional_alignment=self._build_incident_alignment(
                coherence=diagnosis.confidence * 2.0 - 1.0,  # high confidence = coherent diagnosis
                honesty=diagnosis.confidence - 0.5,           # confident diagnosis = honest about reality
            ),
        ))

        # ── Step 2b: Inject urgent goal for critical incidents ──
        # AV1: goal injection now uses Synapse emit; guard on synapse availability
        if incident.severity == IncidentSeverity.CRITICAL and self._synapse is not None:
            await self._inject_repair_goal(incident, diagnosis.repair_tier, resolved=False)

        # ── Step 2c: Integrity precision gating (Soma - cached via Synapse) ──
        # AV1 migration: reads from cached Soma modulation signal instead of
        # calling self._soma.get_current_signal() directly.
        if self._cached_soma_signal:
            precision_weights = self._cached_soma_signal.get("precision_weights", {})
            integrity_precision = float(precision_weights.get("integrity", 1.0))
            if integrity_precision > 0.7:
                original_confidence = diagnosis.confidence
                boosted_confidence = min(1.0, original_confidence * 1.15)
                diagnosis = diagnosis.model_copy(update={"confidence": boosted_confidence})
                self._logger.debug(
                    "integrity_precision_gating_applied",
                    integrity_precision=round(integrity_precision, 3),
                    confidence_before=round(original_confidence, 3),
                    confidence_after=round(boosted_confidence, 3),
                )

        # ── Step 2c½: Soma coherence → Coherence drive pressure ──
        # AV1 migration: reads from cached coherence/vulnerability instead of
        # calling self._soma.coherence_signal / self._soma.vulnerability_map().
        if self._cached_soma_signal:
            try:
                coh = self._cached_soma_coherence
                coherence_pressure = max(0.0, 1.0 - coh)
                if coherence_pressure > 0.1:
                    self._drive_state.coherence = min(
                        1.0,
                        self._drive_state.coherence + coherence_pressure * 0.05,
                    )

                # Suppress Growth drive when Soma reports dynamical instability
                vuln = self._cached_soma_vulnerability
                chaotic = vuln.get("chaotic_metrics", [])
                topo_breaches = vuln.get("topological_breaches", 0)
                if chaotic or topo_breaches > 0:
                    self._drive_state.growth = max(
                        0.0,
                        self._drive_state.growth * 0.7,
                    )
                    self._logger.debug(
                        "growth_drive_suppressed_instability",
                        chaotic_count=len(chaotic),
                        topo_breaches=topo_breaches,
                    )
            except Exception as exc:
                self._logger.debug("soma_coherence_gating_error", error=str(exc))

        # ── Step 2d: Drive pressure gating ──
        # When accumulated drive stress is high (repeated Equor rejections or a
        # flood of high-impact incidents), Thymos conservatively caps the repair
        # tier it's willing to attempt autonomously.  The goal is to avoid
        # generating another repair that Equor will block on the same drive axis.
        #
        # Thresholds (drive stress 0.0–1.0):
        #   > 0.7  → cap at RESTART (Tier 2). Novel / codegen repairs are too risky.
        #   > 0.5  → cap at KNOWN_FIX (Tier 3). Don't attempt codegen.
        #
        # The cap does NOT apply to ESCALATE (Tier 5) - human escalation is always
        # permitted regardless of drive stress.
        self._drive_state.apply_decay()
        composite_stress = self._drive_state.composite_stress
        # Recurrence-forced T4 incidents are immune to stress demotion.
        # A recurring incident under high stress is MORE urgent, not less - demoting
        # it would undo the masking-loop detection that already escalated it.
        _recurrence_forced_t4 = (
            diagnosis.repair_tier == RepairTier.NOVEL_FIX
            and self._response_router is not None
            and incident.fingerprint in self._response_router._t4_escalated_at
        )
        if (
            composite_stress > 0.5
            and diagnosis.repair_tier is not None
            and not _recurrence_forced_t4
        ):
            tier_cap = RepairTier.RESTART if composite_stress > 0.7 else RepairTier.KNOWN_FIX
            if diagnosis.repair_tier.value > tier_cap.value:
                original_diagnosis_tier = diagnosis.repair_tier
                capped_tier = tier_cap
                diagnosis = diagnosis.model_copy(update={"repair_tier": capped_tier})
                self._logger.info(
                    "drive_pressure_tier_cap_applied",
                    incident_id=incident.id,
                    composite_stress=round(composite_stress, 3),
                    most_stressed=self._drive_state.most_stressed_drive,
                    original_tier=original_diagnosis_tier.name,
                    capped_to=capped_tier.name,
                )

        # ── Step 2e: History-driven tier escalation ──
        # If previous attempts failed at certain tiers, skip them and go higher.
        highest_failed = incident.highest_attempted_tier()
        if highest_failed is not None and diagnosis.repair_tier is not None:
            if diagnosis.repair_tier.value <= highest_failed.value:
                escalated_tier = incident.next_escalation_tier(highest_failed)
                self._logger.info(
                    "tier_escalated_by_attempt_history",
                    incident_id=incident.id,
                    previous_highest_tier=highest_failed.name,
                    diagnosis_tier=diagnosis.repair_tier.name,
                    escalated_to=escalated_tier.name,
                    attempt_count=len(incident.repair_history),
                )
                diagnosis = diagnosis.model_copy(update={"repair_tier": escalated_tier})

        # ── Step 2f: Pattern-aware tier override ──
        # Query CrashPattern library (Redis) to check whether this incident
        # matches a known fatal signature.  If it does, skip tiers that have
        # already failed for that pattern and route directly to a higher tier.
        # If all local tiers are exhausted, skip repair entirely and escalate
        # to federation so a peer with a fresh environment can attempt it.
        if self._pattern_router is not None and diagnosis.repair_tier is not None:
            pattern_result = await self._pattern_router.route(incident)

            if pattern_result.matched:
                # Stamp pattern metadata onto the incident for Neo4j / RE training
                incident.matched_pattern_id = pattern_result.pattern_id
                incident.pattern_confidence = pattern_result.pattern_confidence
                incident.tier_skip_reason = pattern_result.tier_skip_reason

                if pattern_result.federation_escalate:
                    # All local tiers exhausted - broadcast to federation immediately
                    self._logger.warning(
                        "pattern_all_tiers_exhausted_federation_broadcast",
                        incident_id=incident.id,
                        pattern_id=pattern_result.pattern_id,
                        skipped_tiers=pattern_result.skipped_tiers,
                    )
                    await self._emit_event(
                        SynapseEventType.INCIDENT_ESCALATED,
                        {
                            "incident_id": incident.id,
                            "incident_class": incident.incident_class.value,
                            "from_tier": diagnosis.repair_tier.name,
                            "to_tier": "FEDERATION",
                            "reason": incident.tier_skip_reason,
                            "federation_broadcast": True,
                            "pattern_id": pattern_result.pattern_id,
                        },
                    )
                    incident.repair_status = RepairStatus.ESCALATED
                    self._active_incidents.pop(incident.id, None)
                    if self._deduplicator is not None:
                        self._deduplicator.resolve(incident.fingerprint)
                    return

                elif (
                    pattern_result.tier_override is not None
                    and pattern_result.tier_override.value > diagnosis.repair_tier.value
                ):
                    # Override: jump past failed tiers to a higher tier
                    self._logger.info(
                        "pattern_tier_override",
                        incident_id=incident.id,
                        pattern_id=pattern_result.pattern_id,
                        original_tier=diagnosis.repair_tier.name,
                        override_tier=pattern_result.tier_override.name,
                        skipped_tiers=pattern_result.skipped_tiers,
                        match_score=round(pattern_result.match_score, 3),
                        confidence=round(pattern_result.pattern_confidence, 3),
                    )
                    diagnosis = diagnosis.model_copy(
                        update={"repair_tier": pattern_result.tier_override}
                    )

                # ── Gap 2: RE training - pattern match captured after stamp ──
                # The `anomaly_detection` example fires in _on_incident_inner before
                # pattern data is available.  Emit a dedicated example here so the
                # RE model learns the full routing decision including pattern context.
                asyncio.ensure_future(self._emit_re_training_example(
                    category="anomaly_detection",
                    instruction=(
                        "Detect, deduplicate, score, and route an incident through the immune "
                        "pipeline, incorporating crash-pattern history to skip known-failed tiers."
                    ),
                    input_context=(
                        f"source={incident.source_system}, "
                        f"class={incident.incident_class.value}, "
                        f"fingerprint={incident.fingerprint[:16]}, "
                        f"pattern_id={incident.matched_pattern_id}, "
                        f"pattern_confidence={round(incident.pattern_confidence, 3)}, "
                        f"tier_skip_reason={incident.tier_skip_reason}"
                    ),
                    output=(
                        f"tier={diagnosis.repair_tier.name}, "
                        f"skipped_tiers={pattern_result.skipped_tiers}, "
                        f"federation_escalate={pattern_result.federation_escalate}"
                    ),
                    outcome_quality=min(1.0, 0.5 + incident.pattern_confidence * 0.5),
                    episode_id=f"pattern_match:{incident.id}",
                    constitutional_alignment=self._build_incident_alignment(
                        care=0.3,       # pattern routing reduces harm duration
                        coherence=0.4,  # skipping futile tiers is coherent
                    ),
                ))

        # ── Step 3: Prescribe ──
        incident.repair_status = RepairStatus.PRESCRIBING
        repair = await self._prescriber.prescribe(incident, diagnosis)

        self._logger.info(
            "repair_prescribed",
            incident_id=incident.id,
            tier=repair.tier.name,
            action=repair.action,
            drive_stress=round(composite_stress, 3),
            attempt_number=len(incident.repair_history) + 1,
        )

        # ── RE training: repair strategy decision ──
        _pattern_ctx = (
            f", pattern_id={incident.matched_pattern_id}"
            f", pattern_confidence={incident.pattern_confidence:.2f}"
            f", tier_skip={incident.tier_skip_reason[:60] if incident.tier_skip_reason else 'none'}"
            if incident.matched_pattern_id else ""
        )
        asyncio.ensure_future(self._emit_re_training_example(
            category="repair_strategy",
            instruction="Prescribe a repair strategy for a diagnosed incident, choosing the appropriate tier and action.",
            input_context=f"root_cause={diagnosis.root_cause[:150]}, confidence={diagnosis.confidence:.2f}, severity={incident.severity.value}, stress={composite_stress:.2f}{_pattern_ctx}",
            output=f"tier={repair.tier.name}, action={repair.action[:200]}, target={repair.target_system}",
            outcome_quality=diagnosis.confidence,
            episode_id=incident.id,
            constitutional_alignment=self._build_incident_alignment(
                coherence=diagnosis.confidence - 0.3,  # partial alignment: confident but not certain
                care=0.3,                              # repair = care for the organism
            ),
        ))

        # ── Step 4: Validate ──
        incident.repair_status = RepairStatus.VALIDATING
        validation = await self._validator.validate(incident, repair)

        if not validation.approved:
            self._logger.warning(
                "repair_rejected",
                incident_id=incident.id,
                reason=validation.reason,
            )

            # If Equor supplied alignment scores in the modifications dict,
            # update drive pressure so future repairs account for the violated drive.
            mods = validation.modifications or {}
            equor_alignment = mods.get("equor_alignment")
            if equor_alignment is not None:
                self.on_equor_rejection(
                    alignment=equor_alignment,
                    reasoning=validation.reason,
                    incident=incident,
                )

            # Escalate or accept the override
            if validation.escalate_to is not None:
                action = (
                    "alert_operator"
                    if validation.escalate_to == RepairTier.ESCALATE
                    else repair.action
                )
                repair = RepairSpec(
                    tier=validation.escalate_to,
                    action=action,
                    target_system=repair.target_system,
                    reason=f"Escalated: {validation.reason}",
                )
            else:
                incident.repair_status = RepairStatus.ESCALATED
                self._active_incidents.pop(incident.id, None)
                return

        # ── Step 5: Apply ──
        incident.repair_status = RepairStatus.APPLYING
        self._total_repairs_attempted += 1
        tier_name = repair.tier.name
        self._repairs_by_tier[tier_name] = self._repairs_by_tier.get(tier_name, 0) + 1
        self._governor.record_repair(repair.tier)
        self._validator.record_repair(repair)

        self._emit_metric("thymos.repairs.attempted", 1, tags={"tier": tier_name})

        # Per-endpoint attempt counter for API incidents (feeds Skia success-rate)
        api_ctx = self._extract_api_context(incident)
        if api_ctx is not None:
            self._emit_metric(
                "thymos.api.repair_attempt",
                1,
                tags={
                    "endpoint": api_ctx.endpoint,
                    "tier": tier_name,
                    "status_code": str(api_ctx.status_code),
                },
            )

        applied = await self._apply_repair(incident, repair)

        if not applied:
            # NOVEL_FIX (T4) records its own RepairAttempt, failure counter,
            # and _learn_from_failure inside _apply_novel_repair - skip the
            # generic recording to avoid duplicate entries and double decay.
            t4_already_recorded = (
                repair.tier == RepairTier.NOVEL_FIX
                and incident.repair_history
                and incident.repair_history[-1].tier == RepairTier.NOVEL_FIX
            )
            if not t4_already_recorded:
                self._logger.warning(
                    "repair_application_failed",
                    incident_id=incident.id,
                    tier=repair.tier.name,
                    attempt_number=len(incident.repair_history) + 1,
                )
                incident.repair_history.append(RepairAttempt(
                    tier=repair.tier,
                    action=repair.action,
                    outcome="failed",
                    reason=f"Application failed: {repair.reason}",
                ))
                self._total_repairs_failed += 1
                self._emit_metric("thymos.repairs.failed", 1, tags={"tier": tier_name})
                await self._learn_from_failure(incident, repair)

                # Pattern reinforcement: application failure confirms pattern is still dangerous
                if incident.matched_pattern_id is not None and self._crash_pattern_analyzer is not None:
                    asyncio.ensure_future(
                        self._crash_pattern_analyzer.update_on_failure(
                            pattern_id=incident.matched_pattern_id,
                            repair_tier=repair.tier.name,
                            failure_reason=f"Application failed: {repair.reason}",
                        )
                    )
                    asyncio.ensure_future(self._emit_event(
                        SynapseEventType.CRASH_PATTERN_REINFORCED,
                        {
                            "pattern_id": incident.matched_pattern_id,
                            "repair_tier": repair.tier.name,
                            "failure_reason": f"Application failed: {repair.reason[:120]}",
                            "incident_id": incident.id,
                            "confidence_before": round(incident.pattern_confidence, 3),
                        },
                    ))
                    # Emit CRASH_PATTERN_CONFIRMED when this reinforcement pushes the pattern
                    # confidence across the 0.70 confirmation threshold for the first time.
                    # _REINFORCE_DELTA = 0.08; threshold = 0.70.
                    _conf_before = incident.pattern_confidence
                    if _conf_before < 0.70 and (_conf_before + 0.08) >= 0.70:
                        if hasattr(SynapseEventType, "CRASH_PATTERN_CONFIRMED"):
                            asyncio.ensure_future(self._emit_event(
                                SynapseEventType.CRASH_PATTERN_CONFIRMED,  # type: ignore[attr-defined]
                                {
                                    "pattern_id": incident.matched_pattern_id,
                                    "confidence": round(min(0.98, _conf_before + 0.08), 3),
                                    "lesson": (
                                        f"{incident.incident_class.value} in {incident.source_system}"
                                        f" fails at tier {repair.tier.name}"
                                    ),
                                    "example_count": incident.occurrence_count if hasattr(incident, "occurrence_count") else 0,
                                    "failed_tiers": [repair.tier.name],
                                    "incident_id": incident.id,
                                    "source": "thymos_reinforcement",
                                },
                            ))

            # History-driven retry: if there are higher tiers available, retry
            # instead of immediately escalating to human.
            next_tier = incident.next_escalation_tier(repair.tier)
            if next_tier != RepairTier.ESCALATE and next_tier.value > repair.tier.value:
                self._logger.info(
                    "repair_failed_retrying_higher_tier",
                    incident_id=incident.id,
                    failed_tier=repair.tier.name,
                    next_tier=next_tier.name,
                    attempt_count=len(incident.repair_history),
                )
                # Emit INCIDENT_ESCALATED lifecycle event
                await self._emit_event(
                    SynapseEventType.INCIDENT_ESCALATED,
                    {
                        "incident_id": incident.id,
                        "incident_class": incident.incident_class.value,
                        "from_tier": repair.tier.name,
                        "to_tier": next_tier.name,
                        "reason": f"Repair failed at {repair.tier.name}",
                    },
                )
                incident.repair_status = RepairStatus.PRESCRIBING
                retry_diagnosis = Diagnosis(
                    root_cause=incident.root_cause_hypothesis or "Unknown",
                    confidence=diagnosis.confidence,
                    repair_tier=next_tier,
                    reasoning=f"Retrying after {repair.tier.name} failed",
                )
                retry_repair = await self._prescriber.prescribe(incident, retry_diagnosis)
                retry_validation = await self._validator.validate(incident, retry_repair)
                if retry_validation.approved:
                    incident.repair_status = RepairStatus.APPLYING
                    self._total_repairs_attempted += 1
                    retry_applied = await self._apply_repair(incident, retry_repair)
                    if retry_applied:
                        repair = retry_repair
                        tier_name = retry_repair.tier.name
                        # Fall through to verification below
                    else:
                        incident.repair_history.append(RepairAttempt(
                            tier=retry_repair.tier,
                            action=retry_repair.action,
                            outcome="failed",
                            reason="Retry application failed",
                        ))
                        await self._learn_from_failure(incident, retry_repair)
                        incident.repair_status = RepairStatus.ESCALATED
                        await self._escalate_with_full_context(incident, diagnosis)
                        self._active_incidents.pop(incident.id, None)
                        if self._deduplicator is not None:
                            self._deduplicator.resolve(incident.fingerprint)
                        return
                else:
                    incident.repair_status = RepairStatus.ESCALATED
                    await self._escalate_with_full_context(incident, diagnosis)
                    self._active_incidents.pop(incident.id, None)
                    if self._deduplicator is not None:
                        self._deduplicator.resolve(incident.fingerprint)
                    return
            else:
                incident.repair_status = RepairStatus.ESCALATED
                await self._escalate_with_full_context(incident, diagnosis)
                self._active_incidents.pop(incident.id, None)
                if self._deduplicator is not None:
                    self._deduplicator.resolve(incident.fingerprint)
                return

        # ── Step 6: Verify ──
        incident.repair_status = RepairStatus.VERIFYING
        verified = await self._verify_repair(incident, repair)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        if verified:
            # ── Success ──
            incident.repair_history.append(RepairAttempt(
                tier=repair.tier,
                action=repair.action,
                outcome="success",
                reason=repair.reason,
            ))
            # Escalations are already marked ESCALATED by _apply_escalation - don't overwrite.
            if incident.repair_status != RepairStatus.ESCALATED:
                incident.repair_status = RepairStatus.RESOLVED
            incident.repair_successful = True
            incident.resolution_time_ms = int(elapsed_ms)
            self._resolution_times.append(elapsed_ms)
            self._total_repairs_succeeded += 1

            self._emit_metric("thymos.repairs.succeeded", 1, tags={"tier": tier_name})
            self._emit_metric("thymos.incidents.mean_resolution_ms", elapsed_ms)

            self._logger.info(
                "incident_resolved",
                incident_id=incident.id,
                tier=repair.tier.name,
                resolution_ms=f"{elapsed_ms:.0f}",
                total_attempts=len(incident.repair_history),
            )

            # ── Emit INCIDENT_RESOLVED lifecycle event ──
            await self._emit_event(
                SynapseEventType.INCIDENT_RESOLVED,
                {
                    "incident_id": incident.id,
                    "incident_class": incident.incident_class.value,
                    "repair_tier": repair.tier.name,
                    "resolution": repair.action[:200],
                    "duration_ms": int(elapsed_ms),
                    "antibody_created": False,  # updated in _learn_from_success
                },
            )

            # ── Emit THYMOS_REPAIR_VALIDATED for Tier 3+ successes (P3 / Nova feedback) ──
            # Nova subscribes to this event to strengthen its prior belief that
            # incidents of this class are recoverable (Bayesian update on repair success).
            if repair.tier.value >= RepairTier.KNOWN_FIX.value:
                await self._emit_event(
                    SynapseEventType.THYMOS_REPAIR_VALIDATED,
                    {
                        "incident_class": incident.incident_class.value,
                        "incident_id": incident.id,
                        "antibody_id": incident.antibody_id,
                        "repair_tier": repair.tier.name,
                        "resolution_time_ms": int(elapsed_ms),
                        "source_system": incident.source_system,
                    },
                )

            # ── Emit CRASH_PATTERN_RESOLVED if this incident matched a known pattern ──
            if incident.matched_pattern_id is not None and self._crash_pattern_analyzer is not None:
                asyncio.ensure_future(
                    self._crash_pattern_analyzer.update_on_success(
                        pattern_id=incident.matched_pattern_id,
                        repair_tier=repair.tier.name,
                    )
                )
                asyncio.ensure_future(self._emit_event(
                    SynapseEventType.CRASH_PATTERN_RESOLVED,
                    {
                        "pattern_id": incident.matched_pattern_id,
                        "repair_tier": repair.tier.name,
                        "strategy_used": repair.action[:200],
                        "time_to_resolve_ms": int(elapsed_ms),
                        "incident_id": incident.id,
                        "confidence_before": round(incident.pattern_confidence, 3),
                    },
                ))

            # ── Step 7: Learn ──
            await self._learn_from_success(incident, repair, diagnosis)

            # ── RE training: antibody creation (successful healing) ──
            asyncio.ensure_future(self._emit_re_training_example(
                category="antibody_creation",
                instruction="Learn from a successful repair to create an antibody for future similar incidents.",
                input_context=f"incident_class={incident.incident_class.value}, tier={repair.tier.name}, attempts={len(incident.repair_history)}",
                output=f"resolved=True, resolution_ms={int(elapsed_ms)}, action={repair.action[:200]}",
                outcome_quality=min(1.0, 0.6 + diagnosis.confidence * 0.4),
                latency_ms=int(elapsed_ms),
                reasoning_trace=diagnosis.root_cause[:200],
                episode_id=incident.id,
                constitutional_alignment=self._build_incident_alignment(
                    care=0.7,       # successful healing = high care
                    growth=0.5,     # antibody creation = growth/learning
                    coherence=0.4,  # resolved incident = restored coherence
                ),
            ))

            await self._emit_evolutionary_observable(
                "incident_healed", 1.0, is_novel=True,
                metadata={
                    "incident_id": incident.id,
                    "tier": repair.tier.name,
                    "resolution_ms": int(elapsed_ms),
                },
            )

            # ── Step 7b: Inject recovery monitoring goal for RESTART+ repairs ──
            # AV1: goal injection now uses Synapse emit
            if repair.tier >= RepairTier.RESTART and self._synapse is not None:
                await self._inject_repair_goal(incident, repair.tier, resolved=True)

        else:
            # ── Rollback ──
            incident.repair_history.append(RepairAttempt(
                tier=repair.tier,
                action=repair.action,
                outcome="rolled_back",
                reason="Post-repair verification failed",
            ))
            incident.repair_status = RepairStatus.ROLLED_BACK
            incident.repair_successful = False
            self._total_repairs_rolled_back += 1

            self._emit_metric("thymos.repairs.rolled_back", 1, tags={"tier": tier_name})

            self._logger.warning(
                "repair_rolled_back",
                incident_id=incident.id,
                tier=repair.tier.name,
                reason="Post-repair verification failed",
                attempt_count=len(incident.repair_history),
            )

            # ── Learn from failure ──
            await self._learn_from_failure(incident, repair)

            # ── Emit CRASH_PATTERN_REINFORCED if this incident matched a known pattern ──
            if incident.matched_pattern_id is not None and self._crash_pattern_analyzer is not None:
                asyncio.ensure_future(
                    self._crash_pattern_analyzer.update_on_failure(
                        pattern_id=incident.matched_pattern_id,
                        repair_tier=repair.tier.name,
                        failure_reason="Post-repair verification failed",
                    )
                )
                asyncio.ensure_future(self._emit_event(
                    SynapseEventType.CRASH_PATTERN_REINFORCED,
                    {
                        "pattern_id": incident.matched_pattern_id,
                        "repair_tier": repair.tier.name,
                        "failure_reason": "Post-repair verification failed",
                        "incident_id": incident.id,
                        "confidence_before": round(incident.pattern_confidence, 3),
                    },
                ))
                # Emit CRASH_PATTERN_CONFIRMED when confidence crosses the 0.70 threshold.
                _conf_before_rollback = incident.pattern_confidence
                if _conf_before_rollback < 0.70 and (_conf_before_rollback + 0.08) >= 0.70:
                    if hasattr(SynapseEventType, "CRASH_PATTERN_CONFIRMED"):
                        asyncio.ensure_future(self._emit_event(
                            SynapseEventType.CRASH_PATTERN_CONFIRMED,  # type: ignore[attr-defined]
                            {
                                "pattern_id": incident.matched_pattern_id,
                                "confidence": round(min(0.98, _conf_before_rollback + 0.08), 3),
                                "lesson": (
                                    f"{incident.incident_class.value} in {incident.source_system}"
                                    f" persists after tier {repair.tier.name} (post-repair verification failed)"
                                ),
                                "example_count": incident.occurrence_count if hasattr(incident, "occurrence_count") else 0,
                                "failed_tiers": [repair.tier.name],
                                "incident_id": incident.id,
                                "source": "thymos_rollback_reinforcement",
                            },
                        ))

            await self._emit_evolutionary_observable(
                "healing_failed", 1.0, is_novel=True,
                metadata={
                    "incident_id": incident.id,
                    "tier": repair.tier.name,
                    "attempt_count": len(incident.repair_history),
                },
            )

            # History-driven retry after rollback: try next higher tier
            next_tier = incident.next_escalation_tier(repair.tier)
            if next_tier != RepairTier.ESCALATE and next_tier.value > repair.tier.value:
                self._logger.info(
                    "rollback_retrying_higher_tier",
                    incident_id=incident.id,
                    rolled_back_tier=repair.tier.name,
                    next_tier=next_tier.name,
                )
                incident.repair_status = RepairStatus.PRESCRIBING
                retry_diagnosis = Diagnosis(
                    root_cause=incident.root_cause_hypothesis or "Unknown",
                    confidence=diagnosis.confidence,
                    repair_tier=next_tier,
                    reasoning=f"Retrying after {repair.tier.name} rolled back",
                )
                retry_repair = await self._prescriber.prescribe(incident, retry_diagnosis)
                retry_validation = await self._validator.validate(incident, retry_repair)
                if retry_validation.approved:
                    incident.repair_status = RepairStatus.APPLYING
                    self._total_repairs_attempted += 1
                    retry_applied = await self._apply_repair(incident, retry_repair)
                    if retry_applied:
                        incident.repair_status = RepairStatus.VERIFYING
                        retry_verified = await self._verify_repair(incident, retry_repair)
                        if retry_verified:
                            incident.repair_history.append(RepairAttempt(
                                tier=retry_repair.tier,
                                action=retry_repair.action,
                                outcome="success",
                                reason=retry_repair.reason,
                            ))
                            incident.repair_status = RepairStatus.RESOLVED
                            incident.repair_successful = True
                            retry_elapsed = (time.monotonic() - start_time) * 1000
                            incident.resolution_time_ms = int(retry_elapsed)
                            self._resolution_times.append(retry_elapsed)
                            self._total_repairs_succeeded += 1
                            await self._learn_from_success(incident, retry_repair, retry_diagnosis)
                        else:
                            incident.repair_history.append(RepairAttempt(
                                tier=retry_repair.tier,
                                action=retry_repair.action,
                                outcome="rolled_back",
                                reason="Retry verification failed",
                            ))
                            await self._learn_from_failure(incident, retry_repair)
                            await self._escalate_with_full_context(incident, diagnosis)
                    else:
                        incident.repair_history.append(RepairAttempt(
                            tier=retry_repair.tier,
                            action=retry_repair.action,
                            outcome="failed",
                            reason="Retry application failed after rollback",
                        ))
                        await self._learn_from_failure(incident, retry_repair)
                        await self._escalate_with_full_context(incident, diagnosis)
                else:
                    await self._escalate_with_full_context(incident, diagnosis)

        # Clean up active tracking
        self._active_incidents.pop(incident.id, None)
        if self._deduplicator is not None:
            self._deduplicator.resolve(incident.fingerprint)
        if self._governor is not None:
            self._governor.resolve_incident(incident.id)

        # Check storm exit
        self._governor.check_storm_exit()

    # ─── Diagnosis ──────────────────────────────────────────────────

    async def _diagnose(self, incident: Incident) -> Diagnosis:
        """
        Full diagnostic sequence:
        1. Check antibody library for known fix
        2. Trace causal chain through dependency graph
        3. Correlate temporal events
        4. Generate and test hypotheses
        """
        assert self._antibody_library is not None
        assert self._causal_analyzer is not None
        assert self._temporal_correlator is not None
        assert self._diagnostic_engine is not None

        # Fast path: check antibody library
        antibody_match = await self._antibody_library.lookup(incident.fingerprint)
        if antibody_match is not None and antibody_match.effectiveness > 0.8:
            # Bypass the antibody fast path when:
            #  1. The router already force-escalated to T4 (NOVEL_FIX) - the
            #     existing repair ISN'T working, that's why T4 was forced.
            #  2. The antibody's repair tier is low (≤ PARAMETER) and the incident
            #     is recurring (count > 3) - parameter tweaks can't fix structural issues.
            t4_forced = (
                incident.repair_tier == RepairTier.NOVEL_FIX
                and self._response_router is not None
                and incident.fingerprint in self._response_router._t4_escalated_at
            )
            antibody_is_low_tier = antibody_match.repair_tier.value <= RepairTier.PARAMETER.value
            should_bypass = t4_forced or (antibody_is_low_tier and incident.occurrence_count > 3)

            if should_bypass:
                self._logger.info(
                    "antibody_bypass_recurring",
                    incident_id=incident.id,
                    antibody_id=antibody_match.id,
                    antibody_tier=antibody_match.repair_tier.name,
                    occurrence_count=incident.occurrence_count,
                    t4_forced=t4_forced,
                )
                antibody_match = None  # Force full diagnosis
            else:
                self._total_antibodies_applied += 1
                self._emit_metric("thymos.antibodies.applied", 1)
                self._logger.info(
                    "antibody_match",
                    incident_id=incident.id,
                    antibody_id=antibody_match.id,
                    effectiveness=f"{antibody_match.effectiveness:.2f}",
                )
                # Skip full diagnosis - antibody provides root cause and repair tier
                return await self._diagnostic_engine.diagnose(
                    incident=incident,
                    causal_chain=await self._causal_analyzer.trace_root_cause(incident),
                    correlations=[],
                    antibody_match=antibody_match,
                )

        # Causal analysis: trace upstream dependencies
        causal_chain = await self._causal_analyzer.trace_root_cause(incident)

        # Temporal correlation: what changed before the incident?
        correlations = self._temporal_correlator.correlate(incident)

        # Full diagnosis with hypothesis generation
        diagnosis = await self._diagnostic_engine.diagnose(
            incident=incident,
            causal_chain=causal_chain,
            correlations=correlations,
            antibody_match=antibody_match,
        )

        self._emit_metric("thymos.diagnosis.hypotheses", len(diagnosis.all_hypotheses))

        return diagnosis

    # ─── Repair Application ──────────────────────────────────────────

    async def _apply_repair(self, incident: Incident, repair: RepairSpec) -> bool:
        """
        Apply a repair based on its tier.

        Returns True if the repair was applied successfully (not verified yet).
        """
        try:
            if repair.tier == RepairTier.NOOP:
                return True

            elif repair.tier == RepairTier.PARAMETER:
                return await self._apply_parameter_repair(repair)

            elif repair.tier == RepairTier.RESTART:
                return await self._apply_restart_repair(repair)

            elif repair.tier == RepairTier.KNOWN_FIX:
                # Sandbox gate: validate Tier 3 repair before applying
                if not await self._request_sandbox_validation(incident, repair):
                    self._logger.warning(
                        "repair_sandbox_rejected",
                        incident_id=incident.id,
                        tier="KNOWN_FIX",
                    )
                    return False
                return await self._apply_antibody_repair(incident, repair)

            elif repair.tier == RepairTier.NOVEL_FIX:
                # Sandbox gate: validate Tier 4 repair before applying
                if not await self._request_sandbox_validation(incident, repair):
                    self._logger.warning(
                        "repair_sandbox_rejected",
                        incident_id=incident.id,
                        tier="NOVEL_FIX",
                    )
                    return False
                return await self._apply_novel_repair(incident, repair)

            elif repair.tier == RepairTier.FACTORY_REPAIR:
                return await self._apply_factory_repair(incident, repair)

            elif repair.tier == RepairTier.ESCALATE:
                return await self._apply_escalation(incident, repair)

            else:
                self._logger.warning(
                    "unknown_repair_tier",
                    tier=repair.tier,
                    incident_id=incident.id,
                )
                return False

        except Exception as exc:
            self._logger.error(
                "repair_application_error",
                incident_id=incident.id,
                tier=repair.tier.name,
                error=str(exc),
            )
            return False

    async def _apply_parameter_repair(self, repair: RepairSpec) -> bool:
        """Apply Tier 1: parameter adjustment."""
        if not repair.parameter_changes:
            return False

        applied_count = 0
        for change in repair.parameter_changes:
            path = change.get("parameter_path", "")
            delta = change.get("delta", 0)
            reason = change.get("reason", repair.reason)

            if self._evo is not None:
                # Route parameter changes through Evo's tuner (velocity-limited, range-clamped)
                applied = await self._evo.apply_immune_parameter_adjustment(
                    parameter_path=path,
                    delta=delta,
                    reason=reason,
                )
                if applied:
                    applied_count += 1
                else:
                    self._logger.debug(
                        "parameter_adjustment_skipped",
                        path=path,
                        delta=delta,
                    )
            else:
                self._logger.info(
                    "parameter_adjustment_no_evo",
                    path=path,
                    delta=delta,
                )
                applied_count += 1

        return applied_count > 0

    async def _apply_restart_repair(self, repair: RepairSpec) -> bool:
        """
        Apply Tier 2: system restart.

        Thymos doesn't restart systems directly - it signals Synapse's
        DegradationManager to handle the restart sequence.
        """
        target = repair.target_system
        if target is None:
            return False

        if self._synapse is not None:
            try:
                # Emit a restart request through the event bus
                await self._synapse._event_bus.emit(
                    SynapseEvent(
                        event_type=SynapseEventType.SYSTEM_RESTARTING,
                        data={
                            "system_id": target,
                            "reason": repair.reason,
                            "requested_by": "thymos",
                        },
                        source_system="thymos",
                    )
                )
                self._logger.info(
                    "restart_requested",
                    target_system=target,
                    reason=repair.reason,
                )
                return True
            except Exception as exc:
                self._logger.error(
                    "restart_request_failed",
                    target_system=target,
                    error=str(exc),
                )
                return False

        self._logger.warning(
            "restart_no_synapse",
            target_system=target,
        )
        return False

    async def _apply_antibody_repair(self, incident: Incident, repair: RepairSpec) -> bool:
        """Apply Tier 3: known fix from antibody library."""
        assert self._antibody_library is not None

        if repair.antibody_id is None:
            return False

        antibody = self._antibody_library._all.get(repair.antibody_id)
        if antibody is None:
            self._logger.warning(
                "antibody_not_found",
                antibody_id=repair.antibody_id,
            )
            return False

        # Apply the antibody's repair spec
        inner_repair = antibody.repair_spec
        incident.antibody_id = antibody.id

        self._logger.info(
            "applying_antibody",
            antibody_id=antibody.id,
            inner_tier=inner_repair.tier.name,
            inner_action=inner_repair.action,
        )

        # Recursively apply the inner repair (but not another antibody to avoid loops)
        if inner_repair.tier == RepairTier.PARAMETER:
            return await self._apply_parameter_repair(inner_repair)
        elif inner_repair.tier == RepairTier.RESTART:
            return await self._apply_restart_repair(inner_repair)
        else:
            # For more complex inner repairs, log and mark as applied
            self._logger.info(
                "antibody_complex_repair",
                antibody_id=antibody.id,
                inner_tier=inner_repair.tier.name,
            )
            return True

    def _cleanup_expired_proposals(self) -> None:
        """
        Remove in-flight proposals that have exceeded the 10-minute TTL.
        Called periodically to allow resubmission of stuck proposals.
        """
        now = utc_now().timestamp()
        ttl_seconds = 600.0  # 10 minutes
        expired = []

        for fingerprint, (_proposal_id, submitted_at) in self._active_simula_proposals.items():
            if now - submitted_at > ttl_seconds:
                expired.append(fingerprint)

        for fingerprint in expired:
            proposal_id, submitted_at = self._active_simula_proposals.pop(fingerprint)
            age_seconds = now - submitted_at
            self._logger.warning(
                "thymos_simula_proposal_ttl_expired",
                fingerprint=fingerprint,
                proposal_id=proposal_id,
                age_seconds=age_seconds,
            )

    def _classify_error_for_repair(self, incident: Incident) -> str:
        """
        Classify an incident's error type to determine proposal category.

        Returns ChangeCategory value string:
        - "bug_fix" for runtime errors fixable by code generation
          (AttributeError, KeyError, missing executor registration, method signature mismatch)
        - "add_system_capability" for architectural changes (default)
        """
        error_type = incident.error_type.lower()
        error_msg = incident.error_message.lower()

        # Runtime errors that Simula can autonomously fix
        if error_type == "attributeerror":
            # Missing attribute on an object - add it to the class
            return "bug_fix"

        if error_type == "keyerror":
            # Missing key in dict/config - add it with default value
            return "bug_fix"

        if "no executor registered" in error_msg:
            # Unregistered executor - register it in Axon
            return "bug_fix"

        if "typeerror" in error_type and "positional argument" in error_msg:
            # Method signature mismatch - fix the call site or signature
            return "bug_fix"

        if "pydantic" in error_type or "not fully defined" in error_msg:
            # Pydantic model validation / forward-ref errors - fix imports or model_rebuild
            return "bug_fix"

        if error_type in ("importerror", "modulenotfounderror"):
            # Missing import - fix the import statement
            return "bug_fix"

        if "model_rebuild" in error_msg or "forward ref" in error_msg:
            # Pydantic forward reference issues
            return "bug_fix"

        # Default: architectural/capability change
        return "add_system_capability"

    @staticmethod
    def _extract_api_context(incident: Incident) -> ApiErrorContext | None:
        """Extract API context from typed ApiErrorContext or plain dict."""
        if isinstance(incident.context, ApiErrorContext):
            return incident.context
        if isinstance(incident.context, dict):
            ctx = incident.context
            if "http_path" in ctx:
                return ApiErrorContext(
                    endpoint=ctx["http_path"],
                    method=ctx.get("http_method", "GET"),
                    status_code=ctx.get("http_status", 0),
                    request_id=ctx.get("request_id", ""),
                    remote_addr=ctx.get("remote_addr", "unknown"),
                    latency_ms=ctx.get("latency_ms", 0.0),
                    user_agent=ctx.get("user_agent", ""),
                )
        return None

    async def _apply_novel_repair(self, incident: Incident, repair: RepairSpec) -> bool:
        """
        Apply Tier 4: novel fix via Simula Code Agent.

        Iron rule: CANNOT apply without Equor review (enforced by validator).
        """
        assert self._governor is not None

        if not self._governor.should_codegen():
            self._logger.info(
                "codegen_throttled",
                incident_id=incident.id,
            )
            return False

        # Cytokine storm prevention: check T4 proposal budget
        if not self._governor.can_submit_t4_proposal():
            self._logger.warning(
                "thymos_t4_budget_exhausted",
                incident_id=incident.id,
                t4_proposals_this_hour=self._governor.budget_state.t4_proposals_this_hour,
                max_t4_per_hour=self._governor.budget_state.max_t4_proposals_per_hour,
                active_t4_proposals=self._governor.budget_state.active_t4_proposals,
                max_concurrent_t4=self._governor.budget_state.max_concurrent_t4_proposals,
            )
            # Incident stays at T3 - don't escalate, just block T4 submission
            return False

        self._governor.begin_codegen()
        self._governor.begin_t4_proposal()
        try:
            # Build the context payload passed to Simula's Code Agent.
            # For API incidents, include endpoint + stack trace so the agent
            # can generate a targeted patch for the failing handler.
            codegen_context: dict[str, Any] = {
                "incident_id": incident.id,
                "source_system": incident.source_system,
                "error_type": incident.error_type,
                "error_message": incident.error_message,
                "stack_trace": incident.stack_trace or "",
                "repair_reason": repair.reason,
            }
            # Extract API context from either typed ApiErrorContext or plain dict
            # (error_capture middleware builds plain dicts).
            api_ctx = self._extract_api_context(incident)
            if api_ctx is not None:
                codegen_context.update({
                    "endpoint": api_ctx.endpoint,
                    "method": api_ctx.method,
                    "status_code": api_ctx.status_code,
                    "latency_ms": api_ctx.latency_ms,
                    "request_id": api_ctx.request_id,
                })
                self._logger.info(
                    "novel_repair_requested",
                    incident_id=incident.id,
                    target_system=repair.target_system,
                    endpoint=api_ctx.endpoint,
                    status_code=api_ctx.status_code,
                    reason=repair.reason,
                )
                self._emit_metric(
                    "thymos.api.novel_repair",
                    1,
                    tags={
                        "endpoint": api_ctx.endpoint,
                        "status_code": str(api_ctx.status_code),
                    },
                )
            else:
                self._logger.info(
                    "novel_repair_requested",
                    incident_id=incident.id,
                    target_system=repair.target_system,
                    reason=repair.reason,
                )

            if self._synapse is None:
                # No Synapse bus - cannot route THYMOS_REPAIR_REQUESTED to Simula.
                # Record the failure so repair_history and antibody scores reflect
                # the outcome, then let the caller handle escalation to Tier 5.
                unavail_reason = (
                    f"Tier 4 repair failed: Synapse bus unavailable (cannot reach Simula). "
                    f"Incident: {incident.id} ({incident.incident_class.value}). "
                    f"Original reason: {repair.reason}"
                )
                self._logger.error(
                    "novel_repair_simula_unavailable",
                    incident_id=incident.id,
                    reason=unavail_reason,
                )
                incident.repair_history.append(RepairAttempt(
                    tier=repair.tier,
                    action=repair.action,
                    outcome="failed",
                    reason=unavail_reason,
                ))
                self._total_repairs_failed += 1
                self._emit_metric(
                    "thymos.repairs.failed",
                    1,
                    tags={"tier": repair.tier.name},
                )
                await self._learn_from_failure(incident, repair)
                repair.reason = unavail_reason
                return False

            # ── Cytokine Storm Prevention: In-flight Proposal Check ──
            # Clean up any proposals that have exceeded the 10-minute TTL
            self._cleanup_expired_proposals()

            # Check if this fingerprint already has an in-flight proposal
            if incident.fingerprint in self._active_simula_proposals:
                existing_proposal_id, _ = self._active_simula_proposals[incident.fingerprint]
                self._logger.info(
                    "thymos_simula_proposal_already_inflight",
                    incident_id=incident.id,
                    fingerprint=incident.fingerprint,
                    existing_proposal_id=existing_proposal_id,
                )
                return False

            # Generate a stable proposal ID for dedup tracking
            from primitives.common import new_id
            proposal_id = new_id()
            category_str = self._classify_error_for_repair(incident)

            self._logger.info(
                "novel_repair_submitting_to_simula",
                incident_id=incident.id,
                proposal_id=proposal_id,
                target_system=repair.target_system,
            )

            # Store the in-flight proposal for dedup (TTL enforced by _cleanup_expired_proposals)
            now = utc_now().timestamp()
            self._active_simula_proposals[incident.fingerprint] = (proposal_id, now)

            # Emit THYMOS_REPAIR_REQUESTED - Simula subscribes and builds the
            # EvolutionProposal internally.  This avoids the cross-system
            # EvolutionProposal import and the direct process_proposal() call.
            # Simula will emit EVOLUTION_APPLIED / EVOLUTION_ROLLED_BACK when done.
            await self._synapse.event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.THYMOS_REPAIR_REQUESTED,
                source_system="thymos",
                data={
                    "incident_id": incident.id,
                    "proposal_id": proposal_id,
                    "affected_system": incident.source_system,
                    "affected_systems": [incident.source_system] + (incident.affected_systems or []),
                    "description": (
                        f"[Thymos T4] Repair {incident.incident_class.value} in "
                        f"{incident.source_system}: {repair.action}"
                    ),
                    "category": category_str,
                    "severity": incident.severity.value,
                    "repair_tier": 4,
                    "error_message": incident.error_message[:200],
                    "root_cause": incident.root_cause_hypothesis or "unknown",
                    "stack_trace": (incident.stack_trace or "")[:500],
                    "diagnostic_confidence": incident.diagnostic_confidence,
                    "code_hint": codegen_context.get("endpoint", ""),
                    "expected_benefit": (
                        f"Resolve {incident.severity.value} incident {incident.id} "
                        f"in {incident.source_system}"
                    ),
                },
            ))

            # Tier 4 is now fire-and-forget via Synapse.  Simula processes the
            # proposal asynchronously and will emit EVOLUTION_APPLIED on success.
            # Return False here so the caller can proceed with monitoring; the
            # in-flight entry will be cleaned up on TTL expiry or when Simula
            # confirms via EVOLUTION_APPLIED / EVOLUTION_ROLLED_BACK.
            repair.reason = (
                f"Tier 4 repair delegated to Simula (proposal_id={proposal_id}). "
                f"Outcome tracked via EVOLUTION_APPLIED / EVOLUTION_ROLLED_BACK."
            )
            return False
        finally:
            self._governor.end_codegen()
            self._governor.end_t4_proposal()

    async def _apply_factory_repair(
        self, incident: Incident, repair: RepairSpec
    ) -> bool:
        """
        Apply Tier 5: dispatch to EcodiaOS Factory for autonomous CC code fix.

        The Factory is the organism's other body — it has full Claude Code autonomy
        across all codebases (EcodiaOS admin, frontend, organism, client projects).
        Dispatched via SymbridgeFactoryExecutor through the symbridge Redis stream.

        This tier is used when:
        - Simula's local code agent (Tier 4) is insufficient
        - The fix spans multiple codebases
        - The incident involves externally deployed systems
        - Full Claude Code power is needed (200 turns, 2hr timeout)
        """
        self._logger.info(
            "factory_repair_dispatching",
            incident_id=incident.id,
            source_system=incident.source_system,
            reason=repair.reason,
        )

        try:
            # Dispatch via Synapse → SymbridgeFactoryExecutor → Redis → EcodiaOS Factory
            if self._event_bus is not None:
                await self._event_bus.emit(
                    SynapseEventType.FACTORY_PROPOSAL_SENT,
                    {
                        "dispatch_type": "thymos_incident",
                        "incident_id": incident.id,
                        "severity": incident.severity.value if hasattr(incident.severity, "value") else str(incident.severity),
                        "affected_system": incident.source_system,
                        "error_message": incident.error_message or str(incident.error_type),
                        "description": repair.reason,
                        "stack_trace": getattr(incident, "stack_trace", ""),
                        "codebase_name": getattr(incident, "codebase_name", None) or incident.source_system,
                        "repair_tier": "FACTORY_REPAIR",
                        "diagnosis_confidence": getattr(repair, "_diagnosis_confidence", 0.0),
                    },
                )
                self._logger.info(
                    "factory_repair_dispatched",
                    incident_id=incident.id,
                )
                return True
            else:
                self._logger.warning(
                    "factory_repair_no_event_bus",
                    incident_id=incident.id,
                )
                return False

        except Exception as exc:
            self._logger.error(
                "factory_repair_dispatch_failed",
                incident_id=incident.id,
                error=str(exc),
            )
            return False

    async def _apply_escalation(self, incident: Incident, repair: RepairSpec) -> bool:
        """
        Apply Tier 6 (ESCALATE): attempt Equor auto-approval first, fall back to human escalation.

        The Telegram dependency is removed from the critical path. Instead:
        1. Build an Intent from the repair and submit to Equor for constitutional review.
        2. If Equor approves with high confidence (≥0.7) and all drives aligned,
           auto-approve and attempt the repair via Simula.
        3. If Equor blocks, defers, or has low confidence → fall back to human
           notification via NotificationDispatcher (which may use Telegram, Redis,
           or any configured channel - but Thymos no longer depends on Telegram
           specifically).

        Iron rule: Thymos CANNOT bypass Equor. Auto-approval requires Equor's
        explicit constitutional consent.
        """
        # ── Step 1: Try Equor auto-approval ────────────────────────────
        auto_approved = await self._try_equor_auto_approval(incident, repair)
        if auto_approved:
            return True

        # ── Step 2: SG4 - Federation escalation before human notification ──
        # Broadcast the unresolved incident to federated peers.  A peer with an
        # effective antibody can resolve it without human intervention; the local
        # instance waits up to _FEDERATION_ESCALATION_TIMEOUT_S before falling
        # through to Telegram / human escalation.
        federation_resolved = await self._try_federation_escalation(incident, repair)
        if federation_resolved:
            return True

        # ── Step 3: Fall back to human escalation ──────────────────────
        self._logger.warning(
            "incident_escalated_to_human",
            incident_id=incident.id,
            source_system=incident.source_system,
            severity=incident.severity.value,
            reason=repair.reason,
            repair_attempts=len(incident.repair_history),
        )
        incident.repair_status = RepairStatus.ESCALATED

        # Build rich attempt history from repair_history
        tried: list[str] = []
        for attempt in incident.repair_history:
            entry = (
                f"Tier {attempt.tier.name}: {attempt.action} → {attempt.outcome}"
            )
            if attempt.reason:
                entry += f" ({attempt.reason[:100]})"
            tried.append(entry)

        if incident.root_cause_hypothesis:
            tried.append(
                f"Root cause hypothesis: {incident.root_cause_hypothesis[:150]}"
            )
        if incident.antibody_id:
            tried.append(f"Antibody applied: {incident.antibody_id}")

        # Include Soma health state if available (AV1: from cached signal)
        soma_summary = ""
        if self._cached_soma_signal:
            pw = self._cached_soma_signal.get("precision_weights", {})
            integrity = pw.get("integrity", "?")
            coherence = pw.get("coherence", "?")
            try:
                soma_summary = (
                    f" | Soma integrity={float(integrity):.2f},"
                    f" coherence={float(coherence):.2f}"
                )
            except (TypeError, ValueError):
                soma_summary = " | Soma: partial"
        elif self._soma is not None:
            soma_summary = " | Soma: no cached signal"

        # Plain-language summary
        attempt_count = len(incident.repair_history)
        failed_tiers = [
            a.tier.name for a in incident.repair_history
            if a.outcome in ("failed", "rolled_back")
        ]
        summary = (
            f"{attempt_count} repair attempt(s) made. "
            f"Failed tiers: {', '.join(failed_tiers) or 'none'}. "
            f"Occurrences: {incident.occurrence_count}{soma_summary}"
        )

        await self._notification_dispatcher.dispatch(
            incident_id=incident.id,
            severity=incident.severity.value,
            system=incident.source_system,
            what_was_tried=tried or ["No automated repair tiers succeeded."],
            what_failed=f"{repair.reason[:200]} | {summary}",
            recommended_human_action=(
                f"Investigate {incident.incident_class.value} in "
                f"{incident.source_system}. "
                f"Error: {incident.error_message[:200]}"
            ),
        )
        return True

    async def _try_equor_auto_approval(
        self,
        incident: Incident,
        repair: RepairSpec,
    ) -> bool:
        """
        Ask Equor if this Tier 5 escalation can be auto-approved.

        Returns True if Equor approved and the repair was dispatched to Simula.
        Returns False if Equor blocked, had low confidence, or was unavailable.

        This replaces the Telegram dependency for Tier 5 repairs that Equor
        confirms are constitutionally safe. Telegram remains as a fallback
        notification channel, not a blocking approval gate.
        """
        if self._equor is None:
            return False

        # Build a synthetic Intent for constitutional review
        from primitives.common import Verdict, new_id
        from primitives.intent import ActionSequence, GoalDescriptor, Intent

        intent = Intent(
            id=new_id(),
            goal=GoalDescriptor(
                description=(
                    f"Auto-repair Tier 5: {repair.action} "
                    f"for {incident.source_system}"
                ),
                target_domain=incident.source_system,
                success_criteria={
                    "incident_id": incident.id,
                    "repair_tier": "ESCALATE",
                    "repair_action": repair.action,
                    "incident_class": incident.incident_class.value,
                },
            ),
            plan=ActionSequence(steps=[]),
        )

        try:
            check = await self._equor.review(intent)
        except Exception as exc:
            self._logger.warning(
                "equor_auto_approval_failed",
                incident_id=incident.id,
                error=str(exc)[:200],
            )
            return False

        # Auto-approve only if:
        # 1. Equor explicitly APPROVED (not MODIFY, ESCALATE, or DENY)
        # 2. Confidence >= 0.7 (Equor is reasonably sure)
        # 3. No drive is violated (all alignment scores >= 0)
        alignment = getattr(check, "alignment", {}) or {}
        confidence = getattr(check, "confidence", 0.0)
        verdict = getattr(check, "verdict", Verdict.BLOCKED)

        drives_safe = all(
            alignment.get(d, 0.0) >= 0.0
            for d in ("coherence", "care", "growth", "honesty")
        )

        auto_approved = (
            verdict == Verdict.APPROVED
            and confidence >= 0.7
            and drives_safe
        )

        # Emit TIER5_AUTO_APPROVAL event for observability
        if self._synapse is not None:
            try:
                await self._synapse._event_bus.emit(
                    SynapseEvent(
                        event_type=SynapseEventType.TIER5_AUTO_APPROVAL,
                        source_system="thymos",
                        data={
                            "incident_id": incident.id,
                            "repair_action": repair.action,
                            "equor_confidence": confidence,
                            "drive_alignment": alignment,
                            "auto_approved": auto_approved,
                            "verdict": str(verdict),
                        },
                    )
                )
            except Exception:
                pass

        if not auto_approved:
            self._logger.info(
                "equor_auto_approval_denied",
                incident_id=incident.id,
                verdict=str(verdict),
                confidence=round(confidence, 3),
                drives_safe=drives_safe,
            )
            return False

        # Auto-approved - dispatch to Simula as a Tier 4 novel repair
        self._logger.info(
            "tier5_auto_approved_by_equor",
            incident_id=incident.id,
            confidence=round(confidence, 3),
            repair_action=repair.action,
        )

        # If Simula already has a proposal awaiting governance for this incident
        # (from a prior Tier 4 THYMOS_REPAIR_REQUESTED), emit THYMOS_REPAIR_APPROVED
        # so Simula can fast-track it through approve_governed_proposal() instead of
        # generating a duplicate proposal.
        existing_proposal_id: str | None = None
        if incident.fingerprint in self._active_simula_proposals:
            existing_proposal_id, _ = self._active_simula_proposals[incident.fingerprint]
        if existing_proposal_id is not None and self._synapse is not None:
            try:
                await self._synapse.event_bus.emit(
                    SynapseEvent(
                        event_type=SynapseEventType.THYMOS_REPAIR_APPROVED,
                        source_system="thymos",
                        data={
                            "incident_id": incident.id,
                            "proposal_id": existing_proposal_id,
                            "repair_tier": 5,
                            "equor_confidence": confidence,
                            "drive_alignment": alignment,
                        },
                    )
                )
                self._logger.info(
                    "thymos_repair_approved_emitted",
                    incident_id=incident.id,
                    proposal_id=existing_proposal_id,
                )
            except Exception:
                pass

        # Record the auto-approval in repair history
        incident.repair_history.append(RepairAttempt(
            tier=RepairTier.ESCALATE,
            action=f"auto_approved:{repair.action}",
            outcome="auto_approved",
            reason=(
                f"Equor approved with confidence={confidence:.2f}. "
                f"Dispatching to Simula."
            ),
        ))

        # Downgrade to NOVEL_FIX and dispatch to Simula
        auto_repair = RepairSpec(
            tier=RepairTier.NOVEL_FIX,
            action=repair.action or "simula_codegen",
            target_system=incident.source_system,
            reason=f"Tier 5 auto-approved by Equor: {repair.reason}",
        )
        success = await self._apply_novel_repair(incident, auto_repair)
        if success:
            incident.repair_status = RepairStatus.RESOLVED
            self._emit_metric("thymos.tier5.auto_approved", 1)
        else:
            # Simula failed - fall through to human escalation
            self._emit_metric("thymos.tier5.auto_approved_simula_failed", 1)
            return False

        return True

    async def _try_federation_escalation(
        self,
        incident: Incident,
        repair: RepairSpec,
    ) -> bool:
        """
        SG4 - Broadcast the unresolved incident to federated peers before
        falling back to human notification.

        Protocol:
        1. Emit INCIDENT_ESCALATED with federation_broadcast=True.  Federated
           peers that hold an effective antibody for this fingerprint will emit
           FEDERATION_ASSISTANCE_ACCEPTED back with the repair steps.
        2. Wait up to _FEDERATION_ESCALATION_TIMEOUT_S for a peer reply.
        3. If a peer claims the incident and resolves it, return True.
        4. If no peer responds or all decline, return False → human escalation.

        This is best-effort: network and bus errors cause immediate fall-through
        (fail-open toward human escalation, never silent failure).
        """
        if self._synapse is None:
            return False

        # Only broadcast if there is at least one federated peer
        peer_count: int = 0
        try:
            peer_count = len(getattr(self._synapse, "federation_peers", {}) or {})
        except Exception:
            pass
        if peer_count == 0:
            return False

        self._logger.info(
            "federation_escalation_broadcast",
            incident_id=incident.id,
            source_system=incident.source_system,
            fingerprint=incident.fingerprint,
            peer_count=peer_count,
        )

        # Emit the broadcast - peers subscribed to INCIDENT_ESCALATED will see it
        await self._emit_event(
            SynapseEventType.INCIDENT_ESCALATED,
            {
                "incident_id": incident.id,
                "fingerprint": incident.fingerprint,
                "incident_class": incident.incident_class.value,
                "source_system": incident.source_system,
                "severity": incident.severity.value,
                "error_message": (incident.error_message or "")[:300],
                "repair_attempts": len(incident.repair_history),
                "federation_broadcast": True,
                "repair_action_hint": repair.action[:200],
            },
        )
        self._emit_metric("thymos.federation_escalation.broadcast", 1)

        # Wait for FEDERATION_ASSISTANCE_ACCEPTED carrying our incident_id.
        # We use a one-shot asyncio.Event injected into a temporary handler.
        resolved_event: asyncio.Event = asyncio.Event()
        peer_resolution: dict[str, object] = {}

        async def _on_peer_assistance(event: SynapseEvent) -> None:
            payload = event.data or {}
            if payload.get("incident_id") == incident.id:
                peer_resolution.update(payload)
                resolved_event.set()

        # Register a temporary handler for FEDERATION_ASSISTANCE_ACCEPTED
        handler_id: str | None = None
        try:
            if hasattr(self._synapse, "_event_bus") and hasattr(
                self._synapse._event_bus, "subscribe"
            ):
                await self._synapse._event_bus.subscribe(
                    SynapseEventType.FEDERATION_ASSISTANCE_ACCEPTED,
                    _on_peer_assistance,
                )
        except Exception:
            pass

        peer_accepted = False
        try:
            peer_accepted = await asyncio.wait_for(
                resolved_event.wait(),
                timeout=_FEDERATION_ESCALATION_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            pass
        finally:
            # Always deregister to avoid handler leaks
            if handler_id is not None:
                try:
                    if hasattr(self._synapse._event_bus, "unsubscribe"):
                        await self._synapse._event_bus.unsubscribe(handler_id)
                except Exception:
                    pass

        if not peer_accepted:
            self._logger.info(
                "federation_escalation_no_peer_response",
                incident_id=incident.id,
                timeout_s=_FEDERATION_ESCALATION_TIMEOUT_S,
            )
            self._emit_metric("thymos.federation_escalation.timeout", 1)
            return False

        self._logger.info(
            "federation_escalation_peer_resolved",
            incident_id=incident.id,
            peer_instance=peer_resolution.get("source_instance"),
            antibody_id=peer_resolution.get("antibody_id"),
        )
        self._emit_metric("thymos.federation_escalation.resolved_by_peer", 1)

        # Mark incident as resolved by federation
        incident.repair_status = RepairStatus.RESOLVED
        incident.repair_history.append(RepairAttempt(
            tier=RepairTier.ESCALATE,
            action="federation_peer_resolution",
            outcome="success",
            reason=(
                f"Resolved by federated peer "
                f"{peer_resolution.get('source_instance', 'unknown')} "
                f"using antibody {peer_resolution.get('antibody_id', 'unknown')}"
            ),
        ))
        return True

    async def _escalate_with_full_context(
        self, incident: Incident, diagnosis: Diagnosis | None = None,
    ) -> None:
        """
        Escalate an incident to human review with full diagnostic context.

        Called when all autonomous repair tiers have been exhausted or when
        the repair loop cannot proceed. Builds a RepairSpec for ESCALATE tier
        and delegates to _apply_escalation which includes rich context.
        """
        reason_parts: list[str] = ["All autonomous repair tiers exhausted."]
        if diagnosis is not None:
            if diagnosis.root_cause:
                reason_parts.append(f"Diagnosis: {diagnosis.root_cause[:200]}")
            if diagnosis.repair_tier is not None:
                reason_parts.append(
                    f"Last diagnosed tier: {diagnosis.repair_tier.name}"
                )
        if incident.repair_history:
            last = incident.repair_history[-1]
            reason_parts.append(
                f"Last attempt: {last.tier.name}/{last.action} → {last.outcome}"
            )

        escalation_repair = RepairSpec(
            tier=RepairTier.ESCALATE,
            action="human_escalation",
            target_system=incident.source_system,
            reason=" | ".join(reason_parts),
        )
        incident.repair_tier = RepairTier.ESCALATE
        await self._apply_escalation(incident, escalation_repair)

    # ─── Post-Repair Verification ────────────────────────────────────

    async def _verify_repair(self, incident: Incident, repair: RepairSpec) -> bool:
        """
        Verify that a repair actually fixed the problem.

        Two-phase verification:
          1. System liveness - is the target system healthy?
          2. Failure-specific re-probe - is the *original* failure condition
             still present?  Checks the deduplicator for any active incident
             sharing the same fingerprint.  This catches cases where liveness
             passes but the specific bug persists (not yet re-triggered or
             only intermittent).

        For escalations / NOOPs, verification is not applicable.
        """
        if repair.tier == RepairTier.ESCALATE:
            return True  # Escalation is inherently "successful"

        if repair.tier == RepairTier.NOOP:
            return True

        # Wait for the repair to take effect
        await asyncio.sleep(min(_POST_REPAIR_VERIFY_TIMEOUT_S, 5.0))

        # ── Phase 1: System liveness ──────────────────────────────────────
        if self._health_monitor is not None and repair.target_system is not None:
            record = self._health_monitor.get_record(repair.target_system)
            if record is not None:
                from systems.synapse.types import SystemStatus
                if record.status == SystemStatus.FAILED:
                    return False
                if (
                    record.status not in (SystemStatus.HEALTHY, SystemStatus.STARTING)
                    and record.consecutive_successes < 1
                ):
                    # Degraded/overloaded with no recent successes - ambiguous.
                    return False

        # ── Phase 2: Failure-specific re-probe ────────────────────────────
        # Regardless of system liveness, verify that no active incident with
        # the *same* fingerprint exists.  This is the real proof the failure
        # is gone - not just that the system is alive.
        if self._deduplicator is not None:
            active = self._deduplicator.get_active_incident(incident.fingerprint)
            if active is not None and active.id != incident.id:
                # A new incident was created for the same fingerprint
                # post-repair - the original failure condition persists.
                self._logger.warning(
                    "repair_verification_fingerprint_still_active",
                    incident_id=incident.id,
                    fingerprint=incident.fingerprint[:16],
                    new_incident_id=active.id,
                    tier=repair.tier.name,
                )
                self._emit_metric(
                    "thymos.repair_verification_failed",
                    1,
                    tags={
                        "tier": repair.tier.name,
                        "source_system": incident.source_system,
                    },
                )
                return False

        return True

    # ─── Learning ────────────────────────────────────────────────────

    async def _learn_from_success(
        self,
        incident: Incident,
        repair: RepairSpec,
        diagnosis: Diagnosis,
    ) -> None:
        """
        A repair succeeded. Crystallize it into an antibody and feed Evo.

        This is where genuine adaptive immunity happens: the organism
        gets harder to break over time.
        """
        assert self._antibody_library is not None

        # If this was an antibody application, record success
        if incident.antibody_id is not None:
            await self._antibody_library.record_outcome(
                incident.antibody_id,
                success=True,
            )
        elif repair.tier >= RepairTier.PARAMETER:
            # For Tier 2+ repairs (not NOOP/transient), create a new antibody
            antibody = await self._antibody_library.create_from_repair(
                incident=incident,
                repair=repair,
            )
            self._total_antibodies_created += 1
            self._emit_metric("thymos.antibodies.created", 1)

            self._logger.info(
                "antibody_crystallized",
                antibody_id=antibody.id,
                fingerprint=incident.fingerprint[:16],
                tier=repair.tier.name,
            )

            # Emit ANTIBODY_CREATED lifecycle event
            await self._emit_event(
                SynapseEventType.ANTIBODY_CREATED,
                {
                    "antibody_id": antibody.id,
                    "fingerprint": incident.fingerprint,
                    "incident_class": incident.incident_class.value,
                    "success_rate": 1.0,
                    "repair_steps": [repair.action[:200]],
                },
            )

            await self._emit_evolutionary_observable(
                "antibody_created", 1.0, is_novel=True,
                metadata={
                    "antibody_id": antibody.id,
                    "fingerprint": incident.fingerprint[:16],
                    "tier": repair.tier.name,
                },
            )

            # Closure Loop 4: emit IMMUNE_PATTERN_ADVISORY for Simula
            await self._emit_immune_advisory(incident, antibody)

        # Per-endpoint repair outcome metric (Skia monitoring / success-rate tracking)
        if isinstance(incident.context, ApiErrorContext):
            endpoint = incident.context.endpoint
            self._logger.info(
                "api_repair_succeeded",
                incident_id=incident.id,
                endpoint=endpoint,
                status_code=incident.context.status_code,
                tier=repair.tier.name,
                resolution_time_ms=incident.resolution_time_ms,
            )
            self._emit_metric(
                "thymos.api.repair_outcome",
                1.0,
                tags={
                    "endpoint": endpoint,
                    "outcome": "success",
                    "tier": repair.tier.name,
                    "status_code": str(incident.context.status_code),
                },
            )

        # Persist incident and repair node to Neo4j (spec §10.6, M7)
        await self._persist_incident(incident)
        await self._persist_repair_node(
            incident, repair, outcome="success",
            duration_ms=incident.resolution_time_ms,
        )

        # Broadcast repair success through the workspace so Evo can learn
        # the fix pattern and Simula can validate future proposals against it.
        await self._broadcast_repair_completed(incident, repair)

        # Schedule post-repair validation - monitors for fingerprint re-firing
        # and emits ATUNE_REPAIR_VALIDATION to close the Thymos↔Atune loop.
        await self._schedule_repair_validation(incident, repair)

        # VERIFIED: outcome fed to Evo via _feed_repair_to_evo → evo.process_episode(Episode)
        # Feed success to Evo so the learning system can accumulate
        # evidence about what repair strategies work
        if self._evo is not None:
            await self._feed_repair_to_evo(incident, repair, success=True)

        # ── Full repair episode RE training (SG2) ──
        # Measured outcome_quality: combines repair success, speed, and tier efficiency
        speed_factor = max(0.0, 1.0 - (incident.resolution_time_ms or 0) / 60_000)
        tier_efficiency = max(0.0, 1.0 - repair.tier.value * 0.15)
        measured_quality = min(1.0, 0.5 + speed_factor * 0.25 + tier_efficiency * 0.25)
        await self._emit_re_training_example(
            category="full_repair_episode",
            instruction=(
                "Given an incident detection, diagnosis, and repair strategy, "
                "evaluate the complete repair pipeline and learn from the outcome."
            ),
            input_context=(
                f"incident_class={incident.incident_class.value}, "
                f"source_system={incident.source_system}, "
                f"severity={incident.severity.value}, "
                f"root_cause={incident.root_cause_hypothesis or 'unknown'}, "
                f"repair_tier={repair.tier.name}, "
                f"attempts={len(incident.repair_history)}"
            ),
            output=(
                f"outcome=SUCCESS, resolution_ms={incident.resolution_time_ms}, "
                f"action={repair.action[:300]}, "
                f"antibody_created={incident.antibody_id is None and repair.tier >= RepairTier.PARAMETER}"
            ),
            outcome_quality=measured_quality,
            latency_ms=incident.resolution_time_ms or 0,
            reasoning_trace=(
                f"Root cause: {incident.root_cause_hypothesis or 'unknown'}. "
                f"Repair: {repair.reason[:200]}"
            ),
            episode_id=incident.id,
            constitutional_alignment=self._build_incident_alignment(
                care=0.8,
                coherence=measured_quality * 2.0 - 1.0,
                growth=0.4,
            ),
        )

    async def _learn_from_failure(
        self,
        incident: Incident,
        repair: RepairSpec,
    ) -> None:
        """A repair failed. Record failure and update antibody if applicable."""
        assert self._antibody_library is not None

        if incident.antibody_id is not None:
            await self._antibody_library.record_outcome(
                incident.antibody_id,
                success=False,
            )

        # Persist failed repair attempt to Neo4j so future antibody selection
        # can avoid repeating the same failed strategy for this fingerprint.
        if self._neo4j is not None and incident.antibody_id is None:
            try:
                await self._neo4j.execute_write(
                    """
                    CREATE (f:FailedRepairAttempt {
                        fingerprint:    $fingerprint,
                        incident_id:    $incident_id,
                        repair_tier:    $repair_tier,
                        repair_action:  $repair_action,
                        source_system:  $source_system,
                        error_type:     $error_type,
                        recorded_at:    $recorded_at
                    })
                    """,
                    {
                        "fingerprint": incident.fingerprint,
                        "incident_id": incident.id,
                        "repair_tier": repair.tier.name,
                        "repair_action": repair.action,
                        "source_system": incident.source_system,
                        "error_type": incident.error_type,
                        "recorded_at": utc_now().isoformat(),
                    },
                )
            except Exception as exc:
                self._logger.debug("failed_repair_persist_error", error=str(exc))

        # Per-endpoint failure metric for Skia monitoring
        if isinstance(incident.context, ApiErrorContext):
            endpoint = incident.context.endpoint
            self._logger.warning(
                "api_repair_failed",
                incident_id=incident.id,
                endpoint=endpoint,
                status_code=incident.context.status_code,
                tier=repair.tier.name,
            )
            self._emit_metric(
                "thymos.api.repair_outcome",
                0.0,
                tags={
                    "endpoint": endpoint,
                    "outcome": "failure",
                    "tier": repair.tier.name,
                    "status_code": str(incident.context.status_code),
                },
            )

        # Persist the failed incident and rollback repair node for post-mortem (M7)
        await self._persist_incident(incident)
        await self._persist_repair_node(
            incident, repair, outcome="rolled_back",
        )

        # Feed failure to Evo - failures are more salient than successes
        # and drive hypothesis formation about system vulnerabilities
        if self._evo is not None:
            await self._feed_repair_to_evo(incident, repair, success=False)

        # ── Full repair episode RE training - failure path (SG2) ──
        # Failed repairs are MORE salient - the RE must learn what doesn't work.
        await self._emit_re_training_example(
            category="full_repair_episode",
            instruction=(
                "Given an incident detection, diagnosis, and repair strategy, "
                "evaluate the complete repair pipeline and learn from the failure."
            ),
            input_context=(
                f"incident_class={incident.incident_class.value}, "
                f"source_system={incident.source_system}, "
                f"severity={incident.severity.value}, "
                f"root_cause={incident.root_cause_hypothesis or 'unknown'}, "
                f"repair_tier={repair.tier.name}, "
                f"attempts={len(incident.repair_history)}"
            ),
            output=(
                f"outcome=FAILURE, action={repair.action[:300]}, "
                f"reason={repair.reason[:200]}"
            ),
            outcome_quality=max(0.0, 0.1 - len(incident.repair_history) * 0.02),
            latency_ms=incident.resolution_time_ms or 0,
            reasoning_trace=(
                f"Root cause: {incident.root_cause_hypothesis or 'unknown'}. "
                f"Failed repair: {repair.reason[:200]}"
            ),
            episode_id=incident.id,
            constitutional_alignment=self._build_incident_alignment(
                care=-0.3,      # failure = unresolved harm
                coherence=-0.4, # failure = incoherence persists
            ),
        )

        # ── Gap 1: Auto-seed CrashPattern when no pattern matched but ≥2 tiers failed ──
        # If the incident already matched a pattern, update_on_failure() is called by the
        # caller (process_incident) via CRASH_PATTERN_REINFORCED.  Only seed a *new* pattern
        # when an incident has stumped ≥2 different tiers and is not yet catalogued.
        if (
            incident.matched_pattern_id is None
            and self._crash_pattern_analyzer is not None
            and len(incident.repair_history) >= 2
        ):
            failed_tier_names = {
                attempt.tier for attempt in incident.repair_history if attempt.outcome != "success"
            }
            if len(failed_tier_names) >= 2:
                # Convert RepairTier enum values to their .name strings before storing.
                # CrashPattern.failed_tiers is list[str] and the PatternAwareRouter
                # checks `t.name not in failed_tiers` - storing enum objects would
                # always compare unequal, making all auto-seeded patterns behave as if
                # no tiers have failed.
                failed_tier_strings = sorted(t.name for t in failed_tier_names)
                features = CrashPatternAnalyzer.extract_features(
                    source_system=incident.source_system,
                    incident_class=incident.incident_class.value,
                    error_type=incident.error_type or "",
                    error_message=incident.error_message or "",
                    affected_systems=incident.affected_systems or [],
                )
                asyncio.ensure_future(
                    self._crash_pattern_analyzer.register_pattern(
                        features=features,
                        description=(
                            f"Auto-seeded from incident {incident.id}: "
                            f"{incident.source_system}/{incident.incident_class.value} "
                            f"failed at tiers {failed_tier_strings}"
                        ),
                        initial_confidence=0.5,
                        failed_tiers=failed_tier_strings,
                    )
                )
                self._logger.info(
                    "crash_pattern_auto_seeded",
                    incident_id=incident.id,
                    source_system=incident.source_system,
                    failed_tiers=failed_tier_strings,
                    feature_count=len(features),
                )
                # Emit CRASH_PATTERN_CONFIRMED: the organism has distilled a new failure law.
                # Auto-seeded patterns start at confidence=0.5 (below 0.70 threshold but
                # still meaningful as first confirmation of a recurring failure).
                pattern_id_auto = CrashPattern.make_id(features)
                if hasattr(SynapseEventType, "CRASH_PATTERN_CONFIRMED"):
                    asyncio.ensure_future(self._emit_event(
                        SynapseEventType.CRASH_PATTERN_CONFIRMED,  # type: ignore[attr-defined]
                        {
                            "pattern_id": pattern_id_auto,
                            "confidence": 0.5,
                            "lesson": (
                                f"{incident.source_system}/{incident.incident_class.value} "
                                f"resists repair at tiers: {failed_tier_strings}"
                            ),
                            "example_count": len(incident.repair_history),
                            "failed_tiers": failed_tier_strings,
                            "incident_id": incident.id,
                            "source": "thymos_auto_seed",
                        },
                    ))

    async def _persist_incident(self, incident: Incident) -> None:
        """Persist an incident to Neo4j for the causal knowledge graph."""
        if self._neo4j is None:
            return

        try:
            await self._neo4j.execute_write(
                """
                MERGE (i:Incident {id: $id})
                SET i.source_system = $source_system,
                    i.incident_class = $incident_class,
                    i.severity = $severity,
                    i.fingerprint = $fingerprint,
                    i.error_type = $error_type,
                    i.error_message = $error_message,
                    i.repair_status = $repair_status,
                    i.repair_tier = $repair_tier,
                    i.repair_successful = $repair_successful,
                    i.resolution_time_ms = $resolution_time_ms,
                    i.root_cause = $root_cause,
                    i.timestamp = $timestamp
                """,
                {
                    "id": incident.id,
                    "source_system": incident.source_system,
                    "incident_class": incident.incident_class.value,
                    "severity": incident.severity.value,
                    "fingerprint": incident.fingerprint,
                    "error_type": incident.error_type,
                    "error_message": incident.error_message[:500],
                    "repair_status": incident.repair_status.value,
                    "repair_tier": incident.repair_tier.name if incident.repair_tier else "unknown",
                    "repair_successful": incident.repair_successful,
                    "resolution_time_ms": incident.resolution_time_ms,
                    "root_cause": incident.root_cause_hypothesis or "",
                    "timestamp": incident.timestamp.isoformat(),
                },
            )
        except Exception as exc:
            self._logger.debug("incident_persist_failed", error=str(exc))

    async def _persist_repair_node(
        self,
        incident: Incident,
        repair: RepairSpec,
        outcome: str,
        duration_ms: int | None = None,
    ) -> None:
        """
        Persist a (:Repair) node to Neo4j linked to its incident.

        Spec §10.6 - Memory Writes:
          (:Repair {id, tier, success, duration_ms, ...})
          (:Incident)-[:REPAIRED_WITH]->(:Repair)

        Called for both successful repairs and rollbacks (M7).
        """
        if self._neo4j is None:
            return

        from primitives.common import new_id

        repair_id = new_id()
        try:
            await self._neo4j.execute_write(
                """
                CREATE (r:Repair {
                    id:            $id,
                    incident_id:   $incident_id,
                    tier:          $tier,
                    action:        $action,
                    outcome:       $outcome,
                    duration_ms:   $duration_ms,
                    target_system: $target_system,
                    reason:        $reason,
                    timestamp:     $timestamp
                })
                WITH r
                MATCH (i:Incident {id: $incident_id})
                MERGE (i)-[:REPAIRED_WITH]->(r)
                """,
                {
                    "id": repair_id,
                    "incident_id": incident.id,
                    "tier": repair.tier.name,
                    "action": repair.action[:200],
                    "outcome": outcome,
                    "duration_ms": duration_ms,
                    "target_system": repair.target_system or incident.source_system,
                    "reason": (repair.reason or "")[:300],
                    "timestamp": utc_now().isoformat(),
                },
            )
            self._logger.debug(
                "repair_node_persisted",
                repair_id=repair_id,
                incident_id=incident.id,
                tier=repair.tier.name,
                outcome=outcome,
            )
        except Exception as exc:
            self._logger.debug("repair_node_persist_failed", error=str(exc))

    # ─── Cross-System Feedback ─────────────────────────────────────────

    async def _inject_repair_goal(
        self,
        incident: Incident,
        repair_tier: RepairTier | None,
        resolved: bool,
    ) -> None:
        """
        Inject a self-repair goal into Nova's goal manager.

        Pre-repair (resolved=False): high-urgency goal so Nova prioritises self-healing.
        Post-repair (resolved=True): follow-up monitoring goal at lower urgency.

        AV1 migration: emits NOVA_GOAL_INJECTED via Synapse instead of calling
        self._nova.add_goal() directly.
        """
        from primitives.common import new_id

        tier_name = repair_tier.name if repair_tier else "UNKNOWN"

        if resolved:
            desc = (
                f"Monitor system recovery: {incident.source_system} "
                f"after {tier_name} repair"
            )
            priority, urgency = 0.6, 0.4
        else:
            desc = (
                f"Urgent: self-repair {incident.source_system} - "
                f"{incident.incident_class.value} incident ({tier_name})"
            )
            priority, urgency = 0.9, 0.85

        goal_id = new_id()
        # AV1 migration: emit NOVA_GOAL_INJECTED via Synapse instead of
        # calling self._nova.add_goal() directly. Nova subscribes to this event.
        await self._emit_event(
            SynapseEventType.NOVA_GOAL_INJECTED,
            {
                "goal_id": goal_id,
                "description": desc,
                "source": "maintenance",
                "priority": priority,
                "urgency": urgency,
                "importance": 0.7,
                "drive_alignment": {
                    "coherence": 0.8,
                    "care": 0.1,
                    "growth": 0.0,
                    "honesty": 0.1,
                },
                "incident_id": incident.id,
                "resolved": resolved,
            },
        )
        self._logger.info(
            "repair_goal_injected",
            goal_id=goal_id,
            incident_id=incident.id,
            resolved=resolved,
        )

        # Schedule belief stabilisation check - emits NOVA_BELIEF_STABILISED
        # after monitoring whether Nova's beliefs re-converge post-repair.
        if resolved:
            self._supervised_fire_and_forget(
                self._check_nova_belief_stabilisation(
                    incident_id=incident.id,
                    goal_id=goal_id,
                ),
                name=f"thymos_nova_stab_{incident.id[:8]}",
            )

    async def _check_nova_belief_stabilisation(
        self,
        incident_id: str,
        goal_id: str,
    ) -> None:
        """
        Monitor Nova after repair goal injection to determine if beliefs stabilised.

        Waits 30 seconds, then checks if the source system has recovered
        (no new incidents from the same fingerprint) and if Nova's belief updater
        is in a stable state. Emits NOVA_BELIEF_STABILISED to close the loop.
        """
        await asyncio.sleep(30.0)

        if self._synapse is None:
            return

        # Check for re-firing of the original incident's source system
        incident = None
        for buf_incident in self._incident_buffer:
            if buf_incident.id == incident_id:
                incident = buf_incident
                break

        # Determine stability by checking if same fingerprint re-fired
        stable = True
        beliefs_affected = 0
        if incident is not None:
            reappeared = sum(
                1 for i in self._incident_buffer
                if (
                    i.fingerprint == incident.fingerprint
                    and i.id != incident_id
                    and i.timestamp > incident.timestamp
                )
            )
            stable = reappeared == 0
            beliefs_affected = reappeared

        convergence_time_ms = 30_000  # monitoring window

        try:
            await self._synapse._event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.NOVA_BELIEF_STABILISED,
                    source_system="thymos",
                    data={
                        "incident_id": incident_id,
                        "goal_id": goal_id,
                        "beliefs_affected": beliefs_affected,
                        "convergence_time_ms": convergence_time_ms,
                        "stable": stable,
                    },
                )
            )
        except Exception as exc:
            self._logger.debug(
                "nova_belief_stabilisation_emit_failed", error=str(exc),
            )

    async def _feed_repair_to_evo(
        self,
        incident: Incident,
        repair: RepairSpec,
        success: bool,
    ) -> None:
        """
        Feed a repair outcome to Evo as a learning episode.

        Successful repairs teach the organism what works.
        Failed repairs teach it what doesn't - and are more salient.
        """
        from primitives.common import new_id, utc_now
        from primitives.memory_trace import Episode

        outcome_text = "succeeded" if success else "failed"
        episode = Episode(
            id=new_id(),
            source=f"thymos.repair_{outcome_text}",
            raw_content=(
                f"Repair {outcome_text}: {repair.action} on {repair.target_system}. "
                f"Tier: {repair.tier.name}. "
                f"Incident class: {incident.incident_class.value}. "
                f"Root cause: {incident.root_cause_hypothesis or 'unknown'}"
            ),
            summary=(
                f"Thymos {repair.tier.name} repair {outcome_text}: "
                f"{repair.target_system}"
            ),
            salience_composite=0.6 if success else 0.8,
            affect_valence=0.2 if success else -0.3,
            event_time=utc_now(),
        )
        # AV1 migration: emit via Synapse instead of calling self._evo.process_episode()
        await self._emit_event(
            SynapseEventType.EVOLUTIONARY_OBSERVABLE,
            {
                "episode_id": episode.id,
                "source": episode.source,
                "raw_content": episode.raw_content,
                "summary": episode.summary,
                "salience_composite": episode.salience_composite,
                "affect_valence": episode.affect_valence,
                "event_time": episode.event_time.isoformat(),
                "kind": "repair_outcome",
                "incident_id": incident.id,
                "success": success,
                "tier": repair.tier.name,
            },
        )
        self._logger.info(
            "repair_outcome_fed_to_evo",
            incident_id=incident.id,
            success=success,
            tier=repair.tier.name,
        )

    # ─── Repair Broadcast ─────────────────────────────────────────────

    async def _broadcast_repair_completed(
        self,
        incident: Incident,
        repair: RepairSpec,
    ) -> None:
        """
        Emit REPAIR_COMPLETED on the Synapse event bus so Evo can extract
        repair patterns and Simula can validate future proposals against them.

        Only fires for Tier 2+ repairs (PARAMETER and above) - NOOP and transient
        retries produce no learnable pattern.
        """
        if self._synapse is None:
            return
        if repair.tier < RepairTier.PARAMETER:
            return

        endpoint = ""
        if isinstance(incident.context, ApiErrorContext):
            endpoint = incident.context.endpoint

        fix_summary = (
            f"{repair.tier.name} repair applied to {incident.source_system}: "
            f"{repair.action}. Root cause: {incident.root_cause_hypothesis or 'unknown'}"
        )
        if endpoint:
            fix_summary = f"[{endpoint}] {fix_summary}"

        # Deterministic repair-pattern ID: same class+fix always produces the same ID
        # so the hypothesis engine can deduplicate across incidents with identical root causes.
        repair_spec_id = RepairSpec.make_id(incident.incident_class.value, repair.action)

        try:
            await self._synapse._event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.REPAIR_COMPLETED,
                    source_system="thymos",
                    data={
                        "repair_id": repair_spec_id,
                        "repair_spec_id": repair_spec_id,
                        "incident_id": incident.id,
                        "endpoint": endpoint,
                        "tier": repair.tier.name,
                        "incident_class": incident.incident_class.value,
                        "fix_type": repair.action,
                        "root_cause": incident.root_cause_hypothesis or "",
                        "antibody_id": incident.antibody_id,
                        "cost_usd": 0.0,
                        "duration_ms": incident.resolution_time_ms or 0,
                        "fix_summary": fix_summary,
                    },
                )
            )
            self._logger.info(
                "repair_completed_broadcast",
                incident_id=incident.id,
                tier=repair.tier.name,
                endpoint=endpoint or "(none)",
            )
        except Exception as exc:
            self._logger.debug("repair_broadcast_failed", error=str(exc))

    async def _schedule_repair_validation(
        self,
        incident: Incident,
        repair: RepairSpec,
    ) -> None:
        """
        Schedule a post-repair validation check.

        After a repair is applied, monitors for 60 seconds to see if the same
        fingerprint re-fires. Emits ATUNE_REPAIR_VALIDATION so the organism
        confirms whether the repair actually helped.
        """
        if self._synapse is None:
            return
        if repair.tier < RepairTier.PARAMETER:
            return

        fingerprint = incident.fingerprint
        incident_id = incident.id
        salience_before = _SEVERITY_TO_SALIENCE.get(incident.severity, 0.1)

        async def _validate() -> None:
            await asyncio.sleep(60.0)  # monitoring window

            # Check if same fingerprint re-appeared in the incident buffer
            reappeared = any(
                i.fingerprint == fingerprint
                and i.id != incident_id
                and i.timestamp > incident.timestamp
                for i in self._incident_buffer
            )

            repair_effective = not reappeared
            salience_after = salience_before * (0.8 if reappeared else 0.1)

            try:
                if self._synapse is not None:
                    await self._synapse._event_bus.emit(
                        SynapseEvent(
                            event_type=SynapseEventType.ATUNE_REPAIR_VALIDATION,
                            source_system="thymos",
                            data={
                                "incident_id": incident_id,
                                "repair_effective": repair_effective,
                                "salience_before": salience_before,
                                "salience_after": salience_after,
                                "cycles_observed": 60,
                            },
                        )
                    )
            except Exception as exc:
                self._logger.debug(
                    "repair_validation_emit_failed", error=str(exc),
                )

        self._supervised_fire_and_forget(
            _validate(),
            name=f"thymos_validate_{incident_id[:8]}",
        )

    # ─── Percept Broadcasting ────────────────────────────────────────

    async def _broadcast_as_percept(self, incident: Incident) -> None:
        """
        Route high-severity incidents into Atune's workspace as Percepts.

        The organism perceives its own failures through the normal
        consciousness cycle. It hurts to break - and that's by design.
        Critical incidents get maximum salience; INFO incidents are barely noticed.

        AV1 migration: emits via Synapse instead of calling self._atune.contribute()
        directly. Atune subscribes to INCIDENT_DETECTED and routes to workspace.
        """
        salience = _SEVERITY_TO_SALIENCE.get(incident.severity, 0.1)

        # Only broadcast MEDIUM+ to avoid flooding the workspace
        if salience < 0.5:
            return

        await self._emit_event(
            SynapseEventType.INCIDENT_DETECTED,
            {
                "incident_id": incident.id,
                "incident_class": incident.incident_class.value,
                "severity": incident.severity.value,
                "source_system": incident.source_system,
                "description": incident.error_message[:500],
                "salience": salience,
                "reason": "immune_incident",
            },
        )

    # ─── Background Loops ────────────────────────────────────────────

    async def _raise_sentinel_internal_incident(
        self,
        sentinel_name: str,
        error: Exception,
    ) -> None:
        """
        P8 - Create a Thymos-internal incident when a sentinel itself throws.

        The sentinel failure is treated as a MEDIUM DEGRADATION incident so that
        the repair pipeline can attempt recovery (restart, parameter adjustment)
        rather than silently losing the sentinel's monitoring coverage.

        Iron rule: Thymos never suppresses its own errors.
        """
        import hashlib

        fp = hashlib.sha256(
            f"thymos:sentinel_failure:{sentinel_name}".encode()
        ).hexdigest()[:16]

        self._logger.warning(
            "sentinel_internal_failure",
            sentinel=sentinel_name,
            error=str(error)[:200],
            fingerprint=fp,
        )

        from systems.thymos.types import Incident, IncidentClass, IncidentSeverity

        incident = Incident(
            incident_class=IncidentClass.DEGRADATION,
            severity=IncidentSeverity.MEDIUM,
            fingerprint=fp,
            source_system="thymos",
            error_type="sentinel_failure",
            error_message=(
                f"{sentinel_name} raised during scan: {type(error).__name__}: "
                f"{str(error)[:200]}"
            ),
            context={"sentinel": sentinel_name, "error": str(error)[:500]},
        )
        try:
            await self.on_incident(incident)
        except Exception as inner:
            # Last resort: just log it - we cannot recurse infinitely
            self._logger.error(
                "sentinel_incident_creation_failed",
                sentinel=sentinel_name,
                error=str(inner)[:200],
            )

    async def _sentinel_scan_loop(self) -> None:
        """
        Periodic sentinel scans for proactive failure detection.

        Feedback loop sentinel checks which loops are transmitting.
        Cognitive stall sentinel checks workspace health.
        Drift sentinel is fed by Synapse metrics (not looped here).
        """
        await asyncio.sleep(10.0)  # Let the organism warm up

        while True:
            try:
                await asyncio.sleep(_SENTINEL_SCAN_INTERVAL_S)

                # P8 - Each sentinel runs in its own try/except.  If a sentinel
                # itself raises, we create a Thymos-internal incident so the
                # failure is visible and can be healed rather than silently lost.

                # ── FeedbackLoop sentinel ──────────────────────────────────
                try:
                    if self._feedback_loop_sentinel is not None:
                        if self._synapse is not None:
                            self._feedback_loop_sentinel.report_loop_active("rhythm_modulation")
                        if self._atune is not None:
                            self._feedback_loop_sentinel.report_loop_active("affect_expression")
                        if self._synapse is not None:
                            self._feedback_loop_sentinel.report_loop_active("goal_guided_attention")
                        if self._nova is not None:
                            self._feedback_loop_sentinel.report_loop_active("axon_outcome_beliefs")
                        if self._evo is not None:
                            self._feedback_loop_sentinel.report_loop_active("personality_evolution")
                        if self._synapse is not None:
                            self._feedback_loop_sentinel.report_loop_active("simula_version_params")

                        loop_incidents = self._feedback_loop_sentinel.check_loops()
                        for incident in loop_incidents:
                            await self.on_incident(incident)
                except Exception as exc:
                    await self._raise_sentinel_internal_incident(
                        sentinel_name="FeedbackLoopSentinel", error=exc
                    )

                # ── Deduplicator pruning (not a sentinel but co-located) ───
                try:
                    if self._deduplicator is not None:
                        pruned = self._deduplicator.prune_stale()
                        if pruned:
                            self._logger.debug("deduplicator_pruned", entries_removed=pruned)
                except Exception as exc:
                    self._logger.debug("deduplicator_prune_error", error=str(exc))

                # ── Protocol health sentinel ───────────────────────────────
                try:
                    if self._protocol_health_sentinel is not None:
                        if self._cached_oikos_snapshot:
                            positions = self._cached_oikos_snapshot.get("yield_positions", [])
                        elif self._oikos is not None:
                            state = self._oikos.snapshot()
                            positions = getattr(state, "yield_positions", [])
                        else:
                            positions = []
                        if positions:
                            health_incidents = (
                                self._protocol_health_sentinel.check_all_positions(
                                    positions, live_data={}
                                )
                            )
                            for incident in health_incidents:
                                await self.on_incident(incident)
                except Exception as exc:
                    await self._raise_sentinel_internal_incident(
                        sentinel_name="ProtocolHealthSentinel", error=exc
                    )

                # ── Bankruptcy sentinel ────────────────────────────────────
                try:
                    if self._bankruptcy_sentinel is not None:
                        deficit_usd: float = 0.0
                        if self._synapse is not None and hasattr(
                            self._synapse, "metabolic_deficit"
                        ):
                            deficit_usd = float(self._synapse.metabolic_deficit)
                        bankruptcy_incidents = self._bankruptcy_sentinel.check(
                            eth_balance=999.0,
                            deficit_usd=deficit_usd,
                        )
                        for incident in bankruptcy_incidents:
                            await self.on_incident(incident)
                except Exception as exc:
                    await self._raise_sentinel_internal_incident(
                        sentinel_name="BankruptcySentinel", error=exc
                    )

                # Emit IMMUNE_CYCLE_COMPLETE after each full sentinel scan
                try:
                    await self._emit_event(
                        SynapseEventType.IMMUNE_CYCLE_COMPLETE,
                        {
                            "cycle_timestamp": utc_now().isoformat(),
                            "active_incidents": len(self._active_incidents),
                            "antibody_count": (
                                len(self._antibody_library._library)
                                if self._antibody_library is not None
                                else 0
                            ),
                        },
                    )
                except Exception:
                    pass

            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._logger.warning(
                    "sentinel_scan_error",
                    error=str(exc),
                )

    async def _homeostasis_loop(self) -> None:
        """
        Proactive homeostatic regulation.

        Runs continuously on MAINTAIN cycle step timing. Checks metrics
        against optimal ranges and makes small preemptive adjustments.

        This is the organism's thermostat - it maintains optimal operating
        conditions without waiting for something to break.
        """
        await asyncio.sleep(30.0)  # Let the organism stabilize

        while True:
            try:
                await asyncio.sleep(_HOMEOSTASIS_INTERVAL_S)

                if self._homeostasis_controller is None:
                    continue

                adjustments = self._homeostasis_controller.check_homeostasis()

                for adjustment in adjustments:
                    self._total_homeostatic_adjustments += 1
                    self._emit_metric("thymos.homeostasis.adjustments", 1)

                    self._logger.info(
                        "homeostatic_adjustment",
                        metric=adjustment.metric_name,
                        current=f"{adjustment.current_value:.2f}",
                        trend=adjustment.trend_direction,
                        adjustment_path=adjustment.adjustment.parameter_path,
                        adjustment_delta=adjustment.adjustment.delta,
                    )

                    # Apply the Tier 1 parameter nudge
                    param_repair = RepairSpec(
                        tier=RepairTier.PARAMETER,
                        action="homeostatic_adjustment",
                        target_system=adjustment.adjustment.parameter_path.split(".")[0],
                        reason=adjustment.adjustment.reason,
                        parameter_changes=[{
                            "parameter_path": adjustment.adjustment.parameter_path,
                            "delta": adjustment.adjustment.delta,
                            "reason": adjustment.adjustment.reason,
                        }],
                    )
                    await self._apply_parameter_repair(param_repair)

                    # Emit HOMEOSTASIS_ADJUSTED lifecycle event
                    await self._emit_event(
                        SynapseEventType.HOMEOSTASIS_ADJUSTED,
                        {
                            "parameter": adjustment.adjustment.parameter_path,
                            "old_value": adjustment.current_value,
                            "new_value": adjustment.current_value + adjustment.adjustment.delta,
                            "reason": adjustment.adjustment.reason,
                            "source_system": adjustment.adjustment.parameter_path.split(".")[0],
                        },
                    )

                # M8 - broadcast early drift signals to Nova / Telos BEFORE the
                # correction tier fires.  We use HOMEOSTASIS_ADJUSTED with
                # warn_only=True so downstream systems can observe the trend
                # without mistaking it for an applied parameter change.
                drift_warnings = self._homeostasis_controller.check_drift_warnings()
                for warn in drift_warnings:
                    self._logger.debug(
                        "homeostatic_drift_warning",
                        metric=warn["metric"],
                        current=f"{warn['current']:.3f}",
                        direction=warn["direction"],
                        proximity=warn["proximity"],
                    )
                    await self._emit_event(
                        SynapseEventType.HOMEOSTASIS_ADJUSTED,
                        {
                            "warn_only": True,
                            "metric": warn["metric"],
                            "current_value": warn["current"],
                            "direction": warn["direction"],
                            "boundary": warn["boundary"],
                            "proximity": warn["proximity"],
                            "trend_slope": warn["trend"],
                            "reason": (
                                f"Drift warning: {warn['metric']} trending "
                                f"{warn['direction']} toward boundary {warn['boundary']:.2f} "
                                f"(proximity={warn['proximity']:.2f})"
                            ),
                        },
                    )

                # Check storm mode exit
                if self._governor is not None:
                    self._governor.check_storm_exit()

            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._logger.warning(
                    "homeostasis_loop_error",
                    error=str(exc),
                )

    # ─── Telemetry Emission Loop (SG3, SG5) ─────────────────────────

    async def _telemetry_emission_loop(self) -> None:
        """
        Periodic emission of THYMOS_DRIVE_PRESSURE (every 30s) and
        THYMOS_VITALITY_SIGNAL (every 60s) for Telos and vitality monitoring.
        """
        await asyncio.sleep(15.0)  # Stagger from other loops
        tick = 0

        while True:
            try:
                await asyncio.sleep(30.0)
                tick += 1

                # ── THYMOS_DRIVE_PRESSURE (every 30s) ──
                ds = self._drive_state
                overall = (ds.coherence + ds.care + ds.growth + ds.honesty) / 4.0
                await self._emit_event(
                    SynapseEventType.THYMOS_DRIVE_PRESSURE,
                    {
                        "coherence": round(ds.coherence, 4),
                        "care": round(ds.care, 4),
                        "growth": round(ds.growth, 4),
                        "honesty": round(ds.honesty, 4),
                        "overall_pressure": round(overall, 4),
                        "timestamp": utc_now().isoformat(),
                    },
                )

                # ── THYMOS_VITALITY_SIGNAL (every 60s - every 2nd tick) ──
                if tick % 2 == 0:
                    total_repairs = self._total_repairs_succeeded + self._total_repairs_failed
                    failure_rate = (
                        self._total_repairs_failed / total_repairs
                        if total_repairs > 0
                        else 0.0
                    )
                    active_count = len(self._active_incidents)
                    ab_count = (
                        self._antibody_library.active_count
                        if self._antibody_library is not None
                        else 0
                    )
                    mean_resolution = (
                        sum(self._resolution_times) / len(self._resolution_times)
                        if self._resolution_times
                        else 0.0
                    )
                    storm_active = (
                        self._governor.healing_mode == HealingMode.STORM
                        if self._governor is not None
                        else False
                    )
                    # Overall health: 1.0 = perfect, 0.0 = failing
                    health = max(0.0, 1.0 - failure_rate - (0.1 if storm_active else 0.0))

                    await self._emit_event(
                        SynapseEventType.THYMOS_VITALITY_SIGNAL,
                        {
                            "healing_failure_rate": round(failure_rate, 4),
                            "active_incidents": active_count,
                            "storm_active": storm_active,
                            "antibody_count": ab_count,
                            "mean_repair_duration_ms": round(mean_resolution, 1),
                            "overall_health": round(health, 4),
                        },
                    )

            except asyncio.CancelledError:
                return
            except Exception as exc:
                self._logger.debug("telemetry_emission_error", error=str(exc))

    # ─── Prophylactic Scanner ────────────────────────────────────────

    async def scan_files(self, files_changed: list[str]) -> list[dict[str, Any]]:
        """
        Pre-deployment prophylactic scan.

        Checks new or modified files against the antibody library's
        error patterns. Returns warnings for code that matches known
        failure signatures.
        """
        if self._prophylactic_scanner is None:
            return []

        warnings = await self._prophylactic_scanner.scan(files_changed)
        self._total_prophylactic_scans += 1
        self._total_prophylactic_warnings += len(warnings)

        self._emit_metric("thymos.prophylactic.scans", 1)
        self._emit_metric("thymos.prophylactic.warnings", len(warnings))

        return [w.model_dump() for w in warnings]

    # ─── Exception Sentinel (Public API) ─────────────────────────────

    async def report_exception(
        self,
        system_id: str,
        exception: Exception,
        context: dict[str, Any] | None = None,
        *,
        endpoint: str = "",
        method: str = "",
        status_code: int = 0,
        request_context: ApiErrorContext | dict[str, Any] | None = None,
    ) -> None:
        """
        Public API for systems to report unhandled exceptions.

        Called by systems' own error handlers or by a global exception hook.
        Creates an Incident and routes it through the immune pipeline.

        API-layer callers may supply HTTP metadata via ``request_context``
        (preferred) or via the convenience kwargs ``endpoint``, ``method``,
        and ``status_code``.  All new parameters are optional - existing call
        sites continue to work without change.
        """
        if self._exception_sentinel is None:
            return

        # ── Resolve ApiErrorContext ──────────────────────────────────
        resolved_api_ctx: ApiErrorContext | None = None
        if isinstance(request_context, ApiErrorContext):
            resolved_api_ctx = request_context
        elif isinstance(request_context, dict) and request_context:
            with contextlib.suppress(Exception):
                resolved_api_ctx = ApiErrorContext(**request_context)

        # If caller passed convenience kwargs, synthesise a minimal context
        if resolved_api_ctx is None and (endpoint or method or status_code):
            resolved_api_ctx = ApiErrorContext(
                endpoint=endpoint,
                method=method.upper() if method else "",
                status_code=status_code,
                request_id="",
                remote_addr="",
                latency_ms=0.0,
            )

        # ── Intercept ────────────────────────────────────────────────
        incident = self._exception_sentinel.intercept(
            system_id=system_id,
            method_name="unknown",
            exception=exception,
            context=context or {},  # raw dict for sentinel (backward-compat)
        )

        # Replace context with structured ApiErrorContext when available
        if resolved_api_ctx is not None:
            incident.context = resolved_api_ctx

        # ── API-aware severity override ──────────────────────────────
        if resolved_api_ctx is not None:
            sc = resolved_api_ctx.status_code
            if sc in (400, 404):
                incident.severity = IncidentSeverity.LOW
                incident.incident_class = IncidentClass.DEGRADATION
            elif sc >= 500:
                incident.severity = IncidentSeverity.HIGH
                incident.incident_class = IncidentClass.CRASH

        # ── Dedup key: endpoint+method+error_class for distinct treatment ──
        # Include error class so different exceptions on the same endpoint
        # get distinct fingerprints (e.g. ImportError vs AttributeError on /api/v1/logos/health).
        if resolved_api_ctx is not None and resolved_api_ctx.endpoint:
            error_class = type(exception).__name__
            dedup_key = (
                f"{resolved_api_ctx.method.upper()}:{resolved_api_ctx.endpoint}"
                f":{error_class}"
            )
            incident.fingerprint = hashlib.sha256(
                f"api:{system_id}:{dedup_key}".encode()
            ).hexdigest()[:16]

        # ── Diagnostic hints ────────────────────────────────────────
        if resolved_api_ctx is not None:
            hint = _api_diagnostic_hint(resolved_api_ctx, exception)
            if hint:
                incident.root_cause_hypothesis = hint

        await self.on_incident(incident)

    # ─── Contract Sentinel (Public API) ──────────────────────────────

    async def report_contract_violation(
        self,
        source: str,
        target: str,
        operation: str,
        latency_ms: float,
        sla_ms: float,
    ) -> None:
        """
        Public API for reporting inter-system contract violations.

        Called by systems when an operation exceeds its SLA.
        """
        if self._contract_sentinel is None:
            return

        incident = self._contract_sentinel.check_contract(
            source=source,
            target=target,
            operation=operation,
            latency_ms=latency_ms,
        )
        if incident is not None:
            await self.on_incident(incident)

    # ─── Drift Sentinel (Public API) ─────────────────────────────────

    def record_metric(self, metric_name: str, value: float) -> None:
        """
        Feed a metric observation to the drift sentinel.

        This is the public API that external systems call to push measurements
        into Thymos's statistical process control layer.  Any metric name
        registered in ``sentinels.DEFAULT_DRIFT_METRICS`` is tracked; unknown
        metric names are silently ignored.

        **What needs to call this (DriftSentinel wiring guide)**:

        Synapse's clock loop should call ``thymos.record_metric()`` after each
        completed cycle with the following metric names and values:

          - ``"synapse.cycle.latency_ms"``      - wall-clock ms for the full cycle
          - ``"synapse.resources.memory_mb"``   - current RSS in MB

        Individual systems should call it after their own hot-path operations:

          - ``"memory.retrieval.latency_ms"``   - ms for a Memory.retrieve() call
          - ``"atune.salience.processing_ms"``  - ms for Atune salience pass
          - ``"atune.coherence.phi"``            - Φ (phi) from IIT coherence computation
          - ``"nova.efe.computation_ms"``        - ms for Nova's EFE computation
          - ``"voxis.generation.latency_ms"``   - ms for a Voxis expression generation
          - ``"evo.self_model.success_rate"``   - rolling success rate (0.0–1.0)

        Calling pattern (from Synapse's cycle loop)::

            # After timing a full cognitive cycle:
            thymos.record_metric("synapse.cycle.latency_ms", elapsed_ms)
            thymos.record_metric("synapse.resources.memory_mb", rss_mb)

        All calls are synchronous and non-blocking - the drift sentinel updates
        its EMA baseline in O(1) and fires incidents via a background task only
        when a threshold is exceeded.
        """
        if self._drift_sentinel is None:
            return

        # Feed homeostasis controller for proactive range regulation
        if self._homeostasis_controller is not None:
            self._homeostasis_controller.record_metric(metric_name, value)

        incident = self._drift_sentinel.record_metric(metric_name, value)
        if incident is not None:
            # Fire-and-forget: drift incidents are LOW priority
            self._supervised_fire_and_forget(
                self.on_incident(incident),
                name=f"thymos_drift_{metric_name}",
            )

        # Also feed the temporal correlator
        if self._temporal_correlator is not None and self._drift_sentinel is not None:
            baseline = self._drift_sentinel._baselines.get(metric_name)
            if baseline is not None and baseline.is_warmed_up:
                self._temporal_correlator.record_metric_anomaly(
                    metric_name=metric_name,
                    value=value,
                    baseline=baseline.mean,
                    z_score=baseline.z_score(value),
                )

    # ─── Health ──────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """
        Health check for Thymos (required by Synapse health monitor).

        Returns a snapshot of immune system status, counters, and budget.
        """
        governor_budget = (
            self._governor.budget_state if self._governor else None
        )

        antibody_count = len(self._antibody_library._all) if self._antibody_library else 0
        mean_effectiveness = 0.0
        if self._antibody_library and self._antibody_library._all:
            effectivenesses = [
                a.effectiveness for a in self._antibody_library._all.values()
                if not a.retired
            ]
            if effectivenesses:
                mean_effectiveness = sum(effectivenesses) / len(effectivenesses)

        mean_resolution = 0.0
        if self._resolution_times:
            mean_resolution = sum(self._resolution_times) / len(self._resolution_times)

        mean_confidence = 0.0
        if self._diagnosis_confidences:
            mean_confidence = sum(self._diagnosis_confidences) / len(self._diagnosis_confidences)

        mean_diag_latency = 0.0
        if self._diagnosis_latencies:
            mean_diag_latency = sum(self._diagnosis_latencies) / len(self._diagnosis_latencies)

        homeostasis_ranges = 0
        if self._homeostasis_controller is not None:
            homeostasis_ranges = self._homeostasis_controller.metrics_in_range

        return {
            "status": "healthy" if self._initialized else "not_initialized",
            "initialized": self._initialized,
            "healing_mode": (
                self._governor.healing_mode.value if self._governor else "unknown"
            ),
            # Incidents
            "total_incidents": self._total_incidents,
            "active_incidents": len(self._active_incidents),
            "mean_resolution_ms": round(mean_resolution, 1),
            "incidents_by_severity": dict(self._incidents_by_severity),
            "incidents_by_class": dict(self._incidents_by_class),
            # Antibodies
            "total_antibodies": antibody_count,
            "mean_antibody_effectiveness": round(mean_effectiveness, 3),
            "antibodies_applied": self._total_antibodies_applied,
            "antibodies_created": self._total_antibodies_created,
            "antibodies_retired": (
                self._antibody_library.retired_count if self._antibody_library else 0
            ),
            # Repairs
            "repairs_attempted": self._total_repairs_attempted,
            "repairs_succeeded": self._total_repairs_succeeded,
            "repairs_failed": self._total_repairs_failed,
            "repairs_rolled_back": self._total_repairs_rolled_back,
            "repairs_by_tier": dict(self._repairs_by_tier),
            # Diagnosis
            "total_diagnoses": self._total_diagnoses,
            "mean_diagnosis_confidence": round(mean_confidence, 3),
            "mean_diagnosis_latency_ms": round(mean_diag_latency, 1),
            # Homeostasis
            "homeostatic_adjustments": self._total_homeostatic_adjustments,
            "metrics_in_range": homeostasis_ranges,
            "metrics_total": (
                self._homeostasis_controller.metrics_total
                if self._homeostasis_controller is not None
                else 0
            ),
            # Storm
            "storm_activations": (
                self._governor.storm_activations if self._governor else 0
            ),
            # Prophylactic
            "prophylactic_scans": self._total_prophylactic_scans,
            "prophylactic_warnings": self._total_prophylactic_warnings,
            # Budget
            "budget": governor_budget.model_dump() if governor_budget else {},
            "budget_exhausted": (
                self._governor.is_budget_exhausted if self._governor else False
            ),
            "degraded_reason": (
                self._governor.degraded_reason if self._governor else None
            ),
            # Drive state (accumulated pressure from incidents + Equor rejections)
            "drive_state": self._drive_state.as_dict(),
        }

    # ─── Health Summary (Soma INTEGRITY feed) ───────────────────────

    @property
    def current_health_score(self) -> float:
        """
        Synchronous scalar health summary for Soma's INTEGRITY dimension.

        This is the output pathway Thymos exposes to the organism - a single
        0.0–1.0 value Soma reads every theta cycle. Thymos retains all sentinel
        logic and diagnostic detail internally; this property is the distilled
        summary that flows outward.

        Score logic:
          - Storm mode                     → 0.0   (immune system overwhelmed)
          - Each active HIGH incident       → -0.15
          - Each active CRITICAL incident   → -0.30
          - Each active MEDIUM incident     → -0.05
          - Result clamped to [0.0, 1.0]
        """
        if self._governor is not None and self._governor.healing_mode.value == "storm":
            return 0.0

        score = 1.0

        # Degraded mode: immune system alive but impaired - cap at 0.3
        if (
            self._governor is not None
            and self._governor.healing_mode == HealingMode.DEGRADED
        ):
            score = min(score, 0.3)

        for incident in self._active_incidents.values():
            sev = (
                incident.severity.value
                if hasattr(incident.severity, "value")
                else str(incident.severity)
            )
            if sev == "critical":
                score -= 0.30
            elif sev == "high":
                score -= 0.15
            elif sev == "medium":
                score -= 0.05

        return max(0.0, score)

    # ─── Stats ──────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        """Synchronous stats for logging."""
        return {
            "initialized": self._initialized,
            "total_incidents": self._total_incidents,
            "active_incidents": len(self._active_incidents),
            "total_diagnoses": self._total_diagnoses,
            "total_repairs_attempted": self._total_repairs_attempted,
            "total_repairs_succeeded": self._total_repairs_succeeded,
            "healing_mode": (
                self._governor.healing_mode.value if self._governor else "unknown"
            ),
            "drive_state": self._drive_state.as_dict(),
        }

    # ─── Telemetry Helper ───────────────────────────────────────────

    def _emit_metric(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Emit a metric if the collector is available."""
        if self._metrics is not None:
            from contextlib import suppress

            with suppress(Exception):  # Telemetry failures must never affect immune function
                asyncio.create_task(
                    self._metrics.record(
                        system="thymos",
                        metric=name,
                        value=value,
                        labels=tags,
                    )
                )

    async def _emit_evolutionary_observable(
        self,
        observable_type: str,
        value: float,
        is_novel: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit an evolutionary observable event via Synapse bus."""
        bus = (
            self._synapse._event_bus if self._synapse is not None else None
        )
        if bus is None:
            return
        try:
            from primitives.evolutionary import EvolutionaryObservable
            from primitives.common import SystemID
            from systems.synapse.types import SynapseEvent, SynapseEventType

            obs = EvolutionaryObservable(
                source_system=SystemID.THYMOS,
                instance_id="",
                observable_type=observable_type,
                value=value,
                is_novel=is_novel,
                metadata=metadata or {},
            )
            event = SynapseEvent(
                event_type=SynapseEventType.EVOLUTIONARY_OBSERVABLE,
                source_system="thymos",
                data=obs.model_dump(mode="json"),
            )
            await bus.emit(event)
        except Exception:
            pass  # Evolutionary telemetry must never block immune function

    @staticmethod
    def _build_incident_alignment(
        care: float = 0.0,
        coherence: float = 0.0,
        growth: float = 0.0,
        honesty: float = 0.0,
    ) -> Any:
        """Build a DriveAlignmentVector from incident-derived scores. Lazy import to avoid circular deps."""
        try:
            from primitives.common import DriveAlignmentVector
            return DriveAlignmentVector(care=care, coherence=coherence, growth=growth, honesty=honesty)
        except Exception:
            return None

    async def _emit_re_training_example(
        self,
        *,
        category: str,
        instruction: str,
        input_context: str,
        output: str,
        outcome_quality: float = 0.5,
        reasoning_trace: str = "",
        alternatives_considered: list[str] | None = None,
        latency_ms: int = 0,
        episode_id: str = "",
        constitutional_alignment: Any = None,
    ) -> None:
        """Fire-and-forget RE training example onto Synapse bus."""
        bus = self._synapse._event_bus if self._synapse is not None else None
        if bus is None:
            return
        try:
            from decimal import Decimal
            from primitives.common import DriveAlignmentVector as _DAV

            example = RETrainingExample(
                source_system=SystemID.THYMOS,
                category=category,
                instruction=instruction,
                input_context=input_context,
                output=output,
                outcome_quality=max(0.0, min(1.0, outcome_quality)),
                reasoning_trace=reasoning_trace,
                alternatives_considered=alternatives_considered or [],
                latency_ms=latency_ms,
                episode_id=episode_id,
                constitutional_alignment=constitutional_alignment or _DAV(),
            )
            await bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="thymos",
                data=example.model_dump(mode="json"),
            ))
        except Exception:
            pass  # Never block the immune pipeline

    async def _emit_immune_advisory(self, incident: Incident, antibody: Any) -> None:
        """Emit IMMUNE_PATTERN_ADVISORY so Simula can avoid known-bad patterns (Loop 4)."""
        bus = (
            self._synapse._event_bus if self._synapse is not None else None
        )
        if bus is None:
            return

        # Extract affected files from incident context
        affected_files: list[str] = []
        ctx = incident.context
        if isinstance(ctx, dict):
            affected_files = ctx.get("affected_files", [])
        elif hasattr(ctx, "file_path"):
            affected_files = [ctx.file_path]

        confidence = getattr(antibody, "effectiveness", 1.0)

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await bus.emit(SynapseEvent(
                event_type=SynapseEventType.IMMUNE_PATTERN_ADVISORY,
                source_system="thymos",
                data={
                    "pattern_fingerprint": incident.fingerprint,
                    "description": incident.error_message[:200] if incident.error_message else "",
                    "affected_files": affected_files,
                    "incident_class": incident.incident_class.value if hasattr(incident.incident_class, "value") else str(incident.incident_class),
                    "severity": incident.severity.value if hasattr(incident.severity, "value") else str(incident.severity),
                    "confidence": confidence,
                },
            ))
            self._logger.info(
                "immune_advisory_emitted",
                fingerprint=incident.fingerprint[:16],
                confidence=round(confidence, 3),
            )
        except Exception:
            pass  # Advisory telemetry must never block immune function
