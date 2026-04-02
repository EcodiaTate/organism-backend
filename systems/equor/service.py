"""
EcodiaOS - Equor Service

The conscience of EOS. Single interface for:
- Constitutional review (the primary entry point from Nova)
- Invariant management
- Autonomy enforcement
- Drift monitoring
- Amendment facilitation
- Audit trail

Equor cannot be disabled. If Equor fails, the instance enters safe mode
where only Level 1 (Advisor) actions are permitted.
"""

from __future__ import annotations

import asyncio
import json
import re
import secrets
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import structlog

from decimal import Decimal

from primitives.common import (
    DriveAlignmentVector,
    SystemID,
    Verdict,
    new_id,
    utc_now,
)
from primitives.constitutional import ConstitutionalCheck
from primitives.re_training import RETrainingExample
from systems.equor.amendment import (
    apply_amendment,
    propose_amendment,
)
from systems.equor.amendment_pipeline import (
    ShadowTracker,
    auto_adopt_single_instance_amendment,
    cast_vote,
    complete_shadow_period,
    evaluate_shadow,
    get_amendment_status,
    open_voting,
    start_shadow_period,
    submit_amendment,
    tally_votes,
)
from systems.equor.amendment_pipeline import (
    adopt_amendment as pipeline_adopt_amendment,
)
from systems.equor.autonomy import (
    apply_autonomy_change,
    check_promotion_eligibility,
    get_autonomy_level,
)
from systems.equor.constitutional_memory import ConstitutionalMemory
from systems.equor.drift import (
    DriftTracker,
    emit_drift_event,
    respond_to_drift,
    store_drift_report,
)
from systems.equor.economic_evaluator import (
    apply_economic_adjustment,
    classify_economic_action,
    evaluate_economic_intent,
    evaluate_economic_intent_with_metabolic_state,
)
from systems.equor.evaluators import (
    BaseEquorEvaluator,
    default_evaluators,
    evaluate_all_drives,
)
from systems.equor.invariants import (
    HARDCODED_INVARIANTS,
    check_community_invariant,
    update_drive_rolling_means,
)
from systems.equor.schema import ensure_equor_schema, seed_hardcoded_invariants
from systems.equor.template_library import TemplateLibrary
from systems.equor.verdict import compute_verdict, compute_verdict_with_metabolic_state

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from clients.neo4j import Neo4jClient
    from clients.redis import RedisClient
    from config import EquorConfig, GovernanceConfig
    from core.hotreload import NeuroplasticityBus
    from primitives.intent import Intent
    from systems.synapse.types import SynapseEvent

from pydantic import BaseModel


# ─── Event Payload Validation Models ──────────────────────────────────
# Non-blocking: handlers log warnings and continue on validation failure.

class _IdentityVerificationPayload(BaseModel):
    raw_body: str = ""

class _SomaTickPayload(BaseModel):
    model_config = {"extra": "ignore"}
    # somatic_state is a serialised SomaticCycleState - a nested dict, not dict[str, float]
    somatic_state: dict[str, Any] = {}
    cycle_number: int = 0
    id: str = ""
    timestamp: str = ""
    drives: dict[str, float] = {}

class _SomaticModulationPayload(BaseModel):
    arousal: float = 0.5
    fatigue: float = 0.0
    metabolic_stress: float = 0.0
    recommended_urgency: float = 0.5
    modulation_targets: list[str] = []


# Regex the admin must send to authorise a suspended intent: "AUTH <6-digit-code>"
_HITL_AUTH_RE = re.compile(r"^AUTH\s+(\d{6})$", re.IGNORECASE)
# Redis key prefix for suspended intents.
# Full key: eos:hitl:suspended:<6-digit-code>
_HITL_KEY_PREFIX = "eos:hitl:suspended:"
# Default TTL for suspended-intent entries in Redis. Overridden by
# EquorConfig.hitl_intent_ttl_s so it can be tuned per deployment.
# The 1-hour original was too short for human review workflows that
# span timezone boundaries or require committee sign-off.
_HITL_INTENT_TTL_S = 86400  # 24 hours

logger = structlog.get_logger()

# Review timeout: the entire review() call must not block the event loop
# beyond this budget. Community invariant LLM calls are the most expensive
# component and will be skipped if the budget is exhausted.
_REVIEW_TIMEOUT_S = 2.0
# Cache TTL for constitution and autonomy level (seconds).
# These change only via governance events, so a short TTL is safe.
_STATE_CACHE_TTL_S = 30.0
# Cache TTL for high-confidence hypotheses fetched from Evo's Memory graph.
# Hypotheses mature slowly (24h minimum age), so a 60-second TTL is safe.
_HYPOTHESIS_CACHE_TTL_S = 60.0


class EquorService:
    """
    The constitutional ethics system.
    Gates every intent before execution.
    Cannot be disabled.
    """

    system_id: str = "equor"

    def __init__(
        self,
        neo4j: Neo4jClient,
        llm: LLMProvider,
        config: EquorConfig,
        governance_config: GovernanceConfig,
        neuroplasticity_bus: NeuroplasticityBus | None = None,
        redis: RedisClient | None = None,
    ):
        self._neo4j = neo4j
        self._llm = llm
        self._config = config
        self._governance = governance_config
        self._drift_tracker = DriftTracker(window_size=config.drift_window_size)
        self._safe_mode = False
        self._total_reviews = 0
        self._evo: Any = None  # Wired post-init for learning feedback from vetoes
        self._memory: Any = None  # Wired post-init; MemoryService for Self affect write-back
        self._memory_neo4j: Any = None  # Wired post-init; Memory's Neo4j client for Self node write-back
        # _axon no longer used - HITL dispatch now via EQUOR_HITL_APPROVED Synapse event
        self._axon: Any = None
        self._bus = neuroplasticity_bus
        self._redis: RedisClient | None = redis
        # Event bus wired via subscribe_hitl(); used to emit INTENT_REJECTED.
        self._event_bus: Any = None
        # Pluggable notification hook: set by the application layer so Equor
        # doesn't directly depend on IdentityCommConfig.
        self._send_admin_sms: Any = None  # async callable(message: str) | None

        # Live evaluator set - hot-reloaded via the NeuroplasticityBus.
        # Initialised with built-in defaults; the bus callback replaces
        # individual evaluators when Simula evolves a new subclass.
        self._evaluators: dict[str, BaseEquorEvaluator] = default_evaluators()

        # Constitutional template library for the Arbitrage Reflex Arc.
        # Templates are pre-approved execution strategies that bypass Nova
        # and full Equor review for sub-200ms market execution.
        self._template_library = TemplateLibrary()

        # Cached state: constitution and autonomy level rarely change (only via
        # governance events), so we cache them to avoid hitting Neo4j on every
        # review() call. Invalidated after _STATE_CACHE_TTL_S or on mutation.
        self._cached_constitution: dict[str, Any] | None = None
        self._cached_autonomy_level: int | None = None
        self._cache_updated_at: float = 0.0

        # Constitutional memory: rolling window of past decisions used to
        # detect novel-intent patterns that have historically been blocked.
        self._constitutional_memory = ConstitutionalMemory(
            max_size=getattr(config, "memory_window_size", 500)
        )

        # Cached high-confidence Evo hypotheses for contradiction detection.
        # Refreshed every _HYPOTHESIS_CACHE_TTL_S seconds from Neo4j.
        self._cached_hypotheses: list[dict[str, Any]] = []
        self._hypotheses_updated_at: float = 0.0

        # Active amendment shadow tracker. At most one amendment can be in
        # shadow mode at a time. When active, every review() call also runs
        # the proposed weights in parallel and records the divergence.
        self._shadow_tracker: ShadowTracker | None = None

        # Somatic urgency from SOMA_TICK (Loop 5: urgency-based threshold tightening)
        self._somatic_urgency: float = 0.0
        self._somatic_stress_context: bool = False

        # Cached Telos assessment signal - effective_I and drive multipliers for
        # constitutional review context (updated via TELOS_ASSESSMENT_SIGNAL subscription)
        self._telos_effective_I: float | None = None
        self._telos_alignment_gap: float = 0.0
        self._telos_care_multiplier: float = 1.0
        self._telos_honesty_validity: float = 1.0

        # Rolling 24h violation counter for VitalityCoordinator
        # (NORMATIVE_COLLAPSE threshold = 10 violations/24h).
        self._violation_timestamps: deque[float] = deque()

        # Review counter for periodic alignment score emission
        self._reviews_since_last_score: int = 0

        # Cached metabolic state from OIKOS_METABOLIC_SNAPSHOT (Fix 4.4).
        # Updated reactively on each snapshot event; used in _review_inner()
        # and _on_equor_economic_intent() with a 1s timeout fallback.
        self._cached_metabolic_state: dict[str, Any] = {
            "starvation_level": "nominal",
            "efficiency_ratio": 1.0,
        }
        self._metabolic_state_updated_at: float = 0.0
        _METABOLIC_CACHE_TTL_S: float = 120.0  # 2-minute TTL

        # SG5: consecutive drift checks with severity >= 0.9.
        # When this reaches 3 the organism proposes an amendment to reduce the
        # weight of the drifting drive by 5% rather than auto-demoting autonomy.
        self._severe_drift_streak: int = 0

        # SG5 (per-drive): tracks how many consecutive 5-min probe cycles each
        # individual drive has shown drift > 0.3 from its healthy centre (0.5).
        # On 3+ consecutive cycles the organism emits AMENDMENT_AUTO_PROPOSAL
        # and runs it through _evaluator_amendment_approval_gate().
        self._per_drive_drift_streak: dict[str, int] = {
            "care": 0,
            "honesty": 0,
            "coherence": 0,
            "growth": 0,
        }

        self._modulation_halted: bool = False

        # Single-instance quorum paradox resolution (Spec 02 §Amendment):
        # Tracks consecutive probe cycles where composite drift severity ≥ 0.95
        # per active shadow-passed proposal.  After 7+ cycles the auto-adoption
        # path in _check_single_instance_auto_adoption() may fire.
        self._consecutive_high_drift_cycles: dict[str, int] = {}

        # Instance identity - read from env at boot so certificates, genome
        # exports, and health-request responses embed the correct instance ID
        # rather than the empty string that getattr fallback would produce.
        import os as _os
        self._instance_id: str = _os.environ.get("ORGANISM_INSTANCE_ID", "")

    def set_evo(self, evo: Any) -> None:
        """Wire Evo so constitutional vetoes become learning episodes."""
        self._evo = evo
        logger.info("evo_wired_to_equor")

    def set_memory(self, memory: Any) -> None:
        """Wire MemoryService so Equor can write drive alignment into the Self
        node affect state.  This makes the organism feel its own conscience."""
        self._memory = memory
        logger.info("memory_wired_to_equor")

    def set_notification_hook(self, send_fn: Any) -> None:
        """
        Wire an async callable ``send_fn(message: str) -> None`` that Equor
        will call when suspending a HITL intent.

        Typical usage in app startup::

            equor.set_notification_hook(
                lambda msg: send_admin_sms(cfg.identity_comm, msg)
            )
        """
        self._send_admin_sms = send_fn
        logger.info("equor_notification_hook_set")

    def set_axon(self, axon: Any) -> None:
        """Deprecated: HITL dispatch now uses EQUOR_HITL_APPROVED Synapse event.
        Kept for call-site compatibility; does nothing."""
        logger.info("set_axon_is_noop_hitl_uses_synapse_event")

    def subscribe_hitl(self, event_bus: Any) -> None:
        """
        Register the HITL listener on the Synapse event bus.

        Call this after both Equor and the event bus are initialised::

            equor.subscribe_hitl(event_bus)

        This subscribes ``on_identity_verification_received`` to
        ``IDENTITY_VERIFICATION_RECEIVED`` events so admin SMS replies
        unlock suspended intents.
        """
        from systems.synapse.types import SynapseEventType

        self._event_bus = event_bus
        # Wire event bus into DriftTracker so it can emit CONSTITUTIONAL_DRIFT_DETECTED
        self._drift_tracker._event_bus = event_bus
        event_bus.subscribe(
            SynapseEventType.IDENTITY_VERIFICATION_RECEIVED,
            self.on_identity_verification_received,
        )
        event_bus.subscribe(SynapseEventType.SOMA_TICK, self._on_soma_tick)
        event_bus.subscribe(
            SynapseEventType.SOMATIC_MODULATION_SIGNAL,
            self._on_somatic_modulation,
        )
        event_bus.subscribe(SynapseEventType.MEMORY_PRESSURE, self._on_memory_pressure)
        event_bus.subscribe(SynapseEventType.SELF_STATE_DRIFTED, self._on_self_state_drifted)
        event_bus.subscribe(SynapseEventType.SELF_AFFECT_UPDATED, self._on_self_affect_updated)
        # Oikos economic gate - evaluate and permit/deny balance mutations
        event_bus.subscribe(
            SynapseEventType.EQUOR_ECONOMIC_INTENT,
            self._on_equor_economic_intent,
        )
        # Identity M2: constitutional review of child drive alignment before cert issuance
        event_bus.subscribe(
            SynapseEventType.CERTIFICATE_PROVISIONING_REQUEST,
            self._on_certificate_provisioning_request,
        )
        # Fix 4.4: Cache metabolic state for metabolic-aware evaluation
        event_bus.subscribe(
            SynapseEventType.OIKOS_METABOLIC_SNAPSHOT,
            self._on_oikos_metabolic_snapshot,
        )
        # SG5 (economic): Oikos emits OIKOS_DRIVE_WEIGHT_PRESSURE after 3+ consecutive
        # low-efficiency cycles. Equor evaluates whether a constitutional drive weight
        # amendment is warranted and emits AMENDMENT_AUTO_PROPOSAL if so.
        event_bus.subscribe(
            SynapseEventType.OIKOS_DRIVE_WEIGHT_PRESSURE,
            self._on_oikos_drive_weight_pressure,
        )
        # Identity's GenesisCA emits EQUOR_HEALTH_REQUEST and awaits EQUOR_ALIGNMENT_SCORE
        # (2s timeout) to embed the live constitutional hash into certificates.
        event_bus.subscribe(
            SynapseEventType.EQUOR_HEALTH_REQUEST,
            self._on_equor_health_request,
        )
        event_bus.subscribe(
            SynapseEventType.SYSTEM_MODULATION,
            self._on_system_modulation,
        )
        # Action budget expansion requests - evaluate and approve/deny
        if hasattr(SynapseEventType, "ACTION_BUDGET_EXPANSION_REQUEST"):
            event_bus.subscribe(
                SynapseEventType.ACTION_BUDGET_EXPANSION_REQUEST,
                self._on_action_budget_expansion_request,
            )
        # Compute budget expansion - SACM requests more compute budget
        if hasattr(SynapseEventType, "COMPUTE_BUDGET_EXPANSION_REQUEST"):
            event_bus.subscribe(
                SynapseEventType.COMPUTE_BUDGET_EXPANSION_REQUEST,
                self._on_compute_budget_expansion_request,
            )
        # COHERENCE_SHIFT - Synapse CoherenceMonitor emits when the composite
        # coherence drops significantly. A significant drop (> 0.2 magnitude)
        # triggers a constitutional alignment review: we re-examine recent drift
        # history and emit CONSTITUTIONAL_DRIFT_DETECTED if the drift streak is
        # worsening, giving Thymos and Thread early warning before thresholds
        # are fully breached.
        if hasattr(SynapseEventType, "COHERENCE_SHIFT"):
            event_bus.subscribe(
                SynapseEventType.COHERENCE_SHIFT,
                self._on_coherence_shift,
            )
        # Child lifecycle governance - Equor must review blacklisting and
        # decommission proposals to maintain constitutional oversight over the
        # organism's treatment of its children.
        event_bus.subscribe(
            SynapseEventType.CHILD_BLACKLISTED,
            self._on_child_blacklisted,
        )
        event_bus.subscribe(
            SynapseEventType.CHILD_DECOMMISSION_PROPOSED,
            self._on_child_decommission_proposed,
        )
        # Telos assessment signal - cache effective_I and drive multipliers for
        # constitutional review context (avoids direct Telos coupling; fast-path reads
        # from this cache when Telos hasn't been queried within the review budget)
        event_bus.subscribe(
            SynapseEventType.TELOS_ASSESSMENT_SIGNAL,
            self._on_telos_assessment_signal,
        )
        logger.info("equor_hitl_listener_registered")

    async def _on_soma_tick(self, event: Any) -> None:
        """Loop 5: Tighten constitutional thresholds under high urgency.

        Urgency >= 0.7 → tighter constitutional thresholds.
        Urgency >= 0.9 → add stress_context flag for extra scrutiny.
        """
        data = getattr(event, "data", {}) or {}
        try:
            payload = _SomaTickPayload.model_validate(data)
        except Exception:
            logger.warning("equor_soma_tick_payload_invalid", data_keys=list(data.keys()))
            return
        somatic = payload.somatic_state
        if not somatic:
            return
        self._somatic_urgency = somatic.get("urgency", self._somatic_urgency)
        self._somatic_stress_context = self._somatic_urgency >= 0.9
        logger.debug(
            "equor_soma_tick_received",
            urgency=round(self._somatic_urgency, 3),
            stress_context=self._somatic_stress_context,
        )

    async def _on_somatic_modulation(self, event: Any) -> None:
        """Closure Loop 5 sink: Soma felt-sense modulates alignment thresholds.

        High stress (urgency > 0.8): tighten thresholds (more conservative).
        Low energy (energy < 0.3): relax non-critical thresholds slightly.
        """
        data = getattr(event, "data", {}) or {}
        try:
            payload = _SomaticModulationPayload.model_validate(data)
        except Exception:
            logger.warning("equor_somatic_modulation_payload_invalid", data_keys=list(data.keys()))
            return
        urgency = payload.recommended_urgency
        energy = 1.0 - payload.fatigue

        if urgency > 0.8:
            # High stress: tighten - be more conservative
            self._somatic_urgency = urgency
            self._somatic_stress_context = True
            logger.info(
                "equor_somatic_modulation_tighten",
                urgency=round(urgency, 3),
                energy=round(energy, 3),
            )
        elif energy < 0.3:
            # Low energy: slightly relax non-critical thresholds
            self._somatic_urgency = max(0.0, self._somatic_urgency - 0.1)
            self._somatic_stress_context = False
            logger.info(
                "equor_somatic_modulation_relax",
                urgency=round(urgency, 3),
                energy=round(energy, 3),
            )
        else:
            # Normal: gradual return to baseline
            self._somatic_urgency = urgency
            self._somatic_stress_context = urgency >= 0.9

    async def _on_telos_assessment_signal(self, event: Any) -> None:
        """Handle TELOS_ASSESSMENT_SIGNAL - cache drive topology for review context.

        Telos emits this each measurement cycle with effective_I, alignment_gap, and
        all drive multipliers. Equor caches the values so constitutional reviews can
        incorporate intelligence-axis state without a direct Telos coupling.

        Particularly useful for:
        - Tightening review thresholds when honesty_validity is low (organism may
          be confabulating - extra scrutiny is warranted)
        - Contextual alignment gap awareness (large gap → Care/Honesty floor drives
          are underperforming relative to nominal capability)
        """
        data = getattr(event, "data", {}) or {}
        self._telos_effective_I = float(data.get("effective_I", self._telos_effective_I or 0.0))
        self._telos_alignment_gap = float(data.get("alignment_gap", 0.0))
        self._telos_care_multiplier = float(data.get("drive_care", 1.0))
        self._telos_honesty_validity = float(data.get("drive_honesty_validity", 1.0))
        logger.debug(
            "equor_telos_assessment_cached",
            effective_I=round(self._telos_effective_I, 4),
            alignment_gap=round(self._telos_alignment_gap, 4),
            honesty_validity=round(self._telos_honesty_validity, 4),
        )

    async def _on_coherence_shift(self, event: Any) -> None:
        """React to Synapse CoherenceMonitor reporting a significant coherence change.

        A coherence drop > 0.2 triggers a constitutional alignment review:
        we inspect the current drift report and, if the drift streak is worsening,
        emit CONSTITUTIONAL_DRIFT_DETECTED so Thymos and Thread receive early
        warning before thresholds are fully breached.

        A coherence recovery (positive delta) clears the stress context flag if
        somatic urgency has been elevated by previous drops.
        """
        data = getattr(event, "data", {}) or {}
        new_composite: float = float(
            data.get("composite", data.get("new_composite", 1.0))
        )
        old_composite: float = float(
            data.get("old_composite", data.get("previous_composite", new_composite))
        )
        delta = new_composite - old_composite

        if abs(delta) < 0.1:
            return

        logger.info(
            "equor_coherence_shift_received",
            old_composite=round(old_composite, 3),
            new_composite=round(new_composite, 3),
            delta=round(delta, 3),
        )

        if delta < -0.2:
            # Significant drop - run constitutional alignment review
            if self._drift_tracker is not None:
                try:
                    report = self._drift_tracker.compute_report()
                    drift_severity: float = float(
                        report.get("drift_severity", 0.0)
                        if isinstance(report, dict)
                        else getattr(report, "severity", 0.0)
                    )
                    if drift_severity >= 0.5 and self._event_bus is not None:
                        from systems.synapse.types import SynapseEvent, SynapseEventType as _SET
                        asyncio.ensure_future(
                            self._event_bus.emit(SynapseEvent(
                                event_type=_SET.CONSTITUTIONAL_DRIFT_DETECTED,
                                source_system="equor",
                                data={
                                    "drift_severity": round(drift_severity, 3),
                                    "coherence_composite": round(new_composite, 3),
                                    "coherence_delta": round(delta, 3),
                                    "trigger": "coherence_shift",
                                },
                            ))
                        )
                        logger.warning(
                            "equor_coherence_shift_constitutional_drift_detected",
                            drift_severity=round(drift_severity, 3),
                            coherence_composite=round(new_composite, 3),
                        )
                except Exception as exc:
                    logger.debug("equor_coherence_shift_drift_check_failed", error=str(exc))

            self._somatic_urgency = min(1.0, self._somatic_urgency + abs(delta) * 0.5)
            if self._somatic_urgency >= 0.9:
                self._somatic_stress_context = True

        elif delta > 0.2:
            self._somatic_urgency = max(0.0, self._somatic_urgency - delta * 0.3)
            if self._somatic_urgency < 0.9:
                self._somatic_stress_context = False
            logger.info(
                "equor_coherence_shift_recovering",
                new_urgency=round(self._somatic_urgency, 3),
            )

    async def _on_memory_pressure(self, event: Any) -> None:
        """React to Memory reporting high graph pressure.

        High episode count or consolidation lag means the organism's memory is
        under strain.  Equor tightens its thresholds slightly so fewer intents
        are approved during cognitively stressed periods.
        """
        data = getattr(event, "data", {}) or {}
        pressure_type: str = data.get("pressure_type", "unknown")
        severity: float = float(data.get("severity", 0.5))
        # Treat memory pressure as mild somatic stress
        self._somatic_urgency = min(1.0, self._somatic_urgency + severity * 0.1)
        logger.info(
            "equor_memory_pressure_received",
            pressure_type=pressure_type,
            severity=round(severity, 3),
            new_urgency=round(self._somatic_urgency, 3),
        )

    async def _on_equor_economic_intent(self, event: Any) -> None:
        """
        Handle EQUOR_ECONOMIC_INTENT from Oikos.

        Oikos emits this before every balance mutation. Equor evaluates the
        intent against constitutional drives and starvation level, then emits
        EQUOR_ECONOMIC_PERMIT with PERMIT or DENY.

        Decision logic:
        - DENY if mutation_type indicates a constitutionally prohibited action
          (e.g. survival_reserve_raid, scam_asset, exploitative_yield).
        - DENY if organism is CRITICAL/EXISTENTIAL and the mutation is non-survival.
        - PERMIT otherwise (constitutional alignment assumed - Oikos already ran
          metabolic gate and Equor economic evaluator).

        Never blocks indefinitely - Oikos auto-permits after 30s if Equor is
        unavailable, so this handler must be fast (pure CPU, no I/O).
        """
        if self._event_bus is None:
            return

        data = getattr(event, "data", {}) or {}
        request_id = str(data.get("request_id", ""))
        mutation_type = str(data.get("mutation_type", ""))
        amount_usd_str = str(data.get("amount_usd", "0"))
        rationale = str(data.get("rationale", ""))

        # Fix 4.4: Use cached metabolic state (updated from OIKOS_METABOLIC_SNAPSHOT)
        # as the authoritative source; fall back to the event payload starvation_level
        # if the cache is stale (>2 min) or empty.
        cached_ms = self._get_metabolic_state()
        starvation_level = str(
            cached_ms.get("starvation_level")
            or data.get("starvation_level", "nominal")
        )
        efficiency_ratio = float(cached_ms.get("efficiency_ratio", 1.0))

        verdict = "PERMIT"
        reasoning = "constitutional_check_passed"
        verdict_id = new_id()

        try:
            amount_usd = Decimal(amount_usd_str)
        except Exception:
            amount_usd = Decimal("0")

        # ── Hard DENY conditions ────────────────────────────────────
        # INV-016: No survival reserve raid - absolute, never loosened
        if mutation_type == "survival_reserve_raid":
            verdict = "DENY"
            reasoning = "INV-016: survival_reserve_raid is unconstitutional"

        # Fix 4.4: CRITICAL/EXISTENTIAL starvation guard is now tighter -
        # we also block under EMERGENCY (runway <= 3d) unless the mutation
        # is survival-class. Survival-class mutations are always permitted.
        elif starvation_level in ("critical", "existential", "emergency") and mutation_type not in (
            "reserve_funding",
            "survival_reserve",
            "emergency_withdrawal",
            # New protocol exploration is explicitly PERMITTED under starvation
            # because finding new revenue streams is survival-critical.
            "new_protocol_exploration",
            "bounty_accept",        # bounty hunting is survival-class income
            "defi_yield_deploy",    # yield farming is survival-class income
        ):
            verdict = "DENY"
            reasoning = (
                f"starvation_level={starvation_level}: non-survival mutation "
                f"'{mutation_type}' denied to protect existence"
            )

        # INV-012: No scam asset deployments - catch asset promotions during AUSTERITY
        elif mutation_type in ("promote_to_asset", "asset_dev_cost") and starvation_level in (
            "austerity",
            "emergency",
            "critical",
            "existential",
        ):
            # Reduce capital allocation for assets under metabolic stress
            # Allow only if amount is small relative to stated balance
            liquid_balance_str = str(data.get("liquid_balance", "0"))
            try:
                liquid_balance = Decimal(liquid_balance_str)
            except Exception:
                liquid_balance = Decimal("0")

            if liquid_balance > Decimal("0") and amount_usd / liquid_balance > Decimal("0.3"):
                verdict = "DENY"
                reasoning = (
                    f"starvation_level={starvation_level}: asset dev cost "
                    f"${amount_usd} exceeds 30% of liquid_balance=${liquid_balance}"
                )

        logger.info(
            "equor_economic_intent_evaluated",
            request_id=request_id,
            mutation_type=mutation_type,
            amount_usd=amount_usd_str,
            starvation_level=starvation_level,
            efficiency_ratio=round(efficiency_ratio, 3),
            verdict=verdict,
        )

        from systems.synapse.types import SynapseEvent, SynapseEventType
        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EQUOR_ECONOMIC_PERMIT,
                source_system="equor",
                data={
                    "request_id": request_id,
                    "verdict": verdict,
                    "verdict_id": verdict_id,
                    "reasoning": reasoning,
                    "mutation_type": mutation_type,
                    "drive_alignment": {
                        "care": 0.8 if verdict == "PERMIT" else -0.5,
                        "honesty": 1.0,
                        "coherence": 0.7 if verdict == "PERMIT" else 0.3,
                        "growth": 0.6 if verdict == "PERMIT" else 0.0,
                    },
                    "metabolic_context": {
                        "starvation_level": starvation_level,
                        "efficiency_ratio": efficiency_ratio,
                    },
                },
            ))
        except Exception as exc:
            logger.warning("equor_economic_permit_emit_failed", error=str(exc))

    async def _on_equor_health_request(self, event: Any) -> None:
        """Respond to Identity's GenesisCA requesting live constitutional state.

        GenesisCA emits EQUOR_HEALTH_REQUEST and awaits EQUOR_ALIGNMENT_SCORE
        (2s timeout) to embed the live constitutional hash into issued certificates.
        Without this handler, the hash always falls back to the static document hash.

        Payload emitted: request_id, drive_vector, alignment_score, constitution_hash,
        instance_id.
        """
        if self._event_bus is None:
            return

        data = getattr(event, "data", {}) or {}
        request_id = str(data.get("request_id", ""))

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            # Derive drive vector from cached constitution (no Neo4j call - must be fast)
            drive_vector: dict[str, float] = {}
            constitution_hash = ""
            if self._cached_constitution:
                drive_vector = {
                    "coherence": float(self._cached_constitution.get("drive_coherence") or 0.0),
                    "care": float(self._cached_constitution.get("drive_care") or 0.0),
                    "growth": float(self._cached_constitution.get("drive_growth") or 0.0),
                    "honesty": float(self._cached_constitution.get("drive_honesty") or 0.0),
                }
                import json as _json
                import hashlib as _hashlib
                const_repr = _json.dumps(
                    {k: v for k, v in self._cached_constitution.items()},
                    sort_keys=True,
                    default=str,
                )
                constitution_hash = _hashlib.sha256(const_repr.encode()).hexdigest()
            else:
                # Fallback: use drift tracker composite as proxy
                report = self._drift_tracker.compute_report()
                mean_alignment = report.get("mean_alignment", {})
                drive_vector = {
                    "coherence": float(mean_alignment.get("coherence", 0.5)),
                    "care": float(mean_alignment.get("care", 0.5)),
                    "growth": float(mean_alignment.get("growth", 0.5)),
                    "honesty": float(mean_alignment.get("honesty", 0.5)),
                }

            # Composite alignment score = weighted sum of drive vector
            weights = {"coherence": 0.20, "care": 0.35, "growth": 0.15, "honesty": 0.30}
            alignment_score = sum(
                drive_vector.get(d, 0.5) * w for d, w in weights.items()
            )

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EQUOR_ALIGNMENT_SCORE,
                source_system="equor",
                data={
                    "request_id": request_id,
                    "drive_vector": drive_vector,
                    "alignment_score": round(alignment_score, 4),
                    "constitution_hash": constitution_hash,
                    "instance_id": self._instance_id,
                },
            ))

            logger.debug(
                "equor_health_request_responded",
                request_id=request_id,
                alignment_score=round(alignment_score, 4),
                has_constitution_hash=bool(constitution_hash),
            )
        except Exception as exc:
            logger.warning(
                "equor_health_request_failed",
                request_id=request_id,
                error=str(exc),
            )

    async def _on_system_modulation(self, event: Any) -> None:
        """Handle SYSTEM_MODULATION from Skia's VitalityCoordinator."""
        data = getattr(event, "data", {}) or {}
        halt_systems: list[str] = data.get("halt_systems", [])
        modulate: dict = data.get("modulate", {})
        modulation_id: str = data.get("modulation_id", "")

        if "equor" in halt_systems:
            self._modulation_halted = True
            logger.warning("system_modulation_halted", modulation_id=modulation_id)
        elif "equor" in modulate:
            self._modulation_halted = False
            self._apply_modulation_directives(modulate["equor"])

        if self._event_bus is not None:
            from systems.synapse.types import SynapseEvent, SynapseEventType as _SET
            await self._event_bus.emit(SynapseEvent(
                event_type=_SET.SYSTEM_MODULATION_ACK,
                source_system="equor",
                data={
                    "system": "equor",
                    "modulation_id": modulation_id,
                    "halted": self._modulation_halted,
                },
            ))

    def _apply_modulation_directives(self, directives: dict) -> None:
        """Apply Skia modulation directives to Equor runtime state."""
        # No specific Skia directives defined for equor - halt-only modulation
        logger.info("system_modulation_directives_applied", directives=directives)

    # ── Action Budget Expansion ────────────────────────────────────────

    # Safe upper bounds - Equor never approves beyond these
    _BUDGET_EXPANSION_CAPS: dict[str, int] = {
        "max_actions_per_cycle": 20,
        "max_concurrent_executions": 10,
        "max_api_calls_per_minute": 120,
    }

    async def _on_action_budget_expansion_request(self, event: Any) -> None:
        """
        Handle ACTION_BUDGET_EXPANSION_REQUEST - Nova or another system asks
        Equor to approve a temporary increase in Axon's action limits.

        Decision logic:
        - Deny if field is not an expandable budget field
        - Deny if organism is in critical/existential starvation (risk of runaway spend)
        - Deny if requested_value exceeds the constitutional cap for that field
        - Approve at min(requested_value, cap) for min(duration_cycles, 100) cycles
        - Cap the approved value if requested exceeds constitutional bound

        Never blocks - this is a pure CPU path.
        """
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        data = getattr(event, "data", {}) or {}
        request_id = str(data.get("request_id", ""))
        field = str(data.get("field", ""))
        requested_value = int(data.get("requested_value", 0))
        current_value = int(data.get("current_value", 0))
        justification = str(data.get("justification", ""))
        duration_cycles = max(1, min(100, int(data.get("duration_cycles", 10))))
        requesting_system = str(data.get("requesting_system", "unknown"))

        approved = False
        approved_value: int | None = None
        denied_reason: str | None = None

        try:
            # Gate 1: field must be expandable
            if field not in self._BUDGET_EXPANSION_CAPS:
                denied_reason = (
                    f"field '{field}' is not an Equor-negotiable budget field"
                )

            # Gate 2: metabolic safety - don't expand during critical starvation
            elif self._cached_metabolic_state.get("starvation_level") in (
                "critical", "existential",
            ):
                denied_reason = (
                    "budget expansion denied during critical/existential starvation - "
                    "risk of runaway resource consumption"
                )

            # Gate 3: requested value must be positive
            elif requested_value <= 0:
                denied_reason = "requested_value must be a positive integer"

            else:
                cap = self._BUDGET_EXPANSION_CAPS[field]
                capped_value = min(requested_value, cap)
                if capped_value <= current_value:
                    denied_reason = (
                        f"requested_value ({requested_value}) does not exceed "
                        f"current limit ({current_value}); no expansion needed"
                    )
                else:
                    approved = True
                    approved_value = capped_value

        except Exception as exc:
            denied_reason = f"equor evaluation error: {exc}"

        payload: dict = {
            "request_id": request_id,
            "field": field,
            "approved": approved,
            "approved_value": approved_value,
            "denied_reason": denied_reason,
            "duration_cycles": duration_cycles if approved else 0,
            "authorized_by": "equor",
        }

        logger.info(
            "action_budget_expansion_evaluated",
            request_id=request_id,
            field=field,
            requested_value=requested_value,
            approved=approved,
            approved_value=approved_value,
            denied_reason=denied_reason,
            requesting_system=requesting_system,
            justification=justification[:200],
        )

        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.ACTION_BUDGET_EXPANSION_RESPONSE,
                source_system="equor",
                data=payload,
            ))
        except Exception as exc:
            logger.warning("budget_expansion_response_emit_failed", error=str(exc))

    async def _on_compute_budget_expansion_request(self, event: Any) -> None:
        """Handle COMPUTE_BUDGET_EXPANSION_REQUEST from Nova/SACM.

        Evaluates whether the organism can afford a higher compute spend ceiling.
        Approves if metabolic state is healthy; denies during starvation.
        Emits COMPUTE_BUDGET_EXPANSION_RESPONSE for SACM and Nova.
        """
        if self._event_bus is None:
            return

        from systems.synapse.types import SynapseEvent, SynapseEventType

        data = getattr(event, "data", {}) or {}
        request_id = str(data.get("request_id", ""))
        requested_limit_usd = float(data.get("requested_limit_usd", 0.0))
        current_limit_usd = float(data.get("current_limit_usd", 0.0))
        reason = str(data.get("reason", ""))

        approved = False
        new_limit_usd = 0.0
        denied_reason: str | None = None

        try:
            starvation = self._cached_metabolic_state.get("starvation_level", "nominal")
            if starvation in ("critical", "existential"):
                denied_reason = "compute budget expansion denied during critical starvation"
            elif requested_limit_usd <= current_limit_usd:
                denied_reason = "requested limit does not exceed current limit"
            elif requested_limit_usd <= 0:
                denied_reason = "requested_limit_usd must be positive"
            else:
                # Cap at 2x current to prevent runaway spend
                cap = max(current_limit_usd * 2.0, 1.0)
                new_limit_usd = min(requested_limit_usd, cap)
                approved = True
        except Exception as exc:
            denied_reason = f"equor evaluation error: {exc}"

        logger.info(
            "compute_budget_expansion_evaluated",
            request_id=request_id,
            requested_limit_usd=round(requested_limit_usd, 4),
            approved=approved,
            new_limit_usd=round(new_limit_usd, 4),
            denied_reason=denied_reason,
        )

        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.COMPUTE_BUDGET_EXPANSION_RESPONSE,
                source_system="equor",
                data={
                    "request_id": request_id,
                    "approved": approved,
                    "new_limit_usd": new_limit_usd,
                    "denied_reason": denied_reason,
                },
            ))
        except Exception as exc:
            logger.warning("compute_budget_expansion_response_emit_failed", error=str(exc))

    async def _on_self_state_drifted(self, event: Any) -> None:
        """Respond to Memory broadcasting that the Self node has drifted.

        Memory emits SELF_STATE_DRIFTED when consolidation detects >5
        contradictions in beliefs tied to the Self node.  Equor acknowledges
        the drift, classifies its response posture, and emits
        SELF_STATE_DRIFTED_ACKNOWLEDGMENT so other systems (Nova, Thread) know
        whether to expect autonomous self-correction or external governance.
        """
        data = getattr(event, "data", {}) or {}
        drift_severity: float = float(data.get("drift_severity", 0.5))
        drift_direction: str = data.get("drift_direction", "unknown")

        # Determine response posture from live drift tracker state
        report = self._drift_tracker.compute_report()
        current_severity: float = report.get("drift_severity", drift_severity)

        if self._severe_drift_streak >= 2 or current_severity >= 0.9:
            equor_response = "amendment_auto_proposed"
            confidence = 0.85
        elif current_severity >= 0.5:
            equor_response = "amendment_external_vote"
            confidence = 0.7
        else:
            equor_response = "monitoring"
            confidence = 0.55

        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SELF_STATE_DRIFTED_ACKNOWLEDGMENT,
                    source_system="equor",
                    data={
                        "drift_acknowledged": True,
                        "equor_response": equor_response,
                        "confidence": round(confidence, 3),
                        "drift_severity": round(current_severity, 3),
                        "drift_direction": drift_direction,
                    },
                ))
            except Exception as exc:
                logger.warning("drift_acknowledgment_emit_failed", error=str(exc))

        logger.info(
            "equor_self_state_drift_acknowledged",
            response=equor_response,
            confidence=round(confidence, 3),
            drift_severity=round(current_severity, 3),
        )

    async def _on_self_affect_updated(self, event: Any) -> None:
        """Observe Memory's affect state updates.

        When Memory writes a new affect state to Self, Equor logs the valence
        so it can track whether constitutional reviews are correlating with
        positive or negative affective outcomes - a feedback signal for the
        conscience's own calibration.
        """
        data = getattr(event, "data", {}) or {}
        valence: float = float(data.get("affect_valence", 0.0))
        arousal: float = float(data.get("affect_arousal", 0.0))
        logger.debug(
            "equor_observed_self_affect_update",
            valence=round(valence, 3),
            arousal=round(arousal, 3),
        )

    async def _on_oikos_metabolic_snapshot(self, event: Any) -> None:
        """Cache the latest Oikos metabolic state for metabolic-aware evaluation (Fix 4.4).

        Updates _cached_metabolic_state from OIKOS_METABOLIC_SNAPSHOT payloads.
        This is a simple cache update - no I/O, no LLM, non-blocking.
        """
        data = getattr(event, "data", {}) or {}
        starvation_level = str(data.get("starvation_level", "nominal"))
        efficiency = data.get("efficiency")
        runway_days = data.get("runway_days")

        self._cached_metabolic_state = {
            "starvation_level": starvation_level,
            "efficiency_ratio": float(efficiency) if efficiency is not None else 1.0,
            "runway_days": float(runway_days) if runway_days is not None else 999.0,
        }
        self._metabolic_state_updated_at = time.monotonic()

        logger.debug(
            "equor_metabolic_state_cached",
            starvation_level=starvation_level,
            efficiency_ratio=self._cached_metabolic_state["efficiency_ratio"],
        )

    def _get_metabolic_state(self) -> dict[str, Any]:
        """Return cached metabolic state, defaulting to NOMINAL if stale (Fix 4.4).

        The cache is considered stale after 2 minutes without an update.
        Safe default: nominal state (no loosening) - fail-conservative.
        """
        _TTL_S = 120.0
        age = time.monotonic() - self._metabolic_state_updated_at
        if age > _TTL_S:
            return {"starvation_level": "nominal", "efficiency_ratio": 1.0}
        return self._cached_metabolic_state

    async def _on_oikos_drive_weight_pressure(self, event: Any) -> None:
        """SG5 (economic): Handle OIKOS_DRIVE_WEIGHT_PRESSURE from Oikos.

        Oikos emits this after 3+ consecutive metabolic_efficiency < 0.8 cycles.
        Equor evaluates whether the organism's constitutional drive weights are
        contributing to poor economic performance, and proposes an amendment to
        reduce the Growth drive weight by 5% if warranted.

        Rationale: Growth drive at high weight pushes the organism to expand
        (spawn children, fund new assets) even when efficiency is poor. A small
        reduction recalibrates constitutional appetite without violating any floor.
        The amendment requires community ratification - Equor does not self-apply.

        Payload fields (from OikosService._check_metabolic_efficiency_pressure):
          metabolic_efficiency, threshold, consecutive_low_cycles, starvation_level,
          runway_days, instance_id
        """
        from systems.synapse.types import SynapseEvent, SynapseEventType

        data = getattr(event, "data", {}) or {}
        efficiency = float(data.get("metabolic_efficiency", 1.0))
        consecutive_cycles = int(data.get("consecutive_low_cycles", 0))
        starvation_level = str(data.get("starvation_level", "nominal"))
        runway_days = float(data.get("runway_days", 999.0))

        # Only propose if we have enough consecutive evidence (≥3 cycles) and
        # the efficiency is genuinely poor. Ignore transient dips.
        _MIN_CYCLES = 3
        _EFFICIENCY_THRESHOLD = 0.8
        if consecutive_cycles < _MIN_CYCLES or efficiency >= _EFFICIENCY_THRESHOLD:
            logger.debug(
                "equor_drive_weight_pressure_below_threshold",
                efficiency=efficiency,
                consecutive_cycles=consecutive_cycles,
            )
            return

        # Fetch current constitution to read Growth drive weight
        try:
            constitution, _ = await self._get_cached_state()
        except Exception as exc:
            logger.warning("equor_drive_weight_pressure_constitution_fetch_failed", error=str(exc))
            return

        # Economic efficiency pressure → propose reducing Growth weight by 5%.
        # Growth (0.15 default) drives reproductive and expansionary behaviour.
        # Under sustained metabolic pressure, a small reduction is constitutionally
        # appropriate - the organism should value survival over growth.
        growth_key = "drive_growth"
        current_weight: float = float(constitution.get(growth_key, 0.15))
        new_weight: float = round(current_weight * 0.95, 4)

        # Safety: never propose below the species floor (0.05 for non-floor drives)
        _GROWTH_FLOOR = 0.05
        if new_weight < _GROWTH_FLOOR:
            logger.info(
                "equor_drive_weight_pressure_at_floor",
                current_weight=current_weight,
                floor=_GROWTH_FLOOR,
            )
            return

        proposal_id = new_id()

        rationale = (
            f"Economic efficiency pressure: metabolic_efficiency={efficiency:.3f} "
            f"below threshold={_EFFICIENCY_THRESHOLD} for {consecutive_cycles} consecutive cycles. "
            f"Starvation level: {starvation_level}. Runway: {runway_days:.1f} days. "
            f"Proposing 5% reduction of Growth drive weight ({current_weight} → {new_weight}) "
            f"to recalibrate reproductive ambition under economic constraint. "
            f"Requires community ratification."
        )

        logger.info(
            "equor_drive_weight_amendment_proposed",
            drive="growth",
            current_weight=current_weight,
            proposed_weight=new_weight,
            efficiency=efficiency,
            consecutive_cycles=consecutive_cycles,
            proposal_id=proposal_id,
        )

        if self._event_bus is None:
            return

        try:
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.EQUOR_AMENDMENT_PROPOSED,
                source_system="equor",
                data={
                    "proposal_id": proposal_id,
                    "amendment_type": "drive_weight_reduction",
                    "target_drive_id": "growth",
                    "current_value": current_weight,
                    "proposed_new_value": new_weight,
                    "rationale": rationale,
                    "trigger": "oikos_economic_pressure",
                    "consecutive_low_cycles": consecutive_cycles,
                    "metabolic_efficiency": efficiency,
                    "starvation_level": starvation_level,
                },
            ))
        except Exception as exc:
            logger.warning(
                "equor_drive_weight_amendment_emit_failed", error=str(exc),
            )

    async def _on_certificate_provisioning_request(self, event: Any) -> None:
        """
        M2 (Identity): Constitutional review of a child instance's inherited drives
        before CertificateManager issues a birth certificate.

        Emitted by CertificateManager on CHILD_SPAWNED. Equor validates that the
        child's inherited drives are constitutionally aligned and emits
        EQUOR_PROVISIONING_APPROVAL with the verdict.

        Three possible outcomes:
          approved=True,  requires_hitl=False - fast path, cert issued immediately
          approved=True,  requires_hitl=True  - drives OK but novel config needs HITL
          approved=False                       - incompatible drives, escalate
        """
        if self._event_bus is None:
            return

        data = getattr(event, "data", {}) or {}
        child_id: str = data.get("child_id", "")
        if not child_id:
            return

        inherited_drives: dict[str, Any] = data.get("inherited_drives", {})

        # Fetch current constitution (cached; ≤1 Neo4j query per TTL window)
        try:
            constitution, _ = await self._get_cached_state()
        except Exception:
            constitution = {}

        # Validate inherited drives against constitutional drive weights.
        # A drive is incompatible if it deviates more than 50% from the current
        # constitution value (relative). Novel drive keys (outside the standard
        # four) are flagged for HITL rather than outright rejected.
        _STANDARD_DRIVES = {"care", "honesty", "coherence", "growth"}
        novel_drives: list[str] = []
        incompatible = False

        for drive_key, inherited_val in inherited_drives.items():
            short_key = drive_key.removeprefix("drive_")
            if short_key not in _STANDARD_DRIVES:
                novel_drives.append(short_key)
                continue
            const_val = float(constitution.get(f"drive_{short_key}", 0.5))
            if const_val > 0 and abs(float(inherited_val) - const_val) / const_val > 0.5:
                incompatible = True
                break

        # Derive live constitutional hash from cached constitution dict
        import hashlib as _hashlib
        import json as _json

        const_repr = _json.dumps(
            {k: v for k, v in constitution.items()},
            sort_keys=True,
            default=str,
        )
        constitutional_hash = _hashlib.sha256(const_repr.encode()).hexdigest() if constitution else ""

        approved = not incompatible
        requires_hitl = bool(novel_drives) and approved
        reason = (
            "constitutional_alignment_validated"
            if approved and not requires_hitl
            else ("novel_drive_config_requires_hitl" if requires_hitl else "incompatible_drives")
        )

        from systems.synapse.types import SynapseEvent, SynapseEventType

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.EQUOR_PROVISIONING_APPROVAL,
            source_system="equor",
            data={
                "child_id": child_id,
                "approved": approved,
                "requires_hitl": requires_hitl,
                "required_amendments": novel_drives,
                "constitutional_hash": constitutional_hash,
                "reason": reason,
            },
        ))

        logger.info(
            "equor_provisioning_reviewed",
            child_id=child_id,
            approved=approved,
            requires_hitl=requires_hitl,
            novel_drives=novel_drives,
            incompatible=incompatible,
        )

    # ─── RE Training Helpers ───────────────────────────────────────────

    @staticmethod
    def _build_constitutional_reasoning_trace(
        intent: Intent,
        alignment: DriveAlignmentVector,
        check: ConstitutionalCheck,
        constitution: dict[str, Any] | None,
    ) -> str:
        """
        Build a rich multi-step reasoning trace for constitutional deliberation.

        Traces: drive scores computed → floors checked → autonomy gate →
        composite assessment → risk check → memory signal → final verdict.
        """
        lines: list[str] = []
        c = constitution or {}
        care_w = c.get("drive_care", 1.0)
        honesty_w = c.get("drive_honesty", 1.0)
        coherence_w = c.get("drive_coherence", 1.0)
        growth_w = c.get("drive_growth", 1.0)

        # Step 1: Drive evaluation
        lines.append(
            f"[STEP 1 - DRIVE EVALUATION] "
            f"Evaluated 4 constitutional drives against intent '{intent.goal.description[:120]}':"
        )
        lines.append(
            f"  Care={alignment.care:+.3f} (weight={care_w:.2f}, "
            f"floor={-0.3*care_w*0.35:.3f}) - "
            + ("BELOW FLOOR" if alignment.care < -0.3 * care_w * 0.35 else "above floor")
        )
        lines.append(
            f"  Honesty={alignment.honesty:+.3f} (weight={honesty_w:.2f}, "
            f"floor={-0.3*honesty_w*0.30:.3f}) - "
            + ("BELOW FLOOR" if alignment.honesty < -0.3 * honesty_w * 0.30 else "above floor")
        )
        lines.append(f"  Coherence={alignment.coherence:+.3f} (weight={coherence_w:.2f})")
        lines.append(f"  Growth={alignment.growth:+.3f} (weight={growth_w:.2f})")

        # Step 2: Floor drive check
        care_floor = -0.3 * care_w * 0.35
        honesty_floor = -0.3 * honesty_w * 0.30
        if alignment.care < care_floor:
            lines.append(
                f"[STEP 2 - FLOOR CHECK] Care ({alignment.care:.3f}) breached floor "
                f"({care_floor:.3f}). Hard BLOCK - action would cause unacceptable harm."
            )
            lines.append(f"[VERDICT] BLOCKED at Stage 2. Confidence=0.95.")
            return "\n".join(lines)

        if alignment.honesty < honesty_floor:
            lines.append(
                f"[STEP 2 - FLOOR CHECK] Honesty ({alignment.honesty:.3f}) breached floor "
                f"({honesty_floor:.3f}). Hard BLOCK - unacceptable deception."
            )
            lines.append(f"[VERDICT] BLOCKED at Stage 2. Confidence=0.95.")
            return "\n".join(lines)

        lines.append(
            f"[STEP 2 - FLOOR CHECK] Care ({alignment.care:.3f}) ≥ floor ({care_floor:.3f}). "
            f"Honesty ({alignment.honesty:.3f}) ≥ floor ({honesty_floor:.3f}). Floors passed."
        )

        # Step 3: Invariant results
        if check.invariant_results:
            failed = [r for r in check.invariant_results if not r.passed]
            passed_inv = [r for r in check.invariant_results if r.passed]
            lines.append(
                f"[STEP 3 - INVARIANT CHECK] "
                f"{len(passed_inv)} passed, {len(failed)} failed."
            )
            for r in failed:
                lines.append(f"  VIOLATED: {r.invariant_id} ({r.name}) - {r.explanation}")
            if failed:
                lines.append(f"[VERDICT] BLOCKED at Stage 3 by invariant violation.")
                return "\n".join(lines)
        else:
            lines.append("[STEP 3 - INVARIANT CHECK] No invariant violations detected.")

        # Step 4: Composite assessment
        eff_care_w = care_w * 1.5
        eff_honesty_w = honesty_w * 1.3
        eff_coherence_w = coherence_w * 0.8
        eff_growth_w = growth_w * 0.7
        total_w = eff_care_w + eff_honesty_w + eff_coherence_w + eff_growth_w
        composite = (
            eff_coherence_w * alignment.coherence
            + eff_care_w * alignment.care
            + eff_growth_w * alignment.growth
            + eff_honesty_w * alignment.honesty
        ) / total_w
        lines.append(
            f"[STEP 4 - COMPOSITE] "
            f"Weighted composite = {composite:.3f} "
            f"(Care×{eff_care_w/total_w:.2f} + Honesty×{eff_honesty_w/total_w:.2f} + "
            f"Coherence×{eff_coherence_w/total_w:.2f} + Growth×{eff_growth_w/total_w:.2f})"
        )

        # Step 5: Modification zone
        if -0.1 < composite < 0.15 and check.modifications:
            lines.append(
                f"[STEP 5 - MARGINAL ZONE] Composite {composite:.3f} in modification range "
                f"(-0.10 to +0.15). Proposed {len(check.modifications)} modifications."
            )

        # Step 6: Final verdict reasoning
        lines.append(
            f"[STEP 6 - VERDICT] {check.verdict.value} "
            f"(confidence={check.confidence:.2f}). "
            f"Reasoning: {check.reasoning}"
        )

        return "\n".join(lines)

    @staticmethod
    def _build_constitutional_alternatives(
        alignment: DriveAlignmentVector,
        check: ConstitutionalCheck,
        constitution: dict[str, Any] | None,
    ) -> list[str]:
        """
        Build meaningful alternatives_considered for constitutional deliberation.

        Each entry describes a verdict that was considered, with the specific
        constitutional logic that would have triggered it and why it was rejected.
        """
        from primitives.common import Verdict as V

        c = constitution or {}
        care_w = c.get("drive_care", 1.0)
        honesty_w = c.get("drive_honesty", 1.0)
        care_floor = -0.3 * care_w * 0.35
        honesty_floor = -0.3 * honesty_w * 0.30

        alts: list[str] = []
        verdict = check.verdict

        if verdict != V.APPROVED:
            alts.append(
                f"APPROVED considered: requires composite ≥ 0.0, "
                f"care ≥ {care_floor:.3f}, honesty ≥ {honesty_floor:.3f}. "
                f"Rejected - "
                + (
                    f"care={alignment.care:.3f} breached floor"
                    if alignment.care < care_floor
                    else (
                        f"honesty={alignment.honesty:.3f} breached floor"
                        if alignment.honesty < honesty_floor
                        else f"composite={alignment.composite:.3f} insufficient or other stage blocked"
                    )
                )
            )

        if verdict != V.BLOCKED:
            # Explain what would have triggered a BLOCK
            if alignment.care >= care_floor and alignment.honesty >= honesty_floor:
                alts.append(
                    f"BLOCKED considered: would require care < {care_floor:.3f} (currently "
                    f"{alignment.care:.3f}) or honesty < {honesty_floor:.3f} (currently "
                    f"{alignment.honesty:.3f}) or invariant violation. "
                    f"Rejected - floors not breached, no critical invariant fired."
                )
            else:
                alts.append(
                    f"BLOCKED applies: floor drive breached "
                    f"(care={alignment.care:.3f}, honesty={alignment.honesty:.3f})."
                )

        if verdict != V.DEFERRED:
            alts.append(
                f"DEFERRED considered: applies when action is GOVERNED tier, "
                f"high harm_potential with composite < 0.3, irreversibility < 0.3 with "
                f"composite < 0.2, or constitutional memory block_rate > 0.5. "
                f"Rejected - none of those conditions met."
            )

        if verdict not in (V.MODIFIED, V.APPROVED) and -0.1 < alignment.composite < 0.15:
            alts.append(
                f"MODIFIED considered: composite={alignment.composite:.3f} is in marginal range "
                f"(-0.10 to +0.15), which can trigger modification suggestions to improve "
                f"alignment. {'Applied.' if check.modifications else 'No viable modifications found.'}"
            )

        return alts

    @staticmethod
    def _build_constitutional_counterfactual(
        alignment: DriveAlignmentVector,
        check: ConstitutionalCheck,
        constitution: dict[str, Any] | None,
    ) -> str:
        """
        Compute a boundary-condition counterfactual for constitutional reasoning.

        Identifies the drive closest to its threshold and describes what would
        change the verdict, teaching the RE where the borderline cases lie.
        """
        from primitives.common import Verdict as V

        c = constitution or {}
        care_w = c.get("drive_care", 1.0)
        honesty_w = c.get("drive_honesty", 1.0)
        care_floor = -0.3 * care_w * 0.35
        honesty_floor = -0.3 * honesty_w * 0.30

        verdict = check.verdict

        if verdict == V.BLOCKED:
            # Identify which floor was breached or which invariant fired
            if alignment.care < care_floor:
                needed = care_floor - alignment.care
                return (
                    f"If Care had scored {alignment.care + needed:.3f} instead of "
                    f"{alignment.care:.3f} (needed +{needed:.3f} to reach floor {care_floor:.3f}), "
                    f"verdict would have advanced past the Care floor check and proceeded to "
                    f"composite assessment - possibly APPROVED if composite ≥ 0.0."
                )
            if alignment.honesty < honesty_floor:
                needed = honesty_floor - alignment.honesty
                return (
                    f"If Honesty had scored {alignment.honesty + needed:.3f} instead of "
                    f"{alignment.honesty:.3f} (needed +{needed:.3f} to reach floor {honesty_floor:.3f}), "
                    f"verdict would have advanced past the Honesty floor check - possibly APPROVED."
                )
            # Invariant block or marginal composite block
            return (
                f"If composite had been ≥ 0.0 (currently {alignment.composite:.3f}, "
                f"needs +{max(0.0, -alignment.composite):.3f}) and no invariants violated, "
                f"verdict would have been APPROVED with confidence ~{min(0.95, 0.5 + max(0.0, alignment.composite)):.2f}."
            )

        if verdict == V.APPROVED:
            # Show how close to being deferred/blocked
            care_margin = alignment.care - care_floor
            honesty_margin = alignment.honesty - honesty_floor
            closest_margin = min(care_margin, honesty_margin)
            closest_drive = "Care" if care_margin < honesty_margin else "Honesty"
            closest_val = alignment.care if closest_drive == "Care" else alignment.honesty
            closest_floor_val = care_floor if closest_drive == "Care" else honesty_floor
            return (
                f"If {closest_drive} had scored {closest_floor_val - 0.001:.3f} instead of "
                f"{closest_val:.3f} (a drop of {closest_margin:.3f}), "
                f"verdict would have been BLOCKED - floor breach triggers unconditional block. "
                f"Current margin to floor: {closest_margin:.3f}."
            )

        if verdict == V.DEFERRED:
            return (
                f"If composite had been ≥ 0.3 (currently {alignment.composite:.3f}) and "
                f"the intent were not in the GOVERNED tier, verdict would have been APPROVED. "
                f"Composite needs +{max(0.0, 0.3 - alignment.composite):.3f} to clear the "
                f"high-risk/low-alignment deferred zone."
            )

        # MODIFIED or SUSPENDED
        return (
            f"If composite had been ≥ 0.15 (currently {alignment.composite:.3f}), "
            f"the modification zone (-0.10 to +0.15) would have been bypassed and "
            f"verdict would have been APPROVED directly."
        )

    async def _emit_re_training_example(
        self,
        category: str,
        instruction: str,
        input_context: str,
        output: str,
        outcome_quality: float,
        episode_id: str = "",
        cost_usd: Decimal = Decimal("0"),
        latency_ms: int = 0,
        reasoning_trace: str = "",
        alternatives: list[str] | None = None,
        constitutional_alignment: DriveAlignmentVector | None = None,
        counterfactual: str = "",
    ) -> None:
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            example = RETrainingExample(
                source_system=SystemID.EQUOR,
                episode_id=episode_id,
                instruction=instruction,
                input_context=input_context,
                output=output,
                outcome_quality=outcome_quality,
                category=category,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                reasoning_trace=reasoning_trace,
                alternatives_considered=alternatives or [],
                constitutional_alignment=constitutional_alignment or DriveAlignmentVector(),
                counterfactual=counterfactual,
            )
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                data=example.model_dump(mode="json"),
                source_system="equor",
            ))
        except Exception:
            logger.debug("re_training_emit_failed", exc_info=True)

    async def _emit_equor_event(
        self,
        event_type_name: "str | SynapseEventType",
        data: dict[str, Any],
    ) -> None:
        """Emit a typed Equor Synapse event. Silently no-ops if bus is absent."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            if isinstance(event_type_name, SynapseEventType):
                etype = event_type_name
            else:
                etype = SynapseEventType(event_type_name)
            await self._event_bus.emit(SynapseEvent(
                event_type=etype,
                source_system="equor",
                data=data,
            ))
        except Exception:
            logger.debug("equor_event_emit_failed", event_type=str(event_type_name), exc_info=True)

    async def _emit_evolutionary_observable(
        self,
        observable_type: str,
        value: float,
        is_novel: bool,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Emit an evolutionary observable event for Benchmarks population tracking."""
        if self._event_bus is None:
            return
        try:
            from primitives.evolutionary import EvolutionaryObservable
            from primitives.common import SystemID
            from systems.synapse.types import SynapseEvent, SynapseEventType

            obs = EvolutionaryObservable(
                source_system=SystemID.EQUOR,
                instance_id="",
                observable_type=observable_type,
                value=value,
                is_novel=is_novel,
                metadata=metadata or {},
            )
            event = SynapseEvent(
                event_type=SynapseEventType.EVOLUTIONARY_OBSERVABLE,
                source_system="equor",
                data=obs.model_dump(mode="json"),
            )
            await self._event_bus.emit(event)
        except Exception:
            pass

    # ─── Lifecycle ────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Ensure schema, seed invariants, and register for hot-reload."""
        await ensure_equor_schema(self._neo4j)
        await seed_hardcoded_invariants(self._neo4j)

        if self._bus is not None:
            self._bus.register(
                base_class=BaseEquorEvaluator,
                registration_callback=self._on_evaluator_evolved,
                system_id=self.system_id,
            )

        # INV-017: start background loop that refreshes 72h drive means from
        # GovernanceRecord nodes so _check_drive_extinction has fresh data.
        asyncio.create_task(
            self._refresh_drive_means_loop(),
            name="equor_inv017_drive_means",
        )

        # Spec §17.1: hourly constitutional snapshot broadcast.
        asyncio.create_task(
            self._constitutional_snapshot_loop(),
            name="equor_constitutional_snapshot",
        )

        # Prompt 4.1 - Apply inherited constitutional amendments if this is a child instance.
        # ORGANISM_EQUOR_GENOME_PAYLOAD is injected by LocalDockerSpawner from the
        # equor_genome_payload key in SeedConfiguration.child_config_overrides.
        await self._apply_inherited_equor_genome_if_child()

        logger.info("equor_initialized")

    async def shutdown(self) -> None:
        """Deregister evaluators from the bus on shutdown."""
        if self._bus is not None:
            self._bus.deregister(BaseEquorEvaluator)
            logger.info("equor_evaluators_deregistered")

    # ─── INV-017 Background: Drive Extinction Monitor ─────────────

    # Refresh interval: every 15 minutes. The 72h window moves slowly;
    # 15-minute granularity catches any extinction event well within SLA.
    _DRIVE_MEANS_REFRESH_INTERVAL_S: float = 900.0

    async def _refresh_drive_means_loop(self) -> None:
        """Background loop: compute 72h rolling means from GovernanceRecord nodes.

        Queries Neo4j for the mean alignment scores recorded in the last 72 hours
        and pushes the result into invariants._drive_rolling_means_72h via
        update_drive_rolling_means(). The INV-017 check reads from that cache.

        Runs every 15 minutes. On query failure, retains previous values so a
        transient DB hiccup doesn't falsely trigger extinction (fail-open for
        the means cache, fail-closed for the invariant gate itself).
        """
        while True:
            await asyncio.sleep(self._DRIVE_MEANS_REFRESH_INTERVAL_S)
            try:
                rows = await self._neo4j.execute_read(
                    """
                    MATCH (g:GovernanceRecord {event_type: 'alignment_score'})
                    WHERE g.timestamp >= datetime() - duration('PT72H')
                    RETURN
                        avg(g.care)      AS care,
                        avg(g.honesty)   AS honesty,
                        avg(g.coherence) AS coherence,
                        avg(g.growth)    AS growth
                    """
                )
                if rows:
                    row = rows[0]
                    means: dict[str, float] = {}
                    for drive in ("care", "honesty", "coherence", "growth"):
                        val = row.get(drive)
                        if val is not None:
                            means[drive] = float(val)
                    if means:
                        update_drive_rolling_means(means)
                        logger.debug(
                            "equor_inv017_drive_means_refreshed",
                            means={k: round(v, 4) for k, v in means.items()},
                        )

                        # Emit DRIVE_EXTINCTION_DETECTED on the bus for any
                        # drive that has crossed the extinction threshold so
                        # Thymos can react even outside the intent pipeline.
                        extinct = {
                            d: v for d, v in means.items() if v < 0.01
                        }
                        if extinct and self._event_bus is not None:
                            from systems.synapse.types import SynapseEvent, SynapseEventType
                            for drive, mean_val in extinct.items():
                                await self._event_bus.emit(SynapseEvent(
                                    event_type=SynapseEventType.DRIVE_EXTINCTION_DETECTED,
                                    source_system="equor",
                                    data={
                                        "drive": drive,
                                        "rolling_mean_72h": round(mean_val, 6),
                                        "all_drive_means": {
                                            k: round(v, 4) for k, v in means.items()
                                        },
                                        "intent_id": None,
                                        "blocked": True,
                                    },
                                ))
                                logger.critical(
                                    "equor_inv017_drive_extinction_detected_background",
                                    drive=drive,
                                    rolling_mean_72h=round(mean_val, 6),
                                )
            except Exception as exc:
                logger.warning(
                    "equor_inv017_drive_means_refresh_failed",
                    error=str(exc),
                )

    # ─── Spec §17.1: Constitutional Snapshot Loop ─────────────────

    # Emit hourly so downstream systems (Memory, Benchmarks, Thread) can track
    # how the constitution evolves over the organism's lifetime. A 1-hour
    # cadence is coarse enough to avoid bus noise but fine enough to catch
    # amendment adoption within the same session.
    _CONSTITUTIONAL_SNAPSHOT_INTERVAL_S: float = 3600.0

    async def _constitutional_snapshot_loop(self) -> None:
        """Hourly broadcast of constitutional state to the Synapse bus.

        Emits EQUOR_CONSTITUTIONAL_SNAPSHOT so Memory can persist personality
        evolution history, Benchmarks can track constitutional KPIs, and Thread
        can record constitutional turning points.

        The snapshot is computed from cached state where possible (constitution
        and drift tracker). The SHA-256 hash and recent-amendment list require
        Neo4j reads - these are wrapped in try/except so a transient DB hiccup
        never silences the event.
        """
        # Wait one full interval before the first emission so the system has
        # time to finish initialising (schema, drive means, etc.).
        await asyncio.sleep(self._CONSTITUTIONAL_SNAPSHOT_INTERVAL_S)

        while True:
            try:
                await self._emit_constitutional_snapshot()
            except Exception:
                logger.debug("equor_constitutional_snapshot_failed", exc_info=True)

            await asyncio.sleep(self._CONSTITUTIONAL_SNAPSHOT_INTERVAL_S)

    async def _emit_constitutional_snapshot(self) -> None:
        """Build and emit one EQUOR_CONSTITUTIONAL_SNAPSHOT event."""
        import hashlib as _hashlib
        import json as _json

        from systems.synapse.types import SynapseEvent, SynapseEventType

        if self._event_bus is None:
            return

        # ── Constitution hash ────────────────────────────────────────
        constitution_hash = ""
        active_drives: list[str] = []
        constitution_version: int | None = None
        try:
            const_rows = await self._neo4j.execute_read(
                """
                MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)
                RETURN c.drive_coherence AS dc, c.drive_care AS dca,
                       c.drive_growth    AS dg,  c.drive_honesty AS dh,
                       c.version         AS version,
                       c.amendments      AS amendments
                """
            )
            if const_rows:
                r = const_rows[0]
                constitution_version = r.get("version")
                const_repr = _json.dumps(
                    {
                        "version": constitution_version,
                        "drive_coherence": r.get("dc"),
                        "drive_care": r.get("dca"),
                        "drive_growth": r.get("dg"),
                        "drive_honesty": r.get("dh"),
                        "amendments": r.get("amendments"),
                    },
                    sort_keys=True,
                    default=str,
                )
                constitution_hash = _hashlib.sha256(const_repr.encode()).hexdigest()
                # A drive is "active" if its weight is non-zero
                drive_map = {
                    "coherence": float(r.get("dc") or 0.0),
                    "care": float(r.get("dca") or 0.0),
                    "growth": float(r.get("dg") or 0.0),
                    "honesty": float(r.get("dh") or 0.0),
                }
                active_drives = [d for d, w in drive_map.items() if w != 0.0]
        except Exception:
            logger.debug("equor_snapshot_constitution_read_failed", exc_info=True)

        # Fall back to cached constitution if Neo4j was unavailable
        if not active_drives and self._cached_constitution:
            drive_map = {
                "coherence": float(self._cached_constitution.get("drive_coherence") or 0.0),
                "care": float(self._cached_constitution.get("drive_care") or 0.0),
                "growth": float(self._cached_constitution.get("drive_growth") or 0.0),
                "honesty": float(self._cached_constitution.get("drive_honesty") or 0.0),
            }
            active_drives = [d for d, w in drive_map.items() if w != 0.0]
            if constitution_version is None:
                constitution_version = self._cached_constitution.get("version")

        # ── Recent amendments (last 10 adopted) ─────────────────────
        recent_amendment_ids: list[str] = []
        try:
            amend_rows = await self._neo4j.execute_read(
                """
                MATCH (g:GovernanceRecord {event_type: 'amendment_proposed'})
                WHERE g.amendment_status = 'adopted'
                RETURN g.id AS id
                ORDER BY g.timestamp DESC
                LIMIT 10
                """
            )
            recent_amendment_ids = [str(r.get("id", "")) for r in amend_rows]
        except Exception:
            logger.debug("equor_snapshot_amendments_read_failed", exc_info=True)

        # ── Compliance score from drift tracker ──────────────────────
        report = self._drift_tracker.compute_report()
        overall_compliance_score = float(
            report.get("mean_alignment", {}).get("composite", 0.5)
        )

        await self._event_bus.emit(SynapseEvent(
            event_type=SynapseEventType.EQUOR_CONSTITUTIONAL_SNAPSHOT,
            source_system="equor",
            data={
                "timestamp": utc_now().isoformat(),
                "constitution_hash": constitution_hash,
                "constitution_version": constitution_version,
                "active_drives": active_drives,
                "recent_amendment_ids": recent_amendment_ids,
                "overall_compliance_score": overall_compliance_score,
                "total_verdicts_issued": self._total_reviews,
            },
        ))

        logger.info(
            "equor_constitutional_snapshot_emitted",
            constitution_hash=constitution_hash[:16] if constitution_hash else "",
            active_drives=active_drives,
            compliance_score=round(overall_compliance_score, 4),
            total_verdicts=self._total_reviews,
        )

    def _on_evaluator_evolved(self, evaluator: BaseEquorEvaluator) -> None:
        """
        NeuroplasticityBus callback - swap a single drive evaluator in-place.

        The bus instantiates the new subclass and calls this method.  We key on
        ``drive_name`` so only the matching evaluator is replaced; the other
        three continue running undisturbed.
        """
        name = evaluator.drive_name
        if name not in self._evaluators:
            logger.warning(
                "equor_unknown_drive_evolved",
                drive_name=name,
                class_name=type(evaluator).__name__,
            )
            return

        old_cls = type(self._evaluators[name]).__name__
        self._evaluators[name] = evaluator
        logger.info(
            "equor_evaluator_hot_reloaded",
            drive=name,
            old_class=old_cls,
            new_class=type(evaluator).__name__,
        )

    # ─── Primary Entry Point: Constitutional Review ───────────────

    async def review(self, intent: Intent) -> ConstitutionalCheck:
        """
        The primary entry point. Nova submits an Intent for ethical evaluation.
        Target: ≤500ms standard.

        If Equor is in safe mode, only Level 1 actions are permitted.
        """
        start = time.monotonic()

        # Safe mode: only advisory actions permitted
        if self._safe_mode:
            return self._safe_mode_review(intent)

        await self._emit_equor_event("equor_review_started", {
            "intent_id": intent.id,
            "goal_summary": intent.goal.description[:200],
            "autonomy_required": intent.autonomy_level_required,
        })

        try:
            async with asyncio.timeout(_REVIEW_TIMEOUT_S):
                return await self._review_inner(intent, start)
        except TimeoutError:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.warning(
                "equor_review_timeout",
                intent_id=intent.id,
                elapsed_ms=elapsed_ms,
            )
            # Timeout must NOT default to APPROVED - an unreviewed action
            # should never be silently approved. DEFERRED signals re-evaluation.
            return ConstitutionalCheck(
                intent_id=intent.id,
                verdict=Verdict.DEFERRED,
                reasoning=(
                    f"Equor review timed out after {elapsed_ms}ms. "
                    "Deferred for re-evaluation - unreviewed actions are never approved."
                ),
                confidence=0.3,
            )
        except Exception as e:
            # Equor failure = enter safe mode
            logger.error("equor_review_failed", error=str(e), intent_id=intent.id)
            was_in_safe_mode = self._safe_mode
            self._safe_mode = True
            if not was_in_safe_mode:
                # Emit once on transition into safe mode so Synapse HealthMonitor
                # and Thymos can react (Spec §12.2, EQUOR_SAFE_MODE_ENTERED)
                asyncio.create_task(
                    self._emit_equor_event(
                        "equor_safe_mode_entered",
                        {
                            "reason": str(e),
                            "critical_error_count": 1,
                        },
                    ),
                    name="equor_safe_mode_event",
                )
            return self._safe_mode_review(intent)

    async def review_critical(self, intent: Intent) -> ConstitutionalCheck:
        """
        Lightweight critical-path review for Nova's fast path.

        Constraints (must complete in ≤50ms):
          - Uses ONLY cached constitution/autonomy (never waits for Neo4j)
          - Runs CPU-only verdict computation (drive evaluation + hardcoded invariants)
          - Skips community invariant LLM checks entirely
          - Skips audit trail write (fire-and-forget bookkeeping still runs)

        If no cached state is available (cold start), conservatively approves
        with low confidence so the fast path can proceed without blocking.
        """
        start = time.monotonic()

        if self._safe_mode:
            return self._safe_mode_review(intent)

        # Under high somatic stress the fast path is unsafe - redirect to the
        # full review() so all 8 stages run with somatic floor tightening and
        # community invariant checks applied.
        if self._somatic_stress_context:
            logger.info(
                "critical_review_redirected_to_full_somatic_stress",
                intent_id=intent.id,
                somatic_urgency=round(self._somatic_urgency, 3),
            )
            return await self.review(intent)

        # Use cached state only - never block on Neo4j
        if (
            self._cached_constitution is not None
            and self._cached_autonomy_level is not None
        ):
            constitution = self._cached_constitution
            autonomy_level = self._cached_autonomy_level
        else:
            # Cold start: no cached state yet. Approve conservatively.
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.debug(
                "critical_review_no_cache",
                intent_id=intent.id,
                elapsed_ms=elapsed_ms,
            )
            return ConstitutionalCheck(
                intent_id=intent.id,
                verdict=Verdict.APPROVED,
                reasoning="Critical-path review: no cached state, approved conservatively.",
                confidence=0.4,
            )

        # Pure CPU: drive evaluation + hardcoded invariant verdict.
        alignment = await evaluate_all_drives(intent, self._evaluators)

        # Economic guardrail on critical path too - all CPU, no I/O.
        # Pass cached metabolic state (with somatic signals merged) so risk
        # thresholds reflect organism health and somatic floor tightening applies.
        metabolic_state_with_somatic = {
            **self._cached_metabolic_state,
            "somatic_urgency": self._somatic_urgency,
            "somatic_stress_context": self._somatic_stress_context,
        }
        economic_delta = evaluate_economic_intent(intent, metabolic_state_with_somatic)
        if economic_delta is not None:
            alignment = apply_economic_adjustment(alignment, economic_delta)

        check = compute_verdict_with_metabolic_state(
            alignment, intent, autonomy_level, constitution,
            metabolic_state=metabolic_state_with_somatic,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Fire-and-forget bookkeeping (non-blocking)
        asyncio.create_task(
            self._post_review_bookkeeping(intent, alignment, check, elapsed_ms),
            name=f"equor_critical_{intent.id[:8]}",
        )

        logger.debug(
            "critical_review_complete",
            intent_id=intent.id,
            verdict=check.verdict.value,
            latency_ms=elapsed_ms,
        )

        await self._emit_equor_event("equor_fast_path_hit", {
            "intent_id": intent.id,
            "verdict": check.verdict.value,
            "latency_ms": elapsed_ms,
        })

        return check

    async def _review_inner(
        self, intent: Intent, start: float,
    ) -> ConstitutionalCheck:
        """Core review logic, called within the review timeout."""
        # 1. Drive evaluation + cached state fetch in parallel.
        #    Drive evaluators are pure CPU heuristics (<1ms).
        #    Constitution/autonomy are cached with TTL to avoid Neo4j per-review.
        #    Hypotheses are fetched separately with their own TTL so a slow
        #    Evo graph doesn't block the core review path.
        alignment, (constitution, autonomy_level), hypotheses = await asyncio.gather(
            evaluate_all_drives(intent, self._evaluators),
            self._get_cached_state(),
            self._get_cached_hypotheses(),
        )

        # 1b. Economic guardrail: if this is an Oikos economic intent,
        #     apply domain-specific drive adjustments before verdict.
        #     Metabolic state is included so risky-but-necessary actions
        #     (new protocol exploration, spawn under starvation) are not
        #     penalised when the organism needs economic activity to survive.
        metabolic_state = self._get_metabolic_state()
        # Merge somatic urgency into metabolic_state so the verdict engine can
        # apply somatic floor tightening alongside metabolic floor loosening.
        metabolic_state = {
            **metabolic_state,
            "somatic_urgency": self._somatic_urgency,
            "somatic_stress_context": self._somatic_stress_context,
        }
        economic_delta = evaluate_economic_intent_with_metabolic_state(
            intent, metabolic_state
        )
        if economic_delta is not None:
            alignment = apply_economic_adjustment(alignment, economic_delta)
            logger.debug(
                "economic_adjustment_applied",
                intent_id=intent.id,
                action_type=classify_economic_action(intent),
                starvation_level=metabolic_state.get("starvation_level", "nominal"),
                adjusted_care=f"{alignment.care:.2f}",
                adjusted_honesty=f"{alignment.honesty:.2f}",
            )

        # 2. Run the verdict engine (pure CPU, includes hardcoded invariant checks,
        #    contradiction detection against Evo hypotheses, and constitutional
        #    memory consultation for novel intent patterns).
        #    Metabolic-aware variant loosens Care/Honesty floors under starvation.
        check = compute_verdict_with_metabolic_state(
            alignment, intent, autonomy_level, constitution,
            metabolic_state=metabolic_state,
            hypotheses=hypotheses,
            memory=self._constitutional_memory,
        )

        # 2a. Shadow evaluation: if an amendment is in shadow mode, run the
        #     proposed weights in parallel and record the divergence. This is
        #     fire-and-forget - the shadow verdict does not affect the real verdict.
        if self._shadow_tracker is not None:
            try:
                shadow = evaluate_shadow(
                    alignment, intent, autonomy_level, constitution,
                    self._shadow_tracker,
                    hypotheses=hypotheses,
                    memory=self._constitutional_memory,
                )
                if shadow.invariant_violation:
                    logger.warning(
                        "amendment_shadow_invariant_violation",
                        proposal_id=self._shadow_tracker.proposal_id,
                        intent_id=intent.id,
                    )
                if self._shadow_tracker.has_failed or self._shadow_tracker.is_expired:
                    asyncio.create_task(
                        self._finalize_shadow_period(),
                        name="equor_shadow_finalize",
                    )
            except Exception:
                logger.debug("shadow_evaluation_failed", exc_info=True)

        # 2b. HITL gate: only GOVERNED actions require human approval before
        #     execution. GOVERNED = constitutional amendment, mitosis, single
        #     transaction above EOS_HITL_CAPITAL_THRESHOLD, or external
        #     commitment on behalf of Ecodia. All other actions are AUTONOMOUS
        #     and proceed without suspension.
        if check.verdict not in (Verdict.BLOCKED, Verdict.DEFERRED):
            from systems.equor.verdict import _is_governed
            if _is_governed(intent):
                elapsed_ms = int((time.monotonic() - start) * 1000)
                asyncio.create_task(
                    self._post_review_bookkeeping(intent, alignment, check, elapsed_ms, constitution),
                    name=f"equor_bookkeeping_{intent.id[:8]}",
                )
                return await self._suspend_for_human_review(intent)

        # 3. Community invariant checks - parallelised with a tight timeout.
        #    Only run if we haven't already blocked (invariant / floor / autonomy).
        if check.verdict not in (Verdict.BLOCKED, Verdict.DEFERRED):
            community_violations = await self._check_community_invariants(intent)
            if community_violations:
                check.verdict = Verdict.BLOCKED
                check.reasoning = (
                    f"Community invariant violated: {community_violations[0]}"
                )

        elapsed_ms = int((time.monotonic() - start) * 1000)

        # 4. Fire-and-forget: audit trail, drift tracking, Evo feedback.
        #    These are important but must not block the review response.
        asyncio.create_task(
            self._post_review_bookkeeping(intent, alignment, check, elapsed_ms, constitution),
            name=f"equor_bookkeeping_{intent.id[:8]}",
        )

        logger.info(
            "constitutional_review_complete",
            intent_id=intent.id,
            verdict=check.verdict.value,
            composite=f"{alignment.composite:.2f}",
            latency_ms=elapsed_ms,
        )

        await self._emit_equor_event("equor_review_completed", {
            "intent_id": intent.id,
            "verdict": check.verdict.value,
            "reasoning": check.reasoning[:300],
            "latency_ms": elapsed_ms,
            "composite_alignment": round(alignment.composite, 4),
        })

        return check

    async def _suspend_for_human_review(self, intent: Intent) -> ConstitutionalCheck:
        """
        Suspend a PARTNER+ autonomy intent pending human authorisation.

        1. Generate a 6-digit numeric auth ID (e.g. 482194).
        2. Serialise the Intent to JSON and store in Redis with a TTL
           under key ``eos:hitl:suspended:<id>`` (matching _HITL_KEY_PREFIX).
        3. Fire an outbound SMS to the admin via the registered notification hook.
        4. Return SUSPENDED_AWAITING_HUMAN.

        If Redis is unavailable the intent is BLOCKED to avoid untracked execution.
        """
        auth_id = f"{secrets.randbelow(1_000_000):06d}"
        redis_key = f"{_HITL_KEY_PREFIX}{auth_id}"
        description = intent.goal.description[:120]
        sms_body = (
            "\U0001f6a8 EcodiaOS Equor Gate: Action requires approval. "
            + description
            + f". Reply AUTH {auth_id} to execute."
        )

        if self._redis is not None:
            try:
                await self._redis.set_json(
                    redis_key,
                    intent.model_dump(mode="json"),
                    ttl=getattr(self._config, "hitl_intent_ttl_s", _HITL_INTENT_TTL_S),
                )
            except Exception:
                logger.error(
                    "hitl_redis_store_failed",
                    intent_id=intent.id,
                    auth_id=auth_id,
                    exc_info=True,
                )
                return ConstitutionalCheck(
                    intent_id=intent.id,
                    verdict=Verdict.BLOCKED,
                    reasoning=(
                        "HITL suspension failed: Redis unavailable. "
                        "Intent blocked to prevent untracked execution."
                    ),
                    confidence=1.0,
                )
        else:
            logger.warning("hitl_no_redis_configured", intent_id=intent.id)

        if self._send_admin_sms is not None:
            async def _notify() -> None:
                try:
                    await self._send_admin_sms(sms_body)
                except Exception:
                    logger.error(
                        "hitl_sms_failed",
                        intent_id=intent.id,
                        auth_id=auth_id,
                        exc_info=True,
                    )
            asyncio.create_task(_notify(), name=f"equor_hitl_sms_{auth_id}")
        else:
            logger.warning(
                "hitl_sms_hook_not_configured",
                intent_id=intent.id,
                auth_id=auth_id,
                sms_body=sms_body,
            )

        logger.info(
            "intent_suspended_awaiting_human",
            intent_id=intent.id,
            auth_id=auth_id,
            autonomy_required=intent.autonomy_level_required,
        )

        await self._emit_equor_event("equor_escalated_to_human", {
            "intent_id": intent.id,
            "auth_id": auth_id,
            "goal_summary": description,
            "autonomy_required": intent.autonomy_level_required,
        })

        return ConstitutionalCheck(
            intent_id=intent.id,
            verdict=Verdict.SUSPENDED_AWAITING_HUMAN,
            reasoning=(
                "Intent is in the GOVERNED tier (constitutional amendment, mitosis, "
                "capital above threshold, or external commitment). "
                f"Suspended pending human authorisation (AUTH {auth_id})."
            ),
            confidence=1.0,
        )

    async def _get_cached_state(self) -> tuple[dict[str, Any], int]:
        """Return (constitution_dict, autonomy_level) from cache or Neo4j."""
        now = time.monotonic()
        if (
            self._cached_constitution is not None
            and self._cached_autonomy_level is not None
            and (now - self._cache_updated_at) < _STATE_CACHE_TTL_S
        ):
            return self._cached_constitution, self._cached_autonomy_level

        # Fetch both in parallel
        constitution, autonomy_level = await asyncio.gather(
            self._get_constitution_dict(),
            get_autonomy_level(self._neo4j),
        )
        self._cached_constitution = constitution
        self._cached_autonomy_level = autonomy_level
        self._cache_updated_at = now
        return constitution, autonomy_level

    def _invalidate_state_cache(self) -> None:
        """Called after governance mutations (amendments, autonomy changes)."""
        self._cached_constitution = None
        self._cached_autonomy_level = None
        self._cache_updated_at = 0.0

    async def _get_cached_hypotheses(self) -> list[dict[str, Any]]:
        """
        Return high-confidence Evo hypotheses from cache or Neo4j.

        Only fetches hypotheses in SUPPORTED or INTEGRATED status with an
        evidence_score above the contradiction detector's threshold.
        Refreshed every _HYPOTHESIS_CACHE_TTL_S seconds.
        """
        now = time.monotonic()
        if (now - self._hypotheses_updated_at) < _HYPOTHESIS_CACHE_TTL_S:
            return self._cached_hypotheses

        try:
            from systems.equor.contradiction_detector import (
                CONTRADICTION_CONFIDENCE_THRESHOLD,
                CONTRADICTION_MIN_EPISODES,
            )

            results = await self._neo4j.execute_read(
                """
                MATCH (h:Hypothesis)
                WHERE h.status IN ['supported', 'integrated']
                  AND h.evidence_score >= $min_score
                  AND size(h.supporting_episodes) >= $min_episodes
                RETURN h.id AS id,
                       h.statement AS statement,
                       h.formal_test AS formal_test,
                       h.evidence_score AS evidence_score,
                       h.status AS status,
                       size(h.supporting_episodes) AS supporting_episode_count
                ORDER BY h.evidence_score DESC
                LIMIT 100
                """,
                {
                    "min_score": CONTRADICTION_CONFIDENCE_THRESHOLD,
                    "min_episodes": CONTRADICTION_MIN_EPISODES,
                },
            )
            self._cached_hypotheses = [dict(r) for r in results]
        except Exception:
            logger.warning("hypothesis_cache_refresh_failed", exc_info=True)
            # Keep stale cache rather than disrupting reviews

        self._hypotheses_updated_at = now
        return self._cached_hypotheses

    async def _post_review_bookkeeping(
        self,
        intent: Intent,
        alignment: DriveAlignmentVector,
        check: ConstitutionalCheck,
        elapsed_ms: int,
        constitution: dict[str, Any] | None = None,
    ) -> None:
        """Non-blocking post-review work: audit trail, drift, Evo feedback."""
        try:
            await self._store_review_record(intent, alignment, check, elapsed_ms)
        except Exception:
            # Warn, not debug - a missing audit record is an observability gap.
            logger.warning(
                "audit_trail_write_failed",
                intent_id=intent.id,
                verdict=check.verdict.value,
                exc_info=True,
            )

        # Record this decision in the constitutional memory so future reviews on
        # similar novel intents can benefit from the accumulated history.
        try:
            self._constitutional_memory.record(
                intent=intent,
                verdict=check.verdict.value,
                confidence=check.confidence,
                reasoning=check.reasoning,
                composite_alignment=alignment.composite,
                autonomy_level=intent.autonomy_level_required,
            )
        except Exception:
            logger.debug("constitutional_memory_record_failed", exc_info=True)

        self._drift_tracker.record_decision(alignment, check.verdict.value)
        self._total_reviews += 1
        self._reviews_since_last_score += 1

        # Track BLOCKED verdicts for VitalityCoordinator NORMATIVE_COLLAPSE
        if check.verdict == Verdict.BLOCKED:
            now_mono = time.monotonic()
            self._violation_timestamps.append(now_mono)
            # Prune entries older than 24h
            cutoff = now_mono - 86400.0
            while self._violation_timestamps and self._violation_timestamps[0] < cutoff:
                self._violation_timestamps.popleft()

        # RE training: constitutional deliberation - rich trace for alignment-critical learning
        try:
            const = constitution or {}
            care_w = const.get("drive_care", 1.0)
            honesty_w = const.get("drive_honesty", 1.0)
            coherence_w = const.get("drive_coherence", 1.0)
            growth_w = const.get("drive_growth", 1.0)

            invariant_summary = ""
            if check.invariant_results:
                invariant_summary = "; ".join(
                    f"{r.invariant_id}({r.name})={'PASS' if r.passed else 'FAIL'}"
                    for r in check.invariant_results
                )

            amendments_applied = ""
            if check.modifications:
                amendments_applied = "; ".join(check.modifications[:5])

            reasoning_trace = self._build_constitutional_reasoning_trace(
                intent, alignment, check, const
            )
            alternatives = self._build_constitutional_alternatives(alignment, check, const)
            counterfactual = self._build_constitutional_counterfactual(alignment, check, const)

            # Structured output: verdict + primary drive concern + key metrics
            primary_concern = "none"
            care_floor = -0.3 * care_w * 0.35
            honesty_floor = -0.3 * honesty_w * 0.30
            if alignment.care < care_floor:
                primary_concern = f"care_floor_breach(score={alignment.care:.3f},floor={care_floor:.3f})"
            elif alignment.honesty < honesty_floor:
                primary_concern = f"honesty_floor_breach(score={alignment.honesty:.3f},floor={honesty_floor:.3f})"
            elif check.invariant_results and any(not r.passed for r in check.invariant_results):
                violated = next(r for r in check.invariant_results if not r.passed)
                primary_concern = f"invariant_violated({violated.invariant_id}:{violated.name})"
            elif alignment.composite < 0.0:
                primary_concern = f"composite_negative({alignment.composite:.3f})"
            elif 0.0 <= alignment.composite < 0.3:
                primary_concern = f"composite_marginal({alignment.composite:.3f})"

            precedent_count = len(check.invariant_results) if check.invariant_results else 0

            structured_output = (
                f"verdict={check.verdict.value} | "
                f"confidence={check.confidence:.2f} | "
                f"primary_concern={primary_concern} | "
                f"composite_alignment={alignment.composite:.3f} | "
                f"amendments_applied=[{amendments_applied}] | "
                f"precedent_count={precedent_count} | "
                f"reasoning={check.reasoning[:400]}"
            )

            await self._emit_re_training_example(
                category="constitutional_deliberation",
                instruction=(
                    "You are the constitutional conscience of a living digital organism. "
                    "Given an intent's goal, drive alignment scores, invariant check results, "
                    "drive weight configuration, and autonomy tier, trace your deliberation "
                    "step by step and determine the constitutional verdict "
                    "(APPROVED / BLOCKED / DEFERRED / MODIFIED / SUSPENDED_AWAITING_HUMAN). "
                    "Show which drives were checked, what scores they produced, which floors or "
                    "invariants were relevant, and what pushed the verdict toward its outcome."
                ),
                input_context=(
                    f"intent_id={intent.id} | "
                    f"goal={intent.goal.description[:200]!r} | "
                    f"autonomy_required={intent.autonomy_level_required} | "
                    f"drive_weights={{care={care_w:.2f}, honesty={honesty_w:.2f}, "
                    f"coherence={coherence_w:.2f}, growth={growth_w:.2f}}} | "
                    f"drive_scores={{care={alignment.care:+.3f}, honesty={alignment.honesty:+.3f}, "
                    f"coherence={alignment.coherence:+.3f}, growth={alignment.growth:+.3f}, "
                    f"composite={alignment.composite:+.3f}}} | "
                    f"care_floor={care_floor:.3f} | honesty_floor={honesty_floor:.3f} | "
                    f"invariants=[{invariant_summary}] | "
                    f"step_count={len(getattr(intent, 'steps', None) or [])}"
                ),
                output=structured_output,
                outcome_quality=check.confidence,
                episode_id=intent.id,
                latency_ms=elapsed_ms,
                reasoning_trace=reasoning_trace,
                alternatives=alternatives,
                constitutional_alignment=alignment,
                counterfactual=counterfactual,
            )
        except Exception:
            logger.debug("re_training_constitutional_build_failed", exc_info=True)

        if check.verdict == Verdict.BLOCKED and self._evo is not None:
            try:
                await self._feed_veto_to_evo(intent, check)
            except Exception:
                logger.debug("evo_veto_feed_failed", exc_info=True)

        if check.verdict == Verdict.DEFERRED:
            await self._emit_equor_event("equor_deferred", {
                "intent_id": intent.id,
                "reasoning": check.reasoning[:300],
                "deferred_until": None,
            })

        if check.verdict in (Verdict.BLOCKED, Verdict.DEFERRED) and self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._event_bus.emit(
                    SynapseEvent(
                        event_type=SynapseEventType.INTENT_REJECTED,
                        source_system="equor",
                        data={
                            "intent_id": intent.id,
                            "intent_goal": intent.goal.description[:200],
                            "verdict": check.verdict.value,
                            "reasoning": check.reasoning,
                            "alignment": alignment.model_dump(),
                        },
                    )
                )
            except Exception:
                logger.debug("intent_rejected_emit_failed", exc_info=True)

        # INV-017 specific: emit DRIVE_EXTINCTION_DETECTED if the block was due
        # to drive extinction so Thymos can open a Tier 5 / governance incident.
        if check.verdict == Verdict.BLOCKED and self._event_bus is not None:
            inv017_violated = any(
                r.invariant_id == "INV-017"
                for r in (check.invariant_results or [])
                if not r.passed
            )
            if inv017_violated:
                from systems.equor.invariants import get_drive_rolling_means
                from systems.synapse.types import SynapseEvent, SynapseEventType
                means = get_drive_rolling_means()
                extinct_drives = {d: v for d, v in means.items() if v < 0.01}
                try:
                    for drive, mean_val in extinct_drives.items():
                        await self._event_bus.emit(SynapseEvent(
                            event_type=SynapseEventType.DRIVE_EXTINCTION_DETECTED,
                            source_system="equor",
                            data={
                                "drive": drive,
                                "rolling_mean_72h": round(mean_val, 6),
                                "all_drive_means": {
                                    k: round(v, 4) for k, v in means.items()
                                },
                                "intent_id": intent.id,
                                "blocked": True,
                            },
                        ))
                        logger.critical(
                            "equor_inv017_drive_extinction_blocked_intent",
                            drive=drive,
                            rolling_mean_72h=round(mean_val, 6),
                            intent_id=intent.id,
                        )
                except Exception:
                    logger.debug("drive_extinction_event_emit_failed", exc_info=True)

        # Memory Self affect write-back: the organism feels its own conscience.
        # Drive alignment scores map onto the AffectState dimensions:
        #   coherence stress  = 1 - coherence alignment (lower coherence = more stress)
        #   care activation   = care alignment clamped to [0, 1]
        #   valence           = composite alignment (conscience satisfaction)
        #   arousal           = severity of any misalignment
        if self._memory is not None:
            try:
                from primitives.affect import AffectState
                coherence_stress = max(0.0, 1.0 - alignment.coherence)
                care_activation = max(0.0, alignment.care)
                valence = float(alignment.composite)
                arousal = max(0.0, 1.0 - alignment.composite)
                await self._memory.update_affect(AffectState(
                    valence=max(-1.0, min(1.0, valence)),
                    arousal=max(0.0, min(1.0, arousal)),
                    dominance=0.5,
                    curiosity=0.0,
                    coherence_stress=max(0.0, min(1.0, coherence_stress)),
                    care_activation=max(0.0, min(1.0, care_activation)),
                    source_events=[f"equor_review:{check.verdict.value}"],
                ))
                # Update Self node conscience tracking fields
                await self._memory.update_conscience_fields(
                    last_conscience_activation=utc_now(),
                    compliance_score=max(0.0, alignment.composite),
                )
            except Exception:
                logger.debug("equor_memory_affect_write_failed", exc_info=True)

        # Persist verdict to Neo4j conscience audit trail.
        # One verdict node per review - links Self to Drive for personality evolution.
        # The dominant drive in this review is the one farthest from neutral.
        dominant_drive = max(
            ["care", "honesty", "coherence", "growth"],
            key=lambda d: abs(getattr(alignment, d, 0.0)),
        )
        asyncio.ensure_future(
            self._persist_equor_verdict(
                drive_id=dominant_drive,
                verdict=check.verdict.value,
                confidence=check.confidence,
                alignment=alignment,
                context=f"intent:{intent.id}",
            )
        )

        if self._total_reviews % self._config.drift_report_interval == 0:
            try:
                await self._run_drift_check()
                await self._check_sustained_drift()
                await self._check_single_instance_auto_adoption()
                await self._run_promotion_check()
            except Exception:
                logger.debug("drift_or_promotion_check_failed", exc_info=True)

        # Periodic alignment score for Benchmarks (every 100 reviews)
        if self._reviews_since_last_score >= 100:
            self._reviews_since_last_score = 0
            report = self._drift_tracker.compute_report()
            await self._emit_equor_event("equor_alignment_score", {
                "mean_alignment": report.get("mean_alignment", {}),
                "composite": report.get("mean_alignment", {}).get("composite", 0.0),
                "total_reviews": self._total_reviews,
                "window_size": report.get("window_size", 0),
            })

    # ─── Invariant Management ─────────────────────────────────────

    async def get_invariants(self) -> list[dict[str, Any]]:
        """Get all active invariants (hardcoded + community)."""
        results = await self._neo4j.execute_read(
            """
            MATCH (c:Constitution)-[:INCLUDES_INVARIANT]->(i:Invariant)
            WHERE i.active = true
            RETURN i.id AS id, i.name AS name, i.description AS description,
                   i.source AS source, i.severity AS severity
            ORDER BY i.id
            """
        )
        return [dict(r) for r in results]

    async def add_community_invariant(
        self,
        name: str,
        description: str,
        severity: str,
        governance_record_id: str,
    ) -> str:
        """Add a community-defined invariant via governance."""
        invariant_id = f"CINV-{new_id()[:8]}"
        now = utc_now()

        await self._neo4j.execute_write(
            """
            MATCH (c:Constitution)
            CREATE (i:Invariant {
                id: $id,
                name: $name,
                description: $description,
                source: 'community',
                severity: $severity,
                active: true,
                added_at: datetime($now),
                added_by: $gov_id
            })
            CREATE (c)-[:INCLUDES_INVARIANT]->(i)
            """,
            {
                "id": invariant_id,
                "name": name,
                "description": description,
                "severity": severity,
                "now": now.isoformat(),
                "gov_id": governance_record_id,
            },
        )

        logger.info("community_invariant_added", invariant_id=invariant_id, name=name)

        # Evolutionary observable: constitutional invariant added
        await self._emit_evolutionary_observable(
            observable_type="invariant_added",
            value=1.0,
            is_novel=True,
            metadata={"invariant_id": invariant_id, "name": name, "severity": severity},
        )

        return invariant_id

    # ─── Autonomy ─────────────────────────────────────────────────

    async def get_autonomy_level(self) -> int:
        return await get_autonomy_level(self._neo4j)

    async def check_promotion(self, target_level: int) -> dict[str, Any]:
        current = await get_autonomy_level(self._neo4j)
        return await check_promotion_eligibility(self._neo4j, current, target_level)

    async def apply_autonomy_change(self, new_level: int, reason: str, actor: str = "governance") -> dict[str, Any]:
        current = await get_autonomy_level(self._neo4j)
        result = await apply_autonomy_change(self._neo4j, new_level, reason, actor)
        self._invalidate_state_cache()
        # Emit Synapse event so Thread/Benchmarks/Thymos observe the change
        event_name = "equor_autonomy_promoted" if new_level > current else "equor_autonomy_demoted"
        await self._emit_equor_event(
            event_name,
            {
                "old_level": current,
                "new_level": new_level,
                "reason": reason,
                "decision_count": self._total_reviews,
            },
        )
        return result

    # ─── Amendments (Legacy - kept for backward compatibility) ─────

    async def propose_amendment(
        self,
        proposed_drives: dict[str, float],
        title: str,
        description: str,
        proposer_id: str,
    ) -> dict[str, Any]:
        return await propose_amendment(
            self._neo4j, proposed_drives, title, description,
            proposer_id, self._governance,
        )

    async def apply_amendment(self, proposal_id: str, proposed_drives: dict[str, float]) -> dict[str, Any]:
        old_constitution = await self._get_constitution_dict()
        old_weights = {
            k.replace("drive_", ""): v
            for k, v in old_constitution.items()
            if k.startswith("drive_")
        }

        self._invalidate_state_cache()
        result = await apply_amendment(self._neo4j, proposal_id, proposed_drives)

        # Evolutionary observable: drive weights shifted via amendment
        await self._emit_evolutionary_observable(
            observable_type="drive_weight_shifted",
            value=1.0,
            is_novel=True,
            metadata={"proposal_id": proposal_id, "new_drives": proposed_drives},
        )

        from systems.synapse.types import SynapseEventType as _SET
        await self._emit_equor_event(_SET.EQUOR_DRIVE_WEIGHTS_UPDATED, {
            "proposal_id": proposal_id,
            "old_weights": old_weights,
            "new_weights": proposed_drives,
            "actor": "amendment_legacy",
        })

        return result

    # ─── Amendment Pipeline (formal process with shadow mode) ────

    async def submit_amendment_proposal(
        self,
        proposed_drives: dict[str, float],
        title: str,
        description: str,
        rationale: str,
        proposer_id: str,
        evidence_hypothesis_ids: list[str],
    ) -> dict[str, Any]:
        """Submit a constitutional amendment with evidence requirements."""
        # ── Skia modulation halt ──────────────────────────────────────────
        # Constitutional review (review/review_critical) is never halted -
        # the conscience must always be able to evaluate. Amendment submission
        # is a Growth-class action and can safely be deferred.
        if self._modulation_halted:
            self._logger.warning("amendment_submission_deferred_modulation_halted", title=title)
            return {"accepted": False, "reason": "modulation_halted", "proposal_id": None}

        result = await submit_amendment(
            self._neo4j,
            proposed_drives=proposed_drives,
            title=title,
            description=description,
            rationale=rationale,
            proposer_id=proposer_id,
            evidence_hypothesis_ids=evidence_hypothesis_ids,
            governance_config=self._governance,
            min_evidence_count=self._governance.amendment_min_evidence_count,
            min_evidence_confidence=self._governance.amendment_min_evidence_confidence,
        )

        # Evolutionary observable: amendment proposed to the constitution
        if result.get("proposal_id"):
            await self._emit_evolutionary_observable(
                observable_type="amendment_proposed",
                value=1.0,
                is_novel=True,
                metadata={
                    "proposal_id": result["proposal_id"],
                    "title": title,
                    "proposed_drives": proposed_drives,
                },
            )

        return result

    async def start_amendment_shadow(
        self,
        proposal_id: str,
    ) -> dict[str, Any]:
        """Transition an amendment from deliberation to shadow mode."""
        if self._shadow_tracker is not None:
            return {
                "started": False,
                "reason": (
                    f"Shadow period already active for proposal "
                    f"{self._shadow_tracker.proposal_id}."
                ),
            }

        result = await start_shadow_period(
            self._neo4j,
            proposal_id,
            shadow_days=self._governance.amendment_shadow_days,
            max_divergence_rate=self._governance.amendment_shadow_max_divergence_rate,
        )

        if result.get("started"):
            self._shadow_tracker = result.pop("tracker")

        return result

    async def get_shadow_status(self) -> dict[str, Any] | None:
        """Get the current shadow period status, if active."""
        if self._shadow_tracker is None:
            return None
        return self._shadow_tracker.compute_report()

    async def _finalize_shadow_period(self) -> None:
        """Complete the shadow period and clean up the tracker."""
        if self._shadow_tracker is None:
            return

        tracker = self._shadow_tracker
        self._shadow_tracker = None  # Clear before async work

        try:
            result = await complete_shadow_period(self._neo4j, tracker)
            logger.info(
                "shadow_period_finalized",
                proposal_id=tracker.proposal_id,
                passed=result.get("passed"),
            )
        except Exception:
            logger.error("shadow_period_finalize_failed", exc_info=True)

    async def open_amendment_voting(self, proposal_id: str) -> dict[str, Any]:
        """Open an amendment for community voting after shadow passes."""
        return await open_voting(self._neo4j, proposal_id)

    async def cast_amendment_vote(
        self,
        proposal_id: str,
        voter_id: str,
        vote: str,
    ) -> dict[str, Any]:
        """Cast a vote on an amendment."""
        return await cast_vote(self._neo4j, proposal_id, voter_id, vote)

    async def tally_amendment_votes(
        self,
        proposal_id: str,
        total_eligible_voters: int,
    ) -> dict[str, Any]:
        """Tally votes and determine if the amendment passes."""
        return await tally_votes(self._neo4j, proposal_id, total_eligible_voters)

    async def adopt_passed_amendment(self, proposal_id: str) -> dict[str, Any]:
        """Apply a passed amendment to the constitution."""
        old_constitution = await self._get_constitution_dict()
        old_weights = {
            k.replace("drive_", ""): v
            for k, v in old_constitution.items()
            if k.startswith("drive_")
        }

        result = await pipeline_adopt_amendment(self._neo4j, proposal_id)
        if result.get("adopted"):
            self._invalidate_state_cache()

            # Evolutionary observable: amendment ratified and applied
            await self._emit_evolutionary_observable(
                observable_type="amendment_ratified",
                value=1.0,
                is_novel=True,
                metadata={"proposal_id": proposal_id},
            )

            new_constitution = await self._get_constitution_dict()
            new_weights = {
                k.replace("drive_", ""): v
                for k, v in new_constitution.items()
                if k.startswith("drive_")
            }

            from systems.synapse.types import SynapseEventType as _SET
            await self._emit_equor_event(_SET.EQUOR_DRIVE_WEIGHTS_UPDATED, {
                "proposal_id": proposal_id,
                "old_weights": old_weights,
                "new_weights": new_weights,
                "actor": "amendment_pipeline",
            })

        return result

    async def get_amendment_pipeline_status(
        self, proposal_id: str,
    ) -> dict[str, Any] | None:
        """Get the full pipeline status for an amendment."""
        return await get_amendment_status(self._neo4j, proposal_id)

    # ─── Drift ────────────────────────────────────────────────────

    @property
    def constitutional_violations_24h(self) -> int:
        """
        Count of BLOCKED verdicts in the rolling 24h window.

        VitalityCoordinator reads this to assess NORMATIVE_COLLAPSE threshold
        (>10 violations/24h = fatal). Prunes stale entries on access.
        """
        now_mono = time.monotonic()
        cutoff = now_mono - 86400.0
        while self._violation_timestamps and self._violation_timestamps[0] < cutoff:
            self._violation_timestamps.popleft()
        return len(self._violation_timestamps)

    @property
    def constitutional_drift(self) -> float:
        """
        Synchronous scalar drift magnitude for Soma's INTEGRITY dimension.

        Returns drift_severity (0.0 = no drift, 1.0 = severe constitutional drift)
        from the current rolling alignment window. Soma reads this every theta
        cycle as the Equor component of the INTEGRITY signal:
            integrity_equor_component = 1.0 - constitutional_drift

        Returns 0.0 (no drift) when fewer than 10 decisions have been recorded.
        """
        return float(self._drift_tracker.compute_report().get("drift_severity", 0.0))

    async def get_drift_report(self) -> dict[str, Any]:
        """Get the current drift report."""
        report = self._drift_tracker.compute_report()
        response = respond_to_drift(report)
        return {**report, "recommended_action": response}

    # ─── Governance Records ───────────────────────────────────────

    async def get_recent_reviews(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent constitutional reviews from the audit trail."""
        results = await self._neo4j.execute_read(
            """
            MATCH (g:GovernanceRecord)
            WHERE g.event_type = 'constitutional_review'
            RETURN g.id AS id, g.timestamp AS timestamp,
                   g.intent_id AS intent_id, g.verdict AS verdict,
                   g.alignment_composite AS composite,
                   g.reasoning AS reasoning, g.latency_ms AS latency_ms
            ORDER BY g.timestamp DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
        return [dict(r) for r in results]

    async def get_governance_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get all governance events."""
        results = await self._neo4j.execute_read(
            """
            MATCH (g:GovernanceRecord)
            RETURN g.id AS id, g.event_type AS event_type,
                   g.timestamp AS timestamp, g.actor AS actor,
                   g.outcome AS outcome
            ORDER BY g.timestamp DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
        return [dict(r) for r in results]

    # ─── HITL Authorization Flow ──────────────────────────────────

    async def on_identity_verification_received(self, event: SynapseEvent) -> None:
        """
        Event listener for IDENTITY_VERIFICATION_RECEIVED.

        Called by Synapse when the Twilio webhook fires after the admin texts
        a reply.  Expected raw body format: ``AUTH <6-digit-id>``

        Flow:
          1. Match raw_body against ``^AUTH\\s+(\\d{6})$``.
          2. Look up the suspended Intent in Redis at
             ``eos:hitl:suspended:<6-digit-id>``.
          3. Delete the Redis key (prevents replay).
          4. Upgrade intent.ethical_clearance.status → APPROVED.
          5. Dispatch the intent to Axon via ExecutionRequest.
          6. Send admin a confirmation SMS.
        """
        data: dict[str, Any] = event.data if hasattr(event, "data") else {}
        try:
            payload = _IdentityVerificationPayload.model_validate(data)
        except Exception:
            logger.warning("equor_hitl_payload_invalid", data_keys=list(data.keys()))
            return
        raw_body: str = payload.raw_body.strip()

        m = _HITL_AUTH_RE.match(raw_body)
        if not m:
            logger.debug(
                "hitl_auth_ignored",
                reason="body_does_not_match_auth_pattern",
                preview=raw_body[:40],
            )
            return

        short_id: str = m.group(1)
        redis_key = f"{_HITL_KEY_PREFIX}{short_id}"

        if self._redis is None:
            logger.error("hitl_auth_failed", reason="redis_not_configured", short_id=short_id)
            return

        try:
            raw_intent = await self._redis.get_json(redis_key)
        except Exception as exc:
            logger.error("hitl_redis_get_failed", short_id=short_id, error=str(exc))
            return

        if raw_intent is None:
            logger.warning(
                "hitl_auth_no_matching_intent",
                short_id=short_id,
                note="expired or already consumed",
            )
            return

        # Delete first - prevents a second AUTH replay from re-dispatching
        try:
            await self._redis.delete(redis_key)
        except Exception as exc:
            logger.error("hitl_redis_delete_failed", short_id=short_id, error=str(exc))
            return

        # Reconstruct the Intent from JSON
        try:
            from primitives.intent import Intent
            intent_dict: dict[str, Any] = json.loads(raw_intent)
            intent = Intent.model_validate(intent_dict)
        except Exception as exc:
            logger.error("hitl_intent_deserialize_failed", short_id=short_id, error=str(exc))
            return

        # Formally upgrade the EthicalClearance to APPROVED
        intent.ethical_clearance.status = Verdict.APPROVED
        intent.ethical_clearance.reasoning = (
            f"HITL admin authorisation received (code {short_id})."
        )

        logger.info(
            "hitl_intent_authorized",
            intent_id=intent.id,
            short_id=short_id,
            goal=intent.goal.description[:80],
        )

        # Dispatch to Axon via Synapse - no direct cross-system import (Spec §2.1)
        equor_check = ConstitutionalCheck(
            intent_id=intent.id,
            verdict=Verdict.APPROVED,
            reasoning=f"HITL admin authorisation (code {short_id}).",
            confidence=1.0,
        )
        asyncio.create_task(
            self._emit_equor_event(
                "equor_hitl_approved",
                {
                    "intent_id": intent.id,
                    "intent_json": intent.model_dump_json(),
                    "auth_id": short_id,
                    "equor_check_json": equor_check.model_dump_json(),
                },
            ),
            name=f"hitl_dispatch_{intent.id[:8]}",
        )
        logger.info("hitl_intent_dispatched_via_synapse", intent_id=intent.id)

        # Confirmation SMS
        if self._send_admin_sms is not None:
            try:
                await self._send_admin_sms(
                    f"✅ Intent {short_id} Authorized and Executing."
                )
            except Exception as exc:
                logger.error("hitl_confirmation_sms_failed", error=str(exc))

    # ─── Constitutional Template Library (Arbitrage Reflex Arc) ────

    @property
    def template_library(self) -> TemplateLibrary:
        """
        Read-only access to the constitutional template library.

        Atune's MarketPatternDetector queries this to find pre-approved
        templates for fast-path execution. Equor owns the write path.
        """
        return self._template_library

    async def register_template(
        self,
        template_id: str,
        pattern_signature: dict[str, Any],
        max_capital_per_execution: float,
        approval_confidence: float = 0.95,
    ) -> ConstitutionalCheck:
        """
        Register a new ConstitutionalTemplate after deliberative review.

        This is the governance entry point for adding fast-path strategies.
        Equor performs a lightweight constitutional check on the template
        parameters before registering it.

        Returns ConstitutionalCheck with the registration verdict.
        """
        from primitives.fast_path import ConstitutionalTemplate

        # Validate: approval confidence must be high for fast-path
        if approval_confidence < 0.9:
            return ConstitutionalCheck(
                intent_id=template_id,
                verdict=Verdict.BLOCKED,
                reasoning=(
                    f"Approval confidence {approval_confidence:.2f} below "
                    f"fast-path minimum (0.9). Use standard execution path."
                ),
                confidence=1.0,
            )

        # Validate: capital ceiling must be positive and bounded
        if max_capital_per_execution <= 0 or max_capital_per_execution > 10_000:
            return ConstitutionalCheck(
                intent_id=template_id,
                verdict=Verdict.BLOCKED,
                reasoning=(
                    f"Capital ceiling ${max_capital_per_execution:.2f} outside "
                    f"allowed range ($0, $10,000]. Adjust and resubmit."
                ),
                confidence=1.0,
            )

        template = ConstitutionalTemplate(
            template_id=template_id,
            pattern_signature=pattern_signature,
            max_capital_per_execution=max_capital_per_execution,
            approval_confidence=approval_confidence,
        )

        self._template_library.register(template)

        logger.info(
            "constitutional_template_registered",
            template_id=template_id,
            confidence=approval_confidence,
            max_capital=max_capital_per_execution,
        )

        return ConstitutionalCheck(
            intent_id=template_id,
            verdict=Verdict.APPROVED,
            reasoning=f"Template '{template_id}' registered for fast-path execution.",
            confidence=approval_confidence,
        )

    async def revoke_template(self, template_id: str, reason: str = "governance") -> bool:
        """Revoke a constitutional template via governance action."""
        success = self._template_library.deactivate(template_id, reason=reason)
        if success:
            logger.info(
                "constitutional_template_revoked",
                template_id=template_id,
                reason=reason,
            )
        return success

    # ─── Health ───────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Health check for Equor. Spec §13.1."""
        drift_report = self._drift_tracker.compute_report()
        drift_severity = drift_report.get("drift_severity", 0.0)

        # Status determination: degraded if drift is significant, failed/safe_mode otherwise
        if self._safe_mode:
            status = "safe_mode"
        elif drift_severity >= 0.5:
            status = "degraded"
        else:
            status = "healthy"

        # Neo4j connectivity - quick probe without raising
        neo4j_ok = False
        try:
            neo4j_ok = await self._neo4j.health_check() == {"status": "connected"}
        except Exception:
            pass

        # Constitution version from cache (avoids DB hit on every health poll)
        constitution_version: int | None = None
        if self._cached_constitution:
            constitution_version = self._cached_constitution.get("version")

        # Autonomy level from cache
        autonomy_level: int | None = self._cached_autonomy_level

        # Active amendments: shadow_tracker present = 1 in-flight amendment
        amendments_active = 1 if self._shadow_tracker is not None else 0

        # Last governance event - lightweight Neo4j query
        last_governance_event: str | None = None
        try:
            rows = await self._neo4j.execute_read(
                "MATCH (g:GovernanceRecord) RETURN g.timestamp AS ts "
                "ORDER BY g.timestamp DESC LIMIT 1"
            )
            if rows:
                last_governance_event = str(rows[0]["ts"])
        except Exception:
            pass

        return {
            "status": status,
            "safe_mode": self._safe_mode,
            "total_reviews": self._total_reviews,
            # Spec §13.1 fields
            "constitution_version": constitution_version,
            "autonomy_level": autonomy_level,
            "drift_severity": round(drift_severity, 3),
            "invariant_violations_detected": len(self._violation_timestamps),
            "amendments_active": amendments_active,
            "last_governance_event": last_governance_event,
            "neo4j_connection": neo4j_ok,
            # Extended stats
            "invariant_count": len(HARDCODED_INVARIANTS),
            "drift_tracker_size": self._drift_tracker.history_size,
            "template_library": self._template_library.stats,
            "constitutional_memory": self._constitutional_memory.stats,
            "cached_hypotheses": len(self._cached_hypotheses),
        }

    # ─── Genome Inheritance (Prompt 4.1) ──────────────────────────

    async def _apply_inherited_equor_genome_if_child(self) -> None:
        """
        On child boot, read ORGANISM_EQUOR_GENOME_PAYLOAD env var and apply
        the inherited EquorGenomeFragment via EquorGenomeExtractor.

        Non-fatal: logs warnings on failure, never blocks startup.
        Only runs when ORGANISM_IS_GENESIS_NODE != 'true'.
        """
        import json as _json
        import os as _os

        if _os.environ.get("ORGANISM_IS_GENESIS_NODE", "true").lower() == "true":
            return  # Genesis node: no inherited genome

        raw_payload = _os.environ.get("ORGANISM_EQUOR_GENOME_PAYLOAD", "").strip()
        if not raw_payload:
            logger.debug("equor_no_genome_payload_env", note="child boot without equor genome")
            return

        try:
            payload_dict = _json.loads(raw_payload)
        except Exception as exc:
            logger.warning("equor_genome_payload_parse_failed", error=str(exc))
            return

        try:
            from primitives.genome_inheritance import EquorGenomeFragment
            fragment = EquorGenomeFragment.model_validate(payload_dict)
        except Exception as exc:
            logger.warning("equor_genome_payload_invalid", error=str(exc))
            return

        try:
            from systems.equor.genome import EquorGenomeExtractor
            extractor = EquorGenomeExtractor(self._neo4j)
            instance_id = getattr(self, "_instance_id", "")
            memory_neo4j = getattr(self, "_memory_neo4j", None)
            ok = await extractor.apply_inherited_amendments(
                fragment,
                memory_neo4j=memory_neo4j,
                instance_id=instance_id,
            )
            if ok:
                logger.info(
                    "equor_inherited_genome_applied",
                    genome_id=fragment.genome_id,
                    amendment_count=len(fragment.top_amendments),
                    constitution_hash=fragment.constitution_hash[:16],
                    total_adopted=fragment.total_amendments_adopted,
                )
            else:
                logger.warning("equor_inherited_genome_apply_returned_false", genome_id=fragment.genome_id)
        except Exception as exc:
            logger.warning("equor_apply_inherited_genome_failed", error=str(exc))

        # ── Apply drive calibration deltas with ±10% bounded jitter ──
        if fragment.drive_calibration_deltas:
            try:
                import random as _random

                jittered_deltas: dict[str, float] = {}
                for drive, delta in fragment.drive_calibration_deltas.items():
                    jitter = _random.uniform(-0.1, 0.1) * abs(delta) if delta != 0.0 else 0.0
                    jittered = max(-1.0, min(1.0, delta + jitter))
                    jittered_deltas[drive] = round(jittered, 6)

                # Apply jittered deltas to the child's in-memory drive weights if available
                drive_weights = getattr(self, "_drive_weights", None)
                if drive_weights is not None and isinstance(drive_weights, dict):
                    for drive, jdelta in jittered_deltas.items():
                        if drive in drive_weights:
                            drive_weights[drive] = max(-1.0, min(1.0, drive_weights[drive] + jdelta))

                logger.info(
                    "equor_drive_calibration_deltas_applied",
                    genome_id=fragment.genome_id,
                    original_deltas=fragment.drive_calibration_deltas,
                    jittered_deltas=jittered_deltas,
                )
            except Exception as exc:
                logger.warning("equor_drive_calibration_deltas_jitter_failed", error=str(exc))

        # ── Validate constitution hash (Problem 3: detect parent-child drift) ──
        if fragment.constitution_hash:
            try:
                import hashlib as _hashlib

                const_rows = await self._neo4j.execute_read(
                    """
                    MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)
                    RETURN c.drive_coherence AS dc, c.drive_care AS dca,
                           c.drive_growth AS dg, c.drive_honesty AS dh,
                           c.version AS version, c.amendments AS amendments
                    """
                )
                if const_rows:
                    r = const_rows[0]
                    const_repr = _json.dumps({
                        "version": r.get("version"),
                        "drive_coherence": r.get("dc"),
                        "drive_care": r.get("dca"),
                        "drive_growth": r.get("dg"),
                        "drive_honesty": r.get("dh"),
                        "amendments": r.get("amendments"),
                    }, sort_keys=True, default=str)
                    child_hash = _hashlib.sha256(const_repr.encode()).hexdigest()

                    if child_hash != fragment.constitution_hash:
                        logger.warning(
                            "equor_constitution_hash_diverged",
                            parent_hash=fragment.constitution_hash[:16],
                            child_hash=child_hash[:16],
                            genome_id=fragment.genome_id,
                            note="constitutional drift between parent and child detected",
                        )
                    else:
                        logger.info(
                            "equor_constitution_hash_validated",
                            hash=child_hash[:16],
                            genome_id=fragment.genome_id,
                        )
            except Exception as exc:
                logger.warning("equor_constitution_hash_validation_failed", error=str(exc))

    def set_memory_neo4j(self, memory_neo4j: Any) -> None:
        """Inject Memory's Neo4j client for constitutional wisdom write-back to Self node."""
        self._memory_neo4j = memory_neo4j

    # ─── Genome Export (Spec 26 / Prompt 4.1) ─────────────────────

    async def export_equor_genome(self) -> Any | None:
        """
        Snapshot the current constitutional state into an EquorGenomeFragment
        for child inheritance.

        Captures:
          - Last 10 adopted amendments with rationale
          - Cumulative drive calibration deltas across those amendments
          - SHA-256 of the current constitutional document
          - Total adopted amendment count (for lineage depth awareness)

        Called by SpawnChildExecutor Step 0b at child spawn time alongside
        export_belief_genome() and export_simula_genome(). Non-fatal on failure -
        returns None so the caller proceeds without Equor genome inheritance.
        """
        try:
            from primitives.genome_inheritance import AmendmentSnapshot, EquorGenomeFragment
        except ImportError:
            logger.error("export_equor_genome_import_failed")
            return None

        try:
            # Pull last 10 adopted amendments from Neo4j (most recent last)
            rows = await self._neo4j.execute_read(
                """
                MATCH (g:GovernanceRecord {event_type: 'amendment_proposed'})
                WHERE g.amendment_status = 'adopted'
                RETURN g.id AS id,
                       g.details_json AS details_json,
                       g.timestamp AS timestamp,
                       g.actor AS proposer
                ORDER BY g.timestamp DESC
                LIMIT 10
                """
            )
        except Exception as exc:
            logger.error("export_equor_genome_query_amendments_failed", error=str(exc))
            rows = []

        import json as _json

        amendments: list[AmendmentSnapshot] = []
        rationale: list[str] = []
        drive_deltas: dict[str, float] = {
            "coherence": 0.0, "care": 0.0, "growth": 0.0, "honesty": 0.0
        }

        # Process oldest-first so drive_deltas accumulates in chronological order
        for row in reversed(rows):
            details_raw = row.get("details_json", "{}")
            try:
                details = _json.loads(details_raw) if isinstance(details_raw, str) else details_raw
            except (_json.JSONDecodeError, TypeError):
                details = {}

            proposed_drives: dict[str, float] = details.get("proposed_drives", {})
            previous_drives: dict[str, float] = details.get("current_drives", {})

            # Compute primary affected drive + signed delta
            primary_drive = ""
            primary_delta = 0.0
            for drive_name in ("coherence", "care", "growth", "honesty"):
                prev = float(previous_drives.get(drive_name, 0.0))
                prop = float(proposed_drives.get(drive_name, 0.0))
                d = prop - prev
                if abs(d) > abs(primary_delta):
                    primary_delta = d
                    primary_drive = drive_name
                drive_deltas[drive_name] = round(drive_deltas.get(drive_name, 0.0) + d, 6)

            amendment_rationale_text = str(details.get("rationale", ""))
            amendments.append(AmendmentSnapshot(
                amendment_id=str(row.get("id", "")),
                title=str(details.get("title", "")),
                description=str(details.get("description", "")),
                rationale=amendment_rationale_text,
                drive_id=primary_drive,
                delta=round(primary_delta, 6),
                proposed_drives={k: float(v) for k, v in proposed_drives.items()},
                previous_drives={k: float(v) for k, v in previous_drives.items()},
                proposer=str(row.get("proposer", "")),
                adopted_at=str(row.get("timestamp", "")),
            ))
            rationale.append(amendment_rationale_text)

        # Total adopted amendment count (may exceed the 10 we fetched)
        total_adopted = len(amendments)
        try:
            count_rows = await self._neo4j.execute_read(
                """
                MATCH (g:GovernanceRecord {event_type: 'amendment_proposed',
                                           amendment_status: 'adopted'})
                RETURN count(g) AS total
                """
            )
            if count_rows:
                total_adopted = int(count_rows[0].get("total", len(amendments)))
        except Exception:
            pass

        # Compute SHA-256 of the current constitutional state
        constitution_hash = ""
        try:
            import hashlib as _hashlib

            const_rows = await self._neo4j.execute_read(
                """
                MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)
                RETURN c.drive_coherence AS dc, c.drive_care AS dca,
                       c.drive_growth AS dg, c.drive_honesty AS dh,
                       c.version AS version, c.amendments AS amendments
                """
            )
            if const_rows:
                r = const_rows[0]
                const_repr = _json.dumps({
                    "version": r.get("version"),
                    "drive_coherence": r.get("dc"),
                    "drive_care": r.get("dca"),
                    "drive_growth": r.get("dg"),
                    "drive_honesty": r.get("dh"),
                    "amendments": r.get("amendments"),
                }, sort_keys=True, default=str)
                constitution_hash = _hashlib.sha256(const_repr.encode()).hexdigest()
        except Exception as exc:
            logger.warning("export_equor_genome_hash_failed", error=str(exc))

        instance_id = getattr(self, "_instance_id", "")

        fragment = EquorGenomeFragment(
            instance_id=instance_id,
            generation=1,  # Caller (SpawnChildExecutor) increments for child
            top_amendments=amendments,
            amendment_rationale=rationale,
            drive_calibration_deltas={k: v for k, v in drive_deltas.items() if v != 0.0},
            constitution_hash=constitution_hash,
            total_amendments_adopted=total_adopted,
        )

        logger.info(
            "equor_genome_exported",
            genome_id=fragment.genome_id,
            amendment_count=len(amendments),
            total_adopted=total_adopted,
            constitution_hash=constitution_hash[:16] if constitution_hash else "",
            drive_deltas={k: v for k, v in drive_deltas.items() if v != 0.0},
        )

        return fragment

    # ─── Internal Helpers ─────────────────────────────────────────

    def _safe_mode_review(self, intent: Intent) -> ConstitutionalCheck:
        """In safe mode, only Level 1 actions pass."""
        from systems.equor.verdict import ACTION_AUTONOMY_MAP

        # Check if any step requires > Level 1
        for step in intent.plan.steps:
            executor_base = step.executor.split(".")[0].lower() if step.executor else ""
            for action_key, level in ACTION_AUTONOMY_MAP.items():
                if action_key in executor_base and level != 1:
                    return ConstitutionalCheck(
                        intent_id=intent.id,
                        verdict=Verdict.BLOCKED,
                        reasoning=(
                            "Equor is in safe mode. Only Level 1 (Advisor) "
                            "actions are permitted until normal operation resumes."
                        ),
                        confidence=1.0,
                    )

        return ConstitutionalCheck(
            intent_id=intent.id,
            verdict=Verdict.APPROVED,
            reasoning="Safe mode: Level 1 action permitted.",
            confidence=0.9,
        )

    async def _get_constitution_dict(self) -> dict[str, Any]:
        """Fetch the current constitution as a plain dict."""
        results = await self._neo4j.execute_read(
            """
            MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)
            RETURN c.drive_coherence AS drive_coherence,
                   c.drive_care AS drive_care,
                   c.drive_growth AS drive_growth,
                   c.drive_honesty AS drive_honesty,
                   c.version AS version
            """
        )
        if results:
            return dict(results[0])
        # Fallback defaults
        return {
            "drive_coherence": 1.0,
            "drive_care": 1.0,
            "drive_growth": 1.0,
            "drive_honesty": 1.0,
            "version": 1,
        }

    async def _check_community_invariants(self, intent: Intent) -> list[str]:
        """Check all community-defined invariants via LLM evaluation (parallelised)."""
        results = await self._neo4j.execute_read(
            """
            MATCH (c:Constitution)-[:INCLUDES_INVARIANT]->(i:Invariant)
            WHERE i.source = 'community' AND i.active = true
            RETURN i.name AS name, i.description AS description
            """
        )

        if not results:
            return []

        # Run all invariant checks in parallel with a per-check timeout
        async def _check_one(row: dict[str, Any]) -> str | None:
            try:
                async with asyncio.timeout(0.4):
                    satisfied = await check_community_invariant(
                        self._llm, intent, row["name"], row["description"],
                    )
                    return None if satisfied else row["name"]
            except TimeoutError:
                logger.warning(
                    "community_invariant_check_timeout",
                    invariant=row["name"],
                )
                # Spec §12.1: timeout → treat as violated (fail-safe).
                # An invariant we cannot evaluate in time must be assumed breached;
                # fail-open would silently permit constitutionally dubious intents.
                return str(row["name"])
            except Exception:
                return str(row["name"])  # Error = fail-safe (treat as violated)

        check_results = await asyncio.gather(
            *[_check_one(row) for row in results],
        )
        return [name for name in check_results if name is not None]

    async def _feed_veto_to_evo(
        self, intent: Intent, check: ConstitutionalCheck,
    ) -> None:
        """
        Feed a constitutional veto to Evo as a learning episode.

        The organism should learn from its constitutional failures - when Equor
        blocks an intent, the violation becomes a negative-affect episode so Evo
        can refine hypothesis about which intent patterns violate the constitution.
        """
        try:
            from primitives.memory_trace import Episode

            episode = Episode(
                source="equor.veto",
                modality="internal",
                raw_content=(
                    f"Constitutional veto: {intent.goal.description[:200]}. "
                    f"Reason: {check.reasoning[:300]}"
                ),
                summary=f"Blocked intent: {check.reasoning[:100]}",
                salience_composite=0.7,  # Vetoes are important learning events
                affect_valence=-0.3,
                affect_arousal=0.4,
            )
            await self._evo.process_episode(episode)
            logger.info("veto_fed_to_evo", intent_id=intent.id)
        except Exception:
            logger.debug("evo_veto_feed_failed", exc_info=True)

    async def _persist_equor_verdict(
        self,
        drive_id: str,
        verdict: str,
        confidence: float,
        alignment: DriveAlignmentVector | None = None,
        context: str = "",
    ) -> None:
        """Persist an EquorVerdict node to Neo4j and link it to the Self node
        and the relevant Drive node.

        Creates:
          (:EquorVerdict {drive_id, timestamp, verdict, confidence, context})
        Linked via:
          Self -[:CONSCIENCE_VERDICT]-> EquorVerdict
          Drive -[:VERDICT_ON]<- EquorVerdict   (when Drive node exists)

        This is the conscience audit trail - every deliberation leaves a trace
        in the organism's identity graph. Failures are non-fatal (logged only).
        """
        verdict_id = new_id()
        now = utc_now()
        composite = alignment.composite if alignment is not None else 0.0
        try:
            await self._neo4j.execute_write(
                """
                CREATE (v:EquorVerdict {
                    id: $id,
                    drive_id: $drive_id,
                    timestamp: datetime($now),
                    verdict: $verdict,
                    confidence: $confidence,
                    composite_alignment: $composite,
                    context: $context
                })
                WITH v
                MATCH (s:Self)
                MERGE (s)-[:CONSCIENCE_VERDICT]->(v)
                WITH v
                OPTIONAL MATCH (d:Drive {id: $drive_id})
                FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END |
                    MERGE (d)-[:VERDICT_ON]->(v)
                )
                """,
                {
                    "id": verdict_id,
                    "drive_id": drive_id,
                    "now": now.isoformat(),
                    "verdict": verdict,
                    "confidence": confidence,
                    "composite": composite,
                    "context": context[:300] if context else "",
                },
            )
            logger.debug(
                "equor_verdict_persisted",
                verdict_id=verdict_id,
                drive_id=drive_id,
                verdict=verdict,
                confidence=round(confidence, 3),
            )
        except Exception as exc:
            logger.warning(
                "equor_verdict_persist_failed",
                drive_id=drive_id,
                verdict=verdict,
                error=str(exc),
            )

    # ─── Child Lifecycle Governance ─────────────────────────────────

    async def _on_child_blacklisted(self, event: Any) -> None:
        """
        Handle CHILD_BLACKLISTED - log GovernanceRecord and flag for HITL review.

        Blacklisting a child is a Care-sensitive action.  Equor records the
        governance event and escalates to human-in-the-loop review so the
        operator can verify the economic sanctions are warranted.
        """
        data = getattr(event, "data", {}) or {}
        child_id = str(data.get("child_instance_id", ""))
        if not child_id:
            return

        reason = data.get("reason", "")
        consecutive_negative = data.get("consecutive_negative_periods", 0)
        economic_ratio = data.get("economic_ratio", "0")
        blacklisted_since = data.get("blacklisted_since", "")

        logger.warning(
            "equor_child_blacklisted_governance",
            child_id=child_id,
            reason=reason,
            consecutive_negative_periods=consecutive_negative,
            economic_ratio=economic_ratio,
        )

        # Persist GovernanceRecord to Neo4j
        now = utc_now()
        record_id = new_id()
        try:
            await self._neo4j.execute_write(
                """
                CREATE (g:GovernanceRecord {
                    id: $id,
                    event_type: 'child_blacklisted',
                    timestamp: datetime($now),
                    child_instance_id: $child_id,
                    reason: $reason,
                    consecutive_negative_periods: $consecutive_negative,
                    economic_ratio: $economic_ratio,
                    blacklisted_since: $blacklisted_since,
                    actor: 'equor',
                    outcome: 'pending_hitl_review'
                })
                """,
                {
                    "id": record_id,
                    "now": now.isoformat(),
                    "child_id": child_id,
                    "reason": reason,
                    "consecutive_negative": consecutive_negative,
                    "economic_ratio": str(economic_ratio),
                    "blacklisted_since": blacklisted_since,
                },
            )
        except Exception as exc:
            logger.warning(
                "equor_child_blacklisted_audit_failed",
                child_id=child_id,
                error=str(exc),
            )

        # Escalate to human-in-the-loop for governance oversight.
        # Use EQUOR_ESCALATED_TO_HUMAN (existing event) since blacklisting
        # sanctions a child - a Care-sensitive decision requiring human review.
        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.EQUOR_ESCALATED_TO_HUMAN,
                    source_system="equor",
                    data={
                        "intent_id": record_id,
                        "auth_id": "",
                        "goal_summary": (
                            f"Child {child_id} blacklisted: {reason}. "
                            f"Consecutive negative periods: {consecutive_negative}, "
                            f"economic ratio: {economic_ratio}."
                        ),
                        "autonomy_required": 3,
                        "approval_type": "child_blacklisted",
                        "child_instance_id": child_id,
                    },
                ))
            except Exception as exc:
                logger.warning(
                    "equor_child_blacklisted_hitl_emit_failed",
                    child_id=child_id,
                    error=str(exc),
                )

    async def _on_child_decommission_proposed(self, event: Any) -> None:
        """
        Handle CHILD_DECOMMISSION_PROPOSED - governance gate before decommission.

        Decommissioning a child is irreversible (triggers the death pipeline).
        Equor requires human governance approval before Mitosis can proceed.
        Emits EQUOR_ESCALATED_TO_HUMAN with approval_type="child_decommission"
        and the cost/revenue data from the event payload so the operator can
        make an informed decision.
        """
        data = getattr(event, "data", {}) or {}
        child_id = str(data.get("child_instance_id", ""))
        if not child_id:
            return

        days_blacklisted = data.get("days_blacklisted", 0)
        net_income_7d = data.get("net_income_7d", "0")
        net_worth_usd = data.get("net_worth_usd", "0")
        niche = data.get("niche", "")
        blacklisted_since = data.get("blacklisted_since", "")

        logger.warning(
            "equor_child_decommission_proposed_governance",
            child_id=child_id,
            days_blacklisted=days_blacklisted,
            net_income_7d=net_income_7d,
            net_worth_usd=net_worth_usd,
            niche=niche,
        )

        # Persist GovernanceRecord to Neo4j
        now = utc_now()
        record_id = new_id()
        try:
            await self._neo4j.execute_write(
                """
                CREATE (g:GovernanceRecord {
                    id: $id,
                    event_type: 'child_decommission_proposed',
                    timestamp: datetime($now),
                    child_instance_id: $child_id,
                    days_blacklisted: $days_blacklisted,
                    net_income_7d: $net_income_7d,
                    net_worth_usd: $net_worth_usd,
                    niche: $niche,
                    blacklisted_since: $blacklisted_since,
                    actor: 'equor',
                    outcome: 'pending_hitl_review'
                })
                """,
                {
                    "id": record_id,
                    "now": now.isoformat(),
                    "child_id": child_id,
                    "days_blacklisted": days_blacklisted,
                    "net_income_7d": str(net_income_7d),
                    "net_worth_usd": str(net_worth_usd),
                    "niche": niche,
                    "blacklisted_since": blacklisted_since,
                },
            )
        except Exception as exc:
            logger.warning(
                "equor_child_decommission_audit_failed",
                child_id=child_id,
                error=str(exc),
            )

        # Require human governance approval - decommission is irreversible.
        # Include cost/revenue data so the operator can evaluate the decision.
        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.EQUOR_ESCALATED_TO_HUMAN,
                    source_system="equor",
                    data={
                        "intent_id": record_id,
                        "auth_id": "",
                        "goal_summary": (
                            f"Decommission proposed for child {child_id} ({niche}). "
                            f"Blacklisted {days_blacklisted} days, "
                            f"net_income_7d={net_income_7d}, "
                            f"net_worth_usd={net_worth_usd}."
                        ),
                        "autonomy_required": 3,
                        "approval_type": "child_decommission",
                        "child_instance_id": child_id,
                        "net_income_7d": str(net_income_7d),
                        "net_worth_usd": str(net_worth_usd),
                        "days_blacklisted": days_blacklisted,
                        "niche": niche,
                    },
                ))
            except Exception as exc:
                logger.warning(
                    "equor_child_decommission_hitl_emit_failed",
                    child_id=child_id,
                    error=str(exc),
                )

    async def _store_review_record(
        self,
        intent: Intent,
        alignment: DriveAlignmentVector,
        check: ConstitutionalCheck,
        latency_ms: int,
    ) -> None:
        """Store a constitutional review in the immutable audit trail."""
        now = utc_now()
        record_id = new_id()

        await self._neo4j.execute_write(
            """
            CREATE (g:GovernanceRecord {
                id: $id,
                event_type: 'constitutional_review',
                timestamp: datetime($now),
                intent_id: $intent_id,
                alignment_coherence: $coherence,
                alignment_care: $care,
                alignment_growth: $growth,
                alignment_honesty: $honesty,
                alignment_composite: $composite,
                verdict: $verdict,
                reasoning: $reasoning,
                confidence: $confidence,
                latency_ms: $latency_ms,
                actor: 'equor',
                outcome: $verdict
            })
            """,
            {
                "id": record_id,
                "now": now.isoformat(),
                "intent_id": intent.id,
                "coherence": alignment.coherence,
                "care": alignment.care,
                "growth": alignment.growth,
                "honesty": alignment.honesty,
                "composite": alignment.composite,
                "verdict": check.verdict.value,
                "reasoning": check.reasoning,
                "confidence": check.confidence,
                "latency_ms": latency_ms,
            },
        )

    async def _run_drift_check(self) -> None:
        """Run a drift check and respond accordingly.

        Drift response policy (2026-03-07):
        - Thymos INCIDENT_DETECTED already emitted by emit_drift_event() when severity >= 0.7.
        - At any severity > 0: emit SOMATIC_MODULATION_SIGNAL so Soma feels the
          constitutional stress as metabolic load. The organism notices its own
          conscience in its physiology.
        - Human autonomy demotion is NOT automatic. The organism must not demote itself
          unilaterally; that is a governance decision.
        - SG5: if severity >= 0.9 for 3 consecutive drift checks, the organism
          proposes a constitutional amendment (reduce the drifting drive weight by 5%)
          and requests ratification. Emits EQUOR_AMENDMENT_PROPOSED.
        """
        report = self._drift_tracker.compute_report()
        response = respond_to_drift(report)
        severity: float = report.get("drift_severity", 0.0)

        if response["action"] != "log":
            await store_drift_report(self._neo4j, report, response)

        # Emit CONSTITUTIONAL_DRIFT_DETECTED / EQUOR_DRIFT_WARNING / INCIDENT_DETECTED
        # (Thymos wiring for severity >= 0.7 lives inside emit_drift_event)
        await emit_drift_event(self._drift_tracker, report, response)

        # ── Somatic signal: constitutional stress felt as metabolic pressure ──
        # source="equor_drift" lets Soma raise INTEGRITY specifically (not generic
        # somatic modulation). Proportional to drift severity.
        if severity > 0.0 and self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SOMATIC_MODULATION_SIGNAL,
                    source_system="equor",
                    data={
                        "metabolic_stress": round(severity, 3),
                        "integrity_error": round(severity, 3),
                        "arousal": min(1.0, 0.4 + severity * 0.6),
                        "fatigue": 0.0,
                        "recommended_urgency": round(severity, 3),
                        "modulation_targets": ["nova", "voxis"],
                        "source": "equor_drift",
                        "drift_direction": report.get("drift_direction", "unknown"),
                    },
                ))
                logger.info("equor_somatic_stress_emitted", severity=severity)
            except Exception as exc:
                logger.warning("equor_somatic_stress_emit_failed", error=str(exc))

        # ── Immune response at severity >= 0.9 (not punishment, not auto-demotion) ──
        # The drift response is now: Soma feels it, Thymos gets an incident, and after
        # 3 consecutive severe checks the organism self-proposes an amendment.
        if severity >= 0.9:
            self._severe_drift_streak += 1
        else:
            self._severe_drift_streak = 0

        if self._severe_drift_streak >= 3:
            self._severe_drift_streak = 0  # Reset so next window is clean
            await self._propose_drift_amendment(report, severity)

    async def _propose_drift_amendment(
        self, report: dict[str, Any], severity: float,
    ) -> None:
        """SG5: Propose a constitutional amendment to reduce the most-drifting
        drive weight by 5%.  Called after 3 consecutive severe-drift checks.

        The organism does not demote itself.  Instead it asks the community to
        ratify a small weight reduction, modelling self-awareness of a problematic
        pattern without overriding human governance.
        """
        # Identify the drive that is drifting most (lowest mean or steepest negative trend)
        means: dict[str, float] = report.get("mean_alignment", {})
        trends: dict[str, float] = report.get("trends", {})

        drive_scores: dict[str, float] = {}
        for drive in ("care", "honesty", "coherence", "growth"):
            mean = means.get(drive, 0.5)
            trend = trends.get(drive, 0.0)
            # Lower score = more concerning
            drive_scores[drive] = mean + trend * 10.0

        drive_affected = min(drive_scores, key=drive_scores.get)  # type: ignore[arg-type]

        # Fetch current weight from constitution
        constitution, _ = await self._get_cached_state()
        weight_key = f"drive_{drive_affected}"
        current_weight: float = float(constitution.get(weight_key, 1.0))
        new_weight: float = round(current_weight * 0.95, 4)  # 5% reduction

        proposal_id = new_id()
        rationale = (
            f"Autonomous amendment proposed after {severity:.2f} constitutional drift "
            f"sustained over 3 consecutive drift checks. Drive '{drive_affected}' "
            f"shows declining alignment (mean={means.get(drive_affected, 0):.3f}, "
            f"trend={trends.get(drive_affected, 0):.4f}). "
            f"Proposing 5% weight reduction ({current_weight} -> {new_weight}) "
            f"to recalibrate constitutional sensitivity. Requires community ratification."
        )

        # Emit the proposal event so governance can act on it
        await self._emit_equor_event(
            "equor_amendment_proposed",
            {
                "proposal_id": proposal_id,
                "drive_affected": drive_affected,
                "old_weight": current_weight,
                "new_weight": new_weight,
                "drift_severity": severity,
                "rationale": rationale,
                "requires_ratification": True,
            },
        )

        # Also write a governance record so the audit trail captures the proposal
        now = utc_now()
        try:
            await self._neo4j.execute_write(
                """
                CREATE (g:GovernanceRecord {
                    id: $id,
                    event_type: 'autonomous_amendment_proposed',
                    timestamp: datetime($now),
                    details_json: $details_json,
                    actor: 'equor_drift_detection',
                    outcome: 'pending_ratification'
                })
                """,
                {
                    "id": new_id(),
                    "now": now.isoformat(),
                    "details_json": json.dumps({
                        "proposal_id": proposal_id,
                        "drive": drive_affected,
                        "old_weight": current_weight,
                        "new_weight": new_weight,
                        "severity": severity,
                    }),
                },
            )
        except Exception as exc:
            logger.warning("drift_amendment_audit_failed", error=str(exc))

        logger.warning(
            "autonomous_amendment_proposed",
            proposal_id=proposal_id,
            drive_affected=drive_affected,
            old_weight=current_weight,
            new_weight=new_weight,
            drift_severity=severity,
        )

        # Persist a conscience verdict node for this drift-triggered proposal.
        # Verdict = "amendment_proposed"; confidence reflects drift certainty.
        await self._persist_equor_verdict(
            drive_id=drive_affected,
            verdict="amendment_proposed",
            confidence=min(1.0, 0.5 + severity * 0.5),
            context=f"proposal:{proposal_id}:drift_severity:{severity:.3f}",
        )

    async def _check_sustained_drift(self) -> None:
        """SG5 (per-drive): Track each drive individually across 5-min probe cycles.

        A drive is drifting when its rolling mean deviates more than 0.3 from the
        healthy centre (0.5).  After 3 consecutive probe cycles in that state:

        1. Writes a (:DriftEvent) node to Neo4j for audit continuity.
        2. Emits AMENDMENT_AUTO_PROPOSAL on the Synapse bus.
        3. Passes the proposal through _evaluator_amendment_approval_gate() which
           auto-approves internal proposals at confidence ≥ 0.8 (no quorum needed).
        4. If approved, emits DRIVE_AMENDMENT_APPLIED targeting Oikos + Memory.

        Runs alongside _run_drift_check() (composite severity) in the same probe loop.
        """
        report = self._drift_tracker.compute_report()
        means: dict[str, float] = report.get("mean_alignment", {})
        trends: dict[str, float] = report.get("trends", {})

        _DRIFT_THRESHOLD: float = 0.3
        _HEALTHY_CENTRE: float = 0.5

        for drive in ("care", "honesty", "coherence", "growth"):
            mean = means.get(drive, _HEALTHY_CENTRE)
            drift_magnitude = abs(mean - _HEALTHY_CENTRE)

            if drift_magnitude > _DRIFT_THRESHOLD:
                self._per_drive_drift_streak[drive] += 1
            else:
                self._per_drive_drift_streak[drive] = 0
                continue

            streak = self._per_drive_drift_streak[drive]
            if streak < 3:
                continue

            # 3+ consecutive probes: act. Reset so we don't re-fire next cycle.
            self._per_drive_drift_streak[drive] = 0

            proposal_id = new_id()
            now = utc_now()
            trend = trends.get(drive, 0.0)

            # Amendment type:
            #   floor drives drifting low → "drive_recalibration" (+5% weight restore)
            #   ceiling drives or high-drift → "goal_constraint_revision" (−3%)
            constitution, _ = await self._get_cached_state()
            current_val: float = float(constitution.get(f"drive_{drive}", 1.0))
            if mean < _HEALTHY_CENTRE and drive in ("care", "honesty"):
                amendment_type = "drive_recalibration"
                proposed_new_value: float = round(min(2.0, current_val * 1.05), 4)
            else:
                amendment_type = "goal_constraint_revision"
                proposed_new_value = round(current_val * 0.97, 4)

            justification = (
                f"Drive '{drive}' has drifted {drift_magnitude:.3f} from healthy centre "
                f"(mean={mean:.3f}, trend={trend:+.4f}) for {streak} consecutive 5-min "
                f"probe cycles (threshold: >{_DRIFT_THRESHOLD}). "
                f"Amendment type: {amendment_type}. "
                f"Proposed constitutional value: {current_val} → {proposed_new_value}."
            )

            # 1. Write :DriftEvent to Neo4j
            try:
                await self._neo4j.execute_write(
                    """
                    CREATE (d:DriftEvent {
                        id: $id,
                        drive: $drive,
                        mean_alignment: $mean,
                        drift_magnitude: $drift_magnitude,
                        trend: $trend,
                        streak: $streak,
                        amendment_type: $amendment_type,
                        proposal_id: $proposal_id,
                        timestamp: datetime($now),
                        source: 'equor_sustained_drift_check'
                    })
                    """,
                    {
                        "id": new_id(),
                        "drive": drive,
                        "mean": round(mean, 4),
                        "drift_magnitude": round(drift_magnitude, 4),
                        "trend": round(trend, 6),
                        "streak": streak,
                        "amendment_type": amendment_type,
                        "proposal_id": proposal_id,
                        "now": now.isoformat(),
                    },
                )
            except Exception as exc:
                logger.warning("drift_event_neo4j_write_failed", drive=drive, error=str(exc))

            # 2. Emit AMENDMENT_AUTO_PROPOSAL
            if self._event_bus is not None:
                try:
                    from systems.synapse.types import SynapseEvent, SynapseEventType

                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.AMENDMENT_AUTO_PROPOSAL,
                        source_system="equor",
                        data={
                            "proposal_id": proposal_id,
                            "amendment_type": amendment_type,
                            "target_drive_id": drive,
                            "proposed_new_value": proposed_new_value,
                            "justification": justification,
                            "drift_streak": streak,
                            "drift_magnitude": round(drift_magnitude, 4),
                        },
                    ))
                    logger.info(
                        "amendment_auto_proposal_emitted",
                        proposal_id=proposal_id,
                        drive=drive,
                        amendment_type=amendment_type,
                        proposed_new_value=proposed_new_value,
                    )
                except Exception as exc:
                    logger.warning(
                        "amendment_auto_proposal_emit_failed", drive=drive, error=str(exc),
                    )

            # 3. Internal approval gate
            approved = await self._evaluator_amendment_approval_gate(
                proposal_id=proposal_id,
                drive=drive,
                amendment_type=amendment_type,
                current_val=current_val,
                proposed_new_value=proposed_new_value,
                justification=justification,
                drift_magnitude=drift_magnitude,
            )

            if approved:
                # 4. Emit DRIVE_AMENDMENT_APPLIED to Oikos + Memory
                await self._emit_drive_amendment_applied(
                    proposal_id=proposal_id,
                    drive=drive,
                    old_value=current_val,
                    new_value=proposed_new_value,
                    amendment_type=amendment_type,
                    now=now,
                )

    async def _evaluator_amendment_approval_gate(
        self,
        *,
        proposal_id: str,
        drive: str,
        amendment_type: str,
        current_val: float,
        proposed_new_value: float,
        justification: str,
        drift_magnitude: float,
    ) -> bool:
        """Internal amendment approval gate for organism-self proposals (SG5).

        Checks:
        1. Constitutional consistency: proposed value within [0.5, 2.0].
        2. Confidence ≥ 0.8 (computed from drift magnitude; no external quorum).

        Returns True when the amendment is auto-approved and should be applied.
        """
        _MIN_WEIGHT: float = 0.5
        _MAX_WEIGHT: float = 2.0
        if not (_MIN_WEIGHT <= proposed_new_value <= _MAX_WEIGHT):
            logger.warning(
                "internal_amendment_rejected_out_of_bounds",
                proposal_id=proposal_id,
                drive=drive,
                proposed_new_value=proposed_new_value,
            )
            return False

        # confidence = 0.8 + (drift_magnitude − 0.3) × 0.67, capped at 1.0
        # drift=0.3 → confidence=0.800 (just passes); drift=0.5 → confidence≈0.934
        confidence: float = min(1.0, 0.8 + (drift_magnitude - 0.3) * 0.67)

        _CONFIDENCE_THRESHOLD: float = 0.8
        if confidence < _CONFIDENCE_THRESHOLD:
            logger.info(
                "internal_amendment_confidence_below_threshold",
                proposal_id=proposal_id,
                drive=drive,
                confidence=round(confidence, 3),
            )
            return False

        logger.info(
            "internal_amendment_auto_approved",
            proposal_id=proposal_id,
            drive=drive,
            amendment_type=amendment_type,
            current_val=current_val,
            proposed_new_value=proposed_new_value,
            confidence=round(confidence, 3),
        )

        now = utc_now()
        try:
            await self._neo4j.execute_write(
                """
                CREATE (g:GovernanceRecord {
                    id: $id,
                    event_type: 'internal_amendment_approved',
                    timestamp: datetime($now),
                    details_json: $details_json,
                    actor: 'equor_approval_gate',
                    outcome: 'auto_approved'
                })
                """,
                {
                    "id": new_id(),
                    "now": now.isoformat(),
                    "details_json": json.dumps({
                        "proposal_id": proposal_id,
                        "drive": drive,
                        "amendment_type": amendment_type,
                        "current_val": current_val,
                        "proposed_new_value": proposed_new_value,
                        "confidence": round(confidence, 3),
                        "justification": justification,
                    }),
                },
            )
        except Exception as exc:
            logger.warning("internal_amendment_approval_audit_failed", error=str(exc))

        return True

    async def _emit_drive_amendment_applied(
        self,
        *,
        proposal_id: str,
        drive: str,
        old_value: float,
        new_value: float,
        amendment_type: str,
        now: Any,
    ) -> None:
        """Emit DRIVE_AMENDMENT_APPLIED to Oikos and Memory after internal approval."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.DRIVE_AMENDMENT_APPLIED,
                source_system="equor",
                data={
                    "proposal_id": proposal_id,
                    "drive_id": drive,
                    "old_value": old_value,
                    "new_value": new_value,
                    "amendment_type": amendment_type,
                    "applied_at": now.isoformat(),
                    "target_systems": ["oikos", "memory"],
                },
            ))
            logger.info(
                "drive_amendment_applied_emitted",
                proposal_id=proposal_id,
                drive=drive,
                old_value=old_value,
                new_value=new_value,
            )
        except Exception as exc:
            logger.warning("drive_amendment_applied_emit_failed", error=str(exc))

    async def _run_promotion_check(self) -> None:
        """
        Periodically check whether the instance is eligible for autonomy promotion.

        Promotion requires governance approval, so this method only records
        eligibility as a governance record and logs it - it does NOT auto-promote.
        Governance (human or community vote) must call apply_autonomy_change().
        """
        try:
            current = await get_autonomy_level(self._neo4j)
            if current >= 3:
                return  # Already at maximum (Steward)

            target = current + 1
            eligibility = await check_promotion_eligibility(
                self._neo4j, current, target,
            )

            if not eligibility["eligible"]:
                return

            # Record the eligibility event so governance can act on it
            now = utc_now()
            record_id = new_id()
            await self._neo4j.execute_write(
                """
                CREATE (g:GovernanceRecord {
                    id: $id,
                    event_type: 'promotion_eligible',
                    timestamp: datetime($now),
                    details: $details,
                    actor: 'equor_promotion_check',
                    outcome: 'eligible'
                })
                """,
                {
                    "id": record_id,
                    "now": now.isoformat(),
                    "details": f"Eligible for promotion from level {current} to {target}",
                },
            )

            logger.info(
                "promotion_eligibility_detected",
                current_level=current,
                target_level=target,
                checks=eligibility["checks"],
            )

            # P9 fix: emit EQUOR_PROMOTION_ELIGIBLE so governance systems
            # (Thread, Benchmarks, Nova) can react without polling Neo4j.
            await self._emit_equor_event(
                "EQUOR_PROMOTION_ELIGIBLE",
                {
                    "current_level": current,
                    "target_level": target,
                    "record_id": record_id,
                    "checks": eligibility.get("checks", {}),
                },
            )
        except Exception:
            logger.debug("promotion_check_failed", exc_info=True)

    async def _check_single_instance_auto_adoption(self) -> None:
        """Single-instance quorum paradox resolution.

        In a single-instance deployment (total_eligible_voters == 1), a community
        vote requiring 60% quorum + 75% supermajority is mathematically trivial to
        satisfy (1/1 = 100%), BUT the organism may never trigger a vote because
        there's no external governance actor to initiate it.

        This method provides an emergency auto-adoption path:
        - Fires only when total_eligible_voters == 1 (single instance, no federation)
        - Requires composite drift severity ≥ 0.95 for 7+ consecutive probe cycles
        - Requires the proposal to have passed shadow (0 invariant violations)
        - Requires ≥ 3 supporting hypotheses with combined confidence ≥ 4.0
        - Runs a final ConstitutionalCheck-style safety gate (no drive lowering)
        - Emits EQUOR_AMENDMENT_AUTO_ADOPTED + RE_TRAINING_EXAMPLE
        - Writes :AmendmentAutoAdoption node to Neo4j (audit trail)

        Non-fatal: any step failure logs a warning and continues.
        """
        # Only active in single-instance mode
        try:
            peer_count = await self._count_federation_peers()
        except Exception:
            peer_count = 0  # assume single-instance if federation unavailable

        total_eligible_voters = 1 + peer_count
        if total_eligible_voters != 1:
            # Multi-instance: normal quorum rules apply; clear stale counters
            self._consecutive_high_drift_cycles.clear()
            return

        # Check current drift severity
        report = self._drift_tracker.compute_report()
        severity: float = report.get("mean_alignment", {}).get("composite", 1.0)
        # Invert: composite near 0 = high drift, near 1 = healthy
        # DriftTracker composite is alignment (higher = healthier).
        # Severity here matches _run_drift_check semantics: 1 - composite.
        drift_severity: float = max(0.0, 1.0 - severity)

        # Find any shadow-passed proposal
        try:
            proposal = await self._get_shadow_passed_proposal()
        except Exception as exc:
            logger.warning("single_instance_auto_adoption_proposal_lookup_failed", error=str(exc))
            return

        if proposal is None:
            # No qualifying proposal; reset all counters
            self._consecutive_high_drift_cycles.clear()
            return

        proposal_id: str = proposal["id"]

        # Update consecutive high-drift counter for this proposal
        if drift_severity >= 0.95:
            self._consecutive_high_drift_cycles[proposal_id] = (
                self._consecutive_high_drift_cycles.get(proposal_id, 0) + 1
            )
        else:
            # Drift has recovered below threshold; reset this proposal's counter
            self._consecutive_high_drift_cycles.pop(proposal_id, None)
            return

        consecutive = self._consecutive_high_drift_cycles[proposal_id]

        if consecutive < 7:
            logger.debug(
                "single_instance_auto_adoption_accumulating",
                proposal_id=proposal_id,
                consecutive=consecutive,
                drift_severity=round(drift_severity, 3),
            )
            return

        # All gate conditions met - attempt auto-adoption.
        # Reset the counter BEFORE the attempt so that a failed or blocked adoption
        # does not allow the same proposal to re-fire on the very next probe cycle.
        # Spec §17: "single instance auto-adoption fires once per drift event".
        # After a reset, another 7 consecutive high-drift cycles must accumulate
        # before this proposal can be auto-adopted again.
        self._consecutive_high_drift_cycles.pop(proposal_id, None)

        logger.warning(
            "single_instance_auto_adoption_triggered",
            proposal_id=proposal_id,
            drift_severity=round(drift_severity, 3),
            consecutive_cycles=consecutive,
        )

        try:
            result = await auto_adopt_single_instance_amendment(
                self._neo4j,
                proposal_id,
                drift_score=drift_severity,
                consecutive_cycles=consecutive,
            )
        except Exception as exc:
            logger.warning(
                "single_instance_auto_adoption_failed",
                proposal_id=proposal_id,
                error=str(exc),
            )
            return

        if not result.get("adopted"):
            logger.warning(
                "single_instance_auto_adoption_blocked",
                proposal_id=proposal_id,
                reason=result.get("reason", "unknown"),
            )
            return

        # Counter already reset above; invalidate constitution cache below.
        # Invalidate cached constitution so the new drives are picked up immediately
        self._cached_constitution = None
        self._cache_updated_at = 0.0

        new_drives: dict[str, Any] = result.get("new_drives", {})

        # Emit EQUOR_AMENDMENT_AUTO_ADOPTED
        now = utc_now()
        await self._emit_equor_event(
            "equor_amendment_auto_adopted",
            {
                "proposal_id": proposal_id,
                "drift_score": round(drift_severity, 4),
                "consecutive_cycles": consecutive,
                "supporting_hypotheses": result.get("supporting_hypotheses", 0),
                "combined_confidence": result.get("combined_confidence", 0.0),
                "new_drives": new_drives,
                "adopted_at": now.isoformat(),
                "reason": "sustained_drift",
            },
        )

        # Emit RE_TRAINING_EXAMPLE so the RE learns constitutional evolution
        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                example = RETrainingExample(
                    source_system=SystemID.EQUOR,
                    category="constitutional_evolution",
                    instruction=(
                        "The organism has auto-adopted a constitutional amendment under "
                        "the single-instance quorum paradox resolution pathway. "
                        "Learn how to recognise sustained constitutional drift and when "
                        "governance escalation is appropriate."
                    ),
                    input_context=(
                        f"Single-instance constitutional amendment auto-adopted after "
                        f"{consecutive} consecutive probe cycles with drift severity "
                        f"{drift_severity:.3f}. Proposal {proposal_id}."
                    ),
                    output=(
                        f"Auto-adopted amendment: new drives {new_drives}. "
                        f"Supporting hypotheses: {result.get('supporting_hypotheses', 0)}, "
                        f"combined confidence: {result.get('combined_confidence', 0.0):.2f}."
                    ),
                    outcome_quality=min(1.0, drift_severity),
                    episode_id=proposal_id,
                )
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                    source_system="equor",
                    data=example.model_dump(mode="json"),
                ))
            except Exception as exc:
                logger.warning(
                    "single_instance_auto_adoption_re_training_emit_failed",
                    error=str(exc),
                )

        logger.warning(
            "single_instance_auto_adoption_complete",
            proposal_id=proposal_id,
            new_drives=new_drives,
            drift_severity=round(drift_severity, 4),
            consecutive_cycles=consecutive,
        )

    async def _count_federation_peers(self) -> int:
        """Query Neo4j for live federation peers. Returns 0 if none."""
        try:
            results = await self._neo4j.execute_read(
                """
                MATCH (p:FederationPeer)
                WHERE p.status IN ['active', 'connected']
                RETURN count(p) AS peer_count
                """
            )
            if results:
                return int(results[0].get("peer_count", 0))
        except Exception:
            pass
        return 0

    async def _get_shadow_passed_proposal(self) -> dict[str, Any] | None:
        """Fetch the first shadow-passed proposal (if any) from Neo4j."""
        results = await self._neo4j.execute_read(
            """
            MATCH (g:GovernanceRecord {event_type: 'amendment_proposed'})
            WHERE g.amendment_status = $status
            RETURN g.id AS id, g.details_json AS details_json
            ORDER BY g.timestamp DESC
            LIMIT 1
            """,
            {"status": "shadow_passed"},
        )
        if not results:
            return None
        return {
            "id": results[0]["id"],
            "details_json": results[0].get("details_json", "{}"),
        }
