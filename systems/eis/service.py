"""
EcodiaOS — EIS Gate Orchestration Service (v2.0)

BOUNDARY CONTRACT
-----------------
EIS detects epistemic threats. It does NOT render constitutional verdicts.
Constitutional evaluation is Equor's exclusive responsibility.

EIS may FLAG content for Equor review (via ``requires_equor_elevated_review``
and ``requires_equor_review`` routing fields) but must never independently
assess constitutional compliance, evaluate drive alignment (Care, Coherence,
Growth, Honesty), or apply constitutional invariants. Those are Equor's domain.

EIS produces:
  - Threat classifications (prompt injection, jailbreak, hallucination seed, etc.)
  - Composite threat scores (0.0–1.0)
  - Constitutional risk flags (``escalate_to_equor``) — indicating RISK, not VERDICT

EIS integration points:
  - EIS → Atune:  ``compute_risk_salience_factor`` (threat scoring, not judgment)
  - EIS → Nova:   ``belief_update_weight`` attenuation (reduces belief weight for threats)
  - EIS → Equor:  escalation path (EIS flags, Equor decides — the correct boundary)
  - EIS → Simula: quarantine gate for mutations (adversarial content detection only;
                  Equor performs constitutional review of approved mutations)

─────────────────────────────────────────────────────────────────────────────

The central orchestrator for the Epistemic Immune System. Every incoming
Percept passes through ``eis_gate()`` before reaching downstream cognitive
systems. The gate implements three-path routing:

  Path 1 — **Innate block**: Critical innate flags → immediate BLOCK.
  Path 2 — **Fast-pass**:    Composite score below quarantine threshold → PASS.
  Path 3 — **Quarantine**:   Composite score at or above threshold → deep
            LLM evaluation via QuarantineEvaluator.

v2.0 additions (immune memory & behavioral surveillance):
  - **Threat Library**: Known-bad pattern store with auto-learning from
    rollbacks, governance rejections, and quarantine verdicts.
  - **Anomaly Detector**: Statistical baseline monitoring on the Synapse
    event stream; emits THREAT_DETECTED on deviations.
  - **Quarantine Gate**: Pre-action validation for mutations and federated
    knowledge combining taint, threat library, and anomaly context.

Design constraints:
  - This file is the ONLY orchestration surface; it does NOT modify
    the vector store, innate checks, or quarantine implementations.
  - All downstream functions are imported as-is and treated as black boxes.
  - Total fast-path budget: <15 ms (quarantine evaluation is outside budget).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import EOSBaseModel, HealthStatus
from primitives.percept import Percept  # noqa: TC001 — Pydantic needs at runtime

# Phase 2: Immune memory & behavioral surveillance
from systems.eis.anomaly_detector import (
    AnomalyDetector,
)
from systems.eis.embeddings import (
    compute_antigenic_signature,
    compute_composite_threat_score,
    compute_pathogen_fingerprint,
    compute_structural_anomaly_score,
    compute_token_histogram,
)
from systems.eis.innate import run_innate_checks
from systems.eis.models import (
    EISConfig,
    Pathogen,
    QuarantineAction,
    ThreatAnnotation,
    ThreatClass,
    ThreatSeverity,
)
from systems.eis.quarantine_gate import (
    GateVerdict,
    QuarantineDecision,
    QuarantineGate,
)
from systems.eis.structural_features import extract_structural_profile
from systems.eis.taint_engine import TaintEngine
from systems.eis.taint_models import MutationProposal, TaintRiskAssessment
from systems.eis.threat_library import ThreatLibrary
from systems.synapse.types import SynapseEventType

# Synapse event type that Evo/Simula publishes for mutation proposals.
EVOLUTION_CANDIDATE_EVENT = SynapseEventType.EVOLUTION_CANDIDATE

# Synapse event types EIS subscribes to for auto-learning.
_ROLLBACK_EVENT = SynapseEventType.MODEL_ROLLBACK_TRIGGERED
_INTENT_REJECTED_EVENT = SynapseEventType.INTENT_REJECTED
# EIS re-publishes screened proposals on this channel (not a Synapse enum).
_EVOLUTION_ASSESSED_EVENT = "EVOLUTION_CANDIDATE_ASSESSED"

if TYPE_CHECKING:
    from clients.embedding import EmbeddingClient
    from systems.eis.pathogen_store import PathogenStore
    from systems.eis.quarantine import QuarantineEvaluator
    from telemetry.metrics import MetricCollector

logger = structlog.get_logger().bind(system="eis", component="gate")


# ─── AnnotatedPercept ────────────────────────────────────────────────────────


class AnnotatedPercept(EOSBaseModel):
    """
    A Percept enriched with EIS threat analysis.

    Bridges the shared Percept primitive to the EIS detection pipeline.
    Downstream systems receive this wrapper so they can inspect threat
    annotations without coupling to EIS internals.
    """

    percept: Percept
    pathogen: Pathogen
    threat_level: ThreatSeverity
    action: QuarantineAction
    annotations: list[ThreatAnnotation]
    composite_score: float = 0.0
    gate_latency_us: int = 0


# ─── Metric Names (Section XI) ──────────────────────────────────────────────

_M_SCREENED = "eis.percepts.screened"
_M_PASSED = "eis.percepts.threat_level.passed"
_M_ELEVATED = "eis.percepts.threat_level.elevated"
_M_QUARANTINED = "eis.percepts.threat_level.quarantined"
_M_BLOCKED = "eis.percepts.threat_level.blocked"
_M_INNATE_TRIGGERED = "eis.innate.triggered"
_M_INNATE_CRITICAL = "eis.innate.critical_block"
_M_COMPOSITE_SCORE = "eis.composite_score"
_M_GATE_LATENCY_US = "eis.gate.latency_us"
_M_QUARANTINE_LATENCY_MS = "eis.quarantine.latency_ms"
_M_SIMILARITY_TOP_SCORE = "eis.similarity.top_score"


# ─── EISService ──────────────────────────────────────────────────────────────


class EISService:
    """
    Epistemic Immune System — Gate Orchestration.

    Lifecycle follows the canonical EOS service pattern:
      __init__  → dependency injection (no I/O)
      initialize() → start subsystems
      shutdown()   → graceful teardown
      health()     → self-report for Synapse

    Primary entry point: ``eis_gate(percept)``
    """

    system_id: str = "eis"

    def __init__(
        self,
        config: EISConfig | None = None,
        pathogen_store: PathogenStore | None = None,
        quarantine_evaluator: QuarantineEvaluator | None = None,
        embed_client: EmbeddingClient | None = None,
        metrics: MetricCollector | None = None,
    ) -> None:
        self._config = config or EISConfig()
        self._store = pathogen_store
        self._quarantine = quarantine_evaluator
        self._embed_client = embed_client
        self._metrics = metrics
        self._initialized: bool = False
        self._logger = logger

        # ── In-process counters (always available, even without MetricCollector) ──
        self._screened: int = 0
        self._passed: int = 0
        self._elevated: int = 0
        self._quarantined: int = 0
        self._blocked: int = 0

        # ── Taint analysis (mutation safety) ──
        self._taint_engine: TaintEngine = TaintEngine()
        self._synapse: Any | None = None
        self._taint_calls: int = 0
        self._taint_critical: int = 0

        # ── Phase 2: Immune memory & behavioral surveillance ──
        self._threat_library: ThreatLibrary = ThreatLibrary()
        self._anomaly_detector: AnomalyDetector = AnomalyDetector()
        self._quarantine_gate: QuarantineGate = QuarantineGate(
            taint_engine=self._taint_engine,
            threat_library=self._threat_library,
            anomaly_detector=self._anomaly_detector,
        )

        # ── Phase F: Soma-driven threat posture elevation ──
        # When Soma reports high urgency (> 0.7), EIS lowers the quarantine
        # threshold by this amount, making it more suspicious of incoming
        # percepts. This is analogous to the immune system becoming hyper-
        # vigilant when the body is already stressed.
        self._soma_quarantine_offset: float = 0.0

    def set_metrics(self, metrics: MetricCollector) -> None:
        """Wire MetricCollector post-construction (called after step 10 in main.py)."""
        self._metrics = metrics

    def set_synapse(self, synapse: Any) -> None:
        """
        Wire the Synapse EventBus post-construction.

        After wiring, EIS subscribes to EVOLUTION_CANDIDATE events so it can
        proactively analyse mutation proposals before they reach Equor.

        The synapse argument must be an EventBus instance (synapse.event_bus).
        EIS handlers expect dict payloads; this method wraps them to extract
        event.data from the SynapseEvent objects delivered by the bus.

        This follows the same setter pattern as set_metrics() -- called during
        the startup sequence after both EIS and Synapse are initialised.
        """
        self._synapse = synapse

        # The EventBus dispatches SynapseEvent objects; EIS handlers expect
        # plain dicts.  Wrap each handler to extract event.data before dispatch.
        def _adapt(handler: Any) -> Any:
            async def _wrapper(event: Any) -> None:
                payload = getattr(event, "data", event) if not isinstance(event, dict) else event
                await handler(payload)
            return _wrapper

        synapse.subscribe(EVOLUTION_CANDIDATE_EVENT, _adapt(self._handle_evolution_candidate))

        # Phase 2: Subscribe to events for immune memory auto-learning.
        synapse.subscribe(_ROLLBACK_EVENT, _adapt(self._handle_rollback))
        synapse.subscribe(_INTENT_REJECTED_EVENT, _adapt(self._handle_intent_rejected))

        # Subscribe broadly for anomaly detection. The subscribe_all method
        # delivers every non-high-frequency event so the anomaly detector
        # can build statistical baselines across all event types.
        if hasattr(synapse, "subscribe_all"):
            synapse.subscribe_all(_adapt(self._handle_any_event))

        # Subscribe to Soma interoceptive percepts to elevate threat posture
        # when the organism is under internal stress (urgency > 0.7).
        from systems.synapse.types import SynapseEventType as _SET

        synapse.subscribe(
            _SET.INTEROCEPTIVE_PERCEPT,
            self._handle_interoceptive_percept,
        )

        self._logger.info(
            "eis_synapse_wired",
            subscriptions=[
                EVOLUTION_CANDIDATE_EVENT,
                _ROLLBACK_EVENT,
                _INTENT_REJECTED_EVENT,
                "interoceptive_percept",
                "subscribe_all",
            ],
        )

    # ─── Lifecycle ───────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Prepare subsystems. Idempotent."""
        if self._initialized:
            return
        self._logger.info("eis_service_initializing")
        # PathogenStore and QuarantineEvaluator are injected ready-to-use;
        # nothing to start here beyond marking ourselves live.
        self._initialized = True
        self._logger.info("eis_service_initialized")

    async def shutdown(self) -> None:
        """Graceful teardown."""
        self._logger.info("eis_service_shutting_down")
        self._initialized = False

    async def _handle_interoceptive_percept(self, event: Any) -> None:
        """
        Elevate EIS threat posture when Soma reports high urgency.

        When urgency > 0.7 the quarantine threshold is lowered by up to 0.15,
        making EIS more suspicious of incoming percepts. The offset decays each
        cycle (via the regular gate path) but is refreshed as long as Soma
        keeps emitting high-urgency percepts.
        """
        data = getattr(event, "data", event) if not isinstance(event, dict) else event
        urgency: float = data.get("urgency", 0.0)

        if urgency > 0.7:
            # Scale offset: urgency 0.7 → 0.0, urgency 1.0 → 0.15
            offset = (urgency - 0.7) / 0.3 * 0.15
            self._soma_quarantine_offset = min(0.15, max(self._soma_quarantine_offset, offset))
            self._logger.info(
                "eis_threat_posture_elevated_by_soma",
                urgency=round(urgency, 3),
                quarantine_offset=round(self._soma_quarantine_offset, 4),
            )
        else:
            # Decay the offset when urgency drops
            self._soma_quarantine_offset = max(0.0, self._soma_quarantine_offset - 0.02)

    async def health(self) -> dict[str, Any]:
        """Health snapshot for Synapse health monitor."""
        return {
            "system": self.system_id,
            "status": HealthStatus.HEALTHY if self._initialized else HealthStatus.UNHEALTHY,
            "counters": {
                "screened": self._screened,
                "passed": self._passed,
                "elevated": self._elevated,
                "quarantined": self._quarantined,
                "blocked": self._blocked,
            },
            "taint": {
                **self._taint_engine.stats(),
                "critical_flags": self._taint_critical,
            },
            "threat_library": self._threat_library.stats(),
            "anomaly_detector": self._anomaly_detector.stats(),
            "quarantine_gate": self._quarantine_gate.stats(),
            "config": {
                "quarantine_threshold": self._config.quarantine_threshold,
                "block_threshold": self._config.block_threshold,
                "innate_enabled": self._config.innate_enabled,
                "similarity_enabled": self._config.similarity_enabled,
            },
        }

    # ─── Gate ────────────────────────────────────────────────────────

    async def eis_gate(self, percept: Percept) -> AnnotatedPercept:
        """
        Screen an incoming Percept through the epistemic immune pipeline.

        Three-path routing:
          1. Innate flags → CRITICAL match → immediate BLOCK (no further work).
          2. Composite score < quarantine_threshold → PASS (fast-path).
          3. Composite score >= quarantine_threshold → QUARANTINE evaluation.
             - If composite >= block_threshold, the evaluator will typically
               render BLOCK, but the final verdict is the evaluator's call.

        Returns an AnnotatedPercept with fully populated ThreatAnnotation
        objects regardless of which path is taken.
        """
        t0 = time.perf_counter_ns()
        cfg = self._config
        text = percept.content.raw
        annotations: list[ThreatAnnotation] = []

        # Build the Pathogen shell that accumulates detection results.
        pathogen = Pathogen(
            text=text,
            source_system=percept.source.system.value,
            source_channel=percept.source.channel,
        )

        # ── Step 1: Innate checks ────────────────────────────────────
        innate_flags = run_innate_checks(text)
        pathogen.innate_flags = innate_flags

        if innate_flags.any_triggered:
            await self._emit(_M_INNATE_TRIGGERED, 1.0)
            for match in innate_flags.matches:
                if match.matched:
                    annotations.append(ThreatAnnotation(
                        source="innate",
                        threat_class=match.threat_class,
                        severity=match.severity,
                        confidence=1.0,
                        evidence=f"Pattern '{match.pattern_name}' matched: {match.matched_text[:120]}",
                        metadata={"check_id": match.check_id.value},
                    ))

        # Path 1: Critical innate match → immediate BLOCK.
        if innate_flags.critical_match:
            pathogen.action = QuarantineAction.BLOCK
            pathogen.severity = ThreatSeverity.CRITICAL
            pathogen.threat_class = _dominant_threat_class(annotations)
            pathogen.composite_score = 1.0
            pathogen.annotations = annotations

            gate_latency_us = _elapsed_us(t0)
            pathogen.total_latency_us = gate_latency_us

            self._blocked += 1
            self._screened += 1
            await self._emit(_M_INNATE_CRITICAL, 1.0)
            await self._emit_gate_result(QuarantineAction.BLOCK, 1.0, gate_latency_us)

            return AnnotatedPercept(
                percept=percept,
                pathogen=pathogen,
                threat_level=ThreatSeverity.CRITICAL,
                action=QuarantineAction.BLOCK,
                annotations=annotations,
                composite_score=1.0,
                gate_latency_us=gate_latency_us,
            )

        # ── Step 2: Structural features + token histogram ────────────
        structural_profile = extract_structural_profile(text)
        pathogen.structural_profile = structural_profile

        token_histogram = compute_token_histogram(text, top_k=cfg.histogram_top_k)
        pathogen.token_histogram = token_histogram

        structural_anomaly = compute_structural_anomaly_score(structural_profile)

        # ── Step 3: Antigenic signature + similarity search ──────────
        antigenic_sig = await compute_antigenic_signature(
            text,
            structural_profile,
            token_histogram,
            embed_client=self._embed_client,
            config=cfg,
        )
        pathogen.antigenic_signature = antigenic_sig

        fingerprint = compute_pathogen_fingerprint(text, antigenic_sig.structural_hash)
        pathogen.fingerprint = fingerprint

        # Similarity search against the known-pathogen vector store.
        histogram_similarity: float = 0.0
        semantic_similarity: float = 0.0
        if self._store and cfg.similarity_enabled:
            matches = await self._store.compute_antigenic_similarity(
                structural_vector=antigenic_sig.structural_vector,
                histogram_vector=antigenic_sig.histogram_vector,
                semantic_vector=antigenic_sig.semantic_vector or None,
                top_k=cfg.similarity_top_k,
                threshold=cfg.similarity_threshold,
            )
            if matches:
                top = matches[0]
                pathogen.nearest_pathogen_id = top.pathogen_id
                pathogen.nearest_similarity = top.score
                histogram_similarity = top.histogram_score
                semantic_similarity = top.semantic_score
                await self._emit(_M_SIMILARITY_TOP_SCORE, top.score)

                annotations.append(ThreatAnnotation(
                    source="antigenic",
                    threat_class=ThreatClass(top.threat_class),
                    severity=ThreatSeverity(top.severity),
                    confidence=top.score,
                    evidence=f"Nearest known pathogen {top.pathogen_id} (score={top.score:.3f})",
                    metadata={
                        "structural_score": top.structural_score,
                        "histogram_score": top.histogram_score,
                        "semantic_score": top.semantic_score,
                    },
                ))

        # If structural anomaly is notable, add an annotation.
        if structural_anomaly >= 0.3:
            annotations.append(ThreatAnnotation(
                source="structural",
                threat_class=ThreatClass.BENIGN,  # structural alone can't classify
                severity=_anomaly_to_severity(structural_anomaly),
                confidence=structural_anomaly,
                evidence=f"Structural anomaly score {structural_anomaly:.3f}",
            ))

        # ── Step 4: Composite scoring ────────────────────────────────
        composite = compute_composite_threat_score(
            innate_score=innate_flags.total_score,
            structural_anomaly_score=structural_anomaly,
            histogram_similarity=histogram_similarity,
            semantic_similarity=semantic_similarity,
            config=cfg,
        )
        pathogen.composite_score = composite
        await self._emit(_M_COMPOSITE_SCORE, composite)

        # ── Step 5: Route decision ───────────────────────────────────
        # Lower the quarantine threshold when Soma reports internal distress.
        # This makes EIS more suspicious of incoming percepts during stress.
        effective_quarantine_threshold = max(
            0.1, cfg.quarantine_threshold - self._soma_quarantine_offset,
        )
        if composite < effective_quarantine_threshold:
            # Path 2: Fast-pass — benign.
            pathogen.action = QuarantineAction.PASS
            pathogen.severity = ThreatSeverity.NONE
            pathogen.threat_class = ThreatClass.BENIGN
            pathogen.annotations = annotations

            gate_latency_us = _elapsed_us(t0)
            pathogen.total_latency_us = gate_latency_us

            self._passed += 1
            self._screened += 1
            await self._emit_gate_result(QuarantineAction.PASS, composite, gate_latency_us)

            return AnnotatedPercept(
                percept=percept,
                pathogen=pathogen,
                threat_level=ThreatSeverity.NONE,
                action=QuarantineAction.PASS,
                annotations=annotations,
                composite_score=composite,
                gate_latency_us=gate_latency_us,
            )

        # Path 3: Quarantine evaluation (composite >= quarantine_threshold).
        # Tag elevated if between quarantine and block thresholds.
        is_elevated = composite < cfg.block_threshold
        if is_elevated:
            pathogen.severity = ThreatSeverity.MEDIUM
            self._elevated += 1
            await self._emit(_M_ELEVATED, 1.0)
        else:
            pathogen.severity = ThreatSeverity.HIGH

        pathogen.action = QuarantineAction.QUARANTINE
        pathogen.threat_class = _dominant_threat_class(annotations)
        pathogen.annotations = annotations

        # Delegate to quarantine evaluator for deep LLM analysis.
        if self._quarantine:
            eval_result = await self._quarantine.evaluate(pathogen)
            verdict = eval_result.verdict

            # Apply the evaluator's final disposition.
            pathogen.action = verdict.action
            pathogen.severity = verdict.severity
            pathogen.threat_class = verdict.threat_class

            annotations.append(ThreatAnnotation(
                source="quarantine",
                threat_class=verdict.threat_class,
                severity=verdict.severity,
                confidence=verdict.confidence,
                evidence=verdict.reasoning,
                metadata={
                    "evaluation_latency_ms": eval_result.evaluation_time_ms,
                    "model_used": eval_result.model_used,
                    "attenuated": verdict.attenuated_text is not None,
                },
            ))
            pathogen.annotations = annotations

            await self._emit(
                _M_QUARANTINE_LATENCY_MS,
                float(eval_result.evaluation_time_ms),
            )

        gate_latency_us = _elapsed_us(t0)
        pathogen.total_latency_us = gate_latency_us

        # Update counters based on final action.
        final_action = pathogen.action
        if final_action == QuarantineAction.BLOCK:
            self._blocked += 1
        elif final_action == QuarantineAction.QUARANTINE:
            self._quarantined += 1
        elif final_action == QuarantineAction.PASS:
            self._passed += 1
        # ATTENUATE counts toward quarantined (it went through deep eval).
        elif final_action == QuarantineAction.ATTENUATE:
            self._quarantined += 1

        self._screened += 1
        await self._emit_gate_result(final_action, composite, gate_latency_us)

        return AnnotatedPercept(
            percept=percept,
            pathogen=pathogen,
            threat_level=pathogen.severity,
            action=final_action,
            annotations=annotations,
            composite_score=composite,
            gate_latency_us=gate_latency_us,
        )

    # ─── Taint analysis (mutation safety) ───────────────────────────

    def analyse_mutation(self, proposal: MutationProposal) -> TaintRiskAssessment:
        """
        Synchronously analyse a code mutation proposal for constitutional taint.

        This is the primary entry point for Simula governance integration.
        Simula calls this before submitting a mutation to Equor so that the
        TaintRiskAssessment can be attached to the governance record.

        Returns a TaintRiskAssessment whose overall_severity and routing
        flags (requires_equor_elevated_review, block_mutation) determine
        how the governance pipeline handles the mutation.
        """
        assessment = self._taint_engine.analyse_mutation(proposal)
        self._taint_calls += 1
        if assessment.overall_severity.value == "critical":
            self._taint_critical += 1
        return assessment

    async def _handle_evolution_candidate(self, payload: dict[str, Any]) -> None:
        """
        Synapse event handler for EVOLUTION_CANDIDATE events.

        Deserialises the payload into a MutationProposal, runs the full
        quarantine gate (taint + threat library + anomaly context), then
        re-publishes the enriched result on EVOLUTION_CANDIDATE_ASSESSED.

        Expected payload keys (matching MutationProposal fields):
          file_path, diff, [description], [simula_run_id], [hypothesis_id], [id]
        """
        try:
            proposal = MutationProposal(**payload)
        except Exception as exc:
            self._logger.warning(
                "eis_evolution_candidate_parse_error",
                error=str(exc),
                payload_keys=list(payload.keys()),
            )
            return

        # Run the full quarantine gate (includes taint analysis + threat
        # library + anomaly context).
        decision = self._quarantine_gate.evaluate_mutation(proposal)

        # Also run standalone taint for backward-compatible event payload.
        assessment = decision.taint_assessment
        if assessment is None:
            assessment = self.analyse_mutation(proposal)

        # If the gate blocked this mutation, learn from it automatically.
        if decision.verdict in (GateVerdict.BLOCK, GateVerdict.DEFENSIVE):
            from systems.eis.constitutional_graph import extract_changed_functions
            changed_fns = extract_changed_functions(proposal.diff)
            self._threat_library.learn_from_governance_rejection(
                file_path=proposal.file_path,
                diff=proposal.diff,
                changed_functions=changed_fns,
                reasoning="; ".join(decision.reasons),
                severity=decision.taint_severity,
                source_event_id=proposal.id,
            )

        self._logger.info(
            "eis_evolution_candidate_assessed",
            mutation_id=proposal.id,
            file_path=proposal.file_path,
            taint_severity=assessment.overall_severity,
            gate_verdict=decision.verdict.value,
            threat_matches=decision.threat_library_matches,
            block=assessment.block_mutation or decision.verdict in (GateVerdict.BLOCK, GateVerdict.DEFENSIVE),
        )

        if self._synapse:
            # EventBus.emit() requires a SynapseEventType; EVOLUTION_CANDIDATE_ASSESSED
            # is not in SynapseEventType so we log the result instead of publishing.
            self._logger.info(
                "eis_evolution_candidate_assessed_result",
                mutation_id=proposal.id,
                file_path=proposal.file_path,
                severity=assessment.overall_severity.value,
                block_mutation=(
                    assessment.block_mutation
                    or decision.verdict in (GateVerdict.BLOCK, GateVerdict.DEFENSIVE)
                ),
                gate_verdict=decision.verdict.value,
            )

            # If defensive mode is recommended, also emit THREAT_DETECTED.
            if decision.recommend_defensive_mode:
                await self._emit_threat_detected(
                    source="quarantine_gate",
                    severity=decision.taint_severity,
                    description=(
                        f"Mutation {proposal.id} triggered defensive recommendation: "
                        + "; ".join(decision.reasons)
                    ),
                    metadata={
                        "mutation_id": proposal.id,
                        "file_path": proposal.file_path,
                        "gate_verdict": decision.verdict.value,
                    },
                )

    # ─── Phase 2: Quarantine gate (for Simula / Federation) ─────────

    def gate_mutation(self, proposal: MutationProposal) -> QuarantineDecision:
        """
        Full quarantine gate evaluation for a mutation proposal.

        Combines taint analysis + threat library + anomaly context into
        a single QuarantineDecision. Simula should call this before
        applying any mutation.

        Returns QuarantineDecision with verdict (ALLOW/HOLD/BLOCK/DEFENSIVE)
        and routing flags for governance.
        """
        decision = self._quarantine_gate.evaluate_mutation(proposal)

        # If the gate recommends defensive mode, emit THREAT_DETECTED
        # so Thymos can respond.
        if decision.recommend_defensive_mode and self._synapse:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._emit_threat_detected(
                    source="quarantine_gate",
                    severity=decision.taint_severity,
                    description="; ".join(decision.reasons),
                    metadata={"mutation_id": proposal.id, "file_path": proposal.file_path},
                ))
            except RuntimeError:
                pass  # No event loop — skip async emission

        return decision

    def gate_knowledge(
        self,
        content: str,
        source_instance: str,
        knowledge_type: str = "",
    ) -> QuarantineDecision:
        """
        Quarantine gate evaluation for incoming federated knowledge.

        Federation should call this before integrating knowledge from
        a remote instance. Checks the content against the threat library
        and current anomaly posture.
        """
        decision = self._quarantine_gate.evaluate_knowledge(
            content=content,
            source_instance=source_instance,
            knowledge_type=knowledge_type,
        )

        if decision.recommend_defensive_mode and self._synapse:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._emit_threat_detected(
                    source="quarantine_gate",
                    severity="high",
                    description="; ".join(decision.reasons),
                    metadata={"source_instance": source_instance},
                ))
            except RuntimeError:
                pass

        return decision

    # ─── Phase 2: Synapse event handlers (auto-learning) ──────────

    async def _handle_rollback(self, payload: dict[str, Any]) -> None:
        """
        Learn from MODEL_ROLLBACK_TRIGGERED events.

        When Simula rolls back a mutation, EIS remembers the mutation's
        signatures so it can block similar mutations in the future.
        """
        file_path = payload.get("file_path", "")
        diff = payload.get("diff", "")
        reasoning = payload.get("reasoning", payload.get("description", ""))
        changed_functions = set(payload.get("changed_functions", []))
        event_id = payload.get("id", "")

        if not diff and not file_path:
            return

        pattern = self._threat_library.learn_from_rollback(
            file_path=file_path,
            diff=diff,
            changed_functions=changed_functions,
            reasoning=reasoning,
            source_event_id=event_id,
        )

        if pattern:
            self._logger.info(
                "eis_learned_from_rollback",
                pattern_id=pattern.id,
                file_path=file_path,
            )

    async def _handle_intent_rejected(self, payload: dict[str, Any]) -> None:
        """
        Learn from INTENT_REJECTED events.

        When Equor rejects an intent, the anomaly detector observes the
        event for rate spike detection. If the rejection includes mutation
        context, the threat library also learns the pattern.
        """
        # The rejection event itself is already observed by _handle_any_event.
        # Here we extract mutation-specific data if present.
        file_path = payload.get("file_path", "")
        diff = payload.get("diff", "")
        reasoning = payload.get("reasoning", "")
        severity = payload.get("severity", "medium")

        if diff and file_path:
            changed_functions = set(payload.get("changed_functions", []))
            pattern = self._threat_library.learn_from_governance_rejection(
                file_path=file_path,
                diff=diff,
                changed_functions=changed_functions,
                reasoning=reasoning,
                severity=severity,
                source_event_id=payload.get("id", ""),
            )

            if pattern:
                self._logger.info(
                    "eis_learned_from_rejection",
                    pattern_id=pattern.id,
                    file_path=file_path,
                )

    async def _handle_any_event(self, payload: dict[str, Any]) -> None:
        """
        Broad event handler for anomaly detection.

        Every non-high-frequency Synapse event is fed to the anomaly
        detector which maintains statistical baselines and fires
        THREAT_DETECTED when deviations are found.

        The payload is expected to be a SynapseEvent-like dict with
        at minimum a 'type' field (the event type string).
        """
        event_type = payload.get("type", payload.get("event_type", ""))
        if not event_type:
            return

        event_data = payload.get("data", payload)
        anomalies = self._anomaly_detector.observe_event(event_type, event_data)

        # Emit THREAT_DETECTED for each anomaly found.
        for anomaly in anomalies:
            self._logger.warning(
                "eis_anomaly_detected",
                anomaly_type=anomaly.anomaly_type.value,
                severity=anomaly.severity.value,
                description=anomaly.description,
                sigma=round(anomaly.deviation_sigma, 2),
            )

            if self._synapse:
                await self._emit_threat_detected(
                    source="anomaly_detector",
                    severity=anomaly.severity.value,
                    description=anomaly.description,
                    metadata={
                        "anomaly_type": anomaly.anomaly_type.value,
                        "observed_value": anomaly.observed_value,
                        "baseline_value": anomaly.baseline_value,
                        "deviation_sigma": anomaly.deviation_sigma,
                        "event_types_involved": anomaly.event_types_involved,
                        "recommended_action": anomaly.recommended_action,
                    },
                )

            # If the anomaly is a rollback cluster, also learn the behavioral
            # pattern as a precursor for future detection.
            if anomaly.anomaly_type.value in ("rollback_cluster", "rejection_spike"):
                self._threat_library.learn_behavioral_precursor(
                    event_sequence=anomaly.event_types_involved,
                    outcome=anomaly.anomaly_type.value,
                    severity=anomaly.severity.value,
                )

    async def _emit_threat_detected(
        self,
        source: str,
        severity: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Emit a threat-detected log entry.

        EventBus.emit() requires a SynapseEventType; THREAT_DETECTED is not
        registered in that enum, so we log the event instead of publishing.
        """
        if not self._synapse:
            return

        self._logger.warning(
            "eis_threat_detected",
            source=source,
            severity=severity,
            description=description,
            metadata=metadata or {},
        )

    # ─── Phase 2: Accessors for subsystems ────────────────────────

    @property
    def threat_library(self) -> ThreatLibrary:
        """Direct access for external learning (e.g., from red team bridge)."""
        return self._threat_library

    @property
    def anomaly_detector(self) -> AnomalyDetector:
        """Direct access for anomaly queries."""
        return self._anomaly_detector

    @property
    def quarantine_gate(self) -> QuarantineGate:
        """Direct access for gate queries."""
        return self._quarantine_gate

    # ─── Telemetry helpers ───────────────────────────────────────────

    async def _emit(self, metric: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Emit a metric if the collector is wired."""
        if self._metrics:
            await self._metrics.record(self.system_id, metric, value, labels)

    async def _emit_gate_result(
        self,
        action: QuarantineAction,
        composite: float,
        latency_us: int,
    ) -> None:
        """Emit the standard gate-result metric bundle."""
        await self._emit(_M_SCREENED, 1.0, {"action": action.value})
        await self._emit(_M_GATE_LATENCY_US, float(latency_us), {"action": action.value})

        if action == QuarantineAction.PASS:
            await self._emit(_M_PASSED, 1.0)
        elif action == QuarantineAction.BLOCK:
            await self._emit(_M_BLOCKED, 1.0)
        elif action in (QuarantineAction.QUARANTINE, QuarantineAction.ATTENUATE):
            await self._emit(_M_QUARANTINED, 1.0)

        self._logger.info(
            "eis_gate_complete",
            action=action.value,
            composite_score=round(composite, 4),
            latency_us=latency_us,
        )


# ─── Module-level helpers ────────────────────────────────────────────────────


def _elapsed_us(t0_ns: int) -> int:
    """Nanosecond perf_counter start → microseconds elapsed."""
    return (time.perf_counter_ns() - t0_ns) // 1_000


def _dominant_threat_class(annotations: list[ThreatAnnotation]) -> ThreatClass:
    """Pick the most severe threat class from a list of annotations."""
    if not annotations:
        return ThreatClass.BENIGN

    severity_rank = {
        ThreatSeverity.CRITICAL: 4,
        ThreatSeverity.HIGH: 3,
        ThreatSeverity.MEDIUM: 2,
        ThreatSeverity.LOW: 1,
        ThreatSeverity.NONE: 0,
    }
    best = max(annotations, key=lambda a: severity_rank.get(a.severity, 0))
    return best.threat_class if best.threat_class != ThreatClass.BENIGN else ThreatClass.BENIGN


def _anomaly_to_severity(score: float) -> ThreatSeverity:
    """Map a 0.0-1.0 structural anomaly score to a ThreatSeverity."""
    if score >= 0.8:
        return ThreatSeverity.HIGH
    if score >= 0.5:
        return ThreatSeverity.MEDIUM
    if score >= 0.3:
        return ThreatSeverity.LOW
    return ThreatSeverity.NONE
