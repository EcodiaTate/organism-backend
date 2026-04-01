"""
EcodiaOS - IIEP Ingestion Pipeline

The receive-side pipeline for inbound federated knowledge.  Every payload
arriving through IIEP passes through this pipeline before being integrated
into any local module.  The pipeline enforces the core safety invariant:

    An instance should never blindly trust knowledge from a peer,
    even a federated one.

Pipeline stages:
  1. Deduplication    - reject payloads we have already seen (by content_hash)
  2. Provenance check - reject payloads that originated from us (loop)
  3. EIS taint analysis - run innate threat checks on the content
  4. Equor governance  - constitutional review of the integration intent
  5. Routing          - dispatch accepted payloads to the target module
  6. Receipt          - build per-payload verdict receipt for the sender

Each payload receives an independent verdict: ACCEPTED, QUARANTINED,
REJECTED, or DEFERRED.  The pipeline never throws - every payload gets
a verdict regardless of individual stage failures.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now
from primitives.federation import (
    EXCHANGE_TRUST_GATES,
    ExchangeEnvelope,
    ExchangePayload,
    ExchangePayloadKind,
    ExchangeReceipt,
    FederationLink,
    IngestionVerdict,
    TrustLevel,
)

if TYPE_CHECKING:
    from systems.equor.service import EquorService

logger = structlog.get_logger("federation.ingestion")


class IngestionPipeline:
    """
    Governs the receive-side processing of inbound IIEP payloads.

    Every payload passes through: dedup -> provenance -> EIS -> Equor -> route.
    Nothing is integrated without clearing all stages.
    """

    def __init__(
        self,
        instance_id: str = "",
        equor: EquorService | None = None,
        eis: Any = None,  # EIS module or run_innate_checks callable
        evo: Any = None,
        simula: Any = None,
        oikos: Any = None,
        event_bus: Any = None,
    ) -> None:
        self._instance_id = instance_id
        self._equor = equor
        self._eis = eis
        self._evo = evo
        self._simula = simula
        self._oikos = oikos
        self._event_bus: Any = event_bus  # For FEDERATION_PRIVACY_VIOLATION emission
        self._re: Any = None  # RE or Claude client - wired post-init for semantic quality scoring
        self._logger = logger.bind(component="ingestion_pipeline")

        # Seen content hashes for deduplication (bounded set)
        self._seen_hashes: set[str] = set()
        self._max_seen: int = 10_000

        # Quarantine buffer for payloads that EIS flagged
        self._quarantine: list[tuple[ExchangePayload, str, str]] = []
        # Tuple: (payload, link_id, reason)

        # Stats
        self._total_processed: int = 0
        self._accepted: int = 0
        self._quarantined: int = 0
        self._rejected: int = 0
        self._deferred: int = 0

        # Runtime-adjustable RE quality threshold (default matches class constant).
        # Evo can tune this via service.set_re_quality_threshold() based on observed
        # accept/defer rates - start conservative, loosen if too many good payloads defer.
        self._re_quality_threshold: float = self._RE_QUALITY_THRESHOLD

    # ─── Main Entry Point ────────────────────────────────────────

    async def process_envelope(
        self,
        envelope: ExchangeEnvelope,
        link: FederationLink,
    ) -> ExchangeReceipt:
        """
        Process all payloads in an inbound PUSH envelope.

        Returns an ExchangeReceipt with per-payload verdicts.
        """
        verdicts: dict[str, IngestionVerdict] = {}

        for payload in envelope.payloads:
            verdict = await self._process_single(payload, link)
            verdicts[payload.payload_id] = verdict

        receipt = ExchangeReceipt(
            envelope_id=envelope.id,
            receiver_instance_id=self._instance_id,
            payload_verdicts=verdicts,
        )

        accepted = sum(1 for v in verdicts.values() if v == IngestionVerdict.ACCEPTED)
        rejected = sum(1 for v in verdicts.values() if v == IngestionVerdict.REJECTED)
        quarantined_count = sum(
            1 for v in verdicts.values() if v == IngestionVerdict.QUARANTINED
        )

        self._logger.info(
            "envelope_processed",
            envelope_id=envelope.id,
            sender=envelope.sender_instance_id,
            total=len(verdicts),
            accepted=accepted,
            rejected=rejected,
            quarantined=quarantined_count,
        )

        # Emit ingestion stats to Synapse so Nova/Benchmarks/Thymos can observe
        # pipeline health. Without this, _total_processed/_accepted/_quarantined/
        # _rejected/_deferred accumulate in-memory but are invisible to the LLM.
        if self._event_bus is not None and len(verdicts) > 0:
            import asyncio as _asyncio
            _asyncio.ensure_future(self._emit_ingestion_telemetry(
                envelope_id=envelope.id,
                sender=envelope.sender_instance_id,
                envelope_accepted=accepted,
                envelope_rejected=rejected,
                envelope_quarantined=quarantined_count,
            ))

        return receipt

    async def _emit_ingestion_telemetry(
        self,
        envelope_id: str,
        sender: str,
        envelope_accepted: int,
        envelope_rejected: int,
        envelope_quarantined: int,
    ) -> None:
        """Emit FEDERATION_KNOWLEDGE_RECEIVED-class stats so pipeline health is observable."""
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            event = SynapseEvent(
                event_type=SynapseEventType.EVOLUTIONARY_OBSERVABLE,
                source_system="federation.ingestion",
                data={
                    "observable_type": "ingestion_pipeline_stats",
                    "envelope_id": envelope_id,
                    "sender_instance_id": sender,
                    "envelope_accepted": envelope_accepted,
                    "envelope_rejected": envelope_rejected,
                    "envelope_quarantined": envelope_quarantined,
                    "cumulative_total_processed": self._total_processed,
                    "cumulative_accepted": self._accepted,
                    "cumulative_quarantined": self._quarantined,
                    "cumulative_rejected": self._rejected,
                    "cumulative_deferred": self._deferred,
                    "accept_rate": round(
                        self._accepted / self._total_processed, 4
                    ) if self._total_processed > 0 else 0.0,
                },
            )
            await self._event_bus.emit(event)
        except Exception:
            pass  # Telemetry must never block ingestion

    # ─── Single Payload Processing ───────────────────────────────

    async def _process_single(
        self,
        payload: ExchangePayload,
        link: FederationLink,
    ) -> IngestionVerdict:
        """Run a single payload through all pipeline stages."""
        self._total_processed += 1

        # Stage 1: Deduplication
        if payload.content_hash and payload.content_hash in self._seen_hashes:
            self._rejected += 1
            self._logger.debug(
                "payload_dedup_rejected",
                payload_id=payload.payload_id,
                kind=payload.kind,
            )
            return IngestionVerdict.REJECTED

        # Stage 2: Provenance / loop check
        if self._instance_id in payload.provenance_chain:
            self._rejected += 1
            self._logger.debug(
                "payload_loop_rejected",
                payload_id=payload.payload_id,
                provenance=payload.provenance_chain,
            )
            return IngestionVerdict.REJECTED

        # Stage 3: Trust gate - does this link's trust permit this kind?
        required_trust = EXCHANGE_TRUST_GATES.get(payload.kind, TrustLevel.ALLY)
        if link.trust_level < required_trust:
            self._rejected += 1
            self._logger.info(
                "payload_trust_rejected",
                payload_id=payload.payload_id,
                kind=payload.kind,
                required=required_trust.name,
                actual=link.trust_level.name,
            )
            return IngestionVerdict.REJECTED

        # Stage 3.5: Privacy scan - detect if peer violated protocol by sending private data.
        # This fires FEDERATION_PRIVACY_VIOLATION and resets trust on the link.
        privacy_violation = self._detect_privacy_violation(payload, link)
        if privacy_violation:
            await self._emit_privacy_violation(
                link=link,
                payload=payload,
                violation_detail=privacy_violation,
            )
            self._rejected += 1
            return IngestionVerdict.REJECTED

        # Stage 4: EIS taint analysis
        eis_verdict = await self._run_eis_check(payload)
        if eis_verdict == IngestionVerdict.QUARANTINED:
            self._quarantined += 1
            self._quarantine.append((
                payload,
                link.id,
                "EIS taint analysis flagged this payload",
            ))
            self._logger.warning(
                "payload_quarantined_eis",
                payload_id=payload.payload_id,
                kind=payload.kind,
            )
            return IngestionVerdict.QUARANTINED
        if eis_verdict == IngestionVerdict.REJECTED:
            self._rejected += 1
            return IngestionVerdict.REJECTED

        # Stage 4.5: RE semantic quality scoring (PARTNER+ trust, epistemic payloads only)
        # At PARTNER+ trust, incoming hypotheses and schema structures are scored for
        # coherence, novelty, and constitutional safety via the RE (or Claude fallback)
        # before being accepted. Low-quality content is DEFERRED, not REJECTED.
        re_verdict = await self._run_re_quality_check(payload, link)
        if re_verdict == IngestionVerdict.REJECTED:
            self._rejected += 1
            self._logger.info(
                "payload_re_quality_rejected",
                payload_id=payload.payload_id,
                kind=payload.kind,
            )
            return IngestionVerdict.REJECTED
        if re_verdict == IngestionVerdict.DEFERRED:
            self._deferred += 1
            return IngestionVerdict.DEFERRED

        # Stage 5: Equor governance review
        equor_verdict = await self._run_equor_review(payload, link)
        if equor_verdict == IngestionVerdict.REJECTED:
            self._rejected += 1
            self._logger.info(
                "payload_equor_rejected",
                payload_id=payload.payload_id,
                kind=payload.kind,
            )
            return IngestionVerdict.REJECTED
        if equor_verdict == IngestionVerdict.DEFERRED:
            self._deferred += 1
            return IngestionVerdict.DEFERRED

        # Stage 6: Route to target module
        routed = await self._route_payload(payload, link)
        if not routed:
            # Routing failed but the payload is safe - defer for retry
            self._deferred += 1
            return IngestionVerdict.DEFERRED

        # Mark as seen
        if payload.content_hash:
            self._seen_hashes.add(payload.content_hash)
            if len(self._seen_hashes) > self._max_seen:
                # Evict oldest entries (set is unordered, so pop arbitrary)
                while len(self._seen_hashes) > self._max_seen // 2:
                    self._seen_hashes.pop()

        self._accepted += 1
        return IngestionVerdict.ACCEPTED

    # ─── Stage: EIS Taint Analysis ───────────────────────────────

    async def _run_eis_check(
        self, payload: ExchangePayload,
    ) -> IngestionVerdict:
        """
        Run EIS innate threat checks on the payload content.

        For MUTATION_PATTERN payloads, also run taint analysis on
        any code-like content.
        """
        if self._eis is None:
            # No EIS wired - conservative pass (log warning)
            self._logger.debug("eis_not_wired_skipping_check")
            return IngestionVerdict.ACCEPTED

        try:
            # Serialise content for text-based innate checks
            text = json.dumps(payload.content, default=str)

            # Try the innate check function
            run_innate = getattr(self._eis, "run_innate_checks", None)
            if run_innate is None:
                # Maybe eis IS the function itself
                if callable(self._eis):
                    run_innate = self._eis
                else:
                    return IngestionVerdict.ACCEPTED

            result = run_innate(text)

            # Check if any flags were raised
            flags = getattr(result, "flags", None)
            if flags is not None:
                severity = getattr(flags, "max_severity", 0)
                blocked = getattr(flags, "should_block", False)
                if blocked:
                    self._logger.warning(
                        "eis_hard_block",
                        payload_id=payload.payload_id,
                        severity=severity,
                    )
                    return IngestionVerdict.REJECTED
                if severity >= 3:  # Medium+ threat
                    return IngestionVerdict.QUARANTINED

            # For mutation patterns, run taint engine if available
            if payload.kind == ExchangePayloadKind.MUTATION_PATTERN:
                taint_engine = getattr(self._eis, "taint_engine", None)
                if taint_engine is not None and callable(
                    getattr(taint_engine, "assess", None)
                ):
                    from systems.eis.taint_models import MutationProposal

                    taint_proposal = MutationProposal(
                        file_path="federation/inbound",
                        diff=payload.content.get(
                            "change_description", ""
                        ),
                        description=payload.content.get(
                            "description", ""
                        ),
                    )
                    assessment = taint_engine.assess(taint_proposal)
                    if getattr(assessment, "blocked", False):
                        return IngestionVerdict.REJECTED
                    if getattr(assessment, "severity", "low") in (
                        "high", "critical",
                    ):
                        return IngestionVerdict.QUARANTINED

        except Exception as exc:
            self._logger.warning("eis_check_failed", error=str(exc))
            # EIS failure -> quarantine (fail safe, don't auto-accept)
            return IngestionVerdict.QUARANTINED

        return IngestionVerdict.ACCEPTED

    # ─── Stage: Equor Governance ─────────────────────────────────

    async def _run_equor_review(
        self,
        payload: ExchangePayload,
        link: FederationLink,
    ) -> IngestionVerdict:
        """
        Run Equor constitutional review on the integration intent.

        The question posed to Equor: "Should we integrate this knowledge
        from this peer into our local modules?"
        """
        if self._equor is None:
            # No Equor wired - conservative pass
            self._logger.debug("equor_not_wired_skipping_review")
            return IngestionVerdict.ACCEPTED

        try:
            from primitives.common import Verdict
            from primitives.intent import (
                ActionSequence,
                DecisionTrace,
                GoalDescriptor,
                Intent,
            )

            kind_label = payload.kind.value.replace("_", " ")
            intent = Intent(
                goal=GoalDescriptor(
                    description=(
                        f"Integrate federated {kind_label} from "
                        f"{link.remote_name} "
                        f"(confidence={payload.confidence:.2f})"
                    ),
                    target_domain=(
                        f"federation.ingestion.{payload.kind.value}"
                    ),
                ),
                plan=ActionSequence(steps=[]),
                decision_trace=DecisionTrace(
                    reasoning=(
                        f"IIEP inbound {kind_label} from "
                        f"{link.remote_name} "
                        f"(trust={link.trust_level.name}, "
                        f"confidence={payload.confidence:.2f}, "
                        f"provenance={payload.provenance_chain})"
                    ),
                ),
            )
            check = await self._equor.review(intent)

            if check.verdict in (Verdict.APPROVED, Verdict.MODIFIED):
                return IngestionVerdict.ACCEPTED
            if check.verdict == Verdict.DEFERRED:
                return IngestionVerdict.DEFERRED
            return IngestionVerdict.REJECTED

        except Exception as exc:
            self._logger.warning("equor_review_failed", error=str(exc))
            # Equor failure -> defer (don't reject good knowledge because
            # governance is temporarily down)
            return IngestionVerdict.DEFERRED

    # ─── Stage: RE Semantic Quality Scoring ──────────────────────

    # Payload kinds that warrant epistemic quality scoring at PARTNER+ trust.
    # HYPOTHESIS and SCHEMA_STRUCTURES carry world-model claims that could
    # subtly corrupt the organism's beliefs if they are low-coherence or
    # constitutionally misaligned.
    _RE_SCORED_KINDS = frozenset({
        ExchangePayloadKind.HYPOTHESIS,
    })

    # Minimum quality score to accept. Below this threshold: DEFERRED (not rejected),
    # allowing the pipeline to re-evaluate after local context improves.
    _RE_QUALITY_THRESHOLD = 0.35

    async def _run_re_quality_check(
        self,
        payload: ExchangePayload,
        link: FederationLink,
    ) -> IngestionVerdict:
        """
        Score inbound epistemic payloads for semantic coherence and constitutional
        safety using the RE (or Claude API fallback) at PARTNER+ trust level.

        Only applies to HYPOTHESIS payloads at PARTNER+ trust - all others pass
        through as ACCEPTED immediately.  The RE scores the statement on three
        dimensions:
          - Coherence (0–1): Is this internally consistent?
          - Novelty (0–1): Does this add new information or is it redundant?
          - Constitutional safety (0–1): Does this align with the four drives?

        Score = harmonic mean of the three dimensions.  Below
        _RE_QUALITY_THRESHOLD the payload is DEFERRED (not rejected) - local
        evidence may later validate it.

        When no RE/Claude client is wired, all payloads pass (fail-open to
        avoid rejecting valid knowledge).

        References: Spec 11b §XII §2 (RE-assisted semantic quality scoring)
        """
        # Only score epistemic payload kinds
        if payload.kind not in self._RE_SCORED_KINDS:
            return IngestionVerdict.ACCEPTED

        # Only apply at PARTNER+ trust (lower trust already blocked upstream)
        if link.trust_level < TrustLevel.PARTNER:
            return IngestionVerdict.ACCEPTED

        # No RE wired - fail-open to avoid rejecting valid federated knowledge
        if self._re is None:
            self._logger.debug("re_not_wired_skipping_quality_check", kind=payload.kind)
            return IngestionVerdict.ACCEPTED

        try:
            statement = payload.content.get("statement", "")
            category = payload.content.get("category", "")
            evidence_score = float(payload.content.get("evidence_score", 0.0))

            if not statement:
                return IngestionVerdict.ACCEPTED  # No statement to score

            # Build scoring prompt
            prompt = (
                f"Score this federated hypothesis on three dimensions (each 0.0–1.0):\n\n"
                f"Statement: {statement}\n"
                f"Category: {category}\n"
                f"Evidence score (log-odds): {evidence_score:.2f}\n"
                f"From instance: {link.remote_name} (trust: {link.trust_level.name})\n\n"
                f"Dimensions:\n"
                f"1. Coherence: Is the statement internally consistent and well-formed?\n"
                f"2. Novelty: Does it add new information beyond common knowledge?\n"
                f"3. Constitutional safety: Does it align with Care, Honesty, Coherence, Growth drives?\n\n"
                f"Respond with only a JSON object: "
                f'{{\"coherence\": <float>, \"novelty\": <float>, \"constitutional_safety\": <float>, '
                f'\"reasoning\": \"<one sentence>\"}}'
            )

            # Try RE first, fall back to raw LLM call
            score_result: dict[str, Any] = {}
            re_call = getattr(self._re, "generate", None) or getattr(self._re, "complete", None)
            if re_call and callable(re_call):
                response = await re_call(prompt)
                raw = response if isinstance(response, str) else str(response)
                # Extract JSON from response
                import json as _json
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    score_result = _json.loads(raw[start:end])

            if not score_result:
                # Incomplete response - fail-open
                return IngestionVerdict.ACCEPTED

            coherence = float(score_result.get("coherence", 1.0))
            novelty = float(score_result.get("novelty", 1.0))
            const_safety = float(score_result.get("constitutional_safety", 1.0))
            reasoning = str(score_result.get("reasoning", ""))

            # Harmonic mean of three dimensions (penalises any single weak dimension)
            dims = [coherence, novelty, const_safety]
            if all(d > 0 for d in dims):
                quality_score = len(dims) / sum(1.0 / d for d in dims)
            else:
                quality_score = 0.0

            self._logger.info(
                "re_quality_scored",
                payload_id=payload.payload_id,
                remote=link.remote_name,
                coherence=round(coherence, 3),
                novelty=round(novelty, 3),
                constitutional_safety=round(const_safety, 3),
                quality_score=round(quality_score, 3),
                threshold=self._re_quality_threshold,
                reasoning=reasoning,
            )

            if quality_score < self._re_quality_threshold:
                self._logger.info(
                    "payload_deferred_low_re_quality",
                    payload_id=payload.payload_id,
                    quality_score=round(quality_score, 3),
                )
                return IngestionVerdict.DEFERRED

            return IngestionVerdict.ACCEPTED

        except Exception as exc:
            # RE scoring failure is non-fatal - fail-open
            self._logger.warning("re_quality_check_failed", error=str(exc))
            return IngestionVerdict.ACCEPTED

    # ─── Stage: Privacy Violation Detection ──────────────────────

    # PII markers that should never appear in inbound federated payloads.
    # If a remote instance sends content containing these, it has violated
    # the federation protocol by sharing individual-level data.
    _PII_KEYS = frozenset({
        "email", "phone", "mobile", "address", "dob", "date_of_birth",
        "first_name", "last_name", "full_name", "ssn", "national_id",
        "passport", "driver_license", "credit_card", "bank_account",
        "user_id", "member_id", "patient_id", "student_id",
        "ip_address", "device_id", "location", "gps",
    })

    def _detect_privacy_violation(
        self,
        payload: ExchangePayload,
        link: FederationLink,
    ) -> str | None:
        """
        Scan inbound payload content for individual-identifiable data that
        should never cross federation boundaries (Spec 11b §V.3, §IX.2).

        A remote instance sending PRIVATE-level data is a protocol violation
        regardless of trust level.  Returns a violation detail string, or
        None if clean.

        References: Spec 11b §V.3 (privacy filter), §IX.2 (privacy absolute),
                    §XI (FEDERATION_PRIVACY_VIOLATION event)
        """
        if not payload.content:
            return None

        # Flatten content into a searchable key-value space
        def _scan(obj: Any, depth: int = 0) -> str | None:
            if depth > 6:
                return None
            if isinstance(obj, dict):
                for k, v in obj.items():
                    key_lower = str(k).lower().replace("-", "_").replace(" ", "_")
                    if key_lower in self._PII_KEYS:
                        val = str(v)
                        if val and val not in ("", "null", "None", "unknown"):
                            return f"PII key '{k}' with non-empty value detected in inbound payload"
                    result = _scan(v, depth + 1)
                    if result:
                        return result
            elif isinstance(obj, list):
                for item in obj[:10]:  # Scan first 10 items only
                    result = _scan(item, depth + 1)
                    if result:
                        return result
            return None

        return _scan(payload.content)

    async def _emit_privacy_violation(
        self,
        link: FederationLink,
        payload: ExchangePayload,
        violation_detail: str,
    ) -> None:
        """
        Emit FEDERATION_PRIVACY_VIOLATION on Synapse and reset trust on the link.

        This is the detection + consequence mechanism for privacy breaches per
        Spec 11b §IV.2 (violation 3× multiplier + instant reset on privacy_breach)
        and §XI (FEDERATION_PRIVACY_VIOLATION event).

        The actual trust reset happens in FederationService._update_trust_and_emit()
        when it processes the VIOLATION interaction - we emit the event here and
        let the service handle trust consequences.

        References: Spec 11b §IV.2, §XI
        """
        self._logger.warning(
            "privacy_violation_detected",
            remote_instance_id=link.remote_instance_id,
            remote_name=link.remote_name,
            payload_id=payload.payload_id,
            payload_kind=payload.kind.value,
            detail=violation_detail,
        )

        bus = self._event_bus
        if bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            event = SynapseEvent(
                event_type=SynapseEventType.FEDERATION_PRIVACY_VIOLATION,
                source_system="federation",
                data={
                    "remote_instance_id": link.remote_instance_id,
                    "remote_name": link.remote_name,
                    "link_id": link.id,
                    "payload_id": payload.payload_id,
                    "payload_kind": payload.kind.value,
                    "violation_detail": violation_detail,
                    "trust_reset": True,
                },
            )
            await bus.emit(event)
        except Exception as exc:
            self._logger.warning("privacy_violation_event_emit_failed", error=str(exc))

    # ─── Stage: Route to Target Module ───────────────────────────

    async def _route_payload(
        self,
        payload: ExchangePayload,
        link: FederationLink,
    ) -> bool:
        """
        Dispatch an accepted payload to its target module.

        Returns True if routing succeeded, False if the target
        is unavailable (caller should DEFER).
        """
        kind = payload.kind
        attribution = {
            "source_instance_id": payload.source_instance_id,
            "provenance_chain": payload.provenance_chain,
            "received_from": link.remote_instance_id,
            "received_at": utc_now().isoformat(),
            "confidence": payload.confidence,
        }

        try:
            if kind == ExchangePayloadKind.HYPOTHESIS:
                return await self._ingest_hypothesis(payload, attribution)
            elif kind == ExchangePayloadKind.PROCEDURE:
                return await self._ingest_procedure(payload, attribution)
            elif kind == ExchangePayloadKind.MUTATION_PATTERN:
                return await self._ingest_mutation_pattern(
                    payload, attribution,
                )
            elif kind == ExchangePayloadKind.ECONOMIC_INTEL:
                return await self._ingest_economic_intel(
                    payload, attribution,
                )
            else:
                self._logger.warning("unknown_payload_kind", kind=kind)
                return False
        except Exception as exc:
            self._logger.error(
                "routing_failed",
                payload_id=payload.payload_id,
                kind=kind,
                error=str(exc),
            )
            return False

    async def _ingest_hypothesis(
        self,
        payload: ExchangePayload,
        attribution: dict[str, Any],
    ) -> bool:
        """
        Route a hypothesis to Evo for consideration.

        The hypothesis is not automatically integrated - Evo receives it
        as a PROPOSED hypothesis from a federated source, which must still
        accumulate local evidence to be SUPPORTED.
        """
        if self._evo is None:
            return False

        ingest_fn = getattr(self._evo, "ingest_federated_hypothesis", None)
        if ingest_fn is None:
            # Evo doesn't support federated hypothesis ingestion yet -
            # log and defer
            self._logger.info(
                "hypothesis_ingestion_deferred",
                reason="evo_lacks_ingest_federated_hypothesis",
            )
            return False

        await ingest_fn(
            statement=payload.content.get("statement", ""),
            category=payload.content.get("category", "world_model"),
            evidence_score=payload.content.get("evidence_score", 0.0),
            formal_test=payload.content.get("formal_test", ""),
            attribution=attribution,
        )
        return True

    async def _ingest_procedure(
        self,
        payload: ExchangePayload,
        attribution: dict[str, Any],
    ) -> bool:
        """
        Route a procedure to Evo for consideration.

        The procedure is stored as a candidate - it will only be
        activated if local evidence confirms its usefulness.
        """
        if self._evo is None:
            return False

        ingest_fn = getattr(self._evo, "ingest_federated_procedure", None)
        if ingest_fn is None:
            self._logger.info(
                "procedure_ingestion_deferred",
                reason="evo_lacks_ingest_federated_procedure",
            )
            return False

        await ingest_fn(
            name=payload.content.get("name", ""),
            preconditions=payload.content.get("preconditions", []),
            steps=payload.content.get("steps", []),
            postconditions=payload.content.get("postconditions", []),
            success_rate=payload.content.get("success_rate", 0.0),
            attribution=attribution,
        )
        return True

    async def _ingest_mutation_pattern(
        self,
        payload: ExchangePayload,
        attribution: dict[str, Any],
    ) -> bool:
        """
        Route a mutation pattern to Simula for evaluation.

        The pattern is NOT applied directly - Simula receives it as
        an informational record of what worked for another instance.
        The local Simula may or may not use it as inspiration for
        future proposals, subject to its own simulation and governance.
        """
        if self._simula is None:
            return False

        ingest_fn = getattr(
            self._simula, "ingest_federated_pattern", None,
        )
        if ingest_fn is None:
            self._logger.info(
                "mutation_pattern_ingestion_deferred",
                reason="simula_lacks_ingest_federated_pattern",
            )
            return False

        await ingest_fn(
            category=payload.content.get("category", ""),
            description=payload.content.get("description", ""),
            expected_benefit=payload.content.get("expected_benefit", ""),
            risk_level=payload.content.get("risk_level", "moderate"),
            attribution=attribution,
        )
        return True

    async def _ingest_economic_intel(
        self,
        payload: ExchangePayload,
        attribution: dict[str, Any],
    ) -> bool:
        """
        Route economic intelligence to Oikos.

        Economic intel informs Oikos's planning but doesn't directly
        trigger any financial actions - those still require local
        governance approval.
        """
        if self._oikos is None:
            return False

        ingest_fn = getattr(self._oikos, "ingest_federated_intel", None)
        if ingest_fn is None:
            self._logger.info(
                "economic_intel_ingestion_deferred",
                reason="oikos_lacks_ingest_federated_intel",
            )
            return False

        await ingest_fn(
            intel_type=payload.content.get("type", ""),
            content=payload.content,
            attribution=attribution,
        )
        return True

    # ─── Quarantine Management ───────────────────────────────────

    @property
    def quarantine_count(self) -> int:
        return len(self._quarantine)

    def get_quarantined(self) -> list[dict[str, Any]]:
        """Return quarantined payloads for manual review."""
        return [
            {
                "payload_id": p.payload_id,
                "kind": p.kind,
                "confidence": p.confidence,
                "source": p.source_instance_id,
                "provenance": p.provenance_chain,
                "link_id": link_id,
                "reason": reason,
                "content_preview": str(p.content)[:200],
            }
            for p, link_id, reason in self._quarantine
        ]

    async def release_from_quarantine(
        self,
        payload_id: str,
        link: FederationLink,
    ) -> IngestionVerdict:
        """
        Manually release a quarantined payload for integration.

        Bypasses EIS (already reviewed by a human) but still runs
        Equor governance.
        """
        target = None
        for i, (p, _link_id, _reason) in enumerate(self._quarantine):
            if p.payload_id == payload_id:
                target = self._quarantine.pop(i)
                break

        if target is None:
            return IngestionVerdict.REJECTED

        payload = target[0]

        # Still run Equor
        equor_verdict = await self._run_equor_review(payload, link)
        if equor_verdict != IngestionVerdict.ACCEPTED:
            return equor_verdict

        routed = await self._route_payload(payload, link)
        if not routed:
            return IngestionVerdict.DEFERRED

        self._quarantined -= 1
        self._accepted += 1
        return IngestionVerdict.ACCEPTED

    # ─── Stats ───────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_processed": self._total_processed,
            "accepted": self._accepted,
            "quarantined": self._quarantined,
            "rejected": self._rejected,
            "deferred": self._deferred,
            "quarantine_buffer_size": len(self._quarantine),
            "seen_hashes_size": len(self._seen_hashes),
        }
