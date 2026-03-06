"""
EcodiaOS — IIEP Ingestion Pipeline

The receive-side pipeline for inbound federated knowledge.  Every payload
arriving through IIEP passes through this pipeline before being integrated
into any local module.  The pipeline enforces the core safety invariant:

    An instance should never blindly trust knowledge from a peer,
    even a federated one.

Pipeline stages:
  1. Deduplication    — reject payloads we have already seen (by content_hash)
  2. Provenance check — reject payloads that originated from us (loop)
  3. EIS taint analysis — run innate threat checks on the content
  4. Equor governance  — constitutional review of the integration intent
  5. Routing          — dispatch accepted payloads to the target module
  6. Receipt          — build per-payload verdict receipt for the sender

Each payload receives an independent verdict: ACCEPTED, QUARANTINED,
REJECTED, or DEFERRED.  The pipeline never throws — every payload gets
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
    ) -> None:
        self._instance_id = instance_id
        self._equor = equor
        self._eis = eis
        self._evo = evo
        self._simula = simula
        self._oikos = oikos
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

        return receipt

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

        # Stage 3: Trust gate — does this link's trust permit this kind?
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
            # Routing failed but the payload is safe — defer for retry
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
            # No EIS wired — conservative pass (log warning)
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
            # No Equor wired — conservative pass
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

        The hypothesis is not automatically integrated — Evo receives it
        as a PROPOSED hypothesis from a federated source, which must still
        accumulate local evidence to be SUPPORTED.
        """
        if self._evo is None:
            return False

        ingest_fn = getattr(self._evo, "ingest_federated_hypothesis", None)
        if ingest_fn is None:
            # Evo doesn't support federated hypothesis ingestion yet —
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

        The procedure is stored as a candidate — it will only be
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

        The pattern is NOT applied directly — Simula receives it as
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
        trigger any financial actions — those still require local
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
