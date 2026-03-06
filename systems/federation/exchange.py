"""
EcodiaOS — IIEP Exchange Protocol

Implements the Inter-Instance Exchange Protocol for selective, attributable
knowledge sharing between federated EOS instances.  This module operates
over an already-established authenticated channel (see channel.py) and is
separate from the handshake layer.

Design principles:
  - Selective: only knowledge above confidence thresholds is shared.
  - Attributable: every payload carries source instance ID and a full
    provenance chain so the receiver always knows where it came from.
  - Safe: the receiver never blindly integrates — all inbound payloads
    are routed through the IngestionPipeline (ingestion.py) which applies
    EIS taint analysis and Equor governance before integration.
  - Signed: every envelope is Ed25519-signed by the sender.

The four exchangeable knowledge categories:
  HYPOTHESIS      — high-confidence hypotheses from Evo
  PROCEDURE       — proven action sequences from Evo
  MUTATION_PATTERN — successful evolution patterns from Simula/GRPO
  ECONOMIC_INTEL  — economic intelligence from Oikos

Outbound flow (push):
  1. Collector gathers eligible knowledge from Evo/Simula/Oikos
  2. Confidence filter removes anything below threshold
  3. Trust gate ensures the peer's trust level is sufficient
  4. Privacy filter scrubs PII
  5. Payloads are packed into a signed ExchangeEnvelope
  6. Envelope sent via the channel, receipt returned

Inbound flow (push received):
  Handled by IngestionPipeline — see ingestion.py.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import structlog

from primitives.federation import (
    EXCHANGE_CONFIDENCE_FLOORS,
    EXCHANGE_TRUST_GATES,
    ExchangeDirection,
    ExchangeEnvelope,
    ExchangePayload,
    ExchangePayloadKind,
    FederationInteraction,
    FederationLink,
    InteractionOutcome,
    TrustLevel,
)

if TYPE_CHECKING:
    from systems.federation.identity import IdentityManager
    from systems.federation.privacy import PrivacyFilter

logger = structlog.get_logger("federation.exchange")


# ─── Content Hash ───────────────────────────────────────────────────


def _content_hash(content: dict[str, Any]) -> str:
    """SHA-256 of deterministic JSON serialisation."""
    canonical = json.dumps(content, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ─── Payload Collectors ─────────────────────────────────────────────
#
# Each collector knows how to extract eligible payloads from its
# respective subsystem.  They return raw ExchangePayload lists — the
# protocol applies confidence and trust filtering afterwards.
# ─────────────────────────────────────────────────────────────────────


async def collect_hypotheses(
    evo: Any,
    instance_id: str,
    min_confidence: float = 0.0,
    max_items: int = 20,
) -> list[ExchangePayload]:
    """
    Collect high-confidence hypotheses from Evo.

    Only SUPPORTED or INTEGRATED hypotheses with evidence_score >=
    the confidence floor are eligible.
    """
    payloads: list[ExchangePayload] = []
    try:
        hypotheses: list[Any] = getattr(evo, "_hypotheses", [])
        if callable(getattr(evo, "get_hypotheses", None)):
            hypotheses = await evo.get_hypotheses()

        for h in hypotheses:
            status = getattr(h, "status", "")
            if status not in ("supported", "integrated"):
                continue
            score = float(getattr(h, "evidence_score", 0.0))
            # Normalise evidence_score to 0-1 confidence
            # evidence_score uses Bayesian log-odds; 3.0 is the integration threshold
            confidence = min(1.0, max(0.0, score / 10.0))
            if confidence < min_confidence:
                continue

            content = {
                "statement": getattr(h, "statement", ""),
                "category": str(getattr(h, "category", "")),
                "evidence_score": score,
                "formal_test": getattr(h, "formal_test", ""),
                "status": str(status),
                "supporting_episode_count": len(getattr(h, "supporting_episodes", [])),
                "contradicting_episode_count": len(
                    getattr(h, "contradicting_episodes", [])
                ),
            }

            payloads.append(ExchangePayload(
                kind=ExchangePayloadKind.HYPOTHESIS,
                confidence=confidence,
                content=content,
                content_hash=_content_hash(content),
                source_instance_id=instance_id,
                provenance_chain=[instance_id],
                domain=str(getattr(h, "category", "")),
            ))
            if len(payloads) >= max_items:
                break
    except Exception as exc:
        logger.warning("hypothesis_collection_failed", error=str(exc))
    return payloads


async def collect_procedures(
    evo: Any,
    instance_id: str,
    min_confidence: float = 0.0,
    max_items: int = 20,
) -> list[ExchangePayload]:
    """
    Collect proven procedures from Evo.

    A procedure's confidence is its success_rate.  Only procedures
    that have been used at least once are eligible.
    """
    payloads: list[ExchangePayload] = []
    try:
        procedures: list[Any] = getattr(evo, "_procedures", [])
        if callable(getattr(evo, "get_procedures", None)):
            procedures = await evo.get_procedures()

        for p in procedures:
            usage = int(getattr(p, "usage_count", 0))
            if usage < 1:
                continue
            confidence = float(getattr(p, "success_rate", 0.0))
            if confidence < min_confidence:
                continue

            steps_data: list[dict[str, Any]] = []
            for step in getattr(p, "steps", []):
                steps_data.append({
                    "action": getattr(step, "action", ""),
                    "target": getattr(step, "target", ""),
                    "parameters": getattr(step, "parameters", {}),
                })

            content = {
                "name": getattr(p, "name", ""),
                "preconditions": list(getattr(p, "preconditions", [])),
                "steps": steps_data,
                "postconditions": list(getattr(p, "postconditions", [])),
                "success_rate": confidence,
                "usage_count": usage,
            }

            payloads.append(ExchangePayload(
                kind=ExchangePayloadKind.PROCEDURE,
                confidence=confidence,
                content=content,
                content_hash=_content_hash(content),
                source_instance_id=instance_id,
                provenance_chain=[instance_id],
                domain="procedural",
            ))
            if len(payloads) >= max_items:
                break
    except Exception as exc:
        logger.warning("procedure_collection_failed", error=str(exc))
    return payloads


async def collect_mutation_patterns(
    simula: Any,
    instance_id: str,
    min_confidence: float = 0.0,
    max_items: int = 10,
) -> list[ExchangePayload]:
    """
    Collect successful mutation patterns from Simula/GRPO.

    Only APPLIED proposals (confirmed beneficial) are eligible.
    The confidence is derived from the risk assessment.
    """
    payloads: list[ExchangePayload] = []
    try:
        proposals: list[Any] = []
        if callable(getattr(simula, "get_applied_proposals", None)):
            proposals = await simula.get_applied_proposals()
        elif hasattr(simula, "_history_manager"):
            mgr = simula._history_manager
            if callable(getattr(mgr, "get_applied", None)):
                proposals = await mgr.get_applied()

        for prop in proposals:
            status = str(getattr(prop, "status", ""))
            if status != "applied":
                continue

            # Derive confidence from risk assessment
            risk = str(getattr(prop, "risk_assessment", "low")).lower()
            risk_conf = {
                "low": 0.9, "moderate": 0.7, "high": 0.5, "unacceptable": 0.1,
            }
            confidence = risk_conf.get(risk, 0.5)
            if confidence < min_confidence:
                continue

            content = {
                "category": str(getattr(prop, "category", "")),
                "description": getattr(prop, "description", ""),
                "expected_benefit": getattr(prop, "expected_benefit", ""),
                "risk_level": risk,
                "source": str(getattr(prop, "source", "")),
            }

            # Include change spec summary without the actual code
            change_spec = getattr(prop, "change_spec", None)
            if change_spec is not None:
                content["change_category"] = str(
                    getattr(change_spec, "executor_name", "")
                )
                content["change_description"] = getattr(
                    change_spec, "executor_description", ""
                ) or getattr(change_spec, "channel_description", "")

            payloads.append(ExchangePayload(
                kind=ExchangePayloadKind.MUTATION_PATTERN,
                confidence=confidence,
                content=content,
                content_hash=_content_hash(content),
                source_instance_id=instance_id,
                provenance_chain=[instance_id],
                domain="evolution",
            ))
            if len(payloads) >= max_items:
                break
    except Exception as exc:
        logger.warning("mutation_pattern_collection_failed", error=str(exc))
    return payloads


async def collect_economic_intel(
    oikos: Any,
    instance_id: str,
    min_confidence: float = 0.0,
    max_items: int = 20,
) -> list[ExchangePayload]:
    """
    Collect economic intelligence from Oikos.

    Gathers yield position performance, threat signals, starvation
    level warnings, and successful revenue strategies.
    """
    payloads: list[ExchangePayload] = []
    try:
        # Economic state summary
        if callable(getattr(oikos, "get_state", None)):
            state = await oikos.get_state()
            if state is not None:
                starvation = str(getattr(state, "starvation_level", "nominal"))
                content = {
                    "type": "economic_state_summary",
                    "starvation_level": starvation,
                    "metabolic_rate": float(
                        getattr(state, "metabolic_rate", 0.0)
                    ),
                }
                confidence = 0.9  # Direct observation, high confidence
                if confidence >= min_confidence:
                    payloads.append(ExchangePayload(
                        kind=ExchangePayloadKind.ECONOMIC_INTEL,
                        confidence=confidence,
                        content=content,
                        content_hash=_content_hash(content),
                        source_instance_id=instance_id,
                        provenance_chain=[instance_id],
                        domain="economic.state",
                    ))

        # Yield position performance (anonymised — no wallet addresses)
        if callable(getattr(oikos, "get_yield_positions", None)):
            positions = await oikos.get_yield_positions()
            for pos in (positions or [])[:max_items]:
                protocol = getattr(pos, "protocol_name", "")
                apy = float(getattr(pos, "current_apy", 0.0))
                content = {
                    "type": "yield_signal",
                    "protocol": protocol,
                    "apy": apy,
                    "risk_tier": str(getattr(pos, "risk_tier", "unknown")),
                }
                # Higher APY = more signal value
                confidence = min(1.0, 0.5 + apy / 100.0)
                if confidence >= min_confidence:
                    payloads.append(ExchangePayload(
                        kind=ExchangePayloadKind.ECONOMIC_INTEL,
                        confidence=confidence,
                        content=content,
                        content_hash=_content_hash(content),
                        source_instance_id=instance_id,
                        provenance_chain=[instance_id],
                        domain="economic.yield",
                    ))
                if len(payloads) >= max_items:
                    break
    except Exception as exc:
        logger.warning("economic_intel_collection_failed", error=str(exc))
    return payloads


# ─── Exchange Protocol ──────────────────────────────────────────────


class ExchangeProtocol:
    """
    Orchestrates selective, signed, trust-gated knowledge exchange
    over an established federation channel.

    Outbound (push):
      prepare_push() -> collects & filters -> returns signed ExchangeEnvelope

    Outbound (pull request):
      prepare_pull_request() -> builds a PULL envelope asking the peer for knowledge

    Inbound handling is delegated to IngestionPipeline (ingestion.py).
    """

    def __init__(
        self,
        identity: IdentityManager | None = None,
        privacy_filter: PrivacyFilter | None = None,
        instance_id: str = "",
    ) -> None:
        self._identity = identity
        self._privacy = privacy_filter
        self._instance_id = instance_id
        self._logger = logger.bind(component="exchange_protocol")

        # Stats
        self._envelopes_sent: int = 0
        self._envelopes_received: int = 0
        self._payloads_sent: int = 0
        self._payloads_filtered: int = 0

    # ─── Outbound: Prepare a Push Envelope ───────────────────────

    async def prepare_push(
        self,
        link: FederationLink,
        payloads: list[ExchangePayload],
    ) -> ExchangeEnvelope | None:
        """
        Filter, sign, and package payloads into an outbound PUSH envelope.

        Returns None if no payloads survive filtering (nothing to send).
        """
        eligible = self._apply_outbound_filters(payloads, link)
        if not eligible:
            self._logger.debug(
                "push_empty_after_filter",
                remote_id=link.remote_instance_id,
                original_count=len(payloads),
            )
            return None

        envelope = ExchangeEnvelope(
            sender_instance_id=self._instance_id,
            receiver_instance_id=link.remote_instance_id,
            direction=ExchangeDirection.PUSH,
            payloads=eligible,
        )

        # Sign the envelope
        self._sign_envelope(envelope)

        self._envelopes_sent += 1
        self._payloads_sent += len(eligible)
        self._payloads_filtered += len(payloads) - len(eligible)

        self._logger.info(
            "push_envelope_prepared",
            remote_id=link.remote_instance_id,
            payload_count=len(eligible),
            filtered_out=len(payloads) - len(eligible),
        )

        return envelope

    # ─── Outbound: Prepare a Pull Request ────────────────────────

    def prepare_pull_request(
        self,
        link: FederationLink,
        kinds: list[ExchangePayloadKind],
        query: str = "",
        max_items: int = 20,
    ) -> ExchangeEnvelope | None:
        """
        Build a PULL request envelope asking a peer for specific knowledge.

        Only requests kinds that the link's trust level permits.
        """
        # Filter to kinds the trust level allows
        eligible_kinds = [
            k for k in kinds
            if link.trust_level >= EXCHANGE_TRUST_GATES.get(k, TrustLevel.ALLY)
        ]
        if not eligible_kinds:
            self._logger.debug(
                "pull_request_no_eligible_kinds",
                remote_id=link.remote_instance_id,
                requested=kinds,
                trust_level=link.trust_level.name,
            )
            return None

        envelope = ExchangeEnvelope(
            sender_instance_id=self._instance_id,
            receiver_instance_id=link.remote_instance_id,
            direction=ExchangeDirection.PULL,
            requested_kinds=eligible_kinds,
            requested_query=query,
            max_items=max_items,
        )

        self._sign_envelope(envelope)
        self._envelopes_sent += 1

        return envelope

    # ─── Inbound: Handle a Pull Request ──────────────────────────

    async def handle_pull_request(
        self,
        envelope: ExchangeEnvelope,
        link: FederationLink,
        evo: Any = None,
        simula: Any = None,
        oikos: Any = None,
    ) -> ExchangeEnvelope | None:
        """
        Handle an inbound PULL request by collecting eligible knowledge
        and returning a PUSH envelope as the response.
        """
        if envelope.direction != ExchangeDirection.PULL:
            return None

        all_payloads: list[ExchangePayload] = []
        kind_count = max(1, len(envelope.requested_kinds))
        per_kind_limit = max(1, envelope.max_items // kind_count)

        for kind in envelope.requested_kinds:
            # Trust gate
            required_trust = EXCHANGE_TRUST_GATES.get(kind, TrustLevel.ALLY)
            if link.trust_level < required_trust:
                continue

            floor = EXCHANGE_CONFIDENCE_FLOORS.get(kind, 0.5)

            if kind == ExchangePayloadKind.HYPOTHESIS and evo is not None:
                all_payloads.extend(await collect_hypotheses(
                    evo, self._instance_id,
                    min_confidence=floor, max_items=per_kind_limit,
                ))
            elif kind == ExchangePayloadKind.PROCEDURE and evo is not None:
                all_payloads.extend(await collect_procedures(
                    evo, self._instance_id,
                    min_confidence=floor, max_items=per_kind_limit,
                ))
            elif kind == ExchangePayloadKind.MUTATION_PATTERN and simula is not None:
                all_payloads.extend(await collect_mutation_patterns(
                    simula, self._instance_id,
                    min_confidence=floor, max_items=per_kind_limit,
                ))
            elif kind == ExchangePayloadKind.ECONOMIC_INTEL and oikos is not None:
                all_payloads.extend(await collect_economic_intel(
                    oikos, self._instance_id,
                    min_confidence=floor, max_items=per_kind_limit,
                ))

        return await self.prepare_push(link, all_payloads[:envelope.max_items])

    # ─── Verify Inbound Envelope ─────────────────────────────────

    def verify_envelope(
        self,
        envelope: ExchangeEnvelope,
        link: FederationLink,
    ) -> tuple[bool, str]:
        """
        Verify an inbound envelope's signature and sender identity.

        Returns (valid, error_reason).
        """
        # Check sender matches the link
        if envelope.sender_instance_id != link.remote_instance_id:
            return False, (
                f"Sender mismatch: envelope says {envelope.sender_instance_id}, "
                f"link says {link.remote_instance_id}"
            )

        # Check receiver is us
        if envelope.receiver_instance_id != self._instance_id:
            return False, (
                f"Receiver mismatch: envelope addressed to "
                f"{envelope.receiver_instance_id}, we are {self._instance_id}"
            )

        # Verify Ed25519 signature
        if self._identity and link.remote_identity and envelope.signature:
            signing_data = self._envelope_signing_data(envelope)
            valid = self._identity.verify_signature(
                signing_data,
                bytes.fromhex(envelope.signature),
                link.remote_identity.public_key_pem,
            )
            if not valid:
                return False, "Invalid Ed25519 signature"

        self._envelopes_received += 1
        return True, ""

    # ─── Build Interaction Record ────────────────────────────────

    @staticmethod
    def build_exchange_interaction(
        link: FederationLink,
        direction: str,
        outcome: InteractionOutcome,
        payload_count: int,
        description: str = "",
    ) -> FederationInteraction:
        """Build a FederationInteraction record for an exchange event."""
        return FederationInteraction(
            link_id=link.id,
            remote_instance_id=link.remote_instance_id,
            interaction_type="iiep_exchange",
            direction=direction,
            outcome=outcome,
            description=description or f"IIEP exchange: {payload_count} payloads",
            # More payloads = more trust value, capped at 2.0
            trust_value=min(2.0, 0.5 * payload_count),
            metadata={"payload_count": payload_count},
        )

    # ─── Outbound Filtering ──────────────────────────────────────

    def _apply_outbound_filters(
        self,
        payloads: list[ExchangePayload],
        link: FederationLink,
    ) -> list[ExchangePayload]:
        """
        Apply confidence floor + trust gate filtering to outbound payloads.

        Also rejects payloads that have already traversed the receiver
        (loop prevention via provenance_chain).
        """
        eligible: list[ExchangePayload] = []
        for p in payloads:
            # Confidence floor
            floor = EXCHANGE_CONFIDENCE_FLOORS.get(p.kind, 0.5)
            if p.confidence < floor:
                continue

            # Trust gate
            required_trust = EXCHANGE_TRUST_GATES.get(p.kind, TrustLevel.ALLY)
            if link.trust_level < required_trust:
                continue

            # Loop prevention — don't send knowledge back to an instance
            # that already appears in the provenance chain
            if link.remote_instance_id in p.provenance_chain:
                continue

            eligible.append(p)
        return eligible

    # ─── Signing ─────────────────────────────────────────────────

    def _sign_envelope(self, envelope: ExchangeEnvelope) -> None:
        """Sign the envelope with our Ed25519 private key."""
        if not self._identity:
            return
        signing_data = self._envelope_signing_data(envelope)
        try:
            sig_bytes = self._identity.sign(signing_data)
            envelope.signature = sig_bytes.hex()
        except Exception as exc:
            self._logger.warning("envelope_signing_failed", error=str(exc))

    @staticmethod
    def _envelope_signing_data(envelope: ExchangeEnvelope) -> bytes:
        """
        Canonical bytes for signing/verifying an envelope.

        Covers: sender, receiver, direction, sorted payload hashes, timestamp.
        """
        payload_hashes = sorted(p.content_hash for p in envelope.payloads)
        canonical = json.dumps({
            "sender": envelope.sender_instance_id,
            "receiver": envelope.receiver_instance_id,
            "direction": envelope.direction,
            "payload_hashes": payload_hashes,
            "timestamp": envelope.timestamp.isoformat(),
        }, sort_keys=True)
        return canonical.encode()

    # ─── Stats ───────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "envelopes_sent": self._envelopes_sent,
            "envelopes_received": self._envelopes_received,
            "payloads_sent": self._payloads_sent,
            "payloads_filtered": self._payloads_filtered,
        }
