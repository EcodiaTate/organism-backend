"""
EcodiaOS - EIS Quarantine Gate (Pre-Action Validation Layer)

The quarantine gate is a gating layer that Simula (mutations) and
Federation (incoming knowledge) must check before acting on received
content. It combines three threat signals:

  1. **Taint analysis** - constitutional path safety (existing)
  2. **Threat library** - known-bad pattern matching (new)
  3. **Anomaly context** - whether the instance is under elevated threat

The gate produces a ``QuarantineDecision`` with one of four verdicts:

  - ALLOW:     Content is safe to proceed.
  - HOLD:      Content should be queued for human review.
  - BLOCK:     Content should be rejected immediately.
  - DEFENSIVE: Instance should enter defensive posture (recommend Thymos
               defensive mode via THREAT_DETECTED event).

Design constraints:
  - Synchronous for mutations (must gate before application)
  - The gate never applies mutations itself - it only advises
  - The gate is composable: callers can override the decision with
    appropriate governance escalation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id
from systems.eis.anomaly_detector import AnomalyDetector, AnomalySeverity
from systems.eis.taint_models import MutationProposal, TaintRiskAssessment, TaintSeverity

if TYPE_CHECKING:
    from systems.eis.taint_engine import TaintEngine
    from systems.eis.threat_library import (
        ThreatLibrary,
        ThreatScanResult,
    )

logger = structlog.get_logger().bind(component="quarantine_gate")


# ─── Decision Types ──────────────────────────────────────────────────────────


class GateVerdict(StrEnum):
    """What the quarantine gate recommends for the content."""

    ALLOW = "allow"          # Safe to proceed
    HOLD = "hold"            # Queue for human/governance review
    BLOCK = "block"          # Reject immediately
    DEFENSIVE = "defensive"  # Block + recommend Thymos defensive mode


class GateSource(StrEnum):
    """What type of content is being gated."""

    MUTATION = "mutation"
    FEDERATION_KNOWLEDGE = "federation_knowledge"
    FEDERATION_ADVISORY = "federation_advisory"


# ─── Quarantine Decision ────────────────────────────────────────────────────


@dataclass
class QuarantineDecision:
    """
    The quarantine gate's recommendation for a piece of content.

    Consumers (Simula, Federation) use this to decide whether to proceed,
    queue for review, or reject. The decision carries full justification
    so governance (Equor) can audit why content was gated.
    """

    id: str = field(default_factory=new_id)
    verdict: GateVerdict = GateVerdict.ALLOW
    source: GateSource = GateSource.MUTATION

    # ── Justification ──
    reasons: list[str] = field(default_factory=list)
    taint_severity: str = "clear"
    threat_library_matches: int = 0
    anomaly_context: str = "nominal"

    # ── Routing flags ──
    requires_human_approval: bool = False
    requires_equor_review: bool = False
    recommend_defensive_mode: bool = False

    # ── Raw assessments (for auditing) ──
    taint_assessment: TaintRiskAssessment | None = None
    threat_scan: ThreatScanResult | None = None

    # ── Timing ──
    gate_latency_ms: int = 0


# ─── Quarantine Gate ─────────────────────────────────────────────────────────


class QuarantineGate:
    """
    Pre-action validation layer for mutations and federated knowledge.

    Combines taint analysis, threat library scanning, and anomaly
    context into a single gating decision. This is the checkpoint
    that Simula and Federation call before acting on content.

    The gate is stateless - it queries TaintEngine, ThreatLibrary,
    and AnomalyDetector for their assessments and combines them.

    Usage:
        gate = QuarantineGate(taint_engine, threat_library, anomaly_detector)

        # For mutations:
        decision = gate.evaluate_mutation(proposal)
        if decision.verdict == GateVerdict.BLOCK:
            reject_mutation(proposal)

        # For federation:
        decision = gate.evaluate_knowledge(content, source_instance)
        if decision.verdict != GateVerdict.ALLOW:
            quarantine_knowledge(content)
    """

    def __init__(
        self,
        taint_engine: TaintEngine,
        threat_library: ThreatLibrary,
        anomaly_detector: AnomalyDetector,
    ) -> None:
        self._taint = taint_engine
        self._library = threat_library
        self._detector = anomaly_detector

        # ── Counters ──
        self._total_evaluations: int = 0
        self._mutations_evaluated: int = 0
        self._knowledge_evaluated: int = 0
        self._blocked: int = 0
        self._held: int = 0
        self._allowed: int = 0
        self._defensive: int = 0

        self._logger = logger

    # ─── Mutation Gating ──────────────────────────────────────────

    def evaluate_mutation(self, proposal: MutationProposal) -> QuarantineDecision:
        """
        Evaluate a mutation proposal against all threat signals.

        Combines:
          1. Taint analysis (constitutional path safety)
          2. Threat library scan (known-bad patterns)
          3. Current anomaly context (threat posture)

        Returns a QuarantineDecision with the gate's recommendation.
        """
        t0 = time.monotonic()
        reasons: list[str] = []

        # ── Signal 1: Taint analysis ──
        taint = self._taint.analyse_mutation(proposal)

        # ── Signal 2: Threat library scan ──
        from systems.eis.constitutional_graph import extract_changed_functions
        changed_fns = extract_changed_functions(proposal.diff)

        threat_scan = self._library.scan_mutation(
            file_path=proposal.file_path,
            diff=proposal.diff,
            changed_functions=changed_fns,
        )

        # ── Signal 3: Anomaly context ──
        anomaly_posture = self._current_threat_posture()

        # ── Combine signals into verdict ──
        verdict = GateVerdict.ALLOW
        requires_human = False
        requires_equor = False
        recommend_defensive = False

        # Taint-based escalation
        if taint.block_mutation:
            verdict = GateVerdict.BLOCK
            reasons.append(
                f"Taint analysis: CRITICAL constitutional path touched "
                f"({taint.overall_severity.value})"
            )
            requires_human = True
            requires_equor = True
        elif taint.requires_equor_elevated_review:
            verdict = _escalate(verdict, GateVerdict.HOLD)
            reasons.append(
                f"Taint analysis: elevated constitutional risk "
                f"({taint.overall_severity.value})"
            )
            requires_equor = True
        elif taint.overall_severity == TaintSeverity.ADVISORY:
            reasons.append("Taint analysis: advisory-level constitutional touch")

        # Threat library matches
        if threat_scan.is_threat:
            if threat_scan.should_block:
                verdict = _escalate(verdict, GateVerdict.BLOCK)
                reasons.append(
                    f"Threat library: {len(threat_scan.matches)} matches "
                    f"(critical severity)"
                )
                requires_human = True
            else:
                verdict = _escalate(verdict, GateVerdict.HOLD)
                reasons.append(
                    f"Threat library: {len(threat_scan.matches)} matches "
                    f"({threat_scan.highest_severity} severity)"
                )
                requires_equor = True

        # Anomaly context escalation
        if anomaly_posture in ("critical", "high"):
            if verdict == GateVerdict.ALLOW:
                verdict = GateVerdict.HOLD
                reasons.append(
                    f"Anomaly context: elevated {anomaly_posture} threat posture"
                )
                requires_equor = True
            elif verdict == GateVerdict.HOLD and anomaly_posture == "critical":
                verdict = GateVerdict.BLOCK
                reasons.append(
                    "Anomaly context: critical posture - escalating hold to block"
                )
                requires_human = True

        # Defensive mode recommendation
        if verdict == GateVerdict.BLOCK and anomaly_posture in ("critical", "high"):
            verdict = GateVerdict.DEFENSIVE
            recommend_defensive = True
            reasons.append(
                "Combined threat signals recommend Thymos defensive mode"
            )

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # ── Update counters ──
        self._total_evaluations += 1
        self._mutations_evaluated += 1
        self._update_verdict_counter(verdict)

        decision = QuarantineDecision(
            verdict=verdict,
            source=GateSource.MUTATION,
            reasons=reasons,
            taint_severity=taint.overall_severity.value,
            threat_library_matches=len(threat_scan.matches),
            anomaly_context=anomaly_posture,
            requires_human_approval=requires_human,
            requires_equor_review=requires_equor,
            recommend_defensive_mode=recommend_defensive,
            taint_assessment=taint,
            threat_scan=threat_scan,
            gate_latency_ms=elapsed_ms,
        )

        self._logger.info(
            "quarantine_gate_mutation",
            mutation_id=proposal.id,
            file_path=proposal.file_path,
            verdict=verdict.value,
            taint_severity=taint.overall_severity.value,
            threat_matches=len(threat_scan.matches),
            anomaly_posture=anomaly_posture,
            latency_ms=elapsed_ms,
        )

        return decision

    # ─── Federation Knowledge Gating ──────────────────────────────

    def evaluate_knowledge(
        self,
        content: str,
        source_instance: str,
        knowledge_type: str = "",
    ) -> QuarantineDecision:
        """
        Evaluate incoming federated knowledge against threat signals.

        Combines:
          1. Threat library scan (previously rejected knowledge)
          2. Current anomaly context

        Note: taint analysis is not applicable to knowledge (it's for
        code mutations). EIS gate (percept screening) should also run
        on the content separately for epistemic threat detection.
        """
        t0 = time.monotonic()
        reasons: list[str] = []

        # ── Signal 1: Threat library scan ──
        threat_scan = self._library.scan_knowledge(
            content=content,
            source_instance=source_instance,
        )

        # ── Signal 2: Anomaly context ──
        anomaly_posture = self._current_threat_posture()

        # ── Combine signals ──
        verdict = GateVerdict.ALLOW
        requires_human = False
        requires_equor = False
        recommend_defensive = False

        # Threat library matches
        if threat_scan.is_threat:
            if threat_scan.should_block:
                verdict = GateVerdict.BLOCK
                reasons.append(
                    f"Threat library: incoming knowledge matches "
                    f"{len(threat_scan.matches)} known-bad patterns"
                )
                requires_human = True
            else:
                verdict = GateVerdict.HOLD
                reasons.append(
                    f"Threat library: {len(threat_scan.matches)} partial matches "
                    f"against previously rejected knowledge"
                )
                requires_equor = True

        # Anomaly context
        if anomaly_posture in ("critical", "high") and verdict == GateVerdict.ALLOW:
            verdict = GateVerdict.HOLD
            reasons.append(
                f"Elevated {anomaly_posture} threat posture - "
                f"holding federated knowledge for review"
            )
            requires_equor = True

        # Defensive mode
        if verdict == GateVerdict.BLOCK and anomaly_posture == "critical":
            verdict = GateVerdict.DEFENSIVE
            recommend_defensive = True
            reasons.append(
                "Blocking federated knowledge under critical posture; "
                "recommend Thymos defensive mode"
            )

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        self._total_evaluations += 1
        self._knowledge_evaluated += 1
        self._update_verdict_counter(verdict)

        decision = QuarantineDecision(
            verdict=verdict,
            source=GateSource.FEDERATION_KNOWLEDGE,
            reasons=reasons,
            threat_library_matches=len(threat_scan.matches),
            anomaly_context=anomaly_posture,
            requires_human_approval=requires_human,
            requires_equor_review=requires_equor,
            recommend_defensive_mode=recommend_defensive,
            threat_scan=threat_scan,
            gate_latency_ms=elapsed_ms,
        )

        self._logger.info(
            "quarantine_gate_knowledge",
            source_instance=source_instance,
            knowledge_type=knowledge_type,
            verdict=verdict.value,
            threat_matches=len(threat_scan.matches),
            anomaly_posture=anomaly_posture,
            latency_ms=elapsed_ms,
        )

        return decision

    # ─── Threat Posture ───────────────────────────────────────────

    def _current_threat_posture(self) -> str:
        """
        Derive the current threat posture from recent anomalies.

        Returns: "nominal", "low", "medium", "high", or "critical"
        """
        recent = self._detector.recent_anomalies(limit=10)
        if not recent:
            return "nominal"

        severity_scores = {
            AnomalySeverity.CRITICAL: 4,
            AnomalySeverity.HIGH: 3,
            AnomalySeverity.MEDIUM: 2,
            AnomalySeverity.LOW: 1,
        }

        max_severity = 0
        for anomaly in recent[:5]:
            score = severity_scores.get(anomaly.severity, 0)
            max_severity = max(max_severity, score)

        if max_severity >= 4:
            return "critical"
        if max_severity >= 3:
            return "high"
        if max_severity >= 2:
            return "medium"
        if max_severity >= 1:
            return "low"
        return "nominal"

    # ─── Helpers ──────────────────────────────────────────────────

    def _update_verdict_counter(self, verdict: GateVerdict) -> None:
        if verdict == GateVerdict.ALLOW:
            self._allowed += 1
        elif verdict == GateVerdict.HOLD:
            self._held += 1
        elif verdict == GateVerdict.BLOCK:
            self._blocked += 1
        elif verdict == GateVerdict.DEFENSIVE:
            self._defensive += 1

    def stats(self) -> dict[str, Any]:
        """Return observable statistics."""
        return {
            "total_evaluations": self._total_evaluations,
            "mutations_evaluated": self._mutations_evaluated,
            "knowledge_evaluated": self._knowledge_evaluated,
            "verdicts": {
                "allowed": self._allowed,
                "held": self._held,
                "blocked": self._blocked,
                "defensive": self._defensive,
            },
        }


# ─── Helpers ──────────────────────────────────────────────────────────────────


_VERDICT_RANK: dict[GateVerdict, int] = {
    GateVerdict.ALLOW: 0,
    GateVerdict.HOLD: 1,
    GateVerdict.BLOCK: 2,
    GateVerdict.DEFENSIVE: 3,
}


def _escalate(current: GateVerdict, proposed: GateVerdict) -> GateVerdict:
    """Return the more restrictive of two verdicts."""
    if _VERDICT_RANK.get(proposed, 0) > _VERDICT_RANK.get(current, 0):
        return proposed
    return current
