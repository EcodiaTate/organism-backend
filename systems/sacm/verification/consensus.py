"""
EcodiaOS — SACM Consensus Verifier

Combines deterministic replay and probabilistic canary auditing
into a single verification decision with configurable weights.

Decision logic:
  - BOTH_PASS:  Accept unconditionally.
  - BOTH_FAIL:  Reject unconditionally.
  - ONE_PASS:   Apply weighted scoring:
      score = w_det × det_score + w_prob × prob_score
      where det_score  = 1.0 if replay passed, 0.0 if failed
            prob_score = 1.0 if canaries passed, 0.0 if failed
      Accept if score ≥ acceptance_threshold.

  By default, weights are equal (0.5, 0.5) and threshold is 0.75,
  meaning both must pass.  But operators can bias toward one strategy:
  e.g., for non-deterministic workloads, lower det weight so canary
  results dominate.

The consensus verifier also tracks provider trust scores over time,
decaying trust on failures and recovering on successes.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped

if TYPE_CHECKING:
    from collections.abc import Sequence

    from systems.sacm.verification.deterministic import (
        DeterministicReplayVerifier,
        ReplayItem,
    )
    from systems.sacm.verification.deterministic import (
        VerificationReport as DeterministicReport,
    )
    from systems.sacm.verification.probabilistic import (
        CanaryItem,
        ProbabilisticAuditReport,
        ProbabilisticAuditVerifier,
    )

logger = structlog.get_logger("systems.sacm.verification.consensus")


# ─── Types ──────────────────────────────────────────────────────


class ConsensusOutcome(enum.StrEnum):
    """Possible outcomes of the consensus decision."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    DEGRADED_ACCEPT = "degraded_accept"   # accepted despite partial failure


class ConsensusWeights(EOSBaseModel):
    """Weights for combining verification strategies."""

    deterministic: float = 0.5
    probabilistic: float = 0.5
    acceptance_threshold: float = 0.75

    def validate_weights(self) -> None:
        total = self.deterministic + self.probabilistic
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.4f} "
                f"(det={self.deterministic}, prob={self.probabilistic})"
            )


class ProviderTrust(EOSBaseModel):
    """Rolling trust score for a remote compute provider."""

    provider_id: str
    trust_score: float = 1.0       # 0.0 = untrusted, 1.0 = fully trusted
    total_batches: int = 0
    accepted_batches: int = 0
    rejected_batches: int = 0
    consecutive_failures: int = 0

    # Exponential decay on failure, linear recovery on success
    decay_factor: float = 0.7     # multiply trust by this on failure
    recovery_increment: float = 0.05  # add this on success (capped at 1.0)

    # If trust drops below this, provider is quarantined
    quarantine_threshold: float = 0.3

    @property
    def is_quarantined(self) -> bool:
        return self.trust_score < self.quarantine_threshold

    def record_success(self) -> None:
        self.total_batches += 1
        self.accepted_batches += 1
        self.consecutive_failures = 0
        self.trust_score = min(1.0, self.trust_score + self.recovery_increment)

    def record_failure(self) -> None:
        self.total_batches += 1
        self.rejected_batches += 1
        self.consecutive_failures += 1
        self.trust_score = max(0.0, self.trust_score * self.decay_factor)


class ConsensusReport(Identified, Timestamped):
    """Full report from a consensus verification run."""

    outcome: ConsensusOutcome = ConsensusOutcome.REJECTED
    weighted_score: float = 0.0
    weights: ConsensusWeights = Field(default_factory=ConsensusWeights)
    deterministic_report: DeterministicReport | None = None
    probabilistic_report: ProbabilisticAuditReport | None = None
    provider_trust: ProviderTrust | None = None
    error: str = ""


# ─── Consensus Verifier ────────────────────────────────────────


class ConsensusVerifier:
    """
    Orchestrates both verification strategies and makes a combined decision.

    Usage:
        consensus = ConsensusVerifier(
            deterministic=det_verifier,
            probabilistic=prob_verifier,
            weights=ConsensusWeights(deterministic=0.4, probabilistic=0.6),
        )
        report = await consensus.verify(
            replay_items=replay_items,
            canaries=canaries,
            remote_results=remote_results,
            provider_id="akash-provider-xyz",
        )
    """

    def __init__(
        self,
        deterministic: DeterministicReplayVerifier,
        probabilistic: ProbabilisticAuditVerifier,
        weights: ConsensusWeights | None = None,
    ) -> None:
        self._det = deterministic
        self._prob = probabilistic
        self._weights = weights or ConsensusWeights()
        self._weights.validate_weights()
        self._trust_store: dict[str, ProviderTrust] = {}
        self._log = logger.bind(component="sacm.verification.consensus")

    def get_provider_trust(self, provider_id: str) -> ProviderTrust:
        """Get or create a trust record for a provider."""
        if provider_id not in self._trust_store:
            self._trust_store[provider_id] = ProviderTrust(
                provider_id=provider_id,
            )
        return self._trust_store[provider_id]

    async def verify(
        self,
        replay_items: Sequence[ReplayItem],
        canaries: Sequence[CanaryItem],
        remote_results: Sequence[bytes],
        provider_id: str = "",
    ) -> ConsensusReport:
        """
        Run both verification strategies and produce a consensus decision.

        Args:
            replay_items:   Items for deterministic replay verification.
            canaries:       Canary items with expected outputs.
            remote_results: Full result batch from the remote provider.
            provider_id:    Provider identifier for trust tracking.

        Returns:
            ConsensusReport with the combined verdict.
        """
        trust = self.get_provider_trust(provider_id) if provider_id else None

        # Pre-check: reject if provider is quarantined
        if trust is not None and trust.is_quarantined:
            self._log.warning(
                "provider_quarantined",
                provider_id=provider_id,
                trust_score=trust.trust_score,
            )
            return ConsensusReport(
                outcome=ConsensusOutcome.REJECTED,
                weighted_score=0.0,
                weights=self._weights,
                provider_trust=trust,
                error=(
                    f"Provider {provider_id} is quarantined "
                    f"(trust={trust.trust_score:.2f} < {trust.quarantine_threshold})"
                ),
            )

        # Run both verifiers
        det_report: DeterministicReport | None = None
        prob_report: ProbabilisticAuditReport | None = None
        det_error: str = ""
        prob_error: str = ""

        # Deterministic replay
        try:
            det_report = await self._det.verify(replay_items)
        except Exception as exc:
            det_error = f"deterministic replay failed: {exc}"
            self._log.error("det_verify_error", error=str(exc), exc_info=True)

        # Probabilistic canary audit
        try:
            prob_report = await self._prob.verify(canaries, remote_results)
        except Exception as exc:
            prob_error = f"probabilistic audit failed: {exc}"
            self._log.error("prob_verify_error", error=str(exc), exc_info=True)

        # Compute weighted score
        det_score = 0.0
        prob_score = 0.0

        if det_report is not None:
            det_score = 1.0 if det_report.accepted else 0.0
        if prob_report is not None:
            prob_score = 1.0 if prob_report.accepted else 0.0

        weighted = (
            self._weights.deterministic * det_score
            + self._weights.probabilistic * prob_score
        )

        # Decision
        if weighted >= self._weights.acceptance_threshold:
            if det_score == 1.0 and prob_score == 1.0:
                outcome = ConsensusOutcome.ACCEPTED
            else:
                outcome = ConsensusOutcome.DEGRADED_ACCEPT
        else:
            outcome = ConsensusOutcome.REJECTED

        # Update trust
        if trust is not None:
            if outcome in (ConsensusOutcome.ACCEPTED, ConsensusOutcome.DEGRADED_ACCEPT):
                trust.record_success()
            else:
                trust.record_failure()

        errors = " | ".join(filter(None, [det_error, prob_error]))

        report = ConsensusReport(
            outcome=outcome,
            weighted_score=weighted,
            weights=self._weights,
            deterministic_report=det_report,
            probabilistic_report=prob_report,
            provider_trust=trust,
            error=errors,
        )

        self._log.info(
            "consensus_verdict",
            outcome=outcome.value,
            weighted_score=round(weighted, 3),
            det_accepted=det_report.accepted if det_report else None,
            prob_accepted=prob_report.accepted if prob_report else None,
            provider_id=provider_id,
            trust=trust.trust_score if trust else None,
        )
        return report
