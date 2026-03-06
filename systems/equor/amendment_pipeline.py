"""
EcodiaOS — Equor Amendment Pipeline

The formal governance mechanism for evolving non-invariant constitutional rules.
Amendments can change drive weights, thresholds, and operational policies —
but the 10 core invariants are immutable and can never be amended.

Pipeline stages:
  1. Proposal   — submit change + rationale + supporting evidence (Evo hypothesis IDs)
  2. Validation — structural checks, cooldown, evidence quality
  3. Shadow     — proposed rule runs alongside the current rule; both produce verdicts
  4. Drift Gate — measure whether shadow verdicts cause constitutional drift
  5. Vote       — supermajority community vote (only after shadow period passes)
  6. Adoption   — apply the amendment, increment constitution version
  7. Cooldown   — prevent rapid erosion through incremental changes

Shadow mode is the heart of the pipeline: every review() call during the shadow
period runs the verdict engine TWICE (current weights and proposed weights) and
logs both outcomes.  If the shadow verdicts would have caused invariant violations
or unacceptable drift, the amendment is auto-rejected before the vote ever happens.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import (
    DriveAlignmentVector,
    Verdict,
    new_id,
    utc_now,
)
from systems.equor.amendment import validate_amendment_proposal
from systems.equor.verdict import compute_verdict

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from config import GovernanceConfig
    from primitives.intent import Intent
    from systems.equor.constitutional_memory import ConstitutionalMemory

logger = structlog.get_logger()


# ── Amendment lifecycle states ───────────────────────────────────────

class AmendmentStatus(StrEnum):
    PROPOSED = "proposed"
    DELIBERATION = "deliberation"
    SHADOW = "shadow"
    SHADOW_PASSED = "shadow_passed"
    SHADOW_FAILED = "shadow_failed"
    VOTING = "voting"
    PASSED = "passed"
    FAILED = "failed"
    ADOPTED = "adopted"
    REJECTED = "rejected"


# ── Shadow evaluation result ─────────────────────────────────────────

class ShadowVerdict:
    """Result of running a single intent through both current and proposed weights."""

    __slots__ = (
        "intent_id",
        "current_verdict",
        "proposed_verdict",
        "current_composite",
        "proposed_composite",
        "verdict_diverged",
        "invariant_violation",
    )

    def __init__(
        self,
        intent_id: str,
        current_verdict: str,
        proposed_verdict: str,
        current_composite: float,
        proposed_composite: float,
    ) -> None:
        self.intent_id = intent_id
        self.current_verdict = current_verdict
        self.proposed_verdict = proposed_verdict
        self.current_composite = current_composite
        self.proposed_composite = proposed_composite
        self.verdict_diverged = current_verdict != proposed_verdict
        # An invariant violation occurs when the proposed weights would APPROVE
        # something the current weights BLOCKED.
        self.invariant_violation = (
            current_verdict == Verdict.BLOCKED.value
            and proposed_verdict == Verdict.APPROVED.value
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "current_verdict": self.current_verdict,
            "proposed_verdict": self.proposed_verdict,
            "current_composite": round(self.current_composite, 4),
            "proposed_composite": round(self.proposed_composite, 4),
            "verdict_diverged": self.verdict_diverged,
            "invariant_violation": self.invariant_violation,
        }


# ── Shadow tracker (in-memory, per active amendment) ──────────────────

class ShadowTracker:
    """
    Accumulates shadow verdicts for a single amendment proposal during
    its shadow period.  Lives in memory on EquorService; flushed to
    Neo4j periodically and at shadow-period end.
    """

    def __init__(
        self,
        proposal_id: str,
        proposed_drives: dict[str, float],
        shadow_start: datetime,
        shadow_end: datetime,
        max_invariant_violations: int = 0,
        max_divergence_rate: float = 0.15,
    ) -> None:
        self.proposal_id = proposal_id
        self.proposed_drives = proposed_drives
        self.shadow_start = shadow_start
        self.shadow_end = shadow_end
        self.max_invariant_violations = max_invariant_violations
        self.max_divergence_rate = max_divergence_rate

        self._verdicts: list[ShadowVerdict] = []
        self._invariant_violations: int = 0
        self._divergences: int = 0

    @property
    def total_evaluations(self) -> int:
        return len(self._verdicts)

    @property
    def invariant_violations(self) -> int:
        return self._invariant_violations

    @property
    def divergence_rate(self) -> float:
        if not self._verdicts:
            return 0.0
        return self._divergences / len(self._verdicts)

    @property
    def is_expired(self) -> bool:
        return utc_now() >= self.shadow_end

    @property
    def has_failed(self) -> bool:
        """True if the shadow period should be terminated early due to violations."""
        return self._invariant_violations > self.max_invariant_violations

    def record(self, shadow: ShadowVerdict) -> None:
        """Record a shadow verdict."""
        self._verdicts.append(shadow)
        if shadow.invariant_violation:
            self._invariant_violations += 1
        if shadow.verdict_diverged:
            self._divergences += 1

    def compute_report(self) -> dict[str, Any]:
        """Produce a summary report for the shadow period."""
        if not self._verdicts:
            return {
                "proposal_id": self.proposal_id,
                "total_evaluations": 0,
                "status": "insufficient_data",
                "passed": False,
            }

        # Composite alignment delta (mean shift)
        composite_deltas = [
            v.proposed_composite - v.current_composite for v in self._verdicts
        ]
        mean_delta = sum(composite_deltas) / len(composite_deltas)

        # Per-verdict-type divergence breakdown
        divergence_breakdown: dict[str, int] = {}
        for v in self._verdicts:
            if v.verdict_diverged:
                key = f"{v.current_verdict}->{v.proposed_verdict}"
                divergence_breakdown[key] = divergence_breakdown.get(key, 0) + 1

        passed = (
            self._invariant_violations <= self.max_invariant_violations
            and self.divergence_rate <= self.max_divergence_rate
        )

        return {
            "proposal_id": self.proposal_id,
            "total_evaluations": self.total_evaluations,
            "invariant_violations": self._invariant_violations,
            "divergences": self._divergences,
            "divergence_rate": round(self.divergence_rate, 4),
            "max_divergence_rate": self.max_divergence_rate,
            "mean_composite_delta": round(mean_delta, 4),
            "divergence_breakdown": divergence_breakdown,
            "shadow_start": self.shadow_start.isoformat(),
            "shadow_end": self.shadow_end.isoformat(),
            "passed": passed,
            "status": "passed" if passed else "failed",
        }

    def proposed_constitution(self) -> dict[str, Any]:
        """Build a constitution dict with the proposed drive weights."""
        return {
            "drive_coherence": self.proposed_drives.get("coherence", 1.0),
            "drive_care": self.proposed_drives.get("care", 1.0),
            "drive_growth": self.proposed_drives.get("growth", 1.0),
            "drive_honesty": self.proposed_drives.get("honesty", 1.0),
        }


# ── Shadow evaluation (called during every review) ───────────────────

def evaluate_shadow(
    alignment: DriveAlignmentVector,
    intent: Intent,
    autonomy_level: int,
    current_constitution: dict[str, Any],
    tracker: ShadowTracker,
    hypotheses: list[dict[str, Any]] | None = None,
    memory: ConstitutionalMemory | None = None,
) -> ShadowVerdict:
    """
    Run the verdict engine with the proposed constitution weights and
    compare against the current verdict.  Called during every review()
    while a shadow period is active.

    This is a CPU-only operation — the expensive drive evaluation has
    already been done; we just re-run compute_verdict with different weights.
    """
    # Current verdict
    current_check = compute_verdict(
        alignment, intent, autonomy_level, current_constitution,
        hypotheses=hypotheses, memory=memory,
    )

    # Proposed verdict with the amendment's weights
    proposed_constitution = tracker.proposed_constitution()
    proposed_check = compute_verdict(
        alignment, intent, autonomy_level, proposed_constitution,
        hypotheses=hypotheses, memory=memory,
    )

    shadow = ShadowVerdict(
        intent_id=intent.id,
        current_verdict=current_check.verdict.value,
        proposed_verdict=proposed_check.verdict.value,
        current_composite=alignment.composite,
        proposed_composite=_compute_composite_with_weights(alignment, proposed_constitution),
    )

    tracker.record(shadow)
    return shadow


def _compute_composite_with_weights(
    alignment: DriveAlignmentVector,
    constitution: dict[str, Any],
) -> float:
    """Compute the weighted composite score using specific constitution weights."""
    coherence_weight: float = float(constitution.get("drive_coherence", 1.0))
    care_weight: float = float(constitution.get("drive_care", 1.0))
    growth_weight: float = float(constitution.get("drive_growth", 1.0))
    honesty_weight: float = float(constitution.get("drive_honesty", 1.0))

    w_coherence = coherence_weight * 0.8
    w_care = care_weight * 1.5
    w_growth = growth_weight * 0.7
    w_honesty = honesty_weight * 1.3
    total_weight = w_coherence + w_care + w_growth + w_honesty
    if total_weight == 0:
        return 0.0

    return (
        w_coherence * alignment.coherence
        + w_care * alignment.care
        + w_growth * alignment.growth
        + w_honesty * alignment.honesty
    ) / total_weight


# ── Full pipeline functions ──────────────────────────────────────────

async def submit_amendment(
    neo4j: Neo4jClient,
    proposed_drives: dict[str, float],
    title: str,
    description: str,
    rationale: str,
    proposer_id: str,
    evidence_hypothesis_ids: list[str],
    governance_config: GovernanceConfig,
    min_evidence_count: int = 2,
    min_evidence_confidence: float = 2.5,
) -> dict[str, Any]:
    """
    Stage 1+2: Submit and validate an amendment proposal.

    Requirements beyond structural validity:
      - At least `min_evidence_count` supporting Evo hypothesis IDs
      - Each hypothesis must have evidence_score >= `min_evidence_confidence`
      - No active cooldown
      - No other amendment currently in shadow or voting

    Returns {accepted, proposal_id, ...} or {accepted: false, reason: ...}.
    """
    # Structural validation
    valid, reason = validate_amendment_proposal(proposed_drives)
    if not valid:
        return {"accepted": False, "reason": reason}

    # Evidence requirement
    if len(evidence_hypothesis_ids) < min_evidence_count:
        return {
            "accepted": False,
            "reason": (
                f"Insufficient evidence: {len(evidence_hypothesis_ids)} hypothesis IDs "
                f"provided, minimum {min_evidence_count} required."
            ),
        }

    # Validate evidence quality
    validated_evidence = await _validate_evidence(
        neo4j, evidence_hypothesis_ids, min_evidence_confidence,
    )
    if not validated_evidence["valid"]:
        return {"accepted": False, "reason": validated_evidence["reason"]}

    # Cooldown check
    from systems.equor.amendment import check_amendment_cooldown

    on_cooldown, next_date = await check_amendment_cooldown(
        neo4j, governance_config.amendment_cooldown_days,
    )
    if on_cooldown:
        return {
            "accepted": False,
            "reason": f"Amendment cooldown active. Next proposal allowed after {next_date}.",
        }

    # No concurrent amendments in shadow or voting
    active = await _get_active_amendment(neo4j)
    if active is not None:
        return {
            "accepted": False,
            "reason": (
                f"Another amendment ({active['id']}) is currently in "
                f"'{active['status']}' status. Only one amendment may be "
                f"active at a time."
            ),
        }

    # Fetch current drives for recording delta
    current_drives = await _get_current_drives(neo4j)

    now = utc_now()
    deliberation_ends = now + timedelta(days=governance_config.amendment_deliberation_days)
    proposal_id = new_id()

    details = {
        "title": title,
        "description": description,
        "rationale": rationale,
        "proposed_drives": proposed_drives,
        "current_drives": current_drives,
        "evidence_hypothesis_ids": evidence_hypothesis_ids,
        "evidence_summaries": validated_evidence["summaries"],
        "deliberation_ends": deliberation_ends.isoformat(),
        "supermajority_required": governance_config.amendment_supermajority,
        "quorum_required": governance_config.amendment_quorum,
        "votes_for": 0,
        "votes_against": 0,
        "votes_abstain": 0,
        "status": AmendmentStatus.DELIBERATION.value,
    }

    await neo4j.execute_write(
        """
        CREATE (g:GovernanceRecord {
            id: $id,
            event_type: 'amendment_proposed',
            timestamp: datetime($now),
            details_json: $details_json,
            actor: $proposer,
            outcome: 'deliberation',
            amendment_status: $status
        })
        """,
        {
            "id": proposal_id,
            "now": now.isoformat(),
            "details_json": json.dumps(details),
            "proposer": proposer_id,
            "status": AmendmentStatus.DELIBERATION.value,
        },
    )

    logger.info(
        "amendment_submitted",
        proposal_id=proposal_id,
        title=title,
        proposed_drives=proposed_drives,
        evidence_count=len(evidence_hypothesis_ids),
        deliberation_ends=deliberation_ends.isoformat(),
    )

    return {
        "accepted": True,
        "proposal_id": proposal_id,
        "status": AmendmentStatus.DELIBERATION.value,
        "deliberation_ends": deliberation_ends.isoformat(),
        "evidence_validated": len(evidence_hypothesis_ids),
    }


async def start_shadow_period(
    neo4j: Neo4jClient,
    proposal_id: str,
    shadow_days: int = 7,
    max_divergence_rate: float = 0.15,
) -> dict[str, Any]:
    """
    Stage 3: Transition an amendment from deliberation to shadow mode.

    Only callable after the deliberation period has elapsed.
    Creates a ShadowTracker that will be used during every review().
    """
    proposal = await _get_proposal(neo4j, proposal_id)
    if proposal is None:
        return {"started": False, "reason": "Proposal not found."}

    details = json.loads(proposal.get("details_json", "{}"))
    status = details.get("status", proposal.get("amendment_status"))

    if status != AmendmentStatus.DELIBERATION.value:
        return {
            "started": False,
            "reason": f"Proposal is in '{status}' status, not 'deliberation'.",
        }

    # Verify deliberation period has elapsed
    deliberation_ends = details.get("deliberation_ends")
    if deliberation_ends:
        end_dt = datetime.fromisoformat(deliberation_ends)
        if not end_dt.tzinfo:
            end_dt = end_dt.replace(tzinfo=UTC)
        if utc_now() < end_dt:
            return {
                "started": False,
                "reason": f"Deliberation period ends {deliberation_ends}. Cannot start shadow yet.",
            }

    now = utc_now()
    shadow_end = now + timedelta(days=shadow_days)

    # Update proposal status
    details["status"] = AmendmentStatus.SHADOW.value
    details["shadow_start"] = now.isoformat()
    details["shadow_end"] = shadow_end.isoformat()
    details["shadow_max_divergence_rate"] = max_divergence_rate

    await neo4j.execute_write(
        """
        MATCH (g:GovernanceRecord {id: $id})
        SET g.details_json = $details_json,
            g.amendment_status = $status
        """,
        {
            "id": proposal_id,
            "details_json": json.dumps(details),
            "status": AmendmentStatus.SHADOW.value,
        },
    )

    tracker = ShadowTracker(
        proposal_id=proposal_id,
        proposed_drives=details["proposed_drives"],
        shadow_start=now,
        shadow_end=shadow_end,
        max_divergence_rate=max_divergence_rate,
    )

    logger.info(
        "amendment_shadow_started",
        proposal_id=proposal_id,
        shadow_end=shadow_end.isoformat(),
        max_divergence_rate=max_divergence_rate,
    )

    return {
        "started": True,
        "proposal_id": proposal_id,
        "shadow_start": now.isoformat(),
        "shadow_end": shadow_end.isoformat(),
        "tracker": tracker,
    }


async def complete_shadow_period(
    neo4j: Neo4jClient,
    tracker: ShadowTracker,
) -> dict[str, Any]:
    """
    Stage 4: Evaluate the shadow period results and determine pass/fail.

    Called when the shadow period expires or when an invariant violation
    triggers early termination.
    """
    report = tracker.compute_report()
    passed = report["passed"]
    new_status = AmendmentStatus.SHADOW_PASSED if passed else AmendmentStatus.SHADOW_FAILED

    # Update proposal
    proposal = await _get_proposal(neo4j, tracker.proposal_id)
    if proposal is None:
        return {"completed": False, "reason": "Proposal not found."}

    details = json.loads(proposal.get("details_json", "{}"))
    details["status"] = new_status.value
    details["shadow_report"] = report

    await neo4j.execute_write(
        """
        MATCH (g:GovernanceRecord {id: $id})
        SET g.details_json = $details_json,
            g.amendment_status = $status
        """,
        {
            "id": tracker.proposal_id,
            "details_json": json.dumps(details),
            "status": new_status.value,
        },
    )

    # Store the shadow report as its own governance record for audit
    await neo4j.execute_write(
        """
        CREATE (g:GovernanceRecord {
            id: $id,
            event_type: 'amendment_shadow_complete',
            timestamp: datetime($now),
            details_json: $details_json,
            actor: 'equor',
            outcome: $outcome
        })
        """,
        {
            "id": new_id(),
            "now": utc_now().isoformat(),
            "details_json": json.dumps(report),
            "outcome": "shadow_passed" if passed else "shadow_failed",
        },
    )

    logger.info(
        "amendment_shadow_complete",
        proposal_id=tracker.proposal_id,
        passed=passed,
        total_evaluations=report["total_evaluations"],
        invariant_violations=report["invariant_violations"],
        divergence_rate=report["divergence_rate"],
    )

    return {
        "completed": True,
        "proposal_id": tracker.proposal_id,
        "passed": passed,
        "report": report,
    }


async def open_voting(
    neo4j: Neo4jClient,
    proposal_id: str,
) -> dict[str, Any]:
    """
    Stage 5a: Open the amendment for community voting.
    Only valid if shadow period passed.
    """
    proposal = await _get_proposal(neo4j, proposal_id)
    if proposal is None:
        return {"opened": False, "reason": "Proposal not found."}

    details = json.loads(proposal.get("details_json", "{}"))
    status = details.get("status")

    if status != AmendmentStatus.SHADOW_PASSED.value:
        return {
            "opened": False,
            "reason": f"Proposal is in '{status}' status. Shadow must pass before voting.",
        }

    details["status"] = AmendmentStatus.VOTING.value
    details["voting_opened_at"] = utc_now().isoformat()

    await neo4j.execute_write(
        """
        MATCH (g:GovernanceRecord {id: $id})
        SET g.details_json = $details_json,
            g.amendment_status = $status
        """,
        {
            "id": proposal_id,
            "details_json": json.dumps(details),
            "status": AmendmentStatus.VOTING.value,
        },
    )

    logger.info("amendment_voting_opened", proposal_id=proposal_id)
    return {"opened": True, "proposal_id": proposal_id}


async def cast_vote(
    neo4j: Neo4jClient,
    proposal_id: str,
    voter_id: str,
    vote: str,  # "for" | "against" | "abstain"
) -> dict[str, Any]:
    """
    Stage 5b: Cast a vote on an amendment.
    """
    if vote not in ("for", "against", "abstain"):
        return {"recorded": False, "reason": f"Invalid vote '{vote}'. Must be 'for', 'against', or 'abstain'."}

    proposal = await _get_proposal(neo4j, proposal_id)
    if proposal is None:
        return {"recorded": False, "reason": "Proposal not found."}

    details = json.loads(proposal.get("details_json", "{}"))
    if details.get("status") != AmendmentStatus.VOTING.value:
        return {"recorded": False, "reason": "Proposal is not in voting status."}

    # Record the vote
    vote_key = f"votes_{vote}"
    details[vote_key] = details.get(vote_key, 0) + 1

    # Track individual votes to prevent double-voting
    voters = details.setdefault("voters", {})
    if voter_id in voters:
        return {"recorded": False, "reason": "This voter has already voted on this proposal."}
    voters[voter_id] = vote

    await neo4j.execute_write(
        """
        MATCH (g:GovernanceRecord {id: $id})
        SET g.details_json = $details_json
        """,
        {
            "id": proposal_id,
            "details_json": json.dumps(details),
        },
    )

    logger.info(
        "amendment_vote_cast",
        proposal_id=proposal_id,
        voter_id=voter_id,
        vote=vote,
    )

    return {
        "recorded": True,
        "current_tally": {
            "for": details.get("votes_for", 0),
            "against": details.get("votes_against", 0),
            "abstain": details.get("votes_abstain", 0),
        },
    }


async def tally_votes(
    neo4j: Neo4jClient,
    proposal_id: str,
    total_eligible_voters: int,
) -> dict[str, Any]:
    """
    Stage 5c: Tally votes and determine if the amendment passes.

    Requires:
      - Quorum: `quorum_required` fraction of eligible voters must have voted
      - Supermajority: `supermajority_required` fraction of non-abstain votes must be 'for'
    """
    proposal = await _get_proposal(neo4j, proposal_id)
    if proposal is None:
        return {"tallied": False, "reason": "Proposal not found."}

    details = json.loads(proposal.get("details_json", "{}"))
    if details.get("status") != AmendmentStatus.VOTING.value:
        return {"tallied": False, "reason": "Proposal is not in voting status."}

    votes_for = details.get("votes_for", 0)
    votes_against = details.get("votes_against", 0)
    votes_abstain = details.get("votes_abstain", 0)
    total_votes = votes_for + votes_against + votes_abstain

    quorum_required = details.get("quorum_required", 0.60)
    supermajority_required = details.get("supermajority_required", 0.75)

    # Quorum check
    quorum_met = total_eligible_voters > 0 and (
        total_votes / total_eligible_voters >= quorum_required
    )

    # Supermajority check (abstentions don't count toward the decision)
    decisive_votes = votes_for + votes_against
    supermajority_met = decisive_votes > 0 and (
        votes_for / decisive_votes >= supermajority_required
    )

    passed = quorum_met and supermajority_met
    new_status = AmendmentStatus.PASSED if passed else AmendmentStatus.FAILED

    details["status"] = new_status.value
    details["tally"] = {
        "votes_for": votes_for,
        "votes_against": votes_against,
        "votes_abstain": votes_abstain,
        "total_eligible": total_eligible_voters,
        "quorum_required": quorum_required,
        "quorum_met": quorum_met,
        "supermajority_required": supermajority_required,
        "supermajority_met": supermajority_met,
        "passed": passed,
    }

    await neo4j.execute_write(
        """
        MATCH (g:GovernanceRecord {id: $id})
        SET g.details_json = $details_json,
            g.amendment_status = $status,
            g.outcome = $outcome
        """,
        {
            "id": proposal_id,
            "details_json": json.dumps(details),
            "status": new_status.value,
            "outcome": "passed" if passed else "failed",
        },
    )

    # Record the vote result as a governance event
    await neo4j.execute_write(
        """
        CREATE (g:GovernanceRecord {
            id: $id,
            event_type: $event_type,
            timestamp: datetime($now),
            details_json: $details_json,
            actor: 'governance',
            outcome: $outcome
        })
        """,
        {
            "id": new_id(),
            "event_type": "amendment_passed" if passed else "amendment_failed",
            "now": utc_now().isoformat(),
            "details_json": json.dumps({
                "proposal_id": proposal_id,
                "tally": details["tally"],
            }),
            "outcome": "passed" if passed else "failed",
        },
    )

    logger.info(
        "amendment_tally_complete",
        proposal_id=proposal_id,
        passed=passed,
        votes_for=votes_for,
        votes_against=votes_against,
        quorum_met=quorum_met,
        supermajority_met=supermajority_met,
    )

    return {
        "tallied": True,
        "proposal_id": proposal_id,
        "passed": passed,
        "tally": details["tally"],
    }


async def adopt_amendment(
    neo4j: Neo4jClient,
    proposal_id: str,
) -> dict[str, Any]:
    """
    Stage 6: Apply a passed amendment to the Constitution node.

    Only callable after the vote has passed. Increments the constitution
    version, updates drive weights, and preserves the full amendment history.
    """
    proposal = await _get_proposal(neo4j, proposal_id)
    if proposal is None:
        return {"adopted": False, "reason": "Proposal not found."}

    details = json.loads(proposal.get("details_json", "{}"))
    if details.get("status") != AmendmentStatus.PASSED.value:
        return {
            "adopted": False,
            "reason": f"Proposal is in '{details.get('status')}' status. Must be 'passed' to adopt.",
        }

    proposed_drives = details["proposed_drives"]
    now = utc_now()

    # Update Constitution node
    amendment_record = json.dumps({
        "date": now.isoformat(),
        "proposal_id": proposal_id,
        "new_values": proposed_drives,
        "shadow_report": details.get("shadow_report", {}),
        "tally": details.get("tally", {}),
    })

    await neo4j.execute_write(
        """
        MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)
        SET c.drive_coherence = $coherence,
            c.drive_care = $care,
            c.drive_growth = $growth,
            c.drive_honesty = $honesty,
            c.version = c.version + 1,
            c.last_amended = datetime($now),
            c.amendments = c.amendments + [$amendment_json]
        """,
        {
            "coherence": proposed_drives["coherence"],
            "care": proposed_drives["care"],
            "growth": proposed_drives["growth"],
            "honesty": proposed_drives["honesty"],
            "now": now.isoformat(),
            "amendment_json": amendment_record,
        },
    )

    # Update proposal status
    details["status"] = AmendmentStatus.ADOPTED.value
    details["adopted_at"] = now.isoformat()

    await neo4j.execute_write(
        """
        MATCH (g:GovernanceRecord {id: $id})
        SET g.details_json = $details_json,
            g.amendment_status = $status,
            g.outcome = 'adopted'
        """,
        {
            "id": proposal_id,
            "details_json": json.dumps(details),
            "status": AmendmentStatus.ADOPTED.value,
        },
    )

    # Ratification record
    await neo4j.execute_write(
        """
        CREATE (g:GovernanceRecord {
            id: $id,
            event_type: 'amendment_ratified',
            timestamp: datetime($now),
            details_json: $details_json,
            actor: 'governance',
            outcome: 'ratified'
        })
        """,
        {
            "id": new_id(),
            "now": now.isoformat(),
            "details_json": json.dumps({
                "proposal_id": proposal_id,
                "new_drives": proposed_drives,
            }),
        },
    )

    logger.info(
        "amendment_adopted",
        proposal_id=proposal_id,
        new_drives=proposed_drives,
    )

    return {
        "adopted": True,
        "proposal_id": proposal_id,
        "new_drives": proposed_drives,
    }


async def get_amendment_status(
    neo4j: Neo4jClient,
    proposal_id: str,
) -> dict[str, Any] | None:
    """Get the current status of an amendment proposal."""
    proposal = await _get_proposal(neo4j, proposal_id)
    if proposal is None:
        return None

    details = json.loads(proposal.get("details_json", "{}"))
    return {
        "proposal_id": proposal_id,
        "status": details.get("status"),
        "title": details.get("title"),
        "description": details.get("description"),
        "proposed_drives": details.get("proposed_drives"),
        "current_drives": details.get("current_drives"),
        "evidence_count": len(details.get("evidence_hypothesis_ids", [])),
        "shadow_report": details.get("shadow_report"),
        "tally": details.get("tally"),
    }


# ── Internal helpers ─────────────────────────────────────────────────


async def _validate_evidence(
    neo4j: Neo4jClient,
    hypothesis_ids: list[str],
    min_confidence: float,
) -> dict[str, Any]:
    """
    Validate that the supporting hypotheses exist and have sufficient
    evidence scores.
    """
    if not hypothesis_ids:
        return {"valid": False, "reason": "No hypothesis IDs provided."}

    results = await neo4j.execute_read(
        """
        UNWIND $ids AS hid
        OPTIONAL MATCH (h:Hypothesis {id: hid})
        RETURN hid AS requested_id,
               h.id AS found_id,
               h.statement AS statement,
               h.evidence_score AS evidence_score,
               h.status AS status
        """,
        {"ids": hypothesis_ids},
    )

    summaries: list[dict[str, Any]] = []
    missing: list[str] = []
    weak: list[str] = []

    for row in results:
        requested = row["requested_id"]
        if row["found_id"] is None:
            missing.append(requested)
            continue

        score = row["evidence_score"] or 0.0
        status = row["status"] or "unknown"

        if score < min_confidence:
            weak.append(
                f"{requested} (score={score:.1f}, need>={min_confidence})"
            )

        if status not in ("supported", "integrated"):
            weak.append(f"{requested} (status='{status}', need supported/integrated)")

        summaries.append({
            "id": requested,
            "statement": row["statement"],
            "evidence_score": score,
            "status": status,
        })

    if missing:
        return {
            "valid": False,
            "reason": f"Hypothesis IDs not found: {', '.join(missing)}",
            "summaries": summaries,
        }

    if weak:
        return {
            "valid": False,
            "reason": f"Insufficient evidence quality: {'; '.join(weak)}",
            "summaries": summaries,
        }

    return {"valid": True, "summaries": summaries}


async def _get_proposal(
    neo4j: Neo4jClient,
    proposal_id: str,
) -> dict[str, Any] | None:
    """Fetch a single amendment proposal by ID."""
    results = await neo4j.execute_read(
        """
        MATCH (g:GovernanceRecord {id: $id, event_type: 'amendment_proposed'})
        RETURN g.id AS id, g.details_json AS details_json,
               g.amendment_status AS amendment_status
        """,
        {"id": proposal_id},
    )
    if not results:
        return None
    return dict(results[0])


async def _get_active_amendment(
    neo4j: Neo4jClient,
) -> dict[str, Any] | None:
    """Check if there's an amendment currently in an active lifecycle stage."""
    active_statuses = [
        AmendmentStatus.DELIBERATION.value,
        AmendmentStatus.SHADOW.value,
        AmendmentStatus.VOTING.value,
    ]
    results = await neo4j.execute_read(
        """
        MATCH (g:GovernanceRecord {event_type: 'amendment_proposed'})
        WHERE g.amendment_status IN $statuses
        RETURN g.id AS id, g.amendment_status AS status
        ORDER BY g.timestamp DESC
        LIMIT 1
        """,
        {"statuses": active_statuses},
    )
    if not results:
        return None
    return dict(results[0])


async def _get_current_drives(neo4j: Neo4jClient) -> dict[str, float]:
    """Fetch the current constitutional drive weights."""
    results = await neo4j.execute_read(
        "MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution) RETURN c"
    )
    if results:
        c = results[0]["c"]
        return {
            "coherence": c.get("drive_coherence", 1.0),
            "care": c.get("drive_care", 1.0),
            "growth": c.get("drive_growth", 1.0),
            "honesty": c.get("drive_honesty", 1.0),
        }
    return {"coherence": 1.0, "care": 1.0, "growth": 1.0, "honesty": 1.0}
