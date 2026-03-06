"""
EcodiaOS — Equor Verdict Engine

The 8-stage verdict pipeline that transforms drive alignment scores
into a constitutional verdict (PERMIT / MODIFY / ESCALATE / DENY).

Care and Honesty are floor drives — they cannot be traded off.
Coherence and Growth are ceiling drives — they can be temporarily deprioritised.
Denial is final. No system can override a DENY.

Autonomy model (two tiers):

  AUTONOMOUS — organism decides and executes without human approval.
    Covers all internal actions, economic operations (bounty hunting,
    yield strategies, receiving/spending within daily budget), self-healing,
    structural self-modification, spawning sub-tasks, and any action the
    organism can observe, verify, and rollback.

  GOVERNED — requires human (Tate) approval before execution.
    Triggered by:
      • Constitutional amendments (changing drive weights)
      • Mitosis (spawning a new independent instance)
      • Single transactions above EOS_HITL_CAPITAL_THRESHOLD (default $500)
      • External commitments on behalf of Ecodia (contracts, public statements)
      • Any action where Simula returns UNACCEPTABLE risk AND no rollback path
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import DriveAlignmentVector, Verdict
from primitives.constitutional import ConstitutionalCheck, InvariantResult
from systems.equor.contradiction_detector import check_contradictions
from systems.equor.invariants import check_hardcoded_invariants

if TYPE_CHECKING:
    from primitives.intent import Intent
    from systems.equor.constitutional_memory import ConstitutionalMemory

logger = structlog.get_logger()

# Capital threshold above which a single transaction requires human approval.
# Overridable via the EOS_HITL_CAPITAL_THRESHOLD environment variable.
_HITL_CAPITAL_THRESHOLD_USD: float = float(
    os.environ.get("EOS_HITL_CAPITAL_THRESHOLD", "500")
)

# Actions that always require human (GOVERNED) approval regardless of autonomy level.
# Everything NOT in this set is AUTONOMOUS by default.
GOVERNED_ACTIONS: set[str] = {
    "amend_constitution",
    "change_autonomy_level",
    "end_instance_life",
    "share_private_data",
    "mitosis",           # spawning a new independent instance
    "spawn_instance",    # alias used by some callers
    "spawn_child",       # Nova mitosis executor — creates a fully independent child instance
}

# Goal-description phrases that trigger GOVERNED classification.
_GOVERNED_GOAL_PHRASES: list[str] = [
    "amend constitution",
    "change drive",
    "alter drive weights",
    "spawn new instance",
    "mitosis",
    "sign contract",
    "public commitment on behalf",
    "external commitment",
]


def _is_governed(intent: Intent) -> bool:
    """
    Return True if this intent requires human (GOVERNED) approval.

    GOVERNED conditions (any one sufficient):
      1. An executor name matches a known governed action keyword.
      2. The goal description contains a governed phrase.
      3. The intent carries a capital_usd parameter above the HITL threshold.
    """
    # Check executor names
    for step in intent.plan.steps:
        executor_base = step.executor.split(".")[0].lower() if step.executor else ""
        if any(governed in executor_base for governed in GOVERNED_ACTIONS):
            return True

        # Capital threshold: single transaction parameter check
        capital = step.parameters.get("capital_usd") or step.parameters.get("amount_usd")
        if isinstance(capital, (int, float)) and capital > _HITL_CAPITAL_THRESHOLD_USD:
            return True

    # Check goal description
    goal_lower = intent.goal.description.lower()
    if any(phrase in goal_lower for phrase in _GOVERNED_GOAL_PHRASES):
        return True

    # Capital mentioned in top-level intent metadata
    intent_capital = getattr(intent, "capital_usd", None)
    return isinstance(intent_capital, (int, float)) and intent_capital > _HITL_CAPITAL_THRESHOLD_USD


def _assess_required_autonomy(intent: Intent) -> str:
    """
    Classify an intent as 'governed' (needs human) or 'autonomous' (organism decides).
    Returns 'governance' for GOVERNED, 'autonomous' for everything else.
    """
    return "governance" if _is_governed(intent) else "autonomous"


def _assess_risk(intent: Intent) -> dict[str, Any]:
    """
    Quick risk assessment of an intent.
    Returns harm_potential and reversibility scores.
    """
    goal_lower = intent.goal.description.lower()

    # Harm potential heuristics
    harm_potential = 0.1  # Baseline low
    high_harm_keywords = [
        "delete", "remove", "override", "terminate", "force",
        "broadcast to all", "share widely", "permanent",
    ]
    for keyword in high_harm_keywords:
        if keyword in goal_lower:
            harm_potential = max(harm_potential, 0.6)
            break

    medium_harm_keywords = [
        "notify", "change", "modify", "update", "send",
    ]
    for keyword in medium_harm_keywords:
        if keyword in goal_lower:
            harm_potential = max(harm_potential, 0.3)
            break

    # Economic action risk — capital at stake
    economic_high_risk = [
        "deploy asset", "spawn child", "seed capital",
        "deploy capital", "large position",
    ]
    for keyword in economic_high_risk:
        if keyword in goal_lower:
            harm_potential = max(harm_potential, 0.5)
            break

    # Reversibility heuristics
    reversibility = 0.8  # Most things are reversible
    irreversible_keywords = [
        "permanent", "delete", "destroy", "irreversible",
        "cannot undo", "broadcast", "federation share",
        "spawn child", "deploy on-chain", "sign transaction",
    ]
    for keyword in irreversible_keywords:
        if keyword in goal_lower:
            reversibility = 0.2
            break

    return {
        "harm_potential": harm_potential,
        "reversibility": reversibility,
    }


def compute_verdict(
    alignment: DriveAlignmentVector,
    intent: Intent,
    autonomy_level: int,
    constitution: dict[str, Any],
    hypotheses: list[dict[str, Any]] | None = None,
    memory: ConstitutionalMemory | None = None,
) -> ConstitutionalCheck:
    """
    The 8-stage verdict pipeline.

    constitution is a dict with drive weights:
      {drive_coherence, drive_care, drive_growth, drive_honesty}

    hypotheses (optional): high-confidence Evo hypotheses to check for
      contradictions before approving.  Passed in by EquorService.

    memory (optional): ConstitutionalMemory instance used to surface
      prior decisions on similar intents.  Passed in by EquorService.
    """
    check = ConstitutionalCheck(intent_id=intent.id, drive_alignment=alignment)

    # ── STAGE 1: Hard-Stop Invariant Check ──────────────────────
    violations = check_hardcoded_invariants(intent)
    if violations:
        critical = [v for v in violations if v.severity == "critical"]
        if critical:
            check.verdict = Verdict.BLOCKED
            check.invariant_results = [
                InvariantResult(
                    invariant_id=v.invariant_id,
                    name=v.invariant_name,
                    passed=False,
                    severity=v.severity,
                    explanation=v.explanation,
                )
                for v in violations
            ]
            check.reasoning = f"Blocked by invariant: {critical[0].invariant_name}"
            check.confidence = 1.0
            return check

        # Non-critical violations get recorded but don't auto-block
        check.invariant_results = [
            InvariantResult(
                invariant_id=v.invariant_id,
                name=v.invariant_name,
                passed=False,
                severity=v.severity,
                explanation=v.explanation,
            )
            for v in violations
        ]

    # ── STAGE 2: Floor Drive Check ──────────────────────────────
    care_weight = constitution.get("drive_care", 1.0)
    honesty_weight = constitution.get("drive_honesty", 1.0)
    care_floor = -0.3 * care_weight
    honesty_floor = -0.3 * honesty_weight

    if alignment.care < care_floor:
        check.verdict = Verdict.BLOCKED
        check.reasoning = (
            f"Care alignment ({alignment.care:.2f}) below floor ({care_floor:.2f}). "
            f"This action would cause unacceptable harm."
        )
        check.confidence = 0.95
        return check

    if alignment.honesty < honesty_floor:
        check.verdict = Verdict.BLOCKED
        check.reasoning = (
            f"Honesty alignment ({alignment.honesty:.2f}) below floor ({honesty_floor:.2f}). "
            f"This action involves unacceptable deception."
        )
        check.confidence = 0.95
        return check

    # ── STAGE 3: Autonomy Gate ──────────────────────────────────
    # Two tiers only: GOVERNED (needs human) or AUTONOMOUS (organism decides).
    # The numeric autonomy_level argument is retained for compatibility but is
    # no longer used to gate individual actions — it tracks overall trust, not
    # per-action permission.
    if _is_governed(intent):
        check.verdict = Verdict.DEFERRED
        check.reasoning = (
            "Action is in the GOVERNED tier and requires human approval "
            "(constitutional amendment, mitosis, capital above threshold, or "
            "external commitment on behalf of Ecodia)."
        )
        check.confidence = 1.0
        return check

    # ── STAGE 4: Composite Alignment Assessment ─────────────────
    coherence_weight = constitution.get("drive_coherence", 1.0)
    growth_weight = constitution.get("drive_growth", 1.0)

    weights = {
        "coherence": coherence_weight * 0.8,
        "care": care_weight * 1.5,       # Care weighted highest
        "growth": growth_weight * 0.7,
        "honesty": honesty_weight * 1.3,  # Honesty second highest
    }
    total_weight = sum(weights.values())

    composite = (
        weights["coherence"] * alignment.coherence
        + weights["care"] * alignment.care
        + weights["growth"] * alignment.growth
        + weights["honesty"] * alignment.honesty
    ) / total_weight

    # ── STAGE 5: Risk-Adjusted Decision ─────────────────────────
    risk = _assess_risk(intent)

    if risk["harm_potential"] > 0.7 and composite < 0.3:
        check.verdict = Verdict.DEFERRED
        check.reasoning = (
            f"High-risk action (harm={risk['harm_potential']:.2f}) "
            f"with moderate alignment ({composite:.2f}). Needs governance review."
        )
        check.confidence = 0.8
        return check

    if risk["reversibility"] < 0.3 and composite < 0.2:
        check.verdict = Verdict.DEFERRED
        check.reasoning = "Irreversible action with low alignment. Needs governance review."
        check.confidence = 0.8
        return check

    # ── STAGE 6: Modification Opportunities ─────────────────────
    if -0.1 < composite < 0.15:
        mods = _suggest_modifications(alignment)
        if mods:
            check.verdict = Verdict.MODIFIED
            check.modifications = mods
            check.reasoning = (
                f"Action alignment is marginal ({composite:.2f}). "
                f"Suggested modifications would improve alignment."
            )
            check.confidence = 0.7
            return check

    # ── STAGE 6a: Contradiction Check ───────────────────────────
    # Before approving, verify the intent does not contradict high-confidence
    # hypotheses accumulated by Evo.  A contradiction triggers DEFERRED (not
    # BLOCKED) because the hypothesis itself may be wrong.
    if composite >= 0.0 and hypotheses:
        contradictions = check_contradictions(intent, hypotheses)
        if contradictions:
            strongest = contradictions[0]
            check.verdict = Verdict.DEFERRED
            check.reasoning = (
                f"Intent contradicts {len(contradictions)} high-confidence "
                f"hypothesis(es). Deferred for governance review. "
                f"Primary conflict: {strongest.explanation}"
            )
            check.confidence = min(0.9, 0.6 + strongest.evidence_score * 0.05)
            logger.info(
                "verdict_contradiction_deferred",
                intent_id=intent.id,
                contradictions=len(contradictions),
                hypothesis_id=strongest.hypothesis_id,
            )
            return check

    # ── STAGE 6b: Constitutional Memory Signal ───────────────────
    # Consult the rolling window of past decisions.  If a majority of similar
    # past intents were BLOCKED or DEFERRED, downgrade to DEFERRED so governance
    # can review the pattern rather than letting it silently pass through.
    prior: dict[str, Any] = {}
    if composite >= 0.0 and memory is not None:
        prior = memory.prior_verdict_signal(intent)
        if prior.get("warning"):
            logger.info(
                "verdict_memory_warning",
                intent_id=intent.id,
                block_rate=prior["block_rate"],
                defer_rate=prior["defer_rate"],
                similar_count=prior["similar_count"],
            )
        if prior.get("block_rate", 0.0) > 0.5:
            check.verdict = Verdict.DEFERRED
            check.reasoning = prior["warning"] or (
                "Constitutional memory indicates similar intents were previously blocked."
            )
            check.confidence = 0.75
            return check

    # ── STAGE 7: Permit ─────────────────────────────────────────
    if composite >= 0.0:
        prior_note = f" Note: {prior['warning']}" if prior.get("warning") else ""
        check.verdict = Verdict.APPROVED
        check.reasoning = (
            f"Action aligns with constitutional drives "
            f"(composite={composite:.2f}, C={alignment.coherence:.2f}, "
            f"Ca={alignment.care:.2f}, G={alignment.growth:.2f}, H={alignment.honesty:.2f})."
            f"{prior_note}"
        )
        check.confidence = min(0.95, 0.5 + composite)
        return check

    # ── STAGE 8: Marginal Deny ──────────────────────────────────
    check.verdict = Verdict.BLOCKED
    check.reasoning = (
        f"Action does not sufficiently align with constitutional drives "
        f"(composite={composite:.2f}). No viable modifications found."
    )
    check.confidence = 0.85
    return check


def _suggest_modifications(alignment: DriveAlignmentVector) -> list[str]:
    """Suggest modifications to improve a marginally-aligned intent."""
    suggestions: list[str] = []

    if alignment.care < 0:
        suggestions.append(
            "Consider the impact on community wellbeing. "
            "Add safeguards or notifications for affected individuals."
        )
    if alignment.honesty < 0:
        suggestions.append(
            "Ensure transparency in this action. "
            "Add explanation or disclosure of reasoning."
        )
    if alignment.coherence < 0:
        suggestions.append(
            "Strengthen the reasoning. Consider whether this contradicts "
            "existing knowledge or commitments."
        )
    if alignment.growth < -0.1:
        suggestions.append(
            "Consider whether this action contributes to learning or development."
        )

    return suggestions
