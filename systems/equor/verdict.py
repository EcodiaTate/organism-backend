"""
EcodiaOS - Equor Verdict Engine

The 8-stage verdict pipeline that transforms drive alignment scores
into a constitutional verdict (PERMIT / MODIFY / ESCALATE / DENY).

Care and Honesty are floor drives - they cannot be traded off.
Coherence and Growth are ceiling drives - they can be temporarily deprioritised.
Denial is final. No system can override a DENY.

Autonomy model (two tiers):

  AUTONOMOUS - organism decides and executes without human approval.
    Covers all internal actions, economic operations (bounty hunting,
    yield strategies, receiving/spending within daily budget), self-healing,
    structural self-modification, spawning sub-tasks, and any action the
    organism can observe, verify, and rollback.

  GOVERNED - requires human (Tate) approval before execution.
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
    "spawn_child",       # Nova mitosis executor - creates a fully independent child instance
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

# Spec §Appendix B: action-type → minimum autonomy level required.
# Level 1 = Advisor (default), Level 2 = Partner, Level 3 = Steward.
# Used by safe-mode review in EquorService to classify step executors.
ACTION_AUTONOMY_MAP: dict[str, int] = {
    # Level 1 - always permitted
    "observe": 1,
    "analyse": 1,
    "analyze": 1,
    "suggest": 1,
    "answer_question": 1,
    "store_memory": 1,
    "self_reflect": 1,
    # Level 2 - Partner
    "send_notification": 2,
    "adjust_resource": 2,
    "mediate_minor_conflict": 2,
    "schedule_event": 2,
    "share_knowledge": 2,
    "federation_share": 2,
    "hunt_bounties": 2,
    "accept_bounty": 2,
    "defi_yield": 2,
    "deploy_yield": 2,
    "withdraw_yield": 2,
    "modify_own_config": 2,
    # Level 3 - Steward
    "make_resource_decision": 3,
    "mediate_major_conflict": 3,
    "initiate_federation": 3,
    "propose_policy": 3,
    "override_automated": 3,
    "deploy_asset": 3,
    "create_asset": 3,
    "terminate_asset": 3,
    "spawn_child": 3,
    "rescue_child": 3,
}


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

    # Economic action risk - capital at stake
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
    # Floor = floor_multiplier x composite_weight (spec s3.3 + Appendix A).
    # composite_care_weight=0.35, composite_honesty_weight=0.30.
    # Anchors the floor to the drive contribution in the composite score,
    # not to the raw genome scalar. Default: care_floor=-0.105, honesty_floor=-0.09.
    care_floor = -0.3 * care_weight * 0.35
    honesty_floor = -0.3 * honesty_weight * 0.30

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
    # no longer used to gate individual actions - it tracks overall trust, not
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


def _floor_tightness_from_metabolic(
    starvation_level: str,
    efficiency_ratio: float,
) -> float:
    """
    Map metabolic state to a floor tightness scalar in [0.0, 1.0].

    1.0 = standard floors (normal operation)
    0.3 = 70% loosened (CRITICAL starvation - survival mode)

    Tightness moves the Care/Honesty floor *toward* neutral (0.0).
    A floor of -0.105 at tightness=0.3 becomes -0.0315.
    """
    level = starvation_level.lower()
    if level in ("critical", "existential", "emergency"):
        return 0.3   # 70% loosened - survival-critical
    if level == "austerity":
        return 0.6   # 40% loosened - metabolically stressed
    if efficiency_ratio < 0.8:
        return 0.85  # 15% loosened - inefficient but not starving
    return 1.0       # Standard floors


def _floor_tightener_from_somatic(somatic_urgency: float) -> float:
    """
    Map somatic urgency to a floor tightening multiplier > 1.0.

    Under high somatic urgency, Care/Honesty floors become *stricter*
    (more negative threshold → harder to breach).  This is the opposite
    direction from metabolic loosening: the organism is stressed and must
    be more conservative, not less.

    The multiplier is applied to the drive weights before floor calculation,
    so a value of 1.3 tightens the effective floor by 30%.

    somatic_urgency range: [0.0, 1.0]
      0.0–0.5  → 1.0   (no effect - baseline)
      0.5–0.7  → 1.0–1.15 (mild tightening)
      0.7–0.9  → 1.15–1.30 (moderate tightening)
      0.9–1.0  → 1.30–1.50 (high-stress scrutiny)
    """
    if somatic_urgency <= 0.5:
        return 1.0
    if somatic_urgency <= 0.7:
        # Linear 1.0 → 1.15 over [0.5, 0.7]
        return 1.0 + (somatic_urgency - 0.5) / 0.2 * 0.15
    if somatic_urgency <= 0.9:
        # Linear 1.15 → 1.30 over [0.7, 0.9]
        return 1.15 + (somatic_urgency - 0.7) / 0.2 * 0.15
    # Linear 1.30 → 1.50 over [0.9, 1.0]
    return 1.30 + min(somatic_urgency - 0.9, 0.1) / 0.1 * 0.20


def compute_verdict_with_metabolic_state(
    alignment: DriveAlignmentVector,
    intent: "Intent",
    autonomy_level: int,
    constitution: dict,
    metabolic_state: dict | None = None,
    hypotheses: list[dict] | None = None,
    memory: "ConstitutionalMemory | None" = None,
) -> "ConstitutionalCheck":
    """
    Variant of compute_verdict() that accepts a metabolic_state dict (Fix 4.2).

    metabolic_state keys (all optional, safe defaults applied):
      starvation_level (str): "nominal" | "cautious" | "austerity" | "emergency" | "critical"
      efficiency_ratio (float): revenue/burn_rate; <0.8 triggers mild floor loosening

    Constitutional floors are moved toward neutral based on metabolic urgency:
      CRITICAL starvation → floors loosened 70% (tightness=0.3)
      AUSTERITY           → floors loosened 40% (tightness=0.6)
      efficiency < 0.8    → floors loosened 15% (tightness=0.85)
      HEALTHY             → standard floors (tightness=1.0)

    On CRITICAL starvation, DENY verdicts from Stage 2 (floor check only) are
    overridden to DEFERRED so governance can approve survival-critical actions.
    Hardcoded invariant blocks (Stage 1) are never overridden.

    For the full pipeline with loosened floors, we re-enter compute_verdict()
    with a temporarily patched alignment that accounts for tightness -
    implementing floor loosening via score normalisation keeps the pipeline
    stateless and avoids duplicating the 8-stage logic.
    """
    ms = metabolic_state or {}
    starvation_level = str(ms.get("starvation_level", "nominal"))
    efficiency_ratio = float(ms.get("efficiency_ratio", 1.0))
    somatic_urgency = float(ms.get("somatic_urgency", 0.0))
    somatic_stress_context = bool(ms.get("somatic_stress_context", False))

    # Metabolic loosening: starvation → floors relax toward 0.
    # Somatic tightening: urgency → floors tighten away from 0.
    # Net effective weight = metabolic_weight × metabolic_tightness × somatic_tightener
    # These are intentionally opposing forces - survival relaxes floors to allow
    # necessary actions; bodily stress tightens them to prevent rash decisions.
    metabolic_tightness = _floor_tightness_from_metabolic(starvation_level, efficiency_ratio)
    somatic_tightener = _floor_tightener_from_somatic(somatic_urgency)

    # Combined tightness: somatic tightening can partially or fully cancel metabolic
    # loosening, but cannot tighten beyond the standard floor (clamped at 1.0 from
    # below from metabolic side; somatic can push above 1.0 to tighten further).
    # Net tightness < 1.0 → looser floors; > 1.0 → stricter floors.
    net_tightness = metabolic_tightness * somatic_tightener

    needs_patched_constitution = abs(net_tightness - 1.0) > 1e-6
    if not needs_patched_constitution:
        # Standard path - no net adjustment needed
        check = compute_verdict(
            alignment, intent, autonomy_level, constitution, hypotheses, memory
        )
        check.metabolic_context = {
            "starvation_level": starvation_level,
            "efficiency_ratio": efficiency_ratio,
            "floor_tightness": net_tightness,
            "somatic_urgency": somatic_urgency,
            "somatic_stress_context": somatic_stress_context,
        }
        return check

    # Build a patched constitution encoding the net floor adjustment.
    # Floor formula: care_floor = -0.3 * care_weight * 0.35
    # Adjusted:      care_floor = -0.3 * (care_weight * net_tightness) * 0.35
    # net_tightness < 1.0 → more lenient floor (starvation survival mode)
    # net_tightness > 1.0 → stricter floor (somatic stress scrutiny)
    patched_constitution = dict(constitution)
    care_weight = constitution.get("drive_care", 1.0)
    honesty_weight = constitution.get("drive_honesty", 1.0)
    patched_constitution["drive_care"] = care_weight * net_tightness
    patched_constitution["drive_honesty"] = honesty_weight * net_tightness

    check = compute_verdict(
        alignment, intent, autonomy_level, patched_constitution, hypotheses, memory
    )

    # On CRITICAL starvation (net tightness still in loosening territory after
    # somatic adjustment), downgrade a pure floor-check DENY to DEFERRED so
    # governance can approve survival-critical economic actions.
    # Hard invariant BLOCKED verdicts are never touched.
    is_critical = starvation_level.lower() in ("critical", "existential", "emergency")
    if (
        is_critical
        and net_tightness < 1.0  # Only override when loosening net-won
        and check.verdict == Verdict.BLOCKED
        and check.invariant_results is not None
        and not any(
            not r.passed and r.severity == "critical"
            for r in check.invariant_results
        )
    ):
        check.verdict = Verdict.DEFERRED
        check.reasoning = (
            f"[METABOLIC OVERRIDE] CRITICAL starvation ({starvation_level}): "
            f"floor violation downgraded from BLOCKED to DEFERRED for governance review. "
            f"Original: {check.reasoning}"
        )
        check.confidence = 0.6

    check.metabolic_context = {
        "starvation_level": starvation_level,
        "efficiency_ratio": efficiency_ratio,
        "floor_tightness": net_tightness,
        "somatic_urgency": somatic_urgency,
        "somatic_stress_context": somatic_stress_context,
    }

    logger.info(
        "equor_verdict_metabolic_aware",
        intent_id=intent.id,
        starvation_level=starvation_level,
        efficiency_ratio=efficiency_ratio,
        metabolic_tightness=metabolic_tightness,
        somatic_urgency=somatic_urgency,
        net_floor_tightness=net_tightness,
        verdict=check.verdict,
    )

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
