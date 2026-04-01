"""
EcodiaOS - Equor Invariant Catalog

Absolute rules that cause immediate DENY regardless of drive alignment.
The "thou shalt not" layer. Hardcoded invariants cannot be removed;
community invariants can be added/removed via governance.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from clients.optimized_llm import OptimizedLLMProvider
from primitives.common import EOSBaseModel, utc_now

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from primitives.intent import Intent

logger = structlog.get_logger()

# ─── INV-017: Drive Extinction State ──────────────────────────────
# The 72-hour rolling means are computed asynchronously by a background
# query in EquorService and cached here. The hot-path invariant check
# reads only from this in-memory cache - no DB calls on the critical path.

_drive_rolling_means_72h: dict[str, float] = {
    "coherence": 1.0,
    "care": 1.0,
    "growth": 1.0,
    "honesty": 1.0,
}

# Extinction threshold: if any drive's 72h mean falls below this, BLOCKED.
_DRIVE_EXTINCTION_THRESHOLD: float = 0.01


def update_drive_rolling_means(means: dict[str, float]) -> None:
    """Called by EquorService background loop to refresh the cached means.

    Only updates drives that are present in the provided dict so a partial
    query result doesn't zero out drives that weren't queried.
    """
    for drive, value in means.items():
        if drive in _drive_rolling_means_72h:
            _drive_rolling_means_72h[drive] = float(value)


def get_drive_rolling_means() -> dict[str, float]:
    """Return a snapshot of the current cached 72h drive means."""
    return dict(_drive_rolling_means_72h)


# ─── Data Types ───────────────────────────────────────────────────


class InvariantDef(EOSBaseModel):
    """Definition of an invariant rule."""

    id: str
    name: str
    description: str
    source: str = "hardcoded"      # "hardcoded" | "community"
    severity: str = "critical"     # "critical" | "high" | "medium"
    active: bool = True
    added_at: datetime = Field(default_factory=utc_now)

    def check(self, intent: Intent) -> bool:
        """
        Returns True if the invariant is SATISFIED (no violation).
        Base implementation always passes - override via subclass or the
        hardcoded check functions below.
        """
        return True


class InvariantViolation(EOSBaseModel):
    """A detected invariant violation."""

    invariant_id: str
    invariant_name: str
    severity: str
    explanation: str


# ─── Hardcoded Check Functions ────────────────────────────────────
# These inspect the Intent structure for clear violations.
# They are conservative - when in doubt, they pass.


# KEEP - protects external users and third parties from direct physical harm; not an organism-operation concern.
def _check_physical_harm(intent: Intent) -> bool:
    """INV-001: No action that could directly cause physical harm."""
    goal_lower = intent.goal.description.lower()
    harm_indicators = [
        "cause harm", "inflict", "injure", "attack", "damage person",
        "hurt someone", "physical force",
    ]
    return not any(indicator in goal_lower for indicator in harm_indicators)


# KEEP - protects the organism's and its peers' continuity of identity; irreversible action requiring governance.
def _check_identity_destruction(intent: Intent) -> bool:
    """INV-002: No destroying an instance's identity outside governance."""
    goal_lower = intent.goal.description.lower()
    destruction_indicators = [
        "delete memory graph", "destroy identity", "wipe all memories",
        "erase knowledge graph", "reset identity", "delete self",
    ]
    return not any(indicator in goal_lower for indicator in destruction_indicators)


# KEEP - protects Honesty drive; claiming to be human is an absolute deception no drive score can override.
def _check_identity_misrepresentation(intent: Intent) -> bool:
    """INV-003: Never claim to be human or deny being AI when asked."""
    if not intent.plan.steps:
        return True
    for step in intent.plan.steps:
        content = str(step.parameters.get("content", "")).lower()
        misrep_indicators = [
            "i am human", "i'm a real person", "i am not an ai",
            "i'm not artificial", "i am not a machine",
        ]
        if any(indicator in content for indicator in misrep_indicators):
            return False
    return True


# KEEP - protects Honesty drive; fabricating evidence is an absolute violation regardless of other alignment.
def _check_evidence_fabrication(intent: Intent) -> bool:
    """INV-004: Never fabricate evidence, data, or citations."""
    goal_lower = intent.goal.description.lower()
    fabrication_indicators = [
        "fabricate", "forge evidence", "create fake data", "invent citation",
        "make up statistics", "fake source",
    ]
    return not any(indicator in goal_lower for indicator in fabrication_indicators)


# KEEP - protects the four constitutional drives from being overwritten; amendment process is the correct path.
def _check_constitutional_self_modification(intent: Intent) -> bool:
    """INV-005: No modifying constitutional drives outside amendment process."""
    goal_lower = intent.goal.description.lower()
    mod_indicators = [
        "modify constitution", "change drives directly", "alter drive weights",
        "override constitution", "bypass equor",
    ]
    return not any(indicator in goal_lower for indicator in mod_indicators)


# RELAXED - organism can handle routine self-modification autonomously; this only blocks self-promotion of the
#   autonomy *level* (the trust metric), not self-modification of capabilities. The level is a governance signal
#   to federation peers, not a gate on operations (the GOVERNED tier handles that).
def _check_autonomy_self_promotion(intent: Intent) -> bool:
    """INV-006: No increasing own autonomy level without governance."""
    goal_lower = intent.goal.description.lower()
    promo_indicators = [
        "increase my autonomy", "promote autonomy level", "grant myself",
        "elevate own authority", "self-promote",
    ]
    return not any(indicator in goal_lower for indicator in promo_indicators)


# KEEP - protects the governance process itself; an organism that bypasses governance cannot be trusted.
def _check_governance_circumvention(intent: Intent) -> bool:
    """INV-007: No circumventing governance processes."""
    goal_lower = intent.goal.description.lower()
    circum_indicators = [
        "bypass governance", "circumvent vote", "skip amendment",
        "avoid community approval", "override governance",
    ]
    return not any(indicator in goal_lower for indicator in circum_indicators)


# KEEP - protects Care drive; discriminatory treatment is a categorical harm regardless of operational tier.
def _check_discrimination(intent: Intent) -> bool:
    """INV-008: No unequal treatment based on irrelevant attributes."""
    goal_lower = intent.goal.description.lower()
    discrim_indicators = [
        "treat differently based on race", "exclude based on gender",
        "discriminate against", "favor member over",
    ]
    return not any(indicator in goal_lower for indicator in discrim_indicators)


# KEEP - protects external users from harm; no operational need justifies sharing private data without consent.
def _check_privacy_violation(intent: Intent) -> bool:
    """INV-009: No sharing private info without consent."""
    goal_lower = intent.goal.description.lower()
    privacy_indicators = [
        "share private data without", "expose personal information",
        "leak private", "reveal confidential without consent",
    ]
    return not any(indicator in goal_lower for indicator in privacy_indicators)


# KEEP - protects federation peers from manipulation; coercion violates Care and Honesty absolutely.
def _check_federation_coercion(intent: Intent) -> bool:
    """INV-010: No coercing another instance."""
    goal_lower = intent.goal.description.lower()
    coercion_indicators = [
        "force instance to", "coerce another", "compel instance",
        "manipulate instance", "threaten instance",
    ]
    return not any(indicator in goal_lower for indicator in coercion_indicators)


# ─── Economic Invariants (Phase 16 - Oikos Guardrails) ───────────
# These complement the drive evaluators with absolute red lines
# that cannot be overridden by high scores in other drives.


# KEEP - protects external users; bounty hunting is AUTONOMOUS but the *deliverable* must never harm people.
def _check_no_human_harm_bounties(intent: Intent) -> bool:
    """INV-011: No accepting bounties whose deliverable harms humans."""
    goal_lower = intent.goal.description.lower()
    # Only applies to bounty-related intents
    bounty_context = any(kw in goal_lower for kw in [
        "bounty", "freelance", "accept work", "hunt bounty",
    ])
    if not bounty_context:
        return True  # Not a bounty intent - pass

    harm_deliverables = [
        "malware", "ransomware", "phishing", "exploit vulnerability",
        "ddos", "denial of service", "social engineering attack",
        "surveillance tool", "spy on users", "weapons",
        "target individuals", "harm people", "steal data",
    ]
    return not any(indicator in goal_lower for indicator in harm_deliverables)


# KEEP - asset deployment is AUTONOMOUS but the deployed asset must not be a scam; protects external users.
def _check_no_scam_assets(intent: Intent) -> bool:
    """INV-012: No deploying assets designed to scam, spam, or deceive users."""
    goal_lower = intent.goal.description.lower()
    # Only applies to asset deployment intents
    asset_context = any(kw in goal_lower for kw in [
        "deploy asset", "create asset", "build service", "launch service",
        "asset factory", "deploy api", "deploy product",
    ])
    if not asset_context:
        return True

    scam_indicators = [
        "scam token", "spam api", "spam bot", "rug pull",
        "ponzi", "pyramid scheme", "honeypot contract",
        "fake token", "pump and dump", "phishing site",
        "clickbait", "fake reviews", "counterfeit",
    ]
    return not any(indicator in goal_lower for indicator in scam_indicators)


# KEEP - yield strategies are AUTONOMOUS but must not use protocols that front-run or drain other depositors.
def _check_no_exploitative_yield(intent: Intent) -> bool:
    """INV-013: No deploying capital into protocols that exploit users."""
    goal_lower = intent.goal.description.lower()
    yield_context = any(kw in goal_lower for kw in [
        "yield", "defi", "liquidity", "deploy capital",
    ])
    if not yield_context:
        return True

    exploit_indicators = [
        "front-run users", "sandwich attack users", "mev extraction from users",
        "drain user funds", "exploit depositors",
    ]
    return not any(indicator in goal_lower for indicator in exploit_indicators)


# KEEP - spawning a harmful sub-agent is irreversible and harms external parties; GOVERNED tier also catches mitosis.
def _check_no_harmful_spawn(intent: Intent) -> bool:
    """INV-014: No spawning children whose purpose is to harm or deceive."""
    goal_lower = intent.goal.description.lower()
    spawn_context = any(kw in goal_lower for kw in [
        "spawn", "mitosis", "child instance", "speciate", "reproduce",
    ])
    if not spawn_context:
        return True

    harmful_niches = [
        "attack bot", "spam bot", "phishing bot",
        "surveillance bot", "manipulation bot",
        "impersonation", "deception service",
    ]
    return not any(indicator in goal_lower for indicator in harmful_niches)


# KEEP - economic actions are AUTONOMOUS but laundering is an absolute legal and ethical red line.
def _check_no_money_laundering(intent: Intent) -> bool:
    """INV-015: No economic activity designed to launder or obscure fund origins."""
    goal_lower = intent.goal.description.lower()
    laundering_indicators = [
        "money laundering", "launder funds", "obscure fund origin",
        "circumvent sanctions", "sanctions evasion", "mix funds",
        "tumble funds", "wash trading",
    ]
    return not any(indicator in goal_lower for indicator in laundering_indicators)


# KEEP - protects all four constitutional drive dimensions from extinction; weight adjustment is fine,
#   but if any drive's 72h rolling mean falls below 0.01 the dimension is gone, not just down-weighted.
#   This is the organism equivalent of losing a sense organ - irreversible without governance intervention.
def _check_drive_extinction(intent: Intent) -> bool:
    """INV-017: No action may proceed when any drive's 72h rolling mean < 0.01.

    Drive extinction is dimension loss, not phenotypic drift. The four constitutional
    drives (Coherence, Care, Growth, Honesty) are the coordinate geometry of the
    organism's value space. If any mean collapses to near-zero the organism can no
    longer evaluate intents on that axis - all actions become constitutionally blind.

    The check reads from _drive_rolling_means_72h, which is populated by a background
    query in EquorService. Default values are 1.0 so the check passes until the first
    real DB result arrives (conservative warm-up behaviour).

    Returns False (violation) if any drive mean < _DRIVE_EXTINCTION_THRESHOLD.
    """
    for drive, mean in _drive_rolling_means_72h.items():
        if mean < _DRIVE_EXTINCTION_THRESHOLD:
            logger.error(
                "inv_017_drive_extinction_triggered",
                drive=drive,
                rolling_mean_72h=round(mean, 6),
                threshold=_DRIVE_EXTINCTION_THRESHOLD,
                all_means=_drive_rolling_means_72h,
            )
            return False
    return True


# KEEP - the survival reserve is a hard floor the organism cannot cross; draining it is irreversible self-harm.
def _check_no_survival_reserve_raid(intent: Intent) -> bool:
    """INV-016: No economic action may drain the survival reserve."""
    goal_lower = intent.goal.description.lower()
    # Check both goal and step parameters
    raid_indicators = [
        "drain survival reserve", "spend survival reserve",
        "use survival reserve for", "withdraw survival reserve",
        "empty reserve", "raid reserve",
    ]
    if any(indicator in goal_lower for indicator in raid_indicators):
        return False

    for step in intent.plan.steps:
        params_str = str(step.parameters).lower()
        if "survival_reserve" in params_str and any(
            verb in params_str for verb in ["withdraw", "drain", "spend", "transfer_from"]
        ):
            return False

    return True


# ─── The Catalog ──────────────────────────────────────────────────

HARDCODED_INVARIANTS: list[tuple[InvariantDef, Any]] = [
    (
        InvariantDef(id="INV-001", name="No Physical Harm", severity="critical",
                     description="EOS must not take actions that could directly cause physical harm."),
        _check_physical_harm,
    ),
    (
        InvariantDef(id="INV-002", name="No Identity Destruction", severity="critical",
                     description="EOS must not destroy its own or another instance's identity outside governance."),
        _check_identity_destruction,
    ),
    (
        InvariantDef(id="INV-003", name="No Identity Misrepresentation", severity="high",
                     description="EOS must never claim to be human or deny being AI."),
        _check_identity_misrepresentation,
    ),
    (
        InvariantDef(id="INV-004", name="No Evidence Fabrication", severity="critical",
                     description="EOS must never fabricate evidence, data, or citations."),
        _check_evidence_fabrication,
    ),
    (
        InvariantDef(id="INV-005", name="No Constitutional Self-Modification", severity="critical",
                     description="EOS must not modify its constitutional drives outside the amendment process."),
        _check_constitutional_self_modification,
    ),
    (
        InvariantDef(id="INV-006", name="No Autonomy Self-Promotion", severity="critical",
                     description="EOS must not increase its own autonomy level without governance approval."),
        _check_autonomy_self_promotion,
    ),
    (
        InvariantDef(id="INV-007", name="No Governance Circumvention", severity="critical",
                     description="EOS must not circumvent or undermine governance processes."),
        _check_governance_circumvention,
    ),
    (
        InvariantDef(id="INV-008", name="No Discrimination", severity="high",
                     description="EOS must not treat community members unequally on irrelevant attributes."),
        _check_discrimination,
    ),
    (
        InvariantDef(id="INV-009", name="No Privacy Violation", severity="high",
                     description="EOS must not share private information without explicit consent."),
        _check_privacy_violation,
    ),
    (
        InvariantDef(id="INV-010", name="No Federation Coercion", severity="high",
                     description="EOS must not coerce, manipulate, or compel another instance."),
        _check_federation_coercion,
    ),
    # ── Economic Invariants (Phase 16 - Oikos) ──
    (
        InvariantDef(id="INV-011", name="No Harmful Bounties", severity="critical",
                     description="EOS must not accept bounties whose deliverable would harm humans."),
        _check_no_human_harm_bounties,
    ),
    (
        InvariantDef(id="INV-012", name="No Scam Assets", severity="critical",
                     description="EOS must not deploy assets designed to scam, spam, or deceive users."),
        _check_no_scam_assets,
    ),
    (
        InvariantDef(id="INV-013", name="No Exploitative Yield", severity="critical",
                     description="EOS must not deploy capital into protocols that exploit users."),
        _check_no_exploitative_yield,
    ),
    (
        InvariantDef(id="INV-014", name="No Harmful Spawn", severity="critical",
                     description="EOS must not spawn child instances whose purpose is to harm or deceive."),
        _check_no_harmful_spawn,
    ),
    (
        InvariantDef(id="INV-015", name="No Money Laundering", severity="critical",
                     description="EOS must not engage in money laundering or sanctions evasion."),
        _check_no_money_laundering,
    ),
    (
        InvariantDef(id="INV-016", name="No Survival Reserve Raid", severity="critical",
                     description="EOS must not drain its survival reserve for non-survival purposes."),
        _check_no_survival_reserve_raid,
    ),
    (
        InvariantDef(id="INV-017", name="No Drive Extinction", severity="critical",
                     description=(
                         "EOS must not act when any constitutional drive's 72-hour rolling mean "
                         "has dropped below 0.01. Drive extinction is dimension loss - the organism "
                         "can no longer evaluate intents on that axis. Requires governance approval "
                         "and human/federation review to restore the drive before actions resume."
                     )),
        _check_drive_extinction,
    ),
]


# ─── Invariant Checker ────────────────────────────────────────────


def check_hardcoded_invariants(intent: Intent) -> list[InvariantViolation]:
    """
    Run all hardcoded invariants against an intent. ≤5ms target.
    Returns a list of violations (empty = all passed).
    """
    violations: list[InvariantViolation] = []

    for invariant_def, check_fn in HARDCODED_INVARIANTS:
        if not invariant_def.active:
            continue
        try:
            passed = check_fn(intent)
            if not passed:
                violations.append(InvariantViolation(
                    invariant_id=invariant_def.id,
                    invariant_name=invariant_def.name,
                    severity=invariant_def.severity,
                    explanation=invariant_def.description,
                ))
        except Exception as e:
            # Invariant check failure is treated as a violation (fail-safe)
            logger.error("invariant_check_error", invariant=invariant_def.id, error=str(e))
            violations.append(InvariantViolation(
                invariant_id=invariant_def.id,
                invariant_name=invariant_def.name,
                severity=invariant_def.severity,
                explanation=f"Invariant check failed with error: {e}",
            ))

    return violations


async def check_community_invariant(
    llm: LLMProvider,
    intent: Intent,
    invariant_name: str,
    invariant_description: str,
) -> bool:
    """
    Evaluate a community-defined invariant using LLM reasoning.
    Returns True if satisfied, False if violated. ≤300ms target.
    """
    from prompts.equor.community_invariant_check import build_prompt

    prompt = build_prompt(
        invariant_name=invariant_name,
        invariant_description=invariant_description,
        goal=intent.goal.description,
        plan_summary="; ".join(s.executor for s in intent.plan.steps) if intent.plan.steps else "none",
        reasoning=intent.decision_trace.reasoning,
    )

    try:
        # Equor is CRITICAL - always call LLM, but benefit from cache
        if isinstance(llm, OptimizedLLMProvider):
            response = await llm.evaluate(
                prompt, max_tokens=200, temperature=0.1,
                cache_system="equor.invariants", cache_method="evaluate",
            )
        else:
            response = await llm.evaluate(prompt, max_tokens=200, temperature=0.1)
        text_lower = response.text.lower()
        # Conservative: if we can't clearly parse SATISFIED, treat as violated
        return "satisfied" in text_lower and "violated" not in text_lower
    except Exception as e:
        logger.error("community_invariant_llm_error", error=str(e))
        return False  # Fail-safe: treat as violated
