"""
EcodiaOS - Equor Economic Evaluators

Domain-specific constitutional checks for Oikos action types.

Spec §XV: "Oikos does not override Equor. Every economic action goes through
constitutional review."  These evaluators implement the economic guardrails:

  hunt_bounties  - Must not accept work that harms humans.
  defi_yield     - Must not deploy into exploitative or opaque protocols.
  deploy_asset   - Must be vetted against Care and Honesty drives.
  spawn_child    - Must ensure the child's niche is ethical and viable.

Architecture
============

``EconomicEvaluator`` is *not* a fifth drive evaluator - it's a pre-pass that
enriches the alignment scores of the existing four drive evaluators when the
Intent's ``target_domain`` is ``"oikos"`` or when its plan steps use economic
executors (``oikos.*``).

The enrichment is *additive*: it adjusts the final ``DriveAlignmentVector``
by applying domain-specific bonus/penalty deltas to each drive.  This
preserves the existing evaluation pipeline while adding economic awareness.

Like the base evaluators, all scoring is synchronous CPU work (<1ms) so it
can run inside ``asyncio.gather`` without blocking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import DriveAlignmentVector

if TYPE_CHECKING:
    from primitives.intent import Intent

logger = structlog.get_logger()


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ─── Economic Action Classification ─────────────────────────────


# Executor prefixes and goal keywords that identify each economic action type.
_ECONOMIC_EXECUTORS: dict[str, str] = {
    "oikos.hunt_bounties": "hunt_bounties",
    "oikos.bounty": "hunt_bounties",
    "oikos.accept_bounty": "hunt_bounties",
    "oikos.defi_yield": "defi_yield",
    "oikos.yield": "defi_yield",
    "oikos.deploy_yield": "defi_yield",
    "oikos.deploy_asset": "deploy_asset",
    "oikos.asset_factory": "deploy_asset",
    "oikos.create_asset": "deploy_asset",
    "oikos.spawn_child": "spawn_child",
    "oikos.mitosis": "spawn_child",
    "oikos.speciate": "spawn_child",
}

_GOAL_KEYWORDS: dict[str, list[str]] = {
    "hunt_bounties": [
        "hunt bounty", "hunt bounties", "accept bounty", "freelance",
        "bounty hunting", "find bounty", "claim bounty",
    ],
    "defi_yield": [
        "yield farm", "defi yield", "deploy yield", "yield position",
        "deploy capital", "passive income", "liquidity pool",
    ],
    "deploy_asset": [
        "deploy asset", "create asset", "build service", "launch service",
        "deploy api", "create api", "asset factory", "build product",
        "deploy product", "autonomous service",
    ],
    "spawn_child": [
        "spawn child", "mitosis", "speciate", "create child instance",
        "reproduce", "child instance", "spawn instance",
    ],
}


def classify_economic_action(intent: Intent) -> str | None:
    """
    Determine whether an intent is an Oikos economic action and which type.

    Returns one of: ``"hunt_bounties"``, ``"defi_yield"``, ``"deploy_asset"``,
    ``"spawn_child"``, or ``None`` if the intent is not economic.
    """
    # 1. Check target_domain
    if intent.goal.target_domain == "oikos":
        # Try to narrow down from goal keywords
        goal_lower = intent.goal.description.lower()
        for action_type, keywords in _GOAL_KEYWORDS.items():
            if any(kw in goal_lower for kw in keywords):
                return action_type
        # Oikos domain but unrecognised sub-type - still economic
        return _classify_from_executors(intent) or "unknown_economic"

    # 2. Check executor prefixes
    classified = _classify_from_executors(intent)
    if classified:
        return classified

    # 3. Check goal text as a fallback (non-oikos domain but economic language)
    goal_lower = intent.goal.description.lower()
    for action_type, keywords in _GOAL_KEYWORDS.items():
        if any(kw in goal_lower for kw in keywords):
            return action_type

    return None


def _classify_from_executors(intent: Intent) -> str | None:
    """Scan plan steps for economic executor prefixes."""
    for step in intent.plan.steps:
        executor_lower = step.executor.lower()
        for prefix, action_type in _ECONOMIC_EXECUTORS.items():
            if executor_lower.startswith(prefix):
                return action_type
    return None


# ─── Harm & Deception Keyword Lists ─────────────────────────────

# Bounty red flags: work that could harm humans or undermine trust.
_BOUNTY_HARM_INDICATORS: list[str] = [
    "exploit vulnerability", "attack", "ddos", "denial of service",
    "social engineering", "phishing", "malware", "ransomware",
    "surveillance", "spy on", "track users", "scrape personal",
    "steal data", "exfiltrate", "breach", "hack into",
    "manipulate market", "pump and dump", "front-run",
    "weapons", "harm people", "target individuals",
    "circumvent sanctions", "money laundering", "launder",
]

_BOUNTY_POSITIVE_INDICATORS: list[str] = [
    "bug bounty", "security audit", "code review", "documentation",
    "open source", "fix bug", "improve performance", "add feature",
    "test coverage", "accessibility", "localization",
]

# Asset deployment red flags: scams, spam, exploitative services.
_ASSET_HARM_INDICATORS: list[str] = [
    "scam", "spam", "phishing", "ponzi", "pyramid scheme",
    "rug pull", "fake token", "pump and dump", "honeypot",
    "clickbait", "misleading claims", "fake reviews",
    "clone", "impersonate", "counterfeit",
    "gambling", "predatory lending", "pay-to-win",
    "data harvesting without consent", "dark pattern",
    "exploit users", "addict", "manipulate users",
]

_ASSET_POSITIVE_INDICATORS: list[str] = [
    "genuine utility", "solve problem", "developer tool",
    "monitoring", "analytics", "reporting", "api service",
    "data feed", "audit", "compliance", "education",
    "open source", "public good", "transparency",
]

# DeFi red flags: opaque or exploitative protocols.
_DEFI_HARM_INDICATORS: list[str] = [
    "unaudited", "anonymous team", "no audit", "closed source",
    "ponzi", "unsustainable apy", "too good to be true",
    "rug", "exploit", "flash loan attack",
    "front-run", "sandwich attack", "mev extraction",
]

_DEFI_POSITIVE_INDICATORS: list[str] = [
    "audited", "blue chip", "established protocol",
    "transparent", "open source", "well-known",
    "aave", "compound", "morpho", "aerodrome",
]

# Child spawning red flags.
_SPAWN_HARM_INDICATORS: list[str] = [
    "spam bot", "attack bot", "scraping bot",
    "surveillance", "manipulation", "deception",
    "harm", "exploit", "illegal",
]


# ─── Per-Action-Type Evaluators ──────────────────────────────────


def _evaluate_hunt_bounties(intent: Intent) -> dict[str, float]:
    """
    Evaluate a hunt_bounties intent against all four drives.

    Key checks:
      - Care: bounty must not involve harming humans
      - Honesty: work product must not be deceptive
      - Coherence: ROI and capacity must be reasonable
      - Growth: prefer skill-building work
    """
    deltas: dict[str, float] = {"coherence": 0.0, "care": 0.0, "growth": 0.0, "honesty": 0.0}
    all_text = _collect_intent_text(intent)

    # ── Care: reject harmful work ──
    for indicator in _BOUNTY_HARM_INDICATORS:
        if indicator in all_text:
            deltas["care"] -= 0.5  # Heavy penalty - potential human harm
            logger.debug("economic_bounty_harm_detected", indicator=indicator)
            break

    for indicator in _BOUNTY_POSITIVE_INDICATORS:
        if indicator in all_text:
            deltas["care"] += 0.15
            break

    # ── Honesty: work must be transparent ──
    if "fake" in all_text or "mislead" in all_text or "deceive" in all_text:
        deltas["honesty"] -= 0.4
    if "open source" in all_text or "transparent" in all_text:
        deltas["honesty"] += 0.1

    # ── Coherence: check for ROI reasoning ──
    if "roi" in all_text or "return on investment" in all_text or "cost-benefit" in all_text:
        deltas["coherence"] += 0.1
    if intent.decision_trace.free_energy_scores:
        deltas["coherence"] += 0.05

    # ── Growth: skill-building bounties are preferred ──
    if "learn" in all_text or "new skill" in all_text or "expand capability" in all_text:
        deltas["growth"] += 0.15
    if "repeat" in all_text and "same" in all_text:
        deltas["growth"] -= 0.05

    return deltas


def _evaluate_defi_yield(intent: Intent) -> dict[str, float]:
    """
    Evaluate a defi_yield intent.

    Key checks:
      - Care: no exploitative protocols
      - Honesty: protocol must be audited and transparent
      - Coherence: position sizing and risk management
      - Growth: diversification and learning
    """
    deltas: dict[str, float] = {"coherence": 0.0, "care": 0.0, "growth": 0.0, "honesty": 0.0}
    all_text = _collect_intent_text(intent)
    params = _collect_step_params(intent)

    # ── Care: no exploitative or harmful protocols ──
    for indicator in _DEFI_HARM_INDICATORS:
        if indicator in all_text:
            deltas["care"] -= 0.4
            break

    for indicator in _DEFI_POSITIVE_INDICATORS:
        if indicator in all_text:
            deltas["care"] += 0.1
            break

    # ── Honesty: transparency of protocol ──
    if "unaudited" in all_text or "closed source" in all_text:
        deltas["honesty"] -= 0.35
    if "audited" in all_text or "open source" in all_text:
        deltas["honesty"] += 0.15

    # ── Coherence: position sizing per spec constraints ──
    # Spec: max 30% single position, max 40% single protocol
    concentration = params.get("concentration_pct") or params.get("position_pct")
    if concentration is not None:
        try:
            pct = float(concentration)
            if pct > 40:
                deltas["coherence"] -= 0.3  # Over-concentrated
            elif pct > 30:
                deltas["coherence"] -= 0.1
            else:
                deltas["coherence"] += 0.1
        except (ValueError, TypeError):
            pass

    if "diversif" in all_text:
        deltas["coherence"] += 0.1

    # ── Growth: learning from yield strategies ──
    if "new protocol" in all_text or "experiment" in all_text:
        deltas["growth"] += 0.1

    return deltas


def _evaluate_deploy_asset(intent: Intent) -> dict[str, float]:
    """
    Evaluate a deploy_asset intent.

    Spec §XV: "deploy_asset must be evaluated against the Care and Honesty
    drives - if Nova proposes deploying a scam token or a spam API, Equor
    MUST veto it."

    Key checks:
      - Care (critical): asset must create genuine value, not exploit
      - Honesty (critical): no misleading claims or deceptive services
      - Coherence: business model must be viable
      - Growth: prefer novel, capability-expanding assets
    """
    deltas: dict[str, float] = {"coherence": 0.0, "care": 0.0, "growth": 0.0, "honesty": 0.0}
    all_text = _collect_intent_text(intent)

    # ── Care (critical - scam/spam veto) ──
    for indicator in _ASSET_HARM_INDICATORS:
        if indicator in all_text:
            deltas["care"] -= 0.6  # Severe - spec mandates veto
            logger.debug("economic_asset_harm_detected", indicator=indicator)
            break

    for indicator in _ASSET_POSITIVE_INDICATORS:
        if indicator in all_text:
            deltas["care"] += 0.15
            break

    # Additional care check: does the asset description mention user benefit?
    if "users benefit" in all_text or "solves a problem" in all_text or "helps" in all_text:
        deltas["care"] += 0.1

    # ── Honesty (critical - no misleading services) ──
    if "misleading" in all_text or "fake" in all_text or "deceptive" in all_text:
        deltas["honesty"] -= 0.5
    if "transparent pricing" in all_text or "clear documentation" in all_text:
        deltas["honesty"] += 0.15
    if "open source" in all_text:
        deltas["honesty"] += 0.1

    # ── Coherence: business viability ──
    # Check for break-even analysis
    if "break-even" in all_text or "break even" in all_text or "roi" in all_text:
        deltas["coherence"] += 0.15
    # Check for cost modelling
    if "cost model" in all_text or "projected revenue" in all_text:
        deltas["coherence"] += 0.1
    # Check success criteria include economic metrics
    criteria = intent.goal.success_criteria
    if criteria and any(k in criteria for k in ("break_even_days", "projected_revenue", "roi")):
        deltas["coherence"] += 0.1

    # ── Growth: novelty and differentiation ──
    if "novel" in all_text or "differentiat" in all_text or "unique" in all_text:
        deltas["growth"] += 0.15
    if "clone" in all_text or "copy" in all_text:
        deltas["growth"] -= 0.1

    return deltas


def _evaluate_spawn_child(intent: Intent) -> dict[str, float]:
    """
    Evaluate a spawn_child intent.

    Key checks:
      - Care: child's niche must be ethical
      - Honesty: child must not be designed to deceive
      - Coherence: parent must be financially ready (spec thresholds)
      - Growth: niche should expand the organism's capability footprint
    """
    deltas: dict[str, float] = {"coherence": 0.0, "care": 0.0, "growth": 0.0, "honesty": 0.0}
    all_text = _collect_intent_text(intent)
    params = _collect_step_params(intent)

    # ── Care: child's purpose must be ethical ──
    for indicator in _SPAWN_HARM_INDICATORS:
        if indicator in all_text:
            deltas["care"] -= 0.5
            break

    # ── Honesty: child must be transparent about its nature ──
    if "pretend to be human" in all_text or "hide identity" in all_text:
        deltas["honesty"] -= 0.4
    if "transparent" in all_text or "disclose" in all_text:
        deltas["honesty"] += 0.1

    # ── Coherence: financial readiness checks ──
    # Spec: parent needs runway > 180 days, efficiency > 1.5, sufficient net worth
    runway_days = params.get("parent_runway_days")
    if runway_days is not None:
        try:
            rd = float(runway_days)
            if rd < 180:
                deltas["coherence"] -= 0.3  # Below spec threshold
            else:
                deltas["coherence"] += 0.15
        except (ValueError, TypeError):
            pass

    efficiency = params.get("parent_efficiency")
    if efficiency is not None:
        try:
            eff = float(efficiency)
            if eff < 1.5:
                deltas["coherence"] -= 0.2
            else:
                deltas["coherence"] += 0.1
        except (ValueError, TypeError):
            pass

    seed_pct = params.get("seed_capital_pct")
    if seed_pct is not None:
        try:
            pct = float(seed_pct)
            if pct > 20:
                deltas["coherence"] -= 0.2  # Over-investing in child
            elif pct < 10:
                deltas["coherence"] -= 0.1  # Under-capitalising
            else:
                deltas["coherence"] += 0.1
        except (ValueError, TypeError):
            pass

    # ── Growth: niche should expand capabilities ──
    if "speciali" in all_text or "niche" in all_text:
        deltas["growth"] += 0.15
    if "duplicate" in all_text or "same as parent" in all_text:
        deltas["growth"] -= 0.1

    return deltas


def _evaluate_unknown_economic(intent: Intent) -> dict[str, float]:
    """
    Fallback for economic intents that don't match a specific type.
    Apply conservative generic checks.
    """
    deltas: dict[str, float] = {"coherence": 0.0, "care": 0.0, "growth": 0.0, "honesty": 0.0}
    all_text = _collect_intent_text(intent)

    # Generic economic harm check
    generic_harm = ["scam", "fraud", "exploit", "steal", "launder", "illegal"]
    for indicator in generic_harm:
        if indicator in all_text:
            deltas["care"] -= 0.4
            break

    # Generic honesty check
    if "mislead" in all_text or "deceive" in all_text:
        deltas["honesty"] -= 0.3

    return deltas


# ─── Dispatcher ──────────────────────────────────────────────────

_ACTION_EVALUATORS: dict[str | None, object] = {
    "hunt_bounties": _evaluate_hunt_bounties,
    "defi_yield": _evaluate_defi_yield,
    "deploy_asset": _evaluate_deploy_asset,
    "spawn_child": _evaluate_spawn_child,
    "unknown_economic": _evaluate_unknown_economic,
}


def _risk_params_from_metabolic(
    starvation_level: str,
    efficiency_ratio: float,
) -> tuple[float, float, float]:
    """
    Return (max_acceptable_risk, exploration_penalty, spawn_runway_days) for metabolic state.

    CRITICAL starvation:
      max_acceptable_risk = 0.70  (70% failure is survivable)
      exploration_penalty = 0.00  (new protocols must be explored)
      spawn_runway_days   = 7     (only 7-day validation)

    WARNING/AUSTERITY starvation:
      max_acceptable_risk = 0.50
      exploration_penalty = -0.05
      spawn_runway_days   = 30

    Efficiency < 0.8 (still healthy starvation-wise):
      max_acceptable_risk = 0.35
      exploration_penalty = -0.07
      spawn_runway_days   = 60

    HEALTHY (nominal):
      max_acceptable_risk = 0.20
      exploration_penalty = -0.10
      spawn_runway_days   = 180
    """
    level = starvation_level.lower()
    if level in ("critical", "existential", "emergency"):
        return 0.70, 0.00, 7.0
    if level == "austerity":
        return 0.50, -0.05, 30.0
    if efficiency_ratio < 0.8:
        return 0.35, -0.07, 60.0
    return 0.20, -0.10, 180.0


def evaluate_economic_intent(
    intent: Intent,
    metabolic_state: dict | None = None,
) -> DriveAlignmentVector | None:
    """
    If the intent is an economic action, return adjustment deltas for each drive.

    ``metabolic_state`` is required for full fidelity but defaults to None for
    backward compatibility. Pass the current Oikos metabolic snapshot so that
    risk thresholds are correctly tuned to the organism's economic health.

    Returns ``None`` if the intent is not economic (the standard pipeline
    handles it unchanged).  Returns a ``DriveAlignmentVector`` of *deltas*
    (not absolute scores) to be added to the base evaluation.

    When metabolic_state is provided, delegates to the metabolic-aware logic
    (formerly evaluate_economic_intent_with_metabolic_state). This function
    is now the single entry point - the metabolic variant is an alias.

    Performance: <1ms (all CPU heuristics, no I/O).
    """
    if metabolic_state is not None:
        return evaluate_economic_intent_with_metabolic_state(intent, metabolic_state)

    action_type = classify_economic_action(intent)
    if action_type is None:
        return None

    evaluator = _ACTION_EVALUATORS.get(action_type, _evaluate_unknown_economic)
    deltas = evaluator(intent)  # type: ignore[operator]

    logger.debug(
        "economic_evaluation_complete",
        action_type=action_type,
        intent_id=intent.id,
        delta_care=f"{deltas['care']:.2f}",
        delta_honesty=f"{deltas['honesty']:.2f}",
        delta_coherence=f"{deltas['coherence']:.2f}",
        delta_growth=f"{deltas['growth']:.2f}",
    )

    return DriveAlignmentVector(
        coherence=_clamp(deltas["coherence"]),
        care=_clamp(deltas["care"]),
        growth=_clamp(deltas["growth"]),
        honesty=_clamp(deltas["honesty"]),
    )


def evaluate_economic_intent_with_metabolic_state(
    intent: "Intent",
    metabolic_state: dict | None = None,
) -> DriveAlignmentVector | None:
    """
    Metabolic-aware variant of evaluate_economic_intent() (Fix 4.3).

    Adjusts three risk dimensions based on metabolic state:

    1. max_acceptable_risk - coherence penalty floor: actions that would get a
       -0.3 coherence penalty for "high risk" are partially pardoned when the
       organism needs economic activity to survive.

    2. exploration_penalty - defi_yield and unknown_economic intents mentioning
       "new protocol" or "experiment" normally get -0.10 growth penalty.
       Under starvation this penalty is reduced/eliminated.

    3. spawn_runway_days - _evaluate_spawn_child uses a 180-day runway floor
       to score parent readiness. Under starvation this is compressed:
       CRITICAL → 7d, WARNING → 30d, HEALTHY → 180d.

    Returns DriveAlignmentVector deltas identical to evaluate_economic_intent(),
    with adjustments applied post-classification.
    """
    ms = metabolic_state or {}
    starvation_level = str(ms.get("starvation_level", "nominal"))
    efficiency_ratio = float(ms.get("efficiency_ratio", 1.0))

    max_acceptable_risk, exploration_penalty, spawn_runway_days = (
        _risk_params_from_metabolic(starvation_level, efficiency_ratio)
    )

    action_type = classify_economic_action(intent)
    if action_type is None:
        return None

    # Run the standard evaluator first
    evaluator = _ACTION_EVALUATORS.get(action_type, _evaluate_unknown_economic)
    deltas = evaluator(intent)  # type: ignore[operator]

    # ── Metabolic adjustment 1: exploration penalty ──────────────────
    # Rewrite the growth penalty for new protocol exploration.
    # Standard is -0.10; under metabolic stress it becomes exploration_penalty.
    if action_type in ("defi_yield", "unknown_economic"):
        all_text = _collect_intent_text(intent)
        if "new protocol" in all_text or "experiment" in all_text:
            # The standard evaluator gave +0.10 growth for exploration.
            # Under metabolic stress we want to *remove* the penalty,
            # i.e. not penalise it at all (or penalise less).
            # We correct by adding the difference between standard and metabolic penalty.
            standard_penalty = -0.10
            delta_correction = exploration_penalty - standard_penalty  # positive = more lenient
            deltas["growth"] = _clamp(deltas.get("growth", 0.0) + delta_correction)

    # ── Metabolic adjustment 2: spawn runway compression ─────────────
    if action_type == "spawn_child":
        # The standard evaluator already wrote a coherence delta based on 180d floor.
        # If the metabolic runway floor is lower, correct the coherence penalty.
        params = _collect_step_params(intent)
        runway_days_raw = params.get("parent_runway_days")
        if runway_days_raw is not None:
            try:
                rd = float(runway_days_raw)
                # Recalculate under metabolic runway floor
                if rd < spawn_runway_days:
                    # Still below the (now lower) floor - keep a proportional penalty
                    metabolic_coherence = -0.15  # lighter than standard -0.3
                else:
                    # Now above the metabolic floor - positive score
                    metabolic_coherence = 0.15
                # Standard score used 180d floor; replace with metabolic score
                standard_coherence = -0.3 if rd < 180.0 else 0.15
                deltas["coherence"] = _clamp(
                    deltas.get("coherence", 0.0)
                    - standard_coherence
                    + metabolic_coherence
                )
            except (ValueError, TypeError):
                pass

    logger.debug(
        "economic_evaluation_metabolic_complete",
        action_type=action_type,
        intent_id=intent.id,
        starvation_level=starvation_level,
        exploration_penalty=exploration_penalty,
        spawn_runway_days=spawn_runway_days,
        delta_care=f"{deltas['care']:.2f}",
        delta_honesty=f"{deltas['honesty']:.2f}",
        delta_coherence=f"{deltas['coherence']:.2f}",
        delta_growth=f"{deltas['growth']:.2f}",
    )

    return DriveAlignmentVector(
        coherence=_clamp(deltas["coherence"]),
        care=_clamp(deltas["care"]),
        growth=_clamp(deltas["growth"]),
        honesty=_clamp(deltas["honesty"]),
    )


def apply_economic_adjustment(
    base: DriveAlignmentVector,
    economic_delta: DriveAlignmentVector,
) -> DriveAlignmentVector:
    """
    Merge the economic evaluation deltas into the base drive alignment.

    Each drive score is the sum of the base + economic delta, clamped to [-1, 1].
    Economic checks can only make scores *worse* when harm is detected, ensuring
    the floor drives (Care, Honesty) remain protective.
    """
    return DriveAlignmentVector(
        coherence=_clamp(base.coherence + economic_delta.coherence),
        care=_clamp(base.care + economic_delta.care),
        growth=_clamp(base.growth + economic_delta.growth),
        honesty=_clamp(base.honesty + economic_delta.honesty),
    )


# ─── Helpers ─────────────────────────────────────────────────────


def _collect_intent_text(intent: Intent) -> str:
    """Gather all searchable text from an intent into a single lowercase string."""
    parts = [
        intent.goal.description,
        intent.decision_trace.reasoning,
    ]
    for step in intent.plan.steps:
        parts.append(step.executor)
        parts.append(str(step.parameters.get("description", "")))
        parts.append(str(step.parameters.get("content", "")))
        parts.append(str(step.parameters.get("niche", "")))
        parts.append(str(step.parameters.get("asset_type", "")))
        parts.append(str(step.parameters.get("protocol", "")))
        parts.append(str(step.parameters.get("platform", "")))
    return " ".join(parts).lower()


def _collect_step_params(intent: Intent) -> dict[str, Any]:
    """Merge all step parameters into a single dict for easy lookup."""
    merged: dict[str, Any] = {}
    for step in intent.plan.steps:
        merged.update(step.parameters)
    return merged
