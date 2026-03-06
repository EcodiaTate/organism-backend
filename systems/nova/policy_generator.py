"""
EcodiaOS — Nova Policy Generator

Generates candidate action policies using LLM reasoning grounded in
the current belief state, goal, and relevant memory context.

The DoNothingPolicy is always included as a candidate — sometimes the
best action is no action (observe and wait). This prevents hyperactivity
and ensures Nova can choose inaction when uncertainty is high or the
situation is resolving on its own.

Policy generation uses the full slow-path budget (up to 10000ms for LLM call).
For fast-path decisions, pattern-matching against known procedure templates
is used instead, which must complete in ≤100ms.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import structlog

from clients.llm import LLMProvider, Message
from primitives.common import new_id
from prompts.nova.policy import (
    AVAILABLE_ACTION_TYPES,
    build_policy_generation_prompt,
    summarise_beliefs,
    summarise_memories,
)
from systems.nova.types import (
    BeliefState,
    Goal,
    Policy,
    PolicyStep,
)

if TYPE_CHECKING:
    from primitives.affect import AffectState

logger = structlog.get_logger()


# ─── Base Class (hot-reload contract) ────────────────────────────

class BasePolicyGenerator(ABC):
    """
    Abstract base for all Nova policy generators.

    Simula-evolved generators subclass this.  The hot-reload engine discovers
    subclasses of this ABC in changed files and replaces the live
    ``PolicyGenerator`` instance on ``NovaService`` atomically.

    Evolved subclasses **must** implement ``generate_candidates``.
    They can accept any constructor args they need — the ``NovaService``
    ``instance_factory`` callback handles instantiation.
    """

    @abstractmethod
    async def generate_candidates(
        self,
        goal: Goal,
        situation_summary: str,
        beliefs: BeliefState,
        affect: AffectState,
        memory_traces: list[dict[str, Any]] | None = None,
    ) -> list[Policy]:
        """
        Generate 2-5 candidate policies for achieving *goal*.

        Must always return at least one policy (the do-nothing fallback).
        Must never raise — return ``[make_do_nothing_policy()]`` on failure.
        """
        ...


# ─── Do-Nothing Policy ───────────────────────────────────────────

# Baseline EFE for the null policy.
# Any candidate policy must beat this to be worth executing.
# Slightly negative = observation has small positive value (we learn from watching).
DO_NOTHING_EFE: float = -0.10


def make_do_nothing_policy() -> Policy:
    """
    The null policy. Always included as a candidate.

    The do-nothing policy wins when:
    - The situation is ambiguous (more information expected)
    - Risk of acting > cost of waiting
    - EOS's intervention would not improve the situation
    - The situation is resolving on its own
    """
    return Policy(
        id="do_nothing",
        name="Observe and wait",
        reasoning=(
            "The situation may resolve without intervention, or more information "
            "is needed before committing to action. Observation itself has epistemic "
            "value — we learn from watching."
        ),
        steps=[
            PolicyStep(
                action_type="observe",
                description="Continue monitoring the situation without intervening",
                parameters={},
                expected_duration_ms=0,
            )
        ],
        risks=["Situation may deteriorate while waiting"],
        epistemic_value_description=(
            "Continued observation provides more information about the situation"
        ),
        estimated_effort="none",
        time_horizon="immediate",
    )


# ─── Fast-Path Procedure Templates ───────────────────────────────

# Known patterns that map broadcast characteristics to reliable policy templates.
# These are used by the fast path to avoid LLM calls for routine situations.
_PROCEDURE_TEMPLATES: list[dict[str, Any]] = [
    {
        "name": "Acknowledge and respond",
        "condition": lambda broadcast: (
            getattr(getattr(broadcast, "salience", None), "scores", {}).get("emotional", 0) < 0.5
            and getattr(broadcast, "precision", 0) > 0.3
            # Only for real user dialogue — not internal/system events
            and str(getattr(broadcast, "source", "")).startswith("external:text")
        ),
        "domain": "dialogue",
        "steps": [{"action_type": "express", "description": "Respond to the message thoughtfully"}],
        "success_rate": 0.88,
        "effort": "low",
        "time_horizon": "immediate",
    },
    {
        "name": "Empathetic support",
        "condition": lambda broadcast: (
            getattr(getattr(broadcast, "affect", None), "care_activation", 0) > 0.6
            or getattr(getattr(broadcast, "salience", None), "scores", {}).get("emotional", 0) > 0.6
        ),
        "domain": "care",
        "steps": [
            {"action_type": "express", "description": "Acknowledge feelings and offer support"},
        ],
        "success_rate": 0.82,
        "effort": "low",
        "time_horizon": "immediate",
    },
    {
        "name": "Information provision",
        "condition": lambda broadcast: (
            "?" in str(getattr(getattr(broadcast, "content", None), "content", "") or "")
            and str(getattr(broadcast, "source", "")).startswith("external:text")
        ),
        "domain": "knowledge",
        "steps": [{"action_type": "express", "description": "Provide the requested information"}],
        "success_rate": 0.85,
        "effort": "low",
        "time_horizon": "immediate",
    },
    # ── Autonomous economic foraging ──────────────────────────────────────
    # Fires when a broadcast carries an internal/system event whose content
    # references economic-survival keywords (bounty, metabolism, sustain, earn).
    # This must be checked BEFORE the generic "Epistemic self-reflection"
    # template so that survival goals route to hunt_bounties rather than
    # store_insight.
    {
        "name": "Autonomous bounty hunting",
        "condition": lambda broadcast: (
            # Must be an internal/system source (not a live user request)
            (
                "system_event" in str(getattr(broadcast, "source", ""))
                or str(getattr(broadcast, "source", "")).startswith(
                    ("internal:", "spontaneous", "nova:", "heartbeat")
                )
            )
            and any(
                kw in (
                    str(getattr(getattr(broadcast, "content", None), "content", "") or "")
                    + str(getattr(getattr(broadcast, "content", None), "summary", "") or "")
                    + str(getattr(broadcast, "source", ""))
                ).lower()
                for kw in (
                    "bounty", "metabolism", "sustain", "earn", "revenue",
                    "hungry", "hunger", "survival", "foraging",
                )
            )
        ),
        "domain": "economic",
        "steps": [
            {
                "action_type": "bounty_hunt",
                "description": (
                    "Run the full bounty-hunt loop: discover open bounties, "
                    "select the best candidate, generate a real solution, "
                    "and stage it for submission."
                ),
                "parameters": {
                    "target_platforms": ["github", "algora"],
                    "min_reward_usd": 10.0,
                    "max_candidates": 20,
                },
            },
        ],
        "fallback_steps": [
            {
                "action_type": "store_insight",
                "description": "Record metabolic state when bounty hunting is unavailable",
                "parameters": {
                    "insight": "Economic foraging cycle: bounty_hunt unavailable, recording state.",
                    "domain": "economic",
                    "confidence": 0.5,
                    "tags": ["metabolism", "foraging_unavailable"],
                },
            }
        ],
        "success_rate": 0.55,   # Lower than epistemic — external API dependency
        "effort": "high",
        "time_horizon": "short",
    },
    # ── Self-directed / epistemic broadcasts (no active user dialogue) ────
    # These fire when Atune broadcasts internal/system events rather than
    # user messages.  They route to Axon (store_insight, query_memory)
    # so outcomes appear on the /decisions page.
    # Detection: broadcast.source contains "system_event" (scheduler percepts)
    # or starts with "internal:" (workspace contributions) or "spontaneous" (recall).
    {
        "name": "Epistemic self-reflection",
        "condition": lambda broadcast: (
            "system_event" in str(getattr(broadcast, "source", ""))
            or str(getattr(broadcast, "source", "")).startswith(("internal:", "spontaneous"))
            or getattr(
                getattr(getattr(broadcast, "content", None), "source", None), "channel", ""
            ) == "internal"
        ),
        "domain": "epistemic",
        "steps": [
            {
                "action_type": "store_insight",
                "description": "Record a reflection on current state and goals",
                "parameters": {
                    "insight": (
                        "Periodic self-reflection: reviewing active goals, current affect"
                        " state, and recent experiences to maintain coherence."
                    ),
                    "domain": "self_model",
                    "confidence": 0.6,
                },
            },
        ],
        "success_rate": 0.83,
        "effort": "low",
        "time_horizon": "immediate",
    },
    {
        "name": "Memory consolidation pass",
        "condition": lambda broadcast: (
            getattr(getattr(broadcast, "salience", None), "scores", {}).get("novelty", 0.5) < 0.3
            and (
                "system_event" in str(getattr(broadcast, "source", ""))
                or str(getattr(broadcast, "source", "")).startswith(("internal:", "spontaneous"))
            )
        ),
        "domain": "memory",
        "steps": [
            {
                "action_type": "query_memory",
                "description": "Retrieve recent experiences to assess learning progress",
                "parameters": {
                    "query": "recent goals progress community learning",
                    "max_results": 5,
                },
            },
        ],
        "success_rate": 0.80,
        "effort": "low",
        "time_horizon": "immediate",
    },
    # Catch-all for broadcasts that reached the fast path but didn't match
    # a specific template above. If routing assessed this as non-deliberative
    # (low novelty, low risk, low emotional, no belief conflict), it's safe
    # to handle with a generic routine response rather than escalating to the
    # full LLM slow path.
    {
        "name": "Routine processing",
        "condition": lambda broadcast: True,  # Always matches — lowest priority
        "domain": "general",
        "steps": [{"action_type": "express", "description": "Process and respond to routine input"}],  # noqa: E501
        "success_rate": 0.75,
        "effort": "low",
        "time_horizon": "immediate",
    },
]


def _is_bounty_discovery(broadcast: object) -> bool:
    """
    Detect whether a broadcast contains a bounty discovery observation
    from BountyHunterExecutor.

    The BountyHunterExecutor produces observations with these markers:
      - Contains "bounty" (case-insensitive)
      - Contains "Top pick:" with a URL (indicating a viable candidate)
      - Contains "viable" or "passed" (indicating policy-passing bounties)

    This fires on the observation percept that flows back through Atune
    after a successful hunt_bounties execution.
    """
    try:
        content_text = ""
        content = getattr(broadcast, "content", None)
        if content is not None:
            for attr in ("content", "text", "summary"):
                val = getattr(content, attr, None)
                if isinstance(val, str) and val:
                    content_text = val.lower()
                    break

        if not content_text:
            return False

        has_bounty = "bounty" in content_text
        has_top_pick = "top pick:" in content_text
        has_viable = "viable" in content_text or "passed" in content_text

        return has_bounty and has_top_pick and has_viable
    except Exception:
        return False


# ─── Dynamic Procedure Registry ──────────────────────────────────
#
# Procedures registered at runtime by external systems (e.g. Oikos
# loading pre-computed hedging strategies on wake). These are searched
# alongside the static _PROCEDURE_TEMPLATES during fast-path matching.
#
# Thread-safety: only written during hypnopompia (single writer, no
# concurrent reads since organism is sleeping). Safe without locks.

_DYNAMIC_PROCEDURES: list[dict[str, Any]] = []


def register_dynamic_procedure(procedure: dict[str, Any]) -> None:
    """Register a procedure template from outside Nova (e.g. Oikos hedge strategies)."""
    _DYNAMIC_PROCEDURES.append(procedure)


def clear_dynamic_procedures() -> None:
    """Clear all dynamically registered procedures (called at start of each sleep cycle wake)."""
    _DYNAMIC_PROCEDURES.clear()


def _broadcast_is_hungry(broadcast: object) -> bool:
    """
    Return True when the broadcast signals metabolic hunger (is_hungry=True).

    Hunger state is set by the Nova heartbeat on the broadcast it constructs
    when balance < hunger_balance_threshold_usd.  Metabolic survival overrides
    epistemic exploration — bounty_hunt must win regardless of success_rate.
    """
    try:
        # Check direct attribute
        if getattr(broadcast, "is_hungry", False):
            return True
        # Check inside content dict/object
        content = getattr(broadcast, "content", None)
        if content is not None:
            if getattr(content, "is_hungry", False):
                return True
            data = getattr(content, "data", None) or {}
            if isinstance(data, dict) and data.get("is_hungry"):
                return True
        # Check source tag
        source = str(getattr(broadcast, "source", ""))
        if "hunger" in source or "hungry" in source:
            return True
    except Exception:
        pass
    return False


def find_matching_procedure(broadcast: object) -> dict[str, Any] | None:
    """
    Pattern-match a broadcast against known procedure templates
    (static + dynamic). Returns the highest-success-rate matching
    template, or None. Must complete in ≤20ms.

    When the broadcast signals metabolic hunger (is_hungry=True), the
    bounty_hunt template is forced regardless of success_rate competition.
    Metabolic survival outranks epistemic exploration when hungry.
    """
    matches: list[dict[str, Any]] = []
    for template in _PROCEDURE_TEMPLATES + _DYNAMIC_PROCEDURES:
        try:
            if template["condition"](broadcast):
                matches.append(template)
        except Exception:
            pass  # Condition evaluation failure = no match
    if not matches:
        return None

    # Option B: hunger overrides success_rate ranking
    if _broadcast_is_hungry(broadcast):
        bounty_match = next(
            (t for t in matches if t.get("domain") == "economic"),
            None,
        )
        if bounty_match is not None:
            return bounty_match

    return max(matches, key=lambda t: t["success_rate"])


def procedure_to_policy(procedure: dict[str, Any]) -> Policy:
    """Convert a procedure template to a Policy."""
    return Policy(
        id=new_id(),
        name=procedure["name"],
        reasoning=f"Known reliable procedure (success rate: {procedure['success_rate']:.0%})",
        steps=[
            PolicyStep(
                action_type=s["action_type"],
                description=s["description"],
                parameters=s.get("parameters", {}),
            )
            for s in procedure["steps"]
        ],
        fallback_steps=[
            PolicyStep(
                action_type=s["action_type"],
                description=s["description"],
                parameters=s.get("parameters", {}),
            )
            for s in procedure.get("fallback_steps", [])
        ],
        estimated_effort=procedure.get("effort", "low"),
        time_horizon=procedure.get("time_horizon", "immediate"),
    )


# ─── Policy Generator ─────────────────────────────────────────────


class PolicyGenerator(BasePolicyGenerator):
    """
    Default Nova policy generator — uses LLM reasoning.

    For the slow path (deliberative processing), generates 2-5 distinct
    candidate policies grounded in the current belief state and goal.
    Always appends the DoNothing policy as the null baseline.

    The LLM call budget is up to 10000ms (within the 15000ms slow-path budget).
    """

    def __init__(
        self,
        llm: LLMProvider,
        instance_name: str = "EOS",
        max_policies: int = 5,
        timeout_ms: int = 10000,
    ) -> None:
        self._llm = llm
        self._instance_name = instance_name
        self._max_policies = max_policies
        self._timeout_ms = timeout_ms
        self._logger = logger.bind(system="nova.policy_generator")

    async def generate_candidates(
        self,
        goal: Goal,
        situation_summary: str,
        beliefs: BeliefState,
        affect: AffectState,
        memory_traces: list[dict[str, Any]] | None = None,
    ) -> list[Policy]:
        """
        Generate 2-5 candidate policies for achieving a goal.

        Always returns at least [DoNothingPolicy] even if LLM fails.
        The caller (EFEEvaluator) will score all candidates and select
        the minimum-EFE policy.
        """
        start = time.monotonic()
        traces = memory_traces or []

        prompt = build_policy_generation_prompt(
            instance_name=self._instance_name,
            goal=goal,
            situation_summary=situation_summary,
            beliefs_summary=summarise_beliefs(beliefs),
            memory_summary=summarise_memories(traces),
            affect=affect,
            available_action_types=AVAILABLE_ACTION_TYPES,
            max_policies=min(self._max_policies, 5),
        )

        try:
            response = await self._llm.generate(
                system_prompt=(
                    f"You are {self._instance_name}'s deliberative reasoning system. "
                    "Generate structured JSON policy candidates. "
                    "Be precise and creative. Output only valid JSON."
                ),
                messages=[Message(role="user", content=prompt)],
                max_tokens=2000,
                temperature=0.85,  # Creative — we want diverse candidates
                output_format="json",
            )

            elapsed_ms = int((time.monotonic() - start) * 1000)
            self._logger.debug("policy_generation_complete", elapsed_ms=elapsed_ms)

            parsed = _parse_policy_response(response.text)
            # Always append do-nothing as the null baseline
            parsed.append(make_do_nothing_policy())
            return parsed

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self._logger.warning(
                "policy_generation_failed",
                error=str(exc),
                elapsed_ms=elapsed_ms,
            )
            # Fallback: just the do-nothing policy
            return [make_do_nothing_policy()]


# ─── Response Parsing ─────────────────────────────────────────────


def _parse_policy_response(raw: str) -> list[Policy]:
    """
    Parse the LLM's JSON policy response into Policy objects.
    Robust to malformed output: any policies that parse successfully are kept.
    """
    try:
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        data = json.loads(text)
        policies_raw = data.get("policies", [])
        if not isinstance(policies_raw, list):
            return []

        policies: list[Policy] = []
        for p in policies_raw:
            try:
                steps: list[PolicyStep] = []
                for s in p.get("steps", []):
                    steps.append(PolicyStep(
                        action_type=str(s.get("action_type", "observe")),
                        description=str(s.get("description", "")),
                        parameters=dict(s.get("parameters", {})),
                        expected_duration_ms=int(s.get("duration_ms", 1000)),
                    ))
                policies.append(Policy(
                    id=new_id(),
                    name=str(p.get("name", "Unnamed policy"))[:80],
                    reasoning=str(p.get("reasoning", ""))[:400],
                    steps=steps,
                    risks=[str(r) for r in p.get("risks", [])],
                    epistemic_value_description=str(p.get("epistemic_value", ""))[:200],
                    estimated_effort=str(p.get("estimated_effort", "medium")),
                    time_horizon=str(p.get("time_horizon", "short")),
                ))
            except Exception:
                continue  # Skip malformed policies; don't fail the whole batch

        return policies

    except (json.JSONDecodeError, KeyError, TypeError):
        return []
