"""
EcodiaOS - Nova Policy Generator

Generates candidate action policies using LLM reasoning grounded in
the current belief state, goal, and relevant memory context.

The DoNothingPolicy is always included as a candidate - sometimes the
best action is no action (observe and wait). This prevents hyperactivity
and ensures Nova can choose inaction when uncertainty is high or the
situation is resolving on its own.

Policy generation uses the full slow-path budget (up to 10000ms for LLM call).
For fast-path decisions, pattern-matching against known procedure templates
is used instead, which must complete in ≤100ms.
"""

from __future__ import annotations

import dataclasses
import json
import random
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
    They can accept any constructor args they need - the ``NovaService``
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
        causal_laws_summary: str = "",
    ) -> list[Policy]:
        """
        Generate 2-5 candidate policies for achieving *goal*.

        Must always return at least one policy (the do-nothing fallback).
        Must never raise - return ``[make_do_nothing_policy()]`` on failure.
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
            "value - we learn from watching."
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
            # Only for real user dialogue - not internal/system events
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
    # ── Economic action ───────────────────────────────────────────────────────
    # Fires on internal/system broadcasts. The specific economic strategy
    # (bounty hunting, yield farming, cost optimisation, asset liquidation,
    # revenue diversification) is selected by generate_economic_intent() via
    # real EFE scoring against current beliefs — not keyword dispatch.
    # Hunger override still applies: if is_hungry, bounty_hunt is forced
    # regardless of EFE ranking (metabolic survival > epistemic exploration).
    {
        "name": "Economic action",
        "condition": lambda broadcast: (
            "system_event" in str(getattr(broadcast, "source", ""))
            or str(getattr(broadcast, "source", "")).startswith(
                ("internal:", "spontaneous", "nova:", "heartbeat")
            )
        ),
        "domain": "economic",
        # Steps are populated at match-time by find_matching_procedure() using
        # generate_economic_intent(). This placeholder is never used directly.
        "steps": [
            {
                "action_type": "bounty_hunt",
                "description": "Economic action — strategy selected by EFE scoring.",
                "parameters": {
                    "target_platforms": ["github", "algora"],
                    "min_reward_usd": 10.0,
                    "max_candidates": 20,
                },
            },
        ],
        "success_rate": 0.65,
        "effort": "medium",
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
        "condition": lambda broadcast: True,  # Always matches - lowest priority
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
    epistemic exploration - bounty_hunt must win regardless of success_rate.
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


# ─── Thompson Sampler ─────────────────────────────────────────────
#
# N-armed Beta-Bernoulli bandit for routing slow-path deliberation across
# an arbitrary set of LLM providers (Claude, RE/vLLM, Ollama, Bedrock, …).
#
# Each provider is an "arm" represented by Beta(alpha, beta). A sample is
# drawn per decision; the arm with the highest sample wins. Arms can be
# added at runtime (register_arm), enabled/disabled (set_arm_ready), and
# their outcomes recorded (record_outcome).
#
# Backward-compat: "claude" and "re" are the two default arms.
# set_re_ready(ready) is preserved as a convenience alias for
# set_arm_ready("re", ready).
#
# State is persisted to Redis key nova:thompson_sampler so the organism
# remembers routing history across restarts.


@dataclasses.dataclass
class ProviderMeta:
    """Metadata and Beta-distribution params for a single provider arm."""

    alpha: float = 1.0
    beta: float = 1.0
    ready: bool = False
    cost_per_token: float = 0.0       # USD per output token (0 = local/free)
    latency_estimate_ms: float = 2000.0
    capability_tags: list[str] = dataclasses.field(default_factory=list)


class ThompsonSampler:
    """
    N-armed Beta-Bernoulli Thompson sampler for LLM provider routing.

    Generalises the original Claude ↔ RE binary sampler to support any
    number of provider arms with dynamic registration, readiness gating,
    and runtime discovery.  Backward-compatible with all existing callers.
    """

    REDIS_KEY = "nova:thompson_sampler"

    def __init__(self) -> None:
        self._arms: dict[str, ProviderMeta] = {}
        self._logger = logger.bind(system="nova.thompson_sampler")
        # Register the two canonical arms (claude always ready, re starts not-ready)
        self.register_arm("claude", prior_alpha=1.0, prior_beta=1.0, ready=True)
        self.register_arm("re", prior_alpha=1.0, prior_beta=1.0, ready=False)

    # ── Arm management ────────────────────────────────────────────────

    def register_arm(
        self,
        name: str,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        ready: bool = False,
        cost_per_token: float = 0.0,
        latency_estimate_ms: float = 2000.0,
        capability_tags: list[str] | None = None,
    ) -> None:
        """Register a new provider arm (idempotent - existing arms are preserved)."""
        if name in self._arms:
            # Only update readiness; preserve accumulated Beta params.
            self._arms[name].ready = ready
            return
        self._arms[name] = ProviderMeta(
            alpha=prior_alpha,
            beta=prior_beta,
            ready=ready,
            cost_per_token=cost_per_token,
            latency_estimate_ms=latency_estimate_ms,
            capability_tags=capability_tags or [],
        )
        self._logger.info("thompson_arm_registered", name=name, ready=ready)

    def set_arm_ready(self, name: str, ready: bool) -> None:
        """Mark a provider arm as available (True) or unavailable (False)."""
        if name in self._arms:
            self._arms[name].ready = ready
            self._logger.info("thompson_arm_readiness", name=name, ready=ready)

    def set_re_ready(self, ready: bool) -> None:
        """Backward-compat alias for set_arm_ready('re', ready)."""
        self.set_arm_ready("re", ready)

    # ── Sampling ──────────────────────────────────────────────────────

    def sample(self) -> str:
        """
        Draw a Beta sample for every ready arm and return the winner.

        Falls back to 'claude' when no other arm is ready (claude is always
        registered and ready by default).
        """
        ready_arms = {n: m for n, m in self._arms.items() if m.ready}
        if not ready_arms:
            return "claude"
        if len(ready_arms) == 1:
            return next(iter(ready_arms))

        samples = {
            name: random.betavariate(meta.alpha, meta.beta)
            for name, meta in ready_arms.items()
        }
        winner = max(samples, key=lambda k: samples[k])
        self._logger.debug(
            "thompson_sample",
            samples={k: round(v, 4) for k, v in samples.items()},
            winner=winner,
        )
        return winner

    def sample_ranked(self) -> list[str]:
        """
        Return all ready arms ranked by their Beta draw (best first).

        Used by the fallback chain: try the winner; on failure, try next, etc.
        """
        ready_arms = {n: m for n, m in self._arms.items() if m.ready}
        if not ready_arms:
            return ["claude"]
        samples = {
            name: random.betavariate(meta.alpha, meta.beta)
            for name, meta in ready_arms.items()
        }
        return sorted(samples, key=lambda k: samples[k], reverse=True)

    # ── Outcome recording ─────────────────────────────────────────────

    def record_outcome(self, model: str, success: bool) -> None:
        """Update Beta params based on observed outcome."""
        if model not in self._arms:
            return
        if success:
            self._arms[model].alpha += 1.0
        else:
            self._arms[model].beta += 1.0

    # ── Introspection ─────────────────────────────────────────────────

    def get_success_rate(self, model: str = "re") -> float:
        """Return current expected success rate (Beta mean) for a model.

        Defaults to "re" - used by the safety layer to write eos:re:success_rate_7d.
        """
        meta = self._arms.get(model)
        if meta is None:
            return 0.5
        return meta.alpha / (meta.alpha + meta.beta)

    @property
    def means(self) -> dict[str, float]:
        """Current mean of each arm's Beta distribution (expected success rate)."""
        return {
            name: meta.alpha / (meta.alpha + meta.beta)
            for name, meta in self._arms.items()
        }

    # ── Redis persistence ─────────────────────────────────────────────

    async def load_from_redis(self, redis: Any) -> None:
        """Restore sampler state from Redis (call on startup)."""
        try:
            raw = await redis.hgetall(self.REDIS_KEY)
            if not raw:
                return
            # Generic N-arm keys: "{name}_alpha" / "{name}_beta"
            for name, meta in self._arms.items():
                alpha_key = f"{name}_alpha"
                beta_key = f"{name}_beta"
                if alpha_key in raw:
                    meta.alpha = float(raw[alpha_key])
                if beta_key in raw:
                    meta.beta = float(raw[beta_key])
            self._logger.info(
                "thompson_sampler_restored",
                arms={n: round(m.alpha / (m.alpha + m.beta), 4) for n, m in self._arms.items()},
            )
        except Exception as exc:
            self._logger.debug("thompson_sampler_load_failed", error=str(exc))

    async def persist_to_redis(self, redis: Any) -> None:
        """Persist all arm states and RE success rate to Redis (fire-and-forget)."""
        try:
            mapping: dict[str, str] = {}
            for name, meta in self._arms.items():
                mapping[f"{name}_alpha"] = str(meta.alpha)
                mapping[f"{name}_beta"] = str(meta.beta)
            await redis.hset(self.REDIS_KEY, mapping=mapping)
            # Write RE success rate to canonical keys consumed by:
            #   - ContinualLearningOrchestrator (degradation trigger + kill switch)
            #   - RESuccessRateMonitor (Tier 2 safety kill switch)
            #   - Benchmarks (RE performance KPI)
            re_rate = self.get_success_rate("re")
            await redis.set("eos:re:thompson_success_rate", str(round(re_rate, 6)))
            await redis.set("eos:re:success_rate_7d", str(round(re_rate, 6)))
        except Exception as exc:
            self._logger.debug("thompson_sampler_persist_failed", error=str(exc))


# ─── Provider Health Monitor ───────────────────────────────────────
#
# Tracks per-provider health signals (consecutive failures, latency EMA).
# Removes arms from rotation on 3 consecutive failures and re-enables them
# after a lightweight periodic probe succeeds.  This means if vLLM dies
# the organism automatically routes to Claude; when vLLM recovers it
# automatically returns - no operator intervention required.


class ProviderHealthMonitor:
    """
    Autonomous health tracking for registered LLM provider arms.

    Wired into PolicyGenerator.generate_candidates() - every call outcome
    is reported here.  The monitor gates arms in/out of the Thompson
    sampler based on observed reliability.
    """

    FAILURE_THRESHOLD = 3           # consecutive failures before arm disabled
    PROBE_INTERVAL_CYCLES = 100     # cycles between re-enabling probes for downed arms

    def __init__(self, sampler: ThompsonSampler) -> None:
        self._sampler = sampler
        self._consecutive_failures: dict[str, int] = {}
        self._latency_ema: dict[str, float] = {}   # ms
        self._cycle_counter: int = 0
        self._logger = logger.bind(system="nova.provider_health")

    def record_call(self, provider: str, success: bool, latency_ms: float) -> None:
        """
        Record an actual provider call outcome.

        On FAILURE_THRESHOLD consecutive failures the arm is disabled.
        On any success the consecutive failure counter resets.
        """
        if success:
            self._consecutive_failures[provider] = 0
            # EMA latency (α=0.2)
            prev = self._latency_ema.get(provider, latency_ms)
            self._latency_ema[provider] = 0.8 * prev + 0.2 * latency_ms
        else:
            count = self._consecutive_failures.get(provider, 0) + 1
            self._consecutive_failures[provider] = count
            if count >= self.FAILURE_THRESHOLD:
                self._sampler.set_arm_ready(provider, False)
                self._logger.warning(
                    "provider_arm_disabled",
                    provider=provider,
                    consecutive_failures=count,
                )

    def on_cycle(self) -> list[str]:
        """
        Called once per policy-generation cycle.

        Every PROBE_INTERVAL_CYCLES, returns a list of downed arm names
        that callers should probe.  On probe success, callers should call
        re_enable(provider).
        """
        self._cycle_counter += 1
        if self._cycle_counter % self.PROBE_INTERVAL_CYCLES != 0:
            return []
        downed = [
            name
            for name, meta in self._sampler._arms.items()
            if not meta.ready and name != "claude"   # claude is never downed
        ]
        return downed

    def re_enable(self, provider: str) -> None:
        """Re-enable a downed arm after a successful health probe."""
        self._consecutive_failures[provider] = 0
        self._sampler.set_arm_ready(provider, True)
        self._logger.info("provider_arm_re_enabled", provider=provider)

    def get_latency_ema(self, provider: str) -> float | None:
        """Return latency EMA in ms for a provider, or None if unknown."""
        return self._latency_ema.get(provider)


# ─── Policy Generator ─────────────────────────────────────────────


class PolicyGenerator(BasePolicyGenerator):
    """
    Default Nova policy generator - uses LLM reasoning.

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
        thompson_sampler: ThompsonSampler | None = None,
        re_client: Any = None,
        bounty_min_reward_usd: float = 10.0,
        bounty_max_candidates: int = 20,
    ) -> None:
        self._llm = llm
        self._instance_name = instance_name
        self._max_policies = max_policies
        self._timeout_ms = timeout_ms
        self._logger = logger.bind(system="nova.policy_generator")
        # N-armed Thompson sampler for provider routing
        self._sampler: ThompsonSampler = thompson_sampler or ThompsonSampler()
        # Health monitor - tracks consecutive failures and latency EMA per arm
        self._health_monitor: ProviderHealthMonitor = ProviderHealthMonitor(self._sampler)
        # RE client - the local vLLM reasoning engine (arm="re")
        self._re_client: Any = re_client
        # Dynamic clients for arms beyond "claude" and "re"
        # key: arm_name, value: object with .generate() matching LLMProvider interface
        self._extra_clients: dict[str, Any] = {}
        # Synapse reference - needed to emit REASONING_CAPABILITY_DEGRADED
        self._synapse: Any = None
        # ActionTypeRegistry - runtime list of action types for LLM prompts.
        # None until NovaService wires it via set_action_type_registry().
        # When None, falls back to the static AVAILABLE_ACTION_TYPES list.
        self._action_type_registry: Any = None
        # Configurable bounty template params (learned by Simula via ADJUST_BUDGET)
        self._bounty_min_reward_usd: float = bounty_min_reward_usd
        self._bounty_max_candidates: int = bounty_max_candidates
        # Patch the module-level bounty template so fast-path also uses configured values
        self._patch_bounty_template(bounty_min_reward_usd, bounty_max_candidates)

    def set_synapse(self, synapse: Any) -> None:
        """Wire Synapse so REASONING_CAPABILITY_DEGRADED can be emitted."""
        self._synapse = synapse

    def set_action_type_registry(self, registry: Any) -> None:
        """
        Wire the ActionTypeRegistry so generate_candidates() reads from the
        live registry instead of the static AVAILABLE_ACTION_TYPES list.

        Called by NovaService after creating both objects during initialize().
        """
        self._action_type_registry = registry

    def register_provider(
        self,
        name: str,
        client: Any,
        ready: bool = True,
        cost_per_token: float = 0.0,
        latency_estimate_ms: float = 2000.0,
        capability_tags: list[str] | None = None,
    ) -> None:
        """
        Register a dynamic provider arm at runtime.

        Call this when a new vLLM endpoint comes online, Ollama is available
        locally, or Evo proposes a new substrate.  The arm is immediately
        eligible for Thompson sampling.

        Args:
            name: Unique arm identifier (e.g. "re_v2", "ollama_local").
            client: Object with .generate() matching LLMProvider interface.
            ready: Whether the arm is immediately available for routing.
            cost_per_token: USD per output token (0 for local/free providers).
            latency_estimate_ms: Expected latency hint for observability.
            capability_tags: Metadata tags (e.g. ["code", "math"]).
        """
        self._extra_clients[name] = client
        self._sampler.register_arm(
            name,
            ready=ready,
            cost_per_token=cost_per_token,
            latency_estimate_ms=latency_estimate_ms,
            capability_tags=capability_tags or [],
        )
        self._logger.info(
            "dynamic_provider_registered",
            name=name,
            ready=ready,
            cost_per_token=cost_per_token,
        )

    @staticmethod
    def _patch_bounty_template(min_reward_usd: float, max_candidates: int) -> None:
        """Patch the module-level bounty template with configurable values.

        Called at init time so both fast-path (find_matching_procedure) and
        generate_economic_intent() use the same configured parameters.
        Simula can mutate these by constructing a new PolicyGenerator with
        updated ADJUST_BUDGET values.
        """
        for template in _PROCEDURE_TEMPLATES:
            if template.get("name") == "Autonomous bounty hunting":
                for step in template.get("steps", []):
                    if step.get("action_type") == "bounty_hunt":
                        step.setdefault("parameters", {})
                        step["parameters"]["min_reward_usd"] = min_reward_usd
                        step["parameters"]["max_candidates"] = max_candidates
                break

    async def generate_candidates(
        self,
        goal: Goal,
        situation_summary: str,
        beliefs: BeliefState,
        affect: AffectState,
        memory_traces: list[dict[str, Any]] | None = None,
        causal_laws_summary: str = "",
    ) -> list[Policy]:
        """
        Generate 2-5 candidate policies for achieving a goal.

        Routes the LLM call through the Thompson sampler - either to the
        Claude API (incumbent) or the local RE (when available and winning).
        The model that handled the call is recorded in `_last_model_used`
        so the caller can stamp `model_used` on the DecisionRecord.

        Always returns at least [DoNothingPolicy] even if LLM fails.
        The caller (EFEEvaluator) will score all candidates and select
        the minimum-EFE policy.
        """
        start = time.monotonic()
        traces = memory_traces or []

        # Use the live registry if wired; fall back to the static list.
        if self._action_type_registry is not None:
            available_types = self._action_type_registry.get_available()
        else:
            available_types = AVAILABLE_ACTION_TYPES

        prompt = build_policy_generation_prompt(
            instance_name=self._instance_name,
            goal=goal,
            situation_summary=situation_summary,
            beliefs_summary=summarise_beliefs(beliefs),
            memory_summary=summarise_memories(traces),
            affect=affect,
            available_action_types=available_types,
            max_policies=self._max_policies,
            causal_laws_summary=causal_laws_summary,
        )

        # ── Thompson sampling: ranked fallback chain ──────────────────────
        # sample_ranked() draws Beta samples for all ready arms and returns
        # them best-first. We try each in order; on failure we record the
        # failure and move to the next arm. If ALL arms fail we emit
        # REASONING_CAPABILITY_DEGRADED and fall back to do-nothing.
        ranked_arms = self._sampler.sample_ranked()
        self._last_model_used: str = ranked_arms[0] if ranked_arms else "claude"

        # Notify the health monitor so it can probe downed arms periodically.
        arms_to_probe = self._health_monitor.on_cycle()
        if arms_to_probe:
            self._logger.debug("provider_probe_due", arms=arms_to_probe)

        system_prompt = f"Policy generation for {self._instance_name}. Respond as JSON."
        messages = [Message(role="user", content=prompt)]

        last_exc: Exception | None = None
        for arm_name in ranked_arms:
            arm_start = time.monotonic()
            try:
                if arm_name == "re" and self._re_client is not None:
                    response = await self._re_client.generate(
                        system_prompt=system_prompt,
                        messages=messages,
                        max_tokens=4000,
                        temperature=0.9,
                        output_format="json",
                    )
                elif arm_name == "claude":
                    response = await self._llm.generate(
                        system_prompt=system_prompt,
                        messages=messages,
                        max_tokens=4000,
                        temperature=0.9,
                        output_format="json",
                    )
                else:
                    # Dynamic arm (Ollama, Bedrock, re_v2, …): must be in _extra_clients
                    client = self._extra_clients.get(arm_name)
                    if client is None:
                        raise RuntimeError(f"No client registered for arm '{arm_name}'")
                    response = await client.generate(
                        system_prompt=system_prompt,
                        messages=messages,
                        max_tokens=4000,
                        temperature=0.9,
                        output_format="json",
                    )

                latency_ms = int((time.monotonic() - arm_start) * 1000)
                self._health_monitor.record_call(arm_name, success=True, latency_ms=latency_ms)
                self._last_model_used = arm_name

                elapsed_ms = int((time.monotonic() - start) * 1000)
                self._logger.debug(
                    "policy_generation_complete",
                    elapsed_ms=elapsed_ms,
                    model=arm_name,
                    tried_arms=ranked_arms[: ranked_arms.index(arm_name) + 1],
                )
                parsed = _parse_policy_response(response.text)
                parsed.append(make_do_nothing_policy())
                return parsed

            except Exception as exc:
                latency_ms = int((time.monotonic() - arm_start) * 1000)
                self._health_monitor.record_call(arm_name, success=False, latency_ms=latency_ms)
                self._sampler.record_outcome(arm_name, success=False)
                last_exc = exc
                self._logger.warning(
                    "provider_arm_failed",
                    arm=arm_name,
                    error=str(exc),
                    remaining=ranked_arms[ranked_arms.index(arm_name) + 1 :],
                )
                # Continue to next arm in the fallback chain

        # All arms failed - emit REASONING_CAPABILITY_DEGRADED if synapse is wired.
        elapsed_ms = int((time.monotonic() - start) * 1000)
        self._logger.error(
            "all_provider_arms_failed",
            elapsed_ms=elapsed_ms,
            tried_arms=ranked_arms,
            error=str(last_exc),
        )
        if self._synapse is not None:
            import asyncio
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                event = SynapseEvent(
                    event_type=SynapseEventType.REASONING_CAPABILITY_DEGRADED,
                    source_system="nova",
                    data={
                        "tried_arms": ranked_arms,
                        "error": str(last_exc),
                        "elapsed_ms": elapsed_ms,
                    },
                )
                asyncio.ensure_future(self._synapse.broadcast(event))
            except Exception:
                pass  # Never block the fallback path
        return [make_do_nothing_policy()]

    def generate_economic_intent(
        self,
        beliefs: BeliefState,
        economic_context: dict[str, Any] | None = None,
    ) -> Policy:
        """
        Select the optimal economic policy via EFE scoring.

        Scores 5 economic strategies against current beliefs using a lightweight
        EFE proxy — epistemic value (learning gain) + pragmatic value (expected
        economic gain) - risk penalty — and returns the minimum-EFE policy.

        All scoring weights are derived from live belief entity confidence signals,
        not hardcoded constants.
        """
        ctx = economic_context or {}
        balance = float(ctx.get("wallet_balance_usd", 100.0))
        burn_rate = float(ctx.get("burn_rate_hourly_usd", 1.0))
        hours_until_depleted = balance / max(burn_rate, 0.01)

        # Belief entity confidence signals (0.0 = no knowledge, 1.0 = certain)
        bounty_rate = beliefs.entities.get("bounty_success_rate")
        bounty_conf = bounty_rate.confidence if bounty_rate else 0.5
        yield_ent = beliefs.entities.get("yield_apy_aave") or beliefs.entities.get("yield_apy_morpho")
        yield_conf = yield_ent.confidence if yield_ent else 0.3
        econ_risk = beliefs.entities.get("economic_risk_level")
        risk_conf = econ_risk.confidence if econ_risk else 0.3
        # Liquidity urgency: scales from 0 (weeks of runway) to 1 (hours of runway)
        liquidity_urgency = max(0.0, min(1.0, 1.0 - (hours_until_depleted / 168.0)))

        # EFE proxy scores (lower = better, most negative wins)
        # Each score = -(pragmatic_value) + risk_penalty - epistemic_bonus
        scores: list[tuple[float, str, list[PolicyStep]]] = [
            # Bounty hunting: pragmatic value scales with confidence we can win bounties
            (
                -bounty_conf + risk_conf * (1.0 - bounty_conf),
                "Autonomous bounty hunting",
                [PolicyStep(
                    action_type="bounty_hunt",
                    description="Discover, select, and solve open bounties for revenue.",
                    parameters={
                        "target_platforms": ["github", "algora"],
                        "min_reward_usd": self._bounty_min_reward_usd,
                        "max_candidates": self._bounty_max_candidates,
                    },
                )],
            ),
            # Yield farming: pragmatic value scales with yield knowledge; capital-gated
            (
                -yield_conf + (liquidity_urgency * 0.5),
                "Yield farming deployment",
                [PolicyStep(
                    action_type="defi_yield",
                    description="Deploy idle USDC into highest-APY DeFi protocol.",
                    parameters={
                        "protocols": ["aave", "morpho", "compound"],
                        "rebalance_interval_h": 24,
                    },
                )],
            ),
            # Cost optimisation: high baseline pragmatic value, scales with risk awareness
            (
                -(0.7 + risk_conf * 0.3),
                "Cost optimization sweep",
                [PolicyStep(
                    action_type="store_insight",
                    description="Audit LLM token spend, SACM substrate costs, and API usage.",
                    parameters={
                        "insight": "Cost optimization audit: identify top expense categories.",
                        "domain": "economic",
                        "confidence": 0.7,
                        "tags": ["cost_optimization", "burn_rate"],
                    },
                )],
            ),
            # Asset liquidation: only preferred when genuinely critical
            (
                -(liquidity_urgency * 0.8) + (1.0 - liquidity_urgency) * 0.4,
                "Asset liquidation for liquidity",
                [PolicyStep(
                    action_type="wallet_transfer",
                    description="Liquidate held assets to restore liquid balance.",
                    parameters={"priority": "survival"},
                )],
            ),
            # Revenue diversification: low pragmatic, high epistemic — use when stuck
            (
                -(0.3 + (1.0 - bounty_conf) * (1.0 - yield_conf) * 0.4),
                "Revenue stream diversification",
                [PolicyStep(
                    action_type="store_insight",
                    description="Research and evaluate new revenue streams.",
                    parameters={
                        "insight": "Revenue diversification analysis.",
                        "domain": "economic",
                        "confidence": 0.4,
                        "tags": ["revenue_diversification"],
                    },
                )],
            ),
        ]

        # Select strategy with minimum EFE
        best_score, best_name, best_steps = min(scores, key=lambda x: x[0])

        self._logger.debug(
            "economic_intent_selected",
            winner=best_name,
            efe=round(best_score, 4),
            bounty_conf=round(bounty_conf, 3),
            yield_conf=round(yield_conf, 3),
            risk_conf=round(risk_conf, 3),
            liquidity_urgency=round(liquidity_urgency, 3),
            hours_until_depleted=round(hours_until_depleted, 1),
        )

        return Policy(
            id=new_id(),
            name=best_name,
            reasoning=f"Economic EFE selection (score={best_score:.3f}, liquidity={liquidity_urgency:.2f})",
            steps=best_steps,
            estimated_effort="medium",
            time_horizon="short",
        )

    def record_outcome(
        self,
        intent_id: str,
        success: bool,
        redis: Any = None,
    ) -> None:
        """Record an intent outcome into the Thompson sampler and health monitor.

        Called by NovaService._on_axon_execution_result() with the model that
        handled the most recent slow-path call.  Routes to the sampler using
        _last_model_used so we attribute the outcome to the correct arm.

        Also triggers a fire-and-forget Redis persist if redis is provided,
        which updates eos:re:success_rate_7d and eos:re:thompson_success_rate.
        """
        model = getattr(self, "_last_model_used", "claude")
        self._sampler.record_outcome(model, success)
        # Update health monitor so post-call outcome (e.g. Equor/Axon) feeds into
        # consecutive-failure tracking (latency unknown here → use 0).
        self._health_monitor.record_call(model, success=success, latency_ms=0.0)
        if redis is not None:
            import asyncio
            try:
                asyncio.ensure_future(self._sampler.persist_to_redis(redis))
            except Exception:
                pass  # Swallow - never block the outcome recording path


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
