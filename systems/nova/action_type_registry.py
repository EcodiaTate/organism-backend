"""
EcodiaOS - Nova Action Type Registry

A runtime registry of action types available to Nova's deliberation engine.

At startup the registry is pre-populated with the 18 static action types that
have corresponding Axon executors.  New types can be added at runtime when
Simula generates and hot-loads a novel executor in response to a
NOVEL_ACTION_REQUESTED event.

The registry is the *single source of truth* for what Nova's LLM is told it
can do.  prompts/nova/policy.py no longer holds a hardcoded list - it asks the
registry for the current set.

Usage
-----
    from systems.nova.action_type_registry import ActionTypeRegistry

    registry = ActionTypeRegistry()
    # pre-populated with static types on first call to get_available()
    types = registry.get_available()       # list[str] suitable for prompt
    registry.register_dynamic(
        name="my_new_action",
        description="Does something novel",
        capabilities=["http_client"],
        risk_tier="low",
    )
    assert registry.is_registered("my_new_action")
"""

from __future__ import annotations

import dataclasses
import threading
from datetime import datetime
from typing import Any

import structlog

logger = structlog.get_logger()

# ─── Data model ───────────────────────────────────────────────────────────────


@dataclasses.dataclass
class ActionTypeEntry:
    """A single registered action type."""

    name: str
    description: str
    # One-line summary suitable for inclusion in the policy-generation prompt.
    # Format: "{name}: {description}"
    prompt_line: str
    # Whether this was generated at runtime by Simula vs. present at startup.
    is_dynamic: bool = False
    # Capabilities the executor requires (for feasibility checks).
    capabilities: list[str] = dataclasses.field(default_factory=list)
    # Risk tier: "low" | "medium" | "high"
    risk_tier: str = "low"
    # How many times this action type has been used; tracked for Evo promotion.
    use_count: int = 0
    # Success rate [0.0, 1.0]; updated by NovaService.process_outcome().
    success_rate: float = 1.0
    # ISO-8601 timestamp when the dynamic type was registered (static = "")
    registered_at: str = ""
    # Equor pre-approved this type before first use.
    equor_approved: bool = True
    # Evo hypothesis ID tracking effectiveness (dynamic types only).
    source_hypothesis_id: str = ""


# ─── Static action types (matches Axon executors) ─────────────────────────────
#
# These are the 18 canonical types that have always existed.  The descriptions
# here are richer than the one-liners in the old AVAILABLE_ACTION_TYPES list
# because the registry also drives capability checks - not just prompt text.

_STATIC_ENTRIES: list[dict[str, Any]] = [
    # ── Voxis-routed (expression) ──────────────────────────────────────────
    {
        "name": "express",
        "description": "Send a text message or response to the user/community",
        "capabilities": ["voxis_expression"],
        "risk_tier": "low",
    },
    {
        "name": "request_info",
        "description": "Ask a clarifying question of a community member",
        "capabilities": ["voxis_expression"],
        "risk_tier": "low",
    },
    # ── Axon-routed (internal cognition) ──────────────────────────────────
    {
        "name": "store_insight",
        "description": "Persist a structured insight or learning to long-term memory",
        "capabilities": ["memory_write"],
        "risk_tier": "low",
    },
    {
        "name": "query_memory",
        "description": "Retrieve relevant memories or past experiences to inform action",
        "capabilities": ["memory_read"],
        "risk_tier": "low",
    },
    {
        "name": "update_goal",
        "description": "Revise progress, priority, or status of an existing goal",
        "capabilities": ["goal_management"],
        "risk_tier": "low",
    },
    {
        "name": "analyse",
        "description": "Examine a piece of information deeply to extract meaning or patterns",
        "capabilities": ["llm_reasoning"],
        "risk_tier": "low",
    },
    {
        "name": "search",
        "description": "Search for external information relevant to the current goal",
        "capabilities": ["http_client"],
        "risk_tier": "low",
    },
    {
        "name": "trigger_consolidation",
        "description": (
            "Initiate a learning consolidation cycle to integrate recent experiences"
        ),
        "capabilities": ["oneiros_trigger"],
        "risk_tier": "low",
    },
    # ── Axon-routed (communication / scheduling) ──────────────────────────
    {
        "name": "respond_text",
        "description": "Compose and send a structured text response via Axon pipeline",
        "capabilities": ["voxis_expression"],
        "risk_tier": "low",
    },
    {
        "name": "schedule_event",
        "description": "Create a scheduled event or reminder",
        "capabilities": ["calendar_write"],
        "risk_tier": "low",
    },
    # ── Axon-routed (foraging / bounty solving) ───────────────────────────
    {
        "name": "bounty_hunt",
        "description": (
            "Full autonomous bounty-hunt loop - discover open bounties, select the best "
            "candidate, generate a real solution via LLM or Simula, and stage it for PR "
            "submission. Use this when the goal is 'earn revenue now'. "
            "Optional parameters: target_platforms (list, default [github, algora]), "
            "min_reward_usd (float, default 10.0), max_candidates (int, default 20)"
        ),
        "capabilities": ["http_client", "code_generation", "git_write"],
        "risk_tier": "medium",
    },
    {
        "name": "hunt_bounties",
        "description": (
            "Scan GitHub/Algora for paid bounty issues and evaluate against BountyPolicy"
        ),
        "capabilities": ["http_client"],
        "risk_tier": "low",
    },
    {
        "name": "solve_bounty",
        "description": (
            "Solve a discovered bounty by cloning the repo, generating code via Simula's "
            "evolution pipeline, and submitting a PR. "
            "Required parameters: bounty_id, issue_url (HTTPS), repository_url "
            "(owner/repo or HTTPS URL), title (issue title), description (issue body). "
            "Optional: reward_usd, difficulty, labels, platform"
        ),
        "capabilities": ["http_client", "code_generation", "git_write"],
        "risk_tier": "medium",
    },
    # ── Internal (no delivery) ────────────────────────────────────────────
    {
        "name": "observe",
        "description": "Continue monitoring without acting (gather more information)",
        "capabilities": [],
        "risk_tier": "low",
    },
    {
        "name": "wait",
        "description": "Pause and let the situation develop",
        "capabilities": [],
        "risk_tier": "low",
    },
    # ── Economic / DeFi ───────────────────────────────────────────────────
    {
        "name": "defi_yield",
        "description": (
            "Deploy capital to a DeFi yield protocol (Aave / Morpho / Compound). "
            "Required autonomy level: SOVEREIGN (4). "
            "Parameters: protocol, amount_usd, action (deploy|withdraw)"
        ),
        "capabilities": ["defi_write", "wallet_access"],
        "risk_tier": "high",
    },
    {
        "name": "wallet_transfer",
        "description": (
            "Transfer USDC to an external address. Requires SOVEREIGN autonomy. "
            "Parameters: to_address, amount_usd, reason"
        ),
        "capabilities": ["wallet_access"],
        "risk_tier": "high",
    },
    {
        "name": "spawn_child",
        "description": (
            "Spawn a specialised child instance via Mitosis. Requires SOVEREIGN autonomy "
            "and Oikos metabolic gate approval. "
            "Parameters: specialisation, seed_capital_usd"
        ),
        "capabilities": ["mitosis_spawn"],
        "risk_tier": "high",
    },
]


# ─── Registry class ────────────────────────────────────────────────────────────


class ActionTypeRegistry:
    """
    Runtime registry of action types available to Nova's deliberation engine.

    Thread-safe: a threading.Lock protects all mutations.

    Usage
    -----
        registry = ActionTypeRegistry()
        types = registry.get_available()          # list[str] for prompt
        registry.register_dynamic(...)            # add Simula-generated type
        registry.record_outcome("my_action", True)  # track success rate
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, ActionTypeEntry] = {}
        self._logger = logger.bind(system="nova.action_type_registry")
        self._populate_static()

    def _populate_static(self) -> None:
        for spec in _STATIC_ENTRIES:
            name = spec["name"]
            desc = spec["description"]
            entry = ActionTypeEntry(
                name=name,
                description=desc,
                prompt_line=f"{name}: {desc}",
                is_dynamic=False,
                capabilities=list(spec.get("capabilities", [])),
                risk_tier=spec.get("risk_tier", "low"),
                equor_approved=True,
            )
            self._entries[name] = entry

    # ── Read API ──────────────────────────────────────────────────────────

    def get_available(self) -> list[str]:
        """
        Return all action type prompt lines in a stable order suitable for
        inclusion in the policy-generation prompt.

        Static types come first (in their original order), then dynamic types
        sorted alphabetically.
        """
        with self._lock:
            static = [
                e.prompt_line
                for e in self._entries.values()
                if not e.is_dynamic
            ]
            dynamic = sorted(
                (e.prompt_line for e in self._entries.values() if e.is_dynamic),
            )
            return static + dynamic

    def is_registered(self, name: str) -> bool:
        with self._lock:
            return name in self._entries

    def get_entry(self, name: str) -> ActionTypeEntry | None:
        with self._lock:
            return self._entries.get(name)

    def list_dynamic(self) -> list[ActionTypeEntry]:
        """Return all dynamically registered action types."""
        with self._lock:
            return [e for e in self._entries.values() if e.is_dynamic]

    # ── Write API ─────────────────────────────────────────────────────────

    def register_dynamic(
        self,
        name: str,
        description: str,
        capabilities: list[str] | None = None,
        risk_tier: str = "medium",
        equor_approved: bool = True,
        source_hypothesis_id: str = "",
    ) -> ActionTypeEntry:
        """
        Register a novel action type generated by Simula.

        If ``name`` is already registered as a static type, a ValueError is
        raised - dynamic types cannot shadow static ones.  If it is already
        registered as a dynamic type the existing entry is returned unchanged
        (idempotent re-registration).
        """
        with self._lock:
            existing = self._entries.get(name)
            if existing is not None:
                if not existing.is_dynamic:
                    raise ValueError(
                        f"Cannot register dynamic action type {name!r}: "
                        "a static type with that name already exists."
                    )
                # Idempotent - already registered.
                return existing

            entry = ActionTypeEntry(
                name=name,
                description=description,
                prompt_line=f"{name}: {description}",
                is_dynamic=True,
                capabilities=list(capabilities or []),
                risk_tier=risk_tier,
                equor_approved=equor_approved,
                registered_at=datetime.utcnow().isoformat(),
                source_hypothesis_id=source_hypothesis_id,
            )
            self._entries[name] = entry

        self._logger.info(
            "novel_action_type_registered",
            name=name,
            risk_tier=risk_tier,
            equor_approved=equor_approved,
        )
        return entry

    def deprecate(self, name: str) -> bool:
        """
        Remove a dynamic action type from the registry.

        Only dynamic types can be deprecated; static types are permanent.
        Returns True if removed, False if not found or is static.
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is None or not entry.is_dynamic:
                return False
            del self._entries[name]

        self._logger.info("novel_action_type_deprecated", name=name)
        return True

    def record_outcome(self, name: str, success: bool) -> None:
        """
        Update use_count and success_rate for the named action type.

        Called by NovaService.process_outcome() after each execution.
        Uses exponential moving average (α = 0.1) for the success rate.
        """
        with self._lock:
            entry = self._entries.get(name)
            if entry is None:
                return
            entry.use_count += 1
            alpha = 0.1
            entry.success_rate = (
                alpha * (1.0 if success else 0.0) + (1.0 - alpha) * entry.success_rate
            )
