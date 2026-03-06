"""
EcodiaOS — Axon Executor Registry

The registry maps action type names to their Executor instances.

Registration happens at AxonService initialisation. The registry is then
immutable at runtime — no executor can be added or removed during a cognitive
cycle. This makes executor lookup O(1) and side-effect-free.

Lookup normalisation:
  Intents use dot-notation executor names (e.g. "executor.observe", "data.store").
  The registry normalises these to the canonical action_type used at registration.
  Normalisation strips a leading "executor." prefix and maps common aliases.

This means executors register under their canonical name ("observe", "call_api")
and the registry handles the dotted forms that Nova's PolicyGenerator may produce.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from systems.axon.executor import Executor

logger = structlog.get_logger()

# Map dotted/aliased executor names → canonical action_type
_ALIAS_MAP: dict[str, str] = {
    # Observation
    "executor.observe": "observe",
    "executor.query_memory": "query_memory",
    "executor.analyse": "analyse",
    "executor.search": "search",
    # Communication
    "executor.respond": "respond_text",
    "respond": "respond_text",
    "executor.respond_text": "respond_text",
    "executor.notify": "send_notification",
    "notify": "send_notification",
    "executor.notification": "send_notification",
    "executor.post": "post_message",
    "post": "post_message",
    "executor.post_message": "post_message",
    "executor.email": "send_email",
    # Data operations
    "executor.create": "create_record",
    "create": "create_record",
    "executor.create_record": "create_record",
    "executor.update": "update_record",
    "update": "update_record",
    "executor.update_record": "update_record",
    "executor.schedule": "schedule_event",
    "schedule": "schedule_event",
    "executor.reminder": "set_reminder",
    "reminder": "set_reminder",
    # Integration
    "executor.api": "call_api",
    "api": "call_api",
    "executor.call_api": "call_api",
    "executor.webhook": "webhook_trigger",
    "webhook": "webhook_trigger",
    # Internal
    "executor.store": "store_insight",
    "store": "store_insight",
    "executor.store_insight": "store_insight",
    "executor.update_goal": "update_goal",
    "executor.consolidate": "trigger_consolidation",
    "consolidate": "trigger_consolidation",
    # Resource & config
    "executor.allocate": "allocate_resource",
    "executor.config": "adjust_config",
    # Federation
    "executor.federate": "federation_send",
    "federate": "federation_send",
    "executor.federation_send": "federation_send",
    "executor.federation_share": "federation_share",
    # Financial — transfers
    "executor.wallet_transfer": "wallet_transfer",
    "executor.transfer": "wallet_transfer",
    "transfer": "wallet_transfer",
    "send_funds": "wallet_transfer",
    "executor.send_funds": "wallet_transfer",
    # Financial — funding requests (survival plea)
    "executor.request_funding": "request_funding",
    "request_funding": "request_funding",
    "executor.funding_request": "request_funding",
    "funding_request": "request_funding",
    "ask_for_funds": "request_funding",
    "executor.ask_for_funds": "request_funding",
    "plea_for_capital": "request_funding",
    # Financial — DeFi yield (resting metabolism)
    "executor.defi_yield": "defi_yield",
    "yield": "defi_yield",
    "deploy_yield": "defi_yield",
    "executor.deploy_yield": "defi_yield",
    "resting_metabolism": "defi_yield",
    "executor.resting_metabolism": "defi_yield",
    # Financial — Phantom Liquidity sensor network (Phase 16q)
    "executor.phantom_liquidity": "phantom_liquidity",
    "phantom_lp": "phantom_liquidity",
    "executor.phantom_lp": "phantom_liquidity",
    "sensor_liquidity": "phantom_liquidity",
    "executor.sensor_liquidity": "phantom_liquidity",
    "deploy_sensor": "phantom_liquidity",
    "executor.deploy_sensor": "phantom_liquidity",
    # Entrepreneurship — asset deployment (Phase 16d)
    "executor.deploy_asset": "deploy_asset",
    "deploy": "deploy_asset",
    "create_asset": "deploy_asset",
    "executor.create_asset": "deploy_asset",
    "launch_asset": "deploy_asset",
    "executor.launch_asset": "deploy_asset",
    # Mitosis — child spawning (Phase 16e)
    "executor.spawn_child": "spawn_child",
    "spawn": "spawn_child",
    "reproduce": "spawn_child",
    "mitosis": "spawn_child",
    "executor.mitosis": "spawn_child",
    # Mitosis — dividend collection (Phase 16e)
    "executor.collect_dividend": "collect_dividend",
    "dividend": "collect_dividend",
    "executor.dividend": "collect_dividend",
    "receive_dividend": "collect_dividend",
    # Federated Telecom Marketplace — phone number provisioning (Phase 16j)
    "executor.request_telecom": "request_telecom",
    "request_telecom": "request_telecom",
    "provision_phone": "request_telecom",
    "executor.provision_phone": "request_telecom",
    "get_phone_number": "request_telecom",
    # Foraging — bounty solving (Phase 16b / Phase 3 solve loop)
    "executor.solve_bounty": "axon.solve_bounty",
    "solve_bounty": "axon.solve_bounty",
    "executor.solve": "axon.solve_bounty",
    "solve": "axon.solve_bounty",
    "solve_issue": "axon.solve_bounty",
    "executor.solve_issue": "axon.solve_bounty",
    # Remote compute — SACM bridge (Section XI)
    "executor.remote_compute": "remote_compute",
    "compute": "remote_compute",
    "executor.compute": "remote_compute",
    "sacm": "remote_compute",
    "executor.sacm": "remote_compute",
    "remote_exec": "remote_compute",
    "executor.remote_exec": "remote_compute",
    # Fast-path arbitrage (Reflex Arc)
    "executor.fast_path_arb": "defi_yield",
    "fast_path_arb": "defi_yield",
    "reflex_arc": "defi_yield",
    "executor.reflex_arc": "defi_yield",
}


def _normalise(action_type: str) -> str:
    """Normalise an action type string to its canonical registry key."""
    normalised = _ALIAS_MAP.get(action_type)
    if normalised:
        return normalised
    # Strip "executor." prefix if present
    if action_type.startswith("executor."):
        return action_type[len("executor."):]
    return action_type


class ExecutorRegistry:
    """
    Immutable-at-runtime registry of available action executors.

    Built at startup, queried at execution time.
    """

    def __init__(self) -> None:
        self._executors: dict[str, Executor] = {}
        self._logger = logger.bind(system="axon.registry")

    def register(self, executor: Executor, *, replace: bool = False) -> None:
        """
        Register an executor under its canonical action_type.

        Raises ValueError if the action_type is already registered, unless
        ``replace=True`` is passed (used by the hot-reload path).
        """
        key = executor.action_type
        if not key:
            raise ValueError(f"Executor {executor!r} has no action_type set")
        already_registered = key in self._executors
        if already_registered and not replace:
            raise ValueError(
                f"Executor for action_type {key!r} already registered — "
                f"existing: {self._executors[key]!r}, new: {executor!r}"
            )
        self._executors[key] = executor
        self._logger.debug(
            "executor_registered",
            action_type=key,
            replaced=already_registered and replace,
        )

    def get(self, action_type: str) -> Executor | None:
        """
        Look up an executor by action type, with alias normalisation.

        Returns None if no executor is registered for the given type.
        """
        canonical = _normalise(action_type)
        return self._executors.get(canonical)

    def get_strict(self, action_type: str) -> Executor:
        """
        Look up an executor; raise KeyError if not found.
        """
        executor = self.get(action_type)
        if executor is None:
            canonical = _normalise(action_type)
            raise KeyError(
                f"No executor registered for action_type {action_type!r} "
                f"(normalised: {canonical!r}). "
                f"Available: {sorted(self._executors.keys())}"
            )
        return executor

    def list_types(self) -> list[str]:
        """Return all registered canonical action type names."""
        return sorted(self._executors.keys())

    def capabilities(self) -> list[dict[str, object]]:
        """Return capability descriptors for every registered executor."""
        caps: list[dict[str, object]] = []
        for key in sorted(self._executors):
            ex = self._executors[key]
            caps.append({
                "action_type": ex.action_type,
                "description": ex.description,
                "required_autonomy": ex.required_autonomy,
                "reversible": ex.reversible,
                "max_duration_ms": ex.max_duration_ms,
                "counts_toward_budget": ex.counts_toward_budget,
                "emits_to_atune": ex.emits_to_atune,
                "rate_limit": {
                    "max_calls": ex.rate_limit.max_calls,
                    "window_seconds": ex.rate_limit.window_seconds,
                },
            })
        return caps

    def find_by_autonomy(self, max_level: int) -> list[str]:
        """Return action types executable at or below *max_level*."""
        return [
            key
            for key, ex in self._executors.items()
            if ex.required_autonomy <= max_level
        ]

    def deregister(self, action_type: str) -> bool:
        """Remove an executor (for self-healing auto-eviction)."""
        canonical = _normalise(action_type)
        if canonical in self._executors:
            del self._executors[canonical]
            self._logger.info("executor_deregistered", action_type=canonical)
            return True
        return False

    def __contains__(self, action_type: str) -> bool:
        return self.get(action_type) is not None

    def __len__(self) -> int:
        return len(self._executors)

    def __repr__(self) -> str:
        return f"<ExecutorRegistry executors={self.list_types()}>"
