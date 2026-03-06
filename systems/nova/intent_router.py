"""
EcodiaOS — Nova Intent Router

Once Equor approves an Intent, Nova routes it to the appropriate executor.

Current routing:
  - Expression intents → Voxis (via VoxisService.express())
  - Action intents     → Axon (via AxonService.execute())
  - Hybrid intents     → Axon first, then express outcome via Voxis
  - Internal intents   → no external delivery (memory/goal updates only)

Routing classification is based on the executor field of the first action step:
  executor.express   → Voxis
  executor.observe   → internal (no delivery)
  executor.wait      → internal (do nothing)
  executor.store     → internal (memory write)
  executor.*         → Axon

Intent routing must complete in ≤20ms (per spec) — the dispatch is async
but the routing classification is synchronous and fast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from primitives.affect import AffectState
    from primitives.constitutional import ConstitutionalCheck
    from primitives.intent import Intent
    from systems.axon.service import AxonService
    from systems.voxis.service import VoxisService
    from systems.voxis.types import ExpressionTrigger

logger = structlog.get_logger()

# Action executor prefixes that route to Voxis
_VOXIS_EXECUTORS = {"executor.express", "express", "executor.request_info", "request_info"}
# Action executor prefixes that are internal (no delivery needed)
_INTERNAL_EXECUTORS = {
    "executor.observe", "observe", "executor.wait", "wait", "executor.store", "store",
}


class IntentRouter:
    """
    Routes approved intents to their appropriate executor system.

    Supports Voxis (expression) and Axon (action) routing.
    Internal intents (observe, wait, store) require no external delivery.
    Hybrid intents execute in Axon first, then express the outcome via Voxis.
    """

    def __init__(
        self,
        voxis: VoxisService,
        axon: AxonService | None = None,
    ) -> None:
        self._voxis = voxis
        self._axon = axon
        self._logger = logger.bind(system="nova.intent_router")
        self._routed_to_voxis: int = 0
        self._routed_to_axon: int = 0
        self._routed_internal: int = 0

    def set_axon(self, axon: AxonService) -> None:
        """Wire Axon after both services are initialised."""
        self._axon = axon
        self._logger.info("axon_wired", system="nova.intent_router")

    async def route(
        self,
        intent: Intent,
        affect: AffectState,
        conversation_id: str | None = None,
        equor_check: ConstitutionalCheck | None = None,
    ) -> str:
        """
        Route an approved intent to its executor.

        Returns the route taken: "voxis" | "axon" | "internal" | "hybrid"

        Args:
            equor_check: The Equor verdict that approved this intent.
                         Required for Axon routing (passed into ExecutionRequest).
        """
        route = _classify_route(intent)

        if route == "voxis":
            await self._route_to_voxis(intent, affect, conversation_id)
            self._routed_to_voxis += 1
        elif route == "axon":
            await self._route_to_axon(intent, equor_check)
            self._routed_to_axon += 1
        elif route == "hybrid":
            # Execute in Axon first, then express the result via Voxis
            await self._route_to_axon(intent, equor_check)
            self._routed_to_axon += 1
            await self._route_to_voxis(intent, affect, conversation_id)
            self._routed_to_voxis += 1
        else:
            # Internal / observe / wait — no delivery needed
            self._routed_internal += 1
            self._logger.debug("intent_internal_route", intent_id=intent.id, route=route)

        self._logger.info(
            "intent_routed",
            intent_id=intent.id,
            route=route,
            goal=intent.goal.description[:60],
        )
        return route

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "routed_to_voxis": self._routed_to_voxis,
            "routed_to_axon": self._routed_to_axon,
            "routed_internal": self._routed_internal,
        }

    # ─── Private ──────────────────────────────────────────────────

    async def _route_to_voxis(
        self,
        intent: Intent,
        affect: AffectState,
        conversation_id: str | None,
    ) -> None:
        """
        Extract the expression content from the intent and deliver via Voxis.
        """

        # Extract the expression content from the first express action step
        content = intent.goal.description  # Default to goal description
        for step in intent.plan.steps:
            params = step.parameters
            if "description" in params:
                content = str(params["description"])
                break

        # Determine the appropriate Voxis trigger from the intent context
        trigger = _classify_voxis_trigger(intent)

        try:
            await self._voxis.express(
                content=content,
                trigger=trigger,
                conversation_id=conversation_id,
                affect=affect,
                urgency=intent.priority,
            )
        except Exception as exc:
            self._logger.error(
                "voxis_routing_failed",
                intent_id=intent.id,
                error=str(exc),
            )

    async def _route_to_axon(
        self,
        intent: Intent,
        equor_check: ConstitutionalCheck | None,
    ) -> None:
        """
        Route an action intent to Axon for execution.

        Builds an ExecutionRequest and calls AxonService.execute().
        The outcome is delivered to Nova by Axon's pipeline directly —
        this method fire-and-forgets (does not await the full execution).
        """
        if self._axon is None:
            self._logger.warning(
                "axon_not_wired",
                intent_id=intent.id,
                goal=intent.goal.description[:80],
                message="Axon not wired — intent cannot be executed",
            )
            return

        from primitives.common import Verdict
        from primitives.constitutional import ConstitutionalCheck
        from systems.axon.types import ExecutionRequest

        # Security default: BLOCKED if no equor_check provided.
        # Intents should always carry a real Equor check from the deliberation
        # engine.  The BLOCKED default prevents bypass if route() is called
        # without going through the full deliberation pipeline.
        check = equor_check or ConstitutionalCheck(
            intent_id=intent.id,
            verdict=Verdict.BLOCKED,
            reasoning="No Equor check provided — blocked by security default.",
        )

        request = ExecutionRequest(
            intent=intent,
            equor_check=check,
            timeout_ms=intent.budget.compute_ms,
        )

        try:
            await self._axon.execute(request)
        except Exception as exc:
            self._logger.error(
                "axon_routing_failed",
                intent_id=intent.id,
                error=str(exc),
            )


# ─── Classification Helpers ───────────────────────────────────────


def _classify_route(intent: Intent) -> str:
    """
    Determine the routing destination from the intent's action steps.
    Returns "voxis" | "axon" | "internal" | "hybrid"
    """
    if not intent.plan.steps:
        return "internal"

    executors = {step.executor for step in intent.plan.steps}

    has_voxis = any(e in _VOXIS_EXECUTORS for e in executors)
    has_internal = all(e in _INTERNAL_EXECUTORS for e in executors)
    has_axon = any(
        e not in _VOXIS_EXECUTORS and e not in _INTERNAL_EXECUTORS
        for e in executors
    )

    if has_internal:
        return "internal"
    if has_voxis and has_axon:
        return "hybrid"
    if has_voxis:
        return "voxis"
    if has_axon:
        return "axon"
    return "internal"


def _classify_voxis_trigger(intent: Intent) -> ExpressionTrigger:
    """
    Map an intent's goal and plan to the most appropriate Voxis trigger.
    """
    from systems.voxis.types import ExpressionTrigger

    goal_desc = intent.goal.description.lower()
    executors = {step.executor for step in intent.plan.steps}
    params_combined = " ".join(
        str(v) for step in intent.plan.steps
        for v in step.parameters.values()
    ).lower()

    # Care / distress response
    if "distress" in goal_desc or "support" in goal_desc or "care" in goal_desc:
        return ExpressionTrigger.NOVA_RESPOND

    # Informing the user
    if "inform" in goal_desc or "inform" in params_combined:
        return ExpressionTrigger.NOVA_INFORM

    # Mediation or conflict resolution
    if "mediat" in goal_desc or "conflict" in goal_desc:
        return ExpressionTrigger.NOVA_MEDIATE

    # Celebration
    if "celebrat" in goal_desc or "success" in goal_desc:
        return ExpressionTrigger.NOVA_CELEBRATE

    # Warning / alert
    if "warn" in goal_desc or "alert" in goal_desc or "risk" in goal_desc:
        return ExpressionTrigger.NOVA_RESPOND

    # Request for clarification
    if "clarif" in goal_desc or "request" in goal_desc or "request_info" in executors:
        return ExpressionTrigger.NOVA_REQUEST

    # Default: standard response
    return ExpressionTrigger.NOVA_RESPOND
