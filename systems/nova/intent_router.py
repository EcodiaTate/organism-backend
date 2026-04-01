"""
EcodiaOS - Nova Intent Router

Once Equor approves an Intent, Nova routes it to the appropriate executor
via Synapse events - no direct cross-system references are held.

Current routing:
  - Expression intents → NOVA_EXPRESSION_REQUEST event → Voxis subscribes
  - Action intents     → AXON_EXECUTION_REQUEST event  → Axon subscribes
  - Hybrid intents     → Axon first, then Voxis (both via events)
  - Internal intents   → no external delivery (memory/goal updates only)

Routing classification is based on the executor field of the first action step:
  executor.express   → Voxis
  executor.observe   → internal (no delivery)
  executor.wait      → internal (do nothing)
  executor.store     → internal (memory write)
  executor.*         → Axon

Intent routing must complete in ≤20ms (per spec) - the dispatch is async
but the routing classification is synchronous and fast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from primitives.affect import AffectState
    from primitives.constitutional import ConstitutionalCheck
    from primitives.intent import Intent
    from systems.synapse.event_bus import EventBus
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
    Routes approved intents to their appropriate executor system via Synapse events.

    Voxis subscribes to NOVA_EXPRESSION_REQUEST.
    Axon subscribes to AXON_EXECUTION_REQUEST.
    No live service references are held - all routing is bus-mediated.
    """

    def __init__(self, event_bus: EventBus) -> None:
        self._bus = event_bus
        self._logger = logger.bind(system="nova.intent_router")
        self._routed_to_voxis: int = 0
        self._routed_to_axon: int = 0
        self._routed_internal: int = 0

    async def route(
        self,
        intent: Intent,
        affect: AffectState,
        conversation_id: str | None = None,
        equor_check: ConstitutionalCheck | None = None,
    ) -> str:
        """
        Route an approved intent to its executor via Synapse events.

        Returns the route taken: "voxis" | "axon" | "internal" | "hybrid"

        Args:
            equor_check: The Equor verdict that approved this intent.
                         Required for Axon routing (embedded in the event payload).
        """
        route = _classify_route(intent)

        if route == "voxis":
            await self._route_to_voxis(intent, affect, conversation_id)
            self._routed_to_voxis += 1
        elif route == "axon":
            await self._route_to_axon(intent, equor_check)
            self._routed_to_axon += 1
        elif route == "hybrid":
            # Axon first, then Voxis - both via bus events
            await self._route_to_axon(intent, equor_check)
            self._routed_to_axon += 1
            await self._route_to_voxis(intent, affect, conversation_id)
            self._routed_to_voxis += 1
        else:
            # Internal / observe / wait - no delivery needed
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
        """Emit NOVA_EXPRESSION_REQUEST - Voxis subscribes and handles expression."""
        content = intent.goal.description
        for step in intent.plan.steps:
            if "description" in step.parameters:
                content = str(step.parameters["description"])
                break

        trigger = _classify_voxis_trigger(intent)

        try:
            await self._bus.emit(SynapseEvent(
                event_type=SynapseEventType.NOVA_EXPRESSION_REQUEST,
                data={
                    "intent_id": intent.id,
                    "content": content,
                    "trigger": trigger.value if hasattr(trigger, "value") else str(trigger),
                    "conversation_id": conversation_id,
                    "affect": affect.model_dump() if affect is not None else None,
                    "urgency": intent.priority,
                },
            ))
        except Exception as exc:
            self._logger.error(
                "voxis_event_emit_failed",
                intent_id=intent.id,
                error=str(exc),
            )

    async def _route_to_axon(
        self,
        intent: Intent,
        equor_check: ConstitutionalCheck | None,
    ) -> None:
        """
        Emit AXON_EXECUTION_REQUEST - Axon subscribes and executes.

        Security default: if no equor_check is provided, embeds a BLOCKED
        verdict so Axon's Stage 0 gate rejects the request.
        """
        from primitives.common import Verdict
        from primitives.constitutional import ConstitutionalCheck

        check = equor_check or ConstitutionalCheck(
            intent_id=intent.id,
            verdict=Verdict.BLOCKED,
            reasoning="No Equor check provided - blocked by security default.",
        )

        try:
            await self._bus.emit(SynapseEvent(
                event_type=SynapseEventType.AXON_EXECUTION_REQUEST,
                data={
                    "intent": intent.model_dump(),
                    "equor_check": check.model_dump(),
                    "timeout_ms": intent.budget.compute_ms,
                },
            ))
        except Exception as exc:
            self._logger.error(
                "axon_event_emit_failed",
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
