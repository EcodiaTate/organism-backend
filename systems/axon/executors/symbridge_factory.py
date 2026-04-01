"""
EcodiaOS - Symbridge Factory Executor

Dispatches code change proposals, capability requests, and repair tasks
to the EcodiaOS Factory via the Symbridge (Redis Streams primary,
shared Neo4j secondary, HTTP tertiary).

The Factory is EcodiaOS's human-facing cortex — the organism's hands
for executing code changes, deployments, and creating new capabilities.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.executor import Executor
from systems.axon.types import ExecutionContext, ExecutionResult, RateLimit, ValidationResult
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


class SymbridgeFactoryExecutor(Executor):
    action_type = "symbridge_factory_dispatch"
    description = (
        "Dispatch work to the EcodiaOS Factory for code execution, "
        "deployment, or capability creation via the Symbridge"
    )
    required_autonomy = 3  # TRUSTED — code changes are significant
    reversible = True  # Factory supports git revert
    max_duration_ms = 600_000  # 10 min — Factory work is async
    rate_limit = RateLimit.per_hour(10)

    def __init__(self, event_bus: Any = None, redis_client: Any = None) -> None:
        self._event_bus = event_bus
        self._redis = redis_client
        self._ecodiaos_api_url: str = ""

    def set_event_bus(self, bus: Any) -> None:
        self._event_bus = bus

    def set_redis(self, redis_client: Any) -> None:
        self._redis = redis_client

    def set_ecodiaos_url(self, url: str) -> None:
        self._ecodiaos_api_url = url

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        dispatch_type = params.get("dispatch_type")
        if dispatch_type not in (
            "proposal", "thymos_incident", "capability_request", "scheduled",
        ):
            return ValidationResult.fail(
                f"Invalid dispatch_type: {dispatch_type}. "
                "Must be one of: proposal, thymos_incident, capability_request, scheduled"
            )

        description = params.get("description")
        if not description or not isinstance(description, str):
            return ValidationResult.fail("'description' is required and must be a string")

        return ValidationResult.ok()

    async def execute(
        self, params: dict[str, Any], context: ExecutionContext,
    ) -> ExecutionResult:
        dispatch_type = params["dispatch_type"]
        description = params["description"]

        message = {
            "type": dispatch_type,
            "payload": {
                "description": description,
                "category": params.get("category", "unknown"),
                "priority": params.get("priority", "medium"),
                "codebase_name": params.get("codebase_name"),
                "workspace_root": params.get("workspace_root"),
                "target_repository_url": params.get("target_repository_url"),
                "change_spec": params.get("change_spec"),
                "expected_benefit": params.get("expected_benefit"),
                "risk_assessment": params.get("risk_assessment"),
                "evidence": params.get("evidence", []),
                "proposed_implementation": params.get("proposed_implementation"),
                "id": params.get("proposal_id") or context.execution_id,
                "severity": params.get("severity"),
                "error_message": params.get("error_message"),
                "affected_system": params.get("affected_system"),
                "stack_trace": params.get("stack_trace"),
            },
            "source": "organism",
            "correlationId": context.execution_id,
        }

        delivered = False
        delivery_details: dict[str, bool] = {"redis": False, "neo4j": False, "http": False}

        # Layer 1: Redis Streams (primary)
        if self._redis is not None:
            try:
                await self._redis.xadd(
                    "symbridge:organism_to_ecodiaos",
                    {"data": json.dumps(message)},
                )
                delivery_details["redis"] = True
                delivered = True
            except Exception as exc:
                logger.debug(
                    "Symbridge Redis delivery failed",
                    error=str(exc),
                    dispatch_type=dispatch_type,
                )

        # Layer 2: HTTP REST (tertiary — used when Redis unavailable)
        if self._ecodiaos_api_url:
            try:
                import httpx

                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(
                        f"{self._ecodiaos_api_url}/api/symbridge/inbound",
                        json=message,
                    )
                    if resp.status_code < 400:
                        delivery_details["http"] = True
                        delivered = True
            except Exception as exc:
                logger.debug(
                    "Symbridge HTTP delivery failed",
                    error=str(exc),
                    dispatch_type=dispatch_type,
                )

        if not delivered:
            logger.error(
                "Symbridge: message failed ALL delivery layers",
                dispatch_type=dispatch_type,
                execution_id=context.execution_id,
            )
            return ExecutionResult(
                success=False,
                error="Failed to deliver message to EcodiaOS Factory via any transport layer",
                data={"delivery_details": delivery_details},
            )

        # Emit event on Synapse bus
        if self._event_bus is not None:
            try:
                await self._event_bus.emit(SynapseEvent(
                    type=SynapseEventType.FACTORY_PROPOSAL_SENT,
                    payload={
                        "proposal_id": context.execution_id,
                        "description": description[:200],
                        "dispatch_type": dispatch_type,
                        "codebase": params.get("codebase_name", "unknown"),
                        "priority": params.get("priority", "medium"),
                    },
                ))
            except Exception:
                pass  # Non-blocking

        logger.info(
            "Symbridge dispatch successful",
            dispatch_type=dispatch_type,
            execution_id=context.execution_id,
            delivery=delivery_details,
        )

        return ExecutionResult(
            success=True,
            data={
                "dispatch_type": dispatch_type,
                "correlation_id": context.execution_id,
                "delivery_details": delivery_details,
                "message": f"Dispatched {dispatch_type} to EcodiaOS Factory",
            },
        )

    async def rollback(
        self, execution_id: str, context: ExecutionContext,
    ) -> ExecutionResult:
        """Request Factory to revert changes via git revert."""
        if not self._ecodiaos_api_url:
            return ExecutionResult(
                success=False,
                error="No EcodiaOS API URL configured for rollback",
            )

        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self._ecodiaos_api_url}/api/symbridge/inbound",
                    json={
                        "type": "rollback_request",
                        "payload": {"execution_id": execution_id},
                        "source": "organism",
                        "correlationId": execution_id,
                    },
                )
                if resp.status_code < 400:
                    return ExecutionResult(success=True, data={"rollback": "requested"})
        except Exception as exc:
            logger.error("Symbridge rollback failed", error=str(exc))

        return ExecutionResult(success=False, error="Rollback request failed")
