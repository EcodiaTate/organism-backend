"""
EcodiaOS - Remote Compute Executor (SACM Bridge)

Axon executor that bridges action execution to the SACM pipeline.
Receives action_type="remote_compute" intents from Nova and dispatches
them to SACMClient.submit_and_await() for remote execution.

Owned by Axon (it is an Axon Executor). SACMClient is injected at
registration time - no runtime SACM service coupling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)
from systems.sacm.workload import (
    OffloadClass,
    ResourceEnvelope,
    WorkloadDescriptor,
    WorkloadPriority,
)

if TYPE_CHECKING:
    from systems.sacm.service import SACMClient

logger = structlog.get_logger()


# ─── RemoteComputeExecutor (Axon bridge) ────────────────────────


class RemoteComputeExecutor(Executor):
    """
    Axon Executor that bridges action execution to the SACM pipeline.

    When Nova formulates an intent with action_type="remote_compute",
    Axon routes it here.  The executor:
      1. Validates the params contain a valid workload specification.
      2. Builds a WorkloadDescriptor from the params.
      3. Calls SACMClient.submit_and_await() to run it remotely.
      4. Returns the result as an Axon ExecutionResult.

    Expected params:
      - image (str):              Container image URI
      - entrypoint (str):         Override entrypoint (optional)
      - items (list[str]):        Base64-encoded input items
      - offload_class (str):      One of OffloadClass values
      - priority (str|int):       WorkloadPriority name or value
      - resources (dict):         ResourceEnvelope fields (optional)
      - max_latency_s (float):    Latency constraint (optional)
      - max_cost_usd (float):     Cost constraint (optional)
      - timeout_s (float):        Execution timeout (optional, default 600)
    """

    action_type: str = "remote_compute"
    description: str = "Execute a workload on a remote compute substrate via SACM"
    required_autonomy: int = 2  # At least PARTNER level - remote execution is visible
    reversible: bool = False
    max_duration_ms: int = 600_000  # 10 minutes
    rate_limit: RateLimit = RateLimit.per_minute(10)

    def __init__(self, sacm_client: SACMClient) -> None:
        self._client = sacm_client
        self._log = logger.bind(component="axon.remote_compute_executor")

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        """
        Validate that params describe a valid remote compute workload.

        Required: 'image' (non-empty string).
        Optional but validated: 'offload_class', 'priority', 'resources'.
        """
        image = params.get("image", "")
        if not isinstance(image, str) or not image.strip():
            return ValidationResult.fail(
                "Missing or empty 'image' parameter",
                image="Container image URI is required",
            )

        # Validate offload_class if provided
        offload_class = params.get("offload_class")
        if offload_class is not None:
            valid_classes = {c.value for c in OffloadClass}
            if offload_class not in valid_classes:
                return ValidationResult.fail(
                    f"Invalid offload_class: {offload_class}",
                    offload_class=f"Must be one of {sorted(valid_classes)}",
                )

        # Validate priority if provided
        priority = params.get("priority")
        if priority is not None:
            valid_priorities = {p.name.lower() for p in WorkloadPriority} | {str(p.value) for p in WorkloadPriority}
            if str(priority).lower() not in valid_priorities:
                return ValidationResult.fail(
                    f"Invalid priority: {priority}",
                    priority=f"Must be one of {sorted(p.name for p in WorkloadPriority)} or 0-4",
                )

        # Validate resources if provided
        resources = params.get("resources")
        if resources is not None:
            if not isinstance(resources, dict):
                return ValidationResult.fail(
                    "Invalid resources: must be a dict",
                    resources="Expected dict with ResourceEnvelope fields",
                )
            for key, val in resources.items():
                if isinstance(val, (int, float)) and val < 0:
                    return ValidationResult.fail(
                        f"Negative resource value: {key}={val}",
                        **{key: "Resource values must be >= 0"},
                    )

        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Build a WorkloadDescriptor from params and execute via SACM.

        Returns ExecutionResult with data containing the remote output
        metadata, or an error if execution/verification fails.
        """
        try:
            workload = self._build_workload(params, context)
            timeout_s = float(params.get("timeout_s", 600.0))

            self._log.info(
                "remote_compute_start",
                execution_id=context.execution_id,
                workload_id=workload.workload_id,
                image=workload.image,
                offload_class=workload.offload_class.value,
                priority=workload.priority.name,
            )

            result = await self._client.submit_and_await(workload, timeout_s=timeout_s)

            if result.accepted:
                return ExecutionResult(
                    success=True,
                    data={
                        "workload_id": workload.workload_id,
                        "provider_id": result.provider_id,
                        "accepted": True,
                        "result_count": len(result.results),
                        "duration_ms": round(result.total_duration_ms, 1),
                        "batch_size": result.batch_size,
                    },
                    side_effects=[
                        f"Executed workload {workload.workload_id} on provider {result.provider_id}"
                    ],
                )
            else:
                return ExecutionResult(
                    success=False,
                    data={
                        "workload_id": workload.workload_id,
                        "provider_id": result.provider_id,
                        "accepted": False,
                        "phase": result.phase.value,
                    },
                    error=result.error or "Remote execution failed verification",
                )

        except ValueError as exc:
            return ExecutionResult(
                success=False,
                error=f"Placement failed: {exc}",
            )
        except TimeoutError:
            return ExecutionResult(
                success=False,
                error="Remote execution timed out",
            )
        except Exception as exc:
            self._log.error(
                "remote_compute_error",
                error=str(exc),
                exc_info=True,
            )
            return ExecutionResult(
                success=False,
                error=f"Remote compute error: {exc}",
            )

    def _build_workload(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> WorkloadDescriptor:
        """Construct a WorkloadDescriptor from Axon execution params."""
        import base64

        # Parse offload class
        offload_class_str = params.get("offload_class", OffloadClass.GENERAL.value)
        offload_class = OffloadClass(offload_class_str)

        # Parse priority
        priority_raw = params.get("priority", WorkloadPriority.NORMAL.value)
        if isinstance(priority_raw, str):
            priority = WorkloadPriority[priority_raw.upper()]
        else:
            priority = WorkloadPriority(int(priority_raw))

        # Parse resource envelope
        resources_dict = params.get("resources", {})
        resources = ResourceEnvelope(**resources_dict) if resources_dict else ResourceEnvelope()

        # Parse items (base64-encoded strings → bytes)
        raw_items = params.get("items", [])
        items: list[bytes] = []
        for item in raw_items:
            if isinstance(item, str):
                items.append(base64.b64decode(item))
            elif isinstance(item, bytes):
                items.append(item)

        return WorkloadDescriptor(
            tenant_id=context.instance_id,
            name=params.get("name", f"axon-{context.execution_id}"),
            image=params["image"],
            entrypoint=params.get("entrypoint", ""),
            env_vars=params.get("env_vars", {}),
            offload_class=offload_class,
            priority=priority,
            resources=resources,
            estimated_duration_s=float(params.get("estimated_duration_s", 300.0)),
            max_latency_s=float(params.get("max_latency_s", 0.0)),
            max_cost_usd=float(params.get("max_cost_usd", 0.0)),
            require_encryption=bool(params.get("require_encryption", True)),
            require_verification=bool(params.get("require_verification", True)),
            items=items,
            metadata={
                "source": "axon",
                "execution_id": context.execution_id,
                "intent_id": context.intent.id if hasattr(context.intent, "id") else "",
            },
        )
