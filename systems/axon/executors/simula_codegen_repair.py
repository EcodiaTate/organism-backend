"""
EcodiaOS -- Simula Code Generation Repair Executor

Repairs stalled code generation in Simula's evolution pipeline.
When Simula's proposal processing, synthesis, or verification stages
stall, this executor intervenes by:

1. Checking Simula health and identifying the stuck stage
2. Clearing blocked proposal queues
3. Resetting synthesis solver state
4. Emitting recovery signals for Thymos/Evo to learn from

Wired to the real SimulaService via dependency injection.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel
from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)
from systems.synapse.types import SynapseEvent, SynapseEventType

logger = structlog.get_logger("simula_codegen_repair")


class SimulaCodegenRepairInput(EOSBaseModel):
    """Input schema for Simula code generation stall repair."""

    stall_duration_s: float = Field(
        ..., description="Duration the generation has been stalled", ge=0.0
    )
    last_successful_generation: str | None = Field(
        default=None, description="ID of the last successfully processed proposal"
    )
    current_intent: str = Field(
        ..., description="What Simula was attempting when it stalled"
    )
    stuck_stage: str = Field(
        default="unknown",
        description="Which pipeline stage is stuck (synthesis/verification/application)",
    )


class SimulaCodegenRepairExecutor(Executor):
    """
    Repairs cognitive stalls in Simula's code generation pipeline.

    Simula processes EvolutionProposals through multiple stages:
    synthesis -> verification -> application. Any stage can stall
    due to LLM timeouts, solver deadlocks, or resource exhaustion.

    This executor performs targeted recovery based on the stuck stage.
    """

    action_type: str = "simula_codegen_repair"
    description: str = "Repair code generation stalls in Simula's evolution pipeline"
    required_autonomy: int = 3
    reversible: bool = False
    max_duration_ms: int = 15_000
    rate_limit: RateLimit = RateLimit.per_minute(3)

    def __init__(
        self,
        simula: Any = None,
        synapse: Any = None,
    ) -> None:
        self._simula = simula
        self._synapse = synapse
        self._repairs_completed: int = 0
        self._logger = logger.bind(system="axon.executor.simula_codegen_repair")

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        try:
            SimulaCodegenRepairInput(**params)
            return ValidationResult(valid=True)
        except Exception as e:
            return ValidationResult(valid=False, reason=str(e))

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        validation = await self.validate_params(params)
        if not validation.valid:
            return ExecutionResult(success=False, error=validation.reason)

        input_data = SimulaCodegenRepairInput(**params)
        t0 = time.monotonic()
        repair_log: list[dict[str, Any]] = []

        self._logger.info(
            "simula_codegen_repair_starting",
            stall_duration_s=input_data.stall_duration_s,
            stuck_stage=input_data.stuck_stage,
            current_intent=input_data.current_intent[:80],
        )

        try:
            # Phase 1: Diagnose -- check Simula health
            diag_result = await self._diagnose_stall(input_data)
            repair_log.append({"phase": "diagnosis", **diag_result})

            # Phase 2: Stage-specific recovery
            if input_data.stuck_stage == "synthesis":
                recovery = await self._recover_synthesis(input_data)
            elif input_data.stuck_stage == "verification":
                recovery = await self._recover_verification(input_data)
            elif input_data.stuck_stage == "application":
                recovery = await self._recover_application(input_data)
            else:
                recovery = await self._recover_generic(input_data)
            repair_log.append({"phase": "recovery", **recovery})

            # Phase 3: Clear queued proposals that accumulated during stall
            queue_result = await self._clear_stale_proposals(input_data)
            repair_log.append({"phase": "queue_cleanup", **queue_result})

            # Phase 4: Emit recovery event
            await self._emit_recovery(input_data, repair_log, t0)

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            self._repairs_completed += 1

            # Determine if stall was too long for automatic recovery
            if input_data.stall_duration_s > 300:
                return ExecutionResult(
                    success=True,
                    data={
                        "repair_phases": repair_log,
                        "elapsed_ms": elapsed_ms,
                        "warning": (
                            "Stall exceeded 5 minutes; recovery applied "
                            "but monitoring recommended"
                        ),
                        "stuck_stage": input_data.stuck_stage,
                    },
                    side_effects=[
                        f"Simula codegen repair completed "
                        f"({input_data.stuck_stage}, {elapsed_ms}ms) "
                        f"-- stall was {input_data.stall_duration_s:.0f}s",
                    ],
                    new_observations=[
                        f"Simula codegen stall in {input_data.stuck_stage} lasted "
                        f"{input_data.stall_duration_s:.0f}s -- may indicate deeper issue",
                    ],
                )

            return ExecutionResult(
                success=True,
                data={
                    "repair_phases": repair_log,
                    "elapsed_ms": elapsed_ms,
                    "stuck_stage": input_data.stuck_stage,
                },
                side_effects=[
                    f"Simula codegen stall repaired ({input_data.stuck_stage}, {elapsed_ms}ms)",
                ],
            )

        except Exception as e:
            self._logger.error("simula_codegen_repair_failed", error=str(e))
            return ExecutionResult(
                success=False,
                error=f"Simula codegen repair failed: {e}",
                data={"partial_repair_log": repair_log},
            )

    async def _diagnose_stall(
        self, input_data: SimulaCodegenRepairInput
    ) -> dict[str, Any]:
        """Check Simula health to understand the stall cause."""
        result: dict[str, Any] = {"success": False}

        if self._simula is None:
            result["skipped"] = True
            result["reason"] = "no_simula_service"
            return result

        try:
            if hasattr(self._simula, "health"):
                health = await self._simula.health()
                result["simula_health"] = health.get("status", "unknown")
                result["initialized"] = health.get("initialized", False)

            # Check if there's a pending proposal count
            if hasattr(self._simula, "_pending_proposals"):
                result["pending_proposals"] = len(self._simula._pending_proposals)
            if hasattr(self._simula, "_active_proposal"):
                result["has_active_proposal"] = self._simula._active_proposal is not None

            result["success"] = True
        except Exception as e:
            result["error"] = str(e)

        return result

    async def _recover_synthesis(
        self, input_data: SimulaCodegenRepairInput
    ) -> dict[str, Any]:
        """Recover from a synthesis stage stall (LLM/solver timeout)."""
        result: dict[str, Any] = {"success": False, "stage": "synthesis"}

        if self._simula is None:
            result["skipped"] = True
            return result

        try:
            # Reset the sketch solver if it exists
            if hasattr(self._simula, "_sketch_solver"):
                solver = self._simula._sketch_solver
                if hasattr(solver, "reset"):
                    solver.reset()
                    result["solver_reset"] = True

            # Reset the synthesis strategy selector
            if hasattr(self._simula, "_strategy_selector"):
                selector = self._simula._strategy_selector
                if hasattr(selector, "reset"):
                    selector.reset()
                    result["strategy_selector_reset"] = True

            # Clear the active proposal to unblock the pipeline
            if hasattr(self._simula, "_active_proposal"):
                self._simula._active_proposal = None
                result["active_proposal_cleared"] = True

            result["success"] = True
        except Exception as e:
            result["error"] = str(e)

        return result

    async def _recover_verification(
        self, input_data: SimulaCodegenRepairInput
    ) -> dict[str, Any]:
        """Recover from a verification stage stall (Z3/test timeout)."""
        result: dict[str, Any] = {"success": False, "stage": "verification"}

        if self._simula is None:
            result["skipped"] = True
            return result

        try:
            # Reset the verifier if accessible
            if hasattr(self._simula, "_verifier"):
                verifier = self._simula._verifier
                if hasattr(verifier, "cancel_pending"):
                    await verifier.cancel_pending()
                    result["pending_verifications_cancelled"] = True

            # Clear the active proposal
            if hasattr(self._simula, "_active_proposal"):
                self._simula._active_proposal = None
                result["active_proposal_cleared"] = True

            result["success"] = True
        except Exception as e:
            result["error"] = str(e)

        return result

    async def _recover_application(
        self, input_data: SimulaCodegenRepairInput
    ) -> dict[str, Any]:
        """Recover from an application stage stall (file write/hot-reload)."""
        result: dict[str, Any] = {"success": False, "stage": "application"}

        if self._simula is None:
            result["skipped"] = True
            return result

        try:
            # If there's a pending application, clear it
            if hasattr(self._simula, "_applying"):
                self._simula._applying = False
                result["application_flag_cleared"] = True

            if hasattr(self._simula, "_active_proposal"):
                self._simula._active_proposal = None
                result["active_proposal_cleared"] = True

            result["success"] = True
        except Exception as e:
            result["error"] = str(e)

        return result

    async def _recover_generic(
        self, input_data: SimulaCodegenRepairInput
    ) -> dict[str, Any]:
        """Generic recovery when the stuck stage is unknown."""
        result: dict[str, Any] = {"success": False, "stage": "generic"}

        if self._simula is None:
            result["skipped"] = True
            return result

        try:
            # Clear any active work
            if hasattr(self._simula, "_active_proposal"):
                self._simula._active_proposal = None
                result["active_proposal_cleared"] = True

            result["success"] = True
        except Exception as e:
            result["error"] = str(e)

        return result

    async def _clear_stale_proposals(
        self, input_data: SimulaCodegenRepairInput
    ) -> dict[str, Any]:
        """Clear proposals that accumulated during the stall period."""
        result: dict[str, Any] = {"success": False, "proposals_cleared": 0}

        if self._simula is None:
            result["skipped"] = True
            return result

        try:
            # If there's a proposal queue, drain stale entries
            if hasattr(self._simula, "_pending_proposals"):
                queue = self._simula._pending_proposals
                if hasattr(queue, "clear"):
                    count = len(queue) if hasattr(queue, "__len__") else 0
                    queue.clear()
                    result["proposals_cleared"] = count
                elif isinstance(queue, list):
                    result["proposals_cleared"] = len(queue)
                    queue.clear()

            result["success"] = True
        except Exception as e:
            result["error"] = str(e)

        return result

    async def _emit_recovery(
        self,
        input_data: SimulaCodegenRepairInput,
        repair_log: list[dict[str, Any]],
        start_time: float,
    ) -> None:
        """Emit REPAIR_COMPLETED for the Simula codegen recovery."""
        if self._synapse is None:
            return

        # Use the same event bus access pattern as the rest of the codebase
        event_bus = getattr(self._synapse, "_event_bus", None)
        if event_bus is None:
            event_bus = getattr(self._synapse, "event_bus", None)
        if event_bus is None or not hasattr(event_bus, "emit"):
            return

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        all_phases_ok = all(p.get("success", False) for p in repair_log)
        event = SynapseEvent(
            event_type=SynapseEventType.REPAIR_COMPLETED,
            source_system="axon.simula_codegen_repair",
            data={
                "repair_type": "simula_codegen",
                "stuck_stage": input_data.stuck_stage,
                "stall_duration_s": input_data.stall_duration_s,
                "phases_completed": len(repair_log),
                "elapsed_ms": elapsed_ms,
                "success": all_phases_ok,
                "incident_class": "cognitive_stall",
                "fix_type": f"simula_{input_data.stuck_stage}_recovery",
                "root_cause": input_data.current_intent[:200],
            },
        )
        try:
            await event_bus.emit(event)
        except Exception as e:
            self._logger.debug("recovery_emit_failed", error=str(e))
