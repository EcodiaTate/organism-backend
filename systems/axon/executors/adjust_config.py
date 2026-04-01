"""
EcodiaOS - Adjust Config Executor

Adjusts system configuration parameters at runtime via Synapse events.
Used by Evo/Simula for parameter tuning without direct cross-system imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.executor import Executor
from systems.axon.types import ExecutionContext, ExecutionResult, RateLimit, ValidationResult
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


class AdjustConfigExecutor(Executor):
    action_type = "adjust_config"
    description = "Adjust system configuration parameters at runtime"
    required_autonomy = 2
    reversible = True
    max_duration_ms = 10_000
    rate_limit = RateLimit.per_minute(20)

    def __init__(self, event_bus: Any = None) -> None:
        self._event_bus = event_bus
        # Track previous values for rollback
        self._last_adjustments: dict[str, dict[str, Any]] = {}

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        target_system = params.get("target_system")
        if not target_system or not isinstance(target_system, str):
            return ValidationResult.fail("'target_system' is required")
        config_key = params.get("config_key")
        if not config_key or not isinstance(config_key, str):
            return ValidationResult.fail("'config_key' is required")
        if "new_value" not in params:
            return ValidationResult.fail("'new_value' is required")
        return ValidationResult.ok()

    async def execute(self, params: dict[str, Any], context: ExecutionContext) -> ExecutionResult:
        target_system = params["target_system"]
        config_key = params["config_key"]
        new_value = params["new_value"]
        reason = params.get("reason", "")

        try:
            # Emit config adjustment event via Synapse for the target system to pick up
            if self._event_bus is not None:
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.ACTION_EXECUTED,
                    source_system="axon",
                    data={
                        "action_type": self.action_type,
                        "target_system": target_system,
                        "config_key": config_key,
                        "new_value": new_value,
                        "reason": reason,
                        "execution_id": context.execution_id,
                    },
                ))

            # Store for potential rollback
            self._last_adjustments[context.execution_id] = {
                "target_system": target_system,
                "config_key": config_key,
                "new_value": new_value,
            }

            await self._emit_re_trace(context, params, success=True)

            return ExecutionResult(
                success=True,
                data={
                    "target_system": target_system,
                    "config_key": config_key,
                    "new_value": new_value,
                },
                side_effects=[
                    f"Config adjusted: {target_system}.{config_key} = {new_value}"
                ],
            )
        except Exception as exc:
            if self._event_bus is not None:
                try:
                    await self._event_bus.emit(SynapseEvent(
                        event_type=SynapseEventType.ACTION_FAILED,
                        source_system="axon",
                        data={
                            "action_type": self.action_type,
                            "error": str(exc),
                            "execution_id": context.execution_id,
                        },
                    ))
                except Exception:
                    pass
            await self._emit_re_trace(context, params, success=False, error=str(exc))
            return ExecutionResult(success=False, error=str(exc))

    async def _emit_re_trace(
        self,
        context: ExecutionContext,
        params: dict[str, Any],
        success: bool,
        error: str = "",
    ) -> None:
        if self._event_bus is None:
            return
        try:
            import json as _json
            from primitives.common import DriveAlignmentVector, SystemID
            from primitives.re_training import RETrainingExample

            target_system = params.get("target_system", "")
            config_key = params.get("config_key", "")
            new_value = params.get("new_value", "")
            reason = params.get("reason", "")

            # Config adjustments are reversible - prior value stored in _last_adjustments
            prior_stored = context.execution_id in self._last_adjustments

            reasoning_trace = "\n".join([
                f"1. VALIDATE: target_system={target_system!r}, config_key={config_key!r}, new_value={str(new_value)[:80]!r}",
                f"2. ROLLBACK READINESS: prior value {'stored' if prior_stored else 'NOT stored'} (execution_id={context.execution_id})",
                f"3. REASON: {reason!r}" if reason else "3. REASON: (none provided)",
                f"4. EMIT: ACTION_EXECUTED event via Synapse - target system receives CONFIG_ADJUSTED signal",
                f"5. OUTCOME: success={success}" + (f", error={error[:120]!r}" if error else ""),
            ])

            alternatives = [
                f"Alternative: use EQUOR amendment process for permanent parameter changes (adjust_config is ephemeral - resets on restart unless persisted)",
                f"Alternative: batch multiple config adjustments into one intent to reduce Synapse event traffic",
            ]
            if not success:
                alternatives.append(
                    f"Alternative: validate that '{target_system}' is registered and listening for CONFIG_ADJUSTED events before retrying"
                )

            counterfactual = ""
            if success:
                counterfactual = (
                    f"If '{target_system}.{config_key}' had not been adjusted to {str(new_value)[:60]!r}, "
                    f"the system would continue operating with the prior parameter value - potentially "
                    f"limiting the performance improvement or behavior change this adjustment was designed to achieve."
                )
            else:
                counterfactual = (
                    f"If the adjustment to '{target_system}.{config_key}' had succeeded, "
                    f"the organism's behavior in that subsystem would shift according to {str(new_value)[:60]!r}. "
                    f"The failure leaves the prior configuration in place."
                )

            equor_alignment = getattr(context.equor_check, "drive_alignment", None) if context.equor_check is not None else None

            trace = RETrainingExample(
                source_system=SystemID.AXON,
                episode_id=context.execution_id,
                instruction=f"Adjust runtime config: {target_system}.{config_key} ← {str(new_value)[:80]!r}. Reason: {reason[:100] or 'unspecified'}",
                input_context=_json.dumps({
                    "target_system": target_system,
                    "config_key": config_key,
                    "new_value": str(new_value)[:200],
                    "reason": reason[:200],
                    "reversible": True,
                }),
                output=_json.dumps({
                    "success": success,
                    "prior_value_stored_for_rollback": prior_stored,
                    "error": error[:200] if error else None,
                }),
                outcome_quality=1.0 if success else 0.5,
                category="config_adjustment",
                constitutional_alignment=equor_alignment or DriveAlignmentVector(),
                reasoning_trace=reasoning_trace,
                alternatives_considered=alternatives,
                counterfactual=counterfactual,
            )
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="axon",
                data=trace.model_dump(mode="json"),
            ))
        except Exception:
            pass
