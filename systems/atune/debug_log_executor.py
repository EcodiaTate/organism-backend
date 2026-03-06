"""
EcodiaOS — Atune DebugLogExecutor

A pure, side-effect-free executor that emits structured JSON debug output to stdout.
Designed for tracing and understanding system behavior without introducing
complex logging frameworks or creating persistent log artifacts.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import structlog

from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)

logger = structlog.get_logger("atune.debug_log_executor")

class DebugLogExecutor(Executor):
    """
    A debug logging executor that emits structured JSON to stdout.

    Characteristics:
    - Pure: No side effects beyond stdout
    - Stateless: No internal state maintained
    - Safe: Minimal computational overhead
    - Transparent: Full context preservation
    """

    action_type: str = "debug_log"
    description: str = "Structured debug logging to stdout for Atune system"

    # Safety parameters
    required_autonomy: int = 1
    reversible: bool = True
    max_duration_ms: int = 100
    rate_limit: RateLimit = RateLimit.per_minute(60)

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Emit a structured JSON debug log to stdout.

        Args:
            params: Dictionary of parameters to log
            context: Execution context with additional metadata

        Returns:
            ExecutionResult indicating successful logging
        """
        debug_payload = {
            "timestamp": context.timestamp.isoformat(),
            "intent_id": str(context.intent_id),
            "system": "atune",
            "executor": self.action_type,
            "params": params,
            "context": {
                k: str(v) for k, v in context.model_dump().items()
                if v is not None
            }
        }

        try:
            json_output = json.dumps(debug_payload, indent=2)
            print(json_output, file=sys.stdout, flush=True)

            return ExecutionResult(
                success=True,
                output={"debug_log": "Emitted to stdout"},
                new_observations={}
            )
        except Exception as e:
            logger.error("Debug log emission failed", error=str(e))
            return ExecutionResult(
                success=False,
                error=f"Debug log emission error: {e}",
                output={}
            )

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        """
        Validate input parameters for the debug log executor.

        Always returns a successful validation since this executor
        is designed to be flexible and log any input.

        Args:
            params: Input parameters to validate

        Returns:
            ValidationResult indicating parameter acceptance
        """
        return ValidationResult(
            valid=True,
            reason="All parameters accepted for debug logging"
        )
