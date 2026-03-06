"""
EcodiaOS — Atune Logging Utilities

Provides a pure, zero-dependency debug logging executor for structured output.
"""

from __future__ import annotations

import json
from typing import Any

from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)


class DebugLogExecutor(Executor):
    """
    Emit a structured debug log entry to stdout.

    Pure executor with no side effects beyond stdout.
    Useful for development and tracing without adding logging framework complexity.
    """

    action_type = "debug_log"
    description = "Structured debug logging to stdout"
    reversible = False
    max_duration_ms = 100  # Very fast operation
    rate_limit = RateLimit.per_minute(60)  # Allow frequent debug logging

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Emit a structured JSON debug log to stdout.

        Args:
            params: Dictionary with 'label' and 'data' keys
            context: Execution context (not used in this pure executor)

        Returns:
            ExecutionResult indicating success
        """
        label = params.get("label", "DEBUG")
        data = params.get("data", {})

        print(f"[{label}] {json.dumps(data, indent=2, default=str)}")

        return ExecutionResult(
            success=True,
            output={"label": label, "logged": True},
        )

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        """
        Validate input parameters for debug logging.

        Args:
            params: Input parameters to validate

        Returns:
            ValidationResult indicating parameter validity
        """
        if not isinstance(params.get("label", ""), str):
            return ValidationResult(
                valid=False,
                error="'label' must be a string",
            )

        if not isinstance(params.get("data", {}), dict):
            return ValidationResult(
                valid=False,
                error="'data' must be a dictionary",
            )

        return ValidationResult(valid=True)
