"""
Unit tests for Axon ExecutorRegistry.

Tests registration, lookup, alias normalisation, and error handling.
"""

from __future__ import annotations

import pytest

from systems.axon.executor import Executor
from systems.axon.registry import ExecutorRegistry, _normalise
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)

# ─── Fixtures ─────────────────────────────────────────────────────


class _ConcreteExecutor(Executor):
    """Minimal concrete executor for testing."""

    action_type = "test_action"
    description = "Test executor"
    required_autonomy = 1
    reversible = False
    max_duration_ms = 1000
    rate_limit = RateLimit.unlimited()

    async def validate_params(self, params: dict) -> ValidationResult:
        return ValidationResult.ok()

    async def execute(self, params: dict, context: ExecutionContext) -> ExecutionResult:
        return ExecutionResult(success=True, data={"echo": params})


class _AnotherExecutor(_ConcreteExecutor):
    action_type = "another_action"


# ─── Tests: Registration ──────────────────────────────────────────


def test_register_and_get():
    registry = ExecutorRegistry()
    executor = _ConcreteExecutor()
    registry.register(executor)

    result = registry.get("test_action")
    assert result is executor


def test_register_duplicate_raises():
    registry = ExecutorRegistry()
    registry.register(_ConcreteExecutor())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_ConcreteExecutor())


def test_register_no_action_type_raises():
    class _NoType(Executor):
        action_type = ""
        description = ""

        async def validate_params(self, params):
            return ValidationResult.ok()

        async def execute(self, params, context):
            return ExecutionResult(success=True)

    registry = ExecutorRegistry()
    with pytest.raises(ValueError, match="no action_type"):
        registry.register(_NoType())


def test_list_types():
    registry = ExecutorRegistry()
    registry.register(_ConcreteExecutor())
    registry.register(_AnotherExecutor())
    types = registry.list_types()
    assert "test_action" in types
    assert "another_action" in types
    assert types == sorted(types)  # Should be sorted


def test_len():
    registry = ExecutorRegistry()
    assert len(registry) == 0
    registry.register(_ConcreteExecutor())
    assert len(registry) == 1


def test_contains():
    registry = ExecutorRegistry()
    registry.register(_ConcreteExecutor())
    assert "test_action" in registry
    assert "nonexistent" not in registry


# ─── Tests: Alias Normalisation ───────────────────────────────────


def test_normalise_strips_executor_prefix():
    assert _normalise("executor.observe") == "observe"
    assert _normalise("executor.query_memory") == "query_memory"
    assert _normalise("executor.analyse") == "analyse"


def test_normalise_known_aliases():
    assert _normalise("executor.respond") == "respond_text"
    assert _normalise("respond") == "respond_text"
    assert _normalise("executor.notify") == "send_notification"
    assert _normalise("notify") == "send_notification"
    assert _normalise("executor.create") == "create_record"
    assert _normalise("create") == "create_record"
    assert _normalise("executor.api") == "call_api"
    assert _normalise("api") == "call_api"
    assert _normalise("store") == "store_insight"


def test_normalise_canonical_unchanged():
    assert _normalise("observe") == "observe"
    assert _normalise("call_api") == "call_api"
    assert _normalise("store_insight") == "store_insight"


def test_get_with_alias():
    """Registry lookup should work with aliased names."""
    registry = ExecutorRegistry()

    class _ObserveExecutor(_ConcreteExecutor):
        action_type = "observe"

    registry.register(_ObserveExecutor())
    # Lookup via alias
    assert registry.get("executor.observe") is not None
    assert registry.get("observe") is not None


def test_get_strict_raises_on_missing():
    registry = ExecutorRegistry()
    with pytest.raises(KeyError, match="No executor registered"):
        registry.get_strict("nonexistent")


def test_get_returns_none_on_missing():
    registry = ExecutorRegistry()
    assert registry.get("nonexistent") is None
