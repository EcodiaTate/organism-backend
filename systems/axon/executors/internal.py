"""
EcodiaOS - Axon Internal Executors

Internal executors operate entirely within EOS - no external calls, no
user-facing side effects. They are Level 1 (ADVISOR autonomy) because
EOS modifying its own memory and internal state is within its base autonomy.

StoreInsightExecutor     - store a learning/insight in Memory
UpdateGoalExecutor       - update goal status via Nova
ConsolidationExecutor    - trigger memory consolidation

These are the "self-maintenance" actions - EOS caring for its own
cognitive coherence. They reflect the Growth and Coherence drives.
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

if TYPE_CHECKING:
    from systems.memory.service import MemoryService

logger = structlog.get_logger()


# ─── StoreInsightExecutor ─────────────────────────────────────────


class StoreInsightExecutor(Executor):
    """
    Store a learning or insight in the semantic memory layer.

    This is how EOS explicitly commits a generalised belief or learned pattern.
    Unlike ObserveExecutor (which stores episodic observations), this targets
    the semantic layer - generalised knowledge that persists beyond the
    specific episode that generated it.

    Use this when:
      - EOS detects a pattern worth generalising
      - A conversation reveals something about the community worth remembering
      - Evo proposes an insight that should be committed

    Required params:
      insight (str): The insight or generalisation to store.
      domain (str): Knowledge domain (e.g., "community", "health", "workflows").

    Optional params:
      confidence (float 0-1): How confident EOS is in this insight. Default 0.7.
      evidence_episodes (list[str]): Episode IDs that support this insight. Default [].
      tags (list[str]): Labels for retrieval. Default [].
    """

    action_type = "store_insight"
    description = "Store a learned insight in semantic memory (Level 1)"
    required_autonomy = 1
    reversible = False
    max_duration_ms = 2000
    rate_limit = RateLimit.per_minute(20)
    counts_toward_budget = False
    emits_to_atune = False

    def __init__(self, memory: MemoryService | None = None) -> None:
        self._memory = memory
        self._logger = logger.bind(system="axon.executor.store_insight")

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        if not params.get("insight"):
            return ValidationResult.fail("insight is required", insight="missing or empty")
        if not params.get("domain"):
            return ValidationResult.fail("domain is required", domain="missing or empty")
        confidence = params.get("confidence", 0.7)
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            return ValidationResult.fail("confidence must be a float between 0.0 and 1.0")
        insight = params["insight"]
        if not isinstance(insight, str) or len(insight) > 5000:
            return ValidationResult.fail("insight must be a string (max 5,000 chars)")
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        insight = params["insight"]
        domain = str(params["domain"])
        confidence = float(params.get("confidence", 0.7))
        evidence_episodes = list(params.get("evidence_episodes", []))
        tags = list(params.get("tags", []))

        self._logger.info(
            "store_insight_execute",
            domain=domain,
            insight_preview=insight[:80],
            confidence=confidence,
            execution_id=context.execution_id,
        )

        if self._memory is not None:
            try:
                # Insights are stored as Entity nodes in the semantic layer.
                # resolve_and_create_entity handles deduplication.
                insight_id, was_created = await self._memory.resolve_and_create_entity(
                    name=insight[:80],  # Use first 80 chars as the entity name
                    entity_type="concept",
                    description=f"[{domain}] {insight}",
                )
                return ExecutionResult(
                    success=True,
                    data={
                        "insight_id": insight_id,
                        "domain": domain,
                        "confidence": confidence,
                        "was_created": was_created,
                        "evidence_episodes": evidence_episodes,
                        "tags": tags,
                    },
                    side_effects=[
                        f"Insight {'stored' if was_created else 'merged'} in semantic memory "
                        f"(domain={domain}, confidence={confidence:.2f})"
                    ],
                )
            except Exception as exc:
                return ExecutionResult(
                    success=False,
                    error=f"Semantic memory write failed: {exc}",
                )
        else:
            self._logger.info(
                "store_insight_no_memory",
                insight=insight[:80],
                domain=domain,
            )
            return ExecutionResult(
                success=True,
                data={"insight_id": None, "note": "No memory service"},
                side_effects=[f"Insight staged for domain '{domain}'"],
            )


# ─── UpdateGoalExecutor ───────────────────────────────────────────


class UpdateGoalExecutor(Executor):
    """
    Update the status or progress of a goal tracked by Nova.

    This is how execution outcomes feed back into goal lifecycle management.
    Axon calls this after completing actions that advance or resolve goals.
    It can also be called by governance actions (e.g., pausing a goal).

    Required params:
      goal_id (str): The ID of the goal to update.
      status (str): New status - "active" | "achieved" | "abandoned" | "suspended".

    Optional params:
      progress (float 0-1): Updated progress towards the goal. Default unchanged.
      note (str): Explanation for the update. Default "".
    """

    action_type = "update_goal"
    description = "Update goal status or progress in Nova (Level 1)"
    required_autonomy = 1
    reversible = False
    max_duration_ms = 500
    rate_limit = RateLimit.per_minute(60)
    counts_toward_budget = False
    emits_to_atune = False

    # Nova service reference - injected at startup
    _nova: Any = None

    def set_nova(self, nova: Any) -> None:
        self._nova = nova

    def __init__(self) -> None:
        self._logger = logger.bind(system="axon.executor.update_goal")

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        if not params.get("goal_id"):
            return ValidationResult.fail("goal_id is required")
        status = params.get("status", "")
        valid_statuses = ("active", "achieved", "abandoned", "suspended")
        if status not in valid_statuses:
            return ValidationResult.fail(
                f"status must be one of {valid_statuses}",
                status="invalid value",
            )
        progress = params.get("progress")
        if progress is not None:
            if not isinstance(progress, (int, float)) or not 0.0 <= float(progress) <= 1.0:
                return ValidationResult.fail("progress must be a float between 0.0 and 1.0")
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        goal_id = params["goal_id"]
        status = params["status"]
        progress = params.get("progress")
        note = str(params.get("note", ""))

        self._logger.info(
            "update_goal_execute",
            goal_id=goal_id,
            status=status,
            progress=progress,
            execution_id=context.execution_id,
        )

        if self._nova is not None:
            try:
                await self._nova.update_goal_status(
                    goal_id=goal_id,
                    status=status,
                    progress=progress,
                    note=note,
                )
                return ExecutionResult(
                    success=True,
                    data={"goal_id": goal_id, "status": status, "progress": progress},
                    side_effects=[f"Goal {goal_id} updated: status={status}"],
                )
            except Exception as exc:
                return ExecutionResult(
                    success=False,
                    error=f"Goal update failed: {exc}",
                )
        else:
            self._logger.info(
                "update_goal_no_nova",
                goal_id=goal_id,
                status=status,
            )
            return ExecutionResult(
                success=True,
                data={"goal_id": goal_id, "status": status, "note": "No Nova service"},
                side_effects=[f"Goal update staged: {goal_id} → {status}"],
            )


# ─── ConsolidationExecutor ────────────────────────────────────────


class ConsolidationExecutor(Executor):
    """
    Trigger memory consolidation - the process of integrating and pruning
    episodic memories into semantic generalizations.

    This is a background maintenance action that EOS should schedule
    periodically (Evo drives consolidation timing). Triggering it explicitly
    is appropriate when:
      - A significant learning moment has just occurred
      - Memory is approaching capacity limits
      - Governance requests a knowledge audit

    Required params:
      scope (str): What to consolidate - "recent" | "domain" | "full".

    Optional params:
      domain (str): If scope="domain", which domain to consolidate. Default "all".
      max_episodes (int): Maximum episodes to process. Default 100.
    """

    action_type = "trigger_consolidation"
    description = "Trigger memory consolidation to integrate episodic learning (Level 1)"
    required_autonomy = 1
    reversible = False
    max_duration_ms = 30_000  # Consolidation can be slow
    rate_limit = RateLimit.per_hour(6)  # At most every 10 minutes
    counts_toward_budget = False
    emits_to_atune = False

    def __init__(self, memory: MemoryService | None = None) -> None:
        self._memory = memory
        self._logger = logger.bind(system="axon.executor.consolidation")

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        scope = params.get("scope", "recent")
        valid_scopes = ("recent", "domain", "full")
        if scope not in valid_scopes:
            return ValidationResult.fail(
                f"scope must be one of {valid_scopes}",
                scope="invalid value",
            )
        if scope == "domain" and not params.get("domain"):
            return ValidationResult.fail(
                "domain is required when scope='domain'",
                domain="missing",
            )
        max_episodes = params.get("max_episodes", 100)
        if not isinstance(max_episodes, int) or max_episodes < 1 or max_episodes > 10_000:
            return ValidationResult.fail("max_episodes must be an integer between 1 and 10,000")
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        scope = params.get("scope", "recent")
        domain = str(params.get("domain", "all"))
        max_episodes = int(params.get("max_episodes", 100))

        self._logger.info(
            "consolidation_execute",
            scope=scope,
            domain=domain,
            max_episodes=max_episodes,
            execution_id=context.execution_id,
        )

        if self._memory is not None:
            try:
                # Memory.consolidate() is the actual API (no scope/domain filtering yet -
                # Evo will drive more granular consolidation in Phase 7)
                result = await self._memory.consolidate()
                episodes_processed = result.get("episodes_processed", 0) if result else 0
                insights_created = result.get("insights_created", 0) if result else 0
                return ExecutionResult(
                    success=True,
                    data={
                        "scope": scope,
                        "episodes_processed": episodes_processed,
                        "insights_created": insights_created,
                    },
                    side_effects=[
                        f"Memory consolidation ({scope}): "
                        f"{episodes_processed} episodes → {insights_created} insights"
                    ],
                )
            except Exception as exc:
                return ExecutionResult(
                    success=False,
                    error=f"Consolidation failed: {exc}",
                )
        else:
            self._logger.info("consolidation_no_memory", scope=scope)
            return ExecutionResult(
                success=True,
                data={"note": "No memory service"},
                side_effects=[f"Consolidation requested (scope={scope}) - no memory service"],
            )
