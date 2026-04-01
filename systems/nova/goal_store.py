"""
EcodiaOS -- Nova Goal Store

Async Neo4j persistence for Nova goal lifecycle.
Goals live in-memory (GoalManager) for fast deliberation, but are
persisted to Neo4j as :Goal nodes so they survive process restarts.

Node label  : :Goal
Node primary: id (string)

Functions:
  persist_goal(neo4j, goal)           -- MERGE and SET all fields
  load_active_goals(neo4j)            -- MATCH active + suspended on startup
  update_goal_status(neo4j, id, ...)  -- partial SET (status + progress)
  abandon_stale_goals(neo4j, hours)   -- SET abandoned on old suspended nodes

All functions catch and log exceptions so a Neo4j outage never blocks
the deliberation path.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import DriveAlignmentVector
from systems.nova.types import Goal, GoalSource, GoalStatus

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from systems.synapse.types import SystemHealthRecord

_logger = structlog.get_logger()


def _dt_str(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat()


def _goal_to_props(goal: Goal) -> dict[str, Any]:
    da = goal.drive_alignment
    return {
        "id": goal.id,
        "description": goal.description,
        "target_domain": goal.target_domain,
        "success_criteria": goal.success_criteria,
        "priority": goal.priority,
        "urgency": goal.urgency,
        "importance": goal.importance,
        "source": goal.source.value,
        "status": goal.status.value,
        "progress": goal.progress,
        "deadline": _dt_str(goal.deadline),
        "created_at": _dt_str(goal.created_at),
        "updated_at": _dt_str(goal.updated_at),
        "drive_coherence": da.coherence,
        "drive_care": da.care,
        "drive_growth": da.growth,
        "drive_honesty": da.honesty,
        "depends_on_json": json.dumps(goal.depends_on),
        "blocks_json": json.dumps(goal.blocks),
        "intents_issued_count": len(goal.intents_issued),
    }


def _parse_neo4j_dt(v: Any) -> datetime | None:
    if v is None:
        return None
    if hasattr(v, "to_native"):
        return v.to_native()  # type: ignore[no-any-return]
    if hasattr(v, "isoformat"):
        return datetime.fromisoformat(v.isoformat())
    if isinstance(v, str):
        return datetime.fromisoformat(v)
    return None


def _row_to_goal(row: dict[str, Any]) -> Goal | None:
    try:
        g: dict[str, Any] = row.get("g", row)
        da_raw = {
            "coherence": float(g.get("drive_coherence", 0.5)),
            "care": float(g.get("drive_care", 0.5)),
            "growth": float(g.get("drive_growth", 0.5)),
            "honesty": float(g.get("drive_honesty", 0.5)),
        }
        depends_on: list[str] = json.loads(g.get("depends_on_json", "[]") or "[]")
        blocks: list[str] = json.loads(g.get("blocks_json", "[]") or "[]")
        now = datetime.now(tz=UTC)
        return Goal(
            id=str(g["id"]),
            description=g.get("description", ""),
            target_domain=g.get("target_domain", ""),
            success_criteria=g.get("success_criteria", ""),
            priority=float(g.get("priority", 0.5)),
            urgency=float(g.get("urgency", 0.3)),
            importance=float(g.get("importance", 0.5)),
            drive_alignment=DriveAlignmentVector(**da_raw),
            source=GoalSource(g.get("source", GoalSource.USER_REQUEST.value)),
            status=GoalStatus(g.get("status", GoalStatus.ACTIVE.value)),
            progress=float(g.get("progress", 0.0)),
            deadline=_parse_neo4j_dt(g.get("deadline")),
            created_at=_parse_neo4j_dt(g.get("created_at")) or now,
            updated_at=_parse_neo4j_dt(g.get("updated_at")) or now,
            depends_on=depends_on,
            blocks=blocks,
        )
    except Exception as exc:
        _logger.warning("goal_store_row_parse_failed", error=str(exc), row_keys=list(row.keys()))
        return None


async def persist_goal(neo4j: Neo4jClient, goal: Goal) -> None:
    props = _goal_to_props(goal)
    query = (
        "MERGE (g:Goal {id: $id}) "
        "SET g += $props, g.persisted_at = datetime()"
    )
    try:
        await neo4j.execute_write(query, {"id": goal.id, "props": props})
        _logger.debug("goal_persisted", goal_id=goal.id, status=goal.status.value)
    except Exception as exc:
        _logger.warning("goal_persist_failed", goal_id=goal.id, error=str(exc))


def _is_stale_maintenance_goal(
    goal: Goal,
    now: datetime,
    health_records: dict[str, SystemHealthRecord] | None,
    stale_minutes: float = 30.0,
) -> bool:
    """
    Return True if this goal should be suppressed on load.

    A maintenance goal is stale - and should not enter active memory - when:
      1. source = MAINTENANCE
      2. created_at is older than `stale_minutes`
      3. the target system is currently healthy (per Synapse's last known record)

    Condition 3 is the guard: if the system is still sick, keep the goal alive.
    If health_records is None (Synapse not yet wired), skip the filter entirely
    to avoid suppressing goals during early startup.
    """
    if goal.source != GoalSource.MAINTENANCE:
        return False
    age = (now - goal.created_at.replace(tzinfo=UTC) if goal.created_at.tzinfo is None
           else now - goal.created_at)
    if age.total_seconds() < stale_minutes * 60:
        return False
    if health_records is None:
        # Can't confirm health - load the goal conservatively.
        return False
    from systems.synapse.types import SystemStatus
    record = health_records.get(goal.target_domain)
    if record is None:
        # Unknown system - suppress (no active incident to monitor).
        return True
    return record.status == SystemStatus.HEALTHY


async def load_active_goals(
    neo4j: Neo4jClient,
    health_records: dict[str, SystemHealthRecord] | None = None,
) -> list[Goal]:
    """
    Load active and suspended goals from Neo4j on startup.

    Maintenance goals older than 30 minutes whose target system is currently
    healthy are filtered out before returning - they represent stale recovery
    monitors from prior sessions and must not fill active capacity.
    Suppressed goals are immediately marked completed in Neo4j so they don't
    re-appear on the next boot.
    """
    query = (
        "MATCH (g:Goal) "
        "WHERE g.status IN ['active', 'suspended'] "
        "RETURN g "
        "ORDER BY g.priority DESC"
    )
    try:
        rows = await neo4j.execute_read(query, {})
        now = datetime.now(tz=UTC)
        goals: list[Goal] = []
        suppressed_ids: list[str] = []
        for row in rows:
            goal = _row_to_goal(row)
            if goal is None:
                continue
            if _is_stale_maintenance_goal(goal, now, health_records):
                suppressed_ids.append(goal.id)
                _logger.debug(
                    "stale_maintenance_goal_suppressed",
                    goal_id=goal.id,
                    target_domain=goal.target_domain,
                    age_minutes=round(
                        (now - goal.created_at.replace(tzinfo=UTC)
                         if goal.created_at.tzinfo is None else
                         now - goal.created_at).total_seconds() / 60,
                        1,
                    ),
                )
                continue
            goals.append(goal)

        # Mark suppressed goals as completed in Neo4j so they don't re-appear on next boot.
        if suppressed_ids:
            now_str = now.isoformat()
            mark_query = (
                "MATCH (g:Goal) "
                "WHERE g.id IN $ids "
                "SET g.status = 'completed', g.updated_at = $now"
            )
            try:
                await neo4j.execute_write(mark_query, {"ids": suppressed_ids, "now": now_str})
                _logger.info(
                    "stale_maintenance_goals_marked_completed",
                    count=len(suppressed_ids),
                )
            except Exception as mark_exc:
                _logger.warning(
                    "stale_maintenance_goals_mark_failed",
                    error=str(mark_exc),
                    count=len(suppressed_ids),
                )

        _logger.info(
            "goals_loaded_from_neo4j",
            count=len(goals),
            suppressed_stale_maintenance=len(suppressed_ids),
        )
        return goals
    except Exception as exc:
        _logger.warning("goals_load_failed", error=str(exc))
        return []


async def update_goal_status(
    neo4j: Neo4jClient,
    goal_id: str,
    status: GoalStatus,
    progress: float,
) -> None:
    query = (
        "MATCH (g:Goal {id: $id}) "
        "SET g.status = $status, g.progress = $progress, g.updated_at = $updated_at"
    )
    try:
        await neo4j.execute_write(query, {
            "id": goal_id,
            "status": status.value,
            "progress": progress,
            "updated_at": datetime.now(tz=UTC).isoformat(),
        })
        _logger.debug(
            "goal_status_updated",
            goal_id=goal_id,
            status=status.value,
            progress=round(progress, 3),
        )
    except Exception as exc:
        _logger.warning("goal_status_update_failed", goal_id=goal_id, error=str(exc))


async def abandon_stale_goals(neo4j: Neo4jClient, suspended_hours: float = 48.0) -> int:
    cutoff = (datetime.now(tz=UTC) - timedelta(hours=suspended_hours)).isoformat()
    now_str = datetime.now(tz=UTC).isoformat()
    query = (
        "MATCH (g:Goal) "
        "WHERE g.status = 'suspended' AND g.updated_at < $cutoff "
        "SET g.status = 'abandoned', g.updated_at = $now "
        "RETURN count(g) AS abandoned_count"
    )
    try:
        rows = await neo4j.execute_write(query, {"cutoff": cutoff, "now": now_str})
        count: int = int(rows[0].get("abandoned_count", 0)) if rows else 0
        if count:
            _logger.info("stale_goals_abandoned_in_neo4j", count=count)
        return count
    except Exception as exc:
        _logger.warning("abandon_stale_goals_failed", error=str(exc))
        return 0
