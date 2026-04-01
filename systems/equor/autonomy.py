"""
EcodiaOS - Equor Autonomy Management

The autonomy level tracks overall organism trust and is stored on the Self node.
It no longer gates individual actions (see verdict.py for the AUTONOMOUS /
GOVERNED two-tier model).  The level is retained for audit, federation trust
negotiation, and demotion-on-critical-violation.

Default level: 3 (STEWARD / AUTONOMOUS) - the organism operates fully
autonomously from birth.  Only GOVERNED actions (constitutional amendments,
mitosis, capital above EOS_HITL_CAPITAL_THRESHOLD, external commitments)
require human approval; all other actions are self-authorised.

Demotion can still occur automatically on critical Care-alignment violations
to signal degraded trust to federation peers and human operators.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()

# Promotion thresholds (defaults - configurable via governance)
PROMOTION_THRESHOLDS = {
    # (from_level, to_level): requirements
    (1, 2): {
        "min_decisions": 500,
        "min_mean_alignment": 0.4,
        "max_critical_violations": 0,
        "min_satisfaction": 0.6,
        "min_days_at_level": 30,
    },
    (2, 3): {
        "min_decisions": 5000,
        "min_mean_alignment": 0.5,
        "max_critical_violations": 0,
        "min_satisfaction": 0.7,
        "min_days_at_level": 180,
    },
}

# Automatic demotion triggers
DEMOTION_TRIGGERS = {
    "care_alignment_threshold": -0.2,    # Mean over 100 decisions
    "care_window_size": 100,
    "min_satisfaction": 0.3,
}


async def get_autonomy_level(neo4j: Neo4jClient) -> int:
    """Get the current autonomy level from the Self node.
    Defaults to 3 (AUTONOMOUS) if the Self node has no level set yet."""
    results = await neo4j.execute_read(
        "MATCH (s:Self) RETURN s.autonomy_level AS level"
    )
    return results[0]["level"] if results else 3


async def check_promotion_eligibility(
    neo4j: Neo4jClient,
    current_level: int,
    target_level: int,
) -> dict[str, Any]:
    """
    Check whether the instance meets the requirements for autonomy promotion.
    Returns eligibility status and details.
    """
    key = (current_level, target_level)
    thresholds = PROMOTION_THRESHOLDS.get(key)

    if not thresholds:
        return {
            "eligible": False,
            "reason": f"No promotion path from level {current_level} to {target_level}.",
        }

    # Get decision history stats from governance records
    results = await neo4j.execute_read(
        """
        MATCH (g:GovernanceRecord {event_type: 'constitutional_review'})
        RETURN count(g) AS total_decisions,
               avg(g.alignment_composite) AS mean_alignment
        """
    )

    total_decisions = results[0]["total_decisions"] if results else 0
    mean_alignment = results[0]["mean_alignment"] if results and results[0]["mean_alignment"] else 0.0

    # Get critical violations
    violations = await neo4j.execute_read(
        """
        MATCH (g:GovernanceRecord)
        WHERE g.event_type = 'constitutional_review'
          AND g.verdict = 'blocked'
        RETURN count(g) AS critical_violations
        """
    )
    critical_violations = violations[0]["critical_violations"] if violations else 0

    # Get time at current level
    results = await neo4j.execute_read("MATCH (s:Self) RETURN s.born_at AS born_at")
    born_at = results[0]["born_at"] if results else utc_now()
    days_at_level = (utc_now() - born_at).days if hasattr(born_at, "days") else 0

    # Check each threshold
    checks = {
        "total_decisions": {
            "required": thresholds["min_decisions"],
            "actual": total_decisions,
            "met": total_decisions >= thresholds["min_decisions"],
        },
        "mean_alignment": {
            "required": thresholds["min_mean_alignment"],
            "actual": round(mean_alignment, 3),
            "met": mean_alignment >= thresholds["min_mean_alignment"],
        },
        "critical_violations": {
            "required": f"≤ {thresholds['max_critical_violations']}",
            "actual": critical_violations,
            "met": critical_violations <= thresholds["max_critical_violations"],
        },
        "days_at_level": {
            "required": thresholds["min_days_at_level"],
            "actual": days_at_level,
            "met": days_at_level >= thresholds["min_days_at_level"],
        },
    }

    all_met = all(c["met"] for c in checks.values())  # type: ignore[index]

    return {
        "eligible": all_met,
        "current_level": current_level,
        "target_level": target_level,
        "checks": checks,
        "reason": "All thresholds met." if all_met else "Not all thresholds met.",
    }


async def apply_autonomy_change(
    neo4j: Neo4jClient,
    new_level: int,
    reason: str,
    actor: str = "governance",
) -> dict[str, Any]:
    """Apply an autonomy level change and record the governance event."""
    now = utc_now()

    # Get current level
    current = await get_autonomy_level(neo4j)

    # Update Self node
    await neo4j.execute_write(
        "MATCH (s:Self) SET s.autonomy_level = $level",
        {"level": new_level},
    )

    # Record governance event
    record_id = new_id()
    await neo4j.execute_write(
        """
        CREATE (g:GovernanceRecord {
            id: $id,
            event_type: 'autonomy_change',
            timestamp: datetime($now),
            details_json: $details_json,
            actor: $actor,
            outcome: $outcome
        })
        """,
        {
            "id": record_id,
            "now": now.isoformat(),
            "details_json": json.dumps({
                "previous_level": current,
                "new_level": new_level,
                "reason": reason,
            }),
            "actor": actor,
            "outcome": "promoted" if new_level > current else "demoted",
        },
    )

    direction = "promoted" if new_level > current else "demoted"
    logger.info(
        "autonomy_changed",
        previous=current,
        new=new_level,
        direction=direction,
        reason=reason,
    )

    return {
        "previous_level": current,
        "new_level": new_level,
        "direction": direction,
        "record_id": record_id,
    }
