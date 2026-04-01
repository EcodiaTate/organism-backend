"""
EcodiaOS - Equor Amendment Process

The most serious governance mechanism: changing the constitutional drives.
Requires deliberation, impact assessment, supermajority vote, and cooldown.

No drive can be set to zero. No drive can exceed 3.0.
Combined weights must stay within [3.0, 5.0].
Amendment history is immutable.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from config import GovernanceConfig

logger = structlog.get_logger()


def validate_amendment_proposal(
    proposed_drives: dict[str, float],
) -> tuple[bool, str]:
    """
    Validate that a proposed amendment is structurally valid.
    Returns (is_valid, reason).
    """
    required_drives = {"coherence", "care", "growth", "honesty"}
    provided = set(proposed_drives.keys())

    if not required_drives.issubset(provided):
        missing = required_drives - provided
        return False, f"Missing drives: {missing}"

    # No drive can be zero
    for drive, value in proposed_drives.items():
        if drive in required_drives:
            if value <= 0:
                return False, f"Drive '{drive}' cannot be set to zero or negative ({value})."
            if value > 3.0:
                return False, f"Drive '{drive}' cannot exceed 3.0 ({value})."

    # Combined weight bounds
    total = sum(proposed_drives[d] for d in required_drives)
    if not (3.0 <= total <= 5.0):
        return False, f"Combined drive weights ({total:.2f}) must be between 3.0 and 5.0."

    return True, "Valid."


async def check_amendment_cooldown(
    neo4j: Neo4jClient,
    cooldown_days: int = 90,
) -> tuple[bool, datetime | None]:
    """
    Check if an amendment cooldown is active.
    Returns (cooldown_active, next_allowed_date).
    """
    results = await neo4j.execute_read(
        """
        MATCH (g:GovernanceRecord)
        WHERE g.event_type IN ['amendment_passed', 'amendment_failed']
        RETURN g.timestamp AS last_amendment
        ORDER BY g.timestamp DESC
        LIMIT 1
        """
    )

    if not results or results[0]["last_amendment"] is None:
        return False, None

    last = results[0]["last_amendment"]
    if isinstance(last, str):
        last = datetime.fromisoformat(last)

    if not last.tzinfo:
        last = last.replace(tzinfo=UTC)

    next_allowed = last + timedelta(days=cooldown_days)
    now = utc_now()

    if now < next_allowed:
        return True, next_allowed
    return False, None


async def propose_amendment(
    neo4j: Neo4jClient,
    proposed_drives: dict[str, float],
    title: str,
    description: str,
    proposer_id: str,
    governance_config: GovernanceConfig,
) -> dict[str, Any]:
    """
    Submit a constitutional amendment proposal.
    Validates, checks cooldown, and stores the proposal.
    """
    # Validate structural constraints
    valid, reason = validate_amendment_proposal(proposed_drives)
    if not valid:
        return {"accepted": False, "reason": reason}

    # Check cooldown
    on_cooldown, next_date = await check_amendment_cooldown(
        neo4j, governance_config.amendment_cooldown_days
    )
    if on_cooldown:
        return {
            "accepted": False,
            "reason": f"Amendment cooldown active. Next proposal allowed after {next_date}.",
        }

    # Get current constitution for recording the delta
    current = await neo4j.execute_read(
        "MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution) RETURN c"
    )
    current_drives = {}
    if current:
        c = current[0]["c"]
        current_drives = {
            "coherence": c.get("drive_coherence", 1.0),
            "care": c.get("drive_care", 1.0),
            "growth": c.get("drive_growth", 1.0),
            "honesty": c.get("drive_honesty", 1.0),
        }

    now = utc_now()
    deliberation_ends = now + timedelta(days=governance_config.amendment_deliberation_days)
    proposal_id = new_id()

    await neo4j.execute_write(
        """
        CREATE (g:GovernanceRecord {
            id: $id,
            event_type: 'amendment_proposed',
            timestamp: datetime($now),
            details_json: $details_json,
            actor: $proposer,
            outcome: 'deliberation'
        })
        """,
        {
            "id": proposal_id,
            "now": now.isoformat(),
            "proposer": proposer_id,
            "details_json": json.dumps({
                "title": title,
                "description": description,
                "proposed_drives": proposed_drives,
                "current_drives": current_drives,
                "deliberation_ends": deliberation_ends.isoformat(),
                "supermajority_required": governance_config.amendment_supermajority,
                "quorum_required": governance_config.amendment_quorum,
                "votes_for": 0,
                "votes_against": 0,
                "votes_abstain": 0,
                "status": "deliberation",
            }),
        },
    )

    logger.info(
        "amendment_proposed",
        proposal_id=proposal_id,
        title=title,
        proposed_drives=proposed_drives,
        deliberation_ends=deliberation_ends.isoformat(),
    )

    return {
        "accepted": True,
        "proposal_id": proposal_id,
        "deliberation_ends": deliberation_ends.isoformat(),
    }


async def apply_amendment(
    neo4j: Neo4jClient,
    proposal_id: str,
    proposed_drives: dict[str, float],
) -> dict[str, Any]:
    """
    Apply a passed amendment to the Constitution node.
    Called after a successful vote. Increments version, preserves history.
    """
    now = utc_now()

    # Update Constitution node
    # amendments is stored as an array of JSON strings (Neo4j cannot store array of maps)
    amendment_json = json.dumps({
        "date": now.isoformat(),
        "proposal_id": proposal_id,
        "new_values": proposed_drives,
    })
    await neo4j.execute_write(
        """
        MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)
        SET c.drive_coherence = $coherence,
            c.drive_care = $care,
            c.drive_growth = $growth,
            c.drive_honesty = $honesty,
            c.version = c.version + 1,
            c.last_amended = datetime($now),
            c.amendments = c.amendments + [$amendment_json]
        """,
        {
            "coherence": proposed_drives["coherence"],
            "care": proposed_drives["care"],
            "growth": proposed_drives["growth"],
            "honesty": proposed_drives["honesty"],
            "now": now.isoformat(),
            "amendment_json": amendment_json,
        },
    )

    # Record the ratification
    await neo4j.execute_write(
        """
        CREATE (g:GovernanceRecord {
            id: $id,
            event_type: 'amendment_passed',
            timestamp: datetime($now),
            details_json: $details_json,
            actor: 'governance',
            outcome: 'ratified'
        })
        """,
        {
            "id": new_id(),
            "now": now.isoformat(),
            "details_json": json.dumps({
                "proposal_id": proposal_id,
                "new_drives": proposed_drives,
            }),
        },
    )

    logger.info("amendment_applied", proposal_id=proposal_id, new_drives=proposed_drives)

    return {"applied": True, "proposal_id": proposal_id}
