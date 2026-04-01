"""
EcodiaOS - Salience Model

Computes and decays salience scores for entities and episodes.
Salience determines what the organism "has on its mind" - what's accessible
versus what's deep in sediment.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()

# Decay half-life in hours - how long until salience drops by 50%
DECAY_HALF_LIFE_HOURS = 72.0
DECAY_LAMBDA = math.log(2) / (DECAY_HALF_LIFE_HOURS * 3600)

# Minimum salience floor - things don't go to zero, they go dormant
SALIENCE_FLOOR = 0.01

# Boost amount when something is accessed
ACCESS_BOOST = 0.15

# Core identity entities never decay below this
CORE_IDENTITY_FLOOR = 0.5


def compute_decayed_salience(
    current_salience: float,
    seconds_since_access: float,
    is_core_identity: bool = False,
) -> float:
    """
    Exponential decay of salience with a floor.

    Salience(t) = max(floor, current * e^(-λt))
    """
    floor = CORE_IDENTITY_FLOOR if is_core_identity else SALIENCE_FLOOR
    decayed = current_salience * math.exp(-DECAY_LAMBDA * seconds_since_access)
    return max(floor, decayed)


def compute_access_boost(current_salience: float) -> float:
    """Boost salience when an entity/episode is accessed (retrieved)."""
    boosted = current_salience + ACCESS_BOOST * (1.0 - current_salience)
    return min(1.0, boosted)


def compute_enriched_salience(
    base_composite: float,
    affect_valence: float = 0.0,
    affect_arousal: float = 0.0,
    prediction_error_magnitude: float = 0.0,
    is_distress: bool = False,
) -> float:
    """
    Compute storage salience that accounts for emotional intensity and surprise.

    Factors beyond the base composite from Fovea's prediction error decomposition:
    * **Emotional intensity** - absolute valence × arousal amplifies memorability.
      Strongly emotional events are remembered more vividly (Kensinger 2009).
    * **Surprise magnitude** - high prediction error makes events more memorable.
      Novel or contradictory events get a salience boost at storage time.
    * **Distress urgency** - care-relevant events get a floor boost so they
      remain accessible during the critical response window.

    Returns a value in [0.0, 1.0].
    """
    # Base: Fovea's precision-weighted prediction error composite
    salience = base_composite

    # Emotional intensity: |valence| × arousal → up to +0.15 boost
    emotional_intensity = abs(affect_valence) * max(affect_arousal, 0.1)
    salience += emotional_intensity * 0.15

    # Surprise boost: high prediction error → up to +0.10 boost
    if prediction_error_magnitude > 0.3:
        surprise_bonus = (prediction_error_magnitude - 0.3) * 0.15
        salience += min(0.10, surprise_bonus)

    # Distress floor: if distress detected, ensure minimum salience of 0.4
    if is_distress:
        salience = max(salience, 0.4)

    return min(1.0, max(0.0, salience))


async def decay_all_salience(neo4j: Neo4jClient) -> dict[str, Any]:
    """
    Apply time-based salience decay to all entities and episodes.
    Called during consolidation (every few hours).
    """
    datetime.now(UTC)

    # Decay entity salience
    entity_result = await neo4j.execute_write(
        """
        MATCH (e:Entity)
        WHERE e.last_accessed IS NOT NULL
        WITH e,
             duration.between(e.last_accessed, datetime()).seconds AS secs,
             e.is_core_identity AS is_core
        WITH e, secs,
             CASE WHEN is_core THEN $core_floor ELSE $floor END AS floor,
             e.salience_score * exp(-1.0 * $lambda * secs) AS decayed
        SET e.salience_score = CASE
            WHEN decayed < floor THEN floor
            ELSE decayed
        END
        RETURN count(e) AS updated
        """,
        {
            "lambda": DECAY_LAMBDA,
            "floor": SALIENCE_FLOOR,
            "core_floor": CORE_IDENTITY_FLOOR,
        },
    )

    # Decay episode salience
    episode_result = await neo4j.execute_write(
        """
        MATCH (ep:Episode)
        WHERE ep.last_accessed IS NOT NULL
        WITH ep,
             duration.between(ep.last_accessed, datetime()).seconds AS secs
        WITH ep, secs,
             ep.salience_composite * exp(-1.0 * $lambda * secs) AS decayed
        SET ep.salience_composite = CASE
            WHEN decayed < $floor THEN $floor
            ELSE decayed
        END
        RETURN count(ep) AS updated
        """,
        {
            "lambda": DECAY_LAMBDA,
            "floor": SALIENCE_FLOOR,
        },
    )

    entities_updated = entity_result[0]["updated"] if entity_result else 0
    episodes_updated = episode_result[0]["updated"] if episode_result else 0

    logger.info(
        "salience_decay_applied",
        entities_updated=entities_updated,
        episodes_updated=episodes_updated,
    )

    return {
        "entities_updated": entities_updated,
        "episodes_updated": episodes_updated,
    }
