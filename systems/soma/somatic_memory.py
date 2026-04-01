"""
EcodiaOS - Soma Somatic Memory

Somatic marker creation and embodied retrieval reranking.

The organism remembers not just what happened, but how it felt.
Somatic markers are 19D snapshots (9 sensed + 9 errors + 1 PE)
attached to memory traces. During retrieval, memories with similar
somatic markers are boosted - state-congruent recall.

Budget: 1ms for marker creation, linear in candidates for reranking.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import structlog

from systems.soma.types import (
    ALL_DIMENSIONS,
    InteroceptiveDimension,
    InteroceptiveState,
    SomaticMarker,
)

if TYPE_CHECKING:
    from systems.soma.types import AllostaticSignal

logger = structlog.get_logger("systems.soma.somatic_memory")

# Somatic marker vector dimensionality: 9 sensed + 9 errors + 1 PE
MARKER_VECTOR_DIM = 19


class SomaticMemoryIntegration:
    """
    Creates somatic markers and reranks memory retrievals by somatic similarity.

    Somatic markers are the bridge between interoception and memory -
    they enable the organism to recall what situations *felt like*,
    not just what happened. This is the computational basis of intuition.
    """

    def __init__(self, rerank_boost: float = 0.3) -> None:
        self._rerank_boost = rerank_boost

    def create_marker(
        self,
        state: InteroceptiveState,
        attractor_label: str = "",
    ) -> SomaticMarker:
        """
        Snapshot current interoceptive state as a somatic marker.

        The marker captures:
        - The 9D sensed state (what the organism feels)
        - The 9D moment-horizon errors (what the organism wants to change)
        - The prediction error magnitude (how surprised the organism is)
        - The allostatic context (which attractor we're in)

        Budget: <=1ms.
        """
        moment_errors = state.errors.get("moment", {})
        return SomaticMarker(
            interoceptive_snapshot={d: state.sensed.get(d, 0.0) for d in ALL_DIMENSIONS},
            allostatic_error_snapshot={d: moment_errors.get(d, 0.0) for d in ALL_DIMENSIONS},
            prediction_error_at_encoding=state.max_error_magnitude,
            allostatic_context=attractor_label,
        )

    def somatic_rerank(
        self,
        candidates: list[Any],
        current_state: InteroceptiveState,
        financial_modifiers: dict[str, float] | None = None,
    ) -> list[Any]:
        """
        Boost memories with similar somatic markers.

        For each candidate with a somatic marker, computes cosine similarity
        between its 19D marker vector and the current state's marker vector.
        Applies up to +30% (configurable) salience boost.

        When financial_modifiers is provided (from TemporalDepthManager),
        additionally boosts exploration-tagged or revenue-tagged memories
        based on the organism's financial horizon:
          - exploration_boost: multiplier for memories with exploration/research context
          - revenue_boost: multiplier for memories with revenue/survival context

        Returns candidates sorted by boosted salience descending.
        """
        if not candidates:
            return candidates

        current_vector = current_state.to_marker_vector()
        current_norm = _vector_norm(current_vector)

        if current_norm < 1e-10:
            return candidates  # Zero state - no meaningful comparison

        exploration_boost = 1.0
        revenue_boost = 1.0
        if financial_modifiers is not None:
            exploration_boost = financial_modifiers.get("exploration_boost", 1.0)
            revenue_boost = financial_modifiers.get("revenue_boost", 1.0)

        boosted: list[tuple[float, Any]] = []

        for candidate in candidates:
            salience = _get_salience(candidate)
            marker_vector = _get_marker_vector(candidate)

            if marker_vector is not None and len(marker_vector) == MARKER_VECTOR_DIM:
                similarity = _cosine_similarity(current_vector, marker_vector, current_norm)
                # Boost: salience *= (1.0 + boost_factor * similarity)
                # similarity is in [-1, 1], but somatic states are mostly positive
                boost = max(0.0, similarity) * self._rerank_boost
                boosted_salience = salience * (1.0 + boost)
            else:
                boosted_salience = salience

            # Apply financial horizon modifiers based on memory context
            if financial_modifiers is not None:
                context = _get_allostatic_context(candidate)
                if context in _EXPLORATION_CONTEXTS:
                    boosted_salience *= exploration_boost
                elif context in _REVENUE_CONTEXTS:
                    boosted_salience *= revenue_boost

            boosted.append((boosted_salience, candidate))

        # Sort by boosted salience descending
        boosted.sort(key=lambda x: x[0], reverse=True)

        # Update salience on candidates if they have the attribute
        for new_salience, candidate in boosted:
            if hasattr(candidate, "salience_score"):
                candidate.salience_score = new_salience
            elif isinstance(candidate, dict) and "salience_score" in candidate:
                candidate["salience_score"] = new_salience

        return [c for _, c in boosted]


# ─── Utility Functions ────────────────────────────────────────────


def _cosine_similarity(
    a: list[float],
    b: list[float],
    a_norm: float | None = None,
) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = a_norm if a_norm is not None else _vector_norm(a)
    norm_b = _vector_norm(b)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return dot / (norm_a * norm_b)


def _vector_norm(v: list[float]) -> float:
    """Euclidean norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def _get_salience(candidate: Any) -> float:
    """Extract salience score from a candidate (object or dict)."""
    if hasattr(candidate, "salience_score"):
        return float(candidate.salience_score or 0.0)
    if isinstance(candidate, dict):
        return float(candidate.get("salience_score", 0.0))
    return 0.0


# Context classifications for financial horizon modulation
_EXPLORATION_CONTEXTS: frozenset[str] = frozenset({
    "creative_ferment", "flow", "wonder", "curiosity",
    "exploration", "deep_processing", "research",
})
_REVENUE_CONTEXTS: frozenset[str] = frozenset({
    "anxiety_spiral", "frustration", "revenue", "bounty",
    "tollbooth", "survival", "austerity", "emergency",
})


def _get_allostatic_context(candidate: Any) -> str:
    """Extract allostatic context label from a candidate."""
    if hasattr(candidate, "somatic_marker"):
        marker = candidate.somatic_marker
        if marker is not None and hasattr(marker, "allostatic_context"):
            return str(marker.allostatic_context).lower()
    if isinstance(candidate, dict):
        marker = candidate.get("somatic_marker")
        if isinstance(marker, dict):
            return str(marker.get("allostatic_context", "")).lower()
        if marker is not None and hasattr(marker, "allostatic_context"):
            return str(marker.allostatic_context).lower()
    return ""


def _get_marker_vector(candidate: Any) -> list[float] | None:
    """Extract somatic marker vector from a candidate."""
    # Try somatic_marker.to_vector()
    if hasattr(candidate, "somatic_marker"):
        marker = candidate.somatic_marker
        if marker is not None and hasattr(marker, "to_vector"):
            result: list[float] = marker.to_vector()
            return result
    # Try somatic_vector attribute (from Neo4j)
    if hasattr(candidate, "somatic_vector"):
        vec = candidate.somatic_vector
        if vec is not None and isinstance(vec, list):
            return list(vec)
    # Try dict access
    if isinstance(candidate, dict):
        if "somatic_marker" in candidate:
            marker = candidate["somatic_marker"]
            if isinstance(marker, dict) and "to_vector" not in dir(marker):
                # Raw dict marker - reconstruct
                return None
            if hasattr(marker, "to_vector"):
                result2: list[float] = marker.to_vector()
                return result2
        if "somatic_vector" in candidate:
            val = candidate["somatic_vector"]
            if isinstance(val, list):
                return list(val)
    return None


# ─── GAP 5: Somatic Marker Write Protocol ────────────────────────


class SomaticMarkerWriter:
    """
    Writes SomaticMarker nodes to Neo4j and links them to Episode nodes.

    Protocol (GAP 5):
    - After each allostatic adjustment, Soma calls write_marker_for_episode()
      off the critical path (fire-and-forget via asyncio.create_task).
    - Creates a (:SomaticMarker) Neo4j node with the full 19D vector.
    - Links it to the triggering (:Episode) via [:MARKS] relationship.
    - The marker captures valence, arousal_delta, metabolic_cost, body_system
      as first-class properties for downstream query efficiency.

    Timing note: The marker reflects state-at-allostatic-adjustment (the cycle
    when an urgency threshold was crossed), not state-at-encoding. This is the
    correct semantics - we want to know how the organism felt *when it decided
    to act*, not just when the memory was filed.
    """

    def __init__(self, neo4j_driver: Any | None = None) -> None:
        self._driver = neo4j_driver
        self._log = logger.bind(subsystem="soma.marker_writer")
        # Rolling stats for observability
        self._write_count: int = 0
        self._error_count: int = 0

    def set_driver(self, driver: Any) -> None:
        """Wire Neo4j driver (called from SomaService.initialize)."""
        self._driver = driver

    async def write_marker_for_episode(
        self,
        marker: SomaticMarker,
        episode_id: str,
        signal: AllostaticSignal,
        body_system: str = "soma",
    ) -> bool:
        """
        Write a SomaticMarker node to Neo4j and link it to an Episode.

        Creates:
            (:SomaticMarker {
                marker_id, episode_id, valence, arousal_delta,
                metabolic_cost, body_system, allostatic_context,
                prediction_error, somatic_vector[19], created_at
            })
            -[:MARKS {valence, arousal_delta, body_system}]->
            (:Episode {episode_id})

        Returns True on success, False on failure (non-fatal - markers are
        best-effort; the organism continues without them).
        """
        if self._driver is None:
            return False

        try:
            from primitives.common import utc_now
            sensed = marker.interoceptive_snapshot
            errors = marker.allostatic_error_snapshot

            # Derive marker fields for query efficiency
            valence = float(sensed.get(InteroceptiveDimension.VALENCE, 0.0))
            # Arousal delta = current arousal error (how far from setpoint)
            arousal_delta = float(errors.get(InteroceptiveDimension.AROUSAL, 0.0))
            # Metabolic cost proxy: energy deficit magnitude
            metabolic_cost = abs(float(errors.get(InteroceptiveDimension.ENERGY, 0.0)))

            somatic_vector = marker.to_vector()
            now_iso = utc_now().isoformat()

            query = """
            MERGE (ep:Episode {episode_id: $episode_id})
            CREATE (sm:SomaticMarker {
                marker_id:         $marker_id,
                episode_id:        $episode_id,
                valence:           $valence,
                arousal_delta:     $arousal_delta,
                metabolic_cost:    $metabolic_cost,
                body_system:       $body_system,
                allostatic_context: $allostatic_context,
                prediction_error:  $prediction_error,
                somatic_vector:    $somatic_vector,
                urgency_at_mark:   $urgency,
                dominant_error:    $dominant_error,
                created_at:        $created_at,
                ingestion_time:    $created_at
            })
            CREATE (sm)-[:MARKS {
                valence:       $valence,
                arousal_delta: $arousal_delta,
                body_system:   $body_system,
                marked_at:     $created_at
            }]->(ep)
            RETURN sm.marker_id AS marker_id
            """

            params = {
                "marker_id": f"sm_{episode_id}_{int(time.time() * 1000)}",
                "episode_id": episode_id,
                "valence": valence,
                "arousal_delta": arousal_delta,
                "metabolic_cost": metabolic_cost,
                "body_system": body_system,
                "allostatic_context": marker.allostatic_context,
                "prediction_error": marker.prediction_error_at_encoding,
                "somatic_vector": somatic_vector,
                "urgency": round(signal.urgency, 4),
                "dominant_error": signal.dominant_error.value if signal.dominant_error else "",
                "created_at": now_iso,
            }

            async with self._driver.session() as session:
                result = await session.run(query, **params)
                record = await result.single()
                if record:
                    self._write_count += 1
                    self._log.debug(
                        "somatic_marker_written",
                        marker_id=record["marker_id"],
                        episode_id=episode_id,
                        valence=round(valence, 3),
                        allostatic_context=marker.allostatic_context,
                    )
                    return True

        except Exception as exc:
            self._error_count += 1
            self._log.warning(
                "somatic_marker_write_error",
                episode_id=episode_id,
                error=str(exc),
            )
        return False

    async def write_marker_for_adjustment(
        self,
        marker: SomaticMarker,
        signal: AllostaticSignal,
        adjustment_type: str,
        body_system: str = "soma",
    ) -> bool:
        """
        Write a standalone SomaticMarker node for an allostatic adjustment
        that has no associated episode (e.g. urgency-critical responses).

        Creates a (:SomaticMarker) with adjustment_type property but no
        [:MARKS] relationship - it represents an organism-level somatic
        event rather than an episodic memory annotation.
        """
        if self._driver is None:
            return False

        try:
            from primitives.common import new_id, utc_now
            sensed = marker.interoceptive_snapshot

            valence_key = next((k for k in sensed if str(k) == "valence"), None)
            valence = float(sensed.get(valence_key, 0.0)) if valence_key else 0.0

            somatic_vector = marker.to_vector()
            now_iso = utc_now().isoformat()
            marker_id = f"sm_adj_{new_id()}"

            query = """
            CREATE (sm:SomaticMarker {
                marker_id:         $marker_id,
                adjustment_type:   $adjustment_type,
                valence:           $valence,
                body_system:       $body_system,
                allostatic_context: $allostatic_context,
                prediction_error:  $prediction_error,
                somatic_vector:    $somatic_vector,
                urgency_at_mark:   $urgency,
                dominant_error:    $dominant_error,
                created_at:        $created_at,
                ingestion_time:    $created_at
            })
            RETURN sm.marker_id AS marker_id
            """

            params = {
                "marker_id": marker_id,
                "adjustment_type": adjustment_type,
                "valence": valence,
                "body_system": body_system,
                "allostatic_context": marker.allostatic_context,
                "prediction_error": marker.prediction_error_at_encoding,
                "somatic_vector": somatic_vector,
                "urgency": round(signal.urgency, 4),
                "dominant_error": signal.dominant_error.value if signal.dominant_error else "",
                "created_at": now_iso,
            }

            async with self._driver.session() as session:
                result = await session.run(query, **params)
                record = await result.single()
                if record:
                    self._write_count += 1
                    self._log.debug(
                        "somatic_marker_adjustment_written",
                        marker_id=record["marker_id"],
                        adjustment_type=adjustment_type,
                    )
                    return True

        except Exception as exc:
            self._error_count += 1
            self._log.warning(
                "somatic_marker_adjustment_write_error",
                adjustment_type=adjustment_type,
                error=str(exc),
            )
        return False

    @property
    def write_count(self) -> int:
        return self._write_count

    @property
    def error_count(self) -> int:
        return self._error_count
