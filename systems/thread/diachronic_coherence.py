"""
EcodiaOS — Thread Diachronic Coherence Monitor

Detects whether the organism's identity is drifting or growing.
Uses 29-dimensional behavioural fingerprints and Wasserstein distance
to measure change between epochs.

The genuine innovation: the same behavioural change can be "growth" or
"drift" depending on narrative context. A 0.3 W-distance is "growth" if
it aligns with active schemas and recent turning points, but "drift" if
it contradicts commitments with no narrative explanation.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now
from systems.thread.types import (
    BehavioralFingerprint,
    DriftClassification,
    ThreadConfig,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()


def wasserstein_identity_distance(
    current: BehavioralFingerprint,
    baseline: BehavioralFingerprint,
    config: ThreadConfig,
) -> float:
    """
    Weighted L1 distance between 29D feature vectors, normalized per dimension.

    Personality and drive dimensions are weighted more heavily than
    goal/interaction dimensions because personality drift is more
    identity-relevant than behavioural adaptation.

    Returns: 0.0 (identical) to 1.0+ (substantial drift).
    """
    segments: dict[str, tuple[int, int, float]] = {
        "personality": (0, 9, config.fingerprint_weight_personality),
        "drive_alignment": (9, 13, config.fingerprint_weight_drive),
        "affect": (13, 19, config.fingerprint_weight_affect),
        "goal_source": (19, 25, config.fingerprint_weight_goal),
        "interaction": (25, 29, config.fingerprint_weight_interaction),
    }

    vec_current = current.feature_vector
    vec_baseline = baseline.feature_vector

    # Ensure both vectors are the correct length (pad with zeros if needed)
    target_len = 29
    while len(vec_current) < target_len:
        vec_current.append(0.0)
    while len(vec_baseline) < target_len:
        vec_baseline.append(0.0)

    distance = 0.0
    for _segment_name, (start, end, weight) in segments.items():
        seg_c = vec_current[start:end]
        seg_b = vec_baseline[start:end]
        if len(seg_c) > 0:
            seg_dist = sum(abs(c - b) for c, b in zip(seg_c, seg_b, strict=True)) / len(seg_c)
            distance += weight * seg_dist

    return distance


class DiachronicCoherenceMonitor:
    """
    Monitors identity coherence over time via behavioural fingerprints.

    Computes 29D fingerprints at regular intervals and uses Wasserstein
    distance to detect and classify change as stable/growth/transition/drift.
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        llm: LLMProvider,
        config: ThreadConfig,
    ) -> None:
        self._neo4j = neo4j
        self._llm = llm
        self._config = config
        self._logger = logger.bind(system="thread.diachronic_coherence")

        # Fingerprint history (in-memory ring buffer)
        self._fingerprints: list[BehavioralFingerprint] = []
        self._max_fingerprints = 100  # Keep last 100 in memory

    @property
    def latest_fingerprint(self) -> BehavioralFingerprint | None:
        return self._fingerprints[-1] if self._fingerprints else None

    @property
    def baseline_fingerprint(self) -> BehavioralFingerprint | None:
        """The earliest fingerprint in the buffer — the comparison baseline."""
        return self._fingerprints[0] if self._fingerprints else None

    async def compute_fingerprint(
        self,
        personality_centroid: list[float],
        drive_alignment_centroid: list[float],
        goal_source_distribution: list[float],
        affect_centroid: list[float],
        interaction_style_distribution: list[float],
        episodes_in_window: int = 0,
        epoch_label: str = "",
    ) -> BehavioralFingerprint:
        """
        Compute a 29-dimensional distributional snapshot of recent behaviour.

        The caller (ThreadService) is responsible for aggregating data from
        Voxis, Equor, Nova, Atune to provide the centroid vectors.
        """
        fp = BehavioralFingerprint(
            epoch_label=epoch_label or f"fp_{len(self._fingerprints)}",
            window_start=utc_now(),
            window_end=utc_now(),
            personality_centroid=personality_centroid,
            drive_alignment_centroid=drive_alignment_centroid,
            goal_source_distribution=goal_source_distribution,
            affect_centroid=affect_centroid,
            interaction_style_distribution=interaction_style_distribution,
            episodes_in_window=episodes_in_window,
        )

        self._fingerprints.append(fp)
        if len(self._fingerprints) > self._max_fingerprints:
            self._fingerprints = self._fingerprints[-self._max_fingerprints:]

        # Persist to Neo4j
        await self._persist_fingerprint(fp)

        self._logger.debug(
            "fingerprint_computed",
            fingerprint_id=fp.id,
            epoch=fp.epoch_label,
            episodes=episodes_in_window,
        )
        return fp

    async def assess_change(
        self,
        current_fp: BehavioralFingerprint | None = None,
        baseline_fp: BehavioralFingerprint | None = None,
    ) -> tuple[float, DriftClassification]:
        """
        Compute W-distance between current and baseline fingerprints
        and classify the change.

        Returns: (distance, classification)
        """
        current = current_fp or self.latest_fingerprint
        baseline = baseline_fp or self.baseline_fingerprint

        if current is None or baseline is None:
            return (0.0, DriftClassification.STABLE)

        # Compute Wasserstein distance
        distance = wasserstein_identity_distance(current, baseline, self._config)

        # Classify the change
        classification = await self._classify_change(distance, current, baseline)

        self._logger.info(
            "change_assessed",
            distance=round(distance, 4),
            classification=classification.value,
        )
        return (distance, classification)

    async def _classify_change(
        self,
        distance: float,
        current_fp: BehavioralFingerprint,
        baseline_fp: BehavioralFingerprint,
    ) -> DriftClassification:
        """
        Classify behavioural change as stable/growth/transition/drift.

        This is the novel contribution: narrative-contextualized drift assessment.
        """
        cfg = self._config

        # 1. Stable: negligible change
        if distance < cfg.wasserstein_stable_threshold:
            return DriftClassification.STABLE

        # 2. Check schema alignment
        schema_aligned = await self._check_schema_alignment(current_fp, baseline_fp)

        # 3. Check turning point context
        turning_point_explains = await self._check_turning_point_context(current_fp.window_start)

        # 4. Check commitment consistency
        commitment_violated = await self._check_commitment_consistency(current_fp, baseline_fp)

        if schema_aligned and not commitment_violated:
            return DriftClassification.GROWTH

        if turning_point_explains:
            return DriftClassification.TRANSITION

        if commitment_violated:
            return DriftClassification.DRIFT

        # Ambiguous: default to transition for moderate changes, drift for major
        if distance >= cfg.wasserstein_major_threshold:
            return DriftClassification.DRIFT
        return DriftClassification.TRANSITION

    async def _check_schema_alignment(
        self,
        current_fp: BehavioralFingerprint,
        baseline_fp: BehavioralFingerprint,
    ) -> bool:
        """
        Check if the direction of change aligns with active schemas.
        Returns True if the change is consistent with who the organism
        believes itself to be.
        """
        try:
            results = await self._neo4j.execute_read(
                """
                MATCH (s:Self)-[:HAS_SCHEMA]->(schema:IdentitySchema)
                WHERE schema.strength IN ['established', 'core']
                RETURN schema.statement AS statement, schema.evidence_ratio AS ratio
                LIMIT 10
                """,
                {},
            )
            # If there are strong schemas and no major contradictions,
            # consider it aligned
            strong_schemas = [r for r in results if float(r.get("ratio", 0)) > 0.7]
            return len(strong_schemas) >= 2
        except Exception:
            return False

    async def _check_turning_point_context(self, window_start: Any) -> bool:
        """Check if a recent TurningPoint explains the change."""
        try:
            results = await self._neo4j.execute_read(
                """
                MATCH (tp:TurningPoint)
                WHERE tp.timestamp >= datetime($since)
                RETURN count(tp) AS count
                """,
                {
                    "since": window_start.isoformat()
                    if hasattr(window_start, "isoformat")
                    else str(window_start),
                },
            )
            for r in results:
                return int(r.get("count", 0)) > 0
            return False
        except Exception:
            return False

    async def _check_commitment_consistency(
        self,
        current_fp: BehavioralFingerprint,
        baseline_fp: BehavioralFingerprint,
    ) -> bool:
        """Check if the change violates active commitments."""
        try:
            results = await self._neo4j.execute_read(
                """
                MATCH (s:Self)-[:HOLDS_COMMITMENT]->(c:Commitment)
                WHERE c.status = 'active'
                RETURN c.fidelity AS fidelity
                """,
                {},
            )
            # If any active commitment has low fidelity, flag violation
            for r in results:
                fidelity = float(r.get("fidelity", 1.0))
                if fidelity < self._config.commitment_strain_threshold:
                    return True
            return False
        except Exception:
            return False

    async def _persist_fingerprint(self, fp: BehavioralFingerprint) -> None:
        """Store a fingerprint node in Neo4j, linked to Self."""
        try:
            await self._neo4j.execute_write(
                """
                MATCH (s:Self)
                CREATE (f:BehavioralFingerprint {
                    id: $id,
                    epoch_label: $epoch_label,
                    window_start: datetime($window_start),
                    window_end: datetime($window_end),
                    personality_centroid_json: $personality_centroid_json,
                    drive_alignment_centroid_json: $drive_alignment_centroid_json,
                    goal_source_distribution_json: $goal_source_distribution_json,
                    affect_centroid_json: $affect_centroid_json,
                    interaction_style_distribution_json: $interaction_style_distribution_json,
                    episodes_in_window: $episodes_in_window,
                    mean_surprise: $mean_surprise,
                    mean_coherence: $mean_coherence
                })
                CREATE (f)-[:SNAPSHOT_OF]->(s)
                """,
                {
                    "id": fp.id,
                    "epoch_label": fp.epoch_label,
                    "window_start": fp.window_start.isoformat(),
                    "window_end": fp.window_end.isoformat(),
                    "personality_centroid_json": json.dumps(fp.personality_centroid),
                    "drive_alignment_centroid_json": json.dumps(fp.drive_alignment_centroid),
                    "goal_source_distribution_json": json.dumps(fp.goal_source_distribution),
                    "affect_centroid_json": json.dumps(fp.affect_centroid),
                    "interaction_style_distribution_json": json.dumps(
                        fp.interaction_style_distribution
                    ),
                    "episodes_in_window": fp.episodes_in_window,
                    "mean_surprise": fp.mean_surprise,
                    "mean_coherence": fp.mean_coherence,
                },
            )

            # Link to previous fingerprint
            if len(self._fingerprints) >= 2:
                prev = self._fingerprints[-2]
                await self._neo4j.execute_write(
                    """
                    MATCH (curr:BehavioralFingerprint {id: $curr_id})
                    MATCH (prev:BehavioralFingerprint {id: $prev_id})
                    MERGE (curr)-[:PRECEDED_BY]->(prev)
                    """,
                    {"curr_id": fp.id, "prev_id": prev.id},
                )
        except Exception as exc:
            self._logger.warning("fingerprint_persist_failed", error=str(exc))
