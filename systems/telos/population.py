"""
EcodiaOS - Telos: Population-Level Intelligence Aggregator (M3)

Subscribes to CHILD_HEALTH_REPORT events and aggregates drive alignment scores
across the fleet to compute population-level effective intelligence.

Drive weight diversity IS the speciation signal. If fleet instances cluster
into distinct drive phenotypes (Growth-heavy vs Care-heavy), the organism is
differentiating into proto-species. This module detects and reports that signal.

Population I = mean(individual_I) + variance_bonus
variance_bonus = k * variance(I) where k=0.25 (diverse phenotypes → higher
collective intelligence because they explore more of the drive topology space)

Speciation signal = mean inter-cluster Euclidean distance in 4D drive space,
normalised to [0, 1]. Above ~0.3 indicates meaningful phenotype divergence.
"""

from __future__ import annotations

import math
from collections import deque
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from systems.telos.types import (
    ConstitutionalPhenotypeCluster,
    DriveWeightDistribution,
    DriveWeightStats,
    PopulationIntelligenceSnapshot,
)

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()

# Diversity bonus coefficient: how much variance in I adds to collective I.
# 0.25 chosen so a perfectly uniform fleet (variance=0) loses nothing, while
# a highly diverse fleet (variance=0.1) gains ~2.5% collective intelligence.
_VARIANCE_BONUS_K: float = 0.25

# Maximum history entries per child instance (rolling window ~ 10 minutes
# at the 60s CHILD_HEALTH_REPORT interval child-side).
_MAX_CHILD_HISTORY: int = 10

# Minimum instances required before emitting a population snapshot.
# Set to 1 so a solo genesis instance still produces snapshot telemetry.
# Population-level variance/speciation statistics simply reflect N=1 state.
_MIN_INSTANCES_FOR_SNAPSHOT: int = 1

# Clustering: number of passes for the simple k-means-lite phenotype detector.
_CLUSTER_PASSES: int = 3

# Drives in canonical order for vector arithmetic.
_DRIVES: tuple[str, ...] = ("care", "coherence", "growth", "honesty")


class _ChildRecord:
    """Rolling window of drive scores and I for a single child instance."""

    __slots__ = ("instance_id", "_history")

    def __init__(self, instance_id: str) -> None:
        self.instance_id = instance_id
        self._history: deque[dict[str, float]] = deque(maxlen=_MAX_CHILD_HISTORY)

    def update(self, care: float, coherence: float, growth: float, honesty: float, effective_I: float) -> None:
        self._history.append({
            "care": max(0.0, min(1.0, care)),
            "coherence": max(0.0, min(1.0, coherence)),
            "growth": max(-1.0, min(1.0, growth)),
            "honesty": max(0.0, min(1.0, honesty)),
            "effective_I": max(0.0, effective_I),
        })

    @property
    def latest(self) -> dict[str, float] | None:
        return self._history[-1] if self._history else None


class PopulationIntelligenceAggregator:
    """
    Aggregates drive alignment data from the fleet and computes population I.

    Fed by CHILD_HEALTH_REPORT events. Queried every 60s by TelosService to
    produce a PopulationIntelligenceSnapshot for Benchmarks.

    The self-instance (the parent EOS) is always included in the population
    using the most recent EffectiveIntelligenceReport from TelosService.
    """

    def __init__(self) -> None:
        self._children: dict[str, _ChildRecord] = {}
        self._logger = logger.bind(component="telos.population")

    def ingest_child_health_report(self, event_data: dict) -> None:
        """
        Ingest a CHILD_HEALTH_REPORT event payload.

        Expected fields (all optional; missing fields default to 0.5):
          child_instance_id (str)
          drive_care (float)       - child's care multiplier [0, 1]
          drive_coherence (float)  - child's coherence multiplier [0, 1]
          drive_growth (float)     - child's growth score [-1, 1]
          drive_honesty (float)    - child's honesty coefficient [0, 1]
          effective_I (float)      - child's effective intelligence
        """
        child_id = str(event_data.get("child_instance_id", ""))
        if not child_id:
            return

        care = float(event_data.get("drive_care", 0.5))
        coherence = float(event_data.get("drive_coherence", 0.5))
        growth = float(event_data.get("drive_growth", 0.0))
        honesty = float(event_data.get("drive_honesty", 0.5))
        effective_I = float(event_data.get("effective_I", 0.0))

        if child_id not in self._children:
            self._children[child_id] = _ChildRecord(child_id)
            self._logger.debug("population_child_registered", child_id=child_id)

        self._children[child_id].update(care, coherence, growth, honesty, effective_I)

    def remove_child(self, child_id: str) -> None:
        """Remove a child from the population (called on CHILD_DIED)."""
        self._children.pop(child_id, None)

    def compute_snapshot(
        self,
        self_care: float,
        self_coherence: float,
        self_growth: float,
        self_honesty: float,
        self_effective_I: float,
    ) -> PopulationIntelligenceSnapshot | None:
        """
        Compute a population-level intelligence snapshot.

        Always includes the self-instance (parent EOS) alongside children.
        Returns None if fewer than _MIN_INSTANCES_FOR_SNAPSHOT instances exist.
        """
        # Collect current drive vectors for all instances
        instances: list[dict[str, float]] = []

        # Self-instance always included
        instances.append({
            "care": max(0.0, min(1.0, self_care)),
            "coherence": max(0.0, min(1.0, self_coherence)),
            "growth": max(-1.0, min(1.0, self_growth)),
            "honesty": max(0.0, min(1.0, self_honesty)),
            "effective_I": max(0.0, self_effective_I),
        })

        # Children with recent data
        for record in self._children.values():
            latest = record.latest
            if latest is not None:
                instances.append(latest)

        if len(instances) < _MIN_INSTANCES_FOR_SNAPSHOT:
            return None

        # ── Step 1: mean_I and variance_I ─────────────────────────────
        i_values = [inst["effective_I"] for inst in instances]
        n = len(i_values)
        mean_I = sum(i_values) / n
        variance_I = sum((x - mean_I) ** 2 for x in i_values) / n

        variance_bonus = _VARIANCE_BONUS_K * variance_I
        population_I = mean_I + variance_bonus

        # ── Step 2: drive_weight_distribution ─────────────────────────
        drive_dist = _compute_drive_distribution(instances)

        # ── Step 3: constitutional phenotype clusters ──────────────────
        clusters = _detect_phenotype_clusters(instances)

        # ── Step 4: speciation_signal ──────────────────────────────────
        speciation_signal = _compute_speciation_signal(clusters)

        self._logger.debug(
            "population_snapshot_computed",
            instance_count=n,
            mean_I=round(mean_I, 4),
            variance_I=round(variance_I, 4),
            population_I=round(population_I, 4),
            cluster_count=len(clusters),
            speciation_signal=round(speciation_signal, 3),
        )

        return PopulationIntelligenceSnapshot(
            instance_count=n,
            mean_I=round(mean_I, 6),
            variance_I=round(variance_I, 6),
            population_I=round(population_I, 6),
            variance_bonus=round(variance_bonus, 6),
            drive_weight_distribution=drive_dist,
            constitutional_phenotype_clusters=clusters,
            speciation_signal=round(speciation_signal, 4),
            timestamp=datetime.now(UTC),
        )

    @property
    def child_count(self) -> int:
        return len(self._children)


# ─── Internal helpers ─────────────────────────────────────────────────


def _compute_drive_distribution(instances: list[dict[str, float]]) -> DriveWeightDistribution:
    """Compute mean and std for each drive across all instances."""
    n = len(instances)
    result: dict[str, DriveWeightStats] = {}
    for drive in _DRIVES:
        values = [inst[drive] for inst in instances]
        mean = sum(values) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
        result[drive] = DriveWeightStats(mean=round(mean, 4), std=round(std, 4))

    return DriveWeightDistribution(
        care=result["care"],
        coherence=result["coherence"],
        growth=result["growth"],
        honesty=result["honesty"],
    )


def _detect_phenotype_clusters(
    instances: list[dict[str, float]],
) -> list[ConstitutionalPhenotypeCluster]:
    """
    Detect constitutional phenotype clusters using a lightweight 2-centroid
    k-means pass over the 4D drive-weight space.

    With small fleets (< 4 instances), grouping is less meaningful -
    we return a single "fleet" cluster.

    Growth drive range is [-1, 1] while others are [0, 1], so we normalise
    growth to [0, 1] before clustering, then restore for the centroid output.
    """
    n = len(instances)
    if n < 4:
        # Not enough instances for meaningful clustering
        centroid = {
            drive: sum(inst[drive] for inst in instances) / n
            for drive in _DRIVES
        }
        dominant = max(_DRIVES, key=lambda d: centroid[d])
        return [ConstitutionalPhenotypeCluster(
            label="fleet",
            centroid={d: round(centroid[d], 3) for d in _DRIVES},
            size=n,
            dominant_drive=dominant,
        )]

    # Normalise growth to [0, 1] for distance computation
    normed = [
        {
            "care": inst["care"],
            "coherence": inst["coherence"],
            "growth": (inst["growth"] + 1.0) / 2.0,
            "honesty": inst["honesty"],
        }
        for inst in instances
    ]

    # Seed centroids: instance with lowest and highest care (maximally diverse seed)
    sorted_by_care = sorted(range(n), key=lambda i: normed[i]["care"])
    c0 = {d: normed[sorted_by_care[0]][d] for d in _DRIVES}
    c1 = {d: normed[sorted_by_care[-1]][d] for d in _DRIVES}

    assignments = [0] * n

    for _ in range(_CLUSTER_PASSES):
        # Assign each instance to nearest centroid
        for i, vec in enumerate(normed):
            d0 = _l2(vec, c0)
            d1 = _l2(vec, c1)
            assignments[i] = 0 if d0 <= d1 else 1

        # Update centroids
        for k, centroid in enumerate([c0, c1]):
            members = [normed[i] for i in range(n) if assignments[i] == k]
            if not members:
                continue
            m = len(members)
            for drive in _DRIVES:
                centroid[drive] = sum(mem[drive] for mem in members) / m

    # Build cluster objects (restore growth normalisation for output)
    clusters: list[ConstitutionalPhenotypeCluster] = []
    for k, (centroid_normed, label_hint) in enumerate([(c0, "a"), (c1, "b")]):
        members_idx = [i for i in range(n) if assignments[i] == k]
        if not members_idx:
            continue

        # Restore growth from normalised [0,1] to actual [-1,1]
        centroid_out = {
            "care": round(centroid_normed["care"], 3),
            "coherence": round(centroid_normed["coherence"], 3),
            "growth": round(centroid_normed["growth"] * 2.0 - 1.0, 3),
            "honesty": round(centroid_normed["honesty"], 3),
        }

        dominant = max(
            ("care", "coherence", "honesty"),
            key=lambda d: centroid_normed[d],
        )
        # Growth dominant if it exceeds 0.6 normalised (actual > 0.2)
        if centroid_normed["growth"] > 0.6 and centroid_normed["growth"] > centroid_normed[dominant]:
            dominant = "growth"

        label = f"{dominant}_dominant_{label_hint}"

        clusters.append(ConstitutionalPhenotypeCluster(
            label=label,
            centroid=centroid_out,
            size=len(members_idx),
            dominant_drive=dominant,
        ))

    return clusters


def _l2(a: dict[str, float], b: dict[str, float]) -> float:
    """Euclidean distance between two drive-weight vectors."""
    return math.sqrt(sum((a[d] - b[d]) ** 2 for d in _DRIVES))


def _compute_speciation_signal(clusters: list[ConstitutionalPhenotypeCluster]) -> float:
    """
    Compute speciation signal as mean inter-cluster distance in 4D drive space,
    normalised to [0, 1].

    The maximum possible L2 distance in [0,1]^4 is sqrt(4) ≈ 2.0.
    We normalise by that.

    Single cluster → 0.0.  Two well-separated clusters → approaches 1.0.
    """
    if len(clusters) < 2:
        return 0.0

    total_dist = 0.0
    pair_count = 0
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            ca = clusters[i].centroid
            cb = clusters[j].centroid
            # Use normalised growth for distance (already stored in [-1,1] in centroid)
            ca_n = {d: ca[d] if d != "growth" else (ca[d] + 1.0) / 2.0 for d in _DRIVES}
            cb_n = {d: cb[d] if d != "growth" else (cb[d] + 1.0) / 2.0 for d in _DRIVES}
            dist = _l2(ca_n, cb_n)
            total_dist += dist
            pair_count += 1

    if pair_count == 0:
        return 0.0

    mean_dist = total_dist / pair_count
    max_possible = math.sqrt(len(_DRIVES))  # sqrt(4) ≈ 2.0
    return min(1.0, mean_dist / max_possible)
