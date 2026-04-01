"""
EcodiaOS - Evolutionary Tracker

Subscribes to EVOLUTIONARY_OBSERVABLE events and computes Bedau-Packard
population-level statistics. Persists observable history to Redis for
restart survival.

Reference: Bedau & Packard (1992) - "Measurement of Evolutionary Activity,
Teleology, and Life"
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID, utc_now
from primitives.evolutionary import (
    BedauPackardStats,
    EvolutionaryActivity,
    EvolutionaryObservable,
)
from systems.mitosis.genome_distance import GenomeDistanceCalculator
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.benchmarks.evolutionary_tracker")

_REDIS_KEY = "eos:benchmarks:evolutionary_observables:{instance_id}"
_MAX_OBSERVABLES = 10_000


class EvolutionaryTracker:
    """
    Collects EvolutionaryObservable events from all systems and computes
    Bedau-Packard population statistics.

    Lifecycle:
      tracker = EvolutionaryTracker(instance_id, generation, parent_instance_id)
      tracker.attach(event_bus)        # subscribes to EVOLUTIONARY_OBSERVABLE
      activity = await tracker.compute_activity()
      stats = await tracker.compute_bedau_packard([activity])
      snapshot = await tracker.snapshot()  # emits BEDAU_PACKARD_SNAPSHOT
    """

    def __init__(
        self,
        instance_id: str,
        generation: int = 1,
        parent_instance_id: str | None = None,
        redis: Any | None = None,
        speciation_threshold: float = 0.3,
    ) -> None:
        self._instance_id = instance_id
        self._generation = generation
        self._parent_instance_id = parent_instance_id
        self._distance_calc = GenomeDistanceCalculator(
            speciation_threshold=speciation_threshold,
        )
        self._redis = redis

        self._observables: list[EvolutionaryObservable] = []
        self._novel_count: int = 0
        self._total_count: int = 0
        self._event_bus: EventBus | None = None

    def attach(self, event_bus: EventBus) -> None:
        """Subscribe to evolutionary observable events."""
        self._event_bus = event_bus
        event_bus.subscribe(
            SynapseEventType.EVOLUTIONARY_OBSERVABLE,
            self.on_evolutionary_observable,
        )

    async def on_evolutionary_observable(self, event: SynapseEvent) -> None:
        """Collect an observable as it arrives via Synapse."""
        try:
            obs = EvolutionaryObservable(**event.data)
        except Exception:
            logger.warning(
                "evolutionary_observable_parse_error",
                data=event.data,
            )
            return

        self._observables.append(obs)
        self._total_count += 1
        if obs.is_novel:
            self._novel_count += 1

        # Cap in-memory list
        if len(self._observables) > _MAX_OBSERVABLES:
            self._observables = self._observables[-_MAX_OBSERVABLES:]

        # Persist to Redis (fire-and-forget)
        await self._persist_to_redis(obs)

    async def compute_activity(self) -> EvolutionaryActivity:
        """Compute this instance's evolutionary activity record."""
        return EvolutionaryActivity(
            instance_id=self._instance_id,
            generation=self._generation,
            novel_mutations=self._novel_count,
            total_mutations=self._total_count,
            fitness=self._novel_count / max(self._total_count, 1),
            parent_instance_id=self._parent_instance_id,
        )

    async def compute_bedau_packard(
        self, population: list[EvolutionaryActivity]
    ) -> BedauPackardStats:
        """
        Compute population-level Bedau-Packard statistics.

        For single-instance (no children), computes self-only stats.
        When children exist, population includes all living instances.
        """
        if not population:
            return BedauPackardStats()

        # Total activity: sum of novel mutations still active
        total_activity = float(sum(p.novel_mutations for p in population))

        # Mean activity
        mean_activity = total_activity / len(population)

        # Diversity index: Shannon entropy of observable_type distribution
        diversity_index = self._shannon_entropy()

        # Evolutionary rate: novel observables per generation
        evolutionary_rate = (
            self._novel_count / max(self._generation, 1)
        )

        # Persistence: fraction of parent's observables that survive in child
        # For genesis instances (no parent), persistence = 1.0
        persistence = await self._compute_persistence()

        return BedauPackardStats(
            total_activity=total_activity,
            mean_activity=round(mean_activity, 4),
            diversity_index=round(diversity_index, 4),
            evolutionary_rate=round(evolutionary_rate, 4),
            persistence=persistence,
        )

    async def snapshot(self) -> BedauPackardStats:
        """Compute stats and emit BEDAU_PACKARD_SNAPSHOT via Synapse."""
        activity = await self.compute_activity()
        stats = await self.compute_bedau_packard([activity])

        if self._event_bus is not None:
            event = SynapseEvent(
                event_type=SynapseEventType.BEDAU_PACKARD_SNAPSHOT,
                source_system=SystemID.BENCHMARKS,
                data=stats.model_dump(mode="json"),
            )
            await self._event_bus.emit(event)

        return stats

    async def _compute_persistence(self) -> float:
        """
        Compute persistence: fraction of parent's observable types that survive in child.

        If no parent (genesis instance): persistence = 1.0.
        If parent observables can't be retrieved: persistence = 0.0 (conservative).
        """
        if self._parent_instance_id is None:
            return 1.0

        if self._redis is None:
            return 0.0

        try:
            parent_key = _REDIS_KEY.format(instance_id=self._parent_instance_id)
            parent_raw = await self._redis.lrange(parent_key, 0, -1)
            if not parent_raw:
                return 0.0

            # Collect parent's unique observable types
            parent_types: set[str] = set()
            for raw in parent_raw:
                try:
                    obs = EvolutionaryObservable.model_validate_json(raw)
                    parent_types.add(obs.observable_type)
                except Exception:
                    continue

            if not parent_types:
                return 0.0

            # Collect child's (our) unique observable types
            child_types = {obs.observable_type for obs in self._observables}

            # Persistence = fraction of parent types that appear in child
            surviving = parent_types & child_types
            return round(len(surviving) / len(parent_types), 4)
        except Exception:
            logger.debug("persistence_computation_failed")
            return 0.0

    def _shannon_entropy(self) -> float:
        """Shannon entropy of observable_type distribution across collected observables."""
        if not self._observables:
            return 0.0

        counts = Counter(obs.observable_type for obs in self._observables)
        total = sum(counts.values())
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)
        return entropy

    async def _persist_to_redis(self, obs: EvolutionaryObservable) -> None:
        """Append observable to Redis list for restart survival."""
        if self._redis is None:
            return
        try:
            key = _REDIS_KEY.format(instance_id=self._instance_id)
            await self._redis.rpush(key, obs.model_dump_json())
            # Trim to max length
            await self._redis.ltrim(key, -_MAX_OBSERVABLES, -1)
        except Exception:
            logger.debug("evolutionary_tracker_redis_persist_failed")

    async def restore_from_redis(self) -> None:
        """Restore observable history from Redis on startup."""
        if self._redis is None:
            return
        try:
            key = _REDIS_KEY.format(instance_id=self._instance_id)
            raw_list = await self._redis.lrange(key, 0, -1)
            for raw in raw_list:
                obs = EvolutionaryObservable.model_validate_json(raw)
                self._observables.append(obs)
                self._total_count += 1
                if obs.is_novel:
                    self._novel_count += 1
            if self._observables:
                logger.info(
                    "evolutionary_tracker_restored",
                    count=len(self._observables),
                    novel=self._novel_count,
                )
        except Exception:
            logger.debug("evolutionary_tracker_redis_restore_failed")

    async def detect_speciation_events(
        self, fleet_genomes: list[dict]
    ) -> list[dict]:
        """
        Detect speciation: clusters of instances with genome distance > threshold.

        Uses single-linkage agglomerative clustering on the pairwise distance
        matrix. If the fleet diverges into > 1 cluster, emits SPECIATION_DETECTED
        via Synapse.

        Parameters
        ----------
        fleet_genomes : list[dict]
            Each dict must have keys: "instance_id" (str) and "genome" (dict with
            optional sub-keys "evo", "simula", "telos", "equor").

        Returns
        -------
        list[dict]
            Detected species clusters, each with:
            {
              "cluster_id": int,
              "instance_ids": list[str],
              "size": int,
              "mean_intra_distance": float,
            }
        """
        if len(fleet_genomes) < 2:
            return []

        n = len(fleet_genomes)
        instance_ids = [g["instance_id"] for g in fleet_genomes]
        genomes = [g.get("genome") or {} for g in fleet_genomes]

        # -- Compute pairwise distance matrix (upper triangle) ---------------
        distances: list[list[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = self._distance_calc.compute(genomes[i], genomes[j]).total_distance
                distances[i][j] = d
                distances[j][i] = d

        # -- Single-linkage agglomerative clustering -------------------------
        # Each instance starts in its own cluster.
        cluster_of: list[int] = list(range(n))

        def _merge(ci: int, cj: int) -> None:
            for k in range(n):
                if cluster_of[k] == cj:
                    cluster_of[k] = ci

        threshold = self._distance_calc._threshold
        for i in range(n):
            for j in range(i + 1, n):
                if distances[i][j] <= threshold:
                    # Merge clusters - single-linkage: any link below threshold joins them
                    ci, cj = cluster_of[i], cluster_of[j]
                    if ci != cj:
                        _merge(min(ci, cj), max(ci, cj))

        # -- Build cluster list ----------------------------------------------
        cluster_members: dict[int, list[int]] = {}
        for idx, cid in enumerate(cluster_of):
            cluster_members.setdefault(cid, []).append(idx)

        clusters: list[dict] = []
        for rank, (_, members) in enumerate(sorted(cluster_members.items())):
            # Mean intra-cluster distance
            intra: list[float] = []
            for i in members:
                for j in members:
                    if i < j:
                        intra.append(distances[i][j])
            mean_intra = round(sum(intra) / len(intra), 6) if intra else 0.0
            clusters.append({
                "cluster_id": rank,
                "instance_ids": [instance_ids[i] for i in members],
                "size": len(members),
                "mean_intra_distance": mean_intra,
            })

        species_count = len(clusters)

        if species_count > 1:
            # Mean distance between cluster centroids (inter-cluster)
            inter_distances: list[float] = []
            for ci, c1 in enumerate(clusters):
                for cj, c2 in enumerate(clusters):
                    if ci < cj:
                        for i_id in c1["instance_ids"]:
                            for j_id in c2["instance_ids"]:
                                i_idx = instance_ids.index(i_id)
                                j_idx = instance_ids.index(j_id)
                                inter_distances.append(distances[i_idx][j_idx])
            mean_inter = (
                round(sum(inter_distances) / len(inter_distances), 6)
                if inter_distances else 0.0
            )

            logger.info(
                "speciation_detected",
                species_count=species_count,
                mean_inter_distance=mean_inter,
                fleet_size=n,
            )

            if self._event_bus is not None:
                try:
                    event = SynapseEvent(
                        event_type=SynapseEventType.SPECIATION_DETECTED,
                        source_system=SystemID.BENCHMARKS,
                        data={
                            "species_count": species_count,
                            "clusters": [
                                {
                                    "cluster_id": c["cluster_id"],
                                    "instance_ids": c["instance_ids"],
                                    "size": c["size"],
                                }
                                for c in clusters
                            ],
                            "mean_inter_distance": mean_inter,
                            "threshold": threshold,
                            "fleet_size": n,
                            "instance_id": self._instance_id,
                        },
                    )
                    await self._event_bus.emit(event)
                except Exception as exc:
                    logger.warning(
                        "speciation_detected_emit_failed", error=str(exc)
                    )

        return clusters

    @property
    def stats(self) -> dict[str, Any]:
        """Summary stats for health/observability."""
        return {
            "total_observables": self._total_count,
            "novel_observables": self._novel_count,
            "novelty_rate": round(
                self._novel_count / max(self._total_count, 1), 4
            ),
            "diversity_index": round(self._shannon_entropy(), 4),
            "generation": self._generation,
        }
