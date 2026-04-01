"""
EcodiaOS - Shadow-Reset Controller

Implements the bible §6.2 shadow-reset control for measuring genuine adaptive
dynamics versus random drift in the population's evolutionary behaviour.

The shadow-reset is NON-DESTRUCTIVE.  It snapshots the current observable
distribution into Redis and later compares the live distribution against that
snapshot.  A dramatic activity drop post-"reset" proves that the dynamics
are genuinely adaptive (organisms are reacting to the change), not statistical
drift (which would be unaffected by a population-state perturbation).

Key concepts
────────────
• Snapshot - saved population state keyed by timestamp.
• Delta - comparison of current state vs. snapshot: activity_drop_pct,
  diversity_recovery_time, is_adaptive flag.
• "is_adaptive" = True when activity_drop_pct > 50 %, meaning the population's
  novel-mutation rate was tied to historical state and has not yet recovered.

See: Bedau & Packard (1992), §6.4 of the speciation bible.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.benchmarks.shadow_reset")

# ─── Redis key schema ─────────────────────────────────────────────────────────
_SNAPSHOT_KEY = "eos:benchmarks:shadow_snapshot:{snapshot_id}"
_SNAPSHOT_INDEX_KEY = "eos:benchmarks:shadow_snapshots:{instance_id}"
_SNAPSHOT_TTL_S = 60 * 60 * 24 * 30  # 30 days


@dataclass
class ObservableFrequency:
    """Frequency record for a single observable type in the population."""

    observable_type: str
    count: int
    fraction: float
    # Mean age in cycles (0 if age data unavailable)
    mean_age_cycles: float = 0.0


@dataclass
class ShadowSnapshot:
    """
    Saved population state at a point in time.

    Captured fields
    ───────────────
    • observable_types: all distinct observable types + their counts
    • total_count: total observables seen at snapshot time
    • novel_count: novel observables seen at snapshot time
    • novelty_rate: novel / total at snapshot time
    • diversity_index: Shannon entropy of observable type distribution
    • snapshot_id: opaque identifier used to retrieve this snapshot
    • taken_at_iso: UTC ISO-8601 timestamp
    • instance_id: which organism took this snapshot
    """

    snapshot_id: str
    instance_id: str
    taken_at_iso: str
    total_count: int
    novel_count: int
    novelty_rate: float
    diversity_index: float
    observable_frequencies: list[ObservableFrequency] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "instance_id": self.instance_id,
            "taken_at_iso": self.taken_at_iso,
            "total_count": self.total_count,
            "novel_count": self.novel_count,
            "novelty_rate": self.novelty_rate,
            "diversity_index": self.diversity_index,
            "observable_frequencies": [
                {
                    "observable_type": f.observable_type,
                    "count": f.count,
                    "fraction": f.fraction,
                    "mean_age_cycles": f.mean_age_cycles,
                }
                for f in self.observable_frequencies
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShadowSnapshot":
        freqs = [
            ObservableFrequency(
                observable_type=f["observable_type"],
                count=f["count"],
                fraction=f["fraction"],
                mean_age_cycles=f.get("mean_age_cycles", 0.0),
            )
            for f in d.get("observable_frequencies", [])
        ]
        return cls(
            snapshot_id=d["snapshot_id"],
            instance_id=d["instance_id"],
            taken_at_iso=d["taken_at_iso"],
            total_count=d["total_count"],
            novel_count=d["novel_count"],
            novelty_rate=d["novelty_rate"],
            diversity_index=d["diversity_index"],
            observable_frequencies=freqs,
        )


@dataclass
class ShadowResetResult:
    """
    Comparison of current population state vs. a historical snapshot.

    Fields
    ──────
    snapshot_id:
        The snapshot this delta was computed against.
    activity_drop_pct:
        How much adaptive activity has dropped since the snapshot, as a
        percentage of the snapshot's novelty_rate.
        Positive = activity declined (expected post-reset in adaptive system).
        Negative = activity increased.
    diversity_change_pct:
        Change in Shannon entropy (diversity_index) as a percentage of the
        snapshot's diversity_index.  Positive = more diverse, negative = less.
    jaccard_overlap:
        Jaccard similarity between the observable type sets at snapshot time
        and now.  Low overlap = the population has generated many new types.
    is_adaptive:
        True when activity_drop_pct > 50 %.  A dramatic drop confirms the
        dynamics were tied to historical state and are genuinely adaptive.
        A near-zero drop suggests dynamics are drift, not adaptive.
    elapsed_seconds:
        Seconds between snapshot and this delta computation.
    diversity_recovery_time:
        Rough estimate of how long diversity took to recover (seconds).
        Set to None when diversity has not yet recovered past snapshot level.
    current_novelty_rate:
        Live novelty rate at comparison time.
    snapshot_novelty_rate:
        Novelty rate captured in the snapshot.
    """

    snapshot_id: str
    activity_drop_pct: float
    diversity_change_pct: float
    jaccard_overlap: float
    is_adaptive: bool
    elapsed_seconds: float
    diversity_recovery_time: float | None
    current_novelty_rate: float
    snapshot_novelty_rate: float
    computed_at_iso: str


class ShadowResetController:
    """
    Non-destructive shadow-reset controller.

    Usage
    ─────
      ctrl = ShadowResetController(instance_id="genesis-001", redis=redis)
      snapshot_id = await ctrl.take_shadow_snapshot()
      ...later...
      result = await ctrl.compute_shadow_delta(snapshot_id)
      print(result.is_adaptive)

    The controller reads current population state from the EvolutionaryTracker
    that is injected at construction time.  If the tracker is not available,
    the snapshot captures zeros (safe, non-fatal).
    """

    def __init__(
        self,
        instance_id: str,
        redis: Any | None = None,
        tracker: Any | None = None,
    ) -> None:
        self._instance_id = instance_id
        self._redis = redis
        self._tracker = tracker  # EvolutionaryTracker (duck-typed)
        # In-memory index of snapshots taken this session
        self._session_snapshots: dict[str, ShadowSnapshot] = {}

    def set_tracker(self, tracker: Any) -> None:
        """Inject EvolutionaryTracker after construction."""
        self._tracker = tracker

    def set_redis(self, redis: Any) -> None:
        self._redis = redis

    # ─── Public API ───────────────────────────────────────────────────────────

    async def take_shadow_snapshot(self) -> str:
        """
        Capture the current population state and persist it to Redis.

        Returns
        ───────
        snapshot_id: str  - opaque ID used to retrieve this snapshot later.
        """
        snapshot_id = new_id()
        now_iso = utc_now().isoformat()

        # Read current population state from the tracker
        total_count, novel_count, novelty_rate, diversity_index, frequencies = (
            await self._read_tracker_state()
        )

        snapshot = ShadowSnapshot(
            snapshot_id=snapshot_id,
            instance_id=self._instance_id,
            taken_at_iso=now_iso,
            total_count=total_count,
            novel_count=novel_count,
            novelty_rate=novelty_rate,
            diversity_index=diversity_index,
            observable_frequencies=frequencies,
        )

        # Persist to Redis + keep in-memory index
        await self._persist_snapshot(snapshot)
        self._session_snapshots[snapshot_id] = snapshot

        logger.info(
            "shadow_snapshot_taken",
            snapshot_id=snapshot_id,
            total_count=total_count,
            novelty_rate=round(novelty_rate, 4),
            diversity_index=round(diversity_index, 4),
        )
        return snapshot_id

    async def compute_shadow_delta(self, snapshot_id: str) -> ShadowResetResult:
        """
        Compare current population state against a historical snapshot.

        Parameters
        ──────────
        snapshot_id: the ID returned by a previous take_shadow_snapshot() call.

        Returns
        ───────
        ShadowResetResult with activity_drop_pct, diversity_change_pct,
        jaccard_overlap, is_adaptive, and related fields.

        Raises
        ──────
        ValueError if the snapshot_id cannot be found in Redis or session cache.
        """
        snapshot = await self._load_snapshot(snapshot_id)
        if snapshot is None:
            raise ValueError(f"Shadow snapshot {snapshot_id!r} not found")

        now_iso = utc_now().isoformat()
        now_ts = time.time()
        snap_ts = _iso_to_timestamp(snapshot.taken_at_iso)
        elapsed = now_ts - snap_ts

        # Read current state
        (
            _curr_total,
            _curr_novel,
            curr_novelty_rate,
            curr_diversity,
            curr_frequencies,
        ) = await self._read_tracker_state()

        # ── Activity drop ──────────────────────────────────────────────────
        snap_novelty = snapshot.novelty_rate
        if snap_novelty > 0.0:
            activity_drop_pct = (snap_novelty - curr_novelty_rate) / snap_novelty * 100.0
        else:
            activity_drop_pct = 0.0

        # ── Diversity change ───────────────────────────────────────────────
        snap_diversity = snapshot.diversity_index
        if snap_diversity > 0.0:
            diversity_change_pct = (
                (curr_diversity - snap_diversity) / snap_diversity * 100.0
            )
        else:
            diversity_change_pct = 0.0

        # ── Jaccard overlap of observable type sets ────────────────────────
        snap_types = {f.observable_type for f in snapshot.observable_frequencies}
        curr_types = {f.observable_type for f in curr_frequencies}
        if snap_types or curr_types:
            union = snap_types | curr_types
            intersect = snap_types & curr_types
            jaccard_overlap = len(intersect) / len(union)
        else:
            jaccard_overlap = 1.0

        # ── Adaptive flag ──────────────────────────────────────────────────
        # Bible §6.4: >50% activity drop = dynamics are genuinely adaptive
        is_adaptive = activity_drop_pct > 50.0

        # ── Diversity recovery time ────────────────────────────────────────
        # If diversity has recovered past the snapshot level, we estimate
        # recovery time as the elapsed time (conservative upper bound).
        # If diversity has NOT yet recovered, return None.
        diversity_recovery_time: float | None = None
        if curr_diversity >= snap_diversity:
            diversity_recovery_time = elapsed

        logger.info(
            "shadow_delta_computed",
            snapshot_id=snapshot_id,
            activity_drop_pct=round(activity_drop_pct, 2),
            diversity_change_pct=round(diversity_change_pct, 2),
            jaccard_overlap=round(jaccard_overlap, 4),
            is_adaptive=is_adaptive,
            elapsed_seconds=round(elapsed, 1),
        )

        return ShadowResetResult(
            snapshot_id=snapshot_id,
            activity_drop_pct=round(activity_drop_pct, 4),
            diversity_change_pct=round(diversity_change_pct, 4),
            jaccard_overlap=round(jaccard_overlap, 4),
            is_adaptive=is_adaptive,
            elapsed_seconds=round(elapsed, 2),
            diversity_recovery_time=(
                round(diversity_recovery_time, 2)
                if diversity_recovery_time is not None
                else None
            ),
            current_novelty_rate=round(curr_novelty_rate, 4),
            snapshot_novelty_rate=round(snap_novelty, 4),
            computed_at_iso=now_iso,
        )

    async def list_snapshots(self) -> list[str]:
        """Return snapshot IDs stored in Redis for this instance."""
        if self._redis is None:
            return list(self._session_snapshots.keys())
        try:
            key = _SNAPSHOT_INDEX_KEY.format(instance_id=self._instance_id)
            raw = await self._redis.lrange(key, 0, -1)
            return [r.decode() if isinstance(r, bytes) else r for r in raw]
        except Exception:
            return list(self._session_snapshots.keys())

    # ─── Internal helpers ─────────────────────────────────────────────────────

    async def _read_tracker_state(
        self,
    ) -> tuple[int, int, float, float, list[ObservableFrequency]]:
        """Extract current state from the EvolutionaryTracker (duck-typed)."""
        if self._tracker is None:
            return 0, 0, 0.0, 0.0, []

        try:
            stats: dict[str, Any] = self._tracker.stats
            total_count: int = stats.get("total_observables", 0)
            novel_count: int = stats.get("novel_observables", 0)
            novelty_rate: float = stats.get("novelty_rate", 0.0)
            diversity_index: float = stats.get("diversity_index", 0.0)

            # Build per-type frequency list from in-memory observables
            frequencies: list[ObservableFrequency] = []
            observables = getattr(self._tracker, "_observables", [])
            if observables:
                counts: Counter[str] = Counter(
                    getattr(obs, "observable_type", "unknown")
                    for obs in observables
                )
                total = sum(counts.values()) or 1
                for obs_type, count in counts.most_common():
                    frequencies.append(
                        ObservableFrequency(
                            observable_type=obs_type,
                            count=count,
                            fraction=round(count / total, 6),
                        )
                    )
            return total_count, novel_count, novelty_rate, diversity_index, frequencies
        except Exception:
            logger.debug("shadow_reset_tracker_read_failed")
            return 0, 0, 0.0, 0.0, []

    async def _persist_snapshot(self, snapshot: ShadowSnapshot) -> None:
        if self._redis is None:
            return
        try:
            key = _SNAPSHOT_KEY.format(snapshot_id=snapshot.snapshot_id)
            await self._redis.set(
                key, json.dumps(snapshot.to_dict()), ex=_SNAPSHOT_TTL_S
            )
            # Update index list for this instance
            index_key = _SNAPSHOT_INDEX_KEY.format(instance_id=self._instance_id)
            await self._redis.rpush(index_key, snapshot.snapshot_id)
            await self._redis.expire(index_key, _SNAPSHOT_TTL_S)
        except Exception:
            logger.debug("shadow_snapshot_persist_failed", snapshot_id=snapshot.snapshot_id)

    async def _load_snapshot(self, snapshot_id: str) -> ShadowSnapshot | None:
        # Try session cache first
        if snapshot_id in self._session_snapshots:
            return self._session_snapshots[snapshot_id]

        # Try Redis
        if self._redis is None:
            return None
        try:
            key = _SNAPSHOT_KEY.format(snapshot_id=snapshot_id)
            raw = await self._redis.get(key)
            if raw is None:
                return None
            data = json.loads(raw)
            snap = ShadowSnapshot.from_dict(data)
            self._session_snapshots[snapshot_id] = snap
            return snap
        except Exception:
            logger.debug("shadow_snapshot_load_failed", snapshot_id=snapshot_id)
            return None


# ─── Utility ──────────────────────────────────────────────────────────────────


def _iso_to_timestamp(iso: str) -> float:
    """Parse an ISO-8601 string (with or without TZ) to a POSIX timestamp."""
    from datetime import datetime, timezone

    try:
        # Python 3.11+ supports Z suffix natively
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return time.time()
