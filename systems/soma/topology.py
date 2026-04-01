"""
EcodiaOS - Soma Topological Analyzer (Persistent Homology)

The persistence diagram encodes the *shape* of healthy behavior. Novel
pathologies that have never been seen before still register as topological
deformation. This is fundamentally more powerful than pattern matching.

Uses Vietoris-Rips persistent homology via ripser on a sliding window of
state vectors, then compares against a locked baseline barcode via
bottleneck distance (persim).

Runs every ~75s (500 theta cycles). Not on the critical path.

Dependencies: ripser, persim, numpy
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger("systems.soma.topology")

# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TopologicalFeature:
    """A single persistent feature extracted from the persistence diagram."""

    dimension: int  # H0, H1, H2
    birth: float  # Scale at which feature appears
    death: float  # Scale at which feature disappears
    persistence: float  # death - birth (lifetime)
    contributing_dims: tuple[int, ...]  # State vector dimensions with highest variance contribution


@dataclass(slots=True)
class PersistenceDiagnosis:
    """Result of a single topological analysis run."""

    timestamp: float = 0.0

    # Bottleneck distances from baseline per homology dimension
    h0_bottleneck: float = 0.0  # Component structure change
    h1_bottleneck: float = 0.0  # Cyclic behavior change
    h2_bottleneck: float = 0.0  # Void structure change

    # Composite topological health score (0 = identical to baseline, higher = more deformed)
    topological_health: float = 0.0

    # Structural changes detected
    new_features: list[TopologicalFeature] = field(default_factory=list)
    lost_features: list[TopologicalFeature] = field(default_factory=list)

    # Classified changes
    breaches: list[TopologicalFeature] = field(default_factory=list)  # Voids filling in
    fractures: list[TopologicalFeature] = field(default_factory=list)  # Components splitting
    novel_cycles: list[TopologicalFeature] = field(default_factory=list)  # New feedback loops

    def to_dict(self) -> dict[str, Any]:
        """Serialize for signal composition and logging."""
        return {
            "timestamp": self.timestamp,
            "h0_bottleneck": round(self.h0_bottleneck, 6),
            "h1_bottleneck": round(self.h1_bottleneck, 6),
            "h2_bottleneck": round(self.h2_bottleneck, 6),
            "topological_health": round(self.topological_health, 6),
            "new_feature_count": len(self.new_features),
            "lost_feature_count": len(self.lost_features),
            "breach_count": len(self.breaches),
            "fracture_count": len(self.fractures),
            "novel_cycle_count": len(self.novel_cycles),
        }


# ---------------------------------------------------------------------------
# TopologicalAnalyzer
# ---------------------------------------------------------------------------

# Minimum persistence threshold: features below this are noise
_MIN_PERSISTENCE: float = 0.01

# Weights for composite topological health score
_H0_WEIGHT: float = 0.4  # Component changes are most diagnostic
_H1_WEIGHT: float = 0.35  # Cyclic changes matter a lot
_H2_WEIGHT: float = 0.25  # Void changes are subtler


class TopologicalAnalyzer:
    """
    Computes persistent homology of the organism's state trajectory and
    compares against a stored healthy baseline barcode.

    The sliding window collects state vectors (numpy arrays). Every
    ``compute_interval_cycles`` theta cycles the analyzer subsamples
    the window and runs Vietoris-Rips homology via ripser. The resulting
    persistence diagram is compared to the baseline via bottleneck distance
    (persim). Structural changes are classified as breaches (voids filling),
    fractures (components splitting), or novel cycles (feedback instability).
    """

    def __init__(
        self,
        window_size: int = 5000,
        subsample_rate: int = 10,
        compute_interval_cycles: int = 500,
        max_homology_dim: int = 2,
    ) -> None:
        # Sliding window of raw state vectors
        self._window: deque[np.ndarray] = deque(maxlen=window_size)
        self._subsample_rate = max(1, subsample_rate)
        self._compute_interval_cycles = compute_interval_cycles
        self._max_dim = max_homology_dim  # H0, H1, H2

        # Healthy baseline persistence diagrams: {dim: ndarray of (birth, death) pairs}
        self._baseline: dict[int, np.ndarray] | None = None
        self._baseline_locked: bool = False
        self._baseline_lock = threading.Lock()

        # Last computed diagnosis (cached between runs)
        self._last_diagnosis: PersistenceDiagnosis | None = None
        self._cycle_counter: int = 0

    # -- Public API --------------------------------------------------------

    def push_state(self, state_vector: np.ndarray) -> None:
        """Buffer a new state vector. Called every theta cycle."""
        self._window.append(state_vector)
        self._cycle_counter += 1

    def should_compute(self) -> bool:
        """Whether enough cycles have passed for a new analysis."""
        return self._cycle_counter >= self._compute_interval_cycles

    def compute_persistence(self) -> PersistenceDiagnosis:
        """
        Run persistent homology on the current trajectory window.

        1. Subsample trajectory to manageable size (~500 points)
        2. Compute Vietoris-Rips filtration via ripser
        3. Extract persistence diagrams for H0, H1, H2
        4. Compare against baseline using bottleneck distance
        5. Classify structural changes
        """
        self._cycle_counter = 0

        points = self._subsample()
        if points is None or len(points) < 10:
            return PersistenceDiagnosis(timestamp=time.monotonic())

        diagrams = self._run_ripser(points)
        if diagrams is None:
            return PersistenceDiagnosis(timestamp=time.monotonic())

        diagnosis = self._compare_to_baseline(diagrams)

        self._last_diagnosis = diagnosis
        return diagnosis

    @property
    def last_diagnosis(self) -> PersistenceDiagnosis | None:
        return self._last_diagnosis

    @property
    def baseline_locked(self) -> bool:
        return self._baseline_locked

    @property
    def window_size(self) -> int:
        return len(self._window)

    # -- Baseline Management -----------------------------------------------

    def update_baseline(self, diagrams: dict[int, np.ndarray] | None = None) -> None:
        """
        Update the healthy baseline barcode.

        If ``diagrams`` is None, computes from the current window.
        Called during Oneiros sleep cycles for slow adaptation,
        or manually during initial calibration.
        """
        if diagrams is None:
            points = self._subsample()
            if points is None or len(points) < 10:
                return
            diagrams = self._run_ripser(points)
            if diagrams is None:
                return

        with self._baseline_lock:
            self._baseline = {
                dim: self._filter_noise(dgm)
                for dim, dgm in diagrams.items()
            }
        logger.info(
            "topology_baseline_updated",
            features={dim: len(dgm) for dim, dgm in diagrams.items()},
        )

    def lock_baseline(self) -> None:
        """Lock the baseline. Further updates require explicit unlock."""
        self._baseline_locked = True
        logger.info("topology_baseline_locked")

    def unlock_baseline(self) -> None:
        """Unlock the baseline for sleep-cycle recalibration."""
        self._baseline_locked = False

    # -- Internals ---------------------------------------------------------

    def _subsample(self) -> np.ndarray | None:
        """Subsample the trajectory window to ~500 points."""
        if len(self._window) < self._subsample_rate:
            return None
        indices = range(0, len(self._window), self._subsample_rate)
        points = np.array([self._window[i] for i in indices], dtype=np.float64)
        return points

    def _run_ripser(self, points: np.ndarray) -> dict[int, np.ndarray] | None:
        """Run Vietoris-Rips persistent homology via ripser."""
        try:
            from ripser import ripser as run_ripser  # type: ignore[import-untyped]

            result = run_ripser(
                points,
                maxdim=self._max_dim,
                do_cocycles=False,
            )
            raw_diagrams: list[np.ndarray] = result["dgms"]

            diagrams: dict[int, np.ndarray] = {}
            for dim in range(min(len(raw_diagrams), self._max_dim + 1)):
                dgm = raw_diagrams[dim]
                # Remove infinite-death features (the single connected component in H0)
                finite_mask = np.isfinite(dgm[:, 1]) if len(dgm) > 0 else np.array([], dtype=bool)
                diagrams[dim] = dgm[finite_mask] if np.any(finite_mask) else np.empty((0, 2))

            return diagrams

        except Exception as exc:
            logger.warning("topology_ripser_failed", error=str(exc))
            return None

    def _filter_noise(self, dgm: np.ndarray) -> np.ndarray:
        """Remove features with persistence below threshold."""
        if len(dgm) == 0:
            return dgm
        persistence = dgm[:, 1] - dgm[:, 0]
        mask = persistence >= _MIN_PERSISTENCE
        return np.asarray(dgm[mask])

    def _compare_to_baseline(self, diagrams: dict[int, np.ndarray]) -> PersistenceDiagnosis:
        """Compare current diagrams to baseline. Compute bottleneck distances."""
        ts = time.monotonic()

        # If no baseline yet, set it from this first computation and return healthy
        with self._baseline_lock:
            if self._baseline is None:
                self._baseline = {
                    dim: self._filter_noise(dgm)
                    for dim, dgm in diagrams.items()
                }
                logger.info("topology_baseline_initialized_from_first_run")
                return PersistenceDiagnosis(timestamp=ts)

            baseline = dict(self._baseline)

        # Compute bottleneck distances per dimension
        bottleneck_distances: dict[int, float] = {}
        for dim in range(self._max_dim + 1):
            current = self._filter_noise(diagrams.get(dim, np.empty((0, 2))))
            base = baseline.get(dim, np.empty((0, 2)))
            bottleneck_distances[dim] = self._bottleneck_distance(current, base)

        h0_bn = bottleneck_distances.get(0, 0.0)
        h1_bn = bottleneck_distances.get(1, 0.0)
        h2_bn = bottleneck_distances.get(2, 0.0)

        # Composite health score
        health = _H0_WEIGHT * h0_bn + _H1_WEIGHT * h1_bn + _H2_WEIGHT * h2_bn

        # Detect structural changes
        new_features: list[TopologicalFeature] = []
        lost_features: list[TopologicalFeature] = []

        for dim in range(self._max_dim + 1):
            current = self._filter_noise(diagrams.get(dim, np.empty((0, 2))))
            base = baseline.get(dim, np.empty((0, 2)))
            nf, lf = self._diff_diagrams(dim, current, base)
            new_features.extend(nf)
            lost_features.extend(lf)

        # Classify structural changes
        breaches: list[TopologicalFeature] = []
        fractures: list[TopologicalFeature] = []
        novel_cycles: list[TopologicalFeature] = []

        for f in new_features:
            if f.dimension == 0:
                fractures.append(f)
            elif f.dimension == 1:
                novel_cycles.append(f)

        for f in lost_features:
            if f.dimension == 2:
                breaches.append(f)

        diag = PersistenceDiagnosis(
            timestamp=ts,
            h0_bottleneck=h0_bn,
            h1_bottleneck=h1_bn,
            h2_bottleneck=h2_bn,
            topological_health=health,
            new_features=new_features,
            lost_features=lost_features,
            breaches=breaches,
            fractures=fractures,
            novel_cycles=novel_cycles,
        )

        if health > 0.1:
            logger.info(
                "topology_deformation_detected",
                h0=round(h0_bn, 4),
                h1=round(h1_bn, 4),
                h2=round(h2_bn, 4),
                health=round(health, 4),
                breaches=len(breaches),
                fractures=len(fractures),
                novel_cycles=len(novel_cycles),
            )

        return diag

    def _bottleneck_distance(self, dgm_a: np.ndarray, dgm_b: np.ndarray) -> float:
        """Compute bottleneck distance between two persistence diagrams."""
        try:
            from persim import bottleneck  # type: ignore[import-untyped]

            # persim expects (n, 2) arrays; empty arrays need special handling
            if len(dgm_a) == 0 and len(dgm_b) == 0:
                return 0.0
            if len(dgm_a) == 0:
                dgm_a = np.empty((0, 2))
            if len(dgm_b) == 0:
                dgm_b = np.empty((0, 2))

            dist: float = bottleneck(dgm_a, dgm_b)
            return dist

        except Exception:
            # Fallback: approximate via max persistence difference
            return self._approximate_bottleneck(dgm_a, dgm_b)

    @staticmethod
    def _approximate_bottleneck(dgm_a: np.ndarray, dgm_b: np.ndarray) -> float:
        """Fallback bottleneck approximation when persim is unavailable."""
        def _stats(dgm: np.ndarray) -> tuple[float, float, int]:
            if len(dgm) == 0:
                return 0.0, 0.0, 0
            pers = dgm[:, 1] - dgm[:, 0]
            return float(np.max(pers)), float(np.mean(pers)), len(pers)

        max_a, mean_a, count_a = _stats(dgm_a)
        max_b, mean_b, count_b = _stats(dgm_b)

        return max(
            abs(max_a - max_b),
            abs(mean_a - mean_b),
            abs(count_a - count_b) * 0.01,
        )

    def _diff_diagrams(
        self,
        dimension: int,
        current: np.ndarray,
        baseline: np.ndarray,
    ) -> tuple[list[TopologicalFeature], list[TopologicalFeature]]:
        """
        Identify new features (in current but not baseline) and
        lost features (in baseline but not current).

        Uses a greedy matching by birth-death proximity.
        """
        new_features: list[TopologicalFeature] = []
        lost_features: list[TopologicalFeature] = []

        if len(current) == 0 and len(baseline) == 0:
            return new_features, lost_features

        # Match current features to nearest baseline feature
        matched_baseline: set[int] = set()
        match_threshold = 0.05  # Birth-death L-inf threshold for matching

        for i in range(len(current)):
            birth_c, death_c = current[i]
            persistence_c = death_c - birth_c

            best_j: int | None = None
            best_dist = float("inf")

            for j in range(len(baseline)):
                if j in matched_baseline:
                    continue
                birth_b, death_b = baseline[j]
                dist = max(abs(birth_c - birth_b), abs(death_c - death_b))
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j is not None and best_dist <= match_threshold:
                matched_baseline.add(best_j)
            else:
                new_features.append(TopologicalFeature(
                    dimension=dimension,
                    birth=float(birth_c),
                    death=float(death_c),
                    persistence=float(persistence_c),
                    contributing_dims=(),
                ))

        # Unmatched baseline features are lost
        for j in range(len(baseline)):
            if j not in matched_baseline:
                birth_b, death_b = baseline[j]
                lost_features.append(TopologicalFeature(
                    dimension=dimension,
                    birth=float(birth_b),
                    death=float(death_b),
                    persistence=float(death_b - birth_b),
                    contributing_dims=(),
                ))

        return new_features, lost_features
