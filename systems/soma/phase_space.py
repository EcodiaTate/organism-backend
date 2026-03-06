"""
EcodiaOS — Soma Phase-Space Model

Topological model of the organism's interoceptive state space.
Detects attractors (stable states the organism settles into),
maps bifurcation boundaries (tipping points between regimes),
and tracks trajectory navigation (which basin we're in, heading where).

Updated every 100 theta cycles (~15s). Budget: 2ms per update.
Uses online k-means for centroid updates, DBSCAN-inspired clustering
for new attractor discovery, and simple linear boundary fitting
for bifurcation detection.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import structlog

from systems.soma.types import (
    ALL_DIMENSIONS,
    SEED_ATTRACTORS,
    Attractor,
    Bifurcation,
    InteroceptiveDimension,
    PhaseSpaceSnapshot,
)

logger = structlog.get_logger("systems.soma.phase_space")


class PhaseSpaceModel:
    """
    Topological model of the organism's 9D interoceptive state space.

    Maintains attractors, bifurcation boundaries, and navigation state.
    Attractors are the organism's "moods" — stable basins in the landscape.
    Bifurcations are the tipping points between them.
    """

    def __init__(
        self,
        max_attractors: int = 20,
        min_dwell_cycles: int = 50,
        detection_enabled: bool = True,
    ) -> None:
        self._attractors: list[Attractor] = []
        self._bifurcations: list[Bifurcation] = []
        self._max_attractors = max_attractors
        self._min_dwell_cycles = min_dwell_cycles
        self._detection_enabled = detection_enabled

        # Navigation state
        self._current_attractor: Attractor | None = None
        self._previous_attractor: Attractor | None = None
        self._dwell_cycles: int = 0
        self._trajectory_heading: str = "transient"
        self._previous_distance: float = float("inf")

        # Transient state buffer for new attractor discovery
        self._transient_buffer: deque[dict[InteroceptiveDimension, float]] = deque(maxlen=200)

        # Seed initial attractors
        self._seed_attractors()

    def _seed_attractors(self) -> None:
        """Load seed attractors from constants."""
        for seed in SEED_ATTRACTORS:
            center = {
                InteroceptiveDimension(k): v
                for k, v in seed["center"].items()
            }
            self._attractors.append(Attractor(
                label=seed["label"],
                center=center,
                basin_radius=seed["basin_radius"],
                valence=seed["valence"],
            ))

    @property
    def attractors(self) -> list[Attractor]:
        return list(self._attractors)

    @property
    def bifurcations(self) -> list[Bifurcation]:
        return list(self._bifurcations)

    @property
    def current_attractor(self) -> Attractor | None:
        return self._current_attractor

    @property
    def attractor_count(self) -> int:
        return len(self._attractors)

    @property
    def bifurcation_count(self) -> int:
        return len(self._bifurcations)

    def snapshot(self) -> PhaseSpaceSnapshot:
        """Get current navigation state as a snapshot."""
        nearest_dist = float("inf")
        nearest_label: str | None = None
        bif_dist = float("inf")
        bif_time: float | None = None

        if self._current_attractor is not None:
            nearest_label = self._current_attractor.label
            nearest_dist = 0.0  # We're in the basin

        if self._bifurcations:
            bif_dist = min(b.distance({}) for b in self._bifurcations) if self._bifurcations else float("inf")

        return PhaseSpaceSnapshot(
            nearest_attractor=nearest_label or self._find_nearest_label(),
            nearest_attractor_distance=nearest_dist if nearest_label else self._find_nearest_distance(),
            trajectory_heading=self._trajectory_heading,
            distance_to_nearest_bifurcation=bif_dist,
            time_to_nearest_bifurcation=bif_time,
            attractor_count=len(self._attractors),
            bifurcation_count=len(self._bifurcations),
        )

    def snapshot_dict(self) -> dict[str, Any]:
        """Get phase-space state as a plain dict for signal construction."""
        snap = self.snapshot()
        return {
            "nearest_attractor": snap.nearest_attractor,
            "nearest_attractor_distance": snap.nearest_attractor_distance,
            "trajectory_heading": snap.trajectory_heading,
            "distance_to_nearest_bifurcation": snap.distance_to_nearest_bifurcation,
            "time_to_nearest_bifurcation": snap.time_to_nearest_bifurcation,
        }

    def update(
        self,
        trajectory: deque[dict[InteroceptiveDimension, float]],
        velocity: dict[InteroceptiveDimension, float],
    ) -> None:
        """
        Full phase-space update. Called every 100 theta cycles (~15s).

        Steps:
        1. Assign recent states to nearest attractor (or mark transient)
        2. Update attractor centroids (online k-means step)
        3. Detect new attractors from transient buffer
        4. Update transition probabilities
        5. Update bifurcation boundaries
        6. Compute navigation state
        """
        if not trajectory:
            return

        current_state = trajectory[-1]

        # 1. Find nearest attractor
        nearest, dist = self._find_nearest(current_state)

        # 2. Update dwell tracking and navigation
        if nearest is not None and dist <= nearest.basin_radius:
            if self._current_attractor != nearest:
                # Transition detected
                if self._current_attractor is not None:
                    self._record_transition(self._current_attractor, nearest)
                    self._previous_attractor = self._current_attractor
                self._current_attractor = nearest
                self._dwell_cycles = 0
            self._dwell_cycles += 1
            nearest.visits += 1
            self._transient_buffer.clear()
        else:
            # Transient — not in any basin
            if self._current_attractor is not None:
                self._previous_attractor = self._current_attractor
            self._current_attractor = None
            self._dwell_cycles = 0
            self._transient_buffer.append(dict(current_state))

        # 3. Update attractor centroids (online k-means)
        if nearest is not None and dist <= nearest.basin_radius:
            self._update_centroid(nearest, current_state)

        # 4. Detect new attractors from transient buffer
        if self._detection_enabled and len(self._transient_buffer) >= self._min_dwell_cycles:
            self._detect_new_attractors()

        # 5. Update trajectory heading
        if nearest is not None:
            if dist < self._previous_distance:
                self._trajectory_heading = "toward_attractor"
            elif dist > self._previous_distance + 0.01:
                self._trajectory_heading = "away"
            self._previous_distance = dist
        else:
            self._trajectory_heading = "transient"

        # 6. Check bifurcation proximity
        self._check_bifurcation_proximity(current_state, velocity)

    def get_nearest_attractor_label(
        self,
        state: dict[InteroceptiveDimension, float],
    ) -> str:
        """Get the label of the nearest attractor to a state."""
        nearest, _ = self._find_nearest(state)
        return nearest.label if nearest is not None else "transient"

    # ─── Internal Methods ─────────────────────────────────────────

    def _find_nearest(
        self, state: dict[InteroceptiveDimension, float],
    ) -> tuple[Attractor | None, float]:
        """Find the nearest attractor and its distance."""
        if not self._attractors:
            return None, float("inf")

        nearest: Attractor | None = None
        min_dist = float("inf")

        for attractor in self._attractors:
            dist = attractor.distance_to(state)
            if dist < min_dist:
                min_dist = dist
                nearest = attractor

        return nearest, min_dist

    def _find_nearest_label(self) -> str | None:
        if not self._attractors:
            return None
        return self._attractors[0].label  # Placeholder if no state available

    def _find_nearest_distance(self) -> float:
        return float("inf")

    def _update_centroid(
        self,
        attractor: Attractor,
        state: dict[InteroceptiveDimension, float],
    ) -> None:
        """Online k-means centroid update with learning rate 1/visits."""
        if attractor.visits <= 0:
            return
        lr = 1.0 / (attractor.visits + 1)
        for dim in ALL_DIMENSIONS:
            current = attractor.center.get(dim, 0.0)
            new_val = state.get(dim, 0.0)
            attractor.center[dim] = current + lr * (new_val - current)

    def _detect_new_attractors(self) -> None:
        """
        DBSCAN-inspired detection: if transient states cluster tightly,
        declare a new attractor. Simple distance-based approach.
        """
        if len(self._attractors) >= self._max_attractors:
            return

        if len(self._transient_buffer) < self._min_dwell_cycles:
            return

        # Compute centroid of transient buffer
        centroid: dict[InteroceptiveDimension, float] = {}
        n = len(self._transient_buffer)
        for dim in ALL_DIMENSIONS:
            total = sum(s.get(dim, 0.0) for s in self._transient_buffer)
            centroid[dim] = total / n

        # Compute spread (mean distance from centroid)
        distances = []
        for state in self._transient_buffer:
            dist = math.sqrt(sum(
                (state.get(d, 0.0) - centroid.get(d, 0.0)) ** 2
                for d in ALL_DIMENSIONS
            ))
            distances.append(dist)

        mean_spread = sum(distances) / len(distances) if distances else float("inf")

        # If spread is tight enough (<0.2 in 9D), declare a new attractor
        if mean_spread < 0.2:
            # Check it's not too close to an existing attractor
            for existing in self._attractors:
                if existing.distance_to(centroid) < existing.basin_radius * 2:
                    return  # Too close to existing

            new_attractor = Attractor(
                label=f"discovered_{len(self._attractors)}",
                center=centroid,
                basin_radius=max(mean_spread * 1.5, 0.1),
                stability=0.5,
                valence=0.0,  # Unknown valence initially
                visits=n,
            )
            self._attractors.append(new_attractor)
            self._transient_buffer.clear()

            logger.info(
                "new_attractor_discovered",
                label=new_attractor.label,
                radius=new_attractor.basin_radius,
            )

    def _record_transition(self, from_a: Attractor, to_a: Attractor) -> None:
        """Record a transition between attractors for probability tracking."""
        count = from_a.transitions.get(to_a.id, 0.0)
        from_a.transitions[to_a.id] = count + 1.0

        # Check if this transition defines a bifurcation
        total_transitions = sum(from_a.transitions.values())
        if total_transitions > 5:
            prob = from_a.transitions[to_a.id] / total_transitions
            # If this is a rare transition, it might be a bifurcation
            if prob < 0.3 and len(self._bifurcations) < 10:
                # Check if we already have this bifurcation
                existing = any(
                    b.pre_regime == from_a.label and b.post_regime == to_a.label
                    for b in self._bifurcations
                )
                if not existing:
                    self._bifurcations.append(Bifurcation(
                        label=f"{from_a.label}_to_{to_a.label}",
                        dimensions=list(ALL_DIMENSIONS),  # All dims initially
                        boundary_condition=f"{from_a.label} -> {to_a.label}",
                        pre_regime=from_a.label,
                        post_regime=to_a.label,
                        crossing_count=1,
                    ))

    def _check_bifurcation_proximity(
        self,
        state: dict[InteroceptiveDimension, float],
        velocity: dict[InteroceptiveDimension, float],
    ) -> None:
        """Check if approaching a bifurcation and update heading if so."""
        for bif in self._bifurcations:
            dist = bif.distance(state)
            if 0 < dist < 0.2:  # Approaching
                ttc = bif.time_to_crossing(state, velocity)
                if ttc is not None and ttc < 60.0:
                    self._trajectory_heading = "bifurcation_approach"
                    break
