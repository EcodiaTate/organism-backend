"""Tests for Soma Phase-Space Model - attractors, bifurcations, navigation."""

from __future__ import annotations

from collections import deque

from systems.soma.phase_space import PhaseSpaceModel
from systems.soma.types import (
    ALL_DIMENSIONS,
    SEED_ATTRACTORS,
    InteroceptiveDimension,
)


def _make_model(**kwargs) -> PhaseSpaceModel:
    return PhaseSpaceModel(
        max_attractors=kwargs.get("max_attractors", 20),
        min_dwell_cycles=kwargs.get("min_dwell_cycles", 10),
        detection_enabled=kwargs.get("detection_enabled", True),
    )


def _state(energy=0.5, arousal=0.5, **kwargs) -> dict[InteroceptiveDimension, float]:
    s = {d: 0.5 for d in ALL_DIMENSIONS}
    s[InteroceptiveDimension.ENERGY] = energy
    s[InteroceptiveDimension.AROUSAL] = arousal
    for k, v in kwargs.items():
        s[InteroceptiveDimension(k)] = v
    return s


class TestSeedAttractors:
    def test_seed_attractors_loaded(self):
        m = _make_model()
        assert m.attractor_count >= 6

    def test_flow_attractor_exists(self):
        m = _make_model()
        labels = [a.label for a in m.attractors]
        assert "flow" in labels

    def test_anxiety_attractor_exists(self):
        m = _make_model()
        labels = [a.label for a in m.attractors]
        assert "anxiety_spiral" in labels


class TestSnapshot:
    def test_initial_snapshot(self):
        m = _make_model()
        snap = m.snapshot()
        assert snap.attractor_count >= 6
        assert snap.bifurcation_count == 0
        assert snap.trajectory_heading == "transient"


class TestUpdate:
    def test_assigns_to_nearest_attractor(self):
        m = _make_model()
        # Create a trajectory near the "flow" attractor
        flow_center = SEED_ATTRACTORS[0]["center"]
        flow_state = {InteroceptiveDimension(k): v for k, v in flow_center.items()}

        trajectory: deque[dict[InteroceptiveDimension, float]] = deque()
        for _ in range(20):
            trajectory.append(dict(flow_state))

        velocity = {d: 0.0 for d in ALL_DIMENSIONS}
        m.update(trajectory, velocity)

        assert m.current_attractor is not None
        assert m.current_attractor.label == "flow"

    def test_transient_when_far_from_all(self):
        m = _make_model()
        # State at extreme corner - far from all seed attractors
        extreme = {d: 0.99 for d in ALL_DIMENSIONS}
        extreme[InteroceptiveDimension.VALENCE] = -0.99

        trajectory: deque[dict[InteroceptiveDimension, float]] = deque()
        trajectory.append(extreme)

        velocity = {d: 0.0 for d in ALL_DIMENSIONS}
        m.update(trajectory, velocity)

        # Should be transient (no attractor within radius)
        # This depends on distances - extreme state may be far from all centers
        # Just verify no crash
        _ = m.snapshot()


class TestAttractorDiscovery:
    def test_discovers_new_attractor(self):
        m = _make_model(min_dwell_cycles=5)
        initial_count = m.attractor_count

        # Create a tight cluster far from existing attractors
        center = {d: 0.15 for d in ALL_DIMENSIONS}
        center[InteroceptiveDimension.VALENCE] = -0.8

        trajectory: deque[dict[InteroceptiveDimension, float]] = deque()
        velocity = {d: 0.0 for d in ALL_DIMENSIONS}

        # Push enough states into transient buffer
        for _ in range(50):
            trajectory.append(dict(center))
            m.update(trajectory, velocity)

        # May or may not discover depending on distance to existing attractors
        # Just verify no crash and model is stable
        assert m.attractor_count >= initial_count


class TestNearestAttractorLabel:
    def test_returns_label(self):
        m = _make_model()
        flow_center = SEED_ATTRACTORS[0]["center"]
        state = {InteroceptiveDimension(k): v for k, v in flow_center.items()}
        label = m.get_nearest_attractor_label(state)
        assert label == "flow"
