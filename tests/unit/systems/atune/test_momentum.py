"""
Unit tests for Atune — Salience Momentum Tracker.

Covers:
    * Ring buffer storage and windowing
    * First and second derivative computation
    * Trajectory classification (steady, rising, falling, accelerating)
    * Time-to-threshold estimation
    * Momentum bonus calculation
    * Overall trajectory aggregation
    * Edge cases (single sample, identical timestamps, stationary scores)
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from systems.atune.momentum import SalienceMomentumTracker

from systems.atune.types import HeadMomentum, ThreatTrajectory

# ─── Helpers ──────────────────────────────────────────────────────


def _ts(seconds: float) -> datetime:
    """Create a UTC datetime at a given offset in seconds from epoch 0."""
    return datetime.fromtimestamp(seconds, tz=UTC)


# ─── Basic recording and derivative computation ──────────────────


class TestDerivatives:
    """First and second derivative computation via finite difference."""

    def test_single_sample_returns_neutral(self) -> None:
        tracker = SalienceMomentumTracker()
        tracker.record("risk", _ts(1.0), 0.3)
        m = tracker.compute_momentum("risk")
        assert m.first_derivative == 0.0
        assert m.second_derivative == 0.0
        assert m.trajectory == ThreatTrajectory.STEADY

    def test_two_samples_computes_first_derivative(self) -> None:
        tracker = SalienceMomentumTracker()
        tracker.record("risk", _ts(1.0), 0.2)
        tracker.record("risk", _ts(2.0), 0.4)
        m = tracker.compute_momentum("risk")
        # dv/dt = (0.4 - 0.2) / 1.0 = 0.2
        assert m.first_derivative == pytest.approx(0.2, abs=1e-5)
        assert m.second_derivative == 0.0  # Need 3 samples for 2nd derivative

    def test_three_samples_computes_second_derivative(self) -> None:
        tracker = SalienceMomentumTracker()
        tracker.record("risk", _ts(1.0), 0.1)
        tracker.record("risk", _ts(2.0), 0.3)
        tracker.record("risk", _ts(3.0), 0.6)
        m = tracker.compute_momentum("risk")
        # dv/dt[t2->t3] = (0.6 - 0.3) / 1.0 = 0.3
        assert m.first_derivative == pytest.approx(0.3, abs=1e-5)
        # dv/dt[t1->t2] = (0.3 - 0.1) / 1.0 = 0.2
        # d2v/dt2 = (0.3 - 0.2) / 1.0 = 0.1
        assert m.second_derivative == pytest.approx(0.1, abs=1e-5)

    def test_falling_first_derivative(self) -> None:
        tracker = SalienceMomentumTracker()
        tracker.record("novelty", _ts(1.0), 0.8)
        tracker.record("novelty", _ts(2.0), 0.5)
        m = tracker.compute_momentum("novelty")
        assert m.first_derivative == pytest.approx(-0.3, abs=1e-5)

    def test_unknown_head_returns_neutral(self) -> None:
        tracker = SalienceMomentumTracker()
        m = tracker.compute_momentum("nonexistent")
        assert m == HeadMomentum()


# ─── Trajectory classification ────────────────────────────────────


class TestTrajectory:
    """Trajectory labels derived from derivative thresholds."""

    def test_steady_when_flat(self) -> None:
        tracker = SalienceMomentumTracker()
        tracker.record("risk", _ts(1.0), 0.5)
        tracker.record("risk", _ts(2.0), 0.5)
        tracker.record("risk", _ts(3.0), 0.5)
        m = tracker.compute_momentum("risk")
        assert m.trajectory == ThreatTrajectory.STEADY

    def test_rising_when_positive_velocity(self) -> None:
        tracker = SalienceMomentumTracker(rising_threshold=0.01)
        tracker.record("risk", _ts(1.0), 0.3)
        tracker.record("risk", _ts(2.0), 0.35)
        tracker.record("risk", _ts(3.0), 0.4)
        m = tracker.compute_momentum("risk")
        assert m.trajectory == ThreatTrajectory.RISING

    def test_falling_when_negative_velocity(self) -> None:
        tracker = SalienceMomentumTracker(falling_threshold=-0.01)
        tracker.record("risk", _ts(1.0), 0.8)
        tracker.record("risk", _ts(2.0), 0.7)
        tracker.record("risk", _ts(3.0), 0.6)
        m = tracker.compute_momentum("risk")
        assert m.trajectory == ThreatTrajectory.FALLING

    def test_accelerating_when_high_second_derivative(self) -> None:
        tracker = SalienceMomentumTracker(accel_threshold=0.05)
        # Scores: 0.1 → 0.2 → 0.5 (accelerating rise)
        tracker.record("risk", _ts(1.0), 0.1)
        tracker.record("risk", _ts(2.0), 0.2)
        tracker.record("risk", _ts(3.0), 0.5)
        m = tracker.compute_momentum("risk")
        # dv/dt = 0.3, d2v/dt2 = 0.2 → both positive → accelerating
        assert m.trajectory == ThreatTrajectory.ACCELERATING

    def test_not_accelerating_when_decelerating_rise(self) -> None:
        """Rising but slowing down → RISING, not ACCELERATING."""
        tracker = SalienceMomentumTracker(accel_threshold=0.05)
        tracker.record("risk", _ts(1.0), 0.1)
        tracker.record("risk", _ts(2.0), 0.4)
        tracker.record("risk", _ts(3.0), 0.5)
        m = tracker.compute_momentum("risk")
        # dv/dt[2->3] = 0.1, dv/dt[1->2] = 0.3 → d2v = -0.2 → negative accel
        assert m.trajectory == ThreatTrajectory.RISING


# ─── Time-to-threshold estimation ────────────────────────────────


class TestTimeToThreshold:
    """Linear extrapolation of when a head will cross the ignition threshold."""

    def test_eta_when_rising(self) -> None:
        tracker = SalienceMomentumTracker(ignition_threshold=0.7)
        tracker.record("risk", _ts(1.0), 0.2)
        tracker.record("risk", _ts(2.0), 0.4)
        m = tracker.compute_momentum("risk")
        # current = 0.4, dv/dt = 0.2, remaining = 0.3
        # eta = 0.3 / 0.2 = 1.5 seconds
        assert m.time_to_threshold == pytest.approx(1.5, abs=0.01)

    def test_no_eta_when_above_threshold(self) -> None:
        tracker = SalienceMomentumTracker(ignition_threshold=0.5)
        tracker.record("risk", _ts(1.0), 0.4)
        tracker.record("risk", _ts(2.0), 0.6)
        m = tracker.compute_momentum("risk")
        assert m.time_to_threshold is None

    def test_no_eta_when_falling(self) -> None:
        tracker = SalienceMomentumTracker(ignition_threshold=0.7)
        tracker.record("risk", _ts(1.0), 0.5)
        tracker.record("risk", _ts(2.0), 0.4)
        m = tracker.compute_momentum("risk")
        assert m.time_to_threshold is None

    def test_no_eta_when_too_far(self) -> None:
        """If ETA exceeds 5 minutes, return None (not actionable)."""
        tracker = SalienceMomentumTracker(ignition_threshold=0.7)
        tracker.record("risk", _ts(1.0), 0.0)
        tracker.record("risk", _ts(2.0), 0.001)
        m = tracker.compute_momentum("risk")
        # dv/dt = 0.001, remaining = 0.699 → eta = 699s > 300 → None
        assert m.time_to_threshold is None


# ─── Momentum bonus ──────────────────────────────────────────────


class TestMomentumBonus:
    """Bonus injected when acceleration exceeds threshold."""

    def test_no_bonus_when_steady(self) -> None:
        tracker = SalienceMomentumTracker()
        tracker.record("risk", _ts(1.0), 0.3)
        tracker.record("risk", _ts(2.0), 0.3)
        tracker.record("risk", _ts(3.0), 0.3)
        m = tracker.compute_momentum("risk")
        assert m.momentum_bonus == 0.0

    def test_bonus_when_accelerating(self) -> None:
        tracker = SalienceMomentumTracker(accel_threshold=0.05, momentum_bonus_scale=0.15)
        tracker.record("risk", _ts(1.0), 0.1)
        tracker.record("risk", _ts(2.0), 0.2)
        tracker.record("risk", _ts(3.0), 0.5)
        m = tracker.compute_momentum("risk")
        assert m.momentum_bonus > 0.0
        assert m.momentum_bonus <= 0.15

    def test_bonus_scales_with_headroom(self) -> None:
        """Low current score + high accel = bigger bonus (pre-emptive)."""
        tracker = SalienceMomentumTracker(accel_threshold=0.01, momentum_bonus_scale=0.15)
        # Low current score scenario
        tracker.record("risk", _ts(1.0), 0.05)
        tracker.record("risk", _ts(2.0), 0.10)
        tracker.record("risk", _ts(3.0), 0.25)
        m_low = tracker.compute_momentum("risk")

        # High current score scenario (less headroom)
        tracker2 = SalienceMomentumTracker(accel_threshold=0.01, momentum_bonus_scale=0.15)
        tracker2.record("risk", _ts(1.0), 0.75)
        tracker2.record("risk", _ts(2.0), 0.80)
        tracker2.record("risk", _ts(3.0), 0.95)
        m_high = tracker2.compute_momentum("risk")

        assert m_low.momentum_bonus > m_high.momentum_bonus

    def test_no_bonus_when_falling(self) -> None:
        """Even with negative acceleration, no bonus when dv/dt < 0."""
        tracker = SalienceMomentumTracker(accel_threshold=0.05)
        tracker.record("risk", _ts(1.0), 0.8)
        tracker.record("risk", _ts(2.0), 0.6)
        tracker.record("risk", _ts(3.0), 0.5)
        m = tracker.compute_momentum("risk")
        assert m.momentum_bonus == 0.0


# ─── Window size and ring buffer behaviour ────────────────────────


class TestRingBuffer:
    """Fixed-size window evicts oldest samples."""

    def test_window_capped(self) -> None:
        tracker = SalienceMomentumTracker(window_size=3)
        for i in range(10):
            tracker.record("risk", _ts(float(i)), float(i) * 0.1)
        assert tracker.head_history_size("risk") == 3

    def test_old_samples_evicted(self) -> None:
        tracker = SalienceMomentumTracker(window_size=3)
        tracker.record("risk", _ts(1.0), 0.1)
        tracker.record("risk", _ts(2.0), 0.2)
        tracker.record("risk", _ts(3.0), 0.3)
        tracker.record("risk", _ts(4.0), 0.4)
        # Only 3 most recent: (2.0, 0.2), (3.0, 0.3), (4.0, 0.4)
        m = tracker.compute_momentum("risk")
        assert m.first_derivative == pytest.approx(0.1, abs=1e-5)


# ─── Overall trajectory ──────────────────────────────────────────


class TestOverallTrajectory:
    """Worst-case trajectory across all heads."""

    def test_accelerating_dominates(self) -> None:
        tracker = SalienceMomentumTracker(accel_threshold=0.05)
        # Head A: steady
        tracker.record("identity", _ts(1.0), 0.5)
        tracker.record("identity", _ts(2.0), 0.5)
        tracker.record("identity", _ts(3.0), 0.5)
        # Head B: accelerating
        tracker.record("risk", _ts(1.0), 0.1)
        tracker.record("risk", _ts(2.0), 0.2)
        tracker.record("risk", _ts(3.0), 0.5)
        all_m = tracker.compute_all_momentum()
        overall = tracker.overall_trajectory(all_m)
        assert overall == ThreatTrajectory.ACCELERATING

    def test_rising_when_no_acceleration(self) -> None:
        tracker = SalienceMomentumTracker(rising_threshold=0.01, accel_threshold=0.5)
        tracker.record("risk", _ts(1.0), 0.3)
        tracker.record("risk", _ts(2.0), 0.35)
        tracker.record("risk", _ts(3.0), 0.4)
        all_m = tracker.compute_all_momentum()
        overall = tracker.overall_trajectory(all_m)
        assert overall == ThreatTrajectory.RISING

    def test_empty_returns_steady(self) -> None:
        tracker = SalienceMomentumTracker()
        assert tracker.overall_trajectory({}) == ThreatTrajectory.STEADY


# ─── Multiple heads tracked independently ────────────────────────


class TestMultiHead:
    """Each head has an independent history."""

    def test_independent_tracking(self) -> None:
        tracker = SalienceMomentumTracker()
        tracker.record("risk", _ts(1.0), 0.2)
        tracker.record("risk", _ts(2.0), 0.4)
        tracker.record("novelty", _ts(1.0), 0.8)
        tracker.record("novelty", _ts(2.0), 0.6)

        m_risk = tracker.compute_momentum("risk")
        m_novelty = tracker.compute_momentum("novelty")

        assert m_risk.first_derivative == pytest.approx(0.2, abs=1e-5)
        assert m_novelty.first_derivative == pytest.approx(-0.2, abs=1e-5)

    def test_tracked_heads_list(self) -> None:
        tracker = SalienceMomentumTracker()
        tracker.record("risk", _ts(1.0), 0.1)
        tracker.record("goal", _ts(1.0), 0.2)
        assert set(tracker.tracked_heads) == {"risk", "goal"}


# ─── Edge cases ───────────────────────────────────────────────────


class TestEdgeCases:
    """Degenerate inputs should not crash or produce nonsense."""

    def test_identical_timestamps_handled(self) -> None:
        """Two samples at the same time should not divide by zero."""
        tracker = SalienceMomentumTracker()
        tracker.record("risk", _ts(1.0), 0.2)
        tracker.record("risk", _ts(1.0), 0.4)
        m = tracker.compute_momentum("risk")
        # Uses _MIN_DT_SECONDS as floor, so derivative is large but finite
        assert m.first_derivative > 0
        assert m.first_derivative == m.first_derivative  # Not NaN

    def test_score_at_zero(self) -> None:
        tracker = SalienceMomentumTracker()
        tracker.record("risk", _ts(1.0), 0.0)
        tracker.record("risk", _ts(2.0), 0.0)
        m = tracker.compute_momentum("risk")
        assert m.first_derivative == 0.0
        assert m.trajectory == ThreatTrajectory.STEADY

    def test_score_at_one(self) -> None:
        tracker = SalienceMomentumTracker()
        tracker.record("risk", _ts(1.0), 1.0)
        tracker.record("risk", _ts(2.0), 1.0)
        m = tracker.compute_momentum("risk")
        assert m.first_derivative == 0.0
        assert m.time_to_threshold is None  # Already at max
