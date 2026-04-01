"""
EcodiaOS - Soma Interoceptive Predictor

Multi-horizon generative model predicting future interoceptive states.
Pure numerical computation - NOT an LLM call. Total prediction budget: 1ms.

The predictor maintains:
  - A trajectory ring buffer (~150s at 150ms/tick)
  - A 9x9 dynamics coupling matrix (how dimensions influence each other)
  - A list of scheduled allostatic events (known upcoming exogenous impacts)

For each of 6 prediction horizons, it extrapolates:
  predicted = current + velocity * dt + 0.5 * acceleration * dt^2
with cross-dimension coupling from the dynamics matrix.
"""

from __future__ import annotations

from collections import deque

import structlog

from systems.soma.base import BaseSomaPredictor
from systems.soma.types import (
    ALL_DIMENSIONS,
    HORIZONS,
    InteroceptiveDimension,
    ScheduledAllostaticEvent,
    _clamp_dimension,
)

logger = structlog.get_logger("systems.soma.predictor")


class InteroceptivePredictor(BaseSomaPredictor):
    """
    Multi-horizon generative model for interoceptive state prediction.

    Computes velocity via exponential weighted mean (EWM) over the trajectory
    buffer, applies a 9x9 cross-dimension coupling matrix, adds scheduled
    event impacts, and extrapolates to each horizon.
    """

    def __init__(
        self,
        buffer_size: int = 1000,
        ewm_span: int = 20,
    ) -> None:
        self._trajectory: deque[dict[InteroceptiveDimension, float]] = deque(maxlen=buffer_size)
        self._ewm_span = ewm_span
        self._ewm_alpha = 2.0 / (ewm_span + 1)

        # 9x9 dynamics coupling matrix: dynamics[i][j] = effect of dim_j on dim_i
        # Updated by Evo during consolidation
        self._dynamics: list[list[float]] = self._default_dynamics()

        # Known upcoming events that affect predictions
        self._scheduled_events: list[ScheduledAllostaticEvent] = []

        # Recent error snapshots for error rate computation
        self._error_history: deque[dict[InteroceptiveDimension, float]] = deque(maxlen=10)

    def push_state(self, sensed: dict[InteroceptiveDimension, float]) -> None:
        """Add a new sensed state to the trajectory buffer."""
        self._trajectory.append(dict(sensed))

    def predict(
        self,
        horizon_name: str,
        current: dict[InteroceptiveDimension, float],
        available_horizons: list[str],
    ) -> dict[InteroceptiveDimension, float]:
        """
        Predict interoceptive state at a given horizon.

        Steps:
        1. Compute velocity per dimension from trajectory (EWM)
        2. Apply dynamics coupling (cross-dimension effects)
        3. Add scheduled event impacts
        4. Extrapolate: predicted = current + velocity*dt + 0.5*accel*dt^2
        5. Clamp to valid ranges
        """
        if horizon_name not in available_horizons:
            return dict(current)

        dt = HORIZONS.get(horizon_name, 0.15)
        velocity = self._compute_velocity()
        acceleration = self._compute_acceleration(velocity)

        # Apply dynamics coupling to velocity
        coupled_velocity = self._apply_dynamics(velocity)

        # Extrapolate
        predicted: dict[InteroceptiveDimension, float] = {}
        for dim in ALL_DIMENSIONS:
            v = coupled_velocity.get(dim, 0.0)
            a = acceleration.get(dim, 0.0)
            pred = current.get(dim, 0.0) + v * dt + 0.5 * a * dt * dt
            predicted[dim] = _clamp_dimension(dim, pred)

        # Apply scheduled event impacts
        predicted = self._apply_scheduled_events(predicted, dt)

        return predicted

    def predict_all_horizons(
        self,
        current: dict[InteroceptiveDimension, float],
        available_horizons: list[str],
    ) -> dict[str, dict[InteroceptiveDimension, float]]:
        """Predict at all available horizons."""
        predictions: dict[str, dict[InteroceptiveDimension, float]] = {}
        for horizon_name in available_horizons:
            predictions[horizon_name] = self.predict(
                horizon_name, current, available_horizons,
            )
        return predictions

    def compute_allostatic_errors(
        self,
        predictions: dict[str, dict[InteroceptiveDimension, float]],
        setpoints: dict[InteroceptiveDimension, float],
    ) -> dict[str, dict[InteroceptiveDimension, float]]:
        """
        Compute allostatic errors: predicted - setpoint per horizon per dimension.

        Positive error = overshoot (above target)
        Negative error = undershoot (below target)
        Magnitude is what the organism "feels"; sign determines regulatory action.
        """
        errors: dict[str, dict[InteroceptiveDimension, float]] = {}
        for horizon_name, predicted in predictions.items():
            horizon_errors: dict[InteroceptiveDimension, float] = {}
            for dim in ALL_DIMENSIONS:
                pred_val = predicted.get(dim, 0.0)
                setpoint = setpoints.get(dim, 0.0)
                horizon_errors[dim] = pred_val - setpoint
            errors[horizon_name] = horizon_errors
        return errors

    def compute_error_rates(
        self,
        current_errors: dict[InteroceptiveDimension, float],
    ) -> dict[InteroceptiveDimension, float]:
        """
        Compute d(error)/dt - is each dimension's error growing or shrinking?

        Positive rate = getting worse (alarm)
        Negative rate = resolving (relief)
        """
        self._error_history.append(dict(current_errors))

        if len(self._error_history) < 2:
            return {dim: 0.0 for dim in ALL_DIMENSIONS}

        rates: dict[InteroceptiveDimension, float] = {}
        for dim in ALL_DIMENSIONS:
            # EWM slope over recent error history
            values = [h.get(dim, 0.0) for h in self._error_history]
            if len(values) >= 2:
                # Simple weighted slope from last few values
                n = len(values)
                weights = [(1 - self._ewm_alpha) ** (n - 1 - i) for i in range(n)]
                w_sum = sum(weights)
                if w_sum > 0:
                    weighted_slope = sum(
                        w * (values[i] - values[max(0, i - 1)])
                        for i, w in enumerate(weights)
                    ) / w_sum
                    rates[dim] = weighted_slope
                else:
                    rates[dim] = 0.0
            else:
                rates[dim] = 0.0

        return rates

    def compute_temporal_dissonance(
        self,
        predictions: dict[str, dict[InteroceptiveDimension, float]],
    ) -> dict[InteroceptiveDimension, float]:
        """
        Temporal dissonance: moment prediction minus session prediction.

        Positive = feels good now, heading bad (temptation)
        Negative = feels bad now, heading good (perseverance)
        High |dissonance| signals Nova to deliberate time-horizon tradeoffs.
        """
        moment = predictions.get("moment", {})
        session = predictions.get("session", {})

        if not moment or not session:
            return {dim: 0.0 for dim in ALL_DIMENSIONS}

        return {
            dim: moment.get(dim, 0.0) - session.get(dim, 0.0)
            for dim in ALL_DIMENSIONS
        }

    def add_scheduled_event(self, event: ScheduledAllostaticEvent) -> None:
        """Register a known upcoming event that will impact predictions."""
        self._scheduled_events.append(event)
        # Keep sorted by timestamp
        self._scheduled_events.sort(key=lambda e: e.timestamp)

    def clear_expired_events(self, now_ts: float) -> None:
        """Remove events whose time + duration has passed."""
        self._scheduled_events = [
            e for e in self._scheduled_events
            if e.timestamp.timestamp() + e.duration_s > now_ts
        ]

    def update_dynamics(self, new_dynamics: list[list[float]]) -> None:
        """
        Evo updates the 9x9 cross-dimension coupling matrix.
        Validates dimensions before accepting.
        """
        n = len(ALL_DIMENSIONS)
        if len(new_dynamics) == n and all(len(row) == n for row in new_dynamics):
            self._dynamics = [list(row) for row in new_dynamics]
        else:
            logger.warning(
                "dynamics_update_rejected",
                expected=f"{n}x{n}",
                got=f"{len(new_dynamics)}x{len(new_dynamics[0]) if new_dynamics else 0}",
            )

    @property
    def trajectory_length(self) -> int:
        return len(self._trajectory)

    # ─── Internal Methods ─────────────────────────────────────────

    def _compute_velocity(self) -> dict[InteroceptiveDimension, float]:
        """
        Compute velocity per dimension from trajectory using EWM.
        Returns d(value)/dt in units per tick (150ms).
        """
        if len(self._trajectory) < 2:
            return {dim: 0.0 for dim in ALL_DIMENSIONS}

        velocity: dict[InteroceptiveDimension, float] = {}
        recent = list(self._trajectory)

        for dim in ALL_DIMENSIONS:
            # Compute deltas between consecutive states
            deltas: list[float] = []
            for i in range(1, len(recent)):
                deltas.append(recent[i].get(dim, 0.0) - recent[i - 1].get(dim, 0.0))

            if not deltas:
                velocity[dim] = 0.0
                continue

            # EWM over deltas
            ewm = deltas[0]
            alpha = self._ewm_alpha
            for d in deltas[1:]:
                ewm = alpha * d + (1 - alpha) * ewm

            velocity[dim] = ewm

        return velocity

    def _compute_acceleration(
        self, velocity: dict[InteroceptiveDimension, float],
    ) -> dict[InteroceptiveDimension, float]:
        """
        Compute acceleration (second derivative) from trajectory.
        Simple approach: difference of recent velocity estimates.
        """
        if len(self._trajectory) < 3:
            return {dim: 0.0 for dim in ALL_DIMENSIONS}

        # Use midpoint and endpoint for rough acceleration
        n = len(self._trajectory)
        mid = max(1, n // 2)
        recent = list(self._trajectory)

        acceleration: dict[InteroceptiveDimension, float] = {}
        for dim in ALL_DIMENSIONS:
            # Velocity at midpoint
            if mid > 0 and mid < n:
                v_mid = recent[mid].get(dim, 0.0) - recent[mid - 1].get(dim, 0.0)
            else:
                v_mid = 0.0
            # Velocity at end
            v_end = velocity.get(dim, 0.0)
            # Acceleration ≈ (v_end - v_mid) / (n - mid) ticks
            ticks = max(n - mid, 1)
            acceleration[dim] = (v_end - v_mid) / ticks

        return acceleration

    def _apply_dynamics(
        self, velocity: dict[InteroceptiveDimension, float],
    ) -> dict[InteroceptiveDimension, float]:
        """
        Apply 9x9 dynamics coupling matrix to velocity.
        coupled[i] = sum(dynamics[i][j] * velocity[j] for j in dims)
        """
        coupled: dict[InteroceptiveDimension, float] = {}
        dim_list = ALL_DIMENSIONS

        for i, dim_i in enumerate(dim_list):
            total = 0.0
            for j, dim_j in enumerate(dim_list):
                total += self._dynamics[i][j] * velocity.get(dim_j, 0.0)
            coupled[dim_i] = total

        return coupled

    def _apply_scheduled_events(
        self,
        predicted: dict[InteroceptiveDimension, float],
        dt: float,
    ) -> dict[InteroceptiveDimension, float]:
        """Add impacts from scheduled events whose time falls within the horizon."""
        result = dict(predicted)
        for event in self._scheduled_events:
            # Simple: apply event impact proportional to overlap with prediction window
            for dim, impact in event.dimension_impacts.items():
                if dim in result:
                    result[dim] = _clamp_dimension(dim, result[dim] + impact * 0.1)
        return result

    @staticmethod
    def _default_dynamics() -> list[list[float]]:
        """
        Default 9x9 dynamics coupling matrix.

        Diagonal = 1.0 (each dimension propagates itself).
        Off-diagonal entries encode known cross-dimension influences:
          - energy depletion increases arousal (stress response)
          - high arousal depletes energy faster
          - coherence loss reduces confidence
          - social engagement boosts valence
          - etc.
        """
        n = len(ALL_DIMENSIONS)
        # Start with identity - each dimension carries its own velocity
        matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        # Named indices for readability
        idx = {d: i for i, d in enumerate(ALL_DIMENSIONS)}
        e, a, v, co, ch, sc, cu, it, tp = (
            idx[InteroceptiveDimension.ENERGY],
            idx[InteroceptiveDimension.AROUSAL],
            idx[InteroceptiveDimension.VALENCE],
            idx[InteroceptiveDimension.CONFIDENCE],
            idx[InteroceptiveDimension.COHERENCE],
            idx[InteroceptiveDimension.SOCIAL_CHARGE],
            idx[InteroceptiveDimension.CURIOSITY_DRIVE],
            idx[InteroceptiveDimension.INTEGRITY],
            idx[InteroceptiveDimension.TEMPORAL_PRESSURE],
        )

        # Cross-dimension couplings (biologically inspired)
        matrix[a][e] = -0.15    # Low energy → arousal rises (stress response)
        matrix[e][a] = -0.10    # High arousal → energy depletes faster
        matrix[co][ch] = 0.05   # Higher coherence stabilises confidence
        matrix[v][sc] = 0.10    # Social engagement boosts valence
        matrix[v][co] = 0.05    # Confidence boosts valence
        matrix[ch][co] = 0.08   # Coherence → confidence
        matrix[cu][e] = 0.05    # More energy → curiosity grows
        matrix[cu][co] = 0.05   # Confidence enables curiosity
        matrix[tp][a] = 0.10    # Arousal amplifies temporal pressure
        matrix[tp][e] = -0.08   # Low energy → urgency rises
        matrix[it][ch] = 0.05   # Coherence supports integrity
        matrix[a][tp] = 0.05    # Temporal pressure feeds arousal

        return matrix

    # ─── ABC Bridge Properties ────────────────────────────────────

    @property
    def raw_trajectory(self) -> deque[dict[InteroceptiveDimension, float]]:
        """Expose trajectory buffer for PhaseSpaceModel."""
        return self._trajectory

    def compute_velocity(self) -> dict[InteroceptiveDimension, float]:
        """Public velocity accessor for PhaseSpaceModel heading."""
        return self._compute_velocity()

    @property
    def dynamics_matrix(self) -> list[list[float]]:
        """Expose dynamics matrix for CounterfactualEngine."""
        return self._dynamics
