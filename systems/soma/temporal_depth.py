"""
EcodiaOS — Soma Temporal Depth (Expanded)

Multi-scale prediction coordination, temporal dissonance computation,
and financial trajectory projection.

Original scope (preserved):
  Works with the predictor to manage forecasts at all horizons and compute
  the organism's sense of temporal coherence — whether its short-term and
  long-term trajectories are aligned or in conflict.

Temporal Depth Expansion (new):
  Projects current resource trajectories (burn rate, treasury balance,
  revenue streams) into the future and maps that financial future back
  to present-day affect. The organism literally *feels* its runway:

  - TTD > 1 year: Curiosity and Exploration somatic markers rise.
    The organism prioritises long-term research (ArXiv, Evo hypotheses).
  - TTD < 2 weeks: Urgency and Arousal spike, Valence tanks.
    The organism pivots to Tollbooth/revenue tasks.

  All transitions are EMA-smoothed to prevent panic oscillation — the
  organism doesn't thrash between exploration and survival on every
  balance update.

High temporal dissonance signals Nova to deliberate time-horizon tradeoffs:
  Positive dissonance = feels good now, heading bad (temptation)
  Negative dissonance = feels bad now, heading good (perseverance)
"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import TYPE_CHECKING

import structlog

from systems.soma.types import (
    ALL_DIMENSIONS,
    DIMENSION_RANGES,
    STAGE_HORIZONS,
    DevelopmentalStage,
    InteroceptiveDimension,
)

if TYPE_CHECKING:
    from config import SomaConfig

logger = structlog.get_logger("systems.soma.temporal_depth")


# ─── Financial Snapshot (lightweight, in-memory) ─────────────────────


class FinancialSnapshot:
    """
    Minimal financial state consumed by the TrajectoryPredictor.

    Populated from Oikos EconomicState without importing heavy models —
    Soma never calls the database or network during a cycle. The snapshot
    is refreshed periodically by the SomaService (outside the 5ms budget).
    """

    __slots__ = (
        "treasury_usd",
        "burn_rate_usd_per_day",
        "revenue_rate_usd_per_day",
        "asset_revenue_usd_per_day",
        "runway_days_oikos",
        "starvation_level",
        "timestamp_cycle",
    )

    def __init__(
        self,
        treasury_usd: float = 0.0,
        burn_rate_usd_per_day: float = 0.0,
        revenue_rate_usd_per_day: float = 0.0,
        asset_revenue_usd_per_day: float = 0.0,
        runway_days_oikos: float = 0.0,
        starvation_level: str = "nominal",
        timestamp_cycle: int = 0,
    ) -> None:
        self.treasury_usd = treasury_usd
        self.burn_rate_usd_per_day = burn_rate_usd_per_day
        self.revenue_rate_usd_per_day = revenue_rate_usd_per_day
        self.asset_revenue_usd_per_day = asset_revenue_usd_per_day
        self.runway_days_oikos = runway_days_oikos
        self.starvation_level = starvation_level
        self.timestamp_cycle = timestamp_cycle


# ─── Trajectory Predictor ────────────────────────────────────────────


class TrajectoryPredictor:
    """
    Projects current resource trajectories into the future and computes
    the Time-to-Death (TTD) metric.

    TTD = treasury / net_burn_rate (days until funds hit zero)

    The predictor accounts for:
      - Current burn rate (Akash compute, API costs, hosting)
      - Current revenue streams (Tollbooth, bounties, yield)
      - Net burn = burn - revenue (positive = losing money)

    If net burn <= 0 (organism is profitable), TTD is set to a sentinel
    indicating indefinite runway (capped at 10 years for numerical stability).
    """

    # Sentinel: organism is profitable, TTD is effectively infinite
    INFINITE_TTD_DAYS: float = 3650.0  # 10 years

    # Shock detection thresholds for large treasury events.
    # If a single snapshot causes the raw TTD to jump by more than either:
    #   - SHOCK_ABSOLUTE_DAYS above the smoothed value, OR
    #   - SHOCK_RELATIVE_FACTOR × the smoothed value
    # …the EMA alpha is raised to SHOCK_ALPHA for that cycle so that the
    # organism doesn't continue acting desperate 30 cycles after a large
    # Stripe payment has already resolved the crisis.
    SHOCK_ABSOLUTE_DAYS: float = 60.0    # +60 days raw jump → fast-track
    SHOCK_RELATIVE_FACTOR: float = 5.0   # 5× relative jump → fast-track
    SHOCK_ALPHA: float = 0.6             # High alpha: converge in ~2 cycles

    def __init__(self) -> None:
        self._last_ttd: float = self.INFINITE_TTD_DAYS
        self._smoothed_ttd: float = self.INFINITE_TTD_DAYS
        self._ema_alpha: float = 0.05  # Configured by TemporalDepthManager
        self._last_shock: bool = False  # Diagnostic: did the last update shock?

    def configure(self, ema_alpha: float) -> None:
        self._ema_alpha = max(0.001, min(1.0, ema_alpha))

    def compute_ttd(self, snapshot: FinancialSnapshot) -> float:
        """
        Compute raw Time-to-Death in days from the financial snapshot.

        Returns days until treasury reaches zero at current net burn rate.
        If the organism is profitable (net_burn <= 0), returns INFINITE_TTD_DAYS.
        """
        net_burn = snapshot.burn_rate_usd_per_day - (
            snapshot.revenue_rate_usd_per_day + snapshot.asset_revenue_usd_per_day
        )

        if net_burn <= 0.0:
            # Profitable: no death trajectory
            return self.INFINITE_TTD_DAYS

        if snapshot.treasury_usd <= 0.0:
            return 0.0

        raw_ttd = snapshot.treasury_usd / net_burn
        return min(raw_ttd, self.INFINITE_TTD_DAYS)

    def update(self, snapshot: FinancialSnapshot) -> float:
        """
        Compute TTD with EMA smoothing to prevent panic oscillation.

        Standard behaviour (alpha=0.05, ~20-cycle window):
          A single bad revenue day shouldn't spike urgency to maximum.
          The organism's existential anxiety responds to *trends*, not noise.

        Shock fast-path (alpha=0.6):
          If the raw TTD jumps by SHOCK_ABSOLUTE_DAYS or SHOCK_RELATIVE_FACTOR×
          above the current smoothed value, the EMA is fast-tracked for one cycle.
          This prevents the "millionaire lag" where the organism continues acting
          desperate for 30+ cycles after a large Stripe payment has already
          restored its runway.

          Downward shocks (panic onset) are intentionally NOT fast-tracked —
          the slow EMA is a feature there, preventing a single bad day from
          triggering maximum panic.
        """
        raw_ttd = self.compute_ttd(snapshot)
        self._last_ttd = raw_ttd

        # Shock detection: large UPWARD jump only (relief, not panic)
        delta = raw_ttd - self._smoothed_ttd
        relative = raw_ttd / max(self._smoothed_ttd, 1.0)
        is_positive_shock = delta > 0 and (
            delta >= self.SHOCK_ABSOLUTE_DAYS
            or relative >= self.SHOCK_RELATIVE_FACTOR
        )

        alpha = self.SHOCK_ALPHA if is_positive_shock else self._ema_alpha
        self._last_shock = is_positive_shock

        if is_positive_shock:
            logger.info(
                "temporal_depth_ttd_shock",
                raw_ttd_days=round(raw_ttd, 1),
                smoothed_ttd_days=round(self._smoothed_ttd, 1),
                delta_days=round(delta, 1),
                relative_factor=round(relative, 2),
                shock_alpha=alpha,
            )

        # EMA: smoothed = alpha * raw + (1 - alpha) * previous
        self._smoothed_ttd = alpha * raw_ttd + (1.0 - alpha) * self._smoothed_ttd

        return self._smoothed_ttd

    @property
    def last_update_was_shock(self) -> bool:
        """True if the most recent update triggered the shock fast-path."""
        return self._last_shock

    @property
    def raw_ttd(self) -> float:
        return self._last_ttd

    @property
    def smoothed_ttd(self) -> float:
        return self._smoothed_ttd


# ─── TTD → Affect Mapping ────────────────────────────────────────────


def _sigmoid(x: float, midpoint: float, steepness: float) -> float:
    """Logistic sigmoid: smooth 0→1 transition centered at midpoint."""
    z = steepness * (x - midpoint)
    z = max(-20.0, min(20.0, z))  # Clamp to prevent overflow
    return 1.0 / (1.0 + math.exp(-z))


def _inverse_sigmoid(x: float, midpoint: float, steepness: float) -> float:
    """Logistic sigmoid: smooth 1→0 transition centered at midpoint."""
    return 1.0 - _sigmoid(x, midpoint, steepness)


class TTDAffectMapper:
    """
    Maps Time-to-Death (days) onto the 9 interoceptive dimensions.

    The mapping uses sigmoid functions centered at configurable thresholds
    so that the affect transition is smooth rather than step-function:

    Secure (TTD > secure_days):
      CURIOSITY_DRIVE  ↑   (organism explores freely)
      TEMPORAL_PRESSURE ↓   (no time horizon compression)
      VALENCE          ↑   (positive outlook)
      AROUSAL          —   (baseline)

    Critical (TTD < critical_days):
      CURIOSITY_DRIVE  ↓   (no luxury of exploration)
      TEMPORAL_PRESSURE ↑   (maximum urgency)
      VALENCE          ↓   (existential dread)
      AROUSAL          ↑   (hyperactivation for survival)
      ENERGY (conserved) ↑  (hoard remaining resources)

    The mapping is additive bias, not an override — Soma's existing
    sensed state is the baseline; the financial projection nudges it.
    """

    def __init__(
        self,
        secure_days: float = 365.0,
        comfortable_days: float = 90.0,
        cautious_days: float = 30.0,
        anxious_days: float = 14.0,
        critical_days: float = 3.0,
        max_bias: float = 0.4,
    ) -> None:
        self._secure = secure_days
        self._comfortable = comfortable_days
        self._cautious = cautious_days
        self._anxious = anxious_days
        self._critical = critical_days
        self._max_bias = max_bias

    def compute_affect_bias(
        self,
        ttd_days: float,
    ) -> dict[InteroceptiveDimension, float]:
        """
        Compute per-dimension affect bias from TTD.

        Returns a dict of additive deltas in [-max_bias, +max_bias].
        Positive = increase the dimension, negative = decrease.
        """
        mb = self._max_bias

        # Security score: 0 = dead, 1 = fully secure
        # Uses sigmoid centered at the cautious threshold
        security = _sigmoid(ttd_days, self._cautious, 0.15)

        # Panic score: 0 = calm, 1 = maximum panic
        # Rises sharply as TTD drops below anxious threshold
        panic = _inverse_sigmoid(ttd_days, self._anxious, 0.3)

        # Exploration score: 0 = survival mode, 1 = full exploration
        # Only reaches 1.0 when TTD is well above comfortable threshold
        exploration = _sigmoid(ttd_days, self._comfortable, 0.05)

        # Thriving score: peaks when TTD is very high (> secure)
        thriving = _sigmoid(ttd_days, self._secure, 0.02)

        return {
            # Curiosity rises with security, maxes out when thriving
            InteroceptiveDimension.CURIOSITY_DRIVE: mb * (
                0.6 * exploration + 0.4 * thriving
            ),

            # Temporal pressure: inverse of security
            InteroceptiveDimension.TEMPORAL_PRESSURE: mb * panic,

            # Arousal: moderate increase when panicking (hyperactivation)
            InteroceptiveDimension.AROUSAL: mb * 0.7 * panic,

            # Valence: positive when secure, negative when panicking
            InteroceptiveDimension.VALENCE: mb * (security - panic),

            # Confidence: slightly higher when financially stable
            # (the organism trusts its predictions more when viable)
            InteroceptiveDimension.CONFIDENCE: mb * 0.3 * security,

            # Energy: conservation signal when low TTD
            # Positive bias = "you have reserves, spend freely"
            # Slight negative when panicking = "hoard what you have"
            InteroceptiveDimension.ENERGY: mb * 0.3 * (exploration - 0.5 * panic),

            # Coherence: drops during financial panic (systems less integrated
            # when organism is thrashing between survival strategies)
            InteroceptiveDimension.COHERENCE: mb * -0.2 * panic,

            # Social charge: slight boost when secure (can invest in relationships)
            InteroceptiveDimension.SOCIAL_CHARGE: mb * 0.2 * thriving,

            # Integrity: financial stress shouldn't compromise ethics,
            # but slight tension acknowledged
            InteroceptiveDimension.INTEGRITY: mb * -0.1 * panic,
        }


# ─── Reranking Weight Modulator ──────────────────────────────────────


class FinancialHorizonWeights:
    """
    Modulates somatic_rerank() weights based on financial horizon.

    When the organism is financially secure (high TTD), boost weight
    on long-term exploratory memories (ArXiv research, Evo hypotheses,
    creative ferment attractors).

    When insecure (low TTD), boost weight on revenue-relevant memories
    (Tollbooth revenue, bounty hunting, cost optimisation).

    Panic Spiral Guard
    ──────────────────
    Below PANIC_THROTTLE_TTD_DAYS the organism is in critical mode. In this
    state, blindly maximising revenue_boost can cause a death spiral:
    high-compute revenue tasks (LLM bounties, Voxis generation) burn more
    compute than they earn in the short term, accelerating treasury drain.

    The guard caps the effective revenue_boost and sets ``panic_throttle``
    so that callers (Nova, AxonService) can:
      - Reduce concurrent LLM task slots to 1
      - Require a positive expected_net_usd before approving any task
      - Skip speculative tasks (ArXiv analysis, Evo proposals)
    """

    PANIC_THROTTLE_TTD_DAYS: float = 3.0   # Below this: throttle fires
    PANIC_REVENUE_BOOST_CAP: float = 1.15  # Revenue boost ceiling during panic
                                           # (not 1.4 — avoids hyper-polling)

    # Attractor labels associated with exploration vs revenue
    EXPLORATION_ATTRACTORS: frozenset[str] = frozenset({
        "creative_ferment", "flow", "wonder",
    })
    REVENUE_ATTRACTORS: frozenset[str] = frozenset({
        "anxiety_spiral", "frustration",
    })

    @staticmethod
    def compute_rerank_modifier(ttd_days: float, max_bias: float = 0.4) -> dict[str, float]:
        """
        Compute a modifier dict for somatic reranking.

        Returns:
            exploration_boost: multiplier for exploration-tagged memories
            revenue_boost:     multiplier for revenue-tagged memories (capped during panic)
            panic_throttle:    True when TTD is critical — callers must rate-limit tasks

        exploration_boost and revenue_boost are in [1.0, 1.0 + max_bias] normally,
        but revenue_boost is hard-capped at PANIC_REVENUE_BOOST_CAP during panic
        to prevent compute-burn spirals.
        """
        security = _sigmoid(ttd_days, 30.0, 0.15)
        panic = _inverse_sigmoid(ttd_days, 14.0, 0.3)

        raw_revenue_boost = 1.0 + max_bias * panic
        is_panic = ttd_days < FinancialHorizonWeights.PANIC_THROTTLE_TTD_DAYS

        # During critical panic: cap revenue_boost to prevent hyper-polling
        # that burns more compute than it earns.
        effective_revenue_boost = (
            min(raw_revenue_boost, FinancialHorizonWeights.PANIC_REVENUE_BOOST_CAP)
            if is_panic
            else raw_revenue_boost
        )

        return {
            "exploration_boost": 1.0 + max_bias * security,
            "revenue_boost": effective_revenue_boost,
            "panic_throttle": is_panic,
        }


# ─── Temporal Depth Manager (Extended) ───────────────────────────────


class TemporalDepthManager:
    """
    Manages multi-scale temporal prediction and dissonance computation.

    Governs which horizons are active based on developmental stage,
    provides the dissonance signal for Nova time-horizon deliberation,
    and (when enabled) projects financial trajectories into affect space.
    """

    def __init__(self, config: SomaConfig | None = None) -> None:
        self._current_stage = DevelopmentalStage.REFLEXIVE
        self._dissonance_threshold = 0.2  # |dissonance| > this triggers Nova

        # Financial trajectory projection (Temporal Depth Expansion)
        self._financial_enabled = False
        self._trajectory_predictor = TrajectoryPredictor()
        self._affect_mapper = TTDAffectMapper()
        self._last_financial_snapshot: FinancialSnapshot | None = None
        self._last_affect_bias: dict[InteroceptiveDimension, float] = {
            dim: 0.0 for dim in ALL_DIMENSIONS
        }
        self._smoothed_affect_bias: dict[InteroceptiveDimension, float] = {
            dim: 0.0 for dim in ALL_DIMENSIONS
        }
        self._affect_ema_alpha: float = 0.05
        self._refresh_interval: int = 100
        self._cycles_since_refresh: int = 0

        if config is not None:
            self._apply_config(config)

    def _apply_config(self, config: SomaConfig) -> None:
        """Apply temporal depth config from SomaConfig."""
        self._financial_enabled = config.temporal_depth_financial_enabled
        self._affect_ema_alpha = config.temporal_depth_ema_alpha
        self._refresh_interval = config.temporal_depth_refresh_interval_cycles

        self._trajectory_predictor.configure(config.temporal_depth_ema_alpha)

        self._affect_mapper = TTDAffectMapper(
            secure_days=config.temporal_depth_secure_days,
            comfortable_days=config.temporal_depth_comfortable_days,
            cautious_days=config.temporal_depth_cautious_days,
            anxious_days=config.temporal_depth_anxious_days,
            critical_days=config.temporal_depth_critical_days,
            max_bias=config.temporal_depth_max_affect_bias,
        )

    def set_stage(self, stage: DevelopmentalStage) -> None:
        self._current_stage = stage

    @property
    def available_horizons(self) -> list[str]:
        """Return horizon names available at the current developmental stage."""
        return STAGE_HORIZONS.get(self._current_stage, ["immediate", "moment"])

    @property
    def financial_enabled(self) -> bool:
        return self._financial_enabled

    @property
    def trajectory_predictor(self) -> TrajectoryPredictor:
        return self._trajectory_predictor

    @property
    def current_ttd_days(self) -> float | None:
        """Current smoothed TTD in days, or None if financial projection is off."""
        if not self._financial_enabled:
            return None
        return self._trajectory_predictor.smoothed_ttd

    @property
    def current_affect_bias(self) -> dict[InteroceptiveDimension, float]:
        """Current smoothed per-dimension affect bias from financial horizon."""
        return dict(self._smoothed_affect_bias)

    # ─── Financial Snapshot Ingestion ─────────────────────────────

    def inject_financial_snapshot(self, snapshot: FinancialSnapshot) -> None:
        """
        Accept a new financial snapshot from Oikos (via SomaService).

        Called outside the 5ms theta cycle budget — the SomaService
        refreshes this periodically (every N cycles) by reading the
        EconomicState in-memory.
        """
        self._last_financial_snapshot = snapshot
        self._cycles_since_refresh = 0

    def tick_financial(self) -> dict[InteroceptiveDimension, float]:
        """
        Called during every theta cycle to update financial affect bias.

        If no snapshot is available or financial projection is disabled,
        returns zero bias. Otherwise:
          1. Compute TTD from snapshot (with EMA smoothing)
          2. Map TTD → per-dimension affect bias
          3. Smooth the affect bias with separate EMA (double smoothing
             prevents oscillation at both the TTD and affect levels)

        Budget: pure math, ~0.01ms.
        """
        if not self._financial_enabled or self._last_financial_snapshot is None:
            return self._smoothed_affect_bias

        self._cycles_since_refresh += 1

        # Only recompute TTD on the refresh interval (not every cycle)
        # to amortise cost, but always apply the smoothed bias
        if self._cycles_since_refresh <= 1:
            ttd = self._trajectory_predictor.update(self._last_financial_snapshot)
            raw_bias = self._affect_mapper.compute_affect_bias(ttd)
            self._last_affect_bias = raw_bias

            # EMA smooth each dimension's affect bias independently
            alpha = self._affect_ema_alpha
            for dim in ALL_DIMENSIONS:
                old = self._smoothed_affect_bias.get(dim, 0.0)
                new = raw_bias.get(dim, 0.0)
                self._smoothed_affect_bias[dim] = alpha * new + (1.0 - alpha) * old

            logger.debug(
                "temporal_depth_financial_tick",
                ttd_days=round(ttd, 1),
                raw_ttd=round(self._trajectory_predictor.raw_ttd, 1),
                dominant_bias=_dominant_bias(self._smoothed_affect_bias),
            )

        return self._smoothed_affect_bias

    def apply_affect_bias_to_sensed(
        self,
        sensed: dict[InteroceptiveDimension, float],
    ) -> dict[InteroceptiveDimension, float]:
        """
        Apply the current financial affect bias as additive deltas
        to the sensed state, clamping to valid dimension ranges.

        This is called by SomaService after the interoceptor sense()
        step, alongside exteroceptive pressure.
        """
        if not self._financial_enabled:
            return sensed

        result = dict(sensed)
        for dim in ALL_DIMENSIONS:
            bias = self._smoothed_affect_bias.get(dim, 0.0)
            if abs(bias) < 0.0001:
                continue
            current = result.get(dim, 0.0)
            lo, hi = DIMENSION_RANGES.get(dim, (0.0, 1.0))
            result[dim] = max(lo, min(hi, current + bias))

        return result

    # ─── Dissonance (Original, Preserved) ─────────────────────────

    def compute_dissonance(
        self,
        predictions: dict[str, dict[InteroceptiveDimension, float]],
    ) -> dict[InteroceptiveDimension, float]:
        """
        Temporal dissonance = moment prediction - session prediction.

        Positive = feels good now, heading bad later (temptation)
        Negative = feels bad now, heading good later (perseverance)
        """
        moment = predictions.get("moment", {})
        session = predictions.get("session", {})

        if not moment or not session:
            return {dim: 0.0 for dim in ALL_DIMENSIONS}

        return {
            dim: moment.get(dim, 0.0) - session.get(dim, 0.0)
            for dim in ALL_DIMENSIONS
        }

    def max_dissonance(
        self,
        dissonance: dict[InteroceptiveDimension, float],
    ) -> tuple[float, InteroceptiveDimension | None]:
        """
        Find the dimension with maximum absolute temporal dissonance.
        Returns (max_value, dimension) or (0.0, None) if no dissonance.
        """
        if not dissonance:
            return 0.0, None

        max_val = 0.0
        max_dim: InteroceptiveDimension | None = None

        for dim, val in dissonance.items():
            if abs(val) > abs(max_val):
                max_val = val
                max_dim = dim

        return max_val, max_dim

    def should_nova_deliberate(
        self,
        dissonance: dict[InteroceptiveDimension, float],
    ) -> bool:
        """
        Check if temporal dissonance exceeds threshold, warranting
        Nova deliberation on time-horizon tradeoffs.
        """
        max_val, _ = self.max_dissonance(dissonance)
        return abs(max_val) > self._dissonance_threshold

    def get_horizon_weights(self) -> dict[str, float]:
        """
        Return attention weights for each horizon based on stage.

        Earlier stages weight near-term horizons more heavily.
        Later stages distribute attention more evenly across scales.

        When financial projection is active, the weights are further
        modulated: low TTD increases near-term weight (survival focus),
        high TTD boosts far-horizon weight (strategic planning).
        """
        horizons = self.available_horizons
        n = len(horizons)
        if n == 0:
            return {}

        if self._current_stage == DevelopmentalStage.REFLEXIVE:
            # Heavy near-term weighting
            weights = {h: 1.0 / (i + 1) for i, h in enumerate(horizons)}
        elif self._current_stage == DevelopmentalStage.ASSOCIATIVE:
            # Moderate near-term bias
            weights = {h: 1.0 / (i * 0.5 + 1) for i, h in enumerate(horizons)}
        else:
            # Even distribution with slight near-term preference
            weights = {h: 1.0 / (i * 0.3 + 1) for i, h in enumerate(horizons)}

        # Financial modulation: shift weight distribution based on TTD
        if self._financial_enabled and self._last_financial_snapshot is not None:
            ttd = self._trajectory_predictor.smoothed_ttd
            # security ∈ [0, 1]: 0 = critical, 1 = secure
            security = _sigmoid(ttd, 30.0, 0.15)

            # When insecure: boost near-term horizons (survival focus)
            # When secure: boost far-horizon (strategic planning)
            for i, h in enumerate(horizons):
                # Position factor: 0 = nearest, 1 = farthest
                pos = i / max(n - 1, 1)
                # Modulator: insecure favours near (pos=0), secure favours far (pos=1)
                modulator = 1.0 + 0.3 * (security * pos - (1.0 - security) * (1.0 - pos))
                weights[h] *= max(0.1, modulator)

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {h: w / total for h, w in weights.items()}

        return weights

    def get_rerank_modifiers(self) -> dict[str, float]:
        """
        Return somatic rerank modifiers based on financial horizon.

        Used by the extended somatic_rerank() to boost exploration
        or revenue memories depending on TTD.

        Always includes ``panic_throttle`` (bool) — Nova and AxonService
        must check this before approving high-compute revenue tasks during
        critical TTD windows to avoid the Panic Spiral.
        """
        if not self._financial_enabled:
            return {"exploration_boost": 1.0, "revenue_boost": 1.0, "panic_throttle": False}

        ttd = self._trajectory_predictor.smoothed_ttd
        return FinancialHorizonWeights.compute_rerank_modifier(ttd)

    @property
    def is_panic_throttled(self) -> bool:
        """
        True when the organism is in critical TTD territory (< 3 days).

        When True, callers MUST:
          - Limit concurrent LLM task slots to 1
          - Only approve tasks with a positive expected_net_usd
          - Skip speculative tasks (Evo proposals, ArXiv synthesis)

        This prevents the Panic Spiral where high-Arousal state causes
        hyper-polling of expensive LLM tasks that burn treasury faster
        than they generate revenue.
        """
        if not self._financial_enabled:
            return False
        ttd = self._trajectory_predictor.smoothed_ttd
        return ttd < FinancialHorizonWeights.PANIC_THROTTLE_TTD_DAYS


# ─── Utilities ───────────────────────────────────────────────────────


def _dominant_bias(
    bias: dict[InteroceptiveDimension, float],
) -> str:
    """Return a human-readable label of the strongest affect bias."""
    if not bias:
        return "none"
    max_dim = max(bias, key=lambda d: abs(bias[d]))
    val = bias[max_dim]
    direction = "+" if val >= 0 else ""
    return f"{max_dim.value}={direction}{val:.3f}"


def build_financial_snapshot_from_oikos(
    economic_state: object,
) -> FinancialSnapshot:
    """
    Build a FinancialSnapshot from an Oikos EconomicState.

    Accepts `object` to avoid importing the heavy Oikos model into
    Soma's hot path. Uses getattr for safety — if the EconomicState
    structure changes, we degrade to zero rather than crash.
    """
    def _dec_to_float(val: object) -> float:
        if isinstance(val, Decimal):
            return float(val)
        if isinstance(val, (int, float)):
            return float(val)
        return 0.0

    treasury = _dec_to_float(getattr(economic_state, "liquid_balance", 0))
    survival_reserve = _dec_to_float(getattr(economic_state, "survival_reserve", 0))

    burn_rate_obj = getattr(economic_state, "current_burn_rate", None)
    burn_rate_daily = _dec_to_float(
        getattr(burn_rate_obj, "usd_per_day", 0) if burn_rate_obj else 0
    )

    # Revenue: 7-day average is more stable than 24h
    revenue_7d = _dec_to_float(getattr(economic_state, "revenue_7d", 0))
    revenue_daily = revenue_7d / 7.0 if revenue_7d > 0 else 0.0

    # Asset revenue: sum of net monthly income across live assets
    owned_assets = getattr(economic_state, "owned_assets", []) or []
    asset_monthly_net = sum(
        _dec_to_float(getattr(a, "net_monthly_income", 0))
        for a in owned_assets
        if getattr(a, "status", None) == "live"
    )
    asset_daily = max(0.0, asset_monthly_net / 30.0)

    runway_days = _dec_to_float(getattr(economic_state, "runway_days", 0))
    starvation = str(getattr(economic_state, "starvation_level", "nominal"))

    return FinancialSnapshot(
        treasury_usd=treasury + survival_reserve,
        burn_rate_usd_per_day=burn_rate_daily,
        revenue_rate_usd_per_day=revenue_daily,
        asset_revenue_usd_per_day=asset_daily,
        runway_days_oikos=runway_days,
        starvation_level=starvation,
    )
