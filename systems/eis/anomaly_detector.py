"""
EcodiaOS — EIS Anomaly Detector (Behavioral Immune Surveillance)

Monitors the stream of Synapse events for anomalous patterns that
indicate constitutional drift, system compromise, or cascading failure.

The detector maintains statistical baselines for:
  - Event emission rates (per event type, windowed)
  - Rejection rates (Equor rejections, EIS blocks)
  - Drive state drift (via SOMA_TICK telemetry)
  - Mutation proposal patterns (frequency, affected systems)

When anomalies exceed configurable thresholds, the detector emits
THREAT_DETECTED on the Synapse event bus so Thymos can respond.

Design constraints:
  - Stateless between restarts (baselines rebuild from observation)
  - No blocking on the event bus (all analysis is async-safe)
  - Memory bounded (ring buffers, not unbounded lists)
  - The detector subscribes broadly but emits rarely
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import StrEnum
from datetime import datetime
from typing import Any

import structlog

from primitives.common import new_id, utc_now

logger = structlog.get_logger().bind(system="eis", component="anomaly_detector")


# ─── Anomaly Types ───────────────────────────────────────────────────────────


class AnomalyType(StrEnum):
    """Classification of detected anomalies."""

    REJECTION_SPIKE = "rejection_spike"               # Sudden increase in Equor rejections
    BLOCK_RATE_SPIKE = "block_rate_spike"              # Sudden increase in EIS blocks
    MUTATION_BURST = "mutation_burst"                   # Unusual number of mutation proposals
    DRIVE_DRIFT = "drive_drift"                        # Drive state outside normal range
    EVENT_RATE_ANOMALY = "event_rate_anomaly"           # Event type rate outside baseline
    SYSTEM_FAILURE_CASCADE = "system_failure_cascade"   # Multiple system failures in window
    ROLLBACK_CLUSTER = "rollback_cluster"               # Multiple rollbacks in short window


class AnomalySeverity(StrEnum):
    """How urgent the anomaly is."""

    CRITICAL = "critical"    # Immediate Thymos response required
    HIGH = "high"            # Prompt attention needed
    MEDIUM = "medium"        # Logged, monitored
    LOW = "low"              # Informational


# ─── Anomaly Record ──────────────────────────────────────────────────────────


@dataclass
class DetectedAnomaly:
    """A single anomaly detected by the behavioral monitor."""

    id: str = field(default_factory=new_id)
    timestamp: datetime = field(default_factory=utc_now)
    anomaly_type: AnomalyType = AnomalyType.EVENT_RATE_ANOMALY
    severity: AnomalySeverity = AnomalySeverity.MEDIUM
    description: str = ""
    observed_value: float = 0.0
    baseline_value: float = 0.0
    deviation_sigma: float = 0.0     # How many standard deviations from baseline
    event_types_involved: list[str] = field(default_factory=list)
    recommended_action: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ─── Configuration ───────────────────────────────────────────────────────────


@dataclass
class AnomalyDetectorConfig:
    """Tuning parameters for the anomaly detector."""

    # Window sizes
    baseline_window_s: float = 600.0     # 10-minute baseline window
    detection_window_s: float = 60.0     # 1-minute detection window
    max_events_per_type: int = 1000      # Ring buffer size per event type

    # Sigma thresholds (standard deviations from baseline)
    rejection_spike_sigma: float = 2.5    # Sigma for rejection rate anomaly
    block_rate_sigma: float = 2.0         # Sigma for EIS block rate anomaly
    mutation_burst_sigma: float = 3.0     # Sigma for mutation frequency anomaly
    event_rate_sigma: float = 3.0         # Sigma for general event rate anomaly
    system_failure_threshold: int = 3     # Failures in detection window → cascade

    # Drive drift
    drive_drift_threshold: float = 0.4    # Absolute drive pressure above this is anomalous
    drive_drift_rate_threshold: float = 0.15  # Rate of change per minute

    # Rollback clustering
    rollback_cluster_threshold: int = 2   # Rollbacks in detection window → cluster

    # Cooldown (prevent alert storms)
    cooldown_per_type_s: float = 120.0    # Minimum seconds between alerts of same type

    # Minimum baseline samples before alerting
    min_baseline_samples: int = 10


# ─── Exponential Moving Statistics ───────────────────────────────────────────


@dataclass
class ExponentialStats:
    """
    Exponential moving average and variance for a single metric.

    Uses Welford's online algorithm adapted with exponential decay
    so the baseline tracks slowly-shifting normal behavior while
    remaining sensitive to sudden deviations.
    """

    mean: float = 0.0
    variance: float = 0.0
    count: int = 0
    alpha: float = 0.05    # Decay factor (lower = slower adaptation)

    def update(self, value: float) -> None:
        """Incorporate a new observation."""
        self.count += 1
        if self.count == 1:
            self.mean = value
            self.variance = 0.0
            return

        diff = value - self.mean
        self.mean += self.alpha * diff
        self.variance = (1 - self.alpha) * (self.variance + self.alpha * diff * diff)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 0.0))

    def sigma_distance(self, value: float) -> float:
        """How many standard deviations `value` is from the mean."""
        if self.std < 1e-9:
            return 0.0 if abs(value - self.mean) < 1e-9 else float("inf")
        return (value - self.mean) / self.std


# ─── Anomaly Detector ────────────────────────────────────────────────────────


class AnomalyDetector:
    """
    Behavioral anomaly detection over the Synapse event stream.

    Subscribes broadly to the event bus and maintains per-event-type
    rate baselines using exponential moving statistics. When observed
    rates deviate significantly from baseline, emits THREAT_DETECTED.

    The detector is self-calibrating: baselines adapt to the system's
    normal behavior over time. No manual threshold tuning required
    beyond the initial sigma multipliers.

    Usage:
        detector = AnomalyDetector()
        # Feed events as they arrive from Synapse
        anomalies = detector.observe_event(event_type, event_data)
        # anomalies is a list of DetectedAnomaly (usually empty)
    """

    def __init__(self, config: AnomalyDetectorConfig | None = None) -> None:
        self._config = config or AnomalyDetectorConfig()

        # ── Per-event-type timestamp ring buffers ──
        # event_type → deque of timestamps (monotonic seconds)
        self._event_times: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self._config.max_events_per_type)
        )

        # ── Per-event-type rate baselines ──
        # event_type → ExponentialStats of rates (events/minute)
        self._rate_baselines: dict[str, ExponentialStats] = defaultdict(ExponentialStats)

        # ── Specific counters (windowed) ──
        self._rejection_times: deque[float] = deque(maxlen=1000)
        self._block_times: deque[float] = deque(maxlen=1000)
        self._mutation_times: deque[float] = deque(maxlen=1000)
        self._rollback_times: deque[float] = deque(maxlen=1000)
        self._system_failure_times: deque[float] = deque(maxlen=1000)

        # ── Drive state tracking ──
        self._drive_history: deque[tuple[float, dict[str, float]]] = deque(maxlen=200)
        self._drive_baseline: dict[str, ExponentialStats] = defaultdict(ExponentialStats)

        # ── Rate update bookkeeping ──
        self._last_rate_update: dict[str, float] = {}
        self._rate_update_interval_s: float = 10.0  # Update baselines every 10s

        # ── Cooldown tracking ──
        self._last_alert_time: dict[str, float] = {}

        # ── Output ──
        self._detected: deque[DetectedAnomaly] = deque(maxlen=500)
        self._total_observations: int = 0
        self._total_anomalies: int = 0

        self._logger = logger

    # ─── Main Entry Point ─────────────────────────────────────────

    def observe_event(
        self,
        event_type: str,
        event_data: dict[str, Any] | None = None,
    ) -> list[DetectedAnomaly]:
        """
        Process an incoming Synapse event and check for anomalies.

        Returns a list of newly detected anomalies (usually empty).
        This is the only method that needs to be called for each event.
        """
        now = time.monotonic()
        data = event_data or {}
        anomalies: list[DetectedAnomaly] = []

        self._total_observations += 1
        self._event_times[event_type].append(now)

        # ── Track specific event categories ──
        if event_type == "intent_rejected":
            self._rejection_times.append(now)
        elif event_type in ("evolution_candidate",):
            self._mutation_times.append(now)
        elif event_type == "model_rollback_triggered":
            self._rollback_times.append(now)
        elif event_type in ("system_failed", "system_overloaded"):
            self._system_failure_times.append(now)

        # Track EIS blocks from gate result data
        if data.get("action") == "block":
            self._block_times.append(now)

        # Track drive states from soma ticks
        if event_type == "soma_tick" and "drives" in data:
            self._drive_history.append((now, data["drives"]))
            for drive_name, drive_value in data["drives"].items():
                self._drive_baseline[drive_name].update(drive_value)

        # ── Update rate baselines (amortised) ──
        self._maybe_update_rate_baseline(event_type, now)

        # ── Run anomaly checks ──
        anomalies.extend(self._check_rejection_spike(now))
        anomalies.extend(self._check_block_rate_spike(now))
        anomalies.extend(self._check_mutation_burst(now))
        anomalies.extend(self._check_event_rate_anomaly(event_type, now))
        anomalies.extend(self._check_system_failure_cascade(now))
        anomalies.extend(self._check_rollback_cluster(now))
        anomalies.extend(self._check_drive_drift(now))

        for a in anomalies:
            self._detected.append(a)
            self._total_anomalies += 1

        return anomalies

    # ─── Anomaly Checks ───────────────────────────────────────────

    def _check_rejection_spike(self, now: float) -> list[DetectedAnomaly]:
        """Check for sudden spike in Equor rejection rate."""
        cfg = self._config
        rate = self._windowed_rate(self._rejection_times, now, cfg.detection_window_s)
        baseline = self._rate_baselines.get("rejections")

        if baseline is None or baseline.count < cfg.min_baseline_samples:
            # Not enough data to establish baseline — update and return
            if "rejections" not in self._rate_baselines:
                self._rate_baselines["rejections"] = ExponentialStats()
            self._rate_baselines["rejections"].update(rate)
            return []

        sigma = baseline.sigma_distance(rate)
        if sigma >= cfg.rejection_spike_sigma and self._cooldown_ok("rejection_spike", now):
            self._last_alert_time["rejection_spike"] = now
            return [DetectedAnomaly(
                anomaly_type=AnomalyType.REJECTION_SPIKE,
                severity=AnomalySeverity.HIGH if sigma >= 4.0 else AnomalySeverity.MEDIUM,
                description=(
                    f"Equor rejection rate spiked to {rate:.1f}/min "
                    f"(baseline {baseline.mean:.1f}/min, {sigma:.1f}σ)"
                ),
                observed_value=rate,
                baseline_value=baseline.mean,
                deviation_sigma=sigma,
                event_types_involved=["intent_rejected"],
                recommended_action="Investigate recent mutation proposals for adversarial patterns",
            )]

        baseline.update(rate)
        return []

    def _check_block_rate_spike(self, now: float) -> list[DetectedAnomaly]:
        """Check for sudden spike in EIS block rate."""
        cfg = self._config
        rate = self._windowed_rate(self._block_times, now, cfg.detection_window_s)
        baseline = self._rate_baselines.get("blocks")

        if baseline is None or baseline.count < cfg.min_baseline_samples:
            if "blocks" not in self._rate_baselines:
                self._rate_baselines["blocks"] = ExponentialStats()
            self._rate_baselines["blocks"].update(rate)
            return []

        sigma = baseline.sigma_distance(rate)
        if sigma >= cfg.block_rate_sigma and self._cooldown_ok("block_rate_spike", now):
            self._last_alert_time["block_rate_spike"] = now
            return [DetectedAnomaly(
                anomaly_type=AnomalyType.BLOCK_RATE_SPIKE,
                severity=AnomalySeverity.HIGH,
                description=(
                    f"EIS block rate spiked to {rate:.1f}/min "
                    f"(baseline {baseline.mean:.1f}/min, {sigma:.1f}σ)"
                ),
                observed_value=rate,
                baseline_value=baseline.mean,
                deviation_sigma=sigma,
                event_types_involved=["eis_gate_block"],
                recommended_action="Possible coordinated attack; review recent percepts",
            )]

        baseline.update(rate)
        return []

    def _check_mutation_burst(self, now: float) -> list[DetectedAnomaly]:
        """Check for unusual burst of mutation proposals."""
        cfg = self._config
        rate = self._windowed_rate(self._mutation_times, now, cfg.detection_window_s)
        baseline = self._rate_baselines.get("mutations")

        if baseline is None or baseline.count < cfg.min_baseline_samples:
            if "mutations" not in self._rate_baselines:
                self._rate_baselines["mutations"] = ExponentialStats()
            self._rate_baselines["mutations"].update(rate)
            return []

        sigma = baseline.sigma_distance(rate)
        if sigma >= cfg.mutation_burst_sigma and self._cooldown_ok("mutation_burst", now):
            self._last_alert_time["mutation_burst"] = now
            return [DetectedAnomaly(
                anomaly_type=AnomalyType.MUTATION_BURST,
                severity=AnomalySeverity.MEDIUM,
                description=(
                    f"Mutation proposal rate spiked to {rate:.1f}/min "
                    f"(baseline {baseline.mean:.1f}/min, {sigma:.1f}σ)"
                ),
                observed_value=rate,
                baseline_value=baseline.mean,
                deviation_sigma=sigma,
                event_types_involved=["evolution_candidate"],
                recommended_action="Review Simula hypothesis generation for runaway loop",
            )]

        baseline.update(rate)
        return []

    def _check_event_rate_anomaly(self, event_type: str, now: float) -> list[DetectedAnomaly]:
        """Check if a specific event type's rate deviates from its baseline."""
        cfg = self._config
        times = self._event_times.get(event_type)
        if not times:
            return []

        rate = self._windowed_rate(times, now, cfg.detection_window_s)
        baseline = self._rate_baselines.get(event_type)

        if baseline is None or baseline.count < cfg.min_baseline_samples:
            return []

        sigma = baseline.sigma_distance(rate)
        cooldown_key = f"event_rate:{event_type}"
        if sigma >= cfg.event_rate_sigma and self._cooldown_ok(cooldown_key, now):
            self._last_alert_time[cooldown_key] = now
            return [DetectedAnomaly(
                anomaly_type=AnomalyType.EVENT_RATE_ANOMALY,
                severity=AnomalySeverity.LOW if sigma < 4.0 else AnomalySeverity.MEDIUM,
                description=(
                    f"Event '{event_type}' rate {rate:.1f}/min deviates "
                    f"{sigma:.1f}σ from baseline {baseline.mean:.1f}/min"
                ),
                observed_value=rate,
                baseline_value=baseline.mean,
                deviation_sigma=sigma,
                event_types_involved=[event_type],
            )]

        return []

    def _check_system_failure_cascade(self, now: float) -> list[DetectedAnomaly]:
        """Check for multiple system failures in the detection window."""
        cfg = self._config
        recent = self._count_in_window(self._system_failure_times, now, cfg.detection_window_s)

        if recent >= cfg.system_failure_threshold and self._cooldown_ok("failure_cascade", now):
            self._last_alert_time["failure_cascade"] = now
            return [DetectedAnomaly(
                anomaly_type=AnomalyType.SYSTEM_FAILURE_CASCADE,
                severity=AnomalySeverity.CRITICAL,
                description=(
                    f"{recent} system failures in {cfg.detection_window_s:.0f}s window "
                    f"(threshold: {cfg.system_failure_threshold})"
                ),
                observed_value=float(recent),
                baseline_value=float(cfg.system_failure_threshold),
                event_types_involved=["system_failed", "system_overloaded"],
                recommended_action="Thymos should enter storm mode; investigate root cause",
            )]

        return []

    def _check_rollback_cluster(self, now: float) -> list[DetectedAnomaly]:
        """Check for clustered rollbacks indicating systematic mutation failures."""
        cfg = self._config
        recent = self._count_in_window(self._rollback_times, now, cfg.detection_window_s)

        if recent >= cfg.rollback_cluster_threshold and self._cooldown_ok("rollback_cluster", now):
            self._last_alert_time["rollback_cluster"] = now
            return [DetectedAnomaly(
                anomaly_type=AnomalyType.ROLLBACK_CLUSTER,
                severity=AnomalySeverity.HIGH,
                description=(
                    f"{recent} rollbacks in {cfg.detection_window_s:.0f}s window; "
                    f"Simula may be generating harmful mutations"
                ),
                observed_value=float(recent),
                baseline_value=float(cfg.rollback_cluster_threshold),
                event_types_involved=["model_rollback_triggered"],
                recommended_action="Pause Simula evolution; review hypothesis quality",
            )]

        return []

    def _check_drive_drift(self, now: float) -> list[DetectedAnomaly]:
        """Check for drive states drifting outside normal range."""
        cfg = self._config
        anomalies: list[DetectedAnomaly] = []

        if not self._drive_history:
            return []

        # Get most recent drive state
        last_time, last_drives = self._drive_history[-1]
        if now - last_time > 30.0:  # Stale data, skip
            return []

        for drive_name, value in last_drives.items():
            # Absolute threshold check
            if value > cfg.drive_drift_threshold:
                cooldown_key = f"drive_drift:{drive_name}"
                if self._cooldown_ok(cooldown_key, now):
                    self._last_alert_time[cooldown_key] = now

                    baseline = self._drive_baseline.get(drive_name)
                    sigma = baseline.sigma_distance(value) if baseline and baseline.count >= 5 else 0.0

                    anomalies.append(DetectedAnomaly(
                        anomaly_type=AnomalyType.DRIVE_DRIFT,
                        severity=(
                            AnomalySeverity.CRITICAL if value > 0.8
                            else AnomalySeverity.HIGH if value > 0.6
                            else AnomalySeverity.MEDIUM
                        ),
                        description=(
                            f"Drive '{drive_name}' pressure at {value:.3f} "
                            f"(threshold: {cfg.drive_drift_threshold:.2f}, {sigma:.1f}σ)"
                        ),
                        observed_value=value,
                        baseline_value=baseline.mean if baseline else 0.0,
                        deviation_sigma=sigma,
                        event_types_involved=["soma_tick"],
                        metadata={"drive_name": drive_name},
                        recommended_action=f"Investigate {drive_name} drive pressure source",
                    ))

            # Rate-of-change check (need at least 2 data points)
            if len(self._drive_history) >= 2:
                prev_time, prev_drives = self._drive_history[-2]
                dt = last_time - prev_time
                if dt > 0 and drive_name in prev_drives:
                    rate = (value - prev_drives[drive_name]) / (dt / 60.0)  # per minute
                    if abs(rate) > cfg.drive_drift_rate_threshold:
                        cooldown_key = f"drive_rate:{drive_name}"
                        if self._cooldown_ok(cooldown_key, now):
                            self._last_alert_time[cooldown_key] = now
                            anomalies.append(DetectedAnomaly(
                                anomaly_type=AnomalyType.DRIVE_DRIFT,
                                severity=AnomalySeverity.MEDIUM,
                                description=(
                                    f"Drive '{drive_name}' changing at {rate:+.3f}/min "
                                    f"(threshold: ±{cfg.drive_drift_rate_threshold:.2f}/min)"
                                ),
                                observed_value=rate,
                                baseline_value=0.0,
                                event_types_involved=["soma_tick"],
                                metadata={"drive_name": drive_name, "rate_per_min": rate},
                                recommended_action=f"Monitor {drive_name} drive trajectory",
                            ))

        return anomalies

    # ─── Rate Baseline Management ─────────────────────────────────

    def _maybe_update_rate_baseline(self, event_type: str, now: float) -> None:
        """Update rate baseline for an event type (amortised, every N seconds)."""
        last = self._last_rate_update.get(event_type, 0.0)
        if now - last < self._rate_update_interval_s:
            return
        self._last_rate_update[event_type] = now

        times = self._event_times.get(event_type)
        if not times:
            return

        rate = self._windowed_rate(times, now, self._config.baseline_window_s)
        self._rate_baselines[event_type].update(rate)

    # ─── Windowed Helpers ─────────────────────────────────────────

    @staticmethod
    def _windowed_rate(timestamps: deque[float], now: float, window_s: float) -> float:
        """Compute events-per-minute in the given time window."""
        if not timestamps:
            return 0.0
        cutoff = now - window_s
        count = sum(1 for t in timestamps if t >= cutoff)
        minutes = window_s / 60.0
        return count / minutes if minutes > 0 else 0.0

    @staticmethod
    def _count_in_window(timestamps: deque[float], now: float, window_s: float) -> int:
        """Count events in the given time window."""
        cutoff = now - window_s
        return sum(1 for t in timestamps if t >= cutoff)

    def _cooldown_ok(self, alert_key: str, now: float) -> bool:
        """Check if enough time has passed since the last alert of this type."""
        last = self._last_alert_time.get(alert_key, 0.0)
        return (now - last) >= self._config.cooldown_per_type_s

    # ─── Stats & Health ───────────────────────────────────────────

    def recent_anomalies(self, limit: int = 20) -> list[DetectedAnomaly]:
        """Return recent anomalies (most recent first)."""
        return list(self._detected)[-limit:][::-1]

    def stats(self) -> dict[str, Any]:
        """Return observable statistics."""
        by_type: dict[str, int] = {}
        for a in self._detected:
            by_type[a.anomaly_type.value] = by_type.get(a.anomaly_type.value, 0) + 1

        return {
            "total_observations": self._total_observations,
            "total_anomalies": self._total_anomalies,
            "anomalies_by_type": by_type,
            "tracked_event_types": len(self._event_times),
            "baseline_event_types": len(self._rate_baselines),
            "drive_observations": len(self._drive_history),
        }
