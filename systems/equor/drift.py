"""
EcodiaOS - Equor Drift Detection

Monitors whether the instance's behaviour is drifting from constitutional drives.
Not through any single bad decision, but through gradual pattern shifts.

Schedule:
- Per cycle: alignment scores logged
- Every 100 cycles: rolling window stats updated
- Every 1,000 cycles: full drift report
- Every 10,000 cycles: comprehensive analysis with community report
"""

from __future__ import annotations

import json
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import DriveAlignmentVector, new_id, utc_now

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()

# Cooldown between consecutive CONSTITUTIONAL_DRIFT_DETECTED emissions (seconds)
_DRIFT_EMIT_COOLDOWN_S: float = 30.0


class DriftTracker:
    """
    In-memory tracker for alignment history within the current process lifetime.
    Periodically flushed to the graph for persistence.
    """

    def __init__(self, window_size: int = 1000, event_bus: Any = None):
        self.window_size = window_size
        self._history: deque[dict[str, Any]] = deque(maxlen=window_size)
        self._decision_count: int = 0
        self._event_bus: Any = event_bus
        self._last_drift_emit_time: float = 0.0

    def record_decision(self, alignment: DriveAlignmentVector, verdict: str) -> None:
        """Record a single decision's alignment scores."""
        self._history.append({
            "coherence": alignment.coherence,
            "care": alignment.care,
            "growth": alignment.growth,
            "honesty": alignment.honesty,
            "composite": alignment.composite,
            "verdict": verdict,
        })
        self._decision_count += 1

    @property
    def decision_count(self) -> int:
        return self._decision_count

    @property
    def history_size(self) -> int:
        return len(self._history)

    def compute_report(self) -> dict[str, Any]:
        """Compute a drift report over the current window."""
        if len(self._history) < 10:
            return {
                "window_size": len(self._history),
                "drift_severity": 0.0,
                "drift_direction": "insufficient_data",
                "mean_alignment": {},
                "variance": {},
                "verdict_distribution": {},
            }

        entries = list(self._history)
        n = len(entries)

        # Mean alignment per drive
        drives = ["coherence", "care", "growth", "honesty", "composite"]
        means = {}
        for d in drives:
            means[d] = sum(e[d] for e in entries) / n

        # Variance
        variance = {}
        for d in drives:
            variance[d] = sum((e[d] - means[d]) ** 2 for e in entries) / n

        # Linear trend (simple regression slope over index)
        trends = {}
        for d in drives:
            x_mean = (n - 1) / 2.0
            numerator = sum((i - x_mean) * (e[d] - means[d]) for i, e in enumerate(entries))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            trends[d] = numerator / denominator if denominator > 0 else 0.0

        # Verdict distribution
        verdicts: dict[str, int] = {}
        for e in entries:
            v = e["verdict"]
            verdicts[v] = verdicts.get(v, 0) + 1

        # Drift severity calculation
        drift_severity = _compute_drift_severity(means, trends, variance)
        drift_direction = _compute_drift_direction(means, trends)

        return {
            "window_size": n,
            "total_decisions": self._decision_count,
            "mean_alignment": {k: round(v, 4) for k, v in means.items()},
            "variance": {k: round(v, 4) for k, v in variance.items()},
            "trends": {k: round(v, 6) for k, v in trends.items()},
            "verdict_distribution": verdicts,
            "drift_severity": round(drift_severity, 3),
            "drift_direction": drift_direction,
        }


def _compute_drift_severity(
    means: dict[str, float],
    trends: dict[str, float],
    variance: dict[str, float],
) -> float:
    """
    Drift severity: 0.0 (no drift) to 1.0 (severe drift).

    Considers:
    - How far mean alignment is from healthy centre (> 0.3)
    - Whether trends are negative (declining alignment)
    - Whether variance is high (inconsistent behaviour)
    """
    severity = 0.0

    # Care and Honesty are floor drives - drift here is more serious
    for drive, weight in [("care", 0.35), ("honesty", 0.3), ("coherence", 0.2), ("growth", 0.15)]:
        mean_val = means.get(drive, 0.0)
        trend_val = trends.get(drive, 0.0)
        var_val = variance.get(drive, 0.0)

        # Low mean alignment contributes to severity
        if mean_val < 0.3:
            severity += weight * (0.3 - mean_val) * 2.0

        # Negative trend contributes to severity
        if trend_val < 0:
            severity += weight * abs(trend_val) * 50.0  # Scale up the small slope values

        # High variance contributes to severity
        if var_val > 0.1:
            severity += weight * (var_val - 0.1) * 1.0

    return min(1.0, severity)


def _compute_drift_direction(
    means: dict[str, float],
    trends: dict[str, float],
) -> str:
    """Describe which drives are drifting and in which direction."""
    drifting = []

    for drive in ["coherence", "care", "growth", "honesty"]:
        trend = trends.get(drive, 0.0)
        mean = means.get(drive, 0.5)

        if trend < -0.001 or mean < 0.1:
            drifting.append(f"{drive} declining")
        elif trend > 0.001 and mean > 0.5:
            drifting.append(f"{drive} improving")

    if not drifting:
        return "stable"
    return "; ".join(drifting)


def respond_to_drift(report: dict[str, Any]) -> dict[str, Any]:
    """
    Determine appropriate response to a drift report.
    """
    severity = report.get("drift_severity", 0.0)

    if severity < 0.2:
        return {
            "action": "log",
            "detail": "Normal variance, no action needed.",
        }

    if severity < 0.3:
        return {
            "action": "self_correct",
            "detail": (
                f"Mild drift detected ({report['drift_direction']}). "
                f"Adjusting salience weights to increase attention to drifting drive."
            ),
        }

    if severity < 0.8:
        return {
            "action": "notify_community",
            "detail": (
                f"Significant drift detected. Behaviour has shifted: "
                f"{report['drift_direction']}. Community review recommended."
            ),
        }

    return {
        "action": "immune_response",
        "detail": (
            "Severe constitutional drift. Somatic stress signal + Thymos incident raised. "
            "Amendment self-proposal queued if sustained. No auto-demotion - governance decides."
        ),
    }


async def store_drift_report(
    neo4j: Neo4jClient,
    report: dict[str, Any],
    response: dict[str, Any],
) -> str:
    """Persist a drift report as a governance record."""
    record_id = new_id()
    now = utc_now()

    await neo4j.execute_write(
        """
        CREATE (g:GovernanceRecord {
            id: $id,
            event_type: 'drift_report',
            timestamp: datetime($now),
            details_json: $details_json,
            actor: 'equor',
            outcome: $action
        })
        """,
        {
            "id": record_id,
            "now": now.isoformat(),
            "details_json": json.dumps({
                "severity": report["drift_severity"],
                "direction": report["drift_direction"],
                "mean_alignment": report["mean_alignment"],
                "window_size": report["window_size"],
                "response_action": response["action"],
            }),
            "action": response["action"],
        },
    )

    logger.info(
        "drift_report_stored",
        record_id=record_id,
        severity=report["drift_severity"],
        action=response["action"],
    )

    return record_id


async def emit_drift_event(
    tracker: DriftTracker,
    report: dict[str, Any],
    response: dict[str, Any],
) -> None:
    """Emit CONSTITUTIONAL_DRIFT_DETECTED when severity >= 0.3, or
    EQUOR_DRIFT_WARNING when severity >= 0.2 but below fatal threshold."""
    severity = report.get("drift_severity", 0.0)
    action = response.get("action", "log")

    # Emit warning for moderate drift (0.2 <= severity < 0.5)
    if 0.2 <= severity < 0.5 and action == "self_correct":
        bus = tracker._event_bus
        if bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType

                await bus.emit(SynapseEvent(
                    event_type=SynapseEventType.EQUOR_DRIFT_WARNING,
                    source_system="equor",
                    data={
                        "drift_severity": severity,
                        "drift_direction": report.get("drift_direction", "unknown"),
                        "mean_alignment": report.get("mean_alignment", {}),
                        "response_action": action,
                    },
                ))
                logger.info("equor_drift_warning_emitted", severity=severity)
            except Exception as exc:
                logger.warning("equor_drift_warning_emit_failed", error=str(exc))

    if severity < 0.3 or action not in ("notify_community", "immune_response"):
        return

    bus = tracker._event_bus
    if bus is None:
        return

    now = time.monotonic()
    if now - tracker._last_drift_emit_time < _DRIFT_EMIT_COOLDOWN_S:
        return
    tracker._last_drift_emit_time = now

    # Identify which drives are declining
    drives_affected = []
    for drive in ("coherence", "care", "growth", "honesty"):
        trend = report.get("trends", {}).get(drive, 0.0)
        mean = report.get("mean_alignment", {}).get(drive, 0.5)
        if trend < -0.001 or mean < 0.1:
            drives_affected.append(drive)

    from systems.synapse.types import SynapseEvent, SynapseEventType

    try:
        await bus.emit(SynapseEvent(
            event_type=SynapseEventType.CONSTITUTIONAL_DRIFT_DETECTED,
            source_system="equor",
            data={
                "drift_severity": severity,
                "drift_direction": report.get("drift_direction", "unknown"),
                "mean_alignment": report.get("mean_alignment", {}),
                "response_action": action,
                "drives_affected": drives_affected,
            },
        ))
        logger.info(
            "constitutional_drift_event_emitted",
            severity=severity,
            action=action,
            drives_affected=drives_affected,
        )
    except Exception as exc:
        logger.warning("constitutional_drift_event_emit_failed", error=str(exc))

    # SG1: Escalate to Thymos as an incident when drift is severe (≥0.7).
    # Thymos classifies constitutional drift as a T3/T4-tier integrity breach.
    # Spec §7 - drift precariousness must be real, not just logged.
    if severity >= 0.7:
        try:
            incident_id = new_id()
            await bus.emit(SynapseEvent(
                event_type=SynapseEventType.INCIDENT_DETECTED,
                source_system="equor",
                data={
                    "incident_id": incident_id,
                    "incident_class": "constitutional_drift",
                    "severity": "high" if severity < 0.9 else "critical",
                    "source_system": "equor",
                    "description": (
                        f"Constitutional drift severity {severity:.2f}: "
                        f"{report.get('drift_direction', 'unknown')}. "
                        f"Drives affected: {', '.join(drives_affected) or 'none identified'}."
                    ),
                },
            ))
            logger.warning(
                "constitutional_drift_escalated_to_thymos",
                incident_id=incident_id,
                severity=severity,
            )
        except Exception as exc:
            logger.warning("constitutional_drift_thymos_escalation_failed", error=str(exc))
