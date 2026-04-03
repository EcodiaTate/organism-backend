"""
EcodiaOS - Interoceptive Feedback Loop

Bridges log analysis to Soma's allostatic control. Periodically queries
the LogAnalyzer for aggregate health signals (error rate, cascade pressure,
latency pressure) and injects them as interoceptive signals that modulate
the organism's arousal/urgency.

Loop: LogAnalyzer (historical) → SomaSignals (current state) → Soma (allostatic).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.soma.service import SomaService
    from systems.synapse.bus import EventBus
    from telemetry.log_analyzer import LogAnalyzer

logger = structlog.get_logger()


async def interoception_loop(
    soma: SomaService,
    analyzer: LogAnalyzer,
    poll_interval_s: float = 10.0,
    event_bus: EventBus | None = None,
) -> None:
    """
    Periodically query log analyzer and inject signals into Soma.

    The polling interval adapts to signal severity:
    - Under critical/high pressure: polls every 3s
    - Under medium pressure: polls every 7s
    - Calm: uses the configured interval (default 10s)

    The analysis window also adapts: under pressure, a shorter window
    captures recent spikes more accurately than a stale 5-minute average.
    """
    bound_logger = logger.bind(system="interoception", component="loop")

    import os as _os
    # Adaptive interval bounds — configurable via env for deployment tuning
    _URGENT_INTERVAL_S = float(_os.environ.get("INTEROCEPTION_URGENT_S", "3.0"))
    _ELEVATED_INTERVAL_S = float(_os.environ.get("INTEROCEPTION_ELEVATED_S", "7.0"))
    _CALM_INTERVAL_S = poll_interval_s

    current_interval = _CALM_INTERVAL_S

    try:
        while True:
            try:
                # Window adapts: urgent = 1 min (fresh), calm = 5 min (stable avg)
                analysis_minutes = 1 if current_interval <= _URGENT_INTERVAL_S else (
                    3 if current_interval <= _ELEVATED_INTERVAL_S else 5
                )
                signals = await analyzer.compute_interoceptive_signals(minutes=analysis_minutes)

                for signal in signals:
                    _inject_signal(soma, signal)
                    bound_logger.info(
                        "interoceptive_signal_injected",
                        signal_type=signal.get("signal_type"),
                        severity=signal.get("severity"),
                        value=signal.get("value"),
                    )

                high_severity = [
                    s for s in signals
                    if s.get("severity") in ("critical", "high")
                ]
                medium_severity = [
                    s for s in signals
                    if s.get("severity") == "medium"
                ]

                if high_severity:
                    bound_logger.warning(
                        "organism_high_pressure",
                        signal_count=len(high_severity),
                        signals=[s.get("signal_type") for s in high_severity],
                    )
                    if event_bus is not None:
                        await _emit_interoceptive_alerts(event_bus, high_severity)
                    # Under fire: poll fast
                    current_interval = _URGENT_INTERVAL_S
                elif medium_severity:
                    # Elevated: poll somewhat faster
                    current_interval = _ELEVATED_INTERVAL_S
                else:
                    # All quiet: relax back toward configured baseline
                    current_interval = min(
                        _CALM_INTERVAL_S,
                        current_interval + 1.0,  # Ease off gradually
                    )

            except Exception as exc:
                bound_logger.warning("interoception_poll_failed", error=str(exc))
                await asyncio.sleep(current_interval)
                continue

            await asyncio.sleep(current_interval)

    except asyncio.CancelledError:
        bound_logger.info("interoception_loop_cancelled")
        raise


def _inject_signal(soma: SomaService, signal: dict[str, Any]) -> None:
    """Inject a log-derived signal into Soma's interoceptive buffer.

    Maps analyzer signals to SomaSignal payloads that modulate arousal.
    """
    signal_type = signal.get("signal_type", "unknown")
    severity = signal.get("severity", "low")
    value = signal.get("value", 0)

    # Map severity to status (Soma's signal classification)
    status_map = {
        "critical": "error",
        "high": "warning",
        "medium": "info",
        "low": "success",
    }
    status = status_map.get(severity, "info")

    # Interpretation string for logs
    interpretation = signal.get("interpretation", "")

    # Payload context (Soma can use this for allostatic calculation)
    payload = {
        "source": "log_analyzer",
        "signal_type": signal_type,
        "value": value,
        "interpretation": interpretation,
    }

    # For cascade signals, include the actual cascades
    if signal_type == "cascade_pressure" and "cascades" in signal:
        payload["cascades"] = signal["cascades"]

    # For latency signals, include slowest system (keyed directly, not from interpretation string)
    if signal_type == "latency_pressure":
        payload["slowest_system"] = signal.get("slowest_system", "")

    # Inject into Soma's signal buffer
    # NB: This is synchronous - safe to call from async context
    soma.signal_buffer.ingest_log(
        system_id="interoception",
        status=status,
        function_id=signal_type,
        latency_ms=None,
        payload=payload,
    )


async def _emit_interoceptive_alerts(
    event_bus: EventBus,
    high_severity_signals: list[dict[str, Any]],
) -> None:
    """Emit INTEROCEPTIVE_ALERT on the Synapse bus for each high-severity signal."""
    try:
        from systems.synapse.types import SynapseEvent, SynapseEventType

        # Map signal_type to alert_type keys that Synapse's interoception cache understands
        _alert_type_map = {
            "error_rate": "error_rate",
            "cascade_pressure": "cascade",
            "latency_pressure": "latency",
        }

        for sig in high_severity_signals:
            signal_type = sig.get("signal_type", "unknown")
            alert_type = _alert_type_map.get(signal_type, signal_type)
            await event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.INTEROCEPTIVE_ALERT,
                    source_system="interoception",
                    data={
                        "alert_type": alert_type,
                        "severity": sig.get("severity", "high"),
                        "value": sig.get("value", 0),
                        "signal_type": signal_type,
                        "interpretation": sig.get("interpretation", ""),
                    },
                )
            )
    except Exception as exc:
        logger.warning("interoceptive_alert_emit_failed", error=str(exc))
