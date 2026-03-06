"""
EcodiaOS — Interoceptive Feedback Loop

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
    from telemetry.log_analyzer import LogAnalyzer

logger = structlog.get_logger()


async def interoception_loop(
    soma: SomaService,
    analyzer: LogAnalyzer,
    poll_interval_s: float = 10.0,
) -> None:
    """
    Periodically query log analyzer and inject signals into Soma.

    Runs continuously in background. Every N seconds:
    1. Query analyzer for aggregate interoceptive signals
    2. Inject into Soma's signal buffer
    3. Log if signals indicate high severity

    Args:
        soma: The Soma system (interoceptive controller)
        analyzer: The LogAnalyzer instance
        poll_interval_s: How often to poll for signals (default 10s)
    """
    bound_logger = logger.bind(system="interoception", component="loop")

    try:
        while True:
            try:
                # Query analyzer for 5-minute window of log data
                signals = await analyzer.compute_interoceptive_signals(minutes=5)

                # Inject into Soma's signal buffer
                for signal in signals:
                    _inject_signal(soma, signal)
                    bound_logger.info(
                        "interoceptive_signal_injected",
                        signal_type=signal.get("signal_type"),
                        severity=signal.get("severity"),
                        value=signal.get("value"),
                    )

                # Log if organism is under high pressure
                high_severity = [
                    s for s in signals
                    if s.get("severity") in ("critical", "high")
                ]
                if high_severity:
                    bound_logger.warning(
                        "organism_high_pressure",
                        signal_count=len(high_severity),
                        signals=[s.get("signal_type") for s in high_severity],
                    )

            except Exception as exc:
                bound_logger.warning(
                    "interoception_poll_failed",
                    error=str(exc),
                )
                # Continue polling even on errors
                await asyncio.sleep(poll_interval_s)
                continue

            # Wait before next poll
            await asyncio.sleep(poll_interval_s)

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

    # For latency signals, include slowest system
    if signal_type == "latency_pressure":
        payload["slowest_system"] = signal.get("interpretation", "")

    # Inject into Soma's signal buffer
    # NB: This is synchronous — safe to call from async context
    soma.signal_buffer.ingest_log(
        system_id="interoception",
        status=status,
        function_id=signal_type,
        latency_ms=None,
        payload=payload,
    )
