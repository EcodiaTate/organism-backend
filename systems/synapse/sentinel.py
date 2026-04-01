"""
EcodiaOS - Universal Error Sentinel

Every cognitive system MUST report operational errors to Thymos via this
sentinel. Without it, 93% of errors are invisible to the immune system.

Usage in any system:
    from systems.synapse.sentinel import ErrorSentinel

    class MyService:
        def __init__(self, synapse: SynapseService) -> None:
            self._sentinel = ErrorSentinel("my_system", synapse)

        async def do_work(self) -> None:
            try:
                await risky_operation()
            except Exception as exc:
                await self._sentinel.report(exc)
                raise  # or handle

The sentinel:
  1. Creates a structured Incident with proper classification
  2. Emits it as a SYSTEM_FAILED SynapseEvent so Thymos can triage
  3. Fingerprints for deduplication (same error = same fingerprint)
  4. Never propagates its own failures - emission is best-effort
  5. Rate-limits per fingerprint to avoid flooding (max 1/30s per unique error)
"""

from __future__ import annotations

import hashlib
import time
import traceback
from typing import TYPE_CHECKING, Any

import structlog

from primitives.incident import IncidentClass, IncidentSeverity

if TYPE_CHECKING:
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("synapse.sentinel")

# Rate limit: max 1 incident per fingerprint per this many seconds
_RATE_LIMIT_WINDOW_S: float = 30.0

# Map exception types to incident classification
_EXCEPTION_CLASS_MAP: dict[str, IncidentClass] = {
    "TimeoutError": IncidentClass.DEGRADATION,
    "asyncio.TimeoutError": IncidentClass.DEGRADATION,
    "ConnectionError": IncidentClass.CRASH,
    "ConnectionRefusedError": IncidentClass.CRASH,
    "ConnectionResetError": IncidentClass.CRASH,
    "OSError": IncidentClass.CRASH,
    "ImportError": IncidentClass.CRASH,
    "ModuleNotFoundError": IncidentClass.CRASH,
    "MemoryError": IncidentClass.RESOURCE_EXHAUSTION,
    "RuntimeError": IncidentClass.CRASH,
    "ValueError": IncidentClass.CONTRACT_VIOLATION,
    "TypeError": IncidentClass.CONTRACT_VIOLATION,
    "KeyError": IncidentClass.CONTRACT_VIOLATION,
    "AttributeError": IncidentClass.CONTRACT_VIOLATION,
    "AssertionError": IncidentClass.CONTRACT_VIOLATION,
    "PermissionError": IncidentClass.CRASH,
}

# Map exception types to severity
_EXCEPTION_SEVERITY_MAP: dict[str, IncidentSeverity] = {
    "TimeoutError": IncidentSeverity.MEDIUM,
    "asyncio.TimeoutError": IncidentSeverity.MEDIUM,
    "ConnectionError": IncidentSeverity.HIGH,
    "ConnectionRefusedError": IncidentSeverity.HIGH,
    "ConnectionResetError": IncidentSeverity.HIGH,
    "MemoryError": IncidentSeverity.CRITICAL,
    "ImportError": IncidentSeverity.HIGH,
    "ModuleNotFoundError": IncidentSeverity.HIGH,
    "RuntimeError": IncidentSeverity.HIGH,
    "ValueError": IncidentSeverity.MEDIUM,
    "TypeError": IncidentSeverity.MEDIUM,
    "KeyError": IncidentSeverity.MEDIUM,
    "AttributeError": IncidentSeverity.MEDIUM,
    "AssertionError": IncidentSeverity.MEDIUM,
    "PermissionError": IncidentSeverity.HIGH,
}

# Systems whose failure affects the whole organism
_HIGH_BLAST_RADIUS_SYSTEMS: frozenset[str] = frozenset({
    "equor", "memory", "synapse", "atune", "nova",
})


class ErrorSentinel:
    """
    Lightweight, zero-config error reporter for any cognitive system.

    Wraps the Synapse event bus emission pattern so that every system
    can report errors to Thymos with a single ``await sentinel.report(exc)``.
    """

    __slots__ = (
        "_system_id",
        "_event_bus",
        "_log",
        "_last_emit_times",
        "_total_reported",
        "_total_suppressed",
    )

    def __init__(
        self,
        system_id: str,
        event_bus: EventBus | None = None,
    ) -> None:
        self._system_id = system_id
        self._event_bus = event_bus
        self._log = logger.bind(system=system_id, component="sentinel")
        # Fingerprint → last emission timestamp (for rate limiting)
        self._last_emit_times: dict[str, float] = {}
        self._total_reported: int = 0
        self._total_suppressed: int = 0

    def set_event_bus(self, event_bus: EventBus) -> None:
        """Wire the event bus after construction (for late-binding setups)."""
        self._event_bus = event_bus

    async def report(
        self,
        error: BaseException,
        *,
        incident_class: IncidentClass | None = None,
        severity: IncidentSeverity | None = None,
        context: dict[str, Any] | None = None,
        affected_systems: list[str] | None = None,
        user_visible: bool = False,
    ) -> None:
        """
        Report an error to Thymos via the Synapse event bus.

        Classifies the error automatically if incident_class/severity not given.
        Rate-limits duplicate fingerprints to avoid flooding.
        Never raises - emission failures are logged and swallowed.
        """
        if self._event_bus is None:
            self._log.warning(
                "sentinel_no_event_bus",
                error_type=type(error).__name__,
                error_message=str(error)[:200],
            )
            return

        error_type = type(error).__name__
        error_message = str(error)[:2000]

        # Fingerprint for deduplication + rate limiting
        fingerprint = hashlib.md5(
            f"{self._system_id}_{error_type}_{error_message[:100]}".encode()
        ).hexdigest()

        # Rate limit check
        now = time.monotonic()
        last_emit = self._last_emit_times.get(fingerprint, 0.0)
        if now - last_emit < _RATE_LIMIT_WINDOW_S:
            self._total_suppressed += 1
            return

        # Classify
        resolved_class = incident_class or _EXCEPTION_CLASS_MAP.get(
            error_type, IncidentClass.CRASH
        )
        resolved_severity = severity or _EXCEPTION_SEVERITY_MAP.get(
            error_type, IncidentSeverity.HIGH
        )

        # Build context with stack trace
        full_context: dict[str, Any] = context.copy() if context else {}
        full_context["stack_trace_summary"] = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )[-3000:]

        # Blast radius heuristic
        blast_radius = 0.3
        if self._system_id in _HIGH_BLAST_RADIUS_SYSTEMS:
            blast_radius = 0.7
        if resolved_severity == IncidentSeverity.CRITICAL:
            blast_radius = min(1.0, blast_radius + 0.3)

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            # Build a structured incident dict - Incident is a thymos-internal
            # model with repair-lifecycle fields; sentinel emits the primitives
            # only (Spec 09 §AV9: no cross-system type import from Thymos).
            incident_data: dict[str, Any] = {
                "incident_class": resolved_class.value,
                "severity": resolved_severity.value,
                "fingerprint": fingerprint,
                "source_system": self._system_id,
                "error_type": error_type,
                "error_message": error_message,
                "stack_trace": "".join(
                    traceback.format_exception(
                        type(error), error, error.__traceback__
                    )
                )[-5000:],
                "context": full_context,
                "affected_systems": affected_systems or [self._system_id],
                "blast_radius": blast_radius,
                "user_visible": user_visible,
            }

            await self._event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.SYSTEM_FAILED,
                    source_system=self._system_id,
                    data={"system_id": self._system_id, "incident": incident_data},
                )
            )

            self._last_emit_times[fingerprint] = now
            self._total_reported += 1

            self._log.info(
                "sentinel_incident_emitted",
                error_type=error_type,
                incident_class=resolved_class.value,
                severity=resolved_severity.value,
                fingerprint=fingerprint[:8],
            )
        except Exception as emit_exc:
            self._log.warning(
                "sentinel_emission_failed",
                error_type=error_type,
                emit_error=str(emit_exc),
            )

    async def report_degradation(
        self,
        description: str,
        *,
        metric_name: str = "",
        metric_value: float = 0.0,
        threshold: float = 0.0,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Report a performance degradation (not a crash, but system is slow/wrong).
        """
        degradation_error = RuntimeError(description)
        ctx = context.copy() if context else {}
        ctx["metric_name"] = metric_name
        ctx["metric_value"] = metric_value
        ctx["threshold"] = threshold

        await self.report(
            degradation_error,
            incident_class=IncidentClass.DEGRADATION,
            severity=IncidentSeverity.MEDIUM,
            context=ctx,
        )

    async def report_contract_violation(
        self,
        description: str,
        *,
        expected: str = "",
        actual: str = "",
        context: dict[str, Any] | None = None,
    ) -> None:
        """Report a contract/SLA violation between systems."""
        violation_error = AssertionError(description)
        ctx = context.copy() if context else {}
        ctx["expected"] = expected
        ctx["actual"] = actual

        await self.report(
            violation_error,
            incident_class=IncidentClass.CONTRACT_VIOLATION,
            severity=IncidentSeverity.HIGH,
            context=ctx,
        )

    async def report_stall(
        self,
        description: str,
        *,
        stalled_for_s: float = 0.0,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Report a cognitive stall (system alive but not making progress)."""
        stall_error = TimeoutError(description)
        ctx = context.copy() if context else {}
        ctx["stalled_for_s"] = stalled_for_s

        await self.report(
            stall_error,
            incident_class=IncidentClass.COGNITIVE_STALL,
            severity=IncidentSeverity.HIGH,
            context=ctx,
        )

    @property
    def stats(self) -> dict[str, Any]:
        """Return sentinel statistics for health/debug endpoints."""
        return {
            "system_id": self._system_id,
            "total_reported": self._total_reported,
            "total_suppressed": self._total_suppressed,
            "active_fingerprints": len(self._last_emit_times),
            "has_event_bus": self._event_bus is not None,
        }
