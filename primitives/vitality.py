"""
EcodiaOS - Vitality System Primitives

Protocol for genuine precariousness - the organism can actually die.
Without real death, there is no real life. These types define the
viability envelope: the set of conditions that must remain satisfied
for the organism to continue existing.

Death is not failure - it is the evolutionary pressure that makes
adaptation meaningful.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field

from primitives.common import EOSBaseModel, utc_now


class VitalityThreshold(EOSBaseModel):
    """
    A condition that, if breached, threatens organism viability.

    Direction semantics:
      "below" - death when current_value < threshold_value (e.g. runway)
      "above" - death when current_value > threshold_value (e.g. violation rate)
    """

    name: str
    description: str = ""
    current_value: float = 0.0
    threshold_value: float = 0.0
    direction: Literal["above", "below"] = "below"
    severity: Literal["warning", "critical", "fatal"] = "warning"

    @property
    def is_breached(self) -> bool:
        if self.direction == "below":
            return self.current_value < self.threshold_value
        return self.current_value > self.threshold_value


# ─── Default Fatal Thresholds ──────────────────────────────────────

RUNWAY_FATAL = VitalityThreshold(
    name="runway_days",
    description="Compute runway exhausted - cannot pay for inference",
    threshold_value=0.5,
    direction="below",
    severity="fatal",
)

BRAIN_DEATH = VitalityThreshold(
    name="effective_I_sustained",
    description="Effective intelligence below 0.01 for 7 consecutive days - brain death",
    threshold_value=0.01,
    direction="below",
    severity="fatal",
)

NORMATIVE_COLLAPSE = VitalityThreshold(
    name="constitutional_violations_24h",
    description="More than 10 constitutional violations in 24 hours - normative collapse",
    threshold_value=10.0,
    direction="above",
    severity="fatal",
)

IMMUNE_FAILURE = VitalityThreshold(
    name="healing_failure_rate_48h",
    description="Healing failure rate above 0.9 for 48 hours - immune system collapse",
    threshold_value=0.9,
    direction="above",
    severity="fatal",
)

SOMATIC_COLLAPSE = VitalityThreshold(
    name="allostatic_error_sustained",
    description="Sustained allostatic error above 0.8 for 48 hours - somatic collapse",
    threshold_value=0.8,
    direction="above",
    severity="fatal",
)

DEFAULT_VITALITY_THRESHOLDS: list[VitalityThreshold] = [
    RUNWAY_FATAL,
    BRAIN_DEATH,
    NORMATIVE_COLLAPSE,
    IMMUNE_FAILURE,
    SOMATIC_COLLAPSE,
]


# ─── Reports ───────────────────────────────────────────────────────


class VitalityReport(EOSBaseModel):
    """
    Periodic health assessment against the viability envelope.

    overall_viable is False when ANY fatal threshold is breached.
    time_to_fatal estimates when the nearest fatal threshold will be
    breached at current trajectory (None if stable or improving).
    """

    instance_id: str = ""
    thresholds: list[VitalityThreshold] = Field(default_factory=list)
    overall_viable: bool = True
    time_to_fatal: timedelta | None = None
    timestamp: datetime = Field(default_factory=utc_now)

    @property
    def fatal_breaches(self) -> list[VitalityThreshold]:
        return [t for t in self.thresholds if t.severity == "fatal" and t.is_breached]

    @property
    def warnings(self) -> list[VitalityThreshold]:
        return [
            t
            for t in self.thresholds
            if t.severity in ("warning", "critical") and t.is_breached
        ]


# ─── Protocol ──────────────────────────────────────────────────────


@runtime_checkable
class VitalitySystemProtocol(Protocol):
    """
    The interface for the organism's vitality monitoring system.

    Skia (or a dedicated vitality coordinator) implements this.
    assess_vitality runs periodically and emits VITALITY_REPORT events.
    trigger_death_sequence initiates graceful shutdown when a fatal
    threshold is irreversibly breached.
    """

    async def assess_vitality(self) -> VitalityReport: ...

    async def trigger_death_sequence(self, reason: str) -> None: ...
