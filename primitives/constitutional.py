"""
EcodiaOS - Constitutional Check Primitive

The fundamental unit of ethical evaluation.
Every Intent passes through Equor and receives one of these.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import Field

from primitives.common import (
    DriveAlignmentVector,
    EOSBaseModel,
    Identified,
    Verdict,
    utc_now,
)


class InvariantResult(EOSBaseModel):
    """Result of checking a single invariant."""

    invariant_id: str
    name: str
    passed: bool = True
    severity: str = "warning"    # "info" | "warning" | "critical"
    explanation: str = ""


class ConstitutionalCheck(Identified):
    """
    The result of Equor evaluating an Intent against the constitution.
    """

    intent_id: str
    timestamp: datetime = Field(default_factory=utc_now)

    drive_alignment: DriveAlignmentVector = Field(default_factory=DriveAlignmentVector)
    invariant_results: list[InvariantResult] = Field(default_factory=list)

    verdict: Verdict = Verdict.APPROVED
    confidence: float = 0.8
    reasoning: str = ""
    modifications: list[str] = Field(default_factory=list)

    review_time_ms: int = 0

    # Metabolic+somatic context injected by compute_verdict_with_metabolic_state()
    # Keys:
    #   starvation_level (str)       - Oikos metabolic tier
    #   efficiency_ratio (float)     - revenue/burn_rate from Oikos
    #   floor_tightness (float)      - net floor multiplier (metabolic × somatic)
    #   somatic_urgency (float)      - urgency from Soma SOMA_TICK / SOMATIC_MODULATION
    #   somatic_stress_context (bool) - True when urgency >= 0.9
    metabolic_context: dict[str, Any] | None = None

    @property
    def has_violations(self) -> bool:
        return any(not r.passed and r.severity == "critical" for r in self.invariant_results)

    @property
    def has_warnings(self) -> bool:
        return any(not r.passed and r.severity == "warning" for r in self.invariant_results)
