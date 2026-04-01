"""
EcodiaOS - Experimental Primitives

Canonical types for Evo's experiment lifecycle. When a hypothesis
transitions to TESTING, an ExperimentDesign is created; when the test
completes, an ExperimentResult is recorded.

These are shared primitives because Benchmarks, RE training, and Simula
all consume experiment data for fitness tracking, training examples,
and sandbox validation respectively.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from pydantic import Field

from primitives.common import EOSBaseModel, Identified, new_id, utc_now


class ExperimentDesign(Identified):
    """
    Design for a hypothesis test - what will be measured, how, and
    what constitutes success.

    Created when a Hypothesis transitions from PROPOSED to TESTING.
    Pure data - no system-specific logic.
    """

    hypothesis_id: str
    experiment_type: str  # "A/B", "before_after", "controlled"
    description: str
    success_criteria: str
    duration_estimate: timedelta | None = None
    created_at: datetime = Field(default_factory=utc_now)


class ExperimentResult(EOSBaseModel):
    """
    Outcome of a completed experiment.

    Recorded when a hypothesis test concludes - confirmed, refuted,
    or inconclusive. Metrics dict is free-form to accommodate different
    experiment types.
    """

    id: str = Field(default_factory=new_id)
    experiment_id: str
    hypothesis_id: str
    outcome: Literal["confirmed", "refuted", "inconclusive"]
    metrics: dict[str, float] = Field(default_factory=dict)
    observations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    completed_at: datetime = Field(default_factory=utc_now)
