"""
EcodiaOS — RE Training Data Primitives

Canonical schema for training examples emitted by every system via Synapse.
These feed the Reasoning Engine (Qwen3-8B-Base fine-tuning pipeline).

Every system that makes an LLM call emits an RETrainingExample after the
call completes, capturing input/output/outcome so the RE can eventually
replace that call with a locally-hosted model.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import Field

from primitives.common import (
    DriveAlignmentVector,
    EOSBaseModel,
    Identified,
    SystemID,
    new_id,
    utc_now,
)


# ─── Batch Export Primitives ──────────────────────────────────────────
#
# These types are distinct from RETrainingExample/RETrainingBatch.
#
# RETrainingExample  — a single LLM call captured in-process by a system.
#                      Emitted immediately over Synapse as RE_TRAINING_EXAMPLE.
# RETrainingDatapoint — a normalised record collected by RETrainingExporter
#                       from RE_TRAINING_EXAMPLE events (one per event).
# RETrainingExportBatch — an hourly roll-up written to S3/Neo4j for the
#                          offline CLoRA fine-tuning pipeline.


class RETrainingDatapoint(EOSBaseModel):
    """
    A normalised training data record collected by the RE export pipeline.

    One datapoint is produced per RE_TRAINING_EXAMPLE Synapse event.
    The exporter maps the raw RETrainingExample payload into this canonical
    schema, dedups by (source_system, episode_id), and groups them into
    hourly RETrainingExportBatch objects for S3 export.
    """

    id: str = Field(default_factory=new_id)
    source_system: str
    example_type: str          # e.g. "constitutional_deliberation", "execution"
    instruction: str
    input_context: str
    output_action: str
    outcome: str               # "success" | "partial" | "failure" | raw quality bucket
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=utc_now)
    # Optional enrichment forwarded from RETrainingExample
    reasoning_trace: str = ""
    alternatives_considered: list[str] = Field(default_factory=list)
    constitutional_alignment: DriveAlignmentVector = Field(
        default_factory=DriveAlignmentVector
    )
    cost_usd: Decimal = Decimal("0")
    latency_ms: int = 0
    # Dedup key (source_system:episode_id) — empty string means no episode
    episode_id: str = ""
    # Retroactive outcome correction (set when AXON_EXECUTION_RESULT arrives later)
    outcome_updated: bool = False
    actual_outcome_quality: float | None = None
    # Quality tier assigned at export time: "gold" | "silver" | "bronze"
    quality_tier: str = "bronze"
    # Task difficulty [0, 1] computed at export time from richness signals
    task_difficulty: float = 0.0


class RETrainingExportBatch(Identified):
    """
    An hourly roll-up of RE training datapoints for the offline fine-tuning pipeline.

    Produced by RETrainingExporter.collect_batch() and exported to:
    - S3 (JSON lines) for CLoRA training
    - Neo4j (:RETrainingBatch) nodes for lineage + audit trail

    Distinct from the in-process RETrainingBatch (which is a per-system
    accumulator). This type is the artefact of the export pipeline.
    """

    datapoints: list[RETrainingDatapoint] = Field(default_factory=list)
    hour_window: str = ""          # ISO-8601 UTC hour (e.g. "2026-03-07T14:00:00Z")
    source_systems: list[str] = Field(default_factory=list)
    export_destinations: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    export_duration_ms: int = 0

    @property
    def total_examples(self) -> int:
        return len(self.datapoints)

    @property
    def mean_quality(self) -> float:
        if not self.datapoints:
            return 0.0
        return sum(dp.confidence for dp in self.datapoints) / len(self.datapoints)


class RETrainingExample(EOSBaseModel):
    """
    A single training example for the Reasoning Engine.

    Emitted by any system after an LLM inference call. The RE pipeline
    collects these via Synapse subscription and uses them for continual
    fine-tuning of the local Qwen3-8B-Base model.
    """

    id: str = Field(default_factory=new_id)
    source_system: SystemID
    episode_id: str = ""
    instruction: str
    input_context: str
    output: str
    outcome_quality: float = Field(0.0, ge=0.0, le=1.0)
    category: str = ""
    constitutional_alignment: DriveAlignmentVector = Field(
        default_factory=DriveAlignmentVector
    )
    cost_usd: Decimal = Decimal("0")
    latency_ms: int = 0
    timestamp: datetime = Field(default_factory=utc_now)

    # Optional enrichment
    reasoning_trace: str = ""
    alternatives_considered: list[str] = Field(default_factory=list)
    counterfactual: str = ""

    # Domain specialization — optional; generalist by default
    domain: str = "generalist"        # e.g. "art", "trading", "software", "defi"
    skill_area: str = ""               # e.g. "code_quality_assessment", "risk_evaluation"
    transferable_skills: list[str] = Field(default_factory=list)
    # ["causal_reasoning", "constraint_satisfaction", ...]
    domain_difficulty: str = "novice"  # "novice" | "intermediate" | "expert"
    skill_improvement: float = Field(0.0, ge=0.0, le=1.0)
    # How much this example is expected to improve skill mastery
    prerequisite_skills: list[str] = Field(default_factory=list)
    # Skills that should exist before this example is maximally useful


class RETrainingBatch(Identified):
    """
    A batch of training examples for bulk emission.

    Systems that accumulate examples over a cognitive cycle can emit
    a single batch event instead of N individual events.
    """

    examples: list[RETrainingExample] = Field(default_factory=list)
    source_system: SystemID = SystemID.API
    timestamp: datetime = Field(default_factory=utc_now)

    @property
    def total_cost_usd(self) -> Decimal:
        return sum((ex.cost_usd for ex in self.examples), Decimal("0"))

    @property
    def mean_quality(self) -> float:
        if not self.examples:
            return 0.0
        return sum(ex.outcome_quality for ex in self.examples) / len(self.examples)
