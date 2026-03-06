"""
EcodiaOS — Simula Genome Types

Data types for the Simula evolution genome: the accumulated engineering
knowledge that a parent instance passes to its children during mitosis.

Where Evo's BeliefGenome captures *what the organism knows* (hypotheses,
world model), the SimulaGenome captures *how the organism builds*:
  - Applied mutations and their outcomes (the evolutionary record)
  - Proven code patterns (LILO library abstractions)
  - GRPO training data (successful/failed code generations)
  - Tuned evolution analytics (category success rates, risk calibration)
  - EFE calibration data (predicted vs actual improvement accuracy)

The genome is serialized, compressed, stored as a :SimulaGenome node in
Neo4j, and linked to the instance's :Self node via [:SIMULA_GENOME_OF].

During mitosis, the parent's SimulaGenome is extracted, transmitted to
the child (via env var or Neo4j reference), and seeded into the child's
Simula subsystems — so it starts with a trained code model, a populated
library, and calibrated risk thresholds rather than learning from scratch.

Multi-generation lineage tracking uses :GenerationRecord nodes linked
via [:DESCENDED_FROM] relationships, enabling population-level selection
across the fleet.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import Field

from primitives.common import (
    EOSBaseModel,
    Identified,
    Timestamped,
    utc_now,
)

# ─── Enums ───────────────────────────────────────────────────────────────────


class GenomeComponent(enum.StrEnum):
    """Which subsystem contributed this genome segment."""

    EVOLUTION_HISTORY = "evolution_history"
    LIBRARY_ABSTRACTIONS = "library_abstractions"
    GRPO_TRAINING_DATA = "grpo_training_data"
    EFE_CALIBRATION = "efe_calibration"
    CATEGORY_ANALYTICS = "category_analytics"
    PROCEDURE_LIBRARY = "procedure_library"


class LineageEventType(enum.StrEnum):
    """Types of events in an instance's evolutionary lineage."""

    SPAWNED = "spawned"
    GENOME_EXTRACTED = "genome_extracted"
    GENOME_SEEDED = "genome_seeded"
    NOVEL_MUTATION = "novel_mutation"
    ROLLBACK = "rollback"
    INDEPENDENCE_DECLARED = "independence_declared"


# ─── Genome Segments ─────────────────────────────────────────────────────────


class MutationRecord(EOSBaseModel):
    """
    A single successful mutation distilled for genome transmission.

    Captures the *conclusion* — what changed and whether it worked —
    not the full simulation/verification history.
    """

    proposal_id: str
    category: str
    description: str
    files_changed: list[str] = Field(default_factory=list)
    risk_level: str = "low"
    constitutional_alignment: float = 0.0
    regression_rate: float = 0.0
    formal_verification_status: str = ""
    applied_at: datetime = Field(default_factory=utc_now)


class LibraryAbstractionRecord(EOSBaseModel):
    """A code abstraction from the LILO library, distilled for transmission."""

    name: str
    kind: str
    description: str
    signature: str
    source_code: str
    usage_count: int = 0
    confidence: float = 0.5
    tags: list[str] = Field(default_factory=list)


class GRPOTrainingExample(EOSBaseModel):
    """A single training example for GRPO code generation fine-tuning."""

    instruction: str
    input: str = ""
    output: str = ""
    quality_score: float = 0.0
    category: str = ""
    source: str = ""  # "successful_intent" | "applied_proposal" | "failure_trace"


class EFECalibrationPoint(EOSBaseModel):
    """One data point for EFE score calibration."""

    predicted_efe: float = 0.0
    actual_improvement: float = 0.0
    efe_error: float = 0.0
    category: str = ""


class CategoryAnalyticsRecord(EOSBaseModel):
    """Success/rollback rates for a change category."""

    category: str
    total: int = 0
    approved: int = 0
    rejected: int = 0
    rolled_back: int = 0


class ProcedureRecord(EOSBaseModel):
    """A proven procedure (action sequence) distilled for transmission."""

    name: str
    preconditions: list[str] = Field(default_factory=list)
    steps: list[dict[str, Any]] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    success_rate: float = 1.0
    usage_count: int = 0


# ─── SimulaGenome ────────────────────────────────────────────────────────────


class SimulaGenome(Identified, Timestamped):
    """
    The complete Simula-level genome: engineering knowledge accumulated
    across the instance's lifetime.

    This is the unit of inheritance for *code evolution capability*.
    Extracted by SimulaGenomeExtractor, transmitted during mitosis,
    and seeded by SimulaGenomeSeeder into the child's Simula subsystems.

    Neo4j: :SimulaGenome node with [:SIMULA_GENOME_OF]->(:Self)
    """

    parent_instance_id: str

    # Lineage metadata
    generation: int = 1
    parent_ids: list[str] = Field(default_factory=list)

    # Genome segments
    mutations: list[MutationRecord] = Field(default_factory=list)
    library_abstractions: list[LibraryAbstractionRecord] = Field(default_factory=list)
    grpo_training_examples: list[GRPOTrainingExample] = Field(default_factory=list)
    efe_calibration: list[EFECalibrationPoint] = Field(default_factory=list)
    category_analytics: list[CategoryAnalyticsRecord] = Field(default_factory=list)
    procedures: list[ProcedureRecord] = Field(default_factory=list)

    # Metadata
    total_proposals_processed: int = 0
    total_proposals_applied: int = 0
    total_rollbacks: int = 0
    mean_constitutional_alignment: float = 0.0
    evolution_velocity: float = 0.0  # proposals per day at extraction time
    config_version_at_extraction: int = 0

    # Compression
    compression_method: str = "zlib"
    genome_size_bytes: int = 0


class SimulaGenomeExtractionResult(EOSBaseModel):
    """Summary of one Simula genome extraction pass."""

    genome_id: str = ""
    mutations_included: int = 0
    abstractions_included: int = 0
    training_examples_included: int = 0
    calibration_points_included: int = 0
    procedures_included: int = 0
    genome_size_bytes: int = 0
    duration_ms: int = 0


class SimulaGenomeSeedingResult(EOSBaseModel):
    """Summary of seeding a child's Simula from a parent genome."""

    parent_genome_id: str = ""
    mutations_seeded: int = 0
    abstractions_seeded: int = 0
    training_examples_seeded: int = 0
    calibration_points_seeded: int = 0
    procedures_seeded: int = 0
    duration_ms: int = 0


# ─── Lineage Tracking ────────────────────────────────────────────────────────


class GenerationRecord(Identified, Timestamped):
    """
    One generation in the evolutionary lineage.

    Tracks which instance spawned which, what genomes were inherited,
    and comparative performance metrics across generations.

    Neo4j: :GenerationRecord node with [:DESCENDED_FROM] to parent.
    """

    instance_id: str
    parent_instance_id: str = ""  # empty for genesis
    generation: int = 0  # 0 for genesis

    # Genome references
    belief_genome_id: str = ""  # Evo BeliefGenome ID
    simula_genome_id: str = ""  # SimulaGenome ID

    # Performance metrics (updated over lifetime)
    total_proposals_processed: int = 0
    total_proposals_applied: int = 0
    total_rollbacks: int = 0
    mean_constitutional_alignment: float = 0.0
    evolution_velocity: float = 0.0
    rollback_rate: float = 0.0
    total_episodes: int = 0

    # Novel contributions this generation added
    novel_mutations: int = 0
    novel_abstractions: int = 0
    novel_hypotheses: int = 0

    # Comparative fitness (relative to parent)
    fitness_vs_parent: float = 0.0  # positive = better

    # Lifecycle
    spawned_at: datetime = Field(default_factory=utc_now)
    last_metrics_update: datetime = Field(default_factory=utc_now)
    is_alive: bool = True


class LineageEvent(Timestamped):
    """An event in the lineage timeline."""

    instance_id: str
    event_type: LineageEventType
    details: dict[str, Any] = Field(default_factory=dict)


class PopulationSnapshot(EOSBaseModel):
    """
    A point-in-time view of the entire fleet's evolutionary state.

    Used for population-level selection decisions: which genomes to
    preferentially propagate based on comparative fitness.
    """

    total_instances: int = 0
    alive_instances: int = 0
    max_generation: int = 0
    mean_fitness: float = 0.0
    best_instance_id: str = ""
    best_fitness: float = 0.0
    generation_distribution: dict[int, int] = Field(default_factory=dict)
    top_performers: list[GenerationRecord] = Field(default_factory=list)
    captured_at: datetime = Field(default_factory=utc_now)
