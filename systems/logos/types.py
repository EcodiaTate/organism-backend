"""
EcodiaOS — Logos Type Definitions

All data types for the Universal Compression Engine: cognitive budget,
MDL scoring, compression cascade stages, world model structures,
intelligence metrics, entropic decay, and Schwarzschild threshold detection.
"""

from __future__ import annotations

import enum
import math
from datetime import UTC, datetime
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, Identified, utc_now

# ─── Enums ────────────────────────────────────────────────────────


class MemoryTier(enum.StrEnum):
    """Knowledge tiers tracked by the cognitive budget."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    HYPOTHESIS = "hypothesis"
    WORLD_MODEL = "world_model"


class KnowledgeItemType(enum.StrEnum):
    """Types of knowledge items scored by the MDL estimator."""

    EPISODE = "episode"
    SEMANTIC_NODE = "semantic_node"
    HYPOTHESIS = "hypothesis"
    PROCEDURE = "procedure"
    SCHEMA = "schema"


class CompressionStage(enum.StrEnum):
    """Four stages of the compression cascade."""

    HOLOGRAPHIC_ENCODING = "holographic_encoding"
    EPISODIC_COMPRESSION = "episodic_compression"
    SEMANTIC_DISTILLATION = "semantic_distillation"
    WORLD_MODEL_INTEGRATION = "world_model_integration"


class DecayType(enum.StrEnum):
    """Three types of entropic decay."""

    ACCESS = "access"
    COMPRESSION = "compression"
    CONTRADICTION = "contradiction"


class WorldModelUpdateType(enum.StrEnum):
    """Types of world model integration outcomes."""

    PRIOR_UPDATED = "prior_updated"
    SCHEMA_EXTENDED = "schema_extended"
    SCHEMA_CREATED = "schema_created"
    CAUSAL_REVISED = "causal_revised"
    INVARIANT_VIOLATED = "invariant_violated"


# ─── Cognitive Budget ────────────────────────────────────────────


class CognitiveBudgetState(EOSBaseModel):
    """
    The hard capacity limit of the EOS knowledge system.
    Exceeding it is not permitted. The pressure of this limit
    is the primary driver of abstraction.
    """

    total_budget: int = 1_000_000  # Hard ceiling in knowledge units (KU)

    # Allocation by memory tier (fractions of total_budget)
    episodic_allocation: float = 0.20
    semantic_allocation: float = 0.35
    procedural_allocation: float = 0.20
    hypothesis_allocation: float = 0.15
    world_model_allocation: float = 0.10

    # Pressure thresholds
    compression_pressure_start: float = 0.75
    emergency_compression: float = 0.90
    critical_eviction: float = 0.95

    # Current utilization per tier (absolute KU counts)
    current_utilization: dict[str, float] = Field(default_factory=dict)

    @property
    def total_used(self) -> float:
        return sum(self.current_utilization.values())

    @property
    def total_pressure(self) -> float:
        """0.0 = empty, 1.0 = full."""
        return self.total_used / self.total_budget

    @property
    def compression_urgency(self) -> float:
        """
        Non-linear pressure curve. Quadratic: mild early, severe near capacity.
        Models thermodynamic pressure approaching entropy limit.
        """
        p = self.total_pressure
        if p < self.compression_pressure_start:
            return 0.0
        normalized = (p - self.compression_pressure_start) / (
            1.0 - self.compression_pressure_start
        )
        return min(normalized**2, 1.0)

    def tier_budget(self, tier: MemoryTier) -> int:
        """Absolute KU budget for a specific tier."""
        alloc_map = {
            MemoryTier.EPISODIC: self.episodic_allocation,
            MemoryTier.SEMANTIC: self.semantic_allocation,
            MemoryTier.PROCEDURAL: self.procedural_allocation,
            MemoryTier.HYPOTHESIS: self.hypothesis_allocation,
            MemoryTier.WORLD_MODEL: self.world_model_allocation,
        }
        return int(self.total_budget * alloc_map.get(tier, 0.0))

    def tier_pressure(self, tier: MemoryTier) -> float:
        """Pressure for a single tier."""
        budget = self.tier_budget(tier)
        if budget <= 0:
            return 1.0
        used = self.current_utilization.get(tier.value, 0.0)
        return min(used / budget, 1.0)


# ─── MDL Score ───────────────────────────────────────────────────


class MDLScore(EOSBaseModel):
    """
    Minimum Description Length score for a knowledge item.

    Core formula: MDL = -log P(data | model) + |model|

    A good score = explains a lot using few bits.
    A bad score = explains little, or requires many bits to express.
    """

    item_id: str
    item_type: KnowledgeItemType

    # What this item explains
    observations_covered: int = 0
    observation_complexity: float = 0.0  # Total bits of covered observations

    # What this item costs
    description_length: float = 0.0  # Bits required to describe this item

    # Derived scores
    compression_ratio: float = 0.0  # observation_complexity / description_length
    marginal_value: float = 0.0  # What disappears if this is removed

    # Temporal dynamics
    last_accessed: datetime = Field(default_factory=utc_now)
    access_frequency: float = 0.0  # Hz
    decay_rate: float = 0.1  # Entropy-driven forgetting rate

    @property
    def survival_score(self) -> float:
        """
        Probability this item survives the next compression cycle.
        High compression ratio + high access = survives.
        Low compression ratio + low access = evicted/compressed.
        """
        compression_value = min(self.compression_ratio / 10.0, 1.0)
        recency_value = self._recency_decay()
        access_value = min(self.access_frequency * 10, 1.0)

        return (compression_value * 0.5) + (recency_value * 0.2) + (access_value * 0.3)

    def _recency_decay(self) -> float:
        age_hours = (datetime.now(UTC) - self.last_accessed).total_seconds() / 3600
        return math.exp(-self.decay_rate * max(age_hours, 0.0))


# ─── Experience & Delta ──────────────────────────────────────────


class RawExperience(Identified):
    """Raw uncompressed experience entering the compression cascade."""

    context: dict[str, Any] = Field(default_factory=dict)
    content: dict[str, Any] = Field(default_factory=dict)
    raw_complexity: float = 0.0  # Estimated bits
    source_system: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


class SemanticDelta(EOSBaseModel):
    """The semantic difference between prediction and reality."""

    information_content: float = 0.0  # 0.0 = perfectly predicted, 1.0 = fully novel
    novel_entities: list[str] = Field(default_factory=list)
    violated_priors: list[str] = Field(default_factory=list)
    novel_relations: list[str] = Field(default_factory=list)
    content: dict[str, Any] = Field(default_factory=dict)


class ExperienceDelta(EOSBaseModel):
    """
    The holographically encoded delta between reality and the world model's prediction.
    Only the delta needs storage — the prediction is free (generated by the model).
    """

    experience_id: str
    delta_content: SemanticDelta | None = None
    information_content: float = 0.0
    world_model_update_required: bool = False
    discard_after_encoding: bool = False


# ─── Compression Cascade ─────────────────────────────────────────


class StageMetrics(EOSBaseModel):
    """Compression metrics for a single cascade stage."""

    stage: CompressionStage
    bits_in: float = 0.0
    bits_out: float = 0.0
    items_in: int = 0
    items_out: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.bits_out <= 0:
            return 0.0
        return self.bits_in / self.bits_out


class SalientEpisode(Identified):
    """
    Stage 2 output: an experience delta that survived episodic compression.
    Only prediction errors with significant information content survive.
    """

    experience_id: str
    delta: ExperienceDelta
    prediction_error: float = 0.0  # How wrong the world model was
    salience: float = 0.0  # Combined novelty + violation weight
    context: dict[str, Any] = Field(default_factory=dict)
    content: dict[str, Any] = Field(default_factory=dict)
    raw_bits: float = 0.0
    compressed_bits: float = 0.0


class SemanticExtraction(EOSBaseModel):
    """
    Stage 3 output: structured knowledge extracted from episodes.
    Multiple episodes sharing a pattern get replaced by 1 semantic node
    plus N lightweight references.
    """

    entities: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)
    schemas: list[str] = Field(default_factory=list)  # Schema IDs created/extended
    hypotheses: list[str] = Field(default_factory=list)
    episode_refs: list[str] = Field(default_factory=list)  # Lightweight back-refs
    raw_bits: float = 0.0  # Total bits of source episodes
    distilled_bits: float = 0.0  # Bits after distillation


class CascadeResult(EOSBaseModel):
    """Result of one item passing through the compression cascade."""

    experience_id: str
    stage_reached: CompressionStage
    delta: ExperienceDelta | None = None
    salient_episode: SalientEpisode | None = None
    semantic_extraction: SemanticExtraction | None = None
    world_model_update: WorldModelUpdate | None = None
    compressed_item_id: str | None = None
    is_irreducible: bool = False
    anchor_memory: bool = False
    compression_ratio: float = 0.0
    bits_saved: float = 0.0
    stage_metrics: list[StageMetrics] = Field(default_factory=list)


class CompressionCycleReport(EOSBaseModel):
    """Summary of a full compression cycle."""

    timestamp: datetime = Field(default_factory=utc_now)
    items_processed: int = 0
    items_evicted: int = 0
    items_distilled: int = 0
    items_reinforced: int = 0
    anchors_created: int = 0
    bits_saved: float = 0.0
    mdl_improvement: float = 0.0
    cycle_duration_ms: float = 0.0
    stage_metrics: list[StageMetrics] = Field(default_factory=list)


# ─── World Model ─────────────────────────────────────────────────


class GenerativeSchema(Identified):
    """A compressed rule for generating entity types, relations, or causal chains."""

    name: str = ""
    domain: str = ""
    description: str = ""
    pattern: dict[str, Any] = Field(default_factory=dict)
    instance_count: int = 0
    compression_ratio: float = 0.0
    created_at: datetime = Field(default_factory=utc_now)
    last_instantiated: datetime = Field(default_factory=utc_now)
    instantiation_frequency: float = 0.0


class CausalLink(EOSBaseModel):
    """A directed causal connection in the causal graph."""

    cause_id: str
    effect_id: str
    strength: float = 0.5  # 0-1 confidence
    domain: str = ""
    observations: int = 0
    last_observed: datetime = Field(default_factory=utc_now)


class PriorDistribution(EOSBaseModel):
    """Predictive prior: expected observation distribution by context."""

    context_key: str
    mean_embedding: list[float] = Field(default_factory=list)
    variance: float = 1.0
    sample_count: int = 0
    last_updated: datetime = Field(default_factory=utc_now)


class EmpiricalInvariant(Identified):
    """
    A rule never violated in EOS's experience.
    The most compressed knowledge: a single rule covers infinite instances.
    """

    statement: str = ""
    domain: str = ""
    observation_count: int = 0  # How many observations this invariant covers
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=utc_now)
    last_tested: datetime = Field(default_factory=utc_now)
    source: str = ""  # "empirical", "causal_ingestion", "cross_domain"


class Prediction(EOSBaseModel):
    """A world model prediction for a given context."""

    expected_content: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    generating_schemas: list[str] = Field(default_factory=list)  # Schema IDs


class WorldModelUpdate(EOSBaseModel):
    """Result of integrating an experience delta into the world model."""

    update_type: WorldModelUpdateType
    schemas_added: int = 0
    schemas_extended: int = 0
    priors_updated: int = 0
    causal_links_added: int = 0
    causal_links_revised: int = 0
    invariants_tested: int = 0
    invariants_violated: int = 0
    complexity_delta: float = 0.0  # Change in model complexity (bits)
    coverage_delta: float = 0.0  # Change in explanatory coverage


# ─── Entropic Decay ─────────────────────────────────────────────


class DecayReport(EOSBaseModel):
    """Result of a single entropic decay cycle."""

    timestamp: datetime = Field(default_factory=utc_now)
    evicted: list[str] = Field(default_factory=list)
    distilled: list[str] = Field(default_factory=list)
    reinforced: list[str] = Field(default_factory=list)
    total_items_scanned: int = 0
    total_bits_freed: float = 0.0


# ─── Schwarzschild Threshold ────────────────────────────────────


class SelfPredictionRecord(EOSBaseModel):
    """Record of a self-prediction attempt and its outcome."""

    predicted_state: dict[str, Any] = Field(default_factory=dict)
    actual_state: dict[str, Any] = Field(default_factory=dict)
    accuracy: float = 0.0  # 0-1 how close the prediction was
    timestamp: datetime = Field(default_factory=utc_now)


class CrossDomainTransfer(EOSBaseModel):
    """Record of a schema transferring predictive power across domains."""

    schema_id: str
    source_domain: str
    target_domain: str
    prediction_accuracy: float = 0.0
    discovered_at: datetime = Field(default_factory=utc_now)


class SchwarzchildIndicators(EOSBaseModel):
    """
    The five measured indicators for Schwarzschild threshold detection.
    Each is independently tracked; the threshold requires conjunction.
    """

    # 1. Self-prediction: can the model predict its own next states?
    self_prediction_accuracy: float = 0.0
    self_prediction_trend: float = 0.0  # Derivative: improving or degrading

    # 2. Cross-domain transfer: predictions generalizing to unseen domains
    cross_domain_transfer_count: int = 0
    cross_domain_accuracy: float = 0.0

    # 3. Generative surplus: model generating more causal candidates than observed
    hypotheses_generated: int = 0
    hypotheses_received: int = 0
    generative_surplus_ratio: float = 0.0

    # 4. Compression acceleration: compression ratio improving faster than data arrives
    compression_ratio_velocity: float = 0.0  # d(ratio)/d(data)
    data_arrival_rate: float = 0.0

    # 5. Novel structure emergence: schemas with no direct observational basis
    novel_schemas_count: int = 0
    novel_schema_ids: list[str] = Field(default_factory=list)


class SchwarzchildStatus(EOSBaseModel):
    """
    Measurement of proximity to the Schwarzschild Cognition Threshold.

    When threshold_met becomes True, EOS has crossed the event horizon:
    the world model generates predictions about its own future states,
    produces more knowledge than it consumes, and transfers schemas
    across domains.
    """

    self_prediction_accuracy: float = 0.0
    intelligence_ratio: float = 0.0
    hypothesis_ratio: float = 0.0  # generated vs received
    novel_concept_rate: float = 0.0
    cross_domain_transfers: int = 0
    compression_acceleration: float = 0.0
    novel_structures: int = 0
    indicators: SchwarzchildIndicators | None = None
    threshold_met: bool = False
    measured_at: datetime = Field(default_factory=utc_now)


# ─── Intelligence Metrics ───────────────────────────────────────


class IntelligenceMetrics(EOSBaseModel):
    """
    Broadcast on Synapse every 60 seconds as INTELLIGENCE_METRICS.
    EOS's vital signs for AGI progress.
    """

    timestamp: datetime = Field(default_factory=utc_now)

    # Primary metrics
    intelligence_ratio: float = 0.0
    cognitive_pressure: float = 0.0
    compression_efficiency: float = 0.0  # Fraction with MDL score > 1.0

    # World model quality
    world_model_coverage: float = 0.0
    world_model_complexity: float = 0.0  # Bits
    prediction_accuracy: float = 0.0

    # Learning velocity
    schema_growth_rate: float = 0.0  # New schemas per hour
    hypothesis_confirmation_rate: float = 0.0
    cross_domain_transfers_today: int = 0

    # Schwarzschild proximity
    self_prediction_accuracy: float = 0.0
    hypothesis_generation_ratio: float = 0.0
    schwarzschild_threshold_met: bool = False

    # Compression cascade throughput (today)
    experiences_holographically_encoded: int = 0
    experiences_discarded_as_redundant: int = 0
    anchor_memories_created: int = 0

    # Delta from last 24h
    intelligence_ratio_delta: float = 0.0
    coverage_delta: float = 0.0
    compression_efficiency_delta: float = 0.0


# ─── Logos Configuration ─────────────────────────────────────────


class LogosConfig(EOSBaseModel):
    """Configuration for the Logos compression engine."""

    # Budget
    total_budget_ku: int = 1_000_000

    # Pressure broadcasting interval (seconds)
    pressure_broadcast_interval_s: float = 30.0
    metrics_broadcast_interval_s: float = 60.0

    # Decay cycle interval (seconds)
    decay_cycle_interval_s: float = 300.0

    # Compression cascade
    holographic_discard_threshold: float = 0.01
    world_model_update_threshold: float = 0.5

    # Episodic compression (Stage 2)
    episodic_salience_threshold: float = 0.05  # Min info content to survive

    # Semantic distillation (Stage 3)
    pattern_merge_threshold: int = 3  # Min episodes sharing pattern to merge

    # Entropic decay
    eviction_survival_threshold: float = 0.1
    reinforcement_survival_threshold: float = 0.8
    reinforcement_factor: float = 1.2
    anchor_compression_ratio_threshold: float = 1.0
    access_decay_rate: float = 0.1  # Exponential decay rate (per hour)
    contradiction_decay_multiplier: float = 2.0

    # Schwarzschild thresholds
    schwarzschild_self_prediction: float = 0.70
    schwarzschild_intelligence_ratio: float = 100.0
    schwarzschild_hypothesis_ratio: float = 1.0
    schwarzschild_measurement_interval_s: float = 120.0  # How often to measure
    self_prediction_window: int = 20  # Rolling window for accuracy
