"""
EcodiaOS - Oneiros Type Definitions

All data types for the dream engine: sleep stages, dreams, insights,
consolidation results, sleep debt, and circadian phases.

Every dream, every insight, and every sleep cycle is a first-class
primitive - the organism's inner life made observable.
"""

from __future__ import annotations

import enum
from datetime import datetime  # noqa: TC003
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ─── Enums ────────────────────────────────────────────────────────


class SleepStage(enum.StrEnum):
    """States of consciousness in the circadian cycle."""

    WAKE = "wake"                   # Normal cognitive cycle
    HYPNAGOGIA = "hypnagogia"       # Transition in (~30s)
    NREM = "nrem"                   # Consolidation (40% of sleep)
    REM = "rem"                     # Creative dreaming (40%)
    LUCID = "lucid"                 # Self-directed dreaming (10%)
    HYPNOPOMPIA = "hypnopompia"    # Transition out (~30s)


class DreamType(enum.StrEnum):
    """What kind of dream is this?"""

    RECOMBINATION = "recombination"             # Random co-activation bridge
    THREAT_REHEARSAL = "threat_rehearsal"        # Hypothetical failure simulation
    AFFECT_PROCESSING = "affect_processing"     # Emotional charge dampening
    ETHICAL_RUMINATION = "ethical_rumination"    # Constitutional edge case
    LUCID_EXPLORATION = "lucid_exploration"      # Directed creative variation
    META_OBSERVATION = "meta_observation"        # Self-observing dream patterns
    SOMATIC = "somatic"                          # Soma sleep analysis insight


class DreamCoherence(enum.StrEnum):
    """How meaningful was the dream's creative bridge?"""

    INSIGHT = "insight"       # High coherence - genuine creative discovery
    FRAGMENT = "fragment"     # Medium - store for future recombination
    NOISE = "noise"           # Low - random noise, discard


class InsightStatus(enum.StrEnum):
    """Lifecycle of a dream insight in the waking world."""

    PENDING = "pending"           # Not yet validated in wake
    VALIDATED = "validated"       # Confirmed useful in wake state
    INVALIDATED = "invalidated"   # Turned out to be noise
    INTEGRATED = "integrated"     # Became permanent semantic knowledge


class SleepQuality(enum.StrEnum):
    """How restful was this sleep cycle?"""

    DEEP = "deep"               # Full cycle, all stages completed
    NORMAL = "normal"           # Standard quality
    FRAGMENTED = "fragmented"   # Interrupted, partial consolidation
    DEPRIVED = "deprived"       # Emergency wake, minimal benefit


# ─── Sleep Pressure ───────────────────────────────────────────────


class SleepPressure(EOSBaseModel):
    """
    Homeostatic sleep drive - rises with wake time and cognitive load.

    Like adenosine accumulation in biological brains, sleep pressure
    builds during wakefulness from four independent sources. When it
    crosses the threshold, the organism must sleep.
    """

    # Raw counters
    cycles_since_sleep: int = 0
    unprocessed_affect_residue: float = 0.0     # Sum of high-affect traces
    unconsolidated_episode_count: int = 0
    hypothesis_backlog: int = 0

    # Computed
    composite_pressure: float = 0.0             # 0.0 (rested) → 1.0+ (exhausted)

    # Thresholds
    threshold: float = 0.70                     # Triggers DROWSY signal
    critical_threshold: float = 0.95            # Forces sleep unconditionally

    # Tracking
    last_sleep_completed: datetime | None = None
    last_computation: datetime = Field(default_factory=utc_now)


class CircadianPhase(EOSBaseModel):
    """Current position in the circadian cycle."""

    wake_duration_target_s: float = 79200.0     # 22 hours default
    sleep_duration_target_s: float = 7200.0     # 2 hours default
    current_phase: SleepStage = SleepStage.WAKE
    phase_elapsed_s: float = 0.0
    total_cycles_completed: int = 0


# ─── Dreams ───────────────────────────────────────────────────────


class Dream(EOSBaseModel):
    """
    A single dream experience.

    Dreams emerge from the intersection of what happened recently
    (episodic replay), what's emotionally charged (affect residue),
    random activation (noise → creativity), and what the organism
    is uncertain about (predictive model gaps).

    Every dream is recorded. The organism can see its own dream
    patterns over time - a therapist for its own psyche.
    """

    id: str = Field(default_factory=new_id)
    dream_type: DreamType
    sleep_cycle_id: str
    timestamp: datetime = Field(default_factory=utc_now)

    # Source traces
    seed_episode_ids: list[str] = Field(default_factory=list)
    activated_episode_ids: list[str] = Field(default_factory=list)

    # Creative bridge
    bridge_narrative: str = ""                  # LLM-generated connection text
    coherence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    coherence_class: DreamCoherence = DreamCoherence.NOISE

    # Affect
    affect_valence: float = 0.0
    affect_arousal: float = 0.0

    # Semantics
    themes: list[str] = Field(default_factory=list)
    summary: str = ""

    # Context
    context: dict[str, Any] = Field(default_factory=dict)


class DreamInsight(EOSBaseModel):
    """
    A high-coherence dream discovery.

    When a dream produces a genuinely meaningful connection between
    distant memories, that connection becomes a DreamInsight. Insights
    are queued for broadcast on the first wake cycle, where they enter
    the Global Workspace like any other percept.

    Over time, validated insights become part of the organism's
    semantic memory - creative knowledge that emerges from sleep.
    """

    id: str = Field(default_factory=new_id)
    dream_id: str
    sleep_cycle_id: str

    # Content
    insight_text: str
    insight_embedding: list[float] | None = None
    coherence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    domain: str = ""                            # What area this concerns

    # Lifecycle
    status: InsightStatus = InsightStatus.PENDING
    validated_at: datetime | None = None
    validation_context: str = ""                # How it was validated
    wake_applications: int = 0                  # Times used in wake decisions

    # Source context
    seed_summary: str = ""
    activated_summary: str = ""
    bridge_narrative: str = ""

    # Timestamps
    created_at: datetime = Field(default_factory=utc_now)


# ─── Sleep Cycles ─────────────────────────────────────────────────


class SleepCycle(EOSBaseModel):
    """
    Record of a complete sleep cycle.

    Each cycle is a journey through NREM (consolidation), REM
    (creative dreaming), and optionally LUCID (self-directed
    exploration). The metrics accumulated here are the organism's
    sleep diary - observable, queryable, learnable.
    """

    id: str = Field(default_factory=new_id)
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    quality: SleepQuality = SleepQuality.NORMAL
    interrupted: bool = False
    interrupt_reason: str = ""

    # ── NREM Metrics ──
    episodes_replayed: int = 0
    semantic_nodes_created: int = 0
    traces_pruned: int = 0
    salience_reduction_mean: float = 0.0
    beliefs_compressed: int = 0
    hypotheses_pruned: int = 0
    hypotheses_promoted: int = 0

    # ── REM Metrics ──
    dreams_generated: int = 0
    insights_discovered: int = 0
    affect_traces_processed: int = 0
    affect_reduction_mean: float = 0.0
    threats_simulated: int = 0
    ethical_cases_digested: int = 0

    # ── Lucid Metrics ──
    lucid_explorations: int = 0
    meta_observations: int = 0

    # ── Lucid Dreaming (self-modification proposals) ──
    lucid_proposals_submitted: int = 0      # High-coherence insights → Simula
    lucid_proposals_accepted: int = 0       # Passed governance pipeline
    lucid_proposals_rejected: int = 0       # Rejected by Simula pipeline

    # ── Pressure ──
    pressure_before: float = 0.0
    pressure_after: float = 0.0


# ─── Consolidation Results ────────────────────────────────────────


class DreamCycleResult(EOSBaseModel):
    """Result of a single dream within REM."""

    dream: Dream
    insight: DreamInsight | None = None
    affect_delta: float = 0.0       # Change in coherence_stress
    duration_ms: int = 0


# ─── Wake Degradation ─────────────────────────────────────────────


class WakeDegradation(EOSBaseModel):
    """
    Current degradation effects from sleep deprivation.

    These are not simulated penalties - they are actual multipliers
    applied to the respective systems. The organism genuinely
    thinks worse when sleep-deprived.
    """

    salience_noise: float = 0.0             # Added noise to salience scoring (0.0-0.15)
    efe_precision_loss: float = 0.0         # Reduced Nova EFE precision (0.0-0.20)
    expression_flatness: float = 0.0        # Reduced Voxis personality (0.0-0.25)
    learning_rate_reduction: float = 0.0    # Reduced Evo learning rate (0.0-0.30)
    composite_impairment: float = 0.0       # Overall impairment (0.0-1.0)

    @classmethod
    def from_pressure(
        cls,
        pressure: float,
        threshold: float,
        critical: float,
        *,
        noise_max: float = 0.15,
        efe_max: float = 0.20,
        flatness_max: float = 0.25,
        learning_max: float = 0.30,
    ) -> WakeDegradation:
        """Compute degradation from current sleep pressure."""
        if pressure <= threshold:
            return cls()

        impairment = min(1.0, max(0.0, (pressure - threshold) / (critical - threshold)))
        return cls(
            salience_noise=impairment * noise_max,
            efe_precision_loss=impairment * efe_max,
            expression_flatness=impairment * flatness_max,
            learning_rate_reduction=impairment * learning_max,
            composite_impairment=impairment,
        )


# ─── Health Snapshot ──────────────────────────────────────────────


class OneirosHealthSnapshot(EOSBaseModel):
    """Oneiros system health and observability."""

    status: str = "healthy"
    current_stage: SleepStage = SleepStage.WAKE

    # Sleep pressure
    sleep_pressure: float = 0.0
    wake_degradation: float = 0.0
    current_sleep_debt_hours: float = 0.0

    # Lifetime metrics
    total_sleep_cycles: int = 0
    total_dreams: int = 0
    total_insights: int = 0
    insights_validated: int = 0
    insights_invalidated: int = 0
    insights_integrated: int = 0
    mean_dream_coherence: float = 0.0
    mean_sleep_quality: float = 0.0

    # Consolidation metrics
    episodes_consolidated: int = 0
    semantic_nodes_created: int = 0
    traces_pruned: int = 0
    hypotheses_pruned: int = 0
    hypotheses_promoted: int = 0

    # Affect processing
    affect_traces_processed: int = 0
    mean_affect_reduction: float = 0.0

    # Threat simulation
    threats_simulated: int = 0
    response_plans_created: int = 0

    # Lucid dreaming (self-modification proposals)
    lucid_proposals_submitted: int = 0
    lucid_proposals_accepted: int = 0
    lucid_proposals_rejected: int = 0

    # Last sleep
    last_sleep_completed: datetime | None = None
    last_sleep_quality: SleepQuality | None = None

    timestamp: datetime = Field(default_factory=utc_now)


# ═══════════════════════════════════════════════════════════════════
# v2 - Sleep as Batch Compiler (Spec 14)
# ═══════════════════════════════════════════════════════════════════


class SleepStageV2(enum.StrEnum):
    """
    The four stages of EOS sleep (v2 architecture).

    Each stage does work that cannot be done in prior stages.
    Order is non-negotiable - each stage prepares inputs for the next.
    """

    DESCENT = "descent"         # ~10% duration. Safe state capture.
    SLOW_WAVE = "slow_wave"     # ~50% duration. Deep compression. Causal reconstruction.
    REM = "rem"                 # ~30% duration. Cross-domain synthesis. Dream generation.
    EMERGENCE = "emergence"     # ~10% duration. World model integration. Wake preparation.


# Stage duration fractions (must sum to 1.0)
STAGE_DURATION_FRACTION: dict[SleepStageV2, float] = {
    SleepStageV2.DESCENT: 0.10,
    SleepStageV2.SLOW_WAVE: 0.50,
    SleepStageV2.REM: 0.30,
    SleepStageV2.EMERGENCE: 0.10,
}


class SleepTrigger(enum.StrEnum):
    """Why did the organism go to sleep?"""

    SCHEDULED = "scheduled"
    COGNITIVE_PRESSURE = "cognitive_pressure"
    COMPRESSION_BACKLOG = "compression_backlog"


class MemoryClassification(enum.StrEnum):
    """What happened to a memory on the ladder."""

    CLIMBED = "climbed"             # Promoted to higher rung
    ANCHOR = "anchor"               # Irreducibly novel - kept as-is
    DECAY_FLAGGED = "decay_flagged"  # Low MDL - candidate for forgetting


class HypothesisDisposition(enum.StrEnum):
    """What happened to a hypothesis in the graveyard."""

    CONFIRMED = "confirmed"     # Good MDL - kept
    RETIRED = "retired"         # Bad MDL, multiple cycles - buried
    DEFERRED = "deferred"       # Not enough data yet


# ─── Sleep Checkpoint ────────────────────────────────────────────


class SleepCheckpoint(EOSBaseModel):
    """
    Consistent snapshot captured at Descent. If sleep is interrupted,
    the system restores from this checkpoint.
    """

    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)

    # Intelligence state at sleep entry
    intelligence_ratio_at_sleep: float = 0.0
    active_hypothesis_count: int = 0
    unprocessed_error_count: int = 0
    world_model_complexity: float = 0.0  # bits

    # Trigger
    trigger: SleepTrigger = SleepTrigger.SCHEDULED
    cognitive_pressure_at_sleep: float = 0.0


# ─── Memory Ladder Reports ──────────────────────────────────────


class RungResult(EOSBaseModel):
    """Result of a single memory ladder rung."""

    rung: int  # 1-4
    items_in: int = 0
    items_promoted: int = 0
    items_anchored: int = 0
    items_decay_flagged: int = 0
    compression_ratio: float = 0.0


class MemoryLadderReport(EOSBaseModel):
    """
    Complete report from the four-rung memory ladder.

    Rung 1: Episodic → Semantic
    Rung 2: Semantic → Schema
    Rung 3: Schema → Procedure
    Rung 4: Procedure → World Model
    """

    memories_processed: int = 0
    semantic_nodes_created: int = 0
    schemas_created: int = 0
    procedures_extracted: int = 0
    world_model_updates: int = 0
    anchor_memories: int = 0
    compression_ratio: float = 0.0
    rung_details: list[RungResult] = Field(default_factory=list)


# ─── Hypothesis Graveyard ───────────────────────────────────────


class HypothesisGraveyardReport(EOSBaseModel):
    """Report from hypothesis graveyard processing."""

    hypotheses_evaluated: int = 0
    hypotheses_confirmed: int = 0
    hypotheses_retired: int = 0
    hypotheses_deferred: int = 0
    total_mdl_freed: float = 0.0  # bits recovered from retired hypotheses


# ─── Causal Graph Reconstruction ────────────────────────────────


class CausalReconstructionReport(EOSBaseModel):
    """Report from causal graph reconstruction."""

    nodes_in_graph: int = 0
    edges_in_graph: int = 0
    contradictions_resolved: int = 0
    invariants_discovered: int = 0
    change_magnitude: float = 0.0  # 0.0 = no change, 1.0 = complete rebuild


# ─── Stage Reports ──────────────────────────────────────────────


class WorldModelConsistencyReport(EOSBaseModel):
    """
    Results of the world model consistency audit (Spec 14 §3.3.4).

    Three classes of inconsistency are detected:
    - Orphaned schemas: GenerativeSchema nodes with no linked episodes or causal links
    - Circular causal structures: causal cycles that violate directed acyclic graph semantics
    - Deprecated hypotheses: Evo hypotheses promoted to world model beliefs but
      since invalidated or superseded
    """

    orphaned_schemas_found: int = 0
    orphaned_schemas_pruned: int = 0
    circular_structures_found: int = 0
    circular_structures_resolved: int = 0
    deprecated_hypotheses_found: int = 0
    deprecated_hypotheses_retired: int = 0
    audit_skipped: bool = False
    duration_ms: float = 0.0


class SlowWaveReport(EOSBaseModel):
    """Complete Slow Wave stage report."""

    compression: MemoryLadderReport = Field(default_factory=MemoryLadderReport)
    hypotheses: HypothesisGraveyardReport = Field(
        default_factory=HypothesisGraveyardReport
    )
    causal: CausalReconstructionReport = Field(
        default_factory=CausalReconstructionReport
    )
    consistency: WorldModelConsistencyReport = Field(
        default_factory=WorldModelConsistencyReport
    )
    duration_ms: float = 0.0


class EmergenceReport(EOSBaseModel):
    """Report from the Emergence (wake preparation) stage."""

    intelligence_ratio_before: float = 0.0
    intelligence_ratio_after: float = 0.0
    intelligence_improvement: float = 0.0
    world_model_finalized: bool = False
    input_channels_resumed: bool = False


class SleepCycleV2Report(EOSBaseModel):
    """Complete v2 sleep cycle report."""

    id: str = Field(default_factory=new_id)
    trigger: SleepTrigger = SleepTrigger.SCHEDULED
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    interrupted: bool = False
    interrupt_reason: str = ""

    checkpoint: SleepCheckpoint | None = None
    slow_wave: SlowWaveReport | None = None
    rem: REMStageReport | None = None
    lucid: LucidDreamingReport | None = None
    emergence: EmergenceReport | None = None

    # Summary
    intelligence_improvement: float = 0.0
    total_duration_ms: float = 0.0


# ─── Scheduler Config ───────────────────────────────────────────


class SleepSchedulerConfig(EOSBaseModel):
    """Configuration for the sleep scheduler."""

    # Scheduled sleep interval
    scheduled_interval_hours: float = 6.0

    # Cognitive pressure trigger
    cognitive_pressure_threshold: float = 0.85

    # Compression backlog trigger
    unprocessed_error_threshold: int = 10_000

    # Target sleep duration
    target_sleep_duration_s: float = 7200.0  # 2 hours default


# ═══════════════════════════════════════════════════════════════════
# Phase C - REM: Cross-Domain Synthesis Types
# ═══════════════════════════════════════════════════════════════════


class AbstractStructure(EOSBaseModel):
    """
    Domain-stripped relational shape of a schema.

    Takes a GenerativeSchema from Logos and removes all domain labels,
    keeping only the relational topology: how many entities, what kinds
    of relations connect them, their arities.
    """

    schema_id: str = ""
    domain: str = ""  # original domain, for reference
    entity_count: int = 0
    relation_types: list[str] = Field(default_factory=list)
    relation_arities: list[int] = Field(default_factory=list)
    pattern_hash: str = ""  # deterministic hash of the relational shape


class CrossDomainMatch(EOSBaseModel):
    """
    A structural isomorphism found between schemas from different domains.

    isomorphism_score > 0.8 = strong match → propose unified schema
    isomorphism_score > 0.9 = submit to Evo as new schema candidate
    """

    id: str = Field(default_factory=new_id)
    schema_a_id: str = ""
    schema_b_id: str = ""
    domain_a: str = ""
    domain_b: str = ""
    isomorphism_score: float = 0.0
    abstract_structure: AbstractStructure = Field(default_factory=AbstractStructure)
    proposed_unified_schema: dict[str, Any] = Field(default_factory=dict)
    mdl_improvement: float = 0.0  # bits saved by unifying


class DreamScenario(EOSBaseModel):
    """A hypothetical scenario generated from world model predictions."""

    id: str = Field(default_factory=new_id)
    domain: str = ""
    scenario_context: dict[str, Any] = Field(default_factory=dict)
    world_model_prediction: dict[str, Any] = Field(default_factory=dict)
    prediction_quality: float = 0.0  # 0.0 = terrible, 1.0 = perfect
    generated_hypotheses: list[dict[str, Any]] = Field(default_factory=list)


class PreAttentionEntry(EOSBaseModel):
    """
    Pre-generated prediction cached for Fovea on wake.

    When Fovea wakes, it has these pre-computed predictions for likely
    contexts, reducing first-cycle prediction latency.
    """

    context_key: str = ""
    domain: str = ""
    predicted_content: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    generating_schema_ids: list[str] = Field(default_factory=list)


class AnalogicalTransfer(EOSBaseModel):
    """
    A causal invariant that applies across multiple domains.

    Same causal structure, different domain labels. The most powerful
    form of knowledge compression: one rule explaining many domains.
    """

    id: str = Field(default_factory=new_id)
    invariant_id: str = ""
    invariant_statement: str = ""
    source_domains: list[str] = Field(default_factory=list)
    domain_count: int = 0
    predictive_transfer_value: float = 0.0  # domain_count × coverage
    mdl_improvement: float = 0.0  # bits saved by replacing domain-specific schemas


class CrossDomainSynthesisReport(EOSBaseModel):
    """Report from cross-domain structural comparison."""

    schemas_compared: int = 0
    domain_pairs_evaluated: int = 0
    strong_matches: int = 0  # score > 0.8
    evo_candidates: int = 0  # score > 0.9
    matches: list[CrossDomainMatch] = Field(default_factory=list)
    total_mdl_improvement: float = 0.0


class DreamGenerationReport(EOSBaseModel):
    """Report from dream generation (constructive simulation)."""

    domains_targeted: int = 0
    scenarios_generated: int = 0
    low_quality_predictions: int = 0  # quality < 0.7
    hypotheses_extracted: int = 0
    pre_attention_entries_cached: int = 0


class AnalogyDiscoveryReport(EOSBaseModel):
    """Report from analogy discovery across causal invariants."""

    invariants_scanned: int = 0
    analogies_found: int = 0
    analogies_applied: int = 0  # top 10 per cycle
    total_mdl_improvement: float = 0.0
    transfers: list[AnalogicalTransfer] = Field(default_factory=list)


class REMStageReport(EOSBaseModel):
    """Complete REM stage report - all three Phase C operations."""

    cross_domain: CrossDomainSynthesisReport = Field(
        default_factory=CrossDomainSynthesisReport
    )
    dreams: DreamGenerationReport = Field(default_factory=DreamGenerationReport)
    analogies: AnalogyDiscoveryReport = Field(default_factory=AnalogyDiscoveryReport)
    duration_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# Phase D - Lucid Dreaming: Mutation Testing Types
# ═══════════════════════════════════════════════════════════════════


class MutationTestResult(EOSBaseModel):
    """Result of testing one scenario against original vs. mutated world model."""

    scenario: DreamScenario = Field(default_factory=DreamScenario)
    original_prediction: dict[str, Any] = Field(default_factory=dict)
    mutated_prediction: dict[str, Any] = Field(default_factory=dict)
    performance_delta: float = 0.0  # positive = mutation is better
    constitutional_violation: bool = False
    violation_detail: str = ""


class MutationSimulationReport(EOSBaseModel):
    """Aggregate report from testing a single mutation proposal."""

    mutation_id: str = ""
    mutation_description: str = ""
    scenarios_tested: int = 0
    results: list[MutationTestResult] = Field(default_factory=list)
    overall_performance_delta: float = 0.0
    any_constitutional_violations: bool = False
    violation_details: list[str] = Field(default_factory=list)
    recommendation: str = "reject"  # "apply" | "reject"


class LucidDreamingReport(EOSBaseModel):
    """Complete Lucid Dreaming report - metacognition + exploration + mutation tests."""

    # Mutation testing
    mutations_tested: int = 0
    mutations_recommended_apply: int = 0
    mutations_recommended_reject: int = 0
    constitutional_violations_found: int = 0
    reports: list[MutationSimulationReport] = Field(default_factory=list)
    # MetaCognition results (Spec 13 §4.5)
    concepts_discovered: int = 0    # Theme clusters promoted to CONCEPT nodes
    # DirectedExploration results (Spec 13 §4.5)
    variations_generated: int = 0   # Systematic variations stored as DreamInsights
    duration_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════
# Phase D - Full Emergence: Wake Preparation Types
# ═══════════════════════════════════════════════════════════════════


class PreAttentionCache(EOSBaseModel):
    """Cache of pre-generated predictions for Fovea's first wake cycles."""

    entries: list[PreAttentionEntry] = Field(default_factory=list)
    domains_covered: int = 0
    total_predictions: int = 0


class SleepNarrative(EOSBaseModel):
    """
    What happened during sleep - composed for Thread.

    The organism's sleep diary: what it compressed, what it discovered,
    what improved, what was dreamed.
    """

    sleep_cycle_id: str = ""
    compression_summary: str = ""
    hypotheses_retired: int = 0
    cross_domain_matches: int = 0
    analogies_discovered: int = 0
    dreams_generated: int = 0
    mutations_tested: int = 0
    intelligence_improvement: float = 0.0
    narrative_text: str = ""  # human-readable summary for Thread


class WakeStatePreparation(EOSBaseModel):
    """Full payload for WAKE_INITIATED broadcast."""

    intelligence_ratio_before: float = 0.0
    intelligence_ratio_after: float = 0.0
    intelligence_improvement: float = 0.0
    world_model_finalized: bool = False
    pre_attention_cache: PreAttentionCache = Field(default_factory=PreAttentionCache)
    sleep_narrative: SleepNarrative = Field(default_factory=SleepNarrative)
    genome_update_prepared: bool = False
    input_channels_resumed: bool = False
    average_intelligence_improvement_per_cycle: float = 0.0


# ─── Sleep Performance Measurement ──────────────────────────────


class SleepOutcome(EOSBaseModel):
    """
    Post-sleep performance comparison result.

    Captures the KPI delta between pre-sleep baseline and the post-sleep
    stabilised state (measured after 100 wake cycles). The verdict drives
    adaptive threshold adjustment and Evo hypothesis generation.
    """

    sleep_cycle_id: str = ""
    sleep_duration_ms: int = 0
    stages_completed: list[str] = Field(default_factory=list)

    # Per-KPI delta: (post - pre) / pre as a fraction (positive = improvement)
    kpi_deltas: dict[str, float] = Field(default_factory=dict)

    # Aggregate signals
    net_improvement: float = 0.0    # mean of positive deltas
    net_degradation: float = 0.0    # |mean of negative deltas|

    # Verdict
    verdict: str = "neutral"        # "beneficial" | "neutral" | "harmful"

    # Adaptive threshold info
    pressure_threshold_adjusted: bool = False
    new_pressure_threshold: float = 0.85

    timestamp: datetime = Field(default_factory=utc_now)
