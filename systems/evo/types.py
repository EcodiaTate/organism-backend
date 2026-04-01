"""
EcodiaOS - Evo Internal Types

All data types internal to the Evo learning system.
These are NOT shared primitives - they model Evo's cognitive structures:
hypotheses, pattern candidates, parameter adjustments, procedures,
consolidation state, and self-model statistics.
"""

from __future__ import annotations

import enum
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import Field

from primitives.affect import AffectState
from primitives.common import (
    EOSBaseModel,
    Identified,
    Timestamped,
    new_id,
    utc_now,
)

# ─── Enums ────────────────────────────────────────────────────────────────────


class PatternType(enum.StrEnum):
    """Categories of patterns Evo can detect."""

    COOCCURRENCE = "cooccurrence"
    ACTION_SEQUENCE = "action_sequence"
    TEMPORAL = "temporal"
    AFFECT_PATTERN = "affect_pattern"


class HypothesisCategory(enum.StrEnum):
    """What kind of claim does this hypothesis make?"""

    WORLD_MODEL = "world_model"    # Claim about external world structure
    SELF_MODEL = "self_model"      # Claim about EOS's own capabilities
    SOCIAL = "social"              # Claim about community member patterns
    PROCEDURAL = "procedural"      # Claim about action sequence effectiveness
    PARAMETER = "parameter"        # Claim about optimal system parameters


class HypothesisStatus(enum.StrEnum):
    """Lifecycle states for a hypothesis."""

    PROPOSED = "proposed"      # Just generated, not yet tested
    TESTING = "testing"        # Accumulating evidence
    SUPPORTED = "supported"    # Evidence_score > threshold AND enough episodes
    REFUTED = "refuted"        # Evidence_score below threshold
    INTEGRATED = "integrated"  # Mutation applied; hypothesis closed
    ARCHIVED = "archived"      # Stale or superseded


class MutationType(enum.StrEnum):
    """What kind of change does a confirmed hypothesis propose?"""

    PARAMETER_ADJUSTMENT = "parameter_adjustment"  # Nudge a system parameter
    PROCEDURE_CREATION = "procedure_creation"       # Codify a successful sequence
    SCHEMA_ADDITION = "schema_addition"             # Add entity/relation type
    EVOLUTION_PROPOSAL = "evolution_proposal"       # Structural change → Simula
    EXPLORATION = "exploration"                     # Low-evidence experiment (Phase 8.5)


class EvidenceDirection(enum.StrEnum):
    """How does a piece of evidence relate to a hypothesis?"""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


# ─── Pattern Candidate ────────────────────────────────────────────────────────


class PatternCandidate(EOSBaseModel):
    """
    A pattern candidate detected during online or offline processing.
    Candidates accumulate into hypotheses when they reach the min_occurrences
    threshold. Raw signal, not yet a claim.
    """

    type: PatternType
    elements: list[str]                                # What was detected
    count: int                                          # How many times seen
    confidence: float = 0.5                             # Detector confidence
    examples: list[str] = Field(default_factory=list)  # Episode IDs (evidence)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_detector: str = ""                          # Which detector produced this


# ─── Pattern Context (mutable accumulator) ────────────────────────────────────


@dataclass
class PatternContext:
    """
    Mutable state accumulated across episodes during wake mode.
    Holds sliding-window counters for all four detector types.
    Reset after each consolidation cycle.

    Not a Pydantic model because it is mutated in-place continuously.
    """

    # CooccurrenceDetector: canonical_pair_key → count
    # Key format: "{entity_a}::{entity_b}" (sorted for stability)
    cooccurrence_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    # SequenceDetector: sequence_hash → count
    sequence_counts: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    # SequenceDetector: sequence_hash → [episode_id, ...]
    sequence_examples: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # TemporalDetector: "{source}::h{hour}" or "{source}::d{weekday}" → count
    temporal_bins: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    # Temporal baselines: source_type → expected count per bin
    temporal_baselines: dict[str, float] = field(default_factory=dict)

    # AffectPatternDetector: stimulus_type → [(valence_delta, arousal_delta), ...]
    affect_responses: dict[str, list[tuple[float, float]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Current affect (set by service before each scan, used by affect detector)
    previous_affect: AffectState | None = None
    current_affect: AffectState | None = None

    # Recent entity IDs from the last workspace broadcast (CooccurrenceDetector)
    recent_entity_ids: list[str] = field(default_factory=list)

    # Running episode counter since last reset
    episodes_scanned: int = 0

    def get_mature_sequences(self, min_occurrences: int = 3) -> list[PatternCandidate]:
        """Return action sequence candidates that have met the threshold."""
        candidates: list[PatternCandidate] = []
        for seq_hash, count in self.sequence_counts.items():
            if count >= min_occurrences:
                candidates.append(
                    PatternCandidate(
                        type=PatternType.ACTION_SEQUENCE,
                        elements=[seq_hash],
                        count=count,
                        confidence=min(0.9, 0.5 + count * 0.05),
                        examples=self.sequence_examples.get(seq_hash, [])[:10],
                        metadata={"sequence_hash": seq_hash},
                    )
                )
        return candidates

    def reset(self) -> None:
        """Reset all counters. Called after each consolidation cycle."""
        self.cooccurrence_counts.clear()
        self.sequence_counts.clear()
        self.sequence_examples.clear()
        self.temporal_bins.clear()
        self.temporal_baselines.clear()
        self.affect_responses.clear()
        self.recent_entity_ids.clear()
        self.episodes_scanned = 0
        self.previous_affect = None
        self.current_affect = None


# ─── Mutation ─────────────────────────────────────────────────────────────────


class Mutation(EOSBaseModel):
    """
    A proposed change to the organism's model, parameters, or structure.
    Attached to a Hypothesis; applied only when hypothesis status = SUPPORTED.
    """

    type: MutationType
    target: str          # Param name, procedure name, or schema element
    value: float = 0.0   # Delta for parameter adjustments; ignored for others
    description: str = ""


# ─── Hypothesis ───────────────────────────────────────────────────────────────


class Hypothesis(Identified, Timestamped):
    """
    A testable hypothesis about the world, self, or processing parameters.
    Stored as a :Hypothesis node in the Memory graph.

    Lifecycle:
      proposed → testing → supported | refuted → integrated | archived

    Evidence scoring follows approximate Bayesian model comparison:
      evidence_score += strength * (1 - complexity_penalty * 0.1)  [for support]
      evidence_score -= strength                                     [for contradiction]

    Integration thresholds (from VELOCITY_LIMITS):
      - evidence_score > 3.0
      - len(supporting_episodes) >= 10
      - hypothesis age >= 24 hours
    """

    category: HypothesisCategory
    statement: str                 # Natural language claim
    formal_test: str               # How we would falsify this

    # Structured repair metadata (populated for PROCEDURAL hypotheses only)
    repair_endpoint: str = ""      # Endpoint/subject that failed (e.g. "/api/v1/logos")
    repair_fix_type: str = ""      # Normalised fix identifier (e.g. "add_route_handler")

    # Evidence tracking
    supporting_episodes: list[str] = Field(default_factory=list)
    contradicting_episodes: list[str] = Field(default_factory=list)
    evidence_score: float = 0.0
    last_evidence_at: datetime = Field(default_factory=utc_now)

    # Lifecycle
    status: HypothesisStatus = HypothesisStatus.PROPOSED

    # Which detector/source generated this hypothesis (for meta-learning)
    source_detector: str = ""

    # Occam's razor - simpler hypotheses are preferred
    complexity_penalty: float = 0.1

    # Minimum age before integration is allowed.
    # Repair hypotheses use 4.0 h (ground-truth, fast convergence);
    # all others default to the spec-defined 24 h gate (VELOCITY_LIMITS).
    min_age_hours: float = 24.0

    # Novelty score: 1.0 = completely novel, 0.0 = redundant with existing.
    # Computed as 1.0 - max_similarity against confirmed hypothesis embeddings.
    # Required for Bedau-Packard "new activities" metric.
    novelty_score: float = 0.0

    # What to apply if hypothesis reaches SUPPORTED
    proposed_mutation: Mutation | None = None

    # ── Exploration Hypotheses (Phase 8.5 gap closure) ───────────────────────
    # For low-evidence, low-confidence hypotheses that fast-track to Simula
    # exploration pipeline instead of waiting for full evidence accumulation.
    is_exploration: bool = False            # Marks this hypothesis as exploration path
    exploration_budget_usd: float = 0.0     # Hard cap on total spend for this exploration
    exploration_attempts: int = 0           # Times we've tried this exploration
    exploration_max_attempts: int = 3       # Give up after N failures
    exploration_outcomes: list[str] = Field(default_factory=list)  # Brief outcome descriptions

    # ── Volatility tracking ──────────────────────────────────────────────────
    # confidence_flip_log records the direction (+1 up / -1 down) of the last
    # N outcome nudges. When the sign flips repeatedly the hypothesis is
    # oscillating and becomes less trustworthy.
    # Excluded from JSON serialisation (ephemeral, in-memory only).
    confidence_flip_log: list[int] = Field(default_factory=list, exclude=True)
    confidence_oscillations: int = 0        # Count of direction reversals
    volatility_flag: str = "normal"         # "normal" | "HIGH_VOLATILITY"
    volatility_weight: float = 1.0          # Multiplier applied to outcome nudges


# ─── Evidence Result ──────────────────────────────────────────────────────────


class EvidenceResult(EOSBaseModel):
    """Result of evaluating a single episode against a hypothesis."""

    hypothesis_id: str
    episode_id: str
    direction: EvidenceDirection
    strength: float = 0.0
    reasoning: str = ""
    new_score: float = 0.0
    new_status: HypothesisStatus = HypothesisStatus.TESTING


# ─── Parameter Tuning ─────────────────────────────────────────────────────────


class ParameterSpec(EOSBaseModel):
    """Defines the valid range and step size for a tunable parameter."""

    min_val: float
    max_val: float
    step: float


class ParameterAdjustment(EOSBaseModel):
    """A proposed or applied adjustment to a system parameter."""

    parameter: str
    old_value: float
    new_value: float
    delta: float = 0.0
    hypothesis_id: str
    evidence_score: float
    supporting_count: int
    applied_at: datetime = Field(default_factory=utc_now)


# ─── Procedures ───────────────────────────────────────────────────────────────


class ProcedureStep(EOSBaseModel):
    """One step in a procedural memory."""

    action_type: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_duration_ms: int = 1000


class Procedure(Identified, Timestamped):
    """
    A reusable action sequence extracted from successful episodes.
    Stored as :Procedure nodes in the Memory graph.
    These become the "habits" Nova's fast path can use.
    """

    name: str
    preconditions: list[str] = Field(default_factory=list)
    steps: list[ProcedureStep] = Field(default_factory=list)
    postconditions: list[str] = Field(default_factory=list)
    success_rate: float = 1.0          # Updated as procedure is used
    source_episodes: list[str] = Field(default_factory=list)
    usage_count: int = 0


# ─── Schema Induction ─────────────────────────────────────────────────────────


class SchemaInduction(EOSBaseModel):
    """
    A proposed structural change to the Memory graph's schema.
    New entity types, relation types, or community patterns from regularities.
    """

    entities: list[dict[str, str]] = Field(default_factory=list)
    relations: list[dict[str, str]] = Field(default_factory=list)
    communities: list[dict[str, str]] = Field(default_factory=list)
    source_hypothesis: str = ""


# ─── Evolution Proposals ──────────────────────────────────────────────────────


class EvolutionProposal(EOSBaseModel):
    """
    A structural change proposal submitted to Simula.
    Evo can propose; Simula gates the actual change.
    """

    description: str
    rationale: str
    supporting_hypotheses: list[str] = Field(default_factory=list)
    proposed_at: datetime = Field(default_factory=utc_now)


# ─── Exploration Proposals (Phase 8.5 gap closure) ───────────────────────────


class ExplorationProposal(EOSBaseModel):
    """
    Lightweight proposal for low-evidence, low-confidence exploration.

    Submitted to Simula's lightweight pipeline which skips SIMULATE
    and goes straight to VALIDATE → GATE → APPLY → VERIFY → RECORD.

    Budget is hard-capped; Equor still gates for constitutional alignment.
    """

    hypothesis_id: str                      # Link back to the originating hypothesis
    hypothesis_statement: str               # Natural language claim being explored
    evidence_score: float                   # Current evidence (2.0–5.0 range)
    proposed_mutation: Mutation | None = None  # The proposed change (lightweight)
    budget_usd: float                       # Hard cap on spending for this exploration
    max_attempts: int = 3                   # Give up after N failures
    success_criteria: str = ""              # Natural language description of success
    rollback_plan: str = ""                 # Mitigation if exploration fails
    metabolic_tier: str = "NOMINAL"         # Starvation level when proposal was generated


# ─── Self-Model ───────────────────────────────────────────────────────────────


class CapabilityScore(EOSBaseModel):
    """Success rate for a specific named capability."""

    capability: str
    success_count: int = 0
    total_count: int = 0

    @property
    def rate(self) -> float:
        return self.success_count / max(1, self.total_count)


class RegretStats(EOSBaseModel):
    """
    Aggregated regret statistics from resolved counterfactual episodes.

    Computed during Evo Phase 6 (self-model update) by mining Neo4j
    :Counterfactual nodes that have been resolved with outcome data.
    Feeds into hypothesis generation for policy-selection calibration.
    """

    mean_regret: float = 0.0
    regret_by_policy_type: dict[str, float] = Field(default_factory=dict)
    regret_by_goal_domain: dict[str, float] = Field(default_factory=dict)
    total_resolved: int = 0
    high_regret_count: int = 0  # |regret| > 0.3


class SelfModelStats(EOSBaseModel):
    """
    What EOS knows about itself: overall effectiveness and per-capability scores.
    Updated during each consolidation cycle.
    """

    success_rate: float = 0.5
    mean_alignment: float = 0.5
    total_outcomes_evaluated: int = 0
    capability_scores: dict[str, CapabilityScore] = Field(default_factory=dict)
    regret: RegretStats = Field(default_factory=RegretStats)
    updated_at: datetime = Field(default_factory=utc_now)


# ─── Belief Consolidation (Phase 2.75) ────────────────────────────────────────


# ─── Genetic Memory (Phase 2.8 - Belief Inheritance) ─────────────────────────


class InheritedHypothesisRecord(EOSBaseModel):
    """
    A single hypothesis serialized for genome transmission.

    Captures the essential claim, its domain, and the confidence at fixation time.
    Evidence details are compressed away - the child inherits the *conclusion*,
    not the full evidentiary history.
    """

    domain: str                          # e.g., "market_property", "capability"
    category: str                        # HypothesisCategory value
    statement: str                       # Natural language claim
    formal_test: str = ""                # Falsification condition
    confidence: float                    # Confidence at fixation (>= 0.95)
    volatility: float = 0.0             # Low = stable (< 0.1 for fixation)
    age_days: float = 0.0               # Age at fixation time
    parameters: dict[str, Any] = Field(default_factory=dict)
    content_hash: str = ""               # SHA-256 of canonical (domain + statement)


class BeliefGenome(Identified, Timestamped):
    """
    Compressed belief inheritance package passed from parent to child instance.

    A mature instance (>10,000 episodes, >100 confirmed hypotheses) serializes
    its stable, high-confidence beliefs into this genome. Child instances
    decompress it at birth and seed their hypothesis engine, skipping redundant
    re-learning.

    Neo4j representation: :BeliefGenome node with INHERITED_FROM relationship
    to the parent :Self node, and SEEDED_BY back-links from child instances.
    """

    parent_instance_id: str              # Which instance generated this genome
    stable_hypotheses: list[InheritedHypothesisRecord] = Field(default_factory=list)
    compression_method: str = "lz4"      # "lz4", "zlib", "json+schema_hash"
    version: int = 1                     # Genome format version (for compatibility)
    generation: int = 1                  # Lineage depth (parent=1, grandchild=3)
    parent_ids: list[str] = Field(default_factory=list)  # Full lineage chain
    total_episodes_at_fixation: int = 0  # Parent's episode count when genome was built
    total_hypotheses_confirmed: int = 0  # Parent's confirmed hypothesis count
    genome_size_bytes: int = 0           # Compressed payload size


class GenomeExtractionResult(EOSBaseModel):
    """Summary of one genetic fixation pass during Evo Phase 2.8."""

    candidates_scanned: int = 0
    candidates_fixed: int = 0            # Hypotheses meeting fixation criteria
    genome_size_bytes: int = 0
    genome_id: str = ""
    duration_ms: int = 0


class GenomeInheritanceReport(EOSBaseModel):
    """
    Monitoring report for genome inheritance fidelity in a child instance.

    Tracks how the child's learning diverges from (or confirms) inherited beliefs.
    """

    parent_genome_id: str = ""
    total_inherited: int = 0
    kept_unchanged: int = 0              # Child confirmed parent's belief
    downgraded: int = 0                  # Child lowered confidence
    refuted: int = 0                     # Child contradicted parent's belief
    novel_hypotheses: int = 0            # Child's new discoveries not in genome
    learning_speedup_pct: float = 0.0    # % faster than cold-start baseline


# ─── Hypothesis Tournament (Thompson Sampling A/B) ───────────────────────────


class BetaDistribution(EOSBaseModel):
    """
    Beta distribution parameterisation for Thompson sampling.
    α counts successes, β counts failures. Prior is Beta(1, 1) = uniform.
    """

    alpha: float = 1.0   # successes + prior
    beta: float = 1.0    # failures + prior

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def sample_count(self) -> int:
        """Total observations (excluding the prior)."""
        return int(self.alpha + self.beta - 2)  # subtract uniform prior (1, 1)

    def update_success(self) -> None:
        self.alpha += 1.0

    def update_failure(self) -> None:
        self.beta += 1.0


class HypothesisRef(EOSBaseModel):
    """Lightweight reference to a hypothesis competing in a tournament."""

    hypothesis_id: str
    statement: str = ""
    evidence_score: float = 0.0


class TournamentStage(enum.StrEnum):
    """Lifecycle stages for a hypothesis tournament."""

    RUNNING = "running"
    CONVERGED = "converged"
    ARCHIVED = "archived"


class HypothesisTournament(Identified, Timestamped):
    """
    A competitive A/B experiment between 2–4 hypotheses with similar fitness.

    Uses Thompson sampling (Beta-Bernoulli conjugate model) to route real
    decision contexts to competing hypotheses and incrementally converge
    on the winner with minimal regret.

    Lifecycle: running → converged → archived
    """

    hypotheses: list[HypothesisRef] = Field(default_factory=list)
    stage: TournamentStage = TournamentStage.RUNNING
    sample_count: int = 0  # Total decision contexts routed
    beta_parameters: dict[str, BetaDistribution] = Field(default_factory=dict)
    winner_id: str | None = None
    convergence_threshold: float = 0.95  # Posterior prob required to declare winner
    burn_in_trials: int = 10  # First N trials are 50/50 regardless of posterior

    @property
    def is_converged(self) -> bool:
        return self.stage == TournamentStage.CONVERGED

    @property
    def is_running(self) -> bool:
        return self.stage == TournamentStage.RUNNING


class TournamentContext(EOSBaseModel):
    """
    Metadata attached to a policy when its source hypothesis is in a tournament.
    Emitted by Nova's policy selection phase; consumed by Evo when outcomes arrive.
    """

    tournament_id: str
    hypothesis_id: str   # Which hypothesis was selected for this trial
    policy_id: str = ""  # The policy that was dispatched
    sampled_at: datetime = Field(default_factory=utc_now)


class TournamentOutcome(EOSBaseModel):
    """
    Outcome of a single tournament trial, reported by Axon after execution.
    """

    tournament_id: str
    hypothesis_id: str
    success: bool
    intent_id: str = ""
    recorded_at: datetime = Field(default_factory=utc_now)


# ─── Belief Consolidation (Phase 2.75) ────────────────────────────────────────


class ConsolidatedBelief(EOSBaseModel):
    """
    A hardened, read-only reference copy of a high-confidence, low-volatility belief.

    Created during Evo consolidation Phase 2.75. Once a belief is consolidated,
    its original :Belief node is marked as superseded and this node becomes the
    authoritative reference. Retrieval paths (Nova, Atune) should prefer
    ConsolidatedBelief nodes for faster, safer reads.
    """

    id: str = Field(default_factory=new_id)
    source_belief_id: str                    # Original :Belief node ID
    domain: str = ""
    statement: str = ""
    precision: float = 0.0                   # Confidence at consolidation time
    consolidated_at: datetime = Field(default_factory=utc_now)
    consolidation_generation: int = 1        # Incremented on belief upgrades
    mutable: bool = False                    # Always False - app-level immutability


class FoundationConflict(EOSBaseModel):
    """A hypothesis that contradicts a consolidated (foundational) belief."""

    hypothesis_id: str
    consolidated_belief_id: str
    hypothesis_statement: str = ""
    consolidated_statement: str = ""
    severity: str = "high"


class BeliefConsolidationResult(EOSBaseModel):
    """Summary of one belief-consolidation pass during Evo Phase 2.75."""

    beliefs_scanned: int = 0
    beliefs_consolidated: int = 0
    foundation_conflicts: int = 0
    conflicts: list[FoundationConflict] = Field(default_factory=list)
    duration_ms: int = 0


# ─── Consolidation ────────────────────────────────────────────────────────────


class ConsolidationResult(EOSBaseModel):
    """Summary of what happened during one consolidation cycle."""

    duration_ms: int = 0
    hypotheses_evaluated: int = 0
    hypotheses_integrated: int = 0
    hypotheses_archived: int = 0
    procedures_extracted: int = 0
    schemas_induced: int = 0
    parameters_adjusted: int = 0
    total_parameter_delta: float = 0.0
    self_model_updated: bool = False
    triggered_at: datetime = Field(default_factory=utc_now)
    # Belief half-life aging (Phase 2.5)
    beliefs_stale: int = 0
    beliefs_critical: int = 0
    # Belief consolidation (Phase 2.75)
    beliefs_consolidated: int = 0
    foundation_conflicts: int = 0
    # Hypothesis tournaments (Phase 2 - tournament update)
    tournaments_active: int = 0
    tournaments_converged: int = 0
    # Genetic memory (Phase 2.8 - belief inheritance)
    genome_candidates_fixed: int = 0
    genome_size_bytes: int = 0
    # Cognitive speciation (Phase 2.9 - speciation + niche forking)
    niches_created: int = 0
    niches_extinct: int = 0
    speciation_events: int = 0
    ring_species_detected: int = 0
    niche_forks_proposed: int = 0
    worldview_forks: int = 0
    # Exploration hypotheses (Phase 8.5 - gap closure)
    exploration_proposals_generated: int = 0
    explorations_skipped: int = 0
    # Telos topology-aware hypothesis prioritisation (Phase D integration)
    telos_hypothesis_rankings: list[dict[str, Any]] = Field(default_factory=list)


# ─── Constants ────────────────────────────────────────────────────────────────


# All parameters Evo is permitted to adjust (spec Section V)
TUNABLE_PARAMETERS: dict[str, ParameterSpec] = {
    # Atune - salience head weights
    "atune.head.novelty.weight":     ParameterSpec(min_val=0.05, max_val=0.40, step=0.01),
    "atune.head.risk.weight":        ParameterSpec(min_val=0.05, max_val=0.40, step=0.01),
    "atune.head.identity.weight":    ParameterSpec(min_val=0.05, max_val=0.30, step=0.01),
    "atune.head.goal.weight":        ParameterSpec(min_val=0.05, max_val=0.30, step=0.01),
    "atune.head.emotional.weight":   ParameterSpec(min_val=0.05, max_val=0.30, step=0.01),
    "atune.head.causal.weight":      ParameterSpec(min_val=0.05, max_val=0.25, step=0.01),
    "atune.head.keyword.weight":     ParameterSpec(min_val=0.05, max_val=0.25, step=0.01),
    # Nova - EFE weights
    "nova.efe.pragmatic":            ParameterSpec(min_val=0.15, max_val=0.55, step=0.02),
    "nova.efe.epistemic":            ParameterSpec(min_val=0.05, max_val=0.40, step=0.02),
    "nova.efe.constitutional":       ParameterSpec(min_val=0.10, max_val=0.40, step=0.02),
    "nova.efe.feasibility":          ParameterSpec(min_val=0.05, max_val=0.30, step=0.02),
    "nova.efe.risk":                 ParameterSpec(min_val=0.05, max_val=0.25, step=0.02),
    # Voxis - personality vector
    "voxis.personality.warmth":      ParameterSpec(min_val=-1.0, max_val=1.0, step=0.03),
    "voxis.personality.directness":  ParameterSpec(min_val=-1.0, max_val=1.0, step=0.03),
    "voxis.personality.verbosity":   ParameterSpec(min_val=-1.0, max_val=1.0, step=0.03),
    "voxis.personality.formality":   ParameterSpec(min_val=-1.0, max_val=1.0, step=0.03),
    "voxis.personality.humour":      ParameterSpec(min_val=0.0,  max_val=1.0, step=0.03),
    # Memory - salience model weights
    "memory.salience.recency":       ParameterSpec(min_val=0.10, max_val=0.40, step=0.02),
    "memory.salience.frequency":     ParameterSpec(min_val=0.05, max_val=0.25, step=0.02),
    "memory.salience.affect":        ParameterSpec(min_val=0.05, max_val=0.30, step=0.02),
    "memory.salience.surprise":      ParameterSpec(min_val=0.05, max_val=0.25, step=0.02),
    "memory.salience.relevance":     ParameterSpec(min_val=0.10, max_val=0.40, step=0.02),
    # Nova - free energy budget (information-theoretic pressure valve)
    "nova.fe_budget.budget_nats":        ParameterSpec(min_val=1.0, max_val=20.0, step=0.5),
    "nova.fe_budget.threshold_fraction": ParameterSpec(min_val=0.5, max_val=0.95, step=0.05),
    # Nova - cognition cost (metabolic budgeting, λ frugality weight)
    "nova.efe.cognition_cost":           ParameterSpec(min_val=0.0, max_val=0.30, step=0.02),

    # Belief half-life tuning (Spec §VIII gap fix - learnable domain half-lives).
    # Values are half-life in days. Evo parameter hypotheses can shift these as
    # the organism observes how quickly beliefs in each domain actually become stale.
    "belief.halflife.sentiment":   ParameterSpec(min_val=0.1, max_val=2.0, step=0.05),
    "belief.halflife.preference":  ParameterSpec(min_val=3.0, max_val=30.0, step=0.5),
    "belief.halflife.capability":  ParameterSpec(min_val=30.0, max_val=180.0, step=5.0),
    "belief.halflife.context":     ParameterSpec(min_val=0.5, max_val=14.0, step=0.25),
    "belief.halflife.social":      ParameterSpec(min_val=7.0, max_val=90.0, step=2.0),
    "belief.halflife.policy":      ParameterSpec(min_val=30.0, max_val=365.0, step=10.0),

    # Atune workspace - evolvable curiosity / spontaneous recall rhythm.
    # Higher base_prob → more frequent spontaneous recall (wider curiosity).
    # Lower cooldown_cycles → recalls can fire more frequently.
    # Higher curiosity_boost → affect.curiosity has stronger influence on recall.
    "atune.workspace.base_prob":       ParameterSpec(min_val=0.005, max_val=0.10, step=0.003),
    "atune.workspace.cooldown_cycles": ParameterSpec(min_val=5.0, max_val=50.0, step=1.0),
    "atune.workspace.curiosity_boost": ParameterSpec(min_val=0.01, max_val=0.10, step=0.005),
}

# Default initial values (mid-range or from spec defaults)
PARAMETER_DEFAULTS: dict[str, float] = {
    "atune.head.novelty.weight":     0.20,
    "atune.head.risk.weight":        0.20,
    "atune.head.identity.weight":    0.15,
    "atune.head.goal.weight":        0.15,
    "atune.head.emotional.weight":   0.15,
    "atune.head.causal.weight":      0.10,
    "atune.head.keyword.weight":     0.05,
    "nova.efe.pragmatic":            0.35,
    "nova.efe.epistemic":            0.20,
    "nova.efe.constitutional":       0.20,
    "nova.efe.feasibility":          0.15,
    "nova.efe.risk":                 0.10,
    "voxis.personality.warmth":      0.0,
    "voxis.personality.directness":  0.0,
    "voxis.personality.verbosity":   0.0,
    "voxis.personality.formality":   0.0,
    "voxis.personality.humour":      0.0,
    "memory.salience.recency":       0.25,
    "memory.salience.frequency":     0.15,
    "memory.salience.affect":        0.20,
    "memory.salience.surprise":      0.15,
    "memory.salience.relevance":     0.25,
    "nova.fe_budget.budget_nats":        5.0,
    "nova.fe_budget.threshold_fraction": 0.8,
    "nova.efe.cognition_cost":           0.10,
    # Belief half-life defaults (days) - match DEFAULT_DOMAIN_HALFLIFES in belief_halflife.py
    "belief.halflife.sentiment":   0.3,
    "belief.halflife.preference":  14.0,
    "belief.halflife.capability":  90.0,
    "belief.halflife.context":     2.0,
    "belief.halflife.social":      14.0,
    "belief.halflife.policy":      90.0,
    # Atune workspace curiosity defaults
    "atune.workspace.base_prob":       0.02,
    "atune.workspace.cooldown_cycles": 20.0,
    "atune.workspace.curiosity_boost": 0.03,
}

# Change velocity limits (spec Section IX)
VELOCITY_LIMITS: dict[str, Any] = {
    "max_total_parameter_delta_per_cycle": 0.15,
    "max_single_parameter_delta":          0.03,
    "min_evidence_for_integration":        10,
    "min_hypothesis_age_hours":            24,
    "max_active_hypotheses":               50,
    "max_new_procedures_per_cycle":        3,
}

# What Evo cannot touch (spec Section IX)
EVO_CONSTRAINTS: dict[str, str] = {
    "equor_evaluation":          "forbidden",
    "constitutional_drives":     "forbidden",
    "invariants":                "forbidden",
    "self_evaluation_criteria":  "forbidden",
    "parameters":                "permitted_within_range",
    "knowledge_structures":      "permitted",
    "evolution_proposals":       "permitted_as_proposal",
}


# ─── Curiosity-Driven Exploration ────────────────────────────────────────────


class EpistemicIntent(EOSBaseModel):
    """
    An action the organism takes purely to gather evidence for a hypothesis,
    not to achieve a pragmatic user goal.

    Priority is computed as: EIG × hypothesis_importance × (1 / cost_estimate).
    Emitted via Synapse as EVO_EPISTEMIC_INTENT_PROPOSED.
    """

    hypothesis_id: str
    hypothesis_statement: str = ""
    expected_information_gain: float = 0.0  # KL-divergence: current vs expected posterior
    hypothesis_importance: float = 0.5      # From Telos drive topology or evidence score
    cost_estimate: float = 1.0              # Estimated cost of running this probe
    priority: float = 0.0                   # EIG × importance × (1/cost)
    proposed_action: str = ""               # Natural-language description of the probe
    target_domain: str = ""                 # Which domain the hypothesis lives in


class CuriosityState(EOSBaseModel):
    """Snapshot of the curiosity engine's internal state."""

    total_intents_proposed: int = 0
    total_intents_executed: int = 0
    exploration_ratio: float = 0.2          # Current epistemic / total ratio
    target_exploration_ratio: float = 0.2   # Target ~20% epistemic / 80% pragmatic
    active_hypotheses_seeking: int = 0      # Hypotheses in active evidence-seeking mode
    soma_curiosity_multiplier: float = 1.0  # From Soma curiosity drive


# ─── Evolutionary Pressure System ───────────────────────────────────────────


class FitnessScore(EOSBaseModel):
    """Fitness score for a knowledge unit (hypothesis, schema, or procedure)."""

    entity_id: str
    entity_type: str  # "hypothesis" | "schema" | "procedure" | "parameter"
    fitness: float = 0.0
    prediction_accuracy: float = 0.0
    compression_ratio: float = 0.0
    age_survival_days: float = 0.0
    reuse_frequency: int = 0
    metabolic_cost: float = 0.0  # Resource cost of maintaining this knowledge


class SelectionEvent(EOSBaseModel):
    """Record of a selection event (tournament, pruning, or extinction)."""

    event_type: str  # "tournament" | "fitness_proportionate" | "extinction" | "niche"
    selected_ids: list[str] = Field(default_factory=list)
    pruned_ids: list[str] = Field(default_factory=list)
    reason: str = ""
    population_fitness_mean: float = 0.0
    population_fitness_std: float = 0.0


class CognitiveSpecies(EOSBaseModel):
    """
    A coherent worldview module - a cluster of hypotheses/schemas/procedures
    that have diverged enough from the main population to be declared a
    separate cognitive 'species'.
    """

    id: str = Field(default_factory=new_id)
    name: str = ""
    member_ids: list[str] = Field(default_factory=list)
    centroid_domain: str = ""           # Primary domain this species covers
    mean_fitness: float = 0.0
    diversity_score: float = 0.0        # Internal diversity within the species
    graph_distance_from_main: float = 0.0  # How far from the general population
    discovered_at: datetime = Field(default_factory=utc_now)


class PressureState(EOSBaseModel):
    """Snapshot of the evolutionary pressure system."""

    total_selection_events: int = 0
    total_extinctions: int = 0
    total_species_detected: int = 0
    population_mean_fitness: float = 0.0
    population_std_fitness: float = 0.0
    metabolic_pressure: float = 0.0     # From Oikos - high pressure = aggressive pruning
    active_species: list[CognitiveSpecies] = Field(default_factory=list)


# ─── Recursive Self-Modification ────────────────────────────────────────────


class DetectorReplacementProposal(EOSBaseModel):
    """
    Proposal to replace a failing pattern detector with a new one.

    Generated when a detector's effectiveness drops below 0.1 for 5 consecutive
    cycles. Contains a natural-language specification that Simula can use
    to synthesize a replacement detector.
    """

    old_detector_name: str
    specification: str           # NL description of what the new detector should catch
    underserved_categories: list[str] = Field(default_factory=list)
    effectiveness_history: list[float] = Field(default_factory=list)
    proposed_at: datetime = Field(default_factory=utc_now)


class LearningArchitectureProposal(EOSBaseModel):
    """
    A proposal for the organism to modify its own learning architecture.

    Generated when meta-learning detects a systematic failure mode. These
    proposals are the organism REQUESTING its own cognitive upgrade.
    """

    failure_mode: str                    # What systematic failure was detected
    proposed_change: str                 # NL description of the architectural change
    expected_improvement: str = ""       # What should improve
    supporting_evidence: list[str] = Field(default_factory=list)  # Hypothesis IDs
    success_probability: float = 0.5    # Estimated chance of success
    proposed_at: datetime = Field(default_factory=utc_now)


class SelfModificationRecord(EOSBaseModel):
    """Record of a self-modification that was proposed and its outcome."""

    # "detector_replacement" | "evidence_adaptation" | "schedule_adaptation" | "architecture"
    proposal_type: str
    description: str
    applied: bool = False
    outcome: str = ""      # "success" | "failure" | "pending"
    metric_delta: float = 0.0  # Change in the target metric after application
    proposed_at: datetime = Field(default_factory=utc_now)
    resolved_at: datetime | None = None


class ConsolidationSchedule(EOSBaseModel):
    """Adaptive consolidation schedule learned by meta-learning."""

    current_interval_hours: float = 6.0
    min_interval_hours: float = 1.0
    max_interval_hours: float = 24.0
    hypothesis_throughput: float = 0.0      # Hypotheses per hour
    integration_success_rate: float = 0.5
    schema_discovery_rate: float = 0.0      # Schemas per consolidation cycle
    last_adaptation_reason: str = ""


# ─── Schema Algebra ─────────────────────────────────────────────────────────


class SchemaVersion(EOSBaseModel):
    """Version tracking for evolving schemas."""

    schema_id: str
    version: int = 1
    parent_version: int = 0
    fitness: float = 0.0       # How much compression this version provides
    instance_coverage: int = 0  # How many graph nodes this version covers
    active: bool = True
    created_at: datetime = Field(default_factory=utc_now)


class SchemaRelationship(EOSBaseModel):
    """A relationship between two schemas in the schema DAG."""

    parent_id: str
    child_id: str
    relationship_type: str  # "SPECIALIZES" | "COMPOSED_OF" | "ABSTRACTS"
    mdl_gain: float = 0.0  # MDL gain from this relationship
    properties_inherited: int = 0
    properties_added: int = 0


class CrossDomainTransfer(EOSBaseModel):
    """A proposed transfer of schema structure across domains."""

    source_schema_id: str
    source_domain: str
    target_domain: str
    isomorphism_score: float = 0.0
    proposed_abstract_schema: str = ""  # Name of the domain-independent schema
    mdl_gain: float = 0.0


# ─── Helpers ──────────────────────────────────────────────────────────────────


def hash_sequence(sequence: list[str]) -> str:
    """Stable, deterministic hash of an action sequence."""
    canonical = json.dumps(sequence, sort_keys=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
