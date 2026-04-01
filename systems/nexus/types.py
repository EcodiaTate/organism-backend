"""
EcodiaOS - Nexus Type Definitions

All data types for epistemic triangulation: shareable world model fragments,
convergence detection, triangulation metadata, divergence measurement,
and speciation incentives.

Core insight: instances should not share beliefs - they share the structure
beneath beliefs. Convergence across maximally diverse compression paths is
the primary evidence for ground truth.
"""

from __future__ import annotations

import enum
from datetime import datetime
import math
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, Identified, new_id, utc_now

# ─── Enums ────────────────────────────────────────────────────────


class DivergenceClassification(enum.StrEnum):
    """Classification of how different two instances are."""

    SAME_KIND = "same_kind"          # overall < 0.2 - near-duplicate, zero triangulation value
    RELATED_KIND = "related_kind"    # overall < 0.5 - same species, different subspecies
    DISTINCT_KIND = "distinct_kind"  # overall >= 0.5 - true speciation threshold
    ALIEN_KIND = "alien_kind"        # overall >= 0.8 - convergence from here is near-proof


class FragmentShareOutcome(enum.StrEnum):
    """Outcome of attempting to share a fragment."""

    ACCEPTED = "accepted"
    REJECTED_LOW_QUALITY = "rejected_low_quality"
    REJECTED_NO_SLEEP_CERT = "rejected_no_sleep_cert"
    REJECTED_DUPLICATE = "rejected_duplicate"
    REJECTED_TRUST_INSUFFICIENT = "rejected_trust_insufficient"


# ─── Sleep Certification ──────────────────────────────────────────


class SleepCertification(EOSBaseModel):
    """
    Proof that a fragment survived offline consolidation.

    Only fragments that have survived at least one full sleep cycle
    (slow-wave compression + REM creative recombination) are eligible
    for sharing. This prevents broadcasting ephemeral noise.
    """

    survived_slow_wave: bool = False
    survived_rem: bool = False
    sleep_cycles_survived: int = 0

    @property
    def is_certified(self) -> bool:
        """A fragment must survive at least one complete sleep cycle."""
        return self.survived_slow_wave and self.survived_rem and self.sleep_cycles_survived >= 1


# ─── Abstract Structure (Formal) ─────────────────────────────────


class AbstractStructure(EOSBaseModel):
    """
    Formally typed representation of a world model fragment's abstract
    relational skeleton.

    Replaces the previous ``dict[str, Any]`` used as
    ``ShareableWorldModelFragment.abstract_structure`` so the wire format is
    typed, validated, and directly comparable by ConvergenceDetector.

    Domain-specific labels are stripped before building this object.  The
    structure captures SHAPE, not CONTENT.

    Gap HIGH-2 (Federation Spec, 2026-03-07): formalised from dict[str, Any].
    Gap MEDIUM-6 (Federation Spec, 2026-03-07): ``sleep_certified`` flag
      added for the sleep certification handshake gate.
    """

    node_count: int = 0
    """Total number of nodes in the relational graph."""

    edge_count: int = 0
    """Total number of directed or undirected edges."""

    schema_type: str = ""
    """
    High-level topology class:
    "chain" | "cycle" | "star" | "tree" | "complete" | "bipartite" |
    "lattice" | "arbitrary".
    """

    compression_ratio: float = 0.0
    """MDL compression ratio achieved for this structure (higher = more parsimonious)."""

    provenance_hash: str = ""
    """
    SHA-256 of the canonical abstract structure JSON.
    Used for structural dedup across the federation.
    """

    sleep_certified: bool = False
    """
    True only after this fragment has survived at least one complete
    Oneiros sleep cycle (slow-wave + REM consolidation).

    Fragments are NOT eligible for WORLD_MODEL_FRAGMENT_SHARE emission
    until this flag is True.  Set by NexusService when
    ONEIROS_CONSOLIDATION_COMPLETE fires with this fragment's schema_id
    in the payload.

    Gap MEDIUM-6: sleep certification handshake.
    """

    node_type_distribution: dict[str, int] = Field(default_factory=dict)
    """Count of each node type label (e.g. {"entity": 3, "event": 2})."""

    edge_type_distribution: dict[str, int] = Field(default_factory=dict)
    """Count of each edge type label (e.g. {"causal": 4, "temporal": 1})."""

    symmetry: str = ""
    """Declared symmetry class (``chain`` | ``cycle`` | ``star`` | ``tree``)."""

    invariants: list[str] = Field(default_factory=list)
    """Named structural invariants preserved under isomorphism (e.g. "acyclic")."""

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    """
    Ordered node descriptors for WL-1 hashing.
    Each dict: {"type": str, ...}.  Domain-specific names must be stripped.
    """

    edges: list[dict[str, Any]] = Field(default_factory=list)
    """
    Ordered edge descriptors for WL-1 hashing.
    Each dict: {"type": str, "from": int, "to": int, ...}.
    Integer indices reference positions in ``nodes``.
    """

    def to_legacy_dict(self) -> dict[str, Any]:
        """Backwards-compatible dict export for existing convergence helpers."""
        return {
            "nodes": self.nodes if self.nodes else self.node_count,
            "edges": self.edges,
            "symmetry": self.symmetry or self.schema_type,
            "invariants": self.invariants,
        }


# ─── Compression Path (Provenance) ───────────────────────────────


class CompressionPathStep(EOSBaseModel):
    """A single step in the compression path that produced a fragment."""

    stage: str  # e.g. "holographic_encoding", "semantic_distillation"
    compression_ratio: float = 0.0
    bits_in: float = 0.0
    bits_out: float = 0.0
    timestamp: datetime = Field(default_factory=utc_now)


class CompressionPath(EOSBaseModel):
    """
    Full provenance record of how a fragment was compressed.

    Gap HIGH-3 (Federation Spec, 2026-03-07): formalised with explicit typed
    fields (source_system, target_system, compression_steps, fidelity_score,
    created_at) replacing the prior implicit dict-like structure.

    This is the transformation record, NOT the raw data. Shared so other
    instances can compare compression strategies and detect when different
    paths converge on the same structure.
    """

    source_system: str = ""
    """System ID that produced this fragment (e.g. ``logos``, ``oneiros``)."""

    target_system: str = ""
    """Intended consumer system. For world model fragments, always ``nexus``."""

    compression_steps: list[str] = Field(default_factory=list)
    """
    Ordered list of stage names applied during compression.
    E.g. ["holographic_encoding", "slow_wave_distillation", "schema_induction"].
    Fast comparison path - mirrors step[*].stage without traversing step records.
    """

    fidelity_score: float = 0.0
    """
    Fraction of original semantic content preserved after compression (0-1).
    Computed as 1.0 - (bits_lost / bits_in_first_step).
    """

    created_at: datetime = Field(default_factory=utc_now)
    """When this compression path was recorded."""

    # Detailed per-step records (optional, richer than compression_steps)
    steps: list[CompressionPathStep] = Field(default_factory=list)
    total_compression_ratio: float = 0.0
    source_domain: str = ""
    source_modality: str = ""

    @property
    def path_length(self) -> int:
        return len(self.steps) or len(self.compression_steps)


# ─── Triangulation State ─────────────────────────────────────────


class TriangulationSource(EOSBaseModel):
    """Record of an independent source that confirmed a structure."""

    instance_id: str
    divergence_score: float = 0.0  # How different this instance is from us
    fragment_id: str = ""
    confirmed_at: datetime = Field(default_factory=utc_now)


class TriangulationMetadata(EOSBaseModel):
    """
    Attached to world model structures to track epistemic triangulation.

    Confidence formula: log1p(count) / log1p(100) * (0.3 + 0.7 * diversity).
    - 1 source → 0.0 (no triangulation possible)
    - 2 diverse sources (div=1.0) → ~0.21
    - 5 diverse sources (div=1.0) → ~0.50
    - 10 diverse sources (div=1.0) → ~0.65
    - Low diversity collapses the multiplier toward 0.3 (near-useless)
    """

    independent_sources: list[TriangulationSource] = Field(default_factory=list)

    @property
    def independent_source_count(self) -> int:
        return len(self.independent_sources)

    @property
    def source_diversity_score(self) -> float:
        """Average divergence score across all independent sources."""
        if not self.independent_sources:
            return 0.0
        return sum(s.divergence_score for s in self.independent_sources) / len(
            self.independent_sources
        )

    @property
    def source_diversity(self) -> float:
        """Alias for source_diversity_score for API consistency.

        NexusService accesses ``fragment.triangulation.source_diversity`` in
        several places (get_divergence_weighted_fragments, _emit_re_training_example,
        etc.).  This alias avoids AttributeError on those call-sites without
        changing the canonical property name used by the promotion pipeline.
        """
        return self.source_diversity_score

    @property
    def triangulation_confidence(self) -> float:
        """
        Confidence in the structure based on independent confirmation.

        Formula: log1p(count) / log1p(100) * (0.3 + 0.7 * diversity)

        - Logarithmic in source count: 1→2 sources is more informative
          than 100→101 (diminishing returns on additional confirmation).
        - Linear in diversity: diverse paths eliminate shared bias.
        - Low diversity makes count nearly worthless (0.3 floor only).
        - 1 source → 0.0 (no triangulation possible with a single source).
        """
        if self.independent_source_count <= 1:
            return 0.0
        count_factor = math.log1p(self.independent_source_count) / math.log1p(100)
        diversity_factor = self.source_diversity_score
        return min(count_factor * (0.3 + 0.7 * diversity_factor), 1.0)


# ─── Shareable World Model Fragment ──────────────────────────────


class ShareableWorldModelFragment(Identified):
    """
    A fragment of the world model stripped of domain-specific labels,
    retaining only abstract relational structure.

    This is WHAT Nexus shares. The domain labels are kept separately
    so the receiving instance can map them to its own domain vocabulary
    without being biased by the sender's labeling.
    """

    # Identity
    fragment_id: str = Field(default_factory=new_id)
    source_instance_id: str = ""
    source_instance_divergence_score: float = 0.0

    # Abstract structure (domain labels stripped)
    abstract_structure: dict[str, Any] = Field(default_factory=dict)
    """
    The relational skeleton: node types, edge types, cardinalities,
    symmetries, invariants - everything except domain-specific names.
    Example: {"nodes": 3, "edges": [{"type": "causal", "from": 0, "to": 1},
              {"type": "causal", "from": 1, "to": 2}], "symmetry": "chain"}

    Typed variant: use ``typed_structure`` (AbstractStructure) when available.
    The dict form is kept for backward compatibility with Logos adapters.
    """

    typed_structure: AbstractStructure | None = None
    """
    Formally typed version of abstract_structure.  When present, ConvergenceDetector
    uses WL-1 hashing from this field.  When absent, falls back to the legacy
    dict form in abstract_structure.

    Gap HIGH-2: AbstractStructure added as typed alternative to dict[str, Any].
    """

    # Original domain labels (kept separate from structure)
    domain_labels: list[str] = Field(default_factory=list)

    # MDL quality metrics
    observations_explained: int = 0
    description_length: float = 0.0  # Bits
    compression_ratio: float = 0.0

    # Compression provenance
    compression_path: CompressionPath = Field(default_factory=CompressionPath)

    # Sleep certification
    sleep_certification: SleepCertification = Field(default_factory=SleepCertification)

    # Triangulation state
    triangulation: TriangulationMetadata = Field(default_factory=TriangulationMetadata)

    # Economic metadata (NEXUS-ECON-2/3): preserved so federation peers can
    # recover economic meaning from structurally-equivalent schemas.
    domain_hints: list[str] = Field(default_factory=list)
    """
    High-level domain tags detected from schema content.
    E.g. ["economic", "yield_farming", "defi"].
    Sent alongside the abstract structure so receivers can map the schema
    to the correct economic domain vocabulary without biasing convergence.
    """

    economic_context: dict[str, Any] | None = None
    """
    Optional economic metadata extracted from the schema source.
    E.g. {"protocol": "Aave", "apy_pct": 8.5, "token": "USDC"}.
    None when the schema has no economic content.
    Allows federation peers to triangulate economic ground truth.
    """

    # Timestamps
    created_at: datetime = Field(default_factory=utc_now)
    last_confirmed_at: datetime = Field(default_factory=utc_now)

    @property
    def is_shareable(self) -> bool:
        """A fragment is shareable only if it has survived sleep consolidation."""
        return self.sleep_certification.is_certified

    @property
    def quality_score(self) -> float:
        """
        Composite quality: compression_ratio * observations_explained weight.
        Higher = more valuable to share.
        """
        if self.description_length <= 0:
            return 0.0
        obs_factor = min(self.observations_explained / 10.0, 1.0)
        comp_factor = min(self.compression_ratio / 5.0, 1.0)
        return obs_factor * 0.4 + comp_factor * 0.6


# ─── Convergence Detection ───────────────────────────────────────


class ConvergenceResult(EOSBaseModel):
    """Result of comparing two fragments for structural isomorphism."""

    fragment_a_id: str
    fragment_b_id: str
    convergence_score: float = Field(0.0, ge=0.0, le=1.0)
    """0.0 = no structural match, 1.0 = identical structure, different domains."""

    matched_nodes: int = 0
    total_nodes_a: int = 0
    total_nodes_b: int = 0
    matched_edges: int = 0
    total_edges_a: int = 0
    total_edges_b: int = 0

    # Whether the domains are different (required for true triangulation)
    domains_are_independent: bool = False

    # Source instance diversity
    source_a_instance_id: str = ""
    source_b_instance_id: str = ""
    source_diversity: float = 0.0

    # WL-1 hashing path used (True = more accurate, False = legacy heuristic)
    wl1_used: bool = False

    detected_at: datetime = Field(default_factory=utc_now)

    @property
    def is_convergent(self) -> bool:
        """Convergence requires score >= 0.7 AND independent domains."""
        return self.convergence_score >= 0.7 and self.domains_are_independent

    @property
    def triangulation_value(self) -> float:
        """
        How much triangulation evidence this convergence provides.
        Maximised when high convergence_score AND high source_diversity.
        """
        return self.convergence_score * self.source_diversity


# ─── Divergence Measurement (Phase B) ────────────────────────────


class DivergenceDimensionScore(EOSBaseModel):
    """Score for a single divergence dimension."""

    dimension: str
    score: float = Field(0.0, ge=0.0, le=1.0)
    weight: float = 0.0
    weighted_score: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class DivergenceScore(EOSBaseModel):
    """
    Multi-dimensional divergence measurement between two instances.

    Five dimensions weighted to capture what makes triangulation valuable:
    - Domain diversity (0.25): overlap of domain coverage maps
    - Structural diversity (0.30): world model schema structural difference
    - Attentional diversity (0.20): Fovea weight profile difference
    - Hypothesis diversity (0.15): active hypothesis overlap
    - Temporal divergence (0.10): age/experience gap
    """

    instance_a_id: str = ""
    instance_b_id: str = ""

    # Per-dimension scores
    domain_diversity: DivergenceDimensionScore = Field(
        default_factory=lambda: DivergenceDimensionScore(
            dimension="domain_diversity", score=0.0, weight=0.25
        )
    )
    structural_diversity: DivergenceDimensionScore = Field(
        default_factory=lambda: DivergenceDimensionScore(
            dimension="structural_diversity", score=0.0, weight=0.30
        )
    )
    attentional_diversity: DivergenceDimensionScore = Field(
        default_factory=lambda: DivergenceDimensionScore(
            dimension="attentional_diversity", score=0.0, weight=0.20
        )
    )
    hypothesis_diversity: DivergenceDimensionScore = Field(
        default_factory=lambda: DivergenceDimensionScore(
            dimension="hypothesis_diversity", score=0.0, weight=0.15
        )
    )
    temporal_divergence: DivergenceDimensionScore = Field(
        default_factory=lambda: DivergenceDimensionScore(
            dimension="temporal_divergence", score=0.0, weight=0.10
        )
    )

    measured_at: datetime = Field(default_factory=utc_now)

    @property
    def dimensions(self) -> list[DivergenceDimensionScore]:
        return [
            self.domain_diversity,
            self.structural_diversity,
            self.attentional_diversity,
            self.hypothesis_diversity,
            self.temporal_divergence,
        ]

    @property
    def overall(self) -> float:
        """Weighted sum across all five dimensions."""
        return sum(d.score * d.weight for d in self.dimensions)

    @property
    def classification(self) -> DivergenceClassification:
        """Classify the divergence level."""
        score = self.overall
        if score >= 0.8:
            return DivergenceClassification.ALIEN_KIND
        if score >= 0.5:
            return DivergenceClassification.DISTINCT_KIND
        if score >= 0.2:
            return DivergenceClassification.RELATED_KIND
        return DivergenceClassification.SAME_KIND

    @property
    def is_same_kind(self) -> bool:
        return self.overall < 0.2

    @property
    def is_related_kind(self) -> bool:
        return 0.2 <= self.overall < 0.5

    @property
    def is_distinct_kind(self) -> bool:
        return 0.5 <= self.overall < 0.8

    @property
    def is_alien_kind(self) -> bool:
        return self.overall >= 0.8


# ─── Divergence Incentives (Phase B) ─────────────────────────────


class DivergencePressure(EOSBaseModel):
    """
    Signal sent to Thymos as a GROWTH drive when an instance is
    too similar to the federation average.

    Pushes toward frontier domains (unexplored by the federation)
    and away from saturated domains (well-covered already).
    """

    instance_id: str = ""
    triangulation_weight: float = 0.0
    """Current weight of this instance's contributions. Near-zero = near-duplicate."""

    pressure_magnitude: float = Field(0.0, ge=0.0, le=1.0)
    """How strongly to push toward divergence. 0 = no pressure, 1 = urgent."""

    frontier_domains: list[str] = Field(default_factory=list)
    """Domains that the federation lacks coverage in - explore these."""

    saturated_domains: list[str] = Field(default_factory=list)
    """Domains where the federation has excess coverage - avoid these."""

    recommended_direction: str = ""
    """Natural-language description of the recommended divergence direction."""

    generated_at: datetime = Field(default_factory=utc_now)

    @property
    def should_apply(self) -> bool:
        """Pressure only applies when triangulation_weight < 0.4."""
        return self.triangulation_weight < 0.4


# ─── IIEP Message Type ──────────────────────────────────────────


class WorldModelFragmentShare(EOSBaseModel):
    """
    IIEP message payload for sharing a world model fragment via
    WORLD_MODEL_FRAGMENT_SHARE Synapse event.

    This is the wire format - what actually gets sent over federation.
    The receiving instance's Nexus validates, compares structures,
    and updates triangulation metadata.

    Gap HIGH-4 (Federation Spec, 2026-03-07): explicit fields documented.
    """

    message_id: str = Field(default_factory=new_id)
    sender_instance_id: str = ""
    sender_divergence_score: float = 0.0
    """Overall divergence of sender relative to federation average (0-1)."""

    fragment: ShareableWorldModelFragment
    """The fragment being shared (includes typed_structure if built with AbstractStructure)."""

    # Sender's self-assessment
    sender_quality_claim: float = 0.0
    """Sender's assessment of this fragment's quality_score (0-1)."""

    sender_triangulation_confidence: float = 0.0
    """Sender's current triangulation_confidence for this fragment (0-1)."""

    # Sleep certification metadata (Gap MEDIUM-6)
    sleep_certified: bool = False
    """
    Mirror of fragment.sleep_certification.is_certified - lifted to top level
    so receivers can gate without deserialising the full fragment.
    Must be True for the receiver to accept the fragment.
    """

    consolidation_cycle_id: str = ""
    """
    ID of the Oneiros sleep cycle that certified this fragment.
    Populated from the ONEIROS_CONSOLIDATION_COMPLETE event payload.
    Allows receivers to cross-reference and verify certification provenance.
    """

    timestamp: datetime = Field(default_factory=utc_now)


class WorldModelFragmentShareResponse(EOSBaseModel):
    """Response to a WORLD_MODEL_FRAGMENT_SHARE message."""

    message_id: str = ""
    outcome: FragmentShareOutcome = FragmentShareOutcome.ACCEPTED
    convergence_detected: bool = False
    convergence_score: float = 0.0
    wl1_used: bool = False
    """True if WL-1 graph isomorphism was used for this comparison."""
    reason: str = ""
    timestamp: datetime = Field(default_factory=utc_now)


# ─── IIEP: Inter-Instance Epistemic Protocol ─────────────────────


class IIEPFragmentType(enum.StrEnum):
    """Type of epistemic payload within an IIEP session message."""

    WORLD_MODEL_FRAGMENT = "world_model_fragment"
    """Standard sleep-certified world model fragment (most common)."""

    CAUSAL_INVARIANT = "causal_invariant"
    """Kairos Tier 3 substrate-independent invariant."""

    CONVERGENCE_SUMMARY = "convergence_summary"
    """Summary of convergence results for triangulation book-keeping."""

    SESSION_OPEN = "session_open"
    """Protocol handshake - sent once per session to establish context."""

    SESSION_CLOSE = "session_close"
    """Graceful session termination."""


class IIEPMessage(EOSBaseModel):
    """
    Inter-Instance Epistemic Protocol (IIEP) wire message.

    Every epistemic exchange across the federation is wrapped in this
    envelope.  The message is carried as the ``payload`` of a
    WORLD_MODEL_FRAGMENT_SHARE Synapse event.

    Design rationale:
    - ``session_id`` groups all messages in one sharing session so
      receivers can correlate convergence results back to a session.
    - ``fragment_type`` lets receivers fast-path routing without
      deserialising the full payload.
    - ``sleep_certified`` is duplicated from the inner fragment so
      receivers can gate without full deserialization.
    - ``convergence_round`` supports multi-round triangulation: round 0
      is the initial share, round N is a re-share after N confirmations.

    Gap CRITICAL (Spec 19, 2026-03-07): formalises the IIEP schema that
    was previously implicit in WorldModelFragmentShare.  NexusService
    wraps every outbound fragment in this envelope and unwraps inbound.
    """

    # Session envelope
    session_id: str = Field(default_factory=new_id)
    """Unique ID for this epistemic sharing session."""

    initiator_id: str = ""
    """Instance ID of the session initiator."""

    responder_id: str = ""
    """Instance ID of the intended recipient (empty = broadcast)."""

    # Payload classification
    fragment_type: IIEPFragmentType = IIEPFragmentType.WORLD_MODEL_FRAGMENT

    # Core payload - the actual epistemic content being shared.
    # Typed as dict[str, Any] so any serialisable payload fits;
    # receivers use fragment_type to deserialise to the correct type.
    payload: dict[str, Any] = Field(default_factory=dict)

    # Sleep certification (fast-path gate - receivers check before full parse)
    sleep_certified: bool = False
    """True only when the carried fragment survived full Oneiros consolidation."""

    # Triangulation round counter
    convergence_round: int = 0
    """
    0 = initial share.
    N = re-share after N independent confirmations.
    Receivers use this to weight the triangulation evidence appropriately -
    re-shared fragments already have prior confirmation and require more
    diverse sources to advance the next triangulation level.
    """

    timestamp: datetime = Field(default_factory=utc_now)


# ─── Instance Profile (for divergence measurement) ───────────────


class InstanceDivergenceProfile(EOSBaseModel):
    """
    A snapshot of an instance's characteristics relevant to divergence
    measurement. Exchanged between instances during divergence assessment.
    """

    instance_id: str = ""

    # Domain coverage: which domains this instance has world model schemas for
    domain_coverage: list[str] = Field(default_factory=list)

    # Structural fingerprint: hash of the world model's schema topology
    structural_fingerprint: str = ""

    # Attentional profile: Fovea weight distribution
    attention_weights: dict[str, float] = Field(default_factory=dict)

    # Active hypotheses: IDs of hypotheses currently being tested
    active_hypothesis_ids: list[str] = Field(default_factory=list)

    # Temporal: when this instance was born, total experience count
    born_at: datetime = Field(default_factory=utc_now)
    total_experiences: int = 0
    total_schemas: int = 0

    # Economic divergence (NEXUS-ECON-4): how differently this instance
    # deploys economic strategy compared to federation peers.
    # 0.0 = converged (same strategy), 1.0 = fully diverged.
    # Measured as revenue-per-strategy variance normalised across peers.
    # Used by Oikos and Evo to detect when instances have independently
    # discovered different profitable strategies - valuable for triangulation.
    economic_divergence: float = Field(0.0, ge=0.0, le=1.0)

    # Snapshot of per-strategy revenue rates for variance computation.
    # Keys are strategy identifiers (e.g. "yield_farming", "bounty", "asset").
    # Values are recent USD revenue per unit time.
    strategy_revenue_rates: dict[str, float] = Field(default_factory=dict)

    captured_at: datetime = Field(default_factory=utc_now)


# ─── Nexus Configuration ────────────────────────────────────────


class NexusConfig(EOSBaseModel):
    """Configuration for the Nexus epistemic triangulation system."""

    # Fragment sharing
    min_compression_ratio_to_share: float = 2.0
    min_observations_to_share: int = 5
    min_sleep_cycles_to_share: int = 1

    # Convergence detection
    convergence_threshold: float = 0.7
    min_node_match_ratio: float = 0.6

    # WL-1 hashing (Gap HIGH-1)
    wl1_iterations: int = 3
    """Number of WL-1 refinement rounds. More = more discriminative, O(n·d·k) cost."""

    wl1_min_nodes_to_activate: int = 3
    """Minimum node count to use WL-1 hashing; smaller graphs use legacy comparison."""

    # Divergence measurement intervals
    divergence_measurement_interval_s: float = 300.0  # 5 minutes
    divergence_pressure_threshold: float = 0.4  # Below this, apply pressure

    # Triangulation
    max_triangulation_confidence: float = 1.0

    # Fragment store limits
    max_stored_fragments: int = 1000
    max_stored_convergences: int = 500

    # Speciation (Phase C)
    speciation_divergence_threshold: float = 0.8  # Overall divergence >= this → speciation

    # Ground truth promotion (Phase D)
    level_1_min_sources: int = 2
    level_2_min_sources: int = 3
    level_2_min_confidence: float = 0.75
    level_2_min_diversity: float = 0.5
    level_3_min_sources: int = 5
    level_3_min_confidence: float = 0.9
    level_3_min_diversity: float = 0.7
    level_4_adversarial_required: bool = True
    level_4_competition_required: bool = True

    # Oikos metabolic coupling (Gap HIGH-5)
    convergence_growth_bonus: float = 0.15
    """
    GROWTH metabolic allocation bonus emitted when triangulation convergence
    is achieved (convergence_score >= convergence_threshold AND
    domains_are_independent). Sent as NEXUS_CONVERGENCE_METABOLIC_SIGNAL.
    """

    # Oikos economic reward (HIGH-2: selection pressure toward knowledge sharing)
    convergence_economic_reward_usdc_per_tier: float = 0.001
    """
    Economic reward in USDC per convergence tier when triangulation convergence
    is confirmed.  Formula: convergence_tier × 0.001 USDC.

    convergence_tier is derived from the local fragment's current EpistemicLevel:
      HYPOTHESIS (0)         → tier 0 → $0.000 (no reward yet)
      CORROBORATED (1)       → tier 1 → $0.001
      TRIANGULATED (2)       → tier 2 → $0.002
      GROUND_TRUTH_CANDIDATE (3) → tier 3 → $0.003
      EMPIRICAL_INVARIANT (4)    → tier 4 → $0.004

    Instances that achieve convergence receive this reward;
    divergent instances receive nothing - selection pressure toward sharing.
    Emitted as ``economic_reward_usd`` in NEXUS_CONVERGENCE_METABOLIC_SIGNAL.
    """

    divergence_penalty_threshold: int = 5
    """
    Number of consecutive divergence cycles without any convergence before
    a metabolic penalty signal is emitted to Oikos.
    """

    divergence_metabolic_penalty: float = -0.10
    """
    GROWTH metabolic penalty magnitude (negative) emitted when an instance
    is persistently stuck in divergence (no convergence for
    divergence_penalty_threshold cycles).
    """


# ─── Phase C: Speciation Types ─────────────────────────────────


class EpistemicLevel(enum.IntEnum):
    """
    Epistemic status of a knowledge fragment.

    Each level requires strictly more evidence than the last.
    Level 4 is constitutional - protected by Equor governance.
    """

    HYPOTHESIS = 0            # Single instance. Could be experience path artifact.
    CORROBORATED = 1          # independent_source_count >= 2.
    TRIANGULATED = 2          # confidence > 0.75, diversity > 0.5, sources >= 3.
    GROUND_TRUTH_CANDIDATE = 3  # confidence > 0.9, diversity > 0.7, sources >= 5, survived bridge.
    EMPIRICAL_INVARIANT = 4   # Level 3 + survived Oneiros adversarial + Evo competition.


class SpeciationEvent(EOSBaseModel):
    """
    Record that two instances have diverged beyond the speciation threshold.

    After speciation, normal fragment sharing is impossible - only causal
    invariants (the most compressed structures) can cross the boundary
    via InvariantBridge.

    Gap MEDIUM-7 (Federation Spec, 2026-03-07): ``genome_distance`` and
    ``is_new_species`` fields added for Neo4j (:SpeciationEvent) storage.
    """

    id: str = Field(default_factory=new_id)
    instance_a_id: str
    instance_b_id: str
    timestamp: datetime = Field(default_factory=utc_now)
    divergence_score: float
    shared_invariant_count: int = 0
    incompatible_schema_count: int = 0
    new_cognitive_kind_registered: bool = False

    # Gap MEDIUM-7: speciation registry storage fields
    genome_distance: float = 0.0
    """
    Genetic distance between the two instance genomes at the time of speciation.
    Computed from Mitosis genome divergence (Hamming or cosine distance on
    genome parameter vectors). 0.0 if genomes are unavailable.
    """

    is_new_species: bool = False
    """
    True if this speciation event created a new cognitive kind entry in the
    SpeciationRegistry - i.e., at least one of the instances was not
    previously classified in any existing cognitive kind.
    """


class ConvergedInvariant(EOSBaseModel):
    """
    Two invariants from speciated (alien-kind) instances that match
    at the purest structural level.

    This is the strongest possible evidence for ground truth - two
    instances with incompatible structural languages independently
    arrived at the same abstract form.
    """

    id: str = Field(default_factory=new_id)
    invariant_a_id: str
    invariant_b_id: str
    source_instance_a: str = ""
    source_instance_b: str = ""
    abstract_form: dict[str, Any] = Field(default_factory=dict)
    """The ultra-abstract structure shared by both invariants."""

    triangulation_confidence: float = 0.95
    is_ground_truth_candidate: bool = True
    converged_at: datetime = Field(default_factory=utc_now)


class InvariantExchangeReport(EOSBaseModel):
    """Result of exchanging invariants across a speciation boundary."""

    bridge_id: str = ""
    instance_a_id: str = ""
    instance_b_id: str = ""
    invariants_compared: int = 0
    converged_invariants: list[ConvergedInvariant] = Field(default_factory=list)
    abstract_equivalences_found: int = 0
    exchange_timestamp: datetime = Field(default_factory=utc_now)


class CognitiveKindEntry(EOSBaseModel):
    """
    Registry entry for a cognitive kind - a class of instances that
    share structural language compatibility.

    Instances within the same kind can share fragments normally.
    Across kinds, only InvariantBridge exchange is possible.
    """

    kind_id: str = Field(default_factory=new_id)
    member_instance_ids: list[str] = Field(default_factory=list)
    founding_speciation_event_id: str = ""
    established_at: datetime = Field(default_factory=utc_now)


class SpeciationRegistryState(EOSBaseModel):
    """
    Federation-level registry of all speciation events, cognitive kinds,
    and active invariant bridges.
    """

    speciation_events: list[SpeciationEvent] = Field(default_factory=list)
    cognitive_kinds: list[CognitiveKindEntry] = Field(default_factory=list)
    active_bridge_pairs: list[tuple[str, str]] = Field(default_factory=list)
    """Pairs of (instance_a_id, instance_b_id) with active InvariantBridge connections."""


# ─── Phase D: Ground Truth Promotion Types ────────────────────


class PromotionDecision(EOSBaseModel):
    """
    Result of evaluating a fragment for epistemic level promotion.

    Contains the current level, whether promotion was granted, and
    the evidence that supports the decision.
    """

    fragment_id: str
    current_level: EpistemicLevel
    proposed_level: EpistemicLevel | None = None
    promoted: bool = False
    reason: str = ""

    # Evidence metrics
    independent_source_count: int = 0
    triangulation_confidence: float = 0.0
    source_diversity: float = 0.0
    survived_speciation_bridge: bool = False
    survived_adversarial_test: bool = False
    survived_hypothesis_competition: bool = False

    evaluated_at: datetime = Field(default_factory=utc_now)
