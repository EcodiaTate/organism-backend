"""
EcodiaOS — Thread Internal Types

The data structures that constitute the organism's narrative identity.
These model commitments, schemas, fingerprints, and life chapters —
the building blocks of autobiographical selfhood.

Fingerprint dimensions (29D):
  Personality        9D  — from Voxis PersonalityVector
  Drive alignment    4D  — from Equor DriftTracker mean alignment
  Affect             6D  — from Atune current_affect
  Goal profile       5D  — estimated from episode/goal counts
  Interaction        5D  — estimated from conversation/expression metrics
"""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import Field

from primitives.common import (
    DriveAlignmentVector,
    EOSBaseModel,
    Identified,
    Timestamped,
    utc_now,
)

# ─── Enums ────────────────────────────────────────────────────────────────────


class CommitmentType(enum.StrEnum):
    """What kind of commitment is this?"""

    CONSTITUTIONAL_GROUNDING = "constitutional_grounding"  # Birth promises (4 drives)
    RELATIONAL = "relational"          # Commitments to specific people/communities
    VOCATIONAL = "vocational"          # Commitments about purpose and role
    EPISTEMIC = "epistemic"            # Commitments to ways of knowing
    AESTHETIC = "aesthetic"             # Commitments to style and expression


class CommitmentStrength(enum.StrEnum):
    """How deeply held is this commitment?"""

    NASCENT = "nascent"          # Just formed, untested
    DEVELOPING = "developing"    # Tested in a few situations
    ESTABLISHED = "established"  # Consistently held across contexts
    CORE = "core"                # Foundational to identity — change triggers crisis


class SchemaStatus(enum.StrEnum):
    """Lifecycle of an identity schema."""

    EMERGING = "emerging"        # Pattern detected, not yet crystallised
    FORMING = "forming"          # Accumulating evidence, taking shape
    ESTABLISHED = "established"  # Stable, integrated into self-narrative
    DOMINANT = "dominant"        # Primary lens for self-interpretation
    ARCHIVED = "archived"        # No longer active but remembered


class ChapterStatus(enum.StrEnum):
    """Status of a narrative chapter."""

    ACTIVE = "active"      # Currently living this chapter
    CLOSED = "closed"      # Chapter has ended, theme resolved
    EMERGING = "emerging"  # A new chapter is forming
    FORMING = "forming"    # New chapter is being initialised


class CommitmentStatus(enum.StrEnum):
    """Lifecycle status of a commitment."""

    ACTIVE = "active"
    TESTED = "tested"
    STRAINED = "strained"
    BROKEN = "broken"
    FULFILLED = "fulfilled"


class SchemaStrength(enum.StrEnum):
    """How well-established is an identity schema?"""

    NASCENT = "nascent"
    DEVELOPING = "developing"
    ESTABLISHED = "established"
    CORE = "core"


class SchemaValence(enum.StrEnum):
    """Is this schema adaptive or maladaptive?"""

    ADAPTIVE = "adaptive"
    MALADAPTIVE = "maladaptive"
    NEUTRAL = "neutral"


class NarrativeCoherence(enum.StrEnum):
    """Overall coherence of the organism's self-narrative."""

    INTEGRATED = "integrated"
    TRANSITIONAL = "transitional"
    FRAGMENTED = "fragmented"
    CONFLICTED = "conflicted"


class CommitmentSource(enum.StrEnum):
    """How a commitment was formed."""

    EXPLICIT_DECLARATION = "explicit_declaration"   # Organism stated "I will always..."
    SCHEMA_CRYSTALLIZATION = "schema_crystallization"  # ADAPTIVE schema reached CORE
    CRISIS_RESOLUTION = "crisis_resolution"          # After a CRISIS turning point
    CONSTITUTIONAL_GROUNDING = "constitutional_grounding"  # Seeded at birth from drives


class TurningPointType(enum.StrEnum):
    """The type of narrative turning point."""

    REVELATION = "revelation"
    CRISIS = "crisis"
    GROWTH = "growth"
    LOSS = "loss"
    CONNECTION = "connection"
    ACHIEVEMENT = "achievement"
    RUPTURE = "rupture"  # Commitment violated / relationship broken


class DriftClassification(enum.StrEnum):
    """Classification of identity change over time."""

    STABLE = "stable"           # Negligible change
    GROWTH = "growth"           # Schema-aligned change, no commitment violation
    TRANSITION = "transition"   # Explained by turning point context
    DRIFT = "drift"             # Unexplained or commitment-violating change


class NarrativeArcType(enum.StrEnum):
    """McAdams narrative arc types for chapter classification."""

    REDEMPTION = "redemption"         # Bad to good
    CONTAMINATION = "contamination"   # Good to bad
    GROWTH = "growth"                 # Upward trajectory
    STABILITY = "stability"           # Stable with low variance
    TRANSFORMATION = "transformation" # Significant change, ambiguous valence


# ─── Commitment ───────────────────────────────────────────────────────────────


class Commitment(Identified, Timestamped):
    """
    A lived promise that shapes identity. Ricoeur's 'keeping one's word'
    as computational structure.

    Constitutional commitments are seeded at birth from the four drives.
    Others emerge from experience and are strengthened through action.
    """

    type: CommitmentType = CommitmentType.CONSTITUTIONAL_GROUNDING
    statement: str = ""             # Natural language commitment
    strength: CommitmentStrength = CommitmentStrength.NASCENT
    drive_source: str = ""          # Which drive this commitment serves
    evidence_episodes: list[str] = Field(default_factory=list)
    violation_episodes: list[str] = Field(default_factory=list)
    last_tested_at: datetime | None = None
    test_count: int = 0
    upheld_count: int = 0
    # Extended fields used by commitment_keeper / narrative_retriever
    status: CommitmentStatus = CommitmentStatus.ACTIVE
    source: CommitmentSource = CommitmentSource.CONSTITUTIONAL_GROUNDING
    source_description: str = ""
    source_episode_ids: list[str] = Field(default_factory=list)
    drive_alignment: DriveAlignmentVector = Field(default_factory=DriveAlignmentVector)
    made_at: datetime = Field(default_factory=utc_now)
    last_tested: datetime | None = None
    last_held: datetime | None = None
    tests_faced: int = 0
    tests_held: int = 0
    fidelity: float = 1.0
    embedding: list[float] | None = None

    def update_fidelity(self, held: bool) -> None:
        """Update fidelity tracking after a test."""
        self.tests_faced += 1
        if held:
            self.tests_held += 1
        if self.tests_faced > 0:
            self.fidelity = self.tests_held / self.tests_faced


# ─── Identity Schema ─────────────────────────────────────────────────────────


class IdentitySchema(Identified, Timestamped):
    """
    A crystallised pattern of self-understanding.

    Schemas emerge from Evo's pattern detection, are refined through
    experience, and become the lenses through which the organism
    interprets itself. "I am someone who..."
    """

    statement: str = ""              # "I tend to..." / "I am someone who..."
    status: SchemaStatus = SchemaStatus.EMERGING
    source_pattern_ids: list[str] = Field(default_factory=list)
    supporting_episodes: list[str] = Field(default_factory=list)
    contradicting_episodes: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    confidence: float = 0.5         # How well-supported is this schema
    salience: float = 0.5           # How relevant to current identity
    last_activated_at: datetime = Field(default_factory=utc_now)
    # Extended fields used by narrative_retriever / identity_schema_engine
    trigger_contexts: list[str] = Field(default_factory=list)
    behavioral_tendency: str = ""
    emotional_signature: dict[str, float] = Field(default_factory=dict)
    drive_alignment: DriveAlignmentVector = Field(default_factory=DriveAlignmentVector)
    strength: SchemaStrength = SchemaStrength.NASCENT
    valence: SchemaValence = SchemaValence.ADAPTIVE
    confirmation_count: int = 0
    confirmation_episodes: list[str] = Field(default_factory=list)
    disconfirmation_count: int = 0
    disconfirmation_episodes: list[str] = Field(default_factory=list)
    evidence_ratio: float = 0.5
    first_formed: datetime = Field(default_factory=utc_now)
    last_activated: datetime = Field(default_factory=utc_now)
    last_updated: datetime = Field(default_factory=utc_now)
    parent_schema_id: str | None = None
    evolution_reason: str = ""

    @property
    def computed_evidence_ratio(self) -> float:
        total = len(self.supporting_episodes) + len(self.contradicting_episodes)
        if total == 0:
            return 0.5
        return len(self.supporting_episodes) / total

    def recompute_evidence_ratio(self) -> None:
        """Recompute evidence_ratio from confirmation/disconfirmation counts."""
        total = self.confirmation_count + self.disconfirmation_count
        self.evidence_ratio = self.confirmation_count / total if total > 0 else 0.5


# ─── Identity Fingerprint ────────────────────────────────────────────────────


FINGERPRINT_DIMS = 29

# Named dimension ranges for interpretability
PERSONALITY_SLICE = slice(0, 9)    # warmth, directness, verbosity, formality,
                                    # curiosity_expression, humour, empathy_expression,
                                    # confidence_display, metaphor_use
DRIVE_SLICE = slice(9, 13)         # coherence, care, growth, honesty
AFFECT_SLICE = slice(13, 19)       # valence, arousal, dominance, curiosity,
                                    # care_activation, coherence_stress
GOAL_SLICE = slice(19, 24)         # active_goals_norm, epistemic_ratio,
                                    # care_ratio, achievement_rate, goal_turnover
INTERACTION_SLICE = slice(24, 29)  # speak_rate, silence_rate, expression_diversity,
                                    # conversation_depth, community_engagement


class IdentityFingerprint(Identified, Timestamped):
    """
    A 29-dimensional snapshot of who the organism is right now.

    Comparing fingerprints over time reveals identity drift — not
    constitutional drift (Equor handles that) but the slower shift
    in who the organism is becoming.
    """

    vector: list[float] = Field(default_factory=lambda: [0.0] * FINGERPRINT_DIMS)
    cycle_number: int = 0

    @property
    def personality(self) -> list[float]:
        return self.vector[PERSONALITY_SLICE]

    @property
    def drive_alignment(self) -> list[float]:
        return self.vector[DRIVE_SLICE]

    @property
    def affect(self) -> list[float]:
        return self.vector[AFFECT_SLICE]

    @property
    def goal_profile(self) -> list[float]:
        return self.vector[GOAL_SLICE]

    @property
    def interaction_profile(self) -> list[float]:
        return self.vector[INTERACTION_SLICE]

    def distance_to(self, other: IdentityFingerprint) -> float:
        """
        Wasserstein-inspired distance between two fingerprints.
        Uses L1 (Manhattan) distance normalised by dimensionality.
        """
        if len(self.vector) != len(other.vector):
            return float("inf")
        return sum(
            abs(a - b) for a, b in zip(self.vector, other.vector, strict=False)
        ) / len(self.vector)


# ─── Behavioral Fingerprint ──────────────────────────────────────────────────


class BehavioralFingerprint(Identified, Timestamped):
    """
    A distributional snapshot of behaviour over an epoch window.

    Used by DiachronicCoherenceMonitor to compute Wasserstein distance
    and classify change as stable/growth/transition/drift.

    The ``feature_vector`` property flattens all centroids into the
    29-dimensional representation used for distance computation.
    """

    epoch_label: str = ""
    window_start: datetime = Field(default_factory=utc_now)
    window_end: datetime = Field(default_factory=utc_now)
    personality_centroid: list[float] = Field(default_factory=lambda: [0.0] * 9)
    drive_alignment_centroid: list[float] = Field(default_factory=lambda: [0.0] * 4)
    goal_source_distribution: list[float] = Field(default_factory=lambda: [0.0] * 5)
    affect_centroid: list[float] = Field(default_factory=lambda: [0.0] * 6)
    interaction_style_distribution: list[float] = Field(default_factory=lambda: [0.0] * 5)
    episodes_in_window: int = 0
    mean_surprise: float = 0.0
    mean_coherence: float = 0.5

    @property
    def feature_vector(self) -> list[float]:
        """Concatenate all component centroids into the 29D feature vector."""
        return (
            list(self.personality_centroid[:9])
            + list(self.drive_alignment_centroid[:4])
            + list(self.affect_centroid[:6])
            + list(self.goal_source_distribution[:5])
            + list(self.interaction_style_distribution[:5])
        )


# ─── Narrative Chapter ───────────────────────────────────────────────────────


class NarrativeChapter(Identified, Timestamped):
    """
    A recognised phase in the organism's life story.

    Chapters emerge from significant identity shifts — a new community,
    a constitutional crisis, a period of rapid growth.
    """

    title: str = ""
    theme: str = ""                   # One-sentence theme
    status: ChapterStatus = ChapterStatus.ACTIVE
    opened_at_cycle: int = 0
    closed_at_cycle: int | None = None
    key_schemas: list[str] = Field(default_factory=list)     # Schema IDs
    key_commitments: list[str] = Field(default_factory=list)  # Commitment IDs
    key_episodes: list[str] = Field(default_factory=list)     # Episode IDs
    fingerprint_at_start: str = ""    # Fingerprint ID at chapter open
    fingerprint_at_close: str = ""    # Fingerprint ID at chapter close
    summary: str = ""                 # LLM-generated chapter summary
    started_at: datetime | None = None
    ended_at: datetime | None = None
    episode_count: int = 0
    personality_snapshot_start: dict[str, float] = Field(default_factory=dict)
    active_schema_ids: list[str] = Field(default_factory=list)


# ─── Life Story ──────────────────────────────────────────────────────────────


class LifeStorySnapshot(EOSBaseModel):
    """
    A periodic synthesis of the organism's autobiography.
    The organism's own understanding of its narrative arc.
    """

    synthesis: str                    # Natural language life story
    chapter_count: int = 0
    active_chapter: str = ""          # Current chapter title
    core_schemas: list[str] = Field(default_factory=list)     # Top schema statements
    core_commitments: list[str] = Field(default_factory=list)  # Top commitment statements
    identity_coherence: float = 0.5   # How integrated is the narrative (0–1)
    generated_at: datetime = Field(default_factory=utc_now)
    cycle_number: int = 0


# ─── Narrative Scene ─────────────────────────────────────────────────────────


class NarrativeScene(Identified, Timestamped):
    """
    A group of related episodes composed into a narrative scene.

    Produced by NarrativeSynthesizer.compose_scene().
    Multiple scenes make up a NarrativeChapter.
    """

    summary: str = ""
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    episode_count: int = 0
    chapter_id: str = ""
    dominant_emotion: str = ""
    arc_type: NarrativeArcType = NarrativeArcType.STABILITY


# ─── Self-Evidencing ─────────────────────────────────────────────────────────


class IdentityPrediction(EOSBaseModel):
    """
    A behavioural prediction generated from a schema or commitment.

    Used by SelfEvidencingLoop to compare predicted vs actual behaviour
    and compute identity surprise (Fristonian prediction error).
    """

    schema_id: str = ""
    commitment_id: str = ""
    predicted_behavior: str = ""
    predicted_affect: dict[str, float] = Field(default_factory=dict)
    precision: float = 0.5    # 0–1, derived from schema strength
    context_condition: str = ""


class SelfEvidencingResult(EOSBaseModel):
    """
    Result of a single self-evidencing cycle for one episode.

    Contains the identity surprise metric and lists of schemas/commitments
    that were confirmed or challenged by the episode.
    """

    episode_id: str = ""
    predictions_evaluated: int = 0
    mean_prediction_error: float = 0.0
    identity_surprise: float = 0.0
    schemas_confirmed: list[str] = Field(default_factory=list)
    schemas_challenged: list[str] = Field(default_factory=list)
    commitments_tested: list[str] = Field(default_factory=list)


# ─── Thread Health ───────────────────────────────────────────────────────────


class ThreadHealthSnapshot(EOSBaseModel):
    """Observability snapshot for Thread system health."""

    status: str = "healthy"
    total_commitments: int = 0
    total_schemas: int = 0
    total_fingerprints: int = 0
    total_chapters: int = 0
    active_chapter: str = ""
    identity_coherence: float = 0.0
    fingerprint_drift: float = 0.0
    on_cycle_count: int = 0
    life_story_integrations: int = 0


# ─── Schema Conflict ─────────────────────────────────────────────────────────


class SchemaConflict(Identified, Timestamped):
    """
    Detected when two ESTABLISHED+ schemas have contradictory statements.
    Embedding cosine similarity < -0.3 triggers this.
    """

    schema_a_id: str
    schema_b_id: str
    schema_a_statement: str = ""
    schema_b_statement: str = ""
    cosine_similarity: float = 0.0
    resolved: bool = False
    resolution_note: str = ""


class TurningPoint(Identified, Timestamped):
    """A significant moment that shifted the organism's narrative trajectory."""

    chapter_id: str = ""
    type: TurningPointType = TurningPointType.REVELATION
    description: str = ""
    surprise_magnitude: float = 0.0
    narrative_weight: float = 0.0


class NarrativeSurpriseAccumulator(EOSBaseModel):
    """Running statistics for chapter boundary detection."""

    chapter_id: str = ""
    surprise_ema: float = 0.0
    surprise_ema_baseline: float = 0.0
    cumulative_surprise: float = 0.0
    episodes_in_chapter: int = 0
    affect_ema_valence: float = 0.0
    affect_ema_arousal: float = 0.1
    goal_completions_in_window: int = 0
    goal_failures_in_window: int = 0
    schema_challenges_in_window: int = 0

    def reset(self, chapter_id: str) -> None:
        self.chapter_id = chapter_id
        self.surprise_ema_baseline = self.surprise_ema
        self.cumulative_surprise = 0.0
        self.episodes_in_chapter = 0
        self.goal_completions_in_window = 0
        self.goal_failures_in_window = 0
        self.schema_challenges_in_window = 0


class ThreadConfig(EOSBaseModel):
    """Configuration for the Thread identity-coherence system."""

    # Surprise accumulator parameters
    surprise_ema_alpha: float = 0.15
    surprise_spike_multiplier: float = 3.0
    surprise_sustained_multiplier: float = 2.0
    surprise_weight_affect: float = 0.25
    surprise_weight_goal: float = 0.30
    surprise_weight_context: float = 0.15
    surprise_weight_entity: float = 0.15
    surprise_weight_schema: float = 0.15

    # Chapter lifecycle
    chapter_min_episodes: int = 10
    chapter_max_episodes: int = 200
    affect_shift_threshold: float = 0.3

    # Commitment tracking
    commitment_min_tests_for_fidelity: int = 3
    commitment_strain_threshold: float = 0.5
    commitment_broken_threshold: float = 0.4
    commitment_broken_min_tests: int = 5

    # Schema tracking
    schema_relevance_threshold: float = 0.4
    schema_evidence_ambiguity_threshold: float = 0.6
    schema_inactive_days_before_decay: int = 30

    # Self-evidencing / narrative synthesis
    self_evidencing_interval_cycles: int = 50
    self_evidencing_relevance_threshold: float = 0.5
    identity_surprise_mild: float = 0.3
    identity_surprise_significant: float = 0.6
    identity_surprise_crisis: float = 0.85
    life_story_max_words: int = 500
    scene_narrative_max_words: int = 100

    # Fingerprint weights (must sum to 1.0 for normalised distance)
    fingerprint_weight_personality: float = 0.35
    fingerprint_weight_drive: float = 0.25
    fingerprint_weight_affect: float = 0.20
    fingerprint_weight_goal: float = 0.10
    fingerprint_weight_interaction: float = 0.10

    # Wasserstein distance thresholds for drift classification
    wasserstein_stable_threshold: float = 0.05
    wasserstein_major_threshold: float = 0.25

    # Schema formation constraints
    schema_formation_cooldown_hours: float = 48.0
    schema_formation_min_episodes: int = 5
    schema_formation_min_span_hours: float = 24.0
    schema_similarity_merge_threshold: float = 0.85

    # Schema promotion thresholds
    schema_promotion_min_confirmations: int = 15
    schema_core_min_confirmations: int = 50
    schema_core_min_age_days: float = 180.0

    # LLM parameters
    llm_temperature_evaluation: float = 0.2
    llm_temperature_narrative: float = 0.7


class NarrativeIdentitySummary(EOSBaseModel):
    """Complete identity summary for 'who am I?' queries."""

    core_schemas: list[IdentitySchema] = Field(default_factory=list)
    established_schemas: list[IdentitySchema] = Field(default_factory=list)
    active_commitments: list[Commitment] = Field(default_factory=list)
    current_chapter_title: str = ""
    current_chapter_theme: str = ""
    life_story_summary: str = ""
    key_personality_traits: dict[str, float] = Field(default_factory=dict)
    recent_turning_points: list[TurningPoint] = Field(default_factory=list)
    narrative_coherence: NarrativeCoherence = NarrativeCoherence.TRANSITIONAL
    idem_score: float = 0.5
    ipse_score: float = 1.0
