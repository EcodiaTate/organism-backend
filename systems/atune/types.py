"""
Atune-specific data types.

These types are internal to Atune and not part of the shared primitives.
They wrap primitives with Atune-specific context for the perception and
workspace pipeline.
"""

from __future__ import annotations

import enum
from datetime import datetime  # noqa: TC003 — Pydantic needs at runtime
from typing import Any

from pydantic import BaseModel, Field

from primitives.affect import AffectState  # noqa: TC001 — Pydantic needs at runtime
from primitives.common import EOSBaseModel, new_id, utc_now
from primitives.memory_trace import MemoryTrace  # noqa: TC001 — Pydantic needs at runtime

# ---------------------------------------------------------------------------
# Input channels
# ---------------------------------------------------------------------------


class InputChannel(enum.StrEnum):
    """All channels from which Atune can receive raw input."""

    # User-facing
    TEXT_CHAT = "text_chat"
    VOICE = "voice"
    GESTURE = "gesture"

    # Environmental
    SENSOR_IOT = "sensor_iot"
    CALENDAR = "calendar"
    EXTERNAL_API = "external_api"

    # Internal
    SYSTEM_EVENT = "system_event"
    MEMORY_BUBBLE = "memory_bubble"
    AFFECT_SHIFT = "affect_shift"
    EVO_INSIGHT = "evo_insight"

    # Federation
    FEDERATION_MSG = "federation_msg"


# ---------------------------------------------------------------------------
# Raw input (pre-normalisation)
# ---------------------------------------------------------------------------


class RawInput(BaseModel):
    """Raw data before normalisation into a Percept."""

    data: str | bytes
    channel_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prediction error
# ---------------------------------------------------------------------------


class PredictionErrorDirection(enum.StrEnum):
    """Category of surprise."""

    CONTRADICTS_BELIEF = "contradicts_belief"
    NOVEL = "novel"
    CONFIRMS_UNEXPECTED = "confirms_unexpected"
    EXPECTED = "expected"


class PredictionError(BaseModel):
    """How surprising a Percept is given current beliefs."""

    magnitude: float = Field(ge=0.0, le=1.0)
    direction: PredictionErrorDirection
    domain: str = ""
    expected_embedding: list[float] | None = None
    actual_embedding: list[float] | None = None


# ---------------------------------------------------------------------------
# Salience
# ---------------------------------------------------------------------------


class GradientAttentionVector(BaseModel):
    """
    Token-level salience attribution via embedding gradient analysis.

    For a given salience head that uses embedding similarity, this captures
    *which tokens* in the percept are driving the similarity score —
    essentially a Jacobian-based saliency map over the input tokens.
    """

    per_token_importance: list[float] = Field(default_factory=list)
    """Normalised importance scores (sum to 1.0), one per token."""

    load_bearing_tokens: list[int] = Field(default_factory=list)
    """Indices of the top-K tokens driving the salience decision."""

    gradient_magnitude: float = 0.0
    """Overall gradient norm — higher means a sharper, more decisive similarity signal."""

    gradient_direction_conflicts: list[int] = Field(default_factory=list)
    """Token indices whose gradient direction opposes the reference embedding (contradiction)."""

    head_name: str = ""
    """Which salience head produced this attribution."""


class ThreatTrajectory(enum.StrEnum):
    """Qualitative label for how a salience head's score is changing."""

    STEADY = "steady"
    RISING = "rising"
    FALLING = "falling"
    ACCELERATING = "accelerating"


class HeadMomentum(BaseModel):
    """Momentum summary for a single salience head."""

    first_derivative: float = 0.0
    """dv/dt — rate of change of the head's score (units per second)."""

    second_derivative: float = 0.0
    """d²v/dt² — acceleration of the head's score (units per second²)."""

    trajectory: ThreatTrajectory = ThreatTrajectory.STEADY
    """Qualitative trajectory label derived from derivatives."""

    time_to_threshold: float | None = None
    """Seconds until linear extrapolation predicts crossing the salience threshold.
    ``None`` when the head is not rising or is already above threshold."""

    momentum_bonus: float = 0.0
    """Additional salience weight awarded when acceleration exceeds threshold."""


class SalienceVector(BaseModel):
    """Per-head salience scores plus composite."""

    scores: dict[str, float] = Field(default_factory=dict)
    composite: float = Field(ge=0.0, le=1.0, default=0.0)
    prediction_error: PredictionError | None = None
    gradient_attention: dict[str, GradientAttentionVector] = Field(default_factory=dict)
    """Per-head gradient attention vectors (only for embedding-similarity heads)."""

    momentum: dict[str, HeadMomentum] = Field(default_factory=dict)
    """Per-head momentum (first/second derivatives) when momentum tracking is active."""

    threat_trajectory: ThreatTrajectory = ThreatTrajectory.STEADY
    """Overall threat trajectory — worst-case across all heads."""


# ---------------------------------------------------------------------------
# Workspace types
# ---------------------------------------------------------------------------


class WorkspaceCandidate(BaseModel):
    """A candidate competing for workspace broadcast."""

    content: Any  # Percept or other content
    salience: SalienceVector
    source: str = ""
    prediction_error: PredictionError | None = None


class MemoryContext(BaseModel):
    """Memory retrieval results attached to a broadcast."""

    traces: list[MemoryTrace] = Field(default_factory=list)
    entities: list[Any] = Field(default_factory=list)
    communities: list[Any] = Field(default_factory=list)


class WorkspaceContext(BaseModel):
    """Contextual information accompanying a workspace broadcast."""

    recent_broadcast_ids: list[str] = Field(default_factory=list)
    active_goal_ids: list[str] = Field(default_factory=list)
    memory_context: MemoryContext = Field(default_factory=MemoryContext)
    prediction_error: PredictionError | None = None


class WorkspaceBroadcast(BaseModel):
    """The output of a workspace cycle — broadcast to all systems."""

    broadcast_id: str = Field(default_factory=lambda: new_id())
    timestamp: datetime = Field(default_factory=utc_now)
    content: Any  # Percept or contributed content
    salience: SalienceVector
    affect: AffectState
    context: WorkspaceContext = Field(default_factory=WorkspaceContext)
    precision: float = Field(ge=0.0, le=1.0, default=0.5)
    source: str = ""  # e.g. "internal:axon", "spontaneous_recall", percept channel_id


# Ensure forward refs are resolved (AffectState must be importable at runtime).
WorkspaceBroadcast.model_rebuild()


class WorkspaceContribution(BaseModel):
    """Content submitted by another system for workspace consideration."""

    system: str
    content: Any
    priority: float = Field(ge=0.0, le=1.0, default=0.5)
    reason: str = ""


# ---------------------------------------------------------------------------
# Attention context (passed to Fovea prediction error computation)
# ---------------------------------------------------------------------------


class ActiveGoalSummary(BaseModel):
    """Minimal goal info needed by Fovea for goal-relevance weighting."""

    id: str
    target_embedding: list[float]
    priority: float = Field(ge=0.0, le=1.0, default=0.5)


class RiskCategory(BaseModel):
    """A known risk category with its embedding."""

    name: str
    embedding: list[float]


class LearnedPattern(BaseModel):
    """A pattern Evo has identified as important."""

    pattern: str
    weight: float = 1.0


class Alert(BaseModel):
    """An active alert pattern set by governance or Equor."""

    pattern: str
    severity: float = Field(ge=0.0, le=1.0, default=0.5)


class PendingDecision(BaseModel):
    """A decision awaiting information."""

    id: str
    description: str
    embedding: list[float] | None = None


class AttentionContext(BaseModel):
    """
    Context assembled once per Percept and passed to Fovea for
    prediction error decomposition and salience computation.
    """

    prediction_error: PredictionError
    affect_state: AffectState
    active_goals: list[ActiveGoalSummary] = Field(default_factory=list)
    core_identity_embeddings: list[list[float]] = Field(default_factory=list)
    community_embedding: list[float] = Field(default_factory=list)
    source_habituation: dict[str, float] = Field(default_factory=dict)
    risk_categories: list[RiskCategory] = Field(default_factory=list)
    learned_patterns: list[LearnedPattern] = Field(default_factory=list)
    community_vocabulary: set[str] = Field(default_factory=set)
    active_alerts: list[Alert] = Field(default_factory=list)
    pending_decisions: list[PendingDecision] = Field(default_factory=list)
    community_size: int = 0
    instance_name: str = ""

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


class EntityCandidate(BaseModel):
    """An entity extracted from a Percept by LLM."""

    name: str
    type: str
    description: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class RelationCandidate(BaseModel):
    """A relation between entities extracted by LLM."""

    from_entity: str
    to_entity: str
    type: str
    strength: float = Field(ge=0.0, le=1.0, default=0.5)
    temporal: str | None = None


class ExtractionResult(BaseModel):
    """Output of entity/relation extraction from a Percept."""

    entities: list[EntityCandidate] = Field(default_factory=list)
    relations: list[RelationCandidate] = Field(default_factory=list)
    source_percept_id: str = ""


# ---------------------------------------------------------------------------
# EIS gate compatibility — Atune-local annotations
# ---------------------------------------------------------------------------


class ThreatAnnotationLocal(EOSBaseModel):
    """
    Atune-side threat annotation, structurally compatible with EIS
    ``ThreatAnnotation``.

    Populated by the normalisation pipeline (initially empty) so that the
    EIS fast-path can append its own annotations without schema mismatch.
    Kept as a separate model from ``eis.models.ThreatAnnotation`` to avoid
    a hard import dependency from Atune → EIS at the type level.
    """

    source: str = ""
    threat_class: str = "benign"
    severity: str = "none"
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    evidence: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SentimentAnalysis(EOSBaseModel):
    """LLM-derived sentiment analysis result attached during normalisation."""

    valence: float = Field(0.0, ge=-1.0, le=1.0)
    arousal: float = Field(0.0, ge=0.0, le=1.0)
    dominant_emotion: str = ""
    summary: str = ""


class NormalisedPercept(EOSBaseModel):
    """
    Atune-specific wrapper around a shared ``Percept``.

    Extends the primitive via composition (not subclassing) with fields
    required by the EIS gate and Fovea prediction error computation.  The
    ``threat_annotations`` list is initialised empty during normalisation
    and populated by the EIS fast-path before Fovea scoring begins.
    """

    percept_id: str = Field(default_factory=new_id)
    threat_annotations: list[ThreatAnnotationLocal] = Field(default_factory=list)
    evidence_tags: list[str] = Field(default_factory=list)
    confidence_score: float = Field(0.5, ge=0.0, le=1.0)
    sentiment: SentimentAnalysis = Field(default_factory=lambda: SentimentAnalysis())
    entities: list[EntityCandidate] = Field(default_factory=list)
    relations: list[RelationCandidate] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Meta-attention
# ---------------------------------------------------------------------------


class MetaContext(BaseModel):
    """Context for the meta-attention controller."""

    risk_level: float = Field(ge=0.0, le=1.0, default=0.0)
    recent_broadcast_count: int = 0
    cycles_since_last_broadcast: int = 0
    active_goal_count: int = 0
    pending_hypothesis_count: int = 0
    # Rhythm state from Synapse (e.g. "flow", "stress", "boredom", "normal")
    rhythm_state: str = "normal"


# ---------------------------------------------------------------------------
# System load (for affect computation)
# ---------------------------------------------------------------------------


class SystemLoad(BaseModel):
    """Current system resource utilisation."""

    cpu_utilisation: float = Field(ge=0.0, le=1.0, default=0.0)
    memory_utilisation: float = Field(ge=0.0, le=1.0, default=0.0)
    queue_depth: int = 0


# ---------------------------------------------------------------------------
# Cache structure
# ---------------------------------------------------------------------------


class AtuneCache(BaseModel):
    """Slowly-changing data cached to meet latency requirements."""

    core_identity_embeddings: list[list[float]] = Field(default_factory=list)
    community_embedding: list[float] = Field(default_factory=list)
    risk_categories: list[RiskCategory] = Field(default_factory=list)
    learned_patterns: list[LearnedPattern] = Field(default_factory=list)
    community_vocabulary: set[str] = Field(default_factory=set)
    active_alerts: list[Alert] = Field(default_factory=list)
    instance_name: str = ""

    # Refresh counters
    cycles_since_identity_refresh: int = 0
    cycles_since_risk_refresh: int = 0
    cycles_since_vocab_refresh: int = 0
    cycles_since_alert_refresh: int = 0

    class Config:
        arbitrary_types_allowed = True
