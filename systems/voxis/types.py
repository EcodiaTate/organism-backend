"""
EcodiaOS - Voxis Internal Types

All Voxis-specific types that don't belong in the shared primitives layer.
These are working types used within the Voxis pipeline; the shared primitives
(Expression, ExpressionStrategy, PersonalityVector) are the external contracts.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import Field

from primitives.affect import AffectState
from primitives.common import EOSBaseModel, Identified, Timestamped, new_id, utc_now
from primitives.expression import PersonalityVector

# ─── Trigger & Channel Enums ─────────────────────────────────────


class ExpressionTrigger(enum.StrEnum):
    """What caused Voxis to consider expressing."""

    # From Nova (deliberate communicative intent)
    NOVA_RESPOND = "nova_respond"          # Responding to user input
    NOVA_INFORM = "nova_inform"            # Proactively sharing information
    NOVA_REQUEST = "nova_request"          # Requesting something from a user
    NOVA_MEDIATE = "nova_mediate"          # Mediating a conflict
    NOVA_CELEBRATE = "nova_celebrate"      # Celebrating an achievement
    NOVA_WARN = "nova_warn"                # Warning about a risk

    # From Atune (reactive - workspace broadcast)
    ATUNE_DISTRESS = "atune_distress"      # Detected distress, immediate response
    ATUNE_DIRECT_ADDRESS = "atune_direct_address"  # Someone spoke to EOS directly

    # From self (ambient / spontaneous)
    AMBIENT_INSIGHT = "ambient_insight"    # Spontaneous thought worth sharing
    AMBIENT_STATUS = "ambient_status"      # Periodic status update


class OutputChannel(enum.StrEnum):
    """Delivery channel for the expression."""

    TEXT_CHAT = "text_chat"            # Direct text in conversation UI
    VOICE = "voice"                    # Synthesised speech
    AMBIENT_VISUAL = "ambient_visual"  # Subtle visual cue in Alive interface
    AMBIENT_SOUND = "ambient_sound"    # Non-verbal audio signal (tone, chime)
    NOTIFICATION = "notification"      # System notification
    STATUS_UPDATE = "status_update"    # Status display (mood indicator)
    FEDERATION_MSG = "federation_msg"  # Communication with another EOS instance


# ─── Rich Strategy Working Type ──────────────────────────────────


class StrategyParams(EOSBaseModel):
    """
    Rich internal expression strategy - built and mutated across pipeline stages.

    At render time this is collapsed into the leaner ExpressionStrategy primitive
    that gets persisted on the Expression node.
    """

    # ── Core intent ──
    intent_type: str = "response"
    trigger: ExpressionTrigger = ExpressionTrigger.NOVA_RESPOND
    context_type: str = "conversation"   # "conversation"|"warning"|"celebration"|"observation"
    urgency: float = 0.5                 # 0.0 (ambient) → 1.0 (critical)
    channel: OutputChannel = OutputChannel.TEXT_CHAT

    # ── Length & pacing ──
    target_length: int = 200             # Characters
    sentence_length_preference: str = "medium"  # "shorter" | "medium" | "longer"
    pacing: str = "balanced"             # "energetic" | "balanced" | "reflective"

    # ── Tone & register ──
    speech_register: str = "neutral"     # "formal" | "casual" | "neutral"
    tone_markers: list[str] = Field(default_factory=list)  # e.g. ["warm", "attentive"]
    contraction_use: bool = True
    greeting_style: str = "neutral"      # "personal" | "professional" | "neutral"

    # ── Content guidance ──
    structure: str = "natural"           # "conclusion_first" | "context_first" | "natural"
    hedge_level: str = "minimal"         # "minimal" | "moderate" | "explicit"
    uncertainty_acknowledgment: str = "implicit"  # "implicit" | "explicit"
    information_density: str = "normal"  # "low" | "normal" | "high"
    explanation_depth: str = "appropriate"  # "thorough" | "appropriate" | "concise"
    assume_knowledge: bool = False
    jargon_level: str = "domain_appropriate"  # "none" | "domain_appropriate" | "technical"

    # ── Questions & curiosity ──
    allows_questions: bool = True
    include_followup_question: bool = False
    exploratory_tangents_allowed: bool = False

    # ── Humour ──
    humour_allowed: bool = False
    humour_probability: float = 0.0
    humour_style: str = "light"          # "light" | "dry" - never sarcastic
    context_appropriate_for_humour: bool = False

    # ── Empathy & emotion ──
    emotional_acknowledgment: str = "implicit"  # "minimal" | "implicit" | "explicit"
    empathy_first: bool = False
    include_wellbeing_check: bool = False

    # ── Personality overrides (applied by affect colouring) ──
    confidence_display_override: str | None = None  # "cautious" | "assertive"
    directness_override: str | None = None          # "high" | "low"
    formality_override: str | None = None           # "relaxed" | "polite" | "professional"
    warmth_boost: float = 0.0

    # ── Analogies & metaphors ──
    analogy_encouraged: bool = False
    preferred_analogy_domains: list[str] = Field(default_factory=list)

    # ── Audience/group ──
    address_style: str = "direct"        # "direct" | "collective"
    avoid_singling_out: bool = False
    reference_shared_history: bool = False
    introduce_self_if_first: bool = False

    # ── Formatting ──
    formatting: str = "prose"            # "prose" | "structured" (bullet points etc.)
    language: str = "en"

    # ── Audience-level overrides ──
    audience_type: str = "individual"    # "individual" | "group" | "community"


# ─── Intent ──────────────────────────────────────────────────────


class ExpressionIntent(EOSBaseModel):
    """
    What Voxis has been asked to express.

    Created either from a workspace broadcast (Atune trigger) or a
    direct Nova request. Encodes the communicative goal.
    """

    id: str = Field(default_factory=new_id)
    trigger: ExpressionTrigger = ExpressionTrigger.NOVA_RESPOND
    content_to_express: str = ""          # The raw content / thought to render
    conversation_id: str | None = None
    addressee_id: str | None = None       # User/entity being addressed (if known)
    intent_id: str | None = None          # Nova intent ID (if triggered by Nova)
    insight_value: float = 0.5            # Relevance/value score (for ambient triggers)
    urgency: float = 0.5
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Audience ────────────────────────────────────────────────────


class AffectEstimate(EOSBaseModel):
    """Estimated emotional state of an external entity (user, group)."""

    distress: float = 0.0
    frustration: float = 0.0
    joy: float = 0.0
    curiosity: float = 0.0
    engagement: float = 0.5
    confidence: float = 0.5


class AudienceProfile(EOSBaseModel):
    """
    Characterisation of who Voxis is speaking to.
    Built from Memory retrieval + conversation history.
    """

    audience_type: str = "individual"    # "individual" | "group" | "community" | "instance"

    # Individual properties
    individual_id: str | None = None
    name: str | None = None
    interaction_count: int = 0           # How many previous interactions
    preferred_register: str = "neutral"  # Learned from past interactions
    technical_level: float = 0.5         # 0.0 (non-technical) → 1.0 (expert)
    emotional_state_estimate: AffectEstimate = Field(default_factory=AffectEstimate)
    communication_preferences: dict[str, Any] = Field(default_factory=dict)
    relationship_strength: float = 0.0   # 0 (stranger) → 1 (deep relationship)

    # Group properties
    group_size: int | None = None
    group_context: str | None = None     # "meeting" | "announcement" | "discussion"

    # Accessibility
    language: str = "en"
    accessibility_needs: list[str] = Field(default_factory=list)


# ─── Somatic Expression Context ──────────────────────────────────


class SomaticExpressionContext(EOSBaseModel):
    """
    Somatic state distilled for expression modulation.

    Carries the organism's felt internal state into the rendering pipeline
    so the expression reflects how the organism actually feels - not just
    what it wants to say.
    """

    arousal: float = 0.5         # Activation level → pacing, speed
    valence: float = 0.0         # Net allostatic trend → tone warmth
    energy: float = 0.6          # Metabolic budget → verbosity
    confidence: float = 0.7      # Generative model fit → hedge level
    coherence: float = 0.75      # Inter-system integration → uncertainty display
    temporal_pressure: float = 0.15  # Urgency → pacing, length
    social_charge: float = 0.3   # Relational engagement → warmth boost
    curiosity_drive: float = 0.5 # Epistemic appetite → question tendency
    integrity: float = 0.9       # Constitutional alignment → honesty emphasis
    nearest_attractor: str = ""  # Current mood basin label
    urgency: float = 0.0        # Soma-computed urgency signal


# ─── Expression Context ──────────────────────────────────────────


class ExpressionContext(EOSBaseModel):
    """
    Full context passed into the renderer.
    Aggregates personality, affect, audience, conversation, and memory context.
    """

    instance_name: str = "EOS"
    personality: PersonalityVector = Field(default_factory=PersonalityVector)
    affect: AffectState = Field(default_factory=AffectState.neutral)
    audience: AudienceProfile = Field(default_factory=AudienceProfile)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    relevant_memories: list[str] = Field(default_factory=list)  # Summarised memory strings
    strategy: StrategyParams = Field(default_factory=StrategyParams)
    intent: ExpressionIntent = Field(default_factory=ExpressionIntent)
    somatic: SomaticExpressionContext = Field(default_factory=SomaticExpressionContext)


# ─── Conversation ────────────────────────────────────────────────


class ConversationMessage(EOSBaseModel):
    """A single message in a conversation."""

    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)
    role: str = "user"                   # "user" | "assistant" | "system"
    content: str = ""
    speaker_id: str | None = None
    affect_valence: float | None = None  # Affect estimate at this message


class ConversationState(EOSBaseModel):
    """
    Live conversation state - serialised to Redis with 24h TTL.
    Maintains a rolling message history with LLM-generated rolling summary
    for efficient context window management.
    """

    conversation_id: str = Field(default_factory=new_id)
    participant_ids: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=utc_now)
    last_active: datetime = Field(default_factory=utc_now)

    # Message history - kept as a list for serialisation (deque rebuilt in memory)
    messages: list[ConversationMessage] = Field(default_factory=list)

    # Rolling summary of older messages (LLM-generated)
    older_messages_summary: str = ""
    summarised_message_count: int = 0

    # Extracted conversation context
    topic_summary: str = ""
    active_topics: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)

    # Emotional trajectory (valence estimates at each exchange)
    emotional_arc: list[float] = Field(default_factory=list)

    # Conversation-level style overrides (temporary adjustments for this conversation)
    style_overrides: dict[str, Any] = Field(default_factory=dict)


# ─── Silence ─────────────────────────────────────────────────────


class SilenceContext(EOSBaseModel):
    """Context inputs for the silence decision."""

    trigger: ExpressionTrigger
    humans_actively_conversing: bool = False
    minutes_since_last_expression: float = 999.0
    min_expression_interval: float = 1.0     # From config
    insight_value: float = 0.5
    urgency: float = 0.5
    conversation_active: bool = False


class SilenceDecision(EOSBaseModel):
    """Result of the silence engine evaluation."""

    speak: bool
    reason: str = ""
    queue: bool = False                  # True = queue for later, False = discard


# ─── Reception & Feedback ────────────────────────────────────────


class ReceptionEstimate(EOSBaseModel):
    """
    Estimated quality of how an expression was received.
    Derived from subsequent user response (if any).
    """

    understood: float = 0.5             # Did they seem to understand?
    emotional_impact: float = 0.0       # Did it affect their emotional state?
    engagement: float = 0.5             # Did they engage with it?
    satisfaction: float = 0.5          # Estimated satisfaction


class ExpressionFeedback(Identified, Timestamped):
    """
    Feedback loop payload sent from Voxis back to Atune after each expression.
    Used by Evo for personality refinement over time.
    """

    expression_id: str = ""
    trigger: str = ""
    conversation_id: str | None = None

    # What was expressed
    content_summary: str = ""
    strategy_register: str = "neutral"
    personality_warmth: float = 0.0

    # How it was received
    inferred_reception: ReceptionEstimate = Field(default_factory=ReceptionEstimate)

    # Affect shift caused by this interaction
    affect_before_valence: float = 0.0
    affect_after_valence: float = 0.0
    affect_delta: float = 0.0           # after - before

    # Was there a user response?
    user_responded: bool = False
    user_response_length: int = 0


# ─── Voice ───────────────────────────────────────────────────────


class VoiceParams(EOSBaseModel):
    """
    TTS parameters derived from personality and affect.
    Used when OutputChannel.VOICE is selected.
    """

    base_voice: str = ""                 # Voice model ID from Seed
    speed: float = 1.0                   # 0.8 → 1.2, modulated by arousal
    pitch_shift: float = 0.0             # -0.1 → +0.1, modulated by valence
    emphasis_level: float = 1.0          # 0.5 → 1.5, modulated by confidence
    pause_frequency: float = 0.5         # Natural pause insertion rate
