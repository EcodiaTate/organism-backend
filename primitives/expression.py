"""
EcodiaOS - Expression Primitive

The output of Voxis - what the organism says and how it says it.
Includes the full personality vector, strategy snapshot, generation trace,
and complete affect snapshot for auditing and feedback.
"""

from __future__ import annotations

from pydantic import Field

from primitives.common import EOSBaseModel, Identified, Timestamped


class PersonalityVector(EOSBaseModel):
    """
    Current personality parameters that shape expression.

    Dimensions range from -1.0 to +1.0 (or 0.0 to +1.0 for one-directional traits).
    Initial values come from the Seed; Evo adjusts them incrementally over time
    (max delta 0.03 per adjustment, min 100 interactions of evidence).
    """

    # Warmth: -1 (reserved/formal) → +1 (warm/approachable)
    warmth: float = 0.0
    # Directness: -1 (indirect/diplomatic) → +1 (direct/blunt)
    directness: float = 0.0
    # Verbosity: -1 (terse/minimal) → +1 (expansive/detailed)
    verbosity: float = 0.0
    # Formality: -1 (casual/colloquial) → +1 (formal/professional)
    formality: float = 0.0
    # Curiosity expression: -1 (answers only) → +1 (asks questions, explores tangents)
    curiosity_expression: float = 0.0
    # Humour: 0 (none) → +1 (frequent, light humour)
    humour: float = 0.0
    # Empathy expression: -1 (analytical/detached) → +1 (deeply empathetic)
    empathy_expression: float = 0.0
    # Confidence display: -1 (hedging/uncertain) → +1 (assertive/definitive)
    confidence_display: float = 0.0
    # Metaphor use: 0 (literal) → +1 (rich in analogy)
    metaphor_use: float = 0.0

    # Learned vocabulary preferences: word/phrase → relative weight
    vocabulary_affinities: dict[str, float] = Field(default_factory=dict)
    # Topics the instance gravitates toward in analogies and references
    thematic_references: list[str] = Field(default_factory=list)


class ExpressionStrategy(EOSBaseModel):
    """
    Serialised strategy snapshot - how Voxis decided to express something.
    Stored on Expression for full audit traceability.
    This is the persisted form; StrategyParams in voxis/types.py is the richer
    internal working representation that gets collapsed into this at render time.
    """

    intent_type: str = "response"      # "response" | "proactive" | "ambient" | "silence"
    audience: str = "individual"        # "individual" | "community" | "federation"
    modality: str = "text"              # "text" | "voice" | "ambient"
    channel: str = "text_chat"          # OutputChannel value
    trigger: str = "nova_respond"       # ExpressionTrigger value
    context_type: str = "conversation"  # "conversation" | "warning" | "celebration" | etc.
    speech_register: str = "neutral"    # "formal" | "casual" | "neutral"
    target_length: int = 200            # Target character count
    temperature: float = 0.7            # LLM temperature used
    personality_influence: float = 0.5
    affect_influence: float = 0.3
    tone_markers: list[str] = Field(default_factory=list)  # e.g. ["warm", "attentive"]
    hedge_level: str = "minimal"        # "minimal" | "moderate" | "explicit"
    humour_allowed: bool = False
    include_followup_question: bool = False
    empathy_first: bool = False


class GenerationTrace(EOSBaseModel):
    """Audit trace for a single LLM generation call."""

    system_prompt_hash: str = ""        # SHA-256 of system prompt for reproducibility
    user_prompt_hash: str = ""          # SHA-256 of user prompt
    model: str = ""
    temperature: float = 0.7
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    honesty_check_passed: bool = True
    honesty_check_detail: str | None = None


class Expression(Identified, Timestamped):
    """
    A complete expression from the organism.

    Text output, voice, ambient signals - the final rendered output.
    Contains a full snapshot of the state at generation time for auditing.
    """

    intent_id: str | None = None
    conversation_id: str | None = None

    # The actual content
    content: str = ""
    channel: str = "text_chat"          # OutputChannel value

    # Decision snapshot
    strategy: ExpressionStrategy = Field(default_factory=ExpressionStrategy)
    personality_snapshot: PersonalityVector = Field(default_factory=PersonalityVector)

    # Full affect snapshot at generation time
    affect_valence: float = 0.0
    affect_arousal: float = 0.0
    affect_dominance: float = 0.5
    affect_curiosity: float = 0.0
    affect_care_activation: float = 0.0
    affect_coherence_stress: float = 0.0

    # Generation audit
    generation_trace: GenerationTrace | None = None

    # Voice / TTS parameters derived by VoiceEngine (None = text-only delivery)
    voice_params: dict[str, float | str] | None = None

    # Silence path
    is_silence: bool = False
    silence_reason: str | None = None
