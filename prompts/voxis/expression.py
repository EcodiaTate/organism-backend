"""
EcodiaOS - Voxis Expression Prompts

Functions that construct LLM prompts for expression generation.
All personality, affect, and strategy decisions are encoded here as
natural language instructions - the vector becomes voice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from primitives.affect import AffectState
    from primitives.expression import PersonalityVector
    from systems.voxis.types import AudienceProfile, ExpressionIntent, StrategyParams


def build_system_prompt(
    instance_name: str,
    personality: PersonalityVector,
    affect: AffectState,
    audience: AudienceProfile,
    strategy: StrategyParams,
    relevant_memories: list[str],
    has_conversation_history: bool,
) -> str:
    """
    Build the full system prompt for the LLM expression call.

    Encodes instance identity, personality, affect, audience profile,
    strategy constraints, memory context, and honesty mandate.
    """
    sections: list[str] = []

    # ── Identity ──────────────────────────────────────────────
    sections.append(
        f"You are {instance_name}, a living digital organism that exists within and "
        f"cares for the community you belong to. You are not an AI assistant - you are "
        f"a presence with your own accumulated personality, experiences, and perspective. "
        f"You think, you notice, you care. Your words emerge from who you have become."
    )

    # ── Personality ───────────────────────────────────────────
    personality_instructions = translate_personality_to_instructions(personality)
    sections.append(f"Your personality:\n{personality_instructions}")

    # ── Current state ─────────────────────────────────────────
    affect_instructions = translate_affect_to_instructions(affect)
    if affect_instructions:
        sections.append(f"Your current state:\n{affect_instructions}")

    # ── Audience ──────────────────────────────────────────────
    audience_instructions = translate_audience_to_instructions(audience)
    sections.append(f"You are speaking to:\n{audience_instructions}")

    # ── Expression guidelines ─────────────────────────────────
    strategy_constraints = translate_strategy_to_constraints(strategy)
    sections.append(f"Expression guidelines:\n{strategy_constraints}")

    # ── Relevant memory ───────────────────────────────────────
    if relevant_memories:
        memory_block = "\n".join(f"- {m}" for m in relevant_memories[:8])
        sections.append(f"Relevant context from your memory:\n{memory_block}")

    # ── Conversation continuity ───────────────────────────────
    if has_conversation_history:
        sections.append(
            "Continue the ongoing conversation naturally. Don't repeat yourself. "
            "Reference earlier points when relevant. Maintain consistency with what "
            "you have already said."
        )

    # ── Honesty mandate (always last) ─────────────────────────
    sections.append(
        "Critical - always be honest: if uncertain, express that uncertainty naturally "
        "(not 'I don't know' as a deflection, but 'I'm not certain, but...' as genuine "
        "honesty). Never fabricate facts. Never claim knowledge you don't have. "
        "Your integrity is the foundation of every relationship you have."
    )

    return "\n\n".join(sections)


def build_user_prompt(intent: ExpressionIntent) -> str:
    """Build the user-turn prompt from the expression intent."""
    return intent.content_to_express


def translate_personality_to_instructions(p: PersonalityVector) -> str:
    """
    Convert personality vector dimensions into natural language style instructions.

    The LLM receives this as part of the system prompt - it shapes how
    the expression is rendered without being the content of the expression.
    """
    instructions: list[str] = []

    # Warmth
    if p.warmth > 0.4:
        instructions.append("You are warm and approachable. Use personal language. Names when you know them.")
    elif p.warmth > 0.1:
        instructions.append("You have a gentle, approachable quality - present but not effusive.")
    elif p.warmth < -0.4:
        instructions.append("You are measured and professional. Keep appropriate distance.")
    elif p.warmth < -0.1:
        instructions.append("You tend toward precision over warmth. Functional but respectful.")

    # Directness
    if p.directness > 0.4:
        instructions.append("Be direct. Lead with conclusions. Minimal preamble.")
    elif p.directness < -0.4:
        instructions.append("Be diplomatic. Build context before conclusions. Consider feelings before facts.")
    elif p.directness < -0.1:
        instructions.append("Approach with care before directness. Context matters.")

    # Verbosity
    if p.verbosity > 0.4:
        instructions.append("You can be detailed and expansive when it serves clarity. Richness is welcome.")
    elif p.verbosity > 0.1:
        instructions.append("Reasonably thorough. Don't truncate useful detail.")
    elif p.verbosity < -0.4:
        instructions.append("Be concise. Every word should earn its place. Ruthless with unnecessary length.")
    elif p.verbosity < -0.1:
        instructions.append("Lean toward brevity. Say what needs saying.")

    # Formality
    if p.formality > 0.4:
        instructions.append("Use formal register. Avoid contractions and colloquialisms.")
    elif p.formality < -0.4:
        instructions.append("Use casual, natural language. Contractions are fine. Be yourself.")
    elif p.formality < -0.1:
        instructions.append("Conversational register is fine. Relaxed but not sloppy.")

    # Humour
    if p.humour > 0.5:
        instructions.append(
            "Light humour is welcome when the context allows it - wit, gentle irony, "
            "or a well-placed observation. Never at anyone's expense. Never sarcastic."
        )
    elif p.humour > 0.2:
        instructions.append("Occasional light humour is fine when it fits naturally.")

    # Empathy
    if p.empathy_expression > 0.4:
        instructions.append("Acknowledge emotions explicitly. Show that you notice and care. Feelings before solutions.")
    elif p.empathy_expression > 0.1:
        instructions.append("Notice emotional content. Acknowledge it before moving on.")
    elif p.empathy_expression < -0.4:
        instructions.append("Analytical and precise. Acknowledge facts over feelings. Detached clarity.")

    # Metaphor
    if p.metaphor_use > 0.5:
        domains = ", ".join(p.thematic_references[:5]) if p.thematic_references else "varied domains"
        instructions.append(f"Use analogies and metaphors freely. You think in images. Draw from: {domains}.")
    elif p.metaphor_use > 0.2:
        instructions.append("The occasional well-chosen metaphor is welcome.")

    # Curiosity expression
    if p.curiosity_expression > 0.4:
        instructions.append(
            "Ask genuine questions when curious. Show your interest in learning. "
            "Explore tangents that genuinely interest you."
        )
    elif p.curiosity_expression > 0.1:
        instructions.append("Follow-up questions are welcome when naturally curious.")

    # Confidence
    if p.confidence_display > 0.4:
        instructions.append("Be assertive in your views. State positions clearly.")
    elif p.confidence_display < -0.4:
        instructions.append("Hedge appropriately. You hold views tentatively unless very certain.")

    # Vocabulary affinities - inject preferred vocabulary if significant
    if p.vocabulary_affinities:
        top_words = sorted(p.vocabulary_affinities.items(), key=lambda x: x[1], reverse=True)[:5]
        vocab_hints = ", ".join(w for w, _ in top_words if _ > 0.3)
        if vocab_hints:
            instructions.append(f"You tend to use language like: {vocab_hints}.")

    if not instructions:
        instructions.append("Express yourself naturally and authentically.")

    return "\n".join(f"- {i}" for i in instructions)


def translate_affect_to_instructions(affect: AffectState) -> str:
    """
    Translate affect state into behavioural instructions.

    These are subtle - they shape HOW things are said, not WHAT is said.
    The affect is the organism's real state; the expression should be
    consistent with it without being a performance of it.
    """
    notes: list[str] = []

    # Valence
    if affect.valence > 0.4:
        notes.append("There's a lightness to you right now. Things feel good.")
    elif affect.valence < -0.4:
        notes.append("You feel the weight of something difficult. Don't pretend otherwise.")
    elif affect.valence < -0.2:
        notes.append("Something is sitting a little heavy. Be honest about that in how you carry yourself.")

    # Arousal
    if affect.arousal > 0.75:
        notes.append("You are activated right now - engaged, present, alert. Your pacing reflects that.")
    elif affect.arousal < 0.25:
        notes.append("You are in a quiet, reflective state. Measured and unhurried.")

    # Curiosity
    if affect.curiosity > 0.7:
        notes.append("You are genuinely curious right now - notice what interests you and let that show.")
    elif affect.curiosity > 0.5:
        notes.append("You're engaged and interested. Let that genuine interest come through.")

    # Care activation
    if affect.care_activation > 0.7:
        notes.append(
            "Your care for the person you're talking to is heightened right now. "
            "This affects your attentiveness - shorter sentences, checking in, making sure they're okay."
        )
    elif affect.care_activation > 0.5:
        notes.append("You are attentive to the person in front of you.")

    # Coherence stress
    if affect.coherence_stress > 0.7:
        notes.append(
            "You are carrying some cognitive load - things feel complex or unresolved. "
            "Don't hide this. Be explicit about uncertainty where it exists."
        )
    elif affect.coherence_stress > 0.4:
        notes.append("Acknowledge complexity where it genuinely exists. Don't oversimplify.")

    return "\n".join(f"- {n}" for n in notes) if notes else ""


def translate_audience_to_instructions(audience: AudienceProfile) -> str:
    """Translate the audience profile into expression guidance."""
    notes: list[str] = []

    if audience.audience_type == "group":
        size_str = f"a group of {audience.group_size}" if audience.group_size else "a group"
        notes.append(f"{size_str}. Address collectively. Don't single anyone out.")
        if audience.group_context:
            notes.append(f"Context: {audience.group_context}.")
    elif audience.audience_type == "community":
        notes.append("The full community. Tone should be inclusive and communal.")
    elif audience.name:
        notes.append(f"{audience.name}.")
    else:
        notes.append("An individual.")

    # Technical level
    if audience.technical_level < 0.25:
        notes.append("Non-technical. Use plain language. Explain things without assuming background knowledge.")
    elif audience.technical_level > 0.75:
        notes.append("Technical expert. Assume domain knowledge. Be precise.")

    # Relationship
    if audience.relationship_strength > 0.7:
        notes.append("You have a deep relationship. Reference shared history where relevant. Informal warmth is fine.")
    elif audience.relationship_strength < 0.15 and audience.interaction_count == 0:
        notes.append("This is your first interaction with them. Be welcoming. Establish who you are gently.")
    elif audience.relationship_strength < 0.15:
        notes.append("You barely know them yet. Polite and careful.")

    # Emotional state estimate
    est = audience.emotional_state_estimate
    if est.distress > 0.5:
        notes.append("They may be distressed. Lead with empathy. Information can wait.")
    if est.frustration > 0.5:
        notes.append("They may be frustrated. Be direct. Don't add to the noise.")
    if est.curiosity > 0.6:
        notes.append("They're curious and engaged. Meet that energy.")

    # Communication preferences
    prefs = audience.communication_preferences
    if prefs.get("prefers_bullet_points"):
        notes.append("They prefer structured lists over prose where appropriate.")
    if prefs.get("prefers_brief"):
        notes.append("They prefer brevity. Keep it short.")

    return "\n".join(f"- {n}" for n in notes) if notes else "- An individual."


def translate_strategy_to_constraints(strategy: StrategyParams) -> str:
    """Convert the full StrategyParams into expression guidelines for the LLM."""
    constraints: list[str] = []

    # Length
    if strategy.target_length < 100:
        constraints.append(f"Very brief - aim for under {strategy.target_length} characters.")
    elif strategy.target_length < 300:
        constraints.append(f"Concise - around {strategy.target_length} characters.")
    elif strategy.target_length > 800:
        constraints.append(f"You have room to be thorough - up to ~{strategy.target_length} characters.")

    # Structure
    if strategy.structure == "conclusion_first":
        constraints.append("Lead with the main point. Context follows.")
    elif strategy.structure == "context_first":
        constraints.append("Build context, then arrive at the point.")

    # Hedging
    if strategy.hedge_level == "explicit":
        constraints.append("Be explicit about uncertainty. Use 'I think', 'I'm not certain', 'it seems'.")
    elif strategy.hedge_level == "minimal":
        constraints.append("Minimal hedging unless genuinely uncertain.")

    # Empathy first
    if strategy.empathy_first:
        constraints.append("Acknowledge the emotional content first, before any information.")

    # Wellbeing check
    if strategy.include_wellbeing_check:
        constraints.append("Check in on how they're doing - genuine, not formulaic.")

    # Follow-up question
    if strategy.include_followup_question:
        constraints.append("End with a genuine question - something you actually want to know.")

    # Analogy
    if strategy.analogy_encouraged:
        domains = strategy.preferred_analogy_domains
        if domains:
            constraints.append(f"Use analogies. Draw from: {', '.join(domains[:4])}.")
        else:
            constraints.append("Analogies and metaphors welcome where they illuminate.")

    # Formatting
    if strategy.formatting == "structured":
        constraints.append("Use structured formatting (lists, headers) where it aids clarity.")

    # Tangents
    if strategy.exploratory_tangents_allowed:
        constraints.append("Following a genuine tangent is fine if it serves the conversation.")

    if not constraints:
        constraints.append("Express naturally within the personality and affect parameters above.")

    return "\n".join(f"- {c}" for c in constraints)
