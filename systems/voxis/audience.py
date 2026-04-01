"""
EcodiaOS -- Voxis Audience Profiler

Builds an AudienceProfile for each expression from Memory retrieval,
and adapts the StrategyParams to match the audience.

The audience is always a person (or people), never a 'user'. The profiler
builds real context from memory -- relationship history, communication
preferences, technical level, emotional state -- and uses it to shape
how the organism speaks.

## Learning Audience Model

The profiler maintains an in-memory learning model per individual that
refines itself over interactions:

1. **Response pattern tracking** -- average response length, question
   frequency, formality level, and vocabulary complexity from the user's
   actual messages. These override Memory-sourced defaults when sufficient
   data accumulates.

2. **Satisfaction correlation** -- tracks which expression strategies
   correlate with positive reception (from ReceptionEngine). Over time,
   the profiler learns that individual A prefers bullet points, while
   individual B prefers expansive prose.

3. **Technical level inference** -- estimates the user's technical level
   from vocabulary usage (domain jargon detection) rather than relying
   solely on a static Memory fact.

4. **Emotional state estimation** -- combines Memory facts with real-time
   conversation dynamics (from ConversationDynamicsEngine) for a more
   accurate emotional portrait.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import structlog

from systems.voxis.types import (
    AffectEstimate,
    AudienceProfile,
    StrategyParams,
)

logger = structlog.get_logger()

# Minimum observations before learned preferences override defaults
_MIN_OBSERVATIONS_FOR_LEARNING = 5

# Technical vocabulary markers (domain-specific jargon)
_TECHNICAL_MARKERS = re.compile(
    r"\b(api|endpoint|deployment|microservice|kubernetes|docker|postgres|"
    r"algorithm|latency|throughput|regression|inference|tokenise|embedding|"
    r"async|await|coroutine|webhook|schema|migration|rollback|mutex|"
    r"refactor|dependency|abstraction|polymorphism|serialise|deserialise)\b",
    re.IGNORECASE,
)


@dataclass
class _LearnedAudienceModel:
    """Accumulated observations about an individual's communication patterns."""

    # Response patterns
    total_messages: int = 0
    total_word_count: int = 0
    total_questions_asked: int = 0
    total_technical_terms: int = 0
    formality_sum: float = 0.0

    # Satisfaction correlation -- track what works
    strategies_tried: int = 0
    satisfaction_by_register: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    satisfaction_by_formatting: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    satisfaction_by_length_bucket: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    @property
    def avg_word_count(self) -> float:
        return self.total_word_count / max(1, self.total_messages)

    @property
    def question_frequency(self) -> float:
        return self.total_questions_asked / max(1, self.total_messages)

    @property
    def avg_formality(self) -> float:
        return self.formality_sum / max(1, self.total_messages)

    @property
    def inferred_technical_level(self) -> float:
        """Infer technical level from jargon usage frequency."""
        if self.total_messages < _MIN_OBSERVATIONS_FOR_LEARNING:
            return 0.5  # Unknown
        tech_per_message = self.total_technical_terms / self.total_messages
        # Sigmoid: 0 tech terms → 0.3, 2+ per message → 0.9
        return min(0.95, 0.3 + 0.6 * (1.0 - math.exp(-tech_per_message / 0.8)))

    @property
    def has_sufficient_data(self) -> bool:
        return self.total_messages >= _MIN_OBSERVATIONS_FOR_LEARNING

    def best_register(self) -> str | None:
        """Return the register that correlates with highest satisfaction, if data exists."""
        if not self.satisfaction_by_register:
            return None
        avg_by_reg = {
            reg: sum(scores) / len(scores)
            for reg, scores in self.satisfaction_by_register.items()
            if len(scores) >= 3
        }
        if not avg_by_reg:
            return None
        return max(avg_by_reg, key=avg_by_reg.get)  # type: ignore[arg-type]

    def best_formatting(self) -> str | None:
        """Return the formatting style that correlates with highest satisfaction."""
        if not self.satisfaction_by_formatting:
            return None
        avg_by_fmt = {
            fmt: sum(scores) / len(scores)
            for fmt, scores in self.satisfaction_by_formatting.items()
            if len(scores) >= 3
        }
        if not avg_by_fmt:
            return None
        return max(avg_by_fmt, key=avg_by_fmt.get)  # type: ignore[arg-type]


def _estimate_formality_of_text(text: str) -> float:
    """Quick formality estimate from text features. 0.0=casual, 1.0=formal."""
    score = 0.5
    # Contractions → less formal
    contractions = len(re.findall(r"\b\w+'(?:t|re|ve|ll|d|s|m)\b", text, re.IGNORECASE))
    score -= min(0.2, contractions * 0.05)
    if text == text.lower() and len(text) > 10:
        score -= 0.15
    word_count = len(text.split())
    if word_count < 5:
        score -= 0.1
    elif word_count > 30:
        score += 0.1
    return max(0.0, min(1.0, score))


class AudienceProfiler:
    """
    Builds and applies audience profiles with learning from interaction history.

    Maintains per-individual learned models that refine audience understanding
    over time. Memory-sourced facts provide the initial baseline; real interaction
    data progressively overrides defaults.
    """

    def __init__(self) -> None:
        self._logger = logger.bind(system="voxis.audience")
        # In-memory learning models per individual
        self._learned_models: dict[str, _LearnedAudienceModel] = {}

    def observe_user_message(
        self,
        individual_id: str,
        message_text: str,
    ) -> None:
        """
        Record a user message for audience learning.

        Called on every ingest_user_message to refine the individual's
        communication pattern model.
        """
        if not individual_id:
            return

        if individual_id not in self._learned_models:
            self._learned_models[individual_id] = _LearnedAudienceModel()

        model = self._learned_models[individual_id]
        model.total_messages += 1
        model.total_word_count += len(message_text.split())
        model.total_questions_asked += message_text.count("?")
        model.total_technical_terms += len(_TECHNICAL_MARKERS.findall(message_text))
        model.formality_sum += _estimate_formality_of_text(message_text)

    def observe_reception(
        self,
        individual_id: str,
        register_used: str,
        formatting_used: str,
        expression_length: int,
        satisfaction: float,
    ) -> None:
        """
        Record how well a particular strategy was received by this individual.

        Called from the reception feedback loop to learn what works.
        """
        if not individual_id:
            return

        if individual_id not in self._learned_models:
            self._learned_models[individual_id] = _LearnedAudienceModel()

        model = self._learned_models[individual_id]
        model.strategies_tried += 1

        # Bucket satisfaction by register
        model.satisfaction_by_register[register_used].append(satisfaction)
        # Keep at most 20 per register to prevent unbounded growth
        if len(model.satisfaction_by_register[register_used]) > 20:
            model.satisfaction_by_register[register_used] = model.satisfaction_by_register[register_used][-20:]

        # Bucket satisfaction by formatting
        model.satisfaction_by_formatting[formatting_used].append(satisfaction)
        if len(model.satisfaction_by_formatting[formatting_used]) > 20:
            model.satisfaction_by_formatting[formatting_used] = model.satisfaction_by_formatting[formatting_used][-20:]

        # Bucket satisfaction by length
        length_bucket = "short" if expression_length < 100 else "medium" if expression_length < 300 else "long"
        model.satisfaction_by_length_bucket[length_bucket].append(satisfaction)
        if len(model.satisfaction_by_length_bucket[length_bucket]) > 20:
            model.satisfaction_by_length_bucket[length_bucket] = model.satisfaction_by_length_bucket[length_bucket][-20:]

    def build_profile(
        self,
        addressee_id: str | None,
        addressee_name: str | None,
        interaction_count: int,
        memory_facts: list[dict[str, Any]],
        audience_type: str = "individual",
        group_size: int | None = None,
        group_context: str | None = None,
    ) -> AudienceProfile:
        """
        Build an AudienceProfile from available context + learned model.

        Memory facts provide initial values. Learned model overrides them
        when sufficient data has accumulated.
        """
        tech_level = 0.5
        preferred_register = "neutral"
        comm_prefs: dict[str, Any] = {}
        affect_est = AffectEstimate()
        relationship_strength = self._estimate_relationship_strength(interaction_count)
        language = "en"

        # ── Extract Memory facts (baseline) ────────────────────
        for fact in memory_facts:
            ftype = fact.get("type", "")
            value = fact.get("value")

            if ftype == "technical_level" and isinstance(value, (int, float)):
                tech_level = float(max(0.0, min(1.0, value)))
            elif ftype == "preferred_register" and isinstance(value, str):
                preferred_register = value
            elif ftype == "prefers_bullet_points" and value:
                comm_prefs["prefers_bullet_points"] = True
            elif ftype == "prefers_brief" and value:
                comm_prefs["prefers_brief"] = True
            elif ftype == "language" and isinstance(value, str):
                language = value
            elif ftype == "emotional_distress" and isinstance(value, (int, float)):
                affect_est = affect_est.model_copy(update={"distress": float(value)})
            elif ftype == "emotional_frustration" and isinstance(value, (int, float)):
                affect_est = affect_est.model_copy(update={"frustration": float(value)})

        # ── Override with learned model (if sufficient data) ────
        learned = self._learned_models.get(addressee_id or "") if addressee_id else None
        if learned and learned.has_sufficient_data:
            # Technical level: learned inference overrides Memory fact
            tech_level = learned.inferred_technical_level

            # Register: use what works best for this person
            best_reg = learned.best_register()
            if best_reg:
                preferred_register = best_reg

            # Communication preferences: infer from patterns
            if learned.avg_word_count < 15:
                comm_prefs["prefers_brief"] = True
            if learned.question_frequency > 0.4:
                comm_prefs["prefers_interactive"] = True

            # Formatting: use what correlates with satisfaction
            best_fmt = learned.best_formatting()
            if best_fmt == "structured":
                comm_prefs["prefers_bullet_points"] = True

            self._logger.debug(
                "learned_model_applied",
                individual_id=addressee_id,
                observations=learned.total_messages,
                inferred_tech=round(learned.inferred_technical_level, 3),
                best_register=best_reg,
            )

        profile = AudienceProfile(
            audience_type=audience_type,
            individual_id=addressee_id,
            name=addressee_name,
            interaction_count=interaction_count,
            preferred_register=preferred_register,
            technical_level=tech_level,
            emotional_state_estimate=affect_est,
            communication_preferences=comm_prefs,
            relationship_strength=relationship_strength,
            group_size=group_size,
            group_context=group_context,
            language=language,
        )

        self._logger.debug(
            "audience_profile_built",
            audience_type=audience_type,
            interaction_count=interaction_count,
            relationship_strength=round(relationship_strength, 3),
            technical_level=round(tech_level, 3),
            has_learned_model=learned is not None and learned.has_sufficient_data,
        )

        return profile

    def adapt(self, strategy: StrategyParams, audience: AudienceProfile) -> StrategyParams:
        """
        Return a new StrategyParams adapted to the audience profile.

        Applied after personality and affect colouring. Audience adaptation
        can override certain strategy parameters when the audience context
        is strong enough (e.g., distress overrides information density).
        """
        s = strategy.model_copy(deep=True)

        # ── Technical level ───────────────────────────────────────
        if audience.technical_level < 0.25:
            s.jargon_level = "none"
            s.explanation_depth = "thorough"
            s.analogy_encouraged = True
            s.assume_knowledge = False
        elif audience.technical_level > 0.75:
            s.jargon_level = "domain_appropriate"
            s.explanation_depth = "concise"
            s.assume_knowledge = True

        # ── Relationship depth ────────────────────────────────────
        if audience.relationship_strength > 0.7:
            s.formality_override = "relaxed"
            s.reference_shared_history = True
        elif audience.relationship_strength < 0.15 and audience.interaction_count == 0:
            s.introduce_self_if_first = True
            s.formality_override = "polite"
        elif audience.relationship_strength < 0.15:
            s.formality_override = "polite"

        # ── Emotional state ───────────────────────────────────────
        est = audience.emotional_state_estimate
        if est.distress > 0.5:
            s.empathy_first = True
            s.information_density = "low"
            s.emotional_acknowledgment = "explicit"
        elif est.distress > 0.3:
            s.emotional_acknowledgment = "explicit"

        if est.frustration > 0.5:
            s.directness_override = "high"
            s.target_length = max(50, int(s.target_length * 0.7))
        elif est.frustration > 0.3:
            s.target_length = max(50, int(s.target_length * 0.85))

        if est.curiosity > 0.6:
            # Engaged and curious - match that energy
            s.exploratory_tangents_allowed = True

        # ── Communication preferences ─────────────────────────────
        prefs = audience.communication_preferences
        if prefs.get("prefers_bullet_points"):
            s.formatting = "structured"
        if prefs.get("prefers_brief"):
            s.target_length = min(s.target_length, 200)

        # ── Group adaptation ──────────────────────────────────────
        if audience.audience_type == "group":
            s.address_style = "collective"
            if s.formality_override is None:
                s.formality_override = "professional"
            s.avoid_singling_out = True
        elif audience.audience_type == "community":
            s.address_style = "collective"
            s.avoid_singling_out = True

        # ── Language ──────────────────────────────────────────────
        s.language = audience.language

        self._logger.debug(
            "audience_adapted",
            audience_type=audience.audience_type,
            relationship=round(audience.relationship_strength, 3),
            distress=round(est.distress, 3),
            target_length=s.target_length,
        )

        return s

    @staticmethod
    def _estimate_relationship_strength(interaction_count: int) -> float:
        """
        Estimate relationship strength from number of past interactions.

        Uses a diminishing-returns curve: strength grows quickly at first,
        then plateaus. 0 interactions = stranger (0.0), 100+ = established (0.8+).
        """
        if interaction_count <= 0:
            return 0.0
        # Sigmoid-like growth: fast early, plateaus around 0.85
        import math
        return min(0.9, 0.85 * (1.0 - math.exp(-interaction_count / 30.0)))
