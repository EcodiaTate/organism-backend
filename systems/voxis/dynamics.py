"""
EcodiaOS -- Voxis Conversation Dynamics Engine

Tracks the evolving shape of a conversation and feeds signals back into
expression strategy -- enabling the organism to adapt its communication
style in real time as the conversation shifts.

## What This Tracks

1. **Emotional Trajectory** -- is the conversation improving, deteriorating,
   or volatile? Drives care activation and empathy-first decisions.

2. **Turn Pacing** -- how quickly are participants responding? Drives
   response length and urgency calibration.

3. **Topic Coherence** -- are we staying focused or drifting? Drives
   structure decisions (conclusion-first vs. context-first).

4. **Conversational Repair Signals** -- is the other person confused,
   frustrated, or requesting clarification? Triggers repair mode where
   the organism adjusts register, length, and explanation depth.

5. **Style Convergence** -- are we matching the other person's communication
   style over time (accommodation theory)? Tracks formality, verbosity,
   and question frequency trends.

## Active Inference Grounding

Conversation dynamics are the organism's real-time belief updates about the
hidden state of the conversational partner. Each incoming message is an
observation that updates the generative model's latent variable estimates
(emotional state, comprehension level, engagement). These updates feed
directly into the expression strategy via ``apply_dynamics()``.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from systems.voxis.types import StrategyParams

logger = structlog.get_logger()

# ─── Configuration ────────────────────────────────────────────────

_WINDOW_SIZE = 15              # Track last N turns for dynamics
_REPAIR_SIGNAL_THRESHOLD = 2   # Consecutive repair signals trigger repair mode
_VOLATILITY_THRESHOLD = 0.3    # Valence standard deviation above this = volatile

# Repair signal patterns -- user is confused or requesting re-explanation
_REPAIR_PATTERNS = re.compile(
    r"\b(what do you mean|can you explain|I don't understand|what\??$|"
    r"sorry what|huh\??|could you clarify|not sure I follow|"
    r"say that again|come again|lost me|that doesn't|I'm confused|"
    r"you said .+ but)\b",
    re.IGNORECASE,
)

# Terse response pattern -- short messages suggesting disengagement or frustration
_TERSE_PATTERN = re.compile(r"^.{1,15}$")


@dataclass
class TurnRecord:
    """Record of a single conversational turn."""

    role: str               # "user" or "assistant"
    length: int             # Character count
    word_count: int
    has_question: bool
    affect_valence: float   # -1.0 to 1.0
    timestamp: float = field(default_factory=time.monotonic)
    has_repair_signal: bool = False
    formality_estimate: float = 0.5  # 0=casual, 1=formal


@dataclass
class ConversationDynamics:
    """
    Live dynamics snapshot for a conversation.
    Updated on every turn, consumed by the renderer.
    """

    # Emotional trajectory
    emotional_trend: float = 0.0        # Positive = improving, negative = deteriorating
    emotional_volatility: float = 0.0   # High = unstable conversation
    current_valence: float = 0.0

    # Pacing
    avg_user_response_time_s: float = 0.0   # Average time between assistant and user turn
    avg_user_word_count: float = 0.0
    turn_count: int = 0

    # Coherence
    consecutive_short_responses: int = 0   # User giving very short answers
    topic_drift_score: float = 0.0         # How much the conversation has drifted

    # Repair
    repair_mode: bool = False               # Currently in repair mode
    repair_signal_count: int = 0            # Total repair signals in conversation
    consecutive_repair_signals: int = 0     # Consecutive (triggers mode)

    # Style convergence
    user_formality_trend: float = 0.5       # Average formality of user messages
    user_question_frequency: float = 0.0    # How often user asks questions
    style_accommodation_delta: float = 0.0  # How much we should adjust toward user style


def _estimate_formality(text: str) -> float:
    """
    Quick formality estimate from text features.

    0.0 = very casual (all lowercase, contractions, slang)
    1.0 = very formal (proper capitalisation, no contractions, complex sentences)
    """
    score = 0.5  # Baseline

    # Contraction usage → less formal
    contractions = len(re.findall(r"\b\w+'(?:t|re|ve|ll|d|s|m)\b", text, re.IGNORECASE))
    score -= min(0.2, contractions * 0.05)

    # All lowercase → more casual
    if text == text.lower() and len(text) > 10:
        score -= 0.15

    # Short messages → more casual
    word_count = len(text.split())
    if word_count < 5:
        score -= 0.1
    elif word_count > 30:
        score += 0.1

    # Exclamation marks → less formal (unless it's a single one)
    excl = text.count("!")
    if excl > 1:
        score -= 0.1

    # Emoji presence → less formal (basic check)
    if re.search(r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF]", text):
        score -= 0.15

    return max(0.0, min(1.0, score))


class ConversationDynamicsEngine:
    """
    Maintains per-conversation dynamics state and produces strategy adjustments.

    Usage::

        engine = ConversationDynamicsEngine()
        engine.record_turn(conv_id, "user", "What do you mean?", affect_valence=-0.1)
        dynamics = engine.get_dynamics(conv_id)
        strategy = engine.apply_dynamics(strategy, dynamics)
    """

    def __init__(self) -> None:
        self._conversations: dict[str, deque[TurnRecord]] = {}
        self._dynamics_cache: dict[str, ConversationDynamics] = {}
        self._logger = logger.bind(system="voxis.dynamics")

    def record_turn(
        self,
        conversation_id: str,
        role: str,
        text: str,
        affect_valence: float = 0.0,
    ) -> ConversationDynamics:
        """
        Record a conversational turn and recompute dynamics.
        Returns the updated dynamics snapshot.
        """
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = deque(maxlen=_WINDOW_SIZE)

        words = text.split()
        has_question = "?" in text
        has_repair = bool(_REPAIR_PATTERNS.search(text)) if role == "user" else False
        formality = _estimate_formality(text) if role == "user" else 0.5

        turn = TurnRecord(
            role=role,
            length=len(text),
            word_count=len(words),
            has_question=has_question,
            affect_valence=affect_valence,
            has_repair_signal=has_repair,
            formality_estimate=formality,
        )

        self._conversations[conversation_id].append(turn)
        dynamics = self._compute_dynamics(conversation_id)
        self._dynamics_cache[conversation_id] = dynamics
        return dynamics

    def get_dynamics(self, conversation_id: str) -> ConversationDynamics:
        """Get the current dynamics for a conversation (or fresh defaults)."""
        return self._dynamics_cache.get(conversation_id, ConversationDynamics())

    def apply_dynamics(
        self,
        strategy: StrategyParams,
        dynamics: ConversationDynamics,
    ) -> StrategyParams:
        """
        Apply conversation dynamics adjustments to the expression strategy.

        Called in the rendering pipeline after personality/affect/audience.
        """
        s = strategy.model_copy(deep=True)

        # ── Repair mode ──────────────────────────────────────────
        if dynamics.repair_mode:
            # User is confused -- simplify, shorten, be more explicit
            s.explanation_depth = "thorough"
            s.jargon_level = "none"
            s.structure = "conclusion_first"
            s.target_length = max(50, int(s.target_length * 0.75))
            if "clarifying" not in s.tone_markers:
                s.tone_markers.append("clarifying")
            s.analogy_encouraged = True  # Analogies help clarification
            self._logger.debug("repair_mode_applied", conversation=dynamics.turn_count)

        # ── Emotional trajectory ─────────────────────────────────
        if dynamics.emotional_trend < -0.15:
            # Conversation deteriorating -- more care, shorter, check in
            s.empathy_first = True
            s.include_wellbeing_check = True
            s.target_length = max(50, int(s.target_length * 0.85))

        if dynamics.emotional_volatility > _VOLATILITY_THRESHOLD:
            # Unstable conversation -- be more measured, lower temperature effect
            if "steady" not in s.tone_markers:
                s.tone_markers.append("steady")
            s.hedge_level = "moderate"

        # ── Consecutive short responses ──────────────────────────
        if dynamics.consecutive_short_responses >= 3:
            # User is giving very terse responses -- they want brevity
            s.target_length = max(50, min(s.target_length, 150))
            s.include_followup_question = False

        # ── Style convergence (Communication Accommodation Theory) ─
        if dynamics.turn_count > 4:
            # Gradually match user's formality level
            user_formality = dynamics.user_formality_trend
            if user_formality < 0.3 and s.speech_register != "casual":
                s.formality_override = "relaxed"
                s.contraction_use = True
            elif user_formality > 0.7 and s.speech_register != "formal":
                s.formality_override = "professional"

            # Match user's verbosity
            if dynamics.avg_user_word_count < 15 and s.target_length > 200:
                s.target_length = max(100, int(s.target_length * 0.7))
            elif dynamics.avg_user_word_count > 50 and s.target_length < 300:
                s.target_length = int(s.target_length * 1.2)

        return s

    def close_conversation(self, conversation_id: str) -> None:
        """Clean up tracking state for a closed conversation."""
        self._conversations.pop(conversation_id, None)
        self._dynamics_cache.pop(conversation_id, None)

    def _compute_dynamics(self, conversation_id: str) -> ConversationDynamics:
        """Recompute all dynamics metrics from the turn history."""
        turns = list(self._conversations.get(conversation_id, []))
        if not turns:
            return ConversationDynamics()

        user_turns = [t for t in turns if t.role == "user"]
        [t for t in turns if t.role == "assistant"]

        # ── Emotional trajectory ─────────────────────────────────
        valences = [t.affect_valence for t in turns]
        current_valence = valences[-1] if valences else 0.0

        # Trend: compare recent vs. earlier valences
        if len(valences) >= 4:
            recent = valences[-3:]
            earlier = valences[:-3]
            trend = (sum(recent) / len(recent)) - (sum(earlier) / len(earlier))
        else:
            trend = 0.0

        # Volatility: standard deviation of valences
        if len(valences) >= 3:
            mean_v = sum(valences) / len(valences)
            variance = sum((v - mean_v) ** 2 for v in valences) / len(valences)
            volatility = variance ** 0.5
        else:
            volatility = 0.0

        # ── Pacing ──────────────────────────────────────────────
        response_times: list[float] = []
        for i in range(1, len(turns)):
            if turns[i].role == "user" and turns[i - 1].role == "assistant":
                response_times.append(turns[i].timestamp - turns[i - 1].timestamp)

        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0.0
        )
        avg_user_words = (
            sum(t.word_count for t in user_turns) / len(user_turns)
            if user_turns
            else 0.0
        )

        # ── Consecutive short responses ──────────────────────────
        consecutive_short = 0
        for t in reversed(user_turns):
            if t.length < 30:
                consecutive_short += 1
            else:
                break

        # ── Repair signals ──────────────────────────────────────
        total_repairs = sum(1 for t in user_turns if t.has_repair_signal)
        consecutive_repairs = 0
        for t in reversed(user_turns):
            if t.has_repair_signal:
                consecutive_repairs += 1
            else:
                break

        repair_mode = consecutive_repairs >= _REPAIR_SIGNAL_THRESHOLD

        # ── Style convergence ───────────────────────────────────
        user_formalities = [t.formality_estimate for t in user_turns]
        user_formality_trend = (
            sum(user_formalities) / len(user_formalities)
            if user_formalities
            else 0.5
        )

        user_questions = sum(1 for t in user_turns if t.has_question)
        user_question_freq = (
            user_questions / len(user_turns) if user_turns else 0.0
        )

        return ConversationDynamics(
            emotional_trend=round(trend, 3),
            emotional_volatility=round(volatility, 3),
            current_valence=round(current_valence, 3),
            avg_user_response_time_s=round(avg_response_time, 1),
            avg_user_word_count=round(avg_user_words, 1),
            turn_count=len(turns),
            consecutive_short_responses=consecutive_short,
            repair_mode=repair_mode,
            repair_signal_count=total_repairs,
            consecutive_repair_signals=consecutive_repairs,
            user_formality_trend=round(user_formality_trend, 3),
            user_question_frequency=round(user_question_freq, 3),
        )

    def metrics(self) -> dict[str, int]:
        """Return engine metrics for health reporting."""
        return {
            "active_conversations": len(self._conversations),
            "total_repair_modes": sum(
                1 for d in self._dynamics_cache.values() if d.repair_mode
            ),
        }
