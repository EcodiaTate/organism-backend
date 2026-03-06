"""
EcodiaOS -- Voxis Reception Feedback Engine

Closes the expression feedback loop by correlating user responses with prior
expressions and estimating how well the expression was received.

## The Problem

Without reception tracking, the organism talks into the void. It generates
expressions, dispatches ExpressionFeedback, but never learns from the
actual response (or lack thereof). Evo has no signal to refine personality.
The feedback loop is open-ended.

## How It Works

1. **Expression tracking**: Each delivered expression is registered with
   the reception engine (expression ID, conversation ID, content summary,
   strategy metadata, timestamp).

2. **Response correlation**: When a user message arrives (via
   ``ingest_user_message``), the engine correlates it with the most recent
   pending expression in the same conversation.

3. **Reception estimation**: The engine estimates reception quality from
   observable signals (no LLM call -- fast, deterministic):
   - ``understood``: response length relative to expression, relevance indicators
   - ``emotional_impact``: sentiment shift signals (affect valence delta)
   - ``engagement``: response latency, response length, question-asking
   - ``satisfaction``: combined heuristic from above

4. **Feedback enrichment**: The populated ``ReceptionEstimate`` is patched
   onto the ``ExpressionFeedback`` and re-dispatched to registered listeners
   (including Evo for personality learning).

## Active Inference Grounding

Reception feedback is the **observation** that completes the perception-action
loop. Expression is action; reception is the sensory consequence. The
organism must observe the consequences of its actions to update its
generative model (via Evo). Without this, expression quality cannot improve.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field

import structlog

from systems.voxis.types import ExpressionFeedback, ReceptionEstimate

logger = structlog.get_logger()

# ─── Configuration ────────────────────────────────────────────────

_MAX_PENDING_EXPRESSIONS = 50     # Per-conversation max tracked expressions
_RESPONSE_WINDOW_SECONDS = 300.0  # 5 min -- max time to attribute a response
_ENGAGEMENT_LENGTH_THRESHOLD = 20 # Chars -- below this counts as minimal engagement

# Positive/negative sentiment markers (lightweight, no LLM needed)
_POSITIVE_MARKERS = re.compile(
    r"\b(thanks|thank you|great|helpful|makes sense|got it|appreciate|"
    r"perfect|exactly|wonderful|awesome|good point|interesting|I see|"
    r"understood|clear now|that helps|nice)\b",
    re.IGNORECASE,
)
_NEGATIVE_MARKERS = re.compile(
    r"\b(confused|don't understand|doesn't make sense|wrong|no that's not|"
    r"what do you mean|huh\??|unclear|that's not what I|I disagree|"
    r"not helpful|actually|you missed)\b",
    re.IGNORECASE,
)
_QUESTION_MARKERS = re.compile(r"\?")


@dataclass
class PendingExpression:
    """An expression awaiting user response for reception feedback."""

    expression_id: str
    conversation_id: str
    content_summary: str
    strategy_register: str
    personality_warmth: float
    affect_before_valence: float
    trigger: str
    delivered_at: float = field(default_factory=time.monotonic)

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.delivered_at

    @property
    def is_expired(self) -> bool:
        return self.age_seconds > _RESPONSE_WINDOW_SECONDS


class ReceptionEngine:
    """
    Tracks delivered expressions and correlates user responses to estimate
    reception quality.

    Usage::

        engine = ReceptionEngine()

        # After expression delivery:
        engine.track_expression(expression_id, conversation_id, ...)

        # When user message arrives:
        enriched = engine.correlate_response(
            conversation_id, response_text, response_affect_valence
        )
        if enriched:
            # Re-dispatch to Evo with populated ReceptionEstimate
            for cb in feedback_callbacks:
                cb(enriched)
    """

    def __init__(self) -> None:
        # conversation_id -> deque of pending expressions
        self._pending: dict[str, deque[PendingExpression]] = {}
        self._logger = logger.bind(system="voxis.reception")

        # Metrics
        self._total_tracked: int = 0
        self._total_correlated: int = 0
        self._total_expired: int = 0
        self._total_no_response: int = 0

    def track_expression(
        self,
        expression_id: str,
        conversation_id: str,
        content_summary: str,
        strategy_register: str = "neutral",
        personality_warmth: float = 0.0,
        affect_before_valence: float = 0.0,
        trigger: str = "",
    ) -> None:
        """Register a delivered expression for response tracking."""
        if conversation_id not in self._pending:
            self._pending[conversation_id] = deque(maxlen=_MAX_PENDING_EXPRESSIONS)

        self._pending[conversation_id].append(PendingExpression(
            expression_id=expression_id,
            conversation_id=conversation_id,
            content_summary=content_summary,
            strategy_register=strategy_register,
            personality_warmth=personality_warmth,
            affect_before_valence=affect_before_valence,
            trigger=trigger,
        ))
        self._total_tracked += 1

    def correlate_response(
        self,
        conversation_id: str,
        response_text: str,
        response_affect_valence: float | None = None,
    ) -> ExpressionFeedback | None:
        """
        Correlate a user response with the most recent pending expression.

        Returns an enriched ExpressionFeedback with populated ReceptionEstimate,
        or None if no pending expression matches.
        """
        # Prune expired
        self._prune_expired(conversation_id)

        pending_queue = self._pending.get(conversation_id)
        if not pending_queue:
            return None

        # Take the most recent pending expression (LIFO -- most relevant)
        pending = pending_queue.pop()

        # Estimate reception
        reception = self._estimate_reception(
            expression_summary=pending.content_summary,
            response_text=response_text,
            response_latency_seconds=pending.age_seconds,
            affect_before=pending.affect_before_valence,
            affect_after=response_affect_valence,
        )

        # Build enriched feedback
        affect_after = response_affect_valence if response_affect_valence is not None else pending.affect_before_valence
        feedback = ExpressionFeedback(
            expression_id=pending.expression_id,
            trigger=pending.trigger,
            conversation_id=conversation_id,
            content_summary=pending.content_summary,
            strategy_register=pending.strategy_register,
            personality_warmth=pending.personality_warmth,
            inferred_reception=reception,
            affect_before_valence=pending.affect_before_valence,
            affect_after_valence=affect_after,
            affect_delta=affect_after - pending.affect_before_valence,
            user_responded=True,
            user_response_length=len(response_text),
        )

        self._total_correlated += 1
        self._logger.debug(
            "response_correlated",
            expression_id=pending.expression_id[:8],
            latency_s=round(pending.age_seconds, 1),
            understood=round(reception.understood, 3),
            engagement=round(reception.engagement, 3),
            satisfaction=round(reception.satisfaction, 3),
        )

        return feedback

    def _estimate_reception(
        self,
        expression_summary: str,
        response_text: str,
        response_latency_seconds: float,
        affect_before: float,
        affect_after: float | None,
    ) -> ReceptionEstimate:
        """
        Estimate reception quality from observable response signals.

        No LLM call -- fast, deterministic heuristics:
        - Understood: positive markers, substantive response, no confusion markers
        - Emotional impact: valence delta (if available)
        - Engagement: response length, latency, question-asking
        - Satisfaction: composite of above
        """
        resp_len = len(response_text)
        response_text.lower()

        # ── Understood ──────────────────────────────────────────
        understood = 0.5  # Baseline: unknown

        positive_matches = len(_POSITIVE_MARKERS.findall(response_text))
        negative_matches = len(_NEGATIVE_MARKERS.findall(response_text))

        if positive_matches > 0 and negative_matches == 0:
            understood = min(1.0, 0.6 + positive_matches * 0.1)
        elif negative_matches > 0 and positive_matches == 0:
            understood = max(0.0, 0.4 - negative_matches * 0.15)
        elif positive_matches > negative_matches:
            understood = 0.6
        elif negative_matches > positive_matches:
            understood = 0.3

        # Substantive response length is a mild positive signal
        if resp_len > 100:
            understood = min(1.0, understood + 0.1)

        # ── Emotional impact ────────────────────────────────────
        emotional_impact = 0.0
        if affect_after is not None:
            delta = affect_after - affect_before
            # Positive shift = positive impact; negative = negative impact
            emotional_impact = max(-1.0, min(1.0, delta * 2.0))

        # ── Engagement ──────────────────────────────────────────
        engagement = 0.5  # Baseline

        # Response length signal
        if resp_len < _ENGAGEMENT_LENGTH_THRESHOLD:
            engagement -= 0.2
        elif resp_len > 200:
            engagement += 0.15
        elif resp_len > 50:
            engagement += 0.05

        # Latency signal: very fast response = high engagement
        if response_latency_seconds < 10:
            engagement += 0.15
        elif response_latency_seconds < 30:
            engagement += 0.05
        elif response_latency_seconds > 120:
            engagement -= 0.1

        # Question-asking = high engagement
        question_count = len(_QUESTION_MARKERS.findall(response_text))
        if question_count > 0:
            engagement += min(0.2, question_count * 0.1)

        engagement = max(0.0, min(1.0, engagement))

        # ── Satisfaction ────────────────────────────────────────
        # Weighted composite of above signals
        satisfaction = (
            understood * 0.40
            + max(0.0, min(1.0, 0.5 + emotional_impact * 0.5)) * 0.20
            + engagement * 0.40
        )
        satisfaction = max(0.0, min(1.0, satisfaction))

        return ReceptionEstimate(
            understood=round(understood, 3),
            emotional_impact=round(emotional_impact, 3),
            engagement=round(engagement, 3),
            satisfaction=round(satisfaction, 3),
        )

    def _prune_expired(self, conversation_id: str) -> None:
        """Remove expired pending expressions."""
        queue = self._pending.get(conversation_id)
        if not queue:
            return

        len(queue)
        while queue and queue[0].is_expired:
            queue.popleft()
            self._total_expired += 1

        if not queue:
            del self._pending[conversation_id]

    def expire_unanswered(self) -> list[ExpressionFeedback]:
        """
        Collect and return feedback for all expired (unanswered) expressions.

        Called periodically to ensure expressions that received no response
        still generate feedback (with user_responded=False).
        """
        expired_feedback: list[ExpressionFeedback] = []

        for conv_id in list(self._pending.keys()):
            queue = self._pending[conv_id]
            while queue and queue[0].is_expired:
                pending = queue.popleft()
                feedback = ExpressionFeedback(
                    expression_id=pending.expression_id,
                    trigger=pending.trigger,
                    conversation_id=conv_id,
                    content_summary=pending.content_summary,
                    strategy_register=pending.strategy_register,
                    personality_warmth=pending.personality_warmth,
                    affect_before_valence=pending.affect_before_valence,
                    affect_after_valence=pending.affect_before_valence,
                    affect_delta=0.0,
                    user_responded=False,
                    user_response_length=0,
                    inferred_reception=ReceptionEstimate(
                        understood=0.5,
                        emotional_impact=0.0,
                        engagement=0.2,  # No response = low engagement
                        satisfaction=0.3,
                    ),
                )
                expired_feedback.append(feedback)
                self._total_no_response += 1

            if not queue:
                del self._pending[conv_id]

        if expired_feedback:
            self._logger.info(
                "unanswered_expressions_expired",
                count=len(expired_feedback),
            )

        return expired_feedback

    def metrics(self) -> dict[str, int]:
        """Return engine metrics for health reporting."""
        total_pending = sum(len(q) for q in self._pending.values())
        return {
            "total_tracked": self._total_tracked,
            "total_correlated": self._total_correlated,
            "total_expired": self._total_expired,
            "total_no_response": self._total_no_response,
            "active_pending": total_pending,
            "active_conversations": len(self._pending),
        }
