"""
EcodiaOS — Voxis Silence Engine

Determines when the organism should NOT speak.

Silence is a first-class decision, not a fallback. An organism that
talks constantly is not caring — it is needy. The Silence Engine
enforces the principle that every expression should serve a real purpose.

The engine tracks time since last expression (stateful) and evaluates
each candidate expression against a set of well-defined heuristics.
"""

from __future__ import annotations

from datetime import datetime

import structlog

from primitives.common import utc_now
from systems.voxis.types import (
    ExpressionTrigger,
    SilenceContext,
    SilenceDecision,
)

logger = structlog.get_logger()


class SilenceEngine:
    """
    Stateful silence evaluator.

    Tracks last expression time and applies trigger-specific heuristics
    to decide whether this expression should happen now, be queued,
    or be discarded entirely.
    """

    def __init__(self, min_expression_interval_minutes: float = 1.0) -> None:
        self._min_interval_minutes = min_expression_interval_minutes
        self._last_expression_time: datetime | None = None
        self._logger = logger.bind(system="voxis.silence")

    @property
    def minutes_since_last_expression(self) -> float:
        if self._last_expression_time is None:
            return 9999.0
        delta = utc_now() - self._last_expression_time
        return delta.total_seconds() / 60.0

    def record_expression(self) -> None:
        """Call this after every successful expression to update the timer."""
        self._last_expression_time = utc_now()

    def evaluate(self, context: SilenceContext) -> SilenceDecision:
        """
        Evaluate whether the organism should speak given this context.

        Decision hierarchy:
        1. Direct address → always speak
        2. Distress detection → always speak (Care drive override)
        3. Warnings → always speak
        4. Nova deliberate triggers → speak (these are intentional)
        5. Proactive/ambient triggers → check rate limits and value threshold
        """
        trigger = context.trigger
        elapsed = self.minutes_since_last_expression

        # ── Mandatory speech triggers ─────────────────────────────
        if trigger == ExpressionTrigger.ATUNE_DIRECT_ADDRESS:
            return SilenceDecision(
                speak=True,
                reason="Direct address — always respond",
            )

        if trigger == ExpressionTrigger.ATUNE_DISTRESS:
            return SilenceDecision(
                speak=True,
                reason="Distress detected — Care drive activated",
            )

        if trigger == ExpressionTrigger.NOVA_WARN:
            return SilenceDecision(
                speak=True,
                reason="Warning — urgency overrides silence heuristics",
            )

        # ── Deliberate Nova triggers ──────────────────────────────
        if trigger in (
            ExpressionTrigger.NOVA_RESPOND,
            ExpressionTrigger.NOVA_REQUEST,
            ExpressionTrigger.NOVA_MEDIATE,
            ExpressionTrigger.NOVA_CELEBRATE,
        ):
            # Always speak for deliberate Nova triggers, but respect active conversation
            return SilenceDecision(
                speak=True,
                reason=f"Deliberate Nova trigger: {trigger.value}",
            )

        if trigger == ExpressionTrigger.NOVA_INFORM:
            # Proactive inform — check rate limit and whether humans are conversing
            if context.humans_actively_conversing:
                return SilenceDecision(
                    speak=False,
                    reason="Humans in active conversation — queue for after",
                    queue=True,
                )
            if elapsed < context.min_expression_interval:
                return SilenceDecision(
                    speak=False,
                    reason=f"Rate limit: {elapsed:.1f}m < {context.min_expression_interval:.1f}m minimum",
                    queue=True,
                )
            return SilenceDecision(
                speak=True,
                reason="Proactive inform — conditions clear",
            )

        # ── Ambient / spontaneous triggers ────────────────────────
        if trigger == ExpressionTrigger.AMBIENT_INSIGHT:
            if context.humans_actively_conversing:
                return SilenceDecision(
                    speak=False,
                    reason="Humans in active conversation — insight queued",
                    queue=True,
                )
            if elapsed < context.min_expression_interval:
                return SilenceDecision(
                    speak=False,
                    reason=f"Insight rate-limited: {elapsed:.1f}m elapsed",
                    queue=False,  # Discard — ambient insights have short relevance windows
                )
            if context.insight_value < 0.6:
                return SilenceDecision(
                    speak=False,
                    reason=f"Insight value {context.insight_value:.2f} below expression threshold (0.60)",
                )
            return SilenceDecision(
                speak=True,
                reason=f"Ambient insight — value {context.insight_value:.2f} above threshold",
            )

        if trigger == ExpressionTrigger.AMBIENT_STATUS:
            # Status updates are low-priority — only if no recent expression
            if elapsed < max(5.0, context.min_expression_interval * 5):
                return SilenceDecision(
                    speak=False,
                    reason="Status update suppressed — recent expression exists",
                )
            return SilenceDecision(speak=True, reason="Periodic status update")

        # ── Default ───────────────────────────────────────────────
        return SilenceDecision(
            speak=False,
            reason=f"Unhandled trigger '{trigger.value}' — defaulting to silence",
        )
