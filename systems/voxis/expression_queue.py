"""
EcodiaOS -- Voxis Expression Queue

Implements a priority queue for expressions that the SilenceEngine suppressed
with ``queue=True`` (as opposed to outright discard).

## Why this exists

The SilenceEngine legitimately suppresses proactive expressions when:
- Humans are actively conversing (don't interrupt)
- Rate limit hasn't elapsed yet (don't be needy)

But these suppressed expressions aren't worthless -- they represent genuine
communicative intent. An organism that silently discards its impulse to speak
loses contextual relevance: by the time the rate limit expires, the insight may
be stale. The queue preserves intent and delivers it when conditions clear,
unless the expression has decayed past its relevance window.

## Relevance Decay

Each queued expression carries:
- ``enqueued_at``: timestamp of suppression
- ``relevance_halflife_seconds``: how quickly the expression loses relevance
  (ambient insights decay fast; deliberate Nova informs decay slowly)
- ``initial_relevance``: the insight_value/urgency at queue time

Current relevance = initial_relevance * 2^(-elapsed / halflife)

Expressions below the ``delivery_threshold`` are pruned on each drain cycle.

## Capacity

The queue has a hard cap (default 20). When full, the lowest-relevance item
is evicted to make room.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from systems.voxis.types import ExpressionIntent, ExpressionTrigger

if TYPE_CHECKING:
    from primitives.affect import AffectState

logger = structlog.get_logger()


# ─── Relevance Halflife by Trigger ────────────────────────────────

_HALFLIFE_BY_TRIGGER: dict[ExpressionTrigger, float] = {
    # Ambient insights lose relevance fast -- the moment passes
    ExpressionTrigger.AMBIENT_INSIGHT: 120.0,       # 2 minutes
    ExpressionTrigger.AMBIENT_STATUS: 300.0,         # 5 minutes
    # Nova proactive informs are more durable -- deliberate intent
    ExpressionTrigger.NOVA_INFORM: 600.0,            # 10 minutes
    # Everything else: 5 minute default
}

_DEFAULT_HALFLIFE = 300.0


@dataclass(order=True)
class QueuedExpression:
    """
    A suppressed expression waiting for delivery conditions to clear.

    Ordered by negative relevance for min-heap (highest relevance = lowest priority value).
    """

    # Sort key (negative relevance so heapq gives us highest-relevance first)
    _sort_key: float = field(init=False, repr=False)

    # Expression data
    intent: ExpressionIntent = field(compare=False)
    affect_snapshot: AffectState = field(compare=False)

    # Decay parameters
    enqueued_at: float = field(compare=False, default_factory=time.monotonic)
    initial_relevance: float = field(compare=False, default=0.5)
    halflife_seconds: float = field(compare=False, default=_DEFAULT_HALFLIFE)

    def __post_init__(self) -> None:
        self._sort_key = -self.initial_relevance

    @property
    def current_relevance(self) -> float:
        """Compute time-decayed relevance using exponential decay."""
        elapsed = time.monotonic() - self.enqueued_at
        return self.initial_relevance * math.pow(2.0, -elapsed / self.halflife_seconds)

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.enqueued_at


class ExpressionQueue:
    """
    Priority queue of suppressed expressions with relevance decay.

    Thread-safe for single-writer (VoxisService) usage within asyncio.

    Usage::

        queue.enqueue(intent, affect, initial_relevance=0.7)
        ...
        deliverable = queue.drain(delivery_threshold=0.3, max_items=3)
        for item in deliverable:
            await voxis.express(item.intent.content_to_express, ...)
    """

    def __init__(
        self,
        max_size: int = 20,
        delivery_threshold: float = 0.3,
    ) -> None:
        self._heap: list[QueuedExpression] = []
        self._max_size = max_size
        self._delivery_threshold = delivery_threshold
        self._total_enqueued: int = 0
        self._total_delivered: int = 0
        self._total_expired: int = 0
        self._total_evicted: int = 0
        self._logger = logger.bind(system="voxis.queue")

    @property
    def size(self) -> int:
        return len(self._heap)

    def enqueue(
        self,
        intent: ExpressionIntent,
        affect: AffectState,
        initial_relevance: float | None = None,
    ) -> None:
        """
        Add a suppressed expression to the queue.

        Uses insight_value/urgency from the intent as initial relevance if not
        explicitly provided.
        """
        relevance = initial_relevance or max(intent.insight_value, intent.urgency)
        halflife = _HALFLIFE_BY_TRIGGER.get(intent.trigger, _DEFAULT_HALFLIFE)

        item = QueuedExpression(
            intent=intent,
            affect_snapshot=affect,
            initial_relevance=relevance,
            halflife_seconds=halflife,
        )

        # Evict lowest-relevance if at capacity
        if len(self._heap) >= self._max_size:
            self._prune_expired()
            if len(self._heap) >= self._max_size:
                # Still full after pruning -- evict lowest
                worst = max(self._heap, key=lambda x: x._sort_key)  # Highest _sort_key = lowest relevance
                self._heap.remove(worst)
                heapq.heapify(self._heap)
                self._total_evicted += 1
                self._logger.debug(
                    "expression_evicted",
                    trigger=worst.intent.trigger.value,
                    relevance=round(worst.current_relevance, 3),
                )

        heapq.heappush(self._heap, item)
        self._total_enqueued += 1
        self._logger.debug(
            "expression_queued",
            trigger=intent.trigger.value,
            relevance=round(relevance, 3),
            halflife=halflife,
            queue_size=len(self._heap),
        )

    def drain(
        self,
        max_items: int = 3,
        delivery_threshold: float | None = None,
    ) -> list[QueuedExpression]:
        """
        Remove and return the highest-relevance expressions that are still
        above the delivery threshold.

        Called by VoxisService when conditions clear (e.g., humans stop
        conversing, rate limit window passes).
        """
        threshold = delivery_threshold or self._delivery_threshold
        self._prune_expired()

        deliverable: list[QueuedExpression] = []
        remaining: list[QueuedExpression] = []

        while self._heap and len(deliverable) < max_items:
            item = heapq.heappop(self._heap)
            if item.current_relevance >= threshold:
                deliverable.append(item)
            else:
                self._total_expired += 1

        # Put back anything we didn't deliver
        for item in self._heap:
            remaining.append(item)
        self._heap = remaining
        heapq.heapify(self._heap)

        self._total_delivered += len(deliverable)

        if deliverable:
            self._logger.info(
                "expressions_drained",
                count=len(deliverable),
                triggers=[d.intent.trigger.value for d in deliverable],
                relevances=[round(d.current_relevance, 3) for d in deliverable],
            )

        return deliverable

    def peek_highest_relevance(self) -> float:
        """Return the current relevance of the highest-priority queued item."""
        if not self._heap:
            return 0.0
        return self._heap[0].current_relevance

    def _prune_expired(self) -> None:
        """Remove items below the delivery threshold."""
        before = len(self._heap)
        self._heap = [
            item for item in self._heap
            if item.current_relevance >= self._delivery_threshold
        ]
        pruned = before - len(self._heap)
        if pruned > 0:
            heapq.heapify(self._heap)
            self._total_expired += pruned

    def clear(self) -> None:
        """Discard all queued expressions."""
        self._heap.clear()

    def metrics(self) -> dict[str, int | float]:
        """Return queue metrics for health reporting."""
        return {
            "queue_size": len(self._heap),
            "total_enqueued": self._total_enqueued,
            "total_delivered": self._total_delivered,
            "total_expired": self._total_expired,
            "total_evicted": self._total_evicted,
            "highest_relevance": round(self.peek_highest_relevance(), 3),
        }
