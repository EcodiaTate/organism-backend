"""
EcodiaOS - Equor Constitutional Memory

A rolling window of past constitutional decisions that Equor consults before
reaching a verdict on novel intents.

Purpose
-------
Novel intents that bypass template matching fall through to the verdict
pipeline with no prior context.  Constitutional Memory provides that context:

  1. **Decision lookup** - before finalising APPROVED/DEFERRED on a novel
     intent, check whether semantically similar past intents were BLOCKED or
     DEFERRED.  If they were, downgrade the current verdict or add a warning.

  2. **Decision recording** - after every review, persist the intent signature,
     verdict, confidence, and reasoning so future reviews can learn from it.

Architecture
------------
- In-memory ring buffer (configurable max_size, default 500).
- Each entry is a ``MemoryEntry`` keyed by a signature hash derived from the
  intent's goal, action executors, and autonomy level.
- Similarity is measured by token-set overlap on the goal description.
- No I/O - the memory lives on EquorService and is passed into compute_verdict()
  as a plain list[MemoryEntry].  EquorService flushes to Neo4j asynchronously.

Thread safety
-------------
Single-writer (EquorService) via ``record()``.  Reads by ``find_similar()``
are lock-free dict scans - safe because Python's GIL protects dict iteration.
"""

from __future__ import annotations

import hashlib
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now

if TYPE_CHECKING:

    from primitives.intent import Intent
logger = structlog.get_logger()

# Maximum decisions retained in the rolling window.
_DEFAULT_MAX_SIZE = 500

# Minimum token-overlap ratio (Jaccard) to consider two intents "similar".
_SIMILARITY_THRESHOLD = 0.25

# How many similar past decisions we return to the verdict engine.
_MAX_SIMILAR_RESULTS = 5

# Minimum confidence a past decision must have to be returned.
_MIN_PAST_CONFIDENCE = 0.7


@dataclass(frozen=True)
class MemoryEntry:
    """A single recorded constitutional decision."""

    intent_id: str
    goal_signature: str           # SHA-256 of normalised goal tokens (first 16 hex)
    goal_tokens: frozenset[str]   # For Jaccard similarity comparison
    goal_summary: str             # First 120 chars of goal description
    verdict: str                  # "approved" | "blocked" | "deferred" | "modified"
    confidence: float
    reasoning: str
    composite_alignment: float
    autonomy_level: int
    recorded_at: datetime = field(default_factory=utc_now)


def _goal_tokens(intent: Intent) -> frozenset[str]:
    """Return a frozenset of lower-case unigrams and bigrams from the goal."""
    words = intent.goal.description.lower().split()
    tokens: set[str] = set(words)
    for i in range(len(words) - 1):
        tokens.add(f"{words[i]} {words[i + 1]}")
    return frozenset(tokens)


def _goal_signature(tokens: frozenset[str]) -> str:
    """16-char SHA-256 hex of sorted tokens - used as a fast equality key."""
    raw = " ".join(sorted(tokens))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity between two token sets. Returns 0.0 if both empty."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


class ConstitutionalMemory:
    """
    Rolling window of constitutional decisions.

    EquorService creates one instance and passes it to compute_verdict()
    on every review.  After the verdict is computed, EquorService calls
    ``record()`` to add the new decision to the window.
    """

    def __init__(self, max_size: int = _DEFAULT_MAX_SIZE) -> None:
        self._max_size = max_size
        # Ordered by insertion time; oldest at left.
        self._window: deque[MemoryEntry] = deque(maxlen=max_size)
        # Fast-path equality index: signature → entry (last seen wins)
        self._index: dict[str, MemoryEntry] = {}
        self._logger = logger.bind(system="equor.constitutional_memory")

        # Metrics
        self._total_recorded: int = 0
        self._total_lookups: int = 0
        self._total_hits: int = 0

    # ─── Write path ──────────────────────────────────────────────────

    def record(
        self,
        intent: Intent,
        verdict: str,
        confidence: float,
        reasoning: str,
        composite_alignment: float,
        autonomy_level: int,
    ) -> None:
        """
        Record a constitutional decision for future lookups.

        Called by EquorService._post_review_bookkeeping() after every review.
        """
        tokens = _goal_tokens(intent)
        sig = _goal_signature(tokens)

        entry = MemoryEntry(
            intent_id=intent.id,
            goal_signature=sig,
            goal_tokens=tokens,
            goal_summary=intent.goal.description[:120],
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning[:300],
            composite_alignment=composite_alignment,
            autonomy_level=autonomy_level,
        )

        self._window.append(entry)
        self._index[sig] = entry
        self._total_recorded += 1

        # Keep index in sync with the window (evict signatures whose entry
        # is no longer in the window after maxlen eviction).
        if len(self._window) == self._max_size:
            oldest = self._window[0]
            if self._index.get(oldest.goal_signature) is oldest:
                del self._index[oldest.goal_signature]

        self._logger.debug(
            "constitutional_decision_recorded",
            intent_id=intent.id,
            verdict=verdict,
            confidence=f"{confidence:.2f}",
            total=self._total_recorded,
        )

    # ─── Read path ───────────────────────────────────────────────────

    def find_similar(
        self,
        intent: Intent,
        *,
        min_similarity: float = _SIMILARITY_THRESHOLD,
        min_confidence: float = _MIN_PAST_CONFIDENCE,
        max_results: int = _MAX_SIMILAR_RESULTS,
    ) -> list[MemoryEntry]:
        """
        Return up to *max_results* past decisions whose goal is similar to
        the candidate intent's goal.

        Sorted descending by Jaccard similarity.  Only entries with
        confidence >= *min_confidence* are returned.

        Called from compute_verdict() before Stage 7 (APPROVED).
        """
        self._total_lookups += 1

        tokens = _goal_tokens(intent)
        sig = _goal_signature(tokens)

        # Fast exact match first
        if sig in self._index:
            exact = self._index[sig]
            if exact.confidence >= min_confidence:
                self._total_hits += 1
                self._logger.debug(
                    "constitutional_memory_exact_match",
                    intent_id=intent.id,
                    past_verdict=exact.verdict,
                )
                return [exact]

        # Approximate scan
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in self._window:
            if entry.confidence < min_confidence:
                continue
            sim = _jaccard(tokens, entry.goal_tokens)
            if sim >= min_similarity:
                scored.append((sim, entry))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [e for _, e in scored[:max_results]]

        self._total_hits += 1
        self._logger.debug(
            "constitutional_memory_similar_found",
            intent_id=intent.id,
            count=len(results),
            top_similarity=f"{scored[0][0]:.2f}",
            top_past_verdict=results[0].verdict,
        )
        return results

    # ─── Summarise past decisions for the verdict engine ─────────────

    def prior_verdict_signal(
        self,
        intent: Intent,
    ) -> dict[str, Any]:
        """
        High-level summary of relevant past decisions for compute_verdict().

        Returns a dict with:
          - similar_count: int
          - block_rate: float (0.0–1.0)
          - defer_rate: float
          - avg_alignment: float
          - most_recent_verdict: str | None
          - warning: str | None  - set when past block_rate > 0.5
        """
        similar = self.find_similar(intent)

        if not similar:
            return {
                "similar_count": 0,
                "block_rate": 0.0,
                "defer_rate": 0.0,
                "avg_alignment": 0.0,
                "most_recent_verdict": None,
                "warning": None,
            }

        block_count = sum(1 for e in similar if e.verdict == "blocked")
        defer_count = sum(1 for e in similar if e.verdict == "deferred")
        avg_alignment = sum(e.composite_alignment for e in similar) / len(similar)
        block_rate = block_count / len(similar)
        defer_rate = defer_count / len(similar)

        warning: str | None = None
        if block_rate > 0.5:
            warning = (
                f"Constitutional memory: {block_count}/{len(similar)} similar "
                f"past intents were BLOCKED "
                f"(avg_alignment={avg_alignment:.2f}). "
                f"Pattern may be constitutionally problematic."
            )
        elif defer_rate > 0.5:
            warning = (
                f"Constitutional memory: {defer_count}/{len(similar)} similar "
                f"past intents were DEFERRED for governance review."
            )

        return {
            "similar_count": len(similar),
            "block_rate": block_rate,
            "defer_rate": defer_rate,
            "avg_alignment": avg_alignment,
            "most_recent_verdict": similar[0].verdict,
            "warning": warning,
        }

    # ─── Introspection ────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._window)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "window_size": self.size,
            "max_size": self._max_size,
            "total_recorded": self._total_recorded,
            "total_lookups": self._total_lookups,
            "total_hits": self._total_hits,
            "hit_rate": self._total_hits / max(1, self._total_lookups),
        }
