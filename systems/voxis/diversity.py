"""
EcodiaOS -- Voxis Expression Diversity Tracker

Prevents repetitive expression by tracking recent output patterns and scoring
new expressions against a sliding window of past content.

## Problem

Without diversity tracking, the organism can fall into ruts:
- Repeating the same opener ("I noticed that...")
- Using identical sentence structures across consecutive expressions
- Cycling through the same vocabulary when personality dimensions are strong
- Producing nearly identical ambient insights

## Approach

Three complementary mechanisms:

1. **Structural fingerprinting** -- n-gram overlap detection between the current
   expression and recent history. High overlap triggers a diversity penalty.

2. **Opener tracking** -- tracks the first N words of each expression. Flags
   when consecutive expressions start the same way.

3. **Semantic similarity** -- lightweight embedding-free approach using
   Jaccard similarity on content word sets. Catches paraphrased repetition.

The tracker returns a ``DiversityScore`` with per-mechanism scores and an
overall ``diversity`` float (0.0 = identical to recent, 1.0 = fully novel).
When diversity falls below a configurable threshold, the renderer can inject
a diversity instruction into the system prompt.

## Active Inference Grounding

In AIF terms, repetitive expression is a failure of epistemic foraging --
the organism is stuck in a local minimum of its generative model, producing
the same action (expression) and therefore not reducing surprise. The
diversity tracker acts as a prediction error signal: "you said something
too similar to what you already said" increases expected free energy for
the repeated policy, pushing the system toward novel expression.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()

# ─── Configuration ────────────────────────────────────────────────

_DEFAULT_WINDOW_SIZE = 20         # Track last N expressions
_DEFAULT_NGRAM_SIZE = 3           # Trigrams for structural fingerprinting
_DEFAULT_OPENER_WORDS = 5         # First N words as opener signature
_DEFAULT_DIVERSITY_THRESHOLD = 0.4  # Below this, inject diversity instruction

# Stopwords to exclude from semantic similarity (common function words)
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "what", "which", "who", "whom", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "into", "through", "during", "before",
    "after", "above", "below", "between", "but", "and", "or", "not", "no",
    "if", "then", "so", "too", "very", "just", "about", "up", "out", "also",
})


@dataclass
class DiversityScore:
    """Score breakdown for a candidate expression's novelty."""

    # Per-mechanism scores (0.0 = identical, 1.0 = fully novel)
    ngram_diversity: float = 1.0       # Structural: low n-gram overlap
    opener_diversity: float = 1.0      # Opener: not starting the same way
    semantic_diversity: float = 1.0    # Content word overlap

    # Composite score
    diversity: float = 1.0

    # Which recent expression was most similar (for diagnostics)
    most_similar_index: int = -1
    most_similar_overlap: float = 0.0

    @property
    def is_repetitive(self) -> bool:
        return self.diversity < _DEFAULT_DIVERSITY_THRESHOLD


@dataclass
class _ExpressionFingerprint:
    """Compact representation of a past expression for comparison."""

    content_words: frozenset[str]         # Content words (no stopwords)
    ngrams: Counter[tuple[str, ...]]      # N-gram frequency counter
    opener: tuple[str, ...]               # First N words
    raw_length: int = 0
    trigger: str = ""


def _tokenise(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser."""
    return re.findall(r"[a-z']+", text.lower())


def _extract_content_words(tokens: list[str]) -> frozenset[str]:
    """Remove stopwords, keep content-bearing tokens."""
    return frozenset(t for t in tokens if t not in _STOPWORDS and len(t) > 2)


def _extract_ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    """Extract n-gram frequency counter."""
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity between two word sets."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def _ngram_overlap(a: Counter[tuple[str, ...]], b: Counter[tuple[str, ...]]) -> float:
    """Overlap coefficient between two n-gram counters."""
    if not a or not b:
        return 0.0
    shared = sum((a & b).values())
    smaller = min(sum(a.values()), sum(b.values()))
    return shared / smaller if smaller > 0 else 0.0


class DiversityTracker:
    """
    Tracks recent expression fingerprints and scores new expressions for novelty.

    Usage::

        tracker = DiversityTracker()
        score = tracker.score("I noticed the garden is blooming today.")
        if score.is_repetitive:
            # Inject diversity instruction into prompt
            ...
        tracker.record("I noticed the garden is blooming today.", trigger="ambient_insight")
    """

    def __init__(
        self,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        ngram_size: int = _DEFAULT_NGRAM_SIZE,
        opener_words: int = _DEFAULT_OPENER_WORDS,
        diversity_threshold: float = _DEFAULT_DIVERSITY_THRESHOLD,
    ) -> None:
        self._window: deque[_ExpressionFingerprint] = deque(maxlen=window_size)
        self._ngram_size = ngram_size
        self._opener_words = opener_words
        self._threshold = diversity_threshold
        self._logger = logger.bind(system="voxis.diversity")

        # Counters
        self._total_scored: int = 0
        self._total_flagged: int = 0

    def score(self, text: str) -> DiversityScore:
        """
        Score a candidate expression against the recent history window.

        Returns a DiversityScore. Does NOT record the expression.
        Call ``record()`` after the expression is delivered.
        """
        if not self._window:
            return DiversityScore()  # First expression is always maximally diverse

        tokens = _tokenise(text)
        content_words = _extract_content_words(tokens)
        ngrams = _extract_ngrams(tokens, self._ngram_size)
        opener = tuple(tokens[: self._opener_words])

        best_ngram_overlap = 0.0
        best_semantic_overlap = 0.0
        opener_match_count = 0
        most_similar_idx = -1
        most_similar_overlap = 0.0

        for idx, fp in enumerate(self._window):
            # N-gram structural overlap
            ng_overlap = _ngram_overlap(ngrams, fp.ngrams)
            if ng_overlap > best_ngram_overlap:
                best_ngram_overlap = ng_overlap

            # Semantic (content word) similarity
            sem_sim = _jaccard_similarity(content_words, fp.content_words)
            if sem_sim > best_semantic_overlap:
                best_semantic_overlap = sem_sim
                most_similar_idx = idx
                most_similar_overlap = sem_sim

            # Opener match
            if opener and fp.opener and opener == fp.opener:
                opener_match_count += 1

        # Convert overlaps to diversity scores (1.0 - overlap)
        ngram_diversity = max(0.0, 1.0 - best_ngram_overlap)
        semantic_diversity = max(0.0, 1.0 - best_semantic_overlap)

        # Opener diversity: penalise more if consecutive recent expressions share opener
        recent_openers = [fp.opener for fp in list(self._window)[-5:]]
        consecutive_opener_matches = sum(1 for o in recent_openers if o == opener)
        opener_diversity = max(0.0, 1.0 - (consecutive_opener_matches * 0.35))

        # Composite: weighted geometric mean (penalises any single bad dimension)
        composite = (
            ngram_diversity ** 0.35
            * semantic_diversity ** 0.40
            * opener_diversity ** 0.25
        )

        self._total_scored += 1
        if composite < self._threshold:
            self._total_flagged += 1

        result = DiversityScore(
            ngram_diversity=round(ngram_diversity, 3),
            opener_diversity=round(opener_diversity, 3),
            semantic_diversity=round(semantic_diversity, 3),
            diversity=round(composite, 3),
            most_similar_index=most_similar_idx,
            most_similar_overlap=round(most_similar_overlap, 3),
        )

        if result.is_repetitive:
            self._logger.info(
                "expression_flagged_repetitive",
                diversity=result.diversity,
                ngram=result.ngram_diversity,
                semantic=result.semantic_diversity,
                opener=result.opener_diversity,
            )

        return result

    def record(self, text: str, trigger: str = "") -> None:
        """Record a delivered expression into the history window."""
        tokens = _tokenise(text)
        fp = _ExpressionFingerprint(
            content_words=_extract_content_words(tokens),
            ngrams=_extract_ngrams(tokens, self._ngram_size),
            opener=tuple(tokens[: self._opener_words]),
            raw_length=len(text),
            trigger=trigger,
        )
        self._window.append(fp)

    def build_diversity_instruction(self, score: DiversityScore) -> str:
        """
        Build a natural-language instruction for the LLM when diversity is low.

        Injected into the system prompt to push the LLM toward novel expression.
        """
        parts: list[str] = [
            "IMPORTANT -- your recent expressions have been repetitive. Vary your approach:"
        ]

        if score.opener_diversity < 0.5:
            parts.append("- Start differently. Do NOT use the same opening words as your recent messages.")

        if score.ngram_diversity < 0.5:
            parts.append("- Use different sentence structures and phrasings than your recent messages.")

        if score.semantic_diversity < 0.5:
            parts.append("- Approach from a different angle. Cover new ground rather than restating.")

        parts.append("- Be genuinely fresh. Surprise yourself.")

        return "\n".join(parts)

    def metrics(self) -> dict[str, int | float]:
        """Return tracker metrics for health reporting."""
        return {
            "window_size": len(self._window),
            "total_scored": self._total_scored,
            "total_flagged": self._total_flagged,
            "flag_rate": round(
                self._total_flagged / max(1, self._total_scored), 4
            ),
        }
