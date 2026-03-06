"""
Atune utility functions.

Low-level helpers used across multiple Atune modules: vector math, text
analysis primitives, and hashing for provenance chains.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Vector math
# ---------------------------------------------------------------------------


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 on degenerate input."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to the [lo, hi] range."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Hashing (for provenance)
# ---------------------------------------------------------------------------


def hash_content(data: str | bytes) -> str:
    """SHA-256 hex digest of arbitrary content."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def compute_hash_chain(*items: str | bytes) -> str:
    """Chain-hash multiple items into a single integrity digest."""
    h = hashlib.sha256()
    for item in items:
        if isinstance(item, str):
            item = item.encode("utf-8")
        h.update(item)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Lightweight text analysis helpers
#
# These provide fast, heuristic-level signals. Heavy analysis (sentiment,
# entity extraction) goes through the LLM client.
# ---------------------------------------------------------------------------

# Common risk-signal patterns
_RISK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(urgent|emergency|critical|danger|warning|alert|crisis)\b", re.I),
    re.compile(r"\b(fail(ed|ure|ing)?|crash(ed|ing)?|down|outage|breach|broken)\b", re.I),
    re.compile(r"\b(threat|risk|hazard|vulnerab(le|ility))\b", re.I),
]

_DISTRESS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(help|please|struggling|scared|worried|anxious|overwhelmed)\b", re.I),
    re.compile(r"\b(hurt|pain|afraid|lonely|depressed|upset)\b", re.I),
]

_CONFLICT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(disagree|argument|conflict|dispute|angry|furious|unfair)\b", re.I),
    re.compile(r"\b(fight|hostile|blame|accuse|betray)\b", re.I),
]

_POSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(thank|grateful|celebrate|wonderful|amazing|excellent|joy)\b", re.I),
    re.compile(r"\b(happy|excited|proud|love|appreciate|congratulat)\b", re.I),
]

_URGENCY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(now|immediately|asap|right away|hurry|deadline)\b", re.I),
    re.compile(r"\b(time.sensitive|urgent(ly)?|critical(ly)?)\b", re.I),
]

_CAUSAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(because|therefore|consequently|leads?\s+to|causes?|results?\s+in)\b", re.I),
    re.compile(r"\b(due\s+to|as\s+a\s+result|hence|thus|so\s+that)\b", re.I),
]

_DIRECT_ADDRESS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(hey|hi|hello|yo)\b", re.I),
    re.compile(r"\?$"),  # Questions often imply addressing
]


def _pattern_score(text: str, patterns: list[re.Pattern[str]]) -> float:
    """Return a 0-1 score based on how many pattern groups match."""
    if not text:
        return 0.0
    matches = sum(1 for p in patterns if p.search(text))
    return clamp(matches / max(len(patterns), 1), 0.0, 1.0)


def detect_risk_patterns(text: str | None) -> float:
    """Heuristic risk signal in [0, 1]."""
    return _pattern_score(text or "", _RISK_PATTERNS)


def detect_distress(text: str | None) -> float:
    """Heuristic distress signal in [0, 1]."""
    return _pattern_score(text or "", _DISTRESS_PATTERNS)


def detect_conflict(text: str | None) -> float:
    """Heuristic conflict signal in [0, 1]."""
    return _pattern_score(text or "", _CONFLICT_PATTERNS)


def detect_positive_emotion(text: str | None) -> float:
    """Heuristic positive emotion signal in [0, 1]."""
    return _pattern_score(text or "", _POSITIVE_PATTERNS)


def detect_urgency(text: str | None) -> float:
    """Heuristic urgency signal in [0, 1]."""
    return _pattern_score(text or "", _URGENCY_PATTERNS)


def detect_causal_language(text: str | None) -> float:
    """Heuristic causal-language signal in [0, 1]."""
    return _pattern_score(text or "", _CAUSAL_PATTERNS)


def detect_direct_address(text: str | None) -> float:
    """Heuristic: is the text addressing EOS directly?"""
    return _pattern_score(text or "", _DIRECT_ADDRESS_PATTERNS)


def match_keyword_set(text: str | None, keywords: set[str]) -> float:
    """Fraction of *keywords* found in *text* (case-insensitive)."""
    if not text or not keywords:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return clamp(hits / max(len(keywords), 1), 0.0, 1.0)


def estimate_consequence_scope(text: str | None, community_size: int) -> float:
    """
    Very rough estimate of how many people a described event might affect.
    Returns 0-1 normalised by community size.
    """
    if not text:
        return 0.0
    scope_keywords = {
        "everyone": 1.0,
        "all members": 1.0,
        "community": 0.8,
        "team": 0.5,
        "group": 0.4,
        "several": 0.3,
        "some": 0.2,
        "a few": 0.1,
    }
    text_lower = text.lower()
    for phrase, score in scope_keywords.items():
        if phrase in text_lower:
            return score
    return 0.1  # default: individual-scale


def estimate_temporal_proximity(text: str | None) -> float:
    """How imminent are the consequences described? 1.0 = now, 0.0 = far future."""
    if not text:
        return 0.0
    immediate = re.compile(r"\b(now|today|immediately|tonight|this hour)\b", re.I)
    soon = re.compile(r"\b(tomorrow|this week|soon|shortly|within days)\b", re.I)
    later = re.compile(r"\b(next month|next year|eventually|long.term)\b", re.I)
    if immediate.search(text):
        return 0.9
    if soon.search(text):
        return 0.5
    if later.search(text):
        return 0.2
    return 0.3  # uncertain
