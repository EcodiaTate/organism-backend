"""
EcodiaOS - Crash Pattern Analyzer

Maintains a Redis-backed library of CrashPatterns: known incident signatures
that have proven difficult or impossible to repair at lower tiers.

Each CrashPattern records:
  - A feature signature (frozenset of feature strings extracted from incidents)
  - Which repair tiers have been attempted and failed
  - A confidence score (0.0–1.0) reflecting how reliably this pattern is fatal
  - The highest tier that successfully resolved an incident matching this pattern

The analyzer is stateless between calls - all durable state lives in Redis under
the key prefix ``crash_pattern:``.

Integration points:
  - PatternAwareRouter (thymos/pattern_router.py): queries patterns before tier
    assignment, then calls update_on_success() / update_on_failure() after outcomes.
  - CRASH_PATTERN_RESOLVED: downstream signal consumed here to decay confidence.
  - CRASH_PATTERN_REINFORCED: downstream signal consumed here to raise confidence.

Redis key schema:
  ``crash_pattern:{pattern_id}``  → JSON-serialised CrashPattern

The pattern_id is a deterministic SHA-256 hex prefix (12 chars) of the sorted
signature features, so the same feature set always maps to the same key.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from pydantic import Field
from primitives.common import EOSBaseModel, new_id, utc_now

if TYPE_CHECKING:
    from clients.redis import RedisClient

logger = structlog.get_logger("core.crash_pattern_analyzer")

# ─── Prefix for all crash-pattern keys in Redis ──────────────────────────────

_KEY_PREFIX = "crash_pattern:"

# How much to nudge confidence up on a reinforcement (failure confirms the pattern)
_REINFORCE_DELTA = 0.08

# How much to nudge confidence down on a resolution (pattern is not always fatal)
_RESOLVE_DELTA = 0.12

# Minimum confidence a pattern can decay to (prevents complete erasure of history)
_MIN_CONFIDENCE = 0.10

# Maximum confidence (1.0 means always fatal - avoid division by zero in callers)
_MAX_CONFIDENCE = 0.98


# ─── Data Model ──────────────────────────────────────────────────────────────


class CrashPattern(EOSBaseModel):
    """
    A known fatal or near-fatal incident signature.

    ``signature`` is a sorted list of feature strings extracted from incident
    attributes (source_system, incident_class, error_type keywords, etc.).
    Overlap between an incoming incident's feature set and this signature
    drives the match_score used by PatternAwareRouter.

    ``failed_tiers`` records which RepairTier names have been attempted and
    failed for incidents matching this pattern.  The router uses this to skip
    those tiers and jump straight to a higher one.

    ``highest_resolved_tier`` is the RepairTier name at which a repair was
    last confirmed successful.  Stored so the router can route *to* that tier
    directly when it sees a match.
    """

    id: str                                     # Deterministic from signature hash
    signature: list[str]                        # Sorted feature strings (dedup'd)
    description: str = ""                       # Human-readable summary
    confidence: float = 0.5                     # P(pattern → fatal at low tiers)
    failed_tiers: list[str] = Field(default_factory=list)  # RepairTier names that proved useless
    highest_resolved_tier: str | None = None    # Tier that last succeeded for this pattern
    occurrence_count: int = 0                   # Total incidents matched
    resolution_count: int = 0                   # Times a repair succeeded
    first_seen: datetime = Field(default_factory=utc_now)
    last_seen: datetime = Field(default_factory=utc_now)

    @classmethod
    def make_id(cls, features: frozenset[str]) -> str:
        """Deterministic 12-char hex ID from a feature set."""
        canonical = "|".join(sorted(features))
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]


# ─── Analyzer ────────────────────────────────────────────────────────────────


class CrashPatternAnalyzer:
    """
    Redis-backed crash pattern library.

    Query flow (used by PatternAwareRouter):
      1. ``load_all_patterns()`` - fetch all crash_pattern:* keys.
      2. Score each pattern: ``match_score = |incident ∩ signature| / |signature|``.
      3. If best_match_score >= 0.7 and pattern.confidence >= 0.6 → PATTERN_MATCH.
      4. Determine skippable tiers from ``pattern.failed_tiers``.
      5. Route above the highest failed tier (or broadcast to federation if all exhausted).

    Update flow (called after repair outcome):
      - ``update_on_success(pattern_id, repair_tier)`` - decays confidence, records tier.
      - ``update_on_failure(pattern_id, repair_tier)`` - raises confidence, records tier.
    """

    def __init__(self, redis: RedisClient | None = None) -> None:
        self._redis = redis
        self._logger = logger.bind(component="crash_pattern_analyzer")

    def set_redis(self, redis: RedisClient) -> None:
        self._redis = redis

    # ── Feature extraction ────────────────────────────────────────────────────

    @staticmethod
    def extract_features(
        source_system: str,
        incident_class: str,
        error_type: str,
        error_message: str,
        affected_systems: list[str],
    ) -> frozenset[str]:
        """
        Extract a normalised feature set from incident attributes.

        Features are lowercase strings prefixed by dimension so
        ``source:nova`` and ``class:nova`` don't collide.
        """
        feats: set[str] = set()

        feats.add(f"source:{source_system.lower()}")
        feats.add(f"class:{incident_class.lower()}")

        if error_type:
            feats.add(f"etype:{error_type.lower()}")

        # Keyword extraction from error_message (top-4 tokens, ignore stopwords)
        _STOP = {"the", "a", "an", "is", "in", "at", "of", "to", "and", "or", "for"}
        tokens = error_message.lower().split()
        keywords = [t for t in tokens if len(t) > 3 and t not in _STOP][:4]
        for kw in keywords:
            feats.add(f"kw:{kw}")

        for sys in affected_systems:
            feats.add(f"affects:{sys.lower()}")

        return frozenset(feats)

    # ── Redis I/O ─────────────────────────────────────────────────────────────

    def _key(self, pattern_id: str) -> str:
        return f"{_KEY_PREFIX}{pattern_id}"

    async def _load_pattern(self, pattern_id: str) -> CrashPattern | None:
        if self._redis is None:
            return None
        try:
            raw = await self._redis.client.get(self._key(pattern_id))  # type: ignore[misc]
            if raw is None:
                return None
            return CrashPattern.model_validate_json(raw)
        except Exception as exc:
            self._logger.debug("crash_pattern_load_error", pattern_id=pattern_id, error=str(exc))
            return None

    async def _save_pattern(self, pattern: CrashPattern) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.client.set(  # type: ignore[misc]
                self._key(pattern.id),
                pattern.model_dump_json(),
            )
        except Exception as exc:
            self._logger.debug("crash_pattern_save_error", pattern_id=pattern.id, error=str(exc))

    async def load_all_patterns(self) -> list[CrashPattern]:
        """Fetch every crash_pattern:* key from Redis using SCAN (non-blocking cursor iteration)."""
        if self._redis is None:
            return []
        try:
            results: list[CrashPattern] = []
            cursor = 0
            match = f"{_KEY_PREFIX}*"
            while True:
                cursor, keys = await self._redis.client.scan(  # type: ignore[misc]
                    cursor, match=match, count=100
                )
                for key in keys:
                    raw = await self._redis.client.get(key)  # type: ignore[misc]
                    if raw:
                        try:
                            results.append(CrashPattern.model_validate_json(raw))
                        except Exception:
                            pass
                if cursor == 0:
                    break
            return results
        except Exception as exc:
            self._logger.debug("crash_pattern_load_all_error", error=str(exc))
            return []

    # ── Pattern registration (called by external tooling / CrashPatternAnalyzer subscribers) ──

    async def register_pattern(
        self,
        features: frozenset[str],
        description: str = "",
        initial_confidence: float = 0.5,
        failed_tiers: list[str] | None = None,
    ) -> CrashPattern:
        """
        Register or update a crash pattern from a feature set.

        If a pattern with the same signature already exists it is returned
        as-is (no overwrite); callers should use update_on_failure() to
        evolve existing patterns.
        """
        pattern_id = CrashPattern.make_id(features)
        existing = await self._load_pattern(pattern_id)
        if existing is not None:
            return existing

        pattern = CrashPattern(
            id=pattern_id,
            signature=sorted(features),
            description=description,
            confidence=max(_MIN_CONFIDENCE, min(_MAX_CONFIDENCE, initial_confidence)),
            failed_tiers=failed_tiers or [],
        )
        await self._save_pattern(pattern)
        self._logger.info(
            "crash_pattern_registered",
            pattern_id=pattern_id,
            feature_count=len(features),
            confidence=pattern.confidence,
        )
        return pattern

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_incident(
        self,
        incident_features: frozenset[str],
        pattern: CrashPattern,
    ) -> float:
        """
        Compute match_score = |incident ∩ signature| / |signature|.

        Returns 0.0 if pattern signature is empty.
        """
        if not pattern.signature:
            return 0.0
        sig_set = frozenset(pattern.signature)
        intersection = incident_features & sig_set
        return len(intersection) / len(sig_set)

    # ── Update on outcome ─────────────────────────────────────────────────────

    async def update_on_success(
        self,
        pattern_id: str,
        repair_tier: str,
    ) -> None:
        """
        A repair succeeded on a matched incident.

        Decays confidence (pattern not always fatal) and records the
        successful repair tier.
        """
        pattern = await self._load_pattern(pattern_id)
        if pattern is None:
            self._logger.debug("crash_pattern_not_found_on_success", pattern_id=pattern_id)
            return

        old_confidence = pattern.confidence
        pattern.confidence = max(_MIN_CONFIDENCE, pattern.confidence - _RESOLVE_DELTA)
        pattern.resolution_count += 1
        pattern.highest_resolved_tier = repair_tier
        pattern.last_seen = utc_now()

        await self._save_pattern(pattern)
        self._logger.info(
            "crash_pattern_confidence_decayed",
            pattern_id=pattern_id,
            repair_tier=repair_tier,
            confidence_before=round(old_confidence, 3),
            confidence_after=round(pattern.confidence, 3),
            resolution_count=pattern.resolution_count,
        )

    async def update_on_failure(
        self,
        pattern_id: str,
        repair_tier: str,
        failure_reason: str = "",
    ) -> None:
        """
        A repair failed on a matched incident.

        Raises confidence (pattern proven fatal at this tier) and records
        the failed tier so future incidents skip it.
        """
        pattern = await self._load_pattern(pattern_id)
        if pattern is None:
            self._logger.debug("crash_pattern_not_found_on_failure", pattern_id=pattern_id)
            return

        old_confidence = pattern.confidence
        pattern.confidence = min(_MAX_CONFIDENCE, pattern.confidence + _REINFORCE_DELTA)
        pattern.occurrence_count += 1

        if repair_tier not in pattern.failed_tiers:
            pattern.failed_tiers.append(repair_tier)

        pattern.last_seen = utc_now()

        await self._save_pattern(pattern)
        self._logger.info(
            "crash_pattern_confidence_raised",
            pattern_id=pattern_id,
            repair_tier=repair_tier,
            failure_reason=failure_reason[:120],
            confidence_before=round(old_confidence, 3),
            confidence_after=round(pattern.confidence, 3),
            failed_tiers=pattern.failed_tiers,
        )
