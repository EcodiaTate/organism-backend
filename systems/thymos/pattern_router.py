"""
EcodiaOS - Thymos Pattern-Aware Router

Intercepts every repair request before tier assignment to check whether the
incoming incident matches a known fatal CrashPattern stored in Redis.

If a match is found (match_score >= 0.7 AND pattern.confidence >= 0.6):
  1. Record which tiers have already failed for this pattern.
  2. Route ABOVE the highest failed tier.
  3. If ALL tiers up to NOVEL_FIX have failed → skip local repair entirely,
     return ESCALATE with federation_broadcast=True and the pattern_id attached.

The router is pure logic - it does not write to Redis or emit events.
Outcome recording (update_on_success / update_on_failure) is performed by
ThymosService after it receives the repair result.

Usage in ThymosService.process_incident():
    result = await self._pattern_router.route(incident)
    if result.federation_escalate:
        # emit INCIDENT_ESCALATED with federation_broadcast=True, pattern_id
        ...
        return
    # apply result.tier_override to diagnosis before prescribing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from core.crash_pattern_analyzer import CrashPattern, CrashPatternAnalyzer
from systems.thymos.types import Incident, RepairTier

if TYPE_CHECKING:
    from clients.redis import RedisClient

logger = structlog.get_logger("systems.thymos.pattern_router")

# Match thresholds - both conditions must be satisfied for a PATTERN_MATCH
_SCORE_THRESHOLD = 0.7    # fraction of pattern signature overlapping incident features
_CONFIDENCE_THRESHOLD = 0.6  # pattern.confidence must be >= this


@dataclass
class PatternRouteResult:
    """
    Decision returned by PatternAwareRouter.route().

    ``matched``           - True if a pattern with sufficient score + confidence matched.
    ``pattern_id``        - ID of the best-matching CrashPattern (None if no match).
    ``pattern_confidence`` - Confidence of the matched pattern (0 if no match).
    ``match_score``       - Feature overlap score (0.0–1.0).
    ``tier_override``     - Suggested RepairTier to use instead of normal routing.
                           None means no change - proceed with normal tier selection.
    ``tier_skip_reason``  - Human-readable reason for any tier skip.
    ``skipped_tiers``     - Tiers that were skipped because the pattern recorded
                           them as already failed.
    ``federation_escalate`` - True when all local tiers are exhausted for this
                              pattern - caller must broadcast to federation immediately.
    """

    matched: bool = False
    pattern_id: str | None = None
    pattern_confidence: float = 0.0
    match_score: float = 0.0
    tier_override: RepairTier | None = None
    tier_skip_reason: str | None = None
    skipped_tiers: list[str] = field(default_factory=list)
    federation_escalate: bool = False


class PatternAwareRouter:
    """
    Pattern-aware tier router for Thymos.

    Instantiated once in ThymosService.initialize() and queried on every
    THYMOS_REPAIR_REQUESTED path inside process_incident().

    The router loads all CrashPatterns from Redis on each call.  This is
    intentionally non-cached: patterns evolve as the organism heals and
    the Redis round-trip is cheap compared with the LLM diagnosis step.

    The caller (ThymosService) is responsible for:
      - Stamping result.pattern_id, pattern_confidence, tier_skip_reason
        onto the Incident before it enters the pipeline.
      - Calling CrashPatternAnalyzer.update_on_success/failure after outcome.
      - Emitting CRASH_PATTERN_RESOLVED / CRASH_PATTERN_REINFORCED.
    """

    # Repair tiers in ascending order (NOOP excluded - it means no action needed)
    _TIER_ORDER: list[RepairTier] = [
        RepairTier.PARAMETER,
        RepairTier.RESTART,
        RepairTier.KNOWN_FIX,
        RepairTier.NOVEL_FIX,
        RepairTier.ESCALATE,
    ]

    def __init__(self, analyzer: CrashPatternAnalyzer) -> None:
        self._analyzer = analyzer
        self._logger = logger.bind(component="pattern_aware_router")

    async def route(self, incident: Incident) -> PatternRouteResult:
        """
        Score the incident against all known CrashPatterns.

        Returns a PatternRouteResult with routing guidance.
        No side-effects (caller handles Redis writes and event emission).
        """
        # Extract features from the incident
        incident_features = CrashPatternAnalyzer.extract_features(
            source_system=incident.source_system,
            incident_class=incident.incident_class.value,
            error_type=incident.error_type,
            error_message=incident.error_message,
            affected_systems=incident.affected_systems,
        )

        # Load all known patterns (Redis round-trip)
        patterns = await self._analyzer.load_all_patterns()

        if not patterns:
            return PatternRouteResult()

        # Score each pattern and find the best match
        best_score = 0.0
        best_pattern: CrashPattern | None = None

        for pattern in patterns:
            score = self._analyzer.score_incident(incident_features, pattern)
            if score > best_score:
                best_score = score
                best_pattern = pattern

        if best_pattern is None or best_score < _SCORE_THRESHOLD:
            return PatternRouteResult()

        if best_pattern.confidence < _CONFIDENCE_THRESHOLD:
            self._logger.debug(
                "pattern_match_below_confidence_threshold",
                incident_id=incident.id,
                pattern_id=best_pattern.id,
                match_score=round(best_score, 3),
                confidence=round(best_pattern.confidence, 3),
                threshold=_CONFIDENCE_THRESHOLD,
            )
            return PatternRouteResult()

        # --- PATTERN_MATCH ---
        failed_tiers = set(best_pattern.failed_tiers)

        # Build the ordered list of tiers that have NOT already failed
        available_tiers = [
            t for t in self._TIER_ORDER if t.name not in failed_tiers
        ]

        self._logger.info(
            "crash_pattern_matched",
            incident_id=incident.id,
            pattern_id=best_pattern.id,
            match_score=round(best_score, 3),
            confidence=round(best_pattern.confidence, 3),
            failed_tiers=list(failed_tiers),
            available_tiers=[t.name for t in available_tiers],
        )

        skipped = list(failed_tiers)

        # If no local tiers remain → federation escalation
        local_tiers = [t for t in available_tiers if t != RepairTier.ESCALATE]
        if not local_tiers:
            return PatternRouteResult(
                matched=True,
                pattern_id=best_pattern.id,
                pattern_confidence=best_pattern.confidence,
                match_score=best_score,
                tier_override=RepairTier.ESCALATE,
                tier_skip_reason=(
                    f"All local tiers exhausted by CrashPattern {best_pattern.id}: "
                    f"{', '.join(sorted(failed_tiers))}"
                ),
                skipped_tiers=skipped,
                federation_escalate=True,
            )

        # Route to the lowest available tier that is ABOVE the highest failed tier
        # (i.e. skip everything that's already proven useless for this pattern)
        target_tier = local_tiers[0]

        # Compose skip reason
        if skipped:
            skip_reason = (
                f"CrashPattern {best_pattern.id} (confidence={best_pattern.confidence:.2f}): "
                f"skipping tiers [{', '.join(sorted(skipped))}] - already failed for this pattern"
            )
        else:
            # Pattern matched but no tiers failed yet - suggest the normal first tier
            # but stamp the pattern match so the pipeline knows about it
            skip_reason = (
                f"CrashPattern {best_pattern.id} matched "
                f"(score={best_score:.2f}, confidence={best_pattern.confidence:.2f})"
            )

        return PatternRouteResult(
            matched=True,
            pattern_id=best_pattern.id,
            pattern_confidence=best_pattern.confidence,
            match_score=best_score,
            tier_override=target_tier,
            tier_skip_reason=skip_reason,
            skipped_tiers=skipped,
            federation_escalate=False,
        )
