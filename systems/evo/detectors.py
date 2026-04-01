"""
EcodiaOS - Evo Pattern Detectors

Online pattern detectors that scan the continuous stream of episodes
during wake mode. Each detector looks for one class of regularity.

Detectors are stateless - all accumulation lives in PatternContext.
They run in ≤20ms per episode (latency budget from spec Section X).

Four detectors (spec Section III):
  CooccurrenceDetector  - entities that appear together repeatedly
  SequenceDetector      - action sequences that produce successful outcomes
  TemporalDetector      - time-based patterns (hour-of-day, day-of-week)
  AffectPatternDetector - stimuli that reliably shift emotional state
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from itertools import combinations
from typing import TYPE_CHECKING

import structlog

from systems.evo.types import (
    PatternCandidate,
    PatternContext,
    PatternType,
    hash_sequence,
)

if TYPE_CHECKING:
    from primitives.memory_trace import Episode

logger = structlog.get_logger()


# ─── Abstract Base ────────────────────────────────────────────────────────────


class PatternDetector(ABC):
    """
    Base class for all online pattern detectors.
    Detectors are stateless; PatternContext carries accumulated state.
    """

    name: str = ""
    window_size: int = 500      # Max episodes to keep in sliding window
    min_occurrences: int = 5    # Threshold to emit a candidate

    @abstractmethod
    async def scan(
        self,
        episode: Episode,
        context: PatternContext,
    ) -> list[PatternCandidate]:
        """
        Scan one episode and update context.
        Returns newly-triggered candidates (those crossing min_occurrences now).
        Must complete in ≤20ms.
        """
        ...


# ─── Co-occurrence Detector ───────────────────────────────────────────────────


class CooccurrenceDetector(PatternDetector):
    """
    Detects entities that frequently appear together.

    Reads entity IDs from context.recent_entity_ids, which the EvoService
    populates from each WorkspaceBroadcast's memory context (MemoryTrace.entities).

    On each episode, updates the co-occurrence matrix. Emits a PatternCandidate
    the first time a pair crosses min_occurrences.
    """

    name = "cooccurrence"
    window_size = 500
    min_occurrences = 5

    async def scan(
        self,
        episode: Episode,
        context: PatternContext,
    ) -> list[PatternCandidate]:
        entity_ids = context.recent_entity_ids
        if len(entity_ids) < 2:
            return []

        candidates: list[PatternCandidate] = []
        for pair in combinations(sorted(entity_ids), 2):
            key = f"{pair[0]}::{pair[1]}"
            context.cooccurrence_counts[key] += 1
            count = context.cooccurrence_counts[key]
            # Emit exactly at threshold (not repeatedly)
            if count == self.min_occurrences:
                candidates.append(
                    PatternCandidate(
                        type=PatternType.COOCCURRENCE,
                        elements=list(pair),
                        count=count,
                        confidence=0.5,
                        examples=[episode.id],
                    )
                )

        return candidates


# ─── Sequence Detector ────────────────────────────────────────────────────────


class SequenceDetector(PatternDetector):
    """
    Detects recurring action sequences that lead to successful outcomes.

    Reads action sequences from episode metadata (stored by Axon audit).
    Only processes action_outcome episodes that recorded success=True.
    Emits a candidate when a sequence hash crosses min_occurrences.
    """

    name = "action_sequence"
    window_size = 1000
    min_occurrences = 3

    async def scan(
        self,
        episode: Episode,
        context: PatternContext,
    ) -> list[PatternCandidate]:
        if not _is_successful_action_outcome(episode):
            return []

        sequence = _extract_action_sequence(episode)
        if not sequence:
            return []

        seq_hash = hash_sequence(sequence)
        context.sequence_counts[seq_hash] += 1
        context.sequence_examples[seq_hash].append(episode.id)

        count = context.sequence_counts[seq_hash]
        candidates: list[PatternCandidate] = []
        # Emit at threshold and then at power-of-2 milestones (3, 4, 8, 16, ...)
        # to avoid flooding _pending_candidates every broadcast.
        if count == self.min_occurrences or (count > self.min_occurrences and (count & (count - 1)) == 0):
            candidates.append(
                PatternCandidate(
                    type=PatternType.ACTION_SEQUENCE,
                    elements=sequence,
                    count=count,
                    confidence=min(0.9, 0.5 + count * 0.05),
                    examples=context.sequence_examples[seq_hash][:10],
                    metadata={"sequence_hash": seq_hash},
                )
            )

        return candidates


# ─── Temporal Detector ────────────────────────────────────────────────────────


class TemporalDetector(PatternDetector):
    """
    Detects time-based patterns: hour-of-day and day-of-week concentrations.

    Uses the episode's source channel as the "event type" and bins by time.
    Emits a candidate when a bin exceeds 2x the source-type baseline.
    """

    name = "temporal"
    window_size = 2000
    min_occurrences = 3

    async def scan(
        self,
        episode: Episode,
        context: PatternContext,
    ) -> list[PatternCandidate]:
        hour = episode.event_time.hour
        day = episode.event_time.weekday()

        # Use source channel as the event type for binning
        source_type = _classify_source(episode.source)

        hour_key = f"{source_type}::h{hour}"
        day_key = f"{source_type}::d{day}"

        context.temporal_bins[hour_key] += 1
        context.temporal_bins[day_key] += 1

        # Update baseline (exponential moving average)
        baseline = context.temporal_baselines.get(source_type, 1.0)
        context.temporal_baselines[source_type] = baseline * 0.99 + 1.0 * 0.01

        candidates: list[PatternCandidate] = []
        for key, count in [
            (hour_key, context.temporal_bins[hour_key]),
            (day_key, context.temporal_bins[day_key]),
        ]:
            # Emit at threshold and power-of-2 milestones to avoid flooding candidates.
            should_emit = count == self.min_occurrences or (count > self.min_occurrences and (count & (count - 1)) == 0)
            if should_emit:
                adjusted_baseline = max(1.0, context.temporal_baselines.get(source_type, 1.0))
                if count / adjusted_baseline > 2.0:
                    bin_label = key.split("::")[-1]
                    candidates.append(
                        PatternCandidate(
                            type=PatternType.TEMPORAL,
                            elements=[source_type, bin_label],
                            count=count,
                            confidence=min(0.8, count / (adjusted_baseline * 5)),
                            metadata={"key": key, "baseline": adjusted_baseline},
                        )
                    )

        return candidates


# ─── Affect Pattern Detector ──────────────────────────────────────────────────


class AffectPatternDetector(PatternDetector):
    """
    Detects stimuli that reliably shift emotional state.

    Compares context.previous_affect (from the prior broadcast) with
    context.current_affect (the broadcast that produced this episode).
    Classifies the stimulus type from the episode source and tracks
    the mean affect delta across occurrences.

    Emits when a stimulus_type has caused a consistent shift (≥min_occurrences).
    """

    name = "affect_pattern"
    window_size = 500
    min_occurrences = 5
    min_magnitude: float = 0.1  # Minimum shift to care about

    async def scan(
        self,
        episode: Episode,
        context: PatternContext,
    ) -> list[PatternCandidate]:
        # We need before and after affect states
        affect_before = context.previous_affect
        affect_after = context.current_affect

        if affect_before is None or affect_after is None:
            return []

        delta_valence = affect_after.valence - affect_before.valence
        delta_arousal = affect_after.arousal - affect_before.arousal

        # Only track meaningful shifts
        if abs(delta_valence) < self.min_magnitude and abs(delta_arousal) < self.min_magnitude:
            return []

        stimulus_type = _classify_source(episode.source)
        context.affect_responses[stimulus_type].append((delta_valence, delta_arousal))

        count = len(context.affect_responses[stimulus_type])
        candidates: list[PatternCandidate] = []

        # Emit at threshold and power-of-2 milestones to avoid flooding candidates.
        should_emit = count == self.min_occurrences or (count > self.min_occurrences and (count & (count - 1)) == 0)
        if should_emit:
            responses = context.affect_responses[stimulus_type]
            mean_v = sum(r[0] for r in responses) / count
            mean_a = sum(r[1] for r in responses) / count
            candidates.append(
                PatternCandidate(
                    type=PatternType.AFFECT_PATTERN,
                    elements=[
                        stimulus_type,
                        f"val:{mean_v:+.2f}",
                        f"aro:{mean_a:+.2f}",
                    ],
                    count=count,
                    confidence=0.5,
                    examples=[episode.id],
                    metadata={
                        "mean_valence_delta": round(mean_v, 3),
                        "mean_arousal_delta": round(mean_a, 3),
                        "stimulus_type": stimulus_type,
                    },
                )
            )

        return candidates


# ─── Default Detector Set ─────────────────────────────────────────────────────


def build_default_detectors() -> list[PatternDetector]:
    """Return the standard set of online pattern detectors."""
    return [
        CooccurrenceDetector(),
        SequenceDetector(),
        TemporalDetector(),
        AffectPatternDetector(),
    ]


# ─── Private Helpers ──────────────────────────────────────────────────────────


def _is_successful_action_outcome(episode: Episode) -> bool:
    """
    Determine whether an episode represents a successful action outcome.
    Axon audit records are stored with source="axon:{action_type}" and
    salience boosted on success. We use source prefix as primary signal.
    """
    source = episode.source or ""
    # Axon stores audit records with "axon:" prefix
    if not ("axon" in source or "action" in source):
        return False
    # Successful outcomes have positive affect or high salience
    return episode.affect_valence >= 0.0 and episode.salience_composite > 0.3


def _extract_action_sequence(episode: Episode) -> list[str]:
    """
    Extract the sequence of action types from an episode.
    Axon audit records encode action type in the source field:
    source = "axon:{action_type}" for single-step intents.
    Multi-step intents store the sequence in salience_scores metadata
    under the key "action_sequence".
    """
    source = episode.source or ""
    if source.startswith("axon:"):
        action_type = source[len("axon:"):]
        if action_type:
            return [action_type]

    # Fallback: check salience_scores for action_sequence encoding.
    # Multi-step intents encode the sequence as a JSON list under "_action_sequence".
    seq_raw = episode.salience_scores.get("_action_sequence")
    if seq_raw is not None:
        try:
            parsed = json.loads(str(seq_raw))
            if isinstance(parsed, list) and parsed:
                return [str(a) for a in parsed if a]
        except (ValueError, TypeError):
            pass

    return []


def _classify_source(source: str) -> str:
    """
    Classify an episode's source into a semantic category for temporal binning.
    """
    if not source:
        return "general"
    if "text_chat" in source:
        return "social_text"
    if "axon" in source or "action" in source:
        return "action_outcome"
    if "sensor" in source or "iot" in source:
        return "environmental"
    if "memory_bubble" in source:
        return "memory_recall"
    if "federation" in source:
        return "federation"
    if "system_event" in source:
        return "system"
    if "evo_insight" in source:
        return "evo_insight"
    return "general"
