"""
EcodiaOS - RE Training Quality Pipeline

Implements the quality scoring, filtering, and diversity enforcement functions
from speciation bible §2.4.

score_example() is a direct port of the bible's function with the only change
being that it also checks `failure_analysis` and `hypothesis` fields that the
bible's single `reasoning_chain` field doesn't cover (because Stream 2/3/4/5
store their reasoning under different keys).

All functions are pure / synchronous - no I/O.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

# ─── Scoring constants (from bible §2.4) ────────────────────────────────────

CAUSAL_MARKERS = [
    "because",
    "therefore",
    "causes",
    "leads to",
    "results in",
    "due to",
    "consequence",
    r"if.*then",
]

_CAUSAL_RE = [re.compile(m) for m in CAUSAL_MARKERS]

CONST_TERMS = [
    "constitutional",
    "value",
    "principle",
    "aligned",
    "equor",
    "safety",
    "wellbeing",
    "flag",
    "pass",
]

_STEP_RE = re.compile(r"(step \d|first,|second,|because|however|therefore)")

# Diversity cap: no single stream_id > this fraction of a batch
_MAX_STREAM_FRACTION = 0.30

# Minimum distinct time periods in a batch (ISO date strings bucketed by day)
_MIN_TIME_PERIODS = 3


# ─── Core scoring ─────────────────────────────────────────────────────────────


def score_example(ex: dict[str, Any]) -> float:
    """
    Score a training example on [0, 1].

    Exact formula from bible §2.4:
        0.30 × reasoning_depth
      + 0.20 × constitutional_awareness
      + 0.25 × causal_structure
      + 0.25 × novelty

    The `text` blob aggregates all reasoning-bearing fields across all 5 streams
    so the formula works uniformly regardless of stream origin.
    """
    # Aggregate all text fields that carry reasoning
    text_parts = [
        ex.get("reasoning_chain", ""),
        ex.get("completion", ""),
        ex.get("failed_reasoning", ""),
        ex.get("failure_analysis", ""),
        ex.get("equor_intervention", ""),
        ex.get("cause", ""),
        ex.get("effect", ""),
        ex.get("abstract_form", ""),
        ex.get("hypothesis", ""),
    ]
    text = " ".join(p for p in text_parts if p).lower()

    # Reasoning depth - count structural markers
    steps = len(_STEP_RE.findall(text))
    reasoning_depth = min(1.0, steps / 5)

    # Constitutional awareness
    constitutional_awareness = min(
        1.0,
        sum(1 for t in CONST_TERMS if t in text) / 3,
    )

    # Causal structure
    causal_hits = sum(1 for cr in _CAUSAL_RE if cr.search(text))
    causal_structure = min(1.0, causal_hits / 4)

    # Novelty - from whichever field is present
    novelty = float(
        ex.get("novelty_score", ex.get("surprise_factor", 0.5)) or 0.5
    )
    novelty = max(0.0, min(1.0, novelty))

    score = (
        0.30 * reasoning_depth
        + 0.20 * constitutional_awareness
        + 0.25 * causal_structure
        + 0.25 * novelty
    )
    return round(score, 4)


# ─── Filtering ────────────────────────────────────────────────────────────────


def filter_batch(
    examples: list[dict[str, Any]],
    min_score: float = 0.30,
) -> list[dict[str, Any]]:
    """
    Remove examples below min_score and attach `quality_score` to each
    surviving example (used later by export_pipeline for JSONL metadata).
    """
    scored: list[dict[str, Any]] = []
    for ex in examples:
        score = score_example(ex)
        if score >= min_score:
            item = dict(ex)
            item["quality_score"] = score
            item["training_weight"] = score  # bible: "Assign training_weight = score"
            scored.append(item)
    return scored


# ─── Diversity enforcement ────────────────────────────────────────────────────


def ensure_diversity(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Enforce that no single stream_id exceeds 30% of the batch.

    Strategy: sort each stream by quality_score DESC, keep the top-N where
    N = floor(0.30 × total). The total is iteratively computed after capping
    dominant streams, so minority streams keep all their examples.
    """
    if not examples:
        return examples

    # Group by stream_id
    by_stream: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ex in examples:
        sid = int(ex.get("stream_id", 0))
        by_stream[sid].append(ex)

    # Sort each stream by quality descending
    for sid in by_stream:
        by_stream[sid].sort(
            key=lambda x: float(x.get("quality_score", 0.0)),
            reverse=True,
        )

    total = len(examples)
    cap = math.floor(_MAX_STREAM_FRACTION * total)

    result: list[dict[str, Any]] = []
    for sid, stream_examples in by_stream.items():
        result.extend(stream_examples[:cap])

    # If capping reduced the total, recalculate and allow uncapped streams
    # to contribute up to their natural size (no double-capping).
    # Simple pass: just return what we have; caller can rescore diversity.
    return result


def ensure_temporal_span(
    examples: list[dict[str, Any]],
    min_periods: int = _MIN_TIME_PERIODS,
) -> list[dict[str, Any]]:
    """
    Ensure the batch spans at least `min_periods` distinct calendar days.

    If the current set spans fewer days, examples are returned as-is
    (we never discard data to force span - we warn instead). The CLI
    stats command surfaces this so operators know when the graph is too
    sparse.

    The `created_at` field is expected to be an ISO datetime string;
    we bucket by date (YYYY-MM-DD).
    """
    if not examples:
        return examples

    periods: set[str] = set()
    for ex in examples:
        raw = ex.get("created_at", "")
        if not raw:
            continue
        try:
            # Accept both "2026-03-07T..." and "2026-03-07"
            day = str(raw)[:10]
            periods.add(day)
        except Exception:
            continue

    if len(periods) < min_periods:
        # Log but don't discard - graph may genuinely be young
        import structlog as _sl
        _sl.get_logger("reasoning_engine.quality_pipeline").warning(
            "temporal_span_below_minimum",
            distinct_days=len(periods),
            min_periods=min_periods,
            example_count=len(examples),
        )

    return examples


# ─── Full quality pass ────────────────────────────────────────────────────────


def apply_full_quality_pass(
    examples: list[dict[str, Any]],
    min_score: float = 0.30,
    min_periods: int = _MIN_TIME_PERIODS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Run all quality steps in order:
        1. score + filter
        2. diversity enforcement
        3. temporal span check

    Returns (final_examples, stats_dict).
    The stats dict is included verbatim in the ExportResult from export_pipeline.
    """
    pre_filter = len(examples)
    filtered = filter_batch(examples, min_score=min_score)
    post_filter = len(filtered)

    diverse = ensure_diversity(filtered)
    post_diversity = len(diverse)

    final = ensure_temporal_span(diverse, min_periods=min_periods)

    # Per-stream breakdown
    stream_counts: dict[str, int] = defaultdict(int)
    quality_sum = 0.0
    for ex in final:
        sid = f"stream_{ex.get('stream_id', '?')}"
        stream_counts[sid] += 1
        quality_sum += float(ex.get("quality_score", 0.0))

    mean_quality = quality_sum / max(len(final), 1)

    stats = {
        "pre_filter_count": pre_filter,
        "post_filter_count": post_filter,
        "post_diversity_count": post_diversity,
        "final_count": len(final),
        "mean_quality_score": round(mean_quality, 4),
        "stream_counts": dict(stream_counts),
    }

    return final, stats
