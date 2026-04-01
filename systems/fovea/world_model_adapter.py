"""
Fovea - WorldModelAdapter

A Memory-backed implementation of the LogosWorldModel protocol.

The real WorldModel does not exist as a standalone queryable system.
This adapter bridges Fovea's prediction needs to Memory's Episode graph:

- predict_content(context)  → hybrid-retrieve Episodes matching context,
                              return the centroid embedding of the top-k hits
                              as the expected next content vector.
- predict_timing(event_type) → query average inter-episode gap for this
                                source/event_type from Neo4j FOLLOWED_BY edges.
- get_prediction_confidence  → matching_episodes / window_total, clamped [0.1, 0.9]

Distance helpers
----------------
_semantic_distance  : cosine distance between 768-D sentence-transformer embeddings
_timing_distance    : normalised absolute delta between two timestamps / durations
_source_distance    : 0.0 if sources match, 1.0 otherwise (binary)

This is the live replacement for StubWorldModel until Logos WorldModel
is implemented. Wire it in main.py as:

    from systems.fovea.world_model_adapter import WorldModelAdapter
    wm = WorldModelAdapter(memory_service=memory_svc, neo4j_driver=driver)
    fovea_service.set_world_model(wm)
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.memory.service import MemoryService

logger = structlog.get_logger("systems.fovea.world_model_adapter")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPISODE_WINDOW = 200           # How many recent episodes define the prior
_MATCH_WINDOW = 50              # Context-match search window
_TIMING_HISTORY_MAX = 500       # Max inter-event timings kept in RAM cache
_MIN_CONFIDENCE = 0.1
_MAX_CONFIDENCE = 0.9
_TIMING_NORMALISER = 60.0       # Normalise timing errors against 60s baseline


# ---------------------------------------------------------------------------
# Low-level vector math (no external deps)
# ---------------------------------------------------------------------------


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine_similarity. Returns 1.0 on empty/mismatched inputs."""
    if not a or not b or len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - dot / (norm_a * norm_b)))


def _semantic_distance(embedding_a: list[float], embedding_b: list[float]) -> float:
    """
    Semantic distance between two 768-D sentence-transformer embeddings.

    Returns [0.0, 1.0]. 0.0 = identical, 1.0 = maximally dissimilar.
    Uses cosine distance (1 - cosine_similarity), which is the correct
    metric for normalised sentence embeddings.
    """
    return _cosine_distance(embedding_a, embedding_b)


def _timing_distance(
    expected_seconds: float,
    actual_seconds: float,
    normaliser: float = _TIMING_NORMALISER,
) -> float:
    """
    Normalised absolute difference between expected and actual timing.

    Returns [0.0, 1.0]. Saturates at normaliser (default 60s) offset.
    If expected_seconds <= 0 (no prediction) returns 0.0 (no error).
    """
    if expected_seconds <= 0:
        return 0.0
    delta = abs(actual_seconds - expected_seconds)
    return min(1.0, delta / max(normaliser, 0.001))


def _source_distance(expected_source: str, actual_source: str) -> float:
    """
    Binary source mismatch.

    Returns 0.0 if sources match (exact string), 1.0 if different.
    Returns 0.0 if either source is empty (no prediction made).
    """
    if not expected_source or not actual_source:
        return 0.0
    return 0.0 if expected_source == actual_source else 1.0


def _embedding_centroid(embeddings: list[list[float]]) -> list[float]:
    """Mean embedding over a list. Returns [] on empty input."""
    if not embeddings:
        return []
    dim = len(embeddings[0])
    centroid = [0.0] * dim
    for emb in embeddings:
        if len(emb) == dim:
            for i, v in enumerate(emb):
                centroid[i] += v
    n = len(embeddings)
    return [v / n for v in centroid]


# ---------------------------------------------------------------------------
# WorldModelAdapter
# ---------------------------------------------------------------------------


class WorldModelAdapter:
    """
    Memory-backed WorldModel for Fovea.

    Satisfies the LogosWorldModel protocol by querying Memory's Episode
    graph instead of a dedicated Logos world model.

    All methods are tolerant of Memory being unavailable (returns neutral defaults).
    Query results are cached with a short TTL to keep latency inside Fovea's
    18ms budget for prediction generation (the full budget is 18ms for ALL
    of generate_prediction(); each query here must be fast).

    Wiring:
        adapter = WorldModelAdapter(memory_service=memory_svc)
        fovea_service.set_world_model(adapter)

    For startup without Memory:
        adapter = WorldModelAdapter()  # returns neutral defaults until wired
        adapter.set_memory(memory_svc)
    """

    def __init__(
        self,
        memory_service: MemoryService | None = None,
        *,
        episode_window: int = _EPISODE_WINDOW,
        match_window: int = _MATCH_WINDOW,
        cache_ttl_s: float = 2.0,
    ) -> None:
        self._memory = memory_service
        self._episode_window = episode_window
        self._match_window = match_window
        self._cache_ttl_s = cache_ttl_s
        self._logger = logger.bind(component="world_model_adapter")

        # In-memory timing statistics keyed by source_system
        # deque of inter-event gap_seconds values
        self._timing_history: dict[str, deque[float]] = {}

        # Simple TTL cache: key → (value, expire_at)
        self._cache: dict[str, tuple[Any, float]] = {}

        # Context accuracy tracking for precision weighting
        # context_type → {dimension → (correct_count, total_count)}
        self._dimension_accuracy: dict[str, dict[str, list[int]]] = {}

        # Total episode count in last window (for confidence computation)
        self._window_total: int = 0
        self._window_total_ts: float = 0.0

    def set_memory(self, memory_service: MemoryService) -> None:
        self._memory = memory_service

    # ------------------------------------------------------------------
    # Public API (LogosWorldModel protocol)
    # ------------------------------------------------------------------

    async def predict_content(
        self, source_system: str, context: dict[str, Any]
    ) -> list[float]:
        """
        Return expected content embedding = centroid of top-k matching Episodes.

        Queries Memory with source_system as filter, then averages the
        embeddings of the returned MemoryTraces to form a predicted prior.
        """
        cache_key = f"content:{source_system}:{_ctx_hash(context)}"
        if cached := self._get_cached(cache_key):
            return cached

        if self._memory is None:
            return []

        try:
            response = await asyncio.wait_for(
                self._memory.retrieve(
                    query_text=source_system,
                    max_results=8,
                    salience_floor=0.1,
                ),
                timeout=0.010,  # 10ms hard cap - we're inside PERCEIVE phase
            )
            embeddings = [
                t.embedding
                for t in response.traces
                if t.embedding and t.source == source_system
            ]
            if not embeddings:
                # Fall back to any result if source filter yields nothing
                embeddings = [t.embedding for t in response.traces if t.embedding]

            result = _embedding_centroid(embeddings)
            self._set_cache(cache_key, result)
            return result
        except Exception:
            self._logger.debug("predict_content_failed", source=source_system, exc_info=True)
            return []

    async def predict_timing(
        self, source_system: str, context: dict[str, Any]
    ) -> float:
        """
        Return expected seconds until the next percept from source_system.

        Computed from average inter-episode gap stored in FOLLOWED_BY edges
        for this source. Falls back to in-memory timing cache seeded from
        Synapse PERCEPT_ARRIVED observations.
        """
        cache_key = f"timing:{source_system}"
        if cached := self._get_cached(cache_key):
            return cached

        # In-memory timing history (from PERCEPT_ARRIVED subscriptions)
        if source_system in self._timing_history:
            hist = self._timing_history[source_system]
            if hist:
                avg = sum(hist) / len(hist)
                self._set_cache(cache_key, avg)
                return avg

        # Neo4j fallback: average gap from FOLLOWED_BY edges for this source
        if self._memory is not None:
            try:
                avg_gap = await asyncio.wait_for(
                    self._query_avg_gap(source_system),
                    timeout=0.008,
                )
                if avg_gap > 0:
                    self._set_cache(cache_key, avg_gap)
                    return avg_gap
            except Exception:
                self._logger.debug("predict_timing_neo4j_failed", source=source_system, exc_info=True)

        return 0.0  # No timing prediction available

    async def predict_magnitude(
        self, source_system: str, context: dict[str, Any]
    ) -> float:
        """Return expected intensity: mean salience_composite of matching episodes."""
        cache_key = f"magnitude:{source_system}"
        if cached := self._get_cached(cache_key):
            return cached

        if self._memory is None:
            return 0.5

        try:
            response = await asyncio.wait_for(
                self._memory.retrieve(
                    query_text=source_system,
                    max_results=10,
                    salience_floor=0.0,
                ),
                timeout=0.008,
            )
            matching = [t for t in response.traces if t.source == source_system]
            if not matching:
                matching = response.traces
            if matching:
                mean_sal = sum(t.salience_score for t in matching) / len(matching)
                result = max(0.0, min(1.0, mean_sal))
                self._set_cache(cache_key, result)
                return result
        except Exception:
            self._logger.debug("predict_magnitude_failed", source=source_system, exc_info=True)

        return 0.5

    async def predict_source(
        self, source_system: str, context: dict[str, Any]
    ) -> str:
        """Predict next source = same source (identity prediction)."""
        return source_system

    async def predict_category(
        self, source_system: str, context: dict[str, Any]
    ) -> str:
        """
        Predict the most common modality for this source from recent episodes.
        """
        cache_key = f"category:{source_system}"
        if cached := self._get_cached(cache_key):
            return cached

        if self._memory is None:
            return ""

        try:
            response = await asyncio.wait_for(
                self._memory.retrieve(
                    query_text=source_system,
                    max_results=15,
                    salience_floor=0.0,
                ),
                timeout=0.008,
            )
            matching = [t for t in response.traces if t.source == source_system]
            if not matching:
                matching = response.traces
            if matching:
                # Most common modality
                modality_counts: dict[str, int] = {}
                for t in matching:
                    m = t.modality or ""
                    modality_counts[m] = modality_counts.get(m, 0) + 1
                best = max(modality_counts, key=modality_counts.get)  # type: ignore[arg-type]
                self._set_cache(cache_key, best)
                return best
        except Exception:
            self._logger.debug("predict_category_failed", source=source_system, exc_info=True)

        return ""

    async def predict_causal_context(
        self, source_system: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Predict causal context from the most recent episode for this source.

        Returns source-keyed dict with prior sample count (used by timing
        and magnitude predictors) and mean affect from matching episodes.
        """
        cache_key = f"causal:{source_system}:{_ctx_hash(context)}"
        if cached := self._get_cached(cache_key):
            return cached

        if self._memory is None:
            return {}

        try:
            response = await asyncio.wait_for(
                self._memory.retrieve(
                    query_text=source_system,
                    max_results=5,
                    salience_floor=0.0,
                ),
                timeout=0.008,
            )
            matching = [t for t in response.traces if t.source == source_system]
            if not matching:
                matching = response.traces[:3]

            result: dict[str, Any] = {
                "prior_sample_count": len(response.traces),
                "source_system": source_system,
            }
            if matching:
                result["mean_affect_valence"] = sum(
                    getattr(t, "affect_valence", 0.0) or 0.0 for t in matching
                ) / len(matching)

            self._set_cache(cache_key, result)
            return result
        except Exception:
            self._logger.debug("predict_causal_failed", source=source_system, exc_info=True)

        return {}

    async def get_prediction_confidence(
        self, source_system: str, context: dict[str, Any]
    ) -> float:
        """
        Confidence = matching_episodes / total_episodes_in_window, clamped to [0.1, 0.9].

        Matching episodes: episodes from this source in the last _match_window.
        Total episodes: episodes from any source in the last _episode_window.
        """
        window_total = await self._get_window_total()
        if window_total == 0:
            return 0.5  # No data - neutral

        if self._memory is None:
            return 0.5

        try:
            matching_response = await asyncio.wait_for(
                self._memory.retrieve(
                    query_text=source_system,
                    max_results=self._match_window,
                    salience_floor=0.0,
                ),
                timeout=0.008,
            )
            matching = sum(
                1 for t in matching_response.traces if t.source == source_system
            )
            confidence = matching / window_total
            return max(_MIN_CONFIDENCE, min(_MAX_CONFIDENCE, confidence))
        except Exception:
            return 0.5

    async def get_context_reliability(self, context_type: str) -> float:
        """Derive reliability from per-context dimension accuracy."""
        if context_type in self._dimension_accuracy:
            accuracies = []
            for dim_counts in self._dimension_accuracy[context_type].values():
                total = dim_counts[1]
                if total > 0:
                    accuracies.append(dim_counts[0] / total)
            if accuracies:
                return max(0.1, min(0.9, sum(accuracies) / len(accuracies)))
        return 0.5

    async def get_historical_accuracy(
        self, context_type: str, lookback_window: int = 100
    ) -> float:
        return await self.get_context_reliability(context_type)

    async def get_dimension_accuracy(
        self, context_type: str, dimension: str, lookback_window: int = 100
    ) -> float:
        """Per-dimension accuracy from in-memory tracking."""
        if context_type in self._dimension_accuracy:
            dim_counts = self._dimension_accuracy[context_type].get(dimension)
            if dim_counts and dim_counts[1] > 0:
                return max(0.1, min(0.9, dim_counts[0] / dim_counts[1]))
        return 0.5  # neutral prior

    async def get_context_stability_age(self, context_type: str) -> int:
        """Return 0 - stability age is not tracked by Memory-backed adapter."""
        return 0

    async def get_compression_score(self) -> float:
        """Return 0.5 - compression score requires Logos; not available here."""
        return 0.5

    # ------------------------------------------------------------------
    # Feedback: record timing observations (called from PERCEPT_ARRIVED handler)
    # ------------------------------------------------------------------

    def record_timing_observation(self, source_system: str, gap_seconds: float) -> None:
        """
        Record an inter-event timing observation for a source.

        Called by FoveaService._on_percept_arrived() so timing predictions
        improve with each percept without Neo4j roundtrips.
        """
        if source_system not in self._timing_history:
            self._timing_history[source_system] = deque(maxlen=_TIMING_HISTORY_MAX)
        self._timing_history[source_system].append(gap_seconds)
        # Invalidate timing cache for this source
        self._cache.pop(f"timing:{source_system}", None)

    def record_prediction_outcome(
        self,
        context_type: str,
        dimension: str,
        was_correct: bool,
    ) -> None:
        """
        Record whether a prediction for a context_type/dimension was correct.

        Drives get_dimension_accuracy() and get_context_reliability().
        Called by FoveaService after each percept is resolved.
        """
        if context_type not in self._dimension_accuracy:
            self._dimension_accuracy[context_type] = {}
        if dimension not in self._dimension_accuracy[context_type]:
            self._dimension_accuracy[context_type][dimension] = [0, 0]
        counts = self._dimension_accuracy[context_type][dimension]
        if was_correct:
            counts[0] += 1
        counts[1] += 1
        # Cap per-cell history to avoid unbounded growth
        if counts[1] > 10_000:
            counts[0] = int(counts[0] * 0.9)
            counts[1] = int(counts[1] * 0.9)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get_window_total(self) -> int:
        """Return total episode count in the window (cached 2s)."""
        now = time.monotonic()
        if now - self._window_total_ts < self._cache_ttl_s and self._window_total > 0:
            return self._window_total

        if self._memory is None:
            return 0

        try:
            response = await asyncio.wait_for(
                self._memory.retrieve(
                    max_results=self._episode_window,
                    salience_floor=0.0,
                ),
                timeout=0.010,
            )
            self._window_total = len(response.traces)
            self._window_total_ts = now
            return self._window_total
        except Exception:
            return self._window_total or 0

    async def _query_avg_gap(self, source_system: str) -> float:
        """
        Query average FOLLOWED_BY gap_seconds for source_system from Neo4j.

        Falls back to 0.0 if Memory's Neo4j client is not accessible.
        """
        if self._memory is None:
            return 0.0

        # Access the internal Neo4j client via memory service if available
        neo4j = getattr(self._memory, "_neo4j", None)
        if neo4j is None:
            return 0.0

        try:
            records = await neo4j.execute_read(
                """
                MATCH (a:Episode)-[r:FOLLOWED_BY]->(b:Episode)
                WHERE a.source = $source
                WITH r.gap_seconds AS gap
                WHERE gap IS NOT NULL AND gap > 0
                RETURN avg(gap) AS avg_gap, count(gap) AS n
                LIMIT 1
                """,
                {"source": source_system},
            )
            if records and records[0].get("avg_gap"):
                return float(records[0]["avg_gap"])
        except Exception:
            self._logger.debug("avg_gap_query_failed", source=source_system, exc_info=True)

        return 0.0

    def _get_cached(self, key: str) -> Any | None:
        if key in self._cache:
            value, expire_at = self._cache[key]
            if time.monotonic() < expire_at:
                return value
            del self._cache[key]
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        self._cache[key] = (value, time.monotonic() + self._cache_ttl_s)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _ctx_hash(context: dict[str, Any]) -> str:
    """Lightweight stable hash of a context dict for cache keys."""
    try:
        return str(hash(frozenset((k, str(v)[:40]) for k, v in sorted(context.items()))))
    except Exception:
        return ""
