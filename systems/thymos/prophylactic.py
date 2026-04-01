"""
EcodiaOS - Thymos Prophylactic Layer (Prevention)

Thymos doesn't just react to errors - it prevents them.

Two components:
  1. ProphylacticScanner    - scans new code against the antibody library
  2. HomeostasisController  - maintains optimal operating ranges proactively
"""

from __future__ import annotations

import re
import statistics
import time
from collections import deque
from typing import Any

import structlog

from clients.embedding import EmbeddingClient, cosine_similarity
from systems.thymos.types import (
    ParameterAdjustment,
    ParameterFix,
    ProphylacticWarning,
)

logger = structlog.get_logger()

# Cosine similarity threshold above which an intent/code pattern is flagged
# as prophylactically dangerous (tuned to ~0.85 per spec P2).
_EMBEDDING_SIMILARITY_THRESHOLD = 0.85

# Keyword fallback threshold (used when embedder unavailable)
_KEYWORD_SIMILARITY_THRESHOLD = 0.70


# ─── Prophylactic Scanner ───────────────────────────────────────


class ProphylacticScanner:
    """
    Scans new or modified code against the Antibody Library's error patterns.

    "This code pattern has caused 3 incidents in the past. The error was
    always an unhandled None return from Memory. Consider adding a null check."

    This is vaccination: exposure to weakened versions of past errors
    to build resistance before the real infection hits.

    Similarity engine (P2 upgrade):
      Each antibody's incident fingerprint text is embedded into 768-dim
      sentence-transformer space on first scan.  New code/intent text is
      also embedded.  Cosine similarity > 0.85 → prophylactic warning.

      Falls back to keyword overlap when the embedding client is unavailable.

    Oneiros repair schemas (SG8):
      When Oneiros completes a consolidation cycle, Thymos queries Memory
      for (:Procedure {thymos_repair: true}) nodes and adds their embeddings
      to the fingerprint store so that repair patterns learned during sleep
      are immediately available for prophylactic matching.
    """

    def __init__(
        self,
        antibody_library: Any = None,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self._library = antibody_library
        self._embedder: EmbeddingClient | None = embedding_client
        self._scans_run: int = 0
        self._warnings_issued: int = 0
        self._logger = logger.bind(system="thymos", component="prophylactic_scanner")

        # fingerprint_id → (label, embedding vector)
        # Populated from antibody library on first scan + from Oneiros procedures.
        self._fingerprint_store: dict[str, tuple[str, list[float]]] = {}

    # ── Public API ──────────────────────────────────────────────────

    def set_embedding_client(self, client: EmbeddingClient) -> None:
        """Hot-swap the embedding client (e.g. after lazy init in service)."""
        self._embedder = client

    async def scan(
        self,
        files_changed: list[str],
        file_contents: dict[str, str] | None = None,
    ) -> list[ProphylacticWarning]:
        """
        Scan changed files against known error patterns.

        Args:
            files_changed: List of file paths that were modified.
            file_contents: Optional map of filepath → content. If not provided,
                          only filename-based pattern matching is used.
        """
        self._scans_run += 1
        if self._library is None:
            return []

        warnings: list[ProphylacticWarning] = []
        all_antibodies = await self._library.get_all_active()

        for filepath in files_changed:
            content = (file_contents or {}).get(filepath, "")

            for antibody in all_antibodies:
                # Check if this file is in the same system as the antibody
                if antibody.source_system not in filepath:
                    continue

                # Embedding-based similarity (P2) - preferred path
                if self._embedder is not None:
                    similarity = await self._embedding_similarity(
                        content, antibody, filepath
                    )
                    threshold = _EMBEDDING_SIMILARITY_THRESHOLD
                else:
                    # Fallback: keyword overlap
                    similarity = self._keyword_similarity(
                        content, antibody.error_pattern, filepath
                    )
                    threshold = _KEYWORD_SIMILARITY_THRESHOLD

                if similarity > threshold:
                    warnings.append(
                        ProphylacticWarning(
                            filepath=filepath,
                            antibody_id=antibody.id,
                            warning=(
                                f"Code pattern similar to known error: "
                                f"{antibody.root_cause_description}"
                            ),
                            suggestion=self._extract_fix_suggestion(antibody),
                            confidence=similarity,
                        )
                    )

        self._warnings_issued += len(warnings)

        if warnings:
            self._logger.info(
                "prophylactic_warnings",
                files_scanned=len(files_changed),
                warnings=len(warnings),
                engine="embedding" if self._embedder is not None else "keyword",
            )

        return warnings

    async def check_intent_similarity(
        self, intent_text: str
    ) -> list[tuple[str, float, str]]:
        """
        Check a new intent's text against all fingerprints in the store.

        Returns list of (fingerprint_id, similarity, label) for entries
        with cosine similarity > _EMBEDDING_SIMILARITY_THRESHOLD.

        Used by the service to gate intents prophylactically before execution.
        """
        if self._embedder is None or not self._fingerprint_store:
            return []

        try:
            intent_vec = await self._embedder.embed(intent_text)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "prophylactic_intent_embed_failed", error=str(exc)[:200]
            )
            return []

        hits: list[tuple[str, float, str]] = []
        for fp_id, (label, fp_vec) in self._fingerprint_store.items():
            sim = cosine_similarity(intent_vec, fp_vec)
            if sim > _EMBEDDING_SIMILARITY_THRESHOLD:
                hits.append((fp_id, sim, label))

        hits.sort(key=lambda h: h[1], reverse=True)
        return hits

    async def add_fingerprints_from_procedures(
        self, procedures: list[dict[str, Any]]
    ) -> int:
        """
        Ingest (:Procedure {thymos_repair: true}) nodes from Memory into the
        fingerprint store.  Called by the Oneiros consolidation handler (SG8).

        Each procedure dict is expected to contain:
          - id:          str  - Neo4j node ID
          - name:        str  - human-readable procedure name
          - description: str  - textual summary of the repair pattern
          - fingerprint: str  - (optional) hex fingerprint if known

        Returns the number of new fingerprints added.
        """
        if self._embedder is None or not procedures:
            return 0

        texts = [
            p.get("description") or p.get("name") or ""
            for p in procedures
        ]
        ids = [p.get("id", "") or p.get("fingerprint", "") for p in procedures]
        labels = [p.get("name", "") or p.get("id", "") for p in procedures]

        # Filter out already-known fingerprints
        new_indices = [
            i for i, fid in enumerate(ids)
            if fid and fid not in self._fingerprint_store and texts[i]
        ]
        if not new_indices:
            return 0

        batch_texts = [texts[i] for i in new_indices]
        try:
            vectors = await self._embedder.embed_batch(batch_texts)
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "prophylactic_procedure_embed_failed",
                error=str(exc)[:200],
                count=len(batch_texts),
            )
            return 0

        added = 0
        for pos, idx in enumerate(new_indices):
            self._fingerprint_store[ids[idx]] = (labels[idx], vectors[pos])
            added += 1

        self._logger.info(
            "prophylactic_fingerprints_added_from_oneiros",
            added=added,
            store_size=len(self._fingerprint_store),
        )
        return added

    # ── Private helpers ─────────────────────────────────────────────

    async def _embedding_similarity(
        self,
        content: str,
        antibody: Any,
        filepath: str,
    ) -> float:
        """
        Compute cosine similarity between code content and antibody fingerprint
        using sentence-transformer embeddings (768-dim).

        The antibody's embedding is cached in _fingerprint_store keyed by
        antibody.id so each antibody is only embedded once per process lifetime.
        """
        assert self._embedder is not None  # caller checks

        fingerprint_text = (
            antibody.error_pattern or antibody.root_cause_description or ""
        )
        if not fingerprint_text:
            return 0.0

        # Ensure antibody fingerprint is cached
        if antibody.id not in self._fingerprint_store:
            try:
                fp_vec = await self._embedder.embed(fingerprint_text)
                self._fingerprint_store[antibody.id] = (
                    antibody.root_cause_description[:120],
                    fp_vec,
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.debug(
                    "prophylactic_fingerprint_embed_failed",
                    antibody_id=antibody.id,
                    error=str(exc)[:200],
                )
                return self._keyword_similarity(
                    content, antibody.error_pattern, filepath
                )

        _, fp_vec = self._fingerprint_store[antibody.id]

        if not content:
            return 0.0

        try:
            content_vec = await self._embedder.embed(content[:2000])
        except Exception as exc:  # noqa: BLE001
            self._logger.debug(
                "prophylactic_content_embed_failed", error=str(exc)[:200]
            )
            return self._keyword_similarity(content, antibody.error_pattern, filepath)

        return cosine_similarity(content_vec, fp_vec)

    def _keyword_similarity(
        self,
        content: str,
        error_pattern: str,
        filepath: str,
    ) -> float:
        """Fallback keyword-overlap similarity when embedder is unavailable."""
        if not content or not error_pattern:
            return 0.3 if error_pattern else 0.0

        content_lower = content.lower()
        pattern_lower = error_pattern.lower()

        pattern_words = set(
            w for w in re.split(r"[^a-z_]+", pattern_lower)
            if len(w) > 3 and w not in {"none", "error", "failed", "the", "and", "was"}
        )

        if not pattern_words:
            return 0.0

        matches = sum(1 for w in pattern_words if w in content_lower)
        return min(1.0, matches / len(pattern_words))

    def _extract_fix_suggestion(self, antibody: Any) -> str:
        """Generate a fix suggestion from an antibody."""
        if antibody.repair_spec.action == "adjust_parameters":
            return f"Consider parameter adjustment: {antibody.root_cause_description}"
        if antibody.repair_spec.action == "restart_system":
            return "This pattern has historically required a restart to resolve."
        ab_short = antibody.id[:8]
        return f"Known fix available (antibody {ab_short}): {antibody.root_cause_description}"

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "scans_run": self._scans_run,
            "warnings_issued": self._warnings_issued,
            "fingerprints_cached": len(self._fingerprint_store),
            "embedding_engine": self._embedder is not None,
        }


# ─── Homeostasis Controller ─────────────────────────────────────


# Optimal ranges for key metrics
DEFAULT_HOMEOSTATIC_RANGES: dict[str, tuple[float, float, str]] = {
    # Metric → (optimal_min, optimal_max, unit)
    "synapse.cycle.latency_ms": (80, 180, "ms"),
    "memory.retrieval.latency_ms": (10, 150, "ms"),
    "synapse.resources.memory_mb": (0, 3072, "MB"),
    "atune.coherence.phi": (0.3, 1.0, "phi"),
    "evo.self_model.success_rate": (0.5, 1.0, "rate"),
    "nova.intent_rate": (0.01, 0.5, "per_cycle"),
}


class _AdaptiveBaseline:
    """
    Rolling 7-day median baseline for a single metric.

    Stores timestamped samples. Computes adaptive optimal range as:
      median ± tolerance_fraction × median

    Falls back to static range when fewer than MIN_SAMPLES samples exist.
    """

    WINDOW_S = 7 * 86400.0  # 7 days
    MIN_SAMPLES = 50  # minimum samples before adaptive range activates
    TOLERANCE_FRACTION = 0.25  # ±25% of median

    def __init__(self, static_min: float, static_max: float) -> None:
        self._static_min = static_min
        self._static_max = static_max
        self._samples: deque[tuple[float, float]] = deque()  # (timestamp, value)

    def record(self, value: float) -> None:
        now = time.monotonic()
        self._samples.append((now, value))
        # Prune samples older than 7 days
        cutoff = now - self.WINDOW_S
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    @property
    def adaptive_range(self) -> tuple[float, float]:
        """Return (opt_min, opt_max). Uses rolling median if enough data, else static."""
        if len(self._samples) < self.MIN_SAMPLES:
            return (self._static_min, self._static_max)
        values = [v for _, v in self._samples]
        med = statistics.median(values)
        tolerance = abs(med) * self.TOLERANCE_FRACTION
        # Clamp so we never produce a negative lower bound for inherently positive metrics
        adaptive_min = max(0.0, med - tolerance)
        adaptive_max = med + tolerance
        return (adaptive_min, adaptive_max)

    @property
    def sample_count(self) -> int:
        return len(self._samples)


class HomeostasisController:
    """
    Maintains optimal operating ranges for key metrics.

    Like body temperature regulation: the organism doesn't wait for
    hypothermia or heatstroke. It actively maintains homeostasis.

    When a metric is trending toward the edge of its optimal range,
    Thymos makes small preemptive adjustments to pull it back.

    Ranges are adaptive: after 50+ samples over 7 days, the optimal
    range is computed as rolling median ± 25%. Before that, static
    DEFAULT_HOMEOSTATIC_RANGES are used.

    This runs on the MAINTAIN step of the cognitive cycle - always-on
    background processing, not triggered by incidents.
    """

    def __init__(
        self,
        ranges: dict[str, tuple[float, float, str]] | None = None,
    ) -> None:
        self._static_ranges = ranges or DEFAULT_HOMEOSTATIC_RANGES
        # Adaptive baselines per metric (P3)
        self._baselines: dict[str, _AdaptiveBaseline] = {
            name: _AdaptiveBaseline(opt_min, opt_max)
            for name, (opt_min, opt_max, _unit) in self._static_ranges.items()
        }
        # Metric → list of recent values for trend detection
        self._history: dict[str, list[float]] = {
            name: [] for name in self._static_ranges
        }
        self._max_history = 200
        self._adjustments_made: int = 0
        self._logger = logger.bind(system="thymos", component="homeostasis")

    @property
    def _ranges(self) -> dict[str, tuple[float, float, str]]:
        """Return current effective ranges - adaptive when data is sufficient."""
        result: dict[str, tuple[float, float, str]] = {}
        for name, (static_min, static_max, unit) in self._static_ranges.items():
            baseline = self._baselines.get(name)
            if baseline is not None:
                a_min, a_max = baseline.adaptive_range
                result[name] = (a_min, a_max, unit)
            else:
                result[name] = (static_min, static_max, unit)
        return result

    def record_metric(self, metric_name: str, value: float) -> None:
        """Record a metric value for trend tracking and adaptive baseline."""
        if metric_name not in self._static_ranges:
            return
        history = self._history[metric_name]
        history.append(value)
        if len(history) > self._max_history:
            self._history[metric_name] = history[-self._max_history:]
        # Feed adaptive baseline (P3: rolling 7-day median)
        baseline = self._baselines.get(metric_name)
        if baseline is not None:
            baseline.record(value)

    def check_homeostasis(self) -> list[ParameterAdjustment]:
        """
        Check all homeostatic ranges and propose micro-adjustments
        for any metric trending toward the edge.

        Returns parameter adjustments (if any). These are Tier 1 -
        no governance required, just config nudges.
        """
        adjustments: list[ParameterAdjustment] = []

        for metric, (opt_min, opt_max, _unit) in self._ranges.items():
            history = self._history[metric]
            if len(history) < 10:
                continue  # Need enough data

            current = history[-1]
            trend = self._compute_trend(history, window=min(100, len(history)))

            # Approaching upper bound and trending up
            if current > opt_max * 0.85 and trend > 0:
                adj = self._prescribe_cooling(metric, current, opt_max, trend)
                if adj is not None:
                    adjustments.append(adj)

            # Approaching lower bound and trending down
            elif current < opt_min * 1.15 + 0.01 and trend < 0:
                adj = self._prescribe_warming(metric, current, opt_min, trend)
                if adj is not None:
                    adjustments.append(adj)

        if adjustments:
            self._adjustments_made += len(adjustments)

        return adjustments

    def check_drift_warnings(self) -> list[dict[str, Any]]:
        """
        Return metrics in the warn zone (70–85% toward a boundary, trending that
        direction) WITHOUT prescribing an adjustment.

        These early drift signals are broadcast to Nova and Telos before the
        homeostatic correction tier fires (M8).  Format per item::

            {
                "metric":    str,          # e.g. "nova.intent_rate"
                "current":   float,
                "direction": "rising" | "falling",
                "boundary":  float,        # the bound being approached
                "proximity": float,        # 0-1, 1 = at boundary
                "trend":     float,        # regression slope
            }
        """
        warnings: list[dict[str, Any]] = []

        for metric, (opt_min, opt_max, _unit) in self._ranges.items():
            history = self._history[metric]
            if len(history) < 10:
                continue

            current = history[-1]
            trend = self._compute_trend(history, window=min(100, len(history)))

            # Upper warn zone: 70–85% of upper bound, trending up
            upper_warn = opt_max * 0.70
            upper_act = opt_max * 0.85
            if upper_warn < current <= upper_act and trend > 0:
                proximity = (current - opt_min) / max(opt_max - opt_min, 1e-9)
                warnings.append(
                    {
                        "metric": metric,
                        "current": current,
                        "direction": "rising",
                        "boundary": opt_max,
                        "proximity": round(proximity, 3),
                        "trend": round(trend, 6),
                    }
                )
                continue

            # Lower warn zone: between opt_min and 115% of opt_min (trending down)
            lower_act = opt_min * 1.15 + 0.01
            lower_warn = opt_min * 1.30 + 0.01
            if lower_act < current <= lower_warn and trend < 0:
                proximity = 1.0 - (current - opt_min) / max(opt_max - opt_min, 1e-9)
                warnings.append(
                    {
                        "metric": metric,
                        "current": current,
                        "direction": "falling",
                        "boundary": opt_min,
                        "proximity": round(proximity, 3),
                        "trend": round(trend, 6),
                    }
                )

        return warnings

    def _compute_trend(self, values: list[float], window: int) -> float:
        """Compute trend direction using simple linear regression slope."""
        if len(values) < 2:
            return 0.0
        recent = values[-window:]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _prescribe_cooling(
        self,
        metric: str,
        current: float,
        opt_max: float,
        trend: float,
    ) -> ParameterAdjustment | None:
        """Prescribe a small decrease to pull metric back from upper bound."""
        # Map metrics to adjustable parameters
        cooling_params: dict[str, tuple[str, float]] = {
            "synapse.cycle.latency_ms": ("synapse.clock.current_period_ms", 10),
            "memory.retrieval.latency_ms": ("memory.retrieval.cache_ttl_seconds", 60),
            "synapse.resources.memory_mb": ("evo.consolidation.batch_size", -2),
        }
        entry = cooling_params.get(metric)
        if entry is None:
            return None

        param_path, delta = entry
        opt_min = self._ranges[metric][0]

        return ParameterAdjustment(
            metric_name=metric,
            current_value=current,
            optimal_min=opt_min,
            optimal_max=opt_max,
            adjustment=ParameterFix(
                parameter_path=param_path,
                delta=delta,
                reason=f"Homeostatic cooling: {metric} at {current:.1f} (max: {opt_max})",
            ),
            trend_direction="rising",
        )

    def _prescribe_warming(
        self,
        metric: str,
        current: float,
        opt_min: float,
        trend: float,
    ) -> ParameterAdjustment | None:
        """Prescribe a small increase to pull metric back from lower bound."""
        warming_params: dict[str, tuple[str, float]] = {
            "atune.coherence.phi": ("synapse.clock.current_period_ms", -5),
            "evo.self_model.success_rate": ("evo.hypothesis.min_evidence", -1),
            "nova.intent_rate": ("synapse.clock.current_period_ms", -10),
        }
        entry = warming_params.get(metric)
        if entry is None:
            return None

        param_path, delta = entry
        opt_max = self._ranges[metric][1]

        return ParameterAdjustment(
            metric_name=metric,
            current_value=current,
            optimal_min=opt_min,
            optimal_max=opt_max,
            adjustment=ParameterFix(
                parameter_path=param_path,
                delta=delta,
                reason=f"Homeostatic warming: {metric} at {current:.2f} (min: {opt_min})",
            ),
            trend_direction="falling",
        )

    @property
    def metrics_in_range(self) -> int:
        """Count of metrics currently within optimal range."""
        count = 0
        for metric, (opt_min, opt_max, _) in self._ranges.items():
            history = self._history.get(metric, [])
            if history:
                current = history[-1]
                if opt_min <= current <= opt_max:
                    count += 1
        return count

    @property
    def metrics_total(self) -> int:
        return len(self._static_ranges)

    @property
    def adjustments_count(self) -> int:
        return self._adjustments_made

    @property
    def adaptive_baselines(self) -> dict[str, dict[str, Any]]:
        """Current adaptive baseline state for all monitored metrics."""
        result: dict[str, dict[str, Any]] = {}
        for name, baseline in self._baselines.items():
            a_min, a_max = baseline.adaptive_range
            static_min, static_max, unit = self._static_ranges[name]
            result[name] = {
                "adaptive_min": round(a_min, 3),
                "adaptive_max": round(a_max, 3),
                "static_min": static_min,
                "static_max": static_max,
                "samples": baseline.sample_count,
                "is_adaptive": baseline.sample_count >= _AdaptiveBaseline.MIN_SAMPLES,
                "unit": unit,
            }
        return result
