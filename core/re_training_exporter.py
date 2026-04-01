"""
EcodiaOS - RE Training Data Exporter

Collects RE_TRAINING_EXAMPLE events from the Synapse bus, assembles them
into hourly RETrainingExportBatch objects, and ships them to:
  1. S3 / local filesystem - JSON lines for the offline CLoRA fine-tuning pipeline
  2. Neo4j - (:RETrainingBatch) nodes with [:CONTAINS]→(:RETrainingDatapoint) edges
              for audit lineage and Benchmarks tracking

The exporter runs as a supervised background task started in Phase 11 of
registry.py.  It is intentionally decoupled from every cognitive system -
it only reads from the event bus ring buffer + subscribes to
RE_TRAINING_EXAMPLE events.  No direct imports from any system module.

Export cadence: every 3600s (1 hour).  Batches with 0 datapoints are skipped.

PERMITTED training data sources (internal organism execution only):
  - Axon:      execution outcomes (confidence, cost, latency)
  - Nova:      belief updates, goal reasoning
  - Evo:       hypothesis confirm/refute
  - Equor:     constitutional alignment checks
  - Simula:    code mutation results
  - Kairos:    causal invariant discovery
  - Soma:      interoception signals
  - Memory:    Neo4j query outcomes
  - Logos:     compression examples
  - Synapse:   coordination outcomes
  - Identity:  connector/credential outcomes
  - Thread:    narrative coherence signals
  - Benchmarks: KPI deltas
  - Any other internal EOS cognitive system

FORBIDDEN training data sources (must NEVER feed into RE training):
  - X (Twitter) API data - posts, replies, timelines, search results
    (X Developer Agreement §5 prohibits use of X Content for model training)
  - Any other third-party platform content (LinkedIn, Reddit, etc.)
  - Web-scraped external content (Brave/SerpAPI results)
  - GitHub issue/PR bodies from external repositories
  - Any inbound data from external social or content platforms

If you add a new training data source, verify it is internal organism
telemetry before wiring it into RE_TRAINING_EXAMPLE emission.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from primitives.re_training import RETrainingDatapoint, RETrainingExportBatch
from primitives.common import new_id

# Deferred import - only resolved at export time to avoid circular deps at module load
def _get_scaffold_validator():  # type: ignore[return]
    try:
        from systems.reasoning_engine.scaffold_formatter import validate
        return validate
    except Exception:
        return None

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from clients.redis import RedisClient
    from systems.synapse.event_bus import EventBus
    from systems.synapse.types import SynapseEvent

logger = structlog.get_logger("core.re_training_exporter")

# How many seconds in each collection window
_EXPORT_INTERVAL_S: int = 3600  # 1 hour

# Local fallback export path (used when S3 is unavailable)
_LOCAL_EXPORT_DIR: str = os.environ.get(
    "RE_TRAINING_EXPORT_DIR", "data/re_training_batches"
)

# S3 config from environment
_S3_BUCKET: str = os.environ.get("RE_TRAINING_S3_BUCKET", "ecodiaos-re-training")
_S3_PREFIX: str = os.environ.get("RE_TRAINING_S3_PREFIX", "batches/")


def _outcome_from_quality(quality: float) -> str:
    """Map [0,1] outcome_quality → human-readable outcome label."""
    if quality >= 0.85:
        return "success"
    if quality >= 0.4:
        return "partial"
    return "failure"


_MIN_INSTRUCTION_LEN = 10   # Reject trivially empty instructions
_MIN_OUTPUT_LEN = 3         # Reject trivially empty outputs


def _datapoint_from_event(event: SynapseEvent) -> RETrainingDatapoint | None:
    """
    Convert a raw RE_TRAINING_EXAMPLE SynapseEvent into a RETrainingDatapoint.

    Returns None if the payload is malformed, missing required fields, or fails
    validation checks. The conversion is best-effort - never raises.

    Validation rules:
    - instruction must be a non-empty string with at least _MIN_INSTRUCTION_LEN chars
    - output must be a non-empty string with at least _MIN_OUTPUT_LEN chars
    - outcome_quality must be in [0.0, 1.0] (clamped - not rejected on range)
    - source_system must not be empty
    """
    try:
        data = event.data
        instruction = str(data.get("instruction", "")).strip()
        output = str(data.get("output", "")).strip()

        # Reject datapoints with trivially empty required fields
        if len(instruction) < _MIN_INSTRUCTION_LEN:
            logger.debug(
                "re_datapoint_rejected_short_instruction",
                source=event.source_system,
                length=len(instruction),
            )
            return None
        if len(output) < _MIN_OUTPUT_LEN:
            logger.debug(
                "re_datapoint_rejected_short_output",
                source=event.source_system,
                length=len(output),
            )
            return None

        quality_raw = data.get("outcome_quality", 0.0)
        try:
            quality = float(quality_raw)
        except (TypeError, ValueError):
            quality = 0.0
        # Clamp to valid range - don't reject, just fix
        quality = min(max(quality, 0.0), 1.0)

        source_system = str(data.get("source_system", event.source_system)).strip()
        if not source_system:
            source_system = event.source_system or "unknown"

        return RETrainingDatapoint(
            source_system=source_system,
            example_type=str(data.get("category", "unknown")),
            instruction=instruction[:2000],
            input_context=str(data.get("input_context", ""))[:4000],
            output_action=output[:2000],
            outcome=_outcome_from_quality(quality),
            confidence=quality,
            timestamp=event.timestamp,
            reasoning_trace=str(data.get("reasoning_trace", ""))[:4000],
            alternatives_considered=list(data.get("alternatives_considered", [])),
            cost_usd=Decimal(str(data.get("cost_usd", "0"))),
            latency_ms=int(data.get("latency_ms", 0)),
            episode_id=str(data.get("episode_id", "")),
        )
    except Exception:
        logger.debug("re_datapoint_conversion_failed", exc_info=True)
        return None


class RETrainingExporter:
    """
    Subscribes to RE_TRAINING_EXAMPLE events, assembles hourly export batches,
    and ships them to S3 + Neo4j.

    Lifecycle:
        exporter = RETrainingExporter(event_bus, neo4j, redis)
        exporter.attach()          # subscribe to event bus
        await exporter.run_loop()  # blocking; call via supervised_task
        exporter.detach()          # unsubscribe (called on shutdown)

    Thread safety: all state is accessed only from the asyncio event loop.
    """

    def __init__(
        self,
        event_bus: EventBus,
        neo4j: Neo4jClient | None = None,
        redis: RedisClient | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._neo4j = neo4j
        self._redis = redis
        # In-memory accumulator for the current window
        self._pending: list[RETrainingDatapoint] = []
        # Dedup: (source_system, episode_id) - episode_id="" means no dedup
        self._seen_episode_ids: set[str] = set()
        # Index for O(1) retroactive quality corrections: episode_id → datapoint
        self._episode_index: dict[str, RETrainingDatapoint] = {}
        self._window_start: datetime = datetime.now(UTC)
        self._total_exported = 0
        self._total_batches = 0
        self._attached = False
        # Metabolic-aware priority boost for economic training examples.
        # Updated by _on_metabolic_pressure(); applied in _enrich_batch().
        self._starvation_level: str = "nominal"
        self._metabolic_boost: float = 1.0

    # ─── Event Bus Integration ────────────────────────────────────────

    def attach(self) -> None:
        """Subscribe to RE_TRAINING_EXAMPLE, AXON_EXECUTION_RESULT, METABOLIC_PRESSURE,
        and EVO hypothesis lifecycle events."""
        from systems.synapse.types import SynapseEventType

        self._event_bus.subscribe(
            SynapseEventType.RE_TRAINING_EXAMPLE,
            self._on_re_training_example,
        )
        self._event_bus.subscribe(
            SynapseEventType.AXON_EXECUTION_RESULT,
            self._on_axon_execution_result,
        )
        if hasattr(SynapseEventType, "METABOLIC_PRESSURE"):
            self._event_bus.subscribe(
                SynapseEventType.METABOLIC_PRESSURE,
                self._on_metabolic_pressure,
            )
        # Hypothesis lifecycle → RE training examples
        # EVO_HYPOTHESIS_CONFIRMED/REFUTED are high-signal training datapoints:
        # they represent Evo's adjudicated view of a hypothesis after real-world
        # testing - exactly the kind of reasoning quality signal the RE should learn.
        if hasattr(SynapseEventType, "EVO_HYPOTHESIS_CONFIRMED"):
            self._event_bus.subscribe(
                SynapseEventType.EVO_HYPOTHESIS_CONFIRMED,
                self._on_hypothesis_confirmed,
            )
        if hasattr(SynapseEventType, "EVO_HYPOTHESIS_REFUTED"):
            self._event_bus.subscribe(
                SynapseEventType.EVO_HYPOTHESIS_REFUTED,
                self._on_hypothesis_refuted,
            )
        self._attached = True
        logger.info("re_training_exporter_attached")

    def detach(self) -> None:
        """Unsubscribe (best-effort - EventBus may not support unsubscribe)."""
        self._attached = False
        logger.info("re_training_exporter_detached")

    async def _on_re_training_example(self, event: SynapseEvent) -> None:
        """Hot-path handler - called by EventBus on every RE_TRAINING_EXAMPLE event."""
        dp = _datapoint_from_event(event)
        if dp is None:
            return

        # Dedup episodes: skip if we've already captured this episode in this window
        if dp.episode_id:
            dedup_key = f"{dp.source_system}:{dp.episode_id}"
            if dedup_key in self._seen_episode_ids:
                return
            self._seen_episode_ids.add(dedup_key)
            # Index by episode_id for retroactive quality corrections
            self._episode_index[dp.episode_id] = dp

        self._pending.append(dp)

    async def _on_axon_execution_result(self, event: SynapseEvent) -> None:
        """
        Retroactively correct outcome_quality when Axon's actual result arrives.

        AXON_EXECUTION_RESULT payload is expected to contain:
          - episode_id: str
          - success: bool
          - value_gained: float (optional)
          - quality: float (optional, [0,1])

        We derive an actual_quality from these and update the buffered datapoint
        if the correction differs from the estimated quality by > 0.1.
        """
        try:
            data = event.data
            episode_id = str(data.get("episode_id", ""))
            if not episode_id:
                return
            dp = self._episode_index.get(episode_id)
            if dp is None:
                return

            # Derive ground-truth quality from execution result
            actual_quality = float(data.get("quality", -1.0))
            if actual_quality < 0:
                success = bool(data.get("success", False))
                value_gained = float(data.get("value_gained", 0.0))
                actual_quality = min(1.0, value_gained) if success else max(0.0, value_gained * 0.3)

            self.update_outcome_quality(episode_id, actual_quality, source_system="axon")
        except Exception:
            logger.debug("re_exporter_axon_result_update_failed", exc_info=True)

    async def _on_metabolic_pressure(self, event: Any) -> None:
        """
        METABOLIC_PRESSURE → adjust economic training example priority boost.

        During metabolic crisis, the organism most needs to learn from economic
        decisions. Boost economic domain examples so the RE training curriculum
        prioritises the reasoning traces that led to revenue or failure.

        Boost table (mirrors StarvationLevel thresholds in oikos/models.py):
          CRITICAL / EMERGENCY → 2.0× (urgent - organism near death)
          AUSTERITY            → 1.5× (significant pressure)
          NOMINAL / CAUTIOUS   → 1.0× (no boost)
        """
        try:
            data = getattr(event, "data", {}) or {}
            level = str(data.get("starvation_level", "")).lower()
            old_boost = self._metabolic_boost
            if level in ("critical", "emergency"):
                self._metabolic_boost = 2.0
            elif level == "austerity":
                self._metabolic_boost = 1.5
            else:
                self._metabolic_boost = 1.0
            self._starvation_level = level or "nominal"
            if self._metabolic_boost != old_boost:
                logger.info(
                    "re_exporter_metabolic_boost_changed",
                    starvation_level=self._starvation_level,
                    old_boost=old_boost,
                    new_boost=self._metabolic_boost,
                )
        except Exception:
            logger.debug("re_exporter_metabolic_pressure_update_failed", exc_info=True)

    async def _on_hypothesis_confirmed(self, event: Any) -> None:
        """
        EVO_HYPOTHESIS_CONFIRMED → generate a RE training datapoint.

        Confirmed hypotheses are ground-truth adjudications: Evo has accumulated
        real evidence across ≥10 episodes with evidence_score ≥ 3.0. This is
        exactly the kind of slow, careful reasoning chain the RE should internalize.

        outcome_quality = 1.0 (success - the hypothesis was correct)
        """
        try:
            data = getattr(event, "data", {}) or {}
            statement = str(data.get("statement", data.get("hypothesis_statement", ""))).strip()
            if not statement:
                return
            evidence_score = float(data.get("evidence_score", 1.0))
            category = str(data.get("category", "hypothesis_validation"))
            supporting = int(data.get("supporting_count", 0))
            confidence = float(data.get("confidence", 0.8))
            hypothesis_id = str(data.get("hypothesis_id", ""))

            instruction = f"Evaluate hypothesis: '{statement[:500]}'"
            output = (
                f"CONFIRMED. Evidence score: {evidence_score:.2f}, "
                f"supporting episodes: {supporting}, confidence: {confidence:.2f}. "
                f"This hypothesis is supported by empirical evidence and should be integrated "
                f"into the organism's belief system."
            )
            reasoning = (
                f"1. HYPOTHESIS: {statement[:300]}\n"
                f"2. EVIDENCE: evidence_score={evidence_score:.2f}, supporting_count={supporting}\n"
                f"3. CONFIDENCE: {confidence:.2f} after real-world episode accumulation\n"
                f"4. VERDICT: CONFIRMED - hypothesis passed empirical testing"
            )

            dp = RETrainingDatapoint(
                source_system="evo",
                example_type=f"hypothesis_confirmed:{category}",
                instruction=instruction[:2000],
                input_context=f'{{"hypothesis_id": "{hypothesis_id}", "evidence_score": {evidence_score}, "category": "{category}"}}',
                output_action=output[:2000],
                outcome="success",
                confidence=min(max(confidence, 0.0), 1.0),
                timestamp=getattr(event, "timestamp", None),
                reasoning_trace=reasoning[:4000],
                alternatives_considered=[
                    "Alternative: hypothesis could have been refuted if evidence was contradictory",
                    "Alternative: hypothesis could remain in testing if evidence was insufficient",
                ],
                episode_id=hypothesis_id,
            )
            # Dedup - hypothesis_id is stable; don't double-count multiple CONFIRMED events
            if hypothesis_id:
                dedup_key = f"evo_confirmed:{hypothesis_id}"
                if dedup_key in self._seen_episode_ids:
                    return
                self._seen_episode_ids.add(dedup_key)
                self._episode_index[hypothesis_id] = dp
            self._pending.append(dp)
        except Exception:
            logger.debug("re_exporter_hypothesis_confirmed_failed", exc_info=True)

    async def _on_hypothesis_refuted(self, event: Any) -> None:
        """
        EVO_HYPOTHESIS_REFUTED → generate a RE training datapoint.

        Refuted hypotheses are negative ground-truth: what NOT to believe.
        Training the RE to recognise refuted reasoning patterns prevents the
        organism from repeatedly proposing the same failed hypotheses.

        outcome_quality = 0.0 (failure - the hypothesis was wrong)
        """
        try:
            data = getattr(event, "data", {}) or {}
            statement = str(data.get("statement", data.get("hypothesis_statement", ""))).strip()
            if not statement:
                return
            category = str(data.get("category", "hypothesis_validation"))
            contradictions = int(data.get("contradicting_count", 0))
            evidence_score = float(data.get("evidence_score", 0.0))
            hypothesis_id = str(data.get("hypothesis_id", ""))
            refutation_reason = str(data.get("refutation_reason", "accumulated contradictions"))

            instruction = f"Evaluate hypothesis: '{statement[:500]}'"
            output = (
                f"REFUTED. Contradicting episodes: {contradictions}, evidence_score: {evidence_score:.2f}. "
                f"This hypothesis contradicts empirical evidence and should be archived. "
                f"Reason: {refutation_reason[:200]}"
            )
            reasoning = (
                f"1. HYPOTHESIS: {statement[:300]}\n"
                f"2. COUNTER-EVIDENCE: {contradictions} contradicting episodes, evidence_score={evidence_score:.2f}\n"
                f"3. REFUTATION REASON: {refutation_reason[:200]}\n"
                f"4. VERDICT: REFUTED - hypothesis failed empirical testing"
            )

            dp = RETrainingDatapoint(
                source_system="evo",
                example_type=f"hypothesis_refuted:{category}",
                instruction=instruction[:2000],
                input_context=f'{{"hypothesis_id": "{hypothesis_id}", "evidence_score": {evidence_score}, "contradictions": {contradictions}}}',
                output_action=output[:2000],
                outcome="failure",
                confidence=0.0,
                timestamp=getattr(event, "timestamp", None),
                reasoning_trace=reasoning[:4000],
                alternatives_considered=[
                    "Alternative: hypothesis might have been confirmed with more supporting evidence",
                    "Alternative: domain scope could be narrowed to find a valid sub-hypothesis",
                ],
                episode_id=hypothesis_id,
            )
            if hypothesis_id:
                dedup_key = f"evo_refuted:{hypothesis_id}"
                if dedup_key in self._seen_episode_ids:
                    return
                self._seen_episode_ids.add(dedup_key)
                self._episode_index[hypothesis_id] = dp
            self._pending.append(dp)
        except Exception:
            logger.debug("re_exporter_hypothesis_refuted_failed", exc_info=True)

    def update_outcome_quality(
        self,
        episode_id: str,
        actual_quality: float,
        source_system: str = "external",
    ) -> bool:
        """
        Retroactively update outcome_quality for a buffered datapoint.

        Only applies the correction if |actual_quality - current_confidence| > 0.1.
        Returns True if the correction was applied, False otherwise.
        """
        dp = self._episode_index.get(episode_id)
        if dp is None:
            return False

        actual_quality = max(0.0, min(1.0, actual_quality))
        if abs(actual_quality - dp.confidence) <= 0.1:
            return False

        dp.confidence = actual_quality
        dp.outcome = _outcome_from_quality(actual_quality)
        dp.outcome_updated = True
        dp.actual_outcome_quality = actual_quality
        logger.debug(
            "re_training_outcome_corrected",
            episode_id=episode_id,
            actual_quality=actual_quality,
            source=source_system,
        )
        return True

    # ─── Batch Collection ─────────────────────────────────────────────

    def collect_batch(self) -> RETrainingExportBatch:
        """
        Drain the current accumulator into an RETrainingExportBatch.

        Resets the accumulator and dedup set for the next window.
        Returns an empty batch if no examples were collected.
        """
        now = datetime.now(UTC)
        hour_window = self._window_start.strftime("%Y-%m-%dT%H:00:00Z")

        datapoints = list(self._pending)
        source_systems = sorted({dp.source_system for dp in datapoints})

        # Reset for next window
        self._pending = []
        self._seen_episode_ids = set()
        self._episode_index = {}
        self._window_start = now

        return RETrainingExportBatch(
            id=new_id(),
            datapoints=datapoints,
            hour_window=hour_window,
            source_systems=source_systems,
        )

    # ─── Export Enrichment ────────────────────────────────────────────

    def _compute_task_difficulty(self, dp: RETrainingDatapoint) -> float:
        """
        Compute task_difficulty ∈ [0, 1] from richness signals.

        Formula (from spec):
          len(alternatives_considered) * 0.3
          + bool(counterfactual) * 0.3          ← sourced from raw event via reasoning_trace heuristic
          + outcome_quality_variance * 0.4

        We use |confidence - 0.5| * 2 as a proxy for outcome_quality_variance
        (decisions near 0.5 confidence are harder; near 0 or 1 are clearer).
        """
        alt_score = min(1.0, len(dp.alternatives_considered) / 5) * 0.3
        # Heuristic: counterfactual present if reasoning_trace contains "if" + "instead"
        has_counterfactual = bool(
            dp.reasoning_trace
            and "if" in dp.reasoning_trace.lower()
            and "instead" in dp.reasoning_trace.lower()
        )
        counterfactual_score = float(has_counterfactual) * 0.3
        variance_proxy = (1.0 - abs(dp.confidence - 0.5) * 2) * 0.4
        return round(min(1.0, alt_score + counterfactual_score + variance_proxy), 4)

    def _assign_quality_tier(
        self,
        dp: RETrainingDatapoint,
        scaffold_valid: bool,
    ) -> str:
        """
        Classify a datapoint into a quality tier.

        gold   - scaffold-compliant, rich trace, has alternatives or counterfactual,
                 high confidence
        silver - scaffold-compliant OR moderate richness
        bronze - minimal data or scaffold-non-compliant
        """
        has_trace = len(dp.reasoning_trace) > 100
        has_alternatives = len(dp.alternatives_considered) >= 2
        high_confidence = dp.confidence >= 0.7

        if scaffold_valid and has_trace and (has_alternatives or dp.task_difficulty >= 0.5) and high_confidence:
            return "gold"
        if scaffold_valid or (has_trace and has_alternatives):
            return "silver"
        return "bronze"

    def _detect_multi_step_strategy(
        self, datapoints: list[RETrainingDatapoint]
    ) -> None:
        """
        Detect multi-step economic strategies within a batch and annotate datapoints.

        Patterns recognised (by example_type sequence sharing the same episode root):
          "recovery_via_diversification": cost_reduce → yield_deploy → spawn_child
          "yield_optimization":           yield_rebalance → protocol_explore → consolidate
          "bounty_then_diversify":        bounty_hunt → asset_create → liquidate

        Strategy detection groups datapoints by their episode root (prefix before
        the first ':' in episode_id, or a shared oikos/economic category prefix).
        Groups with 2+ steps are annotated with strategy metadata.

        Priority boost: strategic datapoints with 2+ steps get their task_difficulty
        multiplied by 1.3× (capped at 1.0) to surface them during curriculum building.
        """
        # Known strategy patterns: ordered action sequences → strategy name
        _STRATEGY_PATTERNS: list[tuple[list[str], str]] = [
            (["cost_reduce", "yield_deploy", "spawn_child"], "recovery_via_diversification"),
            (["yield_rebalance", "protocol_explore", "consolidate"], "yield_optimization"),
            (["bounty_hunt", "asset_create", "liquidate"], "bounty_then_diversify"),
        ]

        # Group economic datapoints by episode root (first segment of episode_id)
        # and by timestamp proximity (within 72 hours)
        economic_types = {
            "oikos", "economic", "yield", "bounty", "asset", "spawn",
            "cost_reduce", "yield_deploy", "yield_rebalance", "protocol_explore",
            "consolidate", "bounty_hunt", "asset_create", "liquidate", "spawn_child",
        }

        def _is_economic(dp: RETrainingDatapoint) -> bool:
            return (
                dp.example_type in economic_types
                or dp.example_type.startswith(("oikos.", "economic."))
                or any(kw in dp.example_type for kw in ("yield", "bounty", "asset", "spawn"))
            )

        econ_dps = [dp for dp in datapoints if _is_economic(dp)]
        if len(econ_dps) < 2:
            return

        # Sort by timestamp for sequence detection
        econ_dps_sorted = sorted(econ_dps, key=lambda dp: dp.timestamp)

        # Build 72-hour windows
        window_seconds = 72 * 3600

        # For each pair, check if any known pattern matches the sequence
        def _action_key(dp: RETrainingDatapoint) -> str:
            return dp.example_type.split(".")[-1].lower()

        def _ts_seconds(dp: RETrainingDatapoint) -> float:
            return dp.timestamp.timestamp()

        for start_idx, start_dp in enumerate(econ_dps_sorted):
            t0 = _ts_seconds(start_dp)
            # Collect all dps within 72h window from start_dp
            window = [
                dp for dp in econ_dps_sorted[start_idx:]
                if _ts_seconds(dp) - t0 <= window_seconds
            ]
            if len(window) < 2:
                continue

            action_seq = [_action_key(dp) for dp in window]

            for pattern_steps, strategy_name in _STRATEGY_PATTERNS:
                # Check if all pattern steps appear in order within the window
                matched_indices: list[int] = []
                search_from = 0
                for step in pattern_steps:
                    for i in range(search_from, len(action_seq)):
                        if action_seq[i] == step or step in action_seq[i]:
                            matched_indices.append(i)
                            search_from = i + 1
                            break

                if len(matched_indices) == len(pattern_steps):
                    # Strategy detected - annotate the matched datapoints
                    matched_dps = [window[i] for i in matched_indices]
                    t_start = _ts_seconds(matched_dps[0])
                    t_end = _ts_seconds(matched_dps[-1])
                    duration_hours = (t_end - t_start) / 3600.0

                    for step_num, dp in enumerate(matched_dps, start=1):
                        dp.is_strategic = True
                        dp.strategy_name = strategy_name
                        dp.strategy_step_number = step_num
                        dp.strategy_total_steps = len(matched_dps)
                        dp.strategy_duration_hours = round(duration_hours, 2)
                        # 1.3× priority boost - surface in curriculum
                        dp.task_difficulty = min(1.0, dp.task_difficulty * 1.3)

                    logger.info(
                        "re_training_strategy_detected",
                        strategy=strategy_name,
                        steps=len(matched_dps),
                        duration_hours=round(duration_hours, 2),
                    )
                    break  # Only assign one strategy per window start

    def _enrich_batch(self, datapoints: list[RETrainingDatapoint]) -> None:
        """
        Mutate datapoints in-place before export:
        1. Compute task_difficulty
        2. Validate scaffold compliance
        3. Assign quality_tier
        4. Apply metabolic boost to economic examples during starvation
        5. Detect multi-step economic strategies (with 1.3× difficulty boost)
        """
        validate = _get_scaffold_validator()

        for dp in datapoints:
            dp.task_difficulty = self._compute_task_difficulty(dp)
            scaffold_valid = False
            if validate is not None:
                try:
                    # Pass as dict so validate() can read reasoning_trace
                    scaffold_valid = validate({"reasoning_trace": dp.reasoning_trace})
                except Exception:
                    pass
            dp.quality_tier = self._assign_quality_tier(dp, scaffold_valid)

        # Apply metabolic boost: during starvation, economic examples are
        # the most urgent curriculum content - surface them by boosting confidence
        # (which acts as a priority proxy for curriculum building).
        if self._metabolic_boost > 1.0:
            _economic_keywords = ("economic", "oikos", "yield", "bounty", "asset", "spawn")
            for dp in datapoints:
                if any(kw in dp.example_type for kw in _economic_keywords):
                    dp.confidence = min(1.0, dp.confidence * self._metabolic_boost)

        # Run strategy detection after task_difficulty is set (boost applied inside)
        self._detect_multi_step_strategy(datapoints)

    # ─── Export Destinations ──────────────────────────────────────────

    async def export_to_s3(self, batch: RETrainingExportBatch) -> bool:
        """
        Export batch as JSON lines to S3 (or local filesystem fallback).

        Returns True if the export succeeded, False otherwise.
        S3 upload uses boto3 if available; falls back to local file write.
        """
        if not batch.datapoints:
            return True

        # Enrich datapoints with task_difficulty, scaffold validation, quality_tier
        self._enrich_batch(batch.datapoints)

        # Log quality tier distribution for monitoring
        tier_counts: dict[str, int] = {}
        for dp in batch.datapoints:
            tier_counts[dp.quality_tier] = tier_counts.get(dp.quality_tier, 0) + 1
        updated_count = sum(1 for dp in batch.datapoints if dp.outcome_updated)
        logger.info(
            "re_training_quality_tier_distribution",
            batch_id=batch.id,
            gold=tier_counts.get("gold", 0),
            silver=tier_counts.get("silver", 0),
            bronze=tier_counts.get("bronze", 0),
            outcome_corrected=updated_count,
            total=batch.total_examples,
        )

        # cost_usd and latency_ms are included via model_dump (needed for curriculum learning)
        lines = "\n".join(
            json.dumps(dp.model_dump(mode="json"), default=str)
            for dp in batch.datapoints
        )
        filename = f"{batch.hour_window.replace(':', '-')}_{batch.id}.jsonl"

        # Try S3 first
        try:
            import boto3  # type: ignore[import]

            s3 = boto3.client("s3")
            key = f"{_S3_PREFIX}{filename}"
            s3.put_object(
                Bucket=_S3_BUCKET,
                Key=key,
                Body=lines.encode("utf-8"),
                ContentType="application/x-ndjson",
            )
            batch.export_destinations.append(f"s3://{_S3_BUCKET}/{key}")
            logger.info(
                "re_training_exported_s3",
                batch_id=batch.id,
                examples=batch.total_examples,
                key=key,
            )
            return True
        except ImportError:
            logger.debug("boto3_not_available_falling_back_to_local")
        except Exception:
            logger.warning("re_training_s3_export_failed", exc_info=True)

        # Local filesystem fallback
        try:
            export_dir = Path(_LOCAL_EXPORT_DIR)
            export_dir.mkdir(parents=True, exist_ok=True)
            dest = export_dir / filename
            dest.write_text(lines, encoding="utf-8")
            batch.export_destinations.append(f"local://{dest}")
            logger.info(
                "re_training_exported_local",
                batch_id=batch.id,
                examples=batch.total_examples,
                path=str(dest),
            )
            return True
        except Exception:
            logger.error("re_training_local_export_failed", exc_info=True)
            return False

    async def sync_to_memory(self, batch: RETrainingExportBatch) -> None:
        """
        Write batch lineage to Neo4j as (:RETrainingBatch) + (:RETrainingDatapoint)
        nodes.  Non-fatal - failures are logged and swallowed.
        """
        if self._neo4j is None or not batch.datapoints:
            return
        try:
            await self._neo4j.execute_write(
                """
                MERGE (b:RETrainingBatch {id: $batch_id})
                SET b.hour_window     = $hour_window,
                    b.total_examples  = $total_examples,
                    b.mean_quality    = $mean_quality,
                    b.source_systems  = $source_systems,
                    b.destinations    = $destinations,
                    b.created_at      = $created_at
                """,
                {
                    "batch_id": batch.id,
                    "hour_window": batch.hour_window,
                    "total_examples": batch.total_examples,
                    "mean_quality": round(batch.mean_quality, 4),
                    "source_systems": batch.source_systems,
                    "destinations": batch.export_destinations,
                    "created_at": batch.created_at.isoformat(),
                },
            )
            # Write per-source summary nodes
            for system in batch.source_systems:
                system_dps = [dp for dp in batch.datapoints if dp.source_system == system]
                mean_q = sum(dp.confidence for dp in system_dps) / len(system_dps)
                await self._neo4j.execute_write(
                    """
                    MATCH (b:RETrainingBatch {id: $batch_id})
                    MERGE (s:RETrainingSource {system: $system, batch_id: $batch_id})
                    SET s.count      = $count,
                        s.mean_quality = $mean_quality
                    MERGE (b)-[:CONTAINS_SOURCE]->(s)
                    """,
                    {
                        "batch_id": batch.id,
                        "system": system,
                        "count": len(system_dps),
                        "mean_quality": round(mean_q, 4),
                    },
                )

            # Write individual datapoints to Neo4j for autonomous training queries.
            # Batched via UNWIND to avoid N separate writes.
            dp_rows = [
                {
                    "id": dp.id,
                    "source_system": dp.source_system,
                    "example_type": dp.example_type,
                    "instruction": dp.instruction[:2000],
                    "input_context": dp.input_context[:4000],
                    "output_action": dp.output_action[:2000],
                    "outcome": dp.outcome,
                    "confidence": round(dp.confidence, 4),
                    "reasoning_trace": dp.reasoning_trace[:4000],
                    "alternatives": dp.alternatives_considered[:5],
                    "coherence": round(dp.constitutional_alignment.coherence, 4)
                    if dp.constitutional_alignment else 0.0,
                    "care": round(dp.constitutional_alignment.care, 4)
                    if dp.constitutional_alignment else 0.0,
                    "growth": round(dp.constitutional_alignment.growth, 4)
                    if dp.constitutional_alignment else 0.0,
                    "honesty": round(dp.constitutional_alignment.honesty, 4)
                    if dp.constitutional_alignment else 0.0,
                    "cost_usd": str(dp.cost_usd),
                    "latency_ms": dp.latency_ms,
                    "episode_id": dp.episode_id,
                    "timestamp": dp.timestamp.isoformat(),
                    "batch_id": batch.id,
                    "outcome_updated": dp.outcome_updated,
                    "actual_outcome_quality": dp.actual_outcome_quality,
                    "quality_tier": dp.quality_tier,
                    "task_difficulty": dp.task_difficulty,
                    "is_strategic": dp.is_strategic,
                    "strategy_name": dp.strategy_name,
                    "strategy_step_number": dp.strategy_step_number,
                    "strategy_total_steps": dp.strategy_total_steps,
                    "strategy_duration_hours": dp.strategy_duration_hours,
                }
                for dp in batch.datapoints
            ]
            await self._neo4j.execute_write(
                """
                UNWIND $rows AS r
                MERGE (d:RETrainingDatapoint {id: r.id})
                SET d.source_system          = r.source_system,
                    d.example_type           = r.example_type,
                    d.instruction            = r.instruction,
                    d.input_context          = r.input_context,
                    d.output_action          = r.output_action,
                    d.outcome                = r.outcome,
                    d.confidence             = r.confidence,
                    d.reasoning_trace        = r.reasoning_trace,
                    d.alternatives           = r.alternatives,
                    d.coherence              = r.coherence,
                    d.care                   = r.care,
                    d.growth                 = r.growth,
                    d.honesty                = r.honesty,
                    d.cost_usd               = r.cost_usd,
                    d.latency_ms             = r.latency_ms,
                    d.episode_id             = r.episode_id,
                    d.timestamp              = r.timestamp,
                    d.outcome_updated        = r.outcome_updated,
                    d.actual_outcome_quality = r.actual_outcome_quality,
                    d.quality_tier           = r.quality_tier,
                    d.task_difficulty        = r.task_difficulty,
                    d.is_strategic           = r.is_strategic,
                    d.strategy_name          = r.strategy_name,
                    d.strategy_step_number   = r.strategy_step_number,
                    d.strategy_total_steps   = r.strategy_total_steps,
                    d.strategy_duration_hours = r.strategy_duration_hours
                WITH d, r
                MATCH (b:RETrainingBatch {id: r.batch_id})
                MERGE (b)-[:CONTAINS_DATAPOINT]->(d)
                """,
                {"rows": dp_rows},
            )
            logger.info(
                "re_training_synced_to_neo4j",
                batch_id=batch.id,
                examples=batch.total_examples,
            )
        except Exception:
            logger.warning("re_training_neo4j_sync_failed", exc_info=True)

    # ─── Export Broadcast ────────────────────────────────────────────

    async def _emit_export_complete(self, batch: RETrainingExportBatch) -> None:
        """Emit RE_TRAINING_EXPORT_COMPLETE on the Synapse bus."""
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXPORT_COMPLETE,
                source_system="re_training_exporter",
                data={
                    "batch_id": batch.id,
                    "total_examples": batch.total_examples,
                    "source_systems": batch.source_systems,
                    "mean_quality": round(batch.mean_quality, 4),
                    "export_destinations": batch.export_destinations,
                    "export_duration_ms": batch.export_duration_ms,
                    "hour_window": batch.hour_window,
                },
            ))
        except Exception:
            logger.debug("re_training_export_emit_failed", exc_info=True)

    # ─── Main Export Cycle ────────────────────────────────────────────

    async def export_cycle(self) -> RETrainingExportBatch:
        """
        Run a single collect → export → sync cycle.

        Called once per hour by run_loop(). Safe to call directly for testing.
        """
        t0 = time.monotonic()
        batch = self.collect_batch()

        if batch.total_examples == 0:
            logger.debug("re_training_export_skipped_empty_window", hour=batch.hour_window)
            return batch

        await self.export_to_s3(batch)
        await self.sync_to_memory(batch)

        batch.export_duration_ms = int((time.monotonic() - t0) * 1000)
        self._total_exported += batch.total_examples
        self._total_batches += 1

        await self._emit_export_complete(batch)

        logger.info(
            "re_training_export_cycle_complete",
            batch_id=batch.id,
            examples=batch.total_examples,
            systems=batch.source_systems,
            mean_quality=round(batch.mean_quality, 4),
            duration_ms=batch.export_duration_ms,
            total_exported=self._total_exported,
        )
        return batch

    # ─── Background Loop ──────────────────────────────────────────────

    async def run_loop(self) -> None:
        """
        Supervised background coroutine.  Runs indefinitely, exporting once per hour.

        To be wrapped in utils.supervision.supervised_task() by registry.py.
        """
        logger.info(
            "re_training_export_loop_started",
            interval_s=_EXPORT_INTERVAL_S,
            s3_bucket=_S3_BUCKET,
        )
        while True:
            await asyncio.sleep(_EXPORT_INTERVAL_S)
            await self.export_cycle()

    # ─── Stats ───────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        outcome_updated = sum(1 for dp in self._pending if dp.outcome_updated)
        return {
            "pending_examples": len(self._pending),
            "total_exported": self._total_exported,
            "total_batches": self._total_batches,
            "window_start": self._window_start.isoformat(),
            "seen_episode_ids": len(self._seen_episode_ids),
            "pending_outcome_corrected": outcome_updated,
            "attached": self._attached,
        }
