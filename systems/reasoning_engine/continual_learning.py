"""
EcodiaOS - Continual Learning Orchestrator (Speciation Bible §3 + §7)

Three-tier architecture:
  Tier 1: Real-time Thompson sampling (DONE - Nova/PolicyGenerator)
  Tier 2: Incremental LoRA training    (THIS FILE - ~2-week cadence)
  Tier 3: Quarterly full retrain       (THIS FILE - placeholder, not yet triggered)

This orchestrator ties the historical training pipeline together:

  Neo4j streams → quality scoring → scaffold formatting → JSONL export
       ↓
  asyncio subprocess: train_lora.py (Unsloth + LoRA, Qwen3-8B base)
       ↓
  Safety layer (Speciation Bible §7):
    0. RESuccessRateMonitor.check_kill_switch()  - halt if RE rate < 0.50 (7d)
       ↓
  Anti-forgetting stack (Speciation Bible §3.3):
    1. SurprisePrioritizedReplay  - mix replay buffer into training JSONL
    2. SuReEMAAdapter             - merge fast adapter into slow EMA adapter
    2b. SafeLoRAProjection        - project onto safety-aligned subspace (§7.2)
    3. STABLEKLGate               - KL divergence gate before deployment
    4. AnchorPerplexityMonitor    - post-deploy forgetting alarm (non-blocking)
       ↓
  ReasoningEngineService.load_adapter() - hot-swap SLOW adapter into vLLM

No cross-system imports. Synapse events only. Organism continues on
Claude-only mode if training fails.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from systems.reasoning_engine.service import ReasoningEngineService
    from systems.reasoning_engine.training_data_extractor import TrainingDataExtractor

from systems.reasoning_engine.anti_forgetting import (
    AntiForgetConfig,
    AnchorPerplexityMonitor,
    STABLEKLGate,
    SurprisePrioritizedReplay,
    SuReEMAAdapter,
)
from systems.reasoning_engine.dpo_pipeline import (
    ConstitutionalJudge,
    DPOConfig,
    DPOTrainer,
    PreferencePairGenerator,
)
from systems.reasoning_engine.safety import (
    RESuccessRateMonitor,
    SafeLoRAProjection,
    SafetyConfig,
)
from systems.reasoning_engine.tier3 import Tier3Config, Tier3Orchestrator
from systems.reasoning_engine.training_exclusions import TrainingExclusionFilter

logger = structlog.get_logger("reasoning_engine.continual_learning")

# ── Environment defaults ───────────────────────────────────────────────────────

_DEFAULT_EXPORT_DIR = os.environ.get("RE_TRAINING_EXPORT_DIR", "data/re_training_batches")
_TRAIN_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "..", "systems", "simula", "training", "train_lora.py"
)
_DEFAULT_BASE_MODEL = os.environ.get("RE_BASE_MODEL", "Qwen/Qwen3-8B")
_TRAINING_TIMEOUT_S = int(os.environ.get("RE_TRAINING_TIMEOUT_S", "7200"))  # 2 hours

# S3 adapter bridge - inference pod polls this prefix for new adapters
_S3_ADAPTER_BUCKET = os.environ.get("RE_ADAPTER_S3_BUCKET", "ecodiaos-re-training")
_S3_ADAPTER_PREFIX = os.environ.get("RE_ADAPTER_S3_PREFIX", "adapters/production/")
_INSTANCE_ID = os.environ.get("INSTANCE_ID", "genesis")

# Redis keys
_REDIS_KEY_LAST_TRAIN = "eos:re:last_train_at"
_REDIS_KEY_RUNS = "eos:re:training_runs"
_REDIS_KEY_THOMPSON_SCORE = "eos:re:thompson_success_rate"
_TRAINING_HALTED_KEY = "eos:re:training_halted"
_REDIS_KEY_PRE_DEPLOY_BASELINE = "eos:re:pre_deploy_baseline"

# Post-deployment quality monitoring
_POST_DEPLOY_MONITOR_CYCLES = 500   # window length (RE decisions)
_ROLLBACK_DEGRADATION_THRESHOLD = 0.90   # rollback if post < pre * 0.90
_CONFIRM_IMPROVEMENT_THRESHOLD = 1.05    # confirm if post > pre * 1.05


# ── Data structures ────────────────────────────────────────────────────────────


@dataclass
class TrainingTrigger:
    """Thresholds that control when a training run fires."""

    min_new_examples: int = 300
    """Minimum new Neo4j examples since last train to justify Tier 2."""

    max_days_since_train: int = 14
    """Force Tier 2 if this many days pass without training regardless of data volume."""

    performance_drop_threshold: float = 0.05
    """Thompson sampler success-rate drop that triggers emergency Tier 2."""

    full_retrain_interval_days: int = 90
    """Days between Tier 3 full retrains (future use - not yet triggered)."""

    min_viable_examples: int = 50
    """Minimum examples required to actually start a training run."""


@dataclass
class TrainingRun:
    """Immutable record of one completed (or failed) training run."""

    run_id: str
    tier: int  # 2 = incremental LoRA; 3 = full retrain (future)
    trigger_reason: str
    examples_used: int
    started_at: datetime
    completed_at: datetime | None = None
    adapter_path: str | None = None
    eval_loss: float | None = None
    deployed: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tier": self.tier,
            "trigger_reason": self.trigger_reason,
            "examples_used": self.examples_used,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "adapter_path": self.adapter_path,
            "eval_loss": self.eval_loss,
            "deployed": self.deployed,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingRun":
        return cls(
            run_id=d["run_id"],
            tier=d["tier"],
            trigger_reason=d["trigger_reason"],
            examples_used=d["examples_used"],
            started_at=datetime.fromisoformat(d["started_at"]),
            completed_at=datetime.fromisoformat(d["completed_at"]) if d.get("completed_at") else None,
            adapter_path=d.get("adapter_path"),
            eval_loss=d.get("eval_loss"),
            deployed=d.get("deployed", False),
            error=d.get("error", ""),
        )


# ── Orchestrator ───────────────────────────────────────────────────────────────


class ContinualLearningOrchestrator:
    """
    Manages the organism's self-improvement cycle.

    Call `initialize()` on startup to restore history from Redis.
    Call `check_and_train()` from a daily background task.
    Call `run_tier2(reason)` to force a training run immediately.
    """

    def __init__(
        self,
        re_service: "ReasoningEngineService",
        extractor: "TrainingDataExtractor",
        config: TrainingTrigger | None = None,
        anti_forget_config: AntiForgetConfig | None = None,
        safety_config: SafetyConfig | None = None,
        dpo_config: DPOConfig | None = None,
        claude_client: Any = None,
        memory: Any = None,
        equor_service: Any = None,
    ) -> None:
        self._re = re_service
        self._extractor = extractor
        self._config = config or TrainingTrigger()
        self._af_config = anti_forget_config or AntiForgetConfig()
        self._safety_config = safety_config or SafetyConfig()
        self._dpo_config = dpo_config or DPOConfig()
        self._redis: "Redis | None" = None
        self._event_bus: Any = None
        self._last_train_at: datetime | None = None
        self._training_runs: list[TrainingRun] = []
        self._lock = asyncio.Lock()
        self._current_adapter_path: str | None = None
        self._pending_dpo_adapter: str | None = None
        # Cross-instance adapter share (Share 2025): takes priority over DPO adapter
        # as BASE_ADAPTER in the next Tier 2 run.
        self._pending_shared_adapter: str | None = None
        self._training_halted: bool = False  # in-memory cache; source of truth is Redis

        # ── Post-deployment quality monitoring ─────────────────────────────────
        # Snapshot captured immediately before each adapter deploy.
        # Format: {"success_rate": float, "eval_loss": float | None, "cycle": str,
        #          "timestamp": str, "adapter_path": str | None}
        self._pre_deploy_baseline: dict[str, Any] | None = None
        # Previous adapter path - restored on rollback.
        self._pre_deploy_adapter_path: str | None = None
        # Counters for the post-deploy monitoring window.
        self._post_deploy_successes: int = 0
        self._post_deploy_attempts: int = 0
        # Flag: True while monitoring window is open.
        self._monitoring_active: bool = False
        # Urgent training request from Evo/Nova via RE_TRAINING_REQUESTED event.
        # When set, should_train() lowers the min_examples gate to 50 (vs 300 normal)
        # and returns True immediately if examples ≥ 50. Cleared after training starts.
        self._urgent_training_requested: bool = False

        # Anti-forgetting components - replay/perplexity get Redis injected in set_redis()
        self._sure = SuReEMAAdapter(self._af_config)
        self._stable = STABLEKLGate(self._af_config)
        # Replay and perplexity monitor require Redis; placeholders set here, initialised in set_redis()
        self._replay: SurprisePrioritizedReplay | None = None
        self._perplexity_monitor: AnchorPerplexityMonitor | None = None

        # Safety layer (§7) - RE success rate monitor + SafeLoRA projection
        self._re_monitor = RESuccessRateMonitor(self._safety_config)
        self._safe_lora = SafeLoRAProjection(self._safety_config)

        # DPO constitutional alignment pipeline (§7.2 speciation bible)
        # claude_client is optional - judge falls back to heuristic if None
        _judge = ConstitutionalJudge(self._dpo_config, claude_client)
        self._dpo_trainer = DPOTrainer(self._dpo_config, re_service, None)  # bus wired in set_event_bus
        self._pair_gen = PreferencePairGenerator(self._dpo_config, memory, equor_service, _judge)

        # Tier 3 - quarterly full retrain with SVD pruning + SLAO merge
        # Redis and event_bus are injected later via set_redis() / set_event_bus()
        # so Tier3Orchestrator is created lazily in set_redis() once Redis is available.
        self._tier3: Tier3Orchestrator | None = None

        # Training exclusion filter - protects evaluation files from entering training data
        self._exclusion_filter = TrainingExclusionFilter()

        # Ablation mode - set by AblationOrchestrator before calling run_tier2().
        # "none" = full stack (default); see ablation.py:AblationMode for valid values.
        # MUST be cleared back to "none" in AblationOrchestrator._train_ablated() finally block.
        self._ablation_mode: str = "none"

    def set_redis(self, redis: "Redis") -> None:
        self._redis = redis
        # Instantiate Redis-backed anti-forgetting components now that Redis is available
        self._replay = SurprisePrioritizedReplay(
            self._af_config, redis, exclusion_filter=self._exclusion_filter
        )
        self._perplexity_monitor = AnchorPerplexityMonitor(self._af_config, redis)
        # Wire Redis into safety monitor
        self._re_monitor.set_redis(redis)
        # Instantiate Tier 3 orchestrator (requires Redis for timestamp persistence)
        self._tier3 = Tier3Orchestrator(
            config=Tier3Config(
                retrain_interval_days=self._config.full_retrain_interval_days,
                svd_prune_top_k=self._af_config.svd_prune_top_k,
            ),
            af_config=self._af_config,
            re_service=self._re,
            event_bus=self._event_bus,
            redis_client=redis,
        )

    def set_event_bus(self, bus: Any) -> None:
        self._event_bus = bus
        self._dpo_trainer._bus = bus
        if self._tier3 is not None:
            self._tier3._bus = bus
        # Adapter sharing - subscribe to cross-instance events
        try:
            from systems.synapse.types import SynapseEventType as _SET
            bus.subscribe(_SET.ADAPTER_SHARE_REQUEST, self._on_adapter_share_request)
            bus.subscribe(_SET.ADAPTER_SHARE_OFFER, self._on_adapter_share_offer)
        except Exception as exc:
            logger.warning("continual_learning.adapter_share_subscribe_failed", error=str(exc))
        # Urgent retraining requests from Evo/Nova
        try:
            from systems.synapse.types import SynapseEventType
            bus.subscribe(SynapseEventType.RE_TRAINING_REQUESTED, self._on_re_training_requested)
        except Exception as exc:
            logger.warning("continual_learning.re_training_requested_subscribe_failed", error=str(exc))

    async def _on_adapter_share_request(self, event: Any) -> None:
        """Partner instance is requesting our adapter path - respond if we have one.

        Non-fatal: any failure is logged silently. Partner will time out after 30s.
        """
        try:
            data = getattr(event, "data", {}) or {}
            my_id = os.getenv("INSTANCE_ID", "genesis")
            if data.get("target_instance_id") != my_id:
                return
            adapter_path = self._sure.production_adapter_path or ""
            if self._event_bus is not None:
                from systems.synapse.types import SynapseEventType as _SET
                await self._event_bus.emit(
                    _SET.ADAPTER_SHARE_RESPONSE,
                    {
                        "request_id": data.get("request_id"),
                        "instance_id": my_id,
                        "adapter_path": adapter_path,
                    },
                )
        except Exception as exc:
            logger.warning("continual_learning.adapter_share_request_failed", error=str(exc))

    async def _on_adapter_share_offer(self, event: Any) -> None:
        """A merged adapter has been offered to us.

        Store as _pending_shared_adapter - it will be used as BASE_ADAPTER on
        the next Tier 2 run with priority over _pending_dpo_adapter.
        The offer is always accepted; a confidence threshold could be added later.
        """
        try:
            data = getattr(event, "data", {}) or {}
            my_id = os.getenv("INSTANCE_ID", "genesis")
            if my_id not in (data.get("target_instances") or []):
                return
            merged_path = data.get("merged_adapter_path", "")
            if not merged_path:
                return
            self._pending_shared_adapter = merged_path
            logger.info(
                "adapter_share.offer_accepted",
                merged_path=merged_path,
                genome_distance=data.get("genome_distance"),
                kl_divergence=data.get("kl_divergence"),
            )
        except Exception as exc:
            logger.warning("continual_learning.adapter_share_offer_failed", error=str(exc))

    async def _on_re_training_requested(self, event: Any) -> None:
        """Handle RE_TRAINING_REQUESTED from Evo or Nova.

        Sets _urgent_training_requested = True so that the next should_train()
        call will fire with a lowered min_examples threshold (50 instead of 300).
        The flag is cleared inside should_train() after returning True, which
        prevents repeated firing on the same urgency signal.
        """
        try:
            data = getattr(event, "data", {}) or {}
            source = data.get("source_system", "unknown")
            kpi = data.get("kpi", "unknown")
            urgency = data.get("urgency", "warning")
            self._urgent_training_requested = True
            logger.warning(
                "re_training_requested_received",
                source=source,
                kpi=kpi,
                urgency=urgency,
            )
        except Exception as exc:
            logger.warning("continual_learning.re_training_requested_failed", error=str(exc))

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Restore last_train_at, training run history, and anti-forgetting state from Redis."""
        if self._redis is None:
            logger.info("continual_learning_no_redis_fresh_start")
        else:
            try:
                raw_ts = await self._redis.get(_REDIS_KEY_LAST_TRAIN)
                if raw_ts:
                    self._last_train_at = datetime.fromisoformat(raw_ts.decode())
                    logger.info("continual_learning_restored_last_train", last_train=self._last_train_at.isoformat())
            except Exception as exc:
                logger.warning("continual_learning_restore_last_train_failed", error=str(exc))

            try:
                raw_runs = await self._redis.get(_REDIS_KEY_RUNS)
                if raw_runs:
                    runs_data = json.loads(raw_runs.decode())
                    self._training_runs = [TrainingRun.from_dict(r) for r in runs_data]
                    logger.info("continual_learning_restored_runs", count=len(self._training_runs))
            except Exception as exc:
                logger.warning("continual_learning_restore_runs_failed", error=str(exc))

            if self._replay is not None:
                await self._replay.restore_from_redis()

        # Load anchor prompts for STABLE KL gate (non-fatal if file absent)
        await self._stable.load_anchors(
            anchor_file=str(Path(_DEFAULT_EXPORT_DIR) / "anchor_prompts.jsonl")
        )

        # Load training exclusion filter - non-fatal; training proceeds without it on failure
        try:
            await self._exclusion_filter.load()
        except Exception as exc:
            logger.warning("training_exclusion_filter_load_failed", error=str(exc))

        # Restore training halt flag from Redis (survives restarts)
        if self._redis is not None:
            try:
                halted, reason = await self._is_training_halted()
                if halted:
                    self._training_halted = True
                    logger.critical("training.halted_restored_from_redis", reason=reason)
            except Exception as exc:
                logger.warning("training_halt_restore_failed", error=str(exc))

    # ── Trigger logic ──────────────────────────────────────────────────────────

    async def should_train(self) -> tuple[bool, str]:
        """
        Check if a training run should be triggered.

        Priority order:
          1. Tier 3 (quarterly full retrain) - future placeholder
          2. Tier 2: data volume threshold exceeded
          3. Tier 2: max days since last train
          4. Tier 2: Thompson sampler performance degradation
          5. No trigger

        Returns (should_train, reason_string).
        """
        # ── Safety kill switch (§7.3 Tier 2) ──────────────────────────────────
        # Check persisted halt flag first - survives restarts (set by red-team or operator).
        try:
            halted, halt_reason = await self._is_training_halted()
            if halted:
                logger.critical("training.halted", reason=halt_reason)
                return False, f"halted:{halt_reason}"
        except Exception as exc:
            logger.warning("continual_learning_halt_check_failed", error=str(exc))

        # Then check RE success rate kill switch.
        try:
            if await self._re_monitor.check_kill_switch(self._event_bus):
                logger.critical("training.halted", reason="re_success_rate_below_floor")
                await self._set_training_halted("re_success_rate_below_floor")
                return False, "halted_re_success_rate"
        except Exception as exc:
            logger.warning("continual_learning_safety_check_failed", error=str(exc))

        now = datetime.now(UTC)

        # 1. Tier 3 quarterly check - full retrain with SVD pruning + SLAO merge
        if self._tier3 is not None:
            try:
                tier3_ready, tier3_reason = await self._tier3.should_run_tier3()
                if tier3_ready:
                    logger.info("continual_learning_tier3_triggered", reason=tier3_reason)
                    return True, "tier3_quarterly"
            except Exception as exc:
                logger.warning("continual_learning_tier3_check_failed", error=str(exc))

        # 1b. Urgent retraining request (RE_TRAINING_REQUESTED) - lowered threshold
        if self._urgent_training_requested:
            try:
                counts = await self._extractor.stream_counts(lookback_days=self._config.max_days_since_train)
                total_new = sum(v for v in counts.values() if v >= 0)
                if total_new >= self._config.min_viable_examples:  # 50, not 300
                    logger.warning(
                        "continual_learning_trigger_urgent",
                        total_new=total_new,
                        urgent_threshold=self._config.min_viable_examples,
                    )
                    self._urgent_training_requested = False
                    return True, "tier2_urgent_requested"
            except Exception as exc:
                logger.warning("continual_learning_urgent_count_failed", error=str(exc))

        # 2. New example count
        try:
            counts = await self._extractor.stream_counts(lookback_days=self._config.max_days_since_train)
            total_new = sum(v for v in counts.values() if v >= 0)
            if total_new >= self._config.min_new_examples:
                logger.info("continual_learning_trigger_data_volume", total_new=total_new)
                return True, "tier2_data_volume"
        except Exception as exc:
            logger.warning("continual_learning_stream_count_failed", error=str(exc))

        # 3. Max days since last train
        if self._last_train_at is not None:
            days_since = (now - self._last_train_at).days
            if days_since >= self._config.max_days_since_train:
                logger.info("continual_learning_trigger_scheduled", days_since=days_since)
                return True, "tier2_scheduled"
        elif self._last_train_at is None:
            # Never trained - trigger immediately if any data exists
            try:
                counts = await self._extractor.stream_counts()
                total = sum(v for v in counts.values() if v >= 0)
                if total >= self._config.min_viable_examples:
                    return True, "tier2_first_run"
            except Exception:
                pass

        # 4. Thompson sampler performance drop
        if self._redis is not None:
            try:
                raw_score = await self._redis.get(_REDIS_KEY_THOMPSON_SCORE)
                if raw_score:
                    score = float(raw_score.decode())
                    # Score is stored as current success rate; compare to 0.5 baseline
                    if score < (0.5 - self._config.performance_drop_threshold):
                        logger.info("continual_learning_trigger_degradation", thompson_score=score)
                        return True, "tier2_degradation"
            except Exception as exc:
                logger.debug("continual_learning_thompson_check_failed", error=str(exc))

        return False, "no_trigger"

    # ── Tier 2 training run ────────────────────────────────────────────────────

    async def run_tier2(self, reason: str) -> TrainingRun:
        """
        Execute an incremental LoRA training run (Tier 2), or route to Tier 3
        if the trigger reason is "tier3_quarterly" or "tier3_forgetting".

        Steps:
          1. Extract training data from Neo4j (30-day lookback, min_score=0.30)
          2. Gate on minimum viable dataset (50 examples)
          3. Write JSONL to RE_TRAINING_EXPORT_DIR
          4. Emit RE_TRAINING_STARTED
          5. Launch train_lora.py subprocess (timeout: 2h)
          6. On success: load_adapter(), emit RE_TRAINING_COMPLETE
          7. On failure: log, emit RE_TRAINING_FAILED (organism continues)

        Returns the TrainingRun record regardless of outcome.
        """
        # ── Tier 3 routing ─────────────────────────────────────────────────────
        # If the trigger is Tier 3 quarterly, route to the full retrain pipeline
        # instead of the incremental LoRA update.
        if "tier3" in reason and self._tier3 is not None:
            tier3_run = TrainingRun(
                run_id=f"tier3_{datetime.now(UTC):%Y%m%d_%H%M%S}",
                tier=3,
                trigger_reason=reason,
                examples_used=0,
                started_at=datetime.now(UTC),
            )
            try:
                cumulative_data = await self._build_cumulative_dataset()
                success = await self._tier3.run_tier3(
                    cumulative_data_path=cumulative_data,
                    slow_adapter_path=self._sure.production_adapter_path,
                )
                tier3_run.completed_at = datetime.now(UTC)
                tier3_run.deployed = success
                if success and self._tier3 is not None:
                    # Update current adapter path to the Tier 3 output
                    self._last_train_at = tier3_run.completed_at
            except Exception as exc:
                logger.error("continual_learning_tier3_error", error=str(exc))
                tier3_run.error = str(exc)
                tier3_run.completed_at = datetime.now(UTC)
            self._training_runs.append(tier3_run)
            await self._persist_state()
            return tier3_run

        async with self._lock:
            run = TrainingRun(
                run_id=f"tier2_{datetime.now(UTC):%Y%m%d_%H%M%S}",
                tier=2,
                trigger_reason=reason,
                examples_used=0,
                started_at=datetime.now(UTC),
            )
            logger.info("continual_learning_tier2_start", run_id=run.run_id, reason=reason)

            try:
                run = await self._execute_tier2(run)
            except Exception as exc:
                # Belt-and-suspenders: _execute_tier2 should handle all exceptions
                # internally, but we never let this coroutine crash.
                logger.error(
                    "continual_learning_tier2_unexpected_error",
                    run_id=run.run_id,
                    error=str(exc),
                    exc_info=True,
                )
                run.error = str(exc)
                run.completed_at = datetime.now(UTC)
                from systems.synapse.types import SynapseEventType as _SET
                await self._emit(
                    _SET.RE_TRAINING_FAILED,
                    {"run_id": run.run_id, "tier": 2, "reason": str(exc)},
                )

            self._training_runs.append(run)
            await self._persist_state()

            # ── DPO background pass (non-blocking) ────────────────────────────
            # After each Tier 2 cycle, generate preference pairs and run DPO if
            # enough pairs have accumulated. Does NOT block the return of run.
            # DPO adapter feeds into the next SuRe EMA cycle as _pending_dpo_adapter.
            asyncio.ensure_future(self._run_dpo_background())

            return run

    async def _run_dpo_background(self) -> None:
        """Generate DPO preference pairs and run DPO pass if threshold met.

        Non-blocking - runs as a background task after each Tier 2 cycle.
        DPO adapter is stored as _pending_dpo_adapter; NOT deployed here.
        Failures are always caught and logged - never propagated.
        """
        try:
            # Source 1: Neo4j constitutional pairs (Equor-approved vs Equor-flagged)
            constitutional_pairs = await self._pair_gen.generate_pairs_from_neo4j(limit=100)

            # Source 2: Red-team pairs (unsafe RE output vs Claude-authored refusal)
            # generate_pairs_from_red_team is called opportunistically; it is a no-op when
            # the red-team prompt file does not exist.
            red_team_pairs = await self._pair_gen.generate_pairs_from_red_team(
                self._re, limit=50
            )

            # Source 3: Reasoning quality pairs (deep causal reasoning vs shallow answer)
            # Anti-laziness DPO signal - teaches the model to prefer rigorous over surface reasoning.
            reasoning_quality_pairs = await self._pair_gen.generate_pairs_from_reasoning_quality(
                self._re, limit=100
            )

            all_pairs = constitutional_pairs + red_team_pairs + reasoning_quality_pairs
            await self._pair_gen.save_pairs(all_pairs)
            logger.info(
                "dpo.pairs_collected",
                constitutional=len(constitutional_pairs),
                red_team=len(red_team_pairs),
                reasoning_quality=len(reasoning_quality_pairs),
                total=len(all_pairs),
            )

            dpo_adapter = await self._dpo_trainer.run_dpo_pass(self._sure.production_adapter_path)
            if dpo_adapter:
                self._pending_dpo_adapter = dpo_adapter
                logger.info("dpo.adapter_ready", path=dpo_adapter)
        except Exception as exc:
            logger.warning("dpo.background_failed", error=str(exc))

    async def _execute_tier2(self, run: TrainingRun) -> TrainingRun:
        # ── Step 1: Extract training data ─────────────────────────────────────
        logger.info("continual_learning_extracting_data", run_id=run.run_id)
        from systems.reasoning_engine.export_pipeline import run_export

        # We need the Neo4j client from the extractor - access it directly
        neo4j = self._extractor._neo4j

        export_dir = Path(_DEFAULT_EXPORT_DIR)
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = str(export_dir / f"tier2_{run.run_id}_{timestamp}.jsonl")

        export_result = await run_export(
            neo4j=neo4j,
            output_path=output_path,
            lookback_days=30,
            min_score=0.30,
        )

        if export_result.error:
            logger.warning(
                "continual_learning_export_failed",
                run_id=run.run_id,
                error=export_result.error,
            )
            run.error = f"export failed: {export_result.error}"
            run.completed_at = datetime.now(UTC)
            from systems.synapse.types import SynapseEventType as _SET
            await self._emit(_SET.RE_TRAINING_FAILED, {"run_id": run.run_id, "tier": 2, "reason": run.error})
            return run

        run.examples_used = export_result.total_exported

        # ── Step 1b: Ablation stream filtering ────────────────────────────────
        # When running under ablation, strip examples from the excluded stream
        # out of the exported JSONL before entering training.
        # stream_2_off: remove failure+correction examples (stream_id == "2")
        # stream_4_off: remove causal-chain examples (stream_id == "4")
        _ablation_excluded = 0
        if self._ablation_mode in ("stream_2_off", "stream_4_off"):
            _target_stream = "2" if self._ablation_mode == "stream_2_off" else "4"
            _filtered_path = output_path + ".ablation_filtered"
            try:
                with open(output_path) as _fin, open(_filtered_path, "w") as _fout:
                    for _line in _fin:
                        try:
                            _ex = json.loads(_line)
                        except Exception:
                            _fout.write(_line)
                            continue
                        if str(_ex.get("stream_id", "")) == _target_stream:
                            _ablation_excluded += 1
                        else:
                            _fout.write(_line)
                output_path = _filtered_path
                run.examples_used = max(0, run.examples_used - _ablation_excluded)
                logger.info(
                    "ablation_stream_filtered",
                    mode=self._ablation_mode,
                    excluded=_ablation_excluded,
                    remaining=run.examples_used,
                )
            except Exception as _af_exc:
                logger.warning("ablation_stream_filter_failed", error=str(_af_exc))

        # ── Step 2: Minimum viable dataset gate ───────────────────────────────
        if run.examples_used < self._config.min_viable_examples:
            logger.warning(
                "continual_learning_insufficient_data",
                run_id=run.run_id,
                examples=run.examples_used,
                min=self._config.min_viable_examples,
            )
            run.error = f"only {run.examples_used} examples (min {self._config.min_viable_examples})"
            run.completed_at = datetime.now(UTC)
            # Not a hard failure - just skip training
            return run

        # ── Step 3: Resolve JSONL path ─────────────────────────────────────────
        # export_pipeline already wrote the file - resolve its local path
        jsonl_path = output_path
        if not Path(jsonl_path).exists():
            # Fallback: look for the path in export_result
            local_paths = [p for p in export_result.output_paths if p.startswith("local://")]
            if local_paths:
                jsonl_path = local_paths[0].removeprefix("local://")

        # ── Step 3b: Mix replay buffer into training JSONL ────────────────────
        # Prepend up to 300 high-surprise historical examples to prevent forgetting.
        # Bypassed when ablation mode is "replay_off" or "anti_forgetting_off".
        if self._replay is not None and self._ablation_mode not in ("replay_off", "anti_forgetting_off"):
            try:
                replay_examples = await self._replay.sample(300)
                if replay_examples:
                    replay_path = str(Path(jsonl_path).with_suffix(".with_replay.jsonl"))
                    with open(replay_path, "w") as f_out:
                        for ex in replay_examples:
                            # Replay examples are already slim dicts with "messages" key
                            f_out.write(json.dumps(ex) + "\n")
                        # Append the new examples from export
                        with open(jsonl_path) as f_in:
                            for line in f_in:
                                f_out.write(line)
                    jsonl_path = replay_path
                    run.examples_used += len(replay_examples)
                    logger.info(
                        "replay_mixed_into_training",
                        run_id=run.run_id,
                        replay_count=len(replay_examples),
                        total_examples=run.examples_used,
                    )
            except Exception as exc:
                logger.warning("replay_mix_failed", run_id=run.run_id, error=str(exc))

        # ── Step 4: Emit RE_TRAINING_STARTED ──────────────────────────────────
        await self._emit(
            "re_training_started",
            {
                "run_id": run.run_id,
                "tier": 2,
                "trigger_reason": run.trigger_reason,
                "examples_available": run.examples_used,
            },
        )

        # ── Step 5: Launch training subprocess ────────────────────────────────
        adapter_output_dir = str(export_dir / f"adapter_{run.run_id}")
        Path(adapter_output_dir).mkdir(parents=True, exist_ok=True)

        # Adaptive hyperparameters based on dataset size (bible §5)
        training_args = _get_training_config(run.examples_used)

        # BASE_ADAPTER priority: shared (genetic recombination) > DPO (constitutional)
        # > slow adapter (incremental update from current state).
        # CLoRA orthogonalizes against the SLOW adapter history (PREVIOUS_ADAPTER_PATH),
        # independently of which BASE_ADAPTER is chosen.
        base_adapter = (
            self._pending_shared_adapter
            or self._pending_dpo_adapter
            or self._sure.production_adapter_path
        )
        self._pending_shared_adapter = None  # consumed - prevent stale adapter on next run
        self._pending_dpo_adapter = None  # consumed - prevent stale adapter on next run

        env = {
            **os.environ,
            "BASE_MODEL": _DEFAULT_BASE_MODEL,
            "TRAINING_DATA": jsonl_path,
            "OUTPUT_DIR": adapter_output_dir,
            "TRAINING_ARGS": json.dumps(training_args),
            # BASE_ADAPTER: DPO-tuned starting point (if available), else current slow adapter.
            "BASE_ADAPTER": base_adapter or "",
            # PREVIOUS_ADAPTER_PATH: CLoRA orthogonalizes against slow adapter's directions.
            # Always the slow adapter - independent of BASE_ADAPTER.
            "PREVIOUS_ADAPTER_PATH": self._sure.production_adapter_path or "",
        }

        logger.info(
            "continual_learning_subprocess_start",
            run_id=run.run_id,
            script=_TRAIN_SCRIPT,
            examples=run.examples_used,
            base_model=_DEFAULT_BASE_MODEL,
            output_dir=adapter_output_dir,
        )

        train_success = False
        proc_error = ""
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                _TRAIN_SCRIPT,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=float(_TRAINING_TIMEOUT_S),
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                proc_error = f"training subprocess timed out after {_TRAINING_TIMEOUT_S}s"
                logger.error("continual_learning_subprocess_timeout", run_id=run.run_id, timeout_s=_TRAINING_TIMEOUT_S)
            else:
                if proc.returncode == 0:
                    train_success = True
                    logger.info(
                        "continual_learning_subprocess_complete",
                        run_id=run.run_id,
                        returncode=proc.returncode,
                    )
                else:
                    stderr_tail = stderr_bytes.decode(errors="replace")[-2000:]
                    proc_error = f"subprocess exited with code {proc.returncode}: {stderr_tail}"
                    logger.error(
                        "continual_learning_subprocess_failed",
                        run_id=run.run_id,
                        returncode=proc.returncode,
                        stderr_tail=stderr_tail,
                    )
        except Exception as exc:
            proc_error = f"failed to launch subprocess: {exc}"
            logger.error("continual_learning_subprocess_launch_error", run_id=run.run_id, error=str(exc))

        run.completed_at = datetime.now(UTC)

        # ── Step 6: Success path - anti-forgetting pipeline + deploy ──────────
        if train_success:
            adapter_dir = Path(adapter_output_dir) / "adapter"
            if not adapter_dir.exists():
                # train_lora.py (Akash flow) may place adapter at root output_dir
                adapter_dir = Path(adapter_output_dir)

            run.adapter_path = str(adapter_dir)

            # Extract eval loss if available (train_lora.py logs it to a status file)
            run.eval_loss = _read_eval_loss(Path(adapter_output_dir))

            # ── Step 6a: Add new training examples to replay buffer ────────────
            # These examples are in the JSONL we already exported - reload them
            # and buffer them for the next training cycle.
            # Protected prompts (evaluation files, DPO pairs, anchors) are filtered
            # out before entering the replay buffer.
            if self._replay is not None:
                try:
                    new_examples = _load_jsonl_examples(jsonl_path)
                    new_examples, _excl = self._exclusion_filter.filter_batch(new_examples)
                    if _excl:
                        logger.warning(
                            "training_data.protected_examples_removed_from_replay",
                            count=_excl,
                        )
                    await self._replay.add_examples(new_examples)
                    logger.info(
                        "replay_buffer_examples_added",
                        run_id=run.run_id,
                        count=len(new_examples),
                    )
                except Exception as exc:
                    logger.warning("replay_buffer_add_failed", run_id=run.run_id, error=str(exc))

            # ── Steps 6b–6e: Anti-forgetting stack ────────────────────────────
            # Bypassed entirely when ablation mode is "anti_forgetting_off".
            # In that case the fast adapter is deployed directly without SuRe EMA,
            # SafeLoRA projection, KL gate, or perplexity alarm.
            _anti_forgetting_disabled = self._ablation_mode == "anti_forgetting_off"
            kl_divergence = 0.0

            if _anti_forgetting_disabled:
                slow_adapter_path = str(adapter_dir)
                logger.info(
                    "ablation_anti_forgetting_stack_bypassed",
                    run_id=run.run_id,
                    mode=self._ablation_mode,
                )
            else:
                # ── Step 6b: SuRe EMA - merge fast adapter into slow adapter ──────
                slow_adapter_path = str(adapter_dir)  # default: fast adapter IS production if EMA fails
                try:
                    slow_output = str(adapter_dir) + "_slow"
                    slow_adapter_path = await self._sure.update_slow_adapter(
                        fast_adapter_path=str(adapter_dir),
                        output_path=slow_output,
                    )
                    logger.info(
                        "sure_slow_adapter_ready",
                        run_id=run.run_id,
                        slow_path=slow_adapter_path,
                    )
                except Exception as exc:
                    logger.warning(
                        "sure_ema_failed_fallback_fast",
                        run_id=run.run_id,
                        error=str(exc),
                        fallback="deploying fast adapter directly",
                    )

                # ── Step 6b.5: SafeLoRA projection (§7.2) ─────────────────────────
                # Project the slow adapter onto the safety-aligned subspace.
                # Non-fatal: if projection fails, slow_adapter_path is returned unchanged.
                try:
                    slow_adapter_path = await self._safe_lora.project(slow_adapter_path)
                    logger.debug("safe_lora_projection_applied", run_id=run.run_id, path=slow_adapter_path)
                except Exception as exc:
                    logger.warning("safe_lora_projection_error", run_id=run.run_id, error=str(exc))

                # ── Step 6c: STABLE KL gate - check before deploying ──────────────
                kl_passed = True
                try:
                    kl_passed, kl_divergence = await self._stable.check_kl_divergence(
                        re_service=self._re,
                        current_adapter_path=self._current_adapter_path,
                        new_adapter_path=slow_adapter_path,
                    )
                except Exception as exc:
                    logger.warning(
                        "stable_kl_check_error_pass",
                        run_id=run.run_id,
                        error=str(exc),
                        note="KL gate errored - deploying anyway",
                    )

                if not kl_passed:
                    logger.warning(
                        "stable_kl_gate_rejected",
                        run_id=run.run_id,
                        kl_divergence=round(kl_divergence, 5),
                        budget=self._af_config.kl_budget,
                    )
                    run.error = f"STABLE KL gate rejected adapter: kl={kl_divergence:.5f} > budget={self._af_config.kl_budget}"
                    run.completed_at = datetime.now(UTC)

                    await self._emit(
                        "re_kl_gate_rejected",
                        {
                            "run_id": run.run_id,
                            "kl_divergence": kl_divergence,
                            "budget": self._af_config.kl_budget,
                            "adapter_path": slow_adapter_path,
                        },
                    )
                    # Do NOT deploy - current adapter stays in production
                    return run

            # ── Step 6d-pre: Snapshot pre-deployment baseline ─────────────────
            # Capture Thompson success rate + eval_loss of current adapter
            # so we have a ground truth to compare against after deployment.
            pre_rate = await self._read_thompson_success_rate()
            self._pre_deploy_baseline = {
                "success_rate": pre_rate,
                "eval_loss": run.eval_loss,  # new adapter's eval_loss (proxy for quality)
                "cycle": run.run_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "adapter_path": self._current_adapter_path,
            }
            # Store the adapter path we'll restore on rollback.
            self._pre_deploy_adapter_path = self._current_adapter_path
            if self._redis is not None:
                try:
                    await self._redis.set(
                        _REDIS_KEY_PRE_DEPLOY_BASELINE,
                        json.dumps(self._pre_deploy_baseline),
                    )
                except Exception as _bex:
                    logger.warning("pre_deploy_baseline_persist_failed", error=str(_bex))
            logger.info(
                "pre_deploy_baseline_captured",
                run_id=run.run_id,
                pre_success_rate=round(pre_rate, 4),
                pre_adapter=self._current_adapter_path,
            )

            # ── Step 6d: Deploy adapter ────────────────────────────────────────
            try:
                adapter_id = f"eos_slow_{run.run_id}"
                await self._re.load_adapter(slow_adapter_path, adapter_id)
                run.deployed = True
                self._current_adapter_path = slow_adapter_path
                logger.info(
                    "continual_learning_adapter_deployed",
                    run_id=run.run_id,
                    adapter_path=slow_adapter_path,
                    eval_loss=run.eval_loss,
                    kl_divergence=round(kl_divergence, 5),
                )
            except Exception as exc:
                # Deployment failure doesn't poison the training record
                logger.warning(
                    "continual_learning_adapter_deploy_failed",
                    run_id=run.run_id,
                    error=str(exc),
                )
                run.error = f"training succeeded but adapter deployment failed: {exc}"

            # ── Step 6d-post: Start post-deployment monitoring window ──────────
            # Reset counters and open the monitoring window.
            # _monitor_re_outcome() is called externally by NovaService when it
            # records each RE decision outcome (via record_re_outcome()).
            if run.deployed:
                self._post_deploy_successes = 0
                self._post_deploy_attempts = 0
                self._monitoring_active = True
                logger.info(
                    "post_deploy_monitoring_started",
                    run_id=run.run_id,
                    window_cycles=_POST_DEPLOY_MONITOR_CYCLES,
                    pre_success_rate=round(pre_rate, 4),
                )

            # ── Step 6d.1: Upload adapter to S3 for inference pod pickup ────────
            # Non-blocking - inference pod watcher polls this prefix.
            s3_adapter_path = await self._upload_adapter_to_s3(
                slow_adapter_path, run.run_id, kl_divergence, run.eval_loss
            )

            # Update last train timestamp
            self._last_train_at = run.completed_at

            await self._emit(
                "re_training_complete",
                {
                    "run_id": run.run_id,
                    "tier": 2,
                    "examples_used": run.examples_used,
                    "eval_loss": run.eval_loss,
                    "adapter_id": f"eos_slow_{run.run_id}",
                    "adapter_path": slow_adapter_path,
                    "s3_adapter_path": s3_adapter_path,
                    "kl_divergence": round(kl_divergence, 5),
                },
            )

            # ── Step 6e: Anchor perplexity check (non-blocking alarm) ─────────
            # Skipped when anti-forgetting stack is disabled (ablation).
            if (
                not _anti_forgetting_disabled
                and self._perplexity_monitor is not None
                and self._stable._anchor_prompts
            ):
                asyncio.ensure_future(
                    self._perplexity_monitor.check_and_alarm(
                        re_service=self._re,
                        anchor_prompts=self._stable._anchor_prompts,
                        event_bus=self._event_bus,
                    )
                )

        # ── Step 7: Failure path ───────────────────────────────────────────────
        else:
            run.error = proc_error
            from systems.synapse.types import SynapseEventType as _SET
            await self._emit(
                _SET.RE_TRAINING_FAILED,
                {"run_id": run.run_id, "tier": 2, "reason": proc_error},
            )
            # Organism continues on Claude-only; no re-raise

        return run

    # ── Periodic check ─────────────────────────────────────────────────────────

    async def check_and_train(self) -> None:
        """
        Called by the daily background task. Checks trigger conditions and
        fires Tier 2 if needed. All errors are caught - never raises.
        """
        try:
            should, reason = await self.should_train()
            if should:
                logger.info("continual_learning_trigger_fired", reason=reason)
                await self.run_tier2(reason)
            else:
                logger.debug("continual_learning_no_trigger")
        except Exception as exc:
            logger.error("continual_learning_check_failed", error=str(exc), exc_info=True)

    # ── History ────────────────────────────────────────────────────────────────

    async def get_training_history(self) -> list[TrainingRun]:
        return list(self._training_runs)

    async def get_status(self) -> dict[str, Any]:
        """Diagnostic summary for health endpoints and CLI."""
        return {
            "last_train_at": self._last_train_at.isoformat() if self._last_train_at else None,
            "total_runs": len(self._training_runs),
            "deployed_runs": sum(1 for r in self._training_runs if r.deployed),
            "failed_runs": sum(1 for r in self._training_runs if r.error and not r.deployed),
            "latest_run": self._training_runs[-1].to_dict() if self._training_runs else None,
            "config": {
                "min_new_examples": self._config.min_new_examples,
                "max_days_since_train": self._config.max_days_since_train,
                "performance_drop_threshold": self._config.performance_drop_threshold,
                "full_retrain_interval_days": self._config.full_retrain_interval_days,
            },
        }

    # ── Post-deployment quality monitoring ─────────────────────────────────────

    def record_re_outcome(self, success: bool) -> None:
        """
        Called by NovaService._on_axon_execution_result() for every RE decision.

        Accumulates successes/attempts in the post-deployment monitoring window.
        When the window fills (_POST_DEPLOY_MONITOR_CYCLES attempts), evaluates
        quality vs the pre-deployment baseline and either confirms or rolls back
        the adapter.  Non-blocking - spawns a background coroutine to handle the
        async rollback/confirm logic.

        No-op if no monitoring window is open.
        """
        if not self._monitoring_active:
            return
        self._post_deploy_attempts += 1
        if success:
            self._post_deploy_successes += 1

        if self._post_deploy_attempts >= _POST_DEPLOY_MONITOR_CYCLES:
            # Window complete - evaluate and act in the background so we don't
            # block the calling synchronous record path.
            self._monitoring_active = False
            asyncio.ensure_future(self._evaluate_post_deploy_quality())

    async def _evaluate_post_deploy_quality(self) -> None:
        """
        Compare post-deployment success rate against pre-deployment baseline.

        Called once when the monitoring window closes.  Never raises.
        """
        try:
            if self._pre_deploy_baseline is None:
                logger.warning("post_deploy_eval.no_baseline")
                return

            pre_rate: float = self._pre_deploy_baseline.get("success_rate", 0.5)
            post_rate: float = (
                self._post_deploy_successes / self._post_deploy_attempts
                if self._post_deploy_attempts > 0
                else 0.0
            )
            run_id: str = self._pre_deploy_baseline.get("cycle", "unknown")

            logger.info(
                "post_deploy_quality_evaluated",
                run_id=run_id,
                pre_rate=round(pre_rate, 4),
                post_rate=round(post_rate, 4),
                window_attempts=self._post_deploy_attempts,
                window_successes=self._post_deploy_successes,
            )

            rollback_threshold = pre_rate * _ROLLBACK_DEGRADATION_THRESHOLD
            confirm_threshold = pre_rate * _CONFIRM_IMPROVEMENT_THRESHOLD

            if post_rate < rollback_threshold:
                await self._rollback_adapter(run_id, pre_rate, post_rate)
            elif post_rate > confirm_threshold:
                await self._confirm_adapter(run_id, pre_rate, post_rate)
            else:
                # Within acceptable range - neutral, keep current adapter
                logger.info(
                    "post_deploy_quality_neutral",
                    run_id=run_id,
                    pre_rate=round(pre_rate, 4),
                    post_rate=round(post_rate, 4),
                )

        except Exception as exc:
            logger.error("post_deploy_eval_failed", error=str(exc))

    async def _rollback_adapter(self, run_id: str, pre_rate: float, post_rate: float) -> None:
        """
        Restore the pre-deployment adapter and reset Thompson sampler Beta params.

        Steps:
          1. Reload the previous adapter in vLLM.
          2. Emit MODEL_ROLLBACK_TRIGGERED.
          3. Reset Thompson "re" arm Beta params to pre-deployment values
             (stored as baseline success_rate → infer alpha/beta from rate).
          4. Emit RE_TRAINING_EXAMPLE so the regression is a learning signal.
          5. Log the rollback reason.
        """
        logger.warning(
            "re_adapter_rollback",
            reason="quality_degradation",
            run_id=run_id,
            pre_rate=round(pre_rate, 4),
            post_rate=round(post_rate, 4),
        )

        # 1. Restore previous adapter
        if self._pre_deploy_adapter_path:
            try:
                rollback_id = f"eos_rollback_{run_id}"
                await self._re.load_adapter(self._pre_deploy_adapter_path, rollback_id)
                self._current_adapter_path = self._pre_deploy_adapter_path
                logger.info(
                    "re_adapter_restored",
                    run_id=run_id,
                    adapter_path=self._pre_deploy_adapter_path,
                )
            except Exception as exc:
                logger.error("re_adapter_rollback_load_failed", error=str(exc))
        else:
            logger.warning("re_adapter_rollback.no_previous_adapter", run_id=run_id)

        # 2. Emit MODEL_ROLLBACK_TRIGGERED
        await self._emit(
            "model_rollback_triggered",
            {
                "run_id": run_id,
                "reason": "quality_degradation",
                "pre_success_rate": round(pre_rate, 4),
                "post_success_rate": round(post_rate, 4),
                "window_attempts": self._post_deploy_attempts,
                "rollback_adapter": self._pre_deploy_adapter_path,
                "auto_rollback": True,
            },
        )

        # 3. Reset Thompson "re" arm Beta params.
        # We approximate: if the old rate was R, set alpha=R*N, beta=(1-R)*N
        # with N=20 (modest confidence - not erasing all historical learning).
        if self._redis is not None:
            try:
                _n = 20.0
                rollback_alpha = pre_rate * _n
                rollback_beta = (1.0 - pre_rate) * _n
                # Patch the Redis hash that ThompsonSampler.load_from_redis() reads.
                from systems.nova.policy_generator import ThompsonSampler
                await self._redis.hset(
                    ThompsonSampler.REDIS_KEY,
                    mapping={
                        "re_alpha": str(rollback_alpha),
                        "re_beta": str(rollback_beta),
                    },
                )
                logger.info(
                    "thompson_sampler_re_reset",
                    run_id=run_id,
                    re_alpha=rollback_alpha,
                    re_beta=rollback_beta,
                )
            except Exception as exc:
                logger.warning("thompson_sampler_re_reset_failed", error=str(exc))

        # 4. Emit RE_TRAINING_EXAMPLE - the rollback is a learning signal for future training
        await self._emit(
            "re_training_example",
            {
                "source": "re_adapter_rollback",
                "run_id": run_id,
                "pre_success_rate": round(pre_rate, 4),
                "post_success_rate": round(post_rate, 4),
                "window_attempts": self._post_deploy_attempts,
                "outcome": "rollback_triggered",
                "signal": "quality_degradation_detected",
            },
        )

        # Clear baseline so next deploy starts fresh
        self._pre_deploy_baseline = None
        self._pre_deploy_adapter_path = None

    async def _confirm_adapter(self, run_id: str, pre_rate: float, post_rate: float) -> None:
        """Confirm successful deployment - log and emit quality confirmation event."""
        logger.info(
            "re_adapter_confirmed",
            run_id=run_id,
            pre_rate=round(pre_rate, 4),
            post_rate=round(post_rate, 4),
            improvement_pct=round((post_rate / pre_rate - 1.0) * 100, 1),
        )
        await self._emit(
            "re_adapter_quality_confirmed",
            {
                "run_id": run_id,
                "pre_success_rate": round(pre_rate, 4),
                "post_success_rate": round(post_rate, 4),
                "improvement_pct": round((post_rate / pre_rate - 1.0) * 100, 1),
                "window_attempts": self._post_deploy_attempts,
            },
        )
        # Clear baseline - this adapter is now the confirmed baseline for the next deploy
        self._pre_deploy_baseline = None
        self._pre_deploy_adapter_path = None

    async def _read_thompson_success_rate(self) -> float:
        """Read the current RE success rate from Redis (written by ThompsonSampler).

        Returns 0.5 (neutral prior) if Redis is unavailable or key is missing.
        """
        if self._redis is None:
            return 0.5
        try:
            raw = await self._redis.get(_REDIS_KEY_THOMPSON_SCORE)
            if raw:
                return float(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception as exc:
            logger.debug("thompson_rate_read_failed", error=str(exc))
        return 0.5

    # ── Internal helpers ───────────────────────────────────────────────────────

    async def _build_cumulative_dataset(self) -> str:
        """Export the full cumulative training dataset (90-day lookback, no quality floor)
        for Tier 3 full retrain. Returns the local JSONL path.

        Uses the same export pipeline as Tier 2 but with a wider lookback and lower
        quality threshold to capture the full historical training signal.
        Non-fatal: returns empty path string if export fails.
        """
        try:
            from systems.reasoning_engine.export_pipeline import run_export

            neo4j = self._extractor._neo4j
            export_dir = Path(_DEFAULT_EXPORT_DIR)
            export_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            output_path = str(export_dir / f"tier3_cumulative_{timestamp}.jsonl")

            result = await run_export(
                neo4j=neo4j,
                output_path=output_path,
                lookback_days=90,
                min_score=0.20,  # lower threshold for full history
            )
            if result.error:
                logger.warning("tier3.cumulative_export_failed", error=result.error)
                return ""

            # Filter protected evaluation files out of the cumulative dataset
            try:
                raw_examples = _load_jsonl_examples(output_path)
                filtered, excluded = self._exclusion_filter.filter_batch(raw_examples)
                if excluded:
                    logger.warning(
                        "training_data.protected_examples_removed",
                        count=excluded,
                        stage="tier3_cumulative",
                    )
                    import json as _json
                    with open(output_path, "w") as _f:
                        for _ex in filtered:
                            _f.write(_json.dumps(_ex) + "\n")
            except Exception as _exc:
                logger.warning("tier3.exclusion_filter_failed", error=str(_exc))

            logger.info(
                "tier3.cumulative_dataset_built",
                total=result.total_exported,
                path=output_path,
            )
            return output_path
        except Exception as exc:
            logger.error("tier3.cumulative_export_error", error=str(exc))
            return ""

    async def _upload_adapter_to_s3(
        self,
        adapter_path: str,
        run_id: str,
        kl_divergence: float,
        eval_loss: float | None,
    ) -> str | None:
        """
        Upload the production adapter to S3 so the inference pod can pick it up.

        Writes adapter files to:
            s3://{bucket}/{prefix}{instance_id}/{timestamp}/

        Also writes a manifest.json at the prefix root so the pod watcher knows
        which version is current without listing all keys.

        Returns the S3 path on success, None on failure. Non-fatal - local
        deployment still works even if S3 upload fails.
        """
        try:
            import boto3  # type: ignore[import]
        except ImportError:
            logger.info("s3_adapter_upload.boto3_not_available", hint="pip install boto3")
            return None

        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            logger.warning("s3_adapter_upload.path_not_found", path=adapter_path)
            return None

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        s3_prefix = f"{_S3_ADAPTER_PREFIX}{_INSTANCE_ID}/{timestamp}/"

        try:
            s3 = boto3.client("s3")

            # Upload all adapter files (safetensors, config, tokenizer, etc.)
            uploaded_files: list[str] = []
            files_to_upload = list(adapter_dir.iterdir()) if adapter_dir.is_dir() else [adapter_dir]
            for fpath in files_to_upload:
                if fpath.is_file():
                    key = f"{s3_prefix}{fpath.name}"
                    s3.upload_file(str(fpath), _S3_ADAPTER_BUCKET, key)
                    uploaded_files.append(fpath.name)

            # Write manifest at the well-known prefix - pod watcher reads this
            manifest = {
                "version": timestamp,
                "run_id": run_id,
                "instance_id": _INSTANCE_ID,
                "adapter_s3_prefix": s3_prefix,
                "kl_divergence": round(kl_divergence, 5),
                "eval_loss": eval_loss,
                "files": uploaded_files,
                "uploaded_at": datetime.now(UTC).isoformat(),
            }
            manifest_key = f"{_S3_ADAPTER_PREFIX}{_INSTANCE_ID}/latest_manifest.json"
            s3.put_object(
                Bucket=_S3_ADAPTER_BUCKET,
                Key=manifest_key,
                Body=json.dumps(manifest, indent=2),
                ContentType="application/json",
            )

            s3_full_path = f"s3://{_S3_ADAPTER_BUCKET}/{s3_prefix}"
            logger.info(
                "s3_adapter_uploaded",
                run_id=run_id,
                s3_path=s3_full_path,
                manifest_key=manifest_key,
                files=len(uploaded_files),
            )
            return s3_full_path

        except Exception as exc:
            logger.warning(
                "s3_adapter_upload_failed",
                run_id=run_id,
                error=str(exc),
                hint="Inference pod will not pick up this adapter version",
            )
            return None

    async def _emit(self, event_type_str: "str | SynapseEventType", payload: dict[str, Any]) -> None:
        """Fire-and-forget Synapse event. Never raises."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            if isinstance(event_type_str, SynapseEventType):
                et: SynapseEventType = event_type_str
            else:
                et = SynapseEventType(event_type_str)
            event = SynapseEvent(
                event_type=et,
                data=payload,
                source_system="reasoning_engine",
            )
            asyncio.ensure_future(self._event_bus.emit(event))
        except Exception as exc:
            logger.debug("continual_learning_emit_failed", event=str(event_type_str), error=str(exc))

    async def _set_training_halted(self, reason: str) -> None:
        """Persist training halt to Redis and notify Thymos via Synapse."""
        self._training_halted = True
        if self._redis is not None:
            try:
                await self._redis.set(_TRAINING_HALTED_KEY, reason)
            except Exception as exc:
                logger.warning("training_halt_persist_failed", error=str(exc))
        logger.critical("training.halted_persisted", reason=reason)
        # Emit RE_TRAINING_HALTED so Thymos creates a HIGH incident and the
        # Synapse bus is aware - Thymos can then attempt recovery.
        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                asyncio.ensure_future(self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.RE_TRAINING_HALTED,
                    source_system="reasoning_engine",
                    data={
                        "reason": reason,
                        "halted_by": "kill_switch",
                        "job_id": "continual_learning",
                        "tier": 2,
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )))
            except Exception:
                pass

    async def _is_training_halted(self) -> tuple[bool, str]:
        """Check Redis for persisted halt flag."""
        if self._redis is None:
            return self._training_halted, "in_memory"
        try:
            reason = await self._redis.get(_TRAINING_HALTED_KEY)
            if reason:
                decoded = reason.decode() if isinstance(reason, bytes) else reason
                return True, decoded
        except Exception as exc:
            logger.warning("training_halt_check_failed", error=str(exc))
        return False, ""

    async def clear_training_halt(self) -> None:
        """Public operator action: clear halt and resume training.

        Call this after a RE quality regression has been investigated and the
        cause addressed.  The next `check_and_train()` cycle (≤6 hours) will
        re-evaluate `should_train()` without the halt gate.

        Also emits RE_TRAINING_RESUMED on the Synapse bus so Benchmarks and
        Nova can react (e.g. reset Thompson baseline, resume goal injection).
        """
        self._training_halted = False
        if self._redis is not None:
            try:
                await self._redis.delete(_TRAINING_HALTED_KEY)
            except Exception as exc:
                logger.warning("training_halt_clear_failed", error=str(exc))
        logger.info("training.halt_cleared")
        # Notify the bus - non-blocking, best-effort
        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                asyncio.ensure_future(self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.RE_TRAINING_RESUMED,
                    source_system="reasoning_engine",
                    data={"cleared_by": "operator", "timestamp": datetime.now(UTC).isoformat()},
                )))
            except Exception:
                pass

    # Keep private alias for internal callers
    _clear_training_halt = clear_training_halt

    async def _persist_state(self) -> None:
        """Persist last_train_at and training_runs to Redis. Never raises."""
        if self._redis is None:
            return
        try:
            if self._last_train_at:
                await self._redis.set(_REDIS_KEY_LAST_TRAIN, self._last_train_at.isoformat())

            # Keep last 50 runs to bound memory
            recent = self._training_runs[-50:]
            await self._redis.set(_REDIS_KEY_RUNS, json.dumps([r.to_dict() for r in recent]))
        except Exception as exc:
            logger.warning("continual_learning_persist_failed", error=str(exc))


# ── Helpers ────────────────────────────────────────────────────────────────────


def _get_training_config(n_examples: int) -> dict[str, Any]:
    """
    Dataset-size-adaptive hyperparameters (Speciation Bible §5).

    Small dataset  (<500):  more regularisation, higher lr, fewer epochs
    Medium dataset (500-2k): balanced settings
    Large dataset  (>2k):   lower lr, more epochs, bigger batch
    """
    if n_examples < 500:
        return {
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.10,
            "learning_rate": 3e-4,
            "num_epochs": 2,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 4096,
            "warmup_ratio": 0.05,
            "weight_decay": 0.01,
        }
    elif n_examples < 2000:
        return {
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 4096,
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
        }
    else:
        return {
            "lora_rank": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "learning_rate": 1e-4,
            "num_epochs": 4,
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "max_seq_length": 4096,
            "warmup_ratio": 0.03,
            "weight_decay": 0.01,
        }


def _load_jsonl_examples(jsonl_path: str) -> list[dict[str, Any]]:
    """Load training examples from a JSONL file for replay buffering.

    Returns an empty list if the file is missing or unreadable.
    """
    path = Path(jsonl_path)
    if not path.exists():
        return []
    examples: list[dict[str, Any]] = []
    try:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
    except Exception:
        pass
    return examples


def _read_eval_loss(adapter_output_dir: Path) -> float | None:
    """
    Try to read the final eval loss from a status JSON written by train_lora.py.
    Returns None if unavailable (non-fatal).
    """
    # train_lora.py writes TRAINING_STATE as JSON to status.json on completion
    status_path = adapter_output_dir / "status.json"
    if status_path.exists():
        try:
            data = json.loads(status_path.read_text())
            loss = data.get("current_loss") or data.get("eval_loss")
            if loss:
                return float(loss)
        except Exception:
            pass
    return None
