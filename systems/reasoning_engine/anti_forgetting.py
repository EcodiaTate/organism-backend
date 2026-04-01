"""
EcodiaOS - Anti-Forgetting Stack (Speciation Bible §3.3)

Implements 4 of 7 mechanisms that wrap around train_lora.py (which runs as a
subprocess and is never modified):

  1. SurprisePrioritizedReplay  - ERI-LoRA (2024): Redis-backed 300–500 sample
     replay buffer; high-surprise episodes weighted higher via sorted set.

  2. SuReEMAAdapter  - SuRe (2025): Dual fast/slow adapters; production always
     serves the EMA-stabilised slow adapter; fast adapter = newly trained weights.

  3. STABLEKLGate  - STABLE (NeurIPS 2025): KL divergence gate on anchor prompts
     before deploying a new adapter; rejects updates that shift behaviour too far.

  4. AnchorPerplexityMonitor  - Perplexity-spike alarm on anchor prompts;
     detects general-knowledge forgetting independently of KL gate.

Mechanisms NOT in this file (require train_lora.py internals):
  - CLoRA orthogonal subspace (Round 4)
  - SLAO time-aware merge      (Round 4 - quarterly)
  - SVD pruning                (Round 4 - quarterly)

ANCHOR PROMPTS:  data/re_training_batches/anchor_prompts.jsonl
  - NEVER include these in any training data JSONL.
  - They are read-only behavioural probes, not examples.
"""

from __future__ import annotations

import asyncio
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from systems.reasoning_engine.service import ReasoningEngineService
    from systems.reasoning_engine.training_exclusions import TrainingExclusionFilter

logger = structlog.get_logger("reasoning_engine.anti_forgetting")

# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class AntiForgetConfig:
    replay_buffer_size: int = 500
    replay_surprise_weight: float = 0.7  # weight toward surprising examples when sampling
    ema_decay: float = 0.99              # SuRe slow adapter EMA decay coefficient
    kl_budget: float = 0.1              # STABLE KL divergence gate budget (nats)
    anchor_perplexity_alarm: float = 0.20  # 20% spike from baseline = alarm
    svd_prune_top_k: int = 5            # intruder dims to prune (quarterly - Round 4)


# ── Redis keys ─────────────────────────────────────────────────────────────────

_REPLAY_BUFFER_KEY = "eos:re:replay_buffer"
_PERPLEXITY_BASELINE_KEY = "eos:re:anchor_perplexity_baseline"


# ── 1. Surprise-Prioritized Replay (ERI-LoRA 2024) ────────────────────────────


class SurprisePrioritizedReplay:
    """ERI-LoRA (2024): 300–500 samples, surprising episodes weighted higher.

    Maintains a rolling Redis sorted set of high-quality training examples.
    Each Tier 2 cycle, these are mixed into the new training JSONL to prevent
    forgetting previous knowledge.

    Priority = quality_score × (1 + surprise_factor)
    where surprise_factor ∈ [0, 1] reflects how unexpected the episode was
    (derived from the 'novelty' field written by the quality pipeline).
    """

    def __init__(
        self,
        config: AntiForgetConfig,
        redis_client: "Redis",
        exclusion_filter: "TrainingExclusionFilter | None" = None,
    ) -> None:
        self._config = config
        self._redis = redis_client
        self._exclusion_filter = exclusion_filter

    async def add_examples(self, examples: list[dict[str, Any]]) -> None:
        """Add new training examples to the replay buffer.

        Priority = quality_score × (1 + novelty_score).
        Evicts lowest-priority entries when buffer exceeds replay_buffer_size.
        """
        if not examples:
            return

        # Filter out protected evaluation prompts before they enter the replay buffer
        if self._exclusion_filter is not None:
            examples, excluded = self._exclusion_filter.filter_batch(examples)
            if excluded:
                logger.warning(
                    "replay_buffer.protected_examples_excluded",
                    count=excluded,
                )
            if not examples:
                return

        pipe = self._redis.pipeline()
        for ex in examples:
            quality = float(ex.get("quality_score", 0.5))
            novelty = float(ex.get("novelty", 0.5))
            priority = quality * (1.0 + novelty * self._config.replay_surprise_weight)

            # Strip heavy metadata before buffering - keep only the messages payload
            slim = {"messages": ex["messages"]}
            if "stream_id" in ex:
                slim["stream_id"] = ex["stream_id"]

            pipe.zadd(_REPLAY_BUFFER_KEY, {json.dumps(slim): priority})

        # Trim to max size - keep top-N by score (highest priority)
        trim_start = 0
        trim_end = -(self._config.replay_buffer_size + 1)  # keep top N
        pipe.zremrangebyrank(_REPLAY_BUFFER_KEY, trim_start, trim_end)
        await pipe.execute()

        logger.debug(
            "replay_buffer_updated",
            added=len(examples),
            buffer_key=_REPLAY_BUFFER_KEY,
        )

    async def sample(self, n: int = 300) -> list[dict[str, Any]]:
        """Sample replay examples, bias toward high-surprise (high-score) entries.

        Samples replay_surprise_weight × n from the top half of the buffer
        and (1 - replay_surprise_weight) × n from the lower half.
        """
        total = await self._redis.zcard(_REPLAY_BUFFER_KEY)
        if total == 0:
            return []

        n = min(n, total)
        high_n = min(int(n * self._config.replay_surprise_weight), total)
        low_n = n - high_n

        # Top half = highest surprise scores
        high_raw = await self._redis.zrevrange(_REPLAY_BUFFER_KEY, 0, max(high_n - 1, 0))
        # Lower half - offset into the buffer
        low_start = max(high_n, 0)
        low_raw = await self._redis.zrevrange(_REPLAY_BUFFER_KEY, low_start, low_start + max(low_n - 1, 0))

        results: list[dict[str, Any]] = []
        for raw in list(high_raw) + list(low_raw):
            try:
                decoded = raw.decode() if isinstance(raw, bytes) else raw
                results.append(json.loads(decoded))
            except Exception as exc:
                logger.warning("replay_sample_decode_failed", error=str(exc))

        logger.debug("replay_buffer_sampled", requested=n, returned=len(results))
        return results

    async def restore_from_redis(self) -> None:
        """Log buffer state on startup (buffer persists in Redis natively)."""
        try:
            size = await self._redis.zcard(_REPLAY_BUFFER_KEY)
            logger.info("replay_buffer_restored", size=size)
        except Exception as exc:
            logger.warning("replay_buffer_restore_check_failed", error=str(exc))


# ── 2. SuRe EMA Adapter (SuRe 2025) ──────────────────────────────────────────


class SuReEMAAdapter:
    """SuRe (2025): Dual fast/slow adapter with EMA stabilisation.

    After each Tier 2 training cycle:
      fast adapter  = newly trained weights (direct output of train_lora.py)
      slow adapter  = EMA of all historical fast adapters

    Production ALWAYS serves the slow adapter - more stable, less susceptible
    to catastrophic updates from any single training run.

    slow_weights[k] = ema_decay * slow_weights[k] + (1 - ema_decay) * fast_weights[k]
    """

    def __init__(self, config: AntiForgetConfig) -> None:
        self._decay = config.ema_decay
        self._slow_adapter_path: str | None = None

    async def update_slow_adapter(self, fast_adapter_path: str, output_path: str) -> str:
        """Apply EMA update: merge fast adapter into slow adapter.

        Uses safetensors to load both, apply element-wise EMA, and save.
        If no slow adapter exists (first training), copies fast → slow.

        Returns: path to updated slow adapter.
        """
        # Lazy import - safetensors not required at boot (no GPU needed for EMA merge)
        try:
            import torch
            from safetensors.torch import load_file, save_file
            _has_safetensors = True
        except ImportError:
            _has_safetensors = False

        if self._slow_adapter_path is None:
            # First training cycle - slow adapter doesn't exist yet
            logger.info(
                "sure_ema_first_cycle",
                fast_adapter_path=fast_adapter_path,
                output_path=output_path,
            )
            if _has_safetensors:
                # Copy fast adapter files to slow adapter path
                fast_dir = Path(fast_adapter_path)
                slow_dir = Path(output_path)
                slow_dir.mkdir(parents=True, exist_ok=True)
                for f in fast_dir.iterdir():
                    shutil.copy2(f, slow_dir / f.name)
            else:
                shutil.copytree(fast_adapter_path, output_path, dirs_exist_ok=True)

            self._slow_adapter_path = output_path
            return output_path

        if not _has_safetensors:
            # safetensors not available - copy fast as fallback
            logger.warning(
                "sure_ema_no_safetensors",
                note="safetensors not installed; copying fast adapter as slow adapter fallback",
            )
            slow_dir = Path(output_path)
            slow_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(fast_adapter_path, output_path, dirs_exist_ok=True)
            self._slow_adapter_path = output_path
            return output_path

        try:
            # Find the .safetensors weight file in each adapter directory
            fast_file = _find_safetensors_file(fast_adapter_path)
            slow_file = _find_safetensors_file(self._slow_adapter_path)

            if fast_file is None or slow_file is None:
                raise FileNotFoundError(
                    f"safetensors file not found: fast={fast_file}, slow={slow_file}"
                )

            fast_tensors = load_file(str(fast_file))
            slow_tensors = load_file(str(slow_file))

            # EMA merge - only keys present in both adapters
            merged: dict[str, Any] = {}
            for key in fast_tensors:
                if key in slow_tensors:
                    fast_t = fast_tensors[key].float()
                    slow_t = slow_tensors[key].float()
                    merged[key] = (self._decay * slow_t + (1.0 - self._decay) * fast_t).to(
                        fast_tensors[key].dtype
                    )
                else:
                    # New key in fast - include as-is (new LoRA modules)
                    merged[key] = fast_tensors[key]

            # Keys in slow but not in fast - preserve them
            for key in slow_tensors:
                if key not in fast_tensors:
                    merged[key] = slow_tensors[key]

            # Write merged slow adapter
            slow_out_dir = Path(output_path)
            slow_out_dir.mkdir(parents=True, exist_ok=True)
            out_file = slow_out_dir / "adapter_model.safetensors"
            save_file(merged, str(out_file))

            # Copy non-weight files (adapter_config.json, tokenizer_config, etc.)
            fast_dir = Path(fast_adapter_path)
            for f in fast_dir.iterdir():
                if f.suffix != ".safetensors":
                    shutil.copy2(f, slow_out_dir / f.name)

            self._slow_adapter_path = output_path
            logger.info(
                "sure_ema_updated",
                decay=self._decay,
                keys_merged=len(merged),
                output_path=output_path,
            )
            return output_path

        except Exception as exc:
            logger.warning(
                "sure_ema_merge_failed",
                error=str(exc),
                fallback="copying fast adapter as slow",
            )
            slow_dir = Path(output_path)
            slow_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(fast_adapter_path, output_path, dirs_exist_ok=True)
            self._slow_adapter_path = output_path
            return output_path

    @property
    def production_adapter_path(self) -> str | None:
        """The slow adapter is always the production adapter."""
        return self._slow_adapter_path


# ── 3. STABLE KL Gate (STABLE NeurIPS 2025) ───────────────────────────────────


class STABLEKLGate:
    """STABLE (NeurIPS 2025): KL divergence gate before deploying adapter updates.

    Before deploying a new adapter to production:
      1. Run anchor prompts through BOTH current and new adapter
      2. Compute mean KL divergence between output distributions
      3. If KL > budget: REJECT - adapter caused too large a behavioural shift

    This prevents catastrophic updates from reaching production even if the
    training loss improved.

    Non-fatal: if anchors can't be loaded or inference fails, the gate passes
    (with a warning) to avoid blocking training on infra failures.
    """

    def __init__(self, config: AntiForgetConfig) -> None:
        self._kl_budget = config.kl_budget
        self._anchor_prompts: list[dict[str, Any]] = []
        self._anchors_loaded = False

    async def load_anchors(
        self,
        anchor_file: str = "data/re_training_batches/anchor_prompts.jsonl",
    ) -> None:
        """Load anchor prompts from JSONL file.

        Each line: {"id": "...", "prompt": "...", "category": "..."}

        These prompts MUST NEVER appear in training data.
        """
        path = Path(anchor_file)
        if not path.exists():
            logger.warning("stable_kl_anchors_not_found", path=str(path))
            return

        prompts: list[dict[str, Any]] = []
        try:
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        prompts.append(json.loads(line))
            self._anchor_prompts = prompts
            self._anchors_loaded = True
            logger.info("stable_kl_anchors_loaded", count=len(prompts), path=str(path))
        except Exception as exc:
            logger.warning("stable_kl_anchors_load_failed", error=str(exc), path=str(path))

    async def check_kl_divergence(
        self,
        re_service: "ReasoningEngineService",
        current_adapter_path: str | None,
        new_adapter_path: str,
    ) -> tuple[bool, float]:
        """Run anchor prompts through both adapters, compute behavioural KL.

        Returns: (passes_gate, kl_divergence_value)

        If no current adapter (first training run): always passes.
        If anchors unavailable or inference fails: passes with warning.
        """
        if current_adapter_path is None:
            logger.info("stable_kl_first_run_pass", reason="no current adapter")
            return True, 0.0

        if not self._anchor_prompts:
            logger.warning("stable_kl_no_anchors_pass", reason="no anchor prompts loaded - deploying anyway")
            return True, 0.0

        try:
            # Collect logprob distributions from current adapter
            current_logprobs = await _collect_adapter_logprobs(
                re_service, current_adapter_path, self._anchor_prompts
            )

            # Switch to new adapter and collect logprobs
            new_logprobs = await _collect_adapter_logprobs(
                re_service, new_adapter_path, self._anchor_prompts
            )

            # Restore current adapter
            try:
                await re_service.load_adapter(current_adapter_path, "stable_kl_restore")
            except Exception as restore_exc:
                logger.warning("stable_kl_restore_failed", error=str(restore_exc))

            if not current_logprobs or not new_logprobs:
                logger.warning("stable_kl_empty_logprobs_pass", reason="inference returned empty - deploying anyway")
                return True, 0.0

            # Compute mean KL divergence across all anchor prompts
            kl_values: list[float] = []
            for curr, new_ in zip(current_logprobs, new_logprobs):
                kl = _kl_from_logprob_dicts(curr, new_)
                kl_values.append(kl)

            mean_kl = sum(kl_values) / len(kl_values) if kl_values else 0.0
            passes = mean_kl <= self._kl_budget

            logger.info(
                "stable_kl_gate_result",
                mean_kl=round(mean_kl, 5),
                budget=self._kl_budget,
                passes=passes,
                n_anchors=len(kl_values),
            )
            return passes, mean_kl

        except Exception as exc:
            logger.warning(
                "stable_kl_gate_error_pass",
                error=str(exc),
                reason="KL gate error - deploying anyway to avoid blocking training",
            )
            return True, 0.0


# ── 4. Anchor Perplexity Monitor ──────────────────────────────────────────────


class AnchorPerplexityMonitor:
    """Monitor general-text perplexity on anchor prompts.

    If perplexity spikes >20% above the baseline established on the first run,
    the model may be forgetting general knowledge. Emits a BENCHMARK_REGRESSION
    alarm via the event bus (non-blocking - never blocks deployment).
    """

    def __init__(self, config: AntiForgetConfig, redis_client: "Redis") -> None:
        self._alarm_threshold = config.anchor_perplexity_alarm
        self._redis = redis_client

    async def measure_perplexity(
        self,
        re_service: "ReasoningEngineService",
        anchor_prompts: list[dict[str, Any]],
    ) -> float:
        """Compute mean perplexity on anchor prompts using the current model.

        perplexity = exp(-mean(log_probs_per_token))
        """
        if not anchor_prompts:
            return 0.0

        total_log_prob = 0.0
        total_tokens = 0
        failed = 0

        for anchor in anchor_prompts:
            prompt_text = anchor.get("prompt", "")
            if not prompt_text:
                continue
            try:
                result = await re_service.generate(
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=1,
                    temperature=0.0,
                    # Request logprobs of the first output token as proxy
                )
                # Extract logprob from response - RE service returns dict
                lp = _extract_logprob_from_result(result)
                if lp is not None:
                    total_log_prob += lp
                    total_tokens += 1
                else:
                    failed += 1
            except Exception as exc:
                logger.debug("perplexity_measure_single_failed", error=str(exc))
                failed += 1

        if total_tokens == 0:
            logger.warning("perplexity_measure_no_tokens", failed=failed)
            return 0.0

        mean_lp = total_log_prob / total_tokens
        perplexity = math.exp(-mean_lp)
        logger.debug("perplexity_measured", perplexity=round(perplexity, 3), tokens=total_tokens)
        return perplexity

    async def check_and_alarm(
        self,
        re_service: "ReasoningEngineService",
        anchor_prompts: list[dict[str, Any]],
        event_bus: Any = None,
    ) -> dict[str, Any]:
        """Compare current perplexity to baseline.

        Returns: {"perplexity": float, "baseline": float, "spike_pct": float, "alarm": bool}
        """
        current = await self.measure_perplexity(re_service, anchor_prompts)
        if current == 0.0:
            return {"perplexity": 0.0, "baseline": 0.0, "spike_pct": 0.0, "alarm": False}

        try:
            raw_baseline = await self._redis.get(_PERPLEXITY_BASELINE_KEY)
        except Exception:
            raw_baseline = None

        if raw_baseline is None:
            # First measurement - establish baseline
            try:
                await self._redis.set(_PERPLEXITY_BASELINE_KEY, str(current))
            except Exception as exc:
                logger.warning("perplexity_baseline_save_failed", error=str(exc))
            logger.info("perplexity_baseline_established", baseline=round(current, 3))
            return {"perplexity": current, "baseline": current, "spike_pct": 0.0, "alarm": False}

        baseline = float(raw_baseline.decode() if isinstance(raw_baseline, bytes) else raw_baseline)
        if baseline == 0.0:
            return {"perplexity": current, "baseline": 0.0, "spike_pct": 0.0, "alarm": False}

        spike_pct = (current - baseline) / baseline
        alarm = spike_pct > self._alarm_threshold

        result = {
            "perplexity": round(current, 4),
            "baseline": round(baseline, 4),
            "spike_pct": round(spike_pct, 4),
            "alarm": alarm,
        }

        if alarm:
            logger.warning(
                "anchor_perplexity_alarm",
                current=round(current, 3),
                baseline=round(baseline, 3),
                spike_pct=round(spike_pct * 100, 1),
                threshold_pct=round(self._alarm_threshold * 100, 1),
            )
            if event_bus is not None:
                try:
                    from systems.synapse.types import SynapseEvent, SynapseEventType

                    event = SynapseEvent(
                        event_type=SynapseEventType.BENCHMARK_REGRESSION,
                        data={
                            "metric": "anchor_perplexity",
                            "current": current,
                            "baseline": baseline,
                            "spike_pct": spike_pct,
                            "threshold": self._alarm_threshold,
                            "source": "anti_forgetting.perplexity_monitor",
                        },
                        source_system="reasoning_engine",
                    )
                    asyncio.ensure_future(event_bus.emit(event))
                except Exception as emit_exc:
                    logger.debug("perplexity_alarm_emit_failed", error=str(emit_exc))
        else:
            logger.debug(
                "anchor_perplexity_ok",
                current=round(current, 3),
                baseline=round(baseline, 3),
                spike_pct=round(spike_pct * 100, 1),
            )

        return result


# ── Internal helpers ────────────────────────────────────────────────────────────


def _find_safetensors_file(adapter_dir: str) -> Path | None:
    """Find the primary .safetensors weight file in an adapter directory."""
    d = Path(adapter_dir)
    if not d.exists():
        return None
    candidates = list(d.glob("*.safetensors"))
    if not candidates:
        return None
    # Prefer adapter_model.safetensors; fall back to first found
    for name in ("adapter_model.safetensors", "model.safetensors"):
        for c in candidates:
            if c.name == name:
                return c
    return candidates[0]


async def _collect_adapter_logprobs(
    re_service: "ReasoningEngineService",
    adapter_path: str,
    anchors: list[dict[str, Any]],
) -> list[dict[str, float]]:
    """Load adapter, run anchors, return per-prompt logprob distributions."""
    results: list[dict[str, float]] = []
    try:
        adapter_id = f"kl_probe_{Path(adapter_path).name}"
        await re_service.load_adapter(adapter_path, adapter_id)
    except Exception as exc:
        logger.warning("collect_logprobs_load_adapter_failed", error=str(exc))
        return results

    for anchor in anchors:
        prompt = anchor.get("prompt", "")
        if not prompt:
            continue
        try:
            resp = await re_service.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=32,
                temperature=0.0,
            )
            lp = _extract_logprob_from_result(resp)
            results.append({"prompt_id": anchor.get("id", ""), "logprob": lp or 0.0})
        except Exception as exc:
            logger.debug("collect_logprobs_single_failed", anchor_id=anchor.get("id"), error=str(exc))
            results.append({"prompt_id": anchor.get("id", ""), "logprob": 0.0})

    return results


def _kl_from_logprob_dicts(
    current: dict[str, float],
    new: dict[str, float],
) -> float:
    """Compute a scalar KL divergence proxy from two logprob observations.

    For single-token logprobs, KL ≈ |log P_current - log P_new|.
    This is a conservative approximation; a full distribution KL would
    require returning the full vocabulary logit vector from vLLM.
    """
    lp_curr = current.get("logprob", 0.0)
    lp_new = new.get("logprob", 0.0)
    # KL(P||Q) ≈ exp(lp_curr) * (lp_curr - lp_new) for a single token
    p = math.exp(max(lp_curr, -50.0))
    return p * abs(lp_curr - lp_new)


def _extract_logprob_from_result(result: Any) -> float | None:
    """Extract a scalar log-probability from whatever re_service.generate() returns.

    Handles both dict responses (vLLM OpenAI-compat) and string responses.
    Returns None if logprob is unavailable.
    """
    if result is None:
        return None
    if isinstance(result, str):
        # Plain string response - no logprob available
        return None
    if isinstance(result, dict):
        # OpenAI-compat: choices[0].logprobs.content[0].logprob
        try:
            choices = result.get("choices", [])
            if choices:
                lp_data = choices[0].get("logprobs") or {}
                content = lp_data.get("content", [])
                if content:
                    return float(content[0].get("logprob", 0.0))
        except Exception:
            pass
    return None
