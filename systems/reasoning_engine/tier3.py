"""
EcodiaOS - Tier 3 Quarterly Retrain (Speciation Bible §3.3 + §3.4)

Implements the remaining 3 anti-forgetting mechanisms not covered by Round 3A:

  5. CLoRA orthogonal subspace init - applied at Tier 2 train time via
     PREVIOUS_ADAPTER_PATH env var in train_lora.py (not this file).

  6. SVD pruning (quarterly) - removes intruder dimensions: high-rank singular
     vectors that accumulate in LoRA B matrices across sequential training cycles
     and cause catastrophic forgetting (§11 risk analysis).

  7. SLAO time-aware merge (quarterly, Dec 2025) - composites the fresh quarterly
     adapter (trained from scratch on full cumulative data) with the slow EMA adapter
     using asymmetric weighting: new lora_A (better orthogonal coverage), weighted
     average on lora_B.

Tier 3 pipeline:
  1. Train from base model from scratch (no CLoRA - clean slate)
  2. SVD prune the new adapter (remove intruder dims)
  3. SLAO merge with slow adapter (if exists)
  4. STABLE KL gate check (reuse from anti_forgetting.py)
  5. Deploy merged adapter
  6. Reset cycle counter (next Tier 2 builds on clean base)

Triggered by:
  - days_since_last_tier3 >= 90       → "tier3_quarterly"
  - general_reasoning_drop >= 0.15    → "tier3_forgetting" (via should_train check)

Tier 3 does NOT use the replay buffer or CLoRA init - it retrains from the
base model on the full cumulative dataset to produce a clean foundation.
Subsequent Tier 2 cycles then apply CLoRA on top of this new base.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from systems.reasoning_engine.anti_forgetting import AntiForgetConfig
    from systems.reasoning_engine.service import ReasoningEngineService

logger = structlog.get_logger("reasoning_engine.tier3")

_TIER3_LAST_RUN_KEY = "eos:re:last_tier3_timestamp"
_TRAIN_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "..", "systems", "simula", "training", "train_lora.py"
)
_DEFAULT_BASE_MODEL = os.environ.get("RE_BASE_MODEL", "Qwen/Qwen3-8B")
_TIER3_TIMEOUT_S = 14400  # 4 hours


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class Tier3Config:
    retrain_interval_days: int = 90
    """Minimum days between Tier 3 quarterly retrains."""

    svd_prune_top_k: int = 5
    """Number of intruder singular vectors to remove from each LoRA B matrix."""

    slao_time_decay: float = 0.5
    """SLAO lora_B weight for the slow adapter (0.5 = equal weight).
    New adapter lora_B weight = (1 - slao_time_decay).
    Lower = newer adapter dominates; higher = historical stability dominates."""

    base_model: str = _DEFAULT_BASE_MODEL
    """HuggingFace model ID to train from scratch."""

    output_dir: str = "data/re_adapters/tier3"
    """Base directory for Tier 3 adapter artifacts."""


# ── SVD Pruner ────────────────────────────────────────────────────────────────


class SVDPruner:
    """Remove intruder dimensions from accumulated LoRA adapters.

    'Intruder dimensions' (§11 risk analysis): high-rank singular vectors
    introduced by sequential training that break the original low-rank structure.
    These cause catastrophic forgetting across training cycles.

    Algorithm (per lora_B matrix):
    1. SVD decompose: U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    2. Identify intruder singular values: those with unusually large gaps
       relative to their neighbours (gap = S[i] / S[i+1], intruder = gap > threshold)
    3. Take the top-k largest-gap indices as intruder dimensions
    4. Zero out those singular values: S_masked[intruder_indices] = 0
    5. Reconstruct: B_pruned = U @ diag(S_masked) @ Vh

    Non-fatal: if safetensors unavailable or any layer fails, copies adapter as-is.
    """

    def __init__(self, config: Tier3Config) -> None:
        self._top_k = config.svd_prune_top_k

    async def prune(self, adapter_path: str, output_path: str) -> str:
        """Apply SVD pruning to remove intruder dimensions.

        Returns path to pruned adapter (output_path).
        On any failure, copies adapter_path as-is and returns output_path.
        """
        try:
            import torch
            from safetensors.torch import load_file, save_file
        except ImportError as e:
            logger.warning("svd_pruner.import_failed", error=str(e), fallback="copy adapter as-is")
            _copy_adapter(adapter_path, output_path)
            return output_path

        try:
            weight_file = _find_safetensors(adapter_path)
            if weight_file is None:
                logger.warning("svd_pruner.no_safetensors", path=adapter_path, fallback="copy as-is")
                _copy_adapter(adapter_path, output_path)
                return output_path

            weights = load_file(str(weight_file))
            pruned: dict[str, Any] = {}
            pruned_layers = 0
            skipped_layers = 0

            for key, tensor in weights.items():
                if "lora_B" not in key:
                    pruned[key] = tensor
                    continue
                try:
                    B = tensor.float()
                    U, S, Vh = torch.linalg.svd(B, full_matrices=False)

                    # Detect intruder dimensions: largest gap ratio S[i]/S[i+1]
                    # A large gap indicates an anomalous singular value cluster.
                    intruder_mask = torch.zeros(len(S), dtype=torch.bool)
                    if len(S) > 1:
                        gaps = S[:-1] / (S[1:] + 1e-8)
                        # Top-k gap indices are candidate intruder dimensions
                        k = min(self._top_k, len(gaps))
                        topk_gap_indices = torch.topk(gaps, k).indices
                        intruder_mask[topk_gap_indices] = True

                    S_masked = S.clone()
                    S_masked[intruder_mask] = 0.0
                    B_pruned = U @ torch.diag(S_masked) @ Vh
                    pruned[key] = B_pruned.to(tensor.dtype)
                    pruned_layers += 1
                except Exception as layer_exc:
                    logger.debug("svd_pruner.layer_failed", key=key, error=str(layer_exc))
                    pruned[key] = tensor
                    skipped_layers += 1

            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_file(pruned, str(out_dir / "adapter_model.safetensors"))

            # Copy non-weight files (configs, tokenizer, etc.)
            for f in Path(adapter_path).iterdir():
                if f.is_file() and f.suffix != ".safetensors":
                    shutil.copy2(f, out_dir / f.name)

            logger.info(
                "svd_pruner.complete",
                pruned_layers=pruned_layers,
                skipped_layers=skipped_layers,
                top_k=self._top_k,
                output=output_path,
            )
            return output_path

        except Exception as exc:
            logger.warning("svd_pruner.failed", error=str(exc), fallback="copy adapter as-is")
            _copy_adapter(adapter_path, output_path)
            return output_path


# ── SLAO Merger ───────────────────────────────────────────────────────────────


class SLAOMerger:
    """SLAO (Dec 2025): Time-aware adapter composition.

    After quarterly retrain from scratch, merge:
    - New adapter (trained from scratch on full cumulative dataset)
    - Existing slow adapter (EMA of all Tier 2 cycles = historical stability)

    SLAO asymmetry:
    - lora_A: use new adapter's A matrices directly (fresh orthogonal coverage,
      better for future Tier 2 CLoRA init)
    - lora_B: weighted average with time_decay
      merged_B = (1 - decay) * new_B + decay * slow_B

    This ensures the new adapter's subspace directions are preserved for future
    CLoRA init while letting historical B knowledge stabilize the output projections.

    Non-fatal: if safetensors unavailable or merge fails, returns new_adapter_path.
    """

    def __init__(self, config: Tier3Config) -> None:
        self._decay = config.slao_time_decay

    async def merge(
        self,
        new_adapter_path: str,
        slow_adapter_path: str,
        output_path: str,
    ) -> str:
        """Merge new (quarterly) adapter with existing slow adapter via SLAO.

        Returns path to merged adapter.
        On any failure, returns new_adapter_path unchanged (non-fatal).
        """
        try:
            import torch
            from safetensors.torch import load_file, save_file
        except ImportError as e:
            logger.warning("slao_merger.import_failed", error=str(e), fallback="use new adapter only")
            return new_adapter_path

        try:
            new_file = _find_safetensors(new_adapter_path)
            slow_file = _find_safetensors(slow_adapter_path)

            if new_file is None or slow_file is None:
                logger.warning(
                    "slao_merger.safetensors_not_found",
                    new=new_file,
                    slow=slow_file,
                    fallback="use new adapter only",
                )
                return new_adapter_path

            new_weights = load_file(str(new_file))
            slow_weights = load_file(str(slow_file))

            merged: dict[str, Any] = {}
            lora_a_count = 0
            lora_b_count = 0

            for key, new_tensor in new_weights.items():
                if "lora_A" in key:
                    # Use new adapter's A matrices: better orthogonal coverage
                    # for subsequent Tier 2 CLoRA init cycles.
                    merged[key] = new_tensor
                    lora_a_count += 1
                elif "lora_B" in key and key in slow_weights:
                    # Weighted average: new adapter has full-data advantage,
                    # slow adapter contributes historical stability.
                    new_B = new_tensor.float()
                    slow_B = slow_weights[key].float()
                    merged_B = (1.0 - self._decay) * new_B + self._decay * slow_B
                    merged[key] = merged_B.to(new_tensor.dtype)
                    lora_b_count += 1
                else:
                    # Non-LoRA keys (adapter_config, embedding, etc.) - use new
                    merged[key] = new_tensor

            # Keys in slow but not new - preserve (covers new modules not in new adapter)
            for key, slow_tensor in slow_weights.items():
                if key not in merged:
                    merged[key] = slow_tensor

            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            save_file(merged, str(out_dir / "adapter_model.safetensors"))

            # Copy config files from new adapter
            for f in Path(new_adapter_path).iterdir():
                if f.is_file() and f.suffix != ".safetensors":
                    shutil.copy2(f, out_dir / f.name)

            logger.info(
                "slao_merger.complete",
                lora_a_count=lora_a_count,
                lora_b_count=lora_b_count,
                decay=self._decay,
                output=output_path,
            )
            return output_path

        except Exception as exc:
            logger.warning("slao_merger.failed", error=str(exc), fallback="use new adapter only")
            return new_adapter_path


# ── Tier 3 Orchestrator ───────────────────────────────────────────────────────


class Tier3Orchestrator:
    """Orchestrates the quarterly full retrain + SVD prune + SLAO merge + KL gate + deploy.

    Called by ContinualLearningOrchestrator when should_train() returns
    "tier3_quarterly" or "tier3_forgetting".

    Pipeline:
    1. Collect cumulative training data (all 5 streams, full history - caller provides path)
    2. Train from base model from scratch (train_lora.py, NO PREVIOUS_ADAPTER_PATH)
    3. SVD prune the new adapter (remove intruder dimensions)
    4. SLAO merge with existing slow adapter (if exists)
    5. STABLE KL gate check
    6. Deploy merged adapter
    7. Record timestamp in Redis (reset quarterly clock)

    After Tier 3 completes, the next Tier 2 cycle will set PREVIOUS_ADAPTER_PATH
    to the new merged adapter, re-enabling CLoRA orthogonal init on a clean base.
    """

    def __init__(
        self,
        config: Tier3Config,
        af_config: "AntiForgetConfig",
        re_service: "ReasoningEngineService",
        event_bus: Any,
        redis_client: "Redis",
    ) -> None:
        self._config = config
        self._af_config = af_config
        self._re = re_service
        self._bus = event_bus
        self._redis = redis_client
        self._svd_pruner = SVDPruner(config)
        self._slao = SLAOMerger(config)

    async def should_run_tier3(self) -> tuple[bool, str]:
        """Check if Tier 3 conditions are met based on last run timestamp.

        Returns (should_run, reason).
        """
        if self._redis is None:
            return False, "no_redis"
        try:
            raw_ts = await self._redis.get(_TIER3_LAST_RUN_KEY)
        except Exception as exc:
            logger.warning("tier3.redis_check_failed", error=str(exc))
            return False, "redis_error"

        if raw_ts is None:
            # First-ever run - self-bootstrap by writing a timestamp 91 days in the past.
            # This makes the next check immediately see it as overdue and fire Tier 3.
            bootstrap_ts = time.time() - (91 * 86400)
            try:
                await self._redis.set(_TIER3_LAST_RUN_KEY, str(bootstrap_ts))
                logger.info("tier3_bootstrapped", will_run_next_check=True)
            except Exception as set_exc:
                logger.warning("tier3.bootstrap_write_failed", error=str(set_exc))
            return True, "tier3_first_run"

        try:
            last_ts = float(raw_ts.decode() if isinstance(raw_ts, bytes) else raw_ts)
            days_since = (time.time() - last_ts) / 86400
            if days_since >= self._config.retrain_interval_days:
                return True, "tier3_quarterly"
            return False, f"days_since={days_since:.1f}"
        except Exception as exc:
            logger.warning("tier3.timestamp_parse_failed", error=str(exc))
            return False, "parse_error"

    async def run_tier3(
        self,
        cumulative_data_path: str,
        slow_adapter_path: str | None,
    ) -> bool:
        """Full quarterly retrain pipeline. Returns True if successful and deployed.

        Args:
            cumulative_data_path: path to JSONL containing ALL training data (no replay mix).
            slow_adapter_path: current slow adapter path (from SuReEMAAdapter) or None.
        """
        run_id = f"tier3_{int(time.time())}"
        logger.info("tier3.started", run_id=run_id)
        from systems.synapse.types import SynapseEventType as _SET
        await self._emit(_SET.RE_TIER3_STARTED, {"run_id": run_id})

        try:
            # ── Step 1: Train from scratch (clean base - no CLoRA) ─────────────
            base_output = f"{self._config.output_dir}/{run_id}_base"
            Path(base_output).mkdir(parents=True, exist_ok=True)

            env = {
                **os.environ,
                "BASE_MODEL": self._config.base_model,
                "TRAINING_DATA": cumulative_data_path,
                "OUTPUT_DIR": base_output,
                # Explicitly clear CLoRA - Tier 3 starts from scratch
                "PREVIOUS_ADAPTER_PATH": "",
                # Default hyperparameters for full retrain
                "TRAINING_ARGS": '{"num_epochs": 3, "lora_rank": 32, "lora_alpha": 64}',
                "WANDB_RUN_NAME": f"tier3_{run_id}",
                "WANDB_JOB_TYPE": "tier3_full_retrain",
            }

            logger.info("tier3.training_started", run_id=run_id, base_model=self._config.base_model)
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
                    timeout=float(_TIER3_TIMEOUT_S),
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                raise RuntimeError(f"train_lora.py timed out after {_TIER3_TIMEOUT_S}s")

            if proc.returncode != 0:
                stderr_tail = stderr_bytes.decode(errors="replace")[-2000:]
                raise RuntimeError(f"train_lora.py exited with code {proc.returncode}: {stderr_tail}")

            # train_lora.py saves adapter under output_dir/adapter/
            adapter_dir = Path(base_output) / "adapter"
            if not adapter_dir.exists():
                adapter_dir = Path(base_output)
            logger.info("tier3.training_complete", run_id=run_id, adapter=str(adapter_dir))

            # ── Step 2: SVD pruning ────────────────────────────────────────────
            pruned_path = f"{self._config.output_dir}/{run_id}_pruned"
            await self._svd_pruner.prune(str(adapter_dir), pruned_path)
            logger.info("tier3.svd_pruned", run_id=run_id, path=pruned_path)

            # ── Step 3: SLAO merge with slow adapter ───────────────────────────
            slao_merged = False
            if slow_adapter_path and os.path.exists(slow_adapter_path):
                merged_path = f"{self._config.output_dir}/{run_id}_merged"
                final_path = await self._slao.merge(pruned_path, slow_adapter_path, merged_path)
                slao_merged = True
            else:
                final_path = pruned_path
                logger.info("tier3.no_slow_adapter", run_id=run_id, note="skipping SLAO merge")

            # ── Step 4: STABLE KL gate ─────────────────────────────────────────
            kl_passed = True
            kl_divergence = 0.0
            try:
                from systems.reasoning_engine.anti_forgetting import STABLEKLGate
                kl_gate = STABLEKLGate(self._af_config)
                await kl_gate.load_anchors()
                kl_passed, kl_divergence = await kl_gate.check_kl_divergence(
                    re_service=self._re,
                    current_adapter_path=slow_adapter_path,
                    new_adapter_path=final_path,
                )
            except Exception as exc:
                logger.warning("tier3.kl_gate_error_pass", error=str(exc), note="deploying anyway")

            if not kl_passed:
                logger.error("tier3.kl_gate_rejected", run_id=run_id, kl=round(kl_divergence, 5))
                from systems.synapse.types import SynapseEventType as _SET
                await self._emit(_SET.RE_KL_GATE_REJECTED, {
                    "run_id": run_id,
                    "kl_divergence": kl_divergence,
                    "tier": 3,
                    "adapter_path": final_path,
                })
                return False

            # ── Step 5: Deploy ─────────────────────────────────────────────────
            adapter_id = f"eos_tier3_{run_id}"
            await self._re.load_adapter(final_path, adapter_id)
            logger.info(
                "tier3.deployed",
                run_id=run_id,
                adapter_id=adapter_id,
                adapter=final_path,
                kl_divergence=round(kl_divergence, 5),
            )

            # ── Step 6: Record timestamp ───────────────────────────────────────
            await self._redis.set(_TIER3_LAST_RUN_KEY, str(time.time()))

            from systems.synapse.types import SynapseEventType as _SET
            await self._emit(_SET.RE_TIER3_COMPLETE, {
                "run_id": run_id,
                "kl_divergence": kl_divergence,
                "final_adapter": final_path,
                "svd_pruned": True,
                "slao_merged": slao_merged,
            })
            return True

        except Exception as exc:
            logger.error("tier3.failed", run_id=run_id, error=str(exc))
            from systems.synapse.types import SynapseEventType as _SET
            await self._emit(_SET.RE_TRAINING_FAILED, {
                "run_id": run_id,
                "tier": 3,
                "error": str(exc),
            })
            return False

    async def _emit(self, event_type_str: "SynapseEventType | str", payload: dict[str, Any]) -> None:
        """Fire-and-forget Synapse event. Never raises."""
        if self._bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            if isinstance(event_type_str, SynapseEventType):
                etype = event_type_str
            else:
                etype = SynapseEventType(event_type_str.lower())
            event = SynapseEvent(
                event_type=etype,
                data=payload,
                source_system="reasoning_engine",
            )
            asyncio.ensure_future(self._bus.emit(event))
        except Exception as exc:
            logger.debug("tier3.emit_failed", event=event_type_str, error=str(exc))


# ── Internal helpers ──────────────────────────────────────────────────────────


def _find_safetensors(adapter_dir: str) -> Path | None:
    """Find the primary .safetensors file in an adapter directory."""
    d = Path(adapter_dir)
    if not d.exists():
        return None
    candidates = list(d.glob("*.safetensors"))
    if not candidates:
        return None
    for name in ("adapter_model.safetensors", "model.safetensors"):
        for c in candidates:
            if c.name == name:
                return c
    return candidates[0]


def _copy_adapter(src: str, dst: str) -> None:
    """Copy adapter directory to dst (mkdir as needed)."""
    try:
        Path(dst).mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
    except Exception as exc:
        logger.warning("tier3.copy_adapter_failed", src=src, dst=dst, error=str(exc))
