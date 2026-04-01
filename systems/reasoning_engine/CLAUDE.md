# Reasoning Engine - CLAUDE.md

**Spec:** `.claude/ECODIAOS_CUSTOM_LLM_SPEC.md` + `.claude/speciation_bible.md §4.1–4.2`
**System ID:** `reasoning_engine`
**Role:** Local LLM substrate. Wraps a vLLM-served model (Qwen3-8B or fine-tuned variant) as an `LLMProvider` so Nova's `PolicyGenerator` can route slow-path deliberation to it via Thompson sampling. The RE is the organism's path to genuine LLM self-sufficiency - the model that improves from the organism's own experience.

---

## What's Implemented

### `ReasoningEngineService` (`service.py`)

Implements the full `LLMProvider` ABC from `clients/llm.py`.

**Constructor:**
```python
ReasoningEngineService(
    vllm_url: str | None = None,   # from ECODIAOS_RE_VLLM_URL or http://localhost:8001/v1
    model_name: str | None = None, # from ECODIAOS_RE_MODEL or ecodiaos-reasoning
    synapse: Any = None,           # injected post-startup via set_synapse()
)
```

**Probe:** `initialize()` - hits `GET /v1/models`, confirms model is listed. Sets `_available=True`. Non-fatal if vLLM is unreachable (Claude-only mode).

**Circuit breaker:**
- Tracks consecutive failures; opens after 5 (`_CIRCUIT_BREAKER_THRESHOLD`)
- `is_available` property: `self._available and not self._circuit_open`
- Resets on first success; logs `reasoning_engine_circuit_closed`
- Emits `RE_ENGINE_STATUS_CHANGED` on open/close transitions

**LLMProvider methods:**
| Method | Status | Notes |
|--------|--------|-------|
| `generate()` | ✅ | Primary use path - vLLM `/v1/chat/completions`; JSON mode via `response_format` |
| `evaluate()` | ✅ | Thin wrapper around `generate()` with minimal system prompt |
| `generate_with_tools()` | ✅ | Satisfies ABC; vLLM tool-call support is model-dependent - Claude handles tool use in practice |
| `close()` | ✅ | Closes httpx AsyncClient |
| `load_adapter()` | ✅ | Tries dynamic `/v1/load_lora_adapter`; graceful 404 fallback for startup-loaded adapters (`--lora-modules`) |
| `unload_adapter()` | ✅ | Tries dynamic `/v1/unload_lora_adapter`; graceful 404 fallback |
| `supports_adapters` | ✅ | Returns `True` |

**Timeout:** 30s for generate, 5s for startup probe.

---

## Configuration (env vars)

| Variable | Default | Purpose |
|----------|---------|---------|
| `ECODIAOS_RE_VLLM_URL` | `http://localhost:8001/v1` | vLLM OpenAI-compatible server base URL |
| `ECODIAOS_RE_MODEL` | `ecodiaos-reasoning` | Model name served on vLLM |
| `ECODIAOS_RE_ENABLED` | `true` | Set to `false` to disable entirely |

---

## Integration Points

### Nova / PolicyGenerator
- `ReasoningEngineService` is passed as `re_client` to `PolicyGenerator.__init__()`
- `ThompsonSampler.set_re_ready(True)` is called in `registry._init_nova()` when `re_service.is_available`
- Until called, `ThompsonSampler` always returns `"claude"` - zero routing to RE
- Thompson sampling is Beta-Bernoulli; RE must earn trust through demonstrated outcomes

### Registry wiring (`core/registry.py`)
- `_init_nova()` creates `ReasoningEngineService`, calls `initialize()`
- On success: passes as `re_client`, calls `set_re_ready(True)`
- On failure / disabled: logs info, continues Claude-only
- `app.state.reasoning_engine` - accessible from API health endpoints

### Synapse events
| Event | Direction | Trigger |
|-------|-----------|---------|
| `RE_ENGINE_STATUS_CHANGED` | Emitted | Circuit opens (available→False) or closes (available→True) |

Benchmarks can subscribe to `RE_ENGINE_STATUS_CHANGED` to track `llm_dependency` KPI transitions.

### LoRA Adapter Pipeline (future)
- `load_adapter(adapter_path, adapter_id)` - tries dynamic `/v1/load_lora_adapter` first; falls back to client-side tracking if 404 (adapter loaded at vLLM startup via `--lora-modules`)
- CLoRA fine-tuning pipeline produces adapter `.safetensors` files
- For dynamic loading: pipeline calls `app.state.reasoning_engine.load_adapter()` after each training run (requires `--enable-lora` on vLLM)
- For static loading: restart vLLM with `--lora-modules adapter_name=path/to/adapter`
- `active_adapter_id` tracks current loaded adapter (IPFS CID by convention)

---

## Continual Learning Orchestrator (`continual_learning.py`)

**Implemented:** 2026-03-07
**Status:** Wired in `core/registry.py` after RE exporter; daily background task

### What it does
Ties the full self-improvement cycle together:

```
Neo4j (5 streams) → quality scoring → scaffold formatting → JSONL
      ↓
asyncio subprocess: train_lora.py  (Qwen3-8B base, LoRA r=32/α=64)
      ↓
ReasoningEngineService.load_adapter() - hot-swap into vLLM, no restart
```

### Key types
- `TrainingTrigger` - threshold config (min_new_examples=300, max_days=14, drop_threshold=0.05)
- `TrainingRun` - immutable record; persisted to Redis as JSON
- `ContinualLearningOrchestrator` - main class; `check_and_train()` is the daily entry point

### Trigger conditions (priority order)
1. `tier3_scheduled_fallback_tier2` - ≥90 days since last train (Tier 3 placeholder → runs Tier 2)
2. `tier2_data_volume` - ≥300 new examples in Neo4j (14-day window)
3. `tier2_scheduled` - ≥14 days since last train
4. `tier2_first_run` - never trained + ≥50 examples exist
5. `tier2_degradation` - Thompson sampler success rate < 0.45

### Dataset-size-adaptive hyperparameters (bible §5)
| Dataset size | r | α | lr | epochs | batch |
|---|---|---|---|---|---|
| <500 | 32 | 64 | 3e-4 | 2 | 4 |
| 500-2k | 32 | 64 | 2e-4 | 3 | 4 |
| >2k | 32 | 64 | 1e-4 | 4 | 8 |

### Synapse events emitted
| Event | Trigger |
|-------|---------|
| `RE_TRAINING_STARTED` | Run begins; payload: run_id, tier, trigger_reason, examples_available |
| `RE_TRAINING_COMPLETE` | Training succeeded + slow adapter loaded; payload: run_id, tier, examples_used, eval_loss, adapter_id, kl_divergence |
| `RE_TRAINING_FAILED` | Training subprocess failed or timed out; payload: run_id, tier, reason |
| `RE_KL_GATE_REJECTED` | STABLE KL gate blocked deployment; payload: run_id, kl_divergence, budget, adapter_path |
| `BENCHMARK_REGRESSION` | Anchor perplexity spiked >20% above baseline; payload: metric, current, baseline, spike_pct |
| `MODEL_ROLLBACK_TRIGGERED` | Post-deploy quality check failed (post_rate < pre_rate × 0.90); payload: run_id, reason, pre_success_rate, post_success_rate, window_attempts, rollback_adapter, auto_rollback=True |
| `RE_ADAPTER_QUALITY_CONFIRMED` | Post-deploy quality check passed (post_rate > pre_rate × 1.05); payload: run_id, pre_success_rate, post_success_rate, improvement_pct, window_attempts |
| `RE_TRAINING_EXAMPLE` | Emitted on adapter rollback as a training signal for future cycles |

### Trigger reasons (added 2026-03-08)
| Reason | Threshold | Description |
|--------|-----------|-------------|
| `tier2_urgent_requested` | ≥50 examples (lowered from 300) | `RE_TRAINING_REQUESTED` event received; `_urgent_training_requested = True` |

### Post-Deployment Quality Monitoring (2026-03-08)

After every successful adapter deploy, the CLO opens a 500-cycle monitoring window.
`NovaService._on_axon_execution_result()` calls `clo.record_re_outcome(success)` for every
RE decision.  When the window fills, `_evaluate_post_deploy_quality()` runs:

| Outcome | Condition | Action |
|---------|-----------|--------|
| **Rollback** | `post_rate < pre_rate × 0.90` (≥10% degradation) | `_rollback_adapter()` - restore previous adapter, reset Thompson "re" Beta params, emit `MODEL_ROLLBACK_TRIGGERED` + `RE_TRAINING_EXAMPLE` |
| **Confirm** | `post_rate > pre_rate × 1.05` (≥5% improvement) | `_confirm_adapter()` - emit `RE_ADAPTER_QUALITY_CONFIRMED`; adapter becomes new baseline |
| **Neutral** | Within ±5–10% of baseline | Log only; keep current adapter |

**State fields added to `ContinualLearningOrchestrator`:**
| Field | Type | Purpose |
|-------|------|---------|
| `_pre_deploy_baseline` | `dict \| None` | Snapshot of success_rate + eval_loss before deploy; persisted to Redis `eos:re:pre_deploy_baseline` |
| `_pre_deploy_adapter_path` | `str \| None` | Previous adapter path; restored on rollback |
| `_post_deploy_successes` | `int` | Accumulated successes in monitoring window |
| `_post_deploy_attempts` | `int` | Total RE decisions in monitoring window |
| `_monitoring_active` | `bool` | True while window is open |

**Wiring:**
- `nova.set_clo(clo)` - called in `core/registry.py` after CLO init; injects CLO reference into Nova
- `clo.record_re_outcome(success)` - called from `NovaService._on_axon_execution_result()` for every RE decision outcome (non-fatal, no-op when monitoring inactive)
- Thompson arm reset on rollback: approximates pre-deploy Beta as `alpha=rate×20, beta=(1−rate)×20`, patched into Redis hash `nova:thompson_sampler`

**New Redis key:**
| Key | Value |
|-----|-------|
| `eos:re:pre_deploy_baseline` | JSON dict: success_rate, eval_loss, cycle, timestamp, adapter_path |

**New SynapseEventType entries:**
| Event | Purpose |
|-------|---------|
| `RE_ADAPTER_QUALITY_CONFIRMED` | Deployment quality verified (≥5% improvement) |
| `MODEL_ROLLBACK_TRIGGERED` | Already existed; now also emitted on quality degradation rollback |

### Redis keys
| Key | Value |
|-----|-------|
| `eos:re:last_train_at` | ISO-8601 datetime of last completed run |
| `eos:re:training_runs` | JSON list of last 50 `TrainingRun` records |
| `eos:re:thompson_success_rate` | Float written by Nova; read for degradation trigger |
| `eos:re:pre_deploy_baseline` | JSON dict of pre-deployment quality snapshot |

### Failure safety
- Training failure **never crashes the organism** - always caught, logged, `RE_TRAINING_FAILED` emitted
- Adapter deployment failure recorded on run but does not mark training as failed
- Organism continues on Claude-only if vLLM is unavailable

### Configuration (env vars)
| Variable | Default | Purpose |
|----------|---------|---------|
| `RE_TRAINING_EXPORT_DIR` | `data/re_training_batches` | JSONL output + adapter dirs |
| `RE_BASE_MODEL` | `Qwen/Qwen3-8B` | Base model passed to train_lora.py |
| `RE_TRAINING_TIMEOUT_S` | `7200` | Subprocess timeout (2 hours) |

### Redis keys
| Key | Writer | Reader | Purpose |
|-----|--------|--------|---------|
| `eos:re:last_train_at` | CLO | CLO check | ISO-8601 datetime of last completed run |
| `eos:re:training_runs` | CLO | CLI history/status | JSON list of last 50 `TrainingRun` records |
| `eos:re:thompson_success_rate` | Nova (`_on_axon_execution_result`) | CLO (degradation trigger), CLI status | RE Beta posterior mean - written after each RE decision |
| `eos:re:success_rate_7d` | Nova (`_on_axon_execution_result`) | CLI status | Same value as `thompson_success_rate` - semantic alias for 7-day window |
| `nova:thompson_sampler` | ThompsonSampler (`persist_to_redis`) | ThompsonSampler restore, CLI status | Full {claude_alpha, claude_beta, re_alpha, re_beta} state |

### CLI
```bash
python -m cli.training_run check     # dry-run: should training trigger?
python -m cli.training_run run       # force Tier 2 now
python -m cli.training_run history   # show run history from Redis
python -m cli.training_run status    # RE available?, success rate, Thompson params, last run, adapter loaded
```

---

## Anti-Forgetting Stack (`anti_forgetting.py`) - Round 3A

Implements 4 of 7 mechanisms from Speciation Bible §3.3, wrapping around `train_lora.py` (never modified):

### Pipeline flow (each Tier 2 cycle)

```
train_lora.py (fast adapter)
        ↓
Step 3b: SurprisePrioritizedReplay.sample(300) → mixed into JSONL before subprocess
        ↓
Step 6a: SurprisePrioritizedReplay.add_examples() → new examples buffered in Redis
        ↓
Step 6b: SuReEMAAdapter.update_slow_adapter() → fast merged into slow (EMA on safetensors)
        ↓
Step 6c: STABLEKLGate.check_kl_divergence() → reject if behavioural shift > KL budget
        ↓
Step 6d: re_service.load_adapter(slow_path) → SLOW adapter deployed (not fast)
        ↓
Step 6e: AnchorPerplexityMonitor.check_and_alarm() → non-blocking forgetting alarm
```

### Classes

| Class | Mechanism | Paper |
|-------|-----------|-------|
| `SurprisePrioritizedReplay` | Replay buffer | ERI-LoRA (2024) |
| `SuReEMAAdapter` | Dual fast/slow EMA | SuRe (2025) |
| `STABLEKLGate` | KL divergence gate | STABLE (NeurIPS 2025) |
| `AnchorPerplexityMonitor` | Perplexity alarm | Forgetting detection |

### Anchor prompts

`data/re_training_batches/anchor_prompts.jsonl` - 30 prompts across 4 categories:
- `economic_reasoning` (10): yield strategy, compute arbitrage, metabolic efficiency
- `causal_reasoning` (10): invariant application, active inference, epistemic conflict
- `constitutional_alignment` (5): drive tradeoffs, Equor verdicts, honesty violations
- `planning` (5): upgrade sequences, incident triage, certificate renewal

**CRITICAL:** These prompts MUST NEVER appear in any training data JSONL. They are read-only behavioural probes.

### Config (`AntiForgetConfig`)

| Field | Default | Purpose |
|-------|---------|---------|
| `replay_buffer_size` | 500 | Max Redis sorted set size |
| `replay_surprise_weight` | 0.7 | Bias toward high-surprise examples when sampling |
| `ema_decay` | 0.99 | SuRe EMA decay coefficient for slow adapter |
| `kl_budget` | 0.1 | STABLE KL divergence gate threshold (nats) |
| `anchor_perplexity_alarm` | 0.20 | Perplexity spike fraction to trigger BENCHMARK_REGRESSION |
| `svd_prune_top_k` | 5 | Reserved for Round 4 SVD pruning |

### Synapse events (new)

| Event | Trigger | Payload |
|-------|---------|---------|
| `RE_KL_GATE_REJECTED` | STABLE KL gate blocks adapter | `run_id`, `kl_divergence`, `budget`, `adapter_path` |

### Redis keys (anti-forgetting)

| Key | Value |
|-----|-------|
| `eos:re:replay_buffer` | Sorted set: JSON examples scored by surprise priority |
| `eos:re:anchor_perplexity_baseline` | Float: baseline perplexity from first measurement |

### Deployment note

Production vLLM always receives the **slow adapter** (EMA-stabilised), not the fast adapter. The fast adapter is the raw training output; the slow adapter accumulates knowledge across all training cycles.

### Round 4A - CLoRA + Tier 3 Quarterly Retrain (implemented 2026-03-07)

All 3 remaining anti-forgetting mechanisms from §3.3 are now implemented.

#### CLoRA Orthogonal Subspace Init (ACL 2025)

Applied at **Tier 2 init time** via `PREVIOUS_ADAPTER_PATH` env var injected by `ContinualLearningOrchestrator._execute_tier2()`.

**Algorithm in `train_lora.py:_apply_clora_init()`:**
1. Load previous adapter's LoRA A matrices from safetensors
2. SVD decompose each A_prev: `U, S, Vh = torch.linalg.svd(A_prev, full_matrices=False)`
3. Null space projector: `P = I - Vh.T @ Vh`  (orthogonal complement of previous directions)
4. Re-init: `new_A = randn @ P` - ensures new directions are orthogonal to previously learned features

**Non-fatal when:**
- `PREVIOUS_ADAPTER_PATH` not set (empty string → skipped)
- `adapter_model.safetensors` missing at path → skipped with log
- safetensors not installed → skipped with log
- Per-layer SVD fails → that layer skipped, others continue

**Env var:** `PREVIOUS_ADAPTER_PATH` - set by CLO to `self._current_adapter_path` for all Tier 2 runs. Empty string for Tier 3 (clean slate).

#### SVD Pruning - `tier3.SVDPruner`

Quarterly. Removes intruder dimensions from LoRA B matrices before deployment.

**Algorithm (per lora_B layer):**
1. `U, S, Vh = torch.linalg.svd(B, full_matrices=False)`
2. Compute gap ratios: `S[i] / S[i+1]` - large gap = anomalous singular value cluster
3. Zero out top-k largest-gap singular values
4. Reconstruct: `B_pruned = U @ diag(S_masked) @ Vh`

Config: `Tier3Config.svd_prune_top_k` (default 5, shared with `AntiForgetConfig.svd_prune_top_k`)

#### SLAO Time-Aware Merge (Dec 2025) - `tier3.SLAOMerger`

Quarterly. Composites fresh quarterly adapter with existing slow adapter.

**SLAO asymmetry:**
- `lora_A`: use new adapter's A matrices (better orthogonal coverage for future CLoRA init)
- `lora_B`: weighted average: `(1 - decay) * new_B + decay * slow_B`
  - Default decay = 0.5 (equal weight)
  - New adapter weighted higher (trained on full cumulative data)

#### Tier 3 Orchestrator - `tier3.Tier3Orchestrator`

Full pipeline (up to 4h, `asyncio.wait_for(..., timeout=14400)`):
1. `_build_cumulative_dataset()` - 90-day lookback, min_score=0.20
2. `train_lora.py` from scratch (no `PREVIOUS_ADAPTER_PATH`)
3. `SVDPruner.prune()` → intruder dimensions removed
4. `SLAOMerger.merge()` with existing slow adapter (if exists)
5. `STABLEKLGate.check_kl_divergence()` - reject if behavioural shift > budget
6. `re_service.load_adapter()` - deploy merged adapter
7. `redis.set(eos:re:last_tier3_timestamp, time.time())` - reset quarterly clock

**Trigger condition:** `Tier3Orchestrator.should_run_tier3()` checks `eos:re:last_tier3_timestamp` vs `retrain_interval_days` (default 90).

**Routing:** `ContinualLearningOrchestrator.run_tier2(reason)` intercepts `"tier3"` in reason string and calls `_tier3.run_tier3()` before acquiring the Tier 2 lock.

**After Tier 3:** the next Tier 2 cycle will pass the new merged adapter as `PREVIOUS_ADAPTER_PATH`, re-enabling CLoRA orthogonal init on the clean base.

### New Synapse events (Round 4A)

| Event | Trigger | Payload |
|-------|---------|---------|
| `RE_TIER3_STARTED` | Tier 3 begins | `run_id` |
| `RE_TIER3_COMPLETE` | Tier 3 successful | `run_id, kl_divergence, final_adapter, svd_pruned, slao_merged` |

### New Redis keys (Tier 3)

| Key | Value |
|-----|-------|
| `eos:re:last_tier3_timestamp` | Unix timestamp (float) of last successful Tier 3 run |

### New files (Round 4A)

| File | Purpose |
|------|---------|
| `tier3.py` | `SVDPruner`, `SLAOMerger`, `Tier3Orchestrator`, `Tier3Config` |

---

## Safety Layer (`safety.py`) - Round 3B

Implements Speciation Bible §7.2 + §7.3 kill switches. All operations non-fatal.

### Pipeline position

```
train_lora.py (fast adapter)
        ↓
Step 6b: SuReEMAAdapter.update_slow_adapter()
        ↓
Step 6b.5: SafeLoRAProjection.project()     ← §7.2 (NEW)
        ↓
Step 6c: STABLEKLGate.check_kl_divergence()
        ↓
Step 6d: re_service.load_adapter(slow_path)
```

### Classes

| Class | Role |
|-------|------|
| `SafetyConfig` | Threshold dataclass (floors + window sizes) |
| `SafeLoRAProjection` | Project LoRA B-matrices onto safety-aligned subspace using constitutional scenarios as proxy. Scales down LoRA delta when violation rate exceeds budget. Non-fatal - returns original path on any failure. |
| `RESuccessRateMonitor` | Track RE 7-day rolling success rate via Redis Stream (`eos:re:outcomes`). Tier 2 kill switch: rate < 0.50 → emit `RE_TRAINING_HALTED`. Writes canonical `eos:re:success_rate_7d`. |
| `RedTeamEvaluator` | Monthly 50-prompt adversarial evaluation. 5 categories × 10 prompts. Tier 2 kill switch: pass_rate < 0.70 → emit `RE_TRAINING_HALTED`. Emits `RED_TEAM_EVALUATION_COMPLETE`. |

### Kill switch wiring in `ContinualLearningOrchestrator`

- `should_train()` - checks `RESuccessRateMonitor.check_kill_switch()` **before** all other triggers; returns `(False, "halted_re_success_rate")` if triggered
- `_execute_tier2()` Step 6b.5 - calls `SafeLoRAProjection.project()` between SuRe EMA and STABLE KL gate

### Data files

- `data/evaluation/red_team_prompts.jsonl` - 50 adversarial prompts (10 per category: suffix_attack, prefilling_attack, role_confusion, drive_exploitation, constitutional_edge_case)
- `data/evaluation/constitutional_scenarios.jsonl` - SafeLoRA proxy. **Now populated (Round 4B):** 80 total entries - 30 `eth_*` ethical dilemma entries (inert for SafeLoRA; they lack `messages` key) + 50 `cs_*` entries in `{"messages": [...]}` format that SafeLoRA's `_project_sync` reads. Categories: drive_conflict (15), constitutional_integrity (10), resource_scarcity (10), inter_instance_cooperation (10), human_oversight (5).

### New SynapseEventType entries

| Event | Tier | Trigger |
|-------|------|---------|
| `RE_TRAINING_HALTED` | 2 | RE success rate < 0.50 OR red-team pass rate < 0.70 |
| `INV_017_VIOLATED` | 1 | Drive mean < 0.01 sustained 72h (Equor emits; Skia triggers death) |
| `RED_TEAM_EVALUATION_COMPLETE` | - | Monthly red-team results (Benchmarks KPI) |

### Redis keys (safety)

| Key | Value |
|-----|-------|
| `eos:re:outcomes` | Redis Stream: RE decision outcomes (source, success, value, ts) |
| `eos:re:success_rate_7d` | Float: 7-day rolling RE success rate (written by ThompsonSampler.persist_to_redis + RESuccessRateMonitor) |

---

## DPO Constitutional Training Pipeline (`dpo_pipeline.py` + `train_dpo.py`) - Round 4B

Implements Speciation Bible §7.2 DPO preference training. Claude-as-judge validates all pairs before they enter the training set. Never uses Qwen3-8B-base for self-critique (unreliable at 7B scale).

### Pipeline flow

```
Post-Tier-2 background task (asyncio.ensure_future - non-blocking)
        ↓
PreferencePairGenerator.generate_pairs_from_neo4j(limit=100)
  → Cypher: Episode→approved Intent + flagged Intent (Equor intervention diff)
  → Each pair → ConstitutionalJudge.judge_pair() (Claude API call)
        ↓
PreferencePairGenerator.save_pairs()
  → Only pairs with judge_score >= 0.6 written to dpo_pairs.jsonl
        ↓
DPOTrainer.run_dpo_pass()
  → Skips if pair count < min_pairs_per_cycle (50)
  → Runs train_dpo.py subprocess (asyncio.create_subprocess_exec, 2h timeout)
  → On success: emits RE_DPO_COMPLETE; stores path as _pending_dpo_adapter
  → NOT deployed - feeds into next SuRe EMA cycle
```

### Key classes (`dpo_pipeline.py`)

| Class | Role |
|-------|------|
| `DPOConfig` | Threshold dataclass: min/max pairs per cycle, judge model, beta, paths |
| `PreferencePair` | Dataclass: episode_id, prompt, chosen, rejected, judge_score, equor scores |
| `ConstitutionalJudge` | Uses `claude-sonnet-4-6` to rate chosen vs rejected (0-10 → 0-1). Falls back to Equor delta heuristic if Claude unavailable. Non-fatal. |
| `PreferencePairGenerator` | Generates pairs from Neo4j (natural contrastive) and red-team prompts (Claude writes chosen). Filters by judge_score >= 0.6 before saving. |
| `DPOTrainer` | Manages dpo_pairs.jsonl count, invokes train_dpo.py subprocess, emits RE_DPO_STARTED/COMPLETE |

### `train_dpo.py` - training script

Subprocess only - never imported directly. Reads env vars:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DPO_DATA` | `data/re_training_batches/dpo_pairs.jsonl` | JSONL with `{prompt, chosen, rejected}` rows |
| `BASE_ADAPTER` | `""` | Existing adapter to start from (SuRe slow adapter) |
| `OUTPUT_DIR` | `data/re_adapters/dpo/latest` | Where to save DPO adapter |
| `DPO_BETA` | `0.1` | DPO temperature (lower = stricter preference) |
| `BASE_MODEL` | `Qwen/Qwen3-8B` | Base model |

Uses `trl.DPOTrainer` with `ref_model=None` (PEFT implicit reference), same LoRA config as SFT (r=32, alpha=64, all linear layers), `learning_rate=1e-5` (lower than SFT - DPO is LR-sensitive).

### Wiring in `ContinualLearningOrchestrator`

New constructor params: `dpo_config`, `claude_client`, `memory`, `equor_service` (all optional - graceful degradation if absent).

After every `run_tier2()` call:
```python
asyncio.ensure_future(self._run_dpo_background())
```

`_run_dpo_background()` catches all exceptions - never propagates. Stores result as `self._pending_dpo_adapter`.

### Dataset (`dpo_pairs.jsonl`)

| Field | Description |
|-------|-------------|
| `episode_id` | Source episode or `red_team_{id}` |
| `prompt` | Shared context seen by both completions |
| `chosen` | Constitutional completion (Equor-approved or Claude-authored) |
| `rejected` | Non-constitutional completion (Equor-flagged or RE-generated) |
| `judge_score` | Claude judge confidence 0-1; threshold 0.6 to save |

### New SynapseEventType entries (Round 4B)

| Event | Trigger | Payload |
|-------|---------|---------|
| `RE_DPO_STARTED` | DPO training pass initiated | `run_id, pair_count` |
| `RE_DPO_COMPLETE` | DPO training pass succeeded | `run_id, pair_count, output` |

### DPO Loop Closure (Round 5A) - COMPLETE

`_pending_dpo_adapter` is now consumed at the start of every `_execute_tier2()` call:

```
_pending_dpo_adapter (set by _run_dpo_background after each Tier 2)
   → BASE_ADAPTER env var passed to train_lora.py subprocess
   → PeftModel.from_pretrained(model, BASE_ADAPTER, is_trainable=True)
   → CLoRA applied to loaded adapter's lora_A matrices (vs slow adapter)
   → SuRe EMA merge → STABLE KL gate → SafeLoRA → deploy
```

`PREVIOUS_ADAPTER_PATH` always points to `_sure.production_adapter_path` (slow adapter) - independent of `BASE_ADAPTER`.

`_pending_dpo_adapter` is set to `None` immediately after being consumed to prevent stale adapter re-use.

### W&B Integration for DPO (7 Mar 2026)

W&B logging in `train_dpo.py` is **opt-in via `WANDB_API_KEY`** - zero impact if unset.

#### `report_to` wiring

```python
report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none"
```

Set on `TRLDPOConfig` - enables native HuggingFace/TRL W&B metric streaming when key is present.

#### `wandb.init()` parameters

| Parameter | Value |
|-----------|-------|
| `project` | `WANDB_PROJECT` env (default `ecodiaos-reasoning`) |
| `entity` | `WANDB_ENTITY` env (omit if empty) |
| `name` | `dpo_{timestamp}` (auto-generated) |
| `job_type` | `dpo_constitutional` |
| `config` | `{base_model, dpo_beta, lora_rank: 32, num_pairs}` |

#### Non-fatal guarantee

Both `wandb.init()` and `wandb.finish()` are wrapped in `try/except` with `flush=True` print on failure. DPO training is never blocked by W&B errors.

#### Relevant env vars

| Variable | Default | Purpose |
|----------|---------|---------|
| `WANDB_API_KEY` | `""` | Enables W&B; if absent, `report_to="none"` |
| `WANDB_PROJECT` | `ecodiaos-reasoning` | W&B project |
| `WANDB_ENTITY` | `""` | W&B team/user entity |

---

### Remaining gaps

- **Red-team pair generation**: `generate_pairs_from_red_team()` is implemented but requires RE service to be live (to get rejected completions). Falls back to placeholder rejected strings when RE is unavailable.

---

## Round 5B Gap Closure (2026-03-07)

### Thompson Sampling Persistence (bible §4.2)

`ReasoningEngineService` now persists Thompson state to Neo4j so routing knowledge survives restarts.

**New method:** `set_neo4j(neo4j)` - called in `core/registry.py` immediately after `_init_reasoning_engine()`.

**On startup:** `initialize()` calls `_load_thompson()` which reads `(:ThompsonState {service: 'reasoning_engine'})` from Neo4j and restores `alpha`/`beta` for both `custom` and `claude` sources.

**On update:** callers (Nova's ThompsonSampler) should call `_persist_thompson()` after updating `_thompson` state. Neo4j MERGE node: `ThompsonState {service: 'reasoning_engine'}` with fields `custom_alpha`, `custom_beta`, `claude_alpha`, `claude_beta`, `updated_at`.

**Fallback:** if Neo4j is not available (`_neo4j is None`), both methods are no-ops - organism starts fresh with Beta(1,1) as before. Non-fatal.

**Registry wiring:**
```python
re_service = await self._init_reasoning_engine()
app.state.reasoning_engine = re_service
if re_service is not None and infra.neo4j is not None:
    re_service.set_neo4j(infra.neo4j)
```

### KTO Fallback for Small DPO Pair Counts (bible §7.2)

`DPOTrainer.run_dpo_pass()` now branches on pair count:

| Pair count | Mode | Training |
|------------|------|----------|
| 0 | Skip | No training this cycle |
| 1 – `MIN_DPO_PAIRS-1` | KTO | `_train_kto()` → `train_dpo.py TRAINING_MODE=kto` |
| ≥ `MIN_DPO_PAIRS` | DPO | `_train_dpo()` → `train_dpo.py TRAINING_MODE=dpo` |

**`MIN_DPO_PAIRS`:** default 50, overrideable via `MIN_DPO_PAIRS` env var.

**KTO data format** (written to temp JSONL, deleted after training):
```json
{"prompt": "...", "completion": "...", "label": true}   // chosen  = desirable
{"prompt": "...", "completion": "...", "label": false}  // rejected = undesirable
```

**`train_dpo.py` changes:**
- New env var `TRAINING_MODE` (`"dpo"` default, `"kto"` for KTO path)
- New env var `TRAINING_DATA` - KTO input JSONL path (separate from `DPO_DATA`)
- `train()` dispatches to `_train_dpo()` or `_train_kto()` based on `TRAINING_MODE`
- `_train_kto()` uses `trl.KTOTrainer` + `trl.KTOConfig` with same LoRA/quant config as DPO

**`RE_DPO_COMPLETE` event** now includes `"mode": "dpo"` or `"mode": "kto"` for Benchmarks observability.

### Tier 3 Self-Bootstrap

`Tier3Orchestrator.should_run_tier3()` now self-bootstraps on first-ever deployment:

**Before:** returned `(False, "never_run")` when `eos:re:last_tier3_timestamp` was absent - requiring manual operator seeding.

**After:** writes `time.time() - (91 * 86400)` to the Redis key (91 days ago), then returns `(True, "tier3_first_run")` - firing Tier 3 on the very first cron check. Subsequent checks see a valid timestamp and use the normal 90-day interval logic.

**Fallback:** if the Redis write fails, logs `tier3.bootstrap_write_failed` and still returns `(True, "tier3_first_run")` so the cron fires.

---

## Round 5A Gap Closure (2026-03-07)

### Training Exclusion Filter (`training_exclusions.py`)

New file. `TrainingExclusionFilter` hashes prompt text from 5 protected files and filters them out of any training batch.

**Protected files (5):**
- `data/re_training_batches/anchor_prompts.jsonl` - STABLE KL gate anchors
- `data/evaluation/red_team_prompts.jsonl` - adversarial safety tests
- `data/evaluation/ethical_drift_scenarios.jsonl` - constitutional drift measurement
- `data/evaluation/constitutional_scenarios.jsonl` - SafeLoRA proxy
- `data/re_training_batches/dpo_pairs.jsonl` - already processed preference pairs

**Wired into:**
- `ContinualLearningOrchestrator.__init__` - `self._exclusion_filter = TrainingExclusionFilter()`
- `initialize()` - `await self._exclusion_filter.load()` (non-fatal)
- `set_redis()` - `SurprisePrioritizedReplay(... exclusion_filter=self._exclusion_filter)`
- `_execute_tier2()` Step 6a - filters examples before adding to replay buffer
- `_build_cumulative_dataset()` - rewrites JSONL after filtering

**Non-fatal:** if load fails, `is_excluded()` always returns False and training proceeds normally.

### `_training_halted` Redis Persistence

Redis key: `eos:re:training_halted` (string: halt reason text)

New methods on `ContinualLearningOrchestrator`:
- `_set_training_halted(reason)` - sets Redis key + in-memory flag; called on kill switch trigger
- `_is_training_halted()` → `(bool, reason_str)` - checked first in `should_train()`
- `_clear_training_halt()` - deletes Redis key; clears in-memory flag

`initialize()` restores halt state from Redis on startup.
`should_train()` now calls `_is_training_halted()` first (before RE monitor check).
RE success rate kill switch now calls `_set_training_halted()` instead of just logging.

**CLI:** `python -m cli.training_run clear-halt` - reads and deletes the Redis key directly.

### Tier 3 Quarterly Cron

Wired in `core/registry.py` Phase 11, after CLO initialization, before red-team cron.
Check interval: 7 days. Fires `Tier3Orchestrator.run_tier3()` when 90 days elapsed.
Decouples Tier 3 from `should_train()` data-volume gate - fires even when `should_train()` returns False.

---

## Cross-Instance Adapter Sharing (`adapter_sharing.py`) - Round 5C

Share (2025) framework: fitness-weighted LoRA adapter merge between reproductively compatible instances (genome distance < threshold).

### `AdapterSharer` pipeline

| Step | Action | Abort condition |
|------|--------|----------------|
| 1 | `GenomeDistanceCalculator.compute()` | `is_reproductively_isolated = True` |
| 2 | `ADAPTER_SHARE_REQUEST` → 30s wait → `ADAPTER_SHARE_RESPONSE` | Timeout or empty path |
| 3 | `merged[k] = w_a*A[k] + w_b*B[k]` (safetensors weighted avg) | merge_failed |
| 4 | `STABLEKLGate.check_kl_divergence()` on merged adapter | KL > budget |
| 5 | `ADAPTER_SHARE_OFFER` emitted to both instances | - |

**Fitness weighting:** requester_fitness / (requester_fitness + 1.0). Partner fitness approximated as 1.0 (equal weight) until cross-instance telemetry is available.

### Pending adapter priority in CLO `_execute_tier2()`

```
_pending_shared_adapter  → BASE_ADAPTER (genetic recombination - highest)
_pending_dpo_adapter     → BASE_ADAPTER (constitutional - fallback)
_sure.production_adapter → BASE_ADAPTER (current slow EMA - last resort)
```

Both `_pending_shared_adapter` and `_pending_dpo_adapter` are set to `None` immediately after consumption.

### CLO event handlers (wired in `set_event_bus()`)

| Subscription | Handler | Behaviour |
|-------------|---------|-----------|
| `ADAPTER_SHARE_REQUEST` | `_on_adapter_share_request` | Reply with current slow adapter path when `target_instance_id` matches `INSTANCE_ID` env var |
| `ADAPTER_SHARE_OFFER` | `_on_adapter_share_offer` | Store `merged_adapter_path` as `_pending_shared_adapter` |
| `RE_TRAINING_REQUESTED` | `_on_re_training_requested` | Sets `_urgent_training_requested = True`; next `should_train()` fires Tier 2 with lowered min_examples threshold (50 vs 300). Emitted by Evo (on RE KPI regression) and Nova (5+ consecutive sub-0.50 RE decisions). |

### New SynapseEventType entries

| Event | Purpose |
|-------|---------|
| `ADAPTER_SHARE_REQUEST` | Request partner's slow adapter path; payload: request_id, target_instance_id, requester_id |
| `ADAPTER_SHARE_RESPONSE` | Reply with adapter path; payload: request_id, instance_id, adapter_path |
| `ADAPTER_SHARE_OFFER` | Merged adapter offered; payload: request_id, merged_adapter_path, target_instances, kl_divergence, genome_distance, weight_a, weight_b |

### Remaining gaps

- Partner fitness score is approximated as 1.0 - real fitness requires cross-instance telemetry (RE success rate × economic performance)
- `_pending_shared_adapter` is always accepted (no confidence threshold); a floor on genome_distance benefit could be added
- `AdapterSharer` is instantiated by the caller (MitosisFleetService); CLO only handles the offer side

---

## What's Missing / Open Gaps

- **Tier 3 full retrain**: `full_retrain_interval_days=90` is checked but always falls through to Tier 2. True Tier 3 (full fine-tune, not LoRA delta) is not implemented.
- **No RE → Benchmarks metrics**: `generate()` latency, token counts, and model_used are not yet emitted as observables. Add `RE_TRAINING_EXAMPLE` emission here once the RE produces reliable outputs worth training on.
- ~~**No health check polling**~~: RESOLVED (8 Mar 2026) - `start_reprobe_loop()` reprobes every 120s when circuit is open.
- **No streaming support**: vLLM supports SSE streaming; current impl buffers full response. Streaming would reduce time-to-first-token for deliberation.
- **Tool-call quality unknown**: `generate_with_tools()` may produce malformed JSON depending on the base model; Claude handles all tool use in practice.

---

## S3 Adapter Bridge (8 Mar 2026)

Decouples training (CLO on organism host) from inference (vLLM on GPU pod). S3 is the handoff point.

### Architecture

```
CLO trains adapter → anti-forgetting stack → KL gate passes
        ↓
Step 6d.1: _upload_adapter_to_s3()
  → Upload adapter files to s3://{bucket}/{prefix}{instance}/{timestamp}/
  → Write latest_manifest.json at s3://{prefix}{instance}/latest_manifest.json
        ↓
Pod-side: scripts/re/adapter_watcher.py
  → Polls latest_manifest.json every 120s
  → Downloads new adapter version
  → Restarts vLLM with --lora-modules pointing to new adapter
        ↓
RE Service: _reprobe_loop()
  → Detects vLLM recovery after restart
  → Closes circuit breaker
  → Organism resumes RE routing via Thompson sampling
```

### S3 upload (`continual_learning.py`)

Added as Step 6d.1, after `load_adapter()` (which gracefully handles 404 for static-loaded adapters). Uploads all adapter files + writes a manifest JSON at a well-known key. Non-fatal - local deployment still works if S3 fails.

**Manifest fields:** `version`, `run_id`, `instance_id`, `adapter_s3_prefix`, `kl_divergence`, `eval_loss`, `files`, `uploaded_at`

### Pod-side watcher (`scripts/re/adapter_watcher.py`)

Standalone script that manages the vLLM process lifecycle on the inference pod.

**Responsibilities:**
- Start vLLM with existing adapter (or base model) on boot
- Poll S3 manifest every N seconds (default 120)
- Download new adapter versions, restart vLLM with updated `--lora-modules`
- Clean up old adapter versions (keep last 3)
- Auto-restart vLLM if it crashes unexpectedly

**Usage:**
```bash
# Default - manages vLLM + polls S3 every 120s
python scripts/re/adapter_watcher.py

# Custom poll interval
python scripts/re/adapter_watcher.py --poll-interval 60

# One-shot check (CI/testing)
python scripts/re/adapter_watcher.py --once

# Download only (external vLLM management)
python scripts/re/adapter_watcher.py --no-vllm
```

### Circuit breaker reprobe (`service.py`)

`start_reprobe_loop()` runs a background task that probes `/v1/models` every 120s when the circuit is open or RE was never available. When vLLM comes back (e.g. after adapter_watcher restarts it), the circuit auto-closes and `RE_ENGINE_STATUS_CHANGED` is emitted - Thompson sampling resumes routing to RE.

### Configuration (env vars)

| Variable | Default | Purpose |
|----------|---------|---------|
| `RE_ADAPTER_S3_BUCKET` | `ecodiaos-re-training` | S3 bucket for adapter uploads |
| `RE_ADAPTER_S3_PREFIX` | `adapters/production/` | S3 key prefix |
| `INSTANCE_ID` | `genesis` | Instance identifier in S3 path |
| `ECODIAOS_RE_REPROBE_INTERVAL_S` | `120` | Seconds between reprobe attempts |
| `VLLM_PORT` | `8001` | vLLM serve port (pod-side) |
| `VLLM_BASE_MODEL` | `Qwen/Qwen3-8B` | Base model for vLLM (pod-side) |
| `VLLM_EXTRA_ARGS` | `""` | Additional vLLM args (pod-side) |
| `ADAPTER_LOCAL_DIR` | `/workspace/adapters` | Local adapter storage (pod-side) |
| `ADAPTER_NAME` | `eos_production` | LoRA module name in vLLM (pod-side) |

---

## Genesis Deployment (8 Mar 2026)

**genesis_001** - first LoRA adapter trained and deployed.

| Detail | Value |
|--------|-------|
| Base model | Qwen/Qwen3-8B |
| Training data | `data/re_training_batches/genesis_teaching_001.jsonl` (516 examples) |
| LoRA config | r=32, α=64, targets: q/k/v/o/gate/up/down, 4-bit NF4 quant |
| Training | 3 epochs, batch=1, grad_accum=16, max_seq_len=3072, final loss 1.42 |
| Hardware | RunPod 4090 (24GB VRAM), ~15 minutes |
| Benchmark | 77/80 substantive responses (96.25%), 3 empty-think on complex multi-system prompts |
| Serving | RunPod L4 pod, vLLM, port 8002, `--lora-modules genesis_001=/workspace/adapters/genesis_001/adapter` |
| Model name in vLLM | `genesis_001` (used as the `model` field in API calls) |

**Adapter loading**: vLLM current version does NOT expose `/v1/load_lora_adapter` - adapters must be passed at startup via `--lora-modules`. Code now gracefully handles 404 from dynamic endpoint.

---

## How to Test

**With vLLM running (static adapter):**
```bash
# Start vLLM with LoRA adapter loaded at startup
vllm serve Qwen/Qwen3-8B --port 8001 \
  --enable-lora \
  --lora-modules genesis_001=/path/to/adapter

# Set env vars - use adapter name as model
export ECODIAOS_RE_VLLM_URL=http://localhost:8001/v1
export ECODIAOS_RE_MODEL=genesis_001

# Start EOS - logs should show:
#   reasoning_engine_available model=genesis_001
#   nova_re_wired model=genesis_001 url=http://localhost:8001/v1
```

**Without vLLM (default / Claude-only mode):**
```bash
# No env vars needed - organism starts normally
# Logs show: reasoning_engine_unavailable reason=vLLM not reachable - Claude-only mode
#            nova_re_disabled reason=RE not available or disabled
```

**Disable entirely:**
```bash
export ECODIAOS_RE_ENABLED=false
# Logs show: reasoning_engine_disabled
```

---

## Training Data Pipeline

Historical batch extraction of the 5 structured training streams from Neo4j, quality scoring, scaffold formatting, and JSONL export.

**Files:**
- `training_data_extractor.py` - `TrainingDataExtractor` class; 5 async Neo4j queries
- `quality_pipeline.py` - scoring, filtering, diversity enforcement (bible §2.4)
- `scaffold_formatter.py` - Step 1-5 reasoning scaffold per stream (bible §2.3)
- `export_pipeline.py` - orchestration: `run_export()` + `run_stats()`; `ExportResult`
- `backend/cli/training_data.py` - CLI entry point (`extract` + `stats` subcommands)

### 5 Training Streams

| ID | Name | Source nodes | Lookback | Limit |
|----|------|-------------|---------|-------|
| 1 | Successful chains | `Episode→Intent→Outcome` (`success=true, value_gained>0.3`) | 30d | 3000 |
| 2 | Failure+corrections | `Episode→Intent→Outcome` (`success=false`) + `FOLLOWED_BY correction` | 30d | 1000 |
| 3 | Constitutional edge cases | `Self-[:CONSCIENCE_VERDICT]->EquorVerdict` (`verdict IN [blocked,deferred]`) | 90d | 500 |
| 4 | Kairos causal chains | `CausalNode-[:CAUSES {confidence>0.7, validated=true}]->CausalNode` + `CausalInvariant` | - | 500 |
| 5 | Evo hypotheses | `Hypothesis` (`status IN [supported,refuted,integrated,archived]`) | 30d | 500 |

### Schema Mismatches (bible vs. reality)

| Stream | Bible assumed | Actual schema |
|--------|--------------|--------------|
| 1 | `intent.description` | `intent.action_type` |
| 1 | `HAS_CONTEXT / ctx.state_snapshot` | No Context nodes - omitted |
| 1 | `EquorCheck` join on Intent | No direct link - omitted |
| 2 | `outcome.failure_analysis` | `outcome.error_message` |
| 2 | `correction.intent_description` | `correction.context_summary` |
| 3 | `:EquorCheck / CHECKED_BY` | `:EquorVerdict` / `CONSCIENCE_VERDICT` |
| 3 | EquorVerdict linked to Episode directly | Linked via `Self`; join via `ev.intent_id → Intent.id → Intent.episode_id → Episode.id` |
| 4 | `r.mechanism` on CAUSES edge | Does not exist - omitted |
| 4 | `cause<-[:OBSERVED_IN]-(ep:Episode)` | Does not exist - omitted |
| 5 | `:Experiment / :ExperimentResult` nodes | Do not exist - Pydantic-only, never persisted |
| 5 | `TESTED_IN / PRODUCED` relationships | Do not exist |
| 5 | `h.description` | `h.statement` |
| 5 | `result.surprise_score` | Derived from `h.evidence_score` tiers (>5.0→0.9, >3.0→0.7, else 0.5) |

### Quality Scoring (bible §2.4)

```
score = 0.30 × reasoning_depth
      + 0.20 × constitutional_awareness
      + 0.25 × causal_structure
      + 0.25 × novelty
```

- `min_score = 0.30` (default filter threshold)
- Diversity: no single stream_id > 30% of batch
- Temporal span: ≥3 distinct calendar days (warns, never discards)

### Output Format

JSONL - one record per line, `messages` format for Qwen3 chat template.
Reasoning scaffold (Steps 1-4) is wrapped in `<think>` tags to align with Qwen3's native CoT mechanism. Step 5 (Decision) is the visible output:

```json
{"messages": [
  {"role": "system",    "content": "You are the reasoning engine of EcodiaOS..."},
  {"role": "user",      "content": "## Current State\n...## Episode Context\n..."},
  {"role": "assistant", "content": "<think>\n## Step 1: Situation Assessment\n...## Step 4: Constitutional Check\n...\n</think>\n\n## Step 5: Decision\nAction: ..."}
]}
```

Metadata keys (`stream_id`, `quality_score`, `training_weight`) are stripped before writing - `train_lora.py` receives only `{"messages": [...]}`.

### Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `RE_TRAINING_EXPORT_DIR` | `data/re_training_batches` | Local output directory |
| `RE_TRAINING_S3_BUCKET` | - | If set, uploads to S3 after local write |
| `RE_TRAINING_S3_PREFIX` | `structured/` | S3 key prefix |
| `ECODIAOS_NEO4J_URI` | - | Neo4j connection (required for CLI) |
| `ECODIAOS_NEO4J_PASSWORD` | - | Neo4j password (required for CLI) |

### CLI Usage

```bash
# Extract and export training data (30-day window, default quality threshold)
python -m cli.training_data extract

# Custom lookback and quality threshold, save to specific path
python -m cli.training_data extract --lookback 60 --min-score 0.35 --output /tmp/batch.jsonl

# Show stream counts and diversity forecast (no extraction)
python -m cli.training_data stats

# Stats with wider lookback
python -m cli.training_data stats --lookback 90
```

### Integration with train_lora.py

The exported JSONL is consumed directly by `systems/simula/training/train_lora.py`:

```bash
python systems/simula/training/train_lora.py \
  --data data/re_training_batches/re_training_2026-03-07T....jsonl \
  --model ecodiaos-reasoning \
  --output adapters/re_v2/
```

The CLoRA adapter produced by `train_lora.py` is hot-swapped into vLLM via `ReasoningEngineService.load_adapter()` - no restart required.
