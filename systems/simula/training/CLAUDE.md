# Simula Training - CLAUDE.md

**File:** `systems/simula/training/train_lora.py`
**Role:** Standalone LoRA fine-tuning script. Runs as an asyncio subprocess launched by `ContinualLearningOrchestrator`. Self-contained - no ecodiaos imports.

---

## What It Does

1. Downloads JSONL training dataset (from IPFS Pinata gateway, or local path via `TRAINING_DATA`)
2. Loads `Qwen/Qwen3-8B` base model with Unsloth (2× faster, 60% less VRAM)
3. Applies LoRA adapters (r=32, α=64, target: q/k/v/o/gate/up/down projections)
4. **CLoRA init** (Round 4A): re-initializes LoRA A matrices in null space of previous adapter (if `PREVIOUS_ADAPTER_PATH` set)
5. Trains with SFTTrainer
6. Saves adapter to `{OUTPUT_DIR}/adapter/` (safetensors)
7. Uploads to IPFS (Akash/cloud mode) or skips upload (local mode)
8. Writes `{OUTPUT_DIR}/status.json` for orchestrator to read `eval_loss`

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `BASE_MODEL` | `Qwen/Qwen3-8B` | HuggingFace model ID |
| `TRAINING_DATA` | `""` | Local JSONL path; if set, skips IPFS download |
| `OUTPUT_DIR` | `""` | Local adapter output directory; if set, skips IPFS upload |
| `BASE_ADAPTER` | `""` | **DPO starting point**: path to an existing adapter (e.g. DPO output) to load as the starting weights instead of a fresh LoRA init. When set, `PeftModel.from_pretrained()` is used. CLoRA is still applied if `PREVIOUS_ADAPTER_PATH` is also set. |
| `PREVIOUS_ADAPTER_PATH` | `""` | **CLoRA**: path to slow (EMA) adapter directory. New/loaded LoRA A matrices are re-initialized in the orthogonal complement of this adapter's directions. Independent of `BASE_ADAPTER`. Empty = no CLoRA. |
| `TRAINING_ARGS` | `"{}"` | JSON-encoded hyperparameter overrides (lora_rank, lora_alpha, learning_rate, num_epochs, batch_size, etc.) |
| `DATASET_CID` | `""` | IPFS CID for Akash/cloud mode |
| `PINATA_JWT` | `""` | Pinata JWT for IPFS upload |
| `PINATA_GATEWAY_URL` | `https://gateway.pinata.cloud` | Pinata gateway URL |
| `STATUS_PORT` | `8080` | HTTP port for `/status` and `/health` endpoints |

---

## BASE_ADAPTER + CLoRA - DPO Loop Closure (Round 5A)

`BASE_ADAPTER` and `PREVIOUS_ADAPTER_PATH` are independent env vars:

| Variable | Role | Set by CLO to |
|---|---|---|
| `BASE_ADAPTER` | Starting weights for training | `_pending_dpo_adapter` (if set), else `production_adapter_path` |
| `PREVIOUS_ADAPTER_PATH` | CLoRA orthogonalization target | Always `_sure.production_adapter_path` (slow adapter) |

**When `BASE_ADAPTER` is set:**
1. `PeftModel.from_pretrained(model, BASE_ADAPTER, is_trainable=True)` - loads DPO-tuned starting point
2. If `PREVIOUS_ADAPTER_PATH` also set: `_apply_clora_init()` modifies the loaded model's `lora_A` tensors in-place to orthogonalize against the slow adapter's directions

**When `BASE_ADAPTER` is NOT set:**
1. Fresh `get_peft_model()` + standard random init
2. If `PREVIOUS_ADAPTER_PATH` set: `_apply_clora_init()` applied as before

**Tier 3:** Both env vars are explicitly empty - clean base model start, no CLoRA.

---

## CLoRA Orthogonal Subspace Init (ACL 2025)

**Function:** `_apply_clora_init(model, previous_adapter_path)`

Called automatically after `get_peft_model()` if `PREVIOUS_ADAPTER_PATH` is set and the path exists.

### Algorithm

1. Load `{PREVIOUS_ADAPTER_PATH}/adapter_model.safetensors`
2. For each `lora_A` parameter in the new model:
   - Match by layer name (strips `base_model.model.` prefix)
   - `_, _, Vh = torch.linalg.svd(A_prev, full_matrices=False)`
   - Null space projector: `P_null = I - Vh.T @ Vh` (shape: `[d_in, d_in]`)
   - Re-initialize: `param.data = (randn_like(param) @ P_null).to(dtype)`
3. Result: new LoRA A matrices are orthogonal to the row space of the previous adapter's A matrices - no interference with previously learned feature directions.

### Non-Fatal Conditions

CLoRA init is always non-fatal. It is skipped (with a print log) when:
- `PREVIOUS_ADAPTER_PATH` is empty or does not exist → standard random init
- `safetensors` package not installed → standard random init
- `adapter_model.safetensors` missing at the path → standard random init
- Per-layer SVD fails → that layer gets standard random init, others get CLoRA init

### When CLoRA Is NOT Applied

- **Tier 3 quarterly retrain**: `PREVIOUS_ADAPTER_PATH=""` is explicitly set - clean base model start
- **First Tier 2 run**: `self._current_adapter_path` is None → empty string → no previous adapter

---

## LoRA Hyperparameters

Default (set by `ContinualLearningOrchestrator._get_training_config()`):

| Dataset size | r | α | lr | epochs | batch |
|---|---|---|---|---|---|
| <500 examples | 32 | 64 | 3e-4 | 2 | 4 |
| 500–2000 | 32 | 64 | 2e-4 | 3 | 4 |
| >2000 | 32 | 64 | 1e-4 | 4 | 8 |

α:r ratio is always 2:1 (Speciation Bible §5).

---

## Target Modules

```python
["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

All attention + MLP projection layers - standard Qwen3 LoRA target set.

---

## Output Layout

```
{OUTPUT_DIR}/
  adapter/
    adapter_model.safetensors   # LoRA weights (fast adapter)
    adapter_config.json
    tokenizer_config.json
    ...
  status.json                   # TRAINING_STATE dict (eval_loss, phase, progress)
```

---

## Modes

### Local Mode (ContinualLearningOrchestrator)
Set by orchestrator when running Tier 2 or Tier 3:
```bash
TRAINING_DATA=/path/to/data.jsonl
OUTPUT_DIR=/path/to/adapter_output/
PREVIOUS_ADAPTER_PATH=/path/to/previous_adapter/   # Tier 2 only
```
No IPFS download or upload. Orchestrator polls `status.json` for result.

### Akash/Cloud Mode
Set when deploying on Akash GPU node:
```bash
DATASET_CID=QmXxx...
PINATA_JWT=eyJ...
```
Downloads from IPFS, uploads adapter back to IPFS. Status server on `STATUS_PORT`.

---

## Integration with ContinualLearningOrchestrator

The orchestrator (`systems/reasoning_engine/continual_learning.py`) launches this script as a subprocess:

```python
proc = await asyncio.create_subprocess_exec(
    sys.executable,
    _TRAIN_SCRIPT,
    env={
        **os.environ,
        "BASE_MODEL": ...,
        "TRAINING_DATA": jsonl_path,
        "OUTPUT_DIR": adapter_output_dir,
        "TRAINING_ARGS": json.dumps(training_args),
        "PREVIOUS_ADAPTER_PATH": self._current_adapter_path or "",  # CLoRA
    },
)
```

Timeout: 2h for Tier 2, 4h for Tier 3 (enforced by `asyncio.wait_for`).

On success, the orchestrator reads `status.json`, runs the SuRe EMA merge, SafeLoRA projection, STABLE KL gate, and deploys the slow adapter via vLLM.

---

## W&B Integration (7 Mar 2026)

W&B logging is **opt-in via env var** - zero impact if `WANDB_API_KEY` is unset.

### Activation

Set `WANDB_API_KEY` in the environment (or `.env`). All other vars have sensible defaults.

### Parameters

| Variable | Default | Purpose |
|----------|---------|---------|
| `WANDB_API_KEY` | `""` | If set, enables W&B logging; if absent, `report_to="none"` |
| `WANDB_PROJECT` | `ecodiaos-reasoning` | W&B project name |
| `WANDB_ENTITY` | `""` | W&B team/user entity (omit for personal account) |
| `WANDB_RUN_NAME` | `tier2_{timestamp}` | Run display name; overridden to `tier3_{run_id}` for Tier 3 runs (injected by `Tier3Orchestrator`) |
| `WANDB_JOB_TYPE` | `tier2_sft` | Job type tag; `tier3_full_retrain` for Tier 3 |

### Logged config

```python
{
    "base_model": BASE_MODEL,
    "lora_rank": lora_rank,
    "lora_alpha": lora_alpha,
    "lora_dropout": lora_dropout,
    "learning_rate": lr,
    "num_epochs": num_epochs,
    "batch_size": batch_size,
    "grad_accum": grad_accum,
    "max_seq_len": max_seq_len,
    "clora_applied": bool(PREVIOUS_ADAPTER_PATH),
    "base_adapter_applied": bool(BASE_ADAPTER),
}
```

### Logged metrics (post-training)

```python
{"final_train_loss": ..., "epochs_completed": num_epochs}
```

### Non-fatal guarantee

`wandb.init()` and `wandb.finish()` are both wrapped in `try/except` - failure is logged as `[EOS] W&B init/finish failed (non-fatal): ...` and training continues normally.

### Tier 3 override

`Tier3Orchestrator.run_tier3()` injects `WANDB_RUN_NAME=tier3_{run_id}` and `WANDB_JOB_TYPE=tier3_full_retrain` into the subprocess env, overriding the defaults above.

---

## Qwen3 `<think>` Tag Training Format (8 Mar 2026)

The instruction-format branch now wraps `reasoning_trace` in Qwen3-native `<think>` tags:

```
<think>
{reasoning_trace content}
</think>

{output_action content}
```

This teaches the model to use its native chain-of-thought mechanism instead of outputting reasoning as visible prose. The `messages` format path is unchanged - if your JSONL already uses `{"messages": [...]}` format, the assistant content should include `<think>` tags directly.

**Before (genesis_001):** reasoning was injected into the user prompt as `Reasoning: {trace}`, and the assistant output was just `output_action`. The model learned to output reasoning as visible text, not inside think tags.

**After:** reasoning moves to the assistant side inside `<think>` tags. The model learns to think internally before responding - matching Qwen3's pretrained behaviour.

---

## Known Issues / Remaining Work

- `signal.pause()` at the end has a Windows fallback (`while True: sleep(3600)`) - this is only needed in Akash mode to keep the status server alive after training.
- The status server (`/status`, `/health`) is not used in local orchestrator mode - it's purely for Akash job monitoring.
