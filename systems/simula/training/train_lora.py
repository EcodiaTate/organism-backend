#!/usr/bin/env python3
"""
EcodiaOS — Standalone LoRA Fine-Tuning Script (Akash GPU Node)

This script runs inside the Akash GPU container. It:
  1. Downloads the JSONL training dataset from IPFS via Pinata gateway
  2. Loads the base model with Unsloth for 2x memory efficiency
  3. Applies LoRA adapters and trains on the dataset
  4. Saves the .safetensors adapter
  5. Uploads the adapter to IPFS via Pinata
  6. Exposes a /status HTTP endpoint for progress monitoring

Environment variables (injected by the SDL template):
  DATASET_CID          — IPFS CID of the training JSONL
  PINATA_JWT           — Pinata JWT for upload
  PINATA_GATEWAY_URL   — Pinata gateway for download
  BASE_MODEL           — HuggingFace model ID
  TRAINING_ARGS        — JSON-encoded hyperparameters
  STATUS_PORT          — Port for the status endpoint (default 8080)

This file is self-contained — no ecodiaos imports. It runs in isolation
on the Akash node with only ML dependencies installed.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── Configuration from environment ───────────────────────────────────────────

DATASET_CID = os.environ.get("DATASET_CID", "")
PINATA_JWT = os.environ.get("PINATA_JWT", "")
PINATA_GATEWAY_URL = os.environ.get("PINATA_GATEWAY_URL", "https://gateway.pinata.cloud")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-8B")
# TRAINING_DATA: local JSONL path, used by ContinualLearningOrchestrator (skips IPFS)
TRAINING_DATA = os.environ.get("TRAINING_DATA", "")
# OUTPUT_DIR: local adapter output path, used by ContinualLearningOrchestrator
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "")
# BASE_ADAPTER: if set, load from this existing adapter (e.g. DPO output) as the
# starting point for training instead of a fresh LoRA init. CLoRA is still applied
# to orthogonalize against PREVIOUS_ADAPTER_PATH directions if that is also set.
BASE_ADAPTER = os.environ.get("BASE_ADAPTER", "")
# PREVIOUS_ADAPTER_PATH: if set, new LoRA A matrices are initialized in the null space
# of the previous adapter's directions (CLoRA, ACL 2025). Prevents interference with
# previously learned features. Non-fatal — skipped if path missing or malformed.
# This is independent of BASE_ADAPTER — always points to the slow (EMA) adapter.
PREVIOUS_ADAPTER_PATH = os.environ.get("PREVIOUS_ADAPTER_PATH", "")
TRAINING_ARGS = json.loads(os.environ.get("TRAINING_ARGS", "{}"))
STATUS_PORT = int(os.environ.get("STATUS_PORT", "8080"))

# ── Global training state (exposed via /status) ─────────────────────────────

TRAINING_STATE: dict[str, object] = {
    "phase": "initializing",
    "progress": 0.0,
    "current_epoch": 0,
    "total_epochs": TRAINING_ARGS.get("num_epochs", 3),
    "current_loss": 0.0,
    "adapter_cid": "",
    "error": "",
    "started_at": time.time(),
    "completed_at": 0.0,
}


# ── Status HTTP Server ──────────────────────────────────────────────────────


class StatusHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that returns training status as JSON."""

    def do_GET(self) -> None:
        if self.path == "/status" or self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(TRAINING_STATE).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        # Suppress default access logging
        pass


def start_status_server() -> HTTPServer:
    """Start the status server in a background thread."""
    server = HTTPServer(("0.0.0.0", STATUS_PORT), StatusHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[EOS] Status server listening on port {STATUS_PORT}")
    return server


# ── IPFS Download / Upload ──────────────────────────────────────────────────


def download_dataset(cid: str, dest_path: Path) -> None:
    """Download training dataset from IPFS gateway."""
    import httpx

    TRAINING_STATE["phase"] = "downloading_dataset"
    url = f"{PINATA_GATEWAY_URL.rstrip('/')}/ipfs/{cid}"
    print(f"[EOS] Downloading dataset from {url}")

    with httpx.Client(timeout=300.0) as client:
        resp = client.get(url)
        resp.raise_for_status()
        dest_path.write_bytes(resp.content)

    size_mb = dest_path.stat().st_size / (1024 * 1024)
    print(f"[EOS] Dataset downloaded: {size_mb:.1f} MB")


def upload_adapter(adapter_dir: Path) -> str:
    """
    Upload the LoRA adapter directory to IPFS via Pinata.

    Returns the IPFS CID.
    """
    import io
    import tarfile

    import httpx

    TRAINING_STATE["phase"] = "uploading_adapter"
    print("[EOS] Packaging adapter for upload...")

    # Tar the adapter directory (contains adapter_model.safetensors + config)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for fpath in adapter_dir.rglob("*"):
            if fpath.is_file():
                tar.add(str(fpath), arcname=fpath.relative_to(adapter_dir))
    adapter_bytes = buf.getvalue()

    print(f"[EOS] Adapter package: {len(adapter_bytes) / (1024 * 1024):.1f} MB")

    with httpx.Client(timeout=600.0) as client:
        resp = client.post(
            "https://api.pinata.cloud/pinning/pinFileToIPFS",
            headers={"Authorization": f"Bearer {PINATA_JWT}"},
            files={"file": ("ecodiaos-lora-adapter.tar.gz", adapter_bytes, "application/gzip")},
            data={"pinataMetadata": json.dumps({"name": f"ecodiaos-lora-{int(time.time())}"})},
        )
        resp.raise_for_status()
        cid = resp.json()["IpfsHash"]

    print(f"[EOS] Adapter uploaded to IPFS: {cid}")
    return cid


# ── Training Pipeline ────────────────────────────────────────────────────────


def load_dataset(dataset_path: Path) -> list[dict[str, str]]:
    """Load JSONL dataset into memory."""
    records: list[dict[str, str]] = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[EOS] Loaded {len(records)} training examples")
    return records


def run_training(dataset_path: Path) -> Path:
    """
    Execute the LoRA fine-tuning loop using Unsloth.

    Returns the path to the saved adapter directory.
    """
    import torch
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel

    TRAINING_STATE["phase"] = "loading_model"
    print(f"[EOS] Loading base model: {BASE_MODEL}")

    # Hyperparameters with defaults (bible §5 — r=32, lora_alpha=64, 2:1 ratio)
    lora_rank = TRAINING_ARGS.get("lora_rank", 32)
    lora_alpha = TRAINING_ARGS.get("lora_alpha", 64)
    lora_dropout = TRAINING_ARGS.get("lora_dropout", 0.05)
    lr = TRAINING_ARGS.get("learning_rate", 2e-4)
    num_epochs = TRAINING_ARGS.get("num_epochs", 3)
    batch_size = TRAINING_ARGS.get("batch_size", 4)
    grad_accum = TRAINING_ARGS.get("gradient_accumulation_steps", 4)
    max_seq_len = TRAINING_ARGS.get("max_seq_length", 4096)
    warmup_ratio = TRAINING_ARGS.get("warmup_ratio", 0.03)
    weight_decay = TRAINING_ARGS.get("weight_decay", 0.01)

    # Load model with Unsloth (2x faster, 60% less memory)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=max_seq_len,
        dtype=None,  # auto-detect (float16 on most GPUs)
        load_in_4bit=True,
    )

    # Qwen3 uses <|im_end|> as EOS — patch tokenizer before Unsloth PEFT setup
    # so get_peft_model(use_gradient_checkpointing="unsloth") doesn't try <EOS_TOKEN>
    qwen3_eos = "<|im_end|>"
    if qwen3_eos in tokenizer.get_vocab():
        tokenizer.eos_token = qwen3_eos
    if tokenizer.pad_token is None or tokenizer.pad_token not in tokenizer.get_vocab():
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA adapters — either load from an existing adapter (BASE_ADAPTER, e.g. DPO
    # output) or initialize fresh LoRA weights. CLoRA orthogonalization is applied in
    # both cases if PREVIOUS_ADAPTER_PATH is set (always the slow EMA adapter).
    if BASE_ADAPTER and os.path.exists(BASE_ADAPTER):
        # Load from DPO-tuned adapter as training starting point
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, BASE_ADAPTER, is_trainable=True)
        print(f"[EOS] Loaded DPO base adapter from: {BASE_ADAPTER}")
        # Apply CLoRA to the loaded adapter's lora_A matrices so new directions
        # are orthogonal to the slow adapter's accumulated history.
        if PREVIOUS_ADAPTER_PATH and os.path.exists(PREVIOUS_ADAPTER_PATH):
            _apply_clora_init(model, PREVIOUS_ADAPTER_PATH)
            print(f"[EOS] CLoRA orthogonalization applied (on DPO base) from: {PREVIOUS_ADAPTER_PATH}")
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing=True,
        )
        # CLoRA (ACL 2025): initialize new LoRA A matrices in the null space of
        # the previous adapter directions — prevents interference with learned features.
        if PREVIOUS_ADAPTER_PATH and os.path.exists(PREVIOUS_ADAPTER_PATH):
            _apply_clora_init(model, PREVIOUS_ADAPTER_PATH)
            print(f"[EOS] CLoRA init applied from: {PREVIOUS_ADAPTER_PATH}")

    TRAINING_STATE["phase"] = "preparing_data"
    print("[EOS] Preparing training data...")

    # Load and format dataset
    raw_data = load_dataset(dataset_path)

    # Format as chat templates for instruction tuning
    formatted: list[dict[str, str]] = []
    for row in raw_data:
        if "messages" in row:
            # Chat format — apply template directly
            text = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False,
            )
        elif "instruction" in row:
            # Build rich user context — supports both legacy {input} and RE training
            # schema {input_context, output_action, reasoning_trace, outcome, ...}
            user_parts = [row["instruction"]]
            ctx = row.get("input_context") or row.get("input", "")
            if ctx:
                user_parts.append(f"Context: {ctx}")
            if row.get("source_system"):
                user_parts.append(f"System: {row['source_system']} | Type: {row.get('example_type', '')}")
            if row.get("reasoning_trace"):
                user_parts.append(f"Reasoning: {row['reasoning_trace']}")
            aligns = row.get("constitutional_alignment") or {}
            if any(v for v in aligns.values() if v):
                align_str = ", ".join(f"{k}={v:.2f}" for k, v in aligns.items() if v)
                user_parts.append(f"Constitutional alignment: {align_str}")

            # Build assistant response — use output_action + outcome if present
            output = row.get("output_action") or row.get("output", "")
            if row.get("outcome"):
                output = f"{output}\nOutcome: {row['outcome']}"
            if row.get("alternatives_considered"):
                alts = row["alternatives_considered"]
                if alts:
                    output += f"\nAlternatives considered: {'; '.join(str(a) for a in alts)}"

            if not output:
                continue  # skip examples with no target output

            messages = [
                {"role": "system", "content": "You are EcodiaOS, a self-evolving digital organism. Reason carefully and act constitutionally."},
                {"role": "user", "content": "\n".join(user_parts)},
                {"role": "assistant", "content": output},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        else:
            continue
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)
    print(f"[EOS] Formatted {len(formatted)} examples for training")

    # Training arguments — prefer OUTPUT_DIR env var for local orchestration
    output_dir = OUTPUT_DIR if OUTPUT_DIR else tempfile.mkdtemp(prefix="eos-finetune-")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=5,
        save_strategy="epoch",
        optim="adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        seed=42,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        max_length=max_seq_len,
        dataset_kwargs={"skip_prepare_dataset": True},
        eos_token=None,
    )

    # Custom callback for progress tracking
    class ProgressCallback:
        def on_log(self, args: object, state: object, control: object, logs: dict[str, float] | None = None, **kwargs: object) -> None:
            if logs:
                TRAINING_STATE["current_loss"] = logs.get("loss", 0.0)
            if hasattr(state, "epoch"):
                TRAINING_STATE["current_epoch"] = int(getattr(state, "epoch", 0))
                total = float(TRAINING_STATE["total_epochs"])  # type: ignore[arg-type]
                if total > 0:
                    TRAINING_STATE["progress"] = float(getattr(state, "epoch", 0)) / total

    TRAINING_STATE["phase"] = "training"
    print("[EOS] Starting LoRA fine-tuning...")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        callbacks=[ProgressCallback()],
    )

    # W&B run — guarded; non-fatal if W&B not available or not authenticated
    _wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb as _wandb
            _wandb_run = _wandb.init(
                project=os.environ.get("WANDB_PROJECT", "ecodiaos-reasoning"),
                entity=os.environ.get("WANDB_ENTITY") or None,
                name=os.environ.get("WANDB_RUN_NAME", f"tier2_{int(time.time())}"),
                job_type=os.environ.get("WANDB_JOB_TYPE", "tier2_sft"),
                config={
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
                },
                resume="allow",
                settings=_wandb.Settings(silent=True),
            )
        except Exception as _wandb_exc:
            print(f"[EOS] W&B init failed (non-fatal): {_wandb_exc}")

    trainer.train()

    # Log final metrics and close W&B run
    if _wandb_run is not None:
        try:
            if trainer.state.log_history:
                last = trainer.state.log_history[-1]
                _wandb_run.log({
                    "final_train_loss": last.get("train_loss", last.get("loss", 0.0)),
                    "epochs_completed": num_epochs,
                })
            _wandb_run.finish()
        except Exception as _wandb_exc:
            print(f"[EOS] W&B finish failed (non-fatal): {_wandb_exc}")

    TRAINING_STATE["phase"] = "saving_adapter"
    print("[EOS] Saving LoRA adapter...")

    # Save only the LoRA adapter (not the full model)
    adapter_dir = Path(output_dir) / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    adapter_size = sum(f.stat().st_size for f in adapter_dir.rglob("*") if f.is_file())
    print(f"[EOS] Adapter saved: {adapter_size / (1024 * 1024):.1f} MB")

    # Record final metrics
    if trainer.state.log_history:
        last_log = trainer.state.log_history[-1]
        TRAINING_STATE["current_loss"] = last_log.get("train_loss", last_log.get("loss", 0.0))

    return adapter_dir


# ── Helpers ───────────────────────────────────────────────────────────────────


def _write_status_json(out_dir: str) -> None:
    """
    Write TRAINING_STATE to status.json in out_dir.
    Called after training completes so ContinualLearningOrchestrator can
    read eval_loss without parsing stdout.
    """
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "status.json").write_text(
            json.dumps(TRAINING_STATE), encoding="utf-8"
        )
    except Exception:
        pass


# ── CLoRA Orthogonal Subspace Init (ACL 2025) ────────────────────────────────


def _apply_clora_init(model: object, previous_adapter_path: str) -> None:
    """CLoRA (ACL 2025): Initialize new LoRA A matrices in the null space of
    previous adapter directions. This prevents interference with previously
    learned features.

    Algorithm:
    1. Load previous adapter weights from safetensors
    2. For each lora_A weight: compute null space via SVD
       (Vh rows span the row space of A_prev)
    3. Null space projector: P = I - Vh.T @ Vh
    4. Re-initialize new model's lora_A with a random matrix projected
       onto that null space: param = randn @ P

    Non-fatal: if safetensors unavailable, path missing, or any per-layer
    error occurs, logs the issue and skips that layer.
    """
    import torch

    try:
        from safetensors.torch import load_file
        prev_weights = load_file(f"{previous_adapter_path}/adapter_model.safetensors")
    except ImportError:
        print("[CLoRA] safetensors not installed — skipping CLoRA init.")
        return
    except Exception as e:
        print(f"[CLoRA] Could not load previous adapter: {e} — skipping CLoRA init.")
        return

    applied = 0
    skipped = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_A" not in name or not param.requires_grad:
                continue
            # Match by layer name (strip "base_model.model." prefix if needed)
            prev_key = name.replace("base_model.model.", "")
            if prev_key not in prev_weights:
                skipped += 1
                continue
            try:
                A_prev = prev_weights[prev_key].to(param.device).float()
                # SVD: A_prev = U @ diag(S) @ Vh; Vh rows span the row space of A_prev
                _, _, Vh = torch.linalg.svd(A_prev, full_matrices=False)
                # Null space projector: P = I - Vh.T @ Vh  (shape: [d_in, d_in])
                d_in = param.shape[1]
                P_null = torch.eye(d_in, device=param.device) - Vh.T @ Vh
                # Project random init onto null space — result lives in orthogonal complement
                random_init = torch.randn_like(param.float())
                param.data = (random_init @ P_null).to(param.dtype)
                applied += 1
            except Exception as e:
                print(f"[CLoRA] Skipping layer {name}: {e}")
                skipped += 1

    print(f"[CLoRA] Orthogonal init applied to {applied} layers, skipped {skipped}.")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the training container."""
    start_status_server()

    try:
        # Local training mode (ContinualLearningOrchestrator): use TRAINING_DATA directly
        if TRAINING_DATA:
            dataset_path = Path(TRAINING_DATA)
            if not dataset_path.exists():
                raise ValueError(f"TRAINING_DATA file not found: {TRAINING_DATA}")
            print(f"[EOS] Local training mode — dataset: {TRAINING_DATA}")
        else:
            # Akash/IPFS mode: download from pinned CID
            if not DATASET_CID:
                raise ValueError("DATASET_CID environment variable is required")
            if not PINATA_JWT:
                raise ValueError("PINATA_JWT environment variable is required")
            dataset_path = Path(tempfile.mkdtemp()) / "training_data.jsonl"
            download_dataset(DATASET_CID, dataset_path)

        # Step 2: Run training
        adapter_dir = run_training(dataset_path)

        # Step 3: Upload adapter to IPFS (skipped in local training mode)
        if TRAINING_DATA:
            adapter_cid = ""  # local mode — no IPFS upload needed
        else:
            adapter_cid = upload_adapter(adapter_dir)

        # Step 4: Mark completion
        TRAINING_STATE["phase"] = "completed"
        TRAINING_STATE["progress"] = 1.0
        TRAINING_STATE["adapter_cid"] = adapter_cid if not TRAINING_DATA else ""
        TRAINING_STATE["completed_at"] = time.time()
        _write_status_json(OUTPUT_DIR or str(adapter_dir.parent))
        print(f"[EOS] Fine-tuning complete. Adapter CID: {adapter_cid if not TRAINING_DATA else '(local)'}")

        # Keep the status server alive for the executor to poll
        signal.pause()

    except Exception as exc:
        TRAINING_STATE["phase"] = "failed"
        TRAINING_STATE["error"] = str(exc)
        print(f"[EOS] FATAL: {exc}", file=sys.stderr)

        # Keep alive so the executor can read the error
        try:
            signal.pause()
        except AttributeError:
            # signal.pause() not available on Windows — use sleep fallback
            while True:
                time.sleep(3600)


if __name__ == "__main__":
    main()
