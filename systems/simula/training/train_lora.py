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
BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/Meta-Llama-3.1-8B-Instruct")
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
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    TRAINING_STATE["phase"] = "loading_model"
    print(f"[EOS] Loading base model: {BASE_MODEL}")

    # Hyperparameters with defaults
    lora_rank = TRAINING_ARGS.get("lora_rank", 64)
    lora_alpha = TRAINING_ARGS.get("lora_alpha", 128)
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

    # Apply LoRA adapters
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
        use_gradient_checkpointing="unsloth",
    )

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
            messages = [
                {"role": "system", "content": "You are EcodiaOS, a self-evolving digital organism."},
                {"role": "user", "content": row["instruction"] + ("\n" + row.get("input", "")).rstrip()},
                {"role": "assistant", "content": row.get("output", "")},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        else:
            continue
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)
    print(f"[EOS] Formatted {len(formatted)} examples for training")

    # Training arguments
    output_dir = tempfile.mkdtemp(prefix="eos-finetune-")
    training_args = TrainingArguments(
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
        report_to="none",
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
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        args=training_args,
        callbacks=[ProgressCallback()],
    )

    trainer.train()

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


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for the training container."""
    start_status_server()

    try:
        # Validate environment
        if not DATASET_CID:
            raise ValueError("DATASET_CID environment variable is required")
        if not PINATA_JWT:
            raise ValueError("PINATA_JWT environment variable is required")

        # Step 1: Download dataset
        dataset_path = Path(tempfile.mkdtemp()) / "training_data.jsonl"
        download_dataset(DATASET_CID, dataset_path)

        # Step 2: Run training
        adapter_dir = run_training(dataset_path)

        # Step 3: Upload adapter to IPFS
        adapter_cid = upload_adapter(adapter_dir)

        # Step 4: Mark completion
        TRAINING_STATE["phase"] = "completed"
        TRAINING_STATE["progress"] = 1.0
        TRAINING_STATE["adapter_cid"] = adapter_cid
        TRAINING_STATE["completed_at"] = time.time()
        print(f"[EOS] Fine-tuning complete. Adapter CID: {adapter_cid}")

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
