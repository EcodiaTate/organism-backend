"""DPO fine-tuning script for constitutional alignment.

Runs as a subprocess invoked by DPOTrainer.run_dpo_pass() - same pattern
as train_lora.py. Do NOT import this module directly; run it via subprocess.

Reads from environment:
  DPO_DATA:    path to JSONL file with {"prompt", "chosen", "rejected"} rows
  BASE_ADAPTER: path to existing adapter to start from (optional)
  OUTPUT_DIR:  where to save the DPO adapter
  DPO_BETA:    DPO temperature (default 0.1)
  BASE_MODEL:  base model ID (default Qwen/Qwen3-8B)

Uses TRL's DPOTrainer with same LoRA config as train_lora.py:
  r=32, alpha=64, all linear projection layers, 4-bit NF4 quant
"""

import os
import sys
import time

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig as TRLDPOConfig
from trl import DPOTrainer

# ── Environment ────────────────────────────────────────────────────────────────

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-8B")
DPO_DATA = os.getenv("DPO_DATA", "data/re_training_batches/dpo_pairs.jsonl")
BASE_ADAPTER = os.getenv("BASE_ADAPTER", "")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data/re_adapters/dpo/latest")
DPO_BETA = float(os.getenv("DPO_BETA", "0.1"))
# TRAINING_MODE: "dpo" (default) or "kto"
# KTO mode reads TRAINING_DATA env var (JSONL with {prompt, completion, label: bool})
TRAINING_MODE = os.getenv("TRAINING_MODE", "dpo")
TRAINING_DATA = os.getenv("TRAINING_DATA", DPO_DATA)  # KTO data path

# ── LoRA config - identical to train_lora.py ──────────────────────────────────

LORA_CONFIG = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)


# ── Training ───────────────────────────────────────────────────────────────────


def _load_model_and_tokenizer() -> tuple:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if BASE_ADAPTER and os.path.exists(BASE_ADAPTER):
        print(f"[train_dpo] Loading base adapter from {BASE_ADAPTER}", flush=True)
        model = PeftModel.from_pretrained(model, BASE_ADAPTER, is_trainable=True)
    else:
        print("[train_dpo] No base adapter - applying fresh LoRA", flush=True)
        model = get_peft_model(model, LORA_CONFIG)

    model.print_trainable_parameters()
    return model, tokenizer


def train() -> None:
    print(f"[train_dpo] BASE_MODEL={BASE_MODEL}", flush=True)
    print(f"[train_dpo] TRAINING_MODE={TRAINING_MODE}", flush=True)
    print(f"[train_dpo] BASE_ADAPTER={BASE_ADAPTER!r}", flush=True)
    print(f"[train_dpo] OUTPUT_DIR={OUTPUT_DIR}", flush=True)

    if TRAINING_MODE == "kto":
        _train_kto()
    else:
        _train_dpo()


def _train_dpo() -> None:
    print(f"[train_dpo] DPO_DATA={DPO_DATA}", flush=True)
    print(f"[train_dpo] DPO_BETA={DPO_BETA}", flush=True)

    model, tokenizer = _load_model_and_tokenizer()

    # ── Dataset - expects {"prompt", "chosen", "rejected"} ────────────────────
    raw_dataset = load_dataset("json", data_files=DPO_DATA, split="train")

    required = {"prompt", "chosen", "rejected"}
    missing = required - set(raw_dataset.column_names)
    if missing:
        raise ValueError(f"DPO dataset missing required columns: {missing}")

    print(f"[train_dpo] Dataset: {len(raw_dataset)} preference pairs", flush=True)

    # ── DPO config ─────────────────────────────────────────────────────────────
    dpo_config = TRLDPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,          # Lower than SFT - DPO is sensitive to LR
        beta=DPO_BETA,
        max_length=2048,
        max_prompt_length=1024,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        optim="paged_adamw_8bit",
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,              # PEFT implicit reference (model before DPO updates)
        args=dpo_config,
        train_dataset=raw_dataset,
        processing_class=tokenizer,
    )

    _wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb as _wandb
            _wandb_run = _wandb.init(
                project=os.environ.get("WANDB_PROJECT", "ecodiaos-reasoning"),
                entity=os.environ.get("WANDB_ENTITY") or None,
                name=f"dpo_{int(time.time())}",
                job_type="dpo_constitutional",
                config={
                    "base_model": BASE_MODEL,
                    "dpo_beta": DPO_BETA,
                    "lora_rank": 32,
                    "num_pairs": len(raw_dataset),
                },
                resume="allow",
                settings=_wandb.Settings(silent=True),
            )
        except Exception as _e:
            print(f"[train_dpo] W&B init failed (non-fatal): {_e}", flush=True)

    print("[train_dpo] Starting DPO training...", flush=True)
    trainer.train()

    if _wandb_run is not None:
        try:
            _wandb_run.finish()
        except Exception as _e:
            print(f"[train_dpo] W&B finish failed (non-fatal): {_e}", flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[train_dpo] DPO complete. Adapter saved to {OUTPUT_DIR}", flush=True)


def _train_kto() -> None:
    """KTO (ICML 2024) training on unpaired preference data.

    Input JSONL format (from TRAINING_DATA env var):
        {"prompt": str, "completion": str, "label": bool}
    label=True  → desirable completion
    label=False → undesirable completion
    """
    from trl import KTOConfig, KTOTrainer

    print(f"[train_dpo] KTO_DATA={TRAINING_DATA}", flush=True)

    raw_dataset = load_dataset("json", data_files=TRAINING_DATA, split="train")

    required = {"prompt", "completion", "label"}
    missing = required - set(raw_dataset.column_names)
    if missing:
        raise ValueError(f"KTO dataset missing required columns: {missing}")

    print(f"[train_dpo] KTO dataset: {len(raw_dataset)} examples", flush=True)

    model, tokenizer = _load_model_and_tokenizer()

    kto_config = KTOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        max_length=2048,
        max_prompt_length=1024,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        optim="paged_adamw_8bit",
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = KTOTrainer(
        model=model,
        args=kto_config,
        train_dataset=raw_dataset,
        processing_class=tokenizer,
    )

    print("[train_dpo] Starting KTO training...", flush=True)
    trainer.train()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[train_dpo] KTO complete. Adapter saved to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    train()
