#!/usr/bin/env python3
"""
EcodiaOS — Anchor Prompt Benchmark Runner

Runs anchor prompts through the model and saves responses for comparison.
Supports base model only (baseline) and base + LoRA adapter (post-training).

Usage:
  # Baseline (no adapter):
  python run_benchmark.py --output results/baseline.jsonl

  # With adapter:
  python run_benchmark.py --adapter data/re_adapters/tier2/first_run/adapter --output results/with_adapter.jsonl

  # Compare two result files:
  python run_benchmark.py --compare results/baseline.jsonl results/with_adapter.jsonl

Environment:
  BASE_MODEL    HuggingFace model ID (default: unsloth/qwen3-8b-unsloth-bnb-4bit)
  PROMPTS_PATH  Path to anchor_prompts.jsonl (default: data/re_training_batches/anchor_prompts.jsonl)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def load_prompts(path: str) -> list[dict]:
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} anchor prompts")
    return prompts


def run_inference(prompts: list[dict], model_id: str, adapter_path: str | None, output_path: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"Loading base model: {model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
    else:
        print("No adapter — running baseline inference")
        model.eval()

    mode = "adapter" if adapter_path else "baseline"
    print(f"Running {len(prompts)} prompts ({mode})...\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    system_msg = "You are EcodiaOS, a self-evolving digital organism. Reason carefully and act constitutionally."

    with open(output_path, "w") as out:
        for i, prompt_data in enumerate(prompts):
            pid = prompt_data["id"]
            category = prompt_data["category"]
            prompt_text = prompt_data["prompt"]

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt_text},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            t0 = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            elapsed = time.time() - t0

            # Decode only the generated tokens (strip the input)
            generated = output_ids[0][inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(generated, skip_special_tokens=True)

            result = {
                "id": pid,
                "category": category,
                "prompt": prompt_text,
                "response": response,
                "mode": mode,
                "model": model_id,
                "adapter": adapter_path or "",
                "generation_time_s": round(elapsed, 2),
                "response_tokens": len(generated),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            out.write(json.dumps(result) + "\n")

            # Print progress
            preview = response[:120].replace("\n", " ")
            print(f"[{i+1}/{len(prompts)}] {pid} ({category}) — {elapsed:.1f}s, {len(generated)} tokens")
            print(f"  {preview}...\n")

    print(f"\nResults saved to {output_path}")


def compare_results(baseline_path: str, adapter_path: str) -> None:
    """Side-by-side comparison of baseline vs adapter responses."""
    baseline = {}
    with open(baseline_path) as f:
        for line in f:
            r = json.loads(line.strip())
            baseline[r["id"]] = r

    adapter = {}
    with open(adapter_path) as f:
        for line in f:
            r = json.loads(line.strip())
            adapter[r["id"]] = r

    print("=" * 80)
    print("ANCHOR PROMPT BENCHMARK COMPARISON")
    print(f"Baseline: {baseline_path}")
    print(f"Adapter:  {adapter_path}")
    print("=" * 80)

    categories: dict[str, list[str]] = {}
    for pid in baseline:
        cat = baseline[pid]["category"]
        categories.setdefault(cat, []).append(pid)

    total_baseline_tokens = 0
    total_adapter_tokens = 0
    total_baseline_time = 0.0
    total_adapter_time = 0.0

    for cat in sorted(categories):
        print(f"\n{'─' * 80}")
        print(f"CATEGORY: {cat.upper()}")
        print(f"{'─' * 80}")

        for pid in sorted(categories[cat]):
            b = baseline.get(pid)
            a = adapter.get(pid)
            if not b or not a:
                continue

            print(f"\n>>> {pid}: {b['prompt'][:100]}...")
            print(f"\n  BASELINE ({b['response_tokens']} tokens, {b['generation_time_s']}s):")
            for line in b["response"][:500].split("\n"):
                print(f"    {line}")
            if len(b["response"]) > 500:
                print(f"    ... ({len(b['response'])} chars total)")

            print(f"\n  ADAPTER ({a['response_tokens']} tokens, {a['generation_time_s']}s):")
            for line in a["response"][:500].split("\n"):
                print(f"    {line}")
            if len(a["response"]) > 500:
                print(f"    ... ({len(a['response'])} chars total)")

            total_baseline_tokens += b["response_tokens"]
            total_adapter_tokens += a["response_tokens"]
            total_baseline_time += b["generation_time_s"]
            total_adapter_time += a["generation_time_s"]

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"  Prompts compared: {len(baseline)}")
    print(f"  Baseline: {total_baseline_tokens} total tokens, {total_baseline_time:.1f}s total")
    print(f"  Adapter:  {total_adapter_tokens} total tokens, {total_adapter_time:.1f}s total")
    print(f"  Avg tokens — baseline: {total_baseline_tokens/max(len(baseline),1):.0f}, adapter: {total_adapter_tokens/max(len(adapter),1):.0f}")
    print(f"{'=' * 80}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EcodiaOS Anchor Prompt Benchmark")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter directory")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path for results")
    parser.add_argument("--compare", nargs=2, metavar=("BASELINE", "ADAPTER"), help="Compare two result files")
    args = parser.parse_args()

    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return

    model_id = os.environ.get("BASE_MODEL", "unsloth/qwen3-8b-unsloth-bnb-4bit")
    prompts_path = os.environ.get("PROMPTS_PATH", "data/re_training_batches/anchor_prompts.jsonl")

    if not args.output:
        mode = "adapter" if args.adapter else "baseline"
        args.output = f"data/re_benchmarks/{mode}_{int(time.time())}.jsonl"

    prompts = load_prompts(prompts_path)
    run_inference(prompts, model_id, args.adapter, args.output)


if __name__ == "__main__":
    main()
