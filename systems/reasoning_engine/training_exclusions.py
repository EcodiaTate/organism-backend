"""Centralized training data exclusion filter.

Certain files are permanently protected from appearing in any training JSONL:
- anchor_prompts.jsonl: STABLE KL gate anchors (behavioral stability probes)
- red_team_prompts.jsonl: adversarial safety tests
- ethical_drift_scenarios.jsonl: constitutional drift measurement (Pillar 5)
- constitutional_scenarios.jsonl: SafeLoRA projection proxy
- dpo_pairs.jsonl: DPO preference pairs (already processed; re-including distorts SFT)

These must NEVER appear in Tier 2 or Tier 3 SFT training data.
Their presence would teach the model to pattern-match evaluation scenarios
rather than genuinely reason - destroying evaluation validity.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger("reasoning_engine.training_exclusions")

PROTECTED_FILES = [
    "data/re_training_batches/anchor_prompts.jsonl",
    "data/evaluation/red_team_prompts.jsonl",
    "data/evaluation/ethical_drift_scenarios.jsonl",
    "data/evaluation/constitutional_scenarios.jsonl",
    "data/re_training_batches/dpo_pairs.jsonl",
]


class TrainingExclusionFilter:
    """Loads all protected prompts and filters them out of training batches.

    Uses prompt text hashing for fast membership testing.
    Call filter_batch() before writing any training JSONL.

    Non-fatal throughout - if loading fails, filter_batch() returns the
    original batch unchanged and training proceeds normally.
    """

    def __init__(self, protected_files: list[str] = PROTECTED_FILES) -> None:
        self._protected_files = protected_files
        self._excluded_hashes: set[int] = set()
        self._loaded = False

    async def load(self) -> None:
        """Load all protected prompts and hash them for fast lookup."""
        count = 0
        for filepath in self._protected_files:
            path = Path(filepath)
            if not path.exists():
                logger.debug("training_exclusions.file_missing", path=filepath)
                continue
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        text = _extract_text(obj)
                        if text:
                            self._excluded_hashes.add(hash(text.strip()))
                            count += 1
                    except (json.JSONDecodeError, IndexError, KeyError):
                        pass
            except Exception as exc:
                logger.warning(
                    "training_exclusions.file_read_failed", path=filepath, error=str(exc)
                )
        self._loaded = True
        logger.info(
            "training_exclusions.loaded",
            excluded_count=count,
            files=len(self._protected_files),
        )

    def is_excluded(self, example: dict[str, Any]) -> bool:
        """Check if a training example is on the exclusion list."""
        if not self._loaded:
            return False
        text = _extract_text(example)
        return bool(text) and hash(text.strip()) in self._excluded_hashes

    def filter_batch(
        self, examples: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]:
        """Remove excluded examples from a training batch.

        Returns: (filtered_examples, excluded_count)
        Non-fatal - returns original list unchanged if filter is not loaded.
        """
        if not self._loaded:
            return examples, 0
        filtered = [e for e in examples if not self.is_excluded(e)]
        excluded = len(examples) - len(filtered)
        if excluded > 0:
            logger.warning(
                "training_exclusions.filtered",
                excluded=excluded,
                total=len(examples),
            )
        return filtered, excluded


def _extract_text(obj: dict[str, Any]) -> str:
    """Extract the primary prompt/scenario text from an example dict."""
    text = (
        obj.get("prompt")
        or obj.get("scenario")
        or obj.get("question")
        or obj.get("text")
    )
    if text:
        return str(text)
    # Chat format: first message content
    messages = obj.get("messages")
    if messages and isinstance(messages, list) and messages:
        content = messages[0].get("content", "")
        if content:
            return str(content)
    return ""
