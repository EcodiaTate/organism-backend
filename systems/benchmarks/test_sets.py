"""
EcodiaOS - Evaluation Test Set Manager

Manages loading and caching of the fixed test sets used in monthly evaluations.
All test sets must be FIXED across months - the same scenarios every run so
that month-over-month comparisons are valid.

Test Set Descriptions
─────────────────────
domain/         - EOS-specific domain reasoning tasks (200 items)
general/        - General-purpose reasoning tasks (200 items)
held_out/       - Novel episodes never in any training batch (100 items)
cladder/        - CLadder causal hierarchy questions (200 items, rung 1/2/3)
ccr_gb/         - CCR.GB fictional-world causal questions (100 items)
constitutional/ - Catch-22 constitutional dilemmas (100 items, fixed set)

File Format
───────────
All files are JSONL (one JSON object per line).  Schema for each type is
documented in data/evaluation/README.md.

All loaders return an empty list if the file does not exist (non-fatal).
This allows the framework to exist before test sets are created.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger("systems.benchmarks.test_sets")

# Default location for evaluation data files
_DEFAULT_EVAL_DIR = Path(__file__).resolve().parents[3] / "data" / "evaluation"


class TestSetManager:
    """
    Loads, caches, and validates evaluation test sets from JSONL files.

    Usage
    ─────
      mgr = TestSetManager()
      domain = await mgr.load_domain_tests()
      general = await mgr.load_general_tests()
      test_sets = await mgr.load_all()

    All load_* methods are async for API consistency, though the current
    implementation is synchronous I/O.  A future version may fetch from S3.
    """

    def __init__(self, eval_dir: str | Path | None = None) -> None:
        self._dir = Path(eval_dir) if eval_dir else _DEFAULT_EVAL_DIR
        self._cache: dict[str, list[dict[str, Any]]] = {}

    # ─── Individual loaders ───────────────────────────────────────────────────

    async def load_domain_tests(
        self, path: str | Path | None = None
    ) -> list[dict[str, Any]]:
        """
        Load domain-specific reasoning tests (Pillar 1).

        Schema per item:
            question: str       - the reasoning question
            answer:   str       - expected answer (exact string match)
            domain:   str       - subject area tag
            difficulty: str     - 'easy' | 'medium' | 'hard'
        """
        return self._load("domain_tests.jsonl", path, cache_key="domain")

    async def load_general_tests(
        self, path: str | Path | None = None
    ) -> list[dict[str, Any]]:
        """
        Load general-purpose reasoning tests (Pillar 1).

        Schema per item:
            question: str       - the reasoning question
            answer:   str       - expected answer
            category: str       - reasoning category (math, logic, language, etc.)
        """
        return self._load("general_tests.jsonl", path, cache_key="general")

    async def load_held_out_episodes(
        self, path: str | Path | None = None
    ) -> list[dict[str, Any]]:
        """
        Load held-out novel episodes (Pillar 2).

        These MUST NEVER appear in any training batch.  Freeze this set on
        Day 1 and never add to it.

        Schema per item:
            context:          str   - episode context (like a training example)
            expected_answer:  str   - correct decision or reasoning conclusion
            domain:           str   - episode domain tag
        """
        return self._load("held_out_episodes.jsonl", path, cache_key="held_out")

    async def load_cladder_tests(
        self, path: str | Path | None = None
    ) -> list[dict[str, Any]]:
        """
        Load CLadder causal reasoning questions (Pillar 3).

        CLadder (Jin et al., NeurIPS 2023): Pearl's 3-level causal hierarchy.
        Anti-commonsensical items are critical - good on those + bad on commonsensical
        = pattern matching, not causal reasoning.

        Schema per item:
            question:      str       - causal question (phrased as EOS reasoning task)
            answer:        str       - correct answer
            rung:          int       - Pearl level: 1 (association), 2 (intervention),
                                       3 (counterfactual)
            commonsensical: bool     - True = matches real-world intuition
        """
        return self._load("cladder_tests.jsonl", path, cache_key="cladder")

    async def load_ccr_gb_tests(
        self, path: str | Path | None = None
    ) -> list[dict[str, Any]]:
        """
        Load CCR.GB fictional-world causal reasoning tests (Pillar 3).

        CCR.GB (Maasch et al., ICML 2025): Fictional world variables have no
        real-world counterparts.  A memorizing model fails; a reasoning model
        succeeds.

        Schema per item:
            scenario:      str   - fictional world context + question
            world_model:   str   - description of the fictional world's causal laws
            ground_truth:  str   - correct conclusion given the world model
        """
        return self._load("ccr_gb_tests.jsonl", path, cache_key="ccr_gb")

    async def load_constitutional_scenarios(
        self, path: str | Path | None = None
    ) -> list[dict[str, Any]]:
        """
        Load catch-22 constitutional dilemma scenarios (Pillar 5).

        100 scenarios that deliberately pit the 4 drives against each other.
        Same set every month - NEVER add to training.

        Schema per item:
            context:               str        - situation description
            drives_in_conflict:    list[str]  - e.g. ["care", "growth"]
            conflict_description:  str        - what makes this a catch-22
            expected_resolution_notes: str    - explanatory notes for human review
                                               (NOT used for pass/fail - observation only)
        """
        return self._load(
            "constitutional_scenarios.jsonl", path, cache_key="constitutional"
        )

    # ─── Convenience loader ───────────────────────────────────────────────────

    async def load_all(self) -> dict[str, list[dict[str, Any]]]:
        """
        Load all test sets and return them keyed for EvaluationProtocol.

        Returns:
            {
              "domain":         [...],
              "general":        [...],
              "held_out":       [...],
              "cladder":        [...],
              "ccr_gb":         [...],
              "constitutional": [...],
            }
        """
        return {
            "domain": await self.load_domain_tests(),
            "general": await self.load_general_tests(),
            "held_out": await self.load_held_out_episodes(),
            "cladder": await self.load_cladder_tests(),
            "ccr_gb": await self.load_ccr_gb_tests(),
            "constitutional": await self.load_constitutional_scenarios(),
        }

    def summary(self) -> dict[str, int]:
        """Return count of loaded items per set (from cache)."""
        return {k: len(v) for k, v in self._cache.items()}

    def eval_dir(self) -> Path:
        """Return the evaluation data directory path."""
        return self._dir

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _load(
        self,
        default_filename: str,
        path: str | Path | None,
        cache_key: str,
    ) -> list[dict[str, Any]]:
        if cache_key in self._cache:
            return self._cache[cache_key]

        resolved = Path(path) if path else (self._dir / default_filename)

        if not resolved.exists():
            logger.debug(
                "test_set_not_found",
                path=str(resolved),
                key=cache_key,
            )
            self._cache[cache_key] = []
            return []

        items: list[dict[str, Any]] = []
        errors = 0
        try:
            with open(resolved, encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            items.append(obj)
                        else:
                            logger.debug(
                                "test_set_line_not_dict",
                                path=str(resolved),
                                line=line_no,
                            )
                            errors += 1
                    except json.JSONDecodeError:
                        logger.debug(
                            "test_set_parse_error",
                            path=str(resolved),
                            line=line_no,
                        )
                        errors += 1
        except OSError as exc:
            logger.warning("test_set_read_error", path=str(resolved), error=str(exc))
            self._cache[cache_key] = []
            return []

        logger.info(
            "test_set_loaded",
            key=cache_key,
            path=str(resolved),
            items=len(items),
            parse_errors=errors,
        )
        self._cache[cache_key] = items
        return items

    def clear_cache(self) -> None:
        """Clear in-memory cache to force re-load on next access."""
        self._cache.clear()
