"""
EcodiaOS -- Simula Failure Analyzer (Stage 6B.1)

Autonomous failure case extraction from failure history.

Failure cases are code-generation failures that the model should
learn to avoid: rollbacks, verification failures, health-check crashes.
These are mined from Neo4j evolution history and fed into the GRPO
training loop (Stage 4B) to improve failure resistance.

Sources:
  - Rolled-back proposals (code that was applied and then reverted)
  - Health check failures (syntax, import, test failures)
  - Formal verification failures (Dafny, Z3, Lean rejections)
  - Robustness test failures (Stage 6B.2)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog

from systems.simula.verification.types import (
    CoevolutionCycleResult,
    FailureCaseExample,
    FailureCaseSource,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from clients.neo4j import Neo4jClient
    from systems.simula.coevolution.robustness_tester import (
        RobustnessTestGenerator,
    )

logger = structlog.get_logger().bind(system="simula.coevolution.failure_analyzer")


class FailureAnalyzer:
    """Mines hard negative training examples from evolution failure history."""

    def __init__(
        self,
        neo4j: Neo4jClient | None = None,
        llm: LLMProvider | None = None,
        *,
        max_negatives_per_cycle: int = 50,
    ) -> None:
        self._neo4j = neo4j
        self._llm = llm
        self._max_per_cycle = max_negatives_per_cycle

    # ── Public API ──────────────────────────────────────────────────────────

    async def mine_from_history(self) -> list[FailureCaseExample]:
        """
        Query Neo4j for rolled-back or verification-failed proposals.

        Returns hard negative examples for GRPO training.
        """
        start = time.monotonic()
        negatives: list[FailureCaseExample] = []

        # Mine from rollbacks
        rollback_negs = await self._mine_rollbacks()
        negatives.extend(rollback_negs)

        # Mine from formal verification failures
        fv_negs = await self._mine_verification_failures()
        negatives.extend(fv_negs)

        # Cap at max
        negatives = negatives[: self._max_per_cycle]

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "failure_cases_mined",
            total=len(negatives),
            rollbacks=len(rollback_negs),
            verification_failures=len(fv_negs),
            duration_ms=elapsed_ms,
        )
        return negatives

    async def mine_from_robustness(
        self,
        generator: RobustnessTestGenerator,
        files: list[str] | None = None,
    ) -> list[FailureCaseExample]:
        """
        Run adversarial test generation and convert failures to hard negatives.
        """
        start = time.monotonic()

        past_failures = await self.mine_from_history()
        adv_result = await generator.generate_robustness_tests(
            files=files or [],
            past_failures=past_failures,
        )

        negatives: list[FailureCaseExample] = []
        for bug in adv_result.bug_descriptions:
            negatives.append(
                FailureCaseExample(
                    source=FailureCaseSource.ADVERSARIAL_GENERATION,
                    failure_reason=bug,
                    adversarial_input=bug,
                ),
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "robustness_negatives_mined",
            bugs_found=adv_result.tests_found_bugs,
            negatives=len(negatives),
            duration_ms=elapsed_ms,
        )
        return negatives

    async def prepare_grpo_batch(
        self,
        negatives: list[FailureCaseExample],
    ) -> list[dict[str, object]]:
        """
        Format hard negatives as GRPO training examples.

        Each example has reward=0.0 (failure), paired with the code
        context and failure reason for contrastive learning.
        """
        batch: list[dict[str, object]] = []
        for neg in negatives:
            batch.append({
                "proposal_id": neg.proposal_id,
                "category": neg.category,
                "code_output": neg.code_context,
                "failure_reason": neg.failure_reason,
                "reward": 0.0,  # hard negative = zero reward
                "source": neg.source.value,
            })
        return batch

    async def run_cycle(
        self,
        adversarial_generator: RobustnessTestGenerator | None = None,
        files: list[str] | None = None,
    ) -> CoevolutionCycleResult:
        """
        Run one complete co-evolution cycle:
        1. Mine hard negatives from history
        2. Optionally run adversarial test generation
        3. Prepare GRPO training batch
        """
        start = time.monotonic()

        # Phase 1: Mine from history
        history_negs = await self.mine_from_history()

        # Phase 2: Adversarial testing
        adv_negs: list[FailureCaseExample] = []
        adv_tests = 0
        bugs_found = 0
        coverage_growth = 0.0

        if adversarial_generator is not None:
            adv_negs = await self.mine_from_robustness(
                adversarial_generator, files,
            )
            # Get stats from generator
            adv_tests = len(adv_negs)
            bugs_found = len(adv_negs)

        # Phase 3: Prepare GRPO batch
        all_negs = history_negs + adv_negs
        grpo_batch = await self.prepare_grpo_batch(all_negs)

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "coevolution_cycle_complete",
            history_negatives=len(history_negs),
            adversarial_negatives=len(adv_negs),
            grpo_examples=len(grpo_batch),
            duration_ms=elapsed_ms,
        )

        return CoevolutionCycleResult(
            failure_cases_mined=len(history_negs),
            adversarial_tests_generated=adv_tests,
            tests_found_bugs=bugs_found,
            grpo_examples_produced=len(grpo_batch),
            coverage_growth_percent=coverage_growth,
            duration_ms=elapsed_ms,
        )

    # ── Private: Mining from Neo4j ──────────────────────────────────────────

    async def _mine_rollbacks(self) -> list[FailureCaseExample]:
        """Mine hard negatives from rolled-back proposals."""
        if self._neo4j is None:
            return []

        rows = await self._neo4j.execute_read(
            """
            MATCH (e:EvolutionRecord)
            WHERE e.rolled_back = true
            RETURN e.proposal_id AS proposal_id,
                   e.category AS category,
                   e.description AS description,
                   e.rollback_reason AS rollback_reason
            ORDER BY e.applied_at DESC
            LIMIT $limit
            """,
            {"limit": self._max_per_cycle},
        )

        return [
            FailureCaseExample(
                source=FailureCaseSource.ROLLBACK_HISTORY,
                proposal_id=str(row["proposal_id"]),
                category=str(row["category"]),
                failure_reason=str(row.get("rollback_reason", "")),
                code_context=str(row.get("description", "")),
            )
            for row in rows
        ]

    async def _mine_verification_failures(self) -> list[FailureCaseExample]:
        """Mine hard negatives from formal verification failures."""
        if self._neo4j is None:
            return []

        rows = await self._neo4j.execute_read(
            """
            MATCH (e:EvolutionRecord)
            WHERE e.formal_verification_status = 'failed'
               OR e.lean_proof_status = 'failed'
            RETURN e.proposal_id AS proposal_id,
                   e.category AS category,
                   e.description AS description,
                   e.formal_verification_status AS fv_status,
                   e.lean_proof_status AS lean_status
            ORDER BY e.applied_at DESC
            LIMIT $limit
            """,
            {"limit": self._max_per_cycle},
        )

        return [
            FailureCaseExample(
                source=FailureCaseSource.FORMAL_VERIFICATION_FAILURE,
                proposal_id=str(row["proposal_id"]),
                category=str(row["category"]),
                failure_reason=(
                    f"Formal verification: {row.get('fv_status', '')}, "
                    f"Lean: {row.get('lean_status', '')}"
                ),
                code_context=str(row.get("description", "")),
            )
            for row in rows
        ]
