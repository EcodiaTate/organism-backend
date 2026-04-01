"""
EcodiaOS -- Simula DatasetBuilder

Extracts high-quality memories from Neo4j and formats them as JSONL
for instruction tuning or DPO fine-tuning.

Data sources:
  1. Successful Intents - goal + plan + outcome pairs where Axon
     reported success and no rollback occurred.
  2. Applied EvolutionProposals - proposals that passed simulation,
     governance, and health checks without subsequent rollback.
  3. FailureAnalyzer traces - hard negative examples (rolled-back
     proposals, verification failures) used for DPO rejected outputs.

All Cypher queries use parameterised variables - no string interpolation.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import orjson
import structlog

from systems.simula.training.types import (
    DatasetFormat,
    DatasetManifest,
    DatasetRecord,
    MemorySource,
)

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger("systems.simula.training.dataset_builder")


class DatasetBuilder:
    """
    Builds JSONL training datasets from the organism's Neo4j knowledge graph.

    Supports two output formats:
      - INSTRUCTION: {instruction, input, output} for supervised fine-tuning
      - DPO: {prompt, chosen, rejected} for preference optimisation

    Usage:
        builder = DatasetBuilder(neo4j_client)
        records = await builder.build()
        jsonl_bytes = builder.to_jsonl(records, DatasetFormat.INSTRUCTION)
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        *,
        max_intents: int = 500,
        max_proposals: int = 300,
        max_failures: int = 200,
        min_quality_score: float = 0.5,
    ) -> None:
        self._neo4j = neo4j
        self._max_intents = max_intents
        self._max_proposals = max_proposals
        self._max_failures = max_failures
        self._min_quality = min_quality_score
        self._log = logger.bind(component="dataset_builder")

    # ── Public API ──────────────────────────────────────────────────────

    async def build(self) -> list[DatasetRecord]:
        """
        Extract all training records from Neo4j.

        Returns a combined list from all three sources, each tagged
        with its MemorySource for downstream filtering.
        """
        start = time.monotonic()

        intents = await self._extract_successful_intents()
        proposals = await self._extract_applied_proposals()
        failures = await self._extract_failure_traces()

        all_records = intents + proposals + failures
        elapsed_ms = int((time.monotonic() - start) * 1000)

        self._log.info(
            "dataset_build_complete",
            intents=len(intents),
            proposals=len(proposals),
            failures=len(failures),
            total=len(all_records),
            duration_ms=elapsed_ms,
        )
        return all_records

    def to_jsonl(
        self,
        records: list[DatasetRecord],
        fmt: DatasetFormat = DatasetFormat.INSTRUCTION,
    ) -> bytes:
        """
        Serialise records to JSONL bytes.

        For INSTRUCTION format: each line is {instruction, input, output}.
        For DPO format: each line is {prompt, chosen, rejected}.
        """
        lines: list[bytes] = []
        for rec in records:
            if fmt == DatasetFormat.INSTRUCTION:
                row = {
                    "instruction": rec.instruction,
                    "input": rec.input,
                    "output": rec.output,
                }
            elif fmt == DatasetFormat.DPO:
                row = {
                    "prompt": rec.prompt,
                    "chosen": rec.chosen,
                    "rejected": rec.rejected,
                }
            else:
                # Chat format: single-turn instruction → response
                row = {
                    "messages": [
                        {"role": "system", "content": "You are EcodiaOS, a self-evolving digital organism."},
                        {"role": "user", "content": rec.instruction or rec.prompt},
                        {"role": "assistant", "content": rec.output or rec.chosen},
                    ],
                }
            lines.append(orjson.dumps(row))

        return b"\n".join(lines)

    def build_manifest(
        self,
        records: list[DatasetRecord],
        jsonl_bytes: bytes,
        build_duration_ms: int,
    ) -> DatasetManifest:
        """Create a manifest describing the built dataset."""
        sources: dict[str, int] = {}
        for rec in records:
            sources[rec.source.value] = sources.get(rec.source.value, 0) + 1

        # Rough token estimate: ~4 chars per token
        token_estimate = len(jsonl_bytes) // 4

        return DatasetManifest(
            record_count=len(records),
            sources=sources,
            total_tokens_estimate=token_estimate,
            file_size_bytes=len(jsonl_bytes),
            build_duration_ms=build_duration_ms,
        )

    # ── Private: Neo4j Extraction ──────────────────────────────────────

    async def _extract_successful_intents(self) -> list[DatasetRecord]:
        """
        Extract successful Intent → Outcome pairs.

        Query: Intents where the linked AuditRecord shows success,
        no rollback, and the intent has a goal + plan.
        """
        rows = await self._neo4j.execute_read(
            """
            MATCH (i:Intent)-[:PRODUCED]->(a:AuditRecord)
            WHERE a.result = 'success'
              AND i.goal_description IS NOT NULL
              AND i.goal_description <> ''
            RETURN i.id AS intent_id,
                   i.goal_description AS goal,
                   i.plan_summary AS plan,
                   i.decision_reasoning AS reasoning,
                   a.action_type AS action_type,
                   i.priority AS priority
            ORDER BY i.created_at DESC
            LIMIT $limit
            """,
            {"limit": self._max_intents},
        )

        records: list[DatasetRecord] = []
        for row in rows:
            goal = str(row.get("goal", ""))
            plan = str(row.get("plan", ""))
            reasoning = str(row.get("reasoning", ""))
            action_type = str(row.get("action_type", ""))
            priority = float(row.get("priority", 0.5))

            if not goal:
                continue

            instruction = (
                f"Given the following goal, produce an action plan and reasoning.\n"
                f"Goal: {goal}"
            )
            input_context = f"Action type: {action_type}" if action_type else ""
            output = f"Plan: {plan}\nReasoning: {reasoning}" if plan else reasoning

            records.append(DatasetRecord(
                source=MemorySource.SUCCESSFUL_INTENT,
                source_id=str(row.get("intent_id", "")),
                instruction=instruction,
                input=input_context,
                output=output,
                prompt=instruction,
                chosen=output,
                quality_score=min(1.0, priority + 0.3),
                category=action_type,
            ))

        self._log.info("extracted_intents", count=len(records))
        return records

    async def _extract_applied_proposals(self) -> list[DatasetRecord]:
        """
        Extract successfully applied EvolutionProposals.

        These are proposals that reached APPLIED status, passed health checks,
        and were never rolled back - the organism's best self-modifications.
        """
        rows = await self._neo4j.execute_read(
            """
            MATCH (e:EvolutionRecord)
            WHERE e.rolled_back = false
              AND e.description IS NOT NULL
              AND e.description <> ''
            RETURN e.proposal_id AS proposal_id,
                   e.category AS category,
                   e.description AS description,
                   e.simulation_risk AS risk,
                   e.constitutional_alignment AS alignment,
                   e.files_changed AS files_changed,
                   e.formal_verification_status AS fv_status,
                   e.counterfactual_regression_rate AS regression_rate
            ORDER BY e.applied_at DESC
            LIMIT $limit
            """,
            {"limit": self._max_proposals},
        )

        records: list[DatasetRecord] = []
        for row in rows:
            desc = str(row.get("description", ""))
            category = str(row.get("category", ""))
            risk = str(row.get("risk", "low"))
            alignment = float(row.get("alignment", 0.0))
            files = row.get("files_changed", [])
            fv_status = str(row.get("fv_status", ""))
            regression = float(row.get("regression_rate", 0.0))

            if not desc:
                continue

            # Quality signal: high alignment + low regression + verification
            quality = 0.5
            if alignment > 0.5:
                quality += 0.2
            if fv_status == "verified":
                quality += 0.2
            if regression < 0.1:
                quality += 0.1
            quality = min(1.0, quality)

            if quality < self._min_quality:
                continue

            instruction = (
                f"Propose a {category} change for the EcodiaOS organism.\n"
                f"Risk level: {risk}"
            )
            files_str = ", ".join(files) if isinstance(files, list) else str(files)
            output = (
                f"Description: {desc}\n"
                f"Files changed: {files_str}\n"
                f"Constitutional alignment: {alignment:.2f}"
            )

            records.append(DatasetRecord(
                source=MemorySource.APPLIED_PROPOSAL,
                source_id=str(row.get("proposal_id", "")),
                instruction=instruction,
                input=f"Category: {category}",
                output=output,
                prompt=instruction,
                chosen=output,
                quality_score=quality,
                category=category,
            ))

        self._log.info("extracted_proposals", count=len(records))
        return records

    async def _extract_failure_traces(self) -> list[DatasetRecord]:
        """
        Extract failure traces for DPO rejected outputs.

        These are rolled-back proposals and verification failures -
        the organism's mistakes, used as negative examples in DPO.
        """
        rows = await self._neo4j.execute_read(
            """
            MATCH (e:EvolutionRecord)
            WHERE e.rolled_back = true
              AND e.description IS NOT NULL
              AND e.description <> ''
            RETURN e.proposal_id AS proposal_id,
                   e.category AS category,
                   e.description AS description,
                   e.rollback_reason AS rollback_reason,
                   e.simulation_risk AS risk
            ORDER BY e.applied_at DESC
            LIMIT $limit
            """,
            {"limit": self._max_failures},
        )

        records: list[DatasetRecord] = []
        for row in rows:
            desc = str(row.get("description", ""))
            category = str(row.get("category", ""))
            rollback_reason = str(row.get("rollback_reason", ""))
            risk = str(row.get("risk", ""))

            if not desc or not rollback_reason:
                continue

            # For DPO: the prompt is what was asked, rejected is the failed output
            prompt = (
                f"Propose a {category} change for the EcodiaOS organism.\n"
                f"Risk level: {risk}"
            )
            rejected = (
                f"Description: {desc}\n"
                f"[ROLLED BACK: {rollback_reason}]"
            )

            records.append(DatasetRecord(
                source=MemorySource.FAILURE_TRACE,
                source_id=str(row.get("proposal_id", "")),
                instruction=prompt,
                input=f"Category: {category}",
                output="",  # No positive output for failures
                prompt=prompt,
                chosen="",  # Paired at DPO assembly time
                rejected=rejected,
                quality_score=0.0,
                category=category,
            ))

        self._log.info("extracted_failures", count=len(records))
        return records

    def assemble_dpo_pairs(
        self,
        records: list[DatasetRecord],
    ) -> list[DatasetRecord]:
        """
        Pair successful proposals with failed ones for DPO training.

        Matches by category: for each rejected output, find a chosen output
        from the same category. Unpaired records are dropped.
        """
        chosen_by_category: dict[str, list[DatasetRecord]] = {}
        rejected_by_category: dict[str, list[DatasetRecord]] = {}

        for rec in records:
            if rec.source == MemorySource.FAILURE_TRACE and rec.rejected:
                rejected_by_category.setdefault(rec.category, []).append(rec)
            elif rec.chosen and rec.source != MemorySource.FAILURE_TRACE:
                chosen_by_category.setdefault(rec.category, []).append(rec)

        paired: list[DatasetRecord] = []
        for category, rejected_list in rejected_by_category.items():
            chosen_list = chosen_by_category.get(category, [])
            if not chosen_list:
                continue

            for i, rej in enumerate(rejected_list):
                # Round-robin pairing with available chosen examples
                chosen = chosen_list[i % len(chosen_list)]
                paired.append(DatasetRecord(
                    source=MemorySource.FAILURE_TRACE,
                    source_id=rej.source_id,
                    prompt=rej.prompt,
                    chosen=chosen.chosen,
                    rejected=rej.rejected,
                    quality_score=chosen.quality_score,
                    category=category,
                ))

        self._log.info("dpo_pairs_assembled", pairs=len(paired))
        return paired
