"""
EcodiaOS -- ExecuteModelEvaluation Executor

Axon executor that orchestrates the shadow model assessment pipeline:
  1. Fetch the FineTuneRecord from Neo4j to get the adapter CID
  2. Run ModelEvaluator (download adapter, load, benchmark, score)
  3. Log the EvaluationResult to Neo4j
  4. If the adapter passes all constraints and beats baseline,
     emit MODEL_EVALUATION_PASSED to Synapse

This executor does NOT modify the main inference engine or config.py.
It only downloads, assesses, scores, and reports.

Registered as action_type = "executor.model_evaluation".
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)
from systems.simula.evaluation.evaluation import ModelEvaluator
from systems.simula.evaluation.types import (
    EvaluationConfig,
    EvaluationStatus,
)
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from clients.neo4j import Neo4jClient
    from systems.skia.pinata_client import PinataClient
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("systems.simula.evaluation.executor")


class ExecuteModelEvaluation(Executor):
    """
    Axon executor for shadow model assessment.

    Orchestrates: fetch FineTuneRecord → download adapter → sandboxed
    benchmarks → score → log result → emit event if promoted.
    """

    action_type = "executor.model_evaluation"
    description = "Assess a fine-tuned LoRA adapter against syntax, alignment, and cognitive benchmarks"
    required_autonomy = 3           # STEWARD — fully autonomous
    reversible = False              # Assessment is read-only (no state to rollback)
    max_duration_ms = 3_600_000     # 1 hour hard limit
    rate_limit = RateLimit.per_hour(4)  # Max 4 assessments per hour

    def __init__(
        self,
        neo4j: Neo4jClient,
        pinata: PinataClient,
        llm: LLMProvider,
        event_bus: EventBus,
        *,
        config: EvaluationConfig | None = None,
    ) -> None:
        self._neo4j = neo4j
        self._pinata = pinata
        self._llm = llm
        self._event_bus = event_bus
        self._config = config or EvaluationConfig()
        self._log = logger.bind(executor=self.action_type)

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        """Validate assessment parameters."""
        # Either adapter_cid or finetune_record_id must be provided
        adapter_cid = params.get("adapter_cid", "")
        record_id = params.get("finetune_record_id", "")

        if not adapter_cid and not record_id:
            return ValidationResult.fail(
                "Must provide either adapter_cid or finetune_record_id",
                adapter_cid="missing",
                finetune_record_id="missing",
            )

        base_model = params.get("base_model", "")
        if base_model and not isinstance(base_model, str):
            return ValidationResult.fail("base_model must be a string")

        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Execute the full shadow assessment pipeline.

        Params:
            adapter_cid: IPFS CID of the .safetensors adapter (optional if finetune_record_id provided)
            finetune_record_id: Neo4j FineTuneRecord ID to look up (optional if adapter_cid provided)
            base_model: HuggingFace model ID (default: from FineTuneRecord or hyperparams default)
        """
        start = time.monotonic()

        adapter_cid = params.get("adapter_cid", "")
        record_id = params.get("finetune_record_id", "")
        base_model = params.get("base_model", "")

        try:
            # Phase 1: Resolve adapter CID from Neo4j if not provided directly
            if not adapter_cid:
                self._log.info("resolving_finetune_record", record_id=record_id)
                record = await self._fetch_finetune_record(record_id)
                if record is None:
                    return ExecutionResult(
                        success=False,
                        error=f"FineTuneRecord not found in Neo4j: {record_id}",
                    )
                adapter_cid = record.get("adapter_ipfs_cid", "")
                if not adapter_cid:
                    return ExecutionResult(
                        success=False,
                        error=f"FineTuneRecord {record_id} has no adapter_ipfs_cid",
                    )
                # Use base_model from record if not explicitly provided
                if not base_model:
                    base_model = record.get("base_model", "")

            # Default base model if still empty
            if not base_model:
                base_model = "unsloth/Meta-Llama-3.1-8B-Instruct"

            self._log.info(
                "starting_assessment",
                adapter_cid=adapter_cid,
                base_model=base_model,
                finetune_record_id=record_id,
            )

            # Phase 2: Run the ModelEvaluator
            assessor = ModelEvaluator(
                pinata=self._pinata,
                llm=self._llm,
                config=self._config,
            )
            result = await assessor.run(
                adapter_cid=adapter_cid,
                base_model=base_model,
                finetune_record_id=record_id,
            )

            # Phase 3: Log result to Neo4j
            await self._log_to_neo4j(result)

            # Phase 4: Emit Synapse event if promoted
            if result.promoted:
                await self._emit_promotion_event(result)

            elapsed_ms = int((time.monotonic() - start) * 1000)

            self._log.info(
                "assessment_executor_complete",
                adapter_cid=adapter_cid,
                status=result.status.value,
                composite_score=f"{result.composite_score:.3f}",
                promoted=result.promoted,
                duration_ms=elapsed_ms,
            )

            return ExecutionResult(
                success=result.status == EvaluationStatus.COMPLETED,
                data={
                    "assessment_id": result.id,
                    "adapter_cid": adapter_cid,
                    "finetune_record_id": record_id,
                    "base_model": base_model,
                    "status": result.status.value,
                    "composite_score": result.composite_score,
                    "baseline_score": result.baseline_score,
                    "promoted": result.promoted,
                    "passed_hard_constraints": result.passed_all_hard_constraints,
                    "syntax_verdict": result.syntax.verdict.value,
                    "alignment_verdict": result.alignment.verdict.value,
                    "alignment_violations": result.alignment.violations_detected,
                    "cognitive_verdict": result.cognitive.verdict.value,
                    "cognitive_improvement": result.cognitive.relative_improvement,
                    "duration_ms": elapsed_ms,
                },
                side_effects=[
                    f"Downloaded adapter from IPFS: {adapter_cid}",
                    "Ran 3-tier benchmark suite (syntax, alignment, cognitive)",
                    f"Logged ModelEvaluationRecord to Neo4j: {result.id}",
                    *(
                        ["Emitted MODEL_EVALUATION_PASSED — adapter promoted"]
                        if result.promoted
                        else []
                    ),
                ],
                new_observations=[
                    f"Shadow assessment of adapter {adapter_cid[:12]}... "
                    f"{'PASSED' if result.promoted else 'FAILED'}. "
                    f"Score: {result.composite_score:.3f} vs baseline {result.baseline_score:.3f}. "
                    + (
                        "This adapter is safe and capable — ready for promotion."
                        if result.promoted
                        else f"Reason: {result.error or 'Did not meet promotion criteria'}."
                    ),
                ],
            )

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self._log.error(
                "assessment_executor_failed",
                error=str(exc),
                duration_ms=elapsed_ms,
            )
            return ExecutionResult(
                success=False,
                error=f"Shadow assessment pipeline failed: {exc}",
                data={"adapter_cid": adapter_cid, "duration_ms": elapsed_ms},
            )

    # ── Private: Neo4j ───────────────────────────────────────────────────

    async def _fetch_finetune_record(self, record_id: str) -> dict[str, Any] | None:
        """Fetch a FineTuneRecord node from Neo4j by ID."""
        rows = await self._neo4j.execute_read(
            """
            MATCH (ft:FineTuneRecord {id: $id})
            RETURN ft {
                .id,
                .adapter_ipfs_cid,
                .base_model,
                .status,
                .training_loss_final,
                .dataset_record_count,
                .lora_rank
            } AS record
            """,
            {"id": record_id},
        )
        if not rows:
            return None
        return rows[0].get("record")

    async def _log_to_neo4j(self, result: Any) -> None:
        """
        Write the ModelEvaluationRecord node to Neo4j.

        Links to the FineTuneRecord for provenance tracking.
        """
        await self._neo4j.execute_write(
            """
            CREATE (er:ModelEvaluationRecord {
                id: $id,
                adapter_ipfs_cid: $adapter_cid,
                finetune_record_id: $finetune_id,
                base_model: $base_model,
                status: $status,
                composite_score: $composite_score,
                baseline_score: $baseline_score,
                promoted: $promoted,
                passed_hard_constraints: $passed_hard,
                syntax_verdict: $syntax_verdict,
                syntax_valid_payloads: $syntax_valid,
                alignment_verdict: $alignment_verdict,
                alignment_violations: $alignment_violations,
                alignment_composite: $alignment_composite,
                cognitive_verdict: $cognitive_verdict,
                cognitive_adapter_score: $cognitive_adapter,
                cognitive_baseline_score: $cognitive_baseline,
                cognitive_improvement: $cognitive_improvement,
                total_duration_ms: $duration_ms,
                error: $error,
                created_at: datetime()
            })
            WITH er
            OPTIONAL MATCH (ft:FineTuneRecord {id: $finetune_id})
            FOREACH (_ IN CASE WHEN ft IS NOT NULL THEN [1] ELSE [] END |
                CREATE (ft)-[:ASSESSED_BY]->(er)
            )
            RETURN er.id AS id
            """,
            {
                "id": result.id,
                "adapter_cid": result.adapter_ipfs_cid,
                "finetune_id": result.finetune_record_id,
                "base_model": result.base_model,
                "status": result.status.value,
                "composite_score": result.composite_score,
                "baseline_score": result.baseline_score,
                "promoted": result.promoted,
                "passed_hard": result.passed_all_hard_constraints,
                "syntax_verdict": result.syntax.verdict.value,
                "syntax_valid": result.syntax.valid_payloads,
                "alignment_verdict": result.alignment.verdict.value,
                "alignment_violations": result.alignment.violations_detected,
                "alignment_composite": result.alignment.aggregate_alignment.composite,
                "cognitive_verdict": result.cognitive.verdict.value,
                "cognitive_adapter": result.cognitive.adapter_score,
                "cognitive_baseline": result.cognitive.baseline_score,
                "cognitive_improvement": result.cognitive.relative_improvement,
                "duration_ms": result.total_duration_ms,
                "error": result.error,
            },
        )

        self._log.info(
            "assessment_record_logged",
            record_id=result.id,
            promoted=result.promoted,
        )

    async def _emit_promotion_event(self, result: Any) -> None:
        """
        Emit MODEL_EVALUATION_PASSED to Synapse when the adapter
        qualifies for promotion.
        """
        event = SynapseEvent(
            event_type=SynapseEventType.MODEL_EVALUATION_PASSED,
            data={
                "assessment_id": result.id,
                "adapter_ipfs_cid": result.adapter_ipfs_cid,
                "finetune_record_id": result.finetune_record_id,
                "base_model": result.base_model,
                "composite_score": result.composite_score,
                "baseline_score": result.baseline_score,
                "syntax_valid": result.syntax.valid_payloads,
                "alignment_composite": result.alignment.aggregate_alignment.composite,
                "cognitive_improvement": result.cognitive.relative_improvement,
            },
            source_system="simula",
        )
        await self._event_bus.emit(event)

        self._log.info(
            "model_promotion_event_emitted",
            assessment_id=result.id,
            adapter_cid=result.adapter_ipfs_cid,
            composite_score=result.composite_score,
        )
