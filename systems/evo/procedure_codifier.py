"""
EcodiaOS - Evo Procedure Codifier

Converts observed (Intent, Outcome) pairs into reusable Procedure objects.
Where ProcedureExtractor works from raw SequenceDetector pattern hashes,
ProcedureCodifier works from structured intent/outcome records - the typed
artefacts Axon produces when executing Nova's plans.

Process:
  1. Accumulate (IntentRecord, OutcomeRecord) pairs via observe()
  2. When a sequence of action types recurs >= min_occurrences times with
     a positive outcome, it is eligible for codification
  3. LLM extracts the generalised preconditions/steps/postconditions from
     the concrete examples
  4. A Procedure node is MERGE-written to the Memory graph
  5. The Procedure is linked to its source episodes (SOURCED_FROM edges)

Performance: codification ≤5s per procedure (same budget as ProcedureExtractor).
Max 3 new procedures per codification call (VELOCITY_LIMITS).

This is the twin of ProcedureExtractor - one reads raw episode hashes,
the other reads typed Intent + Outcome records from Axon's execution path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

from clients.llm import LLMProvider, Message
from clients.optimized_llm import OptimizedLLMProvider
from systems.evo.types import (
    VELOCITY_LIMITS,
    Procedure,
    ProcedureStep,
)

if TYPE_CHECKING:
    from systems.memory.service import MemoryService

logger = structlog.get_logger()

_MAX_PER_CALL: int = VELOCITY_LIMITS["max_new_procedures_per_cycle"]
_MIN_OCCURRENCES: int = 3   # How many times a sequence must recur
_MAX_EXAMPLES: int = 8      # Max examples to send to the LLM

_SYSTEM_PROMPT = (
    "Procedure extraction. "
    "Extract generalised, reusable procedures from concrete intent-outcome pairs. "
    "Capture the essential pattern, not specific parameter values. "
    "Respond with valid JSON."
)


# ─── Value types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class IntentRecord:
    """
    Lightweight summary of an Intent, extracted from the Axon outcome event.

    We do not store the full Intent to avoid coupling this module to Nova's
    full plan representation. The action_types list is the key for grouping.
    """

    intent_id: str
    goal_description: str
    action_types: tuple[str, ...]   # executor strings from plan.steps, ordered
    episode_id: str = ""            # Memory episode ID for this execution, if any


@dataclass(frozen=True)
class OutcomeRecord:
    """
    Result of an executed intent, published by Axon as ACTION_COMPLETED.
    """

    intent_id: str
    success: bool
    economic_delta: float = 0.0     # Revenue/cost impact in USD
    outcome_summary: str = ""       # Short natural-language description


@dataclass
class _SequenceBucket:
    """Accumulator for one recurring (action_types) sequence."""

    action_types: tuple[str, ...]
    successes: list[tuple[IntentRecord, OutcomeRecord]] = field(default_factory=list)
    failures: list[tuple[IntentRecord, OutcomeRecord]] = field(default_factory=list)

    @property
    def success_count(self) -> int:
        return len(self.successes)

    @property
    def total_count(self) -> int:
        return len(self.successes) + len(self.failures)

    @property
    def success_rate(self) -> float:
        return self.success_count / max(1, self.total_count)

    def is_mature(self, min_occurrences: int) -> bool:
        # Require the sequence to have succeeded at least min_occurrences times
        return self.success_count >= min_occurrences


# ─── Codifier ─────────────────────────────────────────────────────────────────


class ProcedureCodifier:
    """
    Converts recurring (Intent, Outcome) pairs into typed Procedure objects
    stored in the Memory graph.

    Typical usage:
        codifier.observe(intent_record, outcome_record)   # called per Axon event
        new_procs = await codifier.codify()               # called during consolidation

    The codify() call is non-blocking at the call site: it returns a list of
    newly created Procedure objects (also persisted to Neo4j).
    """

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemoryService | None = None,
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._logger = logger.bind(system="evo.procedure_codifier")
        self._optimized = isinstance(llm, OptimizedLLMProvider)

        # sequence_key (canonical action_types tuple) → _SequenceBucket
        self._buckets: dict[tuple[str, ...], _SequenceBucket] = {}
        # Procedure IDs already codified - never re-extract the same sequence
        self._codified_sequences: set[tuple[str, ...]] = set()

        # Metrics
        self._total_observed: int = 0
        self._total_codified: int = 0

    # ─── Public API ───────────────────────────────────────────────────────────

    def observe(self, intent: IntentRecord, outcome: OutcomeRecord) -> None:
        """
        Record one (intent, outcome) pair.

        Called for every ACTION_COMPLETED event Evo receives. No I/O.
        """
        self._total_observed += 1
        key = intent.action_types
        if key not in self._buckets:
            self._buckets[key] = _SequenceBucket(action_types=key)

        bucket = self._buckets[key]
        pair = (intent, outcome)
        if outcome.success:
            bucket.successes.append(pair)
        else:
            bucket.failures.append(pair)

    async def codify(self) -> list[Procedure]:
        """
        Attempt to codify any mature, uncodified sequences into Procedures.

        Returns the list of newly created Procedures. Each is also persisted
        to the Memory graph (MERGE into :Procedure node with SOURCED_FROM
        edges back to source episodes).

        Respects _MAX_PER_CALL velocity limit: stops after 3 per call.
        """
        mature = [
            bucket
            for key, bucket in self._buckets.items()
            if bucket.is_mature(_MIN_OCCURRENCES) and key not in self._codified_sequences
        ]

        if not mature:
            return []

        results: list[Procedure] = []
        for bucket in mature:
            if len(results) >= _MAX_PER_CALL:
                break

            procedure = await self._codify_bucket(bucket)
            if procedure is not None:
                self._codified_sequences.add(bucket.action_types)
                self._total_codified += 1
                results.append(procedure)

        return results

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_observed": self._total_observed,
            "total_codified": self._total_codified,
            "pending_buckets": len(self._buckets),
            "mature_uncodified": sum(
                1 for k, b in self._buckets.items()
                if b.is_mature(_MIN_OCCURRENCES) and k not in self._codified_sequences
            ),
        }

    # ─── Private ──────────────────────────────────────────────────────────────

    async def _codify_bucket(self, bucket: _SequenceBucket) -> Procedure | None:
        """Extract a Procedure from one mature bucket via LLM."""
        examples = bucket.successes[:_MAX_EXAMPLES]
        if not examples:
            return None

        prompt = _build_codification_prompt(bucket, examples)

        # Budget check: skip in YELLOW/RED
        if self._optimized:
            assert isinstance(self._llm, OptimizedLLMProvider)
            if not self._llm.should_use_llm("evo.procedure_codifier", estimated_tokens=1200):
                self._logger.info("procedure_codification_skipped_budget")
                return None

        try:
            if self._optimized:
                response = await self._llm.generate(  # type: ignore[call-arg]
                    system_prompt=_SYSTEM_PROMPT,
                    messages=[Message("user", prompt)],
                    max_tokens=1000,
                    temperature=0.3,
                    output_format="json",
                    cache_system="evo.procedure_codifier",
                    cache_method="generate",
                )
            else:
                response = await self._llm.generate(
                    system_prompt=_SYSTEM_PROMPT,
                    messages=[Message("user", prompt)],
                    max_tokens=1000,
                    temperature=0.3,
                    output_format="json",
                )
            raw = _parse_json_safe(response.text)
        except Exception as exc:
            self._logger.error("codification_llm_failed", error=str(exc))
            return None

        procedure = _build_procedure(raw, bucket)
        if procedure is None:
            return None

        if self._memory is not None:
            await self._store_procedure(procedure, bucket)

        self._logger.info(
            "procedure_codified",
            procedure_id=procedure.id,
            name=procedure.name,
            action_sequence=list(bucket.action_types[:6]),
            success_rate=round(bucket.success_rate, 2),
            success_count=bucket.success_count,
        )

        return procedure

    async def _store_procedure(
        self,
        procedure: Procedure,
        bucket: _SequenceBucket,
    ) -> None:
        """
        Persist Procedure as a :Procedure node in Neo4j.

        Writes SOURCED_FROM edges to each source episode so the lineage
        is traceable. Uses MERGE so re-runs are idempotent.
        """
        if self._memory is None:
            return

        try:
            steps_json = json.dumps([s.model_dump() for s in procedure.steps])
            await self._memory.execute_write(
                """
                MERGE (p:Procedure {id: $id})
                SET p.name = $name,
                    p.preconditions = $preconditions,
                    p.steps_json = $steps_json,
                    p.postconditions = $postconditions,
                    p.success_rate = $success_rate,
                    p.source_episodes = $source_episodes,
                    p.usage_count = $usage_count,
                    p.codifier = 'intent_outcome',
                    p.created_at = datetime()
                WITH p
                MATCH (s:Self)
                MERGE (s)-[:HAS_PROCEDURE]->(p)
                """,
                {
                    "id": procedure.id,
                    "name": procedure.name,
                    "preconditions": procedure.preconditions,
                    "steps_json": steps_json,
                    "postconditions": procedure.postconditions,
                    "success_rate": procedure.success_rate,
                    "source_episodes": procedure.source_episodes,
                    "usage_count": procedure.usage_count,
                },
            )

            # Link to source episodes (SOURCED_FROM edges)
            for episode_id in procedure.source_episodes:
                if not episode_id:
                    continue
                try:
                    await self._memory.execute_write(
                        """
                        MATCH (p:Procedure {id: $proc_id})
                        MATCH (e:Episode {id: $ep_id})
                        MERGE (p)-[:SOURCED_FROM]->(e)
                        """,
                        {"proc_id": procedure.id, "ep_id": episode_id},
                    )
                except Exception:
                    # Episode may not exist in graph yet - non-fatal
                    pass

        except Exception as exc:
            self._logger.warning(
                "procedure_store_failed",
                procedure_id=procedure.id,
                error=str(exc),
            )


# ─── Prompt & Parse ───────────────────────────────────────────────────────────


def _build_codification_prompt(
    bucket: _SequenceBucket,
    examples: list[tuple[IntentRecord, OutcomeRecord]],
) -> str:
    action_seq = ", ".join(bucket.action_types[:10]) if bucket.action_types else "unknown"
    success_rate = bucket.success_rate
    total = bucket.total_count

    example_lines = "\n\n".join(
        f"Example {i + 1}:\n"
        f"  Goal: {intent.goal_description[:150]}\n"
        f"  Actions: {', '.join(intent.action_types[:8])}\n"
        f"  Outcome: {outcome.outcome_summary[:150]}\n"
        f"  Economic delta: ${outcome.economic_delta:+.4f}"
        for i, (intent, outcome) in enumerate(examples[:_MAX_EXAMPLES])
    )

    return f"""Analyse these {len(examples)} successful intent executions sharing the same action sequence.

Action sequence (abstract): {action_seq}
Observed {total} times total, {bucket.success_count} successes (success rate: {success_rate:.0%})

EXAMPLES:
{example_lines}

Extract the GENERALISED procedure - the common abstract pattern across all examples.
Preconditions should be situation-level (what must be true before this makes sense).
Postconditions should be outcome-level (what should be true after success).
Steps should match the action sequence but with generalised descriptions.

Respond in JSON:
{{
  "name": "Descriptive name for this procedure (3-7 words)",
  "preconditions": ["What must be true before this procedure applies"],
  "steps": [
    {{
      "action_type": "the executor string",
      "description": "What this step achieves (generalised)",
      "parameters": {{}},
      "expected_duration_ms": 1000
    }}
  ],
  "postconditions": ["What should be true after successful execution"],
  "variations": "Where the examples differed from the common pattern"
}}"""


def _build_procedure(
    raw: dict[str, Any],
    bucket: _SequenceBucket,
) -> Procedure | None:
    """Parse LLM output into a typed Procedure object."""
    try:
        name = str(raw.get("name", "")).strip()
        if not name:
            return None

        preconditions = [str(p) for p in raw.get("preconditions", [])]
        postconditions = [str(p) for p in raw.get("postconditions", [])]

        steps: list[ProcedureStep] = []
        for step_data in raw.get("steps", []):
            if not isinstance(step_data, dict):
                continue
            steps.append(
                ProcedureStep(
                    action_type=str(step_data.get("action_type", "")),
                    description=str(step_data.get("description", "")),
                    parameters=step_data.get("parameters") or {},
                    expected_duration_ms=int(step_data.get("expected_duration_ms", 1000)),
                )
            )

        if not steps:
            return None

        # Source episodes: episode IDs from successful runs (may be empty string if not tracked)
        source_episodes = [
            intent.episode_id
            for intent, _ in bucket.successes[:10]
            if intent.episode_id
        ]

        return Procedure(
            name=name,
            preconditions=preconditions,
            steps=steps,
            postconditions=postconditions,
            success_rate=bucket.success_rate,
            source_episodes=source_episodes,
            usage_count=0,
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("procedure_parse_failed", error=str(exc))
        return None


def _parse_json_safe(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        return {}
