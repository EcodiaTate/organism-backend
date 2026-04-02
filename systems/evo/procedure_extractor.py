"""
EcodiaOS - Evo Procedure Extractor

Converts mature action-sequence patterns into reusable Procedures stored
in the Memory graph. Procedures are the "habits" Nova's fast path can use.

Process (spec Section VI):
  1. A SequenceDetector pattern reaches min_occurrences
  2. Example episodes are retrieved from Memory
  3. LLM extracts the common structure: preconditions, steps, postconditions
  4. A Procedure node is stored in the Memory graph
  5. Nova can discover and reuse it via procedural memory retrieval

Performance: procedure extraction ≤5s per procedure (spec Section X).
Max 3 new procedures per consolidation cycle (VELOCITY_LIMITS).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from clients.llm import LLMProvider, Message
from clients.optimized_llm import OptimizedLLMProvider
from primitives.common import new_id
from primitives.memory_trace import Episode
from systems.evo.types import (
    VELOCITY_LIMITS,
    PatternCandidate,
    PatternType,
    Procedure,
    ProcedureStep,
)

if TYPE_CHECKING:
    from systems.memory.service import MemoryService

logger = structlog.get_logger()

import os as _os_pe
_MAX_PER_CYCLE: int = VELOCITY_LIMITS["max_new_procedures_per_cycle"]
_MIN_OCCURRENCES: int = int(_os_pe.getenv("EVO_PROCEDURE_MIN_OCCURRENCES", "3"))
_MAX_EXAMPLES: int = int(_os_pe.getenv("EVO_PROCEDURE_MAX_EXAMPLES", "0")) or 999  # 0 = unlimited

_SYSTEM_PROMPT = (
    "Extract generalised, reusable procedures from concrete examples. "
    "Capture the essential pattern, not the specific details. Respond as JSON."
)


class ProcedureExtractor:
    """
    Extracts reusable Procedures from mature action-sequence patterns.

    Dependencies:
      llm    - for LLM-based extraction
      memory - for episode retrieval and procedure storage
    """

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemoryService | None = None,
    ) -> None:
        self._llm = llm
        self._memory = memory
        self._logger = logger.bind(system="evo.procedure_extractor")
        self._optimized = isinstance(llm, OptimizedLLMProvider)
        self._extracted_this_cycle: int = 0
        self._total_extracted: int = 0

    def begin_cycle(self) -> None:
        """Reset per-cycle counter."""
        self._extracted_this_cycle = 0

    async def extract_procedure(
        self,
        pattern: PatternCandidate,
    ) -> Procedure | None:
        """
        Attempt to extract a Procedure from an action-sequence pattern.

        Returns None if:
          - Pattern is not an action sequence
          - Count below threshold
          - Cycle limit reached
          - Memory unavailable for episode retrieval
          - LLM extraction fails
        """
        if pattern.type != PatternType.ACTION_SEQUENCE:
            return None
        if pattern.count < _MIN_OCCURRENCES:
            return None
        if self._extracted_this_cycle >= _MAX_PER_CYCLE:
            self._logger.info(
                "procedure_cycle_limit_reached",
                limit=_MAX_PER_CYCLE,
            )
            return None

        # Retrieve example episodes for LLM analysis
        examples = await self._fetch_example_episodes(pattern.examples[:_MAX_EXAMPLES])
        if not examples:
            self._logger.warning(
                "procedure_no_examples",
                sequence_hash=pattern.metadata.get("sequence_hash", ""),
            )
            return None

        # Build extraction prompt
        prompt = _build_extraction_prompt(
            elements=pattern.elements,
            count=pattern.count,
            examples=examples,
        )

        # Budget check: skip procedure extraction in YELLOW/RED (low priority)
        if self._optimized:
            assert isinstance(self._llm, OptimizedLLMProvider)
            if not await self._llm.should_use_llm("evo.procedure", estimated_tokens=1000):
                self._logger.info("procedure_extraction_skipped_budget")
                return None

        try:
            if self._optimized:
                response = await self._llm.generate(  # type: ignore[call-arg]
                    system_prompt=_SYSTEM_PROMPT,
                    messages=[Message("user", prompt)],
                    max_tokens=1000,
                    temperature=0.3,
                    output_format="json",
                    cache_system="evo.procedure",
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
            self._logger.error("procedure_extraction_llm_failed", error=str(exc))
            return None

        procedure = _build_procedure(raw, pattern)
        if procedure is None:
            return None

        # Store in Memory graph
        if self._memory is not None:
            await self._store_procedure(procedure)

        self._extracted_this_cycle += 1
        self._total_extracted += 1

        self._logger.info(
            "procedure_extracted",
            procedure_id=procedure.id,
            name=procedure.name,
            steps=len(procedure.steps),
            source_count=len(procedure.source_episodes),
        )

        return procedure

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_extracted": self._total_extracted,
            "extracted_this_cycle": self._extracted_this_cycle,
        }

    # ─── Private ──────────────────────────────────────────────────────────────

    async def _fetch_example_episodes(
        self,
        episode_ids: list[str],
    ) -> list[Episode]:
        """Retrieve episodes from Memory for LLM analysis."""
        if not episode_ids or self._memory is None:
            return []

        episodes: list[Episode] = []
        try:
            for ep_id in episode_ids[:_MAX_EXAMPLES]:
                results = await self._memory.execute_read(
                    """
                    MATCH (e:Episode {id: $id})
                    RETURN e
                    """,
                    {"id": ep_id},
                )
                if results:
                    data = results[0]["e"]
                    episodes.append(_dict_to_episode(data))
        except Exception as exc:
            self._logger.warning("episode_fetch_failed", error=str(exc))

        return episodes

    async def _store_procedure(self, procedure: Procedure) -> None:
        """Persist the Procedure to the Memory graph as a :Procedure node."""
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
        except Exception as exc:
            self._logger.warning(
                "procedure_store_failed",
                procedure_id=procedure.id,
                error=str(exc),
            )


# ─── Prompt & Parse ───────────────────────────────────────────────────────────


def _build_extraction_prompt(
    elements: list[str],
    count: int,
    examples: list[Episode],
) -> str:
    example_lines = "\n\n".join(
        f"Example {i + 1}:\n"
        f"  Source: {ep.source}\n"
        f"  Content: {(ep.raw_content or ep.summary)[:200]}\n"
        f"  Affect: valence={ep.affect_valence:.2f}, arousal={ep.affect_arousal:.2f}\n"
        f"  Salience: {ep.salience_composite:.2f}"
        for i, ep in enumerate(examples[:8])
    )
    action_sequence = ", ".join(elements[:10]) if elements else "unknown"

    return f"""Analyse these {len(examples)} successful action sequences (pattern observed {count} times).
Detected action types: {action_sequence}

EXAMPLES:
{example_lines}

Extract the GENERALISED procedure - the common abstract pattern, not the specific details.

Respond in JSON:
{{
  "name": "Descriptive name for this procedure (3-7 words)",
  "preconditions": ["What must be true before this procedure applies"],
  "steps": [
    {{
      "action_type": "the action type string",
      "description": "What this step does (generalised, not specific)",
      "parameters": {{}},
      "expected_duration_ms": 1000
    }}
  ],
  "postconditions": ["What should be true after successful execution"],
  "variations": "Where the examples differed from the common pattern"
}}"""


def _build_procedure(raw: dict[str, Any], pattern: PatternCandidate) -> Procedure | None:
    """Parse LLM output into a Procedure object."""
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
                    parameters=step_data.get("parameters", {}),
                    expected_duration_ms=int(step_data.get("expected_duration_ms", 1000)),
                )
            )

        if not steps:
            return None

        return Procedure(
            name=name,
            preconditions=preconditions,
            steps=steps,
            postconditions=postconditions,
            success_rate=1.0,
            source_episodes=pattern.examples[:10],
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


def _dict_to_episode(data: dict[str, Any]) -> Episode:
    """Convert a Neo4j node dict to an Episode object."""

    from dateutil.parser import parse as parse_dt

    from primitives.common import utc_now as _utc_now

    def _safe_dt(v: Any) -> Any:
        if v is None:
            return _utc_now()
        if isinstance(v, str):
            try:
                return parse_dt(v)
            except Exception:
                return _utc_now()
        return v

    return Episode(
        id=str(data.get("id", new_id())),
        event_time=_safe_dt(data.get("event_time")),
        ingestion_time=_safe_dt(data.get("ingestion_time")),
        source=str(data.get("source", "")),
        modality=str(data.get("modality", "text")),
        raw_content=str(data.get("raw_content", "")),
        summary=str(data.get("summary", "")),
        salience_composite=float(data.get("salience_composite", 0.0)),
        salience_scores=json.loads(data.get("salience_scores_json", "{}")) if isinstance(data.get("salience_scores_json"), str) else data.get("salience_scores_json") or {},
        affect_valence=float(data.get("affect_valence", 0.0)),
        affect_arousal=float(data.get("affect_arousal", 0.0)),
    )
