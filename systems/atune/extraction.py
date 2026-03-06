"""
Atune — Entity Extraction Pipeline.

When a Percept is stored in Memory, Atune triggers entity extraction.
This is the bridge between raw experience and structured knowledge.

Extraction is performed asynchronously and does **not** block the
workspace cycle (budget: ≤2 000 ms, performed outside the theta rhythm).
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Protocol

import structlog

from .types import EntityCandidate, ExtractionResult, RelationCandidate

if TYPE_CHECKING:
    from primitives.percept import Percept

logger = structlog.get_logger("systems.atune.extraction")

# ---------------------------------------------------------------------------
# LLM client protocol
# ---------------------------------------------------------------------------


class ExtractionLLMClient(Protocol):
    """Minimal interface for the LLM client used by entity extraction."""

    async def generate(
        self,
        system_prompt: str,
        messages: list[Any],
        max_tokens: int = ...,
        temperature: float = ...,
        output_format: str | None = ...,
        *,
        cache_system: str = ...,
        cache_method: str = ...,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
Extract entities and relationships from the following content.

CONTENT:
{content}

SOURCE: {source_system} ({modality})
TIMESTAMP: {timestamp}

For each entity, provide:
- name: canonical name
- type: one of [person, place, organisation, concept, object, event, emotion, value]
- description: brief description in context
- confidence: 0.0 to 1.0

For each relationship between entities, provide:
- from_entity: source entity name
- to_entity: target entity name
- type: relationship type (e.g., works_for, located_in, caused_by, part_of, etc.)
- strength: 0.0 to 1.0
- temporal: is this relationship time-bounded? If so, from when to when?

Respond in JSON format with keys "entities" and "relations".
Be precise. Only extract entities and relations that are clearly present.
Prefer specificity over coverage — better to miss an entity than fabricate one.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def extract_entities_and_relations(
    percept: Percept,
    llm_client: ExtractionLLMClient,
) -> ExtractionResult:
    """
    Use an LLM to extract structured entities and relations from *percept*.

    This function is designed to be run as a background task (``asyncio.create_task``)
    after a Percept wins workspace broadcast.  It must NOT block the cognitive cycle.

    Returns
    -------
    ExtractionResult
        Parsed entities and relations, or an empty result on failure.
    """
    text = percept.content.parsed if isinstance(percept.content.parsed, str) else percept.content.raw
    if not text or (isinstance(text, str) and len(text.strip()) < 5):
        return ExtractionResult(source_percept_id=percept.id)

    prompt = _EXTRACTION_PROMPT.format(
        content=text,
        source_system=percept.source.system,
        modality=percept.source.modality,
        timestamp=percept.timestamp.isoformat() if percept.timestamp else "unknown",
    )

    try:
        from clients.llm import Message

        response = await llm_client.generate(
            system_prompt="You are a precise entity extraction engine. Always respond with valid JSON only.",
            messages=[Message(role="user", content=prompt)],
            max_tokens=2000,
            temperature=0.2,
            output_format="json",
            cache_system="atune.entity_extraction",
            cache_method="extract",
        )
        raw_text = response.text or ""
        # Strip markdown code fences if present
        json_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
        result: dict[str, Any] = json.loads(json_text)
    except Exception:
        logger.warning("extraction_llm_failed", percept_id=percept.id, exc_info=True)
        return ExtractionResult(source_percept_id=percept.id)

    # Parse entities
    entities: list[EntityCandidate] = []
    for raw in result.get("entities", []):
        try:
            entities.append(EntityCandidate(
                name=raw["name"],
                type=raw.get("type", "concept"),
                description=raw.get("description", ""),
                confidence=float(raw.get("confidence", 0.5)),
            ))
        except (KeyError, ValueError, TypeError):
            continue

    # Parse relations
    relations: list[RelationCandidate] = []
    for raw in result.get("relations", []):
        try:
            relations.append(RelationCandidate(
                from_entity=raw["from_entity"],
                to_entity=raw["to_entity"],
                type=raw.get("type", "related_to"),
                strength=float(raw.get("strength", 0.5)),
                temporal=raw.get("temporal"),
            ))
        except (KeyError, ValueError, TypeError):
            continue

    logger.info(
        "entities_extracted",
        percept_id=percept.id,
        entity_count=len(entities),
        relation_count=len(relations),
    )

    return ExtractionResult(
        entities=entities,
        relations=relations,
        source_percept_id=percept.id,
    )
