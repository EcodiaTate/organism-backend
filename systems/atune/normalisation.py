"""
Atune — Input Normalisation.

Converts raw input from any :class:`InputChannel` into the standard
:class:`Percept` format and produces a :class:`NormalisedPercept` with
LLM-derived sentiment, entities, and EIS-ready threat annotation slots.

Every channel has a dedicated normaliser that knows how to extract textual
content, assign a default modality, and set a salience hint.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

import structlog

from primitives.common import Modality, SourceDescriptor, SystemID, new_id, utc_now
from primitives.percept import (
    Content,
    Percept,
    Provenance,
    TransformRecord,
)

from .helpers import compute_hash_chain, hash_content
from .types import (
    EntityCandidate,
    InputChannel,
    NormalisedPercept,
    RawInput,
    RelationCandidate,
    SentimentAnalysis,
)

logger = structlog.get_logger("systems.atune.normalisation")


# ---------------------------------------------------------------------------
# LLM client protocol
# ---------------------------------------------------------------------------


class NormalisationLLMClient(Protocol):
    """Minimal interface for the LLM client used during normalisation."""

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
# LLM prompt templates
# ---------------------------------------------------------------------------

_SENTIMENT_PROMPT = """\
Analyse the sentiment of the following text and return a JSON object.

TEXT:
{text}

Return exactly this JSON structure:
{{
  "valence": <float from -1.0 (very negative) to 1.0 (very positive)>,
  "arousal": <float from 0.0 (calm) to 1.0 (highly activated)>,
  "dominant_emotion": "<single word: joy, sadness, anger, fear, surprise, disgust, trust, anticipation, or neutral>",
  "summary": "<one sentence describing the emotional tone>"
}}

Be precise. Base your analysis only on the text provided.
"""

_ENTITY_EXTRACTION_PROMPT = """\
Extract entities and relationships from the following text.

TEXT:
{text}

For each entity, provide:
- name: canonical name
- type: one of [person, place, organisation, concept, object, event, emotion, value]
- description: brief description in context
- confidence: 0.0 to 1.0

For each relationship between entities, provide:
- from_entity: source entity name
- to_entity: target entity name
- type: relationship type (e.g., works_for, located_in, caused_by, part_of)
- strength: 0.0 to 1.0

Return exactly this JSON structure:
{{
  "entities": [{{...}}],
  "relations": [{{...}}]
}}

Only extract entities and relations clearly present in the text.
"""


# ---------------------------------------------------------------------------
# LLM analysis helpers
# ---------------------------------------------------------------------------


async def _analyse_sentiment_llm(
    text: str,
    llm_client: NormalisationLLMClient,
) -> SentimentAnalysis:
    """Call the LLM to produce a structured sentiment analysis."""
    if not text or len(text.strip()) < 3:
        return SentimentAnalysis()

    prompt = _SENTIMENT_PROMPT.format(text=text)

    try:
        from clients.llm import Message

        response = await llm_client.generate(
            system_prompt="You are a precise sentiment analysis engine. Always respond with valid JSON only.",
            messages=[Message(role="user", content=prompt)],
            max_tokens=300,
            temperature=0.1,
            output_format="json",
            cache_system="atune.normalisation",
            cache_method="sentiment",
        )
        raw_text = response.text or ""
        json_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
        result: dict[str, Any] = json.loads(json_text)
    except Exception:
        logger.warning("sentiment_llm_failed", text_length=len(text), exc_info=True)
        return SentimentAnalysis()

    return SentimentAnalysis(
        valence=max(-1.0, min(1.0, float(result.get("valence", 0.0)))),
        arousal=max(0.0, min(1.0, float(result.get("arousal", 0.0)))),
        dominant_emotion=str(result.get("dominant_emotion", "")),
        summary=str(result.get("summary", "")),
    )


async def _extract_entities_llm(
    text: str,
    llm_client: NormalisationLLMClient,
) -> tuple[list[EntityCandidate], list[RelationCandidate]]:
    """Call the LLM to extract entities and relations from text."""
    if not text or len(text.strip()) < 5:
        return [], []

    prompt = _ENTITY_EXTRACTION_PROMPT.format(text=text)

    try:
        from clients.llm import Message

        response = await llm_client.generate(
            system_prompt="You are a precise entity extraction engine. Always respond with valid JSON only.",
            messages=[Message(role="user", content=prompt)],
            max_tokens=2000,
            temperature=0.2,
            output_format="json",
            cache_system="atune.normalisation",
            cache_method="entity_extraction",
        )
        raw_text = response.text or ""
        json_text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.MULTILINE)
        result: dict[str, Any] = json.loads(json_text)
    except Exception:
        logger.warning("entity_extraction_llm_failed", text_length=len(text), exc_info=True)
        return [], []

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

    return entities, relations


# ---------------------------------------------------------------------------
# Per-channel normaliser definitions
# ---------------------------------------------------------------------------


class ChannelNormaliser:
    """Base normaliser — subclass per channel for custom logic."""

    modality: str = "text"
    default_salience_hint: float = 0.5

    def extract_text(self, raw: RawInput) -> str:
        """Return the plain-text representation of the raw input."""
        if isinstance(raw.data, bytes):
            return raw.data.decode("utf-8", errors="replace")
        return raw.data


class TextChatNormaliser(ChannelNormaliser):
    modality = "text"
    default_salience_hint = 0.6  # User messages are inherently important


class VoiceNormaliser(ChannelNormaliser):
    modality = "audio_transcript"
    default_salience_hint = 0.6


class GestureNormaliser(ChannelNormaliser):
    modality = "interaction"
    default_salience_hint = 0.3


class SensorIoTNormaliser(ChannelNormaliser):
    modality = "sensor"
    default_salience_hint = 0.2


class CalendarNormaliser(ChannelNormaliser):
    modality = "temporal"
    default_salience_hint = 0.4


class ExternalAPINormaliser(ChannelNormaliser):
    modality = "api"
    default_salience_hint = 0.3


class SystemEventNormaliser(ChannelNormaliser):
    modality = "internal"
    default_salience_hint = 0.4


class MemoryBubbleNormaliser(ChannelNormaliser):
    modality = "internal"
    default_salience_hint = 0.5


class AffectShiftNormaliser(ChannelNormaliser):
    modality = "internal"
    default_salience_hint = 0.4


class EvoInsightNormaliser(ChannelNormaliser):
    modality = "internal"
    default_salience_hint = 0.5


class FederationMsgNormaliser(ChannelNormaliser):
    modality = "federation"
    default_salience_hint = 0.5


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CHANNEL_NORMALISERS: dict[InputChannel, ChannelNormaliser] = {
    InputChannel.TEXT_CHAT: TextChatNormaliser(),
    InputChannel.VOICE: VoiceNormaliser(),
    InputChannel.GESTURE: GestureNormaliser(),
    InputChannel.SENSOR_IOT: SensorIoTNormaliser(),
    InputChannel.CALENDAR: CalendarNormaliser(),
    InputChannel.EXTERNAL_API: ExternalAPINormaliser(),
    InputChannel.SYSTEM_EVENT: SystemEventNormaliser(),
    InputChannel.MEMORY_BUBBLE: MemoryBubbleNormaliser(),
    InputChannel.AFFECT_SHIFT: AffectShiftNormaliser(),
    InputChannel.EVO_INSIGHT: EvoInsightNormaliser(),
    InputChannel.FEDERATION_MSG: FederationMsgNormaliser(),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def _build_percept(
    raw_input: RawInput,
    channel: InputChannel,
    embed_fn: object,
    normaliser: ChannelNormaliser,
) -> tuple[Percept, str]:
    """Shared Percept construction used by both ``normalise`` and ``normalise_enriched``."""
    text = normaliser.extract_text(raw_input)
    embedding: list[float] = await embed_fn(text)  # type: ignore[operator]

    now = utc_now()
    raw_hash = hash_content(raw_input.data if isinstance(raw_input.data, str) else raw_input.data)
    text_hash = hash_content(text)

    provenance = Provenance(
        chain=[
            TransformRecord(
                step="normalise",
                system="atune",
                timestamp=now,
                input_hash=raw_hash,
                output_hash=text_hash,
            )
        ],
        integrity=compute_hash_chain(
            raw_input.data if isinstance(raw_input.data, str) else raw_input.data,
            text,
        ),
    )

    try:
        source_system = SystemID(channel.value)
    except ValueError:
        source_system = SystemID.ATUNE

    try:
        source_modality = Modality(normaliser.modality)
    except ValueError:
        source_modality = Modality.TEXT

    percept = Percept(
        id=new_id(),
        timestamp=now,
        source=SourceDescriptor(
            system=source_system,
            channel=raw_input.channel_id or channel.value,
            modality=source_modality,
        ),
        content=Content(
            raw=raw_input.data if isinstance(raw_input.data, str) else raw_input.data.decode("utf-8", errors="replace"),
            parsed=text,
            embedding=embedding,
        ),
        provenance=provenance,
        salience_hint=normaliser.default_salience_hint,
        metadata=raw_input.metadata,
    )

    return percept, text


def _resolve_normaliser(channel: InputChannel) -> ChannelNormaliser:
    """Look up the normaliser for *channel*, falling back to the base class."""
    normaliser = CHANNEL_NORMALISERS.get(channel)
    if normaliser is None:
        logger.warning("unknown_channel", channel=channel.value)
        normaliser = ChannelNormaliser()
    return normaliser


async def normalise(
    raw_input: RawInput,
    channel: InputChannel,
    embed_fn: object,  # async callable (str) -> list[float]
) -> Percept:
    """
    Convert any raw input into a standard :class:`Percept`.

    This is the backward-compatible entry point used by
    :class:`AtuneService`.  It returns a bare ``Percept`` with no
    LLM enrichment.  Use :func:`normalise_enriched` for the full
    pipeline that also returns a :class:`NormalisedPercept`.

    Parameters
    ----------
    raw_input:
        The raw data arriving on *channel*.
    channel:
        Which input channel produced this data.
    embed_fn:
        Async callable that returns a 768-dim embedding for a text string.

    Returns
    -------
    Percept
        Normalised percept ready for Fovea prediction error evaluation.
    """
    normaliser = _resolve_normaliser(channel)
    percept, text = await _build_percept(raw_input, channel, embed_fn, normaliser)

    logger.debug(
        "percept_normalised",
        percept_id=percept.id,
        channel=channel.value,
        text_length=len(text),
    )

    return percept


async def normalise_enriched(
    raw_input: RawInput,
    channel: InputChannel,
    embed_fn: object,  # async callable (str) -> list[float]
    llm_client: NormalisationLLMClient,
) -> tuple[Percept, NormalisedPercept]:
    """
    Convert raw input into a :class:`Percept` **and** an Atune-specific
    :class:`NormalisedPercept` with LLM-derived enrichment.

    Runs LLM-based sentiment analysis and entity extraction.  The
    ``threat_annotations`` list is initialised empty — EIS populates it
    before Fovea prediction error evaluation.

    Parameters
    ----------
    raw_input:
        The raw data arriving on *channel*.
    channel:
        Which input channel produced this data.
    embed_fn:
        Async callable that returns a 768-dim embedding for a text string.
    llm_client:
        LLM client for sentiment and entity extraction.

    Returns
    -------
    tuple[Percept, NormalisedPercept]
        The normalised percept (shared primitive) and its Atune-local
        enrichment with ``threat_annotations`` ready for EIS population.
    """
    normaliser = _resolve_normaliser(channel)
    percept, text = await _build_percept(raw_input, channel, embed_fn, normaliser)

    # LLM-based enrichment
    sentiment = SentimentAnalysis()
    entities: list[EntityCandidate] = []
    relations: list[RelationCandidate] = []

    if text and len(text.strip()) >= 3:
        sentiment = await _analyse_sentiment_llm(text, llm_client)
        entities, relations = await _extract_entities_llm(text, llm_client)

    # Build evidence tags from high-confidence extracted entities
    evidence_tags = [e.name for e in entities if e.confidence >= 0.6]

    # Confidence score: blend of embedding availability and sentiment clarity
    has_embedding = 1.0 if percept.content.embedding else 0.0
    sentiment_clarity = abs(sentiment.valence) if sentiment.valence != 0.0 else 0.0
    confidence_score = max(0.1, min(1.0, 0.4 * has_embedding + 0.3 * sentiment_clarity + 0.3))

    normalised = NormalisedPercept(
        percept_id=percept.id,
        threat_annotations=[],  # Empty — EIS fast-path populates this
        evidence_tags=evidence_tags,
        confidence_score=confidence_score,
        sentiment=sentiment,
        entities=entities,
        relations=relations,
    )

    logger.debug(
        "percept_normalised",
        percept_id=percept.id,
        channel=channel.value,
        text_length=len(text),
        entity_count=len(entities),
        sentiment_valence=sentiment.valence,
        confidence=confidence_score,
    )

    return percept, normalised
