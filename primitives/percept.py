"""
EcodiaOS — Percept Primitive

The fundamental unit of incoming information.
Everything that enters EOS is normalised into a Percept.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import Field

from primitives.common import (
    EOSBaseModel,
    Identified,
    Modality,
    SourceDescriptor,
    SystemID,
    Timestamped,
    utc_now,
)


class Content(EOSBaseModel):
    """The content of a percept."""

    raw: str = ""                             # Original content, preserved exactly
    parsed: str | dict[str, Any] = Field(default_factory=dict)  # Structured extraction; str when text, dict for structured channels
    embedding: list[float] | None = None      # Dense vector (computed async)
    token_embeddings: list[list[float]] | None = None  # Per-token embeddings (cached)


class TransformRecord(EOSBaseModel):
    """A single step in a percept's transformation chain."""

    step: str                        # e.g. "normalise", "embed", "salience_score"
    system: str                      # System that performed the step
    timestamp: datetime = Field(default_factory=utc_now)
    input_hash: str = ""             # Hash of the input at this step
    output_hash: str = ""            # Hash of the output at this step


class Provenance(EOSBaseModel):
    """Chain of custody for a percept — where it came from and how it was transformed."""

    chain: list[TransformRecord] = Field(default_factory=list)
    integrity: str | None = None     # Hash chain integrity check


class Percept(Identified, Timestamped):
    """
    The fundamental unit of incoming information.

    A user message, a sensor reading, an internal system notification,
    a federation event — all normalised to this shape.
    """

    timestamp: datetime = Field(default_factory=utc_now)
    source: SourceDescriptor
    content: Content
    provenance: Provenance = Field(default_factory=Provenance)
    salience_hint: float | None = None    # Optional pre-scoring from source (0.0-1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_user_message(cls, text: str, channel: str = "text_chat") -> Percept:
        """Convenience constructor for user text input."""
        return cls(
            source=SourceDescriptor(
                system=SystemID.API,
                channel=channel,
                modality=Modality.TEXT,
            ),
            content=Content(raw=text),
            metadata={"type": "user_message"},
        )

    @classmethod
    def from_internal(
        cls, system: SystemID, content: str, metadata: dict[str, Any] | None = None
    ) -> Percept:
        """Convenience constructor for internal system events."""
        return cls(
            source=SourceDescriptor(
                system=system,
                channel="internal",
                modality=Modality.INTERNAL,
            ),
            content=Content(raw=content),
            metadata=metadata or {},
        )
