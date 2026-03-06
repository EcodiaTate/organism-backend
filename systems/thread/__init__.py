"""
EcodiaOS — Thread: The Narrative Identity System

Thread is the organism's autobiographical self — the Ricoeurian ipse that
maintains continuity through change. It manages:
  - Identity schemas (who I am, what I value, what I've learned)
  - Constitutional commitments (the four drives as lived promises)
  - 29-dimensional identity fingerprints (for drift detection)
  - Life story integration (periodic autobiographical synthesis)
  - Narrative chapters (life phases the organism recognises in itself)

Public API:
  ThreadService
  BaseNarrativeSynthesizer, BaseChapterDetector  (ABCs for hot-reload)
  CommitmentType, CommitmentStrength, SchemaStatus
  IdentitySchema, Commitment, IdentityFingerprint, NarrativeChapter
"""

from systems.thread.processors import (
    BaseChapterDetector,
    BaseNarrativeSynthesizer,
)
from systems.thread.service import ThreadService
from systems.thread.types import (
    Commitment,
    CommitmentStrength,
    CommitmentType,
    IdentityFingerprint,
    IdentitySchema,
    LifeStorySnapshot,
    NarrativeChapter,
    NarrativeIdentitySummary,
    SchemaStatus,
    ThreadHealthSnapshot,
)

__all__ = [
    "ThreadService",
    "BaseChapterDetector",
    "BaseNarrativeSynthesizer",
    "Commitment",
    "CommitmentStrength",
    "CommitmentType",
    "IdentityFingerprint",
    "IdentitySchema",
    "LifeStorySnapshot",
    "NarrativeChapter",
    "NarrativeIdentitySummary",
    "SchemaStatus",
    "ThreadHealthSnapshot",
]
