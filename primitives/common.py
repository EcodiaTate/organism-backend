"""
EcodiaOS - Common Primitives

Shared enums, base classes, and utilities used across all systems.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime

from pydantic import BaseModel, Field
from ulid import ULID


def new_id() -> str:
    """Generate a new ULID string. Time-sortable, globally unique."""
    return str(ULID())


def utc_now() -> datetime:
    """Current UTC time, timezone-aware."""
    return datetime.now(UTC)


# ─── Enums ────────────────────────────────────────────────────────


class SystemID(enum.StrEnum):
    MEMORY = "memory"
    EQUOR = "equor"
    ATUNE = "atune"
    VOXIS = "voxis"
    NOVA = "nova"
    AXON = "axon"
    EVO = "evo"
    SIMULA = "simula"
    SYNAPSE = "synapse"
    THYMOS = "thymos"
    FEDERATION = "federation"
    ONEIROS = "oneiros"
    THREAD = "thread"
    SOMA = "soma"
    OIKOS = "oikos"
    SKIA = "skia"
    FOVEA = "fovea"
    LOGOS = "logos"
    KAIROS = "kairos"
    NEXUS = "nexus"
    TELOS = "telos"
    BENCHMARKS = "benchmarks"
    EIS = "eis"
    IDENTITY = "identity"
    SACM = "sacm"
    MITOSIS = "mitosis"
    PHANTOM = "phantom"
    ALIVE = "alive"
    API = "api"


class Modality(enum.StrEnum):
    TEXT = "text"
    VOICE = "voice"
    SENSOR = "sensor"
    INTERNAL = "internal"
    FEDERATION = "federation"


class EntityType(enum.StrEnum):
    PERSON = "person"
    PLACE = "place"
    ORGANISATION = "organisation"
    CONCEPT = "concept"
    OBJECT = "object"
    EVENT = "event"
    EMOTION = "emotion"
    VALUE = "value"


class ConsolidationLevel(int, enum.Enum):
    RAW = 0
    EXTRACTED = 1
    INTEGRATED = 2
    DEEP = 3


class AutonomyLevel(int, enum.Enum):
    ADVISOR = 1     # Recommends, human decides
    PARTNER = 2     # Acts with human approval
    STEWARD = 3     # Acts autonomously within bounds


class Verdict(enum.StrEnum):
    APPROVED = "approved"
    MODIFIED = "modified"
    BLOCKED = "blocked"
    DEFERRED = "deferred"
    SUSPENDED_AWAITING_HUMAN = "suspended_awaiting_human"


class HealthStatus(enum.StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


# ─── Base Models ──────────────────────────────────────────────────


class EOSBaseModel(BaseModel):
    """Base model for all EOS primitives. Uses ULID IDs and UTC timestamps."""

    model_config = {"populate_by_name": True, "from_attributes": True}


class Timestamped(EOSBaseModel):
    """Mixin for models with creation and update timestamps."""

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class Identified(EOSBaseModel):
    """Mixin for models with ULID IDs."""

    id: str = Field(default_factory=new_id)


class SourceDescriptor(EOSBaseModel):
    """Describes where a piece of information came from."""

    system: SystemID
    channel: str = ""       # e.g., "text_chat", "webhook:github", "sensor:temperature"
    modality: Modality = Modality.TEXT


class DriveAlignmentVector(EOSBaseModel):
    """Alignment scores for the four constitutional drives. Range: -1.0 to 1.0."""

    coherence: float = 0.0
    care: float = 0.0
    growth: float = 0.0
    honesty: float = 0.0

    @property
    def composite(self) -> float:
        """Weighted average alignment. Higher = more aligned."""
        return (self.coherence + self.care + self.growth + self.honesty) / 4.0

    @property
    def min_score(self) -> float:
        """Lowest individual drive alignment."""
        return min(self.coherence, self.care, self.growth, self.honesty)


class ResourceBudget(EOSBaseModel):
    """Resource constraints for an action or intent."""

    compute_ms: int = 5000
    memory_mb: int = 256
    api_calls: int = 1
    llm_tokens: int = 2000


class SalienceVector(EOSBaseModel):
    """Multi-head salience scores."""

    scores: dict[str, float] = Field(default_factory=dict)
    composite: float = 0.0

    @classmethod
    def from_heads(cls, **kwargs: float) -> SalienceVector:
        """Create from named head scores with auto-computed composite."""
        scores = {k: v for k, v in kwargs.items()}
        composite = sum(scores.values()) / max(len(scores), 1)
        return cls(scores=scores, composite=composite)
