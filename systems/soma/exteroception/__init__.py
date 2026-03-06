"""
EcodiaOS — Exteroception (Cross-Modal Synesthesia)

Maps arbitrary external data streams into the organism's interoceptive
dimensions. The organism *feels* the weather of the external world —
market volatility becomes arousal, negative sentiment becomes valence
pressure, uncertainty becomes temporal compression.

Public API:
  ExteroceptionService   — top-level coordinator
  ExteroceptiveMappingEngine — the translation math
  MarketDataAdapter      — crypto/equity data ingestion
  NewsSentimentAdapter   — news sentiment firehose
  ExteroceptiveReading   — a normalised external datum
  ExteroceptivePressure  — aggregated pressure for Soma injection
"""

from systems.soma.exteroception.adapters import (
    BaseExteroceptiveAdapter,
    MarketDataAdapter,
    NewsSentimentAdapter,
)
from systems.soma.exteroception.mapping_engine import (
    ExteroceptiveMappingEngine,
)
from systems.soma.exteroception.service import ExteroceptionService
from systems.soma.exteroception.types import (
    DEFAULT_MODALITY_MAPPINGS,
    ExteroceptiveModality,
    ExteroceptivePressure,
    ExteroceptiveReading,
    ModalityMapping,
    ReadingQuality,
)

__all__ = [
    "BaseExteroceptiveAdapter",
    "DEFAULT_MODALITY_MAPPINGS",
    "ExteroceptiveMappingEngine",
    "ExteroceptiveModality",
    "ExteroceptivePressure",
    "ExteroceptiveReading",
    "ExteroceptionService",
    "MarketDataAdapter",
    "ModalityMapping",
    "NewsSentimentAdapter",
    "ReadingQuality",
]
