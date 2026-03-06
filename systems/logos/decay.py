"""
EcodiaOS -- Logos: Entropic Decay Engine

Thermodynamic-inspired knowledge decay.  Knowledge that is not
reinforced decays -- compressible patterns are extracted and what
remains is entropy: low-MDL residue with no generative power.

Three decay types:
  1. ACCESS       -- unaccessed items lose coherence (exponential)
  2. COMPRESSION  -- low MDL score items targeted for distillation/eviction
  3. CONTRADICTION-- items contradicted by new evidence decay faster
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from primitives.common import utc_now
from systems.logos.types import (
    DecayReport,
    KnowledgeItemType,
    MDLScore,
)

if TYPE_CHECKING:
    from systems.logos.cascade import CompressionCascade
    from systems.logos.mdl import MDLEstimator

logger = structlog.get_logger("logos.decay")


# --- Memory Store Protocol ----------------------------------------


@runtime_checkable
class ScoredItem(Protocol):
    """Minimal interface for a memory item that can be scored and decayed."""

    @property
    def id(self) -> str: ...

    @property
    def item_type(self) -> str: ...

    @property
    def content(self) -> dict[str, Any]: ...

    @property
    def description_length(self) -> float: ...

    @property
    def last_accessed(self) -> datetime: ...

    @property
    def access_count(self) -> int: ...

    @property
    def contradiction_count(self) -> int: ...


@runtime_checkable
class MemoryStoreProtocol(Protocol):
    """Protocol for the memory store that the decay engine operates on."""

    async def get_all_scored_items(self) -> list[ScoredItem]: ...

    async def evict(self, item_id: str) -> None: ...

    async def replace(self, item_id: str, replacement: Any) -> None: ...

    async def mark_anchor(self, item_id: str) -> None: ...

    async def reinforce(self, item_id: str, factor: float) -> None: ...


class EntropicDecayEngine:
    """
    Three decay types driven by the survival_score from MDLScore.

    Decision tree (per item):
      survival_score < 0.1 AND compression_ratio < 1.0
          -> evict (explains less than it costs)
      survival_score < 0.1 AND compression_ratio >= 1.0
          -> force distill; if irreducible -> mark as anchor memory
      survival_score > 0.8
          -> reinforce (resist future decay, factor 1.2)
    """

    def __init__(
        self,
        mdl_estimator: MDLEstimator,
        cascade: CompressionCascade,
        *,
        eviction_threshold: float = 0.1,
        reinforcement_threshold: float = 0.8,
        reinforcement_factor: float = 1.2,
        anchor_compression_threshold: float = 1.0,
        access_decay_rate: float = 0.1,
        contradiction_decay_multiplier: float = 2.0,
    ) -> None:
        self._mdl = mdl_estimator
        self._cascade = cascade
        self._eviction_threshold = eviction_threshold
        self._reinforcement_threshold = reinforcement_threshold
        self._reinforcement_factor = reinforcement_factor
        self._anchor_compression_threshold = anchor_compression_threshold
        self._access_decay_rate = access_decay_rate
        self._contradiction_multiplier = contradiction_decay_multiplier

    async def run_decay_cycle(
        self,
        memory_store: MemoryStoreProtocol,
        anchor_ids: set[str],
        *,
        max_items: int = 0,
    ) -> DecayReport:
        """
        Run a full decay cycle over all scored items in the memory store.

        Applies access decay, compression decay, and contradiction decay,
        then uses the survival_score decision tree to evict, distill, or
        reinforce each item.
        """
        all_items = await memory_store.get_all_scored_items()
        if max_items > 0:
            all_items = all_items[:max_items]

        evicted: list[str] = []
        distilled: list[str] = []
        reinforced: list[str] = []
        total_bits_freed = 0.0

        for item in all_items:
            # Skip anchor memories -- permanent
            if item.id in anchor_ids:
                continue

            score = await self._score_with_decay(item)

            if score.survival_score < self._eviction_threshold:
                if score.compression_ratio < self._anchor_compression_threshold:
                    # Explains less than it costs -> evict
                    await memory_store.evict(item.id)
                    evicted.append(item.id)
                    total_bits_freed += score.description_length
                    logger.debug(
                        "decay_evicted",
                        item_id=item.id,
                        survival=score.survival_score,
                        compression_ratio=score.compression_ratio,
                    )
                else:
                    # High compression but low survival -> try distillation
                    distilled_form = await self._cascade.force_distill(
                        item.id, item.content
                    )
                    if distilled_form is not None:
                        await memory_store.replace(item.id, distilled_form)
                        distilled.append(item.id)
                        bits_saved = distilled_form.raw_bits - distilled_form.distilled_bits
                        total_bits_freed += max(bits_saved, 0.0)
                        logger.debug(
                            "decay_distilled",
                            item_id=item.id,
                            bits_saved=bits_saved,
                        )
                    else:
                        # Genuinely irreducible -> anchor memory
                        await memory_store.mark_anchor(item.id)
                        anchor_ids.add(item.id)
                        reinforced.append(item.id)
                        logger.info(
                            "decay_anchor_created",
                            item_id=item.id,
                            compression_ratio=score.compression_ratio,
                        )

            elif score.survival_score > self._reinforcement_threshold:
                await memory_store.reinforce(item.id, self._reinforcement_factor)
                reinforced.append(item.id)

        report = DecayReport(
            timestamp=utc_now(),
            evicted=evicted,
            distilled=distilled,
            reinforced=reinforced,
            total_items_scanned=len(all_items),
            total_bits_freed=total_bits_freed,
        )

        logger.info(
            "decay_cycle_complete",
            scanned=report.total_items_scanned,
            evicted=len(evicted),
            distilled=len(distilled),
            reinforced=len(reinforced),
            bits_freed=total_bits_freed,
        )

        return report

    async def _score_with_decay(self, item: ScoredItem) -> MDLScore:
        """
        Score an item with all three decay adjustments applied.

        1. Access decay: exponential decay based on time since last access
        2. Compression decay: items with ratio < 1.0 get penalised
        3. Contradiction decay: contradicted items decay faster
        """
        # Compute access frequency from access_count and age
        age_hours = max(
            (datetime.now(UTC) - item.last_accessed).total_seconds() / 3600,
            0.01,
        )
        access_freq = float(item.access_count) / age_hours

        # observation_complexity: how much reality this item covers.
        # We don't have a direct measure here, so we use description_length as
        # the baseline (ratio = 1.0 neutral). Access count modulates the ratio:
        # frequently-accessed items are assumed to cover proportionally more
        # reality (they are being used in predictions, i.e. explaining things).
        # This means items that are never accessed drift toward ratio < 1.0 as
        # access_decay reduces access_frequency, triggering eviction/distillation.
        observation_complexity = item.description_length * max(
            1.0, math.log1p(float(item.access_count))
        )
        base_score = await self._mdl.score_generic(
            item_id=item.id,
            item_type=(
                KnowledgeItemType(item.item_type)
                if item.item_type in {e.value for e in KnowledgeItemType}
                else KnowledgeItemType.EPISODE
            ),
            description_length=item.description_length,
            observation_complexity=observation_complexity,
            last_accessed=item.last_accessed,
            access_frequency=access_freq,
        )

        # 1) Access decay: exponential
        access_decay = math.exp(-self._access_decay_rate * age_hours)

        # 2) Contradiction decay: multiplicative penalty
        contradiction_factor = 1.0
        if item.contradiction_count > 0:
            contradiction_factor = 1.0 / (
                1.0 + self._contradiction_multiplier * item.contradiction_count
            )

        # Adjust the decay_rate upward for stale/contradicted items
        adjusted_decay_rate = base_score.decay_rate * (
            1.0 / max(access_decay * contradiction_factor, 0.01)
        )

        return MDLScore(
            item_id=base_score.item_id,
            item_type=base_score.item_type,
            observations_covered=base_score.observations_covered,
            observation_complexity=base_score.observation_complexity,
            description_length=base_score.description_length,
            compression_ratio=base_score.compression_ratio,
            marginal_value=base_score.marginal_value,
            last_accessed=base_score.last_accessed,
            access_frequency=base_score.access_frequency * access_decay,
            decay_rate=adjusted_decay_rate,
        )
