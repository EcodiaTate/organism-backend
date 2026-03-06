"""
EcodiaOS — Logos: Compression Cascade

The four-stage pipeline that every experience traverses before reaching
long-term storage.  Each stage extracts more structure and discards more
redundancy, producing measurable compression ratios (bits-in vs bits-out).

Stage 1 — Holographic Encoding:  delta vs world model prediction (holographic.py)
Stage 2 — Episodic Compression:  prediction-errors only, near-zero delta discarded
Stage 3 — Semantic Distillation: pattern-extraction into entities/relations/schemas
Stage 4 — World Model Integration: distilled structures -> WorldModel.integrate()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id
from systems.logos.types import (
    CascadeResult,
    CompressionStage,
    ExperienceDelta,
    RawExperience,
    SalientEpisode,
    SemanticExtraction,
    StageMetrics,
    WorldModelUpdate,
)

if TYPE_CHECKING:
    from systems.logos.holographic import HolographicEncoder
    from systems.logos.world_model import WorldModel

logger = structlog.get_logger("logos.cascade")


class CompressionCascade:
    """
    Four-stage compression pipeline.

    Every experience entering EOS traverses this before reaching long-term
    storage.  At each stage information is either compressed further
    (its pattern extracted) or declared irreducible (stored as-is, flagged
    as high-novelty anchor material).

    Measurable compression ratios at every stage.
    """

    def __init__(
        self,
        holographic_encoder: HolographicEncoder,
        world_model: WorldModel,
        *,
        salience_threshold: float = 0.05,
        pattern_merge_threshold: int = 3,
    ) -> None:
        self._encoder = holographic_encoder
        self._world_model = world_model
        self._salience_threshold = salience_threshold
        self._pattern_merge_threshold = pattern_merge_threshold

        # Accumulator: episodes pending semantic distillation
        self._episode_buffer: list[SalientEpisode] = []

        # Throughput counters (per cycle, reset externally)
        self.total_cascaded: int = 0
        self.total_discarded_stage2: int = 0
        self.total_distilled: int = 0
        self.total_integrated: int = 0
        self.total_anchors: int = 0

    # --- Public API ---------------------------------------------------

    async def run(self, raw_experience: RawExperience) -> CascadeResult:
        """
        Run a single experience through all four stages.

        Returns a CascadeResult with per-stage metrics, the final
        compressed output, and anchor/irreducibility flags.
        """
        stage_metrics: list[StageMetrics] = []
        bits_in_total = max(raw_experience.raw_complexity, 1.0)

        # -- Stage 1: Holographic Encoding --------------------------
        delta = await self._encoder.encode(raw_experience)

        s1_bits_out = (
            delta.information_content * bits_in_total
            if not delta.discard_after_encoding
            else 0.0
        )
        stage_metrics.append(StageMetrics(
            stage=CompressionStage.HOLOGRAPHIC_ENCODING,
            bits_in=bits_in_total,
            bits_out=s1_bits_out,
            items_in=1,
            items_out=0 if delta.discard_after_encoding else 1,
        ))

        if delta.discard_after_encoding:
            self.total_cascaded += 1
            return CascadeResult(
                experience_id=raw_experience.id,
                stage_reached=CompressionStage.HOLOGRAPHIC_ENCODING,
                delta=delta,
                compression_ratio=bits_in_total / max(s1_bits_out, 0.001),
                bits_saved=bits_in_total,
                stage_metrics=stage_metrics,
            )

        # -- Stage 2: Episodic Compression --------------------------
        episode, s2_metrics = self._episodic_compress(delta, raw_experience)
        stage_metrics.append(s2_metrics)

        if episode is None:
            self.total_cascaded += 1
            self.total_discarded_stage2 += 1
            total_out = s1_bits_out
            return CascadeResult(
                experience_id=raw_experience.id,
                stage_reached=CompressionStage.EPISODIC_COMPRESSION,
                delta=delta,
                compression_ratio=bits_in_total / max(total_out, 0.001),
                bits_saved=bits_in_total - total_out,
                stage_metrics=stage_metrics,
            )

        # -- Stage 3: Semantic Distillation -------------------------
        # Distill against the EXISTING buffer (before appending the current episode),
        # so the current episode is not matched against itself in the pattern scan.
        # After distillation, append to the buffer for future episodes to match against.
        extraction, s3_metrics = self._semantic_distill(episode)
        self._episode_buffer.append(episode)
        stage_metrics.append(s3_metrics)

        # -- Stage 4: World Model Integration -----------------------
        wm_update: WorldModelUpdate | None = None
        s4_bits_in = (
            extraction.distilled_bits
            if extraction.distilled_bits > 0
            else episode.compressed_bits
        )
        s4_bits_out = s4_bits_in  # Default: no further compression

        if delta.world_model_update_required or extraction.schemas:
            wm_update = await self._world_model.integrate(delta)
            self.total_integrated += 1
            # World model integration achieves further compression
            # by absorbing structure into existing schemas/priors
            absorbed = (wm_update.schemas_extended + wm_update.priors_updated) * 0.1
            s4_bits_out = s4_bits_in * max(0.3, 1.0 - absorbed)

        stage_metrics.append(StageMetrics(
            stage=CompressionStage.WORLD_MODEL_INTEGRATION,
            bits_in=s4_bits_in,
            bits_out=s4_bits_out,
            items_in=1,
            items_out=1,
        ))

        # Determine irreducibility / anchor status
        is_irreducible = self._check_irreducible(episode, extraction)
        anchor = False
        if is_irreducible:
            anchor = True
            self.total_anchors += 1

        self.total_cascaded += 1
        self.total_distilled += 1

        total_bits_out = s4_bits_out
        return CascadeResult(
            experience_id=raw_experience.id,
            stage_reached=CompressionStage.WORLD_MODEL_INTEGRATION,
            delta=delta,
            salient_episode=episode,
            semantic_extraction=extraction,
            world_model_update=wm_update,
            compressed_item_id=episode.id,
            is_irreducible=is_irreducible,
            anchor_memory=anchor,
            compression_ratio=bits_in_total / max(total_bits_out, 0.001),
            bits_saved=bits_in_total - total_bits_out,
            stage_metrics=stage_metrics,
        )

    async def force_distill(
        self, item_id: str, item_content: dict[str, Any]
    ) -> SemanticExtraction | None:
        """
        Force distillation of a single item (called by EntropicDecayEngine).

        Attempts to extract any compressible pattern from the item.
        Returns None if the item is genuinely irreducible.
        """
        raw_bits = float(len(str(item_content))) * 4.5  # Shannon estimate

        entities: list[str] = []
        relations: list[str] = []
        for key, val in item_content.items():
            if isinstance(val, str):
                entities.append(val)
            elif isinstance(val, dict):
                relations.append(key)

        if not entities and not relations:
            return None  # Irreducible

        distilled_bits = raw_bits * 0.6  # ~40% compression from distillation
        return SemanticExtraction(
            entities=entities,
            relations=relations,
            episode_refs=[item_id],
            raw_bits=raw_bits,
            distilled_bits=distilled_bits,
        )

    def flush_episode_buffer(self) -> list[SalientEpisode]:
        """Drain the episode buffer (for batch processing by decay engine)."""
        buf = list(self._episode_buffer)
        self._episode_buffer.clear()
        return buf

    def reset_counters(self) -> None:
        self.total_cascaded = 0
        self.total_discarded_stage2 = 0
        self.total_distilled = 0
        self.total_integrated = 0
        self.total_anchors = 0

    # --- Stage 2: Episodic Compression ----------------------------

    def _episodic_compress(
        self,
        delta: ExperienceDelta,
        raw: RawExperience,
    ) -> tuple[SalientEpisode | None, StageMetrics]:
        """
        Only prediction errors survive.  Items with near-zero delta
        (information_content < salience_threshold) are discarded.
        """
        bits_in = delta.information_content * max(raw.raw_complexity, 1.0)

        if (
            delta.delta_content is None
            or delta.information_content < self._salience_threshold
        ):
            return None, StageMetrics(
                stage=CompressionStage.EPISODIC_COMPRESSION,
                bits_in=bits_in,
                bits_out=0.0,
                items_in=1,
                items_out=0,
            )

        # Salience = information content weighted by violation severity
        violation_weight = len(delta.delta_content.violated_priors) * 0.2
        salience = min(delta.information_content + violation_weight, 1.0)

        # Compressed representation: only the novel/violated fields
        content = delta.delta_content.content
        compressed_bits = bits_in * (0.3 + 0.7 * delta.information_content)

        episode = SalientEpisode(
            id=new_id(),
            experience_id=raw.id,
            delta=delta,
            prediction_error=delta.information_content,
            salience=salience,
            context=raw.context,
            content=content,
            raw_bits=bits_in,
            compressed_bits=compressed_bits,
        )

        return episode, StageMetrics(
            stage=CompressionStage.EPISODIC_COMPRESSION,
            bits_in=bits_in,
            bits_out=compressed_bits,
            items_in=1,
            items_out=1,
        )

    # --- Stage 3: Semantic Distillation ---------------------------

    def _semantic_distill(
        self,
        episode: SalientEpisode,
    ) -> tuple[SemanticExtraction, StageMetrics]:
        """
        Extract structured knowledge: entities, relations, schemas, hypotheses.

        Multiple episodes sharing a pattern get replaced by 1 semantic node
        plus N lightweight references.  Check the buffer for pattern matches.
        """
        bits_in = episode.compressed_bits

        # Extract entities and relations from the episode content
        entities: list[str] = []
        relations: list[str] = []
        for key, val in episode.content.items():
            if isinstance(val, str):
                entities.append(val)
            elif isinstance(val, dict):
                relations.append(key)
            elif isinstance(val, list):
                entities.extend(str(v) for v in val)

        # Check for pattern sharing across buffered episodes
        schemas: list[str] = []
        merged_refs: list[str] = [episode.id]
        merge_savings = 0.0

        if len(self._episode_buffer) >= self._pattern_merge_threshold:
            shared_entities = self._find_shared_patterns(entities)
            if shared_entities:
                matching = [
                    ep for ep in self._episode_buffer
                    if self._shares_pattern(ep, shared_entities)
                ]
                if len(matching) >= self._pattern_merge_threshold:
                    schema_id = new_id()
                    schemas.append(schema_id)
                    merged_refs = [ep.id for ep in matching]
                    # Replace N full episodes with 1 schema + N refs
                    full_cost = sum(ep.compressed_bits for ep in matching)
                    ref_cost = len(matching) * (bits_in * 0.1)
                    schema_cost = bits_in * 0.5
                    merge_savings = max(0.0, full_cost - (schema_cost + ref_cost))
                    # Remove merged episodes from the buffer
                    merged_ids = {ep.id for ep in matching}
                    self._episode_buffer = [
                        ep for ep in self._episode_buffer
                        if ep.id not in merged_ids
                    ]

        # Distilled bits: compressed form minus merge savings
        distilled_bits = max(bits_in * 0.7 - merge_savings, bits_in * 0.1)

        extraction = SemanticExtraction(
            entities=entities,
            relations=relations,
            schemas=schemas,
            episode_refs=merged_refs,
            raw_bits=bits_in,
            distilled_bits=distilled_bits,
        )

        return extraction, StageMetrics(
            stage=CompressionStage.SEMANTIC_DISTILLATION,
            bits_in=bits_in,
            bits_out=distilled_bits,
            items_in=1,
            items_out=1,
        )

    # --- Irreducibility Check -------------------------------------

    def _check_irreducible(
        self,
        episode: SalientEpisode,
        extraction: SemanticExtraction,
    ) -> bool:
        """
        An item is irreducibly novel if:
        - High prediction error (> 0.8) and low distillation savings (< 20%)
        - Contains invariant violations (violated_priors non-empty)
        These become anchor memories — permanent, never evicted.
        """
        if episode.prediction_error > 0.8:
            savings = 1.0 - (
                extraction.distilled_bits / max(extraction.raw_bits, 0.001)
            )
            if savings < 0.2:
                return True

        return bool(
            episode.delta.delta_content
            and episode.delta.delta_content.violated_priors
        )

    # --- Pattern helpers ------------------------------------------

    def _find_shared_patterns(self, entities: list[str]) -> set[str]:
        """Find entities that appear across multiple buffered episodes."""
        entity_set = set(entities)
        shared: set[str] = set()
        for ep in self._episode_buffer:
            ep_entities: set[str] = set()
            for val in ep.content.values():
                if isinstance(val, str):
                    ep_entities.add(val)
                elif isinstance(val, list):
                    ep_entities.update(str(v) for v in val)
            overlap = entity_set & ep_entities
            shared.update(overlap)
        return shared

    def _shares_pattern(
        self, episode: SalientEpisode, pattern: set[str]
    ) -> bool:
        """Check if an episode shares at least one entity from the pattern."""
        for val in episode.content.values():
            if isinstance(val, str) and val in pattern:
                return True
            if isinstance(val, list):
                for v in val:
                    if str(v) in pattern:
                        return True
        return False
