"""
EcodiaOS - Oneiros v2: Emergence Stage (Full Wake Preparation)

The transition back to wake state. Not passive awakening - active preparation.

Phase D expansion:
1. Finalize all world model integration via Logos
2. Measure intelligence ratio improvement (now vs checkpoint)
3. Build pre-attention cache for Fovea (pre-generated predictions for top-N contexts)
4. Prepare genome update for potential instance spawning (Mitosis)
5. Compose sleep narrative for Thread
6. Resume input channels
7. Broadcast WAKE_INITIATED with full WakeStatePreparation payload

Tracks: average_intelligence_improvement_per_sleep_cycle - the single metric
predicting growth rate. If declining, signals to Telos Growth that new domain
exposure is needed.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID
from primitives.genome import OrganGenomeSegment
from systems.oneiros.types import (
    EmergenceReport,
    LucidDreamingReport,
    PreAttentionCache,
    PreAttentionEntry,
    REMStageReport,
    SleepCheckpoint,
    SleepNarrative,
    SlowWaveReport,
    WakeStatePreparation,
)
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from systems.logos.service import LogosService
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("oneiros.emergence")

_SOURCE = "oneiros"


class EmergenceStage:
    """
    Stage 4: Emergence (~10% of sleep duration).

    Full wake preparation:
    1. Finalize world model integration
    2. Measure intelligence improvement
    3. Build pre-attention cache from REM dream generation
    4. Prepare genome update for Mitosis
    5. Compose sleep narrative for Thread
    6. Resume input channels
    7. Broadcast WAKE_INITIATED with full WakeStatePreparation

    Tracks cumulative intelligence improvement across cycles.
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._logger = logger.bind(stage="emergence")

        # Track intelligence improvement history for growth rate metric
        self._improvement_history: list[float] = []

    @property
    def average_intelligence_improvement(self) -> float:
        """The single metric predicting growth rate."""
        if not self._improvement_history:
            return 0.0
        return sum(self._improvement_history) / len(self._improvement_history)

    @property
    def intelligence_improvement_declining(self) -> bool:
        """True if recent improvements are below historical average."""
        if len(self._improvement_history) < 3:
            return False
        recent = self._improvement_history[-3:]
        older = self._improvement_history[:-3]
        if not older:
            return False
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        return recent_avg < older_avg * 0.8  # 20% decline threshold

    async def execute(
        self,
        checkpoint: SleepCheckpoint,
        logos: LogosService | None = None,
        sleep_start_time: float | None = None,
        *,
        slow_wave_report: SlowWaveReport | None = None,
        rem_report: REMStageReport | None = None,
        lucid_report: LucidDreamingReport | None = None,
        pre_attention_entries: list[PreAttentionEntry] | None = None,
        sleep_cycle_id: str = "",
    ) -> EmergenceReport:
        """
        Execute the full Emergence stage.

        1. Finalize world model integration
        2. Measure intelligence ratio improvement
        3. Build pre-attention cache
        4. Prepare genome update
        5. Compose sleep narrative
        6. Resume input channels
        7. Broadcast WAKE_INITIATED with full payload
        """
        self._logger.info("emergence_starting", checkpoint_id=checkpoint.id)

        ratio_before = checkpoint.intelligence_ratio_at_sleep
        ratio_after = ratio_before
        finalized = False

        # 1. Finalize world model integration
        if logos is not None:
            await logos.run_batch_compression(force=True, max_items=50)
            finalized = True
            # 2. Measure intelligence ratio improvement
            ratio_after = logos.get_intelligence_ratio()

        improvement = ratio_after - ratio_before

        # Track improvement history
        self._improvement_history.append(improvement)
        avg_improvement = self.average_intelligence_improvement

        # 3. Build pre-attention cache from REM dream generation
        cache = self._build_pre_attention_cache(pre_attention_entries or [])

        # 4. Prepare genome update for Mitosis
        genome_prepared = await self._prepare_genome_update(
            improvement, logos
        )

        # 5. Compose sleep narrative for Thread
        narrative = self._compose_sleep_narrative(
            sleep_cycle_id=sleep_cycle_id,
            improvement=improvement,
            slow_wave=slow_wave_report,
            rem=rem_report,
            lucid=lucid_report,
        )

        # 6. Resume input channels
        await self._resume_input_channels()

        # 7. Broadcast WAKE_INITIATED with full payload
        sleep_duration_s = 0.0
        if sleep_start_time is not None:
            sleep_duration_s = time.monotonic() - sleep_start_time

        wake_prep = WakeStatePreparation(
            intelligence_ratio_before=ratio_before,
            intelligence_ratio_after=ratio_after,
            intelligence_improvement=improvement,
            world_model_finalized=finalized,
            pre_attention_cache=cache,
            sleep_narrative=narrative,
            genome_update_prepared=genome_prepared,
            input_channels_resumed=True,
            average_intelligence_improvement_per_cycle=avg_improvement,
        )

        await self._broadcast_wake_initiated(wake_prep, sleep_duration_s)

        # Signal if growth rate is declining - emit Synapse event for Telos Growth
        if self.intelligence_improvement_declining:
            self._logger.warning(
                "intelligence_improvement_declining",
                avg_improvement=round(avg_improvement, 4),
                recent_improvement=round(improvement, 4),
                note="signal_telos_growth_new_domain_exposure_needed",
            )
            if self._event_bus is not None:
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.INTELLIGENCE_IMPROVEMENT_DECLINING,
                    source_system=_SOURCE,
                    data={
                        "average_improvement": avg_improvement,
                        "recent_improvement": improvement,
                        "history_length": len(self._improvement_history),
                        "signal": "new_domain_exposure_needed",
                    },
                ))

        report = EmergenceReport(
            intelligence_ratio_before=ratio_before,
            intelligence_ratio_after=ratio_after,
            intelligence_improvement=improvement,
            world_model_finalized=finalized,
            input_channels_resumed=True,
        )

        self._logger.info(
            "emergence_complete",
            ratio_before=round(ratio_before, 4),
            ratio_after=round(ratio_after, 4),
            improvement=round(improvement, 4),
            avg_improvement=round(avg_improvement, 4),
            pre_attention_size=cache.total_predictions,
            genome_prepared=genome_prepared,
            sleep_duration_s=round(sleep_duration_s, 1),
        )

        return report

    def _build_pre_attention_cache(
        self, entries: list[PreAttentionEntry]
    ) -> PreAttentionCache:
        """Build the pre-attention cache from REM dream generation results."""
        if not entries:
            return PreAttentionCache()

        domains = {e.domain for e in entries}
        return PreAttentionCache(
            entries=entries,
            domains_covered=len(domains),
            total_predictions=len(entries),
        )

    async def _prepare_genome_update(
        self,
        improvement: float,
        logos: LogosService | None,
    ) -> bool:
        """Prepare genome update for potential Mitosis instance spawning.

        Extracts consolidated beliefs, schemas, and hypotheses from the world
        model into an OrganGenomeSegment and emits ONEIROS_GENOME_READY so
        Mitosis can incorporate it into child genomes.
        """
        if logos is None:
            return False

        if improvement <= 0.001:
            return False

        schemas = logos.get_generative_schemas()
        causal = logos.get_causal_structure()
        wm = logos.world_model

        # Extract heritable state from the world model
        payload: dict[str, Any] = {
            "intelligence_ratio": logos.get_intelligence_ratio(),
            "complexity": logos.get_current_complexity(),
            "schema_count": len(schemas),
            "invariant_count": len(wm.empirical_invariants),
            "causal_link_count": causal.link_count if hasattr(causal, "link_count") else 0,
            "schemas": {
                sid: {
                    "domain": getattr(s, "domain", "general"),
                    "pattern": getattr(s, "pattern", {}),
                }
                for sid, s in list(schemas.items())[:50]
            },
            "invariants": [
                {
                    "id": getattr(inv, "id", ""),
                    "statement": getattr(inv, "statement", ""),
                    "confidence": getattr(inv, "confidence", 0.0),
                    "domain": getattr(inv, "domain", "general"),
                }
                for inv in list(wm.empirical_invariants)[:50]
            ],
            "improvement_history": self._improvement_history[-20:],
        }

        payload_json = json.dumps(payload, sort_keys=True, default=str)
        payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()

        segment = OrganGenomeSegment(
            system_id=SystemID.ONEIROS,
            payload=payload,
            payload_hash=payload_hash,
            size_bytes=len(payload_json.encode()),
        )

        # Emit ONEIROS_GENOME_READY via Synapse (fire-and-forget)
        if self._event_bus is not None:
            event = SynapseEvent(
                event_type=SynapseEventType.ONEIROS_GENOME_READY,
                source_system=_SOURCE,
                data=segment.model_dump(mode="json"),
            )
            try:
                await self._event_bus.emit(event)
            except Exception as exc:
                self._logger.debug("genome_event_emit_failed", error=str(exc))

        self._logger.debug(
            "genome_update_prepared",
            improvement=round(improvement, 4),
            complexity=round(logos.get_current_complexity(), 1),
            schema_count=len(schemas),
            payload_size_bytes=segment.size_bytes,
        )
        return True

    def _compose_sleep_narrative(
        self,
        sleep_cycle_id: str,
        improvement: float,
        slow_wave: SlowWaveReport | None,
        rem: REMStageReport | None,
        lucid: LucidDreamingReport | None,
    ) -> SleepNarrative:
        """Compose a narrative of what happened during sleep for Thread."""
        parts: list[str] = []

        # Compression summary
        compression_summary = ""
        hypotheses_retired = 0
        if slow_wave is not None:
            comp = slow_wave.compression
            hypotheses_retired = slow_wave.hypotheses.hypotheses_retired
            compression_summary = (
                f"Compressed {comp.memories_processed} memories through "
                f"4-rung ladder: {comp.semantic_nodes_created} semantic nodes, "
                f"{comp.schemas_created} schemas, {comp.procedures_extracted} procedures, "
                f"{comp.world_model_updates} world model updates. "
                f"Compression ratio: {comp.compression_ratio:.2f}."
            )
            parts.append(compression_summary)

            if hypotheses_retired > 0:
                parts.append(
                    f"Retired {hypotheses_retired} hypotheses "
                    f"(freed {slow_wave.hypotheses.total_mdl_freed:.0f} bits)."
                )

            if slow_wave.causal.invariants_discovered > 0:
                parts.append(
                    f"Discovered {slow_wave.causal.invariants_discovered} "
                    f"causal invariants."
                )

        # REM summary
        cross_domain_matches = 0
        analogies_discovered = 0
        dreams_generated = 0
        if rem is not None:
            cross_domain_matches = rem.cross_domain.strong_matches
            analogies_discovered = rem.analogies.analogies_found
            dreams_generated = rem.dreams.scenarios_generated

            if cross_domain_matches > 0:
                parts.append(
                    f"Found {cross_domain_matches} cross-domain structural matches "
                    f"({rem.cross_domain.evo_candidates} Evo candidates)."
                )
            if dreams_generated > 0:
                parts.append(
                    f"Generated {dreams_generated} dream scenarios, "
                    f"extracted {rem.dreams.hypotheses_extracted} hypotheses."
                )
            if analogies_discovered > 0:
                parts.append(
                    f"Discovered {analogies_discovered} analogical transfers "
                    f"(applied {rem.analogies.analogies_applied})."
                )

        # Lucid summary
        mutations_tested = 0
        if lucid is not None and lucid.mutations_tested > 0:
            mutations_tested = lucid.mutations_tested
            parts.append(
                f"Tested {mutations_tested} mutations in lucid dreaming: "
                f"{lucid.mutations_recommended_apply} recommended, "
                f"{lucid.mutations_recommended_reject} rejected."
            )

        # Intelligence improvement
        if improvement > 0:
            parts.append(f"Intelligence ratio improved by {improvement:.4f}.")
        elif improvement < 0:
            parts.append(
                f"Intelligence ratio decreased by {abs(improvement):.4f} - "
                f"world model may need recalibration."
            )

        narrative_text = " ".join(parts) if parts else "Uneventful sleep cycle."

        return SleepNarrative(
            sleep_cycle_id=sleep_cycle_id,
            compression_summary=compression_summary,
            hypotheses_retired=hypotheses_retired,
            cross_domain_matches=cross_domain_matches,
            analogies_discovered=analogies_discovered,
            dreams_generated=dreams_generated,
            mutations_tested=mutations_tested,
            intelligence_improvement=improvement,
            narrative_text=narrative_text,
        )

    async def _resume_input_channels(self) -> None:
        """Resume real-time input processing.

        Channel suspension/resumption is managed at the SleepCycleEngine level
        (engine._is_sleeping flag + Descent._input_suspended).  The WAKE_INITIATED
        broadcast signals downstream systems to resume perception cycles.
        """
        self._logger.debug("input_channels_resumed")

    async def _broadcast_wake_initiated(
        self,
        wake_prep: WakeStatePreparation,
        sleep_duration_s: float,
    ) -> None:
        """Broadcast WAKE_INITIATED with full WakeStatePreparation payload.

        Also emits FOVEA_PREATTENTION_CACHE_READY (Gap 2 fix) when the cache is
        non-empty so Fovea can load pre-generated predictions as precision priors.
        """
        if self._event_bus is None:
            return

        event = SynapseEvent(
            event_type=SynapseEventType.WAKE_INITIATED,
            source_system=_SOURCE,
            data={
                "intelligence_improvement": wake_prep.intelligence_improvement,
                "intelligence_ratio_before": wake_prep.intelligence_ratio_before,
                "intelligence_ratio_after": wake_prep.intelligence_ratio_after,
                "sleep_duration_s": sleep_duration_s,
                "world_model_finalized": wake_prep.world_model_finalized,
                "pre_attention_cache_size": wake_prep.pre_attention_cache.total_predictions,
                "genome_update_prepared": wake_prep.genome_update_prepared,
                "sleep_narrative": wake_prep.sleep_narrative.narrative_text,
                "average_intelligence_improvement_per_cycle": (
                    wake_prep.average_intelligence_improvement_per_cycle
                ),
            },
        )
        await self._event_bus.emit(event)

        # Wire ORGANISM_WAKE - Axon, Identity, SACM, Simula subscribe to this.
        # WAKE_INITIATED is Oneiros-internal; ORGANISM_WAKE is the organism-wide signal.
        organism_wake_event = SynapseEvent(
            event_type=SynapseEventType.ORGANISM_WAKE,
            source_system=_SOURCE,
            data={
                "sleep_duration_s": sleep_duration_s,
                "intelligence_improvement": wake_prep.intelligence_improvement,
            },
        )
        await self._event_bus.emit(organism_wake_event)

        # Gap 2 fix: deliver full pre-attention cache to Fovea via a dedicated event.
        # WAKE_INITIATED carries only the count (for lightweight consumers).
        # Fovea subscribes to FOVEA_PREATTENTION_CACHE_READY for the full payload.
        cache = wake_prep.pre_attention_cache
        if cache.total_predictions > 0:
            try:
                entries_payload = [
                    {
                        "domain": getattr(e, "domain", ""),
                        "context_key": getattr(e, "context_key", ""),
                        "predicted_content": getattr(e, "predicted_content", {}),
                        "confidence": getattr(e, "confidence", 0.0),
                        "generating_schema_ids": getattr(e, "generating_schema_ids", []),
                    }
                    for e in (cache.entries or [])
                ]
            except Exception:
                entries_payload = []

            cache_event = SynapseEvent(
                event_type=SynapseEventType.FOVEA_PREATTENTION_CACHE_READY,
                source_system=_SOURCE,
                data={
                    "entries": entries_payload,
                    "domains_covered": cache.domains_covered,
                    "total_predictions": cache.total_predictions,
                    "sleep_cycle_id": wake_prep.sleep_narrative.sleep_cycle_id,
                },
            )
            await self._event_bus.emit(cache_event)
