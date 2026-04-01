"""
EcodiaOS - Synapse Genome Extraction & Seeding

Heritable state: resource allocation weights, rhythm classification
thresholds, degradation strategy config.

Child starts with parent's tuned resource allocation instead of defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from primitives.common import SystemID
from primitives.genome import OrganGenomeSegment
from systems.genome_helpers import build_segment, check_schema_version, verify_segment

if TYPE_CHECKING:
    from systems.synapse.service import SynapseService

logger = structlog.get_logger()


class SynapseGenomeExtractor:
    """Extracts Synapse resource allocation and rhythm config for genome transmission."""

    def __init__(self, service: SynapseService) -> None:
        self._service = service
        self._log = logger.bind(subsystem="synapse.genome")

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        try:
            allocation_weights = self._extract_allocation_weights()
            rhythm_thresholds = self._extract_rhythm_thresholds()
            degradation_config = self._extract_degradation_config()

            if not allocation_weights and not rhythm_thresholds:
                return build_segment(SystemID.SYNAPSE, {}, version=0)

            payload = {
                "allocation_weights": allocation_weights,
                "rhythm_thresholds": rhythm_thresholds,
                "degradation_config": degradation_config,
            }

            self._log.info(
                "synapse_genome_extracted",
                allocation_keys=len(allocation_weights),
            )
            return build_segment(SystemID.SYNAPSE, payload, version=1)

        except Exception as exc:
            self._log.error("synapse_genome_extract_failed", error=str(exc))
            return build_segment(SystemID.SYNAPSE, {}, version=0)

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        if segment.version == 0 or not segment.payload:
            return True

        if not check_schema_version(segment):
            return False
        if not verify_segment(segment):
            return False

        try:
            payload = segment.payload
            self._apply_allocation_weights(payload.get("allocation_weights", {}))
            self._apply_rhythm_thresholds(payload.get("rhythm_thresholds", {}))
            self._apply_degradation_config(payload.get("degradation_config", {}))

            self._log.info("synapse_genome_seeded")
            return True

        except Exception as exc:
            self._log.error("synapse_genome_seed_failed", error=str(exc))
            return False

    # ── Extraction helpers ─────────────────────────────────────────

    def _extract_allocation_weights(self) -> dict:
        """Extract per-system resource allocation weights from the allocator."""
        try:
            config = self._service._config
            weights: dict[str, float] = {}
            # Extract from config if available
            if hasattr(config, "resource_weights"):
                for sys_id, weight in config.resource_weights.items():
                    weights[str(sys_id)] = float(weight)
            # Also capture cycle period as it's a tuned parameter
            if hasattr(config, "cycle_period_ms"):
                weights["_cycle_period_ms"] = float(config.cycle_period_ms)
            return weights
        except Exception:
            return {}

    def _extract_rhythm_thresholds(self) -> dict:
        """Extract rhythm detection thresholds."""
        try:
            thresholds: dict[str, float] = {}
            config = self._service._config
            if hasattr(config, "coherence_threshold"):
                thresholds["coherence_threshold"] = float(config.coherence_threshold)
            if hasattr(config, "stall_threshold_cycles"):
                thresholds["stall_threshold_cycles"] = float(
                    config.stall_threshold_cycles
                )
            if hasattr(config, "metabolic_pressure_threshold"):
                thresholds["metabolic_pressure_threshold"] = float(
                    config.metabolic_pressure_threshold
                )
            return thresholds
        except Exception:
            return {}

    def _extract_degradation_config(self) -> dict:
        """Extract graceful degradation strategy config."""
        try:
            config = self._service._config
            deg: dict[str, object] = {}
            if hasattr(config, "degradation_levels"):
                deg["degradation_levels"] = config.degradation_levels
            if hasattr(config, "max_degradation_depth"):
                deg["max_degradation_depth"] = config.max_degradation_depth
            return deg
        except Exception:
            return {}

    # ── Seeding helpers ────────────────────────────────────────────

    def _apply_allocation_weights(self, weights: dict) -> None:
        """Apply inherited resource allocation weights."""
        if not weights:
            return
        config = self._service._config
        if hasattr(config, "resource_weights"):
            for sys_id, weight in weights.items():
                if sys_id.startswith("_"):
                    continue
                config.resource_weights[sys_id] = float(weight)
        cycle_ms = weights.get("_cycle_period_ms")
        if cycle_ms is not None and hasattr(config, "cycle_period_ms"):
            config.cycle_period_ms = int(cycle_ms)

    def _apply_rhythm_thresholds(self, thresholds: dict) -> None:
        """Apply inherited rhythm thresholds."""
        if not thresholds:
            return
        config = self._service._config
        for key, value in thresholds.items():
            if hasattr(config, key):
                setattr(config, key, type(getattr(config, key))(value))

    def _apply_degradation_config(self, deg: dict) -> None:
        """Apply inherited degradation config."""
        if not deg:
            return
        config = self._service._config
        for key, value in deg.items():
            if hasattr(config, key):
                setattr(config, key, value)
