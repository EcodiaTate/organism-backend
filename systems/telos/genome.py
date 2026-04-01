"""
EcodiaOS - Telos Genome Extraction & Seeding

Heritable state: drive topology weights, intelligence measurement
calibration, alignment gap thresholds.

Child inherits parent's calibrated measurement apparatus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from primitives.common import SystemID
from primitives.genome import OrganGenomeSegment
from systems.genome_helpers import build_segment, check_schema_version, verify_segment

if TYPE_CHECKING:
    from systems.telos.service import TelosService

logger = structlog.get_logger()


class TelosGenomeExtractor:
    """Extracts Telos measurement calibration for genome transmission."""

    def __init__(self, service: TelosService) -> None:
        self._service = service
        self._log = logger.bind(subsystem="telos.genome")

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        try:
            drive_topology = self._extract_drive_topology()
            measurement_calibration = self._extract_measurement_calibration()
            alignment_thresholds = self._extract_alignment_thresholds()

            if not drive_topology and not measurement_calibration:
                return build_segment(SystemID.TELOS, {}, version=0)

            payload = {
                "drive_topology": drive_topology,
                "measurement_calibration": measurement_calibration,
                "alignment_thresholds": alignment_thresholds,
            }

            self._log.info("telos_genome_extracted")
            return build_segment(SystemID.TELOS, payload, version=1)

        except Exception as exc:
            self._log.error("telos_genome_extract_failed", error=str(exc))
            return build_segment(SystemID.TELOS, {}, version=0)

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        if segment.version == 0 or not segment.payload:
            return True

        if not check_schema_version(segment):
            return False
        if not verify_segment(segment):
            return False

        try:
            payload = segment.payload
            self._apply_drive_topology(payload.get("drive_topology", {}))
            self._apply_measurement_calibration(
                payload.get("measurement_calibration", {})
            )
            self._apply_alignment_thresholds(
                payload.get("alignment_thresholds", {})
            )

            self._log.info("telos_genome_seeded")
            return True

        except Exception as exc:
            self._log.error("telos_genome_seed_failed", error=str(exc))
            return False

    # ── Extraction helpers ─────────────────────────────────────────

    def _extract_drive_topology(self) -> dict:
        """Extract drive topology weights and multipliers."""
        try:
            topology: dict[str, object] = {}
            config = self._service._config
            if config is None:
                return topology

            drives = ["care", "coherence", "growth", "honesty"]
            for drive in drives:
                weight_key = f"{drive}_weight"
                if hasattr(config, weight_key):
                    topology[weight_key] = float(getattr(config, weight_key))

            if hasattr(config, "effective_i_window_size"):
                topology["effective_i_window_size"] = int(
                    config.effective_i_window_size
                )
            if hasattr(config, "nominal_i_decay"):
                topology["nominal_i_decay"] = float(config.nominal_i_decay)

            return topology
        except Exception:
            return {}

    def _extract_measurement_calibration(self) -> dict:
        """Extract intelligence measurement calibration data."""
        try:
            calibration: dict[str, object] = {}

            # Recent alignment history for calibration
            if hasattr(self._service, "_recent_alignments"):
                alignments = list(self._service._recent_alignments)
                if alignments:
                    composites = [
                        a.composite if hasattr(a, "composite") else 0.0
                        for a in alignments
                    ]
                    calibration["mean_alignment_composite"] = sum(composites) / len(
                        composites
                    )
                    calibration["alignment_sample_size"] = len(composites)

            config = self._service._config
            if config is not None:
                if hasattr(config, "measurement_interval_s"):
                    calibration["measurement_interval_s"] = int(
                        config.measurement_interval_s
                    )
                if hasattr(config, "stagnation_window_cycles"):
                    calibration["stagnation_window_cycles"] = int(
                        config.stagnation_window_cycles
                    )

            return calibration
        except Exception:
            return {}

    def _extract_alignment_thresholds(self) -> dict:
        """Extract alignment gap thresholds."""
        try:
            thresholds: dict[str, float] = {}
            config = self._service._config
            if config is None:
                return thresholds

            if hasattr(config, "care_gap_threshold"):
                thresholds["care_gap_threshold"] = float(config.care_gap_threshold)
            if hasattr(config, "coherence_gap_threshold"):
                thresholds["coherence_gap_threshold"] = float(
                    config.coherence_gap_threshold
                )
            if hasattr(config, "growth_stagnation_threshold"):
                thresholds["growth_stagnation_threshold"] = float(
                    config.growth_stagnation_threshold
                )
            if hasattr(config, "honesty_drop_threshold"):
                thresholds["honesty_drop_threshold"] = float(
                    config.honesty_drop_threshold
                )

            return thresholds
        except Exception:
            return {}

    # ── Seeding helpers ────────────────────────────────────────────

    def _apply_drive_topology(self, topology: dict) -> None:
        """Apply inherited drive topology weights."""
        if not topology:
            return
        config = self._service._config
        if config is None:
            return
        for key, value in topology.items():
            if hasattr(config, key):
                setattr(config, key, type(getattr(config, key))(value))

    def _apply_measurement_calibration(self, calibration: dict) -> None:
        """Apply inherited measurement calibration."""
        if not calibration:
            return
        config = self._service._config
        if config is None:
            return
        for key, value in calibration.items():
            if key.startswith("mean_") or key.startswith("alignment_sample"):
                continue  # Informational, not directly applied
            if hasattr(config, key):
                setattr(config, key, type(getattr(config, key))(value))

    def _apply_alignment_thresholds(self, thresholds: dict) -> None:
        """Apply inherited alignment gap thresholds."""
        if not thresholds:
            return
        config = self._service._config
        if config is None:
            return
        for key, value in thresholds.items():
            if hasattr(config, key):
                setattr(config, key, float(value))
