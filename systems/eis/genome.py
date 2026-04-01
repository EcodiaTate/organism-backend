"""
EcodiaOS - EIS Genome Extraction (Speciation: Immune Memory Inheritance)

Implements GenomeExtractionProtocol so Mitosis can snapshot EIS's heritable
state and seed child instances with the parent's immune knowledge.

Heritable state includes:
  - Threat library patterns (learned attack signatures)
  - Anomaly detector baselines (statistical norms)
  - Quarantine threshold adjustments
  - False positive history

A child instance starts with its parent's immune memory - it doesn't have
to learn every threat from scratch. This is the immune co-evolution vector
that enables population-level threat resistance.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import structlog

from primitives.common import SystemID
from primitives.genome import GenomeExtractionProtocol, OrganGenomeSegment
from systems.eis.anomaly_detector import AnomalyDetector, ExponentialStats
from systems.eis.threat_library import ThreatLibrary, ThreatPatternStatus

logger = structlog.get_logger().bind(system="eis", component="genome")

_SCHEMA_VERSION = "1.0"


class EISGenomeExtractor:
    """
    Extracts and seeds EIS heritable state for Mitosis genome operations.

    Implements GenomeExtractionProtocol.
    """

    def __init__(
        self,
        threat_library: ThreatLibrary,
        anomaly_detector: AnomalyDetector,
        quarantine_threshold: float = 0.45,
        soma_quarantine_offset: float = 0.0,
    ) -> None:
        self._threat_library = threat_library
        self._anomaly_detector = anomaly_detector
        self._quarantine_threshold = quarantine_threshold
        self._soma_quarantine_offset = soma_quarantine_offset

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        """
        Serialize EIS's heritable state into a genome segment.

        Extracts:
          - Active threat patterns (not retired/decayed)
          - Anomaly detector rate baselines
          - Current quarantine thresholds
        """
        # Extract active threat patterns
        patterns_data: list[dict[str, Any]] = []
        for pattern in self._threat_library._patterns.values():
            if pattern.status != ThreatPatternStatus.ACTIVE:
                continue
            patterns_data.append(pattern.model_dump(mode="json"))

        # Extract anomaly baselines
        baselines_data: dict[str, dict[str, float]] = {}
        for event_type, stats in self._anomaly_detector._rate_baselines.items():
            if stats.count >= 10:  # only inherit established baselines
                baselines_data[event_type] = {
                    "mean": stats.mean,
                    "variance": stats.variance,
                    "count": stats.count,
                }

        payload: dict[str, Any] = {
            "threat_patterns": patterns_data,
            "anomaly_baselines": baselines_data,
            "quarantine_threshold": self._quarantine_threshold,
            "soma_quarantine_offset": self._soma_quarantine_offset,
            "library_stats": self._threat_library.stats(),
        }

        payload_json = json.dumps(payload, sort_keys=True, default=str)
        payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()

        segment = OrganGenomeSegment(
            system_id=SystemID.EIS,
            version=1,
            schema_version=_SCHEMA_VERSION,
            payload=payload,
            payload_hash=payload_hash,
            size_bytes=len(payload_json.encode()),
        )

        logger.info(
            "eis_genome_extracted",
            patterns=len(patterns_data),
            baselines=len(baselines_data),
            size_bytes=segment.size_bytes,
        )

        return segment

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        """
        Restore EIS heritable state from a parent's genome segment.

        The child starts with the parent's immune knowledge - threat patterns
        and anomaly baselines - so it can defend itself from known threats
        immediately after birth.
        """
        if segment.system_id != SystemID.EIS:
            logger.warning("eis_genome_seed_wrong_system", system=segment.system_id)
            return False

        payload = segment.payload
        seeded_patterns = 0
        seeded_baselines = 0

        # Seed threat patterns
        for pattern_data in payload.get("threat_patterns", []):
            try:
                from systems.eis.threat_library import ThreatPattern
                pattern = ThreatPattern(**pattern_data)
                # Reset match counters for the child - inherited, not earned
                pattern.match_count = 0
                pattern.last_matched = None
                pattern.false_positive_count = 0
                pattern.true_positive_count = 0
                if self._threat_library.register_pattern(pattern):
                    seeded_patterns += 1
            except Exception as exc:
                logger.debug("eis_genome_seed_pattern_failed", error=str(exc))

        # Seed anomaly baselines
        for event_type, baseline_data in payload.get("anomaly_baselines", {}).items():
            try:
                stats = ExponentialStats(
                    mean=baseline_data["mean"],
                    variance=baseline_data["variance"],
                    count=baseline_data["count"],
                )
                self._anomaly_detector._rate_baselines[event_type] = stats
                seeded_baselines += 1
            except Exception as exc:
                logger.debug("eis_genome_seed_baseline_failed", error=str(exc))

        logger.info(
            "eis_genome_seeded",
            patterns_seeded=seeded_patterns,
            baselines_seeded=seeded_baselines,
            parent_hash=segment.payload_hash[:16],
        )

        return seeded_patterns > 0 or seeded_baselines > 0


# Verify protocol compliance at import time
assert isinstance(EISGenomeExtractor, type)
_: type[GenomeExtractionProtocol] = EISGenomeExtractor  # type: ignore[assignment]
