"""
EcodiaOS - Identity Genome Extraction & Seeding

Heritable state: constitutional hash, identity parameters (generation,
instance config), certificate configuration.

The identity genome is critical for speciation - it defines what makes
this organism _this_ organism. A child inherits the parent's constitutional
hash and identity parameters, establishing lineage continuity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from primitives.common import SystemID
from primitives.genome import OrganGenomeSegment
from systems.genome_helpers import build_segment, check_schema_version, verify_segment

if TYPE_CHECKING:
    from systems.identity.identity import IdentitySystem

logger = structlog.get_logger()


class IdentityGenomeExtractor:
    """Extracts identity heritable state for genome transmission to child organisms."""

    def __init__(
        self,
        identity_system: IdentitySystem | None = None,
        certificate_manager: object | None = None,
    ) -> None:
        self._identity = identity_system
        self._cert_mgr = certificate_manager
        self._log = logger.bind(subsystem="identity.genome")

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        """
        Extract the identity genome segment.

        Heritable state:
          - constitutional_hash: the hash of the constitutional document
          - generation: current generation number
          - parent_instance_id: who spawned this organism
          - certificate_config: certificate validity, renewal settings
          - identity_parameters: drift threshold, verification config
        """
        try:
            if self._identity is None:
                return build_segment(SystemID.IDENTITY, {}, version=0)

            payload: dict[str, object] = {
                "constitutional_hash": self._identity.constitutional_hash,
                "generation": self._identity.generation,
                "parent_instance_id": self._identity.parent_instance_id or "",
                "identity_parameters": self._extract_identity_parameters(),
                "certificate_config": self._extract_certificate_config(),
            }

            self._log.info(
                "identity_genome_extracted",
                generation=self._identity.generation,
                hash_prefix=self._identity.constitutional_hash[:16],
            )
            return build_segment(SystemID.IDENTITY, payload, version=1)

        except Exception as exc:
            self._log.error("identity_genome_extract_failed", error=str(exc))
            return build_segment(SystemID.IDENTITY, {}, version=0)

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        """
        Seed identity state from a parent's genome segment.

        Applies inherited constitutional hash, generation lineage,
        and certificate configuration to the child organism.
        """
        if segment.version == 0 or not segment.payload:
            return True

        if not check_schema_version(segment):
            return False
        if not verify_segment(segment):
            return False

        try:
            payload = segment.payload

            if self._identity is not None:
                # The child inherits the parent's constitutional hash as its starting point
                inherited_hash = str(payload.get("constitutional_hash", ""))
                if inherited_hash:
                    self._identity._constitutional_hash = inherited_hash

                parent_gen = int(payload.get("generation", 0))
                self._identity._generation = parent_gen + 1

            self._log.info(
                "identity_genome_seeded",
                inherited_generation=payload.get("generation", 0),
            )
            return True

        except Exception as exc:
            self._log.error("identity_genome_seed_failed", error=str(exc))
            return False

    def _extract_identity_parameters(self) -> dict[str, object]:
        """Extract identity verification and drift detection parameters."""
        return {
            "drift_threshold": 0.7,
            "verification_interval_s": 3600,
        }

    def _extract_certificate_config(self) -> dict[str, object]:
        """Extract certificate lifecycle configuration."""
        config: dict[str, object] = {
            "validity_days": 30,
            "expiry_warning_days": 7,
            "protocol_version": "1.0",
        }

        if self._cert_mgr is not None:
            if hasattr(self._cert_mgr, "_validity_days"):
                config["validity_days"] = int(self._cert_mgr._validity_days)
            if hasattr(self._cert_mgr, "_expiry_warning_days"):
                config["expiry_warning_days"] = int(self._cert_mgr._expiry_warning_days)

        return config
