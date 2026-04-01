"""
EcodiaOS - Federation Genome Extraction & Seeding

Heritable state: trust policy defaults (NOT actual trust scores - those are
earned), protocol preferences, known threat patterns.

Child starts with parent's trust policy and threat awareness. Does NOT
inherit parent's relationships.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from primitives.common import SystemID
from primitives.genome import OrganGenomeSegment
from systems.genome_helpers import build_segment, check_schema_version, verify_segment

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()

_MAX_THREATS = 200


class FederationGenomeExtractor:
    """Extracts Federation trust policy and threat awareness for genome transmission."""

    def __init__(self, neo4j: Neo4jClient | None = None, config: object | None = None) -> None:
        self._neo4j = neo4j
        self._config = config
        self._log = logger.bind(subsystem="federation.genome")

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        try:
            trust_policy = self._extract_trust_policy()
            protocol_preferences = self._extract_protocol_preferences()
            known_threats = await self._extract_known_threats()

            if not trust_policy and not known_threats:
                return build_segment(SystemID.FEDERATION, {}, version=0)

            payload = {
                "trust_policy": trust_policy,
                "protocol_preferences": protocol_preferences,
                "known_threats": known_threats,
            }

            self._log.info(
                "federation_genome_extracted",
                threats=len(known_threats),
            )
            return build_segment(SystemID.FEDERATION, payload, version=1)

        except Exception as exc:
            self._log.error("federation_genome_extract_failed", error=str(exc))
            return build_segment(SystemID.FEDERATION, {}, version=0)

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        if segment.version == 0 or not segment.payload:
            return True

        if not check_schema_version(segment):
            return False
        if not verify_segment(segment):
            return False

        try:
            payload = segment.payload
            self._apply_trust_policy(payload.get("trust_policy", {}))
            self._apply_protocol_preferences(payload.get("protocol_preferences", {}))
            await self._seed_known_threats(payload.get("known_threats", []))

            self._log.info("federation_genome_seeded")
            return True

        except Exception as exc:
            self._log.error("federation_genome_seed_failed", error=str(exc))
            return False

    # ── Extraction helpers ─────────────────────────────────────────

    def _extract_trust_policy(self) -> dict:
        """Extract trust policy defaults from config."""
        try:
            if self._config is None:
                return {}
            policy: dict[str, object] = {}
            if hasattr(self._config, "default_trust_level"):
                policy["default_trust_level"] = str(self._config.default_trust_level)
            if hasattr(self._config, "trust_decay_rate"):
                policy["trust_decay_rate"] = float(self._config.trust_decay_rate)
            if hasattr(self._config, "probation_duration_hours"):
                policy["probation_duration_hours"] = int(
                    self._config.probation_duration_hours
                )
            if hasattr(self._config, "max_concurrent_links"):
                policy["max_concurrent_links"] = int(
                    self._config.max_concurrent_links
                )
            if hasattr(self._config, "violation_thresholds"):
                policy["violation_thresholds"] = self._config.violation_thresholds
            return policy
        except Exception:
            return {}

    def _extract_protocol_preferences(self) -> dict:
        """Extract protocol preferences."""
        try:
            if self._config is None:
                return {}
            prefs: dict[str, object] = {}
            if hasattr(self._config, "preferred_exchange_protocol"):
                prefs["preferred_exchange_protocol"] = str(
                    self._config.preferred_exchange_protocol
                )
            if hasattr(self._config, "knowledge_sharing_threshold"):
                prefs["knowledge_sharing_threshold"] = float(
                    self._config.knowledge_sharing_threshold
                )
            if hasattr(self._config, "assistance_acceptance_threshold"):
                prefs["assistance_acceptance_threshold"] = float(
                    self._config.assistance_acceptance_threshold
                )
            return prefs
        except Exception:
            return {}

    async def _extract_known_threats(self) -> list[dict]:
        """Extract known threat patterns from Neo4j."""
        if self._neo4j is None:
            return []
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (t:ThreatAdvisory)
                RETURN t.threat_type AS threat_type,
                       t.pattern AS pattern,
                       t.severity AS severity,
                       t.source_instance AS source_instance,
                       t.description AS description,
                       t.confidence AS confidence
                ORDER BY t.confidence DESC
                LIMIT $limit
                """,
                {"limit": _MAX_THREATS},
            )
        except Exception as exc:
            self._log.warning("federation_extract_threats_failed", error=str(exc))
            return []

        return [
            {
                "threat_type": str(row.get("threat_type", "")),
                "pattern": str(row.get("pattern", "")),
                "severity": str(row.get("severity", "")),
                "description": str(row.get("description", "")),
                "confidence": float(row.get("confidence", 0.0)),
            }
            for row in rows
        ]

    # ── Seeding helpers ────────────────────────────────────────────

    def _apply_trust_policy(self, policy: dict) -> None:
        """Apply inherited trust policy to config."""
        if not policy or self._config is None:
            return
        for key, value in policy.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    def _apply_protocol_preferences(self, prefs: dict) -> None:
        """Apply inherited protocol preferences to config."""
        if not prefs or self._config is None:
            return
        for key, value in prefs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    async def _seed_known_threats(self, threats: list[dict]) -> int:
        """Seed known threat patterns into Neo4j."""
        if not threats or self._neo4j is None:
            return 0
        seeded = 0
        for threat in threats:
            try:
                from primitives.common import new_id, utc_now

                await self._neo4j.execute_write(
                    """
                    CREATE (:ThreatAdvisory {
                        id: $id,
                        threat_type: $threat_type,
                        pattern: $pattern,
                        severity: $severity,
                        description: $description,
                        confidence: $confidence,
                        source: "parent_genome",
                        created_at: datetime($now)
                    })
                    """,
                    {
                        "id": new_id(),
                        "threat_type": threat.get("threat_type", ""),
                        "pattern": threat.get("pattern", ""),
                        "severity": threat.get("severity", ""),
                        "description": threat.get("description", ""),
                        "confidence": float(threat.get("confidence", 0.0)) * 0.9,
                        "now": utc_now().isoformat(),
                    },
                )
                seeded += 1
            except Exception as exc:
                self._log.debug("federation_seed_threat_failed", error=str(exc))
        return seeded
