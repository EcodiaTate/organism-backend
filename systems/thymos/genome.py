"""
EcodiaOS - Thymos Genome Extraction & Seeding

Heritable state: antibody library (known-bad patterns + fixes),
incident pattern history, healing strategy effectiveness scores.

Child starts with parent's immune memory - knows what threats look like.
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

_MAX_ANTIBODIES = 500
_MAX_INCIDENT_PATTERNS = 200


class ThymosGenomeExtractor:
    """Extracts Thymos immune memory for genome transmission."""

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j
        self._log = logger.bind(subsystem="thymos.genome")

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        antibodies = await self._extract_antibodies()
        incident_patterns = await self._extract_incident_patterns()
        healing_effectiveness = await self._extract_healing_effectiveness()

        if not antibodies and not incident_patterns:
            return build_segment(SystemID.THYMOS, {}, version=0)

        payload = {
            "antibodies": antibodies,
            "incident_patterns": incident_patterns,
            "healing_effectiveness": healing_effectiveness,
        }

        self._log.info(
            "thymos_genome_extracted",
            antibodies=len(antibodies),
            incident_patterns=len(incident_patterns),
        )
        return build_segment(SystemID.THYMOS, payload, version=1)

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        if segment.version == 0 or not segment.payload:
            return True

        if not check_schema_version(segment):
            return False
        if not verify_segment(segment):
            return False

        try:
            payload = segment.payload
            seeded_antibodies = await self._seed_antibodies(
                payload.get("antibodies", [])
            )
            seeded_patterns = await self._seed_incident_patterns(
                payload.get("incident_patterns", [])
            )

            self._log.info(
                "thymos_genome_seeded",
                antibodies=seeded_antibodies,
                incident_patterns=seeded_patterns,
            )
            return True

        except Exception as exc:
            self._log.error("thymos_genome_seed_failed", error=str(exc))
            return False

    # ── Extraction helpers ─────────────────────────────────────────

    async def _extract_antibodies(self) -> list[dict]:
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (a:AntibodyTemplate)
                RETURN a.fingerprint AS fingerprint,
                       a.repair_steps AS repair_steps,
                       a.success_rate AS success_rate,
                       a.incident_class AS incident_class,
                       a.usage_count AS usage_count,
                       a.description AS description
                ORDER BY a.success_rate * a.usage_count DESC
                LIMIT $limit
                """,
                {"limit": _MAX_ANTIBODIES},
            )
        except Exception as exc:
            self._log.warning("thymos_extract_antibodies_failed", error=str(exc))
            return []

        return [
            {
                "fingerprint": str(row.get("fingerprint", "")),
                "repair_steps": row.get("repair_steps", []),
                "success_rate": float(row.get("success_rate", 0.0)),
                "incident_class": str(row.get("incident_class", "")),
                "usage_count": int(row.get("usage_count", 0)),
                "description": str(row.get("description", "")),
            }
            for row in rows
        ]

    async def _extract_incident_patterns(self) -> list[dict]:
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (i:Incident)
                WITH i.incident_class AS cls,
                     count(i) AS total,
                     avg(i.severity_score) AS avg_severity,
                     collect(DISTINCT i.root_cause)[..5] AS root_causes
                WHERE total >= 2
                RETURN cls, total, avg_severity, root_causes
                ORDER BY total DESC
                LIMIT $limit
                """,
                {"limit": _MAX_INCIDENT_PATTERNS},
            )
        except Exception as exc:
            self._log.warning("thymos_extract_patterns_failed", error=str(exc))
            return []

        return [
            {
                "incident_class": str(row.get("cls", "")),
                "total_occurrences": int(row.get("total", 0)),
                "avg_severity": float(row.get("avg_severity", 0.0) or 0.0),
                "root_causes": row.get("root_causes", []),
            }
            for row in rows
        ]

    async def _extract_healing_effectiveness(self) -> dict:
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (i:Incident)
                WHERE i.repair_status IS NOT NULL
                WITH i.repair_tier AS tier,
                     count(i) AS total,
                     sum(CASE WHEN i.repair_status = 'success' THEN 1 ELSE 0 END) AS successes
                RETURN tier, total, successes
                """,
            )
        except Exception as exc:
            self._log.warning("thymos_extract_effectiveness_failed", error=str(exc))
            return {}

        effectiveness: dict[str, dict] = {}
        for row in rows:
            tier = str(row.get("tier", "unknown"))
            total = int(row.get("total", 0))
            successes = int(row.get("successes", 0))
            effectiveness[tier] = {
                "total": total,
                "successes": successes,
                "success_rate": successes / total if total > 0 else 0.0,
            }
        return effectiveness

    # ── Seeding helpers ────────────────────────────────────────────

    async def _seed_antibodies(self, antibodies: list[dict]) -> int:
        seeded = 0
        for ab in antibodies:
            try:
                from primitives.common import new_id, utc_now

                await self._neo4j.execute_write(
                    """
                    CREATE (:AntibodyTemplate {
                        id: $id,
                        fingerprint: $fingerprint,
                        repair_steps: $repair_steps,
                        success_rate: $success_rate,
                        incident_class: $incident_class,
                        usage_count: $usage_count,
                        description: $description,
                        source: "parent_genome",
                        created_at: datetime($now)
                    })
                    """,
                    {
                        "id": new_id(),
                        "fingerprint": ab.get("fingerprint", ""),
                        "repair_steps": ab.get("repair_steps", []),
                        "success_rate": float(ab.get("success_rate", 0.0)) * 0.9,
                        "incident_class": ab.get("incident_class", ""),
                        "usage_count": 0,
                        "description": ab.get("description", ""),
                        "now": utc_now().isoformat(),
                    },
                )
                seeded += 1
            except Exception as exc:
                self._log.debug("thymos_seed_antibody_failed", error=str(exc))
        return seeded

    async def _seed_incident_patterns(self, patterns: list[dict]) -> int:
        seeded = 0
        for pat in patterns:
            try:
                from primitives.common import new_id, utc_now

                await self._neo4j.execute_write(
                    """
                    CREATE (:IncidentPattern {
                        id: $id,
                        incident_class: $cls,
                        total_occurrences: $total,
                        avg_severity: $severity,
                        root_causes: $causes,
                        source: "parent_genome",
                        created_at: datetime($now)
                    })
                    """,
                    {
                        "id": new_id(),
                        "cls": pat.get("incident_class", ""),
                        "total": pat.get("total_occurrences", 0),
                        "severity": pat.get("avg_severity", 0.0),
                        "causes": pat.get("root_causes", []),
                        "now": utc_now().isoformat(),
                    },
                )
                seeded += 1
            except Exception as exc:
                self._log.debug("thymos_seed_pattern_failed", error=str(exc))
        return seeded
