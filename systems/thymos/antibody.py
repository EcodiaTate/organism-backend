"""
EcodiaOS — Thymos Antibody Library (Immune Memory)

The accumulated immune intelligence of the organism. Every successful
repair becomes an antibody that can be instantly applied to future
incidents with the same fingerprint.

Lifecycle:
  1. CREATION    — A successful Tier 3+ repair → new antibody
  2. APPLICATION — Matching fingerprint → instant Tier 3 repair
  3. FEEDBACK    — Track success/failure on each application
  4. REFINEMENT  — Effectiveness drops below 0.6 → regenerate repair
  5. RETIREMENT  — Effectiveness below 0.3 after 5+ applications → retire

The library grows over time. An old organism has hundreds of antibodies
and resolves most incidents instantly without LLM-powered diagnosis.
This is genuine adaptive immunity — it compounds.
"""

from __future__ import annotations

import json
import re
from typing import Any

import structlog

from primitives.common import new_id, utc_now
from systems.thymos.types import (
    Antibody,
    Incident,
    IncidentClass,
    RepairSpec,
    RepairTier,
)

logger = structlog.get_logger()


class AntibodyLibrary:
    """
    The organism's immune memory.

    Uses Neo4j for persistent storage so antibodies survive restarts.
    Falls back to in-memory storage when Neo4j is unavailable.
    """

    EFFECTIVENESS_REFINEMENT_THRESHOLD = 0.6
    EFFECTIVENESS_RETIREMENT_THRESHOLD = 0.3
    MIN_APPLICATIONS_FOR_RETIREMENT = 5
    # Time-based decay: effectiveness halves every 7 days of non-use
    DECAY_HALF_LIFE_HOURS = 7 * 24.0

    def __init__(self, neo4j_client: Any = None) -> None:
        self._neo4j = neo4j_client
        # In-memory cache for fast lookups
        self._cache: dict[str, Antibody] = {}  # fingerprint → antibody
        self._all: dict[str, Antibody] = {}  # id → antibody
        self._logger = logger.bind(system="thymos", component="antibody_library")

    async def initialize(self) -> None:
        """Load antibodies from Neo4j into memory cache."""
        if self._neo4j is None:
            self._logger.info("antibody_library_in_memory_only")
            return

        try:
            await self._ensure_schema()
            results = await self._neo4j.execute_read(
                """
                MATCH (a:Antibody)
                WHERE a.retired = false
                RETURN a
                ORDER BY a.effectiveness DESC
                """
            )

            for record in results:
                node = record["a"] if isinstance(record, dict) else record[0]
                antibody = self._node_to_antibody(node)
                self._cache[antibody.fingerprint] = antibody
                self._all[antibody.id] = antibody

            self._logger.info(
                "antibody_library_loaded",
                total=len(self._all),
                active=len(self._cache),
            )
        except Exception as exc:
            self._logger.warning(
                "antibody_library_load_failed",
                error=str(exc),
            )

    async def _ensure_schema(self) -> None:
        """Create Neo4j indexes for antibody queries."""
        if self._neo4j is None:
            return
        try:
            await self._neo4j.execute_write(
                "CREATE INDEX antibody_fingerprint IF NOT EXISTS "
                "FOR (a:Antibody) ON (a.fingerprint)"
            )
            await self._neo4j.execute_write(
                "CREATE INDEX antibody_retired IF NOT EXISTS "
                "FOR (a:Antibody) ON (a.retired)"
            )
        except Exception:
            pass  # Indexes may already exist

    def _apply_time_decay(self, antibody: Antibody) -> None:
        """
        Apply time-based effectiveness decay for antibodies not recently used.

        Effectiveness decays toward a floor of 0.5 (not zero) based on how
        long since the antibody was last applied. This ensures stale fixes
        don't maintain artificially high confidence while still recognizing
        they once worked.
        """
        if antibody.last_applied is None:
            return
        hours_since = (utc_now() - antibody.last_applied).total_seconds() / 3600.0
        if hours_since < 1.0:
            return  # Applied recently, no decay

        import math
        decay_factor = math.exp(
            -hours_since * math.log(2) / self.DECAY_HALF_LIFE_HOURS
        )
        # Decay toward 0.5 floor (not zero — the fix did work at some point)
        floor = 0.5
        antibody.effectiveness = floor + (antibody.effectiveness - floor) * decay_factor

    async def lookup(self, fingerprint: str) -> Antibody | None:
        """
        History-driven antibody selection for this fingerprint.

        Queries all non-retired antibodies for this fingerprint from Neo4j,
        ranks them by a composite score that penalises recent failures, and
        returns the best candidate.  Falls back to in-memory cache when
        Neo4j is unavailable.
        """
        # Try Neo4j for full history-driven ranking
        if self._neo4j is not None:
            try:
                results = await self._neo4j.execute_read(
                    """
                    MATCH (a:Antibody {fingerprint: $fingerprint, retired: false})
                    RETURN a
                    ORDER BY a.effectiveness DESC, a.success_count DESC
                    LIMIT 10
                    """,
                    {"fingerprint": fingerprint},
                )
                if results:
                    candidates: list[Antibody] = []
                    for record in results:
                        node = record["a"] if isinstance(record, dict) else record[0]
                        candidates.append(self._node_to_antibody(node))

                    best = self._rank_candidates(candidates)
                    if best is not None:
                        self._cache[fingerprint] = best
                        self._all[best.id] = best
                        return best
            except Exception as exc:
                self._logger.debug("neo4j_antibody_lookup_failed", error=str(exc))

        # Fall back to in-memory cache
        cached = self._cache.get(fingerprint)
        if cached is not None and not cached.retired:
            return cached

        return None

    def _rank_candidates(self, candidates: list[Antibody]) -> Antibody | None:
        """
        Rank antibody candidates by history-aware composite score.

        Score = effectiveness * recency_boost * tier_preference
        - effectiveness: success_count / (success + failure), baseline 1.0
        - recency_boost: antibodies applied recently and succeeded get a bonus;
          antibodies that failed recently get a penalty
        - tier_preference: prefer least-invasive tier (lower tier = higher score)
        """
        if not candidates:
            return None

        # Apply time-based decay before ranking
        for c in candidates:
            self._apply_time_decay(c)

        from primitives.common import utc_now
        now = utc_now()

        def _score(ab: Antibody) -> float:
            eff = ab.effectiveness

            # Penalise antibodies with recent failures (last 3 applications)
            if ab.application_count > 0 and ab.failure_count > 0:
                recent_failure_ratio = ab.failure_count / ab.application_count
                eff *= (1.0 - recent_failure_ratio * 0.5)

            # Recency: antibodies not applied in >24h get slight discount
            # (they may be stale); recently-successful ones get a boost
            recency_boost = 1.0
            if ab.last_applied is not None:
                hours_since = (now - ab.last_applied).total_seconds() / 3600.0
                if hours_since < 1.0 and ab.effectiveness > 0.7:
                    recency_boost = 1.1  # Recently worked — prefer it
                elif hours_since > 24.0:
                    recency_boost = 0.95  # Stale — slight discount

            # Tier preference: lower tier = less invasive = preferable
            # NOOP=0, PARAMETER=1, RESTART=2, KNOWN_FIX=3, NOVEL_FIX=4
            tier_preference = max(0.7, 1.0 - ab.repair_tier.value * 0.05)

            return eff * recency_boost * tier_preference

        return max(candidates, key=_score)

    async def record_outcome(self, antibody_id: str, success: bool) -> None:
        """
        Record whether an antibody application succeeded or failed.
        Updates effectiveness. Triggers refinement or retirement.
        """
        antibody = self._all.get(antibody_id)
        if antibody is None:
            return

        if success:
            antibody.success_count += 1
        else:
            antibody.failure_count += 1

        total = antibody.success_count + antibody.failure_count
        antibody.effectiveness = antibody.success_count / total if total > 0 else 1.0
        antibody.application_count += 1
        antibody.last_applied = utc_now()

        # Persist to Neo4j
        await self._persist_antibody(antibody)

        # Check for retirement
        if (
            antibody.effectiveness < self.EFFECTIVENESS_RETIREMENT_THRESHOLD
            and antibody.application_count >= self.MIN_APPLICATIONS_FOR_RETIREMENT
        ):
            await self._retire(antibody)
            self._logger.info(
                "antibody_retired",
                antibody_id=antibody.id,
                effectiveness=antibody.effectiveness,
                applications=antibody.application_count,
            )
            return

        # Check for refinement
        if antibody.effectiveness < self.EFFECTIVENESS_REFINEMENT_THRESHOLD:
            self._logger.info(
                "antibody_needs_refinement",
                antibody_id=antibody.id,
                effectiveness=antibody.effectiveness,
            )

    async def create_from_repair(
        self,
        incident: Incident,
        repair: RepairSpec,
    ) -> Antibody:
        """
        When a repair succeeds, crystallize it into an antibody.
        The next time this fingerprint appears, we'll apply this fix
        instantly instead of going through diagnosis.
        """
        antibody = Antibody(
            id=new_id(),
            fingerprint=incident.fingerprint,
            incident_class=incident.incident_class,
            source_system=incident.source_system,
            error_pattern=self._extract_error_pattern(incident),
            repair_tier=repair.tier,
            repair_spec=repair,
            root_cause_description=incident.root_cause_hypothesis or "Auto-discovered",
            source_incident_id=incident.id,
            created_at=utc_now(),
        )

        # Cache it
        self._cache[antibody.fingerprint] = antibody
        self._all[antibody.id] = antibody

        # Persist to Neo4j
        await self._persist_antibody(antibody)

        # Link to source incident in Neo4j
        if self._neo4j is not None:
            from contextlib import suppress

            with suppress(Exception):  # Non-critical — incident node may not exist yet
                await self._neo4j.execute_write(
                    """
                    MATCH (a:Antibody {id: $aid})
                    MATCH (i:Incident {id: $iid})
                    MERGE (a)-[:GENERATED_FROM]->(i)
                    """,
                    {"aid": antibody.id, "iid": incident.id},
                )

        self._logger.info(
            "antibody_created",
            antibody_id=antibody.id,
            fingerprint=incident.fingerprint,
            source_system=incident.source_system,
            tier=repair.tier.name,
        )

        return antibody

    async def _persist_antibody(self, antibody: Antibody) -> None:
        """Persist an antibody to Neo4j."""
        if self._neo4j is None:
            return

        try:
            # Flatten repair_spec for Neo4j storage (no nested dicts)
            repair_spec_json = json.dumps(antibody.repair_spec.model_dump())

            await self._neo4j.execute_write(
                """
                MERGE (a:Antibody {id: $id})
                SET a.fingerprint = $fingerprint,
                    a.incident_class = $incident_class,
                    a.source_system = $source_system,
                    a.error_pattern = $error_pattern,
                    a.repair_tier = $repair_tier,
                    a.repair_spec_json = $repair_spec_json,
                    a.root_cause_description = $root_cause_description,
                    a.application_count = $application_count,
                    a.success_count = $success_count,
                    a.failure_count = $failure_count,
                    a.effectiveness = $effectiveness,
                    a.created_at = $created_at,
                    a.last_applied = $last_applied,
                    a.source_incident_id = $source_incident_id,
                    a.retired = $retired,
                    a.generation = $generation,
                    a.parent_antibody_id = $parent_antibody_id
                """,
                {
                    "id": antibody.id,
                    "fingerprint": antibody.fingerprint,
                    "incident_class": antibody.incident_class.value,
                    "source_system": antibody.source_system,
                    "error_pattern": antibody.error_pattern,
                    "repair_tier": antibody.repair_tier.value,
                    "repair_spec_json": repair_spec_json,
                    "root_cause_description": antibody.root_cause_description,
                    "application_count": antibody.application_count,
                    "success_count": antibody.success_count,
                    "failure_count": antibody.failure_count,
                    "effectiveness": antibody.effectiveness,
                    "created_at": antibody.created_at.isoformat(),
                    "last_applied": (
                        antibody.last_applied.isoformat() if antibody.last_applied else None
                    ),
                    "source_incident_id": antibody.source_incident_id,
                    "retired": antibody.retired,
                    "generation": antibody.generation,
                    "parent_antibody_id": antibody.parent_antibody_id,
                },
            )
        except Exception as exc:
            self._logger.warning("antibody_persist_failed", error=str(exc))

    async def _retire(self, antibody: Antibody) -> None:
        """Retire an antibody that's no longer effective."""
        antibody.retired = True
        # Remove from fingerprint cache (but keep in _all for history)
        if self._cache.get(antibody.fingerprint) is antibody:
            del self._cache[antibody.fingerprint]
        await self._persist_antibody(antibody)

    def _extract_error_pattern(self, incident: Incident) -> str:
        """Extract a reusable error pattern from an incident."""
        # Use the first 200 chars of error message as a pattern
        msg = incident.error_message[:200]
        # Remove variable parts (numbers, ULIDs, timestamps) for better matching
        msg = re.sub(r"\b[0-9a-f]{8,}\b", "<ID>", msg)
        msg = re.sub(r"\b\d+\.\d+\b", "<NUM>", msg)
        msg = re.sub(r"\b\d+\b", "<N>", msg)
        return msg

    def _node_to_antibody(self, node: Any) -> Antibody:
        """Convert a Neo4j node to an Antibody."""
        props = dict(node) if hasattr(node, "__iter__") else {}

        # Parse repair_spec from JSON
        repair_spec_json = props.get("repair_spec_json", "{}")
        try:
            repair_data = json.loads(repair_spec_json) if isinstance(repair_spec_json, str) else {}
            repair_spec = RepairSpec(**repair_data)
        except Exception:
            repair_spec = RepairSpec(tier=RepairTier.NOOP, action="unknown")

        # Parse incident class
        try:
            incident_class = IncidentClass(props.get("incident_class", "crash"))
        except ValueError:
            incident_class = IncidentClass.CRASH

        # Parse repair tier
        try:
            repair_tier = RepairTier(props.get("repair_tier", 0))
        except ValueError:
            repair_tier = RepairTier.NOOP

        from datetime import datetime

        def _parse_dt(val: Any) -> Any:
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(val)
                except ValueError:
                    return utc_now()
            return utc_now()

        return Antibody(
            id=props.get("id", new_id()),
            fingerprint=props.get("fingerprint", ""),
            incident_class=incident_class,
            source_system=props.get("source_system", "unknown"),
            error_pattern=props.get("error_pattern", ""),
            repair_tier=repair_tier,
            repair_spec=repair_spec,
            root_cause_description=props.get("root_cause_description", ""),
            application_count=props.get("application_count", 0),
            success_count=props.get("success_count", 0),
            failure_count=props.get("failure_count", 0),
            effectiveness=props.get("effectiveness", 1.0),
            created_at=_parse_dt(props.get("created_at")) or utc_now(),
            last_applied=_parse_dt(props.get("last_applied")),
            source_incident_id=props.get("source_incident_id", ""),
            retired=props.get("retired", False),
            generation=props.get("generation", 1),
            parent_antibody_id=props.get("parent_antibody_id"),
        )

    async def get_tier_effectiveness(
        self, source_system: str,
    ) -> dict[str, float]:
        """
        Return historical effectiveness of each repair tier for a system.

        Used by DiagnosticEngine to adjust confidence priors dynamically
        instead of relying on hardcoded values. Returns a dict mapping
        tier names to mean effectiveness (0.0–1.0), defaulting to 0.5.
        """
        from systems.thymos.types import RepairTier

        tier_stats: dict[str, list[float]] = {}
        for ab in self._all.values():
            if ab.retired:
                continue
            if ab.source_system != source_system and source_system != "*":
                continue
            tier_name = ab.repair_tier.name
            if tier_name not in tier_stats:
                tier_stats[tier_name] = []
            tier_stats[tier_name].append(ab.effectiveness)

        result: dict[str, float] = {}
        for tier in RepairTier:
            values = tier_stats.get(tier.name, [])
            if values:
                result[tier.name] = sum(values) / len(values)
            else:
                result[tier.name] = 0.5  # Neutral prior when no history
        return result

    @property
    def total(self) -> int:
        return len(self._all)

    @property
    def active_count(self) -> int:
        return sum(1 for a in self._all.values() if not a.retired)

    @property
    def retired_count(self) -> int:
        return sum(1 for a in self._all.values() if a.retired)

    @property
    def mean_effectiveness(self) -> float:
        active = [a for a in self._all.values() if not a.retired]
        if not active:
            return 1.0
        return sum(a.effectiveness for a in active) / len(active)

    async def get_all_active(self) -> list[Antibody]:
        """Get all non-retired antibodies."""
        return [a for a in self._all.values() if not a.retired]
