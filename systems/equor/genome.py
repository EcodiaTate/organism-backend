"""
EcodiaOS - Equor Genome Extraction & Seeding

Implements GenomeExtractionProtocol for the Equor (constitutional conscience)
system. Heritable state includes:

- Constitution text (version, drive weights, amendments list)
- Invariant definitions (hardcoded + community)
- Drive weights (coherence, care, growth, honesty)
- Learned evaluation thresholds (care floor, honesty floor, HITL capital,
  drift window, governance parameters)

The genome allows a child instance to inherit the parent's constitutional
identity - its moral framework, accumulated amendments, and calibrated
governance parameters - so it starts with the same ethical foundation
rather than bare defaults.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID
from primitives.genome import OrganGenomeSegment
from systems.genome_helpers import build_segment, check_schema_version, verify_segment

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()

_SYSTEM_ID: SystemID = SystemID.EQUOR


class EquorGenomeExtractor:
    """
    Extracts and seeds the Equor constitutional genome.

    Extract: reads the Constitution node, its amendments, linked Invariant
    nodes, drive weights, and governance/evaluation thresholds from Neo4j.

    Seed: recreates the Constitution node with inherited drives and
    amendments, and links Invariant nodes, in a child instance's graph.
    """

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j
        self._log = logger.bind(subsystem="equor.genome")

    # ─── GenomeExtractionProtocol ────────────────────────────────────

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        """
        Serialise Equor's heritable state into an OrganGenomeSegment.

        Returns an empty segment (version=0) if no Constitution node exists.
        """
        try:
            constitution = await self._extract_constitution()
        except Exception as exc:
            self._log.error("equor_genome_extract_constitution_failed", error=str(exc))
            return build_segment(_SYSTEM_ID, {}, version=0)

        if constitution is None:
            self._log.warning("equor_genome_no_constitution")
            return build_segment(_SYSTEM_ID, {}, version=0)

        try:
            invariants = await self._extract_invariants()
        except Exception as exc:
            self._log.error("equor_genome_extract_invariants_failed", error=str(exc))
            invariants = []

        try:
            amendments = await self._extract_amendments()
        except Exception as exc:
            self._log.error("equor_genome_extract_amendments_failed", error=str(exc))
            amendments = []

        try:
            evaluation_thresholds = await self._extract_evaluation_thresholds()
        except Exception as exc:
            self._log.error("equor_genome_extract_thresholds_failed", error=str(exc))
            evaluation_thresholds = {}

        version = int(constitution.get("version", 1))
        drive_weights = {
            "coherence": float(constitution.get("drive_coherence", 1.0)),
            "care": float(constitution.get("drive_care", 1.0)),
            "growth": float(constitution.get("drive_growth", 1.0)),
            "honesty": float(constitution.get("drive_honesty", 1.0)),
        }

        try:
            drift_history = await self._extract_drift_history()
        except Exception as exc:
            self._log.error("equor_genome_extract_drift_history_failed", error=str(exc))
            drift_history = []

        try:
            floor_thresholds = await self._extract_floor_thresholds()
        except Exception as exc:
            self._log.error("equor_genome_extract_floor_thresholds_failed", error=str(exc))
            floor_thresholds = {}

        payload: dict[str, Any] = {
            "constitution_version": version,
            "amendments": amendments,
            "invariants": invariants,
            "drive_weights": drive_weights,
            "evaluation_thresholds": evaluation_thresholds,
            "drift_history": drift_history,
            # Evolvable floor thresholds: heritable with ±10% noise at Mitosis.
            # Children inherit the parent's floor sensitivity, not bare defaults.
            "floor_thresholds": floor_thresholds,
        }

        segment = build_segment(_SYSTEM_ID, payload, version=version)

        self._log.info(
            "equor_genome_extracted",
            constitution_version=version,
            amendment_count=len(amendments),
            invariant_count=len(invariants),
            drive_weights=drive_weights,
        )

        return segment

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        """
        Restore Equor's heritable state from a parent's genome segment.

        Creates the Constitution node with inherited drive weights and
        amendments, then creates Invariant nodes linked to the Constitution.

        Returns True on success, False on any failure.
        """
        if not verify_segment(segment):
            self._log.error("equor_genome_seed_hash_mismatch")
            return False

        if not check_schema_version(segment):
            self._log.error("equor_genome_seed_schema_incompatible")
            return False

        payload = segment.payload
        if not payload:
            self._log.warning("equor_genome_seed_empty_payload")
            return False

        constitution_version = int(payload.get("constitution_version", 1))
        amendments: list[dict[str, Any]] = payload.get("amendments", [])
        invariants: list[dict[str, Any]] = payload.get("invariants", [])
        drive_weights: dict[str, float] = payload.get("drive_weights", {})
        evaluation_thresholds: dict[str, Any] = payload.get("evaluation_thresholds", {})

        coherence = float(drive_weights.get("coherence", 1.0))
        care = float(drive_weights.get("care", 1.0))
        growth = float(drive_weights.get("growth", 1.0))
        honesty = float(drive_weights.get("honesty", 1.0))

        # Seed Constitution node
        try:
            await self._seed_constitution(
                version=constitution_version,
                coherence=coherence,
                care=care,
                growth=growth,
                honesty=honesty,
                amendments=amendments,
            )
        except Exception as exc:
            self._log.error("equor_genome_seed_constitution_failed", error=str(exc))
            return False

        # Seed Invariant nodes
        invariants_seeded = 0
        for inv in invariants:
            try:
                await self._seed_invariant(inv)
                invariants_seeded += 1
            except Exception as exc:
                self._log.warning(
                    "equor_genome_seed_invariant_failed",
                    invariant_name=inv.get("name", "<unknown>"),
                    error=str(exc),
                )

        # Seed evaluation thresholds into the Constitution node
        if evaluation_thresholds:
            try:
                await self._seed_evaluation_thresholds(evaluation_thresholds)
            except Exception as exc:
                self._log.warning(
                    "equor_genome_seed_thresholds_failed",
                    error=str(exc),
                )

        # Seed floor thresholds with ±10% noise (heritable evolvable parameters).
        # Children start with the parent's floor sensitivity, not bare defaults.
        floor_thresholds: dict[str, float] = payload.get("floor_thresholds", {})
        if floor_thresholds:
            try:
                await self._seed_floor_thresholds_with_noise(floor_thresholds)
            except Exception as exc:
                self._log.warning(
                    "equor_genome_seed_floor_thresholds_failed",
                    error=str(exc),
                )

        self._log.info(
            "equor_genome_seeded",
            constitution_version=constitution_version,
            amendments_count=len(amendments),
            invariants_seeded=invariants_seeded,
            drive_weights=drive_weights,
            floor_thresholds_inherited=bool(floor_thresholds),
        )

        return True

    # ─── Extraction Helpers ──────────────────────────────────────────

    async def _extract_constitution(self) -> dict[str, Any] | None:
        """Fetch the Constitution node properties."""
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)
                RETURN c.version AS version,
                       c.drive_coherence AS drive_coherence,
                       c.drive_care AS drive_care,
                       c.drive_growth AS drive_growth,
                       c.drive_honesty AS drive_honesty,
                       c.amendments AS amendments,
                       c.last_amended AS last_amended
                """
            )
        except Exception as exc:
            self._log.error("equor_genome_query_constitution_failed", error=str(exc))
            raise

        if not rows:
            return None

        return dict(rows[0])

    async def _extract_amendments(self) -> list[dict[str, Any]]:
        """
        Extract adopted amendments from GovernanceRecord nodes.

        Amendments are stored both as a JSON list on the Constitution node
        (c.amendments) and as individual GovernanceRecord nodes. We pull
        from GovernanceRecord for richer detail, ordered by timestamp.
        """
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (g:GovernanceRecord {event_type: 'amendment_proposed'})
                WHERE g.amendment_status = 'adopted'
                RETURN g.id AS id,
                       g.details_json AS details_json,
                       g.timestamp AS timestamp,
                       g.actor AS proposer
                ORDER BY g.timestamp ASC
                """
            )
        except Exception as exc:
            self._log.error("equor_genome_query_amendments_failed", error=str(exc))
            raise

        amendments: list[dict[str, Any]] = []
        for row in rows:
            details_raw = row.get("details_json", "{}")
            try:
                details = json.loads(details_raw) if isinstance(details_raw, str) else details_raw
            except (json.JSONDecodeError, TypeError):
                details = {}

            amendments.append({
                "id": str(row.get("id", "")),
                "title": str(details.get("title", "")),
                "description": str(details.get("description", "")),
                "rationale": str(details.get("rationale", "")),
                "proposed_drives": details.get("proposed_drives", {}),
                "previous_drives": details.get("current_drives", {}),
                "proposer": str(row.get("proposer", "")),
                "timestamp": str(row.get("timestamp", "")),
            })

        return amendments

    async def _extract_invariants(self) -> list[dict[str, Any]]:
        """Extract all active invariants linked to the Constitution."""
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (c:Constitution)-[:INCLUDES_INVARIANT]->(i:Invariant)
                WHERE i.active = true
                RETURN i.id AS id,
                       i.name AS name,
                       i.description AS description,
                       i.source AS source,
                       i.severity AS severity
                ORDER BY i.source, i.name
                """
            )
        except Exception as exc:
            self._log.error("equor_genome_query_invariants_failed", error=str(exc))
            raise

        return [
            {
                "id": str(row.get("id", "")),
                "name": str(row.get("name", "")),
                "description": str(row.get("description", "")),
                "source": str(row.get("source", "hardcoded")),
                "severity": str(row.get("severity", "critical")),
            }
            for row in rows
        ]

    async def _extract_evaluation_thresholds(self) -> dict[str, Any]:
        """
        Extract learned evaluation thresholds from the Constitution node
        and governance configuration stored in Neo4j.

        These thresholds are calibrated over the organism's lifetime and
        are heritable so children start with the parent's tuned values.
        """
        thresholds: dict[str, Any] = {}

        # Constitution-level thresholds (set by amendments or Evo)
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)
                RETURN c.care_floor AS care_floor,
                       c.honesty_floor AS honesty_floor,
                       c.drift_window_size AS drift_window_size,
                       c.hitl_capital_threshold AS hitl_capital_threshold
                """
            )
            if rows:
                r = rows[0]
                if r.get("care_floor") is not None:
                    thresholds["care_floor"] = float(r["care_floor"])
                if r.get("honesty_floor") is not None:
                    thresholds["honesty_floor"] = float(r["honesty_floor"])
                if r.get("drift_window_size") is not None:
                    thresholds["drift_window_size"] = int(r["drift_window_size"])
                if r.get("hitl_capital_threshold") is not None:
                    thresholds["hitl_capital_threshold"] = float(r["hitl_capital_threshold"])
        except Exception as exc:
            self._log.warning("equor_genome_extract_constitution_thresholds_failed", error=str(exc))

        # Governance parameters (amendment pipeline calibration)
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)
                RETURN c.amendment_supermajority AS amendment_supermajority,
                       c.amendment_quorum AS amendment_quorum,
                       c.amendment_deliberation_days AS amendment_deliberation_days,
                       c.amendment_cooldown_days AS amendment_cooldown_days,
                       c.amendment_shadow_days AS amendment_shadow_days,
                       c.amendment_shadow_max_divergence_rate AS amendment_shadow_max_divergence_rate
                """
            )
            if rows:
                r = rows[0]
                governance: dict[str, Any] = {}
                for key in (
                    "amendment_supermajority",
                    "amendment_quorum",
                    "amendment_deliberation_days",
                    "amendment_cooldown_days",
                    "amendment_shadow_days",
                    "amendment_shadow_max_divergence_rate",
                ):
                    val = r.get(key)
                    if val is not None:
                        governance[key] = float(val) if isinstance(val, (int, float)) else val
                if governance:
                    thresholds["governance"] = governance
        except Exception as exc:
            self._log.warning("equor_genome_extract_governance_thresholds_failed", error=str(exc))

        return thresholds

    async def _extract_drift_history(self) -> list[dict[str, Any]]:
        """Extract recent drift reports for heritable pattern awareness.

        A child instance benefits from knowing the parent's drift patterns
        so it can start with calibrated sensitivity.
        """
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (g:GovernanceRecord {event_type: 'drift_report'})
                RETURN g.id AS id,
                       g.details_json AS details_json,
                       g.timestamp AS timestamp,
                       g.outcome AS action
                ORDER BY g.timestamp DESC
                LIMIT 20
                """
            )
        except Exception as exc:
            self._log.error("equor_genome_query_drift_history_failed", error=str(exc))
            raise

        history: list[dict[str, Any]] = []
        for row in rows:
            details_raw = row.get("details_json", "{}")
            try:
                details = json.loads(details_raw) if isinstance(details_raw, str) else details_raw
            except (json.JSONDecodeError, TypeError):
                details = {}

            history.append({
                "severity": details.get("severity", 0.0),
                "direction": details.get("direction", "unknown"),
                "action": str(row.get("action", "")),
                "timestamp": str(row.get("timestamp", "")),
            })

        return history

    async def _extract_floor_thresholds(self) -> dict[str, float]:
        """Extract evolvable floor threshold multipliers from the Constitution node.

        Returns:
            dict with keys:
              - care_floor_multiplier: float (default -0.3)
              - honesty_floor_multiplier: float (default -0.3)

        These are heritable: children inherit parent's floor sensitivity and
        apply ±10% noise at seeding time so the population can explore the
        space of constitutional strictness.
        """
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)
                RETURN c.care_floor_multiplier   AS care_floor_multiplier,
                       c.honesty_floor_multiplier AS honesty_floor_multiplier
                """
            )
        except Exception as exc:
            self._log.error("equor_genome_query_floor_thresholds_failed", error=str(exc))
            raise

        thresholds: dict[str, float] = {}
        if rows:
            r = rows[0]
            if r.get("care_floor_multiplier") is not None:
                thresholds["care_floor_multiplier"] = float(r["care_floor_multiplier"])
            if r.get("honesty_floor_multiplier") is not None:
                thresholds["honesty_floor_multiplier"] = float(r["honesty_floor_multiplier"])

        return thresholds

    # ─── Seeding Helpers ─────────────────────────────────────────────

    async def _seed_constitution(
        self,
        *,
        version: int,
        coherence: float,
        care: float,
        growth: float,
        honesty: float,
        amendments: list[dict[str, Any]],
    ) -> None:
        """
        Create or update the Constitution node with inherited state.

        Uses MERGE on the Self->Constitution relationship so this is
        idempotent if run during a birth sequence that already created
        a bare Constitution.
        """
        # Serialise amendments list for Neo4j storage
        amendments_json_list = [
            json.dumps(a, sort_keys=True, default=str) for a in amendments
        ]

        try:
            await self._neo4j.execute_write(
                """
                MATCH (s:Self)
                MERGE (s)-[:GOVERNED_BY]->(c:Constitution)
                SET c.version = $version,
                    c.drive_coherence = $coherence,
                    c.drive_care = $care,
                    c.drive_growth = $growth,
                    c.drive_honesty = $honesty,
                    c.amendments = $amendments,
                    c.source = 'parent_genome'
                """,
                {
                    "version": version,
                    "coherence": coherence,
                    "care": care,
                    "growth": growth,
                    "honesty": honesty,
                    "amendments": amendments_json_list,
                },
            )
        except Exception as exc:
            self._log.error("equor_genome_seed_constitution_write_failed", error=str(exc))
            raise

    async def _seed_invariant(self, invariant: dict[str, Any]) -> None:
        """
        Create an Invariant node and link it to the Constitution.

        Uses MERGE on invariant ID to avoid duplicates if hardcoded
        invariants were already seeded by the birth sequence.
        """
        try:
            await self._neo4j.execute_write(
                """
                MERGE (i:Invariant {id: $id})
                ON CREATE SET
                    i.name = $name,
                    i.description = $description,
                    i.source = $source,
                    i.severity = $severity,
                    i.active = true
                ON MATCH SET
                    i.description = $description,
                    i.severity = $severity,
                    i.active = true
                WITH i
                MATCH (c:Constitution)
                MERGE (c)-[:INCLUDES_INVARIANT]->(i)
                """,
                {
                    "id": invariant.get("id", ""),
                    "name": invariant.get("name", ""),
                    "description": invariant.get("description", ""),
                    "source": invariant.get("source", "hardcoded"),
                    "severity": invariant.get("severity", "critical"),
                },
            )
        except Exception as exc:
            self._log.error(
                "equor_genome_seed_invariant_write_failed",
                invariant_id=invariant.get("id"),
                error=str(exc),
            )
            raise

    async def _seed_evaluation_thresholds(self, thresholds: dict[str, Any]) -> None:
        """
        Apply inherited evaluation thresholds to the Constitution node.

        Only sets properties that are present in the thresholds dict;
        does not clear properties that are absent.
        """
        # Build a dynamic SET clause for only the keys we have
        set_clauses: list[str] = []
        params: dict[str, Any] = {}

        direct_keys = [
            "care_floor",
            "honesty_floor",
            "drift_window_size",
            "hitl_capital_threshold",
        ]
        for key in direct_keys:
            if key in thresholds:
                set_clauses.append(f"c.{key} = ${key}")
                params[key] = thresholds[key]

        # Governance sub-keys
        governance = thresholds.get("governance", {})
        if isinstance(governance, dict):
            for key, val in governance.items():
                safe_key = f"gov_{key}"
                set_clauses.append(f"c.{key} = ${safe_key}")
                params[safe_key] = val

        if not set_clauses:
            return

        query = (
            "MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)\n"
            "SET " + ", ".join(set_clauses)
        )

        try:
            await self._neo4j.execute_write(query, params)
        except Exception as exc:
            self._log.error("equor_genome_seed_thresholds_write_failed", error=str(exc))
            raise

    # ─── Spawn-Time Fragment Application (Prompt 4.1) ───────────────

    async def apply_inherited_amendments(
        self,
        equor_genome_fragment: Any,
        *,
        memory_neo4j: Any = None,
        instance_id: str = "",
    ) -> bool:
        """
        Apply an EquorGenomeFragment received at child spawn time.

        This is distinct from seed_from_genome_segment() (which restores the
        full OrganGenomeSegment from the organism-wide genome). This method
        applies the lightweight spawn-time snapshot produced by
        EquorService.export_equor_genome() and passed via CHILD_SPAWNED payload.

        Actions:
          1. Apply drive calibration deltas from the parent's amendment history
             to the live Constitution node (additive on top of defaults).
          2. Write GovernanceRecord nodes for each inherited amendment so the
             child has a traceable audit trail of its normative ancestry.
          3. Write inherited_constitutional_wisdom to Memory.Self.

        Returns True on success (partial failures are non-fatal, logged as warnings).
        """
        if equor_genome_fragment is None:
            self._log.warning("equor_apply_inherited_amendments_no_fragment")
            return False

        # Unpack the fragment (accepts EquorGenomeFragment pydantic model or plain dict)
        if hasattr(equor_genome_fragment, "model_dump"):
            fragment_dict: dict[str, Any] = equor_genome_fragment.model_dump(mode="json")
        else:
            fragment_dict = dict(equor_genome_fragment)

        top_amendments: list[dict[str, Any]] = fragment_dict.get("top_amendments", [])
        amendment_rationale: list[str] = fragment_dict.get("amendment_rationale", [])
        drive_deltas: dict[str, float] = fragment_dict.get("drive_calibration_deltas", {})
        constitution_hash: str = str(fragment_dict.get("constitution_hash", ""))
        total_adopted: int = int(fragment_dict.get("total_amendments_adopted", 0))
        genome_id: str = str(fragment_dict.get("genome_id", ""))

        self._log.info(
            "equor_applying_inherited_amendments",
            genome_id=genome_id,
            amendment_count=len(top_amendments),
            constitution_hash=constitution_hash,
            total_adopted=total_adopted,
        )

        # ── Step 1: Apply drive calibration deltas to Constitution node ──
        if drive_deltas:
            try:
                await self._apply_drive_calibration_deltas(drive_deltas)
            except Exception as exc:
                self._log.warning("equor_apply_drive_deltas_failed", error=str(exc))

        # ── Step 2: Write GovernanceRecord nodes for inherited amendments ──
        for idx, amendment in enumerate(top_amendments):
            # Attach inherited rationale from parallel list if available
            rationale = amendment_rationale[idx] if idx < len(amendment_rationale) else ""
            try:
                await self._write_inherited_amendment_record(amendment, genome_id, inherited_rationale=rationale)
            except Exception as exc:
                self._log.warning(
                    "equor_write_inherited_amendment_failed",
                    amendment_id=amendment.get("amendment_id", ""),
                    error=str(exc),
                )

        # ── Step 3: Write inherited_constitutional_wisdom to Memory.Self ──
        if memory_neo4j is not None and constitution_hash:
            try:
                await self._write_constitutional_wisdom_to_self(
                    memory_neo4j=memory_neo4j,
                    constitution_hash=constitution_hash,
                    genome_id=genome_id,
                    total_adopted=total_adopted,
                    instance_id=instance_id,
                )
            except Exception as exc:
                self._log.warning(
                    "equor_write_constitutional_wisdom_failed",
                    error=str(exc),
                )

        self._log.info(
            "equor_inherited_amendments_applied",
            genome_id=genome_id,
            amendments_written=len(top_amendments),
            drive_deltas_applied=bool(drive_deltas),
        )
        return True

    async def _apply_drive_calibration_deltas(
        self, deltas: dict[str, float],
    ) -> None:
        """Additively apply parent drive calibration deltas to the child Constitution node.

        Only updates drive dimensions present in the delta dict. Clamps to [-1.0, 1.0].
        """
        set_clauses: list[str] = []
        params: dict[str, Any] = {}

        drive_col_map = {
            "coherence": "drive_coherence",
            "care": "drive_care",
            "growth": "drive_growth",
            "honesty": "drive_honesty",
        }

        for drive, delta in deltas.items():
            col = drive_col_map.get(drive)
            if col is None:
                continue
            param_key = f"delta_{drive}"
            params[param_key] = float(delta)
            # Add the delta to the current weight, clamp to [-1.0, 1.0]
            set_clauses.append(
                f"c.{col} = CASE WHEN c.{col} IS NULL THEN toFloat(${param_key}) "
                f"ELSE min(1.0, max(-1.0, c.{col} + toFloat(${param_key}))) END"
            )

        if not set_clauses:
            return

        query = (
            "MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)\n"
            "SET " + ", ".join(set_clauses) + ",\n"
            "    c.inherited_genome_id = $genome_id,\n"
            "    c.inheritance_applied_at = datetime()"
        )
        params["genome_id"] = ""  # set at call site if needed
        await self._neo4j.execute_write(query, params)

    async def _write_inherited_amendment_record(
        self,
        amendment: dict[str, Any],
        genome_id: str,
        *,
        inherited_rationale: str = "",
    ) -> None:
        """Persist an inherited amendment as a GovernanceRecord in the child's graph.

        Args:
            amendment: Amendment snapshot dict from the parent's genome fragment.
            genome_id: Parent genome ID for lineage tracking.
            inherited_rationale: Plain-text rationale from the parent's
                amendment_rationale list (parallel to top_amendments). Used to
                enrich the GovernanceRecord with the reasoning behind adoption.
        """
        import json as _json

        # Prefer the per-amendment rationale from the snapshot; fall back to
        # the parallel inherited_rationale list from the fragment
        rationale = amendment.get("rationale", "") or inherited_rationale

        details = {
            "title": amendment.get("title", ""),
            "description": amendment.get("description", ""),
            "rationale": rationale,
            "inherited_rationale": inherited_rationale,
            "proposed_drives": amendment.get("proposed_drives", {}),
            "current_drives": amendment.get("previous_drives", {}),
            "inherited": True,
            "parent_genome_id": genome_id,
        }
        await self._neo4j.execute_write(
            """
            MERGE (g:GovernanceRecord {id: $id})
            ON CREATE SET
                g.event_type    = 'amendment_proposed',
                g.amendment_status = 'adopted',
                g.actor         = $proposer,
                g.timestamp     = $adopted_at,
                g.details_json  = $details_json,
                g.inherited     = true,
                g.source_genome = $genome_id
            """,
            {
                "id": f"inherited_{amendment.get('amendment_id', '')}_{genome_id[:8]}",
                "proposer": amendment.get("proposer", "parent_genome"),
                "adopted_at": amendment.get("adopted_at", ""),
                "details_json": _json.dumps(details, sort_keys=True, default=str),
                "genome_id": genome_id,
            },
        )

    async def _write_constitutional_wisdom_to_self(
        self,
        *,
        memory_neo4j: Any,
        constitution_hash: str,
        genome_id: str,
        total_adopted: int,
        instance_id: str,
    ) -> None:
        """Write inherited_constitutional_wisdom field to Memory.Self node."""
        await memory_neo4j.execute_write(
            """
            MATCH (s:Self)
            WHERE s.instance_id = $instance_id OR $instance_id = ''
            SET s.inherited_constitutional_wisdom = $constitution_hash,
                s.inherited_equor_genome_id       = $genome_id,
                s.inherited_amendment_count       = $total_adopted,
                s.constitutional_lineage_at       = datetime()
            """,
            {
                "instance_id": instance_id,
                "constitution_hash": constitution_hash,
                "genome_id": genome_id,
                "total_adopted": total_adopted,
            },
        )

    async def _seed_floor_thresholds_with_noise(
        self, floor_thresholds: dict[str, float],
    ) -> None:
        """Apply inherited floor multipliers with ±10% mutation noise.

        Floor thresholds are evolvable: children inherit the parent's strictness
        but with small random perturbation so the population can drift toward
        stricter or more permissive floors over generations.

        Noise model: uniform ±10% of the absolute value of the parent threshold.
        The result is clamped to [-1.0, 0.0] - floors are always negative and
        cannot exceed -1.0 (total block) or go above 0.0 (meaningless).
        """
        import random

        params: dict[str, float] = {}
        set_clauses: list[str] = []

        for key, parent_val in floor_thresholds.items():
            if key not in ("care_floor_multiplier", "honesty_floor_multiplier"):
                continue  # Only seed known evolvable floor keys
            noise = random.uniform(-0.1, 0.1) * abs(parent_val)
            child_val = max(-1.0, min(0.0, parent_val + noise))
            params[key] = round(child_val, 4)
            set_clauses.append(f"c.{key} = ${key}")

        if not set_clauses:
            return

        query = (
            "MATCH (s:Self)-[:GOVERNED_BY]->(c:Constitution)\n"
            "SET " + ", ".join(set_clauses)
        )

        try:
            await self._neo4j.execute_write(query, params)
            self._log.info(
                "equor_genome_floor_thresholds_seeded_with_noise",
                parent_values=floor_thresholds,
                child_values=params,
            )
        except Exception as exc:
            self._log.error("equor_genome_seed_floor_thresholds_write_failed", error=str(exc))
            raise
