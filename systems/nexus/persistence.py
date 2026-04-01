"""
EcodiaOS - Nexus: Neo4j Persistence Layer

Persists speciation registry state, epistemic level promotions,
converged invariants, world model fragments, and instance divergence
profiles to Neo4j so they survive restarts.

Node types:
  (:NexusSpeciationEvent {id, instance_a, instance_b, divergence_score, ...})
  (:NexusCognitiveKind {kind_id, founding_instances, created_at})
  (:NexusEpistemicPromotion {fragment_id, from_level, to_level, evidence, ...})
  (:NexusBridgeSurvivor {fragment_id, marked_at})
  (:NexusFragment {fragment_id, source_instance_id, abstract_structure_json,
                   domain_labels, compression_ratio, observations_explained,
                   sleep_certified, source_divergence_score, created_at})
  (:NexusConvergedInvariant {id, invariant_a_id, invariant_b_id,
                              source_instance_a, source_instance_b,
                              abstract_form_json, triangulation_confidence,
                              converged_at})
  (:NexusDivergenceProfile {instance_id, domain_coverage_json,
                             structural_fingerprint, total_experiences,
                             total_schemas, captured_at})

All writes are batched via UNWIND for efficiency.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import utc_now
from systems.nexus.types import (
    CognitiveKindEntry,
    CompressionPath,
    ConvergedInvariant,
    EpistemicLevel,
    InstanceDivergenceProfile,
    ShareableWorldModelFragment,
    SleepCertification,
    SpeciationEvent,
    SpeciationRegistryState,
    TriangulationMetadata,
    TriangulationSource,
)

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger("nexus.persistence")


class NexusPersistence:
    """
    Neo4j persistence for Nexus epistemic triangulation state.

    All writes are batched - call persist_*() after each speciation
    check / promotion cycle, not per-event.
    """

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j

    # ─── Write: Speciation Events ────────────────────────────────

    async def persist_speciation_events(
        self, events: list[SpeciationEvent]
    ) -> int:
        """Batch-upsert speciation events. Returns count written."""
        if not events:
            return 0

        rows = [
            {
                "id": e.id,
                "instance_a": e.instance_a_id,
                "instance_b": e.instance_b_id,
                "divergence_score": e.divergence_score,
                "shared_invariant_count": e.shared_invariant_count,
                "incompatible_schema_count": e.incompatible_schema_count,
                "new_cognitive_kind_registered": e.new_cognitive_kind_registered,
                "timestamp": e.timestamp.isoformat(),
                # Gap MEDIUM-7 (Federation Spec, 2026-03-07)
                "genome_distance": e.genome_distance,
                "is_new_species": e.is_new_species,
            }
            for e in events
        ]

        query = """
        UNWIND $rows AS r
        MERGE (e:NexusSpeciationEvent {id: r.id})
        SET e.instance_a = r.instance_a,
            e.instance_b = r.instance_b,
            e.divergence_score = r.divergence_score,
            e.shared_invariant_count = r.shared_invariant_count,
            e.incompatible_schema_count = r.incompatible_schema_count,
            e.new_cognitive_kind_registered = r.new_cognitive_kind_registered,
            e.timestamp = r.timestamp,
            e.genome_distance = r.genome_distance,
            e.is_new_species = r.is_new_species,
            e.ingestion_time = datetime()
        """
        await self._neo4j.execute_write(query, {"rows": rows})
        logger.debug("speciation_events_persisted", count=len(rows))
        return len(rows)

    # ─── Write: Cognitive Kinds ──────────────────────────────────

    async def persist_cognitive_kinds(
        self, kinds: list[CognitiveKindEntry]
    ) -> int:
        """Batch-upsert cognitive kinds. Returns count written."""
        if not kinds:
            return 0

        rows = [
            {
                "kind_id": k.kind_id,
                "member_instance_ids": k.member_instance_ids,
                "founding_speciation_event_id": k.founding_speciation_event_id,
                "established_at": k.established_at.isoformat(),
            }
            for k in kinds
        ]

        query = """
        UNWIND $rows AS r
        MERGE (k:NexusCognitiveKind {kind_id: r.kind_id})
        SET k.member_instance_ids = r.member_instance_ids,
            k.founding_speciation_event_id = r.founding_speciation_event_id,
            k.established_at = r.established_at,
            k.ingestion_time = datetime()
        """
        await self._neo4j.execute_write(query, {"rows": rows})
        logger.debug("cognitive_kinds_persisted", count=len(rows))
        return len(rows)

    # ─── Write: Epistemic Level Promotions ───────────────────────

    async def persist_epistemic_promotions(
        self,
        promotions: list[dict[str, Any]],
    ) -> int:
        """
        Batch-write epistemic level promotions.

        Each dict: {fragment_id, from_level, to_level, evidence, timestamp}
        """
        if not promotions:
            return 0

        query = """
        UNWIND $rows AS r
        MERGE (p:NexusEpistemicPromotion {fragment_id: r.fragment_id, to_level: r.to_level})
        SET p.from_level = r.from_level,
            p.evidence = r.evidence,
            p.timestamp = r.timestamp,
            p.ingestion_time = datetime()
        """
        await self._neo4j.execute_write(query, {"rows": promotions})
        logger.debug("epistemic_promotions_persisted", count=len(promotions))
        return len(promotions)

    # ─── Write: Bridge Survivors ─────────────────────────────────

    async def persist_bridge_survivors(
        self, fragment_ids: list[str]
    ) -> int:
        """Batch-upsert bridge survivor markers."""
        if not fragment_ids:
            return 0

        rows = [
            {"fragment_id": fid, "marked_at": utc_now().isoformat()}
            for fid in fragment_ids
        ]

        query = """
        UNWIND $rows AS r
        MERGE (s:NexusBridgeSurvivor {fragment_id: r.fragment_id})
        SET s.marked_at = r.marked_at,
            s.ingestion_time = datetime()
        """
        await self._neo4j.execute_write(query, {"rows": rows})
        logger.debug("bridge_survivors_persisted", count=len(rows))
        return len(rows)

    # ─── Read: Restore on Startup ────────────────────────────────

    async def load_speciation_events(self) -> list[SpeciationEvent]:
        """Load all speciation events from Neo4j."""
        query = """
        MATCH (e:NexusSpeciationEvent)
        RETURN e ORDER BY e.timestamp
        """
        records = await self._neo4j.execute_read(query)
        events: list[SpeciationEvent] = []
        for record in records:
            node = record.get("e", {})
            events.append(SpeciationEvent(
                id=node.get("id", ""),
                instance_a_id=node.get("instance_a", ""),
                instance_b_id=node.get("instance_b", ""),
                divergence_score=float(node.get("divergence_score", 0.0)),
                shared_invariant_count=int(node.get("shared_invariant_count", 0)),
                incompatible_schema_count=int(node.get("incompatible_schema_count", 0)),
                new_cognitive_kind_registered=bool(node.get("new_cognitive_kind_registered", False)),
                timestamp=datetime.fromisoformat(node["timestamp"])
                if "timestamp" in node else utc_now(),
            ))
        logger.info("speciation_events_loaded", count=len(events))
        return events

    async def load_cognitive_kinds(self) -> list[CognitiveKindEntry]:
        """Load all cognitive kinds from Neo4j."""
        query = """
        MATCH (k:NexusCognitiveKind)
        RETURN k ORDER BY k.established_at
        """
        records = await self._neo4j.execute_read(query)
        kinds: list[CognitiveKindEntry] = []
        for record in records:
            node = record.get("k", {})
            kinds.append(CognitiveKindEntry(
                kind_id=node.get("kind_id", ""),
                member_instance_ids=node.get("member_instance_ids", []),
                founding_speciation_event_id=node.get("founding_speciation_event_id", ""),
                established_at=datetime.fromisoformat(node["established_at"])
                if "established_at" in node else utc_now(),
            ))
        logger.info("cognitive_kinds_loaded", count=len(kinds))
        return kinds

    async def load_epistemic_levels(self) -> dict[str, EpistemicLevel]:
        """Load fragment epistemic levels (latest promotion per fragment)."""
        query = """
        MATCH (p:NexusEpistemicPromotion)
        WITH p.fragment_id AS fid, max(p.to_level) AS max_level
        RETURN fid, max_level
        """
        records = await self._neo4j.execute_read(query)
        levels: dict[str, EpistemicLevel] = {}
        for record in records:
            fid = record.get("fid", "")
            level_val = int(record.get("max_level", 0))
            if fid:
                levels[fid] = EpistemicLevel(level_val)
        logger.info("epistemic_levels_loaded", count=len(levels))
        return levels

    async def load_bridge_survivors(self) -> set[str]:
        """Load all bridge survivor fragment IDs."""
        query = """
        MATCH (s:NexusBridgeSurvivor)
        RETURN s.fragment_id AS fid
        """
        records = await self._neo4j.execute_read(query)
        survivors = {r["fid"] for r in records if r.get("fid")}
        logger.info("bridge_survivors_loaded", count=len(survivors))
        return survivors

    # ─── Write: Fragments (R1) ───────────────────────────────────

    async def persist_fragments(
        self, fragments: list[ShareableWorldModelFragment]
    ) -> int:
        """
        Batch-upsert local world model fragments.

        Serialises abstract_structure and triangulation sources as JSON so
        Neo4j can store them as string properties (no nested objects).
        Returns count written.

        Spec §XI Memory Writes - previously in-memory only (R1).
        """
        if not fragments:
            return 0

        rows = [
            {
                "fragment_id": f.fragment_id,
                "source_instance_id": f.source_instance_id,
                "source_divergence_score": f.source_instance_divergence_score,
                "abstract_structure_json": json.dumps(f.abstract_structure),
                "domain_labels": f.domain_labels,
                "observations_explained": f.observations_explained,
                "description_length": f.description_length,
                "compression_ratio": f.compression_ratio,
                "survived_slow_wave": f.sleep_certification.survived_slow_wave,
                "survived_rem": f.sleep_certification.survived_rem,
                "sleep_cycles_survived": f.sleep_certification.sleep_cycles_survived,
                "triangulation_sources_json": json.dumps(
                    [s.model_dump(mode="json") for s in f.triangulation.independent_sources]
                ),
                "created_at": f.created_at.isoformat(),
                "last_confirmed_at": f.last_confirmed_at.isoformat(),
            }
            for f in fragments
        ]

        query = """
        UNWIND $rows AS r
        MERGE (f:NexusFragment {fragment_id: r.fragment_id})
        SET f.source_instance_id = r.source_instance_id,
            f.source_divergence_score = r.source_divergence_score,
            f.abstract_structure_json = r.abstract_structure_json,
            f.domain_labels = r.domain_labels,
            f.observations_explained = r.observations_explained,
            f.description_length = r.description_length,
            f.compression_ratio = r.compression_ratio,
            f.survived_slow_wave = r.survived_slow_wave,
            f.survived_rem = r.survived_rem,
            f.sleep_cycles_survived = r.sleep_cycles_survived,
            f.triangulation_sources_json = r.triangulation_sources_json,
            f.created_at = r.created_at,
            f.last_confirmed_at = r.last_confirmed_at,
            f.ingestion_time = datetime()
        """
        await self._neo4j.execute_write(query, {"rows": rows})
        logger.debug("fragments_persisted", count=len(rows))
        return len(rows)

    async def load_fragments(self) -> list[ShareableWorldModelFragment]:
        """
        Load all persisted local fragments from Neo4j.

        Restores abstract_structure and triangulation sources from JSON.
        Spec §XI Memory Writes (R1).
        """
        query = """
        MATCH (f:NexusFragment)
        RETURN f ORDER BY f.created_at
        """
        records = await self._neo4j.execute_read(query)
        fragments: list[ShareableWorldModelFragment] = []
        for record in records:
            node = record.get("f", {})
            try:
                abstract_structure: dict[str, Any] = json.loads(
                    node.get("abstract_structure_json", "{}")
                )
                sources_raw: list[dict[str, Any]] = json.loads(
                    node.get("triangulation_sources_json", "[]")
                )
                sources = [TriangulationSource(**s) for s in sources_raw]

                fragments.append(
                    ShareableWorldModelFragment(
                        fragment_id=node.get("fragment_id", ""),
                        source_instance_id=node.get("source_instance_id", ""),
                        source_instance_divergence_score=float(
                            node.get("source_divergence_score", 0.0)
                        ),
                        abstract_structure=abstract_structure,
                        domain_labels=list(node.get("domain_labels", [])),
                        observations_explained=int(
                            node.get("observations_explained", 0)
                        ),
                        description_length=float(node.get("description_length", 0.0)),
                        compression_ratio=float(node.get("compression_ratio", 0.0)),
                        sleep_certification=SleepCertification(
                            survived_slow_wave=bool(node.get("survived_slow_wave", False)),
                            survived_rem=bool(node.get("survived_rem", False)),
                            sleep_cycles_survived=int(node.get("sleep_cycles_survived", 0)),
                        ),
                        triangulation=TriangulationMetadata(
                            independent_sources=sources
                        ),
                        created_at=datetime.fromisoformat(node["created_at"])
                        if "created_at" in node
                        else utc_now(),
                        last_confirmed_at=datetime.fromisoformat(
                            node["last_confirmed_at"]
                        )
                        if "last_confirmed_at" in node
                        else utc_now(),
                    )
                )
            except Exception:
                logger.warning(
                    "fragment_restore_failed",
                    fragment_id=node.get("fragment_id", "?"),
                    exc_info=True,
                )
        logger.info("fragments_loaded", count=len(fragments))
        return fragments

    # ─── Write: Converged Invariants (R1) ────────────────────────

    async def persist_converged_invariants(
        self, invariants: list[ConvergedInvariant]
    ) -> int:
        """
        Batch-upsert converged invariants from InvariantBridge exchanges.

        These represent the highest-confidence ground truth candidates -
        structures that two speciated (alien-kind) instances independently
        arrived at. Previously created but never written (R1).
        Spec §XI Memory Writes.
        """
        if not invariants:
            return 0

        rows = [
            {
                "id": ci.id,
                "invariant_a_id": ci.invariant_a_id,
                "invariant_b_id": ci.invariant_b_id,
                "source_instance_a": ci.source_instance_a,
                "source_instance_b": ci.source_instance_b,
                "abstract_form_json": json.dumps(ci.abstract_form),
                "triangulation_confidence": ci.triangulation_confidence,
                "is_ground_truth_candidate": ci.is_ground_truth_candidate,
                "converged_at": ci.converged_at.isoformat(),
            }
            for ci in invariants
        ]

        query = """
        UNWIND $rows AS r
        MERGE (ci:NexusConvergedInvariant {id: r.id})
        SET ci.invariant_a_id = r.invariant_a_id,
            ci.invariant_b_id = r.invariant_b_id,
            ci.source_instance_a = r.source_instance_a,
            ci.source_instance_b = r.source_instance_b,
            ci.abstract_form_json = r.abstract_form_json,
            ci.triangulation_confidence = r.triangulation_confidence,
            ci.is_ground_truth_candidate = r.is_ground_truth_candidate,
            ci.converged_at = r.converged_at,
            ci.ingestion_time = datetime()
        """
        await self._neo4j.execute_write(query, {"rows": rows})
        logger.debug("converged_invariants_persisted", count=len(rows))
        return len(rows)

    async def load_converged_invariants(self) -> list[ConvergedInvariant]:
        """Load all persisted converged invariants from Neo4j."""
        query = """
        MATCH (ci:NexusConvergedInvariant)
        RETURN ci ORDER BY ci.converged_at
        """
        records = await self._neo4j.execute_read(query)
        invariants: list[ConvergedInvariant] = []
        for record in records:
            node = record.get("ci", {})
            try:
                abstract_form: dict[str, Any] = json.loads(
                    node.get("abstract_form_json", "{}")
                )
                invariants.append(
                    ConvergedInvariant(
                        id=node.get("id", ""),
                        invariant_a_id=node.get("invariant_a_id", ""),
                        invariant_b_id=node.get("invariant_b_id", ""),
                        source_instance_a=node.get("source_instance_a", ""),
                        source_instance_b=node.get("source_instance_b", ""),
                        abstract_form=abstract_form,
                        triangulation_confidence=float(
                            node.get("triangulation_confidence", 0.95)
                        ),
                        is_ground_truth_candidate=bool(
                            node.get("is_ground_truth_candidate", True)
                        ),
                        converged_at=datetime.fromisoformat(node["converged_at"])
                        if "converged_at" in node
                        else utc_now(),
                    )
                )
            except Exception:
                logger.warning(
                    "converged_invariant_restore_failed",
                    id=node.get("id", "?"),
                    exc_info=True,
                )
        logger.info("converged_invariants_loaded", count=len(invariants))
        return invariants

    # ─── Write: Divergence Profiles (R5) ─────────────────────────

    async def persist_divergence_profiles(
        self, profiles: list[InstanceDivergenceProfile]
    ) -> int:
        """
        Batch-upsert remote instance divergence profiles.

        Enables cross-session divergence tracking - without this, every
        restart loses all divergence history and measurements restart from
        zero. Spec §XI Memory Reads (R5).
        """
        if not profiles:
            return 0

        rows = [
            {
                "instance_id": p.instance_id,
                "domain_coverage_json": json.dumps(p.domain_coverage),
                "structural_fingerprint": p.structural_fingerprint,
                "attention_weights_json": json.dumps(p.attention_weights),
                "active_hypothesis_ids_json": json.dumps(p.active_hypothesis_ids),
                "born_at": p.born_at.isoformat(),
                "total_experiences": p.total_experiences,
                "total_schemas": p.total_schemas,
                "captured_at": p.captured_at.isoformat(),
            }
            for p in profiles
        ]

        query = """
        UNWIND $rows AS r
        MERGE (p:NexusDivergenceProfile {instance_id: r.instance_id})
        SET p.domain_coverage_json = r.domain_coverage_json,
            p.structural_fingerprint = r.structural_fingerprint,
            p.attention_weights_json = r.attention_weights_json,
            p.active_hypothesis_ids_json = r.active_hypothesis_ids_json,
            p.born_at = r.born_at,
            p.total_experiences = r.total_experiences,
            p.total_schemas = r.total_schemas,
            p.captured_at = r.captured_at,
            p.ingestion_time = datetime()
        """
        await self._neo4j.execute_write(query, {"rows": rows})
        logger.debug("divergence_profiles_persisted", count=len(rows))
        return len(rows)

    async def load_divergence_profiles(self) -> list[InstanceDivergenceProfile]:
        """
        Load all persisted remote divergence profiles from Neo4j.

        Profiles are ordered by captured_at so the most recent snapshot
        for each instance is used on conflict (dict key = instance_id).
        Spec §XI Memory Reads (R5).
        """
        query = """
        MATCH (p:NexusDivergenceProfile)
        RETURN p ORDER BY p.captured_at
        """
        records = await self._neo4j.execute_read(query)
        profiles: list[InstanceDivergenceProfile] = []
        for record in records:
            node = record.get("p", {})
            try:
                profiles.append(
                    InstanceDivergenceProfile(
                        instance_id=node.get("instance_id", ""),
                        domain_coverage=json.loads(
                            node.get("domain_coverage_json", "[]")
                        ),
                        structural_fingerprint=node.get("structural_fingerprint", ""),
                        attention_weights=json.loads(
                            node.get("attention_weights_json", "{}")
                        ),
                        active_hypothesis_ids=json.loads(
                            node.get("active_hypothesis_ids_json", "[]")
                        ),
                        born_at=datetime.fromisoformat(node["born_at"])
                        if "born_at" in node
                        else utc_now(),
                        total_experiences=int(node.get("total_experiences", 0)),
                        total_schemas=int(node.get("total_schemas", 0)),
                        captured_at=datetime.fromisoformat(node["captured_at"])
                        if "captured_at" in node
                        else utc_now(),
                    )
                )
            except Exception:
                logger.warning(
                    "divergence_profile_restore_failed",
                    instance_id=node.get("instance_id", "?"),
                    exc_info=True,
                )
        logger.info("divergence_profiles_loaded", count=len(profiles))
        return profiles

    # ─── Full Restore ────────────────────────────────────────────

    async def restore_full_state(
        self,
    ) -> tuple[
        list[SpeciationEvent],
        list[CognitiveKindEntry],
        dict[str, EpistemicLevel],
        set[str],
        list[ShareableWorldModelFragment],
        list[ConvergedInvariant],
        list[InstanceDivergenceProfile],
    ]:
        """
        Restore all Nexus persistent state from Neo4j.

        Returns (speciation_events, cognitive_kinds, epistemic_levels,
                 bridge_survivors, fragments, converged_invariants,
                 divergence_profiles).
        """
        events = await self.load_speciation_events()
        kinds = await self.load_cognitive_kinds()
        levels = await self.load_epistemic_levels()
        survivors = await self.load_bridge_survivors()
        fragments = await self.load_fragments()
        invariants = await self.load_converged_invariants()
        profiles = await self.load_divergence_profiles()
        return events, kinds, levels, survivors, fragments, invariants, profiles
