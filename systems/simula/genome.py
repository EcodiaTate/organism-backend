"""
EcodiaOS - Simula Genetic Memory: Evolution Genome Extraction & Seeding

Implements the mechanism to serialize a parent's accumulated Simula
knowledge into a SimulaGenome and deserialize it into a child instance.

SimulaGenomeExtractor:
    Scans Neo4j for successful mutations, LILO abstractions, GRPO training
    data, EFE calibration, and category analytics. Compresses into a
    SimulaGenome and persists as a :SimulaGenome node.

SimulaGenomeSeeder:
    Decompresses a parent's SimulaGenome and seeds the child's Simula
    subsystems: pre-populates the history manager with ancestor records,
    loads library abstractions, injects training data, and calibrates
    risk thresholds from inherited analytics.

Integration points:
    - MitosisEngine.build_seed_config(): extract genome before spawning
    - birth.py / seed_simula_from_parent_genome(): seed after birth
    - Axon SpawnChildExecutor: transmit genome reference in env vars
"""

from __future__ import annotations

import base64
import json
import time
import zlib
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now
from systems.simula.genome_types import (
    CategoryAnalyticsRecord,
    EFECalibrationPoint,
    GRPOTrainingExample,
    LibraryAbstractionRecord,
    MutationRecord,
    ProcedureRecord,
    SimulaGenome,
    SimulaGenomeExtractionResult,
    SimulaGenomeSeedingResult,
)

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger()

# ─── Compression ─────────────────────────────────────────────────────────────

try:
    import lz4.frame as lz4_frame

    _HAS_LZ4 = True
except ImportError:
    _HAS_LZ4 = False


def _compress_genome(genome: SimulaGenome) -> tuple[str, str, int]:
    """
    Serialize and compress a SimulaGenome into a base64 string.

    Returns (base64_payload, compression_method, raw_byte_count).
    """
    payload = json.dumps(
        genome.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    if _HAS_LZ4:
        compressed = lz4_frame.compress(payload)
        method = "lz4"
    else:
        compressed = zlib.compress(payload, level=6)
        method = "zlib"

    encoded = base64.b64encode(compressed).decode("ascii")
    return encoded, method, len(compressed)


def _decompress_genome(payload_b64: str, method: str) -> SimulaGenome:
    """
    Decompress a base64-encoded SimulaGenome string.

    Raises ValueError on format or decompression errors.
    """
    raw = base64.b64decode(payload_b64)

    if method == "lz4":
        if not _HAS_LZ4:
            raise ValueError("lz4 module not available; cannot decompress genome")
        decompressed = lz4_frame.decompress(raw)
    elif method == "zlib":
        decompressed = zlib.decompress(raw)
    else:
        raise ValueError(f"Unknown compression method: {method}")

    data = json.loads(decompressed)
    if not isinstance(data, dict):
        raise ValueError("SimulaGenome payload is not a dict")

    return SimulaGenome.model_validate(data)


# ─── Extraction Thresholds ───────────────────────────────────────────────────

# Minimum successful (non-rolled-back) mutations to include in genome
_MIN_MUTATIONS_FOR_GENOME: int = 5

# Maximum records per segment (prevents genome bloat)
_MAX_MUTATIONS: int = 500
_MAX_ABSTRACTIONS: int = 200
_MAX_TRAINING_EXAMPLES: int = 300
_MAX_CALIBRATION_POINTS: int = 100
_MAX_PROCEDURES: int = 100


# ─── Genome Extractor ───────────────────────────────────────────────────────


class SimulaGenomeExtractor:
    """
    Extracts the Simula evolution genome from Neo4j.

    Scans the immutable evolution history, LILO library, GRPO training
    data, EFE calibration records, and category analytics. Compresses
    everything into a SimulaGenome and persists it.

    Call during mitosis preparation, after MitosisEngine confirms
    reproductive fitness but before spawning the child container.
    """

    def __init__(self, neo4j: Neo4jClient, instance_id: str) -> None:
        self._neo4j = neo4j
        self._instance_id = instance_id
        self._log = logger.bind(subsystem="simula.genome.extractor")

        from systems.synapse.sentinel import ErrorSentinel
        self._sentinel = ErrorSentinel("simula.genome.extractor")

    async def is_eligible(self) -> bool:
        """Check whether this instance has enough evolution history."""
        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (e:EvolutionRecord)
                WHERE e.rolled_back = false
                RETURN count(e) AS applied_count
                """,
            )
            count = int(records[0].get("applied_count", 0)) if records else 0
            if count < _MIN_MUTATIONS_FOR_GENOME:
                self._log.debug(
                    "simula_genome_not_eligible",
                    reason="insufficient_mutations",
                    count=count,
                    required=_MIN_MUTATIONS_FOR_GENOME,
                )
                return False
            return True
        except Exception as exc:
            self._log.warning("simula_genome_eligibility_failed", error=str(exc))
            await self._sentinel.report(
                exc, context={"operation": "genome_eligibility_check"},
            )
            return False

    async def extract_genome(
        self,
        parent_ids: list[str] | None = None,
        generation: int = 1,
    ) -> tuple[SimulaGenome | None, SimulaGenomeExtractionResult]:
        """
        Extract the full Simula genome from Neo4j.

        Returns (genome, result). Genome is None if not eligible.
        """
        start = time.monotonic()
        result = SimulaGenomeExtractionResult()

        if not await self.is_eligible():
            result.duration_ms = int((time.monotonic() - start) * 1000)
            return None, result

        # Extract all segments
        mutations = await self._extract_mutations()
        abstractions = await self._extract_abstractions()
        training_examples = await self._extract_training_data()
        calibration = await self._extract_efe_calibration()
        analytics = await self._extract_category_analytics()
        procedures = await self._extract_procedures()
        metadata = await self._extract_metadata()
        dafny_specs = await self._extract_dafny_specifications()
        lean_lemmas = await self._extract_lean_lemma_catalog()
        router_weights = await self._extract_reasoning_router_weights()
        proof_heuristics = await self._extract_proof_search_heuristics()
        inspector_calibration = await self._extract_inspector_confidence_calibration()

        result.mutations_included = len(mutations)
        result.abstractions_included = len(abstractions)
        result.training_examples_included = len(training_examples)
        result.calibration_points_included = len(calibration)
        result.procedures_included = len(procedures)

        if not mutations:
            result.duration_ms = int((time.monotonic() - start) * 1000)
            return None, result

        genome_id = new_id()
        genome = SimulaGenome(
            id=genome_id,
            parent_instance_id=self._instance_id,
            generation=generation,
            parent_ids=parent_ids or [self._instance_id],
            mutations=mutations,
            library_abstractions=abstractions,
            grpo_training_examples=training_examples,
            efe_calibration=calibration,
            category_analytics=analytics,
            procedures=procedures,
            dafny_specifications=dafny_specs,
            lean_lemma_catalog=lean_lemmas,
            reasoning_router_weights=router_weights,
            proof_search_heuristics=proof_heuristics,
            inspector_confidence_calibration=inspector_calibration,
            total_proposals_processed=metadata.get("total_processed", 0),
            total_proposals_applied=metadata.get("total_applied", 0),
            total_rollbacks=metadata.get("total_rollbacks", 0),
            mean_constitutional_alignment=metadata.get("mean_alignment", 0.0),
            evolution_velocity=metadata.get("velocity", 0.0),
            config_version_at_extraction=metadata.get("config_version", 0),
        )

        # Compress and persist
        payload_b64, method, byte_count = _compress_genome(genome)
        genome.compression_method = method
        genome.genome_size_bytes = byte_count
        result.genome_size_bytes = byte_count

        await self._persist_genome(genome, payload_b64)

        result.genome_id = genome_id
        result.duration_ms = int((time.monotonic() - start) * 1000)

        self._log.info(
            "simula_genome_extracted",
            genome_id=genome_id,
            mutations=len(mutations),
            abstractions=len(abstractions),
            training_examples=len(training_examples),
            genome_bytes=byte_count,
            compression=method,
            generation=generation,
        )

        return genome, result

    # ─── Segment Extractors ──────────────────────────────────────────────────

    async def _extract_mutations(self) -> list[MutationRecord]:
        """Extract successful (non-rolled-back) evolution records."""
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (e:EvolutionRecord)
                WHERE e.rolled_back = false
                RETURN e.proposal_id AS proposal_id,
                       e.category AS category,
                       e.description AS description,
                       e.files_changed AS files_changed,
                       e.simulation_risk AS risk,
                       e.constitutional_alignment AS alignment,
                       e.counterfactual_regression_rate AS regression,
                       e.formal_verification_status AS fv_status,
                       e.applied_at AS applied_at
                ORDER BY e.applied_at DESC
                LIMIT $limit
                """,
                {"limit": _MAX_MUTATIONS},
            )
        except Exception as exc:
            self._log.warning("genome_extract_mutations_failed", error=str(exc))
            await self._sentinel.report(
                exc, context={"operation": "extract_mutations"},
            )
            return []

        records: list[MutationRecord] = []
        for row in rows:
            files = row.get("files_changed", [])
            if isinstance(files, str):
                files = [files]

            records.append(MutationRecord(
                proposal_id=str(row.get("proposal_id", "")),
                category=str(row.get("category", "")),
                description=str(row.get("description", "")),
                files_changed=files if isinstance(files, list) else [],
                risk_level=str(row.get("risk", "low")),
                constitutional_alignment=float(row.get("alignment", 0.0)),
                regression_rate=float(row.get("regression", 0.0)),
                formal_verification_status=str(row.get("fv_status", "")),
            ))
        return records

    async def _extract_abstractions(self) -> list[LibraryAbstractionRecord]:
        """Extract LILO library abstractions."""
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (a:LibraryAbstraction)
                RETURN a.name AS name,
                       a.kind AS kind,
                       a.description AS description,
                       a.signature AS signature,
                       a.source_code AS source_code,
                       a.usage_count AS usage_count,
                       a.confidence AS confidence,
                       a.tags AS tags
                ORDER BY a.usage_count * a.confidence DESC
                LIMIT $limit
                """,
                {"limit": _MAX_ABSTRACTIONS},
            )
        except Exception as exc:
            self._log.warning("genome_extract_abstractions_failed", error=str(exc))
            return []

        records: list[LibraryAbstractionRecord] = []
        for row in rows:
            tags_raw = row.get("tags", [])
            if isinstance(tags_raw, str):
                try:
                    tags_raw = json.loads(tags_raw)
                except (json.JSONDecodeError, TypeError):
                    tags_raw = []

            records.append(LibraryAbstractionRecord(
                name=str(row.get("name", "")),
                kind=str(row.get("kind", "")),
                description=str(row.get("description", "")),
                signature=str(row.get("signature", "")),
                source_code=str(row.get("source_code", "")),
                usage_count=int(row.get("usage_count", 0)),
                confidence=float(row.get("confidence", 0.5)),
                tags=tags_raw if isinstance(tags_raw, list) else [],
            ))
        return records

    async def _extract_training_data(self) -> list[GRPOTrainingExample]:
        """Extract high-quality training examples from evolution history."""
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (e:EvolutionRecord)
                WHERE e.rolled_back = false
                  AND e.description IS NOT NULL
                  AND e.description <> ''
                RETURN e.proposal_id AS id,
                       e.category AS category,
                       e.description AS description,
                       e.constitutional_alignment AS alignment,
                       e.formal_verification_status AS fv_status,
                       e.counterfactual_regression_rate AS regression
                ORDER BY e.constitutional_alignment DESC
                LIMIT $limit
                """,
                {"limit": _MAX_TRAINING_EXAMPLES},
            )
        except Exception as exc:
            self._log.warning("genome_extract_training_failed", error=str(exc))
            return []

        examples: list[GRPOTrainingExample] = []
        for row in rows:
            desc = str(row.get("description", ""))
            cat = str(row.get("category", ""))
            alignment = float(row.get("alignment", 0.0))
            fv = str(row.get("fv_status", ""))
            regression = float(row.get("regression", 0.0))

            # Quality signal
            quality = 0.5
            if alignment > 0.5:
                quality += 0.2
            if fv == "verified":
                quality += 0.2
            if regression < 0.1:
                quality += 0.1

            examples.append(GRPOTrainingExample(
                instruction=f"Propose a {cat} change for the EcodiaOS organism.",
                output=f"Description: {desc}\nAlignment: {alignment:.2f}",
                quality_score=min(1.0, quality),
                category=cat,
                source="applied_proposal",
            ))
        return examples

    async def _extract_efe_calibration(self) -> list[EFECalibrationPoint]:
        """Extract EFE prediction vs actual improvement data."""
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (c:EFECalibration)
                RETURN c.predicted_efe AS predicted,
                       c.actual_improvement AS actual,
                       c.efe_error AS error
                ORDER BY c.measured_at DESC
                LIMIT $limit
                """,
                {"limit": _MAX_CALIBRATION_POINTS},
            )
        except Exception as exc:
            self._log.warning("genome_extract_efe_failed", error=str(exc))
            return []

        return [
            EFECalibrationPoint(
                predicted_efe=float(row.get("predicted", 0.0)),
                actual_improvement=float(row.get("actual", 0.0)),
                efe_error=float(row.get("error", 0.0)),
            )
            for row in rows
        ]

    async def _extract_category_analytics(self) -> list[CategoryAnalyticsRecord]:
        """Extract per-category success/rollback rates."""
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (e:EvolutionRecord)
                WITH e.category AS cat,
                     count(e) AS total,
                     sum(CASE WHEN e.rolled_back = false THEN 1 ELSE 0 END) AS approved,
                     sum(CASE WHEN e.rolled_back = true THEN 1 ELSE 0 END) AS rolled_back
                RETURN cat, total, approved, rolled_back
                """,
            )
        except Exception as exc:
            self._log.warning("genome_extract_analytics_failed", error=str(exc))
            return []

        return [
            CategoryAnalyticsRecord(
                category=str(row.get("cat", "")),
                total=int(row.get("total", 0)),
                approved=int(row.get("approved", 0)),
                rolled_back=int(row.get("rolled_back", 0)),
            )
            for row in rows
        ]

    async def _extract_procedures(self) -> list[ProcedureRecord]:
        """Extract proven procedures from the memory graph."""
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (p:Procedure)
                WHERE p.success_rate >= 0.7
                  AND p.usage_count >= 2
                RETURN p.name AS name,
                       p.preconditions AS preconditions,
                       p.postconditions AS postconditions,
                       p.success_rate AS success_rate,
                       p.usage_count AS usage_count
                ORDER BY p.usage_count * p.success_rate DESC
                LIMIT $limit
                """,
                {"limit": _MAX_PROCEDURES},
            )
        except Exception as exc:
            self._log.warning("genome_extract_procedures_failed", error=str(exc))
            return []

        records: list[ProcedureRecord] = []
        for row in rows:
            pre = row.get("preconditions", [])
            post = row.get("postconditions", [])
            records.append(ProcedureRecord(
                name=str(row.get("name", "")),
                preconditions=pre if isinstance(pre, list) else [],
                postconditions=post if isinstance(post, list) else [],
                success_rate=float(row.get("success_rate", 1.0)),
                usage_count=int(row.get("usage_count", 0)),
            ))
        return records

    async def _extract_dafny_specifications(self) -> list[dict[str, Any]]:
        """Extract learned Dafny specifications from the spec library."""
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (s:DafnySpec)
                WHERE s.verified = true
                RETURN s.name AS name,
                       s.module AS module,
                       s.spec_text AS spec_text,
                       s.usage_count AS usage_count,
                       s.confidence AS confidence
                ORDER BY s.usage_count DESC
                LIMIT 200
                """,
            )
        except Exception as exc:
            self._log.warning("genome_extract_dafny_specs_failed", error=str(exc))
            return []

        return [
            {
                "name": str(row.get("name", "")),
                "module": str(row.get("module", "")),
                "spec_text": str(row.get("spec_text", "")),
                "usage_count": int(row.get("usage_count", 0)),
                "confidence": float(row.get("confidence", 0.5)),
            }
            for row in rows
        ]

    async def _extract_lean_lemma_catalog(self) -> list[dict[str, Any]]:
        """Extract proven Lean lemmas from the lemma catalog."""
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (l:LeanLemma)
                WHERE l.proven = true
                RETURN l.name AS name,
                       l.statement AS statement,
                       l.proof_sketch AS proof_sketch,
                       l.category AS category,
                       l.usage_count AS usage_count
                ORDER BY l.usage_count DESC
                LIMIT 200
                """,
            )
        except Exception as exc:
            self._log.warning("genome_extract_lean_lemmas_failed", error=str(exc))
            return []

        return [
            {
                "name": str(row.get("name", "")),
                "statement": str(row.get("statement", "")),
                "proof_sketch": str(row.get("proof_sketch", "")),
                "category": str(row.get("category", "")),
                "usage_count": int(row.get("usage_count", 0)),
            }
            for row in rows
        ]

    async def _extract_reasoning_router_weights(self) -> dict[str, dict[str, float]]:
        """Extract Thompson sampling weights from the ReasoningRouter."""
        try:
            from systems.simula.reasoning_router import ReasoningRouter

            # Try to load persisted weights from Neo4j
            rows = await self._neo4j.execute_read(
                """
                MATCH (r:ReasoningRouterState)
                RETURN r.weights AS weights
                ORDER BY r.updated_at DESC
                LIMIT 1
                """,
            )
            if rows:
                raw = rows[0].get("weights", "{}")
                if isinstance(raw, str):
                    return json.loads(raw)
                if isinstance(raw, dict):
                    return raw
        except Exception as exc:
            self._log.warning("genome_extract_router_weights_failed", error=str(exc))
        return {}

    async def _extract_proof_search_heuristics(self) -> dict[str, Any]:
        """Extract proof search heuristic parameters (timeouts, strategies, priorities)."""
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (h:ProofSearchHeuristic)
                RETURN h.name AS name,
                       h.value AS value,
                       h.updated_at AS updated_at
                ORDER BY h.updated_at DESC
                LIMIT 50
                """,
            )
        except Exception as exc:
            self._log.warning("genome_extract_proof_heuristics_failed", error=str(exc))
            return {}

        heuristics: dict[str, Any] = {}
        for row in rows:
            name = str(row.get("name", ""))
            if name:
                heuristics[name] = row.get("value")
        return heuristics

    async def _extract_inspector_confidence_calibration(self) -> list[dict[str, Any]]:
        """Extract inspector confidence calibration data (predicted vs actual vuln severity)."""
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (c:InspectorCalibration)
                RETURN c.predicted_severity AS predicted_severity,
                       c.actual_severity AS actual_severity,
                       c.target AS target,
                       c.confidence AS confidence,
                       c.measured_at AS measured_at
                ORDER BY c.measured_at DESC
                LIMIT 100
                """,
            )
        except Exception as exc:
            self._log.warning("genome_extract_inspector_calibration_failed", error=str(exc))
            return []

        return [
            {
                "predicted_severity": str(row.get("predicted_severity", "")),
                "actual_severity": str(row.get("actual_severity", "")),
                "target": str(row.get("target", "")),
                "confidence": float(row.get("confidence", 0.5)),
            }
            for row in rows
        ]

    async def _extract_metadata(self) -> dict[str, float | int]:
        """Extract aggregate metadata for the genome."""
        metadata: dict[str, float | int] = {}
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (e:EvolutionRecord)
                RETURN count(e) AS total,
                       sum(CASE WHEN e.rolled_back = false THEN 1 ELSE 0 END) AS applied,
                       sum(CASE WHEN e.rolled_back = true THEN 1 ELSE 0 END) AS rollbacks,
                       avg(e.constitutional_alignment) AS mean_alignment
                """,
            )
            if rows:
                r = rows[0]
                metadata["total_processed"] = int(r.get("total", 0))
                metadata["total_applied"] = int(r.get("applied", 0))
                metadata["total_rollbacks"] = int(r.get("rollbacks", 0))
                metadata["mean_alignment"] = float(r.get("mean_alignment", 0.0) or 0.0)
        except Exception:
            pass

        # Config version
        try:
            rows = await self._neo4j.execute_read(
                """
                MATCH (v:ConfigVersion)
                RETURN max(v.version) AS max_version
                """,
            )
            if rows and rows[0].get("max_version") is not None:
                metadata["config_version"] = int(rows[0]["max_version"])
        except Exception:
            pass

        return metadata

    # ─── Neo4j Persistence ───────────────────────────────────────────────────

    async def _persist_genome(self, genome: SimulaGenome, payload_b64: str) -> None:
        """Persist the SimulaGenome as a :SimulaGenome node in Neo4j."""
        try:
            await self._neo4j.execute_write(
                """
                CREATE (g:SimulaGenome {
                    id: $id,
                    parent_instance_id: $parent_instance_id,
                    generation: $generation,
                    parent_ids: $parent_ids,
                    compression_method: $compression_method,
                    genome_size_bytes: $genome_size_bytes,
                    mutations_count: $mutations_count,
                    abstractions_count: $abstractions_count,
                    training_examples_count: $training_examples_count,
                    total_proposals_processed: $total_processed,
                    total_proposals_applied: $total_applied,
                    total_rollbacks: $total_rollbacks,
                    mean_constitutional_alignment: $mean_alignment,
                    evolution_velocity: $velocity,
                    config_version_at_extraction: $config_version,
                    payload: $payload,
                    created_at: datetime($created_at)
                })
                WITH g
                MATCH (s:Self {instance_id: $parent_instance_id})
                CREATE (g)-[:SIMULA_GENOME_OF]->(s)
                """,
                {
                    "id": genome.id,
                    "parent_instance_id": genome.parent_instance_id,
                    "generation": genome.generation,
                    "parent_ids": genome.parent_ids,
                    "compression_method": genome.compression_method,
                    "genome_size_bytes": genome.genome_size_bytes,
                    "mutations_count": len(genome.mutations),
                    "abstractions_count": len(genome.library_abstractions),
                    "training_examples_count": len(genome.grpo_training_examples),
                    "total_processed": genome.total_proposals_processed,
                    "total_applied": genome.total_proposals_applied,
                    "total_rollbacks": genome.total_rollbacks,
                    "mean_alignment": genome.mean_constitutional_alignment,
                    "velocity": genome.evolution_velocity,
                    "config_version": genome.config_version_at_extraction,
                    "payload": payload_b64,
                    "created_at": genome.created_at.isoformat(),
                },
            )
        except Exception as exc:
            self._log.error("simula_genome_persist_failed", error=str(exc))
            await self._sentinel.report(
                exc, context={"operation": "genome_persist", "instance_id": self._instance_id},
            )


# ─── Genome Seeder ───────────────────────────────────────────────────────────


class SimulaGenomeSeeder:
    """
    Seeds a child instance's Simula subsystems from a parent's SimulaGenome.

    Called during instance birth when a SimulaGenome reference is available.
    Seeds four subsystems:

    1. Evolution history: writes ancestor :EvolutionRecord nodes tagged
       with source="parent_genome" so the child's analytics engine has
       historical context (risk calibration, category success rates).

    2. LILO library: creates :LibraryAbstraction nodes pre-populated with
       the parent's proven code patterns, discounted by 0.9x confidence.

    3. GRPO training data: persists training examples as :TrainingRecord
       nodes for the child's next fine-tuning run.

    4. EFE calibration: writes :EFECalibration nodes so the child's
       architecture scorer starts calibrated rather than guessing.
    """

    def __init__(self, neo4j: Neo4jClient, child_instance_id: str) -> None:
        self._neo4j = neo4j
        self._child_instance_id = child_instance_id
        self._log = logger.bind(subsystem="simula.genome.seeder")

        from systems.synapse.sentinel import ErrorSentinel
        self._sentinel = ErrorSentinel("simula.genome.seeder")

    async def seed_from_genome(
        self,
        genome: SimulaGenome,
    ) -> SimulaGenomeSeedingResult:
        """
        Decompress and seed all Simula subsystems from the parent genome.
        """
        start = time.monotonic()
        result = SimulaGenomeSeedingResult(parent_genome_id=genome.id)

        # Seed mutations as ancestor records
        result.mutations_seeded = await self._seed_mutations(genome.mutations)

        # Seed library abstractions (confidence discounted 0.9x)
        result.abstractions_seeded = await self._seed_abstractions(
            genome.library_abstractions
        )

        # Seed training examples
        result.training_examples_seeded = await self._seed_training_data(
            genome.grpo_training_examples
        )

        # Seed EFE calibration
        result.calibration_points_seeded = await self._seed_efe_calibration(
            genome.efe_calibration
        )

        # Seed procedures
        result.procedures_seeded = await self._seed_procedures(genome.procedures)

        # Link child Self to parent genome
        try:
            await self._neo4j.execute_write(
                """
                MATCH (s:Self {instance_id: $child_id})
                MATCH (g:SimulaGenome {id: $genome_id})
                CREATE (s)-[:SIMULA_SEEDED_BY {
                    seeded_at: datetime($now),
                    mutations_seeded: $mutations,
                    abstractions_seeded: $abstractions
                }]->(g)
                """,
                {
                    "child_id": self._child_instance_id,
                    "genome_id": genome.id,
                    "now": utc_now().isoformat(),
                    "mutations": result.mutations_seeded,
                    "abstractions": result.abstractions_seeded,
                },
            )
        except Exception as exc:
            self._log.warning("simula_seeded_by_link_failed", error=str(exc))
            await self._sentinel.report(
                exc, context={"operation": "seeded_by_link"},
            )

        result.duration_ms = int((time.monotonic() - start) * 1000)

        self._log.info(
            "simula_genome_seeded",
            child_id=self._child_instance_id,
            genome_id=genome.id,
            parent_id=genome.parent_instance_id,
            mutations=result.mutations_seeded,
            abstractions=result.abstractions_seeded,
            training=result.training_examples_seeded,
            calibration=result.calibration_points_seeded,
            procedures=result.procedures_seeded,
        )

        return result

    async def load_genome_from_neo4j(self, genome_id: str) -> SimulaGenome | None:
        """Load a persisted SimulaGenome by ID."""
        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (g:SimulaGenome {id: $genome_id})
                RETURN g.payload AS payload,
                       g.compression_method AS method
                """,
                {"genome_id": genome_id},
            )
            if not records:
                return None

            payload = str(records[0].get("payload", ""))
            method = str(records[0].get("method", "zlib"))
            return _decompress_genome(payload, method)

        except Exception as exc:
            self._log.error(
                "simula_genome_load_failed",
                genome_id=genome_id,
                error=str(exc),
            )
            await self._sentinel.report(
                exc, context={"operation": "genome_load", "genome_id": genome_id},
            )
            return None

    async def load_genome_from_base64(
        self,
        payload_b64: str,
        compression_method: str = "zlib",
    ) -> SimulaGenome | None:
        """Load a SimulaGenome from a raw base64 string (CLI/env var)."""
        try:
            return _decompress_genome(payload_b64, compression_method)
        except Exception as exc:
            self._log.error("simula_genome_b64_load_failed", error=str(exc))
            return None

    # ─── Segment Seeders ─────────────────────────────────────────────────────

    async def _seed_mutations(self, mutations: list[MutationRecord]) -> int:
        """Write ancestor mutation records for analytics context."""
        seeded = 0
        for m in mutations:
            try:
                await self._neo4j.execute_write(
                    """
                    CREATE (:EvolutionRecord {
                        id: $id,
                        proposal_id: $proposal_id,
                        category: $category,
                        description: $description,
                        files_changed: $files_changed,
                        simulation_risk: $risk,
                        constitutional_alignment: $alignment,
                        counterfactual_regression_rate: $regression,
                        formal_verification_status: $fv_status,
                        rolled_back: false,
                        source: "parent_genome",
                        parent_instance_id: $parent_instance_id,
                        from_version: 0,
                        to_version: 0,
                        created_at: datetime($now)
                    })
                    """,
                    {
                        "id": new_id(),
                        "proposal_id": m.proposal_id,
                        "category": m.category,
                        "description": m.description,
                        "files_changed": m.files_changed,
                        "risk": m.risk_level,
                        "alignment": m.constitutional_alignment,
                        "regression": m.regression_rate,
                        "fv_status": m.formal_verification_status,
                        "parent_instance_id": self._child_instance_id,
                        "now": utc_now().isoformat(),
                    },
                )
                seeded += 1
            except Exception as exc:
                self._log.debug("seed_mutation_failed", error=str(exc))
        return seeded

    async def _seed_abstractions(
        self, abstractions: list[LibraryAbstractionRecord],
    ) -> int:
        """Write LILO library abstractions with discounted confidence."""
        seeded = 0
        _CONFIDENCE_DISCOUNT = 0.9

        for a in abstractions:
            try:
                import orjson

                await self._neo4j.execute_write(
                    """
                    CREATE (:LibraryAbstraction {
                        name: $name,
                        kind: $kind,
                        description: $description,
                        signature: $signature,
                        source_code: $source_code,
                        usage_count: $usage_count,
                        confidence: $confidence,
                        tags: $tags,
                        source: "parent_genome",
                        created_at: datetime($now)
                    })
                    """,
                    {
                        "name": a.name,
                        "kind": a.kind,
                        "description": a.description,
                        "signature": a.signature,
                        "source_code": a.source_code,
                        "usage_count": a.usage_count,
                        "confidence": a.confidence * _CONFIDENCE_DISCOUNT,
                        "tags": orjson.dumps(a.tags).decode(),
                        "now": utc_now().isoformat(),
                    },
                )
                seeded += 1
            except Exception as exc:
                self._log.debug("seed_abstraction_failed", error=str(exc))
        return seeded

    async def _seed_training_data(
        self, examples: list[GRPOTrainingExample],
    ) -> int:
        """Write training examples for the child's GRPO pipeline."""
        seeded = 0
        for ex in examples:
            try:
                await self._neo4j.execute_write(
                    """
                    CREATE (:TrainingRecord {
                        id: $id,
                        instruction: $instruction,
                        input: $input,
                        output: $output,
                        quality_score: $quality,
                        category: $category,
                        source: "parent_genome",
                        created_at: datetime($now)
                    })
                    """,
                    {
                        "id": new_id(),
                        "instruction": ex.instruction,
                        "input": ex.input,
                        "output": ex.output,
                        "quality": ex.quality_score,
                        "category": ex.category,
                        "now": utc_now().isoformat(),
                    },
                )
                seeded += 1
            except Exception as exc:
                self._log.debug("seed_training_failed", error=str(exc))
        return seeded

    async def _seed_efe_calibration(
        self, calibration: list[EFECalibrationPoint],
    ) -> int:
        """Write EFE calibration records for the child's scorer."""
        seeded = 0
        for c in calibration:
            try:
                await self._neo4j.execute_write(
                    """
                    CREATE (:EFECalibration {
                        predicted_efe: $predicted,
                        actual_improvement: $actual,
                        efe_error: $error,
                        source: "parent_genome",
                        measured_at: datetime($now)
                    })
                    """,
                    {
                        "predicted": c.predicted_efe,
                        "actual": c.actual_improvement,
                        "error": c.efe_error,
                        "now": utc_now().isoformat(),
                    },
                )
                seeded += 1
            except Exception as exc:
                self._log.debug("seed_efe_calibration_failed", error=str(exc))
        return seeded

    async def _seed_procedures(self, procedures: list[ProcedureRecord]) -> int:
        """Write proven procedures into the child's memory graph."""
        seeded = 0
        for p in procedures:
            try:
                await self._neo4j.execute_write(
                    """
                    CREATE (:Procedure {
                        id: $id,
                        name: $name,
                        preconditions: $preconditions,
                        postconditions: $postconditions,
                        success_rate: $success_rate,
                        usage_count: $usage_count,
                        source: "parent_genome",
                        created_at: datetime($now)
                    })
                    """,
                    {
                        "id": new_id(),
                        "name": p.name,
                        "preconditions": p.preconditions,
                        "postconditions": p.postconditions,
                        "success_rate": p.success_rate,
                        "usage_count": p.usage_count,
                        "now": utc_now().isoformat(),
                    },
                )
                seeded += 1
            except Exception as exc:
                self._log.debug("seed_procedure_failed", error=str(exc))
        return seeded
