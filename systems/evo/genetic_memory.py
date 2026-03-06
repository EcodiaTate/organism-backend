"""
EcodiaOS — Genetic Memory: Compressed Belief Inheritance

Implements belief genome extraction, compression, and seeding for parent→child
instance knowledge transfer.

A mature instance (>10,000 episodes, >100 confirmed hypotheses) compresses its
stable, high-confidence beliefs into a BeliefGenome.  Child instances
decompress the genome at birth and seed their hypothesis engine, skipping
redundant re-learning of facts the parent already proved.

Integration points:
    - Evo Phase 2.8 (after belief consolidation, before schema induction):
        GenomeExtractor.extract_genome()
    - Instance birth (birth.py):
        GenomeSeeder.seed_from_genome()
    - Neo4j: :BeliefGenome node, INHERITED_FROM / SEEDED_BY relationships
    - Monitoring: GenomeInheritanceMonitor tracks fidelity over time
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import zlib
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now
from systems.evo.types import (
    BeliefGenome,
    GenomeExtractionResult,
    GenomeInheritanceReport,
    InheritedHypothesisRecord,
)

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from systems.evo.hypothesis import HypothesisEngine

logger = structlog.get_logger()

# ─── Fixation Thresholds ────────────────────────────────────────────────────

# Hypothesis must have confidence >= this to be genome-eligible
_MIN_FIXATION_CONFIDENCE: float = 0.95

# Hypothesis volatility must be < this (low volatility = stable)
_MAX_FIXATION_VOLATILITY: float = 0.1

# Hypothesis must have been alive for at least this many days
_MIN_FIXATION_AGE_DAYS: int = 30

# Instance must have processed at least this many episodes to produce a genome
_MIN_EPISODES_FOR_GENOME: int = 10_000

# Instance must have confirmed at least this many hypotheses
_MIN_CONFIRMED_HYPOTHESES: int = 100

# Inherited confidence discount — children start slightly below parent
_INHERITED_CONFIDENCE_FACTOR: float = 0.95  # parent 0.98 → child 0.931

# ─── Compression ─────────────────────────────────────────────────────────────

# Try LZ4 first (faster), fall back to zlib (always available)
try:
    import lz4.frame as lz4_frame
    _HAS_LZ4 = True
except ImportError:
    _HAS_LZ4 = False


def _content_hash(domain: str, statement: str) -> str:
    """SHA-256 hash of the canonical (domain, statement) pair."""
    canonical = json.dumps({"domain": domain, "statement": statement}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def compress_genome(records: list[InheritedHypothesisRecord]) -> tuple[str, str, int]:
    """
    Serialize and compress a list of hypothesis records into a base64 string.

    Returns (base64_payload, compression_method, raw_byte_count).
    """
    payload = json.dumps(
        [r.model_dump() for r in records],
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


def decompress_genome(
    payload_b64: str,
    method: str,
) -> list[InheritedHypothesisRecord]:
    """
    Decompress a base64-encoded genome string back into hypothesis records.

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

    items = json.loads(decompressed)
    if not isinstance(items, list):
        raise ValueError("Genome payload is not a list")

    return [InheritedHypothesisRecord(**item) for item in items]


# ─── Genome Extractor ────────────────────────────────────────────────────────


class GenomeExtractor:
    """
    Scans the Neo4j belief/hypothesis graph for fixation-eligible beliefs
    and compresses them into a BeliefGenome.

    Run during Evo consolidation Phase 2.8 (after belief consolidation at
    2.75, before schema induction at Phase 3).
    """

    def __init__(self, neo4j: Neo4jClient, instance_id: str) -> None:
        self._neo4j = neo4j
        self._instance_id = instance_id
        self._logger = logger.bind(system="evo.genetic_memory")

    async def is_eligible(self) -> bool:
        """Check whether this instance is mature enough to produce a genome."""
        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (s:Self {instance_id: $instance_id})
                RETURN s.total_episodes AS episodes
                """,
                {"instance_id": self._instance_id},
            )
            if not records:
                return False
            episodes = int(records[0].get("episodes", 0))
            if episodes < _MIN_EPISODES_FOR_GENOME:
                self._logger.debug(
                    "genome_not_eligible",
                    reason="insufficient_episodes",
                    episodes=episodes,
                    required=_MIN_EPISODES_FOR_GENOME,
                )
                return False
        except Exception as exc:
            self._logger.warning("genome_eligibility_check_failed", error=str(exc))
            return False

        # Count confirmed hypotheses (INTEGRATED status)
        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (h:Hypothesis)
                WHERE h.status = "integrated"
                RETURN count(h) AS confirmed
                """,
            )
            confirmed = int(records[0].get("confirmed", 0)) if records else 0
            if confirmed < _MIN_CONFIRMED_HYPOTHESES:
                self._logger.debug(
                    "genome_not_eligible",
                    reason="insufficient_confirmed_hypotheses",
                    confirmed=confirmed,
                    required=_MIN_CONFIRMED_HYPOTHESES,
                )
                return False
        except Exception as exc:
            self._logger.warning("genome_confirmed_count_failed", error=str(exc))
            return False

        return True

    async def extract_genome(
        self,
        parent_ids: list[str] | None = None,
        generation: int = 1,
    ) -> tuple[BeliefGenome | None, GenomeExtractionResult]:
        """
        Extract fixation-eligible beliefs and build a compressed genome.

        Fixation criteria (all must be met):
            - Precision (confidence) >= 0.95
            - Volatility < 0.1
            - Age > 30 days
            - No active contradictions (no hypothesis in TESTING state
              with contradicting evidence against the same domain)

        Returns (genome, extraction_result).  genome is None if the
        instance is not mature enough or no candidates meet criteria.
        """
        start = time.monotonic()
        result = GenomeExtractionResult()

        if not await self.is_eligible():
            result.duration_ms = int((time.monotonic() - start) * 1000)
            return None, result

        # Query consolidated beliefs meeting fixation thresholds
        now = utc_now()
        cutoff = now - timedelta(days=_MIN_FIXATION_AGE_DAYS)

        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (cb:ConsolidatedBelief)
                WHERE cb.precision >= $min_confidence
                  AND cb.consolidated_at IS NOT NULL
                  AND cb.consolidated_at <= $cutoff
                  AND (cb.volatility_percentile IS NULL OR cb.volatility_percentile < $max_volatility)
                RETURN cb.id AS id,
                       cb.domain AS domain,
                       cb.statement AS statement,
                       cb.precision AS precision,
                       coalesce(cb.volatility_percentile, 0.0) AS volatility_percentile
                ORDER BY cb.precision DESC
                """,
                {
                    "min_confidence": _MIN_FIXATION_CONFIDENCE,
                    "cutoff": cutoff.isoformat(),
                    "max_volatility": _MAX_FIXATION_VOLATILITY,
                },
            )
        except Exception as exc:
            self._logger.error("genome_extraction_query_failed", error=str(exc))
            result.duration_ms = int((time.monotonic() - start) * 1000)
            return None, result

        result.candidates_scanned = len(records)

        if not records:
            result.duration_ms = int((time.monotonic() - start) * 1000)
            return None, result

        # Also pull integrated hypotheses for richer records
        try:
            hyp_records = await self._neo4j.execute_read(
                """
                MATCH (h:Hypothesis)
                WHERE h.status = "integrated"
                  AND h.evidence_score >= $min_score
                RETURN h.hypothesis_id AS id,
                       h.category AS category,
                       h.statement AS statement,
                       h.evidence_score AS evidence_score,
                       h.created_at AS created_at
                """,
                {"min_score": _MIN_FIXATION_CONFIDENCE * 3.0},  # ~2.85
            )
        except Exception:
            hyp_records = []

        # Build lookup of integrated hypothesis details by statement prefix
        hyp_by_statement: dict[str, dict[str, Any]] = {}
        for hr in hyp_records:
            stmt = str(hr.get("statement", ""))[:100]
            hyp_by_statement[stmt] = dict(hr)

        # Check for active contradictions
        contradicted_domains: set[str] = set()
        try:
            contradiction_records = await self._neo4j.execute_read(
                """
                MATCH (h:Hypothesis)
                WHERE h.status IN ["proposed", "testing"]
                  AND h.contradicting_count > 0
                RETURN DISTINCT h.category AS domain
                """,
            )
            for cr in contradiction_records:
                domain = str(cr.get("domain", ""))
                if domain:
                    contradicted_domains.add(domain)
        except Exception as exc:
            self._logger.warning(
                "genome_contradiction_lookup_failed",
                error=str(exc),
            )

        # Build hypothesis records for the genome
        fixed_records: list[InheritedHypothesisRecord] = []

        for record in records:
            domain = str(record.get("domain", ""))
            statement = str(record.get("statement", ""))
            precision = float(record.get("precision", 0.0))
            volatility = float(record.get("volatility_percentile", 0.0))

            # Skip if domain has active contradictions
            if domain in contradicted_domains:
                continue

            # Compute content hash
            c_hash = _content_hash(domain, statement)

            # Look up hypothesis details for richer record
            hyp_detail = hyp_by_statement.get(statement[:100], {})
            category = str(hyp_detail.get("category", "world_model"))
            formal_test = ""  # Compressed away in genome

            # Estimate age
            created_raw = hyp_detail.get("created_at")
            if isinstance(created_raw, str):
                try:
                    from datetime import datetime

                    created = datetime.fromisoformat(created_raw)
                    age_days = (now - created).total_seconds() / 86400
                except (ValueError, TypeError):
                    age_days = float(_MIN_FIXATION_AGE_DAYS)
            else:
                age_days = float(_MIN_FIXATION_AGE_DAYS)

            fixed_records.append(
                InheritedHypothesisRecord(
                    domain=domain,
                    category=category,
                    statement=statement,
                    formal_test=formal_test,
                    confidence=precision,
                    volatility=volatility,
                    age_days=round(age_days, 1),
                    content_hash=c_hash,
                )
            )

        result.candidates_fixed = len(fixed_records)

        if not fixed_records:
            result.duration_ms = int((time.monotonic() - start) * 1000)
            return None, result

        # Compress into genome
        payload_b64, compression_method, byte_count = compress_genome(fixed_records)
        result.genome_size_bytes = byte_count

        # Get episode count for metadata
        try:
            ep_records = await self._neo4j.execute_read(
                """
                MATCH (s:Self {instance_id: $instance_id})
                RETURN s.total_episodes AS episodes
                """,
                {"instance_id": self._instance_id},
            )
            total_episodes = int(ep_records[0].get("episodes", 0)) if ep_records else 0
        except Exception:
            total_episodes = 0

        genome_id = new_id()
        genome = BeliefGenome(
            id=genome_id,
            parent_instance_id=self._instance_id,
            stable_hypotheses=fixed_records,
            compression_method=compression_method,
            version=1,
            generation=generation,
            parent_ids=parent_ids or [self._instance_id],
            total_episodes_at_fixation=total_episodes,
            total_hypotheses_confirmed=len(hyp_records),
            genome_size_bytes=byte_count,
        )

        # Persist genome to Neo4j
        await self._persist_genome(genome, payload_b64)

        result.genome_id = genome_id
        result.duration_ms = int((time.monotonic() - start) * 1000)

        self._logger.info(
            "genome_extracted",
            genome_id=genome_id,
            fixed_count=len(fixed_records),
            genome_bytes=byte_count,
            compression=compression_method,
            generation=generation,
        )

        return genome, result

    async def _persist_genome(self, genome: BeliefGenome, payload_b64: str) -> None:
        """Persist the genome as a :BeliefGenome node in Neo4j."""
        try:
            await self._neo4j.execute_write(
                """
                CREATE (g:BeliefGenome {
                    id: $id,
                    parent_instance_id: $parent_instance_id,
                    compression_method: $compression_method,
                    version: $version,
                    generation: $generation,
                    parent_ids: $parent_ids,
                    hypothesis_count: $hypothesis_count,
                    genome_size_bytes: $genome_size_bytes,
                    total_episodes_at_fixation: $total_episodes,
                    total_hypotheses_confirmed: $total_confirmed,
                    payload: $payload,
                    created_at: datetime($created_at)
                })
                WITH g
                MATCH (s:Self {instance_id: $parent_instance_id})
                CREATE (g)-[:INHERITED_FROM]->(s)
                """,
                {
                    "id": genome.id,
                    "parent_instance_id": genome.parent_instance_id,
                    "compression_method": genome.compression_method,
                    "version": genome.version,
                    "generation": genome.generation,
                    "parent_ids": genome.parent_ids,
                    "hypothesis_count": len(genome.stable_hypotheses),
                    "genome_size_bytes": genome.genome_size_bytes,
                    "total_episodes": genome.total_episodes_at_fixation,
                    "total_confirmed": genome.total_hypotheses_confirmed,
                    "payload": payload_b64,
                    "created_at": genome.created_at.isoformat(),
                },
            )
        except Exception as exc:
            self._logger.error("genome_persist_failed", error=str(exc))


# ─── Genome Seeder ────────────────────────────────────────────────────────────


class GenomeSeeder:
    """
    Seeds a child instance's hypothesis engine with inherited beliefs
    from a parent genome.

    Called during instance birth (birth.py) when a --parent_genome flag
    is provided.
    """

    def __init__(self, neo4j: Neo4jClient, child_instance_id: str) -> None:
        self._neo4j = neo4j
        self._child_instance_id = child_instance_id
        self._logger = logger.bind(system="evo.genetic_memory.seeder")

    async def seed_from_genome(
        self,
        genome: BeliefGenome,
        hypothesis_engine: HypothesisEngine | None = None,
    ) -> GenomeInheritanceReport:
        """
        Decompress a parent genome and seed the child's hypothesis engine.

        For each inherited hypothesis:
            1. Create a hypothesis in the child's engine with inherited confidence
               (discounted by _INHERITED_CONFIDENCE_FACTOR)
            2. Tag with source="parent_genome", parent_id, generation
            3. Persist as :Hypothesis node with SEEDED_BY relationship

        The child continues learning: inherited hypotheses start high confidence
        but can be adjusted down if contradicted by the child's own experience.
        """
        report = GenomeInheritanceReport(
            parent_genome_id=genome.id,
            total_inherited=len(genome.stable_hypotheses),
        )

        seeded_count = 0
        for record in genome.stable_hypotheses:
            inherited_confidence = record.confidence * _INHERITED_CONFIDENCE_FACTOR

            try:
                await self._neo4j.execute_write(
                    """
                    CREATE (h:Hypothesis {
                        hypothesis_id: $hypothesis_id,
                        type: "hypothesis",
                        category: $category,
                        statement: $statement,
                        status: "testing",
                        evidence_score: $evidence_score,
                        supporting_count: 0,
                        contradicting_count: 0,
                        created_at: $created_at,
                        source: "parent_genome",
                        parent_genome_id: $genome_id,
                        parent_instance_id: $parent_instance_id,
                        inherited_confidence: $inherited_confidence,
                        generation: $generation,
                        content_hash: $content_hash
                    })
                    WITH h
                    MATCH (g:BeliefGenome {id: $genome_id})
                    CREATE (h)-[:SEEDED_BY]->(g)
                    """,
                    {
                        "hypothesis_id": new_id(),
                        "category": record.category,
                        "statement": record.statement,
                        "evidence_score": inherited_confidence * 3.0,
                        "created_at": utc_now().isoformat(),
                        "genome_id": genome.id,
                        "parent_instance_id": genome.parent_instance_id,
                        "inherited_confidence": inherited_confidence,
                        "generation": genome.generation + 1,
                        "content_hash": record.content_hash,
                    },
                )
                seeded_count += 1
            except Exception as exc:
                self._logger.warning(
                    "genome_seed_hypothesis_failed",
                    statement=record.statement[:60],
                    error=str(exc),
                )

        # Record the SEEDED_BY relationship from child Self to genome
        try:
            await self._neo4j.execute_write(
                """
                MATCH (s:Self {instance_id: $child_id})
                MATCH (g:BeliefGenome {id: $genome_id})
                CREATE (s)-[:SEEDED_BY {
                    seeded_at: datetime($now),
                    hypotheses_seeded: $seeded_count
                }]->(g)
                """,
                {
                    "child_id": self._child_instance_id,
                    "genome_id": genome.id,
                    "now": utc_now().isoformat(),
                    "seeded_count": seeded_count,
                },
            )
        except Exception as exc:
            self._logger.warning("genome_seeded_by_link_failed", error=str(exc))

        report.kept_unchanged = seeded_count

        self._logger.info(
            "genome_seeded",
            child_instance_id=self._child_instance_id,
            genome_id=genome.id,
            parent_instance_id=genome.parent_instance_id,
            hypotheses_seeded=seeded_count,
            generation=genome.generation + 1,
        )

        return report

    async def load_genome_from_neo4j(self, genome_id: str) -> BeliefGenome | None:
        """Load a persisted genome by ID (for cross-instance transfer)."""
        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (g:BeliefGenome {id: $genome_id})
                RETURN g.id AS id,
                       g.parent_instance_id AS parent_instance_id,
                       g.compression_method AS compression_method,
                       g.version AS version,
                       g.generation AS generation,
                       g.parent_ids AS parent_ids,
                       g.hypothesis_count AS hypothesis_count,
                       g.genome_size_bytes AS genome_size_bytes,
                       g.total_episodes_at_fixation AS total_episodes,
                       g.total_hypotheses_confirmed AS total_confirmed,
                       g.payload AS payload,
                       g.created_at AS created_at
                """,
                {"genome_id": genome_id},
            )
            if not records:
                return None

            r = records[0]
            payload_b64 = str(r.get("payload", ""))
            method = str(r.get("compression_method", "zlib"))

            hypotheses = decompress_genome(payload_b64, method)

            return BeliefGenome(
                id=str(r.get("id", "")),
                parent_instance_id=str(r.get("parent_instance_id", "")),
                stable_hypotheses=hypotheses,
                compression_method=method,
                version=int(r.get("version", 1)),
                generation=int(r.get("generation", 1)),
                parent_ids=list(r.get("parent_ids", [])),
                total_episodes_at_fixation=int(r.get("total_episodes", 0)),
                total_hypotheses_confirmed=int(r.get("total_confirmed", 0)),
                genome_size_bytes=int(r.get("genome_size_bytes", 0)),
            )

        except Exception as exc:
            self._logger.error(
                "genome_load_failed",
                genome_id=genome_id,
                error=str(exc),
            )
            return None

    async def load_genome_from_base64(
        self,
        payload_b64: str,
        compression_method: str = "zlib",
        parent_instance_id: str = "",
    ) -> BeliefGenome | None:
        """
        Load a genome from a raw base64 string (e.g., from CLI --parent_genome flag).

        This is the primary entry point for child instances that receive genomes
        as command-line arguments rather than Neo4j references.
        """
        try:
            hypotheses = decompress_genome(payload_b64, compression_method)
            return BeliefGenome(
                parent_instance_id=parent_instance_id,
                stable_hypotheses=hypotheses,
                compression_method=compression_method,
                version=1,
                generation=1,
                parent_ids=[parent_instance_id] if parent_instance_id else [],
                total_episodes_at_fixation=0,
                total_hypotheses_confirmed=len(hypotheses),
                genome_size_bytes=len(base64.b64decode(payload_b64)),
            )
        except Exception as exc:
            self._logger.error(
                "genome_base64_load_failed",
                error=str(exc),
            )
            return None


# ─── Inheritance Monitor ──────────────────────────────────────────────────────


class GenomeInheritanceMonitor:
    """
    Tracks how a child instance's learning diverges from its inherited genome.

    Queried during consolidation or on-demand for monitoring dashboards.
    Reports:
        - Inheritance fidelity (% beliefs kept unchanged)
        - Confidence downgrades
        - Refutations
        - Novel hypotheses not in parent genome
        - Learning speedup estimate
    """

    def __init__(self, neo4j: Neo4jClient, instance_id: str) -> None:
        self._neo4j = neo4j
        self._instance_id = instance_id
        self._logger = logger.bind(system="evo.genetic_memory.monitor")

    async def generate_report(self) -> GenomeInheritanceReport | None:
        """Generate a fidelity report for this instance's inherited genome."""
        # Find genome this instance was seeded from
        try:
            seed_records = await self._neo4j.execute_read(
                """
                MATCH (s:Self {instance_id: $instance_id})-[:SEEDED_BY]->(g:BeliefGenome)
                RETURN g.id AS genome_id,
                       g.hypothesis_count AS total_inherited
                LIMIT 1
                """,
                {"instance_id": self._instance_id},
            )
        except Exception as exc:
            self._logger.warning("inheritance_report_seed_query_failed", error=str(exc))
            return None

        if not seed_records:
            return None  # Not a child instance (no genome inheritance)

        genome_id = str(seed_records[0].get("genome_id", ""))
        total_inherited = int(seed_records[0].get("total_inherited", 0))

        report = GenomeInheritanceReport(
            parent_genome_id=genome_id,
            total_inherited=total_inherited,
        )

        # Count inherited hypotheses by current status
        try:
            status_records = await self._neo4j.execute_read(
                """
                MATCH (h:Hypothesis)-[:SEEDED_BY]->(g:BeliefGenome {id: $genome_id})
                RETURN h.status AS status,
                       count(h) AS count
                """,
                {"genome_id": genome_id},
            )
            for sr in status_records:
                status = str(sr.get("status", ""))
                count = int(sr.get("count", 0))
                if status in ("supported", "integrated"):
                    report.kept_unchanged += count
                elif status == "testing":
                    report.kept_unchanged += count  # Still being evaluated
                elif status == "refuted":
                    report.refuted += count
                elif status == "archived":
                    report.downgraded += count
        except Exception as exc:
            self._logger.warning("inheritance_report_status_failed", error=str(exc))

        # Count novel hypotheses (not from genome)
        try:
            novel_records = await self._neo4j.execute_read(
                """
                MATCH (h:Hypothesis)
                WHERE (h.source IS NULL OR h.source <> "parent_genome")
                  AND h.status IN ["testing", "supported", "integrated"]
                RETURN count(h) AS novel
                """,
            )
            report.novel_hypotheses = (
                int(novel_records[0].get("novel", 0)) if novel_records else 0
            )
        except Exception as exc:
            self._logger.warning(
                "inheritance_report_novel_query_failed",
                genome_id=genome_id,
                error=str(exc),
            )

        # Learning speedup estimate: compare cycle count to reach N hypotheses
        # vs cold-start baseline (heuristic: inherited beliefs save ~30-40% cycles)
        if total_inherited > 0 and report.kept_unchanged > 0:
            fidelity = report.kept_unchanged / max(1, total_inherited)
            report.learning_speedup_pct = round(fidelity * 40.0, 1)

        self._logger.info(
            "inheritance_report_generated",
            genome_id=genome_id,
            total_inherited=total_inherited,
            kept=report.kept_unchanged,
            downgraded=report.downgraded,
            refuted=report.refuted,
            novel=report.novel_hypotheses,
            speedup_pct=report.learning_speedup_pct,
        )

        return report
