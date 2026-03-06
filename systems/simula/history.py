"""
EcodiaOS -- Simula Evolution History

Maintains the complete, immutable record of all structural changes
applied to this EOS instance. Records are (:EvolutionRecord) nodes
in the Memory graph, linked as:
  (:ConfigVersion)-[:EVOLVED_FROM]->(:ConfigVersion)

Stage 1B enhancement: EvolutionRecord nodes store description embeddings
via voyage-code-3 in a Neo4j vector index for semantic similarity search
across the full evolution history.

The Honesty drive demands this record be permanent and complete.
No record can be deleted or modified after writing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from systems.simula.evolution_types import (
    ConfigVersion,
    EFECalibrationRecord,
    EvolutionRecord,
    ProposalEFEBreakdown,
)

if TYPE_CHECKING:
    from clients.embedding import EmbeddingClient
    from clients.neo4j import Neo4jClient

logger = structlog.get_logger().bind(system="simula.history")

# Voyage-code-3 produces 1024-dimensional embeddings
_EMBEDDING_DIMENSION = 1024
_VECTOR_INDEX_NAME = "evolution_record_embedding"


class EvolutionHistoryManager:
    """
    Writes and queries the immutable evolution history stored in Neo4j.
    Every applied structural change produces exactly one EvolutionRecord node.
    Records are never deleted or updated.

    With an embedding client configured, each record's description is embedded
    and stored in a Neo4j vector index for semantic similarity search.
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self._neo4j = neo4j
        self._embedding = embedding_client
        self._log = logger
        self._vector_index_ensured = False

    async def ensure_vector_index(self) -> None:
        """
        Create the Neo4j vector index on EvolutionRecord.embedding if it
        doesn't already exist. Idempotent — safe to call multiple times.

        Requires Neo4j 5.11+ with vector index support.
        """
        if self._vector_index_ensured:
            return

        try:
            await self._neo4j.execute_write(
                f"""
                CREATE VECTOR INDEX {_VECTOR_INDEX_NAME} IF NOT EXISTS
                FOR (r:EvolutionRecord)
                ON (r.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {_EMBEDDING_DIMENSION},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
                """
            )
            self._vector_index_ensured = True
            self._log.info(
                "vector_index_ensured",
                index_name=_VECTOR_INDEX_NAME,
                dimension=_EMBEDDING_DIMENSION,
            )
        except Exception as exc:
            # Non-fatal: vector search degrades gracefully without the index
            self._log.warning(
                "vector_index_creation_failed",
                error=str(exc),
                hint="Neo4j 5.11+ required for vector indexes",
            )

    async def record(self, record: EvolutionRecord) -> None:
        """
        Write an immutable EvolutionRecord node to Neo4j.
        This is the permanent history of every structural change.

        If an embedding client is configured, the description is embedded
        and stored alongside the record for vector similarity search.
        """
        # Embed the description for vector search
        embedding: list[float] | None = None
        if self._embedding is not None:
            try:
                await self.ensure_vector_index()
                embedding = await self._embedding.embed(record.description)
            except Exception as exc:
                self._log.warning(
                    "embedding_generation_failed",
                    record_id=record.id,
                    error=str(exc),
                )

        # Build the CREATE query — include embedding property if available
        if embedding is not None:
            query = """
            CREATE (:EvolutionRecord {
                id: $id,
                proposal_id: $proposal_id,
                category: $category,
                description: $description,
                from_version: $from_version,
                to_version: $to_version,
                files_changed: $files_changed,
                simulation_risk: $simulation_risk,
                applied_at: $applied_at,
                rolled_back: $rolled_back,
                rollback_reason: $rollback_reason,
                simulation_episodes_tested: $simulation_episodes_tested,
                counterfactual_regression_rate: $counterfactual_regression_rate,
                dependency_blast_radius: $dependency_blast_radius,
                constitutional_alignment: $constitutional_alignment,
                resource_tokens_per_hour: $resource_tokens_per_hour,
                caution_reasoning: $caution_reasoning,
                created_at: $created_at,
                embedding: $embedding
            })
            """
        else:
            query = """
            CREATE (:EvolutionRecord {
                id: $id,
                proposal_id: $proposal_id,
                category: $category,
                description: $description,
                from_version: $from_version,
                to_version: $to_version,
                files_changed: $files_changed,
                simulation_risk: $simulation_risk,
                applied_at: $applied_at,
                rolled_back: $rolled_back,
                rollback_reason: $rollback_reason,
                simulation_episodes_tested: $simulation_episodes_tested,
                counterfactual_regression_rate: $counterfactual_regression_rate,
                dependency_blast_radius: $dependency_blast_radius,
                constitutional_alignment: $constitutional_alignment,
                resource_tokens_per_hour: $resource_tokens_per_hour,
                caution_reasoning: $caution_reasoning,
                created_at: $created_at
            })
            """

        params: dict[str, Any] = {
            "id": record.id,
            "proposal_id": record.proposal_id,
            "category": record.category.value,
            "description": record.description,
            "from_version": record.from_version,
            "to_version": record.to_version,
            "files_changed": record.files_changed,
            "simulation_risk": record.simulation_risk.value,
            "applied_at": record.applied_at.isoformat(),
            "rolled_back": record.rolled_back,
            "rollback_reason": record.rollback_reason,
            "simulation_episodes_tested": record.simulation_episodes_tested,
            "counterfactual_regression_rate": record.counterfactual_regression_rate,
            "dependency_blast_radius": record.dependency_blast_radius,
            "constitutional_alignment": record.constitutional_alignment,
            "resource_tokens_per_hour": record.resource_tokens_per_hour,
            "caution_reasoning": record.caution_reasoning,
            "created_at": record.created_at.isoformat(),
        }
        if embedding is not None:
            params["embedding"] = embedding

        await self._neo4j.execute_write(query, params)
        self._log.info(
            "evolution_recorded",
            record_id=record.id,
            proposal_id=record.proposal_id,
            category=record.category.value,
            from_version=record.from_version,
            to_version=record.to_version,
            has_embedding=embedding is not None,
        )

    async def find_similar_records(
        self,
        description: str,
        top_k: int = 5,
        min_score: float = 0.6,
    ) -> list[tuple[EvolutionRecord, float]]:
        """
        Find EvolutionRecords semantically similar to a description.

        Uses the Neo4j vector index to perform approximate nearest-neighbor
        search on stored embeddings. Returns (record, score) pairs.
        """
        if self._embedding is None:
            return []

        try:
            from clients.embedding import VoyageEmbeddingClient

            if isinstance(self._embedding, VoyageEmbeddingClient):
                query_vec = await self._embedding.embed_query(description)
            else:
                query_vec = await self._embedding.embed(description)
        except Exception as exc:
            self._log.warning("similar_records_embed_failed", error=str(exc))
            return []

        try:
            rows = await self._neo4j.execute_read(
                """
                CALL db.index.vector.queryNodes(
                    $index_name, $top_k, $query_vector
                )
                YIELD node, score
                WHERE score >= $min_score
                RETURN node, score
                ORDER BY score DESC
                """,
                {
                    "index_name": _VECTOR_INDEX_NAME,
                    "top_k": top_k,
                    "query_vector": query_vec,
                    "min_score": min_score,
                },
            )
        except Exception as exc:
            self._log.warning("vector_search_failed", error=str(exc))
            return []

        results: list[tuple[EvolutionRecord, float]] = []
        for row in rows:
            try:
                data = dict(row["node"])
                # Remove embedding from reconstruction (not part of the model)
                data.pop("embedding", None)
                record = EvolutionRecord(**data)
                results.append((record, float(row["score"])))
            except Exception as exc:
                self._log.warning("record_reconstruction_failed", error=str(exc))
                continue

        return results

    async def record_version(self, version: ConfigVersion, previous_version: int | None) -> None:
        """""""""
        Write a ConfigVersion node and optionally chain it to the previous version.
        """""""""
        await self._neo4j.execute_write(
            """
            MERGE (:ConfigVersion {
                version: $version,
                timestamp: $timestamp,
                proposal_ids: $proposal_ids,
                config_hash: $config_hash
            })
            """,
            {
                "version": version.version,
                "timestamp": version.timestamp.isoformat(),
                "proposal_ids": version.proposal_ids,
                "config_hash": version.config_hash,
            },
        )
        if previous_version is not None:
            await self._neo4j.execute_write(
                """
                MATCH (new:ConfigVersion {version: $new_v})
                MATCH (prev:ConfigVersion {version: $prev_v})
                MERGE (new)-[:EVOLVED_FROM]->(prev)
                """,
                {"new_v": version.version, "prev_v": previous_version},
            )
        self._log.info(
            "config_version_recorded",
            version=version.version,
            previous_version=previous_version,
        )

    async def get_history(self, limit: int = 50) -> list[EvolutionRecord]:
        """""""""
        Retrieve the most recent N evolution records, newest first.
        """""""""
        rows = await self._neo4j.execute_read(
            """
            MATCH (r:EvolutionRecord)
            RETURN r
            ORDER BY r.created_at DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
        records: list[EvolutionRecord] = []
        for row in rows:
            data = row["r"]
            records.append(EvolutionRecord(**data))
        return records

    async def get_version_chain(self) -> list[ConfigVersion]:
        """""""""
        Retrieve all ConfigVersion nodes ordered by version ASC.
        """""""""
        rows = await self._neo4j.execute_read(
            """
            MATCH (v:ConfigVersion)
            RETURN v
            ORDER BY v.version ASC
            """
        )
        versions: list[ConfigVersion] = []
        for row in rows:
            data = row["v"]
            versions.append(ConfigVersion(**data))
        return versions

    async def get_current_version(self) -> int:
        """""""""
        Return the highest config version number, or 0 if none exists.
        """""""""
        rows = await self._neo4j.execute_read(
            """
            MATCH (v:ConfigVersion)
            RETURN max(v.version) AS max_version
            """
        )
        if not rows or rows[0]["max_version"] is None:
            return 0
        return int(rows[0]["max_version"])

    # ─── Architecture EFE Persistence ────────────────────────────────────────

    async def record_proposal_efe(
        self,
        proposal_id: str,
        breakdown: ProposalEFEBreakdown,
        rank: int,
        total_in_batch: int,
    ) -> None:
        """
        Persist a ProposalEFE node in Neo4j, linked to the proposal.

        Stores the full EFE breakdown for observability, calibration
        feedback, and monitoring approval rates by EFE quartile.
        """
        await self._neo4j.execute_write(
            """
            CREATE (:ProposalEFE {
                proposal_id: $proposal_id,
                pragmatic_value: $pragmatic_value,
                epistemic_value: $epistemic_value,
                complexity_penalty: $complexity_penalty,
                efe_raw: $efe_raw,
                efe_penalised: $efe_penalised,
                confidence: $confidence,
                reasoning: $reasoning,
                rank: $rank,
                total_in_batch: $total_in_batch,
                scored_at: datetime()
            })
            """,
            {
                "proposal_id": proposal_id,
                "pragmatic_value": breakdown.pragmatic_value,
                "epistemic_value": breakdown.epistemic_value,
                "complexity_penalty": breakdown.complexity_penalty,
                "efe_raw": breakdown.efe_raw,
                "efe_penalised": breakdown.efe_penalised,
                "confidence": breakdown.confidence,
                "reasoning": breakdown.reasoning[:500],
                "rank": rank,
                "total_in_batch": total_in_batch,
            },
        )
        self._log.info(
            "proposal_efe_recorded",
            proposal_id=proposal_id,
            efe=breakdown.efe_penalised,
            rank=rank,
        )

    async def record_efe_calibration(
        self, calibration: EFECalibrationRecord,
    ) -> None:
        """
        Persist an EFE calibration feedback record to Neo4j.

        Links predicted EFE to actual outcome for calibration learning.
        """
        await self._neo4j.execute_write(
            """
            CREATE (:EFECalibration {
                proposal_id: $proposal_id,
                predicted_efe: $predicted_efe,
                actual_improvement: $actual_improvement,
                efe_error: $efe_error,
                pragmatic_predicted: $pragmatic_predicted,
                epistemic_predicted: $epistemic_predicted,
                complexity_predicted: $complexity_predicted,
                applied_at: $applied_at,
                measured_at: datetime()
            })
            """,
            {
                "proposal_id": calibration.proposal_id,
                "predicted_efe": calibration.predicted_efe,
                "actual_improvement": calibration.actual_improvement,
                "efe_error": calibration.efe_error,
                "pragmatic_predicted": calibration.pragmatic_predicted,
                "epistemic_predicted": calibration.epistemic_predicted,
                "complexity_predicted": calibration.complexity_predicted,
                "applied_at": calibration.applied_at.isoformat(),
            },
        )
        self._log.info(
            "efe_calibration_recorded",
            proposal_id=calibration.proposal_id,
            efe_error=round(calibration.efe_error, 4),
        )

    async def get_efe_scores_by_quartile(self) -> dict[str, dict[str, float]]:
        """
        Aggregate proposal approval rates by EFE quartile.

        Returns quartile buckets (Q1=best, Q4=worst) with:
        - count: total proposals in quartile
        - approved_rate: fraction that were applied
        - mean_efe: average EFE score in quartile
        """
        rows = await self._neo4j.execute_read(
            """
            MATCH (e:ProposalEFE)
            WITH e.efe_penalised AS efe, e.proposal_id AS pid
            ORDER BY efe ASC
            WITH collect({efe: efe, pid: pid}) AS all_scores
            WITH all_scores, size(all_scores) AS total
            UNWIND range(0, total - 1) AS idx
            WITH all_scores[idx] AS item, idx, total,
                 CASE
                   WHEN idx < total / 4 THEN 'Q1'
                   WHEN idx < total / 2 THEN 'Q2'
                   WHEN idx < 3 * total / 4 THEN 'Q3'
                   ELSE 'Q4'
                 END AS quartile
            RETURN quartile,
                   count(*) AS count,
                   avg(item.efe) AS mean_efe
            ORDER BY quartile
            """
        )
        result: dict[str, dict[str, float]] = {}
        for row in rows:
            result[row["quartile"]] = {
                "count": float(row["count"]),
                "mean_efe": round(float(row["mean_efe"]), 4),
            }
        return result
