"""
EcodiaOS - Evo Genome: OrganGenomeSegment Adapter

Wraps the existing Evo BeliefGenome / ParameterTuner / hypothesis graph into the
unified OrganGenomeSegment interface used by Mitosis for organism-wide inheritance.

This does NOT replace the existing GenomeExtractor in genetic_memory.py - it
delegates to it for belief extraction and layers on parameter values, experiment
summaries, and consolidation config so that a child instance inherits the full
Evo learning state through a single segment.

Heritable state captured:
    - Tunable parameter values (all 23+)
    - Validated hypotheses (INTEGRATED / SUPPORTED, confidence > 0.8, top 300)
    - Experiment result summaries (top 200 by evidence_score)
    - Consolidation configuration (velocity limits, interval, thresholds)

Payload schema (version "1.0.0"):
    {
        "parameters": {"atune.head.novelty.weight": 0.22, ...},
        "validated_hypotheses": [
            {"id": "...", "category": "...", "statement": "...",
             "status": "...", "confidence": 0.95, "supporting_count": 12},
            ...
        ],
        "experiment_summaries": [
            {"hypothesis_id": "...", "outcome": "supported",
             "evidence_score": 3.2, "episodes_evaluated": 48},
            ...
        ],
        "consolidation_config": {
            "velocity_limits": {...},
            "consolidation_interval_s": 21600,
            "hypothesis_generation_interval": 200,
            "evidence_evaluation_interval": 50,
        }
    }
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID
from primitives.genome import OrganGenomeSegment
from systems.genome_helpers import build_segment, check_schema_version, verify_segment

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from systems.evo.parameter_tuner import ParameterTuner

logger = structlog.get_logger()

# ─── Extraction Limits ────────────────────────────────────────────────────────

_MAX_VALIDATED_HYPOTHESES: int = 300
_MIN_HYPOTHESIS_CONFIDENCE: float = 0.8
_MAX_EXPERIMENT_SUMMARIES: int = 200

# Consolidation config constants mirrored from service.py and types.py
_DEFAULT_CONSOLIDATION_INTERVAL_S: float = 6.0 * 3600  # 6 hours
_DEFAULT_HYPOTHESIS_GENERATION_INTERVAL: int = 200
_DEFAULT_EVIDENCE_EVALUATION_INTERVAL: int = 50


class EvoGenomeExtractor:
    """
    Implements GenomeExtractionProtocol for the Evo system.

    Serialises Evo's heritable state - parameter values, validated hypotheses,
    experiment summaries, and consolidation config - into an OrganGenomeSegment
    for organism-wide Mitosis inheritance.

    This wraps (not replaces) the existing BeliefGenome infrastructure in
    genetic_memory.py: it reads the same Neo4j graph but outputs the unified
    OrganGenomeSegment format that Mitosis expects.
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        parameter_tuner: ParameterTuner | None = None,
        tournament_engine: Any | None = None,
        hypothesis_engine: Any | None = None,
    ) -> None:
        self._neo4j = neo4j
        self._parameter_tuner = parameter_tuner
        self._tournament_engine = tournament_engine
        self._hypothesis_engine = hypothesis_engine
        self._log = logger.bind(subsystem="evo.genome")

    # ─── GenomeExtractionProtocol ─────────────────────────────────────────────

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        """
        Serialise all Evo heritable state into an OrganGenomeSegment.

        Returns an empty segment (version=0) if no validated hypotheses exist,
        signalling to Mitosis that this organ has nothing heritable yet.
        """
        try:
            parameters = self._extract_parameters()
            validated_hypotheses = await self._extract_validated_hypotheses()
            experiment_summaries = await self._extract_experiment_summaries()
            consolidation_config = self._build_consolidation_config()

            # If no validated hypotheses, return an empty segment so Mitosis
            # knows Evo hasn't matured enough yet.
            if not validated_hypotheses:
                self._log.info(
                    "evo_genome_empty",
                    reason="no_validated_hypotheses",
                )
                return build_segment(
                    system_id=SystemID.EVO,
                    payload={},
                    version=0,
                )

            payload: dict[str, Any] = {
                "parameters": parameters,
                "validated_hypotheses": validated_hypotheses,
                "experiment_summaries": experiment_summaries,
                "consolidation_config": consolidation_config,
                "thompson_priors": self._extract_thompson_priors(),
                "active_experiment_designs": self._extract_active_experiments(),
            }

            segment = build_segment(
                system_id=SystemID.EVO,
                payload=payload,
                version=1,
            )

            self._log.info(
                "evo_genome_extracted",
                parameters_count=len(parameters),
                hypotheses_count=len(validated_hypotheses),
                experiments_count=len(experiment_summaries),
                size_bytes=segment.size_bytes,
            )

            return segment

        except Exception as exc:
            self._log.error("evo_genome_extraction_failed", error=str(exc))
            return build_segment(
                system_id=SystemID.EVO,
                payload={},
                version=0,
            )

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        """
        Restore Evo heritable state from a parent's OrganGenomeSegment.

        Writes parameter values to Neo4j EvoParameter nodes and validated
        hypothesis nodes with source="parent_genome". Verifies payload
        integrity via hash and schema version before seeding.

        Returns True on success, False on any failure.
        """
        try:
            # ── Integrity checks ──────────────────────────────────────────
            if not check_schema_version(segment):
                self._log.warning(
                    "evo_genome_seed_rejected",
                    reason="incompatible_schema_version",
                    schema_version=segment.schema_version,
                )
                return False

            if not verify_segment(segment):
                self._log.warning(
                    "evo_genome_seed_rejected",
                    reason="payload_hash_mismatch",
                )
                return False

            payload = segment.payload
            if not payload:
                self._log.info("evo_genome_seed_skipped", reason="empty_payload")
                return True  # Nothing to seed - not a failure

            # ── Seed parameters ───────────────────────────────────────────
            parameters: dict[str, float] = payload.get("parameters", {})
            params_seeded = await self._seed_parameters(parameters)

            # ── Seed validated hypotheses ──────────────────────────────────
            validated: list[dict[str, Any]] = payload.get("validated_hypotheses", [])
            hypotheses_seeded = await self._seed_hypotheses(validated)

            self._log.info(
                "evo_genome_seeded",
                params_seeded=params_seeded,
                hypotheses_seeded=hypotheses_seeded,
                total_params=len(parameters),
                total_hypotheses=len(validated),
            )

            return True

        except Exception as exc:
            self._log.error("evo_genome_seed_failed", error=str(exc))
            return False

    # ─── Extraction Helpers ───────────────────────────────────────────────────

    def _extract_thompson_priors(self) -> list[dict[str, Any]]:
        """
        Extract Thompson sampling Beta priors from active tournaments.
        Child inherits learned exploration/exploitation balance but must
        re-validate through its own experience.
        """
        if self._tournament_engine is None:
            return []
        try:
            tournaments = getattr(self._tournament_engine, "_tournaments", {})
            priors: list[dict[str, Any]] = []
            for tid, t in tournaments.items():
                if not getattr(t, "is_running", False):
                    continue
                beta_params = getattr(t, "beta_parameters", {})
                for hid, beta in beta_params.items():
                    priors.append({
                        "tournament_id": tid,
                        "hypothesis_id": hid,
                        "alpha": beta.alpha,
                        "beta": beta.beta,
                    })
            return priors[:50]  # Cap to prevent genome bloat
        except Exception as exc:
            self._log.debug("evo_genome_thompson_extract_failed", error=str(exc))
            return []

    def _extract_active_experiments(self) -> list[dict[str, Any]]:
        """
        Extract active experiment designs (not results - child must re-validate).
        """
        if self._hypothesis_engine is None:
            return []
        try:
            experiments = getattr(self._hypothesis_engine, "_experiments", {})
            designs: list[dict[str, Any]] = []
            for hid, exp in experiments.items():
                designs.append({
                    "hypothesis_id": hid,
                    "experiment_type": exp.experiment_type,
                    "description": exp.description[:200],
                    "success_criteria": exp.success_criteria[:200],
                })
            return designs[:100]  # Cap
        except Exception as exc:
            self._log.debug("evo_genome_experiment_extract_failed", error=str(exc))
            return []

    def _extract_parameters(self) -> dict[str, float]:
        """
        Get current tunable parameter values from the ParameterTuner.

        Falls back to an empty dict if no tuner is wired.
        """
        if self._parameter_tuner is None:
            self._log.debug("evo_genome_no_parameter_tuner")
            return {}

        try:
            return self._parameter_tuner.get_all_parameters()
        except Exception as exc:
            self._log.warning("evo_genome_parameter_extract_failed", error=str(exc))
            return {}

    async def _extract_validated_hypotheses(self) -> list[dict[str, Any]]:
        """
        Query Neo4j for high-confidence validated hypotheses.

        Selects hypotheses with status INTEGRATED or SUPPORTED and
        confidence > 0.8, ordered by evidence_score descending, capped
        at 300 records.
        """
        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (h:Hypothesis)
                WHERE h.status IN ["integrated", "supported"]
                  AND h.evidence_score >= $min_confidence
                RETURN h.hypothesis_id AS id,
                       h.category AS category,
                       h.statement AS statement,
                       h.status AS status,
                       h.evidence_score AS confidence,
                       h.supporting_count AS supporting_count
                ORDER BY h.evidence_score DESC
                LIMIT $limit
                """,
                {
                    "min_confidence": _MIN_HYPOTHESIS_CONFIDENCE,
                    "limit": _MAX_VALIDATED_HYPOTHESES,
                },
            )
        except Exception as exc:
            self._log.warning(
                "evo_genome_hypothesis_query_failed",
                error=str(exc),
            )
            return []

        hypotheses: list[dict[str, Any]] = []
        for row in records:
            hypotheses.append({
                "id": str(row.get("id", "")),
                "category": str(row.get("category", "")),
                "statement": str(row.get("statement", "")),
                "status": str(row.get("status", "")),
                "confidence": float(row.get("confidence", 0.0)),
                "supporting_count": int(row.get("supporting_count", 0)),
            })

        return hypotheses

    async def _extract_experiment_summaries(self) -> list[dict[str, Any]]:
        """
        Query Neo4j for experiment result summaries.

        Pulls the top 200 experiments ordered by evidence_score, capturing
        the outcome of each hypothesis test.
        """
        try:
            records = await self._neo4j.execute_read(
                """
                MATCH (h:Hypothesis)
                WHERE h.evidence_score IS NOT NULL
                  AND h.status IN ["integrated", "supported", "refuted"]
                RETURN h.hypothesis_id AS hypothesis_id,
                       h.status AS outcome,
                       h.evidence_score AS evidence_score,
                       h.supporting_count + h.contradicting_count AS episodes_evaluated
                ORDER BY h.evidence_score DESC
                LIMIT $limit
                """,
                {"limit": _MAX_EXPERIMENT_SUMMARIES},
            )
        except Exception as exc:
            self._log.warning(
                "evo_genome_experiment_query_failed",
                error=str(exc),
            )
            return []

        summaries: list[dict[str, Any]] = []
        for row in records:
            summaries.append({
                "hypothesis_id": str(row.get("hypothesis_id", "")),
                "outcome": str(row.get("outcome", "")),
                "evidence_score": float(row.get("evidence_score", 0.0)),
                "episodes_evaluated": int(row.get("episodes_evaluated", 0)),
            })

        return summaries

    def _build_consolidation_config(self) -> dict[str, Any]:
        """
        Build the consolidation configuration dict for the genome payload.

        Captures velocity limits and scheduling constants so child instances
        start with the parent's operational tuning.
        """
        from systems.evo.types import VELOCITY_LIMITS

        return {
            "velocity_limits": dict(VELOCITY_LIMITS),
            "consolidation_interval_s": _DEFAULT_CONSOLIDATION_INTERVAL_S,
            "hypothesis_generation_interval": _DEFAULT_HYPOTHESIS_GENERATION_INTERVAL,
            "evidence_evaluation_interval": _DEFAULT_EVIDENCE_EVALUATION_INTERVAL,
        }

    # ─── Seeding Helpers ──────────────────────────────────────────────────────

    async def _seed_parameters(self, parameters: dict[str, float]) -> int:
        """
        Write inherited parameter values to Neo4j EvoParameter nodes.

        Also updates the in-memory ParameterTuner if available.
        Returns count of parameters successfully seeded.
        """
        from systems.evo.types import TUNABLE_PARAMETERS

        seeded = 0
        for name, value in parameters.items():
            # Only seed known tunable parameters
            spec = TUNABLE_PARAMETERS.get(name)
            if spec is None:
                self._log.debug(
                    "evo_genome_skip_unknown_param",
                    parameter=name,
                )
                continue

            # Clamp to valid range
            clamped = max(spec.min_val, min(spec.max_val, float(value)))

            try:
                await self._neo4j.execute_write(
                    """
                    MERGE (p:EvoParameter {name: $name})
                    SET p.current_value = $value,
                        p.last_adjusted = datetime(),
                        p.source = "parent_genome"
                    """,
                    {"name": name, "value": clamped},
                )
                seeded += 1
            except Exception as exc:
                self._log.debug(
                    "evo_genome_seed_param_failed",
                    parameter=name,
                    error=str(exc),
                )

        # Update in-memory tuner if wired
        if self._parameter_tuner is not None and seeded > 0:
            try:
                await self._parameter_tuner.load_from_memory()
            except Exception as exc:
                self._log.warning(
                    "evo_genome_tuner_reload_failed",
                    error=str(exc),
                )

        return seeded

    async def _seed_hypotheses(self, hypotheses: list[dict[str, Any]]) -> int:
        """
        Write inherited validated hypotheses to Neo4j.

        Each hypothesis is created with source="parent_genome" and status
        "testing" so the child re-validates them against its own experience.
        Returns count of hypotheses successfully seeded.
        """
        from primitives.common import new_id, utc_now

        _INHERITED_CONFIDENCE_FACTOR = 0.95

        seeded = 0
        for hyp in hypotheses:
            confidence = float(hyp.get("confidence", 0.0))
            inherited_confidence = confidence * _INHERITED_CONFIDENCE_FACTOR

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
                        inherited_confidence: $inherited_confidence
                    })
                    """,
                    {
                        "hypothesis_id": new_id(),
                        "category": str(hyp.get("category", "")),
                        "statement": str(hyp.get("statement", "")),
                        "evidence_score": inherited_confidence * 3.0,
                        "created_at": utc_now().isoformat(),
                        "inherited_confidence": inherited_confidence,
                    },
                )
                seeded += 1
            except Exception as exc:
                self._log.debug(
                    "evo_genome_seed_hypothesis_failed",
                    statement=str(hyp.get("statement", ""))[:60],
                    error=str(exc),
                )

        return seeded
