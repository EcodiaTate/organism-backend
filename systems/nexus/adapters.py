"""
EcodiaOS - Nexus: Integration Adapters

Bridges between concrete service implementations and the Nexus protocols.

  LogosWorldModelAdapter   - adapts LogosService.world_model (WorldModel)
                             to LogosWorldModelProtocol
  EvoHypothesisSourceAdapter - adapts EvoService to EvoHypothesisSourceProtocol
  ThymosNexusSinkAdapter   - adapts ThymosService to ThymosDriveSinkProtocol

These adapters let Nexus work against protocols rather than concrete types,
keeping the speciation boundary clean.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.evo.service import EvoService
    from systems.kairos.pipeline import KairosPipeline
    from systems.logos.service import LogosService
    from systems.nexus.types import DivergencePressure
    from systems.thymos.service import ThymosService

logger = structlog.get_logger("nexus.adapters")


class LogosWorldModelAdapter:
    """
    Adapts LogosService.world_model to LogosWorldModelProtocol.

    Nexus needs to read schema topology, domain coverage, and structural
    fingerprints for divergence measurement and fragment extraction.
    All reads are in-memory - no I/O or LLM calls.
    """

    def __init__(self, logos: LogosService) -> None:
        self._logos = logos

    @property
    def _wm(self) -> Any:
        return self._logos.world_model

    # ─── LogosWorldModelProtocol ─────────────────────────────────

    def get_schema_ids(self) -> list[str]:
        return list(self._wm.generative_schemas.keys())

    def get_schema(self, schema_id: str) -> dict[str, Any] | None:
        schema = self._wm.generative_schemas.get(schema_id)
        if schema is None:
            return None
        return {
            "id": schema.id,
            "name": schema.name,
            "domain": schema.domain,
            "description": schema.description,
            "pattern": schema.pattern,
            "instance_count": schema.instance_count,
            "compression_ratio": schema.compression_ratio,
            "federation_confidence": getattr(schema, "federation_confidence", 0.0),
        }

    def get_domain_coverage(self) -> list[str]:
        """Unique domains present in the world model."""
        domains: set[str] = set()
        for schema in self._wm.generative_schemas.values():
            if schema.domain:
                domains.add(schema.domain)
        return sorted(domains)

    def get_structural_fingerprint(self) -> str:
        """
        Stable hash of the schema topology (names + domains, sorted).

        Two instances with identical schema graphs produce the same fingerprint
        regardless of insertion order or schema IDs.
        """
        schemas = self._wm.generative_schemas
        keys = sorted(
            f"{s.name}:{s.domain}:{s.instance_count}"
            for s in schemas.values()
        )
        raw = json.dumps(keys, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get_total_schemas(self) -> int:
        return len(self._wm.generative_schemas)

    def get_complexity(self) -> float:
        return float(self._wm.current_complexity)

    def get_causal_link_count(self) -> int:
        return int(self._wm.causal_structure.link_count)

    def get_total_experiences(self) -> int:
        return int(self._wm._total_episodes_received)

    # ─── Nexus feedback: update epistemic confidence in the world model ──

    def update_schema_triangulation_confidence(
        self,
        schema_id: str,
        new_confidence: float,
    ) -> bool:
        """
        Update the federation-derived epistemic confidence for a schema.

        Writes to the dedicated `federation_confidence` field on the schema,
        preserving `compression_ratio` for its original Logos-internal semantics
        (MDL scoring, Schwarzschild threshold).

        Returns True if the schema was found and updated.
        """
        schema = self._wm.generative_schemas.get(schema_id)
        if schema is None:
            return False
        if new_confidence > getattr(schema, "federation_confidence", 0.0):
            schema.federation_confidence = min(new_confidence, 1.0)
            logger.debug(
                "schema_federation_confidence_updated",
                schema_id=schema_id,
                new_confidence=round(new_confidence, 3),
            )
        return True

    def ingest_empirical_invariant_from_nexus(
        self,
        abstract_structure: dict[str, Any],
        observations_explained: int,
        triangulation_confidence: float,
    ) -> None:
        """
        Promote a Nexus-confirmed EMPIRICAL_INVARIANT into the Logos world model.

        Level 4 structures that have survived adversarial + competition testing
        across multiple diverse instances are the highest-quality world model
        components.  Logos ingests them as EmpiricalInvariants so they are
        protected from entropic decay.

        Constructs the EmpiricalInvariant lazily - the import is deferred to
        runtime (not module level) to keep the adapter bridge narrowly scoped.
        Adapter files are the sanctioned cross-system bridge; this lazy import
        is the approved pattern for type construction across system boundaries.
        """
        from primitives.common import new_id, utc_now
        from primitives.logos import EmpiricalInvariant

        statement = abstract_structure.get("description", str(abstract_structure)[:120])
        invariant = EmpiricalInvariant(
            id=new_id(),
            statement=statement,
            confidence=triangulation_confidence,
            observation_count=observations_explained,
            last_tested=utc_now(),
            domain=abstract_structure.get("domain", "triangulated"),
            source="nexus_triangulation",
        )
        self._logos.ingest_invariant(invariant)
        logger.info(
            "nexus_invariant_ingested_into_logos",
            statement=statement[:60],
            confidence=round(triangulation_confidence, 3),
        )

    def get_federation_confidence(self, schema_id: str) -> float | None:
        """
        Return the federation-derived confidence for a schema.

        Delegates to the NexusService if wired; returns None otherwise.
        This enables Logos to query Nexus on-demand for epistemic status.
        """
        nexus = getattr(self, "_nexus", None)
        if nexus is not None:
            return nexus.get_federation_confidence(schema_id)
        return None

    def set_nexus(self, nexus: Any) -> None:
        """Wire back-reference to NexusService for on-demand confidence queries."""
        self._nexus = nexus


class EvoHypothesisSourceAdapter:
    """
    Adapts EvoService to EvoHypothesisSourceProtocol.

    Returns the IDs of hypotheses currently under active evaluation,
    used by Nexus's InstanceDivergenceMeasurer for hypothesis diversity scoring.
    """

    def __init__(self, evo: EvoService) -> None:
        self._evo = evo

    def get_active_hypothesis_ids(self) -> list[str]:
        engine = getattr(self._evo, "_hypothesis_engine", None)
        if engine is None:
            return []
        try:
            return [h.id for h in engine.get_all_active()]
        except Exception:
            logger.debug("evo_hypothesis_ids_unavailable", exc_info=True)
            return []


class ThymosNexusSinkAdapter:
    """
    Adapts ThymosService to ThymosDriveSinkProtocol.

    Routes Nexus divergence pressure into Thymos's internal drive state
    as a growth signal.  Thymos's drive_state.growth is nudged up when
    the instance is too similar to the federation average.
    """

    def __init__(self, thymos: ThymosService) -> None:
        self._thymos = thymos

    def receive_divergence_pressure(self, pressure: DivergencePressure) -> None:
        """
        Translate divergence pressure into a Thymos growth drive increment.

        Delegates to ThymosService.receive_divergence_pressure() - the public
        API that encapsulates the drive-state mutation.
        """
        self._thymos.receive_divergence_pressure(pressure)
        logger.debug(
            "divergence_pressure_routed",
            magnitude=round(pressure.pressure_magnitude, 3),
        )


class KairosCausalSourceAdapter:
    """
    Adapts KairosPipeline to KairosCausalSourceProtocol.

    Enables Nexus to pull Tier 3 causal invariants from Kairos,
    closing the bidirectional sync loop.
    """

    def __init__(self, kairos: KairosPipeline) -> None:
        self._kairos = kairos

    def get_tier3_invariants(self) -> list[dict[str, Any]]:
        from primitives.causal import CausalInvariantTier

        hierarchy = self._kairos.hierarchy
        tier3 = hierarchy.get_by_tier(CausalInvariantTier.TIER_3_SUBSTRATE)
        return [self._invariant_to_dict(inv) for inv in tier3]

    def get_invariants_since(self, since_timestamp: str) -> list[dict[str, Any]]:
        from datetime import datetime, timezone
        from primitives.causal import CausalInvariantTier

        try:
            cutoff = datetime.fromisoformat(since_timestamp)
        except (ValueError, TypeError):
            return []

        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)

        hierarchy = self._kairos.hierarchy
        all_invariants = hierarchy.get_all()
        return [
            self._invariant_to_dict(inv)
            for inv in all_invariants
            if getattr(inv, "discovered_at", None) is not None
            and inv.discovered_at >= cutoff
        ]

    @staticmethod
    def _invariant_to_dict(inv: Any) -> dict[str, Any]:
        return {
            "id": inv.id,
            "abstract_form": getattr(inv, "abstract_form", {}),
            "invariance_hold_rate": getattr(inv, "invariance_hold_rate", 0.0),
            "applicable_domains": [
                {
                    "domain": d.domain,
                    "substrate": getattr(d, "substrate", ""),
                    "observation_count": getattr(d, "observation_count", 0),
                }
                for d in getattr(inv, "applicable_domains", [])
            ],
            "description_length_bits": getattr(inv, "description_length_bits", 0.0),
            "discovered_at": str(getattr(inv, "discovered_at", "")),
        }
