"""
EcodiaOS — Nexus: Integration Adapters

Bridges between concrete service implementations and the Nexus protocols.

  LogosWorldModelAdapter   — adapts LogosService.world_model (WorldModel)
                             to LogosWorldModelProtocol
  EvoHypothesisSourceAdapter — adapts EvoService to EvoHypothesisSourceProtocol
  ThymosNexusSinkAdapter   — adapts ThymosService to ThymosDriveSinkProtocol

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
    from systems.logos.service import LogosService
    from systems.nexus.types import DivergencePressure
    from systems.thymos.service import ThymosService

logger = structlog.get_logger("nexus.adapters")


class LogosWorldModelAdapter:
    """
    Adapts LogosService.world_model to LogosWorldModelProtocol.

    Nexus needs to read schema topology, domain coverage, and structural
    fingerprints for divergence measurement and fragment extraction.
    All reads are in-memory — no I/O or LLM calls.
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
        Update the compression_ratio of a schema as a proxy for epistemic
        confidence.  Nexus calls this when convergence raises a fragment's
        triangulation_confidence, feeding the improvement back into the
        world model.

        Returns True if the schema was found and updated.
        """
        schema = self._wm.generative_schemas.get(schema_id)
        if schema is None:
            return False
        # Use compression_ratio as the epistemic confidence carrier.
        # Triangulation confidence replaces the raw ratio when it is higher,
        # embodying the principle that cross-instance validation increases
        # the epistemic status of a schema.
        if new_confidence > schema.compression_ratio:
            schema.compression_ratio = min(new_confidence * 10.0, schema.compression_ratio * 1.5)
            logger.debug(
                "schema_epistemic_confidence_updated",
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
        """
        from primitives.common import new_id, utc_now
        from systems.logos.types import EmpiricalInvariant

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

        High pressure_magnitude means the instance is too similar to peers —
        it should explore frontier domains.  We nudge growth proportionally.
        """
        drive_state = getattr(self._thymos, "_drive_state", None)
        if drive_state is None:
            return

        nudge = pressure.pressure_magnitude * 0.15
        current = getattr(drive_state, "growth", 0.0)
        drive_state.growth = min(1.0, current + nudge)

        logger.debug(
            "divergence_pressure_applied_to_thymos",
            magnitude=round(pressure.pressure_magnitude, 3),
            growth_nudge=round(nudge, 3),
            new_growth=round(drive_state.growth, 3),
        )
