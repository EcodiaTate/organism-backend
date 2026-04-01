"""
EcodiaOS - Logos Genome Extraction & Seeding

Implements GenomeExtractionProtocol for the Logos compression engine.

Heritable state:
  1. Generative schemas - top N by MDL compression ratio. These represent
     the organism's learned patterns for compressing reality.
  2. Causal graph edges - high-confidence causal links (strength > 0.3).
  3. Domain priors - predictive priors with sufficient sample count.
  4. Empirical invariants - high-confidence substrate-independent rules.

Payload format:
    {
        "generative_schemas": [
            {"id": str, "name": str, "domain": str, "description": str,
             "pattern": dict, "instance_count": int, "compression_ratio": float},
            ...
        ],
        "causal_edges": [
            {"cause_id": str, "effect_id": str, "strength": float,
             "domain": str, "observations": int},
            ...
        ],
        "domain_priors": [
            {"context_key": str, "variance": float, "sample_count": int},
            ...
        ],
        "empirical_invariants": [
            {"id": str, "statement": str, "domain": str,
             "confidence": float, "source": str},
            ...
        ],
        "metrics": {
            "current_complexity": float,
            "coverage": float,
            "intelligence_ratio": float,
        },
    }

Integration:
    MitosisEngine calls extract_genome_segment() before spawning a child.
    birth.py calls seed_from_genome_segment() after child instance starts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import SystemID
from primitives.genome import OrganGenomeSegment
from systems.genome_helpers import build_segment, verify_segment

if TYPE_CHECKING:
    from systems.logos.world_model import WorldModel

logger = structlog.get_logger()

# ─── Extraction Limits ───────────────────────────────────────────────────────

_MAX_SCHEMAS: int = 100
_MIN_SCHEMA_COMPRESSION_RATIO: float = 0.0  # Include all; sorted by ratio
_MAX_CAUSAL_EDGES: int = 500
_MIN_CAUSAL_STRENGTH: float = 0.3
_MAX_PRIORS: int = 200
_MIN_PRIOR_SAMPLES: int = 2
_MAX_INVARIANTS: int = 50
_MIN_INVARIANT_CONFIDENCE: float = 0.5

# Confidence discount applied to seeded schemas (child starts slightly less certain)
_SEEDED_SCHEMA_INSTANCE_DISCOUNT: float = 0.8


class LogosGenomeExtractor:
    """
    Extracts and seeds Logos's heritable state from the in-memory world model.

    Implements GenomeExtractionProtocol:
        extract_genome_segment() -> OrganGenomeSegment
        seed_from_genome_segment(segment) -> bool
    """

    def __init__(self, world_model: WorldModel) -> None:
        self._world_model = world_model
        self._log = logger.bind(subsystem="logos.genome")

    # ─── GenomeExtractionProtocol ────────────────────────────────────────────

    async def extract_genome_segment(self) -> OrganGenomeSegment:
        """Serialize Logos's heritable state into an OrganGenomeSegment.

        Extracts top schemas by MDL compression ratio, high-confidence
        causal edges, domain priors with sufficient samples, and
        empirical invariants.
        """
        schemas = self._extract_schemas()
        causal_edges = self._extract_causal_edges()
        domain_priors = self._extract_domain_priors()
        invariants = self._extract_invariants()

        payload: dict[str, Any] = {
            "generative_schemas": schemas,
            "causal_edges": causal_edges,
            "domain_priors": domain_priors,
            "empirical_invariants": invariants,
            "metrics": {
                "current_complexity": self._world_model.current_complexity,
                "coverage": self._world_model.coverage,
                "intelligence_ratio": self._world_model.measure_intelligence_ratio(),
            },
        }

        version = 1 if schemas else 0
        segment = build_segment(
            system_id=SystemID.LOGOS,
            payload=payload,
            version=version,
        )

        self._log.info(
            "logos_genome_extracted",
            schemas_count=len(schemas),
            causal_edges_count=len(causal_edges),
            priors_count=len(domain_priors),
            invariants_count=len(invariants),
            size_bytes=segment.size_bytes,
        )

        return segment

    async def seed_from_genome_segment(self, segment: OrganGenomeSegment) -> bool:
        """Restore Logos world model state from a parent's genome segment.

        Returns True if seeding succeeded.
        """
        if not verify_segment(segment):
            self._log.warning("logos_genome_seed_verification_failed")
            return False

        payload = segment.payload
        seeded = 0

        # Seed schemas (with instance count discount)
        from systems.logos.types import GenerativeSchema
        for sdata in payload.get("generative_schemas", []):
            schema = GenerativeSchema(
                id=sdata["id"],
                name=sdata.get("name", ""),
                domain=sdata.get("domain", ""),
                description=sdata.get("description", ""),
                pattern=sdata.get("pattern", {}),
                instance_count=max(
                    1,
                    int(sdata.get("instance_count", 1) * _SEEDED_SCHEMA_INSTANCE_DISCOUNT),
                ),
                compression_ratio=sdata.get("compression_ratio", 0.0),
            )
            self._world_model.register_schema(schema)
            seeded += 1

        # Seed causal edges
        from systems.logos.types import CausalLink
        for edata in payload.get("causal_edges", []):
            link = CausalLink(
                cause_id=edata["cause_id"],
                effect_id=edata["effect_id"],
                strength=edata.get("strength", 0.5),
                domain=edata.get("domain", ""),
                observations=edata.get("observations", 0),
            )
            self._world_model.causal_structure.add_link(link)
            seeded += 1

        # Seed priors
        from systems.logos.types import PriorDistribution
        for pdata in payload.get("domain_priors", []):
            prior = PriorDistribution(
                context_key=pdata["context_key"],
                variance=pdata.get("variance", 1.0),
                sample_count=pdata.get("sample_count", 0),
            )
            self._world_model.predictive_priors[prior.context_key] = prior
            seeded += 1

        # Seed invariants
        from systems.logos.types import EmpiricalInvariant
        for idata in payload.get("empirical_invariants", []):
            invariant = EmpiricalInvariant(
                id=idata["id"],
                statement=idata.get("statement", ""),
                domain=idata.get("domain", ""),
                confidence=idata.get("confidence", 1.0),
                source="genome_inheritance",
            )
            self._world_model.ingest_invariant(invariant)
            seeded += 1

        self._log.info("logos_genome_seeded", total_items=seeded)
        return True

    # ─── Internal extraction helpers ──────────────────────────────────────────

    def _extract_schemas(self) -> list[dict[str, Any]]:
        """Extract top schemas sorted by compression ratio."""
        schemas = sorted(
            self._world_model.generative_schemas.values(),
            key=lambda s: s.compression_ratio,
            reverse=True,
        )[:_MAX_SCHEMAS]

        return [
            {
                "id": s.id,
                "name": s.name,
                "domain": s.domain,
                "description": s.description,
                "pattern": s.pattern,
                "instance_count": s.instance_count,
                "compression_ratio": s.compression_ratio,
            }
            for s in schemas
        ]

    def _extract_causal_edges(self) -> list[dict[str, Any]]:
        """Extract high-confidence causal edges."""
        links = [
            link for link in self._world_model.causal_structure.links.values()
            if link.strength >= _MIN_CAUSAL_STRENGTH
        ]
        links.sort(key=lambda l: l.strength, reverse=True)
        links = links[:_MAX_CAUSAL_EDGES]

        return [
            {
                "cause_id": l.cause_id,
                "effect_id": l.effect_id,
                "strength": l.strength,
                "domain": l.domain,
                "observations": l.observations,
            }
            for l in links
        ]

    def _extract_domain_priors(self) -> list[dict[str, Any]]:
        """Extract priors with sufficient sample count."""
        priors = [
            p for p in self._world_model.predictive_priors.values()
            if p.sample_count >= _MIN_PRIOR_SAMPLES
        ]
        priors.sort(key=lambda p: p.sample_count, reverse=True)
        priors = priors[:_MAX_PRIORS]

        return [
            {
                "context_key": p.context_key,
                "variance": p.variance,
                "sample_count": p.sample_count,
            }
            for p in priors
        ]

    def _extract_invariants(self) -> list[dict[str, Any]]:
        """Extract high-confidence empirical invariants."""
        invariants = [
            inv for inv in self._world_model.empirical_invariants
            if inv.confidence >= _MIN_INVARIANT_CONFIDENCE
        ]
        invariants.sort(key=lambda i: i.confidence, reverse=True)
        invariants = invariants[:_MAX_INVARIANTS]

        return [
            {
                "id": inv.id,
                "statement": inv.statement,
                "domain": inv.domain,
                "confidence": inv.confidence,
                "source": inv.source,
            }
            for inv in invariants
        ]
