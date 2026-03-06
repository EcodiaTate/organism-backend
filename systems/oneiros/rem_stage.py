"""
EcodiaOS — Oneiros v2: REM Stage (Cross-Domain Synthesis)

The most computationally expensive and most valuable sleep stage.
The workspace is free — no real-time obligations. Use it for breadth.

Three operations:
1. Cross-Domain Synthesis — structural isomorphism detection across schema domains
2. Dream Generation — constructive simulation on highest-error domains
3. Analogy Discovery — causal invariant transfer across domains

Broadcasts:
    CROSS_DOMAIN_MATCH_FOUND — when structural isomorphism > 0.8
    DREAM_HYPOTHESES_GENERATED — when dream predictions fail (quality < 0.7)
    ANALOGY_DISCOVERED — when invariant applies to 2+ domains
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from primitives.common import new_id
from systems.oneiros.types import (
    AbstractStructure,
    AnalogicalTransfer,
    AnalogyDiscoveryReport,
    CrossDomainMatch,
    CrossDomainSynthesisReport,
    DreamGenerationReport,
    DreamScenario,
    PreAttentionEntry,
    REMStageReport,
    SleepCheckpoint,
)
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from systems.logos.service import LogosService
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("oneiros.rem_stage")


# ─── Protocols ──────────────────────────────────────────────────


@runtime_checkable
class FoveaErrorDomainProtocol(Protocol):
    """Get error counts by domain from Fovea."""

    async def get_top_error_domains(self, n: int = 5) -> list[dict[str, Any]]:
        """Return top-N domains by remaining prediction error.

        Each dict: {"domain": str, "error_count": int, "mean_error": float}
        """
        ...


@runtime_checkable
class EvoHypothesisProtocol(Protocol):
    """Extract hypotheses from Evo for dream-based testing."""

    async def extract_new_hypotheses(
        self, *, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Return recent hypotheses not yet integrated.

        Each dict: {"id": str, "statement": str, "category": str,
                     "evidence_score": float, "proposed_mutation": dict | None}
        """
        ...


# ─── Constants ──────────────────────────────────────────────────

# Cross-domain synthesis thresholds
STRONG_MATCH_THRESHOLD: float = 0.8
EVO_CANDIDATE_THRESHOLD: float = 0.9

# Dream generation threshold
LOW_PREDICTION_QUALITY: float = 0.7

# Analogy discovery
MAX_ANALOGIES_PER_CYCLE: int = 10


# ═══════════════════════════════════════════════════════════════════
# Cross-Domain Synthesis
# ═══════════════════════════════════════════════════════════════════


class CrossDomainSynthesizer:
    """
    Compare all schema pairs across domain boundaries for structural isomorphism.

    Algorithm:
    1. Get all schemas from Logos world model
    2. Convert each to AbstractStructure (strip domain labels, keep relational shape)
    3. For all cross-domain pairs: compute isomorphism score
    4. Strong matches (>0.8) → propose unified schema
    5. Very strong matches (>0.9) → submit to Evo as schema candidate
    """

    def __init__(
        self,
        logos: LogosService | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._logos = logos
        self._event_bus = event_bus
        self._logger = logger.bind(component="cross_domain_synthesis")

    async def run(self) -> CrossDomainSynthesisReport:
        """Run full cross-domain structural comparison."""
        t0 = time.monotonic()

        if self._logos is None:
            self._logger.warning("no_logos_available")
            return CrossDomainSynthesisReport()

        # 1. Get all schemas from world model
        schemas = self._logos.world_model.generative_schemas
        if not schemas:
            self._logger.info("no_schemas_to_compare")
            return CrossDomainSynthesisReport()

        # 2. Convert to abstract structures
        abstracts: list[AbstractStructure] = []
        for schema_id, schema in schemas.items():
            abstract = self._to_abstract_structure(schema_id, schema)
            abstracts.append(abstract)

        # 3. Group by domain, then compare across domain boundaries
        by_domain: dict[str, list[AbstractStructure]] = defaultdict(list)
        for a in abstracts:
            by_domain[a.domain].append(a)

        domains = list(by_domain.keys())
        matches: list[CrossDomainMatch] = []
        domain_pairs_evaluated = 0

        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                domain_a = domains[i]
                domain_b = domains[j]
                domain_pairs_evaluated += 1

                for struct_a in by_domain[domain_a]:
                    for struct_b in by_domain[domain_b]:
                        score = self._compute_isomorphism_score(struct_a, struct_b)

                        if score >= STRONG_MATCH_THRESHOLD:
                            match = CrossDomainMatch(
                                schema_a_id=struct_a.schema_id,
                                schema_b_id=struct_b.schema_id,
                                domain_a=domain_a,
                                domain_b=domain_b,
                                isomorphism_score=score,
                                abstract_structure=struct_a,
                                proposed_unified_schema=self._propose_unified(
                                    struct_a, struct_b, score
                                ),
                                mdl_improvement=self._estimate_mdl_improvement(
                                    struct_a, struct_b, score
                                ),
                            )
                            matches.append(match)

                            await self._broadcast_match(match)

                            if score >= EVO_CANDIDATE_THRESHOLD:
                                self._logger.info(
                                    "evo_schema_candidate",
                                    schema_a=struct_a.schema_id,
                                    schema_b=struct_b.schema_id,
                                    score=round(score, 3),
                                )

        strong_matches = len(matches)
        evo_candidates = sum(
            1 for m in matches if m.isomorphism_score >= EVO_CANDIDATE_THRESHOLD
        )
        total_mdl = sum(m.mdl_improvement for m in matches)

        elapsed = (time.monotonic() - t0) * 1000

        self._logger.info(
            "cross_domain_synthesis_complete",
            schemas_compared=len(abstracts),
            domain_pairs=domain_pairs_evaluated,
            strong_matches=strong_matches,
            evo_candidates=evo_candidates,
            total_mdl_improvement=round(total_mdl, 1),
            elapsed_ms=round(elapsed, 1),
        )

        return CrossDomainSynthesisReport(
            schemas_compared=len(abstracts),
            domain_pairs_evaluated=domain_pairs_evaluated,
            strong_matches=strong_matches,
            evo_candidates=evo_candidates,
            matches=matches,
            total_mdl_improvement=total_mdl,
        )

    def _to_abstract_structure(
        self, schema_id: str, schema: Any
    ) -> AbstractStructure:
        """Strip domain labels, keep only relational shape."""
        pattern = getattr(schema, "pattern", {}) or {}

        # Extract relation types from pattern keys
        relation_types: list[str] = []
        relation_arities: list[int] = []

        for _key, value in pattern.items():
            if isinstance(value, dict):
                relation_types.append("nested")
                relation_arities.append(len(value))
            elif isinstance(value, list):
                relation_types.append("collection")
                relation_arities.append(len(value))
            elif isinstance(value, str):
                relation_types.append("attribute")
                relation_arities.append(1)
            elif isinstance(value, (int, float)):
                relation_types.append("scalar")
                relation_arities.append(1)

        # Count entities as top-level pattern keys
        entity_count = len(pattern)

        # Deterministic hash of the relational shape
        shape_repr = f"{entity_count}:{sorted(relation_types)}:{sorted(relation_arities)}"
        pattern_hash = hashlib.sha256(shape_repr.encode()).hexdigest()[:16]

        return AbstractStructure(
            schema_id=schema_id,
            domain=getattr(schema, "domain", "general"),
            entity_count=entity_count,
            relation_types=sorted(relation_types),
            relation_arities=sorted(relation_arities),
            pattern_hash=pattern_hash,
        )

    def _compute_isomorphism_score(
        self, a: AbstractStructure, b: AbstractStructure
    ) -> float:
        """
        Compute structural isomorphism score between two abstract structures.

        Considers: entity count similarity, relation type overlap, arity match.
        Returns 0.0 (no match) to 1.0 (identical structure).
        """
        if a.entity_count == 0 and b.entity_count == 0:
            return 0.0

        # 1. Entity count similarity (40% weight)
        max_entities = max(a.entity_count, b.entity_count, 1)
        min_entities = min(a.entity_count, b.entity_count)
        entity_sim = min_entities / max_entities

        # 2. Relation type overlap (35% weight) — Jaccard similarity
        types_a = set(a.relation_types)
        types_b = set(b.relation_types)
        if types_a or types_b:
            type_overlap = len(types_a & types_b) / len(types_a | types_b)
        else:
            type_overlap = 1.0 if a.entity_count == b.entity_count else 0.0

        # 3. Arity distribution match (25% weight) — histogram similarity
        all_arities = sorted(set(a.relation_arities + b.relation_arities))
        if all_arities:
            hist_a = {ar: a.relation_arities.count(ar) for ar in all_arities}
            hist_b = {ar: b.relation_arities.count(ar) for ar in all_arities}
            total = sum(max(hist_a.get(ar, 0), hist_b.get(ar, 0)) for ar in all_arities)
            intersection = sum(
                min(hist_a.get(ar, 0), hist_b.get(ar, 0)) for ar in all_arities
            )
            arity_sim = intersection / total if total > 0 else 0.0
        else:
            arity_sim = 1.0 if entity_sim == 1.0 else 0.0

        return 0.40 * entity_sim + 0.35 * type_overlap + 0.25 * arity_sim

    def _propose_unified(
        self, a: AbstractStructure, b: AbstractStructure, score: float
    ) -> dict[str, Any]:
        """Propose a unified schema covering both domains."""
        return {
            "id": new_id(),
            "source_schemas": [a.schema_id, b.schema_id],
            "source_domains": [a.domain, b.domain],
            "entity_count": max(a.entity_count, b.entity_count),
            "relation_types": sorted(set(a.relation_types + b.relation_types)),
            "relation_arities": sorted(set(a.relation_arities + b.relation_arities)),
            "isomorphism_score": score,
            "pattern_hash": a.pattern_hash,
            "source": "oneiros_rem_cross_domain",
        }

    def _estimate_mdl_improvement(
        self, a: AbstractStructure, b: AbstractStructure, score: float
    ) -> float:
        """
        Estimate MDL improvement from unifying two schemas.

        1 schema covering 2 domains = dramatic MDL improvement.
        Rough estimate: bits saved = score x min_entity_count x 4.5 bits/entity.
        """
        min_entities = min(a.entity_count, b.entity_count)
        return score * min_entities * 4.5

    async def _broadcast_match(self, match: CrossDomainMatch) -> None:
        if self._event_bus is None:
            return

        event = SynapseEvent(
            event_type=SynapseEventType.CROSS_DOMAIN_MATCH_FOUND,
            source_system="oneiros",
            data={
                "match_id": match.id,
                "schema_a_id": match.schema_a_id,
                "schema_b_id": match.schema_b_id,
                "domain_a": match.domain_a,
                "domain_b": match.domain_b,
                "isomorphism_score": match.isomorphism_score,
                "abstract_structure": match.abstract_structure.model_dump(),
                "proposed_unified_schema": match.proposed_unified_schema,
                "mdl_improvement": match.mdl_improvement,
            },
        )
        await self._event_bus.emit(event)


# ═══════════════════════════════════════════════════════════════════
# Dream Generation
# ═══════════════════════════════════════════════════════════════════


class DreamGenerator:
    """
    Constructive simulation using world model as generator.

    1. Get highest-remaining-error domains from Fovea
    2. For each: generate hypothetical scenarios using Logos predictions
    3. Test world model on these scenarios — measure prediction quality
    4. prediction_quality < 0.7 → extract new hypotheses via Evo
    5. Build pre-attention cache for Fovea's first wake cycles
    """

    def __init__(
        self,
        logos: LogosService | None = None,
        fovea: FoveaErrorDomainProtocol | None = None,
        evo: EvoHypothesisProtocol | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._logos = logos
        self._fovea = fovea
        self._evo = evo
        self._event_bus = event_bus
        self._pre_attention_entries: list[PreAttentionEntry] = []
        self._logger = logger.bind(component="dream_generator")

    @property
    def pre_attention_entries(self) -> list[PreAttentionEntry]:
        return list(self._pre_attention_entries)

    async def run(
        self,
        priority_seeds: list[dict[str, Any]] | None = None,
    ) -> DreamGenerationReport:
        """Run dream generation on highest-error domains."""
        t0 = time.monotonic()

        if self._logos is None:
            self._logger.warning("no_logos_available")
            return DreamGenerationReport()

        # 1. Get highest-error domains from Fovea, then prepend Kairos priority seeds
        error_domains = await self._get_error_domains(priority_seeds=priority_seeds)
        if not error_domains:
            self._logger.info("no_error_domains_found")
            return DreamGenerationReport()

        domains_targeted = len(error_domains)
        scenarios_generated = 0
        low_quality_count = 0
        hypotheses_extracted = 0
        all_hypotheses: list[dict[str, Any]] = []

        # 2. For each domain, generate and test dream scenarios
        for domain_info in error_domains:
            domain = domain_info.get("domain", "general")

            scenarios = await self._generate_scenarios(domain)
            scenarios_generated += len(scenarios)

            for scenario in scenarios:
                # 3. Test world model on this scenario
                prediction = await self._logos.predict(scenario.scenario_context)
                scenario.world_model_prediction = prediction.expected_content
                scenario.prediction_quality = prediction.confidence

                if prediction.confidence < LOW_PREDICTION_QUALITY:
                    low_quality_count += 1

                    # 4. Extract new hypotheses for poor predictions
                    hyps = await self._extract_hypotheses(scenario, domain)
                    scenario.generated_hypotheses = hyps
                    all_hypotheses.extend(hyps)
                    hypotheses_extracted += len(hyps)
                else:
                    # 5. Good prediction → cache for pre-attention
                    entry = PreAttentionEntry(
                        context_key=f"{domain}:{scenario.id}",
                        domain=domain,
                        predicted_content=prediction.expected_content,
                        confidence=prediction.confidence,
                        generating_schema_ids=prediction.generating_schemas,
                    )
                    self._pre_attention_entries.append(entry)

        # Broadcast hypotheses if any generated
        if all_hypotheses:
            await self._broadcast_hypotheses(
                all_hypotheses, [d.get("domain", "") for d in error_domains]
            )

        elapsed = (time.monotonic() - t0) * 1000

        self._logger.info(
            "dream_generation_complete",
            domains=domains_targeted,
            scenarios=scenarios_generated,
            low_quality=low_quality_count,
            hypotheses=hypotheses_extracted,
            pre_attention=len(self._pre_attention_entries),
            elapsed_ms=round(elapsed, 1),
        )

        return DreamGenerationReport(
            domains_targeted=domains_targeted,
            scenarios_generated=scenarios_generated,
            low_quality_predictions=low_quality_count,
            hypotheses_extracted=hypotheses_extracted,
            pre_attention_entries_cached=len(self._pre_attention_entries),
        )

    async def _get_error_domains(
        self,
        priority_seeds: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Get top-5 error domains from Fovea (or schema fallback).

        Kairos priority seeds are prepended so those invariant domains are
        explored first during REM synthesis. Duplicate domains are de-duped
        while preserving the priority seed order.
        """
        if self._fovea is not None:
            fovea_domains = await self._fovea.get_top_error_domains(n=5)
        else:
            # Fallback: use world model schema domains
            if self._logos is None:
                fovea_domains = []
            else:
                schemas = self._logos.world_model.generative_schemas
                domain_counts: dict[str, int] = defaultdict(int)
                for schema in schemas.values():
                    domain = getattr(schema, "domain", "general")
                    domain_counts[domain] += 1
                fovea_domains = [
                    {"domain": d, "error_count": 0, "mean_error": 0.5}
                    for d in sorted(domain_counts.keys())[:5]
                ]

        # Build priority domains from Kairos seeds (untested domains first)
        seen: set[str] = set()
        domains: list[dict[str, Any]] = []
        if priority_seeds:
            for seed in priority_seeds:
                for d in seed.get("untested_domains", []):
                    if d and d not in seen:
                        domains.append({"domain": d, "error_count": 0, "mean_error": 0.5,
                                        "kairos_priority": True})
                        seen.add(d)

        # Append Fovea/fallback domains that are not already included
        for entry in fovea_domains:
            d = entry.get("domain", "")
            if d not in seen:
                domains.append(entry)
                seen.add(d)

        return domains[:5]

    async def _generate_scenarios(
        self, domain: str
    ) -> list[DreamScenario]:
        """Generate hypothetical scenarios for a domain using world model patterns."""
        if self._logos is None:
            return []

        schemas = self._logos.world_model.generative_schemas
        domain_schemas = [
            (sid, s) for sid, s in schemas.items()
            if getattr(s, "domain", "") == domain
        ]

        scenarios: list[DreamScenario] = []
        for schema_id, schema in domain_schemas[:3]:  # max 3 scenarios per domain
            pattern = getattr(schema, "pattern", {}) or {}
            # Generate a variation of the schema pattern as a hypothetical context
            scenario_context = {
                "domain": domain,
                "source_schema": schema_id,
                "hypothetical": True,
                **{k: v for k, v in pattern.items() if not isinstance(v, dict)},
            }
            scenarios.append(
                DreamScenario(
                    domain=domain,
                    scenario_context=scenario_context,
                )
            )

        return scenarios

    async def _extract_hypotheses(
        self, scenario: DreamScenario, domain: str
    ) -> list[dict[str, Any]]:
        """Extract hypotheses from a poor prediction using Evo if available."""
        if self._evo is not None:
            hyps = await self._evo.extract_new_hypotheses(limit=3)
            return [h for h in hyps if h.get("category", "") != "integrated"]

        # Fallback: generate a basic hypothesis from the prediction gap
        return [
            {
                "id": new_id(),
                "statement": (
                    f"World model underperforms in domain '{domain}' — "
                    f"prediction quality {scenario.prediction_quality:.2f}"
                ),
                "category": "world_model",
                "domain": domain,
                "source": "oneiros_dream_generation",
                "evidence_score": 1.0 - scenario.prediction_quality,
            }
        ]

    async def _broadcast_hypotheses(
        self,
        hypotheses: list[dict[str, Any]],
        target_domains: list[str],
    ) -> None:
        if self._event_bus is None:
            return

        event = SynapseEvent(
            event_type=SynapseEventType.DREAM_HYPOTHESES_GENERATED,
            source_system="oneiros",
            data={
                "hypotheses": hypotheses,
                "count": len(hypotheses),
                "target_domains": target_domains,
            },
        )
        await self._event_bus.emit(event)


# ═══════════════════════════════════════════════════════════════════
# Analogy Discovery
# ═══════════════════════════════════════════════════════════════════


class AnalogyDiscoverer:
    """
    Discover causal invariants that apply across multiple domains.

    Same causal structure, different domain labels. When an invariant
    applies to 2+ domains, it's an analogy — and we can delete
    domain-specific schemas, replacing them with transfers.

    predictive_transfer_value = domain_count x invariant.coverage
    mdl_improvement = delete domain-specific schemas, replace with transfers
    """

    def __init__(
        self,
        logos: LogosService | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._logos = logos
        self._event_bus = event_bus
        self._logger = logger.bind(component="analogy_discovery")

    async def run(self) -> AnalogyDiscoveryReport:
        """Scan all causal invariants and find cross-domain analogies."""
        t0 = time.monotonic()

        if self._logos is None:
            self._logger.warning("no_logos_available")
            return AnalogyDiscoveryReport()

        wm = self._logos.world_model

        # 1. Get all causal invariants
        invariants = wm.empirical_invariants
        if not invariants:
            self._logger.info("no_invariants_to_scan")
            return AnalogyDiscoveryReport()

        # 2. Get all causal links to find domain context
        link_domains = self._build_link_domain_map()

        # 3. For each invariant, find all domains it applies to
        transfers: list[AnalogicalTransfer] = []
        for inv in invariants:
            inv_id = getattr(inv, "id", "")
            statement = getattr(inv, "statement", "")
            confidence = getattr(inv, "confidence", 0.0)

            # Parse cause/effect from statement or use domain map
            domains = self._find_invariant_domains(inv, link_domains)

            if len(domains) >= 2:
                domain_count = len(domains)
                # coverage is the invariant's confidence
                predictive_transfer_value = domain_count * confidence
                # MDL improvement: approximate as domain_count x observation_count x 2 bits
                obs_count = getattr(inv, "observation_count", 0)
                mdl_improvement = domain_count * obs_count * 2.0

                transfer = AnalogicalTransfer(
                    invariant_id=inv_id,
                    invariant_statement=statement,
                    source_domains=sorted(domains),
                    domain_count=domain_count,
                    predictive_transfer_value=predictive_transfer_value,
                    mdl_improvement=mdl_improvement,
                )
                transfers.append(transfer)

        # 4. Sort by predictive_transfer_value, take top N
        transfers.sort(key=lambda t: t.predictive_transfer_value, reverse=True)
        top_transfers = transfers[:MAX_ANALOGIES_PER_CYCLE]

        # 5. Apply top analogies to world model
        applied = 0
        for transfer in top_transfers:
            success = await self._apply_analogy(transfer)
            if success:
                applied += 1
                await self._broadcast_analogy(transfer)

        total_mdl = sum(t.mdl_improvement for t in top_transfers)
        elapsed = (time.monotonic() - t0) * 1000

        self._logger.info(
            "analogy_discovery_complete",
            invariants_scanned=len(invariants),
            analogies_found=len(transfers),
            analogies_applied=applied,
            total_mdl_improvement=round(total_mdl, 1),
            elapsed_ms=round(elapsed, 1),
        )

        return AnalogyDiscoveryReport(
            invariants_scanned=len(invariants),
            analogies_found=len(transfers),
            analogies_applied=applied,
            total_mdl_improvement=total_mdl,
            transfers=top_transfers,
        )

    def _build_link_domain_map(self) -> dict[str, set[str]]:
        """Map each causal entity to its domain(s) based on causal links."""
        if self._logos is None:
            return {}

        domain_map: dict[str, set[str]] = defaultdict(set)
        causal = self._logos.world_model.causal_structure
        links = causal.links if hasattr(causal, "links") else {}

        for _link_key, link in links.items():
            domain = getattr(link, "domain", "general")
            cause_id = getattr(link, "cause_id", "")
            effect_id = getattr(link, "effect_id", "")
            if cause_id:
                domain_map[cause_id].add(domain)
            if effect_id:
                domain_map[effect_id].add(domain)

        return domain_map

    def _find_invariant_domains(
        self,
        invariant: Any,
        link_domains: dict[str, set[str]],
    ) -> list[str]:
        """Find all domains an invariant applies to."""
        inv_domain = getattr(invariant, "domain", "general")
        statement = getattr(invariant, "statement", "")

        domains: set[str] = set()

        # If invariant is explicitly cross-domain
        if inv_domain == "cross_domain":
            # Parse entity names from statement "X causes Y"
            parts = statement.split(" causes ")
            if len(parts) == 2:
                cause_entity = parts[0].strip()
                effect_entity = parts[1].strip()
                domains.update(link_domains.get(cause_entity, set()))
                domains.update(link_domains.get(effect_entity, set()))
        else:
            domains.add(inv_domain)

        # Also check the causal structure for matching entities
        if self._logos is not None:
            causal = self._logos.world_model.causal_structure
            links = causal.links if hasattr(causal, "links") else {}
            for _link_key, link in links.items():
                link_statement = f"{link.cause_id} causes {link.effect_id}"
                if (
                    link_statement == statement
                    or getattr(link, "cause_id", "") in statement
                ):
                    link_domain = getattr(link, "domain", "general")
                    if link_domain:
                        domains.add(link_domain)

        # Remove empty domain strings
        domains.discard("")
        return sorted(domains)

    async def _apply_analogy(self, transfer: AnalogicalTransfer) -> bool:
        """Apply an analogical transfer to the world model.

        Replaces domain-specific schemas with the cross-domain invariant,
        reducing total description length.
        """
        if self._logos is None:
            return False

        from systems.logos.types import EmpiricalInvariant

        # Mark the invariant as cross-domain in the world model
        wm = self._logos.world_model
        invariant = EmpiricalInvariant(
            id=transfer.invariant_id or new_id(),
            statement=transfer.invariant_statement,
            domain="cross_domain",
            observation_count=transfer.domain_count,
            confidence=transfer.predictive_transfer_value / max(transfer.domain_count, 1),
            source="oneiros_analogy_discovery",
        )
        wm.ingest_invariant(invariant)

        self._logger.debug(
            "analogy_applied",
            invariant_id=transfer.invariant_id,
            domains=transfer.source_domains,
            mdl_improvement=round(transfer.mdl_improvement, 1),
        )

        return True

    async def _broadcast_analogy(self, transfer: AnalogicalTransfer) -> None:
        if self._event_bus is None:
            return

        event = SynapseEvent(
            event_type=SynapseEventType.ANALOGY_DISCOVERED,
            source_system="oneiros",
            data={
                "transfer_id": transfer.id,
                "invariant_id": transfer.invariant_id,
                "invariant_statement": transfer.invariant_statement,
                "source_domains": transfer.source_domains,
                "domain_count": transfer.domain_count,
                "predictive_transfer_value": transfer.predictive_transfer_value,
                "mdl_improvement": transfer.mdl_improvement,
            },
        )
        await self._event_bus.emit(event)


# ═══════════════════════════════════════════════════════════════════
# REM Stage Orchestrator
# ═══════════════════════════════════════════════════════════════════


class REMStage:
    """
    Stage 3: REM (~30% of sleep duration).

    Cross-domain synthesis. The workspace is free — no real-time obligations.
    Orchestrates three operations:
    1. Cross-Domain Synthesis
    2. Dream Generation
    3. Analogy Discovery

    Broadcasts CROSS_DOMAIN_MATCH_FOUND, DREAM_HYPOTHESES_GENERATED, ANALOGY_DISCOVERED.
    """

    def __init__(
        self,
        logos: LogosService | None = None,
        fovea: FoveaErrorDomainProtocol | None = None,
        evo: EvoHypothesisProtocol | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._logos = logos
        self._fovea = fovea
        self._evo = evo
        self._event_bus = event_bus

        self._synthesizer = CrossDomainSynthesizer(
            logos=logos, event_bus=event_bus
        )
        self._dreamer = DreamGenerator(
            logos=logos, fovea=fovea, evo=evo, event_bus=event_bus
        )
        self._analogy = AnalogyDiscoverer(
            logos=logos, event_bus=event_bus
        )

        self._logger = logger.bind(stage="rem")

    @property
    def pre_attention_entries(self) -> list[PreAttentionEntry]:
        """Pre-attention cache built during dream generation."""
        return self._dreamer.pre_attention_entries

    async def execute(
        self,
        checkpoint: SleepCheckpoint,
        kairos_priority_seeds: list[dict[str, Any]] | None = None,
    ) -> REMStageReport:
        """Execute the full REM stage.

        Args:
            checkpoint: The Descent checkpoint from this sleep cycle.
            kairos_priority_seeds: Priority REM seeds from Kairos (Loop 5).
                These are prepended to Fovea's error domains so Tier 3 invariant
                domains are explored first during dream generation.
        """
        t0 = time.monotonic()
        self._logger.info(
            "rem_stage_starting",
            checkpoint_id=checkpoint.id,
            kairos_seeds=len(kairos_priority_seeds) if kairos_priority_seeds else 0,
        )

        # 1. Cross-Domain Synthesis
        cross_domain_report = await self._synthesizer.run()

        # 2. Dream Generation (Kairos priority seeds bias domain selection)
        dream_report = await self._dreamer.run(priority_seeds=kairos_priority_seeds)

        # 3. Analogy Discovery
        analogy_report = await self._analogy.run()

        elapsed = (time.monotonic() - t0) * 1000

        report = REMStageReport(
            cross_domain=cross_domain_report,
            dreams=dream_report,
            analogies=analogy_report,
            duration_ms=elapsed,
        )

        self._logger.info(
            "rem_stage_complete",
            cross_domain_matches=cross_domain_report.strong_matches,
            evo_candidates=cross_domain_report.evo_candidates,
            scenarios_generated=dream_report.scenarios_generated,
            hypotheses_extracted=dream_report.hypotheses_extracted,
            analogies_found=analogy_report.analogies_found,
            analogies_applied=analogy_report.analogies_applied,
            elapsed_ms=round(elapsed, 1),
        )

        return report
