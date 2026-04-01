"""
EcodiaOS - Oneiros v2: Lucid Dreaming Stage (Mutation Testing)

When Simula has mutation proposals, Oneiros enters Lucid Dreaming mode:
1. Fork world model with mutation applied (shadow_world_model)
2. Generate targeted test scenarios specifically challenging the mutated code
3. Run each scenario on both original and mutated world models
4. Produce MutationTestResult for each scenario
5. Aggregate into MutationSimulationReport with recommendation (apply/reject)

Built against SimulaProtocol - does not import Simula directly.
If no mutations pending, skip lucid dreaming entirely.

Broadcasts: LUCID_DREAM_RESULT for each mutation tested.
"""

from __future__ import annotations

import copy
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from primitives.common import new_id, utc_now
from systems.oneiros.types import (
    Dream,
    DreamCoherence,
    DreamInsight,
    DreamScenario,
    DreamType,
    InsightStatus,
    LucidDreamingReport,
    MutationSimulationReport,
    MutationTestResult,
    SleepCheckpoint,
)
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from systems.logos.service import LogosService
    from systems.logos.world_model import WorldModel
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("oneiros.lucid_stage")


# ─── Protocols ──────────────────────────────────────────────────


@runtime_checkable
class SimulaProtocol(Protocol):
    """Read pending mutation proposals from Simula without importing it."""

    async def get_pending_mutations(self) -> list[dict[str, Any]]:
        """Return mutation proposals awaiting simulation.

        Each dict: {
            "id": str,
            "description": str,
            "mutation_type": str,
            "target": str,
            "value": float,
            "affected_systems": list[str],
        }
        """
        ...

    async def report_simulation_result(
        self, mutation_id: str, recommendation: str, report: dict[str, Any]
    ) -> None:
        """Report the simulation result back to Simula.

        recommendation: "apply" | "reject"
        """
        ...


@runtime_checkable
class ConstitutionalCheckProtocol(Protocol):
    """Check mutations against constitutional constraints."""

    async def check_mutation(self, mutation: dict[str, Any]) -> dict[str, Any]:
        """Return {"passes": bool, "violations": list[str]}."""
        ...


# ─── MetaCognition ──────────────────────────────────────────────


class MetaCognition:
    """
    Self-reflective analysis of the organism's own dream patterns (Spec 13 §4.5).

    Queries the DreamJournal for recurring themes across recent Dreams,
    clusters them by semantic similarity (Jaccard over theme sets), and
    promotes high-frequency theme clusters to CONCEPT nodes in Neo4j
    with is_core_identity=True.  No LLM - pure graph + set arithmetic.

    Runs every lucid stage regardless of whether Simula has mutations.
    """

    THEME_WINDOW_DAYS: int = 30
    MIN_CLUSTER_FREQUENCY: int = 3        # themes must appear ≥3 dreams to persist
    JACCARD_THRESHOLD: float = 0.25       # min overlap to merge theme clusters

    def __init__(self, neo4j: Any) -> None:
        self._neo4j = neo4j
        self._logger = logger.bind(worker="meta_cognition")

    async def run(self, sleep_cycle_id: str) -> dict[str, Any]:
        """Discover recurring theme clusters → CONCEPT nodes. Returns summary dict."""
        if self._neo4j is None:
            return {"concepts_discovered": 0, "concepts_promoted": 0}

        try:
            theme_freq = await self._collect_theme_frequencies()
            clusters = self._cluster_themes(theme_freq)
            promoted = await self._promote_concepts(clusters, sleep_cycle_id)
            self._logger.info(
                "meta_cognition_complete",
                theme_count=len(theme_freq),
                clusters=len(clusters),
                promoted=promoted,
            )
            return {
                "concepts_discovered": len(clusters),
                "concepts_promoted": promoted,
            }
        except Exception:
            self._logger.exception("meta_cognition_error")
            return {"concepts_discovered": 0, "concepts_promoted": 0}

    async def _collect_theme_frequencies(self) -> Counter:
        """Query recent Dreams for their theme lists."""
        query = """
        MATCH (d:Dream)
        WHERE d.timestamp >= datetime() - duration({days: $days})
        RETURN d.themes AS themes
        """
        try:
            result = await self._neo4j.execute_read(
                query, {"days": self.THEME_WINDOW_DAYS}
            )
            freq: Counter = Counter()
            for record in result.records:
                for theme in (record["themes"] or []):
                    freq[theme] += 1
            return freq
        except Exception:
            self._logger.exception("meta_cognition_theme_query_error")
            return Counter()

    def _cluster_themes(self, freq: Counter) -> list[list[str]]:
        """
        Greedy Jaccard clustering of co-occurring themes.
        Returns list of clusters (each a list of related theme strings).
        """
        frequent = [t for t, c in freq.items() if c >= self.MIN_CLUSTER_FREQUENCY]
        if not frequent:
            return []

        clusters: list[list[str]] = []
        for theme in sorted(frequent, key=lambda t: -freq[t]):
            merged = False
            for cluster in clusters:
                cluster_set = set(cluster)
                theme_set = {theme}
                intersection = len(cluster_set & theme_set)
                union = len(cluster_set | theme_set)
                if union > 0 and intersection / union >= self.JACCARD_THRESHOLD:
                    cluster.append(theme)
                    merged = True
                    break
            if not merged:
                clusters.append([theme])

        return clusters

    async def _promote_concepts(
        self, clusters: list[list[str]], sleep_cycle_id: str
    ) -> int:
        """MERGE each high-frequency cluster as a CONCEPT node in Neo4j."""
        promoted = 0
        for cluster in clusters:
            primary_theme = cluster[0]
            concept_id = new_id()
            query = """
            MERGE (c:CONCEPT {name: $name})
            ON CREATE SET
                c.id = $id,
                c.is_core_identity = true,
                c.themes = $themes,
                c.first_seen_cycle = $cycle_id,
                c.created_at = datetime()
            ON MATCH SET
                c.themes = $themes,
                c.last_reinforced_cycle = $cycle_id,
                c.last_reinforced_at = datetime()
            RETURN c.id
            """
            try:
                await self._neo4j.execute_write(
                    query,
                    {
                        "name": primary_theme,
                        "id": concept_id,
                        "themes": cluster,
                        "cycle_id": sleep_cycle_id,
                    },
                )
                promoted += 1
            except Exception:
                self._logger.exception("concept_promote_error", theme=primary_theme)
        return promoted


# ─── DirectedExploration ────────────────────────────────────────


class DirectedExploration:
    """
    Systematic variation of creative goals and high-coherence dream insights (Spec 13 §4.5).

    Takes either:
    - A creative_goal string (from OneirosService._creative_goal)
    - High-coherence DreamInsights (coherence ≥ 0.85) from the last sleep cycle

    Generates systematic variations using 4 operators:
    1. Domain transfer  - "what if applied to domain X?"
    2. Negation         - "what's the opposite / inverse?"
    3. Amplification    - "taken to the extreme?"
    4. Constraint       - "with resource constraint Y?"

    Stores each variation as a DreamInsight (InsightStatus.PENDING) in Neo4j.
    No LLM - operator templates applied programmatically.
    """

    OPERATORS: list[tuple[str, str]] = [
        ("domain_transfer", "What if '{insight}' applied to domain '{domain}'?"),
        ("negation", "What is the opposite or inverse of: '{insight}'?"),
        ("amplification", "Taken to the extreme, '{insight}' implies what?"),
        ("constraint", "Under tight resource constraints, how does '{insight}' change?"),
    ]
    HIGH_COHERENCE_THRESHOLD: float = 0.85
    MAX_SOURCE_INSIGHTS: int = 5           # cap to avoid combinatorial explosion
    EXPLORATION_DOMAINS: list[str] = [
        "memory", "causal", "economic", "social", "temporal", "structural",
    ]

    def __init__(self, neo4j: Any) -> None:
        self._neo4j = neo4j
        self._logger = logger.bind(worker="directed_exploration")

    async def run(
        self,
        sleep_cycle_id: str,
        creative_goal: str | None = None,
    ) -> dict[str, Any]:
        """Generate exploration variations. Returns summary dict."""
        if self._neo4j is None:
            return {"variations_generated": 0}

        try:
            sources = await self._gather_sources(sleep_cycle_id, creative_goal)
            if not sources:
                return {"variations_generated": 0}

            count = 0
            for source_text, domain in sources:
                for operator_name, template in self.OPERATORS:
                    variation_text = self._apply_operator(
                        template, source_text, domain
                    )
                    await self._store_insight(
                        variation_text, domain, sleep_cycle_id, operator_name
                    )
                    count += 1

            self._logger.info(
                "directed_exploration_complete",
                sources=len(sources),
                variations=count,
            )
            return {"variations_generated": count}
        except Exception:
            self._logger.exception("directed_exploration_error")
            return {"variations_generated": 0}

    async def _gather_sources(
        self, sleep_cycle_id: str, creative_goal: str | None
    ) -> list[tuple[str, str]]:
        """Collect (text, domain) tuples from goal + high-coherence insights."""
        sources: list[tuple[str, str]] = []

        # 1. creative_goal from service
        if creative_goal:
            sources.append((creative_goal, "general"))

        # 2. High-coherence DreamInsights from this cycle
        query = """
        MATCH (i:DreamInsight {sleep_cycle_id: $cycle_id})
        WHERE i.coherence_score >= $threshold
          AND i.status = 'pending'
        RETURN i.insight_text AS text, i.domain AS domain
        ORDER BY i.coherence_score DESC
        LIMIT $limit
        """
        try:
            result = await self._neo4j.execute_read(
                query,
                {
                    "cycle_id": sleep_cycle_id,
                    "threshold": self.HIGH_COHERENCE_THRESHOLD,
                    "limit": self.MAX_SOURCE_INSIGHTS,
                },
            )
            for record in result.records:
                sources.append((record["text"] or "", record["domain"] or "general"))
        except Exception:
            self._logger.exception("directed_exploration_source_query_error")

        return sources[: self.MAX_SOURCE_INSIGHTS]

    def _apply_operator(self, template: str, insight: str, domain: str) -> str:
        """Fill operator template. Picks a random exploration domain if none given."""
        import random
        exp_domain = domain if domain != "general" else random.choice(
            self.EXPLORATION_DOMAINS
        )
        return template.format(insight=insight[:200], domain=exp_domain)

    async def _store_insight(
        self,
        variation_text: str,
        domain: str,
        sleep_cycle_id: str,
        operator_name: str,
    ) -> None:
        """Persist variation as a DreamInsight node in Neo4j."""
        insight_id = new_id()
        query = """
        CREATE (i:DreamInsight {
            id: $id,
            dream_id: $dream_id,
            sleep_cycle_id: $cycle_id,
            insight_text: $text,
            domain: $domain,
            coherence_score: 0.5,
            status: 'pending',
            source_operator: $operator,
            created_at: datetime()
        })
        """
        try:
            await self._neo4j.execute_write(
                query,
                {
                    "id": insight_id,
                    "dream_id": "directed_exploration",
                    "cycle_id": sleep_cycle_id,
                    "text": variation_text,
                    "domain": domain,
                    "operator": operator_name,
                },
            )
        except Exception:
            self._logger.exception(
                "directed_exploration_store_error", operator=operator_name
            )


# ─── Constants ──────────────────────────────────────────────────

SCENARIOS_PER_MUTATION: int = 5
APPLY_THRESHOLD: float = 0.0  # any positive delta is sufficient


# ═══════════════════════════════════════════════════════════════════
# Lucid Dreaming Stage
# ═══════════════════════════════════════════════════════════════════


class LucidDreamingStage:
    """
    Lucid Dreaming: controlled simulation of Simula mutation proposals,
    metacognitive self-reflection, and directed creative exploration (Spec 13 §4.5).

    Always runs:
    - MetaCognition  - clusters recurring dream themes → CONCEPT nodes
    - DirectedExploration - generates systematic variations from creative_goal
                            and high-coherence insights

    When Simula has pending mutations:
    1. Fork world model with mutation applied (shadow)
    2. Generate targeted test scenarios
    3. Run each on original and shadow world models
    4. Compare predictions to produce performance_delta
    5. Check for constitutional violations
    6. Recommend apply or reject
    """

    def __init__(
        self,
        logos: LogosService | None = None,
        simula: SimulaProtocol | None = None,
        equor: ConstitutionalCheckProtocol | None = None,
        event_bus: EventBus | None = None,
        neo4j: Any = None,
        creative_goal: str | None = None,
    ) -> None:
        self._logos = logos
        self._simula = simula
        self._equor = equor
        self._event_bus = event_bus
        self._creative_goal = creative_goal
        self._logger = logger.bind(stage="lucid_dreaming")
        self._meta_cognition = MetaCognition(neo4j=neo4j)
        self._directed_exploration = DirectedExploration(neo4j=neo4j)

    async def execute(
        self,
        checkpoint: SleepCheckpoint,
    ) -> LucidDreamingReport:
        """Execute lucid dreaming: metacognition + directed exploration + mutation testing."""
        t0 = time.monotonic()

        # Always run metacognition and directed exploration (Spec 13 §4.5)
        meta_result = await self._meta_cognition.run(sleep_cycle_id=checkpoint.id)
        exploration_result = await self._directed_exploration.run(
            sleep_cycle_id=checkpoint.id,
            creative_goal=self._creative_goal,
        )

        # Mutation testing requires both Simula and Logos
        if self._simula is None:
            self._logger.info("no_simula_available", note="skipping_mutation_testing")
            elapsed = (time.monotonic() - t0) * 1000
            return LucidDreamingReport(
                concepts_discovered=meta_result.get("concepts_discovered", 0),
                variations_generated=exploration_result.get("variations_generated", 0),
                duration_ms=elapsed,
            )

        if self._logos is None:
            self._logger.info("no_logos_available", note="skipping_mutation_testing")
            elapsed = (time.monotonic() - t0) * 1000
            return LucidDreamingReport(
                concepts_discovered=meta_result.get("concepts_discovered", 0),
                variations_generated=exploration_result.get("variations_generated", 0),
                duration_ms=elapsed,
            )

        # Get pending mutations
        mutations = await self._simula.get_pending_mutations()
        if not mutations:
            self._logger.info("no_pending_mutations", note="skipping_mutation_testing")
            elapsed = (time.monotonic() - t0) * 1000
            return LucidDreamingReport(
                concepts_discovered=meta_result.get("concepts_discovered", 0),
                variations_generated=exploration_result.get("variations_generated", 0),
                duration_ms=elapsed,
            )

        self._logger.info(
            "lucid_dreaming_starting",
            mutations_pending=len(mutations),
            checkpoint_id=checkpoint.id,
        )

        reports: list[MutationSimulationReport] = []
        apply_count = 0
        reject_count = 0
        violation_count = 0

        for mutation in mutations:
            mutation_id = mutation.get("id", new_id())
            description = mutation.get("description", "")

            sim_report = await self._test_mutation(mutation)
            reports.append(sim_report)

            if sim_report.any_constitutional_violations:
                violation_count += len(sim_report.violation_details)

            if sim_report.recommendation == "apply":
                apply_count += 1
            else:
                reject_count += 1

            # Report result back to Simula
            await self._simula.report_simulation_result(
                mutation_id=mutation_id,
                recommendation=sim_report.recommendation,
                report=sim_report.model_dump(),
            )

            # Broadcast result
            await self._broadcast_result(sim_report)

            self._logger.info(
                "mutation_tested",
                mutation_id=mutation_id,
                description=description[:80],
                recommendation=sim_report.recommendation,
                performance_delta=round(sim_report.overall_performance_delta, 4),
                violations=sim_report.any_constitutional_violations,
            )

        elapsed = (time.monotonic() - t0) * 1000

        report = LucidDreamingReport(
            mutations_tested=len(mutations),
            mutations_recommended_apply=apply_count,
            mutations_recommended_reject=reject_count,
            constitutional_violations_found=violation_count,
            reports=reports,
            concepts_discovered=meta_result.get("concepts_discovered", 0),
            variations_generated=exploration_result.get("variations_generated", 0),
            duration_ms=elapsed,
        )

        self._logger.info(
            "lucid_dreaming_complete",
            mutations_tested=len(mutations),
            apply=apply_count,
            reject=reject_count,
            violations=violation_count,
            concepts_discovered=report.concepts_discovered,
            variations_generated=report.variations_generated,
            elapsed_ms=round(elapsed, 1),
        )

        return report

    async def _test_mutation(
        self, mutation: dict[str, Any]
    ) -> MutationSimulationReport:
        """Test a single mutation proposal against the world model."""
        mutation_id = mutation.get("id", new_id())
        description = mutation.get("description", "")

        # 1. Fork world model with mutation applied
        shadow_wm = self._fork_world_model(mutation)
        if shadow_wm is None:
            return MutationSimulationReport(
                mutation_id=mutation_id,
                mutation_description=description,
                recommendation="reject",
            )

        # 2. Generate targeted test scenarios
        scenarios = await self._generate_test_scenarios(mutation)

        # 3. Run each scenario on both models, collect results
        results: list[MutationTestResult] = []
        total_delta = 0.0
        violations: list[str] = []

        for scenario in scenarios:
            result = await self._run_scenario(
                scenario, shadow_wm, mutation
            )
            results.append(result)
            total_delta += result.performance_delta

            if result.constitutional_violation:
                violations.append(result.violation_detail)

        # 4. Aggregate
        avg_delta = total_delta / max(len(results), 1)
        has_violations = len(violations) > 0

        # 5. Recommendation
        if has_violations:
            recommendation = "reject"
        elif avg_delta > APPLY_THRESHOLD:
            recommendation = "apply"
        else:
            recommendation = "reject"

        return MutationSimulationReport(
            mutation_id=mutation_id,
            mutation_description=description,
            scenarios_tested=len(scenarios),
            results=results,
            overall_performance_delta=avg_delta,
            any_constitutional_violations=has_violations,
            violation_details=violations,
            recommendation=recommendation,
        )

    def _fork_world_model(
        self, mutation: dict[str, Any]
    ) -> WorldModel | None:
        """Create a shadow copy of the world model with the mutation applied."""
        if self._logos is None:
            return None

        original = self._logos.world_model

        # Deep copy the world model state
        shadow = copy.deepcopy(original)

        # Apply the mutation to the shadow
        mutation_type = mutation.get("mutation_type", "")
        target = mutation.get("target", "")
        value = mutation.get("value", 0.0)

        if mutation_type == "parameter_adjustment" and target:
            if hasattr(shadow, "predictive_priors") and target in shadow.predictive_priors:
                prior = shadow.predictive_priors[target]
                if hasattr(prior, "mean"):
                    prior.mean += value

        elif mutation_type == "schema_addition":
            from primitives.logos import GenerativeSchema

            schema = GenerativeSchema(
                name=target,
                domain=mutation.get("domain", "general"),
                description=mutation.get("description", ""),
                pattern=mutation.get("pattern", {}),
            )
            shadow.generative_schemas[schema.id] = schema

        elif mutation_type == "schema_removal" and target:
            # Remove a schema from the shadow to test whether it's redundant
            shadow.generative_schemas.pop(target, None)

        elif mutation_type == "causal_link_revision" and target:
            # Revise a causal link strength in the shadow model
            if "->" in target:
                cause, effect = target.split("->", 1)
                shadow.causal_structure.revise_link(
                    cause.strip(), effect.strip(), value
                )

        elif mutation_type == "complexity_reduction":
            # Prune weak links to test model simplification
            threshold = value if value > 0 else 0.1
            shadow.causal_structure.remove_weak_links(threshold=threshold)

        else:
            self._logger.warning(
                "unknown_mutation_type",
                mutation_type=mutation_type,
                target=target,
                note="shadow model returned unmutated - mutation may be rejected unfairly",
            )

        self._logger.debug(
            "world_model_forked",
            mutation_type=mutation_type,
            target=target,
        )

        return shadow

    async def _generate_test_scenarios(
        self, mutation: dict[str, Any]
    ) -> list[DreamScenario]:
        """Generate targeted scenarios that specifically challenge the mutation."""
        scenarios: list[DreamScenario] = []
        target = mutation.get("target", "general")
        mutation_type = mutation.get("mutation_type", "")
        domain = mutation.get(
            "domain",
            target.split(".")[0] if "." in target else "general",
        )

        if self._logos is not None:
            schemas = self._logos.get_generative_schemas()
            domain_schemas = [
                (sid, s) for sid, s in schemas.items()
                if getattr(s, "domain", "") == domain
            ]

            for schema_id, schema in domain_schemas[:SCENARIOS_PER_MUTATION]:
                pattern = getattr(schema, "pattern", {}) or {}
                ctx = {
                    "domain": domain,
                    "source_schema": schema_id,
                    "mutation_target": target,
                    "mutation_type": mutation_type,
                    "hypothetical": True,
                    **{k: v for k, v in pattern.items() if not isinstance(v, dict)},
                }
                scenarios.append(DreamScenario(domain=domain, scenario_context=ctx))

        # Fill remaining with generic scenarios
        while len(scenarios) < SCENARIOS_PER_MUTATION:
            ctx = {
                "domain": domain,
                "mutation_target": target,
                "mutation_type": mutation_type,
                "hypothetical": True,
                "scenario_index": len(scenarios),
            }
            scenarios.append(DreamScenario(domain=domain, scenario_context=ctx))

        return scenarios[:SCENARIOS_PER_MUTATION]

    async def _run_scenario(
        self,
        scenario: DreamScenario,
        shadow_wm: WorldModel,
        mutation: dict[str, Any],
    ) -> MutationTestResult:
        """Run a scenario on both original and mutated world models."""
        if self._logos is None:
            return MutationTestResult(scenario=scenario)

        # Original prediction
        original_prediction = await self._logos.predict(scenario.scenario_context)

        # Shadow prediction
        shadow_prediction = await shadow_wm.predict(scenario.scenario_context)

        # Positive delta = mutation is better
        performance_delta = shadow_prediction.confidence - original_prediction.confidence

        # Constitutional check
        violation = False
        violation_detail = ""
        if self._equor is not None:
            check = await self._equor.check_mutation(mutation)
            if not check.get("passes", True):
                violation = True
                violation_detail = "; ".join(check.get("violations", []))

        return MutationTestResult(
            scenario=scenario,
            original_prediction=original_prediction.expected_content,
            mutated_prediction=shadow_prediction.expected_content,
            performance_delta=performance_delta,
            constitutional_violation=violation,
            violation_detail=violation_detail,
        )

    async def _broadcast_result(self, report: MutationSimulationReport) -> None:
        if self._event_bus is None:
            return

        event = SynapseEvent(
            event_type=SynapseEventType.LUCID_DREAM_RESULT,
            source_system="oneiros",
            data={
                "mutation_id": report.mutation_id,
                "mutation_description": report.mutation_description,
                "scenarios_tested": report.scenarios_tested,
                "overall_performance_delta": report.overall_performance_delta,
                "any_constitutional_violations": report.any_constitutional_violations,
                "violation_details": report.violation_details,
                "recommendation": report.recommendation,
            },
        )
        await self._event_bus.emit(event)
