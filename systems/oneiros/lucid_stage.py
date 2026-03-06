"""
EcodiaOS — Oneiros v2: Lucid Dreaming Stage (Mutation Testing)

When Simula has mutation proposals, Oneiros enters Lucid Dreaming mode:
1. Fork world model with mutation applied (shadow_world_model)
2. Generate targeted test scenarios specifically challenging the mutated code
3. Run each scenario on both original and mutated world models
4. Produce MutationTestResult for each scenario
5. Aggregate into MutationSimulationReport with recommendation (apply/reject)

Built against SimulaProtocol — does not import Simula directly.
If no mutations pending, skip lucid dreaming entirely.

Broadcasts: LUCID_DREAM_RESULT for each mutation tested.
"""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import structlog

from primitives.common import new_id
from systems.oneiros.types import (
    DreamScenario,
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


# ─── Constants ──────────────────────────────────────────────────

SCENARIOS_PER_MUTATION: int = 5
APPLY_THRESHOLD: float = 0.0  # any positive delta is sufficient


# ═══════════════════════════════════════════════════════════════════
# Lucid Dreaming Stage
# ═══════════════════════════════════════════════════════════════════


class LucidDreamingStage:
    """
    Lucid Dreaming: controlled simulation of Simula mutation proposals.

    When Simula has pending mutations:
    1. Fork world model with mutation applied (shadow)
    2. Generate targeted test scenarios
    3. Run each on original and shadow world models
    4. Compare predictions to produce performance_delta
    5. Check for constitutional violations
    6. Recommend apply or reject

    If no mutations pending, returns empty report immediately.
    """

    def __init__(
        self,
        logos: LogosService | None = None,
        simula: SimulaProtocol | None = None,
        equor: ConstitutionalCheckProtocol | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._logos = logos
        self._simula = simula
        self._equor = equor
        self._event_bus = event_bus
        self._logger = logger.bind(stage="lucid_dreaming")

    async def execute(
        self,
        checkpoint: SleepCheckpoint,
    ) -> LucidDreamingReport:
        """Execute lucid dreaming: test all pending mutations."""
        t0 = time.monotonic()

        # If no Simula wired or no Logos available, skip entirely
        if self._simula is None:
            self._logger.info("no_simula_available", note="skipping_lucid_dreaming")
            return LucidDreamingReport()

        if self._logos is None:
            self._logger.info("no_logos_available", note="skipping_lucid_dreaming")
            return LucidDreamingReport()

        # Get pending mutations
        mutations = await self._simula.get_pending_mutations()
        if not mutations:
            self._logger.info("no_pending_mutations", note="skipping_lucid_dreaming")
            return LucidDreamingReport()

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
            duration_ms=elapsed,
        )

        self._logger.info(
            "lucid_dreaming_complete",
            mutations_tested=len(mutations),
            apply=apply_count,
            reject=reject_count,
            violations=violation_count,
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
            from systems.logos.types import GenerativeSchema

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
                note="shadow model returned unmutated — mutation may be rejected unfairly",
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
            schemas = self._logos.world_model.generative_schemas
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
