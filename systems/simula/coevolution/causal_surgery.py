"""
EcodiaOS -- Causal Self-Surgery Engine (Prompt #16)

When Evo detects a consistent failure pattern (e.g. "policy type X fails
when market volatility > 30%"), this module uses do-calculus on the
episode DAG to identify the **causal intervention point** - which system,
which parameter, which decision produced the failure - then generates a
surgical Simula proposal targeting only that node.

Two-tier counterfactual approach:
  Tier 1 (rule-based, zero LLM tokens): For tunable parameters in
    TUNABLE_PARAMETERS, propagate perturbation through the DAG using
    a linear SCM approximation.
  Tier 2 (LLM, batched): For complex conditional interventions,
    batch 10 episodes per LLM call (~800 tokens).

The cognitive cycle's known causal order (Percept → Belief → Goal →
Policy → Action → Outcome) satisfies the front-door criterion - no
unobserved confounders between consecutive stages.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id
from systems.evo.types import (
    PARAMETER_DEFAULTS,
    TUNABLE_PARAMETERS,
)
from systems.simula.coevolution.causal_surgery_types import (
    CausalInterventionPoint,
    CausalSurgeryResult,
    CognitiveCausalDAG,
    CognitiveEdge,
    CognitiveNode,
    CognitiveStage,
    CounterfactualScenario,
    FailurePattern,
    InterventionDirection,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from clients.neo4j import Neo4jClient
    from systems.evo.types import RegretStats

logger = structlog.get_logger()

# ── Stage → system mapping ─────────────────────────────────────────────────

_STAGE_SYSTEM: dict[CognitiveStage, str] = {
    CognitiveStage.PERCEPT: "atune",
    CognitiveStage.BELIEF_UPDATE: "memory",
    CognitiveStage.GOAL_SELECTION: "nova",
    CognitiveStage.POLICY_GENERATION: "nova",
    CognitiveStage.POLICY_SELECTION: "nova",
    CognitiveStage.EQUOR_REVIEW: "equor",
    CognitiveStage.ACTION_EXECUTION: "axon",
    CognitiveStage.OUTCOME: "memory",
}

# Which parameters are relevant at each cognitive stage
_STAGE_PARAMETERS: dict[CognitiveStage, list[str]] = {
    CognitiveStage.PERCEPT: [
        "atune.head.novelty.weight",
        "atune.head.risk.weight",
        "atune.head.identity.weight",
        "atune.head.goal.weight",
        "atune.head.emotional.weight",
        "atune.head.causal.weight",
        "atune.head.keyword.weight",
    ],
    CognitiveStage.BELIEF_UPDATE: [
        "memory.salience.recency",
        "memory.salience.frequency",
        "memory.salience.affect",
        "memory.salience.surprise",
        "memory.salience.relevance",
        "nova.fe_budget.budget_nats",
        "nova.fe_budget.threshold_fraction",
    ],
    CognitiveStage.GOAL_SELECTION: [],  # Goal selection is not directly tunable
    CognitiveStage.POLICY_GENERATION: [
        "nova.efe.cognition_cost",
    ],
    CognitiveStage.POLICY_SELECTION: [
        "nova.efe.pragmatic",
        "nova.efe.epistemic",
        "nova.efe.constitutional",
        "nova.efe.feasibility",
        "nova.efe.risk",
    ],
    CognitiveStage.EQUOR_REVIEW: [],  # Equor is immutable
    CognitiveStage.ACTION_EXECUTION: [],  # Axon execution is not tunable
    CognitiveStage.OUTCOME: [],
}

# Stages eligible for intervention (not percept - given; not outcome - observed)
_INTERVENTION_ELIGIBLE: frozenset[CognitiveStage] = frozenset({
    CognitiveStage.BELIEF_UPDATE,
    CognitiveStage.POLICY_GENERATION,
    CognitiveStage.POLICY_SELECTION,
})

# LLM prompt for batched counterfactual estimation
_COUNTERFACTUAL_PROMPT = """You are the causal reasoning module of EcodiaOS.

A recurring failure pattern has been detected:
{pattern_description}

Below are {count} episodes from this pattern. For each, determine:
If the system had applied this intervention: "{intervention}",
would the episode outcome have changed from failure to success?

EPISODES:
{episode_list}

Reply as a numbered list:
<number>. <yes|no> | <confidence 0.0-1.0> | <1 sentence reason>
Only include episodes where the answer is 'yes'."""

# ── Minimum thresholds ─────────────────────────────────────────────────────

_MIN_EPISODES_FOR_PATTERN = 5
_MIN_EPISODES_FOR_CONFIDENCE = 3
_MIN_SUCCESS_RATE = 0.3  # Intervention must help at least 30% of episodes
_Z_SCORE_95 = 1.96  # For Wilson confidence interval


# ═══════════════════════════════════════════════════════════════════════════
# CausalDAGBuilder
# ═══════════════════════════════════════════════════════════════════════════


class CausalDAGBuilder:
    """
    Constructs episode-level cognitive causal chains from Neo4j episode data.

    Rule-based (zero LLM tokens). Reconstructs the 8-stage cognitive chain
    by mapping stored episode properties to tunable parameter values.
    """

    def __init__(self, neo4j: Neo4jClient) -> None:
        self._neo4j = neo4j
        self._log = logger.bind(system="simula.causal_surgery.dag_builder")

    async def build_dag_for_episode(
        self, episode_id: str,
    ) -> CognitiveCausalDAG | None:
        """
        Query Neo4j for an episode and its linked context, then
        reconstruct the cognitive causal chain.

        Returns None if the episode is not found.
        """
        try:
            results = await self._neo4j.execute_read(
                """
                MATCH (e:Episode {id: $episode_id})
                OPTIONAL MATCH (prev:Episode)-[fb:FOLLOWED_BY]->(e)
                OPTIONAL MATCH (cf:Counterfactual)-[:ALTERNATIVE_TO]->(e)
                RETURN e,
                    collect(DISTINCT {
                        id: prev.id,
                        summary: prev.summary,
                        source: prev.source,
                        free_energy: prev.free_energy,
                        salience_composite: prev.salience_composite,
                        causal_strength: fb.causal_strength
                    }) AS predecessors,
                    collect(DISTINCT {
                        id: cf.id,
                        policy_type: cf.policy_type,
                        policy_name: cf.policy_name,
                        efe_total: cf.efe_total,
                        estimated_pragmatic_value: cf.estimated_pragmatic_value,
                        estimated_epistemic_value: cf.estimated_epistemic_value,
                        constitutional_alignment: cf.constitutional_alignment,
                        feasibility: cf.feasibility,
                        risk_expected_harm: cf.risk_expected_harm,
                        regret: cf.regret,
                        resolved: cf.resolved
                    }) AS counterfactuals
                """,
                {"episode_id": episode_id},
            )
        except Exception as exc:
            self._log.warning(
                "neo4j_episode_fetch_failed",
                episode_id=episode_id,
                error=str(exc),
            )
            return None

        if not results:
            return None

        row = results[0]
        episode_data = row["e"]
        predecessors = row.get("predecessors", [])
        counterfactuals = row.get("counterfactuals", [])

        dag = self._reconstruct_cognitive_chain(
            episode_data, predecessors, counterfactuals,
        )

        self._log.debug(
            "dag_built",
            episode_id=episode_id,
            nodes=len(dag.nodes),
            edges=len(dag.edges),
        )
        return dag

    async def build_dags_for_pattern(
        self,
        failure_pattern: FailurePattern,
        max_episodes: int = 50,
    ) -> list[CognitiveCausalDAG]:
        """Build cognitive DAGs for all matching episodes of a failure pattern."""
        episode_ids = failure_pattern.matching_episode_ids[:max_episodes]
        dags: list[CognitiveCausalDAG] = []

        # Batch fetch to reduce Neo4j round-trips
        for eid in episode_ids:
            dag = await self.build_dag_for_episode(eid)
            if dag is not None:
                dags.append(dag)

        self._log.info(
            "dags_built_for_pattern",
            pattern_id=failure_pattern.pattern_id,
            requested=len(episode_ids),
            built=len(dags),
        )
        return dags

    def _reconstruct_cognitive_chain(
        self,
        episode: dict[str, Any],
        predecessors: list[dict[str, Any]],
        counterfactuals: list[dict[str, Any]],
    ) -> CognitiveCausalDAG:
        """
        Reconstruct the 8-stage cognitive chain from stored Neo4j properties.

        Maps episode properties to variable states at each stage:
          - salience_composite, salience_scores_json → PERCEPT
          - free_energy, affect_valence, affect_arousal → BELIEF_UPDATE
          - goal info (inferred from source) → GOAL_SELECTION
          - Counterfactual efe_total, policy_type → POLICY_SELECTION
          - regret → OUTCOME
        """
        episode_id = episode.get("id", "")
        nodes: list[CognitiveNode] = []
        edges: list[CognitiveEdge] = []

        # Collect outcome info
        # For counterfactual-linked episodes, regret is on the counterfactual
        regret = 0.0
        policy_type = ""
        for cf in counterfactuals:
            if cf.get("resolved"):
                regret = cf.get("regret", 0.0) or 0.0
                policy_type = cf.get("policy_type", "") or ""
                break

        # Build nodes for each cognitive stage
        prev_node_id: str | None = None

        for stage in CognitiveStage:
            nid = f"csn_{new_id()[:8]}"
            variables = self._infer_variables_at_stage(
                stage, episode, counterfactuals,
            )
            node = CognitiveNode(
                node_id=nid,
                stage=stage,
                episode_id=episode_id,
                system=_STAGE_SYSTEM[stage],
                variables=variables,
            )
            nodes.append(node)

            # Create edge from previous stage
            if prev_node_id is not None:
                # Compute causal influence from variable deltas
                prev_vars = nodes[-2].variables
                curr_vars = variables
                deltas = {
                    k: curr_vars.get(k, 0.0) - prev_vars.get(k, 0.0)
                    for k in set(prev_vars) | set(curr_vars)
                    if abs(curr_vars.get(k, 0.0) - prev_vars.get(k, 0.0)) > 1e-6
                }
                # Causal influence: higher if more variables changed significantly
                influence = min(1.0, 0.5 + 0.1 * len(deltas))

                # Use FOLLOWED_BY causal_strength if available
                for pred in predecessors:
                    cs = pred.get("causal_strength")
                    if cs is not None and cs > 0:
                        influence = max(influence, cs)
                        break

                edges.append(CognitiveEdge(
                    from_node=prev_node_id,
                    to_node=nid,
                    causal_influence=influence,
                    mechanism=f"{nodes[-2].stage.value}_to_{stage.value}",
                    variables_changed=deltas,
                ))

            prev_node_id = nid

        fe_start = float(episode.get("free_energy", 0.0) or 0.0)
        # Outcome free energy: assume reduction if outcome was successful
        outcome_success = regret <= 0.0

        return CognitiveCausalDAG(
            episode_id=episode_id,
            nodes=nodes,
            edges=edges,
            outcome_success=outcome_success,
            outcome_value=-regret,  # positive = good outcome
            free_energy_start=fe_start,
            free_energy_end=fe_start * (0.7 if outcome_success else 1.3),
            policy_type=policy_type,
            regret=regret,
        )

    def _infer_variables_at_stage(
        self,
        stage: CognitiveStage,
        episode: dict[str, Any],
        counterfactuals: list[dict[str, Any]],
    ) -> dict[str, float]:
        """
        Extract system variable values at a cognitive stage from episode properties.

        Rule-based (zero LLM tokens). Maps episode properties to parameter
        names defined in TUNABLE_PARAMETERS.
        """
        variables: dict[str, float] = {}

        if stage == CognitiveStage.PERCEPT:
            # Parse salience scores from JSON if available
            salience_json = episode.get("salience_scores_json", "")
            if salience_json:
                try:
                    scores = json.loads(salience_json)
                    if isinstance(scores, dict):
                        for head_name, score in scores.items():
                            param = f"atune.head.{head_name}.weight"
                            if param in TUNABLE_PARAMETERS:
                                variables[param] = float(score)
                except (json.JSONDecodeError, TypeError):
                    pass
            # Composite salience as a proxy
            salience = episode.get("salience_composite", 0.0)
            if salience is not None:
                variables["salience_composite"] = float(salience)

        elif stage == CognitiveStage.BELIEF_UPDATE:
            fe = episode.get("free_energy", 0.0)
            if fe is not None:
                variables["free_energy"] = float(fe)
            valence = episode.get("affect_valence", 0.0)
            if valence is not None:
                variables["affect_valence"] = float(valence)
            arousal = episode.get("affect_arousal", 0.0)
            if arousal is not None:
                variables["affect_arousal"] = float(arousal)
            # Map to tunable parameters at their default values
            for param in _STAGE_PARAMETERS[stage]:
                if param in PARAMETER_DEFAULTS:
                    variables[param] = PARAMETER_DEFAULTS[param]

        elif stage == CognitiveStage.POLICY_SELECTION:
            # Extract EFE components from counterfactual records
            for cf in counterfactuals:
                efe = cf.get("efe_total")
                if efe is not None:
                    variables["efe_total"] = float(efe)
                prag = cf.get("estimated_pragmatic_value")
                if prag is not None:
                    variables["estimated_pragmatic_value"] = float(prag)
                epist = cf.get("estimated_epistemic_value")
                if epist is not None:
                    variables["estimated_epistemic_value"] = float(epist)
                const = cf.get("constitutional_alignment")
                if const is not None:
                    variables["constitutional_alignment"] = float(const)
                feas = cf.get("feasibility")
                if feas is not None:
                    variables["feasibility"] = float(feas)
                risk_harm = cf.get("risk_expected_harm")
                if risk_harm is not None:
                    variables["risk_expected_harm"] = float(risk_harm)
                break  # Use first counterfactual

            # Map to tunable EFE weights
            for param in _STAGE_PARAMETERS[stage]:
                if param in PARAMETER_DEFAULTS:
                    variables[param] = PARAMETER_DEFAULTS[param]

        elif stage == CognitiveStage.POLICY_GENERATION:
            for param in _STAGE_PARAMETERS[stage]:
                if param in PARAMETER_DEFAULTS:
                    variables[param] = PARAMETER_DEFAULTS[param]

        elif stage == CognitiveStage.OUTCOME:
            for cf in counterfactuals:
                regret = cf.get("regret")
                if regret is not None:
                    variables["regret"] = float(regret)
                break

        return variables


# ═══════════════════════════════════════════════════════════════════════════
# CausalFailureAnalyzer
# ═══════════════════════════════════════════════════════════════════════════


class CausalFailureAnalyzer:
    """
    Identifies causal intervention points from episode failure patterns
    using do-calculus on the cognitive causal DAG.

    Integration points:
      - Evo Phase 6.5: detect_failure_patterns() from RegretStats
      - Evo Phase 8: analyze_failure_pattern() produces CausalSurgeryResult
      - Bridge: CausalInterventionPoint → surgical EvolutionProposal
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        llm: LLMProvider,
        dag_builder: CausalDAGBuilder | None = None,
        *,
        max_counterfactuals_per_episode: int = 5,
        min_episodes_for_confidence: int = _MIN_EPISODES_FOR_CONFIDENCE,
        llm_budget_tokens: int = 2000,
    ) -> None:
        self._neo4j = neo4j
        self._llm = llm
        self._dag_builder = dag_builder or CausalDAGBuilder(neo4j)
        self._max_cf = max_counterfactuals_per_episode
        self._min_episodes = min_episodes_for_confidence
        self._llm_budget = llm_budget_tokens
        self._log = logger.bind(system="simula.causal_surgery.analyzer")

    # ── Public API ─────────────────────────────────────────────────────────

    async def detect_failure_patterns(
        self,
        regret_stats: RegretStats,
        min_regret: float = 0.3,
        max_patterns: int = 5,
    ) -> list[FailurePattern]:
        """
        Query Neo4j for recurring high-regret episodes, cluster by
        policy_type × goal_domain to form FailurePatterns.

        Rule-based (zero LLM tokens).
        """
        if regret_stats.high_regret_count < _MIN_EPISODES_FOR_PATTERN:
            return []

        try:
            results = await self._neo4j.execute_read(
                """
                MATCH (cf:Counterfactual)
                WHERE cf.resolved = true
                  AND abs(cf.regret) > $min_regret
                WITH cf.policy_type AS ptype,
                     CASE
                       WHEN cf.goal_id IS NOT NULL THEN cf.goal_id
                       ELSE 'unknown'
                     END AS domain,
                     collect(cf.id) AS episode_ids,
                     count(*) AS occurrence,
                     avg(cf.regret) AS mean_regret
                WHERE occurrence >= $min_occurrences
                RETURN ptype, domain, episode_ids, occurrence, mean_regret
                ORDER BY mean_regret DESC
                LIMIT $max_patterns
                """,
                {
                    "min_regret": min_regret,
                    "min_occurrences": _MIN_EPISODES_FOR_PATTERN,
                    "max_patterns": max_patterns,
                },
            )
        except Exception as exc:
            self._log.warning("failure_pattern_query_failed", error=str(exc))
            return []

        patterns: list[FailurePattern] = []
        for row in results:
            ptype = row.get("ptype", "") or ""
            domain = row.get("domain", "") or ""
            episode_ids = row.get("episode_ids", []) or []
            occurrence = row.get("occurrence", 0) or 0
            mean_regret_val = row.get("mean_regret", 0.0) or 0.0

            description = (
                f"policy_type={ptype!r} AND goal_domain={domain!r} "
                f"=> failure (mean_regret={mean_regret_val:.2f}, "
                f"occurrences={occurrence})"
            )

            patterns.append(FailurePattern(
                description=description,
                condition_predicates=[
                    f"policy_type == {ptype!r}",
                    f"goal_domain == {domain!r}",
                ],
                matching_episode_ids=episode_ids,
                policy_type=ptype,
                goal_domain=domain,
                occurrence_count=occurrence,
                mean_regret=mean_regret_val,
            ))

        self._log.info(
            "failure_patterns_detected",
            count=len(patterns),
            top_pattern=patterns[0].description[:80] if patterns else "",
        )
        return patterns

    async def analyze_failure_pattern(
        self,
        failure_pattern: FailurePattern,
    ) -> CausalSurgeryResult:
        """
        Full causal surgery pipeline:
          1. Build cognitive DAGs for matching episodes
          2. Compute counterfactuals at each eligible stage (two-tier)
          3. Aggregate to find critical intervention point(s)
          4. Estimate normal-condition cost
          5. Persist results to Neo4j
        """
        start = time.monotonic()
        total_tokens = 0

        # Phase 1: Build DAGs
        dags = await self._dag_builder.build_dags_for_pattern(failure_pattern)
        if len(dags) < self._min_episodes:
            self._log.info(
                "insufficient_dags_for_analysis",
                pattern_id=failure_pattern.pattern_id,
                dags=len(dags),
                required=self._min_episodes,
            )
            return CausalSurgeryResult(
                failure_pattern=failure_pattern,
                dags_built=len(dags),
                total_duration_ms=int((time.monotonic() - start) * 1000),
            )

        # Phase 2: Compute counterfactuals
        all_counterfactuals: list[CounterfactualScenario] = []
        for dag in dags:
            scenarios = self._compute_rule_based_counterfactuals(dag)
            all_counterfactuals.extend(scenarios)

        # Tier 2: LLM-based counterfactuals for conditional interventions
        if total_tokens < self._llm_budget and len(dags) >= 3:
            llm_scenarios, tokens = await self._llm_counterfactual_batch(
                dags, failure_pattern,
            )
            all_counterfactuals.extend(llm_scenarios)
            total_tokens += tokens

        # Phase 3: Aggregate into intervention points
        intervention_points = self._aggregate_intervention_points(
            all_counterfactuals, dags,
        )

        # Phase 4: Estimate normal-condition cost for top interventions
        for ip in intervention_points[:3]:
            ip.estimated_normal_cost = self._estimate_normal_cost(ip)

        # Select best intervention
        best = intervention_points[0] if intervention_points else None

        elapsed_ms = int((time.monotonic() - start) * 1000)

        result = CausalSurgeryResult(
            failure_pattern=failure_pattern,
            dags_built=len(dags),
            counterfactuals_evaluated=len(all_counterfactuals),
            intervention_points=intervention_points,
            best_intervention=best,
            total_duration_ms=elapsed_ms,
            llm_tokens_used=total_tokens,
        )

        # Phase 5: Persist to Neo4j
        await self._persist_surgery_result(result)

        if best is not None:
            self._log.info(
                "causal_intervention_identified",
                system=best.system,
                parameter=best.parameter,
                direction=best.direction.value,
                success_rate=round(best.intervention_success_rate, 2),
                confidence=round(best.confidence, 2),
                mean_ate=round(best.mean_ate, 3),
                episodes_analyzed=best.episodes_analyzed,
                condition=best.condition or "unconditional",
                duration_ms=elapsed_ms,
                llm_tokens=total_tokens,
            )
        else:
            self._log.info(
                "no_intervention_found",
                pattern_id=failure_pattern.pattern_id,
                dags_built=len(dags),
                counterfactuals=len(all_counterfactuals),
                duration_ms=elapsed_ms,
            )

        return result

    # ── Tier 1: Rule-Based Counterfactuals ─────────────────────────────────

    def _compute_rule_based_counterfactuals(
        self, dag: CognitiveCausalDAG,
    ) -> list[CounterfactualScenario]:
        """
        For each tunable parameter at each intervention-eligible stage,
        compute the ATE by propagating a perturbation through the DAG.

        Uses linear SCM approximation:
          E[Y | do(X = x + delta)] ≈ E[Y|X=x] + delta * sum(path_coefficients)
        """
        scenarios: list[CounterfactualScenario] = []

        for node in dag.nodes:
            if node.stage not in _INTERVENTION_ELIGIBLE:
                continue

            relevant_params = _STAGE_PARAMETERS.get(node.stage, [])
            for param in relevant_params:
                if param not in TUNABLE_PARAMETERS:
                    continue

                spec = TUNABLE_PARAMETERS[param]
                current_val = node.variables.get(
                    param, PARAMETER_DEFAULTS.get(param, 0.0),
                )

                # Try perturbation in the direction that might help
                # (if outcome was bad, try increasing risk weight, etc.)
                for delta_sign in (1.0, -1.0):
                    delta = delta_sign * spec.step
                    new_val = current_val + delta

                    # Clamp to valid range
                    if new_val < spec.min_val or new_val > spec.max_val:
                        continue

                    ate = self._propagate_perturbation(dag, node, param, delta)

                    # For a failed episode (negative outcome_value), a positive
                    # ATE means the intervention would have improved the outcome
                    outcome_flipped = (
                        dag.outcome_value < 0 and ate > abs(dag.outcome_value) * 0.5
                    )

                    direction = (
                        InterventionDirection.INCREASE
                        if delta > 0
                        else InterventionDirection.DECREASE
                    )

                    scenarios.append(CounterfactualScenario(
                        episode_id=dag.episode_id,
                        intervention_node_id=node.node_id,
                        intervention_stage=node.stage,
                        parameter=param,
                        original_variables={param: current_val},
                        counterfactual_variables={param: new_val},
                        original_outcome_value=dag.outcome_value,
                        counterfactual_outcome_value=dag.outcome_value + ate,
                        ate=ate,
                        outcome_flipped=outcome_flipped,
                        confidence=0.6 if outcome_flipped else 0.4,
                        reasoning=(
                            f"do({param} = {new_val:.3f}) [{direction.value} by "
                            f"{abs(delta):.3f}] -> ATE={ate:.4f}"
                        ),
                    ))

        return scenarios[:self._max_cf * len(_INTERVENTION_ELIGIBLE)]

    def _propagate_perturbation(
        self,
        dag: CognitiveCausalDAG,
        intervention_node: CognitiveNode,
        parameter: str,
        delta: float,
    ) -> float:
        """
        Propagate a variable perturbation through the DAG using a linear
        SCM approximation.

        ATE = delta * sum over all directed paths from intervention to outcome:
            product of causal_influence along path

        This implements Pearl's do-calculus adjustment formula for linear SCMs.
        """
        # Find all directed paths from intervention_node to outcome node
        node_index = {n.node_id: i for i, n in enumerate(dag.nodes)}
        adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for edge in dag.edges:
            adj[edge.from_node].append((edge.to_node, edge.causal_influence))

        # DFS to find all paths from intervention node to outcome
        outcome_nodes = [
            n.node_id for n in dag.nodes
            if n.stage == CognitiveStage.OUTCOME
        ]
        if not outcome_nodes:
            return 0.0

        total_path_coefficient = 0.0

        def dfs(current: str, target: str, path_coeff: float) -> None:
            nonlocal total_path_coefficient
            if current == target:
                total_path_coefficient += path_coeff
                return
            for neighbor, weight in adj.get(current, []):
                if node_index.get(neighbor, 0) > node_index.get(current, 0):
                    dfs(neighbor, target, path_coeff * weight)

        for outcome_nid in outcome_nodes:
            dfs(intervention_node.node_id, outcome_nid, 1.0)

        return delta * total_path_coefficient

    # ── Tier 2: LLM-Based Counterfactuals ──────────────────────────────────

    async def _llm_counterfactual_batch(
        self,
        dags: list[CognitiveCausalDAG],
        failure_pattern: FailurePattern,
    ) -> tuple[list[CounterfactualScenario], int]:
        """
        Batched LLM call for complex conditional counterfactuals.

        Asks: "If the system had this conditional constraint, would the
        failure have been avoided?"

        ~800 tokens per batch of 10 episodes.
        """
        # Build a conditional intervention description from the pattern
        intervention_desc = (
            f"When {' AND '.join(failure_pattern.condition_predicates)}, "
            f"apply extra caution: increase nova.efe.risk weight by 0.05 "
            f"and decrease nova.efe.pragmatic weight by 0.05"
        )

        # Build episode list
        episode_summaries: list[str] = []
        dag_map: dict[int, CognitiveCausalDAG] = {}
        for i, dag in enumerate(dags[:10], start=1):
            summary = (
                f"policy_type={dag.policy_type!r}, "
                f"regret={dag.regret:.2f}, "
                f"outcome_value={dag.outcome_value:.2f}"
            )
            episode_summaries.append(f"{i}. [{dag.episode_id[:12]}] {summary}")
            dag_map[i] = dag

        prompt = _COUNTERFACTUAL_PROMPT.format(
            pattern_description=failure_pattern.description,
            count=len(episode_summaries),
            intervention=intervention_desc,
            episode_list="\n".join(episode_summaries),
        )

        try:
            response = await asyncio.wait_for(
                self._llm.evaluate(
                    prompt=prompt, max_tokens=500, temperature=0.2,
                ),
                timeout=15.0,
            )
        except Exception as exc:
            self._log.warning("llm_counterfactual_failed", error=str(exc))
            return [], 0

        tokens = getattr(response, "input_tokens", 0) + getattr(
            response, "output_tokens", 0,
        )

        # Parse response
        scenarios: list[CounterfactualScenario] = []
        for match in re.finditer(
            r"(\d+)\.\s*(yes|no)\s*\|\s*([\d.]+)\s*\|\s*(.+)",
            response.text,
            re.IGNORECASE,
        ):
            num = int(match.group(1))
            answered_yes = match.group(2).lower() == "yes"
            confidence = min(1.0, max(0.0, float(match.group(3))))
            reason = match.group(4).strip()

            if not answered_yes or num not in dag_map:
                continue

            dag = dag_map[num]
            scenarios.append(CounterfactualScenario(
                episode_id=dag.episode_id,
                intervention_node_id="conditional",
                intervention_stage=CognitiveStage.POLICY_SELECTION,
                parameter="nova.efe.risk+nova.efe.pragmatic",
                original_outcome_value=dag.outcome_value,
                counterfactual_outcome_value=abs(dag.outcome_value),
                ate=abs(dag.outcome_value) - dag.outcome_value,
                outcome_flipped=True,
                confidence=confidence,
                reasoning=reason,
            ))

        self._log.info(
            "llm_counterfactual_batch",
            episodes=len(episode_summaries),
            flipped=len(scenarios),
            tokens=tokens,
        )
        return scenarios, tokens

    # ── Aggregation ────────────────────────────────────────────────────────

    def _aggregate_intervention_points(
        self,
        all_counterfactuals: list[CounterfactualScenario],
        dags: list[CognitiveCausalDAG],
    ) -> list[CausalInterventionPoint]:
        """
        Group counterfactuals by (system, parameter), compute success rate,
        mean ATE, and confidence. Returns sorted by score descending.
        """
        # Group by parameter
        groups: dict[str, list[CounterfactualScenario]] = defaultdict(list)
        for scenario in all_counterfactuals:
            if scenario.outcome_flipped and scenario.ate > 0:
                groups[scenario.parameter].append(scenario)

        total_episodes = len(dags)
        interventions: list[CausalInterventionPoint] = []

        for param, scenarios in groups.items():
            if len(scenarios) < self._min_episodes:
                continue

            # Determine system from parameter name
            system = param.split(".")[0] if "." in param else "unknown"

            # Determine direction from majority of scenarios
            positive_deltas = sum(
                1 for s in scenarios
                if any(v > 0 for v in s.counterfactual_variables.values())
            )
            direction = (
                InterventionDirection.INCREASE
                if positive_deltas > len(scenarios) / 2
                else InterventionDirection.DECREASE
            )

            episodes_helped = len(set(s.episode_id for s in scenarios))
            success_rate = episodes_helped / max(1, total_episodes)
            mean_ate = sum(s.ate for s in scenarios) / max(1, len(scenarios))
            mean_confidence = sum(s.confidence for s in scenarios) / max(
                1, len(scenarios),
            )

            # Wilson score lower bound for confidence interval
            n = total_episodes
            p = success_rate
            wilson_lower = 0.0
            if n > 0:
                denominator = 1 + _Z_SCORE_95**2 / n
                center = p + _Z_SCORE_95**2 / (2 * n)
                spread = _Z_SCORE_95 * math.sqrt(
                    (p * (1 - p) + _Z_SCORE_95**2 / (4 * n)) / n
                )
                wilson_lower = max(0.0, (center - spread) / denominator)

            # Suggested value
            current_val = PARAMETER_DEFAULTS.get(param)
            suggested_val: float | None = None
            if current_val is not None and param in TUNABLE_PARAMETERS:
                step = TUNABLE_PARAMETERS[param].step
                if direction == InterventionDirection.INCREASE:
                    suggested_val = current_val + step
                else:
                    suggested_val = current_val - step

            # Check if this is a conditional (LLM-derived) intervention
            is_conditional = any(
                s.intervention_node_id == "conditional" for s in scenarios
            )
            condition = ""
            if is_conditional:
                condition = (
                    f"when policy_type matches failure pattern "
                    f"(regret > {abs(dags[0].regret):.2f})"
                )

            interventions.append(CausalInterventionPoint(
                system=system,
                parameter=param,
                direction=direction,
                suggested_value=suggested_val,
                current_value=current_val,
                episodes_analyzed=total_episodes,
                episodes_where_intervention_helps=episodes_helped,
                intervention_success_rate=success_rate,
                mean_ate=mean_ate,
                confidence=wilson_lower * mean_confidence,
                failure_pattern_id="",  # Set by caller
                counterfactual_scenarios=scenarios[:10],  # Keep top 10
                reasoning=(
                    f"Changing {param} ({direction.value}) would have avoided "
                    f"failure in {episodes_helped}/{total_episodes} episodes "
                    f"(mean ATE={mean_ate:.3f})"
                ),
                condition=condition,
                is_conditional=is_conditional,
            ))

        # Sort by composite score: success_rate * confidence
        interventions.sort(
            key=lambda ip: ip.intervention_success_rate * ip.confidence,
            reverse=True,
        )

        # Filter low-quality interventions
        return [
            ip for ip in interventions
            if ip.intervention_success_rate >= _MIN_SUCCESS_RATE
        ]

    # ── Normal-Condition Cost ──────────────────────────────────────────────

    def _estimate_normal_cost(
        self,
        intervention: CausalInterventionPoint,
    ) -> float:
        """
        Estimate the cost of the intervention in normal (non-failure) conditions.

        Uses the parameter's step size and the EFE weight budget to estimate
        how much pragmatic value would be lost in normal operation.

        Returns a value in [0, 1] where 0 = no cost, 1 = maximum cost.
        """
        param = intervention.parameter
        if param not in TUNABLE_PARAMETERS:
            return 0.0

        spec = TUNABLE_PARAMETERS[param]
        current = intervention.current_value or PARAMETER_DEFAULTS.get(param, 0.0)

        # Cost is proportional to how much we're moving the parameter
        # relative to its valid range
        param_range = spec.max_val - spec.min_val
        if param_range <= 0:
            return 0.0

        delta = spec.step  # One step in the intervention direction
        cost = delta / param_range

        # Higher cost if we're moving away from the default
        default = PARAMETER_DEFAULTS.get(param, current)
        if intervention.direction == InterventionDirection.INCREASE:
            new_val = current + delta
        else:
            new_val = current - delta

        distance_from_default = abs(new_val - default) / param_range
        cost = cost * (1.0 + distance_from_default)

        return min(1.0, cost)

    # ── Neo4j Persistence ──────────────────────────────────────────────────

    async def _persist_surgery_result(
        self,
        result: CausalSurgeryResult,
    ) -> None:
        """Store the causal surgery result in Neo4j for audit and learning."""
        best = result.best_intervention
        try:
            await self._neo4j.execute_write(
                """
                CREATE (cs:EpisodeChain:CausalSurgery {
                    id: $id,
                    pattern_id: $pattern_id,
                    pattern_description: $description,
                    dags_built: $dags_built,
                    counterfactuals_evaluated: $counterfactuals,
                    best_system: $best_system,
                    best_parameter: $best_parameter,
                    best_direction: $best_direction,
                    intervention_success_rate: $success_rate,
                    mean_ate: $mean_ate,
                    confidence: $confidence,
                    analyzed_at: datetime()
                })
                WITH cs
                UNWIND $episode_ids AS eid
                MATCH (e:Episode {id: eid})
                CREATE (cs)-[:ANALYZED {role: 'failure_episode'}]->(e)
                """,
                {
                    "id": result.id,
                    "pattern_id": result.failure_pattern.pattern_id,
                    "description": result.failure_pattern.description[:500],
                    "dags_built": result.dags_built,
                    "counterfactuals": result.counterfactuals_evaluated,
                    "best_system": best.system if best else "",
                    "best_parameter": best.parameter if best else "",
                    "best_direction": best.direction.value if best else "",
                    "success_rate": best.intervention_success_rate if best else 0.0,
                    "mean_ate": best.mean_ate if best else 0.0,
                    "confidence": best.confidence if best else 0.0,
                    "episode_ids": result.failure_pattern.matching_episode_ids[:50],
                },
            )
        except Exception as exc:
            self._log.warning(
                "persist_surgery_result_failed",
                result_id=result.id,
                error=str(exc),
            )
