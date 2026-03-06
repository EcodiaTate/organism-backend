"""
EcodiaOS — Inspector Phase 5: Trust Analyzer (Orchestrator)

Single-entry orchestrator for the Phase 5 trust-graph influence expansion pipeline:

  TrustGraphBuilder  → (TrustGraph, list[FootholdBinding])
  PropagationEngine  → ReachabilityMap (per-foothold simulations + ranked corridors)
  → Phase5Result

Usage
-----
  # From Phase 4 output (recommended):
  analyzer = TrustAnalyzer()
  result = analyzer.analyze(
      phase3_result=phase3_result,
      phase4_result=phase4_result,
  )

  # From Phase 3 output only:
  result = analyzer.analyze(
      phase3_result=phase3_result,
  )

Exit criterion
--------------
Phase5Result.exit_criterion_met = True when:
  - ≥1 foothold node has been identified AND
  - ≥1 propagation path has been produced AND
  - ≥1 ExpansionCorridor has a non-empty description explaining the trust edges
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from systems.simula.inspector.propagation_engine import PropagationEngine
from systems.simula.inspector.trust_graph import TrustGraphBuilder
from systems.simula.inspector.trust_types import (
    ExpansionCorridor,
    FootholdBinding,
    Phase5Result,
    ReachabilityMap,
)

if TYPE_CHECKING:
    from systems.simula.inspector.constraint_types import Phase4Result
    from systems.simula.inspector.static_types import Phase3Result

logger = structlog.get_logger().bind(system="simula.inspector.trust_analyzer")


class TrustAnalyzer:
    """
    Phase 5 orchestrator — builds a Phase5Result for a target.

    The result combines:
    - A TrustGraph (principal, service, resource, role, credential, session nodes
      + typed trust edges)
    - FootholdBindings mapping Phase 4 steerable regions to trust-graph entry points
    - A ReachabilityMap with per-foothold PropagationSimulations
    - Ranked ExpansionCorridors describing propagation pathways
    - Overall exit criterion flag

    Parameters
    ----------
    max_depth       — BFS hop limit per simulation (default 6)
    strict_gradient — only report monotonically increasing privilege paths (default True)
    high_threshold  — minimum privilege_value for a node to be a corridor terminal (default 50)
    """

    def __init__(
        self,
        max_depth: int = 6,
        strict_gradient: bool = True,
        high_threshold: int = 50,
    ) -> None:
        self._builder = TrustGraphBuilder()
        self._engine  = PropagationEngine(
            max_depth=max_depth,
            strict_gradient=strict_gradient,
            high_threshold=high_threshold,
        )
        self._log = logger

    # ── Primary entry point ───────────────────────────────────────────────────

    def analyze(
        self,
        phase3_result: Phase3Result,
        phase4_result: Phase4Result | None = None,
    ) -> Phase5Result:
        """
        Build a Phase5Result from Phase 3 (and optionally Phase 4) data.

        Args:
            phase3_result: Complete Phase 3 static analysis output.
            phase4_result: Optional Phase 4 steerability model output.
                           When provided, enables Phase 4→5 foothold binding and
                           richer trust edge inference.

        Returns:
            Phase5Result with TrustGraph, ReachabilityMap, and ranked corridors.
        """
        target_id = phase3_result.target_id
        log = self._log.bind(target_id=target_id)
        log.info("trust_analysis_started")

        # 1. Build trust graph + foothold bindings
        graph, bindings = self._builder.build(
            target_id=target_id,
            phase3_result=phase3_result,
            phase4_result=phase4_result,
        )

        # 2. Run propagation simulations
        rmap = self._engine.simulate_all(
            target_id=target_id,
            graph=graph,
            bindings=bindings,
        )

        # 3. Deduplicate + rank corridors across all simulations
        ranked_corridors = rmap.ranked_corridors  # already ranked by simulate_all

        # 4. Check exit criterion
        exit_met = self._check_exit_criterion(bindings, rmap, ranked_corridors)

        result = Phase5Result(
            target_id=target_id,
            trust_graph=graph,
            reachability_map=rmap,
            foothold_bindings=bindings,
            ranked_corridors=ranked_corridors,
            total_trust_nodes=graph.total_nodes,
            total_trust_edges=graph.total_edges,
            total_foothold_bindings=len(bindings),
            total_simulations=len(rmap.simulations),
            total_corridors=rmap.total_corridors,
            critical_corridors=rmap.critical_corridors,
            high_corridors=rmap.high_corridors,
            exit_criterion_met=exit_met,
        )

        log.info(
            "trust_analysis_complete",
            nodes=graph.total_nodes,
            edges=graph.total_edges,
            footholds=len(bindings),
            simulations=len(rmap.simulations),
            corridors=rmap.total_corridors,
            critical=rmap.critical_corridors,
            exit_criterion_met=exit_met,
        )

        return result

    # ── Targeted query ────────────────────────────────────────────────────────

    def explain_corridor(
        self,
        result: Phase5Result,
        corridor_id: str,
    ) -> dict:
        """
        Return a structured explanation dict for a specific corridor.

        This is the exit-criterion delivery: given a corridor_id, produce a
        defensible explanation of the propagation pathway and the trust edges
        enabling it.

        Returns a dict with:
          corridor_id, risk_tier, foothold, terminal, hops,
          trust_edges (list of {kind, strength, from, to, description}),
          privilege_delta, weakest_link, description, confidence
        """
        corridor = next(
            (c for c in result.ranked_corridors if c.corridor_id == corridor_id),
            None,
        )
        if not corridor:
            return {"error": f"corridor '{corridor_id}' not found"}

        graph = result.trust_graph
        path  = corridor.primary_path

        trust_edges = []
        for step in path.steps:
            edge = graph.edges.get(step.edge_id, None)
            from_node = graph.nodes.get(
                path.node_ids[step.step_index] if step.step_index < len(path.node_ids) else "",
                None,
            )
            graph.nodes.get(step.arrived_node_id, None)
            trust_edges.append({
                "edge_id":      step.edge_id,
                "kind":         step.edge_kind.value,
                "strength":     step.edge_strength.value,
                "traversal_cost": step.edge_traversal_cost,
                "from":         from_node.name if from_node else "unknown",
                "from_kind":    from_node.kind.value if from_node else "unknown",
                "to":           step.arrived_node_name,
                "to_kind":      step.arrived_node_kind.value,
                "to_privilege": step.arrived_privilege_value,
                "description":  edge.description if edge else "",
            })

        foothold_node = graph.nodes.get(corridor.foothold_node_id)
        terminal_node = graph.nodes.get(corridor.terminal_node_id)

        return {
            "corridor_id":        corridor.corridor_id,
            "risk_tier":          corridor.risk_tier.value,
            "foothold": {
                "node_id":   corridor.foothold_node_id,
                "name":      corridor.foothold_node_name,
                "kind":      foothold_node.kind.value if foothold_node else "unknown",
                "privilege": foothold_node.privilege_value if foothold_node else 0,
            },
            "terminal": {
                "node_id":   corridor.terminal_node_id,
                "name":      corridor.terminal_node_name,
                "kind":      terminal_node.kind.value if terminal_node else "unknown",
                "privilege": terminal_node.privilege_value if terminal_node else 0,
                "impact":    corridor.terminal_privilege_impact.value,
            },
            "hops":              corridor.min_hops,
            "privilege_delta":   corridor.max_privilege_delta,
            "weakest_link": {
                "edge_id": corridor.weakest_edge_id,
                "cost":    corridor.weakest_edge_cost,
            },
            "trust_edges":       trust_edges,
            "exploited_edge_kinds": [k.value for k in corridor.exploited_edge_kinds],
            "description":       corridor.description,
            "alternative_path_count": len(corridor.alternative_paths),
            "phase4_condition_set_ids": corridor.evidence_condition_set_ids,
        }

    # ── Reporting helper ──────────────────────────────────────────────────────

    def model_summary(self, result: Phase5Result) -> dict:
        """
        Return a concise reporting dict suitable for logging or display.
        """
        graph = result.trust_graph

        # Node breakdown by kind
        node_breakdown: dict[str, int] = {}
        for node in graph.nodes.values():
            node_breakdown[node.kind.value] = node_breakdown.get(node.kind.value, 0) + 1

        # Edge breakdown by kind
        edge_breakdown: dict[str, int] = {}
        for edge in graph.edges.values():
            edge_breakdown[edge.kind.value] = edge_breakdown.get(edge.kind.value, 0) + 1

        # Top 5 corridors
        top_corridors = [
            {
                "corridor_id":        c.corridor_id,
                "risk_tier":          c.risk_tier.value,
                "foothold":           c.foothold_node_name,
                "terminal":           c.terminal_node_name,
                "terminal_impact":    c.terminal_privilege_impact.value,
                "hops":               c.min_hops,
                "privilege_delta":    c.max_privilege_delta,
                "weakest_edge_cost":  c.weakest_edge_cost,
                "edge_kinds":         [k.value for k in c.exploited_edge_kinds],
            }
            for c in result.ranked_corridors[:5]
        ]

        # Footholds
        foothold_summaries = [
            {
                "node_id":              b.foothold_node_id,
                "name":                 b.foothold_node_name,
                "kind":                 b.foothold_node_kind.value,
                "steerability_class":   b.steerability_class,
                "condition_set_id":     b.condition_set_id,
                "confidence":           b.confidence,
            }
            for b in result.foothold_bindings
        ]

        return {
            "target_id":             result.target_id,
            "exit_criterion_met":    result.exit_criterion_met,
            "trust_nodes":           result.total_trust_nodes,
            "trust_edges":           result.total_trust_edges,
            "node_breakdown":        node_breakdown,
            "edge_breakdown":        edge_breakdown,
            "footholds":             result.total_foothold_bindings,
            "simulations":           result.total_simulations,
            "total_corridors":       result.total_corridors,
            "critical_corridors":    result.critical_corridors,
            "high_corridors":        result.high_corridors,
            "global_max_privilege":  result.reachability_map.global_max_privilege,
            "top_corridors":         top_corridors,
            "foothold_bindings":     foothold_summaries,
        }

    # ── Exit criterion ────────────────────────────────────────────────────────

    @staticmethod
    def _check_exit_criterion(
        bindings: list[FootholdBinding],
        rmap: ReachabilityMap,
        corridors: list[ExpansionCorridor],
    ) -> bool:
        """
        Phase 5 exit criterion:
        - ≥1 foothold identified (bindings or steerability-adjacent nodes)
        - ≥1 propagation path produced in any simulation
        - ≥1 ExpansionCorridor with a non-empty description
        """
        has_foothold = bool(bindings) or any(
            s.total_reachable_nodes > 0 for s in rmap.simulations.values()
        )
        has_paths = any(s.total_paths > 0 for s in rmap.simulations.values())
        has_corridor = any(bool(c.description) for c in corridors)

        return has_foothold and has_paths and has_corridor
