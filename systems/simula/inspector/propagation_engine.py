"""
EcodiaOS - Inspector Phase 5: Propagation Engine

Runs influence propagation simulations across the TrustGraph and extracts
ranked ExpansionCorridors.

Algorithm
---------
For each foothold node F:

  1. BFS reachability (depth ≤ max_depth)
     Visit all nodes reachable from F via any trust edge.
     Record the path to each node (shortest-path parent pointers).

  2. Privilege gradient filtering
     Build all simple paths from F to nodes with privilege_value > F.privilege_value.
     If strict_gradient=True, only keep paths that are monotonically non-decreasing
     in privilege_value at every hop.

  3. Corridor extraction
     For each terminal node T (high-value, privilege_value ≥ HIGH_THRESHOLD):
       - Take the shortest monotonic path F→T as the primary path.
       - Collect alternative paths (up to MAX_ALTERNATIVES).
       - Classify risk tier from terminal PrivilegeImpact + weakest edge strength.
       - Build a prose description of the corridor.
       - Emit one ExpansionCorridor.

  4. Risk-ranked corridor list
     Sort all corridors: CRITICAL → HIGH → MEDIUM → LOW, then by privilege_delta desc.

Design decisions
----------------
- BFS rather than DFS: shortest-path footholds matter more than exhaustive paths.
- max_depth=6: beyond 6 hops the noise-to-signal ratio degrades and runtime grows.
- HIGH_THRESHOLD=50: nodes below 50 privilege_value are not interesting terminal targets.
- MAX_ALTERNATIVES=3: one primary + three alternatives per corridor (report clarity).
- We deliberately do not use Dijkstra's traversal_cost minimisation - we want
  the *easiest* path (min cost), not the *cheapest* in a weighted-shortest-path sense.
  A separate "easiest attack" rank is produced via the weakest_edge_cost field.
"""

from __future__ import annotations

import collections

import structlog

from systems.simula.inspector.trust_types import (
    CorridorRiskTier,
    ExpansionCorridor,
    FootholdBinding,
    PrivilegeImpact,
    PropagationPath,
    PropagationSimulation,
    PropagationStep,
    ReachabilityMap,
    TrustEdgeKind,
    TrustGraph,
    TrustNode,
    TrustNodeKind,
)

logger = structlog.get_logger().bind(system="simula.inspector.propagation_engine")

# Tuning constants
MAX_DEPTH         = 6
HIGH_THRESHOLD    = 50   # minimum privilege_value for a node to be a corridor terminal
MAX_ALTERNATIVES  = 3
MAX_PATHS_PER_SIM = 200  # guard against path explosion on dense graphs


def _risk_tier(
    impact: PrivilegeImpact,
    weakest_cost: float,
) -> CorridorRiskTier:
    """
    Determine corridor risk tier from terminal impact + weakest edge cost.

    Critical terminal → CRITICAL regardless of edge cost.
    High terminal + weak edge (cost ≤ 0.3) → CRITICAL; else HIGH.
    Medium terminal → HIGH if edge is weak, else MEDIUM.
    """
    if impact == PrivilegeImpact.CRITICAL:
        return CorridorRiskTier.CRITICAL
    if impact == PrivilegeImpact.HIGH:
        return CorridorRiskTier.CRITICAL if weakest_cost <= 0.3 else CorridorRiskTier.HIGH
    if impact == PrivilegeImpact.MEDIUM:
        return CorridorRiskTier.HIGH if weakest_cost <= 0.2 else CorridorRiskTier.MEDIUM
    return CorridorRiskTier.LOW


_TIER_ORDER = {
    CorridorRiskTier.CRITICAL: 0,
    CorridorRiskTier.HIGH:     1,
    CorridorRiskTier.MEDIUM:   2,
    CorridorRiskTier.LOW:      3,
}


class PropagationEngine:
    """
    Propagation simulator - runs per-foothold BFS and extracts ExpansionCorridors.
    """

    def __init__(
        self,
        max_depth: int = MAX_DEPTH,
        strict_gradient: bool = True,
        high_threshold: int = HIGH_THRESHOLD,
        max_alternatives: int = MAX_ALTERNATIVES,
    ) -> None:
        self._max_depth       = max_depth
        self._strict_gradient = strict_gradient
        self._high_threshold  = high_threshold
        self._max_alternatives = max_alternatives
        self._log = logger

    # ── Primary entry point ───────────────────────────────────────────────────

    def simulate_all(
        self,
        target_id: str,
        graph: TrustGraph,
        bindings: list[FootholdBinding],
    ) -> ReachabilityMap:
        """
        Run propagation simulations for every foothold and build a ReachabilityMap.

        Args:
            target_id: Target identifier.
            graph: Fully built TrustGraph.
            bindings: FootholdBindings from TrustGraphBuilder.

        Returns:
            ReachabilityMap with ranked ExpansionCorridors.
        """
        log = self._log.bind(target_id=target_id)
        log.info("propagation_simulation_started", foothold_count=len(bindings))

        rmap = ReachabilityMap(target_id=target_id)

        # Collect unique foothold node IDs
        foothold_ids: list[str] = []
        seen_fh: set[str] = set()
        for b in bindings:
            if b.foothold_node_id not in seen_fh and b.foothold_node_id in graph.nodes:
                foothold_ids.append(b.foothold_node_id)
                seen_fh.add(b.foothold_node_id)

        # If no bindings, use all steerability-adjacent nodes as footholds
        if not foothold_ids:
            for node in graph.nodes.values():
                if node.steerability_adjacent and node.node_id not in seen_fh:
                    foothold_ids.append(node.node_id)
                    seen_fh.add(node.node_id)

        # Fall back further: use all CREDENTIAL and PRINCIPAL nodes
        if not foothold_ids:
            for kind in (TrustNodeKind.CREDENTIAL, TrustNodeKind.PRINCIPAL):
                for node in graph.nodes_of_kind(kind):
                    if node.node_id not in seen_fh:
                        foothold_ids.append(node.node_id)
                        seen_fh.add(node.node_id)

        # Run per-foothold simulation
        all_corridors: list[ExpansionCorridor] = []
        for fh_id in foothold_ids:
            sim = self.simulate(
                target_id=target_id,
                graph=graph,
                foothold_node_id=fh_id,
                binding=next((b for b in bindings if b.foothold_node_id == fh_id), None),
            )
            rmap.simulations[fh_id] = sim
            all_corridors.extend(sim.corridors)

        # Deduplicate corridors (same foothold + terminal may appear in multiple sims)
        seen_pairs: set[tuple[str, str]] = set()
        deduped: list[ExpansionCorridor] = []
        for c in all_corridors:
            key = (c.foothold_node_id, c.terminal_node_id)
            if key not in seen_pairs:
                seen_pairs.add(key)
                deduped.append(c)

        # Rank
        deduped.sort(key=lambda c: (
            _TIER_ORDER.get(c.risk_tier, 99),
            -c.max_privilege_delta,
            c.min_hops,
        ))

        rmap.ranked_corridors = deduped
        rmap.total_footholds         = len(foothold_ids)
        rmap.total_reachable_nodes   = sum(s.total_reachable_nodes for s in rmap.simulations.values())
        rmap.total_corridors         = len(deduped)
        rmap.critical_corridors      = sum(1 for c in deduped if c.risk_tier == CorridorRiskTier.CRITICAL)
        rmap.high_corridors          = sum(1 for c in deduped if c.risk_tier == CorridorRiskTier.HIGH)
        rmap.medium_corridors        = sum(1 for c in deduped if c.risk_tier == CorridorRiskTier.MEDIUM)
        rmap.low_corridors           = sum(1 for c in deduped if c.risk_tier == CorridorRiskTier.LOW)
        rmap.global_max_privilege    = max(
            (n.privilege_value for n in graph.nodes.values()), default=0
        )

        log.info(
            "propagation_simulation_complete",
            simulations=len(rmap.simulations),
            total_corridors=rmap.total_corridors,
            critical=rmap.critical_corridors,
            high=rmap.high_corridors,
        )

        return rmap

    # ── Per-foothold simulation ───────────────────────────────────────────────

    def simulate(
        self,
        target_id: str,
        graph: TrustGraph,
        foothold_node_id: str,
        binding: FootholdBinding | None = None,
    ) -> PropagationSimulation:
        """
        BFS from foothold_node_id; produce PropagationSimulation with corridors.
        """
        foothold = graph.nodes.get(foothold_node_id)
        if not foothold:
            return PropagationSimulation(
                target_id=target_id,
                foothold_node_id=foothold_node_id,
                foothold_node_name="unknown",
                foothold_privilege_value=0,
            )

        # BFS to find all reachable nodes + shortest paths
        reachable, parent_edge = self._bfs(graph, foothold_node_id)

        sim = PropagationSimulation(
            target_id=target_id,
            foothold_node_id=foothold_node_id,
            foothold_node_name=foothold.name,
            foothold_privilege_value=foothold.privilege_value,
            reachable_node_ids=list(reachable),
            max_depth=self._max_depth,
        )

        sim.privilege_gain_node_ids = [
            nid for nid in reachable
            if graph.nodes[nid].privilege_value > foothold.privilege_value
        ]

        # Build paths to interesting (high-value) terminals
        paths: list[PropagationPath] = []
        terminal_nodes = [
            graph.nodes[nid]
            for nid in reachable
            if graph.nodes[nid].privilege_value >= self._high_threshold
            and nid != foothold_node_id
        ]

        # Sort terminals by privilege value descending
        terminal_nodes.sort(key=lambda n: n.privilege_value, reverse=True)

        for terminal in terminal_nodes[:MAX_PATHS_PER_SIM]:
            path = self._reconstruct_path(
                graph=graph,
                target_id=target_id,
                foothold=foothold,
                terminal=terminal,
                parent_edge=parent_edge,
            )
            if path and (not self._strict_gradient or path.is_monotonic):
                paths.append(path)

        sim.paths = paths
        sim.total_paths          = len(paths)
        sim.total_reachable_nodes = len(reachable)
        sim.total_privilege_gains = len(sim.privilege_gain_node_ids)
        if paths:
            sim.max_privilege_reached = max(p.terminal_privilege_value for p in paths)
            sim.max_privilege_delta   = max(p.privilege_delta for p in paths)

        # Extract corridors
        corridors = self._extract_corridors(
            graph=graph,
            target_id=target_id,
            foothold=foothold,
            paths=paths,
            binding=binding,
        )
        sim.corridors       = corridors
        sim.total_corridors = len(corridors)

        return sim

    # ── BFS ───────────────────────────────────────────────────────────────────

    def _bfs(
        self,
        graph: TrustGraph,
        start_id: str,
    ) -> tuple[set[str], dict[str, tuple[str, str]]]:
        """
        BFS from start_id.

        Returns:
            reachable: set of all visited node IDs (excluding start)
            parent_edge: node_id → (parent_node_id, edge_id) for path reconstruction
        """
        reachable: set[str] = set()
        parent_edge: dict[str, tuple[str, str]] = {}
        depth: dict[str, int] = {start_id: 0}
        queue: collections.deque[str] = collections.deque([start_id])

        while queue:
            current_id = queue.popleft()
            current_depth = depth[current_id]

            if current_depth >= self._max_depth:
                continue

            for edge, neighbor in graph.successors(current_id):
                nid = neighbor.node_id
                if nid not in depth:
                    depth[nid] = current_depth + 1
                    parent_edge[nid] = (current_id, edge.edge_id)
                    reachable.add(nid)
                    queue.append(nid)

        return reachable, parent_edge

    # ── Path reconstruction ───────────────────────────────────────────────────

    def _reconstruct_path(
        self,
        graph: TrustGraph,
        target_id: str,
        foothold: TrustNode,
        terminal: TrustNode,
        parent_edge: dict[str, tuple[str, str]],
    ) -> PropagationPath | None:
        """
        Walk parent_edge backwards from terminal to foothold to reconstruct the path.
        """
        if terminal.node_id not in parent_edge and terminal.node_id != foothold.node_id:
            return None

        # Collect reversed sequence of (node_id, edge_id)
        steps_reversed: list[tuple[str, str | None]] = []  # (node_id, arriving_edge_id)
        current = terminal.node_id
        while current != foothold.node_id:
            if current not in parent_edge:
                return None
            parent_id, edge_id = parent_edge[current]
            steps_reversed.append((current, edge_id))
            current = parent_id

        steps_reversed.reverse()  # now foothold→terminal order

        if not steps_reversed:
            return None

        # Build PropagationStep list
        prop_steps: list[PropagationStep] = []
        node_ids = [foothold.node_id]
        edge_ids: list[str] = []
        total_cost = 0.0
        is_monotonic = True
        prev_priv = foothold.privilege_value
        weakest_eid = ""
        weakest_cost = 1.0

        for idx, (arrived_id, eid) in enumerate(steps_reversed):
            if eid is None:
                continue
            arrived_node = graph.nodes.get(arrived_id)
            edge         = graph.edges.get(eid)
            if not arrived_node or not edge:
                continue

            ec = edge.traversal_cost
            total_cost += ec
            node_ids.append(arrived_id)
            edge_ids.append(eid)

            if arrived_node.privilege_value < prev_priv:
                is_monotonic = False
            prev_priv = arrived_node.privilege_value

            if ec < weakest_cost:
                weakest_cost = ec
                weakest_eid  = eid

            prop_steps.append(PropagationStep(
                step_index=idx,
                edge_id=eid,
                edge_kind=edge.kind,
                edge_strength=edge.strength,
                edge_traversal_cost=ec,
                arrived_node_id=arrived_id,
                arrived_node_name=arrived_node.name,
                arrived_node_kind=arrived_node.kind,
                arrived_privilege_value=arrived_node.privilege_value,
                arrived_privilege_impact=arrived_node.privilege_impact,
            ))

        if not prop_steps:
            return None

        # Evidence: fragments from nodes on the path
        evidence_frags: list[str] = []
        for nid in node_ids:
            n = graph.nodes.get(nid)
            if n:
                evidence_frags.extend(n.derived_from_fragment_ids[:2])

        confidence = min(
            1.0,
            (foothold.privilege_value / 100) * 0.3
            + (terminal.privilege_value / 100) * 0.5
            + (1.0 - total_cost / max(len(prop_steps), 1)) * 0.2,
        )

        return PropagationPath(
            target_id=target_id,
            foothold_node_id=foothold.node_id,
            foothold_node_name=foothold.name,
            terminal_node_id=terminal.node_id,
            terminal_node_name=terminal.name,
            terminal_privilege_value=terminal.privilege_value,
            terminal_privilege_impact=terminal.privilege_impact,
            steps=prop_steps,
            node_ids=node_ids,
            edge_ids=edge_ids,
            path_length=len(prop_steps),
            total_traversal_cost=round(total_cost, 4),
            privilege_delta=terminal.privilege_value - foothold.privilege_value,
            is_monotonic=is_monotonic,
            weakest_edge_id=weakest_eid,
            weakest_edge_kind=graph.edges[weakest_eid].kind if weakest_eid and weakest_eid in graph.edges else TrustEdgeKind.UNKNOWN,
            weakest_edge_cost=round(weakest_cost, 4),
            evidence_fragment_ids=list(dict.fromkeys(evidence_frags))[:6],
            confidence=round(confidence, 4),
        )

    # ── Corridor extraction ───────────────────────────────────────────────────

    def _extract_corridors(
        self,
        graph: TrustGraph,
        target_id: str,
        foothold: TrustNode,
        paths: list[PropagationPath],
        binding: FootholdBinding | None,
    ) -> list[ExpansionCorridor]:
        """
        Group paths by terminal node and emit one ExpansionCorridor per terminal.
        """
        if not paths:
            return []

        # Group by terminal
        terminal_paths: dict[str, list[PropagationPath]] = {}
        for p in paths:
            terminal_paths.setdefault(p.terminal_node_id, []).append(p)

        corridors: list[ExpansionCorridor] = []
        for terminal_id, tpaths in terminal_paths.items():
            terminal = graph.nodes.get(terminal_id)
            if not terminal:
                continue

            # Sort by path_length (shortest first), then by total_traversal_cost (cheapest)
            tpaths.sort(key=lambda p: (p.path_length, p.total_traversal_cost))
            primary     = tpaths[0]
            alternatives = tpaths[1:1 + self._max_alternatives]

            # Weakest edge across all paths for this corridor
            all_paths = [primary] + alternatives
            weakest_cost = min((p.weakest_edge_cost for p in all_paths), default=1.0)
            weakest_eid  = min(all_paths, key=lambda p: p.weakest_edge_cost).weakest_edge_id

            # Edge kinds exploited
            edge_kinds: list[TrustEdgeKind] = list({
                step.edge_kind
                for p in all_paths
                for step in p.steps
            })

            # Risk tier
            tier = _risk_tier(terminal.privilege_impact, weakest_cost)

            # Narrative
            desc = self._build_corridor_description(
                foothold=foothold,
                terminal=terminal,
                primary=primary,
                tier=tier,
            )

            # Evidence
            frag_ids = list(dict.fromkeys(
                fid
                for p in all_paths
                for fid in p.evidence_fragment_ids
            ))[:8]
            cs_ids = [binding.condition_set_id] if binding and binding.condition_set_id else []

            c = ExpansionCorridor(
                target_id=target_id,
                risk_tier=tier,
                primary_path=primary,
                alternative_paths=alternatives,
                foothold_binding=binding,
                foothold_node_id=foothold.node_id,
                foothold_node_name=foothold.name,
                terminal_node_id=terminal_id,
                terminal_node_name=terminal.name,
                terminal_privilege_impact=terminal.privilege_impact,
                exploited_edge_kinds=edge_kinds,
                weakest_edge_id=weakest_eid,
                weakest_edge_cost=round(weakest_cost, 4),
                min_hops=primary.path_length,
                max_privilege_delta=max(p.privilege_delta for p in all_paths),
                description=desc,
                evidence_fragment_ids=frag_ids,
                evidence_condition_set_ids=cs_ids,
            )
            corridors.append(c)

        # Sort within this simulation's corridors
        corridors.sort(key=lambda c: (
            _TIER_ORDER.get(c.risk_tier, 99),
            -c.max_privilege_delta,
            c.min_hops,
        ))
        return corridors

    # ── Narrative builder ─────────────────────────────────────────────────────

    @staticmethod
    def _build_corridor_description(
        foothold: TrustNode,
        terminal: TrustNode,
        primary: PropagationPath,
        tier: CorridorRiskTier,
    ) -> str:
        """
        Build a one-paragraph prose description of an expansion corridor.
        """
        hop_count = primary.path_length
        edge_kinds_used = list({step.edge_kind.value for step in primary.steps})
        edge_summary = ", ".join(edge_kinds_used[:3]) + ("…" if len(edge_kinds_used) > 3 else "")

        intermediate = [
            step.arrived_node_name
            for step in primary.steps[:-1]  # all steps except the terminal
        ]
        via_clause = ""
        if intermediate:
            via_clause = f" via {' → '.join(intermediate[:3])}{'…' if len(intermediate) > 3 else ''}"

        weakness_note = (
            f"The weakest link is a '{primary.weakest_edge_kind.value}' edge "
            f"(traversal cost {primary.weakest_edge_cost:.2f}), enabling low-effort traversal."
            if primary.weakest_edge_cost < 0.35
            else f"All edges require non-trivial trust conditions (min cost {primary.weakest_edge_cost:.2f})."
        )

        return (
            f"[{tier.value.upper()}] Starting from foothold node '{foothold.name}' "
            f"({foothold.kind.value}), influence propagates across {hop_count} trust "
            f"hop{'s' if hop_count != 1 else ''}{via_clause} to reach '{terminal.name}' "
            f"({terminal.kind.value}, impact: {terminal.privilege_impact.value}). "
            f"The path exploits {edge_summary} trust edge types. "
            f"{weakness_note} "
            f"Privilege delta: +{primary.privilege_delta} ({foothold.privilege_value} → "
            f"{terminal.privilege_value})."
        )
