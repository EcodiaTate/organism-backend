"""
EcodiaOS — Inspector Phase 8: Correlation Engine

Three-component engine for Phase 8 cross-layer correlation:

  CausalGraphBuilder       — constructs EventNode graph from Phase 3–7 artifacts
  CausalChainAssembler     — DFS path-finding to produce end-to-end CausalChains
  PropagationMotifMiner    — pattern mining for recurring sub-graph motifs
  SynthesisBuilder         — narrative assembly for the FinalSynthesis

Design
------
CausalGraphBuilder traverses all available Phase artifacts and creates one
EventNode per meaningful finding (protocol boundary state, trust corridor,
distinguishable channel, etc.).  Edges are created by applying a set of
typed LinkingRules that match cross-phase artifact relationships.

Linking rules (priority order):
  1. Phase 6 boundary failure → Phase 7 channel (via source_fsm_state_id match)
  2. Phase 5 corridor → Phase 6 FSM state (via trust-edge privilege escalation)
  3. Phase 4 region → Phase 5 corridor (via foothold binding)
  4. Phase 4 region → Phase 6 scenario (via steerable branch → boundary path)
  5. Phase 7 channel → timing oracle event (timing leakage terminal)
  6. Synthetic input node → Phase 4 region (root anchoring)

CausalChainAssembler runs DFS from each root_node to each terminal_node,
collecting all paths that cross ≥2 layers.  Each path becomes a CausalChain
with an assembled narrative and cumulative confidence score.

PropagationMotifMiner applies a structural fingerprinting algorithm:
each chain is represented as a (layer_sequence, mechanism_sequence) tuple.
Chains that share ≥3-step subsequences form motif groups.

SynthesisBuilder ranks motifs by (security_relevance, occurrence_count,
impact_score), assembles key findings, and builds the final thesis paragraph.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

import structlog

from systems.simula.inspector.correlation_types import (
    CausalChain,
    CausalEdge,
    CausalGraph,
    CausalMechanism,
    ChainStatus,
    ChainStep,
    CorrelationID,
    CorrelationLayer,
    EventNode,
    FinalSynthesis,
    MotifCatalog,
    MotifInstance,
    MotifKind,
    PropagationMotif,
    SynthesisKeyFinding,
)

if TYPE_CHECKING:
    from systems.simula.inspector.constraint_types import Phase4Result
    from systems.simula.inspector.protocol_types import Phase6Result
    from systems.simula.inspector.static_types import Phase3Result
    from systems.simula.inspector.trust_types import Phase5Result
    from systems.simula.inspector.variance_types import Phase7Result

logger = structlog.get_logger().bind(system="simula.inspector.correlation_engine")


# ── CausalGraphBuilder ─────────────────────────────────────────────────────────


class CausalGraphBuilder:
    """
    Constructs a CausalGraph from available Phase 3–7 artifacts.

    Parameters
    ----------
    min_edge_weight  — minimum confidence to include an edge (default 0.3)
    """

    def __init__(self, min_edge_weight: float = 0.3) -> None:
        self._min_w = min_edge_weight
        self._log = logger.bind(component="CausalGraphBuilder")

    def build(
        self,
        target_id: str,
        phase3_result: Phase3Result | None = None,
        phase4_result: Phase4Result | None = None,
        phase5_result: Phase5Result | None = None,
        phase6_result: Phase6Result | None = None,
        phase7_result: Phase7Result | None = None,
    ) -> CausalGraph:
        graph = CausalGraph(target_id=target_id)

        # Layer 0: synthetic input root node
        root = EventNode(
            correlation_id=CorrelationID(
                layer=CorrelationLayer.INPUT,
                artifact_id="input_root",
                artifact_kind="SyntheticInput",
                label="Input",
            ),
            layer=CorrelationLayer.INPUT,
            sensitivity=10,
            is_root=True,
            description="External input / initial observation",
            evidence_quality=1.0,
        )
        graph.add_node(root)

        # Phase 3: static fragments
        if phase3_result:
            self._add_phase3_nodes(graph, phase3_result, root)

        # Phase 4: control-flow regions
        if phase4_result:
            self._add_phase4_nodes(graph, phase4_result, root)

        # Phase 5: trust corridors
        if phase5_result:
            self._add_phase5_nodes(graph, phase5_result, graph)

        # Phase 6: boundary failures + FSM states
        if phase6_result:
            self._add_phase6_nodes(graph, phase6_result, graph)

        # Phase 7: distinguishable channels (timing terminals)
        if phase7_result:
            self._add_phase7_nodes(graph, phase7_result, graph)

        # Cross-phase linking
        self._link_phases(
            graph,
            phase4_result=phase4_result,
            phase5_result=phase5_result,
            phase6_result=phase6_result,
            phase7_result=phase7_result,
        )

        self._log.debug(
            "graph_built",
            nodes=graph.total_nodes,
            edges=graph.total_edges,
            cross_layer=graph.total_cross_layer_edges,
            layers=graph.distinct_layers,
        )
        return graph

    # ── Node population ───────────────────────────────────────────────────────

    def _add_phase3_nodes(
        self,
        graph: CausalGraph,
        p3: Phase3Result,
        root: EventNode,
    ) -> None:
        catalog = getattr(p3, "fragment_catalog", None)
        if not catalog:
            return
        fragments = getattr(catalog, "fragments", {})
        for frag_id, frag in list(fragments.items())[:5]:
            node = EventNode(
                correlation_id=CorrelationID(
                    layer=CorrelationLayer.STATIC,
                    artifact_id=frag_id,
                    artifact_kind="Phase3.Fragment",
                    label=getattr(frag, "name", frag_id[:8]),
                ),
                layer=CorrelationLayer.STATIC,
                sensitivity=30,
                description=getattr(frag, "description", f"Static fragment {frag_id[:8]}"),
                evidence_quality=0.8,
            )
            graph.add_node(node)
            self._add_edge(graph, CausalEdge(
                from_node_id=root.node_id,
                to_node_id=node.node_id,
                mechanism=CausalMechanism.DATA_DEPENDENCY,
                is_cross_layer=True,
                from_layer=CorrelationLayer.INPUT,
                to_layer=CorrelationLayer.STATIC,
                weight=0.6,
                description=f"Input drives execution through static fragment {frag_id[:8]}",
            ))

    def _add_phase4_nodes(
        self,
        graph: CausalGraph,
        p4: Phase4Result,
        root: EventNode,
    ) -> None:
        regions = getattr(p4, "steerable_regions", {})
        for region_id, region in list(regions.items()):
            desc = getattr(region, "description", f"Control-flow region {region_id[:8]}")
            sens = min(100, int(getattr(region, "sensitivity", 50)))
            node = EventNode(
                correlation_id=CorrelationID(
                    layer=CorrelationLayer.CONTROL_FLOW,
                    artifact_id=region_id,
                    artifact_kind="Phase4.ControlFlowRegion",
                    label=desc[:30],
                ),
                layer=CorrelationLayer.CONTROL_FLOW,
                sensitivity=sens,
                description=desc,
                evidence_quality=0.75,
            )
            graph.add_node(node)
            self._add_edge(graph, CausalEdge(
                from_node_id=root.node_id,
                to_node_id=node.node_id,
                mechanism=CausalMechanism.CONTROL_DEPENDENCY,
                is_cross_layer=True,
                from_layer=CorrelationLayer.INPUT,
                to_layer=CorrelationLayer.CONTROL_FLOW,
                weight=0.65,
                description=f"Input value steers branch in region {region_id[:8]}",
            ))

    def _add_phase5_nodes(
        self,
        graph: CausalGraph,
        p5: Phase5Result,
        _graph: CausalGraph,
    ) -> None:
        corridors = getattr(p5, "corridors", [])
        for corridor in corridors:
            c_id = getattr(corridor, "corridor_id", "")
            desc = getattr(corridor, "description", f"Trust corridor {c_id[:8]}")
            priv = getattr(corridor, "privilege_gain", 50)
            node = EventNode(
                correlation_id=CorrelationID(
                    layer=CorrelationLayer.PROCESS,
                    artifact_id=c_id,
                    artifact_kind="Phase5.ExpansionCorridor",
                    label=desc[:30],
                ),
                layer=CorrelationLayer.PROCESS,
                sensitivity=min(100, int(priv)),
                description=desc,
                evidence_quality=0.8,
            )
            graph.add_node(node)

    def _add_phase6_nodes(
        self,
        graph: CausalGraph,
        p6: Phase6Result,
        _graph: CausalGraph,
    ) -> None:
        for fsm in p6.fsms:
            for state in fsm.boundary_states()[:4]:
                node = EventNode(
                    correlation_id=CorrelationID(
                        layer=CorrelationLayer.PROTOCOL,
                        artifact_id=state.state_id,
                        artifact_kind="Phase6.ProtocolFsmState",
                        label=state.name,
                    ),
                    layer=CorrelationLayer.PROTOCOL,
                    sensitivity=70,
                    description=f"Protocol boundary state '{state.name}' ({state.layer.value})",
                    evidence_quality=0.85,
                )
                graph.add_node(node)

        for failure in p6.boundary_failures[:6]:
            is_sec = failure.is_security_relevant
            node = EventNode(
                correlation_id=CorrelationID(
                    layer=CorrelationLayer.PROTOCOL,
                    artifact_id=failure.failure_id,
                    artifact_kind="Phase6.BoundaryFailure",
                    label=f"Failure:{failure.boundary_kind.value[:10]}",
                ),
                layer=CorrelationLayer.PROTOCOL,
                sensitivity=80 if is_sec else 60,
                is_security_relevant=is_sec,
                description=failure.anomaly_description or f"Boundary failure {failure.failure_id[:8]}",
                evidence_quality=0.9,
            )
            graph.add_node(node)

    def _add_phase7_nodes(
        self,
        graph: CausalGraph,
        p7: Phase7Result,
        _graph: CausalGraph,
    ) -> None:
        for dr in p7.distinguishability_results:
            is_dist = dr.is_distinguishable
            node = EventNode(
                correlation_id=CorrelationID(
                    layer=CorrelationLayer.TIMING,
                    artifact_id=dr.result_id,
                    artifact_kind="Phase7.DistinguishabilityResult",
                    label=f"{dr.op_name[:20]}",
                ),
                layer=CorrelationLayer.TIMING,
                sensitivity=85 if dr.is_security_relevant else 50,
                is_terminal=True,
                is_security_relevant=dr.is_security_relevant,
                description=dr.evidence_summary[:100],
                evidence_quality=0.9 if is_dist else 0.5,
            )
            graph.add_node(node)

    # ── Cross-phase linking ───────────────────────────────────────────────────

    def _link_phases(
        self,
        graph: CausalGraph,
        phase4_result: Phase4Result | None,
        phase5_result: Phase5Result | None,
        phase6_result: Phase6Result | None,
        phase7_result: Phase7Result | None,
    ) -> None:
        # Build lookup maps
        cf_nodes   = {n.correlation_id.artifact_id: n for n in graph.nodes_at_layer(CorrelationLayer.CONTROL_FLOW)}
        proc_nodes = {n.correlation_id.artifact_id: n for n in graph.nodes_at_layer(CorrelationLayer.PROCESS)}
        proto_nodes = {n.correlation_id.artifact_id: n for n in graph.nodes_at_layer(CorrelationLayer.PROTOCOL)}
        timing_nodes = {n.correlation_id.artifact_id: n for n in graph.nodes_at_layer(CorrelationLayer.TIMING)}

        # Rule 1: Phase 4 regions → Phase 5 corridors (via foothold binding)
        if phase5_result:
            footholds = getattr(phase5_result, "foothold_bindings", [])
            for fh in footholds:
                region_id = getattr(fh, "region_id", "")
                corridor_id = getattr(fh, "corridor_id", "")
                cf_node = cf_nodes.get(region_id)
                proc_node = proc_nodes.get(corridor_id)
                if cf_node and proc_node:
                    self._add_edge(graph, CausalEdge(
                        from_node_id=cf_node.node_id,
                        to_node_id=proc_node.node_id,
                        mechanism=CausalMechanism.TRUST_DELEGATION,
                        is_cross_layer=True,
                        from_layer=CorrelationLayer.CONTROL_FLOW,
                        to_layer=CorrelationLayer.PROCESS,
                        weight=0.75,
                        description=(
                            f"Steerable region {region_id[:8]} maps to "
                            f"trust corridor {corridor_id[:8]} via foothold binding"
                        ),
                        evidence_ids=[region_id, corridor_id],
                    ))

        # Rule 2: Phase 5 corridors → Phase 6 FSM boundary states
        # (corridors targeting high-privilege auth nodes link to auth-related FSM states)
        if phase5_result and phase6_result:
            corridors = getattr(phase5_result, "corridors", [])
            for corridor in corridors:
                c_id = getattr(corridor, "corridor_id", "")
                proc_node = proc_nodes.get(c_id)
                if not proc_node:
                    continue
                # Link to Phase 6 auth-window boundary states
                for fsm in phase6_result.fsms:
                    for state in fsm.boundary_states():
                        if "auth" in state.name.lower() or "cred" in state.name.lower():
                            proto_node = proto_nodes.get(state.state_id)
                            if proto_node:
                                self._add_edge(graph, CausalEdge(
                                    from_node_id=proc_node.node_id,
                                    to_node_id=proto_node.node_id,
                                    mechanism=CausalMechanism.PROTOCOL_TRANSITION,
                                    is_cross_layer=True,
                                    from_layer=CorrelationLayer.PROCESS,
                                    to_layer=CorrelationLayer.PROTOCOL,
                                    weight=0.65,
                                    description=(
                                        f"Trust corridor {c_id[:8]} traversal "
                                        f"leads to protocol boundary state '{state.name}'"
                                    ),
                                    evidence_ids=[c_id, state.state_id],
                                ))

        # Rule 3: Phase 6 FSM states → Phase 6 boundary failures (within same FSM)
        if phase6_result:
            for fsm in phase6_result.fsms:
                for failure in phase6_result.boundary_failures:
                    if failure.failing_state_id in fsm.states:
                        state_node = proto_nodes.get(failure.failing_state_id)
                        fail_node = proto_nodes.get(failure.failure_id)
                        if state_node and fail_node:
                            self._add_edge(graph, CausalEdge(
                                from_node_id=state_node.node_id,
                                to_node_id=fail_node.node_id,
                                mechanism=CausalMechanism.STATE_POLLUTION,
                                is_cross_layer=False,
                                from_layer=CorrelationLayer.PROTOCOL,
                                to_layer=CorrelationLayer.PROTOCOL,
                                weight=0.85,
                                description=(
                                    f"Boundary state leads to failure "
                                    f"({failure.boundary_kind.value})"
                                ),
                                evidence_ids=[failure.failing_state_id, failure.failure_id],
                            ))

        # Rule 4: Phase 6 failures → Phase 7 channels (via source_fsm_state_id)
        if phase6_result and phase7_result:
            for dr in phase7_result.distinguishability_results:
                fsm_state_id = dr.profile.source_fsm_state_id
                if not fsm_state_id:
                    continue
                # Find the corresponding boundary failure
                matching_failures = [
                    f for f in phase6_result.boundary_failures
                    if f.failing_state_id == fsm_state_id
                ]
                for failure in matching_failures[:1]:
                    fail_node = proto_nodes.get(failure.failure_id)
                    timing_node = timing_nodes.get(dr.result_id)
                    if fail_node and timing_node:
                        w = 0.8 if dr.is_distinguishable else 0.45
                        self._add_edge(graph, CausalEdge(
                            from_node_id=fail_node.node_id,
                            to_node_id=timing_node.node_id,
                            mechanism=CausalMechanism.TIMING_ORACLE,
                            is_cross_layer=True,
                            from_layer=CorrelationLayer.PROTOCOL,
                            to_layer=CorrelationLayer.TIMING,
                            weight=w,
                            description=(
                                f"Protocol boundary failure propagates to "
                                f"timing channel '{dr.op_name[:30]}'"
                            ),
                            evidence_ids=[failure.failure_id, dr.result_id],
                        ))

        # Rule 5: Phase 4 regions → Phase 6 scenarios (steerable branch → boundary path)
        if phase4_result and phase6_result:
            regions = getattr(phase4_result, "steerable_regions", {})
            for fsm in phase6_result.fsms:
                for state in fsm.boundary_states()[:2]:
                    # Match by name similarity (heuristic)
                    proto_node = proto_nodes.get(state.state_id)
                    if not proto_node:
                        continue
                    for region_id, _region in list(regions.items())[:3]:
                        cf_node = cf_nodes.get(region_id)
                        if cf_node:
                            self._add_edge(graph, CausalEdge(
                                from_node_id=cf_node.node_id,
                                to_node_id=proto_node.node_id,
                                mechanism=CausalMechanism.CONTROL_DEPENDENCY,
                                is_cross_layer=True,
                                from_layer=CorrelationLayer.CONTROL_FLOW,
                                to_layer=CorrelationLayer.PROTOCOL,
                                weight=0.55,
                                description=(
                                    f"Control-flow region {region_id[:8]} branch "
                                    f"leads to protocol boundary state '{state.name}'"
                                ),
                                evidence_ids=[region_id, state.state_id],
                            ))

        # Rule 6: Phase 7 channels directly from Phase 5 corridors (timing_covariation)
        if phase5_result and phase7_result:
            corridors = getattr(phase5_result, "corridors", [])
            for corridor in corridors:
                c_id = getattr(corridor, "corridor_id", "")
                proc_node = proc_nodes.get(c_id)
                if not proc_node:
                    continue
                for dr in phase7_result.distinguishability_results:
                    if dr.profile.source_corridor_id == c_id:
                        timing_node = timing_nodes.get(dr.result_id)
                        if timing_node:
                            self._add_edge(graph, CausalEdge(
                                from_node_id=proc_node.node_id,
                                to_node_id=timing_node.node_id,
                                mechanism=CausalMechanism.TIMING_COVARIATION,
                                is_cross_layer=True,
                                from_layer=CorrelationLayer.PROCESS,
                                to_layer=CorrelationLayer.TIMING,
                                weight=0.7 if dr.is_distinguishable else 0.4,
                                description=(
                                    f"Trust corridor {c_id[:8]} traversal "
                                    f"covaries with timing channel '{dr.op_name[:30]}'"
                                ),
                                evidence_ids=[c_id, dr.result_id],
                            ))

    def _add_edge(self, graph: CausalGraph, edge: CausalEdge) -> None:
        if edge.weight >= self._min_w:
            graph.add_edge(edge)


# ── CausalChainAssembler ───────────────────────────────────────────────────────


class CausalChainAssembler:
    """
    Assembles CausalChains from a CausalGraph by DFS path-finding.

    Finds all paths from root nodes to terminal nodes that span ≥2 distinct
    layers.  Paths are converted to CausalChain objects with assembled
    narratives and cumulative confidence scores.

    Parameters
    ----------
    max_depth     — maximum chain length in hops (default 8)
    min_layers    — minimum distinct layers a chain must span (default 2)
    max_chains    — maximum chains to return (default 30)
    """

    def __init__(
        self,
        max_depth: int = 8,
        min_layers: int = 2,
        max_chains: int = 30,
    ) -> None:
        self._max_depth = max_depth
        self._min_layers = min_layers
        self._max_chains = max_chains
        self._log = logger.bind(component="CausalChainAssembler")

    def assemble(self, graph: CausalGraph) -> list[CausalChain]:
        """Find all chains from root to terminal nodes."""
        chains: list[CausalChain] = []

        for root_id in graph.root_node_ids:
            for terminal_id in graph.terminal_node_ids:
                paths = self._dfs_paths(graph, root_id, terminal_id)
                for path_nodes, path_edges in paths:
                    chain = self._build_chain(graph, path_nodes, path_edges)
                    if chain:
                        chains.append(chain)
                    if len(chains) >= self._max_chains:
                        break
                if len(chains) >= self._max_chains:
                    break

        # Sort by (distinct_layer_count desc, confidence desc)
        chains.sort(key=lambda c: (c.distinct_layer_count, c.confidence), reverse=True)

        self._log.debug("chains_assembled", total=len(chains))
        return chains

    def _dfs_paths(
        self,
        graph: CausalGraph,
        start_id: str,
        end_id: str,
    ) -> list[tuple[list[str], list[str]]]:
        """DFS to find all paths from start to end up to max_depth."""
        results: list[tuple[list[str], list[str]]] = []

        def dfs(
            node_id: str,
            node_path: list[str],
            edge_path: list[str],
            visited: set[str],
        ) -> None:
            if len(node_path) > self._max_depth:
                return
            if node_id == end_id and len(node_path) > 1:
                results.append((list(node_path), list(edge_path)))
                return
            for edge, succ in graph.successors(node_id):
                if succ.node_id not in visited:
                    visited.add(succ.node_id)
                    node_path.append(succ.node_id)
                    edge_path.append(edge.edge_id)
                    dfs(succ.node_id, node_path, edge_path, visited)
                    node_path.pop()
                    edge_path.pop()
                    visited.discard(succ.node_id)

        dfs(start_id, [start_id], [], {start_id})
        return results[:20]  # cap per (start, end) pair

    def _build_chain(
        self,
        graph: CausalGraph,
        node_ids: list[str],
        edge_ids: list[str],
    ) -> CausalChain | None:
        steps: list[ChainStep] = []
        layers_seen: set[CorrelationLayer] = set()
        cross_hops = 0
        cumconf = 1.0
        reaches_timing = False
        is_sec = False

        for i, nid in enumerate(node_ids):
            node = graph.nodes.get(nid)
            if not node:
                return None
            layers_seen.add(node.layer)
            if node.layer == CorrelationLayer.TIMING:
                reaches_timing = True
            if node.is_security_relevant:
                is_sec = True

            eid = edge_ids[i - 1] if i > 0 else ""
            edge = graph.edges.get(eid)
            mech = edge.mechanism if edge else CausalMechanism.UNKNOWN
            is_cross = edge.is_cross_layer if edge else False
            if is_cross:
                cross_hops += 1
            if edge:
                cumconf *= edge.weight

            steps.append(ChainStep(
                step_index=i,
                node_id=nid,
                node_label=node.correlation_id.label,
                layer=node.layer,
                edge_id=eid,
                mechanism=mech,
                description=node.description[:80],
                is_cross_layer_step=is_cross,
                cumulative_confidence=round(cumconf, 4),
            ))

        distinct_layers = list(layers_seen)
        if len(distinct_layers) < self._min_layers:
            return None

        status = self._classify_status(distinct_layers)
        narrative = self._build_narrative(steps)

        return CausalChain(
            target_id=graph.target_id,
            steps=steps,
            layers_represented=distinct_layers,
            distinct_layer_count=len(distinct_layers),
            cross_layer_hops=cross_hops,
            status=status,
            confidence=round(cumconf, 4),
            reaches_timing_channel=reaches_timing,
            is_security_relevant=is_sec,
            narrative=narrative,
        )

    @staticmethod
    def _classify_status(layers: list[CorrelationLayer]) -> ChainStatus:
        layer_set = set(layers)
        has_timing = CorrelationLayer.TIMING in layer_set
        has_protocol = CorrelationLayer.PROTOCOL in layer_set
        has_process = CorrelationLayer.PROCESS in layer_set
        n = len(layer_set)
        if n >= 4 and has_timing:
            return ChainStatus.COMPLETE
        if n >= 3:
            return ChainStatus.PARTIAL
        if has_timing and has_protocol and not has_process:
            return ChainStatus.HYPOTHETICAL
        return ChainStatus.PARTIAL

    @staticmethod
    def _build_narrative(steps: list[ChainStep]) -> str:
        parts = []
        for step in steps:
            if step.step_index == 0:
                parts.append(f"Starting from {step.node_label}")
            else:
                mech_desc = step.mechanism.value.replace("_", " ")
                parts.append(f"→ [{mech_desc}] → {step.node_label} ({step.layer.value})")
        confidence_str = f" [end-to-end confidence: {steps[-1].cumulative_confidence:.2f}]"
        return " ".join(parts) + confidence_str


# ── PropagationMotifMiner ──────────────────────────────────────────────────────


class PropagationMotifMiner:
    """
    Mines recurring structural sub-graph patterns (motifs) from CausalChains.

    Algorithm:
    1. Each chain is converted to a (layer_seq, mechanism_seq) fingerprint tuple.
    2. All k-length (k=3..max_motif_len) subsequences are extracted.
    3. Subsequences appearing in ≥ min_occurrences chains form motif candidates.
    4. Candidates are classified by MotifKind and ranked by occurrence + impact.

    Parameters
    ----------
    min_occurrences  — minimum chain count for a motif (default 2)
    max_motif_len    — maximum motif pattern length (default 4)
    """

    def __init__(
        self,
        min_occurrences: int = 2,
        max_motif_len: int = 4,
    ) -> None:
        self._min_occ = min_occurrences
        self._max_len = max_motif_len
        self._log = logger.bind(component="PropagationMotifMiner")

    def mine(
        self,
        chains: list[CausalChain],
        graph: CausalGraph,
        target_id: str,
    ) -> MotifCatalog:
        catalog = MotifCatalog(target_id=target_id)

        if not chains:
            return catalog

        # Build fingerprints
        fps: dict[str, list[tuple[int, int]]] = {}  # chain_id → list[(layer_ord, mech_ord)]
        for chain in chains:
            fp = [
                (self._layer_ord(s.layer), self._mech_ord(s.mechanism))
                for s in chain.steps
            ]
            fps[chain.chain_id] = fp

        # Count subsequence occurrences
        subseq_chains: dict[tuple, list[str]] = defaultdict(list)
        for chain_id, fp in fps.items():
            for length in range(3, min(self._max_len + 1, len(fp) + 1)):
                for start in range(len(fp) - length + 1):
                    subseq = tuple(fp[start : start + length])
                    subseq_chains[subseq].append(chain_id)

        # Filter by min_occurrences + deduplicate by chain set
        motifs_added: set[frozenset] = set()
        for subseq, chain_ids in subseq_chains.items():
            unique_chains = list(dict.fromkeys(chain_ids))  # preserve order, deduplicate
            if len(unique_chains) < self._min_occ:
                continue
            frozen = frozenset(unique_chains)
            if frozen in motifs_added:
                continue
            motifs_added.add(frozen)

            kind = self._classify_motif(subseq)
            mech = CausalMechanism(
                list(CausalMechanism)[min(subseq[-1][1], len(CausalMechanism) - 1)]
            )
            layers = [
                CorrelationLayer(list(CorrelationLayer)[min(s[0], len(CorrelationLayer) - 1)])
                for s in subseq
            ]
            is_sec = CorrelationLayer.TIMING in layers and CorrelationLayer.PROTOCOL in layers
            impact = 50 + (len(unique_chains) * 5) + (20 if is_sec else 0)

            # Build MotifInstances
            instances = [
                MotifInstance(
                    chain_id=cid,
                    layers_spanned=list(set(layers)),
                )
                for cid in unique_chains
            ]

            confidence = min(1.0, math.sqrt(len(unique_chains) / max(len(chains), 1)))

            motif = PropagationMotif(
                target_id=target_id,
                motif_kind=kind,
                occurrence_count=len(unique_chains),
                instance_chain_ids=unique_chains,
                instances=instances,
                participating_layers=list(set(layers)),
                dominant_mechanism=mech,
                confidence=confidence,
                description=self._describe_motif(kind, layers, unique_chains),
                is_security_relevant=is_sec,
                security_impact=(
                    "Timing leakage via protocol boundary" if is_sec else ""
                ),
                impact_score=min(100, impact),
            )
            catalog.add_motif(motif)

        # Sort motifs by impact_score descending
        catalog.motifs.sort(key=lambda m: m.impact_score, reverse=True)

        self._log.debug("motifs_mined", total=catalog.total_motifs, chains=len(chains))
        return catalog

    @staticmethod
    def _layer_ord(layer: CorrelationLayer) -> int:
        order = list(CorrelationLayer)
        try:
            return order.index(layer)
        except ValueError:
            return len(order) - 1

    @staticmethod
    def _mech_ord(mech: CausalMechanism) -> int:
        order = list(CausalMechanism)
        try:
            return order.index(mech)
        except ValueError:
            return len(order) - 1

    @staticmethod
    def _classify_motif(subseq: tuple) -> MotifKind:
        layers = [s[0] for s in subseq]
        mechs  = [s[1] for s in subseq]

        timing_ord = PropagationMotifMiner._layer_ord(CorrelationLayer.TIMING)
        PropagationMotifMiner._layer_ord(CorrelationLayer.PROTOCOL)
        oracle_ord = PropagationMotifMiner._mech_ord(CausalMechanism.TIMING_ORACLE)
        trust_ord = PropagationMotifMiner._mech_ord(CausalMechanism.TRUST_DELEGATION)
        PropagationMotifMiner._layer_ord(CorrelationLayer.PROCESS)

        if timing_ord in layers and oracle_ord in mechs:
            return MotifKind.TIMING_LEAKAGE
        if trust_ord in mechs:
            return MotifKind.TRUST_BRIDGE
        if layers == sorted(set(layers)):
            return MotifKind.PRIVILEGE_ESCALATION
        if layers[0] == layers[-1] and len(set(layers)) < len(layers):
            return MotifKind.AMPLIFICATION_LOOP
        if max(layers) - min(layers) > 2:
            return MotifKind.LAYER_JUMP
        return MotifKind.LINEAR_CHAIN

    @staticmethod
    def _describe_motif(
        kind: MotifKind,
        layers: list[CorrelationLayer],
        chain_ids: list[str],
    ) -> str:
        layer_str = " → ".join(l.value for l in layers)
        return (
            f"{kind.value}: appears in {len(chain_ids)} chains, "
            f"spanning layers: {layer_str}."
        )


# ── SynthesisBuilder ───────────────────────────────────────────────────────────


class SynthesisBuilder:
    """
    Assembles the FinalSynthesis from the CausalGraph, MotifCatalog, and chains.

    Produces:
    - A ranked list of SynthesisKeyFindings
    - A propagation_model abstract description
    - A thesis paragraph
    - A limitations section
    - A reproducibility section
    """

    def __init__(self) -> None:
        self._log = logger.bind(component="SynthesisBuilder")

    def build(
        self,
        target_id: str,
        graph: CausalGraph,
        motif_catalog: MotifCatalog,
        chains: list[CausalChain],
    ) -> FinalSynthesis:
        # Key findings: top motifs by impact
        key_findings = self._build_key_findings(motif_catalog, chains)

        # Propagation model: dominant layer sequence
        prop_model = self._build_propagation_model(chains, graph)

        # Thesis
        thesis = self._build_thesis(key_findings, graph, chains)

        # Limitations
        limitations = self._build_limitations(graph, chains, motif_catalog)

        # Reproducibility
        repro = self._build_reproducibility(key_findings, motif_catalog)

        # Contributing phases
        contrib = self._contributing_phases(graph)

        confidence = self._thesis_confidence(key_findings, motif_catalog)
        is_sec = any(f.is_security_relevant for f in key_findings)

        synth = FinalSynthesis(
            target_id=target_id,
            thesis=thesis,
            key_findings=key_findings,
            propagation_model=prop_model,
            limitations=limitations,
            reproducibility=repro,
            contributing_phases=contrib,
            thesis_confidence=confidence,
            is_security_relevant=is_sec,
        )

        self._log.debug(
            "synthesis_built",
            key_findings=len(key_findings),
            confidence=round(confidence, 3),
            is_security_relevant=is_sec,
        )
        return synth

    def _build_key_findings(
        self,
        catalog: MotifCatalog,
        chains: list[CausalChain],
    ) -> list[SynthesisKeyFinding]:
        findings: list[SynthesisKeyFinding] = []
        top_motifs = sorted(
            catalog.motifs,
            key=lambda m: (m.is_security_relevant, m.occurrence_count, m.impact_score),
            reverse=True,
        )[:5]

        for rank, motif in enumerate(top_motifs, 1):
            motif_chains = [
                c for c in chains if c.chain_id in motif.instance_chain_ids
            ]
            evidence = (
                f"Observed in {motif.occurrence_count} scenarios across "
                f"{len(motif.participating_layers)} layers. "
                f"Confidence: {motif.confidence:.2f}. "
                f"Mechanism: {motif.dominant_mechanism.value}."
            )
            finding = SynthesisKeyFinding(
                rank=rank,
                motif_id=motif.motif_id,
                chain_ids=[c.chain_id for c in motif_chains[:3]],
                claim=self._motif_to_claim(motif),
                evidence=evidence,
                layers_involved=motif.participating_layers,
                is_security_relevant=motif.is_security_relevant,
                confidence=motif.confidence,
                impact_score=motif.impact_score,
            )
            findings.append(finding)

        # Add a finding for the best complete chain (if not already covered)
        complete_chains = [c for c in chains if c.status.value == "complete"]
        if complete_chains and not findings:
            best = max(complete_chains, key=lambda c: (c.distinct_layer_count, c.confidence))
            findings.append(SynthesisKeyFinding(
                rank=1,
                chain_ids=[best.chain_id],
                claim=f"Cross-layer chain spanning {best.distinct_layer_count} layers: {best.narrative[:100]}",
                evidence=f"Single end-to-end chain with confidence {best.confidence:.2f}.",
                layers_involved=best.layers_represented,
                is_security_relevant=best.is_security_relevant,
                confidence=best.confidence,
                impact_score=70,
            ))

        return findings

    @staticmethod
    def _motif_to_claim(motif: PropagationMotif) -> str:
        layer_str = " → ".join(l.value for l in motif.participating_layers[:4])
        return (
            f"{motif.motif_kind.value.replace('_', ' ').title()}: "
            f"propagation pattern {layer_str} "
            f"({'security-relevant' if motif.is_security_relevant else 'functional'}) "
            f"appears in {motif.occurrence_count} scenarios."
        )

    @staticmethod
    def _build_propagation_model(
        chains: list[CausalChain],
        graph: CausalGraph,
    ) -> str:
        if not chains:
            return "Insufficient data to build propagation model."
        # Find most representative chain (most layers, highest confidence)
        best = max(chains, key=lambda c: (c.distinct_layer_count, c.confidence))
        layer_seq = " → ".join(l.value for l in best.layers_represented)
        return (
            f"Dominant propagation pattern ({best.distinct_layer_count} layers): "
            f"{layer_seq}. "
            f"Cross-layer hops: {best.cross_layer_hops}. "
            f"Confidence: {best.confidence:.2f}."
        )

    @staticmethod
    def _build_thesis(
        findings: list[SynthesisKeyFinding],
        graph: CausalGraph,
        chains: list[CausalChain],
    ) -> str:
        n_chains = len(chains)
        n_layers = graph.distinct_layers
        n_findings = len(findings)
        timing_chains = sum(1 for c in chains if c.reaches_timing_channel)
        sec_findings = [f for f in findings if f.is_security_relevant]

        if not findings:
            return (
                f"Phase 8 analysis identified {n_chains} causal chains across "
                f"{n_layers} layers but produced insufficient evidence for a "
                f"cross-layer thesis. Additional artifact coverage required."
            )

        top = findings[0]
        sec_note = (
            f"The highest-impact pattern ({sec_findings[0].claim[:80]}) "
            f"is security-relevant with confidence {sec_findings[0].confidence:.2f}. "
            if sec_findings else ""
        )
        return (
            f"Across {n_chains} cross-layer causal chains spanning {n_layers} distinct "
            f"analysis layers, {n_findings} recurring propagation patterns were identified. "
            f"The dominant pattern — {top.claim[:120]} — "
            f"appears in {top.chain_ids[0][:8] if top.chain_ids else 'multiple'} chains "
            f"with confidence {top.confidence:.2f}. "
            f"{sec_note}"
            f"{timing_chains} chains reach timing-channel terminals, "
            f"indicating that internal state information may be recoverable "
            f"from observable execution variance."
        )

    @staticmethod
    def _build_limitations(
        graph: CausalGraph,
        chains: list[CausalChain],
        catalog: MotifCatalog,
    ) -> str:
        broken = sum(1 for c in chains if c.status.value == "broken")
        hyp = sum(1 for c in chains if c.status.value == "hypothetical")
        missing = []
        for layer in CorrelationLayer:
            if not graph.nodes_at_layer(layer):
                missing.append(layer.value)
        return (
            f"Limitations: {broken} chains are broken (cannot be connected), "
            f"{hyp} chains include hypothetical links. "
            f"Layers with no coverage: {', '.join(missing) or 'none'}. "
            f"Synthetic timing models were used (not live measurements); "
            f"results should be validated against actual execution."
        )

    @staticmethod
    def _build_reproducibility(
        findings: list[SynthesisKeyFinding],
        catalog: MotifCatalog,
    ) -> str:
        if not findings:
            return "No findings to reproduce."
        motif_ids = [f.motif_id for f in findings[:3] if f.motif_id]
        return (
            f"To reproduce key findings: re-run Phase 6 boundary stress scenarios "
            f"associated with motifs {', '.join(m[:8] for m in motif_ids)}, "
            f"then re-run Phase 7 timing measurements with tighter isolation "
            f"(QUIESCE_SYSTEM or PERF_EVENT_GUARD) to confirm distinguishability."
        )

    @staticmethod
    def _contributing_phases(graph: CausalGraph) -> list[int]:
        phases = []
        layer_phase = {
            CorrelationLayer.STATIC: 3,
            CorrelationLayer.CONTROL_FLOW: 4,
            CorrelationLayer.PROCESS: 5,
            CorrelationLayer.PROTOCOL: 6,
            CorrelationLayer.TIMING: 7,
        }
        for layer, phase in layer_phase.items():
            if graph.nodes_at_layer(layer):
                phases.append(phase)
        return sorted(set(phases))

    @staticmethod
    def _thesis_confidence(
        findings: list[SynthesisKeyFinding],
        catalog: MotifCatalog,
    ) -> float:
        if not findings:
            return 0.0
        weights = [f.confidence * (f.impact_score / 100) for f in findings]
        return round(sum(weights) / len(weights), 3) if weights else 0.0
