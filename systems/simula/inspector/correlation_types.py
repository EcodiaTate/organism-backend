"""
EcodiaOS — Inspector Phase 8: Cross-Layer Correlation Types

All domain models for cross-layer cause → propagation → effect analysis.

Design philosophy
-----------------
Phase 8 answers the question: "how does influence travel across the full
analysis stack, from an initial input through all measured layers, and what
does it leak at the physical execution level?"

It takes findings from Phases 3–7 and builds a unified, navigable graph that
links:
  - Input instances (what was sent / observed)
  - Protocol state paths (Phase 6 FSM traces)
  - Process interactions (Phase 5 trust corridor traversals)
  - Control-flow regions (Phase 4 steerable branches)
  - Timing signatures (Phase 7 distinguishable channels)

Each hop in this graph is a CausalEdge with a typed CausalMechanism and a
confidence weight.  The graph is then mined for recurring PropagationMotifs —
structural patterns that appear across multiple scenarios — and ranked by
cross-layer impact.

The final deliverable is a Phase8Result containing:
  - A CausalGraph (directed, typed, multi-layer)
  - A MotifCatalog (recurring patterns with confidence)
  - A list of CausalChain (end-to-end stories per scenario)
  - A FinalSynthesis (thesis-ready narrative)

Layer map
---------
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Phase 3 (Phase3Result — static CFG, fragments)                          │
  │  Phase 4 (Phase4Result — steerable regions, condition sets)              │
  │  Phase 5 (Phase5Result — trust corridors, foothold bindings)             │
  │  Phase 6 (Phase6Result — FSM boundary failures, scenarios)               │
  │  Phase 7 (Phase7Result — distinguishable channels, signatures)           │
  │    ↓  CausalGraphBuilder                                                 │
  │  CausalGraph — nodes (EventNode) + edges (CausalEdge)                   │
  │    ↓  PropagationMotifMiner                                              │
  │  MotifCatalog — recurring structural patterns across scenarios           │
  │    ↓  CausalChainAssembler                                               │
  │  CausalChain — end-to-end story for one scenario                        │
  │    ↓  SynthesisBuilder                                                   │
  │  FinalSynthesis — thesis-ready summary                                  │
  │    ↓                                                                     │
  │  Phase8Result — top-level output with exit criterion                    │
  └──────────────────────────────────────────────────────────────────────────┘

Exit criterion
--------------
Phase8Result.exit_criterion_met = True when:
- ≥1 CausalChain spans ≥3 distinct layers AND
- ≥1 PropagationMotif appears in ≥2 scenarios AND
- FinalSynthesis.thesis is non-empty (coherent narrative produced).

Key concepts
------------
CorrelationLayer     — the analysis layer a node belongs to
CausalMechanism      — the type of edge (how one event causes the next)
CausalEdge           — a typed, weighted directed edge between EventNodes
EventNode            — a node in the causal graph, anchored to a Phase artifact
CausalChain          — an ordered sequence of EventNodes forming a scenario story
PropagationMotif     — a structural sub-graph pattern that recurs ≥2 times
FinalSynthesis       — thesis-ready narrative + ranked findings
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ── Enumerations ───────────────────────────────────────────────────────────────


class CorrelationLayer(enum.StrEnum):
    """
    Which analysis layer an EventNode belongs to.

    INPUT            — the initial input or observation that started the scenario
    CONTROL_FLOW     — Phase 4 control-flow region / steerable branch
    PROCESS          — Phase 5 service interaction / trust edge traversal
    PROTOCOL         — Phase 6 FSM state / protocol boundary
    TIMING           — Phase 7 timing channel observation
    STATIC           — Phase 3 static fragment / CFG node
    UNKNOWN          — unclassified
    """

    INPUT        = "input"
    CONTROL_FLOW = "control_flow"
    PROCESS      = "process"
    PROTOCOL     = "protocol"
    TIMING       = "timing"
    STATIC       = "static"
    UNKNOWN      = "unknown"


class CausalMechanism(enum.StrEnum):
    """
    The typed mechanism by which one event causes the next.

    DATA_DEPENDENCY      — output value of A is input to B
    CONTROL_DEPENDENCY   — branch outcome at A determines whether B executes
    TRUST_DELEGATION     — A's trust relationship enables B
    PROTOCOL_TRANSITION  — A fires a protocol state transition leading to B
    TIMING_COVARIATION   — A's execution time correlates with B's timing channel
    RESOURCE_CONTENTION  — A's resource usage causes B to behave differently
    STATE_POLLUTION      — A's side effect on shared state influences B
    TIMING_ORACLE        — B's latency reveals information about A's internal value
    AMPLIFICATION        — A's repeated execution amplifies B's observable effect
    UNKNOWN              — mechanism not identified
    """

    DATA_DEPENDENCY     = "data_dependency"
    CONTROL_DEPENDENCY  = "control_dependency"
    TRUST_DELEGATION    = "trust_delegation"
    PROTOCOL_TRANSITION = "protocol_transition"
    TIMING_COVARIATION  = "timing_covariation"
    RESOURCE_CONTENTION = "resource_contention"
    STATE_POLLUTION     = "state_pollution"
    TIMING_ORACLE       = "timing_oracle"
    AMPLIFICATION       = "amplification"
    UNKNOWN             = "unknown"


class ChainStatus(enum.StrEnum):
    """
    Completeness status of a CausalChain.

    COMPLETE    — all layers represented; chain is end-to-end
    PARTIAL     — some layers missing; chain has gaps
    BROKEN      — chain cannot be connected across a layer gap
    HYPOTHETICAL — chain includes one or more inferred / unobserved links
    """

    COMPLETE     = "complete"
    PARTIAL      = "partial"
    BROKEN       = "broken"
    HYPOTHETICAL = "hypothetical"


class MotifKind(enum.StrEnum):
    """
    Structural category of a PropagationMotif.

    LINEAR_CHAIN     — A → B → C, each hop same mechanism (rare)
    BRANCH_MERGE     — one cause fans out to multiple effects, then converges
    LAYER_JUMP       — mechanism skips one or more intermediate layers
    AMPLIFICATION_LOOP — effect feeds back to amplify its own cause
    PRIVILEGE_ESCALATION — monotonically increasing sensitivity/privilege along path
    TIMING_LEAKAGE   — path terminates at a timing channel disclosing internal state
    TRUST_BRIDGE     — path crosses a trust boundary at a single delegation edge
    MIXED            — more than one of the above patterns
    """

    LINEAR_CHAIN          = "linear_chain"
    BRANCH_MERGE          = "branch_merge"
    LAYER_JUMP            = "layer_jump"
    AMPLIFICATION_LOOP    = "amplification_loop"
    PRIVILEGE_ESCALATION  = "privilege_escalation"
    TIMING_LEAKAGE        = "timing_leakage"
    TRUST_BRIDGE          = "trust_bridge"
    MIXED                 = "mixed"


# ── Event node ─────────────────────────────────────────────────────────────────


class CorrelationID(EOSBaseModel):
    """
    Unified correlation identifier that anchors an EventNode to its source.

    Each artifact produced by Phases 3–7 has a unique ID.  A CorrelationID
    stores that ID alongside the layer and a human-readable label.
    """

    layer: CorrelationLayer
    artifact_id: str = Field(
        ...,
        description="The ID of the Phase artifact this node represents",
    )
    artifact_kind: str = Field(
        ...,
        description=(
            "e.g. 'Phase4.ControlFlowRegion', 'Phase5.ExpansionCorridor', "
            "'Phase6.BoundaryFailure', 'Phase7.DistinguishabilityResult'"
        ),
    )
    label: str = Field(
        ...,
        description="Short human-readable label for graph visualisation",
    )


class EventNode(EOSBaseModel):
    """
    A node in the CausalGraph representing a meaningful event at one layer.

    Nodes are the anchors for cross-layer correlation.  Each node wraps exactly
    one artifact from a previous phase.

    The correlation_id is the unified cross-layer key: every edge in the graph
    connects two CorrelationIDs, enabling traversal across layers.
    """

    node_id: str = Field(default_factory=new_id)

    # Source artifact
    correlation_id: CorrelationID

    # Position in analysis stack
    layer: CorrelationLayer

    # Sensitivity / privilege value (0–100); higher = more attractive target
    sensitivity: int = Field(default=50, ge=0, le=100)

    # Whether this node is the root cause of a chain
    is_root: bool = False

    # Whether this node is the terminal observable effect
    is_terminal: bool = False

    # Whether this node is a security-relevant finding
    is_security_relevant: bool = False

    # Human-readable description
    description: str = Field(default="")

    # Evidence quality: how confident we are that this node is real
    evidence_quality: float = Field(default=0.7, ge=0.0, le=1.0)

    # Outgoing edge IDs (populated by CausalGraph)
    outgoing_edge_ids: list[str] = Field(default_factory=list)

    # Incoming edge IDs (populated by CausalGraph)
    incoming_edge_ids: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


class CausalEdge(EOSBaseModel):
    """
    A directed, typed edge in the CausalGraph.

    An edge represents the *mechanism* by which one event at one layer
    causes or enables another event at the same or different layer.

    The weight encodes confidence that the causal link is real.
    """

    edge_id: str = Field(default_factory=new_id)

    from_node_id: str
    to_node_id: str

    # How the cause produces the effect
    mechanism: CausalMechanism

    # Whether this edge crosses a layer boundary
    is_cross_layer: bool = False

    # Layer pair (if cross-layer)
    from_layer: CorrelationLayer = CorrelationLayer.UNKNOWN
    to_layer: CorrelationLayer = CorrelationLayer.UNKNOWN

    # Confidence that this causal link is real (not coincidental)
    weight: float = Field(default=0.7, ge=0.0, le=1.0)

    # Human-readable description of the causal link
    description: str = Field(
        default="",
        description=(
            "e.g. 'Sequence counter overflow at ESTABLISHED state causes "
            "parser to accept out-of-window segment, enabling trust edge "
            "traversal to high-privilege session.'"
        ),
    )

    # Supporting evidence (artifact IDs from source phases)
    evidence_ids: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Causal graph ───────────────────────────────────────────────────────────────


class CausalGraph(EOSBaseModel):
    """
    A directed graph of EventNodes connected by CausalEdges.

    This is the primary Phase 8 data structure.  All cross-layer correlations
    are expressed as paths in this graph.

    The graph is built bottom-up by CausalGraphBuilder from Phase 4/5/6/7
    artifacts, then mined for chains and motifs.
    """

    graph_id: str = Field(default_factory=new_id)
    target_id: str

    # Node and edge stores
    nodes: dict[str, EventNode] = Field(
        default_factory=dict, description="node_id → EventNode"
    )
    edges: dict[str, CausalEdge] = Field(
        default_factory=dict, description="edge_id → CausalEdge"
    )

    # Adjacency indices
    out_edges: dict[str, list[str]] = Field(
        default_factory=dict, description="node_id → list[edge_id]"
    )
    in_edges: dict[str, list[str]] = Field(
        default_factory=dict, description="node_id → list[edge_id]"
    )

    # Quick lookup
    nodes_by_layer: dict[str, list[str]] = Field(
        default_factory=dict, description="CorrelationLayer.value → list[node_id]"
    )
    root_node_ids: list[str] = Field(default_factory=list)
    terminal_node_ids: list[str] = Field(default_factory=list)
    cross_layer_edge_ids: list[str] = Field(default_factory=list)

    # Statistics
    total_nodes: int = 0
    total_edges: int = 0
    total_cross_layer_edges: int = 0
    distinct_layers: int = 0

    built_at: datetime = Field(default_factory=utc_now)

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def add_node(self, node: EventNode) -> None:
        self.nodes[node.node_id] = node
        self.total_nodes = len(self.nodes)
        self.nodes_by_layer.setdefault(node.layer.value, []).append(node.node_id)
        self.distinct_layers = len(self.nodes_by_layer)
        if node.is_root and node.node_id not in self.root_node_ids:
            self.root_node_ids.append(node.node_id)
        if node.is_terminal and node.node_id not in self.terminal_node_ids:
            self.terminal_node_ids.append(node.node_id)

    def add_edge(self, edge: CausalEdge) -> None:
        self.edges[edge.edge_id] = edge
        self.total_edges = len(self.edges)
        self.out_edges.setdefault(edge.from_node_id, []).append(edge.edge_id)
        self.in_edges.setdefault(edge.to_node_id, []).append(edge.edge_id)

        # Update node adjacency lists
        from_node = self.nodes.get(edge.from_node_id)
        if from_node and edge.edge_id not in from_node.outgoing_edge_ids:
            from_node.outgoing_edge_ids.append(edge.edge_id)
        to_node = self.nodes.get(edge.to_node_id)
        if to_node and edge.edge_id not in to_node.incoming_edge_ids:
            to_node.incoming_edge_ids.append(edge.edge_id)

        if edge.is_cross_layer:
            self.cross_layer_edge_ids.append(edge.edge_id)
            self.total_cross_layer_edges = len(self.cross_layer_edge_ids)

    # ── Query helpers ─────────────────────────────────────────────────────────

    def successors(self, node_id: str) -> list[tuple[CausalEdge, EventNode]]:
        """Return (edge, destination_node) pairs from node_id."""
        result = []
        for eid in self.out_edges.get(node_id, []):
            e = self.edges.get(eid)
            if e and e.to_node_id in self.nodes:
                result.append((e, self.nodes[e.to_node_id]))
        return result

    def predecessors(self, node_id: str) -> list[tuple[CausalEdge, EventNode]]:
        """Return (edge, source_node) pairs arriving at node_id."""
        result = []
        for eid in self.in_edges.get(node_id, []):
            e = self.edges.get(eid)
            if e and e.from_node_id in self.nodes:
                result.append((e, self.nodes[e.from_node_id]))
        return result

    def nodes_at_layer(self, layer: CorrelationLayer) -> list[EventNode]:
        return [
            self.nodes[nid]
            for nid in self.nodes_by_layer.get(layer.value, [])
            if nid in self.nodes
        ]


# ── Causal chain ───────────────────────────────────────────────────────────────


class ChainStep(EOSBaseModel):
    """A single step in a CausalChain."""

    step_index: int
    node_id: str
    node_label: str
    layer: CorrelationLayer
    edge_id: str = Field(default="", description="Edge arriving at this step (empty for root)")
    mechanism: CausalMechanism = CausalMechanism.UNKNOWN
    description: str = Field(default="")
    is_cross_layer_step: bool = False
    cumulative_confidence: float = 1.0


class CausalChain(EOSBaseModel):
    """
    An ordered sequence of EventNodes forming an end-to-end scenario story.

    A CausalChain answers: "given input X, trace the complete path through
    control-flow decisions → service interactions → protocol state transitions
    → timing channel observations."

    Chains are the primary human-readable output of Phase 8.
    """

    chain_id: str = Field(default_factory=new_id)
    target_id: str

    # Source scenario (Phase 6 StressScenario that initiated this chain)
    source_scenario_id: str = Field(default="")

    # The ordered steps
    steps: list[ChainStep] = Field(default_factory=list)

    # Layers represented (derived from steps)
    layers_represented: list[CorrelationLayer] = Field(default_factory=list)
    distinct_layer_count: int = 0

    # Cross-layer hops count
    cross_layer_hops: int = 0

    # Overall chain status
    status: ChainStatus = ChainStatus.PARTIAL

    # End-to-end confidence (product of edge weights along the path)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Whether this chain reaches a timing-channel terminal
    reaches_timing_channel: bool = False

    # Whether this chain is security-relevant
    is_security_relevant: bool = False

    # Human-readable narrative
    narrative: str = Field(
        default="",
        description=(
            "End-to-end story: 'Input containing boundary sequence number "
            "drives control-flow to FSM boundary state WRAPPING, enabling "
            "trust corridor traversal to session-management service, resulting "
            "in a distinguishable timing channel at HMAC verification.'"
        ),
    )

    assembled_at: datetime = Field(default_factory=utc_now)


# ── Propagation motif ──────────────────────────────────────────────────────────


class MotifInstance(EOSBaseModel):
    """
    A single occurrence of a PropagationMotif in a specific CausalChain.
    """

    instance_id: str = Field(default_factory=new_id)
    chain_id: str
    # Node IDs within the chain that form the motif pattern
    node_ids: list[str] = Field(default_factory=list)
    # Edge IDs within the chain that form the motif pattern
    edge_ids: list[str] = Field(default_factory=list)
    # The layers this instance spans
    layers_spanned: list[CorrelationLayer] = Field(default_factory=list)


class PropagationMotif(EOSBaseModel):
    """
    A recurring structural sub-graph pattern that appears in ≥2 CausalChains.

    Motifs are the core discovery of Phase 8: if the same structural pattern
    appears repeatedly across independent scenarios, it is evidence of a
    *systematic* propagation mechanism — not a one-off coincidence.
    """

    motif_id: str = Field(default_factory=new_id)
    target_id: str

    # Classification
    motif_kind: MotifKind

    # Number of chains this motif appears in
    occurrence_count: int = 0
    instance_chain_ids: list[str] = Field(default_factory=list)
    instances: list[MotifInstance] = Field(default_factory=list)

    # Structural description
    participating_layers: list[CorrelationLayer] = Field(default_factory=list)
    dominant_mechanism: CausalMechanism = CausalMechanism.UNKNOWN

    # Confidence that this motif is real (not a coincidence of small sample)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Human-readable description
    description: str = Field(
        default="",
        description=(
            "e.g. 'TIMING_LEAKAGE: In 4/7 scenarios, a boundary counter overflow "
            "at the protocol layer produces a timing-distinguishable path at the "
            "crypto layer, leaking internal counter state to a network observer.'"
        ),
    )

    # Impact assessment
    is_security_relevant: bool = False
    security_impact: str = Field(default="")

    # Estimated impact score (0–100)
    impact_score: int = Field(default=50, ge=0, le=100)

    discovered_at: datetime = Field(default_factory=utc_now)


class MotifCatalog(EOSBaseModel):
    """
    A structured catalog of all PropagationMotifs found during Phase 8.
    """

    catalog_id: str = Field(default_factory=new_id)
    target_id: str

    motifs: list[PropagationMotif] = Field(default_factory=list)

    total_motifs: int = 0
    security_relevant_motifs: int = 0

    # Index by MotifKind
    motifs_by_kind: dict[str, list[str]] = Field(
        default_factory=dict, description="MotifKind.value → list[motif_id]"
    )

    # Chain → motif mapping for fast lookup
    motifs_by_chain: dict[str, list[str]] = Field(
        default_factory=dict, description="chain_id → list[motif_id]"
    )

    built_at: datetime = Field(default_factory=utc_now)

    def add_motif(self, motif: PropagationMotif) -> None:
        self.motifs.append(motif)
        self.total_motifs = len(self.motifs)
        if motif.is_security_relevant:
            self.security_relevant_motifs += 1
        self.motifs_by_kind.setdefault(motif.motif_kind.value, []).append(motif.motif_id)
        for chain_id in motif.instance_chain_ids:
            self.motifs_by_chain.setdefault(chain_id, []).append(motif.motif_id)


# ── Final synthesis ────────────────────────────────────────────────────────────


class SynthesisKeyFinding(EOSBaseModel):
    """
    One ranked finding in the FinalSynthesis.

    A key finding links a specific propagation motif (or unique chain) to a
    concrete claim about system behaviour: what consistently causes influence
    shifts, how it spreads, and what it leaks.
    """

    finding_id: str = Field(default_factory=new_id)
    rank: int = Field(..., description="1 = highest impact")

    # The motif or chain this finding describes
    motif_id: str = Field(default="")
    chain_ids: list[str] = Field(default_factory=list)

    # Claim
    claim: str = Field(
        ...,
        description=(
            "One sentence: 'Counter overflow at the TCP sequence number boundary "
            "enables trust-corridor traversal that results in a timing-distinguishable "
            "authentication decision, leaking boundary state to a network observer.'"
        ),
    )

    # Supporting evidence
    evidence: str = Field(
        default="",
        description="Paragraph: which scenarios, which layers, which statistical tests.",
    )

    # Layers involved
    layers_involved: list[CorrelationLayer] = Field(default_factory=list)

    # Impact
    is_security_relevant: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    impact_score: int = Field(default=50, ge=0, le=100)


class FinalSynthesis(EOSBaseModel):
    """
    Thesis-ready synthesis of all Phase 8 cross-layer correlation findings.

    The FinalSynthesis is the human-facing output of the full Phase 1–8 pipeline:
    it answers in plain language what consistently causes influence shifts, how
    that influence propagates, and what it ultimately leaks.

    Structure
    ---------
    thesis          — one-paragraph top-level claim (reproducible, cross-layer supported)
    key_findings    — ranked list of specific sub-claims with evidence
    propagation_model — abstract description of the dominant propagation pattern
    limitations     — known gaps, noise effects, unobserved layers
    reproducibility — how to reproduce the strongest findings
    """

    synthesis_id: str = Field(default_factory=new_id)
    target_id: str

    # Top-level thesis
    thesis: str = Field(
        default="",
        description=(
            "Paragraph-length claim supported by cross-layer evidence: "
            "'Sensitive protocol boundary states consistently produce "
            "observable timing signatures that propagate through trust "
            "corridors to authentication decisions, enabling an adversary "
            "to reconstruct internal protocol state from network timing alone.'"
        ),
    )

    # Ranked key findings
    key_findings: list[SynthesisKeyFinding] = Field(default_factory=list)

    # Abstract propagation model
    propagation_model: str = Field(
        default="",
        description=(
            "Abstract description of the dominant pattern: "
            "'Input → Boundary Counter Overflow → FSM Layer Desync → "
            "Trust Edge Traversal → Auth-Decision Timing Oracle → "
            "Distinguishable Channel'"
        ),
    )

    # Limitations and open questions
    limitations: str = Field(default="")

    # Reproducibility guidance
    reproducibility: str = Field(
        default="",
        description=(
            "Which Phase 6 scenarios + Phase 7 operations need to be re-run "
            "to reproduce the key findings."
        ),
    )

    # Coverage: which phases contributed findings
    contributing_phases: list[int] = Field(default_factory=list)

    # Confidence in the overall thesis
    thesis_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Whether the thesis is security-relevant
    is_security_relevant: bool = False

    produced_at: datetime = Field(default_factory=utc_now)


# ── Phase 8 result ─────────────────────────────────────────────────────────────


class Phase8Result(EOSBaseModel):
    """
    Top-level output of a Phase 8 cross-layer correlation session.

    Wraps the CausalGraph, MotifCatalog, CausalChains, and FinalSynthesis,
    with aggregate statistics and the exit criterion flag.

    Exit criterion
    --------------
    exit_criterion_met = True when:
    - ≥1 CausalChain spans ≥3 distinct layers (cross-layer story), AND
    - ≥1 PropagationMotif with occurrence_count ≥ 2 (recurring pattern), AND
    - FinalSynthesis.thesis is non-empty (coherent narrative produced).
    """

    id: str = Field(default_factory=new_id)
    target_id: str

    # Core artifacts
    causal_graph: CausalGraph
    motif_catalog: MotifCatalog
    causal_chains: list[CausalChain] = Field(default_factory=list)
    synthesis: FinalSynthesis

    # Aggregate statistics
    total_nodes: int = 0
    total_edges: int = 0
    total_cross_layer_edges: int = 0
    total_chains: int = 0
    total_motifs: int = 0
    security_relevant_motifs: int = 0
    chains_spanning_3plus_layers: int = 0
    chains_reaching_timing_channel: int = 0
    chains_security_relevant: int = 0

    # Layer coverage
    distinct_layers_in_graph: int = 0
    distinct_layers_in_chains: int = 0

    # Exit criterion
    exit_criterion_met: bool = Field(
        default=False,
        description=(
            "True when: ≥1 chain spans ≥3 layers, ≥1 motif appears in ≥2 "
            "scenarios, and synthesis.thesis is non-empty."
        ),
    )

    produced_at: datetime = Field(default_factory=utc_now)
