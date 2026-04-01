"""
EcodiaOS - Inspector Phase 5: Trust-Graph Influence Expansion Types

All domain models for post-boundary propagation analysis.

Design philosophy
-----------------
Phase 5 answers the question: "once external influence reaches some node,
where can it propagate through existing trust/identity relationships?"

It models the system as a directed graph of principals, services, resources,
roles, credentials, and sessions, with typed trust edges.  Given a foothold
(initial influenced node), the engine computes which other nodes become
reachable and what privilege gradient each path represents.

Layer map
---------
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Phase 4 (Phase4Result, SteerabilityModel, ConditionSet)                  │
  │    ↓  TrustGraphBuilder                                                  │
  │  TrustGraph   - principals, services, resources, roles, credentials,     │
  │                 sessions + typed trust edges                              │
  │    ↓  PropagationEngine                                                  │
  │  PropagationResult - per-foothold reachability + privilege gradient       │
  │    ↓                                                                     │
  │  ExpansionCorridor - ranked propagation pathway (graph finding)           │
  │    ↓                                                                     │
  │  Phase5Result  - top-level output with exit criterion                    │
  └──────────────────────────────────────────────────────────────────────────┘

Exit criterion
--------------
Phase5Result.exit_criterion_met = True when, given an initial influenced node,
the engine has produced a defensible set of propagation pathways and explained
the trust edges enabling each one.

Key concepts
------------
TrustEdgeKind   - the mechanism by which trust is transferred between nodes
PrivilegeValue  - a numeric weight (0–100) assigned to each node; higher = more
                  valuable/sensitive; paths must monotonically increase to qualify
                  as privilege-gradient corridors
ExpansionCorridor - a ranked pathway that starts from a foothold node and
                  monotonically traverses nodes of increasing privilege value
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ── Enumerations ───────────────────────────────────────────────────────────────


class TrustNodeKind(enum.StrEnum):
    """
    Semantic category of a node in the trust graph.

    PRINCIPAL   - a human user, service account, or bot identity
    SERVICE     - a process, microservice, or daemon (can act + be delegated to)
    RESOURCE    - a data store, file, secret, key, config item - things that hold value
    ROLE        - a named permission bundle (IAM role, RBAC role, sudo group)
    CREDENTIAL  - a token, certificate, API key, password hash, session cookie
    SESSION     - an authenticated runtime context binding a principal to capabilities
    """

    PRINCIPAL  = "principal"
    SERVICE    = "service"
    RESOURCE   = "resource"
    ROLE       = "role"
    CREDENTIAL = "credential"
    SESSION    = "session"
    UNKNOWN    = "unknown"


class TrustEdgeKind(enum.StrEnum):
    """
    The mechanism by which trust transfers along a directed edge A → B.

    AUTHENTICATION      - A proves identity to B (credential presented + accepted)
    DELEGATION          - A explicitly grants a subset of its capabilities to B
    ASSUMED_TRUST       - B assumes A is trustworthy without explicit proof
                          (e.g., same-host loopback, private subnet co-tenancy)
    IMPLICIT_PERMISSION - B inherits A's capabilities by default configuration
                          (e.g., sudo NOPASSWD, IAM wildcard, inherited role)
    CREDENTIAL_REUSE    - A's credential is stored/forwarded and reused by B
    SERVICE_ASSUMPTION  - B is configured to trust all calls from A's service
                          (e.g., mTLS peer bypass, allow-all network policy)
    SHARED_SECRET       - A and B share a secret that acts as bilateral trust anchor
    PRIVILEGE_GRANT     - A explicitly grants B elevated privilege (GRANT SQL, chmod)
    INHERITANCE         - B is a child process / sub-service that inherits A's context
    LATERAL_MOVEMENT    - forensic/inferred edge: A was observed pivoting to B
    """

    AUTHENTICATION      = "authentication"
    DELEGATION          = "delegation"
    ASSUMED_TRUST       = "assumed_trust"
    IMPLICIT_PERMISSION = "implicit_permission"
    CREDENTIAL_REUSE    = "credential_reuse"
    SERVICE_ASSUMPTION  = "service_assumption"
    SHARED_SECRET       = "shared_secret"
    PRIVILEGE_GRANT     = "privilege_grant"
    INHERITANCE         = "inheritance"
    LATERAL_MOVEMENT    = "lateral_movement"
    UNKNOWN             = "unknown"


class TrustStrength(enum.StrEnum):
    """
    How robust the trust relationship is - inversely proportional to how easy
    it is to exploit the edge.

    EXPLICIT   - cryptographically enforced, mutual, logged (hard to abuse)
    VERIFIED   - checked at runtime but not cryptographic (e.g., HMAC, bearer token)
    IMPLICIT   - assumed, inferred, or based on network position (easy to abuse)
    BLIND      - no verification at all; one side fully trusts the other
    """

    EXPLICIT = "explicit"
    VERIFIED = "verified"
    IMPLICIT = "implicit"
    BLIND    = "blind"


class PrivilegeImpact(enum.StrEnum):
    """
    The consequence of an attacker reaching this node.

    CRITICAL  - full system / tenant compromise (root, admin, DBA superuser)
    HIGH      - significant capability gain (read all data, impersonate user)
    MEDIUM    - partial gain (read some data, limited write)
    LOW       - minimal gain (unauthenticated read-only public data)
    NONE      = no useful privilege at this node
    """

    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"
    NONE     = "none"


class CorridorRiskTier(enum.StrEnum):
    """
    Ranked risk tier for an ExpansionCorridor.

    CRITICAL - corridor terminates at a CRITICAL node and all edges are exploitable
    HIGH     - corridor terminates at HIGH node or has ≥1 BLIND/IMPLICIT edge
    MEDIUM   - corridor terminates at MEDIUM node
    LOW      - corridor terminates at LOW node or all edges are EXPLICIT
    """

    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"


# ── Trust graph nodes ─────────────────────────────────────────────────────────


class TrustNode(EOSBaseModel):
    """
    A single node in the trust graph.

    Nodes are inferred from Phase 3/4 artifacts (service names, object types,
    identity contexts, credential variables) and optionally from supplied
    topology data.
    """

    node_id: str = Field(default_factory=new_id)
    target_id: str

    kind: TrustNodeKind
    name: str = Field(..., description="Human-readable identifier, e.g. 'postgres-svc', 'admin-role'")

    # Optional: structural location
    service_name: str = Field(default="")
    file_path: str = Field(default="")

    # Privilege weight - higher = more valuable for an attacker (0–100)
    privilege_value: int = Field(
        default=0,
        ge=0,
        le=100,
        description=(
            "Attacker value of this node: 0 = no value, 100 = full system compromise. "
            "Assigned by TrustGraphBuilder based on kind + metadata."
        ),
    )

    # Impact if compromised
    privilege_impact: PrivilegeImpact = PrivilegeImpact.NONE

    # Source evidence
    derived_from_fragment_ids: list[str] = Field(default_factory=list)
    derived_from_variable_ids: list[str] = Field(default_factory=list)
    derived_from_state_variable_kinds: list[str] = Field(default_factory=list)

    # Whether this node was directly touched by Phase 4 steerability analysis
    steerability_adjacent: bool = False

    # Extra metadata (e.g. IAM ARN, DB connection string schema, role name)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Trust edges ───────────────────────────────────────────────────────────────


class TrustEdge(EOSBaseModel):
    """
    A directed trust relationship A → B.

    An edge represents a pathway along which influence can flow from A to B
    (not necessarily a data flow - trust is about who-can-act-as or
    who-is-assumed-equivalent-to).
    """

    edge_id: str = Field(default_factory=new_id)
    target_id: str

    from_node_id: str
    to_node_id: str

    kind: TrustEdgeKind
    strength: TrustStrength = TrustStrength.IMPLICIT

    description: str = Field(
        default="",
        description="One-sentence explanation of why this trust relationship exists",
    )

    # Formal evidence or source location
    evidence_location: str = Field(
        default="",
        description="File path + line, config key, policy rule, or protocol spec reference",
    )
    evidence_fragment_ids: list[str] = Field(default_factory=list)

    # Whether this edge requires special conditions to traverse
    conditional: bool = Field(
        default=False,
        description="True if the edge only exists under certain runtime conditions",
    )
    condition_description: str = Field(default="")

    # Whether this edge was inferred (vs. directly observed)
    inferred: bool = False

    # Confidence in this edge (0.0–1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Cost of traversal: 0 = free (unauthenticated), 1 = high barrier
    traversal_cost: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "How hard it is to traverse this edge for an attacker. "
            "0.0 = trivial (blind trust), 1.0 = near-impossible (MFA + audit)."
        ),
    )


# ── Trust graph ───────────────────────────────────────────────────────────────


class TrustGraph(EOSBaseModel):
    """
    The complete directed trust graph for one analysis target.

    Produced by TrustGraphBuilder.  Supports BFS/DFS reachability,
    privilege-gradient path queries, and corridor extraction.
    """

    id: str = Field(default_factory=new_id)
    target_id: str

    nodes: dict[str, TrustNode] = Field(default_factory=dict)
    edges: dict[str, TrustEdge] = Field(default_factory=dict)

    # Adjacency index: node_id → list of edge_ids going OUT from that node
    out_edges: dict[str, list[str]] = Field(default_factory=dict)
    # Adjacency index: node_id → list of edge_ids coming INTO that node
    in_edges: dict[str, list[str]] = Field(default_factory=dict)

    # Indices for efficient lookup
    nodes_by_kind: dict[str, list[str]] = Field(default_factory=dict)     # kind → node_ids
    nodes_by_name: dict[str, str] = Field(default_factory=dict)           # name → node_id

    total_nodes: int = 0
    total_edges: int = 0

    built_at: datetime = Field(default_factory=utc_now)

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def add_node(self, node: TrustNode) -> None:
        self.nodes[node.node_id] = node
        self.total_nodes = len(self.nodes)

        kind_key = node.kind.value
        if kind_key not in self.nodes_by_kind:
            self.nodes_by_kind[kind_key] = []
        if node.node_id not in self.nodes_by_kind[kind_key]:
            self.nodes_by_kind[kind_key].append(node.node_id)

        self.nodes_by_name[node.name] = node.node_id

    def add_edge(self, edge: TrustEdge) -> None:
        self.edges[edge.edge_id] = edge
        self.total_edges = len(self.edges)

        if edge.from_node_id not in self.out_edges:
            self.out_edges[edge.from_node_id] = []
        self.out_edges[edge.from_node_id].append(edge.edge_id)

        if edge.to_node_id not in self.in_edges:
            self.in_edges[edge.to_node_id] = []
        self.in_edges[edge.to_node_id].append(edge.edge_id)

    # ── Query helpers ─────────────────────────────────────────────────────────

    def successors(self, node_id: str) -> list[tuple[TrustEdge, TrustNode]]:
        """Return (edge, destination_node) pairs reachable in one hop."""
        result = []
        for eid in self.out_edges.get(node_id, []):
            edge = self.edges.get(eid)
            if edge and edge.to_node_id in self.nodes:
                result.append((edge, self.nodes[edge.to_node_id]))
        return result

    def predecessors(self, node_id: str) -> list[tuple[TrustEdge, TrustNode]]:
        """Return (edge, source_node) pairs that reach node_id in one hop."""
        result = []
        for eid in self.in_edges.get(node_id, []):
            edge = self.edges.get(eid)
            if edge and edge.from_node_id in self.nodes:
                result.append((edge, self.nodes[edge.from_node_id]))
        return result

    def nodes_of_kind(self, kind: TrustNodeKind) -> list[TrustNode]:
        """Return all nodes of a given kind."""
        ids = self.nodes_by_kind.get(kind.value, [])
        return [self.nodes[i] for i in ids if i in self.nodes]

    def high_value_nodes(self, threshold: int = 70) -> list[TrustNode]:
        """Return nodes whose privilege_value ≥ threshold."""
        return [n for n in self.nodes.values() if n.privilege_value >= threshold]


# ── Propagation path ──────────────────────────────────────────────────────────


class PropagationStep(EOSBaseModel):
    """
    A single hop in a propagation path.

    Records both the trust edge traversed and the node arrived at,
    so the complete sequence of nodes + edges is preserved.
    """

    step_index: int
    edge_id: str
    edge_kind: TrustEdgeKind
    edge_strength: TrustStrength
    edge_traversal_cost: float

    arrived_node_id: str
    arrived_node_name: str
    arrived_node_kind: TrustNodeKind
    arrived_privilege_value: int
    arrived_privilege_impact: PrivilegeImpact


class PropagationPath(EOSBaseModel):
    """
    A complete path from a foothold node to a destination node through
    the trust graph.

    Paths are produced by PropagationEngine BFS/DFS and filtered to those
    that monotonically increase privilege_value (gradient constraint).
    """

    path_id: str = Field(default_factory=new_id)
    target_id: str

    # Start
    foothold_node_id: str
    foothold_node_name: str

    # End
    terminal_node_id: str
    terminal_node_name: str
    terminal_privilege_value: int
    terminal_privilege_impact: PrivilegeImpact

    # Path sequence
    steps: list[PropagationStep] = Field(default_factory=list)
    node_ids: list[str] = Field(default_factory=list)  # [foothold, ..., terminal]
    edge_ids: list[str] = Field(default_factory=list)

    # Aggregate path properties
    path_length: int = 0
    total_traversal_cost: float = 0.0  # sum of edge traversal costs
    privilege_delta: int = 0           # terminal_value - foothold_value
    is_monotonic: bool = True          # all steps increase or maintain privilege

    # Weakest link - the edge with the lowest traversal cost (easiest to abuse)
    weakest_edge_id: str = Field(default="")
    weakest_edge_kind: TrustEdgeKind = TrustEdgeKind.UNKNOWN
    weakest_edge_cost: float = 1.0

    # Evidence
    evidence_fragment_ids: list[str] = Field(default_factory=list)

    # Confidence in this path (0.0–1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# ── Steerable region binding ──────────────────────────────────────────────────


class FootholdBinding(EOSBaseModel):
    """
    Binds a Phase 4 steerability finding to an initial trust-graph node.

    When a Phase 4 ConditionSet predicts that execution becomes influence-
    permissive, the FootholdBinding maps that steerable region to the node
    where influence enters the trust graph.
    """

    binding_id: str = Field(default_factory=new_id)
    target_id: str

    # Phase 4 references
    condition_set_id: str
    steerable_region_id: str
    steerability_class: str

    # Trust graph entry point
    foothold_node_id: str
    foothold_node_name: str
    foothold_node_kind: TrustNodeKind

    # Why this node was chosen as the foothold
    rationale: str = Field(
        default="",
        description="Why this trust-graph node corresponds to the Phase 4 steerable region",
    )

    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# ── Expansion corridor ────────────────────────────────────────────────────────


class ExpansionCorridor(EOSBaseModel):
    """
    A ranked propagation pathway from a foothold to a high-value destination.

    This is the primary Phase 5 deliverable: a graph-level finding that
    describes how influence spreads, which trust edges it exploits, and
    what privilege elevation results.

    Corridors are produced by PropagationEngine and ranked by risk.
    They are *findings*, not exploit templates - they describe what the
    graph topology makes possible.
    """

    corridor_id: str = Field(default_factory=new_id)
    target_id: str

    # Risk classification
    risk_tier: CorridorRiskTier = CorridorRiskTier.LOW

    # The propagation path (primary path; alternatives in alternative_paths)
    primary_path: PropagationPath

    # Alternative paths to the same terminal node (different routes)
    alternative_paths: list[PropagationPath] = Field(default_factory=list)

    # Binding to Phase 4 foothold (optional - not all footholds come from Phase 4)
    foothold_binding: FootholdBinding | None = None

    # ── Corridor summary ──────────────────────────────────────────────────────
    foothold_node_id: str
    foothold_node_name: str
    terminal_node_id: str
    terminal_node_name: str
    terminal_privilege_impact: PrivilegeImpact

    # Trust edge kinds exploited across all paths
    exploited_edge_kinds: list[TrustEdgeKind] = Field(default_factory=list)

    # Weakest edge across all paths (minimum traversal cost)
    weakest_edge_id: str = Field(default="")
    weakest_edge_cost: float = 1.0

    # Number of hops in the shortest path
    min_hops: int = 0

    # Privilege delta across the corridor
    max_privilege_delta: int = 0

    # ── Narrative ────────────────────────────────────────────────────────────
    description: str = Field(
        default="",
        description=(
            "One-paragraph prose description suitable for a research report. "
            "Format: 'Starting from [foothold], influence propagates via [edges] "
            "to reach [terminal], where [impact].'"
        ),
    )

    # Supporting evidence
    evidence_fragment_ids: list[str] = Field(default_factory=list)
    evidence_condition_set_ids: list[str] = Field(default_factory=list)

    derived_at: datetime = Field(default_factory=utc_now)


# ── Propagation simulation ────────────────────────────────────────────────────


class PropagationSimulation(EOSBaseModel):
    """
    The result of a single propagation simulation from one foothold node.

    Produced by PropagationEngine.simulate() - BFS from the foothold, tracking
    all reachable nodes within a configurable depth and the privilege gradient
    along each path.
    """

    simulation_id: str = Field(default_factory=new_id)
    target_id: str

    foothold_node_id: str
    foothold_node_name: str
    foothold_privilege_value: int

    # All reachable nodes (within depth limit)
    reachable_node_ids: list[str] = Field(default_factory=list)

    # Subset: nodes with higher privilege_value than foothold
    privilege_gain_node_ids: list[str] = Field(default_factory=list)

    # All paths found (monotonic gradient filter applied if strict_gradient=True)
    paths: list[PropagationPath] = Field(default_factory=list)

    # Corridors extracted from the paths (ranked by risk)
    corridors: list[ExpansionCorridor] = Field(default_factory=list)

    # Statistics
    total_reachable_nodes:   int = 0
    total_privilege_gains:   int = 0
    total_paths:             int = 0
    total_corridors:         int = 0
    max_privilege_reached:   int = 0
    max_privilege_delta:     int = 0

    # Depth limit used in this simulation
    max_depth: int = Field(default=5)

    simulated_at: datetime = Field(default_factory=utc_now)


# ── Reachability + privilege gradient map ─────────────────────────────────────


class ReachabilityMap(EOSBaseModel):
    """
    A complete reachability + privilege gradient map for one target.

    Aggregates all per-foothold simulations into a single view:
    - which nodes can be reached from each foothold
    - the privilege gradient along each path
    - the ranked set of expansion corridors

    This is one of the two top-level Phase 5 deliverables (alongside corridors).
    """

    map_id: str = Field(default_factory=new_id)
    target_id: str

    # Per-foothold simulations
    simulations: dict[str, PropagationSimulation] = Field(
        default_factory=dict,
        description="foothold_node_id → PropagationSimulation",
    )

    # All corridors ranked by risk (CRITICAL first)
    ranked_corridors: list[ExpansionCorridor] = Field(default_factory=list)

    # Aggregate statistics
    total_footholds:         int = 0
    total_reachable_nodes:   int = 0
    total_corridors:         int = 0
    critical_corridors:      int = 0
    high_corridors:          int = 0
    medium_corridors:        int = 0
    low_corridors:           int = 0

    # Highest privilege value reachable from any foothold
    global_max_privilege: int = 0

    built_at: datetime = Field(default_factory=utc_now)


# ── Phase 5 result ────────────────────────────────────────────────────────────


class Phase5Result(EOSBaseModel):
    """
    Top-level output of a Phase 5 trust-graph influence expansion session.

    Wraps the TrustGraph, ReachabilityMap, ranked ExpansionCorridors, and
    aggregate statistics suitable for logging and reporting.

    Exit criterion
    --------------
    exit_criterion_met = True when:
    - ≥1 foothold node has been identified, AND
    - ≥1 PropagationSimulation has produced ≥1 propagation path, AND
    - ≥1 ExpansionCorridor has been produced with a description explaining
      the trust edges enabling it.
    """

    id: str = Field(default_factory=new_id)
    target_id: str

    # Core artefacts
    trust_graph: TrustGraph
    reachability_map: ReachabilityMap

    # Foothold bindings (Phase 4 steerable regions → trust-graph nodes)
    foothold_bindings: list[FootholdBinding] = Field(default_factory=list)

    # Ranked corridors (deduplicated across all simulations)
    ranked_corridors: list[ExpansionCorridor] = Field(default_factory=list)

    # Aggregate statistics
    total_trust_nodes:       int = 0
    total_trust_edges:       int = 0
    total_foothold_bindings: int = 0
    total_simulations:       int = 0
    total_corridors:         int = 0
    critical_corridors:      int = 0
    high_corridors:          int = 0

    # Exit criterion
    exit_criterion_met: bool = Field(
        default=False,
        description=(
            "True when: ≥1 foothold identified, ≥1 propagation path produced, "
            "and ≥1 ExpansionCorridor describes trust edges enabling propagation."
        ),
    )

    produced_at: datetime = Field(default_factory=utc_now)
