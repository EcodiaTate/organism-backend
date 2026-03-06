"""
EcodiaOS — Inspector Phase 5: Trust Graph Builder

Constructs a TrustGraph from Phase 3/4 artifacts by running four inference passes:

  Pass 1 — Node extraction
    Scan Fragment catalog and Phase 4 StateVariables for node-class signals:
    IDENTITY_CONTEXT vars → PRINCIPAL nodes
    FUNCTION_POINTER vars adjacent to indirect dispatch → SERVICE nodes
    MEMORY_REGION vars near credential patterns → CREDENTIAL nodes
    PROTOCOL_STATE vars → SESSION nodes
    Failure-adjacent fragments containing auth/role patterns → ROLE nodes
    All other high-interest fragments → RESOURCE nodes

  Pass 2 — Structural edge inference
    Derive explicit edges from CFG call patterns and taint flows:
    Taint flows from CREDENTIAL nodes into SERVICE nodes → AUTHENTICATION
    Call chains from one SERVICE to another via shared credential → CREDENTIAL_REUSE
    Delegation patterns (capability-narrowing calls) → DELEGATION
    Inheritance edges from forked/child services → INHERITANCE
    PRIVILEGE_GRANT edges from ROLE nodes to PRINCIPAL/SERVICE nodes

  Pass 3 — Implicit / assumed-trust edges
    Infer soft trust from co-location, subnet assumptions, and defaults:
    Nodes sharing a service_name with no auth between them → ASSUMED_TRUST
    Loopback/localhost calls without TLS → ASSUMED_TRUST
    Wildcard permission patterns → IMPLICIT_PERMISSION
    Shared secret patterns in source → SHARED_SECRET

  Pass 4 — Privilege value assignment
    Score each node 0–100 based on kind, impact heuristics, fragment content,
    and Phase 4 steerability adjacency:
    CRITICAL roles / root credentials → 90–100
    Admin services / DB superusers → 70–89
    Authenticated sessions → 40–69
    Ordinary principals / resources → 10–39
    Unknown / public nodes → 0–9

Usage
-----
  builder = TrustGraphBuilder()
  graph = builder.build(
      target_id=...,
      phase3_result=phase3_result,
      phase4_result=phase4_result,   # optional but recommended
  )
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

from systems.simula.inspector.trust_types import (
    FootholdBinding,
    PrivilegeImpact,
    TrustEdge,
    TrustEdgeKind,
    TrustGraph,
    TrustNode,
    TrustNodeKind,
    TrustStrength,
)

if TYPE_CHECKING:
    from systems.simula.inspector.constraint_types import Phase4Result
    from systems.simula.inspector.static_types import (
        Phase3Result,
    )

logger = structlog.get_logger().bind(system="simula.inspector.trust_graph")


# ── Privilege scoring heuristics ──────────────────────────────────────────────

# Patterns in node names that indicate high privilege
_CRITICAL_NAME_PATTERNS = re.compile(
    r"\b(root|superuser|admin|dba|sudo|sysadmin|sa|godmode|master_key|"
    r"private_key|signing_key|root_token|service_account_token|cluster_admin)\b",
    re.IGNORECASE,
)
_HIGH_NAME_PATTERNS = re.compile(
    r"\b(admin|iam_role|owner|manager|privileged|elevated|write_access|"
    r"db_admin|api_key|access_token|client_secret|refresh_token|"
    r"jwt_secret|hmac_key|tls_cert|ca_cert)\b",
    re.IGNORECASE,
)
_MEDIUM_NAME_PATTERNS = re.compile(
    r"\b(session|token|auth|user|authenticated|logged_in|identity|"
    r"principal|credential|login|bearer)\b",
    re.IGNORECASE,
)

# Fragment description patterns signalling credential or auth behaviour
_CRED_FRAGMENT_PATTERN = re.compile(
    r"\b(password|secret|token|key|credential|auth|login|encrypt|decrypt|sign|verify)\b",
    re.IGNORECASE,
)
_ROLE_FRAGMENT_PATTERN = re.compile(
    r"\b(role|permission|grant|acl|policy|access_control|rbac|iam|entitlement|"
    r"authorize|authz|privilege|capability)\b",
    re.IGNORECASE,
)
_SERVICE_FRAGMENT_PATTERN = re.compile(
    r"\b(service|server|daemon|handler|worker|dispatch|listen|accept|"
    r"connect|socket|rpc|grpc|http|endpoint)\b",
    re.IGNORECASE,
)
_DELEGATION_PATTERN = re.compile(
    r"\b(delegate|impersonate|assume_role|sts|oidc|oauth|saml|"
    r"act_as|run_as|sudo|su |privilege_drop|setuid|seteuid)\b",
    re.IGNORECASE,
)
_SHARED_SECRET_PATTERN = re.compile(
    r"\b(shared_secret|psk|pre_shared|hmac|symmetric_key|aes_key|"
    r"internal_api_key|service_secret)\b",
    re.IGNORECASE,
)
_IMPLICIT_PERM_PATTERN = re.compile(
    r"\b(nopasswd|allow_all|\*|wildcard|bypass|skip_auth|no_verify|"
    r"trust_all|unconditional|unauthenticated)\b",
    re.IGNORECASE,
)


def _privilege_score_for_name(name: str) -> tuple[int, PrivilegeImpact]:
    """Heuristic privilege score based on node name patterns."""
    if _CRITICAL_NAME_PATTERNS.search(name):
        return 92, PrivilegeImpact.CRITICAL
    if _HIGH_NAME_PATTERNS.search(name):
        return 75, PrivilegeImpact.HIGH
    if _MEDIUM_NAME_PATTERNS.search(name):
        return 45, PrivilegeImpact.MEDIUM
    return 15, PrivilegeImpact.LOW


def _base_privilege_for_kind(kind: TrustNodeKind) -> int:
    """Base privilege score by node kind (before name heuristics)."""
    return {
        TrustNodeKind.ROLE:        60,
        TrustNodeKind.CREDENTIAL:  55,
        TrustNodeKind.PRINCIPAL:   40,
        TrustNodeKind.SESSION:     35,
        TrustNodeKind.SERVICE:     30,
        TrustNodeKind.RESOURCE:    20,
        TrustNodeKind.UNKNOWN:     5,
    }.get(kind, 5)


# ── Trust graph builder ───────────────────────────────────────────────────────


class TrustGraphBuilder:
    """
    Constructs a TrustGraph from Phase 3 + Phase 4 artifacts in four passes.
    """

    def __init__(self) -> None:
        self._log = logger

    def build(
        self,
        target_id: str,
        phase3_result: Phase3Result,
        phase4_result: Phase4Result | None = None,
    ) -> tuple[TrustGraph, list[FootholdBinding]]:
        """
        Build the TrustGraph and FootholdBindings for a target.

        Args:
            target_id: Target identifier.
            phase3_result: Phase 3 static analysis output.
            phase4_result: Optional Phase 4 steerability model output.

        Returns:
            (TrustGraph, list[FootholdBinding])
        """
        log = self._log.bind(target_id=target_id)
        log.info("trust_graph_build_started")

        graph = TrustGraph(target_id=target_id)

        # Collect phase-specific data
        atlas = phase3_result.atlas
        catalog = atlas.fragment_catalog
        phase4_model = phase4_result.model if phase4_result else None

        # Collect steerability-adjacent fragment IDs
        steer_fragment_ids: set[str] = set()
        if phase4_model:
            for region in phase4_model.steerable_regions:
                steer_fragment_ids.update(region.fragment_ids)
                steer_fragment_ids.update(region.high_interest_fragment_ids)

        # Pass 1: Extract nodes from fragments + state variables
        self._pass1_extract_nodes(
            graph=graph,
            target_id=target_id,
            catalog=catalog,
            phase4_model=phase4_model,
            steer_fragment_ids=steer_fragment_ids,
        )

        # Pass 2: Structural edges (call patterns + taint flows)
        self._pass2_structural_edges(
            graph=graph,
            target_id=target_id,
            catalog=catalog,
            phase3_result=phase3_result,
            steer_fragment_ids=steer_fragment_ids,
        )

        # Pass 3: Implicit / assumed-trust edges
        self._pass3_implicit_edges(
            graph=graph,
            target_id=target_id,
            catalog=catalog,
        )

        # Pass 4: Privilege value scoring
        self._pass4_privilege_scoring(graph=graph, steer_fragment_ids=steer_fragment_ids)

        # Build foothold bindings from Phase 4 condition sets
        bindings = self._build_foothold_bindings(
            graph=graph,
            target_id=target_id,
            phase4_result=phase4_result,
        )

        log.info(
            "trust_graph_built",
            nodes=graph.total_nodes,
            edges=graph.total_edges,
            foothold_bindings=len(bindings),
        )

        return graph, bindings

    # ── Pass 1: Node extraction ───────────────────────────────────────────────

    def _pass1_extract_nodes(
        self,
        graph: TrustGraph,
        target_id: str,
        catalog,
        phase4_model,
        steer_fragment_ids: set[str],
    ) -> None:
        """
        Extract TrustNodes from fragment catalog and Phase 4 state variables.
        """
        # From Phase 4 state variables
        if phase4_model:
            from systems.simula.inspector.constraint_types import StateVariableKind

            kind_to_node: dict[StateVariableKind, TrustNodeKind] = {
                StateVariableKind.IDENTITY_CONTEXT: TrustNodeKind.PRINCIPAL,
                StateVariableKind.PROTOCOL_STATE:   TrustNodeKind.SESSION,
                StateVariableKind.TAINT_LABEL:      TrustNodeKind.CREDENTIAL,
                StateVariableKind.FUNCTION_POINTER: TrustNodeKind.SERVICE,
                StateVariableKind.MEMORY_REGION:    TrustNodeKind.RESOURCE,
            }

            for var in phase4_model.state_variables.values():
                node_kind = kind_to_node.get(var.kind, TrustNodeKind.UNKNOWN)
                if node_kind == TrustNodeKind.UNKNOWN:
                    continue

                # Deduplicate by name
                if var.name in graph.nodes_by_name:
                    continue

                node = TrustNode(
                    target_id=target_id,
                    kind=node_kind,
                    name=var.name,
                    service_name=var.func_name,
                    file_path=var.file_path,
                    steerability_adjacent=(
                        bool(set(var.fragment_ids) & steer_fragment_ids)
                    ),
                    derived_from_variable_ids=[var.var_id],
                    derived_from_state_variable_kinds=[var.kind.value],
                    derived_from_fragment_ids=list(
                        set(var.fragment_ids) & steer_fragment_ids
                    ),
                )
                graph.add_node(node)

        # From fragment catalog — scan fragment descriptions for node signals
        for frag in catalog.fragments.values():
            desc = (frag.description or "") + " " + frag.func_name
            self._node_from_fragment(
                graph=graph,
                target_id=target_id,
                frag=frag,
                desc=desc,
                steer_fragment_ids=steer_fragment_ids,
            )

        # Ensure every unique function that appears in fragments becomes a SERVICE node
        for frag in catalog.fragments.values():
            if frag.func_name and frag.func_name not in graph.nodes_by_name:
                node = TrustNode(
                    target_id=target_id,
                    kind=TrustNodeKind.SERVICE,
                    name=frag.func_name,
                    service_name=frag.func_name,
                    file_path=frag.file_path,
                    steerability_adjacent=(frag.fragment_id in steer_fragment_ids),
                    derived_from_fragment_ids=[frag.fragment_id],
                )
                graph.add_node(node)

    def _node_from_fragment(
        self,
        graph: TrustGraph,
        target_id: str,
        frag,
        desc: str,
        steer_fragment_ids: set[str],
    ) -> None:
        """Classify a fragment as a trust node based on description patterns."""
        # Only emit one node per unique (kind, name) from fragments
        if _ROLE_FRAGMENT_PATTERN.search(desc):
            name = f"role:{frag.func_name}"
            kind = TrustNodeKind.ROLE
        elif _CRED_FRAGMENT_PATTERN.search(desc):
            name = f"cred:{frag.func_name}"
            kind = TrustNodeKind.CREDENTIAL
        else:
            return  # Non-role/cred fragments become SERVICE nodes in the loop above

        if name in graph.nodes_by_name:
            existing_id = graph.nodes_by_name[name]
            existing = graph.nodes.get(existing_id)
            if existing and frag.fragment_id not in existing.derived_from_fragment_ids:
                existing.derived_from_fragment_ids.append(frag.fragment_id)
            return

        node = TrustNode(
            target_id=target_id,
            kind=kind,
            name=name,
            service_name=frag.func_name,
            file_path=frag.file_path,
            steerability_adjacent=(frag.fragment_id in steer_fragment_ids),
            derived_from_fragment_ids=[frag.fragment_id],
        )
        graph.add_node(node)

    # ── Pass 2: Structural edge inference ─────────────────────────────────────

    def _pass2_structural_edges(
        self,
        graph: TrustGraph,
        target_id: str,
        catalog,
        phase3_result: Phase3Result,
        steer_fragment_ids: set[str],
    ) -> None:
        """
        Infer structural trust edges from CFG call patterns.
        """
        atlas = phase3_result.atlas

        # Build a map: func_name → list of callee func_names from CFG edges
        caller_to_callees: dict[str, list[str]] = {}
        cfg = atlas.cfg
        if cfg:
            for func in cfg.functions.values():
                callees: list[str] = []
                for block in func.blocks.values():
                    for edge in block.out_edges:
                        if edge.callee_name:
                            callees.append(edge.callee_name)
                if callees:
                    caller_to_callees[func.name] = callees

        # For each call edge: if caller is a SERVICE and callee is a SERVICE,
        # check what kind of trust the call implies
        for caller_name, callees in caller_to_callees.items():
            caller_id = graph.nodes_by_name.get(caller_name)
            if not caller_id:
                continue
            caller_node = graph.nodes.get(caller_id)
            if not caller_node:
                continue

            for callee_name in callees:
                callee_id = graph.nodes_by_name.get(callee_name)
                if not callee_id or callee_id == caller_id:
                    continue
                callee_node = graph.nodes.get(callee_id)
                if not callee_node:
                    continue

                # Decide edge kind from callee name patterns
                edge_kind, strength = self._classify_call_edge(
                    caller=caller_node,
                    callee=callee_node,
                )

                # Avoid duplicate edges
                if self._edge_exists(graph, caller_id, callee_id, edge_kind):
                    continue

                edge = TrustEdge(
                    target_id=target_id,
                    from_node_id=caller_id,
                    to_node_id=callee_id,
                    kind=edge_kind,
                    strength=strength,
                    description=(
                        f"{caller_name} → {callee_name} via {edge_kind.value} "
                        f"(inferred from CFG call edge)"
                    ),
                    inferred=True,
                    confidence=0.65,
                    traversal_cost=self._traversal_cost_for_strength(strength),
                )
                graph.add_edge(edge)

        # Credential→Service authentication edges from credential-named nodes
        cred_nodes = graph.nodes_of_kind(TrustNodeKind.CREDENTIAL)
        svc_nodes  = graph.nodes_of_kind(TrustNodeKind.SERVICE)
        for cred in cred_nodes:
            cred_func = cred.service_name
            for svc in svc_nodes:
                if cred_func and cred_func == svc.service_name:
                    # Credential is defined in the same function as the service → auth edge
                    if not self._edge_exists(graph, cred.node_id, svc.node_id, TrustEdgeKind.AUTHENTICATION):
                        edge = TrustEdge(
                            target_id=target_id,
                            from_node_id=cred.node_id,
                            to_node_id=svc.node_id,
                            kind=TrustEdgeKind.AUTHENTICATION,
                            strength=TrustStrength.VERIFIED,
                            description=(
                                f"Credential '{cred.name}' is used to authenticate to "
                                f"service '{svc.name}' (co-located in {cred_func})"
                            ),
                            inferred=True,
                            confidence=0.6,
                            traversal_cost=0.35,
                        )
                        graph.add_edge(edge)

        # Role → Principal/Service privilege-grant edges
        role_nodes = graph.nodes_of_kind(TrustNodeKind.ROLE)
        for role in role_nodes:
            for principal in graph.nodes_of_kind(TrustNodeKind.PRINCIPAL):
                if not self._edge_exists(graph, role.node_id, principal.node_id, TrustEdgeKind.PRIVILEGE_GRANT):
                    edge = TrustEdge(
                        target_id=target_id,
                        from_node_id=role.node_id,
                        to_node_id=principal.node_id,
                        kind=TrustEdgeKind.PRIVILEGE_GRANT,
                        strength=TrustStrength.IMPLICIT,
                        description=(
                            f"Role '{role.name}' grants capabilities to principal '{principal.name}'"
                        ),
                        inferred=True,
                        confidence=0.5,
                        traversal_cost=0.4,
                    )
                    graph.add_edge(edge)

        # Delegation edges from Phase 4 IDENTITY_CONTEXT → SERVICE patterns
        for frag in catalog.fragments.values():
            desc = frag.description or ""
            if _DELEGATION_PATTERN.search(desc):
                # Fragment likely contains a delegation call; create delegation edge
                # from whatever SERVICE contains this fragment to called services
                src_name = frag.func_name
                src_id = graph.nodes_by_name.get(src_name)
                if src_id:
                    for callee_name in caller_to_callees.get(src_name, []):
                        dst_id = graph.nodes_by_name.get(callee_name)
                        if dst_id and dst_id != src_id:
                            if not self._edge_exists(graph, src_id, dst_id, TrustEdgeKind.DELEGATION):
                                edge = TrustEdge(
                                    target_id=target_id,
                                    from_node_id=src_id,
                                    to_node_id=dst_id,
                                    kind=TrustEdgeKind.DELEGATION,
                                    strength=TrustStrength.VERIFIED,
                                    description=(
                                        f"'{src_name}' delegates to '{callee_name}' "
                                        f"(delegation pattern in fragment {frag.fragment_id[:8]})"
                                    ),
                                    evidence_fragment_ids=[frag.fragment_id],
                                    inferred=True,
                                    confidence=0.7,
                                    traversal_cost=0.3,
                                )
                                graph.add_edge(edge)

    def _classify_call_edge(
        self,
        caller: TrustNode,
        callee: TrustNode,
    ) -> tuple[TrustEdgeKind, TrustStrength]:
        """
        Classify a call edge between two nodes.
        Returns (TrustEdgeKind, TrustStrength).
        """
        # Credential reuse: if caller has a CREDENTIAL kind ancestor
        if callee.kind == TrustNodeKind.CREDENTIAL:
            return TrustEdgeKind.CREDENTIAL_REUSE, TrustStrength.IMPLICIT
        # Service calling service → service assumption
        if caller.kind == TrustNodeKind.SERVICE and callee.kind == TrustNodeKind.SERVICE:
            return TrustEdgeKind.SERVICE_ASSUMPTION, TrustStrength.IMPLICIT
        # Principal → session
        if caller.kind == TrustNodeKind.PRINCIPAL and callee.kind == TrustNodeKind.SESSION:
            return TrustEdgeKind.AUTHENTICATION, TrustStrength.VERIFIED
        # Session → resource
        if caller.kind == TrustNodeKind.SESSION and callee.kind == TrustNodeKind.RESOURCE:
            return TrustEdgeKind.IMPLICIT_PERMISSION, TrustStrength.VERIFIED
        # Default
        return TrustEdgeKind.ASSUMED_TRUST, TrustStrength.IMPLICIT

    # ── Pass 3: Implicit / assumed-trust edges ────────────────────────────────

    def _pass3_implicit_edges(
        self,
        graph: TrustGraph,
        target_id: str,
        catalog,
    ) -> None:
        """
        Infer implicit/assumed-trust edges from co-location and default patterns.
        """
        # Nodes sharing the same service_name with no existing auth edge → ASSUMED_TRUST
        service_name_to_nodes: dict[str, list[str]] = {}
        for node in graph.nodes.values():
            if node.service_name:
                service_name_to_nodes.setdefault(node.service_name, []).append(node.node_id)

        for svc_name, nids in service_name_to_nodes.items():
            if len(nids) < 2:
                continue
            for i, a_id in enumerate(nids):
                for b_id in nids[i + 1 :]:
                    if a_id == b_id:
                        continue
                    # Only add if no direct edge already exists in either direction
                    if (
                        not self._edge_exists(graph, a_id, b_id, TrustEdgeKind.ASSUMED_TRUST)
                        and not self._any_edge_between(graph, a_id, b_id)
                    ):
                        a_node = graph.nodes[a_id]
                        b_node = graph.nodes[b_id]
                        edge = TrustEdge(
                            target_id=target_id,
                            from_node_id=a_id,
                            to_node_id=b_id,
                            kind=TrustEdgeKind.ASSUMED_TRUST,
                            strength=TrustStrength.IMPLICIT,
                            description=(
                                f"'{a_node.name}' and '{b_node.name}' share service "
                                f"context '{svc_name}' with no explicit auth between them"
                            ),
                            inferred=True,
                            conditional=False,
                            confidence=0.55,
                            traversal_cost=0.2,
                        )
                        graph.add_edge(edge)

        # Scan fragments for wildcard/implicit-permission patterns
        for frag in catalog.fragments.values():
            desc = frag.description or ""
            if _IMPLICIT_PERM_PATTERN.search(desc):
                src_id = graph.nodes_by_name.get(frag.func_name)
                if src_id:
                    # Add IMPLICIT_PERMISSION edges to all resources
                    for res in graph.nodes_of_kind(TrustNodeKind.RESOURCE):
                        if not self._edge_exists(graph, src_id, res.node_id, TrustEdgeKind.IMPLICIT_PERMISSION):
                            edge = TrustEdge(
                                target_id=target_id,
                                from_node_id=src_id,
                                to_node_id=res.node_id,
                                kind=TrustEdgeKind.IMPLICIT_PERMISSION,
                                strength=TrustStrength.BLIND,
                                description=(
                                    f"'{frag.func_name}' contains implicit permission "
                                    f"bypass pattern (fragment {frag.fragment_id[:8]})"
                                ),
                                evidence_fragment_ids=[frag.fragment_id],
                                inferred=True,
                                confidence=0.65,
                                traversal_cost=0.1,
                            )
                            graph.add_edge(edge)

        # Shared-secret patterns → SHARED_SECRET edges between co-occurring service pairs
        for frag in catalog.fragments.values():
            desc = frag.description or ""
            if _SHARED_SECRET_PATTERN.search(desc):
                src_id = graph.nodes_by_name.get(frag.func_name)
                if src_id:
                    # Shared secret → trust all nodes in same service namespace
                    peer_ids = service_name_to_nodes.get(frag.func_name, [])
                    for peer_id in peer_ids:
                        if peer_id != src_id and not self._edge_exists(
                            graph, src_id, peer_id, TrustEdgeKind.SHARED_SECRET
                        ):
                            peer = graph.nodes[peer_id]
                            src  = graph.nodes[src_id]
                            edge = TrustEdge(
                                target_id=target_id,
                                from_node_id=src_id,
                                to_node_id=peer_id,
                                kind=TrustEdgeKind.SHARED_SECRET,
                                strength=TrustStrength.IMPLICIT,
                                description=(
                                    f"'{src.name}' shares a secret with '{peer.name}' "
                                    f"(shared-secret pattern in fragment {frag.fragment_id[:8]})"
                                ),
                                evidence_fragment_ids=[frag.fragment_id],
                                inferred=True,
                                confidence=0.6,
                                traversal_cost=0.25,
                            )
                            graph.add_edge(edge)

    # ── Pass 4: Privilege scoring ─────────────────────────────────────────────

    def _pass4_privilege_scoring(
        self,
        graph: TrustGraph,
        steer_fragment_ids: set[str],
    ) -> None:
        """
        Assign privilege_value and privilege_impact to every node.
        """
        for node in graph.nodes.values():
            base = _base_privilege_for_kind(node.kind)
            name_score, name_impact = _privilege_score_for_name(node.name)

            # Blend: 40% kind-base, 60% name-heuristic
            raw = int(0.4 * base + 0.6 * name_score)

            # Steerability boost: +10 if adjacent to a Phase 4 steerable region
            if node.steerability_adjacent:
                raw = min(100, raw + 10)

            node.privilege_value  = max(0, min(100, raw))
            node.privilege_impact = name_impact

    # ── Foothold binding ──────────────────────────────────────────────────────

    def _build_foothold_bindings(
        self,
        graph: TrustGraph,
        target_id: str,
        phase4_result: Phase4Result | None,
    ) -> list[FootholdBinding]:
        """
        Map Phase 4 ConditionSets → TrustGraph nodes as footholds.
        """
        if not phase4_result:
            # Fall back to steerability-adjacent nodes as footholds
            bindings = []
            for node in graph.nodes.values():
                if node.steerability_adjacent:
                    b = FootholdBinding(
                        target_id=target_id,
                        condition_set_id="",
                        steerable_region_id="",
                        steerability_class="influence_permissive",
                        foothold_node_id=node.node_id,
                        foothold_node_name=node.name,
                        foothold_node_kind=node.kind,
                        rationale=(
                            f"Node '{node.name}' is adjacent to a Phase 4 steerable "
                            f"fragment but no Phase 4 result was provided."
                        ),
                        confidence=0.4,
                    )
                    bindings.append(b)
            return bindings

        model = phase4_result.model
        bindings: list[FootholdBinding] = []

        for cs in model.condition_sets:
            region = next(
                (r for r in model.steerable_regions if r.region_id == cs.unlocked_region_id),
                None,
            )
            if not region:
                continue

            # Find the best trust-graph node to use as a foothold for this region:
            # prefer SERVICE nodes that appear in the region's fragments/functions
            best_node: TrustNode | None = None
            best_score = -1

            for func_name in region.function_names:
                candidate_id = graph.nodes_by_name.get(func_name)
                if candidate_id:
                    candidate = graph.nodes[candidate_id]
                    # Prefer steerability-adjacent nodes
                    score = candidate.privilege_value + (20 if candidate.steerability_adjacent else 0)
                    if score > best_score:
                        best_score = score
                        best_node = candidate

            # Fallback: any node derived from a region fragment
            if not best_node:
                for node in graph.nodes.values():
                    overlap = len(set(node.derived_from_fragment_ids) & set(region.fragment_ids))
                    if overlap:
                        score = node.privilege_value + overlap * 5
                        if score > best_score:
                            best_score = score
                            best_node = node

            if not best_node:
                continue

            binding = FootholdBinding(
                target_id=target_id,
                condition_set_id=cs.condition_set_id,
                steerable_region_id=region.region_id,
                steerability_class=region.steerability_class.value,
                foothold_node_id=best_node.node_id,
                foothold_node_name=best_node.name,
                foothold_node_kind=best_node.kind,
                rationale=(
                    f"Phase 4 ConditionSet '{cs.description[:80]}' unlocks region "
                    f"'{region.region_id[:8]}'; node '{best_node.name}' is the best "
                    f"trust-graph entry point for that region."
                ),
                confidence=round(cs.confidence * 0.9, 4),
            )
            bindings.append(binding)

        return bindings

    # ── Utility helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _edge_exists(
        graph: TrustGraph,
        from_id: str,
        to_id: str,
        kind: TrustEdgeKind,
    ) -> bool:
        """Return True if an edge of `kind` from → to already exists."""
        for eid in graph.out_edges.get(from_id, []):
            edge = graph.edges.get(eid)
            if edge and edge.to_node_id == to_id and edge.kind == kind:
                return True
        return False

    @staticmethod
    def _any_edge_between(graph: TrustGraph, a_id: str, b_id: str) -> bool:
        """Return True if any edge exists in either direction between a and b."""
        for eid in graph.out_edges.get(a_id, []):
            if graph.edges[eid].to_node_id == b_id:
                return True
        return any(graph.edges[eid].to_node_id == a_id for eid in graph.out_edges.get(b_id, []))

    @staticmethod
    def _traversal_cost_for_strength(strength: TrustStrength) -> float:
        return {
            TrustStrength.EXPLICIT: 0.85,
            TrustStrength.VERIFIED: 0.55,
            TrustStrength.IMPLICIT: 0.25,
            TrustStrength.BLIND:    0.05,
        }.get(strength, 0.3)
