"""
EcodiaOS - Nexus: Convergence Detector

Compares abstract relational structures from different world model fragments
to detect structural isomorphism. When two instances with different domains,
different compression paths, and different experiences arrive at the same
abstract structure, that convergence is evidence for ground truth.

The comparison uses two paths:

  1. **WL-1 graph isomorphism approximation** (primary, Gap HIGH-1):
     Weisfeiler-Lehman 1-dimensional colour refinement.  O(n·d·k) where n is
     node count, d is mean degree, and k is iteration count (default 3).
     WL-1 cannot distinguish all non-isomorphic graphs (e.g. regular graphs
     with identical degree sequences) but catches the vast majority of cases
     seen in world model fragment sizes (n < 100 typical).  When two graphs
     produce the same canonical WL-1 histogram, they are isomorphic with high
     probability.

  2. **Legacy heuristic** (fallback):
     Node count, edge count, type distributions, degree sequence, symmetry,
     and invariant Jaccard.  Used when fragments lack node/edge lists for
     WL-1 (e.g. older fragments serialised before AbstractStructure was typed).

A high convergence_score (>= 0.7) from independent domains is the strongest
epistemic signal Nexus can produce.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import structlog

from primitives.common import utc_now
from systems.nexus.types import (
    AbstractStructure,
    ConvergenceResult,
    ShareableWorldModelFragment,
    TriangulationMetadata,
    TriangulationSource,
)

logger = structlog.get_logger("nexus.convergence")

# ─── WL-1 Graph Isomorphism ──────────────────────────────────────


def _wl1_hash(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    iterations: int = 3,
) -> str:
    """
    Compute a Weisfeiler-Lehman 1-dimensional canonical hash for a graph.

    Algorithm:
      1. Initialise each node's colour from its ``type`` label.
      2. For each iteration:
         a. For every node, collect its current colour + sorted neighbour colours.
         b. Hash that multiset to produce the node's new colour.
      3. Produce a canonical histogram: sorted list of (colour, count) pairs.
      4. SHA-256 the histogram string.

    The resulting hash is the same for isomorphic graphs regardless of node
    ordering.  Two graphs with identical hashes are isomorphic with high
    probability (WL-1 is not complete - it cannot distinguish certain regular
    graphs, but these are rare in world model fragment sizes).

    Complexity: O(n · d · k) where n = node count, d = mean out-degree,
    k = iterations.  Tractable for n < 1 000.

    Returns empty string if the graph has no nodes (caller falls back to
    legacy heuristics).
    """
    if not nodes:
        return ""

    n = len(nodes)

    # Build adjacency list (directed: only outbound edges affect colour)
    adjacency: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        src = edge.get("from", edge.get("source", -1))
        dst = edge.get("to", edge.get("target", -1))
        if isinstance(src, int) and isinstance(dst, int) and 0 <= src < n and 0 <= dst < n:
            adjacency[src].append(dst)

    # Initialise colours from node type labels
    colours: list[str] = [str(node.get("type", "unknown")) for node in nodes]

    for _ in range(iterations):
        new_colours: list[str] = []
        for i in range(n):
            neighbour_colours = sorted(colours[j] for j in adjacency[i])
            # Aggregate: own colour + sorted neighbour colours
            aggregate = colours[i] + "|" + ",".join(neighbour_colours)
            new_colours.append(hashlib.sha256(aggregate.encode()).hexdigest()[:16])
        colours = new_colours

    # Canonical histogram: count occurrences of each colour, sort for stability
    hist: dict[str, int] = defaultdict(int)
    for c in colours:
        hist[c] += 1
    canonical = str(sorted(hist.items()))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _wl1_similarity(hash_a: str, hash_b: str) -> float:
    """
    Convert two WL-1 hashes to a similarity score [0, 1].

    Exact match → 1.0 (structurally isomorphic with high probability).
    Any mismatch → 0.0 (definitely not isomorphic).

    This binary decision is correct for WL-1: the hash is either a proof of
    isomorphism (with high probability) or evidence of non-isomorphism.
    We do NOT interpolate between hashes - Hamming distance on SHA-256 digests
    is meaningless for this purpose.
    """
    if not hash_a or not hash_b:
        return -1.0  # Sentinel: WL-1 unavailable, caller must use legacy path
    return 1.0 if hash_a == hash_b else 0.0


def _extract_graph_from_structure(
    structure: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract (nodes, edges) from a legacy abstract_structure dict."""
    raw_nodes = structure.get("nodes", [])
    nodes: list[dict[str, Any]]
    if isinstance(raw_nodes, int):
        nodes = [{"type": "unknown", "index": i} for i in range(raw_nodes)]
    elif isinstance(raw_nodes, list):
        nodes = raw_nodes
    else:
        nodes = []

    raw_edges = structure.get("edges", [])
    edges: list[dict[str, Any]] = raw_edges if isinstance(raw_edges, list) else []
    return nodes, edges


def _get_graph(
    fragment: ShareableWorldModelFragment,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Return (nodes, edges) preferring the typed AbstractStructure when available.
    """
    if fragment.typed_structure is not None:
        return fragment.typed_structure.nodes, fragment.typed_structure.edges
    return _extract_graph_from_structure(fragment.abstract_structure)


# ─── ConvergenceDetector ────────────────────────────────────────


class ConvergenceDetector:
    """
    Detects structural convergence between world model fragments.

    Primary path: WL-1 graph isomorphism approximation (Gap HIGH-1).
    Fallback path: legacy heuristics (node/edge counts, type distributions,
    symmetry, invariants) for fragments without node/edge lists.

    Core operation: compare_structures strips domain labels and compares
    the abstract relational shape of two fragments. When different instances
    independently compress different domains into the same structure,
    that structure has high triangulation confidence.
    """

    def __init__(self, *, wl1_iterations: int = 3) -> None:
        self._convergence_cache: dict[str, ConvergenceResult] = {}
        self._wl1_iterations = wl1_iterations

    def compare_structures(
        self,
        fragment_a: ShareableWorldModelFragment,
        fragment_b: ShareableWorldModelFragment,
        *,
        peer_divergence_score: float | None = None,
    ) -> ConvergenceResult:
        """
        Compare two fragments for structural isomorphism.

        Strips domain labels from both, compares abstract relational
        shape. Returns a ConvergenceResult with a score from 0.0
        (no match) to 1.0 (identical structure, different domains).

        WL-1 path (preferred):
          - Used when both fragments have nodes lists (either via typed_structure
            or via abstract_structure["nodes"] as a list).
          - Produces an exact binary result: 1.0 (isomorphic) or 0.0 (not).
          - Blended with size similarity (10%) to give partial credit for
            near-matches caused by minor structural differences.

        Legacy path (fallback):
          - Used when node lists are unavailable.
          - Heuristic score from count ratios, type distributions, degree
            sequences, symmetry, and invariant overlap.

        peer_divergence_score: the measured divergence between the two
          source instances (from InstanceDivergenceMeasurer). When
          provided, used directly as source_diversity. When absent,
          falls back to a minimum of 0.3 if the instance IDs differ.
        """
        cache_key = _cache_key(fragment_a.fragment_id, fragment_b.fragment_id)
        cached = self._convergence_cache.get(cache_key)
        if cached is not None:
            return cached

        nodes_a, edges_a = _get_graph(fragment_a)
        nodes_b, edges_b = _get_graph(fragment_b)

        # ── WL-1 path ────────────────────────────────────────────
        wl1_used = False
        wl1_score: float | None = None

        if nodes_a and nodes_b:
            hash_a = _wl1_hash(nodes_a, edges_a, iterations=self._wl1_iterations)
            hash_b = _wl1_hash(nodes_b, edges_b, iterations=self._wl1_iterations)
            raw = _wl1_similarity(hash_a, hash_b)
            if raw >= 0.0:  # Both hashes valid
                wl1_used = True
                # Blend WL-1 result (90%) with node count ratio (10%) to give
                # soft credit for graphs that are "almost" isomorphic structurally
                # but differ by ±1 node (common in growing world models).
                n_a, n_b = len(nodes_a), len(nodes_b)
                size_ratio = min(n_a, n_b) / max(n_a, n_b) if max(n_a, n_b) > 0 else 1.0
                wl1_score = raw * 0.90 + size_ratio * 0.10

        if wl1_used and wl1_score is not None:
            convergence_score = wl1_score
            matched_nodes = min(len(nodes_a), len(nodes_b))
            matched_edges = min(len(edges_a), len(edges_b))
        else:
            # ── Legacy heuristic path ────────────────────────────
            struct_a = (
                fragment_a.typed_structure.to_legacy_dict()
                if fragment_a.typed_structure
                else fragment_a.abstract_structure
            )
            struct_b = (
                fragment_b.typed_structure.to_legacy_dict()
                if fragment_b.typed_structure
                else fragment_b.abstract_structure
            )

            ext_nodes_a = _extract_nodes(struct_a)
            ext_nodes_b = _extract_nodes(struct_b)
            ext_edges_a = _extract_edges(struct_a)
            ext_edges_b = _extract_edges(struct_b)

            node_score, matched_nodes = _compare_node_topology(ext_nodes_a, ext_nodes_b)
            edge_score, matched_edges = _compare_edge_topology(ext_edges_a, ext_edges_b)
            symmetry_score = _compare_symmetry(struct_a, struct_b)
            invariant_score = _compare_invariants(struct_a, struct_b)

            convergence_score = (
                node_score * 0.30
                + edge_score * 0.35
                + symmetry_score * 0.20
                + invariant_score * 0.15
            )

        # ── Domain independence ───────────────────────────────────
        domains_a = set(fragment_a.domain_labels)
        domains_b = set(fragment_b.domain_labels)
        domains_are_independent = (
            len(domains_a & domains_b) == 0
            if (domains_a and domains_b)
            else False
        )

        # ── Source diversity ──────────────────────────────────────
        if peer_divergence_score is not None:
            source_diversity = max(0.0, min(peer_divergence_score, 1.0))
        elif fragment_a.source_instance_id != fragment_b.source_instance_id:
            source_diversity = 0.3
        else:
            source_diversity = 0.0

        result = ConvergenceResult(
            fragment_a_id=fragment_a.fragment_id,
            fragment_b_id=fragment_b.fragment_id,
            convergence_score=min(convergence_score, 1.0),
            matched_nodes=matched_nodes,
            total_nodes_a=len(nodes_a) if nodes_a else len(_extract_nodes(fragment_a.abstract_structure)),
            total_nodes_b=len(nodes_b) if nodes_b else len(_extract_nodes(fragment_b.abstract_structure)),
            matched_edges=matched_edges,
            total_edges_a=len(edges_a) if edges_a else len(_extract_edges(fragment_a.abstract_structure)),
            total_edges_b=len(edges_b) if edges_b else len(_extract_edges(fragment_b.abstract_structure)),
            domains_are_independent=domains_are_independent,
            source_a_instance_id=fragment_a.source_instance_id,
            source_b_instance_id=fragment_b.source_instance_id,
            source_diversity=source_diversity,
            wl1_used=wl1_used,
            detected_at=utc_now(),
        )

        self._convergence_cache[cache_key] = result

        if result.is_convergent:
            logger.info(
                "convergence_detected",
                fragment_a=fragment_a.fragment_id,
                fragment_b=fragment_b.fragment_id,
                score=convergence_score,
                wl1_used=wl1_used,
                domains_independent=domains_are_independent,
                source_diversity=source_diversity,
            )

        return result

    def update_triangulation(
        self,
        metadata: TriangulationMetadata,
        confirming_instance_id: str,
        confirming_divergence_score: float,
        confirming_fragment_id: str,
    ) -> TriangulationMetadata:
        """
        Update triangulation metadata when a new independent source confirms
        the same structure.

        Deduplicates by instance_id - same instance confirming twice
        doesn't increase triangulation confidence.
        """
        existing_ids = {s.instance_id for s in metadata.independent_sources}
        if confirming_instance_id in existing_ids:
            return metadata

        metadata.independent_sources.append(
            TriangulationSource(
                instance_id=confirming_instance_id,
                divergence_score=confirming_divergence_score,
                fragment_id=confirming_fragment_id,
                confirmed_at=utc_now(),
            )
        )

        logger.info(
            "triangulation_updated",
            source_count=metadata.independent_source_count,
            diversity=metadata.source_diversity_score,
            confidence=metadata.triangulation_confidence,
            confirming_instance=confirming_instance_id,
        )

        return metadata

    def clear_cache(self) -> int:
        """Clear the convergence cache. Returns number of entries cleared."""
        count = len(self._convergence_cache)
        self._convergence_cache.clear()
        return count


# ─── Structural Comparison Helpers (Legacy Path) ─────────────────


def _cache_key(id_a: str, id_b: str) -> str:
    """Canonical cache key - order-independent."""
    return f"{min(id_a, id_b)}::{max(id_a, id_b)}"


def _extract_nodes(structure: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract node list from abstract structure."""
    nodes = structure.get("nodes", [])
    if isinstance(nodes, int):
        return [{"index": i} for i in range(nodes)]
    if isinstance(nodes, list):
        return nodes
    return []


def _extract_edges(structure: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract edge list from abstract structure."""
    edges = structure.get("edges", [])
    if isinstance(edges, list):
        return edges
    return []


def _compare_node_topology(
    nodes_a: list[dict[str, Any]],
    nodes_b: list[dict[str, Any]],
) -> tuple[float, int]:
    """
    Compare node topology. Returns (score, matched_count).

    Score based on count similarity and type distribution similarity.
    """
    if not nodes_a and not nodes_b:
        return 1.0, 0
    if not nodes_a or not nodes_b:
        return 0.0, 0

    count_a = len(nodes_a)
    count_b = len(nodes_b)
    count_ratio = min(count_a, count_b) / max(count_a, count_b)

    types_a = _type_distribution(nodes_a)
    types_b = _type_distribution(nodes_b)
    type_sim = _distribution_similarity(types_a, types_b)

    matched = min(count_a, count_b)
    score = count_ratio * 0.5 + type_sim * 0.5

    return score, matched


def _compare_edge_topology(
    edges_a: list[dict[str, Any]],
    edges_b: list[dict[str, Any]],
) -> tuple[float, int]:
    """
    Compare edge topology. Returns (score, matched_count).

    Score based on count similarity, type distribution, and degree sequence.
    """
    if not edges_a and not edges_b:
        return 1.0, 0
    if not edges_a or not edges_b:
        return 0.0, 0

    count_a = len(edges_a)
    count_b = len(edges_b)
    count_ratio = min(count_a, count_b) / max(count_a, count_b)

    types_a = _type_distribution(edges_a)
    types_b = _type_distribution(edges_b)
    type_sim = _distribution_similarity(types_a, types_b)

    degree_a = _degree_sequence(edges_a)
    degree_b = _degree_sequence(edges_b)
    degree_sim = _sequence_similarity(degree_a, degree_b)

    matched = min(count_a, count_b)
    score = count_ratio * 0.3 + type_sim * 0.35 + degree_sim * 0.35

    return score, matched


def _compare_symmetry(
    struct_a: dict[str, Any], struct_b: dict[str, Any]
) -> float:
    """
    Compare declared symmetry properties.
    Exact match = 1.0, no info = 0.5 (neutral), mismatch = 0.0.
    """
    sym_a = struct_a.get("symmetry", "")
    sym_b = struct_b.get("symmetry", "")

    if not sym_a and not sym_b:
        return 0.5

    if not sym_a or not sym_b:
        return 0.25

    if sym_a == sym_b:
        return 1.0

    related_pairs = {
        frozenset({"chain", "path"}),
        frozenset({"cycle", "ring"}),
        frozenset({"star", "hub"}),
        frozenset({"tree", "hierarchy"}),
    }
    if frozenset({sym_a, sym_b}) in related_pairs:
        return 0.8

    return 0.0


def _compare_invariants(
    struct_a: dict[str, Any], struct_b: dict[str, Any]
) -> float:
    """Compare declared invariant properties (Jaccard similarity)."""
    inv_a = set(struct_a.get("invariants", []))
    inv_b = set(struct_b.get("invariants", []))

    if not inv_a and not inv_b:
        return 0.5

    if not inv_a or not inv_b:
        return 0.25

    union = len(inv_a | inv_b)
    if union == 0:
        return 0.5

    return len(inv_a & inv_b) / union


def _type_distribution(items: list[dict[str, Any]]) -> dict[str, int]:
    """Count occurrences of each 'type' value in a list of dicts."""
    dist: dict[str, int] = {}
    for item in items:
        t = str(item.get("type", "unknown"))
        dist[t] = dist.get(t, 0) + 1
    return dist


def _distribution_similarity(
    dist_a: dict[str, int], dist_b: dict[str, int]
) -> float:
    """Normalised intersection / union (Jaccard-like on counts)."""
    if not dist_a and not dist_b:
        return 1.0
    if not dist_a or not dist_b:
        return 0.0

    all_keys = set(dist_a) | set(dist_b)
    intersection = sum(min(dist_a.get(k, 0), dist_b.get(k, 0)) for k in all_keys)
    union = sum(max(dist_a.get(k, 0), dist_b.get(k, 0)) for k in all_keys)

    return intersection / union if union > 0 else 0.0


def _degree_sequence(edges: list[dict[str, Any]]) -> list[int]:
    """Compute sorted degree sequence from edge list."""
    degrees: dict[int, int] = {}
    for edge in edges:
        src = edge.get("from", edge.get("source", 0))
        dst = edge.get("to", edge.get("target", 0))
        if isinstance(src, int):
            degrees[src] = degrees.get(src, 0) + 1
        if isinstance(dst, int):
            degrees[dst] = degrees.get(dst, 0) + 1

    return sorted(degrees.values(), reverse=True) if degrees else []


def _sequence_similarity(seq_a: list[int], seq_b: list[int]) -> float:
    """Compare two sorted integer sequences with zero-padding."""
    if not seq_a and not seq_b:
        return 1.0
    if not seq_a or not seq_b:
        return 0.0

    max_len = max(len(seq_a), len(seq_b))
    padded_a = seq_a + [0] * (max_len - len(seq_a))
    padded_b = seq_b + [0] * (max_len - len(seq_b))

    max_val = max(max(padded_a), max(padded_b), 1)
    total_sim = sum(
        1.0 - abs(a - b) / max_val for a, b in zip(padded_a, padded_b, strict=True)
    )

    return total_sim / max_len
