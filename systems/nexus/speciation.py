"""
EcodiaOS — Nexus: Speciation Detection and Invariant Bridge

Phase C of epistemic triangulation. When two instances diverge beyond
the speciation threshold (overall divergence >= 0.8), they become
"alien kinds" — normal fragment sharing is no longer possible because
their structural languages are incompatible.

Post-speciation, only the most compressed structures (causal invariants)
can cross the boundary via InvariantBridge. These are stripped of ALL
domain context and compared at the purest structural level.

Convergence across speciated instances is the strongest possible evidence
for ground truth — two alien-kind instances with incompatible structural
languages independently arrived at the same abstract form.

Components:
  SpeciationDetector  — monitors divergence scores for speciation threshold
  InvariantBridge     — communication channel between speciated instances
  SpeciationRegistry  — tracks all speciation events, cognitive kinds, bridges
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now
from systems.nexus.types import (
    CognitiveKindEntry,
    ConvergedInvariant,
    DivergenceScore,
    InvariantExchangeReport,
    NexusConfig,
    ShareableWorldModelFragment,
    SpeciationEvent,
    SpeciationRegistryState,
)

if TYPE_CHECKING:
    from systems.nexus.protocols import LogosWorldModelProtocol

logger = structlog.get_logger("nexus.speciation")


class SpeciationDetector:
    """
    Monitors divergence scores and detects speciation events.

    Speciation occurs when overall divergence between two instances
    reaches >= 0.8 (ALIEN_KIND classification). Once speciated,
    normal fragment sharing is blocked — only InvariantBridge remains.
    """

    def __init__(self, *, config: NexusConfig | None = None) -> None:
        self._config = config or NexusConfig()
        self._threshold = self._config.speciation_divergence_threshold

    def check_for_speciation(
        self,
        divergence: DivergenceScore,
        registry: SpeciationRegistry,
    ) -> SpeciationEvent | None:
        """
        Check if a divergence score triggers a speciation event.

        Returns a SpeciationEvent if the threshold is met and these
        instances haven't already speciated. Returns None otherwise.
        """
        if divergence.overall < self._threshold:
            return None

        instance_a = divergence.instance_a_id
        instance_b = divergence.instance_b_id

        # Already speciated — no duplicate events
        if registry.are_speciated(instance_a, instance_b):
            return None

        event = SpeciationEvent(
            instance_a_id=instance_a,
            instance_b_id=instance_b,
            timestamp=utc_now(),
            divergence_score=divergence.overall,
        )

        logger.info(
            "speciation_detected",
            instance_a=instance_a,
            instance_b=instance_b,
            divergence=divergence.overall,
        )

        return event


class InvariantBridge:
    """
    Communication channel between speciated instances.

    Normal fragment sharing is impossible post-speciation (incompatible
    structural languages). The InvariantBridge strips ALL domain context
    and compares at the purest structural level — causal invariants only.

    When two invariants from alien-kind instances match abstractly,
    this constitutes the strongest possible evidence for ground truth.
    """

    def exchange_invariants(
        self,
        logos_a: LogosWorldModelProtocol,
        logos_b: LogosWorldModelProtocol,
        instance_a_id: str,
        instance_b_id: str,
        bridge_id: str = "",
    ) -> InvariantExchangeReport:
        """
        Exchange and compare invariants between two speciated instances.

        Extracts causal invariants from each Logos world model, strips
        all domain context, and compares at the purest structural level.
        """
        if not bridge_id:
            bridge_id = new_id()

        invariants_a = _extract_causal_invariants(logos_a, instance_a_id)
        invariants_b = _extract_causal_invariants(logos_b, instance_b_id)

        converged: list[ConvergedInvariant] = []

        for inv_a in invariants_a:
            for inv_b in invariants_b:
                if _are_abstractly_equivalent(inv_a, inv_b):
                    ci = ConvergedInvariant(
                        invariant_a_id=inv_a.fragment_id,
                        invariant_b_id=inv_b.fragment_id,
                        source_instance_a=instance_a_id,
                        source_instance_b=instance_b_id,
                        abstract_form=_extract_abstract_form(inv_a, inv_b),
                        triangulation_confidence=0.95,
                        is_ground_truth_candidate=True,
                        converged_at=utc_now(),
                    )
                    converged.append(ci)

        report = InvariantExchangeReport(
            bridge_id=bridge_id,
            instance_a_id=instance_a_id,
            instance_b_id=instance_b_id,
            invariants_compared=len(invariants_a) * len(invariants_b),
            converged_invariants=converged,
            abstract_equivalences_found=len(converged),
            exchange_timestamp=utc_now(),
        )

        if converged:
            logger.info(
                "invariant_bridge_convergence",
                bridge_id=bridge_id,
                instance_a=instance_a_id,
                instance_b=instance_b_id,
                equivalences=len(converged),
                total_compared=report.invariants_compared,
            )

        return report


class SpeciationRegistry:
    """
    Federation-level registry of speciation events, cognitive kinds,
    and active invariant bridge connections.

    Tracks which instances are speciated, what cognitive kinds exist,
    and which bridges are active for invariant exchange.
    """

    def __init__(self) -> None:
        self._state = SpeciationRegistryState()
        # Fast lookup: frozenset({a, b}) → SpeciationEvent
        self._speciation_pairs: dict[frozenset[str], SpeciationEvent] = {}
        # instance_id → kind_id
        self._instance_to_kind: dict[str, str] = {}

    @property
    def state(self) -> SpeciationRegistryState:
        return self._state

    def register_speciation(
        self,
        event: SpeciationEvent,
        *,
        logos_a: LogosWorldModelProtocol | None = None,
        logos_b: LogosWorldModelProtocol | None = None,
    ) -> None:
        """
        Register a speciation event and update cognitive kinds.

        If both instances are in the same kind, split them into separate
        kinds. If neither has a kind, create two new kinds.
        """
        pair = frozenset({event.instance_a_id, event.instance_b_id})
        if pair in self._speciation_pairs:
            return  # Already registered

        # Compute schema compatibility metadata if Logos available
        if logos_a is not None and logos_b is not None:
            schemas_a = set(logos_a.get_schema_ids())
            schemas_b = set(logos_b.get_schema_ids())
            shared = schemas_a & schemas_b
            # "Incompatible" schemas: those unique to each instance
            incompatible = len((schemas_a | schemas_b) - shared)
            event.shared_invariant_count = len(shared)
            event.incompatible_schema_count = incompatible

        self._speciation_pairs[pair] = event
        self._state.speciation_events.append(event)

        # Update cognitive kinds
        kind_a = self._instance_to_kind.get(event.instance_a_id)
        kind_b = self._instance_to_kind.get(event.instance_b_id)

        if kind_a is None and kind_b is None:
            # Neither has a kind — create two new ones
            new_kind_a = CognitiveKindEntry(
                member_instance_ids=[event.instance_a_id],
                founding_speciation_event_id=event.id,
                established_at=utc_now(),
            )
            new_kind_b = CognitiveKindEntry(
                member_instance_ids=[event.instance_b_id],
                founding_speciation_event_id=event.id,
                established_at=utc_now(),
            )
            self._state.cognitive_kinds.append(new_kind_a)
            self._state.cognitive_kinds.append(new_kind_b)
            self._instance_to_kind[event.instance_a_id] = new_kind_a.kind_id
            self._instance_to_kind[event.instance_b_id] = new_kind_b.kind_id
            event.new_cognitive_kind_registered = True
        elif kind_a == kind_b and kind_a is not None:
            # Same kind — split instance_b into a new kind
            old_kind = self._find_kind(kind_a)
            if old_kind is not None and event.instance_b_id in old_kind.member_instance_ids:
                old_kind.member_instance_ids.remove(event.instance_b_id)
            new_kind = CognitiveKindEntry(
                member_instance_ids=[event.instance_b_id],
                founding_speciation_event_id=event.id,
                established_at=utc_now(),
            )
            self._state.cognitive_kinds.append(new_kind)
            self._instance_to_kind[event.instance_b_id] = new_kind.kind_id
            event.new_cognitive_kind_registered = True
        elif kind_a is None:
            # Only b has a kind — create a new kind for a
            new_kind = CognitiveKindEntry(
                member_instance_ids=[event.instance_a_id],
                founding_speciation_event_id=event.id,
                established_at=utc_now(),
            )
            self._state.cognitive_kinds.append(new_kind)
            self._instance_to_kind[event.instance_a_id] = new_kind.kind_id
            event.new_cognitive_kind_registered = True
        elif kind_b is None:
            # Only a has a kind — create a new kind for b
            new_kind = CognitiveKindEntry(
                member_instance_ids=[event.instance_b_id],
                founding_speciation_event_id=event.id,
                established_at=utc_now(),
            )
            self._state.cognitive_kinds.append(new_kind)
            self._instance_to_kind[event.instance_b_id] = new_kind.kind_id
            event.new_cognitive_kind_registered = True
        # else: already in different kinds — speciation is consistent

        logger.info(
            "speciation_registered",
            instance_a=event.instance_a_id,
            instance_b=event.instance_b_id,
            new_kind=event.new_cognitive_kind_registered,
            kinds_count=len(self._state.cognitive_kinds),
        )

    def register_bridge(self, instance_a_id: str, instance_b_id: str) -> None:
        """Register an active invariant bridge between two speciated instances."""
        pair = (instance_a_id, instance_b_id)
        if pair not in self._state.active_bridge_pairs:
            self._state.active_bridge_pairs.append(pair)

    def are_speciated(self, instance_a_id: str, instance_b_id: str) -> bool:
        """Check if two instances have undergone speciation."""
        return frozenset({instance_a_id, instance_b_id}) in self._speciation_pairs

    def get_cognitive_kind(self, instance_id: str) -> str | None:
        """Return the kind_id for an instance, or None if unclassified."""
        return self._instance_to_kind.get(instance_id)

    def are_same_kind(self, instance_a_id: str, instance_b_id: str) -> bool:
        """Check if two instances belong to the same cognitive kind."""
        kind_a = self._instance_to_kind.get(instance_a_id)
        kind_b = self._instance_to_kind.get(instance_b_id)
        if kind_a is None or kind_b is None:
            return True  # Unclassified instances are assumed compatible
        return kind_a == kind_b

    def get_speciation_event(
        self, instance_a_id: str, instance_b_id: str
    ) -> SpeciationEvent | None:
        """Retrieve the speciation event between two instances."""
        pair = frozenset({instance_a_id, instance_b_id})
        return self._speciation_pairs.get(pair)

    def _find_kind(self, kind_id: str) -> CognitiveKindEntry | None:
        for kind in self._state.cognitive_kinds:
            if kind.kind_id == kind_id:
                return kind
        return None


# ─── Internal Helpers ────────────────────────────────────────


def _extract_causal_invariants(
    logos: LogosWorldModelProtocol,
    instance_id: str,
) -> list[ShareableWorldModelFragment]:
    """
    Extract causal invariants from a Logos world model.

    Invariants are the most compressed structures — schemas with high
    causal link density relative to their size. Only these can cross
    speciation boundaries.
    """
    schema_ids = logos.get_schema_ids()
    invariants: list[ShareableWorldModelFragment] = []

    for schema_id in schema_ids:
        schema = logos.get_schema(schema_id)
        if schema is None:
            continue

        # A causal invariant has high compression ratio and contains
        # invariant-level structures (most abstract possible)
        structure = schema.get("abstract_structure", schema)
        inv_list = structure.get("invariants", []) if isinstance(structure, dict) else []

        if not inv_list:
            continue

        # Build a stripped fragment — no domain context, pure structure.
        # Use schema_id as fragment_id so the caller can map converged
        # invariants back to the original schema in the fragment store.
        fragment = ShareableWorldModelFragment(
            fragment_id=schema_id,
            source_instance_id=instance_id,
            abstract_structure=_strip_domain_context(structure),
            domain_labels=[],  # Stripped for bridge exchange
            compression_ratio=schema.get("compression_ratio", 0.0),
            observations_explained=schema.get("instance_count", 0),
            description_length=schema.get("description_length", 0.0),
        )
        invariants.append(fragment)

    return invariants


def _strip_domain_context(structure: dict[str, Any]) -> dict[str, Any]:
    """
    Strip all domain-specific labels from a structure.

    Retains only: node count/topology, edge types (causal/temporal/spatial),
    symmetry class, and invariant list. Everything else is domain noise.
    """
    stripped: dict[str, Any] = {}

    # Preserve topology
    nodes = structure.get("nodes")
    if isinstance(nodes, int):
        stripped["nodes"] = nodes
    elif isinstance(nodes, list):
        stripped["nodes"] = len(nodes)

    # Preserve edge types (abstract)
    edges = structure.get("edges", [])
    if isinstance(edges, list):
        stripped["edges"] = [
            {"type": e.get("type", "unknown"), "from": e.get("from"), "to": e.get("to")}
            for e in edges
            if isinstance(e, dict)
        ]

    # Preserve symmetry class
    if "symmetry" in structure:
        stripped["symmetry"] = structure["symmetry"]

    # Preserve invariants (the core abstract content)
    if "invariants" in structure:
        stripped["invariants"] = structure["invariants"]

    return stripped


def _are_abstractly_equivalent(
    fragment_a: ShareableWorldModelFragment,
    fragment_b: ShareableWorldModelFragment,
) -> bool:
    """
    Ultra-abstract comparison: two invariants from alien-kind instances
    that match at the purest structural level.

    Checks: same invariant set, compatible symmetry, compatible topology.
    """
    struct_a = fragment_a.abstract_structure
    struct_b = fragment_b.abstract_structure

    # Invariant set comparison (primary signal)
    inv_a = set(str(i) for i in struct_a.get("invariants", []))
    inv_b = set(str(i) for i in struct_b.get("invariants", []))

    if not inv_a or not inv_b:
        return False

    # Jaccard similarity of invariant sets
    union = len(inv_a | inv_b)
    intersection = len(inv_a & inv_b)
    if union == 0:
        return False
    jaccard = intersection / union
    if jaccard < 0.7:
        return False

    # Symmetry compatibility (if both declare symmetry)
    sym_a = struct_a.get("symmetry")
    sym_b = struct_b.get("symmetry")
    if sym_a and sym_b and sym_a != sym_b:
        # Allow related symmetries
        related = {
            frozenset({"chain", "path"}),
            frozenset({"cycle", "ring"}),
            frozenset({"star", "hub"}),
            frozenset({"tree", "hierarchy"}),
        }
        pair = frozenset({sym_a, sym_b})
        if pair not in related:
            return False

    # Node count compatibility (within 50% tolerance)
    nodes_a = struct_a.get("nodes", 0)
    nodes_b = struct_b.get("nodes", 0)
    if isinstance(nodes_a, int) and isinstance(nodes_b, int) and nodes_a > 0 and nodes_b > 0:
        ratio = min(nodes_a, nodes_b) / max(nodes_a, nodes_b)
        if ratio < 0.5:
            return False

    return True


def _extract_abstract_form(
    fragment_a: ShareableWorldModelFragment,
    fragment_b: ShareableWorldModelFragment,
) -> dict[str, Any]:
    """
    Extract the shared abstract form from two convergent invariants.

    Takes the intersection of structural features present in both.
    """
    struct_a = fragment_a.abstract_structure
    struct_b = fragment_b.abstract_structure

    form: dict[str, Any] = {}

    # Shared invariants
    inv_a = set(str(i) for i in struct_a.get("invariants", []))
    inv_b = set(str(i) for i in struct_b.get("invariants", []))
    form["invariants"] = sorted(inv_a & inv_b)

    # Symmetry (prefer agreement, fall back to either)
    sym_a = struct_a.get("symmetry")
    sym_b = struct_b.get("symmetry")
    if sym_a == sym_b and sym_a is not None or sym_a:
        form["symmetry"] = sym_a
    elif sym_b:
        form["symmetry"] = sym_b

    # Node count (average)
    nodes_a = struct_a.get("nodes", 0)
    nodes_b = struct_b.get("nodes", 0)
    if isinstance(nodes_a, int) and isinstance(nodes_b, int):
        form["nodes"] = (nodes_a + nodes_b) // 2

    return form
