"""
EcodiaOS - Simula Trust Invariant Discovery (Phase 2.3)

What cannot easily be hidden or transformed in the trust model?
  -> Delegation chains cannot be made cyclic
  -> Authority cannot be escalated beyond the origin credential ceiling
  -> Trust boundaries cannot be silently crossed without mediation

Mechanism:
  1. Domain knowledge: known principals, authority ceilings, trust boundaries
  2. Z3 check: encode acyclicity, monotone-non-escalation, boundary mediation
  3. Credential integrity: credential strength ordering constraints
  4. Runtime: delegation chain depth bounds

Zero-LLM for known principals. LLM optional for new/dynamic trust patterns.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from primitives.common import new_id
from systems.simula.verification.invariant_types import (
    AuthorityPreservationInvariant,
    CredentialIntegrityInvariant,
    DelegationChainInvariant,
    EvidenceSource,
    InvariantStrength,
    TrustBoundaryInvariant,
    TrustInvariantDiscovery,
)

logger = structlog.get_logger().bind(system="simula.invariants.trust")


# -- Domain Knowledge ---------------------------------------------------------

_PRINCIPAL_AUTHORITY: dict[str, dict[str, Any]] = {
    "evo_system": {
        "max_credential_strength": 1.0,
        "role": "EVO_SYSTEM",
        "can_delegate": True,
        "delegate_ceiling": 0.95,
        "description": "Evo system - fully trusted for hypothesis formation",
    },
    "admin": {
        "max_credential_strength": 1.0,
        "role": "ADMIN",
        "can_delegate": True,
        "delegate_ceiling": 0.9,
        "description": "System administrator - fully trusted when authenticated",
    },
    "governance_vote": {
        "max_credential_strength": 0.85,
        "role": "GOVERNANCE",
        "can_delegate": False,
        "delegate_ceiling": 0.0,
        "description": "Community governance - moderately strong, cannot delegate",
    },
    "external_api": {
        "max_credential_strength": 0.7,
        "role": "EXTERNAL_API",
        "can_delegate": False,
        "delegate_ceiling": 0.0,
        "description": "External API caller - limited scope, cannot delegate",
    },
    "repair_agent": {
        "max_credential_strength": 0.8,
        "role": "REPAIR_AGENT",
        "can_delegate": False,
        "delegate_ceiling": 0.0,
        "description": "Internal repair system - elevated trust for repair ops only",
    },
    "unknown": {
        "max_credential_strength": 0.0,
        "role": "UNKNOWN",
        "can_delegate": False,
        "delegate_ceiling": 0.0,
        "description": "Unknown principal - zero trust by default",
    },
}

_TRUST_BOUNDARIES: dict[str, dict[str, Any]] = {
    "external_to_simula": {
        "from_zone": "external",
        "to_zone": "simula_core",
        "mediator": "EvoSimulaBridge",
        "crossing_requires": "authenticated_api_token",
        "strength": InvariantStrength.AXIOM,
        "description": "External callers must pass through EvoSimulaBridge",
    },
    "simula_to_filesystem": {
        "from_zone": "simula_core",
        "to_zone": "filesystem",
        "mediator": "ChangeApplicator",
        "crossing_requires": "governance_gate_passed",
        "strength": InvariantStrength.AXIOM,
        "description": "Filesystem writes only via ChangeApplicator after gate",
    },
    "simula_to_llm": {
        "from_zone": "simula_core",
        "to_zone": "llm_provider",
        "mediator": "LLMProvider",
        "crossing_requires": "sandboxed_prompt",
        "strength": InvariantStrength.STRUCTURAL,
        "description": "LLM calls must be sandboxed with constitutional constraints",
    },
    "evo_to_simula": {
        "from_zone": "evo_system",
        "to_zone": "simula_core",
        "mediator": "HypothesisConsolidator",
        "crossing_requires": "evo_service_account",
        "strength": InvariantStrength.STRUCTURAL,
        "description": "Evo proposals enter Simula via hypothesis consolidation only",
    },
}

# Credential kind ordering: higher index = stronger credential
_CREDENTIAL_ORDER: list[str] = [
    "none",
    "api_token",
    "governance_vote",
    "service_account",
    "admin_sign",
]

# Maximum delegation chain depth before acyclicity is assumed violated
_MAX_DELEGATION_DEPTH: int = 5


# -- Trust Invariant Discoverer -----------------------------------------------


class TrustInvariantDiscoverer:
    """
    Discovers trust invariants from system domain knowledge and runtime observations.

    Steps:
      1. Delegation chain acyclicity for each known principal
      2. Authority preservation: delegated strength <= origin ceiling
      3. Credential integrity: credential ordering constraints
      4. Trust boundary: boundary crossing requires mediation
      5. Z3 validation of constraint expressions
    """

    def __init__(self, check_timeout_ms: int = 5000) -> None:
        self._check_timeout_ms = check_timeout_ms
        self._log = logger

    def discover(
        self,
        observed_delegations: list[dict[str, Any]] | None = None,
        target_file: str = "",
    ) -> TrustInvariantDiscovery:
        start = time.monotonic()
        result = TrustInvariantDiscovery(target_file=target_file)

        # Step 1: Delegation chain acyclicity
        for principal_id, meta in _PRINCIPAL_AUTHORITY.items():
            if meta["can_delegate"]:
                result.delegation_chains.append(
                    self._build_delegation_invariant(principal_id, meta, target_file)
                )

        # Step 2: Authority preservation
        for principal_id, meta in _PRINCIPAL_AUTHORITY.items():
            result.authority_preservation.append(
                self._build_authority_invariant(principal_id, meta, target_file)
            )

        # Step 3: Credential integrity ordering
        result.credential_integrity.append(
            self._build_credential_ordering_invariant(target_file)
        )

        # Step 4: Trust boundaries
        for boundary_name, boundary_meta in _TRUST_BOUNDARIES.items():
            result.trust_boundaries.append(
                self._build_boundary_invariant(boundary_name, boundary_meta, target_file)
            )

        # Step 5: Runtime delegation check (if observations provided)
        if observed_delegations:
            violations = self._check_delegation_cycles(observed_delegations)
            for v in violations:
                result.violations_found += 1
                if result.delegation_chains:
                    result.delegation_chains[0].violation_witness = v

        # Step 6: Z3 validation
        z3_verified = 0
        for inv in result.delegation_chains:
            if inv.z3_expression and self._z3_valid(inv.z3_expression, inv.variable_declarations):
                inv.evidence_source = EvidenceSource.Z3_PROOF
                z3_verified += 1
        for inv in result.authority_preservation:  # type: ignore[assignment]
            if inv.z3_expression and self._z3_valid(inv.z3_expression, inv.variable_declarations):
                inv.evidence_source = EvidenceSource.Z3_PROOF
                z3_verified += 1
        for inv in result.credential_integrity:  # type: ignore[assignment]
            if inv.z3_expression and self._z3_valid(inv.z3_expression, inv.variable_declarations):
                inv.evidence_source = EvidenceSource.Z3_PROOF
                z3_verified += 1
        for inv in result.trust_boundaries:  # type: ignore[assignment]
            if inv.z3_expression and self._z3_valid(inv.z3_expression, inv.variable_declarations):
                inv.evidence_source = EvidenceSource.Z3_PROOF
                z3_verified += 1

        result.z3_verified_count = z3_verified
        result.discovery_time_ms = int((time.monotonic() - start) * 1000)

        self._log.info(
            "trust_invariants_discovered",
            file=target_file,
            delegation_chains=len(result.delegation_chains),
            authority_preservation=len(result.authority_preservation),
            credential_integrity=len(result.credential_integrity),
            trust_boundaries=len(result.trust_boundaries),
            z3_verified=z3_verified,
        )
        return result

    # -- Builders -------------------------------------------------------------

    def _build_delegation_invariant(
        self, principal_id: str, meta: dict[str, Any], target_file: str,
    ) -> DelegationChainInvariant:
        n_principals = len(_PRINCIPAL_AUTHORITY)
        var_decls = {f"p{i}": "Int" for i in range(_MAX_DELEGATION_DEPTH)}
        range_constraints = " ".join(
            f"z3.And(p{i} >= 0, p{i} < {n_principals})," for i in range(_MAX_DELEGATION_DEPTH)
        )
        distinct_args = ", ".join(f"p{i}" for i in range(_MAX_DELEGATION_DEPTH))
        z3_expr = (
            f"z3.And("
            f"{range_constraints} "
            f"z3.Distinct({distinct_args})"
            f")"
        )
        return DelegationChainInvariant(
            invariant_id=f"deleg_{new_id()[:8]}",
            description=f"{principal_id}: delegation chain must be acyclic (max depth {_MAX_DELEGATION_DEPTH})",
            strength=InvariantStrength.AXIOM,
            evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
            confidence=0.95,
            z3_expression=z3_expr,
            variable_declarations=var_decls,
            target_file=target_file,
            origin_principal=principal_id,
            max_chain_depth=_MAX_DELEGATION_DEPTH,
            chain_is_acyclic=True,
            delegation_ceiling=meta["delegate_ceiling"],
        )

    def _build_authority_invariant(
        self, principal_id: str, meta: dict[str, Any], target_file: str,
    ) -> AuthorityPreservationInvariant:
        ceiling = meta["max_credential_strength"]
        var_decls = {"origin_strength": "Real", "delegated_strength": "Real"}
        z3_expr = (
            f"z3.And("
            f"origin_strength >= 0.0, "
            f"origin_strength <= {ceiling}, "
            f"delegated_strength >= 0.0, "
            f"delegated_strength <= origin_strength"
            f")"
        )
        return AuthorityPreservationInvariant(
            invariant_id=f"auth_{new_id()[:8]}",
            description=(
                f"{principal_id}: authority cannot exceed credential ceiling "
                f"({ceiling}) through delegation"
            ),
            strength=InvariantStrength.AXIOM,
            evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
            confidence=0.95,
            z3_expression=z3_expr,
            variable_declarations=var_decls,
            target_file=target_file,
            principal_id=principal_id,
            max_authority_level=ceiling,
            authority_is_monotone_non_escalating=True,
        )

    def _build_credential_ordering_invariant(
        self, target_file: str,
    ) -> CredentialIntegrityInvariant:
        n = len(_CREDENTIAL_ORDER)
        var_decls = {"cred_kind": "Int", "cred_strength": "Real"}
        z3_expr = (
            f"z3.And("
            f"cred_kind >= 0, cred_kind < {n}, "
            f"cred_strength >= 0.0, cred_strength <= 1.0"
            f")"
        )
        return CredentialIntegrityInvariant(
            invariant_id=f"cred_{new_id()[:8]}",
            description=(
                "Credential kind has a total order: "
                "none < api_token < governance_vote < service_account < admin_sign"
            ),
            strength=InvariantStrength.STRUCTURAL,
            evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
            confidence=0.9,
            z3_expression=z3_expr,
            variable_declarations=var_decls,
            target_file=target_file,
            credential_kind_order=_CREDENTIAL_ORDER,
            strength_is_consistent_with_kind=True,
        )

    def _build_boundary_invariant(
        self, boundary_name: str, boundary_meta: dict[str, Any], target_file: str,
    ) -> TrustBoundaryInvariant:
        var_decls = {"is_crossing": "Bool", "is_mediated": "Bool"}
        z3_expr = "z3.Implies(is_crossing, is_mediated)"
        return TrustBoundaryInvariant(
            invariant_id=f"boundary_{new_id()[:8]}",
            description=boundary_meta["description"],
            strength=boundary_meta["strength"],
            evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
            confidence=0.9 if boundary_meta["strength"] == InvariantStrength.AXIOM else 0.75,
            z3_expression=z3_expr,
            variable_declarations=var_decls,
            target_file=target_file,
            from_trust_zone=boundary_meta["from_zone"],
            to_trust_zone=boundary_meta["to_zone"],
            mediator_required=boundary_meta["mediator"],
            crossing_condition=boundary_meta["crossing_requires"],
        )

    # -- Runtime Checks -------------------------------------------------------

    def _check_delegation_cycles(
        self, delegations: list[dict[str, Any]],
    ) -> list[str]:
        """Detect cycles in a delegation graph using DFS."""
        graph: dict[str, set[str]] = {}
        for d in delegations:
            frm = d.get("from", "")
            to = d.get("to", "")
            if frm and to:
                graph.setdefault(frm, set()).add(to)

        violations: list[str] = []

        def has_cycle(node: str, visited: set[str], path: list[str]) -> bool:
            if node in visited:
                cycle = " -> ".join(path + [node])
                violations.append(f"delegation cycle detected: {cycle}")
                return True
            visited.add(node)
            path.append(node)
            for neighbor in graph.get(node, set()):
                if has_cycle(neighbor, visited, path):
                    return True
            path.pop()
            visited.discard(node)
            return False

        for start_node in graph:
            has_cycle(start_node, set(), [])

        return violations

    # -- Z3 Validation --------------------------------------------------------

    def _z3_valid(self, z3_expr_code: str, variable_declarations: dict[str, str]) -> bool:
        """
        Check if a Z3 expression holds universally (negation UNSAT).
        Sandboxed: only z3 and declared symbolic variables accessible.
        """
        try:
            import z3 as z3_lib
        except ImportError:
            return False

        solver = z3_lib.Solver()
        solver.set("timeout", self._check_timeout_ms)

        z3_vars: dict[str, Any] = {}
        for name, z3_type in variable_declarations.items():
            if z3_type == "Int":
                z3_vars[name] = z3_lib.Int(name)
            elif z3_type == "Bool":
                z3_vars[name] = z3_lib.Bool(name)
            else:
                z3_vars[name] = z3_lib.Real(name)

        namespace: dict[str, Any] = {"z3": z3_lib, **z3_vars}
        try:
            expr = _sandboxed_eval(z3_expr_code, namespace)
        except Exception:
            return False

        if not isinstance(expr, z3_lib.BoolRef):
            return False

        solver.add(z3_lib.Not(expr))
        return solver.check() == z3_lib.unsat


def _sandboxed_eval(expr_code: str, namespace: dict[str, Any]) -> Any:
    """Evaluate expr_code with only the provided namespace - no builtins."""
    return eval(expr_code, {"__builtins__": {}}, namespace)  # noqa: S307
