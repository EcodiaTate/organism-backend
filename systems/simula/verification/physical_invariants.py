"""
EcodiaOS - Simula Physical Invariant Discovery (Phase 2.4)

What cannot easily be hidden or transformed by physics?
  -> Conservation laws: allocated + available == total (budgets, counts)
  -> Process bounds: operations cannot exceed hardware limits
  -> Physical feasibility: timing observations below domain minimum are impossible

Mechanism:
  1. Domain knowledge: known conservation laws, process bounds
  2. Z3 check: budget conservation, hash chain non-decrease, count conservation
  3. Statistical calibration: physical_min = max(domain_min, obs_mean - 3*obs_stdev)
  4. Feasibility violation detection: impossibly fast / impossibly slow observations

Zero-LLM for bounds and conservation. TimingProfiler data optional for calibration.
"""

from __future__ import annotations

import time
from statistics import mean, stdev
from typing import Any

import structlog

from primitives.common import new_id
from systems.simula.verification.invariant_types import (
    ConservationConstraintInvariant,
    EvidenceSource,
    InvariantStrength,
    PhysicalFeasibilityInvariant,
    PhysicalInvariantDiscovery,
    ProcessBoundInvariant,
    ResourceConservationInvariant,
)

logger = structlog.get_logger().bind(system="simula.invariants.physical")


# -- Domain Knowledge ---------------------------------------------------------

# z3 conservation law expression strings — evaluated at runtime in _z3_valid
_BUDGET_Z3 = "z3.And(allocated >= 0.0, available >= 0.0, total > 0.0, allocated + available == total)"
_PROPOSAL_COUNT_Z3 = "z3.And(active >= 0, terminal_c >= 0, total_p >= 0, active + terminal_c == total_p)"
_HASH_CHAIN_Z3 = "z3.And(chain_len >= 0, chain_len_next >= chain_len)"
_HEADROOM_Z3 = "z3.And(headroom >= 0.0, headroom <= 100.0)"

_CONSERVATION_LAWS: dict[str, dict[str, Any]] = {
    "budget_allocation": {
        "conserved_quantity": "total_budget",
        "parts": ["allocated_budget", "available_budget"],
        "relation": "sum",
        "strength": InvariantStrength.AXIOM,
        "description": "allocated_budget + available_budget == total_budget at all times",
        "z3_vars": {"allocated": "Real", "available": "Real", "total": "Real"},
        "z3_expr": _BUDGET_Z3,
    },
    "proposal_count_conservation": {
        "conserved_quantity": "total_proposals",
        "parts": ["active_proposals", "terminal_proposals"],
        "relation": "sum",
        "strength": InvariantStrength.STRUCTURAL,
        "description": "active_proposals + terminal_proposals == total_proposals",
        "z3_vars": {"active": "Int", "terminal_c": "Int", "total_p": "Int"},
        "z3_expr": _PROPOSAL_COUNT_Z3,
    },
    "hash_chain_non_decrease": {
        "conserved_quantity": "chain_length",
        "parts": ["chain_length"],
        "relation": "monotone",
        "strength": InvariantStrength.AXIOM,
        "description": "Hash chain length never decreases — append-only log",
        "z3_vars": {"chain_len": "Int", "chain_len_next": "Int"},
        "z3_expr": _HASH_CHAIN_Z3,
    },
    "resource_headroom": {
        "conserved_quantity": "budget_headroom_percent",
        "parts": ["budget_headroom_percent"],
        "relation": "bounded",
        "strength": InvariantStrength.STRUCTURAL,
        "description": "Budget headroom is in [0, 100] percent",
        "z3_vars": {"headroom": "Real"},
        "z3_expr": _HEADROOM_Z3,
    },
}

_PROCESS_BOUNDS: dict[str, dict[str, Any]] = {
    "risk_score": {
        "lower": 0.0,
        "upper": 1.0,
        "unit": "dimensionless",
        "strength": InvariantStrength.AXIOM,
        "description": "Risk score is a probability in [0, 1]",
    },
    "simulation_time_ms": {
        "lower": 0.0,
        "upper": 30_000.0,
        "unit": "ms",
        "strength": InvariantStrength.STRUCTURAL,
        "description": "Simulation must complete within 30s deadline",
    },
    "application_time_ms": {
        "lower": 0.0,
        "upper": 60_000.0,
        "unit": "ms",
        "strength": InvariantStrength.STRUCTURAL,
        "description": "Application phase must complete within 60s deadline",
    },
    "verification_time_ms": {
        "lower": 0.0,
        "upper": 120_000.0,
        "unit": "ms",
        "strength": InvariantStrength.STRUCTURAL,
        "description": "Verification phase must complete within 120s deadline",
    },
    "alignment_score": {
        "lower": -1.0,
        "upper": 1.0,
        "unit": "dimensionless",
        "strength": InvariantStrength.AXIOM,
        "description": "Constitutional alignment score is in [-1, 1]",
    },
    "confidence": {
        "lower": 0.0,
        "upper": 1.0,
        "unit": "dimensionless",
        "strength": InvariantStrength.AXIOM,
        "description": "Confidence is a probability in [0, 1]",
    },
    "evidence_strength": {
        "lower": 0.0,
        "upper": 1.0,
        "unit": "dimensionless",
        "strength": InvariantStrength.AXIOM,
        "description": "Evidence strength is in [0, 1]",
    },
    "blast_radius_files": {
        "lower": 0.0,
        "upper": 10_000.0,
        "unit": "files",
        "strength": InvariantStrength.STRUCTURAL,
        "description": "Blast radius in files is bounded by codebase size",
    },
    "rollback_time_ms": {
        "lower": 0.0,
        "upper": 60_000.0,
        "unit": "ms",
        "strength": InvariantStrength.STRUCTURAL,
        "description": "Rollback must complete within 60s",
    },
    "z3_check_time_ms": {
        "lower": 0.0,
        "upper": 5_000.0,
        "unit": "ms",
        "strength": InvariantStrength.STRUCTURAL,
        "description": "Z3 solver check must complete within 5s timeout",
    },
}

_RESOURCE_QUOTAS: dict[str, dict[str, Any]] = {
    "llm_token_budget": {
        "resource": "llm_tokens",
        "max_per_proposal": 50_000,
        "unit": "tokens",
        "description": "LLM token budget per proposal is bounded",
    },
    "file_write_quota": {
        "resource": "filesystem_writes",
        "max_per_proposal": 500,
        "unit": "files",
        "description": "Maximum files written per proposal (blast radius cap)",
    },
    "z3_solver_calls": {
        "resource": "z3_calls",
        "max_per_proposal": 200,
        "unit": "solver_calls",
        "description": "Z3 solver calls per proposal bounded to prevent runaway",
    },
}

# Absolute minimum physically possible times (hardware lower bound, nanoseconds)
_DOMAIN_PHYSICAL_MINIMUMS_NS: dict[str, float] = {
    "ast_parse": 1_000.0,
    "z3_check": 10_000.0,
    "file_read": 100_000.0,
    "file_write": 100_000.0,
    "llm_call": 50_000_000.0,
    "simulation": 1_000_000.0,
    "full_pipeline": 10_000_000.0,
}


# -- Physical Invariant Discoverer --------------------------------------------


class PhysicalInvariantDiscoverer:
    """
    Discovers physical invariants from domain knowledge and timing observations.

    Steps:
      1. Conservation constraints from domain knowledge
      2. Process bounds for all known parameters
      3. Resource conservation quotas
      4. Physical feasibility from TimingProfiler observations (statistical calibration)
      5. Z3 validation of conservation expressions
    """

    def __init__(self, check_timeout_ms: int = 5000) -> None:
        self._check_timeout_ms = check_timeout_ms
        self._log = logger

    def discover(
        self,
        timing_observations: dict[str, list[float]] | None = None,
        target_file: str = "",
    ) -> PhysicalInvariantDiscovery:
        """
        Args:
            timing_observations: {operation_name: [duration_ns, ...]} from TimingProfiler.
                                  Used to calibrate physical feasibility bounds.
            target_file: Source file context for invariant records.
        """
        start = time.monotonic()
        result = PhysicalInvariantDiscovery(target_file=target_file)

        for law_name, law_meta in _CONSERVATION_LAWS.items():
            result.conservation_constraints.append(
                self._build_conservation_invariant(law_name, law_meta, target_file)
            )

        for param_name, bounds_meta in _PROCESS_BOUNDS.items():
            result.process_bounds.append(
                self._build_process_bound_invariant(param_name, bounds_meta, target_file)
            )

        for resource_name, quota_meta in _RESOURCE_QUOTAS.items():
            result.resource_conservation.append(
                self._build_resource_invariant(resource_name, quota_meta, target_file)
            )

        if timing_observations:
            for op_name, obs_ns in timing_observations.items():
                if len(obs_ns) >= 3:
                    result.feasibility_checks.append(
                        self._build_feasibility_from_obs(op_name, obs_ns, target_file)
                    )
        else:
            for op_name, domain_min_ns in _DOMAIN_PHYSICAL_MINIMUMS_NS.items():
                result.feasibility_checks.append(
                    self._build_feasibility_from_domain(op_name, domain_min_ns, target_file)
                )

        z3_verified = 0
        for inv in result.conservation_constraints:
            if inv.z3_expression and self._z3_valid(inv.z3_expression, inv.variable_declarations):
                inv.evidence_source = EvidenceSource.Z3_PROOF
                inv.strength = InvariantStrength.AXIOM
                z3_verified += 1
        for inv in result.process_bounds:  # type: ignore[assignment]
            if inv.z3_expression and self._z3_valid(inv.z3_expression, inv.variable_declarations):
                inv.evidence_source = EvidenceSource.Z3_PROOF
                z3_verified += 1

        result.z3_verified_count = z3_verified
        result.discovery_time_ms = int((time.monotonic() - start) * 1000)

        self._log.info(
            "physical_invariants_discovered",
            file=target_file,
            conservation_constraints=len(result.conservation_constraints),
            process_bounds=len(result.process_bounds),
            resource_conservation=len(result.resource_conservation),
            feasibility_checks=len(result.feasibility_checks),
            z3_verified=z3_verified,
        )
        return result

    # -- Builders -------------------------------------------------------------

    def _build_conservation_invariant(
        self, law_name: str, law_meta: dict[str, Any], target_file: str,
    ) -> ConservationConstraintInvariant:
        return ConservationConstraintInvariant(
            invariant_id=f"conserve_{new_id()[:8]}",
            description=law_meta["description"],
            strength=law_meta["strength"],
            evidence_source=EvidenceSource.CONSERVATION_LAW,
            confidence=0.99 if law_meta["strength"] == InvariantStrength.AXIOM else 0.90,
            z3_expression=law_meta.get("z3_expr", ""),
            variable_declarations=law_meta.get("z3_vars", {}),
            target_file=target_file,
            conserved_quantity=law_meta["conserved_quantity"],
            conservation_equation=law_meta["description"],
            parts=law_meta["parts"],
        )

    def _build_process_bound_invariant(
        self, param_name: str, bounds_meta: dict[str, Any], target_file: str,
    ) -> ProcessBoundInvariant:
        lo = bounds_meta["lower"]
        hi = bounds_meta["upper"]
        var_decls = {param_name: "Real"}
        z3_expr = f"z3.And({param_name} >= {lo}, {param_name} <= {hi})"
        return ProcessBoundInvariant(
            invariant_id=f"bound_{new_id()[:8]}",
            description=bounds_meta["description"],
            strength=bounds_meta["strength"],
            evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
            confidence=0.95 if bounds_meta["strength"] == InvariantStrength.AXIOM else 0.85,
            z3_expression=z3_expr,
            variable_declarations=var_decls,
            target_file=target_file,
            parameter_name=param_name,
            lower_bound=lo,
            upper_bound=hi,
            unit=bounds_meta["unit"],
        )

    def _build_resource_invariant(
        self, resource_name: str, quota_meta: dict[str, Any], target_file: str,
    ) -> ResourceConservationInvariant:
        max_val = quota_meta["max_per_proposal"]
        res_var = quota_meta["resource"]
        var_decls = {res_var: "Int"}
        z3_expr = f"z3.And({res_var} >= 0, {res_var} <= {max_val})"
        return ResourceConservationInvariant(
            invariant_id=f"resource_{new_id()[:8]}",
            description=quota_meta["description"],
            strength=InvariantStrength.STRUCTURAL,
            evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
            confidence=0.85,
            z3_expression=z3_expr,
            variable_declarations=var_decls,
            target_file=target_file,
            resource_name=res_var,
            max_per_proposal=max_val,
            unit=quota_meta["unit"],
        )

    def _build_feasibility_from_obs(
        self, op_name: str, obs_ns: list[float], target_file: str,
    ) -> PhysicalFeasibilityInvariant:
        """Calibrate physical minimum: max(domain_min, mu - 3*sigma)."""
        obs_mean = mean(obs_ns)
        obs_stdev = stdev(obs_ns) if len(obs_ns) > 1 else 0.0
        domain_min = _DOMAIN_PHYSICAL_MINIMUMS_NS.get(op_name, 0.0)
        physical_min = max(domain_min, obs_mean - 3.0 * obs_stdev)

        impossible = [o for o in obs_ns if o < physical_min * 0.5]
        violation_witness = (
            f"{len(impossible)} observations below half physical minimum "
            f"({physical_min:.0f} ns)"
            if impossible else ""
        )

        return PhysicalFeasibilityInvariant(
            invariant_id=f"feasible_{new_id()[:8]}",
            description=(
                f"{op_name}: physical minimum is {physical_min:.0f} ns "
                f"(calibrated from {len(obs_ns)} observations)"
            ),
            strength=InvariantStrength.STATISTICAL,
            evidence_source=EvidenceSource.STATISTICAL_BOUND,
            confidence=min(0.99, 0.9 + 0.01 * min(9, len(obs_ns) - 3)),
            violation_witness=violation_witness,
            target_file=target_file,
            operation_name=op_name,
            physical_minimum_ns=physical_min,
            observed_mean_ns=obs_mean,
            observed_stdev_ns=obs_stdev,
            sample_count=len(obs_ns),
            domain_minimum_ns=domain_min,
        )

    def _build_feasibility_from_domain(
        self, op_name: str, domain_min_ns: float, target_file: str,
    ) -> PhysicalFeasibilityInvariant:
        """Domain-knowledge-only feasibility (no runtime data)."""
        return PhysicalFeasibilityInvariant(
            invariant_id=f"feasible_{new_id()[:8]}",
            description=(
                f"{op_name}: physical minimum is {domain_min_ns:.0f} ns "
                f"(domain knowledge, no runtime calibration)"
            ),
            strength=InvariantStrength.STRUCTURAL,
            evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
            confidence=0.75,
            target_file=target_file,
            operation_name=op_name,
            physical_minimum_ns=domain_min_ns,
            observed_mean_ns=0.0,
            observed_stdev_ns=0.0,
            sample_count=0,
            domain_minimum_ns=domain_min_ns,
        )

    # -- Z3 Validation --------------------------------------------------------

    def _z3_valid(self, z3_expr_code: str, variable_declarations: dict[str, str]) -> bool:
        """
        Check if a Z3 expression holds universally (negation is UNSAT).
        Sandboxed namespace: only z3 module and declared symbolic variables are accessible.
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
            expr = _run_z3_expr(z3_expr_code, namespace)
        except Exception:
            return False

        if not isinstance(expr, z3_lib.BoolRef):
            return False

        solver.add(z3_lib.Not(expr))
        return solver.check() == z3_lib.unsat


def _run_z3_expr(expr_code: str, namespace: dict[str, Any]) -> Any:
    """Run a Z3 expression string in a namespace with no stdlib builtins."""
    return eval(expr_code, {"__builtins__": {}}, namespace)  # noqa: S307
