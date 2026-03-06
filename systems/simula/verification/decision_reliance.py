"""
EcodiaOS -- Simula Phase 3 Decision Reliance Analyser

Three analysers answer: "Where does code rely on belief instead of verification?"

  StateInterpretationAnalyser  -- cached authority, inferred identity,
                                   remembered protocol context
  SourceOfTruthAnalyser        -- live vs. stored divergence, inferred truth,
                                   origin verification gaps
  SessionContinuityAnalyser    -- session assumptions, narrative continuity,
                                   workflow TOCTOU preconditions

Architecture:
  - Zero-LLM by default: O(N) AST walk + domain knowledge tables
  - Z3 validation: every finding with a z3_expression runs solver.add(Not(expr))
    UNSAT => evidence upgraded to Z3_PROOF, strength upgraded to AXIOM
  - Same _sandboxed_eval pattern as trust_invariants.py (no builtins)
"""

from __future__ import annotations

import ast
import time
from typing import Any

import structlog

from primitives.common import new_id
from systems.simula.verification.decision_reliance_types import (
    CachedAuthorityReliance,
    InferredIdentityReliance,
    InferredTruthGap,
    LiveVsStoredTruthGap,
    NarrativeContinuityReliance,
    OriginVerificationGap,
    RelianceRisk,
    RememberedProtocolReliance,
    SessionAssumptionReliance,
    SessionContinuityDiscovery,
    SourceOfTruthDiscovery,
    StateInterpretationDiscovery,
    VerificationStyle,
    WorkflowPreconditionReliance,
)
from systems.simula.verification.invariant_types import (
    EvidenceSource,
    InvariantStrength,
)

logger = structlog.get_logger().bind(system="simula.reliance")


# ── Sandboxed Z3 eval (same pattern as trust_invariants.py) ──────────────────


def _sandboxed_eval(expr_code: str, namespace: dict[str, Any]) -> Any:
    """Execute a Z3 expression string in a namespace with no builtins."""
    return eval(expr_code, {"__builtins__": {}}, namespace)  # noqa: S307


def _z3_check(
    z3_expression: str,
    variable_declarations: dict[str, str],
    timeout_ms: int,
) -> bool:
    """
    Returns True if the negation of z3_expression is UNSAT (finding is provable).
    Silently returns False on any error or timeout.
    """
    if not z3_expression:
        return False
    try:
        import z3  # type: ignore[import]

        namespace: dict[str, Any] = {"z3": z3}
        for name, kind in variable_declarations.items():
            if kind == "Int":
                namespace[name] = z3.Int(name)
            elif kind == "Real":
                namespace[name] = z3.Real(name)
            elif kind == "Bool":
                namespace[name] = z3.Bool(name)

        expr = _sandboxed_eval(z3_expression, namespace)
        solver = z3.Solver()
        solver.set("timeout", timeout_ms)
        solver.add(z3.Not(expr))
        return solver.check() == z3.unsat
    except Exception:
        return False


# ── Domain knowledge tables ───────────────────────────────────────────────────

# Attribute names that suggest a cached identity or authority field is being read
_CACHED_IDENTITY_VARS: set[str] = {
    "principal_role",
    "principal_id",
    "credential_strength",
    "session_role",
    "cached_role",
    "cached_identity",
    "auth_context",
    "identity_context",
    "session_context",
    "user_role",
    "token_role",
    "delegated_role",
}

# Attribute / header names that suggest identity is inferred from a request signal
_INFERRED_IDENTITY_SIGNALS: set[str] = {
    "remote_addr",
    "x_forwarded_for",
    "x_user_id",
    "x_admin_token",
    "client_ip",
    "forwarded_for",
    "http_x_real_ip",
    "request_user",
    "request_principal",
}

# Known caches where decisions are made against stored-state authority claims
_KNOWN_BELIEF_CACHES: list[dict[str, Any]] = [
    {
        "authority_claim": "principal.role == granted_role",
        "cache_location": "IdentityContext (in-memory, per-request)",
        "cache_ttl_seconds": 0,
        "authoritative_source": "principal_registry.get_role(principal_id)",
        "gated_decision": "allow_governance_proposal / allow_admin_operation",
        "reliance_risk": RelianceRisk.HIGH,
        "remediation": (
            "Add a re-validation call to principal_registry at each sensitive operation. "
            "Cache with a short TTL (e.g. 60 s) and force a refresh on CRITICAL decisions."
        ),
        "z3_expression": (
            "z3.Implies(decision_made, z3.Or(cache_age_seconds <= max_ttl, re_verified))"
        ),
        "variable_declarations": {
            "decision_made": "Bool",
            "cache_age_seconds": "Int",
            "max_ttl": "Int",
            "re_verified": "Bool",
        },
    },
    {
        "authority_claim": "constitution invariants are current",
        "cache_location": "constitution_cache (TTL 86400 s)",
        "cache_ttl_seconds": 86400,
        "authoritative_source": "ConstitutionLoader.load_live()",
        "gated_decision": "risk threshold evaluation in ChangeSimulator",
        "reliance_risk": RelianceRisk.MEDIUM,
        "remediation": (
            "On governance proposal that modifies the constitution, "
            "invalidate constitution_cache immediately "
            "(write-through already intended -- verify it fires)."
        ),
        "z3_expression": "z3.Implies(constitution_changed, cache_invalidated)",
        "variable_declarations": {
            "constitution_changed": "Bool",
            "cache_invalidated": "Bool",
        },
    },
    {
        "authority_claim": "incremental verification result is current for this file hash",
        "cache_location": "incremental_verification_hot (TTL 3600 s)",
        "cache_ttl_seconds": 3600,
        "authoritative_source": "IncrementalVerificationEngine (re-run)",
        "gated_decision": "skip re-verification in SimulaService fast path",
        "reliance_risk": RelianceRisk.HIGH,
        "remediation": (
            "Cache key must include content_hash of the target file. "
            "Any write to that file must invalidate the key before the next read."
        ),
        "z3_expression": "z3.Implies(file_modified, cache_entry_invalidated)",
        "variable_declarations": {
            "file_modified": "Bool",
            "cache_entry_invalidated": "Bool",
        },
    },
]

# Known live-vs-stored truth gaps in the domain
_KNOWN_TRUTH_GAPS: list[dict[str, Any]] = [
    {
        "live_truth_source": "principal_registry (authoritative role store)",
        "stored_truth_location": "IdentityContext.principal_role (in-memory)",
        "update_mechanism": "refreshed on re-authentication only",
        "max_staleness_seconds": 60,
        "gated_decision": "permission gates throughout proposal pipeline",
        "reliance_risk": RelianceRisk.HIGH,
        "remediation": "Add periodic re-fetch from principal_registry or short TTL on role field.",
        "z3_expression": "z3.Implies(decision_fired, role_age_seconds <= max_staleness)",
        "variable_declarations": {
            "decision_fired": "Bool",
            "role_age_seconds": "Int",
            "max_staleness": "Int",
        },
    },
    {
        "live_truth_source": "budget ledger (committed spend)",
        "stored_truth_location": "proposal.estimated_cost (snapshot at validation)",
        "update_mechanism": "not updated after VALIDATE stage",
        "max_staleness_seconds": 0,
        "gated_decision": "APPLY stage budget deduction",
        "reliance_risk": RelianceRisk.CRITICAL,
        "remediation": (
            "Re-read live budget_available at APPLY entry "
            "using SELECT FOR UPDATE or equivalent."
        ),
        "z3_expression": "z3.Implies(apply_stage_entered, budget_re_verified_at_apply)",
        "variable_declarations": {
            "apply_stage_entered": "Bool",
            "budget_re_verified_at_apply": "Bool",
        },
    },
    {
        "live_truth_source": "hash chain (append-only evolution log)",
        "stored_truth_location": "chain_position counter (in-memory)",
        "update_mechanism": "incremented on each RECORD -- not re-verified from persistent store",
        "max_staleness_seconds": 0,
        "gated_decision": "HashChainEntry.previous_hash linkage in RECORD stage",
        "reliance_risk": RelianceRisk.CRITICAL,
        "remediation": (
            "Read chain tail from persistent store before appending; "
            "verify hash linkage."
        ),
        "z3_expression": "z3.Implies(record_stage_entered, chain_tail_re_read_from_store)",
        "variable_declarations": {
            "record_stage_entered": "Bool",
            "chain_tail_re_read_from_store": "Bool",
        },
    },
]

# Known origin verification gaps
_KNOWN_ORIGIN_GAPS: list[dict[str, Any]] = [
    {
        "origin_claim": "EvolutionProposal.submitted_by identifies the actual requester",
        "claim_location": "EvolutionProposal.submitted_by (caller-set field)",
        "claim_setter": "API caller (external, unverified)",
        "verification_mechanism": "verify GovernanceCredential.signed_payload_hash at intake",
        "is_verified_at_use": False,
        "gated_decision": "authority ceiling check at VALIDATE stage",
        "reliance_risk": RelianceRisk.CRITICAL,
        "remediation": (
            "At proposal intake, require a GovernanceCredential that signs the submitted_by "
            "field. Verify signature before persisting the proposal."
        ),
        "z3_expression": "z3.Implies(proposal_validated, submitted_by_signature_verified)",
        "variable_declarations": {
            "proposal_validated": "Bool",
            "submitted_by_signature_verified": "Bool",
        },
    },
    {
        "origin_claim": "repair_agent actions originate from the legitimate repair subsystem",
        "claim_location": "RepairContext.agent_id (set by repair agent on construction)",
        "claim_setter": "repair agent itself",
        "verification_mechanism": "HealthChecker.validate_repair_agent_credential()",
        "is_verified_at_use": True,   # gap is closed -- documenting the closed case
        "gated_decision": "allow FORBIDDEN_WRITE_PATH bypass for repair agent",
        "reliance_risk": RelianceRisk.HIGH,
        "remediation": "Already verified -- ensure verification precedes path-bypass check.",
        "z3_expression": "",
        "variable_declarations": {},
    },
]

# Known EcodiaOS sessions and the persistent assumptions they require
_KNOWN_SESSIONS: list[dict[str, Any]] = [
    {
        "session_type": "proposal_lifecycle",
        "persistent_assumption": "identity.credential_strength >= required_credential_floor",
        "assumption_established_at": "VALIDATE stage (credential floor check)",
        "relied_on_at_steps": ["SIMULATE", "GATE", "APPLY", "VERIFY", "RECORD"],
        "invalidating_events": [
            "principal credential revoked between stages",
            "governance quorum drops below threshold after GATE",
        ],
        "revalidation_mechanism": "",
        "fsm_name": "proposal_lifecycle",
        "strength": InvariantStrength.STRUCTURAL,
        "reliance_risk": RelianceRisk.HIGH,
        "remediation": (
            "Re-check credential_strength at APPLY and RECORD stage entry. "
            "Add a short-circuit abort path if credentials changed."
        ),
        "z3_expression": "z3.Implies(proposal_at_record_stage, credential_still_valid)",
        "variable_declarations": {
            "proposal_at_record_stage": "Bool",
            "credential_still_valid": "Bool",
        },
    },
    {
        "session_type": "governance_vote",
        "persistent_assumption": "quorum_count >= quorum_threshold at all vote stages",
        "assumption_established_at": "GATE stage quorum check",
        "relied_on_at_steps": ["APPLY", "RECORD"],
        "invalidating_events": [
            "voter resigns between GATE and APPLY",
            "governance credential expires in the stage gap",
        ],
        "revalidation_mechanism": "",
        "fsm_name": "proposal_lifecycle",
        "strength": InvariantStrength.STRUCTURAL,
        "reliance_risk": RelianceRisk.CRITICAL,
        "remediation": (
            "Re-check live quorum at APPLY entry using a quorum_snapshot_timestamp. "
            "Block if snapshot is older than governance_quorum_recheck_window_seconds."
        ),
        "z3_expression": (
            "z3.Implies(z3.And(quorum_checked, action_applied), quorum_still_valid)"
        ),
        "variable_declarations": {
            "quorum_checked": "Bool",
            "action_applied": "Bool",
            "quorum_still_valid": "Bool",
        },
    },
    {
        "session_type": "repair_session",
        "persistent_assumption": "repair_agent has REPAIR_AGENT credentials for duration",
        "assumption_established_at": "session initialisation",
        "relied_on_at_steps": ["diagnosing", "planning", "applying", "verifying"],
        "invalidating_events": [
            "repair agent token revoked mid-session",
            "repair scope changed by admin during session",
        ],
        "revalidation_mechanism": "HealthChecker validates repair_agent identity at each phase",
        "fsm_name": "repair_phase",
        "strength": InvariantStrength.BEHAVIORAL,
        "reliance_risk": RelianceRisk.MEDIUM,
        "remediation": "Already has partial revalidation -- extend to cover scope changes.",
        "z3_expression": "",
        "variable_declarations": {},
    },
]

# Known narrative continuity reliances
_KNOWN_NARRATIVE_RELIANCES: list[dict[str, Any]] = [
    {
        "narrative_source": "HashChain evolution log",
        "integrity_mechanism": "SHA-256 chain hash (chain_hash field)",
        "is_integrity_verified_before_act": False,
        "gated_decision": "RECORD stage: appending new HashChainEntry",
        "attack_surface": (
            "Insert a forged HashChainEntry between verification and record stages "
            "to shift chain_position without breaking the running hash."
        ),
        "reliance_risk": RelianceRisk.CRITICAL,
        "remediation": (
            "Verify chain_tail.chain_hash == expected_hash immediately before RECORD "
            "using SELECT FOR UPDATE on the chain tail row."
        ),
        "strength": InvariantStrength.AXIOM,
        "z3_expression": (
            "z3.Implies(record_action_taken, chain_integrity_verified_before_record)"
        ),
        "variable_declarations": {
            "record_action_taken": "Bool",
            "chain_integrity_verified_before_record": "Bool",
        },
    },
    {
        "narrative_source": "audit trail (AuditEvent records)",
        "integrity_mechanism": "",
        "is_integrity_verified_before_act": False,
        "gated_decision": "compliance report generation from audit trail",
        "attack_surface": "Delete or modify audit records before compliance report is generated.",
        "reliance_risk": RelianceRisk.HIGH,
        "remediation": (
            "Compute a running Merkle root over audit records. "
            "Verify root before generating compliance reports."
        ),
        "strength": InvariantStrength.STRUCTURAL,
        "z3_expression": "",
        "variable_declarations": {},
    },
]

# Known TOCTOU workflow preconditions
_KNOWN_TOCTOU: list[dict[str, Any]] = [
    {
        "precondition": "budget_available >= proposal_estimated_cost",
        "checked_at_step": "VALIDATE",
        "consumed_at_step": "APPLY",
        "toctou_scenario": "concurrent proposal approved and applied between VALIDATE and APPLY",
        "is_rechecked_before_consume": False,
        "reliance_risk": RelianceRisk.CRITICAL,
        "remediation": (
            "Acquire a budget lock at VALIDATE and hold until APPLY completes. "
            "Alternatively, use optimistic concurrency: re-read budget at APPLY "
            "and abort if balance changed."
        ),
        "z3_expression": "z3.Implies(apply_fires, budget_re_read_at_apply)",
        "variable_declarations": {
            "apply_fires": "Bool",
            "budget_re_read_at_apply": "Bool",
        },
    },
    {
        "precondition": "governance_quorum_count >= quorum_threshold",
        "checked_at_step": "GATE",
        "consumed_at_step": "APPLY",
        "toctou_scenario": "governance voter resigns or credential expires between GATE and APPLY",
        "is_rechecked_before_consume": False,
        "reliance_risk": RelianceRisk.CRITICAL,
        "remediation": (
            "Re-check live quorum count at APPLY entry. "
            "Add quorum_snapshot_timestamp and reject if older than governance_recheck_window."
        ),
        "z3_expression": "z3.Implies(apply_fires, quorum_re_checked_at_apply)",
        "variable_declarations": {
            "apply_fires": "Bool",
            "quorum_re_checked_at_apply": "Bool",
        },
    },
    {
        "precondition": "rollback_snapshot exists and is uncorrupted",
        "checked_at_step": "APPLY (snapshot created)",
        "consumed_at_step": "VERIFY (rollback triggered if verify fails)",
        "toctou_scenario": "snapshot storage fails or is partially written between APPLY and VERIFY",
        "is_rechecked_before_consume": False,
        "reliance_risk": RelianceRisk.HIGH,
        "remediation": (
            "Verify snapshot checksum at VERIFY entry before proceeding. "
            "Abort if snapshot is missing or hash mismatch."
        ),
        "z3_expression": "z3.Implies(rollback_triggered, snapshot_integrity_verified)",
        "variable_declarations": {
            "rollback_triggered": "Bool",
            "snapshot_integrity_verified": "Bool",
        },
    },
]


# ── AST Helpers ───────────────────────────────────────────────────────────────


class _CachedReadVisitor(ast.NodeVisitor):
    """
    Walk an AST collecting attribute reads on names that suggest cached
    identity or authority state (e.g. identity_context.principal_role).

    Produces records: {function, attr_var, attr_name, line}
    """

    def __init__(self, target_functions: set[str] | None = None) -> None:
        self.findings: list[dict[str, Any]] = []
        self._current_function: str = ""
        self._target = target_functions

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self._current_function
        self._current_function = node.name
        self.generic_visit(node)
        self._current_function = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self._target and self._current_function not in self._target:
            self.generic_visit(node)
            return
        if node.attr in _CACHED_IDENTITY_VARS:
            var_name = ""
            if isinstance(node.value, ast.Name):
                var_name = node.value.id
            elif isinstance(node.value, ast.Attribute):
                var_name = node.value.attr
            self.findings.append({
                "function": self._current_function,
                "attr_var": var_name,
                "attr_name": node.attr,
                "line": getattr(node, "lineno", 0),
            })
        self.generic_visit(node)


class _InferredIdentityVisitor(ast.NodeVisitor):
    """
    Walk an AST collecting reads of headers or request attributes that suggest
    inferred identity (e.g. request.remote_addr, headers.get("X-User-ID")).

    Produces records: {function, signal, line}
    """

    def __init__(self, target_functions: set[str] | None = None) -> None:
        self.findings: list[dict[str, Any]] = []
        self._current_function: str = ""
        self._target = target_functions

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self._current_function
        self._current_function = node.name
        self.generic_visit(node)
        self._current_function = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self._target and self._current_function not in self._target:
            self.generic_visit(node)
            return
        attr_lower = node.attr.lower().replace("-", "_")
        if attr_lower in _INFERRED_IDENTITY_SIGNALS:
            self.findings.append({
                "function": self._current_function,
                "signal": node.attr,
                "line": getattr(node, "lineno", 0),
            })
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Catch .headers.get("X-User-ID") style calls."""
        if self._target and self._current_function not in self._target:
            self.generic_visit(node)
            return
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    key_lower = arg.value.lower().replace("-", "_")
                    if any(sig in key_lower for sig in _INFERRED_IDENTITY_SIGNALS):
                        self.findings.append({
                            "function": self._current_function,
                            "signal": arg.value,
                            "line": getattr(node, "lineno", 0),
                        })
        self.generic_visit(node)


# ── Sub-Analysers ─────────────────────────────────────────────────────────────


class StateInterpretationAnalyser:
    """
    Phase 3.1 -- State Interpretation.

    Finds where decisions depend on stored/cached state without re-validating
    the underlying assumption:
      - Cached authority (role/permission cached in-memory)
      - Inferred identity (header/signal used instead of credential)
      - Remembered protocol context (prior-step guarantee reused without re-check)
    """

    def __init__(self, check_timeout_ms: int = 5000) -> None:
        self._timeout_ms = check_timeout_ms
        self._log = logger.bind(analyser="state_interpretation")

    def analyse(
        self,
        python_source: str = "",
        target_file: str = "",
        target_functions: list[str] | None = None,
        include_domain_knowledge: bool = True,
    ) -> StateInterpretationDiscovery:
        t0 = time.monotonic()
        result = StateInterpretationDiscovery(
            target_file=target_file,
            target_functions=target_functions or [],
        )
        tf_set: set[str] | None = set(target_functions) if target_functions else None

        # AST scan: find cached identity reads and inferred identity signals
        if python_source:
            try:
                tree = ast.parse(python_source)
                cached_visitor = _CachedReadVisitor(tf_set)
                cached_visitor.visit(tree)
                inferred_visitor = _InferredIdentityVisitor(tf_set)
                inferred_visitor.visit(tree)
            except SyntaxError:
                cached_visitor = _CachedReadVisitor()
                inferred_visitor = _InferredIdentityVisitor()

            seen_attrs: set[str] = set()
            for finding in cached_visitor.findings:
                key = f"{finding['attr_var']}.{finding['attr_name']}"
                if key in seen_attrs:
                    continue
                seen_attrs.add(key)
                result.cached_authority.append(CachedAuthorityReliance(
                    invariant_id=new_id("reliance"),
                    description=(
                        f"Decision reads cached identity field "
                        f"'{finding['attr_var']}.{finding['attr_name']}' "
                        f"without re-verifying against authoritative source"
                    ),
                    strength=InvariantStrength.BEHAVIORAL,
                    evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
                    confidence=0.7,
                    target_file=target_file,
                    target_function=finding["function"],
                    authority_claim=f"{finding['attr_var']}.{finding['attr_name']}",
                    cache_location=f"local variable / attribute on {finding['attr_var']}",
                    cache_ttl_seconds=0,
                    authoritative_source="(not determined from AST)",
                    gated_decision=f"decision in {finding['function']}",
                    verification_style=VerificationStyle.CACHED,
                    reliance_risk=RelianceRisk.HIGH,
                    remediation="Verify against authoritative source at decision point.",
                    observed_at_function=finding["function"],
                    observed_at_line=finding["line"],
                ))

            seen_signals: set[str] = set()
            for finding in inferred_visitor.findings:
                key = f"{finding['function']}:{finding['signal']}"
                if key in seen_signals:
                    continue
                seen_signals.add(key)
                result.inferred_identity.append(InferredIdentityReliance(
                    invariant_id=new_id("reliance"),
                    description=(
                        f"Identity inferred from signal '{finding['signal']}' "
                        f"in {finding['function']} -- not verified against credential store"
                    ),
                    strength=InvariantStrength.BEHAVIORAL,
                    evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
                    confidence=0.75,
                    target_file=target_file,
                    target_function=finding["function"],
                    identity_claim=f"caller identified via '{finding['signal']}'",
                    inference_signal=finding["signal"],
                    bypassed_mechanism="credential signature verification",
                    gated_decision=f"decision in {finding['function']}",
                    verification_style=VerificationStyle.INFERRED,
                    reliance_risk=RelianceRisk.CRITICAL,
                    remediation=(
                        f"Do not trust '{finding['signal']}' as identity proof. "
                        "Require a signed credential from the authoritative issuer."
                    ),
                    observed_at_function=finding["function"],
                    observed_at_line=finding["line"],
                ))

        # Domain knowledge: known cached-authority findings
        if include_domain_knowledge:
            for kb in _KNOWN_BELIEF_CACHES:
                inv = CachedAuthorityReliance(
                    invariant_id=new_id("reliance"),
                    description=(
                        f"Cached authority: '{kb['authority_claim']}' stored in "
                        f"'{kb['cache_location']}' -- TTL {kb['cache_ttl_seconds']} s"
                    ),
                    strength=InvariantStrength.BEHAVIORAL,
                    evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
                    confidence=0.9,
                    target_file=target_file,
                    authority_claim=kb["authority_claim"],
                    cache_location=kb["cache_location"],
                    cache_ttl_seconds=kb["cache_ttl_seconds"],
                    authoritative_source=kb["authoritative_source"],
                    gated_decision=kb["gated_decision"],
                    reliance_risk=kb["reliance_risk"],
                    remediation=kb["remediation"],
                    z3_expression=kb.get("z3_expression", ""),
                    variable_declarations=kb.get("variable_declarations", {}),
                )
                if _z3_check(
                    kb.get("z3_expression", ""),
                    kb.get("variable_declarations", {}),
                    self._timeout_ms,
                ):
                    inv.evidence_source = EvidenceSource.Z3_PROOF
                    inv.strength = InvariantStrength.AXIOM
                    result.z3_verified_count += 1
                result.cached_authority.append(inv)

            # Remembered protocol: GATE-stage promise trusted at later stages
            for proto_name in ("proposal_lifecycle", "governance_vote"):
                result.remembered_protocol.append(RememberedProtocolReliance(
                    invariant_id=new_id("reliance"),
                    description=(
                        f"Protocol property from '{proto_name}' GATE stage "
                        f"trusted at later stages without re-assertion"
                    ),
                    strength=InvariantStrength.STRUCTURAL,
                    evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
                    confidence=0.85,
                    target_file=target_file,
                    protocol_property=f"{proto_name} gate conditions satisfied",
                    established_at_step="GATE stage",
                    relied_on_at_step="APPLY / RECORD stage",
                    invalidation_scenario=(
                        "Governance quorum drops or credential expires "
                        "between GATE and APPLY"
                    ),
                    verification_style=VerificationStyle.ASSUMED_VALID,
                    reliance_risk=RelianceRisk.HIGH,
                    remediation=(
                        "Re-assert gate conditions at APPLY entry. "
                        "Treat gate result as a hint, not a binding guarantee."
                    ),
                ))

        result.critical_count = sum(
            1 for inv in list(result.inferred_identity) + list(result.cached_authority)
            if inv.reliance_risk == RelianceRisk.CRITICAL
        )
        result.discovery_time_ms = int((time.monotonic() - t0) * 1000)
        self._log.info(
            "state_interpretation_complete",
            file=target_file,
            cached_authority=len(result.cached_authority),
            inferred_identity=len(result.inferred_identity),
            remembered_protocol=len(result.remembered_protocol),
            z3_verified=result.z3_verified_count,
        )
        return result


class SourceOfTruthAnalyser:
    """
    Phase 3.2 -- Source-of-Truth Gaps.

    Finds where live truth, stored truth, and inferred truth diverge at
    the point of a decision:
      - Live vs. stored (stale cache / stale session field)
      - Inferred truth (no primary source consulted)
      - Origin verification gaps (remembered origin claim, not re-checked)
    """

    def __init__(self, check_timeout_ms: int = 5000) -> None:
        self._timeout_ms = check_timeout_ms
        self._log = logger.bind(analyser="source_of_truth")

    def analyse(
        self,
        python_source: str = "",
        target_file: str = "",
        target_functions: list[str] | None = None,
        include_domain_knowledge: bool = True,
    ) -> SourceOfTruthDiscovery:
        t0 = time.monotonic()
        result = SourceOfTruthDiscovery(
            target_file=target_file,
            target_functions=target_functions or [],
        )

        if include_domain_knowledge:
            # Known live-vs-stored gaps
            for kb in _KNOWN_TRUTH_GAPS:
                inv = LiveVsStoredTruthGap(
                    invariant_id=new_id("reliance"),
                    description=(
                        f"Live-vs-stored gap: '{kb['live_truth_source']}' "
                        f"vs '{kb['stored_truth_location']}'"
                    ),
                    strength=InvariantStrength.BEHAVIORAL,
                    evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
                    confidence=0.9,
                    target_file=target_file,
                    live_truth_source=kb["live_truth_source"],
                    stored_truth_location=kb["stored_truth_location"],
                    update_mechanism=kb["update_mechanism"],
                    max_staleness_seconds=kb["max_staleness_seconds"],
                    gated_decision=kb["gated_decision"],
                    reliance_risk=kb["reliance_risk"],
                    remediation=kb["remediation"],
                    z3_expression=kb.get("z3_expression", ""),
                    variable_declarations=kb.get("variable_declarations", {}),
                )
                if _z3_check(
                    kb.get("z3_expression", ""),
                    kb.get("variable_declarations", {}),
                    self._timeout_ms,
                ):
                    inv.evidence_source = EvidenceSource.Z3_PROOF
                    inv.strength = InvariantStrength.AXIOM
                    result.z3_verified_count += 1
                result.live_vs_stored.append(inv)

            # Known origin verification gaps
            for kb in _KNOWN_ORIGIN_GAPS:
                inv = OriginVerificationGap(
                    invariant_id=new_id("reliance"),
                    description=(
                        f"Origin claim '{kb['origin_claim']}' "
                        f"{'verified' if kb['is_verified_at_use'] else 'NOT verified'} at use"
                    ),
                    strength=InvariantStrength.STRUCTURAL,
                    evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
                    confidence=0.9,
                    target_file=target_file,
                    origin_claim=kb["origin_claim"],
                    claim_location=kb["claim_location"],
                    claim_setter=kb["claim_setter"],
                    verification_mechanism=kb["verification_mechanism"],
                    is_verified_at_use=kb["is_verified_at_use"],
                    gated_decision=kb["gated_decision"],
                    reliance_risk=kb["reliance_risk"],
                    remediation=kb["remediation"],
                    z3_expression=kb.get("z3_expression", ""),
                    variable_declarations=kb.get("variable_declarations", {}),
                )
                if not kb["is_verified_at_use"]:
                    inv.violation_witness = (
                        f"Origin claim '{kb['origin_claim']}' used without verification"
                    )
                    result.violations_found += 1
                if _z3_check(
                    kb.get("z3_expression", ""),
                    kb.get("variable_declarations", {}),
                    self._timeout_ms,
                ):
                    inv.evidence_source = EvidenceSource.Z3_PROOF
                    inv.strength = InvariantStrength.AXIOM
                    result.z3_verified_count += 1
                result.origin_verification.append(inv)

            # Known inferred truth gap
            result.inferred_truth.append(InferredTruthGap(
                invariant_id=new_id("reliance"),
                description=(
                    "Health status inferred from absence of error events "
                    "rather than active heartbeat check"
                ),
                strength=InvariantStrength.BEHAVIORAL,
                evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
                confidence=0.8,
                target_file=target_file,
                inferred_truth="repair subsystem is healthy and available",
                inference_signals=[
                    "no ERROR-level log entries in last N seconds",
                    "last_repair_completed_at < threshold",
                ],
                primary_source_exists=True,
                primary_source="HealthChecker.get_live_repair_status()",
                gated_decision="allow new repair cycle to be initiated",
                reliance_risk=RelianceRisk.MEDIUM,
                remediation=(
                    "Issue an active health probe before initiating a repair cycle "
                    "rather than relying on absence-of-error inference."
                ),
                known_failure_modes=[
                    "repair agent crashes silently with no log output",
                    "log pipeline backpressure delays error delivery past decision window",
                ],
            ))

        result.critical_count = sum(
            1 for inv in list(result.origin_verification) + list(result.live_vs_stored)
            if inv.reliance_risk == RelianceRisk.CRITICAL
        )
        result.discovery_time_ms = int((time.monotonic() - t0) * 1000)
        self._log.info(
            "source_of_truth_complete",
            file=target_file,
            live_vs_stored=len(result.live_vs_stored),
            inferred_truth=len(result.inferred_truth),
            origin_verification=len(result.origin_verification),
            violations=result.violations_found,
        )
        return result


class SessionContinuityAnalyser:
    """
    Phase 3.3 -- Session Continuity.

    Finds what assumptions must persist for a multi-step workflow to proceed
    correctly, and whether those assumptions are re-validated between steps:
      - Session assumptions (FSM-level, across proposal stages)
      - Narrative continuity (hash chain / audit trail integrity)
      - Workflow TOCTOU preconditions (check-then-use with a gap)
    """

    def __init__(self, check_timeout_ms: int = 5000) -> None:
        self._timeout_ms = check_timeout_ms
        self._log = logger.bind(analyser="session_continuity")

    def analyse(
        self,
        python_source: str = "",
        target_file: str = "",
        target_functions: list[str] | None = None,
        include_domain_knowledge: bool = True,
    ) -> SessionContinuityDiscovery:
        t0 = time.monotonic()
        result = SessionContinuityDiscovery(
            target_file=target_file,
            target_functions=target_functions or [],
        )

        if include_domain_knowledge:
            # Session assumptions
            for kb in _KNOWN_SESSIONS:
                inv = SessionAssumptionReliance(
                    invariant_id=new_id("reliance"),
                    description=(
                        f"Session '{kb['session_type']}' assumes "
                        f"'{kb['persistent_assumption']}' across all stages "
                        f"without re-validation"
                    ),
                    strength=kb.get("strength", InvariantStrength.STRUCTURAL),
                    evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
                    confidence=0.9,
                    target_file=target_file,
                    session_type=kb["session_type"],
                    persistent_assumption=kb["persistent_assumption"],
                    assumption_established_at=kb["assumption_established_at"],
                    relied_on_at_steps=kb["relied_on_at_steps"],
                    invalidating_events=kb["invalidating_events"],
                    revalidation_mechanism=kb["revalidation_mechanism"],
                    reliance_risk=kb["reliance_risk"],
                    remediation=kb["remediation"],
                    fsm_name=kb["fsm_name"],
                    z3_expression=kb.get("z3_expression", ""),
                    variable_declarations=kb.get("variable_declarations", {}),
                )
                if not kb["revalidation_mechanism"]:
                    inv.violation_witness = (
                        f"No revalidation mechanism in '{kb['session_type']}'"
                    )
                    result.violations_found += 1
                if _z3_check(
                    kb.get("z3_expression", ""),
                    kb.get("variable_declarations", {}),
                    self._timeout_ms,
                ):
                    inv.evidence_source = EvidenceSource.Z3_PROOF
                    inv.strength = InvariantStrength.AXIOM
                    result.z3_verified_count += 1
                result.session_assumptions.append(inv)

            # Narrative continuity reliances
            for kb in _KNOWN_NARRATIVE_RELIANCES:
                inv = NarrativeContinuityReliance(
                    invariant_id=new_id("reliance"),
                    description=(
                        f"Narrative '{kb['narrative_source']}' relied on "
                        f"{'with' if kb['is_integrity_verified_before_act'] else 'without'} "
                        f"integrity verification before action"
                    ),
                    strength=kb.get("strength", InvariantStrength.STRUCTURAL),
                    evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
                    confidence=0.9,
                    target_file=target_file,
                    narrative_source=kb["narrative_source"],
                    integrity_mechanism=kb["integrity_mechanism"],
                    is_integrity_verified_before_act=kb["is_integrity_verified_before_act"],
                    gated_decision=kb["gated_decision"],
                    attack_surface=kb["attack_surface"],
                    reliance_risk=kb["reliance_risk"],
                    remediation=kb["remediation"],
                    z3_expression=kb.get("z3_expression", ""),
                    variable_declarations=kb.get("variable_declarations", {}),
                )
                if not kb["is_integrity_verified_before_act"]:
                    inv.violation_witness = (
                        f"Narrative '{kb['narrative_source']}' used without integrity check"
                    )
                    result.violations_found += 1
                if _z3_check(
                    kb.get("z3_expression", ""),
                    kb.get("variable_declarations", {}),
                    self._timeout_ms,
                ):
                    inv.evidence_source = EvidenceSource.Z3_PROOF
                    inv.strength = InvariantStrength.AXIOM
                    result.z3_verified_count += 1
                result.narrative_continuity.append(inv)

            # TOCTOU workflow preconditions
            for kb in _KNOWN_TOCTOU:
                inv = WorkflowPreconditionReliance(
                    invariant_id=new_id("reliance"),
                    description=(
                        f"TOCTOU: '{kb['precondition']}' checked at "
                        f"'{kb['checked_at_step']}', consumed at "
                        f"'{kb['consumed_at_step']}' -- "
                        f"{'re-checked' if kb['is_rechecked_before_consume'] else 'NOT re-checked'}"
                    ),
                    strength=InvariantStrength.STRUCTURAL,
                    evidence_source=EvidenceSource.STRUCTURAL_ANALYSIS,
                    confidence=0.9,
                    target_file=target_file,
                    precondition=kb["precondition"],
                    checked_at_step=kb["checked_at_step"],
                    consumed_at_step=kb["consumed_at_step"],
                    toctou_scenario=kb["toctou_scenario"],
                    is_rechecked_before_consume=kb["is_rechecked_before_consume"],
                    reliance_risk=kb["reliance_risk"],
                    remediation=kb["remediation"],
                    z3_expression=kb.get("z3_expression", ""),
                    variable_declarations=kb.get("variable_declarations", {}),
                )
                if not kb["is_rechecked_before_consume"]:
                    inv.violation_witness = (
                        f"Precondition '{kb['precondition']}' not re-checked before "
                        f"'{kb['consumed_at_step']}'"
                    )
                    result.violations_found += 1
                if _z3_check(
                    kb.get("z3_expression", ""),
                    kb.get("variable_declarations", {}),
                    self._timeout_ms,
                ):
                    inv.evidence_source = EvidenceSource.Z3_PROOF
                    inv.strength = InvariantStrength.AXIOM
                    result.z3_verified_count += 1
                result.workflow_preconditions.append(inv)

        all_findings = (
            list(result.session_assumptions)
            + list(result.narrative_continuity)
            + list(result.workflow_preconditions)
        )
        result.critical_count = sum(
            1 for inv in all_findings
            if inv.reliance_risk == RelianceRisk.CRITICAL
        )
        result.discovery_time_ms = int((time.monotonic() - t0) * 1000)
        self._log.info(
            "session_continuity_complete",
            file=target_file,
            session_assumptions=len(result.session_assumptions),
            narrative_continuity=len(result.narrative_continuity),
            workflow_preconditions=len(result.workflow_preconditions),
            z3_verified=result.z3_verified_count,
            violations=result.violations_found,
        )
        return result
