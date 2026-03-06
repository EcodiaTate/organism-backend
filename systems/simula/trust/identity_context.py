"""
EcodiaOS — Simula Identity & Trust Context

Tracks who initiated an action and how their identity affects system behavior.
Documents implicit trust decisions and privilege escalation paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import structlog

logger = structlog.get_logger().bind(system="simula.trust.identity")


class PrincipalRole(StrEnum):
    """Role of the principal initiating an action."""
    EVO_SYSTEM = "evo"  # Internal Evo hypothesis consolidation
    ADMIN = "admin"  # System administrator
    EXTERNAL_API = "external_api"  # External API caller
    GOVERNANCE = "governance"  # Community governance vote
    REPAIR_AGENT = "repair_agent"  # Internal repair system
    UNKNOWN = "unknown"


class CredentialKind(StrEnum):
    """Type of credential presented."""
    NONE = "none"
    API_TOKEN = "api_token"
    ADMIN_SIGN = "admin_sign"
    GOVERNANCE_VOTE = "governance_vote"
    SERVICE_ACCOUNT = "service_account"


@dataclass
class IdentityContext:
    """
    Who initiated this action? What trust signals are presented?

    Documents the identity of the principal making a proposal
    and how their identity affects system behavior (gates, permissions, etc.)
    """
    principal_id: str  # Unique identifier of who initiated this
    role: PrincipalRole  # Role in the system
    credential_kind: CredentialKind = CredentialKind.NONE
    credential_strength: float = 0.0  # 0.0 (untrusted) - 1.0 (fully trusted)
    delegation_chain: list[str] = field(default_factory=list)  # Who delegated to whom

    # Trust decisions this identity enables
    self_applicable: bool = False  # Can skip governance gate
    can_modify_self: bool = False  # Can modify Simula itself
    max_risk_allowed: str = "HIGH"  # Risk level allowed (LOW/MEDIUM/HIGH/CRITICAL)

    # Trust assumptions about this principal
    trust_assumptions: list[str] = field(default_factory=list)
    assumption_confidence: float = 0.5

    # Audit trail
    authenticated_at_utc: str = ""
    authentication_method: str = ""

    @classmethod
    def evo_system(cls) -> IdentityContext:
        """Create identity for Evo system (internal)."""
        return cls(
            principal_id="evo_system",
            role=PrincipalRole.EVO_SYSTEM,
            credential_kind=CredentialKind.SERVICE_ACCOUNT,
            credential_strength=1.0,
            self_applicable=True,  # Evo can skip some gates
            can_modify_self=False,  # But cannot modify Simula itself
            max_risk_allowed="CRITICAL",
            trust_assumptions=[
                "Evo system is trusted to form hypotheses",
                "Evo consolidation process is sound",
            ],
            assumption_confidence=0.95,
        )

    @classmethod
    def admin(cls, admin_id: str, authenticated: bool = True) -> IdentityContext:
        """Create identity for admin."""
        return cls(
            principal_id=f"admin_{admin_id}",
            role=PrincipalRole.ADMIN,
            credential_kind=CredentialKind.ADMIN_SIGN,
            credential_strength=1.0 if authenticated else 0.5,
            self_applicable=False,
            can_modify_self=False,
            max_risk_allowed="CRITICAL",
            trust_assumptions=[
                "Admin has been authenticated",
                "Admin is authorized for this operation",
            ],
            assumption_confidence=0.9 if authenticated else 0.5,
        )

    @classmethod
    def external_api(cls, api_key_hash: str) -> IdentityContext:
        """Create identity for external API caller."""
        return cls(
            principal_id=f"external_{api_key_hash[:16]}",
            role=PrincipalRole.EXTERNAL_API,
            credential_kind=CredentialKind.API_TOKEN,
            credential_strength=0.7,  # Moderate trust
            self_applicable=False,
            can_modify_self=False,
            max_risk_allowed="MEDIUM",
            trust_assumptions=[
                "API token is valid",
                "API caller has limited scope",
            ],
            assumption_confidence=0.7,
        )

    @classmethod
    def governance_vote(cls) -> IdentityContext:
        """Create identity for community governance."""
        return cls(
            principal_id="governance_vote",
            role=PrincipalRole.GOVERNANCE,
            credential_kind=CredentialKind.GOVERNANCE_VOTE,
            credential_strength=0.85,  # Moderately strong
            self_applicable=False,
            can_modify_self=False,
            max_risk_allowed="CRITICAL",
            trust_assumptions=[
                "Vote quorum was reached",
                "Vote was conducted fairly",
            ],
            assumption_confidence=0.85,
        )


@dataclass
class TrustAssumption:
    """An implicit trust point in the system."""
    assumption_id: str
    name: str
    location: str  # File path and line number
    description: str
    risk_level: float  # 0.0 = safe, 1.0 = critical
    assumption_text: str  # What we're assuming
    what_could_go_wrong: str


@dataclass
class IdentityBranchingPoint:
    """A control-flow decision that depends on identity context."""
    branch_id: str
    phase: str  # Which stage of the pipeline
    decision: str  # What was decided
    triggered_by: str  # Which identity attribute triggered it (e.g., "role", "self_applicable")
    if_different_identity: str  # What would happen otherwise
    governance_gate_applied: bool = False


class IdentityContextTracer:
    """
    Tracks how identity affects behavior throughout a proposal lifecycle.

    Records:
      - Identity context at each decision point
      - Branching based on identity
      - Trust assumptions relied upon
      - Privilege escalation paths
    """

    def __init__(self, proposal_id: str, initiator_context: IdentityContext) -> None:
        self.proposal_id = proposal_id
        self.initiator_context = initiator_context
        self.branching_points: list[IdentityBranchingPoint] = []
        self.trust_assumptions_relied_upon: list[TrustAssumption] = []
        self._log = logger

    def record_branching_point(
        self,
        phase: str,
        decision: str,
        triggered_by: str,
        alternative: str,
        governance_gate: bool = False,
    ) -> None:
        """Record a decision that was influenced by identity."""
        bp = IdentityBranchingPoint(
            branch_id=f"branch_{self.proposal_id[:8]}_{len(self.branching_points)}",
            phase=phase,
            decision=decision,
            triggered_by=triggered_by,
            if_different_identity=alternative,
            governance_gate_applied=governance_gate,
        )
        self.branching_points.append(bp)

        self._log.debug(
            "identity_branching_point_recorded",
            proposal_id=self.proposal_id,
            phase=phase,
            triggered_by=triggered_by,
            governance_gate=governance_gate,
        )

    def record_trust_assumption(
        self,
        name: str,
        location: str,
        description: str,
        risk_level: float,
        assumption_text: str,
        what_could_go_wrong: str,
    ) -> None:
        """Record an implicit trust assumption."""
        assumption = TrustAssumption(
            assumption_id=f"assumption_{len(self.trust_assumptions_relied_upon)}",
            name=name,
            location=location,
            description=description,
            risk_level=risk_level,
            assumption_text=assumption_text,
            what_could_go_wrong=what_could_go_wrong,
        )
        self.trust_assumptions_relied_upon.append(assumption)

        self._log.debug(
            "trust_assumption_recorded",
            proposal_id=self.proposal_id,
            name=name,
            risk_level=risk_level,
        )

    def get_privilege_gradient(self) -> list[tuple[str, float]]:
        """
        Trace privilege escalation through the pipeline.

        Returns list of (phase, privilege_level) tuples showing how privilege
        changes as proposal flows through stages.
        """
        # Start with initiator privilege
        privilege = self.initiator_context.credential_strength

        gradient = [
            ("INITIAL", privilege),
        ]

        # Track changes at branching points
        current_phase = "INITIAL"
        for bp in self.branching_points:
            if bp.governance_gate_applied:
                # Governance gate might increase privilege
                privilege = min(1.0, privilege + 0.15)  # Governance boost
            current_phase = bp.phase
            gradient.append((current_phase, privilege))

        return gradient

    def get_credential_flows(self) -> list[tuple[str, str]]:
        """
        Identify where credentials flow through the system.

        Returns list of (source, sink) pairs indicating credential transmission.
        """
        flows = []

        # Credential flows (hardcoded patterns from Simula architecture)
        if self.initiator_context.role == PrincipalRole.EVO_SYSTEM:
            flows.append(("evo_system", "code_agent"))
            flows.append(("code_agent", "applicator"))
            flows.append(("applicator", "health_checker"))

        elif self.initiator_context.role == PrincipalRole.ADMIN:
            flows.append(("admin_context", "governance_gate"))
            flows.append(("governance_gate", "simulator"))

        elif self.initiator_context.role == PrincipalRole.EXTERNAL_API:
            flows.append(("api_token", "bridge"))
            flows.append(("bridge", "simulator"))

        return flows

    def get_summary(self) -> dict[str, Any]:
        """Get summary of identity context and its effects."""
        return {
            "proposal_id": self.proposal_id,
            "initiator": {
                "principal_id": self.initiator_context.principal_id,
                "role": self.initiator_context.role.value,
                "self_applicable": self.initiator_context.self_applicable,
                "can_modify_self": self.initiator_context.can_modify_self,
            },
            "branching_points": len(self.branching_points),
            "trust_assumptions": len(self.trust_assumptions_relied_upon),
            "privilege_gradient": self.get_privilege_gradient(),
            "credential_flows": self.get_credential_flows(),
            "high_risk_assumptions": [
                a.name for a in self.trust_assumptions_relied_upon
                if a.risk_level > 0.7
            ],
        }
