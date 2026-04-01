"""
EcodiaOS - Inspector Domain Types

All data models for the vulnerability discovery pipeline.
Uses EOSBaseModel for consistency with the rest of EOS.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator

from primitives.common import EOSBaseModel, new_id, utc_now

if TYPE_CHECKING:
    from systems.simula.inspector.taint_flow_linker import TaintEdge

# ── Enums ────────────────────────────────────────────────────────────────────


class TargetType(enum.StrEnum):
    """Whether the hunt target is internal EOS or an external repository."""

    INTERNAL_EOS = "internal_eos"
    EXTERNAL_REPO = "external_repo"


class AttackSurfaceType(enum.StrEnum):
    """Classification of an exploitable entry point."""

    API_ENDPOINT = "api_endpoint"
    MIDDLEWARE = "middleware"
    SMART_CONTRACT_PUBLIC = "smart_contract_public"
    FUNCTION_EXPORT = "function_export"
    CLI_COMMAND = "cli_command"
    WEBSOCKET_HANDLER = "websocket_handler"
    GRAPHQL_RESOLVER = "graphql_resolver"
    EVENT_HANDLER = "event_handler"
    DATABASE_QUERY = "database_query"
    FILE_UPLOAD = "file_upload"
    AUTH_HANDLER = "auth_handler"
    DESERIALIZATION = "deserialization"
    CROSS_SERVICE_ENDPOINT = "cross_service_endpoint"


class VulnerabilitySeverity(enum.StrEnum):
    """CVSS-aligned severity classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    FALSE_POSITIVE = "false_positive"


class VulnerabilityClass(enum.StrEnum):
    """Common vulnerability taxonomy (OWASP-aligned)."""

    BROKEN_AUTH = "broken_authentication"
    BROKEN_ACCESS_CONTROL = "broken_access_control"
    INJECTION = "injection"
    SQL_INJECTION = "sql_injection"
    XSS = "cross_site_scripting"
    SSRF = "server_side_request_forgery"
    IDOR = "insecure_direct_object_reference"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    REENTRANCY = "reentrancy"
    RACE_CONDITION = "race_condition"
    UNVALIDATED_REDIRECT = "unvalidated_redirect"
    INFORMATION_DISCLOSURE = "information_disclosure"
    INSECURE_DESERIALIZATION = "insecure_deserialization"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    OTHER = "other"


# ── Data Models ──────────────────────────────────────────────────────────────


class AttackSurface(EOSBaseModel):
    """A discovered exploitable entry point in the target codebase."""

    id: str = Field(default_factory=new_id)
    entry_point: str = Field(
        ...,
        description="Qualified name of the entry point (e.g., 'app.routes.get_user')",
    )
    surface_type: AttackSurfaceType
    file_path: str = Field(
        ...,
        description="Relative path within the workspace to the source file",
    )
    line_number: int | None = Field(
        default=None,
        description="Starting line number of the entry point in the file",
    )
    context_code: str = Field(
        default="",
        description="Surrounding function/class source code for Z3 encoding",
    )
    http_method: str | None = Field(
        default=None,
        description="HTTP method if this is an API endpoint (GET, POST, etc.)",
    )
    route_pattern: str | None = Field(
        default=None,
        description="URL route pattern (e.g., '/api/user/{id}')",
    )
    service_name: str | None = Field(
        default=None,
        description="Docker-compose service hosting this surface (None for single-service targets)",
    )
    taint_context: str = Field(
        default="",
        description="JSON-serialized taint summary for cross-service Z3 encoding",
    )
    discovered_at: datetime = Field(default_factory=utc_now)
    # Populated by DynamicTaintInjector: ordered list of service hops proven
    # by firing a live marked request and observing eBPF propagation.
    verified_data_path: list[TaintEdge] = Field(
        default_factory=list,
        description=(
            "Dynamically proven hop chain: [api-service → postgres, ...]. "
            "Empty when no live injection was performed."
        ),
    )


class VulnerabilityReport(EOSBaseModel):
    """A proven vulnerability with Z3 counterexample and optional PoC."""

    id: str = Field(default_factory=new_id)
    target_url: str = Field(
        ...,
        description="GitHub URL or 'internal_eos'",
    )
    vulnerability_class: VulnerabilityClass
    severity: VulnerabilitySeverity
    attack_surface: AttackSurface
    attack_goal: str = Field(
        ...,
        description="The attacker goal that was proven satisfiable",
    )
    z3_counterexample: str = Field(
        ...,
        description="Human-readable Z3 model showing the exploit conditions",
    )
    z3_constraints_code: str = Field(
        default="",
        description="The Z3 Python code that was checked",
    )
    proof_of_concept_code: str = Field(
        default="",
        description="Generated exploit script (Python)",
    )
    verified: bool = Field(
        default=False,
        description="Whether the PoC was sandbox-verified",
    )
    defender_notes: str | None = Field(
        default=None,
        description="Agent Blue's justification when a finding is refuted as a false positive",
    )
    patched_code: str | None = Field(
        default=None,
        description="Z3-verified patched source code generated by RepairAgent",
    )
    xdp_filter_code: str | None = Field(
        default=None,
        description="eBPF XDP C source synthesized by AutonomousShield (dry-run compiled)",
    )
    boundary_test_evidence: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Structured Z3 model mapped to HTTP fields for SSE boundary test results. "
            "Contains 'details.edge_case_input' as an ordered list of sequential "
            "payloads (one per exploit step in the state-machine chain), "
            "'details.state_count' indicating the number of states, and "
            "'details.global_context' for state-invariant variables."
        ),
    )
    discovered_at: datetime = Field(default_factory=utc_now)

    @property
    def has_poc(self) -> bool:
        """True when a proof-of-concept script has been generated."""
        return bool(self.proof_of_concept_code)

    @field_validator("severity")
    @classmethod
    def _validate_severity(cls, v: VulnerabilitySeverity) -> VulnerabilitySeverity:
        if v not in VulnerabilitySeverity:
            raise ValueError(f"Invalid severity: {v}")
        return v


class InspectionResult(EOSBaseModel):
    """Aggregated results from a full hunt against a target."""

    id: str = Field(default_factory=new_id)
    target_url: str
    target_type: TargetType
    surfaces_mapped: int = 0
    attack_surfaces: list[AttackSurface] = Field(default_factory=list)
    vulnerabilities_found: list[VulnerabilityReport] = Field(default_factory=list)
    generated_patches: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of vulnerability ID → patch diff",
    )
    total_duration_ms: int = 0
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None

    @property
    def vulnerability_count(self) -> int:
        return len(self.vulnerabilities_found)

    @property
    def critical_count(self) -> int:
        return sum(
            1 for v in self.vulnerabilities_found
            if v.severity == VulnerabilitySeverity.CRITICAL
        )

    @property
    def high_count(self) -> int:
        return sum(
            1 for v in self.vulnerabilities_found
            if v.severity == VulnerabilitySeverity.HIGH
        )


class InspectorConfig(EOSBaseModel):
    """Configuration for a Inspector instance. Enforces safety constraints."""

    authorized_targets: list[str] = Field(
        default_factory=list,
        description="List of authorized target domains/URLs for PoC execution",
    )
    max_workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Max concurrent attack surface analysis workers",
    )
    sandbox_timeout_seconds: int = Field(
        default=30,
        gt=0,
        description="Timeout for sandboxed PoC execution",
    )
    log_vulnerability_analytics: bool = Field(
        default=True,
        description="Whether to emit structlog analytics events for discoveries",
    )
    clone_depth: int = Field(
        default=1,
        ge=1,
        description="Git clone depth (1 = shallow clone for speed)",
    )

    @field_validator("authorized_targets")
    @classmethod
    def _validate_targets(cls, v: list[str]) -> list[str]:
        """Ensure authorized targets are non-empty strings."""
        for target in v:
            if not target.strip():
                raise ValueError("Authorized target cannot be an empty string")
        return v


# ── Phase 6: Autonomous Remediation Types ─────────────────────────────────────


class RemediationStatus(enum.StrEnum):
    """Terminal outcome of a remediation attempt."""

    PATCHED = "patched"
    PATCH_UNVERIFIED = "patch_unverified"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"
    SKIPPED = "skipped"


class RemediationAttempt(EOSBaseModel):
    """One attempt at generating and verifying a patch for a vulnerability."""

    attempt_number: int = 0
    patch_diff: str = Field(
        default="",
        description="Unified diff of the generated patch",
    )
    patched_code: str = Field(
        default="",
        description="Complete patched source code",
    )
    repair_status: str = Field(
        default="",
        description="Status from the underlying RepairAgent (repaired/failed/etc.)",
    )
    verification_result: str = Field(
        default="",
        description="UNSAT = vulnerability eliminated, SAT = still exploitable",
    )
    vulnerability_eliminated: bool = False
    cost_usd: float = 0.0
    duration_ms: int = 0
    error: str = ""


class RemediationResult(EOSBaseModel):
    """Aggregate result of attempting to remediate a single vulnerability."""

    id: str = Field(default_factory=new_id)
    vulnerability_id: str = Field(
        ...,
        description="ID of the VulnerabilityReport being remediated",
    )
    status: RemediationStatus = RemediationStatus.SKIPPED
    attempts: list[RemediationAttempt] = Field(default_factory=list)
    total_attempts: int = 0
    successful_attempt: int | None = Field(
        default=None,
        description="Which attempt succeeded (0-indexed), None if no success",
    )
    final_patch_diff: str = Field(
        default="",
        description="The verified patch diff (empty if remediation failed)",
    )
    final_patched_code: str = Field(
        default="",
        description="The verified patched source code",
    )
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    remediated_at: datetime = Field(default_factory=utc_now)


# ── Rebuild forward references ──────────────────────────────────────────────
# `from __future__ import annotations` turns all annotations into strings.
# Pydantic resolves them lazily, but models referencing types defined later
# (or in other modules) need an explicit rebuild once all types are available.

from systems.simula.inspector.taint_flow_linker import TaintEdge  # noqa: E402 - deferred to break circular import

AttackSurface.model_rebuild(_types_namespace={"TaintEdge": TaintEdge})
VulnerabilityReport.model_rebuild()
InspectionResult.model_rebuild()
RemediationResult.model_rebuild()
