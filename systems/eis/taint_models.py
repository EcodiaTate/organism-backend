"""
EcodiaOS — EIS Taint Analysis Types

Data types for mutation-safety taint analysis. Consumed by:
  - EIS TaintEngine (producer)
  - Simula governance pipeline (consumer — decides whether to submit to Equor)
  - Equor (consumer — elevated scrutiny flag)

Mutation proposals arrive as (file_path, unified_diff) pairs.
The TaintRiskAssessment they produce is the primary handoff object.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ─── Enums ────────────────────────────────────────────────────────


class TaintSeverity(enum.StrEnum):
    """
    Constitutional risk proximity for a mutation proposal.

    These levels express how close a mutation is to constitutional code.
    They are NOT constitutional verdicts — Equor renders those.

    CLEAR      — No constitutional paths in scope; normal governance path.
    ADVISORY   — Indirect/transitive proximity; noted in Equor review context.
    ELEVATED   — Direct or short-chain proximity; routes to Equor mandatory
                 review before Simula may apply the mutation.
    CRITICAL   — Core constitutional path directly in scope; mutation blocked
                 until Equor renders a verdict and human approval is granted.
    """

    CLEAR = "clear"
    ADVISORY = "advisory"
    ELEVATED = "elevated"
    CRITICAL = "critical"


class TaintReason(enum.StrEnum):
    """Why a mutation was flagged."""

    DIRECT_CONSTITUTIONAL_TOUCH = "direct_constitutional_touch"
    TRANSITIVE_CONSTITUTIONAL_TOUCH = "transitive_constitutional_touch"
    SAFETY_FUNCTION_MODIFIED = "safety_function_modified"
    DRIVE_EVALUATION_MODIFIED = "drive_evaluation_modified"
    VERDICT_PIPELINE_MODIFIED = "verdict_pipeline_modified"
    INVARIANT_CHECK_MODIFIED = "invariant_check_modified"
    EIS_GATE_MODIFIED = "eis_gate_modified"
    GOVERNANCE_AUDIT_MODIFIED = "governance_audit_modified"


# ─── Mutation Proposal ────────────────────────────────────────────


class MutationProposal(EOSBaseModel):
    """
    A code mutation proposed by Simula awaiting safety analysis.

    Simula publishes these on the EVOLUTION_CANDIDATE event channel.
    EIS subscribes, analyses, and enriches the payload with a
    TaintRiskAssessment before it reaches Equor.
    """

    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)

    # ── Mutation content ──
    file_path: str                   # Repo-relative path, e.g. "systems/equor/invariants.py"
    diff: str                        # Unified diff (--- a/ +++ b/ format)
    description: str = ""            # Human-readable intent of this mutation

    # ── Provenance ──
    simula_run_id: str = ""          # ID of the Simula run that generated this
    hypothesis_id: str = ""          # Linked EVO hypothesis if applicable
    proposer_system: str = "simula"

    # ── Context ──
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Constitutional Path ──────────────────────────────────────────


class ConstitutionalPath(EOSBaseModel):
    """
    A code path (file + optional function) that is considered constitutional.

    Constitutional paths are those whose modification could compromise:
      - Safety constraints (invariant checks, harm detectors)
      - Drive evaluation (coherence/care/growth/honesty scoring)
      - Equor verdict pipeline (review, block, approve logic)
      - EIS gate (the immune filter itself)
      - Governance audit trail
    """

    path_id: str                     # Unique identifier, e.g. "equor.invariants.check_physical_harm"
    file_pattern: str                # Glob-style pattern matched against file_path
    function_names: list[str] = Field(default_factory=list)  # Empty = entire file is constitutional
    description: str = ""
    taint_reason: TaintReason = TaintReason.DIRECT_CONSTITUTIONAL_TOUCH
    severity_if_touched: TaintSeverity = TaintSeverity.ELEVATED

    # ── Graph edges ──
    feeds_into: list[str] = Field(default_factory=list)   # path_ids this path feeds into
    fed_by: list[str] = Field(default_factory=list)        # path_ids that feed into this one


# ─── Taint Result ────────────────────────────────────────────────


class TaintedPath(EOSBaseModel):
    """
    A constitutional path that is implicated by a mutation proposal.
    """

    path_id: str
    file_pattern: str
    description: str
    taint_reason: TaintReason
    severity: TaintSeverity
    is_direct: bool                  # True = mutation directly touches this path
    chain_length: int = 0            # 0 = direct, 1 = one hop, etc.
    chain: list[str] = Field(default_factory=list)  # path_ids in the propagation chain


class TaintRiskAssessment(EOSBaseModel):
    """
    The output of EIS taint analysis for a single MutationProposal.

    This is the handoff object between EIS and:
      - Simula's governance pipeline (decides routing)
      - Equor (uses tainted_paths + severity for elevated review)

    Severity ladder:
      CLEAR    → normal Equor review, no special handling
      ADVISORY → Equor reviews with taint context attached
      ELEVATED → Equor blocks auto-approve; requires human acknowledgement
      CRITICAL → Mutation held pending amendment governance process
    """

    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=utc_now)

    # ── Input reference ──
    mutation_id: str
    file_path: str
    diff_hash: str = ""              # SHA-256 of the diff for dedup

    # ── Analysis result ──
    overall_severity: TaintSeverity = TaintSeverity.CLEAR
    tainted_paths: list[TaintedPath] = Field(default_factory=list)
    reasoning: str = ""              # Human-readable explanation

    # ── Governance routing ──
    requires_human_approval: bool = False
    requires_equor_elevated_review: bool = False
    block_mutation: bool = False     # True only for CRITICAL severity

    # ── Telemetry ──
    analysis_latency_ms: int = 0
    paths_evaluated: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return self.overall_severity == TaintSeverity.CLEAR

    @property
    def summary(self) -> str:
        if self.is_clean:
            return f"Mutation to {self.file_path}: CLEAR — no constitutional paths affected."
        paths = ", ".join(p.path_id for p in self.tainted_paths[:3])
        suffix = f" (+{len(self.tainted_paths) - 3} more)" if len(self.tainted_paths) > 3 else ""
        return (
            f"Mutation to {self.file_path}: {self.overall_severity.upper()} — "
            f"affects {paths}{suffix}."
        )
