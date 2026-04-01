"""
EcodiaOS -- EIS Taint Analysis Engine (Constitutional Risk Flagging)

BOUNDARY CONTRACT
-----------------
The TaintEngine identifies CONSTITUTIONAL RISK in mutation proposals.
It does NOT render constitutional verdicts - that is Equor's exclusive role.

The engine flags that a mutation *touches* constitutional paths and routes
the mutation to Equor for actual constitutional review. The severity levels
(ADVISORY/ELEVATED/CRITICAL) express risk proximity, not compliance judgments.

─────────────────────────────────────────────────────────────────────────────

The TaintEngine ingests a MutationProposal (file_path + unified diff),
maps it against the ConstitutionalGraph, and produces a TaintRiskAssessment
suitable for consumption by:

  - Simula governance pipeline (routing decision)
  - Equor (elevated-scrutiny annotation on the governance record)

Entry point: TaintEngine.analyse_mutation(proposal) -> TaintRiskAssessment

Severity ladder (output - risk proximity, not constitutional verdict):
  CLEAR    -- No constitutional paths affected. Normal governance path.
  ADVISORY -- Transitive/low-risk proximity. Noted in Equor review context.
  ELEVATED -- Direct or shallow-chain proximity. Routes to Equor for mandatory
              review; auto-approve blocked until Equor renders a verdict.
  CRITICAL -- Core constitutional path directly in scope. Mutation held;
              Equor performs constitutional review via the amendment pipeline.

The engine is stateless across calls; the ConstitutionalGraph is loaded
once at construction and can be extended via register_path().
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import structlog

from systems.eis.constitutional_graph import (
    ConstitutionalGraph,
    extract_changed_functions,
)
from systems.eis.taint_models import (
    ConstitutionalPath,
    MutationProposal,
    TaintedPath,
    TaintRiskAssessment,
    TaintSeverity,
)

logger = structlog.get_logger().bind(system="eis", component="taint_engine")

# ─── Severity ordering ───────────────────────────────────────────

_SEVERITY_RANK: dict[TaintSeverity, int] = {
    TaintSeverity.CLEAR: 0,
    TaintSeverity.ADVISORY: 1,
    TaintSeverity.ELEVATED: 2,
    TaintSeverity.CRITICAL: 3,
}


def _max_severity(a: TaintSeverity, b: TaintSeverity) -> TaintSeverity:
    return a if _SEVERITY_RANK[a] >= _SEVERITY_RANK[b] else b


# ─── TaintEngine ────────────────────────────────────────────────


class TaintEngine:
    """
    Stateless taint analysis engine for self-modification safety.

    The engine owns a ConstitutionalGraph (loaded at construction) and
    applies it to each MutationProposal via analyse_mutation().

    Thread-safety: analyse_mutation() is read-only on the graph and
    safe to call concurrently. register_path() is not thread-safe
    while analyse_mutation() is running.
    """

    def __init__(self, graph: ConstitutionalGraph | None = None) -> None:
        if graph is None:
            graph = ConstitutionalGraph()
            graph.load_defaults()
        self._graph = graph
        self._logger = logger
        self._calls: int = 0
        self._critical_flags: int = 0

    # ── Public API ───────────────────────────────────────────────

    def analyse_mutation(self, proposal: MutationProposal) -> TaintRiskAssessment:
        """
        Analyse a MutationProposal for constitutional taint.

        Steps:
          1. Extract all identifiers/function names from the diff.
          2. Find directly-matched constitutional paths in the graph.
          3. Propagate taint transitively through feeds_into edges.
          4. Compute overall severity and routing flags.
          5. Build and return TaintRiskAssessment.

        Synchronous and fast (<5ms for typical diffs).
        """
        t0_ms = time.perf_counter() * 1000
        self._calls += 1

        # Step 1: Extract names from the diff
        changed_functions = extract_changed_functions(proposal.diff)

        # Step 2: Direct matches
        direct_matches = self._graph.find_direct_matches(
            file_path=proposal.file_path,
            changed_functions=changed_functions,
        )

        # Step 3: Taint propagation (includes direct nodes)
        tainted_paths = self._graph.propagate_taint(direct_matches)

        # Step 4: Overall severity
        overall_severity = TaintSeverity.CLEAR
        for tp in tainted_paths:
            overall_severity = _max_severity(overall_severity, tp.severity)

        if overall_severity == TaintSeverity.CRITICAL:
            self._critical_flags += 1

        # Step 5: Routing flags
        requires_equor_elevated = overall_severity in (
            TaintSeverity.ELEVATED,
            TaintSeverity.CRITICAL,
        )
        requires_human = overall_severity == TaintSeverity.CRITICAL
        block_mutation = overall_severity == TaintSeverity.CRITICAL

        # Build reasoning narrative
        reasoning = _build_reasoning(
            proposal=proposal,
            tainted_paths=tainted_paths,
            overall_severity=overall_severity,
        )

        latency_ms = int((time.perf_counter() * 1000) - t0_ms)

        assessment = TaintRiskAssessment(
            mutation_id=proposal.id,
            file_path=proposal.file_path,
            diff_hash=_sha256_prefix(proposal.diff),
            overall_severity=overall_severity,
            tainted_paths=tainted_paths,
            reasoning=reasoning,
            requires_human_approval=requires_human,
            requires_equor_elevated_review=requires_equor_elevated,
            block_mutation=block_mutation,
            analysis_latency_ms=latency_ms,
            paths_evaluated=len(self._graph),
            metadata={
                "simula_run_id": proposal.simula_run_id,
                "hypothesis_id": proposal.hypothesis_id,
                "changed_function_count": len(changed_functions),
                "direct_match_count": len(direct_matches),
                "transitive_match_count": len(tainted_paths) - len(direct_matches),
            },
        )

        self._logger.info(
            "taint_analysis_complete",
            mutation_id=proposal.id,
            file_path=proposal.file_path,
            severity=overall_severity,
            direct_matches=len(direct_matches),
            total_tainted=len(tainted_paths),
            latency_ms=latency_ms,
            block=block_mutation,
        )

        return assessment

    def register_path(self, path: ConstitutionalPath) -> None:
        """Register an additional constitutional path at runtime."""
        self._graph.register_path(path)

    # ── Health / introspection ────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "calls": self._calls,
            "critical_flags": self._critical_flags,
            "constitutional_paths": len(self._graph),
        }


# ─── Helpers ─────────────────────────────────────────────────────


def _sha256_prefix(text: str, length: int = 16) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def _build_reasoning(
    proposal: MutationProposal,
    tainted_paths: list[TaintedPath],
    overall_severity: TaintSeverity,
) -> str:
    """Produce a human-readable explanation of the taint assessment."""
    if overall_severity == TaintSeverity.CLEAR:
        return (
            f"Mutation to '{proposal.file_path}' does not touch any constitutional "
            f"paths. Normal governance review applies."
        )

    parts: list[str] = [
        f"Mutation to '{proposal.file_path}' -- severity: {overall_severity.upper()}.",
        "",
    ]

    direct = [t for t in tainted_paths if t.is_direct]
    transitive = [t for t in tainted_paths if not t.is_direct]

    if direct:
        parts.append(f"Direct constitutional touches ({len(direct)}):")
        for tp in direct:
            parts.append(f"  [{tp.severity.upper()}] {tp.path_id}: {tp.description}")
            parts.append(f"    Reason: {tp.taint_reason.replace('_', ' ')}")

    if transitive:
        parts.append("")
        parts.append(f"Transitive constitutional impacts ({len(transitive)}):")
        for tp in transitive[:5]:
            chain_str = " -> ".join(tp.chain)
            parts.append(f"  [{tp.severity.upper()}] {tp.path_id} (via {chain_str})")
        if len(transitive) > 5:
            parts.append(f"  ... and {len(transitive) - 5} more.")

    parts.append("")
    if overall_severity == TaintSeverity.CRITICAL:
        parts.append(
            "CRITICAL: This mutation is in direct proximity to a core constitutional "
            "path. Flagged for Equor constitutional review via the amendment pipeline; "
            "blocked until Equor renders a verdict and human approval is granted."
        )
    elif overall_severity == TaintSeverity.ELEVATED:
        parts.append(
            "ELEVATED: This mutation is in proximity to safety-adjacent constitutional "
            "code. Escalated to Equor for mandatory constitutional review; auto-approve "
            "blocked until Equor renders a verdict."
        )
    elif overall_severity == TaintSeverity.ADVISORY:
        parts.append(
            "ADVISORY: Low-risk transitive proximity to constitutional paths. "
            "Noted in Equor review context; auto-approval path remains open."
        )

    return "\n".join(parts)
