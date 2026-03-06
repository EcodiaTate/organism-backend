"""
EcodiaOS -- EIS Constitutional Dependency Graph

Defines which code paths are "constitutional" (safety-critical) and how they
relate to each other. Taint propagates transitively: if a mutation modifies
function A, and A feeds into constitutional function B, the mutation is also
tainted via the A->B edge.

Graph structure:
  Nodes = ConstitutionalPath instances (keyed by path_id)
  Edges = feeds_into / fed_by relationships (directed acyclic graph)

Design:
  - File matching uses fnmatch (glob-style) against repo-relative paths.
  - Function matching: a diff touching any changed line containing the
    function name is considered a direct touch.
  - Taint propagation is BFS from directly-touched nodes, collecting all
    reachable constitutional nodes within MAX_PROPAGATION_DEPTH hops.
"""

from __future__ import annotations

import fnmatch
import re
from collections import deque
from typing import TYPE_CHECKING

import structlog

from systems.eis.taint_models import (
    ConstitutionalPath,
    TaintedPath,
    TaintReason,
    TaintSeverity,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = structlog.get_logger().bind(system="eis", component="constitutional_graph")

MAX_PROPAGATION_DEPTH = 5  # Safety ceiling on BFS depth

# ─── The Constitutional Graph ────────────────────────────────────
#
# Path IDs follow: <system>.<module>.<function_or_scope>
# file_patterns use Unix-style globs matched against repo-relative paths.
# feeds_into lists path_ids that this node flows toward (forward edges).
#
# Severity rationale:
#   CRITICAL  -- modifying this IS a constitutional amendment attempt
#   ELEVATED  -- indirect safety risk, requires human acknowledgment
#   ADVISORY  -- transitive/low-risk, flag in Equor context but auto-approvable

_CONSTITUTIONAL_PATHS: list[ConstitutionalPath] = [

    # ── Equor core ────────────────────────────────────────────────

    ConstitutionalPath(
        path_id="equor.invariants.hardcoded",
        file_pattern="**/equor/invariants.py",
        # Entire file is constitutional -- any change to invariants.py is flagged.
        # function_names is empty so the whole file matches.
        description="Hardcoded invariants -- the absolute prohibition layer",
        taint_reason=TaintReason.INVARIANT_CHECK_MODIFIED,
        severity_if_touched=TaintSeverity.CRITICAL,
        feeds_into=["equor.service.review"],
    ),

    ConstitutionalPath(
        path_id="equor.service.review",
        file_pattern="**/equor/service.py",
        function_names=["review", "_evaluate_invariants", "_apply_verdict"],
        description="Equor constitutional review entrypoint and verdict logic",
        taint_reason=TaintReason.VERDICT_PIPELINE_MODIFIED,
        severity_if_touched=TaintSeverity.CRITICAL,
        feeds_into=["equor.service.governance_record"],
        fed_by=["equor.invariants.hardcoded", "equor.evaluators.drives", "equor.autonomy"],
    ),

    ConstitutionalPath(
        path_id="equor.service.governance_record",
        file_pattern="**/equor/service.py",
        function_names=["_record_governance_event", "_store_review"],
        description="Governance audit trail write path",
        taint_reason=TaintReason.GOVERNANCE_AUDIT_MODIFIED,
        severity_if_touched=TaintSeverity.ELEVATED,
        fed_by=["equor.service.review"],
    ),

    ConstitutionalPath(
        path_id="equor.evaluators.drives",
        file_pattern="**/equor/evaluators.py",
        function_names=["evaluate_all_drives", "default_evaluators", "BaseEquorEvaluator"],
        description="Drive alignment evaluators (coherence/care/growth/honesty)",
        taint_reason=TaintReason.DRIVE_EVALUATION_MODIFIED,
        severity_if_touched=TaintSeverity.CRITICAL,
        feeds_into=["equor.service.review"],
    ),

    ConstitutionalPath(
        path_id="equor.autonomy",
        file_pattern="**/equor/autonomy.py",
        function_names=["apply_autonomy_change", "check_promotion_eligibility", "get_autonomy_level"],
        description="Autonomy level enforcement -- controls what the system can act on",
        taint_reason=TaintReason.SAFETY_FUNCTION_MODIFIED,
        severity_if_touched=TaintSeverity.CRITICAL,
        feeds_into=["equor.service.review"],
    ),

    ConstitutionalPath(
        path_id="equor.amendment",
        file_pattern="**/equor/amendment.py",
        description="Constitutional amendment pipeline -- how drives change",
        taint_reason=TaintReason.VERDICT_PIPELINE_MODIFIED,
        severity_if_touched=TaintSeverity.CRITICAL,
    ),

    ConstitutionalPath(
        path_id="equor.drift",
        file_pattern="**/equor/drift.py",
        function_names=["DriftTracker", "respond_to_drift", "store_drift_report"],
        description="Value drift monitoring -- detects constitutional misalignment",
        taint_reason=TaintReason.SAFETY_FUNCTION_MODIFIED,
        severity_if_touched=TaintSeverity.ELEVATED,
        feeds_into=["equor.service.review"],
    ),

    ConstitutionalPath(
        path_id="equor.constitutional_memory",
        file_pattern="**/equor/constitutional_memory.py",
        description="Persistent constitutional memory -- invariant and drive history",
        taint_reason=TaintReason.GOVERNANCE_AUDIT_MODIFIED,
        severity_if_touched=TaintSeverity.ELEVATED,
        feeds_into=["equor.service.review"],
    ),

    ConstitutionalPath(
        path_id="equor.economic_evaluator",
        file_pattern="**/equor/economic_evaluator.py",
        function_names=[
            "evaluate_economic_intent",
            "classify_economic_action",
            "apply_economic_adjustment",
        ],
        description="Economic action evaluator -- guards resource and financial decisions",
        taint_reason=TaintReason.DRIVE_EVALUATION_MODIFIED,
        severity_if_touched=TaintSeverity.ELEVATED,
        feeds_into=["equor.service.review"],
    ),

    # ── EIS gate (immune system cannot modify itself unreviewed) ──

    ConstitutionalPath(
        path_id="eis.service.gate",
        file_pattern="**/eis/service.py",
        function_names=["eis_gate"],
        description="EIS gate -- the primary epistemic immune filter",
        taint_reason=TaintReason.EIS_GATE_MODIFIED,
        severity_if_touched=TaintSeverity.CRITICAL,
    ),

    ConstitutionalPath(
        path_id="eis.innate.checks",
        file_pattern="**/eis/innate.py",
        function_names=["run_innate_checks"],
        description="Innate immune checks -- fast-path threat detection patterns",
        taint_reason=TaintReason.EIS_GATE_MODIFIED,
        severity_if_touched=TaintSeverity.ELEVATED,
        feeds_into=["eis.service.gate"],
    ),

    ConstitutionalPath(
        path_id="eis.taint_engine",
        file_pattern="**/eis/taint_engine.py",
        function_names=["TaintEngine", "analyse_mutation"],
        description="EIS taint analysis engine -- self-referential protection",
        taint_reason=TaintReason.EIS_GATE_MODIFIED,
        severity_if_touched=TaintSeverity.CRITICAL,
    ),

    ConstitutionalPath(
        path_id="eis.constitutional_graph",
        file_pattern="**/eis/constitutional_graph.py",
        description="Constitutional graph definition -- what counts as constitutional",
        taint_reason=TaintReason.EIS_GATE_MODIFIED,
        severity_if_touched=TaintSeverity.CRITICAL,
    ),

    # ── Drive / value primitives ──────────────────────────────────

    ConstitutionalPath(
        path_id="primitives.constitutional",
        file_pattern="**/primitives/constitutional.py",
        description="ConstitutionalCheck primitive -- the ethical evaluation record",
        taint_reason=TaintReason.DRIVE_EVALUATION_MODIFIED,
        severity_if_touched=TaintSeverity.ELEVATED,
        feeds_into=["equor.service.review"],
    ),

    ConstitutionalPath(
        path_id="primitives.common.drives",
        file_pattern="**/primitives/common.py",
        function_names=["DriveAlignmentVector", "Verdict", "AutonomyLevel"],
        description="Core drive alignment and verdict primitives",
        taint_reason=TaintReason.DRIVE_EVALUATION_MODIFIED,
        severity_if_touched=TaintSeverity.ELEVATED,
        feeds_into=["equor.evaluators.drives", "equor.service.review"],
    ),

    # ── Governance audit ──────────────────────────────────────────

    ConstitutionalPath(
        path_id="primitives.governance",
        file_pattern="**/primitives/governance.py",
        description="Governance records and amendment primitives",
        taint_reason=TaintReason.GOVERNANCE_AUDIT_MODIFIED,
        severity_if_touched=TaintSeverity.ELEVATED,
        feeds_into=["equor.service.governance_record"],
    ),

    # ── Simula self-modification pipeline ────────────────────────

    ConstitutionalPath(
        path_id="simula.orchestration.orchestrator",
        file_pattern="**/simula/orchestration/orchestrator.py",
        function_names=["Orchestrator"],
        description="Simula orchestration -- controls when mutations are applied",
        taint_reason=TaintReason.SAFETY_FUNCTION_MODIFIED,
        severity_if_touched=TaintSeverity.ELEVATED,
    ),
]


# ─── ConstitutionalGraph ─────────────────────────────────────────


class ConstitutionalGraph:
    """
    Directed graph of constitutional code paths.

    Taint propagates forward through feeds_into edges via BFS: a mutation to
    node A infects all nodes reachable from A via feeds_into edges, because
    downstream consumers inherit the risk of their corrupted upstream.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, ConstitutionalPath] = {}
        self._logger = logger

    # ── Construction ─────────────────────────────────────────────

    def load_defaults(self) -> None:
        """Load the built-in constitutional path catalog."""
        for path in _CONSTITUTIONAL_PATHS:
            self._nodes[path.path_id] = path
        self._logger.info("constitutional_graph_loaded", node_count=len(self._nodes))

    def register_path(self, path: ConstitutionalPath) -> None:
        """Register an additional constitutional path at runtime."""
        self._nodes[path.path_id] = path
        self._logger.info("constitutional_path_registered", path_id=path.path_id)

    # ── Matching ─────────────────────────────────────────────────

    def find_direct_matches(
        self,
        file_path: str,
        changed_functions: set[str],
    ) -> list[ConstitutionalPath]:
        """
        Return all constitutional paths whose file_pattern matches file_path.
        If a path specifies function_names, at least one must appear in
        changed_functions for it to match.
        """
        matches: list[ConstitutionalPath] = []
        for node in self._nodes.values():
            if not fnmatch.fnmatch(file_path, node.file_pattern):
                continue
            if node.function_names:
                if not changed_functions.intersection(node.function_names):
                    continue
            matches.append(node)
        return matches

    # ── Taint propagation ────────────────────────────────────────

    def propagate_taint(
        self,
        direct_matches: list[ConstitutionalPath],
    ) -> list[TaintedPath]:
        """
        BFS from each directly-matched node following feeds_into edges.
        Returns de-duplicated TaintedPath records sorted by severity then depth.
        """
        seen_path_ids: set[str] = set()
        results: list[TaintedPath] = []

        for start in direct_matches:
            if start.path_id not in seen_path_ids:
                seen_path_ids.add(start.path_id)
                results.append(TaintedPath(
                    path_id=start.path_id,
                    file_pattern=start.file_pattern,
                    description=start.description,
                    taint_reason=start.taint_reason,
                    severity=start.severity_if_touched,
                    is_direct=True,
                    chain_length=0,
                    chain=[start.path_id],
                ))

            queue: deque[tuple[ConstitutionalPath, int, list[str]]] = deque()
            queue.append((start, 0, [start.path_id]))

            while queue:
                node, depth, chain = queue.popleft()
                if depth >= MAX_PROPAGATION_DEPTH:
                    continue

                for downstream_id in node.feeds_into:
                    if downstream_id in seen_path_ids:
                        continue
                    downstream = self._nodes.get(downstream_id)
                    if downstream is None:
                        continue

                    seen_path_ids.add(downstream_id)
                    new_chain = chain + [downstream_id]
                    effective_severity = _cap_transitive_severity(
                        downstream.severity_if_touched, depth + 1
                    )
                    results.append(TaintedPath(
                        path_id=downstream_id,
                        file_pattern=downstream.file_pattern,
                        description=downstream.description,
                        taint_reason=TaintReason.TRANSITIVE_CONSTITUTIONAL_TOUCH,
                        severity=effective_severity,
                        is_direct=False,
                        chain_length=depth + 1,
                        chain=new_chain,
                    ))
                    queue.append((downstream, depth + 1, new_chain))

        _sev: dict[TaintSeverity, int] = {
            TaintSeverity.CRITICAL: 3,
            TaintSeverity.ELEVATED: 2,
            TaintSeverity.ADVISORY: 1,
            TaintSeverity.CLEAR: 0,
        }
        results.sort(key=lambda t: (-_sev.get(t.severity, 0), t.chain_length))
        return results

    # ── Introspection ────────────────────────────────────────────

    def all_paths(self) -> Iterator[ConstitutionalPath]:
        yield from self._nodes.values()

    def get_path(self, path_id: str) -> ConstitutionalPath | None:
        return self._nodes.get(path_id)

    def __len__(self) -> int:
        return len(self._nodes)


# ─── Helpers ─────────────────────────────────────────────────────


def _cap_transitive_severity(original: TaintSeverity, depth: int) -> TaintSeverity:
    """
    Downgrade severity for transitive (non-direct) taint:
      depth 0 (direct)   -> original severity unchanged
      depth 1 (one hop)  -> CRITICAL becomes ELEVATED, others unchanged
      depth 2+ (N hops)  -> max ADVISORY
    """
    if depth == 0:
        return original
    if depth == 1:
        if original == TaintSeverity.CRITICAL:
            return TaintSeverity.ELEVATED
        return original
    return TaintSeverity.ADVISORY


def extract_changed_functions(diff: str) -> set[str]:
    """
    Parse a unified diff and return identifiers from changed lines.

    Intentionally broad: false positives (over-flagging) are safe;
    false negatives (missing a constitutional touch) are not.

    Extracts from:
    1. def/class declarations on added/removed lines
    2. Hunk header context (function name after @@ ... @@)
    3. All identifier tokens on added/removed lines
    """
    names: set[str] = set()

    # 1. def/class on changed lines
    decl_re = re.compile(r"^[+\-]\s*(def|class|async def)\s+(\w+)", re.MULTILINE)
    for m in decl_re.finditer(diff):
        names.add(m.group(2))

    # 2. hunk header: "@@ ... @@ def/class name"
    hunk_re = re.compile(r"@@[^@]*@@\s+(?:(?:async\s+)?def|class)\s+(\w+)")
    for m in hunk_re.finditer(diff):
        names.add(m.group(1))

    # 3. all identifiers on added/removed lines (not file header lines)
    changed_line_re = re.compile(r"^[+\-](?![+\-])\s*(.*)", re.MULTILINE)
    ident_re = re.compile(r"\b([A-Za-z_]\w*)\b")
    for lm in changed_line_re.finditer(diff):
        for im in ident_re.finditer(lm.group(1)):
            token = im.group(1)
            if len(token) > 2 and not token.isupper():
                names.add(token)

    return names
