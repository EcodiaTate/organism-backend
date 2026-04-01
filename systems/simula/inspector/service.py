"""
EcodiaOS - Inspector Service Orchestrator (Phase 7)

Coordinates the full hunting pipeline from target to vulnerability reports:

  1. INGEST   - clone external repo or point at internal EOS workspace
  2. MAP      - discover exploitable attack surfaces via AST/regex scanning
  3. PROVE    - encode attacker goals as Z3 constraints, check satisfiability
  4. EXPLOIT  - translate Z3 SAT counterexamples into proof-of-concept scripts
  5. PATCH    - optionally generate + verify patches via RepairAgent
  6. REPORT   - aggregate findings into a InspectionResult with full analytics

Two entry points:
  hunt_external_repo(github_url) - clone + full pipeline
  hunt_internal_eos()            - scan the EOS codebase itself

Iron Rules (non-negotiable):
  - Inspector NEVER writes to EOS source files (only temp workspace files)
  - PoC execution is scoped to authorized_targets only
  - All vulnerabilities logged via structlog with event="vulnerability_discovered"
  - Z3 SAT counterexamples validated before PoC generation
  - Internal Simula self-evolution must remain fully functional
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now
from systems.simula.inspector.analytics import InspectorAnalyticsView
from systems.simula.inspector.ingestor import TargetIngestor
from systems.simula.inspector.safety import InspectorSafetyGates
from systems.simula.inspector.types import (
    AttackSurface,
    AttackSurfaceType,
    InspectionResult,
    InspectorConfig,
    RemediationStatus,
    TargetType,
    VulnerabilityReport,
    VulnerabilitySeverity,
)
from systems.simula.inspector.workspace import TargetWorkspace

if TYPE_CHECKING:
    from systems.simula.inspector.analytics import InspectorAnalyticsEmitter
    from systems.simula.inspector.detonation import LiveDetonationChamber
    from systems.simula.inspector.inference import InvariantInferencer
    from systems.simula.inspector.injector import DynamicTaintInjector
    from systems.simula.inspector.prover import VulnerabilityProver
    from systems.simula.inspector.remediation import InspectorRepairOrchestrator, RepairAgent
    from systems.simula.inspector.shield import AutonomousShield
    from systems.simula.inspector.slicer import SemanticSlicer
    from systems.simula.inspector.taint_types import (
        CrossServiceAttackSurface,
        TaintGraph,
    )
    from systems.simula.inspector.temporal import ConcurrencyProver
    from systems.simula.inspector.topology import TopologyDetonationChamber
    from systems.simula.inspector.verifier import AdversarialVerifier

logger = structlog.get_logger().bind(system="simula.inspector.service")


# ── Predefined Attack Goals ──────────────────────────────────────────────────
# Focused on realistic attack vectors, not fuzzing noise.

PREDEFINED_ATTACK_GOALS: list[str] = [
    "Unauthenticated access to protected resource",
    "SQL injection in user input",
    "Broken access control: user A can access user B's data",
    "Privilege escalation: regular user can call admin function",
    "Reentrancy: contract can call itself recursively",
    "Unvalidated redirect",
    "Race condition: concurrent access violates invariant",
]

# Modules that a PoC script must never import - these allow host-level
# side effects beyond HTTP requests.
_FORBIDDEN_POC_MODULES = frozenset({
    "subprocess", "socket", "ctypes", "pickle", "shelve",
    "marshal", "shutil", "tempfile", "multiprocessing",
})


class InspectorService:
    """
    Coordinates the full hunting pipeline from target to vulnerability reports.

    Integrates:
      - TargetIngestor     (Phase 3) - clone repos, map attack surfaces
      - VulnerabilityProver (Phase 4+5) - Z3 constraint inversion + PoC gen
      - InspectorRepairOrchestrator (Phase 6) - autonomous patch generation
      - InspectorAnalyticsEmitter (Phase 9) - structlog event instrumentation
      - InspectorAnalyticsView (Phase 9) - aggregate vulnerability statistics
    """

    def __init__(
        self,
        prover: VulnerabilityProver,
        config: InspectorConfig,
        *,
        eos_root: Path | None = None,
        analytics: InspectorAnalyticsEmitter | None = None,
        remediation: InspectorRepairOrchestrator | None = None,
        verifier: AdversarialVerifier | None = None,
        temporal_prover: ConcurrencyProver | None = None,
        slicer: SemanticSlicer | None = None,
        repair_agent: RepairAgent | None = None,
        inferencer: InvariantInferencer | None = None,
        detonation_chamber: LiveDetonationChamber | None = None,
        topology_chamber: TopologyDetonationChamber | None = None,
        taint_injector: DynamicTaintInjector | None = None,
        shield: AutonomousShield | None = None,
    ) -> None:
        """
        Args:
            prover: The Z3-backed vulnerability prover.
            config: Inspector authorization and resource configuration.
            eos_root: Path to the internal EOS codebase (for internal hunts).
            analytics: Optional analytics emitter for structured event logging.
            remediation: Optional remediation orchestrator for patch generation.
            verifier: Optional adversarial verifier (Agent Blue) for false-positive
                      elimination via multi-agent debate.
            temporal_prover: Optional concurrency prover for race conditions and
                             double-spend vulnerabilities.
            slicer: Optional semantic slicer for reducing context size before Z3.
            repair_agent: Optional lightweight LLM-direct patcher (Phase 5).
                          Attaches verified patched_code to each VulnerabilityReport.
            inferencer: Optional invariant inferencer for autonomous business-logic
                        goal generation (Phase 13).
            detonation_chamber: Optional live detonation chamber for dynamic concolic
                        execution. When provided, the pipeline builds and runs the
                        target in a Docker container so PoC scripts fire against
                        a live server instead of generating static localhost scripts.
            topology_chamber: Optional topology detonation chamber for distributed
                        microservice targets. When provided, the pipeline parses
                        docker-compose files, injects an eBPF sidecar for cross-
                        boundary taint tracking, and orchestrates the full cluster.
                        Takes priority over the single-container detonation_chamber.
            shield: Optional AutonomousShield for synthesizing and validating
                        eBPF XDP network filters in response to critical anomalies.
                        When provided, proven vulnerabilities trigger LLM-driven
                        XDP filter generation with BCC dry-run compilation.
        """
        self._prover = prover
        self._config = config
        self._eos_root = eos_root
        self._analytics = analytics
        self._remediation = remediation
        self._verifier = verifier
        self._temporal = temporal_prover
        self._slicer = slicer
        self._repair_agent = repair_agent
        self._inferencer = inferencer
        self._detonation = detonation_chamber
        self._topology = topology_chamber
        self._taint_injector = taint_injector
        self._shield = shield
        self._detonation_ctx: Any = None  # Tracks active spin_up context manager
        self._topology_ctx: Any = None  # Tracks active topology spin_up context manager
        self._safety = InspectorSafetyGates()
        self._log = logger.bind(
            max_workers=config.max_workers,
            authorized_targets=len(config.authorized_targets),
        )

        # Pre-validate config via safety gates
        config_check = self._safety.validate_inspector_config(config)
        if not config_check:
            self._log.warning(
                "inspector_config_safety_warning",
                reason=getattr(config_check, "reason", "config validation failed"),
            )

        # Aggregate analytics view - ingests every InspectionResult automatically
        self._analytics_view = InspectorAnalyticsView()

        # Metrics
        self._hunts_completed: int = 0
        self._total_surfaces_mapped: int = 0
        self._total_vulnerabilities_found: int = 0
        self._total_patches_generated: int = 0

        # Hunt history (in-memory, capped)
        self._hunt_history: list[InspectionResult] = []
        self._max_history: int = 50

        # Vulnerability templates loaded once at construction
        self._templates: list[dict[str, Any]] = self._load_templates()

    # ── Public API ────────────────────────────────────────────────────────────

    async def hunt_external_repo(
        self,
        github_url: str,
        *,
        attack_goals: list[str] | None = None,
        generate_pocs: bool = True,
        generate_patches: bool = False,
    ) -> InspectionResult:
        """
        Clone an external GitHub repository and run the full hunting pipeline.

        Pipeline:
          1. Clone repo → TargetWorkspace
          2. Map attack surfaces via AST/regex scanning
          3. For each surface × attack goal, run VulnerabilityProver
          4. Optionally generate PoCs and patches

        Args:
            github_url: HTTPS URL of the repository to hunt.
            attack_goals: Custom attack goals (defaults to PREDEFINED_ATTACK_GOALS).
            generate_pocs: Whether to generate proof-of-concept exploit scripts.
            generate_patches: Whether to generate patches for found vulnerabilities.

        Returns:
            InspectionResult with all discovered vulnerabilities and optional patches.
        """
        # Step A: Authorization gate - target must be in authorized_targets
        if not any(
            github_url.startswith(t) or t in github_url
            for t in self._config.authorized_targets
        ):
            self._log.error(
                "hunt_target_not_authorized",
                url=github_url,
                authorized_targets=self._config.authorized_targets,
            )
            start = time.monotonic()
            return self._build_empty_result(
                new_id(), github_url, TargetType.EXTERNAL_REPO,
                start, utc_now(),
            )

        goals = attack_goals or PREDEFINED_ATTACK_GOALS
        start = time.monotonic()
        started_at = utc_now()
        hunt_id = new_id()

        log = self._log.bind(
            hunt_id=hunt_id,
            target_url=github_url,
            target_type="external_repo",
            attack_goals=len(goals),
        )

        if self._analytics:
            self._analytics.emit_hunt_started(
                github_url, "external_repo", hunt_id=hunt_id,
            )

        log.info("hunt_started", url=github_url)

        # Step 1: Clone and ingest - workspace is an async context manager;
        # the temp directory is nuked from orbit on exit regardless of outcome.
        try:
            ingestor = await TargetIngestor.ingest_from_github(
                github_url, clone_depth=self._config.clone_depth,
            )
        except Exception as exc:
            log.error("hunt_clone_failed", error=str(exc))
            if self._analytics:
                self._analytics.emit_hunt_error(
                    target_url=github_url,
                    hunt_id=hunt_id,
                    pipeline_stage="ingest",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            return self._build_empty_result(
                hunt_id, github_url, TargetType.EXTERNAL_REPO,
                start, started_at,
            )

        async with ingestor.workspace as workspace:
            # Safety gate: validate workspace isolation before proceeding
            ws_check = self._safety.validate_workspace_isolation(
                workspace, eos_root=self._eos_root,
            )
            if not ws_check:
                log.error("safety_gate_workspace_failed", reason=ws_check.reason)
                return self._build_empty_result(
                    hunt_id, github_url, TargetType.EXTERNAL_REPO,
                    start, started_at,
                )

            return await self._run_hunt_pipeline(
                hunt_id=hunt_id,
                ingestor=ingestor,
                workspace=workspace,
                target_url=github_url,
                target_type=TargetType.EXTERNAL_REPO,
                goals=goals,
                generate_pocs=generate_pocs,
                generate_patches=generate_patches,
                start=start,
                started_at=started_at,
                log=log,
            )

    async def hunt_internal_eos(
        self,
        *,
        attack_goals: list[str] | None = None,
        generate_pocs: bool = False,
        generate_patches: bool = False,
    ) -> InspectionResult:
        """
        Run the hunting pipeline against the internal EOS codebase.

        Useful for continuous automated security testing of EOS itself.
        The workspace is read-only - no temp files are created.

        Args:
            attack_goals: Custom attack goals (defaults to PREDEFINED_ATTACK_GOALS).
            generate_pocs: Whether to generate proof-of-concept exploit scripts.
            generate_patches: Whether to generate patches for found vulnerabilities.

        Returns:
            InspectionResult with discovered vulnerabilities.

        Raises:
            RuntimeError: If eos_root was not provided at construction time.
        """
        if self._eos_root is None:
            raise RuntimeError(
                "Cannot hunt internal EOS: eos_root was not provided. "
                "Pass eos_root= to InspectorService constructor."
            )

        goals = attack_goals or PREDEFINED_ATTACK_GOALS
        start = time.monotonic()
        started_at = utc_now()
        hunt_id = new_id()

        log = self._log.bind(
            hunt_id=hunt_id,
            target_url="internal_eos",
            target_type="internal_eos",
            attack_goals=len(goals),
        )

        if self._analytics:
            self._analytics.emit_hunt_started(
                "internal_eos", "internal_eos", hunt_id=hunt_id,
            )

        log.info("hunt_started", target="internal_eos")

        workspace = TargetWorkspace.internal(self._eos_root)
        ingestor = TargetIngestor(workspace=workspace)

        return await self._run_hunt_pipeline(
            hunt_id=hunt_id,
            ingestor=ingestor,
            workspace=workspace,
            target_url="internal_eos",
            target_type=TargetType.INTERNAL_EOS,
            goals=goals,
            generate_pocs=generate_pocs,
            generate_patches=generate_patches,
            start=start,
            started_at=started_at,
            log=log,
        )

    async def generate_patches(
        self,
        hunt_result: InspectionResult,
        workspace: TargetWorkspace | None = None,
    ) -> dict[str, str]:
        """
        Generate patches for all vulnerabilities in a completed InspectionResult.

        For each vulnerability, calls InspectorRepairOrchestrator.generate_patch()
        and returns a mapping of vulnerability_id → patch diff.

        Args:
            hunt_result: A completed InspectionResult with vulnerabilities.
            workspace: Target workspace to patch in. Required if the
                       hunt's workspace has already been cleaned up.

        Returns:
            Dict mapping vulnerability ID → patch diff string.

        Raises:
            RuntimeError: If remediation orchestrator is not available.
        """
        if self._remediation is None:
            raise RuntimeError(
                "Remediation is not available. Provide a InspectorRepairOrchestrator "
                "when constructing InspectorService."
            )

        if not hunt_result.vulnerabilities_found:
            return {}

        log = self._log.bind(
            hunt_id=hunt_result.id,
            vulnerabilities=len(hunt_result.vulnerabilities_found),
        )
        log.info("patch_generation_started")

        # Swap workspace for the remediation run via public API
        if workspace is not None:
            self._remediation.set_workspace(workspace)

        remediation_results = await self._remediation.generate_patches_batch(
            hunt_result.vulnerabilities_found,
        )

        patches: dict[str, str] = {}
        patched_count = 0

        for vuln_id, result in remediation_results.items():
            if result.status == RemediationStatus.PATCHED and result.final_patch_diff:
                patches[vuln_id] = result.final_patch_diff
                patched_count += 1

                if self._analytics:
                    self._analytics.emit_patch_generated(
                        vuln_id=vuln_id,
                        repair_time_ms=result.total_duration_ms,
                        patch_size_bytes=len(result.final_patch_diff.encode("utf-8")),
                        hunt_id=hunt_result.id,
                        target_url=hunt_result.target_url,
                    )

        self._total_patches_generated += patched_count
        log.info(
            "patch_generation_complete",
            total=len(remediation_results),
            patched=patched_count,
            failed=len(remediation_results) - patched_count,
        )

        return patches

    def validate_poc(
        self,
        poc_code: str,
        authorized_target: str | None = None,
    ) -> bool:
        """
        Validate that a proof-of-concept script does not reach unauthorized targets.

        Delegates to InspectorSafetyGates.validate_poc_execution() for deep
        validation (syntax, forbidden imports, dangerous calls, URL domain
        authorization) plus a pre-check on the explicit authorized_target.

        Args:
            poc_code: The Python PoC script to validate.
            authorized_target: The target domain the PoC should hit (if any).

        Returns:
            True if the PoC passes safety validation, False otherwise.
        """
        # Check authorized target is in config
        if authorized_target is not None:
            if authorized_target not in self._config.authorized_targets:
                self._log.warning(
                    "poc_unauthorized_target",
                    target=authorized_target,
                    authorized=self._config.authorized_targets,
                )
                return False

        # Delegate full validation to safety gates
        result = self._safety.validate_poc_execution(
            poc_code,
            self._config.authorized_targets,
            sandbox_timeout_seconds=self._config.sandbox_timeout_seconds,
        )
        if not result:
            self._log.warning(
                "poc_safety_gate_failed",
                gate=getattr(result, "gate", "unknown"),
                reason=getattr(result, "reason", "validation failed"),
            )
            return False
        return result.passed

    def get_hunt_history(self, limit: int = 20) -> list[InspectionResult]:
        """Return recent hunt results (newest first)."""
        return list(reversed(self._hunt_history[-limit:]))

    @property
    def analytics_view(self) -> InspectorAnalyticsView:
        """Aggregate analytics across all completed hunts."""
        return self._analytics_view

    @property
    def stats(self) -> dict[str, Any]:
        """Service-level metrics for observability."""
        result: dict[str, Any] = {
            "hunts_completed": self._hunts_completed,
            "total_surfaces_mapped": self._total_surfaces_mapped,
            "total_vulnerabilities_found": self._total_vulnerabilities_found,
            "total_patches_generated": self._total_patches_generated,
            "hunt_history_size": len(self._hunt_history),
            "config": {
                "max_workers": self._config.max_workers,
                "sandbox_timeout_seconds": self._config.sandbox_timeout_seconds,
                "authorized_targets": len(self._config.authorized_targets),
                "log_analytics": self._config.log_vulnerability_analytics,
                "clone_depth": self._config.clone_depth,
            },
            "remediation_available": self._remediation is not None,
            "verifier_available": self._verifier is not None,
            "temporal_available": self._temporal is not None,
            "detonation_available": self._detonation is not None,
            "analytics_available": self._analytics is not None,
            "analytics_summary": self._analytics_view.summary,
        }
        # Include emitter health metrics when available
        if self._analytics is not None:
            result["emitter_stats"] = self._analytics.stats
        return result

    # ── Core Pipeline ─────────────────────────────────────────────────────────

    async def _run_hunt_pipeline(
        self,
        *,
        hunt_id: str,
        ingestor: TargetIngestor,
        workspace: TargetWorkspace,
        target_url: str,
        target_type: TargetType,
        goals: list[str],
        generate_pocs: bool,
        generate_patches: bool,
        start: float,
        started_at: Any,
        log: Any,
    ) -> InspectionResult:
        """
        Execute the full hunt pipeline: map → prove → (poc) → (patch) → report.

        This is the shared core between hunt_external_repo and hunt_internal_eos.
        """
        # Step 2: Map attack surfaces
        log.info("mapping_attack_surfaces")
        surfaces: list[AttackSurface] = []
        try:
            surfaces = await ingestor.map_attack_surfaces()
        except Exception as exc:
            log.error("surface_mapping_failed", error=str(exc))
            if self._analytics:
                self._analytics.emit_surface_mapping_failed(
                    target_url=target_url,
                    hunt_id=hunt_id,
                    error_message=str(exc),
                )

        for surface in surfaces:
            if self._analytics:
                self._analytics.emit_attack_surface_discovered(
                    surface_type=surface.surface_type.value,
                    entry_point=surface.entry_point,
                    file_path=surface.file_path,
                    target_url=target_url,
                    hunt_id=hunt_id,
                    line_number=surface.line_number,
                )

        log.info("surfaces_mapped", total=len(surfaces))

        if not surfaces:
            log.info("no_surfaces_found")
            return self._build_empty_result(
                hunt_id, target_url, target_type, start, started_at,
            )

        # Step 3: Extract context code for surfaces that don't have it
        for surface in surfaces:
            if not surface.context_code:
                try:
                    context = await ingestor.extract_context_code(surface)
                    surface.context_code = context
                except Exception:
                    pass  # best-effort

        # Step 3b: Semantic slicing - strip boilerplate/logging from each surface's
        # context before it reaches Z3, preventing state explosion on large codebases.
        if self._slicer:
            for surface in surfaces:
                if not surface.context_code:
                    continue
                original_size = len(surface.context_code)
                # Use a representative goal for slicing (first goal is sufficient;
                # the slicer focuses on data-flow paths, not specific goal wording).
                representative_goal = goals[0] if goals else "vulnerability"
                sliced_code = await self._slicer.slice_context(
                    surface.context_code, representative_goal
                )
                surface.context_code = sliced_code
                log.debug(
                    "context_sliced",
                    entry_point=surface.entry_point,
                    original_size=original_size,
                    sliced_size=len(sliced_code),
                )

        # Step 3c: Autonomous Invariant Inference - for each surface, ask the LLM
        # to deduce the business-logic invariants implicit in the sliced code and
        # translate each one into a Z3 violation goal.  These per-surface goals are
        # appended to the surface's own attack_goals (not mutating the global list)
        # by collecting them into a per-surface map consumed in _prove_all.
        #
        # If inference fails for a surface the pipeline continues unchanged -
        # the inferencer always returns [] rather than raising.
        surface_inferred_goals: dict[str, list[str]] = {}
        if self._inferencer:
            for surface in surfaces:
                if not surface.context_code:
                    continue
                inferred = await self._inferencer.infer_vulnerability_goals(
                    surface.context_code
                )
                if inferred:
                    surface_inferred_goals[surface.id] = inferred
                    log.info(
                        "invariants_inferred",
                        entry_point=surface.entry_point,
                        file_path=surface.file_path,
                        inferred_count=len(inferred),
                    )

        # Flatten all inferred goals into the shared goals list (deduplicated) so
        # the prover matrix covers every novel goal across every surface.
        all_inferred: list[str] = []
        for inferred_list in surface_inferred_goals.values():
            for g in inferred_list:
                if g not in goals and g not in all_inferred:
                    all_inferred.append(g)

        if all_inferred:
            goals = goals + all_inferred
            log.info(
                "inference_pass_complete",
                surfaces_with_goals=len(surface_inferred_goals),
                total_inferred_goals=len(all_inferred),
            )

        # Step D: Augment goals with template-derived attack descriptions
        # for surfaces whose type matches a loaded template.
        template_goals = self._template_goals_for_surfaces(surfaces)
        effective_goals = goals + [g for g in template_goals if g not in goals]

        if template_goals:
            log.info(
                "template_goals_applied",
                template_count=len(self._templates),
                extra_goals=len(effective_goals) - len(goals),
            )

        # Step 3d: Dynamic Concolic Execution - spin up a live environment so PoC
        # scripts can fire HTTP requests against the actual running application.
        #
        # Priority order:
        #   1. Topology chamber (distributed docker-compose cluster with eBPF sidecar)
        #   2. Single-container detonation chamber (Dockerfile-based)
        #   3. Static analysis fallback (original target_url)
        live_env: dict[str, Any] | None = None
        topology_env: Any = None  # TopologyContext | None
        prover_target_url = target_url  # default: static analysis URL
        taint_graph: TaintGraph | None = None

        if workspace.is_external:
            # Attempt 1: Topology chamber - distributed microservice cluster
            if self._topology:
                log.info("topology_chamber_starting")
                try:
                    self._topology_ctx = self._topology.spin_up_cluster(workspace.root)
                    topology_env = await self._topology_ctx.__aenter__()
                except Exception as exc:
                    log.error(
                        "topology_chamber_startup_failed",
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                    topology_env = None

                if topology_env is not None:
                    prover_target_url = topology_env.entry_url
                    log.info(
                        "topology_chamber_active",
                        entry_url=prover_target_url,
                        services=list(topology_env.internal_services.keys()),
                        compose_file=str(topology_env.compose_file_path),
                        taint_collector_url=topology_env.taint_collector_url,
                        taint_collector_status=topology_env.taint_collector_status,
                    )

                    # ── Taint collection: inject marker, wait for warmup, collect ──
                    taint_graph: TaintGraph | None = None
                    if topology_env.taint_collector_url and self._topology:
                        # Inject a taint marker so the eBPF collector can track it
                        await self._topology.inject_taint(
                            b"SIMULA_TAINT_MARKER", "inspector_pipeline"
                        )
                        # Brief warmup for eBPF programs to start capturing
                        await asyncio.sleep(2.0)
                        taint_graph = await self._topology.collect_taint_events()
                        if taint_graph:
                            log.info(
                                "taint_graph_collected",
                                nodes=len(taint_graph.nodes),
                                edges=len(taint_graph.edges),
                                sources=len(taint_graph.sources),
                                has_unsanitized_path=taint_graph.has_unsanitized_path,
                            )

                    # ── Annotate surfaces with service_name + taint_context ──
                    if topology_env.internal_services:
                        self._annotate_surfaces_with_taint(
                            surfaces, topology_env, taint_graph, log,
                        )

                    # ── Synthesize cross-service surfaces from taint flows ──
                    if taint_graph and taint_graph.edges:
                        cross_surfaces = self._synthesize_cross_service_surfaces(
                            surfaces, taint_graph, topology_env, workspace, log,
                        )
                        if cross_surfaces:
                            surfaces.extend(cross_surfaces)
                            log.info(
                                "cross_service_surfaces_synthesized",
                                count=len(cross_surfaces),
                            )

            # Attempt 2: Single-container detonation chamber (only if topology
            # didn't yield a result - either absent or no compose file found)
            if topology_env is None and self._detonation:
                log.info("detonation_chamber_starting")
                try:
                    self._detonation_ctx = self._detonation.spin_up(workspace.root)
                    live_env = await self._detonation_ctx.__aenter__()
                except Exception as exc:
                    log.error(
                        "detonation_chamber_startup_failed",
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                    live_env = None

                if live_env is not None:
                    prover_target_url = live_env["mapped_url"]
                    log.info(
                        "detonation_chamber_active",
                        mapped_url=prover_target_url,
                        container_id=live_env["container_id"],
                        detected_language=live_env.get("detected_language"),
                    )

            # If neither chamber produced a live environment, fall back to static
            if topology_env is None and live_env is None:
                log.warning(
                    "detonation_chamber_fallback_to_static",
                    reason="no live environment available (topology and container both absent or failed)",
                )

            # ── Dynamic taint injection ──────────────────────────────────────
            # Fire marked HTTP requests into the live cluster and attach the
            # proven propagation chain to each surface.  Requires both a live
            # URL and an injector instance.
            if self._taint_injector is not None and prover_target_url:
                log.info(
                    "dynamic_taint_injection_starting",
                    surfaces=len(surfaces),
                    target_url=prover_target_url,
                )
                for surface in surfaces:
                    try:
                        chain = await self._taint_injector.inject_and_trace(
                            prover_target_url, surface
                        )
                        if chain:
                            surface.verified_data_path = chain
                            log.info(
                                "dynamic_data_flow_proven",
                                token=chain[0].token if chain else "",
                                hop_count=len(chain),
                                entry_point=surface.entry_point,
                                hops=[
                                    f"{e.src_service} → {e.dst_service}"
                                    for e in chain
                                ],
                            )
                    except Exception as exc:
                        log.debug(
                            "dynamic_taint_injection_error",
                            entry_point=surface.entry_point,
                            error=str(exc),
                        )

        try:
            vulnerabilities, patches = await self._prove_and_verify(
                surfaces=surfaces,
                effective_goals=effective_goals,
                target_url=target_url,
                prover_target_url=prover_target_url,
                generate_pocs=generate_pocs,
                generate_patches=generate_patches,
                hunt_id=hunt_id,
                workspace=workspace,
                taint_graph=taint_graph,
                log=log,
            )
        finally:
            # Tear down whichever chamber was active, regardless of outcome
            if topology_env is not None:
                try:
                    await self._topology_ctx.__aexit__(None, None, None)
                except Exception as exc:
                    log.warning("topology_chamber_teardown_error", error=str(exc))
            if live_env is not None:
                try:
                    await self._detonation_ctx.__aexit__(None, None, None)
                except Exception as exc:
                    log.warning("detonation_chamber_teardown_error", error=str(exc))

        # Step 5b: Autonomous Shield - synthesize XDP filters for critical findings
        # For each HIGH/CRITICAL vulnerability, ask the LLM to generate an XDP
        # program that drops packets matching the malicious signature.  The generated
        # C is compiled in-memory via BCC (dry-run only, no live attachment).
        if self._shield and vulnerabilities:
            shield_candidates = [
                v for v in vulnerabilities
                if v.severity in (
                    VulnerabilitySeverity.CRITICAL,
                    VulnerabilitySeverity.HIGH,
                )
            ]
            for vuln in shield_candidates:
                try:
                    alert_ctx: dict[str, object] = {
                        "vulnerability_id": vuln.id,
                        "vulnerability_class": vuln.vulnerability_class.value,
                        "severity": vuln.severity.value,
                        "attack_goal": vuln.attack_goal,
                        "z3_counterexample": vuln.z3_counterexample,
                        "entry_point": vuln.attack_surface.entry_point,
                        "file_path": vuln.attack_surface.file_path,
                    }
                    xdp_code = await self._shield.synthesize_xdp_filter(alert_ctx)
                    if not xdp_code:
                        log.warning("shield_synthesis_empty", vulnerability_id=vuln.id)
                        continue

                    try:
                        import asyncio as _asyncio
                        _loop = _asyncio.get_event_loop()
                        bpf_module = await _loop.run_in_executor(
                            None, lambda: self._shield.deploy_filter_live("lo", xdp_code)
                        )
                        vuln.xdp_filter_code = xdp_code

                        log.critical(
                            "CRITICAL_SHIELD_ATTACHED",
                            vulnerability_id=vuln.id,
                            vulnerability_class=vuln.vulnerability_class.value,
                            severity=vuln.severity.value,
                            xdp_code_size=len(xdp_code),
                            interface="lo",
                        )

                        # Run telemetry listener off-thread so it doesn't block the event loop
                        _loop.run_in_executor(
                            None, lambda: self._shield.listen_for_telemetry(bpf_module)
                        )

                    except Exception as e:
                        log.error("live_attachment_failed", error=str(e))
                except Exception as exc:
                    log.warning(
                        "shield_synthesis_error",
                        vulnerability_id=vuln.id,
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )

        # Step 6: Build result - timestamps captured at correct points
        elapsed_ms = int((time.monotonic() - start) * 1000)

        result = InspectionResult(
            id=hunt_id,
            target_url=target_url,
            target_type=target_type,
            surfaces_mapped=len(surfaces),
            attack_surfaces=surfaces,
            vulnerabilities_found=vulnerabilities,
            generated_patches=patches,
            total_duration_ms=elapsed_ms,
            started_at=started_at,
            completed_at=utc_now(),
        )

        # Update metrics
        self._hunts_completed += 1
        self._total_surfaces_mapped += len(surfaces)
        self._total_vulnerabilities_found += len(vulnerabilities)

        # Store in history (capped at _max_history)
        self._hunt_history.append(result)
        if len(self._hunt_history) > self._max_history:
            self._hunt_history = self._hunt_history[-self._max_history:]

        # Ingest into aggregate analytics view
        self._analytics_view.ingest_hunt_result(result)

        # Compute severity counts for analytics
        critical = sum(
            1 for v in vulnerabilities
            if v.severity == VulnerabilitySeverity.CRITICAL
        )
        high = sum(
            1 for v in vulnerabilities
            if v.severity == VulnerabilitySeverity.HIGH
        )

        # Emit analytics + flush buffer
        if self._analytics:
            self._analytics.emit_hunt_completed(
                target_url=target_url,
                hunt_id=hunt_id,
                total_surfaces=len(surfaces),
                total_vulnerabilities=len(vulnerabilities),
                total_time_ms=elapsed_ms,
                total_pocs=sum(1 for v in vulnerabilities if v.proof_of_concept_code),
                total_patches=len(patches),
                critical_count=critical,
                high_count=high,
            )
            # Flush any remaining buffered events to TSDB
            await self._analytics.flush()

        # Log each vulnerability as a distinct event (Iron Rule #4)
        for vuln in vulnerabilities:
            log.info(
                "vulnerability_discovered",
                vulnerability_id=vuln.id,
                vulnerability_class=vuln.vulnerability_class.value,
                severity=vuln.severity.value,
                attack_surface=vuln.attack_surface.entry_point,
                file_path=vuln.attack_surface.file_path,
                z3_counterexample=vuln.z3_counterexample[:200],
                has_poc=bool(vuln.proof_of_concept_code),
                has_patch=vuln.id in patches,
            )

        log.info(
            "hunt_completed",
            hunt_id=hunt_id,
            surfaces=len(surfaces),
            vulnerabilities=len(vulnerabilities),
            patches=len(patches),
            duration_ms=elapsed_ms,
        )

        return result

    async def _prove_and_verify(
        self,
        *,
        surfaces: list[AttackSurface],
        effective_goals: list[str],
        target_url: str,
        prover_target_url: str,
        generate_pocs: bool,
        generate_patches: bool,
        hunt_id: str,
        workspace: TargetWorkspace,
        taint_graph: TaintGraph | None = None,
        log: Any,
    ) -> tuple[list[VulnerabilityReport], dict[str, str]]:
        """
        Run the prove → temporal → verify → repair → patch phases.

        Separated from _run_hunt_pipeline so the detonation chamber context
        manager can wrap this entire block - the live container stays up for
        all proving passes, then is torn down after.

        Args:
            prover_target_url: The URL provers use for PoC generation. When a
                live detonation chamber is active this is the mapped container
                URL (e.g. http://127.0.0.1:54321); otherwise it equals target_url.
            target_url: The original target URL used for analytics/reporting.
            taint_graph: Optional eBPF-observed taint graph for cross-service proving.

        Returns:
            (vulnerabilities, patches) tuple.
        """
        # Step 4: Prove vulnerabilities across surfaces × goals
        vulnerabilities = await self._prove_all(
            surfaces=surfaces,
            goals=effective_goals,
            target_url=prover_target_url,
            generate_pocs=generate_pocs,
            hunt_id=hunt_id,
            taint_graph=taint_graph,
            log=log,
        )

        log.info(
            "proving_complete",
            total_vulnerabilities=len(vulnerabilities),
            critical=sum(
                1 for v in vulnerabilities
                if v.severity == VulnerabilitySeverity.CRITICAL
            ),
            high=sum(
                1 for v in vulnerabilities
                if v.severity == VulnerabilitySeverity.HIGH
            ),
            dynamic_execution=prover_target_url != target_url,
        )

        # Step 4a: Temporal engine - concurrent race-condition proofs
        # Runs in parallel across all surfaces, then merged into vulnerabilities
        # before Agent Blue's adversarial debate so races are also scrutinised.
        if self._temporal and surfaces:
            log.info("temporal_engine_started", surfaces=len(surfaces))
            temporal_results = await asyncio.gather(
                *[
                    self._temporal.prove_race_condition(
                        s, target_url=prover_target_url,
                    )
                    for s in surfaces
                ],
                return_exceptions=True,
            )
            temporal_reports: list[VulnerabilityReport] = []
            for r in temporal_results:
                if isinstance(r, VulnerabilityReport):
                    temporal_reports.append(r)
                elif isinstance(r, Exception):
                    log.warning(
                        "temporal_proof_error",
                        error=str(r),
                    )
            if temporal_reports:
                log.info(
                    "temporal_engine_complete",
                    race_conditions_found=len(temporal_reports),
                )
                vulnerabilities = vulnerabilities + temporal_reports
            else:
                log.info("temporal_engine_complete", race_conditions_found=0)

        # Step 4b: Adversarial verification pass - Agent Blue debates each PoC
        if self._verifier and vulnerabilities:
            log.info("adversarial_verification_started", findings=len(vulnerabilities))
            vulnerabilities = [
                await self._verifier.verify_finding(r) for r in vulnerabilities
            ]
            refuted = [
                v for v in vulnerabilities
                if v.severity == VulnerabilitySeverity.FALSE_POSITIVE
            ]
            for fp in refuted:
                log.info(
                    "vulnerability_refuted_by_defender",
                    vulnerability_id=fp.id,
                    vulnerability_class=fp.vulnerability_class.value,
                    defender_notes=(fp.defender_notes or "")[:200],
                )
            # Discard false positives - only real findings proceed to patches/report
            vulnerabilities = [
                v for v in vulnerabilities
                if v.severity != VulnerabilitySeverity.FALSE_POSITIVE
            ]
            log.info(
                "adversarial_verification_complete",
                confirmed=len(vulnerabilities),
                refuted=len(refuted),
            )

        # Step 4c: RepairAgent - attach verified patched_code to each report
        # Runs only when generate_patches=True and the lightweight repair_agent is set.
        # A failed patch attempt is non-fatal; the report is delivered without patched_code.
        if generate_patches and self._repair_agent and vulnerabilities:
            log.info("repair_agent_started", findings=len(vulnerabilities))
            for vuln in vulnerabilities:
                try:
                    patched = await self._repair_agent.generate_and_verify_patch(vuln)
                    if patched:
                        # model_copy produces a new instance; reassign in-place
                        idx = vulnerabilities.index(vuln)
                        vulnerabilities[idx] = vuln.model_copy(
                            update={"patched_code": patched}
                        )
                        log.info(
                            "patch_generated_and_verified",
                            vulnerability_id=vuln.id,
                            vulnerability_class=vuln.vulnerability_class.value,
                            severity=vuln.severity.value,
                        )
                except Exception as exc:
                    log.warning(
                        "repair_agent_patch_failed",
                        vulnerability_id=vuln.id,
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )

        # Step 5: Generate patches if requested and remediation is available
        patches: dict[str, str] = {}
        if generate_patches and vulnerabilities and self._remediation is not None:
            log.info("generating_patches", vulnerabilities=len(vulnerabilities))
            try:
                # Set remediation workspace via public API (not private attribute)
                self._remediation.set_workspace(workspace)

                remediation_results = await self._remediation.generate_patches_batch(
                    vulnerabilities,
                )
                for vuln_id, rem_result in remediation_results.items():
                    if (
                        rem_result.status == RemediationStatus.PATCHED
                        and rem_result.final_patch_diff
                    ):
                        patches[vuln_id] = rem_result.final_patch_diff
                        self._total_patches_generated += 1

                        if self._analytics:
                            self._analytics.emit_patch_generated(
                                vuln_id=vuln_id,
                                repair_time_ms=rem_result.total_duration_ms,
                                patch_size_bytes=len(
                                    rem_result.final_patch_diff.encode("utf-8")
                                ),
                                target_url=target_url,
                                hunt_id=hunt_id,
                            )
            except Exception as exc:
                log.error("patch_generation_failed", error=str(exc))
                if self._analytics:
                    self._analytics.emit_hunt_error(
                        target_url=target_url,
                        hunt_id=hunt_id,
                        pipeline_stage="remediation",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )

        return vulnerabilities, patches

    async def _prove_all(
        self,
        *,
        surfaces: list[AttackSurface],
        goals: list[str],
        target_url: str,
        generate_pocs: bool,
        hunt_id: str = "",
        taint_graph: TaintGraph | None = None,
        log: Any,
    ) -> list[VulnerabilityReport]:
        """
        Prove vulnerabilities across all surfaces × attack goals.

        Uses bounded concurrency (config.max_workers) to limit parallel
        Z3 + LLM calls. Each (surface, goal) pair is independently provable.
        Individual proof attempts are wrapped in asyncio.wait_for to enforce
        the configured sandbox_timeout_seconds.

        When a surface has a non-empty taint_context, the prover automatically
        switches to cross-service constraint encoding for that surface.
        """
        vulnerabilities: list[VulnerabilityReport] = []
        semaphore = asyncio.Semaphore(self._config.max_workers)
        timeout = self._config.sandbox_timeout_seconds

        # Build cross-service surface lookup for surfaces with taint context
        cross_surface_map: dict[str, CrossServiceAttackSurface] = {}
        if taint_graph is not None:
            cross_surface_map = self._build_cross_surface_map(surfaces, taint_graph)

        async def prove_one(
            surface: AttackSurface,
            goal: str,
        ) -> VulnerabilityReport | None:
            async with semaphore:
                try:
                    # Use cross-service prover when taint context is available
                    cross_surface = cross_surface_map.get(surface.id)
                    if cross_surface is not None:
                        report = await asyncio.wait_for(
                            self._prover.prove_cross_service_vulnerability(
                                cross_surface=cross_surface,
                                attack_goal=goal,
                                target_url=target_url,
                                taint_graph=taint_graph,
                                generate_poc=generate_pocs,
                                config=self._config,
                            ),
                            timeout=timeout,
                        )
                    else:
                        report = await asyncio.wait_for(
                            self._prover.prove_vulnerability(
                                surface=surface,
                                attack_goal=goal,
                                target_url=target_url,
                                generate_poc=generate_pocs,
                                config=self._config,
                            ),
                            timeout=timeout,
                        )
                    if report is not None:
                        if self._analytics:
                            self._analytics.emit_vulnerability_proved(
                                vulnerability_class=report.vulnerability_class.value,
                                severity=report.severity.value,
                                z3_time_ms=0,  # prover tracks internally
                                target_url=target_url,
                                hunt_id=hunt_id,
                                vuln_id=report.id,
                                attack_goal=goal,
                                entry_point=surface.entry_point,
                            )
                        # Emit PoC analytics if one was generated
                        if report.proof_of_concept_code and self._analytics:
                            self._analytics.emit_poc_generated(
                                vuln_id=report.id,
                                poc_size_bytes=len(
                                    report.proof_of_concept_code.encode("utf-8")
                                ),
                                target_url=target_url,
                                hunt_id=hunt_id,
                            )
                    return report
                except TimeoutError:
                    log.warning(
                        "prove_vulnerability_timeout",
                        surface=surface.entry_point,
                        goal=goal[:80],
                        timeout_s=timeout,
                    )
                    if self._analytics:
                        self._analytics.emit_proof_timeout(
                            target_url=target_url,
                            hunt_id=hunt_id,
                            entry_point=surface.entry_point,
                            attack_goal=goal,
                            timeout_s=timeout,
                        )
                    return None
                except Exception as exc:
                    log.warning(
                        "prove_vulnerability_error",
                        surface=surface.entry_point,
                        goal=goal[:80],
                        error=str(exc),
                    )
                    if self._analytics:
                        self._analytics.emit_hunt_error(
                            target_url=target_url,
                            hunt_id=hunt_id,
                            pipeline_stage="prove",
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                        )
                    return None

        # Build task matrix: surface × goal
        tasks = [
            prove_one(surface, goal)
            for surface in surfaces
            for goal in goals
        ]

        log.info(
            "proving_started",
            total_tasks=len(tasks),
            surfaces=len(surfaces),
            goals=len(goals),
            max_workers=self._config.max_workers,
            timeout_s=timeout,
        )

        # Execute with bounded concurrency; exceptions already handled
        # inside prove_one, so return_exceptions=False is safe here.
        results = await asyncio.gather(*tasks)

        for result in results:
            if isinstance(result, VulnerabilityReport):
                vulnerabilities.append(result)

        return vulnerabilities

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_templates() -> list[dict[str, Any]]:
        """
        Load vulnerability templates from the templates/ directory.

        Each JSON file describes a framework-specific vulnerability pattern
        with Z3 encoding instructions and reproduction guidance. Templates
        are loaded once at construction and cached on the instance.

        Returns:
            List of parsed template dicts. Empty list if the directory
            does not exist or no JSON files are found.
        """
        templates_dir = Path(__file__).parent / "templates"
        if not templates_dir.is_dir():
            return []

        loaded: list[dict[str, Any]] = []
        for path in sorted(templates_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    loaded.append(data)
            except (json.JSONDecodeError, OSError):
                pass  # skip malformed/unreadable templates
        return loaded

    def _template_goals_for_surfaces(
        self,
        surfaces: list[AttackSurface],
    ) -> list[str]:
        """
        Derive additional attack goals from loaded templates for the given surfaces.

        A template contributes a goal when at least one surface's type appears
        in the template's ``target_surface_types`` list.  The goal string is
        built from the template's ``description`` field so the Prover LLM
        receives human-readable intent rather than raw JSON.

        Returns:
            Deduplicated list of extra goal strings derived from templates.
        """
        surface_types = {s.surface_type.value for s in surfaces}
        extra: list[str] = []
        seen: set[str] = set()

        for tmpl in self._templates:
            target_types: list[str] = tmpl.get("target_surface_types", [])
            if not any(t in surface_types for t in target_types):
                continue

            description: str = tmpl.get("description", "").strip()
            if not description or description in seen:
                continue

            # Attach Z3 encoding hints inline so the Prover LLM has full
            # context without needing to re-load the template itself.
            instructions: list[str] = tmpl.get("z3_encoding_instructions", [])
            if instructions:
                hint = " | ".join(instructions)
                goal = f"{description} [Z3 hints: {hint}]"
            else:
                goal = description

            seen.add(description)
            extra.append(goal)

        return extra

    # ── Taint integration helpers ────────────────────────────────────────────

    @staticmethod
    def _annotate_surfaces_with_taint(
        surfaces: list[AttackSurface],
        topology_env: Any,
        taint_graph: TaintGraph | None,
        log: Any,
    ) -> None:
        """
        Annotate attack surfaces with service_name and taint_context from
        the topology context and eBPF taint graph.

        Matches surfaces to services by comparing file paths against the
        service source directory mappings in the topology context.
        """
        service_dirs: dict[str, str] = {}
        for svc_name, svc_info in (topology_env.internal_services or {}).items():
            if isinstance(svc_info, dict):
                src_dir = svc_info.get("source_dir", "")
                if src_dir:
                    service_dirs[svc_name] = src_dir

        annotated = 0
        for surface in surfaces:
            # Match surface file_path to a service source directory
            for svc_name, src_dir in service_dirs.items():
                if surface.file_path.startswith(src_dir) or src_dir in surface.file_path:
                    surface.service_name = svc_name
                    break

            # Build taint context JSON for surfaces with a matched service
            if surface.service_name and taint_graph:
                taint_ctx = _build_surface_taint_context(
                    surface.service_name, taint_graph,
                )
                if taint_ctx:
                    surface.taint_context = taint_ctx
                    annotated += 1

        if annotated:
            log.info(
                "surfaces_taint_annotated",
                annotated=annotated,
                total=len(surfaces),
            )

    @staticmethod
    def _synthesize_cross_service_surfaces(
        existing_surfaces: list[AttackSurface],
        taint_graph: TaintGraph | None,
        topology_env: Any,
        workspace: TargetWorkspace,
        log: Any,
    ) -> list[AttackSurface]:
        """
        Synthesize CROSS_SERVICE_ENDPOINT surfaces from taint flows that
        connect services without existing surfaces.

        When eBPF observes data flowing from Service A to Service B but
        we don't have an explicit surface for the receiving endpoint in B,
        we create a synthetic surface so the prover can still attempt
        cross-service proving.
        """
        if not taint_graph:
            return []

        existing_services = {s.service_name for s in existing_surfaces if s.service_name}
        synthesized: list[AttackSurface] = []

        for edge in taint_graph.edges:
            # Only synthesize if the target service doesn't already have surfaces
            if edge.to_service in existing_services:
                continue

            surface = AttackSurface(
                entry_point=f"cross_service:{edge.from_service}->{edge.to_service}",
                surface_type=AttackSurfaceType.CROSS_SERVICE_ENDPOINT,
                file_path="<synthesized from eBPF taint flow>",
                service_name=edge.to_service,
                taint_context=json.dumps({
                    "from_service": edge.from_service,
                    "to_service": edge.to_service,
                    "flow_type": edge.flow_type.value,
                    "event_count": edge.event_count,
                }),
                context_code=f"# Synthesized cross-service endpoint\n"
                             f"# Taint flow: {edge.from_service} -> {edge.to_service}\n"
                             f"# Flow type: {edge.flow_type.value}\n"
                             f"# Events observed: {edge.event_count}\n",
            )
            synthesized.append(surface)
            existing_services.add(edge.to_service)

            log.debug(
                "cross_service_surface_synthesized",
                from_service=edge.from_service,
                to_service=edge.to_service,
                flow_type=edge.flow_type.value,
            )

        return synthesized

    def _build_cross_surface_map(
        self,
        surfaces: list[AttackSurface],
        taint_graph: TaintGraph,
    ) -> dict[str, CrossServiceAttackSurface]:
        """
        Build a map of surface ID → CrossServiceAttackSurface for surfaces
        that have taint context (eligible for cross-service proving).
        """
        from systems.simula.inspector.taint_types import (
            CrossServiceAttackSurface,
        )

        result: dict[str, CrossServiceAttackSurface] = {}

        for surface in surfaces:
            if not surface.taint_context:
                continue

            # Determine involved services from taint graph flows
            involved_services: list[str] = []
            if surface.service_name:
                involved_services.append(surface.service_name)

            flows_from = taint_graph.flows_from(surface.service_name or "")
            flows_to = taint_graph.flows_to(surface.service_name or "")
            all_flows = flows_from + flows_to

            for flow in all_flows:
                if flow.from_service not in involved_services:
                    involved_services.append(flow.from_service)
                if flow.to_service not in involved_services:
                    involved_services.append(flow.to_service)

            # Extract taint sources and sinks from graph
            taint_sources = [
                s for s in taint_graph.sources
                if s.source_service in involved_services
            ]
            taint_sinks = [
                s for s in taint_graph.sinks
                if s.sink_service in involved_services
            ]

            cross_surface = CrossServiceAttackSurface(
                primary_surface=surface,
                service_name=surface.service_name or "unknown",
                taint_sources=taint_sources,
                taint_sinks=taint_sinks,
                cross_service_flows=all_flows,
                cross_service_context_code=surface.context_code,
                involved_services=involved_services,
            )
            result[surface.id] = cross_surface

        return result

    @staticmethod
    def _build_empty_result(
        hunt_id: str,
        target_url: str,
        target_type: TargetType,
        start: float,
        started_at: Any,
    ) -> InspectionResult:
        """Build an empty InspectionResult (clone failed or no surfaces found)."""
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return InspectionResult(
            id=hunt_id,
            target_url=target_url,
            target_type=target_type,
            total_duration_ms=elapsed_ms,
            started_at=started_at,
            completed_at=utc_now(),
        )


# ── Module-level helpers ──────────────────────────────────────────────────────


def _build_surface_taint_context(
    service_name: str,
    taint_graph: TaintGraph,
) -> str:
    """
    Build a JSON taint context string for a single surface from the taint graph.

    Returns empty string if no relevant flows exist for this service.
    """
    flows_from = taint_graph.flows_from(service_name)
    flows_to = taint_graph.flows_to(service_name)

    if not flows_from and not flows_to:
        return ""

    context = {
        "service": service_name,
        "outbound_flows": [
            {
                "to_service": f.to_service,
                "flow_type": f.flow_type.value,
                "event_count": f.event_count,
            }
            for f in flows_from
        ],
        "inbound_flows": [
            {
                "from_service": f.from_service,
                "flow_type": f.flow_type.value,
                "event_count": f.event_count,
            }
            for f in flows_to
        ],
        "sources": [
            {
                "variable_name": s.variable_name,
                "entry_point": s.entry_point,
                "taint_level": s.taint_level.value,
            }
            for s in taint_graph.sources
            if s.source_service == service_name
        ],
        "sinks": [
            {
                "variable_name": s.variable_name,
                "sink_type": s.sink_type.value,
                "is_sanitized": s.is_sanitized,
            }
            for s in taint_graph.sinks
            if s.sink_service == service_name
        ],
    }

    return json.dumps(context)
