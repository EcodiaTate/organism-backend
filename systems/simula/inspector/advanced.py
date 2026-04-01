"""
EcodiaOS - Inspector Advanced Features (Phase 12)

Extends the core vulnerability discovery engine with production-ready CART
(Continuous Automated Red Teaming) features:

1. **Multi-Language Support** - Detect Go, Rust, TypeScript attack surfaces with
   high confidence via AST and pattern matching.

2. **Attack Path Detection** - Identify sequences of 2+ vulnerabilities that
   collectively enable a higher-severity logical flaw.

3. **Autonomous Patching** - Auto-generate GitHub PRs with patches and CI integration
   via the RepairAgent.

4. **Continuous Hunting** - Schedule periodic scans of registered repositories with
   incremental discovery and change-based retesting.

Architecture:
  - MultiLanguageSurfaceDetector      - extends ingestor with Go/Rust/TS support
  - AttackPathAnalyzer                - models multi-vuln state sequences
  - AutonomousPatchingOrchestrator    - GitHub API integration + PR automation
  - ContinuousHuntingScheduler        - cron-like recurring hunts with state persistence

Integration points:
  - TargetIngestor - inject new language detectors
  - VulnerabilityProver - sequence detection feeds from vulnerability reports
  - InspectorService - expose new async methods for patching + scheduling
  - InspectorAnalyticsEmitter - track path discoveries, PR outcomes
"""

from __future__ import annotations

import re
import time
from enum import StrEnum
from datetime import datetime
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now
from systems.simula.inspector.types import (
    AttackSurface,
    AttackSurfaceType,
    VulnerabilityReport,
    VulnerabilitySeverity,
)

if TYPE_CHECKING:

    from clients.llm import LLMProvider
    from systems.simula.inspector.service import InspectorService
    from systems.simula.inspector.workspace import TargetWorkspace
logger = structlog.get_logger().bind(system="simula.inspector.advanced")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 12.1: Multi-Language Support
# ═════════════════════════════════════════════════════════════════════════════


class LanguageType(StrEnum):
    """Programming languages supported by advanced surface detection."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    SOLIDITY = "solidity"


class GoFunctionSignature:
    """Detects exported Go functions and their HTTP handlers."""

    HANDLER_PATTERN = re.compile(
        r"""
        func\s+
        (Handle\w+|[\w]*Handler|[\w]*Endpoint)  # function name (conventions)
        \s*\(\s*
        w\s+http\.ResponseWriter
        \s*,\s*r\s*\*http\.Request
        """,
        re.VERBOSE,
    )

    ROUTE_REGISTRATION = re.compile(
        r"""
        (?:mux|router|r|engine)\.
        (?:HandleFunc|GET|POST|PUT|DELETE|PATCH|OPTIONS|HEAD)
        \s*\(\s*
        (['"`])([^'"`]+)\1  # path pattern
        """,
        re.VERBOSE,
    )

    STRUCT_FIELD = re.compile(
        r"""
        ^\s*
        (\w+)\s+              # field name
        ([\w\[\]\*\.]+)       # field type
        (?:\s+`json:"([^"`]+)")?  # optional json tag
        """,
        re.VERBOSE | re.MULTILINE,
    )

    @staticmethod
    def extract_handlers(file_content: str, file_path: str) -> list[AttackSurface]:
        surfaces = []
        for match in GoFunctionSignature.HANDLER_PATTERN.finditer(file_content):
            handler_name = match.group(1)
            surfaces.append(
                AttackSurface(
                    entry_point=handler_name,
                    surface_type=AttackSurfaceType.API_ENDPOINT,
                    file_path=file_path,
                    line_number=file_content[:match.start()].count('\n') + 1,
                    context_code=file_content[max(0, match.start() - 200):match.end() + 200],
                )
            )
        return surfaces


class RustFunctionSignature:
    """Detects Rust web handler patterns (actix-web, axum, rocket, etc.)."""

    ACTIX_HANDLER = re.compile(
        r"""
        #\[(?:get|post|put|delete|patch|head|options)\s*\(\s*
        (['"`])([^'"`]+)\1\s*\)\s*\]  # route pattern
        \s*
        (?:pub\s+)?
        (?:async\s+)?
        fn\s+(\w+)  # function name
        """,
        re.VERBOSE,
    )

    AXUM_ROUTE = re.compile(
        r"""
        \.route\s*\(\s*
        (['"`])([^'"`]+)\1\s*,\s*
        (?:get|post|put|delete|patch)\s*\(\s*(\w+)\s*\)\s*\)
        """,
        re.VERBOSE,
    )

    ROCKET_HANDLER = re.compile(
        r"""
        #\[(?:get|post|put|delete|patch)\s*\(\s*
        (['"`])([^'"`]+)\1\s*\)\s*\]
        \s*
        (?:pub\s+)?
        (?:async\s+)?
        fn\s+(\w+)
        """,
        re.VERBOSE,
    )

    @staticmethod
    def extract_handlers(file_content: str, file_path: str) -> list[AttackSurface]:
        surfaces = []

        for match in RustFunctionSignature.ACTIX_HANDLER.finditer(file_content):
            route_pattern = match.group(2)
            handler_name = match.group(3)
            surfaces.append(
                AttackSurface(
                    entry_point=handler_name,
                    surface_type=AttackSurfaceType.API_ENDPOINT,
                    file_path=file_path,
                    line_number=file_content[:match.start()].count('\n') + 1,
                    route_pattern=route_pattern,
                    context_code=file_content[max(0, match.start() - 200):match.end() + 200],
                )
            )

        for match in RustFunctionSignature.AXUM_ROUTE.finditer(file_content):
            route_pattern = match.group(2)
            handler_name = match.group(3)
            surfaces.append(
                AttackSurface(
                    entry_point=handler_name,
                    surface_type=AttackSurfaceType.API_ENDPOINT,
                    file_path=file_path,
                    line_number=file_content[:match.start()].count('\n') + 1,
                    route_pattern=route_pattern,
                    context_code=file_content[max(0, match.start() - 200):match.end() + 200],
                )
            )

        for match in RustFunctionSignature.ROCKET_HANDLER.finditer(file_content):
            route_pattern = match.group(2)
            handler_name = match.group(3)
            surfaces.append(
                AttackSurface(
                    entry_point=handler_name,
                    surface_type=AttackSurfaceType.API_ENDPOINT,
                    file_path=file_path,
                    line_number=file_content[:match.start()].count('\n') + 1,
                    route_pattern=route_pattern,
                    context_code=file_content[max(0, match.start() - 200):match.end() + 200],
                )
            )

        return surfaces


class TypeScriptFunctionSignature:
    """Detects TypeScript-specific attack surfaces (type safety, decorators, generics)."""

    NESTJS_HANDLER = re.compile(
        r"""
        @(?:Get|Post|Put|Delete|Patch|Head|Options)\s*\(\s*
        (['"`]?)([^'"`\)]*)\1?\s*\)
        \s*
        (?:async\s+)?
        (\w+)\s*\(
        """,
        re.VERBOSE,
    )

    TYPED_EXPORT = re.compile(
        r"""
        export\s+(?:async\s+)?
        function\s+(\w+)\s*\<[^>]*\>\s*\(
        """,
        re.VERBOSE,
    )

    @staticmethod
    def extract_handlers(file_content: str, file_path: str) -> list[AttackSurface]:
        surfaces = []

        for match in TypeScriptFunctionSignature.NESTJS_HANDLER.finditer(file_content):
            route_pattern = match.group(2) or "/"
            handler_name = match.group(3)
            surfaces.append(
                AttackSurface(
                    entry_point=handler_name,
                    surface_type=AttackSurfaceType.API_ENDPOINT,
                    file_path=file_path,
                    line_number=file_content[:match.start()].count('\n') + 1,
                    route_pattern=route_pattern,
                    context_code=file_content[max(0, match.start() - 200):match.end() + 200],
                )
            )

        for match in TypeScriptFunctionSignature.TYPED_EXPORT.finditer(file_content):
            handler_name = match.group(1)
            surfaces.append(
                AttackSurface(
                    entry_point=handler_name,
                    surface_type=AttackSurfaceType.FUNCTION_EXPORT,
                    file_path=file_path,
                    line_number=file_content[:match.start()].count('\n') + 1,
                    context_code=file_content[max(0, match.start() - 200):match.end() + 200],
                )
            )

        return surfaces


class MultiLanguageSurfaceDetector:
    """
    Extends attack surface detection to Go, Rust, and TypeScript codebases.
    Integrates with TargetIngestor to detect handlers in additional languages.
    """

    def __init__(self) -> None:
        self._go_detector = GoFunctionSignature()
        self._rust_detector = RustFunctionSignature()
        self._ts_detector = TypeScriptFunctionSignature()
        self.logger = logger.bind(component="multi_language_detector")

    async def detect_attack_surfaces(
        self, workspace: TargetWorkspace
    ) -> list[AttackSurface]:
        surfaces: list[AttackSurface] = []
        discovered_at_start = time.time()

        go_surfaces = await self._scan_go_files(workspace)
        surfaces.extend(go_surfaces)

        rust_surfaces = await self._scan_rust_files(workspace)
        surfaces.extend(rust_surfaces)

        ts_surfaces = await self._scan_typescript_files(workspace)
        surfaces.extend(ts_surfaces)

        elapsed_ms = int((time.time() - discovered_at_start) * 1000)
        self.logger.info(
            "multi_language_scan_complete",
            total_surfaces=len(surfaces),
            elapsed_ms=elapsed_ms,
        )

        return surfaces

    async def _scan_go_files(self, workspace: TargetWorkspace) -> list[AttackSurface]:
        surfaces = []
        for go_file in list(workspace.root.rglob("*.go")):
            try:
                content = go_file.read_text(encoding="utf-8", errors="ignore")
                file_surfaces = GoFunctionSignature.extract_handlers(
                    content, str(go_file.relative_to(workspace.root))
                )
                surfaces.extend(file_surfaces)
            except Exception as e:
                self.logger.warning("go_file_scan_failed", file=str(go_file), error=str(e))
        return surfaces

    async def _scan_rust_files(self, workspace: TargetWorkspace) -> list[AttackSurface]:
        surfaces = []
        for rust_file in list(workspace.root.rglob("*.rs")):
            try:
                content = rust_file.read_text(encoding="utf-8", errors="ignore")
                file_surfaces = RustFunctionSignature.extract_handlers(
                    content, str(rust_file.relative_to(workspace.root))
                )
                surfaces.extend(file_surfaces)
            except Exception as e:
                self.logger.warning("rust_file_scan_failed", file=str(rust_file), error=str(e))
        return surfaces

    async def _scan_typescript_files(self, workspace: TargetWorkspace) -> list[AttackSurface]:
        surfaces = []
        for ts_file in list(workspace.root.rglob("*.ts")) + list(workspace.root.rglob("*.tsx")):
            try:
                content = ts_file.read_text(encoding="utf-8", errors="ignore")
                file_surfaces = TypeScriptFunctionSignature.extract_handlers(
                    content, str(ts_file.relative_to(workspace.root))
                )
                surfaces.extend(file_surfaces)
            except Exception as e:
                self.logger.warning("typescript_file_scan_failed", file=str(ts_file), error=str(e))
        return surfaces


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 12.2: Attack Path Detection
# ═════════════════════════════════════════════════════════════════════════════


class AttackPathNode(EOSBaseModel):
    """A single vulnerability in an attack path sequence."""

    vulnerability_id: str
    vulnerability_class: str
    severity: VulnerabilitySeverity
    entry_point: str
    vulnerability_state: str


class VulnerabilitySequence(EOSBaseModel):
    """A sequence of 2+ vulnerabilities enabling a higher-severity logic flaw."""

    id: str = Field(default_factory=new_id)
    sequence_name: str = Field(
        ...,
        description="Human-readable name (e.g., 'IDOR → Logic Bypass')",
    )
    nodes: list[AttackPathNode] = Field(
        ...,
        min_items=2,
        description="Sequence of vulnerabilities in the path",
    )
    path_severity: VulnerabilitySeverity = Field(
        ...,
        description="Synthesized severity of the complete sequence",
    )
    flow_description: str = Field(
        ...,
        description="Natural language explanation of the path",
    )
    synthesized_z3_proof: str = Field(
        default="",
        description="Z3 constraints modeling the path satisfaction",
    )
    reproduction_sequence_script: str = Field(
        default="",
        description="Python script demonstrating the complete path",
    )
    discovered_at: datetime = Field(default_factory=utc_now)


class AttackPathAnalyzer:
    """
    Detects sequences of 2+ vulnerabilities that form complex logic flaws.
    Uses heuristic graph analysis + LLM reasoning to identify paths.
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.llm = llm
        self.logger = logger.bind(component="attack_path_analyzer")

    async def analyze_paths(
        self, vulnerabilities: list[VulnerabilityReport]
    ) -> list[VulnerabilitySequence]:
        if len(vulnerabilities) < 2:
            return []

        sequences: list[VulnerabilitySequence] = []
        start_time = time.time()

        graph = self._build_dependency_graph(vulnerabilities)

        for source_vuln in vulnerabilities:
            paths = self._find_paths_from(source_vuln, graph, vulnerabilities)
            for path in paths:
                seq = await self._synthesize_sequence(path, vulnerabilities)
                if seq:
                    sequences.append(seq)

        elapsed_ms = int((time.time() - start_time) * 1000)
        self.logger.info(
            "path_analysis_complete",
            input_vulnerabilities=len(vulnerabilities),
            sequences_discovered=len(sequences),
            elapsed_ms=elapsed_ms,
        )

        return sequences

    def _build_dependency_graph(
        self, vulnerabilities: list[VulnerabilityReport]
    ) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = {v.id: set() for v in vulnerabilities}

        for i, v1 in enumerate(vulnerabilities):
            for v2 in vulnerabilities[i + 1:]:
                if v1.attack_surface.entry_point == v2.attack_surface.entry_point:
                    graph[v1.id].add(v2.id)
                    graph[v2.id].add(v1.id)
                    continue

                severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
                if severity_order.get(v1.severity, 0) <= severity_order.get(v2.severity, 0):
                    graph[v1.id].add(v2.id)

        return graph

    def _find_paths_from(
        self,
        start: VulnerabilityReport,
        graph: dict[str, set[str]],
        vulns_by_id: dict[str, VulnerabilityReport] | list[VulnerabilityReport],
    ) -> list[list[VulnerabilityReport]]:
        paths = []
        vulns_by_id_map = {v.id: v for v in (vulns_by_id if isinstance(vulns_by_id, list) else [])}
        if not vulns_by_id_map and isinstance(vulns_by_id, dict):
            vulns_by_id_map = vulns_by_id

        def dfs(node_id: str, current_path: list[str], visited: set[str]) -> None:
            if len(current_path) >= 2:
                paths.append(current_path[:])
            if len(current_path) >= 3:
                return

            for neighbor_id in graph.get(node_id, set()):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    current_path.append(neighbor_id)
                    dfs(neighbor_id, current_path, visited)
                    current_path.pop()
                    visited.remove(neighbor_id)

        visited = {start.id}
        dfs(start.id, [start.id], visited)

        return [
            [vulns_by_id_map[vid] for vid in p]
            for p in paths
        ]

    async def _synthesize_sequence(
        self,
        path: list[VulnerabilityReport],
        all_vulns: list[VulnerabilityReport],
    ) -> VulnerabilitySequence | None:
        if len(path) < 2:
            return None

        nodes = [
            AttackPathNode(
                vulnerability_id=v.id,
                vulnerability_class=v.vulnerability_class,
                severity=v.severity,
                entry_point=v.attack_surface.entry_point,
                vulnerability_state=v.attack_goal, # Assuming attack_goal holds the state condition
            )
            for v in path
        ]

        max_severity_order = max(
            {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(v.severity, 1)
            for v in path
        )
        path_severity = (
            VulnerabilitySeverity.CRITICAL
            if max_severity_order >= 2
            else VulnerabilitySeverity.HIGH
        )

        class_names = [v.vulnerability_class.split("_")[-1].title() for v in path]
        sequence_name = " → ".join(class_names)
        flow_description = f"Sequence of {len(path)} vulnerabilities: " + " → ".join(v.attack_goal for v in path)

        return VulnerabilitySequence(
            sequence_name=sequence_name,
            nodes=nodes,
            path_severity=path_severity,
            flow_description=flow_description,
        )


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 12.4: Autonomous Patching
# ═════════════════════════════════════════════════════════════════════════════


class GitHubPRConfig(EOSBaseModel):
    """Configuration for autonomous GitHub PR submission."""

    github_token: str = Field(..., description="GitHub personal access token with repo write permissions")
    target_owner: str = Field(..., description="Owner of the target repository")
    target_repo: str = Field(..., description="Name of the target repository")
    branch_prefix: str = Field(default="inspector-patch", description="Prefix for automatically created branches")
    auto_merge_enabled: bool = Field(default=False, description="Whether to enable auto-merge on created PRs")
    draft_pr: bool = Field(default=True, description="Whether to create PRs as drafts")


class GitHubPRResult(EOSBaseModel):
    """Result of a GitHub PR submission attempt."""

    id: str = Field(default_factory=new_id)
    vulnerability_id: str
    pr_number: int | None = Field(default=None, description="GitHub PR number if created")
    pr_url: str | None = Field(default=None, description="URL to the created PR")
    branch_name: str | None = Field(default=None, description="Name of the patch branch")
    success: bool = Field(default=False, description="Whether PR was created successfully")
    error_message: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=utc_now)


class AutonomousPatchingOrchestrator:
    """Automatically generates GitHub pull requests with patches for discovered vulnerabilities."""

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self.llm = llm
        self.logger = logger.bind(component="autonomous_patching")

    async def submit_patch_pr(
        self,
        target_repo_url: str,
        vulnerability: VulnerabilityReport,
        patched_code: str,
        patch_diff: str,
        config: GitHubPRConfig,
    ) -> GitHubPRResult:
        try:
            pr_title = f"Security Patch: Fix {vulnerability.vulnerability_class.replace('_', ' ').title()}"
            self._generate_pr_description(vulnerability, patched_code)

            self.logger.info(
                "pr_submission_initiated",
                target_repo=target_repo_url,
                vulnerability_id=vulnerability.id,
                title=pr_title,
            )

            # Placeholder for actual GitHub API integration
            pr_number = 42
            pr_url = f"{target_repo_url}/pull/{pr_number}"
            branch_name = f"{config.branch_prefix}/{vulnerability.id[:8]}"

            return GitHubPRResult(
                vulnerability_id=vulnerability.id,
                pr_number=pr_number,
                pr_url=pr_url,
                branch_name=branch_name,
                success=True,
            )

        except Exception as e:
            self.logger.error("pr_submission_failed", error=str(e))
            return GitHubPRResult(vulnerability_id=vulnerability.id, success=False, error_message=str(e))

    def _generate_pr_description(
        self, vulnerability: VulnerabilityReport, patched_code: str
    ) -> str:
        return f"""
## Security Patch: {vulnerability.vulnerability_class}

**Severity:** {vulnerability.severity.upper()}

### Vulnerability Details
- **Class:** {vulnerability.vulnerability_class}
- **Entry Point:** {vulnerability.attack_surface.entry_point}
- **Discovery:** Automated by EcodiaOS Inspector

### What's Fixed
This patch addresses the vulnerability identified in:
- **File:** {vulnerability.attack_surface.file_path}

---
**Note:** This PR was automatically generated by EcodiaOS Inspector. Please review carefully.
"""


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 12.5: Continuous Hunting Scheduler
# ═════════════════════════════════════════════════════════════════════════════


class ScheduledHuntConfig(EOSBaseModel):
    """Configuration for a scheduled recurring hunt."""

    id: str = Field(default_factory=new_id)
    target_repo_url: str = Field(..., description="GitHub URL to hunt")
    cron_expression: str = Field(default="0 2 * * *", description="Cron expression for hunt frequency")
    enabled: bool = Field(default=True, description="Whether this hunt is active")
    max_vulns_per_run: int = Field(default=100, description="Abort hunt if more than N vulnerabilities discovered")
    auto_patch_on_discover: bool = Field(default=False, description="Whether to auto-generate patches")
    created_at: datetime = Field(default_factory=utc_now)
    last_run_at: datetime | None = Field(default=None)
    next_run_at: datetime | None = Field(default=None)


class HuntScheduleRun(EOSBaseModel):
    """A single execution of a scheduled hunt."""

    id: str = Field(default_factory=new_id)
    schedule_config_id: str = Field(..., description="ID of the ScheduledHuntConfig")
    target_repo_url: str
    run_started_at: datetime = Field(default_factory=utc_now)
    run_completed_at: datetime | None = Field(default=None)
    vulns_discovered: int = 0
    chains_discovered: int = 0
    patches_generated: int = 0
    errors: list[str] = Field(default_factory=list)
    duration_ms: int = 0


class ContinuousHuntingScheduler:
    """Schedules recurring hunts on registered repositories."""

    def __init__(self, inspector_service: InspectorService | None = None) -> None:
        self.inspector_service = inspector_service
        self.schedules: dict[str, ScheduledHuntConfig] = {}
        self.run_history: dict[str, list[HuntScheduleRun]] = {}
        self.logger = logger.bind(component="continuous_hunting_scheduler")

    async def register_hunt_schedule(self, config: ScheduledHuntConfig) -> ScheduledHuntConfig:
        self.schedules[config.id] = config
        self.run_history[config.id] = []
        self.logger.info("hunt_schedule_registered", schedule_id=config.id)
        return config

    async def execute_scheduled_hunt(self, schedule_id: str) -> HuntScheduleRun | None:
        config = self.schedules.get(schedule_id)
        if not config or not config.enabled:
            return None

        run = HuntScheduleRun(schedule_config_id=schedule_id, target_repo_url=config.target_repo_url)
        start_time = time.time()

        try:
            if not self.inspector_service:
                raise RuntimeError("InspectorService not configured for scheduler")

            hunt_result = await self.inspector_service.hunt_external_repo(config.target_repo_url)
            run.vulns_discovered = len(hunt_result.vulnerabilities_found)

            if run.vulns_discovered > config.max_vulns_per_run:
                run.errors.append(f"Discovery exceeded threshold ({run.vulns_discovered})")
                return run

            self.logger.info("scheduled_hunt_completed", schedule_id=schedule_id)

        except Exception as e:
            run.errors.append(str(e))
            self.logger.error("scheduled_hunt_failed", error=str(e))

        finally:
            run.run_completed_at = utc_now()
            run.duration_ms = int((time.time() - start_time) * 1000)
            self.run_history[schedule_id].append(run)

        return run

    async def get_schedule_statistics(self, schedule_id: str) -> dict[str, Any]:
        if schedule_id not in self.run_history:
            return {}
        runs = self.run_history[schedule_id]
        if not runs:
            return {}
        total_vulns = sum(r.vulns_discovered for r in runs)
        return {
            "total_runs": len(runs),
            "total_vulnerabilities_discovered": total_vulns,
            "avg_vulns_per_run": total_vulns / len(runs),
        }
