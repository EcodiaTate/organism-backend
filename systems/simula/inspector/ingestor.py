"""
EcodiaOS - Inspector Target Ingestor (Phase 3)

Clones external repositories, builds dependency graphs via AST parsing,
and discovers exploitable attack surfaces across Python, JavaScript/
TypeScript, and Solidity codebases.

Attack surface detection is high-confidence only - regex patterns for
non-Python languages will be refined in future phases.
"""

from __future__ import annotations

import ast
import re
import time
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TYPE_CHECKING

import structlog

from systems.simula.inspector.types import (
    AttackSurface,
    AttackSurfaceType,
)
from systems.simula.inspector.workspace import TargetWorkspace

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from systems.simula.inspector.scope import ScopeEnforcer
    from systems.simula.inspector.tracer import CallGraphTracer

logger = structlog.get_logger().bind(system="simula.inspector.ingestor")


# ── Language file extensions ─────────────────────────────────────────────────

_PYTHON_EXTS = {".py"}
_JS_TS_EXTS = {".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"}
_SOLIDITY_EXTS = {".sol"}

# Max file size to parse (skip huge generated files).
_MAX_FILE_BYTES = 512_000  # 512 KB


# ── Regex patterns for non-Python surface detection ─────────────────────────

# JavaScript / TypeScript: route registrations and exported functions.
_JS_ROUTE_PATTERN = re.compile(
    r"""
    (?:app|router|server)               # Express/Koa/Fastify receiver
    \s*\.\s*
    (get|post|put|patch|delete|all)      # HTTP method
    \s*\(\s*
    (['"`])([^'"`]+)\2                   # route path (quoted)
    """,
    re.VERBOSE | re.IGNORECASE,
)

_JS_EXPORT_FUNCTION_PATTERN = re.compile(
    r"""
    export\s+(?:async\s+)?function\s+   # export function / export async function
    (\w+)                                # function name
    """,
    re.VERBOSE,
)

_JS_DEFAULT_EXPORT_HANDLER = re.compile(
    r"""
    export\s+default\s+(?:async\s+)?function\s+  # export default function
    (\w+)?                                         # optional function name
    """,
    re.VERBOSE,
)

# Solidity: public/external functions and state variables.
_SOL_PUBLIC_FUNCTION = re.compile(
    r"""
    function\s+(\w+)\s*\(               # function name(
    [^)]*\)\s+                           # params)
    (?:external|public)                  # visibility
    """,
    re.VERBOSE,
)

_SOL_PUBLIC_STATE_VAR = re.compile(
    r"""
    ^\s*
    (?:uint\d*|int\d*|address|bool|string|bytes\d*|mapping\s*\([^)]*\))\s+  # type
    public\s+                             # visibility
    (\w+)                                 # variable name
    """,
    re.VERBOSE | re.MULTILINE,
)


# ── Python AST route/handler detection ──────────────────────────────────────

# Decorator patterns that indicate HTTP route registration.
_PYTHON_ROUTE_DECORATORS: dict[str, AttackSurfaceType] = {
    "route": AttackSurfaceType.API_ENDPOINT,
    "get": AttackSurfaceType.API_ENDPOINT,
    "post": AttackSurfaceType.API_ENDPOINT,
    "put": AttackSurfaceType.API_ENDPOINT,
    "patch": AttackSurfaceType.API_ENDPOINT,
    "delete": AttackSurfaceType.API_ENDPOINT,
    "head": AttackSurfaceType.API_ENDPOINT,
    "options": AttackSurfaceType.API_ENDPOINT,
    "websocket": AttackSurfaceType.WEBSOCKET_HANDLER,
    "ws": AttackSurfaceType.WEBSOCKET_HANDLER,
}

# Function name patterns for non-decorator-based detection.
_HandlerPattern = tuple[re.Pattern[str], AttackSurfaceType]
_PYTHON_HANDLER_NAME_PATTERNS: list[_HandlerPattern] = [
    (
        re.compile(r"middleware|dispatch|process_request|process_response"),
        AttackSurfaceType.MIDDLEWARE,
    ),
    (
        re.compile(r"handle_upload|upload_file|file_upload"),
        AttackSurfaceType.FILE_UPLOAD,
    ),
    (
        re.compile(r"authenticate|login|logout|verify_token|check_auth"),
        AttackSurfaceType.AUTH_HANDLER,
    ),
    (
        re.compile(r"on_message|on_connect|handle_event|event_handler"),
        AttackSurfaceType.EVENT_HANDLER,
    ),
    (
        re.compile(r"resolve_|query_|mutation_"),
        AttackSurfaceType.GRAPHQL_RESOLVER,
    ),
    (
        re.compile(r"deserialize|loads|from_json|from_yaml|from_pickle"),
        AttackSurfaceType.DESERIALIZATION,
    ),
    (
        re.compile(r"execute_query|run_query|raw_query|cursor\.execute"),
        AttackSurfaceType.DATABASE_QUERY,
    ),
    (
        re.compile(r"cli_|command_|cmd_"),
        AttackSurfaceType.CLI_COMMAND,
    ),
]


class TargetIngestor:
    """
    Ingests a target codebase: clones (if remote), builds a dependency graph,
    discovers exploitable attack surfaces, and extracts context code for Z3
    encoding.
    """

    def __init__(
        self,
        workspace: TargetWorkspace,
        llm: LLMProvider | None = None,
        scope_enforcer: ScopeEnforcer | None = None,
        tracer: CallGraphTracer | None = None,
    ) -> None:
        self._workspace = workspace
        self._llm = llm
        self._scope_enforcer = scope_enforcer
        self._tracer = tracer
        self._log = logger.bind(
            workspace_root=str(workspace.root),
            workspace_type=workspace.workspace_type,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    @classmethod
    async def ingest_from_github(
        cls,
        github_url: str,
        llm: LLMProvider | None = None,
        *,
        clone_depth: int = 1,
    ) -> TargetIngestor:
        """
        Clone a GitHub repository and return an ingestor ready for analysis.

        Args:
            github_url: HTTPS URL of the repository.
            llm: Optional LLM provider for advanced surface detection.
            clone_depth: Git clone depth (1 = shallow clone for speed).

        Returns:
            A TargetIngestor with the cloned workspace.
        """
        workspace = await TargetWorkspace.from_github_url(
            github_url, clone_depth=clone_depth,
        )
        return cls(workspace=workspace, llm=llm)

    @property
    def workspace(self) -> TargetWorkspace:
        return self._workspace

    async def build_codebase_graph(self) -> dict[str, list[str]]:
        """
        Build a dependency graph from the codebase using AST-based import
        analysis (Python files).

        Returns:
            Dict mapping module path → list of imported module paths.
            Non-Python files are excluded.
        """
        start = time.monotonic()
        graph: dict[str, list[str]] = {}
        root = self._workspace.root

        py_files = list(root.rglob("*.py"))

        for py_file in py_files:
            if py_file.stat().st_size > _MAX_FILE_BYTES:
                continue

            rel_path = str(py_file.relative_to(root))
            imports = self._extract_python_imports(py_file)
            if imports:
                graph[rel_path] = imports

        elapsed_ms = int((time.monotonic() - start) * 1000)
        self._log.info(
            "codebase_graph_built",
            total_files=len(py_files),
            modules_with_imports=len(graph),
            total_edges=sum(len(v) for v in graph.values()),
            duration_ms=elapsed_ms,
        )
        return graph

    async def map_attack_surfaces(self) -> list[AttackSurface]:
        """
        Scan the workspace for exploitable entry points.

        Detects:
        - Python: route decorators, handler functions, middleware, etc. (AST)
        - JavaScript/TypeScript: route registrations, exported functions (regex)
        - Solidity: public/external functions and state variables (regex)

        Returns:
            List of AttackSurface objects with high-confidence entry points.
        """
        start = time.monotonic()
        surfaces: list[AttackSurface] = []
        root = self._workspace.root

        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.stat().st_size > _MAX_FILE_BYTES:
                continue

            # Skip common non-source directories.
            rel = str(file_path.relative_to(root))
            if _should_skip_path(rel):
                continue

            # Enforce Bug Bounty scope rules (if a ScopeEnforcer is configured).
            if self._scope_enforcer and not self._scope_enforcer.is_path_allowed(rel):
                self._log.debug(
                    "file_skipped_out_of_scope",
                    file_path=rel,
                )
                continue

            suffix = file_path.suffix.lower()

            if suffix in _PYTHON_EXTS:
                surfaces.extend(self._scan_python_file(file_path, rel))
            elif suffix in _JS_TS_EXTS:
                surfaces.extend(self._scan_js_ts_file(file_path, rel))
            elif suffix in _SOLIDITY_EXTS:
                surfaces.extend(self._scan_solidity_file(file_path, rel))

        # Inter-procedural enrichment: replace each surface's context_code with
        # a multi-file stitched execution context when a tracer is configured.
        if self._tracer is not None:
            enriched_count = 0
            for surface in surfaces:
                try:
                    rich_context = await self._tracer.build_execution_context(
                        workspace=self._workspace,
                        source_file=surface.file_path,
                        entry_code=surface.context_code,
                    )
                    # Only replace if the tracer produced something richer.
                    if len(rich_context) > len(surface.context_code):
                        surface.context_code = rich_context
                        enriched_count += 1
                except Exception as exc:  # noqa: BLE001
                    self._log.warning(
                        "tracer_enrichment_failed",
                        entry_point=surface.entry_point,
                        file_path=surface.file_path,
                        error=str(exc),
                    )
            self._log.debug(
                "tracer_enrichment_complete",
                enriched=enriched_count,
                total=len(surfaces),
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        self._log.info(
            "attack_surfaces_mapped",
            total_surfaces=len(surfaces),
            duration_ms=elapsed_ms,
        )

        for surface in surfaces:
            self._log.info(
                "attack_surface_discovered",
                entry_point=surface.entry_point,
                surface_type=surface.surface_type.value,
                file_path=surface.file_path,
                line_number=surface.line_number,
                http_method=surface.http_method,
                route_pattern=surface.route_pattern,
            )

        return surfaces

    async def extract_context_code(self, surface: AttackSurface) -> str:
        """
        Extract the surrounding function/class definition for a discovered
        attack surface. This context code is passed to the Z3 encoder.

        Args:
            surface: The attack surface to extract context for.

        Returns:
            The source code of the enclosing function/class, or an empty
            string if extraction fails.
        """
        target_file = self._workspace.root / surface.file_path
        if not target_file.exists():
            return ""

        try:
            source = target_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

        suffix = target_file.suffix.lower()

        if suffix in _PYTHON_EXTS:
            return self._extract_python_context(source, surface)
        else:
            # For non-Python, extract a window of lines around the entry point.
            return self._extract_line_window(source, surface.line_number)

    # ── Python scanning (AST-based) ────────────────────────────────────────

    def _extract_python_imports(self, file_path: Path) -> list[str]:
        """Parse a Python file and return its import list."""
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError, OSError):
            return []

        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)

        return imports

    def _scan_python_file(
        self,
        file_path: Path,
        rel_path: str,
    ) -> list[AttackSurface]:
        """
        AST-walk a Python file to discover attack surfaces:
        - Decorated route handlers (@app.route, @router.get, etc.)
        - Functions matching known handler name patterns
        """
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(file_path))
        except (SyntaxError, UnicodeDecodeError, OSError):
            return []

        lines = source.splitlines()
        surfaces: list[AttackSurface] = []
        seen_names: set[str] = set()

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            func_name = node.name
            if func_name.startswith("_") and not func_name.startswith("__"):
                # Skip private helpers (single underscore).
                continue

            # Check decorators for route patterns.
            surface = self._check_python_decorators(
                node, func_name, rel_path, lines,
            )
            if surface is not None and func_name not in seen_names:
                seen_names.add(func_name)
                surfaces.append(surface)
                continue

            # Check function name against known handler patterns.
            for pattern, stype in _PYTHON_HANDLER_NAME_PATTERNS:
                if pattern.search(func_name) and func_name not in seen_names:
                    seen_names.add(func_name)
                    func_start = node.lineno - 1
                    func_end = node.end_lineno or func_start + 1
                    context = "\n".join(lines[func_start:func_end])

                    surfaces.append(AttackSurface(
                        entry_point=func_name,
                        surface_type=stype,
                        file_path=rel_path,
                        line_number=node.lineno,
                        context_code=context[:4000],
                    ))
                    break

        return surfaces

    def _check_python_decorators(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        func_name: str,
        rel_path: str,
        lines: list[str],
    ) -> AttackSurface | None:
        """
        Check if a function's decorators indicate an HTTP route or handler.

        Detects:
        - @app.get("/path"), @router.post("/path"), etc.
        - @app.route("/path", methods=["GET"])
        """
        for decorator in node.decorator_list:
            dec_name: str | None = None
            http_method: str | None = None
            route_pattern: str | None = None
            surface_type: AttackSurfaceType | None = None

            # Handle @app.get(...) style - ast.Call with ast.Attribute func.
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                attr_name = decorator.func.attr.lower()
                if attr_name in _PYTHON_ROUTE_DECORATORS:
                    dec_name = attr_name
                    surface_type = _PYTHON_ROUTE_DECORATORS[attr_name]

                    # Extract route path from first positional arg.
                    if decorator.args:
                        route_pattern = _ast_const_value(decorator.args[0])

                    # Extract HTTP method from decorator name or methods kwarg.
                    if attr_name in ("get", "post", "put", "patch", "delete", "head", "options"):
                        http_method = attr_name.upper()
                    else:
                        # Check methods= keyword arg for @app.route(... methods=[...]).
                        for kw in decorator.keywords:
                            if kw.arg == "methods" and isinstance(kw.value, ast.List):
                                methods = [
                                    _ast_const_value(elt)
                                    for elt in kw.value.elts
                                    if _ast_const_value(elt)
                                ]
                                http_method = ",".join(m.upper() for m in methods if m)

            # Handle @app.get style without call (bare attribute).
            elif isinstance(decorator, ast.Attribute):
                attr_name = decorator.attr.lower()
                if attr_name in _PYTHON_ROUTE_DECORATORS:
                    dec_name = attr_name
                    surface_type = _PYTHON_ROUTE_DECORATORS[attr_name]
                    if attr_name in ("get", "post", "put", "patch", "delete", "head", "options"):
                        http_method = attr_name.upper()

            if dec_name is not None and surface_type is not None:
                func_start = node.lineno - 1
                func_end = node.end_lineno or func_start + 1
                context = "\n".join(lines[func_start:func_end])

                return AttackSurface(
                    entry_point=func_name,
                    surface_type=surface_type,
                    file_path=rel_path,
                    line_number=node.lineno,
                    context_code=context[:4000],
                    http_method=http_method,
                    route_pattern=route_pattern,
                )

        return None

    # ── JavaScript / TypeScript scanning (regex-based) ──────────────────────

    def _scan_js_ts_file(
        self,
        file_path: Path,
        rel_path: str,
    ) -> list[AttackSurface]:
        """
        Regex-based scanning for JS/TS attack surfaces:
        - Express/Koa/Fastify route registrations (app.get, router.post, etc.)
        - Exported functions (potential API handlers)
        """
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        surfaces: list[AttackSurface] = []

        # Detect route registrations: app.get('/path', handler)
        for match in _JS_ROUTE_PATTERN.finditer(source):
            http_method = match.group(1).upper()
            route_path = match.group(3)
            line_number = source[:match.start()].count("\n") + 1

            surfaces.append(AttackSurface(
                entry_point=f"{http_method} {route_path}",
                surface_type=AttackSurfaceType.API_ENDPOINT,
                file_path=rel_path,
                line_number=line_number,
                context_code=self._extract_line_window(source, line_number),
                http_method=http_method,
                route_pattern=route_path,
            ))

        # Detect exported functions.
        for match in _JS_EXPORT_FUNCTION_PATTERN.finditer(source):
            func_name = match.group(1)
            line_number = source[:match.start()].count("\n") + 1

            surfaces.append(AttackSurface(
                entry_point=func_name,
                surface_type=AttackSurfaceType.FUNCTION_EXPORT,
                file_path=rel_path,
                line_number=line_number,
                context_code=self._extract_line_window(source, line_number),
            ))

        # Detect default export handlers.
        for match in _JS_DEFAULT_EXPORT_HANDLER.finditer(source):
            func_name = match.group(1) or "default"
            line_number = source[:match.start()].count("\n") + 1

            surfaces.append(AttackSurface(
                entry_point=f"export_default_{func_name}",
                surface_type=AttackSurfaceType.FUNCTION_EXPORT,
                file_path=rel_path,
                line_number=line_number,
                context_code=self._extract_line_window(source, line_number),
            ))

        return surfaces

    # ── Solidity scanning (regex-based) ─────────────────────────────────────

    def _scan_solidity_file(
        self,
        file_path: Path,
        rel_path: str,
    ) -> list[AttackSurface]:
        """
        Regex-based scanning for Solidity attack surfaces:
        - public/external functions
        - public state variables (auto-generated getters)
        """
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        surfaces: list[AttackSurface] = []

        # Detect public/external functions.
        for match in _SOL_PUBLIC_FUNCTION.finditer(source):
            func_name = match.group(1)
            line_number = source[:match.start()].count("\n") + 1

            surfaces.append(AttackSurface(
                entry_point=func_name,
                surface_type=AttackSurfaceType.SMART_CONTRACT_PUBLIC,
                file_path=rel_path,
                line_number=line_number,
                context_code=self._extract_line_window(source, line_number),
            ))

        # Detect public state variables.
        for match in _SOL_PUBLIC_STATE_VAR.finditer(source):
            var_name = match.group(1)
            line_number = source[:match.start()].count("\n") + 1

            surfaces.append(AttackSurface(
                entry_point=var_name,
                surface_type=AttackSurfaceType.SMART_CONTRACT_PUBLIC,
                file_path=rel_path,
                line_number=line_number,
                context_code=self._extract_line_window(source, line_number),
            ))

        return surfaces

    # ── Context extraction helpers ──────────────────────────────────────────

    def _extract_python_context(
        self,
        source: str,
        surface: AttackSurface,
    ) -> str:
        """
        AST-based context extraction for Python: returns the full function or
        class body surrounding the entry point.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self._extract_line_window(source, surface.line_number)

        lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            if node.name == surface.entry_point and node.lineno == surface.line_number:
                func_start = node.lineno - 1
                func_end = node.end_lineno or func_start + 1
                context = "\n".join(lines[func_start:func_end])
                return context[:4000]

        # Fallback: try matching by name alone (line might not match exactly).
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name == surface.entry_point:
                func_start = node.lineno - 1
                func_end = node.end_lineno or func_start + 1
                context = "\n".join(lines[func_start:func_end])
                return context[:4000]

        return self._extract_line_window(source, surface.line_number)

    def _extract_line_window(
        self,
        source: str,
        line_number: int | None,
        window: int = 30,
    ) -> str:
        """
        Extract a window of lines around a line number.
        Falls back to the first `window` lines if line_number is None.
        """
        lines = source.splitlines()
        if line_number is None or line_number < 1:
            return "\n".join(lines[:window])

        start = max(0, line_number - 1 - (window // 2))
        end = min(len(lines), start + window)
        return "\n".join(lines[start:end])


# ── Module-level helpers ────────────────────────────────────────────────────


def _ast_const_value(node: ast.expr) -> str | None:
    """Extract a string constant value from an AST node."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _should_skip_path(rel_path: str) -> bool:
    """Skip paths that are unlikely to contain exploitable source code."""
    skip_prefixes = (
        "node_modules/",
        ".git/",
        "__pycache__/",
        ".venv/",
        "venv/",
        ".env/",
        "env/",
        ".tox/",
        ".mypy_cache/",
        ".pytest_cache/",
        "dist/",
        "build/",
        ".next/",
        "coverage/",
        ".idea/",
        ".vscode/",
    )
    skip_names = (
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
    )
    for prefix in skip_prefixes:
        if rel_path.startswith(prefix) or f"/{prefix}" in rel_path:
            return True
    parts = rel_path.replace("\\", "/").split("/")
    return bool(parts and parts[-1] in skip_names)
