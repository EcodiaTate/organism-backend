"""
EcodiaOS - Inspector Cross-Service Call Graph Tracer

Extends the single-service CallGraphTracer concept across docker-compose
service boundaries. Given a compose topology, this tracer:

  1. Maps compose service names to their source directories (via build.context
     or volume mounts in the compose file).
  2. Detects inter-service HTTP calls in source code (requests, fetch, axios,
     httpx, etc.) and resolves URL fragments to compose service names.
  3. Stitches multi-service source into a single labelled context string,
     ready for the cross-service VulnerabilityProver.

Output format:
  // === Service: api (Entry Point) ===
  // --- File: api/routes/users.py (Handler) ---
  <code>

  // === Service: user-service (Cross-Service Callee) ===
  // --- Flow: api -> user-service via HTTP POST /api/users ---
  // --- File: user-service/handlers/create_user.py (Handler) ---
  <code>
"""

from __future__ import annotations

import re
from typing import Any, Final
from pathlib import Path

import structlog

logger: Final = structlog.get_logger().bind(system="simula.inspector.cross_service_tracer")

# Hard cap on multi-service stitched context.
_DEFAULT_MAX_CHARS: Final[int] = 24_000

# Per-service code section cap.
_MAX_SERVICE_CHARS: Final[int] = 6_000

# ── Inter-service call detection patterns ───────────────────────────────────

# Python: requests.get/post/put/delete/patch, httpx.get/post/..., urllib
_PY_HTTP_CALL: Final = re.compile(
    r"""
    (?:requests|httpx|aiohttp|urllib\.request)
    \s*\.
    (?:get|post|put|patch|delete|head|options|request)
    \s*\(
    \s*[f"'`]([^"'`\n]+)[f"'`]
    """,
    re.VERBOSE | re.IGNORECASE,
)

# JS/TS: fetch("url"), axios.get("url"), axios.post("url")
_JS_HTTP_CALL: Final = re.compile(
    r"""
    (?:fetch|axios\.(?:get|post|put|patch|delete|head|options|request))
    \s*\(
    \s*[`"']([^`"'\n]+)[`"']
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Generic: any URL-like string containing a compose service name pattern
# e.g., "http://user-service:3000/api/users"
_URL_FRAGMENT: Final = re.compile(
    r"""https?://([a-zA-Z0-9_-]+)(?::\d+)?(/[^\s"'`]*)?""",
    re.IGNORECASE,
)

# Common entry-point file patterns (per language)
_ENTRY_FILE_PATTERNS: Final[list[str]] = [
    "**/routes/**/*.py",
    "**/handlers/**/*.py",
    "**/views/**/*.py",
    "**/controllers/**/*.py",
    "**/api/**/*.py",
    "**/routes/**/*.ts",
    "**/routes/**/*.js",
    "**/handlers/**/*.ts",
    "**/handlers/**/*.js",
    "**/controllers/**/*.ts",
    "**/controllers/**/*.js",
    "**/src/**/*.go",
    "**/src/**/*.rs",
]


class CrossServiceTracer:
    """
    Traces data flows across docker-compose service boundaries by detecting
    inter-service HTTP calls in source code and stitching multi-service context.

    Usage::

        tracer = CrossServiceTracer(workspace_root, compose_data)
        context = await tracer.build_cross_service_context(
            entry_service="api",
            entry_code="def create_user(request): ...",
            source_file="routes/users.py",
        )
    """

    def __init__(
        self,
        workspace_root: Path,
        compose_data: dict[str, Any],
    ) -> None:
        self._workspace_root = workspace_root
        self._compose_data = compose_data
        self._service_dirs = self._map_services_to_source_dirs()
        self._log = logger.bind(
            workspace=str(workspace_root),
            services=list(self._service_dirs.keys()),
        )

    # ── Public API ─────────────────────────────────────────────────────────

    async def build_cross_service_context(
        self,
        entry_service: str,
        entry_code: str,
        source_file: str,
        *,
        max_depth: int = 2,
        max_chars: int = _DEFAULT_MAX_CHARS,
    ) -> str:
        """
        Build a multi-service stitched context string starting from an
        entry point in ``entry_service``.

        Args:
            entry_service: Compose service name of the entry point.
            entry_code: Source code of the entry-point function.
            source_file: Relative path of the entry-point file.
            max_depth: How many cross-service hops to follow.
            max_chars: Hard cap on total output length.

        Returns:
            Labelled multi-service context string for the prover.
        """
        self._log.debug(
            "cross_service_trace_started",
            entry_service=entry_service,
            source_file=source_file,
            max_depth=max_depth,
        )

        sections: list[str] = []
        visited_services: set[str] = {entry_service}

        # Entry point section
        sections.append(
            f"// === Service: {entry_service} (Entry Point) ===\n"
            f"// --- File: {source_file} (Handler) ---\n"
            + entry_code
        )

        # Detect cross-service calls and follow them
        self._trace_cross_service(
            service_name=entry_service,
            source_code=entry_code,
            depth=0,
            max_depth=max_depth,
            visited=visited_services,
            sections=sections,
        )

        result = "\n\n".join(sections)

        if len(result) > max_chars:
            self._log.debug(
                "cross_service_context_truncated",
                original_len=len(result),
                max_chars=max_chars,
            )
            result = result[:max_chars]

        self._log.debug(
            "cross_service_trace_complete",
            sections=len(sections),
            total_chars=len(result),
            services_visited=list(visited_services),
        )
        return result

    # ── Service-to-directory mapping ──────────────────────────────────────

    def _map_services_to_source_dirs(self) -> dict[str, Path]:
        """
        Map compose service names to their source directories on the host.

        Checks (in priority order):
          1. build.context - the Dockerfile build context directory
          2. volumes - host-mounted source directories
          3. Falls back to service name as a subdirectory of workspace root
        """
        services = self._compose_data.get("services", {})
        mapping: dict[str, Path] = {}

        for svc_name, svc_def in services.items():
            if svc_name == "simula-ebpf-tracer":
                continue

            # 1. build.context
            build = svc_def.get("build")
            if isinstance(build, dict):
                ctx = build.get("context", ".")
                candidate = (self._workspace_root / ctx).resolve()
                if candidate.is_dir():
                    mapping[svc_name] = candidate
                    continue
            elif isinstance(build, str):
                candidate = (self._workspace_root / build).resolve()
                if candidate.is_dir():
                    mapping[svc_name] = candidate
                    continue

            # 2. volumes - look for host:container mounts
            volumes = svc_def.get("volumes", [])
            for vol in volumes:
                if isinstance(vol, str) and ":" in vol:
                    host_part = vol.split(":")[0]
                    # Relative or absolute host path
                    if host_part.startswith(".") or host_part.startswith("/"):
                        candidate = (self._workspace_root / host_part).resolve()
                        if candidate.is_dir():
                            mapping[svc_name] = candidate
                            break

            # 3. Fallback: service name as subdirectory
            if svc_name not in mapping:
                candidate = (self._workspace_root / svc_name).resolve()
                if candidate.is_dir():
                    mapping[svc_name] = candidate

        return mapping

    # ── Cross-service call tracing ────────────────────────────────────────

    def _trace_cross_service(
        self,
        service_name: str,
        source_code: str,
        depth: int,
        max_depth: int,
        visited: set[str],
        sections: list[str],
    ) -> None:
        """Detect inter-service calls and recursively stitch their code."""
        if depth >= max_depth:
            return

        calls = self._detect_inter_service_calls(source_code, service_name)

        for target_service, url_fragment in calls:
            if target_service in visited:
                continue
            visited.add(target_service)

            # Find relevant handler code in the target service
            target_code = self._find_handler_code(target_service, url_fragment)
            if not target_code:
                self._log.debug(
                    "cross_service_handler_not_found",
                    from_service=service_name,
                    to_service=target_service,
                    url=url_fragment,
                )
                continue

            handler_file, handler_code = target_code

            sections.append(
                f"// === Service: {target_service} (Cross-Service Callee) ===\n"
                f"// --- Flow: {service_name} -> {target_service} via HTTP {url_fragment} ---\n"
                f"// --- File: {handler_file} (Handler) ---\n"
                + handler_code[:_MAX_SERVICE_CHARS]
            )

            self._log.debug(
                "cross_service_dependency_included",
                from_service=service_name,
                to_service=target_service,
                handler_file=handler_file,
                depth=depth + 1,
            )

            # Recurse into the callee service
            self._trace_cross_service(
                service_name=target_service,
                source_code=handler_code,
                depth=depth + 1,
                max_depth=max_depth,
                visited=visited,
                sections=sections,
            )

    def _detect_inter_service_calls(
        self,
        source_code: str,
        current_service: str,
    ) -> list[tuple[str, str]]:
        """
        Detect inter-service HTTP calls in source code.

        Returns list of (target_service_name, url_fragment) tuples.
        """
        results: list[tuple[str, str]] = []
        seen: set[str] = set()

        # Collect all URL fragments from HTTP call patterns
        url_fragments: list[str] = []
        for pattern in (_PY_HTTP_CALL, _JS_HTTP_CALL):
            for m in pattern.finditer(source_code):
                url_fragments.append(m.group(1))

        # Also scan for raw URL strings that might reference services
        for m in _URL_FRAGMENT.finditer(source_code):
            full_url = m.group(0)
            if full_url not in url_fragments:
                url_fragments.append(full_url)

        for url_frag in url_fragments:
            target = self._resolve_url_to_service(url_frag)
            if target and target != current_service and target not in seen:
                seen.add(target)
                results.append((target, url_frag))

        return results

    def _resolve_url_to_service(self, url_fragment: str) -> str | None:
        """
        Map a URL fragment to a compose service name.

        Handles patterns like:
          - http://user-service:3000/api/users → "user-service"
          - http://db:5432 → "db"
          - ${USER_SERVICE_URL}/api → try to match env var to service
        """
        # Direct hostname match against known services
        m = _URL_FRAGMENT.search(url_fragment)
        if m:
            hostname = m.group(1).lower()
            services = self._compose_data.get("services", {})
            for svc_name in services:
                if svc_name == "simula-ebpf-tracer":
                    continue
                # Exact match or hyphen/underscore normalization
                if (
                    hostname == svc_name.lower()
                    or hostname == svc_name.lower().replace("-", "_")
                    or hostname == svc_name.lower().replace("_", "-")
                ):
                    return str(svc_name)

        # Environment variable reference: ${SERVICE_URL} or $SERVICE_HOST
        env_match = re.search(r"\$\{?(\w+?)(?:_URL|_HOST|_ADDR)\}?", url_fragment)
        if env_match:
            env_prefix = env_match.group(1).lower().replace("_", "-")
            services = self._compose_data.get("services", {})
            for svc_name in services:
                if svc_name == "simula-ebpf-tracer":
                    continue
                if env_prefix in svc_name.lower():
                    return str(svc_name)

        return None

    # ── Handler code discovery ────────────────────────────────────────────

    def _find_handler_code(
        self,
        service_name: str,
        url_fragment: str,
    ) -> tuple[str, str] | None:
        """
        Find the handler code in a target service that corresponds to a URL.

        Returns (relative_file_path, handler_code) or None if not found.
        """
        service_dir = self._service_dirs.get(service_name)
        if not service_dir or not service_dir.is_dir():
            return None

        # Extract the route path from the URL fragment
        route_path = self._extract_route_path(url_fragment)

        # Scan source files for route handlers matching the path
        for pattern in _ENTRY_FILE_PATTERNS:
            for source_file in service_dir.glob(pattern):
                try:
                    content = source_file.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                # Check if this file contains a handler for the route
                if route_path and self._file_matches_route(content, route_path):
                    rel_path = str(source_file.relative_to(self._workspace_root)).replace("\\", "/")
                    return (rel_path, content[:_MAX_SERVICE_CHARS])

        # Fallback: return the first "interesting" file (routes/handlers/controllers)
        for pattern in _ENTRY_FILE_PATTERNS[:6]:  # Python + JS patterns
            for source_file in service_dir.glob(pattern):
                try:
                    content = source_file.read_text(encoding="utf-8", errors="replace")
                    if len(content) > 50:  # Skip trivial files
                        rel_path = str(source_file.relative_to(self._workspace_root)).replace("\\", "/")
                        return (rel_path, content[:_MAX_SERVICE_CHARS])
                except OSError:
                    continue

        return None

    @staticmethod
    def _extract_route_path(url_fragment: str) -> str:
        """Extract the route path from a URL fragment."""
        # Strip protocol and host
        m = _URL_FRAGMENT.search(url_fragment)
        if m and m.group(2):
            return m.group(2)

        # If it starts with /, it's already a path
        if url_fragment.startswith("/"):
            return url_fragment

        return ""

    @staticmethod
    def _file_matches_route(file_content: str, route_path: str) -> bool:
        """Check if a source file contains a handler for the given route path."""
        if not route_path:
            return False

        # Normalize route: /api/users/{id} → /api/users
        clean_route = re.sub(r"/\{[^}]+\}", "", route_path)
        clean_route = re.sub(r"/:\w+", "", clean_route)
        segments = [s for s in clean_route.split("/") if s]

        if not segments:
            return False

        # Check for route decorators/registrations containing the path
        # Python: @app.route("/api/users"), @router.post("/users")
        # JS/TS: router.get("/users"), app.post("/api/users")
        for segment in segments[-2:]:  # Last 2 segments are most specific
            if re.search(
                r"""['"/@]""" + re.escape(segment) + r"""['"/?]""",
                file_content,
            ):
                return True

        return False
