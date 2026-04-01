"""
EcodiaOS - Inspector Safety Gates (Phase 11)

Enforces the Iron Rules that govern Inspector's execution boundaries:

  1. PoC Execution Safety - No requests to unauthorized domains; no forbidden
     modules; sandbox timeout enforcement; no direct execution against live
     targets without explicit authorization.

  2. Workspace Isolation - External workspaces must be temp-only, never the
     EOS source tree; no symlinks escaping the workspace; deterministic temp
     cleanup on exit.

  3. Configuration Validation - authorized_targets must be non-empty for PoC
     execution; sandbox_timeout_seconds must be positive; max_workers bounded
     to [1, 16]; all constraints enforced before the pipeline starts.

These gates are invoked at two points:
  - Pre-hunt: validate_inspector_config() + validate_workspace_isolation()
  - Post-prove: validate_poc_execution() before any PoC is returned/stored

All validation failures are logged via structlog with event="safety_gate_*"
and never silently swallowed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from systems.simula.inspector.types import InspectorConfig
    from systems.simula.inspector.workspace import TargetWorkspace

logger = structlog.get_logger().bind(system="simula.inspector.safety")


# ── Constants ────────────────────────────────────────────────────────────────

# Modules that a PoC must never import - these allow host-level side effects
# beyond HTTP requests. Kept in sync with service._FORBIDDEN_POC_MODULES.
FORBIDDEN_POC_MODULES: frozenset[str] = frozenset({
    "subprocess", "socket", "ctypes", "pickle", "shelve",
    "marshal", "shutil", "tempfile", "multiprocessing",
})

# Additional dangerous patterns in PoC source (beyond import-level checks).
# These catch inline calls that bypass the import system (e.g., __import__,
# exec, eval, compile with code strings).
_DANGEROUS_CALL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"__import__\s*\(", re.MULTILINE),
    re.compile(r"\bexec\s*\(", re.MULTILINE),
    re.compile(r"\beval\s*\(", re.MULTILINE),
    re.compile(r"\bcompile\s*\(", re.MULTILINE),
    re.compile(r"\bos\.system\s*\(", re.MULTILINE),
    re.compile(r"\bos\.popen\s*\(", re.MULTILINE),
    re.compile(r"\bos\.exec", re.MULTILINE),
    re.compile(r"\bopen\s*\(.+,\s*['\"]w", re.MULTILINE),
]

# URL patterns for domain extraction from PoC code.
_URL_PATTERN = re.compile(
    r"""(?:https?://)([a-zA-Z0-9._\-]+(?::\d+)?)""",
    re.IGNORECASE,
)

# BPF helper functions that modify kernel/process state. Any BPF C source
# containing these is rejected by validate_ebpf_program_safety(). Read-only
# observation helpers (bpf_probe_read_*, bpf_ktime_get_ns, bpf_get_current_*)
# are safe and not listed here.
_FORBIDDEN_BPF_HELPERS: frozenset[str] = frozenset({
    "bpf_probe_write_user",
    "bpf_override_return",
    "bpf_send_signal",
    "bpf_send_signal_thread",
    "bpf_skb_store_bytes",
    "bpf_xdp_adjust_head",
    "bpf_xdp_adjust_tail",
    "bpf_msg_redirect_map",
    "bpf_msg_redirect_hash",
    "bpf_sk_redirect_map",
    "bpf_sk_redirect_hash",
    "bpf_fib_lookup",
    "bpf_tcp_send_ack",
})

# Service names that are never valid taint tracking targets.
_SYSTEM_SERVICE_BLOCKLIST: frozenset[str] = frozenset({
    "host",
    "system",
    "kernel",
    "init",
    "systemd",
    "dockerd",
    "containerd",
})


# ── Safety Gates ─────────────────────────────────────────────────────────────


class InspectorSafetyGates:
    """
    Enforces all Inspector safety constraints.

    Three validation surfaces:
      1. validate_poc_execution()       - PoC code safety
      2. validate_workspace_isolation() - workspace boundary enforcement
      3. validate_inspector_config()       - configuration sanity
    """

    def __init__(self) -> None:
        self._log = logger

    # ── 1. PoC Execution Validation ──────────────────────────────────────

    def validate_poc_execution(
        self,
        poc_code: str,
        authorized_targets: list[str],
        *,
        sandbox_timeout_seconds: int = 30,
    ) -> SafetyResult:
        """
        Validate that a proof-of-concept script is safe to store/return.

        Checks (in order):
          1. Python syntax validity (ast.parse)
          2. No imports of forbidden modules (AST walk)
          3. No dangerous inline calls (__import__, exec, eval, os.system)
          4. All hardcoded URLs target authorized domains only
          5. Sandbox timeout is positive

        Args:
            poc_code: The generated Python PoC script.
            authorized_targets: List of authorized domains/hostnames.
            sandbox_timeout_seconds: Timeout for sandboxed execution.

        Returns:
            SafetyResult with passed=True if all checks pass, else the
            first failing check's reason.
        """
        if not poc_code.strip():
            return SafetyResult(passed=True, gate="poc_execution")

        # 1. Syntax
        try:
            tree = ast.parse(poc_code)
        except SyntaxError as exc:
            self._log.warning("safety_gate_poc_syntax_error", error=str(exc))
            return SafetyResult(
                passed=False,
                gate="poc_execution",
                reason=f"PoC has invalid Python syntax: {exc}",
            )

        # 2. Forbidden imports (AST walk)
        import_violation = self._check_forbidden_imports(tree)
        if import_violation:
            self._log.warning(
                "safety_gate_poc_forbidden_import",
                module=import_violation,
            )
            return SafetyResult(
                passed=False,
                gate="poc_execution",
                reason=f"PoC imports forbidden module: {import_violation}",
            )

        # 3. Dangerous inline calls (regex scan - catches __import__, exec, eval)
        dangerous_call = self._check_dangerous_calls(poc_code)
        if dangerous_call:
            self._log.warning(
                "safety_gate_poc_dangerous_call",
                pattern=dangerous_call,
            )
            return SafetyResult(
                passed=False,
                gate="poc_execution",
                reason=f"PoC contains dangerous call pattern: {dangerous_call}",
            )

        # 4. URL domain authorization
        unauthorized = self._check_unauthorized_urls(poc_code, authorized_targets)
        if unauthorized:
            self._log.warning(
                "safety_gate_poc_unauthorized_url",
                unauthorized_domains=unauthorized,
                authorized=authorized_targets,
            )
            return SafetyResult(
                passed=False,
                gate="poc_execution",
                reason=(
                    f"PoC targets unauthorized domain(s): "
                    f"{', '.join(unauthorized)}. "
                    f"Authorized: {authorized_targets}"
                ),
            )

        # 5. Timeout sanity
        if sandbox_timeout_seconds <= 0:
            return SafetyResult(
                passed=False,
                gate="poc_execution",
                reason=f"Sandbox timeout must be positive, got {sandbox_timeout_seconds}",
            )

        self._log.debug("safety_gate_poc_passed")
        return SafetyResult(passed=True, gate="poc_execution")

    # ── 2. Workspace Isolation Validation ────────────────────────────────

    def validate_workspace_isolation(
        self,
        workspace: TargetWorkspace,
        eos_root: Path | None = None,
    ) -> SafetyResult:
        """
        Validate that a workspace is properly isolated.

        Checks:
          1. Workspace root exists and is a directory
          2. If workspace is external, it must NOT overlap with EOS source tree
          3. No symlinks within the workspace point outside its root
          4. External workspaces with temp_directory have a valid temp path
          5. Temp directory is within system temp or an expected location

        Args:
            workspace: The TargetWorkspace to validate.
            eos_root: Path to the EOS source tree (for overlap detection).

        Returns:
            SafetyResult with passed=True if workspace is properly isolated.
        """
        root = workspace.root

        # 1. Root exists and is directory
        if not root.exists():
            return SafetyResult(
                passed=False,
                gate="workspace_isolation",
                reason=f"Workspace root does not exist: {root}",
            )
        if not root.is_dir():
            return SafetyResult(
                passed=False,
                gate="workspace_isolation",
                reason=f"Workspace root is not a directory: {root}",
            )

        # 2. External workspace must not be EOS source tree
        if workspace.is_external and eos_root is not None:
            resolved_eos = eos_root.resolve()
            resolved_root = root.resolve()
            # Check both directions: workspace inside EOS, or EOS inside workspace
            if (
                _is_subpath(resolved_root, resolved_eos)
                or _is_subpath(resolved_eos, resolved_root)
            ):
                self._log.error(
                    "safety_gate_workspace_eos_overlap",
                    workspace_root=str(resolved_root),
                    eos_root=str(resolved_eos),
                )
                return SafetyResult(
                    passed=False,
                    gate="workspace_isolation",
                    reason=(
                        f"External workspace overlaps with EOS source tree. "
                        f"Workspace: {resolved_root}, EOS: {resolved_eos}"
                    ),
                )

        # 3. Check for symlinks escaping workspace boundary
        escape_path = self._check_symlink_escapes(root)
        if escape_path:
            self._log.warning(
                "safety_gate_workspace_symlink_escape",
                symlink=str(escape_path),
                workspace_root=str(root),
            )
            return SafetyResult(
                passed=False,
                gate="workspace_isolation",
                reason=(
                    f"Symlink escapes workspace boundary: {escape_path} "
                    f"resolves outside {root}"
                ),
            )

        # 4. External workspace temp_directory validation
        if workspace.is_external and workspace.temp_directory is not None:
            temp = workspace.temp_directory
            if not temp.exists():
                return SafetyResult(
                    passed=False,
                    gate="workspace_isolation",
                    reason=f"Temp directory does not exist: {temp}",
                )

            # 5. Temp directory must be inside the OS temp root and must follow
            #    the inspector_workspace_<token> naming convention established by
            #    TargetWorkspace.from_github_url(). This prevents an attacker
            #    from constructing a TargetWorkspace whose temp_directory points
            #    to an arbitrary path (e.g. the EOS source tree) and then
            #    triggering cleanup() to nuke it.
            import tempfile as _tempfile  # local import - only needed here
            os_tmp = Path(_tempfile.gettempdir()).resolve()
            resolved_temp = temp.resolve()
            if not _is_subpath(resolved_temp, os_tmp):
                self._log.error(
                    "safety_gate_workspace_temp_outside_os_tmp",
                    temp_directory=str(resolved_temp),
                    os_tmp=str(os_tmp),
                )
                return SafetyResult(
                    passed=False,
                    gate="workspace_isolation",
                    reason=(
                        f"Temp directory '{resolved_temp}' is outside the OS temp "
                        f"root '{os_tmp}'. Refusing to allow cleanup of arbitrary paths."
                    ),
                )
            if not temp.name.startswith("inspector_workspace_"):
                self._log.warning(
                    "safety_gate_workspace_temp_unexpected_name",
                    temp_name=temp.name,
                )
                # Warn but don't fail - local_path workspaces have no temp_directory,
                # so this can only fire if someone constructed a workspace manually.

        self._log.debug(
            "safety_gate_workspace_passed",
            workspace_root=str(root),
            workspace_type=workspace.workspace_type,
        )
        return SafetyResult(passed=True, gate="workspace_isolation")

    # ── 3. Config Validation ─────────────────────────────────────────────

    def validate_inspector_config(
        self,
        config: InspectorConfig,
        *,
        require_authorized_targets: bool = False,
    ) -> SafetyResult:
        """
        Validate InspectorConfig safety constraints.

        Checks:
          1. authorized_targets is non-empty if require_authorized_targets=True
             (required when PoC execution is enabled)
          2. All authorized targets are non-empty, stripped strings
          3. sandbox_timeout_seconds > 0
          4. max_workers in [1, 16]
          5. clone_depth >= 1

        Args:
            config: The InspectorConfig to validate.
            require_authorized_targets: If True, authorized_targets must be
                non-empty (enforced when generate_pocs=True).

        Returns:
            SafetyResult with passed=True if configuration is valid.
        """
        # 1. Authorized targets required for PoC execution
        if require_authorized_targets and not config.authorized_targets:
            return SafetyResult(
                passed=False,
                gate="inspector_config",
                reason=(
                    "authorized_targets must be non-empty when PoC execution "
                    "is enabled. Provide at least one authorized target domain."
                ),
            )

        # 2. No empty/whitespace-only targets
        for i, target in enumerate(config.authorized_targets):
            if not target.strip():
                return SafetyResult(
                    passed=False,
                    gate="inspector_config",
                    reason=f"authorized_targets[{i}] is empty or whitespace-only",
                )

        # 3. Sandbox timeout
        if config.sandbox_timeout_seconds <= 0:
            return SafetyResult(
                passed=False,
                gate="inspector_config",
                reason=(
                    f"sandbox_timeout_seconds must be positive, "
                    f"got {config.sandbox_timeout_seconds}"
                ),
            )

        # 4. Worker bounds
        if not (1 <= config.max_workers <= 16):
            return SafetyResult(
                passed=False,
                gate="inspector_config",
                reason=(
                    f"max_workers must be in [1, 16], got {config.max_workers}"
                ),
            )

        # 5. Clone depth
        if config.clone_depth < 1:
            return SafetyResult(
                passed=False,
                gate="inspector_config",
                reason=f"clone_depth must be >= 1, got {config.clone_depth}",
            )

        self._log.debug(
            "safety_gate_config_passed",
            authorized_targets=len(config.authorized_targets),
            max_workers=config.max_workers,
            sandbox_timeout=config.sandbox_timeout_seconds,
        )
        return SafetyResult(passed=True, gate="inspector_config")

    # ── 4. eBPF Program Safety ─────────────────────────────────────────

    def validate_ebpf_program_safety(self, bpf_source: str) -> SafetyResult:
        """
        Validate that a BPF C program is read-only and safe to load.

        Rejects programs containing kernel-state-modifying helper calls.
        All Inspector eBPF programs must be observation-only - they attach to
        tracepoints/kprobes and emit events via ring buffer. Any program
        that writes to process memory, overrides returns, or manipulates
        packets is rejected.

        Args:
            bpf_source: The BPF C source code string.

        Returns:
            SafetyResult with passed=True if the program is safe.
        """
        if not bpf_source.strip():
            return SafetyResult(passed=True, gate="ebpf_program_safety")

        for helper in _FORBIDDEN_BPF_HELPERS:
            if helper in bpf_source:
                self._log.warning(
                    "safety_gate_ebpf_forbidden_helper",
                    helper=helper,
                )
                return SafetyResult(
                    passed=False,
                    gate="ebpf_program_safety",
                    reason=(
                        f"BPF program contains forbidden helper '{helper}'. "
                        f"Inspector eBPF programs must be read-only - no kernel writes, "
                        f"no return overrides, no packet modification."
                    ),
                )

        self._log.debug("safety_gate_ebpf_passed")
        return SafetyResult(passed=True, gate="ebpf_program_safety")

    # ── 5. Taint Tracking Scope ────────────────────────────────────────

    def validate_taint_tracking_scope(
        self,
        service_names: list[str],
        authorized_services: list[str] | None = None,
    ) -> SafetyResult:
        """
        Validate that taint tracking targets are within scope.

        Rejects tracking of system-level services (host, kernel, dockerd, etc.)
        and optionally enforces an authorized service list.

        Args:
            service_names: Compose services that will be taint-tracked.
            authorized_services: If provided, only these services are allowed.
                When None, all non-system services are allowed.

        Returns:
            SafetyResult with passed=True if all services are in scope.
        """
        for svc in service_names:
            normalized = svc.lower().strip()
            if normalized in _SYSTEM_SERVICE_BLOCKLIST:
                self._log.warning(
                    "safety_gate_taint_scope_blocked",
                    service=svc,
                )
                return SafetyResult(
                    passed=False,
                    gate="taint_tracking_scope",
                    reason=(
                        f"Cannot track system service '{svc}'. "
                        f"Taint tracking is limited to application-level "
                        f"docker-compose services."
                    ),
                )

        if authorized_services is not None:
            authorized_set = {s.lower().strip() for s in authorized_services}
            for svc in service_names:
                if svc.lower().strip() not in authorized_set:
                    self._log.warning(
                        "safety_gate_taint_scope_unauthorized",
                        service=svc,
                        authorized=authorized_services,
                    )
                    return SafetyResult(
                        passed=False,
                        gate="taint_tracking_scope",
                        reason=(
                            f"Service '{svc}' is not in the authorized tracking list. "
                            f"Authorized: {authorized_services}"
                        ),
                    )

        self._log.debug(
            "safety_gate_taint_scope_passed",
            services=service_names,
        )
        return SafetyResult(passed=True, gate="taint_tracking_scope")

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _check_forbidden_imports(tree: ast.AST) -> str:
        """
        Walk AST for forbidden module imports.

        Returns the first forbidden module name found, or empty string.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".")[0]
                    if root_module in FORBIDDEN_POC_MODULES:
                        return alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_module = node.module.split(".")[0]
                if root_module in FORBIDDEN_POC_MODULES:
                    return node.module
        return ""

    @staticmethod
    def _check_dangerous_calls(poc_code: str) -> str:
        """
        Scan PoC source for dangerous call patterns beyond imports.

        Returns the first matching pattern description, or empty string.
        """
        for pattern in _DANGEROUS_CALL_PATTERNS:
            match = pattern.search(poc_code)
            if match:
                return match.group(0).strip()
        return ""

    @staticmethod
    def _check_unauthorized_urls(
        poc_code: str,
        authorized_targets: list[str],
    ) -> list[str]:
        """
        Extract all URLs from PoC code and check against authorized list.

        Returns list of unauthorized domain strings found. Empty if all OK.
        Skips localhost and 127.0.0.1 - these are always considered safe
        for local testing.
        """
        if not authorized_targets:
            # No authorized targets configured - cannot verify any URLs
            return []

        # Always-safe targets (local testing)
        safe_domains = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}

        found_domains = set()
        for match in _URL_PATTERN.finditer(poc_code):
            domain = match.group(1)
            # Strip port number for comparison
            hostname = domain.split(":")[0].lower()
            found_domains.add(hostname)

        unauthorized = []
        for domain in found_domains:
            if domain in safe_domains:
                continue
            # Check if domain matches any authorized target (substring match
            # allows "api.example.com" to match authorized "example.com")
            if not any(
                domain == target or domain.endswith("." + target)
                for target in authorized_targets
            ):
                unauthorized.append(domain)

        return sorted(unauthorized)

    @staticmethod
    def _check_symlink_escapes(workspace_root: Path, max_depth: int = 3) -> Path | None:
        """
        Walk the workspace up to max_depth levels looking for symlinks that
        resolve outside the workspace boundary.

        Returns the first escaping symlink path, or None if all are safe.
        Limits depth to avoid traversing huge cloned repos.
        """
        resolved_root = workspace_root.resolve()

        def _walk(directory: Path, depth: int) -> Path | None:
            if depth > max_depth:
                return None
            try:
                for entry in directory.iterdir():
                    if entry.is_symlink():
                        target = entry.resolve()
                        if not _is_subpath(target, resolved_root):
                            return entry
                    if entry.is_dir() and not entry.is_symlink():
                        result = _walk(entry, depth + 1)
                        if result is not None:
                            return result
            except (PermissionError, OSError):
                pass  # Skip unreadable directories
            return None

        return _walk(workspace_root, 0)


# ── Result Type ──────────────────────────────────────────────────────────────


class SafetyResult:
    """
    Outcome of a safety gate check.

    Attributes:
        passed: True if the check succeeded.
        gate: Which gate produced this result (poc_execution, workspace_isolation,
              inspector_config).
        reason: Human-readable explanation of failure (empty on success).
    """

    __slots__ = ("passed", "gate", "reason")

    def __init__(
        self,
        *,
        passed: bool,
        gate: str,
        reason: str = "",
    ) -> None:
        self.passed = passed
        self.gate = gate
        self.reason = reason

    def __bool__(self) -> bool:
        return self.passed

    def __repr__(self) -> str:
        if self.passed:
            return f"SafetyResult(passed=True, gate={self.gate!r})"
        return (
            f"SafetyResult(passed=False, gate={self.gate!r}, "
            f"reason={self.reason!r})"
        )


# ── Utilities ────────────────────────────────────────────────────────────────


def _is_subpath(child: Path, parent: Path) -> bool:
    """Check if child is equal to or a descendant of parent (resolved paths)."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
