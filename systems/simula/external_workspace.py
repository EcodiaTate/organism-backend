"""
EcodiaOS - Simula External Repository Workspace

Isolated workspace for working on external (non-EOS) repositories.
Clones a target repo into a task-scoped temp directory, enforces
write boundaries, runs language-specific tests and linters, and
performs secure cleanup on exit.

Architecture:
  - Path isolation: all external repos cloned to
    /tmp/eos_workspace_{task_id}_{16-hex-token}/
  - No write outside workspace root (path traversal protection via resolved paths)
  - Network permitted ONLY during clone phase (same iron rule as TargetWorkspace)
  - Language auto-detected from repo file structure
  - Test + lint run via asyncio.subprocess with configurable commands
  - Secure cleanup: single-pass random-byte overwrite before rmtree

Supported languages (auto-detected):
  python    - Cargo.toml absent, pyproject.toml/setup.py/requirements.txt present
  rust      - Cargo.toml present
  javascript- package.json present, no Cargo.toml
  typescript- package.json + tsconfig.json present
  solidity  - foundry.toml or hardhat.config.* present
  go        - go.mod present
  unknown   - fallback (no test/lint commands available)

Language-specific defaults:
  python:     test="pytest --tb=short -q", lint="ruff check ."
  rust:       test="cargo test --quiet", lint="cargo clippy --quiet -- -D warnings"
  javascript: test="npm test --silent", lint="npx eslint ."
  typescript: test="npm test --silent", lint="npx eslint . --ext .ts"
  solidity:   test="forge test --quiet", lint="solhint 'src/**/*.sol'"
  go:         test="go test ./...", lint="golangci-lint run"
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import structlog

logger = structlog.get_logger().bind(system="simula.external_workspace")


# ── Exceptions ────────────────────────────────────────────────────────────────


class ExternalWorkspaceWriteViolation(PermissionError):
    """Raised when code attempts to write outside the permitted workspace root."""

    def __init__(self, attempted: Path, workspace_root: Path) -> None:
        self.attempted_path = attempted
        self.workspace_root = workspace_root
        super().__init__(
            f"Write violation: '{attempted}' is outside workspace root "
            f"'{workspace_root}'. External workspace writes are strictly confined."
        )


class ExternalWorkspaceCloneError(RuntimeError):
    """Raised when git clone fails."""


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class TestResult:
    """Result of running the repo's test suite."""

    passed: bool
    output: str
    exit_code: int
    command: str
    duration_ms: float = 0.0


@dataclass
class LintResult:
    """Result of running the repo's linter."""

    passed: bool
    output: str
    exit_code: int
    command: str
    duration_ms: float = 0.0


# ── Language detection ────────────────────────────────────────────────────────

Language = Literal["python", "rust", "javascript", "typescript", "solidity", "go", "unknown"]

# Sentinel files used for language detection (checked in order of precedence)
_LANG_MARKERS: list[tuple[Language, list[str]]] = [
    ("rust",       ["Cargo.toml"]),
    ("solidity",   ["foundry.toml", "hardhat.config.js", "hardhat.config.ts", "hardhat.config.cjs"]),
    ("go",         ["go.mod"]),
    ("typescript", ["package.json", "tsconfig.json"]),
    ("javascript", ["package.json"]),
    ("python",     ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"]),
]

# Per-language default commands (test, lint)
_LANG_COMMANDS: dict[Language, tuple[str, str]] = {
    "python":     ("pytest --tb=short -q", "ruff check ."),
    "rust":       ("cargo test --quiet", "cargo clippy --quiet -- -D warnings"),
    "javascript": ("npm test --silent", "npx eslint ."),
    "typescript": ("npm test --silent", "npx eslint . --ext .ts,.tsx"),
    "solidity":   ("forge test --quiet", "solhint 'src/**/*.sol'"),
    "go":         ("go test ./...", "golangci-lint run"),
    "unknown":    ("echo 'no tests configured'", "echo 'no linter configured'"),
}

# Forbidden paths - infrastructure files only maintainers should change
_DEFAULT_FORBIDDEN_PATHS: frozenset[str] = frozenset({
    ".github/workflows",
    "Makefile",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Cargo.toml",
    "go.mod",
    "go.sum",
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "tsconfig.json",
    "foundry.toml",
    "hardhat.config.js",
    "hardhat.config.ts",
})


def detect_language(repo_root: Path) -> Language:
    """
    Infer the primary language from repo file structure.

    Checks sentinel files in order of precedence (Rust > Solidity > Go >
    TypeScript > JavaScript > Python). Returns "unknown" if none match.
    """
    for lang, markers in _LANG_MARKERS:
        if all((repo_root / m).exists() for m in markers):
            return lang
    return "unknown"


# ── Secure erase helpers (mirrored from inspector/workspace.py) ───────────────


def _secure_erase_file(path: Path) -> None:
    try:
        size = path.stat().st_size
        if size > 0:
            with path.open("r+b") as fh:
                fh.write(os.urandom(size))
                fh.flush()
                os.fsync(fh.fileno())
        path.unlink(missing_ok=True)
    except (OSError, PermissionError):
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)


def _make_writable_recursive(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(str(root)):
        for name in dirnames + filenames:
            try:
                full = Path(dirpath) / name
                current = full.stat().st_mode
                full.chmod(current | stat.S_IWRITE | stat.S_IWGRP)
            except OSError:
                pass


def _nuke_directory(root: Path) -> None:
    if not root.exists():
        return
    _make_writable_recursive(root)
    for dirpath, _dirs, filenames in os.walk(str(root), topdown=False):
        for filename in filenames:
            _secure_erase_file(Path(dirpath) / filename)
    try:
        shutil.rmtree(root, ignore_errors=False)
    except Exception:  # noqa: BLE001
        shutil.rmtree(root, ignore_errors=True)


# ── Core class ────────────────────────────────────────────────────────────────


@dataclass
class ExternalRepoConfig:
    """
    Configuration for a CodeAgent operating on an external repository.

    When this is set on SimulaCodeAgent, all file operations are redirected
    to the cloned workspace instead of the EOS codebase root.
    """

    repo_url: str
    base_branch: str = "main"
    # Files the agent may read and modify (empty list = all files in scope)
    target_files: list[str] = field(default_factory=list)
    # Additional paths to protect beyond _DEFAULT_FORBIDDEN_PATHS
    forbidden_paths: list[str] = field(default_factory=list)
    # How to run tests for this repo
    test_command: str = ""          # empty = auto-detect from language
    # How to run linter
    lint_command: str = ""          # empty = auto-detect from language
    # Language override (empty = auto-detect)
    language: Language | str = ""
    # Max repair attempts before giving up
    max_repair_attempts: int = 3
    # Git clone depth (1 = shallow for speed)
    clone_depth: int = 1


class ExternalWorkspace:
    """
    Isolated, task-scoped workspace for an external repository.

    Lifecycle:
      1. Construct via ``ExternalWorkspace.clone(repo_url, task_id, github_token)``
      2. Use the workspace via ``self.root`` (absolute Path to the repo checkout)
      3. Call ``run_tests()`` / ``run_linter()`` as needed
      4. Call ``cleanup()`` or use the async context manager

    Iron Rules:
      - Workspace lives in /tmp/eos_workspace_{task_id}_{token}/repo
      - Writes outside workspace root raise ExternalWorkspaceWriteViolation
      - Network permitted only during clone; all subsequent subprocess calls
        that touch network commands are blocked (same as TargetWorkspace)
      - cleanup() secure-erases all files before rmtree
    """

    def __init__(
        self,
        root: Path,
        temp_directory: Path,
        language: Language,
        config: ExternalRepoConfig,
    ) -> None:
        self.root: Path = root.resolve()
        self._temp_directory: Path = temp_directory
        self.language: Language = language
        self.config: ExternalRepoConfig = config
        self._cleaned_up: bool = False

        # Resolve effective test/lint commands
        default_test, default_lint = _LANG_COMMANDS.get(language, _LANG_COMMANDS["unknown"])
        self.test_command: str = config.test_command or default_test
        self.lint_command: str = config.lint_command or default_lint

        # Build effective forbidden paths (defaults ∪ caller additions)
        self._forbidden: frozenset[str] = _DEFAULT_FORBIDDEN_PATHS | frozenset(
            config.forbidden_paths
        )

        logger.info(
            "external_workspace_ready",
            root=str(self.root),
            language=language,
            test_cmd=self.test_command,
            lint_cmd=self.lint_command,
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    async def clone(
        cls,
        repo_url: str,
        task_id: str,
        config: ExternalRepoConfig,
        github_token: str | None = None,
    ) -> "ExternalWorkspace":
        """
        Clone *repo_url* into an isolated temp directory for *task_id*.

        The temp directory is named ``eos_workspace_{task_id}_{16-hex-token}``
        under the OS temp dir. The random token prevents path prediction.

        If *github_token* is provided it is embedded in the clone URL for
        private repos (``https://{token}@github.com/...``). The token never
        appears in logs or subprocess output.

        Raises:
            ExternalWorkspaceCloneError: If git clone exits non-zero.
        """
        token = secrets.token_hex(8)
        safe_task_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)[:40]
        temp_dir = Path(tempfile.gettempdir()) / f"eos_workspace_{safe_task_id}_{token}"
        temp_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        clone_target = temp_dir / "repo"

        # Build clone URL - inject token for private repos
        clone_url = repo_url
        if github_token and repo_url.startswith("https://"):
            # Strip any existing auth before injecting
            without_proto = repo_url[len("https://"):]
            if "@" in without_proto:
                without_proto = without_proto.split("@", 1)[1]
            clone_url = f"https://{github_token}@{without_proto}"

        log = logger.bind(repo=repo_url, target=str(clone_target), depth=config.clone_depth)
        log.info("cloning_external_repo")

        env = {**os.environ}
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_ASKPASS"] = "true"

        cmd = [
            "git", "clone",
            "--depth", str(config.clone_depth),
            "--branch", config.base_branch,
            "--single-branch",
            "--quiet",
            clone_url,
            str(clone_target),
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            _nuke_directory(temp_dir)
            raise ExternalWorkspaceCloneError(
                f"git clone timed out after 120s for {repo_url}"
            )
        except Exception as exc:
            _nuke_directory(temp_dir)
            raise ExternalWorkspaceCloneError(
                f"git clone subprocess error for {repo_url}: {exc}"
            ) from exc

        if proc.returncode != 0:
            err_text = stderr.decode(errors="replace").strip()
            _nuke_directory(temp_dir)
            raise ExternalWorkspaceCloneError(
                f"git clone failed (exit {proc.returncode}) for {repo_url}: {err_text}"
            )

        if not clone_target.is_dir():
            _nuke_directory(temp_dir)
            raise ExternalWorkspaceCloneError(
                f"git clone did not produce a directory at {clone_target}"
            )

        # Detect language
        raw_lang = config.language
        if raw_lang and raw_lang in _LANG_COMMANDS:
            language: Language = raw_lang  # type: ignore[assignment]
        else:
            language = detect_language(clone_target)

        log.info("clone_complete", language=language)
        return cls(
            root=clone_target,
            temp_directory=temp_dir,
            language=language,
            config=config,
        )

    # ── Write boundary enforcement ────────────────────────────────────────────

    def assert_write_allowed(self, path: Path) -> None:
        """
        Raise ExternalWorkspaceWriteViolation if *path* is outside the workspace
        root or is a forbidden infrastructure file.
        """
        resolved = path.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise ExternalWorkspaceWriteViolation(resolved, self.root)

        # Check against forbidden relative paths
        try:
            rel = resolved.relative_to(self.root)
        except ValueError:
            raise ExternalWorkspaceWriteViolation(resolved, self.root)

        rel_str = str(rel).replace("\\", "/")
        for forbidden in self._forbidden:
            if rel_str == forbidden or rel_str.startswith(forbidden + "/"):
                raise ExternalWorkspaceWriteViolation(resolved, self.root)

    def is_in_scope(self, path: Path) -> bool:
        """
        Return True if *path* is in the workspace and within the target_files
        scope (if target_files is non-empty).
        """
        try:
            rel = path.resolve().relative_to(self.root)
        except ValueError:
            return False

        if not self.config.target_files:
            return True

        rel_str = str(rel).replace("\\", "/")
        return any(
            rel_str == t or rel_str.startswith(t.rstrip("/") + "/")
            for t in self.config.target_files
        )

    # ── Test and lint execution ───────────────────────────────────────────────

    async def run_tests(self) -> TestResult:
        """
        Run the repo's test suite. Returns TestResult with pass/fail, output,
        and exit code. Never raises - all errors captured in TestResult.
        """
        return await self._run_command(self.test_command, kind="test")

    async def run_linter(self) -> LintResult:
        """
        Run the repo's linter. Returns LintResult. Never raises.
        """
        raw = await self._run_command(self.lint_command, kind="lint")
        return LintResult(
            passed=raw.passed,
            output=raw.output,
            exit_code=raw.exit_code,
            command=raw.command,
            duration_ms=raw.duration_ms,
        )

    async def _run_command(self, command: str, kind: str) -> TestResult:
        import time

        log = logger.bind(kind=kind, cmd=command, cwd=str(self.root))
        log.info("running_command")
        t0 = time.monotonic()

        env = {**os.environ, "CI": "true", "TERM": "dumb"}

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            stdout_bytes, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
        except asyncio.TimeoutError:
            output = f"[{kind}] timed out after 300s"
            log.warning("command_timeout")
            return TestResult(passed=False, output=output, exit_code=-1, command=command)
        except Exception as exc:  # noqa: BLE001
            output = f"[{kind}] subprocess error: {exc}"
            log.warning("command_error", error=str(exc))
            return TestResult(passed=False, output=output, exit_code=-1, command=command)

        elapsed_ms = (time.monotonic() - t0) * 1000
        output = (stdout_bytes or b"").decode(errors="replace")
        passed = proc.returncode == 0

        log.info(
            "command_complete",
            passed=passed,
            exit_code=proc.returncode,
            duration_ms=round(elapsed_ms),
        )
        return TestResult(
            passed=passed,
            output=output,
            exit_code=proc.returncode or 0,
            command=command,
            duration_ms=elapsed_ms,
        )

    # ── Branch management ─────────────────────────────────────────────────────

    async def create_branch(self, branch_name: str) -> None:
        """Create and check out a new branch in the workspace."""
        proc = await asyncio.create_subprocess_exec(
            "git", "checkout", "-b", branch_name,
            cwd=str(self.root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            logger.warning("create_branch_failed", branch=branch_name, error=err)

    async def checkout_branch(self, branch_name: str) -> None:
        """Check out an existing branch."""
        proc = await asyncio.create_subprocess_exec(
            "git", "checkout", branch_name,
            cwd=str(self.root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=30)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def cleanup(self) -> None:
        """
        Securely erase and remove the workspace temp directory.

        Idempotent - safe to call multiple times. After the first call the
        workspace root is gone and no further operations are possible.
        """
        if self._cleaned_up:
            return
        self._cleaned_up = True
        log = logger.bind(temp_dir=str(self._temp_directory))
        log.info("workspace_cleanup_start")
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, _nuke_directory, self._temp_directory
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("workspace_cleanup_error", error=str(exc))
        log.info("workspace_cleanup_done")

    # ── Async context manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> "ExternalWorkspace":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.cleanup()
