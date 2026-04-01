"""
EcodiaOS - Inspector Target Workspace

Abstraction for a target codebase (internal EOS or externally cloned repo).
Handles lifecycle: clone → analyze → cleanup.

Iron Rules:
  - Inspector NEVER writes to the EOS source tree.
  - All external targets live in randomised /tmp/inspector_workspace_<uuid> directories.
  - Network access during clone is the ONLY permitted network operation; after
    clone completes, the workspace object actively refuses further subprocess
    calls that could reach external hosts.
  - File writes are strictly confined to the temp workspace root. Any attempted
    write outside the boundary raises WorkspaceWriteViolation.
  - Cleanup is secure: every regular file is overwritten with random bytes before
    the directory tree is removed, preventing filesystem forensics.
  - Use as async context manager: `async with TargetWorkspace.from_github_url(…) as ws:`
    - cleanup is guaranteed even if an exception propagates.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Literal, Self

import structlog

logger = structlog.get_logger().bind(system="simula.inspector.workspace")


# ── Exceptions ───────────────────────────────────────────────────────────────


class WorkspaceWriteViolation(PermissionError):
    """
    Raised when code attempts to write outside the permitted workspace root.

    The path that triggered the violation is attached as `.attempted_path`.
    """

    def __init__(self, attempted: Path, workspace_root: Path) -> None:
        self.attempted_path = attempted
        self.workspace_root = workspace_root
        super().__init__(
            f"Write violation: '{attempted}' is outside workspace root '{workspace_root}'. "
            "Inspector may only write within its isolated temp directory."
        )


class WorkspaceNetworkViolation(PermissionError):
    """
    Raised when code attempts a network subprocess after the clone phase is
    complete. Only `git clone` during workspace construction is permitted.
    """

    def __init__(self, cmd: str) -> None:
        super().__init__(
            f"Network violation: subprocess command '{cmd}' is not permitted after "
            "workspace construction. Inspector network access ends when cloning ends."
        )


# ── Helpers ──────────────────────────────────────────────────────────────────

# Subprocess commands that are categorically network-touching. Any call to
# these after clone completes is blocked via assert_no_network_subprocess().
_NETWORK_COMMANDS: frozenset[str] = frozenset({
    "git", "curl", "wget", "pip", "pip3", "npm", "yarn", "pnpm",
    "poetry", "apt", "apt-get", "brew", "docker", "ssh", "scp", "rsync",
    "nc", "ncat", "netcat", "ftp", "sftp", "http", "https",
})


def _is_network_command(argv0: str) -> bool:
    """Return True if the base command name is in the blocked set."""
    return Path(argv0).name.lower() in _NETWORK_COMMANDS


def _secure_erase_file(path: Path) -> None:
    """
    Overwrite a file with random bytes equal to its size, then delete it.

    Best-effort - if the file is unreadable or already gone, skip silently.
    This is a single-pass wipe sufficient to prevent casual filesystem
    forensics; it is not a DoD 5220.22-M multi-pass wipe.
    """
    try:
        size = path.stat().st_size
        if size > 0:
            with path.open("r+b") as fh:
                fh.write(os.urandom(size))
                fh.flush()
                os.fsync(fh.fileno())
        path.unlink(missing_ok=True)
    except (OSError, PermissionError):
        # Best-effort: fall through, rmtree will remove it below.
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)


def _make_writable_recursive(root: Path) -> None:
    """
    Recursively chmod all files/dirs under *root* to be owner-writable so
    that rmtree can delete read-only files (common in git repos).
    """
    for dirpath, dirnames, filenames in os.walk(str(root)):
        for name in dirnames + filenames:
            try:
                full = Path(dirpath) / name
                current = full.stat().st_mode
                full.chmod(current | stat.S_IWRITE | stat.S_IWGRP)
            except OSError:
                pass


def _nuke_directory(root: Path) -> None:
    """
    Securely destroy a directory tree:

      1. Make everything writable (handles git's read-only pack files).
      2. Overwrite every regular file with random bytes (single-pass scrub).
      3. Remove the entire directory tree via rmtree.

    Logs a warning if removal fails - never raises; callers must not rely on
    this for correctness, only security.
    """
    if not root.exists():
        return

    _make_writable_recursive(root)

    # Pass 1: Secure-erase all regular files.
    for dirpath, _dirs, filenames in os.walk(str(root), topdown=False):
        for filename in filenames:
            _secure_erase_file(Path(dirpath) / filename)

    # Pass 2: Remove directory tree.
    try:
        shutil.rmtree(root, ignore_errors=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "workspace_nuke_partial_failure",
            root=str(root),
            error=str(exc),
        )
        # Final fallback - accept ignore_errors only if the explicit pass failed.
        shutil.rmtree(root, ignore_errors=True)


# ── Core Class ───────────────────────────────────────────────────────────────


class TargetWorkspace:
    """
    Isolated sandbox for a target codebase.

    Lifecycle
    ---------
    1. **Construct** via a factory classmethod:
       - ``from_github_url(url)``  - clone into a randomised temp dir
       - ``from_local_path(path)`` - wrap an existing local directory
       - ``internal(eos_root)``    - point at the read-only internal EOS tree

    2. **Use** the workspace via ``self.root`` (a resolved ``Path``).

    3. **Enforce** write boundaries with ``assert_write_allowed(path)`` before
       any file-write operation targeting the workspace; this raises
       ``WorkspaceWriteViolation`` if the target is outside the sandbox root.

    4. **Enforce** network silence with ``assert_no_network_subprocess(argv0)``
       before launching any subprocess; this raises ``WorkspaceNetworkViolation``
       for commands in the blocked set.

    5. **Cleanup** by calling ``cleanup()`` or using the async context manager::

           async with TargetWorkspace.from_github_url(url) as ws:
               # ws.root is available here
               ...
           # temp directory has been nuked from orbit here

    Iron Rules
    ----------
    - Internal workspaces are *never* cleaned up or written to.
    - External workspaces live in ``/tmp/inspector_workspace_<16-hex-char-token>/``
      (or the OS temp dir on Windows).
    - ``cleanup()`` performs a secure overwrite-then-delete pass over all files.
    """

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(
        self,
        root: Path,
        workspace_type: Literal["internal_eos", "external_repo"],
        temp_directory: Path | None = None,
    ) -> None:
        """
        Direct constructor - prefer the factory classmethods.

        Args:
            root: Absolute path to the codebase root (must exist and be a dir).
            workspace_type: ``"internal_eos"`` or ``"external_repo"``.
            temp_directory: The parent temp dir to nuke on cleanup (external only).

        Raises:
            FileNotFoundError: If *root* does not exist.
            NotADirectoryError: If *root* is not a directory.
        """
        resolved = root.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Workspace root does not exist: {resolved}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Workspace root is not a directory: {resolved}")

        self.root: Path = resolved
        self.workspace_type: Literal["internal_eos", "external_repo"] = workspace_type
        self.temp_directory: Path | None = temp_directory

        logger.info(
            "workspace_created",
            root=str(self.root),
            workspace_type=self.workspace_type,
            is_temp=self.temp_directory is not None,
        )

    @property
    def is_external(self) -> bool:
        """True if this workspace targets an external (non-EOS) codebase."""
        return self.workspace_type == "external_repo"

    # ── Factory Classmethods ──────────────────────────────────────────────────

    @classmethod
    async def from_github_url(
        cls,
        github_url: str,
        *,
        clone_depth: int = 1,
    ) -> TargetWorkspace:
        """
        Clone a GitHub repository into a randomised temp directory.

        The temp directory is named ``inspector_workspace_<16-hex-token>`` and
        lives under the OS temp dir (``/tmp`` on Linux/macOS). The random token
        is generated via ``secrets.token_hex(8)`` - cryptographically random,
        not predictable from PID or timestamp.

        Network access is *only* permitted during this clone. Once the workspace
        object is returned, all subsequent subprocess calls through
        ``assert_no_network_subprocess()`` will block network commands.

        Args:
            github_url: HTTPS URL of the repository to clone.
            clone_depth: Git clone depth (1 = shallow clone for speed).

        Returns:
            A ``TargetWorkspace`` ready for analysis; call ``cleanup()`` when done
            or use the async context manager.

        Raises:
            RuntimeError: If git clone fails (non-zero exit code).
        """
        token = secrets.token_hex(8)  # 16 hex chars - 64 bits of randomness
        temp_dir = Path(tempfile.gettempdir()) / f"inspector_workspace_{token}"
        temp_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
        clone_target = temp_dir / "repo"

        log = logger.bind(url=github_url, target=str(clone_target), depth=clone_depth)
        log.info("cloning_repo")

        # Start from the full inherited environment so Windows DLL loading,
        # DNS resolution (SYSTEMROOT, USERPROFILE), and PATH all work correctly.
        # Then apply safety overrides on top to enforce hermetic clone behaviour.
        clone_env = os.environ.copy()
        clone_env.update({
            "GIT_CONFIG_NOSYSTEM": "1",   # Ignore /etc/gitconfig - hermetic clone
            "GIT_TERMINAL_PROMPT": "0",   # Never hang on auth prompts
            "GIT_ASKPASS": "echo",        # Prevent credential-prompt hangs
            "TMPDIR": tempfile.gettempdir(),
            "TEMP": tempfile.gettempdir(),
            "TMP": tempfile.gettempdir(),
        })

        proc = await asyncio.create_subprocess_exec(
            "git", "clone",
            "--depth", str(max(1, clone_depth)),
            "--no-tags",           # Skip tag fetch - reduces attack surface
            "--single-branch",     # Only clone the default branch
            github_url,
            str(clone_target),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=clone_env,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            # Clean up on failure - secure nuke, not just rmtree
            _nuke_directory(temp_dir)
            error_msg = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"git clone failed (exit {proc.returncode}): {error_msg}"
            )

        log.info("clone_complete", root=str(clone_target))

        return cls(
            root=clone_target,
            workspace_type="external_repo",
            temp_directory=temp_dir,
        )

    @classmethod
    def from_local_path(cls, path: Path) -> TargetWorkspace:
        """
        Wrap an existing local directory as an external workspace.

        No temp directory is created; cleanup() is a no-op for this variant.
        Useful for analysing local repos without cloning.
        """
        return cls(
            root=path,
            workspace_type="external_repo",
            temp_directory=None,
        )

    @classmethod
    def internal(cls, eos_root: Path) -> TargetWorkspace:
        """
        Create a read-only workspace pointing at the internal EOS codebase.

        Cleanup and write assertions are enforced normally - if something tries
        to write to EOS source via this workspace, ``WorkspaceWriteViolation``
        is raised immediately.
        """
        return cls(
            root=eos_root,
            workspace_type="internal_eos",
            temp_directory=None,
        )

    # ── Guard Rails ───────────────────────────────────────────────────────────

    def assert_write_allowed(self, target: Path) -> None:
        """
        Assert that *target* is inside the workspace root before any write.

        Resolves symlinks (``Path.resolve()``) to prevent symlink-escape attacks
        where a path inside the workspace is a symlink to a path outside it.

        Args:
            target: Absolute path the caller intends to write to.

        Raises:
            WorkspaceWriteViolation: If *target* escapes the workspace root.
            PermissionError: If this is an internal EOS workspace (read-only).
        """
        if self.workspace_type == "internal_eos":
            raise PermissionError(
                "Internal EOS workspace is read-only. "
                "Inspector NEVER writes to the EOS source tree."
            )

        # Resolve to defeat symlink escapes
        try:
            resolved_target = target.resolve()
        except OSError:
            # Path doesn't exist yet - resolve its closest existing ancestor
            resolved_target = target.absolute()

        try:
            resolved_target.relative_to(self.root)
        except ValueError:
            raise WorkspaceWriteViolation(resolved_target, self.root)

    def assert_no_network_subprocess(self, argv0: str) -> None:
        """
        Assert that *argv0* is not a network-touching command.

        Called before launching any subprocess *after* workspace construction.
        The clone phase is the only permitted network window; this guard
        enforces silence once the workspace object exists.

        Args:
            argv0: First element of the subprocess argument list.

        Raises:
            WorkspaceNetworkViolation: If the command is in the blocked set.
        """
        if _is_network_command(argv0):
            raise WorkspaceNetworkViolation(argv0)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """
        Securely destroy the temporary workspace.

        For external workspaces with a ``temp_directory``:
          1. Make all files/dirs writable (handles git's read-only pack files).
          2. Overwrite every regular file with random bytes (single-pass scrub).
          3. Remove the entire directory tree.

        For internal or local workspaces (``temp_directory is None``): no-op.

        This method is idempotent - calling it multiple times is safe.
        """
        if self.temp_directory is not None and self.temp_directory.exists():
            logger.info(
                "workspace_nuking",
                temp_directory=str(self.temp_directory),
            )
            _nuke_directory(self.temp_directory)
            logger.info(
                "workspace_nuked",
                temp_directory=str(self.temp_directory),
            )
        # Mark cleaned up so repeated calls are no-ops
        self.temp_directory = None

    # ── Async Context Manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> Self:
        """Enter the context - returns self for ``async with … as ws:`` usage."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit the context - always nukes the temp workspace, even on exception."""
        self.cleanup()

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"TargetWorkspace("
            f"root={self.root!r}, "
            f"type={self.workspace_type!r}, "
            f"temp={self.temp_directory is not None}"
            f")"
        )
