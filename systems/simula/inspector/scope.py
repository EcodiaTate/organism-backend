"""
EcodiaOS — Inspector Scope Enforcement

Provides BountyScope and ScopeEnforcer to restrict TargetIngestor scanning to
paths explicitly authorised by Bug Bounty Rules of Engagement.

Design rules (strict, in priority order):
  1. Any path matching an out_of_scope_paths pattern → DENY (regardless of in-scope rules).
  2. If in_scope_paths is empty → ALLOW (default-allow when no allowlist is defined).
  3. Any path matching at least one in_scope_paths pattern → ALLOW.
  4. Otherwise → DENY.

Regex patterns are compiled at construction time. Malformed patterns raise
ValueError immediately so callers discover misconfiguration early.
"""

from __future__ import annotations

import re
from typing import Final

import structlog
from pydantic import Field, field_validator, model_validator

from primitives.common import EOSBaseModel

logger: Final = structlog.get_logger().bind(system="simula.inspector.scope")


class BountyScope(EOSBaseModel):
    """
    Defines the scanning perimeter for a Bug Bounty engagement.

    Each entry in ``in_scope_paths`` / ``out_of_scope_paths`` is a Python
    ``re`` pattern matched against the *relative* path of every file candidate
    (using ``re.search``, so anchoring with ``^`` is optional but recommended
    for prefix matching).

    Examples::

        BountyScope(
            in_scope_paths=[r"^src/api/", r"^contracts/"],
            out_of_scope_paths=[r"^src/api/internal/", r"migrations/"],
        )
    """

    in_scope_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Regex patterns for directories/files allowed to be scanned. "
            "Empty list means 'allow all' (subject to out_of_scope_paths)."
        ),
    )
    out_of_scope_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Regex patterns for directories/files strictly forbidden from scanning. "
            "These take precedence over in_scope_paths unconditionally."
        ),
    )

    @field_validator("in_scope_paths", "out_of_scope_paths", mode="before")
    @classmethod
    def _reject_empty_patterns(cls, patterns: list[str]) -> list[str]:
        """Blank pattern strings cause re.search to match everything — reject them."""
        for p in patterns:
            if not p or not p.strip():
                raise ValueError(
                    "Scope patterns must be non-empty strings; "
                    f"found empty or whitespace-only pattern: {p!r}"
                )
        return patterns

    @model_validator(mode="after")
    def _compile_check(self) -> BountyScope:
        """
        Validate that every pattern is a syntactically valid ``re`` expression.
        Raises ValueError (pydantic-compatible) for any malformed pattern.
        """
        for pattern in self.in_scope_paths + self.out_of_scope_paths:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"Invalid regex pattern in BountyScope: {pattern!r} — {exc}"
                ) from exc
        return self


class ScopeEnforcer:
    """
    Evaluates whether a relative file path falls within a BountyScope.

    Compiled patterns are cached at construction time so that per-file checks
    in ``map_attack_surfaces()`` are pure in-memory operations (no re-compilation
    on the hot path).
    """

    def __init__(self, scope: BountyScope) -> None:
        self._scope = scope
        self._log = logger.bind(
            in_scope_count=len(scope.in_scope_paths),
            out_of_scope_count=len(scope.out_of_scope_paths),
        )

        # Compile once; BountyScope.__post_init__ already validated syntax.
        self._in_scope_compiled: list[re.Pattern[str]] = [
            re.compile(p) for p in scope.in_scope_paths
        ]
        self._out_of_scope_compiled: list[re.Pattern[str]] = [
            re.compile(p) for p in scope.out_of_scope_paths
        ]

        self._log.debug(
            "scope_enforcer_initialised",
            in_scope_patterns=scope.in_scope_paths,
            out_of_scope_patterns=scope.out_of_scope_paths,
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def is_path_allowed(self, relative_path: str) -> bool:
        """
        Determine whether *relative_path* is permitted under the current scope.

        Args:
            relative_path: Path relative to the workspace root (e.g.
                           ``"src/api/users.py"``).  Both forward and
                           back-slash separators are normalised to ``/``
                           before matching.

        Returns:
            ``False`` if the path matches any out-of-scope pattern.
            ``True``  if no allowlist is configured (empty in_scope_paths).
            ``True``  if the path matches at least one in-scope pattern.
            ``False`` otherwise.
        """
        # Normalise path separators so patterns can use forward slashes only.
        normalised = relative_path.replace("\\", "/")

        # Rule 1: out-of-scope always wins.
        for pattern in self._out_of_scope_compiled:
            if pattern.search(normalised):
                return False

        # Rule 2: empty allowlist → default allow.
        if not self._in_scope_compiled:
            return True

        # Rule 3: explicit allowlist — must match at least one pattern.
        return any(pattern.search(normalised) for pattern in self._in_scope_compiled)
