"""
EcodiaOS - Inspector Inter-Procedural Call Graph Tracer

Follows function calls across file boundaries so the Z3 prover receives the
full execution context for an attack surface, not just the entry-point body.

Design
------
For each entry-point code block the tracer:

  1. Extracts all identifiers that look like function calls from the source.
  2. Scans the source file's import statements to map those identifiers to
     source files in the workspace.
  3. Opens each resolved file and extracts the callee function body.
  4. Repeats recursively up to *max_depth* levels.

The output is a single string with clearly labelled sections per file, ready
to be placed into an LLM prompt as structured context.

Language support
----------------
- Python  - AST-based call extraction + import resolution
- JS/TS   - regex-based call extraction + ES-module / CommonJS import resolution
- Solidity - regex-based call extraction + import path resolution
- Go/Rust  - regex-based call extraction + best-effort import resolution

All I/O is synchronous (``pathlib`` reads).  The method signature is
``async`` for interface consistency with the rest of the Inspector pipeline, but
no actual ``await`` is needed today; wrapping in ``asyncio.to_thread`` can be
added when the workspace moves to async I/O.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING, Final

import structlog

if TYPE_CHECKING:
    from systems.simula.inspector.workspace import TargetWorkspace

logger: Final = structlog.get_logger().bind(system="simula.inspector.tracer")

# Hard cap on the total stitched context returned to the caller.
_DEFAULT_MAX_CHARS: Final[int] = 16_000

# Per-section header template.
_SECTION_HEADER = "// --- File: {rel_path} ({label}) ---\n"

# Max characters to include from a single extracted callee body.
_MAX_CALLEE_CHARS: Final[int] = 3_000

# ── Extension sets ────────────────────────────────────────────────────────────

_PYTHON_EXTS: Final = {".py"}
_JS_TS_EXTS: Final = {".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs"}
_SOLIDITY_EXTS: Final = {".sol"}
_GO_EXTS: Final = {".go"}
_RUST_EXTS: Final = {".rs"}


# ── Call-extraction patterns (per language) ───────────────────────────────────

# JS/TS: bare identifier calls  →  foo(  or  await foo(
_JS_CALL_PATTERN: Final = re.compile(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")

# Solidity: same shape as JS
_SOL_CALL_PATTERN: Final = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# Go: identifier( - captures both plain calls and method calls
_GO_CALL_PATTERN: Final = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# Rust: identifier( - same pattern; macro calls (!() ) are excluded below
_RUST_CALL_PATTERN: Final = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# Keywords that look like calls but aren't user-defined functions.
_BUILTIN_BLACKLIST: Final = frozenset({
    # Python
    "print", "len", "range", "enumerate", "zip", "map", "filter", "sorted",
    "list", "dict", "set", "tuple", "int", "str", "float", "bool", "type",
    "isinstance", "issubclass", "hasattr", "getattr", "setattr", "super",
    "open", "repr", "abs", "round", "min", "max", "sum", "any", "all",
    # JS/TS
    "if", "for", "while", "switch", "catch", "function", "class", "new",
    "return", "typeof", "instanceof", "require", "import", "export",
    "console", "Promise", "setTimeout", "setInterval", "fetch", "Array",
    "Object", "String", "Number", "Boolean", "Symbol", "Error", "JSON",
    # Go
    "make", "append", "copy", "delete", "close", "panic", "recover",
    "cap", "println", # Rust
    "eprintln", "format", "vec", "assert", "Some", "None", "Ok", "Err",
    # Solidity
    "revert", "emit", "selfdestruct", "keccak256",
    "abi", "msg", "block", "tx",
})


# ── Import-resolution patterns ────────────────────────────────────────────────

# JS/TS ES-module: import { foo, bar } from './path/to/module'
_JS_ES_IMPORT: Final = re.compile(
    r"""
    import\s+
    (?:
        (?:\*\s+as\s+\w+)           # import * as ns
      | (?:\w+)                      # import DefaultExport
      | (?:\{[^}]*\})                # import { named, ... }
      | (?:\w+\s*,\s*\{[^}]*\})     # import Default, { named }
    )
    \s+from\s+
    ['"`]([^'"`]+)['"`]              # capture the module specifier
    """,
    re.VERBOSE,
)

# CommonJS: const foo = require('./path')  or  const { foo } = require('./path')
_JS_CJS_REQUIRE: Final = re.compile(
    r"""
    require\s*\(\s*['"`]([^'"`]+)['"`]\s*\)
    """,
    re.VERBOSE,
)

# Python: from .relative.module import foo  /  from absolute.module import foo
_PY_FROM_IMPORT: Final = re.compile(
    r"^\s*from\s+(\.{0,3}[\w./]+)\s+import\s+(.+)$",
    re.MULTILINE,
)

# Python: import module  /  import module as alias
_PY_BARE_IMPORT: Final = re.compile(
    r"^\s*import\s+([\w.]+)(?:\s+as\s+\w+)?",
    re.MULTILINE,
)

# Solidity: import "./path/Contract.sol"  or  import {Foo} from "./path.sol"
_SOL_IMPORT: Final = re.compile(
    r"""
    import\s+
    (?:
        \{[^}]*\}\s+from\s+   # named import
      | \*\s+as\s+\w+\s+from\s+
      |                        # plain import
    )
    ['"]([^'"]+)['"]
    """,
    re.VERBOSE,
)

# Go: import "pkg/path"  or  import alias "pkg/path"
_GO_IMPORT: Final = re.compile(
    r'import\s+(?:\w+\s+)?"([^"]+)"',
)

# Rust: use path::to::module;  or  use path::to::{Foo, Bar};
_RUST_USE: Final = re.compile(
    r"^\s*use\s+([\w:]+)(?:::\{[^}]*\})?;",
    re.MULTILINE,
)


class CallGraphTracer:
    """
    Follows function calls from an entry-point code block across file
    boundaries, building a multi-file execution context string.

    Usage::

        tracer = CallGraphTracer()
        context = await tracer.build_execution_context(
            workspace=ws,
            source_file="routes/api.ts",
            entry_code="...",
            max_depth=2,
        )
    """

    # ── Public API ────────────────────────────────────────────────────────────

    async def build_execution_context(
        self,
        workspace: TargetWorkspace,
        source_file: str,
        entry_code: str,
        max_depth: int = 2,
        max_chars: int = _DEFAULT_MAX_CHARS,
    ) -> str:
        """
        Build a stitched multi-file execution context for an attack surface.

        Args:
            workspace:   The target workspace (provides ``workspace.root``).
            source_file: Relative path of the entry-point file within the workspace.
            entry_code:  Source text of the entry-point function/block.
            max_depth:   How many import-hops to follow (default 2).
            max_chars:   Hard cap on the total returned string length.

        Returns:
            A formatted string with one labelled section per file, total length
            capped at *max_chars*.  Returns just the entry-point block on any
            top-level I/O error.
        """
        log = logger.bind(source_file=source_file, max_depth=max_depth)
        log.debug("trace_started")

        sections: list[str] = []

        # Entry-point section - always first.
        sections.append(
            _SECTION_HEADER.format(rel_path=source_file, label="Entry Point")
            + entry_code
        )

        # Visited set prevents cycles (A→B→A).
        visited: set[str] = {source_file}

        try:
            self._trace(
                workspace=workspace,
                source_file=source_file,
                source_code=entry_code,
                depth=0,
                max_depth=max_depth,
                visited=visited,
                sections=sections,
                log=log,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("trace_unexpected_error", error=str(exc))

        result = "\n\n".join(sections)

        if len(result) > max_chars:
            log.debug("trace_truncated", original_len=len(result), max_chars=max_chars)
            result = result[:max_chars]

        log.debug("trace_complete", sections=len(sections), total_chars=len(result))
        return result

    # ── Recursive core ────────────────────────────────────────────────────────

    def _trace(
        self,
        workspace: TargetWorkspace,
        source_file: str,
        source_code: str,
        depth: int,
        max_depth: int,
        visited: set[str],
        sections: list[str],
        log: structlog.stdlib.BoundLogger,
    ) -> None:
        """
        Recursively follow calls in *source_code* (from *source_file*) up to
        *max_depth* levels.  Results are appended to *sections* in place.
        """
        if depth >= max_depth:
            return

        suffix = Path(source_file).suffix.lower()

        # 1. Extract the set of identifiers called in this code block.
        called_names = self._extract_calls(source_code, suffix)
        if not called_names:
            return

        # 2. Read the *full* source of the containing file to resolve imports.
        full_source = self._read_file(workspace.root / source_file)
        if full_source is None:
            return

        # 3. Map identifiers → candidate workspace-relative file paths.
        import_map = self._resolve_imports(
            source_code=full_source,
            source_file=source_file,
            suffix=suffix,
            workspace_root=workspace.root,
        )

        # 4. For each callee we can resolve, extract its body and recurse.
        for name in sorted(called_names):  # sorted for deterministic output
            if name not in import_map:
                continue

            dep_rel_path = import_map[name]
            if dep_rel_path in visited:
                continue
            visited.add(dep_rel_path)

            dep_abs = workspace.root / dep_rel_path
            dep_source = self._read_file(dep_abs)
            if dep_source is None:
                log.debug("dependency_unreadable", dep=dep_rel_path)
                continue

            callee_body = self._extract_function_body(dep_source, name, dep_abs.suffix.lower())
            if not callee_body:
                log.debug("callee_not_found", name=name, dep=dep_rel_path)
                continue

            log.debug(
                "dependency_included",
                callee=name,
                dep=dep_rel_path,
                depth=depth + 1,
            )

            sections.append(
                _SECTION_HEADER.format(
                    rel_path=dep_rel_path,
                    label=f"Dependency - {name}",
                )
                + callee_body[:_MAX_CALLEE_CHARS]
            )

            # Recurse into the dependency using the extracted body as context.
            self._trace(
                workspace=workspace,
                source_file=dep_rel_path,
                source_code=callee_body,
                depth=depth + 1,
                max_depth=max_depth,
                visited=visited,
                sections=sections,
                log=log,
            )

    # ── Call extraction ───────────────────────────────────────────────────────

    def _extract_calls(self, code: str, suffix: str) -> set[str]:
        """
        Return the set of function-name identifiers called in *code*.
        Language-specific patterns are applied; builtins are filtered out.
        """
        if suffix in _PYTHON_EXTS:
            return self._extract_python_calls(code)

        if suffix in _JS_TS_EXTS:
            pattern = _JS_CALL_PATTERN
        elif suffix in _SOLIDITY_EXTS:
            pattern = _SOL_CALL_PATTERN
        elif suffix in _GO_EXTS:
            pattern = _GO_CALL_PATTERN
        elif suffix in _RUST_EXTS:
            pattern = _RUST_CALL_PATTERN
        else:
            # Unknown language - best-effort identifier-call pattern.
            pattern = _JS_CALL_PATTERN

        names: set[str] = set()
        for m in pattern.finditer(code):
            name = m.group(1)
            if name not in _BUILTIN_BLACKLIST and len(name) > 1:
                names.add(name)
        return names

    def _extract_python_calls(self, code: str) -> set[str]:
        """AST-based call extraction for Python - more accurate than regex."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fallback to regex if the snippet isn't a valid standalone module.
            names: set[str] = set()
            for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", code):
                n = m.group(1)
                if n not in _BUILTIN_BLACKLIST and len(n) > 1:
                    names.add(n)
            return names

        called: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id not in _BUILTIN_BLACKLIST:
                called.add(func.id)
            elif isinstance(func, ast.Attribute) and func.attr not in _BUILTIN_BLACKLIST:
                called.add(func.attr)
        return called

    # ── Import resolution ────────────────────────────────────────────────────

    def _resolve_imports(
        self,
        source_code: str,
        source_file: str,
        suffix: str,
        workspace_root: Path,
    ) -> dict[str, str]:
        """
        Parse the import statements in *source_code* and return a mapping of
        ``{identifier_name: workspace_relative_path}`` for identifiers that
        resolve to files within the workspace.

        Unknown/third-party imports are silently dropped.
        """
        if suffix in _PYTHON_EXTS:
            return self._resolve_python_imports(source_code, source_file, workspace_root)
        if suffix in _JS_TS_EXTS:
            return self._resolve_js_imports(source_code, source_file, workspace_root)
        if suffix in _SOLIDITY_EXTS:
            return self._resolve_solidity_imports(source_code, source_file, workspace_root)
        if suffix in _GO_EXTS:
            return self._resolve_go_imports(source_code, source_file, workspace_root)
        if suffix in _RUST_EXTS:
            return self._resolve_rust_imports(source_code, source_file, workspace_root)
        return {}

    # ── Python import resolution ─────────────────────────────────────────────

    def _resolve_python_imports(
        self,
        source: str,
        source_file: str,
        workspace_root: Path,
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        source_dir = (workspace_root / source_file).parent

        # from .module import foo, bar  /  from package.module import Foo
        for m in _PY_FROM_IMPORT.finditer(source):
            raw_module = m.group(1).strip()
            names_str = m.group(2).strip()

            # Parse the imported names (handles aliases: foo as f).
            names: list[str] = []
            for part in names_str.split(","):
                token = part.strip().split(" as ")[0].strip()
                if token and token != "*":
                    names.append(token)

            resolved = self._resolve_python_module(raw_module, source_dir, workspace_root)
            if resolved is None:
                continue
            for name in names:
                result[name] = resolved

        # import module / import module as alias
        for m in _PY_BARE_IMPORT.finditer(source):
            module = m.group(1).strip()
            resolved = self._resolve_python_module(module, source_dir, workspace_root)
            if resolved is not None:
                # The bare module name itself can be used as a call prefix;
                # the last component is the most useful key.
                result[module.split(".")[-1]] = resolved

        return result

    def _resolve_python_module(
        self,
        module: str,
        source_dir: Path,
        workspace_root: Path,
    ) -> str | None:
        """
        Convert a Python module specifier to a workspace-relative file path.
        Returns ``None`` if the module cannot be resolved within the workspace.
        """
        # Count leading dots for relative imports.
        dots = len(module) - len(module.lstrip("."))
        clean = module.lstrip(".")

        if dots:
            # Relative import: climb `dots - 1` parent directories.
            base = source_dir
            for _ in range(dots - 1):
                base = base.parent
            parts = clean.split(".") if clean else []
        else:
            # Absolute import: try from workspace root.
            base = workspace_root
            parts = clean.split(".")

        candidate_dir = base
        for part in parts:
            candidate_dir = candidate_dir / part

        # Try as a package (directory with __init__.py).
        init = candidate_dir / "__init__.py"
        if init.exists():
            try:
                return str(init.relative_to(workspace_root)).replace("\\", "/")
            except ValueError:
                return None

        # Try as a plain module file.
        as_file = candidate_dir.with_suffix(".py")
        if as_file.exists():
            try:
                return str(as_file.relative_to(workspace_root)).replace("\\", "/")
            except ValueError:
                return None

        return None

    # ── JS/TS import resolution ───────────────────────────────────────────────

    def _resolve_js_imports(
        self,
        source: str,
        source_file: str,
        workspace_root: Path,
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        source_dir = (workspace_root / source_file).parent

        specifiers: list[tuple[str, str]] = []  # (raw_specifier, context_source)

        # ES-module imports.
        for m in _JS_ES_IMPORT.finditer(source):
            # Capture the named imports from the match context (the full match).
            specifiers.append((m.group(1), m.group(0)))

        # CommonJS requires.
        for m in _JS_CJS_REQUIRE.finditer(source):
            specifiers.append((m.group(1), m.group(0)))

        for specifier, match_text in specifiers:
            # Only follow relative paths (./  ../); skip node_modules.
            if not specifier.startswith("."):
                continue

            resolved = self._resolve_js_specifier(specifier, source_dir, workspace_root)
            if resolved is None:
                continue

            # Extract named identifiers from the import clause.
            names = _extract_js_imported_names(match_text)
            for name in names:
                result[name] = resolved

            # Also map the module filename stem as a fallback.
            stem = Path(specifier).stem
            if stem:
                result[stem] = resolved

        return result

    def _resolve_js_specifier(
        self,
        specifier: str,
        source_dir: Path,
        workspace_root: Path,
    ) -> str | None:
        """Resolve a relative JS/TS import specifier to a workspace-relative path."""
        candidate = (source_dir / specifier).resolve()

        # Try the path as-is (already has extension).
        if candidate.is_file():
            return _to_rel(candidate, workspace_root)

        # Try appending common extensions.
        for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"):
            with_ext = candidate.with_suffix(ext)
            if with_ext.is_file():
                return _to_rel(with_ext, workspace_root)

        # Try as a directory with index file.
        for ext in (".ts", ".tsx", ".js", ".jsx"):
            index = candidate / f"index{ext}"
            if index.is_file():
                return _to_rel(index, workspace_root)

        return None

    # ── Solidity import resolution ────────────────────────────────────────────

    def _resolve_solidity_imports(
        self,
        source: str,
        source_file: str,
        workspace_root: Path,
    ) -> dict[str, str]:
        result: dict[str, str] = {}
        source_dir = (workspace_root / source_file).parent

        for m in _SOL_IMPORT.finditer(source):
            path_str = m.group(1)
            if not path_str.startswith("."):
                continue

            candidate = (source_dir / path_str).resolve()
            if candidate.is_file():
                rel = _to_rel(candidate, workspace_root)
                if rel:
                    stem = Path(path_str).stem
                    result[stem] = rel

        return result

    # ── Go import resolution ──────────────────────────────────────────────────

    def _resolve_go_imports(
        self,
        source: str,
        source_file: str,
        workspace_root: Path,
    ) -> dict[str, str]:
        """
        Go imports are package paths, not file paths.  We resolve them by
        looking for a directory with a matching suffix under the workspace root.
        """
        result: dict[str, str] = {}

        for m in _GO_IMPORT.finditer(source):
            pkg_path = m.group(1)
            pkg_name = pkg_path.split("/")[-1]  # last segment is the package name

            # Search for a directory in the workspace matching the last segment.
            candidate_dir = workspace_root / pkg_path
            if candidate_dir.is_dir():
                # Return the first .go file in that package.
                for go_file in sorted(candidate_dir.glob("*.go")):
                    rel = _to_rel(go_file, workspace_root)
                    if rel:
                        result[pkg_name] = rel
                        break

        return result

    # ── Rust import resolution ────────────────────────────────────────────────

    def _resolve_rust_imports(
        self,
        source: str,
        source_file: str,
        workspace_root: Path,
    ) -> dict[str, str]:
        """
        Rust ``use`` paths are module paths.  We map the last path segment to
        a workspace ``.rs`` file.
        """
        result: dict[str, str] = {}
        source_dir = (workspace_root / source_file).parent

        for m in _RUST_USE.finditer(source):
            mod_path = m.group(1)
            segments = mod_path.split("::")
            last = segments[-1]

            # Try relative: sibling .rs file or subdirectory/mod.rs.
            rs_file = source_dir / f"{last}.rs"
            if rs_file.is_file():
                rel = _to_rel(rs_file, workspace_root)
                if rel:
                    result[last] = rel
                continue

            mod_rs = source_dir / last / "mod.rs"
            if mod_rs.is_file():
                rel = _to_rel(mod_rs, workspace_root)
                if rel:
                    result[last] = rel

        return result

    # ── Function-body extraction ──────────────────────────────────────────────

    def _extract_function_body(
        self,
        source: str,
        func_name: str,
        suffix: str,
    ) -> str:
        """
        Extract the body of *func_name* from *source*.

        Uses AST for Python (accurate) and brace-counting for other languages
        (best-effort).  Returns an empty string if the function is not found.
        """
        if suffix in _PYTHON_EXTS:
            return self._extract_python_function(source, func_name)

        if suffix in (_JS_TS_EXTS | _SOLIDITY_EXTS | _GO_EXTS | _RUST_EXTS):
            return self._extract_brace_function(source, func_name)

        return ""

    def _extract_python_function(self, source: str, func_name: str) -> str:
        """AST-walk and return the full source of a top-level or class-level function."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ""

        lines = source.splitlines()

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != func_name:
                continue
            start = node.lineno - 1
            end = node.end_lineno or start + 1
            return "\n".join(lines[start:end])

        return ""

    def _extract_brace_function(self, source: str, func_name: str) -> str:
        """
        Locate *func_name* in *source* using a simple brace-counter.

        Handles:
        - ``function foo(``    (JS/TS)
        - ``async function foo(``
        - ``export function foo(``  / ``export async function foo(``
        - ``export default function foo(``
        - ``func foo(``  (Go)
        - ``fn foo(``    (Rust)
        - ``function foo(``  (Solidity)

        Falls back to an empty string if not found or if brace-counting fails
        (e.g., unbalanced braces in a minified file).
        """
        # Pattern that catches the function keyword immediately before the name.
        pattern = re.compile(
            r"""
            (?:export\s+(?:default\s+)?(?:async\s+)?)?  # JS modifiers
            (?:function|async\s+function|func|fn)\s+     # keyword
            """ + re.escape(func_name) + r"""
            \s*[\(<]                                      # open paren or type-param
            """,
            re.VERBOSE,
        )

        m = pattern.search(source)
        if not m:
            return ""

        # Find the opening brace after the signature.
        brace_start = source.find("{", m.end())
        if brace_start == -1:
            return ""

        # Walk forward counting braces until balanced.
        depth = 0
        idx = brace_start
        for idx in range(brace_start, len(source)):
            ch = source[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
        else:
            # Reached EOF without closing - return what we have.
            pass

        # Include a small header: from the function keyword to the closing brace.
        snippet_start = m.start()
        return source[snippet_start : idx + 1]

    # ── Utility ───────────────────────────────────────────────────────────────

    def _read_file(self, path: Path) -> str | None:
        """Read a text file; return None on any I/O or encoding error."""
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None


# ── Module-level helpers ────────────────────────────────────────────────────


def _to_rel(absolute: Path, workspace_root: Path) -> str | None:
    """Return a forward-slash workspace-relative path, or None if outside root."""
    try:
        return str(absolute.relative_to(workspace_root)).replace("\\", "/")
    except ValueError:
        return None


def _extract_js_imported_names(import_text: str) -> list[str]:
    """
    Parse named imports from an ES-module import statement.

    Handles:
    - ``import { foo, bar as baz }``  → ["foo", "baz"]
    - ``import DefaultName``          → ["DefaultName"]
    - ``import DefaultName, { foo }`` → ["DefaultName", "foo"]
    - ``const { foo } = require(...)`` → ["foo"]
    - ``const foo = require(...)``     → ["foo"]
    """
    names: list[str] = []

    # Named import block: { foo, bar as baz, ... }
    named_block = re.search(r"\{([^}]*)\}", import_text)
    if named_block:
        for part in named_block.group(1).split(","):
            # "foo as bar" → keep "bar"; "foo" → keep "foo"
            segments = part.strip().split(" as ")
            alias = segments[-1].strip()
            if alias and re.match(r"^[A-Za-z_$]\w*$", alias):
                names.append(alias)

    # Default import: ``import DefaultExport from …`` or ``const foo = require``
    default_m = re.match(
        r"(?:import|const)\s+([A-Za-z_$]\w*)\s*(?:,|\s+from|=)",
        import_text.strip(),
    )
    if default_m:
        names.append(default_m.group(1))

    return names
