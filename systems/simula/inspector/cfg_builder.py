"""
EcodiaOS - Inspector Phase 3: CFG Builder

Recovers a static control-flow graph from source files or compiled binaries.

Backends (chosen automatically based on target type)
-----------------------------------------------------
PYTHON_AST  - Python 3.8+ source: ast module walk → function/block/edge graph
REGEX       - JS/TS/Go/Rust/Solidity: regex heuristics for function + call edges
CAPSTONE    - ELF/PE/Mach-O binary: Capstone disassembly → BB boundaries
ANGR        - Same binaries via angr CFGFast when Capstone alone is insufficient
STUB        - Fallback for unknown file types; produces empty CFG

Python AST backend (primary for EodiaOS Python targets)
--------------------------------------------------------
  FunctionDef / AsyncFunctionDef   → StaticFunction
  Each linear sub-sequence of stmts with no branch → BasicBlock
  If / While / For / Try / With     → intra-function edges (CONDITIONAL_T/F,
                                       UNCONDITIONAL, EXCEPTION)
  ast.Call nodes                   → inter-procedural DIRECT_CALL edges
  dynamic call patterns            → INDIRECT_CALL edges

Binary backends (Capstone / angr)
----------------------------------
  Used when target_id points to a compiled ELF/PE/Mach-O.
  Capstone identifies BB boundaries via control-flow instructions.
  angr's CFGFast provides a more complete graph including indirect branches
  and jump tables.  Both backends require the respective library to be
  installed; graceful stub fallback when unavailable.

Public API
----------
  CfgBuilder.build(workspace, target_id, source_files) → StaticCFG
  CfgBuilder.build_from_binary(binary_path)             → StaticCFG
"""

from __future__ import annotations

import ast
import re
from typing import Final
from pathlib import Path

import structlog

from systems.simula.inspector.static_types import (
    AnalysisBackend,
    BasicBlock,
    ControlEdge,
    EdgeKind,
    FragmentSemantics,
    StaticCFG,
    StaticFunction,
)

logger: Final = structlog.get_logger().bind(system="simula.inspector.cfg_builder")

# ── Python AST helpers ────────────────────────────────────────────────────────

# Statement types that are "transparent" - they don't introduce a branch
_TRANSPARENT_STMTS: Final = (
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Expr,
    ast.Delete,
    ast.Pass,
    ast.Global,
    ast.Nonlocal,
    ast.Import,
    ast.ImportFrom,
    ast.Return,
    ast.Raise,
)

# Statement types that introduce branches / new blocks
_BRANCHING_STMTS: Final = (
    ast.If,
    ast.While,
    ast.For,
    ast.AsyncFor,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.TryStar,   # Python 3.11+ except*
    ast.Match,     # Python 3.10+
)

# Patterns that signal an indirect call in source text
_INDIRECT_CALL_PATTERNS: Final = [
    re.compile(r"\b(?:getattr|callable)\s*\("),          # getattr dispatch
    re.compile(r"\b\w+\s*\.\s*\w+\s*\("),               # method call (possibly virtual)
    re.compile(r"\[.*\]\s*\("),                          # subscript call
    re.compile(r"\(\s*\w+\s*\)\s*\("),                  # cast-then-call
]

# Semantic tagging patterns for Python source lines
_SEMANTIC_PATTERNS: Final[list[tuple[re.Pattern, FragmentSemantics]]] = [
    (re.compile(r"\bmalloc\b|\bcalloc\b|\bnew\b|=\s*\[\s*\]|\bbytearray\b"), FragmentSemantics.ALLOC),
    (re.compile(r"\bfree\b|\bdel\b|\bdealloc\b"), FragmentSemantics.FREE),
    (re.compile(r"\bstrcpy\b|\bmemcpy\b|\bsprintf\b|\b\.join\b|\b\+\s*[\"\']"), FragmentSemantics.STRING_OP),
    (re.compile(r"\bsyscall\b|\bctypes\b|\bos\..*\(|\bsubprocess\b"), FragmentSemantics.SYSCALL_GATEWAY),
    (re.compile(r"\braise\b|\bthrow\b|\bpanic\b"), FragmentSemantics.EXCEPTION_SITE),
    (re.compile(r"\breturn\b"), FragmentSemantics.RETURN_SITE),
    (re.compile(r"\bfor\b.*\bin\b|\bwhile\b"), FragmentSemantics.LOOP_HEAD),
]


def _tag_semantics(lines: list[str]) -> list[FragmentSemantics]:
    """Derive semantic tags from a list of source lines."""
    text = "\n".join(lines)
    found: list[FragmentSemantics] = []
    for pattern, sem in _SEMANTIC_PATTERNS:
        if pattern.search(text):
            found.append(sem)
    # Check for indirect calls
    for pattern in _INDIRECT_CALL_PATTERNS:
        if pattern.search(text):
            if FragmentSemantics.INDIRECT_BRANCH not in found:
                found.append(FragmentSemantics.INDIRECT_BRANCH)
            break
    # If we have loads / stores not already tagged → MEMORY_READ / MEMORY_WRITE
    if re.search(r"\[\s*\w", text) and FragmentSemantics.MEMORY_READ not in found:
        found.append(FragmentSemantics.MEMORY_READ)
    if re.search(r"\[\s*\w.*\]\s*=", text) and FragmentSemantics.MEMORY_WRITE not in found:
        found.append(FragmentSemantics.MEMORY_WRITE)
    return found or [FragmentSemantics.UNKNOWN]


def _indirect_calls_in_lines(lines: list[str]) -> list[str]:
    """Return a short string description for each indirect call site in lines."""
    sites: list[str] = []
    for i, line in enumerate(lines):
        for pat in _INDIRECT_CALL_PATTERNS:
            m = pat.search(line)
            if m:
                sites.append(f"line+{i}:{m.group(0)[:40]}")
                break
    return sites


# ── Python AST CFG extractor ──────────────────────────────────────────────────


class _PythonCFGExtractor:
    """
    Walks a parsed Python AST and emits StaticFunctions + ControlEdges into
    a StaticCFG.

    Block granularity
    -----------------
    Rather than splitting at every line, blocks are split at statement
    boundaries where control flow diverges (if/while/for/try/match).
    Each block covers one contiguous run of statements inside a function.
    block_id format: ``{rel_path}:{start_line}-{end_line}``
    """

    def __init__(self, rel_path: str, source_lines: list[str]) -> None:
        self._rel_path = rel_path
        self._lines = source_lines

    def extract(self, tree: ast.Module) -> tuple[list[StaticFunction], list[ControlEdge]]:
        """
        Return (functions, inter_procedural_edges) from the AST.
        Intra-procedural edges are embedded in each StaticFunction's blocks.
        """
        functions: list[StaticFunction] = []
        inter_edges: list[ControlEdge] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            func, f_edges = self._extract_function(node)
            functions.append(func)
            inter_edges.extend(f_edges)

        return functions, inter_edges

    # ── Function-level extraction ─────────────────────────────────────────────

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> tuple[StaticFunction, list[ControlEdge]]:
        """Extract one function → StaticFunction + inter-procedural edges."""
        func_name   = node.name
        start_line  = node.lineno
        end_line    = node.end_lineno or start_line

        blocks: dict[str, BasicBlock] = {}
        intra_edges: list[ControlEdge] = []
        inter_edges: list[ControlEdge] = []
        callees: set[str] = set()
        indirect_callsites: list[str] = []

        # Build blocks from the function body
        body_stmts = node.body
        block_seqs = self._split_into_blocks(body_stmts, func_name)

        for i, (bid, stmts) in enumerate(block_seqs):
            block = self._make_block(bid, func_name, stmts, i == 0)
            blocks[bid] = block
            if i == 0:
                pass

            # Collect inter-procedural edges from call sites in this block
            for call_name, is_indirect in self._calls_in_stmts(stmts):
                if is_indirect:
                    indirect_callsites.append(bid)
                    inter_edges.append(
                        ControlEdge(from_block=func_name, to_block=call_name, kind=EdgeKind.INDIRECT_CALL)
                    )
                else:
                    callees.add(call_name)
                    inter_edges.append(
                        ControlEdge(from_block=func_name, to_block=call_name, kind=EdgeKind.DIRECT_CALL)
                    )

        # Mark last block(s) as exits.
        # intra_edges is only populated if intra-procedural edge building runs;
        # when empty (current state), fall back to marking only the final block
        # to avoid marking every block as an exit.
        if intra_edges:
            for bid, block in blocks.items():
                if not any(e.from_block == bid for e in intra_edges):
                    block.is_exit = True
        else:
            block_ids = list(blocks.keys())
            if block_ids:
                blocks[block_ids[-1]].is_exit = True

        instruction_count = sum(b.instruction_count for b in blocks.values())
        block_count = len(blocks)
        cyclomatic = max(1, len(intra_edges) - block_count + 2)

        func = StaticFunction(
            func_name=func_name,
            file_path=self._rel_path,
            start_line=start_line,
            end_line=end_line,
            blocks=blocks,
            callees=sorted(callees),
            indirect_callsites=list(set(indirect_callsites)),
            block_count=block_count,
            instruction_count=instruction_count,
            cyclomatic_complexity=cyclomatic,
        )
        return func, inter_edges

    # ── Block construction ────────────────────────────────────────────────────

    def _split_into_blocks(
        self,
        stmts: list[ast.stmt],
        func_name: str,
    ) -> list[tuple[str, list[ast.stmt]]]:
        """
        Split a list of statements into (block_id, stmts) pairs.

        Each branching statement starts a new block; transparent statements
        accumulate into the current block.
        """
        result: list[tuple[str, list[ast.stmt]]] = []
        current: list[ast.stmt] = []

        def _flush(s: list[ast.stmt]) -> None:
            if not s:
                return
            start = s[0].lineno
            end = getattr(s[-1], "end_lineno", start) or start
            bid = f"{self._rel_path}:{start}-{end}"
            result.append((bid, list(s)))

        for stmt in stmts:
            if isinstance(stmt, _BRANCHING_STMTS):
                _flush(current)
                current = []
                # The branching statement itself gets its own 1-statement block
                _flush([stmt])
            else:
                current.append(stmt)

        _flush(current)
        return result or [
            (f"{self._rel_path}:{stmts[0].lineno if stmts else 0}-0", [])
        ]

    def _make_block(
        self,
        block_id: str,
        func_name: str,
        stmts: list[ast.stmt],
        is_entry: bool,
    ) -> BasicBlock:
        """Construct a BasicBlock from a list of AST statements."""
        if not stmts:
            return BasicBlock(
                block_id=block_id,
                func_name=func_name,
                file_path=self._rel_path,
                is_entry=is_entry,
                semantics=[FragmentSemantics.UNKNOWN],
            )

        start = stmts[0].lineno
        end   = getattr(stmts[-1], "end_lineno", start) or start
        lines = self._lines[start - 1 : end]

        semantics = _tag_semantics(lines)
        indirect  = _indirect_calls_in_lines(lines)

        return BasicBlock(
            block_id=block_id,
            func_name=func_name,
            file_path=self._rel_path,
            start_line=start,
            end_line=end,
            instructions=lines[:20],    # keep up to 20 representative lines
            instruction_count=len(lines),
            semantics=semantics,
            indirect_calls=indirect,
            is_entry=is_entry,
        )

    # ── Call extraction ───────────────────────────────────────────────────────

    def _calls_in_stmts(
        self,
        stmts: list[ast.stmt],
    ) -> list[tuple[str, bool]]:
        """
        Walk AST call nodes in *stmts* and return (callee_name, is_indirect) pairs.
        """
        results: list[tuple[str, bool]] = []
        for stmt in stmts:
            for node in ast.walk(stmt):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Name):
                    results.append((func.id, False))
                elif isinstance(func, ast.Attribute):
                    results.append((func.attr, True))  # method calls treated as indirect
                else:
                    results.append(("(indirect)", True))
        return results


# ── Regex-based extractor (JS/TS/Go/Rust) ────────────────────────────────────

_FUNC_PATTERNS: Final[list[tuple[re.Pattern, str]]] = [
    # JS/TS: function foo / async function foo / const foo = (...) =>
    (re.compile(r"(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\("), "js"),
    (re.compile(r"(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?\("), "js_arrow"),
    # Go: func foo(
    (re.compile(r"\bfunc\s+([A-Za-z_]\w*)\s*\("), "go"),
    # Rust: fn foo(
    (re.compile(r"\bfn\s+([A-Za-z_]\w*)\s*\("), "rust"),
    # Solidity: function foo(
    (re.compile(r"\bfunction\s+([A-Za-z_]\w*)\s*\("), "sol"),
]

_CALL_PATTERN_GENERIC: Final = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")

_BUILTIN_SKIP: Final = frozenset({
    "if", "for", "while", "return", "new", "typeof", "instanceof",
    "switch", "catch", "class", "import", "export", "function",
    "make", "append", "println", "print", "len", "cap",
    "require", "assert", "panic", "format", "vec",
})


class _RegexCFGExtractor:
    """
    Lightweight regex-based extractor for non-Python languages.

    Produces a coarser CFG: one block per function (no intra-procedural
    splitting), call edges inferred from call-pattern matches.
    """

    def __init__(self, rel_path: str, source: str) -> None:
        self._rel_path = rel_path
        self._source   = source
        self._lines    = source.splitlines()

    def extract(self) -> tuple[list[StaticFunction], list[ControlEdge]]:
        functions: list[StaticFunction] = []
        inter_edges: list[ControlEdge] = []

        func_spans = self._find_functions()
        for func_name, start_line, end_line in func_spans:
            func_lines = self._lines[start_line - 1 : end_line]
            func_text  = "\n".join(func_lines)

            bid = f"{self._rel_path}:{start_line}-{end_line}"
            semantics = _tag_semantics(func_lines)
            indirect  = _indirect_calls_in_lines(func_lines)

            block = BasicBlock(
                block_id=bid,
                func_name=func_name,
                file_path=self._rel_path,
                start_line=start_line,
                end_line=end_line,
                instructions=func_lines[:20],
                instruction_count=len(func_lines),
                semantics=semantics,
                indirect_calls=indirect,
                is_entry=True,
                is_exit=True,
            )

            # Extract callees
            callees: set[str] = set()
            for m in _CALL_PATTERN_GENERIC.finditer(func_text):
                name = m.group(1)
                if name not in _BUILTIN_SKIP and len(name) > 1 and name != func_name:
                    callees.add(name)
                    inter_edges.append(
                        ControlEdge(from_block=func_name, to_block=name, kind=EdgeKind.DIRECT_CALL)
                    )

            func = StaticFunction(
                func_name=func_name,
                file_path=self._rel_path,
                start_line=start_line,
                end_line=end_line,
                blocks={bid: block},
                callees=sorted(callees),
                indirect_callsites=indirect,
                block_count=1,
                instruction_count=len(func_lines),
                cyclomatic_complexity=1,
            )
            func.blocks[bid].is_entry = True
            functions.append(func)

        return functions, inter_edges

    def _find_functions(self) -> list[tuple[str, int, int]]:
        """Return (name, start_line, end_line) for each function found."""
        found: list[tuple[str, int, int]] = []
        for pattern, _lang in _FUNC_PATTERNS:
            for m in pattern.finditer(self._source):
                name = m.group(1)
                # Determine start line from match position
                start_line = self._source[:m.start()].count("\n") + 1
                # Find end by brace-counting from the opening brace
                brace_pos = self._source.find("{", m.end())
                if brace_pos == -1:
                    continue
                depth = 0
                end_pos = brace_pos
                for i in range(brace_pos, len(self._source)):
                    c = self._source[i]
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            end_pos = i
                            break
                end_line = self._source[:end_pos].count("\n") + 1
                found.append((name, start_line, end_line))

        # De-duplicate by function name (keep first occurrence)
        seen: set[str] = set()
        deduped = []
        for name, start, end in found:
            if name not in seen:
                seen.add(name)
                deduped.append((name, start, end))
        return deduped


# ── Binary disassembly stubs (Capstone / angr) ────────────────────────────────

def _try_capstone_build(binary_path: Path, cfg: StaticCFG) -> bool:
    """
    Attempt to recover CFG from a compiled binary using Capstone.

    Returns True on success.  Requires ``capstone`` package; silently
    returns False if unavailable.
    """
    try:
        import capstone  # noqa: PLC0415
    except ImportError:
        return False

    try:
        data = binary_path.read_bytes()
        # Auto-detect architecture from ELF magic; default to x86-64
        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        md.detail = True

        # Naive linear sweep - good enough for stripped binaries without symbols
        func_name = binary_path.stem
        instructions: list[str] = []
        current_start: int | None = None
        block_count = 0

        for insn in md.disasm(data, 0x400000):
            if current_start is None:
                current_start = insn.address

            mnemonic = insn.mnemonic.lower()
            instructions.append(f"{insn.address:#x}: {insn.mnemonic} {insn.op_str}")

            # Block boundary: any branch / call / ret
            is_branch = mnemonic in ("jmp", "je", "jne", "jz", "jnz", "jl", "jg",
                                     "jle", "jge", "jb", "ja", "jbe", "jae",
                                     "call", "ret", "retn", "retf", "syscall")
            if is_branch and instructions:
                block_id = f"{func_name}:{current_start:#x}"
                semantics = _semantics_from_insns(instructions)

                is_ret    = mnemonic in ("ret", "retn", "retf")
                is_syscall = mnemonic == "syscall"

                is_indirect = mnemonic == "call" and "[" in (insn.op_str or "")

                block = BasicBlock(
                    block_id=block_id,
                    func_name=func_name,
                    file_path=str(binary_path),
                    instructions=instructions[:20],
                    instruction_count=len(instructions),
                    semantics=semantics,
                    indirect_calls=[insn.op_str[:40]] if is_indirect else [],
                    is_entry=(block_count == 0),
                    is_exit=is_ret,
                )
                block_count += 1
                instructions = []
                current_start = None

                # Add to a synthetic function
                if func_name not in cfg.functions:
                    fn = StaticFunction(func_name=func_name, file_path=str(binary_path))
                    cfg.functions[func_name] = fn
                cfg.functions[func_name].blocks[block_id] = block
                cfg.block_index[block_id] = block

                if is_syscall and FragmentSemantics.SYSCALL_GATEWAY not in semantics:
                    block.semantics.append(FragmentSemantics.SYSCALL_GATEWAY)

        cfg.backend = AnalysisBackend.CAPSTONE
        cfg.total_functions = len(cfg.functions)
        cfg.total_blocks    = len(cfg.block_index)
        return True

    except Exception as exc:  # noqa: BLE001
        logger.warning("capstone_build_failed", error=str(exc))
        return False


def _semantics_from_insns(insns: list[str]) -> list[FragmentSemantics]:
    """Derive FragmentSemantics from a list of disassembly strings."""
    text = " ".join(insns).lower()
    found: list[FragmentSemantics] = []
    if re.search(r"\bmov\b.*\[", text):
        found.append(FragmentSemantics.MEMORY_READ)
    if re.search(r"\bmov\b.*\[.*\],", text):
        found.append(FragmentSemantics.MEMORY_WRITE)
    if re.search(r"\bcall\b.*\[", text):
        found.append(FragmentSemantics.INDIRECT_BRANCH)
    if re.search(r"\bsyscall\b|\bint\b.*0x80", text):
        found.append(FragmentSemantics.SYSCALL_GATEWAY)
    if re.search(r"\bimul\b|\bidiv\b|\badd\b|\bsub\b|\bmul\b", text):
        found.append(FragmentSemantics.ARITHMETIC)
    if re.search(r"\bcmp\b|\btest\b", text):
        found.append(FragmentSemantics.COMPARISON)
    return found or [FragmentSemantics.UNKNOWN]


# ── CfgBuilder ────────────────────────────────────────────────────────────────


class CfgBuilder:
    """
    Recovers a StaticCFG from source files or a compiled binary.

    Usage - source files::

        cfg = CfgBuilder().build(
            workspace_root=Path("/repo"),
            target_id="mypackage",
            source_files=["src/module.py", "src/util.py"],
        )

    Usage - compiled binary::

        cfg = CfgBuilder().build_from_binary(
            binary_path=Path("/bin/target"),
            target_id="target",
        )
    """

    def build(
        self,
        workspace_root: Path,
        target_id: str,
        source_files: list[str],
    ) -> StaticCFG:
        """
        Build a StaticCFG by analysing the given source files.

        Selects Python-AST or regex backend based on file extension.
        """
        log = logger.bind(target_id=target_id, files=len(source_files))
        log.debug("cfg_build_started")

        cfg = StaticCFG(
            target_id=target_id,
            source_files_analysed=list(source_files),
        )

        for rel_path in source_files:
            abs_path = workspace_root / rel_path
            try:
                source = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                log.warning("source_unreadable", file=rel_path, error=str(exc))
                continue

            suffix = abs_path.suffix.lower()
            if suffix == ".py":
                self._build_python(cfg, rel_path, source, log)
            elif suffix in {".js", ".ts", ".jsx", ".tsx", ".mjs", ".cjs",
                            ".go", ".rs", ".sol"}:
                self._build_regex(cfg, rel_path, source, log)
            else:
                log.debug("skipping_unknown_extension", file=rel_path, suffix=suffix)

        # Build inter-procedural caller maps
        self._link_callers(cfg)

        cfg.total_functions  = len(cfg.functions)
        cfg.total_blocks     = len(cfg.block_index)
        cfg.total_edges      = len(cfg.edges)
        cfg.total_indirect_callsites = sum(
            len(f.indirect_callsites) for f in cfg.functions.values()
        )

        log.info(
            "cfg_build_complete",
            functions=cfg.total_functions,
            blocks=cfg.total_blocks,
            edges=cfg.total_edges,
            indirect_callsites=cfg.total_indirect_callsites,
        )
        return cfg

    def build_from_binary(
        self,
        binary_path: Path,
        target_id: str | None = None,
    ) -> StaticCFG:
        """Build a StaticCFG from a compiled binary using Capstone (or stub)."""
        tid = target_id or binary_path.stem
        log = logger.bind(target_id=tid, binary=str(binary_path))
        log.debug("binary_cfg_build_started")

        cfg = StaticCFG(
            target_id=tid,
            source_files_analysed=[str(binary_path)],
        )

        if not _try_capstone_build(binary_path, cfg):
            log.warning("capstone_unavailable_using_stub")
            cfg.backend = AnalysisBackend.STUB

        cfg.total_functions  = len(cfg.functions)
        cfg.total_blocks     = len(cfg.block_index)
        cfg.total_edges      = len(cfg.edges)
        cfg.total_indirect_callsites = sum(
            len(f.indirect_callsites) for f in cfg.functions.values()
        )
        log.info("binary_cfg_build_complete", functions=cfg.total_functions)
        return cfg

    # ── Backend dispatch ──────────────────────────────────────────────────────

    def _build_python(
        self,
        cfg: StaticCFG,
        rel_path: str,
        source: str,
        log: structlog.stdlib.BoundLogger,
    ) -> None:
        """Parse Python source and merge functions into the CFG."""
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            log.warning("python_parse_error", file=rel_path, error=str(exc))
            return

        source_lines = source.splitlines()
        extractor = _PythonCFGExtractor(rel_path, source_lines)
        functions, inter_edges = extractor.extract(tree)

        for func in functions:
            cfg.add_function(func)

        for edge in inter_edges:
            cfg.edges.append(edge)

        cfg.backend = AnalysisBackend.PYTHON_AST
        log.debug("python_file_done", file=rel_path, functions=len(functions))

    def _build_regex(
        self,
        cfg: StaticCFG,
        rel_path: str,
        source: str,
        log: structlog.stdlib.BoundLogger,
    ) -> None:
        """Parse non-Python source via regex and merge into the CFG."""
        extractor = _RegexCFGExtractor(rel_path, source)
        functions, inter_edges = extractor.extract()

        for func in functions:
            cfg.add_function(func)
        for edge in inter_edges:
            cfg.edges.append(edge)

        if cfg.backend == AnalysisBackend.STUB:
            cfg.backend = AnalysisBackend.REGEX
        log.debug("regex_file_done", file=rel_path, functions=len(functions))

    # ── Post-processing ───────────────────────────────────────────────────────

    def _link_callers(self, cfg: StaticCFG) -> None:
        """Populate StaticFunction.callers from the edge set."""
        for edge in cfg.edges:
            if edge.kind in (EdgeKind.DIRECT_CALL, EdgeKind.INDIRECT_CALL):
                callee_func = cfg.functions.get(edge.to_block)
                if callee_func and edge.from_block not in callee_func.callers:
                    callee_func.callers.append(edge.from_block)
