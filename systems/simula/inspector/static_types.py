"""
EcodiaOS — Inspector Phase 3: Binary / Static Analysis Types

All domain models for the static side of the inspector pipeline:
  - CFG recovery (function graph + basic blocks)
  - Indirect callsite / jump-table inventory
  - Reusable fragment catalog (instruction sequences / call chains)
  - Per-target execution atlas
  - Trace↔static mapping result

Design philosophy
-----------------
Phase 3 operates at the *static understanding* layer.  Given source files or a
compiled target it builds a structural map of what the program *could* do,
then connects that map to the Phase 2 runtime observations (what it *did* do).

The vocabulary deliberately mirrors Phase 2 types so the bridge in
TraceStaticMapping is mechanical: a runtime func_name resolves to a
StaticFunction; a runtime bb_id resolves to a BasicBlock.

Layer map
---------
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Source files / compiled binary                                          │
  │    ↓  CfgBuilder  (AST walk or binary disassembly)                       │
  │  StaticCFG  — nodes (BasicBlock) + edges (ControlEdge)                  │
  │    ↓                                                                     │
  │  FragmentCatalog  — instruction-sequence / call-chain fragments          │
  │  indexed by FragmentSemantics                                            │
  │    ↓                                                                     │
  │  TraceMapper  — runtime ControlFlowTrace → TraceStaticMapping            │
  │    ↓                                                                     │
  │  ExecutionAtlas  — complete per-target picture                           │
  │  (CFG + hot paths + failure-adjacent regions + fragment catalog)         │
  └──────────────────────────────────────────────────────────────────────────┘

Exit criterion
--------------
Given a runtime trace segment (list of func_names or bb_ids), you can locate
it in the static CFG and enumerate all reachable fragments from that position.
This is expressed by ExecutionAtlas.can_locate_and_enumerate() returning True.
"""

from __future__ import annotations

import enum
from datetime import datetime

from pydantic import Field

from primitives.common import EOSBaseModel, new_id, utc_now

# ── Enumerations ──────────────────────────────────────────────────────────────


class EdgeKind(enum.StrEnum):
    """The structural type of a CFG control-flow edge."""

    DIRECT_CALL   = "direct_call"      # foo() — statically known callee
    INDIRECT_CALL = "indirect_call"    # (*fp)() / virtual dispatch / dyn
    RETURN        = "return"           # function return arc
    UNCONDITIONAL = "unconditional"    # jmp / goto
    CONDITIONAL_T = "conditional_true"  # branch taken
    CONDITIONAL_F = "conditional_false" # branch not taken
    EXCEPTION     = "exception"        # unwind / raise arc
    JUMP_TABLE    = "jump_table"       # switch dispatch (multiple targets)


class FragmentSemantics(enum.StrEnum):
    """
    High-level semantic category of an instruction / call-chain fragment.

    These categories drive the fragment catalog index — a consumer querying
    "what memory-write fragments are reachable from this BB?" will search on
    MEMORY_WRITE.
    """

    MEMORY_READ    = "memory_read"     # dereference / load / array index
    MEMORY_WRITE   = "memory_write"    # store / write / assignment through ptr
    INDIRECT_BRANCH = "indirect_branch" # call-via-pointer / vtable dispatch
    SYSCALL_GATEWAY = "syscall_gateway" # wraps a syscall / libc entry point
    ALLOC          = "alloc"           # malloc / calloc / new / operator new
    FREE           = "free"            # free / delete / dealloc
    STRING_OP      = "string_op"       # strcpy / memcpy / sprintf / str concat
    ARITHMETIC     = "arithmetic"      # numeric operation that could overflow
    COMPARISON     = "comparison"      # predicate / conditional expression
    LOOP_HEAD      = "loop_head"       # loop entry / back-edge target
    EXCEPTION_SITE = "exception_site"  # explicit raise / throw / panic
    RETURN_SITE    = "return_site"     # function exit (ret instruction)
    CALL_CHAIN     = "call_chain"      # sequence of calls forming a path
    UNKNOWN        = "unknown"         # unclassified


class AnalysisBackend(enum.StrEnum):
    """Which analysis backend produced a CFG."""

    PYTHON_AST   = "python_ast"        # Python ast module
    REGEX        = "regex"             # Language-agnostic regex heuristics
    CAPSTONE     = "capstone"          # Binary disassembly via Capstone
    ANGR         = "angr"              # angr binary analysis framework
    STUB         = "stub"              # Placeholder / test stub


# ── Static structure nodes ────────────────────────────────────────────────────


class BasicBlock(EOSBaseModel):
    """
    A maximal straight-line sequence of instructions with a single entry and
    single exit.

    The ``block_id`` mirrors the ``bb_id`` format used in Phase 2
    (``file:start_line-end_line``) so that runtime BB events can be mapped
    without transformation.
    """

    block_id: str = Field(
        ...,
        description="Stable identifier: 'file:start_line-end_line' or 'func:offset'",
    )
    func_name: str = Field(default="", description="Enclosing function name")
    file_path: str = Field(default="", description="Source / binary file (relative to workspace root)")
    start_line: int | None = None
    end_line:   int | None = None

    # Instruction-level content (optional — available when backend provides it)
    instructions: list[str] = Field(
        default_factory=list,
        description="Disassembled / source lines in this block",
    )
    instruction_count: int = 0

    # Semantic tags inferred by the fragment extractor
    semantics: list[FragmentSemantics] = Field(
        default_factory=list,
        description="Semantic categories present in this block",
    )

    # Indirect callsites within this block
    indirect_calls: list[str] = Field(
        default_factory=list,
        description="Names / offsets of indirect call instructions in this block",
    )

    # Jump-table targets (populated for switch/dispatch blocks)
    jump_table_targets: list[str] = Field(
        default_factory=list,
        description="Resolved block IDs reachable via jump-table dispatch",
    )

    # Dominance / reachability metadata
    is_entry:   bool = False   # entry block of its function
    is_exit:    bool = False   # exit block of its function
    is_handler: bool = False   # exception handler block


class ControlEdge(EOSBaseModel):
    """A directed edge in the CFG between two basic blocks or functions."""

    from_block: str = Field(..., description="Source block_id or func_name")
    to_block:   str = Field(..., description="Target block_id or func_name")
    kind:       EdgeKind


class StaticFunction(EOSBaseModel):
    """
    Static representation of a single function in the target.

    Groups all basic blocks belonging to the function and records
    caller/callee relationships.
    """

    func_name: str
    file_path: str = Field(default="")
    start_line: int | None = None
    end_line:   int | None = None

    # All basic blocks within this function, keyed by block_id
    blocks: dict[str, BasicBlock] = Field(default_factory=dict)

    # Direct callers observed in the static call graph
    callers: list[str] = Field(default_factory=list)

    # Direct callees (statically-known targets)
    callees: list[str] = Field(default_factory=list)

    # Indirect callsites (call instructions with unknown target at static time)
    indirect_callsites: list[str] = Field(
        default_factory=list,
        description="Block IDs containing indirect call instructions",
    )

    # Size / complexity metrics
    block_count:       int = 0
    instruction_count: int = 0
    cyclomatic_complexity: int = 1   # edges − nodes + 2 for connected graph

    @property
    def entry_block(self) -> BasicBlock | None:
        return next((b for b in self.blocks.values() if b.is_entry), None)


# ── CFG graph ─────────────────────────────────────────────────────────────────


class StaticCFG(EOSBaseModel):
    """
    Complete static control-flow graph for one analysis target.

    Contains all functions, all basic blocks, and all inter-procedural
    edges.  This is the primary artefact produced by CfgBuilder.
    """

    id: str = Field(default_factory=new_id)
    target_id: str

    # All functions keyed by func_name
    functions: dict[str, StaticFunction] = Field(default_factory=dict)

    # All CFG edges (intra- and inter-procedural)
    edges: list[ControlEdge] = Field(default_factory=list)

    # Flat index: block_id → BasicBlock (populated from all functions)
    block_index: dict[str, BasicBlock] = Field(default_factory=dict)

    # Analysis metadata
    backend: AnalysisBackend = AnalysisBackend.STUB
    source_files_analysed: list[str] = Field(default_factory=list)
    total_functions:   int = 0
    total_blocks:      int = 0
    total_edges:       int = 0
    total_indirect_callsites: int = 0
    analysed_at: datetime = Field(default_factory=utc_now)

    def add_function(self, func: StaticFunction) -> None:
        """Register a function and index its blocks."""
        self.functions[func.func_name] = func
        for block_id, block in func.blocks.items():
            self.block_index[block_id] = block
        self.total_functions = len(self.functions)
        self.total_blocks    = len(self.block_index)

    def successors(self, block_id: str) -> list[str]:
        """Return block_ids of all successor blocks in the CFG."""
        return [e.to_block for e in self.edges if e.from_block == block_id]

    def predecessors(self, block_id: str) -> list[str]:
        """Return block_ids of all predecessor blocks in the CFG."""
        return [e.from_block for e in self.edges if e.to_block == block_id]

    def reachable_from(self, start_id: str, max_depth: int = 10) -> set[str]:
        """
        BFS/DFS over CFG edges from *start_id*, up to *max_depth* hops.
        Returns the set of reachable block/function IDs (inclusive of start).
        """
        visited: set[str] = set()
        frontier = [start_id]
        depth = 0
        while frontier and depth < max_depth:
            next_frontier: list[str] = []
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                next_frontier.extend(self.successors(node))
            frontier = next_frontier
            depth += 1
        return visited


# ── Fragment catalog ──────────────────────────────────────────────────────────


class CodeFragment(EOSBaseModel):
    """
    A reusable instruction sequence or call chain discoverable in the CFG.

    Fragments are the atomic units enumerated by FragmentCatalog.
    A single fragment may span multiple basic blocks when it represents
    a call chain.
    """

    fragment_id: str = Field(default_factory=new_id)
    target_id: str

    # Primary semantic label
    semantics: FragmentSemantics

    # All semantic tags present (may be multiple)
    all_semantics: list[FragmentSemantics] = Field(default_factory=list)

    # Location — for single-block fragments
    block_id:  str = Field(default="")
    func_name: str = Field(default="")
    file_path: str = Field(default="")
    start_line: int | None = None
    end_line:   int | None = None

    # For call-chain fragments: ordered list of (caller, callee) tuples
    call_chain: list[tuple[str, str]] = Field(default_factory=list)

    # Representative source / disassembly lines
    representative_instructions: list[str] = Field(
        default_factory=list,
        description="Up to 10 representative instructions from this fragment",
    )

    # Reachability: blocks reachable from this fragment within the CFG
    # (populated lazily by FragmentCatalog.enrich_reachability)
    reachable_block_ids: list[str] = Field(
        default_factory=list,
        description="CFG block IDs reachable from this fragment (up to depth 5)",
    )

    # Risk metadata
    is_indirect_dispatch: bool = Field(
        default=False,
        description="True if this fragment contains an indirect call/jump",
    )
    is_fault_adjacent: bool = Field(
        default=False,
        description="True if this fragment was observed near a fault in Phase 2 traces",
    )
    taint_reachable: bool = Field(
        default=False,
        description="True if taint flow analysis found a path from an entry point to this fragment",
    )


class FragmentCatalog(EOSBaseModel):
    """
    Complete catalog of reusable fragments for a target, indexed by semantics.

    This is the primary artefact produced by the fragment extraction pass.
    Consumers query it via ``fragments_by_semantics()`` or
    ``fragments_near_block()``.
    """

    id: str = Field(default_factory=new_id)
    target_id: str

    # All fragments keyed by fragment_id
    fragments: dict[str, CodeFragment] = Field(default_factory=dict)

    # Secondary index: FragmentSemantics value → list[fragment_id]
    semantics_index: dict[str, list[str]] = Field(default_factory=dict)

    # Secondary index: block_id → list[fragment_id] (fragments in/touching that block)
    block_index: dict[str, list[str]] = Field(default_factory=dict)

    total_fragments:          int = 0
    indirect_dispatch_count:  int = 0
    fault_adjacent_count:     int = 0
    built_at: datetime = Field(default_factory=utc_now)

    def add_fragment(self, frag: CodeFragment) -> None:
        """Register a fragment and update all secondary indexes."""
        self.fragments[frag.fragment_id] = frag

        # Semantics index
        for sem in frag.all_semantics:
            key = sem.value
            if key not in self.semantics_index:
                self.semantics_index[key] = []
            self.semantics_index[key].append(frag.fragment_id)

        # Block index
        if frag.block_id:
            if frag.block_id not in self.block_index:
                self.block_index[frag.block_id] = []
            self.block_index[frag.block_id].append(frag.fragment_id)

        self.total_fragments = len(self.fragments)
        if frag.is_indirect_dispatch:
            self.indirect_dispatch_count += 1
        if frag.is_fault_adjacent:
            self.fault_adjacent_count += 1

    def fragments_by_semantics(self, semantics: FragmentSemantics) -> list[CodeFragment]:
        """Return all fragments with the given primary or secondary semantic tag."""
        ids = self.semantics_index.get(semantics.value, [])
        return [self.fragments[fid] for fid in ids if fid in self.fragments]

    def fragments_near_block(self, block_id: str) -> list[CodeFragment]:
        """Return all fragments located in or touching *block_id*."""
        ids = self.block_index.get(block_id, [])
        return [self.fragments[fid] for fid in ids if fid in self.fragments]


# ── Trace ↔ static mapping ────────────────────────────────────────────────────


class TracedBlockMapping(EOSBaseModel):
    """
    Maps a single observed runtime bb_id to a static BasicBlock.

    When the runtime BB id matches a known static block_id exactly, the
    mapping is ``exact=True``.  When the tracer emits a func:line pair
    that falls within a block's range, ``exact=False`` and ``match_reason``
    describes the heuristic used.
    """

    runtime_bb_id: str
    static_block_id: str
    func_name: str
    file_path: str = ""
    exact: bool = True
    match_reason: str = ""

    # Fragments at this block
    fragment_ids: list[str] = Field(default_factory=list)


class TraceStaticMapping(EOSBaseModel):
    """
    The full mapping between one runtime ControlFlowTrace and the static CFG.

    Produced by TraceMapper for a single run.  Answers the question:
    "Given what happened at runtime, where is it in the static graph?"
    """

    id: str = Field(default_factory=new_id)
    run_id: str
    target_id: str

    # Per-BB mappings
    block_mappings: list[TracedBlockMapping] = Field(default_factory=list)

    # Functions from the trace that resolved to static functions
    resolved_functions: list[str] = Field(default_factory=list)

    # Functions from the trace NOT found in the static CFG
    # (dynamic dispatch, generated code, imports not analysed)
    unresolved_functions: list[str] = Field(default_factory=list)

    # Fragments reachable from each mapped block (flattened deduplicated list)
    reachable_fragment_ids: list[str] = Field(default_factory=list)

    # Summary stats
    total_runtime_blocks:    int = 0
    total_mapped_blocks:     int = 0
    total_unmapped_blocks:   int = 0
    mapping_coverage:        float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of runtime blocks successfully mapped to static CFG",
    )

    mapped_at: datetime = Field(default_factory=utc_now)


# ── Hot-path and failure-adjacent region annotations ─────────────────────────


class HotPath(EOSBaseModel):
    """
    A frequently-executed sequence of basic blocks, derived from runtime traces.

    "Hot" = executed in more than ``hot_threshold`` fraction of normal runs.
    """

    path_id: str = Field(default_factory=new_id)
    target_id: str

    # Ordered block IDs in this path
    block_ids: list[str] = Field(default_factory=list)

    # Functions crossed
    functions: list[str] = Field(default_factory=list)

    # Frequency: fraction of normal runs that exercised ALL blocks in this path
    coverage_fraction: float = Field(default=0.0, ge=0.0, le=1.0)

    # Whether this path was also present in failure runs
    appears_in_failure: bool = False


class FailureAdjacentRegion(EOSBaseModel):
    """
    A subgraph of the CFG that appeared in failure/crash runs but not (or rarely)
    in normal runs.

    These regions are the primary interest of Phase 3 — they represent code
    that is only exercised when something goes wrong, and are therefore
    candidates for steerability analysis.
    """

    region_id: str = Field(default_factory=new_id)
    target_id: str

    # Block IDs in this region
    block_ids: list[str] = Field(default_factory=list)
    functions: list[str] = Field(default_factory=list)

    # Number of failure/crash runs that touched this region
    failure_run_count: int = 0

    # Number of normal runs that touched this region (ideally 0)
    normal_run_count: int = 0

    # Fraction of failure runs that visited this region
    failure_coverage: float = Field(default=0.0, ge=0.0, le=1.0)

    # Fragments found within this region
    fragment_ids: list[str] = Field(default_factory=list)

    # Was a Phase 2 fault observed in this region?
    contains_fault_site: bool = False


# ── Execution Atlas ───────────────────────────────────────────────────────────


class ExecutionAtlas(EOSBaseModel):
    """
    The complete Phase 3 artefact for one analysis target.

    Combines:
    - The static CFG
    - The fragment catalog
    - Hot-path annotations (from normal runs)
    - Failure-adjacent region annotations (from failure/crash runs)
    - A summary of trace↔static mapping coverage

    The Phase 3 exit criterion is met when:
      - At least one runtime trace segment has been mapped into the CFG, AND
      - At least one reachable fragment has been enumerated from that location.

    This is checked by ``can_locate_and_enumerate()``.
    """

    id: str = Field(default_factory=new_id)
    target_id: str
    description: str = Field(default="")

    # Core artefacts
    cfg: StaticCFG
    catalog: FragmentCatalog

    # Annotations derived from Phase 2 traces
    hot_paths: list[HotPath] = Field(default_factory=list)
    failure_adjacent_regions: list[FailureAdjacentRegion] = Field(default_factory=list)

    # Trace↔static mapping results (one per run)
    trace_mappings: list[TraceStaticMapping] = Field(default_factory=list)

    # Aggregate coverage
    mean_mapping_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Mean trace→CFG mapping coverage across all mapped runs",
    )

    # Phase 3 exit criterion
    exit_criterion_met: bool = Field(
        default=False,
        description=(
            "True when at least one runtime trace segment has been located in the "
            "static CFG AND at least one reachable fragment has been enumerated."
        ),
    )

    built_at: datetime = Field(default_factory=utc_now)

    def can_locate_and_enumerate(self) -> bool:
        """
        Check the Phase 3 exit criterion:
        ≥1 run has mapped blocks AND those blocks have reachable fragments.
        """
        for mapping in self.trace_mappings:
            if mapping.total_mapped_blocks > 0 and mapping.reachable_fragment_ids:
                return True
        return False

    def fragments_reachable_from_trace(
        self,
        func_names: list[str],
        bb_ids: list[str] | None = None,
    ) -> list[CodeFragment]:
        """
        Given a runtime trace segment (list of func_names and optional bb_ids),
        return all statically reachable fragments.

        Algorithm:
        1. Resolve each func_name to a StaticFunction → gather all block_ids.
        2. Resolve each bb_id directly via block_index.
        3. Run CFG reachability from each resolved block (up to depth 5).
        4. Look up fragments for every reachable block in the catalog.
        """
        start_blocks: set[str] = set()

        for fn in func_names:
            static_fn = self.cfg.functions.get(fn)
            if static_fn:
                start_blocks.update(static_fn.blocks.keys())

        for bid in (bb_ids or []):
            if bid in self.cfg.block_index:
                start_blocks.add(bid)

        all_reachable: set[str] = set()
        for block_id in start_blocks:
            all_reachable.update(self.cfg.reachable_from(block_id, max_depth=5))

        fragment_ids: set[str] = set()
        for bid in all_reachable:
            for frag_id in self.catalog.block_index.get(bid, []):
                fragment_ids.add(frag_id)

        return [
            self.catalog.fragments[fid]
            for fid in fragment_ids
            if fid in self.catalog.fragments
        ]


# ── Phase 3 result container ──────────────────────────────────────────────────


class Phase3Result(EOSBaseModel):
    """
    Top-level output of a Phase 3 static analysis + trace mapping session.

    Wraps the ExecutionAtlas with aggregate statistics suitable for
    logging and reporting.
    """

    id: str = Field(default_factory=new_id)
    target_id: str

    atlas: ExecutionAtlas

    # Aggregate statistics
    total_functions_analysed:      int = 0
    total_blocks_analysed:         int = 0
    total_fragments_catalogued:    int = 0
    total_indirect_callsites:      int = 0
    total_runs_mapped:             int = 0
    total_hot_paths:               int = 0
    total_failure_adjacent_regions: int = 0

    # Exit criterion
    exit_criterion_met: bool = Field(
        default=False,
        description=(
            "Phase 3 exit criterion: given a runtime trace segment, we can "
            "locate it in the static graph and enumerate nearby reachable fragments."
        ),
    )

    produced_at: datetime = Field(default_factory=utc_now)
