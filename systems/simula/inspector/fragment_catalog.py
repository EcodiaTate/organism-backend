"""
EcodiaOS — Inspector Phase 3: Fragment Catalog Builder

Extracts "reusable fragments" from a StaticCFG and builds a FragmentCatalog
indexed by FragmentSemantics.

What is a fragment?
-------------------
A fragment is a basic block or call-chain sequence that has been annotated with
a semantic category (MEMORY_WRITE, INDIRECT_BRANCH, SYSCALL_GATEWAY, etc.) and
enriched with reachability information from the CFG.  Fragments are the atomic
units the steerability model will ask "can I reach this from the fault site?"

Two extraction passes
---------------------
1. Block-level pass — one fragment per BasicBlock that has a non-UNKNOWN
   semantic tag.  Covers individual instructions / short sequences.

2. Call-chain pass — enumerate call-chain fragments: sequences of 2–5 calls
   that were observed in Phase 2 traces.  These represent multi-step execution
   paths that could be exploited for ROP/JOP-style chaining (in native code) or
   logic-level control hijacking in Python.

Enrichment
----------
After extraction, ``enrich_reachability()`` annotates each fragment with the
set of CFG block IDs reachable from its location (BFS depth 5).

Fault-adjacency annotation
--------------------------
``mark_fault_adjacent()`` accepts a Phase2Result and marks fragments whose
containing block appears in a failure/crash run's call sequence.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import structlog

from systems.simula.inspector.static_types import (
    CodeFragment,
    FragmentCatalog,
    FragmentSemantics,
    StaticCFG,
)

if TYPE_CHECKING:
    from systems.simula.inspector.runtime_types import Phase2Result

logger = structlog.get_logger().bind(system="simula.inspector.fragment_catalog")

# Maximum depth for fragment reachability BFS
_REACHABILITY_DEPTH: int = 5

# Maximum number of representative instructions stored per fragment
_MAX_REPR_INSTRUCTIONS: int = 10

# Minimum length of a call-chain fragment
_MIN_CHAIN_LENGTH: int = 2

# Maximum length of a call-chain fragment
_MAX_CHAIN_LENGTH: int = 5

# Semantic categories that qualify a block for fragment extraction
_QUALIFYING_SEMANTICS: frozenset[FragmentSemantics] = frozenset({
    FragmentSemantics.MEMORY_READ,
    FragmentSemantics.MEMORY_WRITE,
    FragmentSemantics.INDIRECT_BRANCH,
    FragmentSemantics.SYSCALL_GATEWAY,
    FragmentSemantics.ALLOC,
    FragmentSemantics.FREE,
    FragmentSemantics.STRING_OP,
    FragmentSemantics.EXCEPTION_SITE,
    FragmentSemantics.LOOP_HEAD,
})


# ── Pass 1: Block-level fragments ─────────────────────────────────────────────


def _extract_block_fragments(cfg: StaticCFG, catalog: FragmentCatalog) -> None:
    """
    Emit one CodeFragment per BasicBlock that carries a qualifying semantic tag.

    Primary semantics: the first qualifying tag in the block's semantics list.
    all_semantics: all tags on the block.
    """
    for block_id, block in cfg.block_index.items():
        qualifying = [s for s in block.semantics if s in _QUALIFYING_SEMANTICS]
        if not qualifying:
            continue

        primary = qualifying[0]
        repr_lines = block.instructions[:_MAX_REPR_INSTRUCTIONS]

        frag = CodeFragment(
            target_id=cfg.target_id,
            semantics=primary,
            all_semantics=list(block.semantics),
            block_id=block_id,
            func_name=block.func_name,
            file_path=block.file_path,
            start_line=block.start_line,
            end_line=block.end_line,
            representative_instructions=repr_lines,
            is_indirect_dispatch=(
                FragmentSemantics.INDIRECT_BRANCH in block.semantics
                or bool(block.indirect_calls)
            ),
        )
        catalog.add_fragment(frag)


# ── Pass 2: Call-chain fragments ──────────────────────────────────────────────


def _extract_call_chain_fragments(cfg: StaticCFG, catalog: FragmentCatalog) -> None:
    """
    Enumerate call-chain fragments of length 2–5 from the CFG edge set.

    Algorithm: DFS over DIRECT_CALL edges from every function.
    Each path of length [MIN, MAX] becomes a CALL_CHAIN fragment.
    """
    # Build adjacency: func_name → list[callee_func_name]
    call_adj: dict[str, list[str]] = defaultdict(list)
    for edge in cfg.edges:
        if edge.kind in ("direct_call", "indirect_call"):
            if edge.from_block in cfg.functions:
                call_adj[edge.from_block].append(edge.to_block)

    seen_chains: set[tuple[str, ...]] = set()

    def _dfs(current: str, path: list[str], visited: set[str]) -> None:
        if len(path) > _MAX_CHAIN_LENGTH:
            return
        if len(path) >= _MIN_CHAIN_LENGTH:
            chain_key = tuple(path)
            if chain_key not in seen_chains:
                seen_chains.add(chain_key)
                _emit_chain(list(path))

        for callee in call_adj.get(current, []):
            if callee not in visited and callee in cfg.functions:
                visited.add(callee)
                path.append(callee)
                _dfs(callee, path, visited)
                path.pop()
                visited.discard(callee)

    def _emit_chain(path: list[str]) -> None:
        pairs = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        # Representative instructions from the last function in the chain
        tail_func = cfg.functions.get(path[-1])
        repr_insns: list[str] = []
        if tail_func:
            entry = tail_func.entry_block
            if entry:
                repr_insns = entry.instructions[:_MAX_REPR_INSTRUCTIONS]

        start_func = cfg.functions.get(path[0])
        frag = CodeFragment(
            target_id=cfg.target_id,
            semantics=FragmentSemantics.CALL_CHAIN,
            all_semantics=[FragmentSemantics.CALL_CHAIN],
            func_name=path[0],
            file_path=start_func.file_path if start_func else "",
            start_line=start_func.start_line if start_func else None,
            call_chain=pairs,
            representative_instructions=repr_insns,
            is_indirect_dispatch=any("indirect" in p[1] for p in pairs),
        )
        catalog.add_fragment(frag)

    for func_name in cfg.functions:
        _dfs(func_name, [func_name], {func_name})


# ── Enrichment: reachability ──────────────────────────────────────────────────


def _enrich_reachability(cfg: StaticCFG, catalog: FragmentCatalog) -> None:
    """
    For each fragment, compute the set of CFG block IDs reachable within
    _REACHABILITY_DEPTH hops and store in ``reachable_block_ids``.

    This is used by TraceMapper to answer "what can be reached from here?"
    """
    for frag in catalog.fragments.values():
        if frag.block_id:
            reachable = cfg.reachable_from(frag.block_id, max_depth=_REACHABILITY_DEPTH)
            frag.reachable_block_ids = sorted(reachable)
        elif frag.call_chain:
            # For call-chain fragments, start from the last function's entry block
            tail_func_name = frag.call_chain[-1][1] if frag.call_chain else frag.func_name
            tail_func = cfg.functions.get(tail_func_name)
            if tail_func and tail_func.entry_block:
                reachable = cfg.reachable_from(
                    tail_func.entry_block.block_id,
                    max_depth=_REACHABILITY_DEPTH,
                )
                frag.reachable_block_ids = sorted(reachable)


# ── Enrichment: fault-adjacency ───────────────────────────────────────────────


def _mark_fault_adjacent(catalog: FragmentCatalog, phase2_result: Phase2Result) -> None:
    """
    Mark fragments whose block (or function) appears in a failure/crash run's
    call sequence as fault_adjacent=True.
    """
    from systems.simula.inspector.runtime_types import RunCategory

    # Collect all function names that appear in failure/crash traces
    fault_funcs: set[str] = set()
    for trace in phase2_result.dataset.traces.values():
        if trace.run_category in (RunCategory.FAILURE, RunCategory.CRASH):
            for caller, callee in trace.call_sequence:
                fault_funcs.add(caller)
                fault_funcs.add(callee)

    updated = 0
    for frag in catalog.fragments.values():
        if frag.func_name in fault_funcs:
            frag.is_fault_adjacent = True
            updated += 1

    catalog.fault_adjacent_count = updated


# ── FragmentCatalogBuilder ─────────────────────────────────────────────────────


class FragmentCatalogBuilder:
    """
    Builds a FragmentCatalog from a StaticCFG, optionally enriched with
    Phase 2 runtime data.

    Usage::

        builder = FragmentCatalogBuilder()
        catalog = builder.build(cfg)

        # With fault-adjacency from Phase 2:
        catalog = builder.build(cfg, phase2_result=result)
    """

    def build(
        self,
        cfg: StaticCFG,
        phase2_result: Phase2Result | None = None,
    ) -> FragmentCatalog:
        """
        Build and return a populated FragmentCatalog.

        Steps:
        1. Block-level fragment extraction
        2. Call-chain fragment extraction
        3. Reachability enrichment
        4. Fault-adjacency annotation (if phase2_result provided)
        """
        log = logger.bind(target_id=cfg.target_id)
        log.debug("fragment_catalog_build_started", functions=cfg.total_functions)

        catalog = FragmentCatalog(target_id=cfg.target_id)

        # Pass 1: block-level fragments
        _extract_block_fragments(cfg, catalog)
        log.debug("block_fragments_extracted", count=catalog.total_fragments)

        # Pass 2: call-chain fragments
        before = catalog.total_fragments
        _extract_call_chain_fragments(cfg, catalog)
        log.debug(
            "chain_fragments_extracted",
            count=catalog.total_fragments - before,
        )

        # Enrichment: reachability
        _enrich_reachability(cfg, catalog)

        # Enrichment: fault-adjacency
        if phase2_result is not None:
            _mark_fault_adjacent(catalog, phase2_result)

        log.info(
            "fragment_catalog_built",
            total=catalog.total_fragments,
            indirect_dispatch=catalog.indirect_dispatch_count,
            fault_adjacent=catalog.fault_adjacent_count,
            semantics_keys=list(catalog.semantics_index.keys()),
        )
        return catalog

    def enrich_with_taint(
        self,
        catalog: FragmentCatalog,
        taint_reachable_funcs: set[str],
    ) -> None:
        """
        Mark fragments as taint_reachable when their function appears in
        *taint_reachable_funcs* (populated by TaintFlowLinker cross-reference).
        """
        for frag in catalog.fragments.values():
            if frag.func_name in taint_reachable_funcs:
                frag.taint_reachable = True

    def fragment_summary(self, catalog: FragmentCatalog) -> list[dict]:
        """
        Return a compact summary table (one row per semantics category).
        Useful for logging and reporting.
        """
        rows = []
        for sem_key, frag_ids in catalog.semantics_index.items():
            fault_adj = sum(
                1 for fid in frag_ids
                if catalog.fragments.get(fid) and catalog.fragments[fid].is_fault_adjacent
            )
            taint_reach = sum(
                1 for fid in frag_ids
                if catalog.fragments.get(fid) and catalog.fragments[fid].taint_reachable
            )
            rows.append({
                "semantics": sem_key,
                "count": len(frag_ids),
                "fault_adjacent": fault_adj,
                "taint_reachable": taint_reach,
            })
        rows.sort(key=lambda r: r["count"], reverse=True)
        return rows
