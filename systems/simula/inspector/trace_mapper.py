"""
EcodiaOS - Inspector Phase 3: Trace ↔ Static CFG Mapper

Bridges Phase 2 runtime observations (ControlFlowTrace) with the Phase 3
static CFG (StaticCFG + FragmentCatalog).

Given a runtime trace for a single run, TraceMapper:
1.  Resolves each observed function name → StaticFunction
2.  Resolves each observed bb_id → BasicBlock (exact or approximate)
3.  Enumerates fragments reachable from every mapped block
4.  Produces a TraceStaticMapping summarising coverage and reachable fragments

BB matching strategy
--------------------
The Phase 2 tracer encodes bb_ids as "file:start_line-end_line".
The static CFG uses the same format.  When an exact match exists the mapping
is ``exact=True``.

When no exact match exists (e.g. the tracer emitted a bare func:line pair),
a fallback scan checks whether the line number falls within any known block
range in that function - this is ``exact=False, match_reason="line_in_range"``.

Hot-path and failure-adjacent detection
----------------------------------------
``build_hot_paths()`` accepts the full TraceDataset and identifies sequences of
blocks visited in a large fraction of normal runs.

``build_failure_adjacent_regions()`` finds blocks that appear only in
failure/crash runs (or rarely in normal runs) - these are the steerability
candidates.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from typing import TYPE_CHECKING

import structlog

from systems.simula.inspector.static_types import (
    BasicBlock,
    FailureAdjacentRegion,
    FragmentCatalog,
    HotPath,
    StaticCFG,
    TracedBlockMapping,
    TraceStaticMapping,
)

if TYPE_CHECKING:
    from systems.simula.inspector.runtime_types import (
        ControlFlowTrace,
        TraceDataset,
    )

logger = structlog.get_logger().bind(system="simula.inspector.trace_mapper")

# A block visited in ≥HOT_THRESHOLD fraction of normal runs is "hot"
_HOT_THRESHOLD:      float = 0.5

# A block visited in ≤NORMAL_THRESHOLD fraction of normal runs (but in ≥1
# failure run) is "failure-adjacent"
_NORMAL_THRESHOLD:   float = 0.1

# Minimum failure runs to declare a region failure-adjacent
_MIN_FAILURE_VISITS: int   = 1


# ── Block resolution helpers ──────────────────────────────────────────────────


def _resolve_block(
    bb_id: str,
    cfg: StaticCFG,
) -> tuple[BasicBlock | None, bool, str]:
    """
    Attempt to resolve a runtime bb_id to a static BasicBlock.

    Returns (block_or_None, exact_match, reason_string).

    Exact: bb_id in cfg.block_index
    Approximate: try line-range lookup within the function implied by bb_id
    """
    # Exact match
    if bb_id in cfg.block_index:
        return cfg.block_index[bb_id], True, "exact"

    # Parse bb_id format "file:start-end" or "file:start_line-end_line"
    # or "func:offset" (binary)
    if ":" not in bb_id:
        return None, False, "unparseable"

    prefix, rest = bb_id.rsplit(":", 1)
    # Try to extract a line number from the rest part
    start_line: int | None = None
    with contextlib.suppress(ValueError):
        start_line = int(rest.split("-")[0]) if "-" in rest else int(rest)

    if start_line is None:
        return None, False, "no_line"

    # Search for a block whose file_path ends with the prefix and whose
    # line range contains start_line
    for block in cfg.block_index.values():
        if not block.file_path.endswith(prefix):
            continue
        if (
            block.start_line is not None
            and block.end_line is not None
            and block.start_line <= start_line <= block.end_line
        ):
            return block, False, "line_in_range"

    return None, False, "unresolved"


def _resolve_func(func_name: str, cfg: StaticCFG) -> bool:
    """True if func_name exists in the static CFG."""
    return func_name in cfg.functions


# ── TraceMapper ───────────────────────────────────────────────────────────────


class TraceMapper:
    """
    Maps runtime ControlFlowTraces to the static CFG and fragment catalog.

    Usage::

        mapper = TraceMapper()

        # Map a single run
        mapping = mapper.map_trace(trace, cfg, catalog)

        # Map all runs in a dataset
        mappings = mapper.map_dataset(dataset, cfg, catalog)

        # Build hot paths and failure-adjacent regions
        hot_paths = mapper.build_hot_paths(dataset, cfg, mappings)
        regions   = mapper.build_failure_adjacent_regions(dataset, cfg, mappings)
    """

    def map_trace(
        self,
        trace: ControlFlowTrace,
        cfg: StaticCFG,
        catalog: FragmentCatalog,
    ) -> TraceStaticMapping:
        """
        Map a single ControlFlowTrace to its static representation.

        Returns a TraceStaticMapping with per-block mappings and reachable
        fragment IDs.
        """
        block_mappings: list[TracedBlockMapping] = []
        resolved_funcs: list[str]  = []
        unresolved_funcs: list[str] = []

        # Track which blocks we've already mapped (avoid duplicates from repeated visits)
        seen_runtime_ids: set[str] = set()

        # --- Map BB observations ---
        bb_trace = trace.bb_trace
        if bb_trace:
            for runtime_bb_id in bb_trace.hits:
                if runtime_bb_id in seen_runtime_ids:
                    continue
                seen_runtime_ids.add(runtime_bb_id)

                block, exact, reason = _resolve_block(runtime_bb_id, cfg)
                if block is None:
                    continue

                frag_ids = catalog.block_index.get(block.block_id, [])
                tbm = TracedBlockMapping(
                    runtime_bb_id=runtime_bb_id,
                    static_block_id=block.block_id,
                    func_name=block.func_name,
                    file_path=block.file_path,
                    exact=exact,
                    match_reason=reason,
                    fragment_ids=list(frag_ids),
                )
                block_mappings.append(tbm)

        # --- Map function observations ---
        for func_name in trace.functions_visited:
            if _resolve_func(func_name, cfg):
                if func_name not in resolved_funcs:
                    resolved_funcs.append(func_name)
                # If no BB trace, synthesize block mappings from function entry block
                if not bb_trace:
                    static_fn = cfg.functions[func_name]
                    entry = static_fn.entry_block
                    if entry and entry.block_id not in seen_runtime_ids:
                        seen_runtime_ids.add(entry.block_id)
                        frag_ids = catalog.block_index.get(entry.block_id, [])
                        tbm = TracedBlockMapping(
                            runtime_bb_id=entry.block_id,  # use static id as proxy
                            static_block_id=entry.block_id,
                            func_name=func_name,
                            file_path=entry.file_path,
                            exact=False,
                            match_reason="function_entry_proxy",
                            fragment_ids=list(frag_ids),
                        )
                        block_mappings.append(tbm)
            else:
                if func_name not in unresolved_funcs:
                    unresolved_funcs.append(func_name)

        # --- Collect all reachable fragment IDs ---
        all_frag_ids: set[str] = set()
        for tbm in block_mappings:
            all_frag_ids.update(tbm.fragment_ids)
            # Also include fragments reachable from each mapped block
            block = cfg.block_index.get(tbm.static_block_id)
            if block:
                reachable = cfg.reachable_from(tbm.static_block_id, max_depth=3)
                for bid in reachable:
                    for fid in catalog.block_index.get(bid, []):
                        all_frag_ids.add(fid)

        total_runtime = len(seen_runtime_ids) + len(
            set(trace.functions_visited) - set(resolved_funcs)
        )
        total_mapped  = len(block_mappings)
        coverage = total_mapped / max(total_runtime, 1)

        return TraceStaticMapping(
            run_id=trace.run_id,
            target_id=cfg.target_id,
            block_mappings=block_mappings,
            resolved_functions=resolved_funcs,
            unresolved_functions=unresolved_funcs,
            reachable_fragment_ids=sorted(all_frag_ids),
            total_runtime_blocks=total_runtime,
            total_mapped_blocks=total_mapped,
            total_unmapped_blocks=max(0, total_runtime - total_mapped),
            mapping_coverage=round(coverage, 4),
        )

    def map_dataset(
        self,
        dataset: TraceDataset,
        cfg: StaticCFG,
        catalog: FragmentCatalog,
    ) -> list[TraceStaticMapping]:
        """Map all traces in a dataset. Returns one mapping per run."""
        mappings = []
        for trace in dataset.traces.values():
            mapping = self.map_trace(trace, cfg, catalog)
            mappings.append(mapping)

        logger.info(
            "dataset_mapped",
            target_id=cfg.target_id,
            runs=len(mappings),
            mean_coverage=round(
                sum(m.mapping_coverage for m in mappings) / max(len(mappings), 1), 3
            ),
        )
        return mappings

    def build_hot_paths(
        self,
        dataset: TraceDataset,
        cfg: StaticCFG,
        mappings: list[TraceStaticMapping],
    ) -> list[HotPath]:
        """
        Identify hot paths: block sequences visited in ≥HOT_THRESHOLD of
        normal runs.

        Algorithm:
        1. Count how many normal runs visit each static block.
        2. A block is "hot" if its coverage fraction ≥ HOT_THRESHOLD.
        3. Chains of consecutive hot blocks in the CFG become HotPath objects.
        """
        from systems.simula.inspector.runtime_types import RunCategory

        normal_run_count = dataset.normal_run_count or 1

        # Map run_id → mapping for efficient lookup
        run_to_mapping = {m.run_id: m for m in mappings}

        # Count per-block normal-run visits
        block_normal_visits: dict[str, int] = defaultdict(int)
        block_failure_visits: dict[str, int] = defaultdict(int)

        for trace in dataset.traces.values():
            mapping = run_to_mapping.get(trace.run_id)
            if not mapping:
                continue
            visited_static = {tbm.static_block_id for tbm in mapping.block_mappings}
            if trace.run_category == RunCategory.NORMAL:
                for bid in visited_static:
                    block_normal_visits[bid] += 1
            else:
                for bid in visited_static:
                    block_failure_visits[bid] += 1

        # Identify hot blocks
        hot_blocks: set[str] = {
            bid
            for bid, count in block_normal_visits.items()
            if count / normal_run_count >= _HOT_THRESHOLD
        }

        # Build hot paths: sequences of hot blocks connected in the CFG
        hot_paths: list[HotPath] = []
        visited_in_chain: set[str] = set()

        def _dfs_chain(start: str, path: list[str]) -> None:
            path.append(start)
            visited_in_chain.add(start)
            successors = [s for s in cfg.successors(start) if s in hot_blocks and s not in visited_in_chain]
            if not successors:
                if len(path) >= 2:
                    funcs = list({cfg.block_index[b].func_name for b in path if b in cfg.block_index})
                    coverage = min(
                        block_normal_visits[b] / normal_run_count for b in path
                    )
                    appears_in_failure = any(
                        block_failure_visits.get(b, 0) > 0 for b in path
                    )
                    hot_paths.append(HotPath(
                        target_id=cfg.target_id,
                        block_ids=list(path),
                        functions=funcs,
                        coverage_fraction=round(coverage, 4),
                        appears_in_failure=appears_in_failure,
                    ))
            else:
                for s in successors:
                    _dfs_chain(s, path)
                    path.pop()
                    visited_in_chain.discard(s)

        for bid in sorted(hot_blocks):
            if bid not in visited_in_chain:
                preds = cfg.predecessors(bid)
                if not any(p in hot_blocks for p in preds):
                    # It's a hot-path entry point
                    _dfs_chain(bid, [])
                    if hot_paths and hot_paths[-1].block_ids and hot_paths[-1].block_ids[0] == bid:
                        pass  # already added

        logger.info("hot_paths_built", count=len(hot_paths), target_id=cfg.target_id)
        return hot_paths

    def build_failure_adjacent_regions(
        self,
        dataset: TraceDataset,
        cfg: StaticCFG,
        mappings: list[TraceStaticMapping],
        catalog: FragmentCatalog,
    ) -> list[FailureAdjacentRegion]:
        """
        Identify failure-adjacent regions: subgraphs of blocks that appear in
        failure/crash runs but rarely (or never) in normal runs.
        """
        from systems.simula.inspector.runtime_types import RunCategory

        normal_total  = max(dataset.normal_run_count, 1)
        failure_total = max(dataset.failure_run_count + dataset.crash_run_count, 1)

        run_to_mapping = {m.run_id: m for m in mappings}

        block_normal_visits:  dict[str, int] = defaultdict(int)
        block_failure_visits: dict[str, int] = defaultdict(int)

        for trace in dataset.traces.values():
            mapping = run_to_mapping.get(trace.run_id)
            if not mapping:
                continue
            visited = {tbm.static_block_id for tbm in mapping.block_mappings}
            if trace.run_category == RunCategory.NORMAL:
                for bid in visited:
                    block_normal_visits[bid] += 1
            else:
                for bid in visited:
                    block_failure_visits[bid] += 1

        # Identify failure-adjacent blocks
        fa_blocks: set[str] = {
            bid
            for bid, count in block_failure_visits.items()
            if count >= _MIN_FAILURE_VISITS
            and block_normal_visits.get(bid, 0) / normal_total <= _NORMAL_THRESHOLD
        }

        if not fa_blocks:
            return []

        # Group connected FA blocks into regions using union-find / BFS
        regions: list[FailureAdjacentRegion] = []
        visited_in_region: set[str] = set()

        for start in sorted(fa_blocks):
            if start in visited_in_region:
                continue

            # BFS within fa_blocks
            region_blocks: list[str] = []
            queue = [start]
            while queue:
                bid = queue.pop()
                if bid in visited_in_region:
                    continue
                visited_in_region.add(bid)
                region_blocks.append(bid)
                for neighbor in cfg.successors(bid) + cfg.predecessors(bid):
                    if neighbor in fa_blocks and neighbor not in visited_in_region:
                        queue.append(neighbor)

            region_funcs = list({
                cfg.block_index[b].func_name
                for b in region_blocks
                if b in cfg.block_index
            })

            # Collect fragment IDs in this region
            frag_ids: list[str] = []
            for bid in region_blocks:
                frag_ids.extend(catalog.block_index.get(bid, []))

            # Check if any fault was observed at a function in this region
            fault_funcs = {
                obs.fault_at_func
                for faults in dataset.faults.values()
                for obs in faults
            }
            contains_fault = bool(set(region_funcs) & fault_funcs)

            failure_count = max(block_failure_visits.get(b, 0) for b in region_blocks)
            normal_count  = max(block_normal_visits.get(b, 0) for b in region_blocks)
            failure_cov   = failure_count / failure_total

            regions.append(FailureAdjacentRegion(
                target_id=cfg.target_id,
                block_ids=region_blocks,
                functions=region_funcs,
                failure_run_count=failure_count,
                normal_run_count=normal_count,
                failure_coverage=round(failure_cov, 4),
                fragment_ids=list(set(frag_ids)),
                contains_fault_site=contains_fault,
            ))

        logger.info(
            "failure_adjacent_regions_built",
            count=len(regions),
            target_id=cfg.target_id,
        )
        return regions
