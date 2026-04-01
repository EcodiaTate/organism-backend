"""
EcodiaOS - Inspector Phase 3: Static Analyzer (Orchestrator)

Single-entry orchestrator for the Phase 3 pipeline:

  CfgBuilder → StaticCFG
  FragmentCatalogBuilder → FragmentCatalog
  TraceMapper → TraceStaticMappings + HotPaths + FailureAdjacentRegions
  → ExecutionAtlas + Phase3Result

Usage
-----
  # From source files only (no Phase 2 data):
  analyzer = StaticAnalyzer(workspace_root=Path("/repo"))
  result = analyzer.analyze(
      target_id="mypackage",
      source_files=["src/module.py", "src/util.py"],
  )

  # From source files + Phase 2 runtime data:
  result = analyzer.analyze(
      target_id="mypackage",
      source_files=["src/module.py"],
      phase2_result=phase2_result,
  )

  # From a compiled binary:
  result = analyzer.analyze_binary(
      binary_path=Path("/bin/target"),
      target_id="target",
      phase2_result=phase2_result,
  )

Exit criterion
--------------
Phase3Result.exit_criterion_met is True when:
  - At least one runtime trace segment (from Phase 2) has been mapped into the
    static CFG, AND
  - At least one reachable fragment has been enumerated from that location.

If no Phase 2 data is provided, the criterion is met if the CFG contains at
least one function AND the fragment catalog contains at least one fragment -
meaning the static understanding layer is operational.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from systems.simula.inspector.cfg_builder import CfgBuilder
from systems.simula.inspector.fragment_catalog import FragmentCatalogBuilder
from systems.simula.inspector.static_types import (
    ExecutionAtlas,
    Phase3Result,
    StaticCFG,
)
from systems.simula.inspector.trace_mapper import TraceMapper

if TYPE_CHECKING:
    from systems.simula.inspector.runtime_types import Phase2Result

logger = structlog.get_logger().bind(system="simula.inspector.static_analyzer")


class StaticAnalyzer:
    """
    Phase 3 orchestrator - builds an ExecutionAtlas for a target.

    The atlas combines:
    - Static CFG (what the program *could* do)
    - Fragment catalog (reusable instruction sequences / call chains)
    - Hot-path annotations (common execution paths from normal runs)
    - Failure-adjacent regions (code only exercised when things go wrong)
    - Trace→CFG mappings (connecting Phase 2 observations to static structure)

    Parameters
    ----------
    workspace_root:
        Root directory of the target workspace (for relative path resolution).
    """

    def __init__(self, workspace_root: Path | str) -> None:
        self._root = Path(workspace_root)
        self._cfg_builder     = CfgBuilder()
        self._catalog_builder = FragmentCatalogBuilder()
        self._mapper          = TraceMapper()
        self._log = logger

    # ── Source-file analysis ──────────────────────────────────────────────────

    def analyze(
        self,
        target_id: str,
        source_files: list[str],
        phase2_result: Phase2Result | None = None,
    ) -> Phase3Result:
        """
        Build a Phase3Result from source files.

        Args:
            target_id:      Identifier for the target (used throughout the atlas).
            source_files:   Workspace-relative paths to source files to analyse.
            phase2_result:  Optional Phase 2 output; enables fault-adjacency
                            annotation, hot-path detection, and trace mapping.

        Returns:
            Phase3Result with populated ExecutionAtlas.
        """
        log = self._log.bind(target_id=target_id, source_files=len(source_files))
        log.info("static_analysis_started")

        # 1. Build CFG
        cfg = self._cfg_builder.build(
            workspace_root=self._root,
            target_id=target_id,
            source_files=source_files,
        )

        return self._build_result(cfg, target_id, phase2_result, log)

    # ── Binary analysis ───────────────────────────────────────────────────────

    def analyze_binary(
        self,
        binary_path: Path | str,
        target_id: str | None = None,
        phase2_result: Phase2Result | None = None,
    ) -> Phase3Result:
        """
        Build a Phase3Result from a compiled binary.

        Args:
            binary_path:    Path to the ELF/PE/Mach-O binary.
            target_id:      Override name; defaults to binary stem.
            phase2_result:  Optional Phase 2 output.

        Returns:
            Phase3Result with populated ExecutionAtlas.
        """
        binary_path = Path(binary_path)
        tid = target_id or binary_path.stem
        log = self._log.bind(target_id=tid, binary=str(binary_path))
        log.info("binary_analysis_started")

        cfg = self._cfg_builder.build_from_binary(binary_path=binary_path, target_id=tid)
        return self._build_result(cfg, tid, phase2_result, log)

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def _build_result(
        self,
        cfg: StaticCFG,
        target_id: str,
        phase2_result: Phase2Result | None,
        log: structlog.stdlib.BoundLogger,
    ) -> Phase3Result:
        """
        Shared pipeline after CFG is available.

        Steps:
        2. Build fragment catalog (optionally enriched with fault-adjacency)
        3. Map traces → static (if Phase 2 data available)
        4. Build hot paths + failure-adjacent regions
        5. Assemble ExecutionAtlas
        6. Check exit criterion
        """

        # 2. Fragment catalog
        catalog = self._catalog_builder.build(
            cfg,
            phase2_result=phase2_result,
        )

        # 3–4. Trace mapping + hot paths + failure-adjacent regions
        trace_mappings = []
        hot_paths      = []
        fa_regions     = []
        mean_coverage  = 0.0

        if phase2_result is not None:
            dataset = phase2_result.dataset

            # Map all runs
            trace_mappings = self._mapper.map_dataset(dataset, cfg, catalog)
            if trace_mappings:
                mean_coverage = round(
                    sum(m.mapping_coverage for m in trace_mappings)
                    / len(trace_mappings),
                    4,
                )

            # Annotate with taint if available from Phase 2
            # (TaintFlowLinker feeds into catalog via separate call when available)

            # Hot paths
            hot_paths = self._mapper.build_hot_paths(dataset, cfg, trace_mappings)

            # Failure-adjacent regions
            fa_regions = self._mapper.build_failure_adjacent_regions(
                dataset, cfg, trace_mappings, catalog
            )

            log.info(
                "trace_mapping_complete",
                runs_mapped=len(trace_mappings),
                mean_coverage=mean_coverage,
                hot_paths=len(hot_paths),
                fa_regions=len(fa_regions),
            )

        # 5. Assemble atlas
        atlas = ExecutionAtlas(
            target_id=target_id,
            cfg=cfg,
            catalog=catalog,
            hot_paths=hot_paths,
            failure_adjacent_regions=fa_regions,
            trace_mappings=trace_mappings,
            mean_mapping_coverage=mean_coverage,
        )

        # 6. Exit criterion
        if trace_mappings:
            atlas.exit_criterion_met = atlas.can_locate_and_enumerate()
        else:
            # No Phase 2 data - criterion met if static understanding is operational
            atlas.exit_criterion_met = (
                cfg.total_functions > 0 and catalog.total_fragments > 0
            )

        # 7. Assemble Phase3Result
        result = Phase3Result(
            target_id=target_id,
            atlas=atlas,
            total_functions_analysed=cfg.total_functions,
            total_blocks_analysed=cfg.total_blocks,
            total_fragments_catalogued=catalog.total_fragments,
            total_indirect_callsites=cfg.total_indirect_callsites,
            total_runs_mapped=len(trace_mappings),
            total_hot_paths=len(hot_paths),
            total_failure_adjacent_regions=len(fa_regions),
            exit_criterion_met=atlas.exit_criterion_met,
        )

        log.info(
            "static_analysis_complete",
            functions=result.total_functions_analysed,
            blocks=result.total_blocks_analysed,
            fragments=result.total_fragments_catalogued,
            exit_criterion_met=result.exit_criterion_met,
        )

        return result

    # ── Query helpers ─────────────────────────────────────────────────────────

    def locate_and_enumerate(
        self,
        result: Phase3Result,
        func_names: list[str],
        bb_ids: list[str] | None = None,
    ) -> dict:
        """
        Given a runtime trace segment, locate it in the atlas and enumerate
        reachable fragments.

        Returns a dict::

            {
                "resolved_functions": [...],
                "mapped_blocks": [...],
                "reachable_fragments": [
                    {
                        "fragment_id": ...,
                        "semantics": ...,
                        "func_name": ...,
                        "file_path": ...,
                        "is_indirect_dispatch": ...,
                        "is_fault_adjacent": ...,
                    },
                    ...
                ],
            }

        This is the canonical answer to the Phase 3 exit criterion query.
        """
        atlas = result.atlas
        cfg   = atlas.cfg

        resolved_funcs = [f for f in func_names if f in cfg.functions]
        fragments = atlas.fragments_reachable_from_trace(func_names, bb_ids)

        mapped_blocks: list[str] = []
        for fn in resolved_funcs:
            static_fn = cfg.functions[fn]
            mapped_blocks.extend(static_fn.blocks.keys())
        for bid in (bb_ids or []):
            if bid in cfg.block_index:
                mapped_blocks.append(bid)

        return {
            "resolved_functions": resolved_funcs,
            "mapped_blocks": list(set(mapped_blocks)),
            "reachable_fragments": [
                {
                    "fragment_id":      f.fragment_id,
                    "semantics":        f.semantics.value,
                    "func_name":        f.func_name,
                    "file_path":        f.file_path,
                    "is_indirect_dispatch": f.is_indirect_dispatch,
                    "is_fault_adjacent":    f.is_fault_adjacent,
                    "taint_reachable":      f.taint_reachable,
                }
                for f in fragments
            ],
        }

    def atlas_summary(self, result: Phase3Result) -> dict:
        """
        Return a concise summary of the Phase3Result suitable for reporting.
        """
        catalog_summary = self._catalog_builder.fragment_summary(result.atlas.catalog)
        fa_summary = [
            {
                "region_id":      r.region_id,
                "functions":      r.functions,
                "failure_runs":   r.failure_run_count,
                "normal_runs":    r.normal_run_count,
                "contains_fault": r.contains_fault_site,
                "fragments":      len(r.fragment_ids),
            }
            for r in result.atlas.failure_adjacent_regions
        ]
        hot_summary = [
            {
                "path_id":          h.path_id,
                "block_count":      len(h.block_ids),
                "functions":        h.functions,
                "coverage":         h.coverage_fraction,
                "in_failure_runs":  h.appears_in_failure,
            }
            for h in result.atlas.hot_paths
        ]
        return {
            "target_id":              result.target_id,
            "functions":              result.total_functions_analysed,
            "blocks":                 result.total_blocks_analysed,
            "fragments":              result.total_fragments_catalogued,
            "indirect_callsites":     result.total_indirect_callsites,
            "runs_mapped":            result.total_runs_mapped,
            "hot_paths":              result.total_hot_paths,
            "failure_adjacent_regions": result.total_failure_adjacent_regions,
            "mean_mapping_coverage":  result.atlas.mean_mapping_coverage,
            "exit_criterion_met":     result.exit_criterion_met,
            "fragment_semantics_breakdown": catalog_summary,
            "failure_adjacent_regions_detail": fa_summary,
            "hot_paths_detail":       hot_summary,
        }
