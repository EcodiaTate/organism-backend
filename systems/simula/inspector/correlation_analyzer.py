"""
EcodiaOS — Inspector Phase 8: Correlation Analyzer (Orchestrator)

Single-entry orchestrator for the Phase 8 cross-layer correlation pipeline:

  CausalGraphBuilder    → CausalGraph
  CausalChainAssembler  → list[CausalChain]
  PropagationMotifMiner → MotifCatalog
  SynthesisBuilder      → FinalSynthesis
  → Phase8Result

Usage
-----
  # Full pipeline (Phases 3–7 → 8):
  analyzer = CorrelationAnalyzer()
  result = analyzer.analyze(
      phase3_result=phase3_result,
      phase4_result=phase4_result,
      phase5_result=phase5_result,
      phase6_result=phase6_result,
      phase7_result=phase7_result,
  )

  # Explain a specific chain end-to-end:
  story = analyzer.explain_chain(result, chain_id)

  # List all findings for a specific motif kind:
  findings = analyzer.findings_for_motif_kind(result, MotifKind.TIMING_LEAKAGE)

  # Generate the thesis-ready narrative dict:
  synthesis = analyzer.full_synthesis(result)

Exit criterion
--------------
Phase8Result.exit_criterion_met = True when:
  - ≥1 CausalChain spans ≥3 distinct layers
  - ≥1 PropagationMotif with occurrence_count ≥ 2
  - FinalSynthesis.thesis is non-empty
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from systems.simula.inspector.correlation_engine import (
    CausalChainAssembler,
    CausalGraphBuilder,
    PropagationMotifMiner,
    SynthesisBuilder,
)
from systems.simula.inspector.correlation_types import (
    CausalChain,
    MotifKind,
    Phase8Result,
    PropagationMotif,
)

if TYPE_CHECKING:
    from systems.simula.inspector.constraint_types import Phase4Result
    from systems.simula.inspector.protocol_types import Phase6Result
    from systems.simula.inspector.static_types import Phase3Result
    from systems.simula.inspector.trust_types import Phase5Result
    from systems.simula.inspector.variance_types import Phase7Result

logger = structlog.get_logger().bind(system="simula.inspector.correlation_analyzer")


class CorrelationAnalyzer:
    """
    Phase 8 orchestrator — builds a Phase8Result for a target.

    The result combines:
    - A CausalGraph linking EventNodes across all Phases 3–7
    - Assembled CausalChains (end-to-end scenario stories)
    - A MotifCatalog of recurring propagation patterns
    - A FinalSynthesis (thesis + key findings + propagation model)
    - Aggregate statistics and exit criterion flag

    Parameters
    ----------
    min_edge_weight     — minimum edge confidence to include (default 0.3)
    max_chain_depth     — maximum chain length in hops (default 8)
    min_chain_layers    — minimum distinct layers per chain (default 2)
    max_chains          — maximum chains to produce (default 30)
    min_motif_occ       — minimum chain count for a motif (default 2)
    max_motif_len       — maximum motif pattern length (default 4)
    """

    def __init__(
        self,
        min_edge_weight: float = 0.3,
        max_chain_depth: int = 8,
        min_chain_layers: int = 2,
        max_chains: int = 30,
        min_motif_occ: int = 2,
        max_motif_len: int = 4,
    ) -> None:
        self._graph_builder = CausalGraphBuilder(min_edge_weight=min_edge_weight)
        self._chain_assembler = CausalChainAssembler(
            max_depth=max_chain_depth,
            min_layers=min_chain_layers,
            max_chains=max_chains,
        )
        self._motif_miner = PropagationMotifMiner(
            min_occurrences=min_motif_occ,
            max_motif_len=max_motif_len,
        )
        self._synthesis_builder = SynthesisBuilder()
        self._log = logger

    # ── Primary entry point ───────────────────────────────────────────────────

    def analyze(
        self,
        phase3_result: Phase3Result | None = None,
        phase4_result: Phase4Result | None = None,
        phase5_result: Phase5Result | None = None,
        phase6_result: Phase6Result | None = None,
        phase7_result: Phase7Result | None = None,
    ) -> Phase8Result:
        """
        Build a Phase8Result from available upstream phase outputs.

        At least one of phase4/5/6/7_result should be provided.
        If only phase7 is given, the graph will be timing-only (minimal).

        Args:
            phase3_result: Phase 3 static analysis (fragments).
            phase4_result: Phase 4 steerability model (regions).
            phase5_result: Phase 5 trust graph (corridors).
            phase6_result: Phase 6 FSM stress (boundary failures).
            phase7_result: Phase 7 variance analysis (channels).

        Returns:
            Phase8Result with causal graph, chains, motif catalog,
            synthesis, and exit criterion flag.
        """
        target_id = (
            getattr(phase7_result, "target_id", None)
            or getattr(phase6_result, "target_id", None)
            or getattr(phase5_result, "target_id", None)
            or getattr(phase4_result, "target_id", None)
            or getattr(phase3_result, "target_id", None)
            or "unknown"
        )
        log = self._log.bind(target_id=target_id)
        log.info("correlation_analysis_started")

        # 1. Build the causal graph
        graph = self._graph_builder.build(
            target_id=target_id,
            phase3_result=phase3_result,
            phase4_result=phase4_result,
            phase5_result=phase5_result,
            phase6_result=phase6_result,
            phase7_result=phase7_result,
        )

        # 2. Assemble causal chains
        chains = self._chain_assembler.assemble(graph)

        # 3. Mine propagation motifs
        motif_catalog = self._motif_miner.mine(chains, graph, target_id)

        # 4. Build synthesis
        synthesis = self._synthesis_builder.build(
            target_id=target_id,
            graph=graph,
            motif_catalog=motif_catalog,
            chains=chains,
        )

        # 5. Aggregate statistics
        chains_3plus = sum(1 for c in chains if c.distinct_layer_count >= 3)
        chains_timing = sum(1 for c in chains if c.reaches_timing_channel)
        chains_sec = sum(1 for c in chains if c.is_security_relevant)
        motifs_recurring = sum(
            1 for m in motif_catalog.motifs if m.occurrence_count >= 2
        )

        # 6. Exit criterion
        exit_met = (
            chains_3plus >= 1
            and motifs_recurring >= 1
            and bool(synthesis.thesis)
        )

        result = Phase8Result(
            target_id=target_id,
            causal_graph=graph,
            motif_catalog=motif_catalog,
            causal_chains=chains,
            synthesis=synthesis,
            total_nodes=graph.total_nodes,
            total_edges=graph.total_edges,
            total_cross_layer_edges=graph.total_cross_layer_edges,
            total_chains=len(chains),
            total_motifs=motif_catalog.total_motifs,
            security_relevant_motifs=motif_catalog.security_relevant_motifs,
            chains_spanning_3plus_layers=chains_3plus,
            chains_reaching_timing_channel=chains_timing,
            chains_security_relevant=chains_sec,
            distinct_layers_in_graph=graph.distinct_layers,
            distinct_layers_in_chains=len(
                set(l for c in chains for l in c.layers_represented)
            ),
            exit_criterion_met=exit_met,
        )

        log.info(
            "correlation_analysis_complete",
            nodes=graph.total_nodes,
            edges=graph.total_edges,
            cross_layer=graph.total_cross_layer_edges,
            chains=len(chains),
            chains_3plus=chains_3plus,
            chains_timing=chains_timing,
            motifs=motif_catalog.total_motifs,
            motifs_recurring=motifs_recurring,
            sec_motifs=motif_catalog.security_relevant_motifs,
            exit_criterion_met=exit_met,
        )

        return result

    # ── Targeted queries ──────────────────────────────────────────────────────

    def explain_chain(
        self,
        result: Phase8Result,
        chain_id: str,
    ) -> dict:
        """
        Return a structured explanation dict for one CausalChain.

        Returns:
          chain_id, narrative, layers, distinct_layer_count, confidence,
          status, reaches_timing_channel, is_security_relevant, steps,
          motifs_containing_this_chain
        """
        chain = next(
            (c for c in result.causal_chains if c.chain_id == chain_id), None
        )
        if not chain:
            return {"error": f"chain '{chain_id}' not found"}

        motifs_with_chain = [
            m.motif_id for m in result.motif_catalog.motifs
            if chain_id in m.instance_chain_ids
        ]

        steps_detail = [
            {
                "step": s.step_index,
                "label": s.node_label,
                "layer": s.layer.value,
                "mechanism": s.mechanism.value,
                "is_cross_layer": s.is_cross_layer_step,
                "cumulative_confidence": s.cumulative_confidence,
                "description": s.description,
            }
            for s in chain.steps
        ]

        return {
            "chain_id":              chain.chain_id,
            "narrative":             chain.narrative,
            "layers":                [l.value for l in chain.layers_represented],
            "distinct_layer_count":  chain.distinct_layer_count,
            "cross_layer_hops":      chain.cross_layer_hops,
            "confidence":            chain.confidence,
            "status":                chain.status.value,
            "reaches_timing_channel": chain.reaches_timing_channel,
            "is_security_relevant":  chain.is_security_relevant,
            "steps":                 steps_detail,
            "motifs_containing_this_chain": motifs_with_chain,
        }

    def findings_for_motif_kind(
        self,
        result: Phase8Result,
        motif_kind: MotifKind,
    ) -> list[PropagationMotif]:
        """Return all PropagationMotifs of a specific kind."""
        motif_ids = result.motif_catalog.motifs_by_kind.get(motif_kind.value, [])
        id_set = set(motif_ids)
        return [m for m in result.motif_catalog.motifs if m.motif_id in id_set]

    def chains_for_motif(
        self,
        result: Phase8Result,
        motif_id: str,
    ) -> list[CausalChain]:
        """Return all CausalChains that participate in a specific motif."""
        motif = next(
            (m for m in result.motif_catalog.motifs if m.motif_id == motif_id), None
        )
        if not motif:
            return []
        chain_id_set = set(motif.instance_chain_ids)
        return [c for c in result.causal_chains if c.chain_id in chain_id_set]

    def full_synthesis(self, result: Phase8Result) -> dict:
        """
        Return the full FinalSynthesis as a structured dict.

        This is the primary thesis-ready report output.
        """
        s = result.synthesis
        return {
            "synthesis_id":       s.synthesis_id,
            "target_id":          s.target_id,
            "thesis":             s.thesis,
            "propagation_model":  s.propagation_model,
            "limitations":        s.limitations,
            "reproducibility":    s.reproducibility,
            "contributing_phases": s.contributing_phases,
            "thesis_confidence":  s.thesis_confidence,
            "is_security_relevant": s.is_security_relevant,
            "key_findings": [
                {
                    "rank":              f.rank,
                    "claim":             f.claim,
                    "evidence":          f.evidence,
                    "layers_involved":   [l.value for l in f.layers_involved],
                    "is_security_relevant": f.is_security_relevant,
                    "confidence":        f.confidence,
                    "impact_score":      f.impact_score,
                    "motif_id":          f.motif_id,
                    "chain_ids":         f.chain_ids[:3],
                }
                for f in s.key_findings
            ],
        }

    # ── Summary helper ────────────────────────────────────────────────────────

    def model_summary(self, result: Phase8Result) -> dict:
        """
        Return a concise reporting dict for the Phase8Result.
        """
        # Top chains
        top_chains = sorted(
            result.causal_chains,
            key=lambda c: (c.distinct_layer_count, c.confidence),
            reverse=True,
        )[:5]
        chain_summaries = [
            {
                "chain_id":             c.chain_id,
                "layers":               [l.value for l in c.layers_represented],
                "distinct_layer_count": c.distinct_layer_count,
                "cross_layer_hops":     c.cross_layer_hops,
                "confidence":           round(c.confidence, 3),
                "status":               c.status.value,
                "reaches_timing":       c.reaches_timing_channel,
                "security_relevant":    c.is_security_relevant,
                "narrative":            c.narrative[:150],
            }
            for c in top_chains
        ]

        # Top motifs
        top_motifs = result.motif_catalog.motifs[:5]
        motif_summaries = [
            {
                "motif_id":       m.motif_id,
                "motif_kind":     m.motif_kind.value,
                "occurrences":    m.occurrence_count,
                "layers":         [l.value for l in m.participating_layers],
                "mechanism":      m.dominant_mechanism.value,
                "confidence":     round(m.confidence, 3),
                "impact_score":   m.impact_score,
                "security":       m.is_security_relevant,
                "description":    m.description[:120],
            }
            for m in top_motifs
        ]

        return {
            "target_id":                    result.target_id,
            "exit_criterion_met":           result.exit_criterion_met,
            "total_nodes":                  result.total_nodes,
            "total_edges":                  result.total_edges,
            "total_cross_layer_edges":      result.total_cross_layer_edges,
            "distinct_layers_in_graph":     result.distinct_layers_in_graph,
            "total_chains":                 result.total_chains,
            "chains_spanning_3plus_layers": result.chains_spanning_3plus_layers,
            "chains_reaching_timing":       result.chains_reaching_timing_channel,
            "chains_security_relevant":     result.chains_security_relevant,
            "total_motifs":                 result.total_motifs,
            "security_relevant_motifs":     result.security_relevant_motifs,
            "thesis_confidence":            result.synthesis.thesis_confidence,
            "thesis_preview":               result.synthesis.thesis[:300],
            "propagation_model":            result.synthesis.propagation_model,
            "contributing_phases":          result.synthesis.contributing_phases,
            "top_chains":                   chain_summaries,
            "top_motifs":                   motif_summaries,
        }
