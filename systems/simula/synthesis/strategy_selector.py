"""
EcodiaOS -- Simula Synthesis Strategy Selector (Stage 5A.4)

Routes each proposal to the best-fit synthesis strategy based on
proposal characteristics. Scores each strategy 0–1 and picks the
highest. Falls back to CEGIS when no strategy scores above threshold.

Routing heuristics:
  - **HySynth**: additive categories (ADD_EXECUTOR, ADD_PATTERN_DETECTOR,
    ADD_INPUT_CHANNEL) with structural hints from exemplar code
  - **Sketch+Solve**: modification categories (MODIFY_CONTRACT,
    ADD_SYSTEM_CAPABILITY) or non-empty code_hint with constraints
  - **ChopChop**: changes touching type-heavy systems with strong contracts
  - **CEGIS fallback**: when no strategy scores above threshold or synthesis fails
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from pathlib import Path

import structlog

from systems.simula.evolution_types import ChangeCategory
from systems.simula.synthesis.types import (
    SynthesisResult,
    SynthesisSelectionReason,
    SynthesisStatus,
    SynthesisStrategy,
)

if TYPE_CHECKING:

    from systems.simula.evolution_types import ChangeSpec, EvolutionProposal
    from systems.simula.synthesis.chopchop import ChopChopEngine
    from systems.simula.synthesis.hysynth import HySynthEngine
    from systems.simula.synthesis.sketch_solver import SketchSolver
logger = structlog.get_logger().bind(system="simula.synthesis.selector")

# ── Strategy routing constants ──────────────────────────────────────────────

# Categories where HySynth excels (structural, additive patterns)
_HYSYNTH_CATEGORIES: frozenset[ChangeCategory] = frozenset({
    ChangeCategory.ADD_EXECUTOR,
    ChangeCategory.ADD_PATTERN_DETECTOR,
    ChangeCategory.ADD_INPUT_CHANNEL,
})

# Categories where Sketch+Solve excels (contract modifications, capability additions)
_SKETCH_CATEGORIES: frozenset[ChangeCategory] = frozenset({
    ChangeCategory.MODIFY_CONTRACT,
    ChangeCategory.ADD_SYSTEM_CAPABILITY,
    ChangeCategory.ADJUST_BUDGET,
})

# Categories where ChopChop excels (timing, consolidation - type-heavy domains)
_CHOPCHOP_CATEGORIES: frozenset[ChangeCategory] = frozenset({
    ChangeCategory.MODIFY_CYCLE_TIMING,
    ChangeCategory.CHANGE_CONSOLIDATION,
})

# Minimum score to attempt a strategy (below this → CEGIS fallback)
_MIN_STRATEGY_SCORE: float = 0.35

# Type-heavy system paths (boost ChopChop for these)
_TYPE_HEAVY_PATHS: frozenset[str] = frozenset({
    "verification",
    "types.py",
    "primitives",
    "governance",
})


class SynthesisStrategySelector:
    """Routes proposals to the optimal synthesis strategy."""

    def __init__(
        self,
        hysynth: HySynthEngine,
        sketch_solver: SketchSolver,
        chopchop: ChopChopEngine,
        codebase_root: Path,
    ) -> None:
        self._hysynth = hysynth
        self._sketch = sketch_solver
        self._chopchop = chopchop
        self._codebase_root = codebase_root

    # ── Public API ──────────────────────────────────────────────────────────

    async def synthesise(
        self,
        proposal: EvolutionProposal,
        exemplar_code: str = "",
        context_code: str = "",
    ) -> SynthesisResult:
        """Score strategies → pick best → run → fall back to CEGIS on failure."""
        start = time.monotonic()

        # Score all strategies
        scores = self._score_strategies(proposal)
        best_strategy, best_score, factors = self._pick_best(scores)

        selection_reason = SynthesisSelectionReason(
            strategy=best_strategy,
            score=best_score,
            factors=factors,
            reasoning=self._explain_selection(best_strategy, factors),
        )

        logger.info(
            "synthesis_strategy_selected",
            strategy=best_strategy.value,
            score=f"{best_score:.2f}",
            category=proposal.category.value,
        )

        # If no strategy is confident enough, signal CEGIS fallback
        if best_score < _MIN_STRATEGY_SCORE:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.info("synthesis_below_threshold", best_score=best_score)
            return SynthesisResult(
                strategy=SynthesisStrategy.CEGIS_FALLBACK,
                status=SynthesisStatus.SKIPPED,
                selection_reason=selection_reason,
                fell_back_to_cegis=True,
                total_duration_ms=elapsed_ms,
            )

        # Run the selected strategy
        result = await self._run_strategy(
            best_strategy, proposal.change_spec, exemplar_code, context_code
        )

        # If strategy failed, fall back to CEGIS
        if result.status in (SynthesisStatus.FAILED, SynthesisStatus.TIMEOUT):
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.info(
                "synthesis_fallback_to_cegis",
                strategy=best_strategy.value,
                status=result.status.value,
            )
            result.strategy = SynthesisStrategy.CEGIS_FALLBACK
            result.fell_back_to_cegis = True
            result.total_duration_ms = elapsed_ms
            return result

        result.strategy = best_strategy
        result.selection_reason = selection_reason
        result.total_duration_ms = int((time.monotonic() - start) * 1000)
        return result

    # ── Strategy scoring ────────────────────────────────────────────────────

    def _score_strategies(
        self, proposal: EvolutionProposal
    ) -> dict[SynthesisStrategy, dict[str, float]]:
        """Score each strategy based on proposal characteristics."""
        spec = proposal.change_spec
        category = proposal.category
        scores: dict[SynthesisStrategy, dict[str, float]] = {
            SynthesisStrategy.HYSYNTH: {},
            SynthesisStrategy.SKETCH_SOLVE: {},
            SynthesisStrategy.CHOPCHOP: {},
        }

        # ── HySynth scoring ────────────────────────────────────────────────
        hy = scores[SynthesisStrategy.HYSYNTH]
        hy["category_match"] = 0.6 if category in _HYSYNTH_CATEGORIES else 0.1
        hy["structural_hint"] = (
            0.3 if spec.executor_name or spec.detector_name or spec.channel_name else 0.0
        )
        hy["code_hint"] = 0.1 if spec.code_hint else 0.0

        # ── Sketch+Solve scoring ───────────────────────────────────────────
        sk = scores[SynthesisStrategy.SKETCH_SOLVE]
        sk["category_match"] = 0.5 if category in _SKETCH_CATEGORIES else 0.1
        sk["code_hint"] = 0.3 if spec.code_hint else 0.0
        sk["constraints"] = (
            0.2 if spec.contract_changes or spec.budget_parameter else 0.0
        )

        # ── ChopChop scoring ──────────────────────────────────────────────
        ch = scores[SynthesisStrategy.CHOPCHOP]
        ch["category_match"] = 0.5 if category in _CHOPCHOP_CATEGORIES else 0.1
        ch["type_heavy"] = (
            0.3 if self._touches_type_heavy(spec.affected_systems) else 0.0
        )
        ch["has_context"] = 0.1 if spec.additional_context else 0.0

        return scores

    @staticmethod
    def _touches_type_heavy(affected_systems: list[str]) -> bool:
        """Check if any affected system is in the type-heavy list."""
        return any(
            path in system.lower()
            for system in affected_systems
            for path in _TYPE_HEAVY_PATHS
        )

    def _pick_best(
        self, scores: dict[SynthesisStrategy, dict[str, float]]
    ) -> tuple[SynthesisStrategy, float, dict[str, float]]:
        """Pick the highest-scoring strategy. Returns (strategy, total_score, factors)."""
        best: SynthesisStrategy = SynthesisStrategy.CEGIS_FALLBACK
        best_total = 0.0
        best_factors: dict[str, float] = {}

        for strategy, factors in scores.items():
            total = sum(factors.values())
            if total > best_total:
                best = strategy
                best_total = total
                best_factors = factors

        return best, best_total, best_factors

    @staticmethod
    def _explain_selection(
        strategy: SynthesisStrategy, factors: dict[str, float]
    ) -> str:
        """Generate human-readable explanation of strategy selection."""
        top_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)[:3]
        factor_str = ", ".join(f"{k}={v:.2f}" for k, v in top_factors)
        return f"Selected {strategy.value} based on: {factor_str}"

    # ── Strategy execution ──────────────────────────────────────────────────

    async def _run_strategy(
        self,
        strategy: SynthesisStrategy,
        change_spec: ChangeSpec,
        exemplar_code: str,
        context_code: str,
    ) -> SynthesisResult:
        """Execute the selected synthesis strategy."""
        if strategy == SynthesisStrategy.HYSYNTH:
            hysynth_result = await self._hysynth.synthesise(
                change_spec, exemplar_code
            )
            return SynthesisResult(
                strategy=strategy,
                status=hysynth_result.status,
                hysynth_result=hysynth_result,
                final_code=hysynth_result.best_candidate_code,
                speedup_vs_cegis=0.0,
                total_llm_tokens=hysynth_result.llm_tokens_for_weights,
                total_duration_ms=hysynth_result.duration_ms,
            )

        if strategy == SynthesisStrategy.SKETCH_SOLVE:
            sketch_result = await self._sketch.synthesise(
                change_spec, exemplar_code, context_code
            )
            return SynthesisResult(
                strategy=strategy,
                status=sketch_result.status,
                sketch_solve_result=sketch_result,
                final_code=sketch_result.final_code,
                speedup_vs_cegis=0.0,
                total_llm_tokens=(
                    sketch_result.template.llm_tokens if sketch_result.template else 0
                ),
                total_duration_ms=sketch_result.duration_ms,
            )

        if strategy == SynthesisStrategy.CHOPCHOP:
            chopchop_result = await self._chopchop.synthesise(
                change_spec, context_code
            )
            return SynthesisResult(
                strategy=strategy,
                status=chopchop_result.status,
                chopchop_result=chopchop_result,
                final_code=chopchop_result.final_code,
                speedup_vs_cegis=0.0,
                total_llm_tokens=chopchop_result.llm_tokens,
                total_duration_ms=chopchop_result.duration_ms,
            )

        # Should not reach here - CEGIS fallback doesn't run synthesis
        return SynthesisResult(
            strategy=SynthesisStrategy.CEGIS_FALLBACK,
            status=SynthesisStatus.SKIPPED,
            fell_back_to_cegis=True,
        )
