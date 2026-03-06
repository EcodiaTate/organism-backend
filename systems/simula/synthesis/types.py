"""
EcodiaOS -- Simula Synthesis Types (Stage 5A)

Types for the neurosymbolic synthesis subsystem — fast-path alternatives
to the expensive CEGIS agentic loop.  Three strategies are supported:

  - HySynth:      Probabilistic CFG bottom-up beam search
  - Sketch+Solve: LLM template with symbolic hole-filling (Z3 / type enum / micro-LLM)
  - ChopChop:     Type-directed constrained generation (generate-then-verify chunks)

A strategy selector routes each proposal to the best-fit strategy and
falls back to CEGIS when no strategy scores above the confidence threshold.
"""

from __future__ import annotations

import enum

from pydantic import Field

from primitives.common import EOSBaseModel

# ── Enums ────────────────────────────────────────────────────────────────────


class SynthesisStrategy(enum.StrEnum):
    """Available synthesis strategies (ordered by typical speed)."""

    HYSYNTH = "hysynth"
    SKETCH_SOLVE = "sketch_solve"
    CHOPCHOP = "chopchop"
    CEGIS_FALLBACK = "cegis_fallback"


class SynthesisStatus(enum.StrEnum):
    """Terminal status of a synthesis attempt."""

    SYNTHESIZED = "synthesized"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class HoleKind(enum.StrEnum):
    """Kind of hole in a sketch template."""

    EXPRESSION = "expression"
    STATEMENT = "statement"
    BLOCK = "block"
    TYPE_ANNOTATION = "type_annotation"
    GUARD_CONDITION = "guard_condition"


# ── HySynth models ──────────────────────────────────────────────────────────


class CFGRule(EOSBaseModel):
    """One production rule in the probabilistic context-free grammar."""

    lhs: str  # non-terminal symbol
    rhs: list[str]  # sequence of terminals / non-terminals
    weight: float = 1.0  # probability weight (normalised per lhs at search time)
    source: str = ""  # "ast_exemplar"|"convention"|"llm_suggested"


class HySynthResult(EOSBaseModel):
    """Output of HySynth probabilistic CFG bottom-up search."""

    status: SynthesisStatus = SynthesisStatus.SKIPPED
    grammar_rules: int = 0
    candidates_explored: int = 0
    candidates_valid: int = 0
    best_candidate_code: str = ""
    best_candidate_score: float = 0.0
    ast_valid: bool = False
    type_valid: bool = False
    duration_ms: int = 0
    llm_tokens_for_weights: int = 0  # one-shot grammar weight call


# ── Sketch+Solve models ────────────────────────────────────────────────────


class SketchHole(EOSBaseModel):
    """One hole in a sketch template, to be filled by the solver."""

    hole_id: str  # e.g. "__HOLE_0__"
    kind: HoleKind = HoleKind.EXPRESSION
    type_hint: str = ""  # expected Python type
    constraints: list[str] = Field(default_factory=list)  # Z3-expressible constraints
    filled_value: str = ""  # populated after solving


class SketchTemplate(EOSBaseModel):
    """LLM-generated code template with typed holes."""

    template_code: str = ""
    holes: list[SketchHole] = Field(default_factory=list)
    llm_tokens: int = 0


class SketchSolveResult(EOSBaseModel):
    """Output of sketch-based synthesis (LLM template + symbolic hole-filling)."""

    status: SynthesisStatus = SynthesisStatus.SKIPPED
    template: SketchTemplate | None = None
    holes_total: int = 0
    holes_filled_z3: int = 0
    holes_filled_enum: int = 0
    holes_filled_llm: int = 0
    holes_unfilled: int = 0
    final_code: str = ""
    ast_valid: bool = False
    type_valid: bool = False
    duration_ms: int = 0
    z3_solver_ms: int = 0


# ── ChopChop models ────────────────────────────────────────────────────────


class GrammarConstraint(EOSBaseModel):
    """A type or grammar constraint enforced on generated code chunks."""

    constraint_type: str = ""  # "type_annotation"|"return_type"|"arg_type"|"import"|"grammar"
    target: str = ""  # e.g. "return type of process()"
    expected: str = ""  # e.g. "list[str]"
    satisfied: bool = False


class ChopChopResult(EOSBaseModel):
    """Output of ChopChop type-directed constrained generation."""

    status: SynthesisStatus = SynthesisStatus.SKIPPED
    chunks_generated: int = 0
    chunks_valid: int = 0
    chunks_retried: int = 0
    constraints_total: int = 0
    constraints_satisfied: int = 0
    final_code: str = ""
    ast_valid: bool = False
    type_valid: bool = False
    duration_ms: int = 0
    llm_tokens: int = 0


# ── Strategy selection & aggregate result ───────────────────────────────────


class SynthesisSelectionReason(EOSBaseModel):
    """Why a particular strategy was chosen by the selector."""

    strategy: SynthesisStrategy
    score: float = 0.0
    factors: dict[str, float] = Field(default_factory=dict)
    reasoning: str = ""


class SynthesisResult(EOSBaseModel):
    """Aggregate synthesis result, wrapping whichever strategy was used."""

    strategy: SynthesisStrategy = SynthesisStrategy.CEGIS_FALLBACK
    status: SynthesisStatus = SynthesisStatus.SKIPPED
    selection_reason: SynthesisSelectionReason | None = None
    # Strategy-specific results (only the chosen one is populated)
    hysynth_result: HySynthResult | None = None
    sketch_solve_result: SketchSolveResult | None = None
    chopchop_result: ChopChopResult | None = None
    # Aggregate metrics
    final_code: str = ""
    files_written: list[str] = Field(default_factory=list)
    speedup_vs_cegis: float = 0.0  # >1.0 means faster than baseline
    total_llm_tokens: int = 0
    total_duration_ms: int = 0
    fell_back_to_cegis: bool = False
