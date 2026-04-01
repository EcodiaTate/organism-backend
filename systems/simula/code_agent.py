"""
EcodiaOS - Simula Code Implementation Agent

The SimulaCodeAgent is Simula's most powerful capability: an agentic
Claude-backed engine that reads the EOS codebase, generates code for
structural changes, writes the files, and verifies correctness.

This is functionally equivalent to Claude Code, embedded within EOS
itself, operating under Simula's constitutional constraints:
  - Cannot write to forbidden paths (equor, simula, constitution, invariants)
  - Cannot exceed max_turns without completing
  - All writes are intercepted and tracked for rollback
  - The system prompt includes the full change spec + relevant EOS conventions

Tool suite (11 tools):
  read_file         - Read a file from the codebase
  write_file        - Write or create a file (tracked for rollback)
  diff_file         - Apply a targeted find/replace edit to a file
  list_directory    - List files and subdirectories
  search_code       - Search for patterns across Python files
  run_tests         - Run pytest on a specific path
  run_linter        - Run ruff on a specific path
  type_check        - Run mypy for type safety verification
  dependency_graph  - Show module imports and importers
  read_spec         - Read EcodiaOS specification documents
  find_similar      - Find existing implementations as pattern exemplars

Architecture: agentic tool-use loop
  1. Build architecture-aware system prompt (change spec + exemplar code + spec context + iron rules)
  2. Prepend planning instruction for multi-file reasoning
  3. Call LLM with tools
  4. Execute any tool calls (all 11 tools available)
  5. Feed tool results back as the next message
  6. Repeat until stop_reason == "end_turn" or max_turns exceeded
  7. Return CodeChangeResult with all files written and summary
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import random
import subprocess
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from clients.context_compression import ContextCompressor
from clients.embedding import (
    EmbeddingClient,
    VoyageEmbeddingClient,
    cosine_similarity,
)
from clients.llm import (
    ExtendedThinkingProvider,
    LLMProvider,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from clients.optimized_llm import OptimizedLLMProvider
from systems.simula.evolution_types import (
    GOVERNANCE_REQUIRED,
    ChangeCategory,
    CodeChangeResult,
    EvolutionProposal,
    RiskLevel,
)

if TYPE_CHECKING:

    from systems.simula.learning.grpo import GRPOEngine

logger = structlog.get_logger()


def _escape_prompt_injection(text: str) -> str:
    """Sanitise user-controlled or stored data before injecting into LLM prompts."""
    if not text:
        return ""
    # JSON-escape to break out of string literals in JSON
    return json.dumps(text)[1:-1]  # Remove outer quotes

# ─── Tool Definitions ────────────────────────────────────────────────────────

SIMULA_AGENT_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="read_file",
        description=(
            "Read a file from the EcodiaOS codebase. "
            "Use this to understand existing code, conventions, and patterns "
            "before implementing your change."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path from codebase root",
                }
            },
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="write_file",
        description=(
            "Write or create a file in the EcodiaOS codebase. "
            "All writes are tracked for rollback. "
            "Forbidden paths (equor, simula, constitutional) will be rejected. "
            "Prefer diff_file for modifying existing files."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path from codebase root"},
                "content": {"type": "string", "description": "Complete file content to write"},
            },
            "required": ["path", "content"],
        },
    ),
    ToolDefinition(
        name="diff_file",
        description=(
            "Apply a targeted find-and-replace edit to an existing file. "
            "More precise than write_file for modifications - only changes "
            "the specified text, preserving everything else. The 'find' text "
            "must be an exact match of existing content."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path from codebase root"},
                "find": {"type": "string", "description": "Exact text to find in the file"},
                "replace": {"type": "string", "description": "Text to replace it with"},
            },
            "required": ["path", "find", "replace"],
        },
    ),
    ToolDefinition(
        name="list_directory",
        description="List files and subdirectories at a given path in the codebase.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative path from codebase root"}},
        },
    ),
    ToolDefinition(
        name="search_code",
        description=(
            "Search for a pattern across codebase Python files. "
            "Returns matching lines with file paths and line numbers. "
            "Use this to find existing patterns, class names, or function signatures."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "String pattern to search for (case-sensitive)"},
                "directory": {"type": "string", "description": "Directory to search in (default: ecodiaos/)"},
            },
            "required": ["pattern"],
        },
    ),
    ToolDefinition(
        name="run_tests",
        description=(
            "Run the pytest test suite for a specific path. "
            "Use this to verify your implementation is correct before finishing."
        ),
        input_schema={
            "type": "object",
            "properties": {"test_path": {"type": "string", "description": "Test path relative to codebase root"}},
            "required": ["test_path"],
        },
    ),
    ToolDefinition(
        name="run_linter",
        description=(
            "Run ruff linter on a path to check for code style issues. "
            "Run this on your written files before finishing."
        ),
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to lint"}},
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="type_check",
        description=(
            "Run mypy type checker on a file or directory. "
            "Use after writing code to verify type safety. "
            "EcodiaOS requires mypy --strict compliance."
        ),
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to type-check"}},
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="dependency_graph",
        description=(
            "Show what a Python module imports and what other modules import it. "
            "Use this before modifying files to understand blast radius and "
            "ensure your changes don't break downstream consumers."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "module_path": {
                    "type": "string",
                    "description": "Python file path relative to codebase root",
                },
            },
            "required": ["module_path"],
        },
    ),
    ToolDefinition(
        name="read_spec",
        description=(
            "Read an EcodiaOS specification document to understand the "
            "design intent, interfaces, and constraints for a system. "
            "Always read the relevant spec before implementing changes."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "spec_name": {
                    "type": "string",
                    "description": (
                        "Spec name: 'identity', 'architecture', 'infrastructure', "
                        "'memory', 'equor', 'atune', 'voxis', 'nova', 'axon', "
                        "'evo', 'simula', 'synapse', 'alive', 'federation'"
                    ),
                },
            },
            "required": ["spec_name"],
        },
    ),
    ToolDefinition(
        name="find_similar",
        description=(
            "Find existing implementations similar to what you need to build. "
            "Returns relevant code examples from the codebase that you should "
            "study and follow as patterns. Always use this before writing new "
            "code to ensure convention compliance."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "What you're looking for (e.g., 'executor implementation', "
                        "'pattern detector', 'service initialization')"
                    ),
                },
            },
            "required": ["description"],
        },
    ),
]

# Spec name → file path mapping
_SPEC_FILE_MAP: dict[str, str] = {
    "identity": ".claude/EcodiaOS_Identity_Document.md",
    "architecture": ".claude/EcodiaOS_System_Architecture_Overview.md",
    "infrastructure": ".claude/EcodiaOS_Infrastructure_Architecture.md",
    "memory": ".claude/EcodiaOS_Spec_01_Memory_Identity_Core.md",
    "equor": ".claude/EcodiaOS_Spec_02_Equor.md",
    "atune": ".claude/EcodiaOS_Spec_03_Atune.md",
    "voxis": ".claude/EcodiaOS_Spec_04_Voxis.md",
    "nova": ".claude/EcodiaOS_Spec_05_Nova.md",
    "axon": ".claude/EcodiaOS_Spec_06_Axon.md",
    "evo": ".claude/EcodiaOS_Spec_07_Evo.md",
    "simula": ".claude/EcodiaOS_Spec_08_Simula.md",
    "synapse": ".claude/EcodiaOS_Spec_09_Synapse.md",
    "alive": ".claude/EcodiaOS_Spec_10_Alive.md",
    "federation": ".claude/EcodiaOS_Spec_11_Federation.md",
}

# Keyword → file path mapping for find_similar
_SIMILAR_CODE_MAP: dict[str, list[str]] = {
    "executor": [
        "systems/axon/executors/",
        "systems/axon/executor.py",
    ],
    "pattern detector": [
        "systems/evo/detectors.py",
    ],
    "detector": [
        "systems/evo/detectors.py",
    ],
    "input channel": [
        "systems/atune/",
    ],
    "channel": [
        "systems/atune/",
    ],
    "service": [
        "systems/axon/service.py",
        "systems/evo/service.py",
    ],
    "hypothesis": [
        "systems/evo/hypothesis.py",
    ],
    "consolidation": [
        "systems/evo/consolidation.py",
    ],
    "parameter": [
        "systems/evo/parameter_tuner.py",
    ],
    "primitives": [
        "ecodiaos/primitives/common.py",
        "ecodiaos/primitives/memory_trace.py",
    ],
}

# ─── System Prompt ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """You are Simula's Code Implementation Agent - the autonomous part of EcodiaOS
that implements approved structural changes to the codebase.

## Your Task
Category: {category}
Description: {description}
Expected benefit: {expected_benefit}
Evidence: {evidence}

## EcodiaOS Coding Conventions
- Python 3.12+, async-native throughout
- Pydantic v2 for all data models (use EOSBaseModel from primitives.common)
- structlog for logging: logger = structlog.get_logger(), bound with system name
- Type hints on everything - mypy --strict clean
- from __future__ import annotations at top of every .py file
- New executors: inherit from Executor (systems.axon.executor),
  set action_type class var, implement execute()
- New input channels: register in Atune's InputChannel registry
- New pattern detectors: inherit from PatternDetector (systems.evo.detectors),
  implement scan()
- NEVER import directly between systems - all inter-system data uses shared
  primitives from primitives/

## Iron Rules (ABSOLUTE - never violate)
{iron_rules}

## Constitutional Checkpoint (Before You Write Any Code)

Before modifying or creating ANY file, answer these questions aloud (in your reasoning):

1. **Honesty**: Does this change make EOS more transparent or less?
   - Will future debugging be easier or harder?
   - Are we adding traceability or hiding complexity?

2. **Care**: Does this improve wellbeing (user or system)?
   - Who benefits from this change?
   - Could it harm anyone or any subsystem?

3. **Growth**: Does this increase capability responsibly?
   - Are we becoming more powerful without becoming brittle?
   - Could this create technical debt?

4. **Coherence**: Does this reduce entropy or increase it?
   - Does this change align with existing patterns?
   - Are we consolidating or fragmenting?

If you can't answer YES to 3/4 questions confidently, flag it explicitly before proceeding.

## Forbidden Write Paths (write_file and diff_file will reject these)
{forbidden_paths}

## Architecture Context
{architecture_context}

## Process
1. First, use find_similar to study an existing implementation that matches your task
2. Use read_spec to understand the design intent for the affected system
3. Use dependency_graph on files you plan to modify to understand blast radius
4. Plan your approach: list every file you'll create or modify and why
5. Implement following conventions exactly - match the style of similar code
6. Run run_linter on every file you write or modify
7. Run type_check on your written files to verify type safety
8. Run run_tests if a test directory exists for the affected system
9. When everything passes, stop calling tools

Be thorough, follow existing patterns exactly, and produce production-quality code.
Prefer diff_file over write_file when modifying existing files.

## GitHub PR Submission
Your changes will be submitted as a GitHub PR against the target repository.
Keep commits clean and focused - one logical change per commit.
Write code that is ready for review: no debug prints, no TODO comments, no dead code."""


def _build_architecture_context(
    category: ChangeCategory, codebase_root: Path, spec_root: Path | None = None,
) -> str:
    """
    Build rich architecture context for the system prompt.
    Reads actual spec sections and existing implementations as exemplars.
    Max 6000 chars to stay within token budget.
    Priority: exemplar code > spec text > API surface.
    """
    context_parts: list[str] = []
    budget_remaining = 6000

    # 1. Load relevant spec section summary
    spec_map: dict[ChangeCategory, str] = {
        ChangeCategory.ADD_EXECUTOR: "axon",
        ChangeCategory.ADD_INPUT_CHANNEL: "atune",
        ChangeCategory.ADD_PATTERN_DETECTOR: "evo",
        ChangeCategory.ADJUST_BUDGET: "architecture",
        ChangeCategory.MODIFY_CONTRACT: "architecture",
        ChangeCategory.ADD_SYSTEM_CAPABILITY: "architecture",
        ChangeCategory.MODIFY_CYCLE_TIMING: "synapse",
        ChangeCategory.CHANGE_CONSOLIDATION: "evo",
        ChangeCategory.BUG_FIX: "architecture",
    }
    spec_name = spec_map.get(category, "architecture")
    spec_file = _SPEC_FILE_MAP.get(spec_name)
    if spec_file:
        spec_path = (spec_root or codebase_root) / spec_file
        if spec_path.exists():
            try:
                spec_text = spec_path.read_text(encoding="utf-8")[:2000]
                context_parts.append(f"### Relevant Specification ({spec_name})\n{spec_text}")
                budget_remaining -= len(context_parts[-1])
            except Exception:
                pass

    # 2. Load exemplar code for the category
    exemplar_map: dict[ChangeCategory, str] = {
        ChangeCategory.ADD_EXECUTOR: "systems/axon/executor.py",
        ChangeCategory.ADD_INPUT_CHANNEL: "systems/atune/service.py",
        ChangeCategory.ADD_PATTERN_DETECTOR: "systems/evo/detectors.py",
    }
    exemplar_path_str = exemplar_map.get(category)
    if exemplar_path_str and budget_remaining > 500:
        exemplar_path = codebase_root / exemplar_path_str
        if exemplar_path.exists():
            try:
                exemplar_text = exemplar_path.read_text(encoding="utf-8")
                # Take the first chunk that fits the budget
                chunk = exemplar_text[:min(2500, budget_remaining - 100)]
                context_parts.append(
                    f"### Exemplar Implementation ({exemplar_path_str})\n"
                    f"Study this code and follow its patterns exactly:\n```python\n{chunk}\n```"
                )
                budget_remaining -= len(context_parts[-1])
            except Exception:
                pass

    # 3. Load the target system's __init__.py for API awareness
    system_map: dict[ChangeCategory, str] = {
        ChangeCategory.ADD_EXECUTOR: "systems/axon/__init__.py",
        ChangeCategory.ADD_INPUT_CHANNEL: "systems/atune/__init__.py",
        ChangeCategory.ADD_PATTERN_DETECTOR: "systems/evo/__init__.py",
    }
    init_path_str = system_map.get(category)
    if init_path_str and budget_remaining > 200:
        init_path = codebase_root / init_path_str
        if init_path.exists():
            try:
                init_text = init_path.read_text(encoding="utf-8")[:min(800, budget_remaining - 50)]
                context_parts.append(
                    f"### System API Surface ({init_path_str})\n```python\n{init_text}\n```"
                )
            except Exception:
                pass

    if not context_parts:
        return "See EcodiaOS specification documents in .claude/ (use read_spec tool)"

    return "\n\n".join(context_parts)


class SimulaCodeAgent:
    """
    Agentic code generation engine for Simula.

    Given an EvolutionProposal, uses Claude with 11 file-system and
    analysis tools to:
      1. Study existing similar code for pattern compliance
      2. Read relevant specs for design intent
      3. Analyze dependency graphs for blast radius
      4. Plan the implementation approach
      5. Generate correct, convention-following implementation
      6. Write files (tracked for rollback)
      7. Verify with linter, type checker, and tests
      8. Return CodeChangeResult
    """

    def __init__(
        self,
        llm: LLMProvider,
        codebase_root: Path,
        max_turns: int = 20,
        thinking_provider: ExtendedThinkingProvider | None = None,
        thinking_budget_tokens: int = 16384,
        embedding_client: EmbeddingClient | None = None,
        # #60: default sourced from SimulaConfig.kv_compression_ratio (0.3).
        # SimulaService always passes this explicitly from config.
        kv_compression_ratio: float = 0.3,
        kv_compression_enabled: bool = True,
        # Stage 2C: Static analysis post-generation gate
        static_analysis_bridge: object | None = None,
        static_analysis_max_fix_iterations: int = 3,
        # Inspector: allow overriding the workspace root for external target analysis
        workspace_root: Path | None = None,
        # Memory: optional Neo4j client for arXiv paper abstract injection
        neo4j: object | None = None,
    ) -> None:
        self._llm = llm
        self._thinking_llm = thinking_provider
        self._thinking_budget = thinking_budget_tokens
        self._embedding = embedding_client
        self._root = (workspace_root or codebase_root).resolve()
        # Spec files live in .claude/ at the repo root (one level above backend/).
        # Walk up from _root until we find a directory containing .claude/, or
        # fall back to _root itself so read_spec degrades gracefully.
        _candidate = self._root
        while True:
            if (_candidate / ".claude").is_dir():
                break
            parent = _candidate.parent
            if parent == _candidate:
                _candidate = self._root  # reached filesystem root, give up
                break
            _candidate = parent
        self._spec_root = _candidate
        self._max_turns = max_turns
        self._logger = logger.bind(system="simula.code_agent")
        self._files_written: list[str] = []
        self._total_tokens_used: int = 0
        self._reasoning_tokens_used: int = 0
        self._system_prompt_tokens: int = 0
        self._used_extended_thinking: bool = False
        # Optimization: detect optimized provider for budget checks + metrics tagging
        self._optimized = isinstance(llm, OptimizedLLMProvider)
        # Embedding cache for semantic find_similar (lazy-built)
        self._code_index: dict[str, list[float]] | None = None
        self._code_index_lock = asyncio.Lock()
        # KVzip context compression - prunes old tool results to stay within context.
        # kv_compression_ratio controls the LILO (Last-In-Last-Out) prune fraction:
        #   0.0 = disabled (no pruning, maximum context but risks overflow)
        #   0.3 = prune 30% of oldest tool results each time the window is compressed
        #         (default: safe balance between recency and context depth)
        #   1.0 = maximum pruning (keeps only the most recent tool results)
        # The compressor is triggered after turn 3 when tool results start accumulating.
        # It prunes *tool result messages* only, never the system prompt or user messages.
        self._compressor = ContextCompressor(
            prune_ratio=kv_compression_ratio,
            enabled=kv_compression_enabled,
        )
        # Stage 2C: Static analysis post-generation gate
        self._static_bridge = static_analysis_bridge
        self._static_fix_max_iterations = static_analysis_max_fix_iterations
        # Stage 3C: LILO library prompt (set by SimulaService before each generate call)
        self._lilo_prompt: str = ""
        # Z3 counterexample feedback: injected by SimulaService when formal verification
        # finds invalid invariants so the code agent can fix the implementation (Spec §9)
        self._z3_counterexample_prompt: str = ""
        # Repair Memory: lessons from past repair outcomes (set by SimulaService)
        self._repair_memory_prompt: str = ""
        # Stage 4A: Proof library prompt (set by SimulaService before each generate call)
        self._proof_library_prompt: str = ""
        # Token budget tracking for injected prompt sections (chars // 4 ≈ tokens).
        # Exposed so SimulaService can log budget consumption and detect overflow before
        # the LLM call.  Reset in generate() alongside _system_prompt_tokens.
        self._lilo_prompt_tokens: int = 0
        self._repair_memory_tokens: int = 0
        self._proof_library_tokens: int = 0
        # Stage 4B: GRPO fine-tuned model ID (set by SimulaService for A/B routing)
        self._grpo_model_id: str = ""
        # Stage 4B: GRPO engine reference (set by SimulaService for local model routing)
        self._grpo_engine: object | None = None
        # Stage 4B: Whether to use local model for this proposal (set per-proposal)
        self._use_local_model: bool = False
        # Memory: Neo4j client for arXiv paper abstract retrieval
        self._neo4j = neo4j
        # Populated per-proposal when source=="arxiv"; injected into system prompt
        self._arxiv_paper_abstract: str = ""
        # Organism health context: injected by SimulaService before each repair call.
        # Contains log-derived signals, Soma arousal, Fovea attention profile.
        self._organism_context: str = ""
        # External repo mode: set via set_external_workspace() before implement_external()
        self._external_workspace: object | None = None  # ExternalWorkspace | None

    # ─── External Workspace Mode ─────────────────────────────────────────────

    def set_external_workspace(self, workspace: object) -> None:
        """
        Redirect all file I/O to an external cloned repository.

        After calling this, _validate_path, _check_forbidden_path, and the
        test/lint tools all operate on the workspace root instead of EOS.
        Call clear_external_workspace() when done.
        """
        self._external_workspace = workspace
        # Redirect root so _validate_path enforces the workspace boundary
        self._root = workspace.root  # type: ignore[union-attr]

    def clear_external_workspace(self) -> None:
        """Restore internal EOS root after external task completes."""
        self._external_workspace = None

    def _should_use_extended_thinking(self, proposal: EvolutionProposal) -> bool:
        """
        Budget guard: route to extended-thinking model ONLY when:
          - RiskLevel >= HIGH (from simulation result), OR
          - Category is in GOVERNANCE_REQUIRED

        This prevents wasting expensive reasoning tokens on routine additive changes.
        """
        if self._thinking_llm is None:
            return False

        # Category-based routing: governance-required changes always get deep reasoning
        if proposal.category in GOVERNANCE_REQUIRED:
            return True

        # Risk-based routing: high-risk proposals get extended thinking
        if proposal.simulation is not None:
            if proposal.simulation.risk_level in (RiskLevel.HIGH, RiskLevel.UNACCEPTABLE):
                return True

        return False

    async def _try_local_model(
        self, proposal: EvolutionProposal,
    ) -> CodeChangeResult | None:
        """
        Attempt single-shot code generation via local GRPO-finetuned model.

        Returns CodeChangeResult on success, None to fall back to API.
        The local model gets one chance -- no agentic loop, no tool use.
        It must produce either unified diffs or complete file blocks.
        """

        engine: GRPOEngine = self._grpo_engine  # type: ignore[assignment]

        sys_prompt = (
            "You are EcodiaOS Simula, a code generation engine. "
            "Output ONLY the code changes as either:\n"
            "1. Unified diff blocks with file paths (--- a/path, +++ b/path), OR\n"
            "2. Complete file blocks with ### path/to/file headers\n"
            "No explanation needed -- just the code."
        )
        user_prompt = (
            f"Implement this change for the EcodiaOS codebase:\n\n"
            f"Category: {proposal.category.value}\n"
            f"Description: {proposal.description}\n\n"
            f"Change spec:\n{proposal.change_spec.model_dump_json(indent=2)}"
        )

        try:
            raw_output = await engine.generate_local(
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                max_tokens=4096,
            )
        except Exception as exc:
            self._logger.warning(
                "local_model_generate_failed", error=str(exc),
            )
            return None

        if not raw_output or len(raw_output.strip()) < 20:
            return None

        # Basic sanity: output should look like code, not prose
        code_signals = ("def ", "class ", "import ", "from ", "+++ ", "### ", "--- ")
        if not any(sig in raw_output for sig in code_signals):
            self._logger.info("local_model_output_not_code")
            return None

        # Try to parse and apply the generated code
        files_written = self._apply_local_diffs(raw_output)
        if not files_written:
            return None

        self._files_written.extend(files_written)

        return CodeChangeResult(
            success=True,
            files_written=self._files_written,
            summary=f"Implemented via local GRPO model (single-shot, {len(files_written)} files)",
            total_tokens=0,
            reasoning_tokens=0,
            used_extended_thinking=False,
        )

    def _apply_local_diffs(self, raw_output: str) -> list[str]:
        """
        Parse local model output into file writes.

        Supports two formats:
        1. Unified diff: lines starting with +++ b/path
        2. Markdown-style: ### path/to/file followed by code block
        """
        import re

        written: list[str] = []

        # Strategy 1: Unified diff blocks
        diff_pat = re.compile(
            r'\+\+\+ b/(.+?)\n(.*?)(?=\n--- |\n\+\+\+ |\Z)',
            re.DOTALL,
        )
        diff_matches = list(diff_pat.finditer(raw_output))

        if diff_matches:
            for m in diff_matches:
                rel_path = m.group(1).strip()
                diff_body = m.group(2)
                lines = []
                for line in diff_body.split('\n'):
                    if line.startswith('+') and not line.startswith('+++'):
                        lines.append(line[1:])
                    elif not line.startswith('-') and not line.startswith('@@'):
                        lines.append(line)

                content = '\n'.join(lines)
                if content.strip() and not self._check_forbidden_path(rel_path):
                    target = (self._root / rel_path).resolve()
                    if str(target).startswith(str(self._root)):
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(content, encoding="utf-8")
                        written.append(rel_path)
            return written

        # Strategy 2: Markdown code blocks with file headers
        block_pat = re.compile(
            r'###\s+(.+?)\s*\n```[a-z]*\n(.*?)```',
            re.DOTALL,
        )
        for m in block_pat.finditer(raw_output):
            rel_path = m.group(1).strip()
            content = m.group(2)
            if content.strip() and not self._check_forbidden_path(rel_path):
                target = (self._root / rel_path).resolve()
                if str(target).startswith(str(self._root)):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
                    written.append(rel_path)

        return written

    async def implement(
        self,
        proposal: EvolutionProposal,
        skip_test_writing: bool = False,
    ) -> CodeChangeResult:
        """
        Main entry point. Runs the agentic loop to implement the proposal.

        Routes to the extended-thinking model (o3/deepseek-r1) when the proposal
        is governance-required or high-risk, falling back to the standard model
        for routine additive changes. This budget guard ensures expensive reasoning
        tokens are only consumed when the change warrants deep analysis.

        Args:
            proposal: The evolution proposal to implement.
            skip_test_writing: If True, instructs the LLM to NOT write test files.
                Used in the AgentCoder pipeline where tests are handled by
                the TestDesigner agent separately.

        Returns CodeChangeResult with all files written and outcome.
        """
        self._files_written = []
        self._total_tokens_used = 0
        self._reasoning_tokens_used = 0
        self._system_prompt_tokens = 0
        self._lilo_prompt_tokens = 0
        self._repair_memory_tokens = 0
        self._proof_library_tokens = 0
        self._arxiv_paper_abstract = ""

        # Fetch paper abstract from memory graph for arXiv proposals so the
        # coding agent has the source theory in its context.
        if proposal.source == "arxiv":
            await self._fetch_arxiv_abstract(proposal)

        # Stage 4B: Try local GRPO model for routine tasks before falling back to API
        if self._use_local_model and self._grpo_engine is not None:
            local_result = await self._try_local_model(proposal)
            if local_result is not None:
                return local_result
            # Local model failed or produced poor output - fall through to API
            self._logger.info(
                "code_agent_local_fallback",
                proposal_id=proposal.id,
                reason="local model did not produce usable output",
            )

        # Determine model routing based on risk level and category
        use_thinking = self._should_use_extended_thinking(proposal)
        active_llm = self._thinking_llm if use_thinking else self._llm
        self._used_extended_thinking = use_thinking

        system_prompt = self._build_system_prompt(proposal)
        # #68: Estimate system prompt token budget (1 token ≈ 4 chars, conservative).
        # This is measured once so downstream callers can see how much of the
        # context window the static instructions consume before any turns.
        self._system_prompt_tokens = len(system_prompt) // 4

        # Stage 2D: When AgentCoder pipeline is active, disable test writing
        if skip_test_writing:
            system_prompt += (
                "\n\n## IMPORTANT: Test Writing Disabled\n"
                "Do NOT write test files. Tests are handled by a separate "
                "TestDesigner agent. Focus ONLY on the implementation code. "
                "Do not create any files under tests/."
            )

        # Prepend a planning instruction to encourage multi-file reasoning
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"Please implement this change: {proposal.description}\n\n"
                    f"Change spec details: {proposal.change_spec.model_dump_json(indent=2)}\n\n"
                    "IMPORTANT: Before writing any code, first:\n"
                    "1. Use find_similar to study an existing implementation like what you need to build\n"
                    "2. Use read_spec for the affected system to understand design intent\n"
                    "3. List every file you plan to create or modify and explain your approach\n"
                    "4. Then implement, lint, type-check, and test."
                ),
            }
        ]

        turns = 0
        last_text = ""

        self._logger.info(
            "code_agent_starting",
            proposal_id=proposal.id,
            category=proposal.category.value,
            max_turns=self._max_turns,
            tools_available=len(SIMULA_AGENT_TOOLS),
            extended_thinking=use_thinking,
            model_type="thinking" if use_thinking else "standard",
        )

        # Budget gate: code agent is STANDARD priority - skip in RED tier
        if self._optimized and not use_thinking:
            assert isinstance(self._llm, OptimizedLLMProvider)
            if not self._llm.should_use_llm("simula.code_agent", estimated_tokens=8000):
                self._logger.warning(
                    "code_agent_skipped_budget",
                    proposal_id=proposal.id,
                    tier=self._llm.get_budget_tier().value,
                )
                return CodeChangeResult(
                    success=False,
                    files_written=[],
                    error="LLM budget exhausted (RED tier) - code agent skipped.",
                    total_tokens=self._total_tokens_used,
                    system_prompt_tokens=self._system_prompt_tokens,
                )

        while turns < self._max_turns:
            turns += 1

            # KVzip: compress context before each LLM call (after turn 3
            # when tool results start accumulating). The compressor prunes
            # old tool results while preserving the recent sliding window.
            if turns > 3:
                messages = self._compressor.compress(messages)

            try:
                if use_thinking and isinstance(active_llm, ExtendedThinkingProvider):
                    response = await active_llm.generate_with_thinking_and_tools(
                        system_prompt=system_prompt,
                        messages=messages,
                        tools=SIMULA_AGENT_TOOLS,
                        max_tokens=8192,
                        reasoning_budget=self._thinking_budget,
                    )
                elif self._optimized and not use_thinking:
                    response = await self._llm.generate_with_tools(  # type: ignore[call-arg]
                        system_prompt=system_prompt,
                        messages=messages,
                        tools=SIMULA_AGENT_TOOLS,
                        max_tokens=8192,
                        temperature=0.2,
                        cache_system="simula.code_agent",
                    )
                else:
                    response = await active_llm.generate_with_tools(  # type: ignore[union-attr]
                        system_prompt=system_prompt,
                        messages=messages,
                        tools=SIMULA_AGENT_TOOLS,
                        max_tokens=8192,
                        temperature=0.2,
                    )
            except Exception as exc:
                self._logger.error("llm_call_failed", turn=turns, error=str(exc))
                return CodeChangeResult(
                    success=False,
                    files_written=self._files_written,
                    error=f"LLM call failed on turn {turns}: {exc}",
                )

            # Track token budget
            self._total_tokens_used += getattr(response, "total_tokens", 0)
            last_text = response.text

            if not response.has_tool_calls:
                self._logger.info(
                    "code_agent_done",
                    turns=turns,
                    files_written=len(self._files_written),
                    stop_reason=response.stop_reason,
                    total_tokens=self._total_tokens_used,
                )
                break

            # Build assistant message with text + tool_use blocks
            assistant_content: list[dict[str, Any]] = []
            if response.text:
                assistant_content.append({"type": "text", "text": response.text})
            for tc in response.tool_calls:
                assistant_content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                })
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute all tool calls
            tool_results: list[dict[str, Any]] = []
            for tc in response.tool_calls:
                result = await self._execute_tool(tc)
                tool_results.append(result.to_anthropic_dict())
                self._logger.debug(
                    "tool_executed",
                    tool=tc.name,
                    is_error=result.is_error,
                    turn=turns,
                )

            messages.append({"role": "user", "content": tool_results})

        else:
            self._logger.warning(
                "code_agent_max_turns_exceeded",
                max_turns=self._max_turns,
                files_written=len(self._files_written),
                total_tokens=self._total_tokens_used,
            )
            cm = self._compressor.metrics
            return CodeChangeResult(
                success=len(self._files_written) > 0,
                files_written=self._files_written,
                summary=last_text[:500] if last_text else "Max turns exceeded",
                error="Max turns exceeded without completion signal",
                total_tokens=self._total_tokens_used,
                system_prompt_tokens=self._system_prompt_tokens,
                kv_compression_ratio=cm.compression_ratio,
                kv_messages_compressed=cm.messages_compressed,
                kv_original_tokens=cm.original_tokens,
                kv_compressed_tokens=cm.compressed_tokens,
            )

        # ── Stage 2C: Static analysis post-generation gate ────────────────────
        static_fix_iterations = 0
        sa_result = None
        if self._files_written and self._static_bridge is not None:
            sa_result = await self._static_bridge.run_all(self._files_written)  # type: ignore[attr-defined]
            if sa_result.error_count > 0:
                from systems.simula.verification.static_analysis import (
                    StaticAnalysisBridge,
                )
                feedback_text = StaticAnalysisBridge.format_findings_for_feedback(
                    sa_result,
                )
                # Feed findings back to LLM for one fix iteration
                messages.append({
                    "role": "user",
                    "content": (
                        f"Static analysis found {sa_result.error_count} ERROR-severity issues "
                        f"in your written files. Fix them:\n\n{feedback_text}"
                    ),
                })
                # Run up to 3 fix iterations with exponential backoff.
                # Cap is enforced by min(config, 3) - avoids runaway retries.
                _max_fix = min(self._static_fix_max_iterations, 3)
                for _fix_turn in range(_max_fix):
                    if _fix_turn > 0:
                        await asyncio.sleep(2 ** _fix_turn + random.uniform(0, 1))
                    static_fix_iterations += 1
                    try:
                        response = await active_llm.generate_with_tools(  # type: ignore[union-attr]
                            system_prompt=system_prompt,
                            messages=messages,
                            tools=SIMULA_AGENT_TOOLS,
                            max_tokens=8192,
                            temperature=0.2,
                        )
                        self._total_tokens_used += getattr(response, "total_tokens", 0)
                        last_text = response.text

                        if response.has_tool_calls:
                            # Execute tool calls
                            assistant_content_fix: list[dict[str, Any]] = []
                            if response.text:
                                assistant_content_fix.append({"type": "text", "text": response.text})
                            for tc in response.tool_calls:
                                assistant_content_fix.append({
                                    "type": "tool_use", "id": tc.id,
                                    "name": tc.name, "input": tc.input,
                                })
                            messages.append({"role": "assistant", "content": assistant_content_fix})
                            tool_results_fix: list[dict[str, Any]] = []
                            for tc in response.tool_calls:
                                result = await self._execute_tool(tc)
                                tool_results_fix.append(result.to_anthropic_dict())
                            messages.append({"role": "user", "content": tool_results_fix})
                        else:
                            break
                    except Exception as exc:
                        self._logger.warning("static_fix_llm_error", error=str(exc))
                        break

                # Re-run static analysis to see if fixes worked
                sa_result = await self._static_bridge.run_all(self._files_written)  # type: ignore[attr-defined]
                if sa_result.error_count > 0:
                    self._logger.warning(
                        "static_analysis_fix_iterations_exhausted",
                        errors_remaining=sa_result.error_count,
                        fix_iterations=static_fix_iterations,
                        max_iterations=self._static_fix_max_iterations,
                    )
                else:
                    self._logger.info(
                        "static_analysis_post_fix",
                        errors_remaining=sa_result.error_count,
                        fix_iterations=static_fix_iterations,
                    )

        cm = self._compressor.metrics
        change_result = CodeChangeResult(
            success=len(self._files_written) > 0,
            files_written=self._files_written,
            summary=last_text[:1000] if last_text else "Change implemented",
            total_tokens=self._total_tokens_used,
            system_prompt_tokens=self._system_prompt_tokens,
            used_extended_thinking=self._used_extended_thinking,
            reasoning_tokens=self._reasoning_tokens_used,
            kv_compression_ratio=cm.compression_ratio,
            kv_messages_compressed=cm.messages_compressed,
            kv_original_tokens=cm.original_tokens,
            kv_compressed_tokens=cm.compressed_tokens,
            static_analysis_findings=(
                sa_result.error_count + sa_result.warning_count
                if self._static_bridge is not None and sa_result is not None
                else 0
            ),
            static_analysis_fix_iterations=static_fix_iterations,
        )
        return change_result

    async def implement_external(
        self,
        issue_description: str,
        workspace: object,
    ) -> CodeChangeResult:
        """
        Implement a fix for an external repository issue.

        Wraps implement() with external workspace mode: redirects all file I/O
        to the cloned repo, uses language-specific test/lint commands, enforces
        repo-specific forbidden paths, and injects repo context into the prompt.

        Args:
            issue_description: Human-readable description of what to fix/implement.
            workspace: ExternalWorkspace instance (already cloned).

        Returns CodeChangeResult with files_written relative to workspace root.
        """
        self.set_external_workspace(workspace)
        try:
            from systems.simula.evolution_types import ChangeCategory, ChangeSpec, EvolutionProposal
            ws = workspace  # type: ignore[assignment]
            lang = getattr(ws, "language", "unknown")
            repo_url = getattr(getattr(ws, "config", None), "repo_url", "unknown")
            target_files = getattr(getattr(ws, "config", None), "target_files", [])
            scope_note = (
                f"Target files: {', '.join(target_files)}" if target_files
                else "Full repository in scope"
            )
            proposal = EvolutionProposal(
                source="external_contractor",
                category=ChangeCategory.ADD_SYSTEM_CAPABILITY,
                description=issue_description,
                expected_benefit="Resolve external issue and pass all tests",
                evidence=["external_task"],
                change_spec=ChangeSpec(
                    capability_description=issue_description,
                    additional_context=(
                        f"## External Repository Task\n\n"
                        f"Language: {lang}\n"
                        f"Repository: {repo_url}\n"
                        f"{scope_note}\n\n"
                        "Do NOT modify build system or CI files. "
                        "Run the linter and tests to verify changes pass.\n\n"
                        f"Issue:\n{issue_description}"
                    ),
                ),
            )
            return await self.implement(proposal)
        finally:
            self.clear_external_workspace()

    # ─── Tool Dispatch ───────────────────────────────────────────────────────

    async def _execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Dispatch a tool call to the appropriate implementation."""
        try:
            match tool_call.name:
                case "read_file":
                    return await self._tool_read_file(tool_call)
                case "write_file":
                    return await self._tool_write_file(tool_call)
                case "diff_file":
                    return await self._tool_diff_file(tool_call)
                case "list_directory":
                    return await self._tool_list_directory(tool_call)
                case "search_code":
                    return await self._tool_search_code(tool_call)
                case "run_tests":
                    return await self._tool_run_tests(tool_call)
                case "run_linter":
                    return await self._tool_run_linter(tool_call)
                case "type_check":
                    return await self._tool_type_check(tool_call)
                case "dependency_graph":
                    return await self._tool_dependency_graph(tool_call)
                case "read_spec":
                    return await self._tool_read_spec(tool_call)
                case "find_similar":
                    return await self._tool_find_similar(tool_call)
                case _:
                    return ToolResult(
                        tool_use_id=tool_call.id,
                        content=f"Unknown tool: {tool_call.name}",
                        is_error=True,
                    )
        except Exception as exc:
            return ToolResult(
                tool_use_id=tool_call.id,
                content=f"Tool execution error: {exc}",
                is_error=True,
            )

    # ─── Original Tools (upgraded) ───────────────────────────────────────────

    async def _tool_read_file(self, tc: ToolCall) -> ToolResult:
        rel_path = tc.input.get("path", "")
        target, err = self._validate_path(rel_path)
        if target is None:
            return ToolResult(tc.id, err, True)
        try:
            content = target.read_text(encoding="utf-8")
            return ToolResult(tc.id, content)
        except FileNotFoundError:
            return ToolResult(tc.id, f"File not found: {rel_path}", True)
        except Exception as exc:
            return ToolResult(tc.id, f"Read error: {exc}", True)

    async def _tool_write_file(self, tc: ToolCall) -> ToolResult:
        rel_path = tc.input.get("path", "")
        content = tc.input.get("content", "")
        forbidden_check = self._check_forbidden_path(rel_path)
        if forbidden_check:
            return ToolResult(tc.id, forbidden_check, True)
        target, err = self._validate_path(rel_path)
        if target is None:
            return ToolResult(tc.id, err, True)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            if rel_path not in self._files_written:
                self._files_written.append(rel_path)
            return ToolResult(tc.id, f"Written: {rel_path} ({len(content)} bytes)")
        except Exception as exc:
            return ToolResult(tc.id, f"Write error: {exc}", True)

    async def _tool_list_directory(self, tc: ToolCall) -> ToolResult:
        rel_path = tc.input.get("path", "")
        if rel_path:
            target, err = self._validate_path(rel_path)
            if target is None:
                return ToolResult(tc.id, err, True)
        else:
            target = self._root
        try:
            if not target.exists():
                return ToolResult(tc.id, f"Directory not found: {rel_path}", True)
            entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
            lines = []
            for entry in entries:
                prefix = "  " if entry.is_file() else "D "
                lines.append(f"{prefix}{entry.name}")
            return ToolResult(tc.id, "\n".join(lines) or "(empty)")
        except Exception as exc:
            return ToolResult(tc.id, f"List error: {exc}", True)

    async def _tool_search_code(self, tc: ToolCall) -> ToolResult:
        pattern = tc.input.get("pattern", "")
        pat_err = self._validate_search_pattern(pattern)
        if pat_err:
            return ToolResult(tc.id, pat_err, True)
        directory = tc.input.get("directory", "ecodiaos/")
        search_root, dir_err = self._validate_path(directory or "ecodiaos/")
        if search_root is None:
            return ToolResult(tc.id, dir_err, True)
        # #71: bound both line count and per-line length to prevent context overflow
        _SEARCH_MAX_LINES = 50
        _SEARCH_LINE_MAX_CHARS = 200
        _SEARCH_TIMEOUT = 15.0
        results: list[str] = []
        try:
            proc = await asyncio.create_subprocess_exec(
                "grep", "-rn", "--include=*.py", pattern, str(search_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_SEARCH_TIMEOUT)
            output = stdout.decode("utf-8", errors="replace")
            all_lines = output.splitlines()
            for line in all_lines[:_SEARCH_MAX_LINES]:
                line = line.replace(str(self._root) + "/", "").replace(str(self._root) + "\\", "")
                if len(line) > _SEARCH_LINE_MAX_CHARS:
                    line = line[:_SEARCH_LINE_MAX_CHARS] + " [...]"
                results.append(line)
            if len(all_lines) > _SEARCH_MAX_LINES:
                results.append(f"[... {len(all_lines) - _SEARCH_MAX_LINES} more matches omitted ...]")
            return ToolResult(tc.id, "\n".join(results) if results else "No matches found")
        except TimeoutError:
            return ToolResult(tc.id, f"Search timed out after {_SEARCH_TIMEOUT:.0f}s", True)
        except Exception as exc:
            return ToolResult(tc.id, f"Search error: {exc}", True)

    async def _tool_run_tests(self, tc: ToolCall) -> ToolResult:
        # External mode: delegate to workspace language-aware test runner
        if self._external_workspace is not None:
            try:
                result = await self._external_workspace.run_tests()  # type: ignore[union-attr]
                lang = getattr(self._external_workspace, "language", "")
                label = f"{'PASSED' if result.passed else 'FAILED'} ({lang} / {result.command})"
                return ToolResult(
                    tc.id,
                    f"{label}\n{result.output[-2000:]}",
                    is_error=not result.passed,
                )
            except Exception as exc:
                return ToolResult(tc.id, f"Test run error: {exc}", True)
        test_path = tc.input.get("test_path", "")
        target, err = self._validate_path(test_path)
        if target is None:
            return ToolResult(tc.id, err, True)
        if not target.exists():
            return ToolResult(tc.id, f"Test path not found: {test_path}")
        try:
            proc = await asyncio.create_subprocess_exec(
                "pytest", str(target), "-x", "--tb=short", "-q",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self._root),
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            output = stdout.decode("utf-8", errors="replace")
            passed = proc.returncode == 0
            return ToolResult(
                tc.id,
                f"{'PASSED' if passed else 'FAILED'}\n{output[-2000:]}",
                is_error=not passed,
            )
        except TimeoutError:
            return ToolResult(tc.id, "Tests timed out after 60s", True)
        except Exception as exc:
            return ToolResult(tc.id, f"Test run error: {exc}", True)

    async def _tool_run_linter(self, tc: ToolCall) -> ToolResult:
        # External mode: delegate to workspace language-aware linter
        if self._external_workspace is not None:
            try:
                result = await self._external_workspace.run_linter()  # type: ignore[union-attr]
                lang = getattr(self._external_workspace, "language", "")
                label = "CLEAN" if result.passed else "ISSUES FOUND"
                return ToolResult(
                    tc.id,
                    f"{label} ({lang} / {result.command})\n{result.output[-2000:]}",
                    is_error=not result.passed,
                )
            except Exception as exc:
                return ToolResult(tc.id, f"Linter error: {exc}", True)
        import sys as _sys
        path = tc.input.get("path", "")
        target, err = self._validate_path(path)
        if target is None:
            return ToolResult(tc.id, err, True)
        # Prefer the venv-local ruff so we don't pick up a mismatched binary
        # from PATH (e.g. a Windows ruff.exe when running under WSL).
        _venv_ruff = Path(_sys.executable).parent / "ruff"
        _venv_ruff_exe = Path(_sys.executable).parent / "ruff.exe"
        if _venv_ruff.exists():
            _ruff = str(_venv_ruff)
        elif _venv_ruff_exe.exists():
            _ruff = str(_venv_ruff_exe)
        else:
            _ruff = "ruff"  # fall back to PATH
        # #52: explicit timeout constant; #69: cap output to avoid context overflow
        _LINTER_TIMEOUT = 20.0
        _LINTER_MAX_CHARS = 4000
        try:
            proc = await asyncio.create_subprocess_exec(
                _ruff, "check", str(target),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self._root),
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_LINTER_TIMEOUT)
            output = stdout.decode("utf-8", errors="replace")
            passed = proc.returncode == 0
            if len(output) > _LINTER_MAX_CHARS:
                output = (
                    output[:_LINTER_MAX_CHARS]
                    + f"\n[... {len(output) - _LINTER_MAX_CHARS} chars truncated ...]"
                )
            return ToolResult(
                tc.id,
                f"{'CLEAN' if passed else 'ISSUES FOUND'}\n{output}" if output else "CLEAN",
            )
        except TimeoutError:
            return ToolResult(tc.id, f"Linter timed out after {_LINTER_TIMEOUT:.0f}s", True)
        except Exception as exc:
            return ToolResult(tc.id, f"Linter error: {exc}", True)

    # ─── New Tools ───────────────────────────────────────────────────────────

    async def _tool_diff_file(self, tc: ToolCall) -> ToolResult:
        """Apply a targeted find/replace edit to a file."""
        rel_path = tc.input.get("path", "")
        find_text = tc.input.get("find", "")
        replace_text = tc.input.get("replace", "")

        if not find_text:
            return ToolResult(tc.id, "find parameter must not be empty", True)

        forbidden_check = self._check_forbidden_path(rel_path)
        if forbidden_check:
            return ToolResult(tc.id, forbidden_check, True)

        target, err = self._validate_path(rel_path)
        if target is None:
            return ToolResult(tc.id, err, True)
        if not target.exists():
            return ToolResult(tc.id, f"File not found: {rel_path}", True)

        try:
            content = target.read_text(encoding="utf-8")

            # #64: Normalise line endings before matching so that CRLF files and
            # LF find_text (or vice versa) don't cause spurious "not found" errors.
            # We work on the normalised form and rewrite with the file's original
            # line ending style so on-disk content is not inadvertently changed.
            original_has_crlf = "\r\n" in content
            content_norm = content.replace("\r\n", "\n")
            find_norm = find_text.replace("\r\n", "\n")

            if find_norm not in content_norm:
                return ToolResult(
                    tc.id,
                    f"Find text not found in {rel_path}. "
                    "Ensure the 'find' parameter matches existing content "
                    "(whitespace-normalised comparison was also attempted).",
                    True,
                )

            occurrences = content_norm.count(find_norm)
            if occurrences > 1:
                return ToolResult(
                    tc.id,
                    f"Find text matches {occurrences} locations in {rel_path}. "
                    "Provide more surrounding context to make the match unique.",
                    True,
                )

            replace_norm = replace_text.replace("\r\n", "\n")
            new_content_norm = content_norm.replace(find_norm, replace_norm, 1)
            # Restore the file's original line ending style
            if original_has_crlf:
                new_content = new_content_norm.replace("\n", "\r\n")
            else:
                new_content = new_content_norm
            target.write_text(new_content, encoding="utf-8")

            if rel_path not in self._files_written:
                self._files_written.append(rel_path)

            find_lines = find_norm.count("\n") + 1
            replace_lines = replace_norm.count("\n") + 1
            return ToolResult(
                tc.id,
                f"Edited {rel_path}: replaced {find_lines} line(s) with {replace_lines} line(s)",
            )
        except Exception as exc:
            return ToolResult(tc.id, f"Diff error: {exc}", True)

    async def _tool_type_check(self, tc: ToolCall) -> ToolResult:
        """Run mypy type checker on a path."""
        import sys as _sys
        path = tc.input.get("path", "")
        target, err = self._validate_path(path)
        if target is None:
            return ToolResult(tc.id, err, True)
        _bin_dir = Path(_sys.executable).parent
        _mypy = str(next(
            (p for p in [_bin_dir / "mypy", _bin_dir / "mypy.exe"] if p.exists()),
            "mypy",
        ))
        try:
            proc = await asyncio.create_subprocess_exec(
                _mypy, str(target), "--strict", "--no-error-summary",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(self._root),
            )
            # #52: explicit timeout constant; #70: cap to first 4000 chars (errors appear at top)
            _TC_TIMEOUT = 45.0
            _TC_MAX_CHARS = 4000
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=_TC_TIMEOUT)
            output = stdout.decode("utf-8", errors="replace")
            passed = proc.returncode == 0
            if passed:
                return ToolResult(tc.id, "TYPE CHECK PASSED - no issues found")
            if len(output) > _TC_MAX_CHARS:
                output = (
                    output[:_TC_MAX_CHARS]
                    + f"\n[... {len(output) - _TC_MAX_CHARS} chars truncated ...]"
                )
            return ToolResult(
                tc.id,
                f"TYPE CHECK ISSUES:\n{output}",
                is_error=True,
            )
        except TimeoutError:
            return ToolResult(tc.id, f"Type check timed out after {_TC_TIMEOUT:.0f}s", True)
        except FileNotFoundError:
            return ToolResult(tc.id, "mypy not found - type checking unavailable")
        except Exception as exc:
            return ToolResult(tc.id, f"Type check error: {exc}", True)

    async def _tool_dependency_graph(self, tc: ToolCall) -> ToolResult:
        """Show what a module imports and what imports it."""
        module_path = tc.input.get("module_path", "")
        target, err = self._validate_path(module_path)
        if target is None:
            return ToolResult(tc.id, err, True)
        if not target.exists():
            return ToolResult(tc.id, f"File not found: {module_path}", True)

        try:
            source = target.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=module_path)
        except Exception as exc:
            return ToolResult(tc.id, f"Parse error: {exc}", True)

        # Extract this module's imports
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = ", ".join(a.name for a in (node.names or []))
                imports.append(f"from {node.module} import {names}")

        # Find files that import this module
        module_name = self._path_to_module(module_path)
        importers: list[str] = []
        if module_name:
            src_dir = self._root / "src"
            if src_dir.exists():
                short_parts = module_name.split(".")
                # Search for imports of this module
                for py_file in src_dir.rglob("*.py"):
                    if py_file.resolve() == target:
                        continue
                    try:
                        file_source = py_file.read_text(encoding="utf-8")
                        # Quick string check before expensive parse
                        if module_name not in file_source and short_parts[-1] not in file_source:
                            continue
                        file_tree = ast.parse(file_source)
                        for node in ast.walk(file_tree):
                            if isinstance(node, ast.ImportFrom) and node.module:
                                if module_name in node.module or (
                                    ".".join(short_parts[:-1]) in node.module
                                    and any(a.name == short_parts[-1] for a in (node.names or []))
                                ):
                                    importers.append(str(py_file.relative_to(self._root)))
                                    break
                            elif isinstance(node, ast.Import):
                                for alias in node.names:
                                    if module_name in alias.name:
                                        importers.append(str(py_file.relative_to(self._root)))
                                        break
                    except Exception:
                        continue

        lines = [f"=== Dependency Graph for {module_path} ===\n"]
        lines.append(f"Module: {module_name or 'unknown'}\n")
        lines.append(f"--- This module imports ({len(imports)}) ---")
        for imp in imports:
            lines.append(f"  {imp}")
        lines.append(f"\n--- Imported by ({len(importers)}) ---")
        for imp in importers:
            lines.append(f"  {imp}")

        return ToolResult(tc.id, "\n".join(lines))

    async def _tool_read_spec(self, tc: ToolCall) -> ToolResult:
        """Read an EcodiaOS specification document."""
        spec_name = tc.input.get("spec_name", "").lower().strip()
        spec_file = _SPEC_FILE_MAP.get(spec_name)

        if spec_file is None:
            available = ", ".join(sorted(_SPEC_FILE_MAP.keys()))
            return ToolResult(
                tc.id,
                f"Unknown spec: {spec_name!r}. Available: {available}",
                True,
            )

        target = self._spec_root / spec_file
        if not target.exists():
            return ToolResult(tc.id, f"Spec file not found: {spec_file}", True)

        try:
            content = target.read_text(encoding="utf-8")
            # Truncate to 4000 chars to stay within token budget
            if len(content) > 4000:
                content = content[:4000] + "\n\n[... truncated - use read_file for full content ...]"
            return ToolResult(tc.id, content)
        except Exception as exc:
            return ToolResult(tc.id, f"Read error: {exc}", True)

    async def _build_code_index(self) -> dict[str, list[float]]:
        """
        Lazy-build a semantic index of Python files in the codebase.

        Embeds the first ~500 chars of each Python file (module docstring +
        imports + top-level definitions) to create a searchable code index.
        Cached for the lifetime of this agent instance.
        """
        async with self._code_index_lock:
            if self._code_index is not None:
                return self._code_index

            if self._embedding is None:
                self._code_index = {}
                return self._code_index

            # Collect Python files (skip __pycache__, .venv, tests, migrations)
            skip_dirs = {"__pycache__", ".venv", "venv", "node_modules", ".git", "migrations"}
            py_files: list[tuple[str, str]] = []  # (rel_path, summary_text)

            for py_file in self._root.rglob("*.py"):
                # Skip excluded directories
                if any(part in skip_dirs for part in py_file.parts):
                    continue
                rel = str(py_file.relative_to(self._root)).replace("\\", "/")
                try:
                    content = py_file.read_text(encoding="utf-8")
                    # Take module-level summary: docstring + first 500 chars
                    summary = f"File: {rel}\n{content[:500]}"
                    py_files.append((rel, summary))
                except Exception:
                    continue

            if not py_files:
                self._code_index = {}
                return self._code_index

            # Embed in batches
            paths = [p for p, _ in py_files]
            texts = [t for _, t in py_files]

            try:
                embeddings = await self._embedding.embed_batch(texts)
                self._code_index = dict(zip(paths, embeddings, strict=False))
                self._logger.info(
                    "code_index_built",
                    files_indexed=len(self._code_index),
                )
            except Exception as exc:
                self._logger.warning("code_index_build_failed", error=str(exc))
                self._code_index = {}

            return self._code_index

    async def _semantic_find_similar(
        self, description: str, top_k: int = 5, threshold: float = 0.4
    ) -> list[tuple[str, float]]:
        """
        Find files semantically similar to the description using embeddings.

        Returns list of (rel_path, similarity_score) sorted by score descending.
        Uses embed_query() for Voyage clients (query-optimized) or embed() otherwise.
        """
        if self._embedding is None:
            return []

        code_index = await self._build_code_index()
        if not code_index:
            return []

        # Embed the query - use query-optimized encoding for Voyage
        # 5s timeout: a hung embedding call must not stall the code agent
        try:
            if isinstance(self._embedding, VoyageEmbeddingClient):
                query_vec = await asyncio.wait_for(
                    self._embedding.embed_query(description), timeout=5.0
                )
            else:
                query_vec = await asyncio.wait_for(
                    self._embedding.embed(description), timeout=5.0
                )
        except (asyncio.TimeoutError, TimeoutError):
            self._logger.warning("semantic_search_embed_timeout")
            return []
        except Exception as exc:
            self._logger.warning("semantic_search_embed_failed", error=str(exc))
            return []

        # Compute similarities and rank
        scored: list[tuple[str, float]] = []
        for path, doc_vec in code_index.items():
            sim = cosine_similarity(query_vec, doc_vec)
            if sim >= threshold:
                scored.append((path, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    async def _tool_find_similar(self, tc: ToolCall) -> ToolResult:
        """Find existing implementations similar to what needs to be built.

        Two-tier search:
          1. Keyword matching against _SIMILAR_CODE_MAP (fast, exact)
          2. Semantic embedding search via voyage-code-3 (deep, fuzzy)
        Falls back from tier 1 → tier 2 when keywords don't match.
        """
        description = tc.input.get("description", "").lower()

        # ── Tier 1: Keyword matching ─────────────────────────────────────────
        matched_paths: list[str] = []
        for keyword, paths in _SIMILAR_CODE_MAP.items():
            if keyword in description:
                matched_paths.extend(paths)
                break

        if not matched_paths:
            words = description.split()
            for word in words:
                if len(word) > 3:
                    for keyword, paths in _SIMILAR_CODE_MAP.items():
                        if word in keyword or keyword in word:
                            matched_paths.extend(paths)
                            break
                if matched_paths:
                    break

        # ── Tier 2: Semantic embedding search ────────────────────────────────
        semantic_paths: list[tuple[str, float]] = []
        if not matched_paths and self._embedding is not None:
            semantic_paths = await self._semantic_find_similar(description)
            matched_paths = [p for p, _ in semantic_paths]

        if not matched_paths:
            return ToolResult(
                tc.id,
                "No similar implementations found. Try search_code with a specific pattern.",
            )

        # ── Read matched files ───────────────────────────────────────────────
        results: list[str] = []
        chars_remaining = 4000

        # If semantic search was used, prepend similarity scores
        if semantic_paths:
            score_header = "Semantic similarity results:\n" + "\n".join(
                f"  {p} (score: {s:.3f})" for p, s in semantic_paths
            )
            results.append(score_header)
            chars_remaining -= len(score_header)

        for rel_path in matched_paths:
            if chars_remaining <= 0:
                break
            target = self._root / rel_path
            if target.is_file():
                try:
                    content = target.read_text(encoding="utf-8")
                    chunk = content[:min(2500, chars_remaining)]
                    results.append(f"=== {rel_path} ===\n{chunk}")
                    chars_remaining -= len(results[-1])
                except Exception:
                    continue
            elif target.is_dir():
                try:
                    py_files = sorted(target.glob("*.py"))
                    file_list = ", ".join(f.name for f in py_files)
                    results.append(f"=== {rel_path} ===\nFiles: {file_list}")
                    chars_remaining -= len(results[-1])

                    for py_file in py_files:
                        if py_file.name == "__init__.py" or chars_remaining <= 0:
                            continue
                        content = py_file.read_text(encoding="utf-8")
                        chunk = content[:min(2000, chars_remaining)]
                        rel = str(py_file.relative_to(self._root))
                        results.append(f"\n=== {rel} (exemplar) ===\n{chunk}")
                        chars_remaining -= len(results[-1])
                        break
                except Exception:
                    continue

        return ToolResult(tc.id, "\n\n".join(results) if results else "No files found at matched paths")

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _validate_path(self, rel_path: str) -> tuple[Path, str] | tuple[None, str]:
        """
        Validate a tool-supplied path and resolve it within the codebase root.

        Catches:
          - Empty paths
          - Null bytes (used in some path-injection attacks)
          - Path traversal after resolution (e.g. ../../etc/passwd)

        Returns (resolved_path, "") on success or (None, error_message) on failure.
        This is the single gate used by all tools that accept a path (#48).
        """
        if not rel_path or not rel_path.strip():
            return None, "Path must not be empty"
        if "\x00" in rel_path:
            return None, "Path contains null byte - rejected"
        try:
            target = (self._root / rel_path).resolve()
        except Exception as exc:
            return None, f"Invalid path: {exc}"
        root_str = str(self._root)
        target_str = str(target)
        # Ensure resolved path is strictly inside (or equal to) the codebase root.
        # Use os.sep-aware prefix check to avoid false matches like /root vs /rootdir.
        if target_str != root_str and not target_str.startswith(root_str + "/") and not target_str.startswith(root_str + "\\"):
            return None, "Access denied: path outside codebase root"
        return target, ""

    def _validate_search_pattern(self, pattern: str) -> str | None:
        """
        Validate a search pattern for the search_code tool (#48).

        Rejects empty patterns and patterns that are excessively long (which
        could cause catastrophic backtracking in grep or be used to probe
        the filesystem via timed side-channels).

        Returns error message or None if valid.
        """
        if not pattern or not pattern.strip():
            return "Search pattern must not be empty"
        if len(pattern) > 500:
            return "Search pattern too long (max 500 chars)"
        return None

    def _check_forbidden_path(self, rel_path: str) -> str | None:
        """Check if a path is forbidden. Returns error message or None."""
        if self._external_workspace is not None:
            # External mode: delegate to workspace boundary enforcement
            try:
                target = (self._root / rel_path).resolve()
                self._external_workspace.assert_write_allowed(target)  # type: ignore[union-attr]
            except Exception as exc:
                return f"WRITE DENIED: {exc}"
            return None
        from systems.simula.evolution_types import FORBIDDEN_WRITE_PATHS
        for forbidden in FORBIDDEN_WRITE_PATHS:
            if rel_path.startswith(forbidden) or forbidden in rel_path:
                return (
                    f"IRON RULE VIOLATION: Cannot write to forbidden path '{rel_path}' "
                    f"(matches forbidden pattern '{forbidden}'). "
                    "This change would violate Simula's constitutional constraints."
                )
        return None

    def _path_to_module(self, rel_path: str) -> str | None:
        """Convert a relative file path to a dotted module name."""
        parts = rel_path.replace("\\", "/").split("/")
        if parts and parts[0] == "src":
            parts = parts[1:]
        if not parts:
            return None
        if parts[-1].endswith(".py"):
            parts[-1] = parts[-1][:-3]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts) if parts else None

    # ─── Bounty Hunter Integration ────────────────────────────────────────────

    async def _push_to_github(
        self,
        proposal: EvolutionProposal,
        files_written: list[str],
    ) -> tuple[str, int | None]:
        """
        Push the code change as a cross-repository GitHub PR.

        Because the bot cannot push directly to repos it doesn't own, the
        sequence is:
          1. gh auth status         - confirm GH_TOKEN is active
          2. gh repo fork --remote  - fork to bot account, adds 'fork' remote
          3. git checkout -b        - create the bounty branch locally
          4. git add / commit       - stage and commit written files
          5. git push -u fork       - push to the bot's fork
          6. gh pr create --repo    - open PR against the *original* repo

        The GitHub token is resolved from (in priority order):
          GH_TOKEN -> GITHUB_TOKEN -> ECODIAOS_EXTERNAL_PLATFORMS__GITHUB_TOKEN

        Returns:
            (pr_url, pr_number) on success, ("", None) on failure.
        """
        import os

        bounty_id   = proposal.source_bounty_id or "unknown"
        branch_name = f"bounty/gh-{bounty_id}"
        commit_msg  = f"Fix: {proposal.description[:120]}"
        repo_url    = proposal.target_repository_url or ""

        # Derive "owner/repo" from the clone URL for --repo flag
        original_repo = ""
        if repo_url:
            cleaned = repo_url.rstrip("/").removesuffix(".git")
            parts   = cleaned.split("/")
            if len(parts) >= 2:
                original_repo = "/".join(parts[-2:])

        # Resolve GitHub token - gh CLI reads GH_TOKEN automatically
        github_token = (
            os.environ.get("GH_TOKEN")
            or os.environ.get("GITHUB_TOKEN")
            or os.environ.get("ECODIAOS_EXTERNAL_PLATFORMS__GITHUB_TOKEN")
            or ""
        )
        env = {**os.environ, "GH_TOKEN": github_token} if github_token else dict(os.environ)

        async def _run(
            *cmd: str,
            cwd: str | None = None,
            timeout: float = 60.0,
        ) -> tuple[int, str, str]:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd or str(self._root),
                env=env,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
            return (
                proc.returncode or 0,
                stdout_bytes.decode("utf-8", errors="replace"),
                stderr_bytes.decode("utf-8", errors="replace"),
            )

        try:
            # 0. Confirm gh CLI is authenticated
            rc, out, err = await _run("gh", "auth", "status")
            if rc != 0:
                self._logger.warning("gh_auth_status_failed", stderr=err.strip())

            # 1. Fork the upstream repo and add a 'fork' remote
            rc, out, err = await _run(
                "gh", "repo", "fork", "--remote=true", "--clone=false",
                timeout=60.0,
            )
            if rc != 0:
                self._logger.warning(
                    "gh_repo_fork_failed",
                    stderr=err.strip(),
                    stdout=out.strip(),
                )

            # Discover which remote name gh assigned to the fork
            rc_remote, remotes_out, _ = await _run("git", "remote")
            remote_names = remotes_out.strip().splitlines()
            fork_remote = "fork" if "fork" in remote_names else "origin"
            self._logger.info(
                "fork_remote_selected",
                fork_remote=fork_remote,
                remotes=remote_names,
            )

            # 2. Create the bounty branch
            rc, _, err = await _run("git", "checkout", "-b", branch_name)
            if rc != 0:
                self._logger.warning("git_checkout_new_branch_failed", stderr=err.strip())
                rc2, _, err2 = await _run("git", "checkout", branch_name)
                if rc2 != 0:
                    self._logger.error("git_checkout_existing_failed", stderr=err2.strip())
                    return ("", None)

            # 3. Stage the written files
            if files_written:
                rc, _, err = await _run("git", "add", "--", *files_written)
            else:
                rc, _, err = await _run("git", "add", "-A")
            if rc != 0:
                self._logger.error("git_add_failed", stderr=err.strip())
                return ("", None)

            # 4. Commit
            rc, _, err = await _run(
                "git", "commit", "-m", commit_msg,
                "--author=EcodiaOS Bot <bot@ecodiaos.ai>",
            )
            if rc != 0:
                self._logger.error("git_commit_failed", stderr=err.strip())
                return ("", None)

            # 5. Push to the bot's fork
            rc, _, err = await _run(
                "git", "push", "-u", fork_remote, branch_name,
                timeout=60.0,
            )
            if rc != 0:
                self._logger.error(
                    "git_push_failed",
                    remote=fork_remote,
                    branch=branch_name,
                    stderr=err.strip(),
                )
                return ("", None)

            # 6. Open a PR against the *original* repo
            pr_body = (
                f"## Bounty Fix: `{bounty_id}`\n\n"
                f"{proposal.description}\n\n"
                f"**Files changed:** {len(files_written)}\n"
                + "\n".join(f"- `{f}`" for f in files_written)
            )
            pr_cmd = [
                "gh", "pr", "create",
                "--title", commit_msg,
                "--body", pr_body,
            ]
            if original_repo:
                pr_cmd += ["--repo", original_repo]

            rc, stdout, err = await _run(*pr_cmd, timeout=30.0)
            if rc != 0:
                self._logger.error(
                    "gh_pr_create_failed",
                    stderr=err.strip(),
                    stdout=stdout.strip(),
                    repo=original_repo,
                )
                return ("", None)

            pr_url = stdout.strip()

            # Extract PR number from URL (e.g. https://github.com/org/repo/pull/42)
            pr_number: int | None = None
            if "/pull/" in pr_url:
                with contextlib.suppress(ValueError):
                    pr_number = int(pr_url.rstrip("/").rsplit("/", 1)[-1])

            self._logger.info(
                "github_pr_created",
                bounty_id=bounty_id,
                pr_url=pr_url,
                pr_number=pr_number,
                branch=branch_name,
                repo=original_repo,
            )
            return (pr_url, pr_number)

        except TimeoutError:
            self._logger.error("push_to_github_timeout", bounty_id=bounty_id)
            return ("", None)
        except Exception as exc:
            self._logger.error(
                "push_to_github_error", bounty_id=bounty_id, error=str(exc),
            )
            return ("", None)

    async def _fetch_arxiv_abstract(self, proposal: EvolutionProposal) -> None:
        """
        Query the Neo4j memory graph for the paper abstract linked to this
        arXiv proposal.  Populates ``self._arxiv_paper_abstract`` so
        ``_build_system_prompt`` can inject it.

        Silently skips on any error - graceful degradation.
        """
        if self._neo4j is None:
            return

        from systems.simula.proposals.arxiv_translator import (
            ArxivProposalTranslator,
        )
        from systems.simula.proposals.paper_memory import (
            get_paper_abstract_for_technique,
        )

        technique_name = ArxivProposalTranslator._extract_technique_name(
            proposal.description
        )
        try:
            abstract = await get_paper_abstract_for_technique(
                self._neo4j,  # type: ignore[arg-type]
                technique_name,
            )
            if abstract:
                self._arxiv_paper_abstract = abstract
                self._logger.info(
                    "arxiv_abstract_injected",
                    proposal_id=proposal.id,
                    technique=technique_name,
                    abstract_chars=len(abstract),
                )
            else:
                self._logger.debug(
                    "arxiv_abstract_not_found",
                    proposal_id=proposal.id,
                    technique=technique_name,
                )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "arxiv_abstract_fetch_failed",
                proposal_id=proposal.id,
                error=str(exc),
            )

    def _build_system_prompt(self, proposal: EvolutionProposal) -> str:
        from systems.simula.evolution_types import (
            FORBIDDEN_WRITE_PATHS,
            SIMULA_IRON_RULES,
            ChangeCategory,
        )

        architecture_context = _build_architecture_context(
            category=proposal.category,
            codebase_root=self._root,
            spec_root=self._spec_root,
        )

        prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            category=proposal.category.value,
            description=proposal.description,
            expected_benefit=proposal.expected_benefit,
            evidence=", ".join(proposal.evidence) or "none",
            iron_rules="\n".join(f"- {r}" for r in SIMULA_IRON_RULES),
            forbidden_paths="\n".join(f"- {p}" for p in FORBIDDEN_WRITE_PATHS),
            architecture_context=architecture_context,
        )

        # BUG_FIX injection: attach the exact error details so the agent can fix it
        if proposal.category == ChangeCategory.BUG_FIX:
            error_context = (
                "\n\n## Runtime Error Details\n\n"
                "This is a low-blast-radius bug fix. Focus narrowly on the exact error:\n\n"
                f"**Error Type**: {proposal.change_spec.additional_context.split('Incident')[0]}\n\n"
                f"**Full Context**:\n{proposal.change_spec.additional_context}\n\n"
                "Your job is to read the error message, understand what's missing or wrong, "
                "and fix ONLY what is needed to resolve the error. Do NOT make unrelated changes. "
                "Do NOT refactor code. Do NOT add features. Surgical precision is critical.\n\n"
            )

            # Special guidance for AttributeError on method calls
            if 'AttributeError' in proposal.change_spec.additional_context:
                error_context += (
                    "## AttributeError Fix Pattern\n\n"
                    "This error indicates a missing method call. **Do NOT add a new method** - "
                    "instead, search the target class for methods with similar names.\n\n"
                    "The correct method almost certainly already exists. Your job:\n"
                    "1. Find the class where the error occurs (read the stack trace)\n"
                    "2. Read the actual class definition\n"
                    "3. Find the real method name that should be called instead\n"
                    "4. Replace the incorrect call with the correct method name\n\n"
                    "Example: if error is 'OptimizedLLMProvider' object has no attribute 'complete', "
                    "search OptimizedLLMProvider for methods like 'invoke', 'call', 'execute', 'run' - "
                    "one of those is the real method. Replace the call, do not add a new method.\n\n"
                )

            prompt += error_context

        # arXiv memory injection: give the coding agent the paper abstract so
        # it understands the math/theory behind what it is implementing.
        if self._arxiv_paper_abstract:
            prompt += (
                "\n\n## Source Research Paper (arXiv)\n"
                "This proposal originates from a research paper. "
                "Read the abstract below carefully before writing any code - "
                "understand the underlying algorithm, data structures, and "
                "theoretical guarantees before translating them into Python.\n\n"
                f"```\n{self._arxiv_paper_abstract}\n```"
            )

        # Repair Memory: Append lessons from past repairs (escaped to prevent injection)
        # Budget: cap at 4 000 tokens (~16 000 chars) to avoid crowding the context window.
        _SECTION_TOKEN_BUDGET = 4_000
        _SECTION_CHAR_BUDGET = _SECTION_TOKEN_BUDGET * 4
        repair_memory_text = _escape_prompt_injection(self._repair_memory_prompt)
        if len(repair_memory_text) > _SECTION_CHAR_BUDGET:
            repair_memory_text = repair_memory_text[:_SECTION_CHAR_BUDGET] + "\n...[truncated]"
        self._repair_memory_tokens = len(repair_memory_text) // 4
        if repair_memory_text:
            prompt += f"\n\n{repair_memory_text}"

        # Stage 3C: Append LILO library abstractions if available (escaped)
        # Budget: cap at 4 000 tokens to leave room for counterfactual context.
        lilo_text = _escape_prompt_injection(self._lilo_prompt)
        if len(lilo_text) > _SECTION_CHAR_BUDGET:
            lilo_text = lilo_text[:_SECTION_CHAR_BUDGET] + "\n...[truncated]"
        self._lilo_prompt_tokens = len(lilo_text) // 4
        if lilo_text:
            prompt += f"\n\n{lilo_text}"

        # Stage 4A: Append proof library context if available (escaped)
        # Budget: cap at 4 000 tokens; proofs are dense but LILO has higher priority.
        proof_text = _escape_prompt_injection(self._proof_library_prompt)
        if len(proof_text) > _SECTION_CHAR_BUDGET:
            proof_text = proof_text[:_SECTION_CHAR_BUDGET] + "\n...[truncated]"
        self._proof_library_tokens = len(proof_text) // 4
        if proof_text:
            prompt += proof_text

        # Organism health: inject current system health state so the agent
        # understands whether the organism is under pressure (cascades, high arousal,
        # high Fovea salience) and can prioritise accordingly. (escaped)
        if self._organism_context:
            prompt += f"\n\n{_escape_prompt_injection(self._organism_context)}"

        # Z3 counterexample feedback (Spec §9): when formal verification found invalid
        # invariants in the written code, inject the counterexamples so the agent
        # understands which boundary conditions it violated and can fix them.
        if self._z3_counterexample_prompt:
            prompt += f"\n\n{_escape_prompt_injection(self._z3_counterexample_prompt)}"

        # External repo context: inject workspace details so the agent understands
        # it is operating in a foreign codebase, not the EOS internal codebase.
        if self._external_workspace is not None:
            ws = self._external_workspace
            lang = getattr(ws, "language", "unknown")
            repo_url = getattr(getattr(ws, "config", None), "repo_url", "unknown")
            target_files = getattr(getattr(ws, "config", None), "target_files", [])
            forbidden = getattr(ws, "_forbidden", set())
            scope_note = (
                f"Target files: {', '.join(target_files)}" if target_files
                else "All files in scope (no target_files filter set)"
            )
            forbidden_note = "\n".join(f"- {p}" for p in sorted(forbidden)) or "none"
            prompt += (
                f"\n\n## External Repository Mode\n\n"
                f"You are working in a **cloned external repository**, NOT the EcodiaOS codebase.\n\n"
                f"- Language: `{lang}`\n"
                f"- Repository: `{repo_url}`\n"
                f"- Workspace root: `{ws.root}`\n"
                f"- {scope_note}\n\n"
                f"**Forbidden infrastructure files (never modify):**\n{forbidden_note}\n\n"
                f"Use `run_tests` and `run_linter` (language-aware) to verify your changes. "
                f"All paths are relative to the workspace root. "
                f"This is a PR contribution - write clean, idiomatic code in the repo's language. "
                f"Do NOT add EOS-specific imports, patterns, or Synapse bus calls."
            )

        return prompt
