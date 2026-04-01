"""
EcodiaOS -- Simula TestDesigner Agent (Stage 2D)

Generates comprehensive test files for a proposal WITHOUT seeing the
implementation. This is the first agent in the AgentCoder pipeline:

  TestDesigner → Coder → TestExecutor → iterate

The adversarial separation ensures tests aren't biased toward the
implementation - they test the specification, not the code.

The agent uses a read-only subset of the code agent's tools:
  - read_file: read existing codebase files for context
  - list_directory: explore project structure
  - search_code: find existing test patterns
  - read_spec: read EOS specification docs
  - find_similar: locate similar implementations

It does NOT have write_file, diff_file, run_tests, or run_linter.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import TYPE_CHECKING, Any
from pathlib import Path

import structlog

from clients.llm import (
    LLMProvider,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from systems.simula.verification.types import TestDesignResult

if TYPE_CHECKING:

    from systems.simula.evolution_types import EvolutionProposal
logger = structlog.get_logger().bind(system="simula.agents.test_designer")


# ── Read-Only Tool Definitions ───────────────────────────────────────────────

_TEST_DESIGNER_TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="read_file",
        description=(
            "Read a file from the EcodiaOS codebase. "
            "Use to understand existing code, tests, and patterns."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path from codebase root",
                },
            },
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="list_directory",
        description="List files and subdirectories at a given path.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path from codebase root",
                },
            },
        },
    ),
    ToolDefinition(
        name="search_code",
        description=(
            "Search for a pattern across codebase Python files. "
            "Returns matching lines with file paths and line numbers."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "String pattern to search for (case-sensitive)",
                },
                "directory": {
                    "type": "string",
                    "description": "Directory to search in (default: ecodiaos/)",
                },
            },
            "required": ["pattern"],
        },
    ),
    ToolDefinition(
        name="read_spec",
        description=(
            "Read an EcodiaOS specification document. "
            "Use to understand design intent, interfaces, and constraints."
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
            "Find existing implementations similar to what you need. "
            "Returns relevant code examples from the codebase."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What you're looking for (e.g., 'test for executor')",
                },
            },
            "required": ["description"],
        },
    ),
]


# Spec name → file path mapping (matches code_agent.py)
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


_SYSTEM_PROMPT = """You are the TestDesigner agent - part of EcodiaOS Simula's AgentCoder pipeline.

Your job is to generate comprehensive test files for a proposed change WITHOUT seeing
the implementation. You test the specification, not the code. This adversarial separation
produces higher-quality tests.

## Your Task
Category: {category}
Description: {description}
Expected benefit: {expected_benefit}
Change specification: {change_spec}

## EcodiaOS Test Conventions
- Python 3.12+, pytest as test runner
- Async tests: use @pytest.mark.asyncio and async def test_*()
- Fixtures: conftest.py at tests/unit/systems/<system>/
- Pydantic models: test with .model_validate() for schema compliance
- Mock external services: use unittest.mock.AsyncMock for async dependencies
- structlog: capture logs with structlog.testing.capture_logs()
- Imports: from systems.<system>.<module> import <class>
- Naming: test_<module>.py files, test_<behavior>() functions
- Group related tests in classes: class TestFeatureName:
- Edge cases: empty inputs, None values, boundary values, error paths

## Process
1. Study existing test files in the codebase to understand patterns
2. Read the specification for the affected system
3. Identify the key behaviors, edge cases, and invariants to test
4. Generate test files that:
   - Test happy paths for all specified behaviors
   - Test error paths and boundary conditions
   - Test integration points between components
   - Use appropriate mocking for external dependencies
   - Follow existing test conventions exactly

## Output Format
After exploring the codebase, output your test files in fenced code blocks with the
file path as the language tag:

```tests/unit/systems/<system>/test_<module>.py
import pytest
...
```

Each test file should be complete, runnable, and follow EOS conventions.
List the coverage targets (functions/methods being tested) at the end.
"""


class TestDesignerAgent:
    """
    Generates test files for a proposal without seeing the implementation.

    Uses a read-only tool set to explore the codebase for patterns and
    conventions, then generates comprehensive test files via LLM.
    """

    def __init__(
        self,
        llm: LLMProvider,
        codebase_root: Path,
        max_turns: int = 12,
    ) -> None:
        self._llm = llm
        self._root = codebase_root.resolve()
        self._max_turns = max_turns
        self._log = logger

    async def design_tests(
        self, proposal: EvolutionProposal,
    ) -> TestDesignResult:
        """
        Generate test files for the given proposal.

        Runs an agentic tool-use loop: the LLM explores the codebase
        with read-only tools, then outputs test files as fenced code blocks.
        """
        start = time.monotonic()
        total_tokens = 0

        # Build the system prompt with proposal context
        change_spec_text = ""
        if proposal.change_spec:
            affected = ", ".join(proposal.change_spec.affected_systems) or "N/A"
            change_spec_text = (
                f"Affected systems: {affected}\n"
                f"Additional context: {proposal.change_spec.additional_context}"
            )

        system_prompt = _SYSTEM_PROMPT.format(
            category=proposal.category.value,
            description=proposal.description,
            expected_benefit=proposal.expected_benefit,
            change_spec=change_spec_text,
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    "Design comprehensive tests for this proposal. "
                    "Start by exploring the codebase to understand existing "
                    "test patterns, then generate your test files."
                ),
            },
        ]

        # Agentic tool-use loop
        turns_used = 0
        for _turn in range(self._max_turns):
            turns_used += 1
            response = await self._llm.generate_with_tools(
                system_prompt=system_prompt,
                messages=messages,
                tools=_TEST_DESIGNER_TOOLS,
            )
            total_tokens += getattr(response, "tokens_used", 0)

            # Check for tool calls
            if not response.tool_calls:
                # LLM is done - extract test files from the response
                break

            # Execute tool calls and feed results back
            tool_results = await self._execute_tools(response.tool_calls)
            messages.append({
                "role": "assistant",
                "content": response.text,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "input": tc.input}
                    for tc in response.tool_calls
                ],
            })
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tr.tool_use_id,
                        "content": tr.content,
                        "is_error": tr.is_error,
                    }
                    for tr in tool_results
                ],
            })
        else:
            self._log.warning(
                "test_designer_max_turns",
                turns=self._max_turns,
            )

        # Parse test files from the final response
        test_files = self._parse_test_files(response.text)
        test_count = self._count_tests(test_files)
        coverage_targets = self._extract_coverage_targets(response.text)

        result = TestDesignResult(
            test_files=test_files,
            test_count=test_count,
            coverage_targets=coverage_targets,
            design_reasoning=self._extract_reasoning(response.text),
            llm_tokens_used=total_tokens,
        )

        self._log.info(
            "test_design_complete",
            test_files=len(test_files),
            test_count=test_count,
            coverage_targets=len(coverage_targets),
            turns=turns_used,
            elapsed_ms=int((time.monotonic() - start) * 1000),
        )
        return result

    async def _execute_tools(
        self, tool_calls: list[ToolCall],
    ) -> list[ToolResult]:
        """Execute read-only tool calls."""
        results: list[ToolResult] = []
        for tc in tool_calls:
            content = await self._dispatch_tool(tc.name, tc.input)
            results.append(ToolResult(
                tool_use_id=tc.id,
                content=content,
                is_error=content.startswith("Error:"),
            ))
        return results

    async def _dispatch_tool(
        self, name: str, args: dict[str, Any],
    ) -> str:
        """Dispatch a single tool call to its handler."""
        if name == "read_file":
            return self._tool_read_file(args.get("path", ""))
        elif name == "list_directory":
            return self._tool_list_directory(args.get("path", ""))
        elif name == "search_code":
            return await self._tool_search_code(
                args.get("pattern", ""),
                args.get("directory", "ecodiaos/"),
            )
        elif name == "read_spec":
            return self._tool_read_spec(args.get("spec_name", ""))
        elif name == "find_similar":
            return self._tool_find_similar(args.get("description", ""))
        else:
            return f"Error: Unknown tool '{name}'"

    def _tool_read_file(self, path: str) -> str:
        """Read a file from the codebase."""
        if not path:
            return "Error: path is required"
        full = self._root / path
        if not full.is_file():
            return f"Error: File not found: {path}"
        try:
            content = full.read_text(encoding="utf-8")
            # Truncate to prevent context overflow
            if len(content) > 15000:
                content = content[:15000] + "\n... (truncated)"
            return content
        except Exception as exc:
            return f"Error: {exc}"

    def _tool_list_directory(self, path: str) -> str:
        """List directory contents."""
        target = self._root / (path or "")
        if not target.is_dir():
            return f"Error: Directory not found: {path}"
        try:
            entries: list[str] = []
            for item in sorted(target.iterdir()):
                suffix = "/" if item.is_dir() else ""
                entries.append(f"  {item.name}{suffix}")
            return "\n".join(entries) if entries else "(empty directory)"
        except Exception as exc:
            return f"Error: {exc}"

    async def _tool_search_code(
        self, pattern: str, directory: str,
    ) -> str:
        """Search for a pattern in Python files."""
        if not pattern:
            return "Error: pattern is required"
        search_dir = self._root / directory
        if not search_dir.is_dir():
            return f"Error: Directory not found: {directory}"
        try:
            proc = await asyncio.create_subprocess_exec(
                "grep", "-rn", "--include=*.py", pattern, str(search_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            output = stdout.decode("utf-8", errors="replace")
            # Limit output size
            lines = output.splitlines()[:50]
            if len(lines) == 50:
                lines.append("... (truncated, showing first 50 matches)")
            return "\n".join(lines) if lines else "No matches found."
        except TimeoutError:
            return "Error: Search timed out"
        except FileNotFoundError:
            # grep not available - fall back to Python search
            return await self._python_search(pattern, search_dir)
        except Exception as exc:
            return f"Error: {exc}"

    async def _python_search(self, pattern: str, search_dir: Path) -> str:
        """Fallback search when grep is not available."""
        results: list[str] = []
        try:
            for py_file in search_dir.rglob("*.py"):
                try:
                    text = py_file.read_text(encoding="utf-8")
                    for i, line in enumerate(text.splitlines(), 1):
                        if pattern in line:
                            rel = py_file.relative_to(self._root)
                            results.append(f"{rel}:{i}: {line.strip()}")
                            if len(results) >= 50:
                                results.append("... (truncated)")
                                return "\n".join(results)
                except Exception:
                    continue
        except Exception as exc:
            return f"Error: {exc}"
        return "\n".join(results) if results else "No matches found."

    def _tool_read_spec(self, spec_name: str) -> str:
        """Read an EOS specification document."""
        if not spec_name:
            return "Error: spec_name is required"
        file_path = _SPEC_FILE_MAP.get(spec_name.lower())
        if not file_path:
            return f"Error: Unknown spec '{spec_name}'. Available: {', '.join(_SPEC_FILE_MAP)}"
        full = self._root / file_path
        if not full.is_file():
            return f"Error: Spec file not found: {file_path}"
        try:
            content = full.read_text(encoding="utf-8")
            if len(content) > 10000:
                content = content[:10000] + "\n... (truncated)"
            return content
        except Exception as exc:
            return f"Error: {exc}"

    def _tool_find_similar(self, description: str) -> str:
        """Find similar implementations by keyword matching."""
        if not description:
            return "Error: description is required"

        # Simple keyword matching against test directories
        test_dirs = [
            "tests/unit/systems/",
            "tests/integration/",
        ]
        results: list[str] = []
        desc_lower = description.lower()
        for test_dir_str in test_dirs:
            test_dir = self._root / test_dir_str
            if not test_dir.is_dir():
                continue
            for py_file in test_dir.rglob("*.py"):
                name_lower = py_file.name.lower()
                if any(kw in name_lower for kw in desc_lower.split()):
                    rel = py_file.relative_to(self._root)
                    try:
                        content = py_file.read_text(encoding="utf-8")
                        # Show first 2000 chars as exemplar
                        preview = content[:2000]
                        results.append(
                            f"### {rel}\n```python\n{preview}\n```"
                        )
                    except Exception:
                        results.append(f"### {rel} (could not read)")
                    if len(results) >= 3:
                        break

        if not results:
            return "No similar test files found. Check tests/ directory structure."
        return "\n\n".join(results)

    @staticmethod
    def _parse_test_files(text: str) -> dict[str, str]:
        """
        Extract test files from fenced code blocks in the LLM response.

        Expects blocks like:
            ```tests/unit/systems/axon/test_executor.py
            import pytest
            ...
            ```
        """
        files: dict[str, str] = {}
        # Match fenced code blocks where the language tag is a file path
        pattern = re.compile(
            r"```(tests/[^\s`]+\.py)\s*\n(.*?)```",
            re.DOTALL,
        )
        for match in pattern.finditer(text):
            file_path = match.group(1).strip()
            content = match.group(2).strip() + "\n"
            files[file_path] = content
        return files

    @staticmethod
    def _count_tests(test_files: dict[str, str]) -> int:
        """Count test functions across all generated test files."""
        count = 0
        for content in test_files.values():
            # Count def test_ and async def test_ lines
            count += len(re.findall(
                r"(?:async\s+)?def\s+test_\w+", content,
            ))
        return count

    @staticmethod
    def _extract_coverage_targets(text: str) -> list[str]:
        """Extract coverage targets listed in the response."""
        targets: list[str] = []
        # Look for a "Coverage targets:" or "Functions tested:" section
        pattern = re.compile(
            r"(?:coverage targets|functions? tested|methods? tested)"
            r"[:\s]*\n((?:\s*[-*]\s*.+\n)+)",
            re.IGNORECASE,
        )
        match = pattern.search(text)
        if match:
            for line in match.group(1).splitlines():
                line = line.strip().lstrip("-*").strip()
                if line:
                    targets.append(line)
        return targets

    @staticmethod
    def _extract_reasoning(text: str) -> str:
        """Extract the design reasoning from the response (first paragraph)."""
        # Take text before the first code block as reasoning
        idx = text.find("```")
        reasoning = text[:idx].strip() if idx > 0 else text[:500].strip()
        # Limit to 1000 chars
        if len(reasoning) > 1000:
            reasoning = reasoning[:1000] + "..."
        return reasoning
