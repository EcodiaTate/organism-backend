"""
EcodiaOS -- Simula Diffusion-Based Code Repair Agent (Stage 4C)

Last-mile code repair via iterative denoising. When the standard code
agent (CEGIS loop) exhausts max iterations without passing tests,
the diffusion repair agent takes over.

Two modes:
  1. Iterative Denoising (DiffuCoder-style):
     Start from the broken code, progressively denoise by fixing
     one category of error per step until tests pass.

  2. Sketch-First (Tree Diffusion-style):
     Generate a code skeleton (structure without implementation detail),
     then fill in the implementation using the standard code agent.

Integration point: service.py calls diffusion repair after code_agent
has failed `diffusion_handoff_after_failures` times.

References:
  - DiffuCoder (7B): iterative denoising for code repair
  - Tree Diffusion: structure-aware diffusion for code generation
"""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from clients.llm import Message
from systems.simula.verification.types import (
    DiffusionDenoiseStep,
    DiffusionRepairResult,
    DiffusionRepairStatus,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from systems.simula.evolution_types import EvolutionProposal

logger = structlog.get_logger().bind(system="simula.agents.diffusion_repair")


# ── System Prompts ──────────────────────────────────────────────────────────

DENOISE_SYSTEM_PROMPT = """EcodiaOS diffusion repair — iterative denoising. Fix broken Python code one error category at a time until tests pass.

EOS conventions: Python 3.12+, Pydantic, structlog, async/await, type hints. Import paths: from systems.<system>.<module> import <class>.

Output the COMPLETE fixed file(s) in fenced blocks (# path/to/file.py as first line). Fix one category of error per step."""


SKETCH_SYSTEM_PROMPT = """EcodiaOS diffusion repair — sketch phase. Generate a code skeleton (structure only) for the proposed change.

EOS conventions: Python 3.12+, Pydantic, structlog. All signatures with type hints, docstrings, `...` bodies. The code agent fills implementations.

Output complete skeleton file(s) in fenced blocks (# path/to/file.py as first line)."""


# ── DiffusionRepairAgent ───────────────────────────────────────────────────


class DiffusionRepairAgent:
    """
    Diffusion-based code repair for last-mile fixes.

    After the standard code agent exhausts its iterations, this agent
    takes the broken code and progressively repairs it through
    iterative denoising steps.

    Each step targets one error category, building on the previous
    step's output. The process continues until tests pass or max
    steps are reached.

    Flow:
      repair()       - full iterative denoising or sketch-first repair
      denoise_step() - one step of iterative repair
      generate_sketch() - generate code skeleton (sketch-first mode)
    """

    def __init__(
        self,
        llm: LLMProvider,
        codebase_root: Path,
        test_command: str = "pytest",
        max_denoise_steps: int = 10,
        timeout_s: float = 120.0,
        sketch_first: bool = False,
    ) -> None:
        self._llm = llm
        self._root = codebase_root
        self._test_command = test_command
        self._max_steps = max_denoise_steps
        self._timeout_s = timeout_s
        self._sketch_first = sketch_first
        self._log = logger

    async def repair(
        self,
        proposal: EvolutionProposal,
        broken_files: dict[str, str],  # path -> content
        test_output: str = "",
        lint_output: str = "",
    ) -> DiffusionRepairResult:
        """
        Full repair pipeline: iterative denoising or sketch-first.

        Args:
            proposal: The evolution proposal that failed.
            broken_files: The broken code (path -> content).
            test_output: Test failure output from the code agent.
            lint_output: Lint error output from the code agent.

        Returns:
            DiffusionRepairResult with repair status and metrics.
        """
        start = time.monotonic()

        if self._sketch_first:
            return await self._repair_sketch_first(
                proposal, broken_files, test_output, lint_output, start,
            )
        else:
            return await self._repair_iterative_denoise(
                proposal, broken_files, test_output, lint_output, start,
            )

    # ── Iterative Denoising ─────────────────────────────────────────────

    async def _repair_iterative_denoise(
        self,
        proposal: EvolutionProposal,
        broken_files: dict[str, str],
        test_output: str,
        lint_output: str,
        start: float,
    ) -> DiffusionRepairResult:
        """
        DiffuCoder-style iterative denoising repair.

        Each step fixes one category of error, progressively
        improving the code until tests pass.
        """
        result = DiffusionRepairResult(
            mode="iterative_denoise",
            original_code=self._files_to_text(broken_files),
        )

        current_files = dict(broken_files)
        current_errors = f"Test output:\n{test_output}\n\nLint output:\n{lint_output}"

        # Count initial test state
        tests_before = self._count_passing_tests(test_output)
        result.tests_passed_before = tests_before

        for step_num in range(1, self._max_steps + 1):
            # Check timeout
            elapsed = time.monotonic() - start
            if elapsed > self._timeout_s:
                self._log.warning("diffusion_timeout", step=step_num, elapsed_s=elapsed)
                result.status = DiffusionRepairStatus.TIMEOUT
                result.error_summary = f"Repair timed out after {elapsed:.0f}s"
                break

            self._log.info(
                "diffusion_denoise_step",
                step=step_num,
                max_steps=self._max_steps,
            )

            # Run one denoising step
            step_result = await self._denoise_step(
                step_num, current_files, current_errors, proposal,
            )
            result.denoise_steps.append(step_result)
            result.total_llm_tokens += step_result.tokens_used

            # Parse repaired files from the step output
            repaired_files = self._parse_code_blocks(step_result.code_snapshot)
            if repaired_files:
                current_files.update(repaired_files)

            # Write repaired files and run tests
            await self._write_files(current_files)
            test_passed, test_out = await self._run_tests(current_files)
            lint_clean, lint_out = await self._run_lint(current_files)

            # Update step metrics
            tests_now = self._count_passing_tests(test_out)
            step_result.tests_passed = tests_now
            step_result.tests_total = self._count_total_tests(test_out)
            step_result.lint_errors = self._count_lint_errors(lint_out)
            step_result.improvement_delta = (
                (tests_now - tests_before) / max(1, step_result.tests_total)
            )
            step_result.noise_level = 1.0 - (step_num / self._max_steps)

            if test_passed and lint_clean:
                result.status = DiffusionRepairStatus.REPAIRED
                result.repaired_code = self._files_to_text(current_files)
                result.files_repaired = list(current_files.keys())
                result.tests_passed_after = tests_now
                result.tests_total = step_result.tests_total
                result.lint_clean = True
                result.repair_success = True
                result.improvement_rate = tests_now / max(1, step_result.tests_total)
                self._log.info(
                    "diffusion_repair_success",
                    steps=step_num,
                    tests_passed=tests_now,
                )
                break

            # Update error context for next step
            current_errors = f"Test output:\n{test_out}\n\nLint output:\n{lint_out}"

        else:
            # Exhausted all steps
            result.status = DiffusionRepairStatus.FAILED
            # Check if there was partial improvement
            if result.denoise_steps:
                last = result.denoise_steps[-1]
                if last.tests_passed > tests_before:
                    result.status = DiffusionRepairStatus.PARTIAL
                    result.repaired_code = self._files_to_text(current_files)
                    result.files_repaired = list(current_files.keys())
                    result.tests_passed_after = last.tests_passed
                    result.tests_total = last.tests_total
                    result.improvement_rate = (
                        last.tests_passed / max(1, last.tests_total)
                    )
            result.error_summary = (
                f"Repair did not fully converge after {self._max_steps} steps"
            )

        result.total_steps = len(result.denoise_steps)
        result.total_time_ms = int((time.monotonic() - start) * 1000)
        return result

    async def _denoise_step(
        self,
        step_num: int,
        current_files: dict[str, str],
        current_errors: str,
        proposal: EvolutionProposal,
    ) -> DiffusionDenoiseStep:
        """
        Execute one denoising step: fix one category of error.
        """
        # Build prompt with current code and errors
        code_context = self._files_to_text(current_files)
        prompt = (
            f"## Step {step_num}/{self._max_steps}: Iterative Repair\n\n"
            f"## Change Specification\n"
            f"Category: {proposal.category.value}\n"
            f"Description: {proposal.description}\n\n"
            f"## Current Code (may have errors)\n"
            f"```python\n{code_context[:8000]}\n```\n\n"
            f"## Current Errors\n{current_errors[:4000]}\n\n"
            f"Fix ONE category of error in this step. Output the complete "
            f"corrected file(s) in fenced code blocks."
        )

        try:
            response = await self._llm.generate(
                system_prompt=DENOISE_SYSTEM_PROMPT,
                messages=[Message(role="user", content=prompt)],
                max_tokens=8192,
                temperature=0.3,
            )

            return DiffusionDenoiseStep(
                step_number=step_num,
                code_snapshot=response.text,
                tokens_used=getattr(response, "total_tokens", 0),
            )

        except Exception as exc:
            self._log.error(
                "diffusion_denoise_error",
                step=step_num,
                error=str(exc),
            )
            return DiffusionDenoiseStep(
                step_number=step_num,
                code_snapshot="",
            )

    # ── Sketch-First Mode ───────────────────────────────────────────────

    async def _repair_sketch_first(
        self,
        proposal: EvolutionProposal,
        broken_files: dict[str, str],
        test_output: str,
        lint_output: str,
        start: float,
    ) -> DiffusionRepairResult:
        """
        Tree Diffusion-style sketch-first repair.

        Phase 1: Generate a code skeleton (correct structure, ... bodies)
        Phase 2: Let the standard code agent fill implementations

        The skeleton constrains the structure, preventing the code agent
        from taking wrong architectural directions.
        """
        result = DiffusionRepairResult(
            mode="sketch_first",
            original_code=self._files_to_text(broken_files),
        )

        # Phase 1: Generate skeleton
        self._log.info("diffusion_sketch_phase_start")

        sketch_prompt = (
            f"## Generate Skeleton for Repair\n\n"
            f"The following code failed tests. Generate a corrected SKELETON:\n\n"
            f"## Change Specification\n"
            f"Category: {proposal.category.value}\n"
            f"Description: {proposal.description}\n\n"
            f"## Broken Code\n"
            f"```python\n{self._files_to_text(broken_files)[:6000]}\n```\n\n"
            f"## Test Failures\n{test_output[:3000]}\n\n"
            f"Generate a corrected skeleton with all signatures, types, "
            f"and docstrings correct. Use `...` for function bodies."
        )

        try:
            response = await self._llm.generate(
                system_prompt=SKETCH_SYSTEM_PROMPT,
                messages=[Message(role="user", content=sketch_prompt)],
                max_tokens=8192,
                temperature=0.2,
            )
        except Exception as exc:
            result.status = DiffusionRepairStatus.FAILED
            result.error_summary = f"Sketch generation failed: {exc}"
            result.total_time_ms = int((time.monotonic() - start) * 1000)
            return result

        skeleton_files = self._parse_code_blocks(response.text)
        result.total_llm_tokens += getattr(response, "total_tokens", 0)

        sketch_step = DiffusionDenoiseStep(
            step_number=0,
            noise_level=1.0,
            code_snapshot=response.text,
            tokens_used=getattr(response, "total_tokens", 0),
        )
        result.denoise_steps.append(sketch_step)

        if not skeleton_files:
            result.status = DiffusionRepairStatus.FAILED
            result.error_summary = "Failed to parse skeleton from LLM output"
            result.total_time_ms = int((time.monotonic() - start) * 1000)
            return result

        # Phase 2: Fill skeleton via iterative denoising
        # The skeleton constrains the structure - now fill implementations
        self._log.info(
            "diffusion_fill_phase_start",
            skeleton_files=len(skeleton_files),
        )

        fill_result = await self._repair_iterative_denoise(
            proposal, skeleton_files, test_output, lint_output, start,
        )

        # Merge results
        result.status = fill_result.status
        result.denoise_steps.extend(fill_result.denoise_steps)
        result.total_steps = len(result.denoise_steps)
        result.repaired_code = fill_result.repaired_code
        result.files_repaired = fill_result.files_repaired
        result.tests_passed_before = fill_result.tests_passed_before
        result.tests_passed_after = fill_result.tests_passed_after
        result.tests_total = fill_result.tests_total
        result.lint_clean = fill_result.lint_clean
        result.repair_success = fill_result.repair_success
        result.improvement_rate = fill_result.improvement_rate
        result.total_llm_tokens += fill_result.total_llm_tokens
        result.total_time_ms = int((time.monotonic() - start) * 1000)

        return result

    # ── Verification Helpers ───────────────────────────────────────────────

    async def _write_files(self, files: dict[str, str]) -> None:
        """Write repaired files to the codebase."""
        for rel_path, content in files.items():
            full_path = self._root / rel_path
            try:
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content, encoding="utf-8")
            except Exception as exc:
                self._log.warning(
                    "diffusion_write_failed",
                    path=rel_path,
                    error=str(exc),
                )

    async def _run_tests(
        self, files: dict[str, str],
    ) -> tuple[bool, str]:
        """Run tests for the repaired files."""
        # Derive test path from file paths
        test_paths: set[str] = set()
        for rel_path in files:
            parts = Path(rel_path).parts
            if len(parts) >= 4 and parts[0] == "src" and parts[2] == "systems":
                system_name = parts[3]
                test_dir = self._root / "tests" / "unit" / "systems" / system_name
                if test_dir.is_dir():
                    test_paths.add(str(test_dir))

        if not test_paths:
            return True, "no tests found"

        # Run pytest on the first matching test directory
        test_path = next(iter(test_paths))
        try:
            proc = await asyncio.create_subprocess_exec(
                self._test_command, test_path, "-x", "--tb=short", "-q",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self._root),
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=30.0,
                )
            except TimeoutError:
                proc.kill()
                await proc.communicate()
                return False, "Test run timed out after 30s"

            output = stdout.decode("utf-8", errors="replace")
            return proc.returncode == 0, output

        except FileNotFoundError:
            return False, f"Test command {self._test_command!r} not found"
        except Exception as exc:
            return False, f"Test error: {exc}"

    async def _run_lint(
        self, files: dict[str, str],
    ) -> tuple[bool, str]:
        """Run linter on repaired files."""
        py_files = [
            str(self._root / p)
            for p in files
            if p.endswith(".py")
        ]
        if not py_files:
            return True, ""

        try:
            proc = await asyncio.create_subprocess_exec(
                "ruff", "check", *py_files,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self._root),
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=15.0,
                )
            except TimeoutError:
                proc.kill()
                await proc.communicate()
                return False, "Lint timed out"

            output = stdout.decode("utf-8", errors="replace")
            return proc.returncode == 0, output

        except FileNotFoundError:
            return True, "ruff not available"
        except Exception as exc:
            return False, f"Lint error: {exc}"

    # ── Parsing Helpers ────────────────────────────────────────────────────

    def _parse_code_blocks(self, text: str) -> dict[str, str]:
        """
        Parse file contents from fenced code blocks.

        Supports two formats:
          ```python\n# path/to/file.py\n...\n```
          ```path/to/file.py\n...\n```
        """
        files: dict[str, str] = {}

        # Pattern 1: ```python with path comment on first line
        pattern1 = re.compile(
            r"```python\s*\n#\s*([\w/._-]+\.py)\s*\n(.*?)```",
            re.DOTALL,
        )
        for match in pattern1.finditer(text):
            path = match.group(1).strip()
            content = match.group(2).strip() + "\n"
            files[path] = content

        # Pattern 2: ```<path> as language tag
        pattern2 = re.compile(
            r"```([\w/._-]+\.py)\s*\n(.*?)```",
            re.DOTALL,
        )
        for match in pattern2.finditer(text):
            path = match.group(1).strip()
            if path not in files:  # don't override pattern 1 matches
                content = match.group(2).strip() + "\n"
                files[path] = content

        return files

    def _files_to_text(self, files: dict[str, str]) -> str:
        """Convert file dict to a text representation."""
        parts: list[str] = []
        for path, content in files.items():
            parts.append(f"# --- {path} ---\n{content}")
        return "\n\n".join(parts)

    def _count_passing_tests(self, test_output: str) -> int:
        """Count passing tests from pytest output."""
        # Look for "N passed" in pytest output
        match = re.search(r"(\d+)\s+passed", test_output)
        return int(match.group(1)) if match else 0

    def _count_total_tests(self, test_output: str) -> int:
        """Count total tests from pytest output."""
        total = 0
        for pattern in [r"(\d+)\s+passed", r"(\d+)\s+failed", r"(\d+)\s+error"]:
            match = re.search(pattern, test_output)
            if match:
                total += int(match.group(1))
        return max(1, total)

    def _count_lint_errors(self, lint_output: str) -> int:
        """Count lint errors from ruff output."""
        # ruff outputs one line per finding
        if not lint_output.strip():
            return 0
        return len([
            line for line in lint_output.splitlines()
            if line.strip() and not line.startswith("Found")
        ])
