"""
EcodiaOS -- Simula Robustness Test Generator (Stage 6B.2)

Continuous self-improvement on idle compute via robustness test generation.

The robustness tester uses the LLM to generate edge-case tests that
target historical failure patterns. Tests that find bugs become failure cases
for GRPO training, closing the self-improvement loop.

Metric tracked: self-generated test coverage growth (6B.4).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from systems.simula.verification.types import (
    FailureCaseExample,
    RobustnessTestResult,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider

logger = structlog.get_logger().bind(system="simula.coevolution.robustness_tester")


_ROBUSTNESS_TEST_PROMPT = """\
Generate pytest robustness test functions targeting edge cases and failure modes.

Files to test: {files}

Known failure patterns from history:
{failure_patterns}

Target: boundary conditions, known failure patterns, malformed/extreme inputs, error handling, race conditions.

Output ONLY valid Python pytest test code with all necessary imports.
Each test function must be independent and self-contained.
"""


class RobustnessTestGenerator:
    """Generates robustness tests from failure patterns and runs them."""

    def __init__(
        self,
        llm: LLMProvider,
        codebase_root: Path,
        *,
        max_tests_per_cycle: int = 20,
        timeout_s: float = 120.0,
    ) -> None:
        self._llm = llm
        self._root = codebase_root
        self._max_tests = max_tests_per_cycle
        self._timeout_s = timeout_s
        self._total_tests_generated: int = 0
        self._total_bugs_found: int = 0

    # ── Public API ──────────────────────────────────────────────────────────

    async def generate_robustness_tests(
        self,
        files: list[str],
        past_failures: list[FailureCaseExample] | None = None,
    ) -> RobustnessTestResult:
        """
        Generate edge-case tests targeting historical failure patterns.

        1. Collect failure patterns from past hard negatives
        2. LLM generates adversarial test code
        3. Write tests to temp file and execute with pytest
        4. Collect results: which tests found bugs
        """
        start = time.monotonic()

        # Build failure pattern summary
        failure_patterns = ""
        if past_failures:
            patterns = [
                f"- {neg.category}: {neg.failure_reason}"
                for neg in past_failures[:10]  # limit context
            ]
            failure_patterns = "\n".join(patterns)
        else:
            failure_patterns = "- No known failure patterns (generate exploratory tests)"

        # Generate test code via LLM
        prompt = _ROBUSTNESS_TEST_PROMPT.format(
            files=", ".join(files[:5]),  # limit context
            failure_patterns=failure_patterns,
        )

        try:
            from clients.llm import Message

            response = await self._llm.complete(  # type: ignore[attr-defined]
                system=None,
                messages=[Message(role="user", content=prompt)],
                max_tokens=4096,
            )
            test_code = response.content if hasattr(response, "content") else str(response)

            # Strip markdown fences if present
            test_code = re.sub(r"```python\n?", "", test_code)
            test_code = re.sub(r"```\n?", "", test_code)

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.warning("robustness_generation_failed", error=str(exc))
            return RobustnessTestResult(duration_ms=elapsed_ms)

        # Write and run tests
        result = await self._run_tests(test_code)

        # Update lifetime stats
        self._total_tests_generated += result.tests_generated
        self._total_bugs_found += result.tests_found_bugs

        elapsed_ms = int((time.monotonic() - start) * 1000)
        result.duration_ms = elapsed_ms

        logger.info(
            "robustness_tests_complete",
            generated=result.tests_generated,
            executed=result.tests_executed,
            bugs_found=result.tests_found_bugs,
            duration_ms=elapsed_ms,
        )
        return result

    async def get_coverage_growth(self) -> float:
        """
        Return cumulative coverage growth from adversarial tests.

        This is a simplified metric: ratio of bugs found to tests generated.
        A higher ratio means the adversarial tests are more effective.
        """
        if self._total_tests_generated == 0:
            return 0.0
        return self._total_bugs_found / self._total_tests_generated

    # ── Private: Test execution ─────────────────────────────────────────────

    async def _run_tests(self, test_code: str) -> RobustnessTestResult:
        """Write adversarial tests to a temp file and run with pytest."""
        test_files: list[str] = []
        bugs: list[str] = []

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix="_adversarial_test.py",
                delete=False,
                dir=str(self._root),
                prefix="test_",
            ) as f:
                f.write(test_code)
                test_path = f.name
                test_files.append(test_path)

            # Count test functions
            import ast

            try:
                tree = ast.parse(test_code)
                test_count = sum(
                    1
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name.startswith("test_")
                )
            except SyntaxError:
                test_count = 0
                return RobustnessTestResult(
                    tests_generated=0,
                    test_files_written=test_files,
                    bug_descriptions=["Generated test code has syntax errors"],
                )

            # Run pytest
            proc = await asyncio.create_subprocess_exec(
                "python", "-m", "pytest", test_path, "--tb=short", "-q",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._root),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._timeout_s,
                )
            except TimeoutError:
                proc.kill()
                return RobustnessTestResult(
                    tests_generated=test_count,
                    tests_executed=0,
                    test_files_written=test_files,
                    bug_descriptions=["Test execution timed out"],
                )

            output = stdout.decode("utf-8", errors="replace")

            # Parse pytest output for failures
            executed = 0
            failed = 0
            for line in output.splitlines():
                if "passed" in line or "failed" in line:
                    import re

                    pass_match = re.search(r"(\d+) passed", line)
                    fail_match = re.search(r"(\d+) failed", line)
                    if pass_match:
                        executed += int(pass_match.group(1))
                    if fail_match:
                        failed += int(fail_match.group(1))
                        executed += int(fail_match.group(1))

                if "FAILED" in line:
                    bugs.append(line.strip())

            return RobustnessTestResult(
                tests_generated=test_count,
                tests_executed=executed,
                tests_found_bugs=failed,
                test_files_written=test_files,
                bug_descriptions=bugs,
            )

        except Exception as exc:
            logger.warning("robustness_test_execution_failed", error=str(exc))
            return RobustnessTestResult(
                test_files_written=test_files,
                bug_descriptions=[f"Execution error: {exc}"],
            )
        finally:
            # Clean up temp test file
            for tf in test_files:
                with contextlib.suppress(OSError):
                    Path(tf).unlink(missing_ok=True)
