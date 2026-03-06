"""
EcodiaOS -- Simula Dafny Bridge (Stage 2A)

Subprocess wrapper for the Dafny verifier, implementing the Clover
(Closed-Loop Verifiable Code Generation) pattern:

  1. LLM generates Dafny specification + implementation from Python code
  2. Dafny verifier checks the spec+impl pair
  3. If verification fails, errors are fed back to the LLM
  4. Iterate up to max_rounds (default: 8)

The Dafny binary must be available at the configured path or on $PATH.
Install via: dotnet tool install --global dafny

Reference: Sun et al., "Clover: Closed-Loop Verifiable Code Generation"
"""

from __future__ import annotations

import asyncio
import re
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from clients.llm import Message
from systems.simula.verification.types import (
    CloverRoundResult,
    DafnyVerificationResult,
    DafnyVerificationStatus,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider

logger = structlog.get_logger().bind(system="simula.verification.dafny")


# ── Clover System Prompt ─────────────────────────────────────────────────────

CLOVER_SYSTEM_PROMPT = """You are a formal verification assistant for EcodiaOS.
Your task: translate Python code into Dafny and produce a verified spec+implementation.

## The Clover Pattern (Three-Way Consistency)

For the given Python function, generate:
1. A Dafny `method` with `requires` (preconditions) and `ensures` (postconditions)
2. A Dafny method body that mirrors the Python logic
3. All code in a single ```dafny fenced block

The three-way consistency check requires:
- The natural language description matches the formal requires/ensures
- The implementation satisfies the requires/ensures
- The ensures clauses capture the essential behavior

## EcodiaOS Domain Rules
- Risk scores are in [0.0, 1.0]
- Budget values must be non-negative
- Drive alignment scores are in [-1.0, 1.0]
- Regression rates are in [0.0, 1.0]
- Episode counts are non-negative integers
- Priority scores are non-negative floats

## Output Format
Respond ONLY with a single ```dafny fenced code block containing:
- Any needed datatype/predicate definitions
- The method with requires/ensures
- The method body

Do NOT include explanatory text outside the code block."""


CLOVER_FEEDBACK_TEMPLATE = """The Dafny verifier reported errors on your previous output.

## Previous Spec + Implementation
```dafny
{previous_code}
```

## Dafny Verifier Errors (round {round_number}/{max_rounds})
```
{dafny_errors}
```

Fix the spec or implementation to resolve the errors while preserving correctness.
Respond ONLY with the corrected ```dafny fenced code block.
Common fixes:
- Strengthen preconditions if body assumptions are unmet
- Weaken postconditions if they are too strong
- Add loop invariants for while/for loops
- Add decreases clauses for termination proofs"""


# ── DafnyBridge ──────────────────────────────────────────────────────────────


class DafnyBridge:
    """
    Manages Dafny subprocess invocation and the Clover iteration loop.

    The bridge writes generated Dafny source to a temp file, invokes
    `dafny verify`, and parses the output. The Clover loop iterates
    between LLM generation and Dafny verification.
    """

    def __init__(
        self,
        dafny_path: str = "dafny",
        verify_timeout_s: float = 30.0,
        max_rounds: int = 8,
        temp_dir: Path | None = None,
    ) -> None:
        self._dafny_path = dafny_path
        self._verify_timeout_s = verify_timeout_s
        self._max_rounds = max_rounds
        self._temp_dir = temp_dir
        self._log = logger

    async def check_available(self) -> bool:
        """Check if the Dafny binary is available."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self._dafny_path, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=10.0)
            available = proc.returncode == 0
            if available:
                self._log.info("dafny_available", path=self._dafny_path)
            return available
        except (TimeoutError, FileNotFoundError):
            self._log.warning("dafny_not_available", path=self._dafny_path)
            return False
        except Exception as exc:
            self._log.warning("dafny_check_error", error=str(exc))
            return False

    async def verify_dafny_source(
        self, dafny_source: str,
    ) -> tuple[bool, str, str, int]:
        """
        Write Dafny source to a temp file and run `dafny verify`.

        Returns:
            (verified, stdout, stderr, exit_code)
        """
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".dfy",
            delete=False,
            dir=str(self._temp_dir) if self._temp_dir else None,
        ) as f:
            f.write(dafny_source)
            temp_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                self._dafny_path, "verify", temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=self._verify_timeout_s,
                )
            except TimeoutError:
                proc.kill()
                await proc.communicate()
                timeout_msg = (
                    f"Dafny verification timed out after {self._verify_timeout_s}s"
                )
                return False, "", timeout_msg, -1

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            exit_code = proc.returncode or 0
            verified = exit_code == 0

            self._log.debug(
                "dafny_verify_result",
                verified=verified,
                exit_code=exit_code,
                stdout_len=len(stdout),
                stderr_len=len(stderr),
            )
            return verified, stdout, stderr, exit_code

        except FileNotFoundError:
            return False, "", f"Dafny binary not found: {self._dafny_path}", -1
        except Exception as exc:
            return False, "", f"Dafny execution error: {exc}", -1
        finally:
            import contextlib
            with contextlib.suppress(Exception):
                Path(temp_path).unlink(missing_ok=True)

    async def run_clover_loop(
        self,
        llm: LLMProvider,
        python_source: str,
        function_name: str,
        context: str = "",
        template: str | None = None,
    ) -> DafnyVerificationResult:
        """
        The Clover pattern: LLM generates Dafny spec+impl,
        Dafny verifies, errors fed back, iterate.

        Args:
            llm: The LLM provider for spec/impl generation.
            python_source: The Python source code being verified.
            function_name: The function/method being formally verified.
            context: Additional context (change spec, description, etc.).
            template: Optional Dafny template to seed the generation.

        Returns:
            DafnyVerificationResult with full round history.
        """
        result = DafnyVerificationResult(rounds_max=self._max_rounds)
        start = time.monotonic()

        # Build initial prompt
        prompt = self._build_initial_prompt(
            python_source, function_name, context, template,
        )
        messages: list[Message] = [Message(role="user", content=prompt)]

        for round_num in range(1, self._max_rounds + 1):
            self._log.info(
                "clover_round_start",
                round=round_num,
                max_rounds=self._max_rounds,
                function=function_name,
            )

            # LLM generates Dafny spec+impl
            try:
                response = await llm.generate(
                    system_prompt=CLOVER_SYSTEM_PROMPT,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.2,
                )
            except Exception as exc:
                self._log.error("clover_llm_error", round=round_num, error=str(exc))
                result.status = DafnyVerificationStatus.FAILED
                result.error_summary = f"LLM call failed on round {round_num}: {exc}"
                result.rounds_attempted = round_num
                break

            # Parse Dafny code from LLM response
            dafny_code = self._parse_dafny_output(response.text)
            if not dafny_code:
                round_result = CloverRoundResult(
                    round_number=round_num,
                    errors=["Failed to parse Dafny code from LLM response"],
                    llm_tokens_used=getattr(response, "total_tokens", 0),
                )
                result.round_history.append(round_result)
                result.total_llm_tokens += round_result.llm_tokens_used

                # Feed parsing failure back
                messages = [Message(role="user", content=(
                    "Your response did not contain a valid ```dafny code block. "
                    "Please respond with ONLY a single ```dafny fenced code block."
                ))]
                continue

            # Run Dafny verifier
            dafny_start = time.monotonic()
            verified, stdout, stderr, exit_code = await self.verify_dafny_source(dafny_code)
            dafny_time = int((time.monotonic() - dafny_start) * 1000)
            result.total_dafny_time_ms += dafny_time

            errors = self._extract_errors(stderr, stdout) if not verified else []
            round_result = CloverRoundResult(
                round_number=round_num,
                spec_generated=dafny_code,
                implementation_generated=dafny_code,
                dafny_stdout=stdout[:2000],
                dafny_stderr=stderr[:2000],
                dafny_exit_code=exit_code,
                verified=verified,
                errors=errors,
                llm_tokens_used=getattr(response, "total_tokens", 0),
            )
            result.round_history.append(round_result)
            result.total_llm_tokens += round_result.llm_tokens_used

            if verified:
                result.status = DafnyVerificationStatus.VERIFIED
                result.final_spec = dafny_code
                result.final_implementation = dafny_code
                result.rounds_attempted = round_num
                result.proof_obligations = self._extract_proof_obligations(dafny_code)
                self._log.info(
                    "clover_verified",
                    round=round_num,
                    function=function_name,
                )
                break

            # Feed errors back for next round
            combined_errors = stderr or stdout
            feedback = CLOVER_FEEDBACK_TEMPLATE.format(
                previous_code=dafny_code,
                round_number=round_num,
                max_rounds=self._max_rounds,
                dafny_errors=combined_errors[:3000],
            )
            messages = [Message(role="user", content=feedback)]

        else:
            # Exhausted all rounds without verification
            result.status = DafnyVerificationStatus.FAILED
            result.rounds_attempted = self._max_rounds
            last_errors = (
                result.round_history[-1].errors
                if result.round_history
                else ["No rounds completed"]
            )
            result.error_summary = (
                f"Failed to verify after {self._max_rounds} rounds. "
                f"Last errors: {'; '.join(last_errors[:3])}"
            )
            self._log.warning(
                "clover_exhausted",
                rounds=self._max_rounds,
                function=function_name,
            )

        result.verification_time_ms = int((time.monotonic() - start) * 1000)
        return result

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_initial_prompt(
        self,
        python_source: str,
        function_name: str,
        context: str,
        template: str | None,
    ) -> str:
        """Build the initial Clover prompt asking LLM to generate Dafny spec+impl."""
        parts = [
            f"Translate the following Python function `{function_name}` into Dafny "
            f"with formal preconditions (requires) and postconditions (ensures).",
            "",
            "## Python Source",
            f"```python\n{python_source}\n```",
        ]

        if context:
            parts.extend(["", "## Context", context])

        if template:
            parts.extend([
                "",
                "## Dafny Template (use as starting point)",
                f"```dafny\n{template}\n```",
            ])

        parts.extend([
            "",
            "Generate a single ```dafny fenced code block with the complete "
            "Dafny translation including requires/ensures clauses.",
        ])

        return "\n".join(parts)

    def _parse_dafny_output(self, llm_text: str) -> str:
        """
        Extract Dafny code from LLM response.
        Looks for ```dafny ... ``` fenced blocks.
        """
        # Try dafny-specific fence first
        pattern = r"```dafny\s*\n(.*?)```"
        matches: list[str] = re.findall(pattern, llm_text, re.DOTALL)
        if matches:
            return matches[0].strip()

        # Fallback: any fenced code block
        pattern = r"```\w*\s*\n(.*?)```"
        matches = re.findall(pattern, llm_text, re.DOTALL)
        if matches:
            return matches[0].strip()

        return ""

    def _extract_errors(self, stderr: str, stdout: str = "") -> list[str]:
        """Extract individual error messages from Dafny output."""
        errors: list[str] = []
        combined = f"{stderr}\n{stdout}"
        for line in combined.splitlines():
            line = line.strip()
            if not line:
                continue
            # Dafny errors follow pattern: file.dfy(line,col): Error: message
            if "Error" in line or "error" in line:
                errors.append(line)
            elif "Warning" in line:
                continue  # Skip warnings
            elif "verification inconclusive" in line.lower():
                errors.append(line)
        return errors[:20]  # Cap at 20 errors

    def _extract_proof_obligations(self, dafny_code: str) -> list[str]:
        """Extract requires/ensures clauses as proof obligations."""
        obligations: list[str] = []
        for line in dafny_code.splitlines():
            stripped = line.strip()
            proof_kws = ("requires", "ensures", "invariant", "decreases")
            if any(stripped.startswith(kw) for kw in proof_kws):
                obligations.append(stripped)
        return obligations
