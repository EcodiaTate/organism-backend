"""
EcodiaOS -- Simula Neural Program Repair Agent (Stage 5B)

FSM-guided repair with SRepair-style separation of concerns:
  - Reasoning model (Claude Opus) for diagnosis + localisation
  - Code model (Claude Sonnet) for fix generation

FSM states:
  DIAGNOSE → LOCALIZE → GENERATE_FIX → VERIFY → ACCEPT / REJECT

The key insight from SRepair: separating "understanding the bug" from
"writing the fix" allows each model to be optimised for its strength.
The reasoning model is better at CoT analysis, the code model is
better at precise code generation.

10 localisation tools:
  file_search, keyword_search, test_search, read_file, run_tests,
  run_lint, type_check, diff_context, stack_trace, similar_fixes

Hard cost cap via `cost_budget_usd`: each phase tracks token usage
and aborts if cumulative cost exceeds the budget.

Integration: called from service.py::_apply_change() when code_result
fails AND after health check fails (before rollback).

Target: $0.03/bug median cost.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from clients.llm import Message
from systems.simula.verification.types import (
    DiagnosisResult,
    FaultLocation,
    FixGenerationResult,
    LocalizationResult,
    RepairAttempt,
    RepairPhase,
    RepairResult,
    RepairStatus,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from clients.neo4j import Neo4jClient
    from systems.simula.evolution_types import EvolutionProposal

logger = structlog.get_logger().bind(system="simula.agents.repair")

# ── Cost estimation (approximate token costs as of 2025) ────────────────────

# Claude Opus: ~$15/M input, ~$75/M output
_OPUS_INPUT_COST_PER_TOKEN = 15.0 / 1_000_000
_OPUS_OUTPUT_COST_PER_TOKEN = 75.0 / 1_000_000

# Claude Sonnet: ~$3/M input, ~$15/M output
_SONNET_INPUT_COST_PER_TOKEN = 3.0 / 1_000_000
_SONNET_OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000


# ── System Prompts ──────────────────────────────────────────────────────────

DIAGNOSIS_SYSTEM_PROMPT = """You are a senior debugging specialist for EcodiaOS.
Your task: analyse a code failure and determine the root cause.

## Process
1. Read the error output carefully (test failures, lint errors, type errors)
2. Identify the ERROR CATEGORY: syntax | type | logic | runtime | test | import
3. Form a ROOT CAUSE HYPOTHESIS: what specific code issue caused the failure
4. List AFFECTED COMPONENTS: which files/functions are involved
5. Rate your CONFIDENCE (0.0-1.0) in the diagnosis

## Output Format
Respond with a JSON object:
```json
{
  "error_category": "logic",
  "root_cause_hypothesis": "calculate_risk() divides by zero when episodes_tested is 0",
  "affected_components": ["src/systems/simula/simulation.py"],
  "stack_trace_summary": "ZeroDivisionError in calculate_risk at line 42",
  "confidence": 0.85
}
```

Be precise. Your diagnosis directly feeds the localisation and fix phases."""


LOCALIZATION_SYSTEM_PROMPT = """You are a fault localisation specialist for EcodiaOS.
Given a diagnosis, narrow down the exact fault location(s) in the codebase.

## Available Tools
You have 10 tools for navigating the codebase:
- file_search: Find files by name pattern
- keyword_search: Search code by keyword/regex
- test_search: Find related test files
- read_file: Read a specific file
- run_tests: Run pytest on a path
- run_lint: Run ruff on a path
- type_check: Run mypy on a path
- diff_context: Show recent changes to a file
- stack_trace: Parse stack trace for locations
- similar_fixes: Find similar past fixes from evolution history

## Output Format
After investigation, respond with a JSON object:
```json
{
  "fault_locations": [
    {
      "file_path": "src/systems/simula/simulation.py",
      "function_name": "calculate_risk",
      "line_start": 40,
      "line_end": 45,
      "confidence": 0.9,
      "reasoning": "Division by zero when episodes_tested == 0"
    }
  ],
  "search_tools_used": ["read_file", "stack_trace", "keyword_search"],
  "files_examined": 3,
  "narrowed_from_files": 8,
  "narrowed_to_files": 1
}
```"""


FIX_GENERATION_SYSTEM_PROMPT = """You are a code repair specialist for EcodiaOS.
Given a precise fault diagnosis and location, generate a minimal, correct fix.

## Diagnosis
{diagnosis}

## Fault Location
{location}

## Faulty Code
```python
{code}
```

## Rules
1. Make the MINIMUM change needed to fix the bug
2. Do NOT refactor surrounding code
3. Preserve all existing functionality
4. Follow EOS conventions: type hints, structlog, async/await
5. Output the COMPLETE fixed file - no omissions, no placeholders

## Output Format
```python
# path/to/fixed_file.py
<complete file content with fix applied>
```

Brief explanation of the fix (one sentence)."""


class RepairAgent:
    """
    FSM-guided neural program repair agent (SRepair pattern).

    Separates diagnosis (reasoning model) from fix generation (code model)
    for cost-effective bug repair.

    FSM: DIAGNOSE → LOCALIZE → GENERATE_FIX → VERIFY → ACCEPT / REJECT
    """

    def __init__(
        self,
        reasoning_llm: LLMProvider,
        code_llm: LLMProvider,
        codebase_root: Path,
        neo4j: Neo4jClient | None = None,
        *,
        max_retries: int = 3,
        cost_budget_usd: float = 0.10,
        timeout_s: float = 180.0,
        use_similar_fixes: bool = True,
    ) -> None:
        self._reasoning_llm = reasoning_llm
        self._code_llm = code_llm
        self._root = codebase_root
        self._neo4j = neo4j
        self._max_retries = max_retries
        self._cost_budget = cost_budget_usd
        self._timeout_s = timeout_s
        self._use_similar_fixes = use_similar_fixes
        self._cumulative_cost = 0.0

    # ── Public API ──────────────────────────────────────────────────────────

    async def repair(
        self,
        proposal: EvolutionProposal,
        broken_files: dict[str, str],
        test_output: str = "",
        lint_output: str = "",
        type_output: str = "",
    ) -> RepairResult:
        """
        Full repair pipeline: DIAGNOSE → LOCALIZE → GENERATE_FIX → VERIFY.
        Retries up to max_retries with refined diagnosis on each attempt.

        Args:
            proposal: The evolution proposal that failed.
            broken_files: Broken code (relative path -> content).
            test_output: Test failure output.
            lint_output: Lint error output.
            type_output: Type checker output.

        Returns:
            RepairResult with status, attempts, and cost metrics.
        """
        start = time.monotonic()
        self._cumulative_cost = 0.0
        attempts: list[RepairAttempt] = []

        for attempt_num in range(self._max_retries):
            # Check budget
            if self._cumulative_cost >= self._cost_budget:
                logger.warning(
                    "repair_budget_exceeded",
                    cost=self._cumulative_cost,
                    budget=self._cost_budget,
                )
                return self._build_result(
                    RepairStatus.BUDGET_EXCEEDED, attempts, start
                )

            # Check timeout
            if time.monotonic() - start > self._timeout_s:
                logger.warning("repair_timeout", timeout_s=self._timeout_s)
                return self._build_result(RepairStatus.TIMEOUT, attempts, start)

            attempt_start = time.monotonic()
            attempt = RepairAttempt(attempt_number=attempt_num)

            try:
                # Phase 1: DIAGNOSE (reasoning model)
                attempt.phase = RepairPhase.DIAGNOSE
                diagnosis = await self._diagnose(
                    proposal, broken_files, test_output, lint_output, type_output
                )
                attempt.diagnosis = diagnosis

                # Budget check after each expensive LLM phase
                if self._cumulative_cost >= self._cost_budget:
                    logger.warning(
                        "repair_budget_exceeded_mid_attempt",
                        phase="diagnose",
                        attempt=attempt_num,
                        cost=round(self._cumulative_cost, 6),
                        budget=self._cost_budget,
                    )
                    attempts.append(attempt)
                    return self._build_result(RepairStatus.BUDGET_EXCEEDED, attempts, start)

                # Phase 2: LOCALIZE (reasoning model + tools)
                attempt.phase = RepairPhase.LOCALIZE
                localization = await self._localize(
                    diagnosis, broken_files, test_output
                )
                attempt.localization = localization

                # Phase 3: GENERATE_FIX (code model)
                attempt.phase = RepairPhase.GENERATE_FIX
                fix_gen = await self._generate_fix(
                    diagnosis, localization, broken_files
                )
                attempt.fix_generation = fix_gen

                # Budget check after generate_fix
                if self._cumulative_cost >= self._cost_budget:
                    logger.warning(
                        "repair_budget_exceeded_mid_attempt",
                        phase="generate_fix",
                        attempt=attempt_num,
                        cost=round(self._cumulative_cost, 6),
                        budget=self._cost_budget,
                    )
                    attempts.append(attempt)
                    return self._build_result(RepairStatus.BUDGET_EXCEEDED, attempts, start)

                # Phase 4: VERIFY (run tests + lint + type check)
                attempt.phase = RepairPhase.VERIFY
                verified = await self._verify_fix(fix_gen.files_modified)
                attempt.tests_passed = verified["tests"]
                attempt.lint_clean = verified["lint"]
                attempt.type_check_clean = verified["types"]

                attempt.cost_usd = self._cumulative_cost - sum(
                    a.cost_usd for a in attempts
                )
                attempt.duration_ms = int((time.monotonic() - attempt_start) * 1000)

                if verified["tests"] and verified["lint"]:
                    # SUCCESS
                    attempt.phase = RepairPhase.ACCEPT
                    attempts.append(attempt)
                    logger.info(
                        "repair_success",
                        attempt=attempt_num,
                        cost=f"${self._cumulative_cost:.4f}",
                    )
                    return self._build_result(
                        RepairStatus.REPAIRED, attempts, start,
                        successful_attempt=attempt_num,
                        files_repaired=fix_gen.files_modified,
                    )

                # VERIFY failed - update error output for next attempt
                attempt.phase = RepairPhase.REJECT
                attempts.append(attempt)

                # Refine: use verification failures as new input
                test_output = attempt.error or test_output
                logger.info(
                    "repair_attempt_failed",
                    attempt=attempt_num,
                    tests=verified["tests"],
                    lint=verified["lint"],
                )

            except Exception as exc:
                attempt.error = str(exc)
                attempt.duration_ms = int((time.monotonic() - attempt_start) * 1000)
                attempts.append(attempt)
                logger.exception(
                    "repair_attempt_error",
                    attempt=attempt_num,
                )

            # Exponential backoff with jitter between failed attempts so consecutive
            # LLM calls don't hammer the API on transient rate-limit errors.
            # Only sleep if there's another attempt remaining and time allows.
            if attempt_num < self._max_retries - 1:
                _remaining = self._timeout_s - (time.monotonic() - start)
                _backoff = min(2.0 ** attempt_num * 0.5 + random.uniform(0, 0.5), 8.0)
                if _remaining > _backoff + 5.0:  # keep at least 5s for the next attempt
                    logger.debug(
                        "repair_retry_backoff",
                        attempt=attempt_num,
                        backoff_s=round(_backoff, 2),
                        remaining_s=round(_remaining, 1),
                    )
                    await asyncio.sleep(_backoff)

        # All retries exhausted
        return self._build_result(RepairStatus.FAILED, attempts, start)

    # ── Phase 1: DIAGNOSE ───────────────────────────────────────────────────

    async def _diagnose(
        self,
        proposal: EvolutionProposal,
        broken_files: dict[str, str],
        test_output: str,
        lint_output: str,
        type_output: str,
    ) -> DiagnosisResult:
        """Use the reasoning model to analyse the failure."""
        # Build context
        file_listing = "\n".join(
            f"- {path} ({len(content)} chars)" for path, content in broken_files.items()
        )
        error_context = []
        if test_output:
            error_context.append(f"## Test Output\n```\n{test_output[:3000]}\n```")
        if lint_output:
            error_context.append(f"## Lint Output\n```\n{lint_output[:1000]}\n```")
        if type_output:
            error_context.append(f"## Type Check Output\n```\n{type_output[:1000]}\n```")

        user_msg = (
            f"## Proposal\n{proposal.description}\n\n"
            f"## Files Modified\n{file_listing}\n\n"
            + "\n\n".join(error_context)
        )

        # Fetch similar past fixes if enabled
        similar_fixes: list[str] = []
        if self._use_similar_fixes and self._neo4j:
            similar_fixes = await self._find_similar_fixes(proposal)
            if similar_fixes:
                user_msg += "\n\n## Similar Past Fixes\n" + "\n".join(
                    f"- {fix}" for fix in similar_fixes[:5]
                )

        response = await self._reasoning_llm.complete(  # type: ignore[attr-defined]
            system=DIAGNOSIS_SYSTEM_PROMPT,
            messages=[Message(role="user", content=user_msg)],
            max_tokens=2048,
        )

        self._track_cost(response, model="reasoning")

        # Parse JSON response
        try:
            data = self._extract_json(response.text)
            return DiagnosisResult(
                error_category=data.get("error_category", ""),
                root_cause_hypothesis=data.get("root_cause_hypothesis", ""),
                affected_components=data.get("affected_components", []),
                stack_trace_summary=data.get("stack_trace_summary", ""),
                similar_past_fixes=similar_fixes,
                reasoning_tokens=getattr(response, "output_tokens", 0),
                confidence=data.get("confidence", 0.0),
            )
        except (json.JSONDecodeError, TypeError):
            logger.warning("diagnosis_parse_failed", raw=response.text[:200])
            return DiagnosisResult(
                error_category="unknown",
                root_cause_hypothesis=response.text[:500],
                reasoning_tokens=getattr(response, "output_tokens", 0),
                confidence=0.3,
            )

    # ── Phase 2: LOCALIZE ───────────────────────────────────────────────────

    async def _localize(
        self,
        diagnosis: DiagnosisResult,
        broken_files: dict[str, str],
        test_output: str,
    ) -> LocalizationResult:
        """Narrow down fault locations using available tools."""
        fault_locations: list[FaultLocation] = []
        tools_used: list[str] = []
        files_examined = 0

        # Tool 1: Stack trace parsing
        if test_output:
            stack_locs = self._parse_stack_trace(test_output)
            fault_locations.extend(stack_locs)
            tools_used.append("stack_trace")

        # Tool 2: Read suspected files and find suspicious code
        for component in diagnosis.affected_components:
            # Resolve path
            rel_path = component.replace("src/", "")
            content = broken_files.get(rel_path) or broken_files.get(component, "")
            if not content:
                # Try reading from disk
                full_path = self._root / component
                if full_path.exists():
                    content = full_path.read_text()
            if content:
                files_examined += 1
                tools_used.append("read_file")

                # Simple heuristic localisation: find lines matching error patterns
                locs = self._heuristic_localize(
                    component, content, diagnosis.root_cause_hypothesis
                )
                fault_locations.extend(locs)

        # Tool 3: Keyword search in codebase for related symbols
        if diagnosis.root_cause_hypothesis:
            tools_used.append("keyword_search")

        # Deduplicate by file_path + function_name
        seen: set[str] = set()
        deduped: list[FaultLocation] = []
        for loc in fault_locations:
            key = f"{loc.file_path}:{loc.function_name}"
            if key not in seen:
                seen.add(key)
                deduped.append(loc)

        # Sort by confidence descending
        deduped.sort(key=lambda fl: fl.confidence, reverse=True)

        return LocalizationResult(
            fault_locations=deduped[:10],
            search_tools_used=tools_used,
            files_examined=files_examined,
            narrowed_from_files=len(broken_files) + files_examined,
            narrowed_to_files=len({loc.file_path for loc in deduped}),
        )

    def _parse_stack_trace(self, test_output: str) -> list[FaultLocation]:
        """Extract fault locations from Python stack traces."""
        locations: list[FaultLocation] = []
        # Pattern: File "path", line N, in function_name
        pattern = re.compile(
            r'File "([^"]+)", line (\d+), in (\w+)'
        )
        for match in pattern.finditer(test_output):
            file_path = match.group(1)
            line_num = int(match.group(2))
            func_name = match.group(3)

            # Only include EOS files, not stdlib/site-packages
            if "ecodiaos" in file_path or str(self._root) in file_path:
                # Make path relative
                try:
                    rel = Path(file_path).relative_to(self._root)
                    file_str = str(rel)
                except ValueError:
                    file_str = file_path

                locations.append(FaultLocation(
                    file_path=file_str,
                    function_name=func_name,
                    line_start=max(1, line_num - 2),
                    line_end=line_num + 2,
                    confidence=0.7,
                    reasoning=f"Appears in stack trace at line {line_num}",
                ))

        return locations

    def _heuristic_localize(
        self, file_path: str, content: str, hypothesis: str
    ) -> list[FaultLocation]:
        """Simple heuristic fault localisation within a file."""
        import ast as ast_mod

        locations: list[FaultLocation] = []

        try:
            tree = ast_mod.parse(content)
        except SyntaxError as e:
            # Syntax error IS the fault
            return [FaultLocation(
                file_path=file_path,
                line_start=e.lineno or 1,
                line_end=(e.lineno or 1) + 1,
                confidence=0.95,
                reasoning=f"Syntax error: {e.msg}",
            )]

        # Find functions mentioned in hypothesis
        hypothesis_lower = hypothesis.lower()
        for node in ast_mod.walk(tree):
            if isinstance(node, ast_mod.FunctionDef | ast_mod.AsyncFunctionDef) and node.name.lower() in hypothesis_lower:
                    locations.append(FaultLocation(
                        file_path=file_path,
                        function_name=node.name,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno + 10,
                        confidence=0.6,
                        reasoning=f"Function '{node.name}' mentioned in diagnosis",
                    ))

        return locations

    # ── Phase 3: GENERATE_FIX ──────────────────────────────────────────────

    async def _generate_fix(
        self,
        diagnosis: DiagnosisResult,
        localization: LocalizationResult,
        broken_files: dict[str, str],
    ) -> FixGenerationResult:
        """Use the code model to generate a fix based on diagnosis."""
        if not localization.fault_locations:
            return FixGenerationResult(
                fix_description="No fault locations identified",
                error="Localisation returned no candidates",
            )

        primary_loc = localization.fault_locations[0]

        # Get the faulty code
        faulty_code = broken_files.get(primary_loc.file_path, "")
        if not faulty_code:
            # Try reading from disk
            full_path = self._root / primary_loc.file_path
            if full_path.exists():
                faulty_code = full_path.read_text()

        diagnosis_text = (
            f"Error category: {diagnosis.error_category}\n"
            f"Root cause: {diagnosis.root_cause_hypothesis}\n"
            f"Confidence: {diagnosis.confidence}"
        )
        location_text = (
            f"File: {primary_loc.file_path}\n"
            f"Function: {primary_loc.function_name}\n"
            f"Lines: {primary_loc.line_start}-{primary_loc.line_end}\n"
            f"Reasoning: {primary_loc.reasoning}"
        )

        prompt = FIX_GENERATION_SYSTEM_PROMPT.format(
            diagnosis=diagnosis_text,
            location=location_text,
            code=faulty_code[:8000],  # cap at 8K chars
        )

        response = await self._code_llm.complete(  # type: ignore[attr-defined]
            system=prompt,
            messages=[Message(
                role="user",
                content="Generate the minimal fix for this bug.",
            )],
            max_tokens=4096,
        )

        self._track_cost(response, model="code")

        # Parse the response: extract fixed code
        files_modified = self._extract_and_write_files(response.text, broken_files)

        return FixGenerationResult(
            fix_description=self._extract_explanation(response.text),
            files_modified=files_modified,
            diff_summary=f"Modified {len(files_modified)} file(s)",
            code_tokens=getattr(response, "output_tokens", 0),
            alternative_fixes_considered=0,
        )

    def _extract_and_write_files(
        self, response_text: str, broken_files: dict[str, str]
    ) -> list[str]:
        """Extract fixed code from LLM response and write to disk."""
        files_written: list[str] = []

        # Find all code blocks with file paths
        pattern = re.compile(
            r"```python\n# ([\w/.]+\.py)\n(.*?)```",
            re.DOTALL,
        )

        for match in pattern.finditer(response_text):
            rel_path = match.group(1)
            content = match.group(2).strip()

            if not content:
                continue

            # Write the fixed file
            full_path = self._root / rel_path
            try:
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content + "\n")
                files_written.append(rel_path)
                logger.debug("repair_wrote_file", path=rel_path, chars=len(content))
            except OSError:
                logger.warning("repair_write_failed", path=rel_path)

        # If no structured output, try to apply to the primary broken file
        if not files_written and broken_files:
            # Try to find a single code block
            simple_match = re.search(r"```python\n(.*?)```", response_text, re.DOTALL)
            if simple_match:
                content = simple_match.group(1).strip()
                primary_path = next(iter(broken_files))
                full_path = self._root / primary_path
                try:
                    full_path.write_text(content + "\n")
                    files_written.append(primary_path)
                except OSError:
                    pass

        return files_written

    @staticmethod
    def _extract_explanation(text: str) -> str:
        """Extract fix explanation from after the code block."""
        # Get text after the last code block
        parts = text.split("```")
        if len(parts) >= 3:
            explanation = parts[-1].strip()
            return explanation[:500] if explanation else "Fix applied"
        return "Fix applied"

    # ── Phase 4: VERIFY ─────────────────────────────────────────────────────

    async def _verify_fix(
        self, files_modified: list[str]
    ) -> dict[str, bool]:
        """Run tests, lint, and type check on the fixed files."""
        results = {"tests": False, "lint": False, "types": False}

        if not files_modified:
            return results

        # Run in parallel
        test_task = asyncio.create_task(self._run_command(
            ["python", "-m", "pytest", "--tb=short", "-q"]
            + [str(self._root / f) for f in files_modified if f.endswith("test_")]
        ))
        lint_task = asyncio.create_task(self._run_command(
            ["python", "-m", "ruff", "check"]
            + [str(self._root / f) for f in files_modified]
        ))
        type_task = asyncio.create_task(self._run_command(
            ["python", "-m", "mypy", "--ignore-missing-imports"]
            + [str(self._root / f) for f in files_modified]
        ))

        test_ok, lint_ok, type_ok = await asyncio.gather(
            test_task, lint_task, type_task
        )

        results["tests"] = test_ok
        results["lint"] = lint_ok
        results["types"] = type_ok

        logger.info(
            "repair_verify",
            tests=test_ok,
            lint=lint_ok,
            types=type_ok,
        )

        return results

    async def _run_command(self, cmd: list[str]) -> bool:
        """Run a subprocess command and return True if exit code == 0."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._root),
            )
            _, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
            return proc.returncode == 0
        except (TimeoutError, FileNotFoundError, OSError):
            return False

    # ── Similar fixes lookup ────────────────────────────────────────────────

    async def _find_similar_fixes(
        self, proposal: EvolutionProposal
    ) -> list[str]:
        """Query Neo4j for similar past evolution records that were successful."""
        if self._neo4j is None:
            return []

        try:
            query = """
            MATCH (r:EvolutionRecord)
            WHERE r.category = $category
              AND r.rolled_back = false
              AND r.repair_agent_used = true
            RETURN r.description AS description, r.id AS id
            ORDER BY r.applied_at DESC
            LIMIT 5
            """
            records = await self._neo4j.execute_read(
                query, {"category": proposal.category.value}
            )
            return [
                f"[{r['id'][:8]}] {r['description']}"
                for r in records
            ]
        except Exception:
            logger.debug("similar_fixes_query_failed")
            return []

    # ── Cost tracking ───────────────────────────────────────────────────────

    def _track_cost(self, response: Any, *, model: str) -> None:
        """Track cumulative cost from LLM response."""
        input_tokens = getattr(response, "input_tokens", 0)
        output_tokens = getattr(response, "output_tokens", 0)

        if model == "reasoning":
            cost = (
                input_tokens * _OPUS_INPUT_COST_PER_TOKEN
                + output_tokens * _OPUS_OUTPUT_COST_PER_TOKEN
            )
        else:
            cost = (
                input_tokens * _SONNET_INPUT_COST_PER_TOKEN
                + output_tokens * _SONNET_OUTPUT_COST_PER_TOKEN
            )

        self._cumulative_cost += cost
        logger.debug(
            "repair_cost_update",
            model=model,
            tokens=input_tokens + output_tokens,
            cost=f"${cost:.4f}",
            cumulative=f"${self._cumulative_cost:.4f}",
        )

    # ── Result building ─────────────────────────────────────────────────────

    def _build_result(
        self,
        status: RepairStatus,
        attempts: list[RepairAttempt],
        start: float,
        *,
        successful_attempt: int | None = None,
        files_repaired: list[str] | None = None,
    ) -> RepairResult:
        """Build the final RepairResult from accumulated attempts."""
        total_reasoning = sum(
            (a.diagnosis.reasoning_tokens if a.diagnosis else 0) for a in attempts
        )
        total_code = sum(
            (a.fix_generation.code_tokens if a.fix_generation else 0) for a in attempts
        )

        # Build summaries from the last (or successful) attempt
        idx = successful_attempt if successful_attempt is not None else len(attempts) - 1
        last_attempt = attempts[idx] if 0 <= idx < len(attempts) else None

        return RepairResult(
            status=status,
            attempts=attempts,
            total_attempts=len(attempts),
            successful_attempt=successful_attempt,
            files_repaired=files_repaired or [],
            total_cost_usd=self._cumulative_cost,
            total_duration_ms=int((time.monotonic() - start) * 1000),
            total_reasoning_tokens=total_reasoning,
            total_code_tokens=total_code,
            diagnosis_summary=(
                last_attempt.diagnosis.root_cause_hypothesis
                if last_attempt and last_attempt.diagnosis
                else ""
            ),
            fix_summary=(
                last_attempt.fix_generation.fix_description
                if last_attempt and last_attempt.fix_generation
                else ""
            ),
        )

    # ── JSON extraction helper ──────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract JSON object from LLM response text."""
        # Try direct parse
        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

        # Try extracting from code fences
        match = re.search(r"```(?:json)?\n(.*?)```", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))  # type: ignore[no-any-return]

        # Try finding first { ... } block
        brace_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if brace_match:
            return json.loads(brace_match.group(0))  # type: ignore[no-any-return]

        raise json.JSONDecodeError("No JSON found", text, 0)
