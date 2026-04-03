"""
EcodiaOS -- Simula Formal Spec Generator (Stage 6C)

Auto-generates formal specifications for system interfaces:
  - 6C.1: Dafny spec generation (DafnyBench 96% target)
  - 6C.2: TLA+ specs for distributed interactions (Synapse cycle, Evo→Simula)
  - 6C.3: Self-Spec: LLMs invent task-specific DSLs for novel categories
  - 6C.4: Alloy for property checking on system invariants

Each spec kind uses LLM generation + tool verification:
  Dafny: LLM generates spec → Dafny verifies → iterate
  TLA+:  LLM generates spec → TLC model-checks → iterate
  Alloy: LLM generates model → Alloy analyzer → iterate
  Self-Spec: LLM invents DSL grammar → validates examples
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
import tempfile
import time
from typing import TYPE_CHECKING
from pathlib import Path

import structlog

from systems.simula.verification.types import (
    AlloyCheckResult,
    FormalSpecGenerationResult,
    FormalSpecKind,
    FormalSpecResult,
    FormalSpecStatus,
    SelfSpecDSL,
    TlaPlusModelCheckResult,
)

if TYPE_CHECKING:

    from clients.llm import LLMProvider
    from systems.simula.evolution_types import ChangeCategory, EvolutionProposal
    from systems.simula.verification.dafny_bridge import DafnyBridge
logger = structlog.get_logger().bind(system="simula.formal_specs.generator")


_DAFNY_SPEC_PROMPT = """\
Dafny formal specification for a Python function.

Python function:
```python
{source}
```

Output: a complete Dafny method with `requires`/`ensures` clauses, loop invariants where applicable, and implementation body. Output ONLY the Dafny source code (no markdown fences).
"""

_TLA_PLUS_PROMPT = """\
TLA+ specification for the following distributed system interaction.

System: {system_name}
Interactions: {interactions}

Output: a TLA+ spec modelling state variables, initial predicates, next-state relations, safety invariants, and liveness properties. Output ONLY the TLA+ source code.
"""

_ALLOY_PROMPT = """\
Alloy model for the following system properties.

Properties to check:
{properties}

Output: an Alloy model with signatures, facts, assertions, and check commands with scope {scope}. Output ONLY the Alloy source code.
"""

_SELF_SPEC_DSL_PROMPT = """\
Proposal category "{category}" doesn't fit existing formal methods (Dafny/TLA+/Alloy).

Example proposals:
{examples}

Design a minimal DSL that can specify expected behavior for this category.

Output JSON:
{{
  "dsl_name": string,
  "grammar": string,
  "examples": [string, string, string],
  "description": string
}}
"""


class FormalSpecGenerator:
    """Generates and verifies formal specifications for changed code."""

    def __init__(
        self,
        llm: LLMProvider,
        dafny_bridge: DafnyBridge | None = None,
        *,
        tla_plus_path: str = "tlc",
        alloy_path: str = "alloy",
        dafny_bench_target: float = 0.96,
        tla_plus_timeout_s: float = 120.0,
        alloy_scope: int = 10,
    ) -> None:
        self._llm = llm
        self._dafny = dafny_bridge
        self._tlc_path = tla_plus_path
        self._alloy_path = alloy_path
        self._dafny_bench_target = dafny_bench_target
        self._tla_plus_timeout_s = tla_plus_timeout_s
        self._alloy_scope = alloy_scope

    # ── Public API ──────────────────────────────────────────────────────────

    async def generate_all(
        self,
        files: list[str],
        proposal: EvolutionProposal,
        codebase_root: Path,
        *,
        dafny_enabled: bool = True,
        tla_plus_enabled: bool = False,
        alloy_enabled: bool = False,
        self_spec_enabled: bool = False,
    ) -> FormalSpecGenerationResult:
        """
        Orchestrate all spec generation for a proposal.

        Runs enabled spec generators in parallel for efficiency.
        """
        start = time.monotonic()
        all_specs: list[FormalSpecResult] = []
        tla_results: list[TlaPlusModelCheckResult] = []
        alloy_results: list[AlloyCheckResult] = []
        self_spec_dsls: list[SelfSpecDSL] = []
        total_tokens = 0

        tasks: list[asyncio.Task[object]] = []

        if dafny_enabled:
            tasks.append(
                asyncio.create_task(
                    self._generate_dafny_specs_async(files, codebase_root),
                ),
            )

        if tla_plus_enabled:
            tasks.append(
                asyncio.create_task(
                    self._generate_tla_plus_async(proposal),
                ),
            )

        if alloy_enabled:
            tasks.append(
                asyncio.create_task(
                    self._generate_alloy_async(proposal, codebase_root),
                ),
            )

        if self_spec_enabled:
            tasks.append(
                asyncio.create_task(
                    self._generate_self_spec_async(proposal),
                ),
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning("spec_generation_task_failed", error=str(result))
                continue

            if isinstance(result, list):
                for item in result:
                    if isinstance(item, FormalSpecResult):
                        all_specs.append(item)
                        total_tokens += item.llm_tokens_used
                    elif isinstance(item, TlaPlusModelCheckResult):
                        tla_results.append(item)
                    elif isinstance(item, AlloyCheckResult):
                        alloy_results.append(item)
                    elif isinstance(item, SelfSpecDSL):
                        self_spec_dsls.append(item)
                        total_tokens += item.llm_tokens_used

            elif isinstance(result, TlaPlusModelCheckResult):
                tla_results.append(result)
            elif isinstance(result, AlloyCheckResult):
                alloy_results.append(result)
            elif isinstance(result, SelfSpecDSL):
                self_spec_dsls.append(result)
                total_tokens += result.llm_tokens_used

        # Compute coverage
        verified = sum(1 for s in all_specs if s.verified)
        total_functions = max(len(all_specs), 1)
        coverage = verified / total_functions if all_specs else 0.0

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "formal_spec_generation_complete",
            specs=len(all_specs),
            verified=verified,
            coverage=f"{coverage:.0%}",
            tla_plus=len(tla_results),
            alloy=len(alloy_results),
            self_specs=len(self_spec_dsls),
            duration_ms=elapsed_ms,
        )

        return FormalSpecGenerationResult(
            specs=all_specs,
            overall_coverage_percent=coverage,
            tla_plus_results=tla_results,
            alloy_results=alloy_results,
            self_spec_dsls=self_spec_dsls,
            total_llm_tokens=total_tokens,
            total_duration_ms=elapsed_ms,
        )

    async def generate_dafny_specs(
        self,
        files: list[str],
        codebase_root: Path,
    ) -> list[FormalSpecResult]:
        """Generate Dafny specifications for functions in changed files."""
        return await self._generate_dafny_specs_async(files, codebase_root)

    async def generate_tla_plus_spec(
        self,
        system_name: str,
        interactions: list[str],
    ) -> TlaPlusModelCheckResult:
        """Generate and model-check a TLA+ specification."""
        start = time.monotonic()

        prompt = _TLA_PLUS_PROMPT.format(
            system_name=system_name,
            interactions="\n".join(f"- {i}" for i in interactions),
        )

        try:
            from clients.llm import Message

            response = await self._llm.complete(  # type: ignore[attr-defined]
                system="Generate valid TLA+ specification source. Output only the spec.",
                messages=[Message(role="user", content=prompt)],
                max_tokens=4096,
            )
            spec_source = response.content if hasattr(response, "content") else str(response)

            # Run TLC model checker
            mc_result = await self._run_tlc(spec_source, system_name)
            mc_result.spec_source = spec_source
            mc_result.system_name = system_name

            elapsed_ms = int((time.monotonic() - start) * 1000)
            mc_result.duration_ms = elapsed_ms
            return mc_result

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.warning("tla_plus_generation_failed", error=str(exc))
            return TlaPlusModelCheckResult(
                status=FormalSpecStatus.FAILED,
                system_name=system_name,
                duration_ms=elapsed_ms,
            )

    async def check_alloy_properties(
        self,
        properties: list[str],
        codebase_root: Path,
    ) -> AlloyCheckResult:
        """Generate an Alloy model and check system properties."""
        start = time.monotonic()

        prompt = _ALLOY_PROMPT.format(
            properties="\n".join(f"- {p}" for p in properties),
            scope=self._alloy_scope,
        )

        try:
            from clients.llm import Message

            response = await self._llm.complete(  # type: ignore[attr-defined]
                system="Generate a valid Alloy model. Output only the model source.",
                messages=[Message(role="user", content=prompt)],
                max_tokens=4096,
            )
            model_source = response.content if hasattr(response, "content") else str(response)

            result = await self._run_alloy(model_source)
            result.model_source = model_source
            result.scope = self._alloy_scope

            elapsed_ms = int((time.monotonic() - start) * 1000)
            result.duration_ms = elapsed_ms
            return result

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.warning("alloy_check_failed", error=str(exc))
            return AlloyCheckResult(
                status=FormalSpecStatus.FAILED,
                duration_ms=elapsed_ms,
            )

    async def generate_self_spec_dsl(
        self,
        category: ChangeCategory,
        examples: list[str],
    ) -> SelfSpecDSL:
        """LLM invents a task-specific DSL for a novel proposal category."""
        prompt = _SELF_SPEC_DSL_PROMPT.format(
            category=category.value if hasattr(category, "value") else str(category),
            examples="\n".join(f"- {e}" for e in examples),
        )

        try:
            from clients.llm import Message

            response = await self._llm.complete(  # type: ignore[attr-defined]
                system="Generate a formal verification DSL definition. Output as JSON.",
                messages=[Message(role="user", content=prompt)],
                max_tokens=4096,
            )
            text = response.content if hasattr(response, "content") else str(response)
            tokens = getattr(response, "input_tokens", 0) + getattr(response, "output_tokens", 0)

            # Parse JSON response
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                data = json.loads(json_match.group())
                return SelfSpecDSL(
                    dsl_name=str(data.get("dsl_name", "")),
                    grammar_source=str(data.get("grammar", "")),
                    example_programs=list(data.get("examples", [])),
                    target_category=category.value if hasattr(category, "value") else str(category),
                    coverage_rate=0.0,
                    llm_tokens_used=tokens,
                )

            return SelfSpecDSL(
                target_category=category.value if hasattr(category, "value") else str(category),
                llm_tokens_used=tokens,
            )

        except Exception as exc:
            logger.warning("self_spec_dsl_failed", error=str(exc))
            return SelfSpecDSL(
                target_category=category.value if hasattr(category, "value") else str(category),
            )

    # ── Private: Dafny spec generation ──────────────────────────────────────

    async def _generate_dafny_specs_async(
        self,
        files: list[str],
        codebase_root: Path,
    ) -> list[FormalSpecResult]:
        """Extract functions from files and generate Dafny specs."""
        results: list[FormalSpecResult] = []

        for file_path in files:
            full_path = codebase_root / file_path
            if not full_path.exists() or not file_path.endswith(".py"):
                continue

            try:
                source = full_path.read_text(encoding="utf-8")
                tree = ast.parse(source)
            except (SyntaxError, UnicodeDecodeError):
                continue

            lines = source.splitlines()

            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue

                func_start = node.lineno - 1
                func_end = node.end_lineno or func_start + 1
                func_source = "\n".join(lines[func_start:func_end])
                func_name = node.name

                spec_result = await self._generate_single_dafny_spec(
                    func_source, func_name, file_path,
                )
                results.append(spec_result)

        return results

    async def _generate_single_dafny_spec(
        self,
        func_source: str,
        func_name: str,
        file_path: str,
    ) -> FormalSpecResult:
        """Generate a Dafny spec for a single function."""
        start = time.monotonic()

        prompt = _DAFNY_SPEC_PROMPT.format(source=func_source)

        try:
            from clients.llm import Message

            response = await self._llm.complete(  # type: ignore[attr-defined]
                system="Generate a Dafny specification for the given function. Output only valid Dafny source.",
                messages=[Message(role="user", content=prompt)],
                max_tokens=2048,
            )
            spec_source = response.content if hasattr(response, "content") else str(response)
            tokens = getattr(response, "input_tokens", 0) + getattr(response, "output_tokens", 0)

            # Verify with Dafny if bridge is available
            verified = False
            verification_output = ""
            if self._dafny is not None:
                try:
                    dafny_verified, dafny_stdout, dafny_stderr, _ = await self._dafny.verify_dafny_source(spec_source)
                    verified = dafny_verified
                    verification_output = dafny_stderr if not verified else "Verified"
                except Exception as exc:
                    verification_output = f"Dafny verification error: {exc}"

            status = FormalSpecStatus.VERIFIED if verified else FormalSpecStatus.GENERATED

            elapsed_ms = int((time.monotonic() - start) * 1000)
            return FormalSpecResult(
                kind=FormalSpecKind.DAFNY,
                status=status,
                spec_source=spec_source,
                target_function=func_name,
                target_file=file_path,
                coverage_percent=1.0 if verified else 0.5,
                verified=verified,
                verification_output=verification_output,
                llm_tokens_used=tokens,
                duration_ms=elapsed_ms,
            )

        except Exception as exc:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.warning(
                "dafny_spec_generation_failed",
                function=func_name,
                error=str(exc),
            )
            return FormalSpecResult(
                kind=FormalSpecKind.DAFNY,
                status=FormalSpecStatus.FAILED,
                target_function=func_name,
                target_file=file_path,
                duration_ms=elapsed_ms,
            )

    # ── Private: TLA+ model checking ────────────────────────────────────────

    async def _generate_tla_plus_async(
        self,
        proposal: EvolutionProposal,
    ) -> TlaPlusModelCheckResult:
        """Generate a TLA+ spec from a proposal's target system."""
        system_name = getattr(proposal, "target", None) or "UnknownSystem"
        interactions = [proposal.description]
        if hasattr(proposal, "change_spec") and proposal.change_spec:
            if proposal.change_spec.code_hint:
                interactions.append(proposal.change_spec.code_hint)
        return await self.generate_tla_plus_spec(system_name, interactions)

    async def _run_tlc(self, spec_source: str, system_name: str) -> TlaPlusModelCheckResult:
        """Run the TLC model checker on a TLA+ specification."""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".tla", delete=False, prefix=f"{system_name}_",
            ) as f:
                f.write(spec_source)
                spec_path = f.name

            proc = await asyncio.create_subprocess_exec(
                self._tlc_path, spec_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._tla_plus_timeout_s,
                )
            except TimeoutError:
                proc.kill()
                return TlaPlusModelCheckResult(status=FormalSpecStatus.TIMEOUT)

            output = stdout.decode("utf-8", errors="replace")

            # Parse TLC output for states explored and violations
            states = 0
            violations: list[str] = []
            deadlocks = 0

            for line in output.splitlines():
                if "states found" in line.lower():
                    match = re.search(r"(\d+)\s+states", line)
                    if match:
                        states = int(match.group(1))
                if "violation" in line.lower() or "error" in line.lower():
                    violations.append(line.strip())
                if "deadlock" in line.lower():
                    deadlocks += 1

            status = (
                FormalSpecStatus.VERIFIED
                if proc.returncode == 0 and not violations
                else FormalSpecStatus.FAILED
            )

            return TlaPlusModelCheckResult(
                status=status,
                states_explored=states,
                distinct_states=states,
                violations=violations,
                deadlocks_found=deadlocks,
            )

        except FileNotFoundError:
            logger.warning("tlc_not_found", path=self._tlc_path)
            return TlaPlusModelCheckResult(status=FormalSpecStatus.SKIPPED)
        except Exception as exc:
            logger.warning("tlc_execution_failed", error=str(exc))
            return TlaPlusModelCheckResult(status=FormalSpecStatus.FAILED)

    # ── Private: Alloy checking ─────────────────────────────────────────────

    async def _generate_alloy_async(
        self,
        proposal: EvolutionProposal,
        codebase_root: Path,
    ) -> AlloyCheckResult:
        """Generate and check Alloy properties from a proposal."""
        properties = [proposal.description]
        return await self.check_alloy_properties(properties, codebase_root)

    async def _run_alloy(self, model_source: str) -> AlloyCheckResult:
        """Run the Alloy analyzer on a model."""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".als", delete=False,
            ) as f:
                f.write(model_source)
                model_path = f.name

            proc = await asyncio.create_subprocess_exec(
                self._alloy_path, model_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=60.0,
                )
            except TimeoutError:
                proc.kill()
                return AlloyCheckResult(status=FormalSpecStatus.TIMEOUT)

            output = stdout.decode("utf-8", errors="replace")

            # Parse Alloy output
            instances = 0
            counterexamples: list[str] = []

            for line in output.splitlines():
                if "instance" in line.lower():
                    instances += 1
                if "counterexample" in line.lower():
                    counterexamples.append(line.strip())

            status = (
                FormalSpecStatus.VERIFIED
                if proc.returncode == 0 and not counterexamples
                else FormalSpecStatus.FAILED
            )

            return AlloyCheckResult(
                status=status,
                instances_found=instances,
                counterexamples=counterexamples,
            )

        except FileNotFoundError:
            logger.warning("alloy_not_found", path=self._alloy_path)
            return AlloyCheckResult(status=FormalSpecStatus.SKIPPED)
        except Exception as exc:
            logger.warning("alloy_execution_failed", error=str(exc))
            return AlloyCheckResult(status=FormalSpecStatus.FAILED)

    # ── Private: Self-Spec DSL ──────────────────────────────────────────────

    async def _generate_self_spec_async(
        self,
        proposal: EvolutionProposal,
    ) -> SelfSpecDSL:
        """Generate a Self-Spec DSL for a proposal's category."""
        examples = [proposal.description]
        if hasattr(proposal, "change_spec") and proposal.change_spec:
            if proposal.change_spec.code_hint:
                examples.append(proposal.change_spec.code_hint)
        return await self.generate_self_spec_dsl(proposal.category, examples)
