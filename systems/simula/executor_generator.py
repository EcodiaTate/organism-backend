"""
EcodiaOS - Simula Executor Generator
Speciation Bible §8.3 - Organizational Closure (dynamic capability expansion).

Enables Simula to generate NEW Axon executor classes at runtime - not on the
next incarnation, but immediately hot-loaded and registered.  This closes the
loop from Evo opportunity discovery through to live action capability.

The closure loop:
  Evo OPPORTUNITY_DISCOVERED (new DeFi protocol / bounty platform, no executor)
  → Evo emits EVOLUTION_CANDIDATE(mutation_type="add_executor", template=...)
  → SimulaService._on_evolution_candidate routes to ExecutorGenerator
  → ExecutorGenerator validates template, generates Python class extending
    DynamicExecutorBase, AST-checks, writes to axon/executors/dynamic/{name}.py
  → Calls AxonService.registry.register_dynamic_executor(template, path)
  → EXECUTOR_REGISTERED emitted - Thymos opens 24h monitoring window

Iron Rules (harder than SubsystemGenerator):
  - Generated class MUST extend DynamicExecutorBase - no direct Executor ABC
  - CANNOT import from systems.* - all comms via Synapse or injected clients
  - CANNOT contain eval(), exec(), __import__(), subprocess, os.system()
  - CANNOT contain wallet private keys, mnemonics, or HMAC/AES secrets inline
  - MUST implement _execute_action() and _validate_action_params()
  - Budget cap lives in DynamicExecutorBase - never in generated code
  - Written to axon/executors/dynamic/{name}.py - never to systems/* directly
  - Auto-registered by this class after validation - NOT deferred to next boot
"""

from __future__ import annotations

import ast
import re
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    from systems.axon.types import ExecutorTemplate
    from systems.simula.code_agent import SimulaCodeAgent
    from systems.simula.rollback import RollbackManager

logger = structlog.get_logger().bind(system="simula.executor_generator")

# ── Iron Rule guards ──────────────────────────────────────────────────────────

_FORBIDDEN_IMPORT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"from\s+systems\.\w+", re.MULTILINE),
    re.compile(r"import\s+systems\.\w+", re.MULTILINE),
]

_FORBIDDEN_CALL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\beval\s*\(", re.MULTILINE),
    re.compile(r"\bexec\s*\(", re.MULTILINE),
    re.compile(r"\b__import__\s*\(", re.MULTILINE),
    re.compile(r"\bsubprocess\b", re.MULTILINE),
    re.compile(r"\bos\.system\s*\(", re.MULTILINE),
    re.compile(r"\bos\.popen\s*\(", re.MULTILINE),
    re.compile(r"\bctypes\b", re.MULTILINE),
]

# Secrets patterns - generated executors must never embed raw credentials
_FORBIDDEN_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"private.?key\s*=\s*['\"][0-9a-fA-F]{32,}", re.MULTILINE | re.IGNORECASE),
    re.compile(r"mnemonic\s*=\s*['\"]", re.MULTILINE | re.IGNORECASE),
]

# Required methods in the generated executor class
_REQUIRED_METHODS: tuple[str, ...] = ("_execute_action", "_validate_action_params")

# Dynamic executor output directory (relative to codebase root)
_DYNAMIC_EXECUTOR_DIR = Path("systems/axon/executors/dynamic")


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class ExecutorGenerationResult:
    """Result of an executor generation + registration attempt."""

    success: bool
    action_type: str
    name: str
    file_path: str = ""
    validation_errors: list[str] = field(default_factory=list)
    reason: str = ""
    generated_at: float = field(default_factory=time.time)


# ── Main class ────────────────────────────────────────────────────────────────


class ExecutorGenerator:
    """
    Generates and immediately registers new Axon executor classes from an
    ExecutorTemplate.

    Unlike SubsystemGenerator (which defers to next boot), ExecutorGenerator
    hot-loads and registers the executor in the current process.  This is safe
    because DynamicExecutorBase enforces all runtime invariants - budget cap,
    Equor approval, audit trail - and the registry's disable_dynamic_executor()
    can instantly gate any misbehaving executor.

    Flow per generation:
      1.  Iron Rule validation (name, imports, dangerous calls, secrets)
      2.  Snapshot target file for rollback (if it exists)
      3.  Build LLM prompt from ExecutorTemplate
      4.  Generate Python class via LLM (60s timeout, scaffold fallback)
      5.  AST syntax check + Iron Rule source scan
      6.  Required method presence check
      7.  Write to axon/executors/dynamic/{name}.py
      8.  Hot-register via ExecutorRegistry.register_dynamic_executor()
      9.  RE_TRAINING_EXAMPLE emitted with category "executor_generation"
    """

    def __init__(
        self,
        code_agent: SimulaCodeAgent,
        rollback_manager: RollbackManager,
        codebase_root: Path | None = None,
    ) -> None:
        self._code_agent = code_agent
        self._llm: Any | None = getattr(code_agent, "_llm", None)
        self._rollback = rollback_manager
        self._root = codebase_root or rollback_manager._root  # noqa: SLF001
        self._event_bus: Any | None = None
        self._axon_registry: Any | None = None  # ExecutorRegistry, injected
        self._generated: list[ExecutorGenerationResult] = []
        self._log = logger

    def set_event_bus(self, bus: Any) -> None:
        self._event_bus = bus

    def set_axon_registry(self, registry: Any) -> None:
        """Inject the live ExecutorRegistry so generated executors can be hot-loaded."""
        self._axon_registry = registry

    # ── Build-error RE training signal ────────────────────────────────────────

    def _emit_build_error(
        self,
        *,
        generated_code: str,
        prompt_used: str,
        error_type: Literal["syntax", "import", "runtime", "verification_timeout",
                            "proof_failed", "sandbox_escape"],
        error_message: str,
        error_traceback: str | None,
        template: ExecutorTemplate,
        lesson: str,
    ) -> None:
        """Fire-and-forget RE_TRAINING_EXAMPLE(outcome_quality=0.0) on any build error.

        Never raises. Uses asyncio.create_task so it cannot block the caller.
        """
        import asyncio

        bus = self._event_bus
        if bus is None:
            return

        async def _emit() -> None:
            try:
                from decimal import Decimal

                from primitives.common import DriveAlignmentVector, SystemID
                from primitives.re_training import RETrainingExample
                from systems.synapse.types import SynapseEvent, SynapseEventType

                safe_code = generated_code[:4000] if generated_code else ""
                safe_prompt = prompt_used[:2000] if prompt_used else ""
                safe_error = error_message[:1000] if error_message else ""
                safe_tb = (error_traceback[:1500] if error_traceback else "")
                safe_lesson = lesson[:500] if lesson else ""

                reasoning_trace = (
                    f"error_type={error_type}\n"
                    f"executor={template.name} action_type={template.action_type}\n"
                    f"protocol={template.protocol_or_platform}\n"
                    f"error={safe_error}\n"
                    + (f"traceback={safe_tb}\n" if safe_tb else "")
                    + f"lesson={safe_lesson}"
                )

                example = RETrainingExample(
                    source_system=SystemID.SIMULA,
                    episode_id=getattr(template, "source_hypothesis_id", "") or "",
                    instruction=safe_prompt or (
                        f"Generate Axon executor {template.name} "
                        f"for {template.protocol_or_platform}: {template.description[:120]}"
                    ),
                    input_context=(
                        f"executor={template.name} "
                        f"action_type={template.action_type} "
                        f"error_type={error_type} "
                        f"strategy=executor_generation"
                    ),
                    output=safe_code,
                    outcome_quality=0.0,
                    category="build_error",
                    reasoning_trace=reasoning_trace,
                    alternatives_considered=[error_type],
                    cost_usd=Decimal("0"),
                    latency_ms=0,
                    constitutional_alignment=DriveAlignmentVector(),
                    domain="software",
                    skill_area="executor_generation",
                    domain_difficulty="expert",
                )
                await bus.emit(SynapseEvent(
                    event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                    source_system="simula.executor_generator",
                    data=example.model_dump(mode="json"),
                ))
            except Exception:
                pass  # Never let emission errors propagate

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_emit())
        except RuntimeError:
            pass  # No running event loop - skip silently

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate_executor(
        self, template: ExecutorTemplate
    ) -> ExecutorGenerationResult:
        """
        Generate and hot-register a new executor from *template*.

        Never raises - errors captured in result.validation_errors.
        """
        log = self._log.bind(
            action_type=template.action_type,
            name=template.name,
            risk_tier=template.risk_tier,
        )
        log.info("executor_generation_started", protocol=template.protocol_or_platform)

        # 1. Iron Rule: template validation
        guard_errors = self._validate_template_iron_rules(template)
        if guard_errors:
            log.warning("executor_template_iron_rule_violation", errors=guard_errors)
            return ExecutorGenerationResult(
                success=False,
                action_type=template.action_type,
                name=template.name,
                validation_errors=guard_errors,
                reason="Iron Rule violation in executor template",
            )

        output_dir = self._root / _DYNAMIC_EXECUTOR_DIR
        output_path = output_dir / f"{template.name}.py"

        # 2. Snapshot for rollback
        if output_path.exists():
            await self._rollback.snapshot(
                proposal_id=f"executor_gen_{template.name}",
                paths=[output_path],
            )

        # 3–4. Generate code via LLM
        prompt = ""
        try:
            prompt = self._build_generation_prompt(template)
            generated_code = await self._generate_code_via_llm(prompt, template)
        except Exception as exc:
            import traceback as _tb
            log.error("executor_code_generation_failed", error=str(exc))
            self._emit_build_error(
                generated_code="",
                prompt_used=prompt,
                error_type="runtime",
                error_message=f"Code generation failed: {exc}",
                error_traceback=_tb.format_exc(),
                template=template,
                lesson=(
                    "Executor code generation raised an exception before producing any output. "
                    "The LLM call or prompt construction failed."
                ),
            )
            return ExecutorGenerationResult(
                success=False,
                action_type=template.action_type,
                name=template.name,
                validation_errors=[f"Code generation failed: {exc}"],
                reason="LLM error",
            )

        # 5. AST + Iron Rule source scan
        validation_errors = self._validate_generated_code(generated_code, template)
        if validation_errors:
            log.warning(
                "executor_code_validation_failed",
                errors=validation_errors,
                preview=generated_code[:300],
            )
            # Classify the dominant error type
            _has_syntax = any(
                "AST syntax" in e or "syntax error" in e.lower()
                for e in validation_errors
            )
            _has_import = any("import" in e.lower() for e in validation_errors)
            _has_dangerous = any(
                "dangerous" in e.lower() or "forbidden" in e.lower()
                for e in validation_errors
            )
            _exec_err_type: Literal[
                "syntax", "import", "runtime", "verification_timeout",
                "proof_failed", "sandbox_escape"
            ] = (
                "syntax" if _has_syntax
                else "sandbox_escape" if _has_dangerous
                else "import" if _has_import
                else "runtime"
            )
            self._emit_build_error(
                generated_code=generated_code,
                prompt_used=prompt,
                error_type=_exec_err_type,
                error_message="; ".join(validation_errors),
                error_traceback=None,
                template=template,
                lesson=(
                    f"Generated executor code failed {_exec_err_type} validation: "
                    + "; ".join(validation_errors[:3])
                ),
            )
            return ExecutorGenerationResult(
                success=False,
                action_type=template.action_type,
                name=template.name,
                validation_errors=validation_errors,
                reason="Generated code failed validation",
            )

        # 6. Write to disk
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path.write_text(generated_code, encoding="utf-8")
            log.info("executor_file_written", path=str(output_path))
        except Exception as exc:
            log.error("executor_file_write_failed", error=str(exc))
            return ExecutorGenerationResult(
                success=False,
                action_type=template.action_type,
                name=template.name,
                validation_errors=[f"File write failed: {exc}"],
                reason="Disk write error",
            )

        # 7. Hot-register in ExecutorRegistry
        if self._axon_registry is not None:
            try:
                self._axon_registry.register_dynamic_executor(
                    template, str(output_path)
                )
                log.info(
                    "executor_hot_registered",
                    action_type=template.action_type,
                )
            except Exception as exc:
                log.error(
                    "executor_hot_registration_failed",
                    error=str(exc),
                )
                # Return the written path but mark as partial failure
                return ExecutorGenerationResult(
                    success=False,
                    action_type=template.action_type,
                    name=template.name,
                    file_path=str(output_path.relative_to(self._root)),
                    validation_errors=[f"Hot-registration failed: {exc}"],
                    reason="Registry error after file write",
                )
        else:
            log.warning(
                "executor_registry_not_wired",
                note="executor written to disk but not hot-loaded - wire set_axon_registry()",
            )

        # 8. Emit RE_TRAINING_EXAMPLE
        await self._emit_re_training(template, generated_code)

        result = ExecutorGenerationResult(
            success=True,
            action_type=template.action_type,
            name=template.name,
            file_path=str(output_path.relative_to(self._root)),
            reason="Generated and hot-registered",
        )
        self._generated.append(result)
        log.info(
            "executor_generation_complete",
            file_path=result.file_path,
            hypothesis_id=template.source_hypothesis_id,
        )
        return result

    def list_generated_executors(self) -> list[dict[str, Any]]:
        """Return metadata about executors generated in this session."""
        return [
            {
                "action_type": r.action_type,
                "name": r.name,
                "file_path": r.file_path,
                "success": r.success,
                "generated_at": r.generated_at,
                "reason": r.reason,
            }
            for r in self._generated
        ]

    # ── Iron Rule validation ──────────────────────────────────────────────────

    def _validate_template_iron_rules(self, template: ExecutorTemplate) -> list[str]:
        """Validate the ExecutorTemplate before generating any code."""
        errors: list[str] = []
        name_lower = template.name.lower()
        action_lower = template.action_type.lower()

        # Cannot target safety-critical systems
        forbidden_targets = {"equor", "simula", "constitution", "invariant", "memory"}
        for fragment in forbidden_targets:
            if fragment in name_lower or fragment in action_lower:
                errors.append(
                    f"Executor name/action_type contains forbidden fragment '{fragment}'"
                )

        if not re.match(r"^[a-z][a-z0-9_]*$", template.name):
            errors.append(
                "Executor name must be snake_case alphanumeric "
                f"(got {template.name!r})"
            )
        if not re.match(r"^[a-z][a-z0-9_.]*$", template.action_type):
            errors.append(
                "action_type must be snake_case (with optional dots) "
                f"(got {template.action_type!r})"
            )
        if template.risk_tier not in ("low", "medium", "high"):
            errors.append(
                f"risk_tier must be 'low'|'medium'|'high' (got {template.risk_tier!r})"
            )
        if template.max_budget_usd <= 0:
            errors.append("max_budget_usd must be positive")

        return errors

    def _validate_generated_code(
        self, code: str, template: ExecutorTemplate
    ) -> list[str]:
        """AST + pattern scan of LLM-generated code."""
        errors: list[str] = []

        # AST syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            errors.append(f"AST syntax error: {exc}")
            return errors  # No point continuing if AST fails

        # Required class present
        class_name = (
            "".join(p.capitalize() for p in template.name.split("_")) + "Executor"
        )
        class_nodes = [
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
        ]
        class_names = [n.name for n in class_nodes]
        if class_name not in class_names:
            errors.append(
                f"Generated code must define class {class_name!r} "
                f"(found: {class_names})"
            )

        # Must extend DynamicExecutorBase
        target_class = next((n for n in class_nodes if n.name == class_name), None)
        if target_class is not None:
            base_names = [
                (b.id if isinstance(b, ast.Name) else
                 b.attr if isinstance(b, ast.Attribute) else "")
                for b in target_class.bases
            ]
            if "DynamicExecutorBase" not in base_names:
                errors.append(
                    f"Class {class_name!r} must extend DynamicExecutorBase "
                    f"(found bases: {base_names})"
                )

        # Required methods
        if target_class is not None:
            method_names = {
                n.name
                for n in ast.walk(target_class)
                if isinstance(n, ast.FunctionDef)
            }
            for method in _REQUIRED_METHODS:
                if method not in method_names:
                    errors.append(
                        f"Class {class_name!r} must implement method {method!r}"
                    )

        # Forbidden import patterns
        for pattern in _FORBIDDEN_IMPORT_PATTERNS:
            if pattern.search(code):
                errors.append(
                    f"Forbidden cross-system import pattern: {pattern.pattern!r}"
                )

        # Forbidden dangerous call patterns
        for pattern in _FORBIDDEN_CALL_PATTERNS:
            if pattern.search(code):
                errors.append(
                    f"Forbidden dangerous call pattern: {pattern.pattern!r}"
                )

        # Forbidden inline secrets
        for pattern in _FORBIDDEN_SECRET_PATTERNS:
            if pattern.search(code):
                errors.append("Detected potential inline secret - remove hardcoded credentials")

        return errors

    # ── Code generation ───────────────────────────────────────────────────────

    def _build_generation_prompt(self, template: ExecutorTemplate) -> str:
        """Build the LLM prompt for executor generation."""
        class_name = (
            "".join(p.capitalize() for p in template.name.split("_")) + "Executor"
        )
        allowed_prefixes = "\n".join(
            f"  - {api}" for api in template.required_apis
        ) or "  (none specified - use _call_api() with any URL from the API)"
        capabilities = ", ".join(template.capabilities) or "general interaction"
        constraints = "\n".join(
            f"  - {c}" for c in template.safety_constraints
        ) or "  (none specified)"

        return textwrap.dedent(f"""
            You are generating an EcodiaOS Axon executor for {template.protocol_or_platform}.
            The class MUST follow the strict template below - no deviations.

            ## Required class signature
            ```python
            from __future__ import annotations
            from typing import Any
            from systems.axon.executors.dynamic_base import DynamicExecutorBase
            from systems.axon.types import ExecutionContext, ExecutionResult

            class {class_name}(DynamicExecutorBase):
                action_type = "{template.action_type}"
                description = "{template.description}"
                required_autonomy = {template.required_autonomy}
                reversible = False
                max_duration_ms = 30_000

                # Allowed API URL prefixes (enforced by _call_api whitelist)
                _allowed_api_prefixes: list[str] = {template.required_apis!r}

                async def _validate_action_params(self, params: dict[str, Any]) -> list[str]:
                    \"\"\"Return a list of validation error strings, or [] if valid.\"\"\"
                    errors = []
                    # TODO: validate required fields
                    return errors

                async def _execute_action(
                    self, params: dict[str, Any], context: ExecutionContext
                ) -> ExecutionResult:
                    \"\"\"Execute the action. Use self._call_api() for all HTTP calls.\"\"\"
                    try:
                        # TODO: implement
                        return ExecutionResult(
                            success=True,
                            data={{}},
                            side_effects=["Action executed on {template.protocol_or_platform}"],
                        )
                    except Exception as exc:
                        return ExecutionResult(success=False, error=str(exc))
            ```

            ## Protocol: {template.protocol_or_platform}
            ## Capabilities: {capabilities}
            ## Required API endpoints:
            {allowed_prefixes}

            ## Safety constraints:
            {constraints}

            ## Iron Rules (NEVER violate):
            1. DO NOT import from systems.* - only from systems.axon.executors.dynamic_base
               and systems.axon.types (these are allowed as they are the base module)
            2. DO NOT use eval(), exec(), __import__(), subprocess, os.system()
            3. DO NOT embed API keys, private keys, or mnemonics as string literals
            4. DO NOT override execute() or validate_params() - use _execute_action() and _validate_action_params()
            5. Budget enforcement is handled by DynamicExecutorBase - do not add your own budget checks
            6. ALL HTTP calls must go through self._call_api() - never use httpx/aiohttp directly

            Write ONLY the Python class code. No markdown fences, no explanation.
        """).strip()

    async def _generate_code_via_llm(
        self, prompt: str, template: ExecutorTemplate
    ) -> str:
        """Call the LLM to generate executor code. Falls back to scaffold on failure."""
        if self._llm is not None:
            try:
                response = await self._llm.generate(
                    prompt=prompt,
                    max_tokens=4096,
                    timeout=60.0,
                    system=(
                        "Output ONLY valid Python source code — no markdown formatting."
                    ),
                )
                code = response.strip()
                # Strip accidental markdown fences
                if code.startswith("```"):
                    code = re.sub(r"^```[a-z]*\n?", "", code)
                    code = re.sub(r"\n?```$", "", code)
                return code
            except Exception as exc:
                self._log.warning(
                    "llm_generation_failed_using_scaffold",
                    error=str(exc),
                )

        # Scaffold fallback - minimal but valid code
        return self._build_scaffold(template)

    def _build_scaffold(self, template: ExecutorTemplate) -> str:
        """Minimal scaffold executor when LLM is unavailable."""
        class_name = (
            "".join(p.capitalize() for p in template.name.split("_")) + "Executor"
        )
        return textwrap.dedent(f"""
            from __future__ import annotations
            from typing import Any
            from systems.axon.executors.dynamic_base import DynamicExecutorBase
            from systems.axon.types import ExecutionContext, ExecutionResult


            class {class_name}(DynamicExecutorBase):
                \"\"\"
                Auto-generated scaffold executor for {template.protocol_or_platform}.

                Description: {template.description}
                Risk tier: {template.risk_tier}
                Max budget: ${template.max_budget_usd} USD per execution

                Safety constraints:
                {chr(10).join(f"    - {c}" for c in template.safety_constraints) or "    (none specified)"}
                \"\"\"

                action_type = "{template.action_type}"
                description = "{template.description}"
                required_autonomy = {template.required_autonomy}
                reversible = False
                max_duration_ms = 30_000

                _allowed_api_prefixes: list[str] = {template.required_apis!r}

                async def _validate_action_params(
                    self, params: dict[str, Any]
                ) -> list[str]:
                    \"\"\"Scaffold - no validation implemented yet.\"\"\"
                    return []

                async def _execute_action(
                    self, params: dict[str, Any], context: ExecutionContext
                ) -> ExecutionResult:
                    \"\"\"
                    Scaffold - returns a not-implemented error.
                    Replace this with actual {template.protocol_or_platform} API calls.
                    \"\"\"
                    return ExecutionResult(
                        success=False,
                        error=(
                            "Executor {class_name} is a scaffold. "
                            "Implement _execute_action() before use."
                        ),
                    )
        """).strip()

    # ── Side effects ──────────────────────────────────────────────────────────

    async def _emit_re_training(
        self, template: ExecutorTemplate, generated_code: str
    ) -> None:
        """Emit a RE_TRAINING_EXAMPLE for the code generation event."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEventType

            await self._event_bus.emit(
                SynapseEventType.RE_TRAINING_EXAMPLE,
                {
                    "category": "executor_generation",
                    "action_type": template.action_type,
                    "protocol_or_platform": template.protocol_or_platform,
                    "risk_tier": template.risk_tier,
                    "capabilities": template.capabilities,
                    "source_hypothesis_id": template.source_hypothesis_id,
                    "code_length": len(generated_code),
                    "is_scaffold": "_execute_action() is a scaffold" in generated_code,
                },
            )
        except Exception:
            self._log.exception(
                "re_training_emit_failed", action_type=template.action_type
            )
