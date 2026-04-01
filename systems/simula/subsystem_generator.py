"""
EcodiaOS - Simula Subsystem Generator
Speciation Bible §8.3 - Organizational Closure.

Enables Simula to generate NEW subsystem modules (not just parameter patches).
This is the constructive test of organizational closure: the organism can create
components it did not have at birth.

The closure loop:
  Evo hypothesis (SUPPORTED, evidence > 5.0, recommended_action="new_subsystem")
  → ConsolidationOrchestrator Phase 8 emits EVOLUTION_CANDIDATE with category="new_subsystem"
  → SimulaService._on_evolution_candidate routes to SubsystemGenerator
  → SubsystemGenerator validates, prompts SimulaCodeAgent, validates generated code
  → Files written to disk: systems/{name}/__init__.py + systems/{name}/CLAUDE.md
  → SUBSYSTEM_GENERATED emitted on Synapse bus
  → Next incarnation picks up the new system via registry.py

Iron Rules enforced:
  - Cannot generate a subsystem whose name or purpose targets Equor, Simula, or constitution
  - Cannot generate code that imports cross-system (all comms via Synapse)
  - Generated code must implement initialize(), shutdown(), health()
  - Generated code must pass AST syntax check before being written to disk
  - No auto-registration - registry.py wiring is manual for the next incarnation

Generated subsystems are NOT auto-loaded. They live on disk for the next boot.
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import structlog

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from systems.simula.code_agent import SimulaCodeAgent
    from systems.simula.constraint_checker import ConstraintSatisfactionChecker
    from systems.simula.rollback import RollbackManager

logger = structlog.get_logger().bind(system="simula.subsystem_generator")

# ── Iron Rule guard: forbidden subsystem names ─────────────────────────────

_FORBIDDEN_NAME_FRAGMENTS: frozenset[str] = frozenset({
    "equor", "simula", "constitution", "invariant", "constitutional",
})

# Forbidden import patterns (regex) that must not appear in generated code
_FORBIDDEN_IMPORT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"from\s+systems\.\w+", re.MULTILINE),
    re.compile(r"import\s+systems\.\w+", re.MULTILINE),
]

# Required method signatures that every EOS service must expose
_REQUIRED_METHODS: tuple[str, ...] = ("initialize", "shutdown", "health")

# Required Synapse subscription pattern - generated service must call subscribe()
_SYNAPSE_SUBSCRIBE_PATTERN: re.Pattern[str] = re.compile(
    r"subscribe\s*\(", re.MULTILINE
)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class SubsystemSpec:
    """Specification for a new subsystem to be generated.

    Created by the Evo→Simula bridge when a hypothesis's recommended_action
    is "new_subsystem".  Can also be constructed manually for CLI testing.
    """

    name: str
    """Short snake_case system name, e.g. 'pattern_detector'."""

    purpose: str
    """Human-readable one-paragraph description of what the system does."""

    trigger_hypothesis_id: str
    """ID of the Evo hypothesis that motivated this subsystem."""

    required_events: list[str] = field(default_factory=list)
    """SynapseEventType names the subsystem must subscribe to."""

    emitted_events: list[str] = field(default_factory=list)
    """SynapseEventType names the subsystem will emit."""

    dependencies: list[str] = field(default_factory=list)
    """Other system IDs this subsystem needs (protocol-injected, not imports)."""

    constraints: list[str] = field(default_factory=list)
    """Architecture invariants and safety constraints from the spec."""


@dataclass
class SubsystemGenerationResult:
    """Result of a subsystem generation attempt."""

    success: bool
    name: str
    purpose: str
    hypothesis_id: str
    file_paths: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    reason: str = ""
    generated_at: float = field(default_factory=time.time)


# ── Main class ───────────────────────────────────────────────────────────────


class SubsystemGenerator:
    """
    Generates new EOS subsystem modules from a SubsystemSpec.

    Flow per generation:
      1. Validate spec against Iron Rules (name, purpose)
      2. Snapshot existing files in target path for rollback
      3. Build LLM prompt (architecture rules + template + spec)
      4. Delegate to SimulaCodeAgent for code generation
      5. Validate generated code (AST, forbidden imports, required methods)
      6. Write files to disk: systems/{name}/__init__.py
      7. Emit SUBSYSTEM_GENERATED on Synapse bus
      8. Record in internal registry

    Does NOT register the subsystem with SynapseService or registry.py.
    That step requires a restart and is performed by the operator.
    """

    def __init__(
        self,
        code_agent: SimulaCodeAgent,
        constraint_checker: ConstraintSatisfactionChecker,
        rollback_manager: RollbackManager,
        codebase_root: Path | None = None,
    ) -> None:
        self._code_agent = code_agent
        # Extract LLM provider from code agent for direct generation calls
        self._llm: LLMProvider | None = getattr(code_agent, "_llm", None)
        self._constraint_checker = constraint_checker
        self._rollback = rollback_manager
        self._root = codebase_root or rollback_manager._root
        self._event_bus: Any = None
        self._generated_subsystems: list[SubsystemGenerationResult] = []
        self._log = logger

    def set_event_bus(self, bus: Any) -> None:
        self._event_bus = bus

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
        spec: SubsystemSpec,
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
                    f"subsystem={spec.name}\n"
                    f"error={safe_error}\n"
                    + (f"traceback={safe_tb}\n" if safe_tb else "")
                    + f"lesson={safe_lesson}"
                )

                example = RETrainingExample(
                    source_system=SystemID.SIMULA,
                    episode_id=spec.trigger_hypothesis_id,
                    instruction=safe_prompt or (
                        f"Generate EOS subsystem module for {spec.name}: {spec.purpose[:120]}"
                    ),
                    input_context=(
                        f"subsystem={spec.name} "
                        f"error_type={error_type} "
                        f"strategy=subsystem_generation"
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
                    skill_area="subsystem_generation",
                    domain_difficulty="expert",
                )
                await bus.emit(SynapseEvent(
                    event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                    source_system="simula.subsystem_generator",
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

    async def generate_subsystem(self, spec: SubsystemSpec) -> SubsystemGenerationResult:
        """
        Generate a new subsystem module from a specification.

        Returns a SubsystemGenerationResult describing success or failure.
        Never raises - all errors are captured in result.validation_errors.
        """
        log = self._log.bind(
            name=spec.name,
            hypothesis_id=spec.trigger_hypothesis_id,
        )
        log.info("subsystem_generation_started", purpose=spec.purpose[:80])

        # 1. Iron Rule validation
        guard_errors = self._validate_spec_iron_rules(spec)
        if guard_errors:
            log.warning("subsystem_spec_iron_rule_violation", errors=guard_errors)
            return SubsystemGenerationResult(
                success=False,
                name=spec.name,
                purpose=spec.purpose,
                hypothesis_id=spec.trigger_hypothesis_id,
                validation_errors=guard_errors,
                reason="Iron Rule violation in subsystem spec",
            )

        target_dir = self._root / "systems" / spec.name
        init_path = target_dir / "__init__.py"

        # 2. Snapshot for rollback (non-fatal if path doesn't exist yet)
        snapshot_paths: list[Path] = []
        if init_path.exists():
            snapshot_paths.append(init_path)
        if snapshot_paths:
            await self._rollback.snapshot(
                proposal_id=f"subsystem_gen_{spec.name}",
                paths=snapshot_paths,
            )

        # 3. Generate service code via direct LLM call
        prompt = ""
        try:
            prompt = await self._build_generation_prompt(spec)
            generated_code = await self._generate_code_via_llm(prompt, spec)
        except Exception as exc:
            import traceback as _tb
            log.error("subsystem_code_generation_failed", error=str(exc))
            self._emit_build_error(
                generated_code="",
                prompt_used=prompt,
                error_type="runtime",
                error_message=f"Code generation failed: {exc}",
                error_traceback=_tb.format_exc(),
                spec=spec,
                lesson=(
                    "Subsystem code generation raised an exception before producing any output. "
                    "The LLM call or prompt construction failed."
                ),
            )
            return SubsystemGenerationResult(
                success=False,
                name=spec.name,
                purpose=spec.purpose,
                hypothesis_id=spec.trigger_hypothesis_id,
                validation_errors=[f"Code generation failed: {exc}"],
                reason="Code agent error",
            )

        # 4. Validate generated code
        validation_errors = await self._validate_generated_code(generated_code, spec)
        if validation_errors:
            log.warning(
                "subsystem_code_validation_failed",
                errors=validation_errors,
                code_preview=generated_code[:200],
            )
            # Classify error type: syntax errors are caught first by _validate_generated_code
            _has_syntax = any("Syntax error" in e for e in validation_errors)
            _has_import = any("import" in e.lower() for e in validation_errors)
            _build_err_type: Literal[
                "syntax", "import", "runtime", "verification_timeout",
                "proof_failed", "sandbox_escape"
            ] = "syntax" if _has_syntax else ("import" if _has_import else "runtime")
            self._emit_build_error(
                generated_code=generated_code,
                prompt_used=prompt,
                error_type=_build_err_type,
                error_message="; ".join(validation_errors),
                error_traceback=None,
                spec=spec,
                lesson=(
                    f"Generated subsystem code failed {_build_err_type} validation: "
                    + "; ".join(validation_errors[:3])
                ),
            )
            return SubsystemGenerationResult(
                success=False,
                name=spec.name,
                purpose=spec.purpose,
                hypothesis_id=spec.trigger_hypothesis_id,
                validation_errors=validation_errors,
                reason="Generated code failed validation",
            )

        # 5. Write to disk
        written_paths: list[str] = []
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            init_path.write_text(generated_code, encoding="utf-8")
            written_paths.append(str(init_path.relative_to(self._root)))
            log.info("subsystem_file_written", path=written_paths[0])
        except Exception as exc:
            log.error("subsystem_file_write_failed", error=str(exc))
            return SubsystemGenerationResult(
                success=False,
                name=spec.name,
                purpose=spec.purpose,
                hypothesis_id=spec.trigger_hypothesis_id,
                validation_errors=[f"File write failed: {exc}"],
                reason="Disk write error",
            )

        # 6. Build and write result
        result = SubsystemGenerationResult(
            success=True,
            name=spec.name,
            purpose=spec.purpose,
            hypothesis_id=spec.trigger_hypothesis_id,
            file_paths=written_paths,
            reason="Generated successfully - available on next incarnation",
        )
        self._generated_subsystems.append(result)

        # 7. Emit SUBSYSTEM_GENERATED on Synapse bus
        await self._emit_generated(spec, written_paths)

        log.info(
            "subsystem_generation_complete",
            files=written_paths,
            hypothesis_id=spec.trigger_hypothesis_id,
        )
        return result

    async def list_generated_subsystems(self) -> list[dict[str, Any]]:
        """Return metadata about all subsystems generated in this session."""
        return [
            {
                "name": r.name,
                "purpose": r.purpose,
                "hypothesis_id": r.hypothesis_id,
                "file_paths": r.file_paths,
                "success": r.success,
                "generated_at": r.generated_at,
                "reason": r.reason,
            }
            for r in self._generated_subsystems
        ]

    # ── Code generation ──────────────────────────────────────────────────────────

    async def _generate_code_via_llm(self, prompt: str, spec: SubsystemSpec) -> str:
        """
        Call the LLM to generate the subsystem Python module code.

        Uses a single non-streaming call with a generous token budget (~4000).
        Falls back to a minimal skeleton if LLM is unavailable.
        """
        if self._llm is None:
            self._log.warning("subsystem_llm_unavailable_using_skeleton", name=spec.name)
            return self._minimal_skeleton(spec)

        import asyncio

        try:
            response = await asyncio.wait_for(
                self._llm.evaluate(
                    prompt=prompt,
                    max_tokens=4096,
                    temperature=0.2,
                    system=(
                        "You are an expert EcodiaOS system architect. "
                        "Generate complete, production-ready Python code only. "
                        "Output the raw Python module - no markdown fences, no explanation."
                    ),
                ),
                timeout=60.0,
            )
            code = response.text.strip()
            # Strip markdown fences if the model added them
            if code.startswith("```"):
                lines = code.splitlines()
                code = "\n".join(
                    line for line in lines
                    if not line.startswith("```")
                )
            return code
        except Exception as exc:
            self._log.warning(
                "subsystem_llm_call_failed_using_skeleton",
                name=spec.name,
                error=str(exc),
            )
            return self._minimal_skeleton(spec)

    def _minimal_skeleton(self, spec: SubsystemSpec) -> str:
        """Return a minimal valid skeleton if LLM generation fails."""
        class_name = _to_class_name(spec.name)
        subscribes = "\n".join(
            f"            # TODO: subscribe to {e}" for e in spec.required_events
        ) or "            pass"
        return f'''"""
EcodiaOS - {class_name} System (auto-generated skeleton)

{spec.purpose}

Triggered by hypothesis: {spec.trigger_hypothesis_id}
"""
from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger().bind(system="{spec.name}")


class {class_name}Service:
    system_id: str = "{spec.name}"

    def __init__(self) -> None:
        self._event_bus: Any = None
        self._log = logger

    def set_synapse(self, synapse: Any) -> None:
        self._event_bus = getattr(synapse, "_event_bus", None)
        if self._event_bus is not None:
            from systems.synapse.types import SynapseEventType  # noqa: F401
{subscribes}

    async def initialize(self) -> None:
        self._log.info("{spec.name}_initialized")

    async def shutdown(self) -> None:
        self._log.info("{spec.name}_shutdown")

    async def health(self) -> dict:
        return {{"status": "healthy", "system": "{spec.name}"}}
'''

    # ── Prompt construction ────────────────────────────────────────────────────

    async def _build_generation_prompt(self, spec: SubsystemSpec) -> str:
        """
        Build the LLM prompt for subsystem code generation.

        Includes:
        - EcodiaOS architecture rules (no cross-system imports, Synapse-only comms)
        - BaseService pattern with all required methods
        - The SubsystemSpec requirements (events, dependencies, purpose)
        - Iron Rules as negative constraints
        - Example skeleton
        """
        deps_text = (
            ", ".join(spec.dependencies) if spec.dependencies
            else "none (standalone)"
        )
        subscribe_text = (
            "\n".join(f"  - {e}" for e in spec.required_events)
            if spec.required_events else "  - (none specified)"
        )
        emit_text = (
            "\n".join(f"  - {e}" for e in spec.emitted_events)
            if spec.emitted_events else "  - (none specified)"
        )
        constraints_text = (
            "\n".join(f"  - {c}" for c in spec.constraints)
            if spec.constraints else "  - (none beyond standard Iron Rules)"
        )

        return f'''Generate a new EcodiaOS subsystem module for `systems/{spec.name}/__init__.py`.

## System purpose
{spec.purpose}

## Triggered by hypothesis
{spec.trigger_hypothesis_id}

## Events to subscribe to (via synapse._event_bus.subscribe)
{subscribe_text}

## Events to emit (via synapse._event_bus.emit with SynapseEvent)
{emit_text}

## Dependencies (inject via set_* methods, never import directly)
{deps_text}

## Additional constraints
{constraints_text}

## Architecture rules (MANDATORY - violations cause rejection)
1. NO cross-system imports. Do not import from `systems.*` directly.
   All inter-system communication via Synapse event bus only.
2. Import types ONLY from `primitives.*` and `systems.synapse.types`.
3. Implement these three methods:
   - `async def initialize(self) -> None`
   - `async def shutdown(self) -> None`
   - `async def health(self) -> dict`
4. Subscribe to required_events in `set_synapse(synapse)` or `initialize()`.
5. Emit events using:
   ```python
   from systems.synapse.types import SynapseEvent, SynapseEventType
   await self._event_bus.emit(SynapseEvent(
       event_type=SynapseEventType.<EVENT_NAME>,
       source_system="{spec.name}",
       data={{...}},
   ))
   ```
6. Use `structlog.get_logger().bind(system="{spec.name}")` for logging.
7. All DB/network I/O must use `async def` with `await`.
8. Class name: `{_to_class_name(spec.name)}Service` (or `{_to_class_name(spec.name)}` if not a service).
9. Add `system_id: str = "{spec.name}"` as a class attribute.

## Skeleton (adapt, do not copy verbatim)
```python
"""
EcodiaOS - {_to_class_name(spec.name)} System

{spec.purpose}

Triggered by hypothesis: {spec.trigger_hypothesis_id}
"""
from __future__ import annotations
import asyncio
from typing import Any
import structlog
from primitives.common import new_id, utc_now

logger = structlog.get_logger().bind(system="{spec.name}")


class {_to_class_name(spec.name)}Service:
    system_id: str = "{spec.name}"

    def __init__(self) -> None:
        self._event_bus: Any = None
        self._log = logger

    def set_synapse(self, synapse: Any) -> None:
        self._event_bus = getattr(synapse, "_event_bus", None)
        if self._event_bus is not None:
            from systems.synapse.types import SynapseEventType
            # Subscribe to required events here
            pass

    async def initialize(self) -> None:
        self._log.info("{spec.name}_initialized")

    async def shutdown(self) -> None:
        self._log.info("{spec.name}_shutdown")

    async def health(self) -> dict:
        return {{"status": "healthy", "system": "{spec.name}"}}
```

Generate the full implementation. Be complete - do not use placeholder comments.
'''

    # ── Code validation ───────────────────────────────────────────────────────

    async def _validate_generated_code(
        self, code: str, spec: SubsystemSpec
    ) -> list[str]:
        """
        Validate generated Python code before writing to disk.

        Checks:
          1. AST syntax parse
          2. No forbidden cross-system imports
          3. Required methods present (initialize, shutdown, health)
          4. At least one Synapse subscribe() call (if required_events specified)
          5. Iron Rules: no references to equor/simula/constitution modifications
        """
        errors: list[str] = []

        # 1. AST syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            errors.append(f"Syntax error: {exc}")
            return errors  # Cannot continue if code won't parse

        # 2. Forbidden cross-system imports
        for pattern in _FORBIDDEN_IMPORT_PATTERNS:
            match = pattern.search(code)
            if match:
                errors.append(
                    f"Forbidden import detected: '{match.group()}'. "
                    "Use Synapse events for all cross-system communication."
                )

        # 3. Required methods
        defined_functions: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined_functions.add(node.name)

        for required in _REQUIRED_METHODS:
            if required not in defined_functions:
                errors.append(
                    f"Missing required method: '{required}'. "
                    "All EOS services must implement initialize(), shutdown(), health()."
                )

        # 4. Synapse subscribe() present when required_events are specified
        if spec.required_events and not _SYNAPSE_SUBSCRIBE_PATTERN.search(code):
            errors.append(
                "Spec requires subscribing to events but no 'subscribe(' call found in generated code."
            )

        # 5. Iron Rule: no modification of forbidden systems
        code_lower = code.lower()
        for forbidden in ("equor", "constitution", "invariant"):
            if re.search(
                rf"\b{forbidden}\b.*\b(modif|replac|rewrit|patch|chang|updat|delet|remov)\b",
                code_lower,
            ):
                errors.append(
                    f"Iron Rule violation: generated code references modifying '{forbidden}'."
                )

        return errors

    # ── Iron Rule spec validation ─────────────────────────────────────────────

    def _validate_spec_iron_rules(self, spec: SubsystemSpec) -> list[str]:
        """Validate the SubsystemSpec against Iron Rules before code generation."""
        errors: list[str] = []
        name_lower = spec.name.lower()
        purpose_lower = spec.purpose.lower()

        for fragment in _FORBIDDEN_NAME_FRAGMENTS:
            if fragment in name_lower:
                errors.append(
                    f"Subsystem name '{spec.name}' contains forbidden fragment '{fragment}'. "
                    "Cannot generate subsystems targeting protected systems."
                )
            if fragment in purpose_lower and "modif" in purpose_lower:
                errors.append(
                    f"Subsystem purpose references modifying '{fragment}' - forbidden by Iron Rules."
                )

        if not spec.name or not re.match(r"^[a-z][a-z0-9_]*$", spec.name):
            errors.append(
                f"Subsystem name '{spec.name}' must be snake_case (lowercase letters, digits, underscores)."
            )

        if len(spec.purpose) < 20:
            errors.append("Subsystem purpose must be at least 20 characters.")

        return errors

    # ── Synapse emission ──────────────────────────────────────────────────────

    async def _emit_generated(
        self, spec: SubsystemSpec, file_paths: list[str]
    ) -> None:
        """Emit SUBSYSTEM_GENERATED on the Synapse event bus (non-fatal)."""
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.SUBSYSTEM_GENERATED,
                    source_system="simula.subsystem_generator",
                    data={
                        "name": spec.name,
                        "purpose": spec.purpose,
                        "file_paths": file_paths,
                        "hypothesis_id": spec.trigger_hypothesis_id,
                        "validation_passed": True,
                        "required_events": spec.required_events,
                        "emitted_events": spec.emitted_events,
                    },
                )
            )
        except Exception as exc:
            self._log.warning("subsystem_generated_emit_failed", error=str(exc))

        # Also emit NOVEL_ACTION_CREATED so Nova and Evo subscribers fire.
        # A new subsystem is a novel organisational action - observers treat it
        # identically to a newly hot-loaded executor.
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.NOVEL_ACTION_CREATED,
                    source_system="simula.subsystem_generator",
                    data={
                        "proposal_id": spec.trigger_hypothesis_id or spec.name,
                        "action_name": spec.name,
                        "description": spec.purpose,
                        "required_capabilities": spec.dependencies,
                        "executor_class": f"{spec.name.title().replace('_', '')}Service",
                        "module_path": f"systems/{spec.name}/__init__.py",
                        "risk_tier": "medium",
                        "max_budget_usd": 0.0,
                        "equor_approved": True,
                        "source_hypothesis_id": spec.trigger_hypothesis_id or "",
                        "created_at": utc_now().isoformat(),
                        "success": True,
                    },
                )
            )
        except Exception as exc:
            self._log.warning("novel_action_created_emit_failed", error=str(exc))


# ── Helpers ───────────────────────────────────────────────────────────────────


def _to_class_name(snake: str) -> str:
    """Convert snake_case system name to PascalCase class name."""
    return "".join(part.capitalize() for part in snake.split("_"))


def spec_from_proposal_data(data: dict[str, Any]) -> SubsystemSpec:
    """
    Parse a SubsystemSpec from the payload of an EVOLUTION_CANDIDATE
    Synapse event where category == 'new_subsystem'.

    Expected payload keys (all optional except name + purpose):
      subsystem_name, subsystem_purpose, hypothesis_id,
      required_events, emitted_events, dependencies, constraints
    """
    return SubsystemSpec(
        name=data.get("subsystem_name", "unnamed_subsystem"),
        purpose=data.get("subsystem_purpose", data.get("mutation_description", "")),
        trigger_hypothesis_id=data.get("hypothesis_id", ""),
        required_events=data.get("required_events", []),
        emitted_events=data.get("emitted_events", []),
        dependencies=data.get("dependencies", []),
        constraints=data.get("constraints", []),
    )
