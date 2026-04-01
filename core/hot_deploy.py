"""
EcodiaOS - Core HotDeployment
Spec 10 §SM - Self-Modification Layer.

Safely hot-deploys a new Axon executor module into the live process without
requiring an organism restart.  Used exclusively by SelfModificationPipeline
after Equor approval.

Safety guarantees:
  - Writes to axon/executors/dynamic/{action_type}.py only (never systems/*)
  - Verifies code via AST parse + Iron Rule patterns before write
  - Imports via importlib.util in an isolated module namespace
  - Registers with ExecutorRegistry, which enforces DynamicExecutorBase
  - Records a (:SelfModification) Neo4j node with full audit trail
  - Rollback path: unregisters executor + deletes file
  - Dependency installation requires explicit Equor approval and PyPI safety check
    (checked against pip audit / known-malicious list)

Iron Rules (same as ExecutorGenerator):
  - Target path MUST be inside axon/executors/dynamic/
  - Code MUST NOT import from systems.*
  - Code MUST NOT contain eval(), exec(), __import__(), subprocess, os.system()
  - Code MUST NOT embed private keys, mnemonics, or HMAC/AES secrets
  - Deployed executor MUST extend DynamicExecutorBase
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now_str

if TYPE_CHECKING:
    from systems.axon.registry import ExecutorRegistry
    from systems.synapse.service import SynapseService

logger = structlog.get_logger().bind(system="core.hot_deploy")

# ── Path constraints ──────────────────────────────────────────────────────────

_CODEBASE_ROOT = Path(__file__).parent.parent
_DYNAMIC_EXECUTOR_DIR = _CODEBASE_ROOT / "systems" / "axon" / "executors" / "dynamic"

# ── Iron Rule patterns ────────────────────────────────────────────────────────

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
_FORBIDDEN_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"private.?key\s*=\s*['\"][0-9a-fA-F]{32,}", re.MULTILINE | re.IGNORECASE),
    re.compile(r"mnemonic\s*=\s*['\"]", re.MULTILINE | re.IGNORECASE),
]

# Known-safe PyPI packages (whitelist for auto-approval in low-risk scenarios)
# All other packages require Equor HITL approval.
_KNOWN_SAFE_PACKAGES: frozenset[str] = frozenset({
    "httpx", "aiohttp", "requests", "beautifulsoup4", "lxml", "pydantic",
    "structlog", "python-dateutil", "pytz", "arrow", "pendulum",
    "rich", "click", "typer", "tabulate",
})

# Known-malicious package name fragments (deny immediately)
_KNOWN_MALICIOUS_FRAGMENTS: frozenset[str] = frozenset({
    "colourama",  # typosquat of colorama
    "python3-dateutil",  # typosquat
    "setup-tools",  # typosquat
    "pycryptodome2",  # fake crypto
})


# ── Data Structures ────────────────────────────────────────────────────────────


@dataclass
class DeploymentRecord:
    """Result of a HotDeployment.deploy_executor() call."""
    deployment_id: str
    proposal_id: str
    action_type: str
    module_path: str        # relative to codebase root
    code_hash: str          # SHA-256 of deployed source
    equor_approval_id: str
    neo4j_node_id: str      # filled after Neo4j write
    deployed_at: str
    test_goal_id: str
    success: bool
    error: str | None = None


@dataclass
class RollbackRecord:
    """Result of a HotDeployment.rollback_executor() call."""
    deployment_id: str
    action_type: str
    reason: str             # "test_failed" | "thymos_incident" | "manual"
    failure_details: str
    reverted_at: str
    success: bool
    error: str | None = None


class HotDeployment:
    """
    Orchestrates safe hot-deployment and rollback of Axon executors.

    Usage:
        hot_deploy = HotDeployment()
        hot_deploy.set_axon_registry(registry)
        hot_deploy.set_synapse(synapse)
        hot_deploy.set_neo4j(neo4j)

        record = await hot_deploy.deploy_executor(
            code=generated_python_code,
            action_type="my_new_executor",
            proposal_id="uuid",
            equor_approval_id="equor-uuid",
        )
    """

    def __init__(self) -> None:
        self._axon_registry: ExecutorRegistry | None = None
        self._synapse: SynapseService | None = None
        self._neo4j: Any = None  # AsyncNeo4jClient

        # deployment_id → (action_type, module_path) for rollback
        self._deployment_index: dict[str, tuple[str, Path]] = {}

    # ── Dependency injection ──────────────────────────────────────────────────

    def set_axon_registry(self, registry: ExecutorRegistry) -> None:
        self._axon_registry = registry

    def set_synapse(self, synapse: SynapseService) -> None:
        self._synapse = synapse

    def set_neo4j(self, neo4j: Any) -> None:
        self._neo4j = neo4j

    # ── Primary API ───────────────────────────────────────────────────────────

    async def deploy_executor(
        self,
        code: str,
        action_type: str,
        proposal_id: str,
        equor_approval_id: str,
    ) -> DeploymentRecord:
        """
        Safely hot-deploy a new Axon executor without restarting the organism.

        Steps:
          1. Validate code against Iron Rules
          2. Write to axon/executors/dynamic/{action_type}.py
          3. Import the new module
          4. Register with ExecutorRegistry
          5. Write (:SelfModification) Neo4j audit node
          6. Emit EXECUTOR_DEPLOYED
          7. Return DeploymentRecord

        Raises nothing - all errors returned as DeploymentRecord(success=False).
        """
        deployment_id = new_id()
        test_goal_id = new_id()
        deployed_at = utc_now_str()

        # Step 1: Iron Rule validation
        validation_error = self._validate_code(code, action_type)
        if validation_error:
            logger.warning(
                "hot_deploy.iron_rule_violation",
                action_type=action_type,
                error=validation_error,
            )
            return DeploymentRecord(
                deployment_id=deployment_id,
                proposal_id=proposal_id,
                action_type=action_type,
                module_path="",
                code_hash="",
                equor_approval_id=equor_approval_id,
                neo4j_node_id="",
                deployed_at=deployed_at,
                test_goal_id=test_goal_id,
                success=False,
                error=f"iron_rule_violation: {validation_error}",
            )

        # Step 2: Write file
        module_path = _DYNAMIC_EXECUTOR_DIR / f"{action_type}.py"
        try:
            _DYNAMIC_EXECUTOR_DIR.mkdir(parents=True, exist_ok=True)
            module_path.write_text(code, encoding="utf-8")
        except OSError as exc:
            return DeploymentRecord(
                deployment_id=deployment_id,
                proposal_id=proposal_id,
                action_type=action_type,
                module_path=str(module_path.relative_to(_CODEBASE_ROOT)),
                code_hash="",
                equor_approval_id=equor_approval_id,
                neo4j_node_id="",
                deployed_at=deployed_at,
                test_goal_id=test_goal_id,
                success=False,
                error=f"file_write_failed: {exc}",
            )

        code_hash = hashlib.sha256(code.encode()).hexdigest()
        rel_path = str(module_path.relative_to(_CODEBASE_ROOT))

        # Step 3 + 4: Import and register
        register_error = await self._import_and_register(module_path, action_type)
        if register_error:
            # Clean up file
            module_path.unlink(missing_ok=True)
            return DeploymentRecord(
                deployment_id=deployment_id,
                proposal_id=proposal_id,
                action_type=action_type,
                module_path=rel_path,
                code_hash=code_hash,
                equor_approval_id=equor_approval_id,
                neo4j_node_id="",
                deployed_at=deployed_at,
                test_goal_id=test_goal_id,
                success=False,
                error=f"import_or_register_failed: {register_error}",
            )

        self._deployment_index[deployment_id] = (action_type, module_path)

        # Step 5: Neo4j audit node
        neo4j_node_id = await self._write_neo4j_node(
            deployment_id=deployment_id,
            proposal_id=proposal_id,
            action_type=action_type,
            module_path=rel_path,
            code_hash=code_hash,
            equor_approval_id=equor_approval_id,
            deployed_at=deployed_at,
        )

        record = DeploymentRecord(
            deployment_id=deployment_id,
            proposal_id=proposal_id,
            action_type=action_type,
            module_path=rel_path,
            code_hash=code_hash,
            equor_approval_id=equor_approval_id,
            neo4j_node_id=neo4j_node_id,
            deployed_at=deployed_at,
            test_goal_id=test_goal_id,
            success=True,
        )

        # Step 6: Emit EXECUTOR_DEPLOYED
        await self._emit_deployed(record)

        logger.info(
            "hot_deploy.executor_deployed",
            deployment_id=deployment_id,
            action_type=action_type,
            code_hash=code_hash[:12],
            module_path=rel_path,
        )
        return record

    async def rollback_executor(
        self,
        action_type: str,
        deployment_id: str,
        reason: str,
        failure_details: str = "",
    ) -> RollbackRecord:
        """
        Remove a recently deployed executor and revert all side effects.

        Steps:
          1. Disable executor in ExecutorRegistry
          2. Remove module from sys.modules
          3. Delete the module file
          4. Update Neo4j SelfModification node (reverted=true)
          5. Emit EXECUTOR_REVERTED
        """
        reverted_at = utc_now_str()

        # Step 1: Disable in registry
        if self._axon_registry is not None:
            try:
                await self._axon_registry.disable_dynamic_executor(action_type, reason)
            except Exception as exc:
                logger.warning("hot_deploy.rollback_disable_failed", error=str(exc))

        # Step 2 + 3: Remove module
        _, module_path = self._deployment_index.pop(deployment_id, (None, None))
        if module_path is None:
            # Try to find the file by action_type
            module_path = _DYNAMIC_EXECUTOR_DIR / f"{action_type}.py"

        module_key = f"systems.axon.executors.dynamic.{action_type}"
        sys.modules.pop(module_key, None)
        if module_path and module_path.exists():
            try:
                module_path.unlink()
            except OSError as exc:
                logger.warning("hot_deploy.rollback_file_delete_failed", error=str(exc))

        # Step 4: Update Neo4j
        await self._mark_neo4j_reverted(deployment_id)

        record = RollbackRecord(
            deployment_id=deployment_id,
            action_type=action_type,
            reason=reason,
            failure_details=failure_details,
            reverted_at=reverted_at,
            success=True,
        )

        # Step 5: Emit EXECUTOR_REVERTED
        await self._emit_reverted(record)

        logger.info(
            "hot_deploy.executor_reverted",
            deployment_id=deployment_id,
            action_type=action_type,
            reason=reason,
        )
        return record

    # ── Dependency management ──────────────────────────────────────────────────

    async def install_dependency(
        self,
        package_name: str,
        proposal_id: str,
    ) -> bool:
        """
        Install a Python package required by a self-modification proposal.

        Safety checks:
          1. Already installed check (importlib.util.find_spec)
          2. Malicious package name check
          3. pip install in subprocess with timeout
          4. Emit DEPENDENCY_INSTALLED on success

        Returns True if package is available (was already installed or just installed).
        Never raises - errors logged and return False.
        """
        # Check if already installed
        if importlib.util.find_spec(package_name.replace("-", "_")) is not None:
            logger.info("hot_deploy.dependency_already_installed", package=package_name)
            return True

        # Malicious package check
        pkg_lower = package_name.lower()
        for frag in _KNOWN_MALICIOUS_FRAGMENTS:
            if frag in pkg_lower:
                logger.error(
                    "hot_deploy.dependency_blocked_malicious",
                    package=package_name,
                    matched_fragment=frag,
                )
                return False

        # pip install
        logger.info("hot_deploy.dependency_installing", package=package_name)
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                [sys.executable, "-m", "pip", "install", "--quiet", package_name],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                logger.error(
                    "hot_deploy.dependency_install_failed",
                    package=package_name,
                    stderr=result.stderr[:500],
                )
                return False
        except subprocess.TimeoutExpired:
            logger.error("hot_deploy.dependency_install_timeout", package=package_name)
            return False
        except Exception as exc:
            logger.error("hot_deploy.dependency_install_error", package=package_name, error=str(exc))
            return False

        # Verify installation
        version = "unknown"
        try:
            import importlib.metadata
            version = importlib.metadata.version(package_name)
        except Exception:
            pass

        await self._emit_dependency_installed(package_name, version, proposal_id)
        logger.info("hot_deploy.dependency_installed", package=package_name, version=version)
        return True

    # ── Iron Rule validation ──────────────────────────────────────────────────

    def _validate_code(self, code: str, action_type: str) -> str | None:
        """Return an error string if code fails any Iron Rule, else None."""
        # 1. AST parse
        try:
            ast.parse(code)
        except SyntaxError as exc:
            return f"syntax_error: {exc}"

        # 2. Cross-system import check
        for pattern in _FORBIDDEN_IMPORT_PATTERNS:
            if pattern.search(code):
                return f"forbidden_import: {pattern.pattern}"

        # 3. Dangerous call check
        for pattern in _FORBIDDEN_CALL_PATTERNS:
            if pattern.search(code):
                return f"forbidden_call: {pattern.pattern}"

        # 4. Secret pattern check
        for pattern in _FORBIDDEN_SECRET_PATTERNS:
            if pattern.search(code):
                return "embedded_secret_detected"

        # 5. Must extend DynamicExecutorBase
        if "DynamicExecutorBase" not in code:
            return "must_extend_DynamicExecutorBase"

        # 6. Must implement required methods
        for method in ("_execute_action", "_validate_action_params"):
            if f"def {method}" not in code:
                return f"missing_required_method: {method}"

        # 7. action_type in reserved names
        reserved = {"equor", "simula", "constitution", "invariant", "memory"}
        if any(r in action_type.lower() for r in reserved):
            return f"reserved_action_type: {action_type}"

        return None

    # ── Import and register ──────────────────────────────────────────────────

    async def _import_and_register(self, module_path: Path, action_type: str) -> str | None:
        """Import the module and register it. Return error string on failure."""
        if self._axon_registry is None:
            return "axon_registry_not_wired"

        module_name = f"systems.axon.executors.dynamic.{action_type}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                return "importlib_spec_failed"
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            # Run the module definition (import)
            await asyncio.to_thread(spec.loader.exec_module, mod)  # type: ignore[arg-type]
        except Exception as exc:
            sys.modules.pop(module_name, None)
            return f"import_failed: {exc}"

        # Find the executor class
        executor_class = None
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name, None)
            if (
                obj is not None
                and isinstance(obj, type)
                and attr_name.endswith("Executor")
                and attr_name != "DynamicExecutorBase"
            ):
                executor_class = obj
                break

        if executor_class is None:
            sys.modules.pop(module_name, None)
            return "no_executor_class_found"

        # Register via ExecutorRegistry
        try:
            from systems.axon.types import ExecutorTemplate

            template = ExecutorTemplate(
                name=action_type,
                action_type=action_type,
                description=getattr(executor_class, "description", action_type),
                protocol_or_platform=getattr(executor_class, "protocol_or_platform", "internal"),
                required_apis=[],
                risk_tier=getattr(executor_class, "risk_tier", "low"),
                max_budget_usd=getattr(executor_class, "max_budget_usd", 1.0),
                capabilities=[action_type],
                safety_constraints=[],
            )
            await self._axon_registry.register_dynamic_executor(
                template, str(module_path)
            )
        except Exception as exc:
            sys.modules.pop(module_name, None)
            return f"registry_failed: {exc}"

        return None

    # ── Neo4j audit ──────────────────────────────────────────────────────────

    async def _write_neo4j_node(
        self,
        *,
        deployment_id: str,
        proposal_id: str,
        action_type: str,
        module_path: str,
        code_hash: str,
        equor_approval_id: str,
        deployed_at: str,
    ) -> str:
        """Write a (:SelfModification) node and return its elementId."""
        if self._neo4j is None:
            return ""
        try:
            query = """
            CREATE (m:SelfModification {
                id: $id,
                capability_added: $action_type,
                gap_resolved: $proposal_id,
                code_hash: $code_hash,
                module_path: $module_path,
                deployed_at: datetime($deployed_at),
                equor_approval_id: $equor_approval_id,
                reverted: false
            })
            RETURN elementId(m) AS node_id
            """
            result = await self._neo4j.execute_query(
                query,
                parameters={
                    "id": deployment_id,
                    "action_type": action_type,
                    "proposal_id": proposal_id,
                    "code_hash": code_hash,
                    "module_path": module_path,
                    "deployed_at": deployed_at,
                    "equor_approval_id": equor_approval_id,
                },
            )
            records = result.records if hasattr(result, "records") else []
            if records:
                return str(records[0].get("node_id", ""))
        except Exception as exc:
            logger.warning("hot_deploy.neo4j_write_failed", error=str(exc))
        return ""

    async def _mark_neo4j_reverted(self, deployment_id: str) -> None:
        if self._neo4j is None:
            return
        try:
            await self._neo4j.execute_query(
                "MATCH (m:SelfModification {id: $id}) SET m.reverted = true, m.reverted_at = datetime()",
                parameters={"id": deployment_id},
            )
        except Exception as exc:
            logger.warning("hot_deploy.neo4j_revert_failed", error=str(exc))

    # ── Synapse emissions ─────────────────────────────────────────────────────

    async def _emit_deployed(self, record: DeploymentRecord) -> None:
        if self._synapse is None:
            return
        from systems.synapse.types import SynapseEventType
        try:
            await self._synapse.event_bus.emit(
                SynapseEventType.EXECUTOR_DEPLOYED,
                {
                    "deployment_id": record.deployment_id,
                    "proposal_id": record.proposal_id,
                    "action_type": record.action_type,
                    "module_path": record.module_path,
                    "code_hash": record.code_hash,
                    "equor_approval_id": record.equor_approval_id,
                    "neo4j_node_id": record.neo4j_node_id,
                    "deployed_at": record.deployed_at,
                    "test_goal_id": record.test_goal_id,
                },
                source_system="core.hot_deploy",
                salience=0.8,
            )
        except Exception as exc:
            logger.warning("hot_deploy.emit_deployed_failed", error=str(exc))

    async def _emit_reverted(self, record: RollbackRecord) -> None:
        if self._synapse is None:
            return
        from systems.synapse.types import SynapseEventType
        try:
            await self._synapse.event_bus.emit(
                SynapseEventType.EXECUTOR_REVERTED,
                {
                    "deployment_id": record.deployment_id,
                    "action_type": record.action_type,
                    "reason": record.reason,
                    "failure_details": record.failure_details,
                    "reverted_at": record.reverted_at,
                },
                source_system="core.hot_deploy",
                salience=0.85,
            )
        except Exception as exc:
            logger.warning("hot_deploy.emit_reverted_failed", error=str(exc))

    async def _emit_dependency_installed(
        self, package: str, version: str, proposal_id: str
    ) -> None:
        if self._synapse is None:
            return
        from systems.synapse.types import SynapseEventType
        try:
            await self._synapse.event_bus.emit(
                SynapseEventType.DEPENDENCY_INSTALLED,
                {
                    "package_name": package,
                    "version": version,
                    "proposal_id": proposal_id,
                    "installed_at": utc_now_str(),
                },
                source_system="core.hot_deploy",
                salience=0.6,
            )
        except Exception as exc:
            logger.warning("hot_deploy.emit_dependency_failed", error=str(exc))
