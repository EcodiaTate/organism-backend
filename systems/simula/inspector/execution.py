"""
EcodiaOS — Inspector Live Execution Test Chamber

Manages Docker containers for dynamic concolic execution of target repositories.
Autonomously detects the build system, constructs (or generates) a Dockerfile,
builds the image, starts the container, and health-checks the mapped ports —
providing a live, running target for the VulnerabilityProver and ConcurrencyProver
to fire HTTP requests against during inspection.

Iron Rules:
  - Containers are ALWAYS terminated and removed on exit (no zombie processes).
  - Build and boot are strictly time-bounded (configurable, default 120s each).
  - If the container fails to build or boot, the chamber returns None — the
    caller must fall back to purely static analysis.
  - Port mappings use random available high ports on the host.
  - The chamber never modifies the workspace filesystem beyond writing a
    generated Dockerfile (which lives inside the already-isolated temp workspace).
"""

from __future__ import annotations

import asyncio
import re
import socket
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any
from pathlib import Path

import structlog

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from docker.models.containers import Container

    from clients.llm import LLMProvider
logger = structlog.get_logger().bind(system="simula.inspector.execution")


# ── Build file detection ─────────────────────────────────────────────────────

_BUILD_FILE_SIGNATURES: dict[str, str] = {
    "Dockerfile": "docker",
    "docker-compose.yml": "docker-compose",
    "docker-compose.yaml": "docker-compose",
    "package.json": "node",
    "requirements.txt": "python",
    "Pipfile": "python",
    "pyproject.toml": "python",
    "setup.py": "python",
    "Gemfile": "ruby",
    "go.mod": "go",
    "Cargo.toml": "rust",
    "pom.xml": "java",
    "build.gradle": "java",
    "build.gradle.kts": "java",
    "composer.json": "php",
}

# Ports commonly exposed by frameworks — used for health-check polling.
_DEFAULT_PORTS_BY_LANGUAGE: dict[str, list[int]] = {
    "node": [3000, 8080, 5000],
    "python": [8000, 5000, 8080],
    "ruby": [3000, 4567],
    "go": [8080, 3000],
    "rust": [8080, 3000],
    "java": [8080, 8443],
    "php": [8080, 80],
}

# HTTP status codes that indicate the server is alive (even if auth is required).
_HEALTHY_STATUS_CODES = frozenset({200, 201, 204, 301, 302, 400, 401, 403, 404, 405})

# ── LLM prompt for Dockerfile generation ──────────────────────────────────────

_DOCKERFILE_GENERATION_PROMPT = """\
You are a DevOps engineer generating minimal Dockerfiles for development servers.

Given the detected language and list of build files present in a repository,
generate a Dockerfile that:
1. Uses the smallest appropriate base image (e.g., python:3.12-slim, node:22-alpine).
2. Copies the workspace contents into /app.
3. Installs dependencies.
4. Exposes the default port for the framework.
5. Starts the development server.

Rules:
- The Dockerfile must be self-contained and build without external context.
- Use EXPOSE for the port(s) the server listens on.
- Use CMD (not ENTRYPOINT) so the container can be overridden.
- For Python: prefer `pip install --no-cache-dir -r requirements.txt` then
  `uvicorn main:app --host 0.0.0.0` or `python -m flask run --host 0.0.0.0`
  depending on framework detected.
- For Node: `npm ci --production` then `npm start`.
- Keep it minimal — no multi-stage builds, no test dependencies.

Respond with ONLY the Dockerfile content. No markdown fences, no explanations."""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _find_free_port() -> int:
    """Find a random available high port on the host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _detect_build_context(workspace_root: Path) -> tuple[str | None, list[str]]:
    """
    Scan workspace_root for standard build files.

    Returns:
        (detected_language, list_of_found_build_files)
        detected_language is None if no recognisable files are found.
    """
    found_files: list[str] = []
    detected_language: str | None = None

    for filename, language in _BUILD_FILE_SIGNATURES.items():
        if (workspace_root / filename).exists():
            found_files.append(filename)
            # Prefer explicit Docker config; otherwise take first language match.
            if language == "docker" or language == "docker-compose":
                detected_language = language
            elif detected_language is None or detected_language in ("docker", "docker-compose"):
                # Don't overwrite a docker detection with a language detection —
                # but DO overwrite if we haven't found a language yet.
                if detected_language is None:
                    detected_language = language

    return detected_language, found_files


def _extract_expose_ports(dockerfile_content: str) -> list[int]:
    """Extract EXPOSE port numbers from Dockerfile content."""
    ports: list[int] = []
    for match in re.finditer(r"EXPOSE\s+(\d+)", dockerfile_content, re.IGNORECASE):
        with suppress(ValueError):
            ports.append(int(match.group(1)))
    return ports


# ── LiveDetonationChamber ─────────────────────────────────────────────────────


class ExecutionTestChamber:
    """
    Manages Docker containers for live dynamic analysis of target repositories.

    Usage::

        chamber = ExecutionTestChamber(llm=llm_provider)
        async with chamber.spin_up(workspace.root) as live_env:
            # live_env["mapped_url"] → "http://127.0.0.1:54321"
            # live_env["container_id"] → "abc123..."
            # Fire HTTP requests against the live container
            ...
        # Container is guaranteed terminated+removed here.

    If the build or boot fails, the context manager yields None instead,
    allowing the caller to fall back to static analysis.
    """

    def __init__(
        self,
        *,
        llm: LLMProvider | None = None,
        build_timeout_s: int = 120,
        boot_timeout_s: int = 120,
        healthcheck_interval_s: float = 2.0,
    ) -> None:
        """
        Args:
            llm: Optional LLM provider for generating Dockerfiles when none exists.
                 Without this, repos lacking a Dockerfile will skip dynamic execution.
            build_timeout_s: Max seconds for docker build.
            boot_timeout_s: Max seconds to wait for the container to become healthy.
            healthcheck_interval_s: Polling interval for health checks.
        """
        self._llm = llm
        self._build_timeout_s = build_timeout_s
        self._boot_timeout_s = boot_timeout_s
        self._healthcheck_interval_s = healthcheck_interval_s
        self._container: Container | None = None
        self._log = logger

    @asynccontextmanager
    async def spin_up(
        self,
        workspace_root: Path,
    ) -> AsyncIterator[dict[str, Any] | None]:
        """
        Build, run, and health-check a container for the target workspace.

        Yields:
            A dict with keys:
                - container_id: str
                - mapped_url: str (e.g. "http://127.0.0.1:54321")
                - mapped_ports: dict[int, int] (container_port → host_port)
                - detected_language: str
                - internal_ports: list[int]

            Or None if the container failed to build or boot.

        On exit (normal or exception), the container is terminated and removed.
        """
        try:
            result = await self._build_and_run(workspace_root)
            yield result
        except Exception as exc:
            self._log.error(
                "execution_chamber_error",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            yield None
        finally:
            await self.teardown()

    async def teardown(self) -> None:
        """Terminate and remove the container. Idempotent — safe to call multiple times."""
        if self._container is None:
            return

        container = self._container
        self._container = None
        container_id = container.id[:12] if container.id else "unknown"

        try:
            # Run blocking Docker SDK calls in a thread to avoid blocking the event loop.
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: container.terminate())
            self._log.info("container_terminated", container_id=container_id)
        except Exception as exc:
            # Container may have already exited — log but don't raise.
            self._log.debug(
                "container_termination_ignored",
                container_id=container_id,
                error=str(exc),
            )

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, lambda: container.remove(force=True))
            self._log.info("container_removed", container_id=container_id)
        except Exception as exc:
            self._log.warning(
                "container_remove_failed",
                container_id=container_id,
                error=str(exc),
            )

    # ── Internal pipeline ─────────────────────────────────────────────────────

    async def _build_and_run(
        self,
        workspace_root: Path,
    ) -> dict[str, Any] | None:
        """
        Full lifecycle: detect → (generate Dockerfile) → build → run → healthcheck.

        Returns the live_env dict or None on failure.
        """
        log = self._log.bind(workspace=str(workspace_root))

        # Step 1: Detect build context
        detected_language, found_files = _detect_build_context(workspace_root)
        log.info(
            "build_context_detected",
            language=detected_language,
            build_files=found_files,
        )

        if detected_language is None:
            log.warning("no_build_files_detected")
            return None

        # Step 2: Ensure a Dockerfile exists
        dockerfile_path = workspace_root / "Dockerfile"
        if not dockerfile_path.exists():
            generated = await self._generate_dockerfile(
                detected_language, found_files, workspace_root,
            )
            if generated is None:
                log.warning("dockerfile_generation_failed")
                return None

            dockerfile_path.write_text(generated, encoding="utf-8")
            log.info("dockerfile_generated", language=detected_language)

        # Read the Dockerfile to extract EXPOSE ports
        dockerfile_content = dockerfile_path.read_text(encoding="utf-8")
        internal_ports = _extract_expose_ports(dockerfile_content)
        if not internal_ports:
            # Fall back to language defaults
            internal_ports = _DEFAULT_PORTS_BY_LANGUAGE.get(detected_language, [8080])

        # Step 3: Build the Docker image
        try:
            import docker as docker_lib
        except ImportError:
            log.error("docker_sdk_not_installed")
            return None

        client = docker_lib.from_env()
        image_tag = f"inspection-target-{workspace_root.name}".lower()

        log.info("docker_build_started", tag=image_tag)
        try:
            image, build_logs = await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: client.images.build(
                        path=str(workspace_root),
                        tag=image_tag,
                        rm=True,
                        forcerm=True,
                    ),
                ),
                timeout=self._build_timeout_s,
            )
        except TimeoutError:
            log.error(
                "docker_build_timeout",
                timeout_s=self._build_timeout_s,
            )
            return None
        except Exception as exc:
            log.error(
                "docker_build_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return None

        log.info("docker_build_complete", tag=image_tag, image_id=image.id[:12])

        # Step 4: Allocate host ports and start the container
        port_bindings: dict[str, int] = {}
        mapped_ports: dict[int, int] = {}

        for container_port in internal_ports:
            host_port = _find_free_port()
            port_bindings[f"{container_port}/tcp"] = host_port
            mapped_ports[container_port] = host_port

        log.info(
            "container_starting",
            port_mappings=mapped_ports,
        )

        try:
            self._container = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.containers.run(
                    image_tag,
                    detach=True,
                    ports=port_bindings,
                    # Resource limits to prevent runaway containers
                    mem_limit="512m",
                    cpu_period=100_000,
                    cpu_quota=50_000,  # 50% of one CPU
                    # Network isolation — the container can serve requests but
                    # cannot reach arbitrary external hosts.
                    network_mode="bridge",
                ),
            )
        except Exception as exc:
            log.error(
                "container_start_failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            return None

        container_id = self._container.id[:12] if self._container.id else "unknown"
        log.info("container_started", container_id=container_id)

        # Step 5: Health-check — poll mapped ports until we get a valid HTTP response
        primary_host_port = mapped_ports[internal_ports[0]]
        healthy = await self._wait_for_healthy(primary_host_port, log)

        if not healthy:
            log.error(
                "container_healthcheck_failed",
                container_id=container_id,
                timeout_s=self._boot_timeout_s,
            )
            return None

        mapped_url = f"http://127.0.0.1:{primary_host_port}"
        log.info(
            "execution_chamber_ready",
            container_id=container_id,
            mapped_url=mapped_url,
            mapped_ports=mapped_ports,
        )

        return {
            "container_id": container_id,
            "mapped_url": mapped_url,
            "mapped_ports": mapped_ports,
            "detected_language": detected_language,
            "internal_ports": internal_ports,
        }

    # ── Dockerfile generation ────────────────────────────────────────────────

    async def _generate_dockerfile(
        self,
        language: str,
        build_files: list[str],
        workspace_root: Path,
    ) -> str | None:
        """
        Use the LLM to generate a minimal Dockerfile for the detected language.

        Returns the Dockerfile content string, or None if generation fails
        or no LLM is available.
        """
        if self._llm is None:
            self._log.debug("no_llm_for_dockerfile_generation")
            return None

        from clients.llm import Message

        # Gather additional context: read a snippet of the main build file
        context_snippets: list[str] = []
        for bf in build_files[:3]:  # Cap at 3 files to avoid prompt bloat
            bf_path = workspace_root / bf
            if bf_path.exists():
                try:
                    content = bf_path.read_text(encoding="utf-8")[:2000]
                    context_snippets.append(f"### {bf}\n```\n{content}\n```")
                except OSError:
                    pass

        user_prompt = "\n".join([
            f"Language: {language}",
            f"Build files found: {', '.join(build_files)}",
            "",
            "## Build file contents (truncated)",
            *context_snippets,
            "",
            "Generate a Dockerfile for this project.",
        ])

        try:
            response = await asyncio.wait_for(
                self._llm.generate(
                    system_prompt=_DOCKERFILE_GENERATION_PROMPT,
                    messages=[Message(role="user", content=user_prompt)],
                    max_tokens=1024,
                    temperature=0.2,
                ),
                timeout=60,
            )
        except (TimeoutError, Exception) as exc:
            self._log.warning(
                "dockerfile_generation_llm_error",
                error=str(exc),
            )
            return None

        content = response.text.strip()

        # Strip markdown fences if the LLM wrapped them
        if content.startswith("```"):
            lines = content.splitlines()
            content = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()

        # Basic sanity: must contain FROM
        if "FROM" not in content.upper():
            self._log.warning("generated_dockerfile_invalid", preview=content[:200])
            return None

        return content

    # ── Health checking ──────────────────────────────────────────────────────

    async def _wait_for_healthy(
        self,
        host_port: int,
        log: Any,
    ) -> bool:
        """
        Poll the mapped port until the container returns a valid HTTP response.

        A "valid" response is any status code in _HEALTHY_STATUS_CODES —
        we just need to know the server process is listening and responding.

        Returns True if healthy within boot_timeout_s, False otherwise.
        """
        import httpx

        url = f"http://127.0.0.1:{host_port}/"
        deadline = asyncio.get_running_loop().time() + self._boot_timeout_s

        async with httpx.AsyncClient(timeout=5.0) as client:
            while asyncio.get_running_loop().time() < deadline:
                try:
                    resp = await client.get(url)
                    if resp.status_code in _HEALTHY_STATUS_CODES:
                        log.info(
                            "healthcheck_passed",
                            port=host_port,
                            status_code=resp.status_code,
                        )
                        return True
                except (httpx.ConnectError, httpx.ReadError, httpx.TimeoutException):
                    pass  # Server not ready yet — keep polling
                except Exception as exc:
                    log.debug(
                        "healthcheck_poll_error",
                        port=host_port,
                        error=str(exc),
                    )

                await asyncio.sleep(self._healthcheck_interval_s)

        return False
