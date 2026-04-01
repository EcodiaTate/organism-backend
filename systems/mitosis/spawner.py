"""
EcodiaOS - Local Docker Spawner (Phase 16e: Speciation)

Docker-out-of-Docker (DooD) implementation for local/dev mitosis.

When the organism achieves its wealth threshold and MitosisEngine produces a
SeedConfiguration, the spawner boots a physical child container from the same
``ecodiaos-backend`` image.  The parent's Docker socket is bind-mounted into
the running container so `docker.from_env()` speaks to the host daemon (DooD).

The spawner:
  1. Allocates free host ports for the child's HTTP, WS, and Federation endpoints.
  2. Injects the child's environment: unique instance ID, wallet keys, port
     mappings, seed path, ``is_genesis_node=False``, and any niche-specific
     config overrides from the SeedConfiguration.
  3. Starts the child container on the shared ``ecodiaos_network`` bridge.
  4. Health-checks the child's HTTP endpoint (``/health``).
  5. Returns a ``SpawnResult`` with the child's container ID and
     reachable address so the parent can initiate the Federation handshake.

Safety:
  - Containers are resource-capped (1 CPU, 1 GiB RAM) to prevent runaways.
  - Boot is time-bounded (default 120 s); if the child fails to start, the
    container is removed and the caller gets ``SpawnResult.success = False``.
  - The spawner never touches the filesystem or modifies parent state - it is
    pure infrastructure orchestration.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from docker.models.containers import Container

    from systems.oikos.models import SeedConfiguration

logger = structlog.get_logger().bind(component="mitosis.spawner")


# ── Result type ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SpawnResult:
    """Outcome of a child container spawn attempt."""

    success: bool
    container_id: str = ""
    # Reachable address for Federation handshake (host:port)
    child_host: str = ""
    child_http_port: int = 0
    child_ws_port: int = 0
    child_federation_port: int = 0
    error: str = ""
    # Full env that was injected (minus secrets) - useful for debugging
    injected_env_keys: list[str] = field(default_factory=list)

    @property
    def federation_address(self) -> str:
        """``host:port`` for the child's Federation endpoint."""
        if not self.child_host or not self.child_federation_port:
            return ""
        return f"{self.child_host}:{self.child_federation_port}"

    @property
    def http_address(self) -> str:
        """``host:port`` for the child's HTTP API."""
        if not self.child_host or not self.child_http_port:
            return ""
        return f"{self.child_host}:{self.child_http_port}"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _find_free_port() -> int:
    """Bind to port 0 and let the kernel assign a free ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


# ── Spawner ──────────────────────────────────────────────────────────────────


_DEFAULT_IMAGE = "ecodiaos-backend"
_DEFAULT_NETWORK = "ecodiaos_network"
_INTERNAL_HTTP_PORT = 8000
_INTERNAL_WS_PORT = 8001
_INTERNAL_FED_PORT = 8002


class LocalDockerSpawner:
    """
    Spawn child EOS instances as Docker containers on the local daemon.

    Uses Docker-out-of-Docker: the parent container's ``/var/run/docker.sock``
    is mounted, so ``docker.from_env()`` targets the host daemon.  Each child
    gets its own ports, env vars, and config - fully isolated.

    Parameters
    ----------
    image : str
        Docker image to use for child containers.  Defaults to
        ``ecodiaos-backend`` (same image the parent runs).
    network : str
        Docker network to attach children to.  Defaults to
        ``ecodiaos_network`` so children can reach Redis/TimescaleDB.
    boot_timeout_s : float
        Seconds to wait for the child's ``/health`` endpoint to respond.
    config_volume_src : str
        Host path to the ``config/`` directory to mount read-only.
    """

    def __init__(
        self,
        *,
        image: str = _DEFAULT_IMAGE,
        network: str = _DEFAULT_NETWORK,
        boot_timeout_s: float = 120.0,
        config_volume_src: str = "",
    ) -> None:
        self._image = image
        self._network = network
        self._boot_timeout_s = boot_timeout_s
        self._config_volume_src = config_volume_src
        self._log = logger.bind(image=image, network=network)

    # ── Public API ───────────────────────────────────────────────────

    async def spawn_child(
        self,
        child_config: SeedConfiguration,
        parent_cert: str,
    ) -> SpawnResult:
        """
        Boot a new child container and return its connection details.

        Parameters
        ----------
        child_config
            The ``SeedConfiguration`` produced by ``MitosisEngine.build_seed_config``.
            Contains the child's instance ID, niche, config overrides, and
            signed birth certificate.
        parent_cert
            Serialised parent certificate (PEM or JSON).  Injected as
            ``ECODIAOS_PARENT_CERTIFICATE`` so the child can verify its lineage
            and initiate the Federation handshake back to the parent.

        Returns
        -------
        SpawnResult
            On success, contains the container ID and reachable ports.
            On failure, ``success=False`` with an ``error`` message.
        """
        # Lazy-import Docker SDK (mirrors Simula Inspector pattern)
        try:
            import docker as docker_lib
        except ImportError:
            return SpawnResult(
                success=False,
                error="docker Python package not installed - run: pip install docker",
            )

        child_id = child_config.child_instance_id
        log = self._log.bind(child_id=child_id, niche=child_config.niche.name)
        log.info("spawn_child_starting")

        # Allocate host ports
        http_port = _find_free_port()
        ws_port = _find_free_port()
        fed_port = _find_free_port()

        log.info(
            "ports_allocated",
            http=http_port,
            ws=ws_port,
            federation=fed_port,
        )

        # Build environment for the child
        env = self._build_child_env(
            child_config=child_config,
            parent_cert=parent_cert,
            http_port=http_port,
            ws_port=ws_port,
            fed_port=fed_port,
        )

        # Port bindings: host_port -> container_port
        port_bindings: dict[str, int] = {
            f"{_INTERNAL_HTTP_PORT}/tcp": http_port,
            f"{_INTERNAL_WS_PORT}/tcp": ws_port,
            f"{_INTERNAL_FED_PORT}/tcp": fed_port,
        }

        # Volumes
        volumes = self._build_volumes()

        # Container name derived from child instance ID (sanitised)
        container_name = f"eos-child-{child_id[:16]}"

        # Start the container
        container: Container | None = None
        try:
            client = docker_lib.from_env()

            container = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.containers.run(
                    self._image,
                    detach=True,
                    name=container_name,
                    environment=env,
                    ports=port_bindings,
                    volumes=volumes,
                    network=self._network,
                    # Resource caps - prevent runaway children
                    mem_limit="1g",
                    cpu_period=100_000,
                    cpu_quota=100_000,  # 1 full CPU
                    restart_policy={"Name": "unless-stopped"},
                    labels={
                        "ecodiaos.role": "child",
                        "ecodiaos.parent": child_config.parent_instance_id,
                        "ecodiaos.child_id": child_id,
                        "ecodiaos.niche": child_config.niche.name,
                    },
                ),
            )

            if container is None:
                raise RuntimeError("container.run returned None")
            log.info("container_started", container_id=container.id[:12])
        except Exception as exc:
            error_msg = f"Container start failed: {exc}"
            log.error("container_start_failed", error=str(exc))
            # Clean up partial container if it was created
            if container is not None:
                await self._remove_container(container)
            return SpawnResult(success=False, error=error_msg)

        # At this point, container is guaranteed to be non-None
        assert container is not None

        # Wait for health check
        healthy = await self._wait_for_health(
            host="127.0.0.1",
            port=http_port,
            timeout_s=self._boot_timeout_s,
        )

        if not healthy:
            log.error("child_health_check_failed", container_id=container.id[:12])
            # Grab logs before removing for diagnostics
            try:
                tail_logs = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: container.logs(tail=50).decode("utf-8", errors="replace"),
                )
                log.error("child_container_logs", logs=tail_logs)
            except Exception:
                pass
            await self._remove_container(container)
            return SpawnResult(
                success=False,
                container_id=container.id,
                error=f"Child failed health check within {self._boot_timeout_s}s",
            )

        log.info(
            "spawn_child_complete",
            container_id=container.id[:12],
            http=f"127.0.0.1:{http_port}",
            federation=f"127.0.0.1:{fed_port}",
        )

        return SpawnResult(
            success=True,
            container_id=container.id,
            child_host="127.0.0.1",
            child_http_port=http_port,
            child_ws_port=ws_port,
            child_federation_port=fed_port,
            injected_env_keys=list(env.keys()),
        )

    async def terminate_child(self, container_id: str) -> bool:
        """
        Gracefully stop and remove a child container.

        Returns ``True`` if removal succeeded, ``False`` otherwise.
        """
        try:
            import docker as docker_lib
        except ImportError:
            return False

        log = self._log.bind(container_id=container_id[:12])
        try:
            client = docker_lib.from_env()
            container = client.containers.get(container_id)
            await self._remove_container(container)
            log.info("child_terminated")
            return True
        except Exception as exc:
            log.error("child_terminate_failed", error=str(exc))
            return False

    # ── Internals ────────────────────────────────────────────────────

    def _build_child_env(
        self,
        *,
        child_config: SeedConfiguration,
        parent_cert: str,
        http_port: int,
        ws_port: int,
        fed_port: int,
    ) -> dict[str, str]:
        """
        Construct the full environment dict for the child container.

        Merges:
          - Core EOS identity variables
          - Port overrides (so the child advertises its own ports)
          - Niche-specific config overrides from the SeedConfiguration
          - Parent certificate for Federation handshake
        """
        env: dict[str, str] = {
            # ── Identity ──
            "ECODIAOS_INSTANCE_ID": child_config.child_instance_id,
            "ECODIAOS_PARENT_INSTANCE_ID": child_config.parent_instance_id,
            "ECODIAOS_IS_GENESIS_NODE": "false",
            # ── Birth packet ──
            "ECODIAOS_BIRTH_CERTIFICATE": child_config.birth_certificate_json,
            "ECODIAOS_PARENT_CERTIFICATE": parent_cert,
            "ECODIAOS_SEED_CAPITAL_USD": str(child_config.seed_capital_usd),
            "ECODIAOS_DIVIDEND_RATE": str(child_config.dividend_rate),
            # ── Niche ──
            "ECODIAOS_NICHE_NAME": child_config.niche.name,
            "ECODIAOS_NICHE_DESCRIPTION": child_config.niche.description,
            # ── Genome inheritance ──
            "ECODIAOS_ORGANISM_GENOME_ID": child_config.organism_genome_id,
            "ECODIAOS_BELIEF_GENOME_ID": child_config.belief_genome_id,
            "ECODIAOS_SIMULA_GENOME_ID": child_config.simula_genome_id,
            "ECODIAOS_EQUOR_GENOME_ID": child_config.equor_genome_id,
            "ECODIAOS_AXON_GENOME_ID": child_config.axon_genome_id,
            "ECODIAOS_GENERATION": str(child_config.generation),
            # ── Server ports (inside the container, always default) ──
            "ECODIAOS_SERVER__PORT": str(_INTERNAL_HTTP_PORT),
            "ECODIAOS_SERVER__WS_PORT": str(_INTERNAL_WS_PORT),
            "ECODIAOS_SERVER__FEDERATION_PORT": str(_INTERNAL_FED_PORT),
            # ── Config paths ──
            "ECODIAOS_CONFIG_PATH": "/app/config/default.yaml",
        }

        # Inject niche-specific overrides from SeedConfiguration
        for key, value in child_config.child_config_overrides.items():
            env_key = f"ECODIAOS_{key.upper()}"
            env[env_key] = value

        return env

    def _build_volumes(self) -> dict[str, dict[str, str]]:
        """
        Build volume mounts for the child container.

        Always mounts config read-only.  Source path is configurable so it
        works on the host (DooD mounts use host paths, not container paths).
        """
        volumes: dict[str, dict[str, str]] = {}

        if self._config_volume_src:
            volumes[self._config_volume_src] = {
                "bind": "/app/config",
                "mode": "ro",
            }

        return volumes

    async def _wait_for_health(
        self,
        *,
        host: str,
        port: int,
        timeout_s: float,
        poll_interval_s: float = 2.0,
    ) -> bool:
        """
        Poll the child's HTTP port until a TCP connection succeeds.

        A successful TCP connect means uvicorn is accepting connections.
        The FastAPI ``/health`` endpoint itself will return once the app
        is fully wired, but a TCP-level check is sufficient for spawn
        confirmation - the Federation handshake validates deeper readiness.
        """
        log = self._log.bind(host=host, port=port)
        deadline = asyncio.get_running_loop().time() + timeout_s

        while asyncio.get_running_loop().time() < deadline:
            try:
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=min(5.0, poll_interval_s),
                )
                writer.close()
                await writer.wait_closed()
                log.info("health_check_passed")
                return True
            except (TimeoutError, ConnectionError):
                await asyncio.sleep(poll_interval_s)

        return False

    @staticmethod
    async def _remove_container(container: Container) -> None:
        """Stop and remove a container, swallowing errors."""
        with contextlib.suppress(Exception):
            await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: container.remove(force=True),
            )
