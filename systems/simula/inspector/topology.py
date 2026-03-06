"""
EcodiaOS — Inspector Topology Detonation Chamber

Upgrades the single-container LiveDetonationChamber to support distributed
microservice targets defined via docker-compose. Parses the workspace's
compose file, dynamically injects a privileged eBPF sidecar for cross-boundary
taint tracking, orchestrates the full cluster, and discovers the entry-point
service's mapped port.

Iron Rules:
  - The entire cluster is ALWAYS torn down on exit (no zombie services).
  - The mutated compose file is written to a temporary location — the original
    workspace compose file is never modified.
  - If no compose file is found, spin_up_cluster returns None so the caller
    can fall back to the single-container LiveDetonationChamber.
  - The eBPF sidecar runs privileged with host PID/network for full
    cross-service taint visibility.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import yaml

from primitives.common import EOSBaseModel
from systems.simula.inspector.ebpf_programs import TAINT_COLLECTOR_PORT

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from systems.simula.inspector.taint_client import TaintCollectorClient
    from systems.simula.inspector.taint_types import TaintGraph

logger = structlog.get_logger().bind(system="simula.inspector.topology")

# Compose filenames to scan, in priority order.
_COMPOSE_CANDIDATES = ("docker-compose.yml", "docker-compose.yaml")

# Service names commonly used as the public entry point (checked in order).
_ENTRY_SERVICE_HEURISTICS = (
    "frontend",
    "web",
    "api",
    "gateway",
    "app",
    "nginx",
    "proxy",
    "server",
)

# ── eBPF Sidecar Builder ─────────────────────────────────────────────────────
# Injected into every topology so taint data can flow across service boundaries.


def _build_ebpf_sidecar_service(collector_script_host_path: str) -> dict[str, Any]:
    """
    Build the compose service definition for the eBPF taint collector sidecar.

    The sidecar:
      - Uses ubuntu:24.04 with python3-bpfcc pre-installed (via entrypoint)
      - Volume-mounts the taint_collector.py script + BPF kernel paths
      - Runs privileged with host PID/network for full taint visibility
      - Exposes a health endpoint on port 9471
    """
    return {
        "image": "ubuntu:24.04",
        "entrypoint": "/bin/bash",
        "command": [
            "-c",
            (
                "apt-get update -qq && "
                "apt-get install -y -qq python3 python3-bpfcc > /dev/null 2>&1 && "
                f"python3 /opt/simula/taint_collector.py --port {TAINT_COLLECTOR_PORT}"
            ),
        ],
        "privileged": True,
        "pid": "host",
        "network_mode": "host",
        "cap_add": ["SYS_ADMIN", "BPF", "PERFMON", "NET_ADMIN"],
        "volumes": [
            f"{collector_script_host_path}:/opt/simula/taint_collector.py:ro",
            "/sys/kernel/debug:/sys/kernel/debug:ro",
            "/sys/fs/bpf:/sys/fs/bpf:ro",
            "/var/run/docker.sock:/var/run/docker.sock:ro",
        ],
        "restart": "no",
        "healthcheck": {
            "test": [
                "CMD", "python3", "-c",
                (
                    "import urllib.request; "
                    f"urllib.request.urlopen('http://127.0.0.1:{TAINT_COLLECTOR_PORT}/health')"
                ),
            ],
            "interval": "5s",
            "timeout": "3s",
            "retries": 10,
            "start_period": "30s",
        },
    }


# Fallback placeholder when the collector script can't be staged.
_EBPF_SIDECAR_FALLBACK: dict[str, Any] = {
    "image": "ubuntu:latest",
    "command": "tail -f /dev/null",
    "privileged": True,
    "pid": "host",
    "network_mode": "host",
    "cap_add": ["SYS_ADMIN", "BPF", "PERFMON"],
    "restart": "no",
}


# ── Models ───────────────────────────────────────────────────────────────────


class TopologyContext(EOSBaseModel):
    """Runtime context for a topology-detonated cluster."""

    entry_url: str
    """Publicly exposed HTTP endpoint (e.g. ``http://127.0.0.1:8080``)."""

    internal_services: dict[str, str]
    """Map of service names to their internal Docker network IPs/ports."""

    compose_file_path: Path
    """Path to the mutated compose file used to boot the cluster."""

    taint_collector_url: str | None = None
    """URL of the eBPF taint collector sidecar (e.g. ``http://127.0.0.1:9471``).
    None if the collector failed to start or was not injected."""

    taint_collector_status: str = "unknown"
    """Collector readiness: 'ready', 'degraded', 'loading', 'error', or 'unknown'."""


# ── Topology Detonation Chamber ──────────────────────────────────────────────


class TopologyDetonationChamber:
    """
    Spins up a full docker-compose cluster with an injected eBPF sidecar.

    Usage::

        chamber = TopologyDetonationChamber()
        async with chamber.spin_up_cluster(workspace.root) as topology:
            if topology is not None:
                # topology.entry_url → "http://127.0.0.1:32771"
                # topology.internal_services → {"api": "172.18.0.3:8000", ...}
                ...
        # Cluster is guaranteed torn down here.

    If the workspace has no docker-compose file, yields ``None`` so the caller
    can fall back to the single-container :class:`LiveDetonationChamber`.
    """

    def __init__(
        self,
        *,
        compose_up_timeout_s: int = 180,
        discovery_timeout_s: int = 30,
        taint_collector_ready_timeout_s: float = 45.0,
    ) -> None:
        self._compose_up_timeout_s = compose_up_timeout_s
        self._discovery_timeout_s = discovery_timeout_s
        self._taint_collector_ready_timeout_s = taint_collector_ready_timeout_s
        self._mutated_compose_path: Path | None = None
        self._workspace_root: Path | None = None
        self._taint_client: TaintCollectorClient | None = None
        self._log = logger

    # ── Public API ─────────────────────────────────────────────────────────

    @asynccontextmanager
    async def spin_up_cluster(
        self,
        workspace_root: Path,
    ) -> AsyncIterator[TopologyContext | None]:
        """
        Parse compose file, inject eBPF sidecar, boot cluster, discover ports.

        Yields:
            A :class:`TopologyContext` if a compose file was found and the
            cluster booted successfully, or ``None`` if no compose file exists.

        On exit the cluster is torn down via ``docker compose down -v``.
        """
        self._workspace_root = workspace_root
        log = self._log.bind(workspace=str(workspace_root))

        # Step 1: Locate compose file
        compose_path = self._find_compose_file(workspace_root)
        if compose_path is None:
            log.debug(
                "no_compose_file_found",
                scanned=list(_COMPOSE_CANDIDATES),
            )
            yield None
            return

        log = log.bind(compose_file=str(compose_path))

        try:
            # Step 2: Stage taint collector script into workspace
            collector_host_path = self._stage_collector_script(workspace_root, log)

            # Step 3: Parse + inject sidecar
            mutated = self._inject_sidecar(compose_path, collector_host_path, log)

            # Step 4: Write mutated compose to temp location
            self._mutated_compose_path = workspace_root / "docker-compose.simula.yml"
            self._mutated_compose_path.write_text(
                yaml.safe_dump(mutated, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
            log.info(
                "topology_injected",
                sidecar="simula-ebpf-tracer",
                mutated_path=str(self._mutated_compose_path),
                total_services=len(mutated.get("services", {})),
            )

            # Step 5: Boot cluster
            log.info("cluster_booting")
            await self._compose_up(log)

            # Step 6: Discover entry-point ports
            topology = await self._discover_topology(mutated, log)

            # Step 7: Wait for taint collector readiness
            await self._wait_for_taint_collector(topology, log)

            yield topology

        except Exception as exc:
            log.error(
                "topology_cluster_error",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            yield None
        finally:
            await self.teardown()

    async def collect_taint_events(self) -> TaintGraph | None:
        """
        Collect the current taint graph from the sidecar.

        Returns None if the taint client is unavailable.
        """
        if self._taint_client is None:
            return None

        try:
            return await self._taint_client.get_taint_graph()
        except Exception as exc:
            self._log.warning("taint_collection_failed", error=str(exc))
            return None

    async def inject_taint(self, pattern: bytes, label: str) -> bool:
        """Inject a synthetic taint marker through the sidecar."""
        if self._taint_client is None:
            return False
        return await self._taint_client.inject_taint(pattern, label)

    async def teardown(self) -> None:
        """
        Tear down the cluster. Idempotent — safe to call multiple times.

        Closes the taint client, runs ``docker compose down -v --remove-orphans``
        against the mutated compose file, then removes the temp file.
        """
        # Close taint client first
        if self._taint_client is not None:
            await self._taint_client.close()
            self._taint_client = None

        if self._mutated_compose_path is None:
            return

        compose_path = self._mutated_compose_path
        self._mutated_compose_path = None  # Prevent double teardown

        log = self._log.bind(compose_file=str(compose_path))
        log.info("cluster_teardown")

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "compose",
                "-f", str(compose_path),
                "down", "-v", "--remove-orphans",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._workspace_root) if self._workspace_root else None,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=60,
            )

            if proc.returncode == 0:
                log.info("cluster_teardown_complete")
            else:
                log.warning(
                    "cluster_teardown_nonzero_exit",
                    returncode=proc.returncode,
                    stderr=stderr.decode(errors="replace")[:500],
                )
        except TimeoutError:
            log.warning("cluster_teardown_timeout")
        except Exception as exc:
            log.warning("cluster_teardown_error", error=str(exc))

        # Clean up the mutated compose file and staged collector script
        for cleanup_path in (
            compose_path,
            self._workspace_root / ".simula-taint-collector.py" if self._workspace_root else None,
        ):
            if cleanup_path is not None:
                with suppress(OSError):
                    cleanup_path.unlink(missing_ok=True)

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _find_compose_file(workspace_root: Path) -> Path | None:
        """Scan for docker-compose.yml or .yaml in the workspace root."""
        for candidate in _COMPOSE_CANDIDATES:
            path = workspace_root / candidate
            if path.is_file():
                return path
        return None

    @staticmethod
    def _stage_collector_script(workspace_root: Path, log: Any) -> str | None:
        """
        Copy the taint_collector.py script into the workspace for volume-mounting.

        Returns the host path to the staged script, or None if staging fails.
        """
        try:
            collector_src = Path(__file__).parent / "taint_collector.py"
            if not collector_src.is_file():
                log.warning(
                    "taint_collector_script_not_found",
                    expected_path=str(collector_src),
                )
                return None

            staged_path = workspace_root / ".simula-taint-collector.py"
            shutil.copy2(collector_src, staged_path)
            log.debug(
                "taint_collector_staged",
                source=str(collector_src),
                staged=str(staged_path),
            )
            return str(staged_path.resolve())
        except Exception as exc:
            log.warning("taint_collector_stage_failed", error=str(exc))
            return None

    @staticmethod
    def _inject_sidecar(
        compose_path: Path,
        collector_host_path: str | None,
        log: Any,
    ) -> dict[str, Any]:
        """
        Parse the compose file and inject the eBPF tracer sidecar.

        If a collector script was staged, injects the full taint collector
        sidecar. Otherwise falls back to a placeholder.

        Returns the mutated compose dict (original file is never touched).
        """
        raw = compose_path.read_text(encoding="utf-8")
        data: dict[str, Any] = yaml.safe_load(raw) or {}

        if "services" not in data:
            data["services"] = {}

        original_service_count = len(data["services"])

        if collector_host_path:
            sidecar = _build_ebpf_sidecar_service(collector_host_path)
        else:
            sidecar = dict(_EBPF_SIDECAR_FALLBACK)

        data["services"]["simula-ebpf-tracer"] = sidecar

        log.debug(
            "sidecar_injected",
            original_services=original_service_count,
            injected_service="simula-ebpf-tracer",
            has_collector=collector_host_path is not None,
        )

        return data

    async def _wait_for_taint_collector(
        self,
        topology: TopologyContext,
        log: Any,
    ) -> None:
        """
        Create a TaintCollectorClient and wait for the sidecar to become ready.

        Updates topology.taint_collector_url and taint_collector_status in place.
        On failure, logs a warning and leaves the fields at their defaults —
        the pipeline proceeds without taint data.
        """
        from systems.simula.inspector.taint_client import TaintCollectorClient

        collector_url = f"http://127.0.0.1:{TAINT_COLLECTOR_PORT}"
        client = TaintCollectorClient(collector_url)

        try:
            ready = await client.wait_for_ready(
                timeout_s=self._taint_collector_ready_timeout_s,
            )
            if ready:
                status = await client.get_status()
                topology.taint_collector_url = collector_url
                topology.taint_collector_status = status.status
                self._taint_client = client
                log.info(
                    "taint_collector_connected",
                    status=status.status,
                    programs_loaded=status.programs_loaded,
                )
            else:
                topology.taint_collector_status = "timeout"
                await client.close()
                log.warning("taint_collector_not_ready_fallback")
        except Exception as exc:
            topology.taint_collector_status = "error"
            await client.close()
            log.warning("taint_collector_connect_error", error=str(exc))

    async def _compose_up(self, log: Any) -> None:
        """Run ``docker compose up -d --build`` and await completion."""
        if self._mutated_compose_path is None:
            msg = "No mutated compose file — call spin_up_cluster first"
            raise RuntimeError(msg)

        proc = await asyncio.create_subprocess_exec(
            "docker", "compose",
            "-f", str(self._mutated_compose_path),
            "up", "-d", "--build",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._workspace_root) if self._workspace_root else None,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=self._compose_up_timeout_s,
        )

        if proc.returncode != 0:
            err_text = stderr.decode(errors="replace")[:1000]
            log.error(
                "compose_up_failed",
                returncode=proc.returncode,
                stderr=err_text,
            )
            msg = f"docker compose up failed (exit {proc.returncode}): {err_text}"
            raise RuntimeError(msg)

        log.info("cluster_up", stdout=stdout.decode(errors="replace")[:500])

    async def _discover_topology(
        self,
        compose_data: dict[str, Any],
        log: Any,
    ) -> TopologyContext:
        """
        Run ``docker compose ps --format json`` to extract dynamically
        assigned host ports and build the :class:`TopologyContext`.
        """
        if self._mutated_compose_path is None:
            msg = "No mutated compose file"
            raise RuntimeError(msg)

        proc = await asyncio.create_subprocess_exec(
            "docker", "compose",
            "-f", str(self._mutated_compose_path),
            "ps", "--format", "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._workspace_root) if self._workspace_root else None,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=self._discovery_timeout_s,
        )

        if proc.returncode != 0:
            err_text = stderr.decode(errors="replace")[:500]
            log.error("compose_ps_failed", returncode=proc.returncode, stderr=err_text)
            msg = f"docker compose ps failed (exit {proc.returncode})"
            raise RuntimeError(msg)

        # docker compose ps --format json emits one JSON object per line
        raw_output = stdout.decode(errors="replace").strip()
        containers: list[dict[str, Any]] = []
        for line in raw_output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                # docker compose v2 may emit a list or individual objects
                if isinstance(parsed, list):
                    containers.extend(parsed)
                else:
                    containers.append(parsed)
            except json.JSONDecodeError:
                log.debug("compose_ps_json_parse_skip", line=line[:200])

        # Build internal_services map and find entry-point
        internal_services: dict[str, str] = {}
        entry_url: str | None = None
        entry_service_name: str | None = None

        list(compose_data.get("services", {}).keys())

        for container in containers:
            service = container.get("Service", container.get("Name", ""))
            # Skip the injected sidecar
            if service == "simula-ebpf-tracer":
                continue

            # Extract published ports
            publishers = container.get("Publishers", [])
            if publishers:
                for pub in publishers:
                    published_port = pub.get("PublishedPort", 0)
                    target_port = pub.get("TargetPort", 0)
                    if published_port and target_port:
                        internal_services[service] = f"127.0.0.1:{published_port}"
                        break
                else:
                    # No valid publisher found, use container IP if available
                    internal_services[service] = service

        # Identify the entry-point service via heuristic matching
        for heuristic in _ENTRY_SERVICE_HEURISTICS:
            for svc_name, addr in internal_services.items():
                if heuristic in svc_name.lower():
                    entry_url = f"http://{addr}"
                    entry_service_name = svc_name
                    break
            if entry_url:
                break

        # Fallback: use the first service with a published port
        if entry_url is None and internal_services:
            first_service = next(iter(internal_services))
            entry_url = f"http://{internal_services[first_service]}"
            entry_service_name = first_service

        if entry_url is None:
            log.warning("no_entry_service_discovered", containers=len(containers))
            msg = "Could not discover any service with published ports"
            raise RuntimeError(msg)

        log.info(
            "topology_discovered",
            entry_service=entry_service_name,
            entry_url=entry_url,
            services=list(internal_services.keys()),
            total_containers=len(containers),
        )

        return TopologyContext(
            entry_url=entry_url,
            internal_services=internal_services,
            compose_file_path=self._mutated_compose_path
            or Path("docker-compose.simula.yml"),
        )
