"""
EcodiaOS - Inspector Taint Collector Client

Async HTTP client for querying the eBPF taint collector sidecar API running
inside the docker-compose topology. All methods catch connection/timeout errors
and return empty defaults with structlog warnings - the taint client never
blocks the pipeline.

Sidecar API endpoints:
  GET  /health                - readiness probe
  GET  /taint/flows           - filtered cross-service flows
  GET  /taint/graph           - full taint propagation graph
  GET  /taint/connections     - simplified adjacency map
  GET  /taint/stats           - event/buffer statistics
  POST /taint/inject          - inject synthetic taint marker
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import httpx
import structlog

from systems.simula.inspector.taint_types import (
    FlowType,
    TaintCollectorStatus,
    TaintFlow,
    TaintGraph,
    TaintGraphNode,
    TaintSource,
)

logger = structlog.get_logger().bind(system="simula.inspector.taint_client")


class TaintCollectorClient:
    """
    Async HTTP client for the eBPF taint collector sidecar.

    Never raises - all methods catch connection/timeout errors and return
    safe defaults. The caller can always proceed with degraded (or empty)
    taint data without special error handling.

    Usage::

        client = TaintCollectorClient("http://127.0.0.1:9471")
        ready = await client.wait_for_ready(timeout_s=30.0)
        if ready:
            graph = await client.get_taint_graph()
            flows = await client.get_flows(source_service="api")
        await client.close()
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 10.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = httpx.Timeout(timeout_s, connect=5.0)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        )
        self._log = logger.bind(collector_url=self._base_url)

    # ── Readiness ──────────────────────────────────────────────────────────

    async def wait_for_ready(self, timeout_s: float = 30.0) -> bool:
        """
        Poll the /health endpoint until the collector reports ready.

        Returns True if the collector became ready within the timeout,
        False otherwise (degraded mode or unreachable).
        """
        deadline = asyncio.get_running_loop().time() + timeout_s
        poll_interval = 1.0

        while asyncio.get_running_loop().time() < deadline:
            try:
                resp = await self._client.get("/health")
                if resp.status_code == 200:
                    data = resp.json()
                    status = data.get("status", "unknown")
                    if status in ("ready", "degraded"):
                        self._log.info(
                            "taint_collector_ready",
                            status=status,
                            programs_loaded=data.get("programs_loaded", 0),
                        )
                        return True
            except (httpx.HTTPError, Exception):
                pass

            await asyncio.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 5.0)

        self._log.warning("taint_collector_not_ready", timeout_s=timeout_s)
        return False

    # ── Status ─────────────────────────────────────────────────────────────

    async def get_status(self) -> TaintCollectorStatus:
        """Query /health for full collector status."""
        try:
            resp = await self._client.get("/health")
            resp.raise_for_status()
            data = resp.json()
            return TaintCollectorStatus(
                status=data.get("status", "unknown"),
                programs_loaded=data.get("programs_loaded", 0),
                events_collected=data.get("events_collected", 0),
                flows_correlated=data.get("flows_correlated", 0),
                buffer_drops=data.get("buffer_drops", 0),
                uptime_seconds=data.get("uptime_seconds", 0.0),
                degraded_reason=data.get("degraded_reason"),
            )
        except Exception as exc:
            self._log.warning("taint_client_status_error", error=str(exc))
            return TaintCollectorStatus(status="error", degraded_reason=str(exc))

    # ── Flows ──────────────────────────────────────────────────────────────

    async def get_flows(
        self,
        *,
        source_service: str | None = None,
        dest_service: str | None = None,
        flow_type: str | None = None,
    ) -> list[TaintFlow]:
        """
        Query /taint/flows with optional filters.

        Returns an empty list on any error.
        """
        try:
            params: dict[str, str] = {}
            if source_service:
                params["source_service"] = source_service
            if dest_service:
                params["dest_service"] = dest_service
            if flow_type:
                params["flow_type"] = flow_type

            resp = await self._client.get("/taint/flows", params=params)
            resp.raise_for_status()
            data = resp.json()

            flows: list[TaintFlow] = []
            for raw in data.get("flows", []):
                flows.append(
                    TaintFlow(
                        from_service=raw.get("from_service", "unknown"),
                        to_service=raw.get("to_service", "unknown"),
                        flow_type=_parse_flow_type(raw.get("flow_type", "network")),
                        payload_signature=raw.get("payload_signature", ""),
                        payload_size=raw.get("payload_size", 0),
                        event_count=raw.get("event_count", 1),
                    )
                )
            return flows
        except Exception as exc:
            self._log.warning("taint_client_flows_error", error=str(exc))
            return []

    # ── Full Graph ─────────────────────────────────────────────────────────

    async def get_taint_graph(self) -> TaintGraph:
        """
        Query /taint/graph for the full taint propagation graph.

        Returns an empty TaintGraph on any error.
        """
        try:
            resp = await self._client.get("/taint/graph")
            resp.raise_for_status()
            data = resp.json()

            nodes = [
                TaintGraphNode(
                    service_name=n.get("service_name", "unknown"),
                    service_type=n.get("service_type", ""),
                    pid=n.get("pid"),
                    container_id=n.get("container_id"),
                )
                for n in data.get("nodes", [])
            ]

            edges = [
                TaintFlow(
                    from_service=e.get("from_service", "unknown"),
                    to_service=e.get("to_service", "unknown"),
                    flow_type=_parse_flow_type(e.get("flow_type", "network")),
                    payload_signature=e.get("payload_signature", ""),
                    payload_size=e.get("payload_size", 0),
                    event_count=e.get("event_count", 1),
                )
                for e in data.get("edges", [])
            ]

            sources = [
                TaintSource(
                    variable_name=s.get("variable_name", ""),
                    source_service=s.get("source_service", "unknown"),
                    entry_point=s.get("entry_point", ""),
                )
                for s in data.get("sources", [])
            ]

            return TaintGraph(
                nodes=nodes,
                edges=edges,
                sources=sources,
            )
        except Exception as exc:
            self._log.warning("taint_client_graph_error", error=str(exc))
            return TaintGraph()

    # ── Connections ────────────────────────────────────────────────────────

    async def get_connections(self) -> dict[str, list[str]]:
        """
        Query /taint/connections for the simplified adjacency map.

        Returns ``{service_name: [connected_service_names, ...], ...}``.
        Empty dict on error.
        """
        try:
            resp = await self._client.get("/taint/connections")
            resp.raise_for_status()
            data = resp.json()
            connections: dict[str, list[str]] = {}
            for svc, targets in data.get("connections", {}).items():
                if isinstance(targets, list):
                    connections[svc] = targets
            return connections
        except Exception as exc:
            self._log.warning("taint_client_connections_error", error=str(exc))
            return {}

    # ── Taint Injection ────────────────────────────────────────────────────

    async def inject_taint(self, pattern: bytes, label: str) -> bool:
        """
        POST /taint/inject to plant a synthetic taint marker.

        The collector will track this pattern through the topology so the
        prover can verify specific data-flow paths.

        Returns True on success, False on any error.
        """
        try:
            import base64

            resp = await self._client.post(
                "/taint/inject",
                json={
                    "pattern": base64.b64encode(pattern).decode("ascii"),
                    "label": label,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            success = data.get("injected", False)
            if success:
                self._log.debug("taint_injected", label=label, pattern_size=len(pattern))
            return success
        except Exception as exc:
            self._log.warning("taint_client_inject_error", error=str(exc), label=label)
            return False

    # ── Statistics ─────────────────────────────────────────────────────────

    async def get_stats(self) -> dict[str, Any]:
        """
        Query /taint/stats for collector runtime statistics.

        Returns raw JSON dict on success, empty dict on error.
        """
        try:
            resp = await self._client.get("/taint/stats")
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            self._log.warning("taint_client_stats_error", error=str(exc))
            return {}

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying HTTP client. Idempotent."""
        with contextlib.suppress(Exception):
            await self._client.aclose()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_flow_type(raw: str) -> FlowType:
    """Parse a flow type string, defaulting to NETWORK on unknown values."""
    try:
        return FlowType(raw)
    except ValueError:
        return FlowType.NETWORK
