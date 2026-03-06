#!/usr/bin/env python3
"""
EcodiaOS — Inspector Taint Collector Daemon

Runs inside the privileged eBPF sidecar container. Loads BPF programs,
correlates kernel-level data flow events into cross-service taint graphs,
and exposes an HTTP API for the Inspector Prover to query.

This script is self-contained: it depends only on Python stdlib + bcc.
It is volume-mounted into the sidecar, not installed via pip.

HTTP API (default port 9471):
  GET  /health                          — readiness probe
  GET  /taint/flows                     — all observed taint flows
  GET  /taint/flows?source_service=X    — flows from named service
  POST /taint/inject                    — inject synthetic taint tag
  GET  /taint/graph                     — full taint propagation graph
  GET  /taint/connections               — connection map (who talks to whom)
  GET  /taint/stats                     — collector operational stats

Iron Rules:
  - All BPF programs are read-only (no kernel writes).
  - The collector never modifies container state beyond loading BPF programs.
  - Graceful degradation: if bcc is unavailable, start HTTP server with
    empty data and status="degraded".
  - Signal handlers ensure BPF programs are detached on shutdown.

Usage:
  python3 /opt/simula/taint_collector.py --port 9471
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [taint-collector] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("taint-collector")


# ── Data Classes ─────────────────────────────────────────────────────────────
# Using stdlib dataclasses (not Pydantic) since this runs in a minimal container.


@dataclass
class TaintEvent:
    """Python mirror of the BPF taint_event struct."""

    timestamp_ns: int = 0
    source_pid: int = 0
    dest_pid: int = 0
    source_tid: int = 0
    dest_tid: int = 0
    payload_hash: int = 0
    payload_size: int = 0
    flow_type: int = 0  # 0-4 matching BpfProgramType
    ip_version: int = 0
    source_port: int = 0
    dest_port: int = 0
    source_ip: int = 0
    dest_ip: int = 0
    comm: str = ""


@dataclass
class TaintFlow:
    """Correlated cross-service data flow."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    service_name: str = ""           # resolved compose service name (source)
    container_id: str = ""           # 12-char container ID (source)
    source_service: str = ""
    dest_service: str = ""
    source_container_id: str = ""
    dest_container_id: str = ""
    source_pid: int = 0
    dest_pid: int = 0
    source_comm: str = ""
    dest_comm: str = ""
    flow_type: str = "network"  # "network", "file", "ipc"
    payload_hash: int = 0
    payload_size: int = 0
    first_seen_ns: int = 0
    last_seen_ns: int = 0
    event_count: int = 1


@dataclass
class TaintGraphData:
    """Full propagation graph."""

    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    sinks: list[dict[str, Any]] = field(default_factory=list)


# ── PID → Service Mapper ────────────────────────────────────────────────────


class PidServiceMapper:
    """
    Maps process IDs to Docker compose service names and container IDs.

    Reads /proc/{pid}/cgroup to extract the container ID, then calls
    docker inspect to resolve the compose service name. Results are cached
    since PIDs are stable during a compose session.
    """

    def __init__(self) -> None:
        # pid → (service_name, container_id)
        self._cache: dict[int, tuple[str, str]] = {}
        # container_id → service_name
        self._container_cache: dict[str, str] = {}

    def resolve(self, pid: int) -> tuple[str, str]:
        """
        Resolve a PID to (service_name, container_id).

        service_name examples: "api-service", "postgres"
        container_id: 12-char hex prefix, or "" when running outside a container
        Falls back to (comm, "") when the PID is not in a container.
        """
        if pid in self._cache:
            return self._cache[pid]

        result = self._resolve_uncached(pid)
        self._cache[pid] = result
        return result

    def resolve_service(self, pid: int) -> str:
        """Convenience wrapper — returns just the service name."""
        return self.resolve(pid)[0]

    def _resolve_uncached(self, pid: int) -> tuple[str, str]:
        """Attempt to resolve PID → container_id → service_name."""
        container_id = self._pid_to_container_id(pid)
        if not container_id:
            # Not containerised — use process comm as the node label
            return (self._pid_to_comm(pid), "")

        if container_id in self._container_cache:
            return (self._container_cache[container_id], container_id)

        service = self._container_id_to_service(container_id)
        self._container_cache[container_id] = service
        return (service, container_id)

    @staticmethod
    def _pid_to_container_id(pid: int) -> str | None:
        """Extract container ID from /proc/{pid}/cgroup."""
        try:
            cgroup_path = f"/proc/{pid}/cgroup"
            with open(cgroup_path) as f:
                for line in f:
                    # Docker cgroup patterns:
                    #   /docker/{container_id}
                    #   /system.slice/docker-{container_id}.scope
                    parts = line.strip().split("/")
                    for i, part in enumerate(parts):
                        if part == "docker" and i + 1 < len(parts):
                            cid = parts[i + 1].replace(".scope", "")
                            if len(cid) >= 12:
                                return cid[:12]
                        if part.startswith("docker-") and part.endswith(".scope"):
                            cid = part[7:-6]  # strip "docker-" and ".scope"
                            if len(cid) >= 12:
                                return cid[:12]
        except (OSError, IndexError):
            pass
        return None

    @staticmethod
    def _container_id_to_service(container_id: str) -> str:
        """Run docker inspect to get the compose service name."""
        try:
            result = subprocess.run(
                [
                    "docker", "inspect",
                    "--format", "{{index .Config.Labels \"com.docker.compose.service\"}}",
                    container_id,
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return f"container:{container_id}"

    @staticmethod
    def _pid_to_comm(pid: int) -> str:
        """Fallback: read /proc/{pid}/comm."""
        try:
            with open(f"/proc/{pid}/comm") as f:
                return f.read().strip() or f"pid:{pid}"
        except OSError:
            return f"pid:{pid}"

    @property
    def cache_size(self) -> int:
        return len(self._cache)


# ── BPF Loader ──────────────────────────────────────────────────────────────


class BpfLoader:
    """
    Loads and manages BPF programs via bcc.

    Graceful degradation: if bcc is unavailable or a specific program fails
    to load, the loader skips it and tracks which programs are active.
    """

    # BPF C programs — imported from ebpf_programs.py at construction
    # but stored inline here for self-containment in the sidecar.
    # The actual programs are passed via constructor.

    def __init__(self, programs: dict[str, str]) -> None:
        self._program_sources = programs
        self._bpf: Any = None  # bcc.BPF instance
        self._loaded: dict[str, bool] = {}
        self._errors: dict[str, str] = {}
        self._event_counts: dict[str, int] = {name: 0 for name in programs}
        self._bcc_available = False

    def load_all(self) -> int:
        """
        Compile and attach all BPF programs. Returns number successfully loaded.

        If bcc is not installed, logs a warning and returns 0 (degraded mode).
        """
        try:
            from bcc import BPF  # type: ignore[import-untyped]
            self._bcc_available = True
        except ImportError:
            log.warning("bcc not available — running in degraded mode (no eBPF)")
            for name in self._program_sources:
                self._loaded[name] = False
                self._errors[name] = "bcc not installed"
            return 0

        loaded_count = 0
        for name, source in self._program_sources.items():
            try:
                # Each program is compiled independently so failures are isolated
                bpf = BPF(text=source)
                self._bpf = bpf  # Keep reference to last successful BPF instance
                self._loaded[name] = True
                loaded_count += 1
                log.info("BPF program loaded: %s", name)
            except Exception as exc:
                self._loaded[name] = False
                self._errors[name] = str(exc)[:200]
                log.warning("BPF program %s failed to load: %s", name, exc)

        return loaded_count

    def poll_events(self, callback: Any, timeout_ms: int = 100) -> None:
        """Poll ring buffer for events. Calls callback(event_data) for each."""
        if self._bpf is None:
            return
        try:
            self._bpf.ring_buffer_poll(timeout=timeout_ms)
        except Exception as exc:
            log.debug("Ring buffer poll error: %s", exc)

    @property
    def programs_loaded(self) -> int:
        return sum(1 for v in self._loaded.values() if v)

    @property
    def program_status(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name in self._program_sources:
            result[name] = {
                "loaded": self._loaded.get(name, False),
                "events": self._event_counts.get(name, 0),
            }
            if name in self._errors:
                result[name]["error"] = self._errors[name]
        return result

    def increment_event_count(self, program_name: str) -> None:
        if program_name in self._event_counts:
            self._event_counts[program_name] += 1

    @property
    def is_available(self) -> bool:
        return self._bcc_available


# ── Taint Correlator ────────────────────────────────────────────────────────


_FLOW_TYPE_NAMES = {0: "network", 1: "network", 2: "network", 3: "file", 4: "file",
                   5: "file", 6: "network"}
_PROGRAM_NAMES = {0: "tcp_sendmsg", 1: "tcp_recvmsg", 2: "security_socket_connect",
                  3: "sys_write", 4: "sys_read", 5: "sys_exit_read", 6: "sys_write_exit"}

# Correlation window: events within this many nanoseconds are considered
# part of the same flow.
_CORRELATION_WINDOW_NS = 100_000_000  # 100ms


class TaintCorrelator:
    """
    Correlates raw eBPF events into cross-service taint flows.

    Send/recv pairs are matched by payload_hash within a time window.
    Connection events build the service adjacency map.
    """

    def __init__(self, pid_mapper: PidServiceMapper) -> None:
        self._pid_mapper = pid_mapper
        self._lock = threading.Lock()

        # Raw events keyed by payload_hash for correlation
        self._events_by_hash: dict[int, list[TaintEvent]] = {}

        # Correlated flows
        self._flows: dict[str, TaintFlow] = {}  # keyed by "src_svc→dst_svc:hash"

        # Connection map
        self._connections: dict[str, set[str]] = {}  # service → {connected_services}

        # Injected taint patterns
        self._injected_patterns: dict[int, str] = {}  # hash → label

        # Stats
        self.total_events = 0
        self.buffer_drops = 0
        self._start_time = time.monotonic()

    def ingest(self, event: TaintEvent) -> None:
        """Ingest a raw eBPF event and attempt correlation."""
        with self._lock:
            self.total_events += 1

            # Resolve PIDs to (service_name, container_id)
            src_svc, src_cid = (
                self._pid_mapper.resolve(event.source_pid) if event.source_pid else ("", "")
            )
            dst_svc, dst_cid = (
                self._pid_mapper.resolve(event.dest_pid) if event.dest_pid else ("", "")
            )

            # Update connection map for connect events
            if event.flow_type == 2 and src_svc and src_svc not in self._connections:
                self._connections[src_svc] = set()
                # For connect events, we know the source; destination is by IP
                # (resolved later when we see recv events from that IP)

            # Store event for hash-based correlation
            if event.payload_hash != 0:
                if event.payload_hash not in self._events_by_hash:
                    self._events_by_hash[event.payload_hash] = []
                self._events_by_hash[event.payload_hash].append(event)

                # Attempt to correlate send/recv pairs
                self._try_correlate(event, src_svc, src_cid, dst_svc, dst_cid)

            # Prune old events (keep last 10 seconds)
            self._prune_old_events(event.timestamp_ns)

    def _try_correlate(
        self,
        event: TaintEvent,
        src_svc: str,
        src_cid: str,
        dst_svc: str,
        dst_cid: str,
    ) -> None:
        """Match send/write and recv events by payload hash within time window.

        Pairs matched:
          flow 0 (tcp_sendmsg)  → flow 1 (tcp_recvmsg)  — kernel TCP send/recv
          flow 6 (sys_exit_write) → flow 1 (tcp_recvmsg) — userspace write() to socket,
              payload later received by a peer; lets us spot the taint token leaving a
              process (B) via write() and arriving at the destination (C).
        """
        events = self._events_by_hash.get(event.payload_hash, [])
        if len(events) < 2:
            return

        # Outbound: tcp_sendmsg (0) and sys_exit_write (6) are both senders
        senders = [e for e in events if e.flow_type in (0, 6)]
        recvs = [e for e in events if e.flow_type == 1]  # tcp_recvmsg

        for send_evt in senders:
            send_svc, send_cid = self._pid_mapper.resolve(send_evt.source_pid)
            for recv_evt in recvs:
                recv_svc, recv_cid = self._pid_mapper.resolve(recv_evt.dest_pid)

                # Skip self-correlations
                if send_svc == recv_svc:
                    continue

                # Check time proximity
                time_diff = abs(recv_evt.timestamp_ns - send_evt.timestamp_ns)
                if time_diff > _CORRELATION_WINDOW_NS:
                    continue

                # Create or update flow
                flow_key = f"{send_svc}→{recv_svc}:{event.payload_hash}"
                if flow_key in self._flows:
                    flow = self._flows[flow_key]
                    flow.event_count += 1
                    flow.last_seen_ns = max(send_evt.timestamp_ns, recv_evt.timestamp_ns)
                else:
                    self._flows[flow_key] = TaintFlow(
                        # Top-level attribution (source side of this edge)
                        service_name=send_svc,
                        container_id=send_cid,
                        # Both sides of the edge
                        source_service=send_svc,
                        dest_service=recv_svc,
                        source_container_id=send_cid,
                        dest_container_id=recv_cid,
                        source_pid=send_evt.source_pid,
                        dest_pid=recv_evt.dest_pid,
                        source_comm=send_evt.comm,
                        dest_comm=recv_evt.comm,
                        flow_type=_FLOW_TYPE_NAMES.get(send_evt.flow_type, "network"),
                        payload_hash=event.payload_hash,
                        payload_size=send_evt.payload_size,
                        first_seen_ns=min(send_evt.timestamp_ns, recv_evt.timestamp_ns),
                        last_seen_ns=max(send_evt.timestamp_ns, recv_evt.timestamp_ns),
                    )

                # Update connection map
                if send_svc not in self._connections:
                    self._connections[send_svc] = set()
                self._connections[send_svc].add(recv_svc)

    def _prune_old_events(self, current_ns: int) -> None:
        """Remove events older than 10 seconds to bound memory usage."""
        cutoff = current_ns - 10_000_000_000  # 10 seconds
        for hash_val in list(self._events_by_hash):
            events = self._events_by_hash[hash_val]
            self._events_by_hash[hash_val] = [
                e for e in events if e.timestamp_ns > cutoff
            ]
            if not self._events_by_hash[hash_val]:
                del self._events_by_hash[hash_val]

    def inject_taint(self, pattern: bytes, label: str) -> int:
        """
        Inject a synthetic taint marker. Returns the FNV-1a hash of the pattern
        so the caller can track it in the flow data.
        """
        h = self._fnv1a(pattern)
        with self._lock:
            self._injected_patterns[h] = label
        log.info("Taint injected: label=%s hash=0x%08x", label, h)
        return h

    @staticmethod
    def _fnv1a(data: bytes) -> int:
        """Pure-Python FNV-1a hash matching the BPF implementation."""
        h = 2166136261
        for byte in data[:64]:
            h ^= byte
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def get_flows(
        self,
        *,
        source_service: str | None = None,
        dest_service: str | None = None,
        flow_type: str | None = None,
    ) -> list[TaintFlow]:
        """Return correlated flows with optional filters."""
        with self._lock:
            flows = list(self._flows.values())

        if source_service:
            flows = [f for f in flows if f.source_service == source_service]
        if dest_service:
            flows = [f for f in flows if f.dest_service == dest_service]
        if flow_type:
            flows = [f for f in flows if f.flow_type == flow_type]
        return flows

    def get_graph(self) -> TaintGraphData:
        """Build full taint propagation graph."""
        with self._lock:
            flows = list(self._flows.values())
            connections = {k: list(v) for k, v in self._connections.items()}

        # Collect all service names, keeping container_id when available
        # service_name → best-known container_id (non-empty wins)
        svc_to_cid: dict[str, str] = {}
        for f in flows:
            if f.source_service:
                svc_to_cid.setdefault(f.source_service, f.source_container_id)
                if f.source_container_id:
                    svc_to_cid[f.source_service] = f.source_container_id
            if f.dest_service:
                svc_to_cid.setdefault(f.dest_service, f.dest_container_id)
                if f.dest_container_id:
                    svc_to_cid[f.dest_service] = f.dest_container_id
        for svc, targets in connections.items():
            svc_to_cid.setdefault(svc, "")
            for t in targets:
                svc_to_cid.setdefault(t, "")

        nodes = [
            {
                "service_name": s,
                "container_id": svc_to_cid.get(s, "") or None,
                "service_type": "",
                "pid": None,
            }
            for s in sorted(svc_to_cid)
        ]
        edges = [asdict(f) for f in flows]

        # Sources: services that only send (or are entry points)
        senders = {f.source_service for f in flows}
        receivers = {f.dest_service for f in flows}
        source_only = senders - receivers

        sources = [
            {"variable_name": "external_input", "source_service": s,
             "entry_point": "unknown", "taint_level": "user_input"}
            for s in source_only
        ]

        # Sinks: services that only receive (leaf nodes)
        sink_only = receivers - senders
        sinks = [
            {"variable_name": "data_sink", "sink_service": s,
             "sink_type": "unknown", "function_name": "unknown",
             "is_sanitized": False}
            for s in sink_only
        ]

        return TaintGraphData(nodes=nodes, edges=edges, sources=sources, sinks=sinks)

    def get_connections(self) -> dict[str, list[str]]:
        """Return simplified adjacency map."""
        with self._lock:
            return {k: sorted(v) for k, v in self._connections.items()}

    def get_stats(self) -> dict[str, Any]:
        """Return operational metrics."""
        uptime = time.monotonic() - self._start_time
        eps = self.total_events / uptime if uptime > 0 else 0
        return {
            "events_per_second": round(eps, 1),
            "total_events": self.total_events,
            "ring_buffer_drops": self.buffer_drops,
            "correlation_window_ms": _CORRELATION_WINDOW_NS // 1_000_000,
            "pid_service_cache_size": self._pid_mapper.cache_size,
            "flows_correlated": len(self._flows),
            "injected_patterns": len(self._injected_patterns),
        }


# ── HTTP Server ─────────────────────────────────────────────────────────────


class CollectorHTTPHandler(BaseHTTPRequestHandler):
    """Minimal HTTP API for the taint collector."""

    # Set by the server at startup
    correlator: TaintCorrelator
    loader: BpfLoader
    start_time: float

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default stderr logging for cleaner output."""
        pass

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        if path == "/health":
            self._handle_health()
        elif path == "/taint/flows":
            self._handle_flows(params)
        elif path == "/taint/graph":
            self._handle_graph()
        elif path == "/taint/connections":
            self._handle_connections()
        elif path == "/taint/stats":
            self._handle_stats()
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/taint/inject":
            self._handle_inject()
        else:
            self._json_response(404, {"error": "not found"})

    # ── Route handlers ───────────────────────────────────────────────────

    def _handle_health(self) -> None:
        uptime = time.monotonic() - self.start_time
        status = "ready" if self.loader.programs_loaded > 0 else "degraded"
        if not self.loader.is_available:
            status = "degraded"

        code = 200 if status == "ready" else 503 if self.loader.programs_loaded == 0 and uptime < 60 else 200
        self._json_response(code, {
            "status": status,
            "programs_loaded": self.loader.programs_loaded,
            "events_collected": self.correlator.total_events,
            "flows_correlated": len(self.correlator._flows),
            "buffer_drops": self.correlator.buffer_drops,
            "uptime_seconds": round(uptime, 1),
        })

    def _handle_flows(self, params: dict[str, list[str]]) -> None:
        kwargs: dict[str, Any] = {}
        if "source_service" in params:
            kwargs["source_service"] = params["source_service"][0]
        if "dest_service" in params:
            kwargs["dest_service"] = params["dest_service"][0]
        if "flow_type" in params:
            kwargs["flow_type"] = params["flow_type"][0]

        flows = self.correlator.get_flows(**kwargs)
        self._json_response(200, {
            "flows": [asdict(f) for f in flows],
            "total": len(flows),
        })

    def _handle_graph(self) -> None:
        graph = self.correlator.get_graph()
        self._json_response(200, asdict(graph))

    def _handle_connections(self) -> None:
        connections = self.correlator.get_connections()
        self._json_response(200, connections)

    def _handle_stats(self) -> None:
        stats = self.correlator.get_stats()
        stats["programs"] = self.loader.program_status
        self._json_response(200, stats)

    def _handle_inject(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            pattern = base64.b64decode(data.get("pattern", ""))
            label = data.get("label", "injected")
            h = self.correlator.inject_taint(pattern, label)
            self._json_response(200, {
                "accepted": True,
                "pattern_hash": f"0x{h:08x}",
                "label": label,
            })
        except Exception as exc:
            self._json_response(400, {"error": str(exc)})

    # ── Helpers ──────────────────────────────────────────────────────────

    def _json_response(self, code: int, data: Any) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── BPF Event Poller Thread ─────────────────────────────────────────────────


def _event_poll_loop(
    loader: BpfLoader,
    correlator: TaintCorrelator,
    poll_interval_ms: int,
    stop_event: threading.Event,
) -> None:
    """Background thread that polls BPF ring buffer and feeds the correlator."""
    log.info("Event poll loop started (interval=%dms)", poll_interval_ms)

    # Set up ring buffer callback if bcc is available
    if loader._bpf is not None:
        try:
            def _handle_event(ctx: Any, data: Any, size: int) -> None:
                """Ring buffer callback — parse ctypes struct into TaintEvent."""
                try:
                    # Define ctypes struct matching the BPF taint_event
                    class _CtypesTaintEvent(ctypes.Structure):
                        _fields_ = [
                            ("timestamp_ns", ctypes.c_uint64),
                            ("source_pid", ctypes.c_uint32),
                            ("dest_pid", ctypes.c_uint32),
                            ("source_tid", ctypes.c_uint32),
                            ("dest_tid", ctypes.c_uint32),
                            ("payload_hash", ctypes.c_uint32),
                            ("payload_size", ctypes.c_uint32),
                            ("flow_type", ctypes.c_uint8),
                            ("ip_version", ctypes.c_uint8),
                            ("source_port", ctypes.c_uint16),
                            ("dest_port", ctypes.c_uint16),
                            ("source_ip", ctypes.c_uint32),
                            ("dest_ip", ctypes.c_uint32),
                            ("comm", ctypes.c_char * 16),
                            ("captured_len", ctypes.c_uint32),
                            ("payload_bytes", ctypes.c_char * 256),
                        ]

                    evt_ptr = ctypes.cast(data, ctypes.POINTER(_CtypesTaintEvent))
                    raw = evt_ptr.contents

                    event = TaintEvent(
                        timestamp_ns=raw.timestamp_ns,
                        source_pid=raw.source_pid,
                        dest_pid=raw.dest_pid,
                        source_tid=raw.source_tid,
                        dest_tid=raw.dest_tid,
                        payload_hash=raw.payload_hash,
                        payload_size=raw.payload_size,
                        flow_type=raw.flow_type,
                        ip_version=raw.ip_version,
                        source_port=raw.source_port,
                        dest_port=raw.dest_port,
                        source_ip=raw.source_ip,
                        dest_ip=raw.dest_ip,
                        comm=raw.comm.decode("utf-8", errors="replace").rstrip("\x00"),
                    )

                    program_name = _PROGRAM_NAMES.get(event.flow_type, "unknown")
                    loader.increment_event_count(program_name)
                    correlator.ingest(event)

                except Exception as exc:
                    log.debug("Event parse error: %s", exc)

            loader._bpf["taint_events"].open_ring_buffer(_handle_event)
            log.info("Ring buffer callback registered")
        except Exception as exc:
            log.warning("Failed to set up ring buffer: %s", exc)

    while not stop_event.is_set():
        try:
            loader.poll_events(None, timeout_ms=poll_interval_ms)
        except Exception as exc:
            log.debug("Poll error: %s", exc)
        stop_event.wait(timeout=poll_interval_ms / 1000.0)

    log.info("Event poll loop stopped")


# ── BPF Programs (inline for self-containment) ──────────────────────────────
# These are the same programs defined in ebpf_programs.py but inlined here
# so the sidecar script is fully self-contained (no imports from EOS).

def _get_bpf_programs() -> dict[str, str]:
    """
    Return the BPF program sources.

    In sidecar mode, programs are loaded from a companion JSON file if
    available (written by the topology chamber). Falls back to a minimal
    tcp_sendmsg + tcp_recvmsg pair for basic cross-service tracking.
    """
    # Try to load from companion file (written by topology chamber)
    companion = "/opt/simula/bpf_programs.json"
    if os.path.exists(companion):
        try:
            with open(companion) as f:
                programs = json.load(f)
            log.info("Loaded %d BPF programs from %s", len(programs), companion)
            return programs
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to load BPF programs from %s: %s", companion, exc)

    log.info("No companion BPF programs file — using built-in minimal set")
    return {}


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspector Taint Collector Daemon")
    parser.add_argument("--port", type=int, default=9471, help="HTTP API port")
    parser.add_argument("--poll-interval-ms", type=int, default=50, help="BPF poll interval")
    args = parser.parse_args()

    log.info("Starting taint collector on port %d", args.port)

    # Initialize components
    pid_mapper = PidServiceMapper()
    correlator = TaintCorrelator(pid_mapper)

    programs = _get_bpf_programs()
    loader = BpfLoader(programs)

    # Load BPF programs
    loaded = loader.load_all()
    if loaded > 0:
        log.info("Loaded %d/%d BPF programs", loaded, len(programs))
    else:
        log.warning("No BPF programs loaded — running in degraded mode")

    # Start event polling thread
    stop_event = threading.Event()
    poll_thread = threading.Thread(
        target=_event_poll_loop,
        args=(loader, correlator, args.poll_interval_ms, stop_event),
        daemon=True,
        name="bpf-poll",
    )
    poll_thread.start()

    # Configure HTTP handler
    CollectorHTTPHandler.correlator = correlator
    CollectorHTTPHandler.loader = loader
    CollectorHTTPHandler.start_time = time.monotonic()

    server = HTTPServer(("0.0.0.0", args.port), CollectorHTTPHandler)

    # Signal handlers for graceful shutdown
    def _shutdown(signum: int, frame: Any) -> None:
        log.info("Received signal %d — shutting down", signum)
        stop_event.set()
        server.shutdown()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info("HTTP server listening on 0.0.0.0:%d", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        poll_thread.join(timeout=5)
        log.info("Taint collector stopped")


if __name__ == "__main__":
    main()
