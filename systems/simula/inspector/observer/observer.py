#!/usr/bin/env python3
"""
Simula Observer — Standalone eBPF Network Event Printer

Attaches BPF kprobes to tcp_sendmsg, tcp_recvmsg, and
security_socket_connect, polls the ring buffer, and prints
PID/comm/flow events continuously to stdout.

Self-contained: depends only on Python stdlib + bcc.
Volume-mounted into the observer sidecar, not installed via pip.

Optionally serves a minimal /health HTTP endpoint for liveness checks.

Usage:
    python3 observer.py --port 9472 --probes tcp_sendmsg,tcp_recvmsg,security_socket_connect

Iron Rules:
    - All BPF programs are read-only (no kernel writes).
    - Graceful degradation: if bcc is unavailable, start HTTP with status=degraded.
    - Signal handlers ensure BPF programs are detached on shutdown.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import logging
import os
import signal
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import urlparse

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [observer] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("observer")


# ── BPF Program Sources (inlined for self-containment) ───────────────────────
# Identical to ebpf_programs.py but embedded here so the sidecar script
# has zero imports from EOS. Only the 3 network probes are included.

RING_BUFFER_PAGE_CNT = 64

_TAINT_EVENT_STRUCT = """
struct taint_event {
    __u64 timestamp_ns;
    __u32 source_pid;
    __u32 dest_pid;
    __u32 source_tid;
    __u32 dest_tid;
    __u32 payload_hash;
    __u32 payload_size;
    __u8  flow_type;       // 0=tcp_send, 1=tcp_recv, 2=connect, 3=file_write, 4=file_read, 5=read_exit, 6=write_exit
    __u8  ip_version;      // 4 or 6
    __u16 source_port;
    __u16 dest_port;
    __u32 source_ip;       // IPv4 only (network byte order)
    __u32 dest_ip;
    char  comm[16];
    __u32 captured_len;    // actual bytes stored in payload_bytes (0..512)
    char  payload_bytes[512];
};
"""

_FNV1A_HASH_C = """
static __always_inline __u32 fnv1a_hash(const void *data, int len) {
    __u32 hash = 2166136261u;
    const unsigned char *p = (const unsigned char *)data;

    #pragma unroll
    for (int i = 0; i < 64 && i < len; i++) {
        unsigned char byte;
        if (bpf_probe_read_kernel(&byte, 1, p + i) != 0)
            break;
        hash ^= byte;
        hash *= 16777619u;
    }
    return hash;
}
"""

_COMMON_PREAMBLE = (
    """
#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <linux/socket.h>
#include <linux/in.h>

"""
    + _TAINT_EVENT_STRUCT
    + """

BPF_RINGBUF_OUTPUT(taint_events, """
    + str(RING_BUFFER_PAGE_CNT)
    + """);

// Port-filter map: populated from Python with ports we care about.
// If a destination port is NOT in this map the probe returns 0 immediately,
// avoiding the cost of struct allocation and ring-buffer submission for
// unmonitored ports.  An empty map means ALL ports are monitored (open mode).
BPF_HASH(monitored_ports, u16, u8);

"""
    + _FNV1A_HASH_C
)


BPF_TCP_SENDMSG = _COMMON_PREAMBLE + """
int kprobe__tcp_sendmsg(struct pt_regs *ctx, struct sock *sk,
                        struct msghdr *msg, size_t size) {
    // ── Port filter ──────────────────────────────────────────────────────
    // Read destination port early so we can gate before allocating an event.
    // Python writes a sentinel entry (key=0) to signal filter-mode is active.
    // If the sentinel is present and this port is not in the map, skip it.
    __u16 dport_raw = 0;
    bpf_probe_read_kernel(&dport_raw, sizeof(dport_raw),
                          &sk->__sk_common.skc_dport);
    __u16 dport = __builtin_bswap16(dport_raw);
    u8 *port_hit = monitored_ports.lookup(&dport);
    {
        u16 sentinel = 0;
        u8 *has_filter = monitored_ports.lookup(&sentinel);
        if (has_filter && !port_hit) {
            return 0;
        }
    }

    struct taint_event *evt = taint_events.ringbuf_reserve(sizeof(struct taint_event));
    if (!evt)
        return 0;

    __builtin_memset(evt, 0, sizeof(*evt));

    evt->timestamp_ns = bpf_ktime_get_ns();
    evt->source_pid   = bpf_get_current_pid_tgid() >> 32;
    evt->source_tid   = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    evt->flow_type    = 0;
    evt->payload_size = (size > 0xFFFFFFFF) ? 0xFFFFFFFF : (__u32)size;
    evt->dest_port    = dport;

    bpf_get_current_comm(evt->comm, sizeof(evt->comm));

    __u16 family = 0;
    bpf_probe_read_kernel(&family, sizeof(family), &sk->__sk_common.skc_family);

    if (family == AF_INET) {
        evt->ip_version = 4;
        bpf_probe_read_kernel(&evt->dest_ip, sizeof(evt->dest_ip),
                              &sk->__sk_common.skc_daddr);
        bpf_probe_read_kernel(&evt->source_ip, sizeof(evt->source_ip),
                              &sk->__sk_common.skc_rcv_saddr);
        bpf_probe_read_kernel(&evt->source_port, sizeof(evt->source_port),
                              &sk->__sk_common.skc_num);
    }

    // Capture first 512 bytes of payload (tokens typically in header / early body)
    if (size > 0) {
        struct iov_iter *iter = &msg->msg_iter;
        const void __user *base = NULL;
        bpf_probe_read_kernel(&base, sizeof(base), &iter->iov->iov_base);
        if (base) {
            __u32 cap = size < 512 ? (__u32)size : 512;
            bpf_probe_read_user(evt->payload_bytes, cap, base);
            evt->captured_len = cap;
            evt->payload_hash = fnv1a_hash(evt->payload_bytes, cap < 64 ? cap : 64);
        }
    }

    taint_events.ringbuf_submit(evt, 0);
    return 0;
}
"""

BPF_TCP_RECVMSG = _COMMON_PREAMBLE + """
int kprobe__tcp_recvmsg(struct pt_regs *ctx, struct sock *sk,
                        struct msghdr *msg, size_t len, int flags,
                        int *addr_len) {
    // ── Port filter ──────────────────────────────────────────────────────
    // For recvmsg the "local" port (skc_num) is what we monitor; this is
    // the server-side port that accepted the connection.
    __u16 lport = 0;
    bpf_probe_read_kernel(&lport, sizeof(lport), &sk->__sk_common.skc_num);
    u8 *port_hit = monitored_ports.lookup(&lport);
    {
        u16 sentinel = 0;
        u8 *has_filter = monitored_ports.lookup(&sentinel);
        if (has_filter && !port_hit) {
            return 0;
        }
    }

    struct taint_event *evt = taint_events.ringbuf_reserve(sizeof(struct taint_event));
    if (!evt)
        return 0;

    __builtin_memset(evt, 0, sizeof(*evt));

    evt->timestamp_ns = bpf_ktime_get_ns();
    evt->dest_pid     = bpf_get_current_pid_tgid() >> 32;
    evt->dest_tid     = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    evt->flow_type    = 1;
    evt->payload_size = (len > 0xFFFFFFFF) ? 0xFFFFFFFF : (__u32)len;
    evt->dest_port    = lport;

    bpf_get_current_comm(evt->comm, sizeof(evt->comm));

    __u16 family = 0;
    bpf_probe_read_kernel(&family, sizeof(family), &sk->__sk_common.skc_family);

    if (family == AF_INET) {
        evt->ip_version = 4;
        bpf_probe_read_kernel(&evt->source_ip, sizeof(evt->source_ip),
                              &sk->__sk_common.skc_daddr);
        bpf_probe_read_kernel(&evt->dest_ip, sizeof(evt->dest_ip),
                              &sk->__sk_common.skc_rcv_saddr);
        bpf_probe_read_kernel(&evt->source_port, sizeof(evt->source_port),
                              &sk->__sk_common.skc_dport);
        evt->source_port = __builtin_bswap16(evt->source_port);
    }

    taint_events.ringbuf_submit(evt, 0);
    return 0;
}
"""

BPF_SECURITY_SOCKET_CONNECT = _COMMON_PREAMBLE + """
#include <linux/un.h>

int kprobe__security_socket_connect(struct pt_regs *ctx, struct socket *sock,
                                     struct sockaddr *address, int addrlen) {
    __u16 family = 0;
    bpf_probe_read_kernel(&family, sizeof(family), &address->sa_family);

    if (family != AF_INET)
        return 0;

    struct taint_event *evt = taint_events.ringbuf_reserve(sizeof(struct taint_event));
    if (!evt)
        return 0;

    __builtin_memset(evt, 0, sizeof(*evt));

    evt->timestamp_ns = bpf_ktime_get_ns();
    evt->source_pid   = bpf_get_current_pid_tgid() >> 32;
    evt->source_tid   = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    evt->flow_type    = 2;
    evt->ip_version   = 4;

    bpf_get_current_comm(evt->comm, sizeof(evt->comm));

    struct sockaddr_in *addr_in = (struct sockaddr_in *)address;
    bpf_probe_read_kernel(&evt->dest_ip, sizeof(evt->dest_ip), &addr_in->sin_addr.s_addr);
    bpf_probe_read_kernel(&evt->dest_port, sizeof(evt->dest_port), &addr_in->sin_port);
    evt->dest_port = __builtin_bswap16(evt->dest_port);

    taint_events.ringbuf_submit(evt, 0);
    return 0;
}
"""

# ── Inbound Payload Capture ───────────────────────────────────────────────────
# Hooks sys_enter_read (stash fd/buf/count) + sys_exit_read (capture bytes).
# Both probes share the read_ctx_map, so they must be compiled together as one
# BPF object.  flow_type=5; captured_len tells userspace how many bytes landed
# in payload_bytes.

_SYS_EXIT_READ_PREAMBLE = _COMMON_PREAMBLE + """
struct read_ctx {
    __u32 fd;
    __u64 buf;
    __u64 count;
};
BPF_PERCPU_HASH(read_ctx_map, __u32, struct read_ctx, 4096);
"""

BPF_SYS_EXIT_READ = _SYS_EXIT_READ_PREAMBLE + """
TRACEPOINT_PROBE(syscalls, sys_enter_read) {
    int fd = args->fd;
    if (fd <= 2)
        return 0;
    if (args->count < 4)
        return 0;
    __u32 tid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    struct read_ctx ctx = {};
    ctx.fd    = (__u32)fd;
    ctx.buf   = (__u64)(uintptr_t)args->buf;
    ctx.count = args->count;
    read_ctx_map.update(&tid, &ctx);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_exit_read) {
    long ret = args->ret;
    if (ret < 4)
        return 0;
    __u32 tid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    struct read_ctx *ctx = read_ctx_map.lookup(&tid);
    if (!ctx)
        return 0;
    __u32 cap = (__u32)ret;
    if (cap > 512)
        cap = 512;
    struct taint_event *evt = taint_events.ringbuf_reserve(sizeof(struct taint_event));
    if (!evt) {
        read_ctx_map.delete(&tid);
        return 0;
    }
    __builtin_memset(evt, 0, sizeof(*evt));
    evt->timestamp_ns = bpf_ktime_get_ns();
    evt->dest_pid     = bpf_get_current_pid_tgid() >> 32;
    evt->dest_tid     = tid;
    evt->flow_type    = 5;
    evt->payload_size = (__u32)ret;
    evt->source_port  = (__u16)(ctx->fd & 0xFFFF);
    bpf_get_current_comm(evt->comm, sizeof(evt->comm));
    void *ubuf = (void *)(uintptr_t)ctx->buf;
    if (ubuf) {
        bpf_probe_read_user(evt->payload_bytes, cap, ubuf);
        evt->captured_len = cap;
        evt->payload_hash = fnv1a_hash(evt->payload_bytes, cap < 64 ? cap : 64);
    }
    read_ctx_map.delete(&tid);
    taint_events.ringbuf_submit(evt, 0);
    return 0;
}
"""


# ── Outbound Payload Capture ──────────────────────────────────────────────────
# Hooks sys_enter_write (stash fd/buf/count) + sys_exit_write (capture bytes
# actually sent).  Both probes share write_ctx_map and are compiled together as
# one BPF object.  flow_type=6; source_pid is the writing process; source_port
# carries the fd so userspace can correlate to the socket.  captured_len tells
# userspace how many bytes are stored in payload_bytes (0..256).

_SYS_EXIT_WRITE_PREAMBLE = _COMMON_PREAMBLE + """
struct write_ctx {
    __u32 fd;
    __u64 buf;
    __u64 count;
};
BPF_PERCPU_HASH(write_ctx_map, __u32, struct write_ctx, 4096);
"""

BPF_SYS_EXIT_WRITE = _SYS_EXIT_WRITE_PREAMBLE + """
TRACEPOINT_PROBE(syscalls, sys_enter_write) {
    int fd = args->fd;
    if (fd <= 2)
        return 0;
    if (args->count < 4)
        return 0;
    __u32 tid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    struct write_ctx ctx = {};
    ctx.fd    = (__u32)fd;
    ctx.buf   = (__u64)(uintptr_t)args->buf;
    ctx.count = args->count;
    write_ctx_map.update(&tid, &ctx);
    return 0;
}

TRACEPOINT_PROBE(syscalls, sys_exit_write) {
    long ret = args->ret;
    if (ret < 4)
        return 0;
    __u32 tid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    struct write_ctx *ctx = write_ctx_map.lookup(&tid);
    if (!ctx)
        return 0;
    __u32 cap = (__u32)ret;
    if (cap > 512)
        cap = 512;
    struct taint_event *evt = taint_events.ringbuf_reserve(sizeof(struct taint_event));
    if (!evt) {
        write_ctx_map.delete(&tid);
        return 0;
    }
    __builtin_memset(evt, 0, sizeof(*evt));
    evt->timestamp_ns = bpf_ktime_get_ns();
    evt->source_pid   = bpf_get_current_pid_tgid() >> 32;
    evt->source_tid   = tid;
    evt->flow_type    = 6;
    evt->payload_size = (__u32)ret;
    evt->source_port  = (__u16)(ctx->fd & 0xFFFF);
    bpf_get_current_comm(evt->comm, sizeof(evt->comm));
    void *ubuf = (void *)(uintptr_t)ctx->buf;
    if (ubuf) {
        bpf_probe_read_user(evt->payload_bytes, cap, ubuf);
        evt->captured_len = cap;
        evt->payload_hash = fnv1a_hash(evt->payload_bytes, cap < 64 ? cap : 64);
    }
    write_ctx_map.delete(&tid);
    taint_events.ringbuf_submit(evt, 0);
    return 0;
}
"""


# Program registry — maps probe name to BPF C source.
_BPF_PROGRAMS: dict[str, str] = {
    "tcp_sendmsg": BPF_TCP_SENDMSG,
    "tcp_recvmsg": BPF_TCP_RECVMSG,
    "security_socket_connect": BPF_SECURITY_SOCKET_CONNECT,
    # sys_exit_read compiles enter+exit together so they share read_ctx_map
    "sys_exit_read": BPF_SYS_EXIT_READ,
    # sys_exit_write compiles enter+exit together so they share write_ctx_map
    "sys_exit_write": BPF_SYS_EXIT_WRITE,
}


# ── Data Types ───────────────────────────────────────────────────────────────

_FLOW_NAMES = {0: "SEND", 1: "RECV", 2: "CONN", 3: "WRIT", 4: "READ", 5: "RINB", 6: "WOUT"}


@dataclass
class ObserverEvent:
    """Python mirror of the BPF taint_event struct."""

    timestamp_ns: int
    pid: int
    tid: int
    comm: str
    flow_type: int
    payload_size: int
    payload_hash: int
    source_port: int
    dest_port: int
    source_ip: str
    dest_ip: str
    # Populated only for flow_type=5 (read_exit inbound capture)
    captured_len: int
    payload_bytes: bytes
    # Container attribution — populated by PidServiceMapper
    service_name: str = ""   # e.g. "api-service", "postgres"
    container_id: str = ""   # 12-char hex prefix, or "" for non-containerised


# ── PID → Service Mapper ─────────────────────────────────────────────────────


class _PidServiceMapper:
    """
    Minimal PID → (service_name, container_id) resolver for the observer sidecar.

    Reads /proc/{pid}/cgroup to extract the container ID, then calls
    docker inspect to get the compose service label. Results are cached since
    PIDs are stable during a compose session.
    """

    def __init__(self) -> None:
        self._cache: dict[int, tuple[str, str]] = {}
        self._container_cache: dict[str, str] = {}  # container_id → service_name

    def resolve(self, pid: int) -> tuple[str, str]:
        if pid in self._cache:
            return self._cache[pid]
        result = self._resolve_uncached(pid)
        self._cache[pid] = result
        return result

    def _resolve_uncached(self, pid: int) -> tuple[str, str]:
        import subprocess

        container_id = self._pid_to_container_id(pid)
        if not container_id:
            return (self._pid_to_comm(pid), "")

        if container_id in self._container_cache:
            return (self._container_cache[container_id], container_id)

        service = self._container_id_to_service(container_id, subprocess)
        self._container_cache[container_id] = service
        return (service, container_id)

    @staticmethod
    def _pid_to_container_id(pid: int) -> str | None:
        try:
            with open(f"/proc/{pid}/cgroup") as f:
                for line in f:
                    parts = line.strip().split("/")
                    for i, part in enumerate(parts):
                        if part == "docker" and i + 1 < len(parts):
                            cid = parts[i + 1].replace(".scope", "")
                            if len(cid) >= 12:
                                return cid[:12]
                        if part.startswith("docker-") and part.endswith(".scope"):
                            cid = part[7:-6]
                            if len(cid) >= 12:
                                return cid[:12]
        except (OSError, IndexError):
            pass
        return None

    @staticmethod
    def _container_id_to_service(container_id: str, subprocess: Any) -> str:
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
        except Exception:
            pass
        return f"container:{container_id}"

    @staticmethod
    def _pid_to_comm(pid: int) -> str:
        try:
            with open(f"/proc/{pid}/comm") as f:
                return f.read().strip() or f"pid:{pid}"
        except OSError:
            return f"pid:{pid}"


class _CtypesTaintEvent(ctypes.Structure):
    """ctypes layout matching the BPF taint_event struct.

    Must stay byte-for-byte identical to the C struct in _TAINT_EVENT_STRUCT.
    payload_bytes is 512 bytes (increased from 256 for split-packet reassembly).
    """

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
        ("payload_bytes", ctypes.c_char * 512),
    ]


def _ip_to_str(ip_int: int) -> str:
    """Convert a network-byte-order u32 to dotted-quad string."""
    try:
        return socket.inet_ntoa(struct.pack("!I", ip_int))
    except (OSError, struct.error):
        return "0.0.0.0"


def _parse_event(data: Any, mapper: _PidServiceMapper | None = None) -> ObserverEvent:
    """Parse a raw ring buffer event into an ObserverEvent."""
    ptr = ctypes.cast(data, ctypes.POINTER(_CtypesTaintEvent))
    raw = ptr.contents

    # For send/connect/write-out events the acting PID is source_pid; for recv/read it's dest_pid.
    pid = raw.source_pid if raw.flow_type in (0, 2, 6) else raw.dest_pid
    tid = raw.source_tid if raw.flow_type in (0, 2, 6) else raw.dest_tid

    cap = raw.captured_len
    payload_bytes = bytes(raw.payload_bytes[:cap]) if cap > 0 else b""

    service_name = ""
    container_id = ""
    if mapper is not None and pid:
        with contextlib.suppress(Exception):
            service_name, container_id = mapper.resolve(pid)

    return ObserverEvent(
        timestamp_ns=raw.timestamp_ns,
        pid=pid,
        tid=tid,
        comm=raw.comm.decode("utf-8", errors="replace").rstrip("\x00"),
        flow_type=raw.flow_type,
        payload_size=raw.payload_size,
        payload_hash=raw.payload_hash,
        source_port=raw.source_port,
        dest_port=raw.dest_port,
        source_ip=_ip_to_str(raw.source_ip),
        dest_ip=_ip_to_str(raw.dest_ip),
        captured_len=cap,
        payload_bytes=payload_bytes,
        service_name=service_name,
        container_id=container_id,
    )


def _safe_ascii(data: bytes, width: int = 64) -> str:
    """Return a printable ASCII snippet of data, replacing non-printable bytes with '.'."""
    return "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in data[:width])


def _format_event(evt: ObserverEvent) -> str:
    """Format an event as a human-readable line for stdout."""
    flow = _FLOW_NAMES.get(evt.flow_type, "????")

    # Prefer resolved service name; fall back to container short-ID then comm.
    if evt.service_name:
        node_label = evt.service_name
    elif evt.container_id:
        node_label = f"container:{evt.container_id}"
    else:
        node_label = evt.comm

    base = (
        f"[{flow}] pid={evt.pid:<6d} service={node_label:<20s} comm={evt.comm:<16s} "
        f"{evt.source_ip}:{evt.source_port} -> {evt.dest_ip}:{evt.dest_port} "
        f"size={evt.payload_size}"
    )
    if evt.container_id:
        base += f" cid={evt.container_id}"
    if evt.flow_type in (5, 6) and evt.captured_len > 0:
        # Print hex prefix + safe-ASCII snippet so TAINT= tokens are visible
        # flow_type=5: inbound read bytes (RINB)
        # flow_type=6: outbound write bytes (WOUT) — same format, direction differs
        hex_prefix = evt.payload_bytes[:32].hex(" ")
        ascii_snip = _safe_ascii(evt.payload_bytes, 64)
        base += f"\n    cap={evt.captured_len} fd={evt.source_port} hex=[{hex_prefix}] ascii=[{ascii_snip}]"
    return base


# ── BPF Loader ───────────────────────────────────────────────────────────────


class ObserverBpfLoader:
    """
    Loads BPF programs via bcc and provides ring buffer polling.

    Graceful degradation: if bcc is unavailable, runs in degraded mode
    with zero probes loaded and empty event stream.
    """

    def __init__(self, probe_names: list[str]) -> None:
        self._probe_names = probe_names
        self._bpf: Any = None
        self._loaded: dict[str, bool] = {}
        self._errors: dict[str, str] = {}
        self._bcc_available = False

    def load_all(self) -> int:
        """Compile and attach BPF programs. Returns count loaded."""
        try:
            from bcc import BPF  # type: ignore[import-untyped]

            self._bcc_available = True
        except ImportError:
            log.warning("bcc not available — running in degraded mode (no eBPF)")
            for name in self._probe_names:
                self._loaded[name] = False
                self._errors[name] = "bcc not installed"
            return 0

        loaded = 0
        for name in self._probe_names:
            source = _BPF_PROGRAMS.get(name)
            if source is None:
                self._loaded[name] = False
                self._errors[name] = f"unknown probe: {name}"
                log.warning("Unknown probe name: %s", name)
                continue

            try:
                bpf = BPF(text=source)
                self._bpf = bpf
                self._loaded[name] = True
                loaded += 1
                log.info("BPF program loaded: %s", name)
            except Exception as exc:
                self._loaded[name] = False
                self._errors[name] = str(exc)[:200]
                log.warning("BPF program %s failed to load: %s", name, exc)

        return loaded

    def setup_ring_buffer(self, callback: Any) -> bool:
        """Register a ring buffer callback. Returns True on success."""
        if self._bpf is None:
            return False
        try:
            self._bpf["taint_events"].open_ring_buffer(callback)
            log.info("Ring buffer callback registered")
            return True
        except Exception as exc:
            log.warning("Failed to set up ring buffer: %s", exc)
            return False

    def poll(self, timeout_ms: int = 100) -> None:
        """Poll the ring buffer for events."""
        if self._bpf is None:
            return
        try:
            self._bpf.ring_buffer_poll(timeout=timeout_ms)
        except Exception as exc:
            log.debug("Ring buffer poll error: %s", exc)

    def populate_monitored_ports(self, ports: list[int]) -> bool:
        """
        Populate the BPF ``monitored_ports`` hash map with *ports*.

        When the map is non-empty the BPF probes skip any packet whose
        destination (or local) port is NOT in this set.  Passing an empty
        list clears the map, restoring open-mode (all ports monitored).

        A sentinel entry with key=0 is always written alongside real ports so
        the BPF probe can distinguish an intentionally empty map (open mode)
        from a populated-but-not-matching map (filter mode).

        Returns True if the map was updated, False if BPF is unavailable.
        """
        if self._bpf is None:
            log.warning("populate_monitored_ports: BPF not loaded, skipping")
            return False

        try:
            port_map = self._bpf["monitored_ports"]

            # Clear existing entries first
            port_map.clear()

            if not ports:
                # Open mode: empty map → all ports accepted
                log.info("monitored_ports cleared — open mode (all ports)")
                return True

            # Write sentinel key=0 so BPF knows filter mode is active
            port_map[ctypes.c_uint16(0)] = ctypes.c_uint8(0)

            for port in ports:
                port_map[ctypes.c_uint16(port)] = ctypes.c_uint8(1)

            log.info("monitored_ports updated: %s", sorted(ports))
            return True
        except Exception as exc:
            log.warning("Failed to update monitored_ports map: %s", exc)
            return False

    @property
    def programs_loaded(self) -> int:
        return sum(1 for v in self._loaded.values() if v)

    @property
    def is_available(self) -> bool:
        return self._bcc_available

    @property
    def program_status(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name in self._probe_names:
            result[name] = {"loaded": self._loaded.get(name, False)}
            if name in self._errors:
                result[name]["error"] = self._errors[name]
        return result


# ── Rate Limiter ─────────────────────────────────────────────────────────────


class _RateLimiter:
    """Token-bucket rate limiter for event output."""

    def __init__(self, max_per_second: int = 10_000) -> None:
        self._max = max_per_second
        self._count = 0
        self._window_start = time.monotonic()
        self._dropped = 0
        self._last_drop_warn = 0.0

    def allow(self) -> bool:
        now = time.monotonic()
        if now - self._window_start >= 1.0:
            if self._dropped > 0 and now - self._last_drop_warn >= 5.0:
                log.warning(
                    "Rate limiter dropped %d events in last window",
                    self._dropped,
                )
                self._last_drop_warn = now
            self._count = 0
            self._dropped = 0
            self._window_start = now

        if self._count >= self._max:
            self._dropped += 1
            return False

        self._count += 1
        return True

    @property
    def total_dropped(self) -> int:
        return self._dropped


# ── Event Printer Thread ─────────────────────────────────────────────────────


class _EventPrinter:
    """Receives ring buffer events, formats them, prints to stdout."""

    def __init__(self, rate_limiter: _RateLimiter, mapper: _PidServiceMapper) -> None:
        self._rate_limiter = rate_limiter
        self._mapper = mapper
        self._events_seen = 0
        self._lock = threading.Lock()

    def handle_event(self, ctx: Any, data: Any, size: int) -> None:
        """Ring buffer callback — called from the poll thread."""
        try:
            evt = _parse_event(data, self._mapper)
            with self._lock:
                self._events_seen += 1

            if self._rate_limiter.allow():
                line = _format_event(evt)
                print(line, flush=True)
        except Exception as exc:
            log.debug("Event parse error: %s", exc)

    @property
    def events_seen(self) -> int:
        with self._lock:
            return self._events_seen


# ── Phase 1: Process Lifecycle Monitor ───────────────────────────────────────


@dataclass
class ProcessLifecycleEvent:
    """A process fork / exec / exit event read from /proc."""

    timestamp_ns: int
    event_type: str  # "process_fork" | "process_exec" | "process_exit"
    pid: int
    ppid: int
    comm: str
    exit_code: int | None = None


class ProcessLifecycleMonitor:
    """
    Polls /proc to detect process fork and exit events for a set of watched PIDs.

    Supplements the eBPF ring buffer (network-focused) with process-tree context.
    Used during Simula proposal runs to reconstruct the complete process tree
    for a given correlation context.

    Design:
    - Snapshot /proc/<pid>/status for each new PID at attach() time.
    - Poll periodically; emit fork events for new PIDs, exit events for gone PIDs.
    - Pure read-only: never writes to kernel or process memory.
    """

    def __init__(self, *, poll_interval_s: float = 0.5) -> None:
        self._poll_interval = poll_interval_s
        self._known_pids: dict[int, dict[str, Any]] = {}  # pid → {ppid, comm, start_ns}
        self._events: list[ProcessLifecycleEvent] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Begin background polling of /proc."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="proc-lifecycle",
        )
        self._thread.start()
        log.info("ProcessLifecycleMonitor started (interval=%.2fs)", self._poll_interval)

    def stop(self) -> None:
        """Stop background polling."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        log.info("ProcessLifecycleMonitor stopped (%d events)", len(self._events))

    def drain_events(self) -> list[ProcessLifecycleEvent]:
        """Return and clear accumulated lifecycle events (thread-safe)."""
        with self._lock:
            result = self._events[:]
            self._events.clear()
            return result

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._scan_proc()
            except Exception as exc:
                log.debug("proc scan error: %s", exc)
            self._stop.wait(timeout=self._poll_interval)

    def _scan_proc(self) -> None:
        """Single /proc scan: detect new and gone PIDs."""
        try:
            current_pids = {
                int(entry)
                for entry in os.listdir("/proc")
                if entry.isdigit()
            }
        except OSError:
            return

        now_ns = int(time.time_ns())

        with self._lock:
            # New PIDs → fork events
            for pid in current_pids - set(self._known_pids):
                info = self._read_pid_info(pid)
                if info is None:
                    continue
                self._known_pids[pid] = info
                self._events.append(ProcessLifecycleEvent(
                    timestamp_ns=now_ns,
                    event_type="process_fork",
                    pid=pid,
                    ppid=info["ppid"],
                    comm=info["comm"],
                ))

            # Gone PIDs → exit events
            for pid in set(self._known_pids) - current_pids:
                info = self._known_pids.pop(pid)
                self._events.append(ProcessLifecycleEvent(
                    timestamp_ns=now_ns,
                    event_type="process_exit",
                    pid=pid,
                    ppid=info["ppid"],
                    comm=info["comm"],
                    exit_code=self._read_exit_code(pid),
                ))

    @staticmethod
    def _read_pid_info(pid: int) -> dict[str, Any] | None:
        """Read ppid and comm from /proc/<pid>/status. Returns None if race."""
        try:
            status: dict[str, str] = {}
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if ":" in line:
                        key, _, val = line.partition(":")
                        status[key.strip()] = val.strip()

            ppid = int(status.get("PPid", "0"))
            comm_raw = status.get("Name", f"pid:{pid}")
            return {"ppid": ppid, "comm": comm_raw[:15]}
        except (OSError, ValueError):
            return None

    @staticmethod
    def _read_exit_code(pid: int) -> int | None:
        """Best-effort exit code from /proc/<pid>/stat field 52 (not always available)."""
        try:
            with open(f"/proc/{pid}/stat") as f:
                fields = f.read().split()
            # field 51 (0-indexed) is exit_signal; not a true exit code but useful
            return int(fields[51]) if len(fields) > 51 else None
        except (OSError, ValueError, IndexError):
            return None


# ── Phase 1: Interaction Graph Builder ───────────────────────────────────────


@dataclass
class _InteractionEdgeRaw:
    """Mutable accumulator for a single interaction edge."""

    source: str
    target: str
    interaction_type: str
    count: int = 1


class InteractionGraphBuilder:
    """
    Builds a process-to-service interaction graph from ObserverEvents.

    Receives network flow events from the ring buffer and process lifecycle
    events from ProcessLifecycleMonitor, then emits a summary interaction
    graph when `build()` is called.

    Graph nodes: service names (from PidServiceMapper) or comm labels.
    Graph edges: {source → target, type: "http_call" | "db_query" | "socket" | "fork"}

    Thread-safe: all mutation protected by _lock.
    """

    # Well-known ports → interaction semantic
    _PORT_LABELS: dict[int, str] = {
        5432: "db_query",    # PostgreSQL
        3306: "db_query",    # MySQL
        1433: "db_query",    # MSSQL
        6379: "cache_access",  # Redis
        6380: "cache_access",
        27017: "db_query",   # MongoDB
        9200: "search_query",  # Elasticsearch
        80: "http_call",
        443: "http_call",
        8080: "http_call",
        8443: "http_call",
        8000: "http_call",
        3000: "http_call",
        5672: "queue_publish",  # RabbitMQ
        4222: "queue_publish",  # NATS
    }

    def __init__(self, mapper: _PidServiceMapper) -> None:
        self._mapper = mapper
        self._edges: dict[tuple[str, str, str], _InteractionEdgeRaw] = {}
        self._nodes: set[str] = set()
        self._lock = threading.Lock()

    def ingest_observer_event(self, evt: ObserverEvent) -> None:
        """Record an interaction from a network ObserverEvent."""
        # Only track outbound connections/sends (flow types 0=SEND, 2=CONN)
        if evt.flow_type not in (0, 2):
            return

        source_label = evt.service_name or evt.comm or f"pid:{evt.pid}"
        dest_port = evt.dest_port

        # Map dest IP:port to a service label via reverse mapper or port heuristic
        dest_label = self._resolve_dest(evt.dest_ip, dest_port)
        interaction_type = self._PORT_LABELS.get(dest_port, "socket")

        self._record_edge(source_label, dest_label, interaction_type)

    def ingest_lifecycle_event(self, evt: ProcessLifecycleEvent) -> None:
        """Record a process fork as a parent→child interaction."""
        if evt.event_type != "process_fork" or evt.ppid == 0:
            return

        parent_label = self._comm_for_pid(evt.ppid) or f"pid:{evt.ppid}"
        child_label = evt.comm or f"pid:{evt.pid}"
        self._record_edge(parent_label, child_label, "fork")

    def _record_edge(self, source: str, target: str, interaction_type: str) -> None:
        key = (source, target, interaction_type)
        with self._lock:
            self._nodes.add(source)
            self._nodes.add(target)
            if key in self._edges:
                self._edges[key].count += 1
            else:
                self._edges[key] = _InteractionEdgeRaw(
                    source=source,
                    target=target,
                    interaction_type=interaction_type,
                )

    def _resolve_dest(self, dest_ip: str, dest_port: int) -> str:
        """Try to resolve dest IP to a service name; fall back to IP:port label."""
        # Scan known PIDs for one listening on dest_port in the same process namespace.
        # Best-effort: if we can't resolve, label as "external" or "ip:port".
        if dest_ip in ("127.0.0.1", "::1", "0.0.0.0"):
            return f"local:{dest_port}"
        if dest_port in self._PORT_LABELS:
            return f"{self._PORT_LABELS[dest_port]}:{dest_port}"
        return f"{dest_ip}:{dest_port}"

    def _comm_for_pid(self, pid: int) -> str | None:
        """Read process name for a PID, for fork edge labels."""
        try:
            with open(f"/proc/{pid}/comm") as f:
                return f.read().strip()
        except OSError:
            return None

    def build(self) -> dict[str, Any]:
        """
        Return a serialisable dict representation of the interaction graph.

        Format matches InteractionGraph from inspector/taint_types.py so the
        caller can construct a full InteractionGraph model.
        """
        with self._lock:
            nodes = sorted(self._nodes)
            edges = [
                {
                    "source": e.source,
                    "target": e.target,
                    "interaction_type": e.interaction_type,
                    "count": e.count,
                }
                for e in self._edges.values()
            ]
            interaction_types = sorted({e.interaction_type for e in self._edges.values()})

        return {
            "nodes": nodes,
            "edges": edges,
            "interaction_types": interaction_types,
            "edge_count": len(edges),
            "node_count": len(nodes),
        }

    def reset(self) -> None:
        """Clear accumulated graph data (call between proposal runs)."""
        with self._lock:
            self._edges.clear()
            self._nodes.clear()


def _poll_loop(
    loader: ObserverBpfLoader,
    poll_interval_ms: int,
    stop_event: threading.Event,
) -> None:
    """Background thread that polls the BPF ring buffer."""
    log.info("Poll loop started (interval=%dms)", poll_interval_ms)
    while not stop_event.is_set():
        try:
            loader.poll(timeout_ms=poll_interval_ms)
        except Exception as exc:
            log.debug("Poll error: %s", exc)
        stop_event.wait(timeout=poll_interval_ms / 1000.0)
    log.info("Poll loop stopped")


# ── Health HTTP Server ───────────────────────────────────────────────────────


class _HealthHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler — serves /health and /graph."""

    loader: ObserverBpfLoader
    printer: _EventPrinter
    start_time: float
    graph_builder: InteractionGraphBuilder | None = None
    lifecycle_monitor: ProcessLifecycleMonitor | None = None

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Suppress default stderr logging

    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/")
        if path == "/health":
            self._handle_health()
        elif path == "/graph":
            self._handle_graph()
        else:
            self._json_response(404, {"error": "not found"})

    def _handle_health(self) -> None:
        uptime = time.monotonic() - self.start_time
        status = "ready" if self.loader.programs_loaded > 0 else "degraded"
        if not self.loader.is_available:
            status = "degraded"

        code = 200 if status == "ready" else 503
        self._json_response(code, {
            "status": status,
            "probes_loaded": self.loader.programs_loaded,
            "events_seen": self.printer.events_seen,
            "uptime_seconds": round(uptime, 1),
            "programs": self.loader.program_status,
        })

    def _handle_graph(self) -> None:
        """Return current interaction graph snapshot (for Phase 8 assembly)."""
        if self.graph_builder is None:
            self._json_response(503, {"error": "graph builder not active"})
            return
        self._json_response(200, self.graph_builder.build())

    def _json_response(self, code: int, data: Any) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Simula Observer — eBPF Network Event Printer")
    parser.add_argument("--port", type=int, default=9472, help="Health endpoint port (default: 9472)")
    parser.add_argument(
        "--probes",
        type=str,
        default="tcp_sendmsg,tcp_recvmsg,security_socket_connect,sys_exit_read,sys_exit_write",
        help="Comma-separated probe names to attach",
    )
    parser.add_argument("--poll-interval-ms", type=int, default=50, help="BPF poll interval in ms")
    parser.add_argument(
        "--monitored-ports",
        type=str,
        default="",
        help=(
            "Comma-separated list of destination ports to monitor. "
            "Empty string (default) monitors ALL ports (open mode). "
            "Example: 80,443,8080,5432,3306,6379"
        ),
    )
    parser.add_argument(
        "--enable-lifecycle",
        action="store_true",
        default=False,
        help="Enable /proc-based process lifecycle monitoring (Phase 1 observability)",
    )
    parser.add_argument(
        "--lifecycle-interval",
        type=float,
        default=0.5,
        help="Process lifecycle poll interval in seconds (default: 0.5)",
    )
    args = parser.parse_args()

    probe_names = [p.strip() for p in args.probes.split(",") if p.strip()]
    log.info("Starting observer on port %d with probes: %s", args.port, probe_names)

    # 1. Load BPF programs
    loader = ObserverBpfLoader(probe_names)
    loaded = loader.load_all()
    if loaded > 0:
        log.info("Loaded %d/%d BPF programs", loaded, len(probe_names))
    else:
        log.warning("No BPF programs loaded — running in degraded mode")

    # 1b. Populate port filter map.
    #     Default: HTTP + HTTPS + common alt-HTTP + known sensitive sinks.
    if loaded > 0:
        if args.monitored_ports:
            ports_to_monitor = [
                int(p.strip()) for p in args.monitored_ports.split(",") if p.strip().isdigit()
            ]
        else:
            ports_to_monitor = [
                # HTTP / HTTPS / common app ports
                80, 443, 8080, 8443, 8000, 3000,
                # Database sinks (must mirror SENSITIVE_SINK_PORTS in taint_flow_linker)
                5432,   # PostgreSQL
                3306,   # MySQL / MariaDB
                1433,   # MSSQL
                1521,   # Oracle
                6379,   # Redis
                6380,   # Redis TLS
                9200,   # Elasticsearch HTTP
                9300,   # Elasticsearch transport
                27017,  # MongoDB
                27018,  # MongoDB shard
                5984,   # CouchDB
                9042,   # Cassandra CQL
                5672,   # RabbitMQ AMQP
                4222,   # NATS
            ]
        loader.populate_monitored_ports(ports_to_monitor)

    # 2. Set up event printer with rate limiter + container attribution mapper
    rate_limiter = _RateLimiter(max_per_second=10_000)
    mapper = _PidServiceMapper()
    printer = _EventPrinter(rate_limiter, mapper)

    # 2b. Phase 1: interaction graph builder (accumulates topology from ring events)
    graph_builder = InteractionGraphBuilder(mapper)

    # Wrap the printer callback to also feed the graph builder
    _original_handle = printer.handle_event

    def _handle_event_with_graph(ctx: Any, data: Any, size: int) -> None:
        _original_handle(ctx, data, size)
        try:
            evt = _parse_event(data, mapper)
            graph_builder.ingest_observer_event(evt)
        except Exception:
            pass

    # 2c. Phase 1: optional process lifecycle monitor
    lifecycle_monitor: ProcessLifecycleMonitor | None = None
    if args.enable_lifecycle:
        lifecycle_monitor = ProcessLifecycleMonitor(
            poll_interval_s=args.lifecycle_interval,
        )
        lifecycle_monitor.start()
        log.info("Process lifecycle monitor active")

    # 3. Register ring buffer callback (with graph ingestion hooked in)
    if loaded > 0:
        loader.setup_ring_buffer(_handle_event_with_graph)

    # Print header
    print("=== Simula Observer — BCC mode ===", flush=True)
    print(f"Probes loaded: {loaded}/{len(probe_names)}", flush=True)
    print(
        f"{'TYPE':<6s} {'PID':<6s} {'SERVICE':<20s} {'COMM':<16s} "
        f"{'SOURCE':<22s} {'DEST':<22s} {'SIZE':<8s}",
        flush=True,
    )
    print("-" * 90, flush=True)

    # 4. Start poll thread
    stop_event = threading.Event()
    poll_thread = threading.Thread(
        target=_poll_loop,
        args=(loader, args.poll_interval_ms, stop_event),
        daemon=True,
        name="bpf-poll",
    )
    poll_thread.start()

    # 5. Configure and start health server
    _HealthHandler.loader = loader
    _HealthHandler.printer = printer
    _HealthHandler.start_time = time.monotonic()
    _HealthHandler.graph_builder = graph_builder
    _HealthHandler.lifecycle_monitor = lifecycle_monitor

    server = HTTPServer(("0.0.0.0", args.port), _HealthHandler)

    def _shutdown(signum: int, frame: Any) -> None:
        log.info("Received signal %d — shutting down", signum)
        stop_event.set()
        if lifecycle_monitor is not None:
            lifecycle_monitor.stop()
        server.shutdown()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info("Health server listening on 0.0.0.0:%d", args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if lifecycle_monitor is not None:
            lifecycle_monitor.stop()
        poll_thread.join(timeout=5)
        graph_snapshot = graph_builder.build()
        log.info(
            "Observer stopped (events_seen=%d, graph_nodes=%d, graph_edges=%d)",
            printer.events_seen,
            graph_snapshot["node_count"],
            graph_snapshot["edge_count"],
        )


if __name__ == "__main__":
    main()
