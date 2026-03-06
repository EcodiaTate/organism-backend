"""
EcodiaOS — Inspector eBPF Program Definitions

BPF C programs stored as Python string constants for runtime compilation via BCC.
Each program attaches to kernel tracepoints/kprobes for cross-service taint
observation. Programs are read-only by design: they observe data flows without
modifying kernel state.

Iron Rules:
  - All BPF programs are BPF_PROG_TYPE_TRACEPOINT or kprobe — never modify
    kernel memory or override function returns.
  - Ring buffer (not perf buffer) for zero-copy userspace reading.
  - Taint tags include timestamp_ns, source_pid, dest_pid, payload_hash.
  - No kernel writes — read-only observation only.
  - FNV-1a hash on first 64 bytes of payload for correlation across send/recv.
"""

from __future__ import annotations

import enum
from typing import Final

import structlog

logger = structlog.get_logger().bind(system="simula.inspector.ebpf_programs")

# Ring buffer size in pages (64 pages = 256KB).
RING_BUFFER_PAGE_CNT: Final[int] = 64

# Taint collector HTTP port inside the sidecar.
TAINT_COLLECTOR_PORT: Final[int] = 9471


# ── Program Types ────────────────────────────────────────────────────────────


class BpfProgramType(enum.StrEnum):
    """Identifies which kernel hook a BPF program attaches to."""

    TCP_SENDMSG = "tcp_sendmsg"
    TCP_RECVMSG = "tcp_recvmsg"
    SOCKET_CONNECT = "security_socket_connect"
    SYS_WRITE = "sys_write"
    SYS_READ = "sys_read"
    SYS_EXIT_READ = "sys_exit_read"


# ── Shared C Definitions ────────────────────────────────────────────────────

# Maximum byte prefix captured per inbound read event.
PAYLOAD_CAPTURE_BYTES: Final[int] = 256

# Taint event struct emitted to ring buffer by all programs.
TAINT_EVENT_STRUCT: Final[str] = """
struct taint_event {
    __u64 timestamp_ns;
    __u32 source_pid;
    __u32 dest_pid;
    __u32 source_tid;
    __u32 dest_tid;
    __u32 payload_hash;
    __u32 payload_size;
    __u8  flow_type;       // 0=tcp_send, 1=tcp_recv, 2=connect, 3=file_write, 4=file_read, 5=read_exit
    __u8  ip_version;      // 4 or 6
    __u16 source_port;
    __u16 dest_port;
    __u32 source_ip;       // IPv4 only (network byte order)
    __u32 dest_ip;
    char  comm[16];
    // Inbound payload byte prefix — populated by sys_exit_read only.
    // Other probes leave this zeroed to keep ring-buffer entries small.
    __u32 captured_len;    // actual bytes stored in payload_bytes (0..256)
    char  payload_bytes[256];
};
"""

# FNV-1a hash — inline C function for hashing first N bytes of payload.
FNV1A_HASH_C: Final[str] = """
static __always_inline __u32 fnv1a_hash(const void *data, int len) {
    __u32 hash = 2166136261u;  // FNV offset basis
    const unsigned char *p = (const unsigned char *)data;

    #pragma unroll
    for (int i = 0; i < 64 && i < len; i++) {
        unsigned char byte;
        if (bpf_probe_read_kernel(&byte, 1, p + i) != 0)
            break;
        hash ^= byte;
        hash *= 16777619u;  // FNV prime
    }
    return hash;
}
"""

# Common includes and ring buffer declaration used by all programs.
_COMMON_PREAMBLE: Final[str] = """
#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <linux/socket.h>
#include <linux/in.h>

""" + TAINT_EVENT_STRUCT + """

BPF_RINGBUF_OUTPUT(taint_events, """ + str(RING_BUFFER_PAGE_CNT) + """);

""" + FNV1A_HASH_C


# ── BPF Programs ────────────────────────────────────────────────────────────


BPF_TCP_SENDMSG: Final[str] = _COMMON_PREAMBLE + """
/*
 * kprobe: tcp_sendmsg
 * Captures outgoing TCP data — source PID, dest socket, payload hash.
 * flow_type = 0
 */
int kprobe__tcp_sendmsg(struct pt_regs *ctx, struct sock *sk,
                        struct msghdr *msg, size_t size) {
    struct taint_event *evt = taint_events.ringbuf_reserve(sizeof(struct taint_event));
    if (!evt)
        return 0;

    __builtin_memset(evt, 0, sizeof(*evt));

    evt->timestamp_ns = bpf_ktime_get_ns();
    evt->source_pid   = bpf_get_current_pid_tgid() >> 32;
    evt->source_tid   = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    evt->flow_type    = 0;  // tcp_send
    evt->payload_size = (size > 0xFFFFFFFF) ? 0xFFFFFFFF : (__u32)size;

    bpf_get_current_comm(evt->comm, sizeof(evt->comm));

    // Extract destination from socket
    __u16 family = 0;
    bpf_probe_read_kernel(&family, sizeof(family), &sk->__sk_common.skc_family);

    if (family == AF_INET) {
        evt->ip_version = 4;
        bpf_probe_read_kernel(&evt->dest_ip, sizeof(evt->dest_ip),
                              &sk->__sk_common.skc_daddr);
        bpf_probe_read_kernel(&evt->source_ip, sizeof(evt->source_ip),
                              &sk->__sk_common.skc_rcv_saddr);
        bpf_probe_read_kernel(&evt->dest_port, sizeof(evt->dest_port),
                              &sk->__sk_common.skc_dport);
        bpf_probe_read_kernel(&evt->source_port, sizeof(evt->source_port),
                              &sk->__sk_common.skc_num);
        evt->dest_port = __builtin_bswap16(evt->dest_port);
    }

    // Hash first 64 bytes of the message for taint correlation
    if (size > 0) {
        struct iov_iter *iter = &msg->msg_iter;
        const void __user *base = NULL;
        bpf_probe_read_kernel(&base, sizeof(base), &iter->iov->iov_base);
        if (base) {
            unsigned char buf[64] = {};
            int read_len = size < 64 ? (__u32)size : 64;
            bpf_probe_read_user(buf, read_len, base);
            evt->payload_hash = fnv1a_hash(buf, read_len);
        }
    }

    taint_events.ringbuf_submit(evt, 0);
    return 0;
}
"""


BPF_TCP_RECVMSG: Final[str] = _COMMON_PREAMBLE + """
/*
 * kprobe: tcp_recvmsg
 * Captures incoming TCP data — receiver PID, source socket, payload hash.
 * flow_type = 1
 *
 * We hook the entry to capture the socket info; the actual payload hash
 * requires a kretprobe or tracepoint on the return path for accurate data.
 * For correlation purposes, socket info + timing is sufficient.
 */
int kprobe__tcp_recvmsg(struct pt_regs *ctx, struct sock *sk,
                        struct msghdr *msg, size_t len, int flags,
                        int *addr_len) {
    struct taint_event *evt = taint_events.ringbuf_reserve(sizeof(struct taint_event));
    if (!evt)
        return 0;

    __builtin_memset(evt, 0, sizeof(*evt));

    evt->timestamp_ns = bpf_ktime_get_ns();
    evt->dest_pid     = bpf_get_current_pid_tgid() >> 32;
    evt->dest_tid     = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    evt->flow_type    = 1;  // tcp_recv
    evt->payload_size = (len > 0xFFFFFFFF) ? 0xFFFFFFFF : (__u32)len;

    bpf_get_current_comm(evt->comm, sizeof(evt->comm));

    // Extract source from socket (the peer address)
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
        bpf_probe_read_kernel(&evt->dest_port, sizeof(evt->dest_port),
                              &sk->__sk_common.skc_num);
        evt->source_port = __builtin_bswap16(evt->source_port);
    }

    taint_events.ringbuf_submit(evt, 0);
    return 0;
}
"""


BPF_SECURITY_SOCKET_CONNECT: Final[str] = _COMMON_PREAMBLE + """
/*
 * kprobe: security_socket_connect
 * Captures connection establishment — PID, target address.
 * flow_type = 2
 *
 * Maps the connection graph between services before data flows.
 */
#include <linux/un.h>

int kprobe__security_socket_connect(struct pt_regs *ctx, struct socket *sock,
                                     struct sockaddr *address, int addrlen) {
    __u16 family = 0;
    bpf_probe_read_kernel(&family, sizeof(family), &address->sa_family);

    // Only trace AF_INET connections (IPv4 TCP/UDP between services)
    if (family != AF_INET)
        return 0;

    struct taint_event *evt = taint_events.ringbuf_reserve(sizeof(struct taint_event));
    if (!evt)
        return 0;

    __builtin_memset(evt, 0, sizeof(*evt));

    evt->timestamp_ns = bpf_ktime_get_ns();
    evt->source_pid   = bpf_get_current_pid_tgid() >> 32;
    evt->source_tid   = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    evt->flow_type    = 2;  // connect
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


BPF_SYS_WRITE: Final[str] = _COMMON_PREAMBLE + """
/*
 * tracepoint: syscalls:sys_enter_write
 * Captures file write operations for file-based taint tracking.
 * flow_type = 3
 *
 * Filters out stdin/stdout/stderr (fd 0/1/2) and tiny writes (< 4 bytes).
 */
TRACEPOINT_PROBE(syscalls, sys_enter_write) {
    int fd = args->fd;

    // Skip stdin/stdout/stderr and tiny writes
    if (fd <= 2)
        return 0;
    if (args->count < 4)
        return 0;

    struct taint_event *evt = taint_events.ringbuf_reserve(sizeof(struct taint_event));
    if (!evt)
        return 0;

    __builtin_memset(evt, 0, sizeof(*evt));

    evt->timestamp_ns = bpf_ktime_get_ns();
    evt->source_pid   = bpf_get_current_pid_tgid() >> 32;
    evt->source_tid   = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    evt->flow_type    = 3;  // file_write
    evt->payload_size = (args->count > 0xFFFFFFFF) ? 0xFFFFFFFF : (__u32)args->count;
    // Store fd in dest_port field (repurposed for file I/O tracking)
    evt->dest_port    = (__u16)(fd & 0xFFFF);

    bpf_get_current_comm(evt->comm, sizeof(evt->comm));

    // Hash first 64 bytes of write buffer
    if (args->buf && args->count > 0) {
        unsigned char buf[64] = {};
        int read_len = args->count < 64 ? (__u32)args->count : 64;
        bpf_probe_read_user(buf, read_len, args->buf);
        evt->payload_hash = fnv1a_hash(buf, read_len);
    }

    taint_events.ringbuf_submit(evt, 0);
    return 0;
}
"""


BPF_SYS_READ: Final[str] = _COMMON_PREAMBLE + """
/*
 * tracepoint: syscalls:sys_enter_read
 * Captures file read operations for file-based taint tracking.
 * flow_type = 4
 *
 * Filters out stdin/stdout/stderr (fd 0/1/2) and tiny reads (< 4 bytes).
 * Note: we capture the read request, not the result — the buffer address
 * is recorded for potential kretprobe correlation.
 */
TRACEPOINT_PROBE(syscalls, sys_enter_read) {
    int fd = args->fd;

    // Skip stdin/stdout/stderr and tiny reads
    if (fd <= 2)
        return 0;
    if (args->count < 4)
        return 0;

    struct taint_event *evt = taint_events.ringbuf_reserve(sizeof(struct taint_event));
    if (!evt)
        return 0;

    __builtin_memset(evt, 0, sizeof(*evt));

    evt->timestamp_ns = bpf_ktime_get_ns();
    evt->dest_pid     = bpf_get_current_pid_tgid() >> 32;
    evt->dest_tid     = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    evt->flow_type    = 4;  // file_read
    evt->payload_size = (args->count > 0xFFFFFFFF) ? 0xFFFFFFFF : (__u32)args->count;
    // Store fd in source_port field (repurposed for file I/O tracking)
    evt->source_port  = (__u16)(fd & 0xFFFF);

    bpf_get_current_comm(evt->comm, sizeof(evt->comm));

    taint_events.ringbuf_submit(evt, 0);
    return 0;
}
"""


# ── Inbound Payload Capture ──────────────────────────────────────────────────
#
# Strategy: hook sys_exit_read (the tracepoint fired after read() returns).
# At this point the kernel has already copied data into the user buffer, so
# bpf_probe_read_user on args->buf is valid and captures real received bytes.
#
# We use a BPF per-CPU hash map to stash the (fd, buf, count) from sys_enter_read
# then read the filled buffer in sys_exit_read, keeping both halves in the same
# BPF object compiled together so they share the same map.
#
# payload_bytes field in taint_event is only populated here (flow_type=5).
# All other probes leave it zeroed — the ring-buffer entry size is the same
# regardless (struct layout is fixed), but captured_len=0 signals "no sample".

_SYS_EXIT_READ_PREAMBLE: Final[str] = _COMMON_PREAMBLE + """
// Per-CPU scratch map: stash enter-side args keyed by tid so exit-side
// can retrieve them. Only non-stdin/stdout/stderr fds are stored.
struct read_ctx {
    __u32 fd;
    __u64 buf;      // user-space buffer pointer (as u64 for portability)
    __u64 count;
};
BPF_PERCPU_HASH(read_ctx_map, __u32, struct read_ctx, 4096);
"""

BPF_SYS_EXIT_READ: Final[str] = _SYS_EXIT_READ_PREAMBLE + """
/*
 * tracepoint: syscalls:sys_enter_read  (enter-side stash)
 * Store (fd, buf, count) for non-trivial reads so the exit-side can
 * capture bytes from the now-filled user buffer.
 */
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

/*
 * tracepoint: syscalls:sys_exit_read  (exit-side capture)
 * flow_type = 5 — inbound payload bytes.
 *
 * Fires after the kernel has filled the user buffer.  We look up the
 * stashed enter-side context, read up to 256 bytes from the user buffer
 * (the actual received data), hash them, and emit an event with the raw
 * byte prefix in payload_bytes.  Tiny or error returns are skipped.
 */
TRACEPOINT_PROBE(syscalls, sys_exit_read) {
    long ret = args->ret;

    // Skip errors and reads that returned fewer than 4 bytes
    if (ret < 4)
        return 0;

    __u32 tid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
    struct read_ctx *ctx = read_ctx_map.lookup(&tid);
    if (!ctx)
        return 0;

    // Determine how many bytes to capture: min(ret, 256)
    __u32 cap = (__u32)ret;
    if (cap > 256)
        cap = 256;

    struct taint_event *evt = taint_events.ringbuf_reserve(sizeof(struct taint_event));
    if (!evt) {
        read_ctx_map.delete(&tid);
        return 0;
    }

    __builtin_memset(evt, 0, sizeof(*evt));

    evt->timestamp_ns = bpf_ktime_get_ns();
    evt->dest_pid     = bpf_get_current_pid_tgid() >> 32;
    evt->dest_tid     = tid;
    evt->flow_type    = 5;  // read_exit (inbound payload capture)
    evt->payload_size = (__u32)ret;
    evt->source_port  = (__u16)(ctx->fd & 0xFFFF);  // fd in source_port (reused)

    bpf_get_current_comm(evt->comm, sizeof(evt->comm));

    // Read the filled user buffer — this is safe post-return
    void *ubuf = (void *)(uintptr_t)ctx->buf;
    if (ubuf) {
        // bpf_probe_read_user requires a compile-time-constant length.
        // We always read the full 256-byte slot; captured_len records actual.
        bpf_probe_read_user(evt->payload_bytes, 256, ubuf);
        evt->captured_len = cap;
        evt->payload_hash = fnv1a_hash(evt->payload_bytes, (__u32)cap < 64 ? (__u32)cap : 64);
    }

    read_ctx_map.delete(&tid);
    taint_events.ringbuf_submit(evt, 0);
    return 0;
}
"""


# ── Program Registry ────────────────────────────────────────────────────────

BPF_PROGRAMS: dict[BpfProgramType, str] = {
    BpfProgramType.TCP_SENDMSG: BPF_TCP_SENDMSG,
    BpfProgramType.TCP_RECVMSG: BPF_TCP_RECVMSG,
    BpfProgramType.SOCKET_CONNECT: BPF_SECURITY_SOCKET_CONNECT,
    BpfProgramType.SYS_WRITE: BPF_SYS_WRITE,
    BpfProgramType.SYS_READ: BPF_SYS_READ,
    # SYS_EXIT_READ contains both sys_enter_read (stash) and sys_exit_read (capture)
    # as a single BPF object so they share the read_ctx_map.
    BpfProgramType.SYS_EXIT_READ: BPF_SYS_EXIT_READ,
}

# Minimum kernel version required for each program type.
MIN_KERNEL_VERSIONS: dict[BpfProgramType, tuple[int, int]] = {
    BpfProgramType.TCP_SENDMSG: (4, 14),
    BpfProgramType.TCP_RECVMSG: (4, 14),
    BpfProgramType.SOCKET_CONNECT: (4, 14),
    BpfProgramType.SYS_WRITE: (4, 17),
    BpfProgramType.SYS_READ: (4, 17),
    # BPF_PERCPU_HASH + sys_exit_read tracepoint require 4.17+
    BpfProgramType.SYS_EXIT_READ: (4, 17),
}
