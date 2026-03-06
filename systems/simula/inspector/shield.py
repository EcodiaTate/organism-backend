"""
EcodiaOS — Autonomous Shield (XDP Filter Synthesizer)
"""

from __future__ import annotations

import asyncio
import ctypes
import json
import re
import socket
import struct
import time
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from clients.llm import LLMProvider

logger = structlog.get_logger().bind(system="simula.inspector.shield")

_XDP_SYNTHESIZER_SYSTEM_PROMPT = """\
You are an expert Linux Kernel and eBPF/XDP Engineer.

Task: I will provide an anomaly report containing violated system invariants, \
state-machine deviations, and the Z3 proof of the malicious input.

Action: Write a highly optimized XDP program in C (compatible with BCC). \
First, check the source IP against the `xdp_blocklist` map and enforce the TTL. \
Next, inspect the TCP/HTTP payload. If it matches the malicious signature from the report, \
return XDP_DROP. Otherwise, return XDP_PASS.

eBPF VERIFIER COMPLIANCE (CRITICAL — DO NOT VIOLATE):
- DO NOT use dynamic loop boundaries. Loops MUST have a strict, hardcoded maximum integer boundary (e.g., `for (int i = 0; i < 64; i++)`).
- You MUST place `#pragma unroll` immediately preceding any loop to force compiler unrolling.
- Inside the loop, you MUST rigorously check bounds against `data_end` before dereferencing any packet bytes (e.g., `if ((void*)payload + i + 1 > data_end) break;`).
- Keep string matching extremely simple, bounded, and avoid complex pointer arithmetic.
- Never use while loops with non-constant bounds. Use only bounded for loops with #pragma unroll.
- Verify every pointer dereference with explicit data_end bounds checks in the loop body.
- NEVER array-index or dereference a `void *`. Always declare packet data pointers as `unsigned char *` or `char *` before checking bytes (e.g., `unsigned char *payload = (unsigned char *)(tcp + 1)`).

Core Constraints:
- You MUST use this exact structure. Do not deviate.
- Do NOT use the SEC("xdp") macro. BCC handles section attachment automatically.
- Perform strict bounds checking on Ethernet, IP, and TCP headers before accessing payload data.
- Include necessary headers: <uapi/linux/bpf.h>, <linux/if_ether.h>, <linux/ip.h>, <linux/in.h>, <linux/tcp.h>.
- The main function MUST be: int xdp_filter(struct xdp_md *ctx)
- Output ONLY raw C code. No markdown fences, no explanation.
- Blocklist Map: You MUST define: BPF_HASH(xdp_blocklist, u32, struct block_info, 100000);
- Blocklist Struct: You MUST define: struct block_info { u64 expiry_ts; u32 anomaly_report_id; };
- Blocklist TTL: Use bpf_ktime_get_ns() to check expiry. If expired, call xdp_blocklist.delete(&src_ip) and return XDP_PASS. If still valid, emit telemetry and return XDP_DROP.
- Telemetry: You MUST declare a perf event output array using: BPF_PERF_OUTPUT(events);
- Telemetry: Define struct alert_event_t { u32 src_ip; u32 vuln_id; u32 anomaly_report_id; };
- Telemetry: For a blocklist hit, set vuln_id = 1 and copy anomaly_report_id from the map value before submitting.
- Telemetry: For a signature match, set vuln_id = 2 and anomaly_report_id = 0 before submitting.
- Telemetry: Submit via: events.perf_submit(ctx, &event, sizeof(event));

Example template:
#include <uapi/linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/in.h>
#include <linux/tcp.h>

BPF_PERF_OUTPUT(events);

struct block_info {
    u64 expiry_ts;
    u32 anomaly_report_id;
};

BPF_HASH(xdp_blocklist, u32, struct block_info, 100000);

struct alert_event_t {
    u32 src_ip;
    u32 vuln_id;
    u32 anomaly_report_id;
};

int xdp_filter(struct xdp_md *ctx) {
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;

    struct ethhdr *eth = data;
    if ((void*)(eth + 1) > data_end) return XDP_PASS;
    if (eth->h_proto != htons(ETH_P_IP)) return XDP_PASS;

    struct iphdr *ip = (void*)(eth + 1);
    if ((void*)(ip + 1) > data_end) return XDP_PASS;

    u32 src_ip = ip->saddr;

    // 1. Audited Blocklist Check with TTL enforcement
    struct block_info *info = xdp_blocklist.lookup(&src_ip);
    if (info) {
        u64 now = bpf_ktime_get_ns();
        if (now >= info->expiry_ts) {
            xdp_blocklist.delete(&src_ip);
            return XDP_PASS;
        }
        struct alert_event_t event = {};
        event.src_ip = src_ip;
        event.vuln_id = 1; // 1 = Blocklist hit
        event.anomaly_report_id = info->anomaly_report_id;
        events.perf_submit(ctx, &event, sizeof(event));
        return XDP_DROP;
    }

    // 2. Zero-Day Payload Inspection
    // EXAMPLE: Bounded HTTP string matching with verifier-safe loop
    if (ip->protocol == IPPROTO_TCP) {
        struct tcphdr *tcp = (void*)(ip + 1);
        if ((void*)(tcp + 1) > data_end) return XDP_PASS;

        unsigned char *payload = (unsigned char *)(tcp + 1);

        // Bounded, unrolled loop for signature matching (max 64 bytes)
        #pragma unroll
        for (int i = 0; i < 64; i++) {
            // CRITICAL: Verify bounds before every dereference
            if ((void*)(payload + i + 1) > data_end) break;

            // Match simple byte patterns (example: check for "GET " at offset 0)
            if (i == 0) {
                if (payload[0] == 'G' && payload[1] == 'E' &&
                    payload[2] == 'T' && payload[3] == ' ') {
                    // Potential HTTP GET request
                    // Further inspection would go here
                    break;
                }
            }
        }
    }

    return XDP_PASS;
}
"""

_CODE_FENCE_RE = re.compile(r"^```(?:c|C)?\s*\n?", re.MULTILINE)
_CODE_FENCE_END_RE = re.compile(r"\n?```\s*$", re.MULTILINE)


class AutonomousShield:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self._log = logger

    async def synthesize_xdp_filter(self, alert_context: dict[str, object]) -> str:
        from bcc import BPF  # type: ignore[import-untyped]

        from clients.llm import Message

        context_str = json.dumps(alert_context, indent=2, default=str)

        messages = [
            Message(
                role="user",
                content=f"Anomaly Report:\n\n{context_str}\n\nGenerate the XDP filter program now."
            )
        ]

        # The Self-Healing Compiler Loop (Max 3 attempts)
        for attempt in range(3):
            self._log.info("xdp_synthesis_started", attempt=attempt)

            response = await self._llm.generate(
                system_prompt=_XDP_SYNTHESIZER_SYSTEM_PROMPT,
                messages=messages,
                max_tokens=4000,
                temperature=0.2,
            )

            raw_code = response.text.strip()
            raw_code = _CODE_FENCE_RE.sub("", raw_code)
            raw_code = _CODE_FENCE_END_RE.sub("", raw_code)
            raw_code = raw_code.strip()

            try:
                # Syntax Check compilation
                BPF(text=raw_code)
                self._log.info("xdp_synthesis_complete_and_verified", code_size=len(raw_code))
                return raw_code  # Success!

            except Exception as e:
                error_msg = str(e)
                self._log.warning("xdp_compilation_failed_retrying", error=error_msg)

                # Append the assistant's failed code, then our error message
                messages.append(Message(role="assistant", content=raw_code))
                messages.append(Message(
                    role="user",
                    content=f"The eBPF compiler threw this error:\n{error_msg}\n\nPlease rewrite the full C-code to fix this syntax error. Output ONLY the raw, corrected C-code."
                ))

        raise RuntimeError("LLM failed to generate compiling XDP code after 3 attempts.")

    def deploy_filter_live(self, interface: str, bpf_c_code: str):
        from bcc import BPF  # type: ignore
        self._log.info("xdp_live_attachment_started", interface=interface)

        bpf = BPF(text=bpf_c_code)
        fn = bpf.load_func("xdp_filter", BPF.XDP)
        bpf.attach_xdp(interface, fn, 0)

        blocklist_map = bpf["xdp_blocklist"]

        self._log.info("xdp_live_attachment_success", interface=interface)
        return bpf, blocklist_map

    def enforce_ip_block(
        self,
        bpf_module,
        ip_string: str,
        anomaly_id: int,
        duration_seconds: int,
    ) -> None:
        """Inject a time-bounded, audited block entry into the kernel XDP blocklist.

        Every entry is tied to a specific anomaly_id so the reason for each block
        is always recoverable from telemetry. The kernel program enforces the TTL
        itself via bpf_ktime_get_ns(), so entries cannot persist beyond their window
        even if user-space crashes.
        """
        # Pack the IP into network-byte-order u32 to match ip->saddr in the kernel.
        ip_packed = socket.inet_aton(ip_string)
        ip_key = ctypes.c_uint32(struct.unpack("I", ip_packed)[0])

        # bpf_ktime_get_ns() measures from boot, not epoch.  time.clock_gettime
        # with CLOCK_BOOTTIME gives the same reference in user-space.
        now_ns = time.clock_gettime_ns(time.CLOCK_BOOTTIME)  # type: ignore[attr-defined]
        expiry_ns = now_ns + duration_seconds * 1_000_000_000

        class BlockInfo(ctypes.Structure):
            _fields_ = [
                ("expiry_ts", ctypes.c_uint64),
                ("anomaly_report_id", ctypes.c_uint32),
            ]

        block_value = BlockInfo(expiry_ts=expiry_ns, anomaly_report_id=anomaly_id)

        bpf_module["xdp_blocklist"][ip_key] = block_value

        self._log.info(
            "ip_block_enforced",
            ip=ip_string,
            anomaly_id=anomaly_id,
            duration_seconds=duration_seconds,
            expiry_ns=expiry_ns,
        )

    def listen_for_telemetry(self, bpf_module):
        self._log.info("Initializing kernel telemetry listener...")

        class AlertEvent(ctypes.Structure):
            _fields_ = [
                ("src_ip", ctypes.c_uint32),
                ("vuln_id", ctypes.c_uint32),
                ("anomaly_report_id", ctypes.c_uint32),
            ]

        def print_event(cpu, data, size):
            event = ctypes.cast(data, ctypes.POINTER(AlertEvent)).contents
            ip_str = socket.inet_ntoa(struct.pack("<I", event.src_ip))
            rule = "Blocklist" if event.vuln_id == 1 else "Signature"

            print("\n" + "="*60)
            print("[!!!] ECODIAOS SHIELD ALERT: EXPLOIT DESTROYED [!!!]")
            print(f"[*] Attacker IP    : {ip_str}")
            print(f"[*] Rule           : {rule} (vuln_id={event.vuln_id})")
            print(f"[*] Anomaly Report : {event.anomaly_report_id or 'N/A'}")
            print("="*60 + "\n")

        bpf_module["events"].open_perf_buffer(print_event)
        print("[*] Telemetry pipeline open. Waiting for live network attacks...")

        try:
            while True:
                bpf_module.perf_buffer_poll()
        except KeyboardInterrupt:
            print("\n[*] Telemetry listener detached.")

    async def stream_telemetry(
        self,
        bpf_module,
        *,
        perf_map_name: str = "filter_events",
        poll_timeout_ms: int = 100,
    ) -> AsyncIterator[dict[str, object]]:
        """Yield telemetry dicts without blocking the asyncio event loop.

        The BCC ``perf_buffer_poll()`` call is inherently blocking, so we
        run each poll iteration inside ``asyncio.to_thread``.  A short
        *poll_timeout_ms* (default 100 ms) keeps latency low while still
        giving the event loop regular opportunities to process other work
        (e.g. SSE keep-alives, client disconnects).

        The generator is designed to be merged into a FastAPI SSE stream::

            async for evt in shield.stream_telemetry(bpf):
                yield _cc_sse("telemetry", evt)

        Yields:
            ``{"type": "telemetry", "src_ip": "1.2.3.4",
              "rule": "Blocklist"|"Signature", "vuln_id": int}``
        """

        class AlertEvent(ctypes.Structure):
            _fields_ = [
                ("src_ip", ctypes.c_uint32),
                ("vuln_id", ctypes.c_uint32),
                # Third field is anomaly_report_id (shield template) or
                # rule_index (filter_generator template) — same offset & size.
                ("rule_index", ctypes.c_uint32),
            ]

        event_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        stop_event = asyncio.Event()
        # Capture the running loop now (on the main thread) so the
        # perf-buffer callback can schedule work on it from the poll thread.
        loop = asyncio.get_running_loop()

        def _on_event(_cpu: int, data, size: int) -> None:
            """Perf-buffer callback — runs on the poll thread."""
            ev = ctypes.cast(data, ctypes.POINTER(AlertEvent)).contents
            ip_str = socket.inet_ntoa(struct.pack("<I", ev.src_ip))
            rule = "Blocklist" if ev.vuln_id == 1 else "Signature"
            # asyncio.Queue is not thread-safe for put(); use
            # call_soon_threadsafe so the item lands on the loop safely.
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                {
                    "type": "telemetry",
                    "src_ip": ip_str,
                    "rule": rule,
                    "vuln_id": ev.vuln_id,
                    "rule_index": ev.rule_index,
                },
            )

        bpf_module[perf_map_name].open_perf_buffer(_on_event)
        self._log.info(
            "stream_telemetry_started",
            perf_map=perf_map_name,
            poll_timeout_ms=poll_timeout_ms,
        )

        def _blocking_poll() -> None:
            """Single blocking poll with a bounded timeout."""
            bpf_module.perf_buffer_poll(timeout=poll_timeout_ms)

        try:
            while not stop_event.is_set():
                # Offload the blocking poll to a thread so the event loop
                # stays responsive.
                await asyncio.to_thread(_blocking_poll)

                # Drain all events that arrived during this poll cycle.
                while not event_queue.empty():
                    yield (await event_queue.get())
        except asyncio.CancelledError:
            self._log.info("stream_telemetry_cancelled")
        finally:
            stop_event.set()
            self._log.info("stream_telemetry_stopped")
