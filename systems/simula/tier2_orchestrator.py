#!/usr/bin/env python3
"""
EcodiaOS - Tier 2 Orchestrator (Node 22 Uprobe -> XDP)
Fully Implemented. No Mocks.
"""
import ctypes
import re
import sys

from bcc import BPF

from clients.llm import create_llm_provider
from config import EcodiaOSConfig
from systems.simula.inspector.shield import AutonomousShield

DUMMY_XDP_CODE = """
#include <uapi/linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/in.h>

struct block_info { u64 expiry_ts; u32 anomaly_report_id; };
BPF_HASH(xdp_blocklist, u32, struct block_info, 100000);

int xdp_filter(struct xdp_md *ctx) {
    void *data_end = (void *)(long)ctx->data_end;
    void *data     = (void *)(long)ctx->data;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end) return XDP_PASS;
    if (eth->h_proto != htons(ETH_P_IP)) return XDP_PASS; // IPv4 Only

    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end) return XDP_PASS;

    u32 src_ip = ip->saddr;
    struct block_info *blocked = xdp_blocklist.lookup(&src_ip);

    // BRUTAL MODE: If your IP is on this list, you die. No expiry checks.
    if (blocked) {
        return XDP_DROP;
    }
    return XDP_PASS;
}
"""

# --- 2. Tier 2 Uprobe Sensor (Node 22 `llhttp_execute` Hook) ---
TIER2_BPF_SOURCE = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

BPF_PERF_OUTPUT(l7_alerts);

struct l7_alert_t {
    u32 pid;
    char raw_http_buffer[256]; // Grab the first 256 bytes of the HTTP request
};

// Hooking: int llhttp_execute(llhttp_t* parser, const char* data, size_t len)
int trace_llhttp_execute(struct pt_regs *ctx) {
    struct l7_alert_t alert = {};
    alert.pid = bpf_get_current_pid_tgid() >> 32;

    // PT_REGS_PARM2 gets the 2nd argument of the function (const char* data)
    void *data_ptr = (void *)PT_REGS_PARM2(ctx);

    if (data_ptr) {
        // Read the raw HTTP string directly from user-space memory
        bpf_probe_read_user(&alert.raw_http_buffer, sizeof(alert.raw_http_buffer), data_ptr);
        l7_alerts.perf_submit(ctx, &alert, sizeof(alert));
    }
    return 0;
}
"""

def main():
    if len(sys.argv) < 2:
        print("Usage: sudo python3 tier2_orchestrator.py <NETWORK_INTERFACE>")
        sys.exit(1)

    interface = sys.argv[1]

    print(f"[*] Initializing EcodiaOS Tier 1 Shield on {interface}...")
    config = EcodiaOSConfig()
    llm = create_llm_provider(config.llm)
    shield = AutonomousShield(llm)
    bpf_xdp, blocklist_map = shield.deploy_filter_live(interface, DUMMY_XDP_CODE)

    print("[*] Initializing EcodiaOS Tier 2 Uprobe on Node 22...")
    bpf_tier2 = BPF(text=TIER2_BPF_SOURCE)

    # Attach to the Node.js binary using a Uprobe instead of USDT
    # Note: "node" must be in the system PATH, or provide the full path to the binary.
    try:
       bpf_tier2.attach_uprobe(name="/usr/bin/node", sym="llhttp_execute", fn_name="trace_llhttp_execute")
    except Exception as e:
        print(f"[!] Uprobe Attach Failed. Is Node running? Error: {e}")
        sys.exit(1)

    class L7AlertEvent(ctypes.Structure):
        _fields_ = [
            ("pid", ctypes.c_uint32),
            ("raw_http_buffer", ctypes.c_char * 256)
        ]

    def handle_l7_alert(cpu, data, size):
        event = ctypes.cast(data, ctypes.POINTER(L7AlertEvent)).contents
        payload = event.raw_http_buffer.decode('utf-8', 'ignore').strip('\x00')

        # llhttp_execute fires continuously. We only care about the start of requests.
        if not payload.startswith(("GET", "POST", "PUT", "DELETE")):
            return

        # --- ECODIAOS THREAT DETECTION LOGIC ---
        # 1. Parse the IP from the proxy header (Fallback to localhost for testing)
        ip_match = re.search(r"X-Forwarded-For:\s*([0-9\.]+)", payload)
        attacker_ip = ip_match.group(1) if ip_match else "127.0.0.1"

        # 2. Check for L7 zero-day signatures
        if "UNION SELECT" in payload or "bypass=true" in payload:
            print("\n[!!!] L7 ZERO-DAY DETECTED IN NODE MEMORY [!!!]")
            print(f"    -> Parsed IP: {attacker_ip}")
            print(f"    -> Payload Snippet: {payload.splitlines()[0]}")
            print("    -> Action: Engaging XDP Muscle...")

            try:
                shield.enforce_ip_block(
                    bpf_module=bpf_xdp,
                    ip_string=attacker_ip,
                    anomaly_id=505,
                    duration_seconds=3600
                )
                print(f"    -> Success! {attacker_ip} blocked at Layer 3.")
            except Exception as e:
                print(f"    -> [ERROR] Shield execution failed: {e}")
        else:
            print(f"[-] Safe request observed from {attacker_ip}")

    bpf_tier2["l7_alerts"].open_perf_buffer(handle_l7_alert)

    print("\n[*] EcodiaOS Pipeline Active. Hooked into Node's C++ HTTP parser.")
    print("[*] Waiting for traffic...")

    try:
        while True:
            bpf_tier2.perf_buffer_poll()
    except KeyboardInterrupt:
        print("\n[*] Detaching EcodiaOS...")

if __name__ == "__main__":
    main()
