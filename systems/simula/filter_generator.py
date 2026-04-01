"""
EcodiaOS - XDP Filter Generator (Deterministic C Template Engine)

Accepts the ``edge_case_input`` JSON extracted by the Inspector prover
(the ``boundary_test_evidence.details.edge_case_input`` list/dict) and emits a
verifier-compliant XDP/C program that drops packets matching those exact
invalid byte patterns.

The generated C code is guaranteed to satisfy the Linux kernel verifier:
  - All loops use ``#pragma unroll`` with hardcoded integer bounds.
  - Packet data pointers are ``unsigned char *`` (never ``void *``).
  - Every single pointer dereference is preceded by a ``data_end`` guard.
  - No dynamic allocations, no unbounded reads, no helper calls that
    require CAP_PERFMON.

Usage (standalone)::

    python -m systems.simula.filter_generator \\
        --json '{"headers": {"X-Evil": "pwned"}, "body": {"payload": "rm -rf /"}}'

    # → writes /tmp/generated_filter.c

Usage (programmatic)::

    from systems.simula.filter_generator import generate_xdp_filter
    c_code = generate_xdp_filter(edge_case_input)
    Path("/tmp/generated_filter.c").write_text(c_code)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Final

import structlog

logger = structlog.get_logger().bind(module="simula.filter_generator")

# ── Constants ────────────────────────────────────────────────────────────────

# Maximum number of bytes we scan into the TCP payload.  The kernel verifier
# needs a compile-time constant here; 256 is generous for HTTP header/body
# prefix matching while staying well under the XDP stack limit.
MAX_SCAN_DEPTH: Final[int] = 256

# Maximum length of any single match pattern injected into the C template.
# Patterns longer than this are truncated - the verifier rejects programs
# with loops that iterate more than ~512 times even with #pragma unroll.
MAX_PATTERN_LEN: Final[int] = 128

# Default output path.
DEFAULT_OUTPUT_PATH: Final[str] = "/tmp/generated_filter.c"


# ── C Template Fragments ────────────────────────────────────────────────────

_C_PREAMBLE: Final[str] = """\
/*
 * Auto-generated XDP filter - EcodiaOS Inspector boundary-test shield.
 *
 * Drops inbound TCP packets whose HTTP payload matches byte patterns
 * extracted from proven edge-case inputs.  All loops are bounded and
 * unrolled; every dereference is guarded by data_end.
 *
 * DO NOT EDIT - regenerate via filter_generator.py.
 */

#include <uapi/linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/in.h>
#include <linux/tcp.h>

BPF_PERF_OUTPUT(filter_events);

struct block_info {
    u64 expiry_ts;
    u32 anomaly_report_id;
};

BPF_HASH(xdp_blocklist, u32, struct block_info, 100000);

struct alert_event_t {
    u32 src_ip;
    u32 vuln_id;
    u32 rule_index;
};
"""

_C_FUNC_OPEN: Final[str] = """\
int xdp_filter(struct xdp_md *ctx) {
    unsigned char *data_end = (unsigned char *)(long)ctx->data_end;
    unsigned char *data     = (unsigned char *)(long)ctx->data;

    /* ── Ethernet header ─────────────────────────────────────── */
    struct ethhdr *eth = (struct ethhdr *)data;
    if ((unsigned char *)(eth + 1) > data_end)
        return XDP_PASS;
    if (eth->h_proto != htons(ETH_P_IP))
        return XDP_PASS;

    /* ── IP header ───────────────────────────────────────────── */
    struct iphdr *ip = (struct iphdr *)((unsigned char *)eth + sizeof(*eth));
    if ((unsigned char *)(ip + 1) > data_end)
        return XDP_PASS;

    u32 src_ip = ip->saddr;

    /* ── Blocklist check with TTL ────────────────────────────── */
    struct block_info *info = xdp_blocklist.lookup(&src_ip);
    if (info) {
        u64 now = bpf_ktime_get_ns();
        if (now >= info->expiry_ts) {
            xdp_blocklist.delete(&src_ip);
        } else {
            struct alert_event_t event = {};
            event.src_ip   = src_ip;
            event.vuln_id  = 1;
            event.rule_index = info->anomaly_report_id;
            filter_events.perf_submit(ctx, &event, sizeof(event));
            return XDP_DROP;
        }
    }

    /* ── TCP only ────────────────────────────────────────────── */
    if (ip->protocol != IPPROTO_TCP)
        return XDP_PASS;

    /* Variable-length IP header (IHL field) */
    unsigned int ip_hdr_len = ip->ihl * 4;
    if (ip_hdr_len < 20)
        return XDP_PASS;

    struct tcphdr *tcp = (struct tcphdr *)((unsigned char *)ip + ip_hdr_len);
    if ((unsigned char *)(tcp + 1) > data_end)
        return XDP_PASS;

    unsigned int tcp_hdr_len = tcp->doff * 4;
    if (tcp_hdr_len < 20)
        return XDP_PASS;

    unsigned char *payload = (unsigned char *)tcp + tcp_hdr_len;
    if (payload >= data_end)
        return XDP_PASS;

"""

_C_FUNC_CLOSE: Final[str] = """\
    return XDP_PASS;
}
"""


# ── Pattern → C code compilation ────────────────────────────────────────────


def _bytes_for_pattern(pattern: str) -> list[int]:
    """Convert a string pattern into a list of byte values (UTF-8)."""
    raw = pattern.encode("utf-8")
    if len(raw) > MAX_PATTERN_LEN:
        raw = raw[:MAX_PATTERN_LEN]
    return list(raw)


def _emit_match_block(
    rule_index: int,
    label: str,
    pattern_bytes: list[int],
) -> str:
    """Emit a single verifier-safe byte-match block.

    Strategy: scan the payload with a bounded, unrolled outer loop looking
    for the first byte of the pattern.  When found, check the remaining
    bytes with an inner bounded/unrolled loop.  If the full pattern is
    confirmed, emit telemetry and return XDP_DROP.
    """
    if not pattern_bytes:
        return f"    /* rule {rule_index} ({label}): empty pattern - skipped */\n"

    plen = len(pattern_bytes)
    first_byte = pattern_bytes[0]

    # The outer scan window: we can start a match at any offset where
    # the full pattern still fits within MAX_SCAN_DEPTH.
    scan_limit = MAX_SCAN_DEPTH - plen + 1
    if scan_limit < 1:
        scan_limit = 1

    lines: list[str] = []
    lines.append(f"    /* ── Rule {rule_index}: {label} ({plen} bytes) ── */")
    lines.append("    {")

    if plen == 1:
        # Single-byte pattern: just scan for that byte.
        lines.append("        #pragma unroll")
        lines.append(f"        for (int i = 0; i < {scan_limit}; i++) {{")
        lines.append("            if (payload + i + 1 > data_end)")
        lines.append("                break;")
        lines.append(f"            if (payload[i] == 0x{first_byte:02x}) {{")
        lines.append("                struct alert_event_t event = {};")
        lines.append("                event.src_ip     = src_ip;")
        lines.append("                event.vuln_id    = 2;")
        lines.append(f"                event.rule_index = {rule_index};")
        lines.append("                filter_events.perf_submit(ctx, &event, sizeof(event));")
        lines.append("                return XDP_DROP;")
        lines.append("            }")
        lines.append("        }")
    else:
        # Multi-byte: outer scan for first byte, inner verify for rest.
        lines.append(f"        int matched_{rule_index} = 0;")
        lines.append("        #pragma unroll")
        lines.append(f"        for (int i = 0; i < {scan_limit}; i++) {{")
        lines.append("            if (payload + i + 1 > data_end)")
        lines.append("                break;")
        lines.append(f"            if (matched_{rule_index})")
        lines.append("                break;")
        lines.append(f"            if (payload[i] != 0x{first_byte:02x})")
        lines.append("                continue;")
        lines.append("")
        lines.append(f"            /* First byte matched at offset i - verify remaining {plen - 1} bytes */")
        lines.append(f"            int ok_{rule_index} = 1;")

        # Inner loop: bounded to the exact remaining pattern length.
        lines.append("            #pragma unroll")
        lines.append(f"            for (int j = 1; j < {plen}; j++) {{")
        lines.append("                if (payload + i + j + 1 > data_end) {")
        lines.append(f"                    ok_{rule_index} = 0;")
        lines.append("                    break;")
        lines.append("                }")

        # Emit a chained byte comparison.  Each case is a constant
        # index - the verifier handles this well since the loop bound
        # and every branch target are compile-time deterministic.
        for byte_idx in range(1, plen):
            lines.append(
                f"                if (j == {byte_idx} && payload[i + j] != 0x{pattern_bytes[byte_idx]:02x})"
            )
            lines.append(f"                    ok_{rule_index} = 0;")

        lines.append("            }")
        lines.append("")
        lines.append(f"            if (ok_{rule_index}) {{")
        lines.append(f"                matched_{rule_index} = 1;")
        lines.append("                struct alert_event_t event = {};")
        lines.append("                event.src_ip     = src_ip;")
        lines.append("                event.vuln_id    = 2;")
        lines.append(f"                event.rule_index = {rule_index};")
        lines.append("                filter_events.perf_submit(ctx, &event, sizeof(event));")
        lines.append("                return XDP_DROP;")
        lines.append("            }")
        lines.append("        }")

    lines.append("    }")
    lines.append("")
    return "\n".join(lines) + "\n"


# ── Public API ──────────────────────────────────────────────────────────────


def _extract_rules_from_step(step: dict[str, Any], step_idx: int = 0) -> list[tuple[str, str]]:
    """Flatten a single step dict into ``(label, pattern_string)`` pairs.

    Each step dict may contain ``headers``, ``body``, ``query``, and ``flags``
    sub-dicts.  Boolean flags are converted to ``"true"``/``"false"`` strings.
    """
    rules: list[tuple[str, str]] = []

    for category in ("headers", "body", "query", "flags"):
        section = step.get(category)
        if not isinstance(section, dict):
            continue
        for field_name, value in section.items():
            if isinstance(value, bool):
                pattern = f'"{field_name}": {str(value).lower()}'
            elif isinstance(value, (int, float)):
                pattern = f'"{field_name}": {value}'
            elif isinstance(value, str):
                pattern = f"{field_name}: {value}" if category == "headers" else str(value)
            else:
                pattern = str(value)

            if pattern:
                rules.append((f"step{step_idx}/{category}/{field_name}", pattern))

    # Also accept top-level string patterns (simple mode).
    if not rules:
        for key, value in step.items():
            if key == "step":
                continue
            if isinstance(value, str) and value:
                rules.append((f"step{step_idx}/{key}", value))

    return rules


def _extract_rules(edge_case_input: dict[str, Any] | list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Flatten the edge_case_input into ``(label, pattern_string)`` pairs.

    Handles both the new multi-step list format (state-machine prover)::

        [
          {"step": 0, "headers": {...}, "body": {...}},
          {"step": 1, "body": {...}, "flags": {...}},
          {"step": 2, "body": {...}}
        ]

    and the legacy single-step dict format::

        {
          "headers": {"X-Evil": "value", ...},
          "body":    {"field": "value", ...},
          "query":   {"param": "value", ...},
          "flags":   {"is_admin": true, ...}
        }

    Boolean flags are converted to the strings ``"true"``/``"false"`` so they
    can still be byte-matched in the raw HTTP payload (e.g., a JSON body
    containing ``"is_admin": true``).
    """
    # Multi-step list format
    if isinstance(edge_case_input, list):
        rules: list[tuple[str, str]] = []
        for step in edge_case_input:
            if not isinstance(step, dict):
                continue
            step_idx = step.get("step", 0)
            rules.extend(_extract_rules_from_step(step, step_idx))
        return rules

    # Legacy single-step dict format
    return _extract_rules_from_step(edge_case_input, step_idx=0)


def generate_xdp_filter(edge_case_input: dict[str, Any] | list[dict[str, Any]]) -> str:
    """Generate a complete, verifier-compliant XDP/C program.

    Args:
        edge_case_input: The ``edge_case_input`` from the Inspector
            prover's boundary_test_evidence output. Accepts either the
            new multi-step list format (one dict per exploit step) or the
            legacy single-step dict with ``headers``/``body``/``query``/
            ``flags`` sub-dicts.

    Returns:
        A complete C source string ready for compilation with BCC or
        ``clang -target bpf``.

    Raises:
        ValueError: If no matchable patterns could be extracted.
    """
    rules = _extract_rules(edge_case_input)
    if not rules:
        raise ValueError(
            "No matchable patterns found in edge_case_input - "
            "expected at least one non-empty string value in headers/body/query/flags."
        )

    logger.info(
        "xdp_filter_generation_started",
        rule_count=len(rules),
        labels=[r[0] for r in rules],
    )

    parts: list[str] = [_C_PREAMBLE, _C_FUNC_OPEN]

    for idx, (label, pattern) in enumerate(rules):
        pattern_bytes = _bytes_for_pattern(pattern)
        parts.append(_emit_match_block(idx, label, pattern_bytes))

    parts.append(_C_FUNC_CLOSE)

    code = "".join(parts)

    logger.info(
        "xdp_filter_generation_complete",
        rule_count=len(rules),
        code_size=len(code),
    )
    return code


def generate_and_write(
    edge_case_input: dict[str, Any] | list[dict[str, Any]],
    output_path: str = DEFAULT_OUTPUT_PATH,
) -> pathlib.Path:
    """Generate the XDP filter and write it to disk.

    Args:
        edge_case_input: See :func:`generate_xdp_filter`.
        output_path: Filesystem path for the generated ``.c`` file.

    Returns:
        The resolved ``Path`` of the written file.
    """
    code = generate_xdp_filter(edge_case_input)
    out = pathlib.Path(output_path)
    out.write_text(code, encoding="utf-8")
    logger.info("xdp_filter_written", path=str(out), size=len(code))
    return out


# ── CLI ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a verifier-compliant XDP/C filter from Inspector "
            "edge_case_input JSON."
        ),
    )
    parser.add_argument(
        "--json",
        required=True,
        help=(
            "JSON string or @filepath containing the edge_case_input dict "
            "(e.g., '{\"headers\": {\"X-Evil\": \"pwned\"}}' or @evidence.json)"
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path for the generated .c file (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    raw = args.json
    if raw.startswith("@"):
        raw = pathlib.Path(raw[1:]).read_text(encoding="utf-8")

    try:
        edge_case_input = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON: {exc}", file=sys.stderr)
        return 1

    if not isinstance(edge_case_input, (dict, list)):
        print("[ERROR] edge_case_input must be a JSON object or array.", file=sys.stderr)
        return 1

    try:
        out = generate_and_write(edge_case_input, args.output)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[OK] Generated XDP filter → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
