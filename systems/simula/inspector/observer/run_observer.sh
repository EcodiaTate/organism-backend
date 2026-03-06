#!/usr/bin/env bash
# Simula Observer — Container Entrypoint
#
# Selects between bpftrace (quick validation) and BCC Python (full observer)
# based on the OBSERVER_MODE environment variable.
#
# Environment:
#   OBSERVER_MODE   — "bcc" (default) or "bpftrace"
#   OBSERVER_PORT   — Health endpoint port (default: 9472, BCC mode only)
#   OBSERVER_PROBES — Comma-separated probe names (default: all three network probes)

set -euo pipefail

MODE="${OBSERVER_MODE:-bcc}"
PORT="${OBSERVER_PORT:-9472}"
PROBES="${OBSERVER_PROBES:-tcp_sendmsg,tcp_recvmsg,security_socket_connect,sys_exit_read}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[observer] mode=$MODE port=$PORT probes=$PROBES"
echo "[observer] kernel=$(uname -r)"

case "$MODE" in
  bpftrace)
    echo "[observer] Starting bpftrace quick-validation mode"
    exec bpftrace "$SCRIPT_DIR/bpftrace_probes.bt"
    ;;
  bcc)
    echo "[observer] Starting BCC Python observer"
    exec python3 "$SCRIPT_DIR/observer.py" \
      --port "$PORT" \
      --probes "$PROBES"
    ;;
  *)
    echo "[observer] ERROR: Unknown mode '$MODE' (expected 'bcc' or 'bpftrace')"
    exit 1
    ;;
esac
