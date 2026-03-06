#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Simula Inspector — Taint Propagation Testbed Runner
#
# Boots the 3-service chain (A → B → C), waits for Service A to
# complete, then checks Service C's logs for taint token propagation.
#
# Exit codes:
#   0 — taint propagated A → B → C (PASS)
#   1 — taint NOT found in C's logs (FAIL)
#   2 — infrastructure error (docker not running, build failed, etc.)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE_FILE="docker-compose.yml"
TAINT_PATTERN='TAINT=[0-9a-f\-]{36};'

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${YELLOW}[testbed]${NC} $*"; }
pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }

# ── Preflight ──────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    fail "docker not found in PATH"
    exit 2
fi

if ! docker info &>/dev/null; then
    fail "Docker daemon not running"
    exit 2
fi

# ── Cleanup trap ───────────────────────────────────────────────────
cleanup() {
    log "Tearing down testbed..."
    docker compose -f "$COMPOSE_FILE" down -v --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT

# ── Build & Boot ───────────────────────────────────────────────────
log "Building images..."
if ! docker compose -f "$COMPOSE_FILE" build --quiet; then
    fail "Docker build failed"
    exit 2
fi

log "Starting services C and B..."
docker compose -f "$COMPOSE_FILE" up -d service-c service-b

log "Waiting for B to be healthy..."
TRIES=0
MAX_TRIES=30
while [ $TRIES -lt $MAX_TRIES ]; do
    STATUS=$(docker compose -f "$COMPOSE_FILE" ps --format json service-b 2>/dev/null | python3 -c "
import sys, json
for line in sys.stdin:
    data = json.loads(line.strip())
    print(data.get('Health', data.get('Status', 'unknown')))
    break
" 2>/dev/null || echo "unknown")

    if echo "$STATUS" | grep -qi "healthy"; then
        break
    fi
    TRIES=$((TRIES + 1))
    sleep 1
done

if [ $TRIES -ge $MAX_TRIES ]; then
    fail "Service B did not become healthy within ${MAX_TRIES}s"
    docker compose -f "$COMPOSE_FILE" logs
    exit 2
fi

log "B is healthy. Launching A (attacker)..."

# ── Run Service A ──────────────────────────────────────────────────
# A runs, sends 3 tainted requests, then exits.
docker compose -f "$COMPOSE_FILE" up service-a

# ── Collect & Verify ───────────────────────────────────────────────
log ""
log "═══════════════════════════════════════════"
log "  Checking Service C logs for taint tokens"
log "═══════════════════════════════════════════"
log ""

C_LOGS=$(docker compose -f "$COMPOSE_FILE" logs service-c 2>&1)

echo "$C_LOGS"
echo ""

# Count taint tokens detected by C
TAINT_COUNT=$(echo "$C_LOGS" | grep -cE "TAINT DETECTED" || true)

if [ "$TAINT_COUNT" -gt 0 ]; then
    pass "Taint propagated A → B → C  ($TAINT_COUNT taint tokens detected by C)"
    log ""

    # Also show B's logs for the full chain
    log "Service B forwarding log:"
    docker compose -f "$COMPOSE_FILE" logs service-b 2>&1 | grep -E "Forwarding|Received" || true
    log ""

    # Verify via C's query log endpoint (if still running)
    QUERY_LOG=$(curl -s http://127.0.0.1:8080/health 2>/dev/null || echo "")
    exit 0
else
    fail "No taint tokens detected by Service C"
    log ""
    log "Full logs from all services:"
    docker compose -f "$COMPOSE_FILE" logs 2>&1
    exit 1
fi
