"""
EcodiaOS — Hello World Evolution Test

End-to-end test for the Simula evolution pipeline. Injects an
add_executor proposal to create a DebugLogExecutor in
systems/atune/logging_utils.py, then waits for the worker
result and reports the outcome.

Usage:
    cd backend
    python test_hello_world.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime

from dotenv import load_dotenv

# load_dotenv MUST run before any os.getenv() call
load_dotenv()

import redis.asyncio as aioredis
from ulid import ULID

# ── Redis connection ──────────────────────────────────────────────────────────


def _redis_url() -> str:
    """Build the Redis URL, injecting password from env if not already embedded."""
    url = os.getenv("ECODIAOS_REDIS__URL", "").strip()
    if not url:
        url = "redis://localhost:6379/0"
        print("[!] ECODIAOS_REDIS__URL not set — falling back to localhost")

    pw = os.getenv("ECODIAOS_REDIS_PASSWORD", "").strip()
    if pw and "://" in url and "@" not in url:
        scheme, rest = url.split("://", 1)
        url = f"{scheme}://:{pw}@{rest}"

    return url


REDIS_URL = _redis_url()
TASKS_STREAM = "eos:simula:tasks"
RESULTS_STREAM = "eos:simula:results"
TIMEOUT_S = 300  # 5 min; give the code agent time to work
TARGET_FILE = "systems/atune/logging_utils.py"


# ── Proposal builder ─────────────────────────────────────────────────────────


def _build_proposal() -> dict:
    now = datetime.now(UTC).isoformat()
    proposal_id = str(ULID())

    return {
        "id": proposal_id,
        "created_at": now,
        "updated_at": now,
        "source": "test",
        "category": "add_executor",
        "description": (
            "Create a DebugLogExecutor that inherits from the base Executor "
            "in systems/atune/logging_utils.py. The executor should "
            "emit structured JSON debug output to stdout."
        ),
        "change_spec": {
            "executor_name": "DebugLogExecutor",
            "executor_description": (
                "Structured debug logging executor for the Atune system. "
                "Inherits from the base Executor class. "
                "Emits a labelled JSON dump to stdout. "
                "Pure, no side effects beyond stdout, no circular EOS imports."
            ),
            "executor_action_type": "debug_log",
            "executor_input_schema": {
                "label": "string",
                "data": "object",
            },
            "affected_systems": ["atune"],
            "additional_context": (
                "Target file: systems/atune/logging_utils.py — "
                "create it if it does not exist. "
                "DebugLogExecutor must inherit from the project base Executor class."
            ),
            "code_hint": (
                "# systems/atune/logging_utils.py\n"
                "import json\n"
                "from systems.base import Executor\n\n\n"
                "class DebugLogExecutor(Executor):\n"
                "    '''Emit a structured debug log entry to stdout.'''\n\n"
                "    action_type = 'debug_log'\n\n"
                "    async def execute(self, label: str, data: dict) -> None:\n"
                "        print(f'[DEBUG] {label}: {json.dumps(data, indent=2, default=str)}')\n"
            ),
        },
        "evidence": [],
        "expected_benefit": (
            "Gives developers a zero-dependency structured debug executor inside "
            "the Atune system without adding logging framework coupling."
        ),
        "risk_assessment": (
            "Purely additive. New file, no changes to any existing file. "
            "Zero rollback risk."
        ),
        "status": "proposed",
        "simulation": None,
        "governance_record_id": None,
        "result": None,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    # ── 1. Connect and verify auth ────────────────────────────────────────
    masked = REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL
    print(f"[*] Connecting to Redis at: {masked}")

    r = await aioredis.from_url(REDIS_URL, decode_responses=True)

    try:
        pong = await r.ping()
        if pong:
            print("[+] Redis AUTH OK — connection verified")
        else:
            print("[!] Redis PING returned falsy — check credentials")
            await r.aclose()
            sys.exit(1)
    except Exception as exc:
        print(f"[!] Redis connection FAILED: {exc}")
        await r.aclose()
        sys.exit(1)

    # ── 2. Build and push the proposal ───────────────────────────────────
    proposal = _build_proposal()
    proposal_id: str = proposal["id"]

    print()
    print(f"[*] Proposal ID   : {proposal_id}")
    print(f"[*] Target file   : {TARGET_FILE}")
    print(f"[*] Category      : {proposal['category']}")
    print()

    await r.xadd(
        TASKS_STREAM,
        {
            "proposal_id": proposal_id,
            "payload": json.dumps(proposal),
        },
    )
    print(f"[+] Pushed to stream: {TASKS_STREAM!r}")
    print(f"[*] Waiting up to {TIMEOUT_S}s for worker response on {RESULTS_STREAM!r} ...")
    print()

    # ── 3. Poll results stream for OUR proposal_id ───────────────────────
    # Use "$" so we only see messages that arrive AFTER this moment,
    # avoiding replaying stale results from previous test runs.
    last_id = "$"
    loop = asyncio.get_event_loop()
    deadline = loop.time() + TIMEOUT_S
    found_result: dict | None = None

    while loop.time() < deadline:
        remaining_ms = int((deadline - loop.time()) * 1000)
        block_ms = min(3000, remaining_ms)
        if block_ms <= 0:
            break

        try:
            entries = await r.xread(
                streams={RESULTS_STREAM: last_id},
                count=10,
                block=block_ms,
            )
        except Exception as exc:
            print(f"[!] XREAD error: {exc}")
            break

        if not entries:
            elapsed = int(TIMEOUT_S - (deadline - loop.time()))
            print(f"[...] Still waiting ({elapsed}s elapsed) ...")
            continue

        for _stream, messages in entries:
            for msg_id, fields in messages:
                last_id = msg_id
                if fields.get("proposal_id") == proposal_id:
                    raw = fields.get("result", "{}")
                    found_result = json.loads(raw)
                    break
            if found_result:
                break

        if found_result:
            break

    # ── 4. Report ─────────────────────────────────────────────────────────
    await r.aclose()

    if found_result is None:
        print(f"\n[!] TIMEOUT — no result received within {TIMEOUT_S}s")
        sys.exit(1)

    status = found_result.get("status", "unknown")
    reason = found_result.get("reason", "")
    files_changed = found_result.get("files_changed", [])
    version = found_result.get("version")

    print("=" * 60)
    print(f"  STATUS        : {status.upper()}")
    print(f"  VERSION       : {version}")
    print(f"  REASON        : {reason or 'n/a'}")
    print(f"  FILES CHANGED : {files_changed or 'n/a'}")
    print("=" * 60)

    if status == "applied":
        print("\n[OK] Pipeline succeeded — evolution was applied.")
    elif status == "rejected":
        print(f"\n[FAIL] Proposal REJECTED: {reason}")
        sys.exit(1)
    else:
        print(f"\n[?] Unexpected status: {status!r}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
