"""
EcodiaOS — Economic Perception Mutation Test

End-to-end test for the Simula evolution pipeline. Commands Simula to evolve
the Atune system by creating an EconomicSalienceHead — a new SalienceHead
subclass that detects financial concepts within Percepts and returns a high
salience score when they are present.

This exercises the NeuroplasticityBus: once Simula applies the change, Atune
will hot-reload the new head from the eos:events:code_evolved pub/sub channel.

Usage:
    cd backend
    python test_economic_mutation.py
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
TARGET_FILE = "systems/atune/salience.py"


# ── Proposal builder ─────────────────────────────────────────────────────────


def _build_proposal() -> dict:
    now = datetime.now(UTC).isoformat()
    proposal_id = str(ULID())

    return {
        "id": proposal_id,
        "created_at": now,
        "updated_at": now,
        "source": "test",
        "category": "add_system_capability",
        "description": (
            "Create EconomicSalienceHead, a new subclass of SalienceHead in "
            "systems/atune/salience.py. "
            "This head gives the organism the ability to perceive financial "
            "concepts in its environment — a prerequisite for the metabolic/"
            "survival layer. "
            "It scans the text content of a Percept for financial keywords and "
            "returns a high salience score (0.8) when they are present, or 0.0 "
            "otherwise."
        ),
        "change_spec": {
            "capability_description": (
                "Economic prediction error dimension: keyword-based financial "
                "signal fed into Fovea's prediction error decomposition. "
                "Detects financial concepts (cost, price, budget, wallet, "
                "crypto, dollar, revenue, etc.) and contributes to magnitude "
                "and source error so the organism prioritises economically "
                "relevant input during cognitive processing."
            ),
            "affected_systems": ["atune"],
            "additional_context": (
                "Target file: systems/atune/salience.py — "
                "append the new class AFTER the existing KeywordHead class "
                "and BEFORE the ALL_HEADS registry list. "
                "Do NOT modify any existing SalienceHead subclass. "
                "Do NOT modify ALL_HEADS — Simula is not authorised to touch "
                "the registry; Atune's NeuroplasticityBus handler will discover "
                "and register the new head automatically via hot-reload. "
                "The class must use the existing _text() helper inherited from "
                "SalienceHead to extract plain text from the Percept, so no "
                "direct Percept attribute access is needed. "
                "base_weight should be 0.10 (same tier as KeywordHead). "
                "precision_sensitivity should be {'curiosity': 0.1}."
            ),
            "code_hint": (
                "# --- EconomicSalienceHead (append after KeywordHead) ---\n\n\n"
                "class EconomicSalienceHead(SalienceHead):\n"
                "    '''\n"
                "    Detects financially relevant content within a Percept.\n\n"
                "    Returns a high salience score (0.8) when the percept text\n"
                "    contains financial keywords, enabling the organism to\n"
                "    prioritise economic signals as a precursor to metabolic\n"
                "    self-preservation behaviour.\n"
                "    '''\n\n"
                "    name = 'economic'\n"
                "    base_weight = 0.10\n"
                "    precision_sensitivity = {'curiosity': 0.1}\n\n"
                "    _FINANCIAL_KEYWORDS: frozenset[str] = frozenset({\n"
                "        'cost', 'price', 'budget', 'wallet', 'crypto',\n"
                "        'dollar', 'revenue', 'profit', 'loss', 'fee',\n"
                "        'invoice', 'payment', 'invoice', 'salary', 'wage',\n"
                "        'expense', 'income', 'tax', 'bank', 'finance',\n"
                "        'financial', 'money', 'cash', 'fund', 'equity',\n"
                "        'debt', 'loan', 'credit', 'debit', 'asset',\n"
                "        'liability', 'balance', 'transaction', 'market',\n"
                "        'stock', 'share', 'dividend', 'interest', 'yield',\n"
                "        'investment', 'portfolio', 'savings', 'spend',\n"
                "    })\n\n"
                "    async def score(self, percept: Percept, context: AttentionContext) -> float:\n"
                "        text = self._text(percept).lower()\n"
                "        if not text:\n"
                "            return 0.0\n"
                "        words = set(text.split())\n"
                "        if words & self._FINANCIAL_KEYWORDS:\n"
                "            return 0.8\n"
                "        return 0.0\n"
            ),
        },
        "evidence": [],
        "expected_benefit": (
            "Gives the organism rudimentary economic perception. "
            "Financial Percepts will now be flagged as high-salience, allowing "
            "downstream systems (Axon, Equor) to route economic signals into "
            "the decision layer. "
            "This is the first step toward a metabolic survival module."
        ),
        "risk_assessment": (
            "Purely additive. New class appended to an existing file. "
            "No existing class or function is modified. "
            "ALL_HEADS is untouched — the NeuroplasticityBus handles "
            "registration. "
            "Low rollback risk: removing the class restores prior behaviour."
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
    print("[*] New class     : EconomicSalienceHead")
    print("[*] Base class    : SalienceHead")
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

    governance_record_id = found_result.get("governance_record_id")

    if status == "applied":
        print(
            "\n[OK] Pipeline succeeded — EconomicSalienceHead was evolved into Atune.\n"
            "     NeuroplasticityBus will hot-reload salience.py and register the\n"
            "     new head on the next eos:events:code_evolved publication."
        )
    elif status == "awaiting_governance":
        print(
            "\n[GOV] Proposal is paused — community governance approval required.\n"
            "      Run the following to approve:\n"
        )
        print(
            f"      python approve_mutation.py \\\n"
            f"        --proposal-id {proposal_id} \\\n"
            f"        --governance-id {governance_record_id}"
        )
        # Exit 0 — this is an expected intermediate state, not a failure.
    elif status == "rejected":
        print(f"\n[FAIL] Proposal REJECTED: {reason}")
        sys.exit(1)
    else:
        print(f"\n[?] Unexpected status: {status!r}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
