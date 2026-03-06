#!/usr/bin/env python3
"""
poke_bounties.py — Standalone live GitHub Bounty Hunter test.

Directly initializes AxonService and fires a hunt_bounties Intent against
GitHub, then prints rich terminal output of every bounty fetched, scored,
and selected.

Usage:
    cd d:/.code/EcodiaOS/backend
    python poke_bounties.py

Required env vars (set in .env or export manually):
    ECODIAOS_EXTERNAL_PLATFORMS__GITHUB_TOKEN=ghp_...
    ANTHROPIC_API_KEY=sk-ant-...  (optional — falls back to score=50 without it)
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# ── bootstrap: make sure the package is importable ──────────────────────────
_BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(_BACKEND))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_BACKEND / ".env", override=True)
except ImportError:
    pass  # python-dotenv optional

import os as _os  # noqa: E402 — after dotenv so .env values are loaded first

# ── colour helpers ───────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_DIM    = "\033[2m"


def _h(text: str) -> str:
    return f"{_BOLD}{_CYAN}{text}{_RESET}"

def _ok(text: str) -> str:
    return f"{_GREEN}{text}{_RESET}"

def _warn(text: str) -> str:
    return f"{_YELLOW}{text}{_RESET}"

def _err(text: str) -> str:
    return f"{_RED}{text}{_RESET}"

def _dim(text: str) -> str:
    return f"{_DIM}{text}{_RESET}"

def _bold(text: str) -> str:
    return f"{_BOLD}{text}{_RESET}"


# ── main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"\n{_h('=' * 60)}")
    print(f"{_h('  EcodiaOS -- Live GitHub Bounty Hunter (poke_bounties.py)')}")
    print(f"{_h('=' * 60)}\n")

    # ── 1. Config ────────────────────────────────────────────────────────────
    print(_h("Step 1: Loading config..."))
    from config import AxonConfig, ExternalPlatformsConfig

    github_token = _os.environ.get("ECODIAOS_EXTERNAL_PLATFORMS__GITHUB_TOKEN", "")
    if not github_token:
        print(_warn("  WARNING: ECODIAOS_EXTERNAL_PLATFORMS__GITHUB_TOKEN is not set."))
        print(_warn("           The GitHub fetch will fail. Set it in .env or export it first."))
    else:
        masked = github_token[:8] + "..." + github_token[-4:]
        print(_ok(f"  OK  GitHub token loaded: {masked}"))

    github_config = ExternalPlatformsConfig(github_token=github_token)
    axon_config   = AxonConfig()
    print(_ok("  OK  Config ready.\n"))

    # ── 2. LLM ───────────────────────────────────────────────────────────────
    print(_h("Step 2: Building LLM client..."))
    llm = None
    anthropic_key = _os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        try:
            from clients.llm import create_llm_provider
            from config import LLMConfig
            llm_config = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001")
            llm = create_llm_provider(llm_config)
            print(_ok("  OK  Anthropic LLM ready (claude-haiku-4-5-20251001)."))
        except Exception as exc:
            print(_warn(f"  WARNING  LLM init failed: {exc}"))
            print(_warn("           Bounties will receive a default ecodian_score of 50."))
    else:
        print(_warn("  WARNING  ANTHROPIC_API_KEY not set -- skipping LLM scoring (score=50 default)."))
    print()

    # ── 3. AxonService ───────────────────────────────────────────────────────
    print(_h("Step 3: Initialising AxonService..."))
    from systems.axon.service import AxonService

    # NOTE: AxonService does NOT accept a `registry=` kwarg.
    # The executor registry is built internally during initialize()
    # via build_default_registry(). Pass github_config + llm directly.
    axon = AxonService(
        config=axon_config,
        github_config=github_config,
        llm=llm,
        instance_id="poke-bounties-dev",
    )
    await axon.initialize()
    stats = axon.stats
    print(_ok(f"  OK  AxonService initialized with {stats['executor_count']} executors."))
    executor_types = stats.get("executor_types", [])
    preview = ", ".join(executor_types[:8])
    if len(executor_types) > 8:
        preview += f"... (+{len(executor_types) - 8} more)"
    print(_dim(f"     Executor types: {preview}"))
    print()

    # ── 4. Build Intent ──────────────────────────────────────────────────────
    print(_h("Step 4: Constructing hunt_bounties Intent..."))
    from primitives.common import AutonomyLevel, Verdict
    from primitives.constitutional import ConstitutionalCheck
    from primitives.intent import (
        Action,
        ActionSequence,
        EthicalClearance,
        GoalDescriptor,
        Intent,
    )
    from systems.axon.types import ExecutionRequest

    intent = Intent(
        goal=GoalDescriptor(
            description="Scan GitHub for live paid bounties and surface the best opportunity.",
            target_domain="oikos.bounty",
        ),
        plan=ActionSequence(
            steps=[
                Action(
                    executor="hunt_bounties",
                    parameters={
                        "target_platforms": ["github", "algora"],
                        "min_reward_usd": 10.0,
                        "max_results": 10,
                        "include_rejected": True,  # show rejects too for debugging
                    },
                    timeout_ms=60_000,
                )
            ]
        ),
        autonomy_level_required=AutonomyLevel.PARTNER,
        autonomy_level_granted=AutonomyLevel.PARTNER,
        ethical_clearance=EthicalClearance(
            status=Verdict.APPROVED,
            reasoning="Manual test -- Equor pre-approved.",
        ),
    )

    equor_check = ConstitutionalCheck(
        intent_id=intent.id,
        verdict=Verdict.APPROVED,
        confidence=1.0,
        reasoning="Manual test -- constitutional check bypassed.",
    )

    request = ExecutionRequest(intent=intent, equor_check=equor_check, timeout_ms=90_000)
    print(_ok(f"  OK  Intent built: {intent.id}"))
    print(_dim(f"     Goal: {intent.goal.description}"))
    print()

    # ── 5. Execute ───────────────────────────────────────────────────────────
    print(_h("Step 5: Executing hunt_bounties via AxonService..."))
    print(_dim("     (This may take up to 60s -- live GitHub API + optional LLM call)\n"))

    outcome = await axon.execute(request)

    # ── 6. Print results ─────────────────────────────────────────────────────
    print(_h("=" * 60))
    if outcome.success:
        print(_ok("  OUTCOME: SUCCESS"))
    else:
        print(_err(f"  OUTCOME: FAILED  -- {outcome.failure_reason}: {outcome.error}"))
    print(_h("=" * 60))
    print()

    if not outcome.step_outcomes:
        print(_warn("  No step outcomes recorded."))
        return

    step   = outcome.step_outcomes[0]
    result = step.result

    if not result.success:
        print(_err(f"  Step failed: {result.error}"))
        return

    data          = result.data or {}
    bounties      = data.get("bounties", [])
    total_scanned = data.get("total_scanned", 0)
    total_passed  = data.get("total_passed", 0)
    total_rejected = data.get("total_rejected", 0)
    scan_id       = data.get("scan_id", "?")
    top_url       = data.get("top_bounty_url")
    scanned_at    = data.get("scanned_at", "?")

    # Summary
    print(_h("SCAN SUMMARY"))
    print(f"  Scan ID     : {_dim(scan_id)}")
    print(f"  Scanned at  : {_dim(scanned_at)}")
    print(f"  Fetched     : {_bold(str(total_scanned))} issues from GitHub")
    print(f"  Passed      : {_ok(str(total_passed))} bounties (passed BountyPolicy)")
    print(f"  Rejected    : {_warn(str(total_rejected))} bounties (failed BountyPolicy)")
    print()

    if top_url:
        print(_h("TOP PICK (highest ROI x score)"))
        print(f"  {_ok(top_url)}")
        print()

    # Bounty table
    if bounties:
        print(_h(f"BOUNTY DETAILS  ({len(bounties)} shown)"))
        print(_dim("  " + "-" * 90))
        for i, b in enumerate(bounties, 1):
            passed      = b.get("policy_passes", False)
            status_icon = "PASS" if passed else "FAIL"
            color       = _ok if passed else _warn

            title   = (b.get("title") or "?")[:55]
            repo    = (b.get("repo") or "?")[:35]
            reward  = b.get("reward_usd", 0.0)
            roi     = b.get("roi", 0.0)
            score   = b.get("ecodian_score", 50)
            diff    = b.get("difficulty", "?")
            url     = b.get("source_url", "")
            reasons = b.get("rejection_reasons", [])

            print(f"  {_dim(str(i).rjust(2))}. [{color(status_icon)}]  {_bold(title)}")
            print(f"      Repo     : {_dim(repo)}")
            print(f"      Reward   : ${reward:.2f}   ROI: {roi:.1f}x   Score: {score}/100   Difficulty: {diff}")
            if url:
                print(f"      URL      : {_dim(url)}")
            if reasons:
                print(f"      Rejected : {_warn(', '.join(reasons))}")
            print()

    # Side effects
    if result.side_effects:
        print(_h("SIDE EFFECTS"))
        for se in result.side_effects:
            print(f"  {_dim(se)}")
        print()

    # Observations
    if result.new_observations:
        print(_h("NEW OBSERVATIONS (-> Atune)"))
        for obs in result.new_observations:
            for line in obs.splitlines():
                print(f"  {line}")
        print()

    print(_h("=" * 60))
    print(_ok("  poke_bounties.py complete."))
    print(_h("=" * 60))
    print()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(main())
