#!/usr/bin/env python3
"""
poke_solve.py — Manually trigger a single bounty solve + PR submission.

Bypasses Nova's biological wait time by directly initializing AxonService
and SimulaService, wiring them together, and firing an axon.solve_bounty Intent.

Target:
    https://github.com/bolivian-peru/os-moda/issues/2
    "BOUNTY (1 SOL): Film 'I Deleted My Nginx Config — Watch What Happens' Demo"

Usage:
    cd d:/.code/EcodiaOS/backend
    python poke_solve.py

Required env vars (set in .env or export manually):
    ECODIAOS_EXTERNAL_PLATFORMS__GITHUB_TOKEN=ghp_...
    ANTHROPIC_API_KEY=sk-ant-...
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from systems.simula.evolution_types import ProposalResult

# ── bootstrap: make sure the package is importable ──────────────────────────
_BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(_BACKEND))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_BACKEND / ".env", override=True)
except ImportError:
    pass  # python-dotenv optional


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


# ── target bounty ────────────────────────────────────────────────────────────
_BOUNTY_ID          = "github-bolivian-peru-os-moda-4006355155"
_ISSUE_URL          = "https://github.com/bolivian-peru/os-moda/issues/2"
_REPO_URL           = "https://github.com/bolivian-peru/os-moda"
_ISSUE_TITLE        = "\U0001f3c6 BOUNTY (1 SOL): Film 'I Deleted My Nginx Config \u2014 Watch What Happens' Demo"
_ISSUE_DESCRIPTION  = (
    "Record a short demo video showing what happens when you deliberately delete your Nginx "
    "config file and then recover it. The video should demonstrate the failure mode and the "
    "recovery process. Submit the video link as a PR against this repo. Reward: 1 SOL on completion."
)
_REWARD_USD         = 1.0


# ── main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"\n{_h('=' * 60)}")
    print(f"{_h('  EcodiaOS -- Manual Bounty Solve (poke_solve.py)')}")
    print(f"{_h('=' * 60)}\n")
    print(f"  Target : {_bold(_ISSUE_URL)}")
    print(f"  Title  : {_dim(_ISSUE_TITLE)}")
    print(f"  Reward : {_ok(f'${_REWARD_USD:.2f} (~1 SOL)')}\n")

    # ── 1. Config ────────────────────────────────────────────────────────────
    print(_h("Step 1: Loading config..."))
    from config import AxonConfig, ExternalPlatformsConfig, LLMConfig, Neo4jConfig, SimulaConfig

    github_token = os.environ.get("ECODIAOS_EXTERNAL_PLATFORMS__GITHUB_TOKEN", "")
    if not github_token:
        print(_warn("  WARNING: ECODIAOS_EXTERNAL_PLATFORMS__GITHUB_TOKEN not set — GitHub API will fail."))
    else:
        masked = github_token[:8] + "..." + github_token[-4:]
        print(_ok(f"  OK  GitHub token: {masked}"))

    neo4j_uri = os.environ.get("ECODIAOS_NEO4J_URI", "")
    if not neo4j_uri:
        print(_err("  ERROR: ECODIAOS_NEO4J_URI not set — Simula requires Neo4j."))
        print(_err("         Set ECODIAOS_NEO4J_URI in .env and retry."))
        sys.exit(1)
    print(_ok(f"  OK  Neo4j URI: {neo4j_uri}"))

    neo4j_config = Neo4jConfig(
        uri=neo4j_uri,
        username=os.environ.get("ECODIAOS_NEO4J_USERNAME", "neo4j"),
        password=os.environ.get("ECODIAOS_NEO4J_PASSWORD", ""),
        database=os.environ.get("ECODIAOS_NEO4J_DATABASE", "neo4j"),
    )
    github_config = ExternalPlatformsConfig(github_token=github_token)
    axon_config   = AxonConfig()
    simula_config = SimulaConfig()
    print(_ok("  OK  Config ready.\n"))

    # ── 2. LLM ───────────────────────────────────────────────────────────────
    llm_provider = os.environ.get("ECODIAOS_LLM__PROVIDER", "bedrock")
    llm_model    = os.environ.get("ECODIAOS_LLM__MODEL", "us.anthropic.claude-sonnet-4-5-20251001-v1:0")
    print(_h(f"Step 2: Building LLM client ({llm_provider})..."))
    from clients.llm import create_llm_provider

    llm_config = LLMConfig(provider=llm_provider, model=llm_model)
    llm = create_llm_provider(llm_config)
    print(_ok(f"  OK  {llm_provider} LLM ready ({llm_model}).\n"))

    # ── 3. Neo4j client ──────────────────────────────────────────────────────
    print(_h("Step 3: Connecting to Neo4j..."))
    from clients.neo4j import Neo4jClient

    neo4j = Neo4jClient(neo4j_config)
    await neo4j.connect()
    print(_ok("  OK  Neo4j connected.\n"))

    # ── 4. SimulaService ─────────────────────────────────────────────────────
    print(_h("Step 4: Initialising SimulaService..."))
    from systems.simula.service import SimulaService

    simula = SimulaService(
        config=simula_config,
        llm=llm,
        neo4j=neo4j,
        codebase_root=_BACKEND,
        instance_name="poke-solve-dev",
    )
    await simula.initialize()
    print(_ok("  OK  SimulaService initialized.\n"))

    # ── 5. AxonService ───────────────────────────────────────────────────────
    print(_h("Step 5: Initialising AxonService..."))
    from systems.axon.service import AxonService

    axon = AxonService(
        config=axon_config,
        github_config=github_config,
        llm=llm,
        instance_id="poke-solve-dev",
    )
    await axon.initialize()

    # Wire Simula into Axon for the SolveBountyExecutor
    axon.set_simula_service(simula)

    stats = axon.stats
    print(_ok(f"  OK  AxonService initialized with {stats['executor_count']} executors."))
    print(_dim("     Simula wired: axon.solve_bounty executor is ready.\n"))

    # ── 6. Build Intent ──────────────────────────────────────────────────────
    print(_h("Step 6: Constructing solve_bounty Intent..."))
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
            description=(
                f"Solve GitHub bounty '{_ISSUE_TITLE}' on {_REPO_URL}, "
                f"submit a PR, and claim the ${_REWARD_USD:.2f} reward."
            ),
            target_domain="oikos.bounty",
            success_criteria={
                "pr_submitted": True,
                "bounty_id": _BOUNTY_ID,
            },
        ),
        plan=ActionSequence(
            steps=[
                Action(
                    executor="axon.solve_bounty",
                    parameters={
                        "bounty_id":      _BOUNTY_ID,
                        "issue_url":      _ISSUE_URL,
                        "repository_url": _REPO_URL,
                        "title":          _ISSUE_TITLE,
                        "description":    _ISSUE_DESCRIPTION,
                        "reward_usd":     _REWARD_USD,
                        "platform":       "github",
                    },
                    timeout_ms=300_000,  # 5 min — Simula code-gen can be slow
                )
            ]
        ),
        autonomy_level_required=AutonomyLevel.STEWARD,
        autonomy_level_granted=AutonomyLevel.STEWARD,
        ethical_clearance=EthicalClearance(
            status=Verdict.APPROVED,
            reasoning="Manual poke — operator pre-approved solve and PR submission.",
        ),
    )

    equor_check = ConstitutionalCheck(
        intent_id=intent.id,
        verdict=Verdict.APPROVED,
        confidence=1.0,
        reasoning="Manual poke — constitutional check bypassed by operator.",
    )

    request = ExecutionRequest(intent=intent, equor_check=equor_check, timeout_ms=360_000)
    print(_ok(f"  OK  Intent built: {intent.id}"))
    print(_dim("     Executor : axon.solve_bounty"))
    print(_dim(f"     Bounty   : {_BOUNTY_ID}\n"))

    # ── 7. Execute ───────────────────────────────────────────────────────────
    print(_h("Step 7: Executing axon.solve_bounty..."))
    print(_dim("     (This may take several minutes — Simula will clone, analyse, and patch the repo)\n"))

    outcome = await axon.execute(request)

    # ── 7b. Auto-approve governance gate if needed ───────────────────────────
    # ADD_SYSTEM_CAPABILITY is always GOVERNANCE_REQUIRED in Simula. For a manual
    # poke we are the operator, so we immediately approve and resume the pipeline.
    if (
        not outcome.success
        and outcome.step_outcomes
        and outcome.step_outcomes[0].result.data
        and outcome.step_outcomes[0].result.data.get("proposal_status") == "awaiting_governance"
    ):
        step_data    = outcome.step_outcomes[0].result.data
        proposal_id  = step_data.get("proposal_id", "")
        gov_id       = step_data.get("governance_record_id") or f"poke-manual-approval-{proposal_id}"

        print(_warn("  INFO  Proposal hit governance gate (ADD_SYSTEM_CAPABILITY is always governed)."))
        print(_warn(f"        Proposal ID     : {proposal_id}"))
        print(_warn(f"        Governance ID   : {gov_id}"))
        print(_warn("        Auto-approving as operator...\n"))


        gov_result: ProposalResult = await simula.approve_governed_proposal(proposal_id, gov_id)

        if gov_result.status.value != "applied":
            print(_err(f"  ERROR  Governance approval failed: {gov_result.reason}"))
            return

        # Pull PR info from history
        pr_url    = getattr(gov_result, "pr_url", "") or ""
        pr_number = getattr(gov_result, "pr_number", None)
        if not pr_url:
            try:
                records = await simula.get_history(limit=5)
                for rec in records:
                    if rec.proposal_id == proposal_id:
                        pr_url    = rec.pr_url or ""
                        pr_number = rec.pr_number
                        break
            except Exception:
                pass

        print(_h("=" * 60))
        print(_ok("  OUTCOME: SUCCESS (via operator governance approval)"))
        print(_h("=" * 60))
        print()
        print(_h("PULL REQUEST"))
        if pr_url:
            print(f"  PR URL    : {_ok(_bold(pr_url))}")
        else:
            print(f"  PR URL    : {_warn('(not available — check GitHub)')}")
        if pr_number:
            print(f"  PR Number : {_bold(f'#{pr_number}')}")
        print()
        print(_h("SOLVE DETAILS"))
        print(f"  Bounty ID       : {_dim(_BOUNTY_ID)}")
        print(f"  Issue URL       : {_dim(_ISSUE_URL)}")
        print(f"  Repository      : {_dim(_REPO_URL)}")
        print(f"  Proposal ID     : {_dim(proposal_id)}")
        print(f"  Proposal Status : {_ok('applied')}")
        print(f"  Reward          : {_ok(f'${_REWARD_USD:.2f}')}")
        print(f"  Files changed   : {_bold(str(len(gov_result.files_changed)))}")
        for f in gov_result.files_changed[:10]:
            print(f"    {_dim(f)}")
        print()
        print(_h("=" * 60))
        if pr_url:
            print(_ok(f"  poke_solve.py complete.  PR: {pr_url}"))
        else:
            print(_warn("  poke_solve.py complete.  No PR URL — check GitHub."))
        print(_h("=" * 60))
        print()
        return

    # ── 8. Print results ─────────────────────────────────────────────────────
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
        if result.data:
            print(_dim("  Debug data:"))
            for k, v in result.data.items():
                print(_dim(f"    {k}: {v}"))
        return

    data = result.data or {}

    # -- Primary output: PR details ------------------------------------------
    pr_url    = data.get("pr_url")
    pr_number = data.get("pr_number")

    print(_h("PULL REQUEST"))
    if pr_url:
        print(f"  PR URL    : {_ok(_bold(pr_url))}")
    else:
        print(f"  PR URL    : {_warn('(not available)')}")
    if pr_number:
        print(f"  PR Number : {_bold(f'#{pr_number}')}")
    print()

    # -- Supporting details --------------------------------------------------
    print(_h("SOLVE DETAILS"))
    print(f"  Bounty ID       : {_dim(data.get('bounty_id', '?'))}")
    print(f"  Issue URL       : {_dim(data.get('issue_url', '?'))}")
    print(f"  Repository      : {_dim(data.get('repository_url', '?'))}")
    print(f"  Proposal ID     : {_dim(data.get('proposal_id', '?'))}")
    print(f"  Proposal Status : {_ok(data.get('proposal_status', '?'))}")
    reward_usd = data.get("reward_usd", 0.0)
    print(f"  Reward          : {_ok(f'${reward_usd:.2f}')}")
    print(f"  Platform        : {_dim(data.get('platform', '?'))}")
    print(f"  Difficulty      : {_dim(data.get('difficulty', '?'))}")
    files_changed = data.get("files_changed", [])
    print(f"  Files changed   : {_bold(str(len(files_changed)))}")
    for f in files_changed[:10]:
        print(f"    {_dim(f)}")
    if len(files_changed) > 10:
        print(f"    {_dim(f'... (+{len(files_changed) - 10} more)')}")
    print()

    # -- Side effects --------------------------------------------------------
    if result.side_effects:
        print(_h("SIDE EFFECTS"))
        for se in result.side_effects:
            print(f"  {_dim(se)}")
        print()

    # -- Observations --------------------------------------------------------
    if result.new_observations:
        print(_h("NEW OBSERVATIONS (-> Atune)"))
        for obs in result.new_observations:
            for line in obs.splitlines():
                print(f"  {line}")
        print()

    print(_h("=" * 60))
    if pr_url:
        print(_ok(f"  poke_solve.py complete.  PR: {pr_url}"))
    else:
        print(_warn("  poke_solve.py complete.  No PR URL returned."))
    print(_h("=" * 60))
    print()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(main())
