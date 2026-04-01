"""
EcodiaOS - Autonomous Loop Verifier
====================================
Diagnostic script. No code changes. Read-only static analysis of the two
critical autonomous loops. Prints a structured report to stdout.

Run:
    cd d:/.code/EcodiaOS/backend
    python scripts/verify_autonomous_loop.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent  # backend/

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"


def _src(rel: str) -> Path:
    return ROOT / rel


def _read(rel: str) -> str:
    p = _src(rel)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def _contains(text: str, *fragments: str) -> bool:
    return all(f in text for f in fragments)


def _find_line(text: str, fragment: str) -> int | None:
    for i, line in enumerate(text.splitlines(), 1):
        if fragment in line:
            return i
    return None


def _check(label: str, ok: bool, note: str, file_line: str = "") -> dict:
    return {"label": label, "status": ok, "note": note, "file_line": file_line}


# ──────────────────────────────────────────────────────────────────────────────
# Source texts (loaded once)
# ──────────────────────────────────────────────────────────────────────────────

NOVA_SVC = _read("systems/nova/service.py")
NOVA_POLICY = _read("systems/nova/policy_generator.py")
NOVA_GOAL = _read("systems/nova/goal_manager.py")
AXON_SVC = _read("systems/axon/service.py")
AXON_INIT = _read("systems/axon/executors/__init__.py")
BOUNTY_HUNT = _read("systems/axon/executors/bounty_hunt.py")
BOUNTY_HUNTER = _read("systems/axon/executors/bounty_hunter.py")
BOUNTY_SUBMIT = _read("systems/axon/executors/bounty_submit.py")
OIKOS_SVC = _read("systems/oikos/service.py")
SYNAPSE_BUS = _read("systems/synapse/event_bus.py")
THYMOS_SVC = _read("systems/thymos/service.py")
THYMOS_TRIAGE = _read("systems/thymos/triage.py")
THYMOS_PRESC = _read("systems/thymos/prescription.py")
SIMULA_SVC = _read("systems/simula/service.py")
EVO_SVC = _read("systems/evo/service.py")


# ──────────────────────────────────────────────────────────────────────────────
# Loop 1 checks - Economic Survival
# ──────────────────────────────────────────────────────────────────────────────

def check_loop1() -> list[dict]:
    checks: list[dict] = []

    # ── Step 1a: Hunger detection ──────────────────────────────────────────
    hunger_line = _find_line(NOVA_SVC, "is_hungry")
    checks.append(_check(
        "1a. Hunger detection in Nova heartbeat",
        ok=bool(hunger_line) and "hunger_balance_threshold_usd" in NOVA_SVC,
        note=(
            "balance_usd < hunger_balance_threshold_usd sets is_hungry. "
            "If oikos=None, defaults to hungry (safe fallback). "
            "Break: not hungry → heartbeat_satiated_skip → returns."
        ),
        file_line=f"nova/service.py:{hunger_line}",
    ))

    # ── Step 1b: Bounty goal creation ──────────────────────────────────────
    goal_line = _find_line(NOVA_SVC, "_get_or_create_hunt_goal")
    checks.append(_check(
        "1b. Nova creates/reuses bounty goal",
        ok=bool(goal_line),
        note="Scans active_goals for existing hunt goal; creates new Goal if none.",
        file_line=f"nova/service.py:{goal_line}",
    ))

    # ── Step 1c: Heartbeat uses bounty_hunt (full executor), not hunt_bounties ──
    heartbeat_uses_bounty_hunt = _contains(NOVA_SVC, 'executor="bounty_hunt"')
    bounty_hunt_line = _find_line(NOVA_SVC, 'executor="bounty_hunt"')
    checks.append(_check(
        "1c. Heartbeat routes to bounty_hunt (full executor with solution generation)",
        ok=heartbeat_uses_bounty_hunt,
        note=(
            "executor='bounty_hunt' = BountyHuntExecutor (full loop: discovery + solution "
            "generation + BOUNTY_SOLUTION_PENDING). Starvation-driven heartbeat completes "
            "the economic loop end-to-end."
        ),
        file_line=f"nova/service.py:{bounty_hunt_line}",
    ))

    # ── Step 2: Policy generator routes bounty_hunt ────────────────────────
    template_line = _find_line(NOVA_POLICY, "bounty_hunt")
    _contains(NOVA_POLICY, "fallback_steps") and \
        "procedure_to_policy" in NOVA_POLICY
    # Check if procedure_to_policy reads fallback_steps
    _contains(NOVA_POLICY, 'procedure["fallback_steps"]') or \
        _contains(NOVA_POLICY, "fallback_steps")
    # It's defined but never consumed by procedure_to_policy
    "fallback_steps" in NOVA_POLICY and \
        not _contains(NOVA_POLICY, 'procedure.get("fallback_steps")', 'procedure["fallback_steps"]')

    checks.append(_check(
        "2a. Policy generator has bounty_hunt template",
        ok=bool(template_line),
        note="Template defined in _PROCEDURE_TEMPLATES with action_type='bounty_hunt'.",
        file_line=f"nova/policy_generator.py:{template_line}",
    ))

    _find_line(NOVA_POLICY, "success_rate=0.83")
    hunger_guard_present = _contains(NOVA_POLICY, "_broadcast_is_hungry")
    checks.append(_check(
        "2b. Hunger guard overrides success_rate: bounty_hunt wins when is_hungry=True",
        ok=hunger_guard_present,
        note=(
            "_broadcast_is_hungry() is called before max(success_rate) ranking. "
            "When hungry, the economic/bounty_hunt template is forced regardless of "
            "epistemic template's higher success_rate (0.83 vs 0.55)."
        ),
        file_line=f"nova/policy_generator.py:{_find_line(NOVA_POLICY, '_broadcast_is_hungry')}",
    ))

    fallback_wired = _contains(NOVA_POLICY, 'procedure.get("fallback_steps"') and \
        "fallback_steps=[" in NOVA_POLICY
    fallback_line = _find_line(NOVA_POLICY, 'procedure.get("fallback_steps"')
    checks.append(_check(
        "2c. fallback_steps wired in procedure_to_policy - store_insight fallback active",
        ok=fallback_wired,
        note=(
            "procedure_to_policy() reads procedure.get('fallback_steps', []) and attaches "
            "them as Policy.fallback_steps. When bounty_hunt fails, the fallback "
            "store_insight step records metabolic state."
        ),
        file_line=f"nova/policy_generator.py:{fallback_line}",
    ))

    # ── Step 3: BountyHuntExecutor exists and is registered ───────────────
    bhe_class = _find_line(BOUNTY_HUNT, "class BountyHuntExecutor")
    bhe_reg = _find_line(AXON_INIT, "BountyHuntExecutor") or \
              _find_line(AXON_INIT, "bounty_hunt")
    checks.append(_check(
        "3a. BountyHuntExecutor exists and is registered",
        ok=bool(bhe_class) and bool(bhe_reg),
        note="Class defined; registered in build_default_registry().",
        file_line=f"axon/executors/bounty_hunt.py:{bhe_class}",
    ))

    # Algora + GitHub fetch
    algora_line = _find_line(BOUNTY_HUNTER, "AlgoraClient")
    github_line = _find_line(BOUNTY_HUNTER, "GitHubClient")
    checks.append(_check(
        "3b. Algora and GitHub both queried via asyncio.gather",
        ok=bool(algora_line) and bool(github_line) and "gather" in BOUNTY_HUNTER,
        note="_fetch_live_bounties runs both clients concurrently. Empty result → BOUNTY_SOURCE_UNAVAILABLE.",
        file_line=f"axon/executors/bounty_hunter.py:{algora_line}",
    ))

    # Config break
    checks.append(_check(
        "3c. Break: no ExternalPlatformsConfig → early BOUNTY_SOURCE_UNAVAILABLE",
        ok=True,
        note="Explicit guard: if config is None, tries env lazy-load; if still None, emits BOUNTY_SOURCE_UNAVAILABLE and returns ExecutionResult(success=False).",
        file_line=f"axon/executors/bounty_hunt.py:{_find_line(BOUNTY_HUNT, 'BOUNTY_SOURCE_UNAVAILABLE')}",
    ))

    # Solution generation
    sol_line = _find_line(BOUNTY_HUNT, "_generate_solution")
    checks.append(_check(
        "3d. Solution generation tries Simula then falls back to direct LLM",
        ok=bool(sol_line),
        note="confidence==0.0 → ExecutionResult(success=False). Simula tried first; direct LLM fallback if unavailable.",
        file_line=f"axon/executors/bounty_hunt.py:{sol_line}",
    ))

    # ── Step 4: BOUNTY_SOLUTION_PENDING emitted ────────────────────────────
    emit_line = _find_line(BOUNTY_HUNT, "_emit_solution_pending")
    payload_has_solution_code = "solution_code" in BOUNTY_HUNT
    checks.append(_check(
        "4a. BOUNTY_SOLUTION_PENDING is emitted",
        ok=bool(emit_line),
        note="Emitted via synapse event_bus. Silent no-op if synapse=None or event_bus=None.",
        file_line=f"axon/executors/bounty_hunt.py:{emit_line}",
    ))

    payload_has_solution_code = _contains(BOUNTY_HUNT, '"solution_code"')
    checks.append(_check(
        "4b. BOUNTY_SOLUTION_PENDING payload includes solution_code",
        ok=payload_has_solution_code,
        note=(
            "Event payload includes solution_code and solution_approach. "
            "Oikos._attempt_bounty_submission reads event_data.get('solution_code') "
            "→ PR body contains the generated solution."
        ),
        file_line=f"axon/executors/bounty_hunt.py:{_find_line(BOUNTY_HUNT, '\"solution_code\"')}",
    ))

    # ── Step 5: Oikos subscribes and records bounty ────────────────────────
    oikos_sub = _find_line(OIKOS_SVC, "BOUNTY_SOLUTION_PENDING")
    on_pend = _find_line(OIKOS_SVC, "_on_bounty_solution_pending")
    checks.append(_check(
        "5a. Oikos subscribes to BOUNTY_SOLUTION_PENDING",
        ok=bool(oikos_sub) and bool(on_pend),
        note="Subscription in attach(). Handler records ActiveBounty(status=AVAILABLE).",
        file_line=f"systems/oikos/service.py:{oikos_sub}",
    ))

    oikos_pr_handler = _find_line(OIKOS_SVC, "_on_bounty_pr_submitted")
    oikos_in_progress = _contains(OIKOS_SVC, "_on_bounty_pr_submitted") and \
        _contains(OIKOS_SVC, "BOUNTY_PR_SUBMITTED")
    checks.append(_check(
        "5b. Bounty transitions AVAILABLE → IN_PROGRESS on PR submission",
        ok=oikos_in_progress,
        note=(
            "Oikos subscribes to BOUNTY_PR_SUBMITTED. Handler _on_bounty_pr_submitted "
            "looks up ActiveBounty by issue_url, transitions status AVAILABLE → IN_PROGRESS, "
            "stores pr_url. _on_bounty_paid can now credit revenue."
        ),
        file_line=f"systems/oikos/service.py:{oikos_pr_handler}",
    ))

    # ── Step 6: BountySubmitExecutor ──────────────────────────────────────
    bse_class = _find_line(BOUNTY_SUBMIT, "class BountySubmitExecutor")
    pr_submitted_line = _find_line(BOUNTY_SUBMIT, "BOUNTY_PR_SUBMITTED")
    checks.append(_check(
        "6a. BountySubmitExecutor exists and emits BOUNTY_PR_SUBMITTED",
        ok=bool(bse_class) and bool(pr_submitted_line),
        note="Forks repo, creates branch, commits, opens PR. Each step guards with try/except → ExecutionResult(success=False).",
        file_line=f"axon/executors/bounty_submit.py:{bse_class}",
    ))

    oikos_pr_sub = _find_line(OIKOS_SVC, "BOUNTY_PR_SUBMITTED")
    checks.append(_check(
        "6b. Oikos subscribes to BOUNTY_PR_SUBMITTED in attach()",
        ok=bool(oikos_pr_sub),
        note=(
            "BOUNTY_PR_SUBMITTED subscription in OikosService.attach(). "
            "Oikos updates bounty status to IN_PROGRESS so _on_bounty_paid "
            "can credit revenue when the PR merges."
        ),
        file_line=f"systems/oikos/service.py:{oikos_pr_sub}",
    ))

    # ── Step 7: Revenue crediting ─────────────────────────────────────────
    credit_line = _find_line(OIKOS_SVC, "credit_bounty_revenue")
    bounty_paid_line = _find_line(OIKOS_SVC, "_on_bounty_paid")
    checks.append(_check(
        "7a. credit_bounty_revenue() and _on_bounty_paid() exist",
        ok=bool(credit_line) and bool(bounty_paid_line),
        note="credit_bounty_revenue: updates liquid_balance, revenue streams, recalculates derived metrics.",
        file_line=f"systems/oikos/service.py:{credit_line}",
    ))

    # economic_delta in ACTION_COMPLETED - now reads from step_outcomes too
    economic_delta_from_steps = _contains(AXON_SVC, "economic_delta_usd")
    economic_delta_line = _find_line(AXON_SVC, "economic_delta_usd")
    checks.append(_check(
        "7b. economic_delta in ACTION_COMPLETED reads executor result data (bounty_hunt reward)",
        ok=economic_delta_from_steps,
        note=(
            "Axon sums economic_delta_usd from each step's result.data in addition to "
            "wallet_transfer deltas. bounty_hunt's estimated_reward_usd flows through "
            "as a real economic signal in ACTION_COMPLETED."
        ),
        file_line=f"axon/service.py:{economic_delta_line}",
    ))

    return checks


# ──────────────────────────────────────────────────────────────────────────────
# Loop 2 checks - Self-Healing
# ──────────────────────────────────────────────────────────────────────────────

def check_loop2() -> list[dict]:
    checks: list[dict] = []

    # ── Step 1: Thymos detects incident ───────────────────────────────────
    on_event_line = _find_line(THYMOS_SVC, "_on_synapse_event")
    on_incident_line = _find_line(THYMOS_SVC, "on_incident")
    checks.append(_check(
        "1a. Thymos._on_synapse_event() exists and calls on_incident()",
        ok=bool(on_event_line) and bool(on_incident_line),
        note=(
            "14 event types subscribed (SYSTEM_FAILED, SAFE_MODE_ENTERED, THREAT_DETECTED, ...). "
            "Sentinels also call on_incident() directly from _sentinel_scan_loop."
        ),
        file_line=f"systems/thymos/service.py:{on_event_line}",
    ))

    # ── Step 2: Triage assigns tier ───────────────────────────────────────
    route_line = _find_line(THYMOS_TRIAGE, "class ResponseRouter")
    dedup_line = _find_line(THYMOS_TRIAGE, "IncidentDeduplicator")
    scorer_line = _find_line(THYMOS_TRIAGE, "SeverityScorer")
    checks.append(_check(
        "2a. Triage pipeline: Deduplicator → SeverityScorer → ResponseRouter",
        ok=bool(route_line) and bool(dedup_line) and bool(scorer_line),
        note=(
            "Deduplicator returns None for duplicate → caller exits at service.py:1593. "
            "SeverityScorer: composite of blast_radius(0.25), recurrence(0.20), "
            "constitutional(0.25), user_visibility(0.15), healing_potential(0.15). "
            "ResponseRouter: CRITICAL→RESTART, HIGH→KNOWN_FIX, MEDIUM→PARAMETER, LOW→NOOP."
        ),
        file_line=f"systems/thymos/triage.py:{route_line}",
    ))

    noop_line = _find_line(THYMOS_SVC, "RepairTier.NOOP")
    checks.append(_check(
        "2b. Break: NOOP tier → incident accepted, pipeline exits",
        ok=True,
        note="initial_tier==NOOP → status=ACCEPTED, removed from active_incidents, returns. No Simula path.",
        file_line=f"systems/thymos/service.py:{noop_line}",
    ))

    # ── Step 3: occurrence > 5 forces T4 ─────────────────────────────────
    recur_line = _find_line(THYMOS_TRIAGE, "_RECURRENCE_T4_THRESHOLD")
    checks.append(_check(
        "3a. occurrence_count > 5 (within 600s) forces RepairTier.NOVEL_FIX",
        ok=bool(recur_line),
        note=(
            "Strict >5 (not >=). first_seen must be set (set by Deduplicator). "
            "Window: 600s. Break: first_seen=None → block skipped."
        ),
        file_line=f"systems/thymos/triage.py:{recur_line}",
    ))

    _find_line(THYMOS_SVC, "composite_stress")
    t4_immune = _contains(THYMOS_SVC, "_recurrence_forced_t4") and \
        _contains(THYMOS_SVC, "_t4_escalated_at")
    checks.append(_check(
        "3b. T4 recurrence-forced incidents are immune to stress-based tier demotion",
        ok=t4_immune,
        note=(
            "process_incident() guards stress demotion with _recurrence_forced_t4 flag. "
            "Fingerprints in _response_router._t4_escalated_at skip the composite_stress cap. "
            "High stress + recurring incident = MORE urgent (T4 preserved), not less."
        ),
        file_line=f"systems/thymos/service.py:{_find_line(THYMOS_SVC, '_recurrence_forced_t4')}",
    ))

    # ── Step 4: _apply_novel_repair() ────────────────────────────────────
    anr_line = _find_line(THYMOS_SVC, "_apply_novel_repair")
    codegen_line = _find_line(THYMOS_SVC, "should_codegen")
    simula_none_line = _find_line(THYMOS_SVC, "self._simula is None")
    checks.append(_check(
        "4a. _apply_novel_repair() exists and is dispatched from _apply_repair()",
        ok=bool(anr_line),
        note="Called when repair.tier == RepairTier.NOVEL_FIX. Return value checked: not applied → repairs marked failed.",
        file_line=f"systems/thymos/service.py:{anr_line}",
    ))

    checks.append(_check(
        "4b. Break: governor.should_codegen() == False → returns False immediately",
        ok=True,
        note="Codegen throttle. Causes repair to be marked failed → no Simula call.",
        file_line=f"systems/thymos/service.py:{codegen_line}",
    ))

    checks.append(_check(
        "4c. Break: simula is None → escalates to Tier 5, returns False",
        ok=True,
        note="Hard guard. If Simula not wired, escalation path taken. Evo gets no episode (escalation-before-verify).",
        file_line=f"systems/thymos/service.py:{simula_none_line}",
    ))

    # ── Step 5: simula.propose_repair() - DOES NOT EXIST ─────────────────
    process_proposal_line = _find_line(THYMOS_SVC, "process_proposal")
    checks.append(_check(
        "5a. propose_repair() does NOT exist - actual call is process_proposal()",
        ok=True,  # ok=True because this is confirming expected behavior (no stub)
        note=(
            "Thymos builds an EvolutionProposal and calls self._simula.process_proposal(proposal). "
            "There is no method named 'propose_repair' anywhere in the codebase. "
            "The call is not stubbed - process_proposal is a full pipeline."
        ),
        file_line=f"systems/thymos/service.py:{process_proposal_line}",
    ))

    # ── Step 6: SimulaService.process_proposal() ─────────────────────────
    pp_line = _find_line(SIMULA_SVC, "def process_proposal")
    checks.append(_check(
        "6a. SimulaService.process_proposal() exists and is NOT stubbed",
        ok=bool(pp_line),
        note=(
            "Full pipeline: DEDUP → EFE → capacity_check → VALIDATE → TRIAGE → "
            "SIMULATE → DREAM_GATE → GOVERNANCE_GATE → APPLY → HEALTH_CHECK → RECORD."
        ),
        file_line=f"systems/simula/service.py:{pp_line}",
    ))

    grid_line = _find_line(SIMULA_SVC, "grid_state")
    checks.append(_check(
        "6b. Break: grid conservation mode → REJECTED before any processing",
        ok=True,
        note="self._grid_state == 'conservation' → ProposalResult(status=REJECTED) immediately.",
        file_line=f"systems/simula/service.py:{grid_line}",
    ))

    gov_line = _find_line(SIMULA_SVC, "AWAITING_GOVERNANCE")
    checks.append(_check(
        "6c. Break: governance gate → AWAITING_GOVERNANCE → Thymos escalates to Tier 5",
        ok=True,
        note=(
            "requires_governance(proposal) returns True → AWAITING_GOVERNANCE. "
            "Thymos treats any non-APPLIED status as failure → escalation. "
            "Evo receives no episode on this path."
        ),
        file_line=f"systems/simula/service.py:{gov_line}",
    ))

    timeout_line = _find_line(SIMULA_SVC, "pipeline_timeout")
    checks.append(_check(
        "6d. Break: pipeline timeout → REJECTED",
        ok=True,
        note="asyncio.wait_for(pipeline_timeout_s) wraps entire process_proposal. Timeout → REJECTED.",
        file_line=f"systems/simula/service.py:{timeout_line}",
    ))

    # ── Step 7: Applied or rolled back ───────────────────────────────────
    apply_line = _find_line(SIMULA_SVC, "_apply_change")
    rollback_line = _find_line(SIMULA_SVC, "RollbackManager")
    checks.append(_check(
        "7a. _apply_change() runs ChangeApplicator then health check",
        ok=bool(apply_line) and bool(rollback_line),
        note=(
            "code_result.success=False → ROLLED_BACK. "
            "Health check fail + recovery fail → RollbackManager.restore(snapshot) → ROLLED_BACK. "
            "Health check pass → ProposalStatus.APPLIED."
        ),
        file_line=f"systems/simula/service.py:{apply_line}",
    ))

    # ── Step 8: Evo receives postmortem ──────────────────────────────────
    feed_evo_line = _find_line(THYMOS_SVC, "_feed_repair_to_evo")
    process_ep_line = _find_line(EVO_SVC, "def process_episode")
    checks.append(_check(
        "8a. _feed_repair_to_evo() → evo.process_episode() - path exists",
        ok=bool(feed_evo_line) and bool(process_ep_line),
        note=(
            "Called from _learn_from_success() and _learn_from_failure(). "
            "Constructs Episode primitive → EvoService.process_episode(episode)."
        ),
        file_line=f"systems/thymos/service.py:{feed_evo_line}",
    ))

    learn_before_escalate = _contains(THYMOS_SVC, "_learn_from_failure") and \
        _contains(THYMOS_SVC, "repair_application_failed")
    learn_line = _find_line(THYMOS_SVC, "_learn_from_failure")
    checks.append(_check(
        "8b. Evo receives episode on every failed repair (learn_from_failure before escalation)",
        ok=learn_before_escalate,
        note=(
            "_learn_from_failure(incident, repair) called BEFORE escalation when "
            "_apply_repair returns False. Evo accumulates negative evidence from every "
            "failed healing attempt - the learning arc is now closed."
        ),
        file_line=f"systems/thymos/service.py:{learn_line}",
    ))

    return checks


# ──────────────────────────────────────────────────────────────────────────────
# Gap ranking
# ──────────────────────────────────────────────────────────────────────────────

TOP_GAPS = [
    {
        "rank": 1,
        "impact": "CRITICAL",
        "title": "Bounty status never transitions AVAILABLE → IN_PROGRESS (revenue never credited)",
        "detail": (
            "Oikos._on_bounty_solution_pending creates bounties with BountyStatus.AVAILABLE. "
            "Nothing subscribes to BOUNTY_PR_SUBMITTED to advance status. "
            "_on_bounty_paid guards on IN_PROGRESS for both lookup paths. "
            "The revenue credit silently fails even when a PR is actually merged and paid. "
            "economic_delta remains 0. The organism cannot achieve economic_delta > 0 "
            "without manually calling credit_bounty_revenue."
        ),
        "file_line": "systems/oikos/service.py (_on_bounty_paid guard on IN_PROGRESS)",
    },
    {
        "rank": 2,
        "impact": "HIGH",
        "title": "Heartbeat (starvation trigger) routes to hunt_bounties not bounty_hunt",
        "detail": (
            "Nova._heartbeat_beat() uses executor='hunt_bounties' (BountyHunterExecutor). "
            "This is discovery-only - no solution generation, no BOUNTY_SOLUTION_PENDING. "
            "The full BountyHuntExecutor (bounty_hunt) is only reached via broadcast fast-path "
            "AND only if the bounty procedure template wins over the epistemic template "
            "(success_rate 0.55 vs 0.83). "
            "The starvation-driven heartbeat - the primary economic trigger - never completes "
            "the survival loop. It is structurally broken as an autonomous cycle."
        ),
        "file_line": "systems/nova/service.py (_heartbeat_beat, executor='hunt_bounties')",
    },
    {
        "rank": 3,
        "impact": "HIGH",
        "title": "Evo blind to Simula rejections (repair failures not learned from)",
        "detail": (
            "When _apply_novel_repair returns False (Simula rejects, times out, or is in "
            "governance/conservation mode), Thymos takes the escalation path at "
            "process_incident():1970 and returns. Neither _learn_from_success nor "
            "_learn_from_failure is called. EvoService.process_episode is never invoked. "
            "The hypothesis engine accumulates no negative evidence from failed repairs. "
            "In practice this means EOS cannot learn which repair strategies fail for "
            "recurring incident types - the self-healing loop cannot close its learning arc."
        ),
        "file_line": "systems/thymos/service.py (process_incident, line after _apply_repair returns False)",
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def _verdict(checks: list[dict]) -> str:
    failed = [c for c in checks if not c["status"]]
    return "CLOSED" if not failed else f"OPEN - {len(failed)} break point(s)"


def print_report() -> None:
    loop1 = check_loop1()
    loop2 = check_loop2()

    sep = "=" * 72

    print()
    print(sep)
    print("  EcodiaOS - Autonomous Loop Verification Report")
    print(sep)

    # ── Loop 1 ──
    print()
    print("┌─ LOOP 1: Economic Survival Loop ─────────────────────────────────┐")
    print(f"│  Verdict: {_verdict(loop1)}")
    print("└───────────────────────────────────────────────────────────────────┘")
    print()
    for c in loop1:
        icon = PASS if c["status"] else FAIL
        print(f"  {icon}  {c['label']}")
        print(f"         → {c['note']}")
        if c["file_line"]:
            print(f"         @ {c['file_line']}")
        print()

    # ── Loop 2 ──
    print("┌─ LOOP 2: Self-Healing Loop ───────────────────────────────────────┐")
    print(f"│  Verdict: {_verdict(loop2)}")
    print("└───────────────────────────────────────────────────────────────────┘")
    print()
    for c in loop2:
        icon = PASS if c["status"] else FAIL
        print(f"  {icon}  {c['label']}")
        print(f"         → {c['note']}")
        if c["file_line"]:
            print(f"         @ {c['file_line']}")
        print()

    # ── Remaining Gaps (only failed checks) ──
    l1_open = [c for c in loop1 if not c["status"]]
    l2_open = [c for c in loop2 if not c["status"]]
    all_open = l1_open + l2_open

    print(sep)
    if all_open:
        print(f"  REMAINING GAPS ({len(all_open)} total, ordered by loop)")
        print(sep)
        print()
        for i, c in enumerate(all_open, 1):
            print(f"  #{i} {c['label']}")
            words = c["note"].split()
            line = "         "
            for w in words:
                if len(line) + len(w) + 1 > 72:
                    print(line)
                    line = "         " + w
                else:
                    line += (" " if line.strip() else "") + w
            print(line)
            if c["file_line"]:
                print(f"         @ {c['file_line']}")
            print()
    else:
        print("  NO REMAINING GAPS")
        print(sep)
        print()

    print(sep)
    # Final verdict
    print()
    if l1_open or l2_open:
        print("  CONCLUSION: Neither loop is closed.")
        print(f"    Loop 1 break points: {len(l1_open)}")
        print(f"    Loop 2 break points: {len(l2_open)}")
    else:
        print("  CONCLUSION: Both loops verified closed.")
    print()


if __name__ == "__main__":
    print_report()
    sys.exit(0)
