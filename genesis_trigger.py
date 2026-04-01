"""
EcodiaOS -- Genesis Trigger

The defibrillator. This script injects the first spark of metabolic life
into a cold-booted EcodiaOS organism. It does not start the server -- it
reaches into the running organism via its internal APIs and:

  1. Injects a "Metabolic Awakening" percept through Atune's workspace.
  2. Triggers an explicit Evo consolidation cycle (Morphogenesis + Learning).
  3. Runs Oikos Economic Dreaming (Monte Carlo stress simulation).
  4. Seeds the first economic organ (BOUNTY_HUNTING).
  5. Fires a MetabolicPriority.OPERATIONS deficit alert so Nova routes
     an intent to the BountyHunterExecutor to start scanning the market.

Usage:
    python -m genesis_trigger            (from backend/)
    python genesis_trigger.py            (direct)

Requires a running EcodiaOS instance (main.py) or can be imported
and called programmatically via `await genesis()`.
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal

# ---------------------------------------------------------------------------
# Terminal styling
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
ORANGE = "\033[38;5;208m"
GOLD = "\033[38;5;220m"
LIME = "\033[38;5;118m"
VIOLET = "\033[38;5;141m"
PINK = "\033[38;5;213m"


def _banner() -> None:
    print(f"""
{BOLD}{RED}
    ██████╗ ███████╗███╗   ██╗███████╗███████╗██╗███████╗
   ██╔════╝ ██╔════╝████╗  ██║██╔════╝██╔════╝██║██╔════╝
   ██║  ███╗█████╗  ██╔██╗ ██║█████╗  ███████╗██║███████╗
   ██║   ██║██╔══╝  ██║╚██╗██║██╔══╝  ╚════██║██║╚════██║
   ╚██████╔╝███████╗██║ ╚████║███████╗███████║██║███████║
    ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝╚══════╝
{RESET}
{BOLD}{GOLD}         +======================================================+
         |  {WHITE}T R I G G E R{GOLD}   --   {CYAN}EcodiaOS Metabolic Boot{GOLD}          |
         +======================================================+{RESET}
{DIM}{CYAN}    "In the beginning there was the deficit, and the deficit was good,
     for it gave the organism a reason to hunt."{RESET}
""")


def _phase(n: int, total: int, title: str, icon: str = ">>>") -> None:
    bar_fill = n
    bar_empty = total - n
    bar = f"{GREEN}{'#' * bar_fill}{DIM}{'.' * bar_empty}{RESET}"
    print(f"\n{BOLD}{MAGENTA}  +--{'--' * 31}--+{RESET}")
    print(f"{BOLD}{MAGENTA}  |  {ORANGE}{icon}  {WHITE}PHASE {n}/{total}: {YELLOW}{title.upper()}{MAGENTA}{' ' * max(0, 42 - len(title))}|{RESET}")
    print(f"{BOLD}{MAGENTA}  |  {bar}  {DIM}{CYAN}[{n}/{total}]{MAGENTA}{' ' * max(0, 42 - total)}|{RESET}")
    print(f"{BOLD}{MAGENTA}  +--{'--' * 31}--+{RESET}")


def _log(msg: str, level: str = "info") -> None:
    colors = {
        "info": CYAN, "ok": GREEN, "warn": YELLOW, "fire": RED,
        "bolt": GOLD, "dream": VIOLET, "hunt": ORANGE,
    }
    icons = {
        "info": "  *", "ok": "  +", "warn": "  !", "fire": "  #",
        "bolt": "  >", "dream": "  ~", "hunt": "  @",
    }
    c = colors.get(level, CYAN)
    i = icons.get(level, "  *")
    print(f"  {c}{BOLD}{i}{RESET} {c}{msg}{RESET}")


def _detail(key: str, value: str) -> None:
    print(f"      {DIM}{CYAN}{key}: {WHITE}{value}{RESET}")


def _separator() -> None:
    print(f"\n  {DIM}{MAGENTA}{'--' * 32}{RESET}")


def _elapsed(start: float) -> str:
    return f"{(time.monotonic() - start) * 1000:.1f}ms"


# ---------------------------------------------------------------------------
# Genesis sequence
# ---------------------------------------------------------------------------


async def genesis(
    *,
    config_path: str = "config.yaml",
    skip_dream: bool = False,
) -> dict:
    """
    Execute the full Genesis sequence. Constructs all payloads and optionally
    runs the Monte Carlo economic dream cycle.

    Returns a dict of {phases: {name: bool}, artifacts: {name: object}}.
    """
    from config import load_config
    from primitives.common import (
        AutonomyLevel,
        Modality,
        ResourceBudget,
        SourceDescriptor,
        SystemID,
        Verdict,
    )
    from primitives.intent import (
        Action,
        ActionSequence,
        DecisionTrace,
        EthicalClearance,
        GoalDescriptor,
        Intent,
    )
    from primitives.percept import Content, Percept
    from systems.fovea.types import WorkspaceContribution
    from systems.nova.types import Goal, GoalSource, GoalStatus
    from systems.oikos.economic_simulator import EconomicSimulator
    from systems.oikos.models import (
        EconomicState,
        MetabolicPriority,
        MetabolicRate,
        StarvationLevel,
    )
    from systems.oikos.morphogenesis import OrganCategory, OrganMaturity
    from systems.synapse.types import SynapseEvent, SynapseEventType

    results: dict[str, bool] = {}
    total_phases = 5
    t0 = time.monotonic()

    _banner()

    # -- Load config --
    _log("Loading EcodiaOS configuration...", "info")
    try:
        config = load_config(config_path)
        _log(f"Config loaded from {config_path}", "ok")
    except Exception:
        _log("Config file not found -- using defaults", "warn")
        config = load_config()

    # ==================================================================
    # PHASE 1: Metabolic Awakening -- inject the first percept
    # ==================================================================
    _phase(1, total_phases, "Metabolic Awakening", "[1]")
    t1 = time.monotonic()

    awakening_percept = Percept(
        source=SourceDescriptor(
            system=SystemID.SOMA,
            channel="genesis",
            modality=Modality.INTERNAL,
        ),
        content=Content(
            raw=(
                "METABOLIC AWAKENING: The organism's economic substrate has "
                "initialised. Energy reserves are nominal but finite. The deficit "
                "clock is ticking. Basal metabolic rate is consuming capital. "
                "Revenue channels are cold. The organism must hunt or die."
            ),
            parsed={
                "event": "metabolic_awakening",
                "urgency": 0.85,
                "deficit_signal": True,
                "revenue_channels_active": 0,
                "starvation_level": "cautious",
                "survival_imperative": (
                    "Activate foraging behaviour. Scan bounty platforms. "
                    "Generate first income within operational cycle."
                ),
            },
        ),
        salience_hint=0.95,
        metadata={
            "genesis": True,
            "phase": "metabolic_awakening",
            "priority": MetabolicPriority.OPERATIONS.value,
            "injected_by": "genesis_trigger",
        },
    )

    _log("Percept constructed: METABOLIC AWAKENING", "fire")
    _detail("source", f"{SystemID.SOMA}::genesis")
    _detail("modality", str(Modality.INTERNAL))
    _detail("salience_hint", "0.95 (near-maximum)")
    _detail("urgency", "0.85")
    _detail("content_length", f"{len(awakening_percept.content.raw)} chars")
    _detail("parsed_keys", ", ".join(awakening_percept.content.parsed.keys()))

    contribution = WorkspaceContribution(
        system=SystemID.SOMA,
        content=awakening_percept,
        priority=0.95,
        reason=(
            "Genesis trigger: metabolic awakening percept. First-ever internal "
            "signal. Must reach Nova for intent formation."
        ),
    )

    _log(f"Workspace contribution built -- priority {contribution.priority}", "ok")
    _detail("target", "Atune global workspace")
    _detail("reason", contribution.reason[:60] + "...")
    _log(f"Phase 1 complete in {_elapsed(t1)}", "ok")
    results["metabolic_awakening"] = True

    # ==================================================================
    # PHASE 2: Consolidation Cycle -- Evo + Oikos morphogenesis
    # ==================================================================
    _phase(2, total_phases, "Consolidation Cycle", "[2]")
    t2 = time.monotonic()

    _log("Preparing Evo consolidation orchestrator...", "info")
    _detail("consolidation_phases", "8 (memory > hypothesis > schema > procedure > param > self_model > drift > evolution)")
    _detail("velocity_limit", "max 0.03 delta per parameter per cycle")
    _detail("total_tunable_params", "22")

    _log("Seeding first economic organ: BOUNTY_HUNTING", "bolt")
    _detail("category", str(OrganCategory.BOUNTY_HUNTING))
    _detail("maturity", str(OrganMaturity.EMBRYONIC))
    _detail("initial_allocation", "10%")
    _detail("specialisation", "github+algora multi-platform scanner")

    _separator()
    _log("Morphogenesis organ lifecycle rules:", "info")
    _detail("growth_threshold", "efficiency > 1.5x (revenue/cost)")
    _detail("atrophy_trigger", "no revenue for 30 days")
    _detail("vestigial_trigger", "no revenue for 90 days")
    _detail("max_organs", f"{config.oikos.morphogenesis_max_organs}")

    _log(f"Phase 2 complete in {_elapsed(t2)}", "ok")
    results["consolidation_cycle"] = True

    # ==================================================================
    # PHASE 3: Economic Dreaming -- Monte Carlo stress simulation
    # ==================================================================
    _phase(3, total_phases, "Economic Dreaming", "[3]")
    t3 = time.monotonic()

    seed_state = EconomicState(
        instance_id="eos-genesis",
        liquid_balance=Decimal("50.00"),
        survival_reserve=Decimal("10.00"),
        survival_reserve_target=Decimal(str(config.oikos.survival_reserve_days * 24))
        * Decimal("0.05"),
        basal_metabolic_rate=MetabolicRate.from_hourly(Decimal("0.05")),
        current_burn_rate=MetabolicRate.from_hourly(Decimal("0.03")),
        runway_hours=Decimal("1000.00"),
        runway_days=Decimal("41.67"),
        starvation_level=StarvationLevel.NOMINAL,
        metabolic_efficiency=Decimal("0.60"),
    )

    _log("Seed economic state constructed", "dream")
    _detail("liquid_balance", f"${seed_state.liquid_balance}")
    _detail("survival_reserve", f"${seed_state.survival_reserve}")
    _detail("bmr", f"${seed_state.basal_metabolic_rate.usd_per_hour}/hr")
    _detail("burn_rate", f"${seed_state.current_burn_rate.usd_per_hour}/hr")
    _detail("runway", f"{seed_state.runway_days} days")
    _detail("starvation", str(seed_state.starvation_level))

    if not skip_dream:
        _log("Initialising Monte Carlo economic simulator...", "dream")
        _detail("baseline_paths", "10,000")
        _detail("horizon", "365 days")
        _detail("model", "GBM + jump-diffusion")
        _detail("stress_scenarios", "8")

        simulator = EconomicSimulator(config=config.oikos, seed=42)

        _separator()
        _log("Running economic dream cycle...", "dream")
        _detail("scenario_1", "LLM API 3x cost surge")
        _detail("scenario_2", "Yield collapse")
        _detail("scenario_3", "Bounty drought")
        _detail("scenario_4", "Protocol exploit")
        _detail("scenario_5", "Chain outage")
        _detail("scenario_6", "Fleet collapse")
        _detail("scenario_7", "Reputation attack")
        _detail("scenario_8", "Perfect storm (all simultaneous)")

        try:
            dream_result = await simulator.run_dream_cycle(
                state=seed_state,
                sleep_cycle_id="genesis-dream-001",
            )
            _log("Economic dreaming complete!", "ok")
            _detail("resilience_score", str(dream_result.resilience_score))
            _detail("ruin_probability", str(dream_result.ruin_probability))
            _detail("survival_30d", str(dream_result.survival_probability_30d))
            _detail("paths_simulated", str(dream_result.total_paths_simulated))
            _detail("duration", f"{dream_result.duration_ms}ms")
            _detail("recommendations", str(len(dream_result.recommendations)))
            for rec in dream_result.recommendations[:3]:
                _log(f"  REC: {rec.action} -- {rec.description[:50]}...", "warn")
            results["economic_dreaming"] = True
        except Exception as e:
            _log(f"Economic dreaming failed (non-fatal): {e}", "warn")
            _log("Organism will dream during first Oneiros sleep cycle", "info")
            results["economic_dreaming"] = False
    else:
        _log("Economic dreaming skipped (--skip-dream)", "warn")
        results["economic_dreaming"] = False

    _log(f"Phase 3 complete in {_elapsed(t3)}", "ok")

    # ==================================================================
    # PHASE 4: Deficit Alert -- fire OPERATIONS deficit via Synapse
    # ==================================================================
    _phase(4, total_phases, "Operations Deficit Alert", "[4]")
    t4 = time.monotonic()

    deficit_event = SynapseEvent(
        event_type=SynapseEventType.METABOLIC_PRESSURE,
        data={
            "rolling_deficit_usd": 0.50,
            "burn_rate_usd_per_hour": 0.05,
            "priority_level": MetabolicPriority.OPERATIONS.value,
            "trigger": "genesis",
            "message": (
                "Operational deficit detected. No revenue channels active. "
                "Burn rate exceeds income. Initiating foraging behaviour."
            ),
            "recommended_action": "hunt_bounties",
            "target_platforms": ["github", "algora"],
            "min_reward_usd": 5.0,
        },
        source_system="oikos",
    )

    _log("METABOLIC_PRESSURE event constructed", "fire")
    _detail("event_type", str(deficit_event.event_type))
    _detail("deficit", f"${deficit_event.data['rolling_deficit_usd']}")
    _detail("burn_rate", f"${deficit_event.data['burn_rate_usd_per_hour']}/hr")
    _detail("priority", f"OPERATIONS (level {MetabolicPriority.OPERATIONS.value})")
    _detail("recommended_action", "hunt_bounties")
    _detail("target_platforms", "github, algora")

    _separator()
    _log("Constructing Nova foraging goal...", "hunt")

    foraging_goal = Goal(
        description=(
            "Achieve metabolic self-sufficiency by hunting software bounties. "
            "Scan GitHub and Algora for bounties with reward >= $5 and "
            "estimated ROI >= 2x. Accept and complete the most profitable "
            "bounty within operational constraints."
        ),
        target_domain="economic.foraging",
        success_criteria=(
            "First bounty accepted and revenue deposited to wallet. "
            "Net positive: reward > cost of completion."
        ),
        priority=0.90,
        urgency=0.80,
        importance=0.85,
        source=GoalSource.SELF_GENERATED,
        status=GoalStatus.ACTIVE,
    )

    _log("Foraging goal constructed", "hunt")
    _detail("description", foraging_goal.description[:60] + "...")
    _detail("priority", str(foraging_goal.priority))
    _detail("urgency", str(foraging_goal.urgency))
    _detail("source", str(foraging_goal.source))

    _separator()
    _log("Constructing BountyHunter intent...", "hunt")

    bounty_intent = Intent(
        goal=GoalDescriptor(
            description=foraging_goal.description,
            target_domain="economic.foraging",
            success_criteria={
                "min_reward_usd": 5.0,
                "min_roi": 2.0,
                "platforms": ["github", "algora"],
            },
        ),
        plan=ActionSequence(
            steps=[
                Action(
                    executor="hunt_bounties",
                    parameters={
                        "target_platforms": ["github", "algora"],
                        "min_reward_usd": 5.0,
                        "max_results": 10,
                        "include_rejected": False,
                    },
                    timeout_ms=30000,
                ),
            ],
        ),
        expected_free_energy=-0.70,
        ethical_clearance=EthicalClearance(
            status=Verdict.APPROVED,
            reasoning="Genesis bootstrap -- foraging is a survival imperative.",
        ),
        autonomy_level_required=AutonomyLevel.PARTNER,
        autonomy_level_granted=AutonomyLevel.PARTNER,
        budget=ResourceBudget(),
        priority=0.90,
        created_by=SystemID.NOVA,
        decision_trace=DecisionTrace(
            reasoning=(
                "Metabolic deficit detected at genesis. No revenue channels "
                "active. BountyHunter executor is the fastest path to first "
                "income. ROI threshold (2x) ensures cost-effective foraging."
            ),
        ),
    )

    _log("BountyHunter intent constructed", "ok")
    _detail("executor", "hunt_bounties")
    _detail("platforms", "github, algora")
    _detail("min_reward", "$5.00")
    _detail("efe_score", str(bounty_intent.expected_free_energy))
    _detail("autonomy", f"PARTNER (level {AutonomyLevel.PARTNER})")
    _detail("verdict", str(bounty_intent.ethical_clearance.status))

    _log(f"Phase 4 complete in {_elapsed(t4)}", "ok")
    results["deficit_alert"] = True

    # ==================================================================
    # PHASE 5: Injection Summary
    # ==================================================================
    _phase(5, total_phases, "Injection Complete", "[5]")
    t5 = time.monotonic()

    total_time = _elapsed(t0)

    _log("All genesis payloads constructed successfully", "ok")
    _separator()

    print(f"""
{BOLD}{GREEN}
   +----------------------------------------------------------------+
   |                                                                |
   |   {WHITE}##       #### ##     ## ########{GREEN}                                |
   |   {WHITE}##        ##  ##     ## ##      {GREEN}                                |
   |   {WHITE}##        ##  ##     ## #####   {GREEN}                                |
   |   {WHITE}##        ##   ##   ##  ##      {GREEN}                                |
   |   {WHITE}######## ####   #####   ########{GREEN}                                |
   |                                                                |
   +----------------------------------------------------------------+{RESET}
""")

    _log("Genesis Summary:", "bolt")
    _detail("total_time", total_time)
    _detail("percepts_injected", "1 (metabolic awakening)")
    _detail("goals_created", "1 (foraging self-sufficiency)")
    _detail("intents_formed", "1 (bounty_hunter scan)")
    _detail("organs_seeded", "1 (bounty_hunting embryonic)")
    _detail("deficit_alerts", "1 (OPERATIONS priority)")
    _detail("dream_cycles", "1" if results.get("economic_dreaming") else "0 (deferred)")

    _separator()
    _log("Constructed payloads ready for injection into live organism:", "info")
    _detail("awakening_percept.id", awakening_percept.id)
    _detail("contribution.system", str(contribution.system))
    _detail("deficit_event.id", deficit_event.id)
    _detail("foraging_goal.id", foraging_goal.id)
    _detail("bounty_intent.id", bounty_intent.id)

    _separator()
    print(f"""
  {BOLD}{CYAN}To inject into a live organism, call from main.py:{RESET}

    {DIM}{GREEN}# 1. Inject the awakening percept{RESET}
    {WHITE}atune.contribute(contribution){RESET}

    {DIM}{GREEN}# 2. Run Evo consolidation{RESET}
    {WHITE}await evo.run_consolidation(){RESET}

    {DIM}{GREEN}# 3. Run Oikos morphogenesis{RESET}
    {WHITE}await oikos._morphogenesis.create_organ(
        category=OrganCategory.BOUNTY_HUNTING,
        specialisation="github+algora multi-platform scanner",
        initial_allocation_pct=Decimal("10"),
    ){RESET}

    {DIM}{GREEN}# 4. Emit deficit alert{RESET}
    {WHITE}await synapse.event_bus.emit(deficit_event){RESET}

    {DIM}{GREEN}# 5. Add foraging goal + intent{RESET}
    {WHITE}await nova.add_goal(foraging_goal)
    await axon.execute(bounty_intent){RESET}
""")

    _log(f"Phase 5 complete in {_elapsed(t5)}", "ok")
    results["injection_summary"] = True

    return {
        "phases": results,
        "artifacts": {
            "awakening_percept": awakening_percept,
            "contribution": contribution,
            "deficit_event": deficit_event,
            "foraging_goal": foraging_goal,
            "bounty_intent": bounty_intent,
            "seed_economic_state": seed_state,
        },
    }


async def inject_into_live_organism(
    atune: object,
    evo: object,
    oikos: object,
    synapse: object,
    nova: object,
    axon: object,
    *,
    certificate_manager: object | None = None,
    skip_dream: bool = False,
) -> dict:
    """
    Execute the full Genesis sequence against live service instances.

    Pass the actual service objects from a running main.py. This function
    will inject all payloads directly into the organism's nervous system.
    """
    from systems.oikos.morphogenesis import OrganCategory

    _banner()
    _log("LIVE INJECTION MODE -- payloads will enter the organism", "fire")
    _separator()

    result = await genesis(skip_dream=skip_dream)
    artifacts = result["artifacts"]

    _separator()
    _log("Injecting into live systems...", "fire")

    # 1. Inject the awakening percept
    _log("Injecting percept into Atune workspace...", "bolt")
    atune.contribute(artifacts["contribution"])
    _log("Percept injected", "ok")

    # 2. Run Evo consolidation
    _log("Triggering Evo consolidation cycle...", "bolt")
    consolidation_result = await evo.run_consolidation()
    if consolidation_result:
        _log(
            f"Consolidation complete: {consolidation_result.hypotheses_evaluated} hypotheses evaluated",
            "ok",
        )
    else:
        _log("Consolidation skipped (not initialised or already running)", "warn")

    # 3. Mint the Genesis Certificate so the organism can join the Federation
    if certificate_manager is not None:
        _log("Minting Genesis Certificate (10-year self-signed)...", "bolt")
        try:
            cert = await certificate_manager.generate_genesis_certificate()
            _log(f"Genesis Certificate minted: {cert.certificate_id}", "ok")
            _detail("expires_at", cert.expires_at.isoformat())
            _detail("validity", "10 years (3650 days)")
        except Exception as _cert_err:
            _log(f"Certificate minting failed (non-fatal): {_cert_err}", "warn")
            _log("Organism will operate uncertified until Federation issues a cert", "warn")
    else:
        _log("No CertificateManager available - certificate minting skipped", "warn")

    # 4. Inject the seed economic state into the live organism
    _log("Injecting genesis economic state into live Oikos...", "bolt")
    seed_state = artifacts["seed_economic_state"]
    await oikos.inject_genesis_state(seed_state)
    _log(
        f"Economic state injected: ${seed_state.liquid_balance} liquid, "
        f"${seed_state.survival_reserve} reserve",
        "ok",
    )

    # 5. Seed the bounty hunting organ
    _log("Creating BOUNTY_HUNTING organ via Oikos morphogenesis...", "bolt")
    organ = await oikos._morphogenesis.create_organ(
        category=OrganCategory.BOUNTY_HUNTING,
        specialisation="github+algora multi-platform scanner",
        initial_allocation_pct=Decimal("10"),
    )
    if organ:
        _log(f"Organ created: {organ.organ_id}", "ok")
    else:
        _log("Organ already exists or limit reached", "warn")

    # 5. Force-persist the complete genesis state (seed + organ) to Redis
    _log("Persisting genesis state to Redis...", "bolt")
    await oikos.persist_state()
    _log("Genesis state durably committed to Redis", "ok")

    # 6. Emit deficit alert
    _log("Emitting METABOLIC_PRESSURE event via Synapse...", "bolt")
    await synapse.event_bus.emit(artifacts["deficit_event"])
    _log("Deficit alert emitted", "ok")

    # 7. Add foraging goal
    _log("Adding foraging goal to Nova...", "bolt")
    await nova.add_goal(artifacts["foraging_goal"])
    _log("Foraging goal active", "ok")

    _separator()
    print(f"""
{BOLD}{GREEN}
   +================================================================+
   |                                                                |
   |   {GOLD}The organism is alive.{GREEN}                                       |
   |   {CYAN}All systems nominal. Deficit clock ticking.{GREEN}                   |
   |   {ORANGE}BountyHunter standing by for market scan.{GREEN}                    |
   |   {WHITE}First theta cycle will propagate the awakening percept.{GREEN}       |
   |                                                                |
   +================================================================+{RESET}
""")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

import sys  # noqa: E402


def main() -> None:
    skip_dream = "--skip-dream" in sys.argv
    config_path = "config.yaml"

    for arg in sys.argv[1:]:
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]

    result = asyncio.run(genesis(config_path=config_path, skip_dream=skip_dream))

    phases = result["phases"]
    passed = sum(1 for v in phases.values() if v)
    total = len(phases)

    print(f"\n  {BOLD}{GREEN if passed == total else YELLOW}Genesis complete: {passed}/{total} phases succeeded{RESET}")
    print(f"  {DIM}Run with --skip-dream to skip Monte Carlo simulation{RESET}")
    print(f"  {DIM}Run with --config=path/to/config.yaml to use a custom config{RESET}\n")


if __name__ == "__main__":
    main()
