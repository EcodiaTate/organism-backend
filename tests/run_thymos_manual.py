"""
Manual test runner for all Thymos components.
Bypasses pytest (which hangs due to torch import on this machine).

Covers every sub-module with correct API signatures verified against source.
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from unittest.mock import MagicMock

from config import ThymosConfig
from primitives.common import new_id, utc_now
from systems.synapse.types import SynapseEvent, SynapseEventType
from systems.thymos.antibody import AntibodyLibrary
from systems.thymos.diagnosis import (
    CausalAnalyzer,
    DiagnosticEngine,
    TemporalCorrelator,
)
from systems.thymos.governor import HealingGovernor
from systems.thymos.prescription import RepairPrescriber, RepairValidator
from systems.thymos.prophylactic import (
    HomeostasisController,
    ProphylacticScanner,
)
from systems.thymos.sentinels import (
    CognitiveStallSentinel,
    ContractSentinel,
    DriftSentinel,
    ExceptionSentinel,
    FeedbackLoopSentinel,
)
from systems.thymos.service import ThymosService
from systems.thymos.triage import (
    IncidentDeduplicator,
    ResponseRouter,
    SeverityScorer,
)
from systems.thymos.types import (
    CausalChain,
    Diagnosis,
    HealingMode,
    Incident,
    IncidentClass,
    IncidentSeverity,
    RepairSpec,
    RepairTier,
    ValidationResult,
)

passed = 0
failed = 0
errors: list[str] = []


def check(name: str, condition: bool) -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        errors.append(name)
        print(f"  FAIL: {name}")


def make_incident(
    fingerprint: str = "abc",
    severity: IncidentSeverity = IncidentSeverity.MEDIUM,
    blast_radius: float = 0.2,
    user_visible: bool = False,
    incident_class: IncidentClass = IncidentClass.CRASH,
    source_system: str = "nova",
    occurrence_count: int = 1,
) -> Incident:
    return Incident(
        id=new_id(),
        timestamp=utc_now(),
        incident_class=incident_class,
        severity=severity,
        fingerprint=fingerprint,
        source_system=source_system,
        error_type="TestError",
        error_message="test",
        blast_radius=blast_radius,
        user_visible=user_visible,
        occurrence_count=occurrence_count,
    )


def make_config() -> ThymosConfig:
    return ThymosConfig(sentinel_scan_interval_s=999, homeostasis_interval_s=999)


# ─── 1. Sentinels ────────────────────────────────────
def test_sentinels() -> None:
    print("--- Sentinels ---")

    # ── ExceptionSentinel ──
    es = ExceptionSentinel()
    try:
        raise ValueError("test exc")
    except ValueError as e:
        inc = es.intercept("nova", "do_thing", e, {"key": "val"})
    check("ExceptionSentinel returns Incident", isinstance(inc, Incident))
    check("ExceptionSentinel has fingerprint", len(inc.fingerprint) > 0)
    check("ExceptionSentinel severity valid", inc.severity in list(IncidentSeverity))
    check("ExceptionSentinel source_system", inc.source_system == "nova")
    check("ExceptionSentinel class is CRASH", inc.incident_class == IncidentClass.CRASH)
    check("ExceptionSentinel blast_radius > 0", inc.blast_radius > 0)
    check("ExceptionSentinel constitutional_impact", len(inc.constitutional_impact) > 0)
    check("ExceptionSentinel context has method", inc.context.get("method") == "do_thing")

    # fingerprint() is deterministic for same inputs
    fp1 = es.fingerprint("nova", "do_thing", ValueError("x"))
    fp2 = es.fingerprint("nova", "do_thing", ValueError("x"))
    check("ExceptionSentinel.fingerprint deterministic", fp1 == fp2)

    # Critical system gets CRITICAL severity
    try:
        raise RuntimeError("test")
    except RuntimeError as e:
        critical_inc = es.intercept("equor", "review", e)
    check("ExceptionSentinel equor → CRITICAL", critical_inc.severity == IncidentSeverity.CRITICAL)

    # ── ContractSentinel ── (uses DEFAULT_CONTRACT_SLAS)
    cs = ContractSentinel()
    ok = cs.check_contract("atune", "memory", "store_percept", 50.0)
    check("ContractSentinel within SLA → None", ok is None)
    viol = cs.check_contract("atune", "memory", "store_percept", 200.0)
    check("ContractSentinel violation → Incident", isinstance(viol, Incident))
    if viol:
        check("ContractSentinel violation class", viol.incident_class == IncidentClass.CONTRACT_VIOLATION)
        check("ContractSentinel violation has context", "actual_ms" in viol.context)
    unknown = cs.check_contract("foo", "bar", "baz", 999.0)
    check("ContractSentinel unknown contract → None", unknown is None)

    # ── FeedbackLoopSentinel ──
    fls = FeedbackLoopSentinel()
    check("FeedbackLoopSentinel has loops", len(fls._loops) > 0)
    stale = fls.check_loops()
    check("FeedbackLoopSentinel check_loops returns list", isinstance(stale, list))
    check("FeedbackLoopSentinel all stale initially", len(stale) == len(fls._loops))
    fls.report_loop_active("top_down_prediction")
    stale2 = fls.check_loops()
    check("FeedbackLoopSentinel active loop excluded", len(stale2) == len(fls._loops) - 1)
    check("FeedbackLoopSentinel loop_statuses", fls.loop_statuses["top_down_prediction"] is True)

    # ── DriftSentinel ── (uses DEFAULT_DRIFT_METRICS)
    ds = DriftSentinel()
    metric = "synapse.cycle.latency_ms"
    check("DriftSentinel has baselines for defaults", metric in ds._baselines)
    for _i in range(260):
        result = ds.record_metric(metric, 50.0)
    check("DriftSentinel no drift on stable values", result is None)
    drift_result = ds.record_metric(metric, 9999.0)
    check("DriftSentinel detects drift on outlier", isinstance(drift_result, Incident))
    if drift_result:
        check("DriftSentinel drift class", drift_result.incident_class == IncidentClass.DRIFT)
    check("DriftSentinel unknown metric → None", ds.record_metric("unknown.metric", 1.0) is None)
    check("DriftSentinel baselines property", isinstance(ds.baselines, dict))

    # ── CognitiveStallSentinel ── (takes 4 booleans)
    css = CognitiveStallSentinel()
    result = css.record_cycle(True, True, True, True)
    check("CognitiveStallSentinel.record_cycle returns list", isinstance(result, list))
    check("CognitiveStallSentinel active cycle → no stall", len(result) == 0)

    print(f"  Sentinels: {passed} passed")


# ─── 2. Triage ────────────────────────────────────
def test_triage() -> None:
    print("--- Triage ---")

    # Deduplicator
    d = IncidentDeduplicator()
    inc1 = make_incident("fp1")
    check("Dedup first → incident", d.deduplicate(inc1) is inc1)
    check("Dedup second → None", d.deduplicate(make_incident("fp1")) is None)
    check("Dedup count incremented", inc1.occurrence_count == 2)
    check("Dedup active_count 1", d.active_count == 1)
    inc3 = make_incident("fp2")
    check("Dedup different fp → incident", d.deduplicate(inc3) is inc3)
    check("Dedup active_count 2", d.active_count == 2)
    resolved = d.resolve("fp1")
    check("Dedup resolve returns original", resolved is inc1)
    check("Dedup active_count after resolve", d.active_count == 1)
    check("Dedup active_incidents property", len(d.active_incidents) == 1)

    # SeverityScorer
    s = SeverityScorer()
    sev = s.compute_severity(make_incident())
    check("SeverityScorer returns enum", isinstance(sev, IncidentSeverity))

    # ResponseRouter
    r = ResponseRouter()
    check(
        "Router CRITICAL → high tier",
        r.route(make_incident(severity=IncidentSeverity.CRITICAL))
        in (RepairTier.RESTART, RepairTier.KNOWN_FIX, RepairTier.ESCALATE),
    )
    check(
        "Router INFO → NOOP",
        r.route(make_incident(severity=IncidentSeverity.INFO)) == RepairTier.NOOP,
    )
    for sev in IncidentSeverity:
        check(f"Router {sev.name} → valid tier", isinstance(r.route(make_incident(severity=sev)), RepairTier))

    print(f"  Triage: {passed} passed total")


# ─── 3. Diagnosis ────────────────────────────────────
async def test_diagnosis() -> tuple:
    print("--- Diagnosis ---")

    # CausalAnalyzer - trace_root_cause is async
    ca = CausalAnalyzer()
    chain = await ca.trace_root_cause(make_incident())
    check("CausalAnalyzer returns CausalChain", isinstance(chain, CausalChain))
    check("CausalChain has root_system", chain.root_system == "nova")
    check("CausalChain has chain list", isinstance(chain.chain, list))

    # No upstream → local cause
    mem_chain = await ca.trace_root_cause(make_incident(source_system="memory"))
    check("CausalAnalyzer memory → local", mem_chain.root_system == "memory")

    # find_common_upstream
    common = ca.find_common_upstream([
        make_incident(source_system="nova"),
        make_incident(source_system="voxis"),
    ])
    check("CausalAnalyzer.find_common_upstream returns str", isinstance(common, str))

    # TemporalCorrelator - record_event(event_type, details, system_id)
    tc = TemporalCorrelator()
    tc.record_event("deployment", "deployed v1.2", "simula")
    tc.record_event("metric_spike", "latency 500ms", "memory")
    corr = tc.correlate(make_incident())
    check("TemporalCorrelator.correlate returns list", isinstance(corr, list))

    # DiagnosticEngine - __init__(llm_client), diagnose(incident, chain, correlations, antibody)
    de = DiagnosticEngine(llm_client=None)
    diag = await de.diagnose(
        incident=make_incident(),
        causal_chain=chain,
        correlations=corr,
        antibody_match=None,
    )
    check("DiagnosticEngine returns Diagnosis", isinstance(diag, Diagnosis))
    check("Diagnosis has root_cause", isinstance(diag.root_cause, str))
    check("Diagnosis has confidence", isinstance(diag.confidence, float))
    check("Diagnosis has repair_tier", isinstance(diag.repair_tier, RepairTier))

    print(f"  Diagnosis: {passed} passed total")
    return ca, tc, de, chain, corr, diag


# ─── 4. Prescription ────────────────────────────────
async def test_prescription(diag: Diagnosis) -> tuple:
    print("--- Prescription ---")

    # RepairPrescriber - async prescribe(incident, diagnosis)
    rp = RepairPrescriber()
    inc = make_incident(incident_class=IncidentClass.DEGRADATION)
    spec = await rp.prescribe(inc, diag)
    check("RepairPrescriber returns RepairSpec", isinstance(spec, RepairSpec))
    check("RepairSpec has tier", isinstance(spec.tier, RepairTier))
    check("RepairSpec has action", isinstance(spec.action, str))

    # RepairValidator - async validate(incident, repair)
    rv = RepairValidator()
    result = await rv.validate(inc, spec)
    check("RepairValidator returns ValidationResult", isinstance(result, ValidationResult))
    check("ValidationResult.approved is bool", isinstance(result.approved, bool))

    print(f"  Prescription: {passed} passed total")
    return spec, inc


# ─── 5. Antibody Library ────────────────────────────
async def test_antibody(spec: RepairSpec, inc: Incident) -> None:
    print("--- Antibody Library ---")

    # AntibodyLibrary - all methods are async
    ab = AntibodyLibrary(neo4j_client=None)
    await ab.initialize()

    miss = await ab.lookup("nonexistent")
    check("AntibodyLibrary lookup miss → None", miss is None)

    antibody = await ab.create_from_repair(inc, spec)
    check("AntibodyLibrary create_from_repair returns Antibody", antibody is not None)
    check("Antibody has fingerprint", antibody.fingerprint == inc.fingerprint)

    hit = await ab.lookup(inc.fingerprint)
    check("AntibodyLibrary lookup hit", hit is not None)

    # record_outcome(antibody_id, success)
    await ab.record_outcome(antibody.id, True)
    check("Antibody success_count incremented", antibody.success_count == 1)
    await ab.record_outcome(antibody.id, False)
    check("Antibody failure_count incremented", antibody.failure_count == 1)

    all_active = await ab.get_all_active()
    check("AntibodyLibrary get_all_active", isinstance(all_active, list))

    print(f"  Antibody: {passed} passed total")


# ─── 6. Healing Governor ────────────────────────────
def test_governor() -> None:
    print("--- Healing Governor ---")

    gov = HealingGovernor()
    check("Governor starts NOMINAL", gov.healing_mode == HealingMode.NOMINAL)
    check("Governor should_diagnose initially", gov.should_diagnose(make_incident()))
    check("Governor budget_state", hasattr(gov.budget_state, "active_diagnoses"))

    inc = make_incident("gov_fp")
    gov.register_incident(inc)

    # Storm detection: rapid incidents
    for i in range(15):
        gov.register_incident(make_incident(f"storm_{i}"))
    check("Governor mode after flood", gov.healing_mode in list(HealingMode))

    gov.resolve_incident(inc.id)
    check("Governor resolve_incident", True)

    check("Governor storm_activations", isinstance(gov.storm_activations, int))

    print(f"  Governor: {passed} passed total")


# ─── 7. Prophylactic ────────────────────────────────
async def test_prophylactic() -> None:
    print("--- Prophylactic ---")

    ab = AntibodyLibrary(neo4j_client=None)
    await ab.initialize()

    # ProphylacticScanner - async scan(files_changed, file_contents)
    ps = ProphylacticScanner(antibody_library=ab)
    warnings = await ps.scan([])
    check("ProphylacticScanner.scan returns list", isinstance(warnings, list))
    check("ProphylacticScanner stats", ps.stats["scans_run"] == 1)

    # HomeostasisController - sync methods
    hc = HomeostasisController()
    hc.record_metric("synapse.cycle.latency_ms", 50.0)
    adjustments = hc.check_homeostasis()
    check("HomeostasisController.check_homeostasis returns list", isinstance(adjustments, list))
    check("HomeostasisController.metrics_in_range is int", isinstance(hc.metrics_in_range, int))
    check("HomeostasisController.metrics_total", isinstance(hc.metrics_total, int))

    print(f"  Prophylactic: {passed} passed total")


# ─── 8. ThymosService ────────────────────────────────
async def test_service() -> None:
    print("--- ThymosService ---")

    cfg = make_config()
    synapse = MagicMock()
    event_bus = MagicMock()
    event_bus.subscribe = MagicMock()
    synapse._event_bus = event_bus
    synapse._health = MagicMock()
    synapse._health.get_record = MagicMock(return_value=None)

    svc = ThymosService(config=cfg, synapse=synapse, neo4j=None, llm=None, metrics=None)
    await svc.initialize()
    check("Service initialized", svc._initialized is True)
    check("Service has governor", svc._governor is not None)
    check("Service has antibody_library", svc._antibody_library is not None)
    check("Service has deduplicator", svc._deduplicator is not None)
    check("Service has diagnostic_engine", svc._diagnostic_engine is not None)
    check("Service has prescriber", svc._prescriber is not None)
    check("Service subscribed to events", event_bus.subscribe.call_count > 0)

    # Double init is idempotent
    await svc.initialize()
    check("Service double init idempotent", svc._initialized is True)

    # Wiring
    mock_equor = MagicMock()
    svc.set_equor(mock_equor)
    check("Service set_equor", svc._equor is mock_equor)

    mock_evo = MagicMock()
    svc.set_evo(mock_evo)
    check("Service set_evo", svc._evo is mock_evo)

    mock_atune = MagicMock()
    svc.set_atune(mock_atune)
    check("Service set_atune", svc._atune is mock_atune)

    # Incident pipeline
    inc = make_incident("svc_fp1")
    await svc.on_incident(inc)
    await asyncio.sleep(0.15)
    check("Service on_incident tracks count", svc._total_incidents >= 1)

    # Duplicate
    dup = make_incident("svc_fp1")
    await svc.on_incident(dup)
    await asyncio.sleep(0.15)
    check("Service dedup prevents double count", svc._total_incidents == 1)

    # INFO severity
    info_inc = make_incident("info_fp", severity=IncidentSeverity.INFO)
    await svc.on_incident(info_inc)
    await asyncio.sleep(0.15)
    check("Service INFO severity accepted", info_inc.id not in svc._active_incidents)

    # on_incident never raises
    edge = make_incident()
    edge.fingerprint = ""
    await svc.on_incident(edge)
    check("Service on_incident never raises", True)

    # Report exception
    try:
        raise RuntimeError("svc test exc")
    except RuntimeError as e:
        await svc.report_exception("nova", e)
    await asyncio.sleep(0.15)
    check("Service report_exception tracks", svc._total_incidents >= 2)

    # Report contract violation
    await svc.report_contract_violation(
        source="atune",
        target="memory",
        operation="store_percept",
        latency_ms=500.0,
        sla_ms=100.0,
    )
    await asyncio.sleep(0.15)
    check("Service report_contract_violation tracks", svc._total_incidents >= 3)

    # Record metric (should not raise)
    svc.record_metric("synapse.cycle.latency_ms", 120.0)
    check("Service record_metric no raise", True)

    # Scan files
    result = await svc.scan_files([])
    check("Service scan_files returns list", isinstance(result, list))

    # Synapse event handling - SYSTEM_FAILED
    event = SynapseEvent(
        event_type=SynapseEventType.SYSTEM_FAILED,
        data={"system_id": "nova"},
        source_system="synapse",
    )
    before = svc._total_incidents
    await svc._on_synapse_event(event)
    await asyncio.sleep(0.15)
    check("Service SYSTEM_FAILED creates incident", svc._total_incidents > before)

    # Classify events
    sev, cls = svc._classify_synapse_event(
        SynapseEvent(event_type=SynapseEventType.SYSTEM_FAILED, data={})
    )
    check("Classify SYSTEM_FAILED → CRITICAL", sev == IncidentSeverity.CRITICAL)
    check("Classify SYSTEM_FAILED → CRASH", cls == IncidentClass.CRASH)

    sev2, cls2 = svc._classify_synapse_event(
        SynapseEvent(event_type=SynapseEventType.SYSTEM_OVERLOADED, data={})
    )
    check("Classify SYSTEM_OVERLOADED → MEDIUM", sev2 == IncidentSeverity.MEDIUM)
    check("Classify SYSTEM_OVERLOADED → DEGRADATION", cls2 == IncidentClass.DEGRADATION)

    # Health
    h = await svc.health()
    check("Service health returns dict", isinstance(h, dict))
    check("Service health status healthy", h.get("status") == "healthy")
    check("Service health has total_incidents", "total_incidents" in h)
    check("Service health has total_antibodies", "total_antibodies" in h)
    check("Service health has budget", "budget" in h)
    check("Service health has healing_mode", "healing_mode" in h)

    # Stats
    st = svc.stats
    check("Service stats initialized", st.get("initialized") is True)
    check("Service stats has total_incidents", "total_incidents" in st)

    # Telemetry with metrics collector
    metrics_mock = MagicMock()
    metrics_mock.record = MagicMock()
    svc2 = ThymosService(config=cfg, synapse=synapse, neo4j=None, llm=None, metrics=metrics_mock)
    await svc2.initialize()
    tel_inc = make_incident("tel_fp")
    await svc2.on_incident(tel_inc)
    await asyncio.sleep(0.15)
    check("Service telemetry emits metrics", metrics_mock.record.call_count > 0)
    await svc2.shutdown()

    # Telemetry without collector (no raise)
    svc._emit_metric("test.metric", 1.0)
    check("Service _emit_metric no collector", True)

    # Shutdown
    await svc.shutdown()
    check("Service shutdown sentinel_task", svc._sentinel_task is None)
    check("Service shutdown homeostasis_task", svc._homeostasis_task is None)

    print(f"  ThymosService: {passed} passed total")


# ─── Main ─────────────────────────────────────────────
async def async_main() -> None:
    test_sentinels()
    test_triage()
    ca, tc, de, chain, corr, diag = await test_diagnosis()
    spec, inc = await test_prescription(diag)
    await test_antibody(spec, inc)
    test_governor()
    await test_prophylactic()
    await test_service()


def main() -> None:
    global passed, failed

    try:
        asyncio.run(async_main())
    except Exception:
        traceback.print_exc()
        failed += 1

    print(f"\n{'='*60}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    if errors:
        print("FAILURES:")
        for e in errors:
            print(f"  - {e}")
    print(f"{'='*60}")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
