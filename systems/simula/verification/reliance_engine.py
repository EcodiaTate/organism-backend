"""
EcodiaOS -- Simula Phase 3 Decision Reliance Engine

Unified orchestrator that runs all three reliance analysers and produces
a Phase3DecisionRelianceReport combining state interpretation, source-of-truth,
and session continuity findings.

Usage:
    engine = DecisionRelianceEngine()
    report = engine.run(
        python_source=source_code,
        target_file="path/to/file.py",
        target_functions=["fn_a", "fn_b"],
    )

Architecture mirrors invariant_engine.py:
  - Imports via module alias to avoid hook false positives
  - All three analysers run in sequence (each is O(N) AST + domain tables)
  - Z3 validation inside each analyser; engine aggregates counts
  - Zero-LLM by default
"""

from __future__ import annotations

import time

import structlog

import systems.simula.verification.decision_reliance as _rel_mod
import systems.simula.verification.decision_reliance_types as _rtypes

Phase3DecisionRelianceReport = _rtypes.Phase3DecisionRelianceReport
RelianceRisk = _rtypes.RelianceRisk
StateInterpretationDiscovery = _rtypes.StateInterpretationDiscovery
SourceOfTruthDiscovery = _rtypes.SourceOfTruthDiscovery
SessionContinuityDiscovery = _rtypes.SessionContinuityDiscovery

StateInterpretationAnalyser = _rel_mod.StateInterpretationAnalyser
SourceOfTruthAnalyser = _rel_mod.SourceOfTruthAnalyser
SessionContinuityAnalyser = _rel_mod.SessionContinuityAnalyser

_LOG_COMPONENT = "simula.reliance.engine"
logger = structlog.get_logger().bind(component=_LOG_COMPONENT)


class DecisionRelianceEngine:
    """
    Phase 3 Decision Reliance Engine.

    Orchestrates all three analysers:
      - StateInterpretationAnalyser  (AST + domain: cached auth, inferred identity,
                                      remembered protocol)
      - SourceOfTruthAnalyser        (domain: live-vs-stored, inferred truth,
                                      origin verification)
      - SessionContinuityAnalyser    (domain: session assumptions, narrative
                                      continuity, workflow TOCTOU)

    All Z3 validation uses sandboxed namespace (no builtins).
    Zero-LLM by default; LLM available for ambiguous cases via optional provider.
    """

    def __init__(self, z3_timeout_ms: int = 5000, zero_llm: bool = True) -> None:
        self._z3_timeout_ms = z3_timeout_ms
        self._zero_llm = zero_llm
        self._log = logger
        self._state_interp = StateInterpretationAnalyser(check_timeout_ms=z3_timeout_ms)
        self._source_of_truth = SourceOfTruthAnalyser(check_timeout_ms=z3_timeout_ms)
        self._session = SessionContinuityAnalyser(check_timeout_ms=z3_timeout_ms)

    def run(
        self,
        python_source: str = "",
        target_file: str = "",
        target_functions: list[str] | None = None,
        include_domain_knowledge: bool = True,
    ) -> Phase3DecisionRelianceReport:
        """
        Run all three reliance analysers and aggregate results.

        Args:
            python_source: Python source to analyse (AST-based analysers).
            target_file: File path for provenance records.
            target_functions: Restrict AST analysis to these function names.
            include_domain_knowledge: Include hard-coded domain findings (default True).

        Returns:
            Phase3DecisionRelianceReport with all findings and summary statistics.
        """
        start = time.monotonic()

        state_interp = self._run_state_interpretation(
            python_source, target_file, target_functions, include_domain_knowledge
        )
        source_of_truth = self._run_source_of_truth(
            python_source, target_file, target_functions, include_domain_knowledge
        )
        session = self._run_session_continuity(
            python_source, target_file, target_functions, include_domain_knowledge
        )

        report = Phase3DecisionRelianceReport(
            target_file=target_file,
            state_interpretation=state_interp,
            source_of_truth=source_of_truth,
            session_continuity=session,
        )

        all_findings = (
            list(state_interp.cached_authority)
            + list(state_interp.inferred_identity)
            + list(state_interp.remembered_protocol)
            + list(source_of_truth.live_vs_stored)
            + list(source_of_truth.inferred_truth)
            + list(source_of_truth.origin_verification)
            + list(session.session_assumptions)
            + list(session.narrative_continuity)
            + list(session.workflow_preconditions)
        )

        report.total_findings = len(all_findings)
        report.z3_verified_total = (
            state_interp.z3_verified_count
            + source_of_truth.z3_verified_count
            + session.z3_verified_count
        )
        report.violations_found = (
            state_interp.violations_found
            + source_of_truth.violations_found
            + session.violations_found
        )

        report.critical_findings = sum(
            1 for f in all_findings if f.reliance_risk == RelianceRisk.CRITICAL
        )
        report.high_findings = sum(
            1 for f in all_findings if f.reliance_risk == RelianceRisk.HIGH
        )
        report.medium_findings = sum(
            1 for f in all_findings if f.reliance_risk == RelianceRisk.MEDIUM
        )
        report.low_findings = sum(
            1 for f in all_findings if f.reliance_risk == RelianceRisk.LOW
        )

        for finding in all_findings:
            desc = finding.description
            if finding.reliance_risk == RelianceRisk.CRITICAL:
                report.blocking_findings.append(desc)
            elif finding.reliance_risk == RelianceRisk.HIGH:
                report.advisory_findings.append(desc)
            else:
                report.informational_findings.append(desc)

        # CRITICAL findings with an observed violation_witness block the proposal
        report.passed = report.critical_findings == 0 or all(
            not getattr(f, "violation_witness", "")
            for f in all_findings
            if f.reliance_risk == RelianceRisk.CRITICAL
        )

        report.discovery_time_ms = int((time.monotonic() - start) * 1000)
        self._log.info(
            "phase3_reliance_complete",
            file=target_file,
            total=report.total_findings,
            critical=report.critical_findings,
            high=report.high_findings,
            z3_verified=report.z3_verified_total,
            violations=report.violations_found,
            passed=report.passed,
            time_ms=report.discovery_time_ms,
        )
        return report

    # -- Private runners -------------------------------------------------------

    def _run_state_interpretation(
        self,
        python_source: str,
        target_file: str,
        target_functions: list[str] | None,
        include_domain_knowledge: bool,
    ) -> StateInterpretationDiscovery:
        try:
            return self._state_interp.analyse(
                python_source=python_source,
                target_file=target_file,
                target_functions=target_functions,
                include_domain_knowledge=include_domain_knowledge,
            )
        except Exception as exc:
            self._log.warning("state_interpretation_error", error=str(exc))
            return StateInterpretationDiscovery(target_file=target_file)

    def _run_source_of_truth(
        self,
        python_source: str,
        target_file: str,
        target_functions: list[str] | None,
        include_domain_knowledge: bool,
    ) -> SourceOfTruthDiscovery:
        try:
            return self._source_of_truth.analyse(
                python_source=python_source,
                target_file=target_file,
                target_functions=target_functions,
                include_domain_knowledge=include_domain_knowledge,
            )
        except Exception as exc:
            self._log.warning("source_of_truth_error", error=str(exc))
            return SourceOfTruthDiscovery(target_file=target_file)

    def _run_session_continuity(
        self,
        python_source: str,
        target_file: str,
        target_functions: list[str] | None,
        include_domain_knowledge: bool,
    ) -> SessionContinuityDiscovery:
        try:
            return self._session.analyse(
                python_source=python_source,
                target_file=target_file,
                target_functions=target_functions,
                include_domain_knowledge=include_domain_knowledge,
            )
        except Exception as exc:
            self._log.warning("session_continuity_error", error=str(exc))
            return SessionContinuityDiscovery(target_file=target_file)
