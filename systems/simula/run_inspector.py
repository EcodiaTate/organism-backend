"""
EcodiaOS - Inspector Engine runner (Full Auto-Patching Pipeline).

Runs the full Clone → Slice → Map → Prove → Debate → Patch pipeline against
a single authorized target repository and prints a structured summary.

Usage:
    python run_inspector.py [--target <github_url>]

The target must be listed in AUTHORIZED_TARGETS below (or passed via --target).
AWS credentials must be set (Bedrock provider) - see .env.example.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Authorized honeypot targets (edit to add your own) ────────────────────────

AUTHORIZED_TARGETS: list[str] = ["file:///tmp/phantom_workspace_0e73f971"]


async def main(target_url: str) -> int:
    # Deferred imports - avoids loading the full EOS stack until we're sure
    # the venv is activated and the target is authorized.
    from clients.llm import create_llm_provider
    from config import LLMConfig
    from systems.simula.inspector.injector import (
        DynamicTaintInjector,  # noqa: F401 - available for future use
    )
    from systems.simula.inspector.prover import VulnerabilityProver
    from systems.simula.inspector.remediation import RepairAgent
    from systems.simula.inspector.service import InspectorService
    from systems.simula.inspector.shield import AutonomousShield
    from systems.simula.inspector.slicer import SemanticSlicer
    from systems.simula.inspector.temporal import ConcurrencyProver
    from systems.simula.inspector.types import InspectorConfig
    from systems.simula.inspector.verifier import AdversarialVerifier
    from systems.simula.verification.z3_bridge import Z3Bridge

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        print("[ERROR] AWS_ACCESS_KEY_ID is not set.", file=sys.stderr)
        return 1
    if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        print("[ERROR] AWS_SECRET_ACCESS_KEY is not set.", file=sys.stderr)
        return 1

    print("[*] Booting EcodiaOS Simula Inspector Engine (Full Auto-Patching Pipeline)...")
    print(f"[*] Target: {target_url}")

    # ── 1. Core engines ────────────────────────────────────────────────────────

    llm_config = LLMConfig(
        provider=os.environ.get("ORGANISM_LLM__PROVIDER", "bedrock"),
        model=os.environ.get("ORGANISM_LLM__MODEL", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
    )
    llm = create_llm_provider(llm_config)

    z3_bridge = Z3Bridge(check_timeout_ms=10_000)

    # ── 2. Multi-agent swarm ───────────────────────────────────────────────────

    prover = VulnerabilityProver(z3_bridge=z3_bridge, llm=llm)
    # ConcurrencyProver takes (llm, z3_bridge) - note argument order
    temporal_prover = ConcurrencyProver(llm=llm, z3_bridge=z3_bridge)
    slicer = SemanticSlicer(llm=llm)
    verifier = AdversarialVerifier(llm=llm)
    repair_agent = RepairAgent(llm=llm, prover=prover, max_retries=3)
    shield = AutonomousShield(llm=llm)
    # DynamicTaintInjector requires a live httpx client + TaintFlowLinker - skip
    # in this standalone runner (the service accepts taint_injector=None).

    # ── 3. Safety & config ─────────────────────────────────────────────────────

    config = InspectorConfig(
        authorized_targets=[target_url],
        max_workers=2,
        sandbox_timeout_seconds=60,
        log_vulnerability_analytics=False,
        clone_depth=1,
    )

    # ── 4. Orchestrator ────────────────────────────────────────────────────────

    inspector = InspectorService(
        prover=prover,
        config=config,
        eos_root=Path(__file__).parent,
        temporal_prover=temporal_prover,
        slicer=slicer,
        verifier=verifier,
        repair_agent=repair_agent,
        shield=shield,
    )

    # ── 5. Pull the trigger ────────────────────────────────────────────────────

    print("[*] Commencing hunt...")
    result = await inspector.hunt_external_repo(
        github_url=target_url,
        generate_pocs=True,
        generate_patches=True,
    )

    # ── Output ─────────────────────────────────────────────────────────────────

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Hunt complete in {result.total_duration_ms}ms")
    print(f"Surfaces mapped : {result.surfaces_mapped}")
    print(f"Vulnerabilities : {len(result.vulnerabilities_found)}")

    if not result.vulnerabilities_found:
        print("\n[OK] No vulnerabilities proved - target appears secure.")
        await llm.close()
        return 0

    for vuln in result.vulnerabilities_found:
        print(f"\n{'-' * 60}")
        print(f"[!] ZERO-DAY CONFIRMED: {vuln.vulnerability_class.value.upper()}")
        print(f"Severity  : {vuln.severity.value}")
        print(f"Surface   : {vuln.attack_surface.file_path}:{vuln.attack_surface.line_number}")
        print(f"Entry     : {vuln.attack_surface.entry_point}")
        print(f"Goal      : {vuln.attack_goal}")
        print(f"\nZ3 counterexample:\n{vuln.z3_counterexample}")

        # Emit structured boundary test evidence as a single JSON line
        # so the SSE pipeline can detect and forward it to CommandCenter.
        evidence = getattr(vuln, "boundary_test_evidence", None)
        if evidence is not None:
            print(json.dumps({"boundary_test": evidence}), flush=True)

        if vuln.proof_of_concept_code:
            print(f"\nReproduction script (Security Unit Test):\n{vuln.proof_of_concept_code}")
        if vuln.patched_code:
            print("\n[+] AUTONOMOUS PATCH GENERATED & FORMALLY VERIFIED:")
            print("-" * 40)
            print(vuln.patched_code)
            print("-" * 40)
        else:
            print("[-] No patch could be verified.")
        if getattr(vuln, "xdp_filter_code", None):
            print(f"\n{'#' * 60}")
            print("[+] AUTONOMOUS XDP KERNEL SHIELD COMPILED (DRY-RUN):")
            print(f"{'#' * 60}")
            print(vuln.xdp_filter_code)
            print(f"{'#' * 60}")

    print(f"\n{sep}")
    print(
        f"[SUMMARY] {len(result.vulnerabilities_found)} finding(s)  "
        f"critical={result.critical_count}  high={result.high_count}"
    )

    await llm.close()
    return 0 if not result.vulnerabilities_found else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspector engine runner - full auto-patching pipeline"
    )
    parser.add_argument(
        "--target",
        default=AUTHORIZED_TARGETS[0] if AUTHORIZED_TARGETS else "",
        help="GitHub HTTPS URL of the authorized target repository",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.target:
        print(
            "[ERROR] No target specified.\n"
            "  Either add a URL to AUTHORIZED_TARGETS in this script\n"
            "  or pass --target https://github.com/yourusername/nextjs-honeypot",
            file=sys.stderr,
        )
        sys.exit(1)

    sys.exit(asyncio.run(main(args.target)))
