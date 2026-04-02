#!/usr/bin/env python3
"""
EcodiaOS Simula CLI

Commands for the Simula self-evolution system, including the organizational
closure capability (Speciation Bible §8.3): generating new subsystem modules.

Usage:
    python -m cli.simula generate-subsystem --name pattern_detector \\
        --purpose "Detect recurring patterns in episode sequences" \\
        --subscribes-to "EPISODE_STORED,MEMORY_EPISODES_DECAYED" \\
        --emits "PATTERN_DETECTED"

    python -m cli.simula generate-subsystem --name anomaly_scanner \\
        --purpose "Scan for statistical anomalies in Soma interoceptive signals" \\
        --subscribes-to "SOMATIC_TICK" --emits "ANOMALY_DETECTED" \\
        --hypothesis-id hyp_abc123 \\
        --dependencies soma

    python -m cli.simula list-generated

Options:
    --codebase-root PATH    Path to EcodiaOS backend root (default: auto-detect)
    --dry-run               Validate spec and show prompt without writing files
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# ── ANSI colors ────────────────────────────────────────────────────────────────
_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_DIM = "\033[2m"


def _ok(msg: str) -> None:
    print(f"{_GREEN}{_BOLD}✓{_RESET} {msg}")


def _err(msg: str) -> None:
    print(f"{_RED}{_BOLD}✗{_RESET} {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"{_CYAN}→{_RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"{_YELLOW}⚠{_RESET} {msg}")


# ── Codebase root detection ───────────────────────────────────────────────────


def _find_codebase_root(hint: str | None = None) -> Path:
    """Find the EcodiaOS backend root directory."""
    if hint:
        return Path(hint).resolve()

    env = os.environ.get("ORGANISM_CODEBASE_ROOT")
    if env:
        return Path(env).resolve()

    # Walk up from this file
    here = Path(__file__).resolve().parent
    for candidate in [here.parent, here.parent.parent]:
        if (candidate / "systems" / "simula").exists():
            return candidate

    # Last resort: current working directory
    return Path.cwd()


# ── generate-subsystem command ────────────────────────────────────────────────


async def cmd_generate_subsystem(args: argparse.Namespace) -> int:
    """
    Generate a new EOS subsystem module using SubsystemGenerator.

    This is the CLI interface for the organizational closure capability
    defined in Speciation Bible §8.3.  The generated module is written to:
        {codebase_root}/systems/{name}/__init__.py

    The module is NOT auto-registered - wire it manually in registry.py
    before the next incarnation.
    """
    from systems.simula.subsystem_generator import SubsystemGenerator, SubsystemSpec

    codebase_root = _find_codebase_root(args.codebase_root)
    _info(f"Codebase root: {codebase_root}")

    # Parse comma-separated event lists
    required_events = [e.strip() for e in args.subscribes_to.split(",") if e.strip()]
    emitted_events = [e.strip() for e in args.emits.split(",") if e.strip()]
    dependencies = [d.strip() for d in args.dependencies.split(",") if d.strip()]

    spec = SubsystemSpec(
        name=args.name,
        purpose=args.purpose,
        trigger_hypothesis_id=args.hypothesis_id or f"cli_{args.name}",
        required_events=required_events,
        emitted_events=emitted_events,
        dependencies=dependencies,
        constraints=args.constraints or [],
    )

    print(f"\n{_BOLD}SubsystemSpec{_RESET}")
    print(f"  Name      : {spec.name}")
    print(f"  Purpose   : {spec.purpose[:100]}")
    print(f"  Hypothesis: {spec.trigger_hypothesis_id}")
    print(f"  Subscribes: {spec.required_events}")
    print(f"  Emits     : {spec.emitted_events}")
    print(f"  Deps      : {spec.dependencies}")
    print()

    if args.dry_run:
        # Validate spec iron rules only
        from systems.simula.subsystem_generator import _FORBIDDEN_NAME_FRAGMENTS, _to_class_name
        import re

        errors: list[str] = []
        name_lower = spec.name.lower()
        for fragment in _FORBIDDEN_NAME_FRAGMENTS:
            if fragment in name_lower:
                errors.append(f"Iron Rule: name contains forbidden fragment '{fragment}'")

        if not re.match(r"^[a-z][a-z0-9_]*$", spec.name):
            errors.append("Name must be snake_case")

        if len(spec.purpose) < 20:
            errors.append("Purpose must be at least 20 characters")

        if errors:
            _err("Spec validation failed:")
            for e in errors:
                print(f"  {_RED}•{_RESET} {e}")
            return 1

        _ok("Spec passes Iron Rule validation")
        _info(f"Would generate: systems/{spec.name}/__init__.py")
        _info(f"Class name   : {_to_class_name(spec.name)}Service")
        return 0

    # Build a minimal stub generator for CLI use (no LLM needed for skeleton)
    # We create a stub code_agent, constraint_checker, and rollback_manager
    class _StubConstraintChecker:
        def check_proposal(self, proposal: Any) -> list:
            return []

    class _StubRollback:
        _root = codebase_root
        async def snapshot(self, proposal_id: str, paths: list) -> Any:
            return None

    class _StubCodeAgent:
        _llm = None  # will use LLM if available via env

    # Try to wire a real LLM if API key is available
    llm = None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            from clients.llm import create_thinking_provider, LLMProvider
            from config import get_config
            cfg = get_config()
            from clients.llm import AnthropicProvider
            llm = AnthropicProvider(
                api_key=api_key,
                model=os.environ.get("SIMULA_LLM_MODEL", "claude-sonnet-4-6"),
            )
            _info("Using Anthropic LLM for code generation")
        except Exception as exc:
            _warn(f"LLM unavailable ({exc}) - will generate skeleton only")

    stub_agent = _StubCodeAgent()
    stub_agent._llm = llm  # type: ignore[attr-defined]

    generator = SubsystemGenerator(
        code_agent=stub_agent,  # type: ignore[arg-type]
        constraint_checker=_StubConstraintChecker(),  # type: ignore[arg-type]
        rollback_manager=_StubRollback(),  # type: ignore[arg-type]
        codebase_root=codebase_root,
    )

    _info("Generating subsystem...")
    result = await generator.generate_subsystem(spec)

    if result.success:
        _ok(f"Subsystem '{result.name}' generated successfully")
        for fp in result.file_paths:
            print(f"  {_GREEN}{fp}{_RESET}")
        print()
        _warn(
            "Next step: wire the new system in backend/core/registry.py "
            "before the next organism incarnation."
        )
        return 0
    else:
        _err(f"Generation failed: {result.reason}")
        for ve in result.validation_errors:
            print(f"  {_RED}•{_RESET} {ve}")
        return 1


# ── list-generated command ─────────────────────────────────────────────────────


async def cmd_list_generated(args: argparse.Namespace) -> int:
    """List subsystems generated in this session (in-memory only)."""
    print(f"\n{_BOLD}Generated Subsystems (this session){_RESET}")
    _warn("Note: this list is in-memory only and empty for fresh CLI invocations.")
    _info(
        "To see all generated systems on disk, run: "
        "ls systems/ in the backend directory."
    )
    return 0


# ── argument parser ───────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cli.simula",
        description="EcodiaOS Simula CLI - self-evolution and organizational closure",
    )
    parser.add_argument(
        "--codebase-root",
        default=None,
        help="Path to EcodiaOS backend root (default: auto-detect)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # generate-subsystem
    gen = sub.add_parser(
        "generate-subsystem",
        help="Generate a new EOS subsystem module (Speciation Bible §8.3)",
    )
    gen.add_argument("--name", required=True, help="Snake_case system name")
    gen.add_argument("--purpose", required=True, help="Purpose description (≥20 chars)")
    gen.add_argument(
        "--subscribes-to",
        default="",
        help="Comma-separated SynapseEventType names to subscribe to",
    )
    gen.add_argument(
        "--emits",
        default="",
        help="Comma-separated SynapseEventType names to emit",
    )
    gen.add_argument(
        "--dependencies",
        default="",
        help="Comma-separated system IDs this subsystem depends on (injected, not imported)",
    )
    gen.add_argument(
        "--hypothesis-id",
        default="",
        help="Evo hypothesis ID that motivated this subsystem (optional)",
    )
    gen.add_argument(
        "--constraints",
        nargs="*",
        default=[],
        help="Additional architecture constraints",
    )
    gen.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate spec and show prompt without writing files",
    )

    # list-generated
    sub.add_parser(
        "list-generated",
        help="List subsystems generated in this session",
    )

    return parser


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Add the backend root to sys.path so imports work
    codebase_root = _find_codebase_root(getattr(args, "codebase_root", None))
    if str(codebase_root) not in sys.path:
        sys.path.insert(0, str(codebase_root))

    command_map = {
        "generate-subsystem": cmd_generate_subsystem,
        "list-generated": cmd_list_generated,
    }

    handler = command_map.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)

    exit_code = asyncio.run(handler(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
