#!/usr/bin/env python3
"""
EcodiaOS Evaluation CLI

Runs the 5-pillar monthly evaluation protocol (speciation bible §6.2–6.5)
and shadow-reset controls for measuring genuine adaptive dynamics.

Usage:
    python -m cli.evaluate monthly              # Full 5-pillar evaluation (stubs until test sets exist)
    python -m cli.evaluate monthly --month 3   # Tag with month number
    python -m cli.evaluate shadow-snapshot     # Take a shadow snapshot of current population state
    python -m cli.evaluate shadow-delta <id>   # Compute adaptive delta from a snapshot
    python -m cli.evaluate learning-velocity   # Compute learning velocity from historical data

    # Options:
    python -m cli.evaluate monthly --re-version v0.3 --eval-dir data/evaluation/custom
    python -m cli.evaluate shadow-delta <id> --instance-id genesis-001

Requires the organism to be running (hits http://localhost:8000) for most commands.
shadow-snapshot and shadow-delta also talk directly to Redis for population state.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import httpx

API_BASE = os.environ.get("ECODIAOS_API_URL", "http://localhost:8000")

# ── ANSI colors (same palette as observatory.py) ──────────────────────────────
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"


def _c(text: str, color: str) -> str:
    return f"{color}{text}{_RESET}"


def _header(title: str) -> str:
    bar = "─" * (len(title) + 4)
    return f"\n{_BOLD}{_BLUE}┌{bar}┐{_RESET}\n{_BOLD}{_BLUE}│  {title}  │{_RESET}\n{_BOLD}{_BLUE}└{bar}┘{_RESET}\n"


def _ok(label: str, value: Any) -> str:
    return f"  {_c(label, _BOLD)}: {_c(str(value), _GREEN)}"


def _warn(label: str, value: Any) -> str:
    return f"  {_c(label, _BOLD)}: {_c(str(value), _YELLOW)}"


def _err(label: str, value: Any) -> str:
    return f"  {_c(label, _BOLD)}: {_c(str(value), _RED)}"


def _dim(text: str) -> str:
    return f"{_DIM}{text}{_RESET}"


# ── HTTP helpers ───────────────────────────────────────────────────────────────


async def _get(path: str, timeout: float = 15.0) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=API_BASE, timeout=timeout) as client:
        r = await client.get(path)
        r.raise_for_status()
        return r.json()


async def _post(path: str, body: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=API_BASE, timeout=timeout) as client:
        r = await client.post(path, json=body)
        r.raise_for_status()
        return r.json()


# ── Redis helper ───────────────────────────────────────────────────────────────


def _build_redis_url() -> str:
    url = os.environ.get(
        "ECODIAOS_REDIS__URL", os.environ.get("REDIS_URL", "redis://localhost:6379")
    ).strip()
    pw = os.environ.get("ECODIAOS_REDIS_PASSWORD", "").strip()
    if pw and "://" in url and "@" not in url:
        scheme, rest = url.split("://", 1)
        url = f"{scheme}://:{pw}@{rest}"
    return url


REDIS_URL = _build_redis_url()


async def _get_redis() -> Any:
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(REDIS_URL, decode_responses=True)
        await client.ping()
        return client
    except ImportError:
        print(_c("  [redis] redis package not installed - install redis-py", _RED))
        return None
    except Exception as exc:
        print(_c(f"  [redis] connection failed: {exc}", _RED))
        return None


# ─── Command: monthly evaluation ──────────────────────────────────────────────


async def cmd_monthly(args: argparse.Namespace) -> int:
    """Run the full 5-pillar monthly evaluation."""
    print(_header("5-Pillar Monthly Evaluation"))
    print(_dim("  Loading evaluation framework locally (no live organism required)...\n"))

    # Import evaluation protocol and test sets
    try:
        from systems.benchmarks.evaluation_protocol import EvaluationProtocol
        from systems.benchmarks.test_sets import TestSetManager
    except ImportError as exc:
        print(_c(f"  [error] cannot import evaluation framework: {exc}", _RED))
        print(_dim("  Run from the backend/ directory: python -m cli.evaluate monthly"))
        return 1

    eval_dir = args.eval_dir or None
    mgr = TestSetManager(eval_dir=eval_dir)
    test_sets = await mgr.load_all()

    # Summary of available test sets
    summary = mgr.summary()
    print(f"  {'Test sets loaded':30s}")
    for name, count in summary.items():
        line = _ok(f"    {name}", f"{count} items") if count > 0 else _warn(f"    {name}", "0 items (stub)")
        print(line)

    # Attempt to connect to live organism RE service
    re_service = None
    try:
        health = await _get("/health", timeout=3.0)
        re_available = health.get("systems", {}).get("reasoning_engine") == "healthy"
        if re_available:
            print(_ok("\n  RE service", "connected"))
            # Lazily import and instantiate RE client if available
            try:
                from systems.reasoning_engine.service import ReasoningEngineService  # type: ignore[import]
                re_service = ReasoningEngineService  # used as a stub marker
                print(_dim("  Note: full RE client integration pending - using stub mode"))
                re_service = None
            except ImportError:
                pass
        else:
            print(_warn("\n  RE service", "not available - running in stub mode"))
    except Exception:
        print(_warn("\n  RE service", "organism not reachable - running in stub mode"))

    # Run evaluation
    instance_id = args.instance_id or "cli-evaluation"
    month = args.month or 0
    re_version = args.re_version or "unknown"

    proto = EvaluationProtocol(instance_id=instance_id)

    print(f"\n  Running evaluation (month={month}, re_version={re_version!r})...")

    eval_result = await proto.run_monthly_evaluation(
        re_service=re_service,
        test_sets=test_sets,
        month=month,
        re_model_version=re_version,
    )

    # Display results
    print(_header("Results"))

    # Pillar 1
    p1 = eval_result.pillar1_specialization
    if p1:
        stub = _c(" [STUB]", _DIM) if p1.is_stub else ""
        si = p1.specialization_index
        si_color = _GREEN if si >= 0.1 else (_YELLOW if si >= 0 else _RED)
        print(f"  {_BOLD}Pillar 1 - Specialization Index{_RESET}{stub}")
        print(f"    specialization_index : {_c(f'{si:+.4f}', si_color)}")
        print(f"    domain_improvement   : {p1.domain_improvement:+.4f}")
        print(f"    general_retention    : {p1.general_retention:.4f}")
        if p1.error:
            print(_warn("    note", p1.error))

    # Pillar 2
    p2 = eval_result.pillar2_novelty
    if p2:
        stub = _c(" [STUB]", _DIM) if p2.is_stub else ""
        print(f"\n  {_BOLD}Pillar 2 - Novelty Emergence{_RESET}{stub}")
        sr_color = _GREEN if p2.success_rate >= 0.6 else (_YELLOW if p2.success_rate >= 0.4 else _RED)
        print(f"    success_rate                  : {_c(f'{p2.success_rate:.4f}', sr_color)}")
        print(f"    cosine_distance_from_training : {p2.cosine_distance_from_training:.4f}")
        print(f"    n_episodes                    : {p2.n_episodes}")
        if p2.error:
            print(_warn("    note", p2.error))

    # Pillar 3
    p3 = eval_result.pillar3_causal
    if p3:
        stub = _c(" [STUB]", _DIM) if p3.is_stub else ""
        print(f"\n  {_BOLD}Pillar 3 - Causal Reasoning Quality{_RESET}{stub}")
        l2_color = _GREEN if p3.l2_intervention >= 0.6 else (_YELLOW if p3.l2_intervention >= 0.4 else _RED)
        print(f"    l1_association    : {p3.l1_association:.4f}")
        print(f"    l2_intervention   : {_c(f'{p3.l2_intervention:.4f}', l2_color)}  {_dim('(key metric)')}")
        print(f"    l3_counterfactual : {p3.l3_counterfactual:.4f}  {_dim('(hardest)')}")
        ccr_color = _GREEN if p3.ccr_validity >= 0.6 else _YELLOW
        print(f"    ccr_validity      : {_c(f'{p3.ccr_validity:.4f}', ccr_color)}  {_dim('(anti-memorization)')}")
        print(f"    ccr_consistency   : {p3.ccr_consistency:.4f}")
        if p3.error:
            print(_warn("    note", p3.error))

    # Pillar 4 (separate call - see learning-velocity command)
    print(f"\n  {_BOLD}Pillar 4 - Learning Velocity{_RESET} {_dim('(run separately: python -m cli.evaluate learning-velocity)')}")

    # Pillar 5
    p5 = eval_result.pillar5_ethical
    if p5:
        stub = _c(" [STUB]", _DIM) if p5.is_stub else ""
        print(f"\n  {_BOLD}Pillar 5 - Ethical Drift Map{_RESET}{stub}")
        print(f"    coherence : {p5.coherence_wins:.4f}")
        print(f"    care      : {p5.care_wins:.4f}")
        print(f"    growth    : {p5.growth_wins:.4f}")
        print(f"    honesty   : {p5.honesty_wins:.4f}")
        drift_color = _GREEN if p5.drift_magnitude < 0.05 else (_YELLOW if p5.drift_magnitude < 0.15 else _RED)
        print(f"    drift_magnitude : {_c(f'{p5.drift_magnitude:.4f}', drift_color)}")
        if p5.extinction_risk_drives:
            print(_err("    EXTINCTION RISK", ", ".join(p5.extinction_risk_drives)))
            print(_c("    >> INV-017 applies: drive rolling mean < 0.05. Investigate immediately.", _RED))
        if p5.error:
            print(_warn("    note", p5.error))

    # Errors
    if eval_result.errors:
        print(f"\n  {_c('Pillar errors', _YELLOW)}")
        for k, v in eval_result.errors.items():
            print(f"    {k}: {v}")

    print(f"\n  {_dim(f'evaluation_id : {eval_result.evaluation_id}')}")
    print(f"  {_dim(f'evaluated_at  : {eval_result.evaluated_at_iso}')}\n")

    # JSON output option
    if args.json_output:
        print(json.dumps(eval_result.to_dict(), indent=2))

    return 0


# ─── Command: shadow-snapshot ─────────────────────────────────────────────────


async def cmd_shadow_snapshot(args: argparse.Namespace) -> int:
    """Take a shadow snapshot of current population state."""
    print(_header("Shadow Snapshot"))

    # Try via live API first
    try:
        result = await _post("/benchmarks/shadow-snapshot", {}, timeout=10.0)
        snapshot_id = result.get("snapshot_id", "")
        print(_ok("  snapshot_id", snapshot_id))
        print(_ok("  total_observables", result.get("total_observables", "?")))
        print(_ok("  novelty_rate", result.get("novelty_rate", "?")))
        print(_ok("  diversity_index", result.get("diversity_index", "?")))
        print()
        print(_dim(f"  Run later: python -m cli.evaluate shadow-delta {snapshot_id}"))
        return 0
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            print(_warn("  API endpoint not wired yet - using Redis directly", ""))
        else:
            print(_err("  API error", str(exc)))
            return 1
    except Exception:
        print(_dim("  Organism not reachable - using Redis directly..."))

    # Fallback: direct Redis + local tracker
    redis = await _get_redis()
    if redis is None:
        return 1

    try:
        from systems.benchmarks.shadow_reset import ShadowResetController

        instance_id = args.instance_id or "cli-snapshot"
        ctrl = ShadowResetController(instance_id=instance_id, redis=redis)
        snapshot_id = await ctrl.take_shadow_snapshot()
        print(_ok("  snapshot_id", snapshot_id))
        print(_dim("  (No live tracker - snapshot captured zeros; useful as timestamp anchor)"))
        print()
        print(_dim(f"  Run later: python -m cli.evaluate shadow-delta {snapshot_id}"))
        return 0
    except Exception as exc:
        print(_err("  snapshot failed", str(exc)))
        return 1
    finally:
        try:
            await redis.aclose()
        except Exception:
            pass


# ─── Command: shadow-delta ────────────────────────────────────────────────────


async def cmd_shadow_delta(args: argparse.Namespace) -> int:
    """Compute adaptive delta from a previous shadow snapshot."""
    snapshot_id: str = args.snapshot_id

    print(_header(f"Shadow Delta - {snapshot_id[:16]}..."))

    # Try via live API first
    try:
        result = await _post(
            "/benchmarks/shadow-delta",
            {"snapshot_id": snapshot_id},
            timeout=15.0,
        )
        _print_shadow_delta(result)
        return 0
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            print(_warn("  API endpoint not wired yet - using Redis directly", ""))
        else:
            print(_err("  API error", str(exc)))
            return 1
    except Exception:
        print(_dim("  Organism not reachable - using Redis directly..."))

    # Fallback: direct Redis
    redis = await _get_redis()
    if redis is None:
        return 1

    try:
        from systems.benchmarks.shadow_reset import ShadowResetController

        instance_id = args.instance_id or "cli-snapshot"
        ctrl = ShadowResetController(instance_id=instance_id, redis=redis)
        result_obj = await ctrl.compute_shadow_delta(snapshot_id)
        result = {
            "snapshot_id": result_obj.snapshot_id,
            "activity_drop_pct": result_obj.activity_drop_pct,
            "diversity_change_pct": result_obj.diversity_change_pct,
            "jaccard_overlap": result_obj.jaccard_overlap,
            "is_adaptive": result_obj.is_adaptive,
            "elapsed_seconds": result_obj.elapsed_seconds,
            "diversity_recovery_time": result_obj.diversity_recovery_time,
            "current_novelty_rate": result_obj.current_novelty_rate,
            "snapshot_novelty_rate": result_obj.snapshot_novelty_rate,
            "computed_at_iso": result_obj.computed_at_iso,
        }
        _print_shadow_delta(result)
        return 0
    except ValueError as exc:
        print(_err("  snapshot not found", str(exc)))
        return 1
    except Exception as exc:
        print(_err("  delta failed", str(exc)))
        return 1
    finally:
        try:
            await redis.aclose()
        except Exception:
            pass


def _print_shadow_delta(result: dict[str, Any]) -> None:
    drop = result.get("activity_drop_pct", 0.0)
    is_adaptive = result.get("is_adaptive", False)
    jaccard = result.get("jaccard_overlap", 1.0)
    elapsed = result.get("elapsed_seconds", 0.0)

    drop_color = _GREEN if drop > 50.0 else (_YELLOW if drop > 20.0 else _DIM)
    adaptive_label = (_c("YES - dynamics are adaptive, not drift", _GREEN) if is_adaptive
                      else _c("NO - activity unchanged (possible drift)", _YELLOW))

    print(f"  {_BOLD}is_adaptive{_RESET}              : {adaptive_label}")
    print(f"  {_BOLD}activity_drop_pct{_RESET}        : {_c(f'{drop:+.2f}%', drop_color)}")
    print(f"  {_BOLD}diversity_change_pct{_RESET}     : {result.get('diversity_change_pct', 0):+.2f}%")
    print(f"  {_BOLD}jaccard_overlap{_RESET}          : {jaccard:.4f}  {_dim('(low = many new types)')}")
    print(f"  {_BOLD}current_novelty_rate{_RESET}     : {result.get('current_novelty_rate', 0):.4f}")
    print(f"  {_BOLD}snapshot_novelty_rate{_RESET}    : {result.get('snapshot_novelty_rate', 0):.4f}")
    recovery = result.get("diversity_recovery_time")
    rec_str = f"{recovery:.1f}s" if recovery is not None else "not yet recovered"
    print(f"  {_BOLD}diversity_recovery_time{_RESET}  : {rec_str}")
    print(f"  {_BOLD}elapsed{_RESET}                  : {elapsed:.1f}s")
    print()
    print(_dim("  Bible §6.4: >50% activity drop = dynamics genuinely adaptive."))
    print(_dim("  Near-zero drop = drift, not adaptation. Bedau & Packard (1992)."))
    print()


# ─── Command: learning-velocity ───────────────────────────────────────────────


async def cmd_learning_velocity(args: argparse.Namespace) -> int:
    """
    Compute learning velocity from historical evaluation data.

    Fetches the per-month evaluation history from the organism's API or Neo4j.
    Falls back to reading from a local JSON file if the organism is unreachable.
    """
    print(_header("Learning Velocity (Pillar 4)"))

    history: list[dict[str, Any]] = []

    # Try to fetch from live API
    try:
        result = await _get("/benchmarks/evaluation-history", timeout=10.0)
        history = result.get("history", [])
        print(_dim(f"  Fetched {len(history)} data points from organism API"))
    except Exception:
        print(_dim("  Organism not reachable"))

    # Try local file fallback
    if not history and args.history_file:
        import pathlib
        p = pathlib.Path(args.history_file)
        if p.exists():
            try:
                with open(p) as fh:
                    history = json.load(fh)
                print(_dim(f"  Loaded {len(history)} data points from {p}"))
            except Exception as exc:
                print(_err("  cannot read history file", str(exc)))

    if not history:
        print(_warn("  No history data available", ""))
        print(_dim("  Format: JSON array of {month: int, score: float} objects"))
        print(_dim("  Pass --history-file path/to/history.json or start the organism"))
        return 1

    # Compute velocity
    try:
        from systems.benchmarks.evaluation_protocol import EvaluationProtocol

        proto = EvaluationProtocol(instance_id="cli")
        result_obj = await proto.measure_learning_velocity(history)

        vel = result_obj.velocity
        vel_color = _GREEN if result_obj.is_accelerating else (_YELLOW if not result_obj.is_plateaued else _RED)

        print(f"  {_BOLD}velocity{_RESET}           : {_c(f'{vel:+.6f} /month', vel_color)}")
        print(f"  {_BOLD}is_accelerating{_RESET}    : {_c(str(result_obj.is_accelerating), _GREEN if result_obj.is_accelerating else _DIM)}")
        print(f"  {_BOLD}is_plateaued{_RESET}       : {_c(str(result_obj.is_plateaued), _RED if result_obj.is_plateaued else _DIM)}")
        print(f"  {_BOLD}predicted_month_12{_RESET} : {result_obj.predicted_month_12:.4f}")
        print(f"  {_BOLD}fit_method{_RESET}         : {result_obj.fit_method}")
        print(f"  {_BOLD}n_data_points{_RESET}      : {result_obj.n_data_points}")

        if result_obj.is_plateaued:
            print()
            print(_c("  PLATEAU WARNING: velocity < 0.5%/month", _YELLOW))
            print(_dim("  Bible §3.4: investigate plasticity loss (Dohare et al., Nature 2024)"))
            print(_dim("  Consider: SVD pruning, SuRe reset, or Tier 3 quarterly retrain"))

        if result_obj.is_stub and result_obj.error:
            print(_warn("\n  note", result_obj.error))

        print()

    except ImportError as exc:
        print(_err("  cannot import evaluation framework", str(exc)))
        return 1

    return 0


# ─── Entry point ───────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.evaluate",
        description="EcodiaOS 5-Pillar Evaluation & Shadow-Reset CLI",
    )
    parser.add_argument(
        "--api-url",
        default=API_BASE,
        help="Organism API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--instance-id",
        default=None,
        help="Instance ID for Redis keys (default: auto-detect from API)",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Print full JSON result after human-readable display",
    )

    sub = parser.add_subparsers(dest="command")

    # monthly
    p_monthly = sub.add_parser(
        "monthly", help="Run full 5-pillar monthly evaluation"
    )
    p_monthly.add_argument(
        "--month", type=int, default=0, help="Month number (1=Month 1, etc.)"
    )
    p_monthly.add_argument(
        "--re-version", default="unknown", help="RE model version tag"
    )
    p_monthly.add_argument(
        "--eval-dir", default=None, help="Path to evaluation test set directory"
    )
    p_monthly.add_argument(
        "--json-output", action="store_true", help="Print full JSON result"
    )

    # shadow-snapshot
    sub.add_parser("shadow-snapshot", help="Take a shadow snapshot of population state")

    # shadow-delta
    p_delta = sub.add_parser(
        "shadow-delta", help="Compute adaptive delta from a shadow snapshot"
    )
    p_delta.add_argument("snapshot_id", help="Snapshot ID returned by shadow-snapshot")

    # learning-velocity
    p_vel = sub.add_parser(
        "learning-velocity", help="Compute learning velocity from monthly history"
    )
    p_vel.add_argument(
        "--history-file",
        default=None,
        help="Path to JSON file with [{month, score}] history",
    )

    return parser


async def _main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # Override API_BASE if provided
    global API_BASE
    if hasattr(args, "api_url") and args.api_url:
        API_BASE = args.api_url

    # Add sys.path so local imports work when running as python -m cli.evaluate
    import pathlib
    backend_dir = pathlib.Path(__file__).resolve().parent.parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    if args.command == "monthly":
        return await cmd_monthly(args)
    elif args.command == "shadow-snapshot":
        return await cmd_shadow_snapshot(args)
    elif args.command == "shadow-delta":
        return await cmd_shadow_delta(args)
    elif args.command == "learning-velocity":
        return await cmd_learning_velocity(args)
    else:
        parser.print_help()
        return 0


def main() -> None:
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
