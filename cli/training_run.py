#!/usr/bin/env python3
"""
EcodiaOS — Continual Learning CLI

Interact with the ContinualLearningOrchestrator from the command line.
The organism does NOT need to be running — this CLI connects directly to
Neo4j and Redis to check state and trigger training runs.

Usage:
    # Dry-run: check if training should trigger
    python -m cli.training_run check

    # Force a Tier 2 training run immediately
    python -m cli.training_run run

    # Force run with a custom reason label
    python -m cli.training_run run --reason "manual_review"

    # Show training run history from Redis
    python -m cli.training_run history

Required environment variables:
    ECODIAOS_NEO4J_URI       — e.g. neo4j+s://xxx.databases.neo4j.io
    ECODIAOS_NEO4J_PASSWORD  — Neo4j password

Optional:
    ECODIAOS_NEO4J_USERNAME  — default "neo4j"
    ECODIAOS_REDIS_URL       — default "redis://localhost:6379"
    ECODIAOS_RE_VLLM_URL     — vLLM endpoint (for adapter deployment)
    ECODIAOS_RE_MODEL        — model name on vLLM
    RE_TRAINING_EXPORT_DIR   — local JSONL output dir
    RE_BASE_MODEL            — base model for training (default: Qwen/Qwen3-8B)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Load .env before any os.getenv() calls — same pattern as all other EOS CLIs
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

# ─── ANSI colours ─────────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_DIM    = "\033[2m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}"


def _ok(text: str) -> str:    return _c(text, _GREEN)
def _warn(text: str) -> str:  return _c(text, _YELLOW)
def _err(text: str) -> str:   return _c(text, _RED)
def _head(text: str) -> str:  return _c(text, _BOLD + _CYAN)
def _dim(text: str) -> str:   return _c(text, _DIM)


# ─── Infrastructure bootstrap ─────────────────────────────────────────────────


def _build_neo4j() -> Any:
    """Construct Neo4jClient from env. Caller must call connect()."""
    from clients.neo4j import Neo4jClient
    from config import Neo4jConfig

    uri = os.environ.get("ECODIAOS_NEO4J_URI", "").strip()
    password = os.environ.get("ECODIAOS_NEO4J_PASSWORD", "").strip()
    username = os.environ.get("ECODIAOS_NEO4J_USERNAME", "neo4j").strip()
    database = os.environ.get("ECODIAOS_NEO4J_DATABASE", "neo4j").strip()

    if not uri:
        print(_err("ERROR: ECODIAOS_NEO4J_URI is not set."), file=sys.stderr)
        sys.exit(1)
    if not password:
        print(_err("ERROR: ECODIAOS_NEO4J_PASSWORD is not set."), file=sys.stderr)
        sys.exit(1)

    cfg = Neo4jConfig(uri=uri, username=username, password=password, database=database)
    return Neo4jClient(cfg)


async def _build_redis() -> Any | None:
    """Connect to Redis. Returns None on failure (non-fatal)."""
    redis_url = os.environ.get("ECODIAOS_REDIS_URL", "redis://localhost:6379")
    try:
        import redis.asyncio as aioredis  # type: ignore[import]
        r = aioredis.from_url(redis_url)
        await r.ping()
        return r
    except Exception as exc:
        print(_warn(f"  Redis unavailable ({exc}) — history/Thompson check will be skipped."))
        return None


def _build_re_service() -> Any | None:
    """Build ReasoningEngineService for adapter deployment. Returns None if disabled."""
    if os.environ.get("ECODIAOS_RE_ENABLED", "true").lower() in {"false", "0", "no"}:
        return None
    try:
        from systems.reasoning_engine.service import ReasoningEngineService
        return ReasoningEngineService()
    except Exception as exc:
        print(_warn(f"  RE service unavailable ({exc}) — adapter will not be deployed."))
        return None


async def _build_orchestrator(neo4j: Any, redis: Any, re_service: Any) -> Any:
    """Build and initialize ContinualLearningOrchestrator."""
    from systems.reasoning_engine.continual_learning import ContinualLearningOrchestrator
    from systems.reasoning_engine.training_data_extractor import TrainingDataExtractor

    extractor = TrainingDataExtractor(neo4j=neo4j)
    orch = ContinualLearningOrchestrator(re_service=re_service, extractor=extractor)
    if redis is not None:
        orch.set_redis(redis)
    await orch.initialize()
    return orch


# ─── Subcommand: check ────────────────────────────────────────────────────────


async def _cmd_check(args: argparse.Namespace) -> int:
    """Dry-run: print whether training should trigger and why."""
    print(_head("\n=== EcodiaOS Continual Learning — Trigger Check ===\n"))

    neo4j = _build_neo4j()
    redis = await _build_redis()
    re_service = _build_re_service()

    try:
        await neo4j.connect()
        orch = await _build_orchestrator(neo4j, redis, re_service)

        # Show current state
        status = await orch.get_status()
        last = status.get("last_train_at")
        print(f"  Last train    : {_ok(last) if last else _dim('never')}")
        print(f"  Total runs    : {status['total_runs']}")
        print(f"  Deployed runs : {status['deployed_runs']}")
        print(f"  Failed runs   : {status['failed_runs']}")
        print()

        # Stream counts
        from systems.reasoning_engine.training_data_extractor import TrainingDataExtractor
        extractor = TrainingDataExtractor(neo4j=neo4j)
        counts = await extractor.stream_counts(lookback_days=14)
        total = sum(v for v in counts.values() if v >= 0)
        print(f"  Stream counts (14-day window):")
        for stream, count in counts.items():
            colour = _ok if count > 50 else (_warn if count > 0 else _dim)
            print(f"    {stream:<40} {colour(str(count))}")
        print(f"  Total available : {_ok(str(total)) if total > 50 else _warn(str(total))}")
        print()

        should, reason = await orch.should_train()
        if should:
            print(_ok(f"  TRIGGER: YES — reason: {reason}"))
            print(_dim("  Run `python -m cli.training_run run` to start training."))
        else:
            print(_dim(f"  TRIGGER: no ({reason})"))

        return 0

    except Exception as exc:
        print(_err(f"\nERROR: {exc}"), file=sys.stderr)
        return 1
    finally:
        await neo4j.close()
        if redis is not None:
            await redis.aclose()


# ─── Subcommand: run ──────────────────────────────────────────────────────────


async def _cmd_run(args: argparse.Namespace) -> int:
    """Force a Tier 2 training run immediately."""
    reason = getattr(args, "reason", "manual_cli")
    print(_head(f"\n=== EcodiaOS Continual Learning — Forced Tier 2 Run ==="))
    print(f"  Reason: {reason}\n")

    neo4j = _build_neo4j()
    redis = await _build_redis()
    re_service = _build_re_service()

    if re_service is not None:
        try:
            await re_service.initialize()
        except Exception:
            pass  # non-fatal — training will still proceed, adapter won't deploy

    try:
        await neo4j.connect()
        orch = await _build_orchestrator(neo4j, redis, re_service)

        print("  Starting Tier 2 run (this may take up to 2 hours on GPU)...")
        print(_dim("  Training subprocess output will appear on stdout.\n"))

        run = await orch.run_tier2(reason)

        print()
        print(_head("  ── Run Summary ──────────────────────────────"))
        print(f"  run_id        : {run.run_id}")
        print(f"  examples_used : {run.examples_used}")
        print(f"  started_at    : {run.started_at.isoformat()}")
        print(f"  completed_at  : {run.completed_at.isoformat() if run.completed_at else 'n/a'}")
        print(f"  eval_loss     : {run.eval_loss if run.eval_loss is not None else _dim('n/a')}")
        print(f"  adapter_path  : {run.adapter_path or _dim('none')}")

        if run.deployed:
            print(f"  deployed      : {_ok('YES')}")
        elif run.error:
            print(f"  deployed      : {_warn('NO')} — {run.error}")
        else:
            print(f"  deployed      : {_dim('NO — training skipped (insufficient data)')}")

        return 0 if not run.error or run.deployed else 1

    except Exception as exc:
        print(_err(f"\nERROR: {exc}"), file=sys.stderr)
        return 1
    finally:
        await neo4j.close()
        if redis is not None:
            await redis.aclose()


# ─── Subcommand: history ──────────────────────────────────────────────────────


async def _cmd_history(args: argparse.Namespace) -> int:
    """Print training run history from Redis."""
    print(_head("\n=== EcodiaOS Continual Learning — Training History ===\n"))

    redis = await _build_redis()
    if redis is None:
        print(_err("  Cannot show history — Redis is unavailable."))
        return 1

    try:
        raw = await redis.get("eos:re:training_runs")
        if not raw:
            print(_dim("  No training runs recorded in Redis."))
            return 0

        runs_data = json.loads(raw.decode())
        if not runs_data:
            print(_dim("  No training runs recorded."))
            return 0

        limit = getattr(args, "limit", 20)
        print(f"  Showing {min(len(runs_data), limit)} of {len(runs_data)} runs:\n")

        for run_dict in runs_data[-limit:]:
            status_str = (
                _ok("DEPLOYED") if run_dict.get("deployed")
                else (_err("FAILED") if run_dict.get("error") else _warn("SKIPPED"))
            )
            loss_str = f"{run_dict['eval_loss']:.4f}" if run_dict.get("eval_loss") else _dim("n/a")
            duration = ""
            if run_dict.get("started_at") and run_dict.get("completed_at"):
                from datetime import datetime
                t0 = datetime.fromisoformat(run_dict["started_at"])
                t1 = datetime.fromisoformat(run_dict["completed_at"])
                secs = int((t1 - t0).total_seconds())
                duration = f"{secs // 60}m{secs % 60}s"

            print(
                f"  {run_dict['run_id']:<36}  "
                f"T{run_dict['tier']}  "
                f"{run_dict['started_at'][:19]}  "
                f"ex={run_dict['examples_used']:<5}  "
                f"loss={loss_str:<8}  "
                f"{status_str:<12}  "
                f"{_dim(duration)}"
            )
            if run_dict.get("error"):
                print(f"    {_dim('error:')} {run_dict['error'][:80]}")

        return 0

    except Exception as exc:
        print(_err(f"\nERROR: {exc}"), file=sys.stderr)
        return 1
    finally:
        await redis.aclose()


# ─── Subcommand: status ───────────────────────────────────────────────────────


async def _cmd_status(args: argparse.Namespace) -> int:
    """Show RE performance status: success rate, Thompson params, training state."""
    print(_head("\n=== EcodiaOS RE Performance Status ===\n"))

    redis = await _build_redis()

    if redis is None:
        print(_err("  Redis unavailable — cannot read RE performance metrics."))
        return 1

    try:
        # ── RE success rate keys ─────────────────────────────────────────
        rate_7d_raw = await redis.get("eos:re:success_rate_7d")
        thompson_rate_raw = await redis.get("eos:re:thompson_success_rate")

        rate_7d = float(rate_7d_raw.decode()) if rate_7d_raw else None
        thompson_rate = float(thompson_rate_raw.decode()) if thompson_rate_raw else None

        def _rate_colour(r: float | None) -> str:
            if r is None:
                return _dim("n/a")
            if r >= 0.75:
                return _ok(f"{r:.3f}")
            if r >= 0.60:
                return _warn(f"{r:.3f}")
            return _err(f"{r:.3f}")

        print(f"  RE success rate (7d)   : {_rate_colour(rate_7d)}")
        print(f"  Thompson success rate  : {_rate_colour(thompson_rate)}")
        print()

        # ── Thompson sampler raw params ──────────────────────────────────
        sampler_raw = await redis.hgetall("nova:thompson_sampler")
        if sampler_raw:
            def _fv(k: str) -> str:
                v = sampler_raw.get(k) or sampler_raw.get(k.encode())
                if v is None:
                    return _dim("?")
                return f"{float(v.decode() if isinstance(v, bytes) else v):.2f}"

            c_alpha = _fv("claude_alpha")
            c_beta  = _fv("claude_beta")
            re_alpha = _fv("re_alpha")
            re_beta  = _fv("re_beta")
            print(f"  Thompson params:")
            print(f"    claude  α={c_alpha}  β={c_beta}")
            print(f"    re      α={re_alpha}  β={re_beta}")
        else:
            print(f"  Thompson params        : {_dim('not yet persisted')}")
        print()

        # ── Training run history (last 3) ────────────────────────────────
        runs_raw = await redis.get("eos:re:training_runs")
        if runs_raw:
            import json as _json
            runs = _json.loads(runs_raw.decode())[-3:]
            print(f"  Recent training runs ({len(runs)}):")
            for r in runs:
                status_str = (
                    _ok("DEPLOYED") if r.get("deployed")
                    else (_err("FAILED") if r.get("error") else _warn("SKIPPED"))
                )
                loss_str = f"{r['eval_loss']:.4f}" if r.get("eval_loss") else _dim("n/a")
                print(
                    f"    {r['run_id'][:20]}…  "
                    f"T{r['tier']}  "
                    f"ex={r['examples_used']:<5}  "
                    f"loss={loss_str:<8}  "
                    f"{status_str}"
                )
        else:
            print(f"  Training runs          : {_dim('none recorded')}")
        print()

        # ── RE availability check ────────────────────────────────────────
        re_service = _build_re_service()
        if re_service is not None:
            try:
                await re_service.initialize()
                available = getattr(re_service, "is_available", False)
                model = getattr(re_service, "_model", "unknown")
                print(f"  RE vLLM available      : {_ok('YES') if available else _warn('NO')}")
                print(f"  RE model               : {model}")
            except Exception as exc:
                print(f"  RE vLLM available      : {_warn(f'NO ({exc})')}")
        else:
            print(f"  RE vLLM available      : {_dim('disabled (ECODIAOS_RE_ENABLED=false)')}")

        return 0

    except Exception as exc:
        print(_err(f"\nERROR: {exc}"), file=sys.stderr)
        return 1
    finally:
        await redis.aclose()


# ─── Argument parser ──────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cli.training_run",
        description="EcodiaOS Continual Learning CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # check
    sub.add_parser("check", help="Dry-run: check if training should trigger")

    # run
    run_p = sub.add_parser("run", help="Force a Tier 2 training run now")
    run_p.add_argument(
        "--reason",
        default="manual_cli",
        help="Trigger reason label stored in the run record (default: manual_cli)",
    )

    # history
    hist_p = sub.add_parser("history", help="Show training run history from Redis")
    hist_p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Max runs to display (default: 20)",
    )

    # status
    sub.add_parser("status", help="Show RE performance status (success rate, Thompson params, adapter)")

    # clear-halt
    sub.add_parser(
        "clear-halt",
        help="Clear a persisted training halt (set by red-team kill switch or operator). "
             "Resumes daily Tier 2 trigger checks.",
    )

    return parser


# ─── Entry point ──────────────────────────────────────────────────────────────


async def _cmd_clear_halt(args: argparse.Namespace) -> int:
    """Clear the persisted training halt flag from Redis."""
    print(_head("\n=== EcodiaOS Continual Learning — Clear Training Halt ===\n"))

    redis = await _build_redis()
    if redis is None:
        print(_err("  Redis unavailable — cannot clear halt flag."))
        return 1

    try:
        _HALT_KEY = "eos:re:training_halted"
        existing = await redis.get(_HALT_KEY)
        if existing is None:
            print(_dim("  No training halt is currently set."))
            return 0

        reason = existing.decode() if isinstance(existing, bytes) else existing
        print(f"  Current halt reason : {_warn(reason)}")
        await redis.delete(_HALT_KEY)
        print(_ok("  Training halt cleared. Daily Tier 2 trigger checks will resume."))
        return 0

    except Exception as exc:
        print(_err(f"\nERROR: {exc}"), file=sys.stderr)
        return 1
    finally:
        await redis.aclose()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    handlers = {
        "check": _cmd_check,
        "run": _cmd_run,
        "history": _cmd_history,
        "status": _cmd_status,
        "clear-halt": _cmd_clear_halt,
    }

    handler = handlers[args.command]
    exit_code = asyncio.run(handler(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
