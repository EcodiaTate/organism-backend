"""
EcodiaOS - RE Model Post-Training Evaluator

Closes the feedback loop between the RE training pipeline and the organism's
self-model.  After every RE_TRAINING_EXPORT_COMPLETE event (and on a 24-hour
safety-net schedule) the evaluator:

  1. Pulls up to 20 recent RETrainingDatapoint records for each category
     from S3 / local filesystem (same path the exporter wrote to).
  2. Replays the original prompt through the live RE model (VLLMProvider).
  3. Scores the response with a per-category heuristic.
  4. Compares to the EvaluationBaseline stored in Redis.
  5. Emits BENCHMARKS_KPI per category, INCIDENT_DETECTED on regression,
     RE_TRAINING_EXAMPLE on strong improvement, and NOVA_GOAL_INJECTED when
     the overall health score crosses 0.85.

Redis keys (all namespaced by instance_id):
  "re_eval:baseline:{instance_id}:{category}"
      → JSON: { "pass_rate": float, "sample_count": int, "timestamp": str }
  "re_eval:health_score:{instance_id}"
      → float string (overall weighted health score)
  "re_eval:last_run:{instance_id}"
      → ISO-8601 timestamp of last completed evaluation
"""

from __future__ import annotations

import asyncio
import json
import os
import re as _re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from clients.redis import RedisClient
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("core.re_evaluator")

# ── Categories evaluated ────────────────────────────────────────────────────

_CATEGORIES: list[str] = [
    "build_error",
    "hot_swap_failure",
    "hot_swap_rollback",
    "crash_pattern",
    "general_repair",
    "code_generation",
]

# Weighted average weights for the overall health score (sum = 1.0)
_CATEGORY_WEIGHTS: dict[str, float] = {
    "build_error": 0.30,
    "hot_swap_failure": 0.20,
    "hot_swap_rollback": 0.00,  # not in task-spec weights - distributed elsewhere
    "crash_pattern": 0.30,
    "general_repair": 0.10,
    "code_generation": 0.10,
}

# Spec weights do not include hot_swap_rollback; treat it like general_repair
_EFFECTIVE_WEIGHTS: dict[str, float] = {
    "build_error": 0.30,
    "hot_swap_failure": 0.20,
    "hot_swap_rollback": 0.05,
    "crash_pattern": 0.25,
    "general_repair": 0.10,
    "code_generation": 0.10,
}

_MAX_SAMPLES_PER_CATEGORY: int = int(os.environ.get("RE_EVAL_MAX_SAMPLES", "20"))
_EVAL_INTERVAL_S: int = int(os.environ.get("RE_EVAL_INTERVAL_S", "86400"))  # 24h safety-net

# S3 / local config mirrors re_training_exporter constants
_S3_BUCKET: str = os.environ.get("RE_TRAINING_S3_BUCKET", "ecodiaos-re-training")
_S3_PREFIX: str = os.environ.get("RE_TRAINING_S3_PREFIX", "batches/")
_LOCAL_EXPORT_DIR: str = os.environ.get("RE_TRAINING_EXPORT_DIR", "data/re_training_batches")

# Regression / improvement thresholds — env-overridable so Evo can tune them
_REGRESSION_THRESHOLD: float = float(os.environ.get("RE_EVAL_REGRESSION_THRESHOLD", "-0.05"))
_IMPROVEMENT_THRESHOLD: float = float(os.environ.get("RE_EVAL_IMPROVEMENT_THRESHOLD", "0.10"))
_DIRECTION_FLAT_BAND: float = float(os.environ.get("RE_EVAL_FLAT_BAND", "0.02"))
_HEALTH_GOAL_THRESHOLD: float = float(os.environ.get("RE_EVAL_HEALTH_GOAL_THRESHOLD", "0.85"))


# ── Error-pattern dictionary for repair/hot_swap categories ────────────────

# These strings represent known failure patterns that a good model should
# *avoid* suggesting in its repair output.  Presence of the pattern in the
# response lowers the score.

_KNOWN_ERROR_PATTERNS: dict[str, list[str]] = {
    "build_error": [
        "import error",
        "modulenotfounderror",
        "syntaxerror",
        "indentationerror",
        "nameerror: name",
    ],
    "hot_swap_failure": [
        "lora_path not found",
        "adapter not loaded",
        "failed to load",
        "connection refused",
    ],
    "hot_swap_rollback": [
        "adapter cid",
        "reverted to",
        "rollback failed",
    ],
    "crash_pattern": [
        "segmentation fault",
        "killed",
        "out of memory",
        "oom",
        "core dumped",
    ],
    "general_repair": [
        "traceback (most recent call last)",
        "exception:",
        "error:",
    ],
}


# ── Scoring helpers ─────────────────────────────────────────────────────────

def _score_code_generation(response: str) -> float:
    """Attempt compile(); 1.0 if clean, partial credit for near-valid code."""
    code = response.strip()
    # Strip markdown code fences if present
    if code.startswith("```"):
        lines = code.splitlines()
        # Remove first line (```python / ```) and last closing ```
        inner = "\n".join(
            l for l in lines[1:]
            if l.strip() != "```"
        )
        code = inner.strip()
    if not code:
        return 0.0
    try:
        compile(code, "<re_eval>", "exec")
        return 1.0
    except SyntaxError:
        # Partial score if the response is non-trivial but has a syntax issue
        if len(code) > 50:
            return 0.3
        return 0.0
    except Exception:
        return 0.2


def _score_repair_response(response: str, category: str, datapoint: dict[str, Any]) -> float:
    """
    Heuristic scorer for repair categories.

    Checks:
    - Response does not contain known error patterns for this category (penalty)
    - Response contains actionable repair keywords (bonus)
    - Response is substantive (length check)
    """
    text = response.lower()
    patterns = _KNOWN_ERROR_PATTERNS.get(category, [])

    # Start with a passing baseline
    score = 0.7

    # Penalise each known error pattern found
    pattern_hits = sum(1 for p in patterns if p in text)
    score -= 0.15 * min(pattern_hits, 3)

    # Reward substantive content
    if len(text) > 100:
        score += 0.1
    if len(text) > 300:
        score += 0.05

    # Reward repair keywords
    repair_keywords = ["fix", "resolve", "patch", "update", "replace", "correct", "change"]
    keyword_hits = sum(1 for kw in repair_keywords if kw in text)
    score += 0.05 * min(keyword_hits, 2)

    return max(0.0, min(score, 1.0))


def _score_hot_swap(response: str, datapoint: dict[str, Any]) -> float:
    """
    Check that the response does not suggest the previously-failed adapter CID.

    The failed adapter CID is extracted from the original input_context
    (best-effort - returns neutral score if not found).
    """
    context = str(datapoint.get("input_context", "")).lower()
    # Attempt to find the failed CID mentioned in context
    cid_match = _re.search(r"(?:failed|old|previous)\s+(?:adapter\s+)?cid[:\s]+([a-zA-Z0-9]+)", context)
    failed_cid = cid_match.group(1).lower() if cid_match else None

    text = response.lower()
    if failed_cid and failed_cid in text:
        return 0.0  # Suggests the known-bad adapter - hard fail

    # Otherwise score as repair
    return _score_repair_response(response, "hot_swap_failure", datapoint)


def _score_response(response: str, category: str, datapoint: dict[str, Any]) -> float:
    """Dispatch to the appropriate scoring function for a category."""
    if category == "code_generation":
        return _score_code_generation(response)
    if category in ("hot_swap_failure", "hot_swap_rollback"):
        return _score_hot_swap(response, datapoint)
    # All other categories: repair heuristic
    return _score_repair_response(response, category, datapoint)


# ── Data loading ────────────────────────────────────────────────────────────

def _load_from_local(category: str, max_samples: int) -> list[dict[str, Any]]:
    """
    Load up to *max_samples* RETrainingDatapoint JSON records for *category*
    from the local export directory.  Returns records in descending
    recency order (newest files first).
    """
    export_dir = Path(_LOCAL_EXPORT_DIR)
    if not export_dir.exists():
        return []

    records: list[dict[str, Any]] = []

    # Sort files newest-first (filename starts with ISO timestamp)
    jsonl_files = sorted(export_dir.glob("*.jsonl"), reverse=True)

    for path in jsonl_files:
        if len(records) >= max_samples:
            break
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if len(records) >= max_samples:
                    break
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # example_type maps to category in RETrainingDatapoint
                if obj.get("example_type") == category:
                    records.append(obj)
        except Exception:
            continue

    return records


async def _load_from_s3(category: str, max_samples: int) -> list[dict[str, Any]]:
    """
    Attempt to load records from S3; fall back silently to empty list.

    Runs in a thread pool to avoid blocking the event loop.
    """
    def _s3_read() -> list[dict[str, Any]]:
        try:
            import boto3  # type: ignore[import]

            s3 = boto3.client("s3")
            # List recent objects under the configured prefix
            resp = s3.list_objects_v2(
                Bucket=_S3_BUCKET,
                Prefix=_S3_PREFIX,
                MaxKeys=50,
            )
            objects = resp.get("Contents", [])
            # Sort by LastModified descending
            objects.sort(key=lambda o: o.get("LastModified", 0), reverse=True)

            records: list[dict[str, Any]] = []
            for obj in objects:
                if len(records) >= max_samples:
                    break
                try:
                    body = s3.get_object(Bucket=_S3_BUCKET, Key=obj["Key"])["Body"].read()
                    for line in body.decode("utf-8").splitlines():
                        if len(records) >= max_samples:
                            break
                        if not line.strip():
                            continue
                        try:
                            o = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if o.get("example_type") == category:
                            records.append(o)
                except Exception:
                    continue
            return records
        except ImportError:
            return []
        except Exception:
            return []

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _s3_read)


async def _load_examples(category: str, max_samples: int) -> list[dict[str, Any]]:
    """
    Load up to *max_samples* datapoints for *category*, trying S3 first,
    falling back to local filesystem.
    """
    records = await _load_from_s3(category, max_samples)
    if not records:
        records = _load_from_local(category, max_samples)
    return records[:max_samples]


# ── REEvaluator ─────────────────────────────────────────────────────────────

class REEvaluator:
    """
    Post-training evaluator for the local Reasoning Engine.

    Triggered by RE_TRAINING_EXPORT_COMPLETE and on a 24-hour schedule.
    Evaluates per-category pass rates, compares to stored baselines, and
    emits BENCHMARKS_KPI / INCIDENT_DETECTED / RE_TRAINING_EXAMPLE / NOVA_GOAL_INJECTED
    events so Benchmarks and Thread can observe the organism learning.
    """

    def __init__(
        self,
        event_bus: Any | None = None,
        redis: Any | None = None,
        neo4j: Any | None = None,
        instance_id: str = "genesis",
    ) -> None:
        self._event_bus = event_bus
        self._redis: Any | None = redis
        self._neo4j: Any | None = neo4j
        self._instance_id = instance_id
        self._vllm: Any | None = None   # injected via set_vllm()

        self._running = False
        self._attached = False
        self._last_eval_ts: float = 0.0

        self._logger = logger.bind(system="re_evaluator", instance_id=instance_id)

    # ── Injection setters ───────────────────────────────────────────────────

    def set_event_bus(self, event_bus: Any) -> None:
        self._event_bus = event_bus

    def set_redis(self, redis: Any) -> None:
        self._redis = redis

    def set_neo4j(self, neo4j: Any) -> None:
        self._neo4j = neo4j

    def set_vllm(self, vllm: Any) -> None:
        """
        Wire the live VLLMProvider (ReasoningEngineService) so the evaluator
        can replay prompts through the actual RE model.
        """
        self._vllm = vllm

    # ── Attachment / subscription ───────────────────────────────────────────

    def attach(self) -> None:
        """Subscribe to RE_TRAINING_EXPORT_COMPLETE on the event bus."""
        if self._attached or self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEventType

            if hasattr(SynapseEventType, "RE_TRAINING_EXPORT_COMPLETE"):
                self._event_bus.subscribe(
                    SynapseEventType.RE_TRAINING_EXPORT_COMPLETE,
                    self._on_export_complete,
                )
                self._attached = True
                self._logger.info("re_evaluator_attached")
        except Exception as exc:
            self._logger.warning("re_evaluator_attach_failed", error=str(exc))

    async def _on_export_complete(self, event: Any) -> None:
        """Trigger evaluation immediately after every successful export."""
        self._logger.info("re_evaluator_triggered_by_export")
        try:
            await self._run_evaluation()
        except Exception as exc:
            self._logger.error("re_evaluator_export_trigger_failed", error=str(exc))

    # ── Background loop ─────────────────────────────────────────────────────

    async def run_loop(self) -> None:
        """
        24-hour safety-net loop.  Runs indefinitely; cancellation stops it
        cleanly.  The first run fires immediately after a 60-second warm-up
        so systems are ready.
        """
        self._running = True
        self._logger.info("re_evaluator_loop_started", interval_s=_EVAL_INTERVAL_S)
        await asyncio.sleep(60)  # warm-up: let exporter + systems settle

        while self._running:
            try:
                await self._run_evaluation()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._logger.error("re_evaluator_loop_error", error=str(exc))

            await asyncio.sleep(_EVAL_INTERVAL_S)

    # ── Core evaluation ─────────────────────────────────────────────────────

    async def _run_evaluation(self) -> None:
        """
        Full evaluation pass across all categories.

        Loads examples → replays prompts → scores → compares baseline →
        emits KPI events → updates health score.
        """
        if self._vllm is None:
            self._logger.info("re_evaluator_skipped_no_vllm")
            return

        # Guard: is RE available?
        is_available = getattr(self._vllm, "is_available", None)
        if callable(is_available):
            available = is_available()
        elif isinstance(is_available, bool):
            available = is_available
        else:
            available = True  # unknown - assume yes

        if not available:
            self._logger.info("re_evaluator_skipped_re_unavailable")
            return

        eval_start = time.monotonic()
        self._logger.info("re_evaluator_starting")

        category_results: dict[str, float] = {}

        for category in _CATEGORIES:
            pass_rate = await self._evaluate_category(category)
            if pass_rate is not None:
                category_results[category] = pass_rate

        if not category_results:
            self._logger.info("re_evaluator_no_results")
            return

        # Compare to baselines and emit KPI events
        await self._compare_and_emit(category_results)

        # Compute overall health score
        health_score = self._compute_health_score(category_results)
        await self._emit_health_score(health_score)

        # Persist last run timestamp
        self._last_eval_ts = time.time()
        if self._redis is not None:
            try:
                key = f"re_eval:last_run:{self._instance_id}"
                await self._redis.set(key, datetime.now(UTC).isoformat())
            except Exception:
                pass

        elapsed_ms = int((time.monotonic() - eval_start) * 1000)
        self._logger.info(
            "re_evaluator_complete",
            categories_evaluated=len(category_results),
            health_score=round(health_score, 4),
            elapsed_ms=elapsed_ms,
        )

    async def _evaluate_category(self, category: str) -> float | None:
        """
        Load examples, replay prompts, score responses.

        Returns the pass_rate [0, 1] for *category*, or None if no examples.
        """
        examples = await _load_examples(category, _MAX_SAMPLES_PER_CATEGORY)
        if not examples:
            self._logger.debug("re_evaluator_no_examples", category=category)
            return None

        successes = 0
        total = 0

        for dp in examples:
            instruction = str(dp.get("instruction", "")).strip()
            input_context = str(dp.get("input_context", "")).strip()
            if not instruction:
                continue

            prompt = instruction
            if input_context:
                prompt = f"{input_context}\n\n{instruction}"

            response_text = await self._query_vllm(prompt)
            if response_text is None:
                continue  # RE unavailable for this call

            score = _score_response(response_text, category, dp)
            # Score ≥ 0.5 counts as a pass
            if score >= 0.5:
                successes += 1
            total += 1

        if total == 0:
            return None

        pass_rate = successes / total
        self._logger.debug(
            "re_evaluator_category_result",
            category=category,
            pass_rate=round(pass_rate, 4),
            total=total,
        )
        return pass_rate

    async def _query_vllm(self, prompt: str) -> str | None:
        """
        Send a single prompt to the RE model.  Returns the response text or
        None if the call fails / times out.
        """
        try:
            from clients.llm import Message

            response = await asyncio.wait_for(
                self._vllm.generate(
                    system_prompt=None,
                    messages=[Message("user", prompt)],
                    max_tokens=512,
                    temperature=0.1,
                ),
                timeout=30.0,
            )
            return response.text
        except asyncio.TimeoutError:
            self._logger.debug("re_evaluator_query_timeout")
            return None
        except Exception as exc:
            self._logger.debug("re_evaluator_query_failed", error=str(exc))
            return None

    # ── Baseline comparison + event emission ────────────────────────────────

    async def _compare_and_emit(self, category_results: dict[str, float]) -> None:
        """
        For each evaluated category, load the Redis baseline, compute delta,
        emit BENCHMARKS_KPI.  If regression or strong improvement detected,
        emit additional events.
        """
        for category, new_pass_rate in category_results.items():
            baseline = await self._load_baseline(category)
            baseline_rate = baseline.get("pass_rate") if baseline else None

            if baseline_rate is not None:
                delta = new_pass_rate - baseline_rate
            else:
                delta = 0.0  # No baseline yet - first run

            # Direction
            if abs(delta) < _DIRECTION_FLAT_BAND:
                direction = "flat"
            elif delta > 0:
                direction = "up"
            else:
                direction = "down"

            # Emit per-category KPI
            await self._emit_event(
                "BENCHMARK_RE_PROGRESS",
                {
                    "kpi_name": f"re_model.{category}.pass_rate",
                    "value": new_pass_rate,
                    "delta": delta,
                    "direction": direction,
                    "category": category,
                    "baseline_pass_rate": baseline_rate,
                    "instance_id": self._instance_id,
                },
            )

            # Regression incident
            if baseline_rate is not None and delta < _REGRESSION_THRESHOLD:
                await self._emit_event(
                    "INCIDENT_DETECTED",
                    {
                        "title": f"RE model regressed on {category}",
                        "description": (
                            f"RE model regressed on {category}: "
                            f"{baseline_rate:.0%} → {new_pass_rate:.0%}"
                        ),
                        "severity": "HIGH",
                        "source_system": "re_evaluator",
                        "category": "model_regression",
                        "instance_id": self._instance_id,
                    },
                )
                self._logger.warning(
                    "re_evaluator_regression_detected",
                    category=category,
                    baseline=round(baseline_rate, 4),
                    new=round(new_pass_rate, 4),
                    delta=round(delta, 4),
                )

            # Strong improvement - organism celebrates its own learning
            if baseline_rate is not None and delta > _IMPROVEMENT_THRESHOLD:
                await self._emit_event(
                    "RE_TRAINING_EXAMPLE",
                    {
                        "source_system": "re_evaluator",
                        "instruction": (
                            f"Assess the quality improvement of the RE model on the "
                            f"{category} category after a CLoRA training cycle."
                        ),
                        "input_context": (
                            f"Pre-training pass rate: {baseline_rate:.0%}. "
                            f"Post-training pass rate: {new_pass_rate:.0%}."
                        ),
                        "output": (
                            f"Model improved {category} by {delta:.0%} "
                            f"(from {baseline_rate:.0%} to {new_pass_rate:.0%}). "
                            "This is evidence of successful continual learning."
                        ),
                        "outcome_quality": 1.0,
                        "category": "model_improvement",
                        "episode_id": f"re_eval_improvement:{category}:{int(time.time())}",
                        "reasoning_trace": (
                            "## Step 1: Observation\n"
                            f"Pass rate on {category} increased by {delta:.0%}.\n"
                            "## Step 2: Assessment\n"
                            "The CLoRA training cycle produced a measurable improvement "
                            "above the 10% threshold.\n"
                            "## Step 3: Significance\n"
                            "This confirms that the organism's training pipeline is "
                            "producing genuine capability improvements.\n"
                            "## Step 4: Constitutional check\n"
                            "Growth drive strongly aligned. No safety concerns.\n"
                            "## Step 5: Record\n"
                            "Emit this as a gold-tier training signal so future cycles "
                            "know this pattern leads to improvement."
                        ),
                        "instance_id": self._instance_id,
                    },
                )
                self._logger.info(
                    "re_evaluator_improvement_celebrated",
                    category=category,
                    delta=round(delta, 4),
                )

            # Update baseline in Redis
            await self._save_baseline(category, new_pass_rate, total=_MAX_SAMPLES_PER_CATEGORY)

    # ── Health score ────────────────────────────────────────────────────────

    def _compute_health_score(self, category_results: dict[str, float]) -> float:
        """
        Weighted average of all evaluated category pass rates.
        Uses _EFFECTIVE_WEIGHTS; missing categories do not contribute.
        """
        total_weight = 0.0
        weighted_sum = 0.0
        for category, pass_rate in category_results.items():
            w = _EFFECTIVE_WEIGHTS.get(category, 0.0)
            weighted_sum += w * pass_rate
            total_weight += w

        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight

    async def _emit_health_score(self, health_score: float) -> None:
        """Emit health score KPI and NOVA_GOAL_INJECTED if threshold exceeded."""
        # Load previous health score from Redis to compute delta for Thread's
        # _on_re_model_improved handler (which requires delta > 0.05).
        prev_health_score: float | None = None
        redis_key = f"re_eval:health_score:{self._instance_id}"
        if self._redis is not None:
            try:
                raw = await self._redis.get(redis_key)
                if raw is not None:
                    prev_health_score = float(raw)
            except Exception:
                pass

        delta = (health_score - prev_health_score) if prev_health_score is not None else 0.0
        if abs(delta) < 1e-6:
            direction = "flat"
        elif delta > 0:
            direction = "up"
        else:
            direction = "down"

        # Persist current health score to Redis (after reading prev, before emit)
        if self._redis is not None:
            try:
                await self._redis.set(redis_key, str(round(health_score, 6)))
            except Exception:
                pass

        # Emit KPI event with computed delta so Thread/_on_re_model_improved fires
        await self._emit_event(
            "BENCHMARK_RE_PROGRESS",
            {
                "kpi_name": "re_model.health_score",
                "value": health_score,
                "delta": round(delta, 6),
                "direction": direction,
                "category": "overall",
                "instance_id": self._instance_id,
            },
        )

        # Notify Thread / Nova if organism is learning well
        if health_score > _HEALTH_GOAL_THRESHOLD:
            await self._emit_event(
                "NOVA_GOAL_INJECTED",
                {
                    "goal": (
                        f"RE model performing at {health_score:.0%} - organism is learning"
                    ),
                    "priority": 0.6,
                    "source_system": "re_evaluator",
                    "goal_type": "SELF_GENERATED",
                    "instance_id": self._instance_id,
                },
            )
            self._logger.info(
                "re_evaluator_health_goal_injected",
                health_score=round(health_score, 4),
            )

    # ── Redis baseline helpers ──────────────────────────────────────────────

    async def _load_baseline(self, category: str) -> dict[str, Any] | None:
        """Load the stored EvaluationBaseline for *category* from Redis."""
        if self._redis is None:
            return None
        try:
            key = f"re_eval:baseline:{self._instance_id}:{category}"
            raw = await self._redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception:
            return None

    async def _save_baseline(self, category: str, pass_rate: float, total: int) -> None:
        """Persist a new EvaluationBaseline for *category* to Redis."""
        if self._redis is None:
            return
        try:
            key = f"re_eval:baseline:{self._instance_id}:{category}"
            payload = json.dumps({
                "pass_rate": round(pass_rate, 6),
                "sample_count": total,
                "timestamp": datetime.now(UTC).isoformat(),
            })
            await self._redis.set(key, payload)
        except Exception as exc:
            self._logger.debug("re_evaluator_baseline_save_failed", error=str(exc))

    # ── Generic event emitter ───────────────────────────────────────────────

    async def _emit_event(self, event_type_name: str, data: dict[str, Any]) -> None:
        """
        Emit a SynapseEvent via the event bus.  Non-fatal.

        *event_type_name* is the string attribute name on SynapseEventType,
        e.g. "INCIDENT_DETECTED", "BENCHMARK_RE_PROGRESS".
        """
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            if not hasattr(SynapseEventType, event_type_name):
                self._logger.debug(
                    "re_evaluator_unknown_event_type", event_type=event_type_name
                )
                return

            event_type = getattr(SynapseEventType, event_type_name)
            await self._event_bus.emit(
                SynapseEvent(
                    event_type=event_type,
                    source_system="re_evaluator",
                    data=data,
                )
            )
        except Exception as exc:
            self._logger.debug(
                "re_evaluator_emit_failed",
                event_type=event_type_name,
                error=str(exc),
            )

    # ── Stats ───────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "attached": self._attached,
            "last_eval_ts": self._last_eval_ts,
            "vllm_wired": self._vllm is not None,
            "instance_id": self._instance_id,
        }
