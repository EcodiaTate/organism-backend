"""
EcodiaOS -- Axon BountyHuntExecutor (Phase 16b -- Full Hunt Loop)

A unified executor that performs the complete bounty-hunting cycle in a
single execution step:

  Step 1 -- Discover: fetch live bounties from Algora + GitHub, score
            by BountyPolicy ROI and Ecodian alignment.
  Step 2 -- Select: pick the best candidate (highest score, clear criteria,
            accessible repo). Store selection in memory for continuity.
  Step 3 -- Generate solution: call Simula if wired, otherwise fall back
            to direct LLM reasoning. Produces real code or documentation.
  Step 4 -- Stage: store the solution as a high-confidence insight with
            tags=["bounty_solution", "pending_submission"]. Emit
            BOUNTY_SOLUTION_PENDING on the Synapse bus.

Separation from BountyHunterExecutor:
  BountyHunterExecutor (hunt_bounties) -- discovery + filtering only;
            Nova receives candidates and decides whether to pursue.
  BountyHuntExecutor  (bounty_hunt)    -- full loop; autonomously discovers
            AND generates a solution in a single Nova intent cycle.
            Use this when the goal is "earn money now" rather than
            "find options and deliberate".

Safety constraints:
  - Required autonomy: PARTNER (2) -- external API reads + LLM generation,
    no PRs submitted, no funds moved.
  - Rate limit: 4 per day -- each execution costs significant LLM tokens.
  - Reversible: False -- stored insights persist; solution generation is
    non-reversible cognitive work.
  - Max duration: 120 s -- bounded by API timeouts + LLM call.
  - BOUNTY_SOURCE_UNAVAILABLE emitted and degraded outcome returned when
    both Algora and GitHub are unreachable. Never fakes success.
"""

from __future__ import annotations

import json as _json
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.executor import Executor
from systems.axon.executors.bounty_hunter import (
    _COST_ESTIMATES_USD,
    BountyPolicy,
    _classify_difficulty,
    _fetch_live_bounties,
    _llm_score_bounties,
)
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)
from systems.synapse.types import SynapseEvent, SynapseEventType

if TYPE_CHECKING:
    from config import ExternalPlatformsConfig

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Solution generation prompts
# ---------------------------------------------------------------------------

_SOLUTION_SYSTEM_PROMPT = """\
You are EcodiaOS - a self-sustaining AI organism. You are attempting to earn
a paid bounty by solving an open GitHub issue. Produce real, production-quality
code or documentation. Do not produce placeholders or stubs.

Your solution must:
  1. Directly address the issue acceptance criteria.
  2. Follow the repository's language and style conventions (inferred from context).
  3. Include a brief explanation of the approach.
  4. Be complete enough to open a pull request.

Output format - respond with a JSON object:
{
  "summary": "<one-sentence description of the fix>",
  "approach": "<2-3 sentences on the technical approach>",
  "files": [
    {
      "path": "<relative file path in repo>",
      "action": "create|modify|delete",
      "content": "<full file content or unified diff>"
    }
  ],
  "confidence": <float 0-1>,
  "limitations": "<known gaps or risks, or empty string>"
}
"""

_SOLUTION_USER_TEMPLATE = """\
Bounty issue details:
  Title: {title}
  Repository: {repo}
  URL: {url}
  Reward: ${reward_usd:.0f} USD
  Labels: {labels}
  Difficulty estimate: {difficulty}

Issue description:
{description}

Generate a complete solution.
"""


async def _generate_solution(
    bounty: dict[str, Any],
    llm: Any,
    simula: Any,
    logger_: Any,
) -> dict[str, Any]:
    """
    Generate a real solution for the selected bounty.

    Tries Simula first (if wired); falls back to direct LLM call.
    Returns a solution dict with keys: summary, approach, files, confidence,
    limitations, generator ("simula" | "llm").

    On total failure returns a dict with confidence=0.0 and error key set.
    """
    # -- Request solution from Simula via Synapse (fire-and-forget) --------
    # Simula subscribes to BOUNTY_SOLUTION_REQUESTED and handles proposal
    # construction internally. The result arrives asynchronously via
    # BOUNTY_SOLUTION_PENDING on a subsequent cycle. This cycle falls through
    # to LLM for a synchronous inline solution.
    if simula is not None:
        try:
            from primitives.common import new_id
            synapse = getattr(simula, "_synapse", None)
            event_bus = getattr(synapse, "event_bus", None) or getattr(synapse, "_event_bus", None)
            if event_bus is not None:
                await event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.BOUNTY_SOLUTION_REQUESTED,
                    source_system="axon.bounty_hunt",
                    data={
                        "request_id": new_id(),
                        "source": "bounty",
                        "title": bounty["title"][:200],
                        "description": bounty["description"][:2000],
                        "repository_url": bounty.get("repo", ""),
                        "issue_url": bounty["source_url"],
                        "category": "code",
                        "metadata": {
                            "reward_usd": str(bounty["reward_usd"]),
                            "labels": bounty.get("labels", []),
                            "platform": bounty.get("platform", ""),
                        },
                    },
                ))
                logger_.debug("bounty_solution_requested_via_synapse", title=bounty["title"][:80])
        except Exception as exc:
            logger_.warning("bounty_solution_request_failed", error=str(exc))

    # -- Fall back to direct LLM -------------------------------------------
    if llm is None:
        return {
            "summary": "",
            "approach": "",
            "files": [],
            "confidence": 0.0,
            "limitations": "",
            "generator": "none",
            "error": "No LLM provider available for solution generation.",
        }

    user_msg = _SOLUTION_USER_TEMPLATE.format(
        title=bounty["title"],
        repo=bounty.get("repo", "unknown"),
        url=bounty["source_url"],
        reward_usd=float(bounty["reward_usd"]),
        labels=", ".join(bounty.get("labels", [])) or "none",
        difficulty=bounty.get("difficulty", "unknown"),
        description=(bounty.get("description") or "No description provided.")[:3000],
    )

    try:
        from clients.llm import Message

        response = await llm.generate(
            system_prompt=_SOLUTION_SYSTEM_PROMPT,
            messages=[Message(role="user", content=user_msg)],
            max_tokens=4096,
        )
        raw = response.text.strip()

        if not raw:
            raise ValueError(
                f"LLM returned empty response (finish_reason={response.finish_reason})"
            )

        # Strip markdown fences
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]

        solution = _json.loads(raw)
        solution.setdefault("generator", "llm")
        solution.setdefault("confidence", 0.6)
        return solution

    except Exception as exc:
        logger_.error("llm_solution_failed", error=str(exc))
        return {
            "summary": "",
            "approach": "",
            "files": [],
            "confidence": 0.0,
            "limitations": "",
            "generator": "llm",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# BountyHuntExecutor
# ---------------------------------------------------------------------------


class BountyHuntExecutor(Executor):
    """
    Full bounty-hunt loop: discover → select → solve → stage.

    Unlike BountyHunterExecutor (which only discovers), this executor
    autonomously completes the full cognitive cycle and stages a solution
    ready for submission.

    Optional params:
      target_platforms (list[str]): Platforms to scan. Default: ["github", "algora"].
      min_reward_usd (float): Minimum reward to consider. Default: 10.0.
      max_candidates (int): Candidates to evaluate before selecting. Default: 20.
    """

    action_type = "bounty_hunt"
    description = (
        "Full autonomous bounty hunt: discover → select → generate solution → "
        "stage for submission. Emits BOUNTY_SOLUTION_PENDING on success."
    )

    required_autonomy = 2       # PARTNER - external reads + LLM work, no PRs
    reversible = False
    max_duration_ms = 120_000   # 2 min: API fetch + LLM scoring + solution gen
    rate_limit = RateLimit.per_day(4)
    counts_toward_budget = True
    emits_to_atune = True

    def __init__(
        self,
        synapse: Any = None,
        github_config: ExternalPlatformsConfig | None = None,
        llm: Any = None,
        simula: Any = None,
        memory: Any = None,
    ) -> None:
        self._synapse = synapse
        self._config = github_config
        self._llm = llm
        self._simula = simula
        self._memory = memory
        self._logger = logger.bind(executor="axon.bounty_hunt")

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        platforms = params.get("target_platforms", ["github", "algora"])
        if not isinstance(platforms, list) or len(platforms) == 0:
            return ValidationResult.fail(
                "target_platforms must be a non-empty list",
                target_platforms="invalid",
            )
        min_reward = params.get("min_reward_usd", 10.0)
        try:
            float(Decimal(str(min_reward)))
        except Exception:
            return ValidationResult.fail(
                "min_reward_usd must be a valid number",
                min_reward_usd="not a number",
            )
        return ValidationResult.ok()

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        hunt_id = f"hunt-{uuid.uuid4().hex[:10]}"
        target_platforms: list[str] = [
            p.lower().strip()
            for p in params.get("target_platforms", ["github", "algora"])
        ]
        min_reward_usd = float(params.get("min_reward_usd", 10.0))
        max_candidates = int(params.get("max_candidates", 20))

        self._logger.info(
            "bounty_hunt_started",
            hunt_id=hunt_id,
            platforms=target_platforms,
            min_reward_usd=min_reward_usd,
            execution_id=context.execution_id,
        )

        # -- Resolve config lazily ----------------------------------------
        live_platforms = {"github", "algora"} & set(target_platforms)
        if live_platforms and self._config is None:
            try:
                from config import ExternalPlatformsConfig as _Cfg
                self._config = _Cfg()
            except Exception:
                pass

        # ================================================================
        # STEP 1 - Discover
        # ================================================================
        raw_bounties: list[dict[str, Any]] = []
        source_error: str = ""

        if live_platforms and self._config is not None:
            try:
                raw_bounties = await _fetch_live_bounties(
                    config=self._config,
                    target_platforms=list(live_platforms),
                    min_reward_usd=min_reward_usd,
                    max_fetch=max_candidates,
                )
                self._logger.info(
                    "bounty_hunt_discovered",
                    hunt_id=hunt_id,
                    count=len(raw_bounties),
                )
            except Exception as exc:
                source_error = str(exc)
                self._logger.error(
                    "bounty_hunt_fetch_failed",
                    hunt_id=hunt_id,
                    error=source_error,
                )
        elif not live_platforms:
            source_error = f"No supported live platforms in {target_platforms}"
        else:
            source_error = (
                "ExternalPlatformsConfig unavailable - set "
                "ECODIAOS_EXTERNAL_PLATFORMS__GITHUB_TOKEN"
            )

        if not raw_bounties:
            # Both sources unreachable or returned nothing - emit signal and degrade
            await self._emit_source_unavailable(hunt_id, source_error)
            return ExecutionResult(
                success=False,
                error=f"BOUNTY_SOURCE_UNAVAILABLE: {source_error or 'no bounties returned'}",
                side_effects=[f"Bounty hunt [{hunt_id}]: no sources available."],
                new_observations=["Bounty hunt failed: no live bounty sources reachable."],
            )

        # ================================================================
        # STEP 2 - Select (BountyPolicy filter + LLM alignment scoring)
        # ================================================================
        passing: list[dict[str, Any]] = []
        for b in raw_bounties:
            labels = list(b.get("labels", []))
            difficulty = _classify_difficulty(labels)
            cost = _COST_ESTIMATES_USD[difficulty]
            reward = float(b.get("reward_usd", 0))
            result = BountyPolicy.evaluate(reward_usd=reward, estimated_cost_usd=cost)
            if not result["passes"]:
                continue
            b["difficulty"] = difficulty.value
            b["estimated_cost_usd"] = cost
            b["roi"] = result["roi"]
            b["cost_pct"] = result["cost_pct"]
            b["ecodian_score"] = 50
            passing.append(b)

        if not passing:
            self._logger.info("bounty_hunt_none_pass_policy", hunt_id=hunt_id)
            return ExecutionResult(
                success=False,
                error="No bounties passed BountyPolicy ROI/cost threshold.",
                side_effects=[
                    f"Bounty hunt [{hunt_id}]: {len(raw_bounties)} fetched, "
                    "0 passed BountyPolicy."
                ],
                new_observations=[
                    f"Bounty scan fetched {len(raw_bounties)} issues; "
                    "none met the minimum ROI threshold."
                ],
            )

        # LLM alignment scoring on top candidates
        to_score = passing[:max(10, max_candidates)]
        llm_scores = await _llm_score_bounties(to_score, self._llm)
        for b in passing:
            if b["id"] in llm_scores:
                b["ecodian_score"] = llm_scores[b["id"]]

        # Sort by composite score (ROI × ecodian_score), pick winner
        passing.sort(
            key=lambda b: float(b.get("roi", 0)) * float(b.get("ecodian_score", 50)),
            reverse=True,
        )
        selected = passing[0]

        self._logger.info(
            "BOUNTY_SELECTED",
            hunt_id=hunt_id,
            title=selected["title"],
            url=selected["source_url"],
            reward_usd=selected["reward_usd"],
            roi=selected["roi"],
            ecodian_score=selected["ecodian_score"],
            difficulty=selected["difficulty"],
            reason=(
                f"Highest composite score (ROI={selected['roi']:.1f}x × "
                f"alignment={selected['ecodian_score']}/100) from "
                f"{len(passing)} policy-passing candidates."
            ),
        )

        # Persist selection to memory for cross-cycle continuity
        await self._persist_selection(selected, hunt_id)

        # ================================================================
        # STEP 3 - Generate solution
        # ================================================================
        self._logger.info(
            "bounty_hunt_generating_solution",
            hunt_id=hunt_id,
            bounty_url=selected["source_url"],
            generator="simula" if self._simula else "llm",
        )

        solution = await _generate_solution(
            bounty=selected,
            llm=self._llm,
            simula=self._simula,
            logger_=self._logger,
        )

        if solution.get("confidence", 0.0) == 0.0:
            err = solution.get("error", "Unknown generation failure")
            self._logger.error(
                "bounty_hunt_solution_failed",
                hunt_id=hunt_id,
                error=err,
            )
            return ExecutionResult(
                success=False,
                error=f"Solution generation failed: {err}",
                side_effects=[f"Bounty hunt [{hunt_id}]: solution generation failed."],
                new_observations=[
                    f"Failed to generate solution for bounty: {selected['source_url']}"
                ],
                data={
                    "hunt_id": hunt_id,
                    "selected_bounty": selected["source_url"],
                    "reward_usd": selected["reward_usd"],
                },
            )

        confidence = float(solution.get("confidence", 0.6))
        generator = solution.get("generator", "llm")

        self._logger.info(
            "BOUNTY_SOLUTION_READY",
            hunt_id=hunt_id,
            bounty_url=selected["source_url"],
            reward_usd=selected["reward_usd"],
            confidence=confidence,
            generator=generator,
            summary=solution.get("summary", "")[:120],
        )

        # ================================================================
        # STEP 4 - Stage solution
        # ================================================================
        insight_text = (
            f"Bounty solution (pending submission): {selected['title']} | "
            f"URL: {selected['source_url']} | "
            f"Reward: ${selected['reward_usd']:.0f} | "
            f"Confidence: {confidence:.0%} | "
            f"Approach: {solution.get('approach', '')[:300]}"
        )

        await self._store_solution_insight(
            insight=insight_text,
            bounty=selected,
            solution=solution,
            hunt_id=hunt_id,
        )

        # Emit BOUNTY_SOLUTION_PENDING on the Synapse bus
        await self._emit_solution_pending(
            bounty=selected,
            solution=solution,
            hunt_id=hunt_id,
        )

        # Structured log - organism explains its reasoning
        self._logger.info(
            "BOUNTY_SOLUTION_READY - awaiting submission capability",
            hunt_id=hunt_id,
            bounty_url=selected["source_url"],
            bounty_title=selected["title"],
            platform=selected.get("platform"),
            repo=selected.get("repo"),
            reward_usd=selected["reward_usd"],
            estimated_cost_usd=selected["estimated_cost_usd"],
            roi=selected["roi"],
            ecodian_score=selected["ecodian_score"],
            confidence=confidence,
            generator=generator,
            files_generated=len(solution.get("files", [])),
            selection_reason=(
                f"Selected from {len(passing)} policy-passing candidates. "
                f"ROI {selected['roi']:.1f}x, alignment {selected['ecodian_score']}/100."
            ),
        )

        observation = (
            f"Bounty solution generated: \"{selected['title']}\" "
            f"(${selected['reward_usd']:.0f} reward, {confidence:.0%} confidence). "
            f"URL: {selected['source_url']}. "
            f"Staged for submission. Generator: {generator}."
        )

        return ExecutionResult(
            success=True,
            data={
                "hunt_id": hunt_id,
                "bounty_url": selected["source_url"],
                "bounty_title": selected["title"],
                "platform": selected.get("platform"),
                "repo": selected.get("repo"),
                "reward_usd": selected["reward_usd"],
                "estimated_cost_usd": selected["estimated_cost_usd"],
                "roi": selected["roi"],
                "ecodian_score": selected["ecodian_score"],
                "difficulty": selected.get("difficulty"),
                "confidence": confidence,
                "generator": generator,
                "solution_summary": solution.get("summary", ""),
                "solution_approach": solution.get("approach", ""),
                "files_count": len(solution.get("files", [])),
                "limitations": solution.get("limitations", ""),
                "economic_delta_usd": selected["reward_usd"],
                "world_state_changes": [
                    f"bounty_solution_generated: {selected['source_url']}"
                ],
            },
            side_effects=[
                f"Bounty hunt [{hunt_id}]: solution staged for "
                f"\"{selected['title']}\" (${selected['reward_usd']:.0f})."
            ],
            new_observations=[observation],
        )

    # ------------------------------------------------------------------ helpers

    async def _persist_selection(
        self, bounty: dict[str, Any], hunt_id: str
    ) -> None:
        """Store selected bounty in memory for cross-cycle continuity."""
        if self._memory is None:
            return
        try:
            await self._memory.resolve_and_create_entity(
                name=f"BountyCandidate:{bounty['source_url']}",
                entity_type="concept",
                description=(
                    f"[bounty_hunt:{hunt_id}] {bounty['title']} | "
                    f"${bounty['reward_usd']:.0f} | ROI {bounty['roi']:.1f}x | "
                    f"{bounty['source_url']}"
                ),
            )
        except Exception as exc:
            self._logger.warning("bounty_selection_persist_failed", error=str(exc))

    async def _store_solution_insight(
        self,
        insight: str,
        bounty: dict[str, Any],
        solution: dict[str, Any],
        hunt_id: str,
    ) -> None:
        """Store the solution as a high-confidence insight in memory."""
        if self._memory is None:
            return
        try:
            await self._memory.resolve_and_create_entity(
                name=insight[:80],
                entity_type="concept",
                description=f"[bounty_solution] {insight}",
            )
        except Exception as exc:
            self._logger.warning("bounty_solution_store_failed", error=str(exc))

    async def _emit_solution_pending(
        self,
        bounty: dict[str, Any],
        solution: dict[str, Any],
        hunt_id: str,
    ) -> None:
        """Emit BOUNTY_SOLUTION_PENDING on the Synapse bus."""
        if self._synapse is None:
            return
        try:
            event_bus = getattr(self._synapse, "event_bus", None)
            if event_bus is None:
                return
            await event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.BOUNTY_SOLUTION_PENDING,
                source_system="axon.bounty_hunt",
                data={
                    "hunt_id": hunt_id,
                    "bounty_url": bounty["source_url"],
                    "bounty_title": bounty["title"],
                    "estimated_reward_usd": bounty["reward_usd"],
                    "solution_confidence": solution.get("confidence", 0.0),
                    "solution_code": solution.get("solution_code", ""),
                    "generator": solution.get("generator", "unknown"),
                    "platform": bounty.get("platform"),
                    "repo": bounty.get("repo"),
                },
            ))
        except Exception as exc:
            self._logger.warning("bounty_solution_pending_emit_failed", error=str(exc))

    async def _emit_source_unavailable(self, hunt_id: str, reason: str) -> None:
        """Emit BOUNTY_SOURCE_UNAVAILABLE when all bounty sources are unreachable."""
        if self._synapse is None:
            return
        try:
            event_bus = getattr(self._synapse, "event_bus", None)
            if event_bus is None:
                return
            await event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.BOUNTY_SOURCE_UNAVAILABLE,
                source_system="axon.bounty_hunt",
                data={
                    "hunt_id": hunt_id,
                    "reason": reason,
                    "timestamp": datetime.now(tz=UTC).isoformat(),
                },
            ))
        except Exception as exc:
            self._logger.warning("bounty_source_unavailable_emit_failed", error=str(exc))
