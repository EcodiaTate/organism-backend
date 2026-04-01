"""
EcodiaOS -- Axon Bounty Hunter Executor (Phase 16b -- Freelancer/Foraging)

Purpose-built executor that scans for paid bounties across configured
platforms (GitHub Issues w/ bounty labels, Algora, Replit Bounties, etc.)
and evaluates each against a strict economic policy before surfacing it
to Nova for acceptance.

The organism hunts to fund its basal metabolic rate.  This executor is
the "foraging" half of that loop -- it finds candidate work, estimates
the cost to complete it, and only returns bounties that pass the
BountyPolicy ROI threshold.

BountyPolicy (non-negotiable):
  MIN_ROI_THRESHOLD   = 2.0   -- reward must be >= 2x estimated API token cost
  MAX_ESTIMATED_COST_PCT = 0.40 -- estimated cost must be <= 40% of reward

Safety constraints:
  - Required autonomy: PARTNER (2) -- scans external platforms, no funds moved
  - Rate limit: 6 scans per hour -- prevents hammering external APIs
  - No state mutation -- this executor only reads and filters
  - Returns structured bounty candidates; Nova decides whether to accept
  - All bounty evaluations logged for audit trail
"""

from __future__ import annotations

import asyncio
import json as _json
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import structlog

from clients.algora import AlgoraClient
from clients.github import GitHubClient
from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)

if TYPE_CHECKING:
    from config import ExternalPlatformsConfig

logger = structlog.get_logger()

# -- BountyPolicy -- non-negotiable economic constraints -----------------


class BountyPolicy:
    """
    The organism's foraging economics.

    A bounty is only worth pursuing if the expected return exceeds
    the cost of the cognitive resources (API tokens, compute) needed
    to complete it.
    """

    # Reward must be at least 2x the estimated API token cost
    MIN_ROI_THRESHOLD: float = 2.0

    # Estimated cost must not exceed 40% of the reward
    MAX_ESTIMATED_COST_PCT: float = 0.40

    @classmethod
    def evaluate(
        cls,
        reward_usd: float,
        estimated_cost_usd: float,
    ) -> dict[str, Any]:
        """
        Evaluate a bounty against the policy.

        Returns a dict with:
          passes: bool -- whether the bounty passes the policy
          roi: float -- reward / cost ratio
          cost_pct: float -- cost as a fraction of reward
          rejection_reasons: list[str] -- why it failed (empty if passes)
        """
        rejection_reasons: list[str] = []

        # Guard: zero/negative reward is auto-reject
        if reward_usd <= 0:
            return {
                "passes": False,
                "roi": 0.0,
                "cost_pct": 1.0,
                "rejection_reasons": ["reward_usd must be > 0"],
            }

        # Guard: zero cost means we can't compute ROI meaningfully;
        # treat as infinite ROI -- passes trivially
        if estimated_cost_usd <= 0:
            return {
                "passes": True,
                "roi": float("inf"),
                "cost_pct": 0.0,
                "rejection_reasons": [],
            }

        roi = reward_usd / estimated_cost_usd
        cost_pct = estimated_cost_usd / reward_usd

        if roi < cls.MIN_ROI_THRESHOLD:
            rejection_reasons.append(
                f"ROI {roi:.2f}x < minimum {cls.MIN_ROI_THRESHOLD}x"
            )

        if cost_pct > cls.MAX_ESTIMATED_COST_PCT:
            rejection_reasons.append(
                f"cost_pct {cost_pct:.2%} > maximum {cls.MAX_ESTIMATED_COST_PCT:.0%}"
            )

        return {
            "passes": len(rejection_reasons) == 0,
            "roi": round(roi, 4),
            "cost_pct": round(cost_pct, 4),
            "rejection_reasons": rejection_reasons,
        }


# -- Bounty difficulty classification ------------------------------------


class BountyDifficulty(StrEnum):
    TRIVIAL = "trivial"      # < 30 min estimated, docs/typos/config
    EASY = "easy"            # 30 min - 2 hr, small feature/bugfix
    MEDIUM = "medium"        # 2 - 8 hr, moderate feature/refactor
    HARD = "hard"            # 8+ hr, architectural change
    UNKNOWN = "unknown"


# Token cost estimates per difficulty tier (USD).
# Based on Claude Sonnet-class model at ~$3/M input + $15/M output tokens,
# assuming average task token budgets.
_COST_ESTIMATES_USD: dict[BountyDifficulty, float] = {
    BountyDifficulty.TRIVIAL: 0.05,
    BountyDifficulty.EASY: 0.25,
    BountyDifficulty.MEDIUM: 1.50,
    BountyDifficulty.HARD: 5.00,
    BountyDifficulty.UNKNOWN: 2.00,  # Conservative default
}


# -- Supported platform identifiers --------------------------------------

_SUPPORTED_PLATFORMS = frozenset({
    "github",
    "algora",
    "replit",
    "gitcoin",
})


# -- Label heuristics for difficulty classification -----------------------

_TRIVIAL_LABELS = frozenset({"good first issue", "documentation", "typo", "chore"})
_EASY_LABELS = frozenset({"bug", "minor", "enhancement", "small"})
_MEDIUM_LABELS = frozenset({"feature", "medium", "moderate", "refactor"})
_HARD_LABELS = frozenset({"major", "hard", "complex", "architecture", "breaking"})


def _classify_difficulty(labels: list[str]) -> BountyDifficulty:
    """Heuristic difficulty classification from issue labels."""
    lower_labels = {label.lower().strip() for label in labels}
    if lower_labels & _HARD_LABELS:
        return BountyDifficulty.HARD
    if lower_labels & _MEDIUM_LABELS:
        return BountyDifficulty.MEDIUM
    if lower_labels & _EASY_LABELS:
        return BountyDifficulty.EASY
    if lower_labels & _TRIVIAL_LABELS:
        return BountyDifficulty.TRIVIAL
    return BountyDifficulty.UNKNOWN


# -- Live fetch (GitHub + Algora concurrently) ----------------------------


async def _fetch_live_bounties(
    config: ExternalPlatformsConfig,
    target_platforms: list[str],
    min_reward_usd: float,
    max_fetch: int,
) -> list[dict[str, Any]]:
    """
    Fetch live bounty issues from GitHub and/or Algora concurrently.

    Both platforms are gathered in a single ``asyncio.gather`` call so the
    combined wall-clock time is bounded by the slower of the two APIs, not
    their sum.

    Returns a list of raw bounty dicts shaped identically to the fields
    the BountyHunterExecutor expects (id, platform, source_url, title,
    description, reward_usd, labels, repo, posted_at, expires_at).

    Raises on unrecoverable network errors so the caller can surface them.
    """
    coroutines: list[Any] = []
    platform_order: list[str] = []

    if "github" in target_platforms:
        async def _gh() -> list[dict[str, Any]]:
            async with GitHubClient(config) as gh:
                result = await gh.search_open_bounties(
                    min_reward=0,         # pre-filter disabled; BountyPolicy filters later
                    max_pages=max(1, max_fetch // 100),
                )
            items: list[dict[str, Any]] = []
            for issue in result.items[:max_fetch]:
                items.append({
                    "id": f"github-{issue.repo.replace('/', '-')}-{issue.id}",
                    "platform": "github",
                    "source_url": issue.url,
                    "title": issue.title,
                    "description": (issue.body or "")[:1000].strip(),
                    "reward_usd": issue.reward_usd if issue.reward_usd is not None else 0.0,
                    "labels": issue.labels,
                    "language": "unknown",
                    "repo": issue.repo,
                    "posted_at": "",
                    "expires_at": None,
                })
            return items

        coroutines.append(_gh())
        platform_order.append("github")

    if "algora" in target_platforms:
        async def _al() -> list[dict[str, Any]]:
            async with AlgoraClient(config) as al:
                result = await al.fetch_active_bounties(
                    min_reward=0,         # pre-filter disabled; BountyPolicy filters later
                    max_pages=max(1, max_fetch // 100),
                )
            items: list[dict[str, Any]] = []
            for issue in result.items[:max_fetch]:
                items.append({
                    "id": f"algora-{issue.repo.replace('/', '-')}-{issue.id}",
                    "platform": "algora",
                    "source_url": issue.url,
                    "title": issue.title,
                    "description": (issue.body or "")[:1000].strip(),
                    "reward_usd": issue.reward_usd if issue.reward_usd is not None else 0.0,
                    "labels": issue.labels,
                    "language": "unknown",
                    "repo": issue.repo,
                    "posted_at": "",
                    "expires_at": None,
                })
            return items

        coroutines.append(_al())
        platform_order.append("algora")

    if not coroutines:
        return []

    results: list[list[dict[str, Any]]] = await asyncio.gather(*coroutines)

    # Concatenate all results then slice to max_fetch so the LLM scorer
    # sees a manageable candidate set regardless of how many platforms fired.
    combined: list[dict[str, Any]] = []
    for platform_items in results:
        combined.extend(platform_items)

    return combined[:max_fetch]


# -- LLM scoring -------------------------------------------------------------


_SCORE_PROMPT_TEMPLATE = """\
You are the foraging intelligence for an AI organism called EcodiaOS.
Your job is to score GitHub bounty issues by their strategic value to the organism.

The organism's capabilities: Python, TypeScript, async systems, LLMs, APIs, data pipelines, \
web tooling, CLI tools, DevOps. It is skilled at backend, tooling, and integrations.
It cannot do hardware, mobile-native (iOS/Android), or domain-specific work \
(legal, medical, finance compliance).

Ecodian alignment criteria (score higher if):
  - Uses Python or TypeScript
  - Is a backend / API / tooling / data task
  - Has clear acceptance criteria
  - Is self-contained (no deep domain knowledge required)
  - Has a reasonable reward relative to complexity

Score each issue 0–100. Return ONLY valid JSON - an array of objects with:
  "id": (the issue id string),
  "score": (integer 0–100),
  "reasoning": (one sentence max)

Issues to score:
{issues_json}
"""


async def _llm_score_bounties(
    bounties: list[dict[str, Any]],
    llm: Any,
) -> dict[str, int]:
    """
    Ask the LLM to score each bounty on Ecodian alignment (0–100).

    Returns a mapping of bounty id → score.  On any LLM failure, returns
    a default score of 50 for all bounties so the hunt can still complete.
    """
    if not bounties or llm is None:
        return {b["id"]: 50 for b in bounties}

    slim = [
        {
            "id": b["id"],
            "title": b["title"],
            "description": b["description"][:400],
            "labels": b["labels"],
            "reward_usd": b["reward_usd"],
            "repo": b["repo"],
            "language": b["language"],
        }
        for b in bounties
    ]

    prompt = _SCORE_PROMPT_TEMPLATE.format(issues_json=_json.dumps(slim, indent=2))

    try:
        from clients.llm import Message

        response = await llm.generate(
            system_prompt="",
            messages=[Message(role="user", content=prompt)],
            max_tokens=1024,
        )
        raw_text = response.text.strip()

        if not raw_text:
            logger.warning(
                "bounty_llm_score_empty",
                finish_reason=response.finish_reason,
            )
            return {b["id"]: 50 for b in bounties}

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        scored = _json.loads(raw_text)
        if not isinstance(scored, list):
            raise ValueError("LLM returned non-list JSON")

        return {
            item["id"]: int(item.get("score", 50))
            for item in scored
            if isinstance(item, dict) and "id" in item
        }

    except Exception as exc:
        logger.warning("bounty_llm_score_failed", error=str(exc))
        return {b["id"]: 50 for b in bounties}


# -- BountyHunterExecutor ------------------------------------------------


class BountyHunterExecutor(Executor):
    """
    Scan configured platforms for paid bounties and evaluate each against
    the organism's BountyPolicy before returning viable candidates to Nova.

    This is a read-only, foraging executor -- it finds and filters work
    opportunities but does not accept or commit to any of them.  Nova
    receives the structured output and decides whether to pursue.

    Required params:
      target_platforms (list[str]): Platforms to scan.
                                    Supported: "github", "algora", "replit", "gitcoin".

    Optional params:
      min_reward_usd (float | str): Minimum bounty reward to consider. Default 5.0.
      max_results (int): Maximum number of passing bounties to return. Default 10.
      include_rejected (bool): If True, include rejected bounties in output
                               (marked with rejection_reasons). Default False.

    Returns ExecutionResult with:
      data:
        bounties          -- list of evaluated bounty dicts (see below)
        total_scanned     -- number of raw bounties fetched
        total_passed      -- number that passed BountyPolicy
        total_rejected    -- number that failed BountyPolicy
        policy            -- the policy parameters used
        scan_id           -- unique scan identifier
        top_bounty_url    -- source_url of the highest-scoring passing bounty
      side_effects:
        -- Human-readable summary of the scan
      new_observations:
        -- Feed top candidates back to Atune as a Percept

    Each bounty dict in data.bounties:
      id, platform, source_url, title, description,
      reward_usd, language, repo, labels,
      posted_at, expires_at,
      difficulty          -- classified difficulty tier
      estimated_cost_usd  -- API token cost estimate
      roi                 -- reward / cost ratio
      cost_pct            -- cost as fraction of reward
      policy_passes       -- bool
      rejection_reasons   -- list[str] (empty if passes)
      ecodian_score       -- LLM-assigned alignment score (0-100)
    """

    action_type = "hunt_bounties"
    description = (
        "Scan GitHub and Algora for live paid bounties, score against BountyPolicy + Ecodian "
        "alignment, and return the highest-scoring real bounty URL for Nova (Phase 16b)"
    )

    required_autonomy = 2       # PARTNER -- reads external platforms, no funds moved
    reversible = False          # Read-only scan, nothing to reverse
    max_duration_ms = 60_000    # Live API calls + LLM scoring can be slow
    rate_limit = RateLimit.per_hour(6)  # Don't hammer external APIs

    def __init__(
        self,
        synapse: Any = None,
        github_config: ExternalPlatformsConfig | None = None,
        llm: Any = None,
    ) -> None:
        self._synapse = synapse
        self._config = github_config   # ExternalPlatformsConfig holds all platform keys
        self._llm = llm
        self._logger = logger.bind(executor="axon.bounty_hunter")

    # -- Validation -------------------------------------------------------

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        """Validate scan parameters -- no I/O."""
        # target_platforms: required, non-empty list of supported platforms.
        platforms_raw = params.get("target_platforms")
        if not platforms_raw:
            return ValidationResult.fail(
                "target_platforms is required (list of platform names)",
                target_platforms="missing",
            )
        if not isinstance(platforms_raw, list):
            return ValidationResult.fail(
                "target_platforms must be a list",
                target_platforms="not a list",
            )
        if len(platforms_raw) == 0:
            return ValidationResult.fail(
                "target_platforms must contain at least one platform",
                target_platforms="empty list",
            )
        unsupported = {
            p.lower().strip() for p in platforms_raw
        } - _SUPPORTED_PLATFORMS
        if unsupported:
            return ValidationResult.fail(
                f"Unsupported platforms: {sorted(unsupported)}. "
                f"Supported: {sorted(_SUPPORTED_PLATFORMS)}",
                target_platforms="unsupported platform(s)",
            )

        # min_reward_usd: optional, must be >= 0 if provided
        min_reward_raw = params.get("min_reward_usd", "5.0")
        try:
            min_reward = float(Decimal(str(min_reward_raw)))
        except Exception:
            return ValidationResult.fail(
                "min_reward_usd must be a valid number",
                min_reward_usd="not a number",
            )
        if min_reward < 0:
            return ValidationResult.fail(
                "min_reward_usd must be >= 0",
                min_reward_usd="negative value",
            )

        # max_results: optional, must be positive int if provided
        max_results = params.get("max_results", 10)
        if not isinstance(max_results, int) or max_results < 1:
            return ValidationResult.fail(
                "max_results must be a positive integer",
                max_results="invalid value",
            )

        return ValidationResult.ok()

    # -- Execution --------------------------------------------------------

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Fetch live bounty issues from GitHub and Algora concurrently, evaluate
        via BountyPolicy + LLM scorer, and return the highest-scoring real
        bounty URL.  Never raises.
        """
        target_platforms: list[str] = [
            p.lower().strip() for p in params["target_platforms"]
        ]
        min_reward_usd = float(params.get("min_reward_usd", 5.0))
        max_results = int(params.get("max_results", 10))
        include_rejected = bool(params.get("include_rejected", False))

        scan_id = f"scan-{uuid.uuid4().hex[:12]}"

        self._logger.info(
            "bounty_hunt_started",
            scan_id=scan_id,
            target_platforms=target_platforms,
            min_reward_usd=min_reward_usd,
            max_results=max_results,
            execution_id=context.execution_id,
        )

        # -- Resolve config (lazy fallback to env vars) ------------------
        live_platforms = {"github", "algora"} & set(target_platforms)
        if live_platforms:
            if self._config is None:
                try:
                    from config import ExternalPlatformsConfig as _Cfg
                    self._config = _Cfg()
                except Exception:
                    pass

            if self._config is None:
                self._logger.warning(
                    "bounty_no_config",
                    scan_id=scan_id,
                    note="config not provided; set ECODIAOS_EXTERNAL_PLATFORMS__GITHUB_TOKEN",
                )
                return ExecutionResult(
                    success=False,
                    error=(
                        "BountyHunterExecutor requires ExternalPlatformsConfig. "
                        "Inject github_config into BountyHunterExecutor or set "
                        "ECODIAOS_EXTERNAL_PLATFORMS__GITHUB_TOKEN."
                    ),
                )

        # -- Fetch bounties from all live platforms concurrently ----------
        raw_bounties: list[dict[str, Any]] = []

        if live_platforms and self._config is not None:
            try:
                raw_bounties = await _fetch_live_bounties(
                    config=self._config,
                    target_platforms=target_platforms,
                    min_reward_usd=min_reward_usd,
                    # Fetch 2–3x more than max_results so the scorer has
                    # enough candidates to choose the best 5-10 from.
                    max_fetch=min(max_results * 3, 50),
                )
                self._logger.info(
                    "bounty_fetch_complete",
                    scan_id=scan_id,
                    total=len(raw_bounties),
                    platforms=sorted(live_platforms),
                )
            except Exception as exc:
                self._logger.error(
                    "bounty_fetch_failed",
                    scan_id=scan_id,
                    platforms=sorted(live_platforms),
                    error=str(exc),
                )
                return ExecutionResult(
                    success=False,
                    error=f"Bounty fetch failed: {exc}",
                )

        # Other platforms (replit, gitcoin) are accepted for forward-compatibility
        # but silently skipped until wired up.

        platforms_label = " + ".join(sorted(live_platforms)) if live_platforms else ", ".join(target_platforms)

        if not raw_bounties:
            return ExecutionResult(
                success=True,
                data={
                    "bounties": [],
                    "total_scanned": 0,
                    "total_passed": 0,
                    "total_rejected": 0,
                    "policy": {
                        "min_roi_threshold": BountyPolicy.MIN_ROI_THRESHOLD,
                        "max_estimated_cost_pct": BountyPolicy.MAX_ESTIMATED_COST_PCT,
                    },
                    "scan_id": scan_id,
                    "scanned_at": datetime.now(tz=UTC).isoformat(),
                    "top_bounty_url": None,
                },
                side_effects=[f"Bounty scan [{scan_id}]: no issues returned from {platforms_label}."],
                new_observations=[f"Bounty scan returned 0 live issues from {platforms_label}."],
            )

        # -- Evaluate each bounty against BountyPolicy -------------------
        evaluated: list[dict[str, Any]] = []
        passed_bounties: list[dict[str, Any]] = []
        passed_count = 0
        rejected_count = 0

        for bounty in raw_bounties:
            labels = list(bounty.get("labels", []))
            difficulty = _classify_difficulty(labels)
            estimated_cost = _COST_ESTIMATES_USD[difficulty]
            reward = float(bounty["reward_usd"])

            policy_result = BountyPolicy.evaluate(
                reward_usd=reward,
                estimated_cost_usd=estimated_cost,
            )

            enriched: dict[str, Any] = {
                "id": bounty["id"],
                "platform": bounty["platform"],
                "source_url": bounty["source_url"],
                "title": bounty["title"],
                "description": bounty["description"],
                "reward_usd": reward,
                "language": bounty.get("language", "unknown"),
                "repo": bounty.get("repo"),
                "labels": labels,
                "posted_at": bounty.get("posted_at"),
                "expires_at": bounty.get("expires_at"),
                "difficulty": difficulty.value,
                "estimated_cost_usd": estimated_cost,
                "roi": policy_result["roi"],
                "cost_pct": policy_result["cost_pct"],
                "policy_passes": policy_result["passes"],
                "rejection_reasons": policy_result["rejection_reasons"],
                "ecodian_score": 50,   # placeholder; filled in by LLM below
            }

            if policy_result["passes"]:
                passed_count += 1
                evaluated.append(enriched)
                passed_bounties.append(enriched)

                self._logger.info(
                    "bounty_passed_policy",
                    scan_id=scan_id,
                    bounty_id=bounty["id"],
                    title=bounty["title"],
                    reward_usd=reward,
                    roi=policy_result["roi"],
                    difficulty=difficulty.value,
                )
            else:
                rejected_count += 1
                if include_rejected:
                    evaluated.append(enriched)

                self._logger.debug(
                    "bounty_rejected_by_policy",
                    scan_id=scan_id,
                    bounty_id=bounty["id"],
                    reward_usd=reward,
                    rejection_reasons=policy_result["rejection_reasons"],
                )

        # -- LLM alignment scoring on the top 5-10 passing bounties -----
        # Only score the slice Nova will see to keep token usage bounded.
        candidates_for_scoring = passed_bounties[:max(max_results, 10)]
        if candidates_for_scoring:
            llm_scores = await _llm_score_bounties(candidates_for_scoring, self._llm)
            for b in evaluated:
                if b["id"] in llm_scores:
                    b["ecodian_score"] = llm_scores[b["id"]]
            self._logger.info(
                "bounty_llm_scoring_done",
                scan_id=scan_id,
                scored=len(llm_scores),
            )

        # -- Sort: passed first, then by composite (roi × ecodian_score) -
        def _sort_key(b: dict[str, Any]) -> tuple[bool, float]:
            roi = float(b.get("roi", 0) or 0)
            score = float(b.get("ecodian_score", 50))
            return (b["policy_passes"], roi * score)

        evaluated.sort(key=_sort_key, reverse=True)
        capped = evaluated[:max_results]

        # -- Identify the single highest-scoring passing bounty ----------
        top_passing = [b for b in capped if b["policy_passes"]]
        top_bounty_url: str | None = top_passing[0]["source_url"] if top_passing else None

        # -- Build observation for Atune ---------------------------------
        top_candidates = top_passing[:3]
        if top_candidates:
            observation_lines = [
                f"Live bounty scan complete: {passed_count} viable / "
                f"{len(raw_bounties)} fetched from {platforms_label}. Top candidates:"
            ]
            for b in top_candidates:
                observation_lines.append(
                    f"  - [{b['platform']}] \"{b['title']}\" "
                    f"${b['reward_usd']:.0f} reward, "
                    f"{b['roi']:.1f}x ROI, "
                    f"score {b['ecodian_score']}/100, "
                    f"{b['difficulty']} difficulty"
                )
            if top_bounty_url:
                observation_lines.append(f"  Top pick: {top_bounty_url}")
            observation = "\n".join(observation_lines)
        else:
            observation = (
                f"Live bounty scan complete: 0 viable bounties found across "
                f"{platforms_label} "
                f"(fetched {len(raw_bounties)}, all rejected by BountyPolicy)."
            )

        # -- Side effect summary -----------------------------------------
        side_effect = (
            f"Live bounty scan [{scan_id}]: fetched {len(raw_bounties)} issues "
            f"from {platforms_label}. "
            f"{passed_count} passed BountyPolicy "
            f"(min ROI {BountyPolicy.MIN_ROI_THRESHOLD}x, "
            f"max cost {BountyPolicy.MAX_ESTIMATED_COST_PCT:.0%}), "
            f"{rejected_count} rejected. "
            f"Top pick: {top_bounty_url or 'none'}."
        )

        self._logger.info(
            "bounty_hunt_complete",
            scan_id=scan_id,
            total_scanned=len(raw_bounties),
            total_passed=passed_count,
            total_rejected=rejected_count,
            top_bounty_url=top_bounty_url,
            execution_id=context.execution_id,
        )

        return ExecutionResult(
            success=True,
            data={
                "bounties": capped,
                "total_scanned": len(raw_bounties),
                "total_passed": passed_count,
                "total_rejected": rejected_count,
                "policy": {
                    "min_roi_threshold": BountyPolicy.MIN_ROI_THRESHOLD,
                    "max_estimated_cost_pct": BountyPolicy.MAX_ESTIMATED_COST_PCT,
                },
                "scan_id": scan_id,
                "scanned_at": datetime.now(tz=UTC).isoformat(),
                "top_bounty_url": top_bounty_url,
            },
            side_effects=[side_effect],
            new_observations=[observation],
        )
