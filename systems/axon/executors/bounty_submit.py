"""
EcodiaOS — BountySubmitExecutor (Phase 16r: Bounty Submission Layer)

Submits a solved bounty to GitHub by forking the target repo, creating a
branch, committing the solution, and opening a pull request that clearly
identifies EcodiaOS as the author (honesty invariant).

Flow:
  1. Acquire GitHub token via GitHubConnector
  2. Parse owner/repo from repository_url
  3. Fork the target repository into EOS's account
  4. Create branch: eos/bounty-{bounty_id}-{timestamp}
  5. Commit the solution file(s) to that branch
  6. Open PR with a body that discloses EOS authorship + solution explanation
  7. Emit BOUNTY_PR_SUBMITTED on Synapse bus
  8. Store PR URL in Redis for MonitorPRsExecutor tracking

Safety constraints:
  - Required autonomy: 3 (TRUSTED) — creates external PRs
  - Rate limit: 5 PRs per hour (GitHub REST API conservative cap)
  - Reversible: False — submitted PRs cannot be atomically recalled
  - Circuit breaker: 3 consecutive failures → OPEN, 5 min cooldown
    (enforced by AxonService pipeline via ExecutorRegistry)
  - Max duration: 120 s — network round-trips to GitHub API only

PR body invariant:
  Every PR body unconditionally contains the EOS author block below.
  This cannot be suppressed via params — it is assembled after all
  caller-supplied content is concatenated.

Parameters (all required unless noted):
  bounty_id (str):              Unique identifier from BountyHuntExecutor.
  solution_code (str):          The solution file content to commit.
  solution_explanation (str):   Human-readable explanation of the fix.
  repository_url (str):         Target repo ("owner/repo" or HTTPS URL).
  target_branch (str, optional): Base branch to target. Default "main".
  solution_filename (str, opt): Filename for the committed code.
                                Default "solution.py".
  bounty_url (str, opt):        URL of the original bounty issue (for PR body).

Returns ExecutionResult with:
  data:
    pr_url          — HTTPS URL of the opened pull request
    pr_number       — PR number (int)
    branch          — branch name created
    fork_full_name  — "fork_owner/repo" of the fork
    bounty_id       — the bounty that was submitted
  side_effects:
    — Human-readable submission summary
  new_observations:
    — Fed back to Nova for foraging success tracking
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from systems.identity.connectors.github import GitHubConnector
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("axon.executor.bounty_submit")

# ── Constants ─────────────────────────────────────────────────────────────────

# Appended unconditionally to every PR body — honesty invariant.
_EOS_AUTHOR_BLOCK = """

---

> **Submitted by EcodiaOS** — an autonomous digital organism.
>
> This pull request was generated autonomously by
> [EcodiaOS](https://github.com/ecodia/ecodiaos), an AI system that discovers
> open-source bounties and attempts to resolve them.  The solution above was
> produced entirely by AI and has not been reviewed by a human operator before
> submission.  Please review the changes carefully before merging.
>
> If you have questions or concerns about this PR, feel free to leave a comment
> or close the PR — the organism monitors outcomes and learns from feedback.
"""

# Redis key prefix for PR URL storage (used by MonitorPRsExecutor)
_PR_REDIS_KEY_PREFIX = "axon:bounty_submit:pr:"

# Maximum characters of solution_explanation in the PR body before truncation
_MAX_EXPLANATION_LEN = 4000


# ── Executor ──────────────────────────────────────────────────────────────────


class BountySubmitExecutor(Executor):
    """
    Submit a solved bounty to GitHub as a pull request.

    See module docstring for full parameter documentation.
    """

    action_type = "submit_bounty_solution"
    description = (
        "Fork a bounty target repo, commit the solution, and open a GitHub PR "
        "with mandatory EOS authorship disclosure (Phase 16r)"
    )

    required_autonomy = 3          # TRUSTED — creates external PRs
    reversible = False             # Submitted PRs cannot be atomically reversed
    max_duration_ms = 120_000      # 2 minutes — GitHub API round-trips only
    rate_limit = RateLimit.per_hour(5)   # Conservative GitHub API cap

    def __init__(
        self,
        github_connector: GitHubConnector | None = None,
        redis: RedisClient | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self._github = github_connector
        self._redis = redis
        self._event_bus = event_bus
        self._logger = logger.bind(executor="axon.bounty_submit")

    # ── Validation ────────────────────────────────────────────────────────

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        bounty_id = str(params.get("bounty_id", "")).strip()
        if not bounty_id:
            return ValidationResult.fail("bounty_id is required", bounty_id="missing")

        solution_code = str(params.get("solution_code", "")).strip()
        if not solution_code:
            return ValidationResult.fail(
                "solution_code is required", solution_code="missing"
            )

        explanation = str(params.get("solution_explanation", "")).strip()
        if not explanation:
            return ValidationResult.fail(
                "solution_explanation is required", solution_explanation="missing"
            )

        repo_url = str(params.get("repository_url", "")).strip()
        if not repo_url:
            return ValidationResult.fail(
                "repository_url is required", repository_url="missing"
            )
        try:
            _parse_owner_repo(repo_url)
        except ValueError as exc:
            return ValidationResult.fail(str(exc), repository_url="invalid")

        return ValidationResult.ok()

    # ── Execution ─────────────────────────────────────────────────────────

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Fork → branch → commit → PR → emit event → cache PR URL.
        Never raises — failures returned in ExecutionResult.
        """
        bounty_id = str(params["bounty_id"]).strip()
        solution_code = str(params["solution_code"]).strip()
        solution_explanation = str(params["solution_explanation"]).strip()
        repository_url = str(params["repository_url"]).strip()
        target_branch = str(params.get("target_branch", "main")).strip() or "main"
        _raw_filename = str(params.get("solution_filename", "solution.py")).strip()
        solution_filename = _raw_filename or "solution.py"
        bounty_url = str(params.get("bounty_url", "")).strip()

        self._logger.info(
            "bounty_submit_started",
            bounty_id=bounty_id,
            repository_url=repository_url,
            execution_id=context.execution_id,
        )

        # ── Guard: GitHubConnector must be wired ─────────────────────────
        if self._github is None:
            return ExecutionResult(
                success=False,
                error=(
                    "BountySubmitExecutor requires a GitHubConnector. "
                    "Wire it via AxonService at startup."
                ),
                data={"bounty_id": bounty_id, "stage": "connector_missing"},
            )

        # ── Step 1: Acquire GitHub token ──────────────────────────────────
        token = await self._github.get_access_token()
        if not token:
            self._logger.warning(
                "bounty_submit_no_github_token",
                bounty_id=bounty_id,
            )
            await self._emit_credentials_missing(bounty_id, bounty_url)
            return ExecutionResult(
                success=False,
                error=(
                    "No GitHub credentials available. "
                    "Set GITHUB_TOKEN env var or configure the GitHub App connector "
                    "(GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY + GITHUB_INSTALLATION_ID)."
                ),
                data={"bounty_id": bounty_id, "stage": "credentials"},
                new_observations=[
                    f"Bounty submission BLOCKED for {bounty_id}: no GitHub credentials. "
                    "Operator must provision GITHUB_TOKEN or GitHub App credentials."
                ],
            )

        # ── Step 2: Parse owner/repo ──────────────────────────────────────
        try:
            owner, repo = _parse_owner_repo(repository_url)
        except ValueError as exc:
            return ExecutionResult(
                success=False,
                error=str(exc),
                data={"bounty_id": bounty_id, "stage": "url_parse"},
            )

        # ── Step 3: Fork the repository ───────────────────────────────────
        try:
            fork_data = await self._github.fork_repository(owner, repo)
        except Exception as exc:
            self._logger.error(
                "fork_failed",
                bounty_id=bounty_id,
                repo=f"{owner}/{repo}",
                error=str(exc),
            )
            return ExecutionResult(
                success=False,
                error=f"Failed to fork repository {owner}/{repo}: {exc}",
                data={"bounty_id": bounty_id, "stage": "fork"},
                new_observations=[
                    f"Bounty submit FAILED for {bounty_id}: "
                    f"could not fork {owner}/{repo} — {str(exc)[:200]}"
                ],
            )

        fork_full_name: str = fork_data.get("full_name", "")
        if "/" in fork_full_name:
            fork_owner, fork_repo = fork_full_name.split("/", 1)
        else:
            fork_owner, fork_repo = owner, repo

        # ── Step 4: Create branch ─────────────────────────────────────────
        timestamp = int(time.time())
        branch_name = f"eos/bounty-{_slugify(bounty_id)}-{timestamp}"

        try:
            await self._github.create_branch(
                owner=fork_owner,
                repo=fork_repo,
                branch=branch_name,
                base_branch=target_branch,
            )
        except Exception as exc:
            self._logger.error(
                "branch_creation_failed",
                bounty_id=bounty_id,
                branch=branch_name,
                error=str(exc),
            )
            return ExecutionResult(
                success=False,
                error=f"Failed to create branch '{branch_name}': {exc}",
                data={"bounty_id": bounty_id, "stage": "branch", "fork": fork_full_name},
                new_observations=[
                    f"Bounty submit FAILED for {bounty_id}: "
                    f"branch creation error — {str(exc)[:200]}"
                ],
            )

        # ── Step 5: Commit the solution ───────────────────────────────────
        commit_message = (
            f"fix: resolve bounty {bounty_id}\n\n"
            f"Automated solution by EcodiaOS.\n"
            + (f"Bounty: {bounty_url}" if bounty_url else "")
        ).strip()

        try:
            await self._github.commit_files(
                owner=fork_owner,
                repo=fork_repo,
                branch=branch_name,
                files={solution_filename: solution_code},
                commit_message=commit_message,
            )
        except Exception as exc:
            self._logger.error(
                "commit_failed",
                bounty_id=bounty_id,
                branch=branch_name,
                error=str(exc),
            )
            return ExecutionResult(
                success=False,
                error=f"Failed to commit solution to '{branch_name}': {exc}",
                data={"bounty_id": bounty_id, "stage": "commit", "branch": branch_name},
                new_observations=[
                    f"Bounty submit FAILED for {bounty_id}: commit error — {str(exc)[:200]}"
                ],
            )

        # ── Step 6: Open pull request ─────────────────────────────────────
        pr_title = f"fix: resolve bounty {bounty_id}"
        pr_body = _build_pr_body(
            bounty_id=bounty_id,
            bounty_url=bounty_url,
            explanation=solution_explanation,
            solution_filename=solution_filename,
        )

        # head is "fork_owner:branch" for cross-repo PRs
        pr_head = f"{fork_owner}:{branch_name}"

        try:
            pr_data = await self._github.open_pull_request(
                owner=owner,
                repo=repo,
                title=pr_title,
                body=pr_body,
                head=pr_head,
                base=target_branch,
                labels=["automated", "bounty"],
            )
        except Exception as exc:
            self._logger.error(
                "pr_open_failed",
                bounty_id=bounty_id,
                head=pr_head,
                error=str(exc),
            )
            return ExecutionResult(
                success=False,
                error=f"Failed to open pull request: {exc}",
                data={
                    "bounty_id": bounty_id,
                    "stage": "pr_open",
                    "branch": branch_name,
                    "fork": fork_full_name,
                },
                new_observations=[
                    f"Bounty submit FAILED for {bounty_id}: PR creation error — {str(exc)[:200]}"
                ],
            )

        pr_url: str = pr_data.get("html_url", "")
        pr_number: int = pr_data.get("number", 0)

        self._logger.info(
            "bounty_submit_complete",
            bounty_id=bounty_id,
            pr_url=pr_url,
            pr_number=pr_number,
            branch=branch_name,
            execution_id=context.execution_id,
        )

        # ── Step 7: Emit BOUNTY_PR_SUBMITTED ─────────────────────────────
        await self._emit_pr_submitted(
            bounty_id=bounty_id,
            bounty_url=bounty_url,
            pr_url=pr_url,
            pr_number=pr_number,
            repository_url=repository_url,
        )

        # ── Step 8: Store PR URL in Redis for MonitorPRsExecutor ──────────
        await self._cache_pr_url(bounty_id, pr_url, pr_number)

        return ExecutionResult(
            success=True,
            data={
                "bounty_id": bounty_id,
                "pr_url": pr_url,
                "pr_number": pr_number,
                "branch": branch_name,
                "fork_full_name": fork_full_name,
                "repository_url": repository_url,
                # economic_delta: will be filled in when bounty pays out via BOUNTY_PAID
                # Oikos registers the receivable from BOUNTY_PR_SUBMITTED event.
            },
            side_effects=[
                f"Bounty {bounty_id}: PR submitted to {repository_url} — {pr_url}"
            ],
            new_observations=[
                f"Bounty SUBMITTED: {bounty_id}. PR opened: {pr_url}. "
                f"Repository: {repository_url}. Awaiting maintainer review and merge for payout."
            ],
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _emit_pr_submitted(
        self,
        bounty_id: str,
        bounty_url: str,
        pr_url: str,
        pr_number: int,
        repository_url: str,
    ) -> None:
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.BOUNTY_PR_SUBMITTED,
                source_system="axon.bounty_submit",
                data={
                    "bounty_id": bounty_id,
                    "bounty_url": bounty_url,
                    "pr_url": pr_url,
                    "pr_number": pr_number,
                    "repository_url": repository_url,
                },
            ))
        except Exception as exc:
            self._logger.warning("bounty_pr_submitted_emit_failed", error=str(exc))

    async def _emit_credentials_missing(
        self,
        bounty_id: str,
        bounty_url: str,
    ) -> None:
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.GITHUB_CREDENTIALS_MISSING,
                source_system="axon.bounty_submit",
                data={
                    "bounty_id": bounty_id,
                    "bounty_url": bounty_url,
                    "message": (
                        "GitHub credentials are required for bounty PR submission. "
                        "Operator must set GITHUB_TOKEN or configure the GitHub App "
                        "(GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY + GITHUB_INSTALLATION_ID)."
                    ),
                },
            ))
        except Exception as exc:
            self._logger.warning("credentials_missing_emit_failed", error=str(exc))

    async def _cache_pr_url(
        self,
        bounty_id: str,
        pr_url: str,
        pr_number: int,
    ) -> None:
        """Store PR URL in Redis so MonitorPRsExecutor can track merge status."""
        if self._redis is None:
            return
        try:
            key = f"{_PR_REDIS_KEY_PREFIX}{bounty_id}"
            await self._redis.set_json(
                key,
                {"pr_url": pr_url, "pr_number": pr_number, "bounty_id": bounty_id},
                ttl=7 * 24 * 3600,   # 7 days — generous window for review + merge
            )
        except Exception as exc:
            # Cache failure is non-fatal; MonitorPRsExecutor can fall back to event bus
            self._logger.warning("pr_url_cache_failed", bounty_id=bounty_id, error=str(exc))


# ── Module-level helpers ───────────────────────────────────────────────────────


def _parse_owner_repo(url: str) -> tuple[str, str]:
    """
    Parse "owner/repo" from a GitHub URL or shorthand.

    Accepts:
      "owner/repo"
      "https://github.com/owner/repo"
      "https://github.com/owner/repo.git"

    Returns (owner, repo) tuple.
    Raises ValueError on unrecognised format.
    """
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]

    # Shorthand: owner/repo
    if "/" in url and not url.startswith("http"):
        parts = url.split("/")
        if len(parts) == 2:
            return parts[0], parts[1]

    # HTTPS: https://github.com/owner/repo
    match = re.match(r"https://github\.com/([^/]+)/([^/]+)$", url)
    if match:
        return match.group(1), match.group(2)

    raise ValueError(
        f"Cannot parse owner/repo from repository_url: {url!r}. "
        "Expected 'owner/repo' or 'https://github.com/owner/repo'."
    )


def _slugify(value: str) -> str:
    """Make value safe for use in a branch name."""
    return re.sub(r"[^a-zA-Z0-9_-]", "-", value)[:40].strip("-")


def _build_pr_body(
    bounty_id: str,
    bounty_url: str,
    explanation: str,
    solution_filename: str,
) -> str:
    """
    Compose the pull request body.

    Structure:
      ## Summary
      <explanation — truncated to _MAX_EXPLANATION_LEN>

      ## Changes
      - <filename>

      ## Bounty
      <issue link if provided>

      <EOS author block — UNCONDITIONAL>
    """
    explanation_safe = explanation[:_MAX_EXPLANATION_LEN]
    if len(explanation) > _MAX_EXPLANATION_LEN:
        explanation_safe += "\n\n*(explanation truncated)*"

    bounty_section = (
        f"\n## Bounty\n\nFixes: {bounty_url}\n" if bounty_url else ""
    )

    return (
        f"## Summary\n\n{explanation_safe}\n\n"
        f"## Changes\n\n- `{solution_filename}`\n"
        f"{bounty_section}"
        f"{_EOS_AUTHOR_BLOCK}"
    )
