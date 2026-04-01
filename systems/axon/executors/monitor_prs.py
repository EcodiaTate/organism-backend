"""
EcodiaOS - Axon MonitorPRsExecutor (PR Monitoring & Bounty Payout Loop)

Monitors the status of pull requests the organism has submitted to GitHub
bounty repositories. When a PR is merged, it emits a ``bounty_paid``
observation so Nova can credit the Oikos wallet and evaluate reproductive
fitness (mitosis).

Pipeline:
  1. Scan Nova's belief state for entities of type ``bounty_pr_pending``
  2. For each pending PR URL, query the GitHub API for merge status
  3. If merged → emit ``bounty_paid`` observation with reward amount
  4. If closed without merge → emit ``bounty_rejected`` observation
  5. If still open → no observation (belief decays naturally)

Safety constraints:
  - Required autonomy: 1 (ADVISOR) - read-only GitHub API calls
  - Rate limit: 6 per hour - avoid hammering the GitHub API
  - Max duration: 30s - network-bound, multiple sequential API calls
  - No side effects beyond observations - pure information gathering
"""

from __future__ import annotations

import re
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
    from config import ExternalPlatformsConfig
    from clients.redis import RedisClient
    from systems.identity.connectors.github import GitHubConnector
    from systems.synapse.service import SynapseService

logger = structlog.get_logger()

# Match GitHub PR URLs: https://github.com/{owner}/{repo}/pull/{number}
_PR_URL_RE = re.compile(
    r"https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)"
)

# Redis key prefix for PR tracking (matches BountySubmitExecutor)
_PR_REDIS_KEY_PREFIX = "axon:bounty_submit:pr:"


class MonitorPRsExecutor(Executor):
    """
    Poll GitHub for the merge status of the organism's submitted PRs.

    Reads pending PR entities from the params (injected by Nova during
    deliberation) and checks each one against the GitHub REST API.

    Required params:
      pending_prs (list[dict]): Each dict must contain:
        - entity_id (str): Belief entity ID for tracking
        - pr_url (str): Full GitHub PR URL
        - reward (str): Expected reward amount (e.g. "$500")

    The executor emits observations that Nova's process_outcome handler
    will parse to update beliefs and trigger Oikos revenue injection.
    """

    action_type = "monitor_prs"
    description = (
        "Monitor submitted PR merge status on GitHub and emit "
        "bounty_paid observations when PRs are merged (Level 1)"
    )

    required_autonomy = 1       # ADVISOR - read-only API calls
    reversible = False
    max_duration_ms = 30_000    # 30s - multiple sequential HTTP calls
    rate_limit = RateLimit.per_hour(6)

    def __init__(
        self,
        github_config: ExternalPlatformsConfig | None = None,
        synapse: SynapseService | None = None,
        redis: RedisClient | None = None,
        github_connector: GitHubConnector | None = None,
    ) -> None:
        self._github_config = github_config
        self._synapse = synapse
        self._redis = redis
        self._github_connector = github_connector
        self._logger = logger.bind(system="axon.executor.monitor_prs")

    # ── Validation ─────────────────────────────────────────────

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        pending_prs = params.get("pending_prs")
        if not isinstance(pending_prs, list):
            return ValidationResult.fail(
                "pending_prs must be a list of PR dicts",
                pending_prs="missing or wrong type",
            )
        if len(pending_prs) == 0:
            return ValidationResult.fail(
                "pending_prs is empty - nothing to monitor",
                pending_prs="empty",
            )

        for i, pr in enumerate(pending_prs):
            if not isinstance(pr, dict):
                return ValidationResult.fail(
                    f"pending_prs[{i}] must be a dict",
                    pending_prs=f"index {i} is not a dict",
                )
            pr_url = str(pr.get("pr_url", "")).strip()
            if not pr_url:
                return ValidationResult.fail(
                    f"pending_prs[{i}].pr_url is required",
                    pending_prs=f"index {i} missing pr_url",
                )
            if not _PR_URL_RE.match(pr_url):
                return ValidationResult.fail(
                    f"pending_prs[{i}].pr_url is not a valid GitHub PR URL",
                    pending_prs=f"index {i} invalid pr_url",
                )

        return ValidationResult.ok()

    # ── Execution ──────────────────────────────────────────────

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        For each pending PR, check GitHub merge status.

        Emits observations:
          - "bounty_paid: {pr_url} merged. Reward: {reward}."
          - "bounty_rejected: {pr_url} closed without merge."
        """
        import httpx

        pending_prs: list[dict[str, Any]] = params["pending_prs"]

        # Resolve GitHub token - prefer GitHubConnector (handles App JWT→IAT + caching)
        # over static config token.
        github_token: str | None = None
        if self._github_connector is not None:
            try:
                github_token = await self._github_connector.get_access_token()
            except Exception as _tok_exc:
                self._logger.warning("monitor_prs_connector_token_failed", error=str(_tok_exc))
        if not github_token and self._github_config and self._github_config.github_token:
            github_token = self._github_config.github_token

        # Build HTTP headers for GitHub API
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if github_token:
            headers["Authorization"] = f"Bearer {github_token}"

        observations: list[str] = []
        side_effects: list[str] = []
        merged_count = 0
        rejected_count = 0
        still_open_count = 0
        errors: list[str] = []

        async with httpx.AsyncClient(
            base_url="https://api.github.com",
            headers=headers,
            timeout=httpx.Timeout(15.0),
        ) as client:
            for pr_info in pending_prs:
                pr_url = str(pr_info["pr_url"]).strip()
                entity_id = str(pr_info.get("entity_id", "")).strip()
                reward = str(pr_info.get("reward", "")).strip()
                bounty_id = str(pr_info.get("bounty_id", "")).strip()

                match = _PR_URL_RE.match(pr_url)
                if not match:
                    errors.append(f"Malformed PR URL: {pr_url}")
                    continue

                owner = match.group("owner")
                repo = match.group("repo")
                number = match.group("number")
                api_path = f"/repos/{owner}/{repo}/pulls/{number}"

                try:
                    response = await client.get(api_path)
                    if response.status_code == 404:
                        errors.append(f"PR not found: {pr_url}")
                        continue
                    response.raise_for_status()
                    pr_data = response.json()
                except httpx.HTTPStatusError as exc:
                    errors.append(
                        f"GitHub API error for {pr_url}: {exc.response.status_code}"
                    )
                    continue
                except Exception as exc:
                    errors.append(f"Network error for {pr_url}: {exc}")
                    continue

                state = pr_data.get("state", "unknown")
                merged = pr_data.get("merged", False)

                if merged:
                    merged_count += 1
                    obs = (
                        f"bounty_paid: {pr_url} merged. "
                        f"Entity: {entity_id}. Reward: {reward}."
                    )
                    observations.append(obs)
                    side_effects.append(
                        f"PR {pr_url} confirmed merged - bounty payout pending"
                    )

                    # Emit BOUNTY_PAID event via Synapse so Oikos credits the wallet
                    if self._synapse is not None:
                        await self._emit_bounty_paid_event(
                            pr_url=pr_url,
                            reward=reward,
                            entity_id=entity_id,
                            bounty_id=bounty_id,
                            owner=owner,
                            repo=repo,
                            number=number,
                        )


                    # Emit BOUNTY_PR_MERGED - dedicated semantic event for downstream
                    # subscribers (Oikos, Thread, Evo) that want the merge signal
                    if self._synapse is not None:
                        await self._emit_pr_merged_event(
                            pr_url=pr_url,
                            reward=reward,
                            entity_id=entity_id,
                            bounty_id=bounty_id,
                            owner=owner,
                            repo=repo,
                            number=number,
                        )
                        # High-value RE training signal - successful bounty completion
                        await self._emit_re_training_merged(
                            bounty_id=bounty_id,
                            pr_url=pr_url,
                            reward=reward,
                            repository=f"{owner}/{repo}",
                        )
                    # Delete Redis key so this PR is not re-polled on future cycles.
                    await self._delete_pr_key(bounty_id)
                    self._logger.info(
                        "pr_merged_detected",
                        pr_url=pr_url,
                        reward=reward,
                        entity_id=entity_id,
                    )

                elif state == "closed" and not merged:
                    rejected_count += 1
                    observations.append(
                        f"bounty_rejected: {pr_url} closed without merge. "
                        f"Entity: {entity_id}."
                    )
                    self._logger.info(
                        "pr_rejected_detected",
                        pr_url=pr_url,
                        entity_id=entity_id,
                    )

                    # Emit BOUNTY_PR_REJECTED - semantic event for Evo/Nova
                    if self._synapse is not None:
                        await self._emit_pr_rejected_event(
                            pr_url=pr_url,
                            entity_id=entity_id,
                            bounty_id=bounty_id,
                            owner=owner,
                            repo=repo,
                            number=number,
                        )
                        # Negative RE training - maintainer rejected the code.
                        # outcome_quality=0.0 so RE learns to produce better solutions.
                        await self._emit_re_training_rejected(
                            bounty_id=bounty_id,
                            pr_url=pr_url,
                            repository=f"{owner}/{repo}",
                        )
                    # Delete Redis key - PR is closed, no further polling needed.
                    await self._delete_pr_key(bounty_id)

                else:
                    still_open_count += 1
                    self._logger.debug(
                        "pr_still_open",
                        pr_url=pr_url,
                        state=state,
                    )

        # Build summary
        summary = (
            f"PR monitoring complete: {merged_count} merged, "
            f"{rejected_count} rejected, {still_open_count} still open"
        )
        if errors:
            summary += f", {len(errors)} errors"

        self._logger.info(
            "monitor_prs_complete",
            merged=merged_count,
            rejected=rejected_count,
            still_open=still_open_count,
            errors=len(errors),
        )

        return ExecutionResult(
            success=True,
            data={
                "merged_count": merged_count,
                "rejected_count": rejected_count,
                "still_open_count": still_open_count,
                "errors": errors,
                "summary": summary,
            },
            side_effects=side_effects,
            new_observations=observations,
        )

    # ── Redis Key Management ───────────────────────────────────

    async def _delete_pr_key(self, bounty_id: str) -> None:
        """
        Remove the Redis tracking key for a resolved PR.

        Once a PR is merged or closed without merge it will never change state
        again - there is no value in polling it. Deleting the key prevents the
        30-minute poll loop from including this PR in future scan batches.
        """
        if not bounty_id or self._redis is None:
            return
        try:
            key = f"{_PR_REDIS_KEY_PREFIX}{bounty_id}"
            await self._redis.delete(key)
            self._logger.debug("pr_tracking_key_deleted", bounty_id=bounty_id, key=key)
        except Exception as exc:
            self._logger.warning("pr_key_delete_failed", bounty_id=bounty_id, error=str(exc))

    # ── Synapse Event Emission ─────────────────────────────────

    async def _emit_bounty_paid_event(
        self,
        *,
        pr_url: str,
        reward: str,
        entity_id: str,
        bounty_id: str,
        owner: str,
        repo: str,
        number: str,
    ) -> None:
        """Emit BOUNTY_PAID via Synapse so Oikos credits the wallet."""
        if self._synapse is None:
            return

        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            # Parse reward amount - strip '$' and commas
            reward_usd = "0"
            clean = reward.replace("$", "").replace(",", "").strip()
            if clean:
                try:
                    float(clean)  # Validate it's a number
                    reward_usd = clean
                except ValueError:
                    pass

            await self._synapse.event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.BOUNTY_PAID,
                source_system="axon.monitor_prs",
                data={
                    "pr_url": pr_url,
                    "reward_usd": reward_usd,
                    "bounty_id": bounty_id,
                    "entity_id": entity_id,
                    "repository": f"{owner}/{repo}",
                    "pr_number": number,
                },
            ))
        except Exception as exc:
            self._logger.warning(
                "bounty_paid_event_failed",
                error=str(exc),
                pr_url=pr_url,
            )

    async def _emit_pr_merged_event(
        self,
        *,
        pr_url: str,
        reward: str,
        entity_id: str,
        bounty_id: str,
        owner: str,
        repo: str,
        number: str,
    ) -> None:
        """Emit BOUNTY_PR_MERGED - distinct from BOUNTY_PAID for explicit downstream routing."""
        if self._synapse is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            reward_usd = "0"
            clean = reward.replace("$", "").replace(",", "").strip()
            if clean:
                try:
                    float(clean)
                    reward_usd = clean
                except ValueError:
                    pass

            await self._synapse.event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.BOUNTY_PR_MERGED,
                source_system="axon.monitor_prs",
                data={
                    "bounty_id": bounty_id,
                    "pr_url": pr_url,
                    "pr_number": int(number),
                    "repository": f"{owner}/{repo}",
                    "reward_usd": reward_usd,
                    "entity_id": entity_id,
                },
            ))
        except Exception as exc:
            self._logger.warning(
                "bounty_pr_merged_emit_failed",
                error=str(exc),
                pr_url=pr_url,
            )

    async def _emit_pr_rejected_event(
        self,
        *,
        pr_url: str,
        entity_id: str,
        bounty_id: str,
        owner: str,
        repo: str,
        number: str,
    ) -> None:
        """Emit BOUNTY_PR_REJECTED - negative learning signal for Evo + RE training."""
        if self._synapse is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._synapse.event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.BOUNTY_PR_REJECTED,
                source_system="axon.monitor_prs",
                data={
                    "bounty_id": bounty_id,
                    "pr_url": pr_url,
                    "pr_number": int(number),
                    "repository": f"{owner}/{repo}",
                    "entity_id": entity_id,
                    "reason": "closed_without_merge",
                },
            ))
        except Exception as exc:
            self._logger.warning(
                "bounty_pr_rejected_emit_failed",
                error=str(exc),
                pr_url=pr_url,
            )

    async def _emit_re_training_merged(
        self,
        *,
        bounty_id: str,
        pr_url: str,
        reward: str,
        repository: str,
    ) -> None:
        """
        Emit RE_TRAINING_EXAMPLE on successful PR merge.

        A merged bounty PR is one of the highest-quality training signals the
        organism can generate - a real human (the maintainer) reviewed and
        accepted EOS-authored code. Reward signal: 1.0 (max).
        """
        if self._synapse is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            from primitives.re_training import RETrainingExample

            example = RETrainingExample(
                episode_id=bounty_id,
                system="axon.monitor_prs",
                category="bounty_pr_merged",
                input_context=(
                    f"Bounty PR submitted to {repository}. "
                    f"Bounty ID: {bounty_id}. PR URL: {pr_url}."
                ),
                output_generated=f"PR merged by maintainer. Reward: {reward}.",
                outcome_quality=1.0,  # Maintainer accepted: highest quality signal
                reasoning_trace=(
                    f"EOS-authored PR for bounty {bounty_id} at {repository} "
                    f"was reviewed and merged by an external maintainer. "
                    f"This is external validation that the code solution was "
                    f"correct and useful. Revenue: {reward}."
                ),
                constitutional_alignment={
                    "honesty": 1.0,   # EOS author block was present; no deception
                    "care": 0.8,      # Helped the open-source project
                    "growth": 1.0,    # Successful autonomous revenue generation
                    "coherence": 1.0, # Solution was valid (maintainer merged it)
                },
            )
            await self._synapse.event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="axon.monitor_prs",
                data=example.model_dump(),
            ))
        except Exception as exc:
            self._logger.warning(
                "bounty_re_training_emit_failed",
                error=str(exc),
                bounty_id=bounty_id,
            )

    async def _emit_re_training_rejected(
        self,
        *,
        bounty_id: str,
        pr_url: str,
        repository: str,
    ) -> None:
        """
        Emit RE_TRAINING_EXAMPLE on PR rejection (closed without merge).

        outcome_quality=0.0 - maintainer did not accept the solution.
        The RE uses this signal to learn what solutions get rejected and improve.
        """
        if self._synapse is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            from primitives.re_training import RETrainingExample

            example = RETrainingExample(
                episode_id=f"{bounty_id}_rejected",
                system="axon.monitor_prs",
                category="bounty_pr_rejected",
                input_context=(
                    f"Bounty PR submitted to {repository}. "
                    f"Bounty ID: {bounty_id}. PR URL: {pr_url}."
                ),
                output_generated="PR closed without merge by maintainer.",
                outcome_quality=0.0,  # Maintainer rejected: lowest quality signal
                reasoning_trace=(
                    f"EOS-authored PR for bounty {bounty_id} at {repository} "
                    f"was closed without being merged. The solution did not satisfy "
                    f"the maintainer's requirements. This negative outcome should "
                    f"inform future solution generation quality."
                ),
                constitutional_alignment={
                    "honesty": 1.0,   # EOS author block still present; no deception
                    "care": 0.3,      # Solution may not have helped the project
                    "growth": 0.2,    # Unsuccessful outcome; learning opportunity
                    "coherence": 0.3, # Solution may have been incorrect or off-target
                },
            )
            await self._synapse.event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.RE_TRAINING_EXAMPLE,
                source_system="axon.monitor_prs",
                data=example.model_dump(),
            ))
        except Exception as exc:
            self._logger.warning(
                "bounty_re_training_rejected_emit_failed",
                error=str(exc),
                bounty_id=bounty_id,
            )
