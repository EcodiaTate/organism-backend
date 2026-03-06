"""
EcodiaOS — Axon MonitorPRsExecutor (PR Monitoring & Bounty Payout Loop)

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
  - Required autonomy: 1 (ADVISOR) — read-only GitHub API calls
  - Rate limit: 6 per hour — avoid hammering the GitHub API
  - Max duration: 30s — network-bound, multiple sequential API calls
  - No side effects beyond observations — pure information gathering
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
    from systems.synapse.service import SynapseService

logger = structlog.get_logger()

# Match GitHub PR URLs: https://github.com/{owner}/{repo}/pull/{number}
_PR_URL_RE = re.compile(
    r"https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)"
)


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

    required_autonomy = 1       # ADVISOR — read-only API calls
    reversible = False
    max_duration_ms = 30_000    # 30s — multiple sequential HTTP calls
    rate_limit = RateLimit.per_hour(6)

    def __init__(
        self,
        github_config: ExternalPlatformsConfig | None = None,
        synapse: SynapseService | None = None,
    ) -> None:
        self._github_config = github_config
        self._synapse = synapse
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
                "pending_prs is empty — nothing to monitor",
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

        # Build HTTP headers for GitHub API
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._github_config and self._github_config.github_token:
            headers["Authorization"] = f"Bearer {self._github_config.github_token}"

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
                        f"PR {pr_url} confirmed merged — bounty payout pending"
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

            # Parse reward amount — strip '$' and commas
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
