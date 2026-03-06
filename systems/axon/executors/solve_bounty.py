"""
EcodiaOS -- Axon SolveBountyExecutor (Phase 3 -- Bounty Hunter / Simula Integration)

Purpose-built executor that takes a bounty discovered by BountyHunterExecutor
and orchestrates the full solve-and-submit loop via Simula's evolution pipeline:

  1. Clone the target repository via TargetIngestor
  2. Analyse the codebase surface to build context
  3. Build an EvolutionProposal from the bounty issue
  4. Feed it through SimulaService.process_proposal() (simulate / apply / PR)
  5. Extract the PR URL from the resulting EvolutionRecord
  6. Return the PR info as an ExecutionResult for Nova

This is the "active foraging" executor:
  BountyHunterExecutor  ->  scans / filters / scores candidates
  SolveBountyExecutor   ->  clones repo, generates code, submits PR

BountyPolicy compliance:
  BountyHunterExecutor already verified ROI >= 2.0x and cost_pct <= 40%.
  This executor trusts that gating -- its job is execution, not re-evaluation.

Safety constraints:
  - Required autonomy: 3 (TRUSTED) -- generates code and creates PRs
  - Rate limit: 2 solves per hour -- each solve is a major LLM budget expenditure
  - Reversible: False -- submitted PRs cannot be atomically recalled
  - Max duration: 600s (10 min) -- repo clone + Simula pipeline + PR creation
  - Constitutional review via Equor is mandatory (enforced by Axon pipeline)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from systems.axon.executor import Executor
from systems.axon.types import (
    ExecutionContext,
    ExecutionResult,
    RateLimit,
    ValidationResult,
)
from systems.simula.evolution_types import (
    ChangeCategory,
    ChangeSpec,
    EvolutionProposal,
    ProposalStatus,
)

if TYPE_CHECKING:
    from clients.llm import LLMProvider
    from systems.identity.connectors.github_app import GitHubAppConnector
    from systems.simula.service import SimulaService

logger = structlog.get_logger()


class SolveBountyExecutor(Executor):
    """
    Solve a bounty by cloning the target repo, generating code via Simula's
    full evolution pipeline, and submitting a PR.

    Takes a bounty that has already passed BountyPolicy evaluation
    (from BountyHunterExecutor) and attempts to complete it autonomously.

    Required params:
      bounty_id (str): Unique bounty identifier from BountyHunterExecutor
      issue_url (str): Full HTTPS URL to the GitHub/Algora issue
      repository_url (str): HTTPS clone URL or "owner/repo" shorthand
      title (str): Issue title (used in proposal description)
      description (str): Issue body (used in proposal context)

    Optional params:
      reward_usd (float): Bounty reward in USD (for logging / observations)
      difficulty (str): Classified difficulty tier from BountyHunterExecutor
      labels (list[str]): Issue labels (for Simula context)
      platform (str): Source platform ("github" | "algora")

    Returns ExecutionResult with:
      data:
        bounty_id         -- the bounty that was attempted
        pr_url            -- URL of the submitted pull request
        pr_number         -- PR number (int) if available
        repository_url    -- the target repository
        proposal_id       -- Simula EvolutionProposal ID
        files_changed     -- list of files modified
        proposal_status   -- final ProposalStatus value
      side_effects:
        -- Human-readable summary of the solve attempt
      new_observations:
        -- Fed back as Percept so Nova can track foraging success
    """

    action_type = "axon.solve_bounty"
    description = (
        "Clone a bounty target repo, generate a fix via Simula's evolution "
        "pipeline, and submit a pull request (Phase 3 Bounty/Simula integration)"
    )

    required_autonomy = 3          # TRUSTED -- generates code, creates external PRs
    reversible = False             # Submitted PRs cannot be atomically reversed
    max_duration_ms = 600_000      # 10 minutes -- clone + Simula pipeline + PR
    rate_limit = RateLimit.per_hour(2)  # Major LLM budget per solve

    def __init__(
        self,
        simula: SimulaService | None = None,
        llm: LLMProvider | None = None,
        github_connector: GitHubAppConnector | None = None,
    ) -> None:
        self._simula = simula
        self._llm = llm
        self._github_connector = github_connector
        self._logger = logger.bind(executor="axon.solve_bounty")

    # -- Validation -----------------------------------------------------------

    async def validate_params(self, params: dict[str, Any]) -> ValidationResult:
        """Validate solve parameters -- no I/O."""
        bounty_id = str(params.get("bounty_id", "")).strip()
        if not bounty_id:
            return ValidationResult.fail(
                "bounty_id is required",
                bounty_id="missing",
            )

        issue_url = str(params.get("issue_url", "")).strip()
        if not issue_url:
            return ValidationResult.fail(
                "issue_url is required (full URL to the bounty issue)",
                issue_url="missing",
            )
        if not issue_url.startswith("https://"):
            return ValidationResult.fail(
                "issue_url must be an HTTPS URL",
                issue_url="not https",
            )

        repository_url = str(params.get("repository_url", "")).strip()
        if not repository_url:
            return ValidationResult.fail(
                "repository_url is required "
                "(e.g., 'https://github.com/owner/repo' or 'owner/repo')",
                repository_url="missing",
            )

        title = str(params.get("title", "")).strip()
        if not title:
            return ValidationResult.fail(
                "title is required (issue title for context)",
                title="missing",
            )

        description = str(params.get("description", "")).strip()
        if not description:
            return ValidationResult.fail(
                "description is required (issue body for Simula context)",
                description="missing",
            )

        return ValidationResult.ok()

    # -- Execution ------------------------------------------------------------

    async def execute(
        self,
        params: dict[str, Any],
        context: ExecutionContext,
    ) -> ExecutionResult:
        """
        Solve a bounty issue via the full Simula evolution pipeline.

        Pipeline:
          1. Guard: verify SimulaService is wired
          2. Clone/analyse the target repo via TargetIngestor
          3. Build an EvolutionProposal with source="bounty"
          4. Feed the proposal through SimulaService.process_proposal()
          5. Extract the PR URL from the EvolutionRecord (via history)
          6. Return ExecutionResult with PR info

        Never raises -- failures returned in result.
        """
        bounty_id = str(params["bounty_id"]).strip()
        issue_url = str(params["issue_url"]).strip()
        repository_url = str(params["repository_url"]).strip()
        title = str(params["title"]).strip()
        description = str(params["description"]).strip()[:2000]
        reward_usd = float(params.get("reward_usd", 0.0))
        difficulty = str(params.get("difficulty", "unknown")).strip()
        labels = list(params.get("labels", []))
        platform = str(params.get("platform", "github")).strip()

        self._logger.info(
            "solve_bounty_started",
            bounty_id=bounty_id,
            issue_url=issue_url,
            repository_url=repository_url,
            reward_usd=reward_usd,
            difficulty=difficulty,
            execution_id=context.execution_id,
        )

        # -- Guard: SimulaService must be wired --------------------------------
        if self._simula is None:
            self._logger.warning("solve_bounty_no_simula", bounty_id=bounty_id)
            return ExecutionResult(
                success=False,
                error=(
                    "SolveBountyExecutor requires SimulaService. "
                    "Call AxonService.set_simula_service() before executing."
                ),
            )

        # -- Acquire GitHub Installation Access Token --------------------------
        # The GitHubAppConnector mints a fresh JWT and exchanges it for an IAT
        # (valid 1 hour). We pass this token to TargetIngestor so that private
        # repos and higher-rate API calls work without a static PAT.
        github_token: str | None = None
        if self._github_connector is not None:
            github_token = await self._github_connector.get_access_token()
            if github_token:
                self._logger.info(
                    "github_iat_acquired",
                    bounty_id=bounty_id,
                    connector=self._github_connector.platform_id,
                )
            else:
                self._logger.warning(
                    "github_iat_unavailable",
                    bounty_id=bounty_id,
                    reason="GitHubAppConnector.get_access_token() returned None",
                )
                return ExecutionResult(
                    success=False,
                    error=(
                        "SolveBountyExecutor requires a valid GitHub IAT but "
                        "GitHubAppConnector returned no token. "
                        "Verify the GitHub App is installed and the connector is active."
                    ),
                    data={"bounty_id": bounty_id, "stage": "github_iat"},
                )
        else:
            self._logger.warning(
                "solve_bounty_no_github_connector",
                bounty_id=bounty_id,
                note="Proceeding without IAT; clone may fail for private repos",
            )

        # -- Step 1: Clone and analyse the target repository -------------------
        github_url = self._normalise_github_url(repository_url)

        try:
            from systems.simula.inspector.ingestor import TargetIngestor

            ingestor = await TargetIngestor.ingest_from_github(
                github_url=github_url,
                llm=self._llm,
                clone_depth=1,
                # Pass the IAT so the ingestor can authenticate git clone and
                # any GitHub API calls (surface discovery, etc.).
                github_token=github_token,
            )
            self._logger.info(
                "target_repo_cloned",
                bounty_id=bounty_id,
                workspace_root=str(ingestor.workspace.root),
            )
        except Exception as exc:
            self._logger.error(
                "target_repo_clone_failed",
                bounty_id=bounty_id,
                github_url=github_url,
                error=str(exc),
            )
            return ExecutionResult(
                success=False,
                error=f"Failed to clone target repository: {exc}",
                data={"bounty_id": bounty_id, "stage": "clone"},
                new_observations=[
                    f"Bounty solve FAILED for '{title}': "
                    f"could not clone {github_url} -- {str(exc)[:200]}"
                ],
            )

        # Discover attack surfaces for extra context (best-effort)
        surfaces_context = ""
        try:
            surfaces = await ingestor.discover_surfaces()
            if surfaces:
                surface_lines = [
                    f"  - {s.surface_type.value}: {s.name} ({s.file_path}:{s.line_number})"
                    for s in surfaces[:20]
                ]
                surfaces_context = (
                    f"\n\nDiscovered {len(surfaces)} attack surfaces:\n"
                    + "\n".join(surface_lines)
                )
                self._logger.info(
                    "surfaces_discovered",
                    bounty_id=bounty_id,
                    count=len(surfaces),
                )
        except Exception as exc:
            self._logger.debug("surface_discovery_skipped", error=str(exc))

        # -- Step 2: Build EvolutionProposal -----------------------------------
        label_context = f"Labels: {', '.join(labels)}" if labels else ""

        proposal_description = (
            f"Bounty fix for: {title}\n\n"
            f"Issue: {issue_url}\n"
            f"Repository: {repository_url}\n"
            f"Difficulty: {difficulty}\n"
            f"{label_context}\n\n"
            f"Description:\n{description}"
            f"{surfaces_context}"
        )

        proposal = EvolutionProposal(
            source="bounty",
            category=ChangeCategory.ADD_SYSTEM_CAPABILITY,
            description=proposal_description,
            change_spec=ChangeSpec(
                capability_description=(
                    f"Fix issue: {title}\n\n{description[:1000]}"
                ),
                additional_context=(
                    f"Target repository: {github_url}\n"
                    f"Issue URL: {issue_url}\n"
                    f"Bounty ID: {bounty_id}\n"
                    f"Difficulty: {difficulty}\n"
                    # Provide the IAT to Simula's applicator so it can
                    # authenticate PR creation without a static PAT.
                    + (f"github_installation_token: {github_token}\n" if github_token else "")
                ),
                code_hint=(
                    "The repository has been cloned to the local workspace. "
                    "Understand the codebase, implement the fix described in "
                    "the issue, write tests where appropriate, and the "
                    "applicator will submit a pull request."
                ),
            ),
            expected_benefit=(
                f"Complete bounty worth ${reward_usd:.0f} on {platform}. "
                f"Estimated difficulty: {difficulty}."
            ),
            risk_assessment="External repository change -- PR requires maintainer review.",
            source_bounty_id=bounty_id,
            target_repository_url=github_url,
            # Direct Simula to operate on the cloned repo, not the organism's own codebase.
            workspace_root=str(ingestor.workspace.root),
        )

        self._logger.info(
            "proposal_built",
            bounty_id=bounty_id,
            proposal_id=proposal.id,
        )

        # -- Step 3: Feed through Simula's evolution pipeline ------------------
        try:
            result = await self._simula.process_proposal(proposal)
        except Exception as exc:
            self._logger.error(
                "simula_process_proposal_failed",
                bounty_id=bounty_id,
                proposal_id=proposal.id,
                error=str(exc),
            )
            return ExecutionResult(
                success=False,
                error=f"Simula evolution pipeline failed: {exc}",
                data={
                    "bounty_id": bounty_id,
                    "proposal_id": proposal.id,
                    "stage": "process_proposal",
                },
                new_observations=[
                    f"Bounty solve FAILED for '{title}': "
                    f"Simula pipeline error -- {str(exc)[:200]}"
                ],
            )

        # -- Step 4: Extract PR URL from the EvolutionRecord -------------------
        pr_url = ""
        pr_number: int | None = None

        if result.status == ProposalStatus.APPLIED:
            # The ProposalResult may carry pr_url/pr_number if the base types
            # have been updated (Phase 3 Terminal 1). Fall back to history query.
            pr_url = getattr(result, "pr_url", "") or ""
            pr_number = getattr(result, "pr_number", None)

            # Fallback: query the most recent EvolutionRecord for this proposal
            if not pr_url:
                try:
                    records = await self._simula.get_history(limit=5)
                    for record in records:
                        if record.proposal_id == proposal.id:
                            pr_url = record.pr_url
                            pr_number = record.pr_number
                            break
                except Exception as exc:
                    self._logger.debug(
                        "history_lookup_failed",
                        proposal_id=proposal.id,
                        error=str(exc),
                    )

        # -- Step 5: Build the response ----------------------------------------
        if result.status != ProposalStatus.APPLIED:
            reason = result.reason or f"Proposal ended with status: {result.status.value}"
            self._logger.warning(
                "solve_bounty_proposal_not_applied",
                bounty_id=bounty_id,
                proposal_id=proposal.id,
                status=result.status.value,
                reason=reason,
            )
            return ExecutionResult(
                success=False,
                error=f"Simula proposal was not applied: {reason}",
                data={
                    "bounty_id": bounty_id,
                    "proposal_id": proposal.id,
                    "proposal_status": result.status.value,
                    "governance_record_id": result.governance_record_id,
                    "stage": "proposal_outcome",
                },
                new_observations=[
                    f"Bounty solve INCOMPLETE for '{title}' ({bounty_id}): "
                    f"proposal {result.status.value} -- {reason[:200]}"
                ],
            )

        if not pr_url:
            self._logger.warning(
                "solve_bounty_no_pr_url",
                bounty_id=bounty_id,
                proposal_id=proposal.id,
            )
            return ExecutionResult(
                success=False,
                error="Simula applied the proposal but no PR URL was produced",
                data={
                    "bounty_id": bounty_id,
                    "proposal_id": proposal.id,
                    "proposal_status": result.status.value,
                    "files_changed": result.files_changed,
                    "stage": "pr_extraction",
                },
                new_observations=[
                    f"Bounty solve INCOMPLETE for '{title}' ({bounty_id}): "
                    f"code generated ({len(result.files_changed)} files) but PR submission failed."
                ],
            )

        # -- Success -----------------------------------------------------------
        self._logger.info(
            "solve_bounty_complete",
            bounty_id=bounty_id,
            proposal_id=proposal.id,
            pr_url=pr_url,
            pr_number=pr_number,
            files_changed=len(result.files_changed),
            reward_usd=reward_usd,
            execution_id=context.execution_id,
        )
        side_effect = (
            f"Bounty solved [{bounty_id}]: PR submitted to {repository_url}. "
            f"PR: {pr_url}. "
            f"Files changed: {len(result.files_changed)}. "
            f"Reward: ${reward_usd:.0f} "
            f"(platform: {platform}, difficulty: {difficulty})."
        )

        observation = (
            f"Bounty SOLVED: '{title}' on {platform}. "
            f"PR submitted: {pr_url}. "
            f"Repository: {repository_url}. "
            f"Reward: ${reward_usd:.0f}. "
            f"Files changed: {', '.join(result.files_changed[:10])}. "
            f"Awaiting review and merge for payout."
        )

        return ExecutionResult(
            success=True,
            data={
                "bounty_id": bounty_id,
                "pr_url": pr_url,
                "pr_number": pr_number,
                "repository_url": repository_url,
                "issue_url": issue_url,
                "proposal_id": proposal.id,
                "proposal_status": result.status.value,
                "files_changed": result.files_changed,
                "reward_usd": reward_usd,
                "platform": platform,
                "difficulty": difficulty,
            },
            side_effects=[side_effect],
            new_observations=[observation],
        )

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _normalise_github_url(raw: str) -> str:
        """
        Normalise a repository identifier to a full GitHub HTTPS URL.

        Accepts:
          - "owner/repo"                 -> "https://github.com/owner/repo"
          - "https://github.com/o/r"     -> "https://github.com/o/r" (pass-through)
          - "https://github.com/o/r.git" -> "https://github.com/o/r" (strip .git)
        """
        url = raw.strip()
        if url.endswith(".git"):
            url = url[:-4]
        if not url.startswith("https://"):
            url = f"https://github.com/{url}"
        return url
