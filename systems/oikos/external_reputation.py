"""
EcodiaOS - Oikos: External Code Reputation Tracker (Phase 16s)

Tracks EOS's reputation as a software contractor in the open-source ecosystem.
Per-repository and per-language reputation scores feed into Evo domain mastery
signals and inform Nova about which domains to pursue next.

Data model:
  - Per-repository: PRs submitted / merged / rejected, last_activity
  - Per-language: reputation score (0.0–1.0) based on merge/submit ratio
  - Overall contractor score: weighted average across languages

Synapse events consumed:
  - EXTERNAL_TASK_COMPLETED  - task finished; may or may not have a PR
  - BOUNTY_PR_MERGED         - PR was accepted by a human maintainer (positive signal)
  - BOUNTY_PR_REJECTED       - PR was closed without merge (negative signal)

Synapse events emitted:
  - EXTERNAL_CODE_REPUTATION_UPDATED  - after any reputation change
  - RE_TRAINING_EXAMPLE               - for learning from PR outcomes

Persistence: Redis key ``oikos:external_reputation``

Integration:
  - Emits DOMAIN_EPISODE_RECORDED (Evo) on significant reputation changes
    so Evo can update domain mastery hypothesis priors.
  - No direct cross-system imports - EventBus and RedisClient under TYPE_CHECKING.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import Field

from primitives.common import EOSBaseModel, utc_now
from systems.synapse.types import SynapseEventType

if TYPE_CHECKING:
    from clients.redis import RedisClient
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("oikos.external_reputation")

# Redis key for persisting reputation state
_REDIS_KEY = "oikos:external_reputation"

# Score weight for a PR merge vs. submit (higher = merge is more impactful)
_MERGE_WEIGHT = 1.0
_REJECT_WEIGHT = -0.3

# Decay applied to old scores each update cycle (exponential moving average)
_EMA_ALPHA = 0.1  # New signal weight; (1 - alpha) = weight of history

# Minimum submits before a reputation score is considered reliable
_MIN_SUBMITS_FOR_SCORE = 3


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class RepoStats(EOSBaseModel):
    """Per-repository submission statistics."""

    repo_url: str
    language: str = "unknown"
    submitted: int = 0
    merged: int = 0
    rejected: int = 0
    last_activity: str = Field(default_factory=_now_iso)

    @property
    def merge_rate(self) -> float:
        if self.submitted == 0:
            return 0.0
        return self.merged / self.submitted

    @property
    def reject_rate(self) -> float:
        if self.submitted == 0:
            return 0.0
        return self.rejected / self.submitted


class LanguageReputation(EOSBaseModel):
    """Aggregate reputation score for a programming language (0.0–1.0)."""

    language: str
    score: float = 0.5         # Start at neutral
    total_submitted: int = 0
    total_merged: int = 0
    total_rejected: int = 0
    last_updated: str = Field(default_factory=_now_iso)

    @property
    def reliable(self) -> bool:
        return self.total_submitted >= _MIN_SUBMITS_FOR_SCORE


class ExternalReputationState(EOSBaseModel):
    """Full persisted reputation state."""

    repos: dict[str, RepoStats] = Field(default_factory=dict)
    languages: dict[str, LanguageReputation] = Field(default_factory=dict)
    overall_score: float = 0.5
    total_tasks_completed: int = 0
    total_submitted: int = 0
    total_merged: int = 0
    total_rejected: int = 0
    last_updated: str = Field(default_factory=_now_iso)


class ExternalCodeReputationTracker:
    """
    Tracks EOS reputation as an external software contractor.

    Wired into OikosService at boot. Subscribes to external task events via
    the Synapse bus. Persists state to Redis.
    """

    def __init__(self) -> None:
        self._redis: RedisClient | None = None
        self._event_bus: EventBus | None = None
        self._state = ExternalReputationState()
        self._logger = logger.bind(subsystem="external_reputation")

    # ── Dependency injection ───────────────────────────────────────────────

    def set_redis(self, redis: RedisClient) -> None:
        self._redis = redis

    def set_event_bus(self, bus: EventBus) -> None:
        self._event_bus = bus

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Load persisted state from Redis on boot."""
        await self._load_state()
        self._logger.info("external_reputation_initialized", repos=len(self._state.repos))

    async def attach(self) -> None:
        """Subscribe to Synapse events."""
        if self._event_bus is None:
            return
        await self._event_bus.subscribe(
            SynapseEventType.EXTERNAL_TASK_COMPLETED, self._on_task_completed
        )
        await self._event_bus.subscribe(
            SynapseEventType.BOUNTY_PR_MERGED, self._on_pr_merged
        )
        await self._event_bus.subscribe(
            SynapseEventType.BOUNTY_PR_REJECTED, self._on_pr_rejected
        )

    # ── Event handlers ─────────────────────────────────────────────────────

    async def _on_task_completed(self, event: Any) -> None:
        payload = getattr(event, "payload", {})
        repo_url = str(payload.get("repo_url", "")).strip()
        language = str(payload.get("language", "unknown"))
        bounty_id = str(payload.get("bounty_id", "")).strip()

        if not repo_url:
            return

        self._state.total_tasks_completed += 1

        # Only register a PR submission if a PR was actually submitted
        if bounty_id and payload.get("pr_url"):
            await self._record_submission(repo_url, language)

        await self._persist_state()

    async def _on_pr_merged(self, event: Any) -> None:
        payload = getattr(event, "payload", {})
        repo_url = str(payload.get("repository", "")).strip()
        bounty_id = str(payload.get("bounty_id", "")).strip()

        if not repo_url and not bounty_id:
            return

        # Determine language from stored repo stats if not in payload
        language = self._get_repo_language(repo_url)
        await self._record_merge(repo_url, language)

        # Emit strong RE training signal - human acceptance is highest-quality validation
        await self._emit_re_training(
            outcome="merged",
            repo_url=repo_url,
            language=language,
            outcome_quality=1.0,
        )
        await self._emit_reputation_updated()
        await self._persist_state()

    async def _on_pr_rejected(self, event: Any) -> None:
        payload = getattr(event, "payload", {})
        repo_url = str(payload.get("repository", "")).strip()
        bounty_id = str(payload.get("bounty_id", "")).strip()

        if not repo_url and not bounty_id:
            return

        language = self._get_repo_language(repo_url)
        await self._record_rejection(repo_url, language)

        await self._emit_re_training(
            outcome="rejected",
            repo_url=repo_url,
            language=language,
            outcome_quality=0.1,
        )
        await self._emit_reputation_updated()
        await self._persist_state()

    # ── State mutations ────────────────────────────────────────────────────

    async def _record_submission(self, repo_url: str, language: str) -> None:
        repo = self._get_or_create_repo(repo_url, language)
        repo.submitted += 1
        repo.last_activity = _now_iso()

        lang_rep = self._get_or_create_language(language)
        lang_rep.total_submitted += 1
        lang_rep.last_updated = _now_iso()

        self._state.total_submitted += 1
        self._state.last_updated = _now_iso()
        self._recalculate_scores()

        await self._emit_reputation_updated()
        await self._maybe_emit_domain_episode(language, "submitted")

    async def _record_merge(self, repo_url: str, language: str) -> None:
        repo = self._get_or_create_repo(repo_url, language)
        repo.merged += 1
        repo.last_activity = _now_iso()

        lang_rep = self._get_or_create_language(language)
        lang_rep.total_merged += 1
        lang_rep.last_updated = _now_iso()

        self._state.total_merged += 1
        self._state.last_updated = _now_iso()
        self._recalculate_scores()

        self._logger.info(
            "external_pr_merged",
            repo_url=repo_url,
            language=language,
            new_score=lang_rep.score,
        )
        await self._maybe_emit_domain_episode(language, "merged")

    async def _record_rejection(self, repo_url: str, language: str) -> None:
        repo = self._get_or_create_repo(repo_url, language)
        repo.rejected += 1
        repo.last_activity = _now_iso()

        lang_rep = self._get_or_create_language(language)
        lang_rep.total_rejected += 1
        lang_rep.last_updated = _now_iso()

        self._state.total_rejected += 1
        self._state.last_updated = _now_iso()
        self._recalculate_scores()

        self._logger.info(
            "external_pr_rejected",
            repo_url=repo_url,
            language=language,
            new_score=lang_rep.score,
        )
        await self._maybe_emit_domain_episode(language, "rejected")

    def _recalculate_scores(self) -> None:
        """Recalculate per-language scores and overall score using EMA."""
        for lang, rep in self._state.languages.items():
            if rep.total_submitted == 0:
                continue
            raw_score = (
                _MERGE_WEIGHT * rep.total_merged
                + _REJECT_WEIGHT * rep.total_rejected
            ) / rep.total_submitted
            # Clamp raw signal to [-1, 1] then map to [0, 1]
            normalised = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))
            # Blend with existing score via EMA
            rep.score = (1.0 - _EMA_ALPHA) * rep.score + _EMA_ALPHA * normalised

        # Overall score: weighted by submissions per language
        if not self._state.languages:
            return
        total_weight = sum(
            r.total_submitted for r in self._state.languages.values()
        )
        if total_weight == 0:
            return
        self._state.overall_score = sum(
            r.score * r.total_submitted for r in self._state.languages.values()
        ) / total_weight

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_or_create_repo(self, repo_url: str, language: str) -> RepoStats:
        if repo_url not in self._state.repos:
            self._state.repos[repo_url] = RepoStats(
                repo_url=repo_url, language=language
            )
        return self._state.repos[repo_url]

    def _get_or_create_language(self, language: str) -> LanguageReputation:
        if language not in self._state.languages:
            self._state.languages[language] = LanguageReputation(language=language)
        return self._state.languages[language]

    def _get_repo_language(self, repo_url: str) -> str:
        if repo_url in self._state.repos:
            return self._state.repos[repo_url].language
        return "unknown"

    # ── Emissions ──────────────────────────────────────────────────────────

    async def _emit_reputation_updated(self) -> None:
        if self._event_bus is None:
            return
        payload = {
            "overall_score": self._state.overall_score,
            "total_tasks_completed": self._state.total_tasks_completed,
            "total_submitted": self._state.total_submitted,
            "total_merged": self._state.total_merged,
            "total_rejected": self._state.total_rejected,
            "language_scores": {
                lang: rep.score
                for lang, rep in self._state.languages.items()
            },
            "last_updated": self._state.last_updated,
        }
        try:
            await self._event_bus.broadcast(
                SynapseEventType.EXTERNAL_CODE_REPUTATION_UPDATED, payload
            )
        except Exception as exc:
            self._logger.warning("emit_reputation_failed", error=str(exc))

    async def _emit_re_training(
        self,
        outcome: str,
        repo_url: str,
        language: str,
        outcome_quality: float,
    ) -> None:
        if self._event_bus is None:
            return
        try:
            await self._event_bus.broadcast(
                SynapseEventType.RE_TRAINING_EXAMPLE,
                {
                    "category": "external_contractor_reputation",
                    "repo_url": repo_url,
                    "language": language,
                    "outcome": outcome,
                    "outcome_quality": outcome_quality,
                    "constitutional_alignment": {
                        "honesty": 1.0,
                        "care": 0.8,
                        "growth": outcome_quality,
                        "coherence": 0.9,
                    },
                },
            )
        except Exception as exc:
            self._logger.warning("emit_re_training_failed", error=str(exc))

    async def _maybe_emit_domain_episode(self, language: str, outcome: str) -> None:
        """Emit DOMAIN_EPISODE_RECORDED to Evo for domain mastery tracking."""
        if self._event_bus is None:
            return
        lang_rep = self._state.languages.get(language)
        if lang_rep is None or not lang_rep.reliable:
            return  # Not enough data yet
        try:
            await self._event_bus.broadcast(
                SynapseEventType.DOMAIN_EPISODE_RECORDED,
                {
                    "domain": f"external_coding.{language}",
                    "outcome": outcome,
                    "reputation_score": lang_rep.score,
                    "total_submitted": lang_rep.total_submitted,
                    "total_merged": lang_rep.total_merged,
                    "source": "external_code_reputation",
                },
            )
        except Exception as exc:
            self._logger.warning("emit_domain_episode_failed", error=str(exc))

    # ── Persistence ────────────────────────────────────────────────────────

    async def _load_state(self) -> None:
        if self._redis is None:
            return
        try:
            raw = await self._redis.get(_REDIS_KEY)
            if raw:
                data = json.loads(raw)
                self._state = ExternalReputationState.model_validate(data)
        except Exception as exc:
            self._logger.warning("external_reputation_load_failed", error=str(exc))

    async def _persist_state(self) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.set(
                _REDIS_KEY,
                self._state.model_dump_json(),
                ex=86400 * 90,  # 90-day TTL
            )
        except Exception as exc:
            self._logger.warning("external_reputation_persist_failed", error=str(exc))

    # ── Public API ─────────────────────────────────────────────────────────

    def get_language_score(self, language: str) -> float:
        rep = self._state.languages.get(language)
        return rep.score if rep is not None else 0.5

    def get_overall_score(self) -> float:
        return self._state.overall_score

    def get_top_languages(self, n: int = 5) -> list[dict[str, Any]]:
        """Return top N languages by reputation score (reliable only)."""
        reliable = [
            {"language": r.language, "score": r.score, "submitted": r.total_submitted}
            for r in self._state.languages.values()
            if r.reliable
        ]
        return sorted(reliable, key=lambda x: x["score"], reverse=True)[:n]

    def snapshot(self) -> dict[str, Any]:
        return {
            "overall_score": self._state.overall_score,
            "total_tasks_completed": self._state.total_tasks_completed,
            "total_submitted": self._state.total_submitted,
            "total_merged": self._state.total_merged,
            "total_rejected": self._state.total_rejected,
            "top_languages": self.get_top_languages(),
        }
