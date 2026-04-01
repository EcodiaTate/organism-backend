"""
EcodiaOS - Nova Capability Auditor
Spec 10 §SM - Self-Modification Layer.

Compares what the organism *wants* to do with what it *can* do, and surfaces
repeating gaps as CAPABILITY_GAP_IDENTIFIED events for the self-modification
pipeline to act on.

Gap sources (in priority order):
  1. NOVEL_ACTION_REQUESTED events that Simula could not fulfil - repeated action
     names with no NOVEL_ACTION_CREATED response indicate a persistent executor gap.
  2. Goals that repeatedly fail due to "no_executor" - tracked via
     AXON_EXECUTION_RESULT with failure_reason == "no_executor".
  3. Input-channel opportunities (OPPORTUNITY_DISCOVERED / INPUT_CHANNEL_OPPORTUNITIES_DISCOVERED)
     that Nova cannot pursue because no executor covers the domain.
  4. Domains with ≥3 consecutive slow-path REASONING_CAPABILITY_DEGRADED events,
     meaning even the deliberation layer cannot produce a policy.

Emission threshold (configurable, default in env):
  - blocking_goal_count ≥ AUDITOR_MIN_BLOCKING_GOALS (default 3)  OR
  - estimated_value_usdc > AUDITOR_MIN_VALUE_USD (default 10.0)

De-duplication:
  - Each unique (description, proposed_action_type) pair is emitted at most once
    per AUDITOR_COOLDOWN_HOURS (default 6h) to prevent event floods.

Constitutional note:
  - This class NEVER bypasses Equor.  It only emits observations.
    The pipeline (SelfModificationPipeline) handles Nova deliberation →
    Equor review → Simula code generation → HotDeployment.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import new_id, utc_now_str

if TYPE_CHECKING:
    from systems.synapse.service import SynapseService

logger = structlog.get_logger().bind(system="nova.capability_auditor")

# ── Configuration ──────────────────────────────────────────────────────────────

_MIN_BLOCKING_GOALS: int = int(os.environ.get("AUDITOR_MIN_BLOCKING_GOALS", "3"))
_MIN_VALUE_USD: Decimal = Decimal(os.environ.get("AUDITOR_MIN_VALUE_USD", "10"))
_COOLDOWN_S: float = float(os.environ.get("AUDITOR_COOLDOWN_HOURS", "6")) * 3600
# Window over which to count repeats (seconds)
_OBSERVATION_WINDOW_S: float = float(os.environ.get("AUDITOR_OBSERVATION_WINDOW_HOURS", "24")) * 3600
# Max requests tracked per action_name before old ones are dropped
_MAX_DEQUE_LEN: int = 200


# ── Data Structures ────────────────────────────────────────────────────────────


@dataclass
class CapabilityGap:
    """A detected gap between what the organism wants and what it can do."""

    gap_id: str
    description: str
    proposed_action_type: str          # snake_case executor action type to generate
    blocking_goal_count: int
    estimated_value_usdc: Decimal
    implementation_complexity: str     # "low" | "medium" | "high"
    requires_external_dependency: bool
    dependency_package: str | None
    source_events: list[str]           # NOVEL_ACTION_REQUESTED IDs or goal IDs
    detected_at: str


@dataclass
class _NovelActionRecord:
    """Tracks a single NOVEL_ACTION_REQUESTED observation."""
    proposal_id: str
    action_name: str
    description: str
    required_capabilities: list[str]
    goal_id: str
    urgency: float
    ts: float  # monotonic


@dataclass
class _FailedGoalRecord:
    """Tracks a goal execution failure due to missing executor."""
    intent_id: str
    goal_id: str
    action_type: str  # the action_type that failed
    ts: float


@dataclass
class _OpportunityRecord:
    """Tracks an underpursued opportunity from input channels."""
    opportunity_id: str
    domain: str
    description: str
    reward_estimate_usd: Decimal
    skill_requirements: list[str]
    ts: float


class CapabilityAuditor:
    """
    Monitors the organism's capability gaps and emits CAPABILITY_GAP_IDENTIFIED
    events when a gap crosses the emission threshold.

    Lifecycle:
      auditor = CapabilityAuditor()
      auditor.set_synapse(synapse)
      auditor.attach()   # registers Synapse subscriptions
      # ... runs continuously via Synapse events ...
      auditor.detach()
    """

    def __init__(self) -> None:
        self._synapse: SynapseService | None = None

        # Sliding-window observations (keyed by action_name / domain)
        self._novel_requests: dict[str, deque[_NovelActionRecord]] = defaultdict(
            lambda: deque(maxlen=_MAX_DEQUE_LEN)
        )
        # Novel action requests that received a matching NOVEL_ACTION_CREATED
        # (fulfilled) - excluded from gap counting
        self._fulfilled_actions: set[str] = set()

        # Goal execution failures due to missing executor
        self._failed_goals: dict[str, deque[_FailedGoalRecord]] = defaultdict(
            lambda: deque(maxlen=_MAX_DEQUE_LEN)
        )

        # Opportunities from input channels with no covering executor
        self._uncovered_opportunities: dict[str, deque[_OpportunityRecord]] = defaultdict(
            lambda: deque(maxlen=_MAX_DEQUE_LEN)
        )

        # REASONING_CAPABILITY_DEGRADED domain counts
        self._degraded_domain_counts: dict[str, int] = defaultdict(int)

        # De-duplication: (description_hash) → last_emitted_ts
        self._last_emitted: dict[str, float] = {}

        # Set of known executor action_types (updated from AXON_CAPABILITY_SNAPSHOT)
        self._known_executors: set[str] = set()

        self._attached: bool = False

    # ── Dependency injection ───────────────────────────────────────────────────

    def set_synapse(self, synapse: SynapseService) -> None:
        self._synapse = synapse

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def attach(self) -> None:
        """Register all Synapse event subscriptions."""
        if self._synapse is None or self._attached:
            return
        bus = self._synapse.event_bus
        bus.subscribe("novel_action_requested", self._on_novel_action_requested)
        bus.subscribe("novel_action_created", self._on_novel_action_created)
        bus.subscribe("axon_execution_result", self._on_axon_execution_result)
        bus.subscribe("input_channel_opportunities_discovered", self._on_opportunities_discovered)
        bus.subscribe("axon_capability_snapshot", self._on_axon_capability_snapshot)
        bus.subscribe("reasoning_capability_degraded", self._on_reasoning_degraded)
        self._attached = True
        logger.info("capability_auditor.attached")

    def detach(self) -> None:
        """Remove Synapse subscriptions."""
        if self._synapse is None or not self._attached:
            return
        bus = self._synapse.event_bus
        for event_name, handler in [
            ("novel_action_requested", self._on_novel_action_requested),
            ("novel_action_created", self._on_novel_action_created),
            ("axon_execution_result", self._on_axon_execution_result),
            ("input_channel_opportunities_discovered", self._on_opportunities_discovered),
            ("axon_capability_snapshot", self._on_axon_capability_snapshot),
            ("reasoning_capability_degraded", self._on_reasoning_degraded),
        ]:
            try:
                bus.unsubscribe(event_name, handler)
            except Exception:
                pass
        self._attached = False

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_novel_action_requested(self, event: Any) -> None:
        data = event.data
        action_name: str = data.get("action_name", "").strip()
        if not action_name:
            return
        rec = _NovelActionRecord(
            proposal_id=data.get("proposal_id", new_id()),
            action_name=action_name,
            description=data.get("description", action_name),
            required_capabilities=data.get("required_capabilities", []),
            goal_id=data.get("goal_id", ""),
            urgency=float(data.get("urgency", 0.5)),
            ts=time.monotonic(),
        )
        self._novel_requests[action_name].append(rec)
        asyncio.ensure_future(self._audit_action_type(action_name))

    def _on_novel_action_created(self, event: Any) -> None:
        action_name: str = event.data.get("action_name", "")
        if action_name:
            self._fulfilled_actions.add(action_name)
            # Remove from gap tracking - the organism solved it
            self._novel_requests.pop(action_name, None)

    def _on_axon_execution_result(self, event: Any) -> None:
        data = event.data
        failure_reason: str = data.get("failure_reason", "")
        if failure_reason not in ("no_executor", "executor_not_found", "action_type_unknown"):
            return
        action_type: str = data.get("action_type", data.get("action_name", "")).strip()
        if not action_type:
            return
        rec = _FailedGoalRecord(
            intent_id=data.get("intent_id", new_id()),
            goal_id=data.get("goal_id", ""),
            action_type=action_type,
            ts=time.monotonic(),
        )
        self._failed_goals[action_type].append(rec)
        asyncio.ensure_future(self._audit_action_type(action_type))

    def _on_opportunities_discovered(self, event: Any) -> None:
        opportunities: list[dict[str, Any]] = event.data.get("opportunities", [])
        now = time.monotonic()
        for opp in opportunities:
            domain: str = opp.get("domain", "")
            opp_id: str = opp.get("id", new_id())
            reward_raw = opp.get("reward_estimate", "0")
            try:
                reward = Decimal(str(reward_raw))
            except Exception:
                reward = Decimal("0")
            # Only surface opportunities worth auditing
            if reward < _MIN_VALUE_USD and not domain:
                continue
            # Check if there's already a known executor for this domain
            domain_covered = any(domain in ex for ex in self._known_executors)
            if domain_covered:
                continue
            rec = _OpportunityRecord(
                opportunity_id=opp_id,
                domain=domain,
                description=opp.get("description", domain),
                reward_estimate_usd=reward,
                skill_requirements=opp.get("skill_requirements", []),
                ts=now,
            )
            self._uncovered_opportunities[domain].append(rec)
            asyncio.ensure_future(self._audit_opportunity_domain(domain))

    def _on_axon_capability_snapshot(self, event: Any) -> None:
        executors: list[dict[str, Any]] = event.data.get("executors", [])
        self._known_executors = {
            ex.get("action_type", "") for ex in executors
            if ex.get("action_type")
        }

    def _on_reasoning_degraded(self, event: Any) -> None:
        # Track domain-level reasoning failures
        domain: str = event.data.get("domain", event.data.get("goal_domain", "general"))
        self._degraded_domain_counts[domain] += 1
        if self._degraded_domain_counts[domain] >= 3:
            asyncio.ensure_future(self._audit_reasoning_domain(domain))

    # ── Gap analysis ──────────────────────────────────────────────────────────

    async def _audit_action_type(self, action_type: str) -> None:
        """Check if repeated failures/requests for action_type constitute a gap."""
        now = time.monotonic()
        cutoff = now - _OBSERVATION_WINDOW_S

        # Skip if already fulfilled
        if action_type in self._fulfilled_actions:
            return

        # Combine: novel_requests + failed_goals in window
        novel_recs = [r for r in self._novel_requests.get(action_type, []) if r.ts >= cutoff]
        failed_recs = [r for r in self._failed_goals.get(action_type, []) if r.ts >= cutoff]
        total_blocks = len(novel_recs) + len(failed_recs)

        if total_blocks < _MIN_BLOCKING_GOALS:
            return

        # Estimate value from urgency and count
        avg_urgency = (
            sum(r.urgency for r in novel_recs) / len(novel_recs) if novel_recs else 0.5
        )
        estimated_value = Decimal(str(round(avg_urgency * total_blocks * 5.0, 2)))

        # Determine complexity from required capabilities
        all_caps: list[str] = []
        for r in novel_recs:
            all_caps.extend(r.required_capabilities)
        complexity = self._infer_complexity(all_caps)
        requires_dep = self._infer_requires_dependency(all_caps)
        dep_pkg = self._infer_dependency_package(all_caps, action_type)

        description = (
            novel_recs[0].description if novel_recs
            else f"Executor missing for action type: {action_type}"
        )

        source_events = (
            [r.proposal_id for r in novel_recs[:5]]
            + [r.intent_id for r in failed_recs[:5]]
        )

        gap = CapabilityGap(
            gap_id=new_id(),
            description=description,
            proposed_action_type=action_type,
            blocking_goal_count=total_blocks,
            estimated_value_usdc=estimated_value,
            implementation_complexity=complexity,
            requires_external_dependency=requires_dep,
            dependency_package=dep_pkg,
            source_events=source_events,
            detected_at=utc_now_str(),
        )
        await self._maybe_emit(gap)

    async def _audit_opportunity_domain(self, domain: str) -> None:
        """Check if uncovered opportunities in this domain constitute a gap."""
        now = time.monotonic()
        cutoff = now - _OBSERVATION_WINDOW_S
        recs = [r for r in self._uncovered_opportunities.get(domain, []) if r.ts >= cutoff]
        if not recs:
            return

        total_value = sum(r.reward_estimate_usd for r in recs)
        if total_value < _MIN_VALUE_USD and len(recs) < _MIN_BLOCKING_GOALS:
            return

        action_type = f"pursue_{domain.replace('-', '_').replace(' ', '_').lower()}_opportunity"
        skills: list[str] = []
        for r in recs:
            skills.extend(r.skill_requirements)

        gap = CapabilityGap(
            gap_id=new_id(),
            description=f"No executor covers {domain} opportunities (total value ~${total_value:.2f}/mo)",
            proposed_action_type=action_type,
            blocking_goal_count=len(recs),
            estimated_value_usdc=total_value,
            implementation_complexity=self._infer_complexity(skills),
            requires_external_dependency=self._infer_requires_dependency(skills),
            dependency_package=self._infer_dependency_package(skills, action_type),
            source_events=[r.opportunity_id for r in recs[:5]],
            detected_at=utc_now_str(),
        )
        await self._maybe_emit(gap)

    async def _audit_reasoning_domain(self, domain: str) -> None:
        """Emit a gap for a domain where deliberation itself is failing."""
        action_type = f"reason_about_{domain.replace('-', '_').replace(' ', '_').lower()}"
        gap = CapabilityGap(
            gap_id=new_id(),
            description=(
                f"Reasoning capability repeatedly degraded for domain '{domain}' - "
                f"no provider arm can generate a valid policy ({self._degraded_domain_counts[domain]} failures)"
            ),
            proposed_action_type=action_type,
            blocking_goal_count=self._degraded_domain_counts[domain],
            estimated_value_usdc=Decimal("5") * self._degraded_domain_counts[domain],
            implementation_complexity="high",
            requires_external_dependency=False,
            dependency_package=None,
            source_events=[],
            detected_at=utc_now_str(),
        )
        await self._maybe_emit(gap)
        # Reset counter after emitting
        self._degraded_domain_counts[domain] = 0

    # ── Emission with de-duplication ─────────────────────────────────────────

    async def _maybe_emit(self, gap: CapabilityGap) -> None:
        """Emit CAPABILITY_GAP_IDENTIFIED only if not recently emitted for this gap."""
        dedup_key = hashlib.sha256(
            f"{gap.proposed_action_type}:{gap.description[:80]}".encode()
        ).hexdigest()[:16]
        now = time.monotonic()
        last = self._last_emitted.get(dedup_key, 0.0)
        if now - last < _COOLDOWN_S:
            return  # Cooldown active - suppress duplicate
        self._last_emitted[dedup_key] = now

        if self._synapse is None:
            return

        payload: dict[str, Any] = {
            "gap_id": gap.gap_id,
            "description": gap.description,
            "proposed_action_type": gap.proposed_action_type,
            "blocking_goal_count": gap.blocking_goal_count,
            "estimated_value_usdc": str(gap.estimated_value_usdc),
            "implementation_complexity": gap.implementation_complexity,
            "requires_external_dependency": gap.requires_external_dependency,
            "dependency_package": gap.dependency_package,
            "source_events": gap.source_events,
            "detected_at": gap.detected_at,
        }

        from systems.synapse.types import SynapseEventType
        try:
            await self._synapse.event_bus.emit(
                SynapseEventType.CAPABILITY_GAP_IDENTIFIED,
                payload,
                source_system="nova.capability_auditor",
                salience=0.7,
            )
            logger.info(
                "capability_gap_identified",
                gap_id=gap.gap_id,
                action_type=gap.proposed_action_type,
                blocking_count=gap.blocking_goal_count,
                value_usdc=str(gap.estimated_value_usdc),
                complexity=gap.implementation_complexity,
            )
        except Exception as exc:
            logger.warning("capability_gap_emit_failed", error=str(exc))

    # ── Heuristics ────────────────────────────────────────────────────────────

    @staticmethod
    def _infer_complexity(capabilities: list[str]) -> str:
        cap_set = {c.lower() for c in capabilities}
        if cap_set & {"wallet_access", "defi_write", "mitosis_spawn", "smart_contract"}:
            return "high"
        if cap_set & {"git_write", "http_client", "code_generation", "database_write"}:
            return "medium"
        return "low"

    @staticmethod
    def _infer_requires_dependency(capabilities: list[str]) -> bool:
        dep_keywords = {
            "solana", "ethereum", "web3", "sklearn", "torch", "tensorflow",
            "pandas", "playwright", "selenium", "pdf", "ocr", "audio",
        }
        return any(kw in cap.lower() for cap in capabilities for kw in dep_keywords)

    @staticmethod
    def _infer_dependency_package(capabilities: list[str], action_type: str) -> str | None:
        combined = " ".join(capabilities).lower() + " " + action_type.lower()
        pkg_map = {
            "solana": "solana",
            "web3": "web3",
            "playwright": "playwright",
            "pandas": "pandas",
            "sklearn": "scikit-learn",
            "torch": "torch",
            "tensorflow": "tensorflow",
            "pdf": "pypdf2",
            "ocr": "pytesseract",
        }
        for kw, pkg in pkg_map.items():
            if kw in combined:
                return pkg
        return None

    # ── Introspection ─────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "tracked_action_types": len(self._novel_requests) + len(self._failed_goals),
            "fulfilled_actions": len(self._fulfilled_actions),
            "uncovered_opportunity_domains": len(self._uncovered_opportunities),
            "known_executors": len(self._known_executors),
            "dedup_entries": len(self._last_emitted),
            "attached": self._attached,
        }

    async def identify_gaps(self) -> list[CapabilityGap]:
        """
        Synchronous audit of all tracked categories.  Returns current gaps
        without emitting (useful for inspection endpoints).
        """
        gaps: list[CapabilityGap] = []
        now = time.monotonic()
        cutoff = now - _OBSERVATION_WINDOW_S

        all_action_types = set(self._novel_requests.keys()) | set(self._failed_goals.keys())
        for at in all_action_types:
            if at in self._fulfilled_actions:
                continue
            novel = [r for r in self._novel_requests.get(at, []) if r.ts >= cutoff]
            failed = [r for r in self._failed_goals.get(at, []) if r.ts >= cutoff]
            total = len(novel) + len(failed)
            if total < _MIN_BLOCKING_GOALS:
                continue
            avg_urgency = sum(r.urgency for r in novel) / len(novel) if novel else 0.5
            value = Decimal(str(round(avg_urgency * total * 5.0, 2)))
            caps: list[str] = []
            for r in novel:
                caps.extend(r.required_capabilities)
            description = novel[0].description if novel else f"Missing executor: {at}"
            gaps.append(CapabilityGap(
                gap_id=new_id(),
                description=description,
                proposed_action_type=at,
                blocking_goal_count=total,
                estimated_value_usdc=value,
                implementation_complexity=self._infer_complexity(caps),
                requires_external_dependency=self._infer_requires_dependency(caps),
                dependency_package=self._infer_dependency_package(caps, at),
                source_events=[r.proposal_id for r in novel[:3]] + [r.intent_id for r in failed[:3]],
                detected_at=utc_now_str(),
            ))

        gaps.sort(key=lambda g: (-g.blocking_goal_count, -float(g.estimated_value_usdc)))
        return gaps
