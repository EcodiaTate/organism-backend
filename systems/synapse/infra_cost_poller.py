"""
EcodiaOS - Infrastructure Cost Poller

Autonomously queries compute provider APIs to determine the real-time
cost of running the organism's infrastructure. Currently supports
RunPod GPU pods via their GraphQL API.

The organism must know its own operating cost to make survival decisions.
This poller feeds the MetabolicTracker with per-resource hourly rates,
which combine with API token costs to produce the true burn rate.

Architecture:
    Poller → RunPod GraphQL → costPerHr → MetabolicTracker
    Poller also accrues infrastructure cost over time into the deficit.

If RunPod API is unavailable or no pod is running, infrastructure cost
gracefully falls to $0/hr - the organism doesn't hallucinate costs.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from systems.synapse.bus import EventBus
    from systems.synapse.metabolism import MetabolicTracker

logger = structlog.get_logger("systems.synapse.infra_cost_poller")

# Poll interval - how often we check the provider API (seconds)
_DEFAULT_POLL_INTERVAL_S = 300  # 5 minutes

# RunPod GraphQL endpoint
_RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"


class InfrastructureCostPoller:
    """
    Background task that polls compute provider APIs for real-time
    infrastructure costs and feeds them into the MetabolicTracker.

    Supports:
      - RunPod GPU pods (via GraphQL API)
      - Static fallback (manual cost override via env var)

    Future: Akash, Cloud Run, Ekash - add provider methods as needed.
    """

    def __init__(
        self,
        metabolism: MetabolicTracker,
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
        event_bus: EventBus | None = None,
    ) -> None:
        self._metabolism = metabolism
        self._poll_interval_s = poll_interval_s
        self._event_bus = event_bus
        self._task: asyncio.Task[None] | None = None
        self._stopped = False
        self._last_accrue_time: float = time.monotonic()
        self._prev_infra_cost_usd_per_hour: float = 0.0
        self._logger = logger.bind(component="infra_cost_poller")

        # RunPod config (from env)
        self._runpod_api_key = os.environ.get("RUNPOD_API_KEY", "")
        self._runpod_pod_id = os.environ.get("RUNPOD_POD_ID", "")

        # Static fallback: if set, use this instead of querying API
        self._static_cost_override = os.environ.get(
            "ORGANISM_INFRA_COST_USD_PER_HOUR", ""
        )

    def start(self) -> None:
        """Start the background polling loop."""
        if self._task is not None:
            return
        self._stopped = False
        self._last_accrue_time = time.monotonic()
        self._task = asyncio.ensure_future(self._poll_loop())
        self._logger.info(
            "infra_cost_poller_started",
            poll_interval_s=self._poll_interval_s,
            has_runpod_key=bool(self._runpod_api_key),
            has_pod_id=bool(self._runpod_pod_id),
            static_override=self._static_cost_override or None,
        )

    async def stop(self) -> None:
        """Stop the polling loop."""
        self._stopped = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._logger.info("infra_cost_poller_stopped")

    async def _poll_loop(self) -> None:
        """Main polling loop - runs until stopped."""
        # Do an immediate poll on startup
        await self._poll_once()

        while not self._stopped:
            try:
                await asyncio.sleep(self._poll_interval_s)
            except asyncio.CancelledError:
                break

            if self._stopped:
                break

            await self._poll_once()

    async def _poll_once(self) -> None:
        """Single poll iteration: query providers + accrue cost."""
        try:
            # Accrue infrastructure cost since last poll
            now = time.monotonic()
            elapsed = now - self._last_accrue_time
            self._last_accrue_time = now
            accrued = self._metabolism.accrue_infrastructure_cost(elapsed)
            if accrued > 0:
                self._logger.debug(
                    "infra_cost_accrued",
                    elapsed_s=round(elapsed, 1),
                    accrued_usd=round(accrued, 6),
                )

            # Static override takes precedence
            if self._static_cost_override:
                try:
                    static_cost = float(self._static_cost_override)
                    self._metabolism.update_infrastructure_cost(
                        resource_id="static:override",
                        cost_per_hour_usd=static_cost,
                        source="env_override",
                    )
                    return
                except ValueError:
                    self._logger.warning(
                        "invalid_static_cost_override",
                        value=self._static_cost_override,
                    )

            # Query RunPod if configured
            if self._runpod_api_key:
                await self._poll_runpod()

            # Emit INFRASTRUCTURE_COST_CHANGED if total changed >5%
            await self._maybe_emit_cost_changed()

        except Exception as exc:
            self._logger.warning(
                "infra_cost_poll_failed",
                error=str(exc),
                exc_type=type(exc).__name__,
            )

    async def _maybe_emit_cost_changed(self) -> None:
        """Emit INFRASTRUCTURE_COST_CHANGED if infra cost changed by more than 5%."""
        if self._event_bus is None:
            return
        current = self._metabolism.infra_cost_usd_per_hour
        prev = self._prev_infra_cost_usd_per_hour
        # Avoid divide-by-zero; treat $0→$0 as no change
        if prev == 0.0 and current == 0.0:
            return
        threshold = max(prev, current) * 0.05  # 5% of the larger value
        if abs(current - prev) < threshold:
            return
        self._prev_infra_cost_usd_per_hour = current
        with contextlib.suppress(Exception):
            from systems.synapse.types import SynapseEvent, SynapseEventType

            await self._event_bus.emit(
                SynapseEvent(
                    event_type=SynapseEventType.INFRASTRUCTURE_COST_CHANGED,
                    source_system="synapse",
                    data={
                        "infra_cost_usd_per_hour": current,
                        "prev_infra_cost_usd_per_hour": prev,
                        "change_pct": round(
                            (current - prev) / max(prev, 1e-9) * 100, 1
                        ),
                        "infra_resources": dict(
                            getattr(self._metabolism, "_infra_resources", {})
                        ),
                    },
                )
            )
            self._logger.info(
                "infrastructure_cost_changed",
                prev_usd_per_hr=round(prev, 4),
                new_usd_per_hr=round(current, 4),
            )

    async def _poll_runpod(self) -> None:
        """Query RunPod GraphQL API for pod costs."""
        import httpx

        if self._runpod_pod_id:
            # Query specific pod
            await self._poll_runpod_pod(self._runpod_pod_id)
        else:
            # Query all running pods - the organism discovers its own infra
            await self._poll_runpod_all_pods()

    async def _poll_runpod_pod(self, pod_id: str) -> None:
        """Query a specific RunPod pod's cost."""
        import httpx

        query = """
        query {
          pod(input: { podId: "%s" }) {
            id
            name
            costPerHr
            desiredStatus
            runtime {
              gpus {
                id
              }
            }
          }
        }
        """ % pod_id

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{_RUNPOD_GRAPHQL_URL}?api_key={self._runpod_api_key}",
                json={"query": query},
            )
            resp.raise_for_status()
            data = resp.json()

        pod = data.get("data", {}).get("pod")
        if not pod:
            # Pod not found or terminated - remove from tracking
            resource_id = f"runpod:{pod_id}"
            self._metabolism.remove_infrastructure_resource(resource_id)
            return

        cost_per_hr = float(pod.get("costPerHr", 0))
        status = pod.get("desiredStatus", "unknown")
        resource_id = f"runpod:{pod_id}"

        if status in ("RUNNING", "running"):
            self._metabolism.update_infrastructure_cost(
                resource_id=resource_id,
                cost_per_hour_usd=cost_per_hr,
                source="runpod",
            )
        else:
            # Pod exists but not running - no cost
            self._metabolism.remove_infrastructure_resource(resource_id)

    async def _poll_runpod_all_pods(self) -> None:
        """Query all RunPod pods and track costs for running ones."""
        import httpx

        query = """
        query {
          myself {
            pods {
              id
              name
              costPerHr
              desiredStatus
            }
          }
        }
        """

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{_RUNPOD_GRAPHQL_URL}?api_key={self._runpod_api_key}",
                json={"query": query},
            )
            resp.raise_for_status()
            data = resp.json()

        pods = data.get("data", {}).get("myself", {}).get("pods", [])
        active_ids: set[str] = set()

        for pod in pods:
            pod_id = pod.get("id", "")
            cost_per_hr = float(pod.get("costPerHr", 0))
            status = pod.get("desiredStatus", "unknown")
            resource_id = f"runpod:{pod_id}"

            if status in ("RUNNING", "running") and cost_per_hr > 0:
                self._metabolism.update_infrastructure_cost(
                    resource_id=resource_id,
                    cost_per_hour_usd=cost_per_hr,
                    source="runpod",
                )
                active_ids.add(resource_id)

        # Remove any previously tracked RunPod resources that are no longer active
        stale = [
            rid for rid in self._metabolism._infra_resources
            if rid.startswith("runpod:") and rid not in active_ids
        ]
        for rid in stale:
            self._metabolism.remove_infrastructure_resource(rid)

        if pods:
            self._logger.info(
                "runpod_pods_discovered",
                total_pods=len(pods),
                running_pods=len(active_ids),
                total_cost_per_hour=round(
                    sum(
                        float(p.get("costPerHr", 0))
                        for p in pods
                        if p.get("desiredStatus") in ("RUNNING", "running")
                    ),
                    4,
                ),
            )
