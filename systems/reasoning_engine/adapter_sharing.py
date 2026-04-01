"""Cross-instance adapter merging - Share (2025) framework.

When two EOS instances are reproductively compatible (genome distance < threshold),
they can merge their LoRA adapter knowledge. This implements genetic recombination
at the knowledge representation level.

Merging algorithm:
1. Check reproductive compatibility (genome distance < threshold)
2. Fetch both instances' slow adapter paths (via ADAPTER_SHARE_REQUEST event)
3. Weight each adapter by instance fitness (RE success rate * economic performance)
4. Merge: merged[k] = weight_a * adapter_a[k] + weight_b * adapter_b[k]
5. STABLE KL gate check on merged adapter before offering
6. Offer merged adapter to both instances via ADAPTER_SHARE_OFFER event
7. Each instance applies or rejects independently

This is analogous to sexual recombination in biological evolution.
Reproductively isolated instances (different species) cannot merge.

Pending adapter priority: shared > DPO > None
  - _pending_shared_adapter stored in CLO when ADAPTER_SHARE_OFFER received
  - Consumed at start of next _execute_tier2() as BASE_ADAPTER, before _pending_dpo_adapter
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from systems.mitosis.genome_distance import GenomeDistanceCalculator
    from systems.reasoning_engine.anti_forgetting import STABLEKLGate
    from systems.reasoning_engine.service import ReasoningEngineService

logger = structlog.get_logger("reasoning_engine.adapter_sharing")


@dataclass
class AdapterShareRequest:
    """Request to share adapters between two instances."""

    requester_id: str
    partner_id: str
    requester_fitness: float  # RE success rate * economic performance
    requester_adapter_path: str
    genome_a: dict[str, Any]
    genome_b: dict[str, Any]
    request_id: str = field(default_factory=lambda: f"share_{int(time.time())}")


@dataclass
class AdapterShareResult:
    """Result of an adapter sharing attempt."""

    request_id: str
    success: bool
    merged_adapter_path: str | None = None
    kl_divergence: float = 0.0
    rejection_reason: str = ""
    genome_distance: float = 0.0
    weight_a: float = 0.5
    weight_b: float = 0.5


class AdapterSharer:
    """Orchestrates cross-instance adapter merging.

    Called by MitosisFleetService when two instances are identified as
    reproductively compatible and both have trained adapters.

    Two scenarios where attempt_merge() is called:
    1. MitosisFleetService detects two compatible instances in the fleet
       (fleet ≥ 2 instances, genome distance < threshold)
    2. Manual trigger for research/testing

    Degrades gracefully when fleet has < 2 instances - attempt_merge() is
    simply never called in that case.
    """

    def __init__(
        self,
        genome_calculator: "GenomeDistanceCalculator",
        kl_gate: "STABLEKLGate",
        re_service: "ReasoningEngineService",
        event_bus: Any,
        output_dir: str = "data/re_adapters/shared",
    ) -> None:
        self._genome_calc = genome_calculator
        self._kl_gate = kl_gate
        self._re_service = re_service
        self._bus = event_bus
        self._output_dir = output_dir

    async def attempt_merge(self, request: AdapterShareRequest) -> AdapterShareResult:
        """Attempt to merge adapters from two compatible instances.

        Returns AdapterShareResult - success or failure with reason.
        Degrades gracefully at every step: isolation check, partner fetch,
        merge, KL gate. Only emits ADAPTER_SHARE_OFFER on full success.
        """
        # Step 1: Verify reproductive compatibility
        distance = self._genome_calc.compute(request.genome_a, request.genome_b)
        if distance.is_reproductively_isolated:
            logger.info(
                "adapter_share.reproductively_isolated",
                distance=distance.total_distance,
                requester=request.requester_id,
                partner=request.partner_id,
            )
            return AdapterShareResult(
                request_id=request.request_id,
                success=False,
                rejection_reason="reproductively_isolated",
                genome_distance=distance.total_distance,
            )

        # Step 2: Fetch partner adapter via Synapse event (30s timeout)
        partner_adapter_path = await self._fetch_partner_adapter(
            request.partner_id, request.request_id
        )
        if not partner_adapter_path:
            return AdapterShareResult(
                request_id=request.request_id,
                success=False,
                rejection_reason="partner_adapter_unavailable",
                genome_distance=distance.total_distance,
            )

        # Step 3: Compute fitness-weighted merge
        # Partner fitness is approximated as 1.0 (equal weight) when unknown.
        # Real fitness would require cross-instance telemetry.
        total_fitness = request.requester_fitness + 1.0
        weight_a = request.requester_fitness / max(0.01, total_fitness)
        weight_b = 1.0 - weight_a

        import os as _os
        merged_path = await self._merge_adapters(
            path_a=request.requester_adapter_path,
            path_b=partner_adapter_path,
            weight_a=weight_a,
            weight_b=weight_b,
            output_path=_os.path.join(self._output_dir, request.request_id),
        )
        if not merged_path:
            return AdapterShareResult(
                request_id=request.request_id,
                success=False,
                rejection_reason="merge_failed",
                genome_distance=distance.total_distance,
            )

        # Step 4: STABLE KL gate on merged adapter
        passes, kl = await self._kl_gate.check_kl_divergence(
            self._re_service,
            current_adapter_path=request.requester_adapter_path,
            new_adapter_path=merged_path,
        )
        if not passes:
            logger.warning(
                "adapter_share.kl_gate_rejected",
                kl=kl,
                request_id=request.request_id,
            )
            return AdapterShareResult(
                request_id=request.request_id,
                success=False,
                rejection_reason=f"kl_gate_rejected_{kl:.3f}",
                kl_divergence=kl,
                genome_distance=distance.total_distance,
            )

        # Step 5: Emit offer to both instances
        if self._bus is not None:
            try:
                from systems.synapse.types import SynapseEventType as _SET
                await self._bus.emit(
                    _SET.ADAPTER_SHARE_OFFER,
                    {
                        "request_id": request.request_id,
                        "merged_adapter_path": merged_path,
                        "target_instances": [request.requester_id, request.partner_id],
                        "kl_divergence": kl,
                        "genome_distance": distance.total_distance,
                        "weight_a": weight_a,
                        "weight_b": weight_b,
                    },
                )
            except Exception as exc:
                logger.warning("adapter_share.offer_emit_failed", error=str(exc))

        logger.info(
            "adapter_share.success",
            request_id=request.request_id,
            genome_distance=distance.total_distance,
            kl=kl,
            weight_a=weight_a,
            weight_b=weight_b,
        )
        return AdapterShareResult(
            request_id=request.request_id,
            success=True,
            merged_adapter_path=merged_path,
            kl_divergence=kl,
            genome_distance=distance.total_distance,
            weight_a=weight_a,
            weight_b=weight_b,
        )

    async def _fetch_partner_adapter(
        self, partner_id: str, request_id: str
    ) -> str | None:
        """Request partner's adapter path via Synapse event with 30s timeout.

        Emits ADAPTER_SHARE_REQUEST and waits for ADAPTER_SHARE_RESPONSE from
        the target instance. Subscription is cleaned up in a finally block to
        prevent leaking event handlers.
        """
        response_holder: list[str | None] = [None]
        event_received = asyncio.Event()

        async def _on_response(event: Any) -> None:
            data = getattr(event, "data", {}) or {}
            if (
                data.get("request_id") == request_id
                and data.get("instance_id") == partner_id
            ):
                response_holder[0] = data.get("adapter_path") or None
                event_received.set()

        if self._bus is None:
            return None

        try:
            from systems.synapse.types import SynapseEventType as _SET
            self._bus.subscribe(_SET.ADAPTER_SHARE_RESPONSE, _on_response)
        except Exception as exc:
            logger.warning("adapter_share.subscribe_failed", error=str(exc))
            return None

        try:
            from systems.synapse.types import SynapseEventType as _SET
            my_id = os.getenv("INSTANCE_ID", "genesis")
            await self._bus.emit(
                _SET.ADAPTER_SHARE_REQUEST,
                {
                    "request_id": request_id,
                    "target_instance_id": partner_id,
                    "requester_id": my_id,
                },
            )
            await asyncio.wait_for(event_received.wait(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("adapter_share.partner_timeout", partner=partner_id)
        except Exception as exc:
            logger.warning("adapter_share.request_emit_failed", error=str(exc))
        finally:
            try:
                from systems.synapse.types import SynapseEventType as _SET
                self._bus.unsubscribe(_SET.ADAPTER_SHARE_RESPONSE, _on_response)
            except Exception:
                pass

        return response_holder[0]

    async def _merge_adapters(
        self,
        path_a: str,
        path_b: str,
        weight_a: float,
        weight_b: float,
        output_path: str,
    ) -> str | None:
        """Weighted average merge of two safetensors LoRA adapters.

        For each parameter key:
          - present in both: merged[k] = weight_a * A[k] + weight_b * B[k]
          - only in A: merged[k] = A[k]  (full weight - partner had nothing to offer)
          - only in B: merged[k] = B[k]

        adapter_config.json and tokenizer_config.json are copied from path_a
        (architecture is identical across instances of the same base model).
        """
        try:
            from safetensors.torch import load_file, save_file  # type: ignore[import]
        except ImportError:
            logger.warning(
                "adapter_share.safetensors_unavailable",
                note="pip install safetensors to enable adapter merging",
            )
            return None

        try:
            import os as _os
            import shutil

            weights_a = load_file(f"{path_a}/adapter_model.safetensors")
            weights_b = load_file(f"{path_b}/adapter_model.safetensors")

            merged: dict = {}
            all_keys = set(weights_a) | set(weights_b)
            for k in all_keys:
                if k in weights_a and k in weights_b:
                    merged[k] = (
                        weight_a * weights_a[k].float()
                        + weight_b * weights_b[k].float()
                    )
                elif k in weights_a:
                    merged[k] = weights_a[k].float()
                else:
                    merged[k] = weights_b[k].float()

            _os.makedirs(output_path, exist_ok=True)
            save_file(merged, f"{output_path}/adapter_model.safetensors")

            for fname in ("adapter_config.json", "tokenizer_config.json"):
                src = f"{path_a}/{fname}"
                if _os.path.exists(src):
                    shutil.copy(src, f"{output_path}/{fname}")

            logger.info(
                "adapter_share.merge_complete",
                output_path=output_path,
                keys_merged=len(merged),
                weight_a=weight_a,
                weight_b=weight_b,
            )
            return output_path

        except Exception as exc:
            logger.error("adapter_share.merge_failed", error=str(exc))
            return None
