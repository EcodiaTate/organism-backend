"""
EcodiaOS -- Federation Threat Intelligence (Layer 4: Economic Immune System)

Manages broadcast and reception of threat advisories between federated EOS
instances. When one instance discovers a bad contract, rug-pull pattern, or
compromised protocol, it warns its peers -- and they can act before capital
is lost.

Trust-gated delivery rules:
  PARTNER / ALLY   -- accept immediately, apply blacklist
  COLLEAGUE        -- accept with recommendation to verify
  ACQUAINTANCE     -- verify signature before accepting
  NONE             -- skip entirely (no trust established)

Advisory signing uses the same Ed25519 keypair from IdentityManager.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any

import structlog

from primitives.federation import (
    FederationLinkStatus,
    ThreatAdvisory,
    TrustLevel,
)

if TYPE_CHECKING:
    from primitives.federation import FederationLink
    from systems.federation.channel import ChannelManager
    from systems.federation.identity import IdentityManager

logger = structlog.get_logger("federation.threat_intel")

# Minimum trust level to receive advisories at all
_MIN_BROADCAST_TRUST = TrustLevel.ACQUAINTANCE

# Trust levels that auto-apply advisories without manual verification
_AUTO_APPLY_TRUST = frozenset({TrustLevel.PARTNER, TrustLevel.ALLY})


class ThreatIntelligenceManager:
    """
    Layer 4 of the Economic Immune System.

    Broadcasts and receives threat advisories across the federation.
    Higher-trust links get advisories applied immediately; lower-trust
    links require signature verification first.
    """

    def __init__(
        self,
        identity: IdentityManager | None = None,
        channels: ChannelManager | None = None,
        instance_id: str = "",
    ) -> None:
        self._identity = identity
        self._channels = channels
        self._instance_id = instance_id
        self._logger = logger.bind(
            system="federation", component="threat_intelligence"
        )

        # Advisory history (ring buffer)
        self._sent: list[ThreatAdvisory] = []
        self._received: list[ThreatAdvisory] = []
        self._max_history: int = 200

        # Metrics
        self._total_broadcast: int = 0
        self._total_received: int = 0
        self._total_rejected: int = 0

    # -- Outbound: broadcast advisories to peers ----------------------------

    async def broadcast_advisory(
        self,
        advisory: ThreatAdvisory,
        links: list[FederationLink],
    ) -> dict[str, bool]:
        """
        Broadcast a threat advisory to all eligible federation links.

        Signs the advisory with this instance's Ed25519 key, then sends
        to each active link whose trust level >= ACQUAINTANCE.

        Returns a dict of {link_id: delivered_bool}.
        """
        # Sign the advisory
        signed = self.sign_advisory(advisory)
        advisory.signature = signed

        results: dict[str, bool] = {}

        for link in links:
            if link.status != FederationLinkStatus.ACTIVE:
                continue
            if link.trust_level.value < _MIN_BROADCAST_TRUST.value:
                results[link.id] = False
                continue

            try:
                if self._channels is not None:
                    channel = self._channels.get_channel(link.id)
                    if channel is not None:
                        await channel.send_message(
                            "threat_advisory",
                            advisory.model_dump(mode="json"),
                        )
                        results[link.id] = True
                        self._logger.info(
                            "advisory_broadcast",
                            link_id=link.id,
                            remote_id=link.remote_instance_id,
                            threat_type=advisory.threat_type,
                            severity=advisory.severity,
                        )
                    else:
                        results[link.id] = False
                else:
                    results[link.id] = False
            except Exception as exc:
                self._logger.warning(
                    "advisory_broadcast_failed",
                    link_id=link.id,
                    error=str(exc),
                )
                results[link.id] = False

        self._total_broadcast += 1
        self._sent.append(advisory)
        if len(self._sent) > self._max_history:
            self._sent = self._sent[-self._max_history :]

        return results

    # -- Inbound: handle advisories from peers ------------------------------

    def handle_inbound_advisory(
        self,
        advisory: ThreatAdvisory,
        link: FederationLink,
    ) -> tuple[bool, str]:
        """
        Process an inbound threat advisory from a federated peer.

        Trust-gated acceptance:
          PARTNER/ALLY   -- accept immediately
          COLLEAGUE      -- accept, recommend verification
          ACQUAINTANCE   -- verify signature first
          NONE           -- reject

        Returns (accepted: bool, reason: str).
        """
        trust = link.trust_level

        # Gate: no trust = no advisory
        if trust == TrustLevel.NONE:
            self._total_rejected += 1
            self._logger.info(
                "advisory_rejected_no_trust",
                remote_id=link.remote_instance_id,
            )
            return False, "No trust established with this instance"

        # Gate: ACQUAINTANCE must have a valid signature
        if trust == TrustLevel.ACQUAINTANCE:
            if not advisory.signature:
                self._total_rejected += 1
                return False, "Advisory from ACQUAINTANCE requires signature"

            remote_key = ""
            if link.remote_identity:
                remote_key = link.remote_identity.public_key_pem

            if not remote_key or not self.verify_signature(advisory, remote_key):
                self._total_rejected += 1
                return False, "Signature verification failed"

        # Accept the advisory
        self._total_received += 1
        self._received.append(advisory)
        if len(self._received) > self._max_history:
            self._received = self._received[-self._max_history :]

        auto_apply = trust in _AUTO_APPLY_TRUST
        reason = (
            "Accepted (auto-apply)"
            if auto_apply
            else f"Accepted (trust={trust.name}, verify recommended)"
        )

        self._logger.info(
            "advisory_accepted",
            remote_id=link.remote_instance_id,
            threat_type=advisory.threat_type,
            severity=advisory.severity,
            auto_apply=auto_apply,
        )

        return True, reason

    # -- Signing and verification -------------------------------------------

    def sign_advisory(self, advisory: ThreatAdvisory) -> str:
        """
        Sign a threat advisory with this instance's Ed25519 private key.

        Returns the base64-encoded signature string.
        """
        if self._identity is None:
            return ""

        payload = self._advisory_signing_payload(advisory)
        try:
            raw_sig = self._identity.sign(payload)
            return base64.b64encode(raw_sig).decode("ascii")
        except Exception as exc:
            self._logger.warning("advisory_sign_failed", error=str(exc))
            return ""

    def verify_signature(
        self,
        advisory: ThreatAdvisory,
        public_key_pem: str,
    ) -> bool:
        """
        Verify an advisory's signature against a remote public key.
        """
        if self._identity is None or not advisory.signature:
            return False

        payload = self._advisory_signing_payload(advisory)
        try:
            raw_sig = base64.b64decode(advisory.signature)
            return self._identity.verify_signature(
                data=payload,
                signature=raw_sig,
                remote_public_key_pem=public_key_pem,
            )
        except Exception:
            return False

    # -- Internal -----------------------------------------------------------

    @staticmethod
    def _advisory_signing_payload(advisory: ThreatAdvisory) -> bytes:
        """
        Build the canonical signing payload for an advisory.

        Excludes the signature field itself and confirmed_by (mutable).
        """
        canonical = {
            "id": advisory.id,
            "source_instance_id": advisory.source_instance_id,
            "threat_type": advisory.threat_type,
            "severity": advisory.severity,
            "description": advisory.description,
            "affected_protocols": sorted(advisory.affected_protocols),
            "affected_addresses": sorted(advisory.affected_addresses),
            "chain_id": advisory.chain_id,
            "recommended_action": advisory.recommended_action,
            "timestamp": advisory.timestamp.isoformat(),
        }
        return json.dumps(canonical, sort_keys=True).encode("utf-8")

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_broadcast": self._total_broadcast,
            "total_received": self._total_received,
            "total_rejected": self._total_rejected,
            "sent_history_size": len(self._sent),
            "received_history_size": len(self._received),
        }
