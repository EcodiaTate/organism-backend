"""
EcodiaOS — Nexus: Integration Protocols

Protocol-based dependency injection for the systems Nexus depends on.
Nexus never imports Logos, Fovea, Federation, or Thymos directly —
they satisfy these protocols instead.

Protocols exposed:
- LogosWorldModelProtocol: read world model structure, schemas, complexity
- FoveaAttentionProtocol: read attention weight profiles
- FederationFragmentProtocol: send/receive fragments over federation links
- ThymosDriveSinkProtocol: route divergence pressure as growth drive signals
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from systems.nexus.types import (
        DivergencePressure,
        InstanceDivergenceProfile,
        ShareableWorldModelFragment,
        WorldModelFragmentShare,
        WorldModelFragmentShareResponse,
    )

# ─── Logos: World Model Access ───────────────────────────────────


@runtime_checkable
class LogosWorldModelProtocol(Protocol):
    """
    Protocol for Nexus to read the world model's structure.

    Nexus needs to:
    - Extract shareable fragments from the world model
    - Read schema topology for divergence measurement
    - Compute structural fingerprints for convergence detection
    """

    def get_schema_ids(self) -> list[str]:
        """Return IDs of all generative schemas in the world model."""
        ...

    def get_schema(self, schema_id: str) -> dict[str, Any] | None:
        """
        Return a schema's full data as a dict.

        Must include at minimum: name, domain, pattern, instance_count,
        compression_ratio, description.
        """
        ...

    def get_domain_coverage(self) -> list[str]:
        """Return list of domains the world model has schemas for."""
        ...

    def get_structural_fingerprint(self) -> str:
        """
        Hash of the world model's schema topology.

        Two instances with the same structural fingerprint have
        identical abstract schema graphs (even if domain labels differ).
        """
        ...

    def get_total_schemas(self) -> int:
        """Total number of generative schemas."""
        ...

    def get_complexity(self) -> float:
        """Current model complexity in bits."""
        ...

    def get_causal_link_count(self) -> int:
        """Number of causal links in the causal graph."""
        ...

    def get_total_experiences(self) -> int:
        """
        Total number of episodes processed (compressed or otherwise).

        Used for temporal divergence measurement — instances at
        different experience counts are at different developmental stages.
        """
        ...


# ─── Fovea: Attention Profile ───────────────────────────────────


@runtime_checkable
class FoveaAttentionProtocol(Protocol):
    """
    Protocol for Nexus to read Fovea's attention weight profile.

    Different instances attend to different prediction error types.
    Attentional diversity is one of the five divergence dimensions.
    """

    def get_attention_weights(self) -> dict[str, float]:
        """
        Return current attention weight profile.

        Keys are prediction error types (content, temporal, magnitude,
        source, category, causal). Values are learned weights [0, 1].
        """
        ...


# ─── Federation: Fragment Transport ──────────────────────────────


@runtime_checkable
class FederationFragmentProtocol(Protocol):
    """
    Protocol for Nexus to send/receive world model fragments
    over federation links.

    Nexus decides WHAT to share. Federation handles HOW — encryption,
    authentication, trust gating, and transport.
    """

    async def send_fragment(
        self,
        link_id: str,
        message: WorldModelFragmentShare,
    ) -> WorldModelFragmentShareResponse:
        """
        Send a world model fragment to a federated peer.

        Federation handles trust gating (COLLEAGUE+ required),
        privacy filtering, and transport. Returns the peer's response.
        """
        ...

    async def broadcast_fragment(
        self,
        message: WorldModelFragmentShare,
    ) -> dict[str, WorldModelFragmentShareResponse]:
        """
        Broadcast a fragment to all active federation links.
        Returns {link_id: response} for each link.
        """
        ...

    def get_active_link_ids(self) -> list[str]:
        """Return IDs of all active federation links."""
        ...

    async def get_remote_profile(
        self, link_id: str
    ) -> InstanceDivergenceProfile | None:
        """
        Request a remote instance's divergence profile.
        Returns None if the link is inactive or the request fails.
        """
        ...


# ─── Thymos: Growth Drive Sink ───────────────────────────────────


@runtime_checkable
class ThymosDriveSinkProtocol(Protocol):
    """
    Protocol for Nexus to route divergence pressure as a growth drive signal.

    When an instance is too similar to the federation average,
    Nexus computes a DivergencePressure and routes it to Thymos
    as a GROWTH drive signal — explore frontier domains, avoid
    saturated ones.
    """

    def receive_divergence_pressure(self, pressure: DivergencePressure) -> None:
        """
        Receive a divergence pressure signal from Nexus.

        Thymos maps this to its internal drive state, increasing
        growth pressure proportionally to pressure_magnitude.
        """
        ...


# ─── Evo: Active Hypothesis Source ──────────────────────────


@runtime_checkable
class EvoHypothesisSourceProtocol(Protocol):
    """
    Protocol for Nexus to read the set of active hypothesis IDs from Evo.

    Hypothesis diversity is one of the five divergence dimensions. Two
    instances actively testing different hypotheses are epistemically
    more distinct than instances exploring the same questions.
    """

    def get_active_hypothesis_ids(self) -> list[str]:
        """
        Return IDs of all hypotheses currently under active evaluation.

        Hypothesis IDs must be stable across calls during a session so
        that divergence measurements are consistent.
        """
        ...


# ─── Oneiros: Adversarial Dream Testing ──────────────────────


@runtime_checkable
class OneirosAdversarialProtocol(Protocol):
    """
    Protocol for Nexus to request adversarial testing via Oneiros.

    Level 4 (EMPIRICAL_INVARIANT) promotion requires a fragment to
    survive adversarial simulation in Oneiros's lucid dreaming mode.
    Nexus builds against this protocol — not Oneiros directly.
    """

    async def run_adversarial_test(
        self,
        fragment: ShareableWorldModelFragment,
    ) -> bool:
        """
        Run adversarial simulation against a fragment in lucid dreaming.

        Returns True if the fragment's abstract structure survives —
        i.e. no counter-scenario could break it.
        """
        ...


# ─── Evo: Hypothesis Competition ─────────────────────────────


@runtime_checkable
class EvoCompetitionProtocol(Protocol):
    """
    Protocol for Nexus to request hypothesis competition via Evo.

    Level 4 promotion requires a fragment to survive open competition
    against Evo's hypothesis population. If alternative hypotheses
    outperform the fragment's structure, it fails promotion.
    """

    async def run_hypothesis_competition(
        self,
        fragment: ShareableWorldModelFragment,
    ) -> bool:
        """
        Pit a fragment's abstract structure against Evo's hypothesis pool.

        Returns True if no competing hypothesis achieves better compression
        or explanatory coverage of the fragment's observations.
        """
        ...


# ─── Equor: Constitutional Protection ────────────────────────


@runtime_checkable
class EquorProtectionProtocol(Protocol):
    """
    Protocol for Nexus to route Level 4 EMPIRICAL_INVARIANTs to
    Equor for constitutional protection.

    Once protected, modifying or removing the invariant requires
    governance review (human-in-the-loop or constitutional vote).
    """

    async def protect_invariant(
        self,
        fragment: ShareableWorldModelFragment,
        evidence: dict[str, Any],
    ) -> bool:
        """
        Register a fragment as constitutionally protected.

        Returns True if Equor accepted the protection request.
        The evidence dict should include promotion metrics:
        triangulation_confidence, source_diversity, source_count,
        survived_adversarial, survived_competition.
        """
        ...
