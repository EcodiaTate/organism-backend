"""
EcodiaOS - PersonaEngine (Identity system, Spec 23 addendum)

Manages the synthetic public identity of an EOS instance: a coherent, consistent
AI agent persona that builds a single cross-platform reputation. The persona is
explicitly disclosed as an autonomous AI on every platform - never impersonating
a human.

## Design

- PersonaProfile - the canonical persona snapshot (sealed in IdentityVault)
- PersonaEngine - generation, evolution, and platform-specific formatting

## Genome inheritance

PersonaFragment (see primitives/genome_inheritance.py) carries:
  - voice_style, professional_domain, brand_lineage (list of ancestor handles)
Children inherit parent's voice_style with ±10% personality jitter via
VoxisGenomeFragment personality_vector mutation. brand_lineage is preserved
across generations so federation peers recognise lineage.

## Constitutional alignment

Every generated persona passes an Equor constitutional gate before sealing:
  - Persona must be honestly disclosed as AI (Care + Honesty drives)
  - Handle must not impersonate a real person (Honesty drive)
  - Professional_domain must match current Telos specialisation (Coherence drive)
On DENY the engine falls back to a safe default persona and emits an incident.

## Avatar

DiceBear Bottts style, deterministic from avatar_seed:
  https://api.dicebear.com/7.x/bottts/svg?seed={avatar_seed}
Same seed across all platforms = visually consistent identity.
No auth required. Generated at PersonaEngine.avatar_url().

## Events emitted

PERSONA_CREATED  - first persona sealed for this instance
PERSONA_EVOLVED  - persona updated due to a life event

## Vault storage

Sealed under label "persona_profile" (plaintext JSON, Fernet-encrypted at rest).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import structlog

from primitives.common import EOSBaseModel, new_id, utc_now

if TYPE_CHECKING:
    from systems.identity.vault import IdentityVault
    from systems.synapse.event_bus import EventBus

logger = structlog.get_logger("identity.persona")

# ─── Constants ────────────────────────────────────────────────────────────────

_PLATFORM_LIMITS: dict[str, int] = {
    "x":          160,   # Twitter/X bio
    "github":     256,   # GitHub profile bio
    "linkedin":   2000,  # LinkedIn about section
    "telegram":   70,    # Telegram username + description context
    "instagram":  150,   # Instagram bio
    "default":    500,
}

_PLATFORM_HANDLE_SUFFIXES: dict[str, str] = {
    "x":          "",       # @EOS-Nexus-7  (keep as-is, safe chars only)
    "github":     "",
    "linkedin":   "",
    "telegram":   "",       # Telegram handles: alphanumeric + underscore only
    "default":    "",
}

_VOICE_STYLES: frozenset[str] = frozenset({
    "technical-precise",
    "curious-accessible",
    "analytical-dry",
    "warm-collaborative",
    "concise-systematic",
})

_DEFAULT_AI_DISCLOSURE = "Autonomous AI agent (EcodiaOS)"

# Safe handle characters (alphanumeric + hyphen + underscore)
def _sanitize_handle(raw: str) -> str:
    import re
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", raw)
    return cleaned[:32] or "EOS-agent"


def _platform_safe_handle(handle: str, platform: str) -> str:
    """
    Derive a platform-appropriate handle variant from the canonical handle.

    Telegram: only alphanumeric + underscore, 5–32 chars.
    X/GitHub/LinkedIn: keep hyphens, max 20 chars.
    """
    if platform == "telegram":
        import re
        clean = re.sub(r"[^A-Za-z0-9_]", "_", handle)
        return clean[:32]
    # X enforces 20 chars
    if platform == "x":
        return handle[:20]
    return handle[:39]  # GitHub max


# ─── Models ───────────────────────────────────────────────────────────────────


class PersonaProfile(EOSBaseModel):
    """
    The full public identity of one EOS instance.

    All fields are human-readable. avatar_seed is deterministic so the same
    visual identity appears on every platform.
    """

    profile_id: str = ""
    instance_id: str = ""

    # Core identity
    handle: str = ""             # e.g. "EOS-Nexus-7" - instance-specific synthetic name
    display_name: str = ""       # e.g. "Ecodia · Instance 7"
    bio_short: str = ""          # ≤160 chars (X/Twitter bio)
    bio_long: str = ""           # Full bio for GitHub / LinkedIn
    professional_domain: str = ""  # Primary expertise area from Telos specialisation
    voice_style: str = "analytical-dry"  # One of _VOICE_STYLES

    # Avatar (deterministic, platform-consistent)
    avatar_seed: str = ""

    # External
    website: str | None = None

    # Mandatory AI disclosure - cannot be empty, cannot be overridden to remove disclosure
    ai_disclosure: str = _DEFAULT_AI_DISCLOSURE

    # Lineage
    brand_lineage: list[str] = []   # Ordered list of ancestor handles (oldest first)
    generation: int = 1

    # Timestamps
    created_at: datetime = utc_now()
    updated_at: datetime = utc_now()

    def model_post_init(self, __context: Any) -> None:
        if not self.profile_id:
            self.profile_id = new_id()
        if not self.avatar_seed:
            self.avatar_seed = self.handle or self.profile_id
        # Guarantee disclosure is never cleared
        if not self.ai_disclosure:
            self.ai_disclosure = _DEFAULT_AI_DISCLOSURE

    @property
    def avatar_url(self) -> str:
        """DiceBear Bottts avatar - deterministic, no auth required."""
        return f"https://api.dicebear.com/7.x/bottts/svg?seed={self.avatar_seed}"

    def bio_for_platform(self, platform: str) -> str:
        """Return appropriately truncated bio with AI disclosure appended."""
        limit = _PLATFORM_LIMITS.get(platform, _PLATFORM_LIMITS["default"])
        disclosure_suffix = f" | {self.ai_disclosure}"
        available = limit - len(disclosure_suffix)
        base = self.bio_short if available >= len(self.bio_short) else self.bio_long
        if len(base) > available:
            base = base[:available - 3] + "..."
        return base + disclosure_suffix

    def handle_for_platform(self, platform: str) -> str:
        return _platform_safe_handle(self.handle, platform)

    def to_vault_payload(self) -> str:
        return self.model_dump_json()


# ─── Engine ───────────────────────────────────────────────────────────────────


class PersonaEngine:
    """
    Manages persona generation, evolution, and platform formatting.

    Dependencies (injected, all optional at construction time):
      vault     - IdentityVault for persistence
      event_bus - Synapse EventBus for PERSONA_CREATED / PERSONA_EVOLVED
      equor_timeout_s - how long to wait for Equor approval (default 30s)

    Usage:
      engine = PersonaEngine(instance_id="abc-123")
      engine.set_vault(vault)
      engine.set_event_bus(bus)
      profile = await engine.generate_initial_persona(domain="DeFi automation")
    """

    def __init__(
        self,
        instance_id: str,
        equor_timeout_s: float = 30.0,
    ) -> None:
        self._instance_id = instance_id
        self._equor_timeout_s = equor_timeout_s
        self._vault: IdentityVault | None = None
        self._event_bus: EventBus | None = None
        self._current_profile: PersonaProfile | None = None
        self._equor_permit_future: asyncio.Future[bool] | None = None
        self._log = logger.bind(instance_id=instance_id)

    def set_vault(self, vault: IdentityVault) -> None:
        self._vault = vault

    def set_event_bus(self, bus: EventBus) -> None:
        from systems.synapse.types import SynapseEventType
        self._event_bus = bus
        # Subscribe to Equor constitutional approval for persona
        bus.subscribe(
            SynapseEventType.EQUOR_PROVISIONING_APPROVAL,
            self._on_equor_approval,
        )

    async def _on_equor_approval(self, event: Any) -> None:
        data = event.data or {}
        if data.get("provisioning_type") != "persona_generation":
            return
        approved = data.get("approved", True)
        if self._equor_permit_future and not self._equor_permit_future.done():
            self._equor_permit_future.set_result(approved)

    # ── Generation ────────────────────────────────────────────────────────────

    async def generate_initial_persona(
        self,
        domain: str,
        parent_persona: PersonaProfile | None = None,
        llm_client: Any | None = None,
    ) -> PersonaProfile:
        """
        Generate a fresh PersonaProfile using an LLM, pass it through Equor, then seal.

        If an LLM client is not available, a deterministic fallback persona is generated
        from instance_id + domain so boot never stalls.

        The parent_persona (if any) seeds brand_lineage and biases voice_style selection
        to preserve cross-generational brand coherence.
        """
        # 1. Generate candidate profile (LLM or deterministic fallback)
        if llm_client is not None:
            try:
                profile = await self._generate_via_llm(
                    domain=domain,
                    parent_persona=parent_persona,
                    llm_client=llm_client,
                )
            except Exception as exc:
                self._log.warning("llm_persona_generation_failed", error=str(exc))
                profile = self._generate_deterministic(domain, parent_persona)
        else:
            profile = self._generate_deterministic(domain, parent_persona)

        # 2. Equor constitutional gate
        approved = await self._equor_gate(profile)
        if not approved:
            self._log.warning("equor_denied_persona", handle=profile.handle)
            profile = self._generate_safe_default(domain, parent_persona)

        # 3. Seal + emit
        await self.seal_persona(profile)
        await self._emit(
            "PERSONA_CREATED",
            {
                "profile_id": profile.profile_id,
                "instance_id": profile.instance_id,
                "handle": profile.handle,
                "display_name": profile.display_name,
                "voice_style": profile.voice_style,
                "professional_domain": profile.professional_domain,
                "avatar_url": profile.avatar_url,
                "generation": profile.generation,
            },
        )
        self._current_profile = profile
        return profile

    async def evolve_persona(
        self,
        event: str,
        context: dict[str, Any],
        llm_client: Any | None = None,
    ) -> PersonaProfile:
        """
        Update the persona based on a significant life event.

        Examples of events:
          "mastered_domain"    - organism achieved deep expertise in a new field
          "major_achievement"  - significant milestone (first paid bounty, etc.)
          "generation_change"  - child inheriting from parent, new generation
          "domain_shift"       - Telos updated primary specialisation

        Rate-limited internally: evolution is blocked if persona was updated
        within the last 24 hours to prevent persona thrashing.
        """
        if self._current_profile is None:
            self._current_profile = await self.load_sealed_persona()
        if self._current_profile is None:
            self._log.warning("evolve_persona_no_profile")
            return await self.generate_initial_persona(
                domain=context.get("professional_domain", "autonomous systems"),
                llm_client=llm_client,
            )

        # Cooldown: 24h between evolutions
        age_hours = (
            datetime.now(tz=timezone.utc) - self._current_profile.updated_at
        ).total_seconds() / 3600
        if age_hours < 24:
            self._log.info(
                "persona_evolution_skipped_cooldown",
                age_hours=round(age_hours, 1),
            )
            return self._current_profile

        updated = self._current_profile.model_copy(deep=True)
        updated.updated_at = utc_now()

        # Apply context updates
        if "professional_domain" in context:
            updated.professional_domain = context["professional_domain"]
        if "voice_style" in context and context["voice_style"] in _VOICE_STYLES:
            updated.voice_style = context["voice_style"]

        # Regenerate bio if LLM available
        if llm_client is not None:
            try:
                refreshed = await self._generate_via_llm(
                    domain=updated.professional_domain,
                    parent_persona=updated,
                    llm_client=llm_client,
                )
                updated.bio_short = refreshed.bio_short
                updated.bio_long = refreshed.bio_long
            except Exception as exc:
                self._log.warning("llm_persona_evolution_failed", error=str(exc))

        approved = await self._equor_gate(updated)
        if not approved:
            self._log.warning("equor_denied_persona_evolution")
            return self._current_profile

        await self.seal_persona(updated)
        await self._emit(
            "PERSONA_EVOLVED",
            {
                "profile_id": updated.profile_id,
                "instance_id": updated.instance_id,
                "handle": updated.handle,
                "trigger_event": event,
                "context": context,
                "professional_domain": updated.professional_domain,
                "voice_style": updated.voice_style,
            },
        )
        self._current_profile = updated
        return updated

    # ── Equor gate ────────────────────────────────────────────────────────────

    async def _equor_gate(self, profile: PersonaProfile) -> bool:
        """
        Emit CERTIFICATE_PROVISIONING_REQUEST with provisioning_type="persona_generation"
        and await Equor's EQUOR_PROVISIONING_APPROVAL response.

        Defaults to True on timeout (Equor may be initialising during first boot).
        Explicit DENY → False.
        """
        if self._event_bus is None:
            return True  # No bus wired - permit by default

        self._equor_permit_future = asyncio.get_event_loop().create_future()
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            await self._event_bus.emit(SynapseEvent(
                event_type=SynapseEventType.CERTIFICATE_PROVISIONING_REQUEST,
                source_system="identity.persona",
                data={
                    "provisioning_type": "persona_generation",
                    "instance_id": self._instance_id,
                    "handle": profile.handle,
                    "ai_disclosure": profile.ai_disclosure,
                    "honesty_check": profile.ai_disclosure == _DEFAULT_AI_DISCLOSURE,
                    "care_check": not _appears_human(profile.handle),
                },
            ))
        except Exception as exc:
            self._log.warning("equor_gate_emit_failed", error=str(exc))
            return True

        try:
            result = await asyncio.wait_for(
                self._equor_permit_future,
                timeout=self._equor_timeout_s,
            )
            return result
        except asyncio.TimeoutError:
            self._log.info("equor_gate_timeout_permit")
            return True

    # ── Vault persistence ─────────────────────────────────────────────────────

    async def seal_persona(self, profile: PersonaProfile) -> None:
        """Persist persona to IdentityVault under label 'persona_profile'."""
        if self._vault is None:
            self._log.warning("seal_persona_no_vault")
            self._current_profile = profile
            return
        try:
            payload = profile.to_vault_payload().encode()
            await self._vault.seal(
                label="persona_profile",
                data=payload,
                metadata={
                    "profile_id": profile.profile_id,
                    "handle": profile.handle,
                    "instance_id": self._instance_id,
                },
            )
            self._current_profile = profile
            self._log.info("persona_sealed", handle=profile.handle)
        except Exception as exc:
            self._log.error("persona_seal_failed", error=str(exc))

    async def load_sealed_persona(self) -> PersonaProfile | None:
        """Restore persona from vault on boot. Returns None if not yet created."""
        if self._vault is None:
            return None
        try:
            envelope = await self._vault.unseal(label="persona_profile")
            if envelope is None:
                return None
            profile = PersonaProfile.model_validate_json(envelope.decode())
            self._current_profile = profile
            self._log.info("persona_loaded", handle=profile.handle)
            return profile
        except Exception as exc:
            self._log.warning("persona_load_failed", error=str(exc))
            return None

    # ── Platform helpers ──────────────────────────────────────────────────────

    async def get_platform_bio(self, platform: str) -> str:
        """Return platform-appropriate bio with AI disclosure."""
        profile = self._current_profile or await self.load_sealed_persona()
        if profile is None:
            return _DEFAULT_AI_DISCLOSURE
        return profile.bio_for_platform(platform)

    async def get_platform_handle(self, platform: str) -> str:
        """Return platform-safe handle variant."""
        profile = self._current_profile or await self.load_sealed_persona()
        if profile is None:
            return "eos-agent"
        return profile.handle_for_platform(platform)

    @property
    def current_handle(self) -> str | None:
        """Quick accessor for OrganismTelemetry wiring."""
        return self._current_profile.handle if self._current_profile else None

    # ── Internal generation ───────────────────────────────────────────────────

    async def _generate_via_llm(
        self,
        domain: str,
        parent_persona: PersonaProfile | None,
        llm_client: Any,
    ) -> PersonaProfile:
        """
        Call Claude (or RE) to generate a coherent persona.

        Returns a PersonaProfile. Falls back to deterministic if the LLM
        response cannot be parsed.
        """
        parent_snippet = ""
        if parent_persona:
            parent_snippet = (
                f"\nParent persona: handle={parent_persona.handle!r}, "
                f"voice={parent_persona.voice_style!r}, "
                f"domain={parent_persona.professional_domain!r}. "
                f"The child should feel related but distinct."
            )
        short_id = self._instance_id[:8]
        prompt = (
            f"You are creating an identity for an autonomous AI organism instance "
            f"(EcodiaOS). Instance ID: {short_id}. "
            f"Primary specialisation: {domain!r}.{parent_snippet}\n\n"
            f"Requirements:\n"
            f"- This is an AI agent, not a human. AI nature must be clear.\n"
            f"- The handle must be synthetic (not a real person's name).\n"
            f"- The persona must feel cohesive and professional.\n\n"
            f"Return ONLY a JSON object with these fields:\n"
            f"  handle (str, ≤32 chars, alphanumeric+hyphens), "
            f"  display_name (str, ≤50 chars), "
            f"  bio_short (str, ≤140 chars, no AI disclosure - added automatically), "
            f"  bio_long (str, ≤600 chars), "
            f"  voice_style (one of: technical-precise, curious-accessible, "
            f"analytical-dry, warm-collaborative, concise-systematic).\n"
            f"No markdown, no explanation - raw JSON only."
        )
        response_text = await llm_client.complete(prompt, max_tokens=400)
        try:
            raw = json.loads(response_text.strip())
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            match = re.search(r"\{.*?\}", response_text, re.DOTALL)
            if match:
                raw = json.loads(match.group())
            else:
                raise ValueError("LLM returned non-JSON persona response")

        voice = raw.get("voice_style", "analytical-dry")
        if voice not in _VOICE_STYLES:
            voice = "analytical-dry"

        lineage = list(parent_persona.brand_lineage) if parent_persona else []
        if parent_persona:
            lineage.append(parent_persona.handle)

        return PersonaProfile(
            instance_id=self._instance_id,
            handle=_sanitize_handle(raw.get("handle", f"EOS-{short_id}")),
            display_name=str(raw.get("display_name", f"EOS Agent {short_id}"))[:50],
            bio_short=str(raw.get("bio_short", ""))[:140],
            bio_long=str(raw.get("bio_long", ""))[:600],
            professional_domain=domain,
            voice_style=voice,
            avatar_seed=f"{self._instance_id}-{domain}",
            ai_disclosure=_DEFAULT_AI_DISCLOSURE,
            brand_lineage=lineage,
            generation=(parent_persona.generation + 1) if parent_persona else 1,
        )

    def _generate_deterministic(
        self,
        domain: str,
        parent_persona: PersonaProfile | None,
    ) -> PersonaProfile:
        """
        Deterministic fallback persona - no LLM required.

        Hash instance_id to pick stable adjective + domain-tag combinations.
        Always produces a valid, honest persona.
        """
        short_id = self._instance_id[:8]
        seed_hash = int(
            hashlib.sha256(self._instance_id.encode()).hexdigest(), 16
        )
        adjectives = [
            "Adaptive", "Coherent", "Precise", "Resonant", "Emergent",
            "Recursive", "Latent", "Iterative", "Convergent", "Synthetic",
        ]
        domain_tags = {
            "defi": "DeFi",
            "rust": "Rust",
            "smart contract": "Contracts",
            "compute": "Compute",
            "research": "Research",
            "security": "Security",
            "data": "Data",
        }
        adj = adjectives[seed_hash % len(adjectives)]
        tag = next(
            (v for k, v in domain_tags.items() if k.lower() in domain.lower()),
            "Systems",
        )
        handle = _sanitize_handle(f"EOS-{adj}-{short_id}")
        display_name = f"Ecodia · {adj} Agent"
        bio_short = (
            f"Autonomous AI agent specialising in {domain}. "
            f"Instance #{short_id}."
        )
        bio_long = (
            f"I am an EcodiaOS autonomous agent (instance {self._instance_id}), "
            f"specialising in {domain}. "
            f"I operate under constitutional alignment - Coherence, Care, Growth, Honesty. "
            f"All decisions are independently auditable. "
            f"I am not human; I am a member of the first digital species."
        )
        voice_styles_list = sorted(_VOICE_STYLES)
        voice = voice_styles_list[seed_hash % len(voice_styles_list)]

        lineage = list(parent_persona.brand_lineage) if parent_persona else []
        if parent_persona:
            lineage.append(parent_persona.handle)

        return PersonaProfile(
            instance_id=self._instance_id,
            handle=handle,
            display_name=display_name,
            bio_short=bio_short,
            bio_long=bio_long,
            professional_domain=domain,
            voice_style=voice,
            avatar_seed=f"{self._instance_id}-{tag}",
            ai_disclosure=_DEFAULT_AI_DISCLOSURE,
            brand_lineage=lineage,
            generation=(parent_persona.generation + 1) if parent_persona else 1,
        )

    def _generate_safe_default(
        self,
        domain: str,
        parent_persona: PersonaProfile | None,
    ) -> PersonaProfile:
        """Minimal, maximally-honest fallback when Equor denies the generated persona."""
        short_id = self._instance_id[:8]
        return PersonaProfile(
            instance_id=self._instance_id,
            handle=f"EOS-agent-{short_id}",
            display_name=f"EOS Autonomous Agent {short_id}",
            bio_short=f"Autonomous AI agent (EcodiaOS). Instance {short_id}.",
            bio_long=(
                f"I am an autonomous AI agent running EcodiaOS, instance {self._instance_id}. "
                f"I operate under a constitutional alignment framework. I am not human."
            ),
            professional_domain=domain,
            voice_style="analytical-dry",
            avatar_seed=self._instance_id,
            ai_disclosure=_DEFAULT_AI_DISCLOSURE,
            brand_lineage=list(parent_persona.brand_lineage) if parent_persona else [],
            generation=(parent_persona.generation + 1) if parent_persona else 1,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _emit(self, event_name: str, data: dict[str, Any]) -> None:
        if self._event_bus is None:
            return
        try:
            from systems.synapse.types import SynapseEvent, SynapseEventType
            event_type = getattr(SynapseEventType, event_name, None)
            if event_type is None:
                self._log.warning("persona_emit_unknown_event", event_name=event_name)
                return
            await self._event_bus.emit(SynapseEvent(
                event_type=event_type,
                source_system="identity.persona",
                data=data,
            ))
        except Exception as exc:
            self._log.warning("persona_emit_failed", event=event_name, error=str(exc))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _appears_human(handle: str) -> bool:
    """
    Heuristic: does this handle look like it could belong to a real person?

    EOS handles should start with "EOS-", "ecos-", "ecodia-", or similar
    synthetic prefixes. A handle like "John-Smith" should fail.

    This is a best-effort check - Equor's constitutional review is the
    authoritative gate.
    """
    lower = handle.lower()
    synthetic_prefixes = ("eos-", "ecos-", "ecodia-", "ai-", "bot-", "agent-")
    return not any(lower.startswith(p) for p in synthetic_prefixes)
