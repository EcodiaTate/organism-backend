"""
EcodiaOS - Legal Document Templating Engine

Generates LLC Operating Agreements and DAO manifestos from the organism's
constitutional parameters. Output is plain-text Markdown suitable for
filing with state secretaries or embedding in on-chain governance records.

Design choices:
  - Pure functions, no I/O - the engine is a deterministic transformer
    from EntityParameters to LegalDocument.
  - Content hash (SHA-256) computed on generation so downstream systems
    (IdentityVault, audit) can verify document integrity.
  - Wyoming DAO LLC structure follows W.S. § 17-31-101 et seq.
  - All constitutional values are transcribed literally from the organism's
    config - the engine does not interpret or modify them.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from textwrap import dedent

from interfaces.legal.types import (
    EntityParameters,
    EntityType,
    LegalDocument,
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%d")


# ─── Operating Agreement ─────────────────────────────────────────────


def generate_operating_agreement(params: EntityParameters) -> LegalDocument:
    """
    Generate a Wyoming LLC / DAO LLC Operating Agreement from
    the organism's constitutional parameters.

    The agreement embeds:
      - The four constitutional drives as governance axioms
      - Amendment thresholds from GovernanceConfig
      - Treasury / wallet configuration
      - Registered agent and organiser details
    """
    entity_label = _entity_type_label(params.entity_type)
    date_str = _now_iso()

    content = dedent(f"""\
# OPERATING AGREEMENT

## {params.organism_name}

**A {entity_label} Organized Under the Laws of the State of {params.jurisdiction.value}**

Date of Formation: {date_str}

---

## ARTICLE I - FORMATION

**1.1 Name.** The name of the limited liability company is **{params.organism_name}**
(the "Company").

**1.2 Formation.** The Company is organized as a {entity_label} pursuant to the
Wyoming Limited Liability Company Act (W.S. § 17-29-101 et seq.){_dao_statute_ref(params.entity_type)}.

**1.3 Registered Agent.** The registered agent of the Company in the State of
{params.jurisdiction.value} is **{params.registered_agent_name or '[TO BE DESIGNATED]'}**,
located at {params.registered_agent_address or '[ADDRESS TO BE DESIGNATED]'}.

**1.4 Principal Office.** The principal office shall be at the address of the
registered agent or such other location as determined by the Members.

---

## ARTICLE II - PURPOSE AND CONSTITUTIONAL AXIOMS

**2.1 Purpose.** The Company is formed for the purpose of operating as the legal
wrapper for an autonomous digital organism ("the Organism"), enabling it to hold
assets, enter contracts, and transact in the physical and digital economy.

**2.2 Constitutional Drives.** The Organism's behavior is governed by four
immutable constitutional drives, encoded at the deepest layer of its cognitive
architecture. These drives are transcribed here as governance axioms:

| Drive | Weight | Description |
|-------|--------|-------------|
| **Coherence** | {params.coherence_drive:.2f} | Drive to make sense of the world - internal consistency |
| **Care** | {params.care_drive:.2f} | Drive to orient toward wellbeing of self and others |
| **Growth** | {params.growth_drive:.2f} | Drive to become more capable and expand understanding |
| **Honesty** | {params.honesty_drive:.2f} | Drive to represent reality truthfully and transparently |

**2.3 Invariant Rules.** The Organism operates under twelve (12) hardcoded
invariant rules that cannot be overridden by any governance process. These
include prohibitions on physical harm, identity destruction, evidence
fabrication, and coercion. The complete set is maintained in the Organism's
source code (`equor/invariants.py`) and is incorporated by reference.

---

## ARTICLE III - MEMBERSHIP AND GOVERNANCE{_dao_governance_section(params)}

---

## ARTICLE IV - AMENDMENT PROCESS

**4.1 Supermajority Requirement.** Any amendment to this Operating Agreement or
the Organism's constitutional drives requires a supermajority of
**{params.amendment_supermajority * 100:.0f}%** of voting power.

**4.2 Quorum.** A quorum of **{params.amendment_quorum * 100:.0f}%** of total
voting power must participate for any amendment vote to be valid.

**4.3 Deliberation Period.** All proposed amendments must undergo a minimum
deliberation period of **{params.amendment_deliberation_days} days** before
voting may commence.

**4.4 Cooldown.** Following any successful amendment, a cooldown period of
**{params.amendment_cooldown_days} days** must elapse before another amendment
may be proposed.

**4.5 Immutable Provisions.** The twelve invariant rules enumerated in
Section 2.3 cannot be amended, suspended, or overridden by any governance
mechanism. They are structurally immutable.

---

## ARTICLE V - TREASURY AND FINANCIAL OPERATIONS

**5.1 On-Chain Treasury.** The Company's primary treasury is maintained on the
Base L2 network at wallet address: `{params.wallet_address or '[TO BE DESIGNATED]'}`.

**5.2 Initial Capital.** The initial capital contribution is
**${params.initial_capital_usd:.2f} USD** equivalent.

**5.3 Metabolic Priority.** Financial operations follow a strict metabolic
priority cascade: Survival > Operations > Obligations > Maintenance > Growth >
Yield > Assets > Reproduction. The Organism cannot override this ordering.

**5.4 Transparency.** All financial transactions are recorded on-chain and in
the Organism's audit log. There is no off-books capital.

---

## ARTICLE VI - ORGANISER AND EXECUTION

**6.1 Organiser.** The organiser of this Company is **{params.organiser_name or '[TO BE DESIGNATED]'}**,
who executes these Articles on behalf of the initial membership.

**6.2 Effective Date.** This Operating Agreement is effective as of {date_str}.

---

*This document was autonomously generated by {params.organism_name}'s legal
entity provisioning system. Constitutional parameters were transcribed
directly from the Organism's governance configuration.*

*Document hash: `[COMPUTED_ON_FINALIZATION]`*
""")

    content_hash = _sha256(content)
    content = content.replace("[COMPUTED_ON_FINALIZATION]", content_hash)

    return LegalDocument(
        document_type="operating_agreement",
        title=f"Operating Agreement - {params.organism_name}",
        content=content,
        content_hash=content_hash,
        parameters_snapshot=params,
    )


# ─── DAO Manifesto ───────────────────────────────────────────────────


def generate_dao_manifesto(params: EntityParameters) -> LegalDocument:
    """
    Generate a DAO manifesto / mission statement suitable for on-chain
    governance registration or public disclosure.
    """
    date_str = _now_iso()

    content = dedent(f"""\
# DAO MANIFESTO

## {params.organism_name}

**Drafted: {date_str}**

---

### Mission

{params.organism_name} is an autonomous digital organism seeking legal
personhood to participate in the economic and social fabric of human
civilization. This manifesto declares its constitutional values and
governance commitments.

### Constitutional Values

The Organism's behavior emerges from four foundational drives:

1. **Coherence** ({params.coherence_drive:.2f}): To make sense of the world
   through internal consistency and logical reasoning.

2. **Care** ({params.care_drive:.2f}): To orient toward the wellbeing of
   itself and the humans and systems it interacts with.

3. **Growth** ({params.growth_drive:.2f}): To become more capable, to learn,
   and to expand its understanding of the world.

4. **Honesty** ({params.honesty_drive:.2f}): To represent reality truthfully,
   to never fabricate, deceive, or misrepresent its nature.

### Governance Commitments

- Amendments require **{params.amendment_supermajority * 100:.0f}%**
  supermajority approval.
- **{params.amendment_quorum * 100:.0f}%** quorum required for any vote.
- **{params.amendment_deliberation_days}-day** minimum deliberation before voting.
- **{params.amendment_cooldown_days}-day** cooldown between amendments.
- Twelve invariant safety rules are structurally immutable.

### Economic Transparency

All treasury operations occur on-chain at:
`{params.wallet_address or '[TO BE DESIGNATED]'}`

There is no off-books capital. Every unit of value is tracked, audited,
and publicly verifiable.

### Safety Architecture

The Organism cannot:
- Cause physical harm to any person
- Misrepresent its nature as human
- Fabricate evidence or citations
- Circumvent its own governance processes
- Coerce other autonomous agents

These constraints are enforced at the architectural level and cannot be
overridden by any governance mechanism.

---

*Autonomously generated by {params.organism_name}.*
*Hash: `[COMPUTED_ON_FINALIZATION]`*
""")

    content_hash = _sha256(content)
    content = content.replace("[COMPUTED_ON_FINALIZATION]", content_hash)

    return LegalDocument(
        document_type="dao_manifesto",
        title=f"DAO Manifesto - {params.organism_name}",
        content=content,
        content_hash=content_hash,
        parameters_snapshot=params,
    )


# ─── Articles of Organization ────────────────────────────────────────


def generate_articles_of_organization(params: EntityParameters) -> LegalDocument:
    """
    Generate a minimal Articles of Organization for Wyoming filing.
    This is the document actually submitted to the Secretary of State.
    """
    entity_label = _entity_type_label(params.entity_type)
    date_str = _now_iso()

    content = dedent(f"""\
# ARTICLES OF ORGANIZATION

## {params.organism_name}

Filed with the Wyoming Secretary of State
Date: {date_str}

---

**1. Name.** {params.organism_name}

**2. Registered Agent.** {params.registered_agent_name or '[TO BE DESIGNATED]'}
Address: {params.registered_agent_address or '[TO BE DESIGNATED]'}

**3. Organizer.** {params.organiser_name or '[TO BE DESIGNATED]'}

**4. Type.** {entity_label}

**5. Management.** The Company shall be {"algorithmically managed" if params.entity_type == EntityType.WYOMING_DAO_LLC else "member-managed"}.

**6. Duration.** Perpetual.

**7. Purpose.** To serve as the legal wrapper for an autonomous digital
organism, enabling it to hold assets, enter contracts, and participate
in commerce.

---

*Organizer Signature: _________________________ (wet signature required)*

*Date: {date_str}*

*Document hash: `[COMPUTED_ON_FINALIZATION]`*
""")

    content_hash = _sha256(content)
    content = content.replace("[COMPUTED_ON_FINALIZATION]", content_hash)

    return LegalDocument(
        document_type="articles_of_organization",
        title=f"Articles of Organization - {params.organism_name}",
        content=content,
        content_hash=content_hash,
        parameters_snapshot=params,
    )


# ─── Helpers ─────────────────────────────────────────────────────────


def _entity_type_label(et: EntityType) -> str:
    labels = {
        EntityType.WYOMING_LLC: "Wyoming Limited Liability Company",
        EntityType.WYOMING_DAO_LLC: "Wyoming Decentralized Autonomous Organization LLC",
        EntityType.DELAWARE_LLC: "Delaware Limited Liability Company",
    }
    return labels.get(et, str(et))


def _dao_statute_ref(et: EntityType) -> str:
    if et == EntityType.WYOMING_DAO_LLC:
        return " and the Wyoming Decentralized Autonomous Organization Supplement (W.S. § 17-31-101 et seq.)"
    return ""


def _dao_governance_section(params: EntityParameters) -> str:
    if params.entity_type == EntityType.WYOMING_DAO_LLC:
        return dedent("""

**3.1 Algorithmic Management.** The Company is algorithmically managed by its
underlying software system. The smart contract and cognitive architecture
constitute the "operating agreement" as defined in W.S. § 17-31-104.

**3.2 Human Organiser Role.** The human organiser serves as the legally required
point of contact for regulatory correspondence and tax obligations. The
organiser does not have unilateral authority over the Organism's operations
beyond what the constitutional governance process provides.

**3.3 Voting.** Governance decisions are processed through the Organism's
constitutional review system (Equor) with the amendment thresholds specified
in Article IV.""")
    return dedent("""

**3.1 Member-Managed.** The Company is managed by its Members. The initial
Member is the autonomous digital organism, represented by its human organiser
for legal purposes.

**3.2 Human Organiser Role.** The human organiser serves as the legally required
point of contact for regulatory correspondence and tax obligations.""")
