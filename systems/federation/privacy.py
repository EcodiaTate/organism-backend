"""
EcodiaOS — Federation Privacy Filter

CRITICAL: Individual private data NEVER crosses federation boundaries
without that individual's explicit consent.

The privacy filter is the last line of defence before knowledge leaves
this instance. It operates on three principles:

1. Items marked PRIVATE are always removed — no exceptions.
2. Items marked COMMUNITY_ONLY require at least COLLEAGUE trust.
3. All items are anonymised: individual identifiers (names, IDs,
   contact information) are stripped or replaced with anonymous tokens.

Even at ALLY trust level, individual data is anonymised. The federation
protocol shares patterns, procedures, and aggregate knowledge — never
individual people's information.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

from primitives.federation import (
    FilteredKnowledge,
    KnowledgeItem,
    PrivacyLevel,
    TrustLevel,
)

logger = structlog.get_logger("systems.federation.privacy")

# Patterns for identifying personal information
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")
_PERSON_NAME_MARKERS = {"person", "individual", "member", "user", "participant"}


class PrivacyFilter:
    """
    Filters knowledge for safe federation sharing.

    Every piece of knowledge passes through this filter before
    crossing the federation boundary. The filter:

    1. Removes items above the trust level's access threshold
    2. Removes items marked as private
    3. Strips individual identifiers from remaining items
    4. Tracks statistics for observability

    The privacy filter is deliberately conservative — it is better to
    over-filter than to leak private data.
    """

    def __init__(self) -> None:
        self._total_filtered: int = 0
        self._total_removed: int = 0
        self._total_anonymised: int = 0
        self._logger = logger.bind(component="privacy_filter")

    async def filter(
        self,
        items: list[KnowledgeItem],
        trust_level: TrustLevel,
    ) -> FilteredKnowledge:
        """
        Apply privacy filtering to a list of knowledge items.

        Returns a FilteredKnowledge containing only items that are
        safe to share at the given trust level, with all individual
        identifiers removed.
        """
        result = FilteredKnowledge()

        for item in items:
            # Rule 1: Private items NEVER cross boundaries
            if item.privacy_level == PrivacyLevel.PRIVATE:
                result.items_removed_by_privacy += 1
                continue

            # Rule 2: Community-only items require COLLEAGUE+
            if item.privacy_level == PrivacyLevel.COMMUNITY_ONLY:
                if trust_level.value < TrustLevel.COLLEAGUE.value:
                    result.items_removed_by_privacy += 1
                    continue

            # Rule 3: Anonymise all remaining items
            anonymised_item = self._anonymise(item)
            if anonymised_item is not item:
                result.items_anonymised += 1

            result.items.append(anonymised_item)

        self._total_filtered += len(items)
        self._total_removed += result.items_removed_by_privacy
        self._total_anonymised += result.items_anonymised

        self._logger.debug(
            "privacy_filter_applied",
            input_count=len(items),
            output_count=len(result.items),
            removed=result.items_removed_by_privacy,
            anonymised=result.items_anonymised,
            trust_level=trust_level.name,
        )

        return result

    # ─── Anonymisation ──────────────────────────────────────────────

    def _anonymise(self, item: KnowledgeItem) -> KnowledgeItem:
        """
        Remove individual identifiers from a knowledge item.

        Operates on the content dict, stripping:
          - Email addresses
          - Phone numbers
          - Fields that look like personal names
          - Fields named 'user_id', 'person_id', 'member_id', etc.
        """
        content = item.content
        if not content:
            return item

        anonymised_content = _deep_anonymise(content)

        if anonymised_content is content:
            return item  # No changes needed

        return KnowledgeItem(
            item_id=item.item_id,
            knowledge_type=item.knowledge_type,
            privacy_level=item.privacy_level,
            content=anonymised_content,
            embedding=item.embedding,
            source_instance_id=item.source_instance_id,
            created_at=item.created_at,
        )

    # ─── Stats ──────────────────────────────────────────────────────

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_filtered": self._total_filtered,
            "total_removed": self._total_removed,
            "total_anonymised": self._total_anonymised,
        }


# ─── Deep Anonymisation ─────────────────────────────────────────

# Keys that likely contain personal identifiers
_PII_KEYS = {
    "name", "full_name", "first_name", "last_name", "user_name", "username",
    "email", "email_address", "phone", "phone_number", "address",
    "user_id", "person_id", "member_id", "participant_id", "individual_id",
    "ssn", "social_security", "date_of_birth", "dob", "birthday",
    "contact", "contact_info",
}


def _deep_anonymise(obj: Any, _depth: int = 0) -> Any:
    """
    Recursively anonymise a nested dict/list structure.

    Strips PII keys, redacts email/phone patterns in string values,
    and recurses into nested structures up to a reasonable depth.
    """
    if _depth > 10:
        return obj

    if isinstance(obj, dict):
        result = {}
        changed = False
        for key, value in obj.items():
            key_lower = key.lower().replace("-", "_")

            # Remove known PII keys entirely
            if key_lower in _PII_KEYS:
                result[key] = "[REDACTED]"
                changed = True
                continue

            # Check for entity type markers that suggest person data
            if key_lower == "type" and isinstance(value, str) and value.lower() in _PERSON_NAME_MARKERS:
                result[key] = "anonymous_entity"
                changed = True
                continue

            # Recurse into nested values
            new_value = _deep_anonymise(value, _depth + 1)
            if new_value is not value:
                changed = True
            result[key] = new_value

        return result if changed else obj

    if isinstance(obj, list):
        new_list = [_deep_anonymise(item, _depth + 1) for item in obj]
        if any(new is not old for new, old in zip(new_list, obj, strict=False)):
            return new_list
        return obj

    if isinstance(obj, str):
        return _anonymise_string(obj)

    return obj


def _anonymise_string(text: str) -> str:
    """Redact email addresses and phone numbers from a string."""
    result = text
    changed = False

    if _EMAIL_PATTERN.search(result):
        result = _EMAIL_PATTERN.sub("[email-redacted]", result)
        changed = True

    if _PHONE_PATTERN.search(result):
        result = _PHONE_PATTERN.sub("[phone-redacted]", result)
        changed = True

    return result if changed else text
