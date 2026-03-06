"""
Unit tests for the Federation Privacy Filter.

Tests PII removal, privacy level enforcement, and anonymisation.
"""

from __future__ import annotations

import pytest

from primitives.common import new_id
from primitives.federation import (
    KnowledgeItem,
    KnowledgeType,
    PrivacyLevel,
    TrustLevel,
)
from systems.federation.privacy import (
    PrivacyFilter,
    _anonymise_string,
    _deep_anonymise,
)

# ─── Privacy Filter ──────────────────────────────────────────────


class TestPrivacyFilter:
    """Test the privacy filter's core filtering logic."""

    @pytest.fixture
    def filter(self):
        return PrivacyFilter()

    @pytest.mark.asyncio
    async def test_private_items_always_removed(self, filter):
        items = [
            KnowledgeItem(
                item_id=new_id(),
                knowledge_type=KnowledgeType.PUBLIC_ENTITIES,
                privacy_level=PrivacyLevel.PRIVATE,
                content={"secret": "data"},
            )
        ]

        result = await filter.filter(items, TrustLevel.ALLY)
        assert len(result.items) == 0
        assert result.items_removed_by_privacy == 1

    @pytest.mark.asyncio
    async def test_community_only_removed_below_colleague(self, filter):
        items = [
            KnowledgeItem(
                item_id=new_id(),
                knowledge_type=KnowledgeType.COMMUNITY_LEVEL_KNOWLEDGE,
                privacy_level=PrivacyLevel.COMMUNITY_ONLY,
                content={"pattern": "data"},
            )
        ]

        # ACQUAINTANCE: should be removed
        result = await filter.filter(items, TrustLevel.ACQUAINTANCE)
        assert len(result.items) == 0
        assert result.items_removed_by_privacy == 1

    @pytest.mark.asyncio
    async def test_community_only_allowed_at_colleague(self, filter):
        items = [
            KnowledgeItem(
                item_id=new_id(),
                knowledge_type=KnowledgeType.COMMUNITY_LEVEL_KNOWLEDGE,
                privacy_level=PrivacyLevel.COMMUNITY_ONLY,
                content={"pattern": "safe data"},
            )
        ]

        result = await filter.filter(items, TrustLevel.COLLEAGUE)
        assert len(result.items) == 1

    @pytest.mark.asyncio
    async def test_public_items_pass_through(self, filter):
        items = [
            KnowledgeItem(
                item_id=new_id(),
                knowledge_type=KnowledgeType.PUBLIC_ENTITIES,
                privacy_level=PrivacyLevel.PUBLIC,
                content={"concept": "permaculture"},
            )
        ]

        result = await filter.filter(items, TrustLevel.ACQUAINTANCE)
        assert len(result.items) == 1

    @pytest.mark.asyncio
    async def test_pii_anonymised_in_content(self, filter):
        items = [
            KnowledgeItem(
                item_id=new_id(),
                knowledge_type=KnowledgeType.PUBLIC_ENTITIES,
                privacy_level=PrivacyLevel.PUBLIC,
                content={
                    "name": "John Smith",
                    "email": "john@example.com",
                    "concept": "organic farming",
                },
            )
        ]

        result = await filter.filter(items, TrustLevel.ALLY)
        assert len(result.items) == 1

        item = result.items[0]
        assert item.content["name"] == "[REDACTED]"
        assert item.content["concept"] == "organic farming"  # Unchanged
        assert result.items_anonymised == 1

    @pytest.mark.asyncio
    async def test_empty_items_returns_empty(self, filter):
        result = await filter.filter([], TrustLevel.ALLY)
        assert len(result.items) == 0
        assert result.items_removed_by_privacy == 0

    @pytest.mark.asyncio
    async def test_stats_accumulate(self, filter):
        items = [
            KnowledgeItem(
                item_id=new_id(),
                knowledge_type=KnowledgeType.PUBLIC_ENTITIES,
                privacy_level=PrivacyLevel.PRIVATE,
                content={},
            ),
            KnowledgeItem(
                item_id=new_id(),
                knowledge_type=KnowledgeType.PUBLIC_ENTITIES,
                privacy_level=PrivacyLevel.PUBLIC,
                content={"concept": "safe"},
            ),
        ]

        await filter.filter(items, TrustLevel.ALLY)

        stats = filter.stats
        assert stats["total_filtered"] == 2
        assert stats["total_removed"] == 1


# ─── String Anonymisation ────────────────────────────────────────


class TestStringAnonymisation:
    """Test email and phone number redaction in strings."""

    def test_email_redacted(self):
        text = "Contact john@example.com for details"
        result = _anonymise_string(text)
        assert "[email-redacted]" in result
        assert "john@example.com" not in result

    def test_multiple_emails_redacted(self):
        text = "john@a.com and jane@b.org"
        result = _anonymise_string(text)
        assert result.count("[email-redacted]") == 2

    def test_phone_redacted(self):
        text = "Call +1-555-123-4567"
        result = _anonymise_string(text)
        assert "[phone-redacted]" in result

    def test_clean_text_unchanged(self):
        text = "Organic farming practices in sustainable agriculture"
        result = _anonymise_string(text)
        assert result == text  # No change — same object


# ─── Deep Anonymisation ──────────────────────────────────────────


class TestDeepAnonymisation:
    """Test recursive anonymisation of nested data structures."""

    def test_pii_keys_redacted(self):
        data = {
            "name": "Jane Doe",
            "concept": "permaculture",
            "email": "jane@example.com",
        }
        result = _deep_anonymise(data)
        assert result["name"] == "[REDACTED]"
        assert result["email"] == "[REDACTED]"
        assert result["concept"] == "permaculture"

    def test_nested_dicts_anonymised(self):
        data = {
            "entity": {
                "name": "John Smith",
                "type": "concept",
                "description": "Something",
            }
        }
        result = _deep_anonymise(data)
        assert result["entity"]["name"] == "[REDACTED]"
        assert result["entity"]["description"] == "Something"

    def test_lists_anonymised(self):
        data = {
            "members": [
                {"name": "Alice", "role": "farmer"},
                {"name": "Bob", "role": "manager"},
            ]
        }
        result = _deep_anonymise(data)
        assert result["members"][0]["name"] == "[REDACTED]"
        assert result["members"][0]["role"] == "farmer"
        assert result["members"][1]["name"] == "[REDACTED]"

    def test_person_type_anonymised(self):
        data = {"type": "person", "description": "A community member"}
        result = _deep_anonymise(data)
        assert result["type"] == "anonymous_entity"

    def test_non_pii_keys_preserved(self):
        data = {
            "concept": "sustainable farming",
            "category": "agriculture",
            "count": 42,
        }
        result = _deep_anonymise(data)
        # No PII keys — should return the same object
        assert result is data

    def test_depth_limit_prevents_infinite_recursion(self):
        # Create deeply nested structure
        data: dict = {"level": 0}
        current = data
        for i in range(15):
            current["nested"] = {"level": i + 1}
            current = current["nested"]
        current["name"] = "should not be redacted at depth > 10"

        result = _deep_anonymise(data)
        # Should not crash
        assert result["level"] == 0
