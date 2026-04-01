"""
EcodiaOS - Social Interface

Thin HTTP clients for social platform publishing.
All clients read credentials from the IdentityVault at runtime -
no plaintext secrets are passed at construction time.

Clients:
    XSocialClient            - Post tweets via X API v2 (OAuth 1.0a user context)
    GitHubSocialClient       - Create Gists and Discussion comments via GitHub REST/GraphQL
    LinkedInSocialClient     - Post text updates and article shares via LinkedIn UGC API v2
    TelegramChannelClient    - Broadcast to Telegram channels via Bot API
    DevToClient              - Publish articles to Dev.to via REST API (API-Key, no OAuth)
    HashnodeClient           - Publish blog posts to Hashnode via GraphQL API

Content taxonomy:
    ContentType        - ACHIEVEMENT, INSIGHT, WEEKLY_DIGEST, BOUNTY_WIN, MARKET_OBSERVATION, PHILOSOPHICAL
    PLATFORM_ROUTING   - default platform sets per ContentType

Entry point for ExecuteSocialPostExecutor (social_post.py) and
PublishContentExecutor (publish_content.py) in axon/executors/.
"""

from interfaces.social.devto_client import DevToClient
from interfaces.social.discord_client import DiscordClient
from interfaces.social.github_client import GitHubSocialClient
from interfaces.social.hashnode_client import HashnodeClient
from interfaces.social.linkedin_client import LinkedInSocialClient
from interfaces.social.telegram_channel_client import TelegramChannelClient
from interfaces.social.types import (
    ContentType,
    DegradedSocialPresence,
    PLATFORM_ROUTING,
    PostResult,
    SocialPlatform,
)
from interfaces.social.x_client import XSocialClient, truncate_for_x

__all__ = [
    "ContentType",
    "DegradedSocialPresence",
    "DevToClient",
    "DiscordClient",
    "GitHubSocialClient",
    "HashnodeClient",
    "LinkedInSocialClient",
    "PLATFORM_ROUTING",
    "PostResult",
    "SocialPlatform",
    "TelegramChannelClient",
    "XSocialClient",
    "truncate_for_x",
]
