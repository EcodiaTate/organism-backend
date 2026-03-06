"""
EcodiaOS — Social Interface

Thin HTTP clients for social platform publishing.
All clients read credentials from the IdentityVault at runtime —
no plaintext secrets are passed at construction time.

Clients:
    XSocialClient       — Post tweets via X API v2 (OAuth 1.0a user context)
    GitHubSocialClient  — Create Gists and Discussion comments via GitHub REST/GraphQL

Entry point for the ExecuteSocialPostExecutor in axon/executors/social_post.py.
"""

from interfaces.social.github_client import GitHubSocialClient
from interfaces.social.types import (
    DegradedSocialPresence,
    PostResult,
    SocialPlatform,
)
from interfaces.social.x_client import XSocialClient, truncate_for_x

__all__ = [
    "DegradedSocialPresence",
    "GitHubSocialClient",
    "PostResult",
    "SocialPlatform",
    "XSocialClient",
    "truncate_for_x",
]
