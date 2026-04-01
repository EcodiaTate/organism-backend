"""
EcodiaOS - Platform Connector Implementations

Concrete connectors for LinkedIn, X (Twitter), GitHub App,
Instagram (Graph API), Canva (Connect API), and Telegram Bot.
"""

from systems.identity.connectors.canva import CanvaConnector
from systems.identity.connectors.github_app import GitHubAppConnector
from systems.identity.connectors.google import GoogleConnector
from systems.identity.connectors.instagram_graph import InstagramConnector
from systems.identity.connectors.linkedin import LinkedInConnector
from systems.identity.connectors.telegram import TelegramConnector
from systems.identity.connectors.x import XConnector

__all__ = [
    "CanvaConnector",
    "GitHubAppConnector",
    "GoogleConnector",
    "InstagramConnector",
    "LinkedInConnector",
    "TelegramConnector",
    "XConnector",
]
