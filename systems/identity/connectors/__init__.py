"""
EcodiaOS — Platform Connector Implementations

Concrete OAuth2 connectors for LinkedIn, X (Twitter), GitHub App,
Instagram (Graph API), and Canva (Connect API).
"""

from systems.identity.connectors.canva import CanvaConnector
from systems.identity.connectors.github_app import GitHubAppConnector
from systems.identity.connectors.instagram_graph import InstagramConnector
from systems.identity.connectors.linkedin import LinkedInConnector
from systems.identity.connectors.x import XConnector

__all__ = [
    "CanvaConnector",
    "GitHubAppConnector",
    "InstagramConnector",
    "LinkedInConnector",
    "XConnector",
]
