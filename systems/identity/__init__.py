"""
EcodiaOS - Identity & Certificate Management (Phase 16g–h)

Phase 16g: Cryptographically signed Certificates of Alignment that govern
participation in the Federation, access to the Knowledge Commons, and
human trust.

Phase 16h: External Identity Layer - encrypted credential vault, platform
connector ABCs for OAuth2 lifecycle, and native TOTP generation for
automated 2FA flows.

Every EOS instance must hold a valid EcodianCertificate - issued by its parent
(birth certificate) or the Genesis Node - to participate in the ecosystem.
"""

from systems.identity.certificate import (
    CertificateStatus,
    CertificateType,
    EcodianCertificate,
    build_certificate,
    compute_lineage_hash,
    sign_certificate,
    verify_certificate_signature,
)
from systems.identity.connector import (
    AuthorizationRequest,
    AuthorizationResponse,
    ConnectorCredentials,
    ConnectorHealthReport,
    ConnectorStatus,
    OAuthClientConfig,
    OAuthGrantType,
    OAuthTokenSet,
    PlatformAuthError,
    PlatformConnector,
    TokenExchangeRequest,
    TokenRefreshResult,
    TokenType,
)
from systems.identity.genome import IdentityGenomeExtractor
from systems.identity.identity import IdentitySystem, compute_constitutional_hash
from systems.identity.manager import CertificateManager
from systems.identity.persona import PersonaEngine, PersonaProfile
from systems.identity.totp import (
    TOTPConfig,
    TOTPGenerator,
    generate_totp,
    verify_totp,
)
from systems.identity.vault import (
    IdentityVault,
    SealedEnvelope,
    VaultConfig,
)

__all__ = [
    # Certificate (Phase 16g)
    "CertificateManager",
    "CertificateStatus",
    "CertificateType",
    "EcodianCertificate",
    "build_certificate",
    "compute_lineage_hash",
    "sign_certificate",
    "verify_certificate_signature",
    # Vault (Phase 16h)
    "IdentityVault",
    "SealedEnvelope",
    "VaultConfig",
    # Connector (Phase 16h)
    "AuthorizationRequest",
    "AuthorizationResponse",
    "ConnectorCredentials",
    "ConnectorHealthReport",
    "ConnectorStatus",
    "OAuthClientConfig",
    "OAuthGrantType",
    "OAuthTokenSet",
    "PlatformAuthError",
    "PlatformConnector",
    "TokenExchangeRequest",
    "TokenRefreshResult",
    "TokenType",
    # Identity System (Spec 23)
    "IdentitySystem",
    "IdentityGenomeExtractor",
    "compute_constitutional_hash",
    # Persona (Spec 23 addendum)
    "PersonaEngine",
    "PersonaProfile",
    # TOTP (Phase 16h)
    "TOTPConfig",
    "TOTPGenerator",
    "generate_totp",
    "verify_totp",
]
