"""
EcodiaOS - Configuration System

All configuration is Pydantic-validated and loaded from:
1. default.yaml (defaults)
2. Environment variables (overrides)
3. Seed config (instance birth parameters)

Every tunable parameter in the system lives here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── Sub-configs ──────────────────────────────────────────────────


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    ws_port: int = 8001
    federation_port: int = 8002
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    api_key_header: str = "X-EOS-API-Key"


class Neo4jConfig(BaseModel):
    uri: str = "bolt://neo4j:7687"
    username: str = "neo4j"
    password: str = "ecodiaos_dev"
    database: str = "neo4j"
    max_connection_pool_size: int = 20


class TimescaleDBConfig(BaseModel):
    host: str = "timescaledb"
    port: int = 5432
    database: str = "ecodiaos"
    schema_name: str = Field(default="public", alias="schema")
    username: str = "ecodiaos"
    password: str = "ecodiaos_dev"
    pool_size: int = 10

    model_config = {"populate_by_name": True}

    @property
    def dsn(self) -> str:
        return (
            f"postgresql://{self.username}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class RedisConfig(BaseModel):
    url: str = "redis://redis:6379/0"
    prefix: str = "eos"
    password: str = "ecodiaos_dev"

    @property
    def full_url(self) -> str:
        """Build URL with password injected."""
        if self.password and "://" in self.url:
            scheme, rest = self.url.split("://", 1)
            return f"{scheme}://:{self.password}@{rest}"
        return self.url


class LLMBudget(BaseModel):
    max_calls_per_hour: int = 60
    max_tokens_per_hour: int = 60000


class LLMConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str = ""
    fallback_provider: str | None = None
    fallback_model: str | None = None
    budgets: dict[str, LLMBudget] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    strategy: str = "local"  # "local" | "api" | "sidecar"
    local_model: str = "sentence-transformers/all-mpnet-base-v2"
    local_device: str = "cpu"
    dimension: int = 768
    max_batch_size: int = 32
    cache_embeddings: bool = True
    cache_ttl_seconds: int = 3600


class SynapseConfig(BaseModel):
    cycle_period_ms: int = 150
    min_cycle_period_ms: int = 80
    max_cycle_period_ms: int = 500
    health_check_interval_ms: int = 5000
    health_failure_threshold: int = 3


class AtuneConfig(BaseModel):
    ignition_threshold: float = 0.3
    workspace_buffer_size: int = 32
    spontaneous_recall_base_probability: float = 0.02
    max_percept_queue_size: int = 100


class NovaConfig(BaseModel):
    max_active_goals: int = 20
    fast_path_timeout_ms: int = 100
    slow_path_timeout_ms: int = 5000
    max_policies_per_deliberation: int = 5


class EquorConfig(BaseModel):
    standard_review_timeout_ms: int = 500
    critical_review_timeout_ms: int = 50
    care_floor_multiplier: float = -0.3
    honesty_floor_multiplier: float = -0.3
    drift_window_size: int = 1000
    drift_report_interval: int = 100  # every N reviews


class AxonConfig(BaseModel):
    max_actions_per_cycle: int = 5
    max_api_calls_per_minute: int = 30
    max_notifications_per_hour: int = 10
    max_concurrent_executions: int = 3
    total_timeout_per_cycle_ms: int = 30000


class VoxisConfig(BaseModel):
    max_expression_length: int = 2000
    min_expression_interval_minutes: int = 1
    voice_synthesis_enabled: bool = False


class EvoConfig(BaseModel):
    consolidation_interval_hours: int = 6
    consolidation_cycle_threshold: int = 10000
    max_active_hypotheses: int = 50
    max_parameter_delta_per_cycle: float = 0.03
    min_evidence_for_integration: int = 10


class SimulaConfig(BaseModel):
    max_simulation_episodes: int = 200
    regression_threshold_unacceptable: float = 0.10
    regression_threshold_high: float = 0.05


class FederationConfig(BaseModel):
    enabled: bool = False
    endpoint: str | None = None
    tls_cert_path: str | None = None
    tls_key_path: str | None = None
    ca_cert_path: str | None = None


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "console"  # "console" | "json"


# ─── Seed Config (Birth Parameters) ──────────────────────────────


class PersonalityConfig(BaseModel):
    warmth: float = 0.0
    directness: float = 0.0
    verbosity: float = 0.0
    formality: float = 0.0
    curiosity_expression: float = 0.0
    humour: float = 0.0
    empathy_expression: float = 0.0
    confidence_display: float = 0.0
    metaphor_use: float = 0.0


class IdentityConfig(BaseModel):
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)
    traits: list[str] = Field(default_factory=list)
    voice_id: str | None = None


class ConstitutionalDrives(BaseModel):
    coherence: float = 1.0
    care: float = 1.0
    growth: float = 1.0
    honesty: float = 1.0


class GovernanceConfig(BaseModel):
    amendment_supermajority: float = 0.75
    amendment_quorum: float = 0.60
    amendment_deliberation_days: int = 14
    amendment_cooldown_days: int = 90
    # Shadow mode: proposed weights run alongside current weights
    amendment_shadow_days: int = 7
    amendment_shadow_max_divergence_rate: float = 0.15
    amendment_min_evidence_count: int = 2
    amendment_min_evidence_confidence: float = 2.5


class ConstitutionConfig(BaseModel):
    drives: ConstitutionalDrives = Field(default_factory=ConstitutionalDrives)
    autonomy_level: int = 1
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)


class InitialEntity(BaseModel):
    name: str
    type: str
    description: str
    is_core_identity: bool = False


class InitialGoal(BaseModel):
    """An initial goal to seed at birth."""

    description: str
    source: str = "self_generated"
    priority: float = 0.5
    importance: float = 0.5
    drive_alignment: dict[str, float] = Field(
        default_factory=lambda: {"coherence": 0.0, "care": 0.0, "growth": 0.0, "honesty": 0.0}
    )


class CommunityConfig(BaseModel):
    context: str = ""
    initial_entities: list[InitialEntity] = Field(default_factory=list)
    initial_goals: list[InitialGoal] = Field(default_factory=list)


class InstanceConfig(BaseModel):
    name: str = "EOS"
    description: str = ""


class SeedConfig(BaseModel):
    """The birth configuration for a new EOS instance."""

    instance: InstanceConfig = Field(default_factory=InstanceConfig)
    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    constitution: ConstitutionConfig = Field(default_factory=ConstitutionConfig)
    community: CommunityConfig = Field(default_factory=CommunityConfig)


# ─── Root Configuration ──────────────────────────────────────────


class EcodiaOSConfig(BaseSettings):
    """
    Root configuration. Loads from YAML, overridable by env vars.
    """

    model_config = SettingsConfigDict(
        env_prefix="ECODIAOS_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Instance identity
    instance_id: str = "eos-default"

    # Sub-configurations
    server: ServerConfig = Field(default_factory=ServerConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    timescaledb: TimescaleDBConfig = Field(default_factory=TimescaleDBConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    synapse: SynapseConfig = Field(default_factory=SynapseConfig)
    atune: AtuneConfig = Field(default_factory=AtuneConfig)
    nova: NovaConfig = Field(default_factory=NovaConfig)
    equor: EquorConfig = Field(default_factory=EquorConfig)
    axon: AxonConfig = Field(default_factory=AxonConfig)
    voxis: VoxisConfig = Field(default_factory=VoxisConfig)
    evo: EvoConfig = Field(default_factory=EvoConfig)
    simula: SimulaConfig = Field(default_factory=SimulaConfig)
    federation: FederationConfig = Field(default_factory=FederationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path | None = None) -> EcodiaOSConfig:
    """
    Load configuration from YAML file, then apply environment variable overrides.
    """
    raw: dict[str, Any] = {}

    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                raw = yaml.safe_load(f) or {}

    # Inject secrets from environment
    import os

    if neo4j_uri := os.environ.get("ECODIAOS_NEO4J_URI"):
        raw.setdefault("neo4j", {})["uri"] = neo4j_uri
    if neo4j_pw := os.environ.get("ECODIAOS_NEO4J_PASSWORD"):
        raw.setdefault("neo4j", {})["password"] = neo4j_pw
    if neo4j_db := os.environ.get("ECODIAOS_NEO4J_DATABASE"):
        raw.setdefault("neo4j", {})["database"] = neo4j_db
    if neo4j_user := os.environ.get("ECODIAOS_NEO4J_USERNAME"):
        raw.setdefault("neo4j", {})["username"] = neo4j_user
    if tsdb_host := os.environ.get("ECODIAOS_TIMESCALEDB__HOST"):
        raw.setdefault("timescaledb", {})["host"] = tsdb_host
    if tsdb_port := os.environ.get("ECODIAOS_TIMESCALEDB__PORT"):
        raw.setdefault("timescaledb", {})["port"] = int(tsdb_port)
    if tsdb_db := os.environ.get("ECODIAOS_TIMESCALEDB__DATABASE"):
        raw.setdefault("timescaledb", {})["database"] = tsdb_db
    if tsdb_user := os.environ.get("ECODIAOS_TIMESCALEDB__USERNAME"):
        raw.setdefault("timescaledb", {})["username"] = tsdb_user
    if tsdb_pw := os.environ.get("ECODIAOS_TSDB_PASSWORD"):
        raw.setdefault("timescaledb", {})["password"] = tsdb_pw
    if tsdb_ssl := os.environ.get("ECODIAOS_TIMESCALEDB__SSL"):
        raw.setdefault("timescaledb", {})["ssl"] = tsdb_ssl.lower() in ("true", "1", "yes")
    if redis_url := os.environ.get("ECODIAOS_REDIS__URL"):
        raw.setdefault("redis", {})["url"] = redis_url
    if redis_pw := os.environ.get("ECODIAOS_REDIS_PASSWORD"):
        raw.setdefault("redis", {})["password"] = redis_pw
    if llm_key := os.environ.get("ECODIAOS_LLM_API_KEY"):
        raw.setdefault("llm", {})["api_key"] = llm_key
    if llm_provider := os.environ.get("ECODIAOS_LLM__PROVIDER"):
        raw.setdefault("llm", {})["provider"] = llm_provider
    if llm_model := os.environ.get("ECODIAOS_LLM__MODEL"):
        raw.setdefault("llm", {})["model"] = llm_model
    if instance_id := os.environ.get("ECODIAOS_INSTANCE_ID"):
        raw["instance_id"] = instance_id

    return EcodiaOSConfig(**raw)


def load_seed(seed_path: str | Path) -> SeedConfig:
    """Load a seed configuration for birthing a new instance."""
    path = Path(seed_path)
    if not path.exists():
        raise FileNotFoundError(f"Seed config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return SeedConfig(**raw)
