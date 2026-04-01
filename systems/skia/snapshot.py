"""
EcodiaOS - Skia State Snapshot Pipeline

Exports critical Neo4j graph state, encrypts with IdentityVault,
and pins to IPFS via Pinata.

Also provides ``restore_from_ipfs()`` for the startup restoration path.

Pipeline:
  1. Cypher queries per label → list[dict]
  2. Edge export between exported nodes
  3. Serialize with orjson → compress with gzip
  4. Encrypt with IdentityVault (Fernet AES-128 + HMAC-SHA256)
  5. Upload encrypted blob to Pinata IPFS
  6. Store SnapshotManifest in Redis
  7. Prune old pins beyond retention limit
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import time
from typing import TYPE_CHECKING, Any

import orjson
import structlog
from neo4j.time import DateTime

from systems.skia.types import SnapshotManifest, SnapshotPayload

if TYPE_CHECKING:
    from clients.neo4j import Neo4jClient
    from clients.redis import RedisClient
    from config import SkiaConfig
    from systems.identity.vault import IdentityVault
    from systems.skia.pinata_client import PinataClient
    from systems.memory.service import MemoryService

logger = structlog.get_logger("systems.skia.snapshot")

_SNAPSHOT_NAME_PREFIX = "eos-skia-snapshot"


def _serialize_neo4j_types(obj: Any) -> Any:
    """Convert Neo4j types to JSON-serializable Python types."""
    if isinstance(obj, DateTime):
        return obj.iso_format()
    if isinstance(obj, dict):
        return {k: _serialize_neo4j_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_neo4j_types(item) for item in obj]
    return obj


class StateSnapshotPipeline:
    """
    Exports Neo4j graph state, encrypts, and pins to IPFS.

    Lifecycle:
        pipeline = StateSnapshotPipeline(neo4j, vault, pinata, redis, config, instance_id)
        await pipeline.start()        # begin periodic snapshots
        manifest = await pipeline.take_snapshot()  # manual trigger
        await pipeline.stop()
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        vault: IdentityVault,
        pinata: PinataClient,
        redis: RedisClient,
        config: SkiaConfig,
        instance_id: str,
        memory: MemoryService | None = None,
    ) -> None:
        self._neo4j = neo4j
        self._vault = vault
        self._pinata = pinata
        self._redis = redis
        self._config = config
        self._instance_id = instance_id
        self._memory: MemoryService | None = memory
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._log = logger.bind(component="skia.snapshot")
        self._event_bus: Any = None  # EventBus | None

        # Stats
        self._total_snapshots: int = 0
        self._last_cid: str = ""
        self._last_constitutional_genome: dict[str, Any] | None = None

    def set_memory(self, memory: MemoryService) -> None:
        """Wire the Memory service so snapshots include the constitutional genome."""
        self._memory = memory

    def set_event_bus(self, event_bus: Any) -> None:
        """Wire Synapse event bus so completed snapshots are broadcast."""
        self._event_bus = event_bus

    @property
    def total_snapshots(self) -> int:
        return self._total_snapshots

    @property
    def last_cid(self) -> str:
        return self._last_cid

    @property
    def last_constitutional_genome(self) -> dict[str, Any] | None:
        """The constitutional genome bundled in the most recent snapshot."""
        return self._last_constitutional_genome

    async def start(self) -> None:
        """Start the periodic snapshot loop."""
        self._running = True
        self._task = asyncio.create_task(
            self._snapshot_loop(), name="skia_snapshot_loop"
        )
        self._log.info("snapshot_pipeline_started", interval_s=self._config.snapshot_interval_s)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._log.info("snapshot_pipeline_stopped")

    async def _snapshot_loop(self) -> None:
        """Periodic snapshot on configurable interval."""
        while self._running:
            try:
                await self.take_snapshot()
            except Exception as exc:
                self._log.error("snapshot_failed", error=str(exc), exc_info=True)
            await asyncio.sleep(self._config.snapshot_interval_s)

    # ── Main pipeline ─────────────────────────────────────────────

    async def take_snapshot(self) -> SnapshotManifest:
        """
        Execute a full snapshot pipeline.

        Returns the SnapshotManifest with IPFS CID and metadata.
        """
        t0 = time.monotonic()
        self._log.info("snapshot_starting")

        # 1. Export nodes
        nodes = await self._export_nodes()
        node_ids = {n["id"] for n in nodes}

        # 2. Export edges between exported nodes
        edges: list[dict[str, Any]] = []
        if self._config.snapshot_include_edges and node_ids:
            edges = await self._export_edges(node_ids)

        # 2b. Export constitutional genome from Memory so the organism's
        #     phenotype survives instance death and can be inherited by any
        #     shadow instance provisioned from this snapshot.
        constitutional_genome: dict[str, Any] | None = None
        if self._memory is not None:
            try:
                constitutional_genome = await self._memory.export_genome()
                self._log.info(
                    "constitutional_genome_exported",
                    genome_keys=list(constitutional_genome.keys()) if constitutional_genome else [],
                )
            except Exception as genome_exc:
                # Non-fatal: snapshot is still valuable without the genome.
                self._log.warning(
                    "constitutional_genome_export_failed",
                    error=str(genome_exc),
                )

        # 3. Serialize + compress
        payload = SnapshotPayload(
            instance_id=self._instance_id,
            nodes=nodes,
            edges=edges,
            constitutional_genome=constitutional_genome,
        )
        raw_bytes = orjson.dumps(payload.model_dump(mode="json"))
        uncompressed_size = len(raw_bytes)

        if self._config.snapshot_compress:
            compressed = gzip.compress(raw_bytes, compresslevel=6)
        else:
            compressed = raw_bytes
        compressed_size = len(compressed)

        # 4. Encrypt with IdentityVault
        envelope = self._vault.encrypt(
            plaintext=compressed,
            platform_id="skia",
            purpose="state_snapshot",
        )
        encrypted_bytes = envelope.ciphertext.encode("ascii")
        encrypted_size = len(encrypted_bytes)

        # Check size limit
        if encrypted_size > self._config.pinata_max_pin_size_bytes:
            raise ValueError(
                f"Snapshot too large ({encrypted_size} bytes) for Pinata "
                f"limit ({self._config.pinata_max_pin_size_bytes} bytes)"
            )

        # 5. Pin to IPFS
        snapshot_name = f"{_SNAPSHOT_NAME_PREFIX}-{self._instance_id}-{int(time.time())}"
        cid, pin_id = await self._pinata.pin_bytes(
            data=encrypted_bytes,
            name=snapshot_name,
            group_name=self._config.pinata_group_name,
        )

        # 5b. Integrity verification - download and size-check before promoting CID.
        # Pinata does not validate gzip/encryption on ingest; a truncated upload
        # produces a valid CID for corrupt content. We verify the round-trip
        # returns at least as many bytes as we uploaded.
        try:
            fetched = await self._pinata.get_by_cid(cid)
            if len(fetched) < encrypted_size:
                raise ValueError(
                    f"IPFS integrity check failed: uploaded {encrypted_size} bytes "
                    f"but got back {len(fetched)} bytes for CID {cid}. "
                    f"Refusing to promote this CID."
                )
        except Exception as integrity_exc:
            # Unpin the bad upload so it cannot be mistakenly promoted later.
            with contextlib.suppress(Exception):
                await self._pinata.unpin(cid)
            raise RuntimeError(
                f"Snapshot integrity verification failed: {integrity_exc}"
            ) from integrity_exc

        # 6. Build manifest
        duration_ms = (time.monotonic() - t0) * 1000
        manifest = SnapshotManifest(
            ipfs_cid=cid,
            instance_id=self._instance_id,
            node_count=len(nodes),
            edge_count=len(edges),
            uncompressed_size_bytes=uncompressed_size,
            compressed_size_bytes=compressed_size,
            encrypted_size_bytes=encrypted_size,
            encryption_key_version=envelope.key_version,
            snapshot_duration_ms=duration_ms,
            pinata_pin_id=pin_id,
        )

        # 7. Stage → promote: write manifest to a staging key first, then atomically
        # promote the CID. If Redis dies between the two writes only the staging key
        # is stale - the live CID key is unchanged.
        # Also append to a sorted-set history (score = unix timestamp) so restoration
        # can fall back to the previous good CID if the latest is corrupt.
        staging_key = f"{self._config.manifest_redis_key}:staging"
        await self._redis.set_json(staging_key, manifest.model_dump(mode="json"))

        # Promote: overwrite live keys only after staging write succeeded
        await self._redis.set_json(
            self._config.manifest_redis_key,
            manifest.model_dump(mode="json"),
        )
        await self._redis.set_json(self._config.state_cid_redis_key, cid)

        # Append CID to rolling history (score = timestamp for ordered retrieval)
        raw = self._redis.client
        snapshot_ts = time.time()
        await raw.zadd(
            f"{self._config.state_cid_redis_key}:history",
            {cid: snapshot_ts},
        )
        # Trim history to retain only the last N entries
        await raw.zremrangebyrank(
            f"{self._config.state_cid_redis_key}:history",
            0,
            -(self._config.pinata_max_retained_pins + 1),
        )

        # 8. Prune old pins
        await self._prune_old_pins()

        # Update stats
        self._total_snapshots += 1
        self._last_cid = cid
        self._last_constitutional_genome = constitutional_genome

        self._log.info(
            "snapshot_completed",
            cid=cid,
            nodes=len(nodes),
            edges=len(edges),
            raw_kb=uncompressed_size // 1024,
            encrypted_kb=encrypted_size // 1024,
            duration_ms=round(duration_ms, 1),
        )

        # Broadcast snapshot completion on Synapse so observers (Benchmarks,
        # Thymos, the observatory tracer) can measure snapshot cadence.
        if self._event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                await self._event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.SKIA_SNAPSHOT_COMPLETED,
                    source_system="skia",
                    data={
                        "instance_id": self._instance_id,
                        "cid": cid,
                        "node_count": len(nodes),
                        "edge_count": len(edges),
                        "encrypted_size_bytes": encrypted_size,
                        "duration_ms": round(duration_ms, 1),
                        "snapshot_number": self._total_snapshots,
                    },
                ))
            except Exception as _emit_exc:
                self._log.debug("snapshot_event_emit_failed", error=str(_emit_exc))

        return manifest

    # ── Neo4j export ──────────────────────────────────────────────

    async def _export_nodes(self) -> list[dict[str, Any]]:
        """Export nodes for configured labels from Neo4j."""
        all_nodes: list[dict[str, Any]] = []
        remaining = self._config.snapshot_max_nodes

        for label in self._config.snapshot_node_labels:
            if remaining <= 0:
                break

            query = (
                f"MATCH (n:`{label}`) "
                f"RETURN labels(n) AS labels, properties(n) AS props, "
                f"elementId(n) AS id "
                f"LIMIT $limit"
            )
            rows = await self._neo4j.execute_read(
                query, {"limit": remaining}
            )
            for row in rows:
                all_nodes.append({
                    "id": row["id"],
                    "labels": row["labels"],
                    "props": _serialize_neo4j_types(row["props"]),
                })
            remaining -= len(rows)

        self._log.debug("nodes_exported", count=len(all_nodes))
        return all_nodes

    async def _export_edges(self, node_ids: set[str]) -> list[dict[str, Any]]:
        """Export relationships between the exported nodes."""
        if not node_ids:
            return []

        # Neo4j elementId() returns strings; pass as list for IN clause
        id_list = list(node_ids)
        query = (
            "MATCH (a)-[r]->(b) "
            "WHERE elementId(a) IN $ids AND elementId(b) IN $ids "
            "RETURN type(r) AS type, properties(r) AS props, "
            "elementId(a) AS source, elementId(b) AS target"
        )
        rows = await self._neo4j.execute_read(query, {"ids": id_list})
        edges = [
            {
                "type": row["type"],
                "props": _serialize_neo4j_types(row["props"]),
                "source": row["source"],
                "target": row["target"],
            }
            for row in rows
        ]
        self._log.debug("edges_exported", count=len(edges))
        return edges

    # ── Pin pruning ───────────────────────────────────────────────

    async def _prune_old_pins(self) -> None:
        """Unpin IPFS entries beyond the retention limit."""
        try:
            pins = await self._pinata.list_pins(
                name_contains=_SNAPSHOT_NAME_PREFIX,
                limit=self._config.pinata_max_retained_pins + 10,
                sort_order="DESC",
            )
            if len(pins) > self._config.pinata_max_retained_pins:
                to_remove = pins[self._config.pinata_max_retained_pins:]
                for pin in to_remove:
                    cid = pin.get("ipfs_pin_hash", "")
                    if cid:
                        await self._pinata.unpin(cid)
                        self._log.info("pin_pruned", cid=cid)
        except Exception as exc:
            self._log.warning("pin_prune_failed", error=str(exc))


# ── Standalone restoration function ────────────────────────────────────


async def restore_from_ipfs(
    cid: str,
    neo4j: Neo4jClient,
    vault_passphrase: str,
    pinata_jwt: str,
    pinata_api_url: str = "https://api.pinata.cloud",
    pinata_gateway_url: str = "https://gateway.pinata.cloud",
    redis_client: RedisClient | None = None,
    instance_id: str = "",
    event_bus: Any = None,
    memory: Any = None,
) -> dict[str, Any] | None:
    """
    Download an encrypted snapshot from IPFS and restore into Neo4j.

    Called during startup when ECODIAOS_SKIA_RESTORE_CID is set.

    Steps:
      1. Download encrypted blob from IPFS gateway
      2. Decrypt with IdentityVault (key_version read from manifest if available)
      3. Decompress gzip
      4. Import nodes and edges into Neo4j
      5. Write restoration_complete flag to Redis so the health endpoint
         can signal to the parent's _verify_handoff() that the graph is
         fully populated (not just that the HTTP server is up).
      6. Apply constitutional genome - emit GENOME_EXTRACT_REQUEST via Synapse
         and call memory.seed_genome() so Memory/Equor reinitialize from the
         parent's drive weights instead of defaults.

    Args:
        event_bus: Optional Synapse EventBus. If provided, GENOME_EXTRACT_REQUEST
            is emitted after restoration so Memory and Equor can reinitialize state.
        memory: Optional MemoryService. If provided, memory.seed_genome() is called
            directly with the extracted constitutional genome.

    Returns the constitutional_genome dict extracted from the snapshot
    (or None if absent), so callers can apply it to Memory/Equor on startup.
    """
    from systems.identity.vault import IdentityVault
    from systems.skia.pinata_client import PinataClient

    log = logger.bind(component="skia.restore", cid=cid)
    log.info("restoration_starting")

    # 1. Download from IPFS
    pinata = PinataClient(
        api_url=pinata_api_url,
        gateway_url=pinata_gateway_url,
        jwt=pinata_jwt,
    )
    await pinata.connect()
    try:
        encrypted_bytes = await pinata.get_by_cid(cid)
    finally:
        await pinata.close()

    log.info("ipfs_download_complete", size_bytes=len(encrypted_bytes))

    # 2. Decrypt - the ciphertext is stored as ASCII Fernet token.
    # Read key_version from the snapshot manifest in Redis if available.
    # Falls back to 1 with a warning when the manifest lacks this field.
    key_version = 1
    if redis_client is not None:
        try:
            manifest_raw = await redis_client.get_json("skia:snapshot:manifest")
            if isinstance(manifest_raw, dict):
                recorded_version = manifest_raw.get("encryption_key_version")
                if recorded_version is not None:
                    key_version = int(recorded_version)
                    log.info("key_version_from_manifest", key_version=key_version)
                else:
                    log.warning("key_version_not_in_manifest", fallback=key_version)
        except Exception as kv_exc:
            log.warning("key_version_lookup_failed", fallback=key_version, error=str(kv_exc))

    vault = IdentityVault(passphrase=vault_passphrase)
    from systems.identity.vault import SealedEnvelope

    envelope = SealedEnvelope(
        platform_id="skia",
        purpose="state_snapshot",
        ciphertext=encrypted_bytes.decode("ascii"),
        key_version=key_version,
    )
    compressed = vault.decrypt(envelope)

    # 3. Decompress
    raw_bytes = gzip.decompress(compressed)
    payload_data = orjson.loads(raw_bytes)
    nodes: list[dict[str, Any]] = payload_data.get("nodes", [])
    edges: list[dict[str, Any]] = payload_data.get("edges", [])

    constitutional_genome: dict[str, Any] | None = payload_data.get("constitutional_genome")
    log.info(
        "snapshot_decoded",
        nodes=len(nodes),
        edges=len(edges),
        has_constitutional_genome=constitutional_genome is not None,
    )

    # 4. Import into Neo4j
    # Use MERGE to avoid duplicates on re-import
    for node in nodes:
        labels_str = ":".join(f"`{lbl}`" for lbl in node.get("labels", []))
        props = node.get("props", {})
        if not labels_str or not props:
            continue

        # Use a stable identifier from props if available
        stable_id = props.get("id") or props.get("instance_id") or props.get("name")
        if stable_id:
            query = (
                f"MERGE (n:{labels_str} {{id: $stable_id}}) "
                f"SET n += $props"
            )
            await neo4j.execute_write(query, {"stable_id": stable_id, "props": props})
        else:
            query = f"CREATE (n:{labels_str}) SET n = $props"
            await neo4j.execute_write(query, {"props": props})

    for edge in edges:
        edge_type = edge.get("type", "RELATED_TO")
        props = edge.get("props", {})
        source_id = edge.get("source", "")
        target_id = edge.get("target", "")

        # We can't use elementId across instances; match by node props.id instead
        # This depends on the exported nodes having stable id props
        query = (
            f"MATCH (a {{id: $source_id}}), (b {{id: $target_id}}) "
            f"MERGE (a)-[r:`{edge_type}`]->(b) "
            f"SET r += $props"
        )
        await neo4j.execute_write(
            query, {"source_id": source_id, "target_id": target_id, "props": props}
        )

    log.info("restoration_complete", nodes_imported=len(nodes), edges_imported=len(edges))

    # Signal to the health endpoint that graph population is finished.
    # _verify_handoff() on the parent will not accept the handoff until
    # this key is present (restoration_complete: true in health response).
    if redis_client is not None:
        restoration_flag_key = f"skia:restoration_complete:{instance_id or cid[:16]}"
        raw = redis_client.client
        await raw.set(restoration_flag_key, "1", ex=3600)

    # 6. Apply constitutional genome so the revived organism inherits parent drive weights.
    # Without this step, Memory and Equor reinitialize with default values and the
    # organism loses its phenotype on every resurrection.
    if constitutional_genome is not None:
        # 6a. Seed Memory directly if available (fastest path - no event round-trip).
        if memory is not None:
            try:
                await memory.seed_genome(constitutional_genome)
                log.info("constitutional_genome_seeded_to_memory")
            except Exception as mem_exc:
                log.warning("memory_seed_genome_failed", error=str(mem_exc))

        # 6b. Emit GENOME_EXTRACT_REQUEST so Memory and Equor can reinitialize
        # their state from the genome payload via the event bus.
        if event_bus is not None:
            try:
                from systems.synapse.types import SynapseEvent, SynapseEventType
                await event_bus.emit(SynapseEvent(
                    event_type=SynapseEventType.GENOME_EXTRACT_REQUEST,
                    source_system="skia",
                    data={
                        "genome": constitutional_genome,
                        "source": "ipfs_restoration",
                        "cid": cid,
                        "instance_id": instance_id,
                    },
                ))
                log.info("genome_extract_request_emitted")
            except Exception as bus_exc:
                log.warning("genome_extract_request_failed", error=str(bus_exc))

    # Return the constitutional genome so callers can apply it further if needed.
    return constitutional_genome
