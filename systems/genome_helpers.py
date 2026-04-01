"""
EcodiaOS - Genome Extraction Helpers

Shared utilities for systems implementing GenomeExtractionProtocol.
Handles SHA-256 hashing, size computation, segment construction,
and schema version validation.
"""

from __future__ import annotations

import hashlib
import json
import sys

import structlog

from primitives.common import SystemID, utc_now
from primitives.genome import OrganGenomeSegment

logger = structlog.get_logger()

_SCHEMA_VERSION = "1.0.0"
_SIZE_WARNING_BYTES = 1_000_000  # 1MB


def _compute_payload_hash(payload: dict) -> str:
    """SHA-256 of the JSON-serialised payload (sorted keys, compact)."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_segment(system_id: SystemID, payload: dict, version: int = 1) -> OrganGenomeSegment:
    """
    Build an OrganGenomeSegment with correct hash and size.

    Every system calls this instead of constructing OrganGenomeSegment directly.
    """
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    size_bytes = sys.getsizeof(raw)
    payload_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()

    if size_bytes > _SIZE_WARNING_BYTES:
        logger.warning(
            "genome_segment_large",
            system_id=system_id,
            size_bytes=size_bytes,
            limit=_SIZE_WARNING_BYTES,
        )

    return OrganGenomeSegment(
        system_id=system_id,
        version=version,
        schema_version=_SCHEMA_VERSION,
        payload=payload,
        payload_hash=payload_hash,
        size_bytes=size_bytes,
        extracted_at=utc_now(),
    )


def verify_segment(segment: OrganGenomeSegment) -> bool:
    """
    Verify a segment's payload_hash matches its payload.

    Returns True if valid, False if corrupted.
    """
    expected = _compute_payload_hash(segment.payload)
    if expected != segment.payload_hash:
        logger.error(
            "genome_segment_hash_mismatch",
            system_id=segment.system_id,
            expected=expected,
            actual=segment.payload_hash,
        )
        return False
    return True


def check_schema_version(segment: OrganGenomeSegment) -> bool:
    """
    Check if the segment's schema_version is compatible.

    For now, only "1.0.0" is supported. Returns True if compatible.
    Unknown versions log a warning and return False.
    """
    if segment.schema_version == _SCHEMA_VERSION:
        return True
    logger.warning(
        "genome_segment_unknown_schema",
        system_id=segment.system_id,
        schema_version=segment.schema_version,
        supported=_SCHEMA_VERSION,
    )
    return False
