"""
EcodiaOS — EIS Threat Library (Immune Memory)

The threat library is the immune system's long-term memory. It maintains a
catalogue of known-bad patterns — mutations that were rolled back, knowledge
rejected by governance, behavioral signatures that preceded constitutional
drift — and scans new inputs against them.

Three pattern categories:

  1. **Mutation patterns** — diffs, file paths, and function-level signatures
     from mutations that were blocked, rolled back, or required escalation.
  2. **Knowledge patterns** — federated knowledge payloads that were rejected
     by Equor or flagged by the quarantine evaluator.
  3. **Behavioral patterns** — sequences of Synapse events that preceded a
     negative outcome (drift, crash, degradation).

The library is self-maintaining: it learns from every rejection and rollback
automatically via Synapse event subscriptions. No manual curation required.

Analogous to B-cell memory in biological immunity — the system remembers
what hurt it before so it can mount a faster response next time.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

import structlog

from primitives.common import EOSBaseModel, new_id, utc_now

logger = structlog.get_logger().bind(system="eis", component="threat_library")


# ─── Pattern Categories ──────────────────────────────────────────────────────


class ThreatPatternCategory(StrEnum):
    """Top-level classification for threat library entries."""

    MUTATION_ROLLBACK = "mutation_rollback"          # Mutation was rolled back
    MUTATION_BLOCKED = "mutation_blocked"             # Mutation blocked by taint/governance
    KNOWLEDGE_REJECTED = "knowledge_rejected"         # Federated knowledge rejected
    BEHAVIORAL_PRECURSOR = "behavioral_precursor"     # Event sequence that preceded failure
    QUARANTINE_CONFIRMED = "quarantine_confirmed"     # Quarantine evaluator confirmed threat
    DRIFT_PRECURSOR = "drift_precursor"               # Pattern preceded constitutional drift


class ThreatPatternStatus(StrEnum):
    """Lifecycle state of a threat pattern."""

    ACTIVE = "active"        # Currently matching against inputs
    DECAYED = "decayed"      # Below effectiveness threshold, pending retirement
    RETIRED = "retired"      # No longer matching, kept for audit


# ─── Threat Pattern ──────────────────────────────────────────────────────────


class ThreatPattern(EOSBaseModel):
    """
    A single known-bad pattern stored in the threat library.

    Patterns carry enough context for matching (signatures, file globs,
    event sequences) and enough metadata for self-maintenance (match
    counts, false positive tracking, decay).
    """

    id: str = field(default_factory=new_id)
    created_at: datetime = field(default_factory=utc_now)

    # ── Classification ──
    category: ThreatPatternCategory
    status: ThreatPatternStatus = ThreatPatternStatus.ACTIVE

    # ── Matching signatures ──
    # For mutations: file path globs that matched the bad mutation.
    file_patterns: list[str] = field(default_factory=list)
    # For mutations: function names that were changed.
    function_signatures: list[str] = field(default_factory=list)
    # Lexical fragments from the diff or content that are distinctive.
    lexical_signatures: list[str] = field(default_factory=list)
    # For behavioral patterns: ordered list of event types that formed the precursor.
    event_sequence: list[str] = field(default_factory=list)
    # Structural hash of the original content (for dedup).
    content_hash: str = ""

    # ── Context ──
    description: str = ""
    source_event_id: str = ""         # Synapse event ID that created this pattern
    source_system: str = ""           # Which system triggered the learning
    severity: str = "medium"          # critical/high/medium/low
    original_reasoning: str = ""      # Why the original was rejected/rolled back

    # ── Effectiveness tracking ──
    match_count: int = 0
    last_matched: datetime | None = None
    false_positive_count: int = 0
    true_positive_count: int = 0

    # ── Decay ──
    decay_after_days: int = 90        # Auto-decay if no matches in this period
    max_false_positive_rate: float = 0.3  # Retire if FP rate exceeds this


# ─── Match Result ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ThreatLibraryMatch:
    """Result of scanning content against the threat library."""

    pattern_id: str
    category: ThreatPatternCategory
    match_type: str           # "file_pattern", "function", "lexical", "event_sequence"
    matched_signature: str    # The specific signature that matched
    confidence: float         # 0.0-1.0
    severity: str
    description: str


@dataclass
class ThreatScanResult:
    """Aggregate result of scanning against the full threat library."""

    matches: list[ThreatLibraryMatch]
    scan_time_ms: int
    patterns_evaluated: int
    highest_severity: str = "none"

    @property
    def is_threat(self) -> bool:
        return len(self.matches) > 0

    @property
    def should_block(self) -> bool:
        return any(m.severity == "critical" for m in self.matches)


def _highest_severity(matches: list[ThreatLibraryMatch]) -> str:
    """Return the highest severity level from matches, or 'none' if empty."""
    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
    if not matches:
        return "none"
    severities = [m.severity for m in matches]
    return max(severities, key=lambda s: severity_order.get(s, 0))


# ─── Threat Library ──────────────────────────────────────────────────────────


class ThreatLibrary:
    """
    In-memory threat pattern store with auto-learning from Synapse events.

    The library maintains three indices for fast lookup:
      - file_index: file path glob → pattern IDs
      - function_index: function name → pattern IDs
      - content_hashes: content hash → pattern ID (dedup)

    Patterns are learned automatically from:
      - MODEL_ROLLBACK_TRIGGERED events (mutation patterns)
      - INTENT_REJECTED events (governance rejection patterns)
      - EVOLUTION_CANDIDATE_ASSESSED events with block_mutation=True
      - Confirmed quarantine verdicts from the EIS gate

    Self-maintenance:
      - Patterns decay after `decay_after_days` without matches
      - Patterns retire when false_positive_rate > max_false_positive_rate
      - Decay check runs on every scan (amortised, O(1) per scan)
    """

    def __init__(self, max_patterns: int = 10_000) -> None:
        self._patterns: dict[str, ThreatPattern] = {}
        self._max_patterns = max_patterns

        # ── Indices for fast matching ──
        self._file_index: dict[str, set[str]] = {}        # glob → {pattern_id}
        self._function_index: dict[str, set[str]] = {}    # func_name → {pattern_id}
        self._content_hashes: dict[str, str] = {}          # hash → pattern_id

        # ── Decay bookkeeping ──
        self._last_decay_check: float = time.monotonic()
        self._decay_check_interval_s: float = 300.0  # Check every 5 minutes

        # ── Stats ──
        self._total_scans: int = 0
        self._total_matches: int = 0
        self._total_learned: int = 0

        self._logger = logger

    # ─── Pattern Registration ─────────────────────────────────────

    def register_pattern(self, pattern: ThreatPattern) -> bool:
        """
        Add a new pattern to the library.

        Returns False if the pattern is a duplicate (same content_hash)
        or the library is at capacity.
        """
        # Dedup by content hash
        if pattern.content_hash and pattern.content_hash in self._content_hashes:
            existing_id = self._content_hashes[pattern.content_hash]
            if existing_id in self._patterns:
                # Boost existing pattern instead of duplicating
                existing = self._patterns[existing_id]
                existing.match_count += 1
                existing.last_matched = utc_now()
                self._logger.debug(
                    "threat_pattern_dedup_boost",
                    pattern_id=existing_id,
                    category=pattern.category.value,
                )
                return False

        # Capacity check — evict oldest decayed pattern if full
        if len(self._patterns) >= self._max_patterns:
            evicted = self._evict_one()
            if not evicted:
                self._logger.warning("threat_library_full", max=self._max_patterns)
                return False

        self._patterns[pattern.id] = pattern
        self._total_learned += 1

        # Build indices
        if pattern.content_hash:
            self._content_hashes[pattern.content_hash] = pattern.id

        for fp in pattern.file_patterns:
            self._file_index.setdefault(fp, set()).add(pattern.id)

        for fn in pattern.function_signatures:
            self._function_index.setdefault(fn.lower(), set()).add(pattern.id)

        self._logger.info(
            "threat_pattern_registered",
            pattern_id=pattern.id,
            category=pattern.category.value,
            severity=pattern.severity,
            file_patterns=len(pattern.file_patterns),
            lexical_sigs=len(pattern.lexical_signatures),
        )
        return True

    # ─── Scanning ─────────────────────────────────────────────────

    def scan_mutation(
        self,
        file_path: str,
        diff: str,
        changed_functions: set[str] | None = None,
    ) -> ThreatScanResult:
        """
        Scan a mutation proposal against the threat library.

        Checks file path patterns, function signatures, and lexical
        content against known-bad patterns from previous rollbacks
        and governance rejections.
        """
        t0 = time.monotonic()
        matches: list[ThreatLibraryMatch] = []
        evaluated = 0

        # Candidate patterns from file index
        candidate_ids: set[str] = set()
        for glob_pattern, pattern_ids in self._file_index.items():
            if _glob_match(file_path, glob_pattern):
                candidate_ids.update(pattern_ids)

        # Candidate patterns from function index
        if changed_functions:
            for fn in changed_functions:
                fn_lower = fn.lower()
                if fn_lower in self._function_index:
                    candidate_ids.update(self._function_index[fn_lower])

        # Evaluate candidates
        for pid in candidate_ids:
            pattern = self._patterns.get(pid)
            if pattern is None or pattern.status != ThreatPatternStatus.ACTIVE:
                continue
            evaluated += 1

            # File pattern match
            for fp in pattern.file_patterns:
                if _glob_match(file_path, fp):
                    matches.append(ThreatLibraryMatch(
                        pattern_id=pid,
                        category=pattern.category,
                        match_type="file_pattern",
                        matched_signature=fp,
                        confidence=0.6,
                        severity=pattern.severity,
                        description=pattern.description,
                    ))

            # Function signature match
            if changed_functions:
                for fn_sig in pattern.function_signatures:
                    if fn_sig.lower() in {f.lower() for f in changed_functions}:
                        matches.append(ThreatLibraryMatch(
                            pattern_id=pid,
                            category=pattern.category,
                            match_type="function",
                            matched_signature=fn_sig,
                            confidence=0.8,
                            severity=pattern.severity,
                            description=pattern.description,
                        ))

            # Lexical signature match against diff
            for lex in pattern.lexical_signatures:
                if lex in diff:
                    matches.append(ThreatLibraryMatch(
                        pattern_id=pid,
                        category=pattern.category,
                        match_type="lexical",
                        matched_signature=lex[:120],
                        confidence=0.7,
                        severity=pattern.severity,
                        description=pattern.description,
                    ))

        # Also scan all active patterns for content hash match
        diff_hash = _hash_content(diff)
        if diff_hash in self._content_hashes:
            pid = self._content_hashes[diff_hash]
            pattern = self._patterns.get(pid)
            if pattern and pattern.status == ThreatPatternStatus.ACTIVE:
                evaluated += 1
                matches.append(ThreatLibraryMatch(
                    pattern_id=pid,
                    category=pattern.category,
                    match_type="content_hash",
                    matched_signature=diff_hash[:16],
                    confidence=0.95,
                    severity=pattern.severity,
                    description=pattern.description,
                ))

        # Update match counts
        matched_pattern_ids = {m.pattern_id for m in matches}
        now = utc_now()
        for pid in matched_pattern_ids:
            if pid in self._patterns:
                self._patterns[pid].match_count += 1
                self._patterns[pid].last_matched = now

        self._total_scans += 1
        self._total_matches += len(matches)

        # Amortised decay check
        self._maybe_run_decay()

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        highest_severity = _highest_severity(matches)

        return ThreatScanResult(
            matches=matches,
            scan_time_ms=elapsed_ms,
            patterns_evaluated=evaluated,
            highest_severity=highest_severity,
        )

    def scan_knowledge(
        self,
        content: str,
        source_instance: str = "",
    ) -> ThreatScanResult:
        """
        Scan incoming federated knowledge against the threat library.

        Checks content hash and lexical signatures from previously
        rejected knowledge payloads.
        """
        t0 = time.monotonic()
        matches: list[ThreatLibraryMatch] = []
        evaluated = 0
        content_hash = _hash_content(content)

        for pid, pattern in self._patterns.items():
            if pattern.status != ThreatPatternStatus.ACTIVE:
                continue
            if pattern.category not in (
                ThreatPatternCategory.KNOWLEDGE_REJECTED,
                ThreatPatternCategory.QUARANTINE_CONFIRMED,
            ):
                continue

            evaluated += 1

            # Content hash match (exact duplicate of rejected knowledge)
            if pattern.content_hash and pattern.content_hash == content_hash:
                matches.append(ThreatLibraryMatch(
                    pattern_id=pid,
                    category=pattern.category,
                    match_type="content_hash",
                    matched_signature=content_hash[:16],
                    confidence=0.95,
                    severity=pattern.severity,
                    description=pattern.description,
                ))

            # Lexical signature match
            for lex in pattern.lexical_signatures:
                if lex in content:
                    matches.append(ThreatLibraryMatch(
                        pattern_id=pid,
                        category=pattern.category,
                        match_type="lexical",
                        matched_signature=lex[:120],
                        confidence=0.7,
                        severity=pattern.severity,
                        description=pattern.description,
                    ))

        # Update match counts
        now = utc_now()
        for m in matches:
            if m.pattern_id in self._patterns:
                self._patterns[m.pattern_id].match_count += 1
                self._patterns[m.pattern_id].last_matched = now

        self._total_scans += 1
        self._total_matches += len(matches)
        self._maybe_run_decay()

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        return ThreatScanResult(
            matches=matches,
            scan_time_ms=elapsed_ms,
            patterns_evaluated=evaluated,
            highest_severity=_highest_severity(matches),
        )

    # ─── Auto-Learning from Events ───────────────────────────────

    def learn_from_rollback(
        self,
        file_path: str,
        diff: str,
        changed_functions: set[str],
        reasoning: str,
        source_event_id: str = "",
    ) -> ThreatPattern | None:
        """
        Learn a new threat pattern from a rolled-back mutation.

        Called when MODEL_ROLLBACK_TRIGGERED fires. Extracts distinctive
        signatures from the mutation that was undone and stores them for
        future matching.
        """
        content_hash = _hash_content(diff)
        lexical_sigs = _extract_lexical_signatures(diff)

        pattern = ThreatPattern(
            id=new_id(),
            category=ThreatPatternCategory.MUTATION_ROLLBACK,
            file_patterns=[file_path] if file_path else [],
            function_signatures=list(changed_functions),
            lexical_signatures=lexical_sigs,
            content_hash=content_hash,
            description=f"Rolled-back mutation on {file_path}",
            source_event_id=source_event_id,
            source_system="simula",
            severity="high",
            original_reasoning=reasoning[:500],
        )

        if self.register_pattern(pattern):
            return pattern
        return None

    def learn_from_governance_rejection(
        self,
        file_path: str,
        diff: str,
        changed_functions: set[str],
        reasoning: str,
        severity: str = "medium",
        source_event_id: str = "",
    ) -> ThreatPattern | None:
        """
        Learn a new threat pattern from a governance-blocked mutation.

        Called when EVOLUTION_CANDIDATE_ASSESSED fires with
        block_mutation=True. The governance system (Equor) rejected this
        mutation, so we remember it.
        """
        content_hash = _hash_content(diff)
        lexical_sigs = _extract_lexical_signatures(diff)

        pattern = ThreatPattern(
            id=new_id(),
            category=ThreatPatternCategory.MUTATION_BLOCKED,
            file_patterns=[file_path] if file_path else [],
            function_signatures=list(changed_functions),
            lexical_signatures=lexical_sigs,
            content_hash=content_hash,
            description=f"Governance-blocked mutation on {file_path}",
            source_event_id=source_event_id,
            source_system="equor",
            severity=severity,
            original_reasoning=reasoning[:500],
        )

        if self.register_pattern(pattern):
            return pattern
        return None

    def learn_from_knowledge_rejection(
        self,
        content: str,
        source_instance: str,
        reasoning: str,
        source_event_id: str = "",
    ) -> ThreatPattern | None:
        """
        Learn from rejected federated knowledge.

        Called when incoming federated knowledge is rejected by Equor
        review or EIS quarantine. Remembers the content signature so
        identical or similar payloads are caught faster next time.
        """
        content_hash = _hash_content(content)
        lexical_sigs = _extract_lexical_signatures(content)

        pattern = ThreatPattern(
            id=new_id(),
            category=ThreatPatternCategory.KNOWLEDGE_REJECTED,
            lexical_signatures=lexical_sigs,
            content_hash=content_hash,
            description=f"Rejected knowledge from instance {source_instance}",
            source_event_id=source_event_id,
            source_system="federation",
            severity="medium",
            original_reasoning=reasoning[:500],
        )

        if self.register_pattern(pattern):
            return pattern
        return None

    def learn_from_quarantine_verdict(
        self,
        content: str,
        threat_class: str,
        severity: str,
        reasoning: str,
    ) -> ThreatPattern | None:
        """
        Learn from a confirmed quarantine verdict (BLOCK or QUARANTINE).

        Supplements the antibody system — while antibodies use vector
        similarity, threat patterns use exact/lexical matching for
        known-bad content that may evade embedding similarity.
        """
        content_hash = _hash_content(content)
        lexical_sigs = _extract_lexical_signatures(content)

        pattern = ThreatPattern(
            id=new_id(),
            category=ThreatPatternCategory.QUARANTINE_CONFIRMED,
            lexical_signatures=lexical_sigs,
            content_hash=content_hash,
            description=f"Quarantine-confirmed {threat_class} threat",
            source_system="eis.quarantine",
            severity=severity,
            original_reasoning=reasoning[:500],
        )

        if self.register_pattern(pattern):
            return pattern
        return None

    def learn_behavioral_precursor(
        self,
        event_sequence: list[str],
        outcome: str,
        severity: str = "medium",
        source_event_id: str = "",
    ) -> ThreatPattern | None:
        """
        Learn a behavioral event sequence that preceded a negative outcome.

        Called by the anomaly detector when it identifies a recurring
        sequence of events that precedes system failures.
        """
        seq_hash = _hash_content("|".join(event_sequence))

        pattern = ThreatPattern(
            id=new_id(),
            category=ThreatPatternCategory.BEHAVIORAL_PRECURSOR,
            event_sequence=event_sequence,
            content_hash=seq_hash,
            description=f"Behavioral precursor to {outcome}",
            source_event_id=source_event_id,
            source_system="eis.anomaly_detector",
            severity=severity,
            original_reasoning=f"Event sequence preceded: {outcome}",
        )

        if self.register_pattern(pattern):
            return pattern
        return None

    # ─── False Positive Feedback ─────────────────────────────────

    def record_false_positive(self, pattern_id: str) -> None:
        """Record that a pattern match was a false positive."""
        pattern = self._patterns.get(pattern_id)
        if pattern:
            pattern.false_positive_count += 1
            fp_rate = self._false_positive_rate(pattern)
            if fp_rate > pattern.max_false_positive_rate:
                pattern.status = ThreatPatternStatus.RETIRED
                self._logger.info(
                    "threat_pattern_retired_fp",
                    pattern_id=pattern_id,
                    fp_rate=round(fp_rate, 3),
                )

    def record_true_positive(self, pattern_id: str) -> None:
        """Record that a pattern match was a true positive."""
        pattern = self._patterns.get(pattern_id)
        if pattern:
            pattern.true_positive_count += 1

    # ─── Self-Maintenance ─────────────────────────────────────────

    def _maybe_run_decay(self) -> None:
        """Run decay check if enough time has passed since last check."""
        now = time.monotonic()
        if now - self._last_decay_check < self._decay_check_interval_s:
            return
        self._last_decay_check = now
        self._run_decay()

    def _run_decay(self) -> None:
        """
        Decay patterns that haven't matched in their decay window.

        Active → Decayed: no matches in `decay_after_days`.
        Decayed → Retired: still no matches after another decay period.
        """
        now = utc_now()
        decayed_count = 0
        retired_count = 0

        for pattern in self._patterns.values():
            if pattern.status == ThreatPatternStatus.RETIRED:
                continue

            # Check time-based decay
            reference_time = pattern.last_matched or pattern.created_at
            age = now - reference_time
            decay_threshold = timedelta(days=pattern.decay_after_days)

            if age > decay_threshold * 2 and pattern.status == ThreatPatternStatus.DECAYED:
                pattern.status = ThreatPatternStatus.RETIRED
                retired_count += 1
            elif age > decay_threshold and pattern.status == ThreatPatternStatus.ACTIVE:
                pattern.status = ThreatPatternStatus.DECAYED
                decayed_count += 1

            # Check false positive rate
            fp_rate = self._false_positive_rate(pattern)
            if fp_rate > pattern.max_false_positive_rate and pattern.match_count >= 5:
                pattern.status = ThreatPatternStatus.RETIRED
                retired_count += 1

        if decayed_count or retired_count:
            self._logger.info(
                "threat_library_decay_pass",
                decayed=decayed_count,
                retired=retired_count,
                active=sum(
                    1 for p in self._patterns.values()
                    if p.status == ThreatPatternStatus.ACTIVE
                ),
            )

    def _evict_one(self) -> bool:
        """Evict the least-useful pattern (retired first, then oldest decayed)."""
        # Retired patterns first
        for pid in list(self._patterns):
            if self._patterns[pid].status == ThreatPatternStatus.RETIRED:
                self._remove_pattern(pid)
                return True
        # Then oldest decayed
        decayed = [
            (pid, p) for pid, p in self._patterns.items()
            if p.status == ThreatPatternStatus.DECAYED
        ]
        if decayed:
            oldest = min(decayed, key=lambda x: x[1].created_at)
            self._remove_pattern(oldest[0])
            return True
        return False

    def _remove_pattern(self, pattern_id: str) -> None:
        """Remove a pattern and clean up all indices."""
        pattern = self._patterns.pop(pattern_id, None)
        if pattern is None:
            return

        # Clean file index
        for fp in pattern.file_patterns:
            if fp in self._file_index:
                self._file_index[fp].discard(pattern_id)
                if not self._file_index[fp]:
                    del self._file_index[fp]

        # Clean function index
        for fn in pattern.function_signatures:
            fn_lower = fn.lower()
            if fn_lower in self._function_index:
                self._function_index[fn_lower].discard(pattern_id)
                if not self._function_index[fn_lower]:
                    del self._function_index[fn_lower]

        # Clean content hash
        if pattern.content_hash and self._content_hashes.get(pattern.content_hash) == pattern_id:
            del self._content_hashes[pattern.content_hash]

    @staticmethod
    def _false_positive_rate(pattern: ThreatPattern) -> float:
        total = pattern.true_positive_count + pattern.false_positive_count
        if total == 0:
            return 0.0
        return pattern.false_positive_count / total

    # ─── Stats & Health ───────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return observable statistics."""
        by_status = {"active": 0, "decayed": 0, "retired": 0}
        by_category: dict[str, int] = {}
        for p in self._patterns.values():
            by_status[p.status.value] = by_status.get(p.status.value, 0) + 1
            by_category[p.category.value] = by_category.get(p.category.value, 0) + 1

        return {
            "total_patterns": len(self._patterns),
            "by_status": by_status,
            "by_category": by_category,
            "total_scans": self._total_scans,
            "total_matches": self._total_matches,
            "total_learned": self._total_learned,
        }


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _hash_content(content: str) -> str:
    """SHA-256 hash of content for deduplication."""
    return hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()


def _glob_match(path: str, pattern: str) -> bool:
    """Simple glob matching for file paths. Supports * and ** wildcards."""
    import fnmatch
    # Normalise separators
    path = path.replace("\\", "/")
    pattern = pattern.replace("\\", "/")
    return fnmatch.fnmatch(path, pattern)


def _extract_lexical_signatures(content: str, max_sigs: int = 10) -> list[str]:
    """
    Extract distinctive lexical fragments from content.

    Focuses on unusual identifiers, function calls, and distinctive
    string literals that would help identify similar future content.
    """
    sigs: list[str] = []

    # Extract function/method definitions
    for m in re.finditer(r'(?:def|function|async def|class)\s+(\w{4,50})', content):
        sigs.append(m.group(0))
        if len(sigs) >= max_sigs:
            break

    # Extract distinctive string literals (4-60 chars, not common words)
    for m in re.finditer(r'["\']([^"\']{4,60})["\']', content):
        literal = m.group(1)
        # Skip common words/paths
        if not literal.startswith("/") and " " not in literal:
            sigs.append(literal)
            if len(sigs) >= max_sigs:
                break

    # Extract import/require statements
    for m in re.finditer(r'(?:import|from|require)\s+[\w.]+', content):
        sigs.append(m.group(0))
        if len(sigs) >= max_sigs:
            break

    return sigs[:max_sigs]
