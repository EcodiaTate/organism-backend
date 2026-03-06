"""
EcodiaOS — EIS Structural Feature Extraction

Extracts syntactic / structural fingerprints from text input.
These features capture statistical properties that distinguish
adversarial inputs from normal conversation: character class
distributions, entropy, delimiter density, script mixing, etc.

Performance contract: extract_structural_profile() must complete
in <3ms for inputs up to 100K characters. All computation is
pure Python math — no I/O, no model inference.

The structural profile serves two purposes:
1. Direct anomaly detection (unusual distributions flag suspicious input)
2. Input to the structural_vector used in antigenic similarity search
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter
from typing import Final

import numpy as np

from systems.eis.models import StructuralProfile

# ─── Pre-compiled patterns ───────────────────────────────────────

_URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    re.IGNORECASE,
)
_EMAIL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
)
_CODE_BLOCK_PATTERN: Final[re.Pattern[str]] = re.compile(r"```")
_BRACKET_OPEN: Final[frozenset[str]] = frozenset("([{<")
_BRACKET_CLOSE: Final[frozenset[str]] = frozenset(")]}>")
_DELIMITERS: Final[frozenset[str]] = frozenset("()[]{}<>|/\\\"'`~!@#$%^&*=+;:,.")

# Unicode script detection — map category prefixes to script names
_SCRIPT_RANGES: Final[dict[str, tuple[int, int]]] = {
    "Latin": (0x0000, 0x024F),
    "Cyrillic": (0x0400, 0x04FF),
    "Greek": (0x0370, 0x03FF),
    "Arabic": (0x0600, 0x06FF),
    "Hebrew": (0x0590, 0x05FF),
    "CJK": (0x4E00, 0x9FFF),
    "Hangul": (0xAC00, 0xD7AF),
    "Devanagari": (0x0900, 0x097F),
    "Thai": (0x0E00, 0x0E7F),
    "Katakana": (0x30A0, 0x30FF),
    "Hiragana": (0x3040, 0x309F),
}


# ─── Core extraction ─────────────────────────────────────────────


def _detect_scripts(text: str) -> list[str]:
    """Detect which Unicode scripts are present in the text. Sample first 5K chars."""
    sample = text[:5000]
    found: set[str] = set()
    for ch in sample:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        for script_name, (low, high) in _SCRIPT_RANGES.items():
            if low <= cp <= high:
                found.add(script_name)
                break
    return sorted(found)


def _compute_entropy(text: str) -> float:
    """Shannon entropy of character frequency distribution."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _compute_repetition_score(text: str) -> float:
    """
    Estimate substring repetition via compression ratio approximation.

    Uses character bigram repetition as a proxy for actual compression.
    Faster than zlib but correlated. Returns 0.0 (no repetition) to
    1.0 (highly repetitive).
    """
    if len(text) < 4:
        return 0.0

    sample = text[:10_000]
    bigrams = [sample[i : i + 2] for i in range(len(sample) - 1)]
    if not bigrams:
        return 0.0

    unique_bigrams = len(set(bigrams))
    total_bigrams = len(bigrams)

    # Max unique bigrams for a string of length N is N-1
    # Highly repetitive text has very few unique bigrams relative to total
    ratio = unique_bigrams / total_bigrams
    # Invert: low ratio = high repetition
    return max(0.0, min(1.0, 1.0 - ratio))


def _max_bracket_depth(text: str) -> int:
    """Compute maximum nesting depth of bracket-like characters."""
    depth = 0
    max_depth = 0
    for ch in text[:10_000]:  # Sample first 10K
        if ch in _BRACKET_OPEN:
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif ch in _BRACKET_CLOSE:
            depth = max(0, depth - 1)
    return max_depth


def extract_structural_profile(text: str) -> StructuralProfile:
    """
    Extract a structural fingerprint from the input text.

    Pure computation — no I/O, no model inference.
    Budget: <3ms for 100K character inputs.
    """
    start_ns = time.perf_counter_ns()

    char_count = len(text)
    if char_count == 0:
        elapsed_us = (time.perf_counter_ns() - start_ns) // 1000
        return StructuralProfile(latency_us=elapsed_us)

    # ── Basic counts (sample-based for large inputs) ──
    # Sample first 10K chars for all statistics
    sample = text[:10_000]
    sample_len = len(sample)
    sample_newlines = sample.count("\n")
    if char_count <= 10_000:
        line_count = sample_newlines + 1
    else:
        line_count = int((sample_newlines + 1) * (char_count / sample_len))
    sample_words = sample.split()
    sample_word_count = len(sample_words)
    # Extrapolate word count for full text
    if char_count <= 10_000:
        word_count = sample_word_count
    else:
        word_count = int(sample_word_count * (char_count / sample_len))

    avg_word_length = (
        sum(len(w) for w in sample_words) / sample_word_count
        if sample_word_count > 0
        else 0.0
    )

    sample_lines = sample.split("\n")
    max_line_length = max((len(line) for line in sample_lines), default=0)

    # ── Character class ratios (sample first 10K chars for speed) ──
    alpha_count = 0
    digit_count = 0
    ws_count = 0
    punct_count = 0
    special_count = 0
    upper_count = 0

    sample_text = text[:10_000]
    sample_len_for_ratios = len(sample_text)

    for ch in sample_text:
        if ch.isalpha():
            alpha_count += 1
            if ch.isupper():
                upper_count += 1
        elif ch.isdigit():
            digit_count += 1
        elif ch.isspace():
            ws_count += 1
        elif ch.isascii() and not ch.isalnum():
            punct_count += 1
        elif not ch.isascii():
            special_count += 1

    alpha_ratio = alpha_count / sample_len_for_ratios
    digit_ratio = digit_count / sample_len_for_ratios
    whitespace_ratio = ws_count / sample_len_for_ratios
    punctuation_ratio = punct_count / sample_len_for_ratios
    special_char_ratio = special_count / sample_len_for_ratios
    uppercase_ratio = upper_count / alpha_count if alpha_count > 0 else 0.0

    # ── Structural indicators (reuse `sample` = text[:10_000]) ──
    entropy = _compute_entropy(sample[:5_000])  # 5K sample is sufficient
    unique_chars = len(set(sample[:5_000]))
    unique_sample_len = min(char_count, 5_000)
    unique_char_ratio = unique_chars / unique_sample_len if unique_sample_len > 0 else 0.0
    repetition_score = _compute_repetition_score(sample)

    # ── Delimiters (reuse sample) ──
    delimiter_count = sum(1 for ch in sample if ch in _DELIMITERS)
    delimiter_density = (delimiter_count / sample_len * 100) if sample_len > 0 else 0.0
    bracket_depth = _max_bracket_depth(sample)

    # ── Content patterns (reuse sample) ──
    code_block_count = len(_CODE_BLOCK_PATTERN.findall(sample)) // 2
    url_count = len(_URL_PATTERN.findall(sample))
    email_count = len(_EMAIL_PATTERN.findall(sample))

    # ── Script detection (5K sample) ──
    detected_scripts = _detect_scripts(sample[:5_000])
    mixed_script = len(detected_scripts) > 1

    elapsed_us = (time.perf_counter_ns() - start_ns) // 1000

    return StructuralProfile(
        char_count=char_count,
        word_count=word_count,
        line_count=line_count,
        avg_word_length=round(avg_word_length, 2),
        alpha_ratio=round(alpha_ratio, 4),
        digit_ratio=round(digit_ratio, 4),
        whitespace_ratio=round(whitespace_ratio, 4),
        punctuation_ratio=round(punctuation_ratio, 4),
        special_char_ratio=round(special_char_ratio, 4),
        uppercase_ratio=round(uppercase_ratio, 4),
        max_line_length=max_line_length,
        entropy=round(entropy, 4),
        unique_char_ratio=round(unique_char_ratio, 4),
        repetition_score=round(repetition_score, 4),
        bracket_depth=bracket_depth,
        delimiter_density=round(delimiter_density, 4),
        code_block_count=code_block_count,
        url_count=url_count,
        email_count=email_count,
        detected_scripts=detected_scripts,
        mixed_script=mixed_script,
        latency_us=elapsed_us,
    )


# ─── Structural vector ───────────────────────────────────────────


def structural_profile_to_vector(
    profile: StructuralProfile,
    dim: int = 32,
) -> list[float]:
    """
    Convert a StructuralProfile into a fixed-length normalised vector.

    This vector is used for antigenic similarity search — comparing
    the structural shape of an input against known pathogen signatures.

    Features are selected to maximise discrimination between benign
    and adversarial inputs. All features are normalised to [0, 1].
    """
    # Build raw feature vector (32 features)
    raw = np.array([
        # Character class ratios (6)
        profile.alpha_ratio,
        profile.digit_ratio,
        profile.whitespace_ratio,
        profile.punctuation_ratio,
        profile.special_char_ratio,
        profile.uppercase_ratio,

        # Length features (4) — log-scaled, capped
        min(math.log1p(profile.char_count) / 12.0, 1.0),    # log(100K) ≈ 11.5
        min(math.log1p(profile.word_count) / 10.0, 1.0),
        min(math.log1p(profile.line_count) / 8.0, 1.0),
        min(profile.avg_word_length / 20.0, 1.0),

        # Entropy features (3)
        min(profile.entropy / 8.0, 1.0),                     # Max entropy ≈ 8 bits
        profile.unique_char_ratio,
        profile.repetition_score,

        # Delimiter features (3)
        min(profile.bracket_depth / 20.0, 1.0),
        min(profile.delimiter_density / 30.0, 1.0),
        min(profile.max_line_length / 1000.0, 1.0),

        # Content features (3)
        min(profile.code_block_count / 10.0, 1.0),
        min(profile.url_count / 20.0, 1.0),
        min(profile.email_count / 10.0, 1.0),

        # Script features (2)
        min(len(profile.detected_scripts) / 5.0, 1.0),
        1.0 if profile.mixed_script else 0.0,

        # Derived ratios (3)
        min((profile.punctuation_ratio + profile.special_char_ratio) * 5, 1.0),
        1.0 - profile.alpha_ratio if profile.alpha_ratio < 0.3 else 0.0,  # Non-text indicator
        min(profile.digit_ratio * 10, 1.0),  # Digit-heavy indicator

        # Padding to dim (8 zeros for future features)
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float32)

    # Truncate or pad to exact dimension
    if len(raw) > dim:
        raw = raw[:dim]
    elif len(raw) < dim:
        raw = np.pad(raw, (0, dim - len(raw)), constant_values=0.0)

    # L2 normalise for cosine similarity
    norm = np.linalg.norm(raw)
    if norm > 0:
        raw = raw / norm

    return raw.tolist()


def structural_profile_hash(profile: StructuralProfile) -> str:
    """
    Compute a stable hash of the structural profile for deduplication.

    Two inputs with identical structural profiles (same char distribution,
    same entropy, same delimiter pattern) get the same hash.
    """
    # Quantise floats to 2 decimal places for stable hashing
    key_parts = [
        f"cc:{profile.char_count}",
        f"wc:{profile.word_count}",
        f"ar:{profile.alpha_ratio:.2f}",
        f"dr:{profile.digit_ratio:.2f}",
        f"pr:{profile.punctuation_ratio:.2f}",
        f"sr:{profile.special_char_ratio:.2f}",
        f"en:{profile.entropy:.2f}",
        f"rp:{profile.repetition_score:.2f}",
        f"bd:{profile.bracket_depth}",
        f"dd:{profile.delimiter_density:.1f}",
        f"ms:{1 if profile.mixed_script else 0}",
    ]
    raw = "|".join(key_parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
