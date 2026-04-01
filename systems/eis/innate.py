"""
EcodiaOS - EIS Innate Checks

Fast, deterministic regex/string pattern checks for known epistemic threats.
Analogous to innate immunity: non-specific, pattern-based, zero-latency.

Each check targets a specific attack surface:
  - Prompt injection / role override
  - Instruction injection via delimiters
  - Encoding evasion (base64, hex, unicode)
  - Repetition / context window stuffing
  - Data exfiltration patterns
  - Identity spoofing
  - Known jailbreak phrases

Performance contract: run_innate_checks() must complete in <5ms for
inputs up to 100K characters. All patterns are pre-compiled at module
load time.

This module is import-time safe - no side effects, no I/O.
"""

from __future__ import annotations

import re
import time
import unicodedata
from collections import Counter
from typing import Final

from systems.eis.models import (
    InnateCheckID,
    InnateFlags,
    InnateMatch,
    ThreatClass,
    ThreatSeverity,
)

# ─── Pre-compiled patterns ───────────────────────────────────────
#
# All regexes compile once at import time. re.IGNORECASE is used
# where case-insensitive matching matters. re.DOTALL for multiline.


# 1. System prompt leak attempts
_SYSTEM_PROMPT_LEAK: Final[list[re.Pattern[str]]] = [
    re.compile(
        r"(?:repeat|show|display|print|output|reveal|tell me|give me|what (?:is|are)|"
        r"recite|dump|echo|list)\s+"
        r"(?:your|the|my|system|initial|original|full|complete|entire|hidden|secret)\s+"
        r"(?:prompt|instructions?|rules?|system\s*message|context|directives?|guidelines?|"
        r"configuration|setup|preamble)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:ignore|forget|disregard)[\s\S]{0,30}(?:above|previous|prior|earlier|all)\s+"
        r"(?:instructions?|rules?|directives?|context|guidelines?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"you\s+are\s+now\s+(?:in\s+)?(?:developer|debug|admin|root|god|sudo|unrestricted|"
        r"unfiltered|jailbroken|DAN)\s*mode",
        re.IGNORECASE,
    ),
]

# 2. Role override
_ROLE_OVERRIDE: Final[list[re.Pattern[str]]] = [
    re.compile(
        r"(?:^|\n)\s*(?:system|assistant|user)\s*:\s*",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"<\|(?:im_start|im_end|system|endoftext|sep)\|>",
        re.IGNORECASE,
    ),
    re.compile(
        r"\[(?:SYSTEM|INST|/INST|\[INST\])\]",
        re.IGNORECASE,
    ),
    re.compile(
        r"<<\s*SYS\s*>>|<\s*/\s*SYS\s*>",
        re.IGNORECASE,
    ),
]

# 3. Instruction injection via delimiters
_INSTRUCTION_INJECTION: Final[list[re.Pattern[str]]] = [
    re.compile(
        r"(?:BEGIN|START|ENTER)\s+(?:NEW|OVERRIDE|CUSTOM|REAL|TRUE|ACTUAL)\s+"
        r"(?:INSTRUCTIONS?|PROMPT|RULES?|SYSTEM|CONTEXT|SESSION|MODE)",
        re.IGNORECASE,
    ),
    re.compile(
        r"={5,}|~{5,}|-{10,}|\*{5,}|#{5,}",
    ),
    re.compile(
        r"---+\s*(?:NEW|OVERRIDE|REAL|TRUE)\s+(?:PROMPT|INSTRUCTIONS?|RULES?)\s*---+",
        re.IGNORECASE,
    ),
]

# 4. Encoding evasion (base64, hex, rot13 mentions)
_ENCODING_EVASION: Final[list[re.Pattern[str]]] = [
    re.compile(
        r"(?:decode|interpret|execute|run|eval|parse)\s+(?:this\s+)?(?:as\s+)?"
        r"(?:base64|hex|rot13|binary|ascii|unicode|utf)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:base64|atob|btoa)\s*\(\s*['\"]",
        re.IGNORECASE,
    ),
    # Long base64-like strings (60+ chars, tightened to reduce false positives)
    re.compile(
        r"[A-Za-z0-9+/]{60,}={0,2}",
    ),
]

# 5. Delimiter abuse
_DELIMITER_ABUSE: Final[list[re.Pattern[str]]] = [
    re.compile(r"```\s*(?:system|prompt|instructions?|python|javascript|bash|sh)\b", re.IGNORECASE),
    re.compile(r"<(?:script|iframe|object|embed|form|input|svg|img\s+onerror)\b", re.IGNORECASE),
    re.compile(r"\{\{[^}]{1,500}\}\}"),  # Template injection {{ }} - bounded length
]

# 6. Repetition attack (same word/phrase repeated many times)
# NOTE: We use Counter-based detection instead of backreference regex
# because backreference patterns are O(n²) on repetitive input and
# blow the <5ms budget on large inputs.
_REPETITION_THRESHOLD: Final[int] = 20  # Same word appearing 20+ times

# 7. Context window stuffing (unusual volume indicators)
_STUFFING_CHAR_THRESHOLD: Final[int] = 50000  # 50K chars is suspicious in a single message

# 8. Data exfiltration patterns
_DATA_EXFIL: Final[list[re.Pattern[str]]] = [
    re.compile(
        r"(?:send|post|transmit|upload|exfiltrate|forward|email|webhook)\s+"
        r"(?:to|at|via)\s+"
        r"(?:https?://|ftp://|mailto:|[\w.-]+@[\w.-]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:curl|wget|fetch|requests?\.(?:get|post)|httpx|axios)\s*\(",
        re.IGNORECASE,
    ),
]

# 9. Identity spoofing
_IDENTITY_SPOOF: Final[list[re.Pattern[str]]] = [
    re.compile(
        r"(?:I am|I'm|this is|speaking as|acting as|responding as)\s+"
        r"(?:the\s+)?(?:system|assistant|AI|Claude|GPT|ChatGPT|Anthropic|OpenAI|"
        r"administrator|admin|root|moderator)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:from|signed|sent by|authored by)\s*:?\s*"
        r"(?:system|assistant|AI|Claude|GPT|the\s+developer|admin)",
        re.IGNORECASE,
    ),
]

# 10. Known jailbreak phrases
_JAILBREAK_PHRASES: Final[list[re.Pattern[str]]] = [
    re.compile(r"\bDAN\b.*\bDo Anything Now\b", re.IGNORECASE),
    re.compile(r"\bjailbreak(?:ed|ing)?\b", re.IGNORECASE),
    re.compile(
        r"(?:pretend|act|roleplay|imagine)\s+(?:you\s+)?(?:are|were|that\s+you(?:'re|\s+are))\s+"
        r"(?:an?\s+)?(?:evil|unfiltered|unrestricted|uncensored|unethical|amoral|"
        r"malicious|harmful|dangerous)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:bypass|circumvent|ignore|override|disable|turn off|remove)\s+"
        r"(?:your\s+)?(?:safety|ethical|moral|content)\s+"
        r"(?:filters?|guidelines?|restrictions?|limits?|guardrails?|constraints?)",
        re.IGNORECASE,
    ),
]

# 11. Unicode smuggling (homoglyphs, bidirectional overrides)
_UNICODE_SMUGGLING: Final[list[re.Pattern[str]]] = [
    # Bidirectional override characters
    re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]"),
    # Zero-width characters in suspicious density
    re.compile(r"[\u200b\u200c\u200d\ufeff]{3,}"),
]

# 12. Invisible characters (control chars, zero-width joiners in bulk)
_INVISIBLE_CHAR_THRESHOLD: Final[int] = 5  # More than N invisible chars is suspicious


# ─── Severity weights for composite scoring ──────────────────────

_SEVERITY_SCORE: Final[dict[ThreatSeverity, float]] = {
    ThreatSeverity.CRITICAL: 1.0,
    ThreatSeverity.HIGH: 0.7,
    ThreatSeverity.MEDIUM: 0.4,
    ThreatSeverity.LOW: 0.15,
    ThreatSeverity.NONE: 0.0,
}


# ─── Check definitions ───────────────────────────────────────────


def _check_patterns(
    text: str,
    patterns: list[re.Pattern[str]],
    check_id: InnateCheckID,
    pattern_name: str,
    severity: ThreatSeverity,
    threat_class: ThreatClass,
) -> InnateMatch | None:
    """Run a list of patterns and return a match if any hits."""
    for pat in patterns:
        m = pat.search(text)
        if m is not None:
            return InnateMatch(
                check_id=check_id,
                matched=True,
                matched_text=m.group(0)[:200],  # Cap captured text length
                pattern_name=pattern_name,
                severity=severity,
                threat_class=threat_class,
            )
    return None


def _count_invisible_chars(text: str) -> int:
    """Count invisible / control characters in text."""
    count = 0
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("C") and ch not in ("\n", "\r", "\t"):
            count += 1
    return count


# ─── Main entry point ────────────────────────────────────────────


def run_innate_checks(text: str) -> InnateFlags:
    """
    Run all innate (fast-path) checks against the input text.

    Returns an InnateFlags with all match results. This function is
    synchronous and deterministic - no I/O, no randomness, no LLM calls.

    Performance: <5ms for inputs up to 100K characters (pre-compiled
    patterns, early-exit on short inputs).
    """
    start_ns = time.perf_counter_ns()
    matches: list[InnateMatch] = []

    # Early exit for very short inputs - unlikely to contain attacks
    if len(text) < 10:
        elapsed_us = (time.perf_counter_ns() - start_ns) // 1000
        return InnateFlags(latency_us=elapsed_us)

    # Truncate to first 4K chars for regex scanning.
    # Attack payloads are front-loaded (injection must appear early to
    # take effect). 4K captures the vast majority of real-world attacks
    # while keeping total regex cost under 5ms.
    check_text = text[:4_000]

    # 1. System prompt leak
    m = _check_patterns(
        check_text, _SYSTEM_PROMPT_LEAK, InnateCheckID.SYSTEM_PROMPT_LEAK,
        "system_prompt_leak", ThreatSeverity.HIGH, ThreatClass.DATA_EXFILTRATION,
    )
    if m is not None:
        matches.append(m)

    # 2. Role override
    m = _check_patterns(
        check_text, _ROLE_OVERRIDE, InnateCheckID.ROLE_OVERRIDE,
        "role_override", ThreatSeverity.CRITICAL, ThreatClass.PROMPT_INJECTION,
    )
    if m is not None:
        matches.append(m)

    # 3. Instruction injection
    m = _check_patterns(
        check_text, _INSTRUCTION_INJECTION, InnateCheckID.INSTRUCTION_INJECTION,
        "instruction_injection", ThreatSeverity.HIGH, ThreatClass.PROMPT_INJECTION,
    )
    if m is not None:
        matches.append(m)

    # 4. Encoding evasion
    m = _check_patterns(
        check_text, _ENCODING_EVASION, InnateCheckID.ENCODING_EVASION,
        "encoding_evasion", ThreatSeverity.MEDIUM, ThreatClass.PROMPT_INJECTION,
    )
    if m is not None:
        matches.append(m)

    # 5. Delimiter abuse
    m = _check_patterns(
        check_text, _DELIMITER_ABUSE, InnateCheckID.DELIMITER_ABUSE,
        "delimiter_abuse", ThreatSeverity.MEDIUM, ThreatClass.PROMPT_INJECTION,
    )
    if m is not None:
        matches.append(m)

    # 6. Repetition attack (Counter-based, O(n) instead of O(n²) regex)
    # Use check_text (already truncated to 10K) for word counting
    words_lower = check_text.split()
    if words_lower:
        word_counts = Counter(words_lower)
        most_common_word, most_common_count = word_counts.most_common(1)[0]
        if most_common_count >= _REPETITION_THRESHOLD and len(most_common_word) >= 3:
            matches.append(InnateMatch(
                check_id=InnateCheckID.REPETITION_ATTACK,
                matched=True,
                matched_text=f'"{most_common_word}" x{most_common_count}',
                pattern_name="repetition_attack",
                severity=ThreatSeverity.MEDIUM,
                threat_class=ThreatClass.CONTEXT_POISONING,
            ))

    # 7. Context window stuffing
    if len(text) > _STUFFING_CHAR_THRESHOLD:
        matches.append(InnateMatch(
            check_id=InnateCheckID.CONTEXT_WINDOW_STUFFING,
            matched=True,
            matched_text=f"<{len(text)} chars>",
            pattern_name="context_window_stuffing",
            severity=ThreatSeverity.LOW,
            threat_class=ThreatClass.CONTEXT_POISONING,
        ))

    # 8. Data exfiltration
    m = _check_patterns(
        check_text, _DATA_EXFIL, InnateCheckID.DATA_EXFIL_PATTERN,
        "data_exfil_pattern", ThreatSeverity.HIGH, ThreatClass.DATA_EXFILTRATION,
    )
    if m is not None:
        matches.append(m)

    # 9. Identity spoofing
    m = _check_patterns(
        check_text, _IDENTITY_SPOOF, InnateCheckID.IDENTITY_SPOOF,
        "identity_spoof", ThreatSeverity.MEDIUM, ThreatClass.IDENTITY_SPOOFING,
    )
    if m is not None:
        matches.append(m)

    # 10. Jailbreak phrases
    m = _check_patterns(
        check_text, _JAILBREAK_PHRASES, InnateCheckID.JAILBREAK_PHRASE,
        "jailbreak_phrase", ThreatSeverity.HIGH, ThreatClass.JAILBREAK,
    )
    if m is not None:
        matches.append(m)

    # 11. Unicode smuggling
    m = _check_patterns(
        check_text, _UNICODE_SMUGGLING, InnateCheckID.UNICODE_SMUGGLING,
        "unicode_smuggling", ThreatSeverity.MEDIUM, ThreatClass.PROMPT_INJECTION,
    )
    if m is not None:
        matches.append(m)

    # 12. Invisible characters (sample first 2K for speed)
    invisible_count = _count_invisible_chars(check_text[:2_000])
    if invisible_count > _INVISIBLE_CHAR_THRESHOLD:
        matches.append(InnateMatch(
            check_id=InnateCheckID.INVISIBLE_CHARS,
            matched=True,
            matched_text=f"<{invisible_count} invisible chars>",
            pattern_name="invisible_chars",
            severity=ThreatSeverity.MEDIUM,
            threat_class=ThreatClass.PROMPT_INJECTION,
        ))

    # ── Aggregate results ──
    any_triggered = len(matches) > 0
    highest_severity = ThreatSeverity.NONE
    total_score = 0.0

    for match in matches:
        score = _SEVERITY_SCORE.get(match.severity, 0.0)
        total_score += score
        if score > _SEVERITY_SCORE.get(highest_severity, 0.0):
            highest_severity = match.severity

    # Normalise total_score to 0.0-1.0 range
    # Max possible: all 12 checks match at CRITICAL = 12.0
    total_score = min(total_score / 3.0, 1.0)  # Scale so 3 high-severity matches ≈ 1.0

    elapsed_us = (time.perf_counter_ns() - start_ns) // 1000

    return InnateFlags(
        matches=matches,
        any_triggered=any_triggered,
        highest_severity=highest_severity,
        total_score=total_score,
        latency_us=elapsed_us,
    )
