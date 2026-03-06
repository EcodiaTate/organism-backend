"""
EcodiaOS — Structured Output Validator

Ensures LLM responses conform to expected schemas without rewrites.
Implements fast validation + auto-correction (no retry loops).

Key principle: Never retry the same LLM call. Instead:
1. Validate response against schema
2. If invalid, apply auto-correction heuristics
3. If still invalid, return None and caller uses fallback
"""

from __future__ import annotations

import json as _json
import re
from typing import Any, TypeVar

import structlog

logger = structlog.get_logger()

T = TypeVar('T')


class OutputValidator:
    """
    Validates LLM outputs against expected formats.
    Implements fast corrections to avoid retry loops.
    """

    @staticmethod
    def extract_json(text: str) -> dict[str, Any] | None:
        """
        Extract JSON from LLM response.

        Handles common issues:
        - Leading/trailing text
        - Markdown code blocks
        - Partial JSON (truncated due to token limit)

        Returns:
            Parsed JSON dict, or None if unfixable
        """
        text = text.strip()

        # Remove markdown code block markers
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
        text = text.strip()

        # Try direct parse
        try:
            parsed: dict[str, Any] | None = _json.loads(text)
            return parsed
        except _json.JSONDecodeError:
            pass

        # Try to find JSON object boundaries { ... }
        start_idx = text.find("{")
        end_idx = text.rfind("}")

        if start_idx >= 0 and end_idx > start_idx:
            candidate = text[start_idx : end_idx + 1]
            try:
                parsed = _json.loads(candidate)
                return parsed
            except _json.JSONDecodeError:
                pass

        # Try array
        start_idx = text.find("[")
        end_idx = text.rfind("]")

        if start_idx >= 0 and end_idx > start_idx:
            candidate = text[start_idx : end_idx + 1]
            try:
                parsed = _json.loads(candidate)
                return parsed
            except _json.JSONDecodeError:
                pass

        # Unable to salvage
        logger.warning(
            "output_json_parse_failed",
            text_preview=text[:200],
        )
        return None

    @staticmethod
    def extract_number(text: str, default: float = 0.0) -> float:
        """
        Extract a floating-point number from text.

        Handles: "0.7", "value: 0.7", "[0.7]", etc.

        Returns:
            Parsed float, or default if not found
        """
        text = text.strip()

        # Look for decimal number pattern
        match = re.search(r"-?\d+\.?\d*", text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass

        return default

    @staticmethod
    def extract_string_list(text: str) -> list[str]:
        """
        Extract list of strings from LLM response.

        Handles: JSON array, newline-separated, comma-separated, bullet points.

        Returns:
            List of non-empty strings
        """
        text = text.strip()

        # Try JSON array first
        try:
            parsed = _json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if item]
        except _json.JSONDecodeError:
            pass

        # Try newline-separated (with bullet points)
        results = []
        for line in text.split("\n"):
            line = line.strip()
            # Remove bullet points, numbers, etc.
            line = re.sub(r"^[\-\*\+\d\.]+\s*", "", line)
            line = line.strip()
            if line and len(line) > 2:  # Skip empty or very short lines
                results.append(line)

        if results:
            return results

        # Fallback: comma-separated
        return [s.strip() for s in text.split(",") if s.strip()]

    @staticmethod
    def validate_enum(text: str, valid_values: list[str]) -> str | None:
        """
        Validate that text matches one of the valid enum values.

        Case-insensitive. Handles partial matches (e.g., "Pragmatic" → "pragmatic").

        Returns:
            Matched enum value (canonical case), or None if no match
        """
        text = text.strip().lower()

        for valid in valid_values:
            if text == valid.lower():
                return valid
            # Partial match (first word)
            if text == valid.lower().split()[0]:
                return valid

        return None

    @staticmethod
    def truncate_at_token_limit(text: str, max_tokens: int = 500) -> str:
        """
        Truncate text to fit within token limit.

        Rough heuristic: ~4 chars per token.

        Returns:
            Truncated text with graceful ending
        """
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars].rstrip()
        # Remove incomplete sentence
        last_period = truncated.rfind(".")
        last_newline = truncated.rfind("\n")
        last_punct = max(last_period, last_newline)

        if last_punct > max_chars * 0.8:
            truncated = truncated[:last_punct + 1]

        return truncated + " [truncated]"

    @staticmethod
    def validate_dict_keys(
        data: dict[str, Any],
        required_keys: list[str],
    ) -> tuple[bool, list[str]]:
        """
        Validate that dict has all required keys.

        Returns:
            (is_valid, list_of_missing_keys)
        """
        missing = [key for key in required_keys if key not in data]
        return (len(missing) == 0, missing)

    @staticmethod
    def auto_fix_dict(
        data: dict[str, Any],
        required_keys: list[str],
        defaults: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Auto-fix dict by adding missing keys with defaults.

        Args:
            data: Parsed dict
            required_keys: Keys that must exist
            defaults: Default values (defaults to sensible types)

        Returns:
            Fixed dict
        """
        defaults = defaults or {}

        for key in required_keys:
            if key not in data:
                # Use provided default or infer type
                if key in defaults:
                    data[key] = defaults[key]
                else:
                    # Sensible defaults based on key name
                    if "score" in key or "value" in key:
                        data[key] = 0.5
                    elif "count" in key or "count" in key:
                        data[key] = 0
                    elif "reason" in key or "explanation" in key:
                        data[key] = ""
                    else:
                        data[key] = None

        return data

    @staticmethod
    def log_validation_error(
        system: str,
        method: str,
        issue: str,
        text_preview: str,
    ) -> None:
        """Log output validation failure."""
        logger.warning(
            "output_validation_failed",
            system=system,
            method=method,
            issue=issue,
            text_preview=text_preview[:200],
        )
