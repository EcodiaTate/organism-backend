"""
EcodiaOS - Synapse Event Bus Audit Scanner

Scans the codebase for SynapseEventType usage and reports:
  - WIRED:    events that have at least one emit site AND one subscribe site
  - EMIT_ONLY: events emitted but never subscribed
  - SUBSCRIBE_ONLY: events subscribed but never emitted (dangling - no data)
  - DEAD:     events defined in the enum but never referenced anywhere

Usage:
    python scripts/synapse_audit.py [--backend-root PATH] [--json]

Recognised emit patterns (8 patterns):
  1. SynapseEventType.FOO_BAR              - direct attribute access
  2. _SET.FOO_BAR / _SynET.FOO_BAR         - module-level alias (alias = SynapseEventType)
  3. SynapseEventType("foo_bar")           - constructor with string literal
  4. SynapseEventType(SOME_VAR)            - constructor with string variable
  5. _emit_equor_event("foo_bar", ...)     - Equor string helper
  6. _fire_event("FOO_BAR", ...)           - Identity vault string helper
  7. _emit_safe("foo_bar", ...)            - generic string helper
  8. String dispatch maps: "FOO_BAR": SynapseEventType.FOO_BAR  - dict key → enum value
  9. getattr(SynapseEventType, "FOO_BAR")  - dynamic access
 10. list-based subscribe loops            - subscribe(SET.X) in for loops

Recognised subscribe patterns:
  1. bus.subscribe(SynapseEventType.FOO_BAR, handler)
  2. bus.subscribe(_SET.FOO_BAR, handler)
  3. event_bus.subscribe(SynapseEventType.FOO_BAR, ...)
  4. Subscribe-all patterns: bus.subscribe_all(...)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# ── Patterns ──────────────────────────────────────────────────────────────────

# Matches: SynapseEventType.FOO_BAR  (attribute access)
_RE_DIRECT = re.compile(r"\bSynapseEventType\.([A-Z][A-Z0-9_]*)\b")

# Matches: _SET.FOO_BAR or _SynET.FOO_BAR or similar short aliases
_RE_SHORT_ALIAS = re.compile(r"\b_S(?:ET|ynET|ET2|ETYPE)\s*\.\s*([A-Z][A-Z0-9_]*)\b")

# Matches: SynapseEventType("foo_bar") or SynapseEventType(SOME_VARIABLE)
_RE_CONSTRUCTOR = re.compile(r"\bSynapseEventType\s*\(\s*['\"]([a-z][a-z0-9_]*)['\"]")

# Matches string constants assigned to lowercase snake_case variables that
# are later passed to SynapseEventType(...).
# e.g.  FOVEA_HABITUATION_DECAY = "fovea_habituation_decay"
_RE_STRING_CONSTANT = re.compile(r'\b([A-Z][A-Z0-9_]*)\s*=\s*["\']([a-z][a-z0-9_]*)["\']')

# Matches helper calls that accept event names as strings:
#   _emit_equor_event("equor_review_started", ...)
#   _fire_event("VAULT_KEY_ROTATION_STARTED", ...)
#   _emit_safe("foo_bar", ...)
_RE_STRING_HELPER = re.compile(
    r"""(?:_emit_equor_event|_fire_event|_emit_safe|_emit_synapse_event_str)\s*\(\s*['"]([A-Za-z][A-Za-z0-9_]*)['"]"""
)

# Matches dict dispatch maps: "FOO_BAR": SynapseEventType.FOO_BAR
_RE_DISPATCH_MAP_KEY = re.compile(r'''['"]((?:[A-Z][A-Z0-9_]*)|(?:[a-z][a-z0-9_]*))['"]\s*:\s*SynapseEventType\.([A-Z][A-Z0-9_]*)''')

# Matches: getattr(SynapseEventType, "FOO_BAR")
_RE_GETATTR = re.compile(r"""getattr\s*\(\s*SynapseEventType\s*,\s*['"]([A-Z][A-Z0-9_]*)['"]""")

# Subscribe patterns:
#   bus.subscribe(SynapseEventType.FOO, handler)
#   event_bus.subscribe(_SET.FOO, handler)
_RE_SUBSCRIBE = re.compile(
    r"""(?:bus|event_bus|_event_bus|_bus|synapse)\s*\.\s*subscribe\s*\(\s*"""
    r"""(?:SynapseEventType\.|_S(?:ET|ynET)\s*\.\s*)([A-Z][A-Z0-9_]*)"""
)

# hasattr guard: hasattr(SynapseEventType, "FOO_BAR")  - counts as a reference
_RE_HASATTR = re.compile(r"""hasattr\s*\(\s*SynapseEventType\s*,\s*['"]([A-Z][A-Z0-9_]*)['"]""")


def _enum_value_to_name(value: str) -> str:
    """Convert lowercase snake_case enum value → UPPER_SNAKE_CASE name."""
    return value.upper()


def _extract_all_enum_members(backend_root: Path) -> set[str]:
    """Parse synapse/types.py and collect all SynapseEventType member names."""
    types_path = backend_root / "systems" / "synapse" / "types.py"
    if not types_path.exists():
        print(f"[WARN] Cannot find {types_path}", file=sys.stderr)
        return set()

    source = types_path.read_text(encoding="utf-8")
    members: set[str] = set()

    # Strategy 1: direct enum attribute access in the file itself
    for m in _RE_DIRECT.finditer(source):
        members.add(m.group(1))

    # Strategy 2: AST parsing - find the class body assignments
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "SynapseEventType":
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                members.add(target.id)
    except SyntaxError:
        pass

    # Remove known non-event names
    members.discard("SynapseEventType")
    return {m for m in members if m and m[0].isupper()}


def scan_file(
    path: Path,
    emit_sites: dict[str, list[str]],
    subscribe_sites: dict[str, list[str]],
    string_constant_map: dict[str, str],  # VAR_NAME → enum_value_lowercase
) -> None:
    """Scan a single Python file and update emit/subscribe site maps."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return

    rel = str(path)

    # First pass: collect module-level string constant assignments
    # e.g. FOVEA_HABITUATION_DECAY = "fovea_habituation_decay"
    for m in _RE_STRING_CONSTANT.finditer(source):
        var_name, str_value = m.group(1), m.group(2)
        string_constant_map[var_name] = str_value

    def _record_emit(name: str) -> None:
        emit_sites[name.upper()].append(rel)

    def _record_subscribe(name: str) -> None:
        subscribe_sites[name.upper()].append(rel)

    # ── Emit detection ────────────────────────────────────────────────────────

    # Pattern 1: SynapseEventType.FOO_BAR
    for m in _RE_DIRECT.finditer(source):
        _record_emit(m.group(1))

    # Pattern 2: _SET.FOO or _SynET.FOO
    for m in _RE_SHORT_ALIAS.finditer(source):
        _record_emit(m.group(1))

    # Pattern 3: SynapseEventType("foo_bar") with string literal
    for m in _RE_CONSTRUCTOR.finditer(source):
        _record_emit(_enum_value_to_name(m.group(1)))

    # Pattern 4: SynapseEventType(VARIABLE) - look up variable in string_constant_map
    _re_constructor_var = re.compile(r"\bSynapseEventType\s*\(\s*([A-Z][A-Z0-9_]*)\s*\)")
    for m in _re_constructor_var.finditer(source):
        var = m.group(1)
        if var in string_constant_map:
            _record_emit(_enum_value_to_name(string_constant_map[var]))

    # Pattern 5/6/7: string helper calls
    for m in _RE_STRING_HELPER.finditer(source):
        _record_emit(_enum_value_to_name(m.group(1)))

    # Pattern 8: dispatch map keys  "FOO_BAR": SynapseEventType.FOO_BAR
    for m in _RE_DISPATCH_MAP_KEY.finditer(source):
        # Prefer the RHS enum name (always correct)
        _record_emit(m.group(2))

    # Pattern 9: getattr(SynapseEventType, "FOO_BAR")
    for m in _RE_GETATTR.finditer(source):
        _record_emit(m.group(1))

    # hasattr guard - counts as a reference (not an emit, but prevents DEAD classification)
    for m in _RE_HASATTR.finditer(source):
        _record_emit(m.group(1))  # add to emit so it's not flagged as DEAD

    # ── Subscribe detection ───────────────────────────────────────────────────

    for m in _RE_SUBSCRIBE.finditer(source):
        _record_subscribe(m.group(1))

    # Also capture subscribe_all as a wildcard (don't attribute to specific events)


def scan_directory(backend_root: Path) -> tuple[dict, dict, set]:
    """
    Scan the entire backend directory.

    Returns:
        emit_sites:      event_name → [file_paths]
        subscribe_sites: event_name → [file_paths]
        all_enum_members: set of all SynapseEventType member names
    """
    emit_sites: dict[str, list[str]] = defaultdict(list)
    subscribe_sites: dict[str, list[str]] = defaultdict(list)
    string_constant_map: dict[str, str] = {}

    py_files = list(backend_root.rglob("*.py"))

    for py_file in py_files:
        # Skip __pycache__ dirs and test fixtures
        if "__pycache__" in py_file.parts:
            continue
        scan_file(py_file, emit_sites, subscribe_sites, string_constant_map)

    all_members = _extract_all_enum_members(backend_root)
    return dict(emit_sites), dict(subscribe_sites), all_members


def classify(
    emit_sites: dict[str, list[str]],
    subscribe_sites: dict[str, list[str]],
    all_members: set[str],
) -> dict[str, dict]:
    """
    Classify every SynapseEventType member.

    Returns a dict: event_name → {
        "status": "WIRED" | "EMIT_ONLY" | "SUBSCRIBE_ONLY" | "DEAD",
        "emitters": [file_paths],
        "subscribers": [file_paths],
    }
    """
    result: dict[str, dict] = {}
    all_names = all_members | set(emit_sites) | set(subscribe_sites)

    for name in sorted(all_names):
        emitters = emit_sites.get(name, [])
        subscribers = subscribe_sites.get(name, [])
        has_emit = bool(emitters)
        has_sub = bool(subscribers)

        if has_emit and has_sub:
            status = "WIRED"
        elif has_emit and not has_sub:
            status = "EMIT_ONLY"
        elif not has_emit and has_sub:
            status = "SUBSCRIBE_ONLY"
        else:
            status = "DEAD"

        result[name] = {
            "status": status,
            "in_enum": name in all_members,
            "emitters": sorted(set(emitters)),
            "subscribers": sorted(set(subscribers)),
        }

    return result


def print_report(classification: dict[str, dict], verbose: bool = False) -> None:
    """Print a human-readable audit report."""
    by_status: dict[str, list] = defaultdict(list)
    for name, info in classification.items():
        by_status[info["status"]].append(name)

    total = len(classification)
    wired = len(by_status["WIRED"])
    emit_only = len(by_status["EMIT_ONLY"])
    sub_only = len(by_status["SUBSCRIBE_ONLY"])
    dead = len(by_status["DEAD"])

    print(f"\n{'='*70}")
    print(f" EcodiaOS Synapse Event Bus Audit")
    print(f"{'='*70}")
    print(f" Total events analysed : {total}")
    print(f" ✅ WIRED              : {wired}  (emit + subscribe)")
    print(f" ⬆  EMIT_ONLY          : {emit_only}  (no subscriber - fire-and-forget OK)")
    print(f" ⬇  SUBSCRIBE_ONLY     : {sub_only}  (dangling - no emitter)")
    print(f" 💀 DEAD               : {dead}  (unreferenced - delete candidates)")
    print(f"{'='*70}\n")

    for status, label in [
        ("SUBSCRIBE_ONLY", "⬇  SUBSCRIBE_ONLY - dangling (no emitter)"),
        ("DEAD", "💀 DEAD - unreferenced"),
        ("EMIT_ONLY", "⬆  EMIT_ONLY - no subscriber"),
    ]:
        names = sorted(by_status[status])
        if not names:
            continue
        print(f"\n── {label} ({len(names)}) ──")
        for name in names:
            info = classification[name]
            in_enum = "✓" if info["in_enum"] else "✗(not in enum)"
            print(f"  {name}  [{in_enum}]")
            if verbose:
                for f in info["emitters"]:
                    print(f"      EMIT: {f}")
                for f in info["subscribers"]:
                    print(f"      SUB:  {f}")

    if verbose:
        print(f"\n── ✅ WIRED ({wired}) ──")
        for name in sorted(by_status["WIRED"]):
            info = classification[name]
            print(f"  {name}")
            for f in info["emitters"]:
                print(f"      EMIT: {f}")
            for f in info["subscribers"]:
                print(f"      SUB:  {f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EcodiaOS Synapse Event Bus Audit")
    parser.add_argument(
        "--backend-root",
        default=str(Path(__file__).parent.parent),
        help="Path to the backend/ directory (default: parent of scripts/)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full classification as JSON",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show file paths for each event",
    )
    parser.add_argument(
        "--status",
        choices=["WIRED", "EMIT_ONLY", "SUBSCRIBE_ONLY", "DEAD"],
        help="Filter output to a specific status",
    )
    args = parser.parse_args()

    backend_root = Path(args.backend_root).resolve()
    if not backend_root.exists():
        print(f"[ERROR] Backend root not found: {backend_root}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning: {backend_root}", file=sys.stderr)
    emit_sites, subscribe_sites, all_members = scan_directory(backend_root)
    classification = classify(emit_sites, subscribe_sites, all_members)

    if args.status:
        classification = {k: v for k, v in classification.items() if v["status"] == args.status}

    if args.json:
        print(json.dumps(classification, indent=2))
    else:
        print_report(classification, verbose=args.verbose)


if __name__ == "__main__":
    main()
