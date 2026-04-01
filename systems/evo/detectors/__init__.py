"""
EcodiaOS - Evo Detector Package

This __init__.py makes the detectors/ directory a proper Python package.
The core detector classes live in the sibling detectors.py flat module.
Since a package directory shadows a flat module of the same name, we
must re-export everything here so existing imports continue to work.

NOTE: Python prioritises the package (directory with __init__.py) over
the flat module. All public symbols from detectors.py are re-exported
below so that `from systems.evo.detectors import PatternDetector` works.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

# Load the flat detectors.py module explicitly via file path
_flat_path = Path(__file__).resolve().parent.parent / "detectors.py"
_spec = importlib.util.spec_from_file_location(
    "systems.evo._detectors_flat", str(_flat_path)
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export public API
PatternDetector = _mod.PatternDetector
CooccurrenceDetector = _mod.CooccurrenceDetector
SequenceDetector = _mod.SequenceDetector
TemporalDetector = _mod.TemporalDetector
AffectPatternDetector = _mod.AffectPatternDetector
build_default_detectors = _mod.build_default_detectors

# Re-export PatternCandidate / PatternType so that lazy imports in service.py
# (`from systems.evo.detectors import PatternCandidate, PatternType`) resolve
# correctly even though these are defined in systems.evo.types.
from systems.evo.types import PatternCandidate, PatternType  # noqa: E402

__all__ = [
    "PatternDetector",
    "CooccurrenceDetector",
    "SequenceDetector",
    "TemporalDetector",
    "AffectPatternDetector",
    "build_default_detectors",
    "PatternCandidate",
    "PatternType",
]
