"""
Root conftest — pre-mock optional external dependencies.

Some clients (e.g. wallet.py) import libraries that aren't installed
in the test environment (cdp). We register lightweight stubs in
sys.modules so the import chain doesn't break during collection.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

# ── cdp (Coinbase Developer Platform) ─────────────────────────────────
# wallet.py does `from cdp import CdpClient` at module level.
# Stub the entire cdp package so tests that transitively import wallet
# don't fail with ModuleNotFoundError.
if "cdp" not in sys.modules:
    _cdp = ModuleType("cdp")
    _cdp.CdpClient = MagicMock()  # type: ignore[attr-defined]

    _cdp_evm_account = ModuleType("cdp.evm_account")
    _cdp_evm_account.EvmAccount = MagicMock()  # type: ignore[attr-defined]

    _cdp_evm_smart = ModuleType("cdp.evm_smart_account")
    _cdp_evm_smart.EvmSmartAccount = MagicMock()  # type: ignore[attr-defined]

    sys.modules["cdp"] = _cdp
    sys.modules["cdp.evm_account"] = _cdp_evm_account
    sys.modules["cdp.evm_smart_account"] = _cdp_evm_smart
