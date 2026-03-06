from __future__ import annotations

from typing import Any

from systems.atune.logging_utils import debug_log
from systems.axon.executor import Executor


class DebugLogExecutor(Executor):
    """
    Executor for structured debug logging in the Atune system.
    Emits a labelled JSON dump to stdout.
    """

    action_type = "debug_log"

    async def execute(self, data: dict[str, Any]) -> None:
        """
        Execute debug log with provided data.

        Args:
            data: Dictionary containing 'label' and 'data' keys for logging
        """
        label = data.get('label', 'Unnamed Debug')
        log_data = data.get('data', {})
        debug_log(label, log_data)
