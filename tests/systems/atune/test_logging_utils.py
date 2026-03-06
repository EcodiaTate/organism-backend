from __future__ import annotations

import json

from systems.atune.logging_utils import debug_log


def test_debug_log(capsys):
    """Test that debug_log emits structured JSON to stdout."""
    test_label = "test_event"
    test_data = {"key1": "value1", "key2": 42}

    debug_log(test_label, test_data)

    captured = capsys.readouterr()
    assert captured.out.startswith(f'[DEBUG] {test_label}: ')

    # Verify JSON is valid and contains expected data
    json_part = captured.out.split(': ', 1)[1].strip()
    parsed_data = json.loads(json_part)
    assert parsed_data == test_data
