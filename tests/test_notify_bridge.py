"""Notification-bridge filter tests.

The bridge (deploy/mac/notify-bridge.py) runs on the user's Mac under
launchd, so only its pure line-filter logic is tested here: which
events.jsonl lines become notifications, and the no-history-replay
rule on first install.
"""

import importlib.util
import json
import os
import unittest

_PATH = os.path.join(os.path.dirname(__file__), "..", "deploy", "mac",
                     "notify-bridge.py")
_spec = importlib.util.spec_from_file_location("notify_bridge", _PATH)
bridge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bridge)


def _line(ts: float, msg: str) -> str:
    return json.dumps({"ts": ts, "msg": msg})


class TestExtract(unittest.TestCase):
    def test_only_notify_lines_newer_than_watermark(self):
        lines = [
            _line(100.0, "NOTIFY: supervisor: stop"),
            _line(101.0, "plain log line"),
            _line(102.0, "NOTIFY: heads up"),
            "not json at all",
            _line(99.0, "NOTIFY: old one"),
        ]
        got = bridge.extract_notifications(lines, after_ts=99.5)
        self.assertEqual([t for _, t in got],
                         ["NOTIFY: supervisor: stop", "NOTIFY: heads up"])

    def test_empty_input(self):
        self.assertEqual(bridge.extract_notifications([], 0.0), [])


class TestState(unittest.TestCase):
    def test_missing_state_file_means_first_run(self):
        old = bridge.STATE_PATH
        bridge.STATE_PATH = "/nonexistent-dir-xyz/state.json"
        try:
            self.assertIsNone(bridge.load_state())
        finally:
            bridge.STATE_PATH = old


if __name__ == "__main__":
    unittest.main()
