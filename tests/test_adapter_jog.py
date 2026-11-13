"""JOG executes as the remote-control 'move' command - one jog
step (backward / forward / spin / stop) with no destination,
answered code 0 by fw 1.61.0 at ~0.35 m per backward step.
"""

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bugnav.deebot_adapter import AdapterConfig, DeebotAdapter
from bugnav.model import Command, CommandKind


class _RecordingDevice:
    def __init__(self):
        self.commands = []

    async def execute_command(self, command):
        self.commands.append(command)
        return {"resp": {"body": {"code": 0},
                         "header": {"ts": "1794374247626", "tzm": 480}}}


def _adapter():
    adapter = DeebotAdapter(AdapterConfig(
        account_id="probe", device_name="", clean_command_v2=True))
    adapter._device = _RecordingDevice()
    return adapter


class TestJog(unittest.TestCase):
    def test_backward_jog_sends_move_command(self):
        adapter = _adapter()
        logs = []
        asyncio.run(adapter.execute(
            Command(CommandKind.JOG, act="backward"), logs.append))
        self.assertEqual(len(adapter._device.commands), 1)
        cmd = adapter._device.commands[0]
        self.assertEqual(cmd.NAME, "move")
        self.assertEqual(cmd._args, {"act": "backward"})
        self.assertTrue(any("move act=backward" in line for line in logs))

    def test_stop_jog_sends_move_stop(self):
        adapter = _adapter()
        asyncio.run(adapter.execute(
            Command(CommandKind.JOG, act="stop"), lambda line: None))
        cmd = adapter._device.commands[0]
        self.assertEqual(cmd.NAME, "move")
        self.assertEqual(cmd._args, {"act": "stop"})


if __name__ == "__main__":
    unittest.main()
