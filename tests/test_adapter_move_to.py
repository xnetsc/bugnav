"""MOVE_TO executes as clean_V2 mapPoint - the directed-movement
type fw 1.61.0 actually implements (probed live 2026-11-12:
customArea/spotArea are rejected 20011, mapPoint returns code 0
and the robot drives to the point, in the same raw frame GetPos
reports)."""

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bugnav.deebot_adapter import AdapterConfig, DeebotAdapter
from bugnav.model import Command, CommandKind, Point


class _RecordingDevice:
    def __init__(self):
        self.commands = []

    async def execute_command(self, command):
        self.commands.append(command)
        return {"resp": {"body": {"code": 0},
                         "header": {"ts": "1794374247626", "tzm": 480}}}


def _adapter(**cfg_kw):
    adapter = DeebotAdapter(AdapterConfig(
        account_id="probe", device_name="", clean_command_v2=True,
        **cfg_kw))
    adapter._device = _RecordingDevice()
    return adapter


class TestMoveTo(unittest.TestCase):
    def test_mappoint_args_and_frame(self):
        adapter = _adapter()
        logs = []
        asyncio.run(adapter.execute(
            Command(CommandKind.MOVE_TO, point=Point(1.5, -2.5)),
            logs.append))
        self.assertEqual(len(adapter._device.commands), 1)
        cmd = adapter._device.commands[0]
        self.assertEqual(cmd.NAME, "clean_V2")
        from deebot_client.models import CleanAction
        self.assertEqual(
            cmd._get_args(CleanAction.START),
            {"act": "start",
             "content": {"type": "mapPoint", "value": "1500,-2500"}})
        self.assertTrue(any("mapPoint" in line for line in logs))

    def test_to_raw_point_rounding_and_origin(self):
        adapter = _adapter(origin_raw_x=100.0, origin_raw_y=-50.0)
        self.assertEqual(adapter.to_raw_point(Point(0.0, 0.0)), "100,-50")
        self.assertEqual(adapter.to_raw_point(Point(-0.0004, 0.0004)),
                         "100,-50")

    def test_start_auto_sends_raw_fresh_start(self):
        # A mapPoint session leaves the robot paused with nothing
        # to resume; the stock START would be rewritten by the
        # library into a bare resume (a no-op, probed live).  The
        # adapter must send the raw start form untouched.
        adapter = _adapter()
        logs = []
        asyncio.run(adapter.execute(
            Command(CommandKind.START_AUTO), logs.append))
        self.assertEqual(len(adapter._device.commands), 1)
        cmd = adapter._device.commands[0]
        from deebot_client.models import CleanAction
        self.assertEqual(
            cmd._get_args(CleanAction.RESUME),
            {"act": "start", "content": {"type": "auto"}})
        self.assertTrue(any("fresh auto" in line for line in logs))

    def test_move_to_rejected_on_v1_config(self):
        adapter = _adapter()
        adapter.cfg.clean_command_v2 = False
        with self.assertRaises(RuntimeError):
            asyncio.run(adapter.execute(
                Command(CommandKind.MOVE_TO, point=Point(0.0, -1.0)),
                lambda msg: None))
        self.assertEqual(adapter._device.commands, [])


if __name__ == "__main__":
    unittest.main()
