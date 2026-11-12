"""Tolerant parsers for the T80S fw 1.61.0 payload shapes, using
EXACT captured payloads (the pinned library's strict parsers died
on both), plus the LLM JSON-mode payload."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bugnav.firmware_parsers import install_tolerant_parsers
from bugnav.llm import build_payload

install_tolerant_parsers()

from deebot_client.commands.json.life_span import GetLifeSpan  # noqa: E402
from deebot_client.commands.json.water_info import GetWaterInfo  # noqa: E402
from deebot_client.events import (  # noqa: E402
    LifeSpan,
    LifeSpanEvent,
    SweepType,
    WaterAmount,
    WaterInfoEvent,
)
from deebot_client.message import HandlingState  # noqa: E402


class _Bus:
    def __init__(self):
        self.events = []

    def notify(self, event):
        self.events.append(event)


class TestWaterInfo(unittest.TestCase):
    def test_firmware_custom_amount_shape(self):
        bus = _Bus()
        result = GetWaterInfo._handle_body_data_dict(
            bus, {"customAmount": 30, "enable": 1, "mopCount": 2,
                  "sideMop": 0, "sweepType": 1, "type": 1})
        self.assertEqual(result.state, HandlingState.SUCCESS)
        self.assertEqual(len(bus.events), 1)
        ev = bus.events[0]
        self.assertIsInstance(ev, WaterInfoEvent)
        self.assertEqual(ev.amount, WaterAmount.MEDIUM)  # 30 -> band 2
        self.assertEqual(ev.sweep_type, SweepType.STANDARD)
        self.assertTrue(ev.mop_attached)

    def test_classic_amount_shape_still_works(self):
        bus = _Bus()
        result = GetWaterInfo._handle_body_data_dict(
            bus, {"amount": 3, "enable": 0})
        self.assertEqual(result.state, HandlingState.SUCCESS)
        self.assertEqual(bus.events[0].amount, WaterAmount.HIGH)
        self.assertFalse(bus.events[0].mop_attached)


class TestLifeSpan(unittest.TestCase):
    def test_unknown_components_skipped_not_fatal(self):
        bus = _Bus()
        payload = [
            {"left": 7829, "total": 18000, "type": "brush"},
            {"left": 0, "total": 100, "type": "cleaningSolution"},
            {"left": 1319, "total": 3000, "type": "dustBag"},
            {"left": 0, "total": 3600, "type": "sewageBox"},
            {"left": 7361, "total": 9000, "type": "sideBrush"},
        ]
        result = GetLifeSpan._handle_body_data_list(bus, payload)
        self.assertEqual(result.state, HandlingState.SUCCESS)
        types = [e.type for e in bus.events
                 if isinstance(e, LifeSpanEvent)]
        self.assertEqual(types, [LifeSpan.BRUSH, LifeSpan.DUST_BAG,
                                 LifeSpan.SIDE_BRUSH])


class TestLLMJsonMode(unittest.TestCase):
    def test_payload_enforces_json_object(self):
        p = build_payload("qwen-plus", [{"role": "user", "content": "x"}])
        self.assertEqual(p["response_format"], {"type": "json_object"})

    def test_payload_fallback_without_json_mode(self):
        p = build_payload("m", [], json_mode=False)
        self.assertNotIn("response_format", p)


if __name__ == "__main__":
    unittest.main()


class TestP2PLifeSpanTolerance(unittest.TestCase):
    """A mirrored resetLifeSpan naming a consumable newer than the
    pinned enum (e.g. 'cleaningSolution') must reconstruct instead
    of dying in create_from_mqtt."""

    def test_unknown_lifespan_reconstructs(self):
        from deebot_client.commands.json.life_span import ResetLifeSpan

        cmd = ResetLifeSpan.create_from_mqtt({"type": "cleaningSolution"})
        self.assertEqual(cmd._args, {"type": "cleaningSolution"})

    def test_known_lifespan_still_real_enum(self):
        from deebot_client.commands.json.life_span import ResetLifeSpan

        cmd = ResetLifeSpan.create_from_mqtt({"type": "brush"})
        self.assertEqual(cmd._args, {"type": "brush"})


class TestMapSetNoiseFilter(unittest.TestCase):
    """The firmware permanently answers getMapSet with 20003; the
    library's per-dispatch 'Could not parse getMapSet' warnings
    (tracebacks included) are pure noise and get filtered, while
    every other library record must still pass."""

    @staticmethod
    def _record(logger_name, msg, args=()):
        import logging

        return logging.LogRecord(logger_name, logging.WARNING,
                                 __file__, 1, msg, args, None)

    def _filters_of(self, logger_name):
        import logging

        from bugnav.firmware_parsers import _UnsupportedMapSetFilter

        return [f for f in logging.getLogger(logger_name).filters
                if isinstance(f, _UnsupportedMapSetFilter)]

    def test_filter_attached_to_both_library_loggers(self):
        self.assertEqual(len(self._filters_of("deebot_client.message")), 1)
        self.assertEqual(len(self._filters_of("deebot_client.command")), 1)

    def test_getmapset_warning_dropped(self):
        f = self._filters_of("deebot_client.message")[0]
        rec = self._record("deebot_client.message",
                           "Could not parse %s: %s",
                           ("getMapSet", {"code": 20003}))
        self.assertFalse(f.filter(rec))

    def test_other_records_pass(self):
        f = self._filters_of("deebot_client.message")[0]
        other_parse = self._record("deebot_client.message",
                                   "Could not parse %s: %s",
                                   ("getPos", {"code": 1}))
        self.assertTrue(f.filter(other_parse))
        unrelated = self._record("deebot_client.message",
                                 "connection lost, retrying")
        self.assertTrue(f.filter(unrelated))
