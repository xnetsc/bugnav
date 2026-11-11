"""The AI's clock basis is the ROBOT's reported time + timezone
(user 2026-11-11: 以扫地机自己上报的时间和时区为准，不可以写死).

parse_device_clock reads the firmware's response header ('ts' =
device epoch ms as a string, 'tzm' = offset minutes; live capture
2026-11-11: ts '1794374247626', tzm 480) and device_now projects
that basis forward on the monotonic clock."""

import time
import unittest

from bugnav.deebot_adapter import (
    AdapterConfig,
    DeebotAdapter,
    parse_device_clock,
)


class TestParseDeviceClock(unittest.TestCase):
    def test_live_header_shape(self):
        parsed = parse_device_clock(
            {"ts": "1794374247626", "tzm": 480}, mono_now=100.0)
        self.assertIsNotNone(parsed)
        epoch_s, mono, tz_hours = parsed
        self.assertAlmostEqual(epoch_s, 1794374247.626, places=3)
        self.assertEqual(mono, 100.0)
        self.assertEqual(tz_hours, 8.0)

    def test_missing_or_garbage_ts(self):
        self.assertIsNone(parse_device_clock(None, 0.0))
        self.assertIsNone(parse_device_clock({}, 0.0))
        self.assertIsNone(parse_device_clock({"ts": "soon"}, 0.0))

    def test_unsynced_device_epoch_rejected(self):
        # A device that has not time-synced yet reports a bogus epoch;
        # only absurd values are rejected - never host-clamped.
        self.assertIsNone(parse_device_clock({"ts": "0"}, 0.0))
        self.assertIsNone(parse_device_clock({"ts": "915148800000"}, 0.0))

    def test_clock_without_tzm_still_valid(self):
        parsed = parse_device_clock({"ts": "1794374247626"}, 5.0)
        self.assertIsNotNone(parsed)
        self.assertIsNone(parsed[2])


class TestDeviceNow(unittest.TestCase):
    def test_projects_from_captured_basis(self):
        adapter = DeebotAdapter(AdapterConfig(
            account_id="probe", device_name=""))
        self.assertIsNone(adapter.device_now())
        mono = time.monotonic()
        adapter._device_clock = (1_794_374_247.0, mono)
        now = adapter.device_now()
        self.assertIsNotNone(now)
        self.assertGreaterEqual(now, 1_794_374_247.0)
        self.assertLess(now, 1_794_374_248.0)

    def test_note_device_clock_from_raw_response(self):
        adapter = DeebotAdapter(AdapterConfig(
            account_id="probe", device_name=""))
        adapter._note_device_clock(
            {"resp": {"header": {"ts": "1794374247626", "tzm": 480}}})
        self.assertEqual(adapter.device_tz_offset_hours, 8.0)
        self.assertIsNotNone(adapter.device_now())
        # A headerless response must not clobber the basis.
        adapter._note_device_clock({})
        self.assertIsNotNone(adapter.device_now())


if __name__ == "__main__":
    unittest.main()
