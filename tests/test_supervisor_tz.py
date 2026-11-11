"""The supervisor's clock strings must follow cfg.tz_offset_hours,
never the process-local zone (regression: 2026-11-11 the container's
UTC clock made the judge read 11:01 local as 03:00 'deep night' and
stop a morning session).  These assertions hold under ANY machine
TZ because the code must not consult localtime at all."""

import unittest

from bugnav.supervisor import (
    SessionWatch,
    SupervisorConfig,
    adopt_device_tz,
)


class TestSupervisorClock(unittest.TestCase):
    def test_digest_now_uses_config_offset(self):
        w = SessionWatch(SupervisorConfig(enabled=True))
        d = w.digest(0.0, "monitor")  # epoch 0 = 1970-01-01 00:00 UTC
        self.assertEqual(d["now"], "08:00（东八区）")  # stated WITH its zone
        self.assertEqual(d["bout_started"], "08:00")

    def test_digest_now_honours_custom_offset(self):
        w = SessionWatch(SupervisorConfig(enabled=True,
                                          tz_offset_hours=0.0))
        self.assertEqual(w.digest(0.0, "monitor")["now"], "00:00（UTC）")
        w2 = SessionWatch(SupervisorConfig(enabled=True,
                                           tz_offset_hours=-5.0))
        self.assertEqual(w2.digest(0.0, "monitor")["now"], "19:00（西五区）")

    def test_adopt_device_tz(self):
        cfg = SupervisorConfig(enabled=True)
        adopted = adopt_device_tz(cfg, config_had_key=False,
                                  device_hours=8.0)
        self.assertTrue(adopted)
        self.assertEqual(cfg.tz_offset_hours, 8.0)

    def test_adopt_device_tz_respects_explicit_config(self):
        cfg = SupervisorConfig(enabled=True, tz_offset_hours=1.0)
        adopted = adopt_device_tz(cfg, config_had_key=True,
                                  device_hours=8.0)
        self.assertFalse(adopted)
        self.assertEqual(cfg.tz_offset_hours, 1.0)

    def test_adopt_device_tz_no_device_value(self):
        cfg = SupervisorConfig(enabled=True)
        adopted = adopt_device_tz(cfg, config_had_key=False,
                                  device_hours=None)
        self.assertFalse(adopted)
        self.assertEqual(cfg.tz_offset_hours, 8.0)  # default stands

    def test_eta_finish_uses_config_offset(self):
        w = SessionWatch(SupervisorConfig(enabled=True))
        eta = w.estimate_eta(0.0, home_cells=0)  # nothing remaining
        self.assertIsNotNone(eta)
        self.assertEqual(eta["eta_finish"], "08:00")


if __name__ == "__main__":
    unittest.main()


class TestPrecomputedTimes(unittest.TestCase):
    """The judge receives READY-MADE time differences, never raw
    clock pairs it would have to subtract itself (user 2026-11-11:
    给 AI 时要给算好的时间，不要让 AI 算时间)."""

    def test_span_over_normal_is_precomputed(self):
        from bugnav.model import Point, RobotState, Sample

        w = SessionWatch(SupervisorConfig(
            enabled=True, normal_session_max_s=3600.0))
        t = 0.0
        for i in range(40):
            t += 25.0
            w.feed(Sample(t=t, pos=Point(i * 0.5, 0.0),
                          state=RobotState.CLEANING),
                   "monitor", wall=1_700_000_000.0 + t,
                   battery=100 - i // 4, water_amount=2)
        d = w.digest(1_700_000_000.0 + t + 25.0, "monitor",
                     home_cells=200)
        self.assertIn("projected_total_span_min", d)
        self.assertIn("span_over_normal_min", d)
        self.assertAlmostEqual(
            d["span_over_normal_min"],
            round(d["projected_total_span_min"]
                  - d["normal_session_max_min"], 1),
            places=1)

    def test_prompt_forbids_clock_arithmetic(self):
        from bugnav.supervisor import SYSTEM

        self.assertIn("时间纪律", SYSTEM)
        self.assertIn("严禁用钟点做加减", SYSTEM)
        self.assertIn("span_over_normal_min", SYSTEM)
