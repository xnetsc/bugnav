"""The supervisor's clock strings must follow cfg.tz_offset_hours,
never the process-local zone (regression: 2026-11-11 the container's
UTC clock made the judge read 11:01 local as 03:00 'deep night' and
stop a morning session).  These assertions hold under ANY machine
TZ because the code must not consult localtime at all."""

import unittest

from bugnav.supervisor import SessionWatch, SupervisorConfig


class TestSupervisorClock(unittest.TestCase):
    def test_digest_now_uses_config_offset(self):
        w = SessionWatch(SupervisorConfig(enabled=True))
        d = w.digest(0.0, "monitor")  # epoch 0 = 1970-01-01 00:00 UTC
        self.assertEqual(d["now"], "08:00")  # default offset +8
        self.assertEqual(d["bout_started"], "08:00")

    def test_digest_now_honours_custom_offset(self):
        w = SessionWatch(SupervisorConfig(enabled=True,
                                          tz_offset_hours=0.0))
        self.assertEqual(w.digest(0.0, "monitor")["now"], "00:00")
        w2 = SessionWatch(SupervisorConfig(enabled=True,
                                           tz_offset_hours=-5.0))
        self.assertEqual(w2.digest(0.0, "monitor")["now"], "19:00")

    def test_eta_finish_uses_config_offset(self):
        w = SessionWatch(SupervisorConfig(enabled=True))
        eta = w.estimate_eta(0.0, home_cells=0)  # nothing remaining
        self.assertIsNotNone(eta)
        self.assertEqual(eta["eta_finish"], "08:00")


if __name__ == "__main__":
    unittest.main()
