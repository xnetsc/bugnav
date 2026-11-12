"""Live-case regression, 2026-11-12: a bout that opened fast on
open floor, then spent ~16 min in one chair pocket.  The rolling
window pace collapsed to ~0.2, the conservative yardstick silently
vanished from the digest (its gate demanded rolling >= 0.25), the
ETA priced ALL remaining area at the pocket pace, and the judge -
left with only the nominal 3.64x ratio - stopped a 21-minute bout
that in reality finished the whole home in 57 minutes.

Pinned here: the ETA's headline pace is the bout's OVERALL
demonstrated pace; the recent pace survives only as a labelled
pessimistic bound; the conservative expectation is built from the
overall pace and does not disappear when the rolling pace dips."""

import unittest

from bugnav.model import Point, RobotState, Sample
from bugnav.supervisor import SessionWatch, SupervisorConfig

WALL0 = 1_700_000_000.0


def _feed(w, t, x, y, batt):
    w.feed(Sample(t=t, pos=Point(x, y), state=RobotState.CLEANING),
           "monitor", wall=WALL0 + t, battery=batt, water_amount=2)


def _live_case_watch() -> SessionWatch:
    w = SessionWatch(SupervisorConfig())
    t = 0.0
    # Open floor: 80 samples x 10 s, a NEW cell every other sample
    # -> 40 cells = 10 m2 in 800 s.
    for i in range(80):
        t += 10.0
        _feed(w, t, (i // 2) * 0.5, 0.0, 95 - i // 16)
    # Pocket: 90 samples x 10 s grinding in visited cells, with a
    # trickle of new cells (4 x 0.25 m2 over 15 min).
    for i in range(90):
        t += 10.0
        if i % 22 == 21:
            _feed(w, t, 0.0, 5.0 + (i // 22) * 0.5, 90 - i // 15)
        else:
            _feed(w, t, (i % 2) * 0.5, 0.0, 90 - i // 15)
    return w


class TestProjectionHonesty(unittest.TestCase):
    def setUp(self):
        self.w = _live_case_watch()
        self.wall = WALL0 + 1700.0

    def test_overall_pace_exists_and_prices_the_mix(self):
        overall = self.w.bout_overall_pace_m2_per_min()
        self.assertIsNotNone(overall)
        # 11 m2 over ~28 cleaning minutes.
        self.assertTrue(0.30 <= overall <= 0.48, overall)
        rolling = self.w.rolling_progress_m2_per_min()
        self.assertTrue(rolling is None or rolling < 0.25, rolling)

    def test_eta_headline_is_overall_not_pocket_pace(self):
        eta = self.w.estimate_eta(self.wall, home_cells=83)
        self.assertIsNotNone(eta)
        self.assertEqual(eta["eta_assumes_progress_m2_per_min"],
                         self.w.bout_overall_pace_m2_per_min())
        if "projected_span_if_recent_pace_min" in eta:
            self.assertGreater(eta["projected_span_if_recent_pace_min"],
                               eta["projected_total_span_min"] * 1.3)

    def test_conservative_yardstick_survives_a_collapsed_window(self):
        d = self.w.digest(self.wall, "monitor", home_cells=83,
                          robot_state="cleaning")
        self.assertIn("expected_total_min_conservative", d)
        self.assertEqual(d["demonstrated_pace_m2_per_min"],
                         d["bout_overall_pace_m2_per_min"])
        ratio = d["projected_vs_conservative_ratio"]
        self.assertTrue(0.6 <= ratio <= 1.6, ratio)
        # And the span stays far inside this household's normal cap:
        # an efficiency ratio alone must never read as "not worth it".
        self.assertLess(d["projected_total_span_min"], 90.0)


if __name__ == "__main__":
    unittest.main()
