"""Rolling drain / ETA / virtual water gauge tests (2026-11-11 batch).

User requirements pinned here:
  * LIVE rolling consumption (recent ~15 min), not just bout averages;
  * a projected finish time that includes the charge cycles implied by
    the remaining energy shortfall;
  * a virtual water gauge: 100% after a real station stay (wash +
    refill), draining by spray amount x newly mopped area, NOT reset
    by a short pass-through dock touch.
"""

import unittest

from bugnav.model import Point, RobotState, Sample
from bugnav.supervisor import SYSTEM, SessionWatch, SupervisorConfig


class TestReasonablenessFrame(unittest.TestCase):
    def test_prompt_carries_resume_and_routine_frame(self):
        # User (2026-11-11): every resume is a fresh decision; half a
        # day for one clean is unreasonable; judge with the current
        # time and ordinary daily routine, not duration alone.
        self.assertIn("续扫", SYSTEM)
        self.assertIn("半天", SYSTEM)
        self.assertIn("作息", SYSTEM)


def _sweep(w: SessionWatch, n: int, t0: float, batt0: float,
           batt1: float, water_amount: int | None = 2,
           x0: int = 0) -> float:
    """Feed n cleaning samples, each entering a NEW 0.5 m cell, with
    battery interpolating batt0 -> batt1.  Returns the last t.
    Sample period is 25 s so the rolling span grows past 5 min fast."""
    t = t0
    for i in range(n):
        t += 25.0
        batt = round(batt0 + (batt1 - batt0) * (i / max(1, n - 1)))
        w.feed(Sample(t=t, pos=Point((x0 + i) * 0.5, 0.0),
                      state=RobotState.CLEANING),
               "monitor", wall=1_700_000_000.0 + t, battery=batt,
               water_amount=water_amount)
    return t


class TestRollingGauges(unittest.TestCase):
    def test_rolling_drain_and_progress(self):
        w = SessionWatch(SupervisorConfig(enabled=True))
        # 40 new cells = 10 m^2 over ~16 min; battery 100 -> 90.
        _sweep(w, 40, 0.0, 100, 90)
        self.assertAlmostEqual(w.rolling_drain_pct_per_m2(), 1.0, delta=0.15)
        prog = w.rolling_progress_m2_per_min()
        self.assertIsNotNone(prog)
        self.assertAlmostEqual(prog, 10.0 / (975.0 / 60.0), delta=0.1)

    def test_rolling_needs_span_and_data(self):
        w = SessionWatch(SupervisorConfig(enabled=True))
        _sweep(w, 3, 0.0, 100, 99)  # ~1 min only
        self.assertIsNone(w.rolling_drain_pct_per_m2())
        self.assertIsNone(w.rolling_progress_m2_per_min())


class TestEta(unittest.TestCase):
    def test_eta_no_charges_when_energy_suffices(self):
        w = SessionWatch(SupervisorConfig(enabled=True))
        _sweep(w, 40, 0.0, 100, 90)
        d = w.digest(1_700_000_000.0 + 1000.0, "monitor",
                     home_cells=40 + 40)  # 10 m^2 remaining
        self.assertEqual(d["eta_charges_left"], 0)
        self.assertGreater(d["eta_minutes_left"], 5)
        self.assertLess(d["eta_minutes_left"], 40)
        self.assertIn("eta_finish", d)

    def test_eta_counts_charge_cycles_on_shortfall(self):
        w = SessionWatch(SupervisorConfig(enabled=True))
        _sweep(w, 40, 0.0, 30, 20)  # ends at 20% battery
        d = w.digest(1_700_000_000.0 + 1000.0, "monitor",
                     home_cells=40 + 400)  # 100 m^2 remaining
        # drain ~1.0 %/m^2 -> need 100%, usable 12%, per cycle 82%
        # -> 2 more charge cycles, each adding serious minutes.
        self.assertEqual(d["eta_charges_left"], 2)
        self.assertGreater(d["eta_minutes_left"], 200)

    def test_eta_zero_when_nothing_remaining(self):
        w = SessionWatch(SupervisorConfig(enabled=True))
        _sweep(w, 40, 0.0, 100, 90)
        eta = w.estimate_eta(1_700_000_000.0, home_cells=40)
        self.assertEqual(eta["eta_minutes_left"], 0.0)
        self.assertEqual(eta["eta_charges_left"], 0)


class TestWaterGauge(unittest.TestCase):
    def test_drains_by_amount_and_area(self):
        w = SessionWatch(SupervisorConfig(enabled=True))
        # 20 new cells = 5 m^2 at amount 3 (1.8 %/m^2) -> 91.0%
        _sweep(w, 20, 0.0, 100, 95, water_amount=3)
        self.assertAlmostEqual(w.water_pct_est(), 91.0, delta=0.6)

    def test_starts_full_and_short_dock_does_not_reset(self):
        cfg = SupervisorConfig(enabled=True)  # reset needs >= 120 s docked
        w = SessionWatch(cfg)
        t = _sweep(w, 20, 0.0, 100, 95, water_amount=3)
        self.assertLess(w.water_pct_est(), 100.0)
        # 60 s docked: pass-through, no reset.
        for k in range(3):
            w.feed(Sample(t=t + 25.0 * (k + 1), pos=Point(0, 0),
                          state=RobotState.DOCKED),
                   "monitor", wall=1_700_001_000.0, battery=50)
        self.assertLess(w.water_pct_est(), 100.0)

    def test_real_station_stay_resets_to_full(self):
        cfg = SupervisorConfig(enabled=True)
        w = SessionWatch(cfg)
        t = _sweep(w, 20, 0.0, 100, 95, water_amount=3)
        self.assertLess(w.water_pct_est(), 100.0)
        # Stay docked for 200 s of sample time: wash + refill happened.
        tt = t
        for _ in range(8):
            tt += 25.0
            w.feed(Sample(t=tt, pos=Point(0, 0), state=RobotState.DOCKED),
                   "monitor", wall=1_700_001_000.0 + tt, battery=50)
        self.assertEqual(w.water_pct_est(), 100.0)
        d = w.digest(1_700_001_000.0 + tt, "monitor")
        self.assertEqual(d["water_pct_est"], 100.0)
        self.assertEqual(d["water_amount"], 3)


class TestAdapterWaterWiring(unittest.TestCase):
    def test_nazy54_water_refresh_includes_get_water_info(self):
        from deebot_client.events import WaterInfoEvent
        from deebot_client.commands.json import GetWaterInfo

        from bugnav.deebot_adapter import AdapterConfig, DeebotAdapter

        adapter = DeebotAdapter(AdapterConfig(country="CN", account_id="x"))
        adapter._register_hardware_aliases(lambda msg: None)

        from deebot_client.hardware.deebot import DEVICES
        cmds = DEVICES["nazy54"].capabilities.get_refresh_commands(
            WaterInfoEvent)
        self.assertTrue(any(isinstance(c, GetWaterInfo) for c in cmds))


if __name__ == "__main__":
    unittest.main()
