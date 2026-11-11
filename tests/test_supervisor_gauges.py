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
from bugnav.supervisor import (
    SYSTEM,
    SessionWatch,
    SupervisorConfig,
    home_consistency,
)


class TestReasonablenessFrame(unittest.TestCase):
    def test_prompt_carries_resume_and_routine_frame(self):
        # User (2026-11-11): every resume is a fresh decision; half a
        # day for one clean is unreasonable; judge with the current
        # time and ordinary daily routine, not duration alone.
        self.assertIn("续扫", SYSTEM)
        self.assertIn("半天", SYSTEM)
        self.assertIn("作息", SYSTEM)

    def test_prompt_carries_no_full_coverage_obligation(self):
        # User (2026-11-11): after 4-5 h with most of the planned area
        # done, end it - no obligation to leave nothing uncleaned.
        # This judgment belongs to the LLM, fed with enough info.
        self.assertIn("不遗漏", SYSTEM)
        self.assertIn("性价比", SYSTEM)
        self.assertIn("coverage_pct_of_home", SYSTEM)


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


class TestPaceStall(unittest.TestCase):
    def test_stall_when_no_new_coverage(self):
        w = SessionWatch(SupervisorConfig(enabled=True))
        t = 0.0
        for i in range(20):  # ~8 min jittering inside one cell
            t += 25.0
            w.feed(Sample(t=t, pos=Point(0.1 if i % 2 else 0.0, 0.0),
                          state=RobotState.CLEANING),
                   "monitor", wall=1_700_000_000.0 + t,
                   battery=100 - i // 4)
        self.assertTrue(w.pace_stalled())
        d = w.digest(1_700_000_000.0 + t, "monitor")
        self.assertTrue(d["pace_stalled"])
        self.assertEqual(d["rolling_new_m2"], 0.0)

    def test_no_stall_when_progressing(self):
        w = SessionWatch(SupervisorConfig(enabled=True))
        _sweep(w, 40, 0.0, 100, 90)
        self.assertFalse(w.pace_stalled())

    def test_prompt_speed_anomaly_is_cache_free(self):
        # User (2026-11-11): a cleaning-speed anomaly has nothing to
        # do with total area; it must be judged at all times, even
        # when the map cache is absent or mismatched.
        self.assertIn("与总面积无关", SYSTEM)
        self.assertIn("pace_stalled", SYSTEM)


class TestEta(unittest.TestCase):
    def test_eta_no_charges_when_energy_suffices(self):
        w = SessionWatch(SupervisorConfig(enabled=True))
        _sweep(w, 40, 0.0, 100, 90)
        d = w.digest(1_700_000_000.0 + 1000.0, "monitor",
                     home_cells=40 + 40)  # 10 m^2 remaining
        self.assertEqual(d["coverage_pct_of_home"], 50.0)
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


def _run_segment(w, t, minutes, cells, batt0, batt1, x_start):
    """Feed one cleaning segment (10 s samples, fresh cells along a
    row starting at cell x_start).  Returns the new t."""
    samples = int(minutes * 6)
    per = max(1, samples // max(1, cells))
    for i in range(samples):
        t += 10.0
        idx = min(cells - 1, i // per) if cells else 0
        batt = round(batt0 + (batt1 - batt0) * (i / max(1, samples - 1)))
        w.feed(Sample(t=t, pos=Point((x_start + idx) * 0.5, 0.0),
                      state=RobotState.CLEANING),
               "monitor", wall=1_700_000_000.0 + t, battery=batt)
    return t


def _dock(w, t, batt):
    w.feed(Sample(t=t + 10.0, pos=Point(0, 0), state=RobotState.RETURNING),
           "monitor", wall=1_700_000_000.0 + t + 10.0, battery=batt)
    w.feed(Sample(t=t + 20.0, pos=Point(0, 0), state=RobotState.DOCKED),
           "monitor", wall=1_700_000_000.0 + t + 20.0, battery=batt)
    return t + 20.0


class TestChargeThrash(unittest.TestCase):
    """User (2026-11-11): charged -> resumed -> flat again in no time,
    repeatedly, means the resume-level handling is broken.  Detect it
    from the segment ledger and force a high resume floor."""

    def test_thrash_detected_after_two_short_post_charge_segments(self):
        w = SessionWatch(SupervisorConfig(enabled=True))
        t = _run_segment(w, 0.0, 60, 80, 100, 18, 0)     # healthy first
        t = _dock(w, t, 18)
        t = _run_segment(w, t, 10, 12, 60, 15, 200)      # short, low end
        t = _dock(w, t, 15)
        t = _run_segment(w, t, 8, 10, 55, 12, 400)       # short again
        t = _dock(w, t, 12)
        self.assertEqual(len(w.segments), 3)
        self.assertFalse(w.segments[0]["after_charge"])
        self.assertTrue(w.segments[1]["after_charge"])
        self.assertTrue(w.short_cycle_thrash())
        d = w.digest(1_700_000_000.0 + t, "monitor")
        self.assertTrue(d["short_cycle_thrashing"])
        self.assertEqual(len(d["charge_segments"]), 3)

    def test_long_post_charge_segments_are_not_thrash(self):
        w = SessionWatch(SupervisorConfig(enabled=True))
        t = _run_segment(w, 0.0, 60, 80, 100, 18, 0)
        t = _dock(w, t, 18)
        t = _run_segment(w, t, 45, 60, 80, 20, 200)
        t = _dock(w, t, 20)
        t = _run_segment(w, t, 40, 55, 85, 22, 400)
        t = _dock(w, t, 22)
        self.assertFalse(w.short_cycle_thrash())

    def test_wash_return_segment_does_not_count(self):
        # A short segment ending at HIGH battery is a mop-wash return,
        # not a charge cycle - it must not feed the thrash detector.
        w = SessionWatch(SupervisorConfig(enabled=True))
        t = _run_segment(w, 0.0, 60, 80, 100, 70, 0)
        t = _dock(w, t, 70)
        t = _run_segment(w, t, 8, 10, 72, 65, 200)   # short, ends high
        t = _dock(w, t, 65)
        t = _run_segment(w, t, 8, 10, 60, 12, 400)   # short, ends low
        t = _dock(w, t, 12)
        self.assertFalse(w.short_cycle_thrash())

    def test_prompt_carries_thrash_logic(self):
        self.assertIn("短周期", SYSTEM)
        self.assertIn("short_cycle_thrashing", SYSTEM)


class TestGuaranteeSoc(unittest.TestCase):
    def test_guarantee_figure_in_digest(self):
        w = SessionWatch(SupervisorConfig(enabled=True))
        _sweep(w, 40, 0.0, 100, 90)  # drain ~1.0 %/m^2, 10 m^2 covered
        d = w.digest(1_700_000_000.0 + 1000.0, "monitor",
                     home_cells=80)  # remaining 10 m^2
        # reserve 8 + 10 m^2 x ~1.0 %/m^2 x margin 1.25 = ~20.5 -> 21
        self.assertGreaterEqual(d["resume_soc_to_guarantee_finish"], 19)
        self.assertLessEqual(d["resume_soc_to_guarantee_finish"], 23)

    def test_guarantee_scales_with_remaining(self):
        w = SessionWatch(SupervisorConfig(enabled=True))
        _sweep(w, 40, 0.0, 100, 90)
        small = w.soc_to_guarantee_finish(10.0)
        big = w.soc_to_guarantee_finish(60.0)
        self.assertLess(small, big)
        self.assertLessEqual(big, 100)

    def test_prompt_carries_guarantee_and_afternoon_anchor(self):
        # User (2026-11-11): with ~10% left and a charge unavoidable,
        # charge enough to GUARANTEE finishing, and do not resume
        # before that; a morning task still resuming at 16-17:00 is a
        # stop signal.  These are judgment logic for the LLM, fed with
        # computed figures - not hardcoded rules.
        self.assertIn("resume_soc_to_guarantee_finish", SYSTEM)
        self.assertIn("确保足够一次扫完", SYSTEM)
        self.assertIn("没充到这个数之前不要放行续扫", SYSTEM)
        self.assertIn("四五点", SYSTEM)
        self.assertIn("不是死规则", SYSTEM)


class TestHomeCacheConsistency(unittest.TestCase):
    """User (2026-11-11): robot taken to another home / new map /
    different rooms -> cached denominators must not be reported as
    fact, and a one-off foreign layout must not poison the cache."""

    CFG = SupervisorConfig(enabled=True)

    @staticmethod
    def _grid(x0, y0, n, m):
        return {(x0 + i, y0 + j) for i in range(n) for j in range(m)}

    def test_consistency_flags_foreign_bout(self):
        cached = self._grid(0, 0, 20, 10)          # 200 cells
        foreign = self._grid(100, 100, 10, 5)      # 50 cells, all new
        cons = home_consistency(
            foreign, cached, self.CFG)
        self.assertEqual(cons["novel_area_share"], 1.0)
        self.assertTrue(cons["suspect"])
        same = self._grid(0, 0, 10, 5)             # subset of cached
        cons2 = home_consistency(
            same, cached, self.CFG)
        self.assertFalse(cons2["suspect"])

    def test_digest_withholds_home_figures_when_suspect(self):
        w = SessionWatch(self.CFG)
        # Foreign sweep: 20 new cells in a far-away row.
        _sweep(w, 20, 0.0, 100, 95, x0=200)
        cached = self._grid(0, 0, 20, 10)
        # Force a substantial bout footprint for the judge: the watch
        # saw 20 cells; lower the bout minimum via cfg for this test.
        w.cfg = SupervisorConfig(enabled=True, home_min_bout_cells=10)
        d = w.digest(1_700_000_000.0, "monitor",
                     home_cells=len(cached), home_set=cached)
        self.assertIn("home_based_figures_withheld", d)
        self.assertNotIn("remaining_est_m2", d)
        self.assertNotIn("eta_finish", d)
        self.assertTrue(d["home_cache"]["suspect"])

    def test_digest_normal_when_consistent(self):
        w = SessionWatch(self.CFG)
        _sweep(w, 20, 0.0, 100, 95)  # cells (0..19, 0) in cached grid
        cached = self._grid(0, 0, 20, 10)
        d = w.digest(1_700_000_000.0, "monitor",
                     home_cells=len(cached), home_set=cached)
        self.assertIn("remaining_est_m2", d)
        self.assertFalse(d["home_cache"]["suspect"])

    def test_candidate_two_sightings_rule(self):
        from bugnav.supervisor import candidate_decision

        cached = self._grid(0, 0, 20, 10)
        foreign1 = self._grid(100, 100, 10, 5)
        foreign2 = self._grid(100, 100, 12, 5)  # overlaps foreign1
        home_bout = self._grid(0, 0, 10, 5)
        self.assertEqual(
            candidate_decision(foreign1, cached, None, self.CFG), "stash")
        self.assertEqual(
            candidate_decision(foreign2, cached, foreign1, self.CFG),
            "adopt")
        self.assertEqual(
            candidate_decision(home_bout, cached, foreign1, self.CFG),
            "merge")
        tiny = self._grid(500, 500, 3, 3)
        self.assertEqual(
            candidate_decision(tiny, cached, None, self.CFG), "none")


class TestHomeProfile(unittest.TestCase):
    def test_digest_carries_home_profile(self):
        cfg = SupervisorConfig(
            enabled=True,
            home_note="public areas + study only, no bedrooms",
            normal_session_max_s=3 * 3600)
        w = SessionWatch(cfg)
        _sweep(w, 10, 0.0, 100, 97)
        d = w.digest(1_700_000_000.0, "monitor")
        self.assertEqual(d["home_note"],
                         "public areas + study only, no bedrooms")
        self.assertEqual(d["normal_session_max_min"], 180)

    def test_prompt_defers_to_home_note(self):
        # User (2026-11-11): this home cleans living/dining/kitchen/
        # study only and normally finishes within 3 h - the household
        # yardstick outranks the generic one.
        self.assertIn("home_note", SYSTEM)

    def test_return_to_dock_command_exists(self):
        from bugnav.model import CommandKind

        self.assertEqual(CommandKind.RETURN_TO_DOCK.value, "return_to_dock")
        from deebot_client.commands.json import Charge  # noqa: F401


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
