"""Tests for the LLM layers: strategist plans, supervisor verdicts,
charge planning, and the controller's consult flow."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bugnav.chargeplan import ChargeModel, plan_resume
from bugnav.controller import BugNavController, ControllerConfig, Phase
from bugnav.geometry import Point
from bugnav.model import RobotState, Sample
from bugnav.strategist import parse_plan
from bugnav.supervisor import SessionWatch, SupervisorConfig, parse_verdict


class TestParsePlan(unittest.TestCase):
    def test_valid_plan_with_fences(self):
        plan = parse_plan(
            '好的：\n```json\n{"actions": [{"type": "try_angle", '
            '"theta_deg": 175}, {"type": "adjust", "step_angle_deg": 15}], '
            '"reason": "开口可能在背面"}\n```')
        self.assertIsNotNone(plan)
        self.assertEqual(plan.angles, [175.0])
        self.assertEqual(plan.step_angle_deg, 15.0)
        self.assertIn("背面", plan.reason)

    def test_out_of_range_is_clamped_not_fatal(self):
        plan = parse_plan('{"actions": [{"type": "adjust", '
                          '"step_angle_deg": 500, "staging_clearance_m": 9}]}')
        self.assertEqual(plan.step_angle_deg, 90.0)
        self.assertEqual(plan.staging_clearance_m, 1.0)

    def test_angles_capped_at_three(self):
        acts = ",".join(
            '{"type": "try_angle", "theta_deg": %d}' % (i * 40)
            for i in range(4))
        plan = parse_plan('{"actions": [%s]}' % acts)
        self.assertEqual(len(plan.angles), 3)

    def test_garbage_degrades_to_none(self):
        self.assertIsNone(parse_plan("我觉得应该往南边试试"))
        self.assertIsNone(parse_plan('{"actions": []}'))
        self.assertIsNone(parse_plan(""))


class TestParseVerdict(unittest.TestCase):
    def test_stop_with_message(self):
        v = parse_verdict('{"action": "stop_session", "message": "扫太久了", '
                           '"reason": "反复清扫"}')
        self.assertEqual(v.action, "stop_session")
        self.assertEqual(v.message, "扫太久了")

    def test_resume_pct_validated(self):
        v = parse_verdict('{"action": "continue", "resume_charge_pct": 55}')
        self.assertEqual(v.resume_charge_pct, 55)
        v = parse_verdict('{"action": "continue", "resume_charge_pct": 3}')
        self.assertIsNone(v.resume_charge_pct)

    def test_garbage_is_continue(self):
        self.assertEqual(parse_verdict("blah").action, "continue")
        self.assertEqual(parse_verdict('{"action": "explode"}').action,
                         "continue")


class TestChargePlan(unittest.TestCase):
    def test_small_remaining_means_small_charge(self):
        plan = plan_resume(20, 15.0, ChargeModel())
        # 15 m^2 needs ~8% + margin: charging to 100% would be wasteful.
        self.assertLess(plan["resume_soc"], 60)
        self.assertEqual(plan["est_charges"], 1)

    def test_huge_remaining_charges_full(self):
        plan = plan_resume(15, 300.0, ChargeModel())
        self.assertEqual(plan["resume_soc"], 100)

    def test_charge_minutes_nonlinear(self):
        m = ChargeModel()
        low = m.charge_minutes(10, 40)   # fast band
        high = m.charge_minutes(70, 100)  # taper band
        self.assertLess(low, high)


class TestSupervisorWatch(unittest.TestCase):
    def _feed_clean(self, w: SessionWatch, n: int, t0: float = 0.0,
                    x0: float = 0.0) -> float:
        t = t0
        for i in range(n):
            t += 2.5
            w.feed(Sample(t=t, pos=Point(x0 + i * 0.1, 0.0),
                          state=RobotState.CLEANING),
                   "monitor", wall=1_700_000_000.0 + t, battery=90)
        return t

    def test_review_due_after_interval(self):
        cfg = SupervisorConfig(enabled=True, review_interval_s=60)
        w = SessionWatch(cfg)
        self._feed_clean(w, 10)  # 25 s of cleaning
        self.assertFalse(w.due())
        self._feed_clean(w, 30, t0=25.0, x0=1.0)
        self.assertTrue(w.due())

    def test_charge_review_on_mid_bout_dock(self):
        cfg = SupervisorConfig(enabled=True, review_interval_s=99999)
        w = SessionWatch(cfg)
        t = self._feed_clean(w, 20)
        w.feed(Sample(t=t + 2.5, pos=Point(0, 0), state=RobotState.RETURNING),
               "monitor", wall=1_700_000_100.0, battery=18)
        self.assertFalse(w.due())
        w.feed(Sample(t=t + 5.0, pos=Point(0, 0), state=RobotState.DOCKED),
               "monitor", wall=1_700_000_102.0, battery=18)
        self.assertTrue(w.charge_review_pending)
        self.assertTrue(w.due())

    def test_drain_learning(self):
        cfg = SupervisorConfig(enabled=True)
        w = SessionWatch(cfg)
        t = 0.0
        # 100 samples * 0.5 m steps = a 10x5 m sweep: 40 cells * 0.25 = 10 m^2
        for i in range(100):
            t += 2.5
            x = (i % 20) * 0.5
            y = (i // 20) * 0.5
            batt = 90 - (i * 10) // 99  # drains 90 -> 80
            w.feed(Sample(t=t, pos=Point(x, y), state=RobotState.CLEANING),
                   "monitor", wall=1_700_000_000.0 + t, battery=batt)
        d = w.drain_pct_per_m2()
        self.assertIsNotNone(d)
        # 100 distinct 0.5 m cells = 25 m^2, battery 90 -> 80: 0.4 %/m^2.
        self.assertAlmostEqual(d, 0.4, delta=0.1)


class TestConsultFlow(unittest.TestCase):
    def _controller_in_episode(self):
        cfg = ControllerConfig(
            llm_strategy_enabled=True, consult_after_failures=2,
            max_consults_per_episode=2)
        ctl = BugNavController(cfg)
        s0 = Sample(t=0.0, pos=Point(-2.0, 0.0))
        ctl._begin_episode(s0, Point(0.0, 0.0), 0.0)
        return ctl

    def _fail_attempt(self, ctl, t):
        ctl.phase = Phase.TO_GOAL
        ctl._deadline = t - 1.0  # already timed out
        return ctl.tick(Sample(t=t, pos=Point(0.3, 0.0)))

    def test_repeated_failures_trigger_consult_then_plan_angle(self):
        ctl = self._controller_in_episode()
        self._fail_attempt(ctl, 10.0)
        cmds = self._fail_attempt(ctl, 20.0)
        self.assertEqual(ctl.phase, Phase.CONSULT)
        self.assertIsNotNone(ctl.pending_consult)
        self.assertEqual(ctl.pending_consult.kind, "repeated_failures")
        plan = parse_plan('{"actions": [{"type": "try_angle", '
                          '"theta_deg": 180}], "reason": "试背面"}')
        s = Sample(t=25.0, pos=Point(0.3, 0.0))
        cmds = ctl.apply_consult_result(plan, s)
        self.assertEqual(ctl.phase, Phase.TO_STAGING)
        self.assertAlmostEqual(ctl._theta_deg, 180.0)

    def test_consult_timeout_falls_back(self):
        ctl = self._controller_in_episode()
        ctl.cfg.consult_timeout_s = 30.0
        self._fail_attempt(ctl, 10.0)
        self._fail_attempt(ctl, 20.0)
        self.assertEqual(ctl.phase, Phase.CONSULT)
        cmds = ctl.tick(Sample(t=100.0, pos=Point(0.3, 0.0)))
        self.assertNotEqual(ctl.phase, Phase.CONSULT)
        self.assertIsNone(ctl.pending_consult)

    def test_disabled_means_no_consult_ever(self):
        cfg = ControllerConfig(llm_strategy_enabled=False,
                               consult_after_failures=1)
        ctl = BugNavController(cfg)
        ctl._begin_episode(Sample(t=0.0, pos=Point(-2.0, 0.0)),
                           Point(0.0, 0.0), 0.0)
        for i in range(5):
            ctl.phase = Phase.TO_GOAL
            ctl._deadline = 0.0
            ctl.tick(Sample(t=10.0 + i, pos=Point(0.3, 0.0)))
            self.assertNotEqual(ctl.phase, Phase.CONSULT)


if __name__ == "__main__":
    unittest.main()
