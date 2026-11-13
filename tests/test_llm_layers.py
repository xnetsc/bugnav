"""Tests for the LLM layer: supervisor verdicts, charge
planning, and the decision trace."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bugnav.chargeplan import ChargeModel, plan_resume
from bugnav.geometry import Point
from bugnav.llm import DecisionTrace
from bugnav.model import RobotState, Sample
from bugnav.supervisor import SessionWatch, SupervisorConfig, parse_verdict


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


class TestDecisionTrace(unittest.TestCase):
    def test_records_round_trip_as_jsonl(self):
        import json
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "llm_trace.jsonl")
            trace = DecisionTrace(path)
            msgs = [{"role": "system", "content": "s"},
                    {"role": "user", "content": "摘要"}]
            trace.record("supervisor", msgs, '{"action": "continue"}',
                         {"action": "continue"}, ts=1794441006.0)
            trace.record("supervisor", msgs, "not json", None,
                         ts=1794441100.0, trigger="stuck")
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["kind"], "supervisor")
            self.assertEqual(rows[0]["messages"], msgs)
            self.assertEqual(rows[0]["response"],
                             '{"action": "continue"}')
            self.assertEqual(rows[0]["parsed"], {"action": "continue"})
            self.assertIn("model", rows[0])
            self.assertIsNone(rows[1]["parsed"])
            self.assertEqual(rows[1]["trigger"], "stuck")

    def test_write_failure_is_silent(self):
        trace = DecisionTrace("/nonexistent-dir-xyz/llm_trace.jsonl")
        trace.record("supervisor", [], "{}", {"action": "continue"},
                     ts=0.0)  # must not raise


if __name__ == "__main__":
    unittest.main()
