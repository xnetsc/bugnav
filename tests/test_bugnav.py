"""Unit + end-to-end (simulator) tests for bugnav."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bugnav.geometry import Point
from bugnav.model import RobotState, Sample
from bugnav.scenarios import run_scenario
from bugnav.trail import BreadcrumbTrail


class TestTrail(unittest.TestCase):
    def test_reversed_waypoints_cover_entry_path(self):
        trail = BreadcrumbTrail(max_length_m=20.0, min_spacing_m=0.05)
        for i in range(21):
            trail.record(Sample(t=float(i), pos=Point(i * 0.1, 0.0)))
        anchor = trail.freeze()
        self.assertEqual(anchor, Point(0.0, 0.0))
        wps = trail.reversed_waypoints(spacing_m=0.45)
        self.assertEqual(wps[0], Point(2.0, 0.0))
        self.assertEqual(wps[-1], Point(0.0, 0.0))
        xs = [p.x for p in wps]
        self.assertEqual(xs, sorted(xs, reverse=True))


class TestScenarios(unittest.TestCase):
    def test_u_base_opening_found(self):
        controller, robot, notes, reached = run_scenario("u_base")
        self.assertTrue(reached, f"robot never reached the pocket; notes={notes}")
        stats = list(controller.stats.values())
        self.assertTrue(any(s.successes > 0 for s in stats), notes)

    def test_table_leg_got_around(self):
        controller, robot, notes, reached = run_scenario("table_leg")
        self.assertTrue(reached, f"robot never got past the leg; notes={notes}")
        stats = list(controller.stats.values())
        self.assertTrue(any(s.successes > 0 for s in stats), notes)

    def test_sealed_box_abandoned_and_backtracked(self):
        controller, robot, notes, reached = run_scenario("sealed_box")
        self.assertFalse(reached)
        stats = list(controller.stats.values())
        self.assertTrue(any(s.abandons > 0 for s in stats), notes)
        # After backtracking the robot must be back outside, near the
        # entry anchor on the west side.
        self.assertLess(robot.pos.x, -1.5, f"robot ended at {robot.pos}")
        # And the spot is blacklisted for this session only.
        self.assertTrue(controller._blacklist)


if __name__ == "__main__":
    unittest.main()
