"""Scripted test: a small obstacle (chair leg) where *every* approach
fails must still end as success after a full 360-degree exploration -
the perimeter has then been fully covered, which is the desired
outcome for a leg."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bugnav.controller import BugNavController, ControllerConfig
from bugnav.geometry import Point
from bugnav.model import CommandKind, RobotState, Sample


class TestCircumnavigation(unittest.TestCase):
    def test_full_circle_on_small_obstacle_is_success(self):
        # The fake leg below blocks the robot centre out to 0.48 m, so
        # the small-obstacle threshold is raised accordingly (threshold
        # tuning is a deployment calibration item).
        ctl = BugNavController(ControllerConfig(small_obstacle_radius_m=0.65))
        leg = Point(0.0, 0.0)
        pos = Point(-2.0, 0.0)
        t = 0.0
        target: Point | None = None
        notes: list[str] = []

        def feed(p: Point, state=RobotState.CLEANING):
            nonlocal t
            t += 1.0
            cmds = ctl.tick(Sample(t=t, pos=p, state=state))
            new_target = None
            for c in cmds:
                if c.kind == CommandKind.NOTIFY:
                    notes.append(c.message)
                if c.kind == CommandKind.CLEAN_AREA:
                    new_target = c.point
            return new_target

        # 1) Free approach eastwards, then jiggle at the leg for >30 s.
        while pos.x < -0.30:
            pos = Point(pos.x + 0.15, 0.0)
            feed(pos)
        contact = Point(-0.28, 0.0)
        jiggle = [Point(-0.28, 0.08), Point(-0.36, 0.0), Point(-0.28, -0.08), Point(-0.36, 0.0)]
        for i in range(80):
            nt = feed(jiggle[i % 4])
            if nt is not None:
                target = nt
        self.assertIsNotNone(target, f"episode never started; notes={notes}")

        # 2) Fake robot: staging/retreat/backtrack targets are reachable;
        #    goal attempts stop 0.30 m short of the leg and stay there
        #    (goal timeout then fails the attempt).
        pos = contact
        for _ in range(4000):
            if ctl.stats and any(s.successes > 0 for s in ctl.stats.values()):
                break
            if target is not None:
                d = pos.distance_to(target)
                goal_attempt = (
                    ctl._goal is not None and target == ctl._goal
                )
                if goal_attempt:
                    # stop short of the leg (blocked disc radius 0.48 m)
                    stop = leg + (pos - leg).normalized().scale(0.48) if pos != leg else Point(0.48, 0)
                    if pos.distance_to(stop) > 0.05:
                        step = min(0.15, pos.distance_to(stop))
                        pos = pos + (stop - pos).normalized().scale(step)
                    # else: stay (blocked)
                elif d > 0.05:
                    step = min(0.15, d)
                    pos = pos + (target - pos).normalized().scale(step)
            nt = feed(pos)
            if nt is not None:
                target = nt
        stats = list(ctl.stats.values())
        self.assertTrue(any(s.successes > 0 for s in stats), notes)
        self.assertFalse(any(s.abandons > 0 for s in stats), notes)
        self.assertTrue(any("circumnavigated" in n for n in notes), notes)


if __name__ == "__main__":
    unittest.main()
