"""Opportunistic goal entry during transit legs: while retreating or
heading to a staging point, passing the goal's mouth must dive at the
goal (or count as reached) instead of completing the leg elsewhere."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bugnav.controller import BugNavController, ControllerConfig, Phase
from bugnav.geometry import Point
from bugnav.model import CommandKind, RobotState, Sample


def _start_episode(ctl: BugNavController, *, large: bool) -> float:
    """Drive the circumnavigate-style jiggle until an episode opens;
    returns the tick time reached.  With large=True the cluster is
    widened to furniture-complex size (the dive only applies there)."""
    t = 0.0

    def feed(p: Point):
        nonlocal t
        t += 1.0
        return ctl.tick(Sample(t=t, pos=p, state=RobotState.CLEANING))

    pos = Point(-2.0, 0.0)
    while pos.x < -0.30:
        pos = Point(pos.x + 0.15, 0.0)
        feed(pos)
    jiggle = [Point(-0.28, 0.08), Point(-0.36, 0.0),
              Point(-0.28, -0.08), Point(-0.36, 0.0)]
    for i in range(120):
        feed(jiggle[i % 4])
        if ctl.phase not in (Phase.MONITOR, Phase.COOLDOWN):
            break
    assert ctl.phase in (Phase.RETREAT, Phase.TO_STAGING), ctl.phase
    assert ctl._goal is not None
    if large:
        c = ctl._cluster.centre
        for dx, dy in ((1.5, 0.0), (-1.5, 0.0), (0.0, 1.5), (0.0, -1.5)):
            ctl._cluster.add_hit(Point(c.x + dx, c.y + dy))
        assert ctl._cluster.radius_m >= 0.80
    return t


class TestOpportunisticGoal(unittest.TestCase):
    def test_passing_the_mouth_dives_at_goal(self):
        ctl = BugNavController(ControllerConfig())
        t = _start_episode(ctl, large=True)
        goal = ctl._goal
        near = Point(goal.x + 0.40, goal.y)  # inside 0.50, outside tol
        cmds = ctl.tick(Sample(t=t + 1.0, pos=near,
                               state=RobotState.CLEANING))
        self.assertEqual(ctl.phase, Phase.TO_GOAL)
        moves = [c for c in cmds if c.kind == CommandKind.MOVE_TO]
        self.assertTrue(any(c.point == goal for c in moves), cmds)

    def test_passing_within_tolerance_is_reached(self):
        ctl = BugNavController(ControllerConfig())
        t = _start_episode(ctl, large=True)
        goal = ctl._goal
        near = Point(goal.x + 0.20, goal.y)  # within arrival tolerance
        cmds = ctl.tick(Sample(t=t + 1.0, pos=near,
                               state=RobotState.CLEANING))
        kinds = [c.kind for c in cmds]
        self.assertIn(CommandKind.START_AUTO, kinds)
        self.assertTrue(
            any(s.successes > 0 for s in ctl.stats.values()))

    def test_passing_farther_off_keeps_the_leg(self):
        ctl = BugNavController(ControllerConfig())
        t = _start_episode(ctl, large=True)
        goal = ctl._goal
        phase_before = ctl.phase
        far = Point(goal.x + 0.60, goal.y)  # outside the dive radius
        cmds = ctl.tick(Sample(t=t + 1.0, pos=far,
                               state=RobotState.CLEANING))
        self.assertEqual(ctl.phase, phase_before)
        moves = [c for c in cmds if c.kind == CommandKind.MOVE_TO]
        self.assertFalse(any(c.point == goal for c in moves), cmds)

    def test_small_obstacle_never_dives(self):
        # Around a chair-leg-sized cluster the rotation samples every
        # side in short hops; a transit pass near the goal must not
        # hijack the leg or falsify the sector bookkeeping.
        ctl = BugNavController(ControllerConfig())
        t = _start_episode(ctl, large=False)
        goal = ctl._goal
        phase_before = ctl.phase
        near = Point(goal.x + 0.40, goal.y)
        cmds = ctl.tick(Sample(t=t + 1.0, pos=near,
                               state=RobotState.CLEANING))
        self.assertEqual(ctl.phase, phase_before)
        moves = [c for c in cmds if c.kind == CommandKind.MOVE_TO]
        self.assertFalse(any(c.point == goal for c in moves), cmds)


if __name__ == "__main__":
    unittest.main()
