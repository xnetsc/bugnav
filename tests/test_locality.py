"""Locality rules: cleaning continues over adjacent ground.

- A cluster whose episode just succeeded must not re-trigger the
  soft detectors when the fresh native plan wanders back into the
  same furniture - each re-takeover resets the native session and
  sends it re-cleaning finished ground in disconnected patches.
- An episode must not retreat to an anchor that lies farther from
  the obstacle than the staging ring when the robot is already at
  working distance; it starts from the nearest staging point.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bugnav.clusters import ObstacleCluster
from bugnav.controller import BugNavController, ControllerConfig, Phase
from bugnav.detector import StruggleEvent
from bugnav.geometry import Point
from bugnav.model import CommandKind, RobotState, Sample


def _jiggle_into_episode(ctl: BugNavController) -> float:
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
    return t


def _force_success(ctl: BugNavController, t: float) -> None:
    ctl.phase = Phase.TO_GOAL
    ctl._pending_target = ctl._goal
    ctl.tick(Sample(t=t + 1.0, pos=ctl._goal, state=RobotState.CLEANING))
    assert ctl._resolved_marks, "success left no resolved mark"


class TestResolvedSuppression(unittest.TestCase):
    def test_retrigger_inside_resolved_cluster_is_suppressed(self):
        ctl = BugNavController(ControllerConfig())
        t = _jiggle_into_episode(ctl)
        _force_success(ctl, t)
        centre = ctl._resolved_marks[0][0]
        ctl.detector.feed = lambda s: StruggleEvent(
            t=s.t, centre=centre, approach_heading_deg=0.0)
        cmds = ctl.tick(Sample(t=t + 10.0, pos=centre,
                               state=RobotState.CLEANING))
        self.assertNotIn(CommandKind.PAUSE,
                         [c.kind for c in cmds])
        self.assertIn(ctl.phase, (Phase.MONITOR, Phase.COOLDOWN))

    def test_same_event_without_mark_starts_episode(self):
        ctl = BugNavController(ControllerConfig())
        centre = Point(-0.31, -0.05)
        ctl.detector.feed = lambda s: StruggleEvent(
            t=s.t, centre=centre, approach_heading_deg=0.0)
        cmds = ctl.tick(Sample(t=10.0, pos=centre,
                               state=RobotState.CLEANING))
        self.assertIn(CommandKind.PAUSE, [c.kind for c in cmds])
        self.assertIn(ctl.phase, (Phase.RETREAT, Phase.TO_STAGING))

    def test_suppression_expires(self):
        ctl = BugNavController(ControllerConfig(resolved_suppress_s=5.0))
        t = _jiggle_into_episode(ctl)
        _force_success(ctl, t)
        centre = ctl._resolved_marks[0][0]
        ctl.detector.feed = lambda s: StruggleEvent(
            t=s.t, centre=centre, approach_heading_deg=0.0)
        cmds = ctl.tick(Sample(t=t + 30.0, pos=centre,
                               state=RobotState.CLEANING))
        self.assertIn(CommandKind.PAUSE, [c.kind for c in cmds])


class TestRetreatLocality(unittest.TestCase):
    def _cluster(self) -> ObstacleCluster:
        return ObstacleCluster(hits=[Point(1.17, 0.0), Point(-1.17, 0.0),
                                     Point(0.0, 1.17), Point(0.0, -1.17)])

    def _trail_from(self, ctl: BugNavController, xs: list[float]) -> None:
        for i, x in enumerate(xs):
            ctl.trail.record(Sample(t=float(i + 1), pos=Point(x, 0.0),
                                    state=RobotState.CLEANING))

    def test_far_anchor_retreat_skipped_nearest_staging_first(self):
        ctl = BugNavController(ControllerConfig())
        self._trail_from(ctl, [-6.0, -5.0, -4.0, -3.0])  # anchor (-6, 0)
        sample = Sample(t=10.0, pos=Point(0.3, 0.0),
                        state=RobotState.CLEANING)
        cmds = ctl._begin_episode(sample, Point(0.9, 0.0), 0.0,
                                  cluster=self._cluster())
        self.assertEqual(ctl.phase, Phase.TO_STAGING)
        moves = [c.point for c in cmds if c.kind == CommandKind.MOVE_TO]
        self.assertTrue(moves)
        # nearest staging point to the robot: ring 1.67 m at 0 deg
        self.assertLess(moves[0].distance_to(Point(1.67, 0.0)), 0.05)
        self.assertFalse(
            any(p.distance_to(Point(-6.0, 0.0)) < 0.05 for p in moves))

    def test_near_anchor_retreat_kept(self):
        ctl = BugNavController(ControllerConfig())
        self._trail_from(ctl, [-1.8, -0.9])  # anchor (-1.8, 0), at the ring
        sample = Sample(t=10.0, pos=Point(0.35, 0.0),
                        state=RobotState.CLEANING)
        cmds = ctl._begin_episode(sample, Point(0.9, 0.0), 0.0,
                                  cluster=self._cluster())
        self.assertEqual(ctl.phase, Phase.RETREAT)
        moves = [c.point for c in cmds if c.kind == CommandKind.MOVE_TO]
        self.assertTrue(
            any(p.distance_to(Point(-1.8, 0.0)) < 0.05 for p in moves))


if __name__ == "__main__":
    unittest.main()
