"""Backtrack is not a blind replay (user 2026-11-11):

* the entry corridor may be blocked by something NEW on the way out;
  blockage (struggle, or a waypoint timing out with no progress)
  starts a perimeter DETOUR whose objective is leaving the area -
  rejoining the corridor at ANY remaining breadcrumb counts, the
  aim point is the area exit at the far end of the trail;
* only when that exploration has circled completely and can neither
  regain the exit nor leave the area at all is the robot declared
  TRAPPED: it stops and alarms for a human instead of grinding on.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bugnav.clusters import ObstacleCluster
from bugnav.controller import BugNavController, ControllerConfig, Phase
from bugnav.geometry import Point
from bugnav.model import CommandKind, RobotState, Sample


def _ctl_in_backtrack(points, pos, deadline=5.0):
    ctl = BugNavController(ControllerConfig())
    ctl.phase = Phase.BACKTRACK
    ctl._backtrack_points = list(points)
    ctl._pending_target = points[0] if points else None
    ctl._deadline = deadline
    ctl._bt_start_dist = (
        pos.distance_to(points[0]) if points else None)
    return ctl


def _kinds(cmds):
    return [c.kind for c in cmds]


class TestBacktrackDetour(unittest.TestCase):
    def test_blocked_waypoint_starts_detour_toward_exit(self):
        pts = [Point(-1.0, 0.0), Point(-2.0, 0.0), Point(-3.0, 0.0)]
        ctl = _ctl_in_backtrack(pts, Point(0.0, 0.0))
        # Waypoint times out with zero progress toward it.
        cmds = ctl.tick(Sample(t=6.0, pos=Point(0.0, 0.0),
                               state=RobotState.CLEANING))
        self.assertEqual(ctl._detour_goal, Point(-3.0, 0.0))
        self.assertIn(ctl.phase, (Phase.TO_STAGING, Phase.RETREAT,
                                  Phase.TO_GOAL))
        notes = [c.message for c in cmds if c.kind == CommandKind.NOTIFY]
        self.assertTrue(any("return path blocked" in n for n in notes),
                        notes)

    def test_timeout_with_progress_just_advances(self):
        pts = [Point(-1.0, 0.0), Point(-2.0, 0.0), Point(-3.0, 0.0)]
        ctl = _ctl_in_backtrack(pts, Point(0.0, 0.0))
        # Half-way to the waypoint when it times out: close enough,
        # move on - no detour.
        cmds = ctl.tick(Sample(t=6.0, pos=Point(-0.5, 0.0),
                               state=RobotState.CLEANING))
        self.assertIsNone(ctl._detour_goal)
        self.assertEqual(ctl.phase, Phase.BACKTRACK)
        self.assertEqual(ctl._pending_target, Point(-2.0, 0.0))
        self.assertIn(CommandKind.CLEAN_AREA, _kinds(cmds))

    def test_detour_rejoins_at_any_breadcrumb(self):
        pts = [Point(-1.0, 0.0), Point(-2.0, 0.0), Point(-3.0, 0.0)]
        ctl = _ctl_in_backtrack(pts, Point(0.0, 0.0))
        ctl._detour_goal = pts[-1]
        ctl._goal = pts[-1]
        ctl.phase = Phase.TO_GOAL
        ctl._pending_target = pts[-1]
        ctl._deadline = 100.0
        # The robot surfaces at the MIDDLE breadcrumb: success -
        # leaving the area matters, not the exact waypoint.
        cmds = ctl.tick(Sample(t=10.0, pos=Point(-2.0, 0.0),
                               state=RobotState.CLEANING))
        self.assertEqual(ctl.phase, Phase.BACKTRACK)
        self.assertEqual(ctl._backtrack_points, [Point(-3.0, 0.0)])
        self.assertEqual(ctl._pending_target, Point(-3.0, 0.0))
        self.assertIn(CommandKind.CLEAN_AREA, _kinds(cmds))

    def test_detour_full_circle_declares_trapped(self):
        ctl = BugNavController(ControllerConfig())
        ctl._cluster = ObstacleCluster(
            hits=[Point(0.5, 0.0), Point(-0.5, 0.0),
                  Point(0.0, 0.5), Point(0.0, -0.5)])
        ctl._detour_goal = Point(-3.0, 0.0)
        ctl._goal = Point(-3.0, 0.0)
        cmds = ctl._conclude_full_circle(
            Sample(t=10.0, pos=Point(0.0, 0.0), state=RobotState.PAUSED))
        self.assertIn(CommandKind.PAUSE, _kinds(cmds))
        self.assertNotIn(CommandKind.RESUME_AUTO, _kinds(cmds))
        # The robot itself must sound the alarm too, not only the
        # program's notification (user 2026-11-11).
        sounds = [c for c in cmds if c.kind == CommandKind.PLAY_SOUND]
        self.assertEqual(len(sounds), 1)
        self.assertEqual(sounds[0].count, 3)
        notes = [c.message for c in cmds if c.kind == CommandKind.NOTIFY]
        self.assertTrue(any("TRAPPED" in n for n in notes), notes)
        self.assertIsNotNone(ctl._trapped_pos)

    def test_trapped_latch_waits_for_human(self):
        ctl = BugNavController(ControllerConfig())
        ctl._trapped_pos = Point(0.0, 0.0)
        ctl.phase = Phase.MONITOR
        # Jiggling in place must not start a new episode.
        jiggle = [Point(0.05, 0.0), Point(-0.05, 0.0),
                  Point(0.0, 0.05), Point(0.0, -0.05)]
        for i in range(40):
            cmds = ctl.tick(Sample(t=float(i), pos=jiggle[i % 4],
                                   state=RobotState.CLEANING))
            self.assertNotIn(CommandKind.CLEAN_AREA, _kinds(cmds))
        self.assertIsNotNone(ctl._trapped_pos)
        # Moved by a human: latch clears.
        ctl.tick(Sample(t=100.0, pos=Point(2.0, 0.0),
                        state=RobotState.CLEANING))
        self.assertIsNone(ctl._trapped_pos)


if __name__ == "__main__":
    unittest.main()
