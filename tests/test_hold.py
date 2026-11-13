"""Pause-hold: the session-preserving intervention.

- Soft triggers (struggle, ramming, perimeter loiter) start a
  hold: PAUSE in place, settle, RESUME the same session, then a
  watch with unlatched detectors judges whether the pattern
  forms again.
- Across the whole lifecycle - attempts, exhaustion included -
  the hold NEVER emits a session-killing command (no MOVE_TO,
  no START_AUTO, no JOG).  That is the invariant this design
  exists for.
- A hard signal during a hold (a fault persisting in place)
  hands the spot to the wedged-rescue path instead.
- End to end against the simulator's naive firmware, a robot
  that jiggles forever gets pause-resume cycles and then an
  exhaustion notice - and is never repositioned, never
  restarted.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bugnav.controller import BugNavController, ControllerConfig, Phase
from bugnav.detector import RammingEvent, StruggleEvent
from bugnav.geometry import Point
from bugnav.model import CommandKind, RobotState, Sample
from bugnav.sim import SimRobot, Wall, World

CENTRE = Point(-0.31, -0.05)
KILLING = (CommandKind.MOVE_TO, CommandKind.START_AUTO,
           CommandKind.JOG)


def _kinds(cmds):
    return [c.kind for c in cmds]


def _stub_struggle(ctl):
    ctl.detector.feed = lambda s: StruggleEvent(
        t=s.t, centre=CENTRE, approach_heading_deg=0.0)


def _begin_hold(ctl, t=0.0):
    _stub_struggle(ctl)
    cmds = ctl.tick(Sample(t=t, pos=CENTRE,
                           state=RobotState.CLEANING))
    assert ctl.phase == Phase.HOLD, ctl.phase
    return cmds


class TestHoldLifecycle(unittest.TestCase):
    def test_struggle_starts_hold_with_pause_only(self):
        ctl = BugNavController(ControllerConfig())
        cmds = _begin_hold(ctl)
        self.assertIn(CommandKind.PAUSE, _kinds(cmds))
        for k in KILLING:
            self.assertNotIn(k, _kinds(cmds))

    def test_ramming_starts_hold(self):
        ctl = BugNavController(ControllerConfig())
        ctl.ramming.feed = lambda s: RammingEvent(
            t=s.t, centre=CENTRE, approach_heading_deg=0.0,
            charges=4, mean_leg_m=0.4)
        cmds = ctl.tick(Sample(t=0.0, pos=CENTRE,
                               state=RobotState.CLEANING))
        self.assertIn(CommandKind.PAUSE, _kinds(cmds))
        self.assertEqual(ctl.phase, Phase.HOLD)

    def test_settle_then_resume_same_session(self):
        ctl = BugNavController(ControllerConfig())
        _begin_hold(ctl, t=0.0)
        cmds = ctl.tick(Sample(t=3.0, pos=CENTRE,
                               state=RobotState.PAUSED))
        self.assertEqual(cmds, [])  # still settling
        cmds = ctl.tick(Sample(t=7.0, pos=CENTRE,
                               state=RobotState.PAUSED))
        self.assertIn(CommandKind.RESUME_AUTO, _kinds(cmds))
        self.assertEqual(ctl._hold_stage, "watch")
        for k in KILLING:
            self.assertNotIn(k, _kinds(cmds))

    def test_watch_without_pattern_resolves(self):
        ctl = BugNavController(ControllerConfig())
        _begin_hold(ctl, t=0.0)
        ctl.tick(Sample(t=7.0, pos=CENTRE, state=RobotState.PAUSED))
        # It resumes and cleans straight away from the spot.
        t = 7.0
        cmds = []
        for i in range(30):
            t += 2.5
            cmds.extend(ctl.tick(Sample(
                t=t, pos=Point(CENTRE.x + 0.1 * i, CENTRE.y),
                state=RobotState.CLEANING)))
        self.assertEqual(ctl.phase, Phase.MONITOR)
        self.assertTrue(
            any(m[3] == "resolved" for m in ctl._quiet_marks))
        st = list(ctl.stats.values())
        self.assertEqual(st[0].successes, 1)
        for k in KILLING:
            self.assertNotIn(k, _kinds(cmds))

    def test_error_during_hold_hands_over_to_unstick(self):
        ctl = BugNavController(ControllerConfig())
        _begin_hold(ctl, t=0.0)
        cmds = []
        for i in range(1, 12):
            cmds.extend(ctl.tick(Sample(
                t=float(i), pos=CENTRE, state=RobotState.PAUSED,
                error_code=103)))
        self.assertIn(CommandKind.JOG, _kinds(cmds))
        self.assertEqual(ctl.phase, Phase.UNSTICK)


class TestHoldInvariant(unittest.TestCase):
    def test_full_lifecycle_never_kills_session(self):
        """Attempts, re-attempts, exhaustion: only PAUSE,
        RESUME_AUTO and NOTIFY may ever come out of a hold."""
        ctl = BugNavController(ControllerConfig())
        all_cmds = list(_begin_hold(ctl, t=0.0))
        ctl._watch_struggle.feed = lambda s: StruggleEvent(
            t=s.t, centre=CENTRE, approach_heading_deg=0.0)
        t = 0.0
        for _ in range(3):
            t += 10.0
            all_cmds.extend(ctl.tick(Sample(
                t=t, pos=CENTRE, state=RobotState.PAUSED)))
            t += 1.0
            all_cmds.extend(ctl.tick(Sample(
                t=t, pos=CENTRE, state=RobotState.CLEANING)))
        self.assertEqual(ctl.phase, Phase.MONITOR)  # exhausted
        self.assertTrue(
            any(m[3] == "blocked" for m in ctl._quiet_marks))
        for k in KILLING:
            self.assertNotIn(k, _kinds(all_cmds), all_cmds)
        self.assertIn(CommandKind.RESUME_AUTO, _kinds(all_cmds))


class TestHoldAgainstSimulator(unittest.TestCase):
    def test_naive_firmware_gets_holds_never_a_restart(self):
        walls = [
            Wall(Point(-0.5, -0.6), Point(-0.5, 0.6)),
            Wall(Point(-0.5, 0.6), Point(0.5, 0.6)),
            Wall(Point(-0.5, -0.6), Point(0.5, -0.6)),
        ]
        world = World(walls)
        goal = Point(0.0, 0.0)
        robot = SimRobot(world=world, pos=Point(-2.4, 0.0), goal=goal)
        ctl = BugNavController(ControllerConfig())
        all_kinds = []
        t = 0.0
        reached = False
        while t < 1500.0:
            sample = robot.tick(t, 0.5)
            if sample.pos.distance_to(goal) < 0.15:
                reached = True
            cmds = ctl.tick(sample)
            all_kinds.extend(_kinds(cmds))
            for cmd in cmds:
                robot.execute(cmd)
            t += 0.5
        self.assertFalse(reached)
        self.assertIn(CommandKind.PAUSE, all_kinds)
        self.assertIn(CommandKind.RESUME_AUTO, all_kinds)
        for k in KILLING:
            self.assertNotIn(k, all_kinds)
        stats = list(ctl.stats.values())
        self.assertTrue(any(s.abandons > 0 for s in stats), stats)


if __name__ == "__main__":
    unittest.main()
