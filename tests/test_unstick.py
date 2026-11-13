"""Wedged rescue: backing out is the whole rescue.

- A fault that persists while the robot stays put (state ERROR
  or paused-with-error - the firmware uses both for one fault)
  triggers a jog back-out after the persistence delay, and only
  then.
- A position frozen while cleaning triggers the same back-out.
- A back-out that frees the robot: the jog has spent the native
  session, so once the fault reads clear a fresh AUTO clean
  starts right there; if the robot resumes on its own, the
  rescue simply closes.
- A back-out that fails - the fault persists, or the rear is
  blocked and the jog budget runs out - stops the robot, sounds
  its alarm and waits for a human (the trapped latch).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bugnav.controller import BugNavController, ControllerConfig, Phase
from bugnav.detector import PinnedConfig, PinnedDetector
from bugnav.geometry import Point
from bugnav.model import CommandKind, RobotState, Sample

WEDGE = Point(-4.63, -2.15)


def _kinds(cmds):
    return [c.kind for c in cmds]


def _feed(ctl, t0, t1, pos, state, err=0):
    """Feed 1 s samples in [t0, t1]; return all commands issued."""
    out = []
    t = t0
    while t <= t1:
        out.extend(ctl.tick(Sample(t=float(t), pos=pos, state=state,
                                   error_code=err)))
        t += 1
    return out


def _error_unstick(ctl):
    """Drive a fresh controller into UNSTICK via a persistent
    paused fault at the wedge; returns the trigger time."""
    _feed(ctl, 0, 9, WEDGE, RobotState.PAUSED, err=103)
    cmds = _feed(ctl, 10, 10, WEDGE, RobotState.PAUSED, err=103)
    assert CommandKind.JOG in _kinds(cmds), cmds
    assert ctl.phase == Phase.UNSTICK, ctl.phase
    return 10.0


class TestPinnedDetector(unittest.TestCase):
    def test_frozen_cleaning_fires(self):
        det = PinnedDetector(PinnedConfig(window_s=60.0))
        event = None
        for i in range(70):
            ev = det.feed(Sample(t=float(i), pos=WEDGE,
                                 state=RobotState.CLEANING))
            if ev is not None:
                event = ev
                break
        self.assertIsNotNone(event)
        self.assertEqual(event.centre, WEDGE)

    def test_moving_cleaning_never_fires(self):
        det = PinnedDetector(PinnedConfig(window_s=60.0))
        for i in range(120):
            ev = det.feed(Sample(t=float(i),
                                 pos=Point(-4.0 + 0.1 * i, -2.0),
                                 state=RobotState.CLEANING))
            self.assertIsNone(ev)

    def test_frozen_paused_never_fires(self):
        det = PinnedDetector(PinnedConfig(window_s=60.0))
        for i in range(120):
            ev = det.feed(Sample(t=float(i), pos=WEDGE,
                                 state=RobotState.PAUSED))
            self.assertIsNone(ev)


class TestErrorTrigger(unittest.TestCase):
    def test_persistent_paused_error_triggers_jog_without_pause(self):
        ctl = BugNavController(ControllerConfig())
        cmds = _feed(ctl, 0, 9, WEDGE, RobotState.PAUSED, err=103)
        self.assertNotIn(CommandKind.JOG, _kinds(cmds))
        self.assertEqual(ctl.phase, Phase.MONITOR)
        cmds = _feed(ctl, 10, 10, WEDGE, RobotState.PAUSED, err=103)
        self.assertIn(CommandKind.JOG, _kinds(cmds))
        self.assertNotIn(CommandKind.PAUSE, _kinds(cmds))
        jog = [c for c in cmds if c.kind == CommandKind.JOG][0]
        self.assertEqual(jog.act, "backward")
        self.assertEqual(ctl.phase, Phase.UNSTICK)

    def test_error_state_form_triggers_too(self):
        ctl = BugNavController(ControllerConfig())
        _feed(ctl, 0, 9, WEDGE, RobotState.ERROR, err=105)
        cmds = _feed(ctl, 10, 10, WEDGE, RobotState.ERROR, err=105)
        self.assertIn(CommandKind.JOG, _kinds(cmds))
        self.assertEqual(ctl.phase, Phase.UNSTICK)

    def test_transient_error_never_triggers(self):
        ctl = BugNavController(ControllerConfig())
        cmds = _feed(ctl, 0, 5, WEDGE, RobotState.PAUSED, err=103)
        cmds += _feed(ctl, 6, 40, WEDGE, RobotState.PAUSED, err=0)
        self.assertNotIn(CommandKind.JOG, _kinds(cmds))
        self.assertEqual(ctl.phase, Phase.MONITOR)

    def test_error_while_driving_away_never_triggers(self):
        ctl = BugNavController(ControllerConfig())
        cmds = []
        for i in range(30):
            cmds.extend(ctl.tick(Sample(
                t=float(i), pos=Point(-4.63 + 0.2 * i, -2.15),
                state=RobotState.PAUSED, error_code=103)))
        self.assertNotIn(CommandKind.JOG, _kinds(cmds))


class TestFrozenTrigger(unittest.TestCase):
    def test_frozen_cleaning_pauses_then_jogs(self):
        ctl = BugNavController(ControllerConfig())
        cmds = _feed(ctl, 0, 53, WEDGE, RobotState.CLEANING)
        self.assertNotIn(CommandKind.JOG, _kinds(cmds))
        cmds = _feed(ctl, 54, 60, WEDGE, RobotState.CLEANING)
        self.assertIn(CommandKind.JOG, _kinds(cmds))
        self.assertIn(CommandKind.PAUSE, _kinds(cmds))
        self.assertEqual(ctl.phase, Phase.UNSTICK)


class TestUnstickOutcomes(unittest.TestCase):
    def test_freed_and_resumed_closes_without_fresh_start(self):
        ctl = BugNavController(ControllerConfig())
        _error_unstick(ctl)
        cmds = _feed(ctl, 11, 11, Point(-4.53, -2.15),
                     RobotState.PAUSED, err=0)
        cmds += _feed(ctl, 12, 12, Point(-4.33, -2.15),
                      RobotState.PAUSED, err=0)
        # Freed: jog stream stopped, and NO resume is sent - the
        # jog spent the session, there is nothing to resume.
        self.assertNotIn(CommandKind.RESUME_AUTO, _kinds(cmds))
        jogs = [c for c in cmds if c.kind == CommandKind.JOG]
        self.assertEqual([j.act for j in jogs], ["stop"])
        cmds = _feed(ctl, 13, 13, Point(-4.33, -2.15),
                     RobotState.CLEANING, err=0)
        self.assertNotIn(CommandKind.START_AUTO, _kinds(cmds))
        self.assertEqual(ctl.phase, Phase.MONITOR)

    def test_freed_but_session_dead_starts_fresh_auto(self):
        ctl = BugNavController(ControllerConfig())
        _error_unstick(ctl)
        _feed(ctl, 11, 11, Point(-4.53, -2.15), RobotState.PAUSED,
              err=0)
        _feed(ctl, 12, 12, Point(-4.33, -2.15), RobotState.PAUSED,
              err=0)  # freed at t=12; fault-clear window ends t=20
        cmds = _feed(ctl, 13, 19, Point(-4.33, -2.15),
                     RobotState.PAUSED, err=0)
        self.assertNotIn(CommandKind.START_AUTO, _kinds(cmds))
        cmds = _feed(ctl, 20, 20, Point(-4.33, -2.15),
                     RobotState.PAUSED, err=0)
        self.assertIn(CommandKind.START_AUTO, _kinds(cmds))
        self.assertEqual(ctl.phase, Phase.MONITOR)

    def test_freed_but_fault_persists_waits_for_rescue(self):
        ctl = BugNavController(ControllerConfig())
        _error_unstick(ctl)
        _feed(ctl, 11, 12, Point(-4.33, -2.15), RobotState.PAUSED,
              err=103)
        cmds = _feed(ctl, 13, 20, Point(-4.33, -2.15),
                     RobotState.PAUSED, err=103)
        self.assertIn(CommandKind.PLAY_SOUND, _kinds(cmds))
        self.assertNotIn(CommandKind.MOVE_TO, _kinds(cmds))
        self.assertEqual(ctl.phase, Phase.MONITOR)
        self.assertIsNotNone(ctl._trapped_pos)
        # Latched: the same persisting fault starts nothing new.
        cmds = _feed(ctl, 21, 40, Point(-4.33, -2.15),
                     RobotState.PAUSED, err=103)
        self.assertNotIn(CommandKind.JOG, _kinds(cmds))

    def test_rear_blocked_waits_for_rescue_after_jog_budget(self):
        ctl = BugNavController(ControllerConfig())
        _error_unstick(ctl)  # step 1 out at t=10, deadline 18
        cmds = _feed(ctl, 11, 18, WEDGE, RobotState.PAUSED, err=103)
        self.assertIn(CommandKind.JOG, _kinds(cmds))  # step 2
        cmds = _feed(ctl, 19, 26, WEDGE, RobotState.PAUSED, err=103)
        self.assertIn(CommandKind.JOG, _kinds(cmds))  # step 3
        cmds = _feed(ctl, 27, 34, WEDGE, RobotState.PAUSED, err=103)
        self.assertIn(CommandKind.PLAY_SOUND, _kinds(cmds))
        self.assertNotIn(CommandKind.MOVE_TO, _kinds(cmds))
        self.assertEqual(ctl.phase, Phase.MONITOR)
        self.assertIsNotNone(ctl._trapped_pos)

    def test_moving_robot_never_unsticks(self):
        ctl = BugNavController(ControllerConfig())
        cmds = []
        for i in range(180):
            cmds.extend(ctl.tick(Sample(
                t=float(i), pos=Point(-4.0 + 0.2 * (i % 20), -2.0),
                state=RobotState.CLEANING)))
        self.assertNotIn(CommandKind.JOG, _kinds(cmds))
        self.assertEqual(ctl.phase, Phase.MONITOR)


if __name__ == "__main__":
    unittest.main()
