"""Quiet marks: a handled spot stays quiet.

- A spot whose pause-hold just succeeded must not re-trigger the
  soft detectors while the robot cleans around the same
  furniture - intervening again buys nothing and stalls a
  productive session.
- A spot that used up its hold attempts is left to native
  cleaning for the session: no soft trigger (struggle, ramming,
  perimeter) starts another hold there.
- Both marks expire; nothing is banned permanently, because the
  obstacle is usually movable furniture.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bugnav.controller import BugNavController, ControllerConfig, Phase
from bugnav.detector import RammingEvent, StruggleEvent
from bugnav.geometry import Point
from bugnav.model import CommandKind, RobotState, Sample

CENTRE = Point(-0.31, -0.05)


def _jiggle_into_hold(ctl: BugNavController) -> float:
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
        if ctl.phase == Phase.HOLD:
            break
    assert ctl.phase == Phase.HOLD, ctl.phase
    return t


def _force_hold_success(ctl: BugNavController, t: float) -> None:
    """Drive the active hold to its watch deadline, still
    cleaning: the pattern did not form again = resolved."""
    ctl._hold_stage = "watch"
    ctl._hold_deadline = t + 1.0
    ctl.tick(Sample(t=t + 2.0, pos=ctl._hold_pos,
                    state=RobotState.CLEANING))
    assert ctl.phase == Phase.MONITOR, ctl.phase
    assert any(m[3] == "resolved" for m in ctl._quiet_marks), \
        "success left no resolved mark"


def _stub_struggle(ctl: BugNavController, centre: Point) -> None:
    ctl.detector.feed = lambda s: StruggleEvent(
        t=s.t, centre=centre, approach_heading_deg=0.0)


class TestResolvedSuppression(unittest.TestCase):
    def test_retrigger_inside_resolved_cluster_is_suppressed(self):
        ctl = BugNavController(ControllerConfig())
        t = _jiggle_into_hold(ctl)
        _force_hold_success(ctl, t)
        centre = ctl._quiet_marks[0][0]
        _stub_struggle(ctl, centre)
        cmds = ctl.tick(Sample(t=t + 10.0, pos=centre,
                               state=RobotState.CLEANING))
        self.assertNotIn(CommandKind.PAUSE,
                         [c.kind for c in cmds])
        self.assertEqual(ctl.phase, Phase.MONITOR)

    def test_same_event_without_mark_starts_hold(self):
        ctl = BugNavController(ControllerConfig())
        _stub_struggle(ctl, CENTRE)
        cmds = ctl.tick(Sample(t=10.0, pos=CENTRE,
                               state=RobotState.CLEANING))
        self.assertIn(CommandKind.PAUSE, [c.kind for c in cmds])
        self.assertEqual(ctl.phase, Phase.HOLD)

    def test_suppression_expires(self):
        ctl = BugNavController(ControllerConfig(resolved_suppress_s=5.0))
        t = _jiggle_into_hold(ctl)
        _force_hold_success(ctl, t)
        centre = ctl._quiet_marks[0][0]
        _stub_struggle(ctl, centre)
        cmds = ctl.tick(Sample(t=t + 30.0, pos=centre,
                               state=RobotState.CLEANING))
        self.assertIn(CommandKind.PAUSE, [c.kind for c in cmds])
        self.assertEqual(ctl.phase, Phase.HOLD)


class TestBlockedSuppression(unittest.TestCase):
    def _exhaust_hold(self, ctl: BugNavController) -> Point:
        """Run one hold through all its attempts with the watch
        detector stubbed to fire at once, ending exhausted."""
        _stub_struggle(ctl, CENTRE)
        ctl.tick(Sample(t=0.0, pos=CENTRE, state=RobotState.CLEANING))
        assert ctl.phase == Phase.HOLD
        ctl._watch_struggle.feed = lambda s: StruggleEvent(
            t=s.t, centre=CENTRE, approach_heading_deg=0.0)
        t = 0.0
        for _ in range(3):
            # settle expires while paused -> RESUME into watch
            t += 10.0
            ctl.tick(Sample(t=t, pos=CENTRE, state=RobotState.PAUSED))
            assert ctl._hold_stage == "watch", ctl._hold_stage
            # watch fires at once -> next attempt, or exhausted
            t += 1.0
            ctl.tick(Sample(t=t, pos=CENTRE,
                            state=RobotState.CLEANING))
        assert ctl.phase == Phase.MONITOR, ctl.phase
        assert any(m[3] == "blocked" for m in ctl._quiet_marks)
        return CENTRE

    def test_exhausted_spot_gets_no_new_hold(self):
        ctl = BugNavController(ControllerConfig())
        centre = self._exhaust_hold(ctl)
        _stub_struggle(ctl, centre)
        cmds = ctl.tick(Sample(t=100.0, pos=centre,
                               state=RobotState.CLEANING))
        self.assertNotIn(CommandKind.PAUSE,
                         [c.kind for c in cmds])
        self.assertEqual(ctl.phase, Phase.MONITOR)

    def test_exhausted_spot_blocks_ramming_too(self):
        ctl = BugNavController(ControllerConfig())
        centre = self._exhaust_hold(ctl)
        ctl.detector.feed = lambda s: None  # struggle stays quiet
        ctl.ramming.feed = lambda s: RammingEvent(
            t=s.t, centre=centre, approach_heading_deg=0.0,
            charges=4, mean_leg_m=0.4)
        cmds = ctl.tick(Sample(t=100.0, pos=centre,
                               state=RobotState.CLEANING))
        self.assertNotIn(CommandKind.PAUSE,
                         [c.kind for c in cmds])
        self.assertEqual(ctl.phase, Phase.MONITOR)


if __name__ == "__main__":
    unittest.main()
