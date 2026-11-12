"""Ramming (charge-and-climb) detection + the rule that handling
follows the data stream, never the commands already sent.

The shape: the firmware backs off a few
decimetres and charges the same U-chair base face again and again,
trying to climb it, until a drive wheel leaves the ground.  The
sawtooth escapes the struggle disc (excursions too wide) and never
circles (so the perimeter detector stays silent too).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bugnav.controller import BugNavController, ConsultRequest, Phase
from bugnav.detector import RammingDetector
from bugnav.geometry import Point
from bugnav.model import CommandKind, RobotState, Sample


def _feed(det, points, dt=2.0, t0=0.0, state=RobotState.CLEANING):
    """Feed a point sequence, return the list of events produced."""
    events = []
    for i, (x, y) in enumerate(points):
        ev = det.feed(Sample(t=t0 + i * dt, pos=Point(x, y), state=state))
        if ev is not None:
            events.append(ev)
    return events


def _sawtooth(retreat, impact, cycles, jitter=0.03):
    """retreat -> impact -> retreat ... with a midpoint per leg."""
    pts = [retreat]
    import random

    rnd = random.Random(42)
    for _ in range(cycles):
        for target in (impact, retreat):
            mid = ((pts[-1][0] + target[0]) / 2, (pts[-1][1] + target[1]) / 2)
            pts.append(mid)
            pts.append((target[0] + rnd.uniform(-jitter, jitter),
                        target[1] + rnd.uniform(-jitter, jitter)))
    return pts


class TestRammingDetector(unittest.TestCase):
    def test_strong_charges_fire(self):
        det = RammingDetector()
        pts = _sawtooth((0.0, 0.0), (0.55, 0.0), cycles=3)
        events = _feed(det, pts)
        self.assertEqual(len(events), 1)
        ev = events[0]
        # Impact face is the +x end; the approach runs +x.
        self.assertAlmostEqual(ev.centre.x, 0.55, delta=0.1)
        self.assertAlmostEqual(ev.centre.y, 0.0, delta=0.1)
        self.assertGreaterEqual(ev.charges, 2)
        self.assertGreater(ev.mean_leg_m, 0.3)

    def test_short_leg_sawtooth_needs_more_cycles(self):
        det = RammingDetector()
        pts = _sawtooth((0.0, 0.0), (0.28, 0.0), cycles=4)
        events = _feed(det, pts)
        self.assertEqual(len(events), 1)

    def test_two_charges_only_do_not_fire(self):
        det = RammingDetector()
        pts = _sawtooth((0.0, 0.0), (0.55, 0.0), cycles=1)
        self.assertEqual(_feed(det, pts), [])

    def test_boustrophedon_lanes_do_not_fire(self):
        pts = []
        y = 0.0
        for lane in range(4):
            xs = [i * 0.5 for i in range(7)] if lane % 2 == 0 else [
                i * 0.5 for i in reversed(range(7))]
            for x in xs:
                pts.append((x, y))
            y += 0.3
            pts.append((xs[-1], y))
        self.assertEqual(_feed(RammingDetector(), pts), [])

    def test_wall_following_square_does_not_fire(self):
        pts = []
        for (x0, y0), (x1, y1) in [((0, 0), (1, 0)), ((1, 0), (1, 1)),
                                   ((1, 1), (0, 1)), ((0, 1), (0, 0))]:
            for k in range(1, 5):
                pts.append((x0 + (x1 - x0) * k / 4, y0 + (y1 - y0) * k / 4))
        self.assertEqual(_feed(RammingDetector(), pts), [])

    def test_fan_scrub_does_not_fire(self):
        import math

        pts = [(0.0, 0.0)]
        for heading in (0, 50, -30, 80, -70, 20, 110, -100, 40, -50):
            r = math.radians(heading)
            x, y = pts[-1]
            pts.append((x + 0.18 * math.cos(r), y + 0.18 * math.sin(r)))
            pts.append((x, y))
        self.assertEqual(_feed(RammingDetector(), pts), [])

    def test_straight_driving_does_not_fire(self):
        pts = [(i * 0.4, 0.0) for i in range(12)]
        self.assertEqual(_feed(RammingDetector(), pts), [])

    def test_latch_is_positional(self):
        det = RammingDetector()
        pts = _sawtooth((0.0, 0.0), (0.55, 0.0), cycles=3)
        self.assertEqual(len(_feed(det, pts)), 1)
        t = len(pts) * 2.0
        # Same spot again: suppressed while the robot stays there.
        more = _sawtooth((0.0, 0.0), (0.55, 0.0), cycles=3)
        self.assertEqual(_feed(det, more, t0=t), [])
        t += len(more) * 2.0
        # Robot leaves, rams somewhere else: fires again.
        away = [(1.5, 0.0), (2.5, 0.0), (3.5, 0.0),
                (3.5, 0.0), (3.5, 0.0), (3.5, 0.0), (3.5, 0.0)]
        _feed(det, away, t0=t)
        t += len(away) * 2.0
        pts2 = _sawtooth((3.5, 0.0), (4.05, 0.0), cycles=3)
        events = _feed(det, pts2, t0=t)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].centre.x, 4.05, delta=0.1)


class TestHandlingFollowsData(unittest.TestCase):
    """The rule at controller level: phases entered by
    sending commands (cooldown after hand-back, consult while
    paused) must not blind the adjustment logic."""

    def test_cooldown_still_detects_ramming(self):
        ctl = BugNavController()
        ctl.phase = Phase.COOLDOWN
        ctl._cooldown_until = 1e9  # timer nowhere near expiry
        kinds = []
        pts = _sawtooth((0.0, 0.0), (0.55, 0.0), cycles=3)
        for s in (Sample(t=i * 2.0, pos=Point(x, y),
                         state=RobotState.CLEANING)
                  for i, (x, y) in enumerate(pts)):
            kinds.extend(c.kind for c in ctl.tick(s))
        self.assertIn(CommandKind.PAUSE, kinds)
        self.assertNotEqual(ctl.phase, Phase.COOLDOWN)

    def test_consult_cancelled_when_robot_cleans_away(self):
        ctl = BugNavController()
        ctl.phase = Phase.CONSULT
        ctl.pending_consult = ConsultRequest(
            kind="full_circle", situation={}, t=0.0)
        ctl._consult_pos = Point(0.0, 0.0)
        cmds = ctl.tick(Sample(t=10.0, pos=Point(1.0, 0.0),
                               state=RobotState.CLEANING))
        self.assertEqual(ctl.phase, Phase.MONITOR)
        self.assertIsNone(ctl.pending_consult)
        self.assertTrue(any(
            c.kind == CommandKind.NOTIFY
            and "consult cancelled" in c.message for c in cmds))

    def test_consult_waits_while_robot_stays_paused(self):
        ctl = BugNavController()
        ctl.phase = Phase.CONSULT
        ctl.pending_consult = ConsultRequest(
            kind="full_circle", situation={}, t=0.0)
        ctl._consult_pos = Point(0.0, 0.0)
        cmds = ctl.tick(Sample(t=10.0, pos=Point(0.05, 0.0),
                               state=RobotState.PAUSED))
        self.assertEqual(cmds, [])
        self.assertEqual(ctl.phase, Phase.CONSULT)


if __name__ == "__main__":
    unittest.main()
