"""Perimeter-loiter trigger: the firmware circles an obstacle and
passes its opening by (live case: U-shaped chair base, 2026-11-11).

First loiter at a spot = observation only.  A repeat loiter at the same
spot in the same session = takeover, goal = centre of the circled area.
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bugnav.controller import BugNavController, Phase
from bugnav.detector import PerimeterDetector
from bugnav.geometry import Point
from bugnav.model import CommandKind, Sample


def circle_samples(centre, t0, n=20, radius=0.45, step_deg=24.0, dt=2.5):
    """Walk a full circle (plus a bit) around `centre`."""
    out = []
    for i in range(n):
        a = math.radians(step_deg * i)
        pos = Point(centre.x + radius * math.cos(a), centre.y + radius * math.sin(a))
        out.append(Sample(t=t0 + dt * i, pos=pos, heading_deg=(step_deg * i + 90) % 360))
    return out


def straight_samples(start, heading_deg, t0, n=12, step_m=0.75, dt=2.5):
    out = []
    rad = math.radians(heading_deg)
    for i in range(n):
        pos = Point(start.x + step_m * i * math.cos(rad), start.y + step_m * i * math.sin(rad))
        out.append(Sample(t=t0 + dt * i, pos=pos, heading_deg=heading_deg))
    return out


class TestPerimeterDetector(unittest.TestCase):
    def test_circling_fires_once(self):
        det = PerimeterDetector()
        events = [det.feed(s) for s in circle_samples(Point(0, 0), 0.0)]
        fired = [e for e in events if e is not None]
        self.assertEqual(len(fired), 1, f"events={fired}")
        self.assertAlmostEqual(fired[0].centre.x, 0.0, delta=0.2)
        self.assertAlmostEqual(fired[0].centre.y, 0.0, delta=0.2)

    def test_straight_bow_lines_do_not_fire(self):
        det = PerimeterDetector()
        samples = []
        t = 0.0
        pos = Point(-6.0, 0.0)
        for line in range(4):
            seg = straight_samples(pos, 0.0, t, n=16)
            samples += seg
            t = seg[-1].t + 2.5
            pos = Point(seg[-1].pos.x, seg[-1].pos.y + 0.4)
            seg = straight_samples(pos, 180.0, t, n=16)
            samples += seg
            t = seg[-1].t + 2.5
            pos = Point(seg[-1].pos.x, seg[-1].pos.y + 0.4)
        fired = [det.feed(s) for s in samples]
        self.assertTrue(all(e is None for e in fired))

    def test_parked_does_not_fire(self):
        det = PerimeterDetector()
        fired = [
            det.feed(Sample(t=2.5 * i, pos=Point(1.0, 1.0)))
            for i in range(40)
        ]
        self.assertTrue(all(e is None for e in fired))

    def test_rearm_requires_leaving_the_disc(self):
        det = PerimeterDetector()
        first = circle_samples(Point(0, 0), 0.0)
        self.assertTrue(any(det.feed(s) for s in first))
        # Keep circling in place: latched, no second event.
        more = circle_samples(Point(0, 0), first[-1].t + 2.5)
        self.assertTrue(all(det.feed(s) is None for s in more))
        # Leave, come back, circle again: fires again.
        away = straight_samples(more[-1].pos, 0.0, more[-1].t + 2.5, n=6)
        for s in away:
            det.feed(s)
        back = straight_samples(away[-1].pos, 180.0, away[-1].t + 2.5, n=11)
        for s in back:
            det.feed(s)
        again = circle_samples(Point(0, 0), back[-1].t + 2.5)
        self.assertTrue(any(det.feed(s) for s in again))


class TestPerimeterEscalation(unittest.TestCase):
    def _run(self, ctl, samples):
        cmds = []
        for s in samples:
            cmds.extend(ctl.tick(s))
        return cmds

    def test_first_loiter_observes_second_takes_over(self):
        ctl = BugNavController()
        centre = Point(0.0, 0.0)
        first = circle_samples(centre, 0.0)
        cmds = self._run(ctl, first)
        notes = [c for c in cmds if c.kind == CommandKind.NOTIFY]
        self.assertTrue(any("perimeter loiter #1" in c.message for c in notes), notes)
        self.assertFalse(any(c.kind == CommandKind.PAUSE for c in cmds))
        self.assertEqual(ctl.phase, Phase.MONITOR)

        away = straight_samples(first[-1].pos, 0.0, first[-1].t + 2.5, n=8)
        self._run(ctl, away)
        back = straight_samples(away[-1].pos, 180.0, away[-1].t + 2.5, n=15)
        self._run(ctl, back)
        self.assertEqual(ctl.phase, Phase.MONITOR)

        second = circle_samples(centre, back[-1].t + 2.5)
        cmds = self._run(ctl, second)
        self.assertTrue(any(c.kind == CommandKind.PAUSE for c in cmds), cmds)
        self.assertNotEqual(ctl.phase, Phase.MONITOR)
        self.assertIsNotNone(ctl._goal)
        self.assertLess(ctl._goal.distance_to(centre), 0.35, f"goal={ctl._goal}")

    def test_loiter_at_new_spot_counts_separately(self):
        ctl = BugNavController()
        first = circle_samples(Point(0.0, 0.0), 0.0)
        self._run(ctl, first)
        away = straight_samples(first[-1].pos, 45.0, first[-1].t + 2.5, n=10)
        self._run(ctl, away)
        other = circle_samples(Point(4.0, 4.0), away[-1].t + 2.5)
        cmds = self._run(ctl, other)
        notes = [c for c in cmds if c.kind == CommandKind.NOTIFY]
        self.assertTrue(any("perimeter loiter #1" in c.message for c in notes), notes)
        self.assertEqual(ctl.phase, Phase.MONITOR)


if __name__ == "__main__":
    unittest.main()
