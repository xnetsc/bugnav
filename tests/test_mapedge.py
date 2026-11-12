"""Wall confirmation by the map (user rule, 2026-11-12).

A bump does not automatically start a gap-hunt: whether the
obstacle is a wall is decided by the map.  At the edge of the
known cleanable area (a big never-cleaned arc around the spot)
it IS a wall/corner and gets no dive-in episode; anything else -
interior furniture, a U-chair included - is not a wall."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bugnav.controller import BugNavController, Phase
from bugnav.geometry import Point
from bugnav.mapedge import outside_arc_deg, wall_by_map
from bugnav.model import CommandKind, Sample

O = Point(0.0, 0.0)


def block(i0, i1, j0, j1):
    return {(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)}


class TestOutsideArc(unittest.TestCase):
    def test_straight_wall_half_plane(self):
        fp = block(-30, 0, -30, 30)  # cleaned: x <= 0 side
        wall, arc = wall_by_map(O, fp, 1.0)
        self.assertTrue(wall)
        self.assertGreaterEqual(arc, 150.0)

    def test_room_corner(self):
        fp = block(-30, 0, -30, 0)  # cleaned: one quadrant only
        wall, arc = wall_by_map(O, fp, 1.0)
        self.assertTrue(wall)
        self.assertGreaterEqual(arc, 240.0)

    def test_interior_obstacle_is_not_a_wall(self):
        fp = block(-30, 30, -30, 30)
        wall, arc = wall_by_map(O, fp, 1.0)
        self.assertFalse(wall)
        self.assertEqual(arc, 0.0)

    def test_unentered_u_chair_is_not_a_wall(self):
        # The chair body + its unentered interior are missing from
        # the footprint, but the ring BEYOND the chair is all room.
        fp = block(-30, 30, -30, 30) - block(-1, 1, -1, 1)
        wall, arc = wall_by_map(O, fp, 1.0)
        self.assertFalse(wall)

    def test_no_footprint_no_wall(self):
        # A wall must be CONFIRMED by the map, never assumed.
        self.assertEqual(wall_by_map(O, set(), 1.0), (False, 0.0))
        self.assertEqual(outside_arc_deg(O, set(), 1.0), 0.0)


def circle_samples(centre, t0, n=20, radius=0.45, step_deg=24.0, dt=2.5):
    out = []
    for i in range(n):
        a = math.radians(step_deg * i)
        pos = Point(centre.x + radius * math.cos(a),
                    centre.y + radius * math.sin(a))
        out.append(Sample(t=t0 + dt * i, pos=pos,
                          heading_deg=(step_deg * i + 90) % 360))
    return out


class TestPerimeterGate(unittest.TestCase):
    def _run(self, ctl, samples):
        cmds = []
        for s in samples:
            cmds.extend(ctl.tick(s))
        return cmds

    def test_loiter_at_map_edge_is_a_wall_no_episode(self):
        ctl = BugNavController()
        ctl.set_footprint(block(-30, 0, -30, 30))
        cmds = self._run(ctl, circle_samples(O, 0.0))
        notes = [c for c in cmds if c.kind == CommandKind.NOTIFY]
        self.assertTrue(
            any("wall/corner by the map edge" in c.message
                for c in notes), notes)
        self.assertFalse(
            any(c.kind == CommandKind.PAUSE for c in cmds), cmds)
        self.assertEqual(ctl.phase, Phase.MONITOR)

    def test_loiter_in_interior_still_takes_over(self):
        ctl = BugNavController()
        ctl.set_footprint(block(-30, 30, -30, 30))
        cmds = self._run(ctl, circle_samples(O, 0.0))
        notes = [c for c in cmds if c.kind == CommandKind.NOTIFY]
        self.assertTrue(
            any("interior obstacle, not a wall" in c.message
                for c in notes), notes)
        self.assertTrue(
            any(c.kind == CommandKind.PAUSE for c in cmds), cmds)
        self.assertNotEqual(ctl.phase, Phase.MONITOR)


if __name__ == "__main__":
    unittest.main()
