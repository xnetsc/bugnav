"""Wall confirmation by the map (user rule, 2026-11-12).

Whether an obstacle is a WALL (a straight edge, a room corner) is
decided by the map, not by the shape of the robot's motion: if the
spot sits at the edge of the known cleanable area - a substantial
arc of the ring around it was never cleaned, because there is no
room there - it IS a wall.  Anything NOT confirmed at the area
edge must not be treated as a wall: an interior obstacle is
furniture, possibly a shape with an opening worth diving into,
and the pocket logic applies.

The footprint is the ever-swept set: every cell ever observed
cleaned, in any bout, updated in real time as new cells are
swept (user rule 2026-11-12: "swept" means swept in history,
and the map must stay live).  It is deliberately NOT the
supervisor's adopted home baseline - that estimate is
conservative and lags; a collision point ringed by historically
swept floor is not a wall whatever the baseline says.  Grid:
shared 0.5 m cells, keys (round(x/0.5), round(y/0.5)), as in
supervisor.py.
"""

import math

from .geometry import Point

CELL_M = 0.5


def cell_of(p: Point) -> tuple[int, int]:
    return (round(p.x / CELL_M), round(p.y / CELL_M))


def outside_arc_deg(centre: Point, footprint: set, radius_m: float,
                    step_deg: float = 5.0) -> float:
    """Largest contiguous arc (degrees) of the ring around centre
    whose probe points fall OUTSIDE the footprint - i.e. in space
    that has never been cleaned."""
    if not footprint:
        return 0.0
    n = int(round(360.0 / step_deg))
    outside = []
    for i in range(n):
        a = math.radians(i * step_deg)
        probe = Point(centre.x + radius_m * math.cos(a),
                      centre.y + radius_m * math.sin(a))
        outside.append(cell_of(probe) not in footprint)
    if all(outside):
        return 360.0
    best = run = 0
    for v in outside + outside:  # circular run, bounded by the all-case
        run = run + 1 if v else 0
        if run > best:
            best = run
        if best >= n:
            break
    return min(best, n) * step_deg


def wall_by_map(centre: Point, footprint: set, radius_m: float,
                min_outside_arc_deg: float = 120.0) -> tuple[bool, float]:
    """(is_wall, outside_arc_deg).  With no footprint there is no
    map evidence, so nothing is a wall (user rule: a wall must be
    CONFIRMED by the map edge, never assumed)."""
    arc = outside_arc_deg(centre, footprint, radius_m)
    return (arc >= min_outside_arc_deg, arc)
