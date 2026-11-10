"""Breadcrumb trail: a rolling record of where the robot has been.

This is the "how did I get here" memory.  It is recorded continuously
while cleaning is healthy, frozen when an obstacle episode starts, and
replayed in reverse to leave a dead end along the proven entry corridor
instead of letting the robot's own planner cut straight through the
obstacle.
"""

from __future__ import annotations

from .geometry import Point
from .model import Sample


class BreadcrumbTrail:
    def __init__(self, max_length_m: float = 12.0, min_spacing_m: float = 0.10):
        self.max_length_m = max_length_m
        self.min_spacing_m = min_spacing_m
        self._points: list[Point] = []
        self._frozen: list[Point] | None = None

    def record(self, sample: Sample) -> None:
        if self._frozen is not None:
            return
        p = sample.pos
        if self._points and p.distance_to(self._points[-1]) < self.min_spacing_m:
            return
        self._points.append(p)
        # Trim from the front so total path length stays within budget.
        total = self.path_length(self._points)
        while total > self.max_length_m and len(self._points) > 2:
            total -= self._points[0].distance_to(self._points[1])
            self._points.pop(0)

    @staticmethod
    def path_length(points: list[Point]) -> float:
        return sum(a.distance_to(b) for a, b in zip(points, points[1:]))

    @property
    def points(self) -> list[Point]:
        return list(self._frozen if self._frozen is not None else self._points)

    def freeze(self) -> Point | None:
        """Freeze the trail at episode start; return the entry anchor.

        The anchor is the oldest point still in the trail: the last place
        the robot was provably moving freely before this local area.
        """
        if not self._points:
            return None
        self._frozen = list(self._points)
        return self._frozen[0]

    def unfreeze(self) -> None:
        self._frozen = None
        self._points = []

    def reversed_waypoints(self, spacing_m: float = 0.45) -> list[Point]:
        """Entry path in reverse, thinned to roughly `spacing_m` steps.

        Every waypoint was actually traversed by the robot on the way in,
        so the exit corridor is proven passable by construction.
        """
        pts = self.points
        if len(pts) < 2:
            return list(reversed(pts))
        out: list[Point] = [pts[-1]]
        acc = 0.0
        for a, b in zip(reversed(pts[:-1]), reversed(pts[1:])):
            acc += a.distance_to(b)
            if acc >= spacing_m:
                out.append(a)
                acc = 0.0
        if out[-1] != pts[0]:
            out.append(pts[0])
        return out
