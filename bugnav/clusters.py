"""Obstacle clusters: hit points that belong to one obstacle are merged.

Four table legs plus a crossbar are one obstacle, not four.  Merging by
distance makes the perimeter exploration walk the outline of the whole
furniture piece and try every leg gap in one episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import Point


@dataclass
class ObstacleCluster:
    hits: list[Point] = field(default_factory=list)
    failed_angles_deg: list[float] = field(default_factory=list)

    def add_hit(self, p: Point) -> None:
        self.hits.append(p)

    @property
    def centre(self) -> Point:
        n = len(self.hits)
        return Point(
            sum(p.x for p in self.hits) / n,
            sum(p.y for p in self.hits) / n,
        )

    @property
    def radius_m(self) -> float:
        c = self.centre
        if not self.hits:
            return 0.0
        return max(p.distance_to(c) for p in self.hits)

    def grid_key(self, cell_m: float = 0.5) -> str:
        c = self.centre
        return f"{round(c.x / cell_m)}:{round(c.y / cell_m)}"


class ClusterManager:
    def __init__(self, merge_distance_m: float = 1.0):
        self.merge_distance_m = merge_distance_m
        self.clusters: list[ObstacleCluster] = []

    def find_or_create(self, hit: Point) -> ObstacleCluster:
        for cl in self.clusters:
            if hit.distance_to(cl.centre) <= self.merge_distance_m + cl.radius_m:
                cl.add_hit(hit)
                return cl
        cl = ObstacleCluster(hits=[hit])
        self.clusters.append(cl)
        return cl

    def nearest(self, p: Point) -> ObstacleCluster | None:
        best, best_d = None, float("inf")
        for cl in self.clusters:
            d = p.distance_to(cl.centre) - cl.radius_m
            if d < best_d:
                best, best_d = cl, d
        return best
