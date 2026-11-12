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

import json
import math
import os

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


class SweptHistory:
    """Ever-swept cells, bound to ONE map at a time (user rule
    2026-11-12: the history must be bound to a specific map - a
    venue change IS a map change, and the same coordinates under
    another map are another place).  One file per map id;
    activate() saves the old map's set before loading the new
    one's, so returning to a map resumes its own history."""

    def __init__(self, data_dir: str):
        self._dir = data_dir
        self.mid: str | None = None
        self.cells: set[tuple[int, int]] = set()

    @staticmethod
    def _key(mid: str) -> str:
        return "".join(
            c if (c.isalnum() or c in "-_") else "_" for c in str(mid))

    def _path(self, mid: str) -> str:
        return os.path.join(
            self._dir, f"swept_cells_{self._key(mid)}.json")

    @property
    def _legacy_path(self) -> str:
        return os.path.join(self._dir, "swept_cells.json")

    def _any_keyed_file(self) -> bool:
        try:
            return any(fn.startswith("swept_cells_")
                       and fn.endswith(".json")
                       for fn in os.listdir(self._dir))
        except OSError:
            return False

    @staticmethod
    def _load(path: str) -> set:
        try:
            with open(path, encoding="utf-8") as f:
                return {tuple(c) for c in json.load(f)["cells"]}
        except Exception:  # noqa: BLE001 - a lost file is just empty
            return set()

    def activate(self, mid: str, seed: set | None = None) -> bool:
        """Bind the history to `mid`.  Returns True when the bound
        map actually changed (first activation included).  Cells
        gathered before any binding (the brief window before the
        first map-id read) are kept: the physical map in use
        during that window is the one being bound."""
        mid = str(mid)
        if mid == self.mid:
            return False
        pending = self.cells if self.mid is None else set()
        self.save()
        self.mid = mid
        path = self._path(mid)
        if os.path.exists(path):
            self.cells = self._load(path)
        elif (os.path.exists(self._legacy_path)
              and not self._any_keyed_file()):
            # Pre-keying builds kept one unbound file; it was
            # accumulated on the then-current map, so the FIRST
            # map bound under keying adopts it - and only that
            # map: once any keyed file exists, the legacy file
            # can never seed another map's history.
            self.cells = self._load(self._legacy_path)
        else:
            self.cells = set(seed or set())
        self.cells |= pending
        return True

    def save(self) -> None:
        if self.mid is None:
            return
        os.makedirs(self._dir, exist_ok=True)
        with open(self._path(self.mid), "w", encoding="utf-8") as f:
            json.dump({"cells": sorted(self.cells)}, f)

    def add(self, cell: tuple[int, int]) -> bool:
        if cell in self.cells:
            return False
        self.cells.add(cell)
        return True
