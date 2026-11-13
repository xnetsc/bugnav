"""A small 2D simulator used to validate the controller end to end.

The simulated *stock* robot is deliberately dumb in the same way the
real firmware is: in AUTO mode it drives straight at its goal, and when
it meets an obstacle it jiggles back and forth forever - it never gives
up and never replans.  When it receives a MOVE_TO command, it uses a
grid A* planner over the walls it can see, like the real robot's
map-based planner would: it can route *around* an obstacle, but it
cannot pass through a closed side.  Pause, resume, start and jog
commands move it the way the real command channel does.

Walls are line segments; the robot is a point with a clearance radius.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from .geometry import Point
from .model import Command, CommandKind, RobotState, Sample


@dataclass(frozen=True)
class Wall:
    a: Point
    b: Point


def _point_segment_distance(p: Point, w: Wall) -> float:
    ax, ay, bx, by = w.a.x, w.a.y, w.b.x, w.b.y
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(p.x - ax, p.y - ay)
    t = max(0.0, min(1.0, ((p.x - ax) * dx + (p.y - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p.x - (ax + t * dx), p.y - (ay + t * dy))


class World:
    def __init__(self, walls: list[Wall], bounds: float = 3.5):
        self.walls = walls
        self.bounds = bounds

    def blocked(self, p: Point, clearance: float) -> bool:
        if abs(p.x) > self.bounds or abs(p.y) > self.bounds:
            return True
        return any(_point_segment_distance(p, w) < clearance for w in self.walls)


class _AStar:
    """Grid A* over the world; returns a waypoint list or None."""

    def __init__(self, world: World, clearance: float, cell: float = 0.10):
        self.world = world
        self.clearance = clearance
        self.cell = cell
        b = world.bounds
        self.n = int(2 * b / cell) + 1
        self.b = b

    def _idx(self, p: Point) -> tuple[int, int]:
        return (
            round((p.x + self.b) / self.cell),
            round((p.y + self.b) / self.cell),
        )

    def _pt(self, i: int, j: int) -> Point:
        return Point(i * self.cell - self.b, j * self.cell - self.b)

    def _free(self, i: int, j: int) -> bool:
        if not (0 <= i < self.n and 0 <= j < self.n):
            return False
        return not self.world.blocked(self._pt(i, j), self.clearance)

    def plan(self, start: Point, goal: Point) -> list[Point] | None:
        s, g = self._idx(start), self._idx(goal)
        if not self._free(*g):
            # Goal cell itself inside an obstacle: aim for the goal anyway;
            # the caller treats "no path" via the direct-drive fallback.
            return None
        open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, s)]
        came: dict[tuple[int, int], tuple[int, int]] = {}
        cost = {s: 0.0}
        while open_heap:
            _, cur = heapq.heappop(open_heap)
            if cur == g:
                path = [cur]
                while path[-1] != s:
                    path.append(came[path[-1]])
                path.reverse()
                pts = [self._pt(i, j) for i, j in path]
                pts[-1] = goal
                return pts
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nb = (cur[0] + di, cur[1] + dj)
                if not self._free(*nb):
                    continue
                nc = cost[cur] + math.hypot(di, dj) * self.cell
                if nb not in cost or nc < cost[nb]:
                    cost[nb] = nc
                    came[nb] = cur
                    h = math.hypot(nb[0] - g[0], nb[1] - g[1]) * self.cell
                    heapq.heappush(open_heap, (nc + h, nb))
        return None


@dataclass
class SimRobot:
    world: World
    pos: Point
    goal: Point  # the cell the stock firmware insists on reaching
    speed: float = 0.30  # m/s
    radius: float = 0.17
    mode: str = "auto"  # auto | command | paused | idle
    _path: list[Point] = field(default_factory=list)
    _direct: bool = False
    _target: Point | None = None
    _jiggle_phase: int = 0
    _contact: Point | None = None

    # ------------------------------------------------------------- motion
    def _step_toward(self, target: Point, dt: float) -> bool:
        """Move one step toward target. Returns True if we moved."""
        d = self.pos.distance_to(target)
        if d < 1e-6:
            return True
        step = min(d, self.speed * dt)
        nxt = self.pos + (target - self.pos).scale(step / d)
        if self.world.blocked(nxt, self.radius):
            return False
        self.pos = nxt
        return True

    def _follow_path(self, dt: float) -> str:
        """Follow the A* path. Returns 'arrived' | 'moving' | 'blocked'."""
        while self._path and self.pos.distance_to(self._path[0]) < 0.05:
            self._path.pop(0)
        if not self._path:
            return "arrived"
        return "moving" if self._step_toward(self._path[0], dt) else "blocked"

    def _plan_to(self, target: Point) -> None:
        self._target = target
        path = _AStar(self.world, self.radius).plan(self.pos, target)
        self._path = path if path else []
        self._direct = path is None  # no route: drive straight until contact

    # ------------------------------------------------------------ commands
    def execute(self, cmd: Command) -> None:
        if cmd.kind == CommandKind.PAUSE:
            self.mode = "paused"
        elif cmd.kind == CommandKind.STOP:
            self.mode = "idle"
            self._path = []
        elif cmd.kind in (CommandKind.RESUME_AUTO,
                          CommandKind.START_AUTO):
            self.mode = "auto"
            self._contact = None
            self._path = []
        elif cmd.kind == CommandKind.MOVE_TO and cmd.point is not None:
            self.mode = "command"
            self._contact = None
            self._plan_to(cmd.point)
        elif cmd.kind == CommandKind.JOG:
            # One dumb straight step, no planner: backward backs
            # away from the goal the firmware insists on reaching,
            # and whatever is behind blocks it, like the real thing.
            if cmd.act in ("backward", "forward"):
                away = self.pos - self.goal
                if away.length() > 1e-6:
                    step = away.normalized().scale(
                        0.35 if cmd.act == "backward" else -0.35
                    )
                    dest = self.pos + step
                    if not self.world.blocked(dest, self.radius):
                        self.pos = dest

    # ---------------------------------------------------------------- tick
    def tick(self, t: float, dt: float) -> Sample:
        error = 0
        if self.mode == "auto":
            # The stock firmware drives straight at the goal and, when
            # blocked, jiggles forever. It does NOT route around.
            if self.pos.distance_to(self.goal) < 0.08:
                self.mode = "idle"
            elif not self._step_toward(self.goal, dt):
                if self._contact is None:
                    self._contact = self.pos
                self._jiggle(t, dt)
        elif self.mode == "command":
            if self._direct and self._target is not None:
                self._step_toward(self._target, dt)  # stops at contact
            else:
                self._follow_path(dt)
        if self.mode in ("auto", "command"):
            state = RobotState.CLEANING
        elif self.mode == "paused":
            state = RobotState.PAUSED
        else:
            state = RobotState.IDLE
        return Sample(t=t, pos=self.pos, state=state, error_code=error)

    def _jiggle(self, t: float, dt: float) -> None:
        """The stock firmware's futile struggle: back and forth forever."""
        base = self._contact or self.pos
        # Direction back toward the goal (it keeps trying the same way).
        toward = (self.goal - base).normalized()
        side = Point(-toward.y, toward.x)
        offsets = [
            side.scale(0.10) + toward.scale(-0.02),
            toward.scale(-0.12),
            side.scale(-0.10) + toward.scale(-0.02),
            toward.scale(-0.12),
        ]
        idx = int(t // 1.5) % len(offsets)
        dest = base + offsets[idx]
        if not self.world.blocked(dest, self.radius):
            # Move at struggle speed toward the jiggle offset.
            d = self.pos.distance_to(dest)
            if d > 1e-6:
                step = min(d, self.speed * 0.6 * dt)
                self.pos = self.pos + (dest - self.pos).scale(step / d)
