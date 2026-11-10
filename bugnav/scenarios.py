"""Built-in simulation scenarios."""

from __future__ import annotations

from .geometry import Point
from .sim import SimRobot, Wall, World


def u_base() -> tuple[World, Point, Point]:
    """U-shaped base, opening facing +x (east). Goal inside the pocket.

    The robot starts west of it and insists on driving straight east,
    into the closed back of the U.
    """
    walls = [
        Wall(Point(-0.5, -0.6), Point(-0.5, 0.6)),  # back (west)
        Wall(Point(-0.5, 0.6), Point(0.5, 0.6)),  # north arm
        Wall(Point(-0.5, -0.6), Point(0.5, -0.6)),  # south arm
    ]
    return World(walls), Point(-2.4, 0.0), Point(0.0, 0.0)


def table_leg() -> tuple[World, Point, Point]:
    """A single table leg (point obstacle). Goal on the far side."""
    walls = [Wall(Point(0.0, 0.0), Point(0.0, 0.0))]
    return World(walls), Point(-2.2, 0.0), Point(0.85, 0.0)


def sealed_box() -> tuple[World, Point, Point]:
    """A fully closed square. The goal inside is unreachable, period."""
    walls = [
        Wall(Point(-0.6, -0.6), Point(-0.6, 0.6)),
        Wall(Point(-0.6, 0.6), Point(0.6, 0.6)),
        Wall(Point(0.6, 0.6), Point(0.6, -0.6)),
        Wall(Point(0.6, -0.6), Point(-0.6, -0.6)),
    ]
    return World(walls), Point(-2.4, 0.0), Point(0.0, 0.0)


SCENARIOS = {"u_base": u_base, "table_leg": table_leg, "sealed_box": sealed_box}


def run_scenario(name: str, max_t: float = 4000.0, dt: float = 0.5):
    """Run controller + SimRobot until the episode resolves.

    Returns (controller, robot, notifications, reached_goal).
    """
    from .controller import BugNavController
    from .model import CommandKind

    world, start, goal = SCENARIOS[name]()
    robot = SimRobot(world=world, pos=start, goal=goal)
    controller = BugNavController()
    notifications: list[str] = []
    reached_goal = False
    end_after: float | None = None
    t = 0.0
    while t < max_t:
        sample = robot.tick(t, dt)
        if sample.pos.distance_to(goal) < 0.15:
            reached_goal = True
        cmds = controller.tick(sample)
        for cmd in cmds:
            if cmd.kind == CommandKind.NOTIFY:
                notifications.append(cmd.message)
            robot.execute(cmd)
        t += dt
        stats = controller.stats
        abandoned = any(s.abandons > 0 for s in stats.values())
        succeeded = any(s.successes > 0 for s in stats.values())
        resumed = any(c.kind == CommandKind.RESUME_AUTO for c in cmds)
        if abandoned and resumed:
            break  # backtrack already finished when RESUME_AUTO is emitted
        if succeeded and resumed and end_after is None:
            end_after = t + 90.0  # give the stock robot time to finish up
        if reached_goal and succeeded:
            break
        if end_after is not None and t >= end_after:
            break
    return controller, robot, notifications, reached_goal
