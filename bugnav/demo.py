"""Run the three built-in scenarios and print a summary.

  python -m bugnav.demo
"""

from __future__ import annotations

from .scenarios import run_scenario


def main() -> None:
    for name in ("u_base", "table_leg", "sealed_box"):
        controller, robot, notes, reached = run_scenario(name)
        stats = list(controller.stats.values())
        outcome = (
            "SUCCESS"
            if any(s.successes > 0 for s in stats)
            else "ABANDONED"
            if any(s.abandons > 0 for s in stats)
            else "UNRESOLVED"
        )
        print(f"== {name}: {outcome}, reached_goal={reached}, "
              f"robot_end=({robot.pos.x:.2f},{robot.pos.y:.2f})")
        for n in notes:
            print(f"   - {n}")


if __name__ == "__main__":
    main()
