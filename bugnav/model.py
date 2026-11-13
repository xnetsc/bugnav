"""Data model shared by the core controller, adapters and the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .geometry import Point


class RobotState(str, Enum):
    IDLE = "idle"
    CLEANING = "cleaning"
    PAUSED = "paused"
    RETURNING = "returning"
    DOCKED = "docked"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Sample:
    """One observation of the robot at time `t` (seconds, monotonic)."""

    t: float
    pos: Point
    state: RobotState = RobotState.CLEANING
    heading_deg: float | None = None  # from GetPos 'a' when the adapter has it
    error_code: int = 0  # 0 = no error


class CommandKind(str, Enum):
    PAUSE = "pause"
    STOP = "stop"
    RESUME_AUTO = "resume_auto"
    # Start a FRESH AUTO clean, whatever session state the robot is
    # in.  Needed after MOVE_TO hops: a mapPoint command replaces
    # the running session, and once it completes the robot sits
    # paused with nothing to resume - deebot-client even rewrites a
    # plain START into a bare resume while paused, which is then a
    # no-op (robot stayed paused 100 s).
    # The adapter sends the raw clean_V2 start form, which the
    # rewrite cannot touch; the robot starts cleaning at once.
    START_AUTO = "start_auto"
    MOVE_TO = "move_to"  # drive to `point` (clean_V2 mapPoint)
    # One remote-control jog step (the 'move' command): a small
    # reposition - back a step, forward a step, turn - that names
    # no destination.  `act` on the Command says which step.
    JOG = "jog"
    RETURN_TO_DOCK = "return_to_dock"  # head back to the station now
    PLAY_SOUND = "play_sound"  # the robot's OWN built-in alert sound
    NOTIFY = "notify"


@dataclass(frozen=True)
class Command:
    kind: CommandKind
    point: Point | None = None  # target for MOVE_TO
    message: str = ""
    count: int = 1  # repetitions (PLAY_SOUND)
    act: str = ""  # jog step for JOG (backward/forward/spinLeft/...)


@dataclass
class EpisodeStats:
    """Cross-session statistics for one obstacle location (never a ban)."""

    grid_key: str
    episodes: int = 0
    successes: int = 0
    abandons: int = 0
    success_headings: list[int] = field(default_factory=list)
