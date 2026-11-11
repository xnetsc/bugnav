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
    CLEAN_AREA = "clean_area"  # go clean a small rect around `point`
    RETURN_TO_DOCK = "return_to_dock"  # head back to the station now
    PLAY_SOUND = "play_sound"  # the robot's OWN built-in alert sound
    NOTIFY = "notify"


@dataclass(frozen=True)
class Command:
    kind: CommandKind
    point: Point | None = None  # centre for CLEAN_AREA
    rect_half_m: float = 0.30  # half side length of the CLEAN_AREA rect
    message: str = ""
    count: int = 1  # repetitions (PLAY_SOUND)


@dataclass
class EpisodeStats:
    """Cross-session statistics for one obstacle location (never a ban)."""

    grid_key: str
    episodes: int = 0
    successes: int = 0
    abandons: int = 0
    success_headings: list[int] = field(default_factory=list)
