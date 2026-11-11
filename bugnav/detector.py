"""Struggle (precursor) detection, and perimeter-loiter detection.

The stock firmware has no "give up" concept: it keeps climbing an
obstacle until a drive wheel leaves the ground and only then reports an
error.  Waiting for the error is too late, so the detector watches for
the precursor pattern instead: over a sliding window the robot stays
inside a small radius, travels a non-trivial path length (it *is*
trying), changes direction repeatedly, and makes almost no net
progress.

Perimeter loiter is the second failure shape, seen live on 2026-11-11:
the firmware *does* walk around a U-shaped chair base, reaches the
opening face - and then simply passes it by, because following the
perimeter is all it knows.  That is not a struggle (the robot moves
freely, over a wider disc, mostly turning), so the struggle detector
rightly ignores it.  The perimeter detector watches for it instead:
over a longer window the robot stays inside a ~0.65 m disc, travels
several metres, and accumulates a large total turn while ending up
about where it started.

Calibration note (real 2026-11-11 session trace): this pattern also
occurs during the firmware's normal intensive local cleaning, roughly
ten times per hour-long session, at about seven distinct spots.  Per
the user's settled model that does not gate anything: an obstacle
encounter goes straight into the along-edge logic in every scenario,
so the controller takes over on the first loiter at a spot (and only
once per spot per session - the episode's outcome then stands).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .geometry import angle_diff_deg, angle_of
from .model import RobotState, Sample


@dataclass
class DetectorConfig:
    window_s: float = 30.0
    radius_m: float = 0.28  # all samples in window inside this radius
    min_path_m: float = 0.55  # it really is trying, not just paused
    max_net_m: float = 0.22  # ...but getting nowhere
    min_reversals: int = 2  # heading reversals inside the window
    attempt_window_s: float = 18.0  # shorter budget per guided attempt


@dataclass
class StruggleEvent:
    t: float
    centre: object  # Point
    approach_heading_deg: float  # direction of travel before the struggle


class StruggleDetector:
    def __init__(self, config: DetectorConfig | None = None):
        self.cfg = config or DetectorConfig()
        self._buf: deque[Sample] = deque()
        self._latched = False

    def reset(self) -> None:
        self._buf.clear()
        self._latched = False

    def feed(self, sample: Sample, *, window_s: float | None = None) -> StruggleEvent | None:
        if sample.state not in (RobotState.CLEANING, RobotState.ERROR):
            self._buf.clear()
            self._latched = False
            return None
        win = window_s if window_s is not None else self.cfg.window_s
        self._buf.append(sample)
        while self._buf and sample.t - self._buf[0].t > win:
            self._buf.popleft()
        if self._latched or len(self._buf) < 3:
            return None
        if sample.t - self._buf[0].t < win * 0.9:
            return None  # window not full yet

        pts = [s.pos for s in self._buf]
        centre = pts[-1]
        if any(p.distance_to(centre) > self.cfg.radius_m for p in pts):
            return None
        path = sum(a.distance_to(b) for a, b in zip(pts, pts[1:]))
        net = pts[0].distance_to(pts[-1])
        if path < self.cfg.min_path_m or net > self.cfg.max_net_m:
            return None

        # Heading reversals: count sign changes of the turn direction
        # between consecutive movement vectors (back-and-forth counts too,
        # via near-180 turns).
        vecs = [
            (b - a)
            for a, b in zip(pts, pts[1:])
            if a.distance_to(b) > 0.03
        ]
        reversals = 0
        for v1, v2 in zip(vecs, vecs[1:]):
            if angle_diff_deg(angle_of(v1), angle_of(v2)) > 100.0:
                reversals += 1
        if reversals < self.cfg.min_reversals:
            return None

        # Approach heading: direction of travel in the first part of the
        # window, before the robot settled into the struggle.
        approach = angle_of(vecs[0]) if vecs else (sample.heading_deg or 0.0)
        self._latched = True
        return StruggleEvent(t=sample.t, centre=centre, approach_heading_deg=approach)


@dataclass
class PerimeterConfig:
    window_s: float = 40.0
    radius_m: float = 0.65  # all samples in window inside this disc
    min_path_m: float = 1.8  # it really is travelling, not parked
    max_net_m: float = 0.45  # ...but ends up about where it started
    min_sweep_deg: float = 270.0  # cumulative absolute turn of travel
    rearm_extra_m: float = 0.5  # must leave disc + this before re-firing


@dataclass
class PerimeterEvent:
    t: float
    centre: object  # Point: centroid of the loiter window
    ring_points: list  # up to 4 extreme window points (obstacle outline)
    path_m: float
    sweep_deg: float


class PerimeterDetector:
    """Detects the robot travelling in circles/loops inside one small
    disc - the signature of perimeter-following around an obstacle it
    never enters.

    Deliberately state-agnostic: on the current firmware the reported
    robot state can be stuck at "docked" for a whole clean (see
    work_state.py), so gating on state would blind this detector
    exactly when it matters.  The geometric criteria gate everything:
    a parked robot has no path, a robot crossing the house has no
    confinement, a robot on straight runs has no sweep.
    """

    def __init__(self, config: PerimeterConfig | None = None):
        self.cfg = config or PerimeterConfig()
        self._buf: deque[Sample] = deque()
        self._latched_centre = None  # Point while suppressed

    def reset(self) -> None:
        self._buf.clear()
        self._latched_centre = None

    def feed(self, sample: Sample) -> PerimeterEvent | None:
        if self._latched_centre is not None:
            if sample.pos.distance_to(self._latched_centre) > (
                self.cfg.radius_m + self.cfg.rearm_extra_m
            ):
                self._latched_centre = None
                self._buf.clear()
            else:
                return None
        self._buf.append(sample)
        while self._buf and sample.t - self._buf[0].t > self.cfg.window_s:
            self._buf.popleft()
        if len(self._buf) < 5 or sample.t - self._buf[0].t < self.cfg.window_s * 0.9:
            return None

        pts = [s.pos for s in self._buf]
        n = len(pts)
        centre = type(pts[0])(
            sum(p.x for p in pts) / n, sum(p.y for p in pts) / n
        )
        if any(p.distance_to(centre) > self.cfg.radius_m for p in pts):
            return None
        path = sum(a.distance_to(b) for a, b in zip(pts, pts[1:]))
        if path < self.cfg.min_path_m:
            return None
        if pts[0].distance_to(pts[-1]) > self.cfg.max_net_m:
            return None

        vecs = [
            (b - a)
            for a, b in zip(pts, pts[1:])
            if a.distance_to(b) > 0.03
        ]
        sweep = 0.0
        for v1, v2 in zip(vecs, vecs[1:]):
            sweep += angle_diff_deg(angle_of(v1), angle_of(v2))
        if sweep < self.cfg.min_sweep_deg:
            return None

        # Outline estimate for clustering: the window's extreme points.
        ring = [
            min(pts, key=lambda p: p.x),
            max(pts, key=lambda p: p.x),
            min(pts, key=lambda p: p.y),
            max(pts, key=lambda p: p.y),
        ]
        ring = list(dict.fromkeys(ring))
        self._latched_centre = centre
        return PerimeterEvent(
            t=sample.t, centre=centre, ring_points=ring,
            path_m=path, sweep_deg=sweep,
        )
