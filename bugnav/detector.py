"""Struggle (precursor) detection.

The stock firmware has no "give up" concept: it keeps climbing an
obstacle until a drive wheel leaves the ground and only then reports an
error.  Waiting for the error is too late, so the detector watches for
the precursor pattern instead: over a sliding window the robot stays
inside a small radius, travels a non-trivial path length (it *is*
trying), changes direction repeatedly, and makes almost no net
progress.
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
