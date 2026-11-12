"""Struggle (precursor) detection, and perimeter-loiter detection.

The stock firmware has no "give up" concept: it keeps climbing an
obstacle until a drive wheel leaves the ground and only then reports an
error.  Waiting for the error is too late, so the detector watches for
the precursor pattern instead: over a sliding window the robot stays
inside a small radius, travels a non-trivial path length (it *is*
trying), changes direction repeatedly, and makes almost no net
progress.

Perimeter loiter is the second failure shape: the firmware *does*
walk around a U-shaped chair base, reaches the opening face - and
then simply passes it by, because following the perimeter is all it
knows.  That is not a struggle (the robot moves freely, over a wider
disc, mostly turning), so the struggle detector rightly ignores it.
The perimeter detector watches for it instead: over a longer window
the robot stays inside a ~0.65 m disc, travels several metres, and
accumulates a large total turn while ending up about where it
started.  This pattern also occurs during the firmware's normal
intensive local cleaning; that gates nothing - an obstacle encounter
goes straight into the along-edge logic in every scenario, so the
controller takes over on the first loiter at a spot (and only once
per spot per session - the episode's outcome then stands).

Ramming is the third shape, and the one that actually hangs wheels:
the firmware backs off a few decimetres and charges the SAME obstacle
face again, trying to climb over it (a U-shaped chair base, charged
frontally until a drive wheel leaves the ground).  The sawtooth
escapes both detectors above - its excursions exceed the struggle
disc and it does not circle - so it gets its own detector:
back-and-forth legs along ONE axis, with the turn-around points
pinned in two tight clusters (the impact face and the run-up end).
It is corrected promptly: the moment the pattern is proven, the
controller pauses the robot and takes over - detection never waits
for an error, and issuing commands never suspends detection
(latches here are positional: a handled spot re-arms by the robot
leaving it, never by a timer or by a command having been sent).
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
        self._latched_centre = None  # Point while suppressed

    def reset(self) -> None:
        self._buf.clear()
        self._latched_centre = None

    def feed(self, sample: Sample, *, window_s: float | None = None) -> StruggleEvent | None:
        if sample.state not in (RobotState.CLEANING, RobotState.ERROR):
            self._buf.clear()
            self._latched_centre = None
            return None
        win = window_s if window_s is not None else self.cfg.window_s
        self._buf.append(sample)
        while self._buf and sample.t - self._buf[0].t > win:
            self._buf.popleft()
        if self._latched_centre is not None:
            # Positional latch (handling never
            # switches off because a command was sent - it re-arms
            # when the robot observably leaves the handled spot).
            if sample.pos.distance_to(self._latched_centre) > (
                self.cfg.radius_m + 0.5
            ):
                self._latched_centre = None
                self._buf.clear()
            else:
                return None
        if len(self._buf) < 3:
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
        self._latched_centre = centre
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


@dataclass
class RammingConfig:
    window_s: float = 60.0
    min_vector_m: float = 0.05  # shorter steps are jitter, not legs
    axis_tolerance_deg: float = 35.0  # legs stay this close to one axis
    min_aligned_share: float = 0.8
    reversal_deg: float = 130.0  # heading flip that ends a leg
    cluster_radius_m: float = 0.22  # turn-around points pin this tight
    # (must stay below min_separation_m, or the two ends merge into
    # one cluster and no pair can ever form)
    min_points_per_end: int = 2  # turn-arounds needed at EACH end
    min_reversals: int = 4  # two full charges at minimum
    strong_leg_m: float = 0.30  # mean leg this long = real run-ups
    min_reversals_weak: int = 5  # shorter legs need one more reversal
    min_separation_m: float = 0.25  # the two ends are distinct...
    max_separation_m: float = 1.10  # ...but it stays one obstacle
    confinement_m: float = 0.95  # whole window near the impact face
    max_net_m: float = 0.60  # it is getting nowhere overall
    min_path_m: float = 1.2  # it really is travelling, back and forth
    rearm_extra_m: float = 0.5  # leaves the face by this -> re-armed


@dataclass
class RammingEvent:
    t: float
    centre: object  # Point: centroid of the impact-face cluster
    approach_heading_deg: float  # run-up end -> impact face direction
    charges: int  # turn-around points counted at the impact face
    mean_leg_m: float


class RammingDetector:
    """Detects frontal charge-and-climb behaviour: repeated runs at
    one obstacle face along a single axis.

    Signature (all over one sliding window): the movement legs are
    collinear (they align with one axis, either direction); the
    heading flips by >130 deg at least four times; the flip points
    form exactly two tight clusters - the impact face and the run-up
    end - a few decimetres apart; and the robot overall gets
    nowhere.  Boustrophedon lane turns fail the confinement and
    cluster tests (their flip points march along the wall), wall
    following fails the flip test (corners turn ~90 deg), and
    corner scrubbing fails the collinearity test (its headings fan
    out).  Like the perimeter detector it latches positionally: a
    handled face re-arms only when the robot leaves it.
    """

    def __init__(self, config: RammingConfig | None = None):
        self.cfg = config or RammingConfig()
        self._buf: deque[Sample] = deque()
        self._latched_centre = None  # Point while suppressed

    def reset(self) -> None:
        self._buf.clear()
        self._latched_centre = None

    def feed(self, sample: Sample) -> RammingEvent | None:
        cfg = self.cfg
        if sample.state not in (RobotState.CLEANING, RobotState.ERROR):
            self._buf.clear()
            return None
        if self._latched_centre is not None:
            if sample.pos.distance_to(self._latched_centre) > (
                cfg.confinement_m + cfg.rearm_extra_m
            ):
                self._latched_centre = None
                self._buf.clear()
            else:
                return None
        self._buf.append(sample)
        while self._buf and sample.t - self._buf[0].t > cfg.window_s:
            self._buf.popleft()
        if len(self._buf) < 6:
            return None

        pts = [s.pos for s in self._buf]
        # Movement legs: (vector, end point, start index in pts).
        # Micro-steps are dropped; what remains must still read as
        # travel.
        legs = []
        for i, (a, b) in enumerate(zip(pts, pts[1:])):
            if a.distance_to(b) > cfg.min_vector_m:
                legs.append((b - a, b, i))
        if len(legs) < 5:
            return None

        # One axis: the longest leg defines it; nearly every leg
        # must run parallel or anti-parallel to it.
        axis = angle_of(max((v for v, _, _ in legs),
                            key=lambda v: v.length()))
        aligned = 0
        for v, _, _ in legs:
            d = angle_diff_deg(angle_of(v), axis)
            if d <= cfg.axis_tolerance_deg or d >= 180.0 - cfg.axis_tolerance_deg:
                aligned += 1
        if aligned / len(legs) < cfg.min_aligned_share:
            return None

        # Flip points: where consecutive legs reverse direction.
        def _sign(v) -> int:
            return 1 if angle_diff_deg(angle_of(v), axis) <= 90.0 else -1

        flips: list = []
        first_flip_leg = None
        for k, ((v1, end1, start1), (v2, _e2, _s2)) in enumerate(
                zip(legs, legs[1:])):
            if _sign(v1) != _sign(v2) and angle_diff_deg(
                angle_of(v1), angle_of(v2)
            ) >= cfg.reversal_deg:
                flips.append(end1)
                if first_flip_leg is None:
                    first_flip_leg = start1
        if len(flips) < cfg.min_reversals:
            return None

        # The pattern's own samples: from the first flip's leg
        # onward.  Whatever the robot did before the pattern (free
        # travel, transit after a previous handling) is not part of
        # it and must not pollute the path / net / confinement
        # measures below.
        pattern = pts[first_flip_leg:] if first_flip_leg is not None else pts
        path = sum(a.distance_to(b) for a, b in zip(pattern, pattern[1:]))
        if path < cfg.min_path_m:
            return None
        if pattern[0].distance_to(pattern[-1]) > cfg.max_net_m:
            return None

        # Cluster the flip points greedily around running centroids.
        clusters: list[list] = []  # [centroid Point, [points]]
        for p in flips:
            for c in clusters:
                if p.distance_to(c[0]) <= cfg.cluster_radius_m:
                    c[1].append(p)
                    n = len(c[1])
                    c[0] = type(p)(
                        sum(q.x for q in c[1]) / n,
                        sum(q.y for q in c[1]) / n,
                    )
                    break
            else:
                clusters.append([p, [p]])
        clusters = [c for c in clusters if len(c[1]) >= cfg.min_points_per_end]
        if len(clusters) < 2:
            return None
        clusters.sort(key=lambda c: -len(c[1]))
        end_a, end_b = clusters[0], clusters[1]
        separation = end_a[0].distance_to(end_b[0])
        if not (cfg.min_separation_m <= separation <= cfg.max_separation_m):
            return None

        # Charge length between consecutive flip points.
        leg_lengths = [
            a.distance_to(b) for a, b in zip(flips, flips[1:])
        ]
        mean_leg = sum(leg_lengths) / len(leg_lengths) if leg_lengths else 0.0
        if len(flips) < cfg.min_reversals_weak and mean_leg < cfg.strong_leg_m:
            return None  # short legs need more proof than 4 flips

        # The impact face is the end the robot hit FIRST in this
        # window (it charged in from free travel); ties and later
        # windows resolve to the busier end, which is the pinned
        # face the robot keeps returning to.
        first = flips[0]
        impact = end_a if first.distance_to(end_a[0]) <= first.distance_to(
            end_b[0]) else end_b
        other = end_b if impact is end_a else end_a
        centre = impact[0]
        if any(p.distance_to(centre) > cfg.confinement_m for p in pattern):
            return None
        approach = angle_of(centre - other[0])
        self._latched_centre = centre
        return RammingEvent(
            t=sample.t, centre=centre,
            approach_heading_deg=approach,
            charges=len(impact[1]), mean_leg_m=mean_leg,
        )
