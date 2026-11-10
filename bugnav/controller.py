"""BugNavController: the Bug-algorithm (perimeter exploration) brain.

Transport-agnostic and step-driven: feed it Samples, it returns Commands.
All timing comes from sample timestamps, which makes the whole state
machine deterministically testable against the simulator.

Strategy (settled with the user):
  * Obstacles are mostly *movable* (a chair).  Nothing is ever banned
    permanently; a blacklist entry lives for one cleaning session only.
  * Retrying in the same direction does not count as a new attempt.
    Attempts are counted in distinct approach directions.
  * On a struggle precursor (not the error - by the time a drive wheel
    is suspended it is far too late), first retreat along the breadcrumb
    trail to the entry anchor, then walk around the obstacle in small
    angular steps until an opening is found (U-shaped base: three sides
    fail, the fourth works; table leg: a full circle *is* the success,
    its perimeter is then fully cleaned).
  * If a full 360 degrees of approach angles fail on a large obstacle,
    the area is provably unreachable this session: leave by replaying
    the breadcrumb trail in reverse (every exit waypoint was actually
    traversed on the way in), blacklist the spot for this session only,
    and resume.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from enum import Enum

from .clusters import ClusterManager, ObstacleCluster
from .detector import DetectorConfig, StruggleDetector
from .geometry import Point, angle_of, polar, quantize_heading
from .model import Command, CommandKind, EpisodeStats, RobotState, Sample
from .trail import BreadcrumbTrail


class Phase(str, Enum):
    MONITOR = "monitor"
    RETREAT = "retreat"  # going back to the entry anchor
    TO_STAGING = "to_staging"  # moving to the next perimeter staging point
    TO_GOAL = "to_goal"  # trying to reach the goal from this staging point
    BACKTRACK = "backtrack"  # leaving along the reversed breadcrumb trail
    COOLDOWN = "cooldown"


@dataclass
class ControllerConfig:
    step_angle_deg: float = 30.0
    staging_clearance_m: float = 0.50  # staging ring distance from cluster edge
    goal_depth_m: float = 0.55  # goal = first hit + approach dir * this
    arrival_tol_m: float = 0.24
    staging_timeout_s: float = 45.0
    goal_timeout_s: float = 35.0
    retreat_timeout_s: float = 45.0
    backtrack_timeout_s: float = 30.0
    # Full circle on an obstacle this small = success (its perimeter is
    # then fully cleaned). Contact points sit one robot-clearance away
    # from the physical object, so a chair leg's cluster measures up to
    # ~0.4 m across even though the leg itself is a few cm wide.
    small_obstacle_radius_m: float = 0.45
    session_blacklist_ttl_s: float = 6 * 3600.0
    cooldown_s: float = 45.0
    backtrack_spacing_m: float = 0.45
    detector: DetectorConfig = field(default_factory=DetectorConfig)


class BugNavController:
    def __init__(
        self,
        config: ControllerConfig | None = None,
        stats_path: str | None = None,
    ):
        self.cfg = config or ControllerConfig()
        self.detector = StruggleDetector(self.cfg.detector)
        self.trail = BreadcrumbTrail()
        self.clusters = ClusterManager()
        self.phase = Phase.MONITOR
        self.stats: dict[str, EpisodeStats] = {}
        self._stats_path = stats_path
        self._load_stats()

        # Per-episode state
        self._cluster: ObstacleCluster | None = None
        self._stats_key: str = ""
        self._goal: Point | None = None
        self._anchor: Point | None = None
        self._theta_deg = 0.0  # current staging angle around the cluster
        self._swept_deg = 0.0
        self._tried_sectors: set[int] = set()
        self._pending_target: Point | None = None
        self._deadline = 0.0
        self._backtrack_points: list[Point] = []
        self._blacklist: list[tuple[Point, float]] = []  # (centre, expiry t)
        self._cooldown_until = 0.0
        self._error_pause_sent = False
        from collections import deque

        self._recent: deque[Sample] = deque()
        self._free_heading: float | None = None

    # ------------------------------------------------------------- public
    def tick(self, sample: Sample) -> list[Command]:
        cmds: list[Command] = []
        if self.phase == Phase.MONITOR:
            cmds = self._tick_monitor(sample)
        elif self.phase == Phase.COOLDOWN:
            if sample.t >= self._cooldown_until:
                self.phase = Phase.MONITOR
                self.detector.reset()
            else:
                self.trail.record(sample)
        else:
            cmds = self._tick_episode(sample)
        return cmds

    # ------------------------------------------------------------ monitor
    def _track_free_heading(self, sample: Sample) -> None:
        """Remember the travel direction while movement is clean.

        Clean = over the last few seconds the robot covered real ground
        and its path was nearly straight. Jiggling fails both tests, so
        the stored heading survives the struggle untouched.
        """
        self._recent.append(sample)
        while self._recent and sample.t - self._recent[0].t > 4.0:
            self._recent.popleft()
        if len(self._recent) < 3:
            return
        pts = [s.pos for s in self._recent]
        net = pts[0].distance_to(pts[-1])
        path = sum(a.distance_to(b) for a, b in zip(pts, pts[1:]))
        if net >= 0.40 and path <= net * 1.35:
            self._free_heading = angle_of(pts[-1] - pts[0])

    def _tick_monitor(self, sample: Sample) -> list[Command]:
        self.trail.record(sample)
        self._track_free_heading(sample)
        event = self.detector.feed(sample)
        if sample.error_code != 0 and sample.state == RobotState.ERROR:
            # A real error (e.g. wheel suspended). Too late to prevent,
            # but still an episode: record the hit at the current spot.
            from .detector import StruggleEvent

            event = event or StruggleEvent(
                t=sample.t,
                centre=sample.pos,
                approach_heading_deg=sample.heading_deg or 0.0,
            )
        if event is None:
            return []
        # Approach heading = the direction it was travelling the last
        # time it was moving freely. (The trail and the struggle window
        # are both polluted by the jiggle itself, whose first move is
        # usually a back-off in the opposite direction.)
        approach = self._free_heading
        if approach is None:
            approach = event.approach_heading_deg
        return self._begin_episode(sample, event.centre, approach)  # type: ignore[arg-type]

    def _begin_episode(
        self, sample: Sample, hit: Point, approach_heading_deg: float
    ) -> list[Command]:
        cluster = self.clusters.find_or_create(hit)  # type: ignore[arg-type]
        self._cluster = cluster
        self._anchor = self.trail.freeze()
        cmds = [Command(CommandKind.PAUSE)]
        key = cluster.grid_key()
        self._stats_key = key
        st = self.stats.setdefault(key, EpisodeStats(grid_key=key))
        st.episodes += 1
        self._save_stats()

        if self._is_blacklisted(cluster.centre, sample.t):
            cmds.append(
                Command(
                    CommandKind.NOTIFY,
                    message=f"session-blacklisted obstacle at {cluster.centre}, leaving",
                )
            )
            self._start_backtrack(sample)
            return cmds

        self._goal = hit + polar(approach_heading_deg, self.cfg.goal_depth_m)  # type: ignore[operator]
        self._theta_deg = angle_of(hit - cluster.centre) if cluster.hits else approach_heading_deg  # type: ignore[operator]
        self._swept_deg = 0.0
        self._tried_sectors = {quantize_heading(approach_heading_deg)}
        self._error_pause_sent = False
        cmds.append(
            Command(
                CommandKind.NOTIFY,
                message=(
                    f"struggle detected at {hit}, approach {approach_heading_deg:.0f} deg "
                    f"failed; starting perimeter exploration"
                ),
            )
        )
        if self._anchor is not None and sample.pos.distance_to(self._anchor) > self.cfg.arrival_tol_m:
            self.phase = Phase.RETREAT
            self._pending_target = self._anchor
            self._deadline = sample.t + self.cfg.retreat_timeout_s
            cmds.append(Command(CommandKind.CLEAN_AREA, point=self._anchor))
        else:
            cmds.extend(self._next_attempt(sample))
        return cmds

    # ------------------------------------------------------------ episode
    def _tick_episode(self, sample: Sample) -> list[Command]:
        cmds: list[Command] = []
        if sample.error_code != 0 and not self._error_pause_sent:
            self._error_pause_sent = True
            cmds.append(Command(CommandKind.PAUSE))
            cmds.append(
                Command(
                    CommandKind.NOTIFY,
                    message=f"robot reported error {sample.error_code} during episode; "
                    "it may already be suspended and need a manual rescue",
                )
            )
            if self.phase != Phase.BACKTRACK:
                self._start_backtrack(sample)
                return cmds

        arrived = (
            self._pending_target is not None
            and sample.pos.distance_to(self._pending_target) <= self.cfg.arrival_tol_m
        )
        timed_out = sample.t >= self._deadline

        if self.phase == Phase.RETREAT:
            if arrived or timed_out:
                cmds.extend(self._next_attempt(sample))
        elif self.phase == Phase.TO_STAGING:
            if arrived:
                self.phase = Phase.TO_GOAL
                self._pending_target = self._goal
                self._deadline = sample.t + self.cfg.goal_timeout_s
                self.detector.reset()
                cmds.append(Command(CommandKind.CLEAN_AREA, point=self._goal))
            elif timed_out:
                cmds.extend(
                    self._attempt_failed(sample, "staging unreachable", physical=False)
                )
        elif self.phase == Phase.TO_GOAL:
            if self._goal is not None and sample.pos.distance_to(self._goal) <= self.cfg.arrival_tol_m:
                cmds.extend(self._episode_success(sample, circumnavigated=False))
            else:
                ev = self.detector.feed(
                    sample, window_s=self.cfg.detector.attempt_window_s
                )
                if ev is not None:
                    cmds.extend(
                        self._attempt_failed(sample, "struggled again", physical=True)
                    )
                elif timed_out:
                    cmds.extend(
                        self._attempt_failed(sample, "goal timeout", physical=False)
                    )
        elif self.phase == Phase.BACKTRACK:
            if arrived or timed_out:
                cmds.extend(self._advance_backtrack(sample))
        return cmds

    def _staging_point(self, theta_deg: float) -> Point:
        cl = self._cluster
        assert cl is not None
        r = cl.radius_m + self.cfg.staging_clearance_m
        return cl.centre + polar(theta_deg, r)

    def _next_attempt(self, sample: Sample) -> list[Command]:
        """Advance the staging angle by one small step and go there."""
        self._theta_deg = (self._theta_deg + self.cfg.step_angle_deg) % 360.0
        self._swept_deg += self.cfg.step_angle_deg
        staging = self._staging_point(self._theta_deg)
        self.phase = Phase.TO_STAGING
        self._pending_target = staging
        self._deadline = sample.t + self.cfg.staging_timeout_s
        return [Command(CommandKind.CLEAN_AREA, point=staging)]

    def _attempt_failed(
        self, sample: Sample, why: str, *, physical: bool
    ) -> list[Command]:
        cl = self._cluster
        assert cl is not None and self._goal is not None
        # Only stops plausibly caused by *this* obstacle are hit points.
        # A timeout while the robot never got near (e.g. it was still on
        # its way to an unreachable staging point, near the anchor) must
        # not corrupt the cluster's centre/radius estimate. During a goal
        # attempt, a stop anywhere along the staging->goal corridor
        # belongs to this cluster - including the far side of a big
        # obstacle, whose contact points are what grow the cluster to
        # its true size.
        admit = physical
        if not admit and self.phase == Phase.TO_GOAL and self._goal is not None:
            admit = sample.pos.distance_to(self._goal) <= 1.8
        if not admit:
            admit = sample.pos.distance_to(cl.centre) <= cl.radius_m + 0.35
        if admit:
            cl.add_hit(sample.pos)
        # Heading actually attempted: staging -> goal.
        staging = self._staging_point(self._theta_deg)
        sector = quantize_heading(angle_of(self._goal - staging))
        self._tried_sectors.add(sector)
        cmds = [
            Command(CommandKind.PAUSE),
            Command(
                CommandKind.NOTIFY,
                message=f"approach at {self._theta_deg:.0f} deg failed ({why}); "
                f"swept {self._swept_deg:.0f}/360 deg",
            ),
        ]
        if self._swept_deg >= 360.0 - 1e-6:
            cmds.extend(self._conclude_full_circle(sample))
        else:
            cmds.extend(self._next_attempt(sample))
        return cmds

    def _conclude_full_circle(self, sample: Sample) -> list[Command]:
        cl = self._cluster
        assert cl is not None
        if cl.radius_m <= self.cfg.small_obstacle_radius_m:
            # Table-leg case: a full circle means the perimeter is fully
            # explored and cleaned. That is the desired outcome.
            return self._episode_success(sample, circumnavigated=True)
        # Large obstacle, every direction failed: provably unreachable
        # this session. Blacklist for this session only, then leave the
        # way we came in.
        self._blacklist.append((cl.centre, sample.t + self.cfg.session_blacklist_ttl_s))
        self.stats[self._stats_key].abandons += 1
        self._save_stats()
        cmds = [
            Command(
                CommandKind.NOTIFY,
                message="full 360 deg explored, no opening; abandoning this "
                "area for this session and backtracking out",
            )
        ]
        self._start_backtrack(sample)
        return cmds

    def _episode_success(
        self, sample: Sample, *, circumnavigated: bool
    ) -> list[Command]:
        cl = self._cluster
        assert cl is not None
        st = self.stats[self._stats_key]
        st.successes += 1
        if self._goal is not None:
            staging = self._staging_point(self._theta_deg)
            st.success_headings.append(quantize_heading(angle_of(self._goal - staging)))
        self._save_stats()
        how = "circumnavigated" if circumnavigated else "reached goal"
        cmds = [
            Command(
                CommandKind.NOTIFY,
                message=f"obstacle resolved ({how}) at {self._theta_deg:.0f} deg; resuming",
            ),
            Command(CommandKind.RESUME_AUTO),
        ]
        self._end_episode(sample)
        return cmds

    # ----------------------------------------------------------- backtrack
    def _start_backtrack(self, sample: Sample) -> None:
        self._backtrack_points = self.trail.reversed_waypoints(
            self.cfg.backtrack_spacing_m
        )
        self.phase = Phase.BACKTRACK
        if self._backtrack_points:
            self._pending_target = self._backtrack_points[0]
            self._deadline = sample.t + self.cfg.backtrack_timeout_s
        else:
            self._pending_target = None
            self._deadline = sample.t

    def _advance_backtrack(self, sample: Sample) -> list[Command]:
        self._backtrack_points.pop(0)
        if self._backtrack_points:
            self._pending_target = self._backtrack_points[0]
            self._deadline = sample.t + self.cfg.backtrack_timeout_s
            return [Command(CommandKind.CLEAN_AREA, point=self._pending_target)]
        cmds = [
            Command(CommandKind.NOTIFY, message="backtrack complete; resuming auto clean"),
            Command(CommandKind.RESUME_AUTO),
        ]
        self._end_episode(sample)
        return cmds

    def _end_episode(self, sample: Sample) -> None:
        self.trail.unfreeze()
        self.detector.reset()
        self._cluster = None
        self._goal = None
        self._pending_target = None
        self.phase = Phase.COOLDOWN
        self._cooldown_until = sample.t + self.cfg.cooldown_s

    # -------------------------------------------------------------- misc
    def _is_blacklisted(self, centre: Point, t: float) -> bool:
        self._blacklist = [(c, e) for c, e in self._blacklist if e > t]
        return any(
            centre.distance_to(c) <= 1.0 + self.cfg.staging_clearance_m
            for c, _ in self._blacklist
        )

    def _load_stats(self) -> None:
        if not self._stats_path or not os.path.exists(self._stats_path):
            return
        try:
            with open(self._stats_path, encoding="utf-8") as f:
                raw = json.load(f)
            for key, v in raw.items():
                self.stats[key] = EpisodeStats(**v)
        except (OSError, ValueError, TypeError):
            self.stats = {}

    def _save_stats(self) -> None:
        if not self._stats_path:
            return
        data = {k: vars(v) for k, v in self.stats.items()}
        tmp = self._stats_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self._stats_path)
