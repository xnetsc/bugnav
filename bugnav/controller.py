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
  * Second trigger, no struggle needed: if the firmware itself keeps
    circling one spot and passing its opening by (perimeter loiter,
    seen live 2026-11-11 at a U-shaped chair base), the second loiter
    at the same spot in one session starts an episode whose goal is
    the centre of the circled area - the pocket it never enters.  The
    staging-ring exploration then tries that goal from successive
    angles; the opening face is one of them.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from enum import Enum

from .clusters import ClusterManager, ObstacleCluster
from .detector import (
    DetectorConfig,
    PerimeterConfig,
    PerimeterDetector,
    PerimeterEvent,
    StruggleDetector,
)
from .geometry import Point, angle_of, polar, quantize_heading
from .model import Command, CommandKind, EpisodeStats, RobotState, Sample
from .strategist import Plan
from .trail import BreadcrumbTrail


class Phase(str, Enum):
    MONITOR = "monitor"
    RETREAT = "retreat"  # going back to the entry anchor
    TO_STAGING = "to_staging"  # moving to the next perimeter staging point
    TO_GOAL = "to_goal"  # trying to reach the goal from this staging point
    BACKTRACK = "backtrack"  # leaving along the reversed breadcrumb trail
    COOLDOWN = "cooldown"
    CONSULT = "consult"  # paused, waiting for the LLM strategy advisor


@dataclass
class ConsultRequest:
    """A pending question to the strategy advisor.

    The runner watches for this, asks the LLM asynchronously (the robot
    is paused meanwhile), and feeds the answer back through
    apply_consult_result(). If nobody answers within the consult
    timeout, tick() falls back to the fixed strategy on its own.
    """

    kind: str  # "repeated_failures" | "full_circle"
    situation: dict
    t: float


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
    perimeter: PerimeterConfig = field(default_factory=PerimeterConfig)
    # A single perimeter loiter is common in the firmware's normal
    # intensive cleaning, so it is only logged.  The robot coming back
    # and loitering at the same spot again in the same session is the
    # takeover signal (it circled the opening and passed it by, twice).
    perimeter_escalations_required: int = 2
    perimeter_repeat_radius_m: float = 0.6
    # LLM strategy advisor (see strategist.py). When enabled AND the
    # runner has an LLM configured, the fixed strategy consults the
    # advisor at its weak points instead of grinding on or giving up
    # blindly. Disabled = behaviour is exactly the fixed strategy.
    llm_strategy_enabled: bool = False
    max_consults_per_episode: int = 2
    consult_after_failures: int = 4
    consult_timeout_s: float = 180.0


class BugNavController:
    def __init__(
        self,
        config: ControllerConfig | None = None,
        stats_path: str | None = None,
    ):
        self.cfg = config or ControllerConfig()
        self.detector = StruggleDetector(self.cfg.detector)
        self.perimeter = PerimeterDetector(self.cfg.perimeter)
        # Perimeter-loiter bookkeeping for this session only (a chair
        # moves; nothing here may outlive the session as a ban).
        self._perimeter_visits: list[tuple[Point, int]] = []  # (centre, count)
        self._perimeter_handled: list[Point] = []  # centres already taken over
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
        # Strategy-advisor state (per episode)
        self.pending_consult: ConsultRequest | None = None
        self._consults_used = 0
        self._failures = 0
        self._planned_angles: list[float] = []
        self._step_deg = self.cfg.step_angle_deg
        self._clearance_m = self.cfg.staging_clearance_m
        from collections import deque

        self._recent: deque[Sample] = deque()
        self._free_heading: float | None = None

    # ------------------------------------------------------------- public
    def tick(self, sample: Sample) -> list[Command]:
        cmds: list[Command] = []
        if self.phase == Phase.MONITOR:
            cmds = self._tick_monitor(sample)
        elif self.phase == Phase.CONSULT:
            # Robot is paused; we only wait for the advisor's answer.
            # On timeout, fall back to exactly what the fixed strategy
            # would have done - the advisor can never wedge the robot.
            if (
                self.pending_consult is not None
                and sample.t - self.pending_consult.t >= self.cfg.consult_timeout_s
            ):
                cmds = self._consult_fallback(sample)
            else:
                cmds = []
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
            pe = self.perimeter.feed(sample)
            if pe is not None:
                return self._on_perimeter_event(sample, pe)
            return []
        # Approach heading = the direction it was travelling the last
        # time it was moving freely. (The trail and the struggle window
        # are both polluted by the jiggle itself, whose first move is
        # usually a back-off in the opposite direction.)
        approach = self._free_heading
        if approach is None:
            approach = event.approach_heading_deg
        return self._begin_episode(sample, event.centre, approach)  # type: ignore[arg-type]

    def _on_perimeter_event(
        self, sample: Sample, event: PerimeterEvent
    ) -> list[Command]:
        centre = event.centre  # type: ignore[assignment]
        if any(
            centre.distance_to(c) <= self.cfg.perimeter_repeat_radius_m
            for c in self._perimeter_handled
        ):
            return []  # already took this spot over once this session
        count = 1
        for i, (c, n) in enumerate(self._perimeter_visits):
            if centre.distance_to(c) <= self.cfg.perimeter_repeat_radius_m:
                count = n + 1
                self._perimeter_visits[i] = (c, count)
                centre = c
                break
        else:
            self._perimeter_visits.append((centre, 1))
        if count < self.cfg.perimeter_escalations_required:
            return [
                Command(
                    CommandKind.NOTIFY,
                    message=(
                        f"perimeter loiter #{count} at {centre} "
                        f"(path {event.path_m:.1f} m, sweep {event.sweep_deg:.0f} deg); "
                        "watching, no takeover yet"
                    ),
                )
            ]
        # Repeat loiter at the same spot: the firmware circled this
        # obstacle and passed its opening by, more than once.  Take
        # over and aim at the middle of the ring it kept walking
        # around - the pocket it never enters.
        self._perimeter_handled.append(centre)
        # Seed the cluster from the loiter outline directly: opposite
        # extremes of the ring can be ~1.3 m apart, beyond the hit-merge
        # distance, so find_or_create per point could split the ring
        # into two clusters.
        cluster = self.clusters.nearest(centre)
        if cluster is not None and centre.distance_to(
            cluster.centre
        ) <= self.clusters.merge_distance_m + cluster.radius_m:
            for p in event.ring_points:
                cluster.add_hit(p)
        else:
            cluster = ObstacleCluster(hits=list(event.ring_points))
            self.clusters.clusters.append(cluster)
        approach = angle_of(cluster.centre - sample.pos)
        return self._begin_episode(
            sample,
            sample.pos,
            approach,
            cluster=cluster,
            goal=cluster.centre,
            tried_sectors=set(),
            notify_text=(
                f"perimeter loiter #{count} at {cluster.centre}; firmware "
                "keeps circling without entering - taking over, goal is "
                "the centre of the circled area"
            ),
        )

    def _begin_episode(
        self,
        sample: Sample,
        hit: Point,
        approach_heading_deg: float,
        *,
        cluster: ObstacleCluster | None = None,
        goal: Point | None = None,
        tried_sectors: set[int] | None = None,
        notify_text: str | None = None,
    ) -> list[Command]:
        if cluster is None:
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

        self._goal = (
            goal
            if goal is not None
            else hit + polar(approach_heading_deg, self.cfg.goal_depth_m)  # type: ignore[operator]
        )
        self._theta_deg = angle_of(hit - cluster.centre) if cluster.hits else approach_heading_deg  # type: ignore[operator]
        self._swept_deg = 0.0
        self._tried_sectors = (
            tried_sectors
            if tried_sectors is not None
            else {quantize_heading(approach_heading_deg)}
        )
        self._error_pause_sent = False
        self._consults_used = 0
        self._failures = 0
        self._planned_angles = []
        self._step_deg = self.cfg.step_angle_deg
        self._clearance_m = self.cfg.staging_clearance_m
        self.pending_consult = None
        cmds.append(
            Command(
                CommandKind.NOTIFY,
                message=notify_text
                or (
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
        r = cl.radius_m + self._clearance_m
        return cl.centre + polar(theta_deg, r)

    def _go_staging(self, sample: Sample, theta_deg: float) -> list[Command]:
        self._theta_deg = theta_deg % 360.0
        staging = self._staging_point(self._theta_deg)
        self.phase = Phase.TO_STAGING
        self._pending_target = staging
        self._deadline = sample.t + self.cfg.staging_timeout_s
        return [Command(CommandKind.CLEAN_AREA, point=staging)]

    def _next_attempt(self, sample: Sample) -> list[Command]:
        """Advance the staging angle by one small step and go there."""
        self._swept_deg += self._step_deg
        return self._go_staging(sample, self._theta_deg + self._step_deg)

    def _attempt_from_planned(self, sample: Sample) -> list[Command]:
        """Jump to an advisor-planned angle (counts as swept arc too)."""
        theta = self._planned_angles.pop(0)
        delta = abs((theta - self._theta_deg + 180.0) % 360.0 - 180.0)
        self._swept_deg += delta
        return self._go_staging(sample, theta)

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
        self._failures += 1
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
        elif self._consult_available() and self._failures >= self.cfg.consult_after_failures:
            cmds.extend(self._enter_consult(sample, "repeated_failures"))
        elif self._planned_angles:
            cmds.extend(self._attempt_from_planned(sample))
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
        # Large obstacle, every direction failed. Before the fixed
        # strategy gives up, give the advisor one look: it may spot a
        # pattern (e.g. an opening just past a quantized step) that the
        # rigid sweep cannot express.
        if self._consult_available():
            return self._enter_consult(sample, "full_circle")
        return self._do_abandon(sample)

    def force_abandon(self, sample: Sample) -> list[Command]:
        """Supervisor order: abandon the current episode right now."""
        if self._cluster is None or self.phase in (Phase.MONITOR, Phase.COOLDOWN):
            return []
        self.pending_consult = None
        return self._do_abandon(sample)

    def _do_abandon(self, sample: Sample) -> list[Command]:
        """Blacklist for this session only, then leave the way we came."""
        cl = self._cluster
        assert cl is not None
        self._blacklist.append((cl.centre, sample.t + self.cfg.session_blacklist_ttl_s))
        self.stats[self._stats_key].abandons += 1
        self._save_stats()
        cmds = [
            Command(
                CommandKind.NOTIFY,
                message="abandoning this area for this session and "
                "backtracking out along the entry path",
            )
        ]
        self._start_backtrack(sample)
        return cmds

    # ------------------------------------------------- strategy advisor
    def _consult_available(self) -> bool:
        return (
            self.cfg.llm_strategy_enabled
            and self._consults_used < self.cfg.max_consults_per_episode
            and self.pending_consult is None
        )

    def _situation(self, kind: str, sample: Sample) -> dict:
        """Structured snapshot for the advisor. Numbers only - the
        advisor gets no say over coordinates, only over angles/params."""
        cl = self._cluster
        assert cl is not None
        st = self.stats.get(self._stats_key)
        return {
            "decision_point": kind,
            "robot_pos": [round(sample.pos.x, 2), round(sample.pos.y, 2)],
            "cluster_centre": [round(cl.centre.x, 2), round(cl.centre.y, 2)],
            "cluster_radius_m": round(cl.radius_m, 2),
            "hit_points": [[round(p.x, 2), round(p.y, 2)] for p in cl.hits],
            "goal": ([round(self._goal.x, 2), round(self._goal.y, 2)]
                     if self._goal else None),
            "swept_deg": round(self._swept_deg),
            "current_theta_deg": round(self._theta_deg),
            "tried_approach_sectors_deg": sorted(self._tried_sectors),
            "failures_this_episode": self._failures,
            "consults_used": self._consults_used,
            "step_angle_deg": self._step_deg,
            "staging_clearance_m": self._clearance_m,
            "history_this_spot": {
                "episodes": st.episodes if st else 0,
                "successes": st.successes if st else 0,
                "abandons": st.abandons if st else 0,
                "success_headings_deg": st.success_headings if st else [],
            },
        }

    def _enter_consult(self, sample: Sample, kind: str) -> list[Command]:
        self._consults_used += 1
        self.phase = Phase.CONSULT
        self.pending_consult = ConsultRequest(
            kind=kind, situation=self._situation(kind, sample), t=sample.t)
        return [
            Command(
                CommandKind.NOTIFY,
                message=f"consulting strategy advisor ({kind}); robot paused",
            )
        ]

    def _consult_fallback(self, sample: Sample) -> list[Command]:
        """No answer (or shadow mode): do what the fixed strategy does."""
        req = self.pending_consult
        self.pending_consult = None
        if req is not None and req.kind == "full_circle":
            return self._do_abandon(sample)
        if self._planned_angles:
            return self._attempt_from_planned(sample)
        return self._next_attempt(sample)

    def apply_consult_result(
        self, plan: Plan | None, sample: Sample
    ) -> list[Command]:
        """Feed the advisor's answer back in. plan=None means fallback.

        The plan is already range-validated by strategist.parse_plan;
        the controller additionally only ever turns it into staging
        angles around the known cluster - never raw coordinates.
        """
        if self.pending_consult is None:
            return []
        if plan is None:
            return self._consult_fallback(sample)
        self.pending_consult = None
        cmds = [
            Command(
                CommandKind.NOTIFY,
                message=f"strategy advisor: {plan.reason or 'plan received'}",
            )
        ]
        if plan.resolved:
            cmds.append(
                Command(CommandKind.NOTIFY,
                        message="advisor judges the obstacle resolved; resuming"))
            cmds.append(Command(CommandKind.RESUME_AUTO))
            self._end_episode(sample)
            return cmds
        if plan.abandon:
            cmds.extend(self._do_abandon(sample))
            return cmds
        if plan.step_angle_deg is not None:
            self._step_deg = plan.step_angle_deg
        if plan.staging_clearance_m is not None:
            self._clearance_m = plan.staging_clearance_m
        if plan.angles:
            self._planned_angles = list(plan.angles)
            cmds.extend(self._attempt_from_planned(sample))
        else:
            cmds.extend(self._next_attempt(sample))
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
        self.perimeter.reset()
        self.pending_consult = None
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
