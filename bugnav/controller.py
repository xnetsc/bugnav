"""BugNavController: watch the native clean, intervene without
ever resetting its session.

Transport-agnostic and step-driven: feed it Samples, it returns
Commands.  All timing comes from sample timestamps, which makes
the whole state machine deterministically testable.

Control model (measured on the target firmware): every externally
directed move - a remote-control jog step or a mapPoint drive -
discards the running native session, and the fresh session that
replaces it re-plans the whole home and re-cleans finished ground.
Rotation jogs are refused outright.  The only commands that keep
the session (and its breakpoint) are PAUSE / RESUME in place, and
the robot's own return-to-dock followed by a breakpoint resume.
Interventions are therefore built from exactly those:

  * Struggle / ramming / perimeter-loiter precursors (the robot is
    still mobile, its session is still productive): pause in
    place, let it settle, resume the SAME session.  A dedicated
    watch then re-checks the pattern: if the same signature forms
    again, pause once more - up to a per-spot attempt budget -
    and if the budget runs out, notify the user and leave the
    spot to native cleaning for the rest of the session.  No
    command in this path can reset the session.
  * A wedge (a fault persisting while the robot stays put, or a
    position frozen while cleaning) is the one case where the
    session is already producing nothing: back out in jog steps -
    forward is what wedged it.  A successful back-out restarts
    cleaning from right there (the jog cost the session; a fresh
    AUTO start is the only way back to work).  If the back-out
    itself fails - rear blocked, or the fault survives it - the
    robot stops, sounds its own alarm and waits for a human.

Obstacles are mostly movable (a chair): nothing is banned
permanently.  Every suppression mark below lives for minutes (a
handled spot) or for the rest of the session (a spot that used up
its attempts), never longer.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum

from .clusters import ClusterManager, ObstacleCluster
from .mapedge import wall_by_map
from .detector import (
    DetectorConfig,
    PerimeterConfig,
    PerimeterDetector,
    PerimeterEvent,
    PinnedConfig,
    PinnedDetector,
    RammingConfig,
    RammingDetector,
    RammingEvent,
    StruggleDetector,
)
from .geometry import Point
from .model import Command, CommandKind, SpotStats, RobotState, Sample


class Phase(str, Enum):
    MONITOR = "monitor"
    HOLD = "hold"  # paused in place / watching a resumed session
    UNSTICK = "unstick"  # backing a wedged robot out, jog step by jog step


@dataclass
class ControllerConfig:
    # Pause-hold for the mobile precursors (struggle, ramming,
    # perimeter loiter): pause in place, settle, resume the same
    # session, then watch for the pattern forming again.
    hold_settle_s: float = 6.0  # paused this long before resuming
    hold_watch_s: float = 45.0  # post-resume watch for a re-formed pattern
    hold_max_attempts: int = 3  # pause-resume cycles per spot per session
    # After a successful hold the spot is handled; the robot will
    # clean around the same furniture and the soft signatures may
    # recur benignly.  Soft triggers stay quiet there this long.
    resolved_suppress_s: float = 900.0
    # A spot that used up its hold attempts is left to native
    # cleaning for this long (session-length in practice).
    hold_blocked_ttl_s: float = 6 * 3600.0
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    perimeter: PerimeterConfig = field(default_factory=PerimeterConfig)
    ramming: RammingConfig = field(default_factory=RammingConfig)
    pinned: PinnedConfig = field(default_factory=PinnedConfig)
    # Wedged rescue (a fault persisting in place, or a frozen
    # position): back out in jog steps - the direct cure for a
    # nose-in wedge.
    error_rescue_delay_s: float = 10.0  # a fault must persist this long
    jog_observe_s: float = 8.0  # watch one jog step this long
    jog_free_m: float = 0.25  # displacement that counts as backed out
    jog_max_steps: int = 3  # backward steps before the rear reads blocked
    jog_error_clear_s: float = 8.0  # after a back-out, wait this long
    # for a persisting fault to clear before deciding the outcome
    # When rescue gives up and the robot waits for a human, the
    # ROBOT ITSELF must also raise the alarm - its built-in sound,
    # this many times - not only the program's notification.
    trapped_sound_count: int = 3
    # Wall confirmation comes from the MAP, not from the motion
    # shape: a trigger centre at the edge of the known cleanable
    # area - at least this much of the ring around it (probed a
    # bit beyond the cluster) was never cleaned - IS a wall or
    # corner and gets no intervention; the firmware's own edge
    # handling is the right tool there.  Anything not confirmed
    # at the area edge is furniture.
    wall_min_outside_arc_deg: float = 120.0
    wall_probe_extra_m: float = 0.55


class BugNavController:
    def __init__(
        self,
        config: ControllerConfig | None = None,
        stats_path: str | None = None,
    ):
        self.cfg = config or ControllerConfig()
        self.detector = StruggleDetector(self.cfg.detector)
        self.perimeter = PerimeterDetector(self.cfg.perimeter)
        self.ramming = RammingDetector(self.cfg.ramming)
        self.pinned = PinnedDetector(self.cfg.pinned)
        # Dedicated, unlatched detector pair for the post-resume
        # watch inside a hold: the main detectors latch
        # positionally when they fire, so they cannot see the same
        # pattern form again at the same spot - the watch pair can.
        self._watch_struggle = StruggleDetector(self.cfg.detector)
        self._watch_ramming = RammingDetector(self.cfg.ramming)
        # Ever-swept footprint (0.5 m cell keys), pushed by the
        # runner: every cell ever observed cleaned, updated live.
        # Used ONLY to confirm walls by the map edge (see
        # mapedge.py); empty = no map evidence = nothing is a wall.
        self._footprint: set[tuple[int, int]] = set()
        self.clusters = ClusterManager()
        self.phase = Phase.MONITOR
        self.stats: dict[str, SpotStats] = {}
        self._stats_path = stats_path
        self._load_stats()

        # The spot the current intervention (hold or unstick) is
        # about, and its per-spot statistics key.
        self._cluster: ObstacleCluster | None = None
        self._stats_key: str = ""
        # Boxed-in latch: set when a rescue gives up; MONITOR stays
        # hands-off until a human visibly intervenes (the robot is
        # moved by hand or reaches the dock).
        self._trapped_pos: Point | None = None
        # Quiet marks: (centre, radius_m, expiry t, kind) - spots
        # where soft triggers stay quiet.  kind "resolved" = a hold
        # succeeded there; kind "blocked" = the spot used up its
        # hold attempts and is left alone for the session.
        self._quiet_marks: list[tuple[Point, float, float, str]] = []
        # Hold state.
        self._hold_centre: Point | None = None
        self._hold_pos: Point | None = None  # robot position at entry
        self._hold_stage: str = ""  # "settle" | "watch"
        self._hold_deadline: float = 0.0
        self._hold_count: int = 0  # attempts in the current series
        # Attempt counts that survive a neutrally closed hold:
        # grid key -> (attempts used, expiry t).
        self._hold_attempts: dict[str, tuple[int, float]] = {}
        # Wedged-rescue state: the spot being backed out of, how
        # many jog steps are out, when the current watch ends, and
        # when the robot first got clear (fault-clear clock).
        self._unstick_pos: Point | None = None
        self._unstick_steps: int = 0
        self._unstick_deadline: float = 0.0
        self._unstick_freed_at: float | None = None
        # Fault-persistence tracker: a fault counts as a wedge only
        # while it persists AND the robot stays put.
        self._err_since: float | None = None
        self._err_pos: Point | None = None

    # ------------------------------------------------------------- public
    def set_footprint(self, cells: set[tuple[int, int]]) -> None:
        """The runner pushes the ever-swept footprint (history,
        live-updated) for map-edge wall checks."""
        self._footprint = set(cells)

    def tick(self, sample: Sample) -> list[Command]:
        if self.phase == Phase.MONITOR:
            return self._tick_monitor(sample)
        if self.phase == Phase.HOLD:
            return self._tick_hold(sample)
        return self._tick_unstick(sample)

    def force_abandon(self, sample: Sample) -> list[Command]:
        """Supervisor order: give up on the current intervention.

        The spot is left to native cleaning for the rest of the
        session (a blocked quiet mark); the session itself is
        untouched.
        """
        if self.phase == Phase.MONITOR or self._cluster is None:
            return []
        self._mark_quiet(
            self._cluster.centre, self._cluster.radius_m, sample.t,
            self.cfg.hold_blocked_ttl_s, "blocked")
        self._hold_attempts.pop(self._stats_key, None)
        cmds = [Command(
            CommandKind.NOTIFY,
            message="supervisor: abandoning the current "
                    "intervention; this spot is left to native "
                    "cleaning for this session")]
        self._close_intervention()
        return cmds

    # ------------------------------------------------------------ monitor
    def _tick_monitor(self, sample: Sample) -> list[Command]:
        if self._trapped_pos is not None:
            # Boxed in and alarmed: start nothing, resume nothing,
            # until a human visibly intervenes (robot moved by hand
            # or back at the dock).
            if (sample.state == RobotState.DOCKED
                    or sample.pos.distance_to(self._trapped_pos) > 1.0):
                self._trapped_pos = None
            else:
                return []
        event = self.detector.feed(sample)
        pin_event = self.pinned.feed(sample)
        if self._track_error(sample):
            return self._begin_unstick(
                sample,
                reason=(
                    f"error {sample.error_code} persisting with "
                    "the robot stationary"),
            )
        if event is not None:
            if (sample.error_code == 0 and self._within_quiet(
                    event.centre, sample.t, ("resolved", "blocked"))):
                # This spot was just handled - the robot is
                # cleaning around the same furniture, not stuck
                # anew.  Intervening again would only pause it
                # pointlessly.
                return []
            cluster = self.clusters.find_or_create(event.centre)
            return self._begin_hold(
                sample, event.centre, cluster,
                notify_text=(
                    f"struggle detected at {event.centre}; "
                    "pausing in place and resuming the same "
                    "session"))
        rev = self.ramming.feed(sample)
        if rev is not None:
            return self._on_ramming_event(sample, rev)
        pe = self.perimeter.feed(sample)
        if pe is not None:
            return self._on_perimeter_event(sample, pe)
        if pin_event is not None:
            return self._begin_unstick(
                sample,
                reason=(
                    "position frozen while cleaning for "
                    f"{self.cfg.pinned.window_s:.0f} s"),
            )
        return []

    def _track_error(self, sample: Sample) -> bool:
        """A reported fault that PERSISTS while the robot stays
        put is a wedge, whatever state label carries it: the
        firmware reports one and the same drive-wheel fault as
        ERROR or as paused-with-error depending on the moment,
        and both forms sit frozen.  A fault that clears, or one
        the robot drives away from, is not a wedge.  Returns True
        once the fault has persisted long enough to rescue."""
        if sample.error_code != 0 and sample.state in (
            RobotState.ERROR,
            RobotState.PAUSED,
        ):
            if (
                self._err_pos is None
                or sample.pos.distance_to(self._err_pos) > 0.3
            ):
                self._err_since = sample.t
                self._err_pos = sample.pos
            return (
                self._err_since is not None
                and sample.t - self._err_since
                >= self.cfg.error_rescue_delay_s
            )
        self._err_since = None
        self._err_pos = None
        return False

    def _on_ramming_event(
        self, sample: Sample, event: RammingEvent
    ) -> list[Command]:
        centre = event.centre
        cluster = self.clusters.find_or_create(centre)
        # Same map gate as every other trigger: a face the map
        # confirms as the edge of the cleanable area is a wall,
        # not a climb target - no intervention, the firmware's own
        # edge handling continues.
        is_wall, arc = wall_by_map(
            centre, self._footprint,
            cluster.radius_m + self.cfg.wall_probe_extra_m,
            self.cfg.wall_min_outside_arc_deg)
        if is_wall:
            return [Command(
                CommandKind.NOTIFY,
                message=(
                    f"repeated charging at {centre} confirmed as a "
                    f"wall by the map edge (never-cleaned arc "
                    f"{arc:.0f} deg around it); no intervention"))]
        if self._within_quiet(centre, sample.t, ("blocked",)):
            return []  # this spot used up its attempts; leave it
        return self._begin_hold(
            sample, centre, cluster,
            notify_text=(
                f"repeated hard charging at {centre}: "
                f"{event.charges} runs into the same face, mean "
                f"run-up {event.mean_leg_m:.2f} m - pausing the "
                "climb attempt in place and resuming the same "
                f"session (map edge arc {arc:.0f} deg: not a "
                "wall)"),
        )

    def _on_perimeter_event(
        self, sample: Sample, event: PerimeterEvent
    ) -> list[Command]:
        centre = event.centre
        # Seed the cluster from the loiter outline directly:
        # opposite extremes of the ring can be ~1.3 m apart, beyond
        # the hit-merge distance, so find_or_create per point could
        # split the ring into two clusters.
        cluster = self.clusters.nearest(centre)
        if cluster is not None and centre.distance_to(
            cluster.centre
        ) <= self.clusters.merge_distance_m + cluster.radius_m:
            for p in event.ring_points:
                cluster.add_hit(p)
        else:
            cluster = ObstacleCluster(hits=list(event.ring_points))
            self.clusters.clusters.append(cluster)
        if self._within_quiet(
                cluster.centre, sample.t, ("resolved", "blocked")):
            return []  # this spot was just handled, or gave up on
        # Map first: a loiter is not automatically a stuck robot.
        # If the map says this spot is at the edge of the
        # cleanable area, it is a wall/corner - the robot's own
        # edge handling continues, and the struggle detector still
        # watches independently.
        is_wall, arc = wall_by_map(
            centre, self._footprint,
            cluster.radius_m + self.cfg.wall_probe_extra_m,
            self.cfg.wall_min_outside_arc_deg)
        if is_wall:
            return [Command(
                CommandKind.NOTIFY,
                message=(
                    f"perimeter loiter at {centre} confirmed as a "
                    f"wall/corner by the map edge (never-cleaned "
                    f"arc {arc:.0f} deg around it); no intervention "
                    "- native edge handling continues"))]
        return self._begin_hold(
            sample, cluster.centre, cluster,
            notify_text=(
                f"perimeter loiter at {cluster.centre} "
                f"(path {event.path_m:.1f} m, sweep "
                f"{event.sweep_deg:.0f} deg); map edge check: "
                f"never-cleaned arc {arc:.0f} deg - interior "
                "obstacle, not a wall; pausing in place and "
                "resuming the same session"),
        )

    # --------------------------------------------------------------- hold
    def _begin_hold(
        self,
        sample: Sample,
        centre: Point,
        cluster: ObstacleCluster,
        *,
        notify_text: str,
    ) -> list[Command]:
        self._cluster = cluster
        key = cluster.grid_key()
        self._stats_key = key
        stored = self._hold_attempts.get(key)
        if stored is not None and stored[1] <= sample.t:
            stored = None  # expired
        if stored is not None and stored[0] >= self.cfg.hold_max_attempts:
            # A previous series already used the budget and the
            # hold closed neutrally; stay out for the session.
            self._mark_quiet(centre, cluster.radius_m, sample.t,
                             self.cfg.hold_blocked_ttl_s, "blocked")
            self._cluster = None
            return []
        if stored is None:
            st = self.stats.setdefault(key, SpotStats(grid_key=key))
            st.interventions += 1
            self._save_stats()
        self._hold_count = (stored[0] if stored else 0) + 1
        self._hold_centre = centre
        self._hold_pos = sample.pos
        self._hold_stage = "settle"
        self._hold_deadline = sample.t + self.cfg.hold_settle_s
        self.phase = Phase.HOLD
        cmds: list[Command] = []
        if sample.state == RobotState.CLEANING:
            cmds.append(Command(CommandKind.PAUSE))
        cmds.append(Command(
            CommandKind.NOTIFY,
            message=(f"{notify_text} (attempt {self._hold_count} "
                     f"of {self.cfg.hold_max_attempts})")))
        return cmds

    def _tick_hold(self, sample: Sample) -> list[Command]:
        # Hard signals outrank the hold at every moment: a fault
        # persisting in place, or a frozen position, is a wedge -
        # hand it to the rescue path instead.
        pin_event = self.pinned.feed(sample)
        if self._track_error(sample):
            self._close_intervention()
            return self._begin_unstick(
                sample,
                reason=(
                    f"error {sample.error_code} persisting with "
                    "the robot stationary"),
            )
        if pin_event is not None:
            self._close_intervention()
            return self._begin_unstick(
                sample,
                reason=(
                    "position frozen while cleaning for "
                    f"{self.cfg.pinned.window_s:.0f} s"),
            )
        if sample.state in (RobotState.DOCKED, RobotState.RETURNING):
            # It is heading home (or home) on its own: the hold is
            # moot, close it and go back to watching.
            self._stash_attempts(sample)
            self._close_intervention()
            return []
        if self._hold_stage == "settle":
            if (sample.state == RobotState.CLEANING
                    and self._hold_pos is not None
                    and sample.pos.distance_to(self._hold_pos) > 0.6):
                # The stream shows it cleaning again well away
                # from where we paused it - our pause never
                # landed, or it was resumed externally.  Handling
                # follows the data stream, never the commands we
                # believe we sent: the spot is moving again, close
                # the hold as handled.
                return self._hold_resolved(
                    sample,
                    "the robot is cleaning again away from the "
                    "paused spot (pause did not hold, or it was "
                    "resumed)")
            if sample.t < self._hold_deadline:
                return []
            if sample.state == RobotState.PAUSED:
                self._enter_watch(sample)
                return [Command(CommandKind.RESUME_AUTO)]
            if sample.state == RobotState.CLEANING:
                # Our pause never landed but it is still here,
                # cleaning: watch it from now.
                self._enter_watch(sample)
                return []
            # IDLE or anything unexpected: nothing to resume.
            self._stash_attempts(sample)
            self._close_intervention()
            return []
        # stage == "watch": the session is resumed; judge by what
        # it does now, with the dedicated unlatched detectors.
        assert self._hold_centre is not None and self._cluster is not None
        if sample.state == RobotState.CLEANING:
            if sample.pos.distance_to(self._hold_centre) > (
                    self._cluster.radius_m + 0.6):
                return self._hold_resolved(
                    sample, "it cleaned its way out of the spot")
            ev = self._watch_struggle.feed(sample)
            if ev is None:
                ev = self._watch_ramming.feed(sample)
            if ev is not None:
                return self._hold_attempt_failed(sample)
        if sample.t >= self._hold_deadline:
            if sample.state == RobotState.CLEANING:
                return self._hold_resolved(
                    sample,
                    "the pattern did not form again after the "
                    "resume")
            self._stash_attempts(sample)
            self._close_intervention()
            return []
        return []

    def _enter_watch(self, sample: Sample) -> None:
        self._hold_stage = "watch"
        self._hold_deadline = sample.t + self.cfg.hold_watch_s
        self._watch_struggle.reset()
        self._watch_ramming.reset()

    def _hold_attempt_failed(self, sample: Sample) -> list[Command]:
        if self._hold_count >= self.cfg.hold_max_attempts:
            return self._hold_exhausted(sample)
        self._hold_count += 1
        self._hold_stage = "settle"
        self._hold_deadline = sample.t + self.cfg.hold_settle_s
        cmds: list[Command] = []
        if sample.state == RobotState.CLEANING:
            cmds.append(Command(CommandKind.PAUSE))
        cmds.append(Command(
            CommandKind.NOTIFY,
            message=(
                f"the pattern formed again at {self._hold_centre} "
                f"after the resume; pausing in place again "
                f"(attempt {self._hold_count} of "
                f"{self.cfg.hold_max_attempts})")))
        return cmds

    def _hold_resolved(self, sample: Sample, how: str) -> list[Command]:
        cl = self._cluster
        assert cl is not None
        self._mark_quiet(cl.centre, cl.radius_m, sample.t,
                         self.cfg.resolved_suppress_s, "resolved")
        st = self.stats.get(self._stats_key)
        if st is not None:
            st.successes += 1
            self._save_stats()
        self._hold_attempts.pop(self._stats_key, None)
        cmds = [Command(
            CommandKind.NOTIFY,
            message=f"pause-hold resolved at {self._hold_centre}: "
                    f"{how}; the same session continues")]
        self._close_intervention()
        return cmds

    def _hold_exhausted(self, sample: Sample) -> list[Command]:
        cl = self._cluster
        assert cl is not None
        self._mark_quiet(cl.centre, cl.radius_m, sample.t,
                         self.cfg.hold_blocked_ttl_s, "blocked")
        st = self.stats.get(self._stats_key)
        if st is not None:
            st.abandons += 1
            self._save_stats()
        self._hold_attempts.pop(self._stats_key, None)
        cmds = [Command(
            CommandKind.NOTIFY,
            message=(
                f"spot at {self._hold_centre} resisted "
                f"{self.cfg.hold_max_attempts} pause-resume "
                "cycles; leaving it to native cleaning for this "
                "session - a hand may be needed if it is truly "
                "stuck there"))]
        self._close_intervention()
        return cmds

    def _stash_attempts(self, sample: Sample) -> None:
        """Remember a neutrally closed hold's attempt count, so a
        later series at the same spot continues the budget instead
        of restarting it."""
        if self._stats_key and self._hold_count > 0:
            self._hold_attempts[self._stats_key] = (
                self._hold_count,
                sample.t + self.cfg.hold_blocked_ttl_s)

    # ------------------------------------------------------------ unstick
    def _begin_unstick(
        self, sample: Sample, *, reason: str
    ) -> list[Command]:
        """Back a wedged robot out, one jog step at a time.

        The wedge is nose-in, and the way out is the way it came -
        backwards.  This is the one intervention that spends the
        native session: a wedged session cleans nothing, and the
        jog is the only way out of the wedge.
        """
        self._err_since = None
        self._err_pos = None
        cluster = self.clusters.find_or_create(sample.pos)
        self._cluster = cluster
        key = cluster.grid_key()
        self._stats_key = key
        st = self.stats.setdefault(key, SpotStats(grid_key=key))
        st.interventions += 1
        self._save_stats()
        self._unstick_pos = sample.pos
        self._unstick_steps = 1
        self._unstick_deadline = sample.t + self.cfg.jog_observe_s
        self._unstick_freed_at = None
        cmds: list[Command] = []
        if sample.state == RobotState.CLEANING:
            # Take over before jogging, as the remote-control flow
            # itself does; a fault-paused robot is already held.
            cmds.append(Command(CommandKind.PAUSE))
        cmds.append(
            Command(
                CommandKind.NOTIFY,
                message=(
                    f"robot wedged at {sample.pos} ({reason}); "
                    "backing out first - forward is what wedged it"
                ),
            )
        )
        cmds.append(Command(CommandKind.JOG, act="backward"))
        self.phase = Phase.UNSTICK
        return cmds

    def _tick_unstick(self, sample: Sample) -> list[Command]:
        assert self._unstick_pos is not None
        if sample.state in (RobotState.DOCKED, RobotState.RETURNING):
            # It got home (or is on its way) without us: the wedge
            # is moot, close the rescue and go back to watching.
            cmds = [
                Command(
                    CommandKind.NOTIFY,
                    message="wedged robot is heading back to the "
                    "dock on its own; unstick closed",
                )
            ]
            self._close_intervention()
            return cmds
        disp = sample.pos.distance_to(self._unstick_pos)
        if self._unstick_freed_at is None:
            if disp >= self.cfg.jog_free_m:
                self._unstick_freed_at = sample.t
                self._unstick_deadline = (
                    sample.t + self.cfg.jog_error_clear_s)
                return [
                    Command(CommandKind.JOG, act="stop"),
                    Command(
                        CommandKind.NOTIFY,
                        message=(
                            f"backed out {disp:.2f} m from the "
                            "wedge; watching whether the fault "
                            "clears"
                        ),
                    ),
                ]
            if sample.t >= self._unstick_deadline:
                if sample.state == RobotState.CLEANING:
                    # Driving again but has not (yet) travelled
                    # the freeing distance: the wedge released it
                    # - monitoring (and the frozen-position
                    # detector) will catch it if it truly stays
                    # pinned here.
                    return self._unstick_resolved(
                        sample, "it is driving again")
                if self._unstick_steps < self.cfg.jog_max_steps:
                    self._unstick_steps += 1
                    self._unstick_deadline = (
                        sample.t + self.cfg.jog_observe_s
                    )
                    return [Command(CommandKind.JOG, act="backward")]
                # The rear is blocked too: no way out under its
                # own power.
                return self._unstick_give_up(
                    sample,
                    "backing out went nowhere in "
                    f"{self.cfg.jog_max_steps} steps - something "
                    "is behind it too")
            return []
        # Backed out.  The jog ended the native session (a jog
        # always does on this firmware), so there is nothing to
        # resume: once the fault reads clear, start a fresh AUTO
        # clean right here; if the fault survives, this is more
        # than a wedge.
        if sample.state == RobotState.CLEANING and sample.error_code == 0:
            return self._unstick_resolved(
                sample, "it resumed cleaning on its own")
        if sample.t >= self._unstick_deadline:
            if sample.error_code != 0:
                return self._unstick_give_up(
                    sample,
                    f"error {sample.error_code} persists after "
                    "backing out")
            cmds = [
                Command(
                    CommandKind.NOTIFY,
                    message=(
                        "backed out and the fault cleared; the "
                        "jog ended the wedged session, so a fresh "
                        "auto clean starts from here"
                    ),
                ),
                Command(CommandKind.START_AUTO),
            ]
            self._mark_quiet_here(sample)
            self._close_intervention()
            return cmds
        return []

    def _mark_quiet_here(self, sample: Sample) -> None:
        cl = self._cluster
        if cl is None:
            return
        self._mark_quiet(cl.centre, cl.radius_m, sample.t,
                         self.cfg.resolved_suppress_s, "resolved")
        st = self.stats.get(self._stats_key)
        if st is not None:
            st.successes += 1
            self._save_stats()

    def _unstick_resolved(
        self, sample: Sample, how: str
    ) -> list[Command]:
        cmds = [
            Command(
                CommandKind.NOTIFY,
                message=f"unstick complete at {self._unstick_pos}: "
                        f"{how}",
            )
        ]
        self._mark_quiet_here(sample)
        self._close_intervention()
        return cmds

    def _unstick_give_up(
        self, sample: Sample, why: str
    ) -> list[Command]:
        """No way out under its own power: stop, sound the alarm,
        and wait for a human.  The session is left exactly as it
        lies - a rescue by hand loses nothing more."""
        self._trapped_pos = sample.pos
        cmds: list[Command] = []
        if sample.state == RobotState.CLEANING:
            cmds.append(Command(CommandKind.PAUSE))
        cmds.append(Command(
            CommandKind.PLAY_SOUND,
            count=self.cfg.trapped_sound_count))
        cmds.append(Command(
            CommandKind.NOTIFY,
            message=(f"WEDGED: backing out failed ({why}); robot "
                     "stopped and needs a human rescue")))
        self._close_intervention()
        return cmds

    # -------------------------------------------------------------- misc
    def _close_intervention(self) -> None:
        """End the current hold/unstick and return to MONITOR.

        The detectors are NOT reset here: closing an intervention
        must not wipe the observation windows - a pattern already
        forming elsewhere has to stay visible.  Their latches are
        positional, so the spot just handled stays suppressed
        until the robot leaves it.
        """
        self._cluster = None
        self._hold_centre = None
        self._hold_pos = None
        self._hold_stage = ""
        self._hold_count = 0
        self._unstick_pos = None
        self._unstick_steps = 0
        self._unstick_freed_at = None
        self.phase = Phase.MONITOR

    def _mark_quiet(
        self, centre: Point, radius_m: float, t: float,
        ttl_s: float, kind: str,
    ) -> None:
        self._quiet_marks.append((centre, radius_m, t + ttl_s, kind))

    def _within_quiet(
        self, centre: Point, t: float, kinds: tuple[str, ...]
    ) -> bool:
        self._quiet_marks = [
            m for m in self._quiet_marks if m[2] > t]
        return any(
            kind in kinds and centre.distance_to(c) <= r + 0.35
            for c, r, _, kind in self._quiet_marks
        )

    def _load_stats(self) -> None:
        if not self._stats_path or not os.path.exists(self._stats_path):
            return
        try:
            with open(self._stats_path, encoding="utf-8") as f:
                raw = json.load(f)
            for key, v in raw.items():
                self.stats[key] = SpotStats(**v)
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
