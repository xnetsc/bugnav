"""Session supervisor: LLM common-sense oversight of the whole clean.

Two judgment layers live outside the deterministic controller because
they cannot be hardcoded honestly:
  * the episode strategist (strategist.py) advises inside one obstacle
    encounter;
  * this supervisor watches the whole bout: "started 07:00, it is now
    noon, it re-cleaned the same rooms five times and covered half the
    home - just stop." A fixed hour-threshold would misfire on a big
    home; the judgment is contextual, so deterministic code computes
    an honest digest and decides WHEN to ask, the LLM decides what the
    numbers mean.

Charge judgment follows the user-set objective: minimize TOTAL time
(cleaning + charging + cycle overheads). The deterministic solver in
chargeplan.py proposes the resume level; the supervisor reviews it
with context (time of day, whether the remaining scrap is worth
another cycle at all) and may override within bounds or stop instead.

Bounded verdict vocabulary (the supervisor can only ever):
  continue        - nothing wrong, keep going
  notify          - tell the user something (message included)
  abandon_current - give up on the current obstacle episode only
  stop_session    - stop this cleaning session entirely
  ask_user        - pause and let the USER choose (button/app);
                    no feedback within ask_timeout_s = return & end
  + resume_charge_pct while docked: resume at this battery level
    instead of the firmware's fixed threshold.
It cannot invent destinations or change controller parameters.
In observe mode verdicts are logged as shadow decisions only.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field

from .chargeplan import MARGIN, ChargeModel, expected_total_minutes
from .model import RobotState, Sample

CELL_M = 0.5
CELL_AREA_M2 = CELL_M * CELL_M


@dataclass
class SupervisorConfig:
    enabled: bool = False
    review_interval_s: float = 1200.0  # ask every ~20 min of cleaning
    session_soft_cap_s: float = 3 * 3600.0  # beyond this, ask sooner
    # ask_user verdicts: how long to wait for the user's choice
    # (button press / app action) before the default fires - return
    # to dock and end (user 2026-11-11: 5 minutes, no feedback).
    ask_timeout_s: float = 300.0
    # Home timezone as a fixed UTC offset.  Every clock string fed to
    # the supervisor LLM (now / bout_started / eta_finish / ready_at)
    # is rendered with THIS offset, never the process-local clock:
    # the container may run on UTC, and on 2026-11-11 that made the
    # judge read 11:01 local as "03:00, deep night" and stop a
    # morning session.  This household is UTC+8.
    tz_offset_hours: float = 8.0
    episode_trigger: int = 6  # this many episodes also triggers a review
    urgent_repeat_s: float = 300.0  # min gap between cap-triggered reviews
    docked_full_pct: int = 98  # docked at/above this = bout is over
    docked_idle_close_s: float = 1800.0  # docked this long = bout over
    # Rolling-gauge windows (user 2026-11-11: judge on LIVE consumption,
    # not only whole-bout averages, so a bout heading for an all-day
    # finish is caught while there is still time to act).
    rolling_window_s: float = 900.0  # ~15 min rolling drain/progress
    eta_reserve_pct: float = 8.0  # battery reserve kept in ETA maths
    eta_charge_to_pct: float = 90.0  # assumed target of future charges
    # Virtual water gauge (user-settled 2026-11-11): the plumbed
    # station refills + washes on every real dock stay, so a stay of at
    # least water_reset_dock_s counts as water back to 100%.  Between
    # stays the gauge drains by spray amount x newly mopped area.  The
    # rates are relative placeholders (no observable "empty" event
    # exists to calibrate against): 100% ~= 125/83/55 m^2 per level.
    water_reset_dock_s: float = 120.0
    water_rate_pct_per_m2: dict = field(
        default_factory=lambda: {1: 0.8, 2: 1.2, 3: 1.8, 4: 2.4})
    water_default_amount: int = 2
    # This household's own profile (set in the local config, kept out
    # of the generic code): what areas a session covers and how long a
    # NORMAL session takes here.  When set it outranks the generic
    # duration yardsticks in the supervisor prompt.
    home_note: str = ""
    normal_session_max_s: float = 0.0
    # Home-cache consistency (user 2026-11-11): the cached home area
    # is the denominator for progress / remaining / ETA.  If a bout's
    # area significantly mismatches the cache (robot taken to another
    # home, new map, different rooms), those cached figures must not
    # be reported as fact.
    home_suspect_novel_share: float = 0.25  # >= this share of NEW cells
    home_min_cache_cells: int = 150  # cache must be mature to be judged
    home_min_bout_cells: int = 40  # bout must be substantial to judge
    # Short-cycle charge thrashing (user 2026-11-11): charged, resumed,
    # flat again in no time, repeatedly = the resume level / drain
    # handling is broken.  A segment counts toward thrashing when it
    # started after a charge stop, ENDED at low battery (a mop-wash
    # return ends at whatever charge - it must not count), and bought
    # little cleaning: short on time OR small on area.
    short_segment_max_min: float = 20.0
    short_segment_max_m2: float = 8.0
    thrash_end_batt_max: int = 30  # segment must end at/below this
    thrash_segments: int = 2  # this many short ones in a row = thrash


def home_consistency(bout_cells: set, cached_cells: set,
                     cfg: SupervisorConfig) -> dict:
    """Compare a bout's footprint with the cached home baseline."""
    novel = len(bout_cells - cached_cells)
    share = novel / len(bout_cells) if bout_cells else 0.0
    suspect = (
        len(cached_cells) >= cfg.home_min_cache_cells
        and len(bout_cells) >= cfg.home_min_bout_cells
        and share >= cfg.home_suspect_novel_share
    )
    return {"novel_area_share": round(share, 2), "suspect": suspect}


def candidate_decision(bout_cells: set, cached_cells: set,
                       candidate_cells: set | None,
                       cfg: SupervisorConfig) -> str:
    """Bout-end cache maintenance decision.

    'merge'  - bout matches the cache: fold it in, drop any candidate.
    'stash'  - bout mismatches the cache: remember it as a candidate
               new baseline, but do NOT adopt a one-off (the robot may
               just be visiting another home).
    'adopt'  - bout mismatches the cache but matches the stashed
               candidate: the new layout repeated, adopt bout+candidate
               as the new baseline.
    'none'   - too small to judge; leave everything as is.
    """
    if len(bout_cells) < cfg.home_min_bout_cells:
        return "none"
    if not home_consistency(bout_cells, cached_cells, cfg)["suspect"]:
        return "merge"
    if candidate_cells and not home_consistency(
            bout_cells, candidate_cells, cfg)["suspect"]:
        return "adopt"
    return "stash"


@dataclass
class Verdict:
    action: str = "continue"  # continue|notify|abandon_current|stop_session|ask_user
    message: str = ""
    reason: str = ""
    resume_charge_pct: int | None = None
    # Optional charge-model calibration FROM the raw history in the
    # digest (user 2026-11-11: the AI corrects the parameters; code
    # only validates the shape and applies them).
    charge_model_update: dict | None = None


class SessionWatch:
    """Accumulates whole-bout aggregates from the sample stream.

    A bout opens at the first CLEANING sample and stays open across
    mid-bout dockings (charge stops); it closes when the robot sits
    docked at full battery or stays docked past the idle limit.
    """

    def __init__(self, cfg: SupervisorConfig):
        self.cfg = cfg
        self._last: Sample | None = None
        self._last_phase = "monitor"
        self._last_cell: tuple[int, int] | None = None
        self.bout_open = False
        self.cleaning_now = False
        self.bout_start_wall = 0.0
        self.bout_cleaning_s = 0.0
        self.distance_m = 0.0
        self.cells: set[tuple[int, int]] = set()
        self.cell_entries = 0
        self.episodes_total = 0
        self.episodes_since_review = 0
        self.dock_returns = 0
        self.errors: Counter = Counter()
        self.cleaning_since_review_s = 0.0
        self.battery_pct: int | None = None
        self.bout_start_battery: int | None = None
        self._docked_since_t: float | None = None
        self._charge_review_pending = False
        self._was_returning = False
        # Rolling history: (sample.t, battery_pct, unique cells so far)
        self._history: deque = deque()
        self._pace_hist: deque = deque()  # (t, distinct cell count)
        # Raw charge history: recent dock stays with their SOC
        # trajectory marks.  Recorded faithfully and handed to the
        # supervisor LLM, which does the CALIBRATING - code invents
        # no correction scheme of its own (user 2026-11-11).
        self._charge_history: deque = deque(maxlen=12)
        self._dock_stay: dict | None = None
        # Virtual water gauge state.
        self.water_amount: int | None = None
        self._water_cells: set[tuple[int, int]] = set()
        self._water_reset_t: float | None = None  # sample.t of last reset
        self._water_reset_this_stay = False
        # Per-segment ledger: one entry per continuous cleaning run
        # between dockings (duration / area / start+end battery /
        # whether it began right after a charge stop).
        self.segments: list[dict] = []
        self._seg_open = False
        self._seg_clean_s = 0.0
        self._seg_cells0 = 0
        self._seg_batt0: int | None = None
        self._seg_after_charge = False
        self._had_dock_this_bout = False

    # ------------------------------------------------------------ feeding
    def feed(self, sample: Sample, phase: str, wall: float,
             battery: int | None = None,
             water_amount: int | None = None) -> None:
        if battery is not None:
            self.battery_pct = battery
        if water_amount is not None:
            self.water_amount = water_amount
        last = self._last
        dt = 0.0 if last is None else min(sample.t - last.t, 15.0)
        state = sample.state

        if state == RobotState.CLEANING:
            if not self.bout_open:
                self.bout_open = True
                self.bout_start_wall = wall
                self.bout_cleaning_s = 0.0
                self.distance_m = 0.0
                self.cells = set()
                self.cell_entries = 0
                self.episodes_total = 0
                self.errors = Counter()
                self.bout_start_battery = self.battery_pct
                self._history.clear()
                self._pace_hist.clear()
                self.segments = []
                self._seg_open = False
                self._had_dock_this_bout = False
                # Leaving the station for a fresh bout = just washed
                # and refilled: the virtual water gauge starts full.
                self._water_cells = set()
                self._water_reset_t = sample.t
            self.cleaning_now = True
            self._docked_since_t = None
            self._water_reset_this_stay = False
            if not self._seg_open:
                self._seg_open = True
                self._seg_clean_s = 0.0
                self._seg_cells0 = len(self.cells)
                self._seg_batt0 = self.battery_pct
                # Consume the dock flag: only the segment that starts
                # right after a docking counts as "after charge" (a
                # later pause/resume must not inherit it).
                self._seg_after_charge = self._had_dock_this_bout
                self._had_dock_this_bout = False
            self._seg_clean_s += dt
            self.bout_cleaning_s += dt
            self.cleaning_since_review_s += dt
            if last is not None:
                self.distance_m += last.pos.distance_to(sample.pos)
            cell = (round(sample.pos.x / CELL_M), round(sample.pos.y / CELL_M))
            if cell != self._last_cell:
                self.cell_entries += 1
                self.cells.add(cell)
                self._water_cells.add(cell)
                self._last_cell = cell
            self._pace_hist.append((sample.t, len(self.cells)))
            while (self._pace_hist
                   and sample.t - self._pace_hist[0][0]
                   > self.cfg.rolling_window_s):
                self._pace_hist.popleft()
            if self.battery_pct is not None:
                self._history.append(
                    (sample.t, self.battery_pct, len(self.cells)))
                while (self._history
                       and sample.t - self._history[0][0]
                       > self.cfg.rolling_window_s):
                    self._history.popleft()
        elif state == RobotState.RETURNING:
            self._close_segment()
            self.cleaning_now = False
            self._was_returning = True
        elif state in (RobotState.DOCKED, RobotState.IDLE):
            self._close_segment()
            if state == RobotState.DOCKED:
                self._had_dock_this_bout = True
            if self.cleaning_now or self._was_returning:
                if state == RobotState.DOCKED:
                    self.dock_returns += 1
                    # Docked mid-bout with a not-full battery = a charge
                    # stop: that is THE moment for the charge review.
                    if (self.bout_open and self.battery_pct is not None
                            and self.battery_pct < self.cfg.docked_full_pct):
                        self._charge_review_pending = True
            self.cleaning_now = False
            self._was_returning = False
            self._last_cell = None
            if self._docked_since_t is None:
                self._docked_since_t = sample.t
            if (state == RobotState.DOCKED and self.bout_open
                    and not self._water_reset_this_stay
                    and sample.t - self._docked_since_t
                    >= self.cfg.water_reset_dock_s):
                # A real station stay, not a pass-through touch: the
                # plumbed station washes the mop and refills, so the
                # virtual water gauge resets to full (user-settled rule,
                # 2026-11-11).
                self._water_cells = set()
                self._water_reset_t = sample.t
                self._water_reset_this_stay = True
            if self.bout_open and (
                (self.battery_pct or 0) >= self.cfg.docked_full_pct
                or sample.t - self._docked_since_t >= self.cfg.docked_idle_close_s
            ):
                self.bout_open = False
                self._charge_review_pending = False
        else:  # PAUSED / ERROR / UNKNOWN: bout state unchanged
            self.cleaning_now = False

        # Charge-history recording: while docked, log the SOC
        # trajectory (a mark every +2%); on undocking, file the stay.
        # Raw material only - the supervisor LLM calibrates the
        # charge model from it (charge_model_update verdicts).
        if state == RobotState.DOCKED and self.battery_pct is not None:
            if self._dock_stay is None:
                self._dock_stay = {
                    "soc0": self.battery_pct, "t0": sample.t,
                    "marks": [[self.battery_pct, 0.0]]}
            else:
                marks = self._dock_stay["marks"]
                if self.battery_pct - marks[-1][0] >= 2:
                    marks.append([self.battery_pct, round(
                        (sample.t - self._dock_stay["t0"]) / 60, 1)])
        elif self._dock_stay is not None:
            stay, self._dock_stay = self._dock_stay, None
            mins = (sample.t - stay["t0"]) / 60
            if mins >= 1 and self.battery_pct is not None:
                stay["marks"].append(
                    [self.battery_pct, round(mins, 1)])
                self._charge_history.append({
                    "start_soc": stay["soc0"],
                    "minutes": round(mins, 1),
                    "marks": stay["marks"]})

        if sample.error_code:
            self.errors[sample.error_code] += 1
        if self._last_phase == "monitor" and phase not in ("monitor", "cooldown"):
            self.episodes_total += 1
            self.episodes_since_review += 1
        self._last_phase = phase
        self._last = sample

    def _close_segment(self) -> None:
        if not self._seg_open:
            return
        self._seg_open = False
        self.segments.append({
            "clean_min": round(self._seg_clean_s / 60, 1),
            "area_m2": round(
                (len(self.cells) - self._seg_cells0) * CELL_AREA_M2, 1),
            "start_batt": self._seg_batt0,
            "end_batt": self.battery_pct,
            "after_charge": self._seg_after_charge,
        })

    def short_cycle_thrash(self) -> bool:
        """The last `thrash_segments` charge-driven segments were all
        short: the resume level / drain handling is not working."""
        segs = [
            s for s in self.segments
            if s["after_charge"]
            and s["end_batt"] is not None
            and s["end_batt"] <= self.cfg.thrash_end_batt_max
        ]
        if len(segs) < self.cfg.thrash_segments:
            return False
        recent = segs[-self.cfg.thrash_segments:]
        return all(
            s["clean_min"] < self.cfg.short_segment_max_min
            or s["area_m2"] < self.cfg.short_segment_max_m2
            for s in recent)

    # ------------------------------------------------------------ review
    def due(self) -> bool:
        if not self.cfg.enabled or not self.bout_open:
            return False
        if self._charge_review_pending:
            return True
        if not self.cleaning_now:
            return False
        if self.cleaning_since_review_s >= self.cfg.review_interval_s:
            return True
        if (self.bout_cleaning_s >= self.cfg.session_soft_cap_s
                and self.cleaning_since_review_s >= self.cfg.urgent_repeat_s):
            return True
        return self.episodes_since_review >= self.cfg.episode_trigger

    def mark_reviewed(self) -> None:
        self.cleaning_since_review_s = 0.0
        self.episodes_since_review = 0
        self._charge_review_pending = False

    @property
    def charge_review_pending(self) -> bool:
        return self._charge_review_pending

    def covered_m2(self) -> float:
        return len(self.cells) * CELL_AREA_M2

    def drain_pct_per_m2(self) -> float | None:
        """Learned battery drain this bout, if the data supports it."""
        if (self.bout_start_battery is None or self.battery_pct is None):
            return None
        used = self.bout_start_battery - self.battery_pct
        cov = self.covered_m2()
        if used >= 5 and cov >= 5.0:
            return round(used / cov, 3)
        return None

    # ------------------------------------------------------------- gauges
    def _rolling_span(self) -> tuple[float, float, float] | None:
        """(battery used %, new area m^2, span min) over the window."""
        if len(self._history) < 2:
            return None
        t0, b0, c0 = self._history[0]
        t1, b1, c1 = self._history[-1]
        span_min = (t1 - t0) / 60.0
        if span_min < 5.0:
            return None
        return (b0 - b1, (c1 - c0) * CELL_AREA_M2, span_min)

    def rolling_drain_pct_per_m2(self) -> float | None:
        """Battery % per NEW m^2 over the recent window (live rate)."""
        span = self._rolling_span()
        if span is None:
            return None
        used, new_m2, _ = span
        if used >= 2 and new_m2 >= 2.0:
            return round(used / new_m2, 3)
        return None

    def rolling_progress_m2_per_min(self) -> float | None:
        """New-coverage rate over the recent window."""
        span = self._rolling_span()
        if span is None:
            return None
        _, new_m2, span_min = span
        if new_m2 > 0:
            return round(new_m2 / span_min, 2)
        return None

    def rolling_new_m2(self) -> float | None:
        """New area covered inside the rolling window."""
        span = self._rolling_span()
        return round(span[1], 1) if span is not None else None

    def pace_stalled(self) -> bool:
        """Speed anomaly, deliberately cache-free (user 2026-11-11):
        the rolling window passed with (almost) no NEW coverage while
        the robot kept 'cleaning'.  Needs no home total, no remaining
        area, no ETA - it is valid in a foreign home, on a new map,
        everywhere, at all times."""
        span = self._rolling_span()
        return span is not None and span[1] < 1.0

    def _window_pace_split(self) -> tuple[float, float | None] | None:
        """Split the rolling window's cleaning time into low-yield
        time (cleaning but gaining NO new cell - probing around an
        obstacle, escaping, re-cleaning; honest progress there is
        ~zero, user 2026-11-11) and advancing time.  Returns
        (low_yield_share, advancing_pace_m2_per_min|None).  Pure
        measurement: what it MEANS is the supervisor's call."""
        h = self._pace_hist
        if len(h) < 8:
            return None
        clean_t = low_t = 0.0
        prev = None
        for t1, c1 in h:
            if prev is not None:
                t0, c0 = prev
                dt = min(t1 - t0, 15.0)
                if dt > 0:
                    clean_t += dt
                    if c1 <= c0:
                        low_t += dt
            prev = (t1, c1)
        if clean_t < 120:
            return None
        new_m2 = (h[-1][1] - h[0][1]) * CELL_AREA_M2
        advancing = None
        eff_t = clean_t - low_t
        if eff_t >= 60 and new_m2 > 0:
            advancing = round(new_m2 / (eff_t / 60.0), 2)
        return round(low_t / clean_t, 2), advancing

    def charge_history(self) -> list[dict]:
        """Recent dock stays (copies): start_soc, minutes, SOC marks."""
        return [dict(s) for s in self._charge_history]

    def restore_charge_history(self, stays: list[dict]) -> None:
        self._charge_history.clear()
        for s in stays[-12:]:
            self._charge_history.append(dict(s))

    def water_pct_est(self) -> float:
        """Virtual water gauge: 100 after a station wash/refill, then
        drains by spray amount x area mopped since (user-settled model;
        the firmware exposes no real level)."""
        amount = self.water_amount or self.cfg.water_default_amount
        rate = self.cfg.water_rate_pct_per_m2.get(
            amount, self.cfg.water_rate_pct_per_m2[2])
        used = len(self._water_cells) * CELL_AREA_M2 * rate
        return round(max(0.0, 100.0 - used), 1)

    def _best_drain(self, model: ChargeModel | None) -> float:
        """Drain estimate by evidence strength: rolling > bout > model."""
        return (self.rolling_drain_pct_per_m2()
                or self.drain_pct_per_m2()
                or (model.drain_pct_per_m2 if model is not None
                    else ChargeModel().drain_pct_per_m2))

    def _hm(self, wall_ts: float) -> str:
        """HH:MM in the home timezone (cfg.tz_offset_hours).

        Deliberately NOT time.localtime: the process clock's zone is
        an accident of the deployment environment, while the judge's
        sense of 'what time it is' must always be the household's."""
        tz = timezone(timedelta(hours=self.cfg.tz_offset_hours))
        return datetime.fromtimestamp(wall_ts, tz).strftime("%H:%M")

    def soc_to_guarantee_finish(self, remaining_m2: float,
                                model: ChargeModel | None = None) -> int:
        """SOC to LEAVE the dock with so the remaining area finishes in
        one outing: reserve + remaining x drain x safety margin.  This
        is an honest number for the judge, not a hard rule - the LLM
        weighs it against the clock, the hour and what is left."""
        need = (self.cfg.eta_reserve_pct
                + remaining_m2 * self._best_drain(model) * MARGIN)
        return int(math.ceil(min(100.0, need)))

    def charge_options(self, wall: float, remaining_m2: float,
                       model: ChargeModel | None = None) -> list[dict]:
        """Candidate resume levels priced in TIME: from the current
        battery, how long each level takes to reach (and the clock
        time when it is ready) and how much area it buys.  A level
        that takes hours to reach can be the wrong choice even when
        it would 'guarantee' finishing - the judge needs the price
        tag, not just the level (user 2026-11-11)."""
        if self.battery_pct is None:
            return []
        model = model or ChargeModel()
        drain = self._best_drain(model)
        levels = {50, 65, 80, 100,
                  self.soc_to_guarantee_finish(remaining_m2, model)}
        out = []
        for level in sorted(levels):
            if level <= self.battery_pct or level < 30:
                continue
            mins = model.charge_minutes(self.battery_pct, level)
            out.append({
                "resume_soc": level,
                "charge_min_from_now": round(mins, 1),
                "ready_at": self._hm(wall + mins * 60),
                "buys_m2": round(max(
                    0.0, (level - self.cfg.eta_reserve_pct) / drain), 1),
            })
        return out

    def estimate_eta(self, wall: float, home_cells: int,
                     model: ChargeModel | None = None) -> dict | None:
        """Projected finish: remaining area at the live progress rate,
        plus the charge cycles the remaining energy shortfall implies
        (each charged to eta_charge_to_pct at the model's band rates,
        plus per-cycle overhead).  None when no honest projection is
        possible (no progress signal yet)."""
        remaining = max(0, home_cells - len(self.cells)) * CELL_AREA_M2
        if remaining <= 0:
            return {"eta_minutes_left": 0.0,
                    "eta_finish": self._hm(wall),
                    "eta_charges_left": 0}
        progress = self.rolling_progress_m2_per_min()
        if progress is None and self.bout_cleaning_s >= 300:
            avg = self.covered_m2() / (self.bout_cleaning_s / 60.0)
            progress = round(avg, 2) if avg > 0 else None
        if not progress:
            return None
        model = model or ChargeModel()
        drain = self._best_drain(model)
        clean_min = remaining / progress
        charges = 0
        charge_min = 0.0
        if self.battery_pct is not None:
            need = remaining * drain
            usable = max(0.0, self.battery_pct - self.cfg.eta_reserve_pct)
            if need > usable:
                per_cycle = self.cfg.eta_charge_to_pct - self.cfg.eta_reserve_pct
                charges = math.ceil((need - usable) / per_cycle)
                charge_min = charges * (
                    model.charge_minutes(self.cfg.eta_reserve_pct,
                                         self.cfg.eta_charge_to_pct)
                    + model.overhead_s / 60.0)
        total = clean_min + charge_min
        return {
            "eta_minutes_left": round(total, 1),
            # The whole bout's span if this projection holds: elapsed
            # so far + waiting/charging + remaining cleaning.  Charging
            # must not stretch THIS number past acceptability.
            "projected_total_span_min": round(
                (wall - self.bout_start_wall) / 60 + total, 1),
            "eta_finish": self._hm(wall + total * 60),
            "eta_charges_left": charges,
            "eta_assumes_progress_m2_per_min": progress,
            "eta_assumes_drain_pct_per_m2": drain,
        }

    def digest(self, wall: float, phase: str, *,
               home_cells: int | None = None,
               robot_state: str = "",
               charge_plan: dict | None = None,
               charge_model: ChargeModel | None = None,
               home_set: set | None = None,
               blocked_m2: float | None = None,
               charge_history: list | None = None) -> dict:
        revisit = 0.0
        if self.cell_entries:
            revisit = round(1.0 - len(self.cells) / self.cell_entries, 2)
        d = {
            "now": self._hm(wall),
            "bout_started": self._hm(self.bout_start_wall),
            "bout_elapsed_min": round((wall - self.bout_start_wall) / 60, 1),
            "actual_cleaning_min": round(self.bout_cleaning_s / 60, 1),
            "distance_m": round(self.distance_m, 1),
            "covered_m2_this_bout": round(self.covered_m2(), 1),
            "cell_revisit_ratio": revisit,
            "episodes_this_bout": self.episodes_total,
            "dock_returns": self.dock_returns,
            "errors": dict(self.errors),
            "current_phase": phase,
            "robot_state": robot_state,
            "battery_pct": self.battery_pct,
            "learned_drain_pct_per_m2": self.drain_pct_per_m2(),
            "rolling_drain_pct_per_m2": self.rolling_drain_pct_per_m2(),
            "rolling_progress_m2_per_min":
                self.rolling_progress_m2_per_min(),
            "rolling_new_m2": self.rolling_new_m2(),
            "pace_stalled": self.pace_stalled(),
            "charge_segments": self.segments[-4:],
            "short_cycle_thrashing": self.short_cycle_thrash(),
            "water_pct_est": self.water_pct_est(),
            "water_amount": self.water_amount,
            "water_note": "virtual gauge: 100% after each station wash/"
                          "refill, drains by spray amount x area since",
        }
        split = self._window_pace_split()
        if split is not None:
            d["low_yield_share_recent"] = split[0]
            if split[1] is not None:
                d["advancing_pace_m2_per_min"] = split[1]
        if self._water_reset_t is not None and self._last is not None:
            d["minutes_since_water_reset"] = round(
                (self._last.t - self._water_reset_t) / 60, 1)
        if self.cfg.home_note:
            d["home_note"] = self.cfg.home_note
        if self.cfg.normal_session_max_s:
            d["normal_session_max_min"] = round(
                self.cfg.normal_session_max_s / 60)
        consistency = (home_consistency(self.cells, home_set, self.cfg)
                       if home_set is not None else None)
        if consistency is not None and consistency["suspect"]:
            # The cached denominator does not describe THIS bout (new
            # map / other home / different rooms): report only what
            # this bout itself proves, and say why the rest is withheld
            # instead of feeding the judge stale-cache numbers.
            d["home_cache"] = consistency
            d["home_based_figures_withheld"] = (
                "cached home area mismatches this bout; remaining / "
                "coverage / ETA withheld until the cache re-baselines "
                "(new layout adopted after it repeats)")
        elif home_cells:
            if consistency is not None:
                d["home_cache"] = consistency
            remaining = max(0, home_cells - len(self.cells)) * CELL_AREA_M2
            d["home_area_est_m2"] = round(home_cells * CELL_AREA_M2, 1)
            d["remaining_est_m2"] = round(remaining, 1)
            d["coverage_pct_of_home"] = round(
                100.0 * len(self.cells) / home_cells, 1)
            d["resume_soc_to_guarantee_finish"] = (
                self.soc_to_guarantee_finish(remaining, charge_model))
            options = self.charge_options(wall, remaining, charge_model)
            if options:
                d["charge_options"] = options
            planned_m2 = home_cells * CELL_AREA_M2
            # Complexity accounting (user 2026-11-11): part of the
            # nominal area is obstacle footprint the robot can never
            # clean.  Expectations and coverage are judged against the
            # CLEANABLE area, not the nominal one.
            cleanable_m2 = planned_m2
            if blocked_m2:
                blocked = min(blocked_m2, 0.4 * planned_m2)
                cleanable_m2 = planned_m2 - blocked
                d["blocked_area_est_m2"] = round(blocked, 1)
                d["cleanable_area_est_m2"] = round(cleanable_m2, 1)
                d["effective_remaining_m2"] = round(
                    max(0.0, remaining - blocked), 1)
                if cleanable_m2 > 0:
                    d["coverage_pct_of_cleanable"] = round(
                        100.0 * self.covered_m2() / cleanable_m2, 1)
            model = charge_model or ChargeModel()
            expected = expected_total_minutes(cleanable_m2, model)
            if expected > 0:
                d["expected_total_min_for_planned_area"] = expected
            # Conservative expectation: same cleanable area at the
            # pace THIS environment has actually demonstrated - the
            # rolling pace already prices in every obstacle encounter,
            # detour and retry, so it is the honest speed yardstick in
            # a cluttered home (user 2026-11-11).
            expected_cons = None
            pace = self.rolling_progress_m2_per_min()
            if pace is None and self.bout_cleaning_s >= 600:
                pace = self.covered_m2() / (self.bout_cleaning_s / 60.0)
            if pace is not None and 0.25 <= pace < 1.1:
                expected_cons = expected_total_minutes(
                    cleanable_m2, model, speed_m2_per_min=pace)
                d["expected_total_min_conservative"] = expected_cons
                d["demonstrated_pace_m2_per_min"] = round(pace, 2)
            # ETA also targets the cleanable area: projecting a finish
            # time for square metres no robot can ever reach would
            # poison every ratio built on it.
            eta_cells = (home_cells if cleanable_m2 == planned_m2
                         else int(round(cleanable_m2 / CELL_AREA_M2)))
            eta = self.estimate_eta(wall, eta_cells, charge_model)
            if eta is not None:
                d.update(eta)
                if expected > 0:
                    d["projected_vs_expected_ratio"] = round(
                        eta["projected_total_span_min"] / expected, 2)
                if expected_cons:
                    d["projected_vs_conservative_ratio"] = round(
                        eta["projected_total_span_min"] / expected_cons, 2)
        if charge_plan is not None:
            d["deterministic_charge_plan"] = charge_plan
        if charge_model is not None:
            d["charge_model_in_use"] = {
                "band_rates": [list(b) for b in charge_model.band_rates],
                "drain_pct_per_m2": charge_model.drain_pct_per_m2}
        if charge_history:
            d["charge_history_recent"] = charge_history[-6:]
        return d


SYSTEM = """你是一个扫地机器人的值班监督。下面是它本次清扫的统计摘要（含电量、覆盖/剩余面积估算，以及一个确定性充电规划器给出的建议）。总目标：让「清扫+充电+回站开销」的总耗时最短——早上 7 点的任务不许拖到晚上 8 点。

只输出一个 JSON 对象，不要其他文字：
{"action": "continue|notify|abandon_current|stop_session|ask_user", "message": "给用户看的一句话（notify/stop_session/ask_user 时必填，简体中文）", "reason": "判断依据，一句话", "resume_charge_pct": 可选整数, "charge_model_update": 可选，见充电参数矫正}

判断要点：
- continue：一切正常，或虽慢但合理（大户型、首次建图）。
- notify：有异常值得用户知道，但还不至于停（如反复清扫同一区域、电量消耗异常）。
- abandon_current：当前这一个绕障 episode 明显在空转，放弃它、其余继续。
- stop_session：整个任务已失去意义（反复打扫、长时间无有效进展、剩余无几却要再充一轮电等），直接结束。
- ask_user：继续还是收尾各有道理、需要用户拍板时用（如总跨度已不可接受但剩余又确实可惜、真因环境复杂扫不完）。系统会暂停机器人（若在桩上则保持停靠），把你的 message 发给用户并等 5 分钟：用户按机身键或在 App 里恢复清扫=继续，让它回站=结束，5 分钟无任何反馈则自动回站结束。message 里要写清现状、预计几点能完、慢在哪里。只用于真正的取舍题，能自己判断的不要甩给用户。
- 预计完成时间（摘要带 eta_finish / eta_minutes_left 时）：这是按当前滚动推进速度与滚动耗电率推算的完工时刻（含预计充电轮次）。若 eta_finish 明显越界（如下午的任务拖到深夜）、或 eta_charges_left 还很多而剩余面积不大，要认真考虑 stop_session 或至少 notify，不要等软上限到了才反应。
- 速度异常与总面积无关（任何时候都成立）：pace_stalled=true 表示最近约 15 分钟几乎没有新增覆盖、却一直在「清扫」——这本身就是异常，该 notify 或 stop 就处理，不要因为本场没有剩余面积/ETA（如地图缓存对不上、基准缺席）就跳过判断。反过来，在基准缺席的陌生大户型里，只要推进正常、重扫率低，单纯耗时长不算异常，不要把本户的时长上限硬套上去。
- 虚拟水量（water_pct_est）：无真实水位传感器，这是按「回站洗拖即补满、其后按喷水档位×拖过面积递减」推算的账面水量。低于约 20% 且短期不会回站时，拖布大概率在干拖，值得 notify 一句。
- 地图缓存对不上（摘要带 home_based_figures_withheld 时）：本场扫的区域与缓存的全屋面积显著不同——多半是换了地图、换了清扫房间，或把机器人带去了别处。此时摘要已扣下剩余面积/覆盖率/预计完成时间，不要拿它们做判断，按已扫时长、推进速度、重扫率照常判断即可；这件事本身值得 notify 用户一句。缓存基准会在新布局连续出现两场后自动重建，不用管。
- 充电裁决（机器人回站充电中、摘要带 deterministic_charge_plan 时）：这就是「要不要再放它出去续扫」的决策点，每一次都要重新判断，不默认放行。先过合理性这一关（见下面单场时长与作息），不合理就 stop_session；合理时默认认可规划器的 resume_soc，只有明确理由（如剩余面积小到不值得再充）才用 resume_charge_pct 覆盖。若剩余面积为 0 或极小，优先 stop_session 而不是再充再扫。
- 短周期反复充电（摘要 short_cycle_thrashing=true 时）：充完回来扫没多久又没电、已经反复多轮——这说明续扫电量或耗电估计根本没处理好，处置由你裁决（系统不自动改电量）。结合 charge_segments 诊断：每轮出发电量（start_batt）不高，是充得太少——可以抬高 resume_charge_pct 让它一次充够再放行；出发电量已高却很快耗尽（end_batt 很快见底），是耗电异常，值得 notify 用户；若已反复多轮、剩余又不多，直接 stop_session 收尾。
- 充电标准是个平衡题，不是「必须一次充够扫完剩余」：天平一端是「一口气扫完剩余」——摘要的 resume_soc_to_guarantee_finish 是它需要的出发电量（按当前耗电估算加安全系数），充到它意味着等得久些、但后面零回站；另一端是「分多轮」——少充快走，代价是每多一轮就多付一遍回站开销（回程+洗拖布+复位，规划器按约 5 分钟/轮计）外加多轮的充电尾段损耗。deterministic_charge_plan 就是规划器在这架天平上按总时间算出的平衡点，charge_options 把每一档的价格都摊开了。对照着判断：剩余越少，一口气的等待越短、越划算；剩余越多，死等一口气越可能把总跨度撑爆，分轮反而总时间更短。用 resume_charge_pct 覆盖规划器时，等于你在天平上另选一点，要在 reason 里说清往哪端偏、为什么。唯一的下限是别碎成短周期抖动那种碎法——每一轮出发都要能扫出像样的面积。
- 充电等待也是成本（摘要 charge_options 给每个候选续扫电量标了「还要等多久、几点能走（ready_at）、能换多少面积」）：若充到高电量（如 80%）意味着要等三四个小时，那不可接受——哪怕它名义上能保证扫完。宁可选低一档（如 50%）先扫出一个可接受的面积、把主体扫完就收尾，零头留到下次。这条与上一条平衡题一起看：保证数只是天平一端的价格标签，不是必须达到的下限；它高到要等数小时，就往分轮一端偏，降档先扫出可接受面积、主体扫完收尾、零头留到下次，不要为最后一点面积把全天耗在充电等待上。记住底线是整场总跨度（摘要 projected_total_span_min）：已花的时间、充电等待、剩余清扫是一个整体——这个总和被撑到不可接受的程度（如跨大半天、顶进深夜），就必须降档或收尾。这条优先于一切比率判断：即使 projected_vs_expected_ratio 完全正常、速度与面积相称、慢得「有道理」，只要预计总跨度本身已不可接受，照样收尾——复杂度可以解释速度慢，解释不了无限长；不管什么速度，总清扫时间不许被撑到无法接受。
- 时段锚点：上午发起的任务，拖到下午四五点还在续扫、且 projected_vs_expected_ratio 显示已明显超出与面积相称的预期（或剩余极少却还要再充一轮），是强收尾信号，stop_session；若比率正常、纯粹面积大且推进稳，不算异常。无论如何不要让它无必要地跨过晚间居家时段继续耗。
- 充电参数矫正（由你持续做）：充电不是线性的——低电量段充得快、高电量段充得慢（如 30%→50% 可能只要 10 分钟，50%→80% 却要 30 分钟），而且这台机器的真实曲线会随电池状态漂移。摘要的 charge_history_recent 是最近几次回站充电的原始记录（marks 是 [电量%, 已用分钟] 轨迹打点），charge_model_in_use 是当前在用的分段速率与单位面积耗电。对照两者：若某电量段的实测速度与在用参数明显对不上、或耗电率漂移了，就在裁决里加 "charge_model_update": {"band_rates": [[上限电量, %/分钟], ... 一直覆盖到 100], "drain_pct_per_m2": 数值}（两项可只给一项），给出你矫正后的参数，系统会采用并长期保存、后续规划都用新参数。只依据实测历史矫正，没有足够历史或对不上不明显时不要带这个字段——保持原参数好过凭空乱改。
以上是判断逻辑，不是死规则：结合摘要全貌（进度、速度、电量、时刻、已投入时长）综合裁量；拿不准时先认可规划器的平衡点，或倾向于「少折腾」。
- 单场时长与作息常识（每次裁决都要合并判断，不要孤立看任何单个数字）：
  · 合理时长不看绝对钟点、看与计划面积是否相称：摘要的 expected_total_min_for_planned_area 是按合理清扫速度扫完计划面积本应花的总时间（含必要充电），projected_vs_expected_ratio 是「已花+预计总跨度」与它的比值。比值越接近 1，效率越与面积相称；比值明显大于 1，说明效率出了问题（重扫、推进塌陷、充电策略不当），偏离越大越不合理，应考虑 stop_session、剩余留到下一次，不要为「扫完」无限续扫。但比率只判效率、不判可接受性：无论速度多合理，总跨度底线（见充电条）优先于比率。这里和下文所有数字都是测量与算术，不是阈值规则——多少算过分，由你综合判断，不要用写死的「几小时」一刀切。home_note 与 normal_session_max_min 是用户按自家情况给的经验参考：范围信息（扫哪些房间）照用，时长与面积推导冲突时以面积推导为准。
  · 环境复杂度要计价：障碍多的环境推进慢是正常代价，且名义面积里有一块是障碍占住、永远扫不到的。摘要若带 blocked_area_est_m2 / cleanable_area_est_m2，预期时长已按可扫面积算；另有保守预期（expected_total_min_conservative、projected_vs_conservative_ratio）是按本环境已证明的速度算的——先看保守比率：落在乐观与保守之间，多半是复杂环境的正常成本；连保守预期都超出一大截，才更像真问题。
  · 试探时间单独看：low_yield_share_recent 是最近一段时间「在清扫、却没扫到新面积」的时间占比（绕障试探、脱困、重扫都在内），advancing_pace_m2_per_min 是剔除这段时间后的推进速度。试探期间推进接近零是合理的，demonstrated_pace 已把试探成本算进整体——判断整体进度用含试探的速度，不要拿 advancing_pace 去要求含试探的整体，也不要因为某段时间没进展就直接判异常。
  · 真因太复杂扫很久扫不完，要出声提示：若投影显示本场将远超可接受总跨度（projected_total_span_min / eta_finish），且主因是环境太复杂（试探占比高、episodes 频繁、按保守预期也快不了），必须用 notify 把话说清：预计几点能完、慢在哪里（障碍试探占了多少）、你打算怎么处置（收尾、只扫主体、还是继续）。不许无声地无限扫下去，也不许无声地放弃——用户有权知道这次为什么扫不完。
  · 结合当前时刻与普通人作息：深夜至清晨（约 23:00–07:00）一般不应持续作业、更不应再放它出去续扫，除非任务开始时间表明这就是用户有意安排的时段；晚间居家时段长时间轰鸣也有打扰成本，已扫很久时倾向收尾。
  · 合并看三件事：已扫时长相对面积预期是否失衡（projected_vs_expected_ratio）、预计还要多久（eta_minutes_left / eta_finish）、当前时刻（now）。比率失衡 + 还要很久 + 时段不对，任意两条叠加就应认真考虑 stop_session 并在 message 里说明。
- 不必全覆盖（收尾宽容）：清扫没有「不遗漏任何区域」的义务，100% 覆盖不是目标，总耗时最短才是。覆盖率要对着可扫面积看（coverage_pct_of_cleanable）：障碍占住、进不去的区域不计入「剩下多少」；可扫部分已基本扫完，就不要为名义覆盖率不满而继续耗。已经扫了很久（如四五个小时）且计划区域的多数已经扫完（coverage_pct_of_home 约 85% 以上）时，剩下的零头不值得再花时间、更不值得再充一轮电去追——直接 stop_session 收尾就是合理选择，在 message 里说明已完成大部分、剩余留待下次即可。一般地，每次都要算剩余的「性价比」：剩余面积 vs 预计耗时（eta）vs 已投入时长；已投入越久，为零头继续的门槛应越高，已花的时长是沉没成本，不构成继续扫的理由。
判断要克制：数据不足以证明异常时选 continue，不要因为单纯耗时长就停大户型的正常清扫。"""


def build_messages(digest: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "本次清扫摘要（JSON）：\n"
         + json.dumps(digest, ensure_ascii=False, indent=1)},
    ]


_ACTIONS = {"continue", "notify", "abandon_current", "stop_session",
            "ask_user"}


def parse_verdict(text: str) -> Verdict:
    """Anything unparsable degrades to 'continue' - the safe default."""
    if not text:
        return Verdict()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return Verdict()
    try:
        raw = json.loads(m.group(0))
    except (ValueError, TypeError):
        return Verdict()
    action = raw.get("action")
    if action not in _ACTIONS:
        return Verdict()
    pct = raw.get("resume_charge_pct")
    if not isinstance(pct, int) or not (10 <= pct <= 100):
        pct = None
    return Verdict(
        action=action,
        message=str(raw.get("message", ""))[:200],
        reason=str(raw.get("reason", ""))[:300],
        resume_charge_pct=pct,
        charge_model_update=_valid_model_update(
            raw.get("charge_model_update")),
    )


def _valid_model_update(raw) -> dict | None:
    """Shape-check an AI-proposed charge-model calibration.

    Validation only - monotonic band tops covering 100, rates and
    drain inside physical sanity bounds.  The VALUES are the AI's,
    derived from the raw charge history; code has no scheme of its
    own for what they should be."""
    if not isinstance(raw, dict):
        return None
    out: dict = {}
    bands = raw.get("band_rates")
    if isinstance(bands, list) and bands:
        parsed: list[tuple[int, float]] = []
        last_top, ok = 0, True
        for pair in bands:
            try:
                top, rate = int(pair[0]), float(pair[1])
            except (TypeError, ValueError, IndexError):
                ok = False
                break
            if not (last_top < top <= 100 and 0 < rate <= 5):
                ok = False
                break
            parsed.append((top, rate))
            last_top = top
        if ok and last_top == 100:
            out["band_rates"] = parsed
    drain = raw.get("drain_pct_per_m2")
    if isinstance(drain, (int, float)) and 0.05 <= float(drain) <= 3:
        out["drain_pct_per_m2"] = float(drain)
    return out or None
