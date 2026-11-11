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
from dataclasses import dataclass, field

from .chargeplan import ChargeModel
from .model import RobotState, Sample

CELL_M = 0.5
CELL_AREA_M2 = CELL_M * CELL_M


@dataclass
class SupervisorConfig:
    enabled: bool = False
    review_interval_s: float = 1200.0  # ask every ~20 min of cleaning
    session_soft_cap_s: float = 3 * 3600.0  # beyond this, ask sooner
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
    thrash_resume_floor_pct: int = 95  # forced resume floor on thrash


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
    action: str = "continue"  # continue|notify|abandon_current|stop_session
    message: str = ""
    reason: str = ""
    resume_charge_pct: int | None = None


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

    def water_pct_est(self) -> float:
        """Virtual water gauge: 100 after a station wash/refill, then
        drains by spray amount x area mopped since (user-settled model;
        the firmware exposes no real level)."""
        amount = self.water_amount or self.cfg.water_default_amount
        rate = self.cfg.water_rate_pct_per_m2.get(
            amount, self.cfg.water_rate_pct_per_m2[2])
        used = len(self._water_cells) * CELL_AREA_M2 * rate
        return round(max(0.0, 100.0 - used), 1)

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
                    "eta_finish": time.strftime(
                        "%H:%M", time.localtime(wall)),
                    "eta_charges_left": 0}
        progress = self.rolling_progress_m2_per_min()
        if progress is None and self.bout_cleaning_s >= 300:
            avg = self.covered_m2() / (self.bout_cleaning_s / 60.0)
            progress = round(avg, 2) if avg > 0 else None
        if not progress:
            return None
        model = model or ChargeModel()
        drain = (self.rolling_drain_pct_per_m2()
                 or self.drain_pct_per_m2()
                 or model.drain_pct_per_m2)
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
            "eta_finish": time.strftime(
                "%H:%M", time.localtime(wall + total * 60)),
            "eta_charges_left": charges,
            "eta_assumes_progress_m2_per_min": progress,
            "eta_assumes_drain_pct_per_m2": drain,
        }

    def digest(self, wall: float, phase: str, *,
               home_cells: int | None = None,
               robot_state: str = "",
               charge_plan: dict | None = None,
               charge_model: ChargeModel | None = None,
               home_set: set | None = None) -> dict:
        revisit = 0.0
        if self.cell_entries:
            revisit = round(1.0 - len(self.cells) / self.cell_entries, 2)
        d = {
            "now": time.strftime("%H:%M", time.localtime(wall)),
            "bout_started": time.strftime(
                "%H:%M", time.localtime(self.bout_start_wall)),
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
            eta = self.estimate_eta(wall, home_cells, charge_model)
            if eta is not None:
                d.update(eta)
        if charge_plan is not None:
            d["deterministic_charge_plan"] = charge_plan
        return d


SYSTEM = """你是一个扫地机器人的值班监督。下面是它本次清扫的统计摘要（含电量、覆盖/剩余面积估算，以及一个确定性充电规划器给出的建议）。总目标：让「清扫+充电+回站开销」的总耗时最短——早上 7 点的任务不许拖到晚上 8 点。

只输出一个 JSON 对象，不要其他文字：
{"action": "continue|notify|abandon_current|stop_session", "message": "给用户看的一句话（notify/stop_session 时必填，简体中文）", "reason": "判断依据，一句话", "resume_charge_pct": 可选整数}

判断要点：
- continue：一切正常，或虽慢但合理（大户型、首次建图）。
- notify：有异常值得用户知道，但还不至于停（如反复清扫同一区域、电量消耗异常）。
- abandon_current：当前这一个绕障 episode 明显在空转，放弃它、其余继续。
- stop_session：整个任务已失去意义（反复打扫、长时间无有效进展、剩余无几却要再充一轮电等），直接结束。
- 预计完成时间（摘要带 eta_finish / eta_minutes_left 时）：这是按当前滚动推进速度与滚动耗电率推算的完工时刻（含预计充电轮次）。若 eta_finish 明显越界（如下午的任务拖到深夜）、或 eta_charges_left 还很多而剩余面积不大，要认真考虑 stop_session 或至少 notify，不要等软上限到了才反应。
- 速度异常与总面积无关（任何时候都成立）：pace_stalled=true 表示最近约 15 分钟几乎没有新增覆盖、却一直在「清扫」——这本身就是异常，该 notify 或 stop 就处理，不要因为本场没有剩余面积/ETA（如地图缓存对不上、基准缺席）就跳过判断。反过来，在基准缺席的陌生大户型里，只要推进正常、重扫率低，单纯耗时长不算异常，不要把本户的时长上限硬套上去。
- 虚拟水量（water_pct_est）：无真实水位传感器，这是按「回站洗拖即补满、其后按喷水档位×拖过面积递减」推算的账面水量。低于约 20% 且短期不会回站时，拖布大概率在干拖，值得 notify 一句。
- 地图缓存对不上（摘要带 home_based_figures_withheld 时）：本场扫的区域与缓存的全屋面积显著不同——多半是换了地图、换了清扫房间，或把机器人带去了别处。此时摘要已扣下剩余面积/覆盖率/预计完成时间，不要拿它们做判断，按已扫时长、推进速度、重扫率照常判断即可；这件事本身值得 notify 用户一句。缓存基准会在新布局连续出现两场后自动重建，不用管。
- 充电裁决（机器人回站充电中、摘要带 deterministic_charge_plan 时）：这就是「要不要再放它出去续扫」的决策点，每一次都要重新判断，不默认放行。先过合理性这一关（见下面单场时长与作息），不合理就 stop_session；合理时默认认可规划器的 resume_soc，只有明确理由（如剩余面积小到不值得再充）才用 resume_charge_pct 覆盖。若剩余面积为 0 或极小，优先 stop_session 而不是再充再扫。
- 短周期反复充电（摘要 short_cycle_thrashing=true 时）：充完回来扫没多久又没电、已经反复多轮——这说明续扫电量或耗电估计根本没处理好。系统已自动把本轮续扫电量下限抬到 95%，不要再按原规划放行。结合 charge_segments 诊断：每轮出发电量（start_batt）不高，是充得太少；出发电量已高却很快耗尽（end_batt 很快见底），是耗电异常，值得 notify 用户。若已反复多轮、剩余又不多，直接 stop_session 收尾。
- 单场时长与作息常识（每次裁决都要合并判断，不要孤立看任何单个数字）：
  · 一次清扫任务的合理跨度：普通户型 1–2 小时，大户型 3–4 小时已是上限；拖到半天（约 6 小时以上）本身即不合理——除非摘要能证明面积确实巨大且推进速度正常，否则 stop_session，剩余留到下一次任务，不要为「扫完」无限续扫。一整天更不用说。摘要若带 home_note 与 normal_session_max_min，那是本户的实际清扫范围与正常时长，以它为准、通用标尺让位：本户正常上限已到、而覆盖又已完成大半，就按异常收尾处理。
  · 结合当前时刻与普通人作息：深夜至清晨（约 23:00–07:00）一般不应持续作业、更不应再放它出去续扫，除非任务开始时间表明这就是用户有意安排的时段；晚间居家时段长时间轰鸣也有打扰成本，已扫很久时倾向收尾。
  · 合并看三件事：已扫总时长（bout_elapsed_min）、预计还要多久（eta_minutes_left / eta_finish）、当前时刻（now）。已扫很久 + 还要很久 + 时段不对，任意两条叠加就应认真考虑 stop_session 并在 message 里说明。
- 不必全覆盖（收尾宽容）：清扫没有「不遗漏任何区域」的义务，100% 覆盖不是目标，总耗时最短才是。已经扫了很久（如四五个小时）且计划区域的多数已经扫完（coverage_pct_of_home 约 85% 以上）时，剩下的零头不值得再花时间、更不值得再充一轮电去追——直接 stop_session 收尾就是合理选择，在 message 里说明已完成大部分、剩余留待下次即可。一般地，每次都要算剩余的「性价比」：剩余面积 vs 预计耗时（eta）vs 已投入时长；已投入越久，为零头继续的门槛应越高，已花的时长是沉没成本，不构成继续扫的理由。
判断要克制：数据不足以证明异常时选 continue，不要因为单纯耗时长就停大户型的正常清扫。"""


def build_messages(digest: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "本次清扫摘要（JSON）：\n"
         + json.dumps(digest, ensure_ascii=False, indent=1)},
    ]


_ACTIONS = {"continue", "notify", "abandon_current", "stop_session"}


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
    )
