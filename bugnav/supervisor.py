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
import re
import time
from collections import Counter
from dataclasses import dataclass

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

    # ------------------------------------------------------------ feeding
    def feed(self, sample: Sample, phase: str, wall: float,
             battery: int | None = None) -> None:
        if battery is not None:
            self.battery_pct = battery
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
            self.cleaning_now = True
            self._docked_since_t = None
            self.bout_cleaning_s += dt
            self.cleaning_since_review_s += dt
            if last is not None:
                self.distance_m += last.pos.distance_to(sample.pos)
            cell = (round(sample.pos.x / CELL_M), round(sample.pos.y / CELL_M))
            if cell != self._last_cell:
                self.cell_entries += 1
                self.cells.add(cell)
                self._last_cell = cell
        elif state == RobotState.RETURNING:
            self.cleaning_now = False
            self._was_returning = True
        elif state in (RobotState.DOCKED, RobotState.IDLE):
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

    def digest(self, wall: float, phase: str, *,
               home_cells: int | None = None,
               robot_state: str = "",
               charge_plan: dict | None = None) -> dict:
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
        }
        if home_cells:
            remaining = max(0, home_cells - len(self.cells)) * CELL_AREA_M2
            d["home_area_est_m2"] = round(home_cells * CELL_AREA_M2, 1)
            d["remaining_est_m2"] = round(remaining, 1)
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
- 充电裁决（机器人回站充电中、摘要带 deterministic_charge_plan 时）：默认认可规划器的 resume_soc；只有你有明确理由（如剩余面积小到不值得再充、时段不合适）才用 resume_charge_pct 覆盖，或用 stop_session 结束。若剩余面积为 0 或极小，优先 stop_session 而不是再充再扫。
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
